import os
import sys
sys.path.append(os.path.abspath('..'))
import collections,torch
from collections import OrderedDict
from.TorchVector_new import TorchVect
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math,time, re
from typing import Sequence, Callable, Optional
from nonsmooth.RWrap import RWrap
from nonsmooth.trustregion import trustregion
from nonsmooth.setDefaultParameters_new import set_default_parameters
from non_smooth.Problem import Problem
from non_smooth.L1norm_new import L1TorchNorm
from non_smooth.L2Vectors_new import L2TVPrimal, L2TVDual
from non_smooth.step_new import Reye

class FullyConnectedNN(nn.Module):
    def __init__(self, sizes:Sequence[int],activation:Optional[nn.Module]=None,last_bias: bool=False,):
        super().__init__()
        assert len(sizes) >= 2, "sizes must have at least [in, out]"
        assert sizes[0] == 2, "First layer input dim must be 2 for 2D inputs"
        act = activation if activation is not None else nn.Tanh()

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=True))
            # use a fresh activation module each time
            layers.append(type(act)())

        # final linear layer (optionally without bias)
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=last_bias))
        self.net = nn.Sequential(*layers)

        # good default init for PINNs
        self._init_weights()
        # in FullyConnectedNN.__init__
        self.cB = torch.nn.Parameter(torch.tensor([2.7726], dtype=torch.float64))  # shape (1,)
  

        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot is a solid default (tanh-friendly)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
        

    def forward(self, x:torch.Tensor):
        #print("inputs_shape:",x.shape)
        
        return self.net(x)
    
    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
        
    def gradient(self,x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        u = self.forward(x)  # (N,1)
        grad = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,  # keep graph for Laplacian
            retain_graph=True
        )[0]  # (N,2)
        
        return grad
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        g = self.gradient(x)  # (N,2), graph kept
        lap = 0.0
        for d in range(g.shape[1]):  # 2D
            g_d = g[:, d:d+1]  # (N,1)
            lap_d = torch.autograd.grad(
                g_d, x,
                grad_outputs=torch.ones_like(g_d),
                create_graph=True,
                retain_graph=True
            )[0][:, d:d+1]  # (N,1) pick same component
            lap = lap + lap_d
            
        return lap  # (N,1)
        
def _stateless_u(var, params, x_xy):
    xg = x_xy.detach().clone().requires_grad_(True)
    u  = torch.func.functional_call(var['NN'], params, (xg,))  # (N,1)
    return u, xg
    
from collections import OrderedDict

def _u_with_hard_bc(var, params, x_xy):
    """
    Stateless forward with hard Dirichlet BC:
      u(x,y) = B(x,y) * N_theta(x,y),   B = x(1-x)*y(1-y).
    IMPORTANT: do NOT detach params, or you break backprop.
    """
    # Accept TorchVect or OrderedDict
    if hasattr(params, "td"):
        params = params.td

    # Put tensors on the model's device/dtype WITHOUT detaching
    dev   = next(var['NN'].parameters()).device
    dtype = next(var['NN'].parameters()).dtype
    params = OrderedDict((k, v.to(device=dev, dtype=dtype)) for k, v in params.items())

    # Inputs with grad (for PDE terms)
    xg = x_xy.detach().clone().requires_grad_(True)

    # Stateless forward that keeps grads to params
    n_raw = torch.func.functional_call(var['NN'], params, (xg,))

    # Hard BC mask
    B = (xg[:, :1] * (1 - xg[:, :1])) * (xg[:, 1:2] * (1 - xg[:, 1:2]))
    #u = B * n_raw
    if 'cB' in params:
        scale = torch.exp(params['cB'])      # works for scalar () or shape (1,)
    else:
        scale = 1.0

    u = scale * B * n_raw
    #u = torch.exp(var['NN'].cB) * B * n_raw

    return u, xg
def hard_bc_forward_and_derivs(var, params, x_xy):
    # Stateless net output N
    xg = x_xy.detach().clone().requires_grad_(True)
    N  = torch.func.functional_call(var['NN'], params, (xg,))  # (N,1)

    x = xg[:, :1]; y = xg[:, 1:2]
    B = x*(1-x)*y*(1-y)
    dBx = (1 - 2*x) * y*(1-y)
    dBy = (1 - 2*y) * x*(1-x)
    lapB = -2*y*(1-y) - 2*x*(1-x)

    # ∇N and ΔN by autodiff
    gN = torch.autograd.grad(N, xg, grad_outputs=torch.ones_like(N), create_graph=True)[0]  # (N,2)
    lapN = 0.0
    for d in range(2):
        gNd = gN[:, d:d+1]
        lapNd = torch.autograd.grad(gNd, xg, grad_outputs=torch.ones_like(gNd), create_graph=True)[0][:, d:d+1]
        lapN = lapN + lapNd
    if 'cB' in params:
        scale = torch.exp(params['cB'])
    elif hasattr(var['NN'],'cB'):
        scale = torch.exp(var['NN'].cB)
    else:
        scale = 1.0

    # u, ∇u, Δu via product rule
    u = scale * B * N
    gu_x = scale * (dBx * N + B * gN[:, :1])
    gu_y = scale * (dBy * N + B * gN[:, 1:2])
    gu = torch.cat([gu_x, gu_y], dim=1)
    lapu = scale * (lapB * N + 2*(dBx * gN[:, :1] + dBy * gN[:, 1:2]) + B * lapN)
    return u, gu, lapu, xg

def apply_best_scale_live(nnset):
    diag   = nnset.diagnose_scale()          # prints s_opt, etc.
    s_opt  = diag["s_opt"]

    with torch.no_grad():
        # 1) If negative, flip last layer to absorb the sign
        if s_opt < 0:
            last = nnset.NN.net[-1]          # final nn.Linear in your Sequential
            last.weight.mul_(-1)
            if last.bias is not None:
                last.bias.mul_(-1)
            s_opt = -s_opt                   # now positive

        # 2) Set cB = log(s_opt)
        if hasattr(nnset.NN, "cB"):
            nnset.NN.cB.copy_(torch.tensor(math.log(max(s_opt, 1e-16)),
                                            dtype=nnset.NN.cB.dtype,
                                            device=nnset.NN.cB.device))
    return diag     


@torch.no_grad()
def param_hash(nn):
    return float(sum(p.abs().sum() for p in nn.parameters()).cpu())  

class NNObjective:
    def __init__(self, var):
        self.var = var
        self.loss_history = []
        
    
    def update(self, x, type):
        return None
    
    
        
    def check_parameters_require_grad(self, params, context=""):
        """Check if all parameters require gradients"""
        if isinstance(params, TorchVect):
            params_dict = params.td
        else:
            params_dict = params
            
        issues = []
        for name, param in params_dict.items():
            if not param.requires_grad:
                issues.append(name)
                
        if issues:
            print(f"GRADIENT ISSUE {context}: Parameters without grad: {issues}")
        return len(issues) == 0
    
    @staticmethod
    def _trainable_params_like(model, param_map):
        """Clone to leaf tensors, correct dtype/device, requires_grad=True, preserve order."""
        if not isinstance(param_map, collections.OrderedDict):
            param_map = collections.OrderedDict(param_map)
        # align to model's state_dict ordering exactly
        sd = model.state_dict()
        assert list(param_map.keys()) == list(sd.keys()), \
            f"Param keys mismatch.\nmodel: {list(sd.keys())}\nparam_map: {list(param_map.keys())}"
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        out = collections.OrderedDict()
        for k in sd.keys():  # preserve exact order
            #t = param_map[k].detach().clone().to(dtype=dtype, device=device).requires_grad_(True)
            t = param_map[k].to(dtype=dtype, device=device)
            if not t.requires_grad:
                t.requires_grad_(True)
            out[k] = t
        return out
    def value_torch(self, params, ftol=None):
        """
        Strong-form PINN with hard Dirichlet BCs:
          u(x) = B(x) * N(x),   B(x,y)=x(1-x)y(1-y)
        PDE: -div(k grad u) = b   (k may vary; gradk provided)
        """
        # params can be TorchVect or OrderedDict
        if isinstance(params, TorchVect):
            params = params.td

        # interior inputs (dof coordinates)
        #x_in = self.var['inputs_xy']  # (N,2) torch
        I = self.var.get('interior_idx',None)
        x_all = self.var['inputs_xy']
        if I is not None:
            x_in = x_all[I]
        else:
            x_in = x_all
        u, gu, lap, xg = hard_bc_forward_and_derivs(self.var, params, x_in)


        # coefficients/RHS aligned to dof coords
        k_all     = self.var['k']      # (N,1)
        gradk_all = self.var['gradk']  # (N,2)
        b_all     = self.var['b']      # (N,1)
        if I is not None:
            k = k_all[I]
            gradk = gradk_all[I]
            b = b_all[I]
        else:
            k,gradk,b = k_all,gradk_all,b_all

        # residual:  - (∇k · ∇u) - k Δu - b
        pde_residual = -(gradk * gu).sum(dim=1, keepdim=True) - k * lap - b
        pde_loss = 0.5 * self.var.get('alpha', 1.0) * (pde_residual.pow(2)).mean()
        
        lam = float(self.var.get('lambda_reg', 0.0))
        reg = torch.tensor(0.0, dtype=pde_loss.dtype, device=pde_loss.device)
        if lam > 0.0:
            excl = tuple(self.var.get('l2_exclude', ()))  # patterns to skip
            reg_sum = []
            for name, p in params.items():                # IMPORTANT: use the passed-in params
                if any(tag in name for tag in excl):      # skip biases, cB, etc.
                    continue
                reg_sum.append((p**2).sum())
            if reg_sum:
                reg = 0.5 * lam * torch.stack(reg_sum).sum()


        # OPTIONAL: soft BC penalty (should be ~0 with hard BC; keep as guardrail)
        loss_bc = torch.tensor(0.0, dtype=pde_loss.dtype, device=pde_loss.device)
        if ('bdry_xy' in self.var and 'bdry_u' in self.var and
            self.var['bdry_xy'] is not None and self.var['bdry_u'] is not None):
            xb = self.var['bdry_xy']          # (Nb,2)
            ub = self.var['bdry_u']           # (Nb,1)
            ub_pred, _ = _u_with_hard_bc(self.var, params, xb)  # uses same hard-BC ansatz
            loss_bc = 0.5 * (ub_pred - ub).pow(2).mean()
            

        beta_bc = 0.0  # with hard BC, you can set 0; or small positive as a safety belt
        loss = pde_loss + beta_bc * loss_bc +reg
        ###consider L2 norm of p
        
        

        self.loss_history.append(loss.item())
        return loss

                
  
        
    
        
    def value_torch_soft(self, params, ftol=None):
        """
        Strong-form loss with SOFT Dirichlet BCs only.
        PDE:  - (∇k · ∇u) - k Δu = b    on Ω
        BC :  u = u_b                 on ∂Ω   (penalized)
        """
        # Accept TorchVect or OrderedDict
        if isinstance(params, TorchVect):
            params = params.td

        # ===== Interior residual =====
        x = self.var['inputs_xy']              # (Ni,2) dof coords (Ω)
        if not x.requires_grad:
            x = x.requires_grad_(True)         # needed for grads/Laplacian

        # Stateless forward (plain network, NO hard BC)
        u = torch.func.functional_call(self.var['NN'], params, (x,))
        #u, xg = _stateless_u(self.var, params, x)

        # ∇u
        gu = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        # Δu
        lap = torch.zeros_like(u)
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap_d = torch.autograd.grad(
                 g_d, x, grad_outputs=torch.ones_like(g_d),
                 create_graph=True, retain_graph=True
            )[0][:, d:d+1]
            lap += lap_d

        # coefficients / RHS (aligned to x)
        k     = self.var['k']      # (Ni,1)
        gradk = self.var['gradk']  # (Ni,2)
        b     = self.var['b']      # (Ni,1)  <-- pointwise f(x), not assembled RHS
        def _reduce_mean(x):  # helper
            return x.mean()
        

        # PDE residual and loss
        pde_residual = -(gradk * gu).sum(dim=1, keepdim=True) - k * lap - b
        #pde_loss = 0.5 * self.var.get('alpha', 1.0) * (pde_residual.pow(2).mean())
        pde_loss = 0.5*self.var['alpha']*_reduce_mean((pde_residual**2))

        # ===== Boundary penalty (soft BC) =====
        loss_bc = torch.tensor(0.0, dtype=pde_loss.dtype, device=pde_loss.device)
        if ('bdry_xy' in self.var and self.var['bdry_xy'] is not None and
            'bdry_u'  in self.var and self.var['bdry_u']  is not None):
            xb = self.var['bdry_xy']           # (Nb,2) boundary coords
            ub = self.var['bdry_u']            # (Nb,1) target boundary values (e.g., zeros)

            # For BC penalty we DON'T need grads wrt xb; just params
            #ub_pred = torch.func.functional_call(self.var['NN'], params, (xb,))
            #loss_bc = 0.5 * (ub_pred - ub).pow(2).mean()
            ub_pred, _ = _stateless_u(self.var, params, xb)  # (Nb,1)
            loss_bc = 0.5 * _reduce_mean((ub_pred - ub)**2)

        # Big boundary weight (tune as needed)
        beta_bc = self.var.get('beta_bc', 1.0)   # e.g., 50–500 good starting range

        loss = pde_loss + beta_bc * loss_bc
        self.loss_history.append(float(loss.item()))
        return loss



    def value(self, x, ftol=None):
        """
        x: TorchVect or OrderedDict
        returns: (val, ferr)
        """
        val = self.value_torch(x, ftol)
        return val, 0.0

    def torch_gradient(self, x, gtol=None):
    # FORCE all parameters to require grad regardless of input
        if isinstance(x, TorchVect):
            params_dict = x.td
        else:
            params_dict = x
        
        # Create fresh parameters that definitely require grad
        fresh_params = collections.OrderedDict()
        for name, param in params_dict.items():
            fresh_params[name] = param.detach().clone().requires_grad_(True)
    
        # Use the fresh parameters for computation
        loss = self.value_torch(fresh_params, gtol)
    
        names, tensors = zip(*fresh_params.items())
        grads = torch.autograd.grad(loss, tensors, retain_graph=False, create_graph=False)
    
        gdict = collections.OrderedDict(zip(names, grads))
        return TorchVect(gdict) if isinstance(x, TorchVect) else gdict



    def gradient(self, x, gtol=None):
        """Wrapper for gradient computation"""
        g = self.torch_gradient(x, gtol)
        
        # Compute total gradient norm
        if isinstance(g, TorchVect):
            grad_tensors = list(g.td.values())
        else:
            grad_tensors = list(g.values())
        
        total_norm = torch.norm(torch.cat([g.flatten() for g in grad_tensors])).item()
        #print(f"Total gradient norm: {total_norm}")
        
        return g, 0.0

    def hessVec(self, v, x, htol=None):
        raw   = x.td if isinstance(x, TorchVect) else collections.OrderedDict(x)
        v_raw = v.td if isinstance(v, TorchVect) else collections.OrderedDict(v)
        params = self._trainable_params_like(self.var['NN'], raw)  # only here

        names, theta = zip(*params.items())
        vtheta = [v_raw[k] for k in names]

        loss = self.value_torch(params, htol)

        g_list = torch.autograd.grad(loss, theta, create_graph=True)
        flat_g = torch.cat([g.reshape(-1) for g in g_list])
        flat_v = torch.cat([vv.reshape(-1) for vv in vtheta])
        gv = torch.dot(flat_g, flat_v)
        Hv_list = torch.autograd.grad(gv, theta, retain_graph=False)

        Hv = collections.OrderedDict((k, hv) for k, hv in zip(names, Hv_list))
        return (TorchVect(Hv) if isinstance(x, TorchVect) else Hv), 0.0
    def hessVec_JJ(self, v, x, htol=None, use_gn=True):
    

        # unwrap inputs
        raw   = x.td if isinstance(x, TorchVect) else collections.OrderedDict(x)
        v_raw = v.td if isinstance(v, TorchVect) else collections.OrderedDict(v)

        # clone into leaf tensors on correct device/dtype and requires_grad=True
        params = self._trainable_params_like(self.var['NN'], raw)
        names, theta = zip(*params.items())
        vtheta = [v_raw[k] for k in names]

        if not use_gn:
            # exact Hessian (as before)
            loss = self.value_torch(params, htol)
            g_list = torch.autograd.grad(loss, theta, create_graph=True)
            flat_g = torch.cat([g.reshape(-1) for g in g_list])
            flat_v = torch.cat([vv.reshape(-1) for vv in vtheta])
            gv = torch.dot(flat_g, flat_v)
            Hv_list = torch.autograd.grad(gv, theta, retain_graph=False)
            Hv = collections.OrderedDict((k, hv) for k, hv in zip(names, Hv_list))
            return (TorchVect(Hv) if isinstance(x, TorchVect) else Hv), 0.0

        # --------- Gauss–Newton path: Hv ≈ α/N Jᵀ(J v) ---------
        x_in = self.var['inputs_xy']

        def r_of_theta(*theta_list):
            local = collections.OrderedDict((kname, t) for kname, t in zip(names, theta_list))
            u, xg = _u_with_hard_bc(self.var, local, x_in)
            gu = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
            lap = 0.0
            for d in range(2):
                g_d = gu[:, d:d+1]
                lap += torch.autograd.grad(g_d, xg, torch.ones_like(g_d), create_graph=True)[0][:, d:d+1]
            k     = self.var['k']
            gradk = self.var['gradk']
            b     = self.var['b']
            r = -(gradk * gu).sum(dim=1, keepdim=True) - k * lap - b
            return r

        # Jv with a graph on r(theta):
        r_val, Jv = torch.autograd.functional.jvp(lambda *t: r_of_theta(*t),tuple(theta),tuple(vtheta),create_graph=True,  strict=True)

        # Hv = α * ∂(r)ᵀ/∂θ · (Jv)
        gJv = torch.autograd.grad(r_val,theta,grad_outputs=Jv,retain_graph=False,allow_unused=True)

        alpha = float(self.var.get('alpha', 1.0))
        N = int(r_val.numel())
        scale = alpha/max(N,1)
        Hv_list = [torch.zeros_like(p) if g is None else scale * g for p, g in zip(theta, gJv)]
        Hv = collections.OrderedDict((k, hv) for k, hv in zip(names, Hv_list))
        return (TorchVect(Hv) if isinstance(x, TorchVect) else Hv), 0.0
    
    
class NNSetup:
    def __init__(self, NN_dim, n, alpha, beta, n_samples=1):
        self.n      = n
        self.NN_dim = NN_dim
        self.nsamps = n_samples
        self.beta   = beta
        self.alpha  = alpha
        self.R      = []

        # ---- (A) Build a uniform grid in [0,1]^2
        xs = np.linspace(0.0, 1.0, n+1)
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        coords_np = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N,2)
        self.inputs_xy = torch.as_tensor(coords_np, dtype=torch.float64)  # (N,2)

        # ---- (B) Boundary detection (grid points on ∂Ω)
        x = self.inputs_xy[:, 0]
        y = self.inputs_xy[:, 1]
        on_bdry = (x == 0) | (x == 1) | (y == 0) | (y == 1)
        self.boundary_idx = on_bdry.nonzero(as_tuple=False).view(-1).cpu().numpy()
        self.boundary_xy  = self.inputs_xy[self.boundary_idx]
        all_idx = np.arange(self.inputs_xy.shape[0], dtype=np.int64)
        self.interior_idx = np.setdiff1d(all_idx, self.boundary_idx)


        # ---- (C) Define manufactured fields and compute tensors
        (self.kappa_value_list,
         self.f_value_list,
         self.u_solution_tensor,
         self.grad_kappa_value_list) = self.generate_analytic_data(self.inputs_xy)

        # shape & dtype like before
        self.kappa_value_list      = self.kappa_value_list.reshape(-1, 1).contiguous()
        self.f_value_list          = self.f_value_list.reshape(-1, 1).contiguous()
        self.u_solution_tensor     = self.u_solution_tensor.reshape(-1, 1).contiguous()
        # boundary values (should be ~0 with our B(x,y) mask)
        self.boundary_u = self.u_solution_tensor[self.boundary_idx]

        # ---- (D) NN
        self.NN = FullyConnectedNN(NN_dim)
        self.NN.to(torch.float64)
        for p in self.NN.parameters(): p.requires_grad_(True)
        print("NN parameter gradient status:")
        for name, p in self.NN.named_parameters():
            print(f"  {name}: requires_grad={p.requires_grad}")

    def get_initial_params(self):
        params = collections.OrderedDict()
        for name, p in self.NN.named_parameters():
            if torch.is_tensor(p):
                params[name] = p.clone().requires_grad_(True)
            
        return params

    def NN_update(self, xvec):
        with torch.no_grad():
            offset = 0
            for p in self.NN.parameters():
                numel = p.numel()
                p.copy_(xvec[offset:offset+numel].view_as(p))
                offset += numel
        for p in self.NN.parameters(): p.requires_grad_(True)

    # ========= NEW: purely analytic data generator =========
    def generate_analytic_data(self, xy: torch.Tensor):
        """
        Inputs:
          xy : (N,2) float64 tensor of points in [0,1]^2
        Returns:
          kappa_values (N,), f_values (N,), u_true_values (N,), grad_kappa_values (N,2)
        """

        # Choose u* and kappa below.
        # (1) Boundary mask B ensures u* = 0 on ∂Ω
        def B(x, y):
            return x*(1.0 - x)*y*(1.0 - y)

        # (2) A smooth interior "shape" g(x,y)
        def g(x, y):
            # mildly nontrivial but smooth
            return 1.0 + 0.25*torch.sin(2*math.pi*x) * torch.sin(1*math.pi*y) + 0.1*x*y

        # Manufactured solution u*(x,y)
        def u_true(x, y):
            return B(x, y) * g(x, y)

        # (3) Choose kappa
        def kappa_fn(x, y):
            # Constant:
            #return 1.1*torch.ones_like(x)
            # Or variable (uncomment to try):
             #return 1.1 + 0.2*torch.sin(2*math.pi*x)*torch.cos(2*math.pi*y)
             return torch.full_like(x, 1.1) 

        # Autodiff to get grad u, lap u, grad k
        xy = xy.clone().detach().requires_grad_(True)
        x = xy[:, :1]
        y = xy[:, 1:2]

        kappa = kappa_fn(x, y)                        # (N,1)
        uStar = u_true(x, y)                          # (N,1)

        # ∇u*
        gu = torch.autograd.grad(uStar, xy,
                                 grad_outputs=torch.ones_like(uStar),
                                 create_graph=True, retain_graph=True)[0]   # (N,2)
        # Δu*
        lap = 0.0
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap_d = torch.autograd.grad(g_d, xy,
                                        grad_outputs=torch.ones_like(g_d),
                                        create_graph=True, retain_graph=True)[0][:, d:d+1]
            lap = lap + lap_d                         # (N,1)

        # ∇k
        #gradk = torch.autograd.grad(kappa, xy,
        #                            grad_outputs=torch.ones_like(kappa),
        #                            create_graph=True, retain_graph=True)[0]  # (N,2)
        gradk = torch.zeros_like(xy)

        # f = - div(k grad u) = -(∇k · ∇u) - k Δu
        f_vals = -(gradk * gu).sum(dim=1, keepdim=True) - kappa * lap   # (N,1)

        # detach all (we just want targets / coefficients)
        return (kappa.detach().view(-1),
                f_vals.detach().view(-1),
                uStar.detach().view(-1),
                gradk.detach())  # (N,2)
    
    def compute_metrics(self, params=None, use_hard_bc=True):
        """
        Returns a dict with L2_rel, Linf, residual_RMSE, residual_max on the class grid.
        """
    
        # unwrap TorchVect
        if params is not None and hasattr(params, "td"):
            params = params.td

        # predict on grid
        X, Y, U_pred = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)

        # true u on same grid
        xy = self.inputs_xy.detach().cpu().numpy()
        ut = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
        order = np.lexsort((xy[:,1], xy[:,0]))
        U_true = ut[order].reshape(X.shape)

        E = U_pred - U_true
        L2_rel = np.linalg.norm(E.ravel()) / max(1e-16, np.linalg.norm(U_true.ravel()))
        Linf   = np.max(np.abs(E))

        # residual on grid (uses stored k, gradk, b)
        k  = self.kappa_value_list.detach().cpu().numpy().reshape(-1,1)
        gk = self.grad_kappa_value_list.detach().cpu().numpy()
        b  = self.f_value_list.detach().cpu().numpy().reshape(-1,1)

        # compute ∇u and Δu from NN
        dev   = next(self.NN.parameters()).device
        dtype = next(self.NN.parameters()).dtype
        xg = self.inputs_xy.to(device=dev, dtype=dtype).detach().clone().requires_grad_(True)

        if params is None:
            u = self.NN(xg)
        else:
            params = collections.OrderedDict((k, v.to(dev, dtype)) for k, v in params.items())
            u = torch.func.functional_call(self.NN, params, (xg,))

        if use_hard_bc:
            B = (xg[:, :1]*(1-xg[:, :1])) * (xg[:, 1:2]*(1-xg[:, 1:2]))
            if params is not None and ('cB' in params):
                scale = torch.exp(params['cB'])
            elif hasattr(self.NN, 'cB'):
                scale = torch.exp(self.NN.cB)
            else:
                scale = torch.tensor(1.0, dtype=dtype, device=dev)
            u = scale * B * u

        gu = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
        lap = 0.0
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap += torch.autograd.grad(g_d, xg, torch.ones_like(g_d), create_graph=True)[0][:, d:d+1]

        gu_np  = gu.detach().cpu().numpy()
        lap_np = lap.detach().cpu().numpy()
        res = -(gk * gu_np).sum(axis=1, keepdims=True) - k * lap_np - b
        residual_RMSE = float(np.sqrt(np.mean(res**2)))
        residual_max  = float(np.max(np.abs(res)))

        return {
            "L2_rel": float(L2_rel),
            "Linf": float(Linf),
            "residual_RMSE": residual_RMSE,
            "residual_max": residual_max,
        }

    def plot_surface(self, X=None, Y=None, Z=None, title=None, prefer_trisurf=False):
        """
        Robust 3D surface plotter.
        - If X/Y/Z are None, it uses self.inputs_xy and self.u_solution_tensor.
        - Accepts torch tensors or numpy arrays.
        - If points form a structured grid, uses plot_surface; else falls back to plot_trisurf.
        """
    
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        def _to_np(a):
            if a is None:
                return None
            if hasattr(a, "detach"):  # torch.Tensor
                a = a.detach().cpu().numpy()
            return np.asarray(a, dtype=np.float64)

        # ---- Gather data ----
        if X is None or Y is None or Z is None:
            xy = _to_np(self.inputs_xy)            # (N,2)
            z  = _to_np(self.u_solution_tensor).reshape(-1)
            # sort to grid order and reshape
            order = np.lexsort((xy[:, 1], xy[:, 0]))
            x_sorted, y_sorted, z_sorted = xy[order, 0], xy[order, 1], z[order]
            nx, ny = np.unique(x_sorted).size, np.unique(y_sorted).size
            Xg = x_sorted.reshape(nx, ny)
            Yg = y_sorted.reshape(nx, ny)
            Zg = z_sorted.reshape(nx, ny)
            use_grid = True
        else:
            Xg, Yg, Zg = _to_np(X), _to_np(Y), _to_np(Z)
            # try to recognize a grid
            if Xg.ndim == Yg.ndim == Zg.ndim == 2 and Xg.shape == Yg.shape == Zg.shape and not prefer_trisurf:
                use_grid = True
            else:
                # 1D point lists → try grid reshape; else trisurf
                X1, Y1, Z1 = Xg.reshape(-1), Yg.reshape(-1), Zg.reshape(-1)
                nx, ny = np.unique(X1).size, np.unique(Y1).size
                if nx * ny == X1.size and not prefer_trisurf:
                    order = np.lexsort((Y1, X1))
                    Xg = X1[order].reshape(nx, ny)
                    Yg = Y1[order].reshape(nx, ny)
                    Zg = Z1[order].reshape(nx, ny)
                    use_grid = True
                else:
                    use_grid = False
                    Xg, Yg, Zg = X1, Y1, Z1  # for trisurf

        # ---- Finite checks ----
        if not np.isfinite(Xg).all() or not np.isfinite(Yg).all() or not np.isfinite(Zg).all():
            raise ValueError("Non-finite values found in plot arrays.")

        # ---- Plot ----
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        if use_grid:
            ax.plot_surface(Xg, Yg, Zg, linewidth=0, antialiased=True)
        else:
            ax.plot_trisurf(Xg, Yg, Zg, linewidth=0, antialiased=True)

        if title: ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("value")
        plt.tight_layout()
        plt.show()
    
    
    # inside class NNSetup
    def diagnose_scale(self, params=None):
        """
        Computes the best scalar s* for u_pred = s * B*N_raw (least-squares fit to u_true),
        returns suggested cB* = log(s*), and (rel L2 before, after).
        """
        import  math

        if params is not None and hasattr(params, "td"):
            params = params.td

        dev   = next(self.NN.parameters()).device
        dtype = next(self.NN.parameters()).dtype
        x = self.inputs_xy.to(device=dev, dtype=dtype)

        with torch.no_grad():
            if params is None:
                n_raw = self.NN(x)
            else:
                params = collections.OrderedDict((k, v.to(dev, dtype)) for k, v in params.items())
                n_raw = torch.func.functional_call(self.NN, params, (x,))

            B = (x[:, :1]*(1 - x[:, :1])) * (x[:, 1:2]*(1 - x[:, 1:2]))
            v = (B * n_raw).detach().cpu().numpy().reshape(-1)  # basis vector to scale
            u = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)

            num = float(np.dot(u, v))
            den = float(np.dot(v, v)) + 1e-16
            s_opt = num / den
            cB_opt = float(np.log(max(s_opt, 1e-16)))

            # current prediction (with existing cB if any)
            if params is not None and ('cB' in params):
                s_cur = float(np.exp(params['cB'].detach().cpu().numpy()))
            elif hasattr(self.NN, 'cB'):
                s_cur = float(np.exp(self.NN.cB.detach().cpu().numpy()))
            else:
                s_cur = 1.0

            U_cur = (s_cur * v)
            U_opt = (s_opt * v)
            L2_rel_cur = np.linalg.norm(U_cur - u) / max(1e-16, np.linalg.norm(u))
            L2_rel_opt = np.linalg.norm(U_opt - u) / max(1e-16, np.linalg.norm(u))

        print(f"[scale diag] s_cur={s_cur:.4e}, s_opt={s_opt:.4e}, ΔcB={math.log(max(s_opt,1e-16))-math.log(max(s_cur,1e-16)):.4e}")
        print(f"[scale diag] rel L2 (current) = {L2_rel_cur:.3e}, (with s_opt) = {L2_rel_opt:.3e}")
        return {"s_cur": s_cur, "s_opt": s_opt, "cB_opt": cB_opt,
                "L2_rel_current": float(L2_rel_cur), "L2_rel_opt": float(L2_rel_opt)}


    def _predict_u_grid(self, params=None, use_hard_bc=True):
        """
        Evaluate the NN on the class grid and return (X, Y, U) as numpy 2D arrays.
        - params: OrderedDict or TorchVect (optional). If None, uses live NN weights.
        - use_hard_bc: if True, returns B(x,y)*N(x,y)*exp(cB if present).
        """
    
        import  collections

        # unwrap TorchVect if needed
        if params is not None and hasattr(params, "td"):
             params = params.td
        # device/dtype
        dev   = next(self.NN.parameters()).device
        dtype = next(self.NN.parameters()).dtype

        # inputs (no grad needed for plotting)
        x = self.inputs_xy.to(device=dev, dtype=dtype)

        # forward (stateless if params provided)
        if params is None:
            with torch.no_grad():
                n_raw = self.NN(x)                             # (N,1)
        else:
            # ensure params tensors on same device/dtype
            params = collections.OrderedDict((k, v.to(dev, dtype)) for k, v in params.items())
            with torch.no_grad():
                n_raw = torch.func.functional_call(self.NN, params, (x,))  # (N,1)

        if use_hard_bc:
            # boundary mask B(x,y)
            B = (x[:, :1] * (1 - x[:, :1])) * (x[:, 1:2] * (1 - x[:, 1:2]))
            # optional global scale exp(cB) if present
            if params is not None and ('cB' in params):
                scale = torch.exp(params['cB'])
            elif hasattr(self.NN, 'cB'):
                scale = torch.exp(self.NN.cB)
            else:
                scale = torch.tensor(1.0, dtype=dtype, device=dev)
            u_pred = scale * B * n_raw
        else:
            u_pred = n_raw

        # convert to numpy and reshape onto grid
        xy = self.inputs_xy.detach().cpu().numpy()
        up = u_pred.detach().cpu().numpy().reshape(-1)

        order = np.lexsort((xy[:,1], xy[:,0]))  # sort by x then y
        x_sorted = xy[order, 0]; y_sorted = xy[order, 1]; u_sorted = up[order]
        nx = np.unique(x_sorted).size
        ny = np.unique(y_sorted).size

        X = x_sorted.reshape(nx, ny)
        Y = y_sorted.reshape(nx, ny)
        U = u_sorted.reshape(nx, ny)
        return X, Y, U

    def plot_nn_solution(self, params=None, title=r"NN prediction $u_\theta(x,y)$", use_hard_bc=True):
        """
        Compute NN prediction on the stored grid and plot a 3D surface.
        - params: OrderedDict or TorchVect with weights (optional)
        - use_hard_bc: multiply by B(x,y) (and exp(cB)) to match training ansatz
        """
        X, Y, U = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)
        self.plot_surface(X, Y, U, title=title)

    def plot_nn_error(self, params=None, title=r"|$u_\theta - u^*$|", use_hard_bc=True):
        """
        Plot absolute error surface |u_pred - u_true| on the grid.
        """
    
        X, Y, U_pred = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)
        # ground truth from class tensor
        xy = self.inputs_xy.detach().cpu().numpy()
        u_true = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
        order = np.lexsort((xy[:,1], xy[:,0]))
        U_true = u_true[order].reshape(X.shape)
        E = np.abs(U_pred - U_true)
        self.plot_surface(X, Y, E, title=title)
        
    def plot_residual_surface(self, params=None, title=r"PDE residual $r(x,y)$", use_hard_bc=True):
        """
        r(x,y) = -(∇k · ∇u) - k Δu - b  evaluated with the NN.
        """
    

        # unwrap TorchVect
        if params is not None and hasattr(params, "td"):
            params = params.td

        dev   = next(self.NN.parameters()).device
        dtype = next(self.NN.parameters()).dtype

        xg = self.inputs_xy.to(device=dev, dtype=dtype).detach().clone().requires_grad_(True)

        # forward (stateless if params is provided)
        if params is None:
            u = self.NN(xg)
        else:
            params = collections.OrderedDict((k, v.to(dev, dtype)) for k, v in params.items())
            u = torch.func.functional_call(self.NN, params, (xg,))

        if use_hard_bc:
            B = (xg[:, :1]*(1 - xg[:, :1])) * (xg[:, 1:2]*(1 - xg[:, 1:2]))
            if params is not None and ('cB' in params):
                scale = torch.exp(params['cB'])
            elif hasattr(self.NN, 'cB'):
                scale = torch.exp(self.NN.cB)
            else:
                scale = torch.tensor(1.0, dtype=dtype, device=dev)
            u = scale * B * u

        # ∇u and Δu
        gu = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
        lap = 0.0
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap += torch.autograd.grad(g_d, xg, torch.ones_like(g_d), create_graph=True)[0][:, d:d+1]

        # coefficients from stored tensors (already aligned to inputs_xy)
        k_np  = self.kappa_value_list.detach().cpu().numpy().reshape(-1, 1)
        gk_np = self.grad_kappa_value_list.detach().cpu().numpy()
        b_np  = self.f_value_list.detach().cpu().numpy().reshape(-1, 1)

        gu_np  = gu.detach().cpu().numpy()
        lap_np = lap.detach().cpu().numpy()

        res = -(gk_np * gu_np).sum(axis=1, keepdims=True) - k_np * lap_np - b_np  # (N,1)

        # reshape to grid and plot
        xy = self.inputs_xy.detach().cpu().numpy()
        order = np.lexsort((xy[:,1], xy[:,0]))
        nx = np.unique(xy[:,0]).size
        ny = np.unique(xy[:,1]).size
        R = res[order].reshape(nx, ny)
        X = xy[order,0].reshape(nx, ny)
        Y = xy[order,1].reshape(nx, ny)

        self.plot_surface(X, Y, R, title=title)
        # inside class NNSetup
    def plot_error_heatmap(self, params=None, title=r"|$u_\theta - u^*$|", use_hard_bc=True):
        
        if params is not None and hasattr(params, "td"):
            params = params.td

        # predict on grid
        X, Y, U_pred = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)

        # true grid
        xy = self.inputs_xy.detach().cpu().numpy()
        u_true = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
        order = np.lexsort((xy[:,1], xy[:,0]))
        U_true = u_true[order].reshape(X.shape)

        E = np.abs(U_pred - U_true)

        plt.figure(figsize=(6,5))
        im = plt.imshow(E.T, extent=[X.min(), X.max(), Y.min(), Y.max()],origin='lower', aspect='auto')
        plt.title(title); plt.xlabel("x"); plt.ylabel("y")
        plt.colorbar(im, label="|error|")
        plt.tight_layout(); plt.show()
        
    def plot_centerline(self, params=None, axis='y', value=0.5, use_hard_bc=True):
        """
        Plot u_pred vs u_true along x=value (axis='x') or y=value (axis='y').
        """
    

        X, Y, U = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)

        if axis == 'y':
            j = np.argmin(np.abs(Y[0,:] - value))
            xs = X[:, j]; up = U[:, j]
            # true
            xy = self.inputs_xy.detach().cpu().numpy()
            ut = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
            order = np.lexsort((xy[:,1], xy[:,0]))
            U_true = ut[order].reshape(X.shape)[:, j]
            plt.plot(xs, U_true, 'k-',  label='u*')
            plt.plot(xs, up,     'b--', label='u_theta')
            plt.xlabel('x'); plt.ylabel('u'); plt.title(f'Centerline y={value}')
            plt.legend(); plt.tight_layout(); plt.show()
        else:
            i = np.argmin(np.abs(X[:,0] - value))
            ys = Y[i, :]; up = U[i, :]
            xy = self.inputs_xy.detach().cpu().numpy()
            ut = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
            order = np.lexsort((xy[:,1], xy[:,0]))
            U_true = ut[order].reshape(X.shape)[i, :]
            plt.plot(ys, U_true, 'k-',  label='u*')
            plt.plot(ys, up,     'b--', label='u_theta')
            plt.xlabel('y'); plt.ylabel('u'); plt.title(f'Centerline x={value}')
            plt.legend(); plt.tight_layout(); plt.show()


    def returnVars(self, useEuclidean):
      var = {'beta':self.beta,
             'n':self.n,
             'nu':self.n,
             'nz':sum(p.numel() for p in self.NN.parameters()),
             'alpha':self.alpha,
             #'A':self.A,
             #'M':self.M,
             'NN':self.NN,
             'k':self.kappa_value_list,
             'b':self.f_value_list,
             'ud':self.u_solution_tensor,
             'useEuclidean':useEuclidean,
             #'mesh':torch.tensor(self.domain.geometry.x[:,0:1], dtype=torch.float64).T,
             #'mesh': torch.as_tensor(self.domain.geometry.x, dtype=torch.float64),  
             'inputs_xy': self.inputs_xy,            # (N,2)
             'gradk': self.grad_kappa_value_list,
             'bdry_xy':self.boundary_xy,
             'bdry_u':self.boundary_u,
             'lambda_reg': 1e-4,            # <-- choose λ here (0 to disable)
             'l2_exclude': ('bias', 'cB'),
             'interior_idx': torch.as_tensor(self.interior_idx,dtype=torch.long)
             
            }
      return var

    


def _group_reduce(m: int, n: int, mode: str, dtype, device):
    """
    Make A in R^{n x m} that groups m->n with disjoint blocks.
    mode='orthonormal' -> entries 1/sqrt(g) in each block (rows orthonormal)
    mode='average'     -> entries 1/g in each block (rows sum to 1)
    """
    g = m // n
    A = torch.zeros(n, m, dtype=dtype, device=device)
    val = (1.0 / (g ** 0.5)) if mode == "orthonormal" else (1.0 / g)
    for r in range(n):
        A[r, r*g:(r+1)*g] = val
    return A

def _linear_interp(m: int, n: int, dtype, device):
    """Simple 1D linear interpolation rows that sum to 1 (not orthonormal)."""
    idx = torch.linspace(0, m - 1, steps=n, dtype=dtype, device=device)
    A = torch.zeros(n, m, dtype=dtype, device=device)
    for r, x in enumerate(idx):
        i0 = int(torch.floor(x).item())
        i1 = min(i0 + 1, m - 1)
        t = float(x - i0)
        A[r, i0] = 1.0 - t
        A[r, i1] += t
    A = A / (A.sum(dim=1, keepdim=True) + 1e-12)
    return A

def _orthonormalize_rows(A: torch.Tensor) -> torch.Tensor:
    """
    Return Ã with orthonormal rows spanning the same row space as A.
    Uses QR on A^T:  A^T = Q R  -> rows(Ã) = rows(Q^T).
    """
    Q, _ = torch.linalg.qr(A.t(), mode='reduced')  # Q: (m x n)
    return Q.t()  # (n x m), rows are orthonormal

def _reduce_matrix(m_from: int, m_to: int, *, mode: str, dtype=torch.float64, device=None):
    """
    Build A in R^{m_to x m_from} mapping fine->coarse hidden units.
    - If equal: identity (orthonormal).
    - If divisible: disjoint grouping (orthonormal or average).
    - Else: linear interp then (if mode='orthonormal') row-orthonormalize.
    """
    device = device or torch.device("cpu")
    if m_from == m_to:
        return torch.eye(m_to, m_from, dtype=dtype, device=device)

    if m_from % m_to == 0:
        return _group_reduce(m_from, m_to, mode=mode, dtype=dtype, device=device)

    # Not divisible: start with interp
    A = _linear_interp(m_from, m_to, dtype=dtype, device=device)
    if mode == "orthonormal":
        A = _orthonormalize_rows(A)
    return A

# -------- layer index inference --------

def _infer_layer_index_from_name(name: str, n_linear_layers: int) -> int:
    m = re.search(r'\.(\d+)\.(weight|bias)$', name)
    if m:
        idx = int(m.group(1))
        return min(max(idx, 0), n_linear_layers - 1)
    m = re.search(r'(\d+)', name)
    if m:
        idx = int(m.group(1))
        if idx > 0: idx -= 1  # fc1/layer1 -> 0-based
        return min(max(idx, 0), n_linear_layers - 1)
    return 0

# -------- build parameter-wise restriction --------

def _build_R_ops(
    cur_sizes,          # fine widths:  [in_f, h1_f, ..., out_f]
    next_sizes,         # coarse widths:[in_c, h1_c, ..., out_c]
    state_dict: OrderedDict,
    *,
    mode: str = "orthonormal",      # "orthonormal" (default) or "average"
    dtype=torch.float64,
    device=None,
    mapping_mode: str = "by_name",  # or "by_order"
):
    """
    Returns:
      R_ops[name]       = 2D op mapping vec(param_fine[name]) -> vec(param_coarse[name])
      next_shapes[name] = target shape on coarse net
    Ensures every key in state_dict appears in R_ops (identity fallback).
    """
    device = device or torch.device("cpu")
    cur, nxt = list(cur_sizes), list(next_sizes)
    L = len(cur) - 1

    R_ops = {}
    next_shapes = {}
    w_idx = b_idx = 0

    for name, tensor in state_dict.items():
        is_weight = name.endswith("weight")
        is_bias   = name.endswith("bias")

        ell = (
            min(w_idx, L-1) if (mapping_mode=="by_order" and is_weight)
            else min(b_idx, L-1) if (mapping_mode=="by_order" and is_bias)
            else _infer_layer_index_from_name(name, L)
        )

        if is_weight and tensor.ndim == 2 and 0 <= ell < L:
            o_f, i_f = tensor.shape
            if o_f == cur[ell+1] and i_f == cur[ell]:
                o_c, i_c = nxt[ell+1], nxt[ell]
                A_out = _reduce_matrix(o_f, o_c, mode=mode, dtype=dtype, device=device)  # (o_c, o_f)
                A_in  = _reduce_matrix(i_f, i_c, mode=mode, dtype=dtype, device=device)  # (i_c, i_f)
                # vec(Wc) = (A_out ⊗ A_in) vec(Wf)
                Rk = torch.kron(A_out, A_in)
                R_ops[name] = Rk.to(dtype=dtype, device=device)
                next_shapes[name] = (o_c, i_c)
                if mapping_mode == "by_order": w_idx += 1
                continue  # handled

        if is_bias and 0 <= ell < L:
            o_f = tensor.numel()
            if o_f == cur[ell+1]:
                o_c = nxt[ell+1]
                A_out = _reduce_matrix(o_f, o_c, mode=mode, dtype=dtype, device=device)  # (o_c, o_f)
                R_ops[name] = A_out.to(dtype=dtype, device=device)
                next_shapes[name] = (o_c,)
                if mapping_mode == "by_order": b_idx += 1
                continue  # handled

        # Fallback for anything unexpected: identity pass-through
        n = tensor.numel()
        R_ops[name] = torch.eye(n, n, dtype=dtype, device=device)
        next_shapes[name] = tuple(tensor.shape)

    # Guarantee every key exists (if any were skipped above)
    for name, t in state_dict.items():
        if name not in R_ops:
            n = t.numel()
            R_ops[name] = torch.eye(n, n, dtype=dtype, device=device)
            next_shapes[name] = tuple(t.shape)

    return R_ops, next_shapes

# -------- user-facing: build TorchVect operator R --------

def restriction_R_from_dims(
    next_sizes,                 # coarse widths (e.g., [2, 50, 50, 1])
    cur_sizes,                  # fine widths   (e.g., [2,100,100,1])
    x_fine,                     # TorchVect fine params (.td = OrderedDict name->tensor)
    *,
    mode: str = "orthonormal",  # "orthonormal" (row-orthonormal) or "average"
    mapping_mode: str = "by_name",
    dtype=torch.float64,
):
    fine_sd = x_fine.td
    device = next(iter(fine_sd.values())).device if len(fine_sd) else torch.device("cpu")
    R_ops, next_shapes = _build_R_ops(
        cur_sizes, next_sizes, fine_sd, mode=mode, dtype=dtype, device=device, mapping_mode=mapping_mode
    )

    # Pack into TorchVect operator with shapes_map; also add a .shape your step checks
    R = TorchVect(OrderedDict(R_ops), isRop=True, shapes_map=next_shapes)

    # Synthetic global shape for legacy checks (sum of block sizes)
    total_rows = sum(W.shape[0] for W in R_ops.values())
    total_cols = sum(W.shape[1] for W in R_ops.values())
    R.shape = [total_rows, total_cols]      # <- keeps your existing `x.shape[0] != R0.shape[0]` check happy

    return R

def _dot_euclid(a, b):
    # a, b: TorchVect
    s = 0.0
    for k, va in a.td.items():
        vb = b.td[k]
        s += float((va.reshape(-1) @ vb.reshape(-1)).item())
    return s

def _as_vect(v):
    """
    Accept either a TorchVect or (TorchVect, ...) tuple and return the TorchVect.
    Raise if it's something unexpected.
    """
    if isinstance(v, tuple):
        v = v[0]
    if hasattr(v, "td"):  # TorchVect-like
        return v
    raise TypeError(f"Expected TorchVect (or tuple of), got {type(v)}")

def _rand_unit_like(x):
    d = x.clone()
    tot = 0.0
    for k, t in d.td.items():
        r = torch.randn_like(t)
        d.td[k] = r
        tot += float((r.reshape(-1) @ r.reshape(-1)).item())
    tot = (tot ** 0.5) + 1e-12
    for k in d.td:
        d.td[k] /= tot
    return d

# --- the robust checker ---

def check_grad_and_hv(problem, x):
    # Make eval deterministic (no dropout/BN drift)
    try:
        problem.var.NN.eval()
    except Exception:
        pass

    d = _rand_unit_like(x)

    f0, _ = problem.obj_smooth.value(x, 0.0)

    g_raw = problem.obj_smooth.gradient(x)
    g = _as_vect(g_raw)
    dirderiv = _dot_euclid(g, d)

    print("\nFinite Difference Gradient Check")
    print(f"{'t':>12s} {'DirDeriv':>14s} {'FinDiff':>14s} {'Error':>12s}")
    for t in [1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
        fp, _ = problem.obj_smooth.value(x + d * t, 0.0)
        fm, _ = problem.obj_smooth.value(x - d * t, 0.0)
        fd = (fp - fm) / (2.0 * t)
        err = abs(dirderiv - fd)
        print(f"{t:12.4e} {dirderiv:14.6e} {fd:14.6e} {err:12.4e}")

    # Hessian-vector (if available)
    if hasattr(problem.obj_smooth, "hessVec"):
        hv_raw = problem.obj_smooth.hessVec(d, x)
        hv = _as_vect(hv_raw)
        hv_dir = _dot_euclid(hv, d)

        print("\nFinite Difference Hessian Check")
        print(f"{'t':>12s} {'Hv·d':>14s} {'FinDiff':>14s} {'Error':>12s}")
        for t in [1.0,1e-1,1e-2,1e-3,1e-4]:
            gp_raw = problem.obj_smooth.gradient(x + d * t)
            gm_raw = problem.obj_smooth.gradient(x - d * t)
            gp = _as_vect(gp_raw); gm = _as_vect(gm_raw)
            fd_vec = (gp - gm) * (1.0 / (2.0 * t))   # TorchVect ops
            fd = _dot_euclid(fd_vec, d)
            err = abs(hv_dir - fd)
            print(f"{t:12.4e} {hv_dir:14.6e} {fd:14.6e} {err:12.4e}")
            


def tr_style_eval(nn, var, use_interior=True, chunk=8192):
    """
    Compute the exact objective that TR uses:
      value := 0.5 * mean( r^2 ) over interior points,
      r = -(∇k·∇u) - k Δu - b     with hard-BC u = exp(cB)*B*n_raw
    Returns (value, rmse).
    """
    dev   = next(nn.parameters()).device
    dtype = next(nn.parameters()).dtype

    X     = var['inputs_xy'].to(device=dev, dtype=dtype)
    k     = var['k'].to(device=dev, dtype=dtype)
    b     = var['b'].to(device=dev, dtype=dtype)
    gradk = var['gradk'].to(device=dev, dtype=dtype)
    alpha = float(var.get('alpha', 1.0))
    lam   = float(var.get('lambda_reg', 0.0))
    excl  = tuple(var.get('l2_exclude', ()))

    if use_interior and 'interior_idx' in var:
        I = var['interior_idx'].to(device=dev)
    elif use_interior:
        # build interior mask if not provided
        xb, yb = X[:,0], X[:,1]
        I = torch.arange(X.shape[0], device=dev)[~((xb==0)|(xb==1)|(yb==0)|(yb==1))]
    else:
        I = torch.arange(X.shape[0], device=dev)

    N = I.numel()
    if N == 0:
        return 0.0, 0.0

    def _forward_hard_bc(NN, x_xy):
        xg = x_xy if x_xy.requires_grad else x_xy.detach().clone().requires_grad_(True)
        n_raw = NN.net(xg)                       # your MLP body
        cB    = getattr(NN, 'cB', torch.tensor(2.7726, dtype=xg.dtype, device=xg.device))
        B     = (xg[:, :1]*(1-xg[:, :1])) * (xg[:, 1:2]*(1-xg[:, 1:2]))
        u     = torch.exp(cB) * B * n_raw
        return u, xg

    r2_accum = 0.0
    count = 0
    for s in range(0, N, chunk):
        j = I[s: s+chunk]

        with torch.enable_grad():
            Xc = X[j].detach().clone().requires_grad_(True)
            uc, Xc = _forward_hard_bc(nn, Xc)
            g  = torch.autograd.grad(uc, Xc, torch.ones_like(uc), create_graph=True, retain_graph=True)[0]
            lap = 0.0
            for d in range(2):
                gd = g[:, d:d+1]
                lap += torch.autograd.grad(gd, Xc, torch.ones_like(gd), create_graph=False, retain_graph=(d<1))[0][:, d:d+1]

            rc = -(gradk[j] * g).sum(1, keepdim=True) - k[j] * lap - b[j]   # (m,1)
            r2_accum += float((rc*rc).sum().item())
            count    += rc.numel()

    mean_r2 = r2_accum / max(count,1)
    value   = 0.5 * mean_r2
    rmse    = mean_r2**0.5
    l2 = 0.0
    if lam > 0.0:
        for n,p in nn.named_parameters():
            if any(tag in n for tag in excl):
                continue
            l2 += (p*p).sum().item()
        l2 = 0.5*lam*l2
    value = 0.5*alpha*mean_r2+l2
    return value, rmse
            
            
@torch.no_grad()
def eval_pinn_metrics(nn, var, chunk=8192):
    """
    Evaluate PINN quality on the *whole* dataset (in chunks to fit memory).
    Metrics:
      - L2_rel: ||u_pred - u_true|| / ||u_true||
      - Linf:   max |u_pred - u_true|
      - residual_RMSE: sqrt(mean r^2) over interior points (r = -(∇k·∇u) - kΔu - b)
      - residual_max: max |r|
    """
    dev   = next(nn.parameters()).device
    dtype = next(nn.parameters()).dtype

    X     = var['inputs_xy'].to(device=dev, dtype=dtype)
    k     = var['k'].to(device=dev, dtype=dtype)
    b     = var['b'].to(device=dev, dtype=dtype)
    gradk = var['gradk'].to(device=dev, dtype=dtype)
    utrue = var['ud'].to(device=dev, dtype=dtype)

    N = X.shape[0]
    xb, yb = X[:,0], X[:,1]
    bdry = (xb==0)|(xb==1)|(yb==0)|(yb==1)
    interior = (~bdry).nonzero(as_tuple=False).view(-1)

    # accumulators
    num = 0.0; den = 0.0
    linf = 0.0
    r2_sum = 0.0; r_abs_max = 0.0
    m = interior.numel()

    def _forward_hard_bc(NN, x_xy):
        xg = x_xy if x_xy.requires_grad else x_xy.detach().clone().requires_grad_(True)
        n_raw = NN.net(xg)
        cB = getattr(NN, 'cB', torch.tensor(2.7726, dtype=xg.dtype, device=xg.device))
        B  = (xg[:, :1]*(1-xg[:, :1]))*(xg[:, 1:2]*(1-xg[:, 1:2]))
        u  = torch.exp(cB) * B * n_raw
        return u, xg

    nn.eval()
    with torch.no_grad():
        ut_norm2 = (utrue*utrue).sum().item()
    den = ut_norm2**0.5 + 1e-16

    # need grads for residual/laplacian → compute in chunks with enable_grad
    for start in range(0, N, chunk):
        end = min(N, start+chunk)

        with torch.enable_grad():
            Xc = X[start:end].detach().clone().requires_grad_(True)
            uc, Xc = _forward_hard_bc(nn, Xc)
            g = torch.autograd.grad(uc, Xc, torch.ones_like(uc), create_graph=True, retain_graph=True)[0]
            lap = 0.0
            for d in range(2):
                gd = g[:, d:d+1]
                lap += torch.autograd.grad(gd, Xc, torch.ones_like(gd), create_graph=False, retain_graph=(d<1))[0][:, d:d+1]

        up = uc.detach()
        diff = up - utrue[start:end]
        num += (diff*diff).sum().item()
        linf = max(linf, float(diff.abs().max().item()))

        # residual only on interior slice indices
        idx_global = torch.arange(start, end, device=dev)
        mask = ~(((Xc[:,0]==0)|(Xc[:,0]==1)|(Xc[:,1]==0)|(Xc[:,1]==1))).view(-1,1)
        r = -(gradk[start:end] * g).sum(1, keepdim=True) - k[start:end] * lap - b[start:end]
        r = r[mask]
        if r.numel() > 0:
            r2_sum += float((r*r).mean().item() * r.numel())
            r_abs_max = max(r_abs_max, float(r.abs().max().item()))

        # free chunk graphs
        del Xc, uc, g, lap

    L2_rel = (num**0.5) / den
    # RMSE over interior points: we averaged per-chunk means → rescale back
    residual_RMSE = (r2_sum / max(m,1))**0.5
    return {
        'L2_rel': L2_rel,
        'Linf': linf,
        'residual_RMSE': residual_RMSE,
        'residual_max': r_abs_max
    }

def _l1_on_params(module, beta, exclude=('bias','cB')):
    if beta <= 0: return torch.tensor(0., dtype=torch.float64, device=next(module.parameters()).device)
    reg = 0.0
    for n,p in module.named_parameters():
        if any(e in n for e in exclude): continue
        reg = reg + p.abs().sum()
    return beta * reg



def adam_warmstart(nnset, var, steps=1000, lr=1e-3, batch=4096,
                   beta_l1=0.0, beta_l2=0.0,
                   plateau=60, lr_decay=0.5, seed=0,
                   eval_every=25, eval_chunk=8192,
                   target_L2_rel=None, target_res_RMSE=None,
                   use_ema=True, ema_decay=0.995):
    """
    Run Adam on minibatches until targets are reached or plateaued.
    Returns: TorchVect of the best weights (EMA if enabled), ready for TR.
    """
    torch.manual_seed(seed)
    dev   = next(nnset.NN.parameters()).device
    dtype = next(nnset.NN.parameters()).dtype

    X     = var['inputs_xy'].to(device=dev, dtype=dtype)
    k     = var['k'].to(device=dev, dtype=dtype)
    b     = var['b'].to(device=dev, dtype=dtype)
    gradk = var['gradk'].to(device=dev, dtype=dtype)
    N     = X.shape[0]

    nn = nnset.NN
    nn.train()

    # interior indices only (hard BC)
    with torch.no_grad():
        xb, yb = X[:,0], X[:,1]
        bdry = (xb==0)|(xb==1)|(yb==0)|(yb==1)
        I_int = torch.arange(N, device=dev)[~bdry]
        if batch > len(I_int): batch = len(I_int)

    opt = torch.optim.Adam(nn.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                       factor=lr_decay, patience=max(5, plateau//5), verbose=False)

    # EMA state
    ema = None
    if use_ema:
        ema = {k: v.detach().clone() for k,v in nn.state_dict().items()}

    def _ema_update():
        if not use_ema: return
        sd = nn.state_dict()
        for k in ema.keys():
            ema[k].mul_(ema_decay).add_(sd[k], alpha=1.0-ema_decay)

    best = float('inf')
    best_sd = {k:v.detach().clone() for k,v in nn.state_dict().items()}
    stagn = 0

    def _forward_hard_bc(NN, x_xy):
        xg = x_xy if x_xy.requires_grad else x_xy.detach().clone().requires_grad_(True)
        n_raw = NN.net(xg)
        cB = getattr(NN, 'cB', torch.tensor(2.7726, dtype=xg.dtype, device=xg.device))
        B  = (xg[:, :1]*(1-xg[:, :1]))*(xg[:, 1:2]*(1-xg[:, 1:2]))
        u  = torch.exp(cB) * B * n_raw
        return u, xg

    for t in range(1, steps+1):
        opt.zero_grad(set_to_none=True)

        # sample interior minibatch
        I = I_int[torch.randint(low=0, high=len(I_int), size=(batch,), device=dev)]
        x_mb = X[I]; k_mb = k[I]; b_mb = b[I]; gradk_mb = gradk[I]

        # forward + residual loss
        u, xg = _forward_hard_bc(nn, x_mb)
        gu = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
        lap = 0.0
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap += torch.autograd.grad(g_d, xg, torch.ones_like(g_d), create_graph=True)[0][:, d:d+1]
        r = -(gradk_mb * gu).sum(1, keepdim=True) - k_mb * lap - b_mb
        loss_pde = 0.5 * r.pow(2).mean()

        loss_l2 = 0.0
        if beta_l2 > 0:
            for n,p in nn.named_parameters():
                if 'bias' in n or 'cB' in n: continue
                loss_l2 = loss_l2 + (p*p).sum()
            loss_l2 = 0.5 * beta_l2 * loss_l2

        loss_l1 = _l1_on_params(nn, beta_l1)
        loss = loss_pde + loss_l2 + loss_l1

        loss.backward()
        opt.step()
        _ema_update()

        # periodic evaluation
        if t % eval_every == 0:
            # evaluate with current (or EMA) weights
            if use_ema:
                cur_sd = {k:v.detach().clone() for k,v in nn.state_dict().items()}
                nn.load_state_dict(ema, strict=True)
            try:
                metrics = eval_pinn_metrics(nn, var, chunk=eval_chunk)
                tr_val, tr_rmse = tr_style_eval(nn, var, use_interior=True, chunk=eval_chunk)
                #print(f"[Adam eval] TR-style value={tr_val:.6e}, RMSE={tr_rmse:.6e}")
            finally:
                if use_ema:
                    nn.load_state_dict(cur_sd, strict=True)

            # scheduler on residual RMSE (robust)
            sched.step(metrics['residual_RMSE'])

            # early stop targets
            hit_l2  = (target_L2_rel is not None and metrics['L2_rel'] <= target_L2_rel)
            hit_res = (target_res_RMSE is not None and metrics['residual_RMSE'] <= target_res_RMSE)

            score = metrics['residual_RMSE']  # ranking key
            improved = score < best*(1-1e-3)
            if improved:
                best = score; stagn = 0
                best_sd = ({k:v.detach().clone() for k,v in (ema if use_ema else nn.state_dict()).items()})
            else:
                stagn += 1

            # stop if good enough or plateau
            if hit_l2 or hit_res or stagn >= plateau:
                break

    # load best weights (EMA best if enabled)
    nn.load_state_dict(best_sd, strict=True)
    nn.eval()
    return TorchVect(nnset.get_initial_params())

def driver(savestats, name):
    print("driver started")
    np.random.seed(0)

    # settings
    n               = [30, 30]
    NN_dim_fine     = np.array([2, 500, 1])
    NN_dim_coarse   = np.array([2, 250, 1])
    if np.array_equal(NN_dim_fine, NN_dim_coarse):
        meshlist = [NN_dim_fine]
        n        = [n[0]]
    else:
        meshlist = [NN_dim_fine, NN_dim_coarse]
    #meshlist        = [NN_dim_fine, NN_dim_coarse]

    alpha      = 1
    beta       = 0
    derivCheck = False
    lambda_warm = 1e-4
    lambda_TR = 1e-3
    
    


    problems = []
    x0 = None

    def _attach_global_shape(R_op,x_fine):
        """
        R_op.td[name]      : torch.Tensor of shape (coarse_numel, fine_numel)
        R_op.shapes_map    : dict name -> coarse param shape (e.g., (250,2))
        x_fine.td[name]    : fine param tensor with .shape (e.g., (500,2))
        """
        out_shapes_map = dict(R_op.shapes_map)                       # coarse
        in_shapes_map  = {name: x_fine.td[name].shape for name in R_op.td.keys()}  # fine

        # ensure scalars like 'cB' are present and identity-mapped
        if 'cB' not in R_op.td and 'cB' in x_fine.td:
            I = torch.eye(1, dtype=x_fine.td['cB'].dtype, device=x_fine.td['cB'].device)
            R_op.td['cB'] = I
            out_shapes_map['cB'] = (1,)
            in_shapes_map['cB']  = (1,)

        return RWrap(R_op.td, out_shapes_map=out_shapes_map, in_shapes_map=in_shapes_map)

    def _build_R_for_level(next_sizes, cur_sizes, x_fine):
        # try name-based first, fall back to encounter-order; use orthonormal averaging
        R_try = restriction_R_from_dims(next_sizes, cur_sizes, x_fine,
                                        mapping_mode="by_name", mode="orthonormal")
        ok = True
        for pname, W in R_try.td.items():
            out_shape = R_try.shapes_map[pname]
            in_size   = x_fine.td[pname].numel()
            out_size  = int(torch.tensor(out_shape).prod().item()) if len(out_shape) > 0 else 1
            if W.shape[1] != in_size or W.shape[0] != out_size:
                ok = False
                break
        if ok:
            return _attach_global_shape(R_try,x_fine)

        R_try = restriction_R_from_dims(next_sizes, cur_sizes, x_fine,
                                        mapping_mode="by_order", mode="orthonormal")
        for pname, W in R_try.td.items():
            out_shape = R_try.shapes_map[pname]
            in_size   = x_fine.td[pname].numel()
            out_size  = int(torch.tensor(out_shape).prod().item()) if len(out_shape) > 0 else 1
            assert W.shape[1] == in_size,  f"{pname}: R cols {W.shape[1]} != param size {in_size}"
            assert W.shape[0] == out_size, f"{pname}: R rows {W.shape[0]} != target size {out_size}"
        return _attach_global_shape(R_try,x_fine)

    for i in range(len(meshlist)):
        cur_sizes = meshlist[i]
        nnset = NNSetup(meshlist[i], n[i], alpha, beta, n_samples=1)
        var = nnset.returnVars(False)
        var['lambda_reg'] = lambda_warm
        x     = adam_warmstart(nnset, var,steps=5000, lr=5e-3, batch=4096,beta_l1=0.0,beta_l2=1e-6, plateau=200, lr_decay=0.5,eval_every=50, eval_chunk=8192,target_L2_rel=None,target_res_RMSE=None,use_ema=True, ema_decay=0.995)
        if i == 0:
            x0 = x
            print("blub")
        

        if i < len(meshlist) - 1:
            next_sizes = meshlist[i + 1]
            R = _build_R_for_level(next_sizes, cur_sizes, x_fine=x)
            print("second blub")

            # optional strict checks
            for name, W in R.td.items():
                out_shape = R.shapes_map[name]
                in_size   = x.td[name].numel()
                out_size  = int(torch.tensor(out_shape).prod().item()) if len(out_shape) > 0 else 1
                assert W.shape[1] == in_size,  f"{name}: R cols {W.shape[1]} != param size {in_size}"
                assert W.shape[0] == out_size, f"{name}: R rows {W.shape[0]} != target size {out_size}"
        else:
            # coarsest level: identity restriction via same sizes (still orthonormal)
            #R = restriction_R_from_dims(cur_sizes, cur_sizes, x, mapping_mode="by_name", mode="orthonormal")
            R = _attach_global_shape(Reye(x),x)
            
            #R = _attach_global_shape(R)
        var['beta'] = beta

        p = Problem(var, R)
        p.obj_smooth    = NNObjective(var)
        p.obj_nonsmooth = L1TorchNorm(var)
        p.pvector       = L2TVPrimal(var)
        p.dvector       = L2TVDual(var)
        problems.append(p)
        check_grad_and_hv(p, x)

        #if derivCheck:
        #    for j in range(len(meshlist)):
        #        x_chk = TorchVect(nnset.NN.state_dict())
        #        d     = x_chk.clone()
        #        for key, vals in d.td.items():
        #            d.td[key] = vals.copy_(torch.randn(vals.size()))
        #        deriv_check(x_chk, d, problems[j], 1e-4 * np.sqrt(np.finfo(float).eps))
        #        vector_check(x_chk, d, problems[j])

    cnt = {}
    params = set_default_parameters("SPG2")
    params["reltol"]  = False
    params['gtol']    = 5e-4
    params["t"]       = 2 / alpha
    params["ocScale"] = 1 / alpha
    params["maxit"]   = 700
    params['deltamax'] = 1e13
    params['RgnormScale'] = 0.5
    params['RgnormScaleTol'] = 0.5
    # Sanity: evaluate TR-style loss with the warm-start weights
    tmp_nnset = NNSetup(meshlist[0], n[0], alpha, beta, n_samples=1)
    tmp_nnset.NN.load_state_dict(x0.td)
    tmp_nnset.NN.eval()
    #var_eval = tmp_nnset.returnVars(False)
    v0, rmse0 = tr_style_eval(tmp_nnset.NN, problems[0].var, use_interior=True)
    print(f"[Pre-TR] TR-style value={v0:.6e}, RMSE={rmse0:.6e}  (should match first TR row up to rounding)")
    

    print("[Pre-TR] param_hash:", param_hash(tmp_nnset.NN))
    



    


    start_time   = time.time()
    x, cnt_tr    = trustregion(0, x0, params['delta'], problems, params)
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")

    pro_tr = []
    cnt[1] = (cnt_tr, pro_tr)

    print("\nSummary")
    print("           niter     nobjs     ngrad     nhess     nobjn     nprox     ")
    print(
        f"   SGP2:  {cnt[1][0]['iter']:6d}    {cnt[1][0]['nobj1']:6d}    {cnt[1][0]['ngrad']:6d}    "
        f"{cnt[1][0]['nhess']:6d}    {cnt[1][0]['nobj2']:6d}    {cnt[1][0]['nprox']:6d}     "
    )
    print("\nBoundary diagnostics:")
    #with torch.no_grad():
    #    ub_pred = torch.func.functional_call(var['NN'], x.td, (var['bdry_xy'],))
    #    max_bc_err = ub_pred.abs().max().item()
    #    mean_bc_err = ub_pred.abs().mean().item()
    #    print(f"  Max |u| on boundary  = {max_bc_err:.3e}")
    #    print(f"  Mean |u| on boundary = {mean_bc_err:.3e}")
    #print({k: type(v) for k, v in params.items()})

    with torch.no_grad():
        ub_pred, _ = _u_with_hard_bc(var, x.td, var['bdry_xy'])
        print("BC max|u|:", float(ub_pred.abs().max()))



    final_nnset = NNSetup(meshlist[0], n[0], alpha, beta, n_samples=1)
    final_nnset.NN.load_state_dict(x.td)
    final_nnset.NN.eval()
    var = final_nnset.returnVars(False)
    

    final_nnset.plot_surface(title=r"Manufactured solution $u^*(x,y)$")
    #diag = nnset.diagnose_scale()                 # computes s_opt and cB_opt
    
    #with torch.no_grad():
    #    if hasattr(nnset.NN, "cB"):
    #        nnset.NN.cB.fill_(math.log(16.0))    # back to sane init

    apply_best_scale_live(final_nnset)
    final_nnset.plot_nn_solution(title=r"NN prediction $u_\theta$ (after sign+scale)")
    print(final_nnset.compute_metrics())

    
    final_nnset.plot_nn_error(title=r"abs error")
    metrics = final_nnset.compute_metrics()
    print(metrics)
    final_nnset.diagnose_scale()
    
    final_nnset.plot_residual_surface(title=r"PDE residual $r(x,y)$")
    final_nnset.plot_error_heatmap(title=r"|$u_\theta - u^*$|")
    final_nnset.plot_centerline()
    



    


    print("Updated neural network is stored in `updated_nn`.")
    return cnt



cnt = driver(False, "test_run")
