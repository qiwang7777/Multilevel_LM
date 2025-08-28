import torch
import torch.nn as nn
import math

import os
import sys
sys.path.append(os.path.abspath('..'))



from typing import Sequence, Callable, Optional
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
