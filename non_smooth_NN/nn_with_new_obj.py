import os
import sys
sys.path.append(os.path.abspath("/Users/wang/Multilevel_LM/Multilevel_LM"))
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
# Fully connected neural network construction
# Set up BB
import ufl, dolfinx
from mpi4py import MPI
#import torch
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import collections
import time, torch


from non_smooth.checks import deriv_check, deriv_check_simopt, vector_check
from non_smooth.setDefaultParameters import set_default_parameters
from non_smooth.Problem import Problem
from non_smooth.trustregion import trustregion
from non_smooth.L1norm import L1TorchNorm
#Change a bit in __matmul__
class TorchVect:
    @torch.no_grad()
    def __init__(self, tensordict, isRop = False): #store tensor dictionary
        self.td    = tensordict
        s          = []
        self.isRop = isRop
        for _, v in self.td.items():
            s.append(v.size()[0])
        self.shape = s
    @torch.no_grad()
    def clone(self):
        td  = copy.deepcopy(self.td)
        ans = TorchVect(td)
        ans.zero()
        return ans

    @property
    def T(self):
        J = len(self.td.items())
        Temp = self.clone()
        for j, (k, v) in enumerate(list(self.td.items())):
              Temp.td[k] = v.T
        return Temp

    @torch.no_grad()
    def zero(self):
        for _, v in self.td.items():
            v.zero_()
    @torch.no_grad()
    def __add__(self, other):
        temp = other.clone()
        for k, v in self.td.items():
            temp.td[k] = other.td[k] + v
        return temp
    @torch.no_grad()
    def __sub__(self, other):
        temp = other.clone()
        for k, v, in self.td.items():
           temp.td[k] = other.td[k] - v
        return -1*temp
    @torch.no_grad()
    def __mul__(self, alpha):
      ans = self.clone()
      for k, v in self.td.items():
          ans.td[k].add_(v, alpha = alpha)
      return ans
    @torch.no_grad()
    def __rmul__(self, alpha):
        return self.__mul__(alpha)
    @torch.no_grad()
    def __matmul__(self, x):
        ans = x.clone()
        I = list(x.td.items())
        J = len(I)
        for index, (k, v) in enumerate(I):
            # print(index, k, self.td[k].size(), v.size())
            if index % 2 == 0 and not x.isRop:
                if index+1 < J:
                    nk = I[index+1][0]
                    ans.td[k] = self.td[nk] @ (v @ self.td[k].T)
                else:
                    ans.td[k] = v @self.td[k].T
              
              
              
            else:
              if index == 0 and x.isRop:
                ans.td[k] =  v @ self.td[k]
              else:
                ans.td[k] = self.td[k] @ v
        return ans

    @torch.no_grad()
    def __truediv__(self, alpha):
        ans = self.clone()
        for k, v in self.td.items():
            ans.td[k].add_(v, alpha = 1/alpha)
        return ans
    @torch.no_grad()
    def __rtruediv__(self, alpha):
        return self.__truediv__(alpha)

from non_smooth.L2Vectors import L2TVPrimal, L2TVDual
def restriction_R(m,n,x):
    matrix_R = x.clone()
    matrix_R.isRop = True
    I = [(k,v) for k,v in matrix_R.td.items() if k in x.td]
    #J = len(I)
    for j,(k,_) in enumerate(I):
        #qm = int(np.sqrt(m[j]))
        #qn = int(np.sqrt(n[j]))
        if m[j]==n[j]:
            R = torch.eye(m[j],n[j],dtype = torch.float64)
        else:
            R = torch.zeros(m[j],n[j],dtype = torch.float64)
            for i in range(m[j]):
                R[i,2*(i+1)-1] = 1/np.sqrt(2)
                R[i,2*i] = 1/np.sqrt(2)
                
        matrix_R.td[k] = R
    
    return matrix_R
    
#Recursive step
def Reye(x):
    if isinstance(x,np.ndarray):
        return np.eye(x.shape[0])
      
    matrix_R       = x.clone()
    matrix_R.isRop = True
    for k  in x.td.keys():
        param = x.td[k]
        
        if len(x.td[k].size()) == 1:
            n = x.td[k].size(0)
            matrix_R.td[k] = torch.eye(n, n, dtype=torch.float64)

        else:
            m = x.td[k].size(1) if len(x.td[k].size())>1 else x.td[k].size(0)
            if k in matrix_R.td:
                matrix_R.td[k] = torch.eye(m, m, dtype=torch.float64)
          

    return matrix_R


class FullyConnectedNN(nn.Module):
    def __init__(self, dims):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[2], dims[3])
        self.fc3 = nn.Linear(dims[4], dims[5],bias=False)
        self.activation = nn.Sigmoid() #changable

    def forward(self, x):
        #print(x.shape)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    def gradient(self,x):
        assert x.requires_grad
        #import pdb
        #pdb.set_trace()
            
        batch_size,num_points_2 = x.shape
        num_points = int(np.sqrt(num_points_2))
        for param in self.parameters():
            param.requires_grad_(True)
        x_grid = x.view(batch_size,num_points,num_points)
        grad_x = torch.zeros_like(x_grid)
        grad_y = torch.zeros_like(x_grid)
        #import pdb
        #pdb.set_trace()
        #print(x.shape)
        #y = self.forward(x.view(batch_size, -1))
        #y = y.view(batch_size, num_points)
        #grad = torch.autograd.grad(outputs=y,inputs=x,grad_outputs=torch.ones_like(y),create_graph=True)[0]
        for i in range(num_points):
            for j in range(num_points):
                # Forward pass for single point
                output = self.forward(x)
                
                # Compute gradient of output w.r.t input (x)
                grad_output = torch.zeros_like(output)
                grad_output[:, 0] = 1.0  # ∂NN/∂x_ij
                
                # Compute gradient
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=x,
                    grad_outputs=grad_output,
                    retain_graph=True,
                    create_graph=True
                )[0]
                
                # Extract ∂NN/∂x and ∂NN/∂y (approximated via finite differences)
                grad_x[:, i, j] = grad[:, i * num_points + j]  # ∂NN/∂x
                grad_y[:, i, j] = grad[:, i * num_points + j]  # ∂NN/∂y
        
        # Stack gradients into [batch, 961, 2]
        grad = torch.stack([grad_x, grad_y], dim=-1)
        #print(grad.view(batch_size, -1, 2).shape)#(1,961,2)
        return grad.view(batch_size, -1, 2)
    #def hessian(self,x):
    #    assert x.requires_grad
    #    y = self.forward(x)
    #    N, input_dim = x.shape
    #    hess = torch.zeros(N, input_dim, input_dim, dtype=x.dtype)
    #    grad = self.gradient(x)
    #    for i in range(input_dim):
    #        grad2 = torch.autograd.grad(
    #            outputs=grad[:, i],
    #            inputs=x,
    #            grad_outputs=torch.ones_like(grad[:, i]),
    #            retain_graph=True
    #        )[0]
    #        hess[:, i, :] = grad2
            
    #    #print("debug for hessian shape:",hess.shape) #([1,961,961])
    #    return hess
    def laplacian(self,x):
        assert x.requires_grad
        grad = self.gradient(x)
        lap = torch.zeros_like(x)
        #hess = self.hessian(x)
        
        for i in range(2):  # x and y directions
            # Compute ∂²NN/∂x_i²
            grad_i = grad[..., i].sum()  # Sum to scalar for autograd
            
            grad2 = torch.autograd.grad(
                outputs=grad_i,
                inputs=x,
                retain_graph=True,
                create_graph=True
            )[0]
            
            lap += grad2  # Accumulate ∂²NN/∂x² + ∂²NN/∂y²
        #print(lap.shape) #(1,961)
        return lap 

        
        

class NNObjective:
    def __init__(self, var):
        self.var = var
    def update(self, x, type):
        return None
    # Compute objective function value
    def value_torch(self, x, ftol):
        #mesh= self.var['mesh'].clone().detach().requires_grad_(True)
        if isinstance(x,  TorchVect):
            nn = torch.func.functional_call(self.var['NN'], x.td, (self.var['mesh'],))
        else:
          #x is the parameters of model
          nn = torch.func.functional_call(self.var['NN'], x, (self.var['mesh'],))
        f_domain = self.var['b']
        
        #pde = torch.matmul(self.var['A'], nn.T)
        
        #gradk_x1 = self.var['gradk'][:,0]
        
        #gradk_x2 = self.var['gradk'][:,1]
        gradk = self.var['gradk']#(1,961,2)
        
        #gradNN = self.var['gradNN']#(1,961,2)
        gradNN = self.var['NN'].gradient(self.var['inputs_x']).clone().detach()
        lapNN = self.var['NN'].laplacian(self.var['inputs_x']).clone().detach()
       
        kappa = self.var['k']#(1,961)
        #self.var['mesh'].requires_grad_(True)
        
        #lapNN = self.var['lapNN'] #(1,961)
        
        # Compute gradient of NN output w.r.t. mesh (for PDE residual)
        
        
        #PDE residual -\nabla\cdot(\kappa\nabla_x NN) = -\nabla\kappa\cdot\nabla_x NN-\kappa\laplacian NN
        pde_residual = -(gradk*gradNN).sum(-1)-kappa*lapNN-f_domain#(1,961)
        
        
        #pde_residual = pde[self.domain_mask] - f_domain
        #pde_residual = pde - f_domain.T
        #loss_pde = 0.5*torch.matmul(torch.matmul(pde_residual,self.var['M']), pde_residual.T)
        loss_pde = 0.5 * (pde_residual ** 2).sum()  

        val = loss_pde.squeeze() # reduce dims for some reason
        #loss = loss_boundary
        return val

    def value(self, x, ftol):
        val = self.value_torch(x, ftol)
        ferr = 0.
        return val, ferr
    def gradient(self, x, gtol):
        #if isinstance(x,  TorchVect):
        #  grad = self.torch_gradient(x.td, gtol)
        #else:
        #  grad = self.torch_gradient(x, gtol)
        grad = self.torch_gradient(x,gtol)
        gerr = 0
        if isinstance(grad, TorchVect) and hasattr(grad, 'td'):
            total_grad_norm = sum(torch.norm(g).item() 
                           for g in grad.td.values() if g is not None)
            #print(f"Total gradient norm: {total_grad_norm:.6f}")
        
            if total_grad_norm < 1e-8:
                print("WARNING: All gradients are effectively zero!")
        return grad, gerr
    def torch_gradient_wrong(self,x,gtol):
        if isinstance(x, TorchVect):
             params = collections.OrderedDict(x.td.items())
        else:
             params = collections.OrderedDict(x.items() if isinstance(x, dict) else [('param', x)])
             
        inputs_x = self.var['inputs_x'].detach().requires_grad_(True)
        
        torch.set_grad_enabled(True)
        with torch.enable_grad():
            original_params = {n:p.clone() for n,p in self.var['NN'].named_parameters()}
            
            try:
                with torch.no_grad():
                    for name, param in self.var['NN'].named_parameters():
                        if name in params:
                            param.copy_(params[name])
                nn_output = self.var['NN'](inputs_x)
                gradNN = self.var['NN'].gradient(inputs_x)
                lapNN = self.var['NN'].laplacian(inputs_x)
                residual = (-self.var['gradk'] * gradNN).sum(-1) - self.var['k'] * lapNN - self.var['b']
                loss = (residual**2).sum()
                grads = torch.autograd.grad(loss,[params[name] for name in self.var['NN'].state_dict() if name in params],retain_graph=True,create_graph=True,allow_unused=True)
            finally:
                with torch.no_grad():
                    for name,param in self.var['NN'].named_parameters():
                        if name in original_params:
                            param.copy_(original_params[name])
        grad_dict = collections.OrderedDict()
        grad_idx = 0
        model_params = self.var['NN'].state_dict()
        for name in model_params:
            if name in params:
                grad = grads[grad_idx] if grads[grad_idx] is not None else torch.zeros_like(params[name])
                grad_dict[name] = grad
                grad_idx +=1
           
        return TorchVect(grad_dict) if isinstance(x,TorchVect) else grad_dict
                
                
                    
            
    

    def torch_gradient(self, x, gtol):
        # 1. Parameter extraction and validation
        if isinstance(x, TorchVect):
             params = collections.OrderedDict(x.td.items())
        else:
             params = collections.OrderedDict(x.items() if isinstance(x, dict) else [('param', x)])
    
        #print("\n=== Gradient Computation Debug ===")
        #print("Parameter structure:")
        #for name, p in params.items():
        #    print(f"{name}: shape {p.shape} | dtype {p.dtype} | requires_grad {p.requires_grad}")

        # 2. Input preparation
        inputs_x = self.var['inputs_x'].detach().requires_grad_(True)
        #print(f"\nInputs_x - shape: {inputs_x.shape} | dtype: {inputs_x.dtype}")
        torch.set_grad_enabled(True)

        # 3. Forward pass with full computation graph
        with torch.enable_grad():
            # Verify parameter usage
            #print("\nRunning forward pass...")
            try:
                # Direct computation without functional_call for debugging
                original_params = dict(self.var['NN'].named_parameters())
            
                # Temporarily replace model parameters
                with torch.no_grad():
                    for name, param in self.var['NN'].named_parameters():
                        if name in params:
                            param.copy_(params[name])
            
                # Forward pass
                nn_output = self.var['NN'](inputs_x)
                #print(f"NN output - mean: {nn_output.mean().item():.4f} | dtype: {nn_output.dtype}")

                # Spatial derivatives
                gradNN = self.var['NN'].gradient(inputs_x)
                lapNN = self.var['NN'].laplacian(inputs_x)
                #print(f"gradNN - mean: {gradNN.mean().item():.4f} | dtype: {gradNN.dtype}")
                #print(f"lapNN - mean: {lapNN.mean().item():.4f} | dtype: {lapNN.dtype}")

                # PDE residual
                residual = (-self.var['gradk'] * gradNN).sum(-1) - self.var['k'] * lapNN - self.var['b']
                loss = (residual**2).sum()
                #print(f"Loss value: {loss.item():.8f}")

            except Exception as e:
                print(f"Forward pass failed: {str(e)}")
                raise

        # 4. Gradient computation with verification
        #print("\nComputing gradients...")
        try:
            # Compute gradients directly for debugging
            grads = []
            for name, param in self.var['NN'].named_parameters():
                if name in params:
                    grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
                    #print(f"{name} gradient norm: {torch.norm(grad).item():.6f}")
                    grads.append(grad)
                else:
                    grads.append(None)
        
            # Package results
            grad_dict = collections.OrderedDict(
                (name, grad if grad is not None else torch.zeros_like(p))
                for (name, p), grad in zip(params.items(), grads)
                if name in params
            )
        
            return TorchVect(grad_dict) if isinstance(x, TorchVect) else grad_dict

        except RuntimeError as e:
            print(f"\nGradient computation failed: {str(e)}")
            print("\nCritical verification:")
        
        # Finite difference check
            eps = 1e-4
            print("\nRunning finite difference verification:")
            with torch.no_grad():
                base_loss = self.value_torch(x, gtol)
                for name, param in params.items():
                    original = param.clone()
                    param.add_(eps)
                    new_loss = self.value_torch(x, gtol)
                    fd_grad = (new_loss - base_loss)/eps
                    param.copy_(original)
                    #print(f"{name} FD grad: {fd_grad.item():.6f}")
        
            raise RuntimeError("Gradient computation failed - see finite difference results above")
            
            
    
        # Return based on input type
        #if isinstance(x, TorchVect):
        #    return TorchVect(grad_dict)
        
        #return grad_dict

    #def hessVec(self, v, x, htol):

    #    grad,_ =self.gradient(x,htol)
    #    if isinstance(x,TorchVect):
    #        v_td = v.td if isinstance(v,TorchVect) else v
    #        x_td = x.td
    #    else:
    #        v_td = v
    #        x_td = x
    #    x_td = {k: v.detach().requires_grad_(True) for k,v in x_td.items()}
    #    def flatten_params(params):
    #        return torch.cat([p.reshape(-1) for p in params.values()])
            
    #        if isinstance(params,(dict,collections.OrderedDict)):
    #            return torch.cat([p.reshape(-1) for p in params.values()])
    #        return params.reshape(-1)
    #    flat_grad = flatten_params(grad.td if isinstance(grad,TorchVect) else grad)
    #    flat_v = flatten_params(v_td)
        
    #    with torch.enable_grad():
    #        loss = self.value_torch(TorchVect(x_td) if isinstance(x,TorchVect) else x_td,htol)
    #        grads = torch.autograd.grad(loss,x_td.values(),create_graph=True)
    #        flat_grad = torch.cat([g.reshape(-1) for g in grads])
    #        gv = torch.dot(flat_grad,flat_v)
    #        gv = torch.dot(flat_grad,flat_v)
    #        Hv = torch.autograd.grad(gv,x_td.values(),retain_graph=True,create_graph=True,allow_unused=True)
    #        Hv_dict = collections.OrderedDict((k,v if v is not None else torch.zeros_like(x_td[k])) for k,v in zip(x_td.keys(),Hv))
            
    #    return TorchVect(Hv_dict) if isinstance(x,TorchVect) else Hv_dict, 0 
            
    #    if isinstance(x_td,(dict,collections.OrderedDict)):
    #        Hv_dict = collections.OrderedDict(zip(x_td.keys(),Hv))
    #        Hv_dict = collections.OrderedDict((k,v if v is not None else torch.zeros_like(x_td[k]))
    #                                          for k,v in Hv_dict.items())
    #        return TorchVect(Hv_dict),0
    #    else:
    #        return TorchVect(Hv[0] if Hv[0] is not None else torch.zeros_like(x_td)),0
    
    
    def hessVec(self, v, x, htol):
        params = x.td if isinstance(x,TorchVect) else x
        v_params = v.td if isinstance(v,TorchVect) else v
        if not isinstance(params,collections.OrderedDict):
            params =  collections.OrderedDict(params)
        if not isinstance(v_params,collections.OrderedDict):
            v_params = collections.OrderedDict(v_params)
        inputs_x = self.var['inputs_x'].detach().requires_grad_(True)
        with torch.no_grad():
            for name, p in self.var['NN'].named_parameters():
                if name in params:
                    p.copy_(params[name])
        torch.set_grad_enabled(True)
        from torch.nn.utils.stateless import functional_call
        nn_output = functional_call(self.var['NN'], params, (inputs_x,))
        gradNN = self.var['NN'].gradient(inputs_x)
        lapNN = self.var['NN'].laplacian(inputs_x)
        residual = (-self.var['gradk'] * gradNN).sum(-1) - self.var['k'] * lapNN - self.var['b']
        loss = (residual**2).sum()
        grads =  torch.autograd.grad(loss, [p for _, p in self.var['NN'].named_parameters()], create_graph=True)

    #    grad,_ =self.gradient(x,htol)
    
        #if isinstance(x,TorchVect):
        #    params = x.td
        #    v_params = v.td
        #else:
        #    params = x
        #    v_params = v
        #params = collections.OrderedDict(params)
        #v_params = collections.OrderedDict(v_params)
        
    #        v_td = v.td if isinstance(v,TorchVect) else v
    #        x_td = x.td
    #    else:
    #        v_td = v
    #        x_td = x
    #    x_td = {k: v.detach().requires_grad_(True) for k,v in x_td.items()}
        #def flatten(p):
        #    return torch.cat([t.reshape(-1) for t in p.values()])
        #for k in params:
        #    params[k] = params[k].detach().clone().requires_grad_(True)
        #loss = self.value_torch(TorchVect(params) if isinstance(x,TorchVect) else params,htol)
        #grads = torch.autograd.grad(loss, params.values(), create_graph=True)

        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        flat_v = torch.cat([v_params[k].reshape(-1) for k in params.keys()])
        gv = torch.dot(flat_grad, flat_v)
        Hv_tensors = torch.autograd.grad(gv, [p for _, p in self.var['NN'].named_parameters()], retain_graph=False,create_graph=True)
        Hv = collections.OrderedDict((k, hv if hv is not None else torch.zeros_like(params[k]))for k, hv in zip(params.keys(), Hv_tensors))
        return TorchVect(Hv) if isinstance(x,TorchVect) else Hv,0
      class NNSetup:
    """
    Solve the NN control problem
    """
    def __init__(self, NN_dim, n, alpha, beta, n_samples = 1):
        self.n      = n
        self.NN_dim = NN_dim
        self.nsamps = n_samples
        # Mesh grid for x, y ((meshsize+1) x (meshsize+1))
        self.domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n,n)

        self.V      = dolfinx.fem.functionspace(self.domain, ("Lagrange", 1))
        # Source term
        self.f      = dolfinx.fem.Constant(self.domain, dolfinx.default_scalar_type(1))
        self.inputs = torch.tensor(self.domain.geometry.x[:,0:1], dtype=torch.float64).T

        # Boundary condition
        self.u_D = dolfinx.fem.Function(self.V)
        self.u_D.interpolate(lambda x: np.zeros(x[0].shape))#1 + x[0]**2 + 2 * x[1]**2)
        tdim            = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim - 1, tdim)
        boundary_facets = dolfinx.mesh.exterior_facet_indices(self.domain.topology)
        boundary_dofs   = dolfinx.fem.locate_dofs_topological(self.V, self.domain.topology.dim - 1, boundary_facets)
        self.bc         = dolfinx.fem.dirichletbc(self.u_D, boundary_dofs)

        self.beta  = beta
        self.alpha = alpha
        self.R     = []



        self.NN    = FullyConnectedNN(NN_dim)
        self.NN.to(torch.float64)
        self.inputs_x = self.inputs.clone().detach().requires_grad_(True)
        self.grad_NN = self.NN.gradient(self.inputs_x).clone().detach()
        
        #self.hess_NN = self.NN.hessian(self.inputs_x).clone().detach()
        self.lap_NN = self.NN.laplacian(self.inputs_x).clone().detach()
        
        

        t = self.generate_fenics_data()
        self.kappa_value_list  = t[0]
        self.f_value_list      = t[1]
        # self.kappa_grad_list   = t[2]
        self.u_solution_tensor = t[2]
        self.grad_kappa_value_list = t[3]

    def generate_fenics_data(self):
        # Generate random values for kx, ky, ax, ay, and alpha
        kx_samples    = np.random.uniform(0.5, 4.0, self.nsamps)
        ky_samples    = np.random.uniform(0.5, 4.0, self.nsamps)
        ax_samples    = np.random.uniform(0.0, 0.5, self.nsamps)
        ay_samples    = np.random.uniform(0.0, 0.5, self.nsamps)
        alpha_samples = np.random.uniform(0.0, np.pi / 2, self.nsamps)


        # Prepare containers for inputs and outputs
        kappa_value_list = []
        f_value_list = []
        u_solution_list = []
        grad_kappa_value_list = []


        for i in range(self.nsamps):
            kx = kx_samples[i]
            ky = ky_samples[i]
            ax = ax_samples[i]
            ay = ay_samples[i]
            alpha = alpha_samples[i]

            # Define rotated coordinates x' and y'
            x     = ufl.SpatialCoordinate(self.domain)
            K     = dolfinx.fem.Function(self.V)
            K.interpolate(lambda x: 1.1* np.ones(x[0].shape)) #+ np.cos(kx*np.pi*(np.cos(alpha)*(x[0]-0.5) - np.sin(alpha)*(x[1]-0.5) + 0.5 + ax)) * np.cos(ky*np.pi*(np.sin(alpha)*(x[0]-0.5) + np.cos(alpha)*(x[1]-0.5) + 0.5 + ay)))
            grad_K_expr = ufl.grad(K)
            
            # Create vector function space
            
            from dolfinx import fem
            V_vec = fem.functionspace(self.domain, ("CG", 1, (self.domain.topology.dim,)))

            # Interpolate using automatically generated lambda
            grad_K_func = dolfinx.fem.Function(V_vec)
            grad_K_func.interpolate(dolfinx.fem.Expression(grad_K_expr, V_vec.element.interpolation_points()))
            
            #reshape
            num_scalar_dofs = self.V.dofmap.index_map.size_local
            gdim = self.domain.topology.dim
            grad_values = grad_K_func.x.array.reshape((num_scalar_dofs, gdim))

            # Store gradient values in a list
            grad_kappa_value_list.append(grad_values)

            V = self.V
            # Define variational problem
            v = ufl.TestFunction(V)
            uh = dolfinx.fem.Function(V)
            u  = ufl.TrialFunction(V)
            F = ufl.dot(K*ufl.grad(uh), ufl.grad(v)) * ufl.dx  - ufl.inner(self.f, v) * ufl.dx
            p = dolfinx.fem.petsc.NonlinearProblem(F, uh, bcs = [self.bc])
            solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, p)
            solver.solve(uh)

            #get operators
            a = ufl.dot(K*ufl.grad(u), ufl.grad(v)) * ufl.dx
            l = self.f * v * ufl.dx
            cost =  1 / 2 * (uh ) * (uh) * ufl.dx #+ alpha / 2 * f**2 * ufl.dx
            ac = dolfinx.fem.form(a)
            H = dolfinx.fem.form(ufl.derivative(ufl.derivative(cost, uh), uh))
            A = dolfinx.fem.create_matrix(ac)
            M = dolfinx.fem.create_matrix(H)
            b = dolfinx.fem.Function(uh.function_space)
            L = dolfinx.fem.form(l)
            # Solve the problem
            # Get the solution and input features
            dolfinx.fem.assemble_matrix(A, ac, bcs = [self.bc])
            dolfinx.fem.assemble_matrix(M, H)
            AT = torch.from_numpy(A.to_dense()) #stiffness
            MT = torch.from_numpy(M.to_dense()) #mass
            dolfinx.fem.assemble_vector(b.x.array, L) #rhs
            dolfinx.fem.apply_lifting(b.x.array, [ac], [[self.bc]])
            b.x.scatter_reverse(dolfinx.la.InsertMode.add)
            [bc.set(b.x.array) for bc in [self.bc]]

            ut = dolfinx.fem.Function(V)
            ut.x.array[:] = np.linalg.solve(AT, b.x.array)
            self.A            = AT.to(dtype=torch.float64)
            self.M            = MT.to(dtype=torch.float64)
            u_array           = uh.x.array[:]

            kappa_values      =  K.x.array[:]
            f_values     = b.x.array[:] #self.f.x.array[:]
            # print(np.dot(A, u_array) - f_values)

            kappa_domain      = kappa_values #[self.domain_mask]
            f_domain          = f_values #[self.domain_mask]

            kappa_value_list.append(kappa_domain)
            f_value_list.append(f_domain)
            u_solution_list.append(u_array)


        kappa_value_list  = torch.tensor(np.array(kappa_value_list), dtype=torch.float64)
        f_value_list      = torch.tensor(np.array(f_value_list), dtype=torch.float64)
        u_solution_tensor = torch.tensor(np.array(u_solution_list),dtype=torch.float64)
        grad_kappa_value_list = torch.tensor(np.array(grad_kappa_value_list), dtype=torch.float64)
        #print("debugging:",grad_kappa_value_list.shape)
        #print("debug for kappa shape:",kappa_value_list.shape)





        return  kappa_value_list, f_value_list, u_solution_tensor, grad_kappa_value_list
    def returnVars(self, useEuclidean):
      var = {'beta':self.beta,
             'n':self.n,
             'nu':self.n,
             'nz':sum(p.numel() for p in self.NN.parameters()),
             'alpha':self.alpha,
             'A':self.A,
             'M':self.M,
             'NN':self.NN,
             'k':self.kappa_value_list,
             'b':self.f_value_list,
             'ud':self.u_solution_tensor,
             'useEuclidean':useEuclidean,
             'mesh':torch.tensor(self.domain.geometry.x[:,0:1], dtype=torch.float64).T,
             'gradk': self.grad_kappa_value_list,
             'gradNN': self.grad_NN,
             #'hessNN': self.hess_NN,
             'lapNN': self.lap_NN,
             'inputs_x': self.inputs_x
            }
      return var

    def plot_solutions(self, pinns_solution=None):

        # Extract mesh coordinates
        geometry = self.domain.geometry.x[:, :2]
        #X = geometry[:, 0]
        #Y = geometry[:, 1]


        # Real solution
        real_solution = self.u_solution_tensor[0].numpy()  # Shape: (1024,)

        # Predict the PINNs solution if not provided
        if pinns_solution is None:
            with torch.no_grad():
                pinns_solution = self.NN(self.inputs).numpy()  # Shape: (1, 1024)
            pinns_solution = pinns_solution.squeeze(0)  # Remove the extra dimension -> (1024,)


        # Plot the true solution
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("True Solution")
        plt.tricontourf(geometry[:, 0], geometry[:, 1], real_solution, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

        # Plot the predicted solution
        plt.subplot(1, 3, 2)
        plt.title("Predicted Solution")
        plt.tricontourf(geometry[:, 0], geometry[:, 1], pinns_solution, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

        # Plot the error (absolute difference)
        error = np.abs(real_solution - pinns_solution)
        plt.subplot(1, 3, 3)
        plt.title("Error (Absolute Difference)")
        plt.tricontourf(geometry[:, 0], geometry[:, 1], error, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

        plt.tight_layout()
        plt.show()


def driver(savestats, name):
    print("driver started")
    np.random.seed(0)

    # Set up optimization problem
    n          = [30, 30] #[30, 15]# Number of cells
    NN_dim     = np.array([(n[0]+1)**2, 100, 100, 100, 100, (n[0]+1)**2]) # Neural network nodes
    meshlist = [NN_dim, np.array([(n[1]+1)**2, 100, 100, 100, 100, (n[1]+1)**2])]
    #meshlist   = [NN_dim, np.array([(n[1]+1)**2, 50, 50, 50, 50, (n[1]+1)**2])]
    # meshlist   = [NN_dim, np.array([(n[0]+1)**2, 49, 49, 25, 25, (n[0]+1)**2])]


    alpha      = 1  # L2 penalty parameter
    beta       = 0  # L1 penalty parameter
    derivCheck = True


    problems = [] #problem list goes from fine to coarse
    x0       = []
    for i in range(0, len(meshlist)):
        nnset = NNSetup(meshlist[i], n[i], alpha, beta, n_samples=1)
        x     = TorchVect(nnset.NN.state_dict())
        if i == 0:
           x0 = x
        var   = nnset.returnVars(False)
        if i < len(meshlist) - 1:
          R = restriction_R(meshlist[i+1], meshlist[i], x) #puts R in preceeding problem
        else:
          R = Reye(x)
        #print(f"Tensor shapes before matmul:")
        #print(f"R: {R.shape}, x: {x.shape}")
        
        p = Problem(var, R)
        p.obj_smooth    = NNObjective(var)
        p.obj_nonsmooth = L1TorchNorm(var)
        p.pvector = L2TVPrimal(var)
        p.dvector = L2TVDual(var)
        problems.append(p)

        if derivCheck:
            for i in range(0, len(meshlist)):
                x =  TorchVect(nnset.NN.state_dict())
                d = x.clone() #just to get something to multiply against
                for key, vals in d.td.items():
                    d.td[key] = vals.copy_(torch.randn(vals.size()))
                    deriv_check(x, d, problems[i], 1e-4 * np.sqrt(np.finfo(float).eps))
                    vector_check(x, d, problems[i])

    cnt = {}
    # Update default parameters
    params = set_default_parameters("SPG2")
    params["reltol"] = False
    params["t"] = 2 / alpha
    params["ocScale"] = 1 / alpha
    params['maxit'] = 300

    #Plot loss
    #loss_values = []
    



    # Solve optimization problem
    start_time   = time.time()
    x, cnt_tr    = trustregion(0, x0, params['delta'],problems, params)
    loss_history = cnt_tr['objhist']
    #loss_history_single = cnt_single['objhist']
    iterations = range(1, len(loss_history) + 1)
    #iterations_single = range(1, len(loss_history_single) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, loss_history, marker="o", color="b")
    #plt.plot(iterations_single, loss_history_single, marker="o", color="r")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Objective Function)")
    plt.title("Loss vs Iterations in Trust-Region Optimization")
    plt.grid(True)
    plt.show()



    elapsed_time = time.time() - start_time

    print(f"Optimization completed in {elapsed_time:.2f} seconds")

    pro_tr =  []


    cnt[1] = (cnt_tr, pro_tr)


    print("\nSummary")
    print(
        "           niter     nobjs     ngrad     nhess     nobjn     nprox     ")


    print(
        f"   SGP2:  {cnt[1][0]['iter']:6d}    {cnt[1][0]['nobj1']:6d}    {cnt[1][0]['ngrad']:6d}    {cnt[1][0]['nhess']:6d}    "
        f"{cnt[1][0]['nobj2']:6d}    {cnt[1][0]['nprox']:6d}     "
    )

    ## Print updated weights of the first layer (fc1)
    #nnset.NN.load_state_dict(x.td)  # Load the updated parameters into the neural network
    #nnset.NN.eval()
    #state_dict_after  = nnset.NN.state_dict()
    #weights_fc1_after = state_dict_after['fc1.weight']
    #print("Weights of fc1 after optimization:", torch.nonzero(weights_fc1_after))

    final_nnset = NNSetup(meshlist[0], n[0], alpha, beta, n_samples=1)
    final_nnset.NN.load_state_dict(x.td)
    final_nnset.NN.eval()





    print("Updated neural network is stored in `updated_nn`.")
    final_nnset.plot_solutions()


    return cnt


cnt = driver(False, "test_run")
