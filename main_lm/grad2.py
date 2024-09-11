#from ROL import Objective

import torch



#Test for objective loss function
from Loss_function import compute_loss
from neural_network_construction import FullyConnectedNN
from PDE_POISSON import PDE
import numpy as np
import torch.nn as nn


class TorchObjective():
    # https://pytorch.org/docs/stable/func.html

    # @staticmethod
    # def _copy(source, target):
    #     target.zero()
    #     target.plus(source)

    def __init__(self):
        super().__init__()
        self.torch_gradient = torch.func.grad(self.torch_value)

    def value(self, x, tol):
        return self.torch_value(x).item()

    def torch_value(self, x):
        # Returns a scalar torch Tensor
        raise NotImplementedError

    def gradient(self, g, x, tol): #should put gradient of model into g
        ans = self.torch_gradient(x)
        g   = ans.copy()

    def _forward_over_reverse(self, input, x, v):
        # https://github.com/google/jax/blob/main/docs/notebooks/autodiff_cookbook.ipynb
        return torch.func.jvp(input, (x,), (v,))

    def hessVec(self, hv, v, x, tol):
        _, ans = self._forward_over_reverse(self.torch_gradient, x, v)
        hv.copy_(ans)


class TrainingObjective(TorchObjective):

    def __init__(self, model, data, loss):
        super().__init__()
        self.model = model
        self.x, self.y = data
        self.loss = loss

    def torch_value(self, x):
        return self.loss(torch.func.functional_call(self.model, x, self.x), self.y)



def real_solution_1d(x):
    return torch.sin(x)




def initialize_parameters(model, input_dim,r,init_type = 'he'):
    np.random.seed(1234)
    parameters = model.state_dict().copy()
    if init_type == 'he':   
      for v in parameters:
        parameters[v] = torch.rand(parameters[v].shape)

    elif init_type == 'xvaier':
        w1 = np.random.randn(r, input_dim) * np.sqrt(1. / input_dim)
        b1 = np.zeros((r, 1))
        w2 = np.random.randn(1, r) * np.sqrt(1. / r)
        b2 = np.zeros((1, 1))
    elif init_type == 'random':
       # Random initialization
       w1 = np.random.randn(r, input_dim) * 0.01
       b1 = np.zeros((r, 1))
       w2 = np.random.randn(1, r) * 0.01
       b2 = np.zeros((1, 1))
    else:
       raise ValueError("Unknown initialization type. Use 'he', 'xavier', or 'random'.")


    return parameters





class Objective_nn_pde(TrainingObjective):
    
    def __init__(self, model, data, pde):
        super().__init__(model, data, self.loss)
        self.model      = model
        self.x, self.y  = data
        self.pde        = pde
   
    def value(self, x, tol):
        return self.torch_value(x)

    def torch_value(self, x):
        return self.loss(x)
    
    def loss(self, params_nn, regularization=False, lambdap = 0.1):
        # Compute model prediction (torch tensor)
        #model_output   = model(model_input)
        model_output   = torch.func.functional_call(self.model, params_nn, self.x)
        # Compute main cost (torch tensor)
        main_cost      = (self.y - model_output)[1:-1]
        main_cost_loss = 0.5 * torch.norm(main_cost)**2 / self.x.shape[0] 
        
        # Compute regularization term if needed (torch tensor)
        if regularization==True:
            real_0   = real_solution_1d(self.x[0])
            real_end = real_solution_1d(self.x[-1])
            nn_0     = self.model(self.x[0].reshape(1, -1))
            nn_end   = self.model(self.x[-1].reshape(1, -1))
            
            regularization_term = (torch.norm(real_0 - nn_0)**2 + torch.norm(real_end - nn_end)**2)
            regularization_term = lambdap * 0.5 * regularization_term
        else:
            regularization_term = torch.tensor(0.0)
        
        # Total loss (torch tensor)
        total_loss = main_cost_loss + regularization_term
        
        return total_loss


def check_first_deriv(obj, x, g, d):

    tolerances          = np.array([1., 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10])
    
    obj.gradient(g, x, 0.) #compute gradient 
    DirDiff = 0.   #compute dot prod (probably standardize this?)
    for v in g: 
      DirDiff += torch.sum(torch.mul(g[v], d[v])) # need the same keys

    fx = obj.value(x, 0.) 


    print(' t    g\'*d   (f(x+td) - f(x))/t   err\n')
    
    for t in tolerances:
     
      with torch.no_grad():
        newx = x.copy()
        for v in newx:  #again probably standardize +/- for ordered dicts
          newx[v] = t*d[v] + x[v]

      fxt    = obj.value(newx, 0.)
      fd_est = (fxt - fx)/t

      err    = DirDiff - fd_est
      print(t, ' ', DirDiff.item(), '  ', fd_est.item(), '   ', torch.norm(err).item())

#Test for function compute_first_derivative
#print(compute_first_derivative(My_model, input_data))
#it works
def compute_second_derivative(model, input_data):
    input_data.requires_grad = True


    # First derivative: Compute the gradient of the output with respect to the input
    first_derivative = compute_first_derivative(model, input_data)

    if first_derivative is None:
        return ValueError("First derivative is None. Check if the input tensor is used in the graph.")


    # Second derivative: Compute the gradient of the first derivative with respect to the input
    second_derivative = torch.zeros_like(input_data)
    for i in range(input_data.size(0)):  # Iterate over input features
        grad2 = torch.autograd.grad(first_derivative[i], input_data, retain_graph=True, allow_unused=True)[0]
        if grad2 is not None:
            second_derivative[i] = grad2[i]

    return second_derivative
    
def main(): 
   input_dim           = 1
   output_dim          = 1
   n_hidden_layers     = 1
   r_nodes_per_layer   = 5
   activation_function = torch.sigmoid
   sample_num          = 3 #could change
   x                   = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
   model               = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)
  
   #Compute source term for the real solution (torch tensor)
   pde                 = PDE((0,1),real_solution_1d) 
   y                   = pde.compute_source_term(x, real_solution_1d).reshape(-1, 1)
   #y                   = real_solution_1d(x)  
   
   data                = (x,y) 
   obj_nn_pde          = Objective_nn_pde(model, data, pde)
   param_test          = initialize_parameters(model, 1, 5)
   dvec                = param_test.copy()
   
   with torch.no_grad():
     for v in dvec:
       dvec[v] = torch.rand(dvec[v].shape)

   value              = obj_nn_pde.value(param_test, 0.)
   #Calculate gradient
   with torch.no_grad():
     g_nn_pde = dvec.copy() 
   check_first_deriv(obj_nn_pde, param_test, g_nn_pde, dvec)


#g = torch.zeros_like(x)  # Initialize g with the same shape as x
#obj.gradient(g, x, tolerance)
#print(f"Gradient: {g}")  
#print(loss_p_tensor(param_test_flatten))
#Calculte the gradient and second derivative of NN w.r.t. input x

if __name__ == "__main__": 
  main()
