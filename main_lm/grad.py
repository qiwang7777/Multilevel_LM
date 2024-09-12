import torch
import numpy as np
import torch.nn as nn
from neural_network_construction import FullyConnectedNN
from PoissonPDE import PoissonPDE
import numpy as np
import torch.nn as nn
# Assuming FullyConnectedNN and PoissonPDE are already defined

import torch
import numpy as np
import torch.nn as nn

class TorchObjective():
    def __init__(self):
        super().__init__()

    def value(self, x, tol):
        return self.torch_value(x).item()

    def torch_value(self, x):
        # Returns a scalar torch Tensor
        raise NotImplementedError

    def gradient(self, g, x, tol):
        # Compute gradients
        scalar_value = self.torch_value(x)
        grads = torch.autograd.grad(scalar_value, x.values(), retain_graph=True, create_graph=True,allow_unused=True)
        for param, grad in zip(x.values(), grads):
            g[param] = grad

    def _forward_over_reverse(self, input, x, v):
        return torch.autograd.grad(input, x, grad_outputs=v, retain_graph=True)

    def hessVec(self, hv, v, x, tol):
        _, ans = self._forward_over_reverse(self.torch_gradient, x, v)
        for param, hessian in zip(x.values(), ans):
            hv[param] = hessian

class TrainingObjective(TorchObjective):
    def __init__(self, model, data, loss):
        super().__init__()
        self.model = model
        self.x, self.y = data
        self.loss = loss

    def torch_value(self, x):
        self.model.load_state_dict(x)
        temp = self.model(self.x)
        val = self.loss(temp, self.y)
        return val

def real_solution_1d(x):
    return torch.sin(x)

def get_1d_boundary(x):
    return torch.tensor([x[0].item(), x[-1].item()], dtype=torch.float32)

def get_2d_boundary(x):
    x_min = x[:,0].min().item()
    x_max = x[:,0].max().item()
    y_min = x[:,1].min().item()
    y_max = x[:,1].max().item()
    boundary_points = x[(x[:,0] == x_min) | (x[:,0] == x_max) |
                         (x[:,1] == y_min) |(x[:,1] == y_max)]
    return boundary_points


def loss_solving_poisson(real_solution,model,x,regularization=True,lambdap = 0.1):
    input_dim = model.hidden_layers[0].in_features
    
    if input_dim == 1:
        pde_1d = PoissonPDE(real_solution,x)#real_solution = real_solution_1d
        real_source = pde_1d._compute_1d_source_term(x)
        nn_pde = PoissonPDE(model,x)#real_solution = neural network
        nn_source = nn_pde._compute_1d_source_term(x)
        main_cost = (real_source-nn_source)[1:-1]
        sample_num = x.shape[0]-2
        main_loss = 0.5*torch.norm(main_cost)**2 / sample_num
        
    if input_dim == 2:
        
        pde_2d = PoissonPDE(real_solution,x)
        real_source = pde_2d._compute_2d_source_term(x)
        nn_pde = PoissonPDE(model,x)
        nn_source = nn_pde._compute_2d_source_term(x)
        main_cost = (real_source-nn_source)
        sample_num = (np.sqrt(x.shape[0])-1)**2 
        main_loss = 0.5*torch.norm(main_cost)**2 / sample_num
        
    #Compute the regularization term if needed
    if regularization == True:
        if input_dim == 1:
            x_boundary = get_1d_boundary(x).reshape(-1,1)
            real = real_solution(x_boundary)
            nn = model(x_boundary)
            re_term = real-nn
            #re_tensor = re_term.clone().detach().requires_grad_(True)
            boundary_num = x_boundary.size(0)
            re_loss = 0.5*lambdap*torch.norm(re_term)**2 / boundary_num
        if input_dim == 2:
            x_boundary = get_2d_boundary(x)
            real = real_solution(x_boundary).reshape(-1,1)
            nn = model(x_boundary)
            re_term = real-nn
            boundary_num = x_boundary.size(0)
            re_loss = 0.5*lambdap*torch.norm(re_term)**2 / boundary_num
    else:
        re_loss = 0
    return main_loss+re_loss

class Objective_nn_pde(TrainingObjective):
    def __init__(self, model, data, real_solution):
        self.model = model
        self.x, self.y = data
        self.real_solution = real_solution
        self.loss_fn = lambda nn_y: torch.nn.MSELoss(reduction='mean')(nn_y, self.y)
        super().__init__(model, data, None)

    def torch_value(self, params_dict):
        try:
            self.model.load_state_dict(params_dict)
        except Exception as e:
            print(f"Error loading state dict: {e}")
            raise
        
        return loss_solving_poisson(self.real_solution,self.model,self.x)

def initialize_parameters(model, init_type='he'):
    np.random.seed(1234)
    parameters = model.state_dict()
    new_parameters = {}

    for name, param in parameters.items():
        if 'weight' in name:
            if init_type == 'he':
                stddev = np.sqrt(2. / param.size(1))
                new_parameters[name] = torch.randn_like(param) * stddev
            elif init_type == 'xavier':
                stddev = np.sqrt(2. / (param.size(0) + param.size(1)))
                new_parameters[name] = torch.randn_like(param) * stddev
            elif init_type == 'random':
                new_parameters[name] = torch.randn_like(param) * 0.01
            else:
                raise ValueError("Unknown initialization type. Use 'he', 'xavier', or 'random'.")
        elif 'bias' in name:
            new_parameters[name] = torch.zeros_like(param)

        # Ensure parameters have requires_grad=True
        new_parameters[name].requires_grad_()

    return new_parameters


def check_first_deriv(obj, x, g, d):
    tolerances = np.array([1., 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10])
    
    obj.gradient(g, x, 0.)# Compute gradient
    print(g)#get the gradient?
    DirDiff = 0.  # Initialize dot product
    for param in g: 
        if param in d:
            DirDiff += torch.sum(torch.mul(g[param], d[param]))  # Compute dot product

    fx = obj.value(x, 0.) 

    print(' t         g\'*d          (f(x+td) - f(x))/t        err     ||t*d||\n')
    
    for t in tolerances:
        newx = {k: v + t * d[k] for k, v in x.items()}
        temp = 0.
        for param in newx:
            temp += torch.norm(newx[param] - x[param])**2

        fxt = obj.value(newx, 0.)
        fd_est = (fxt - fx) / t
        err = DirDiff - fd_est

        print('{:.3e}  {:.3e}  {:.3e}  {:.3e} {:.3e} '.format(t, DirDiff.item(), fd_est.item(), torch.norm(err).item(), temp.item()))

def compute_loss_gradients(real_solution,model,x,regularization=True,lambdap = 0.1):
    #Forward pass: Perform forward pass to compute network output
    x.requires_grad_(True)
    output = model(x)
   
   
    #Compute the function
    loss = loss_solving_poisson(real_solution, model, x, regularization=True,lambdap = 0.1)
    loss.backward()
    #Create a vector to store the flattened gradients
    gradients = []

    for param in model.parameters():
        gradients.append(param.grad.view(-1))
        
    flattened_gradients = torch.cat(gradients)
    
    return flattened_gradients

def main():
    input_dim = 1
    output_dim = 1
    n_hidden_layers = 1
    r_nodes_per_layer = 2
    activation_function = torch.sigmoid
    sample_num = 3
    x = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num, input_dim), dtype=torch.float32, requires_grad=True)
    model = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)
    for param in model.parameters():
        param.requires_grad = True
    
    real_solution = real_solution_1d

    pde = PoissonPDE(real_solution_1d, x)
    y = pde._compute_1d_source_term(x).reshape(-1, 1)

    data = (x, y)
    
    obj_nn_pde = Objective_nn_pde(model, data, real_solution)
    #param_test = model.parameters()
    param_test = initialize_parameters(model)
    model.load_state_dict(param_test)
    for name, param in model.state_dict().items():
        print(f"{name}: {param}")

    print(compute_loss_gradients(real_solution, model, x))
    dvec                = param_test.copy()
    g_nn_pde            = param_test.copy() 
    with torch.no_grad():
        for v in dvec:
            dvec[v] = r_nodes_per_layer * torch.rand(dvec[v].shape)
            g_nn_pde[v] = torch.rand(dvec[v].shape)

    # Calculate the objective value
    value = obj_nn_pde.torch_value(param_test)

    # Calculate gradient
    check_first_deriv(obj_nn_pde, param_test, g_nn_pde, dvec)
    
    

    #dvec = {k: torch.randn_like(v) * 5. for k, v in param_test.items()}
    #g_nn_pde = {k: torch.randn_like(v) for k, v in param_test.items()}

    #value = obj_nn_pde.value(param_test, 0.)

    #check_first_deriv(obj_nn_pde, param_test, g_nn_pde, dvec)

if __name__ == "__main__":
    main()
  





    

