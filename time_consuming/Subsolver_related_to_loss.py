
import numpy as np
import torch
import torch.autograd as autograd
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd import elementwise_grad
from Multilevel_LM.time_consuming.PDE_POISSON import PDE
def Fk(input_data, model, pde, real_solution):
    sample_num = input_data.shape[0]
    grid_points = torch.linspace(0, 1, sample_num, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    
    # Compute source term for the real solution (torch tensor)
    source_term = pde.compute_source_term(grid_points, real_solution).reshape(-1, 1)
    
    # Compute model prediction (torch tensor)
    model_input = torch.tensor(input_data, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    #model_output = model(model_input)
    
    # Compute the source term from the model output (torch tensor)
    NN_source_term = pde.compute_source_term(grid_points, lambda x: model(x)).reshape(-1, 1)
    
    # Compute main cost (torch tensor)
    main_cost = (source_term - NN_source_term)[1:-1]
    
    return main_cost
#PDE_Nontensor
def Jk(input_data, model, pde, real_solution):
    # Compute Fk
    main_cost = Fk(input_data, model, pde, real_solution)
    
    # Ensure requires_grad is True for model parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Initialize an empty list to store the gradients (Jacobian rows)
    jacobian = []
    
    # Compute the Jacobian matrix
    for i in range(main_cost.size(0)):
        # Zero the gradients
        model.zero_grad()
        
        # Compute the gradient of the i-th component of Fk w.r.t. model parameters
        grad_i = autograd.grad(main_cost[i], model.parameters(), retain_graph=True, create_graph=True)
        
        # Flatten and concatenate the gradients into a single vector
        jacobian_row = torch.cat([g.view(-1) for g in grad_i])
        
        # Append the row to the Jacobian matrix
        jacobian.append(jacobian_row)
    
    # Stack the list of Jacobian rows into a matrix
    jacobian = torch.stack(jacobian)
    
    return jacobian

# Define input data
input_data = torch.linspace(0, 1, 40)

# Define your PDE and real solution functions
class PDE_Nontensor:
    def __init__(self, domain, real_solution):
        self.domain = domain
        self.real_solution = real_solution

    def compute_source_term(self, grid_points, solution_func):
        # Example implementation (should be adapted to your PDE)
        return solution_func(grid_points)

from Multilevel_LM.main_lm.test import real_solution_1d

pde = PDE_Nontensor((0,1), real_solution_1d)

def line_A(input_data, model, pde, real_solution, lambda0):
    input_data_tensor = torch.tensor(input_data)
    #F_k = Fk(input_data_tensor, model, PDE((0,1), real_solution_1d), real_solution_1d)
    J_k = Jk(input_data_tensor, model, pde, real_solution_1d)
    B_k = J_k.T@J_k
    shape_eye = B_k.shape[0]
    return B_k+lambda0*torch.eye(shape_eye)


def line_b(input_data, model, pde, real_solution, lambda0):
    input_data_tensor = torch.tensor(input_data)
    F_k = Fk(input_data_tensor, model, pde, real_solution_1d)
    J_k = Jk(input_data_tensor, model, pde, real_solution_1d)
    return -1*J_k.T @F_k


