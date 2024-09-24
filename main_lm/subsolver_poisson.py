#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file, we gave the functions which will be used for subsolver 
i.e. get the linear equation from Taylor model, to solve As = b, 
here, sub_A_solving_poisson is A, but sub_b_solving_poisson is actually -b.
"""
import numpy as np
import torch
from Multilevel_LM.main_lm.PoissonPDE import PoissonPDE
from Multilevel_LM.main_lm.loss_poisson import get_1d_boundary, get_2d_boundary
from Multilevel_LM.main_lm.neural_network_construction import compute_flatten_gradients
def Fk1_solving_poisson(real_solution,model,x,regularization=True,lambdap = 0.1):
    input_dim = model.hidden_layers[0].in_features
    
    if input_dim == 1:
        pde_1d = PoissonPDE(real_solution,x)
        real_source = pde_1d._compute_1d_source_term(x)
        nn_pde = PoissonPDE(model,x)
        nn_source = nn_pde._compute_1d_source_term(x)
        main_cost = (real_source-nn_source)
        
        
    if input_dim == 2:
        
        pde_2d = PoissonPDE(real_solution,x)
        real_source = pde_2d._compute_2d_source_term(x).reshape(-1,1)
        nn_pde = PoissonPDE(model,x)
        nn_source = nn_pde._compute_2d_source_term(x).reshape(-1,1)
        main_cost = (real_source-nn_source)
        
    return main_cost


def Fk2_solving_poisson(real_solution,model,x,regularization=True,lambdap = 0.1):
    input_dim = model.hidden_layers[0].in_features
    if input_dim == 1:
        x_boundary = get_1d_boundary(x).reshape(-1,1)
        real = real_solution(x_boundary)
        nn = model(x_boundary)
        re_term = real-nn
        
    if input_dim == 2:
        x_boundary = get_2d_boundary(x)
        real = real_solution(x_boundary).reshape(-1,1)
        nn = model(x_boundary)
        re_term = real-nn
        
    return re_term


def Jk1_solving_poisson(real_solution,model,x,regularization=True,lambdap = 0.1):
    Fk1 = lambda model,x: Fk1_solving_poisson(real_solution, model, x)
    return compute_flatten_gradients(model,Fk1,x)

def Jk2_solving_poisson(real_solution,model,x,regularization=True,lambdap = 0.1):
    Fk2 = lambda model,x: Fk2_solving_poisson(real_solution, model, x,regularization=True,lambdap = 0.1)
    return compute_flatten_gradients(model,Fk2,x)

def sub_A_solving_poisson(real_solution,model,x,lambdak,regularization=True,lambdap = 0.1):
    input_dim = model.hidden_layers[0].in_features
    s_size = sum(p.numel() for p in model.parameters())
    if input_dim == 1:
        sample_num = x.shape[0]-2
        J1 = Jk1_solving_poisson(real_solution,model,x)
        
        J2 = Jk2_solving_poisson(real_solution, model, x)
        boundary_num = 2
        
    if input_dim == 2:
        sample_num = (np.sqrt(x.shape[0])-1)**2 
        J1 = Jk1_solving_poisson(real_solution, model, x)
        J2 = Jk2_solving_poisson(real_solution, model, x)
        boundary_num = x.shape[0]-sample_num
    return J1.T @ J1/sample_num+lambdap*J2.T@J2/boundary_num+lambdak*torch.eye(s_size)



def sub_b_solving_poisson(real_solution,model,x,lambdak,regularization=True,lambdap = 0.1):
    input_dim = model.hidden_layers[0].in_features
    F1 = Fk1_solving_poisson(real_solution, model, x)
    J1 = Jk1_solving_poisson(real_solution, model, x)
    F2 = Fk2_solving_poisson(real_solution, model, x)
    J2 = Jk2_solving_poisson(real_solution, model, x)
    if input_dim == 1:
        sample_num = x.shape[0]-2
        boundary_num = 2
    if input_dim == 2:
        sample_num = (np.sqrt(x.shape[0])-1)**2 
        boundary_num = x.shape[0]-sample_num
    return J1.T@F1/sample_num+lambdap*J2.T@F2/boundary_num
    
def Taylor_solver(real_solution,model,x,lambdak,s,regularization=True,lambdap=0.1):
    input_dim = model.hidden_layers[0].in_features
    F1 = Fk1_solving_poisson(real_solution, model, x)
    J1 = Jk1_solving_poisson(real_solution, model, x)
    F2 = Fk2_solving_poisson(real_solution, model, x)
    J2 = Jk2_solving_poisson(real_solution, model, x)
    if input_dim == 1:
        sample_num = x.shape[0]-2
        boundary_num = 2
    if input_dim == 2:
        sample_num = (np.sqrt(x.shape[0])-1)**2 
        boundary_num = x.shape[0]-sample_num
    m1 = (torch.norm(F1)**2+2*F1.T@J1@s+s.T@J1.mT@J1@s)/(2*sample_num)
    m2 = (torch.norm(F2)**2+2*F2.T@J2@s+s.T@J2.mT@J2@s)/(2*boundary_num)
    m3 = lambdak*torch.norm(s)**2/2
    return m1+m2+m3

    
    
    
#Test

#def test_func_1d(x):
#    return 2*x+x**3-x**2
#from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
#input_dim = 1
#output_dim = 1
#n_hidden_layers = 1
#r_nodes_per_layer = 2
#model_21 = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim)
#x_1d = torch.tensor(np.linspace(0,1,5).reshape(5,1), dtype=torch.float32)
#print(Jk1_solving_poisson(test_func_1d, model_21, x_1d,regularization=True))
#print(sub_b_solving_poisson(test_func_1d, model_21, x_1d, 0.01))
#print(Taylor_solver(test_func_1d, model_21, x_1d, 0.1, torch.ones(7)))
# Example usage for 2D case
#input_dim_2d = 2
#output_dim = 1
#n_hidden_layers = 1
#r_nodes_per_layer = 2
#model_21_2d = FullyConnectedNN(input_dim_2d, n_hidden_layers, r_nodes_per_layer, output_dim)


#x = np.linspace(0, 1, 5)
#y = np.linspace(0, 1, 5)

# Create a 2D grid
#X, Y = np.meshgrid(x, y)

# Flatten the grid and stack x and y coordinates
#input_data = np.stack([X.flatten(), Y.flatten()], axis=1)
#def test_func_2d(x):
#    return torch.sin(x[:,0])+x[:,1]
# Convert to PyTorch tensor
#x_2d = torch.tensor(input_data, dtype=torch.float32)

#print(Jk1_solving_poisson(test_func_2d,model_21_2d,x_2d)) 
#print(sub_b_solving_poisson(test_func_2d, model_21_2d, x_2d, 0.03))    
#print(Taylor_solver(test_func_2d, model_21_2d, x_2d, 0.03, torch.ones(9)))    
        
    
