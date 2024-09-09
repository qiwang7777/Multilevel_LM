#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:57:07 2024

@author: wang
"""

from Multilevel_LM.main_lm.PoissonPDE import PoissonPDE
import torch
import numpy as np
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
        pde_1d = PoissonPDE(real_solution,x)
        real_source = pde_1d._compute_1d_source_term(x)
        nn_pde = PoissonPDE(model,x)
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

def compute_loss_gradients(real_solution,model,x,regularization=True,lambdap = 0.1):
    #Forward pass: Perform forward pass to compute network output
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
            
#Test
#def test_func_1d(x):
#    return torch.sin(x)
#from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
#input_dim = 1
#output_dim = 1
#n_hidden_layers = 1
#r_nodes_per_layer = 2
#model_21 = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim)
#x_1d = torch.tensor(np.linspace(0,1,5).reshape(5,1), dtype=torch.float32)
#print(loss_solving_poisson(test_func_1d, model_21, x_1d,regularization=True))

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

#print(compute_loss_gradients(test_func_2d,model_21_2d,x_2d)) 
        
    