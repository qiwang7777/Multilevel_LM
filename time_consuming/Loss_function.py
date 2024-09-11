#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:51:34 2024

<<<<<<< Updated upstream:time_consuming/Loss_function.py
@author: wang
"""
from Multilevel_LM.time_consuming.PDE_POISSON import PDE
from Multilevel_LM.main_lm.gradient import initialize_parameters
from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
import torch.nn as nn
import torch
import numpy as np

def real_solution_1d(x):
    return torch.sin(x)

# Network configuration
input_dim = 1
output_dim = 1
n_hidden_layers = 1
r_nodes_per_layer = 5
activation_function = torch.sigmoid

# Create the network
My_model = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)

def model_nn(x,model = My_model):
    x = torch.tensor(x, dtype=torch.float32).reshape(-1,1)
    return model(x).detach().numpy()


def loss_solve_pde(model,input_data,real_solution=real_solution_1d,regularization = False, lambdak = 0.1):
    
    sample_num = input_data.shape[0]
    pde_1d = PDE((0,1),real_solution_1d)
    grid_points_1d = torch.linspace(0, 1, sample_num, dtype=torch.float32).reshape(-1, 1)
    inner_sample = sample_num-2
    source_term_1d = pde_1d.compute_source_term(grid_points_1d,real_solution_1d).reshape(-1,1)
    def model_NN(input_data,Model = model):
        x = torch.tensor(input_data, dtype=torch.float32).reshape(-1,1)
        return Model(x).detach().numpy()
    model_input = torch.tensor(input_data, dtype=torch.float32).reshape(-1, 1)
    model_output = model(model_input) 
    NN_source_term_1d = pde_1d.compute_source_term(grid_points_1d, lambda x: model(x)).reshape(-1, 1)
    #NN_source_term_1d = pde_1d.compute_source_term(grid_points_1d,model_NN)
    main_cost = (source_term_1d-NN_source_term_1d)[1:-1]
    if regularization == True:
        real_0 = real_solution_1d(input_data[0]).reshape(-1,1)
        real_end = real_solution_1d(input_data[-1])
        nn_0 = model_nn(input_data[0])
        nn_end = model_nn(input_data[-1])
        regularization_term = np.linalg.norm(real_0-nn_0)**2+np.linalg.norm(real_end-nn_end)**2
        regularization_term = lambdak*0.5*0.5*regularization_term
    else:
        regularization_term = 0
    
    return 0.5*np.linalg.norm(main_cost)**2/inner_sample +regularization_term


#Test loss_solve_pde

#NN_source_term_1d = pde_1d.compute_source_term(grid_points_1d,model_NN)
#print(loss_solve_pde(My_model, np.linspace(0,1,40),regularization=True))

def compute_loss(input_data, model, pde, real_solution, lambdak=0.1, regularization=False):
    sample_num = input_data.shape[0]
    grid_points = torch.linspace(0, 1, sample_num, dtype=torch.float32).reshape(-1, 1)
    inner_sample = sample_num - 2
    
    # Compute source term for the real solution (torch tensor)
    source_term = pde.compute_source_term(grid_points, real_solution).reshape(-1, 1)
    
    # Compute model prediction (torch tensor)
    model_input = torch.tensor(input_data, dtype=torch.float32).reshape(-1, 1)
    model_output = model(model_input)
    
    # Compute the source term from the model output (torch tensor)
    NN_source_term = pde.compute_source_term(grid_points, lambda x: model(x)).reshape(-1, 1)
    
    # Compute main cost (torch tensor)
    main_cost = (source_term - NN_source_term)[1:-1]
    main_cost_loss = 0.5 * torch.norm(main_cost)**2 / inner_sample
    
    # Compute regularization term if needed (torch tensor)
    if regularization==True:
        real_0 = real_solution(input_data[0])
        real_end = real_solution(input_data[-1])
        nn_0 = model(model_input[0].reshape(1, -1))
        nn_end = model(model_input[-1].reshape(1, -1))
        
        regularization_term = (torch.norm(real_0 - nn_0)**2 + torch.norm(real_end - nn_end)**2)
        regularization_term = lambdak * 0.5 * regularization_term
    else:
        regularization_term = torch.tensor(0.0)
    
    # Total loss (torch tensor)
    total_loss = main_cost_loss + regularization_term
    
    return total_loss

pde = PDE((0,1),real_solution_1d) 
def loss_p(params_nn,regularization=True,lambdap = 0.1):
   """
   params_nn is a list with dictionary 'w1','w2','b1','b2'
   """
   r_nodes_per_layer = params_nn['b1'].size
   input_dim = params_nn['w1'].shape[1]
   output_dim = params_nn['w2'].shape[0]
   n_hidden_layers = 1
   model = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim)
   with torch.no_grad():
       # Set weights and biases for the first hidden layer
        model.hidden_layers[0].weight = nn.Parameter(torch.tensor(params_nn['w1'], dtype=torch.float32))
        model.hidden_layers[0].bias = nn.Parameter(torch.tensor(params_nn['b1'], dtype=torch.float32))
   #
       # Set weights and biases for the remaining hidden layers
       #for i in range(1, n_hidden_layers):
           #model.hidden_layers[i].weight = nn.Parameter(torch.tensor(weights[i], dtype=torch.float32))
           #model.hidden_layers[i].bias = nn.Parameter(torch.tensor(biases[i], dtype=torch.float32))

       # Set weights and biases for the output layer
        model.output_layer.weight = nn.Parameter(torch.tensor(params_nn['w2'], dtype=torch.float32))
        model.output_layer.bias = nn.Parameter(torch.tensor(params_nn['b2'], dtype=torch.float32))
    
    
    
   sample_num = 40
   input_data = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
   loss = compute_loss(input_data, model, pde, real_solution_1d, lambdak=lambdap, regularization=True)
   return loss


   
   

def loss_p_tensor(params_nn_tensor,regularization=True,lambdap = 0.1):
    num_params = params_nn_tensor.shape[0]
    r_nodes_per_layer = int((num_params-1)/3)
    
    params_nn = {
        'w1': params_nn_tensor[0:r_nodes_per_layer].reshape(r_nodes_per_layer,1).numpy(),
        'b1': params_nn_tensor[r_nodes_per_layer:2*r_nodes_per_layer].numpy(),
        'w2': params_nn_tensor[2*r_nodes_per_layer:3*r_nodes_per_layer].reshape(1,r_nodes_per_layer).numpy(),
        'b2': params_nn_tensor[-1].numpy()
         }
    return loss_p(params_nn)
   


def loss_Jk(model,regularization=True,lambdap = 0.1):
    sample_num = 40
    input_data = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
    loss = compute_loss(input_data, model, pde, real_solution_1d, lambdak=lambdap, regularization=True)
    loss.backward()
    params = [p for p in model.parameters()]
    gradients = [p.grad.view(-1) for p in params]
    return gradients

p_tt = initialize_parameters(1,5)
print(loss_p(p_tt))
w1_flat = torch.tensor(p_tt['w1'].flatten())
w2_flat = torch.tensor(p_tt['w2'].flatten())
b1_flat = torch.tensor(p_tt['b1'].flatten())
b2_flat = torch.tensor(p_tt['b2'].flatten())
param_test_flatten = torch.cat([w1_flat,b1_flat,w2_flat,b2_flat])


#param_test_flatten[0:5]
#print(param_test_flatten[5:10].shape[0])
print(param_test_flatten.shape[0])

#print(params['w1'])

#print(p_tt)
#print(params)
print(loss_p_tensor(param_test_flatten))
#Test
#sample_num = 40
#input_data = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
#pde_1d = PDE((0,1),real_solution_1d)
#print(compute_loss(input_data, My_model, pde_1d, real_solution_1d))
#print(loss_solve_pde(My_model, input_data)) #same result with compute_loss
