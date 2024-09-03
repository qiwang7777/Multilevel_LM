#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:32:05 2024

@author: Baraldi
"""

from ROL import Objective

import torch



class TorchObjective(Objective):
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

    def gradient(self, g, x, tol):
        ans = self.torch_gradient(x)
        g.copy_(ans) 

    def _forward_over_reverse(self, input, x, v):
        # https://github.com/google/jax/blob/main/docs/notebooks/autodiff_cookbook.ipynb
        return torch.func.jvp(input, (x,), (v,))

    def hessVec(self, hv, v, x, tol):
        _, ans = self._forward_over_reverse(self.torch_gradient, x, v)
        hv.copy_(ans)


class SquaredNorm(TorchObjective):

    def torch_value(self, x):
        return 0.5 * torch.sum(x.squeeze() ** 2)


class TrainingObjective(TorchObjective):

    def __init__(self, model, data, loss):
        super().__init__()
        self.model = model
        self.x, self.y = data
        self.loss = loss

    def torch_value(self, x):
        return self.loss(torch.func.functional_call(self.model, x, self.x), self.y)


class SquaredErrorObjective(TrainingObjective):

    def __init__(self, model, data):
        loss = torch.nn.MSELoss(reduction='sum')
        super().__init__(model, data, loss)
        
#Test for calculating Jacobian and Hessian
#class MyFuncObjective(TorchObjective):

#    def torch_value(self, x):
        # Assuming x is a torch tensor containing [x, y]
#        z, y = x[0], x[1]
#        return z**2 + 2*z*y + y**2


# Test Example
#x = torch.tensor([1.0, 2.0], requires_grad=True)
#v = torch.tensor([1.0, 0.0])
#hv = torch.tensor([0.0, 0.0])

#obj = MyFuncObjective()
#tolerance = 1e-6

# Calculate value
#value = obj.value(x, tolerance)
#print(f"Objective Value: {value}")

# Calculate gradient
#g = torch.zeros_like(x)  # Initialize g with the same shape as x
#obj.gradient(g, x, tolerance)
#print(f"Gradient: {g}")

# Calculate Hessian-vector product
#obj.hessVec(hv, v, x, tolerance)
#print(f"Hessian-Vector Product: {hv}")

#Usage for Nxx
#import numpy as np
#import Multilevel_LM
#from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN

#input_dim = 1
#output_dim = 1
#n_hidden_layers = 1
#r_nodes_per_layer = 500
#activation_function = torch.sigmoid

# Create the network
#My_model_nn = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)



#class NNObjective(TorchObjective):
    
#    def torch_value(self,x):
#        return torch.sum(My_model_nn(x))
    
#Test Example
#sample_num = 3
#xx = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
#value = torch.zeros_like(xx)
#gradient = torch.zeros_like(xx)
#hessian = torch.zeros_like(xx)
#for i in range(sample_num):
#    hv = torch.tensor([0.0])
#    v = torch.tensor([1.0]) #v could also be -1.0
#    xx_i = xx[i]
#    obj_nn = NNObjective()
#    tolerance = 1e-6
    #Calculte the value
    
#    value[i] = obj_nn.value(xx_i,tolerance)
    #print(f"Ouput of NN with {xx_i}: {value[i]}")
    
#    g_i = torch.zeros_like(xx_i)
   
#    obj_nn.gradient(g_i, xx_i, tolerance)
#    gradient[i] = g_i
    
    #print(f"Gradient of NN with repect to {xx_i} :{g_i}")
    
#    obj_nn.hessVec(hv, v, xx_i, tolerance)
#    hessian[i] = hv
    #print(f"Hessian-Vector Product of NN w.r.t. {xx_i}: {hv}")
#print(gradient)
#print(hessian)



#Test for objective loss function
from Multilevel_LM.main_lm.Loss_function import compute_loss
from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
from Multilevel_LM.main_lm.PDE_POISSON import PDE
import numpy as np
import torch.nn as nn
input_dim = 1
output_dim = 1
n_hidden_layers = 1
r_nodes_per_layer = 500
activation_function = torch.sigmoid
model = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)
def real_solution_1d(x):
    return torch.sin(x)

pde = PDE((0,1),real_solution_1d) 



def initialize_parameters(input_dim,r,init_type = 'he'):
    np.random.seed(1234)
    if init_type == 'he':
        w1 = np.random.randn(r,input_dim)*np.sqrt(2./input_dim)
        b1 = np.zeros((r,1))
        w2 = np.random.randn(1,r)*np.sqrt(2./r)
        b2 = np.zeros((1,1))
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

    parameters = {
        'w1': w1,
        'b1': b1.reshape(-1),
        'w2': w2,
        'b2': b2.reshape(-1)
    }

    return parameters


def loss_p(params_nn,regularization=True,lambdap = 0.1):
   
    input_dim = 1
    output_dim = 1
    n_hidden_layers = 1
    r_nodes_per_layer = 500
    activation_function = torch.sigmoid
    sample_num = 3 #could change
    x = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
    model = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)
    weights = [
        params_nn['w1'],          # First hidden layer
        params_nn['w2']          # Output layer
    ]

    biases = [
        params_nn['b1'],  # First hidden layer
        params_nn['b2']          # Output layer
    ]
    with torch.no_grad():
        # Set weights and biases for the first hidden layer
        model.hidden_layers[0].weight = nn.Parameter(torch.tensor(weights[0], dtype=torch.float32))
        model.hidden_layers[0].bias = nn.Parameter(torch.tensor(biases[0], dtype=torch.float32))
    #
        # Set weights and biases for the remaining hidden layers
        for i in range(1, n_hidden_layers):
            model.hidden_layers[i].weight = nn.Parameter(torch.tensor(weights[i], dtype=torch.float32))
            model.hidden_layers[i].bias = nn.Parameter(torch.tensor(biases[i], dtype=torch.float32))

        # Set weights and biases for the output layer
        model.output_layer.weight = nn.Parameter(torch.tensor(weights[-1], dtype=torch.float32))
        model.output_layer.bias = nn.Parameter(torch.tensor(biases[-1], dtype=torch.float32))
        
    grid_points = torch.linspace(0, 1, sample_num, dtype=torch.float32).reshape(-1, 1)
    inner_sample = sample_num - 2
    
    # Compute source term for the real solution (torch tensor)
    source_term = pde.compute_source_term(grid_points, real_solution_1d).reshape(-1, 1)
    
    # Compute model prediction (torch tensor)
    model_input = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    model_output = model(model_input)
    
    # Compute the source term from the model output (torch tensor)
    NN_source_term = pde.compute_source_term(grid_points, lambda x: model(x)).reshape(-1, 1)
    
    # Compute main cost (torch tensor)
    main_cost = (source_term - NN_source_term)[1:-1]
    main_cost_loss = 0.5 * torch.norm(main_cost)**2 / inner_sample
    
    # Compute regularization term if needed (torch tensor)
    if regularization==True:
        real_0 = real_solution_1d(x[0])
        real_end = real_solution_1d(x[-1])
        nn_0 = model(model_input[0].reshape(1, -1))
        nn_end = model(model_input[-1].reshape(1, -1))
        
        regularization_term = (torch.norm(real_0 - nn_0)**2 + torch.norm(real_end - nn_end)**2)
        regularization_term = lambdap * 0.5 * regularization_term
    else:
        regularization_term = torch.tensor(0.0)
    
    # Total loss (torch tensor)
    total_loss = main_cost_loss + regularization_term
    
    return total_loss

def loss_p_tensor(params_nn,regularization=True,lambdap = 0.1):
   
    input_dim = 1
    output_dim = 1
    n_hidden_layers = 1
    r_nodes_per_layer = 5
    activation_function = torch.sigmoid
    sample_num = 3 #could change
    x = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
    model = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)
    weights = [
        params_nn[0:r_nodes_per_layer].reshape(r_nodes_per_layer,1),          # First hidden layer
        params_nn[2*r_nodes_per_layer:3*r_nodes_per_layer].reshape(1,r_nodes_per_layer)          # Output layer
    ]

    biases = [
        params_nn[r_nodes_per_layer:2*r_nodes_per_layer],  # First hidden layer
        params_nn[-1]          # Output layer
    ]
    with torch.no_grad():
        # Set weights and biases for the first hidden layer
        model.hidden_layers[0].weight = nn.Parameter(torch.tensor(weights[0], dtype=torch.float32))
        model.hidden_layers[0].bias = nn.Parameter(torch.tensor(biases[0], dtype=torch.float32))
    #
        # Set weights and biases for the remaining hidden layers
        for i in range(1, n_hidden_layers):
            model.hidden_layers[i].weight = nn.Parameter(torch.tensor(weights[i], dtype=torch.float32))
            model.hidden_layers[i].bias = nn.Parameter(torch.tensor(biases[i], dtype=torch.float32))

        # Set weights and biases for the output layer
        model.output_layer.weight = nn.Parameter(torch.tensor(weights[-1], dtype=torch.float32))
        model.output_layer.bias = nn.Parameter(torch.tensor(biases[-1], dtype=torch.float32))
        
    grid_points = torch.linspace(0, 1, sample_num, dtype=torch.float32).reshape(-1, 1)
    inner_sample = sample_num - 2
    
    # Compute source term for the real solution (torch tensor)
    source_term = pde.compute_source_term(grid_points, real_solution_1d).reshape(-1, 1)
    
    # Compute model prediction (torch tensor)
    model_input = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    model_output = model(model_input)
    
    # Compute the source term from the model output (torch tensor)
    NN_source_term = pde.compute_source_term(grid_points, lambda x: model(x)).reshape(-1, 1)
    
    # Compute main cost (torch tensor)
    main_cost = (source_term - NN_source_term)[1:-1]
    main_cost_loss = 0.5 * torch.norm(main_cost)**2 / inner_sample
    
    # Compute regularization term if needed (torch tensor)
    if regularization==True:
        real_0 = real_solution_1d(x[0])
        real_end = real_solution_1d(x[-1])
        nn_0 = model(model_input[0].reshape(1, -1))
        nn_end = model(model_input[-1].reshape(1, -1))
        
        regularization_term = (torch.norm(real_0 - nn_0)**2 + torch.norm(real_end - nn_end)**2)
        regularization_term = lambdap * 0.5 * regularization_term
    else:
        regularization_term = torch.tensor(0.0)
    
    # Total loss (torch tensor)
    total_loss = main_cost_loss + regularization_term
    
    return total_loss


class Objective_nn_pde(TorchObjective):
    
    def torch_value(self,params_nn):
        
        return loss_p_tensor(params_nn)
    
    
obj_nn_pde = Objective_nn_pde()
tolerance = 1e-6
param_test = initialize_parameters(1, 5)
w1_flat = torch.tensor(param_test['w1'].flatten())
w2_flat = torch.tensor(param_test['w2'].flatten())
b1_flat = torch.tensor(param_test['b1'].flatten())
b2_flat = torch.tensor(param_test['b2'].flatten())
param_test_flatten = torch.cat([w1_flat,b1_flat,w2_flat,b2_flat])
value = obj_nn_pde.value(param_test_flatten,tolerance)
#print(value)
#Calculate gradient
g_nn_pde = torch.zeros_like(param_test_flatten)
obj_nn_pde.gradient(g_nn_pde,param_test_flatten,tolerance)
print(g_nn_pde)
#g = torch.zeros_like(x)  # Initialize g with the same shape as x
#obj.gradient(g, x, tolerance)
#print(f"Gradient: {g}")  
#print(loss_p_tensor(param_test_flatten))
#Calculte the gradient and second derivative of NN w.r.t. input x
def compute_first_derivative(model,input_data):
    input_data.requires_grad = True

    # Forward pass: Compute the network output
    output = model(input_data)

    # First derivative: Compute the gradient of the output with respect to the input
    first_derivative = torch.autograd.grad(outputs=output, inputs=input_data,
                                           grad_outputs=torch.ones_like(output),
                                           create_graph=True, allow_unused=True)[0]
    return first_derivative
    
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
