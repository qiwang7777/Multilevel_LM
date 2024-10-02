#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
#from Multilevel_LM.main_lm.PoissonPDE import PoissonPDE
from Multilevel_LM.main_lm.loss_poisson import loss_solving_poisson,compute_loss_gradients
from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
from Multilevel_LM.main_lm.LMTR_poisson import update_model_parameters
from Multilevel_LM.mlm_main.average_strategies import average_nodes_model, restriction
from Multilevel_LM.main_lm.subsolver_poisson import sub_A_solving_poisson,sub_b_solving_poisson


def create_block_matrix_torch(R, m):
    """
    Creates a block matrix where the diagonal consists of m copies of matrix R, 
    and the last diagonal element is filled with 1. 
    The output matrix will be non-square.
    
    Args:
        R (torch.Tensor): The matrix R to place on the diagonal.
        m (int): The number of times matrix R should appear on the diagonal.
        
    Returns:
        block_matrix (torch.Tensor): The block matrix.
    """
    r_rows, r_cols = R.shape
    # Total size for block matrix
    total_rows = m * r_rows + 1  # Number of rows
    total_cols = m * r_cols + 1  # Number of columns

    # Initialize the block matrix with zeros
    block_matrix = torch.zeros((total_rows, total_cols))

    # Place matrix R on the diagonal m times
    for i in range(m):
        start_idx = i * r_rows
        block_matrix[start_idx:start_idx + r_rows, i * r_cols:i * r_cols + r_cols] = R

    # Place 1 at the last diagonal element
    block_matrix[-1, -1] = 1

    return block_matrix




def Taylor_H(real_solution,model,x,lambdak,sH,m=2,regularization=True,lambdap=0.1):
    input_dim = model.input_dim
    new_model = average_nodes_model(model,m)
    new_model_s = update_model_parameters(new_model, sH)[1]
    R = restriction(model,m)
    if input_dim == 1 and model.n_hidden_layers == 1:
        R_extend = create_block_matrix_torch(R,3)
    elif input_dim == 2 and model.n_hidden_layers == 1:
        R_extend = create_block_matrix_torch(R,4)
    fHs = loss_solving_poisson(real_solution,new_model_s,x, regularization=True,lambdap=0.1)
    grad_fh = compute_loss_gradients(real_solution, model, x,regularization=True,lambdap=0.1)
    grad_fH = compute_loss_gradients(real_solution, new_model, x)
    return fHs+(R_extend@grad_fh-grad_fH).T@sH + 0.5*lambdak*torch.norm(sH)**2


def sub_A_H(real_solution,model,x,lambdak,m=2,regularization=True,lambdap=0.1):
    #input_dim = model.input_dim
    new_model = average_nodes_model(model, m)
    return sub_A_solving_poisson(real_solution, new_model, x, lambdak)

def sub_b_H(real_solution,model,x,lambdak,m=2,regularization=True,lambdap=0.1):
    input_dim = model.input_dim
    new_model = average_nodes_model(model, m)
    R = restriction(model,m)
    if input_dim == 1 and model.n_hidden_layers == 1:
        R_extend = create_block_matrix_torch(R,3)
    elif input_dim == 2 and model.n_hidden_layers == 1:
        R_extend = create_block_matrix_torch(R,4)
    grad_fh = compute_loss_gradients(real_solution, model, x,regularization=True,lambdap=0.1)
    grad_fH = compute_loss_gradients(real_solution, new_model, x)
    return sub_b_solving_poisson(real_solution, new_model, x, lambdak) + (R_extend@grad_fh-grad_fH).view(-1,1)    
#test

#model = FullyConnectedNN(input_dim=1, n_hidden_layers=1, r_nodes_per_layer=6, output_dim=1)
#new_model = average_nodes_model(model,2)
#def test_func_1d(x):
#    return torch.sin(x)
#real_solution = test_func_1d
#sample_num = 41
#x = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,1), dtype=torch.float32)
#grad_fh = compute_loss_gradients(real_solution, model, x,regularization=True,lambdap=0.1)

#R = restriction(model,2)
#sH=torch.ones(10).view(-1,1)

#grad_fH = compute_loss_gradients(real_solution, new_model, x)

#print(sub_A_H(real_solution,model,x,0.1).shape)
#print(sub_b_H(real_solution, model, x, 0.1))