#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Multilevel_LM.main_lm.params_options import MLM_TR_params_options
from Multilevel_LM.main_lm.loss_poisson import loss_solving_poisson,compute_loss_gradients

from Multilevel_LM.main_lm.LMTR_poisson import update_model_parameters,LMTR_solving_poisson
import torch

#from scipy.sparse.linalg import cg, LinearOperator,splu
import numpy as np
#from scipy.sparse import csc_matrix
from Multilevel_LM.mlm_main.subsolver_two_level import create_block_matrix_torch,Taylor_H, sub_A_H,sub_b_H
from Multilevel_LM.mlm_main.average_strategies import restriction,average_nodes_model
def MLM_TR(real_solution,model,x,lambdak,m=2,regularization=True,lambdap =0.1,l=2):
    #fk = loss_solving_poisson(real_solution,model,x)
    
    options = MLM_TR_params_options()
    eta1 = options.eta1
    eta2 = options.eta2
    gamma1 = options.gamma1
    gamma2 = options.gamma2
    gamma3 = options.gamma3
    lambda_min = options.lambda_min
    epsilon = options.epsilon
    kappaH = options.kappaH
    epsilonH = options.epsilonH
    max_iter = 100
    k=0
    R = restriction(model,m)
    input_dim = model.input_dim
    
   
    if input_dim == 1 and model.n_hidden_layers == 1:
        R_extend = create_block_matrix_torch(R,3)
    elif input_dim == 2 and model.n_hidden_layers == 1:
        R_extend = create_block_matrix_torch(R,4)
    P_extend = R_extend.T
    
    while torch.norm(compute_loss_gradients(real_solution, model, x,regularization=True,lambdap=0.1))>=epsilon and k <= max_iter:
        #print(torch.norm(compute_loss_gradients(real_solution,model,x,regularization=True,lambdap = 0.1)))
        #print(torch.norm(loss_solving_poisson(real_solution, model, x)))
        grad_fh = compute_loss_gradients(real_solution, model, x,regularization=True,lambdap=0.1)
        if l >1 and torch.norm(R_extend@grad_fh)>=kappaH*torch.norm(grad_fh) and torch.norm(R_extend@grad_fh) > epsilonH:
            AH = sub_A_H(real_solution,model,x,lambdak,m=2,regularization=True,lambdap=0.1)
            bH = (-1)*sub_b_H(real_solution,model,x,lambdak,m=2,regularization=True,lambdap=0.1)
            sH = np.linalg.solve(AH.detach().numpy(),bH.detach().numpy().flatten())
            sH = torch.tensor(sH)
            s = P_extend @ sH
            modelH = average_nodes_model(model, m)
            new_modelH = update_model_parameters(modelH,sH)[1]
            new_modelh = update_model_parameters(model, s)[1]
            fhs = loss_solving_poisson(real_solution, new_modelh, x)
            fh = loss_solving_poisson(real_solution, model, x)
            fH = loss_solving_poisson(real_solution, modelH, x)
            fHs = loss_solving_poisson(real_solution, new_modelH, x)
            mHs = Taylor_H(real_solution,model,x,lambdak,sH)
            mH = Taylor_H(real_solution,model,x,lambdak,torch.zeros(sum(p.numel() for p in modelH.parameters())))
            ared = fH-fHs
            pred = mH-mHs
            
            print(ared,pred)
            pho = ared/pred
            #print(pho)
            if pred == 0:
                print("pred = 0")
            else:
                if pho >= eta1:
                    model = new_modelh
                    if pho >= eta2:
                        lambdak = max(lambda_min,gamma2*lambdak)
                    else:
                        lambdak = max(lambda_min,gamma1*lambdak)
                else:
                    model = model
                    lambdak = gamma3*lambdak
            k += 1

        
            
        else:
            print("just fine case")
            return LMTR_solving_poisson(real_solution,model,x,lambdak,regularization=True,lambdap=0.1)
        
    return model(x)
#TEST           
#def test_func_1d(x):
#    return torch.sin(x)
#from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
#input_dim = 1
#output_dim = 1
#n_hidden_layers = 1
#r_nodes_per_layer = 300
#model_21 = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim)
#sample_num = 41
#x_1d = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,1), dtype=torch.float32)
#pred_1d = MLM_TR(test_func_1d, model_21, x_1d, lambdak=0.1).detach().numpy()       
#print(pred_1d) 
#fh=loss_solving_poisson(test_func_1d, model_21, x_1d) 
#modelH = average_nodes_model(model_21, 2)
#fH =loss_solving_poisson(test_func_1d, modelH, x_1d)

                   
#AH = sub_A_H(test_func_1d,model_21,x_1d,lambdak=0.1,m=2,regularization=True,lambdap=0.1)
#bH = (-1)*sub_b_H(test_func_1d,model_21,x_1d,lambdak=0.1,m=2,regularization=True,lambdap=0.1)
#sH = np.linalg.solve(AH.detach().numpy(),bH.detach().numpy().flatten())
#sH = torch.tensor(sH) 
#new_modelH = update_model_parameters(modelH,sH)[1]
#fHs=loss_solving_poisson(test_func_1d, new_modelH, x_1d)
#mHs = Taylor_H(test_func_1d,model_21,x_1d,0.1,sH)
#mH = Taylor_H(test_func_1d,model_21,x_1d,0.1,torch.zeros(sum(p.numel() for p in modelH.parameters())))          
#print(fh)
#print(fH)
#R = restriction(model_21,2)
#R_extend = create_block_matrix_torch(R,3)
#P_extend = R_extend.T
#s = P_extend@sH

#update_model_parameters(model_21, s)
#modelH = average_nodes_model(model_21, 2)
#mHs = Taylor_H(test_func_1d,model_21,x_1d,0.1,sH)       
            
            
            
            