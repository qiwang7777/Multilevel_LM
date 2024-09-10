import torch
import torch.nn as nn
import numpy as np
from Multilevel_LM.main_lm.Subsolver_related_to_loss import line_A,line_b,Fk,Jk
from Multilevel_LM.main_lm.test import real_solution_1d

from Multilevel_LM.main_lm.Loss_function import compute_loss
from scipy.sparse.linalg import cg

def taylor_loss(s,input_data, model, pde, real_solution, lambda0=0.05, regularization=False):
    #s = torch.randn(3*r_nodes_per_layer+1)
    input_data_tensor = torch.tensor(input_data)
    zero_term = compute_loss(input_data, model, pde, real_solution,regularization=False)
    F_k = Fk(input_data_tensor, model, pde, real_solution_1d)
    J_k = Jk(input_data_tensor, model, pde, real_solution_1d)
    first_term = F_k.T@J_k@s
    B_k = J_k.T@J_k
    second_term = 0.5*(s.T @ B_k @ s)
    if regularization == True:
        re_term = 0.5*lambda0*torch.norm(s)**2
    else:
        re_term = 0
    return zero_term+first_term+second_term+re_term

def LMTR(input_data, model, pde, real_solution, lambdak=0.1, regularization=False):
    #input_data = torch.tensor(input_data)
    input_dim = model.hidden_layers[0].in_features
    #n_hidden_layers = 1
    r_nodes_per_layer = model.hidden_layers[0].out_features
    #output_dim = 1
    #activation_function = torch.sigmoid
    #Set up the neural network
    #Model = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)
    #Initialize the neural network
    #model_param_init = initialize_parameters(input_dim, r_nodes_per_layer)
    w10 = model.hidden_layers[0].weight.data.numpy()
    b10 = model.hidden_layers[0].bias.data.numpy()
    w20 = model.output_layer.weight.data.numpy()
    b20 = model.output_layer.bias.data.numpy()
    #model_param_init = initialize_parameters(input_dim, r_nodes_per_layer)
    #w10 = model_param_init['w1']
    #b10 = model_param_init['b1']
    #w20 = model_param_init['w2']
    #b20 = model_param_init['b2']
    
    
    #model_NN = model_nn(input_data,model = Model)
    
    #Loss = compute_loss(input_data, model, pde, real_solution,lambdak,True)
    ###Optional parameters
    eta1 = 0.1
    eta2 = 0.75
    gamma1 = 0.85
    gamma2 = 0.5
    gamma3 = 1.5
    lambda0 = 0.05
    lambda_min = 10**(-4)
    epsilon = 10**(-4)
    max_iter = 1000
    k = 0
    #Get the gradient
    
    input_data_tensor = torch.tensor(input_data)
    #model_tt = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim)
    loss = compute_loss(input_data_tensor, model, pde, real_solution, lambdak, regularization=True)

    # Compute gradients
    model.zero_grad()
    loss.backward()  # This should now work since loss is a PyTorch tensor

    # Access gradients
    gradients = torch.cat([param.grad.view(-1) for param in model.parameters()])
    w10_1d = torch.tensor(w10.squeeze())
    w20_1d = torch.tensor(w20.squeeze())
    b10 = torch.tensor(b10)
    b20 = torch.tensor(b20)
    allparams = torch.cat([w10_1d,b10,w20_1d,b20])
    while k<=max_iter and np.linalg.norm(gradients)>=epsilon:
        #print(np.linalg.norm(gradients))
        A_sub = line_A(input_data, model, pde, real_solution, lambda0)
        b_sub = line_b(input_data, model, pde, real_solution, lambda0)
        A_np = A_sub.detach().numpy()
        b_np = b_sub.detach().numpy()

        s_np, info = cg(A_np,b_np)
        s = torch.from_numpy(s_np)
        #print(s)
        
        pred = taylor_loss(torch.zeros_like(b_sub), input_data, model, pde, real_solution,lambda0,regularization=True)-taylor_loss(s, input_data, model, pde, real_solution,lambda0,regularization=True)
        fk = compute_loss(input_data, model, pde, real_solution, lambdak, regularization=True)
        new_params = allparams+s
        w10_new = new_params[:r_nodes_per_layer].view((r_nodes_per_layer,1))
        b10_new = new_params[r_nodes_per_layer:2*r_nodes_per_layer]
        w20_new = new_params[2*r_nodes_per_layer:3*r_nodes_per_layer].view((1,r_nodes_per_layer))
        b20_new = new_params[-1]
        model_new = model
        with torch.no_grad():
            # Set weights and biases for the first hidden layer
            model_new.hidden_layers[0].weight = nn.Parameter(torch.tensor(w10_new, dtype=torch.float32))
            model_new.hidden_layers[0].bias = nn.Parameter(torch.tensor(b10_new, dtype=torch.float32))

            # Set weights and biases for the output layer
            model_new.output_layer.weight = nn.Parameter(torch.tensor(w20_new, dtype=torch.float32))
            model_new.output_layer.bias = nn.Parameter(torch.tensor(b20_new, dtype=torch.float32))
        
        #model_NN = model_nn(input_data,model = model)
        fks = compute_loss(input_data, model_new, pde, real_solution, lambdak, regularization=True)
        ared = fk-fks
        #print(pred)
        pho = ared/pred
        print(pho)
        if pred == 0:
            break
        elif pho >= eta1:
            w10 = w10_new
            b10 = b10_new
            w20 = w20_new
            b20 = b20_new
            #model = model_new
            if pho>=eta2:
                lambda0 = max(lambda_min,gamma2*lambda0)
            else:
                lambda0 = max(lambda_min,gamma1*lambda0)
                
            with torch.no_grad():
            # Set weights and biases for the first hidden layer
                model_new.hidden_layers[0].weight = nn.Parameter(torch.tensor(w10, dtype=torch.float32))
                model_new.hidden_layers[0].bias = nn.Parameter(torch.tensor(b10, dtype=torch.float32))

            # Set weights and biases for the output layer
                model_new.output_layer.weight = nn.Parameter(torch.tensor(w20, dtype=torch.float32))
                model_new.output_layer.bias = nn.Parameter(torch.tensor(b20, dtype=torch.float32)) 
                model = model_new
        else:
            w10 = w10
            b10 = b10
            w20 = w20
            b20 = b20
            #model = model
            lambda0 = gamma3*lambda0
        
        

       
        loss = compute_loss(input_data_tensor, model, pde, real_solution, lambdak, regularization=True)

        # Compute gradients
        model.zero_grad()
        loss.backward()  # This should now work since loss is a PyTorch tensor

        # Access gradients
        gradients = torch.cat([param.grad.view(-1) for param in model.parameters()])
    
        k += 1
    sample_num = 40    
    input_data = torch.tensor(input_data.reshape(sample_num,input_dim), dtype=torch.float32)
            
    return model(input_data)-real_solution_1d(input_data)
