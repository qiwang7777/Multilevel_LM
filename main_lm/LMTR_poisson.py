from Multilevel_LM.main_lm.params_options import LMTR_params_options
from Multilevel_LM.main_lm.loss_poisson import loss_solving_poisson,compute_loss_gradients
from Multilevel_LM.main_lm.subsolver_poisson import Taylor_solver,sub_A_solving_poisson,sub_b_solving_poisson
import torch
import copy
from scipy.sparse.linalg import cg, LinearOperator,splu
import numpy as np
from scipy.sparse import csc_matrix





# Flatten the model's parameters into a single vector
def flatten_parameters(model):
    return torch.cat([param.view(-1) for param in model.parameters()])

# Unflatten the new parameters and update the model, then return the updated model
def update_model_parameters(model, s):
    model_new = copy.deepcopy(model)
    # Flatten the current model parameters
    flattened_params = flatten_parameters(model_new)
    
    # Add the current flattened parameters to s
    new_flattened_params = flattened_params + s
    
    # Get the iterator for the model's parameters
    param_iterator = iter(model_new.parameters())
    
    # Initialize the current position in the 'new_flattened_params' tensor
    current_pos = 0
    
    # Update each parameter in the model
    for param in param_iterator:
        # Get the size of the current parameter
        num_param_elements = param.numel()
        
        # Extract the corresponding part of 'new_flattened_params' and reshape it to match the param's shape
        new_param = new_flattened_params[current_pos:current_pos + num_param_elements].view(param.size())
        
        # Update the parameter without affecting gradients
        with torch.no_grad():
            param.copy_(new_param)
        
        # Move the current position forward
        current_pos += num_param_elements

    # Return the updated model
    return  model, model_new






        
        
def LMTR_solving_poisson(real_solution,model,x,lambdak,regularization=True,lambdap=0.1):
    fk = loss_solving_poisson(real_solution, model, x)
    
    options = LMTR_params_options()
    eta1 = options.eta1
    eta2 = options.eta2
    gamma1 = options.gamma1
    gamma2 = options.gamma2
    gamma3 = options.gamma3
    lambda_min = options.lambda_min
    epsilon = options.epsilon
    max_iter = 100
    k=0
    while torch.norm(compute_loss_gradients(real_solution,model,x,regularization=True,lambdap = 0.1))>=epsilon and k<=max_iter:
        #print(torch.norm(compute_loss_gradients(real_solution,model,x,regularization=True,lambdap = 0.1))) 
        
        A = sub_A_solving_poisson(real_solution,model,x,lambdak)
        #check the reason why CG didn't converge
        #check if A is poor conditioning
        #AA = A @ A.T
        #singular_values = torch.svd(AA).S
        #cond_number = singular_values[0] / singular_values[-1]  # max/smallest singular value

        # Define a threshold for poor conditioning
        #threshold = 1e10  # Example threshold
        #if cond_number >threshold:
        #    print(f"Matrix A is poorly conditioned")
          
        #check if A is SPD
        #def is_spd(A):
        #    if not torch.allclose(A,A.T):
        #        print("Matrix A is not symetric")
        #        return 
        #    eigenvalues = torch.linalg.eigvals(A).real
        #    if torch.all(eigenvalues <= 0):
        #        print("Matrix A is not positive definite.")
        #        return
            
        #is_spd(A)
        
       
        


        b = (-1)*sub_b_solving_poisson(real_solution,model,x,lambdak)
        #A_sparse = csc_matrix(A.detach().numpy())
        
        s = np.linalg.solve(A.detach().numpy(),b.detach().numpy().flatten())
        #try:
        #    s,info = cg(A.detach().numpy(),b.detach().numpy().flatten())
        #    if info > 0:
        #        print(f"Conjugate gradient did not converge after {info} iterations.")
        #        break
        #except Exception as e:
        #    print(f"Error in conjugate gradient solver: {e}")
        #    break
    
        
        s = torch.tensor(s)
        new_model = update_model_parameters(model, s)[1]
        fks = loss_solving_poisson(real_solution, new_model, x)
        mks = Taylor_solver(real_solution,model,x,lambdak,s)
        fk = loss_solving_poisson(real_solution, model, x)
        #print(fk)
        mk = Taylor_solver(real_solution, model, x, lambdak, torch.zeros(sum(p.numel() for p in model.parameters())))
        #print(mk)
        ared = fk-fks
        pred = mk-mks
        
        pho = ared/pred
        #print(pho)
        print(fk)
        
        if pred == 0 :
            print("pred = 0")
        
        else:
       
    
            if pho >= eta1:
                model = new_model
                if pho >= eta2:
                    lambdak = max(lambda_min,gamma2*lambdak)
                else:
                    lambdak = max(lambda_min,gamma1*lambdak)
            else:
                model = model
                lambdak = gamma3*lambdak
            
        k+=1
    return model(x)
