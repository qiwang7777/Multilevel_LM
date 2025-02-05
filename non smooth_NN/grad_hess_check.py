import numpy as np
import sympy as sp

def gradient(f, x):
    batch_size, input_dim = x.shape
    gradients =[]
    for i in range(batch_size):
        x_sample = x[i].clone().detach().requires_grad_(True)
        output = f(x_sample)
        if output.numel() > 1:
            raise ValueError("The function output must be scalar for gradient computation.")
        

    
        first_grad = torch.autograd.grad(output, x_sample, create_graph=True)[0]
        gradients.append(first_grad)

    return torch.stack(gradients)


def hessian(f, x):
    batch_size, input_dim = x.shape
    hessians = []
    for i in range(batch_size):
        x_sample = x[i].clone().detach().requires_grad_(True)
        output = f(x_sample)
        
        if output.numel() > 1:
            raise ValueError("The function output must be scalar for Hessian computation.")
            
        first_grads = torch.autograd.grad(output,x_sample,create_graph=True)[0]
        
        hessian = torch.zeros(input_dim,input_dim,dtype=x_sample.dtype,device=x_sample.device)
        
        for j in range(input_dim):
            second_grads = torch.autograd.grad(first_grads[j],x_sample,create_graph=True)[0]
            hessian[j,:] = second_grads.view(-1)
            
        hessians.append(hessian)
        
    return torch.stack(hessians)
    

def gradient_check(f, grad_func, x, epsilon=1e-6):
    """
    Check the gradient of f numerically against the provided gradient function.

    Parameters
    ----------
    f : function
        The original scalar function f: R^n -> R.
    grad_func : function
        The gradient function of f.
    x : np.array
        The point at which to check the gradient (vector in R^n).
    epsilon : float, optional
        The step size for finite differences. Default is 1e-6.

    Returns
    -------
    bool
        True if the gradient matches numerically within a tolerance, False otherwise.
    """
    
    if not torch.is_tensor(x):
        raise ValueError("Input x must be a torch.Tensor.")
    
    batch_size,input_dim = x.shape
    numerical_grad = torch.zeros_like(x)
    for i in range(batch_size):
        for j in range(input_dim):
            x_forward = x[i].clone().detach()
            x_backward = x[i].clone().detach()
            x_forward[j] += epsilon
            x_backward[i] -= epsilon
            
            f_forward = f(x_forward)
            f_backward = f(x_backward)
            
            if f_forward.numel()>1 or f_backward.numel()>1:
                raise ValueError("The function output must be scalar for gradient computation.")
                
            numerical_grad[i,j] = (f_forward-f_backward)/(2*epsilon)
            
    analytical_grad = grad_func(f,x)
    if torch.allclose(numerical_grad,analytical_grad,atol=1e-5):
        print("Gradient check passed")
        return True
    else:
        print("Gradient check failed")
        print("Numerical Gradient:\n", numerical_grad)
        print("Analytical Gradient:\n", analytical_grad)
        return False
                
    

    
#def f_try(x):
    
#    return np.linalg.norm(x,ord=2)**2

# Test point in R^2
#x = np.array([1, 2, 3, 0, 5, 0])
#x = [1,2]
#print(f_try(x))
    
# Compute gradient and Hessian
#grad = gradient(f_try, x)
#print(grad)
#hess = hessian(f_try, x)
#print(hess)


# Perform gradient check
#def grad_func(f, x):
#    return gradient(f, x)

# Perform gradient check
#gradient_check(f_try, grad_func, x)
