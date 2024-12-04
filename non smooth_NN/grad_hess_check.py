import numpy as np
import sympy as sp

def gradient(f, x):
    n = len(x)
    symbols = sp.symbols(f'x:{n}')  # Create symbolic variables x0, x1, ..., x(n-1)

    # Wrap f to handle symbolic inputs
    def f_wrapper(*args):
        if isinstance(args[0], sp.Basic):  # Check if symbolic variables are passed
            return sum(arg**2 for arg in args)  # Equivalent to ||x||_2^2 symbolically
        else:
            return f(np.array(args))  # Use original f for numerical inputs

    # Convert the wrapped function to symbolic form
    f_symbolic = f_wrapper(*symbols)

    # Compute the symbolic gradient
    grad_symbolic = [sp.diff(f_symbolic, xi) for xi in symbols]

    # Convert the symbolic gradient to a numerical function
    grad_func = sp.lambdify(symbols, grad_symbolic, 'numpy')

    # Evaluate the gradient at the given point x
    grad = np.array(grad_func(*x), dtype=float)
    return grad


def hessian(f, x):
    n = len(x)
    symbols = sp.symbols(f'x:{n}')  # Create symbolic variables x0, x1, ..., x(n-1)

    # Wrap f to handle symbolic inputs
    def f_wrapper(*args):
        if isinstance(args[0], sp.Basic):  # Check if symbolic variables are passed
            return sum(arg**2 for arg in args)  # Equivalent to ||x||_2^2 symbolically
        else:
            return f(np.array(args))  # Use original f for numerical inputs

    # Convert the wrapped function to symbolic form
    f_symbolic = f_wrapper(*symbols)

    # Compute the gradient symbolically
    grad_symbolic = [sp.diff(f_symbolic, xi) for xi in symbols]

    # Compute the Hessian symbolically
    hessian_symbolic = [[sp.diff(gi, xj) for xj in symbols] for gi in grad_symbolic]

    # Convert the symbolic Hessian to a numerical function
    hess_func = sp.lambdify(symbols, hessian_symbolic, 'numpy')

    # Evaluate the Hessian at the given point x
    hess = np.array(hess_func(*x), dtype=float)
    return hess

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
    x = np.array(x, dtype=float)
    n = len(x)

    # Compute numerical gradient using finite differences
    numerical_grad = np.zeros(n)
    for i in range(n):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += epsilon
        x_backward[i] -= epsilon
        numerical_grad[i] = (f(x_forward) - f(x_backward)) / (2 * epsilon)

    # Compute analytical gradient
    analytical_grad = grad_func(f, x)

    # Compare numerical and analytical gradients
    if np.allclose(numerical_grad, analytical_grad, atol=1e-5):
        print("Gradient check passed")
        return True
    else:
        print("Gradient check failed")
        print("Numerical Gradient:", numerical_grad)
        print("Analytical Gradient:", analytical_grad)
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
