#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:50:29 2024

@author: wang
"""

import sympy as sp
import numpy as np

def derivative_check(f, x_value):
    # Define the symbolic variable
    x = sp.symbols('x')
    
    # Convert the function to a symbolic expression if it's not already
    f_sympy = f(x) if isinstance(f(x), sp.Expr) else sp.sympify(f(x))
    
    # Calculate the symbolic derivative
    symbolic_derivative = sp.diff(f_sympy, x)
    
    # Calculate the symbolic derivative at x_value
    symbolic_derivative_at_x = symbolic_derivative.subs(x, x_value)
    
    # Define the numerical derivative using central difference
    def numerical_derivative(f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)
    
    # Calculate the numerical derivative at x_value
    numerical_derivative_at_x = numerical_derivative(f, x_value)
    
    # Output the results
    print(f"Original function: {f_sympy}")
    print(f"Symbolic derivative: {symbolic_derivative}")
    print(f"Symbolic derivative at x = {x_value}: {symbolic_derivative_at_x}")
    print(f"Numerical derivative at x = {x_value}: {numerical_derivative_at_x}")
    
    # Return both derivatives for further use if needed
    return symbolic_derivative_at_x, numerical_derivative_at_x

# Example usage:
# Define a function, for example, f(x) = x**3 + 2*x**2 + x + 1
def my_function(x):
    return x**3 + 2*x**2 + x + 1

# Check the derivative at x = 2
derivative_check(my_function, 2)
