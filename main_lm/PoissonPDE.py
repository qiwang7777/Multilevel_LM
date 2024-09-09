#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file, we gave the PoissonPDE class, which is aimed to calculate 
the source term.
i.e. once we know the real solution u(x), (x could be in 1d or 2d), 
we want to get the f = -\laplacian u(x).

"""
import torch
import numpy as np

class PoissonPDE:
    def __init__(self, real_solution, x):
        """
        Initialize the PoissonPDE class.

        Parameters:
        - real_solution: Function that provides the true solution of the PDE.
        - dim: Dimensionality of the input (1 or 2).
        """
        self.real_solution = real_solution
        self.x = x

    def compute_source_term(self, x):
        """
        Compute the source term for the PDE given the input.

        Parameters:
        - x: Input tensor (1D or 2D grid)

        Returns:
        - source_term: Computed source term
        """
        if self.x.size(1) == 1:
            return self._compute_1d_source_term(x)
        elif self.x.size(1) == 2:
            return self._compute_2d_source_term(x)
        else:
            raise ValueError("Unsupported dimensionality")

    def _compute_1d_source_term(self, x):
        """
        Compute the source term for 1D PDE.

        Parameters:
        - x: 1D input tensor

        Returns:
        - source_term: Computed source term for 1D
        """
        x.requires_grad_(True)
        output = self.real_solution(x)
        grad1 = torch.autograd.grad(outputs=output, inputs=x,
                                    grad_outputs=torch.ones_like(output),
                                    create_graph=True)[0]
        grad2 = torch.autograd.grad(outputs=grad1, inputs=x,
                                    grad_outputs=torch.ones_like(grad1),
                                    create_graph=True)[0]
        source_term = -grad2
        return source_term

    def _compute_2d_source_term(self, x):
        """
        Compute the source term for 2D PDE.

        Parameters:
        - x: 2D input tensor

        Returns:
        - source_term: Computed source term for 2D
        """
        x.requires_grad_(True)
        output = self.real_solution(x)
        grad1 = torch.autograd.grad(outputs=output, inputs=x,
                                    grad_outputs=torch.ones_like(output),
                                    create_graph=True)[0]
        grad2 = []
        for i in range(2):
            grad2_i = torch.autograd.grad(outputs=grad1[:, i], inputs=x,
                                          grad_outputs=torch.ones_like(grad1[:, i]),
                                          create_graph=True)[0]
            grad2.append(grad2_i)
        grad2 = torch.stack(grad2, dim=-1)
        source_term = -torch.sum(grad2, dim=-1)  # Sum over spatial dimensions
        return source_term










#from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN,nn_xx
#input_dim = 1
#output_dim = 1
#n_hidden_layers = 1
#r_nodes_per_layer = 2
#model_21 = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim)
# Example usage for 1D case
#def u1d(x):
#    return model_21(x)

#x_1d = torch.tensor(np.linspace(0,1,5).reshape(5,1), dtype=torch.float32)
#pde_1d = PoissonPDE(u1d, x=x_1d)
#f_1d = pde_1d._compute_1d_source_term(x_1d)
#print(f"1D Source Term:\n{f_1d}")
#print(nn_xx(model_21,x_1d))
# Example usage for 2D case
#input_dim_2d = 2
#output_dim = 1
#n_hidden_layers = 1
#r_nodes_per_layer = 2
#model_21_2d = FullyConnectedNN(input_dim_2d, n_hidden_layers, r_nodes_per_layer, output_dim)
#def u2d(x):
#    return model_21_2d(x)

#x = np.linspace(0, 1, 5)
#y = np.linspace(0, 1, 5)

# Create a 2D grid
#X, Y = np.meshgrid(x, y)

# Flatten the grid and stack x and y coordinates
#input_data = np.stack([X.flatten(), Y.flatten()], axis=1)

# Convert to PyTorch tensor
#x_2d = torch.tensor(input_data, dtype=torch.float32)


#pde_2d = PoissonPDE(u2d, x=x_2d)
#f_2d = pde_2d._compute_2d_source_term(x_2d)
#print(f"2D Source Term:\n{f_2d}")
#print(nn_xx(model_21_2d,x_2d))

