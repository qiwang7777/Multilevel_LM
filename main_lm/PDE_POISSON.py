#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:57:06 2024

@author: wang
"""
import numpy as np
import torch
from scipy.sparse import diags, kron, eye

class PDE:
    def __init__(self, domain, real_solution=None):
        self.domain = domain
        self.real_solution = real_solution

    def laplacian_1d(self, grid_points):
        n = len(grid_points)
        dx = grid_points[1] - grid_points[0]
        diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)]
        L = diags(diagonals, [0, -1, 1], shape=(n, n), format='csr')
        L /= dx**2
        return L




    def laplacian_2d(self,x_grid, y_grid):
        """
        Create a 2D Laplacian matrix for the given x and y grids.
    
        Parameters:
        x_grid (np.ndarray): 1D array representing the x-coordinates of the grid.
        y_grid (np.ndarray): 1D array representing the y-coordinates of the grid.
    
        Returns:
        scipy.sparse.csr_matrix: The Laplacian matrix for the given grid.
        """
        n_x = x_grid.size
        n_y = y_grid.size
    
        # Compute grid spacing (assume uniform grid)
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
    
        # Create 1D Laplacian matrices for x and y directions
        D_x = diags([1, -2, 1], [1, 0, -1], shape=(n_x, n_x)) / dx**2
        D_y = diags([1, -2, 1], [1, 0, -1], shape=(n_y, n_y)) / dy**2
        I_x = eye(n_x)
        I_y = eye(n_y)
    
        # Create the 2D Laplacian using Kronecker product
        L = kron(I_y, D_x) + kron(D_y, I_x)
    
        return L






    def compute_source_term_DD(self, grid_points, real_solution):
        if isinstance(grid_points, tuple):  # 2D case
            x_grid, y_grid = grid_points
            X,Y = np.meshgrid(x_grid,y_grid)
            L = self.laplacian_2d(x_grid, y_grid)
            u = real_solution(X, Y).ravel()  # Flatten 2D array to 1D
            if L.shape[0] != u.shape[0]:
                raise ValueError("Dimension mismatch between Laplacian and real solution.")
            source_term = L.dot(u)
        else:  # 1D case
            L = self.laplacian_1d(grid_points)
            u = real_solution(grid_points)
            if L.shape[0] != u.shape[0]:
                raise ValueError("Dimension mismatch between Laplacian and real solution.")
            source_term = L.dot(u)
        
        return torch.tensor(source_term)
    
    def compute_source_term(self, grid_points, solution):
        # This should be a PyTorch tensor operation, assume solution is also a torch function
        return torch.tensor([solution(x).item() for x in grid_points], dtype=torch.float32) 