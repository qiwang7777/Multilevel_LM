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
class MyFuncObjective(TorchObjective):

    def torch_value(self, x):
        # Assuming x is a torch tensor containing [x, y]
        z, y = x[0], x[1]
        return z**2 + 2*z*y + y**2


# Test Example
x = torch.tensor([1.0, 2.0], requires_grad=True)
v = torch.tensor([1.0, 0.0])
hv = torch.tensor([0.0, 0.0])

obj = MyFuncObjective()
tolerance = 1e-6

# Calculate value
value = obj.value(x, tolerance)
print(f"Objective Value: {value}")

# Calculate gradient
g = torch.zeros_like(x)  # Initialize g with the same shape as x
obj.gradient(g, x, tolerance)
print(f"Gradient: {g}")

# Calculate Hessian-vector product
obj.hessVec(hv, v, x, tolerance)
print(f"Hessian-Vector Product: {hv}")

