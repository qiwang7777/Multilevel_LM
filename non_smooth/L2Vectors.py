import torch
import numpy as np
#L2TVprimal for NN
class L2TVPrimal:
    def __init__(self, var):
        self.var = var
    @torch.no_grad()
    def dot(self, x, y):
        ans = 0
        for k, v in x.td.items():
            ans += torch.sum(torch.mul(v, y.td[k]))
        return ans.item()
    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x, y)
    @torch.no_grad()
    def norm(self, x):
        return np.sqrt(self.dot(x, x))
    @torch.no_grad()
    def dual(self, x):
        return x
#L2TVDual for NN
class L2TVDual:
    def __init__(self, var):
        self.var = var
    @torch.no_grad()
    def dot(self, x, y):
        ans = 0
        for k, v in x.td.items():
            ans += torch.sum(torch.mul(v, y.td[k]))
        return ans.item()
    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x,y)
    @torch.no_grad()
    def norm(self, x):
        return np.sqrt(self.dot(x, x))
    @torch.no_grad()
    def dual(self, x):
        return x

class L2vectorPrimal:
    def __init__(self, var):
        self.var = var

    def dot(self, x, y):
        return x.T @ (self.var['Rlump'] * y)

    def apply(self, x, y):
        return x.T @ y

    def norm(self, x):
        return np.sqrt(self.dot(x, x))

    def dual(self, x):
        return self.var['Rlump'] * x

class L2vectorDual:
    def __init__(self, var):
        self.var = var

    def dot(self, x, y):
        return x.T @ (y / self.var['Rlump'])

    def apply(self, x, y):
        return x.T @ y

    def norm(self, x):
        return np.sqrt(self.dot(x, x))

    def dual(self, x):
        return x / self.var['Rlump']

class Euclidean:
    def __init__(self, var):
        self.var = var

    def dot(self, x, y):
        return x.T @ y

    def apply(self, x, y):
        return x.T @ y

    def norm(self, x):
        return np.sqrt(self.dot(x, x))

    def dual(self, x):
        return x
