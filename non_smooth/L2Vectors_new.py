import numpy as np
import torch

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
