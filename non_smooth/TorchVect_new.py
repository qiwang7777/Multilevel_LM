import os
import sys
sys.path.append(os.path.abspath("/Users/wang/Multilevel_LM/Multilevel_LM"))
import numpy as np

# Fully connected neural network construction
# Set up BB

#import torch

import collections
import time, torch
import numbers, copy
from collections import OrderedDict

class TorchVect:
    @torch.no_grad()
    def __init__(self, tensordict, isRop: bool = False, shapes_map=None):
        self.td = OrderedDict(tensordict)           # preserve order
        self.isRop = isRop
        self.shapes_map = dict(shapes_map) if shapes_map is not None else {}

        # Collect metadata once (handles scalars with shape == ())
        self.names  = list(self.td.keys())
        self.shapes = []
        self.sizes  = []
        for k, v in self.td.items():
            if not torch.is_tensor(v):
                raise TypeError(f"Param '{k}' is not a tensor.")
            self.shapes.append(self.shapes_map.get(k, v.shape))
            self.sizes.append(v.numel())

        self.numel     = int(sum(self.sizes))
        self.n_params  = len(self.names)

        self.sizes_per_param = list(self.sizes)
        self.shape = [self.numel]
        
    @torch.no_grad()
    def clone(self):
        td = {
            k: (v.clone().detach().requires_grad_(v.requires_grad) if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in self.td.items()
        }
    # IMPORTANT: do NOT zero here; it breaks any code that clones a gradient/vector/operator
        return TorchVect(td, isRop=self.isRop, shapes_map=self.shapes_map)
    def __deepcopy__(self, memo):
        """Support copy.deepcopy by using our clone method"""
        return self.clone()

    @property
    def T(self):
        Temp = self.clone()
        for k, v in list(self.td.items()):
            Temp.td[k] = v.T
        return Temp

    @torch.no_grad()
    def zero(self):
        for _, v in self.td.items():
            if torch.is_tensor(v):
                v.zero_()
    @torch.no_grad()
    def add_(self,other,alpha=1.0):
        if not isinstance(other,TorchVect):
            raise TypeError("add_: other must be a TorchVect")
        if self.isRop!= other.isRop:
            raise RuntimeError("add_: cannot add operator to parameter vector (isRop mismatch)")
            
        for k in self.td.keys():
            a = self.td[k]
            b = other.td[k]
            if not torch.is_tensor(a) or not torch.is_tensor(b):
                raise TypeError(f"add_: non-tensor entry for key'{k}'")
                
            if a.shape != b.shape:
                raise RuntimeError(f"add_: shape mismatch for '{k}':{a.shape} vs {b.shape}")
            self.td[k] = a.add(b.to(dtype=a.dtype,device=a.device),alpha=float(alpha))
        return self

    @torch.no_grad()
    def __add__(self, other):
        #temp = other.clone()
        #for k, v in self.td.items():
        #    temp.td[k] = other.td[k] + v
        #return temp
        out = self.clone()
        out.add_(other,alpha=1.0)
        return out

    @torch.no_grad()
    def __sub__(self, other):
        #temp = other.clone()
        #for k, v in self.td.items():
        #    temp.td[k] = other.td[k] - v
        #return -1 * temp
        out = self.clone()
        out.add_(other,alpha=-1.0)
        return out

    @torch.no_grad()
    def __mul__(self, alpha):
        ans = self.clone()
        for k, v in self.td.items():
            if not torch.is_tensor(v):
                raise TypeError(f"__mul__: non-tensor entry for key '{k}]")
            ans.td[k] = v*float(alpha)
        return ans

    @torch.no_grad()
    def __rmul__(self, alpha):
        return self.__mul__(alpha)

    @torch.no_grad()
    def __matmul__(self, x):
        """
        If self.isRop == True and x is a TorchVect of parameters:
            For each param k:
              y_k = R_k @ vec(x_k)
              reshape y_k to self.shapes_map[k]
        """
        assert isinstance(x, TorchVect), "Right operand must be TorchVect"
        if not self.isRop:
            raise RuntimeError("Left operand must be an operator (isRop=True).")
        #ans = x.clone()

        if x.isRop:
            ans = TorchVect(OrderedDict(), isRop=True, shapes_map={})
            for k in self.td.keys():
                L = self.td[k]                # (m x p)
                
                R = x.td[k]                   # (p x q)
                #print("L_shape:",L.shape,"R_shape:",R.shape)
                if L.shape[1] != R.shape[0]:
                    raise RuntimeError(f"Operator compose mismatch for '{k}': "
                                       f"L cols {L.shape[1]} != R rows {R.shape[0]}")
                ans.td[k] = L @ R             # (m x q)
                # output shape stays that of the LEFT operator
                ans.shapes_map[k] = self.shapes_map.get(k, x.shapes_map.get(k, None))
            return ans

        # Case 2: apply operator to parameter vector (matrix–vector per parameter)
        ans = x.clone()
        for k, v in x.td.items():
            Rk = self.td[k]                  # (out_size x in_size)
            vec = v.reshape(-1)              # (in_size,)
            if Rk.shape[1] != vec.numel():
                raise RuntimeError(f"Op '{k}' expects input {Rk.shape[1]}, got {vec.numel()}")
            y = Rk.to(vec.dtype).to(vec.device) @ vec
            out_shape = self.shapes_map.get(k, v.shape)
            if int(torch.tensor(out_shape).prod().item()) != y.numel():
                raise RuntimeError(f"Op '{k}' output size {y.numel()} cannot reshape to {out_shape}")
            ans.td[k] = y.reshape(out_shape)
        ans.isRop = False
        return ans

    @torch.no_grad()
    def __truediv__(self, alpha):
        ans = self.clone()
        inv = 1.0/float(alpha)
        for k, v in ans.td.items():
            if not torch.is_tensor(v):
                raise TypeError(f"__truediv__: non-tensor entry for key '{k}'")
            ans.td[k] = v*inv
        return ans

    @torch.no_grad()
    def __rtruediv__(self, alpha):
        return self.__truediv__(alpha)
