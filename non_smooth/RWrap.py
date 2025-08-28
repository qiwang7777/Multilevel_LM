import os
import sys
sys.path.append(os.path.abspath('..'))
from collections import OrderedDict
from .TorchVector_new import TorchVect




class RWrap:
    """
    Per-parameter linear operator wrapper.
    td[name]: matrix of shape (out_numel, in_numel)
    shapes_map is the OUTPUT reshape map used by TorchVect.__matmul__.
    We also keep an input map and swap them on transpose.
    """
    def __init__(self, td, out_shapes_map, in_shapes_map):
        self.td = dict(td)
        self._out_shapes_map = dict(out_shapes_map)  # shape to reshape R @ x into
        self._in_shapes_map  = dict(in_shapes_map)   # shape of x
        self.shapes_map = self._out_shapes_map       # TorchVect reads this
        self.isRop = True
        
    @property
    def inner(self):
        return TorchVect(OrderedDict(self.td), isRop=True, shapes_map=self.shapes_map)

    def __matmul__(self,other):
        #RWrap @ TorchVect -> apply operator
        #RWrap @ RWrap -> compose operators
        if isinstance(other, RWrap):
            return self.inner @ other.inner
        elif isinstance(other,TorchVect):
            return self.inner @ other
        else:
            return NotImplemented
        
    def __rmatmul__(self, other):
        """Support TorchVect @ RWrap if your code ever does that."""
        if isinstance(other, TorchVect):
            # (x @ R) is unusual; if you need it, implement as (R.T @ x) with care.
            return NotImplemented
        return NotImplemented


    @property
    def T(self):
        Rt = RWrap.__new__(RWrap)
        Rt.td = {k: v.T.contiguous() for k, v in self.td.items()}
        # swap in/out maps on transpose
        Rt._out_shapes_map = self._in_shapes_map
        Rt._in_shapes_map  = self._out_shapes_map
        Rt.shapes_map = Rt._out_shapes_map
        Rt.isRop = True
        return Rt
    
    @property
    def shape(self):
        rows = sum(W.shape[0] for W in self.td.values())
        cols = sum(W.shape[1] for W in self.td.values())
        return [rows,cols]
