import numpy as np
from scipy.sparse import csr_matrix


class Objective:
    def __init__(self, var):
        self.var = var

    # Compute objective function value
    def value(self, x, ftol):
        
        nu = self.var['nu']
        M = self.var['M']
        R = self.var['R']
        alpha = self.var['alpha']
        ud = self.var['ud']
        
        u = x[0] 
        z = x[1] 
        
        
        
        
        
        diffu = u - ud[1:-1]
        uMu = diffu.T @ (M @ diffu)
        zRz = z.T @ (R @ z)
        val = 0.5 * (uMu + alpha * zRz)
        ferr = 0
        
        return val, ferr
    
    # Compute objective function gradient (gradient_1)
    def gradient_1(self, x, gtol):
        nu = self.var['nu']
        M = self.var['M']
        ud = self.var['ud']
        u = x[0]
        diffu = u - ud[1:-1]
        g = M @ diffu
        gerr = 0
        
        return g, gerr
    
    # Compute objective function gradient (gradient_2)
    def gradient_2(self, x, gtol):
        nu = self.var['nu']
        R = self.var['R']
        alpha = self.var['alpha']
        z = x[1]
        g = alpha * (R @ z)
        gerr = 0
        
        return g, gerr
    
    # Apply objective function Hessian to a vector (hessVec_11)
    def hessVec_11(self, v, x, htol):
        M = self.var['M']
        hv = M @ v
        herr = 0
        
        return hv, herr
    
    # Apply objective function Hessian to a vector (hessVec_12)
    def hessVec_12(self, v, x, htol):
        nu = self.var['nu']
        hv = csr_matrix((nu, 1))
        herr = 0
        
        return hv, herr
    
    # Apply objective function Hessian to a vector (hessVec_21)
    def hessVec_21(self, v, x, htol):
        nz = self.var['nz']
        hv = csr_matrix((nz, 1))
        herr = 0
        
        return hv, herr
    
    # Apply objective function Hessian to a vector (hessVec_22)
    def hessVec_22(self, v, x, htol):
        R = self.var['R']
        alpha = self.var['alpha']
        hv = alpha * (R @ v)
        herr = 0
        
        return hv, herr
