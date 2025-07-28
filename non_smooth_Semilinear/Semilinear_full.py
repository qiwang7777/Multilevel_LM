import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import scipy.sparse as sp
from scipy.sparse import spdiags, diags, lil_matrix,block_diag, kron, eye
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve
import time
from non_smooth.checks import deriv_check, deriv_check_simopt, vector_check
from non_smooth.setDefaultParameters import set_default_parameters

#from non_smooth.L1norm import L1Norm
from non_smooth.trustregion import trustregion
import matplotlib.pyplot as plt

class SemilinearSetup2D:
    def __init__(self, n, alpha, beta):
        if n <= 1:
            raise ValueError("Number of cells must be greater than 1")
        if alpha <= 0:
            raise ValueError("Control penalty parameter (alpha) must be positive")
        
        # Grid parameters
        self.h = 1/(n+1)
        self.n = n
        self.N = n * n  # Total DOFs per variable (state or control)
        
        # Stiffness matrix (A)
        e = np.ones(n)
        T = sp.diags([-e, 2*e, -e], [-1, 0, 1], (n,n))
        I = sp.eye(n)
        self.A = (sp.kron(I,T) + sp.kron(T,I)) / self.h**2
        
        # Mass matrix (M) - size N×N
        self.M = (self.h**2) * sp.eye(self.N)
        
        
        
        M_diag = np.array(self.M.sum(axis=1)).flatten()
        
        self.Rlump = np.concatenate([M_diag,M_diag])
        print("debug",self.Rlump.shape)
        # Target state yd (2D implementation)
        x = np.linspace(0, 1, n+2)[1:-1]  # Interior points
        X1, X2 = np.meshgrid(x, x)
        yd = np.where(X1 <= 0.5,
                     ((X1 - 0.5)**4 + 0.5*(X1 - 0.5)**3) * np.sin(np.pi*X2),
                     0.0)
        self.yd = yd.flatten()
        
        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.mesh = (X1, X2)
    

    def returnVars(self, useEuclidean):
        return {
            'n': self.n,
            'h': self.h,
            'N': self.N,
            'alpha': self.alpha,
            'beta': self.beta,
            'A': self.A,
            'M': self.M,
            'R': self.M,  # Using mass matrix as default
            'B': -sp.eye(self.N),
            'ud': self.yd,
            'useEuclidean': useEuclidean,
            'mesh': self.mesh,
            'Rlump': self.Rlump
        }



class SemilinearConstraintSolver2D:
    def __init__(self, var):
        self.var = var
        self.uprev = np.zeros(var['N'])  # Previous state solution
        
    def begin_counter(self, iter, cnt0):
        return cnt0
    
    def end_counter(self, iter, cnt0):
        return cnt0
        
    def solve(self, x, stol=1e-12):
        """Solve the PDE constraint for given full x=[y,z] vector"""
        y = x[:self.var['N']]  # Current state guess
        z = x[self.var['N']:]  # Control variables
        
        # Newton iteration to solve A*y + y^3 = z
        F = self.var['A'] @ y + y**3 - z
        cnt = 0
        fnorm = np.linalg.norm(F)
        ftol = max(stol, stol*fnorm)
        
        for _ in range(100):
            # Jacobian J = A + diag(3*y^2)
            J = self.var['A'] + sp.diags(3*y**2, 0, shape=(self.var['N'], self.var['N']))
            
            # Newton update
            dy = sp.linalg.spsolve(J, -F)
            y += dy
            
            # Update residual
            F = self.var['A'] @ y + y**3 - z
            fnew = np.linalg.norm(F)
            cnt += 1
            
            if fnew < ftol:
                break
                
        self.uprev = y
        return np.concatenate([y, z]), cnt, fnew
    
    def reset(self):
        self.uprev = np.zeros(self.var['N'])
    
    def value(self, x, vtol=1e-6):
        """Constraint evaluation F(y,z) = A*y + y^3 - z"""
        y = x[:self.var['N']]
        z = x[self.var['N']:]
        return self.var['A'] @ y + y**3 - z, 0
    
    def apply_jacobian(self, v, x, gtol=1e-6):
        """Apply Jacobian [J1, J2] where J1 = A + 3*diag(y^2), J2 = -I"""
        y = x[:self.var['N']]
        v_y = v[:self.var['N']]
        v_z = v[self.var['N']:]
        
        J1 = self.var['A'] + 3*sp.diags(y**2, 0, shape=(self.var['N'], self.var['N']))
        out_y = J1 @ v_y - v_z
        return out_y, 0
    
    def apply_adjoint_jacobian(self, v, x, gtol=1e-6):
        """Apply adjoint Jacobian [J1^T; J2^T]"""
        y = x[:self.var['N']]
        J1 = self.var['A'] + 3*sp.diags(y**2, 0, shape=(self.var['N'], self.var['N']))
        
        out_y = J1.T @ v
        out_z = -v
        return np.concatenate([out_y, out_z]), 0
    
    def apply_hessian(self, u, v, x, htol=1e-6):
        """Apply Hessian-vector product"""
        y = x[:self.var['N']]
        u_y = u[:self.var['N']]
        v_y = v[:self.var['N']]
        
        # Only the y-y block is nonzero
        out_y = 6 * (y * u_y * v_y)
        out_z = np.zeros(self.var['N'])
        return np.concatenate([out_y, out_z]), 0




class SemilinearObjective2D:
    def __init__(self, var):
        self.var = var
        
    def update(self,x,type):
        return None        
    

        
    def value(self, x, ftol=1e-6):
        y = x[:self.var['N']]  # State portion
        z = x[self.var['N']:]  # Control portion
        diff = y - self.var['ud']
        
        # Ensure proper matrix-vector multiplication
        term1 = 0.5 * diff.T @ (self.var['M'] @ diff)
        term2 = 0.5 * self.var['alpha'] * (z.T @ z) * ((self.var['h'])**2)  # Simplified for diagonal M
        
        return term1 + term2, 0
    
    def gradient(self, x, gtol=1e-6):
        y = x[:self.var['N']]
        z = x[self.var['N']:]
        grad_y = self.var['M'] @ (y - self.var['ud'])
        grad_z = self.var['alpha'] * z * ((self.var['h'])**2)  # Simplified for diagonal M
        return np.concatenate([grad_y, grad_z]), 0
    
    def hessVec(self, v, x, htol=1e-6):
        v_y = v[:self.var['N']]
        v_z = v[self.var['N']:]
        hv_y = self.var['M'] @ v_y
        hv_z = self.var['alpha'] * v_z * ((self.var['h'])**2)
        return np.concatenate([hv_y, hv_z]), 0 


class L1Norm:
    def __init__(self,var):
        self.var = var
        
    def value(self,x):
        
        z = x[self.var['N']:]
        return self.var['beta']*np.dot(self.var['Rlump'][self.var['N']:],np.abs(z))
    
    def prox(self,x,t):
        y = x[:self.var['N']]
        z = x[self.var['N']:]
        if self.var['useEuclidean']:
            prox_z = np.maximum(0,np.abs(z)-t*self.var['Rlump']*self.var['beta'])*np.sign(z)
        else:
            prox_z = np.maximum(0,np.abs(z)-t*self.var['beta'])*np.sign(z)
            
        return np.hstack([y,prox_z])
    
    def dir_deriv(self,s,x):
        sz = s[self.var['N']:]
        z = x[self.var['N']:]
        sz_sign = np.sign(z)
        return self.var['beta']*(np.dot(sz_sign,sz)+np.dot((1-np.abs(sz_sign)),np.abs(sz)))
    
    def project_sub_diff(self,g,x):
        z = x[self.var['N']:]
        g_control = g[self.var['N']:]
        sz_sign = np.sign(z)
        projected = self.var['beta']*sz_sign +(1-np.abs(sz_sign))*np.clip(g_control,-self.var['beta'])
        return np.hstack([np.zeros(self.var['N']),projected])
    
    def gen_jac_prox(self,x,t):
        z = x[self.var['N']:]
        px = self.prox(x,t)[self.var['N']:]
        d = np.ones_like(z)
        ind = px==0
        d[ind] = 0
        return np.diag(d),ind
    
    def apply_prox_jacobian(self,v,x,t):
        vz = v[self.var['N']:]
        if self.var['useEuclidean']:
            ind = np.abs(x[self.var['N']:]) <= t*self.var['Rlump'][self.var['N']:]*self.var['beta']
        else:
            ind = np.abs(x[self.var['N']:]) <= t*self.var['beta']
            
        Dv = vz.copy()
        Dv[ind] = 0
        return np.hstack([v[:self.var['N']],Dv])
    
    def get_parameter(self):
        return self.var['beta']
        
        
class L2vectorPrimal:
    def __init__(self,var):
        self.var = var
        
    def dot(self,x,y):
        Ny = self.var['N']
        x1, x2 = x[:Ny], x[Ny:]
        y1, y2 = y[:Ny], y[Ny:]
        return x1.T @(self.var['Rlump'][:Ny]*y1)+x2.T@y2
    
    def apply(self,x,y):
        return x.T@y
    
    def norm(self,x):
        return np.sqrt(self.dot(x,x))
    
    def dual(self,x):
        Ny = self.var['N']
        x1,x2 = x[:Ny], x[Ny:]
        dual1 = self.var['Rlump'][:Ny]*x1
        dual2 = x2
        return np.hstack([dual1,dual2])
    
class L2vectorDual:
    def __init__(self,var):
        self.var = var
        
    def dot(self,x,y):
        Ny = self.var['N']
        x1,x2 = x[:Ny],x[Ny:]
        y1,y2 = y[:Ny],y[Ny:]
        return x1.T@(y1/self.var['Rlump'][:Ny])+x2.T@y2
    
    def apply(self,x,y):
        return x.T@y
    
    def norm(self,x):
        return np.sqrt(self.dot(x,x))
    
    def dual(self,x):
        Ny = self.var['N']
        x1,x2 = x[:Ny],x[Ny:]
        dual1 = x1/self.var['Rlump'][:Ny]
        dual2 = x2
        return np.hstack([dual1,dual2])
    
        
        
        
        
        
        
        
        
from non_smooth.Euclidean import Euclidean
  
class Problem:
    def __init__(self, var, R):
        self.var  = var
        if var['useEuclidean']:
            self.pvector   = Euclidean(var)
            self.dvector   = Euclidean(var)
        else:
            self.pvector   = L2vectorPrimal(var)
            self.dvector   = L2vectorDual(var)
        self.obj_smooth    = SemilinearObjective2D(var)
        self.obj_nonsmooth = L1Norm(var)
        self.R             = R 

def restriction_R_2d(m,n):
    """
    Create 2D restriction matrix for coupled y-u system
    
    m: coarse dimension
    n: fine dimension

    """
    
    r_1d = lil_matrix((m,n))
    ratio = n//m
    for i in range(m):
        r_1d[i,i*ratio:(i+1)*ratio] = 1/(np.sqrt(ratio))
       
        
    r_2d = kron(r_1d,r_1d).tocsc()
    
    R = lil_matrix((2*m*m,2*n*n))
    R[:m*m,:n*n] = r_2d
    R[m*m:,n*n:] = r_2d

    return R.tocsc()
    

import time       
        
def driver(savestats=True, name="semilinear_control_2d"):
    print("2D Driver started")
    np.random.seed(0)
    
    # Problem parameters
    n = 32  # 32x32 grid
    alpha = 1e-4
    beta = 1e-2
    #meshlist = [n]
    meshlist = [n,n//2]
    problems = []
    for i in range(len(meshlist)):
        S = SemilinearSetup2D(meshlist[i],alpha,beta)
        var = S.returnVars(False)
        
        if i < len(meshlist)-1:
            R = restriction_R_2d(meshlist[i+1],meshlist[i])
        else:
            R = sp.eye(2*meshlist[i]*meshlist[i],format='csc')
        
    #Verify dimensions
        assert R.shape == (2*meshlist[i+1]**2,2*meshlist[i]**2) if i <len(meshlist)-1 else (2*meshlist[i]**2,2*meshlist[i]**2)
    
        p = Problem(var,R)
        p.obj_smooth = SemilinearObjective2D(var)
        p.obj_nonsmooth = L1Norm(var)
        p.con_solver = SemilinearConstraintSolver2D(var)
        problems.append(p)
        
    dim = 2*n*n
    x0 = np.ones(dim)
   

    params = set_default_parameters("SPG2")
    params.update({
        "reltol": False,
        "t": 2/alpha,
        "ocScale":1/alpha,
        "maxiter":100,
        "verbose":True,
        "useReduced":False
        })
    # Solve optimization problem
    start_time = time.time()
    x_opt, cnt_tr = trustregion(0, x0, params['delta'], problems, params)
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")

    # Extract results
    var = problems[0].obj_nonsmooth.var
    optimal_state = x_opt[:var['N']].reshape(var['n'], var['n'])
    optimal_control = x_opt[var['N']:].reshape(var['n'], var['n'])
    X1, X2 = var['mesh']

    # Plot results
    plt.figure(figsize=(15,5))
   
    plt.subplot(1,3,1)
    plt.contourf(X1, X2, var['ud'].reshape(var['n'],var['n']), levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("Target State $y_d$")
   
    plt.subplot(1,3,2)
    plt.contourf(X1, X2, optimal_state, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("Optimal State $y$")
   
    plt.subplot(1,3,3)
    plt.contourf(X1, X2, optimal_control, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("Optimal Control $z$")
   
    plt.tight_layout()
    plt.show()

    return x_opt, cnt_tr

    
    
cnt = driver()

