#Reduced Objective Function

class ReducedObjective:
    def __init__(self, obj0, con0):
        self.obj0 = obj0
        self.con0 = con0
        self.is_state_computed = False
        self.is_state_cached = False
        self.is_adjoint_computed = False
        self.is_adjoint_cached = False
        self.uwork = None
        self.ucache = None
        self.pwork = None
        self.pcache = None
        self.cnt = {
            "nstate": 0, "nadjoint": 0, "nstatesens": 0,
            "nadjointsens": 0, "ninvJ1": 0
        }
    
    def begin_counter(self, iter, cnt0):
        if iter == 0:
            cnt0.update({
                "nstatehist": [], "nadjoihist": [], "nstsenhist": [],
                "nadsenhist": [], "ninvJ1hist": []
            })
            for key in self.cnt:
                self.cnt[key] = 0
        return self.con0.begin_counter(iter, cnt0)
    
    def end_counter(self, iter, cnt0):
        cnt0["nstatehist"].append(self.cnt["nstate"])
        cnt0["nadjoihist"].append(self.cnt["nadjoint"])
        cnt0["nstsenhist"].append(self.cnt["nstatesens"])
        cnt0["nadsenhist"].append(self.cnt["nadjointsens"])
        cnt0["ninvJ1hist"].append(self.cnt["ninvJ1"])
        return self.con0.end_counter(iter, cnt0)
    
    def reset(self):
        self.uwork = None
        self.ucache = None
        self.is_state_computed = False
        self.is_state_cached = False
        self.pwork = None
        self.pcache = None
        self.is_adjoint_computed = False
        self.is_adjoint_cached = False
    
    def update(self, x, type):
        if type == "init":
            self.is_state_computed = False
            self.is_state_cached = False
            self.is_adjoint_computed = False
            self.is_adjoint_cached = False
        elif type == "trial":
            self.is_state_cached = self.is_state_computed
            self.is_adjoint_cached = self.is_adjoint_computed
            self.is_state_computed = False
            self.is_adjoint_computed = False
        elif type == "reject":
            if self.is_state_cached:
                self.uwork = self.ucache
                self.is_state_computed = True
            else:
                self.is_state_computed = False
            if self.is_adjoint_cached:
                self.pwork = self.pcache
                self.is_adjoint_computed = True
            else:
                self.is_adjoint_computed = False
        elif type == "accept":
            if self.is_state_computed:
                self.ucache = self.uwork
            if self.is_adjoint_computed:
                self.pcache = self.pwork
        elif type == "temp":
            self.is_state_computed = False
            self.is_adjoint_computed = False
    
    def value(self, z, ftol):
        ferr = 0
        if not self.is_state_computed or self.uwork is None:
            self.uwork, cnt0, serr = self.con0.solve(z, ftol)
            self.cnt["ninvJ1"] += cnt0
            self.is_state_computed = True
            self.cnt["nstate"] += 1
            ferr = max(ferr, serr)
        val, verr = self.obj0.value([self.uwork, z], ftol)
        return val, max(ferr, verr)
    
    def gradient(self, z, gtol):
        gerr = 0
        if not self.is_state_computed or self.uwork is None:
            self.uwork, cnt0, serr = self.con0.solve(z, gtol)
            self.cnt["ninvJ1"] += cnt0
            self.is_state_computed = True
            self.cnt["nstate"] += 1
            gerr = max(gerr, serr)
        if not self.is_adjoint_computed or self.pwork is None:
            rhs, aerr1 = self.obj0.gradient_1([self.uwork, z], gtol)
            rhs = -rhs
            print([self.uwork,z][0].shape)
            self.pwork, aerr2 = self.con0.apply_inverse_adjoint_jacobian_1(rhs, [self.uwork, z][0], gtol)
            self.cnt["ninvJ1"] += 1
            self.is_adjoint_computed = True
            self.cnt["nadjoint"] += 1
            gerr = max(gerr, aerr1, aerr2)
        Bp, jerr = self.con0.apply_adjoint_jacobian_2(self.pwork, [self.uwork, z], gtol)
        grad, gerr1 = self.obj0.gradient_2([self.uwork, z], gtol)
        return grad + Bp, max(gerr, jerr, gerr1)
    
    def profile(self):
        print("\nProfile Reduced Objective")
        print("  #state    #adjoint    #statesens    #adjointsens    #linearsolves")
        print("  {:6d}      {:6d}        {:6d}          {:6d}           {:6d}".format(
            self.cnt["nstate"], self.cnt["nadjoint"], self.cnt["nstatesens"],
            self.cnt["nadjointsens"], self.cnt["ninvJ1"]
        ))
        self.cnt["con"] = self.con0.profile()
        return self.cnt
    


# Objective function
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
    
# Constraint

import scipy.sparse as sp


class ConstraintSolver:
    def __init__(self, var):
        
        self.var = var
        self.uprev = np.ones(self.var['nu'])
        
    def begin_counter(self,iter,cnt0):
        return cnt0
    
    def end_counter(self,iter,cnt0):
        return cnt0
        

    def solve(self, z, stol=1e-12):
        nu = self.var['nu']
        u = self.uprev
        c, _ = self.value(np.hstack([u,z]))
        cnt = 0 
        atol = stol
        rtol = 1
        cnorm = np.linalg.norm(c)
        ctol = min(atol, rtol * cnorm)
        
        for _ in range(100):
            s,_ = self.apply_inverse_jacobian_1(self.value(np.hstack([u, z]))[0], np.hstack([u, z]))
            
            unew = u - s
            cnew = self.value(np.hstack([unew, z]))[0]
            ctmp = np.linalg.norm(cnew)
            
            alpha = 1
            while ctmp > (1 - 1e-4 * alpha) * cnorm:
                alpha *= 0.1
                unew = u - alpha * s
                cnew = self.value(np.hstack([unew, z]))[0]
                ctmp = np.linalg.norm(cnew)
            
            u, c, cnorm = unew, cnew, ctmp
            cnt += 1
            if cnorm < ctol:
                break
        serr = cnorm
        
        self.uprev = u
        return u,cnt,serr

    def reset(self):
        self.uprev = np.ones(self.var['nu'])

    def value(self,x):
        nu = self.var['nu']
        A, B, b = self.var['A'], self.var['B'], self.var['b']
        u, z = x[:nu], x[nu:]
        Nu = self.evaluate_nonlinearity_value(u)
        #print("A_shape:", A.shape) #(n-1,n-1)
        #print("u_shape:",u.shape)  #(n-1,)
        #print("Nu_shape:", Nu.shape) #(n-1,)
        #print("B_shape:", B.shape) #(n-1,n)
        #print("z_shape:",z.shape) #(n,)
        #print("b_shape:",b.shape) #(n-1,)
        c = A @ u + Nu - (B @ z + b)
        return c, 0

    def apply_jacobian_1(self, v, x, gtol=1e-6):
        nu = self.var['nu']
        A = self.var['A']
        u = x[:nu]
        
        J = self.evaluate_nonlinearity_jacobian(u)
        return (A + J) @ v, 0

    def apply_jacobian_2(self, v):
        return -self.var.B @ v, 0

    def apply_adjoint_jacobian_1(self, v, x):
        nu = self.var['nu']
        A = self.var['A']
        u = x[:nu]
        J = self.evaluate_nonlinearity_jacobian(u)
        return (A + J).T @ v, 0

    def apply_adjoint_jacobian_2(self, v,x,gtol=1e-6):
        return -self.var['B'].T @ v, 0

    def apply_inverse_jacobian_1(self, v, x,gtol=1e-6):
        nu = self.var['nu']
        A = self.var['A']
        u = x[:nu]
        
        J = self.evaluate_nonlinearity_jacobian(u)
        
        solution = sp.linalg.spsolve(A+J,v)
        return solution, 0

    def apply_inverse_adjoint_jacobian_1(self, v, x,gtol=1e-6):
        nu = self.var['nu']
        A = self.var['A']
        u = x[:nu]
        J = self.evaluate_nonlinearity_jacobian(u)
        return sp.linalg.spsolve((A + J).T, v), 0

    def apply_adjoint_hessian_11(self, u, v, x):
        nu = self.var['nu']
        ahuv = np.zeros(nu)
        for i in range(nu):
            if i < nu - 1:
                ahuv[i] += (u[i] * v[i + 1] - u[i + 1] * (2 * v[i] + v[i + 1])) / 6
            if i > 0:
                ahuv[i] += (u[i - 1] * (v[i - 1] + 2 * v[i]) - u[i] * v[i - 1]) / 6
        return ahuv, 0

    def apply_adjoint_hessian_12(self, u, v, x):
        return np.zeros(self.var['nz']), 0

    def apply_adjoint_hessian_21(self, u, v, x):
        return np.zeros(self.var['nu']), 0

    def apply_adjoint_hessian_22(self, u, v, x):
        return np.zeros(self.var['nz']), 0

    def evaluate_nonlinearity_value(self, u):
        n = self.var['n']
        Nu = np.zeros(n - 1)
        Nu[:-1] += u[:-1] * u[1:] + u[1:] ** 2
        Nu[1:] -= u[:-1] * u[1:] + u[:-1] ** 2
        return Nu / 6

    def evaluate_nonlinearity_jacobian(self, u):
        n = self.var['n']
        if isinstance(u, list):
             print("u is a list")
        else:
             print("u is a Numpy array")
        
        
       
        d1, d2, d3 = np.zeros(n - 1), np.zeros(n - 1), np.zeros(n - 1)
       
        d1[:-1] = u[:-1] - u[1:]
        d2[0] = u[1]
        d2[1:-1] = u[2:]-u[:-2]
        d2[-1] =  -u[-2]
        d3[1:] = u[:-1] + 2 * u[1:]
        J = spdiags([d1, d2, d3], [-1, 0, 1], n-1, n-1) / 6
        
        return J

    def profile(self):
        return []
    
    
    
# Set up Burgers
from scipy.sparse import spdiags, diags,lil_matrix
from scipy.integrate import quad

class BurgersSetup:
    """
    
    Solve the Burgers, distributed control problem
    
    """
    def __init__(self, n, mu, alpha, beta, usepc=True):
        if n <= 1:
            raise ValueError("Number of cells (n) must be greater than 1.")
        if mu <= 0:
            raise ValueError("Viscosity (mu) must be positive.")
        if alpha <= 0:
            raise ValueError("Control penalty parameter (alpha) must be positive.")

        # Dirichlet conditions
        self.h = 1 / (n + 1)
        self.u0 = 0
        self.u1 = -1
        
        # Build stiffness matrix
        o = -np.ones(n - 1) / self.h
        d = 2 * np.ones(n - 1) / self.h
        A0 = lil_matrix((n-1,n-1))
        A0[0 , 0] = -self.u0 / 6
        A0 = A0.tocsc()
        A1 = lil_matrix((n-1,n-1))
        A1[n-2,n-2] = self.u1 / 6
        A1 = A1.tocsc()
        
        d0 = np.zeros(n - 1)
        d0[0] = -self.u0 * (self.u0 / 6 + mu / self.h)
        d1 = np.zeros(n - 1)
        d1[-1] = self.u1 * (self.u1 / 6 - mu / self.h)
        
        self.A = mu * spdiags([o, d, o], [-1, 0, 1], n - 1, n - 1).tocsc() + A0 + A1

        # Build state observation matrix
        o = (self.h / 6 - 1 / self.h) * np.ones(n - 1)
        d = (2 * self.h / 3 + 2 / self.h) * np.ones(n - 1)
        self.M = spdiags([o, d, o], [-1, 0, 1], n - 1, n - 1).tocsc()

        # Build control operator
        if usepc:
            self.B = spdiags([self.h / 2 * np.ones(n)] * 2, [0, 1], n - 1, n).tocsc()
        else:
            e0 = np.zeros(n - 1)
            e0[0] = self.h / 6
            e1 = np.zeros(n - 1)
            e1[n-2] = self.h / 6
            self.B = diags([e0, self.M.diagonal(), e1]).tocsc()
            #self.B = csc_matrix(np.column_stack((e0, self.M.toarray(), e1)))

        # Build control mass matrix
        if usepc:
            self.R = self.h * diags([1], [0], shape=(n, n)).tocsc()
        else:
            e0 = np.zeros(n + 1)
            e0[0] = 2 * self.h / 6
            e0[1] = self.h / 6
            e1 = np.zeros(n + 1)
            e1[n] = 2 * self.h / 6
            e1[n-1] = self.h / 6
            R = diags([e0, self.B.diagonal(), e1]).tocsc()
            self.R = diags(R.sum(axis=1).A.ravel(), [0], shape=(n + 1, n + 1)).tocsc()

        self.Rlump = np.array(self.R.sum(axis=1)).flatten()

        # Build the right-hand side for the PDE
        self.mesh = np.linspace(0, 1, n + 1)
        self.b = self.integrate_rhs(self.mesh, lambda x: 2 * (mu + x**3)) - d0 - d1

        # Target state
        self.ud = -self.mesh**2

        # Save parameters
        self.n = n
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.nu = n if usepc else n + 1
        self.nz = n if usepc else n + 1

    def integrate_rhs(self, x, f):
        nx = len(x)
        b = np.zeros(nx - 2)
        for i in range(nx - 2):
            x0, x1, x2 = x[i], x[i + 1], x[i + 2]

            def F1(x):
                return f(x) * ((x >= x0) & (x < x1)) * (x - x0) / (x1 - x0)

            def F2(x):
                return f(x) * ((x >= x1) & (x <= x2)) * (x2 - x) / (x2 - x1)

            b[i] = quad(F1, x0, x1)[0] + quad(F2, x1, x2)[0]

        return b
    
# Example usage
Burgers = BurgersSetup(n=512, mu=0.08, alpha=1e-4, beta=1e-2)
n = Burgers.n
nu = Burgers.n - 1
nz = Burgers.n 
alpha = Burgers.alpha
M = Burgers.M
R = Burgers.R
A = Burgers.A

B = Burgers.B
b = Burgers.b
ud = Burgers.ud
var = {
       'n':n,
       'nu':nu,
       'nz':nz,
       'alpha':alpha,
       'A':A,
       'M':M,
       'R':R,
       'B':B,
       'b':b,
       'ud':ud
       }
obj = Objective(var)
con = ConstraintSolver(var)
red_obj = ReducedObjective(obj, con)
z = np.random.rand(nz)
u = np.zeros(nu)
ftol = 1e-6
# Evaluate the reduced objective function value
val,err = red_obj.value(z, ftol)
print("Objective function value:", val)
print("Error estimate:",err)

#Compute the reduced gradient
gtol = 1e-6
grad,gerr = red_obj.gradient(z, gtol)
print("Gradient:",grad)
print("Gradient error estimate:", gerr)
#Profile the computation
red_obj.profile()
