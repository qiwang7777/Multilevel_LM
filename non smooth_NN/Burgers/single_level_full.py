#Reduced Objective Function

import numpy as np

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
            'nstate': 0, 'nadjoint': 0, 'nstatesens': 0, 'nadjointsens': 0, 'ninvJ1': 0
        }
    
    def begin_counter(self, iter, cnt0):
        if iter == 0:
            cnt0.update({'nstatehist': [], 'nadjoihist': [], 'nstsenhist': [], 'nadsenhist': [], 'ninvJ1hist': []})
            for key in self.cnt:
                self.cnt[key] = 0
        return self.con0.begin_counter(iter, cnt0)

    def end_counter(self, iter, cnt0):
        cnt0['nstatehist'].append(self.cnt['nstate'])
        cnt0['nadjoihist'].append(self.cnt['nadjoint'])
        cnt0['nstsenhist'].append(self.cnt['nstatesens'])
        cnt0['nadsenhist'].append(self.cnt['nadjointsens'])
        cnt0['ninvJ1hist'].append(self.cnt['ninvJ1'])
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
        if type == 'init':
            self.is_state_computed = False
            self.is_state_cached = False
            self.is_adjoint_computed = False
            self.is_adjoint_cached = False
        elif type == 'trial':
            self.is_state_cached = self.is_state_computed
            self.is_adjoint_cached = self.is_adjoint_computed
            self.is_state_computed = False
            self.is_adjoint_computed = False
        elif type == 'reject':
            if self.is_state_cached:
                self.uwork = self.ucache
                self.is_state_computed = True
            if self.is_adjoint_cached:
                self.pwork = self.pcache
                self.is_adjoint_computed = True
        elif type == 'accept':
            if self.is_state_computed:
                self.ucache = self.uwork
            if self.is_adjoint_computed:
                self.pcache = self.pwork
        elif type == 'temp':
            self.is_state_computed = False
            self.is_adjoint_computed = False
    
    def value(self, z, ftol):
        ferr = 0
        
        if not self.is_state_computed or self.uwork is None:
            self.uwork, cnt0, serr = self.con0.solve(z, ftol)
            self.cnt['ninvJ1'] += cnt0
            self.is_state_computed = True
            self.cnt['nstate'] += 1
            ferr = max(ferr, serr)
        val, verr = self.obj0.value(np.hstack([self.uwork, z]), ftol)
        return val, max(ferr, verr)

    def gradient(self, z, gtol):
        if not self.is_state_computed or self.uwork is None:
            self.uwork, cnt0, serr = self.con0.solve(z, gtol)
            self.cnt['ninvJ1'] += cnt0
            self.is_state_computed = True
            self.cnt['nstate'] += 1
        if not self.is_adjoint_computed or self.pwork is None:
            rhs, aerr1 = self.obj0.gradient_1(np.hstack([self.uwork,z]), gtol)
            rhs = -rhs
            self.pwork, aerr2 = self.con0.apply_inverse_adjoint_jacobian_1(rhs, [self.uwork, z][0], gtol)
            self.cnt['ninvJ1'] += 1
            self.is_adjoint_computed = True
            self.cnt['nadjoint'] += 1
        Bp, jerr = self.con0.apply_adjoint_jacobian_2(self.pwork, [self.uwork, z], gtol)
        grad, gerr1 = self.obj0.gradient_2(np.hstack([self.uwork,z]), gtol)
        return grad + Bp, max(jerr, gerr1)
    
    def hessVec(self, v, z, htol):
        if not self.is_state_computed or self.uwork is None:
            self.uwork, cnt0, serr = self.con0.solve(z, htol)
            self.cnt['ninvJ1'] += cnt0
            self.is_state_computed = True
            self.cnt['nstate'] += 1
        if not self.is_adjoint_computed or self.pwork is None:
            rhs, aerr1 = self.obj0.gradient_1([self.uwork, z], htol)
            rhs = -rhs
            self.pwork, aerr2 = self.con0.apply_inverse_adjoint_jacobian_1(rhs, [self.uwork, z][0], htol)
            self.cnt['ninvJ1'] += 1
            self.is_adjoint_computed = True
            self.cnt['nadjoint'] += 1
        rhs, sserr1 = self.con0.apply_jacobian_2(v, [self.uwork, z], htol)
        rhs = -rhs
        w, sserr2 = self.con0.apply_inverse_jacobian_1(rhs, [self.uwork, z][0], htol)
        rhs, aserr1 = self.obj0.hessVec_11(w, [self.uwork, z], htol)
        q, aserr5 = self.con0.apply_inverse_adjoint_jacobian_1(rhs, [self.uwork, z][0], htol)
        q = -q
        self.cnt['ninvJ1'] += 1
        hv, herr1 = self.con0.apply_adjoint_jacobian_2(q, [self.uwork, z], htol)
        return hv, max(herr1, aserr1, aserr5)
    
    def profile(self):
        print("\nProfile Reduced Objective")
        print("  #state    #adjoint    #statesens    #adjointsens    #linearsolves")
        print(f"  {self.cnt['nstate']:6d}      {self.cnt['nadjoint']:6d}        {self.cnt['nstatesens']:6d}          {self.cnt['nadjointsens']:6d}           {self.cnt['ninvJ1']:6d}")
        cnt = self.cnt.copy()
        cnt['con'] = self.con0.profile()
        return cnt

    


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
        
        u = x[:nu] 
        z = x[nu:] 
        
        
        
        
        
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
        u = x[:nu][0]
        #print(u[0].shape)
        
        diffu = u - ud[1:-1]
        g = M @ diffu
        gerr = 0
        
        return g, gerr
    
    # Compute objective function gradient (gradient_2)
    def gradient_2(self, x, gtol):
        nu = self.var['nu']
        R = self.var['R']
        alpha = self.var['alpha']
        z = x[nu:]
        #print(R.shape)
        #print(x.shape)
        #print(z.shape)
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

    def value(self,x,vtol=1e-6):
        nu = self.var['nu']
        A, B, b = self.var['A'], self.var['B'], self.var['b']
        u, z = x[:nu], x[nu:]
        Nu = self.evaluate_nonlinearity_value(u)
        #print("x_shape:",x.shape)
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

    def apply_jacobian_2(self, v,x,gtol=1e-6):
        return -self.var['B'] @ v, 0

    def apply_adjoint_jacobian_1(self, v, x,gtol=1e-6):
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

    def apply_adjoint_hessian_11(self, u, v, x,htol=1e-6):
        nu = self.var['nu']
        ahuv = np.zeros(nu)
        for i in range(nu):
            if i < nu - 1:
                ahuv[i] += (u[i] * v[i + 1] - u[i + 1] * (2 * v[i] + v[i + 1])) / 6
            if i > 0:
                ahuv[i] += (u[i - 1] * (v[i - 1] + 2 * v[i]) - u[i] * v[i - 1]) / 6
        return ahuv, 0

    def apply_adjoint_hessian_12(self, u, v, x,htol=1e-6):
        return np.zeros(self.var['nz']), 0

    def apply_adjoint_hessian_21(self, u, v, x,htol=1e-6):
        return np.zeros(self.var['nu']), 0

    def apply_adjoint_hessian_22(self, u, v, x,htol=1e-6):
        return np.zeros(self.var['nz']), 0

    def evaluate_nonlinearity_value(self, u,htol=1e-6):
        n = self.var['n']
        Nu = np.zeros(n - 1)
        Nu[:-1] += u[:-1] * u[1:] + u[1:] ** 2
        Nu[1:] -= u[:-1] * u[1:] + u[:-1] ** 2
        return Nu / 6

    def evaluate_nonlinearity_jacobian(self, u):
        n = self.var['n']
        
        
       
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
beta = Burgers.beta
M = Burgers.M
R = Burgers.R
A = Burgers.A
Rlump = Burgers.Rlump
B = Burgers.B
b = Burgers.b
ud = Burgers.ud
mesh = Burgers.mesh
var = {
       'beta':beta,
       'n':n,
       'nu':nu,
       'nz':nz,
       'alpha':alpha,
       'A':A,
       'M':M,
       'R':R,
       'Rlump':Rlump,
       'B':B,
       'b':b,
       'ud':ud,
       'useEuclidean': False,
       'mesh':mesh
       }
obj = Objective(var)
con = ConstraintSolver(var)
red_obj = ReducedObjective(obj, con)
z = np.random.rand(nz)
u = np.zeros(nu)
ftol = 1e-6
x = np.hstack([u, z])

# Evaluate the reduced objective function value
val,err = red_obj.value(z, ftol)
print("Objective function value:", val)
#print("Error estimate:",err)

#Compute the reduced gradient
#gtol = 1e-6
#grad,gerr = red_obj.gradient(x_try, gtol)
#print("Gradient:",grad.shape)
#print("Gradient error estimate:", gerr)
#Profile the computation
#red_obj.profile()

#Compute the Hessian
#htol = 1e-6
#v = np.random.rand(var['nz'])
#hv, herr = red_obj.hessVec(v, x_try, htol)
#print("Hessian-vector product:", hv.shape)
#print("Hessian-vector Error estimate:", herr)

def trustregion_step_SPG2(x, val, dgrad, phi, problem, params, cnt):
    params.setdefault('maxitsp', 10)
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1)
    params.setdefault('gtol', np.sqrt(np.finfo(float).eps))
    
    safeguard = 1e2 * np.finfo(float).eps
    
    x0 = x
    g0 = dgrad
    snorm = 0
    
    # Evaluate model at GCP
    sHs = 0
    gs = 0
    valold = val
    phiold = phi
    valnew = valold
    phinew = phiold
    
    t0 = max(params['lam_min'], min(params['lam_max'], params['t'] / problem.pvector.norm(dgrad)))
    
    # Set exit flag
    iter_count = 0
    iflag = 1
    
    for iter0 in range(1, params['maxitsp'] + 1):
        snorm0 = snorm
        
        # Compute step
        x1 = problem.obj_nonsmooth.prox(x0 - t0 * g0, t0)
        s = x1 - x0
        
        # Check optimality conditions
        gnorm = problem.pvector.norm(s)
        if gnorm / t0 <= params.get('tolsp', 0) and iter_count > 1:
            iflag = 0
            break
        
        # Compute relaxation parameter
        alphamax = 1
        snorm = problem.pvector.norm(x1 - x)
        if snorm >= params['delta'] - safeguard:
            ds = problem.pvector.dot(s, x0 - x)
            dd = gnorm ** 2
            alphamax = min(1, (-ds + np.sqrt(ds ** 2 + dd * (params['delta'] ** 2 - snorm0 ** 2))) / dd)
        
        #Hs = red_obj.hessVec(v, z, htol)[0]
        Hs = problem.obj_smooth.hessVec(s, x, params['gtol'])[0]
        
        sHs = problem.dvector.apply(Hs, s)
        g0s = problem.pvector.dot(g0, s)
        phinew = problem.obj_nonsmooth.value(x1)
        #eps = 1e-12
        alpha0 = max(-(g0s + phinew - phiold), gnorm ** 2 / t0) / sHs
        
        if sHs <= safeguard:
            alpha = alphamax
            if 0.5 * alphamax < alpha0 and iter0 > 1:
                alpha = 0
                phinew = phiold
                valnew = valold
                snorm = snorm0
                iflag = 3
                break
        else:
            alpha = min(alphamax, alpha0)
        
        # Update iterate
        if alpha == 1:
            x0 = x1
            g0 = problem.dvector.dual(Hs) + g0
            valnew = valold + g0s + 0.5 * sHs
        else:
            x0 = x0 + alpha * s
            g0 = alpha * problem.dvector.dual(Hs) + g0
            valnew = valold + alpha * g0s + 0.5 * alpha ** 2 * sHs
            phinew = problem.obj_nonsmooth.value(x0)
            snorm = problem.pvector.norm(x0 - x)
        
        # Update model information
        valold = valnew
        phiold = phinew
        
        # Check step size
        if snorm >= params['delta'] - safeguard:
            iflag = 2
            break
        
        norm_g0 = problem.pvector.norm(g0)
        
        # Update spectral step length
        if sHs <= safeguard:
            #if norm_g0 == 0:
                #norm_g0 = eps
                
            lambdaTmp = params['t'] / norm_g0
            
        else:
            lambdaTmp = gnorm ** 2 / sHs
        
        t0 = max(params['lam_min'], min(params['lam_max'], lambdaTmp))
    
    s = x0 - x
    pRed = (val + phi) - (valnew + phinew)
    
    iter_count = max(iter_count, iter0)
    
    return s, snorm, pRed, phinew, iflag, iter_count, cnt, params

# Example Problem Class 


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
    
class ObjectiveNonSmooth:
    """
    
    nonsmooth term of obj i.e. beta*\int_{\Omega}|z|\,dx
    
    """
    def __init__(self, var):
        self.var = var
        
    def prox(self, x, t):
        return self.var['beta'] * np.sign(x) * np.maximum(np.abs(x) - 1e-2 * t, 0)

    def value(self, x):
        return self.var['beta']*np.linalg.norm(x)

    


class L1Norm:
    def __init__(self, var):
        self.var = var

    def value(self, x):
        return self.var['beta'] * np.dot(self.var['Rlump'].T, np.abs(x))

    def prox(self, x, t):
        if self.var['useEuclidean']:
            return np.maximum(0, np.abs(x) - t * self.var['Rlump'] * self.var['beta']) * np.sign(x)
        else:
            return np.maximum(0, np.abs(x) - t * self.var['beta']) * np.sign(x)

    def dir_deriv(self, s, x):
        sx = np.sign(x)
        return self.var['beta'] * (np.dot(sx.T, s) + np.dot((1 - np.abs(sx)).T, np.abs(s)))

    def project_sub_diff(self, g, x):
        sx = np.sign(x)
        return self.var['beta'] * sx + (1 - np.abs(sx)) * np.clip(g, -self.var['beta'], self.var['beta'])

    def gen_jac_prox(self, x, t):
        d = np.ones_like(x)
        px = self.prox(x, t)
        ind = px == 0
        d[ind] = 0
        return np.diag(d), ind

    def apply_prox_jacobian(self, v, x, t):
        if self.var['useEuclidean']:
            ind = np.abs(x) <= t * self.var['Rlump'] * self.var['beta']
        else:
            ind = np.abs(x) <= t * self.var['beta']
        Dv = v.copy()
        Dv[ind] = 0
        return Dv

    def get_parameter(self):
        return self.var['beta']
    
        
    
class Problem:
    def __init__(self):
        self.pvector = L2vectorPrimal(var)
        self.dvector = L2vectorDual(var)
        #self.obj_smooth = ReducedObjective(obj, con)
        self.obj_smooth = red_obj
        self.obj_nonsmooth = L1Norm(var)
        

#Example usage
# Initial guess for the control vector

obj_nonsmooth = L1Norm(var)
problem = Problem()
# Parameters for the trust-region solver
params = {
    'maxitsp': 10,
    'lam_min': 1e-12,
    'lam_max': 1e12,
    't': 1,
    'gtol': np.sqrt(np.finfo(float).eps),
    'delta': 1.0  # Trust-region radius
}

# Evaluate the initial objective function value
ftol = 1e-6
val, err = red_obj.value(z, ftol)
phi = obj_nonsmooth.value(z)

# Compute the initial gradient
gtol = 1e-6
grad, gerr = red_obj.gradient(z, gtol)

# Run the trust-region subproblem solver
s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
    z, val, grad, phi, problem, params, cnt={}
)

# Update the control vector
z_new = z + s

# Print results
print("Step:", s)
print("New control vector:", z_new)
print("Predicted reduction:", pRed)
print("Exit flag:", iflag)

#Trust region method

import time

def trustregion(x0, problem, params):
    """
    Trust-region optimization algorithm.

    Parameters:
    x0 (np.array): Initial guess for the control variable.
    problem (Problem): Problem class containing objective functions and vector spaces.
    params (dict): Dictionary of algorithmic parameters.

    Returns:
    x (np.array): Optimized control variable.
    cnt (dict): Dictionary of counters and history.
    """
    start_time = time.time()

    # Check and set default parameters
    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 100)
    params.setdefault('reltol', False)
    params.setdefault('gtol', 1e-6)
    params.setdefault('stol', 1e-6)
    params.setdefault('ocScale', 1.0)
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)
    params.setdefault('eta1', 1e-4)
    params.setdefault('eta2', 0.75)
    params.setdefault('gamma1', 0.25)
    params.setdefault('gamma2', 10.0)
    params.setdefault('delta', 1.0)
    params.setdefault('deltamax', 1e10)
    params.setdefault('useSecant', False)
    params.setdefault('useSecantPrecond', False)
    params.setdefault('secantType', 2)
    params.setdefault('secantSize', 10)
    params.setdefault('initScale', 1.0)
    params.setdefault('useDefault', True)
    params.setdefault('useInexactObj', False)
    params.setdefault('etascale', 1 - 1e-3)
    params.setdefault('maxValTol', 1)
    params.setdefault('scaleValTol', 1e0)
    params.setdefault('forceValTol', 1 - 1e-3)
    params.setdefault('expValTol', 0.9)
    params.setdefault('useInexactGrad', False)
    params.setdefault('maxGradTol', 1)
    params.setdefault('scaleGradTol', 1e0)

    # Initialize counters
    cnt = {
        'AlgType': f"TR-{params.get('spsolver', 'SPG2')}",
        'iter': 0,
        'nobj1': 0,
        'ngrad': 0,
        'nhess': 0,
        'nobj2': 0,
        'nprox': 0,
        'nprec': 0,
        'timetotal': 0,
        'objhist': [],
        'obj1hist': [],
        'obj2hist': [],
        'gnormhist': [],
        'snormhist': [],
        'deltahist': [],
        'nobj1hist': [],
        'nobj2hist': [],
        'ngradhist': [],
        'nhesshist': [],
        'nproxhist': [],
        'timestor': [],
        'valerr': [],
        'valtol': [],
        'graderr': [],
        'gradtol': []
    }

    # Compute initial function information
    if hasattr(problem.obj_smooth, 'begin_counter'):
        cnt = problem.obj_smooth.begin_counter(0, cnt)

    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1)
        cnt['nprox'] += 1
    else:
        x = x0

    problem.obj_smooth.update(x, 'init')
    ftol = 1e-12
    if params['useInexactObj']:
        ftol = params['maxValTol']

    val, _ = problem.obj_smooth.value(x, ftol)
    cnt['nobj1'] += 1

    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    if hasattr(problem.obj_smooth, 'end_counter'):
        cnt = problem.obj_smooth.end_counter(0, cnt)

    # Initialize secant if needed
    #if params['useSecant'] or params['useSecantPrecond']:
    #    problem.secant = SR1(params['secantSize'], params['useDefault'], params['initScale'])
        

    if params['useSecantPrecond']:
        problem.prec.apply = lambda x: problem.secant.apply(x, problem.pvector, problem.dvector)
        problem.prec.apply_inverse = lambda x: problem.secant.apply_inverse(x, problem.pvector, problem.dvector)

    # Output header
    print(f"\nNonsmooth Trust-Region Method using {params.get('spsolver', 'SPG2')} Subproblem Solver")
    print("  iter            value            gnorm              del            snorm       nobjs      ngrad      nhess      nobjn      nprox    iterSP    flagSP")
    print(f"  {0:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}              ---      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}       ---       ---")

    # Storage
    cnt['objhist'].append(val + phi)
    cnt['obj1hist'].append(val)
    cnt['obj2hist'].append(phi)
    cnt['gnormhist'].append(gnorm)
    cnt['snormhist'].append(np.nan)
    cnt['deltahist'].append(params['delta'])
    cnt['nobj1hist'].append(cnt['nobj1'])
    cnt['nobj2hist'].append(cnt['nobj2'])
    cnt['ngradhist'].append(cnt['ngrad'])
    cnt['nhesshist'].append(cnt['nhess'])
    cnt['nproxhist'].append(cnt['nprox'])
    cnt['timestor'].append(np.nan)

    # Set optimality tolerance
    gtol = params['gtol']
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    # Check stopping criterion
    if gnorm <= gtol:
        return x, cnt

    # Iterate
    for i in range(1, params['maxit'] + 1):
        if hasattr(problem.obj_smooth, 'begin_counter'):
            cnt = problem.obj_smooth.begin_counter(i, cnt)

        # Solve trust-region subproblem
        params['tolsp'] = min(params['atol'], params['rtol'] * gnorm ** params['spexp'])
        dgrad = problem.dvector.dual(grad)
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val, dgrad, phi, problem, params, cnt
        )

        # Update function information
        xnew = x + s
        problem.obj_smooth.update(xnew, 'trial')
        valnew, val, cnt = compute_value(xnew, x, val, problem.obj_smooth, pRed, params, cnt)

        # Accept/reject step and update trust-region radius
        aRed = (val + phi) - (valnew + phinew)
        if aRed < params['eta1'] * pRed:
            params['delta'] = params['gamma1'] * min(snorm, params['delta'])
            problem.obj_smooth.update(x, 'reject')
            if params['useInexactGrad']:
                grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
        else:
            x = xnew
            val = valnew
            phi = phinew
            problem.obj_smooth.update(x, 'accept')
            grad0 = grad
            grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
            if aRed > params['eta2'] * pRed:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

            # Update secant
            if params['useSecant'] or params['useSecantPrecond']:
                y = grad - grad0
                problem.secant.update(s, y, problem.pvector, problem.dvector)
                if params['useSecantPrecond']:
                    problem.prec.apply = lambda x: problem.secant.apply(x, problem.pvector, problem.dvector)
                    problem.prec.apply_inverse = lambda x: problem.secant.apply_inverse(x, problem.pvector, problem.dvector)

        # Output iteration history
        if i % params['outFreq'] == 0:
            print(f"  {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")

        # Storage
        cnt['objhist'].append(val + phi)
        cnt['obj1hist'].append(val)
        cnt['obj2hist'].append(phi)
        cnt['gnormhist'].append(gnorm)
        cnt['snormhist'].append(snorm)
        cnt['deltahist'].append(params['delta'])
        cnt['nobj1hist'].append(cnt['nobj1'])
        cnt['nobj2hist'].append(cnt['nobj2'])
        cnt['ngradhist'].append(cnt['ngrad'])
        cnt['nhesshist'].append(cnt['nhess'])
        cnt['nproxhist'].append(cnt['nprox'])
        cnt['timestor'].append(time.time() - start_time)

        if hasattr(problem.obj_smooth, 'end_counter'):
            cnt = problem.obj_smooth.end_counter(i, cnt)

        # Check stopping criterion
        if gnorm <= gtol or snorm <= stol or i >= params['maxit']:
            if i % params['outFreq'] != 0:
                print(f"  {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")
            if gnorm <= gtol:
                flag = 0
            elif i >= params['maxit']:
                flag = 1
            else:
                flag = 2
            break

    cnt['iter'] = i
    cnt['timetotal'] = time.time() - start_time

    print("Optimization terminated because ", end="")
    if flag == 0:
        print("optimality tolerance was met")
    elif flag == 1:
        print("maximum number of iterations was met")
    else:
        print("step tolerance was met")
    print(f"Total time: {cnt['timetotal']:8.6e} seconds")

    return x, cnt


def compute_value(x, xprev, fvalprev, obj, pRed, params, cnt):
    """
    Compute the objective function value with inexactness handling.

    Parameters:
    x (np.array): Current point.
    xprev (np.array): Previous point.
    fvalprev (float): Previous function value.
    obj (Objective): Objective function class.
    pRed (float): Predicted reduction.
    params (dict): Algorithm parameters.
    cnt (dict): Counters.

    Returns:
    fval (float): Current function value.
    fvalprev (float): Previous function value.
    cnt (dict): Updated counters.
    """
    ftol = 1e-12
    valerrprev = 0
    if params['useInexactObj']:
        omega = params['expValTol']
        scale = params['scaleValTol']
        force = params['forceValTol']
        eta = params['etascale'] * min(params['eta1'], 1 - params['eta2'])
        ftol = min(params['maxValTol'], scale * (eta * min(pRed, force ** cnt['nobj1'])) ** (1 / omega))
        fvalprev, valerrprev = obj.value(xprev, ftol)
        cnt['nobj1'] += 1

    obj.update(x, 'trial')
    fval, valerr = obj.value(x, ftol)
    cnt['nobj1'] += 1
    cnt['valerr'].append(max(valerr, valerrprev))
    cnt['valtol'].append(ftol)

    return fval, fvalprev, cnt


def compute_gradient(x, problem, params, cnt):
    """
    Compute the gradient with inexactness handling.

    Parameters:
    x (np.array): Current point.
    problem (Problem): Problem class.
    params (dict): Algorithm parameters.
    cnt (dict): Counters.

    Returns:
    grad (np.array): Gradient.
    dgrad (np.array): Dual gradient.
    gnorm (float): Gradient norm.
    cnt (dict): Updated counters.
    """
    if params['useInexactGrad']:
        scale0 = params['scaleGradTol']
        gtol = min(params['maxGradTol'], scale0 * params['delta'])
        gerr = gtol + 1
        while gerr > gtol:
            grad, gerr = problem.obj_smooth.gradient(x, gtol)
            cnt['ngrad'] += 1
            dgrad = problem.dvector.dual(grad)
            pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
            cnt['nprox'] += 1
            gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']
            gtol = min(params['maxGradTol'], scale0 * min(gnorm, params['delta']))
    else:
        gtol = 1e-12
        grad, gerr = problem.obj_smooth.gradient(x, gtol)
        cnt['ngrad'] += 1
        dgrad = problem.dvector.dual(grad)
        pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
        cnt['nprox'] += 1
        gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']

    params['gradTol'] = gtol
    cnt['graderr'].append(gerr)
    cnt['gradtol'].append(gtol)

    return grad, dgrad, gnorm, cnt

#Example usage
def set_default_parameters(name):
    params = {}
    
    # General Parameters
    params['spsolver'] = name.replace(' ', '')
    params['outFreq'] = 1
    params['debug'] = False
    params['initProx'] = False
    params['t'] = 1
    params['safeguard'] = np.sqrt(np.finfo(float).eps)
    
    # Stopping tolerances
    params['maxit'] = 200
    params['reltol'] = True
    params['gtol'] = 1e-5
    params['stol'] = 1e-10
    params['ocScale'] = params['t']
    
    # Trust-region parameters
    params['eta1'] = 0.05
    params['eta2'] = 0.9
    params['gamma1'] = 0.25
    params['gamma2'] = 2.5
    params['delta'] = 50.0
    params['deltamax'] = 1e10
    
    # Subproblem solve tolerances
    params['atol'] = 1e-5
    params['rtol'] = 1e-3
    params['spexp'] = 2
    params['maxitsp'] = 15
    
    # GCP and subproblem solve parameter
    params['useGCP'] = False
    params['mu1'] = 1e-4
    params['beta_dec'] = 0.1
    params['beta_inc'] = 10.0
    params['maxit_inc'] = 2
    
    # SPG and spectral GCP parameters
    params['lam_min'] = 1e-12
    params['lam_max'] = 1e12
    
    # Inexactness parameters
    params['useInexactObj'] = False
    params['useInexactGrad'] = False
    params['gradTol'] = np.sqrt(np.finfo(float).eps)
    
    
    
    return params


z_opt, cnt = trustregion(z, problem, params)

# Print results
#print("\nOptimized solution:", z_opt)
print("debug")


def deriv_check_simopt(u0, z0, obj, con, tol):
    """
    Perform derivative checks using finite differences.

    Parameters:
    u0 (np.array): Initial state vector.
    z0 (np.array): Initial control vector.
    obj (Objective): Objective function class.
    con (Constraint): Constraint class.
    tol (float): Tolerance for function evaluations.
    """
    # Random directions for finite differences
    u = np.random.randn(*u0.shape)
    udir = np.random.randn(*u0.shape)
    z = np.random.randn(*z0.shape)
    zdir = np.random.randn(*z0.shape)
    
    lambda_ = np.random.randn(*(con.value(np.hstack([u, z]))[0]).shape)

    # Evaluate objective and constraint functions
    f = obj.value(np.hstack([u, z]), tol)[0]
    
    df1 = obj.gradient_1(np.hstack([u, z]), tol)[0]
    
    df2 = obj.gradient_2(np.hstack([u, z]), tol)[0]
    c = con.value(np.hstack([u, z]), tol)[0]
    J1d = con.apply_jacobian_1(udir, np.hstack([u, z]), tol)[0]
    J2d = con.apply_jacobian_2(zdir, np.hstack([u, z]), tol)[0]
    J1 = con.apply_adjoint_jacobian_1(lambda_, np.hstack([u, z]), tol)[0]
    J2 = con.apply_adjoint_jacobian_2(lambda_, np.hstack([u, z]), tol)[0]

    # Check objective gradient_1 using finite differences
    print("\n Objective gradient_1 check using finite differences (FDs)")
    print(" FD step size      grad'*v      FD approx.  absolute error")
    delta = 1
    for d in range(13):
        f1 = obj.value(np.hstack([u + delta * udir, z]), tol)[0]
        error = np.abs(np.dot(df1, udir) - (f1 - f) / delta)
        print(f" {delta:12.6e}  {np.dot(df1, udir):12.6e}  {(f1 - f) / delta:12.6e}  {error:12.6e}")
        delta /= 10

    # Check objective gradient_2 using finite differences
    print("\n Objective gradient_2 check using finite differences (FDs)")
    print(" FD step size      grad'*v      FD approx.  absolute error")
    delta = 1
    for d in range(13):
        f1 = obj.value(np.hstack([u, z + delta * zdir]), tol)[0]
        error = np.abs(np.dot(df2, zdir) - (f1 - f) / delta)
        print(f" {delta:12.6e}  {np.dot(df2, zdir):12.6e}  {(f1 - f) / delta:12.6e}  {error:12.6e}")
        delta /= 10

    # Check objective Hessian_11 using finite differences
    print("\n Objective Hessian_11 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    hv = obj.hessVec_11(udir, np.hstack([u, z]), tol)[0]
    delta = 1
    for d in range(13):
        g1 = obj.gradient_1(np.hstack([u + delta * udir, z]), tol)[0]
        
        fd_approx = (g1 - df1) / delta
        error = np.linalg.norm(hv - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check objective Hessian_12 using finite differences
    print("\n Objective Hessian_12 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    hv = obj.hessVec_12(zdir, np.hstack([u, z]), tol)[0]
    delta = 1
    for d in range(13):
        g1 = obj.gradient_1(np.hstack([u, z + delta * zdir]), tol)[0]
        fd_approx = (g1 - df1) / delta
        fd_approx_for_norm = fd_approx.reshape(-1,1)
        
        
        error = np.linalg.norm(hv - fd_approx_for_norm)
       
        print(f" {delta:12.6e}     {sp.linalg.norm(hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check objective Hessian_21 using finite differences
    print("\n Objective Hessian_21 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    hv = obj.hessVec_21(udir, np.hstack([u, z]), tol)[0]
    delta = 1
    for d in range(13):
        g1 = obj.gradient_2(np.hstack([u + delta * udir, z]), tol)[0]
        fd_approx = (g1 - df2) / delta
        fd_approx_for_norm = fd_approx.reshape(-1,1)
        error = np.linalg.norm(hv - fd_approx_for_norm)
        print(f" {delta:12.6e}     {sp.linalg.norm(hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check objective Hessian_22 using finite differences
    print("\n Objective Hessian_22 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    hv = obj.hessVec_22(zdir, np.hstack([u, z]), tol)[0]
    delta = 1
    for d in range(13):
        g1 = obj.gradient_2(np.hstack([u, z + delta * zdir]), tol)[0]
        fd_approx = (g1 - df2) / delta
        fd_approx_for_norm = fd_approx.reshape(-1,1)
        error = np.linalg.norm(hv - fd_approx_for_norm)
        print(f" {delta:12.6e}     {np.linalg.norm(hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Jacobian_1 using finite differences
    print("\n Constraint Jacobian_1 check using finite differences (FDs)")
    print(" FD step size      norm(Jac*v)     norm(FD approx.)   absolute error")
    delta = 1
    for d in range(13):
        c1 = con.value(np.hstack([u + delta * udir, z]), tol)[0]
        fd_approx = (c1 - c) / delta
        error = np.linalg.norm(J1d - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(J1d):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Jacobian_2 using finite differences
    print("\n Constraint Jacobian_2 check using finite differences (FDs)")
    print(" FD step size      norm(Jac*v)     norm(FD approx.)   absolute error")
    delta = 1
    for d in range(13):
        c1 = con.value(np.hstack([u, z + delta * zdir]), tol)[0]
        fd_approx = (c1 - c) / delta
        error = np.linalg.norm(J2d - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(J2d):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Hessian_11 using finite differences
    print("\n Constraint Hessian_11 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    Hv = con.apply_adjoint_hessian_11(lambda_, udir, np.hstack([u, z]), tol)[0]
    delta = 1
    for d in range(13):
        Jn = con.apply_adjoint_jacobian_1(lambda_, np.hstack([u + delta * udir, z]), tol)[0]
        
        fd_approx = (Jn - J1) / delta
        error = np.linalg.norm(Hv - fd_approx) / (1 + np.linalg.norm(Hv))
        print(f" {delta:12.6e}     {np.linalg.norm(Hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Hessian_12 using finite differences
    print("\n Constraint Hessian_12 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    Hv = con.apply_adjoint_hessian_12(lambda_, udir, np.hstack([u, z]), tol)[0]
    delta = 1
    for d in range(13):
        Jn = con.apply_adjoint_jacobian_2(lambda_, np.hstack([u + delta * udir, z]), tol)[0]
        fd_approx = (Jn - J2) / delta
        error = np.linalg.norm(Hv - fd_approx) / (1 + np.linalg.norm(Hv))
        print(f" {delta:12.6e}     {np.linalg.norm(Hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Hessian_21 using finite differences
    print("\n Constraint Hessian_21 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    Hv = con.apply_adjoint_hessian_21(lambda_, zdir, np.hstack([u, z]), tol)[0]
    delta = 1
    for d in range(13):
        Jn = con.apply_adjoint_jacobian_1(lambda_, np.hstack([u, z + delta * zdir]), tol)[0]
        fd_approx = (Jn - J1) / delta
        error = np.linalg.norm(Hv - fd_approx) / (1 + np.linalg.norm(Hv))
        print(f" {delta:12.6e}     {np.linalg.norm(Hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Hessian_22 using finite differences
    print("\n Constraint Hessian_22 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    Hv = con.apply_adjoint_hessian_22(lambda_, zdir, np.hstack([u, z]), tol)[0]
    delta = 1
    for d in range(13):
        Jn = con.apply_adjoint_jacobian_2(lambda_, np.hstack([u, z + delta * zdir]), tol)[0]
        fd_approx = (Jn - J2) / delta
        error = np.linalg.norm(Hv - fd_approx) / (1 + np.linalg.norm(Hv))
        print(f" {delta:12.6e}     {np.linalg.norm(Hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check solve
    print("\n Solve check")
    uz = con.solve(z, tol)[0]
    _, res = con.value(np.hstack([uz, z]), tol)
    print(f"  Absolute Residual = {res:12.6e}")
    print(f"  Relative Residual = {res / np.linalg.norm(uz):12.6e}")

    # Check applyInverseJacobian_1
    print("\n Check applyInverseJacobian_1")
    uz = con.apply_inverse_jacobian_1(udir, np.hstack([u, z]), tol)[0]
    Juz = con.apply_jacobian_1(uz, np.hstack([u, z]), tol)[0]
    res = np.linalg.norm(Juz - udir)
    print(f"  Absolute Error = {res:12.6e}")
    print(f"  Relative Error = {res / np.linalg.norm(udir):12.6e}")

    # Check applyInverseAdjointJacobian_1
    print("\n Check applyInverseAdjointJacobian_1")
    uz = con.apply_inverse_adjoint_jacobian_1(udir, np.hstack([u, z]), tol)[0]
    Juz = con.apply_adjoint_jacobian_1(uz, np.hstack([u, z]), tol)[0]
    res = np.linalg.norm(Juz - udir)
    print(f"  Absolute Error = {res:12.6e}")
    print(f"  Relative Error = {res / np.linalg.norm(udir):12.6e}")

    # Check applyAdjointJacobian_1
    print("\n Check applyAdjointJacobian_1")
    vdir = np.random.randn(*udir.shape)
    aju = con.apply_adjoint_jacobian_1(udir, np.hstack([u, z]), tol)[0]
    ju = con.apply_jacobian_1(vdir, np.hstack([u, z]), tol)[0]
    res = np.abs(np.dot(aju, vdir) - np.dot(ju, udir))
    print(f"  Absolute Error = {res:12.6e}")

    # Check applyAdjointJacobian_2
    print("\n Check applyAdjointJacobian_2")
    aju = con.apply_adjoint_jacobian_2(udir, np.hstack([u, z]), tol)[0]
    ju = con.apply_jacobian_2(zdir, np.hstack([u, z]), tol)[0]
    res = np.abs(np.dot(aju, zdir) - np.dot(ju, udir))
    print(f"  Absolute Error = {res:12.6e}")



def deriv_check(x, d, problem, tol):
    # Update the problem object
    problem.obj_smooth.update(x, 'temp')
    val, _ = problem.obj_smooth.value(x, tol)
    grad, _ = problem.obj_smooth.gradient(x, tol)
    gd = problem.dvector.apply(grad, d)
    t = 1
    
    print('\n  Finite Difference Gradient Check')
    print('           t       DirDeriv        FinDiff         Error')
    
    for i in range(13):
        xnew = x + t * d
        problem.obj_smooth.update(xnew, 'temp')
        valnew, _ = problem.obj_smooth.value(xnew, tol)
        fd = (valnew - val) / t
        
        print(f'  {t:6.4e}    {gd: 6.4e}    {fd: 6.4e}    {abs(fd - gd):6.4e}')
        t *= 0.1

    problem.obj_smooth.update(x, 'temp')
    hv, _ = problem.obj_smooth.hessVec(d, x, tol)
    hvnorm = problem.dvector.norm(hv)
    t = 1
    
    print('\n  Finite Difference Hessian Check')
    print('           t        HessVec        FinDiff         Error')
    
    for i in range(13):
        xnew = x + t * d
        problem.obj_smooth.update(xnew, 'temp')
        gradnew, _ = problem.obj_smooth.gradient(xnew, tol)
        fd = (gradnew - grad) / t
        fdnorm = problem.dvector.norm(fd)
        
        print(f'  {t:6.4e}    {hvnorm: 6.4e}    {fdnorm: 6.4e}    {problem.dvector.norm(fd - hv):6.4e}')
        t *= 0.1

    problem.obj_smooth.update(x, 'temp')
    d2 = np.random.randn(*d.shape)  # Random d2 of the same shape as d
    hd2, _ = problem.obj_smooth.hessVec(d2, x, tol)
    vhd2 = problem.dvector.apply(hd2, d)
    d2hv = problem.dvector.apply(hv, d2)
    
    print('\n  Hessian Symmetry Check')
    print(f'    <x,Hy> = {vhd2: 6.4e}')
    print(f'    <y,Hx> = {d2hv: 6.4e}')
    print(f'    AbsErr = {abs(vhd2 - d2hv): 6.4e}')


#Vector check
def vector_check(primal, dual, problem):
    print('\n  Vector Check')
    print('  Check Dual Consistency')
    
    xp = problem.dvector.dual(dual)
    xpp = problem.pvector.dual(xp)
    err = problem.dvector.norm(dual - xpp)
    print(f'  norm(x-dual(dual(x))) = {err:.4e}')
    
    xp = problem.pvector.dual(primal)
    xpp = problem.dvector.dual(xp)
    err = problem.pvector.norm(primal - xpp)
    print(f'  norm(y-dual(dual(y))) = {err:.4e}')
    
    print('\n  Check Apply Consistency')
    xp = problem.pvector.dual(primal)
    xydot = problem.dvector.dot(xp, dual)
    xyapply = problem.dvector.apply(dual, primal)
    err = abs(xydot - xyapply)
    print(f' |x.dot(dual(y))-x.apply(y)| = {err:.4e}')
    
    xp = problem.dvector.dual(dual)
    xydot = problem.pvector.dot(xp, primal)
    xyapply = problem.pvector.apply(primal, dual)
    err = abs(xydot - xyapply)
    print(f' |y.dot(dual(x))-y.apply(x)| = {err:.4e}')


#print(vector_check(z, z_opt, problem))


class Euclidean:
    def dot(self, x, y):
        return np.dot(x.flatten(), y.flatten())

    def apply(self, x, y):
        return self.dot(x, y)

    def norm(self, x):
        return np.sqrt(self.dot(x, x))

    def dual(self, x):
        return x




def driver(savestats, name):
    print("driver started")
    np.random.seed(0)

    # Set up optimization problem
    n = 512  # Number of cells
    nu = 0.08  # Viscosity
    alpha = 1e-4  # L2 penalty parameter
    beta = 1e-2  # L1 penalty parameter
    usepc = True  # Use piecewise constant controls
    useInexact = False
    derivCheck = True

    
    obj = Objective(var)
    con = ConstraintSolver(var)
    var['useEuclidean'] = False

    # Define the optimization problem
    #problem = {
    #    "obj_smooth": ReducedObjective(obj, con),
    #    "obj_nonsmooth": L1Norm(var),
    #    "pvector": None,
    #    "dvector": None,
    #}

    if var['useEuclidean']:
        problem.pvector = Euclidean(var)
        problem.dvector = Euclidean(var)
    else:
        problem.pvector = L2vectorPrimal(var)
        problem.dvector = L2vectorDual(var)

    dim = n if usepc else n + 1

    if derivCheck:
        x = np.random.randn(dim)
        d = np.random.randn(dim)
        deriv_check_simopt(np.zeros(var['nu']), x, obj, con, 1e-4 * np.sqrt(np.finfo(float).eps))
        deriv_check(x, d, problem, 1e-4 * np.sqrt(np.finfo(float).eps))
        vector_check(x, d, problem)

    x0 = np.ones(dim)
    cnt = {}

    # Update default parameters
    params = set_default_parameters("SPG2")
    params["reltol"] = False
    params["t"] = 2 / alpha
    params["ocScale"] = 1 / alpha

    # Solve optimization problem
    start_time = time.time()
    problem.obj_smooth.reset()
    con.reset()
    x, cnt_tr = trustregion(x0, problem, params)
    elapsed_time = time.time() - start_time
    
    print(f"Optimization completed in {elapsed_time:.2f} seconds")

    pro_tr = problem.obj_smooth.profile()

    cnt[1] = (cnt_tr, pro_tr)
    

    print("\nSummary")
    print(
        "           niter     nobjs     ngrad     nhess     nobjn     nprox     nstat     nadjo     nssen     nasen"
    )
    print(
        f"   SGP2:  {cnt[1][0]['iter']:6d}    {cnt[1][0]['nobj1']:6d}    {cnt[1][0]['ngrad']:6d}    {cnt[1][0]['nhess']:6d}    "
        f"{cnt[1][0]['nobj2']:6d}    {cnt[1][0]['nprox']:6d}    {cnt[1][1]['nstate']:6d}    {cnt[1][1]['nadjoint']:6d}    "
        f"{cnt[1][1]['nstatesens']:6d}    {cnt[1][1]['nadjointsens']:6d}"
    )

    mesh = 0.5 * (var['mesh'][:-1] + var['mesh'][1:]) if usepc else var['mesh']

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(mesh, x, "b", linewidth=3)
    plt.xlabel("x")
    plt.ylabel("z(x)")
    plt.title("Optimal Control")
    plt.grid()
    plt.show()

    return cnt


cnt = driver(False, "test_run")
