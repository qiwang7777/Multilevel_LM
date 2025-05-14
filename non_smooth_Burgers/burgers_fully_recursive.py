import numpy as np
import scipy.sparse as sp
from scipy.sparse import spdiags, diags,lil_matrix
from scipy.integrate import quad
import time
from checks import deriv_check, deriv_check_simopt, vector_check
from setDefaultParameters import set_default_parameters

#Problem Class
class Problem:
    def __init__(self, var, R):
        self.var  = var
        if var['useEuclidean']:
            self.pvector   = Euclidean(var)
            self.dvector   = Euclidean(var)
        else:
            self.pvector   = L2vectorPrimal(var)
            self.dvector   = L2vectorDual(var)
        self.obj_smooth    = ReducedObjective(Objective(var), ConstraintSolver(var))
        self.obj_nonsmooth = L1Norm(var)
        self.R             = R

#Burger's variables
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

    def returnVars(self, useEuclidean):
      var = {'beta':self.beta,
             'n':self.n,
             'nu':self.n-1,
             'nz':self.n,
             'alpha':self.alpha,
             'A':self.A,
             'M':self.M,
             'R':self.R,
             'Rlump':self.Rlump,
             'B':self.B,
             'b':self.b,
             'ud':self.ud,
             'useEuclidean':useEuclidean,
             'mesh':self.mesh
            }
      return var

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

#Objective function
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
            self.pwork, aerr2 = self.con0.apply_inverse_adjoint_jacobian_1(rhs, np.hstack([self.uwork, z]), gtol)
            self.cnt['ninvJ1'] += 1
            self.is_adjoint_computed = True
            self.cnt['nadjoint'] += 1
        Bp, jerr = self.con0.apply_adjoint_jacobian_2(self.pwork, np.hstack([self.uwork, z]), gtol)
        grad, gerr1 = self.obj0.gradient_2(np.hstack([self.uwork,z]), gtol)
        return grad + Bp, max(jerr, gerr1)

    def hessVec(self, v, z, htol):
        herr = 0
        if not self.is_state_computed or self.uwork is None:
            self.uwork, cnt0, serr  = self.con0.solve(z, htol)
            self.cnt['ninvJ1']     += cnt0
            self.is_state_computed  = True
            self.cnt['nstate']     += 1
        if not self.is_adjoint_computed or self.pwork is None:
            rhs, aerr1                = self.obj0.gradient_1(np.hstack([self.uwork, z]), htol)
            rhs                       = -rhs
            self.pwork, aerr2         = self.con0.apply_inverse_adjoint_jacobian_1(rhs, np.hstack([self.uwork, z]), htol)
            self.cnt['ninvJ1']       += 1
            self.is_adjoint_computed  = True
            self.cnt['nadjoint']     += 1
        # Solve state sensitivity equation
        rhs, sserr1  = self.con0.apply_jacobian_2(v,np.hstack([self.uwork, z]), htol)
        rhs          = -rhs
        w, sserr2    = self.con0.apply_inverse_jacobian_1(rhs, np.hstack([self.uwork, z]), htol)
        # add counters
        # Solve adjoint sensitivity equation
        rhs, aserr1  = self.obj0.hessVec_11(w, np.hstack([self.uwork, z]), htol)
        tmp, aserr2  = self.obj0.hessVec_12(v, np.hstack([self.uwork, z]),   htol)
        rhs         += tmp
        tmp, aserr3  = self.con0.apply_adjoint_hessian_11(self.pwork, w, np.hstack([self.uwork, z]), htol)
        rhs         += tmp
        tmp, aserr4  = self.con0.apply_adjoint_hessian_21(self.pwork, v, np.hstack([self.uwork, z]),htol)
        rhs         += tmp
        q, aserr5    = self.con0.apply_inverse_adjoint_jacobian_1(rhs, np.hstack([self.uwork, z]), htol)
        q            = -q
        self.cnt['ninvJ1'] += 1
        hv, herr1   = self.con0.apply_adjoint_jacobian_2(q, np.hstack([self.uwork, z]), htol)
        tmp,herr2   = self.obj0.hessVec_21(w,np.hstack([self.uwork,z]),htol)
        hv         += tmp
        tmp,herr3   = self.obj0.hessVec_22(v,np.hstack([self.uwork,z]),htol)
        hv         += tmp
        tmp,herr4   = self.con0.apply_adjoint_hessian_12(self.pwork,w,np.hstack([self.uwork,z]),htol)
        hv         += tmp
        tmp,herr5   = self.con0.apply_adjoint_hessian_22(self.pwork,v,np.hstack([self.uwork,z]),htol)
        hv         += tmp
        herr        = max(max(max(max(max(herr,herr1),herr2),herr3),herr4),herr5)

        return hv, max(herr1, aserr1, aserr5)

    def profile(self):
        print("\nProfile Reduced Objective")
        print("  #state    #adjoint    #statesens    #adjointsens    #linearsolves")
        print(f"  {self.cnt['nstate']:6d}      {self.cnt['nadjoint']:6d}        {self.cnt['nstatesens']:6d}          {self.cnt['nadjointsens']:6d}           {self.cnt['ninvJ1']:6d}")
        cnt = self.cnt.copy()
        cnt['con'] = self.con0.profile()
        return cnt
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

    # Compute objective function gradient (w.r.t. u)
    def gradient_1(self, x, gtol):
        nu = self.var['nu']
        M = self.var['M']
        ud = self.var['ud']
        u = x[:nu]

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
        nu   = self.var['nu']
        hv   = np.zeros((nu,))
        herr = 0

        return hv, herr

    # Apply objective function Hessian to a vector (hessVec_21)
    def hessVec_21(self, v, x, htol):
        nz = self.var['nz']
        hv   = np.zeros((nz,))
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
        ahuv = [0.0] * nu
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

        d1[:-1] = -2*u[:-1] - u[1:]
        d2[0] = u[1]
        d2[1:-1] = u[2:]-u[:-2]
        d2[-1] =  -u[-2]
        d3[1:] = u[:-1] + 2 * u[1:]
        J = spdiags([d1, d2, d3], [-1, 0, 1], n-1, n-1) / 6

        return J

    def profile(self):
        return []

class Euclidean:
    def dot(self, x, y):
        return np.dot(x.flatten(), y.flatten())

    def apply(self, x, y):
        return self.dot(x, y)

    def norm(self, x):
        return np.sqrt(self.dot(x, x))

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

#Recursive step
def restriction_R(m,n):
    """
    Construct a sparse orthonormal matrix R in R^{m\times n}

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns (n>>m).

    Returns
    -------
    matrix_R : TYPE
        DESCRIPTION.

    """
    matrix_R = np.zeros((m,n))
    for i in range(m):
        matrix_R[i,2*(i+1)-1] = 1/np.sqrt(2)
        matrix_R[i,2*i] = 1/np.sqrt(2)
    return matrix_R

#Subsolver for Taylor model
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

def trustregion_step(l,x,val,grad,phi,problems,params,cnt):
    #fine level l comes in
    dgrad                   = problems[l].dvector.dual(grad) #dual(grad) puts in primal space
    L                       = len(problems)
    if l < L-1:
      R                       = problems[l].R
      Rdgrad                  = R @ dgrad
      Rgnorm                  = problems[l+1].pvector.norm(Rdgrad)
    else:
      Rgnorm = 0.0
    gnorm                   = problems[l].pvector.norm(dgrad)

    if Rgnorm > 0.5*gnorm and Rgnorm >= 1e-3: #note: counters are off
      problemsL = [] #problem list goes from fine to coarse
      for i in range(0, L):
        if i == l+1:
          p               = Problem(problems[i].obj_nonsmooth.var, problems[i].R) #make next level problem quadratic
          p.obj_smooth    = modelTR(problems, params["useSecant"], 'recursive', l = i, R = problems[i-1].R, grad = grad, x = x)
          p.obj_nonsmooth = phiPrec(problems[l], R = problems[i-1].R)
          # d = np.random.randn(R.shape[0],)
          # deriv_check(R @ x, d, p, 1e-4 * np.sqrt(np.finfo(float).eps))
          # stp
          problemsL.append(p)
        else:
          problemsL.append(problems[i])
      Deltai           = params['delta']
      xnew, cnt_coarse = trustregion(l+1, R @ x, Deltai, problemsL, params)
      #recompute snorm, pRed, phinew
      params['delta'] = Deltai
      s               = R.T @ xnew - x
      snorm           = problems[l].pvector.norm(s)
      #check L = f_{i-1} + <R g, s> + phi_{i-1}(x0 + s)
      #m = L + phi_i - phi_{i-1} so M = f_{i-1} + <Rg, s> + \phi_i
      phinew = problems[l].obj_nonsmooth.value(x + s)
      cnt['nobj2'] += 1
      valnew, _ = problemsL[l+1].obj_smooth.value(xnew, 0.0)
      pRed   = val + phi - valnew - phinew
      iflag  = cnt_coarse['iflag']
      iter   = cnt_coarse['iter']
      cnt    = problemsL[l+1].obj_smooth.addCounter(cnt)
      cnt    = problemsL[l+1].obj_nonsmooth.addCounter(cnt)
    else:
      R                       = np.eye(x.shape[0])
      problemTR               = Problem(problems[l].var, R)
      problemTR.obj_smooth    = modelTR(problems, params["useSecant"], 'spg', l = l, R = R, grad = grad, x = x)
      problemTR.obj_nonsmooth = phiPrec(problems[l], R = R)
      # d = np.random.randn((x.shape[0]))
      # deriv_check(x, d, problemTR, 1e-4 * np.sqrt(np.finfo(float).eps))
      s, snorm, pRed, phinew, iflag, iter, cnt, params = trustregion_step_SPG2(x, val, dgrad, phi, problemTR, params, cnt)
      cnt = problemTR.obj_smooth.addCounter(cnt)
      cnt = problemTR.obj_nonsmooth.addCounter(cnt)
    return s, snorm, pRed, phinew, iflag, iter, cnt, params

# f_{i = {0, r}} -> each of these has a value and gradient and hessian.  


class modelTR:
    def __init__(self, problems, secant, subtype = 'spg', i = 0, R = np.empty(1), grad = np.empty(1), x = np.empty(1)):
        self.problemc = problems[i] #l+1 -> level we are going to 
        self.problemf = problems[0] #l+1 -> level we are going to 
        self.var     = problems[i].var
        self.secant  = secant
        self.l       = l
        self.x       = R @ x
        self.R       = R
        self.Rgrad   = problems[i].pvector.dual(R @ grad) #should be in dual space, dgrad in primal
        self.subtype = subtype
        self.nobj1   = 0
        self.ngrad   = 0
        self.nhess   = 0
        if subtype == 'recursive':
            grad, _      = problems[i].obj_smooth.gradient(R @ x, 0.)
            self.grad    = grad
            self.ngrad  += 1

    def update(self, x, type):
        self.problemc.obj_smooth.update(x, type)
        self.problemf.obj_smooth.update(x, type)
    def value(self, x, ftol):
        val, ferr    = self.problemc.obj_smooth.value(x, ftol) #ok since we evaluate the lower level model
        if self.subtype == 'recursive':
          #gradf, _  = self.problemf.obj_smooth.gradient(x, ftol) #evaluates high level gradient
          #self.Rgrad   = self.problemf.pvector.dual(R @ gradf) #should be in dual space, dgrad in primal
          val      += self.problem.pvector.apply(self.Rgrad - self.grad, x - self.x)
          ferr      = 0
        self.nobj1 += 0
        return val, ferr
    def gradient(self,x,gtol):
      # takes R\nabla f_{0} (x)
      grad, gerr   = self.problemf.obj_smooth.gradient(x, gtol)
      self.Rgrad   = R @ grad
      self.ngrad  += 0
      return self.Rgrad, gerr
    def hessVec(self,v,x,htol):
      if (self.secant):
        hv          = self.problem.secant.apply(v,self.problem.pvector,self.problem.dvector)
        herr        = 0
      else:
        hv, herr    = self.problem.obj_smooth.hessVec(v, x, htol)
        self.nhess += 1
      return hv, herr
    def addCounter(self,cnt):
        #actually zero because you should eval in subprob?
        cnt["nobj1"] += self.nobj1
        cnt["ngrad"] += self.ngrad
        cnt["nhess"] += self.nhess
        return cnt

class phiPrec: # you can definitely clean this up and inherit a bunch of stuff but
               # we can be explicit for now
    def __init__(self, problem, R = np.empty(1)):
        self.problem   = problem
        self.var       = problem.obj_nonsmooth.var
        self.R         = R
        self.nobj2     = 0
        self.nprox     = 0
    def value(self, x):
        val             = self.problem.obj_nonsmooth.value(self.R.T @ x)
        self.nobj2     += 1
        return val
    def prox(self, x, t):
        px          = x + self.R @ (self.problem.obj_nonsmooth.prox(self.R.T @ x, t) - self.R.T @ x)
        self.nprox += 1
        return px
    def addCounter(self, cnt):
        cnt["nobj2"] += self.nobj2
        cnt["nprox"] += self.nprox
        return cnt
    def genJacProx(self, x, t):
        D, ind = self.problem.obj_nonsmooth.genJacProx(x, t)
        return D, ind
    def applyProxJacobian(self, v, x, t):
        Dv = self.problem.obj_nonsmooth.applyProxJacobian(v, x, t)
        return Dv
    def getParameter(self):
        return self.problem.obj_nonsmooth.getParameter()

def trustregion(l, x0, Deltai, problems, params): #inpute Deltai
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
    if hasattr(problems[l].obj_smooth, 'begin_counter'):
        cnt = problems[l].obj_smooth.begin_counter(0, cnt)

    if params['initProx']:
        x = problems[l].obj_nonsmooth.prox(x0, 1)
        cnt['nprox'] += 1
    else:
        x = x0

    problems[l].obj_smooth.update(x, 'init')
    ftol = 1e-12
    if params['useInexactObj']:
        ftol = params['maxValTol']
    params['delta'] = min(params['delta'], Deltai)
    val, _      = problems[l].obj_smooth.value(x, ftol)

    cnt['nobj1'] += 1

    grad, _, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
    phi = problems[l].obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    if hasattr(problems[l].obj_smooth, 'end_counter'):
        cnt = problems[l].obj_smooth.end_counter(0, cnt)

    if params['useSecantPrecond']:
        problems[l].prec.apply         = lambda x: problems[l].secant.apply(x, problems[l].pvector, problems[l].dvector)
        problems[l].prec.apply_inverse = lambda x: problems[l].secant.apply_inverse(x, problems[l].pvector, problems[l].dvector)

    # Output header
    if l == 0:
      print(f"\nRecursive Nonsmooth Trust-Region Method using {params.get('spsolver', 'SPG2')} Subproblem Solver")
      print("level   iter          value           gnorm             del           snorm       nobjs      ngrad      nhess      nobjn      nprox    iterSP    flagSP")
      print(f"{0:4d}   {0:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}             ---      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}       ---      ---")

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
        cnt['iflag'] = 0
        return x, cnt

    # Iterate
    for i in range(1, params['maxit'] + 1):
        if hasattr(problems[l].obj_smooth, 'begin_counter'):
            cnt = problems[l].obj_smooth.begin_counter(i, cnt)

        # Solve trust-region subproblem
        params['tolsp'] = min(params['atol'], params['rtol'] * gnorm ** params['spexp'])
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step(l, x, val, grad, phi, problems, params, cnt)

        # Update function information
        xnew = x + s
        problems[l].obj_smooth.update(xnew, 'trial')
        valnew, val, cnt = compute_value(xnew, x, val, problems[l].obj_smooth, pRed, params, cnt)

        # Accept/reject step and update trust-region radius
        aRed = (val + phi) - (valnew + phinew)
        if aRed < params['eta1'] * pRed:
            params['delta'] = params['gamma1'] * min(snorm, params['delta'])
            problems[l].obj_smooth.update(x, 'reject')
            if params['useInexactGrad']:
                grad, dgrad, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
        else:
            x = xnew
            val = valnew
            phi = phinew
            problems[l].obj_smooth.update(x, 'accept')
            grad0 = grad
            grad, dgrad, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
            if aRed > params['eta2'] * pRed:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])
                params['delta'] = min(params['delta'], Deltai - problems[l].pvector.norm(x - x0))
            # Update secant
            if params['useSecant'] or params['useSecantPrecond']:
                y = grad - grad0
                problems.secant.update(s, y, problems[l].pvector, problems[l].dvector)
                if params['useSecantPrecond']:
                    problems[l].prec.apply = lambda x: problems[l].secant.apply(x, problems[l].pvector, problems[l].dvector)
                    problems[l].prec.apply_inverse = lambda x: problems[l].secant.apply_inverse(x, problems[l].pvector, problems[l].dvector)

        # Output iteration history
        if i % params['outFreq'] == 0:
            print(f"{l:4d}   {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")

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

        if hasattr(problems[l].obj_smooth, 'end_counter'):
            cnt = problems[l].obj_smooth.end_counter(i, cnt)

        # Check stopping criterion
        if gnorm <= gtol or snorm <= stol or i >= params['maxit']:
            if i % params['outFreq'] != 0:
                print(f"  {l:4d}   {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")
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
    cnt['iflag'] = flag
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

def driver(savestats, name):
    print("driver started")
    np.random.seed(0)

    # Set up optimization problem
    n = 1024  # Number of cells
    mu = 0.08  # Viscosity
    alpha = 1e-4  # L2 penalty parameter
    beta = 1e-2  # L1 penalty parameter
    usepc = True  # Use piecewise constant controls
    useInexact = False
    derivCheck = False
    meshlist = [n, int(n/2), int(n/4)]
    problems = [] #problem list goes from fine to coarse
    for i in range(0, len(meshlist)):
        B   = BurgersSetup(meshlist[i], mu=mu, alpha=alpha, beta=beta)
        var = B.returnVars(False)
        if i < len(meshlist)-1:
          R = restriction_R(meshlist[i+1], meshlist[i]) #puts R in preceeding problem
        else:
          R = np.eye(meshlist[i])
        p = Problem(var, R)
        problems.append(p)


    z = np.random.rand(n)
    u = np.zeros(n-1)
    x = np.hstack([u, z])

    # Parameters for the trust-region solver
    params = {
        'maxitsp': 10,
        'lam_min': 1e-12,
        'lam_max': 1e12,
        't': 1,
        'gtol': np.sqrt(np.finfo(float).eps),
        'delta': 1.0  # Trust-region radius
    }


    dim = n if usepc else n + 1

    if derivCheck:
        for i in range(0, len(meshlist)):
          x = np.random.randn(meshlist[i])
          d = np.random.randn(meshlist[i])
          obj = problems[i].obj_smooth.obj0
          con = problems[i].obj_smooth.con0
          deriv_check_simopt(np.zeros(problems[i].obj_nonsmooth.var['nu']), x, obj, con, 1e-4 * np.sqrt(np.finfo(float).eps))
          deriv_check(x, d, problems[i], 1e-4 * np.sqrt(np.finfo(float).eps))
          vector_check(x, d, problems[i])

    x0 = np.ones(dim)
    cnt = {}

    # Update default parameters
    params = set_default_parameters("SPG2")
    params["reltol"] = False
    params["t"] = 2 / alpha
    params["ocScale"] = 1 / alpha

    # Solve optimization problem
    start_time = time.time()
    for p in problems:
      p.obj_smooth.reset()
      p.obj_smooth.con0.reset()




    x, cnt_tr = trustregion(0, x0, params['delta'],problems, params)


    elapsed_time = time.time() - start_time

    print(f"Optimization completed in {elapsed_time:.2f} seconds")

    pro_tr = []
    for p in problems:
        pro_tr.append(p.obj_smooth.profile())

    cnt = (cnt_tr, pro_tr)

    print("\nSummary")
    print(
        "           niter     nobjs     ngrad     nhess     nobjn     nprox     nstat     nadjo     nssen     nasen"
    )

    print(
        f"   SGP2:  {cnt[0]['iter']:6d}    {cnt[0]['nobj1']:6d}    {cnt[0]['ngrad']:6d}    {cnt[0]['nhess']:6d}    "
        f"{cnt[0]['nobj2']:6d}    {cnt[0]['nprox']:6d}    {cnt[1][1]['nstate']:6d}    {cnt[1][1]['nadjoint']:6d}    "
        f"{cnt[1][1]['nstatesens']:6d}    {cnt[1][1]['nadjointsens']:6d}"
    )
    var = problems[0].obj_nonsmooth.var
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








