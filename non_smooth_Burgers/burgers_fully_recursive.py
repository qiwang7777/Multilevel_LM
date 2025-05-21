import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import scipy.sparse as sp
from scipy.sparse import spdiags, diags,lil_matrix
from scipy.integrate import quad
import time
from non_smooth.checks import deriv_check, deriv_check_simopt, vector_check
from non_smooth.setDefaultParameters import set_default_parameters
from non_smooth.Problem import Problem
from non_smooth.L1norm import L1Norm
from non_smooth.trustregion import trustregion

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
        p.obj_smooth    = ReducedObjective(Objective(var), ConstraintSolver(var))
        p.obj_nonsmooth = L1Norm(var)
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








