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
from non_smooth.Problem import Problem
from non_smooth.L1norm import L1NormEuclid as L1Norm

#from non_smooth.L1norm import L1Norm
from non_smooth.trustregion import trustregion
from quadpts import quadpts, sparse
import matplotlib.pyplot as plt
import copy

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

class mesh2d:
#  Input
#          xmin, xmax  size of the rectangle
#          ymin, ymax
#          nx          number of subintervals on x-interval
#          ny          number of subintervals on y-interval
#
#  Output
#          mesh       structure array with the following fields
#
#          mesh.p     Real nn x 2
#                     array containing the x- and y- coordinates
#                     of the self.mesh.ps
#
#          mesh.t     Integer nt x 3
#                     t(i,1:3) contains the indices of the vertices of
#                     triangle i.
#
#          mesh.e     Integer nf x 3
#                     e(i,1:2) contains the indices of the vertices of
#                     edge i.
#                     edge(i,3) contains the boundary marker of edge i.
#                     Currently set to one.
#                     e(i,3) = 1  Dirichlet bdry conds are imposed on edge i
#                     e(i,3) = 2  Neumann bdry conds are imposed on edge i
#                     e(i,3) = 3  Robin bdry conds are imposed on edge i
#
#
#   Vertical ordering:
#   The triangles are ordered column wise, for instance:
#
#     03 -------- 06 -------- 09 -------- 12
#      |  4     /  |  8     /  | 12     /  |
#      |     /     |     /     |     /     |
#      |  /    3   |  /    7   |  /    11  |
#     02 -------- 05 -------- 08 -------- 11
#      |  2     /  |  6     /  | 10     /  |
#      |     /     |     /     |     /     |
#      |  /    1   |  /    5   |  /     9  |
#     01 -------- 04 -------- 07 -------- 10
#
#   The vertices and midpoints in a triangle are numbered
#   counterclockwise, for example
#           triangle 7: (05, 08, 09)
#           triangle 8: (05, 09, 06)
#
#   number of triangles: 2*nx*ny,
#   number of vertices:  (nx+1)*(ny+1),
#
#   AUTHOR:  Matthias Heinkenschloss
#            Department of Computational and Applied Mathematics
#            Rice University
#            November 23, 2005
    def __init__(self, xmin, xmax, ymin, ymax, nx, ny):
        nt = 2*nx*ny
        nP = (nx+1)*(ny+1)

        self.t = np.zeros((nt, 3))
        self.p = np.zeros((nP, 2))
        nxp1 = nx + 1
        nyp1 = ny + 1
        # Create triangles
        nt  = 0
        for ix in range(1, nx+1):
          for iy in range(1,ny+1):

              iv  = (ix-1)*nyp1 + iy
              iv1 = iv + nyp1

              nt           += 1
              self.t[nt-1,0]  = iv
              self.t[nt-1,1]  = iv1
              self.t[nt-1,2]  = iv1 + 1

              nt           += 1
              self.t[nt-1,0]  = iv
              self.t[nt-1,1]  = iv1 + 1
              self.t[nt-1,2]  = iv + 1
          # Create vertex coodinates

        hx   = (xmax-xmin)/nx
        hy   = (ymax-ymin)/ny
        x    = xmin
        for ix in range(1, nx+1):
          # set coordinates for vertices with fixed
          # x-coordinate at x
          i1 = (ix-1)*(ny+1) #+1
          i2 = ix*(ny+1)
          self.p[i1:i2,0] = x * np.ones((nyp1,))
          self.p[i1:i2,1] = np.arange(ymin, ymax+hy, hy).T #linespace?
          x += hx


        # set coordinates for vertices with fixed
        # x-coordinate at xmax
        i1 = nx*(ny+1) #+1
        i2 = (nx+1)*(ny+1)
        self.p[i1:i2,0] = xmax*np.ones((nyp1,))
        self.p[i1:i2,1] = np.arange(ymin, ymax+hy, hy).T


        # Set grid.edge (edges are numbered counter clock wise starting
        # at lower left end).

        self.e = np.ones((2*(nx+ny),3))

        # edges on left on left boundary
        self.e[0:ny,0] = np.arange(1, ny+1).T
        self.e[0:ny,1] = np.arange(2, ny+2).T

        # edges on top boundary
        self.e[ny:nx+ny,0] = np.arange(ny+1, nP , ny+1).T #translate below to linspace
        self.e[ny:nx+ny,1] = np.arange(2*(ny+1),nP+1, ny+1).T

        # edges on right boundary
        self.e[nx+ny:nx+2*ny,0] = np.arange(nP-ny, nP).T
        self.e[nx+ny:nx+2*ny,1] = np.arange(nP-ny+1, nP+1).T

        # edges on lower boundary
        self.e[nx+2*ny:2*(nx+ny),0] = np.arange(1, nP-2*ny, ny+1).T
        self.e[nx+2*ny:2*(nx+ny),1] = np.arange(ny+2, nP-ny+1, ny+1).T

        #grid correct up to here - subtract 1 for python indexing
        self.e -= 1
        self.t -= 1
        self.e = self.e.astype(int)
        self.t = self.t.astype(int)

def gradbasis(node, elem):
    NT = elem.shape[0]
    # $\nabla \phi_i = rotation(l_i)/(2|\tau|)$
    ve1 = node[elem[:,2],:] - node[elem[:,1],:]
    ve2 = node[elem[:,0],:] - node[elem[:,2],:]
    ve3 = node[elem[:,1],:] - node[elem[:,0],:]
    area = 0.5*(-ve3[:,0] * ve2[:,1] + ve3[:,1] * ve2[:,0])
    ##input dimensions
    Dphi = np.zeros((NT, 2, 3)) # is this right?
    Dphi[:NT,0,2] = -ve3[:,1] / (2*area)
    Dphi[:NT,1,2] = ve3[:,0] / (2*area)
    Dphi[:NT,0,0] = -ve1[:,1] / (2*area)
    Dphi[:NT,1,0] = ve1[:,0] / (2*area)
    Dphi[:NT,0,1] = -ve2[:,1] / (2*area)
    Dphi[:NT,1,1] = ve2[:,0] / (2*area)

    return Dphi, area


class SemilinearSetup2D:
    def __init__(self, n, alpha, beta, ctrl):
        self.mesh = mesh2d(0, 1, 0, 1, n, n)
        # self.mesh = mesh2d(0, 1, 0, 1, 4, 4)
        self.NT   = self.mesh.t.shape[0]
        self.N    = self.mesh.p.shape[0]
        self.Ndof = self.N
        self.ctrl = ctrl
        self.alpha = alpha
        self.beta = beta
        # Generate boundary data; reset all boundary markers to Dirichlet / Neumann
        self.mesh.e[:,2] = 1
        self.lamb        = 0.01

        # number of nodes
        N = np.max( np.max(self.mesh.t[:,:2]) )

        ## Initialization of free nodes.
        dirichlet = self.mesh.e[(self.mesh.e[:,2]==1).T,:2]
        self.dirichlet = np.unique( dirichlet )
        self.FreeNodes = np.setdiff1d(np.arange(0,N), self.dirichlet )


        ## Compute geometric quantities and gradient of local basis
        [Dphi,area] = gradbasis(self.mesh.p,self.mesh.t);  # using the code from iFEM by L. Chen
        #Dphi = Dlambda, area = area, node = mesh.p, elem = mesh.t

        # When the triangle is not positive orientated, we reverse the sign of the
        # area. The sign of Dlambda is always right since signed area is used in
        # the computation.

        idx           = (area<0)
        area[idx]     = -area[idx]
        self.area     = area
        elemSign      = np.ones((self.NT,1), dtype=int)
        elemSign[idx] = -1

        ## Generate stiffness and mass matrices
        Mt = np.zeros((self.NT,3,3))
        At = np.zeros((self.NT,3,3))
        for i in range(0,3):
            for j in range(0, 3):
                At[:,i,j] = (Dphi[:,0,i] * Dphi[:,0,j] + Dphi[:,1,i] * Dphi[:,1,j]) * area
                Mt[:,i,j] = area*((i==j)+1)/12

        ## Assemble the mass matrix in Omega
        M = sparse([], [], [], self.N,self.N)
        A = sparse([], [], [], self.N,self.N)
        for i in range(0, 3):
            krow = self.mesh.t[:,i]
            for j in range(0, 3):
                kcol = self.mesh.t[:,j]
                M = M + sparse(krow,kcol,Mt[:,i,j],self.N,self.N)
                A = A + sparse(krow,kcol,At[:,i,j],self.N,self.N)
        # clear At Mt
        A = A.toarray()
        M = M.toarray()
        ## Assemble mass matrices with picewise constants
        B0 = sparse([], [], [], N+1, self.NT)
        M0 = sparse([], [], [], self.NT, self.NT)
        for k in range(0, self.NT):
            # i = [self.mesh.t(k,1);mesh.t(k,2);mesh.t(k,3)]
            i = self.mesh.t[k,:].T
            Bt = area[k]/3 * np.ones((3,1))
            B0[i,k] += Bt
        # M0 = sparse(1:NT,1:NT,area,NT,NT)

        M0 = diags(area,shape=(self.NT, self.NT))
        self.M  = M[self.FreeNodes,:]
        self.M  = lil_matrix(self.M[:, self.FreeNodes])
        self.A  = A[self.FreeNodes,:]
        self.A  = lil_matrix(self.A[:, self.FreeNodes])

        self.ctrl_disc = 'pw_constant'
        self.za        = 0.
        self.zb        = 7.

        if self.ctrl_disc == 'pw_constant':
          self.B0 = B0[self.FreeNodes, :]
          self.M0 = M0

        elif self.ctrl_disc == 'pw_linear':
          self.M0 = M
          self.B0 = B0

        self.R             = np.squeeze(np.array(np.sum(self.M0, axis=1)))
        self.uD            = np.zeros(self.N,)
        self.uD[self.dirichlet] = 0. #self.exactu(self.mesh.p[self.dirichlet, :])
        self.b             = np.zeros(self.FreeNodes.shape[0],) #M[self.FreeNodes,:]@self.f(self.mesh.p, self.lamb) - A[self.FreeNodes,:]@self.uD
        self.c             = -np.ones(self.FreeNodes.shape[0],) #self.udesired(self.mesh.p[self.FreeNodes,:])
        self.nu            = self.N

# exact solution u
    # def exactu(self, p):
      # return np.sin(2*np.pi*p[:,0]) * np.sin(2*np.pi*p[:,1])
# right hand side
#     def f(self, p,lam):
#         t = 8*np.pi**2*(np.sin(2*np.pi*p[:,0]) * np.sin(2*np.pi*p[:,1])) + self.exactu(p)**3
#         if self.ctrl == 1:
#             t -= self.exactz(p,lam)
#         elif self.ctrl == 2:
#             t -= self.exactz_constrained(p, lam, self.za, self.zb)
#         return t
# # ud
#     def udesired(self, p):
#         return self.exactu(p) - (8*np.pi**2+3*self.exactu(p)**2)*self.exactp(p)
# # p
#     def exactp(self, p):
#         return self.exactu(p)
# # z (unconstrained case)
#     def exactz(self, p,lam):
#         t = self.exactp(p)
#         return -t/lam
# # z (constrained case)
#     def exactz_constrained(self, p,lamb,a,b):
#         t = self.exactp(p)
#         return np.minimum(b,np.maximum(-t/lamb,a))
# nonlinear function
    def nonlin(self, x):
        u  = x**3
        du = 3*x**2
        duu = 6*x
        return u, du, duu

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
        self.uprev = np.zeros(var.N)  # Previous state solution

    def begin_counter(self, iter, cnt0):
        return cnt0

    def end_counter(self, iter, cnt0):
        return cnt0

    def solve(self, z, stol=1e-12):
        """Solve the PDE constraint for given full x=[y,z] vector"""
        u = self.uprev
        unew = copy.deepcopy(u)
        c, _ = self.value(np.hstack([u,z]))
        cnt = 0
        atol = stol
        rtol = 1
        cnorm = np.linalg.norm(c)
        ctol = min(atol, rtol * cnorm)
        for _ in range(100):
            s,_ = self.apply_inverse_jacobian_1(self.value(np.hstack([u, z]))[0], np.hstack([u, z]))

            unew[self.var.FreeNodes] = u[self.var.FreeNodes] - s
            cnew = self.value(np.hstack([unew, z]))[0]
            ctmp = np.linalg.norm(cnew)

            alpha = 1
            while ctmp > (1 - 1e-4 * alpha) * cnorm:
                alpha *= 0.1
                unew[self.var.FreeNodes]  = u[self.var.FreeNodes]  - alpha * s
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
        self.uprev = np.zeros(self.var.N,)

    def value(self, x, vtol=1e-6):
        """Constraint evaluation F(y,z) = A*y + y^3 - z"""
        nu = self.var.nu
        u, z = x[:nu], x[nu:]
        (Nu,_, _) = self.evaluate_nonlinearity(u)
        c = self.var.A @ u[self.var.FreeNodes] + Nu[self.var.FreeNodes] - (self.var.B0 @ z + self.var.b)
        return c, 0.

    def apply_jacobian_1(self, v, x, gtol=1e-6):
        """Apply Jacobian [J1, J2] where J1 = A + 3*diag(y^2), J2 = -I"""
        nu = self.var.nu
        u = x[:nu]
        (_, J, _) = self.evaluate_nonlinearity(u)
        J = J[self.var.FreeNodes,:]
        J = J[:, self.var.FreeNodes]
        return  (self.var.A + J) @ v, 0

    def apply_jacobian_2(self, v, x, gtol=1e-6):
        """Apply adjoint Jacobian [J1^T; J2^T]"""
        hv = -self.var.B0 @ v
        return hv,  0.

    def apply_adjoint_jacobian_1(self, v, x,gtol=1e-6):
        nu = self.var.nu
        u = x[:nu]
        (_,J,_) = self.evaluate_nonlinearity(u)
        J = J[self.var.FreeNodes,:]
        J = J[:, self.var.FreeNodes]
        return (self.var.A + J).T @ v, 0

    def apply_adjoint_jacobian_2(self, v,x,gtol=1e-6):
        hv = -self.var.B0.T @ v
        return  hv, 0

    def apply_inverse_jacobian_1(self, v, x,gtol=1e-6):
        nu = self.var.nu
        u = x[:nu]

        (_,J,_) = self.evaluate_nonlinearity(u)
        J = J[self.var.FreeNodes,:]
        J = J[:, self.var.FreeNodes]
        solution = sp.linalg.spsolve(self.var.A+J,v )
        return solution, 0

    def apply_inverse_adjoint_jacobian_1(self, v, x,gtol=1e-6):
        nu = self.var.nu
        u = x[:nu]
        (_, J, _) = self.evaluate_nonlinearity(u)
        J = J[self.var.FreeNodes,:]
        J = J[:, self.var.FreeNodes]
        return sp.linalg.spsolve((self.var.A + J).T, v), 0

    def apply_adjoint_hessian_11(self, w, v, x, htol=1e-6):
        nu = self.var.nu
        u = x[:nu]
        (_, _, D) = self.evaluate_nonlinearity(u, pt = v)
        D = D[self.var.FreeNodes,:]
        D = D[:, self.var.FreeNodes]
        return D @ w, 0.

    def apply_adjoint_hessian_12(self, u, v, x,htol=1e-6):
        return np.zeros(self.var.B0.shape[1]), 0

    def apply_adjoint_hessian_21(self, u, v, x,htol=1e-6):
        return np.zeros(self.var.FreeNodes.shape[0]), 0

    def apply_adjoint_hessian_22(self, u, v, x,htol=1e-6):
        return np.zeros(self.var.B0.shape[1]), 0

    def evaluate_nonlinearity(self, uh, pt=None):
      ## quadrature points and weights
      (lamb,weight) = quadpts(7)
      nQuad = lamb.shape[0]
      elem = self.var.mesh.t
      node = self.var.mesh.p

      ## assemble nonlinearities
      fn  = np.zeros((self.var.NT,3))
      Dfn = np.zeros((self.var.NT,3,3))
      if pt is not None:
        ph   = np.zeros(self.var.mesh.p.shape[0],)
        ph[self.var.FreeNodes] = pt
        DDfn = np.zeros((self.var.NT, 3, 3))
      else:
        DDfn = None
      for p in range(0, nQuad):
          #evaluate uh at quadrature point
          uhp = uh[elem[:,0]]*lamb[p,0] + uh[elem[:,1]]*lamb[p,1] + uh[elem[:,2]]*lamb[p,2]
          if pt is not None:
            php = ph[elem[:,0]]*lamb[p,0] + ph[elem[:,1]]*lamb[p,1] + ph[elem[:,2]]*lamb[p,2]

          (non,dnon, ddnon) = self.var.nonlin(uhp)
          for i in range(0,3):
              for j in range(0,3):
                  Dfn[:,i,j] += self.var.area*weight[p]*dnon*lamb[p,j]*lamb[p,i]
                  if pt is not None:
                      DDfn[:, i, j] += self.var.area * weight[p] * ddnon * php * lamb[p, j] * lamb[p,i]
              fn[:,i] += self.var.area*weight[p]*non*lamb[p,i]

      Newt = np.zeros((self.var.Ndof,))
      DNewt = sparse([], [], [], self.var.N,self.var.N)
      if pt is not None:
          DDNewt = sparse([], [], [], self.var.N, self.var.N)
      else:
          DDNewt = None
      for i in range(0,3):
          krow = elem[:,i]
          for j in range(0,3):
              kcol = elem[:,j]
              DNewt += sparse(krow,kcol,Dfn[:,i,j],self.var.N,self.var.N)
              if pt is not None:
                  DDNewt += sparse(krow, kcol, DDfn[:, i, j], self.var.N, self.var.N)
      # Newt +=  accumarray(elem(:),fn(:),[Ndof,1]);
      temp = np.bincount(elem.reshape(np.prod(elem.shape)), weights=fn.reshape(np.prod(fn.shape))).T
      Newt +=  temp #vectorize elem and fn
      return Newt, DNewt, DDNewt



class SemilinearObjective2D:
    def __init__(self, var):
        self.var = var

    def update(self,x,type):
        return None

    def value(self, x, ftol=1e-6):
        y = x[:self.var.nu]  # State portion
        z = x[self.var.nu:]  # Control portion
        diff = y[self.var.FreeNodes]  - self.var.c
        # Ensure proper matrix-vector multiplication
        term1 = 0.5 * diff.T @ (self.var.M @ diff)
        term2 = 0.5 * self.var.alpha * (z.T @ self.var.M0 @ z)

        return term1 + term2, 0

    def gradient_1(self, x, gtol=1e-6):
        u = x[:self.var.nu]
        gradu = self.var.M @ (u[self.var.FreeNodes]  - self.var.c) # Simplified for diagonal M
        return gradu, 0

    def gradient_2(self, x, gtol=1e-6):
        z = x[self.var.nu:]
        gradu = self.var.alpha * self.var.M0 @ z # Simplified for diagonal M
        return gradu, 0

    def hessVec_11(self, v, x, htol):
        hv = self.var.M @ v
        return hv, 0.

    # Apply objective function Hessian to a vector (hessVec_12)
    def hessVec_12(self, v, x, htol):
        hv   = np.zeros((self.var.B0.shape[0],))
        return hv, 0.

    # Apply objective function Hessian to a vector (hessVec_21)
    def hessVec_21(self, v, x, htol):
        hv   = np.zeros((self.var.B0.shape[1],))
        return hv, 0.

    # Apply objective function Hessian to a vector (hessVec_22)
    def hessVec_22(self, v, x, htol):
        hv = self.var.alpha * (self.var.M0 @ v)
        return hv, 0.

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
    n = 128  # 32x32 grid
    alpha = 1e-4
    beta = 1e-2
    # meshlist = [n]
    meshlist = [n,n//2]
    problems = []
    # if (strcmp(GLB.ctrl_disc,'pw_constant')):
      #  z = zeros(NT,1)
    # elif (strcmp(GLB.ctrl_disc,'pw_linear')):
      #  z = zeros(length(GLB.FreeNodes),1)


    for i in range(len(meshlist)):
        S = SemilinearSetup2D(meshlist[i],alpha,beta, 1)

        if i < len(meshlist)-1:
            R = restriction_R_2d(meshlist[i+1],meshlist[i])
        else:
            R = sp.eye(2*meshlist[i]*meshlist[i],format='csc')

    #Verify dimensions
        assert R.shape == (2*meshlist[i+1]**2,2*meshlist[i]**2) if i <len(meshlist)-1 else (2*meshlist[i]**2,2*meshlist[i]**2)

        p = Problem(S,R)
        p.obj_smooth    = ReducedObjective(SemilinearObjective2D(S), SemilinearConstraintSolver2D(S))
        p.obj_nonsmooth = L1Norm(S)
        problems.append(p)
        # dim = S.NT
        # x   = np.ones(dim)
        # d   = np.ones(dim)
        # deriv_check(x, d, problems[i], 1e-4 * np.sqrt(np.finfo(float).eps))
        # vector_check(x, d, problems[i])

    dim = 2*n*n
    x0 = np.zeros(dim)


    params = set_default_parameters("SPG2")
    params.update({
        "reltol": False,
        "t": 2/alpha,
        "ocScale":1/alpha,
        "maxiter":100,
        "verbose":True,
        "useReduced":False,
        "gtol":1e-6,
        "RgnormScale":1e0 # is v in Rgnorm >= v^i*gtol -> absolute R-step flag
        })
    # Solve optimization problem
    start_time = time.time()
    x_opt, cnt_tr = trustregion(0, x0, params['delta'], problems, params)
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")

    # Extract results
    var = problems[0].obj_nonsmooth.var

    optimal_state = problems[0].obj_smooth.con0.uprev
    X1, X2 = var.mesh.p[:,0], var.mesh.p[:,1]

    # Plot results
    fig = plt.figure(figsize=(15,5))
    c = var.uD
    c[var.FreeNodes] = var.c
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_trisurf(X1, X2, c, cmap='viridis')
    plt.title("Target State $y_d$")

    ax = fig.add_subplot(132, projection='3d')
    ax.plot_trisurf(X1, X2, optimal_state, cmap='viridis')
    plt.title("Optimal State $y$")
    optimal_control = np.tile(x_opt,(1,3)).T.reshape(3*var.NT,)
    nodenew = var.mesh.p[var.mesh.t.reshape(3*var.NT,),:]
    ax = fig.add_subplot(133, projection='3d')
    ax.plot_trisurf(nodenew[:,0], nodenew[:,1], optimal_control, cmap='viridis')
    plt.title("Optimal Control $z$")

    plt.tight_layout()
    plt.show()

    return x_opt, cnt_tr



cnt = driver()

