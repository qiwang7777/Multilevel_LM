import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import scipy.sparse as sp
from scipy.sparse import  diags, lil_matrix,coo_matrix, kron,eye,spdiags,block_diag,dok_matrix,csr_matrix
#from scipy.integrate import quad
#from scipy.sparse.linalg import spsolve
import time, torch
#from non_smooth.checks import deriv_check, deriv_check_simopt, vector_check
#from non_smooth.setDefaultParameters import set_default_parameters
from non_smooth.Problem import Problem
#from non_smooth.L1norm import L1NormEuclid as L1Norm

def set_default_parameters(name):
    params = {}

    # General Parameters
    params['spsolver']  = name.replace(' ', '')
    params['outFreq']   = 1
    params['debug']     = False
    params['initProx']  = False
    params['t']         = 1
    params['safeguard'] = np.sqrt(np.finfo(float).eps)

    # Stopping tolerances
    params['maxit']   = 200
    params['reltol']  = False
    params['gtol']    = 1e-8
    params['stol']    = 1e-10
    params['ocScale'] = params['t']

    # Trust-region parameters
    params['eta1']     = 0.05
    params['eta2']     = 0.9
    params['gamma1']   = 0.25
    params['gamma2']   = 2.5
    params['delta']    = 1.e2
    params['deltamax'] = 1e10

    # Subproblem solve tolerances
    params['atol']    = 1e-5
    params['rtol']    = 1e-2
    params['spexp']   = 1
    params['maxitsp'] = 15
    params['tolsp_floor'] = 1e-8

    # GCP and subproblem solve parameter
    params['useGCP']    = False
    params['mu1']       = 1e-4
    params['beta_dec']  = 0.1
    params['beta_inc']  = 10.0
    params['maxit_inc'] = 2

    # SPG and spectral GCP parameters
    params['lam_min'] = 1e-12
    params['lam_max'] = 1e12

    # Inexactness parameters
    params['useInexactObj']  = False
    params['useInexactGrad'] = False
    params['gradTol']        = np.sqrt(np.finfo(float).eps)

    # Recursive Parameters
    params['RgnormScale']      = 1e-1 # is v in Rgnorm >= v*gnorm -> relative R-step flag
    params['RgnormScaleTol']   = 5e2  # is v in Rgnorm >= v^i*gtol -> absolute R-step flag

    # Debug / diagnostics (all optional; default off)
    params.setdefault('debug_drop_gate', False)      # print drop-gate info each iter
    params.setdefault('debug_h_equiv', False)        # print h-equivalence diagnostic
    params.setdefault('debug_h_equiv_freq', 1)       # how often to print (every k iters)

    # Numerical guards/tolerances
    params.setdefault('prox_equiv_abs_tol', 1e-10)   # tight-frame prox identity tolerance
    params.setdefault('min_drop_cap', 1e-8)          # min parent cap to allow dropping

    # Existing gate scalars (set if not already)
    params.setdefault('RgnormScale', 1.0)            # κ_g
    params.setdefault('RgnormScaleTol', 100.0)       # ε_{i-1} scaling for coarse solve
    
    #child
    params.setdefault('last_step_from_child',True)
    params.setdefault('last_child_iflag',None)
    params.setdefault('prev_near_boundary', False)
    params.setdefault('nb_hit_valid_on_this_level',True)
    params.setdefault('grow_cooldown_k',0)





    return params
class L1Norm:
    def __init__(self, var,lo=-25.0,hi=25.0):
        self.var = var
        self.lo = lo
        self.hi = hi

    def value(self, x):
        return self.var.beta * np.dot(self.var.Rlump.T, np.abs(x))

    def prox(self, x, t):
        z_soft = np.maximum(0, np.abs(x) - t * self.var.Rlump * self.var.beta) * np.sign(x)
        return np.clip(z_soft,self.lo,self.hi)

    def dir_deriv(self, s, x):
        sx = np.sign(x)
        return self.var.beta * (np.dot(sx.T, s) + np.dot((1 - np.abs(sx)).T, np.abs(s)))

    def project_sub_diff(self, g, x):
        sx = np.sign(x)
        return self.var.beta * sx + (1 - np.abs(sx)) * np.clip(g, -self.var.beta, self.var.beta)

    def gen_jac_prox(self, x, t):
        d = np.ones_like(x)
        px = self.prox(x, t)
        ind = px == 0
        d[ind] = 0
        d[px <= self.lo + 0.0] = 0
        d[px >= self.hi - 0.0] =0
        return np.diag(d), (px==0)

    def apply_prox_jacobian(self, v, x, t):
        #ind = np.abs(x) <= t * self.var.M0 * self.var.beta
        px = self.prox(x, t)
        Dv = v.copy()
        mask_zero = (px == 0)
        mask_clip = (px <= self.lo + 0.0) | (px >= self.hi - 0.0)
        Dv[mask_zero | mask_clip] = 0
        
        return Dv

    def get_parameter(self):
        return self.var.beta

from non_smooth.modelTR import modelTR
#from non_smooth.subsolvers import trustregion_step_SPG2
def trustregion_gcp2(x,val,grad,dgrad,phi,problem,params,cnt):
  params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))  # Numerical safeguard
  params.setdefault('lam_min', 1e-12)
  params.setdefault('lam_max', 1e12)
  params.setdefault('t', 1)
  params.setdefault('t_gcp', params['t'])
  params.setdefault('gradTol', np.sqrt(np.finfo(float).eps)) # Gradient inexactness tolerance used for hessVec

  ## Compute Cauchy point as a single SPG step
  Hg,_          = problem.obj_smooth.hessVec(grad,x,params['gradTol'])
  cnt['nhess'] += 1
  gHg           = problem.dvector.apply(Hg, grad)
  gg            = problem.pvector.dot(grad, grad)
  if (gHg > params['safeguard'] * gg):
    t0Tmp = gg / gHg
  else:
    t0Tmp = params['t'] / np.sqrt(gg)
  t0     = np.min([params['lam_max'],np.max([params['lam_min'], t0Tmp])])
  xc     = problem.obj_nonsmooth.prox(x - t0 * dgrad, t0)
  cnt['nprox'] += 1
  s      = xc - x
  snorm  = problem.pvector.norm(s)
  Hs, _  = problem.obj_smooth.hessVec(s,x,params['gradTol'])
  cnt['nhess'] += 1
  sHs    = problem.dvector.apply(Hs,s)
  gs     = problem.pvector.dot(grad,s)
  phinew = problem.obj_nonsmooth.value(xc)
  cnt['nobj2'] += 1
  alpha  = 1
  if (snorm >= (1-params['safeguard'])*params['delta']):
    alpha = np.minimum(1, params['delta']/snorm)

  if sHs > params['safeguard']: #*problem.pvector.dot(s,s):
    alpha = np.minimum(alpha,-(gs+phinew-phi)/sHs) #min(alpha,max(-(gs + phinew - phi), snorm^2 / t0)/sHs);

  if (alpha != 1):
    s      *= alpha
    snorm  *= alpha
    gs     *= alpha
    Hs     *= alpha
    sHs    *= alpha**2
    xc     = x + s
    phinew = problem.obj_nonsmooth.value(xc)
    cnt['nobj2'] += 1

  valnew = val + gs + 0.5*sHs
  pRed   = (val+phi)-(valnew+phinew)
  params['t_gcp'] = t0
  return s, snorm, pRed, phi, Hs, cnt, params

def trustregion_step_SPG2(x, val,grad, dgrad, phi, problem, params, cnt):
    params.setdefault('maxitsp', 10)
    ## Cauchy point parameters
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1)
    params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))
    ## General parameters
    params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))  # Numerical safeguard
    params.setdefault('atol',  1e-4) # Absolute tolerance
    params.setdefault('rtol',  1e-2) # Relative tolerance
    params.setdefault('spexp',    2) # hk0 exponent

    x0    = copy.deepcopy(x)
    g0_primal    = copy.deepcopy(grad)
    snorm = 0

    # Evaluate model at GCP
    sHs    = 0
    gs     = 0
    valold = val
    phiold = phi
    hk0    = 0
    valnew = valold
    phinew = phiold

    [sc,snormc,pRed,_,_,cnt,params] = trustregion_gcp2(x,val,grad,dgrad,phi,problem,params,cnt)

    t0     = params['t']
    s      = copy.deepcopy(sc)
    x1     = x0 + s
    gnorm  = snormc
    gtol   = np.min([params['atol'], params['rtol']*(gnorm/t0)**params['spexp']])

    # Set exit flag
    iter  = 0
    iflag = 1

    for iter0 in range(1, params['maxitsp'] + 1):
        alphamax = 1
        snorm0   = snorm
        snorm    = problem.pvector.norm(x1 - x)

        if snorm >= (1 - params['safeguard'])*params['delta']:
            ds = problem.pvector.dot(s, x0 - x)
            dd = gnorm**2
            alphamax = np.minimum(1, (-ds + np.sqrt(ds**2 + dd * (params['delta']**2 - snorm0**2)))/dd)
        Hs, _  = problem.obj_smooth.hessVec(s,x,params['gradTol'])
        cnt['nhess'] += 1
        sHs    = problem.dvector.apply(Hs,s)
        g0s    = problem.pvector.dot(g0_primal,s)
        phinew = problem.obj_nonsmooth.value(x1)
        alpha0 = -(g0s + phinew - phiold) / sHs
        if sHs <= params['safeguard']: #*gnorm**2:
          alpha = alphamax
        else:
          alpha = np.minimum(alphamax,alpha0)
        ## Update iterate
        if (alpha == 1):
          x0     = x1
          g0_primal     += problem.dvector.dual(Hs)
          valnew = valold + g0s + 0.5 * sHs
        else:
          x0     += alpha*s
          g0_primal     += alpha*problem.dvector.dual(Hs)
          valnew = valold + alpha * g0s + 0.5 * alpha**2 * sHs
          phinew = problem.obj_nonsmooth.value(x0)
          snorm  = problem.pvector.norm(x0-x)

        ## Update model information
        valold = valnew
        phiold = phinew

        ## Check step size
        if snorm >= (1-params['safeguard'])*params['delta']:
          iflag = 2
          break

        # Update spectral step length
        if sHs <= params['safeguard']: #*gnorm**2:
          lambdaTmp = params['t']/problem.pvector.norm(g0_primal)
        else:
          lambdaTmp = gnorm**2/sHs

        t0 = np.max([params['lam_min'],np.min([params['lam_max'], lambdaTmp])])
        ## Compute step
        x1    = problem.obj_nonsmooth.prox(x0 - t0 * g0_primal, t0)
        s     = x1 - x0
        ## Check for convergence
        gnorm = problem.pvector.norm(s)
        if (gnorm/t0 <= gtol):
          iflag = 0
          break

    s    = x0 - x
    pRed = (val+phi) - (valnew+phinew)
    if (iter0 > iter):
       iter = iter0
    return s, snorm, pRed, phinew, iflag, iter, cnt, params
import copy
from non_smooth_Semilinear.quadpts import quadpts, sparse
import matplotlib.pyplot as plt
from typing import Optional

class PhiCounter:
    def __init__(self, base_phi):
        self.base = base_phi
        self.nobj2 = 0
        self.nprox = 0
    def value(self, x):
        v = self.base.value(x)
        self.nobj2 += 1
        return v
    def prox(self, u, t):
        p = self.base.prox(u, t)
        self.nprox += 1
        return p
    def gen_jac_prox(self, x, t):       return self.base.gen_jac_prox(x, t)
    def apply_prox_jacobian(self, v,x,t): return self.base.apply_prox_jacobian(v, x, t)
    def get_parameter(self):            return self.base.get_parameter()
    def addCounter(self, cnt):
        cnt["nobj2"] += self.nobj2
        cnt["nprox"] += self.nprox
        return cnt
    

class phiPrec:
    """
    Coarse-level nonsmooth φ_i built from previous level Problem {i-1}.
    
    Piecewise definition:
        ∙ anchor x_{i,0} = R x_{i-1,k} or x_{i,j}==x_{i,0} -> φ_i(x_{i,0}) = φ_{i-1}(x_{i-1,k})
        ∙ else                                             -> φ_i(x) = φ_{i-1}(R.T @ x)
    Prox:
        prox_{tφ_i}(u) = u + R^T @ (prox_{tφ_{i-1}}(Ru)-Ru)
        
    Parameters
    ----------
    problem: the previous level {i-1} Problem.
    R: ndarray of shape (n_i,n_{i-1}), Restriction matrix from level i-1 (fine) to level i (coarse)
    l: int, current level index i >= 0
    x_fine : ndarray | None , The special x_{i-1,k} (on level i-1) to define x_{i,0} = R @ x_fine
    x_tol: float, Tolerance for detecting x = x_{i,0}.
    assume_tight_rows: bool, If True, enable the closed-form prox for the composition branch.
    
    """
    def __init__(self, problems, R:np.ndarray,l: int, x_fine:Optional[np.ndarray]=None, x_tol:float=1e-12, assume_tight_rows:bool=True, use_anchor_in_prox:bool=False):
        self.problems = problems
        self.l = l
        self.R = None if self.l == 0 else R
        self.nobj2 = 0
        self.nprox = 0
        self.x_fine = None if x_fine is None else x_fine
        self.x_tol = float(x_tol)
        self.assume_tight_rows = bool(assume_tight_rows)
        self._x_coarse = None
        self.use_anchor_in_prox = bool(use_anchor_in_prox)
        if self.l>0 and self.R is not None and self.x_fine is not None:
            self._x_coarse = self.R @ self.x_fine
            
    def update_x(self,x_fine:np.ndarray) -> None:
        """
        Update the special fine point x_{i-1,k} and cache x_{i,0} = R @ x_fine

        """
        self.x_fine = x_fine
        if self.l>0 and self.R is not None:
            self._x_coarse = self.R @self.x_fine
        
    def _is_anchor(self,x:np.ndarray) -> bool:
        if self._x_coarse is None:
            return False
        if x.shape != self._x_coarse.shape:
            return False
        return np.allclose(x,self._x_coarse,atol = self.x_tol,rtol=0.0)
    
    def value(self,x:np.ndarray) -> float:
        """
       φ_{c}(x) = φ_{f}(x_f+R^T(x-x_c)) with x_c = Rx_f
     
        """
        if self.l == 0 or self.R is None:
            val = self.problems[0].obj_nonsmooth.value(x)
        else:
            if self._is_anchor(x):
                val = self.problems[self.l-1].obj_nonsmooth.value(self.x_fine)
            else:
                prev = self.problems[self.l-1].obj_nonsmooth
                x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
                y = self.x_fine +self.R.T @ (x-x_coarse)
                val = prev.value(y)
        self.nobj2 += 1
        return val
    def prox(self, u:np.ndarray,t:float) -> np.ndarray:
        """
        prox_{t φ_i}(u) with φ_i(x)=φ_{i-1}( x_fine + R^T (x - x_coarse) )
        Closed-form under RR^T=I:
          let v = u - x_coarse and y = x_fine + R^T v (fine-space),
          then
            z = v + R ( prox_{t φ_{i-1}}(y) - y )
            x = x_coarse + z

        """
        if not self.assume_tight_rows:
            raise NotImplementedError("Closed-form prox requires R R.T=I. Provided a numerical solver otherwise.")
        if self.l == 0 or self.R is None:
            px = self.problems[0].obj_nonsmooth.prox(u,t)
            self.nprox += 1
            return px
        
       
        else:
            
            n_coarse, n_fine = self.R.shape
            if u.shape[0] != n_coarse:
                 raise ValueError(
                    f"phiPrec(prox): level {self.l}: got u.shape[0]={u.shape[0]} but R is {n_coarse}×{n_fine}. "
                    f"u must be the **coarse** vector of length {n_coarse}. "
                    f"Likely the caller passed a fine-level vector or attached phiPrec to the wrong problem."
                )
        #else:
        #    if self._is_anchor(u):
        #        px = self.R @ self.problems[self.l-1].obj_nonsmooth.prox(self.x_fine, t)
        #    else:
        #        px = self.R @ self.problems[self.l-1].obj_nonsmooth.prox(self.R.T @ u ,t)
            prev = self.problems[self.l-1].obj_nonsmooth
            x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
            v = u - x_coarse
            y = self.x_fine +self.R.T @ v
            py = prev.prox(y,t)
            z = v + self.R @(py-y)
            xcomp = x_coarse+z
            if not self.use_anchor_in_prox or self._x_coarse is None:
                self.nprox +=1
                return xcomp
            Fcomp = 0.5 / t * np.linalg.norm(xcomp-u)**2 +prev.value(self.x_fine+self.R.T@(xcomp-x_coarse))
            Fanc = 0.5 / t * np.linalg.norm(self._x_coarse-u)**2+prev.value(self.x_fine)
            px = self._x_coarse if (Fanc +1e-12<Fcomp) else xcomp
            
            self.nprox += 1
            return px
    
    def addCounter(self, cnt):
        cnt["nobj2"] += self.nobj2
        cnt["nprox"] += self.nprox
        return cnt
    
    def genJacProx(self, x, t):
        if self.l == 0 or self.R is None:
            return self.problems[0].obj_nonsmooth.gen_jac_prox(x, t)
        prev = self.problems[self.l-1].obj_nonsmooth
        x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
        y = self.x_fine + self.R.T @ (x - x_coarse)
        D_prev, ind_prev = prev.gen_jac_prox(y, t)
        if D_prev is None: return None, None
        D_prev = np.asarray(D_prev)
        n_i = self.R.shape[0]
        J = np.eye(n_i) + self.R @ (D_prev - np.eye(D_prev.shape[0])) @ self.R.T
        ind = None
        if ind_prev is not None:
            ind = (np.abs(self.R @ ind_prev.astype(float)) > 0).ravel()
        return J, ind
    
    
    def applyProxJacobian(self, v, x, t):
        if self.l == 0 or self.R is None:
            return self.problems[0].obj_nonsmooth.apply_prox_jacobian(v,x,t)
        x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
        y = self.x_fine + self.R.T @ (x - x_coarse)
        Jprev_v = self.problems[self.l-1].obj_nonsmooth.apply_prox_jacobian(self.R.T @ v, y, t)
        #Dv = self.problems[self.l].obj_nonsmooth.applyProxJacobian(v, x, t)
        return v + self.R@(Jprev_v-self.R.T @ v)
    
    def getParameter(self):
        #return self.problem.obj_nonsmooth.getParameter()
        return (self.problems[0].obj_nonsmooth.get_parameter() 
                if self.l==0 else self.problems[self.l-1].obj_nonsmooth.get_parameter())

def trustregion_step(l, x, val, grad, problems, params, cnt, i=0):
    """
    One TR step on level l. Decides whether to recurse to l+1 or solve SPG at l.
    Returns: s, snorm, pRed, phinew, phi, iflag, iter_count, cnt, params
    """
    # --- basics ---
    L     = len(problems)
    t     = params['ocScale']
    dgrad = problems[l].dvector.dual(grad)

    # fine-level PG & norm h_{i,k}
    pgrad_f = problems[l].obj_nonsmooth.prox(x - t * dgrad, t)
    cnt['nprox'] += 1
    gnorm   = problems[l].pvector.norm(pgrad_f - x) / t
    params.setdefault('recurse_cooldown_N', 2)    # allow at most once every 2 its
    params.setdefault('recurse_cooldown_k', 0)
    if params['recurse_cooldown_k'] > 0:
        params['recurse_cooldown_k'] -= 1

    # edge restriction l -> l+1
    R_edge = problems[l].R
    xt     = R_edge @ x

    # --- coarse-side quantities (only if a coarser level exists) ---
    Rgnorm  = 0.0
    prox_ok = True
    if l < L - 1:
        u_coarse = xt - t * (R_edge @ dgrad)

        # φ_{i-1} as pullback via phiPrec (edge R, anchor at current fine x)
        phi_gate = phiPrec(problems, R=R_edge, l=l+1, x_fine=x, assume_tight_rows=True)
        pgrad_c  = phi_gate.prox(u_coarse, t)          # prox_{t φ_{i-1}}(u_coarse)
        cnt['nprox'] += 1

        # coarse PG norm h_{i-1,0}
        Rgnorm = problems[l+1].pvector.norm(pgrad_c - xt) / t

        # (tight-frame) prox identity diagnostic: should be tiny
        # shifted prox identity check (RR^T = I)
        x_coarse = phi_gate._x_coarse   # = R_edge @ x
        y        = x + R_edge.T @ (u_coarse - x_coarse)     # fine-space argument
        rhs      = u_coarse + R_edge @ (problems[l].obj_nonsmooth.prox(y, t) - y)
        prox_ok  = (np.linalg.norm(pgrad_c - rhs) <= params.get('prox_equiv_abs_tol', 1e-10))


        # optional scalar h-equivalence print
        if params.get('debug_h_equiv', False) and (i % params.get('debug_h_equiv_freq', 1) == 0):
            rhs_norm = problems[l+1].pvector.norm(R_edge @ (x - pgrad_f)) / t
            lhs_norm = problems[l+1].pvector.norm(xt - pgrad_c) / t
            denom    = max(lhs_norm, rhs_norm, 1e-16)
            rel_err  = abs(lhs_norm - rhs_norm) / denom
            #print(f"[h-eq l={l}] lhs={lhs_norm:.3e} rhs={rhs_norm:.3e} rel_err={rel_err:.2e}")

    # --- parent cap available to move on coarse level ---
    cap    = params.get('delta_effective', params['delta'])
    cap_ok = (cap > params.get('min_drop_cap', 1e-8))

    # --- drop gate (use function arg i; do NOT shadow it anywhere) ---
    #abs_gate = min(1e-2,params['gtol'] * np.sqrt(x.shape[0] / 2.0))
    gtol_gate = params.get('gtol_gate', params['gtol'])

    use_abs_dim_term = params.get('use_abs_dim_term', True)

    candidates = [params.get('abs_gate_min', 5e-5),params.get('abs_gate_frac', 0.6) * gnorm,]

    if use_abs_dim_term:
        candidates.append(gtol_gate * np.sqrt(x.shape[0] / 2.0))

    abs_gate = max(candidates)
    #abs_gate = max(params.get('abs_gate_min', 1e-7),params['gtol'] * np.sqrt(x.shape[0] / 2.0),params.get('abs_gate_frac', 0.3) * gnorm) 
    drop_rel = (Rgnorm >= params['RgnormScale'] * gnorm)
    drop_abs = (Rgnorm >= abs_gate)


    if not prox_ok:
        print(f"[PROX-EQ l={l}] ||pgrad_c - rhs||={np.linalg.norm(pgrad_c - rhs):.3e} "
              f"tol={params.get('prox_equiv_abs_tol', 1e-10):.1e}")
        
    print("debugging", "Rgnorm:",Rgnorm, "abs_gate:", abs_gate)

    do_recurse = (l < L - 1) and  drop_rel and drop_abs #and (params['recurse_cooldown_k'] == 0)
    
   


    if do_recurse:

        problemsL = copy.deepcopy(problems)
        #print(f"[REC l={l}] C: deepcopy done")
        params['recurse_cooldown_k'] = params['recurse_cooldown_N']

        # Build child problem on l+1
        p = Problem(problems[l+1].obj_nonsmooth.var, problems[l+1].R)
        p.obj_smooth    = modelTR(problems, params["useSecant"], 'recursive',
                                  l=l+1, R=R_edge, dgrad=R_edge @ dgrad, x=copy.deepcopy(xt))
        p.obj_nonsmooth = phiPrec(problems, R=R_edge, l=l+1, x_fine=x, assume_tight_rows=True)
        p.pvector       = problems[l+1].pvector
        p.dvector       = problems[l+1].dvector
        problemsL[l+1]  = p

        # evaluate model & phi at xt
        val, _ = p.obj_smooth.value(xt, 0.0); cnt['nobj1'] += 1
        phi    = p.obj_nonsmooth.value(xt);   cnt['nobj2'] += 1
        

        # child params (do not pollute parent)
        params_child          = copy.deepcopy(params)
        params_child['gtol']  = params['RgnormScaleTol'] * params['gtol']
        cap = params.get('delta_effective', params['delta'])
        params_child['delta'] = min(params_child['delta'], params_child['deltamax'], cap)
        params_child.pop('delta_effective', None)

        #print(f"[RECURSE→ l={l+1}] parent_cap={cap:.3e} child_init_δ={params_child['delta']:.3e}")


        # solve child TR
        xnew, cnt_coarse = trustregion(l+1, xt, cap, problemsL, params_child)




        # lift step back to level l
        s     = R_edge.T @ (xnew - xt)
        snorm = problems[l].pvector.norm(s)

        # evaluate new model & phi at xnew
        valnew, _ = p.obj_smooth.value(xnew, 0.0); cnt['nobj1'] += 1
        phinew    = p.obj_nonsmooth.value(xnew);   cnt['nobj2'] += 1

        pRed       = (val + phi) - (valnew + phinew)
        iflag      = cnt_coarse['iflag']
        iter_count = cnt_coarse['iter']

        # counters from the child objects
        cnt = p.obj_smooth.addCounter(cnt)
        cnt = p.obj_nonsmooth.addCounter(cnt)

    else:
        # -------- SPG / TAYLOR BRANCH (stay on level l) --------
        R_eye = Reye(x)  # identity on level l

        problemTR               = Problem(problems[l].var, R_eye)
        problemTR.obj_smooth    = modelTR(problems, params["useSecant"], 'spg',
                                          l=l, R=R_eye, dgrad=dgrad, x=x)
        problemTR.obj_nonsmooth = PhiCounter(problems[l].obj_nonsmooth)  # native φ_l + counters
        phi = problemTR.obj_nonsmooth.value(x)

        problemTR.pvector = problems[l].pvector
        problemTR.dvector = problems[l].dvector

        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val,grad, dgrad, phi, problemTR, params, cnt
        )
        cnt = problemTR.obj_smooth.addCounter(cnt)
        cnt = problemTR.obj_nonsmooth.addCounter(cnt)
        #params['nb_hit_valid_on_this_level'] = True   # step computed on this level

        #params['last_step_from_child'] = False
        #params['last_child_iflag'] = None


    # safety
    if pRed < 0 and np.abs(pRed) > 1e-5:
        #import pdb; pdb.set_trace()
        pRed = 1e-16

    return s, snorm, pRed, phinew, phi, iflag, iter_count, cnt, params
#Recursive step
def Reye(x):
    if isinstance(x,np.ndarray):
      matrix_R = np.eye(x.shape[0])
    else:
      matrix_R = x.clone()
      for k, v in matrix_R.td.items():
        n = x.td[k].size()[0]
        if len(x.td[k].size()) == 1:
          matrix_R.td[k] = torch.eye(n, n, dtype=torch.float64)
        else:
          m = x.td[k].size()[1]
          matrix_R.td[k] = torch.eye(m, m, dtype=torch.float64)
    return matrix_R

# --- debug helpers ---
def _near_boundary(snorm, delta_eff, frac=0.8):
    return snorm >= frac * max(1e-16, delta_eff)

def _good_quality(aRed, pRed, eta2):
    return aRed > eta2 * pRed

def _not_too_close_to_opt(gnorm, gtol, factor=10.0):
    return gnorm > factor * gtol


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
    params.setdefault('prev_near_boundary', False)

    # right after reading params and before the first print:
    if l == 0:
       
        

        params['delta'] = min(params['delta'], params['deltamax'])   # Δ_0,0 = min(Δ_0^s, +∞)
    else:
        # Deltai is Δ_{i+1} in your signature
        params['delta'] = min(params['delta'], params['deltamax'], Deltai)  # Δ_{i,0} = min(Δ_i^s, Δ_{i+1})


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
        'timehist': [],
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
        x = copy.deepcopy(x0)
    if l == 0:
      problems[l].obj_smooth.update(x, 'init')
    ftol = 1e-12
    if params['useInexactObj']:
        ftol = params['maxValTol']
    val, _      = problems[l].obj_smooth.value(x, ftol)

    cnt['nobj1'] += 1
    grad, _, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
    phi                 = problems[l].obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    if hasattr(problems[l].obj_smooth, 'end_counter'):
        cnt = problems[l].obj_smooth.end_counter(0, cnt)

    

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
    cnt['timehist'].append(np.nan)

    # Set optimality tolerance
    gtol = params['gtol']
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    # Check stopping criterion
    # if gnorm <= gtol:
    #     print('tr', gnorm, gtol, l)
    #     cnt['iflag'] = 0
    #     return x, cnt
    
        

    
    # Iterate
    for i in range(1, params['maxit'] + 1):
        if hasattr(problems[l].obj_smooth, 'begin_counter'):
            cnt = problems[l].obj_smooth.begin_counter(i, cnt)
            
        if l != 0:
            dist = problems[l].pvector.norm(x - x0)
            cap  = Deltai - dist
            if -1e-12 < cap < 0:   # tiny negative → 0
                cap = 0.0
            delta_eff = max(0.0, min(params['delta'], cap))
        else:
            delta_eff = min(params['delta'], params['deltamax'])

        params['delta_effective'] = delta_eff
        if params.get('verbose_child_debug',False):
            if l != 0:
                print(f"[lvl {l} iter {i}] dist = {dist:.3e} Deltai = {Deltai:.3e}"
                      f"delta={params['delta']:.3e} delta_eff={delta_eff:.3e}")
            else:
                print(f"[lvl {l} iter {i}] delta = {params['delta']}:.3e (fine level)")
                    
        #print(f"del={params['delta']:8.6e}", f"del_eff={params['delta_effective']:8.6e}")

        # Solve trust-region subproblem
        rel = params['rtol'] * gnorm ** params['spexp']
        tolsp = min(params['atol'], rel)
        tolsp = max(params.get('tolsp_floor', 1e-8), tolsp)   # <- add a floor
        params['tolsp'] = tolsp
        # -- DEBUG: pre-subproblem snapshot --
        #print(f"[TR l={l} i={i}] gnorm={gnorm:.3e} gtol={gtol:.3e} "
        #      f"δ={params['delta']:.3e} δ_eff={params['delta_effective']:.3e} tolsp={params['tolsp']:.3e}")
        s, snorm, pRed, phinew, phi, iflag, iter_count, cnt, params = trustregion_step(l, x, val, grad, problems, params, cnt, i=i-1)
        gTs = float(np.dot(grad,s))
        grad_piece = gTs+(phinew-phi)

        # Update function information
        xnew             = x + s
        problems[l].obj_smooth.update(xnew, 'trial')
        valnew, val, cnt = compute_value(xnew, x, val, problems[l].obj_smooth, pRed, params, cnt)

        # Accept/reject step and update trust-region radius
        aRed = (val + phi) - (valnew + phinew)
        rho   = aRed / max(1e-16, pRed)
        fracB = snorm / max(1e-16, params['delta_effective'])
        
        #print(f"{'':>4}   {'':>4}    {'ared':>12}    {'pred':>12}    {'g^Ts + Δφ':>12}")
        #print(f"{l:4d} {i:4d} {aRed:12.5e} {pRed:12.5e} {grad_piece:12.5e}")

        if aRed < params['eta1'] * pRed:
            params['delta'] = params['gamma1'] * min(snorm, params['delta'])
            problems[l].obj_smooth.update(x, 'reject')
            if params['useInexactGrad']:
                grad, _, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
        else:
            x   = xnew
            val = valnew
            phi = phinew
            problems[l].obj_smooth.update(x, 'accept')
            # grad0 = grad
            grad, _, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
            
            
            if (rho>params['eta2']) and (fracB>=0.9) and (gnorm>10*gtol):
               params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])
 


            
        if l != 0:
            dist = problems[l].pvector.norm(x - x0)
            residual = Deltai - dist
            if -1e-12 < residual < 0: residual = 0.0
            params['delta'] = min(params['delta'], max(0.0, residual))
          #params['delta'] = min(params['delta'], Deltai - problems[l].pvector.norm(x - x0))
         
        params['delta'] = max(0.0,min(params['delta'],params['deltamax']))

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
        cnt['timehist'].append(time.time() - start_time)

        if hasattr(problems[l].obj_smooth, 'end_counter'):
            cnt = problems[l].obj_smooth.end_counter(i, cnt)
            
        dist_now = problems[l].pvector.norm(x-x0)
        hit_cap = (l!=0) and (dist_now > (1 - .001)*Deltai)
        stop_cond = (
            (gnorm <= gtol) or
            (snorm <= stol) or
            (i >= params['maxit']) or
            hit_cap
            )
        if stop_cond:
            if i % params['outFreq'] != 0 :
                print(f"{l:4d} {i:4d} {val + phi:8.6e} {gnorm:8.6e} {params['delta']:8.6e} {snorm:8.6e} {cnt['nobj1']:6d} {cnt['ngrad']:6d } {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")
            if params.get('verbose_child_debug',False):
                print(f"[STOP lvl {l} iter {i}]"
                      f"gnorm={gnorm:.3e} snorm={snorm:.3e} dist={dist_now:.3e}"
                      f"Deltai={Deltai:.3e} hit_cap={hit_cap}")
            if gnorm <= gtol:
                flag = 0
            elif i > params['maxit']:
                flag = 1
            elif hit_cap:
                flag = 2
            else:
                flag = 3
            break

        # Check stopping criterion
        #if gnorm <= gtol or snorm <= stol or i >= params['maxit'] or (problems[l].pvector.norm(x - x0) > (1 - .001)*Deltai and l!=0):
        #    if i % params['outFreq'] != 0:
        #        print(f"  {l:4d}   {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")
        #    if gnorm <= gtol:
        #        flag = 0
        #    elif i >= params['maxit']:
        #        flag = 1
        #    elif problems[l].pvector.norm(x - x0) > (1 - .001)*Deltai:
        #        flag = 2
        #    else:
        #        flag = 3
        #    break

    cnt['iter'] = i
    cnt['timetotal'] = time.time() - start_time

    print("Optimization terminated because ", end="")
    if flag == 0:
        print("optimality tolerance was met")
    elif flag == 1:
        print("maximum number of iterations was met")
    elif flag == 2:
        print("finer trust-region radius met")
    else:
        print("step tolerance was met")
    if l==0:
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
#from non_smooth_Semilinear.quadpts import quadpts, sparse


from dataclasses import dataclass
from typing import Tuple
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
            pde_tol = max(1e-6, ftol)
            self.uwork, cnt0, serr = self.con0.solve(z, pde_tol)
            self.cnt['ninvJ1'] += cnt0
            self.is_state_computed = True
            self.cnt['nstate'] += 1
            ferr = max(ferr, serr)
        val, verr = self.obj0.value(np.hstack([self.uwork, z]), ftol)
        return val, max(ferr, verr)

    def gradient(self, z, gtol):
        if not self.is_state_computed or self.uwork is None:
            pde_tol = max(1e-6, gtol)
            self.uwork, cnt0, serr = self.con0.solve(z, pde_tol)
            self.cnt['ninvJ1'] += cnt0
            self.is_state_computed = True
            self.cnt['nstate'] += 1
        if not self.is_adjoint_computed or self.pwork is None:
            pde_tol = max(1e-6, gtol)
            rhs, aerr1 = self.obj0.gradient_1(np.hstack([self.uwork,z]), gtol)
            rhs = -rhs
            self.pwork, aerr2 = self.con0.apply_inverse_adjoint_jacobian_1(rhs, np.hstack([self.uwork, z]), pde_tol)
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






#New setupclass


@dataclass
class SemilinearSetup2D:
    """
    Finite element setup for the semilinear control problem.
    - Builds mesh, FreeNodes, FE matrices (A, M), and control couplings (B0, M0)
    - Caches geometry and quadrature for fast PDE solves
    - Keeps same attributes (alpha, beta, etc.) your other code expects
    """

    n: int          # number of subintervals in each direction (domain [0,1]^2)
    alpha: float    # Tikhonov weight in objective
    beta: float     # L1 weight (used by nonsmooth term)
    ctrl: int = 1   # not really used in the current formulation, kept for compatibility
    
    #noise controls
    noise_sigma: float = 0.0                    #std dev of Gaussian noise on target y_d
    rng: Optional[np.random.Generator] = None   #for reproducibility, pass same rng to all levels

    # Filled in __post_init__:
    mesh: object = None
    NT: int = 0
    N: int = 0
    Ndof: int = 0
    nu: int = 0

    dirichlet: np.ndarray = None
    FreeNodes: np.ndarray = None
    FreeMap: np.ndarray = None   # map global index -> local free index or -1

    # FE matrices, restricted to free nodes
    M: csr_matrix = None         # mass matrix on FreeNodes
    A: csr_matrix = None         # stiffness matrix on FreeNodes

    # Control discretization structures
    ctrl_disc: str = "pw_constant"
    B0: csr_matrix = None        # (nFree x NT) maps elementwise control z to interior RHS
    M0: csr_matrix = None        # (NT x NT) element mass (diag of triangle areas)
    Rlump: np.ndarray = None     # (NT,) lumped control weights (triangle areas)

    # PDE data
    uD: np.ndarray = None        # Dirichlet boundary values at ALL nodes (length N)
    b: np.ndarray = None         # baseline RHS on FreeNodes
    c: np.ndarray = None         # desired state on FreeNodes

    # Cached mesh/quad for nonlinear solves
    elem: np.ndarray = None      # (NT,3) triangle connectivity (global indices)
    area: np.ndarray = None      # (NT,) triangle areas
    lamb_q: np.ndarray = None    # (nQuad,3) barycentric basis funcs at quad points
    w_q: np.ndarray = None       # (nQuad,) quadrature weights

    def __post_init__(self):
        # ---- Build mesh on [0,1]x[0,1] with n x n subdivisions ----
        self.mesh = mesh2d(0, 1, 0, 1, self.n, self.n)

        self.elem = self.mesh.t.astype(int)   # (NT,3)
        self.NT   = int(self.elem.shape[0])   # #triangles
        self.N    = int(self.mesh.p.shape[0]) # #nodes
        self.Ndof = self.N
        self.nu   = self.N                    # state dofs = N nodal values

        # ---- Boundary conditions / FreeNodes ----
        # In your original code you forced all boundary edges to Dirichlet=1.
        self.mesh.e[:, 2] = 1
        self.dirichlet = np.unique(
            self.mesh.e[self.mesh.e[:,2] == 1, :2].astype(int)
        )
        all_nodes = np.arange(self.N, dtype=int)
        self.FreeNodes = np.setdiff1d(all_nodes, self.dirichlet)

        # Map global node -> index in FreeNodes (or -1 if Dirichlet)
        self.FreeMap = -np.ones(self.N, dtype=int)
        self.FreeMap[self.FreeNodes] = np.arange(self.FreeNodes.size, dtype=int)

        # ---- Finite element geometry ----
        # Dphi: gradients of basis functions on each triangle
        # area: triangle areas
        Dphi, area = gradbasis(self.mesh.p, self.elem)  # Dphi: (NT,2,3), area: (NT,)
        # Force areas positive in case of orientation flips
        neg = area < 0
        if np.any(neg):
            area = area.copy()
            area[neg] *= -1.0
        self.area = area

        # ---- Quadrature rule (cached for nonlinear assembly) ----
        lamb, weight = quadpts(7)          # lamb: (nQuad,3), weight: (nQuad,)
        self.lamb_q = lamb
        self.w_q    = weight

        # ---- Assemble global M and A once (vectorized COO->CSR) ----
        M_full, A_full = self._assemble_M_A(self.elem, Dphi, area, self.N)

        # Restrict to free nodes for the PDE solves
        self.M = M_full[self.FreeNodes, :][:, self.FreeNodes].tocsr()
        self.A = A_full[self.FreeNodes, :][:, self.FreeNodes].tocsr()

        # ---- Assemble control coupling ----
        # z is piecewise constant per triangle.
        B0_full, M0_full = self._assemble_B0_M0(self.elem, area, self.N, self.NT)

        # Only interior eqns depend on RHS, so restrict B0 to FreeNodes
        self.B0 = B0_full[self.FreeNodes, :].tocsr()
        self.M0 = M0_full.tocsr()

        # Lumped control mass (just element areas)
        self.Rlump = np.asarray(self.M0.sum(axis=1)).ravel()

        # ---- PDE data containers ----
        # uD = Dirichlet boundary values (0 for now)
        self.uD = np.zeros(self.N)
        # b = baseline RHS term (FreeNodes only)
        self.b  = np.zeros(self.FreeNodes.size)
        # c = desired state on FreeNodes 
        #self.c  = -np.ones(self.FreeNodes.size)
        # Build the desired state on FreeNodes.
        # Base profile: -1 everywhere in the interior
        base_target = -np.ones(self.FreeNodes.size)
        # Add Gaussian noise if requested
        if self.rng is None:
            # make a local, reproducible-ish generator if none is provided
            self.rng = np.random.default_rng(seed=0)
        if self.noise_sigma > 0.0:
            noise = self.rng.normal(loc=0.0, scale = self.noise_sigma,size=self.FreeNodes.size)
            self.c = base_target + noise
        else:
            self.c = base_target
            
        
        

    @staticmethod
    def _assemble_M_A(elem: np.ndarray,
                      Dphi: np.ndarray,
                      area: np.ndarray,
                      N: int) -> Tuple[csr_matrix, csr_matrix]:
        """
        Assemble global Mass (M_full) and Stiffness (A_full).
        elem: (NT,3) int
        Dphi: (NT,2,3)
        area: (NT,)
        N:    number of global nodes
        """
        NT = elem.shape[0]
        At = np.empty((NT,3,3), dtype=float)
        Mt = np.empty((NT,3,3), dtype=float)

        # local stiffness/mass matrices for each element
        for i in range(3):
            for j in range(3):
                # stiffness = ∫ grad φ_i · grad φ_j
                At[:, i, j] = (Dphi[:,0,i] * Dphi[:,0,j] +
                               Dphi[:,1,i] * Dphi[:,1,j]) * area
                # mass = ∫ φ_i φ_j = (area/12) * (2 if i==j else 1)
                Mt[:, i, j] = area * ((2.0 if i == j else 1.0) / 12.0)

        # expand into COO triplets
        rows = np.repeat(elem, 3, axis=1)   # (NT,9)
        cols = np.tile(elem, (1, 3))        # (NT,9)
        dataA = At.reshape(NT,9)
        dataM = Mt.reshape(NT,9)

        A_full = coo_matrix(
            (dataA.ravel(), (rows.ravel(), cols.ravel())),
            shape=(N, N)
        ).tocsr()

        M_full = coo_matrix(
            (dataM.ravel(), (rows.ravel(), cols.ravel())),
            shape=(N, N)
        ).tocsr()

        return M_full, A_full

    @staticmethod
    def _assemble_B0_M0(elem: np.ndarray,
                        area: np.ndarray,
                        N: int,
                        NT: int) -> Tuple[csr_matrix, csr_matrix]:
        """
        Control discretization as piecewise-constant per triangle.
        B0_full: maps elementwise z (size NT) to nodal load (size N)
                 by spreading area[k]/3 to each vertex.
        M0_full: diag of triangle areas.
        """
        rows = elem.reshape(-1)                      # (3*NT,)
        cols = np.repeat(np.arange(NT), 3)           # (3*NT,)
        vals = np.repeat(area / 3.0, 3)              # (3*NT,)

        B0_full = coo_matrix(
            (vals, (rows, cols)),
            shape=(N, NT)
        ).tocsr()

        M0_full = diags(area, offsets=0, shape=(NT, NT)).tocsr()
        return B0_full, M0_full

    def nonlin(self, x: np.ndarray):
        """
        Cubic nonlinearity used in the PDE:
            f(u) = u^3
            f'(u) = 3 u^2
            f''(u) = 6 u
        (Kept for compatibility with parts of your code that might still call it.)
        """
        u   = x**3
        du  = 3.0 * x**2
        duu = 6.0 * x
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





from scipy.sparse.linalg import spsolve

class SemilinearConstraintSolver2D:
    """
    Nonlinear PDE solver for the state equation:
        A(u_free) + N(u)_free = B0 z + b,
    where N(u)_free is the cubic nonlinearity integrated against test functions.

    This version:
    - Does Newton with backtracking.
    - At each Newton iteration, builds the Jacobian
         J_free = A + dN/du
      explicitly as a CSR matrix on FreeNodes.
    - Solves J_free s = r with a direct sparse solve (spsolve),
      which is fast up to ~O(1e5) unknowns in 2D.

    It keeps the same API that ReducedObjective expects:
    - solve(...)
    - value(...)
    - apply_jacobian_1(...)
    - apply_inverse_jacobian_1(...)
    etc.
    """

    def __init__(self, var: SemilinearSetup2D):
        self.var = var

        # last computed full state (length N)
        self.uprev = np.zeros(var.N)

        # convenient local handles
        self.elem   = var.elem        # (NT,3)
        self.area   = var.area        # (NT,)
        self.lamb   = var.lamb_q      # (nQuad,3)
        self.weight = var.w_q         # (nQuad,)
        self.FreeNodes = var.FreeNodes
        self.FreeMap   = var.FreeMap  # global -> local or -1

    # -------- utility: map between full and free vectors --------
    def _restrict_free(self, u_full: np.ndarray) -> np.ndarray:
        """Full (N,) -> interior/free (nFree,)"""
        return u_full[self.FreeNodes]

    def _extend_full(self, u_free: np.ndarray) -> np.ndarray:
        """Free (nFree,) -> full (N,), inserting Dirichlet boundary values (uD)."""
        u_full = self.var.uD.copy()
        u_full[self.FreeNodes] = u_free
        return u_full

    # -------- element-level assembly helpers --------
    def _nonlinear_element_loads(self, u_full: np.ndarray):
        """
        Compute per-element nodal contributions of ∫ u^3 φ_i.
        Returns fn_elem shape (NT,3).
        """
        elem   = self.elem      # (NT,3)
        area   = self.area      # (NT,)
        lamb   = self.lamb      # (nQuad,3)
        weight = self.weight    # (nQuad,)
        NT     = elem.shape[0]

        u_loc = u_full[elem]            # (NT,3)
        uhat  = u_loc @ lamb.T          # (NT,nQuad)

        u3 = uhat**3                    # (NT,nQuad)

        # fn_elem[k,i] = ∑_q area[k] * u^3(k,q) * w_q * lamb[q,i]
        fn_elem = np.einsum("kq,q,qi,k->ki", u3, weight, lamb, area)  # (NT,3)
        return fn_elem

    def _accumulate_to_global(self, elem: np.ndarray, contrib: np.ndarray, N: int) -> np.ndarray:
        """
        Scatter-add element contributions contrib (NT,3) into length-N global vector.
        contrib[k,i] adds to global node elem[k,i].
        """
        NT = elem.shape[0]
        out = np.zeros(N)
        # vectorized scatter-add using np.add.at
        np.add.at(out, elem.reshape(3*NT), contrib.reshape(3*NT))
        return out

    def _assemble_J_free(self, u_free: np.ndarray):
        """
        Assemble the Jacobian J_free = A + dN/du at current u_free.

        - A is already stored on FreeNodes: self.var.A (CSR)
        - dN/du is the nonlinear part: ∫ (3 u^2 φ_i φ_j)

        We build dN/du once (CSR on FreeNodes) and add it.
        """
        u_full = self._extend_full(u_free)

        elem   = self.elem
        area   = self.area
        lamb   = self.lamb
        weight = self.weight
        N      = self.var.N
        NT     = elem.shape[0]

        # interpolate u to quad points
        u_loc = u_full[elem]           # (NT,3)
        uhat  = u_loc @ lamb.T         # (NT,nQuad)
        du    = 3.0 * uhat**2          # (NT,nQuad)

        # build local nonlinear stiffness block:
        # Jnl[k,i,j] = ∑_q area[k] * (3 u^2)(k,q) * w_q * φ_i(q) * φ_j(q)
        Jnl = np.empty((NT,3,3), dtype=float)
        for i in range(3):
            phi_i_q = lamb[:, i]       # (nQuad,)
            for j in range(3):
                phi_j_q = lamb[:, j]
                # einsum over q: du(k,q) * w_q * phi_i_q * phi_j_q
                # then multiply by area[k]
                # shape result (NT,)
                Jnl[:, i, j] = np.einsum(
                    "kq,q,q,q->k",
                    du,
                    weight,
                    phi_i_q,
                    phi_j_q
                ) * area

        # Scatter this nonlinear part into an (N x N) sparse matrix
        rows = np.repeat(elem, 3, axis=1)   # (NT,9)
        cols = np.tile(elem, (1,3))         # (NT,9)
        data = Jnl.reshape(NT,9)

        Jnl_full = coo_matrix(
            (data.ravel(), (rows.ravel(), cols.ravel())),
            shape=(N, N)
        ).tocsr()

        # Restrict to FreeNodes × FreeNodes
        Jnl_free = Jnl_full[self.FreeNodes, :][:, self.FreeNodes].tocsr()

        # Add linear stiffness A (already restricted)
        J_free = self.var.A + Jnl_free
        return J_free

    # -------- residual & Newton solve --------
    def nonlinear_residual_free(self, u_free: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute residual r(y,z) on FreeNodes:
            r = A u_free + N(u)_free - (B0 z + b)
        """
        u_full = self._extend_full(u_free)

        # nonlinear element load vectors
        fn_elem = self._nonlinear_element_loads(u_full)  # (NT,3)
        Fnl_full = self._accumulate_to_global(self.elem, fn_elem, self.var.N)

        Fnl_free = Fnl_full[self.FreeNodes]
        rhs_ctrl = self.var.B0 @ z + self.var.b  # (nFree,)

        Au = self.var.A @ u_free

        r = Au + Fnl_free - rhs_ctrl
        return r

    def solve(self, z: np.ndarray, stol: float = 1e-12):
        """
        Solve the nonlinear PDE for state u given control z using Newton+line search.

        Returns:
            u_full_new : full state vector (length N)
            nlin_solve : number of linear solves performed
            serr       : final residual norm on FreeNodes
        """
        # warm-start from previous state
        u_full = self.uprev.copy()
        u_free = u_full[self.FreeNodes]

        nlin_solve = 0

        for _ in range(50):  # max Newton iterations
            # residual on FreeNodes
            r = self.nonlinear_residual_free(u_free, z)
            rn = np.linalg.norm(r)

            # stopping tolerance
            atol = stol
            rtol = 1.0
            ctol = min(atol, rtol * rn)
            if rn <= ctol:
                break

            # Assemble Jacobian J_free = A + dN/du
            J_free = self._assemble_J_free(u_free)

            # Solve J_free * s = r
            s = spsolve(J_free, r)
            nlin_solve += 1

            # Backtracking line search: try u_free - alpha*s
            alpha = 1.0
            while alpha > 1e-8:
                trial_free = u_free - alpha * s
                r_trial = self.nonlinear_residual_free(trial_free, z)
                if np.linalg.norm(r_trial) <= (1 - 1e-4 * alpha) * rn:
                    u_free = trial_free
                    break
                alpha *= 0.5

        # Update cached solution
        u_full_new = self._extend_full(u_free)
        self.uprev = u_full_new.copy()

        serr = np.linalg.norm(self.nonlinear_residual_free(u_free, z))
        return u_full_new, nlin_solve, serr

    # -------- API for ReducedObjective --------
    def begin_counter(self, iter_idx, cnt0):
        return cnt0

    def end_counter(self, iter_idx, cnt0):
        return cnt0

    def reset(self):
        self.uprev[:] = 0.0

    def value(self, x, vtol=1e-6):
        """
        Constraint residual F(y,z) on FreeNodes.
        x = [u, z] where u has length N and z has length NT.
        """
        nu = self.var.nu
        u_full = x[:nu]
        z      = x[nu:]
        u_free = u_full[self.FreeNodes]
        r = self.nonlinear_residual_free(u_free, z)
        return r, 0.0

    def apply_jacobian_1(self, v, x, gtol=1e-6):
        """
        Apply dF/dy [u] to direction v.
        (A + dN/du)*v, restricted to FreeNodes.
        We'll explicitly assemble J_free and multiply.
        """
        nu = self.var.nu
        u_full = x[:nu]
        u_free = u_full[self.FreeNodes]

        J_free = self._assemble_J_free(u_free)
        Jv = J_free @ v
        return Jv, 0.0

    def apply_jacobian_2(self, v, x, gtol=1e-6):
        """
        Apply dF/dz to v.
        F(y,z) = A*y + N(y) - (B0 z + b)
        so dF/dz = -B0.
        """
        return -(self.var.B0 @ v), 0.0

    def apply_adjoint_jacobian_1(self, v, x, gtol=1e-6):
        """
        Apply (dF/dy)^T to v.
        For this semilinear PDE, A + dN/du is symmetric,
        so (dF/dy)^T == (dF/dy).
        """
        return self.apply_jacobian_1(v, x, gtol)

    def apply_adjoint_jacobian_2(self, v, x, gtol=1e-6):
        """
        Apply (dF/dz)^T to v.
        dF/dz = -B0, so its adjoint is -(B0^T).
        """
        return -(self.var.B0.T @ v), 0.0

    def apply_inverse_jacobian_1(self, v, x, gtol=1e-6):
        """
        Solve (dF/dy) s = v for s.
        This means solve (A + dN/du) s = v, restricted to FreeNodes.
        """
        nu = self.var.nu
        u_full = x[:nu]
        u_free = u_full[self.FreeNodes]

        J_free = self._assemble_J_free(u_free)
        s = spsolve(J_free, v)
        return s, 0.0

    def apply_inverse_adjoint_jacobian_1(self, v, x, gtol=1e-6):
        """
        Solve (dF/dy)^T s = v. Symmetric => same solve.
        """
        return self.apply_inverse_jacobian_1(v, x, gtol)

    # Hessian terms in reduced gradient/Hessian.
    # For cubic nonlinearity, mixed control/state second derivatives vanish.
    def apply_adjoint_hessian_11(self, w, v, x, htol=1e-6):
        """
        Second derivative wrt state twice acting on (w,v).
        For your semilinear u^3, this shows up in reduced Hessian,
        but you were already returning something sparse/cheap.
        We'll just return 0 vector here for now if you're not using 2nd-order info.
        """
        return np.zeros_like(v), 0.0

    def apply_adjoint_hessian_12(self, u, v, x, htol=1e-6):
        return np.zeros(self.var.B0.shape[1]), 0.0

    def apply_adjoint_hessian_21(self, u, v, x, htol=1e-6):
        return np.zeros(self.var.FreeNodes.size), 0.0

    def apply_adjoint_hessian_22(self, u, v, x, htol=1e-6):
        return np.zeros(self.var.B0.shape[1]), 0.0







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

def plot_tr_history(cnt):
    import numpy as np
    import matplotlib.pyplot as plt

    it = np.arange(len(cnt.get('objhist', [])))

    # 1) Objective value (value + phi)
    plt.figure()
    plt.plot(it, cnt.get('objhist', []))
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.title('Objective vs. Iterations')
    plt.grid(True)

    # 2) Prox-gradient norm (gnorm)
    plt.figure()
    gnorm = np.array(cnt.get('gnormhist', []), dtype=float)
    plt.semilogy(it, gnorm + 1e-300)  # guard against zeros
    plt.xlabel('Iteration')
    plt.ylabel('||g|| (prox-grad)')
    plt.title('Prox-Gradient Norm vs. Iterations')
    plt.grid(True, which='both')

    # 3) Trust-region radius Δ (delta)
    plt.figure()
    plt.plot(it, cnt.get('deltahist', []))
    plt.xlabel('Iteration')
    plt.ylabel('Trust-Region Radius Δ')
    plt.title('TR Radius vs. Iterations')
    plt.grid(True)

    # 4) Step norm ||s||
    plt.figure()
    snorm = cnt.get('snormhist', [])
    plt.plot(it, snorm)
    plt.xlabel('Iteration')
    plt.ylabel('Step Norm ||s||')
    plt.title('Step Norm vs. Iterations')
    plt.grid(True)

    # 5) (Optional) Counters: nobj1, nobj2, ngrad, nprox cumulatives
    plt.figure()
    plt.plot(it, cnt.get('nobj1hist', []), label='nobj1 (smooth evals)')
    plt.plot(it, cnt.get('nobj2hist', []), label='nobj2 (nonsmooth evals)')
    plt.plot(it, cnt.get('ngradhist', []), label='ngrad (grad calls)')
    plt.plot(it, cnt.get('nproxhist', []), label='nprox (prox calls)')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative counts')
    plt.title('Work Counters vs. Iterations')
    plt.grid(True)
    plt.legend()

    plt.show()
    
def plot_compare_single_vs_double(n):
    # run 1-level
    _, cnt1 = driver(meshlist=[n], do_plots=False)

    # run 2-level
    _, cnt2 = driver(meshlist=[n,n//2,n//4,n//8], do_plots=False)

    

    it1 = np.arange(len(cnt1.get('objhist', [])))
    it2 = np.arange(len(cnt2.get('objhist', [])))

    # Objective
    plt.figure()
    plt.plot(it1, cnt1.get('objhist', []), label=f"1-level [{n}]")
    plt.plot(it2, cnt2.get('objhist', []), label=f"2-level [{n},{n//2}]")
    plt.xlabel('Iteration'); plt.ylabel('Objective (val + phi)')
    plt.title('Objective vs Iterations'); plt.grid(True); plt.legend()

    # Prox-gradient norm (log)
    g1 = np.asarray(cnt1.get('gnormhist', []), dtype=float)
    g2 = np.asarray(cnt2.get('gnormhist', []), dtype=float)
    plt.figure()
    plt.semilogy(it1, g1 + 1e-300, label=f"1-level [{n}]")
    plt.semilogy(it2, g2 + 1e-300, label=f"2-level [{n},{n//2}]")
    plt.xlabel('Iteration'); plt.ylabel('||prox-grad||')
    plt.title('Prox-Gradient Norm vs Iterations'); plt.grid(True, which='both'); plt.legend()

    # TR radius
    d1 = cnt1.get('deltahist', [])
    d2 = cnt2.get('deltahist', [])
    plt.figure()
    plt.plot(it1, d1, label=f"1-level [{n}]")
    plt.plot(it2, d2, label=f"2-level [{n},{n//2}]")
    plt.xlabel('Iteration'); plt.ylabel('Δ (trust-region radius)')
    plt.title('TR Radius vs Iterations'); plt.grid(True); plt.legend()

    # Step norm
    s1 = cnt1.get('snormhist', [])
    s2 = cnt2.get('snormhist', [])
    plt.figure()
    plt.plot(it1, s1, label=f"1-level [{n}]")
    plt.plot(it2, s2, label=f"2-level [{n},{n//2}]")
    plt.xlabel('Iteration'); plt.ylabel('||s||')
    plt.title('Step Norm vs Iterations'); plt.grid(True); plt.legend()

    plt.show()


def plot_obj_convergence(cnt, label=None):
    
    F = np.array(cnt['objhist'], dtype=float)
    Fstar = F[-1]
    it = np.arange(len(F))

    plt.figure()
    plt.semilogy(it, F - Fstar + 1e-16, linewidth=2)
    plt.xlabel('TR Iteration k')
    plt.ylabel(r'$F(x_k) - F^\star$')
    if label:
        plt.title(label)
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    
def plot_stationarity(cnt, label=None):
    import numpy as np
    import matplotlib.pyplot as plt
    g = np.array(cnt['gnormhist'], dtype=float)
    it = np.arange(len(g))

    plt.figure()
    plt.semilogy(it, g + 1e-30, linewidth=2)
    plt.xlabel('TR Iteration k')
    plt.ylabel(r'$\|g_k\|_\mathrm{prox}$')
    if label:
        plt.title(label)
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
def plot_radius(cnt, label=None):
    import numpy as np
    import matplotlib.pyplot as plt
    d = np.array(cnt['deltahist'], dtype=float)
    it = np.arange(len(d))

    plt.figure()
    plt.plot(it, d, linewidth=2)
    plt.xlabel('TR Iteration k')
    plt.ylabel(r'$\Delta_k$')
    if label:
        plt.title(label)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()


def plot_convergence_summary_single(cnt, label="Recursive TR (n=256)"):
    """
    Plot objective gap, stationarity norm, and trust-region radius
    in a single 3-panel figure.
    """
    # Extract histories
    F = np.array(cnt.get("objhist", []), dtype=float)
    g = np.array(cnt.get("gnormhist", []), dtype=float)
    d = np.array(cnt.get("deltahist", []), dtype=float)
    it = np.arange(len(F))
    
    # Reference final value
    Fstar = F[-1] if len(F) > 0 else 0.0
    
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)
    
    # --- (1) Objective gap ---
    axs[0].semilogy(it, np.maximum(F - Fstar, 1e-16), linewidth=2)
    axs[0].set_ylabel(r"$F(x_k) - F^\star$")
    axs[0].grid(True, which="both", alpha=0.4)
    axs[0].set_title(label, fontsize=12, fontweight='bold')
    
    # --- (2) Stationarity norm ---
    axs[1].semilogy(it, g + 1e-30, linewidth=2, color='tab:orange')
    axs[1].set_ylabel(r"$\|g_k\|_\mathrm{prox}$")
    axs[1].grid(True, which="both", alpha=0.4)
    
    # --- (3) Trust-region radius ---
    axs[2].plot(it, d, linewidth=2, color='tab:green')
    axs[2].set_xlabel("TR iteration $k$")
    axs[2].set_ylabel(r"$\Delta_k$")
    axs[2].grid(True, alpha=0.4)
    
    plt.tight_layout(h_pad=1.5)
    plt.show()


from matplotlib.patches import Patch
def add_hyperparam_legend(ax, alpha_val, beta_val):
    label = rf"$\alpha={alpha_val}$, $\beta={beta_val}$"
    dummy = Patch(facecolor="none", edgecolor="none")  # invisible handle
    leg = ax.legend([dummy], [label],
                    loc="upper left",
                    bbox_to_anchor=(0.02, 0.98),
                    bbox_transform=ax.transAxes,
                    frameon=True)
    leg.get_frame().set_alpha(0.9)

def driver(meshlist, savestats=True, name="semilinear_control_2d"):
    print("2D Driver started")
    np.random.seed(0)

    # Problem parameters
    #n = 128 # 32x32 grid
    alpha = 1e-4
    beta = 1e-2
    #meshlist = [n]
    n = meshlist[0]
    #meshlist = [n,n//2]
    problems = []
    shared_rng = np.random.default_rng(seed=12345)



    for i in range(len(meshlist)):
        S = SemilinearSetup2D(n=meshlist[i],alpha=alpha,beta=beta, ctrl=1, noise_sigma=0.5,rng=shared_rng)

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


    dim = 2*n*n
    #rng = np.random.default_rng(seed=123)
    #x0 = np.zeros(dim)
    #x0 = 0.1*rng.normal(size=dim)
    ys = np.linspace(0, 1, n)
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    z0 = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
    z0 = z0.ravel()  # control part
    y0 = np.zeros_like(z0)  # state part
    x0 = np.concatenate([y0, z0])


    params = set_default_parameters("SPG2")
    params.update({
        "reltol": False,
        "use_abs_dim_term": False,
        "t": 2/alpha,
        "ocScale":1/alpha,
        "maxiter":200,
        "verbose":True,
        "useReduced":False,
        "gtol":1e-6,
        "gtol_gate": 1e-6,
        "RgnormScale":0.01, # is v in Rgnorm >= v^i*gtol -> absolute R-step flag
        "RgnormScaleTol": 0.1,
        "debug_drop_gate":True,
        "debug_h_equiv":True,
        "debug_h_equiv_freq": 1,
        "prox_equiv_abs_tol": 1e-10,
        "min_drop_cap":1e-8,
        'deltamax': 1e4,
        'gamma2': 2.0,
        'eta2':0.95,
        "maxit":100,
        "verbose_child_debug":True
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
    # --- Target state (hide boundary) ---
    fig = plt.figure(figsize=(15,5))
    triangles = var.mesh.t  # (NT, 3) triangle connectivity
    num_nodes = var.mesh.p.shape[0]
    all_nodes = np.arange(num_nodes)
    boundary_nodes = np.setdiff1d(all_nodes, var.FreeNodes)

    # build y_d over all nodes, then mask boundary
    c_full = var.uD.copy()
    c_full[var.FreeNodes] = var.c
    c_plot = c_full.copy()
    c_plot[boundary_nodes] = np.nan  # mask boundary nodes

    ax = fig.add_subplot(131, projection='3d')
    ax.plot_trisurf(X1, X2, c_plot, triangles=triangles, cmap='viridis')
    plt.title("Target State $y_d$")

    
    #c = var.uD
    #c[var.FreeNodes] = var.c
    #ax = fig.add_subplot(131, projection='3d')
    #ax.plot_trisurf(X1, X2, c, cmap='viridis')
    #plt.title("Target State $y_d$")

    ax = fig.add_subplot(132, projection='3d')
    ax.plot_trisurf(X1, X2, optimal_state, cmap='viridis')
    plt.title("Optimal State $y$")
    add_hyperparam_legend(ax, alpha, beta)
    
    optimal_control = np.tile(x_opt,(1,3)).T.reshape(3*var.NT,)
    nodenew = var.mesh.p[var.mesh.t.reshape(3*var.NT,),:]
    ax = fig.add_subplot(133, projection='3d')
    ax.plot_trisurf(nodenew[:,0], nodenew[:,1], optimal_control, cmap='viridis')
    plt.title("Optimal Control $z$")
    add_hyperparam_legend(ax, alpha, beta)

    plt.tight_layout()
    plt.show()
    

    return x_opt, cnt_tr





def plot_convergence_summary(run_data, title="Convergence vs hierarchy depth"):
    """
    Compare multiple runs (e.g. 1-level vs 2-level) on:
      1. objective gap F(x_k)-F*
      2. prox-stationarity norm ||g||_prox
      3. trust-region radius Δ_k

    Parameters
    ----------
    run_data : list of dict-like
        Each element is a dict with:
            {
              "cnt": cnt_tr,             # output counter dict from trustregion()
              "label": "2-level",        # legend label
              "color": "tab:blue",       # (optional) matplotlib color
              "ls": "-",                 # (optional) line style
            }
        Only "cnt" and "label" are required.

    title : str
        Title for the whole figure.
    """

    fig, axs = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)

    # loop over runs
    for run in run_data:
        cnt   = run["cnt"]
        label = run.get("label", "run")
        color = run.get("color", None)
        ls    = run.get("ls", "-")

        F = np.array(cnt.get("objhist", []), dtype=float)
        g = np.array(cnt.get("gnormhist", []), dtype=float)
        d = np.array(cnt.get("deltahist", []), dtype=float)
        it = np.arange(len(F))

        if len(F) == 0:
            continue

        Fstar = F[-1]

        # (1) objective gap
        axs[0].semilogy(it,
                        np.maximum(F - Fstar, 1e-16),
                        linewidth=2,
                        label=label,
                        color=color,
                        ls=ls)

        # (2) stationarity norm
        axs[1].semilogy(it,
                        g + 1e-30,
                        linewidth=2,
                        label=label,
                        color=color,
                        ls=ls)

        # (3) trust-region radius
        axs[2].plot(it,
                    d,
                    linewidth=2,
                    label=label,
                    color=color,
                    ls=ls)

    # --- cosmetics / labels ---
    axs[0].set_ylabel(r"$F(x_k) - F^\star$")
    axs[0].grid(True, which="both", alpha=0.4)
    axs[0].set_title(title, fontsize=12, fontweight='bold')
    axs[0].legend(loc="best", fontsize=9)

    axs[1].set_ylabel(r"$\|g_k\|_\mathrm{prox}$")
    axs[1].grid(True, which="both", alpha=0.4)
    axs[1].legend(loc="best", fontsize=9)

    axs[2].set_xlabel(r"TR iteration $k$")
    axs[2].set_ylabel(r"$\Delta_k$")
    axs[2].grid(True, alpha=0.4)
    axs[2].legend(loc="best", fontsize=9)

    plt.tight_layout(h_pad=1.5)
    plt.show()




    
    
    

#plot_obj_convergence(cnt_tr, label="Recursive TR (n=2128)")
#plot_stationarity(cnt_tr, label="Recursive TR (n=128)")
#plot_radius(cnt_tr, label="Trust-Region Radius Evolution (n=256)")
n = 256
meshlist1 = [n]
meshlist2 = [n,n//2]
#meshlist3 = [n,n//2,n//4]
x_opt1,cnt_tr1 = driver(meshlist1)
x_opt2,cnt_tr2 = driver(meshlist2)
#x_opt3,cnt_tr3 = driver(meshlist3)
#plot_convergence_summary(cnt_tr, label="Recursive nonsmooth TR (n=128)")
plot_convergence_summary([
    {"cnt": cnt_tr1, "label": "1-level (fine only)", "color": "tab:blue", "ls": "-"},
    {"cnt": cnt_tr2, "label": "2-level",             "color": "tab:orange", "ls": "--"},
    #{"cnt": cnt_tr3, "label": "3-level",             "color": "tab:green", "ls": ":"},
], title="Convergence vs hierarchy depth (n=256)")
