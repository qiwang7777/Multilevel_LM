import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import spdiags, diags,lil_matrix
from scipy.integrate import quad
import time
from non_smooth.checks import deriv_check, deriv_check_simopt, vector_check

from non_smooth.Problem import Problem
from non_smooth.L1norm import L1Norm
from non_smooth.modelTR import modelTR
from non_smooth.subsolvers import trustregion_step_SPG2
import copy
import matplotlib.pyplot as plt
#from non_smooth.trustregion import trustregion
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
    params['delta']    = 100
    params['deltamax'] = 1e10

    # Subproblem solve tolerances
    params['atol']    = 1e-5
    params['rtol']    = 1e-3
    params['spexp']   = 2
    params['maxitsp'] = 15

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
        Piecewise value:
            ∙ anchor -> φ_{i-1}(x_{i-1,k})
            ∙ else   -> φ_{i-1}(R.T@x)
        """
        if self.l == 0 or self.R is None:
            val = self.problems[0].obj_nonsmooth.value(x)
        else:
            if self._is_anchor(x):
                val = self.problems[self.l-1].obj_nonsmooth.value(self.x_fine)
            else:
                val = self.problems[self.l-1].obj_nonsmooth.value(self.R.T @ x)
        self.nobj2 += 1
        return val
    def prox(self, u:np.ndarray,t:float) -> np.ndarray:
        """
        Prox for the composition branch (global; anchor is a pointwise value condition).
        prox_{tφ_i}(u) = u + R^T @ (prox_{tφ_{i-1}}(Ru)-Ru)

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
            RuT = self.R.T @ u
            pRuT = self.problems[self.l-1].obj_nonsmooth.prox(RuT,t)
            xcomp = u + self.R @ (pRuT-RuT)
            if not self.use_anchor_in_prox or self._x_coarse is None:
                self.nprox += 1
                return xcomp
            xanc = self._x_coarse
            # evaluate piecewise objective values
            prev = self.problems[self.l-1].obj_nonsmooth
            Fcomp = 0.5/t * np.linalg.norm(xcomp - u)**2 + prev.value(self.R.T @ xcomp)
            Fanc  = 0.5/t * np.linalg.norm(xanc  - u)**2 + prev.value(self.x_fine)

            # pick better (add a tiny tie tolerance)
            if Fanc + 1e-12 < Fcomp:
                px = xanc
            else:
                px = xcomp

            self.nprox += 1
            return px
    
    def addCounter(self, cnt):
        cnt["nobj2"] += self.nobj2
        cnt["nprox"] += self.nprox
        return cnt
    
    def genJacProx(self, x, t):
        if self.l == 0 or self.R is None:
            return self.problems[0].obj_nonsmooth.gen_jac_prox(x,t)
        if self.l > 0:
            D_prev, ind_prev = self.problems[self.l-1].obj_nonsmooth.gen_jac_prox(self.R.T @ x , t)
            if D_prev is None:
                return None, None
            D_prev = np.asarray(D_prev)
            n_i = self.R.shape[0]
            J = np.eye(n_i)+self.R@(D_prev-np.eye(D_prev.shape[0]))@self.R.T
            ind = None
            if ind_prev is not None:
                ind = (np.abs(self.R @ ind_prev.astype(float))>0).ravel()
        #D, ind = self.problems[self.l].obj_nonsmooth.genJacProx(x, t)
        return J, ind
    
    def applyProxJacobian(self, v, x, t):
        if self.l == 0 or self.R is None:
            return self.problems[0].obj_nonsmooth.apply_prox_jacobian(v,x,t)
        Rtv = self.R.T @ v
        Jprev = self.problems[self.l-1].obj_nonsmooth.apply_prox_jacobian(Rtv,self.R.T@x,t)
        #Dv = self.problems[self.l].obj_nonsmooth.applyProxJacobian(v, x, t)
        return v+self.R@(Jprev-Rtv)
    
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
        RuT  = R_edge.T @ u_coarse                     # = P @ (x - t dgrad)
        rhs  = u_coarse + R_edge @ (problems[l].obj_nonsmooth.prox(RuT, t) - RuT)
        prox_ok = (np.linalg.norm(pgrad_c - rhs) <= params.get('prox_equiv_abs_tol', 1e-10))

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
    abs_gate = max(params.get('abs_gate_min', 5e-5),params['gtol'] * np.sqrt(x.shape[0] / 2.0),params.get('abs_gate_frac', 0.6) * gnorm) 
    drop_rel = (Rgnorm >= params['RgnormScale'] * gnorm)
    drop_abs = (Rgnorm >= abs_gate)
    if not prox_ok:
        print(f"[PROX-EQ l={l}] ||pgrad_c - rhs||={np.linalg.norm(pgrad_c - rhs):.3e} "
              f"tol={params.get('prox_equiv_abs_tol', 1e-10):.1e}")

    do_recurse = (l < L - 1) and (i > 0) and prox_ok and cap_ok and drop_rel and drop_abs and (params['recurse_cooldown_k'] == 0)
    


    if do_recurse:
        
        # -------- RECURSIVE BRANCH (to level l+1) --------
        problemsL = copy.deepcopy(problems)
        
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
            x, val, dgrad, phi, problemTR, params, cnt
        )
        cnt = problemTR.obj_smooth.addCounter(cnt)
        cnt = problemTR.obj_nonsmooth.addCounter(cnt)



    # safety
    if pRed < 0 and np.abs(pRed) > 1e-5:
        import pdb; pdb.set_trace()

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
        #print(f"del={params['delta']:8.6e}", f"del_eff={params['delta_effective']:8.6e}")

        # Solve trust-region subproblem
        params['tolsp']                                        = min(params['atol'], params['rtol'] * gnorm ** params['spexp'])
        # -- DEBUG: pre-subproblem snapshot --
        #print(f"[TR l={l} i={i}] gnorm={gnorm:.3e} gtol={gtol:.3e} "
        #      f"δ={params['delta']:.3e} δ_eff={params['delta_effective']:.3e} tolsp={params['tolsp']:.3e}")
        s, snorm, pRed, phinew, phi, iflag, iter_count, cnt, params = trustregion_step(l, x, val, grad, problems, params, cnt, i=i-1)

        # Update function information
        xnew             = x + s
        problems[l].obj_smooth.update(xnew, 'trial')
        valnew, val, cnt = compute_value(xnew, x, val, problems[l].obj_smooth, pRed, params, cnt)

        # Accept/reject step and update trust-region radius
        aRed = (val + phi) - (valnew + phinew)
        rho   = aRed / max(1e-16, pRed)
        fracB = snorm / max(1e-16, params['delta_effective'])
        
        
        
        # Optional mild cooldown so you don't grow every single iter
        
        coolk = params.setdefault('grow_cooldown_k', 0)          # counter
        if coolk > 0:
            params['grow_cooldown_k'] = coolk - 1

        

        
        
        
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
            if aRed > params['eta2'] * pRed:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])
            
           
            #if (rho>params['eta2']) and (fracB>=0.9) and (gnorm>10*gtol):
              # params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])
 

        if l != 0:
            #dist = problems[l].pvector.norm(x - x0)
            #residual = Deltai - dist
            #if -1e-12 < residual < 0: residual = 0.0
            #params['delta'] = min(params['delta'], max(0.0, residual))
            params['delta'] = min(params['delta'], Deltai - problems[l].pvector.norm(x - x0))
         
        #params['delta'] = max(0.0,min(params['delta'],params['deltamax']))

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

        # Check stopping criterion
        if gnorm <= gtol or snorm <= stol or i >= params['maxit'] or (problems[l].pvector.norm(x - x0) > (1 - .001)*Deltai and l!=0):
            if i % params['outFreq'] != 0:
                print(f"  {l:4d}   {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")
            if gnorm <= gtol:
                flag = 0
            elif i >= params['maxit']:
                flag = 1
            elif problems[l].pvector.norm(x - x0) > (1 - .001)*Deltai:
                flag = 2
            else:
                flag = 3
            break

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

def make_noise(ud_clean,noise_level = 0.01, mode="relative",seed=1234,mask=None,clip_sigma=None):
    """
    
    Build target with additive Gaussian noise.
    -mode = "relative" : sigma = noise_level*(ud_max-ud_min)
    -mode = "absolute" : sigma = noise_level (in same units as ud)
    -mask : if provided (bool tensor), only add noise where True
    -clip_sigma : clip noisy target to ud_clean +- clip sigma * sigma 

    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    if mode == "relative":
        data_range = float(np.nanmax(ud_clean)-np.nanmin(ud_clean))
        sigma = noise_level*data_range
    elif mode =="absolute":
        sigma =    float(noise_level)
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")
        
    eps = rng.normal(loc=0.0,scale=sigma,size=ud_clean.shape)
    if mask is not None:
        eps = np.where(mask,eps,0.0)
        
    ud_noisy = ud_clean + eps
    if clip_sigma is not None:
        lo = ud_clean - clip_sigma*sigma
        hi = ud_clean + clip_sigma*sigma
        ud_noisy = np.clip(ud_noisy,lo,hi)
    
    return ud_noisy,sigma

def boundary_mask_1d(n, margin=1):
    """True where you allow noise; False near boundaries."""
    m = np.ones(n, dtype=bool)
    m[:margin]  = False
    m[-margin:] = False
    return m

def step_noise_1d(n, n_jumps=6, amp=0.1, seed=0):
    """
    Make a 1D signal that is constant on random segments.
    amp is the max absolute level of each segment.
    """
    rng = np.random.default_rng(seed)
    if n < 3: return np.zeros(n)
    n_jumps = max(1, min(n_jumps, n-2))
    cuts = np.sort(rng.choice(np.arange(1, n-1), size=n_jumps, replace=False))
    cuts = np.concatenate([[0], cuts, [n]])
    levels = rng.uniform(-amp, amp, size=len(cuts)-1)
    out = np.zeros(n, float)
    for i in range(len(cuts)-1):
        out[cuts[i]:cuts[i+1]] = levels[i]
    return out
        
def block_noise_1d(n, K=10, amp=0.1, min_len=5, max_len=None, seed=0):
    """
    Sum of K random constant intervals with random amplitude in [-amp, amp].
    """
    rng = np.random.default_rng(seed)
    if max_len is None: max_len = max(6, n//8)
    out = np.zeros(n, float)
    for _ in range(K):
        L = int(rng.integers(min_len, max_len+1))
        i0 = int(rng.integers(0, max(1, n-L+1)))
        a  = float(rng.uniform(-amp, amp))
        out[i0:i0+L] += a
    return out 

def salt_pepper_1d(n, density=0.01, amp=0.2, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random(n) < density
    spikes = rng.choice([-amp, amp], size=n)
    out = np.zeros(n, float)
    out[mask] = spikes[mask]
    return out

      
        
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
        # o = (self.h / 6 - 1 / self.h) * np.ones(n - 1)
        # d = (2 * self.h / 3 + 2 / self.h) * np.ones(n - 1)
        o = (self.h / 6 ) * np.ones(n - 1)
        d = (2 * self.h / 3 ) * np.ones(n - 1)
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

        # Target state # add some noise
        self.ud_clean = -(self.mesh**2)
        
        #self.ud = self.ud_clean
        #print(self.ud_clean.shape)
        nn = self.mesh.size
        mm = boundary_mask_1d(nn, margin=1) 
        noise = (0.7*step_noise_1d(nn, n_jumps=8, amp=0.05, seed=1) +0.3*block_noise_1d(nn, K=12, amp=0.04, min_len=max(3, nn//40), max_len=max(6, nn//12), seed=2) +0.1*salt_pepper_1d(nn, density=0.005, amp=0.15, seed=3))
        #print(self.ud_clean.shape,noise.shape,(noise*m).shape)
        self.ud  = self.ud_clean+noise*mm
        
        #self.ud, sigma = make_noise(self.ud_clean, noise_level = 0.01, mode="relative", seed=1234, mask=None, clip_sigma=3)
        #print(self.ud.shape)
        #print(self.ud.shape)

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
        matrix_R[i,2*i]       = 1/np.sqrt(2)
    return matrix_R


def driver(savestats, name):
    print("driver started")
    np.random.seed(0)

    # Set up optimization problem
    n          = 8192 # Number of cells
    mu         = 0.08  # Viscosity
    alpha      = 1e-5  # L2 penalty parameter
    beta       = 1e-3 # L1 penalty parameter
    usepc      = True  # Use piecewise constant controls
    useInexact = False
    derivCheck = False
    meshlist = [n,n//2,n//4,n//8]
    #meshlist   = [n, int(n/2), int(n/4)]#, int(n/8)]#, int(n/16)]
    problems   = [] #problem list goes from fine to coarse
    for i in range(0, len(meshlist)):
        B   = BurgersSetup(meshlist[i], mu=mu, alpha=alpha, beta=beta)
        var = B.returnVars(True)
        if i < len(meshlist)-1:
          R = restriction_R(meshlist[i+1], meshlist[i]) #puts R in preceeding problem
        else:
          R = np.eye(meshlist[i])
        p = Problem(var, R)
        p.obj_smooth    = ReducedObjective(Objective(var), ConstraintSolver(var))
        p.obj_nonsmooth = L1Norm(var)
        problems.append(p)


    z = np.ones(n)
    u = np.zeros(n-1)
    x = np.hstack([u, z])

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

    x0  = np.ones(dim)
    cnt = {}

    # Update default parameters
    params            = set_default_parameters("SPG2")
    params["reltol"]  = False
    params["t"]       = 2 / alpha
    params["ocScale"] = 1 / alpha
    params["gtol"]    = 1e-6
    params['RgnormScale']      = 5e-3 # is v in Rgnorm >= v*gnorm -> relative R-step flag
    params['RgnormScaleTol']   = 5  # is v in Rgnorm >= v^i*gtol -> absolute R-step flag
    params['gamma1'] =0.5
    params['gamma2'] = 2.0
    params['eta1'] = 0.05
    params['eta2'] = 0.9


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

    var = problems[0].obj_nonsmooth.var
    mesh = 0.5 * (var['mesh'][:-1] + var['mesh'][1:]) if usepc else var['mesh']

    # Plot results
    #import matplotlib.pyplot as plt
    # Finest level problem
    prob_fine = problems[0]
    var = prob_fine.obj_nonsmooth.var

    # Optimal control from optimizer
    z_opt = x  # trustregion returns z on finest grid

    # Recover the optimal state once with z_opt
    u_opt, _, _ = prob_fine.obj_smooth.con0.solve(z_opt, 1e-12)

    # Meshes
    mesh_nodes = var['mesh']              # size n+1
    x_u  = mesh_nodes[1:-1]               # interior nodes for state u (size n-1)
    ud   = var['ud']                      # target defined at all nodes

    # Control x-grid: handle piecewise-constant vs nodal controls
    if z_opt.shape[0] == mesh_nodes.shape[0]:
        # nodal control
        x_z = mesh_nodes
    else:
        # piecewise-constant control on cell centers
        x_z = 0.5 * (mesh_nodes[:-1] + mesh_nodes[1:])



    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=False)

    # (1) target state
    axes[0].plot(mesh_nodes, ud, linewidth=2)
    axes[0].set_title("Target state $u_d(x)$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("state")
    axes[0].grid(True)

    # (2) optimal state
    axes[1].plot(x_u, u_opt, linewidth=2)
    axes[1].set_title("Optimal state $u^*(x)$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("state")
    axes[1].grid(True)

    # (3) optimal control (step plot if piecewise-constant)
    if z_opt.shape[0] == mesh_nodes.shape[0]:
        axes[2].plot(x_z, z_opt, linewidth=2)
    else:
        axes[2].step(x_z, z_opt, where="mid", linewidth=2)
    axes[2].set_title("Optimal control $z^*(x)$")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("control")
    axes[2].grid(True)

    fig.tight_layout()
    plt.show()


    return cnt


cnt = driver(False, "test_run")
