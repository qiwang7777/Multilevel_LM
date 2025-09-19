import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import scipy.sparse as sp
from scipy.sparse import spdiags, diags, lil_matrix,block_diag, kron, eye
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve
import time, torch
from non_smooth.checks import deriv_check, deriv_check_simopt, vector_check
from non_smooth.setDefaultParameters import set_default_parameters
from non_smooth.Problem import Problem
#from non_smooth.L1norm import L1NormEuclid as L1Norm
class L1Norm:
    def __init__(self, var,lo=-25.0,hi=25.0):
        self.var = var
        self.lo = lo
        self.hi = hi

    def value(self, x):
        return self.var.beta * np.dot(self.var.R.T, np.abs(x))

    def prox(self, x, t):
        z_soft = np.maximum(0, np.abs(x) - t * self.var.R * self.var.beta) * np.sign(x)
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
from non_smooth.subsolvers import trustregion_step_SPG2
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
    #print("Rnorm:", Rgnorm, "gnorm:", gnorm,
    #  "abs_gate:", min(1e-2, params['gtol']*np.sqrt(x.shape[0]/2.0)))
    #if params.get('debug_drop_gate',False):
    #    print("Rnorm:",Rgnorm, "gnorm:",gnorm,"abs_gate:",abs_gate,"cap:",cap,"prox_ok:",prox_ok,"do_recurse:",do_recurse)
    # -- DEBUG: drop-gate details --
    #print(f"[GATE l={l} i={i}] prox_ok={prox_ok} cap={cap:.3e} cap_ok={cap_ok} "
    #      f"Rgnorm={Rgnorm:.3e} gnorm={gnorm:.3e} "
    #      f"abs_gate={min(1e-2,params['gtol']*np.sqrt(x.shape[0]/2.0)):.3e} "
    #      f"drop_rel={Rgnorm >= params['RgnormScale']*gnorm} "
    #      f"drop_abs={Rgnorm >= min(1e-2,params['gtol']*np.sqrt(x.shape[0]/2.0))} "
    #      f"→ do_recurse={do_recurse}")



    if do_recurse:
        #print(f"[REC l={l}] A: entering recursive branch")
        #print(f"[REC l={l}] B: deepcopy start")
        # -------- RECURSIVE BRANCH (to level l+1) --------
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
        #params['last_step_from_child'] = True
        #params['last_child_iflag'] = cnt_coarse.get('iflag', 3)
        
        #print(f"[RECURSE← l={l+1}] iflag_child={cnt_coarse.get('iflag','?')} "
        #      f"iter_child={cnt_coarse.get('iter','?')}")
        #params['nb_hit_valid_on_this_level'] = False  # step came from child



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
        #params['nb_hit_valid_on_this_level'] = True   # step computed on this level

        #params['last_step_from_child'] = False
        #params['last_child_iflag'] = None


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

    # if params['useSecantPrecond']:
        # problems[l].prec.apply         = lambda x: problems[l].secant.apply(x, problems[l].pvector, problems[l].dvector)
        # problems[l].prec.apply_inverse = lambda x: problems[l].secant.apply_inverse(x, problems[l].pvector, problems[l].dvector)

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
        #is_nb = _near_boundary(snorm, params['delta_effective'])
        is_nb = (fracB>=0.8)
        is_gq = (aRed>params['eta2']*pRed)
        is_ok = (gnorm>10.0*gtol)
        cg = 0.6
        max_growth = 1.6
        frac_thresh = 0.8
        gpress       = max(0.0, fracB - frac_thresh)             # boundary pressure ∈ [0, 0.2..]
        qpress       = max(0.0, rho   - params['eta2'])          # quality pressure
        okpress      = max(0.0, gnorm/(10.0*gtol) - 1.0)         # how far from stationarity
        # Optional mild cooldown so you don't grow every single iter
        coolN = params.setdefault('grow_cooldown_N', 2)          # allow growth at most every N iters
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
from non_smooth_Semilinear.quadpts import quadpts, sparse
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
        #self.c             = self.udesired(self.mesh.p[self.FreeNodes,:])
        

# exact solution u
#    def exactu(self, p):
#       return np.sin(2*np.pi*p[:,0]) * np.sin(2*np.pi*p[:,1])
# right hand side
#    def f(self, p,lam):
#        t = 8*np.pi**2*(np.sin(2*np.pi*p[:,0]) * np.sin(2*np.pi*p[:,1])) + self.exactu(p)**3
#        if self.ctrl == 1:
#            t -= self.exactz(p,lam)
#        elif self.ctrl == 2:
#            t -= self.exactz_constrained(p, lam, self.za, self.zb)
#        return t
# # ud
#    def udesired(self, p):
#        return self.exactu(p) - (8*np.pi**2+3*self.exactu(p)**2)*self.exactp(p)
# # p
#    def exactp(self, p):
#        return self.exactu(p)
# # z (unconstrained case)
#    def exactz(self, p,lam):
#        t = self.exactp(p)
#        return -t/lam
# # z (constrained case)
#    def exactz_constrained(self, p,lamb,a,b):
#        t = self.exactp(p)
#        return np.minimum(b,np.maximum(-t/lamb,a))
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


import time
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

def driver(savestats=True, name="semilinear_control_2d"):
    print("2D Driver started")
    np.random.seed(0)

    # Problem parameters
    n = 128 # 32x32 grid
    alpha = 1e-4
    beta = 1e-2
    #meshlist = [n]
    
    meshlist = [n,n//2,n//4,n//8]
    problems = []



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


    dim = 2*n*n
    x0 = np.zeros(dim)


    params = set_default_parameters("SPG2")
    params.update({
        "reltol": False,
        "t": 2/alpha,
        "ocScale":1/alpha,
        "maxiter":200,
        "verbose":True,
        "useReduced":False,
        "gtol":1e-7,
        "RgnormScale":0.85, # is v in Rgnorm >= v^i*gtol -> absolute R-step flag
        "RgnormScaleTol": 500,
        "debug_drop_gate":True,
        "debug_h_equiv":True,
        "debug_h_equiv_freq": 1,
        "prox_equiv_abs_tol": 1e-10,
        "min_drop_cap":1e-8,
        'deltamax': 1e4,
        'gamma2': 2.0,
        'eta2':0.95
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



x_opt,cnt_tr=driver()
