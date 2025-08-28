import numpy as np
import torch
import copy
import numbers
import os
import sys
sys.path.append(os.path.abspath('..'))
from .modelTR import modelTR
from .phiPrec import phiPrec
from . import trustregion
from .subsolvers import trustregion_step_SPG2
from .Problem import Problem
import pdb
from .RWrap import RWrap


def as_float(x):
    """Return a Python float from a 0-dim tensor or numeric."""
    if torch.is_tensor(x):
        x = x.detach()
        if x.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(x.shape)}")
        return float(x.cpu().item())
    if isinstance(x,np.ndarray):
        if x.size != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(x.shape)}")
        return float(x.reshape(()))
    if isinstance(x, (np.generic, numbers.Number)):
        # numpy.float64, numpy.int64, Python floats/ints, etc.
        return float(x)
    if isinstance(x,numbers.Number) or hasattr(x,"item"):
        try:
            return float(x.item())
        except Exception:
            return float(x)
        
    return float(np.asarray(x).reshape(()))

def as_numpy(x):
    """Detach + cpu + numpy for arrays."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def trustregion_step(l,x,val,grad,phi,problems,params,cnt, i=0):
    #fine level l comes in
    dgrad                   = problems[l].dvector.dual(grad) #dual(grad) puts in primal space
    norm_gl                 = problems[l].pvector.norm(dgrad)
    
    L                       = len(problems)
    pgrad                   = problems[l].obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
    cnt['nprox']           += 1
    debugflag               = 'R-step'
    ##Note: change Rdgnorm and gnorm to h_[i-1, 0] and h_[i,k]
    R0   = problems[0].R
    for j in range(1, l+1): #and adjust here for multilevel
      R0 = problems[j].R @ R0

    R     = problems[l].R
    xt    = R @ x
    gnorm = problems[l].pvector.norm(pgrad - x) / params['ocScale']
    
    
    
    
    
    if l < L-1:
      
      Rgnorm = problems[l+1].pvector.norm(R @ pgrad - xt) / params['ocScale']
      g_l_pr = problems[l].dvector.dual(grad)

      # coarse smooth grad at base point xt
      g_c_dual, _ = problems[l+1].obj_smooth.gradient(xt)
      g_c_pr      = problems[l+1].dvector.dual(g_c_dual)

      # mismatch Δg in coarse (primal) space
      Delta_g_pr  = (R @ g_l_pr) - g_c_pr

      # coarse smooth value at base (always >=0 for squared-residuals)
      Fc0, _ = problems[l+1].obj_smooth.value(xt)
      Fc0_f  = as_float(Fc0)

      # norms (same norm you already use elsewhere)
      nD_f     = as_float(problems[l+1].pvector.norm(Delta_g_pr))
      Rnorm = problems[l+1].pvector.norm(R @ dgrad) / params['ocScale'] 
      n_gc_u  = as_float(problems[l+1].pvector.norm(g_c_pr))       # ||∇F_coarse|| (unscaled)

      # --- positivity safeguard via SAFE COARSE RADIUS ---
      # Dynamic floor: Fc0_eff = max(Fc0, ρ * ||∇F_c|| * Δ)
      rho       = params.get('coarse_pos_floor_rel', 0.5)
      Delta_cur = params['delta']
      Fc0_eff   = max(Fc0_f, rho * n_gc_u * Delta_cur)


      # ===== (A) SAFE COARSE RADIUS: keep worst-case |<Δg, s>| ≤ θ * Fc0 =====
      theta = params.get('lin_corr_theta', 0.9)   # e.g. 0.25–0.75
      eps = 1e-16
      Delta_safe = theta * Fc0_eff / max(nD_f, eps)
      
      try:
          Deltai = min(Deltai,Delta_safe)
      except NameError:
        Deltai = min(Delta_cur, Delta_safe)
          
      Delta_min = params.get('coarse_delta_min_frac', 0.02) * Delta_cur
      ok_pos    = (Deltai >= Delta_min)
      if params.get('verbose_smooth_norms', False):
        print(f"[lvl {l}] Fc0={Fc0_f:.3e} Fc0_eff={Fc0_eff:.3e} ||Δg||={nD_f:.3e} "
              f"Δsafe={Delta_safe:.3e} Δpass={Deltai:.3e} (min={Delta_min:.3e}) ok_pos={ok_pos}")

     
    else:
      Rnorm = 0.0
      Rgnorm = 0.0
      ok_pos = False
    drop_ok = (i > 0 and l < L-1) \
          and ok_pos \
          and (Rgnorm >= params['RgnormScale'] * gnorm) \
          and (Rgnorm >= params['RgnormScaleTol'] * params['gtol'])
    if params.get('verbose_gate', False):
      print(f"[gate l={l}] Rgnorm-τ*gnorm={as_float(Rgnorm) - params['RgnormScale']*as_float(gnorm):.3e} "
          f"ok_pos={ok_pos} drop_ok={drop_ok}")

    #adjust here for multilevel
    #print("debugging",Rgnorm-params['RgnormScale']*gnorm,"ok_pos:",ok_pos)# "RgnormScaleTolgnorm:",params['RgnormScale']*gnorm, "gtol:",params['gtol'])
    #if (i > 0 and l < L-1) and ok_pos and (Rgnorm >= params['RgnormScale']*gnorm and Rgnorm >= params['RgnormScaleTol']*params['gtol']): #note: counters are off
    if drop_ok:
          
      problemsL = copy.deepcopy(problems) #problem list goes from fine to coarse
      # constructs L_{i-1} = f_{i-1} + (Rg_{i,k} - g_{i-1,0})'*(x - x_{i-1,0})

      p               = Problem(problems[l+1].obj_nonsmooth.var, problems[l+1].R) #make next level problem
      p.obj_smooth    = modelTR(problems, params["useSecant"], 'recursive', l = l+1, R = problems[l].R, dgrad = R @ dgrad, x = copy.deepcopy(xt))
      p.obj_nonsmooth = phiPrec(problems[0], R = R0, l = l+1)
      p.pvector       = problems[l+1].pvector
      p.dvector       = problems[l+1].dvector
      problemsL[l+1]  = p
      # d = np.random.randn((problems[i].R.shape[1]))
      # deriv_check(xt, d, p, 1e-4 * np.sqrt(np.finfo(float).eps))
      # print(l+1, problemsL[l+1].obj_smooth.value(R @ x, 0.), np.linalg.norm(grad))
      val, _           = p.obj_smooth.value(xt, 0.0)
      cnt['nobj1']    += 1
      # import pdb
      # pdb.set_trace()
      phi              = p.obj_nonsmooth.value(xt)
      Deltai           = copy.copy(params['delta'])
      gtol_store       = params['gtol']
      params['gtol']   = params['RgnormScaleTol']*params['gtol']
      xnew, cnt_coarse = trustregion(l+1, xt, Deltai, problemsL, params)
      #recompute snorm, pRed, phinew
      params['delta']  = Deltai
      params['gtol']   = gtol_store
      s                = R.T @ (xnew - xt)
      snorm            = problems[l].pvector.norm(s)
      #check L = f_{i-1} + <R g, s> + phi_{i-1}(x0 + s)
      #m = L + phi_i - phi_{i-1} so M = f_{i-1} + <Rg, s> + \phi_i
      ## compute old
      phinew        = problems[0].obj_nonsmooth.value(R0.T @ xnew)
      cnt['nobj2'] += 1
      valnew, _     = p.obj_smooth.value(xnew, 0.0)
      cnt['nobj1'] += 1
      # print('after', val, valnew, phi, phinew, np.linalg.norm(grad))
      # pdb.set_trace()
      pRed   = val + phi - valnew - phinew
      iflag  = cnt_coarse['iflag']
      iter   = cnt_coarse['iter']
      cnt    = problemsL[l+1].obj_smooth.addCounter(cnt)
      cnt    = problemsL[l+1].obj_nonsmooth.addCounter(cnt)
    else:
      R         = Reye(x)
      #print("Reye_shape",R.shape)
      debugflag = 'SPG step'
      #print("debugging:",x.shape,R0.shape)
      if x.shape[0] != R0.shape[0]: #check the dimension if only going up one level
        R0 = problems[0].R
        for j  in range(1, l): #and adjust here for multilevel
          R0 = problems[j].R @ R0

      problemTR               = Problem(problems[l].var, R)
      problemTR.obj_smooth    = modelTR(problems, params["useSecant"], 'spg', l = l, R = R, dgrad = dgrad, x = x)
      problemTR.obj_nonsmooth = phiPrec(problems[0], R = R0, l = l)
      problemTR.pvector       = problems[l].pvector
      problemTR.dvector       = problems[l].dvector
      # d = np.random.randn((x.shape[0]))
      # deriv_check(x, d, problemTR, 1e-4 * np.sqrt(np.finfo(float).eps))
      s, snorm, pRed, phinew, iflag, iter, cnt, params = trustregion_step_SPG2(x, val, dgrad, phi, problemTR, params, cnt)
      cnt = problemTR.obj_smooth.addCounter(cnt)
      cnt = problemTR.obj_nonsmooth.addCounter(cnt)

    if pRed < 0:
      if abs(as_float(pRed)) > 1e-5:
        import pdb
        pdb.set_trace()
        print(debugflag, pRed)
        stp
    return s, snorm, pRed, phinew, iflag, iter, cnt, params
from collections import OrderedDict

def Reye(x):
    """Identity restriction with stable .shape (used by trustregion_step)."""
    ops = OrderedDict()
    out_shapes_map = {}
    in_shapes_map  = {}
    
    for name, t in x.td.items():
        n = int(t.numel())
        I = torch.eye(n, n, dtype=t.dtype, device=t.device)
        ops[name] = I
        shape_t = tuple(t.shape)
        out_shapes_map[name] = shape_t   # output shape (coarse) == input shape (fine) for identity
        in_shapes_map[name]  = shape_t
    return RWrap(ops, out_shapes_map=out_shapes_map, in_shapes_map=in_shapes_map)
