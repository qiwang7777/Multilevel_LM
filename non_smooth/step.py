import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import copy, torch
from .modelTR import modelTR
from .phiPrec import phiPrec
from . import trustregion
from .subsolvers import trustregion_step_SPG2
from .Problem import Problem
from .checks import deriv_check
import pdb

def trustregion_step(l,x,val,grad,phi,problems,params,cnt, i=0):
    #fine level l comes in
    dgrad                   = problems[l].dvector.dual(grad) #dual(grad) puts in primal space
    L                       = len(problems)
    pgrad                   = problems[l].obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
    cnt['nprox']           += 1
    debugflag               = 'R-step'
    ##Note: change Rdgnorm and gnorm to h_[i-1, 0] and h_[i,k]
    R0   = problems[0].R
    for i in range(1, l+1): #and adjust here for multilevel
      R0 = problems[i].R @ R0

    R     = problems[l].R
    xt    = R @ x
    gnorm = problems[l].pvector.norm(pgrad - x) / params['ocScale']
    if l < L-1:
      Rgnorm = problems[l+1].pvector.norm(R @ pgrad - xt) / params['ocScale']
    else:
      Rgnorm = 0.

    #adjust here for multilevel
    if (i > 0 and l < L-1) and (Rgnorm >= params['RgnormScale']*gnorm and Rgnorm >= (params['RgnormScaleTol']**(l+1))*params['gtol']): #note: counters are off
      problemsL = copy.deepcopy(problems) #problem list goes from fine to coarse
      # constructs L_{i-1} = f_{i-1} + (Rg_{i,k} - g_{i-1,0})'*(x - x_{i-1,0})

      p               = Problem(problems[l+1].obj_nonsmooth.var, problems[l+1].R) #make next level problem
      p.obj_smooth    = modelTR(problems, params["useSecant"], 'recursive', l = l+1, R = problems[l].R, dgrad = R @ dgrad, x = xt)
      p.obj_nonsmooth = phiPrec(problems[0], R = R0, l = l+1)
      p.pvector       = problems[l+1].pvector
      p.dvector       = problems[l+1].dvector
      problemsL[l+1]  = p
      # import pdb
      # pdb.set_trace()
      # d = np.random.randn((problems[i].R.shape[1]))
      # deriv_check(xt, d, p, 1e-4 * np.sqrt(np.finfo(float).eps))
      # print(l+1, problemsL[l+1].obj_smooth.value(R @ x, 0.), np.linalg.norm(grad))
      val, _           = p.obj_smooth.value(xt, 0.0)
      cnt['nobj1']    += 1
      phi              = p.obj_nonsmooth.value(xt)
      Deltai           = copy.copy(params['delta'])
      gtol_store       = params['gtol']
      params['gtol']   = (params['RgnormScaleTol']**(l+1))*params['gtol']
      xnew, cnt_coarse = trustregion.trustregion(l+1, xt, Deltai, problemsL, params)
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
      valnew, _     = problemsL[l+1].obj_smooth.value(xnew, 0.0)
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
      debugflag = 'SPG step'
      if x.shape[0] != R0.shape[0]: #check the dimension if only going up one level
        R0 = problems[0].R
        for i in range(1, l): #and adjust here for multilevel
          R0 = problems[i].R @ R0

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
      if np.abs(pRed) > 1e-5:
        print(debugflag, pRed)
        stp
    return s, snorm, pRed, phinew, iflag, iter, cnt, params

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