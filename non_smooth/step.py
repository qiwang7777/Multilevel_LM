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
from non_smooth.checks import deriv_check

def trustregion_step(l,x,val,grad,phi,problems,params,cnt, i=0):
    #fine level l comes in
    dgrad                   = problems[l].dvector.dual(grad) #dual(grad) puts in primal space
    L                       = len(problems)
    pgrad                   = problems[l].obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
    ##Note: change Rdgnorm and gnorm to h_[i-1, 0] and h_[i,k]
    # gnorm                   = problems[l].pvector.norm(dgrad)
    R0 = problems[0].R
    for i in range(1, l): #and adjust here for multilevel
      R0 = problems[i].R @ R0
    cnt['nprox'] += 1
    R = problems[l].R
    gnorm = problems[l].pvector.norm(pgrad - x) / params['ocScale']
    if l < L-2:
      Rgnorm = problems[l+1].pvector.norm(R @ (pgrad - x))
    else:
      Rgnorm = .1*gnorm

    #adjust here for multilevel
    if (i > 0 and l < 1) and (Rgnorm > 0.01*gnorm and Rgnorm >= 1e-3): #note: counters are off
      problemsL = [] #problem list goes from fine to coarse
      for i in range(0, L):
        if i == l+1:
          # constructs L_{i-1} = f_{i-1} + (Rg_{i,k} - g_{i-1,0})'*(x - x_{i-1,0})
          p               = Problem(problems[i].obj_nonsmooth.var, problems[i].R) #make next level problem
          p.obj_smooth    = modelTR(problems, params["useSecant"], 'recursive', l = i, R = problems[i-1].R, grad = grad, x = x)
          p.obj_nonsmooth = phiPrec(problems[0], R = R0, l = i)
          problemsL.append(p)
        else:
          problemsL.append(problems[i])
      Deltai           = copy.copy(params['delta'])
      gtol_store       = params['gtol']
      params['gtol']   = 1e2*params['gtol']
      xnew, cnt_coarse = trustregion.trustregion(l+1, R @ x, Deltai, problemsL, params)
      #recompute snorm, pRed, phinew
      params['delta'] = Deltai
      params['gtol']  = gtol_store
      s               = R.T @ xnew - x
      snorm           = problems[l].pvector.norm(s)
      #check L = f_{i-1} + <R g, s> + phi_{i-1}(x0 + s)
      #m = L + phi_i - phi_{i-1} so M = f_{i-1} + <Rg, s> + \phi_i
      ## compute old
      val, _ = problems[l].obj_smooth.value(x, 0.0)
      cnt['nobj1'] += 1
      phi    = problems[l].obj_nonsmooth.value(x)
      cnt['nobj2'] += 1
      phinew = problems[0].obj_nonsmooth.value(R0.T @ xnew)
      cnt['nobj2'] += 1
      valnew, _ = problemsL[l+1].obj_smooth.value(xnew, 0.0)
      cnt['nobj1'] += 1
      pRed   = val + phi - valnew - phinew
      iflag  = cnt_coarse['iflag']
      iter   = cnt_coarse['iter']
      cnt    = problemsL[l+1].obj_smooth.addCounter(cnt)
      cnt    = problemsL[l+1].obj_nonsmooth.addCounter(cnt)
    else:
      R                       = Reye(x)
      problemTR               = Problem(problems[l].var, R)
      problemTR.obj_smooth    = modelTR(problems, params["useSecant"], 'spg', l = l, R = R, grad = grad, x = x)
      problemTR.obj_nonsmooth = phiPrec(problems[0], R = R0, l = l)
      # d = np.random.randn((x.shape[0]))
      # deriv_check(x, d, problemTR, 1e-4 * np.sqrt(np.finfo(float).eps))
      s, snorm, pRed, phinew, iflag, iter, cnt, params = trustregion_step_SPG2(x, val, dgrad, phi, problemTR, params, cnt)
      cnt = problemTR.obj_smooth.addCounter(cnt)
      cnt = problemTR.obj_nonsmooth.addCounter(cnt)
    return s, snorm, pRed, phinew, iflag, iter, cnt, params

#Recursive step
def Reye(x):
    if isinstance(x, np.ndarray):
      matrix_R = np.eye(x.shape[0])
    else:
      matrix_R = x.clone()
      for k, v in matrix_R.td.items():
        matrix_R.td[k] = torch.eye(x.td[k].size()[0], dtype=torch.float64)

    return matrix_R