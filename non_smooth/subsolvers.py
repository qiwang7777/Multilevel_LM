import numpy as np
import copy
from .gcp import trustregion_gcp2
#Subsolver for Taylor model
def trustregion_step_SPG2(x, val, dgrad, phi, problem, params, cnt):
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
    g0    = copy.deepcopy(dgrad)
    snorm = 0

    # Evaluate model at GCP
    sHs    = 0
    gs     = 0
    valold = val
    phiold = phi
    hk0    = 0
    valnew = valold
    phinew = phiold

    [sc,snormc,pRed,_,_,cnt,params] = trustregion_gcp2(x,val,dgrad,phi,problem,params,cnt)

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
        g0s    = problem.pvector.dot(g0,s)
        phinew = problem.obj_nonsmooth.value(x1)
        alpha0 = -(g0s + phinew - phiold) / sHs
        if sHs <= params['safeguard']: #*gnorm**2:
          alpha = alphamax
        else:
          alpha = np.minimum(alphamax,alpha0)
        ## Update iterate
        if (alpha == 1):
          x0     = x1
          g0     += problem.dvector.dual(Hs)
          valnew = valold + g0s + 0.5 * sHs
        else:
          x0     += alpha*s
          g0     += alpha*problem.dvector.dual(Hs)
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
          lambdaTmp = params['t']/problem.pvector.norm(g0)
        else:
          lambdaTmp = gnorm**2/sHs

        t0 = np.max([params['lam_min'],np.min([params['lam_max'], lambdaTmp])])
        ## Compute step
        x1    = problem.obj_nonsmooth.prox(x0 - t0 * g0, t0)
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
