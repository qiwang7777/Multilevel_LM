import numpy as np
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
