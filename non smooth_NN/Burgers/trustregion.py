import numpy as np
import time

def trustregion(x0, problem, params):
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
        'timestor': [],
        'valerr': [],
        'valtol': [],
        'graderr': [],
        'gradtol': []
    }

    # Compute initial function information
    if hasattr(problem.obj_smooth, 'begin_counter'):
        cnt = problem.obj_smooth.begin_counter(0, cnt)

    if params['initProx']:
        x = problem.obj_nonsmooth.prox(x0, 1)
        cnt['nprox'] += 1
    else:
        x = x0

    problem.obj_smooth.update(x, 'init')
    ftol = 1e-12
    if params['useInexactObj']:
        ftol = params['maxValTol']

    val, _ = problem.obj_smooth.value(x, ftol)
    cnt['nobj1'] += 1

    grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
    phi = problem.obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    if hasattr(problem.obj_smooth, 'end_counter'):
        cnt = problem.obj_smooth.end_counter(0, cnt)

    # Initialize secant if needed
    #if params['useSecant'] or params['useSecantPrecond']:
    #    problem.secant = SR1(params['secantSize'], params['useDefault'], params['initScale'])
        

    if params['useSecantPrecond']:
        problem.prec.apply = lambda x: problem.secant.apply(x, problem.pvector, problem.dvector)
        problem.prec.apply_inverse = lambda x: problem.secant.apply_inverse(x, problem.pvector, problem.dvector)

    # Output header
    print(f"\nNonsmooth Trust-Region Method using {params.get('spsolver', 'SPG2')} Subproblem Solver")
    print("  iter            value            gnorm              del            snorm       nobjs      ngrad      nhess      nobjn      nprox    iterSP    flagSP")
    print(f"  {0:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}              ---      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}       ---       ---")

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
    cnt['timestor'].append(np.nan)

    # Set optimality tolerance
    gtol = params['gtol']
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    # Check stopping criterion
    if gnorm <= gtol:
        return x, cnt

    # Iterate
    for i in range(1, params['maxit'] + 1):
        if hasattr(problem.obj_smooth, 'begin_counter'):
            cnt = problem.obj_smooth.begin_counter(i, cnt)

        # Solve trust-region subproblem
        params['tolsp'] = min(params['atol'], params['rtol'] * gnorm ** params['spexp'])
        dgrad = problem.dvector.dual(grad)
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val, dgrad, phi, problem, params, cnt
        )

        # Update function information
        xnew = x + s
        problem.obj_smooth.update(xnew, 'trial')
        valnew, val, cnt = compute_value(xnew, x, val, problem.obj_smooth, pRed, params, cnt)

        # Accept/reject step and update trust-region radius
        aRed = (val + phi) - (valnew + phinew)
        if aRed < params['eta1'] * pRed:
            params['delta'] = params['gamma1'] * min(snorm, params['delta'])
            problem.obj_smooth.update(x, 'reject')
            if params['useInexactGrad']:
                grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
        else:
            x = xnew
            val = valnew
            phi = phinew
            problem.obj_smooth.update(x, 'accept')
            grad0 = grad
            grad, dgrad, gnorm, cnt = compute_gradient(x, problem, params, cnt)
            if aRed > params['eta2'] * pRed:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])

            # Update secant
            if params['useSecant'] or params['useSecantPrecond']:
                y = grad - grad0
                problem.secant.update(s, y, problem.pvector, problem.dvector)
                if params['useSecantPrecond']:
                    problem.prec.apply = lambda x: problem.secant.apply(x, problem.pvector, problem.dvector)
                    problem.prec.apply_inverse = lambda x: problem.secant.apply_inverse(x, problem.pvector, problem.dvector)

        # Output iteration history
        if i % params['outFreq'] == 0:
            print(f"  {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")

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
        cnt['timestor'].append(time.time() - start_time)

        if hasattr(problem.obj_smooth, 'end_counter'):
            cnt = problem.obj_smooth.end_counter(i, cnt)

        # Check stopping criterion
        if gnorm <= gtol or snorm <= stol or i >= params['maxit']:
            if i % params['outFreq'] != 0:
                print(f"  {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")
            if gnorm <= gtol:
                flag = 0
            elif i >= params['maxit']:
                flag = 1
            else:
                flag = 2
            break

    cnt['iter'] = i
    cnt['timetotal'] = time.time() - start_time

    print("Optimization terminated because ", end="")
    if flag == 0:
        print("optimality tolerance was met")
    elif flag == 1:
        print("maximum number of iterations was met")
    else:
        print("step tolerance was met")
    print(f"Total time: {cnt['timetotal']:8.6e} seconds")

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
