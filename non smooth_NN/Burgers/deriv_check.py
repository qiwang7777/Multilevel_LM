def deriv_check(x, d, problem, tol):
    # Update the problem object
    problem.obj_smooth.update(x, 'temp')
    val, _ = problem.obj_smooth.value(x, tol)
    grad, _ = problem.obj_smooth.gradient(x, tol)
    gd = problem.dvector.apply(grad, d)
    t = 1
    
    print('\n  Finite Difference Gradient Check')
    print('           t       DirDeriv        FinDiff         Error')
    
    for i in range(13):
        xnew = x + t * d
        problem.obj_smooth.update(xnew, 'temp')
        valnew, _ = problem.obj_smooth.value(xnew, tol)
        fd = (valnew - val) / t
        
        print(f'  {t:6.4e}    {gd: 6.4e}    {fd: 6.4e}    {abs(fd - gd):6.4e}')
        t *= 0.1

    problem.obj_smooth.update(x, 'temp')
    hv, _ = problem.obj_smooth.hessVec(d, x, tol)
    hvnorm = problem.dvector.norm(hv)
    t = 1
    
    print('\n  Finite Difference Hessian Check')
    print('           t        HessVec        FinDiff         Error')
    
    for i in range(13):
        xnew = x + t * d
        problem.obj_smooth.update(xnew, 'temp')
        gradnew, _ = problem.obj_smooth.gradient(xnew, tol)
        fd = (gradnew - grad) / t
        fdnorm = problem.dvector.norm(fd)
        
        print(f'  {t:6.4e}    {hvnorm: 6.4e}    {fdnorm: 6.4e}    {problem.dvector.norm(fd - hv):6.4e}')
        t *= 0.1

    problem.obj_smooth.update(x, 'temp')
    d2 = np.random.randn(*d.shape)  # Random d2 of the same shape as d
    hd2, _ = problem.obj_smooth.hessVec(d2, x, tol)
    vhd2 = problem.dvector.apply(hd2, d)
    d2hv = problem.dvector.apply(hv, d2)
    
    print('\n  Hessian Symmetry Check')
    print(f'    <x,Hy> = {vhd2: 6.4e}')
    print(f'    <y,Hx> = {d2hv: 6.4e}')
    print(f'    AbsErr = {abs(vhd2 - d2hv): 6.4e}')
