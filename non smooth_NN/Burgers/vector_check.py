def vector_check(primal, dual, problem):
    print('\n  Vector Check')
    print('  Check Dual Consistency')
    
    xp = problem.dvector.dual(dual)
    xpp = problem.pvector.dual(xp)
    err = problem.dvector.norm(dual - xpp)
    print(f'  norm(x-dual(dual(x))) = {err:.4e}')
    
    xp = problem.pvector.dual(primal)
    xpp = problem.dvector.dual(xp)
    err = problem.pvector.norm(primal - xpp)
    print(f'  norm(y-dual(dual(y))) = {err:.4e}')
    
    print('\n  Check Apply Consistency')
    xp = problem.pvector.dual(primal)
    xydot = problem.dvector.dot(xp, dual)
    xyapply = problem.dvector.apply(dual, primal)
    err = abs(xydot - xyapply)
    print(f' |x.dot(dual(y))-x.apply(y)| = {err:.4e}')
    
    xp = problem.dvector.dual(dual)
    xydot = problem.pvector.dot(xp, primal)
    xyapply = problem.pvector.apply(primal, dual)
    err = abs(xydot - xyapply)
    print(f' |y.dot(dual(x))-y.apply(x)| = {err:.4e}')
