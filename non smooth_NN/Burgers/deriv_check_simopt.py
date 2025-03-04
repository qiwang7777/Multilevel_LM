import numpy as np
def deriv_check_simopt(u0, z0, obj, con, tol):
    """
    Perform derivative checks using finite differences.

    Parameters:
    u0 (np.array): Initial state vector.
    z0 (np.array): Initial control vector.
    obj (Objective): Objective function class.
    con (Constraint): Constraint class.
    tol (float): Tolerance for function evaluations.
    """
    # Random directions for finite differences
    u = np.random.randn(*u0.shape)
    udir = np.random.randn(*u0.shape)
    z = np.random.randn(*z0.shape)
    zdir = np.random.randn(*z0.shape)
    lambda_ = np.random.randn(*con.value(np.hstack([u0, z0])).shape)

    # Evaluate objective and constraint functions
    f = obj.value(np.hstack([u, z]), tol)
    df1 = obj.gradient_1(np.hstack([u, z]), tol)
    df2 = obj.gradient_2(np.hstack([u, z]), tol)
    c = con.value(np.hstack([u, z]), tol)
    J1d = con.apply_jacobian_1(udir, np.hstack([u, z]), tol)
    J2d = con.apply_jacobian_2(zdir, np.hstack([u, z]), tol)
    J1 = con.apply_adjoint_jacobian_1(lambda_, np.hstack([u, z]), tol)
    J2 = con.apply_adjoint_jacobian_2(lambda_, np.hstack([u, z]), tol)

    # Check objective gradient_1 using finite differences
    print("\n Objective gradient_1 check using finite differences (FDs)")
    print(" FD step size      grad'*v      FD approx.  absolute error")
    delta = 1
    for d in range(13):
        f1 = obj.value(np.hstack([u + delta * udir, z]), tol)
        error = np.abs(np.dot(df1, udir) - (f1 - f) / delta)
        print(f" {delta:12.6e}  {np.dot(df1, udir):12.6e}  {(f1 - f) / delta:12.6e}  {error:12.6e}")
        delta /= 10

    # Check objective gradient_2 using finite differences
    print("\n Objective gradient_2 check using finite differences (FDs)")
    print(" FD step size      grad'*v      FD approx.  absolute error")
    delta = 1
    for d in range(13):
        f1 = obj.value(np.hstack([u, z + delta * zdir]), tol)
        error = np.abs(np.dot(df2, zdir) - (f1 - f) / delta)
        print(f" {delta:12.6e}  {np.dot(df2, zdir):12.6e}  {(f1 - f) / delta:12.6e}  {error:12.6e}")
        delta /= 10

    # Check objective Hessian_11 using finite differences
    print("\n Objective Hessian_11 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    hv = obj.hessVec_11(udir, np.hstack([u, z]), tol)
    delta = 1
    for d in range(13):
        g1 = obj.gradient_1(np.hstack([u + delta * udir, z]), tol)
        fd_approx = (g1 - df1) / delta
        error = np.linalg.norm(hv - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check objective Hessian_12 using finite differences
    print("\n Objective Hessian_12 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    hv = obj.hessVec_12(zdir, np.hstack([u, z]), tol)
    delta = 1
    for d in range(13):
        g1 = obj.gradient_1(np.hstack([u, z + delta * zdir]), tol)
        fd_approx = (g1 - df1) / delta
        error = np.linalg.norm(hv - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check objective Hessian_21 using finite differences
    print("\n Objective Hessian_21 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    hv = obj.hessVec_21(udir, np.hstack([u, z]), tol)
    delta = 1
    for d in range(13):
        g1 = obj.gradient_2(np.hstack([u + delta * udir, z]), tol)
        fd_approx = (g1 - df2) / delta
        error = np.linalg.norm(hv - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check objective Hessian_22 using finite differences
    print("\n Objective Hessian_22 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    hv = obj.hessVec_22(zdir, np.hstack([u, z]), tol)
    delta = 1
    for d in range(13):
        g1 = obj.gradient_2(np.hstack([u, z + delta * zdir]), tol)
        fd_approx = (g1 - df2) / delta
        error = np.linalg.norm(hv - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Jacobian_1 using finite differences
    print("\n Constraint Jacobian_1 check using finite differences (FDs)")
    print(" FD step size      norm(Jac*v)     norm(FD approx.)   absolute error")
    delta = 1
    for d in range(13):
        c1 = con.value(np.hstack([u + delta * udir, z]), tol)
        fd_approx = (c1 - c) / delta
        error = np.linalg.norm(J1d - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(J1d):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Jacobian_2 using finite differences
    print("\n Constraint Jacobian_2 check using finite differences (FDs)")
    print(" FD step size      norm(Jac*v)     norm(FD approx.)   absolute error")
    delta = 1
    for d in range(13):
        c1 = con.value(np.hstack([u, z + delta * zdir]), tol)
        fd_approx = (c1 - c) / delta
        error = np.linalg.norm(J2d - fd_approx)
        print(f" {delta:12.6e}     {np.linalg.norm(J2d):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Hessian_11 using finite differences
    print("\n Constraint Hessian_11 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    Hv = con.apply_adjoint_hessian_11(lambda_, udir, np.hstack([u, z]), tol)
    delta = 1
    for d in range(13):
        Jn = con.apply_adjoint_jacobian_1(lambda_, np.hstack([u + delta * udir, z]), tol)
        fd_approx = (Jn - J1) / delta
        error = np.linalg.norm(Hv - fd_approx) / (1 + np.linalg.norm(Hv))
        print(f" {delta:12.6e}     {np.linalg.norm(Hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Hessian_12 using finite differences
    print("\n Constraint Hessian_12 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    Hv = con.apply_adjoint_hessian_12(lambda_, udir, np.hstack([u, z]), tol)
    delta = 1
    for d in range(13):
        Jn = con.apply_adjoint_jacobian_2(lambda_, np.hstack([u + delta * udir, z]), tol)
        fd_approx = (Jn - J2) / delta
        error = np.linalg.norm(Hv - fd_approx) / (1 + np.linalg.norm(Hv))
        print(f" {delta:12.6e}     {np.linalg.norm(Hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Hessian_21 using finite differences
    print("\n Constraint Hessian_21 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    Hv = con.apply_adjoint_hessian_21(lambda_, zdir, np.hstack([u, z]), tol)
    delta = 1
    for d in range(13):
        Jn = con.apply_adjoint_jacobian_1(lambda_, np.hstack([u, z + delta * zdir]), tol)
        fd_approx = (Jn - J1) / delta
        error = np.linalg.norm(Hv - fd_approx) / (1 + np.linalg.norm(Hv))
        print(f" {delta:12.6e}     {np.linalg.norm(Hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check constraint Hessian_22 using finite differences
    print("\n Constraint Hessian_22 check using finite differences (FDs)")
    print(" FD step size      norm(H*v)      norm(FD approx.)    absolute error")
    Hv = con.apply_adjoint_hessian_22(lambda_, zdir, np.hstack([u, z]), tol)
    delta = 1
    for d in range(13):
        Jn = con.apply_adjoint_jacobian_2(lambda_, np.hstack([u, z + delta * zdir]), tol)
        fd_approx = (Jn - J2) / delta
        error = np.linalg.norm(Hv - fd_approx) / (1 + np.linalg.norm(Hv))
        print(f" {delta:12.6e}     {np.linalg.norm(Hv):12.6e}      {np.linalg.norm(fd_approx):12.6e}      {error:12.6e}")
        delta /= 10

    # Check solve
    print("\n Solve check")
    uz = con.solve(z, tol)
    _, res = con.value(np.hstack([uz, z]), tol)
    print(f"  Absolute Residual = {res:12.6e}")
    print(f"  Relative Residual = {res / np.linalg.norm(uz):12.6e}")

    # Check applyInverseJacobian_1
    print("\n Check applyInverseJacobian_1")
    uz = con.apply_inverse_jacobian_1(udir, np.hstack([u, z]), tol)
    Juz = con.apply_jacobian_1(uz, np.hstack([u, z]), tol)
    res = np.linalg.norm(Juz - udir)
    print(f"  Absolute Error = {res:12.6e}")
    print(f"  Relative Error = {res / np.linalg.norm(udir):12.6e}")

    # Check applyInverseAdjointJacobian_1
    print("\n Check applyInverseAdjointJacobian_1")
    uz = con.apply_inverse_adjoint_jacobian_1(udir, np.hstack([u, z]), tol)
    Juz = con.apply_adjoint_jacobian_1(uz, np.hstack([u, z]), tol)
    res = np.linalg.norm(Juz - udir)
    print(f"  Absolute Error = {res:12.6e}")
    print(f"  Relative Error = {res / np.linalg.norm(udir):12.6e}")

    # Check applyAdjointJacobian_1
    print("\n Check applyAdjointJacobian_1")
    vdir = np.random.randn(*udir.shape)
    aju = con.apply_adjoint_jacobian_1(udir, np.hstack([u, z]), tol)
    ju = con.apply_jacobian_1(vdir, np.hstack([u, z]), tol)
    res = np.abs(np.dot(aju, vdir) - np.dot(ju, udir))
    print(f"  Absolute Error = {res:12.6e}")

    # Check applyAdjointJacobian_2
    print("\n Check applyAdjointJacobian_2")
    aju = con.apply_adjoint_jacobian_2(udir, np.hstack([u, z]), tol)
    ju = con.apply_jacobian_2(zdir, np.hstack([u, z]), tol)
    res = np.abs(np.dot(aju, zdir) - np.dot(ju, udir))
    print(f"  Absolute Error = {res:12.6e}")
