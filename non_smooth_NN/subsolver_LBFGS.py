def trustregion_step_LBFGS2(x, val, dgrad, phi, problem, params, cnt):
    """
    Proximal L-BFGS subsolver for the TR subproblem:
        min_s  m(x+s) + phi(x+s)  s.t. ||s|| <= delta_effective

    Matches trustregion_step_SPG2 signature.
    Returns: s, snorm, pRed, phinew, iflag, iter_count, cnt, params
    """
    import numpy as np
    import copy

    # --- params / defaults ---
    Delta      = params.get('delta_effective', params['delta'])
    maxit_sp   = params.setdefault('maxitsp', 50)
    m_hist     = params.setdefault('lbfgs_m', 10)
    t_prox     = params.setdefault('t_lbfgs', params.get('t', 1.0))  # scalar prox step
    c_armijo   = params.setdefault('backtrack_c', 1e-4)
    r_armijo   = params.setdefault('backtrack_r', 0.5)
    safeg      = params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))
    gradTol    = params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))
    # stopping on prox-gradient residual
    atol       = params.setdefault('atol', 1e-4)
    rtol       = params.setdefault('rtol', 1e-2)
    spexp      = params.setdefault('spexp', 2)

    model   = problem.obj_smooth
    phiobj  = problem.obj_nonsmooth
    pspace  = problem.pvector
    dspace  = problem.dvector

    # --- helpers ---
    def f_model(s):
        v, _ = model.value(x + s, 0.0); cnt['nobj1'] += 1
        return v

    def g_model(s):
        g, _ = model.gradient(x + s, gradTol); cnt['ngrad'] += 1
        return g

    def F_comp(s):
        # composite value at x+s
        y = x + s
        return f_model(s) + phiobj.value(y)

    def project_ball(s):
        nrm = pspace.norm(s)
        if nrm > Delta:
            return (Delta / max(1e-16, nrm)) * s
        return s

    # --- init at s=0 ---
    s    = np.zeros_like(x)
    m0   = f_model(s)     # val at x
    phi0 = phi            # caller already has phi(x)
    gk   = g_model(s)     # smooth grad at x
    if hasattr(dspace, "dual"):
        # keep using the same space logic you already have elsewhere
        dgk = dspace.dual(gk)
    else:
        dgk = gk

    # L-BFGS history
    S_hist, Y_hist, RHO_hist = [], [], []

    # stopping tolerance based on initial prox-grad norm proxy
    # we use ||prox(x - t*g) - x|| / t as a residual proxy
    y0   = phiobj.prox(x - t_prox * dgk, t_prox); cnt['nprox'] += 1
    r0   = pspace.norm(y0 - x) / max(1e-16, t_prox)
    gtol = min(atol, rtol * (r0 ** spexp))

    iterSP = 0
    iflag  = 1

    while iterSP < maxit_sp:
        iterSP += 1

        # --- two-loop recursion to get dQN = -H_k * grad(m)(x+s) ---
        q = gk.copy()
        alpha = []
        for Sk, Yk, rhok in zip(reversed(S_hist), reversed(Y_hist), reversed(RHO_hist)):
            a = rhok * np.dot(Sk, q)
            alpha.append(a)
            q -= a * Yk
        if Y_hist:
            sy = np.dot(S_hist[-1], Y_hist[-1]); yy = np.dot(Y_hist[-1], Y_hist[-1])
            H0 = sy / max(1e-16, yy)
        else:
            H0 = 1.0
        r = H0 * q
        for (Sk, Yk, rhok), a in zip(zip(S_hist, Y_hist, RHO_hist), reversed(alpha)):
            b = rhok * np.dot(Yk, r)
            r += Sk * (a - b)
        dQN = -r

        # Map direction into dual if prox expects dual-scaled input (your prox uses primal x and a scalar t)
        step = dQN

        # --- proximal trial with backtracking + TR projection ---
        alpha_bt = 1.0
        accepted = False
        m_curr   = m0
        while alpha_bt > 1e-12:
            s_try = s + alpha_bt * step
            # TR projection (before prox)
            s_try = project_ball(s_try)
            x_try = x + s_try
            # prox (composite) — use standard forward-backward pattern around x_try
            # (here we do a single prox to respect your nonsmooth)
            # Note: using t_prox as fixed scalar; works well for L1(+box).
            y_try = phiobj.prox(x_try, t_prox); cnt['nprox'] += 1
            s_try = y_try - x               # ensure exact feasibility for TR check next
            s_try = project_ball(s_try)     # (clip again if prox pushed slightly out)
            x_try = x + s_try

            m_try   = f_model(s_try)
            phi_try = phiobj.value(x_try); cnt['nobj2'] += 1
            F_try   = m_try + phi_try

            # Armijo condition on F around current (x+s)
            lin = c_armijo * np.dot(gk, (s_try - s))
            if F_try <= (m_curr + phi0) + lin:
                accepted = True
                break
            alpha_bt *= r_armijo

        # prox-gradient residual at (x+s): ||(prox(x+s - t g) - (x+s))/t||
        y_pg  = phiobj.prox(x + s - t_prox * dgk, t_prox); cnt['nprox'] += 1
        res   = pspace.norm(y_pg - (x + s)) / max(1e-16, t_prox)

        # stop if inner converged
        if res <= gtol:
            s, m0, phi0 = s_try, m_try, phi_try
            iflag = 0
            break

        # --- L-BFGS update on smooth part ---
        g_new = g_model(s_try)
        sk = s_try - s
        yk = g_new - gk
        sty = np.dot(sk, yk)
        if sty > 1e-12:
            if len(S_hist) == m_hist:
                S_hist.pop(0); Y_hist.pop(0); RHO_hist.pop(0)
            S_hist.append(sk); Y_hist.append(yk); RHO_hist.append(1.0 / sty)

        s, gk, m0, phi0 = s_try, g_new, m_try, phi_try

        # exit if we hit the TR boundary
        if pspace.norm(s) >= (1 - 1e-8) * Delta:
            iflag = 2
            break

    # --- finalize, compute predicted reduction like SPG2 ---
    snorm     = pspace.norm(s)
    x_fin     = x + s
    m_fin     = f_model(s)
    phi_fin   = phiobj.value(x_fin); cnt['nobj2'] += 1
    pRed      = (val + phi) - (m_fin + phi_fin)

    return s, snorm, pRed, phi_fin, iflag, iterSP, cnt, params
