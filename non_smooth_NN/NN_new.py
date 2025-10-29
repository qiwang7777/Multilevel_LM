import os
import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
from non_smooth.Problem import Problem
from non_smooth.modelTR import modelTR
import copy, numbers, re
import matplotlib.pyplot as plt
from typing import Optional
import collections, math, torch, time
from collections import OrderedDict
import torch.nn as nn
def _spg2_print_header(params, F0, res0, Delta, t):
    if not params.get('subsolver_print', False): return
    print("── SPG2-TR subsolver ─────────────────────────────────────────")
    print(f"    start: F0={F0:.6e}  ||g_prox||≈{res0:.3e}  Δ={Delta:.3e}  t={t:.3e}")
    print("    iter      res      ||s||/Δ   step(α)    ared         pred        g^Ts+Δφ    note")

def _spg2_print_row(params, k, res, s_over_D, alpha, aRed, pRed, gTs_plus_dphi, note=""):
    q = params.get('subsolver_outfreq', 1)
    if not params.get('subsolver_print', False): return
    if k % max(1,q) != 0: return
    print(f"{k:8d}  {res:8.3e}  {s_over_D:7.3f}   {alpha:7.3f}  "
          f"{aRed:+.3e}   {pRed:+.3e}   {gTs_plus_dphi:+.3e}   {note:>10s}")

def _spg2_print_footer(params, it, iflag, reason, aRed, F_end, snorm, Delta):
    if not params.get('subsolver_print', False): return
    print("    done: it={:d}  iflag={}  reason={}".format(it, iflag, reason))
    print("          aRed={:+.6e}  F_end={:.6e}  ||s||={:.3e}  ||s||/Δ={:.3f}".format(aRed, F_end, snorm, snorm/max(1e-16,Delta)))
    print("    ────────────────────────────────────────────────────────────────")

def trustregion_step_SPG2(x, val, grad, dgrad, phi, problem, params, cnt):
    # defaults
    params.setdefault('maxitsp', 10)
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1.0)
    params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))
    params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)
    params.setdefault('subsolver_print', False)
    params.setdefault('subsolver_outfreq', 1)

    Delta  = params.get('delta_effective', params['delta'])
    x0     = copy.deepcopy(x)
    g0_pr  = copy.deepcopy(grad)   # smooth grad in primal space
    snorm  = 0.0

    # initial GCP (gives s, predicted reduction, etc.)
    sc, snormc, pRed_gcp, _, Hs_gcp, cnt, params = trustregion_gcp2(x, val, grad, dgrad, phi, problem, params, cnt)

    t0     = params['t']
    s      = copy.deepcopy(sc)
    x1     = x0 + s
    gnorm  = snormc
    gtol   = np.min([params['atol'], params['rtol'] * (gnorm / max(1e-16, t0)) ** params['spexp']])

    # initial composite value and prox-residual for header
    F0     = val + phi
    y_pg0  = problem.obj_nonsmooth.prox(x0 - t0 * problem.dvector.dual(g0_pr), t0); cnt['nprox'] += 1
    res0   = problem.pvector.norm(y_pg0 - x0) / max(1e-16, t0)
    _spg2_print_header(params, F0, res0, Delta, t0)

    # Set exit flag
    iterSP  = 0
    iflag   = 1
    valold  = val
    phiold  = phi
    valnew  = val
    phinew  = phi

    for iter0 in range(1, params['maxitsp'] + 1):
        iterSP = iter0
        alphamax = 1.0
        snorm0   = snorm
        snorm    = problem.pvector.norm(x1 - x)

        # TR cap for linearized path
        if snorm >= (1 - params['safeguard']) * Delta:
            ds = problem.pvector.dot(s, x0 - x)
            dd = gnorm ** 2
            alphamax = np.minimum(1.0, (-ds + np.sqrt(max(0.0, ds**2 + dd * (Delta**2 - snorm0**2)))) / max(1e-16, dd))

        # quadratic model pieces
        Hs, _  = problem.obj_smooth.hessVec(s, x, params['gradTol']); cnt['nhess'] += 1
        sHs    = problem.dvector.apply(Hs, s)
        g0s    = problem.pvector.dot(g0_pr, s)
        phinew = problem.obj_nonsmooth.value(x1); cnt['nobj2'] += 1

        # one-step line search on the quadratic + φ
        if sHs <= params['safeguard']:
            alpha = alphamax
        else:
            alpha0 = -(g0s + (phinew - phiold)) / sHs
            alpha  = np.minimum(alphamax, alpha0)

        # Update iterate
        if alpha == 1.0:
            x0     = x1
            g0_pr += problem.dvector.dual(Hs)
            valnew = valold + g0s + 0.5 * sHs
        else:
            x0     += alpha * s
            g0_pr  += alpha * problem.dvector.dual(Hs)
            valnew  = valold + alpha * g0s + 0.5 * alpha**2 * sHs
            phinew  = problem.obj_nonsmooth.value(x0); cnt['nobj2'] += 1
            snorm   = problem.pvector.norm(x0 - x)

        # book-keeping
        aRed = (valold + phiold) - (valnew + phinew)  # actual reduction this inner step
        # predicted reduction for the *trial* step direction
        pRed = -(g0s + (phinew - phiold)) - 0.5 * sHs  # based on current model pieces
        gTs_plus_dphi = g0s + (phinew - phiold)

        _spg2_print_row(params, iter0,
                        res=problem.pvector.norm(problem.obj_nonsmooth.prox(x0 - t0 * problem.dvector.dual(g0_pr), t0) - x0) / max(1e-16, t0),
                        s_over_D=snorm / max(1e-16, Delta),
                        alpha=float(alpha),
                        aRed=aRed,
                        pRed=pRed,
                        gTs_plus_dphi=gTs_plus_dphi,
                        note=("on-TR" if snorm >= (1 - params['safeguard']) * Delta else "")
                        )

        # accept x1 move and refresh
        valold = valnew
        phiold = phinew

        # stop if TR boundary
        if snorm >= (1 - params['safeguard']) * Delta:
            iflag = 2
            break

        # spectral t
        if sHs <= params['safeguard']:
            lambdaTmp = params['t'] / max(1e-16, problem.pvector.norm(g0_pr))
        else:
            lambdaTmp = max(1e-16, gnorm**2) / max(1e-16, sHs)
        t0 = np.clip(lambdaTmp, params['lam_min'], params['lam_max'])

        # new prox point
        x1   = problem.obj_nonsmooth.prox(x0 - t0 * g0_pr, t0); cnt['nprox'] += 1
        s    = x1 - x0
        gnorm = problem.pvector.norm(s)

        # inner convergence: prox-grad residual
        if (gnorm / max(1e-16, t0)) <= np.min([params['atol'], params['rtol'] * (gnorm / max(1e-16, t0)) ** params['spexp']]):
            iflag = 0
            break

    # finalize for outer TR
    s     = x0 - x
    snorm = problem.pvector.norm(s)
    F_end = valnew + phinew
    pRed  = (val + phi) - F_end

    reason = {0: "inner tol met", 1: "max inner iterations or other", 2: "hit TR boundary"}.get(iflag, "unknown")
    _spg2_print_footer(params, iterSP, iflag, reason, aRed=(val + phi) - F_end, F_end=F_end, snorm=snorm, Delta=Delta)

    return s, snorm, pRed, phinew, iflag, iterSP, cnt, params


def trustregion_gcp2(x, val, grad, dgrad, phi, problem, params, cnt):
    params.setdefault('safeguard', np.sqrt(np.finfo(float).eps))
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1.0)
    params.setdefault('t_gcp', params['t'])
    params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))

    # Basic Cauchy/SPG step
    Hg, _       = problem.obj_smooth.hessVec(grad, x, params['gradTol']); cnt['nhess'] += 1
    gHg         = problem.dvector.apply(Hg, grad)
    gg          = problem.pvector.dot(grad, grad)
    if gHg > params['safeguard'] * gg:
        t0Tmp = gg / gHg
    else:
        t0Tmp = params['t'] / max(1e-16, np.sqrt(gg))
    t0 = np.clip(t0Tmp, params['lam_min'], params['lam_max'])

    xc        = problem.obj_nonsmooth.prox(x - t0 * dgrad, t0); cnt['nprox'] += 1
    s         = xc - x
    snorm     = problem.pvector.norm(s)
    Hs, _     = problem.obj_smooth.hessVec(s, x, params['gradTol']); cnt['nhess'] += 1
    sHs       = problem.dvector.apply(Hs, s)
    gs        = problem.pvector.dot(grad, s)
    phinew    = problem.obj_nonsmooth.value(xc); cnt['nobj2'] += 1

    alpha = 1.0
    if snorm >= (1 - params['safeguard']) * params['delta']:
        alpha = min(1.0, params['delta'] / max(1e-16, snorm))
    if sHs > params['safeguard']:
        alpha = min(alpha, -(gs + (phinew - phi)) / sHs)

    if alpha != 1.0:
        s      *= alpha
        snorm  *= alpha
        gs     *= alpha
        Hs     *= alpha
        sHs    *= alpha**2
        xc      = x + s
        phinew  = problem.obj_nonsmooth.value(xc); cnt['nobj2'] += 1

    valnew = val + gs + 0.5 * sHs
    pRed   = (val + phi) - (valnew + phinew)

    params['t_gcp'] = t0
    return s, snorm, pRed, phi, Hs, cnt, params

def trustregion_step_LBFGS2(x, val, dgrad, phi, problem, params, cnt):
    """
    Proximal L-BFGS subsolver for the TR subproblem:
        min_s  m(x+s) + phi(x+s)  s.t. ||s|| <= delta_effective

    Returns: s, snorm, pRed, phinew, iflag, iter_count, cnt, params

    Diagnostics (controlled by params):
      - subsolver_print: bool (default True)
      - subsolver_outfreq: int (default 1)
    """
    import numpy as np

    # ---------------- params / defaults ----------------
    Delta      = params.get('delta_effective', params['delta'])
    maxit_sp   = params.setdefault('maxitsp', 50)
    m_hist     = params.setdefault('lbfgs_m', 10)
    t_prox     = params.setdefault('t_lbfgs', params.get('t', 1.0))  # prox stepsize
    c_armijo   = params.setdefault('backtrack_c', 1e-4)
    r_armijo   = params.setdefault('backtrack_r', 0.5)
    gradTol    = params.setdefault('gradTol', np.sqrt(np.finfo(float).eps))
    atol       = params.setdefault('atol', 1e-4)
    rtol       = params.setdefault('rtol', 1e-2)
    spexp      = params.setdefault('spexp', 2)

    # printing controls
    do_print   = params.setdefault('subsolver_print', True)
    outfreq    = params.setdefault('subsolver_outfreq', 1)

    model   = problem.obj_smooth
    phiobj  = problem.obj_nonsmooth
    pspace  = problem.pvector
    dspace  = problem.dvector

    # ------- vector helpers (respect problem geometry) -------
    Z = x - x  # zero with same structure as x
    def vcopy(v): return v + Z
    def vdot(a,b): return pspace.dot(a,b)
    def vnorm(v):  return pspace.norm(v)

    def project_ball(s):
        nrm = vnorm(s)
        if nrm > Delta:
            s = s * (Delta / max(1e-16, float(nrm)))
        return s

    def f_model(s):
        v, _ = model.value(x + s, 0.0); cnt['nobj1'] += 1
        return v

    def g_model(s):
        g, _ = model.gradient(x + s, gradTol); cnt['ngrad'] += 1
        return g

    # --- init at s=0 ---
    s    = vcopy(Z)
    m0   = f_model(s)        # m(x)
    phi0 = phi               # φ(x) (caller provided)
    gk   = g_model(s)        # ∇m(x)
    dgk  = dspace.dual(gk) if hasattr(dspace, "dual") else gk

    # L-BFGS history
    S_hist, Y_hist, RHO_hist = [], [], []

    # stopping tolerance based on initial prox-grad norm proxy at x
    y0   = phiobj.prox(x - t_prox * dgk, t_prox); cnt['nprox'] += 1
    r0   = vnorm(y0 - x) / max(1e-16, t_prox)
    gtol = min(atol, rtol * (r0 ** spexp))

    if do_print:
        print("    ── LBFGS-TR subsolver ─────────────────────────────────────────")
        print(f"    start: F0={m0+phi0:.6e}  ||g_prox||≈{r0:.3e}  Δ={Delta:.3e}  t={t_prox:.3e}")
        print("    iter    res         ||s||/Δ   step(α)   F_try-LHS      RHS(Armijo)    accept  note")

    iterSP = 0
    iflag  = 1
    stop_note = ""

    while iterSP < maxit_sp:
        iterSP += 1

        # ---- two-loop recursion: dQN ~ -H_k * ∇m(x+s) ----
        q = vcopy(gk)
        alpha_hist = []
        for Sk, Yk, rhok in zip(reversed(S_hist), reversed(Y_hist), reversed(RHO_hist)):
            a = rhok * vdot(Sk, q)
            alpha_hist.append(a)
            q -= a * Yk
        if Y_hist:
            sy = vdot(S_hist[-1], Y_hist[-1]); yy = vdot(Y_hist[-1], Y_hist[-1])
            H0 = sy / max(1e-16, yy)
        else:
            H0 = 1.0
        r = H0 * q
        for (Sk, Yk, rhok), a in zip(zip(S_hist, Y_hist, RHO_hist), reversed(alpha_hist)):
            b = rhok * vdot(Yk, r)
            r += Sk * (a - b)

        dQN  = r * (-1)                # descent direction in primal space
        step = dQN

        # ---- backtracking prox line-search with TR projection ----
        alpha_bt = 1.0
        accepted = False
        bt_tries = 0
        s_try = s
        m_try, phi_try, F_try = m0, phi0, m0 + phi0
        armijo_lhs = 0.0
        armijo_rhs = 0.0

        while alpha_bt > 1e-12:
            bt_tries += 1
            # trial step in the ball
            s_try = project_ball(s + alpha_bt * step)
            x_try_pre = x + s_try

            # proximal forward-backward (gradient at z = x+s)
            y_try = phiobj.prox(x_try_pre - t_prox * dgk, t_prox); cnt['nprox'] += 1

            # project exact step back to ball (prox can push slightly out)
            s_try = project_ball(y_try - x)
            x_try = x + s_try

            m_try   = f_model(s_try)
            phi_try = phiobj.value(x_try); cnt['nobj2'] += 1
            F_try   = m_try + phi_try

            # Armijo: F(x+s_try) <= F(x+s) + c * <∇m(x+s), s_try - s>
            armijo_lhs = F_try
            armijo_rhs = (m0 + phi0) + c_armijo * vdot(gk, (s_try - s))

            if F_try <= armijo_rhs:
                accepted = True
                break
            alpha_bt *= r_armijo

        # ---- prox-gradient residual at z = x+s (recompute gradient at z) ----
        gk  = g_model(s)                # ∇m(x+s)
        dgk = dspace.dual(gk) if hasattr(dspace, "dual") else gk
        y_pg = phiobj.prox((x + s) - t_prox * dgk, t_prox); cnt['nprox'] += 1
        res  = vnorm(y_pg - (x + s)) / max(1e-16, t_prox)

        # pretty print this iteration
        if do_print and (iterSP % outfreq == 0):
            snorm_ratio = vnorm(s_try) / max(1e-16, Delta)
            note = []
            if not accepted: note.append("no-accept")
            if snorm_ratio >= 0.999: note.append("on-TR")
            print(f"    {iterSP:4d}  {res:9.3e}  {snorm_ratio:7.3f}   {alpha_bt:7.3f}  "
                  f"{(armijo_lhs - (m0+phi0)):>+11.3e}  "
                  f"{(armijo_rhs - (m0+phi0)):>+11.3e}   "
                  f"{'Y' if accepted else 'N'}     {';'.join(note)}")

        # stopping: prox-gradient residual
        if res <= gtol:
            s, m0, phi0 = s_try, m_try, phi_try
            iflag = 0
            stop_note = "prox-gradient tol met"
            break

        # ---- L-BFGS update on smooth part using accepted trial (or last tried) ----
        g_new = g_model(s_try)          # ∇m(x+s_try)
        sk = s_try - s
        yk = g_new - gk
        sty = vdot(sk, yk)
        if sty > 1e-12:
            if len(S_hist) == m_hist:
                S_hist.pop(0); Y_hist.pop(0); RHO_hist.pop(0)
            S_hist.append(sk); Y_hist.append(yk); RHO_hist.append(1.0 / sty)

        # accept the trial point state for next iter
        s, m0, phi0 = s_try, m_try, phi_try
        gk          = g_new
        dgk         = dspace.dual(gk) if hasattr(dspace, "dual") else gk

        # exit if we hit the TR boundary
        if vnorm(s) >= (1 - 1e-8) * Delta:
            iflag = 2
            stop_note = "hit TR boundary"
            break

    # max-iter stop
    if iflag == 1 and iterSP >= maxit_sp and stop_note == "":
        stop_note = "max inner iterations"

    # --- finalize, compute predicted & actual reduction for report ---
    snorm     = vnorm(s)
    x_fin     = x + s
    m_fin     = f_model(s)
    phi_fin   = phiobj.value(x_fin); cnt['nobj2'] += 1
    F_start   = val + phi
    F_end     = m_fin + phi_fin
    aRed_sub  = F_start - F_end                 # actual reduction on subproblem
    pRed      = aRed_sub                        # (we used final values as "model" end)

    if do_print:
        ratio = snorm / max(1e-16, Delta)
        print(f"    done: it={iterSP}  iflag={iflag}  reason={stop_note}")
        print(f"          aRed={aRed_sub:+.6e}  F_end={F_end:.6e}  ||s||={snorm:.3e}  ||s||/Δ={ratio:.3f}")
        print("    ────────────────────────────────────────────────────────────────")

    return s, snorm, pRed, phi_fin, iflag, iterSP, cnt, params

def trustregion_step_LBFGS2_withoutprint(x, val, dgrad, phi, problem, params, cnt):
    """
    Proximal L-BFGS subsolver for the TR subproblem:
        min_s  m(x+s) + phi(x+s)  s.t. ||s|| <= delta_effective

    Matches trustregion_step_SPG2 signature.
    Returns: s, snorm, pRed, phinew, iflag, iter_count, cnt, params
    """

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
    x0 = copy.deepcopy(x)
    g0 = copy.deepcopy(dgrad)
    snorm = 0
    Z = x-x
    def vcopy(v):
        return v+Z
    def vdot(a,b):
        return pspace.dot(a,b)
    def vnorm(v):
        return pspace.norm(v)
    def project_ball(s):
        nrm = vnorm(s)
        if nrm > Delta:
            s = s * (Delta / max(1e-16,float(nrm)))
        return s

    

    def f_model(s):
        v, _ = model.value(x + s, 0.0); cnt['nobj1'] += 1
        return v

    def g_model(s):
        g, _ = model.gradient(x + s, gradTol); cnt['ngrad'] += 1
        return g

    # --- init at s=0 ---
    s = vcopy(Z)
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
    r0   = vnorm(y0 - x) / max(1e-16, t_prox)
    gtol = min(atol, rtol * (r0 ** spexp))

    iterSP = 0
    iflag  = 1

    while iterSP < maxit_sp:
        iterSP += 1

        # --- two-loop recursion to get dQN = -H_k * grad(m)(x+s) ---
        q = vcopy(gk)
        alpha = []
        for Sk, Yk, rhok in zip(reversed(S_hist), reversed(Y_hist), reversed(RHO_hist)):
            a = rhok * vdot(Sk, q)
            alpha.append(a)
            q -= a * Yk
        if Y_hist:
            sy = vdot(S_hist[-1], Y_hist[-1]); yy = vdot(Y_hist[-1], Y_hist[-1])
            H0 = sy / max(1e-16, yy)
        else:
            H0 = 1.0
        r = H0 * q
        for (Sk, Yk, rhok), a in zip(zip(S_hist, Y_hist, RHO_hist), reversed(alpha)):
            b = rhok * vdot(Yk, r)
            r += Sk * (a - b)
        dQN = r*(-1)

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
            #y_try = phiobj.prox(x_try - t_prox*dgk, t_prox); cnt['nprox'] += 1
            y_try = phiobj.prox(x_try, t_prox); cnt['nprox'] += 1
            s_try = project_ball(y_try - x)              # ensure exact feasibility for TR check next
            s_try = project_ball(s_try)     # (clip again if prox pushed slightly out)
            x_try = x + s_try

            m_try   = f_model(s_try)
            phi_try = phiobj.value(x_try); cnt['nobj2'] += 1
            F_try   = m_try + phi_try

            # Armijo condition on F around current (x+s)
            lin = c_armijo * vdot(gk, (s_try - s))
            if F_try <= (m_curr + phi0) + lin:
                accepted = True
                break
            alpha_bt *= r_armijo
        if not accepted:
            s_try = project_ball(s+step*0.0)
            x_try = x + s_try
            m_try = m0
            phi_try = phi0
        gk = g_model(s)
        dgk = dspace.dual(gk) if hasattr(dspace,"dual") else gk
        y_pg = phiobj.prox((x+s)-t_prox*dgk,t_prox)
        cnt['nprox'] += 1
        res = vnorm(y_pg-(x+s))/max(1e-16,t_prox)
        if res<= gtol:
            s,m0,phi0 = s_try, m_try,phi_try
            iflag = 0
            break

       

        # --- L-BFGS update on smooth part ---
        g_new = g_model(s_try)
        sk = s_try - s
        yk = g_new - gk
        sty = vdot(sk, yk)
        if sty > 1e-12:
            if len(S_hist) == m_hist:
                S_hist.pop(0); Y_hist.pop(0); RHO_hist.pop(0)
            S_hist.append(sk); Y_hist.append(yk); RHO_hist.append(1.0 / sty)

        s, gk, m0, phi0 = s_try, g_new, m_try, phi_try

        # exit if we hit the TR boundary
        if vnorm(s) >= (1 - 1e-8) * Delta:
            iflag = 2
            break

    # --- finalize, compute predicted reduction like SPG2 ---
    snorm     = vnorm(s)
    x_fin     = x + s
    m_fin     = f_model(s)
    phi_fin   = phiobj.value(x_fin); cnt['nobj2'] += 1
    pRed      = (val + phi) - (m_fin + phi_fin)

    return s, snorm, pRed, phi_fin, iflag, iterSP, cnt, params

def _act(name: str):
    if name == "tanh":   return nn.Tanh()
    if name == "relu":   return nn.ReLU()
    if name == "gelu":   return nn.GELU()
    if name == "swish":  return nn.SiLU()
    raise ValueError(f"unknown activation {name}")

class FeatureLinear(nn.Module):
    """
    net.0: Linear(in, hidden)   <-- frozen (feature extractor)
    net.1: Activation
    net.2: Linear(hidden, out)  <-- trainable head (purely linear)
    """
    def __init__(self, dims, activation="tanh", freeze_body=True, head_bias=True):
        super().__init__()
        in_dim, hidden, out_dim = map(int, dims)
        self.net = nn.Sequential(OrderedDict([
            ("0", nn.Linear(in_dim, hidden, bias=True)),  # body
            ("1", _act(activation)),
            ("2", nn.Linear(hidden, out_dim, bias=head_bias)),  # head
        ]))

        if freeze_body:
            # freeze *only* the body (layer 0). activation has no params.
            for name, p in self.net.named_parameters():
                if name.startswith("0."):   # '0.weight' or '0.bias'
                    p.requires_grad_(False)

    def forward(self, x):
        return self.net(x)

def freeze_body(model: FeatureLinear):
    for p in model.body.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
def split_state_dict_for_head(model: FeatureLinear):
    sd = model.state_dict()
    head_sd   = {k: v.clone() for k, v in sd.items() if k.startswith("head.")}
    frozen_sd = {k: v.clone() for k, v in sd.items() if not k.startswith("head.")}
    return head_sd, frozen_sd

def merge_params(frozen_sd, head_sd):
    full = {k: v for k, v in frozen_sd.items()}
    full.update(head_sd)
    return full

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

class TorchVect:
    @torch.no_grad()
    def __init__(self, tensordict, isRop: bool = False, shapes_map=None):
        self.td = OrderedDict(tensordict)           # preserve order
        self.isRop = isRop
        self.shapes_map = dict(shapes_map) if shapes_map is not None else {}

        # Collect metadata once (handles scalars with shape == ())
        self.names  = list(self.td.keys())
        self.shapes = []
        self.sizes  = []
        for k, v in self.td.items():
            if not torch.is_tensor(v):
                raise TypeError(f"Param '{k}' is not a tensor.")
            self.shapes.append(self.shapes_map.get(k, v.shape))
            self.sizes.append(v.numel())

        self.numel     = int(sum(self.sizes))
        self.n_params  = len(self.names)

        self.sizes_per_param = list(self.sizes)
        self.shape = [self.numel]
        
    @torch.no_grad()
    def clone(self):
        td = {
            k: (v.clone().detach().requires_grad_(v.requires_grad) if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in self.td.items()
        }
    # IMPORTANT: do NOT zero here; it breaks any code that clones a gradient/vector/operator
        return TorchVect(td, isRop=self.isRop, shapes_map=self.shapes_map)
    def __deepcopy__(self, memo):
        """Support copy.deepcopy by using our clone method"""
        return self.clone()

    @property
    def T(self):
        Temp = self.clone()
        for k, v in list(self.td.items()):
            Temp.td[k] = v.T
        return Temp

    @torch.no_grad()
    def zero(self):
        for _, v in self.td.items():
            if torch.is_tensor(v):
                v.zero_()
    @torch.no_grad()
    def add_(self,other,alpha=1.0):
        if not isinstance(other,TorchVect):
            raise TypeError("add_: other must be a TorchVect")
        if self.isRop!= other.isRop:
            raise RuntimeError("add_: cannot add operator to parameter vector (isRop mismatch)")
            
        for k in self.td.keys():
            a = self.td[k]
            b = other.td[k]
            if not torch.is_tensor(a) or not torch.is_tensor(b):
                raise TypeError(f"add_: non-tensor entry for key'{k}'")
                
            if a.shape != b.shape:
                raise RuntimeError(f"add_: shape mismatch for '{k}':{a.shape} vs {b.shape}")
            self.td[k] = a.add(b.to(dtype=a.dtype,device=a.device),alpha=float(alpha))
        return self

    @torch.no_grad()
    def __add__(self, other):
        out = self.clone()
        out.add_(other,alpha=1.0)
        return out

    @torch.no_grad()
    def __sub__(self, other):
        out = self.clone()
        out.add_(other,alpha=-1.0)
        return out

    @torch.no_grad()
    def __mul__(self, alpha):
        ans = self.clone()
        for k, v in self.td.items():
            if not torch.is_tensor(v):
                raise TypeError(f"__mul__: non-tensor entry for key '{k}]")
            ans.td[k] = v*float(alpha)
        return ans

    @torch.no_grad()
    def __rmul__(self, alpha):
        return self.__mul__(alpha)

    @torch.no_grad()
    def __matmul__(self, x):
        """
        If self.isRop == True and x is a TorchVect of parameters:
            For each param k:
              y_k = R_k @ vec(x_k)
              reshape y_k to self.shapes_map[k]
        """
        assert isinstance(x, TorchVect), "Right operand must be TorchVect"
        if not self.isRop:
            raise RuntimeError("Left operand must be an operator (isRop=True).")

        if x.isRop:
            ans = TorchVect(OrderedDict(), isRop=True, shapes_map={})
            for k in self.td.keys():
                L = self.td[k]                # (m x p)
                
                R = x.td[k]                   # (p x q)
                if L.shape[1] != R.shape[0]:
                    raise RuntimeError(f"Operator compose mismatch for '{k}': "
                                       f"L cols {L.shape[1]} != R rows {R.shape[0]}")
                ans.td[k] = L @ R             # (m x q)
                # output shape stays that of the LEFT operator
                ans.shapes_map[k] = self.shapes_map.get(k, x.shapes_map.get(k, None))
            return ans

        # Case 2: apply operator to parameter vector (matrix–vector per parameter)
        ans = x.clone()
        for k, v in x.td.items():
            Rk = self.td[k]                  # (out_size x in_size)
            vec = v.reshape(-1)              # (in_size,)
            if Rk.shape[1] != vec.numel():
                raise RuntimeError(f"Op '{k}' expects input {Rk.shape[1]}, got {vec.numel()}")
            y = Rk.to(vec.dtype).to(vec.device) @ vec
            out_shape = self.shapes_map.get(k, v.shape)
            if int(torch.tensor(out_shape).prod().item()) != y.numel():
                raise RuntimeError(f"Op '{k}' output size {y.numel()} cannot reshape to {out_shape}")
            ans.td[k] = y.reshape(out_shape)
        ans.isRop = False
        return ans

    @torch.no_grad()
    def __truediv__(self, alpha):
        ans = self.clone()
        inv = 1.0/float(alpha)
        for k, v in ans.td.items():
            if not torch.is_tensor(v):
                raise TypeError(f"__truediv__: non-tensor entry for key '{k}'")
            ans.td[k] = v*inv
        return ans

    @torch.no_grad()
    def __rtruediv__(self, alpha):
        return self.__truediv__(alpha)

class L1TorchNorm:
    """
    L1 penalty over a *selected* subset of parameters.
    - include_names: iterable of parameter names to which L1 is applied.
      If None, it uses all model params with requires_grad=True.
    - var['beta']: L1 weight (float).
    - var['l1_exclude']: tuple of substrings; any param name containing one is skipped.
      Defaults to ('bias','cB').
    """
    def __init__(self, var, include_names=None):
        self.var      = var
        self.beta     = float(var.get('beta', 0.0))
        self.exclude  = tuple(var.get('l1_exclude', ('bias', 'cB')))
        model         = var["NN"]
        if include_names is None:
            self.trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        else:
            self.trainable = set(include_names)

    # ----- helpers -----
    def _active(self, name: str) -> bool:
        return (name in self.trainable) and (not any(ex in name for ex in self.exclude))

    def _as_td(self, x):
        return x.td if hasattr(x, "td") else x

    # ----- objective interface -----
    def value(self, x):
        td = self._as_td(x)
        total = 0.0
        for k, v in td.items():
            if torch.is_tensor(v) and self._active(k):
                total += v.abs().sum()
        return float(self.beta * (total.item() if torch.is_tensor(total) else total))

    def prox(self, x, t):
        """Soft-threshold on the active set only: prox_{t*beta*||.||_1}(x)."""
        lam = float(self.beta) * float(t)
        out = x.clone() if hasattr(x, "clone") else copy.deepcopy(x)
        td_in  = self._as_td(x)
        td_out = self._as_td(out)
        for k, v in td_in.items():
            if torch.is_tensor(v) and self._active(k) and lam > 0.0:
                td_out[k] = v.sign() * (v.abs() - lam).clamp_min(0.0)
            else:
                td_out[k] = v.clone() if torch.is_tensor(v) else v
        return out

    # ----- optional extras used by some solvers -----
    def dir_deriv(self, s, x):
        td_x = self._as_td(x)
        td_s = self._as_td(s)
        tot = 0.0
        for k, xk in td_x.items():
            if not torch.is_tensor(xk) or not self._active(k):
                continue
            sk = td_s[k]
            contrib = torch.where(xk != 0.0, xk.sign() * sk, sk.abs())
            tot += contrib.sum()
        return float(self.beta * tot.item())

    def project_sub_diff(self, g, x):
        """Project g onto ∂(beta * ||x||_1) at x for the active set; zero elsewhere."""
        td_x = self._as_td(x)
        td_g = self._as_td(g)
        out  = g.clone() if hasattr(g, "clone") else copy.deepcopy(g)
        td_o = self._as_td(out)
        b = float(self.beta)
        for k in td_x.keys():
            gx = td_g[k]
            xv = td_x[k]
            if not torch.is_tensor(xv) or not self._active(k):
                td_o[k] = torch.zeros_like(gx) if torch.is_tensor(gx) else gx
                continue
            s = xv.sign()
            td_o[k] = b * s + (1.0 - s.abs()) * gx.clamp(min=-b, max=b)
        return out

    def gen_jac_prox(self, x, t):
        """Diagonal mask of prox Jacobian (1 where |x|>beta*t on active set; 1 elsewhere if inactive)."""
        lam = float(self.beta) * float(t)
        td = self._as_td(x)
        mask = {}
        for k, v in td.items():
            if not torch.is_tensor(v):
                mask[k] = v
            elif not self._active(k):
                mask[k] = torch.ones_like(v)  # prox is identity on inactive params
            else:
                mask[k] = (v.abs() > lam).to(v.dtype)
        return mask

    def apply_prox_jacobian(self, v, x, t):
        lam = float(self.beta) * float(t)
        out = v.clone() if hasattr(v, "clone") else copy.deepcopy(v)
        td_v = self._as_td(v)
        td_x = self._as_td(x)
        td_o = self._as_td(out)
        for k in td_v.keys():
            vv = td_v[k]; xx = td_x[k]
            if not torch.is_tensor(vv) or not self._active(k):
                td_o[k] = vv.clone() if torch.is_tensor(vv) else vv
            else:
                td_o[k] = vv * (xx.abs() > lam).to(vv.dtype)
        return out

    def get_parameter(self):
        return self.beta


class L2TVPrimal:
    """
    L2 inner-product (primal metric) restricted to a given active set.
    This keeps all dot/norm calculations from ‘seeing’ frozen parameters.
    """
    def __init__(self, var, include_names=None):
        self.var = var
        model = var["NN"]
        if include_names is None:
            self.active = {n for n, p in model.named_parameters() if p.requires_grad}
        else:
            self.active = set(include_names)

    @torch.no_grad()
    def dot(self, x, y):
        td_x = x.td if hasattr(x, "td") else x
        td_y = y.td if hasattr(y, "td") else y
        tot = 0.0
        for k in td_x.keys():
            if k in self.active and torch.is_tensor(td_x[k]):
                tot += (td_x[k] * td_y[k]).sum()
        return float(tot.item())

    @torch.no_grad()
    def apply(self, x, y):
        # In your code, apply() is used as an inner product; keep that behavior.
        return self.dot(x, y)

    @torch.no_grad()
    def norm(self, x):
        return float(self.dot(x, x) ** 0.5)

    @torch.no_grad()
    def dual(self, x):
        # Identity for L2; if you ever add nontrivial preconditioning, change this.
        return x


class L2TVDual:
    """Same active-set aware L2 metric for the dual space (kept identical to primal here)."""
    def __init__(self, var, include_names=None):
        self.var = var
        model = var["NN"]
        if include_names is None:
            self.active = {n for n, p in model.named_parameters() if p.requires_grad}
        else:
            self.active = set(include_names)

    @torch.no_grad()
    def dot(self, x, y):
        td_x = x.td if hasattr(x, "td") else x
        td_y = y.td if hasattr(y, "td") else y
        tot = 0.0
        for k in td_x.keys():
            if k in self.active and torch.is_tensor(td_x[k]):
                tot += (td_x[k] * td_y[k]).sum()
        return float(tot.item())

    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x, y)

    @torch.no_grad()
    def norm(self, x):
        return float(self.dot(x, x) ** 0.5)

    @torch.no_grad()
    def dual(self, x):
        return x



class PhiCounter:
    def __init__(self, base_phi):
        self.base = base_phi
        self.nobj2 = 0
        self.nprox = 0
    def value(self, x):
        v = self.base.value(x)
        self.nobj2 += 1
        return v
    def prox(self, u, t):
        p = self.base.prox(u, t)
        self.nprox += 1
        return p
    def gen_jac_prox(self, x, t):       return self.base.gen_jac_prox(x, t)
    def apply_prox_jacobian(self, v,x,t): return self.base.apply_prox_jacobian(v, x, t)
    def get_parameter(self):            return self.base.get_parameter()
    def addCounter(self, cnt):
        cnt["nobj2"] += self.nobj2
        cnt["nprox"] += self.nprox
        return cnt
    

def set_default_parameters(name):
    params = {}

    # General Parameters
    params['spsolver']  = name.replace(' ', '')
    params['outFreq']   = 1
    params['debug']     = False
    params['initProx']  = False
    params['t']         = 1
    params['safeguard'] = np.sqrt(np.finfo(float).eps)

    # Stopping tolerances
    params['maxit']   = 200
    params['reltol']  = False
    params['gtol']    = 1e-8
    params['stol']    = 1e-10
    params['ocScale'] = params['t']

    # Trust-region parameters
    params['eta1']     = 0.05
    params['eta2']     = 0.7
    params['gamma1']   = 0.25
    params['gamma2']   = 1.5
    params['delta']    = 1e2
    params['deltamax'] = 1e10

    # Subproblem solve tolerances
    params['atol']    = 1e-3
    params['rtol']    = 1e-1
    params['spexp']   = 1
    params['maxitsp'] = 100

    # GCP and subproblem solve parameter
    params['useGCP']    = False
    params['mu1']       = 1e-4
    params['beta_dec']  = 0.1
    params['beta_inc']  = 10.0
    params['maxit_inc'] = 2

    # SPG and spectral GCP parameters
    params['lam_min'] = 1e-12
    params['lam_max'] = 1e12

    # Inexactness parameters
    params['useInexactObj']  = False
    params['useInexactGrad'] = False
    params['gradTol']        = np.sqrt(np.finfo(float).eps)

    # Recursive Parameters
    params['RgnormScale']      = 1e-1 # is v in Rgnorm >= v*gnorm -> relative R-step flag
    params['RgnormScaleTol']   = 5e2  # is v in Rgnorm >= v^i*gtol -> absolute R-step flag

    # Debug / diagnostics (all optional; default off)
    params.setdefault('debug_drop_gate', False)      # print drop-gate info each iter
    params.setdefault('debug_h_equiv', False)        # print h-equivalence diagnostic
    params.setdefault('debug_h_equiv_freq', 1)       # how often to print (every k iters)

    # Numerical guards/tolerances
    params.setdefault('prox_equiv_abs_tol', 1e-10)   # tight-frame prox identity tolerance
    params.setdefault('min_drop_cap', 1e-8)          # min parent cap to allow dropping

    # Existing gate scalars (set if not already)
    params.setdefault('RgnormScale', 1.0)            # κ_g
    params.setdefault('RgnormScaleTol', 100.0)       # ε_{i-1} scaling for coarse solve
    
    return params

class phiPrec:
    """
    
    Parameters
    ----------
    problem: the previous level {i-1} Problem.
    R: ndarray of shape (n_i,n_{i-1}), Restriction matrix from level i-1 (fine) to level i (coarse)
    l: int, current level index i >= 0
    x_fine : ndarray | None , The special x_{i-1,k} (on level i-1) to define x_{i,0} = R @ x_fine
    x_tol: float, Tolerance for detecting x = x_{i,0}.
    assume_tight_rows: bool, If True, enable the closed-form prox for the composition branch.
    
    """
    def __init__(self, problems, R:np.ndarray,l: int, x_fine:Optional[np.ndarray]=None, x_tol:float=1e-12, assume_tight_rows:bool=True, use_anchor_in_prox:bool=False):
        self.problems = problems
        self.l = l
        self.R = None if self.l == 0 else R
        self.nobj2 = 0
        self.nprox = 0
        self.x_fine = None if x_fine is None else x_fine
        self.x_tol = float(x_tol)
        self.assume_tight_rows = bool(assume_tight_rows)
        self._x_coarse = None
        self.use_anchor_in_prox = bool(use_anchor_in_prox)
        if self.l>0 and self.R is not None and self.x_fine is not None:
            self._x_coarse = self.R @ self.x_fine
            
    def update_x(self,x_fine:np.ndarray) -> None:
        """
        Update the special fine point x_{i-1,k} and cache x_{i,0} = R @ x_fine

        """
        self.x_fine = x_fine
        if self.l>0 and self.R is not None:
            self._x_coarse = self.R @self.x_fine
        
    def _is_anchor(self,x:np.ndarray) -> bool:

        if getattr(self,"_x_coarse",None) is None:
            return False
        diff = x - self._x_coarse
        return self.problems[self.l].pvector.norm(diff) <= self.x_tol
    
    def value(self,x:np.ndarray) -> float:
        """
       φ_{c}(x) = φ_{f}(x_f+R^T(x-x_c)) with x_c = Rx_f
     
        """
        if self.l == 0 or self.R is None:
            val = self.problems[0].obj_nonsmooth.value(x)
        else:
            if self._is_anchor(x):
                val = self.problems[self.l-1].obj_nonsmooth.value(self.x_fine)
            else:
                prev = self.problems[self.l-1].obj_nonsmooth
                x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
                y = self.x_fine +self.R.T @ (x-x_coarse)
                val = prev.value(y)
        self.nobj2 += 1
        return val
    def prox(self, u:np.ndarray,t:float) -> np.ndarray:
        """
        prox_{t φ_i}(u) with φ_i(x)=φ_{i-1}( x_fine + R^T (x - x_coarse) )
        Closed-form under RR^T=I:
          let v = u - x_coarse and y = x_fine + R^T v (fine-space),
          then
            z = v + R ( prox_{t φ_{i-1}}(y) - y )
            x = x_coarse + z

        """
        if not self.assume_tight_rows:
            raise NotImplementedError("Closed-form prox requires R R.T=I. Provided a numerical solver otherwise.")
        if self.l == 0 or self.R is None:
            px = self.problems[0].obj_nonsmooth.prox(u,t)
            self.nprox += 1
            return px
        
       
        else:
            
            n_coarse, n_fine = self.R.shape
            if u.shape[0] != n_coarse:
                 raise ValueError(
                    f"phiPrec(prox): level {self.l}: got u.shape[0]={u.shape[0]} but R is {n_coarse}×{n_fine}. "
                    f"u must be the **coarse** vector of length {n_coarse}. "
                    f"Likely the caller passed a fine-level vector or attached phiPrec to the wrong problem."
                )
            prev = self.problems[self.l-1].obj_nonsmooth
            x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
            v = u - x_coarse
            y = self.x_fine +self.R.T @ v
            py = prev.prox(y,t)
            z = v + self.R @(py-y)
            xcomp = x_coarse+z
            if not self.use_anchor_in_prox or self._x_coarse is None:
                self.nprox +=1
                return xcomp
            Fcomp = 0.5 / t * np.linalg.norm(xcomp-u)**2 +prev.value(self.x_fine+self.R.T@(xcomp-x_coarse))
            Fanc = 0.5 / t * np.linalg.norm(self._x_coarse-u)**2+prev.value(self.x_fine)
            px = self._x_coarse if (Fanc +1e-12<Fcomp) else xcomp
            
            self.nprox += 1
            return px
    
    def addCounter(self, cnt):
        cnt["nobj2"] += self.nobj2
        cnt["nprox"] += self.nprox
        return cnt
    
    def genJacProx(self, x, t):
        if self.l == 0 or self.R is None:
            return self.problems[0].obj_nonsmooth.gen_jac_prox(x, t)
        prev = self.problems[self.l-1].obj_nonsmooth
        x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
        y = self.x_fine + self.R.T @ (x - x_coarse)
        D_prev, ind_prev = prev.gen_jac_prox(y, t)
        if D_prev is None: return None, None
        D_prev = np.asarray(D_prev)
        n_i = self.R.shape[0]
        J = np.eye(n_i) + self.R @ (D_prev - np.eye(D_prev.shape[0])) @ self.R.T
        ind = None
        if ind_prev is not None:
            ind = (np.abs(self.R @ ind_prev.astype(float)) > 0).ravel()
        return J, ind
    
    
    def applyProxJacobian(self, v, x, t):
        if self.l == 0 or self.R is None:
            return self.problems[0].obj_nonsmooth.apply_prox_jacobian(v,x,t)
        x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
        y = self.x_fine + self.R.T @ (x - x_coarse)
        Jprev_v = self.problems[self.l-1].obj_nonsmooth.apply_prox_jacobian(self.R.T @ v, y, t)
        #Dv = self.problems[self.l].obj_nonsmooth.applyProxJacobian(v, x, t)
        return v + self.R@(Jprev_v-self.R.T @ v)
    
    def getParameter(self):
        #return self.problem.obj_nonsmooth.getParameter()
        return (self.problems[0].obj_nonsmooth.get_parameter() 
                if self.l==0 else self.problems[self.l-1].obj_nonsmooth.get_parameter())
    
def trustregion_step(l, x, val, grad, problems, params, cnt, i=0):
    """
    One TR step on level l. Decides whether to recurse to l+1 (coarser) or
    solve on level l with LBFGS2.
    Returns: s, snorm, pRed, phinew, phi, iflag, iter_count, cnt, params
    """

    # ---------- tiny helpers ----------
    def scalar(z):
        return float(z.detach()) if hasattr(z, "detach") else float(z)

    class ConstShiftModel:
        """
        Wrap a smooth model with: value += <g_corr_dual, x-x0> + c0; gradient += g_corr (primal).
        No Hessian change; constant shift doesn't affect pRed/aRed or gradients.
        """
        def __init__(self, base_model, pspace, dspace, x_base, g_corr_dual, c0):
            self.base = base_model
            self.ps   = pspace
            self.ds   = dspace
            self.x0   = x_base
            self.gd   = g_corr_dual
            self.c0   = float(c0)
            
        def update(self, x, how):
            if hasattr(self.base, "update"):
                return self.base.update(x, how)
            return None

    
        def begin_counter(self, i, cnt):
            if hasattr(self.base, "begin_counter"):
                return self.base.begin_counter(i, cnt)
            return cnt

        def end_counter(self, i, cnt):
            if hasattr(self.base, "end_counter"):
                return self.base.end_counter(i, cnt)
            return cnt

        def value(self, x, ftol):
            v0, ferr = self.base.value(x, ftol)
            s   = x - self.x0
            lin = self.ds.apply(self.gd, s)  # ⟨g_corr, s⟩
            return v0 + lin + self.c0, ferr

        def gradient(self, x, gtol):
            g0, ferr = self.base.gradient(x, gtol)
            g_corr_primal = self.gd if not hasattr(self.ds, "dual_inv") else self.ds.dual_inv(self.gd)
            return g0 + g_corr_primal, ferr

        def hessVec(self, v, x, htol):
            return self.base.hessVec(v, x, htol)

        def addCounter(self, ccnt):  # passthrough if base tracks counters
            return self.base.addCounter(ccnt) if hasattr(self.base, "addCounter") else ccnt

    # ---------- basics ----------
    L     = len(problems)
    t     = params.get('ocScale', 1.0)
    dgrad = problems[l].dvector.dual(grad)

    # fine-level prox-gradient norm
    pgrad_f = problems[l].obj_nonsmooth.prox(x - t * dgrad, t); cnt['nprox'] += 1
    gnorm   = problems[l].pvector.norm(pgrad_f - x) / t

   

    # restriction to coarser (your convention: l+1 is coarser)
    R_edge = problems[l].R
    xt     = R_edge @ x

    # coarse-side gate metric
    Rgnorm = 0.0
    if l < L - 1:
        u_coarse = xt - t * (R_edge @ dgrad)
        phi_gate = phiPrec(problems, R=R_edge, l=l+1, x_fine=x, assume_tight_rows=True)
        pgrad_c  = phi_gate.prox(u_coarse, t); cnt['nprox'] += 1
        Rgnorm   = problems[l+1].pvector.norm(pgrad_c - xt) / t

    # --- base recurse gate (relative/absolute + boundary guard) ---
    params.setdefault('RgnormScale',    0.5)
    params.setdefault('RgnormScaleTol', 0.5)
    abs_floor = params.setdefault('abs_gate_min', 5e-5)
    abs_frac  = params.setdefault('abs_gate_frac', 0.6)
    gtol_gate = params.get('gtol', 1e-6)
    abs_gate  = max(abs_floor, abs_frac * scalar(gnorm), gtol_gate)

    drop_rel  = (l < L - 1) and (scalar(Rgnorm) >= params['RgnormScale'] * scalar(gnorm))
    drop_abs  = (l < L - 1) and (scalar(Rgnorm) >= abs_gate)

    fracB_ok = True
    if 'last_snorm' in params and 'last_delta_eff' in params:
        fracB = scalar(params['last_snorm']) / max(1e-16, scalar(params['last_delta_eff']))
        if fracB > 0.9:
            fracB_ok = False

    do_recurse = (l < L - 1) and drop_rel and drop_abs and fracB_ok

    # ---------- refine recurse decision with kappa/Δ_safe (inside do_recurse) ----------
    if do_recurse:
        # exact coarse correction (NO scaling): first-order consistency
        g_f_smooth, _ = problems[l].obj_smooth.gradient(x, params.get('gradTol', 1e-8));   cnt['ngrad'] += 1
        g_c_base,  _  = problems[l+1].obj_smooth.gradient(xt, params.get('gradTol', 1e-8)); cnt['ngrad'] += 1
        gf_dual   = problems[l].dvector.dual(g_f_smooth)      # fine dual
        gc_dual   = problems[l+1].dvector.dual(g_c_base)      # coarse dual
        Rgf_dual  = R_edge @ gf_dual                          # to coarse dual (per your API usage)
        g_corr_dual = Rgf_dual - gc_dual

        # coarse base physical value at xt
        val_c0, _ = problems[l+1].obj_smooth.value(xt, 0.0); cnt['nobj1'] += 1
        phi_c0    = problems[l+1].obj_nonsmooth.value(xt);   cnt['nobj2'] += 1
        F0        = val_c0 + phi_c0

        # Δ_safe from kappa
        Delta_parent_cap = params.get('delta_effective', params['delta'])
        gcnorm  = problems[l+1].dvector.norm(g_corr_dual)
        kappa   = params.setdefault('coarse_lb_kappa', 1.0)  # keep ≥0 printed with shift
        if scalar(gcnorm) > 0:
            Delta_safe_raw = scalar(F0) / max(1e-16, kappa * scalar(gcnorm))
        else:
            Delta_safe_raw = Delta_parent_cap
        Delta_safe = min(Delta_parent_cap, Delta_safe_raw)
        Delta_safe = max(0.0, Delta_safe)

        # NEW: tiny-radius floor — skip recursion if Δ_safe is microscopic
        delta_floor_abs  = params.setdefault('delta_child_floor_abs', 1e-6)
        delta_floor_frac = params.setdefault('delta_child_floor_frac', 1e-3)
        delta_floor = max(delta_floor_abs, delta_floor_frac * float(Delta_parent_cap))
        if Delta_safe < delta_floor:
            do_recurse = False  # fall back to same-level LBFGS branch

    # ---------- recurse or stay ----------
    if do_recurse:
        problemsL = copy.deepcopy(problems)
        

        # constant shift so printed model ≥ 0 on ||s||≤Δ_safe; no gradient change
        c0_shift = Delta_safe * scalar(gcnorm)

        # child params (strict cap + tighter gtol)
        params_child              = copy.deepcopy(params)
        params_child['gtol']      = params.get('gtol', 1e-6) * params.setdefault('coarse_gtol_scale', 0.1)
        params_child['delta']     = min(params_child['delta'], Delta_safe)
        params_child['deltamax']  = Delta_safe      # cannot grow later
        params_child.pop('delta_effective', None)

        # child problem with wrapped smooth model
        base_model = modelTR(problems, params.get("useSecant", False), 'recursive',
                             l=l+1, R=R_edge, dgrad=(R_edge @ dgrad), x=copy.deepcopy(xt))

        p = Problem(problems[l+1].obj_nonsmooth.var, problems[l+1].R)
        p.obj_smooth    = ConstShiftModel(base_model, problems[l+1].pvector, problems[l+1].dvector,
                                          x_base=xt, g_corr_dual=g_corr_dual, c0=c0_shift)
        p.obj_nonsmooth = phiPrec(problems, R=R_edge, l=l+1, x_fine=x, assume_tight_rows=True)
        p.pvector       = problems[l+1].pvector
        p.dvector       = problems[l+1].dvector
        problemsL[l+1]  = p

        # evaluate wrapped model & phi at xt
        val_wrapped, _ = p.obj_smooth.value(xt, 0.0); cnt['nobj1'] += 1
        phi            = p.obj_nonsmooth.value(xt);   cnt['nobj2'] += 1

        # solve child TR
        xnew, cnt_coarse = trustregion(l+1, xt, Delta_safe, problemsL, params_child)

        # lift step back to level l
        s     = R_edge.T @ (xnew - xt)
        snorm = problems[l].pvector.norm(s)

        # evaluate trial (wrapped) for pRed
        valnew_wrapped, _ = p.obj_smooth.value(xnew, 0.0); cnt['nobj1'] += 1
        phinew            = p.obj_nonsmooth.value(xnew);   cnt['nobj2'] += 1

        pRed       = (val_wrapped + phi) - (valnew_wrapped + phinew)
        print(pRed)
        iflag      = cnt_coarse.get('iflag', 3)
        iter_count = cnt_coarse.get('iter', 0)

        # collect counters
        cnt = p.obj_smooth.addCounter(cnt)
        cnt = p.obj_nonsmooth.addCounter(cnt)

    else:
        # -------- stay on level l: LBFGS2 subsolver --------
        R_eye = Reye(x)
        problemTR               = Problem(problems[l].var, R_eye)
        problemTR.obj_smooth    = modelTR(problems, params.get("useSecant", False), 'spg',
                                          l=l, R=R_eye, dgrad=dgrad, x=x)
        problemTR.obj_nonsmooth = PhiCounter(problems[l].obj_nonsmooth)
        phi = problemTR.obj_nonsmooth.value(x)

        problemTR.pvector = problems[l].pvector
        problemTR.dvector = problems[l].dvector

        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step_SPG2(
            x, val, grad,dgrad, phi, problemTR, params, cnt
        )
        cnt = problemTR.obj_smooth.addCounter(cnt)
        cnt = problemTR.obj_nonsmooth.addCounter(cnt)

    # safety (Torch-safe scalar)
    if scalar(pRed) < 0 and abs(scalar(pRed)) > 1e-5:
        import pdb; pdb.set_trace()
        #pRed = 1e-16

    # remember boundary info for next gate
    params['last_snorm']     = snorm
    params['last_delta_eff'] = params.get('delta_effective', params['delta'])

    return s, snorm, pRed, phinew, phi, iflag, iter_count, cnt, params



# --- debug helpers ---
def _near_boundary(snorm, delta_eff, frac=0.8):
    return snorm >= frac * max(1e-16, delta_eff)

def _good_quality(aRed, pRed, eta2):
    return aRed > eta2 * pRed

def _not_too_close_to_opt(gnorm, gtol, factor=10.0):
    return gnorm > factor * gtol


def trustregion(l, x0, Deltai, problems, params): #inpute Deltai
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
    params.setdefault('prev_near_boundary', False)

    # right after reading params and before the first print:
    if l == 0:
        params['delta'] = min(params['delta'], params['deltamax'])   # Δ_0,0 = min(Δ_0^s, +∞)
    else:
        # Deltai is Δ_{i+1} in your signature
        params['delta'] = min(params['delta'], params['deltamax'], Deltai)  # Δ_{i,0} = min(Δ_i^s, Δ_{i+1})


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
        'timehist': [],
        'valerr': [],
        'valtol': [],
        'graderr': [],
        'gradtol': []
    }



    # Compute initial function information
    if hasattr(problems[l].obj_smooth, 'begin_counter'):
        cnt = problems[l].obj_smooth.begin_counter(0, cnt)

    if params['initProx']:
        x = problems[l].obj_nonsmooth.prox(x0, 1)
        cnt['nprox'] += 1
    else:
        x = copy.deepcopy(x0)
    if l == 0:
      problems[l].obj_smooth.update(x, 'init')
    ftol = 1e-12
    if params['useInexactObj']:
        ftol = params['maxValTol']
    val, _      = problems[l].obj_smooth.value(x, ftol)

    cnt['nobj1'] += 1
    grad, _, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
    phi                 = problems[l].obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    if hasattr(problems[l].obj_smooth, 'end_counter'):
        cnt = problems[l].obj_smooth.end_counter(0, cnt)

    

    # Output header
    if l == 0:
      print(f"\nRecursive Nonsmooth Trust-Region Method using {params.get('spsolver', 'SPG2')} Subproblem Solver")
      print("level   iter          value           gnorm             del           snorm       nobjs      ngrad      nhess      nobjn      nprox    iterSP    flagSP")
      print(f"{0:4d}   {0:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}             ---      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}       ---      ---")

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
    cnt['timehist'].append(np.nan)

    # Set optimality tolerance
    gtol = params['gtol']
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    # Check stopping criterion
    # if gnorm <= gtol:
    #     print('tr', gnorm, gtol, l)
    #     cnt['iflag'] = 0
    #     return x, cnt
    
        

    
    # Iterate
    for i in range(1, params['maxit'] + 1):
        if hasattr(problems[l].obj_smooth, 'begin_counter'):
            cnt = problems[l].obj_smooth.begin_counter(i, cnt)
            
        if l != 0:
            dist = problems[l].pvector.norm(x - x0)
            cap  = Deltai - dist
            if -1e-12 < cap < 0:   # tiny negative → 0
                cap = 0.0
            delta_eff = max(0.0, min(params['delta'], cap))
        else:
            delta_eff = min(params['delta'], params['deltamax'])

        params['delta_effective'] = delta_eff
        #print(f"del={params['delta']:8.6e}", f"del_eff={params['delta_effective']:8.6e}")

        # Solve trust-region subproblem
        rel = params['rtol'] * gnorm ** params['spexp']
        tolsp = min(params['atol'], rel)
        tolsp = max(params.get('tolsp_floor', 1e-8), tolsp)   # <- add a floor
        params['tolsp'] = tolsp
        # -- DEBUG: pre-subproblem snapshot --
        #print(f"[TR l={l} i={i}] gnorm={gnorm:.3e} gtol={gtol:.3e} "
        #      f"δ={params['delta']:.3e} δ_eff={params['delta_effective']:.3e} tolsp={params['tolsp']:.3e}")
        s, snorm, pRed, phinew, phi, iflag, iter_count, cnt, params = trustregion_step(l, x, val, grad, problems, params, cnt, i=i-1)

        # Update function information
        xnew             = x + s
        problems[l].obj_smooth.update(xnew, 'trial')
        valnew, val, cnt = compute_value(xnew, x, val, problems[l].obj_smooth, pRed, params, cnt)

        # Accept/reject step and update trust-region radius
        aRed = (val + phi) - (valnew + phinew)
        rho   = aRed / max(1e-16, pRed)
        fracB = snorm / max(1e-16, params['delta_effective'])
        
        
        coolk = params.setdefault('grow_cooldown_k', 0)          # counter
        if coolk > 0:
            params['grow_cooldown_k'] = coolk - 1

        

        if aRed < params['eta1'] * pRed:
            params['delta'] = params['gamma1'] * params['delta']
            problems[l].obj_smooth.update(x, 'reject')
            if params['useInexactGrad']:
                grad, _, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
        else:
            x   = xnew
            val = valnew
            phi = phinew
            problems[l].obj_smooth.update(x, 'accept')
            # grad0 = grad
            grad, _, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
            
            
            if (rho>params['eta2']) and (fracB>=0.9) and (gnorm>10*gtol):
               params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])
 


            
        if l != 0:
            dist = problems[l].pvector.norm(x - x0)
            residual = Deltai - dist
            if -1e-12 < residual < 0: residual = 0.0
            
          #params['delta'] = min(params['delta'], Deltai - problems[l].pvector.norm(x - x0))
         
        params['delta'] = max(0.0,min(params['delta'],params['deltamax']))

        # Output iteration history
        if i % params['outFreq'] == 0:
            print(f"{l:4d}   {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")

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
        cnt['timehist'].append(time.time() - start_time)

        if hasattr(problems[l].obj_smooth, 'end_counter'):
            cnt = problems[l].obj_smooth.end_counter(i, cnt)

        # Check stopping criterion
        if gnorm <= gtol or snorm <= stol or i >= params['maxit'] or (problems[l].pvector.norm(x - x0) > (1 - .001)*Deltai and l!=0):
            if i % params['outFreq'] != 0:
                print(f"  {l:4d}   {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")
            if gnorm <= gtol:
                flag = 0
            elif i >= params['maxit']:
                flag = 1
            elif problems[l].pvector.norm(x - x0) > (1 - .001)*Deltai:
                flag = 2
            else:
                flag = 3
            break

    cnt['iter'] = i
    cnt['timetotal'] = time.time() - start_time

    print("Optimization terminated because ", end="")
    if flag == 0:
        print("optimality tolerance was met")
    elif flag == 1:
        print("maximum number of iterations was met")
    elif flag == 2:
        print("finer trust-region radius met")
    else:
        print("step tolerance was met")
    if l==0:
      print(f"Total time: {cnt['timetotal']:8.6e} seconds")
    cnt['iflag'] = flag
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


# -------- row-orthonormal "averaging" blocks --------

def _group_reduce(m: int, n: int, mode: str, dtype, device):
    """
    Make A in R^{n x m} that groups m->n with disjoint blocks.
    mode='orthonormal' -> entries 1/sqrt(g) in each block (rows orthonormal)
    mode='average'     -> entries 1/g in each block (rows sum to 1)
    """
    g = m // n
    A = torch.zeros(n, m, dtype=dtype, device=device)
    val = (1.0 / (g ** 0.5)) if mode == "orthonormal" else (1.0 / g)
    for r in range(n):
        A[r, r*g:(r+1)*g] = val
    return A

def _linear_interp(m: int, n: int, dtype, device):
    """Simple 1D linear interpolation rows that sum to 1 (not orthonormal)."""
    idx = torch.linspace(0, m - 1, steps=n, dtype=dtype, device=device)
    A = torch.zeros(n, m, dtype=dtype, device=device)
    for r, x in enumerate(idx):
        i0 = int(torch.floor(x).item())
        i1 = min(i0 + 1, m - 1)
        t = float(x - i0)
        A[r, i0] = 1.0 - t
        A[r, i1] += t
    A = A / (A.sum(dim=1, keepdim=True) + 1e-12)
    return A

def _orthonormalize_rows(A: torch.Tensor) -> torch.Tensor:
    """
    Return Ã with orthonormal rows spanning the same row space as A.
    Uses QR on A^T:  A^T = Q R  -> rows(Ã) = rows(Q^T).
    """
    Q, _ = torch.linalg.qr(A.t(), mode='reduced')  # Q: (m x n)
    return Q.t()  # (n x m), rows are orthonormal

def _reduce_matrix(m_from: int, m_to: int, *, mode: str, dtype=torch.float64, device=None):
    """
    Build A in R^{m_to x m_from} mapping fine->coarse hidden units.
    - If equal: identity (orthonormal).
    - If divisible: disjoint grouping (orthonormal or average).
    - Else: linear interp then (if mode='orthonormal') row-orthonormalize.
    """
    device = device or torch.device("cpu")
    if m_from == m_to:
        return torch.eye(m_to, m_from, dtype=dtype, device=device)

    if m_from % m_to == 0:
        return _group_reduce(m_from, m_to, mode=mode, dtype=dtype, device=device)

    # Not divisible: start with interp
    A = _linear_interp(m_from, m_to, dtype=dtype, device=device)
    if mode == "orthonormal":
        A = _orthonormalize_rows(A)
    return A

# -------- layer index inference --------

def _infer_layer_index_from_name(name: str, n_linear_layers: int) -> int:
    m = re.search(r'\.(\d+)\.(weight|bias)$', name)
    if m:
        idx = int(m.group(1))
        return min(max(idx, 0), n_linear_layers - 1)
    m = re.search(r'(\d+)', name)
    if m:
        idx = int(m.group(1))
        if idx > 0: idx -= 1  # fc1/layer1 -> 0-based
        return min(max(idx, 0), n_linear_layers - 1)
    return 0

# -------- build parameter-wise restriction --------

def _build_R_ops(
    cur_sizes,          # fine widths:  [in_f, h1_f, ..., out_f]
    next_sizes,         # coarse widths:[in_c, h1_c, ..., out_c]
    state_dict: OrderedDict,
    *,
    mode: str = "orthonormal",      # "orthonormal" (default) or "average"
    dtype=torch.float64,
    device=None,
    mapping_mode: str = "by_name",  # or "by_order"
):
    """
    Returns:
      R_ops[name]       = 2D op mapping vec(param_fine[name]) -> vec(param_coarse[name])
      next_shapes[name] = target shape on coarse net
    Ensures every key in state_dict appears in R_ops (identity fallback).
    """
    device = device or torch.device("cpu")
    cur, nxt = list(cur_sizes), list(next_sizes)
    L = len(cur) - 1

    R_ops = {}
    next_shapes = {}
    w_idx = b_idx = 0

    for name, tensor in state_dict.items():
        is_weight = name.endswith("weight")
        is_bias   = name.endswith("bias")

        ell = (
            min(w_idx, L-1) if (mapping_mode=="by_order" and is_weight)
            else min(b_idx, L-1) if (mapping_mode=="by_order" and is_bias)
            else _infer_layer_index_from_name(name, L)
        )

        if is_weight and tensor.ndim == 2 and 0 <= ell < L:
            o_f, i_f = tensor.shape
            if o_f == cur[ell+1] and i_f == cur[ell]:
                o_c, i_c = nxt[ell+1], nxt[ell]
                A_out = _reduce_matrix(o_f, o_c, mode=mode, dtype=dtype, device=device)  # (o_c, o_f)
                A_in  = _reduce_matrix(i_f, i_c, mode=mode, dtype=dtype, device=device)  # (i_c, i_f)
                # vec(Wc) = (A_out ⊗ A_in) vec(Wf)
                Rk = torch.kron(A_out, A_in)
                R_ops[name] = Rk.to(dtype=dtype, device=device)
                next_shapes[name] = (o_c, i_c)
                if mapping_mode == "by_order": w_idx += 1
                continue  # handled

        if is_bias and 0 <= ell < L:
            o_f = tensor.numel()
            if o_f == cur[ell+1]:
                o_c = nxt[ell+1]
                A_out = _reduce_matrix(o_f, o_c, mode=mode, dtype=dtype, device=device)  # (o_c, o_f)
                R_ops[name] = A_out.to(dtype=dtype, device=device)
                next_shapes[name] = (o_c,)
                if mapping_mode == "by_order": b_idx += 1
                continue  # handled

        # Fallback for anything unexpected: identity pass-through
        n = tensor.numel()
        R_ops[name] = torch.eye(n, n, dtype=dtype, device=device)
        next_shapes[name] = tuple(tensor.shape)

    # Guarantee every key exists (if any were skipped above)
    for name, t in state_dict.items():
        if name not in R_ops:
            n = t.numel()
            R_ops[name] = torch.eye(n, n, dtype=dtype, device=device)
            next_shapes[name] = tuple(t.shape)

    return R_ops, next_shapes

# -------- user-facing: build TorchVect operator R --------

def restriction_R_from_dims(
    next_sizes,                 # coarse widths (e.g., [2, 50, 50, 1])
    cur_sizes,                  # fine widths   (e.g., [2,100,100,1])
    x_fine,                     # TorchVect fine params (.td = OrderedDict name->tensor)
    *,
    mode: str = "orthonormal",  # "orthonormal" (row-orthonormal) or "average"
    mapping_mode: str = "by_name",
    dtype=torch.float64,
):
    fine_sd = x_fine.td
    device = next(iter(fine_sd.values())).device if len(fine_sd) else torch.device("cpu")
    R_ops, next_shapes = _build_R_ops(
        cur_sizes, next_sizes, fine_sd, mode=mode, dtype=dtype, device=device, mapping_mode=mapping_mode
    )

    # Pack into TorchVect operator with shapes_map; also add a .shape your step checks
    R = TorchVect(OrderedDict(R_ops), isRop=True, shapes_map=next_shapes)

    # Synthetic global shape for legacy checks (sum of block sizes)
    total_rows = sum(W.shape[0] for W in R_ops.values())
    total_cols = sum(W.shape[1] for W in R_ops.values())
    R.shape = [total_rows, total_cols]      # <- keeps your existing `x.shape[0] != R0.shape[0]` check happy

    return R

class RWrap:
    """
    Per-parameter linear operator wrapper.
    td[name]: matrix of shape (out_numel, in_numel)
    shapes_map is the OUTPUT reshape map used by TorchVect.__matmul__.
    We also keep an input map and swap them on transpose.
    """
    def __init__(self, td, out_shapes_map, in_shapes_map):
        self.td = dict(td)
        self._out_shapes_map = dict(out_shapes_map)  # shape to reshape R @ x into
        self._in_shapes_map  = dict(in_shapes_map)   # shape of x
        self.shapes_map = self._out_shapes_map       # TorchVect reads this
        self.isRop = True
        
    @property
    def inner(self):
        return TorchVect(OrderedDict(self.td), isRop=True, shapes_map=self.shapes_map)

    def __matmul__(self,other):
        #RWrap @ TorchVect -> apply operator
        #RWrap @ RWrap -> compose operators
        if isinstance(other, RWrap):
            return self.inner @ other.inner
        elif isinstance(other,TorchVect):
            return self.inner @ other
        else:
            return NotImplemented
        
    def __rmatmul__(self, other):
        """Support TorchVect @ RWrap if your code ever does that."""
        if isinstance(other, TorchVect):
            # (x @ R) is unusual; if you need it, implement as (R.T @ x) with care.
            return NotImplemented
        return NotImplemented


    @property
    def T(self):
        Rt = RWrap.__new__(RWrap)
        Rt.td = {k: v.T.contiguous() for k, v in self.td.items()}
        # swap in/out maps on transpose
        Rt._out_shapes_map = self._in_shapes_map
        Rt._in_shapes_map  = self._out_shapes_map
        Rt.shapes_map = Rt._out_shapes_map
        Rt.isRop = True
        return Rt
    
    @property
    def shape(self):
        rows = sum(W.shape[0] for W in self.td.values())
        cols = sum(W.shape[1] for W in self.td.values())
        return [rows,cols]


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


from typing import Sequence, Callable, Optional
class FullyConnectedNN(nn.Module):
    def __init__(self, sizes:Sequence[int],activation:Optional[nn.Module]=None,last_bias: bool=False,):
        super().__init__()
        assert len(sizes) >= 2, "sizes must have at least [in, out]"
        assert sizes[0] == 2, "First layer input dim must be 2 for 2D inputs"
        act = activation if activation is not None else nn.Tanh()

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=True))
            # use a fresh activation module each time
            layers.append(type(act)())

        # final linear layer (optionally without bias)
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=last_bias))
        self.net = nn.Sequential(*layers)

        # good default init for PINNs
        self._init_weights()
        # in FullyConnectedNN.__init__
        self.cB = torch.nn.Parameter(torch.tensor([2.7726], dtype=torch.float64))  # shape (1,)
  

        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot is a solid default (tanh-friendly)
                nn.init.xavier_uniform_(m.weight)
                #nn.init.kaiming_uniform_(m.weight,a=0.1,nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
        

    def forward(self, x:torch.Tensor):
        #print("inputs_shape:",x.shape)
        x = 2.0*x-1.0
        
        return self.net(x)
    
    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
        
    def gradient(self,x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        u = self.forward(x)  # (N,1)
        grad = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,  # keep graph for Laplacian
            retain_graph=True
        )[0]  # (N,2)
        
        return grad
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        g = self.gradient(x)  # (N,2), graph kept
        lap = 0.0
        for d in range(g.shape[1]):  # 2D
            g_d = g[:, d:d+1]  # (N,1)
            lap_d = torch.autograd.grad(
                g_d, x,
                grad_outputs=torch.ones_like(g_d),
                create_graph=True,
                retain_graph=True
            )[0][:, d:d+1]  # (N,1) pick same component
            lap = lap + lap_d
            
        return lap  # (N,1)
        
def _stateless_u(var, params, x_xy):
    xg = x_xy.detach().clone().requires_grad_(True)
    u  = torch.func.functional_call(var['NN'], params, (xg,))  # (N,1)
    return u, xg
    


def _u_with_hard_bc(var, params, x_xy):
    """
    Stateless forward with hard Dirichlet BC:
      u(x,y) = B(x,y) * N_theta(x,y),   B = x(1-x)*y(1-y).
    IMPORTANT: do NOT detach params, or you break backprop.
    """
    # Accept TorchVect or OrderedDict
    if hasattr(params, "td"):
        params = params.td

    # Put tensors on the model's device/dtype WITHOUT detaching
    dev   = next(var['NN'].parameters()).device
    dtype = next(var['NN'].parameters()).dtype
    params = OrderedDict((k, v.to(device=dev, dtype=dtype)) for k, v in params.items())

    # Inputs with grad (for PDE terms)
    xg = x_xy.detach().clone().requires_grad_(True)

    # Stateless forward that keeps grads to params
    n_raw = torch.func.functional_call(var['NN'], params, (xg,))

    # Hard BC mask
    B = (xg[:, :1] * (1 - xg[:, :1])) * (xg[:, 1:2] * (1 - xg[:, 1:2]))
    #u = B * n_raw
    if 'cB' in params:
        scale = torch.exp(params['cB'])      # works for scalar () or shape (1,)
    else:
        scale = 1.0

    u = scale * B * n_raw
    #u = torch.exp(var['NN'].cB) * B * n_raw

    return u, xg
def hard_bc_forward_and_derivs(var, params, x_xy):
    # Stateless net output N
    xg = x_xy.detach().clone().requires_grad_(True)
    N  = torch.func.functional_call(var['NN'], params, (xg,))  # (N,1)

    x = xg[:, :1]; y = xg[:, 1:2]
    B = x*(1-x)*y*(1-y)
    dBx = (1 - 2*x) * y*(1-y)
    dBy = (1 - 2*y) * x*(1-x)
    lapB = -2*y*(1-y) - 2*x*(1-x)

    # ∇N and ΔN by autodiff
    gN = torch.autograd.grad(N, xg, grad_outputs=torch.ones_like(N), create_graph=True)[0]  # (N,2)
    lapN = 0.0
    for d in range(2):
        gNd = gN[:, d:d+1]
        lapNd = torch.autograd.grad(gNd, xg, grad_outputs=torch.ones_like(gNd), create_graph=True)[0][:, d:d+1]
        lapN = lapN + lapNd
    if 'cB' in params:
        scale = torch.exp(params['cB'])
    elif hasattr(var['NN'],'cB'):
        scale = torch.exp(var['NN'].cB)
    else:
        scale = 1.0

    # u, ∇u, Δu via product rule
    u = scale * B * N
    gu_x = scale * (dBx * N + B * gN[:, :1])
    gu_y = scale * (dBy * N + B * gN[:, 1:2])
    gu = torch.cat([gu_x, gu_y], dim=1)
    lapu = scale * (lapB * N + 2*(dBx * gN[:, :1] + dBy * gN[:, 1:2]) + B * lapN)
    return u, gu, lapu, xg

def apply_best_scale_live(nnset):
    diag   = nnset.diagnose_scale()          # prints s_opt, etc.
    s_opt  = diag["s_opt"]

    with torch.no_grad():
        # 1) If negative, flip last layer to absorb the sign
        if s_opt < 0:
            last = nnset.NN.net[-1]          # final nn.Linear in your Sequential
            last.weight.mul_(-1)
            if last.bias is not None:
                last.bias.mul_(-1)
            s_opt = -s_opt                   # now positive

        # 2) Set cB = log(s_opt)
        if hasattr(nnset.NN, "cB"):
            nnset.NN.cB.copy_(torch.tensor(math.log(max(s_opt, 1e-16)),
                                            dtype=nnset.NN.cB.dtype,
                                            device=nnset.NN.cB.device))
    return diag     


@torch.no_grad()
def param_hash(nn):
    return float(sum(p.abs().sum() for p in nn.parameters()).cpu())        

class NNObjective:
    def __init__(self, var):
        self.var = var
        self.loss_history = []
        
    
    def update(self, x, type):
        return None
    
    
        
    def check_parameters_require_grad(self, params, context=""):
        """Check if all parameters require gradients"""
        if isinstance(params, TorchVect):
            params_dict = params.td
        else:
            params_dict = params
            
        issues = []
        for name, param in params_dict.items():
            if not param.requires_grad:
                issues.append(name)
                
        if issues:
            print(f"GRADIENT ISSUE {context}: Parameters without grad: {issues}")
        return len(issues) == 0
    
    @staticmethod
    def _trainable_params_like(model, param_map):
        """Clone to leaf tensors, correct dtype/device, requires_grad=True, preserve order."""
        if not isinstance(param_map, collections.OrderedDict):
            param_map = collections.OrderedDict(param_map)
        # align to model's state_dict ordering exactly
        sd = model.state_dict()
        assert list(param_map.keys()) == list(sd.keys()), \
            f"Param keys mismatch.\nmodel: {list(sd.keys())}\nparam_map: {list(param_map.keys())}"
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        out = collections.OrderedDict()
        for k in sd.keys():  # preserve exact order
            #t = param_map[k].detach().clone().to(dtype=dtype, device=device).requires_grad_(True)
            t = param_map[k].to(dtype=dtype, device=device)
            if not t.requires_grad:
                t.requires_grad_(True)
            out[k] = t
        return out
    def value_torch(self, params, ftol=None):
        """
        Strong-form PINN with hard Dirichlet BCs:
          u(x) = B(x) * N(x),   B(x,y)=x(1-x)y(1-y)
        PDE: -div(k grad u) = b   (k may vary; gradk provided)
        """
        # params can be TorchVect or OrderedDict
        if isinstance(params, TorchVect):
            params = params.td

        # interior inputs (dof coordinates)
        #x_in = self.var['inputs_xy']  # (N,2) torch
        I = self.var.get('interior_idx',None)
        x_all = self.var['inputs_xy']
        if I is not None:
            x_in = x_all[I]
        else:
            x_in = x_all
        u, gu, lap, xg = hard_bc_forward_and_derivs(self.var, params, x_in)


        # coefficients/RHS aligned to dof coords
        k_all     = self.var['k']      # (N,1)
        gradk_all = self.var['gradk']  # (N,2)
        b_all     = self.var['b']      # (N,1)
        if I is not None:
            k = k_all[I]
            gradk = gradk_all[I]
            b = b_all[I]
        else:
            k,gradk,b = k_all,gradk_all,b_all

        # residual:  - (∇k · ∇u) - k Δu - b
        pde_residual = -(gradk * gu).sum(dim=1, keepdim=True) - k * lap - b
        pde_loss = 0.5 * self.var.get('alpha', 1.0) * (pde_residual.pow(2)).mean()
        
        lam = float(self.var.get('lambda_reg', 0.0))
        reg = torch.tensor(0.0, dtype=pde_loss.dtype, device=pde_loss.device)
        if lam > 0.0:
            excl = tuple(self.var.get('l2_exclude', ()))  # patterns to skip
            reg_sum = []
            for name, p in params.items():                # IMPORTANT: use the passed-in params
                if any(tag in name for tag in excl):      # skip biases, cB, etc.
                    continue
                reg_sum.append((p**2).sum())
            if reg_sum:
                reg = 0.5 * lam * torch.stack(reg_sum).sum()


        # OPTIONAL: soft BC penalty (should be ~0 with hard BC; keep as guardrail)
        loss_bc = torch.tensor(0.0, dtype=pde_loss.dtype, device=pde_loss.device)
        if ('bdry_xy' in self.var and 'bdry_u' in self.var and
            self.var['bdry_xy'] is not None and self.var['bdry_u'] is not None):
            xb = self.var['bdry_xy']          # (Nb,2)
            ub = self.var['bdry_u']           # (Nb,1)
            ub_pred, _ = _u_with_hard_bc(self.var, params, xb)  # uses same hard-BC ansatz
            loss_bc = 0.5 * (ub_pred - ub).pow(2).mean()
            

        beta_bc = 0.0  # with hard BC, you can set 0; or small positive as a safety belt
        loss = pde_loss + beta_bc * loss_bc +reg
        ###consider L2 norm of p
        
        

        self.loss_history.append(loss.item())
        return loss

                
  
        
    
        
    def value_torch_soft(self, params, ftol=None):
        """
        Strong-form loss with SOFT Dirichlet BCs only.
        PDE:  - (∇k · ∇u) - k Δu = b    on Ω
        BC :  u = u_b                 on ∂Ω   (penalized)
        """
        # Accept TorchVect or OrderedDict
        if isinstance(params, TorchVect):
            params = params.td

        # ===== Interior residual =====
        x = self.var['inputs_xy']              # (Ni,2) dof coords (Ω)
        if not x.requires_grad:
            x = x.requires_grad_(True)         # needed for grads/Laplacian

        # Stateless forward (plain network, NO hard BC)
        u = torch.func.functional_call(self.var['NN'], params, (x,))
        #u, xg = _stateless_u(self.var, params, x)

        # ∇u
        gu = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        # Δu
        lap = torch.zeros_like(u)
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap_d = torch.autograd.grad(
                 g_d, x, grad_outputs=torch.ones_like(g_d),
                 create_graph=True, retain_graph=True
            )[0][:, d:d+1]
            lap += lap_d

        # coefficients / RHS (aligned to x)
        k     = self.var['k']      # (Ni,1)
        gradk = self.var['gradk']  # (Ni,2)
        b     = self.var['b']      # (Ni,1)  <-- pointwise f(x), not assembled RHS
        def _reduce_mean(x):  # helper
            return x.mean()
        

        # PDE residual and loss
        pde_residual = -(gradk * gu).sum(dim=1, keepdim=True) - k * lap - b
        #pde_loss = 0.5 * self.var.get('alpha', 1.0) * (pde_residual.pow(2).mean())
        pde_loss = 0.5*self.var['alpha']*_reduce_mean((pde_residual**2))

        # ===== Boundary penalty (soft BC) =====
        loss_bc = torch.tensor(0.0, dtype=pde_loss.dtype, device=pde_loss.device)
        if ('bdry_xy' in self.var and self.var['bdry_xy'] is not None and
            'bdry_u'  in self.var and self.var['bdry_u']  is not None):
            xb = self.var['bdry_xy']           # (Nb,2) boundary coords
            ub = self.var['bdry_u']            # (Nb,1) target boundary values (e.g., zeros)

            # For BC penalty we DON'T need grads wrt xb; just params
            #ub_pred = torch.func.functional_call(self.var['NN'], params, (xb,))
            #loss_bc = 0.5 * (ub_pred - ub).pow(2).mean()
            ub_pred, _ = _stateless_u(self.var, params, xb)  # (Nb,1)
            loss_bc = 0.5 * _reduce_mean((ub_pred - ub)**2)

        # Big boundary weight (tune as needed)
        beta_bc = self.var.get('beta_bc', 1.0)   # e.g., 50–500 good starting range

        loss = pde_loss + beta_bc * loss_bc
        self.loss_history.append(float(loss.item()))
        return loss



    def value(self, x, ftol=None):
        """
        x: TorchVect or OrderedDict
        returns: (val, ferr)
        """
        val = self.value_torch(x, ftol)
        return val, 0.0

    def torch_gradient(self, x, gtol=None):
    # FORCE all parameters to require grad regardless of input
        if isinstance(x, TorchVect):
            params_dict = x.td
        else:
            params_dict = x
        
        # Create fresh parameters that definitely require grad
        fresh_params = collections.OrderedDict()
        for name, param in params_dict.items():
            fresh_params[name] = param.detach().clone().requires_grad_(True)
    
        # Use the fresh parameters for computation
        loss = self.value_torch(fresh_params, gtol)
    
        names, tensors = zip(*fresh_params.items())
        grads = torch.autograd.grad(loss, tensors, retain_graph=False, create_graph=False)
    
        gdict = collections.OrderedDict(zip(names, grads))
        return TorchVect(gdict) if isinstance(x, TorchVect) else gdict



    def gradient(self, x, gtol=None):
        """Wrapper for gradient computation"""
        g = self.torch_gradient(x, gtol)
        
        # Compute total gradient norm
        if isinstance(g, TorchVect):
            grad_tensors = list(g.td.values())
        else:
            grad_tensors = list(g.values())
        
        total_norm = torch.norm(torch.cat([g.flatten() for g in grad_tensors])).item()
        #print(f"Total gradient norm: {total_norm}")
        
        return g, 0.0

    def hessVec(self, v, x, htol=None):
        raw   = x.td if isinstance(x, TorchVect) else collections.OrderedDict(x)
        v_raw = v.td if isinstance(v, TorchVect) else collections.OrderedDict(v)
        params = self._trainable_params_like(self.var['NN'], raw)  # only here

        names, theta = zip(*params.items())
        vtheta = [v_raw[k] for k in names]

        loss = self.value_torch(params, htol)

        g_list = torch.autograd.grad(loss, theta, create_graph=True)
        flat_g = torch.cat([g.reshape(-1) for g in g_list])
        flat_v = torch.cat([vv.reshape(-1) for vv in vtheta])
        gv = torch.dot(flat_g, flat_v)
        Hv_list = torch.autograd.grad(gv, theta, retain_graph=False)

        Hv = collections.OrderedDict((k, hv) for k, hv in zip(names, Hv_list))
        return (TorchVect(Hv) if isinstance(x, TorchVect) else Hv), 0.0
    def hessVec_JJ(self, v, x, htol=None, use_gn=True):
    

        # unwrap inputs
        raw   = x.td if isinstance(x, TorchVect) else collections.OrderedDict(x)
        v_raw = v.td if isinstance(v, TorchVect) else collections.OrderedDict(v)

        # clone into leaf tensors on correct device/dtype and requires_grad=True
        params = self._trainable_params_like(self.var['NN'], raw)
        names, theta = zip(*params.items())
        vtheta = [v_raw[k] for k in names]

        if not use_gn:
            # exact Hessian (as before)
            loss = self.value_torch(params, htol)
            g_list = torch.autograd.grad(loss, theta, create_graph=True)
            flat_g = torch.cat([g.reshape(-1) for g in g_list])
            flat_v = torch.cat([vv.reshape(-1) for vv in vtheta])
            gv = torch.dot(flat_g, flat_v)
            Hv_list = torch.autograd.grad(gv, theta, retain_graph=False)
            Hv = collections.OrderedDict((k, hv) for k, hv in zip(names, Hv_list))
            return (TorchVect(Hv) if isinstance(x, TorchVect) else Hv), 0.0
        
        x_all = self.var['inputs_xy']
        I_all = self.var.get('interior_idx',None)
        if I_all is None:
            I_all = torch.arange(x_all.shape[0],dtype=torch.long)
        if not torch.is_tensor(I_all):
            I_all= torch.as_tensor(I_all,dtype=torch.long)
        B = int(self.var.get('batch_hvp',0)) or int(I_all.numel())
        B = min(B, int(I_all.numel()))
        idx = torch.randint(0, int(I_all.numel()), (B,), dtype=torch.long)
        sel = I_all[idx]
        x_batch  = x_all[sel]
        k_batch  = self.var['k'][sel]
        gk_batch = self.var['gradk'][sel]
        b_batch  = self.var['b'][sel]
        

        # --------- Gauss–Newton path: Hv ≈ α/N Jᵀ(J v) ---------
        #x_in = self.var['inputs_xy']

        def r_of_theta(*theta_list):
            local = collections.OrderedDict((kname, t) for kname, t in zip(names, theta_list))
            u, xg = _u_with_hard_bc(self.var, local, x_batch)
            gu = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
            lap = 0.0
            for d in range(2):
                g_d = gu[:, d:d+1]
                lap += torch.autograd.grad(g_d, xg, torch.ones_like(g_d), create_graph=True)[0][:, d:d+1]
            #k     = self.var['k']
            #gradk = self.var['gradk']
            #b     = self.var['b']
            #r = -(gradk * gu).sum(dim=1, keepdim=True) - k * lap - b
            r = -(gk_batch * gu).sum(dim=1, keepdim=True) - k_batch * lap - b_batch
            return r

        # Jv with a graph on r(theta):
        r_val, Jv = torch.autograd.functional.jvp(lambda *t: r_of_theta(*t),tuple(theta),tuple(vtheta),create_graph=True,  strict=True)

        # Hv = α * ∂(r)ᵀ/∂θ · (Jv)
        #gJv = torch.autograd.grad(r_val,theta,grad_outputs=Jv,retain_graph=False,allow_unused=True)

        alpha = float(self.var.get('alpha', 1.0))
        
        B_eff = int(r_val.numel())
        scale = alpha / max(B_eff, 1)
        #N = int(r_val.numel())
        #scale = alpha/max(N,1)
        gJv = torch.autograd.grad(r_val, theta, grad_outputs=Jv, retain_graph=False, allow_unused=True)
        Hv_list = [torch.zeros_like(p) if g is None else scale * g for p, g in zip(theta, gJv)]
        Hv = collections.OrderedDict((k, hv) for k, hv in zip(names, Hv_list))
        return (TorchVect(Hv) if isinstance(x, TorchVect) else Hv), 0.0


            


class NNSetup:
    def __init__(self, NN_dim, n, alpha, beta, n_samples=1):
        self.n      = n
        self.NN_dim = NN_dim
        self.nsamps = n_samples
        self.beta   = beta
        self.alpha  = alpha
        self.R      = []

        # ---- (A) Build a uniform grid in [0,1]^2
        xs = np.linspace(0.0, 1.0, n+1)
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        coords_np = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N,2)
        self.inputs_xy = torch.as_tensor(coords_np, dtype=torch.float64)  # (N,2)

        # ---- (B) Boundary detection (grid points on ∂Ω)
        x = self.inputs_xy[:, 0]
        y = self.inputs_xy[:, 1]
        on_bdry = (x == 0) | (x == 1) | (y == 0) | (y == 1)
        self.boundary_idx = on_bdry.nonzero(as_tuple=False).view(-1).cpu().numpy()
        self.boundary_xy  = self.inputs_xy[self.boundary_idx]
        all_idx = np.arange(self.inputs_xy.shape[0], dtype=np.int64)
        self.interior_idx = np.setdiff1d(all_idx, self.boundary_idx)


        # ---- (C) Define manufactured fields and compute tensors
        (self.kappa_value_list,
         self.f_value_list,
         self.u_solution_tensor,
         self.grad_kappa_value_list) = self.generate_analytic_data(self.inputs_xy)

        # shape & dtype like before
        self.kappa_value_list      = self.kappa_value_list.reshape(-1, 1).contiguous()
        self.f_value_list          = self.f_value_list.reshape(-1, 1).contiguous()
        self.u_solution_tensor     = self.u_solution_tensor.reshape(-1, 1).contiguous()
        # boundary values (should be ~0 with our B(x,y) mask)
        self.boundary_u = self.u_solution_tensor[self.boundary_idx]

        # ---- (D) NN
        self.NN = FeatureLinear(NN_dim, activation="tanh", freeze_body=True, head_bias=True)

        self.NN.to(torch.float64)
        for p in self.NN.parameters(): p.requires_grad_(True)
        print("NN parameter gradient status:")
        for name, p in self.NN.named_parameters():
            print(f"  {name}: requires_grad={p.requires_grad}")

    def get_initial_params(self):
        params = collections.OrderedDict()
        for name, p in self.NN.named_parameters():
            if torch.is_tensor(p):
                params[name] = p.clone().requires_grad_(True)
            
        return params

    def NN_update(self, xvec):
        with torch.no_grad():
            offset = 0
            for p in self.NN.parameters():
                numel = p.numel()
                p.copy_(xvec[offset:offset+numel].view_as(p))
                offset += numel
        for p in self.NN.parameters(): p.requires_grad_(True)

    # ========= NEW: purely analytic data generator =========
    def generate_analytic_data(self, xy: torch.Tensor):
        """
        Inputs:
          xy : (N,2) float64 tensor of points in [0,1]^2
        Returns:
          kappa_values (N,), f_values (N,), u_true_values (N,), grad_kappa_values (N,2)
        """

        # Choose u* and kappa below.
        # (1) Boundary mask B ensures u* = 0 on ∂Ω
        def B(x, y):
            return x*(1.0 - x)*y*(1.0 - y)

        # (2) A smooth interior "shape" g(x,y)
        def g(x, y):
            # mildly nontrivial but smooth
            return 1.0 + 0.25*torch.sin(2*math.pi*x) * torch.sin(1*math.pi*y) + 0.1*x*y

        # Manufactured solution u*(x,y)
        def u_true(x, y):
            return B(x, y) * g(x, y)

        # (3) Choose kappa
        def kappa_fn(x, y):
            # Constant:
            #return 1.1*torch.ones_like(x)
            # Or variable (uncomment to try):
             #return 1.1 + 0.2*torch.sin(2*math.pi*x)*torch.cos(2*math.pi*y)
             return torch.full_like(x, 1.1) 

        # Autodiff to get grad u, lap u, grad k
        xy = xy.clone().detach().requires_grad_(True)
        x = xy[:, :1]
        y = xy[:, 1:2]

        kappa = kappa_fn(x, y)                        # (N,1)
        uStar = u_true(x, y)                          # (N,1)

        # ∇u*
        gu = torch.autograd.grad(uStar, xy,
                                 grad_outputs=torch.ones_like(uStar),
                                 create_graph=True, retain_graph=True)[0]   # (N,2)
        # Δu*
        lap = 0.0
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap_d = torch.autograd.grad(g_d, xy,
                                        grad_outputs=torch.ones_like(g_d),
                                        create_graph=True, retain_graph=True)[0][:, d:d+1]
            lap = lap + lap_d                         # (N,1)

        # ∇k
        #gradk = torch.autograd.grad(kappa, xy,
        #                            grad_outputs=torch.ones_like(kappa),
        #                            create_graph=True, retain_graph=True)[0]  # (N,2)
        gradk = torch.zeros_like(xy)

        # f = - div(k grad u) = -(∇k · ∇u) - k Δu
        f_vals = -(gradk * gu).sum(dim=1, keepdim=True) - kappa * lap   # (N,1)

        # detach all (we just want targets / coefficients)
        return (kappa.detach().view(-1),
                f_vals.detach().view(-1),
                uStar.detach().view(-1),
                gradk.detach())  # (N,2)
    
    def compute_metrics(self, params=None, use_hard_bc=True):
        """
        Returns a dict with L2_rel, Linf, residual_RMSE, residual_max on the class grid.
        """
    
        # unwrap TorchVect
        if params is not None and hasattr(params, "td"):
            params = params.td

        # predict on grid
        X, Y, U_pred = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)

        # true u on same grid
        xy = self.inputs_xy.detach().cpu().numpy()
        ut = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
        order = np.lexsort((xy[:,1], xy[:,0]))
        U_true = ut[order].reshape(X.shape)

        E = U_pred - U_true
        L2_rel = np.linalg.norm(E.ravel()) / max(1e-16, np.linalg.norm(U_true.ravel()))
        Linf   = np.max(np.abs(E))

        # residual on grid (uses stored k, gradk, b)
        k  = self.kappa_value_list.detach().cpu().numpy().reshape(-1,1)
        gk = self.grad_kappa_value_list.detach().cpu().numpy()
        b  = self.f_value_list.detach().cpu().numpy().reshape(-1,1)

        # compute ∇u and Δu from NN
        dev   = next(self.NN.parameters()).device
        dtype = next(self.NN.parameters()).dtype
        xg = self.inputs_xy.to(device=dev, dtype=dtype).detach().clone().requires_grad_(True)

        if params is None:
            u = self.NN(xg)
        else:
            params = collections.OrderedDict((k, v.to(dev, dtype)) for k, v in params.items())
            u = torch.func.functional_call(self.NN, params, (xg,))

        if use_hard_bc:
            B = (xg[:, :1]*(1-xg[:, :1])) * (xg[:, 1:2]*(1-xg[:, 1:2]))
            if params is not None and ('cB' in params):
                scale = torch.exp(params['cB'])
            elif hasattr(self.NN, 'cB'):
                scale = torch.exp(self.NN.cB)
            else:
                scale = torch.tensor(1.0, dtype=dtype, device=dev)
            u = scale * B * u

        gu = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
        lap = 0.0
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap += torch.autograd.grad(g_d, xg, torch.ones_like(g_d), create_graph=True)[0][:, d:d+1]

        gu_np  = gu.detach().cpu().numpy()
        lap_np = lap.detach().cpu().numpy()
        res = -(gk * gu_np).sum(axis=1, keepdims=True) - k * lap_np - b
        residual_RMSE = float(np.sqrt(np.mean(res**2)))
        residual_max  = float(np.max(np.abs(res)))

        return {
            "L2_rel": float(L2_rel),
            "Linf": float(Linf),
            "residual_RMSE": residual_RMSE,
            "residual_max": residual_max,
        }

    def plot_surface(self, X=None, Y=None, Z=None, title=None, prefer_trisurf=False):
        """
        Robust 3D surface plotter.
        - If X/Y/Z are None, it uses self.inputs_xy and self.u_solution_tensor.
        - Accepts torch tensors or numpy arrays.
        - If points form a structured grid, uses plot_surface; else falls back to plot_trisurf.
        """
    
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        def _to_np(a):
            if a is None:
                return None
            if hasattr(a, "detach"):  # torch.Tensor
                a = a.detach().cpu().numpy()
            return np.asarray(a, dtype=np.float64)

        # ---- Gather data ----
        if X is None or Y is None or Z is None:
            xy = _to_np(self.inputs_xy)            # (N,2)
            z  = _to_np(self.u_solution_tensor).reshape(-1)
            # sort to grid order and reshape
            order = np.lexsort((xy[:, 1], xy[:, 0]))
            x_sorted, y_sorted, z_sorted = xy[order, 0], xy[order, 1], z[order]
            nx, ny = np.unique(x_sorted).size, np.unique(y_sorted).size
            Xg = x_sorted.reshape(nx, ny)
            Yg = y_sorted.reshape(nx, ny)
            Zg = z_sorted.reshape(nx, ny)
            use_grid = True
        else:
            Xg, Yg, Zg = _to_np(X), _to_np(Y), _to_np(Z)
            # try to recognize a grid
            if Xg.ndim == Yg.ndim == Zg.ndim == 2 and Xg.shape == Yg.shape == Zg.shape and not prefer_trisurf:
                use_grid = True
            else:
                # 1D point lists → try grid reshape; else trisurf
                X1, Y1, Z1 = Xg.reshape(-1), Yg.reshape(-1), Zg.reshape(-1)
                nx, ny = np.unique(X1).size, np.unique(Y1).size
                if nx * ny == X1.size and not prefer_trisurf:
                    order = np.lexsort((Y1, X1))
                    Xg = X1[order].reshape(nx, ny)
                    Yg = Y1[order].reshape(nx, ny)
                    Zg = Z1[order].reshape(nx, ny)
                    use_grid = True
                else:
                    use_grid = False
                    Xg, Yg, Zg = X1, Y1, Z1  # for trisurf

        # ---- Finite checks ----
        if not np.isfinite(Xg).all() or not np.isfinite(Yg).all() or not np.isfinite(Zg).all():
            raise ValueError("Non-finite values found in plot arrays.")

        # ---- Plot ----
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        if use_grid:
            ax.plot_surface(Xg, Yg, Zg, linewidth=0, antialiased=True)
        else:
            ax.plot_trisurf(Xg, Yg, Zg, linewidth=0, antialiased=True)

        if title: ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("value")
        plt.tight_layout()
        plt.show()
    
    
    # inside class NNSetup
    def diagnose_scale(self, params=None):
        """
        Computes the best scalar s* for u_pred = s * B*N_raw (least-squares fit to u_true),
        returns suggested cB* = log(s*), and (rel L2 before, after).
        """
        import  math

        if params is not None and hasattr(params, "td"):
            params = params.td

        dev   = next(self.NN.parameters()).device
        dtype = next(self.NN.parameters()).dtype
        x = self.inputs_xy.to(device=dev, dtype=dtype)

        with torch.no_grad():
            if params is None:
                n_raw = self.NN(x)
            else:
                params = collections.OrderedDict((k, v.to(dev, dtype)) for k, v in params.items())
                n_raw = torch.func.functional_call(self.NN, params, (x,))

            B = (x[:, :1]*(1 - x[:, :1])) * (x[:, 1:2]*(1 - x[:, 1:2]))
            v = (B * n_raw).detach().cpu().numpy().reshape(-1)  # basis vector to scale
            u = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)

            num = float(np.dot(u, v))
            den = float(np.dot(v, v)) + 1e-16
            s_opt = num / den
            cB_opt = float(np.log(max(s_opt, 1e-16)))

            # current prediction (with existing cB if any)
            if params is not None and ('cB' in params):
                s_cur = float(np.exp(params['cB'].detach().cpu().numpy()))
            elif hasattr(self.NN, 'cB'):
                s_cur = float(np.exp(self.NN.cB.detach().cpu().numpy()))
            else:
                s_cur = 1.0

            U_cur = (s_cur * v)
            U_opt = (s_opt * v)
            L2_rel_cur = np.linalg.norm(U_cur - u) / max(1e-16, np.linalg.norm(u))
            L2_rel_opt = np.linalg.norm(U_opt - u) / max(1e-16, np.linalg.norm(u))

        print(f"[scale diag] s_cur={s_cur:.4e}, s_opt={s_opt:.4e}, ΔcB={math.log(max(s_opt,1e-16))-math.log(max(s_cur,1e-16)):.4e}")
        print(f"[scale diag] rel L2 (current) = {L2_rel_cur:.3e}, (with s_opt) = {L2_rel_opt:.3e}")
        return {"s_cur": s_cur, "s_opt": s_opt, "cB_opt": cB_opt,
                "L2_rel_current": float(L2_rel_cur), "L2_rel_opt": float(L2_rel_opt)}


    def _predict_u_grid(self, params=None, use_hard_bc=True):
        """
        Evaluate the NN on the class grid and return (X, Y, U) as numpy 2D arrays.
        - params: OrderedDict or TorchVect (optional). If None, uses live NN weights.
        - use_hard_bc: if True, returns B(x,y)*N(x,y)*exp(cB if present).
        """
    
        import  collections

        # unwrap TorchVect if needed
        if params is not None and hasattr(params, "td"):
             params = params.td
        # device/dtype
        dev   = next(self.NN.parameters()).device
        dtype = next(self.NN.parameters()).dtype

        # inputs (no grad needed for plotting)
        x = self.inputs_xy.to(device=dev, dtype=dtype)

        # forward (stateless if params provided)
        if params is None:
            with torch.no_grad():
                n_raw = self.NN(x)                             # (N,1)
        else:
            # ensure params tensors on same device/dtype
            params = collections.OrderedDict((k, v.to(dev, dtype)) for k, v in params.items())
            with torch.no_grad():
                n_raw = torch.func.functional_call(self.NN, params, (x,))  # (N,1)

        if use_hard_bc:
            # boundary mask B(x,y)
            B = (x[:, :1] * (1 - x[:, :1])) * (x[:, 1:2] * (1 - x[:, 1:2]))
            # optional global scale exp(cB) if present
            if params is not None and ('cB' in params):
                scale = torch.exp(params['cB'])
            elif hasattr(self.NN, 'cB'):
                scale = torch.exp(self.NN.cB)
            else:
                scale = torch.tensor(1.0, dtype=dtype, device=dev)
            u_pred = scale * B * n_raw
        else:
            u_pred = n_raw

        # convert to numpy and reshape onto grid
        xy = self.inputs_xy.detach().cpu().numpy()
        up = u_pred.detach().cpu().numpy().reshape(-1)

        order = np.lexsort((xy[:,1], xy[:,0]))  # sort by x then y
        x_sorted = xy[order, 0]; y_sorted = xy[order, 1]; u_sorted = up[order]
        nx = np.unique(x_sorted).size
        ny = np.unique(y_sorted).size

        X = x_sorted.reshape(nx, ny)
        Y = y_sorted.reshape(nx, ny)
        U = u_sorted.reshape(nx, ny)
        return X, Y, U

    def plot_nn_solution(self, params=None, title=r"NN prediction $u_\theta(x,y)$", use_hard_bc=True):
        """
        Compute NN prediction on the stored grid and plot a 3D surface.
        - params: OrderedDict or TorchVect with weights (optional)
        - use_hard_bc: multiply by B(x,y) (and exp(cB)) to match training ansatz
        """
        X, Y, U = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)
        self.plot_surface(X, Y, U, title=title)

    def plot_nn_error(self, params=None, title=r"|$u_\theta - u^*$|", use_hard_bc=True):
        """
        Plot absolute error surface |u_pred - u_true| on the grid.
        """
    
        X, Y, U_pred = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)
        # ground truth from class tensor
        xy = self.inputs_xy.detach().cpu().numpy()
        u_true = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
        order = np.lexsort((xy[:,1], xy[:,0]))
        U_true = u_true[order].reshape(X.shape)
        E = np.abs(U_pred - U_true)
        self.plot_surface(X, Y, E, title=title)
        
    def plot_residual_surface(self, params=None, title=r"PDE residual $r(x,y)$", use_hard_bc=True):
        """
        r(x,y) = -(∇k · ∇u) - k Δu - b  evaluated with the NN.
        """
    

        # unwrap TorchVect
        if params is not None and hasattr(params, "td"):
            params = params.td

        dev   = next(self.NN.parameters()).device
        dtype = next(self.NN.parameters()).dtype

        xg = self.inputs_xy.to(device=dev, dtype=dtype).detach().clone().requires_grad_(True)

        # forward (stateless if params is provided)
        if params is None:
            u = self.NN(xg)
        else:
            params = collections.OrderedDict((k, v.to(dev, dtype)) for k, v in params.items())
            u = torch.func.functional_call(self.NN, params, (xg,))

        if use_hard_bc:
            B = (xg[:, :1]*(1 - xg[:, :1])) * (xg[:, 1:2]*(1 - xg[:, 1:2]))
            if params is not None and ('cB' in params):
                scale = torch.exp(params['cB'])
            elif hasattr(self.NN, 'cB'):
                scale = torch.exp(self.NN.cB)
            else:
                scale = torch.tensor(1.0, dtype=dtype, device=dev)
            u = scale * B * u

        # ∇u and Δu
        gu = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
        lap = 0.0
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap += torch.autograd.grad(g_d, xg, torch.ones_like(g_d), create_graph=True)[0][:, d:d+1]

        # coefficients from stored tensors (already aligned to inputs_xy)
        k_np  = self.kappa_value_list.detach().cpu().numpy().reshape(-1, 1)
        gk_np = self.grad_kappa_value_list.detach().cpu().numpy()
        b_np  = self.f_value_list.detach().cpu().numpy().reshape(-1, 1)

        gu_np  = gu.detach().cpu().numpy()
        lap_np = lap.detach().cpu().numpy()

        res = -(gk_np * gu_np).sum(axis=1, keepdims=True) - k_np * lap_np - b_np  # (N,1)

        # reshape to grid and plot
        xy = self.inputs_xy.detach().cpu().numpy()
        order = np.lexsort((xy[:,1], xy[:,0]))
        nx = np.unique(xy[:,0]).size
        ny = np.unique(xy[:,1]).size
        R = res[order].reshape(nx, ny)
        X = xy[order,0].reshape(nx, ny)
        Y = xy[order,1].reshape(nx, ny)

        self.plot_surface(X, Y, R, title=title)
        # inside class NNSetup
    def plot_error_heatmap(self, params=None, title=r"|$u_\theta - u^*$|", use_hard_bc=True):
        
        if params is not None and hasattr(params, "td"):
            params = params.td

        # predict on grid
        X, Y, U_pred = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)

        # true grid
        xy = self.inputs_xy.detach().cpu().numpy()
        u_true = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
        order = np.lexsort((xy[:,1], xy[:,0]))
        U_true = u_true[order].reshape(X.shape)

        E = np.abs(U_pred - U_true)

        plt.figure(figsize=(6,5))
        im = plt.imshow(E.T, extent=[X.min(), X.max(), Y.min(), Y.max()],origin='lower', aspect='auto')
        plt.title(title); plt.xlabel("x"); plt.ylabel("y")
        plt.colorbar(im, label="|error|")
        plt.tight_layout(); plt.show()
        
    def plot_centerline(self, params=None, axis='y', value=0.5, use_hard_bc=True):
        """
        Plot u_pred vs u_true along x=value (axis='x') or y=value (axis='y').
        """
    

        X, Y, U = self._predict_u_grid(params=params, use_hard_bc=use_hard_bc)

        if axis == 'y':
            j = np.argmin(np.abs(Y[0,:] - value))
            xs = X[:, j]; up = U[:, j]
            # true
            xy = self.inputs_xy.detach().cpu().numpy()
            ut = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
            order = np.lexsort((xy[:,1], xy[:,0]))
            U_true = ut[order].reshape(X.shape)[:, j]
            plt.plot(xs, U_true, 'k-',  label='u*')
            plt.plot(xs, up,     'b--', label='u_theta')
            plt.xlabel('x'); plt.ylabel('u'); plt.title(f'Centerline y={value}')
            plt.legend(); plt.tight_layout(); plt.show()
        else:
            i = np.argmin(np.abs(X[:,0] - value))
            ys = Y[i, :]; up = U[i, :]
            xy = self.inputs_xy.detach().cpu().numpy()
            ut = self.u_solution_tensor.detach().cpu().numpy().reshape(-1)
            order = np.lexsort((xy[:,1], xy[:,0]))
            U_true = ut[order].reshape(X.shape)[i, :]
            plt.plot(ys, U_true, 'k-',  label='u*')
            plt.plot(ys, up,     'b--', label='u_theta')
            plt.xlabel('y'); plt.ylabel('u'); plt.title(f'Centerline x={value}')
            plt.legend(); plt.tight_layout(); plt.show()


    def returnVars(self, useEuclidean):
      var = {'beta':self.beta,
             'n':self.n,
             'nu':self.n,
             'nz':sum(p.numel() for p in self.NN.parameters()),
             'alpha':self.alpha,
             #'A':self.A,
             #'M':self.M,
             'NN':self.NN,
             'k':self.kappa_value_list,
             'b':self.f_value_list,
             'ud':self.u_solution_tensor,
             'useEuclidean':useEuclidean,
             #'mesh':torch.tensor(self.domain.geometry.x[:,0:1], dtype=torch.float64).T,
             #'mesh': torch.as_tensor(self.domain.geometry.x, dtype=torch.float64),  
             'inputs_xy': self.inputs_xy,            # (N,2)
             'gradk': self.grad_kappa_value_list,
             'bdry_xy':self.boundary_xy,
             'bdry_u':self.boundary_u,
             'lambda_reg': 1e-4,            # <-- choose λ here (0 to disable)
             'l2_exclude': ('bias', 'cB'),
             'interior_idx': torch.as_tensor(self.interior_idx,dtype=torch.long),
             'batch_hvp': 1024,
             'batch_loss': 4096
             
            }
      return var

def evaluate_pinn(var, params):
    """
    Returns numpy arrays: coords, u_pred, u_fe (if available), err (if available)
    params: TorchVect or OrderedDict of params
    """
    if isinstance(params, TorchVect):
        params = params.td
    if not isinstance(params, OrderedDict):
        params = OrderedDict(params)
    params = OrderedDict((k, v.detach()) for k, v in params.items())  # leaf tensors

    x_xy = var['inputs_xy']                          # (N,2)
    u_pred, _ = _stateless_u(var, params, x_xy)      # (N,1)

    coords = x_xy.cpu().numpy()
    u_np   = u_pred.detach().cpu().numpy().squeeze(-1)

    ud_np = None
    if 'ud' in var and var['ud'] is not None:
        ud_np = np.asarray(var['ud']).reshape(-1)

    err_np = None
    if ud_np is not None and ud_np.shape[0] == u_np.shape[0]:
        err_np = u_np - ud_np

    return coords, u_np, ud_np, err_np


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _dot_euclid(a, b):
    # a, b: TorchVect
    s = 0.0
    for k, va in a.td.items():
        vb = b.td[k]
        s += float((va.reshape(-1) @ vb.reshape(-1)).item())
    return s

def _as_vect(v):
    """
    Accept either a TorchVect or (TorchVect, ...) tuple and return the TorchVect.
    Raise if it's something unexpected.
    """
    if isinstance(v, tuple):
        v = v[0]
    if hasattr(v, "td"):  # TorchVect-like
        return v
    raise TypeError(f"Expected TorchVect (or tuple of), got {type(v)}")

def _rand_unit_like(x):
    d = x.clone()
    tot = 0.0
    for k, t in d.td.items():
        r = torch.randn_like(t)
        d.td[k] = r
        tot += float((r.reshape(-1) @ r.reshape(-1)).item())
    tot = (tot ** 0.5) + 1e-12
    for k in d.td:
        d.td[k] /= tot
    return d

# --- the robust checker ---

def check_grad_and_hv(problem, x):
    # Make eval deterministic (no dropout/BN drift)
    try:
        problem.var.NN.eval()
    except Exception:
        pass

    d = _rand_unit_like(x)

    f0, _ = problem.obj_smooth.value(x, 0.0)

    g_raw = problem.obj_smooth.gradient(x)
    g = _as_vect(g_raw)
    dirderiv = _dot_euclid(g, d)

    print("\nFinite Difference Gradient Check")
    print(f"{'t':>12s} {'DirDeriv':>14s} {'FinDiff':>14s} {'Error':>12s}")
    for t in [1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
        fp, _ = problem.obj_smooth.value(x + d * t, 0.0)
        fm, _ = problem.obj_smooth.value(x - d * t, 0.0)
        fd = (fp - fm) / (2.0 * t)
        err = abs(dirderiv - fd)
        print(f"{t:12.4e} {dirderiv:14.6e} {fd:14.6e} {err:12.4e}")

    # Hessian-vector (if available)
    if hasattr(problem.obj_smooth, "hessVec"):
        hv_raw = problem.obj_smooth.hessVec(d, x)
        hv = _as_vect(hv_raw)
        hv_dir = _dot_euclid(hv, d)

        print("\nFinite Difference Hessian Check")
        print(f"{'t':>12s} {'Hv·d':>14s} {'FinDiff':>14s} {'Error':>12s}")
        for t in [1.0,1e-1,1e-2,1e-3,1e-4]:
            gp_raw = problem.obj_smooth.gradient(x + d * t)
            gm_raw = problem.obj_smooth.gradient(x - d * t)
            gp = _as_vect(gp_raw); gm = _as_vect(gm_raw)
            fd_vec = (gp - gm) * (1.0 / (2.0 * t))   # TorchVect ops
            fd = _dot_euclid(fd_vec, d)
            err = abs(hv_dir - fd)
            print(f"{t:12.4e} {hv_dir:14.6e} {fd:14.6e} {err:12.4e}")
            


def tr_style_eval(nn, var, use_interior=True, chunk=8192):
    """
    Compute the exact objective that TR uses:
      value := 0.5 * mean( r^2 ) over interior points,
      r = -(∇k·∇u) - k Δu - b     with hard-BC u = exp(cB)*B*n_raw
    Returns (value, rmse).
    """
    dev   = next(nn.parameters()).device
    dtype = next(nn.parameters()).dtype

    X     = var['inputs_xy'].to(device=dev, dtype=dtype)
    k     = var['k'].to(device=dev, dtype=dtype)
    b     = var['b'].to(device=dev, dtype=dtype)
    gradk = var['gradk'].to(device=dev, dtype=dtype)
    alpha = float(var.get('alpha', 1.0))
    lam   = float(var.get('lambda_reg', 0.0))
    excl  = tuple(var.get('l2_exclude', ()))

    if use_interior and 'interior_idx' in var:
        I = var['interior_idx'].to(device=dev)
    elif use_interior:
        # build interior mask if not provided
        xb, yb = X[:,0], X[:,1]
        I = torch.arange(X.shape[0], device=dev)[~((xb==0)|(xb==1)|(yb==0)|(yb==1))]
    else:
        I = torch.arange(X.shape[0], device=dev)

    N = I.numel()
    if N == 0:
        return 0.0, 0.0

    def _forward_hard_bc(NN, x_xy):
        xg = x_xy if x_xy.requires_grad else x_xy.detach().clone().requires_grad_(True)
        n_raw = NN.net(xg)                       # your MLP body
        cB    = getattr(NN, 'cB', torch.tensor(2.7726, dtype=xg.dtype, device=xg.device))
        B     = (xg[:, :1]*(1-xg[:, :1])) * (xg[:, 1:2]*(1-xg[:, 1:2]))
        u     = torch.exp(cB) * B * n_raw
        return u, xg

    r2_accum = 0.0
    count = 0
    for s in range(0, N, chunk):
        j = I[s: s+chunk]

        with torch.enable_grad():
            Xc = X[j].detach().clone().requires_grad_(True)
            uc, Xc = _forward_hard_bc(nn, Xc)
            g  = torch.autograd.grad(uc, Xc, torch.ones_like(uc), create_graph=True, retain_graph=True)[0]
            lap = 0.0
            for d in range(2):
                gd = g[:, d:d+1]
                lap += torch.autograd.grad(gd, Xc, torch.ones_like(gd), create_graph=False, retain_graph=(d<1))[0][:, d:d+1]

            rc = -(gradk[j] * g).sum(1, keepdim=True) - k[j] * lap - b[j]   # (m,1)
            r2_accum += float((rc*rc).sum().item())
            count    += rc.numel()

    mean_r2 = r2_accum / max(count,1)
    value   = 0.5 * mean_r2
    rmse    = mean_r2**0.5
    l2 = 0.0
    if lam > 0.0:
        for n,p in nn.named_parameters():
            if any(tag in n for tag in excl):
                continue
            l2 += (p*p).sum().item()
        l2 = 0.5*lam*l2
    value = 0.5*alpha*mean_r2+l2
    return value, rmse
            
            
@torch.no_grad()
def eval_pinn_metrics(nn, var, chunk=8192):
    """
    Evaluate PINN quality on the *whole* dataset (in chunks to fit memory).
    Metrics:
      - L2_rel: ||u_pred - u_true|| / ||u_true||
      - Linf:   max |u_pred - u_true|
      - residual_RMSE: sqrt(mean r^2) over interior points (r = -(∇k·∇u) - kΔu - b)
      - residual_max: max |r|
    """
    dev   = next(nn.parameters()).device
    dtype = next(nn.parameters()).dtype

    X     = var['inputs_xy'].to(device=dev, dtype=dtype)
    k     = var['k'].to(device=dev, dtype=dtype)
    b     = var['b'].to(device=dev, dtype=dtype)
    gradk = var['gradk'].to(device=dev, dtype=dtype)
    utrue = var['ud'].to(device=dev, dtype=dtype)

    N = X.shape[0]
    xb, yb = X[:,0], X[:,1]
    bdry = (xb==0)|(xb==1)|(yb==0)|(yb==1)
    interior = (~bdry).nonzero(as_tuple=False).view(-1)

    # accumulators
    num = 0.0; den = 0.0
    linf = 0.0
    r2_sum = 0.0; r_abs_max = 0.0
    m = interior.numel()

    def _forward_hard_bc(NN, x_xy):
        xg = x_xy if x_xy.requires_grad else x_xy.detach().clone().requires_grad_(True)
        n_raw = NN.net(xg)
        cB = getattr(NN, 'cB', torch.tensor(2.7726, dtype=xg.dtype, device=xg.device))
        B  = (xg[:, :1]*(1-xg[:, :1]))*(xg[:, 1:2]*(1-xg[:, 1:2]))
        u  = torch.exp(cB) * B * n_raw
        return u, xg

    nn.eval()
    with torch.no_grad():
        ut_norm2 = (utrue*utrue).sum().item()
    den = ut_norm2**0.5 + 1e-16

    # need grads for residual/laplacian → compute in chunks with enable_grad
    for start in range(0, N, chunk):
        end = min(N, start+chunk)

        with torch.enable_grad():
            Xc = X[start:end].detach().clone().requires_grad_(True)
            uc, Xc = _forward_hard_bc(nn, Xc)
            g = torch.autograd.grad(uc, Xc, torch.ones_like(uc), create_graph=True, retain_graph=True)[0]
            lap = 0.0
            for d in range(2):
                gd = g[:, d:d+1]
                lap += torch.autograd.grad(gd, Xc, torch.ones_like(gd), create_graph=False, retain_graph=(d<1))[0][:, d:d+1]

        up = uc.detach()
        diff = up - utrue[start:end]
        num += (diff*diff).sum().item()
        linf = max(linf, float(diff.abs().max().item()))

        # residual only on interior slice indices
        idx_global = torch.arange(start, end, device=dev)
        mask = ~(((Xc[:,0]==0)|(Xc[:,0]==1)|(Xc[:,1]==0)|(Xc[:,1]==1))).view(-1,1)
        r = -(gradk[start:end] * g).sum(1, keepdim=True) - k[start:end] * lap - b[start:end]
        r = r[mask]
        if r.numel() > 0:
            r2_sum += float((r*r).mean().item() * r.numel())
            r_abs_max = max(r_abs_max, float(r.abs().max().item()))

        # free chunk graphs
        del Xc, uc, g, lap

    L2_rel = (num**0.5) / den
    # RMSE over interior points: we averaged per-chunk means → rescale back
    residual_RMSE = (r2_sum / max(m,1))**0.5
    return {
        'L2_rel': L2_rel,
        'Linf': linf,
        'residual_RMSE': residual_RMSE,
        'residual_max': r_abs_max
    }

def _l1_on_params(module, beta, exclude=('bias','cB')):
    if beta <= 0: return torch.tensor(0., dtype=torch.float64, device=next(module.parameters()).device)
    reg = 0.0
    for n,p in module.named_parameters():
        if any(e in n for e in exclude): continue
        reg = reg + p.abs().sum()
    return beta * reg



def adam_warmstart(nnset, var, steps=1000, lr=1e-3, batch=4096,
                   beta_l1=0.0, beta_l2=0.0,
                   plateau=60, lr_decay=0.5, seed=0,
                   eval_every=25, eval_chunk=8192,
                   target_L2_rel=None, target_res_RMSE=None,
                   use_ema=True, ema_decay=0.995):
    """
    Run Adam on minibatches until targets are reached or plateaued.
    Returns: TorchVect of the best weights (EMA if enabled), ready for TR.
    """
    print("adam_warmstart began")
    torch.manual_seed(seed)
    dev   = next(nnset.NN.parameters()).device
    dtype = next(nnset.NN.parameters()).dtype

    X     = var['inputs_xy'].to(device=dev, dtype=dtype)
    k     = var['k'].to(device=dev, dtype=dtype)
    b     = var['b'].to(device=dev, dtype=dtype)
    gradk = var['gradk'].to(device=dev, dtype=dtype)
    N     = X.shape[0]

    nn = nnset.NN
    trainable = [p for p in nn.parameters() if p.requires_grad]  # head only
    opt = torch.optim.Adam(trainable, lr=lr)
    nn.train()

    # interior indices only (hard BC)
    with torch.no_grad():
        xb, yb = X[:,0], X[:,1]
        bdry = (xb==0)|(xb==1)|(yb==0)|(yb==1)
        I_int = torch.arange(N, device=dev)[~bdry]
        if batch > len(I_int): batch = len(I_int)

    opt = torch.optim.Adam(nn.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                       factor=lr_decay, patience=max(5, plateau//5), verbose=False)

    # EMA state
    ema = None
    if use_ema:
        ema = {k: v.detach().clone() for k,v in nn.state_dict().items()}

    def _ema_update():
        if not use_ema: return
        sd = nn.state_dict()
        for k in ema.keys():
            ema[k].mul_(ema_decay).add_(sd[k], alpha=1.0-ema_decay)

    best = float('inf')
    best_sd = {k:v.detach().clone() for k,v in nn.state_dict().items()}
    stagn = 0

    def _forward_hard_bc(NN, x_xy):
        xg = x_xy if x_xy.requires_grad else x_xy.detach().clone().requires_grad_(True)
        n_raw = NN.net(xg)
        cB = getattr(NN, 'cB', torch.tensor(2.7726, dtype=xg.dtype, device=xg.device))
        B  = (xg[:, :1]*(1-xg[:, :1]))*(xg[:, 1:2]*(1-xg[:, 1:2]))
        u  = torch.exp(cB) * B * n_raw
        return u, xg

    for t in range(1, steps+1):
        opt.zero_grad(set_to_none=True)

        # sample interior minibatch
        I = I_int[torch.randint(low=0, high=len(I_int), size=(batch,), device=dev)]
        x_mb = X[I]; k_mb = k[I]; b_mb = b[I]; gradk_mb = gradk[I]

        # forward + residual loss
        u, xg = _forward_hard_bc(nn, x_mb)
        gu = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
        lap = 0.0
        for d in range(2):
            g_d = gu[:, d:d+1]
            lap += torch.autograd.grad(g_d, xg, torch.ones_like(g_d), create_graph=True)[0][:, d:d+1]
        r = -(gradk_mb * gu).sum(1, keepdim=True) - k_mb * lap - b_mb
        loss_pde = 0.5 * r.pow(2).mean()

        loss_l2 = 0.0
        if beta_l2 > 0:
            for n,p in nn.named_parameters():
                if 'bias' in n or 'cB' in n: continue
                loss_l2 = loss_l2 + (p*p).sum()
            loss_l2 = 0.5 * beta_l2 * loss_l2

        loss_l1 = _l1_on_params(nn, beta_l1)
        loss = loss_pde + loss_l2 + loss_l1

        loss.backward()
        opt.step()
        _ema_update()

        # periodic evaluation
        if t % eval_every == 0:
            # evaluate with current (or EMA) weights
            if use_ema:
                cur_sd = {k:v.detach().clone() for k,v in nn.state_dict().items()}
                nn.load_state_dict(ema, strict=True)
            try:
                metrics = eval_pinn_metrics(nn, var, chunk=eval_chunk)
                tr_val, tr_rmse = tr_style_eval(nn, var, use_interior=True, chunk=eval_chunk)
                #print(f"[Adam eval] TR-style value={tr_val:.6e}, RMSE={tr_rmse:.6e}")
            finally:
                if use_ema:
                    nn.load_state_dict(cur_sd, strict=True)

            # scheduler on residual RMSE (robust)
            sched.step(metrics['residual_RMSE'])

            # early stop targets
            hit_l2  = (target_L2_rel is not None and metrics['L2_rel'] <= target_L2_rel)
            hit_res = (target_res_RMSE is not None and metrics['residual_RMSE'] <= target_res_RMSE)

            score = metrics['residual_RMSE']  # ranking key
            improved = score < best*(1-1e-3)
            if improved:
                best = score; stagn = 0
                best_sd = ({k:v.detach().clone() for k,v in (ema if use_ema else nn.state_dict()).items()})
            else:
                stagn += 1

            # stop if good enough or plateau
            if hit_l2 or hit_res or stagn >= plateau:
                break

    # load best weights (EMA best if enabled)
    nn.load_state_dict(best_sd, strict=True)
    nn.eval()
    print("adam done")
    return TorchVect(nnset.get_initial_params())

def driver(savestats, name):
    print("driver started")
    np.random.seed(0)

    # settings
    n               = [30, 30]
    #NN_dims = [[2,500,1],[2,250,1],[2,125,1]]
    NN_dim_fine     = np.array([2, 60, 1])
    NN_dim_coarse   = np.array([2, 30, 1])
    #NN_dim_coarse_2 = np.array([2, 125, 1])
    if np.array_equal(NN_dim_fine, NN_dim_coarse):
        meshlist = [NN_dim_fine]
        n        = [n[0]]
    else:
        meshlist = [NN_dim_fine, NN_dim_coarse]#, NN_dim_coarse_2]


    alpha      = 1
    beta       = 1e-8
    #derivCheck = False
    lambda_warm = 1e-4
    lambda_TR = 1e-3
    problems = []
    x0 = None

    def _attach_global_shape(R_op,x_fine):
        """
        R_op.td[name]      : torch.Tensor of shape (coarse_numel, fine_numel)
        R_op.shapes_map    : dict name -> coarse param shape (e.g., (250,2))
        x_fine.td[name]    : fine param tensor with .shape (e.g., (500,2))
        """
        out_shapes_map = dict(R_op.shapes_map)                       # coarse
        in_shapes_map  = {name: x_fine.td[name].shape for name in R_op.td.keys()}  # fine

        # ensure scalars like 'cB' are present and identity-mapped
        if 'cB' not in R_op.td and 'cB' in x_fine.td:
            I = torch.eye(1, dtype=x_fine.td['cB'].dtype, device=x_fine.td['cB'].device)
            R_op.td['cB'] = I
            out_shapes_map['cB'] = (1,)
            in_shapes_map['cB']  = (1,)

        return RWrap(R_op.td, out_shapes_map=out_shapes_map, in_shapes_map=in_shapes_map)

    def _build_R_for_level(next_sizes, cur_sizes, x_fine):
        # try name-based first, fall back to encounter-order; use orthonormal averaging
        R_try = restriction_R_from_dims(next_sizes, cur_sizes, x_fine,
                                        mapping_mode="by_name", mode="orthonormal")
        ok = True
        for pname, W in R_try.td.items():
            out_shape = R_try.shapes_map[pname]
            in_size   = x_fine.td[pname].numel()
            out_size  = int(torch.tensor(out_shape).prod().item()) if len(out_shape) > 0 else 1
            if W.shape[1] != in_size or W.shape[0] != out_size:
                ok = False
                break
        if ok:
            return _attach_global_shape(R_try,x_fine)

        R_try = restriction_R_from_dims(next_sizes, cur_sizes, x_fine,
                                        mapping_mode="by_order", mode="orthonormal")
        for pname, W in R_try.td.items():
            out_shape = R_try.shapes_map[pname]
            in_size   = x_fine.td[pname].numel()
            out_size  = int(torch.tensor(out_shape).prod().item()) if len(out_shape) > 0 else 1
            assert W.shape[1] == in_size,  f"{pname}: R cols {W.shape[1]} != param size {in_size}"
            assert W.shape[0] == out_size, f"{pname}: R rows {W.shape[0]} != target size {out_size}"
        return _attach_global_shape(R_try,x_fine)
    x_seed_next = None

    for i in range(len(meshlist)):
        cur_sizes = meshlist[i]
        nnset = NNSetup(cur_sizes, n[i], alpha, beta, n_samples=1)
        var = nnset.returnVars(False)
        
        
        #x = TorchVect(nnset.NN.state_dict())
        if i == 0:
            var_warm = nnset.returnVars(False)
            var_warm['lambda_reg'] = lambda_warm
            x     = adam_warmstart(nnset, var_warm,steps=300, lr=2e-3, batch=4096,beta_l1=0.0,beta_l2=1e-6, plateau=200, lr_decay=0.5,eval_every=50, eval_chunk=8192,target_L2_rel=None,target_res_RMSE=None,use_ema=True, ema_decay=0.995)
            x0 = x
    #        print("blub")
        else:
            x = x_seed_next
        

        if i < len(meshlist) - 1:
            next_sizes = meshlist[i + 1]
            R_edge = _build_R_for_level(next_sizes, cur_sizes, x_fine=x)
     #       print("second blub")
            x_seed_next = R_edge @ x
            R = R_edge

            # optional strict checks
            for name, W in R.td.items():
                out_shape = R.shapes_map[name]
                in_size   = x.td[name].numel()
                out_size  = int(torch.tensor(out_shape).prod().item()) if len(out_shape) > 0 else 1
                assert W.shape[1] == in_size,  f"{name}: R cols {W.shape[1]} != param size {in_size}"
                assert W.shape[0] == out_size, f"{name}: R rows {W.shape[0]} != target size {out_size}"
        else:
            # coarsest level: identity restriction via same sizes (still orthonormal)
            #R = restriction_R_from_dims(cur_sizes, cur_sizes, x, mapping_mode="by_name", mode="orthonormal")
            R = _attach_global_shape(Reye(x),x)
            
            #R = _attach_global_shape(R)
        var['beta'] = beta

        trainable_names = {n for n, p in nnset.NN.named_parameters()
                   if (n.startswith("net.2") or n == "cB")}
    
 
        p = Problem(var, R)
        p.obj_smooth    = NNObjective(var)
        p.obj_nonsmooth = L1TorchNorm(var,include_names = trainable_names)
        p.pvector       = L2TVPrimal(var,include_names = trainable_names)
        p.dvector       = L2TVDual(var,include_names = trainable_names)
        problems.append(p)
        check_grad_and_hv(p, x)

        

    cnt = {}
    params = set_default_parameters("spg2")
    params["reltol"]  = False
    params['gtol']    = 8e-4
    params["t"]       = 2 / alpha
    params["ocScale"] = 1 / alpha
    #params['rtol']   = 3e-4
    #params['atol'] = 1e-6
    params['maxit'] = 1000
    params['maxitsp'] = 100
    params['gamma2'] = 2.0
    #params['deltamax'] = 1e6
    params["maxit"]   = 800
    params['deltamax'] = 1e13
    params['RgnormScale'] = 0.6
    params['RgnormScaleTol'] = 600
    params['recurse_cooldown_N'] = 3 
    # Sanity: evaluate TR-style loss with the warm-start weights
    tmp_nnset = NNSetup(meshlist[0], n[0], alpha, beta, n_samples=1)
    tmp_nnset.NN.load_state_dict(x0.td)
    tmp_nnset.NN.eval()
    #var_eval = tmp_nnset.returnVars(False)
    v0, rmse0 = tr_style_eval(tmp_nnset.NN, problems[0].var, use_interior=True)
    print(f"[Pre-TR] TR-style value={v0:.6e}, RMSE={rmse0:.6e}  (should match first TR row up to rounding)")

    print("[Pre-TR] param_hash:", param_hash(tmp_nnset.NN))
    
    start_time   = time.time()
    x, cnt_tr    = trustregion(0, x0, params['delta'], problems, params)
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")

    pro_tr = []
    cnt[1] = (cnt_tr, pro_tr)

    print("\nSummary")
    print("           niter     nobjs     ngrad     nhess     nobjn     nprox     ")
    print(
        f"   SGP2:  {cnt[1][0]['iter']:6d}    {cnt[1][0]['nobj1']:6d}    {cnt[1][0]['ngrad']:6d}    "
        f"{cnt[1][0]['nhess']:6d}    {cnt[1][0]['nobj2']:6d}    {cnt[1][0]['nprox']:6d}     "
    )

    final_nnset = NNSetup(meshlist[0], n[0], alpha, beta, n_samples=1)
    final_nnset.NN.load_state_dict(x.td)
    final_nnset.NN.eval()
    var = final_nnset.returnVars(False)
    

    final_nnset.plot_surface(title=r"Manufactured solution $u^*(x,y)$")

    apply_best_scale_live(final_nnset)
    final_nnset.plot_nn_solution(title=r"NN prediction $u_\theta$ (after sign+scale)")
    print(final_nnset.compute_metrics())

    
    final_nnset.plot_nn_error(title=r"abs error")
    metrics = final_nnset.compute_metrics()
    print(metrics)
    final_nnset.diagnose_scale()
    
    final_nnset.plot_residual_surface(title=r"PDE residual $r(x,y)$")
    final_nnset.plot_error_heatmap(title=r"|$u_\theta - u^*$|")
    final_nnset.plot_centerline()
    

    print("Updated neural network is stored in `updated_nn`.")
    return cnt

cnt = driver(False, "test_run")
