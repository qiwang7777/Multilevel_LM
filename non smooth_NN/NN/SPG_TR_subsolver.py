import numpy as np


def tk(Bk,gk,t0=0.1,t_min=1e-30,t_max=1e30):
    if gk.T@Bk@gk>0:
        tk0=np.linalg.norm(gk,2)**2/(gk.T@Bk@gk)
    else:
        tk0=t0/np.linalg.norm(gk,2)
    t = max(tk0,t_min)
    return min(t_max,t)


def prox_l1(y, lam):
    """
    Proximal operator for the L1 norm.

    Parameters
    ----------
    y : np.array
        Input vector.
    lam : float
        Regularization parameter (threshold).

    Returns
    -------
    np.array
        Result after applying the proximal operator.
    """
    return np.sign(y) * np.maximum(np.abs(y) - lam, 0)


def hk(xk,gk,prox,tk):
    """
    hk = ||(xk-Prox_{tkphik}(xk-tkgk))||/tk

    """
    return np.linalg.norm(xk-prox(xk-tk*gk,tk))/tk


def alpha_solver(xk,x0,s,deltak):
    """
    Solve for alpha>0 in ||xk+alpha*s-x0||=deltak 

    """
    a = np.dot(s,s)
    b = 2*np.dot(s,xk-x0)
    c = np.dot(xk-x0,xk-x0)-deltak**2
    discriminant = b**2-4*a*c
    if discriminant<0:
        return None
    sqrt_discriminant = np.sqrt(discriminant)
    alpha1 = (-b+sqrt_discriminant)/(2*a)
    alpha2 = (-b-sqrt_discriminant)/(2*a)
    if alpha1 > 0:
        return alpha1
    if alpha2 > 0:
        return alpha2
    else:
        return None
    


def SPG_TR_subsolver(xk,f,phi,grad_f,Bk,tauk,lambdak,deltak,maxit=1,tau_bar=1e-5,tmin=1e-30,tmax=1e30):
    l=0
    x0 = xk
    tk0 = lambdak
    hk0 = hk(x0,grad_f(f,x0),prox_l1,lambdak)
    hkl = hk(xk,grad_f(f,xk),prox_l1,lambdak)
    while l<maxit and hkl>min(tau_bar,tauk*hk0) and np.linalg.norm(xk-x0,2)<deltak:
        dk = grad_f(f,xk)
        s = prox_l1(xk-lambdak*dk, lambdak)-xk
        alpha_max = 1
        if np.linalg.norm(xk+s-x0)>deltak:
            alpha_max = alpha_solver(xk,x0,s,deltak)
        
        phik = phi(xk+s)
        b = Bk@s
        kappa = np.dot(b,s)
        if kappa <= 0:
            alpha = alpha_max
        else:
            alpha = min(alpha_max,-(np.dot(dk,s)+phik-phi(xk))/kappa)
            
        xk = xk+alpha*s
        dk = dk+alpha*b
        phik = phi(xk)
        if kappa<=0:
            lambda_bar = tk0/np.linalg.norm(dk,2)
        else:
            lambda_bar = np.dot(s,s)/kappa
            
        lambdak = max(tmin,min(tmax,lambda_bar))
        l += 1
    return xk

