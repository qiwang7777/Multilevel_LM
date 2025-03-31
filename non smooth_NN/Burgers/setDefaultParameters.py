import numpy as np
def set_default_parameters(name):
    params = {}

    # General Parameters
    params['spsolver'] = name.replace(' ', '')
    params['outFreq'] = 1
    params['debug'] = False
    params['initProx'] = False
    params['t'] = 1
    params['safeguard'] = np.sqrt(np.finfo(float).eps)

    # Stopping tolerances
    params['maxit'] = 200
    params['reltol'] = True
    params['gtol'] = 1e-5
    params['stol'] = 1e-10
    params['ocScale'] = params['t']

    # Trust-region parameters
    params['eta1'] = 0.05
    params['eta2'] = 0.9
    params['gamma1'] = 0.25
    params['gamma2'] = 2.5
    params['delta'] = 50.0
    params['deltamax'] = 1e10

    # Subproblem solve tolerances
    params['atol'] = 1e-5
    params['rtol'] = 1e-3
    params['spexp'] = 2
    params['maxitsp'] = 15

    # GCP and subproblem solve parameter
    params['useGCP'] = False
    params['mu1'] = 1e-4
    params['beta_dec'] = 0.1
    params['beta_inc'] = 10.0
    params['maxit_inc'] = 2

    # SPG and spectral GCP parameters
    params['lam_min'] = 1e-12
    params['lam_max'] = 1e12

    # Inexactness parameters
    params['useInexactObj'] = False
    params['useInexactGrad'] = False
    params['gradTol'] = np.sqrt(np.finfo(float).eps)



    return params
