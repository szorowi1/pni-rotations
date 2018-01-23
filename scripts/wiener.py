import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()

## Load RWiener package.
rwiener = importr('RWiener')

def wiener_pdf(q, alpha, tau, beta, delta, resp='upper', give_log=False):
    
    ## Force to itererable array.
    Q = np.array(q, dtype=float).reshape(-1)
    
    ## Force to float.
    alpha, tau, beta, delta = [float(x) for x in [alpha, tau, beta, delta]]
    
    ## Iteratively compute PDF.
    return np.array([rwiener.dwiener(q, alpha, tau, beta, delta, resp, give_log) for q in Q]).squeeze()

def wiener_rng(alpha, tau, beta, delta, size=1):
    
    ## Force to float.
    alpha, tau, beta, delta = [float(x) for x in [alpha, tau, beta, delta]]
    
    ## Simulate.
    yz = [rwiener.rwiener(1, alpha, tau, beta, delta) for _ in np.arange(size)]
    yz = np.concatenate(yz, axis=-1).squeeze()
    
    return yz[0], yz[1].astype(int)