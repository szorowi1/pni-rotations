import numpy as np
import scipy.sparse as sp
from .likelihood import matnormal_lpdf

def _matnormal_wrapper(params, indices, Y, X, Omega_k, Omega_v, Sigma_t, Sigma_v):
    """Internal convenience function."""
    
    ## Update covariances.
    for i, cov in enumerate([Omega_k, Omega_v, Sigma_t, Sigma_v]):
        if np.any(indices==i): cov.update(*params[indices==i])
    
    return -matnormal_lpdf(Y, X, Omega_k, Omega_v, Sigma_t, Sigma_v)

def matnormal_regression(Y, X, Omega_k, Omega_v, Sigma_t, Sigma_v, method='L-BFGS-B'):
    """Wrapper function for matrix-normal regression.
    
    Parameters
    ----------
    Y : array, shape (T, V)
        BOLD data.
    X : sparse CSC matrix, shape (T, K)
        Design matrix
    Omega_k : `cov` instance, shape (K, K)
        Regressor weight precision matrix.
    Omega_v : `cov` instance, shape (V, V)
        Spatial weight precision matrix.
    Sigma_t : `cov` instance, shape (T, T)
        Temporal noise covariance matrix.
    Sigma_v : `cov` instance, shape (V, V)
        Spatial noise covariance matrix.
    method : str or callable
        Type of solver. See scipy.optimize.minimize for details.
      
    Returns
    -------
    res : OptimizeResult 
        The optimization result represented as a OptimizeResult object. Important 
        attributes are: `W` regression weights; `x` the solution array; `success` 
        a Boolean flag indicating if the optimizer exited successfully,  `message` 
        which describes the cause of the termination. See `scipy.optimize.OptimizeResult` 
        for details.
    """
    
    ## Define metadata.
    T, V = Y.shape
    T, K = X.shape
    
    ## Prepare matrices.
    X = sp.kron(sp.eye(V), X, format='csc')
    Y = Y.reshape(V*T,1,order='F')
    
    ## Assemble optimization parameters.
    indices = np.concatenate([np.repeat(i, len(cov.params)) for i, cov in 
                              enumerate([Omega_k, Omega_v, Sigma_t, Sigma_v])])
    bounds = Omega_k.bounds + Omega_v.bounds + Sigma_t.bounds + Sigma_v.bounds
    
    ## Optimization.
    x0 = Omega_v.params + Omega_k.params + Sigma_t.params + Sigma_v.params
    res = minimize(_matnormal_wrapper, x0, method=method, bounds=bounds,
                   args=(indices, Y, X, Omega_k, Omega_v, Sigma_t, Sigma_v))
    
    ## Compute W.
    res['W'] = matnormal_inverse(Y, X, Omega_k, Omega_v, Sigma_t, Sigma_v)
    return res