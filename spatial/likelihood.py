import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from sksparse.cholmod import cholesky

def logdet(A):
    return cholesky(A).logdet
    
def solve(A,B):
    return cholesky(A)(B)

def matnormal_inverse(Y, X, Omega_k, Omega_v, Sigma_t, Sigma_v):
    """Solve for regression weights, W.
    
    Parameters
    ----------
    Y : array, shape (T*V, 1)
        Vectorized BOLD data.
    X : sparse CSC matrix, shape (T*V, K*V)
        Block diagonal design matrix.
    Omega_k : sparse CSC matrix, shape (K, K)
        Regressor weight precision matrix.
    Omega_v : sparse CSC matrix, shape (V, V)
        Spatial weight precision matrix.
    Sigma_t : sparse CSC matrix, shape (T, T)
        Temporal noise covariance matrix.
    Sigma_v : sparse CSC matrix, shape (V, V)
        Spatial noise covariance matrix.
    
    Returns
    -------
    W : array, shape (T*V, 1)
        Vectorized regression weights.
    """
    
    ## Precompute matrices.
    Omega_w = sp.kron(Omega_v, Omega_k, format='csc')
    Sigma_n = sp.kron(Sigma_v, Sigma_t, format='csc')
    
    ## Compute term 1.
    M = Omega_w + X.T @ solve(Sigma_n, X)
    
    ## Compute term 2.
    XSnY = X.T @ solve(Sigma_n, Y)
    
    ## Solve for W.
    return solve(M, XSnY)

def matnormal_lpdf(Y, X, Omega_k, Omega_v, Sigma_t, Sigma_v):
    """Matrix-normal log-likelihood.
    
    Parameters
    ----------
    Y : array, shape (T*V, 1)
        Vectorized BOLD data.
    X : sparse CSC matrix, shape (T*V, K*V)
        Block diagonal design matrix.
    Omega_k : sparse CSC matrix, shape (K, K)
        Regressor weight precision matrix.
    Omega_v : sparse CSC matrix, shape (V, V)
        Spatial weight precision matrix.
    Sigma_t : sparse CSC matrix, shape (T, T)
        Temporal noise covariance matrix.
    Sigma_v : sparse CSC matrix, shape (V, V)
        Spatial noise covariance matrix.
    
    Returns
    -------
    L : scalar
        Matrix-normal log-likelihood.
    
    Notes
    -----
    Quadratic form assumes matrix inversion lemma.
    
    .. math::
    
        (A + CBC^T)^{-1} = A^{-1} - A^{-1}C(B^{-1} + C^TA^{-1}C)^{-1}C^TA^{-1}

    Log-determinant assumes matrix determinant lemma.
    
    .. math::
    
        \log| A + CBC^T | = \log| B^{-1} + C^TA^{-1}C) | + \log| B | + \log| A |
        
    """

    ## Precompute matrices.
    Omega_w = sp.kron(Omega_v, Omega_k, format='csc')
    Sigma_n = sp.kron(Sigma_v, Sigma_t, format='csc')
    
    SnX = solve(Sigma_n, X)
    XSnX = X.T @ SnX
    SnY = solve(Sigma_n, Y)
    
    ## Log quadratic term.
    a = Y.T @ SnY
    b = Y.T @ SnX @ solve(Omega_w + XSnX, X.T.tocsc()) @ SnY    
    log_quad = -0.5 * (a - b)    

    ## Log determinant.
    a = logdet( Omega_w + XSnX )
    b = Omega_k.size * Omega_v.logdet
    c = Sigma_t.size * Sigma_v.logdet + Sigma_v.size * Sigma_t.logdet
    log_det = -0.5 * ( a + b + c )
    
    ## Additive constant.
    const = -0.5 * Y.size * np.log(2 * np.pi)

    return const + log_det + log_quad