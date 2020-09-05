import numpy as np
from sksparse.cholmod import cholesky

def logdet(A):
    return cholesky(A).logdet
    
def solve(A,B):
    return cholesky(A)(B)

class GraphLaplacian(object):
    
    def __init__(self, AR=1, tau_w=None, tau_r=None, sigma=None):
        
        ## Should define some instances here.
        
    def fit(Y, X, Z, method='L-BFGS-B'):
        
        def _prepare_
        
        
    
    #def predict
    
    #def score
    
    
    
    

def W_lpdf(Y, X, Omega_w, Omega_n):
    
    ## Precompute matrices.
    M = Omega_w + X.T @ Omega_n @ X
    Sigma = Omega_n @ X @ solve(M, X.T @ Omega_n)
    
    ## Log-likelihood: quadratic form.
    a = Y.T @ Omega_n @ Y
    b = Y.T @ Sigma @ Y
    log_quad = a - b
    
    ## Log-likelihood: log determinant.
    a = logdet( M )
    b = Omega_w.logdet
    c = Omega_n.logdet
    log_det = a + b + c
    
    ## Log-likelihood: additive constant.
    const = Y.size * np.log(2 * np.pi)
    
    ## Log-likelihood.
    loglik = -0.5 * (log_quad + log_det + const)

def W_gradient(Y, X, Omega_w, Omega_n):
    
    ## Precompute matrices.
    M = Omega_w + X.T @ Omega_n @ X
    L = Omega_n.derivative
    
    ## Quadratic form.
    a = Y @ Omega_n @ X
    b = solve(solve(-M, L) @ M)
    c = X.T @ Omega_n @ Y.T
    grad_quad = a @ b @ c
    
    ## Log-determinant.
    a = np.sum(M * L.T)
    b = 
    
class LaplacianRegression(object):
    
    def __init__(self, alpha=0.1, log_lik=False):
        
        self.alpha = 0.1
        self.log_lik = log_lik
        
    def fit()