import numpy as np
from scipy.sparse import csc_matrix, eye
from .cholesky import log_det_sym_tri

def adj_to_ugl(A):
    """Convert adjacency matrix to unweighted graph laplacian.
    
    Parameters
    ----------
    A : sparse CSC matrix
      The adjacency matrix describing the spatial graph structure.
    
    Returns
    -------
    UGL : sparse CSC matrix
      The unweighted graph laplacian matrix corresponding to A.
    """
    
    ## Temporary conversion.
    A = A.copy().tocoo()
    
    ## Extract metadata.
    row, col, data = A.row, A.col, A.data
    
    ## Compute vertex degree.
    D = np.asarray(A.sum(axis=0)).squeeze()
    
    ## Construct new matrix.
    row = np.concatenate([row, np.arange(D.size)])
    col = np.concatenate([col, np.arange(D.size)])
    data = np.concatenate([data * -1, D])
    
    return csc_matrix((data, (row, col)), shape=A.shape)

class CovIdentity(csc_matrix): 
    """Identity covariance matrix.

    Parameters
    ----------
    n : int
        Number of rows/columns of matrix.

    Attributes
    ----------
    params : list
        Covariance parameters.
    bounds : list
        Covariance parameter bounds.
    logdet : float
        Log-determinant.

    Notes
    -----
    The identity matrix is defined as:

    .. math::

        C = I_n
    """
    
    def __init__(self, n):
                
        ## Initialize matrix.
        csc_matrix.__init__(self,eye(n))
        
        ## Store metadata.
        self.params = []
        self.bounds = []
        
        ## Store log-determinant.
        self.logdet = 0
    
class CovIsotropic(csc_matrix):
    """Isotropic covariance matrix.

    Parameters
    ----------
    n : int
        Number of rows/columns of matrix.
    sigma : float
        Standard deviation initial value (default=1).

    Attributes
    ----------
    params : list
        Covariance parameters.
    bounds : list
        Covariance parameter bounds.
    logdet : float
        Log-determinant.

    Notes
    -----
    The isotropic matrix is defined as:

    .. math::
    
        C = \sigma^2 I_n
    """
    
    def __init__(self, n, sigma=None):
        
        ## Initialize parameters.
        if sigma is None: sigma = 1
        self._sigma = sigma 
        
        ## Initialize matrix.
        csc_matrix.__init__(self,eye(n)*sigma**2)
        self._data = self.data
        
        ## Store metadata.
        self.params = [self._sigma]
        self.bounds = [(1e-6, None)]
        
        ## Store log-determinant.
        self.logdet = self.nnz * np.log(self._sigma)
                    
    def update(self, sigma):
        """Update hyperparameters."""
        self._sigma = sigma
        self.data = self._data * self._sigma**2
        self.logdet = self.nnz * np.log(self._sigma)
        self.params = [self._sigma]   
    
class CovGraphLaplacian(csc_matrix):
    """Graph laplacian precision matrix.

    Parameters
    ----------
    A : sparse CSC matrix
        Adjacency matrix.
    sigma : float
        Standard deviation initial value (default=1).
      
    Attributes
    ----------
    params : list
        Covariance parameters.
    bounds : list
        Covariance parameter bounds.
    logdet : float
        Log-determinant.

    Notes
    -----
    The graph laplacian matrix is defined as:

    .. math::
    
        C = \sigma^2(D - A)^{-1}
        
    where :math:`D` is the degree matrix and :math:`A` is the adjacency matrix. :code:`CovGraphLaplacian` is parameterized as the sparse precision matrix, :math:`C^{-1}`.  
    """
    
    def __init__(self, A, sigma=None):

        ## Initialize parameters.
        if sigma is None: sigma = 1
        self._sigma = sigma
        
        ## Construct matrix.
        csc_matrix.__init__(self, adj_to_ugl(A) * self._sigma**-2)
        self._data = self.data
        
        ## Store metadata.
        self.params = [self._sigma]
        self.bounds = [(1e-6, None)]
    
        ## Store log-determinant.
        self.logdet = 2 * (self.shape[0] - 1) * np.log(self._sigma)
    
    def update(self, sigma):
        """Update hyperparameters."""
        self._sigma = sigma
        self.data = self._data * self._sigma**-2
        self.logdet = 2 * (self.shape[0] - 1) * np.log(self._sigma)
        self.params = [self._sigma]
        
class CovBlockDiagAR1(object):
    
    def __init__(self, T, V, sigma=None, rho=None):  
        
        ## Initialize parameters.
        if sigma is None: sigma = 1
        if rho is None: rho = 0
        if not hasattr(rho, '__len__'): rho = np.repeat(rho, V)
        assert V == rho.size
            
        ## Store parameters.
        self.shape = (T,V)
        self.sigma = sigma
        self.rho = rho            
            
        ## Store log-determinant.
        self.logdet = sum(logdet(T,sigma,p) for p in rho)