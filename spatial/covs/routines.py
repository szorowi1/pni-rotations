from math import log, sqrt
from numba import jit

@jit
def logdet_ar1(T, sigma, rho):
    '''Log-determinant of symmetric tridiagonal matrix.
    
    Notes
    -----
    Estimates 50,000 in under 50ms.
    
    '''
    f1, f2 = 1, 0
    rho_2 = rho ** 2

    for i in range(2,T+2):
        
        f = f1 - f2 * rho_2
        f2 = f1
        f1 = f

    return T * 2 * log(sigma) + log(f)

@jit
def cholesky_ar1(arr_1, arr_2, T, V):
    '''Cholesky decomposition of AR(1) matrix.
    
    Parameters
    ----------
    arr_1 : array, shape = (T*V*2-1,)
        Data of csc sparse matrix.
    arr_2 : array, shape = (V,)
        AR(1) parameters per voxel.
    T, V : int
        Number of time points and voxels.
        
    Returns
    -------
    arr_1 : array, shape = (T*V*2-1,)
        Cholesky decomposition of AR(1) matrix.
        
    Notes
    -----
    Takes the data of a sparse AR(1) matrix in csc format.
    Computes the Cholesky decomposition such that:
    
    .. math::
    
        A = LL^T
    
    Using numba to speed up operation. At initial testing,
    could handle 40,000 voxels and 250 time points in 150ms.
    '''
    
    ## Error-catching.
    assert len(arr_2) == V
    
    ## Define metadata.
    n_total = len(arr_1) - 1    # Total elements
    n_per_voxel = T * 2 - 1     # Elements per voxel.
    
    ## Initialize values.
    arr_1[0] = 1
    v, i = 0, 1

    while i < n_total:
                
        ## Initialize block matrix per new voxel. 
        if not i % n_per_voxel: 
            arr_1[i:i+2] = [0, 1]
            v += 1
            i += 2
            
        ## Compute off-diagonal element.
        arr_1[i] = arr_2[v] * arr_1[i-1] ** -1
        i += 1
        
        ## Compute diagonal element.
        arr_1[i] = sqrt(1 - arr_1[i-1] ** 2)
        i += 1
                    
    return arr_1