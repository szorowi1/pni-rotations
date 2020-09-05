import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.special import gammaln
from scipy.special import gamma as fgamma

def surf_to_adj(tris, remap_vertices=True):
    """Compute adjacency matrix from surface mesh triangles.
    
    Parameters
    ----------
    tris : array
      N x 3 array defining triangles.
    remap_vertices : bool
      Reassign vertex indices based on unique values. 
      Useful to process a subset of triangles. Defaults to True.
      
    Returns
    -------
    A : sparse CSC matrix
      The adjacency matrix describing the spatial graph structure.
      
    References
    ----------
    https://martinos.org/mne/stable/generated/mne.spatial_tris_connectivity.html
    """
    from mne import spatial_tris_connectivity
    return spatial_tris_connectivity(tris, remap_vertices, False).tocsc()

def vol_to_adj(rr):
    """Compute adjacency matrix from voxel coordinates.
        
    Parameters
    ----------
    rr : nd array.
      Points in N-dimensional space.

    Returns
    -------
    A : sparse CSC matrix
      The adjacency matrix describing the spatial graph structure.
    """

    ## Compute distances.
    rr = cdist(rr,rr)

    ## Identify adjacent points.
    rows, cols = np.where(rr==1)
    data = np.ones_like(rows)

    ## Construct sparse matrices.
    A = sp.coo_matrix((data, (rows,cols)), shape=rr.shape)

    return A.tocsc()

def spm_hrf(RT, P=np.array([6, 16, 1, 1, 6, 0, 32], dtype=float), fMRI_T=16):
    """Python implementation of spm_hrf.
    
    Parameters
    ----------
    RT : float
      Repetition time (TR)
    p : array, shape = (6,)
      Parameters of response function. See notes for details.
    fMRI_T : float
      Length of kernel (seocnds).
        
    Returns
    -------
    hrf : array
      Hemodynamic response function
    
    Notes
    -----
    1) delay of response               (default =  6)
    2) delay of undershoot             (default = 16) 
    3) dispersion of response          (default =  1)
    4) ratio of response to undershoot (default =  6)
    5) onset (seconds)                 (default =  0)
    6) length of kernel (seconds)	   (default = 32)
    
    References
    ----------
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/modelgen.py
    """
    if len(P) != 6: raise ValueError('Must specify six parameters!')

    _spm_Gpdf = lambda x, h, l: np.exp(h * np.log(l) + (h - 1) * np.log(x) - (l * x) - gammaln(h))
    # modelled hemodynamic response function - {mixture of Gammas}
    dt = RT / float(fMRI_T)
    u = np.arange(0, int(p[6] / dt + 1)) - p[5] / dt
    with np.errstate(divide='ignore'):  # Known division-by-zero
        hrf = _spm_Gpdf(u, p[0] / p[2], dt / p[2]) - _spm_Gpdf(u, p[1] / p[3],
                                                               dt / p[3]) / p[4]
    idx = np.arange(0, int((p[6] / RT) + 1)) * fMRI_T
    hrf = hrf[idx]
    hrf = hrf / np.sum(hrf)
    return hrf