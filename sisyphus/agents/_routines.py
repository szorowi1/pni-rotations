import numpy as np
from scipy.stats import rankdata

def softmax(arr):
    """Scale-robust softmax."""
    arr = np.exp(arr - np.max(arr))
    return arr / arr.sum()

def metric_sampling(v, alpha, err=1e-3):
    """Proportional prioritization sampling.
    
    Parameters
    ----------
    v : array
        Sampling values.
    alpha : float
        Scaling exponent (see notes).
    err : float
        Small positive constant preventing no sampling of values of 0.
    
    Returns
    -------
    ix : int
        Sampling index.
        
    Notes
    -----
    Probability of sampling value i is:
    
    .. math::
    
        P(i) = p_i^\alpha / \sum_k p_k^\alpha
        
    where :math:`\p_i` is the priority of value i defined as:
    
    .. math::
    
        p(i) = | v_i | + \epsilon
    
    The exponent :math:`\alpha` determines how much prioritization is used, 
    with :math:`\alpha=0` corresponding to the uniform case.
    
    References
    ----------
    1. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. 
       arXiv preprint arXiv:1511.05952.
    """
    
    ## Prevent no-sampling issue.
    v = np.abs(v.copy()) + err
    
    ## Compute prioritization values.
    p = v ** alpha / np.sum(v ** alpha)
    
    ## Sampling.
    return np.random.choice(np.arange(v.size), 1, p=p)[0]

def rank_sampling(v, alpha):
    """Rank prioritization sampling.
    
    Parameters
    ----------
    v : array
        Sampling values.
    alpha : float
        Scaling exponent (see notes).
    
    Returns
    -------
    ix : int
        Sampling index.
        
    Notes
    -----
    Probability of sampling value i is:
    
    .. math::
    
        P(i) = p_i^\alpha / \sum_k p_k^\alpha
        
    where :math:`\p_i` is the priority of value i defined as:
    
    .. math::
    
        p(i) = 1 / rank( | v_i | )
    
    The exponent :math:`\alpha` determines how much prioritization is used, 
    with :math:`\alpha=0` corresponding to the uniform case.
    
    References
    ----------
    1. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. 
       arXiv preprint arXiv:1511.05952.
    """
    
    ## Compute ranks.
    r = 1 / (v.size - rankdata(np.abs(v), method='min') + 1)
    
    ## Compute prioritization values.
    p = r ** alpha / np.sum(r ** alpha)
    
    ## Sampling.
    return np.random.choice(np.arange(v.size), 1, p=p)[0]