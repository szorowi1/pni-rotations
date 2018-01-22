import numpy as np
import _pickle as cPickle
from scipy.special import gamma as fgamma
from . psis import psisloo

def inv_logit(arr):
    '''Elementwise inverse logit (logistic) function.'''
    return 1 / (1 + np.exp(-arr))

def phi_approx(arr):
    '''Elementwise fast approximation of the cumulative unit normal. 
    For details, see Bowling et al. (2009). "A logistic approximation 
    to the cumulative normal distribution."'''
    return inv_logit(0.07056 * arr ** 3 + 1.5976 * arr)
                     
def to_shape_rate(mode, sd):
    '''Convert parameters from gamma(mode, sd) to gamma(shape, rate).'''
    rate = ( mode + np.sqrt( mode**2 + 4*sd**2 ) ) / ( 2 * sd**2 )
    shape = 1 + mode * rate
    return shape, rate

def gamma_pdf(x, s, r):
    '''Probability density function for the (shape, rate)-parameterized
    gamma distribution.'''
    return r ** s / fgamma(s) * x ** (s - 1) * np.exp(-r * x)

def HDIofMCMC(arr, credMass=0.95):
    '''
    Computes highest density interval from a sample of representative values,
    estimated as shortest credible interval. Functions for computing HDI's are 
    explained in Chapter 25 of Doing Bayesian Data Analysis, Second Edition.
    
    INPUTS:
    -- arr: a vector of representative values from a probability distribution.
    -- credMass: a scalar between 0 and 1, indicating the mass within the credible
       interval that is to be estimated.
    '''
    sortedPts = np.sort(arr)
    ciIdxInc = np.ceil(credMass * len( sortedPts )).astype(int)
    nCIs = len( sortedPts ) - ciIdxInc
    ciWidth = [ sortedPts[ i + ciIdxInc ] - sortedPts[ i ] for i in np.arange(nCIs).astype(int) ]
    HDImin = sortedPts[ np.argmin( ciWidth ) ]
    HDImax = sortedPts[ np.argmin( ciWidth ) + ciIdxInc ]
    return HDImin, HDImax

def psis_model_comparison(a, b):
    '''Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC'''
    ## Main loop.
    LOO = []
    for model_name in [a,b]:

        ## Load StanFit file.
        f = 'stan_fits/%s/StanFit.pickle' %model_name
        with open(f, 'rb') as f: extract = cPickle.load(f)

        ## Extract log-likelihood values.
        Y_log_lik = extract['Y_log_lik']
        M_log_lik = extract['M_log_lik']
        n_samp, n_subj, n_block, n_trial = Y_log_lik.shape

        ## Reshape data.
        Y_log_lik = Y_log_lik.reshape(n_samp, n_subj*n_block*n_trial)
        M_log_lik = M_log_lik.reshape(n_samp, n_subj*n_block*3)

        ## Remove log-likelihoods corresponding to missing data.
        Y_log_lik = np.where(Y_log_lik, Y_log_lik, np.nan)
        missing = np.isnan(Y_log_lik).mean(axis=0) > 0
        Y_log_lik = Y_log_lik[:,~missing] 

        ## Compute PSIS-LOO.
        _, loo, _ = psisloo(np.concatenate([Y_log_lik, M_log_lik], axis=-1))
        LOO.append(loo)
        
    ## Perform model comparison.
    LOO = -2 * np.array(LOO)
    m1, m2 = np.sum(LOO, axis=-1)
    se = np.sqrt( LOO.shape[-1] * np.var(np.diff(LOO, axis=0)) )
    
    print('Model comparison')
    print('----------------')
    print('PSIS[1] = %0.0f' %m1)
    print('PSIS[2] = %0.0f' %m2)
    print('Diff\t= %0.2f (%0.2f)' %(m1-m2, se))
    
    return m1, m2, se