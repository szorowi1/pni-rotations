import numpy as np
import _pickle as cPickle
from scipy.special import gamma as fgamma

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