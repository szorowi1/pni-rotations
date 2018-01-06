import pickle, pystan 
import numpy as np
from scipy.special import gamma as fgamma

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### General utilities.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Stan utilities.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## - Stan user case studies: http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html
## - Michael Betancourt github: https://github.com/betanalpha/jupyter_case_studies/blob/master/pystan_workflow/stan_utility.py

def check_div(fit):
    """Check transitions that ended with a divergence.
    If divergences, try running with larger adapt_delta to remove the divergences.
    """
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = [x for y in sampler_params for x in y['divergent__']]
    n = sum(divergent)
    N = len(divergent)
    return n / N * 100

def check_treedepth(fit, max_depth = 10):
    """Check transitions that ended prematurely due to maximum tree depth limit
    If saturations exist, Run again with max_depth set to a larger value to avoid saturation.
    """
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    depths = [x for y in sampler_params for x in y['treedepth__']]
    n = sum(1 for x in depths if x == max_depth)
    N = len(depths)
    return n / N * 100

def check_energy(fit):
    """Checks the energy Bayesian fraction of missing information (E-BFMI)
    If E-BFMI below 0.2 indicates you may need to reparameterize your model'"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    warning = False
    for chain_num, s in enumerate(sampler_params):
        energies = s['energy__']
        numer = sum((energies[i] - energies[i - 1])**2 for i in range(1, len(energies))) / len(energies)
        denom = np.var(energies)
        if numer / denom < 0.2:
            print('Chain {}: E-BFMI = {}'.format(chain_num, numer / denom))
            warning = True
    return warning

def check_n_eff(fit):
    """Checks the effective sample size per iteration"""
    fit_summary = fit.summary(probs=[0.5])
    n_effs = [x[4] for x in fit_summary['summary']]
    names = fit_summary['summary_rownames']
    n_iter = len(fit.extract()['lp__'])

    no_warning = True
    for n_eff, name in zip(n_effs, names):
        ratio = n_eff / n_iter
        if (ratio < 0.001):
            print('n_eff / iter for parameter {} is {}!'.format(name, ratio))
            print('E-BFMI below 0.2 indicates you may need to reparameterize your model')
            no_warning = False
    if no_warning:
        print('n_eff / iter looks reasonable for all parameters')
    else:
        print('  n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated')

def check_rhat(fit):
    """Checks the potential scale reduction factors"""
    from math import isnan
    from math import isinf

    fit_summary = fit.summary(probs=[0.5])
    rhats = [x[5] for x in fit_summary['summary']]
    names = fit_summary['summary_rownames']

    no_warning = True
    for rhat, name in zip(rhats, names):
        if (rhat > 1.1 or isnan(rhat) or isinf(rhat)):
            print('Rhat for parameter {} is {}!'.format(name, rhat))
            no_warning = False
    if no_warning:
        print('Rhat looks reasonable for all parameters')
    else:
        print('  Rhat above 1.1 indicates that the chains very likely have not mixed')


def _by_chain(unpermuted_extraction):
    num_chains = len(unpermuted_extraction[0])
    result = [[] for _ in range(num_chains)]
    for c in range(num_chains):
        for i in range(len(unpermuted_extraction)):
            result[c].append(unpermuted_extraction[i][c])
    return np.array(result)

def _shaped_ordered_params(fit):
    ef = fit.extract(permuted=False, inc_warmup=False) # flattened, unpermuted, by (iteration, chain)
    ef = _by_chain(ef)
    ef = ef.reshape(-1, len(ef[0][0]))
    ef = ef[:, 0:len(fit.flatnames)] # drop lp__
    shaped = {}
    idx = 0
    for dim, param_name in zip(fit.par_dims, fit.extract().keys()):
        length = int(np.prod(dim))
        shaped[param_name] = ef[:,idx:idx + length]
        shaped[param_name].reshape(*([-1] + dim))
        idx += length
    return shaped

def partition_div(fit):
    """ Returns parameter arrays separated into divergent and non-divergent transitions"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    div = np.concatenate([x['divergent__'] for x in sampler_params]).astype('int')
    params = _shaped_ordered_params(fit)
    nondiv_params = dict((key, params[key][div == 0]) for key in params)
    div_params = dict((key, params[key][div == 1]) for key in params)
    return nondiv_params, div_params