import numpy as np
from scipy.stats import binom, norm
from . utilities import inv_logit

## Shifted Wald distribution functions.
def shifted_wald_pdf(x, gamma, alpha, theta, err=1e-16):    

    with np.errstate(invalid='ignore'):
        tmp1 = alpha / np.sqrt(2 * np.pi * np.power(x-theta,3))
        tmp2 = np.power(alpha-gamma*(x-theta), 2) / (2*(x-theta))
        p = tmp1 * np.exp(-1*tmp2)
    
    return np.where(np.isnan(p), err, p)

def shifted_wald_lpdf(x, gamma, alpha, theta, err=1e-16):
    return np.log(shifted_wald_pdf(x, gamma, alpha, theta, err))

def shifted_wald_rng(gamma, alpha, theta, ub=3, gridsize=1000):
    
    gamma, alpha, theta = [np.copy(arr).reshape(-1) for arr in [gamma, alpha, theta]]
    x = np.linspace(0,ub,gridsize)
    z = np.zeros_like(gamma)
    
    for i in np.arange(gamma.size):
        p = shifted_wald_pdf(x, gamma[i], alpha[i], theta[i], err=0)
        z[i] = np.random.choice(x, p=p/p.sum())
    
    return z

## Shifted Wald Stan utilities.
def init_shifted_wald(N, H, n_chains=4):
    
    init = dict(
        
        ## Group-level parameters.
        mu_pr = np.random.normal(0, 1, H),
        sigma = np.random.gamma(1, 2, H),
        sigma_m = np.random.gamma(1, 2),
        
        ## Subject-level parameters.
        gama_pr = np.random.normal(0, 1, N),
        alpha_pr = np.random.normal(0, 1, N),
        beta_pr = np.random.normal(0, 1, N),
        eta_v_pr = np.random.normal(0, 1, N),
        
        theta = np.ones(N) * 0.1,
        beta_h = np.random.normal(0, 1, N)
        
    )
    
    return [init] * n_chains

def wald_generate_quantities(fit):
    
    ## Extract task information.
    N, B, T = [fit[param] for param in ['N','B','T']]
    X, Y, R, Z = [fit[param].copy() for param in ['X','Y','R','Z']]
    M, m2 = [fit[param] for param in ['M','m2']]
    
    ## Zero-indexing.
    X -= 1
    Y -= 1
    
    ## Extract RL parameters.
    beta = fit['beta']
    eta_v = fit['eta_v']
    eta_h = fit.get('eta_h', np.zeros_like(beta))
    f = fit.get('f', np.ones_like(beta))
    
    ## Extract Wald parameters.
    gamma = fit['gamma']
    alpha = fit['alpha']
    theta = fit['theta']
    
    ## Extract mood parameters.
    beta_h = fit['beta_h']
    sigma_m = fit['sigma_m']
    S, _ = beta_h.shape
    
    ## Preallocate space.
    Y_pred = np.zeros((S, N, B, T))       # Predicted choice data.
    Z_pred = np.zeros((S, N, B, T))       # Predicted RT data.
    h_pred = np.zeros((S, N, B, T))       # Predicted reward history data.
    Y_log_lik = np.zeros((S, N, B, T))    # Choice log-likelihood
    Z_log_lik = np.zeros((S, N, B, T))    # RT log-likelihood.
    M_log_lik = np.zeros((S, N, B, 3))    # Mood log-likelihood.
        
    for i in np.arange(N):

        ## Initialize values.
        Q = np.zeros((S,9))
        h = np.zeros(S)
        m = np.tanh(beta_h[:,i])

        for j in np.arange(B):

            ## If second block, update reward history/mood
            ## to reflect Wheel of Fortune.
            if np.any(eta_h) and j == 1:
                m = np.ones_like(m) * m2[i]
                h = np.arctanh(m)

            for k in np.arange(T):

                if Z[i,j,k] > 0:
    
                    ## Compute difference in expected values / drift rate.
                    dEV = beta[:,i] * (Q[:, X[i,j,k,1]] - Q[:, X[i,j,k,0]])
                    drift = gamma[:,i] * np.abs((Q[:, X[i,j,k,1]] - Q[:, X[i,j,k,0]]))

                    ## Compute log-likelihood of choice.
                    Y_log_lik[:, i, j, k] = binom.logpmf(Y[i,j,k], 1, inv_logit(dEV))

                    ## Simulate choice given current model.
                    Y_pred[:, i, j, k] = binom.rvs(1, inv_logit(dEV)) + 1

                    ## Compute log-likelihood of RT.
                    Z_log_lik[:, i, j, k] = shifted_wald_lpdf(Z[i,j,k], drift, alpha[:,i], theta[:,i])

                    ## Simulate RT given current model.
                    Z_pred[:, i, j, k] = shifted_wald_rng( drift, alpha[:,i], theta[:,i] )
    
                    ## Compute reward prediction error.
                    delta = (f[:,i] ** m) * R[i,j,k] - Q[:,X[i,j,k,Y[i,j,k]]]
            
                    ## Update expectations.
                    Q[:,X[i,j,k,Y[i,j,k]]] += eta_v[:,i] * delta

                    ## Update history of rewards.
                    h += eta_h[:,i] * (delta - h)

                    ## Store reward history.
                    h_pred[:,i,j,k] = h
                                        
                    ## Update mood.
                    m = np.tanh( beta_h[:,i] + h )
                    
                else:
                    
                    ## Compute log-likelihood of choice.
                    Y_log_lik[:, i, j, k] = 0

                    ## Simulate choice given current model.
                    Y_pred[:, i, j, k] = -1

                    ## Compute log-likelihood of RT.
                    Z_log_lik[:, i, j, k] = 0

                    ## Simulate RT given current model.
                    Z_pred[:, i, j, k] = -1
                    
                    ## Store reward history.
                    h_pred[:,i,j,k] = h
                    
                if k == 7-1:
                    
                    M_log_lik[:, i, j, 0] = norm.logpdf( M[i,j,0], m, sigma_m )
                    
                elif k == 21-1:
                    
                    M_log_lik[:, i, j, 1] = norm.logpdf( M[i,j,1], m, sigma_m )
                    
                elif k == 35-1:
                    
                    M_log_lik[:, i, j, 2] = norm.logpdf( M[i,j,2], m, sigma_m )
    
    ## Store information.
    fit['Y_pred'] = Y_pred
    fit['Z_pred'] = Z_pred
    fit['h_pred'] = h_pred
    fit['Y_log_lik'] = Y_log_lik
    fit['Z_log_lik'] = Z_log_lik
    fit['M_log_lik'] = M_log_lik    
    
    return fit