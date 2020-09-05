import numpy as np
from scipy.sparse import csr_matrix
from warnings import warn
from ._routines import metric_sampling, rank_sampling, softmax

class DynaSR(object):
    """Successor representation with prioritized-sweeping Dyna architecture.
    
    Parameters
    ----------
    beta : float
        Inverse temperature for choice.
    eta_td : float
        Learning rate on reward weights.
    eta_sr : float
        Learning rate on successor matrix.
    gamma : float
        Discount factor.
    replay : int
        Number of replay events at end of trial.
    w : float
        Prioritization bias.
    sampling : str or function
        Sampling method. Can be "metric", "rank", or user-defined function.
    alpha : float
        Prioritization scaling.
    outcome : bool
        If True, replay sampling is prioritized by absolute outcome. 
        Otherwise it is prioritized by absolute RPE. 
        
    References
    ----------
    1. Dayan, P. (1993). Improving generalization for temporal difference learning: 
       The successor representation. Neural Computation, 5(4), 613-624.
    2. Russek, E. M., Momennejad, I., Botvinick, M. M., Gershman, S. J., & Daw, N. D. (2017). 
       Predictive representations can link model-based reinforcement learning to model-free 
       mechanisms. PLoS computational biology, 13(9), e1005768.
    3. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. 
       arXiv preprint arXiv:1511.05952.
    """    
    def __init__(self, beta, eta_td, eta_sr, gamma, replay=False, w=0.5, 
                 sampling='rank', alpha=1, outcome=False):
        
        ## Error catching.
        if abs(beta) > 50:
            warn('Parameter "beta" set very large.')
        if eta_td <= 0 or eta_td > 1:
            raise ValueError('Parameter "eta_td" must be in range (0,1].')
        if eta_sr <= 0 or eta_sr > 1:
            raise ValueError('Parameter "eta_sr" must be in range (0,1].')
        if gamma < 0 or gamma > 1:
            raise ValueError('Parameter "gamma" must be in range [0,1].')
        if w < 0 or w > 1:
            raise ValueError('Parameter "w" must be in range [0,1].') 
        if replay > 1000:
            warn('Parameter "replay" set very large.')
        
        ## Define choice & learning parameters.
        self.beta = beta
        self.eta_td = eta_td
        self.eta_sr = eta_sr
        self.gamma = gamma
        
        ## Define replay parameters.
        self.replay = int(replay)
        self.w = w
        
        ## Define sampling method.
        self.alpha = alpha
        if hasattr(sampling, '__call__'):
            self.sampling = sampling
        elif sampling == 'metric':
            self.sampling = lambda v: metric_sampling(v, self.alpha)
        elif sampling == 'rank':
            self.sampling = lambda v: rank_sampling(v, self.alpha)
        else:
            raise ValueError('Parameter "sampling" must be "metric","rank", or function.')
        
        ## Define prioritization (RPE or outcome).
        self.outcome = outcome
        if outcome: self._search_col = 2    # Outcome-based sampling
        else: self._search_col = 3          # RPE-based sampling
        
    def _select_action(self, env, H, w, s):

        ## Compute Q-values.
        Q = H @ w
    
        ## Identify possible actions.
        a = env.T[s].indices
        sa = env.T[s].data
            
        ## Compute likelihood of action under current policy.
        theta = softmax(Q[sa] * self.beta)
        
        ## Select action.
        i = np.argmax(np.random.multinomial(1, theta))
        return a[i], sa[i]

    def _update_sr(self, sa, sa_prime, H):

        ## Define ones vector.
        t = np.zeros(H[sa].size)
        t[sa] = 1
        
        ## Update state-action SR matrix (H).
        H[sa] = (1 - self.eta_sr) * H[sa] + self.eta_sr * (t + self.gamma * H[sa_prime])
        
        return H 
    
    def _update_w(self, sa, r, sa_prime, H, w):
        
        ## Compute normalizing term.
        norm_feature_rep_s = H[sa] / ( H[sa] @ H[sa].T )
        
        ## Compute reward prediction error.
        delta = r + self.gamma * (H[sa_prime] @ w) - (H[sa] @ w)

        ## Update weights.
        w += self.eta_td * delta * norm_feature_rep_s

        return w, delta
    
    def _update_model(self, H, w, sa, r, sa_prime):
                    
        ## Update state-action SR matrix.
        H = self._update_sr(sa, sa_prime, H)

        ## Update weights.
        w, delta = self._update_w(sa, r, sa_prime, H, w)

        return H, w, delta
        
    def _prioritized_replay(self, env, memory, H, w):
        
        ## Sample event from memory.
        v = memory[:,self._search_col]
        v = np.where(v > 0, self.w * v, (1-self.w) * v)
        i = self.sampling(v)
                
        ## Unpack and delete memory.
        s, a, _, _, s_prime, a_prime, _ = memory[i].astype(int)
        memory = np.delete(memory, i, axis=0)
        
        ## Observe outcome.
        r = env.R[s_prime]

        ## Update model.
        sa = env.T[s,a]
        sa_prime = env.T[s_prime,a_prime]
        H, w, delta = self._update_model(H, w, sa, r, sa_prime)

        ## Store updated memory.
        memory = np.vstack([memory, (s,a,r,delta,s_prime,a_prime,1)])
        
        ## Remove old predecessor states.      
        memory = memory[memory[:,-1].nonzero()]      
        
        ## Identify predecessor states. 
        s_bar = env.T.tocsc()[:,s].indices
        sa_bar = env.T[:,s].data
        if not np.any(s_bar): return H, w, memory
        
        ## Gather predecessor state info.
        a_bar = [np.argmax(env.T[sb].indices == s) for sb in s_bar]
        rb = env.R[s]
        delta_bar = [rb + self.gamma * H[sa] @ w - H[sb] @ w for sb in sa_bar]

        ## Store predecessor states in memory.
        for sb, ab, db in zip(s_bar, a_bar, delta_bar):
            memory = np.vstack([memory, (sb,ab,rb,db,s,a,0)])
            
        return H, w, memory
        
    def train_trial(self, env, H, w, memory=None, start=None, n_steps=100, limit=1000):
        
        ## Define memory.
        if not np.any(memory): memory = np.empty((0,7))
        assert memory.shape[-1] == 7
        
        ## Define starting state.
        if start is None: start = env.start
        assert start is not None
        s = start
        
        ## Select first action under current policy.
        a, sa = self._select_action(env, H, w, s)
        
        ## Main loop.
        for _ in np.arange(n_steps):
            
            ## Check for termination.
            if s in env.terminal: break        
            
            ## Observe sucessor state.
            s_prime = a
            
            ## Observe outcome.
            r = env.R[s_prime]
                        
            ## Select successor action under current policy.
            a_prime, sa_prime = self._select_action(env, H, w, s_prime)

            ## Update model.
            H, w, delta = self._update_model(H, w, sa, r, sa_prime)
            
            ## Store memory.
            memory = np.vstack([memory, (s,a,r,delta,s_prime,a_prime,1)])
            
            ## Update state.
            s, a, sa = s_prime, a_prime, sa_prime          
            
        ## Replay block.
        for _ in np.arange(self.replay): 
            
            H, w, memory = self._prioritized_replay(env, memory, H, w)
        
        ## Clean-up memory.
        memory = memory[memory[:,-1].nonzero()]      
        memory = memory[-limit:]
            
        return H, w, memory