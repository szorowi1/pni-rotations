import numpy as np
from warnings import warn
from ._routines import metric_sampling, rank_sampling, softmax

class DynaQ(object):
    """SARSA-learning with prioritized-sweeping Dyna architecture.
    
    Parameters
    ----------
    beta : float
        Inverse temperature for choice.
    eta : float
        Learning rate.
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
    1. Sutton, R. S. (1991). Dyna, an integrated architecture for learning, planning, 
       and reacting. ACM SIGART Bulletin, 2(4), 160-163.
    2. Peng, J., & Williams, R. J. (1993). Efficient learning and planning within the 
       Dyna framework. Adaptive Behavior, 1(4), 437-454. 
    3. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. 
       arXiv preprint arXiv:1511.05952.
    """
    
    def __init__(self, beta, eta, gamma, replay=False, w=0.5, 
                 sampling='rank', alpha=1, outcome=False):
        
        ## Error catching.
        if abs(beta) > 50:
            warn('Parameter "beta" set very large.')
        if eta <= 0 or eta > 1:
            raise ValueError('Parameter "eta" must be in range (0,1].')
        if gamma < 0 or gamma > 1:
            raise ValueError('Parameter "gamma" must be in range [0,1].') 
        if w < 0 or w > 1:
            raise ValueError('Parameter "w" must be in range [0,1].') 
        if replay > 1000:
            warn('Parameter "replay" set very large.')
        
        ## Define choice & learning parameters.
        self.beta = beta
        self.eta = eta
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
    
    def _select_action(self, env, Q, s):
        
        ## Identify possible actions.
        a = env.T[s].indices
            
        ## Compute likelihood of action under current policy.
        theta = softmax(Q[s, a] * self.beta)
        
        ## Select action.        
        return a[np.argmax(np.random.multinomial(1, theta))]
    
    def _update_model(self, Q, s, a, r, s_prime, a_prime):
    
        ## Compute reward prediction error.
        delta = r + self.gamma * Q[s_prime, a_prime] - Q[s,a]
            
        ## Update values.
        Q[s,a] += self.eta * delta
        
        return Q, delta
    
    def _prioritized_replay(self, env, memory, Q):
        
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
        Q, delta = self._update_model(Q, s, a, r, s_prime, a_prime)

        ## Store updated memory.
        memory = np.vstack([memory, (s,a,r,delta,s_prime,a_prime,1)])
        
        ## Remove old predecessor states.      
        memory = memory[memory[:,-1].nonzero()]      
        
        ## Identify predecessor states. 
        S_bar = env.T.tocsc()[:,s].indices
        if not np.any(S_bar): return Q, memory
        
        ## Gather predecessor state info.
        a_bar = s
        r_bar = env.R[s]
        D_bar = [r_bar + self.gamma * Q[s,a] - Q[s_bar,a_bar] for s_bar in S_bar]
        
        ## Store predecessor states in memory.
        for s_bar, d_bar in zip(S_bar, D_bar):
            memory = np.vstack([memory, (s_bar,a_bar,r_bar,d_bar,s,a,0)])
            
        return Q, memory
    
    def train_trial(self, env, Q, memory=False, start=None, n_steps=100, limit=1000):
        """Run one trial.
        
        Parameters
        ----------
        env : GridWorld or GraphWorld instance
            Simulation environment.
        Q : sparse CSR matrix
            Q-value matrix, mapped to the one-state transition matrix.
        start : int
            Starting state. Defaults to env.start.
        n_steps : int
            Maximum number of steps in trial.
        
        Returns
        -------
        Q : sparse CSR matrix
            Updated Q-value matrix.
        """
        
        ## Define memory.
        if not np.any(memory): memory = np.empty((0,7))
        assert memory.shape[-1] == 7
        
        ## Define starting state.
        if start is None: start = env.start
        assert start is not None
        s = start
        
        ## Select first action under current policy.
        a = self._select_action(env, Q, s)
        
        ## Main loop.
        for _ in np.arange(n_steps):
            
            ## Check for termination.
            if s in env.terminal: break        

            ## Observe sucessor state.
            s_prime = a
            
            ## Observe outcome.
            r = env.R[s_prime]
                        
            ## Select successor action under current policy.
            a_prime = self._select_action(env, Q, s_prime)
                
            ## Update model.
            Q, delta = self._update_model(Q, s, a, r, s_prime, a_prime)
            
            ## Store memory.
            memory = np.vstack([memory, (s,a,r,delta,s_prime,a_prime,1)])
            
            ## Update state.
            s, a = s_prime, a_prime            
            
        ## Replay block.
        for _ in np.arange(self.replay): 
            
            Q, memory = self._prioritized_replay(env, memory, Q)
        
        ## Clean-up memory.
        memory = memory[memory[:,-1].nonzero()]      
        memory = memory[-limit:]
            
        return Q, memory
    
    def test_trial(self, env, Q, start=False, n_steps=100):
        '''Run a single test episode.

        Parameters
        ----------
        env : GridWorld or GraphWorld instance
            Simulation environment.
        Q : array, shape = (n,n)
            Q-value matrix, mapped to the one-state transition matrix.
        start : int
            Starting state. Defaults to env.start.
        n_steps : int
            Maximum number of steps in trial.

        Returns
        -------
        states : array, shape = (n,)
            List of visited states in episode.
        cumsum : float
            Cumulative reward in episode.
        '''   

        ## Define starting state.
        if not start: start = env.start
        s = start  
        
        ## Initialize values.
        states = np.array([start])
        cumsum = 0

        for _ in np.arange(n_steps):

            ## Check for termination.
            s = int(states[-1])
            if s in env.terminal: break

            ## Select action under current policy.
            a = self._select_action(env, Q, s)
            s_prime = a

            ## Observe outcome.
            r = env.R[s_prime]
            cumsum += r

            ## Update state(s).
            s = s_prime
            states = np.append(states, s_prime)

        return states, cumsum