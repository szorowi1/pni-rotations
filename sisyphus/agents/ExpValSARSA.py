import numpy as np
from warnings import warn
from ._routines import softmax

class ExpValSARSA(object):
    '''(Off-policy) expected value SARSA.
    
    Parameters
    ----------
    beta : float
      Inverse temperature for choice.
    eta : float
      Learning rate.
    gamma : float
      Discount factor.
    tau : float
      Inverse temperature for learning.
      
    Notes
    -----
    Expected value SARSA is a variant of the classic SARSA algorithm that is expected
    to have slightly faster convergence by weighing the value of the successor state,
    s', by the likelihood of its respective actions under the current policy. As 
    discussed in Sutton & Barto, the learning rule is as follows:
    
    .. math::
    
        \delta = r + \gamma \sum_a \pi(a' | s')Q(s',a') - Q(s,a)
        
    Note that this is similar to including a softmax function in the learning rule. Thus,
    there are several possible regimes given :math:`\beta` and :math:`\tau`.

    - :math:`\tau >> \beta`: Q-learning (Sutton & Barto, 1998).
    - :math:`\tau = \beta`: expected value SARSA (Sutton & Barto, 1998).
    - :math:`\tau > 0`: soft Q-learning (Nachum et al., 2017).
    - :math:`\tau = 0`: stochastic learning.
    - :math:`\tau < 0`: beta-pessimistic learning (Gaskett, 2003).
    - :math:`\tau << 0`: minimax learning (Heger, 1994).
    
    References
    ----------
    1. Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT press.
    2. Heger, M. (1994). Consideration of risk in reinforcement learning. In Machine Learning 
       Proceedings 1994 (pp. 105-111).
    3. Gaskett, C. (2003). Reinforcement learning under circumstances beyond its control.
    4. Nachum, O., Norouzi, M., Xu, K., & Schuurmans, D. (2017). Bridging the gap between value 
       and policy based reinforcement learning. In Advances in Neural Information Processing 
       Systems (pp. 2775-2785).
    '''
    
    def __init__(self, beta, eta, gamma, tau=False):
        
        ## Error catching.
        if abs(beta) > 50:
            warn('Parameter "beta" set very large.')
        if eta <= 0 or eta > 1:
            raise ValueError('Parameter "eta" must be in range (0,1].')
        if gamma < 0 or gamma > 1:
            raise ValueError('Parameter "gamma" must be in range [0,1].') 
        if abs(tau) > 50:
            warn('Parameter "tau" set very large.')
        
        ## Define parameters.
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        if not tau: self.tau = beta
        else: self.tau = tau
            
    def _select_action(self, env, Q, s):
                
        ## Identify possible actions.
        a = env.T[s].indices
            
        ## Compute likelihood of action under current policy.
        theta = softmax(Q[s, a] * self.beta)
        
        ## Select action.        
        return a[np.argmax(np.random.multinomial(1, theta))]
    
    def _update_model(self, env, Q, s, a, r, s_prime):
        
        ## Identify possible actions.
        a_prime = env.T[s_prime].indices
                
        ## Compute reward prediction error.
        theta = softmax(Q[s_prime,a_prime] * self.tau)
        delta = r + self.gamma * np.sum(Q[s_prime,a_prime] * theta) - Q[s,a]
            
        ## Update values.
        Q[s,a] += self.eta * delta
        
        return Q
            
    def train_trial(self, env, Q, start=None, n_steps=100):
        """Run a single training episode.
        
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
        
        ## Define starting state.
        if start is None: start = env.start
        assert start is not None
        s = start
        
        ## Main loop.
        for _ in np.arange(n_steps):
            
            ## Check for termination.
            if s in env.terminal: break        

            ## Select action under current policy.
            a = self._select_action(env, Q, s)
            s_prime = a
            
            ## Observe outcome.
            r = env.R[s_prime]
                        
            ## Update model.
            Q = self._update_model(env, Q, s, a, r, s_prime)
            
            ## Update state.
            s = s_prime

        return Q

    def test_trial(self, env, Q, start=False, n_steps=100):
        '''Run a single test episode.

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