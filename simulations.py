import numpy as np

def slot_machine_game(n_trials=42, probabilities=(0.2,0.4,0.6), reward=0.25):
    '''Simulates one block of the slot machine game. Based on the
    task presented in Eldar and Niv (2015).
    
    Parameters
    ----------
    n_trials : int
      number of trials in block.
    probabilities : array, shape=(n_machines,)
      probabilities of reward per slot machine.
    reward : float
      reward amount received for successful draw.
      
    Returns
    -------
    game : array, shape=(n_trials, n_machines)
      trials in one block of the game. 
      
    Notes
    -----
    Each row of the game block returned presents one comparison
    of two distinct machines. The values of each presented
    machine are in the set {loss = 0, won = reward}. Unpresented
    machines are set to NaN. 
    
    The code pseudorandomizes machine presentation order such that 
    all possible comparisons of two machines must be presented
    before the same comparison trial can be presented again.
    '''
    
    ## Sort porbabilities.
    p = np.sort(probabilities)
    
    ## Generate miniblock (contains all possible 2-machine comparisons).
    miniblock = []
    for i in np.arange(p.size-1):
        for j in np.arange(i+1, p.size):
            ix = np.in1d(p, (p[i],p[j]))
            miniblock.append( np.where(ix, p, np.nan) )
    miniblock = np.array(miniblock)
    
    ## Iteratively assemble game by stacking miniblocks. The order of
    ## trials within miniblocks are randomized.
    N = np.ceil(n_trials/miniblock.shape[0]).astype(int)
    game = np.concatenate([np.random.permutation(miniblock) for _ in range(N)], axis=0)
    game = game[:n_trials]
    
    ## Randomly generate outcomes based on machine probabilities.
    game = np.random.binomial(1, np.where(np.isnan(game), 0, game)) *\
           np.where(np.isnan(game), np.nan, reward)
    return game

def softmax(arr, beta=1):
    '''Softmax function.
    
    Parameters
    ----------
    arr : 1-d array
      array of values.
    beta : scalar
      inverse temperature parameter.
    
    Returns
    -------
    p : 1-d array
      normalized values (sum to 1).
    '''
    return np.exp(beta * arr) / np.nansum( np.exp( beta * arr ) )

class MoodyAgent(object):
    '''Class for simulated agent playing N-arm bandit task.
    
    Parameters
    ----------
    alpha : scalar
      learning parameter in the range of [0, 1].
    beta : scalar
      inverse temperature parameter.
    '''
    
    def __init__(self, beta=5, eta_v=0.1, eta_h=0.1, f=1):
        
        ## Set values.
        self._beta = float(np.copy(beta))
        self._eta_v = float(np.copy(eta_v))
        self._eta_h = float(np.copy(eta_h))
        self._f = float(np.copy(f))
        self.info = dict(beta = self._beta, eta_v=self._eta_v, eta_h=self._eta_h, f=self._f)
        
    def _setup_parameters(self, R, Q, M):
        '''Convenience function to initialize parameters for 
        single run through of the slot machine game.
        
        Parameters
        ----------
        Q : scalar, array shape=(n_arms,)
          Initial values for Q-table.
        R : array, shape=(n_trials, n_arms)
          Predetermined reward values for bandit task.
        
        Returns
        -------
        Q : array, shape=(n_arms,)
          Initial values for Q-table.
        R : array, shape=(n_trials, n_arms)
          Predetermined reward values for bandit task.
        '''
        
        ## Force to NumPy array.
        R, Q, M = [np.copy(x).astype(float) for x in [R, Q, M]]
        
        ## Initialize Q-table.
        if not np.ndim(Q):
            Q = np.ones_like(R) * Q
        elif np.ndim(Q) == 1: 
            Q = np.array([Q for _ in np.arange(R.shape[0])])
        assert Q.shape == R.shape
            
        ## Initialize mood.
        if not np.ndim(M): 
            M = np.ones(R.shape[0]) * M
        assert M.size == R.shape[0]
        
        ## Initialize choice/history terms.
        h = np.arctanh(M)
        C = np.zeros_like(h, dtype=int)

        return R, Q, M, h, C
        
    def _select_action(self, q):
        '''Action selection function. See simulate function for details.
        
        Parameters
        ----------
        q : 1-d array
          Q-values on a particular trial.
          
        Returns
        -------
        c : int
          integer, corresponding to index of action selected.
        '''
        theta = softmax(q, self._beta)
        choice = np.zeros_like(theta)
        choice[~np.isnan(theta)] = np.random.multinomial(1, theta[~np.isnan(theta)])
        return np.argmax(choice)    
        
    def simulate(self, R, Q=False, M=False):
        '''Simulate bandit task for agent. 
        
        Parameters
        ----------
        R : array, shape=(n_trials, n_arms)
          Predetermined reward values for bandit task.
        Q : array, shape=(n_arms,)
          Initial values for Q-table. If scalar, Q initialized as
          1-d array with all the same value.
        M : scalar
          Initial value for mood.
        
        Returns
        -------
        C : array, shape=(n_trials,)
          Choices (i.e. selected arm) on each trial.
        M : array, shape=(n_trials,)
          Mood on each trial
        Q : array, shape=(n_trials, n_arms)
          Final values in Q-table.

        Notes
        -----
        '''
        
        ## Initialize parameters.
        R, Q, M, h, C = self._setup_parameters(R, Q, M)
        
        ## Run bandit task.
        for i in np.arange(C.size):
            
            ## Select action.
            C[i] = self._select_action( np.where( np.isnan(R[i]), np.nan, Q[i] ) )
            
            ## Skip update if last trial.
            if i + 1 == C.size: continue

            ## Compute reward prediction error.
            delta = (self._f ** M[i]) * R[i,C[i]] - Q[i, C[i]]
            
            ## Update expectations.
            Q[i+1] = Q[i]
            Q[i+1, C[i]] = Q[i, C[i]] + self._eta_v * delta

            ## Update mood.
            h[i+1] = h[i] + self._eta_h * (delta - h[i])
            M[i+1] = np.tanh(h[i+1])
            
        return C, M, Q