import warnings
import numpy as np
from itertools import combinations
from pandas import Panel

## Need to find better fix for Panel at some point.
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def DataFrame3d(arr, items=None, major_axis=None, minor_axis=None, copy=False):
    df = Panel(arr, items=items, major_axis=major_axis, 
               minor_axis=minor_axis, copy=copy).to_frame()
    return df.reset_index().melt(id_vars=['major','minor'])

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
    X : array, shape=(n_trials, 2)
      presented machines for each trial.
    R : array. shape=(n_trials, 2)
      outcome for choosing a machine.
      
    Notes
    -----
    Each row of the trials block, X, represents one comparison
    of two distinct machines. The values of X are integers in 
    the range of [0, n_machines]. Each row of the rewards
    array, R, represents the predetermined reward if the agent
    chooses that machine. The value of R are floats in the 
    range of [0, reward].
    '''
    
   ## Sort porbabilities.
    p = np.sort(probabilities)

    ## Generate trial types.
    x = np.array(list(combinations(np.arange(p.size), 2)))

    ## Iteratively assemble game by stacking the trial types. 
    N = np.ceil(n_trials/x.shape[0]).astype(int)
    X = np.concatenate([x for _ in np.arange(N)])
    for _ in np.arange(1000): np.random.shuffle(X)

    ## Randomly generate outcomes based on machine probabilities.
    R = np.random.binomial(1, p[X]) * reward
    
    return X, R

def optimal_choice(Y, X):
    '''Check if optimal decision made per trial.
    
    Parameters
    ----------
    Y : array, shape=(n_trials,)
      choices made.
    X : array, shape=(n_trials, n_machines)
      the design matrix.
      
    Returns
    -------
    OC : array, shape=(n_trials,)
      integer array: 1 if optimal choice made, otherwise 0.
    '''
    return np.array([np.argmax(x)==y for x,y in zip(X,Y)]).astype(int)


def cumulative_reward(Y, R):
    '''Compute cumulative reward.
    
    Parameters
    ----------
    Y : array, shape=(n_trials,)
      choices made.
    R : array, shape=(n_trials, n_machines)
      predetermined rewards for block.
      
    Returns
    -------
    cumsum : array, shape=(n_trials,)
      Cumulative reward over trials.
    '''
    
    return np.cumsum([R[i,j] for i,j in enumerate(Y)])

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
        
    def _setup_parameters(self, X, Q, M):
        '''Convenience function to initialize parameters for 
        single run through of the slot machine game.
        
        Parameters
        ----------
        X : array, shape=(n_trials, 2)
          Predetermined machine presentation order for bandit task.
        Q : float or array, shape=(n_machines)
          Initial values for Q-table.
        M : float
          Initial value for mood.
        
        Returns
        -------
        Y : array, shape=(n_trials)
          Preallocated space for agent choices.
        Q : array, shape=(n_machines,)
          Initial values for Q-table.
        M : array, shape=(n_trials)
          Initial values for mood.
        h : array, shape=(n_trials)
          Preallocated space for history effects.
        '''
        
        ## Force to NumPy array.
        Q, M = [np.copy(x).astype(float) for x in [Q, M]]
        
        ## Initialize Q-table.
        if not np.ndim(Q):
            Q = np.ones(X.max()+1) * Q
        assert Q.size == X.max()+1
            
        ## Initialize mood.
        M = np.ones(X.shape[0]) * M
        assert M.size == X.shape[0]
        
        ## Initialize choice/history terms.
        h = np.arctanh(M)
        Y = np.zeros_like(h, dtype=int)

        return Y, Q, M, h
        
    def _select_action(self, q):
        '''Action selection function. See simulate function for details.
        
        Parameters
        ----------
        q : 1-d array
          Q-values on a particular trial.
          
        Returns
        -------
        y : int
          integer, corresponding to index of action selected.
        '''
        theta = softmax(q, self._beta)
        choice = np.random.multinomial(1, theta)
        return np.argmax(choice)    
        
    def simulate(self, X, R, Q=False, M=False):
        '''Simulate bandit task for agent. 
        
        Parameters
        ----------
        X : array, shape=(n_trials, 2)
          Predetermined machine presentation order for bandit task.
        R : array, shape=(n_trials, 2)
          Predetermined reward values for bandit task.
        Q : array, shape=(n_machines,)
          Initial values for Q-table. If scalar, Q initialized as
          1-d array with all the same value.
        M : scalar
          Initial value for mood.
        
        Returns
        -------
        Y : array, shape=(n_trials,)
          Choices on each trial.
        M : array, shape=(n_trials,)
          Mood on each trial

        Notes
        -----
        '''
        
        ## Initialize parameters.
        Y, Q, M, h = self._setup_parameters(X, Q, M)
        
        ## Run bandit task.
        for i in np.arange(Y.size):
            
            ## Select action.
            Y[i] = self._select_action( Q[X[i]] )
            
            ## Skip update if last trial.
            if i + 1 == Y.size: continue

            ## Compute reward prediction error.
            delta = (self._f ** M[i]) * R[i,Y[i]] - Q[X[i, Y[i]]]
            
            ## Update expectations.
            Q[X[i, Y[i]]] += self._eta_v * delta

            ## Update mood.
            h[i+1] = h[i] + self._eta_h * (delta - h[i])
            M[i+1] = np.tanh(h[i+1])
            
        return Y, M
    
class DualLearningAgent(object):
    '''Slot machine game-playing agent with two learning rates (positive, negative).
    
    Parameters
    ----------
    beta : scalar
      Inverse temperature parameter.
    eta_p : scalar
      Learning rate, positive RPE.
    eta_n : scalar
      Learning rate, negative RPE.
    '''
    
    def __init__(self, beta=5, eta_p=0.1, eta_n=0.1):
        
        ## Set values.
        self._beta = float(np.copy(beta))
        self._eta_p = float(np.copy(eta_p))
        self._eta_n = float(np.copy(eta_n))
        self.info = dict(beta = self._beta, eta_p=self._eta_p, eta_n=self._eta_n)
        
    def _setup_parameters(self, X, Q):
        '''Convenience function to initialize parameters for 
        single run through of the slot machine game.
        
        Parameters
        ----------
        X : array, shape=(n_trials, 2)
          Predetermined machine presentation order for bandit task.
        Q : float or array, shape=(n_machines)
          Initial values for Q-table.
        
        Returns
        -------
        Y : array, shape=(n_trials)
          Preallocated space for agent choices.
        Q : array, shape=(n_machines,)
          Initial values for Q-table.
        '''   
        
        ## Initialize Q-table.
        if not np.ndim(Q):
            Q = np.ones(X.max()+1) * Q
        assert Q.size == X.max()+1
        
        ## Initialize choice.
        Y = np.zeros(X.shape[0], dtype=int)

        return Y, Q
        
    def _select_action(self, q):
        '''Action selection function. See simulate function for details.
        
        Parameters
        ----------
        q : 1-d array
          Q-values on a particular trial.
          
        Returns
        -------
        y : int
          integer, corresponding to index of action selected.
        '''
        theta = softmax(q, self._beta)
        choice = np.random.multinomial(1, theta)
        return np.argmax(choice)    
        
    def simulate(self, X, R, Q=False):
        '''Simulate bandit task for agent. 
        
        Parameters
        ----------
        X : array, shape=(n_trials, 2)
          Predetermined machine presentation order for bandit task.
        R : array, shape=(n_trials, 2)
          Predetermined reward values for bandit task.
        Q : array, shape=(n_machines,)
          Initial values for Q-table. If scalar, Q initialized as
          1-d array with all the same value.
        
        Returns
        -------
        Y : array, shape=(n_trials,)
          Choices on each trial.
        '''
        
        ## Initialize parameters.
        Y, Q = self._setup_parameters(X, Q)
        
        ## Run bandit task.
        for i in np.arange(Y.size):
            
            ## Select action.
            Y[i] = self._select_action( Q[X[i]] )

            ## Compute reward prediction error.
            delta = R[i,Y[i]] - Q[X[i, Y[i]]]
            
            ## Update expectations. 
            if delta > 0:
                Q[X[i, Y[i]]] += self._eta_p * delta
            else:
                Q[X[i, Y[i]]] += self._eta_n * delta 

        return Y