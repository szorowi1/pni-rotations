import numpy as np
from scipy.sparse import csr_matrix

def sequential_decision_task(cond):
    """Sequential decision task paradigm.
    
    Parameters
    ----------
    cond : int
        Condition of experiment (see notes).
    
    Returns
    -------
    graph : array, shape = (6,6)
        States in SDT environment.
    rewards : array, shape = (6,6)
        Rewards on transitioning to each state.
    
    Notes
    -----
    1. Training
    2. Reward revaluation
    3. Transition revaluation
    
    References
    ----------
    Momennejad, I., Russek, E. M., Cheong, J. H., Botvinick, M. M., Daw, N. D., & Gershman, S. J. (2017). 
    The successor representation in human reinforcement learning. Nature Human Behaviour, 1(9), 680.
    """
    
    ## Training.
    if cond == 1:
        
        ## Define graphworld.
        graph = np.zeros((8,8),dtype=int)
        graph[np.arange(6),[2,3,4,5,6,7]] = 1

        ## Define rewards.
        rewards = np.array([0,0,0,0,0,0,10,1], dtype=float)
        
    ## Reward revaluation.
    elif cond == 2:
        
        ## Define graphworld.
        graph = np.zeros((8,8),dtype=int)
        graph[np.arange(6),[2,3,4,5,6,7]] = 1

        ## Define rewards.
        rewards = np.array([0,0,0,0,0,0,1,10], dtype=float)
        
    ## Transition revaluation.
    elif cond == 3:
        
        ## Define graphworld.
        graph = np.zeros((8,8),dtype=int)
        graph[np.arange(6),[2,3,5,4,6,7]] = 1

        ## Define rewards.
        rewards = np.array([0,0,0,0,0,0,10,1], dtype=float)
        
    return graph, rewards
                
class GraphWorld(object):
    """Generate graphworld environment.
    
    Parameters
    ----------
    graph : array, shape = (n,n)
        Graph adjacency matrix. Terminal states should be all-zero.
    rewards : array, shape = (n,)
        Array denoting the reward for transitioning into corresponding state.
    start : int
        Starting state (default = None).
    
    Attributes
    ----------
    states : array
        List of state indices.
    n_states : int
        Number of unique states.
    viable_states : array
        List of viable state indices
    n_viable_states : int
        Number of viable states.
    R : array
        State-transition rewards.
    T : sparse CSR matrix
        One-step transition matrix.
    """
    
    def __init__(self, graph, rewards, start=None):
            
        ## Error catching: compare graph and rewards.
        assert np.all(np.in1d(graph, [0,1]))
            
        ## Define rewards.
        self.R = rewards
        
        ## Define start / terminal states.
        self.start = start
        self.terminal, = np.where(~np.any(graph, axis=-1))
            
        ## Define state information.
        self.states = np.arange(graph.shape[0])
        self.n_states = self.states.size
        
        self.viable_states, = np.where(np.any(graph, axis=-1))
        self.n_viable_states = self.viable_states.size
        
        ## Define one-step transition matrix.
        self.T = self._one_step_transition_matrix(graph) 
        
    def _one_step_transition_matrix(self, graph):
        
        ## Add self-absorption for terminal states.
        graph = graph.copy()
        graph[self.terminal,self.terminal] = 1
        
        ## Convert to sparse CSR matrix.
        data = np.arange(graph.sum())
        row, col = graph.nonzero()        
        return csr_matrix((data, (row,col)), graph.shape, dtype=int)

    def initialize_Q(self, init=0):
        """Returns an initial Q-table."""
        return np.ones(self.T.shape) * float(init)

    def initialize_H(self):
        """Returns an intial state-action successor representation matrix."""
        H = np.identity(self.T.data.size)
        H[self.terminal] = 0
        return H