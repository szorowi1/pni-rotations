import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

def cliff_walking():
    """Returns grid information for the Cliffwalking environment.
    
    Returns
    -------
    grid : array, shape = (11,12)
        States in Cliffwalking environment.
    rewards : array, shape = (11,12)
        Rewards on transitioning to each state.
    start : int
        Starting state.
    terminal : array, shape = (11)
        Terminal states
    """
    
    ## Define gridworld.
    grid = np.ones((11,12), dtype=int)
    
    ## Define start/terminal states.
    start = 120
    terminal = np.array([121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131])
    
    ## Define rewards.
    rewards = np.ones(grid.size) * -1
    rewards[terminal[:-1]] = -100
    rewards[terminal[-1]] = 0
    rewards = rewards.reshape(grid.shape)
    
    return grid, rewards, start, terminal

def two_step_task():
    """Returns grid information for the two-step task environment.
    
    Returns
    -------
    grid : array, shape = (6,11)
        States in two-step task environment.
    rewards : array, shape = (6,11)
        Rewards on transitioning to each state.
    start : int
        Starting state.
    terminal : array, shape = (4)
        Terminal states
    """
            
    ## Define gridworld.
    grid = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
        
    ## Define start/stop information.     
    start = 60
    terminal = np.array([0, 4, 6, 10])

    ## Define rewards.
    rewards = np.ones(grid.size) * -1
    rewards[0] = 50
    rewards[4] = -100
    rewards[6] = 10
    rewards[10] = 0
    rewards = rewards.reshape(grid.shape)
    
    return grid, rewards, start, terminal

class GridWorld(object):
    """Generate gridworld environment.
    
    Parameters
    ----------
    grid : array, shape = (n,m)
        Binary array where 1 denoting occupiable states and 0 otherwise.
    rewards : array, shape = (n,m)
        Array denoting the reward for transitioning into corresponding state.
    start : int
        Starting state (default = None).
    terminal : list or array
        Terminal states (default = None).
    
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
    shape : tuple
        Size of gridworld.
    R : array
        State-transition rewards.
    T : sparse CSR matrix
        One-step transition matrix.
    """
    
    def __init__(self, grid, rewards, start=None, terminal=None):
        
        ## Define metadata.
        self.shape = grid.shape
        
        ## Define rewards.
        self.R = rewards.flatten()
        
        ## Define start / terminal states.
        if terminal is None: self.terminal = []
        elif isinstance(terminal, int): self.terminal = [terminal]
        else: terminal = self.terminal = np.array(terminal)
        self.start = start
            
        ## Define state information.
        self.states = np.arange(grid.size).reshape(self.shape)
        self.n_states = self.states.size
        
        viable_states = np.logical_xor(grid.flatten(), np.isin(self.states.flatten(), self.terminal))
        self.viable_states = np.argwhere(viable_states).squeeze()
        self.n_viable_states = self.viable_states.size
        
        ## Define one-step transition matrix.
        self.T = self._one_step_transition_matrix() 
        
    def _one_step_transition_matrix(self):
        """Returns the sparse CSR one-step transition matrix."""

        ## Define grid coordinates.
        nx, ny = self.shape
        rr = np.array(np.meshgrid(np.arange(nx),np.arange(ny)))
        rr = rr.reshape(2,np.product(self.shape),order='F').T

        ## Compute distances between states. Binarize.
        T = (cdist(rr,rr)==1).astype(int)
        
        ## Mask terminal states.
        T[self.terminal] = 0
        T[self.terminal, self.terminal] = 1
        
        ## Convert to sparse CSR matrix.
        data = np.arange(T.sum())
        row, col = T.nonzero()        
        return csr_matrix((data, (row,col)), T.shape, dtype=int)

    def initialize_Q(self, init=0):
        """Returns an initial Q-table."""
        return np.ones(self.T.shape) * float(init)

    def initialize_H(self):
        """Returns an intial state-action successor representation matrix."""
        H = np.identity(self.T.data.size)
        H[self.T[self.terminal].data] = 0
        return H