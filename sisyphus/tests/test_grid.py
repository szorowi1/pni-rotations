import numpy as np
from sisyphus.agents import ExpValSARSA, DynaQ, DynaSR
from sisyphus.environments import GridWorld

def test_grid():
    """Test initializing and running GridWorld."""
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Define GridWorld.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    grid = np.array([1,1,1,1]).reshape(1,-1)
    rewards = np.array([0,0,0,10]).reshape(1,-1)
    start = 0
    terminal = 3

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Test GridWorld.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Test initilaization.
    gw = GridWorld(grid, rewards, start, terminal)

    assert np.all(np.equal(gw.shape, grid.shape))
    assert gw.n_states == grid.size
    assert gw.n_viable_states == grid.size - 1
    assert gw.start == start
    assert np.all(np.in1d(terminal, gw.terminal))

    ## Test one-step transition matrix.
    T = np.array([[0,1,0,0],
                  [1,0,1,0],
                  [0,1,0,1],
                  [0,0,0,1]])
    T[T.nonzero()] = np.arange(T.sum())

    assert np.all(gw.T.todense() == T)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Test ExpValSARSA.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Initialize agent / Q-matrix.
    agent = ExpValSARSA(beta=10, eta=0.5, gamma=0.99)
    Q = gw.initialize_Q()

    ## Train agent.
    for _ in np.arange(100): Q = agent.train_trial(gw, Q)

    ## Test agent.
    Q = Q[Q.nonzero()]
    V = gw.R.max() * (agent.gamma ** np.arange(Q.size))
    assert np.allclose(Q, V[::-1])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Test DynaQ.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Initialize agent / Q-matrix.
    agent = DynaQ(beta=10, eta=0.5, gamma=0.99, replay=1)
    Q = gw.initialize_Q()

    ## Train agent.
    for _ in np.arange(100): Q, _ = agent.train_trial(gw, Q)

    ## Test agent.
    Q = Q[Q.nonzero()]
    V = gw.R.max() * (agent.gamma ** np.arange(Q.size))
    assert np.allclose(Q, V[::-1])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Test DynaSR.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Initialize agent / Q-matrix.
    agent = DynaSR(beta=10, eta_td=0.5, eta_sr=0.5, gamma=0.99, replay=1)
    H = gw.initialize_H()
    w = np.zeros(H.shape[0])

    ## Train agent.
    for _ in np.arange(100): H, w, _ = agent.train_trial(gw, H, w)

    ## Test agent.
    Q = (H @ w)[::2]
    V = gw.R.max() * (agent.gamma ** np.arange(Q.size))
    assert np.allclose(Q, V[::-1])