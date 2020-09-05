import numpy as np
from sisyphus.agents import ExpValSARSA, DynaQ, DynaSR
from sisyphus.environments import GraphWorld

def test_graph():
    """Test initializing and running GraphWorld."""

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Define GridWorld.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    graph = np.diag(np.ones(3), k=1)
    rewards = np.array([0,0,0,10])
    terminal, = np.where(~np.any(graph, axis=-1))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Test GridWorld.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Test initilaization.
    gw = GraphWorld(graph, rewards)

    assert gw.n_states == graph.shape[0]
    assert gw.n_viable_states == graph.shape[0] - 1
    assert np.all(gw.terminal == terminal)

    ## Test one-step transition matrix.
    T = graph.copy()
    T[terminal, terminal] = 1
    T[T.nonzero()] = np.arange(T.sum())

    assert np.all(gw.T.todense() == T)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Test ExpValSARSA.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Initialize agent / Q-matrix.
    agent = ExpValSARSA(beta=10, eta=0.5, gamma=0.99)
    Q = gw.initialize_Q()

    ## Train agent.
    for _ in np.arange(100): Q = agent.train_trial(gw, Q, start=0)

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
    for _ in np.arange(100): Q, _ = agent.train_trial(gw, Q, start=0)

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
    for _ in np.arange(100): H, w, _ = agent.train_trial(gw, H, w, start=0)

    ## Test agent.
    Q = (H @ w)[:-1]
    V = gw.R.max() * (agent.gamma ** np.arange(Q.size))
    assert np.allclose(Q, V[::-1])