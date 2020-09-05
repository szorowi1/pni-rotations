import numpy as np
from pandas import DataFrame

def experiment_cliff_walking(agent, cliff, n_agents=50, train_episodes=1000, 
                             train_steps = 50, test_points = [10,50,100,250,500,1000], 
                             test_episodes = 100, test_steps = 50):
    
    ## Preallocate space.
    paths = np.zeros((len(test_points), cliff.n_states))
    df = DataFrame([], columns=('Simulation','Episode','Iter','Reward','Safe','Cliff'))

    ## Main loop.
    for i in np.arange(n_agents)+1:

        ## Initialize Q-values.
        Q = cliff.initialize_Q()

        for j in np.arange(train_episodes)+1:

            ## Training.
            Q, _, _ = agent.train_trial(cliff, Q, n_steps=train_steps)

            ## Testing.
            if j in test_points:

                for k in np.arange(test_episodes)+1:

                    ## Run test trials.
                    states, cumsum = agent.test_trial(cliff, Q, n_steps=test_steps)

                    ## Append info.
                    s = cliff.terminal[-1] == states[-1]
                    c = states[-1] in cliff.terminal[:-1] 
                    df = df.append(Series([i,j,k,cumsum,s,c], index=df.columns), ignore_index=True)

                    states, counts = np.unique(states, return_counts=True)
                    paths[test_points.index(j),states] += counts

    ## Normalize path frequency.
    paths = np.divide(paths.T, paths.sum(axis=-1)).T
    
    ## Fix DataFrame dtypes.
    df[['Simulation','Episode','Iter']] = df[['Simulation','Episode','Iter']].astype(int)
    df[['Safe','Cliff']] = df[['Safe','Cliff']].astype(bool)
    
    return df, paths