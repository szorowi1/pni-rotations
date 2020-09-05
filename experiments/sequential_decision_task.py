import os, warnings
import numpy as np
from pandas import DataFrame, Series
from sisyphus.agents import DynaSR
from sisyphus.environments import GraphWorld, sequential_decision_task 
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5)

def softmax(arr):
    """Scale-robust softmax."""
    arr = np.exp(arr - np.max(arr))
    return arr / arr.sum()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define simulation parameters.
n_simulations = 50
n_training_trials = 5
n_reval_trials = 5

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Initialize agents.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Successor representation agent.
agent_sr = DynaSR(beta=1, eta_td=0.3, eta_sr=0.3, gamma=0.99, replay=False)

## Successor representation with unbiased replay.
agent_rp = DynaSR(beta=1, eta_td=0.3, eta_sr=0.3, gamma=0.99, replay=3, 
                  sampling='metric', alpha=2, outcome=False)

## Successor representation with biased replay.
agent_neg = DynaSR(beta=1, eta_td=0.3, eta_sr=0.3, gamma=0.99, replay=3, 
                   w=0., sampling='metric', alpha=2, outcome=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Preallocate space.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

DATA = DataFrame([], columns=('Model','Domain','Phase','Simulation','Trial','State','Preference'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Part 1: Reward Domain.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Initialize environments.
phase1  = GraphWorld(*sequential_decision_task(1))    # Training
phase2r = GraphWorld(*sequential_decision_task(2))    # Reward revaluation
phase2t = GraphWorld(*sequential_decision_task(3))    # Transition revaluation
domain = 'Positive'

for agent, model in zip([agent_sr, agent_rp, agent_neg], ['SR','Dyna','DynaNeg']):
    
    ## Model-free agent.
    for i in np.arange(n_simulations)+1:

        ## Initialize Q-values.
        H = phase1.initialize_H()    # SR matrix
        w = np.zeros(H.shape[0])     # Reward weights
        M = np.empty((0,7))          # Memory store

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Phase 1: Training.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        for j in np.arange(n_training_trials)+1:

            ## Training trials.
            H, w, m = agent.train_trial(phase1, H, w, M, start=0)
            H, w, m = agent.train_trial(phase1, H, w, M, start=1)

            ## Compute preference.
            theta = softmax((H @ w)[:2].data)

            ## Store data.
            series = Series([model,domain,'Training',i,j,1,theta[0]], DATA.columns)
            DATA = DATA.append(series, ignore_index=True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Phase 2r: Reward Revaluation.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        Hr, wr, Mr = H.copy(), w.copy(), M.copy()
        for j in np.arange(n_reval_trials)+1:

            ## Training trials.
            Hr, wr, Mr = agent.train_trial(phase2r, Hr, wr, Mr, start=2)
            Hr, wr, Mr = agent.train_trial(phase2r, Hr, wr, Mr, start=3)

            ## Compute preference.
            theta = softmax((Hr @ wr)[2:4].data)

            ## Store data.
            series = Series([model,domain,'Reward',i,j,3,theta[0]], DATA.columns)
            DATA = DATA.append(series, ignore_index=True)

        theta = softmax((Hr @ wr)[:2])
        series = Series([model,domain,'Reward',i,-1,1,theta[0]], DATA.columns)
        DATA = DATA.append(series, ignore_index=True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Phase 2t: Transition Revaluation.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        Ht, wt, Mt = H.copy(), w.copy(), M.copy()
        for j in np.arange(n_reval_trials)+1:

            ## Training trials.
            Ht, wt, Mt = agent.train_trial(phase2t, Ht, wt, Mt, start=2)
            Ht, wt, Mt = agent.train_trial(phase2t, Ht, wt, Mt, start=3)

            ## Compute preference.
            theta = softmax((Ht @ wt)[2:4].data)

            ## Store data.
            series = Series([model,domain,'Transition',i,j,3,theta[0]], DATA.columns)
            DATA = DATA.append(series, ignore_index=True)

        theta = softmax((Ht @ wt)[:2])
        series = Series([model,domain,'Transition',i,-1,1,theta[0]], DATA.columns)
        DATA = DATA.append(series, ignore_index=True)
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Part 2: Punishment Domain.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Initialize environments.
phase1.R *= -1
phase2r.R *= -1
phase2t.R *= -1
domain = 'Negative'

for agent, model in zip([agent_sr, agent_rp, agent_neg], ['SR','Dyna','DynaNeg']):
    
    ## Model-free agent.
    for i in np.arange(n_simulations)+1:

        ## Initialize Q-values.
        H = phase1.initialize_H()    # SR matrix
        w = np.zeros(H.shape[0])     # Reward weights
        M = np.empty((0,7))          # Memory store

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Phase 1: Training.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        for j in np.arange(n_training_trials)+1:

            ## Training trials.
            H, w, m = agent.train_trial(phase1, H, w, M, start=0)
            H, w, m = agent.train_trial(phase1, H, w, M, start=1)

            ## Compute preference.
            theta = softmax((H @ w)[:2].data)

            ## Store data.
            series = Series([model,domain,'Training',i,j,1,theta[0]], DATA.columns)
            DATA = DATA.append(series, ignore_index=True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Phase 2r: Reward Revaluation.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        Hr, wr, Mr = H.copy(), w.copy(), M.copy()
        for j in np.arange(n_reval_trials)+1:

            ## Training trials.
            Hr, wr, Mr = agent.train_trial(phase2r, Hr, wr, Mr, start=2)
            Hr, wr, Mr = agent.train_trial(phase2r, Hr, wr, Mr, start=3)

            ## Compute preference.
            theta = softmax((Hr @ wr)[2:4].data)

            ## Store data.
            series = Series([model,domain,'Reward',i,j,3,theta[0]], DATA.columns)
            DATA = DATA.append(series, ignore_index=True)

        theta = softmax((Hr @ wr)[:2])
        series = Series([model,domain,'Reward',i,-1,1,theta[0]], DATA.columns)
        DATA = DATA.append(series, ignore_index=True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Phase 2t: Transition Revaluation.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        Ht, wt, Mt = H.copy(), w.copy(), M.copy()
        for j in np.arange(n_reval_trials)+1:

            ## Training trials.
            Ht, wt, Mt = agent.train_trial(phase2t, Ht, wt, Mt, start=2)
            Ht, wt, Mt = agent.train_trial(phase2t, Ht, wt, Mt, start=3)

            ## Compute preference.
            theta = softmax((Ht @ wt)[2:4].data)

            ## Store data.
            series = Series([model,domain,'Transition',i,j,3,theta[0]], DATA.columns)
            DATA = DATA.append(series, ignore_index=True)

        theta = softmax((Ht @ wt)[:2])
        series = Series([model,domain,'Transition',i,-1,1,theta[0]], DATA.columns)
        DATA = DATA.append(series, ignore_index=True)
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save results.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define results directory.
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
if not os.path.isdir(results_dir): os.makedirs(results_dir)

## Save file.
DATA.to_csv(os.path.join(results_dir, 'sequential_decision_task.csv'), index=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Plot results.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Main loop.
for domain in ['Positive','Negative']:
    
    ## Restrict data.
    data = DATA[DATA.Domain==domain]
    
    ## Initialize canvas.
    fig = plt.figure(figsize=(12,12))
    
    ## Training.
    ax = plt.subplot2grid((3,3),(0,0),colspan=2)
    indices = np.logical_and(data.Phase=='Training', data.Trial!=-1)
    sns.lineplot('Trial','Preference','Model',data=data[indices],lw=3,alpha=0.8,ax=ax)
    ax.set(ylim=(0,1), ylabel='Preference ($S_1 > S_2$)', title='Training')
    ax.legend(loc=4, frameon=False)
    
    ## Reward revaluation.
    ax = plt.subplot2grid((3,3),(1,0),colspan=2)
    indices = np.logical_and(data.Phase=='Reward', data.Trial!=-1)
    sns.lineplot('Trial','Preference','Model',data=data[indices],lw=3,alpha=0.8,ax=ax)
    ax.set(ylim=(0,1), ylabel=r'Preference ($S_3 > S_4$)', title='Reward Revaluation')
    ax.legend(loc=4, frameon=False)
    
    ax = plt.subplot2grid((3,3),(1,2))
    indices = np.logical_and(data.Phase=='Reward', data.Trial==-1)
    sns.stripplot('Model','Preference',data=data[indices],ax=ax)
    ax.set(ylim=(0,1), ylabel='Preference ($S_1 > S_2$)')
    
    ## Reward revaluation.
    ax = plt.subplot2grid((3,3),(2,0),colspan=2)
    indices = np.logical_and(data.Phase=='Transition', data.Trial!=-1)
    sns.lineplot('Trial','Preference','Model',data=data[indices],lw=3,alpha=0.8,ax=ax)
    ax.set(ylim=(0,1), ylabel='Preference ($S_3 > S_4$)', title='Transition Revaluation')
    ax.legend(loc=4, frameon=False)
    
    ax = plt.subplot2grid((3,3),(2,2))
    indices = np.logical_and(data.Phase=='Transition', data.Trial==-1)
    sns.stripplot('Model','Preference',data=data[indices],ax=ax)
    ax.set(ylim=(0,1), ylabel='Preference ($S_1 > S_2$)')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'sdt_%s.png' %domain), dpi=180)
    plt.close('all')