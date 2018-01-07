import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from . utilities import optimal_choice

def plot_summary_table(summary, include, ax=False):
    
    ## Initialize canvas (if applicable).
    if not ax: fig, ax = plt.subplots(1,1,figsize=(12,4))

    ## Plot table.
    include = np.in1d(summary.index, include)
    sns.heatmap(summary.iloc[include], cmap=['w'], annot=True, fmt='0.2f', 
                linewidths=0.25, linecolor='k', cbar=False, ax=ax)
    ax.set_xticklabels(summary.columns, fontsize=12)
    ax.set_yticklabels(summary.index[include], fontsize=14)
    ax.xaxis.tick_top()
    ax.set_ylabel('')

    return ax

def plot_learning(data, ax=False):

    ## Initialize canvas (if applicable).
    if not ax: fig, ax = plt.subplots(1,1,figsize=(12,4))

    ## Plot predicted choices.
    sns.lineplot('Trial', 'Optimal', data=data[data.Source=='Predicted'], lw=2.5, ci=None)
    
    ## Plot observed choices.
    ix = np.logical_and(data.Source=='Observed', data.Optimal)
    ax.scatter(data.loc[ix,'Trial'], np.ones(ix.sum())*1.05, marker='x', s=20, color='k')
    
    ## Add info.
    ax.vlines([42.5, 84.5], 0.3, 1.0, color='k', zorder=10)
    ax.set(xlim=(0,127), xticks=(21,63,105), xticklabels=['Block 1', 'Block 2', 'Block 3'],
           xlabel='', ylim=(0.3, 1.1), ylabel='Optimal Choice')
    sns.despine(ax=ax)
    
    return ax

def plot_likelihood(log_lik, n_trials, ax=False):

    if not ax: fig, ax = plt.subplots(1,1,figsize=(6,6))
        
    ## Compute average likelihood per trial.
    likelihood = np.exp(log_lik / n_trials)
    
    ## Plot.
    sns.kdeplot(likelihood, lw=2.5, ax=ax)
    
    ## Add info.
    ax.set(xlabel=r'Per-Trial Likelihood', ylabel='Density')
    sns.despine(ax=ax)
    return ax