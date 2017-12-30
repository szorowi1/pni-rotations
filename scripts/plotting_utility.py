import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stan_utility import check_div, check_energy, check_treedepth

def optimal_choice(Y, X):
    return np.array([np.argmax(x)+1==y for x,y in zip(X,Y)]).astype(int)

def plot_summary_table(fit, include=[], ax=False):
    
    if not ax: fig, ax = plt.subplots(1,1,figsize=(12,4))
    
    ## Extract data summary.
    summary = fit.summary()
    data = summary['summary']
    xticklabels = summary['summary_colnames']
    yticklabels = summary['summary_rownames']
    
    ## Restrict data (if applicable).
    if np.array(include).size:
        include = np.in1d(yticklabels, include)
        data = data[include]
        yticklabels = yticklabels[include]
        
    ## Plot table.
    sns.heatmap(data, cmap=['w'], annot=True, fmt='0.2f', linewidths=0.25, linecolor='k',
                xticklabels=xticklabels, yticklabels=yticklabels, cbar=False, ax=ax)
    ax.xaxis.tick_top()
    
    ## Add diagnostic information.
    n_samples = fit.extract('log_lik')['log_lik'].size
    msg = '    '.join(['N samples = %s' %n_samples,
                       'Divergence = %0.0f%%' %check_div(fit), 
                       'Treedepth = %0.0f%%' %check_treedepth(fit),
                       'E-BFMI Warning = %s' %check_energy(fit)])
    ax.annotate(msg, (0,0), (0.5,-0.1), xycoords='axes fraction', 
                va='center', ha='center', fontsize=14)

    return ax

def plot_learning(OC, OC_pred, B, ax=False):
    
    if not ax: fig, ax = plt.subplots(1,1,figsize=(12,4))
        
    ## Compute trial information.
    x = np.arange(OC.size) + 1
    boundaries = np.array([np.argmax(B==b) for b in np.unique(B)])
    boundaries = boundaries[1:] + 1
    
    ## Plot predicted optimal choice.
    for b in np.unique(B):
        ax.plot(x[B==b], OC_pred[B==b], lw=2.5, color="#34495e")
        
    ## Plot observed optimal choice.
    colors = ['#1f77b4','#d62728']
    labels = ['Optimal','Nonoptimal']
    for oc, color, label in zip([1,0], colors, labels):
        ax.scatter(x, np.where(OC==oc, 1.025, np.nan), s=25, color=color, label=label )
    
    ## Add info.
    ymin, _ = ax.get_ylim()
    ax.vlines(boundaries-0.5, ymin, 1, lw=0.25)
    ax.set(xlim=(x.min() - 0.5, x.max() + 0.5), ylim=(ymin, 1.05), ylabel='Optimal Choice')
    ax.legend(loc=4, markerscale=1.5, borderpad=0, labelspacing=0.2, handletextpad=0.2)
    sns.despine(ax=ax)
    return ax

def kde_plot(fit, params, labels=False, title=False, cmap=None, ax=False):
    
    if not ax: fig, ax = plt.subplots(1,1,figsize=(6,6))
    if not labels: labels = params
    
    ## Extract data.
    data = fit.extract(params)
    data = np.array([data[param] for param in params]).squeeze()

    ## Plot.
    sns.kdeplot(*data, legend=False, cmap=cmap, ax=ax)
    
    ## Add info.
    if isinstance(labels, str): ax.set_xlabel(labels)
    elif len(labels) < 2: ax.set_xlabel(labels[0])
    else: ax.set(xlabel=labels[0], ylabel=labels[1])
    if title: ax.set_title(title)
    sns.despine(ax=ax)
    
    return ax

def plot_likelihood(fit, ax=False):

    if not ax: fig, ax = plt.subplots(1,1,figsize=(6,6))
        
    ## Extract and compute average likelihood per trial.
    log_lik = fit.extract('log_lik')['log_lik']
    log_lik /= fit.data['Y'].size
    likelihood = np.exp(log_lik)
    
    ## Plot.
    sns.kdeplot(likelihood, lw=2.5, ax=ax)
    
    ## Add info.
    ax.set(xlabel=r'Per-Trial Likelihood', ylabel='Posterior Density')
    sns.despine(ax=ax)
    return ax

def plot_toy_model(f, fit):

    ## Initialize canvas.
    fig = plt.figure(figsize=(12,9))
    sns.set_context('notebook', font_scale=1.5)

    ## Extract information.
    X = fit.data['X']
    Y = fit.data['Y']
    R = fit.data['R']
    B = fit.data['B']
    Y_pred = fit.extract('Y_pred')['Y_pred']

    ## Compute optimal choices.
    OC = optimal_choice(Y, X)
    OC_pred = np.apply_along_axis(optimal_choice, 1, Y_pred, X)

    ## Plot results table.
    ax = plt.subplot2grid((3,3),(0,0),colspan=3)
    include = ['beta_pr', 'eta_v_pr', 'beta', 'eta_v', 'lp__']
    plot_summary_table(fit, include=include, ax=ax)

    ## Plot learning.
    ax = plt.subplot2grid((3,3),(1,0),colspan=3)
    plot_learning(OC, OC_pred.mean(axis=0), B, ax=ax)

    ## Plot KDE of beta vs. eta.
    ax = plt.subplot2grid((3,3),(2,0))
    kde_plot(fit, ['beta','eta_v'], cmap=sns.dark_palette('skyblue', as_cmap=True),
             labels=[r'Inverse Temperature ($\beta$)', r'Learning Rate ($\eta$)'], ax=ax)

    ## Plot KDE of optimal choice.
    ax = plt.subplot2grid((3,3),(2,1))
    sns.kdeplot(OC_pred.mean(axis=1), lw=2.5, ax=ax)
    ax.vlines(OC.mean(), *ax.get_ylim(), linestyle='--', alpha=0.8)
    ax.set(xlabel='Optimal Choice', ylabel='Posterior Density')
    sns.despine(ax=ax)

    ## Plot average likelihood.
    ax = plt.subplot2grid((3,3),(2,2))
    plot_likelihood(fit, ax=ax)

    ## Save figure.
    plt.tight_layout()
    plt.savefig(f, dpi=180)
    plt.close('all')