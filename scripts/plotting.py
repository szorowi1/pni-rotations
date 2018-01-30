import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import _pickle as cPickle
import warnings
from . utilities import load_fit, HDIofMCMC, WAIC
warnings.filterwarnings("ignore", category=RuntimeWarning) 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Subject plots.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def plot_subject_behavior(Y, Yhat, color='#1f77b4', ax=False):
    
    if not ax: fig, ax = plt.subplots(1,1,figsize=(12,4))
        
    ## Compute average.
    Yhat = np.nanmean(Yhat, axis=0)
    n_blocks, n_trials = Y.shape
    
    ## Plot.
    for b in np.arange(n_blocks):
        
        ## Define trial numbers.
        trials = np.arange(n_trials) + b * n_trials
        trials += 1

        ## Plot.
        ax.scatter( trials, Y[b]*1.05, s=20, marker='o', color='k' )
        ax.plot( trials, Yhat[b], lw=2.5, color=color )
        
    ## Add info.
    ax.vlines([42.5, 84.5], 0, 1, lw=1.5, color='k', zorder=10)
    ax.set(xlim=(0.5, 126), xticks=[7,21,35,49,63,77,91,105,119], xlabel='Trial', 
           ylim=(0.3, 1.1), yticks=np.arange(0.4,1.1,0.2), ylabel='Optimal Choice')
    ax.grid(axis='y', alpha=0.25, zorder=0)
    
    return ax

def plot_subject_mood(M, Mhat, color='#1f77b4', ax=False):

    if not ax: fig, ax = plt.subplots(1,1,figsize=(12,4))
    _, n_blocks, n_trials = Mhat.shape
    
    ## Plot.
    for b in np.arange(n_blocks):

        ## Plot median.
        trials = np.arange(n_trials) + b * n_trials + 1                
        ax.plot(trials, np.median(Mhat, axis=0)[b], lw=2.5, color=color)
        
        ## Plot HDI.
        lb, ub = np.apply_along_axis(HDIofMCMC, 0, Mhat[:,b])
        ax.fill_between(trials, lb, ub, color=color, alpha=0.1)
        
        ## Plot observed data.
        trials = np.array([7,21,35]) + b * n_trials
        ax.scatter(trials, M[b], s=150, marker='d',color='k', zorder=10)
        
    ## Add info.
    ax.hlines(0, 0, n_trials*n_blocks+1, linestyle='--', alpha=0.1, zorder=0)
    ax.vlines([42.5, 84.5], -1, 1, lw=1.5, color='k', zorder=10)
    ax.set(xlim=(0.5, 126), xticks=np.arange(7,126,14), xlabel='Trial', ylim=(-1,1), ylabel='Mood')
        
    return ax

def plot_rl_params(beta, eta_v, cmap='Blues', color='w', hdi=0.95, ds=1, ax=False):
    
    if not ax: fig, ax = plt.subplots(1,1,figsize=(6,6))
    
    ## Plot.
    sns.kdeplot(beta, eta_v, cmap=cmap, shade=True, shade_lowest=False, ax=ax)
    ax.scatter(beta[::ds], eta_v[::ds], s=20, marker='+', color=color)
    ax.set(xlim=HDIofMCMC(beta, hdi), xlabel=r'$\beta$', 
           ylim=HDIofMCMC(eta_v, hdi), ylabel=r'$\eta_v$')
    
    return ax

def plot_bias_params(eta_h, f=False, cmap='Blues', color='#1f77b4', hdi=0.95, ds=1, ax=False):
    
    if not ax: fig, ax = plt.subplots(1,1,figsize=(6,6))
    
    if not np.any(f):
        
        sns.distplot(eta_h, kde=False, color=color, hist_kws=dict(alpha=0.9, edgecolor='w'), ax=ax)
        ax.set(xlabel=r'$\eta_h$', ylabel='Count')
        
    else:
        sns.kdeplot(eta_h, f, cmap=cmap, shade=True, shade_lowest=False, ax=ax)
        ax.scatter(eta_h[::ds], f[::ds], s=20, marker='+', color='w')
        ax.set(xlim=HDIofMCMC(eta_h, hdi), xlabel=r'$\eta_h$', 
               ylim=HDIofMCMC(f, hdi), ylabel=r'$f$')
    
    return ax

def plot_subject(fit, ix, cmap='Blues', color='#1f77b4', hdi=0.975, ds=1, figsize=(12,8)):

    sns.set_context('notebook', font_scale=1.25)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Prepare data.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Extract data.
    X = fit['X'][ix]
    Y = fit['Y'][ix]
    M = fit['M'][ix]
    Yhat = fit['Y_pred'][:,ix]
    h_pred = (fit['h_pred'][:,ix].T + fit['beta_h'][:,ix].T).T
    Mhat = np.tanh(h_pred)
    
    ## Mask missing data.
    missing = Y < 0
    Y = np.where(missing, np.nan, Y)
    Yhat = np.array([np.where(missing, np.nan, sample) for sample in Yhat])
        
    ## Define choices as optimal or not.
    optimal_choice = np.argmax(X, axis=-1)
    Y = np.equal(Y-1, optimal_choice).astype(int)
    Yhat = np.array([np.equal(sample, optimal_choice) for sample in Yhat-1]).astype(int)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Plotting.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ## Initialize canvas.
    fig = plt.figure(figsize=figsize)
    axes = []
    
    ## Plot posterior predictive checks.
    gs = gridspec.GridSpec(2, 1)
    gs.update(left=0.075, right=0.975, bottom=0.45,  top=0.975, hspace=0.01)
    
    axes.append(plt.subplot(gs[0]))
    plot_subject_behavior(Y, Yhat, color=color, ax=axes[-1])
    axes[-1].set(xticks=[], xlabel='')
    
    axes.append(plt.subplot(gs[1]))
    plot_subject_mood(M, Mhat, color=color, ax=axes[-1])

    ## Plot distributions.
    gs = gridspec.GridSpec(1, 3)
    gs.update(left=0.075, right=0.975, bottom=0.1,  top=0.35, hspace=0.05)

    axes.append(plt.subplot(gs[0]))
    plot_rl_params(fit['beta'][:,ix], fit['eta_v'][:,ix], cmap=cmap, hdi=hdi, ds=10, ax=axes[-1])
  
    if 'f' in fit.keys():
        axes.append(plt.subplot(gs[1]))
        plot_bias_params(fit['eta_h'][:,0], fit['f'][:,0], cmap=cmap, hdi=hdi, ds=10, ax=axes[-1])
    elif 'eta_h' in fit.keys():
        axes.append(plt.subplot(gs[1]))
        plot_bias_params(fit['eta_h'][:,0], cmap=cmap, hdi=hdi, ds=10, ax=axes[-1])    
    
    if 'beta_h' in fit.keys():
        axes.append(plt.subplot(gs[2]))
        sns.distplot(np.tanh(fit['beta_h'][:,ix]), kde=False, color=color, 
                     hist_kws=dict(alpha=0.9, edgecolor='w'), ax=axes[-1])
        axes[-1].set(xlabel=r'$\hat{M}_\mu$', ylabel='Count')
    
    sns.despine()
    
    return fig, axes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Group plots.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def plot_group_behavior(model, subject=None, observed=False, color=None, label=None,
                        fit_dir='stan_fits', ax=False):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Prepare behavior.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ## Open StanFit.
    f = '%s/%s/StanFit.pickle' %(fit_dir, model)
    with open(f, 'rb') as f: fit = cPickle.load(f)

    ## Extract data and compute optimal choice.
    optimal_choice = np.argmax(fit['X'], axis=-1)
    Y_obs = np.equal(fit['Y']-1, optimal_choice).astype(int)
    Y_pred = np.array([np.equal(sample, optimal_choice) for sample in fit['Y_pred']-1]).astype(int)

    ## Mask missing data.
    missing = fit['Y'] < 0
    Y_obs = np.where(missing, np.nan, Y_obs)
    Y_pred = np.array([np.where(missing, np.nan, sample) for sample in Y_pred])

    ## Compute average over group or subject.
    if subject is None:
        Y_obs = np.nanmean(Y_obs, axis=0)
        Y_pred = np.apply_over_axes(np.nanmean, Y_pred, [0,1]).squeeze()
    else:
        Y_obs = Y_obs[subject]
        Y_pred = np.nanmean(Y_pred, axis=0)[subject]
        
    n_blocks, n_trials = Y_obs.shape
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Plotting.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    if not ax: fig, ax = plt.subplots(1,1,figsize=(12,4))
    
    for block in np.arange(n_blocks):

        ## Define trial numbers.
        trials = np.arange(n_trials) + block * n_trials
        trials += 1

        ## Plot.
        if observed: ax.plot( trials, Y_obs[block], lw=1.5, color='k', alpha=0.8, zorder=0 )
        ax.plot( trials, Y_pred[block], lw=2.5, label=label, color=color, alpha=0.8 )
        label = None

    ## Add info.
    ax.vlines([42.5, 84.5], 0, 1, lw=1.5, color='k', zorder=10)
    ax.set(xlim=(0.5,126), xticks=np.arange(7,126,14), xlabel='Trial',
             ylim=(0.35, 1.01), ylabel='Optimal Choice')
    
    return ax

def scale_bar_yaxis(ax):
    
    ## Get bar heights.
    y = [patch.get_height() for patch in ax.patches]
    
    ## Set ymin.
    if min(y) < 0: ymin = min(y) - 50
    else: ymin = max([min(y) - 50, 0])
        
    ## Set ymax.
    if max(y) < 0: ymax = 0
    else: ymax = max(y) + 50
    
    return ymin, ymax

def plot_group_comparison(model_names, labels, palette=None, show_behavior=True, show_mood=True, show_psis=True):
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Setup figure.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ## Define figure dimensions.
    nrow = sum([show_behavior, show_mood, show_psis])
    ncol = 3

    ## Initialize canvas.
    fig = plt.figure(figsize=(12,nrow*3))
    sns.set_context('notebook', font_scale=1.5)
    if palette is None: palette = sns.color_palette(n_colors=len(model_names))
        
    ## Initialize plots.
    i = 0
    axes = []
    if show_behavior: 
        ax_b = plt.subplot2grid((nrow,ncol),(i,0),colspan=ncol)
        axes.append(ax_b)
        i += 1
    if show_mood:
        ax_m = plt.subplot2grid((nrow,ncol),(i,0),colspan=ncol)
        axes.append(ax_m)
        i += 1
    if show_psis:
        ax_p1 = plt.subplot2grid((nrow,ncol),(i,0))
        ax_p2 = plt.subplot2grid((nrow,ncol),(i,1))
        ax_p3 = plt.subplot2grid((nrow,ncol),(i,2))
        axes += [ax_p1, ax_p2, ax_p3]
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Main loop.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    for i, model_name in enumerate(model_names):
    
        ## Load StanFit file.
        f = 'stan_fits/%s/StanFit.pickle' %model_name
        with open(f, 'rb') as f: extract = cPickle.load(f)

        if show_behavior:
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            ### Plot group-level performance.
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            ## Extract data and compute optimal choice.
            optimal_choice = np.argmax(extract['X'], axis=-1)
            Y_obs = np.equal(extract['Y']-1, optimal_choice).astype(int)
            Y_pred = np.array([np.equal(sample, optimal_choice) for sample in extract['Y_pred']-1]).astype(int)

            ## Mask missing data.
            missing = extract['Y'] < 0
            Y_obs = np.where(missing, np.nan, Y_obs)
            Y_pred = np.array([np.where(missing, np.nan, sample) for sample in Y_pred])

            ## Compute average over subjects.
            Y_obs = np.nanmean(Y_obs, axis=0)
            Y_pred = np.apply_over_axes(np.nanmean, Y_pred, [0,1]).squeeze()
            n_blocks, n_trials = Y_obs.shape

            for block in np.arange(n_blocks):

                ## Define trial numbers.
                trials = np.arange(n_trials) + block * n_trials
                trials += 1

                ## Plot.
                if not i: ax_b.plot( trials, Y_obs[block], lw=1.5, color='k', alpha=0.8 )
                ax_b.plot( trials, Y_pred[block], lw=2.5, label=labels[i], color=palette[i], alpha=0.8 )

            ## Add info.
            ax_b.vlines([42.5, 84.5], 0, 1, lw=1.5, color='k', zorder=10)
            ax_b.set(xlim=(0.5,126), xticks=np.arange(7,126,14), xlabel='Trial',
                     ylim=(0.35, 1.01), ylabel='Optimal Choice')
    
        if show_mood:
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            ### Plot group-level mood change.
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            ## Extract mood data.
            M_obs = extract['M']
            M_pred = np.tanh( np.median(extract['h_pred'], axis=0) )

            ## Compute average across subjects by WoF outcome.
            WoF = extract['WoF']
            M_obs = np.array([M_obs[np.sign(WoF) == v].mean(axis=0) for v in [1,-1]])
            M_pred = np.array([M_pred[np.sign(WoF) == v].mean(axis=0) for v in [1,-1]])
            n_outcome, n_blocks, n_trials = M_pred.shape
            
            for block in np.arange(n_blocks):

                ## Define trial numbers.
                trials = np.array([7,21,35]) + block * n_trials
                trials = np.arange(n_trials) + block * n_trials + 1

                ## Plot.
                for outcome, style in enumerate(['-','--']):

                    if not i: ax_m.scatter(trials[[6,20,34]], M_obs[outcome, block], s=150,
                                         marker='d', color='k', zorder=10)
                    ax_m.plot(trials, M_pred[outcome, block], lw=2.5, linestyle=style,
                              label=labels[i], color=palette[i], alpha=0.8)

            ## Add info.
            ax_m.vlines([42.5, 84.5], -3, 3, lw=1.5, color='k', zorder=10)
            ax_m.hlines(0, 0, n_trials*n_blocks+1, lw=0.5, alpha=0.1, zorder=0)
            ax_m.set(xlim=(0.5,126), xticks=np.arange(7,126,14), xlabel='Trial', ylabel='Mood')

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Plot WAIC.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        if show_psis:

            ## Extract log-likelihood values.
            Y_log_lik = extract['Y_log_lik']
            M_log_lik = extract['M_log_lik']
            n_samp, n_subj, n_block, n_trial = Y_log_lik.shape

            ## Reshape data.
            Y_log_lik = Y_log_lik.reshape(n_samp, n_subj*n_block*n_trial)
            M_log_lik = M_log_lik.reshape(n_samp, n_subj*n_block*3)

            ## Remove log-likelihoods corresponding to missing data.
            Y_log_lik = np.where(Y_log_lik, Y_log_lik, np.nan)
            missing = np.isnan(Y_log_lik).mean(axis=0) > 0
            Y_log_lik = Y_log_lik[:,~missing] 

            ## Compute PSIS-LOO.
            Y_waic = WAIC(Y_log_lik).sum()
            M_waic = WAIC(M_log_lik).sum()
            T_waic = Y_waic + M_waic

            ## Plot.
            ax_p1.bar(i, -2*Y_waic, 1, color=palette[i])
            ax_p2.bar(i, -2*M_waic, 1, color=palette[i])
            ax_p3.bar(i, -2*T_waic, 1, color=palette[i])
            
        ## Add info.
        ax_p1.set(xticks=[], ylim=scale_bar_yaxis(ax_p1), ylabel='elppd', title='Model Fits: Choice')
        ax_p2.set(xticks=[], ylim=scale_bar_yaxis(ax_p2), title='Model Fits: Mood')
        ax_p3.set(xticks=[], ylim=scale_bar_yaxis(ax_p3), title='Model Fits: Overall')
        
    ## Add legend.
    if show_mood: ax_m.legend(loc=7, bbox_to_anchor=(1.15, 0.5), borderpad=0)
    elif show_behavior: ax_b.legend(loc=4, borderpad=0)
    else: show_psis: ax_p3.legend(loc=7, bbox_to_anchor=(1.15, 0.5), borderpad=0)
        
    sns.despine()
    plt.tight_layout()
    return fig, axes