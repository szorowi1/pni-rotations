import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import _pickle as cPickle
import warnings
from . utilities import WAIC

warnings.filterwarnings("ignore", category=RuntimeWarning) 

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