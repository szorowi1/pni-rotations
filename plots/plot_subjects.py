import os
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=1.25)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Metadata.
n_subj = 31

## I/O parameters.
models = ['moodRL_ppool_base', 'moodRL_ppool_mood', 'moodRL_ppool_mood_bias_mod']

## Plotting parameters.
colors = sns.color_palette(n_colors=len(models)).as_hex()
labels = ['base', 'mood', r'mb-fix$h$']

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define paths.
model_dir = '../stan_fits'
out_dir = 'subj_ppc'

for i in np.arange(n_subj):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Initialize figure.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Open canvas.
    fig = plt.figure(figsize=(12,8))
    
    ## Customize axes with GridSpec.
    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(left=0.07, right=0.89, bottom=0.45,  top=0.985, hspace=0.05)

    gs2 = gridspec.GridSpec(1, 4)
    gs2.update(left=0.07, right=0.89, bottom=0.085,  top=0.35, hspace=0, wspace=0.25)
    
    ## Initialize 3x4 grid.
    axes = [plt.subplot(gs1[0]), plt.subplot(gs1[1]), plt.subplot(gs2[0]), 
            plt.subplot(gs2[1]), plt.subplot(gs2[2]), plt.subplot(gs2[3])]
    
    for j, model in enumerate(models):
        
        ## Load StanFit file.
        f = '%s/%s/StanFit.pickle' %(model_dir,model)
        with open(f, 'rb') as f: extract = cPickle.load(f)
    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Plot behavior.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        ## Extract data and compute optimal choice.
        optimal_choice = np.argmax(extract['X'][i], axis=-1)
        Y_obs = np.equal(extract['Y'][i]-1, optimal_choice).astype(int)
        Y_pred = np.array([np.equal(sample, optimal_choice) for sample in extract['Y_pred'][:,i]-1]).astype(int)

        ## Mask missing data.
        missing = extract['Y'][i] < 0
        Y_obs = np.where(missing, np.nan, Y_obs)
        Y_pred = np.array([np.where(missing, np.nan, sample) for sample in Y_pred])
        
        ## Compute average.
        Y_pred = np.nanmean(Y_pred, axis=0)
        n_blocks, n_trials = Y_pred.shape
        
        for block in np.arange(n_blocks):
        
            ## Define trial numbers.
            trials = np.arange(n_trials) + block * n_trials
            trials += 1

            ## Plot.
            if not j: axes[0].scatter( trials, Y_obs[block]*1.05, s=20, marker='o', color='k' )
            axes[0].plot( trials, Y_pred[block], lw=2.5, color=colors[j], alpha=0.8 )
        
        ## Add info.
        axes[0].vlines([42.5, 84.5], 0, 1, lw=1.5, color='k', zorder=10)
        axes[0].set(xlim=(0.5, 126), xticks=[], xlabel='', ylim=(0.3, 1.1), 
                    ylabel='Optimal Choice')
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Plot mood.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        ## Extract predicted h-value and convert to mood.
        M_pred = np.tanh(extract['h_pred'][:,i])
        M_obs = extract['M'][i]
        
        ## Compute average.
        M_pred = np.median(M_pred, axis=0)
        n_blocks, n_trials = M_pred.shape
        
        for block in np.arange(n_blocks):
        
            ## Define trial numbers.
            trials = np.arange(n_trials) + block * n_trials
            trials += 1

            ## Plot.
            if not j: axes[1].scatter( trials[[6,20,34]], M_obs[block], s=150, marker='d', color='k', zorder=100 )
            axes[1].plot( trials, M_pred[block], lw=2.5, label=labels[j], color=colors[j], alpha=0.8 )
        
        ## Add info.
        axes[1].hlines(0, 0, n_trials*n_blocks+1, linestyle='--', alpha=0.1, zorder=0)
        axes[1].vlines([42.5, 84.5], -1, 1, lw=1.5, color='k', zorder=10)
        axes[1].set(xlim=(0.5, 126), xticks=np.arange(7,126,14), xlabel='Trial', ylim=(-1,1), ylabel='Mood')
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Plot paramaters.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        if 'beta' in extract.keys():
            sns.kdeplot(extract['beta'][:,i], lw=2.5, color=colors[j], ax=axes[2])
            axes[2].set(xlim=(0,20), xlabel=r'$\beta$', ylabel='Count')
            
        if 'eta_v' in extract.keys():
            sns.kdeplot(extract['eta_v'][:,i], lw=2.5, color=colors[j], ax=axes[3])
            axes[3].set(xlim=(0,0.5), xlabel=r'$\eta_v$', xticks=np.linspace(0,0.4,3))
        
        if 'eta_h' in extract.keys():
            sns.kdeplot(extract['eta_h'][:,i], lw=2.5, color=colors[j], ax=axes[4])
            axes[4].set(xlim=(0,0.5), xlabel=r'$\eta_h$', xticks=np.linspace(0,0.4,3))
            
        if 'f' in extract.keys():
            sns.kdeplot(extract['f'][:,i], lw=2.5, color=colors[j], ax=axes[5])
            axes[5].set_xscale('log')
            axes[5].set(xlim=np.logspace(-2,1,2), xlabel=r'$f$', xticks=[0.01, 0.1, 1, 10], 
                        xticklabels=['0.01', '0.10', '1.00', '10.0'])
            
    ## Add legend.
    axes[1].legend(loc=7, bbox_to_anchor=(1.135, 0.5), borderpad=0)
        
    ## Save figure.
    sns.despine()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.1, hspace=0.3)
    plt.savefig(os.path.join(out_dir, 'subj%s' %i), dpi=180)
    plt.close('all')