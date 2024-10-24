CODE_DIR = 'C:/Users/mmall/Documents/github/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/assignment_errors/fits/'

import socket
import os
import sys
import pickle as pkl

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.font_manager as mfm
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx
from tqdm import tqdm
import itertools as itt
import pystan as ps
import arviz as az
import umap

sys.path.append(CODE_DIR+'assignment_errors/')
sys.path.append(CODE_DIR+'assignment_errors/jeffcode/')
import general.data_io as gio
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan
import swap_errors.visualization as swv
import general.neural_analysis as na
import general.plotting as gpl
import general.stan_utility as su

import helpers as hlp

sys.path.append(CODE_DIR+'repler/src/')
import util
import plotting as dicplt

special_font = mfm.FontProperties(fname='C:/Windows/Fonts/seguiemj.ttf')

#%%
# these_models = ['null_hierarchical','spatial_error_hierarchical','cue_error_hierarchical','hybrid_error_hierarchical']

these_models = ['null_precue','spatial_error_precue','hybrid_error_precue']


# dset_prm = {'session':list(range(13)),
# 					'regions':['frontal','posterior'],
# 					'tzf': 'WHEEL_ON_diode',
# 					'tbeg':-0.5,
# 					'twindow':0.5,
# 					'tstep':0.5,
# 					'num_bins':6,
# 					'do_pca':'before', #'after'
# 					'pca_thrs':0.95,
# 					'min_trials':40,
# 					'shuffle':False,
# 					'impute_nan':True,
# 					'impute_params':{'weights':'uniform','n_neighbors':5},
# 					'color_weights':'interpolated' # 'softmax'
# 					}

dset_prm = {'session':list(range(13)),
					'regions':['frontal','posterior'],
					'tzf':'CUE2_ON_diode',
					'tbeg':-0.5,
					'twindow':0.5,
					'tstep':0.5,
					'num_bins':6,
					'do_pca':'before', #'after'
					'pca_thrs':0.95,
					'min_trials':40,
					'shuffle':False,
					'impute_nan':True,
					'impute_params':{'weights':'uniform','n_neighbors':5},
					'color_weights':'interpolated' # 'softmax'

## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list }
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list }

# these_sess = dset_prm['these_sess']
var_k, var_v = zip(*variable_prms.items())

az_fits = []
loos = []
for vals in list(itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    wa_az = []
    wa_prm = []
    # for which_sess in these_sess:
    dset_info = {**this_dset}
    folds = hlp.folder_hierarchy(dset_info) 

    fits_az = {p:None for p in these_models}
    for mod in these_models:
        with open(SAVE_DIR+folds+'/arviz_fit_%s_model.pkl'%mod, 'rb') as f:
            fits_az[mod] = pkl.load(f)
        # with open(SAVE_DIR+folds+'/fitted_params_%s_model.pkl'%mod, 'rb') as f:
        #     fits_prm[mod] = pkl.load(open(SAVE_DIR+folds+'/fitted_params_%s_model.pkl'%mod, 'rb'))
        
    comp = az.compare(fits_az)
    # wa_prm.append(np.exp(fits_az['hybrid_error_precue'].posterior['logits'].mean())/(1+np.exp(fits_az['hybrid_error_precue'].posterior['logits'].mean())))
    
    wa_az.append(comp)
    
    loos.append(wa_az)
    # az_fits.append(wa_prm)

vals = [np.array([loos[i][0]['loo'][m] for m in these_models]) for i in range(len(list(itt.product(*var_v))))]
errs = [np.array([loos[i][0]['se'][m] for m in these_models]) for i in range(len(list(itt.product(*var_v))))]
warn = [np.array([loos[i][0]['warning'][m] for m in these_models]) for i in range(len(list(itt.product(*var_v))))]


#%%

row_labs = var_k[1:]
row_lab_vals = var_v[1:]
col_labs = var_k[:1] 
col_lab_vals = var_v[:1]

# col_labs = var_k[1:]
# col_lab_vals = var_v[1:]
# row_labs = var_k[:1]
# row_lab_vals = var_v[:1]


y_ticks = False
# y_ticks = True

# share_y_axis = False
share_y_axis = True

axs = dicplt.hierarchical_labels(row_lab_vals, col_lab_vals,    
                                 row_names=row_labs, col_names=col_labs,
                                 fontsize=13, wmarg=0.3, hmarg=0.1)

n_row_lab = np.flip(np.array([1,]+[len(v) for v in row_lab_vals[1:]]))
n_col_lab = np.flip(np.array([1,]+[len(v) for v in col_lab_vals[1:]]))
for k, this_prm in enumerate(itt.product(*var_v)):
    
    col_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,col_labs))[0]])
    row_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,row_labs))[0]])
    
    c = col_idx@n_col_lab
    r = row_idx@n_row_lab
    
    warned = warn[k]
    
    axs[r,c].errorbar(np.arange(len(these_models)),
                      vals[k],yerr=errs[k],
                      linestyle='none', marker='o',
                      ecolor='k')
    # axs[r,c].errorbar(np.arange(len(these_models))[warned],
    #                   vals[k][warned],yerr=errs[k][warned],linestyle='none', marker='o', 
    #                   markerfacecolor='w', markeredgecolor='w', ecolor='k')
    for i in np.where(warned)[0]:
        axs[r,c].text(np.arange(len(these_models))[i].item(), vals[k][i].item(), 
                 s='💀', fontproperties=special_font, fontsize=12)
    
    if y_ticks:
        # axs[r,c].set_yticks([vals[k].min(), vals[k].max()])
        axs[r,c].set_yticks([util.significant_figures(vals[k].min(),3), util.significant_figures(vals[k].max(),3)])
        # axs[r,c].set_yticklabels([np.format_float_scientific(vals[k].min(),precision=1), 
        #                           np.format_float_scientific(vals[k].max(),precision=1)])
        axs[r,c].set_yticklabels([util.significant_figures(vals[k].min(),3), util.significant_figures(vals[k].max(),3)],
                                 rotation=30)
        # axs[r,c].ticklabel_format(axis='y',style='sci', scilimits=(0,0))
        formatter = tkr.ScalarFormatter(useMathText=True)
        formatter.set_useOffset(True)
        formatter.set_powerlimits((0,0))
        # formatter.set_powerlimits((formatter.orderOfMagnitude - 1,formatter.orderOfMagnitude - 1))
        axs[r,c].yaxis.set_major_formatter(formatter)
    else:
        axs[r,c].set_yticks([])
    
    if r<len(row_labs):
        axs[r,c].set_xticks([])
        axs[r,c].set_xticklabels([])
    else:
        axs[r,c].set_xticks(np.arange(len(these_models)))
        axs[r,c].set_xticklabels(these_models, rotation=30, rotation_mode='anchor', ha='right')
        
        



