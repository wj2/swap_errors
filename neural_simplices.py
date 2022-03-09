# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:53:51 2021

@author: mmall
"""


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
from matplotlib import cm
import matplotlib.colors as clr
import scipy.linalg as la
import scipy.io as sio
import scipy.stats as sts
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

dset_prm = {'session':list(range(13,23)),
 					'regions':['all'],
 					'tzf': 'WHEEL_ON_diode',
 					'tbeg':-0.5,
 					'twindow':0.5,
 					'tstep':0.5,
 					'num_bins':6,
 					'do_pca':'before', #'after'
 					'pca_thrs':0.95,
 					'min_trials':40,
 					'shuffle':False,
 					'impute_nan':True,
                    'shuffle_probs':False,
                    'pro':True,
 					'impute_params':{'weights':'uniform','n_neighbors':5},
 					'color_weights':'interpolated' # 'softmax'
 					}

## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

# simplx_basis = np.array([[1,-1,0],[-0.5,-0.5,1]])
simplx_basis = np.array([[1,0,-1],[-0.5,1,-0.5]])
simplx_basis /= la.norm(simplx_basis,axis=1,keepdims=True)

all_probs = []
for vals in list(itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    sess_probs = [[] for _ in these_sess]
    for idx, which_sess in enumerate(these_sess):
        this_dset['session'] = which_sess
        dset_info = {**this_dset}
        folds = hlp.folder_hierarchy(dset_info) 
    
        with open(SAVE_DIR+folds+'/arviz_fit_super_hybrid_error_hierarchical_model.pkl', 'rb') as f:
            az_fit = pkl.load(f)
            
        probs = az_fit.posterior['p_err'].to_numpy()
        sess_probs[idx] = probs@simplx_basis.T
        
    all_probs.append(sess_probs)

all_probs = np.array(all_probs)

#%%
cmap = 'tab20'

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

share_y_axis = False
# share_y_axis = True

xmin = -0.5*np.sqrt(2)
xmax = 0.5*np.sqrt(2)
ymin = np.sqrt(6)/3 - np.sqrt(1.5)
ymax = np.sqrt(6)/3
xx, yy = np.meshgrid(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100))
foo = (np.stack([xx.flatten(),yy.flatten()]).T@simplx_basis) + [1/3,1/3,1/3]
support = la.norm(foo,1, axis=-1)<1.001

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
    
    if len(col_labs)>=1:
        c = col_idx@n_col_lab
    else:
        c = 0
    if len(row_labs)>=1:
        r = row_idx@n_row_lab
    else:
        r = 0
    
    cols = getattr(cm, cmap)(np.arange(len(these_sess))/len(these_sess))
    for idx, sess in enumerate(these_sess):
        simp = all_probs[k, idx]
        
        kd_pdf = sts.gaussian_kde(simp.reshape((-1,2)).T)
        zz = np.where(support, kd_pdf(np.stack([xx.flatten(),yy.flatten()])), np.nan)
        
        axs[r,c].contour(xx,yy,zz.reshape(100,100,order='A'), 2,
                          colors=clr.to_hex(cols[idx]),
                          linestyles=['solid','dotted'])
        # axs[r,c].contourf(xx,yy,zz.reshape(100,100,order='A'), 2,
        #                  colors=clr.to_hex(cols[idx]),
        #                  alpha=0.7)
        axs[r,c].plot([xmin,xmax,0,xmin], [ymin, ymin, ymax, ymin],'#A6ACAF')
        
    axs[r,c].set_ylim([ymin*1.1,ymax*1.1])
    axs[r,c].set_xlim([xmin*1.1,xmax*1.1])
    axs[r,c].set_aspect('equal')
    axs[r,c].set_axis_off()
    # dicplt.square_axis(axs[r,c])

#%%

dset_prm = {'session':list(range(13)),
 					'regions':['all','frontal','posterior','decision','sensory'],
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
 					'shuffle_probs':False,
 					'impute_params':{'weights':'uniform','n_neighbors':5},
 					'color_weights':'interpolated' # 'softmax'
 					}

## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

all_probs = []
for vals in list(itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    sess_probs = [[] for _ in these_sess]
    for idx, which_sess in enumerate(these_sess):
        this_dset['session'] = which_sess
        dset_info = {**this_dset}
        folds = hlp.folder_hierarchy(dset_info) 
    
        with open(SAVE_DIR+folds+'/arviz_fit_hybrid_error_precue_model.pkl', 'rb') as f:
            az_fit = pkl.load(f)
            
        logits = az_fit.posterior['logits'].to_numpy()
        sess_probs[idx] = np.exp(logits)/(1+np.exp(logits))
        
    all_probs.append(sess_probs)

all_probs = np.array(all_probs)

#%%
cmap = 'tab20'

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

share_y_axis = False
# share_y_axis = True


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
    
    if len(col_labs)>=1:
        c = col_idx@n_col_lab
    else:
        c = 0
    if len(row_labs)>=1:
        r = row_idx@n_row_lab
    else:
        r = 0
    
    cols = getattr(cm, cmap)(np.arange(len(these_sess))/len(these_sess))
    for idx, sess in enumerate(these_sess):
        simp = all_probs[k, idx]
        
        kd_pdf = sts.gaussian_kde(simp.flatten())
        zz = kd_pdf(np.linspace(0,1,100))
        
        axs[r,c].plot(np.linspace(0,1,100), kd_pdf(np.linspace(0,1,100)), color=cols[idx])
    axs[r,c].set_xlabel(r'$p_{\rm{sptl}}$')
    axs[r,c].set_ylabel('Posterior density')
    axs[r,c].set_ylim([0, np.max(axs[r,c].get_ylim())])
    axs[r,c].set_xlim([0,1])  
    
    # dicplt.square_axis(axs[r,c])




