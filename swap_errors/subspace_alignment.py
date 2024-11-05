# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:35:26 2021

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
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
from sklearn.model_selection import cross_validate as skcv
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

BASE_DIR = 'C:/Users/mmall/Documents/uni/columbia/assignment_errors/'

#%%

# data = gio.Dataset.from_readfunc(swa.load_buschman_data, BASE_DIR, max_files=np.inf,seconds=True, 
#                                   load_bhv_model='C:/Users/mmall/Documents/github/assignment_errors/bhv_model.pkl')

data = gio.Dataset.from_readfunc(swa.load_buschman_data, BASE_DIR, max_files=np.inf,
                                  seconds=True,
                                  load_bhv_model='C:/Users/mmall/Documents/github/assignment_errors/bhv_model-pr.pkl',
                                  spks_template=swa.busch_spks_templ_mua)


#%%
tbeg = -1.5
tend = 0.2
twindow = .1
tstep = .1
n_folds = 10


um = data['is_one_sample_displayed'] == 0
# um = um.rs_and(data['StopCondition'] == 1)
# um = um.rs_and(data['Block']>1)
data_single = data.mask(um)

shuffle = False
repl_nan = False
# tzf = 'SAMPLES_ON_diode'
# tzf = 'CUE2_ON_diode'
tzf = 'WHEEL_ON_diode'
min_trials = 80
pre_pca = .99

# try z-scoring
# try pca
# heirarchical
# try only visual (v4) neurons
# pseudopop
num_pop = data_single.n_sessions
# num_pop = 10 

pops, xs = data_single.get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)
xs = xs[:int((tend-tbeg)/tstep)+1]


#%%

grp1 = (data_single['IsUpperSample'] == 0)
grp2 = (data_single['IsUpperSample'] == 1)


n1 = data_single.mask(grp1).get_ntrls()
n2 = data_single.mask(grp2).get_ntrls()

comb_n = gio.combine_ntrls(n1, n2)
  
pop1, xs = data_single.mask(grp1).get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)
pop2, xs = data_single.mask(grp2).get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)

ppop1 = data_single.mask(grp1).make_pseudopop(pop1, comb_n, 300, 10)
ppop2 = data_single.mask(grp2).make_pseudopop(pop2, comb_n, 300, 10)


# ppop = np.concatenate([ppop1, ppop2],axis=-2).squeeze().sum(-1)
ppop = np.concatenate([ppop1, ppop2],axis=-2).squeeze()
labels = np.concatenate([np.zeros(ppop1.shape[-2]), np.ones(ppop2.shape[-2])])

n_pp = ppop.shape[1]

#%%

vv = []
uv = []
vu = []
uu = []
null = []
for pp in range(len(ppop)):
    p1 = ppop[pp,:,labels==0,t].T
    p2 = ppop[pp,:,labels==1,t].T
    
    U, l1 = util.pca(p1)
    V, l2 = util.pca(p2)

    uu.append(np.cumsum(l1)/np.sum(l1))
    vv.append(np.cumsum(l2)/np.sum(l2))
    uv.append(np.array([(V[:,:k].T@p1).var(1).sum(0)/p1.var(1).sum(0) for k in range(np.min(V.shape))]))
    vu.append(np.array([(U[:,:k].T@p2).var(1).sum(0)/p2.var(1).sum(0) for k in range(np.min(U.shape))]))
    
    # la.qr( np.random.randn(n_pp,n_pp).T)[0]
    null.append(np.array([[(la.qr( np.random.randn(n_pp,n_pp).T)[0][:,:k].T@p2).var(1).sum(0)/p2.var(1).sum(0) \
                          for k in range(np.min(U.shape))] for _ in range(15)]))


#%%

# n1 = data_single.get_ntrls()

# pop, xs = data_single.get_populations(twindow, tbeg, tend, tstep, repl_nan=repl_nan,
#                                                    time_zero_field=tzf)

# ppop = data_single.make_pseudopop(pop, n1, 300, 10)


grp1 = (data_single['Block']>1)
grp2 = (data_single['Block']==1)


n1 = data_single.mask(grp1).get_ntrls()
n2 = data_single.mask(grp2).get_ntrls()

comb_n = gio.combine_ntrls(n1, n2)

pop1, xs = data_single.mask(grp1).get_populations(twindow, tbeg, tend, tstep,
                                                  repl_nan=repl_nan,
                                                   time_zero_field=tzf)
pop2, xs = data_single.mask(grp2).get_populations(twindow, tbeg, tend, tstep, 
                                                  repl_nan=repl_nan,
                                                   time_zero_field=tzf)

ppop1 = data_single.mask(grp1).make_pseudopop(pop1, comb_n, 300, 10)
ppop2 = data_single.mask(grp2).make_pseudopop(pop2, comb_n, 300, 10)


# ppop = np.concatenate([ppop1, ppop2],axis=-2).squeeze().sum(-1)
# ppop = np.concatenate([ppop1, ppop2],axis=-2).squeeze()
# labels = np.concatenate([np.zeros(ppop1.shape[-2]), np.ones(ppop2.shape[-2])])

# n_pp = ppop.shape[1]
ppop1 = (ppop1 - ppop1.mean(1, keepdims=True))/(ppop1.std(1, keepdims=True)+1e-8) # zscore
ppop2 = (ppop2 - ppop2.mean(1, keepdims=True))/(ppop2.std(1, keepdims=True)+1e-8)

#%%
 
U1, l1 = util.pca(ppop1[0].transpose((2,1,0)), full_matrices=False)
U2, l2 = util.pca(ppop2[0].transpose((2,1,0)), full_matrices=False)

U1V1 = np.einsum('tki,fkj->tfij',U1,U1) 
U2V2 = np.einsum('tki,fkj->tfij',U2,U2)
U2V1 = np.einsum('tki,fkj->tfij',U2,U1)

rr = np.einsum('tfij,tj->tfi',U1V1**2, l1)
pp = np.einsum('tfij,tj->tfi',U2V2**2, l2)
pr = np.einsum('tfij,tj->tfi',U2V1**2, l1)

#%%




