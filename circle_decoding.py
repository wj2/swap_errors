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
                                  load_bhv_model='C:/Users/mmall/Documents/github/assignment_errors/bhv_model.pkl',
                                  spks_template=swa.busch_spks_templ_mua)

#%%
tbeg = -.5
tend = 1
twindow = .5
tstep = .05
n_folds = 10
num_bins = 16

um = data['StopCondition'] == 1 
um = um.rs_and(data['is_one_sample_displayed'] == 0)
um = um.rs_and(data['Block']>1)
data_single = data.mask(um)

shuffle = False
repl_nan = False
# tzf = 'SAMPLES_ON_diode'
tzf = 'CUE2_ON_diode'
min_trials = 80
pre_pca = .99

this_color = data_single['LABthetaTarget']

bins = np.linspace(0, 2*np.pi, num_bins)

col1 =  this_color > np.pi
col2 = this_color <= np.pi

pop1, xs = data_single.mask(col1).get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)
pop2, xs = data_single.mask(col2).get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)
xs = xs[:int((tend-tbeg)/tstep)+1]

c1_n = data_single.mask(col1).get_ntrls()
c2_n = data_single.mask(col2).get_ntrls()
comb_n = data_single.combine_ntrls(c1_n, c2_n)

# [np.isin(data_single.mask(col1)['neur_regions'][i].values[0], 'v4pit'),...]
these_pop = [pop1[i] for i in range(len(pop1))]
ppop1, labs1 = data_single.mask(col1).make_pseudopop(these_pop, comb_n, min_trials, 10, skl_axs=True)

# [np.isin(data_single.mask(col2)['neur_regions'][i].values[0], 'v4pit'),...]
these_pop = [pop2[i] for i in range(len(pop2))]
ppop2, labs2 = data_single.mask(col2).make_pseudopop(these_pop, comb_n, min_trials, 10, skl_axs=True)

cv_perf = na.fold_skl(ppop1[0], ppop2[0], 
                      n_folds, model=svm.LinearSVC, params={'class_weight':'balanced'}, 
                      mean=False, pre_pca=pre_pca, shuffle=shuffle,
                      impute_missing=repl_nan)

# rbf_feat = kaprx.RBFSampler()
# ppop_all = np.concatenate([ppop1.squeeze()[...,(xs>0)&(xs<=0.5)], ppop2.squeeze()[...,(xs>0)&(xs<=0.5)]], axis=-2)
# n_samp = np.prod(ppop_all.shape[-2:])
# ppop_all = ppop_all.reshape((10,-1,n_samp))

# ppop_rbf = rbf_feat.fit_transform(ppop_all[0].T).T

# cv_perf = na.fold_skl(ppop_rbf[:,None,:,None], ppop_rbf[:,None,:,None], 
#                       n_folds, model=svm.LinearSVC, params={'class_weight':'balanced'}, 
#                       mean=False, pre_pca=pre_pca, shuffle=False,
#                       impute_missing=repl_nan)


#%%
tbeg = -.0
tend = 1
twindow = .5
tstep = .05
n_folds = 10
num_bins = 4

um = data['StopCondition'] == 1 
um = um.rs_and(data['is_one_sample_displayed'] == 0)
um = um.rs_and(data['Block']>1)
data_single = data.mask(um)

shuffle = False
repl_nan = False
# tzf = 'SAMPLES_ON_diode'
tzf = 'CUE2_ON_diode'
min_trials = 10
pre_pca = .99

this_color = data_single['LABthetaTarget']
# this_color = data_single['upper_color']

bins = np.linspace(0, 2*np.pi, num_bins+1)

ntrls = []
masks = []
for i in range(num_bins):
    msk =  (this_color > bins[i]).rs_and(this_color <= bins[i+1])
    ntrls.append(data_single.mask(msk).get_ntrls())
    masks.append(msk)

comb_n = data_single.combine_ntrls(*ntrls)

pops = []
for i in range(num_bins):    
    pop, xs = data_single.mask(masks[i]).get_populations(twindow, tbeg, tend, tstep,
                                                       skl_axes=True, repl_nan=repl_nan,
                                                       time_zero_field=tzf)
    
    ppop = data_single.mask(masks[i]).make_pseudopop(pop, comb_n, min_trials, 10, skl_axs=True)

    pops.append(ppop)
    
xs = xs[:int((tend-tbeg)/tstep)+1]


all_pops = np.concatenate(pops, axis=-2).squeeze()
labs = np.repeat(0.5*bins[:-1] + 0.5*bins[1:], [p.shape[-2] for p in pops])


cv_perf = []
for t in range(len(xs)):
    cv = cv_score(svm.LinearSVR(), all_pops[0][...,t].T, np.cos(labs), cv=n_folds)
    cv_perf.append(cv)
cv_perf = np.array(cv_perf)

# wa = cv_score(svm.LinearSVR(), all_pops[0].reshape((all_pops[0].shape[0],-1)).T, np.repeat(np.cos(labs),len(xs)), cv=n_folds)
#%%
tbeg = -.5
tend = 1
twindow = .5
tstep = .05
n_folds = 10
num_bins = 4

# um = data['StopCondition'] == 1 
um = data['is_one_sample_displayed'] == 0
um = um.rs_and(data['Block']>1)
data_single = data.mask(um)


shuffle = False
repl_nan = False
# tzf = 'SAMPLES_ON_diode'
tzf = 'CUE2_ON_diode'
min_trials = 40
pre_pca = .99

# grp1 = (data_single['IsUpperSample'] == 0).rs_and(data['StopCondition'] == 1)
# grp2 = (data_single['IsUpperSample'] == 1).rs_and(data['StopCondition'] == 1)
# grp1_err = (data_single['IsUpperSample'] == 0).rs_and(data['StopCondition'] == -1)
# grp2_err = (data_single['IsUpperSample'] == 1).rs_and(data['StopCondition'] == -1)
grp1 = (data_single['IsUpperSample'] == 0).rs_and(data['corr_prob']>0.7)
grp2 = (data_single['IsUpperSample'] == 1).rs_and(data['corr_prob']>0.7)
grp1_err = (data_single['IsUpperSample'] == 0).rs_and(data['corr_prob']<0.7)
grp2_err = (data_single['IsUpperSample'] == 1).rs_and(data['corr_prob']<0.7)
# grp1_err = (data_single['IsUpperSample'] == 0).rs_and(data['guess_prob']>0.3)
# grp2_err = (data_single['IsUpperSample'] == 1).rs_and(data['guess_prob']>0.3)

n1 = data_single.mask(grp1).get_ntrls()
n2 = data_single.mask(grp2).get_ntrls()
n1_err = data_single.mask(grp1_err).get_ntrls()
n2_err = data_single.mask(grp2_err).get_ntrls()

comb_n = gio.combine_ntrls(n1, n2, n1_err, n2_err)
  
pop1, xs = data_single.mask(grp1).get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)
pop2, xs = data_single.mask(grp2).get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)

ppop1 = data_single.mask(grp1).make_pseudopop(pop1, comb_n, min_trials, 10, skl_axs=True)
ppop2 = data_single.mask(grp2).make_pseudopop(pop2, comb_n, min_trials, 10, skl_axs=True)
    

pop1_err, xs = data_single.mask(grp1_err).get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)
pop2_err, xs = data_single.mask(grp1_err).get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)

ppop1_err = data_single.mask(grp1_err).make_pseudopop(pop1_err, comb_n, min_trials, 10, skl_axs=True)
ppop2_err = data_single.mask(grp2_err).make_pseudopop(pop1_err, comb_n, min_trials, 10, skl_axs=True)


xs = xs[:int((tend-tbeg)/tstep)+1]

cv_perf = na.fold_skl(ppop1[0], ppop2[0], n_folds, model=svm.LinearSVC, params={'class_weight':'balanced'}, 
                      mean=False, pre_pca=pre_pca, shuffle=False,
                      impute_missing=repl_nan)
cv_base = na.fold_skl(ppop1[0], ppop2[0], n_folds, model=svm.LinearSVC, params={'class_weight':'balanced'}, 
                      mean=False, pre_pca=pre_pca, shuffle=True,
                      impute_missing=repl_nan)

ppop = np.concatenate([ppop1, ppop2],axis=-2).squeeze()
ppop_err = np.concatenate([ppop1_err, ppop2_err],axis=-2).s queeze()
labels = np.concatenate([np.zeros(ppop1.shape[-2]), np.ones(ppop2.shape[-2])])
labels_err = np.concatenate([np.zeros(ppop1_err.shape[-2]), np.ones(ppop2_err.shape[-2])])


err_test_perf = []
clf = svm.LinearSVC()
flat_perf = []
flat_err_test = []
for p in range(len(ppop)):
    wa = []
    for t in range(len(xs)):
        clf.fit(ppop[p][...,t].T, labels)
        wa.append(clf.score(ppop_err[p][...,t].T, labels_err))
    err_test_perf.append(wa)
    
    flattened = ppop[p][...,cv_perf.mean(0)>0.9].reshape((ppop.shape[1],-1))
    flattened_err = ppop_err[p][...,cv_perf.mean(0)>0.9].reshape((ppop_err.shape[1],-1))

    flat_perf.append(cv_score(svm.LinearSVC(), flattened.T, np.repeat(labels, np.sum(cv_perf.mean(0)>0.9))))
    
    clf = svm.LinearSVC()
    clf.fit(flattened.T, np.repeat(labels, np.sum(cv_perf.mean(0)>0.9)))
    flat_err_test.append(clf.score(flattened_err.T, np.repeat(labels_err, np.sum(cv_perf.mean(0)>0.9))))


#%%
tbeg = -.5
tend = 0
twindow = .5
tstep = .5
n_folds = 10


um = data['is_one_sample_displayed'] == 0
# um = um.rs_and(data['StopCondition'] == 1)
um = um.rs_and(data['Block']>1)
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

trn_set = data_single['corr_prob']>0.7
# tst_set = data_single['StopCondition'] == 1
tst_set = data_single['swap_prob']>0.3
# tst_set = data_single['guess_prob']>0.3
# tst_set = data_single['StopCondition'] == -1
# tst_set = data_single['corr_prob']>0.5
# tst_set = data_single['corr_prob']<0.5

# these_times = (xs>0.4)&(xs<=1.0)
these_times = (xs>-0.5)&(xs<=0.0)

cv_perf_all = []
err_perf = []
for i in tqdm(range(num_pop)):
    # these_neur = np.isin(data_single['neur_regions'][i].values[2], 'v4pit')
    # if np.sum(these_neur)==0:
        # print('skipping')
        # continue
    binned = pops[i].squeeze()#[these_neur,...]
    # binned = np.concatenate([ppop1.squeeze()[i,...],ppop2.squeeze()[i,...]], axis=-2)
    
    # U, S, _ = la.svd(binned.reshape((binned.shape[0],-1))-binned.mean((1,2))[:,None], full_matrices=False)
    # n_comp = np.argmax(np.cumsum((S**2)/np.sum(S**2))>pca_thr)
    # binned = np.einsum('ik...,kj...->ij...',U[:,:n_comp].T,binned)
    
    # binned = (binned-binned.mean((1,2),keepdims=True))/(binned.std((1,2),keepdims=True)+1e-4)
    
    for_trn = trn_set[i].array.to_numpy()
    for_tst = tst_set[i].array.to_numpy()
    
    if np.sum(for_tst)==0:
        print('oops!')
        continue
    
    # n_trn = int(0.9*np.sum(for_trn))
    # binned_trn = binned[for_trn][:,:n_trn,:]
    # binned_tst = binned[for_trn][:,n_trn:,:]
    
    nT = binned.shape[2]
    n_comp = binned.shape[0]
    
    labels_trn = data_single['IsUpperSample'][i].array.to_numpy()[for_trn]
    labels_tst = data_single['IsUpperSample'][i].array.to_numpy()[for_tst]
    
    binned_trn = binned[:,for_trn,:][:,:,these_times].reshape((n_comp, -1)).T
    binned_tst = binned[:,for_tst,:][:,:,these_times].reshape((n_comp, -1)).T
    
    clf = svm.LinearSVC()
    # cv_perf_all.append(cv_score(clf, binned_trn, np.repeat(labels_trn,np.sum(these_times)), cv=10))
    
    # this_set = np.random.choice(len(binned_trn),int(len(binned_trn)*9/10), replace=False)
    # clf.fit(binned_trn[this_set,:], np.repeat(labels_trn,np.sum(these_times))[this_set])
    cv_clf = skcv(clf, binned_trn, np.repeat(labels_trn,np.sum(these_times)), cv=10, return_estimator=True)
    
    cv_perf_all.append(cv_clf['test_score'])
    perf = [this_clf.score(binned_tst, np.repeat(labels_tst,np.sum(these_times))) for this_clf in cv_clf['estimator']]
    
    err_perf.append(perf)


trn_err = np.array(cv_perf_all)
tst_err = np.array(err_perf)

elmo = np.arange(num_pop)<=12
wald = np.arange(num_pop)>12

#%%
elm_clr = '#EC7063'
wld_clr = '#3498DB'

plt.errorbar(trn_err[elmo,:].mean(1), tst_err[elmo,:].mean(1), 
             xerr=trn_err[elmo,:].std(1), yerr=tst_err[elmo,:].std(1),
             ls='none', ecolor=elm_clr, marker='.', c=elm_clr, markersize=10)
plt.errorbar(trn_err[wald,:].mean(1), tst_err[wald,:].mean(1), 
             xerr=trn_err[wald,:].std(1), yerr=tst_err[wald,:].std(1),
             ls='none', ecolor=wld_clr, marker='.',c=wld_clr, markersize=10)
# plt.scatter(trn_err.mean(1), tst_err.mean(1), c=np.arange(num_pop)>12)
dicplt.square_axis()
plt.plot(plt.xlim(),plt.xlim(),'k--')


