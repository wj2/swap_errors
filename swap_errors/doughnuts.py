
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

#%%

ndat = 100
nneur = 300

c_up = np.repeat(np.linspace(0,2*np.pi,ndat), ndat) # upper color
c_low = np.tile(np.linspace(0,2*np.pi,ndat), ndat) # lower color

circ1 = np.stack([np.sin(c_up), np.cos(c_up)])
circ2 = np.stack([np.sin(c_low), np.cos(c_low)])


basis = la.qr( np.random.randn(nneur,nneur).T)[0][:8,:]

#%% parallel circles

mu_u = basis[:2,:].T@circ1 + np.random.randn(nneur,ndat**2)*0.1
mu_d_l = 0.6*basis[2:4,:].T@circ2 + np.random.randn(nneur,ndat**2)*0.1


mu_l = basis[:2,:].T@circ2 + basis[[5],:].T + np.random.randn(nneur,ndat**2)*0.1
mu_d_u = 0.6*basis[2:4,:].T@circ1 + basis[[6],:].T + np.random.randn(nneur,ndat**2)*0.1

T1_par = mu_u + mu_d_l   # upper cued torus
T2_par = mu_l + mu_d_u  # lower cued torus

U, mwa = util.pca(T1_par)
V, _ = util.pca(T2_par)

plt.plot([(U[:,:k].T@T2_par).var(1).sum(0)/T2_par.var(1).sum(0) for k in range(nneur)])
plt.plot([(V[:,:k].T@T1_par).var(1).sum(0)/T1_par.var(1).sum(0) for k in range(nneur)])

#%%
mu_u = basis[:2,:].T@circ1 + np.random.randn(nneur,ndat**2)*0.1
mu_d_l = 0.6*basis[2:4,:].T@circ2 + np.random.randn(nneur,ndat**2)*0.1


mu_l = basis[4:6,:].T@circ2 + np.random.randn(nneur,ndat**2)*0.1
mu_d_u = 0.6*basis[6:,:].T@circ1 + np.random.randn(nneur,ndat**2)*0.1

T1_orth = mu_u + mu_d_l  # upper cued torus
T2_orth = mu_l + mu_d_u   # lower cued torus

U, mwa = util.pca(T1_orth)
V, mwa = util.pca(T2_orth)

plt.plot([(U[:,:k].T@T2_orth).var(1).sum(0)/T2_orth.var(1).sum(0) for k in range(54)])
plt.plot([(V[:,:k].T@T1_orth).var(1).sum(0)/T1_orth.var(1).sum(0) for k in range(54)])

#%%

avg_up_cue = np.array([T1_par[:,c_up==c].mean(1) for c in np.unique(c_up)])
avg_low_cue = np.array([T2_par[:,c_low==c].mean(1) for c in np.unique(c_low)])

avg_up_uncue = np.array([T2_par[:,c_up==c].mean(1) for c in np.unique(c_up)])
avg_low_uncue = np.array([T1_par[:,c_low==c].mean(1) for c in np.unique(c_low)])




