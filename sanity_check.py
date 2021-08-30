CODE_DIR = '/home/kelarion/github/'
SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'

import socket
import os
import sys
import pickle as pkl

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx
from tqdm import tqdm
import pystan as ps

sys.path.append(CODE_DIR+'jeffcode/')
import general.data_io as gio
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan
# import swap_errors.visualization as swv
import general.neural_analysis as na
import general.plotting as gpl
import general.stan_utility as su

sys.path.append(CODE_DIR+'repler/src/')
import util

# compile stan model
# su.recompile_model(CODE_DIR+'assignment_errors/linear_interp_color_model.stan')

#%%
def convexify(cols, bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    dc = 2*np.pi/(len(bins))
    
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    dist_near = (distances).max(0)
    nearest = (distances).argmax(0)
    alpha = np.zeros((len(bins),len(cols)))
    alpha[nearest, np.arange(len(cols))] = (dist_near-dc)/dc
    alpha[np.mod(nearest+1,len(bins)), np.arange(len(cols))] = 1 - (dist_near-dc)/dc
    
    return alpha

def linear_interp(num_tri, num_neur, num_col, num_bin):
    mu_u = np.random.randn(num_bin, num_neur)
    mu_l = np.random.randn(num_bin, num_neur)
    mu_d_u = np.random.randn(num_bin, num_neur)
    mu_d_l = np.random.randn(num_bin, num_neur)
        
    c_u = np.random.randint(0, num_bin, num_tri)
    c_l = np.random.randint(0, num_bin, num_tri)
    cue = np.random.randint(0,2,num_tri)
    
    alpha_u = np.zeros((num_bin,num_tri)) # draw the interpolation parameters
    alpha_l = np.zeros((num_bin,num_tri))
    alp_u = np.random.choice(np.linspace(0,1,(num_col//num_bin)+1)[:-1],num_tri)
    alp_l = np.random.choice(np.linspace(0,1,(num_col//num_bin)+1)[:-1],num_tri)
    
    alpha_u[c_u,np.arange(num_tri)] = alp_u # create interpolation matrix
    alpha_u[np.mod(c_u+1, num_bin),np.arange(num_tri)] = 1-alp_u
    
    alpha_l[c_l,np.arange(num_tri)] = alp_l # create interpolation matrix
    alpha_l[np.mod(c_l+1, num_bin),np.arange(num_tri)] = 1-alp_l
    
    y_up = alpha_u.T@mu_u + alpha_l.T@mu_d_l + np.random.randn(num_tri, num_neur)
    y_low = alpha_l.T@mu_l + alpha_u.T@mu_d_u + np.random.randn(num_tri, num_neur)
    
    y = cue[:,None]*y_up + (1-cue[:,None])*y_low
    
    return (alpha_u, alpha_l, cue, y), (mu_u,mu_l,mu_d_u,mu_d_l)
    
#%%

# PCA the neural activity before
# shorten the delay period
# put on the cluster
# sanity check with synthetic data 

num_bins = 4

(c_u, c_l, cue, y), (mu_u,mu_l,mu_d_u,mu_d_l) = linear_interp(5000, 3, 16, 4)

stan_data = dict(T=y.shape[0], N=y.shape[-1], K=num_bins, y=y, C_u=c_u.T, C_l=c_l.T, cue=cue)

#%%

model = pkl.load(open(CODE_DIR+'assignment_errors/linear_interp_color_model.stan','rb'))

fit = model.sampling(data=stan_data, iter=2000, chains=4)

print(fit)

np.save(SAVE_DIR+'fitted_mu_u.npy', fit.extract('mu_u')['mu_u'])
np.save(SAVE_DIR+'fitted_mu_l.npy', fit.extract('mu_l')['mu_l'])
np.save(SAVE_DIR+'fitted_mu_d_u.npy', fit.extract('mu_d_u')['mu_d_u'])
np.save(SAVE_DIR+'fitted_mu_d_l.npy', fit.extract('mu_d_l')['mu_d_l'])

np.save(SAVE_DIR+'real_mu_u.npy', mu_u)
np.save(SAVE_DIR+'real_mu_l.npy', mu_l)
np.save(SAVE_DIR+'real_mu_d_u.npy', mu_d_u)
np.save(SAVE_DIR+'real_mu_d_l.npy', mu_d_l)


