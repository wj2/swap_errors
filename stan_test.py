CODE_DIR = '/home/kelarion/github/'
SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'

import socket
import os
import sys

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

#%% Data loading

# data = gio.Dataset.from_readfunc(swa.load_buschman_data, SAVE_DIR, max_files=np.inf,seconds=True, 
#                                   load_bhv_model=CODE_DIR+'/assignment_errors/bhv_model.pkl')

# data = gio.Dataset.from_readfunc(swa.load_buschman_data, SAVE_DIR, max_files=np.inf,
#                                  seconds=True,
#                                  load_bhv_model=CODE_DIR+'/assignment_errors/bhv_model.pkl',
#                                  spks_template=swa.busch_spks_templ_mua)

# print('Success!')

#%% Stan models

def null_model(num_tri, num_neur, num_col):
    mu_u = np.random.randn(num_col, num_neur)
    mu_l = np.random.randn(num_col, num_neur)
    mu_d_u = np.random.randn(num_col, num_neur)
    mu_d_l = np.random.randn(num_col, num_neur)
        
    c_u = np.random.randint(0,num_col, num_tri)
    c_l = np.random.randint(0,num_col, num_tri)
    cue = np.random.randint(0,2,num_tri)
    
    y_up = mu_u[c_u,:] + mu_d_l[c_l,:] + np.random.randn(num_tri, num_neur)
    y_low = mu_l[c_l,:] + mu_d_u[c_u,:] + np.random.randn(num_tri, num_neur)
    
    y = cue[:,None]*y_up + (1-cue[:,None])*y_low
    
    return (c_u, c_l, cue, y), (mu_u,mu_l,mu_d_u,mu_d_l)


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
    
    
