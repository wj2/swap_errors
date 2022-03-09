# CODE_DIR = '/home/kelarion/github/assignment_errors/'
# SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'
CODE_DIR = 'C:/Users/mmall/Documents//github/assignment_errors/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/assignment_errors/'


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
import arviz as az

sys.path.append(CODE_DIR)
sys.path.append(CODE_DIR+'jeffcode/')
import general.data_io as gio
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan
# import swap_errors.visualization as swv
import general.neural_analysis as na
import general.plotting as gpl
import general.stan_utility as su

sys.path.append('C:/Users/mmall/Documents//github/repler/src/')
# import util

# # compile stan model
# su.recompile_model(CODE_DIR+'assignment_errors/linear_interp_color_model.stan')
# su.recompile_model(CODE_DIR+'assignment_errors/spatial_errors_model.stan')
# su.recompile_model(CODE_DIR+'assignment_errors/cue_mistake_model.stan')

#%%
def convexify(cols, bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    dc = 2*np.pi/(len(bins))
    
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    dist_near = np.abs(distances).min(0)
    nearest = np.abs(distances).argmin(0)
    sec_near = np.sign(distances[nearest,np.arange(len(cols))]+1e-8).astype(int)
    alpha = np.zeros((len(bins),len(cols)))
    alpha[nearest, np.arange(len(cols))] = (dc-dist_near)/dc
    alpha[np.mod(nearest+sec_near,len(bins)), np.arange(len(cols))] = 1 - (dc-dist_near)/dc
    
    return alpha

def linear_interp(num_tri, num_neur, num_col, num_bin, std=1.0):
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
    
    y_up = alpha_u.T@mu_u + alpha_l.T@mu_d_l + np.random.randn(num_tri, num_neur)*std
    y_low = alpha_l.T@mu_l + alpha_u.T@mu_d_u + np.random.randn(num_tri, num_neur)*std
    
    y = cue[:,None]*y_up + (1-cue[:,None])*y_low
    
    probs = np.repeat(np.arange(2)[:,None]==0,num_tri,axis=1).T
    
    return (alpha_u, alpha_l, cue, y, probs), (mu_u,mu_l,mu_d_u,mu_d_l, probs)

def spatial_errors(num_tri, num_neur, num_col, num_bin, alpha=[0.85,0.15], std=1.0):
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
    
    probs = np.random.dirichlet(alpha, num_tri)
    trial_type = np.array([np.random.multinomial(1,p) for p in probs]).T
    
    y_up_correct = alpha_u.T@mu_u + alpha_l.T@mu_d_l + np.random.randn(num_tri, num_neur)*std
    y_up_err = alpha_l.T@mu_u + alpha_u.T@mu_d_l + np.random.randn(num_tri, num_neur)*std
    y_low_correct = alpha_l.T@mu_l + alpha_u.T@mu_d_u + np.random.randn(num_tri, num_neur)*std
    y_low_err = alpha_u.T@mu_l + alpha_l.T@mu_d_u + np.random.randn(num_tri, num_neur)*std
    
    y_correct = cue[:,None]*y_up_correct + (1-cue[:,None])*y_low_correct
    y_err = cue[:,None]*y_up_err + (1-cue[:,None])*y_low_err
    
    y = (trial_type[:,:,None]*np.stack([y_correct,y_err])).sum(0)
    
    return (alpha_u, alpha_l, cue, y, probs), (mu_u,mu_l,mu_d_u,mu_d_l,trial_type)
    
def cue_mistake(num_tri, num_neur, num_col, num_bin, alpha=[0.85,0.15], std=1):
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
    
    probs = np.random.dirichlet(alpha, num_tri)
    trial_type = np.array([np.random.multinomial(1,p) for p in probs]).T
    
    y_up = alpha_u.T@mu_u + alpha_l.T@mu_d_l + np.random.randn(num_tri, num_neur)*std
    y_low = alpha_l.T@mu_l + alpha_u.T@mu_d_u + np.random.randn(num_tri, num_neur)*std
    
    y_correct = cue[:,None]*y_up + (1-cue[:,None])*y_low
    y_err = (1-cue[:,None])*y_up + cue[:,None]*y_low
    
    y = (trial_type[:,:,None]*np.stack([y_correct,y_err])).sum(0)
    
    return (alpha_u, alpha_l, cue, y, probs), (mu_u,mu_l,mu_d_u,mu_d_l,trial_type)


#%%

# PCA the neural activity before
# shorten the delay period
# put on the cluster
# sanity check with synthetic data 

num_dat = 500
num_neur = 3
num_col = 16
num_bins = 4

niter = 500
nchain = 4

noise = 1.0

# data_models = [linear_interp,spatial_errors,cue_mistake]
data_models = [spatial_errors]
# fit_models = ['linear_interp_color','spatial_errors','cue_mistake']
fit_models = ['spatial_errors','cue_mistake']

comps = {k.__name__:None for k in data_models}
for data_model in data_models:
    
    # if data_model.__name__ == 'linear_interp':
    #     (c_u, c_l, cue, y), (mu_u,mu_l,mu_d_u,mu_d_l) = data_model(num_dat, num_neur, num_col, num_bins)
    #     stan_data = dict(T=y.shape[0], N=y.shape[-1], K=num_bins, y=y, C_u=c_u.T, C_l=c_l.T, cue=cue)
    # else:
    (c_u, c_l, cue, y, probs), (mu_u,mu_l,mu_d_u,mu_d_l,trial_type) = data_model(num_dat, num_neur, num_col, num_bins, std=noise)
    
    probs = np.concatenate([probs, np.zeros((num_dat,1))], axis=-1)

    stan_data = dict(T=y.shape[0], N=y.shape[-1], K=num_bins, y=y, C_u=c_u.T, C_l=c_l.T, cue=cue, p=probs)
    
    
    fits = {k:None for k in fit_models}
    for fit_model in fit_models:
        model = pkl.load(open(CODE_DIR+'assignment_errors/%s_model.stan'%fit_model,'rb'))
        
        fit = model.sampling(data=stan_data, iter=niter, chains=nchain)
    
    # print(fit)
    
    # ll = fit.extract('log_lik')['log_lik']
    # eh = fit.extract('err_hat')['err_hat']
    
        model_params = {'observed_data':'y',
                        'log_likelihood':{'y':'log_lik'},
                        'posterior_predictive':'err_hat'}
        fit_az = az.from_pystan(posterior=fit, **model_params)
                                
        fits[fit_model] = fit_az
        
    print('Evaluating fit')
    comps[data_model.__name__] = az.compare(fits)

# prefix = '%s-data_%s-fit_'%(data_model.__name__, fit_model)

# pkl.dump(fit_az,open(SAVE_DIR+prefix+'arviz_fit.pkl','wb'))

# np.save(SAVE_DIR+prefix+'fitted_mu_u.npy', fit.extract('mu_u')['mu_u'])
# np.save(SAVE_DIR+prefix+'fitted_mu_l.npy', fit.extract('mu_l')['mu_l'])
# np.save(SAVE_DIR+prefix+'fitted_mu_d_u.npy', fit.extract('mu_d_u')['mu_d_u'])
# np.save(SAVE_DIR+prefix+'fitted_mu_d_l.npy', fit.extract('mu_d_l')['mu_d_l'])

# np.save(SAVE_DIR+prefix+'real_mu_u.npy', mu_u)
# np.save(SAVE_DIR+prefix+'real_mu_l.npy', mu_l)
# np.save(SAVE_DIR+prefix+'real_mu_d_u.npy', mu_d_u)
# np.save(SAVE_DIR+prefix+'real_mu_d_l.npy', mu_d_l)
# np.save(SAVE_DIR+prefix+'real_probs.npy', probs)

