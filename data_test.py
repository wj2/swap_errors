import socket
import os
import sys

if sys.platform == 'linux':
    CODE_DIR = '/home/kelarion/github/'
    SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'
else:
    CODE_DIR = 'C:/Users/mmall/Documents/github/'
    SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/assignment_errors/'

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


# compile stan model
# su.recompile_model(CODE_DIR+'assignment_errors/linear_interp_color_model.pkl')
# su.recompile_model(CODE_DIR+'assignment_errors/spatial_errors_model.pkl')
# su.recompile_model(CODE_DIR+'assignment_errors/cue_mistake_model.pkl')

#%%
def convexify(cols, bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    dc = 2*np.pi/(len(bins))
    
    # get the nearest bin
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    dist_near = np.abs(distances).min(0)
    nearest = np.abs(distances).argmin(0)
    # see if the color is to the "left" or "right" of that bin
    sec_near = np.sign(distances[nearest,np.arange(len(cols))]+1e-8).astype(int) # add epsilon to handle 0
    # fill in the convex array
    alpha = np.zeros((len(bins),len(cols)))
    alpha[nearest, np.arange(len(cols))] = (dc-dist_near)/dc
    alpha[np.mod(nearest-sec_near,len(bins)), np.arange(len(cols))] = 1 - (dc-dist_near)/dc
    
    return alpha

def smconvexify(cols, num_bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    bins = np.linspace(0,2*np.pi,num_bins+1)[:num_bins]
    dc = 2*np.pi/num_bins
    
    # get the nearest bin
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    alpha = np.exp(np.abs(distances))/np.exp(np.abs(distances)).sum(0)
    
    return alpha

def box_conv(X, len_filt):
    T = x.shape[1]
    N = x.shape[0]
    
    f = np.eye(T+len_filt,T)
    f[np.arange(T)+len_filt,np.arange(T)] = -1
    filt = np.cumsum(f,0)
    
    x_pad = np.stack([np.concatenate([np.zeros((N,len_filt-i)), X, np.zeros((N,i))],axis=1) for i in range(len_filt+1)])
    filted = x_pad@filt
    
    return filted


#%% Data loading

print('Loading data')

data = gio.Dataset.from_readfunc(swa.load_buschman_data, SAVE_DIR, max_files=np.inf,seconds=True, 
                                  load_bhv_model=CODE_DIR+'/assignment_errors/bhv_model.pkl')

# data = gio.Dataset.from_readfunc(swa.load_buschman_data, SAVE_DIR, max_files=np.inf,
#                                  seconds=True,
#                                  load_bhv_model=CODE_DIR+'/assignment_errors/bhv_model.pkl',
#                                  spks_template=swa.busch_spks_templ_mua)

print('Success!')

#%%

# PCA the neural activity before
# shorten the delay period
# put on the cluster
# sanity check with synthetic data 

# make one long bin per trial

tbeg = 0
tend = 0.5
twindow = .5
tstep = .5
num_bins = 4

which_sess = 3

do_pca = True
pca_thrs = 0.95

# um = data['StopCondition'] == 1 
um = data['is_one_sample_displayed'] == 0
# um = um.rs_and(data['Block']>1)
um = um.rs_and(data['corr_prob']<2) # avoid nan trials
data_single = data.mask(um)

shuffle = False
repl_nan = False
# tzf = 'SAMPLES_ON_diode'
tzf = 'CUE2_ON_diode'
min_trials = 40
pre_pca = .99

n = data_single.get_ntrls()

pop, xs = data_single.get_populations(twindow, tbeg, tend, tstep,
                                                   skl_axes=True, repl_nan=repl_nan,
                                                   time_zero_field=tzf)   

# xs = xs[:int((tend-tbeg)/tstep)+1]

# # x = pop[which_sess][...,xs>0].reshape((pop[which_sess].shape[0],-1))
# x = pop[which_sess].sum(-1).reshape((pop[which_sess].shape[0],-1))

# imp = np.any(box_conv(x==0,20)==20,axis=0)
# trash = imp.mean(1)>0.5
# x = x[~trash,:]
# imp = imp[~trash,:]

# imptr = knni(weights='distance')
# x = imptr.fit_transform(np.where(imp,np.nan,x))
    
# if do_pca:
#     x = util.pca_reduce(x, thrs=pca_thrs)

# # z-score
# x -= x.mean(0,keepdims=True)
# x /= (x.std(0, keepdims=True)+1e-8)

# # yup = np.repeat(data_single['upper_color'][which_sess].array.to_numpy(), np.sum(xs>0))
# # ylow = np.repeat(data_single['lower_color'][which_sess].array.to_numpy(), np.sum(xs>0))
# # cue = np.repeat(data_single['IsUpperSample'][which_sess].array.to_numpy(), np.sum(xs>0))

# yup = data_single['upper_color'][which_sess].array.to_numpy()
# ylow = data_single['lower_color'][which_sess].array.to_numpy()
# cue = data_single['IsUpperSample'][which_sess].array.to_numpy()


# bins = np.linspace(0,2*np.pi,num_bins+1)[:num_bins]

# c_u = convexify(yup, bins).T
# c_l = convexify(ylow, bins).T

# probs = np.stack([data_single[tt][which_sess].array.to_numpy() for  tt in ['corr_prob','swap_prob','guess_prob']]).T

# stan_data = dict(T=x.shape[0], N=x.shape[-1], K=num_bins, y=x, C_u=c_u, C_l=c_l, cue=cue, p = probs)

# #%%
# niter = 500
# nchain = 4

# print ('Fitting models')

# # fit_models = ['linear_interp_color','spatial_errors','cue_mistake']
# fit_models = ['spatial_errors']

# fits = []
# az_fits = {k:None for k in fit_models}
# for fit_model in fit_models:
    
#     print('Fitting %s model'%fit_model)
#     model = pkl.load(open(CODE_DIR+'assignment_errors/%s_model.pkl'%fit_model,'rb'))
    
#     fit = model.sampling(data=stan_data, iter=niter, chains=nchain)

#     print('Converting to Arvix ...')
#     model_params = {'observed_data':'y',
#                     'log_likelihood':{'y':'log_lik'},
#                     'posterior_predictive':'err_hat'}
#     fit_az = az.from_pystan(posterior=fit, **model_params)
                            
#     fits.append(fit)
#     az_fits[fit_model] = fit_az
#     print('Done')
    
# print('Evaluating fit')
# comps = az.compare(az_fits)


# np.save(SAVE_DIR+'mu_u.npy', fit.extract('mu_u')['mu_u'])
# np.save(SAVE_DIR+'mu_l.npy', fit.extract('mu_l')['mu_l'])
# np.save(SAVE_DIR+'mu_d_u.npy', fit.extract('mu_d_u')['mu_d_u'])
# np.save(SAVE_DIR+'mu_d_l.npy', fit.extract('mu_d_l')['mu_d_l'])

