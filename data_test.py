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
su.recompile_model(CODE_DIR+'assignment_errors/linear_interp_color_model.stan')

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

#%% Data loading

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

tbeg = 0
tend = 1
twindow = .5
tstep = .05
num_bins = 4

which_sess = 0

do_pca = True
pca_thrs = 0.95

# um = data['StopCondition'] == 1 
um = data['is_one_sample_displayed'] == 0
um = um.rs_and(data['Block']>1)
um = um.rs_and(data['corr_prob']>0.7)
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

xs = xs[:int((tend-tbeg)/tstep)+1]

x = pop[which_sess][...,xs>0].reshape((pop[which_sess].shape[0],-1)).T

if do_pca:
    x = util.pca_reduce(x, thrs=pca_thrs)
    print(x.shape)

yup = np.repeat(data_single['upper_color'][which_sess].array.to_numpy(), np.sum(xs>0))
ylow = np.repeat(data_single['lower_color'][which_sess].array.to_numpy(), np.sum(xs>0))
cue = np.repeat(data_single['IsUpperSample'][which_sess].array.to_numpy(), np.sum(xs>0))

bins = np.linspace(0,2*np.pi,num_bins+1)[:num_bins]

c_u = convexify(yup, bins).T
c_l = convexify(ylow, bins).T

stan_data = dict(T=x.shape[0], N=x.shape[-1], K=num_bins, y=x, C_u=c_u, C_l=c_l, cue=cue)

#%%

model = pkl.load(open(CODE_DIR+'assignment_errors/linear_interp_color_model.stan','rb'))

fit = model.sampling(data=stan_data, iter=2000, chains=4)


np.save(SAVE_DIR+'mu_u.npy', fit.extract('mu_u')['mu_u'])
np.save(SAVE_DIR+'mu_l.npy', fit.extract('mu_l')['mu_l'])
np.save(SAVE_DIR+'mu_d_u.npy', fit.extract('mu_d_u')['mu_d_u'])
np.save(SAVE_DIR+'mu_d_l.npy', fit.extract('mu_d_l')['mu_d_l'])

