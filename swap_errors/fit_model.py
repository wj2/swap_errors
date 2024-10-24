import socket
import os
import sys
import pickle as pkl

import socket
import os
import sys

if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/kelarion/github/'
    SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'
else:    
    # CODE_DIR = '/rigel/home/ma3811/repler/'
    # SAVE_DIR = '/rigel/theory/users/ma3811/'
    CODE_DIR = '/burg/home/ma3811/repler/'
    SAVE_DIR = '/burg/theory/users/ma3811/'

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


def fit_models(stan_data, stan_models, **hmc_args):


    fits = []
    az_fits = {k:None for k in fit_models}
    for fit_model in stan_models:
        
        print('Fitting %s model'%fit_model)
        model = pkl.load(open(CODE_DIR+'assignment_errors/%s_model.stan'%fit_model,'rb'))
        
        fit = model.sampling(data=stan_data, **hmc_args)

        model_params = {'observed_data':'y',
                        'log_likelihood':{'y':'log_lik'},
                        'posterior_predictive':'err_hat'}
        fit_az = az.from_pystan(posterior=fit, **model_params)
                                
        fits.append(fit)
        az_fits[fit_model] = fit_az
        
    comps = az.compare(az_fits)

    return comps, fits
