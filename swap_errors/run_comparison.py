import socket
import os
import sys

if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/kelarion/github/'
    SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'
    LOAD_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/server_cache/'
    repler_DIR = 'repler/src'
else:    
    # CODE_DIR = '/rigel/home/ma3811/repler/'
    # SAVE_DIR = '/rigel/theory/users/ma3811/'  
    CODE_DIR = '/burg/home/ma3811/'
    SAVE_DIR = '/burg/theory/users/ma3811/assignment_errors/'
    LOAD_DIR = SAVE_DIR
    repler_DIR = 'repler/'
    openmind = False

import pickle as pkl
import re

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx
import pystan as ps
import arviz as az

sys.path.append(CODE_DIR+'assignment_errors/')
sys.path.append(CODE_DIR+'assignment_errors/jeffcode/')
import general.data_io as gio
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan
# import swap_errors.visualization as swv
import general.neural_analysis as na
import general.plotting as gpl
import general.stan_utility as su

import helpers as hlp

sys.path.append(CODE_DIR+repler_DIR)
import util


# get the indices
allargs = sys.argv
data_idx = int(allargs[1])
dset_idx = int(allargs[2])

with open(LOAD_DIR+'dataset_cv_%d_%d.pkl'%(dset_idx, data_idx), 'rb') as fil:
    data_dict = pkl.load(fil)

####  Load models  ######
##########################################################

print('Comparing models')

folds = hlp.folder_hierarchy(data_dict)

model_files = [f for f in os.listdir(SAVE_DIR+folds) if 'arviz_fit_' in f]
these_models = [re.findall('arviz_fit_(.+)_model.pkl', f)[0] for f in model_files]

fits_az = {p:None for p in these_models}
for modfil, mod in zip(model_files, these_models):
    with open(SAVE_DIR+folds+modfil, 'rb') as fil:
        fits_az[mod] = pkl.load(fil)
    loo = az.loo(fits_az[mod], pointwise=True)
    with open(SAVE_DIR+folds+'cv_%s_model.pkl'%mod, 'wb') as fil:
        pkl.dump(loo, fil)

comp = az.compare(fits_az)

####  Save comparison  ######
##########################################################
print('Saving ')

with open(SAVE_DIR+folds+'model_comparisons.pkl', 'wb') as fil:
    pkl.dump(comp, fil)