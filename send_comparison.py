CODE_DIR = '/home/kelarion/github/'
SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'

REMOTE_SYNC_SERVER = 'ma3811@motion.rcs.columbia.edu' #must have ssh keys set up
REMOTE_CODE = '/burg/home/ma3811/assignment_errors/'
REMOTE_RESULTS = '/burg/theory/users/ma3811/assignment_errors/'
# REMOTE_SYNC_SERVER = 'kelarion@kelarion' #must have ssh keys set up
# REMOTE_CODE = '/home/kelarion/github/'
# REMOTE_RESULTS = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'

import socket
import os
import sys
import pickle as pkl
import subprocess

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx
from sklearn.impute import KNNImputer as knni
from tqdm import tqdm
import itertools as itt
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

import helpers

sys.path.append(CODE_DIR+'repler/src/')
import util


### Set experiment parameters
##############################
these_dsets = []

these_dsets.append({'session':list(range(23)),
					'regions':['all','frontal','posterior','decision','sensory'],
					'tzf': 'WHEEL_ON_diode',
					'tbeg':-0.5,
					'twindow':0.5,
					'tstep':0.5,
					'num_bins':6,
					'do_pca':'before', #'after'
					'pca_thrs':0.95,
					'min_trials':40,
					'shuffle':False,
					'impute_nan':True,
					'impute_params':{'weights':'uniform','n_neighbors':5},
					'color_weights':'interpolated' # 'softmax'
					})

# these_dsets.append({'these_sess':[3,5],
# 					'tzf':'CUE2_ON_diode',
# 					'tbeg':0,
# 					'twindow':0.5,
# 					'tstep':0.5,
# 					'num_bins':[4,5,6,7,8],
# 					'do_pca':'before', #'after'
# 					'pca_thrs':0.95,
# 					'min_trials':40,
# 					'shuffle':False,
# 					'impute_nan':True,
# 					'impute_params':{'weights':'uniform','n_neighbors':5},
# 					'color_weights':'interpolated' # 'softmax'
# 					})

these_dsets.append({'session':list(range(23)),
					'regions':['all','frontal','posterior','decision','sensory'],
					'tzf':'CUE2_ON_diode',
					'tbeg':-0.5,
					'twindow':0.5,
					'tstep':0.5,
					'num_bins':6,
					'do_pca':'before', #'after'
					'pca_thrs':0.95,
					'min_trials':40,
					'shuffle':False,
					'impute_nan':True,
					'impute_params':{'weights':'uniform','n_neighbors':5},
					'color_weights':'interpolated' # 'softmax'
					})


### Assemble stan data dicts
##############################

for i_dst, dset_prm in enumerate(these_dsets):
	## funky way of iterating over all the parameters in the dictionary
	variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list}
	fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list}

	var_k, var_v = zip(*variable_prms.items())

	for idx, vals in enumerate(itt.product(*var_v)):
		this_dset = dict(zip(var_k, vals), **fixed_prms)
		this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

		pkl.dump(this_dset, open(SAVE_DIR+'server_cache/dataset_cv_%d_%d.pkl'%(i_dst,idx),'wb'))

	###### Send to pickles server
	##########################################
	if 'columbia' in REMOTE_SYNC_SERVER:
		print('[{}] Giving files to {}...'.format(sys.platform, REMOTE_SYNC_SERVER))

		cmd = 'rsync {local}*.pkl {remote} -v'.format(local=SAVE_DIR+'server_cache/',
			remote=REMOTE_SYNC_SERVER+':'+REMOTE_RESULTS)
		subprocess.check_call(cmd, shell=True)

	####### Run job array
	###########################################
	n_sess = np.prod([len(v) for k,v in dset_prm.items() if type(v) is list])

	print('\nSending %d jobs to server ...\n'%(n_sess))

	# update file to have correct array indices
	tmplt_file = open(CODE_DIR+'assignment_errors/cv_script_template.sh','r')
	with open(SAVE_DIR+'server_cache/cv_script_%d.sh'%i_dst,'w') as script_file:
		sbatch_text = tmplt_file.read().format(n_tot=n_sess - 1, dset_idx=i_dst, file_dir=REMOTE_CODE)
		script_file.write(sbatch_text)
	tmplt_file.close()

	## run job
	if 'columbia' in REMOTE_SYNC_SERVER:
		cmd = "ssh ma3811@ginsburg.rcs.columbia.edu 'sbatch -s' < {}".format(SAVE_DIR+'server_cache/cv_script_%d.sh'%i_dst)
		subprocess.call(cmd, shell=True)
