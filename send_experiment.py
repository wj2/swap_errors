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
regions = {'frontal':['pfc','fef'],
		   'posterior':['v4pit','tpot','7ab'],
		   'decision':['pfc','fef','7ab'],
		   'sensory':['v4pit','tpot'],
		   'all': ['7ab', 'fef', 'motor', 'pfc', 'tpot', 'v4pit']}

these_dsets = []

these_dsets.append({'these_sess':list(range(23)),
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
					'shuffle_probs':True,
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

# these_dsets.append({'these_sess':list(range(23)),
# 					'regions':['all','frontal','posterior','decision','sensory'],
# 					'tzf':'CUE2_ON_diode',
# 					'tbeg':-0.5,
# 					'twindow':0.5,
# 					'tstep':0.5,
# 					'num_bins':6,
# 					'do_pca':'before', #'after'
# 					'pca_thrs':0.95,
# 					'min_trials':40,
# 					'shuffle':False,
# 					'impute_nan':True,
# 					'shuffle_probs':True,
# 					'impute_params':{'weights':'uniform','n_neighbors':5},
# 					'color_weights':'interpolated' # 'softmax'
# 					})

these_models = []
# these_models.append(['null_hierarchical','spatial_error_hierarchical','cue_error_hierarchical',
# 				'hybrid_error_hierarchical', 'super_hybrid_error_hierarchical'])
# these_models.append(['null_precue','spatial_error_precue','hybrid_error_precue'])
these_models.append(['super_hybrid_error_hierarchical'])
# these_models.append(['spatial_error_precue','hybrid_error_precue'])

### Assemble stan data dicts
##############################
if os.path.exists(SAVE_DIR+'server_cache/dset_params.pkl'):
	old_prms = pkl.load(open(SAVE_DIR+'server_cache/dset_params.pkl','rb'))
else:
	old_prms = {}

if these_dsets != old_prms:
	print('Overwriting old data ....')

	# loading the data -- takes the longest
	data = gio.Dataset.from_readfunc(swa.load_buschman_data, SAVE_DIR, max_files=np.inf,seconds=True, 
	                                  load_bhv_model=CODE_DIR+'/assignment_errors/bhv_model.pkl',
	                                  spks_template=swa.busch_spks_templ_mua)

	for i_dst, dset_prm in enumerate(these_dsets):
		## funky way of iterating over all the parameters in the dictionary
		variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='these_sess'}
		fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='these_sess'}

		these_sess = dset_prm['these_sess']
		var_k, var_v = zip(*variable_prms.items())

		idx = 0
		for vals in list(itt.product(*var_v)):
			this_dset = dict(zip(var_k, vals), **fixed_prms)
			this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

			# get the parameters for this particular dataset
			tzf = this_dset['tzf']
			tbeg = this_dset['tbeg']
			tend = this_dset['tend'] 
			twindow = this_dset['twindow']
			tstep = this_dset['tstep']
			num_bins = this_dset['num_bins']
			do_pca = this_dset['do_pca']
			pca_thrs = this_dset['pca_thrs']
			min_trials = this_dset['min_trials']
			shuffle = this_dset['shuffle']
			impute_nan = this_dset['impute_nan']
			impute_params = this_dset['impute_params']
			color_weights = this_dset['color_weights']

			if color_weights == 'interpolated':
				spline_func = helpers.convexify
			elif color_weights == 'softmax':
				spline_func = helpers.softmax_cols


			# maybe this should be made more flexible ...
			um = data['is_one_sample_displayed'] == 0
			um = um.rs_and(data['Block']>1)
			um = um.rs_and(data['corr_prob']<2) # avoid nan trials
			# if 
			data_single = data.mask(um)

			n = data_single.get_ntrls()

			pop, xs = data_single.get_populations(twindow, tbeg, tend, tstep,
			                                      skl_axes=True, repl_nan=False,
			                                      time_zero_field=this_dset['tzf'])   

			xs = xs[:int((tend-tbeg)/tstep)+1]

			for i, which_sess in enumerate(these_sess):
				print('Assembling population %d, %d ...'%(i_dst,idx))

				x = pop[which_sess].sum(-1).reshape((pop[which_sess].shape[0],-1))
				in_area = np.isin(data_single['neur_regions'][which_sess].values[0], regions[this_dset['regions']])
				x = x[in_area, :]

				if impute_nan:
					impute = (helpers.box_conv(x==0, 50)==50).max(0)
					trash = impute.mean(1) > 0.5
					x = x[~trash,:]
					impute = impute[~trash,:]
					imptr = knni(**impute_params)
					x = imptr.fit_transform(np.where(impute,np.nan,x))
				num_neur = x.shape[0]

				# z-score
				if do_pca == 'before':
					x = util.pca_reduce(x, thrs=pca_thrs)

				x -= x.mean(0,keepdims=True)
				x /= (x.std(0, keepdims=True)+1e-8)

				if do_pca == 'after':
					x = util.pca_reduce(x, thrs=pca_thrs)

				# yup = np.repeat(data_single['upper_color'][which_sess].array.to_numpy(), np.sum(xs>0))
				# ylow = np.repeat(data_single['lower_color'][which_sess].array.to_numpy(), np.sum(xs>0))
				# cue = np.repeat(data_single['IsUpperSample'][which_sess].array.to_numpy(), np.sum(xs>0))

				yup = data_single['upper_color'][which_sess].array.to_numpy()
				ylow = data_single['lower_color'][which_sess].array.to_numpy()
				cue = data_single['IsUpperSample'][which_sess].array.to_numpy()

				bins = np.linspace(0,2*np.pi,num_bins+1)[:num_bins]

				c_u = spline_func(yup, bins).T
				c_l = spline_func(ylow, bins).T

				probs = np.stack([data_single[tt][which_sess].array.to_numpy() \
					for  tt in ['corr_prob','swap_prob','guess_prob']]).T

				if this_dset['shuffle_probs']:
					probs = probs[np.random.permutation(len(probs)),:]

				stan_data = dict(T=x.shape[0], N=x.shape[-1], K=num_bins, y=x, C_u=c_u, C_l=c_l, 
					cue=cue, p=probs, num_neur=num_neur)

				dset_info = {'session':which_sess, 'stan_data':stan_data, **this_dset}
				pkl.dump(dset_info, open(SAVE_DIR+'server_cache/dataset_%d_%d.pkl'%(i_dst,idx),'wb'))
				idx += 1

	pkl.dump(these_dsets, open(SAVE_DIR+'server_cache/dset_params.pkl','wb'))


##### Make parameter dicts
##########################################

hmc_args = {'iter':2000, 
			'chains':4}

for i_dst, dset_models in enumerate(these_models):
	for j, mod in enumerate(dset_models):
		params = {'stan_model': '%s_model.stan'%mod, 'hmc_args':hmc_args}
		pkl.dump(params, open(SAVE_DIR+'server_cache/params_%d_%d.pkl'%(i_dst,j),'wb'))

		if not os.path.exists(CODE_DIR + 'assignment_errors/' + mod+'_model.pkl'):
			print('Need to compile %s model'%mod)
			su.recompile_model(CODE_DIR + 'assignment_errors/' + mod+'_model.pkl')

	###### Send to pickles server
	##########################################
	if 'columbia' in REMOTE_SYNC_SERVER:
		print('[{}] Giving files to {}...'.format(sys.platform, REMOTE_SYNC_SERVER))

		cmd = 'rsync {local}*.pkl {remote} -v'.format(local=SAVE_DIR+'server_cache/',
			remote=REMOTE_SYNC_SERVER+':'+REMOTE_RESULTS)
		subprocess.check_call(cmd, shell=True)

		cmd = 'rsync {local}*.pkl {remote} -v'.format(local=CODE_DIR + 'assignment_errors/',
			remote=REMOTE_SYNC_SERVER+':'+REMOTE_CODE)
		subprocess.check_call(cmd, shell=True)

	####### Run job array
	###########################################
	# n_sess = len(these_dsets[i_dst]['these_sess'])*len(list(itt.product(*var_v)))
	n_sess = np.prod([len(v) for k,v in these_dsets[i_dst].items() if type(v) is list])
	n_mod = len(dset_models)

	print('\nSending %d jobs to server ...\n'%(n_sess*n_mod))

	# update file to have correct array indices
	tmplt_file = open(CODE_DIR+'assignment_errors/job_script_template.sh','r')
	with open(SAVE_DIR+'server_cache/job_script_%d.sh'%i_dst,'w') as script_file:
		sbatch_text = tmplt_file.read().format(n_tot=n_sess*n_mod - 1, n_dat=n_sess, dset_idx=i_dst, file_dir=REMOTE_CODE)
		script_file.write(sbatch_text)
	tmplt_file.close()

	## run job
	if 'columbia' in REMOTE_SYNC_SERVER:
		cmd = "ssh ma3811@ginsburg.rcs.columbia.edu 'sbatch -s' < {}".format(SAVE_DIR+'server_cache/job_script_%d.sh'%i_dst)
		subprocess.call(cmd, shell=True)
