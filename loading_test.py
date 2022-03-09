import socket
import os
import sys

import re
import pickle as pkl

import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm, manifold, linear_model
from tqdm import tqdm

sys.path.append('C:/Users/mmall/Documents/github/repler/src/')
import students
import assistants
import util
import experiments as exp

sys.path.append('C:/Users/mmall/Documents/github/general-neural/')
import general.data_io as gio
import general.utility as u
import swap_errors.aux_ as swa
import swap_errors.analysis as swan
# import swap_errors.visualization as swv
import general.neural_analysis as na
import general.plotting as gpl

BASE_DIR = 'C:/Users/mmall/Documents/uni/columbia/assignment_errors/'

#%%
busch_bhv_fields = ('StopCondition', 'ReactionTime', 'Block',
                    'is_one_sample_displayed', 'IsUpperSample',
                    'TargetTheta', 'DistTheta', 'ResponseTheta',
                    'LABthetaTarget', 'LABthetaDist', 'LABthetaResp',
                    'CueDelay', 'CueDelay2', 'CueRespDelay', 'FIXATE_ON_diode',
                    'CUE1_ON_diode', 'SAMPLES_ON_diode', 'CUE2_ON_diode',
                    'WHEEL_ON_diode')

bmp = 'swap_errors/behavioral_model/corr_swap_guess.pkl'
default_prior_dict = {'report_var_var_mean':1,
                      'report_var_var_var':3,
                      'report_var_mean_mean':.64,
                      'report_var_mean_var':1,
                      'swap_weight_var_mean':1,
                      'swap_weight_var_var':3,
                      'swap_weight_mean_mean':0,
                      'swap_weight_mean_var':1,
                      'guess_weight_var_mean':1,
                      'guess_weight_var_var':3,
                      'guess_weight_mean_mean':0,
                      'guess_weight_mean_var':1}

def load_bhv_data(fl, flname='bhv.mat', const_fields=('Date', 'Monkey'),
                  extract_fields=busch_bhv_fields):
    bhv = sio.loadmat(os.path.join(fl, flname))['bhv']
    const_dict = {cf:np.squeeze(bhv[cf][0,0]) for cf in const_fields}
    trl_dict = {}
    for tf in extract_fields:
        elements = bhv['Trials'][0,0][tf][0]
        for i, el in enumerate(elements):
            if len(el) == 0:
                elements[i] = np.array([[np.nan]])
        trl_dict[tf] = np.squeeze(np.stack(elements, axis=0))
    return const_dict, trl_dict


# def fit_bhv_model(data, model_path=bmp, targ_field='LABthetaTarget',
#                   dist_field='LABthetaDist', resp_field='LABthetaResp',
#                   prior_dict=None, stan_iters=2000, stan_chains=4,
#                   arviz=mixture_arviz, **stan_params):
#     if prior_dict is None:
#         prior_dict = default_prior_dict
#     targs_is = data[targ_field]
#     session_nums = np.array([], dtype=int)
#     for i, x in enumerate(targs_is):
#         sess = np.ones(len(x), dtype=int)*(i + 1)
#         session_nums = np.concatenate((session_nums,
#                                        sess))
#     targs = np.concatenate(targs_is, axis=0)
#     dists = np.concatenate(data[dist_field], axis=0)
#     resps = np.concatenate(data[resp_field], axis=0)
#     errs = u.normalize_periodic_range(targs - resps)
#     dist_errs = u.normalize_periodic_range(dists - resps)
#     dists_per = u.normalize_periodic_range(dists - targs)
#     stan_data = dict(T=dist_errs.shape[0], S=len(targs_is),
#                      err=errs, dist_err=dist_errs, run_ind=session_nums,
#                      dist_loc=dists_per, **prior_dict)
#     control = {'adapt_delta':stan_params.pop('adapt_delta', .8),
#                'max_treedepth':stan_params.pop('max_treedepth', 10)}
#     sm = pickle.load(open(model_path, 'rb'))
#     fit = sm.sampling(data=stan_data, iter=stan_iters, chains=stan_chains,
#                       control=control, **stan_params)
#     diag = ps.diagnostics.check_hmc_diagnostics(fit)
#     fit_av = az.from_pystan(posterior=fit, **arviz)
#     return fit, diag, fit_av, stan_data

#%%
st = []
which_neuron = []
chan = []
session_st = []

stim_on = []
which_trial = []
fix_time = []
is_up = []
is_one = []
block_type = []
session_bhv = []
m = 0
for i,g in tqdm(enumerate(os.listdir(BASE_DIR))):
    SAVE_DIR = BASE_DIR + g + '/'
    
    spk_files = [f for f in os.listdir(SAVE_DIR+'spikes/') if '-srt.mat' in f]
    
    for f in spk_files:
        ch = int(re.findall('chan(\d+)', f)[0])
        
        dat = sio.loadmat(SAVE_DIR+'spikes/' + f)
        if len(dat['ts'])>0:
            st.append(dat['ts'].squeeze())
            which_neuron.append(dat['id'].squeeze()+m)
            chan.append(np.ones(len(dat['ts']))*ch)
            session_st.append(np.ones(len(dat['ts']))*i)
            m += dat['id'].squeeze().max()
    
    _, bhv = load_bhv_data(SAVE_DIR+'bhv/')
    
    stim_on.append(bhv['SAMPLES_ON_diode'])
    fix_time.append(bhv['FIXATE_ON_diode'])
    is_up.append(bhv['IsUpperSample'])
    is_one.append(bhv['is_one_sample_displayed'])
    block_type.append(bhv['Block'])
    which_trial.append(np.arange(len(bhv['is_one_sample_displayed'])))
    session_bhv.append(np.ones(len(bhv['is_one_sample_displayed']))*i)

st = np.concatenate(st)
which_neuron = np.concatenate(which_neuron)
chan = np.concatenate(chan)
session_st = np.concatenate(session_st)

stim_on = np.concatenate(stim_on)
fix_time = np.concatenate(fix_time)
is_up = np.concatenate(is_up)
is_one = np.concatenate(is_one)
block_type = np.concatenate(block_type)
which_trial = np.concatenate(which_trial)
session_bhv = np.concatenate(session_bhv)

#%%
these_tri = (is_one==1) & ~np.isnan(stim_on)
# these_tri = (is_one==0) & ~np.isnan(stim_on)

num_tri = [these_tri[session_bhv==i].sum() for i in np.unique(session_bhv)]

n_neur = len(np.unique(which_neuron))
n_row = int(np.sqrt(n_neur))
n_col = int(np.ceil(n_neur/n_row))

plt.figure()
for j,neur in enumerate(np.unique(which_neuron)):
    plt.subplot(n_row,n_col,j+1)
    psth_up = (np.concatenate([st[(st>t-2)&(st<t+2)&(which_neuron==neur)]-t \
                            for t in stim_on[these_tri & (is_up==1)]]))
    
    psth_down = (np.concatenate([st[(st>t-2)&(st<t+2)&(which_neuron==neur)]-t \
                            for t in stim_on[these_tri & (is_up==0)]]))
    
    plt.hist(psth_up, density=True, alpha=0.5)
    plt.hist(psth_down, density=True, alpha=0.5)
    plt.plot([0,0],plt.ylim(),'k--')
    plt.xticks([])
    plt.yticks([])

#%%
valid_trials = ~np.isnan(fix_time)&(is_one==1)
# valid_trials = ~np.isnan(fix_time)&(is_one==0)
cond = is_up

# num_tri = np.max([np.min([valid_trials[session_bhv==i].sum() for i in np.unique(session_bhv)]), 20])
num_tri = [[valid_trials[(session_bhv==i)&(cond==j)].sum() for i in np.unique(session_bhv)] for j in np.unique(cond)]
# num_tri_sesh = [(session_bhv==i).sum() for i in np.unique(session_bhv)]
num_samp = np.max([np.min(num_tri), 20])

# trials = [[np.random.choice(which_trial[valid_trials&(session_bhv==i)&(cond==j)], num_samp, replace=False).astype(int) \
#                           for i in np.unique(session_bhv)] for j in np.unique(cond)]

# these_tri_st = np.isin(range(len(session_st)), trials)
# these_tri_bhv = np.isin(range(len(session_bhv)), trials)
# binned = np.array([[np.histogram(st[(which_neuron==i)&(st<(j+4))&(st>=j-1)]-j, bins=np.linspace(-1,4,50))[0] \
#                     for j in stim_on[these_tri]] \
#                    for i in np.unique(which_neuron)])

# pseudopop = np.array([[np.histogram(st[(which_neuron==i)&(st<(j+4))&(st>=j-1)]-j, bins=np.linspace(-1,4,50))[0] \
#                        for j in stim_on[these_tri]] \
#                       for i in np.unique(which_neuron)])

pseudopop = []
for i in tqdm(np.unique(session_bhv)):
    trial_idx = np.concatenate([np.random.choice(which_trial[valid_trials&(session_bhv==i)&(cond==j)], 
                                 num_samp, replace=False).astype(int) for j in np.unique(cond)])
    these_trials = np.isin(which_trial, trial_idx)&(session_bhv==i)
    act = [[np.histogram(st[(which_neuron==j)&(st<(k+4))&(st>=(k-1))&(session_st==i)]-k, bins=np.linspace(-1,4,50))[0] \
            for j in np.unique(which_neuron[session_st==i])] for k in stim_on[these_trials]]
    pseudopop.append(act)

binned = np.concatenate(pseudopop, axis=1).transpose((1,0,2))

print('Binned the data')

#%%
nT = binned.shape[2]
nTri = binned.shape[1]

labels = (np.arange(nTri)>(nTri//2)).astype(int)
# labels = np.random.permutation(np.arange(nTri)>(nTri//2)).astype(int)
times = (np.arange(nT)[None,:]*np.ones((nTri,1))).astype(int)

clf = assistants.LinearDecoder(binned.shape[0], 1, svm.LinearSVC)
clf.fit(binned[:,:70,:].transpose(1,2,0), labels[:70,None,None]*np.ones((1,nT,1)), 
        t_=(np.arange(nT)[None,:]*np.ones((70,1))).astype(int))

perf = clf.test(binned[:,70:,:].transpose(1,2,0), labels[70:,None,None]*np.ones((1,nT,1)), 
                t_=(np.arange(nT)[None,:]*np.ones((nTri-70,1))).astype(int))


#%%

data = gio.Dataset.from_readfunc(swa.load_buschman_data, BASE_DIR, max_files=np.inf,seconds=True)

#%%
tbeg = -.5
tend = 1
twindow = .1
tstep = .05
n_folds = 10

um = data['StopCondition'] == 1
um = um.rs_and(data['is_one_sample_displayed'] == 1)
data_single = data.mask(um)

mask_c1 = data_single['IsUpperSample'] == 1
mask_c2 = data_single['IsUpperSample'] == 0


pseudo = True
repl_nan = False
tzf = 'SAMPLES_ON_diode'
min_trials = 10
pre_pca = .99

cat1 = data_single.mask(mask_c1)
cat2 = data_single.mask(mask_c2)

pop1, xs = cat1.get_populations(twindow, tbeg, tend, tstep,
                                skl_axes=True, repl_nan=repl_nan,
                                time_zero_field=tzf)
pop2, xs = cat2.get_populations(twindow, tbeg, tend, tstep,
                                skl_axes=True, repl_nan=repl_nan,
                                time_zero_field=tzf)

c1_n = cat1.get_ntrls()
c2_n = cat2.get_ntrls()
comb_n = data_single.combine_ntrls(c1_n, c2_n)
pop1 = data_single.make_pseudopop(pop1, comb_n, min_trials, 10, skl_axs=True)
pop2 = data_single.make_pseudopop(pop2, comb_n, min_trials, 10, skl_axs=True)

cv_perf = na.fold_skl(pop1[0], pop2[0], n_folds, model=svm.LinearSVC, params={'class_weight':'balanced'}, 
                      mean=False, pre_pca=pre_pca, shuffle=False,
                      impute_missing=repl_nan)

#%%
binned_trn = np.append(pop1[0],pop2[0], axis=2).squeeze()
binned_tst = np.append(pop1[1],pop2[1], axis=2).squeeze()

nT = binned_trn.shape[2]
nTri = binned_trn.shape[1] 

labels = (np.arange(nTri)>(pop1.shape[-2])).astype(int)
# labels = np.random.permutation(np.arange(nTri)>(nTri//2)).astype(int)
times = (np.arange(nT)[None,:]*np.ones((nTri,1))).astype(int)

clf = assistants.LinearDecoder(binned.shape[0], 1, svm.LinearSVC)
clf.fit(binned_trn[:,:,:].transpose(1,2,0), labels[:,None,None]*np.ones((1,nT,1)), 
        t_=(np.arange(nT)[None,:]*np.ones((nTri,1))).astype(int))

perf = clf.test(binned_tst[:,:,:].transpose(1,2,0), labels[:,None,None]*np.ones((1,nT,1)), 
                t_=(np.arange(nT)[None,:]*np.ones((nTri,1))).astype(int))




