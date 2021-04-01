
import numpy as np
import pickle
import pystan as ps
import arviz

import general.neural_analysis as na
import general.utility as u

def decode_color(data, tbeg, tend, twindow, tstep, time_key='SAMPLES_ON_diode',
                 color_key='TargetTheta', regions=None, n_folds=10, **kwargs):
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                    time_zero_field=time_key, skl_axes=True)
    regs = data[color_key]
    outs = []
    for i, pop in enumerate(pops):
        tcs = na.pop_regression_skl(pop, regs[i], n_folds, mean=False,
                                    **kwargs)
        outs.append(tcs)
    return outs, xs

mixture_arviz = {'observed_data':'err',
                 'log_likelihood':{'err':'log_lik'},
                 'posterior_predictive':'err_hat',
                 'dims':{'report_var':['run_ind'],
                         'swap_prob':['run_ind'],
                         'guess_prob':['run_ind']}}

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
def fit_bhv_model(data, model_path=bmp, targ_field='LABthetaTarget',
                  dist_field='LABthetaDist', resp_field='LABthetaResp',
                  prior_dict=None, stan_iters=2000, stan_chains=4,
                  arviz=mixture_arviz, **stan_params):
    if prior_dict is None:
        prior_dict = default_prior_dict
    targs_is = data[targ_field]
    session_nums = np.array([], dtype=int)
    for i, x in enumerate(targs_is):
        sess = np.ones(len(x), dtype=int)*(i + 1)
        session_nums = np.concatenate((session_nums,
                                       sess))
    targs = np.concatenate(targs_is, axis=0)
    dists = np.concatenate(data[dist_field], axis=0)
    resps = np.concatenate(data[resp_field], axis=0)
    errs = u.normalize_periodic_range(targs - resps)
    dist_errs = u.normalize_periodic_range(dists - resps)
    dists_per = u.normalize_periodic_range(dists - targs)
    stan_data = dict(T=dist_errs.shape[0], S=len(targs_is),
                     err=errs, dist_err=dist_errs, run_ind=session_nums,
                     dist_loc=dists_per, **prior_dict)
    control = {'adapt_delta':stan_params.pop('adapt_delta', .8),
               'max_treedepth':stan_params.pop('max_treedepth', 10)}
    sm = pickle.load(open(model_path, 'rb'))
    fit = sm.sampling(data=stan_data, iter=stan_iters, chains=stan_chains,
                      control=control, **stan_params)
    diag = ps.diagnostics.check_hmc_diagnostics(fit)
    fit_av = az.from_pystan(posterior=fit, **arviz)
    return fit, diag, fit_av
