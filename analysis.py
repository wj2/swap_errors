
import numpy as np
import pickle
import pystan as ps
import arviz as az
import sklearn.manifold as skm

import general.neural_analysis as na
import general.utility as u

def nonlinear_dimred(data, tbeg, tend, twindow=None, tstep=None,
                     time_key='SAMPLES_ON_diode', color_key='LABthetaTarget',
                     regions=None, dim_red=skm.LocallyLinearEmbedding,
                     n_components=2, **kwargs):
    if twindow is not None and tstep is not None:
        pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                        time_zero_field=time_key)
    else:
        pops, xs = data.get_populations(tend - tbeg, tbeg, tend,
                                        time_zero_field=time_key)
    colors = data[color_key]
    emb_pops = []
    drs = []
    for i, pop in enumerate(pops):
        if twindow is not None and tstep is not None:
            pop_swap = np.swapaxes(pop, 0, 1)
            pop_all = np.concatenate(pop_swap, axis=0)
        else:
            pop = pop[..., :1]
            pop_all = np.squeeze(pop)
            xs = xs[:1]
        dr = dim_red(n_components=n_components, **kwargs)
        dr.fit(pop_all)
        emb_pop = np.zeros((pop.shape[0], n_components, len(xs)))
        for j in range(len(xs)):
            emb_pop[..., j] = dr.transform(pop[..., j])
        emb_pops.append(emb_pop)
        drs.append(dr)
    return emb_pops, drs, colors, xs

def decode_color(data, tbeg, tend, twindow, tstep, time_key='SAMPLES_ON_diode',
                 color_key='LABthetaTarget', regions=None, n_folds=10,
                 transform_color=True, **kwargs):
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                    time_zero_field=time_key, skl_axes=True)
    regs = data[color_key]
    outs = []
    for i, pop in enumerate(pops):
        if transform_color:
            regs_i = np.stack((np.cos(regs[i]), np.sin(regs[i])), axis=1)
        else:
            regs_i = regs[i]
        tcs = na.pop_regression_skl(pop, regs_i, n_folds, mean=False,
                                    **kwargs)
        outs.append(tcs)
    return outs, xs

def single_neuron_color(data, tbeg, tend, twindow, tstep,
                        time_key='SAMPLES_ON_diode',
                        color_key='LABthetaTarget', neur_chan='neur_channels',
                        neur_id='neur_ids', neur_region='neur_regions'):
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                    time_zero_field=time_key)
    regs = data[color_key]
    outs = {}
    for i, pop in enumerate(pops):
        for j in range(pop.shape[1]):
            val = (pop[:, j], regs[i])
            k1 = data[neur_chan][i].iloc[0][j]
            k2 = data[neur_id][i].iloc[0][j]
            k3 = data[neur_region][i].iloc[0][j]
            outs[(k1, k2, k3)] = val
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
    return fit, diag, fit_av, stan_data
