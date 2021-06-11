
import numpy as np
import pickle
import pystan as ps
import arviz as az
import scipy.spatial.distance as spsd
import sklearn.manifold as skm
import sklearn.neighbors as skn
import sklearn.model_selection as skms
import sklearn.svm as skc
import scipy.stats as sts
# import ripser as r

import general.data_io as dio
import general.neural_analysis as na
import general.utility as u
import general.decoders as gd

def nonlinear_dimred(data, tbeg, tend, twindow=None, tstep=None,
                     time_key='SAMPLES_ON_diode', color_key='LABthetaTarget',
                     regions=None, dim_red=skm.LocallyLinearEmbedding,
                     n_components=2, **kwargs):
    if twindow is None:
        twindow = tend - tbeg
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
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

def compute_persistent_homology_pops(data, tbeg=.05, tend=.55, twindow=.1,
                                     tstep=None, time_key='SAMPLES_ON_diode',
                                     regions=None, max_pops=np.inf,
                                     **kwargs):
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                    time_zero_field=time_key)
    bettis = []
    for i, pop in enumerate(pops):
        if i >= max_pops:
            break
        pop_flat = np.concatenate(list((pop[..., j]
                                        for j in range(pop.shape[-1]))),
                                  axis=0)
        bs = compute_persistent_homology(pop_flat, **kwargs)
        bettis.append(bs)
    return bettis

def compute_persistent_homology(pop, dim_red=10, dim_red_method=skm.Isomap,
                                neighborhood=True, n_neighbors=5,
                                thresh_percent=20):
    if pop.shape[1] > dim_red:
        dr = dim_red_method(n_components=dim_red, n_neighbors=n_neighbors)
        pop = dr.fit_transform(pop)
    if neighborhood:
        dists = spsd.pdist(pop)
        rad = np.percentile(dists, 1)
        neigh = skn.NearestNeighbors()
        neigh.fit(pop)
        neigh_dists = neigh.radius_neighbors(X=pop, radius=rad,
                                             return_distance=False)
        num_nbrs = np.array(list(map(len, neigh_dists)))
        threshold = np.percentile(num_nbrs, thresh_percent)
        pop = pop[num_nbrs > threshold]
    results = {}
    barcodes = r.ripser(pop, maxdim=1, coeff=2)['dgms']
    results['h0'] = barcodes[0]
    results['h1'] = barcodes[1]

    results['h2'] = r.ripser(pop, maxdim=2, coeff=2)['dgms'][2]
    return results


def decode_fake_data(n_times, n_neurons, n_trials, n_colors, noise_std=.1):
    cols = np.linspace(0, 2*np.pi, n_colors)
    x = np.sin(cols)
    y = np.cos(cols)
    x_code_m = sts.norm(0, 10).rvs((n_neurons, 1))
    x_code = x_code_m + sts.norm(0, 1).rvs((1, n_times))
    y_code_m = sts.norm(0, 10).rvs((n_neurons, 1))
    y_code = x_code_m + sts.norm(0, 1).rvs((1, n_times))
    resp_x = np.expand_dims(x_code, -1)*np.expand_dims(x, (0, 1))
    resp_y = np.expand_dims(y_code, -1)*np.expand_dims(y, (0, 1))
    resp_m = resp_x + resp_y
    t_inds = np.random.choice(range(n_colors), n_trials)
    resps = resp_m[:, :, t_inds]
    resps = resps + sts.norm(0, noise_std).rvs(resps.shape)
    resps = np.swapaxes(resps, 1, 2)
    resps = np.expand_dims(resps, 1)
    cols = cols[t_inds]
    resps[..., 0] = sts.norm(0, noise_std).rvs(resps.shape[:-1])
    out = na.pop_regression_stan(resps, cols, model=gd.PeriodicDecoderTF)
    xs = np.arange(n_times)
    return out, xs

def _get_cmean(trls, trl_cols, targ_col, all_cols, color_window=.2):
    col_dists = np.abs(u.normalize_periodic_range(targ_col - all_cols))
    col_mask = col_dists < color_window
    use_cols = all_cols[col_mask]
    trial_mask = np.isin(trl_cols, use_cols)
    pop_col = trls[:, 0, trial_mask]
    m = np.nanmean(pop_col, axis=1)
    return m   

def _get_trl_dist(trl, m, norm_neurons=True):
    d = np.sqrt(np.nansum((trl - m)**2, axis=0))/m.shape[0]
    return d

def _get_leftout_color_dists(pop_i, targ_cols, dist_cols,
                             splitter=skms.LeaveOneOut, norm=True,
                             u_cols=None, color_window=.2, norm_neurons=True,
                             return_norm=False):
    cols_arr = np.stack((np.array(targ_cols), np.array(dist_cols)), axis=1)
    if u_cols is None:
        u_cols = np.unique(cols_arr)
    if norm:
        m = np.mean(pop_i, axis=2, keepdims=True)
        s = np.std(pop_i, axis=2, keepdims=True)
        s[np.isnan(s)] = 1
        pop_i = (pop_i - m)/s
    spl = splitter()
    n_splits = spl.get_n_splits(cols_arr)
    targ_means = np.zeros(n_splits, dtype=object)
    dist_means = np.zeros(n_splits, dtype=object)
    for i, (tr_ind, te_ind) in enumerate(spl.split(cols_arr)):
        pop_train = pop_i[:, :, tr_ind]
        cols_train = cols_arr[tr_ind]
        pop_test = pop_i[:, :, te_ind]
        cols_test = cols_arr[te_ind]
        if i == 0:
            targ_dists = np.zeros((n_splits, len(te_ind), 2, pop_test.shape[3]))
            dist_dists = np.zeros_like(targ_dists)
        targ_means[i] = {}
        dist_means[i] = {}
        for uc in u_cols:
            targ_means[i][uc] = _get_cmean(pop_train, cols_train[:, 0], uc,
                                           u_cols, color_window=color_window)
            dist_means[i][uc] = _get_cmean(pop_train, cols_train[:, 1], uc,
                                           u_cols, color_window=color_window)
        for j in range(pop_test.shape[2]):
            tc = cols_test[j, 0]
            dc = cols_test[j, 1]
            targ_dists[i, j, 0] = _get_trl_dist(pop_test[:, 0, j],
                                                targ_means[i][tc],
                                                norm_neurons=norm_neurons)
            targ_dists[i, j, 1] = _get_trl_dist(pop_test[:, 0, j],
                                                targ_means[i][dc],
                                                norm_neurons=norm_neurons)
            dist_dists[i, j, 0] = _get_trl_dist(pop_test[:, 0, j],
                                                dist_means[i][tc],
                                                norm_neurons=norm_neurons)
            dist_dists[i, j, 1] = _get_trl_dist(pop_test[:, 0, j],
                                                dist_means[i][dc],
                                                norm_neurons=norm_neurons)
    out = targ_dists, dist_dists, targ_means, dist_means
    if return_norm and norm:
        out = out + (m, s)
    return out

def get_test_color_dists(data, tbeg, tend, twindow, tstep, 
                         targ_means, dist_means,
                         time_key='SAMPLES_ON_diode',
                         targ_key='LABthetaTarget',
                         dist_key='LABthetaDist',
                         resp_field='LABthetaResp',
                         regions=None, norm=True,
                         norm_neurons=True, m=None, s=None,
                         err_thr=2):
    targ_cols = data[targ_key]
    dist_cols = data[dist_key]
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                    time_zero_field=time_key,
                                    skl_axes=True)
    if m is None:
        m = (None,)*len(pops)
        s = (None,)*len(pops)
    out_dists = []
    for i, pop_i in enumerate(pops):
        tc_i = np.array(targ_cols[i])
        dc_i = np.array(dist_cols[i])
        if err_thr is not None:
            resps = np.array(data[resp_field][i])
            err = u.normalize_periodic_range(tc_i - resps)
            mask = np.abs(err) > err_thr
            pop_i = pop_i[:, :, mask]
            dc_i = dc_i[mask]
            tc_i = tc_i[mask]
        out = compute_dists(pop_i, tc_i, dc_i,
                            targ_means[i],
                            norm_neurons=norm_neurons,
                            mean=m[i], std=s[i])
        out_dists.append(out)
    return out_dists
    
def compute_dists(pop_test, trl_targs, trl_dists, col_means,
                  norm_neurons=True, mean=None, std=None):
    if mean is not None:
        pop_test = (pop_test - mean)/std
    n_models = len(col_means)
    n_tests = pop_test.shape[2]
    out_dists = np.zeros((n_models, n_tests, 2, pop_test.shape[3]))
    for i in range(n_models):
        for j in range(pop_test.shape[2]):
            tc = trl_targs[j]
            dc = trl_dists[j]
            out_dists[i, j, 0] = _get_trl_dist(pop_test[:, 0, j],
                                               col_means[i][tc],
                                               norm_neurons=norm_neurons)
            out_dists[i, j, 1] = _get_trl_dist(pop_test[:, 0, j],
                                               col_means[i][dc],
                                               norm_neurons=norm_neurons)
    return out_dists

def get_color_means(data, tbeg, tend, twindow, tstep, color_window=.2,
                    time_key='SAMPLES_ON_diode',
                    targ_key='LABthetaTarget',
                    dist_key='LABthetaDist',
                    regions=None, leave_out=0,
                    norm=True, norm_neurons=True,
                    test_data=None):
    targ_cols = data[targ_key]
    dist_cols = data[dist_key]
    u_cols = np.unique(np.concatenate(targ_cols))
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                    time_zero_field=time_key,
                                    skl_axes=True)
    targ_dists_all = []
    dist_dists_all = []
    targ_means_all = []
    dist_means_all = []
    means = []
    stds = []
    for i, pop_i in enumerate(pops):
        out = _get_leftout_color_dists(pop_i, targ_cols[i], dist_cols[i],
                                       norm=norm, u_cols=u_cols,
                                       color_window=color_window,
                                       return_norm=True)
        targ_dists, dist_dists, targ_means, dist_means, m, s = out
        means.append(m)
        stds.append(s)
        targ_dists_all.append(targ_dists)
        dist_dists_all.append(dist_dists)
        targ_means_all.append(targ_means)
        dist_means_all.append(dist_means)
    out_dists = (targ_dists_all, dist_dists_all)
    if test_data is not None:
        test_dists = get_test_color_dists(test_data, tbeg, tend, twindow, tstep,
                                          targ_means_all, dist_means_all,
                                          time_key=time_key, dist_key=dist_key,
                                          regions=regions, norm=norm,
                                          norm_neurons=norm_neurons, m=means,
                                          s=stds)
        out_dists = out_dists + (test_dists,)
    out_means = (targ_means_all, dist_means_all)
    out = out_dists, out_means, xs
    return out

def decode_color(data, tbeg, tend, twindow, tstep, time_key='SAMPLES_ON_diode',
                 color_key='LABthetaTarget', regions=None, n_folds=10,
                 transform_color=True, model=skc.SVR, n_jobs=-1,
                 use_stan=True, max_pops=np.inf, time=True, pseudo=False,
                 min_trials_pseudo=1, resample_pseudo=10, **kwargs):
    regs = data[color_key]
    if pseudo:
        regs = np.concatenate(regs)
        u_cols = np.unique(regs)
        pis = []
        ns = []
        for i, uc in enumerate(u_cols):
            mi = data[color_key] == uc
            di = data.mask(mi)
            pi, xs = di.get_populations(twindow, tbeg, tend, tstep,
                                        time_zero_field=time_key,
                                        skl_axes=True)
        
            ns.append(di.get_ntrls())
            pis.append(pi)
        comb_n = dio.combine_ntrls(*ns)
        new_pis = []
        cols = []
        for i, pi in enumerate(pis):
            pi = data.make_pseudopop(pi, comb_n, min_trials_pseudo,
                                     resample_pseudo, skl_axs=True)
            new_pis.append(pi)
            cols.append((u_cols[i],)*pi.shape[3])
        pops = np.concatenate(new_pis, axis=3)
        cols_cat = np.concatenate(cols)
        regs = (cols_cat,)*resample_pseudo
    else:
        pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                        time_zero_field=time_key,
                                        skl_axes=True)
    outs = []
    for i, pop in enumerate(pops):
        if i >= max_pops:
            break
        if transform_color:
            regs_i = np.stack((np.cos(regs[i]), np.sin(regs[i])), axis=1)
        else:
            regs_i = np.array(regs[i])
        if use_stan and not time:
            out = na.pop_regression_stan(pop, regs_i, model=model, **kwargs)
        elif use_stan and time:
            out = na.pop_regression_timestan(pop, regs_i, **kwargs)   
        else:
            out = na.pop_regression_skl(pop, regs_i, n_folds, mean=False,
                                        model=model, n_jobs=n_jobs, **kwargs)
        outs.append(out)
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
