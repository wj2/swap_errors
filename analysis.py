
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
import itertools as it
import statsmodels.stats.weightstats as smw

import general.data_io as dio
import general.neural_analysis as na
import general.utility as u
import general.decoders as gd
import swap_errors.auxiliary as swa

def log_likelihood_comparison(model_dict, use_weights=None, thresh=None):
    mw = smw.DescrStatsW
    n_models = len(model_dict)
    m_diff = np.zeros((n_models, n_models))
    sem_diff = np.zeros_like(m_diff)
    names = np.zeros((n_models, n_models, 2), dtype=object)
    keys = list(model_dict.keys())
    prod = it.combinations(range(n_models), 2)
    for (i1, i2) in prod:
        k1, k2 = keys[i1], keys[i2]
        l1 = np.mean(model_dict[k1].log_likelihood.y, axis=(0, 1))
        l2 = np.mean(model_dict[k2].log_likelihood.y, axis=(0, 1))

        if use_weights is not None:
            if thresh is not None:
                weights = use_weights > thresh
            else:
                weights = use_weights*(use_weights.shape[0]/np.sum(use_weights))
        else:
            weights = None
        delt = l1 - l2
        diff_model = mw(delt, weights=weights, ddof=0)
        m_diff[i1, i2] = diff_model.mean
        sem_diff[i1, i2] = diff_model.std_mean
        names[i1, i2] = (k1, k2)
    return m_diff, sem_diff, names

def _get_key_mu(posterior, cols, keys, mean=True, mask=None):
    for i, k in enumerate(keys):
        arr = np.concatenate(posterior[k], axis=0)
        if mask is not None:
            cols_use = cols[i, mask].T
        else:
            cols_use = cols[i].T
        dot_arr = np.dot(arr, cols_use)
        if i == 0:
            mu_arr = dot_arr
        else:
            mu_arr = mu_arr + dot_arr
    if mean:
        mu_arr = np.mean(mu_arr, axis=0)
    return mu_arr.T

def get_normalized_centroid_distance(fit_az, data, eh_key='err_hat',
                                     col_keys=('C_u', 'C_l'), 
                                     cent1_keys=(('mu_d_u', 'mu_l'),
                                                 ('mu_u', 'mu_d_l')),
                                     cent2_keys=(('mu_l', 'mu_d_u'),
                                                 ('mu_d_l', 'mu_u')),
                                     resp_key='y', cue_key='cue',
                                     p_thresh=.5, p_key='p', p_ind=1,
                                     eps=1e-3):
    cols = np.stack(list(data[ck] for ck in col_keys), axis=0)
    pp = np.concatenate(fit_az.posterior_predictive[eh_key], axis=0)
    cues = data[cue_key]
    u_cues = np.unique(cues)
    resp = data[resp_key]
    if p_thresh is not None:
        mask = data[p_key][:, p_ind] > p_thresh
    else:
        mask = np.ones(len(data[p_key]), dtype=bool)
    true_arr = []
    pred_arr = []
    p_vals = []
    for i, cue in enumerate(u_cues):
        cue_mask = np.logical_and(mask, cues == cue)
        p_vals.append(data[p_key][cue_mask])
        mu1 = _get_key_mu(fit_az.posterior, cols, cent1_keys[i],
                          mask=cue_mask)
        mu2 = _get_key_mu(fit_az.posterior, cols, cent2_keys[i],
                          mask=cue_mask)
        v_len = np.sqrt(np.sum((mu2 - mu1)**2, axis=1))
        v_len[v_len < eps] = 1
        resp_c = resp[cue_mask]
        pp_c = pp[:, cue_mask]
        true_arr_i = np.sum((resp_c - mu1)*(mu2 - mu1), axis=1)/v_len**2
        true_arr.append(true_arr_i)
        pred_arr_i = np.sum((pp_c - mu1)*(mu2 - mu1), axis=2)/v_len**2
        pred_arr.append(pred_arr_i)
        big_mask = np.abs(true_arr_i) > 10
        if np.any(big_mask):
            print(true_arr_i[big_mask])
            print(v_len[big_mask])
            print(resp_c[big_mask])
            print(data['C_u'][cue_mask][big_mask])
            print(data['C_l'][cue_mask][big_mask])
    true_arr_full = np.concatenate(true_arr, axis=0)
    pred_arr_full = np.concatenate(pred_arr, axis=1).flatten()
    p_vals_full = np.concatenate(p_vals, axis=0)
    return true_arr_full, pred_arr_full, p_vals_full

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

def get_pareto_k(fit, k_thresh=.7):
    x = az.loo(fit, pointwise=True)
    k_val = x['pareto_k']
    k_mask = k_val > k_thresh
    inds = np.where(k_mask)[0]
    trls = fit.observed_data.y[k_mask]
    return k_val, inds, trls

def get_pareto_k_dict(fit_dict, **kwargs):
    k_dict = {}
    for k, v in fit_dict.items():
        k_dict[k] = get_pareto_k(v, **kwargs)
    return k_dict
        
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

def _get_cmean(trls, trl_cols, targ_col, all_cols, color_window=.2,
               positions=None):
    if positions is None:
        u_pos = [0]
        positions = np.zeros(trls.shape[2])
    else:
        u_pos = np.unique(positions)
    out = np.zeros((len(u_pos), trls.shape[0], trls.shape[-1]))
    for i, pos in enumerate(u_pos):
        pos_mask = pos == positions
        col_dists = np.abs(u.normalize_periodic_range(targ_col - all_cols))
        col_mask = col_dists < color_window
        use_cols = all_cols[col_mask]
        col_trl_mask = np.isin(trl_cols, use_cols)
        trial_mask = np.logical_and(pos_mask, col_trl_mask)
        pop_col = trls[:, 0, trial_mask]
        out[i] = np.nanmean(pop_col, axis=1)
    if positions is None:
        out = out[0]
    return out

def _get_trl_dist(trl, m, norm_neurons=True):
    d = np.sqrt(np.nansum((trl - m)**2, axis=0))/m.shape[0]
    return d

def _get_leftout_color_dists(pop_i, targ_cols, dist_cols, upper_samp,
                             splitter=skms.LeaveOneOut, norm=True,
                             u_cols=None, color_window=.2, norm_neurons=True,
                             return_norm=False):
    cols_arr = np.stack((np.array(targ_cols), np.array(dist_cols)), axis=1)
    cols_pos = np.stack((np.array(upper_samp),
                        np.logical_not(np.array(upper_samp))),
                       axis=1)
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
        pos_train = cols_pos[tr_ind]
        pop_test = pop_i[:, :, te_ind]
        cols_test = cols_arr[te_ind]
        pos_test = cols_pos[te_ind]
        if i == 0:
            targ_dists = np.zeros((n_splits, len(te_ind), 4, pop_test.shape[3]))
            dist_dists = np.zeros_like(targ_dists)
            vec_dists = np.zeros((n_splits, len(te_ind), 3, pop_test.shape[3]))
        targ_means[i] = {}
        dist_means[i] = {}
        for uc in u_cols:
            targ_means[i][uc] = _get_cmean(pop_train, cols_train[:, 0], uc,
                                           u_cols, color_window=color_window,
                                           positions=pos_train[:, 0])
            dist_means[i][uc] = _get_cmean(pop_train, cols_train[:, 1], uc,
                                           u_cols, color_window=color_window,
                                           positions=pos_train[:, 1])
        for j in range(pop_test.shape[2]):
            tc = cols_test[j, 0]
            dc = cols_test[j, 1]
            t_pos = pos_test[j, 0]
            d_pos = pos_test[j, 1]
            targ_dists[i, j, 0] = _get_trl_dist(pop_test[:, 0, j],
                                                targ_means[i][tc][t_pos],
                                                norm_neurons=norm_neurons)
            targ_dists[i, j, 1] = _get_trl_dist(pop_test[:, 0, j],
                                                targ_means[i][dc][t_pos],
                                                norm_neurons=norm_neurons)
            targ_dists[i, j, 2] = _get_trl_dist(pop_test[:, 0, j],
                                                targ_means[i][tc][d_pos],
                                                norm_neurons=norm_neurons)
            targ_dists[i, j, 3] = _get_trl_dist(pop_test[:, 0, j],
                                                targ_means[i][dc][d_pos],
                                                norm_neurons=norm_neurons)
            vec_dists[i, j, 0] = _get_vec_dist(pop_test[:, 0, j],
                                               targ_means[i][tc][t_pos],
                                               targ_means[i][dc][t_pos])
            vec_dists[i, j, 1] = _get_vec_dist(pop_test[:, 0, j],
                                               targ_means[i][tc][t_pos],
                                               targ_means[i][tc][d_pos])
            vec_dists[i, j, 2] = _get_vec_dist(pop_test[:, 0, j],
                                               targ_means[i][tc][t_pos],
                                               targ_means[i][dc][d_pos])
    out = targ_dists, vec_dists, dist_dists, targ_means, dist_means
    if return_norm and norm:
        out = out + (m, s)
    return out

def _get_vec_dist(pop, targ, dist):
    vec = targ - dist
    cent = np.nanmean(np.stack((targ, dist), axis=0), axis=0)
    mid = np.nansum(vec*cent, axis=0)
    vec_len = np.sqrt(np.nansum(vec**2, axis=0, keepdims=True))
    
    proj = (np.nansum(pop*vec, axis=0) - mid)/vec_len
    return proj
    
def get_dist_diff_prop(corr_dist, err_dist, n_boots=1000):
    outs = []
    for i, cd in enumerate(corr_dist):
        ed = err_dist[i]
        out_i = np.zeros((len(ed), n_boots, ed.shape[-1]))
        cd_diff = np.diff(cd, axis=2)[:, :, 0]
        assert cd.shape[1] == 1
        ed_diff = np.squeeze(np.diff(ed, axis=2))
        diff_diff = ed_diff - cd_diff
        func = lambda x: np.mean(x > 0, axis=0)
        for j, dd in enumerate(diff_diff):
            out_i[j] = u.bootstrap_list(dd, func, n=n_boots,
                                        out_shape=(ed.shape[-1],))
        outs.append(out_i)
    return outs    

def get_test_color_dists(data, tbeg, tend, twindow, tstep, 
                         targ_means, dist_means,
                         time_key='SAMPLES_ON_diode',
                         targ_key='LABthetaTarget',
                         dist_key='LABthetaDist',
                         resp_field='LABthetaResp',
                         upper_key='IsUpperSample',
                         regions=None, norm=True,
                         norm_neurons=True, m=None, s=None,
                         err_thr=None, use_cache=False):
    targ_cols = data[targ_key]
    dist_cols = data[dist_key]
    targ_pos = data[upper_key]
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                    time_zero_field=time_key,
                                    skl_axes=True, regions=regions,
                                    cache=use_cache)
    if m is None:
        m = (None,)*len(pops)
        s = (None,)*len(pops)
    out_dists = []
    out_vecs = []
    for i, pop_i in enumerate(pops):
        tc_i = np.array(targ_cols[i])
        dc_i = np.array(dist_cols[i])
        tp_i = np.array(targ_pos[i])
        dp_i = np.logical_not(np.array(targ_pos[i])).astype(int)
        out = compute_dists(pop_i, tc_i, dc_i,
                            tp_i, dp_i, targ_means[i], 
                            norm_neurons=norm_neurons,
                            mean=m[i], std=s[i])
        out_dist, out_vec = out
        out_dists.append(out_dist)
        out_vecs.append(out_vec)
    return out_dists, out_vecs
    
def compute_dists(pop_test, trl_targs, trl_dists, targ_pos, dist_pos,
                  col_means, norm_neurons=True, mean=None, std=None):
    if mean is not None and pop_test.shape[2] > 0:
        pop_test = (pop_test - mean)/std
    n_models = len(col_means)
    n_tests = pop_test.shape[2]
    out_dists = np.zeros((n_models, n_tests, 4, pop_test.shape[3]))
    vec_dists = np.zeros((n_models, n_tests, 3, pop_test.shape[3]))
    for i in range(n_models):
        for j in range(pop_test.shape[2]):
            tc = trl_targs[j]
            dc = trl_dists[j]
            t_pos = targ_pos[j]
            d_pos = dist_pos[j]
            out_dists[i, j, 0] = _get_trl_dist(pop_test[:, 0, j],
                                               col_means[i][tc][t_pos],
                                               norm_neurons=norm_neurons)
            out_dists[i, j, 1] = _get_trl_dist(pop_test[:, 0, j],
                                               col_means[i][dc][t_pos],
                                               norm_neurons=norm_neurons)
            out_dists[i, j, 2] = _get_trl_dist(pop_test[:, 0, j],
                                               col_means[i][tc][d_pos],
                                               norm_neurons=norm_neurons)
            out_dists[i, j, 3] = _get_trl_dist(pop_test[:, 0, j],
                                               col_means[i][dc][d_pos],
                                               norm_neurons=norm_neurons)
            vec_dists[i, j, 0] = _get_vec_dist(pop_test[:, 0, j],
                                               col_means[i][tc][t_pos],
                                               col_means[i][dc][t_pos])
            vec_dists[i, j, 1] = _get_vec_dist(pop_test[:, 0, j],
                                               col_means[i][tc][t_pos],
                                               col_means[i][tc][d_pos])
            vec_dists[i, j, 2] = _get_vec_dist(pop_test[:, 0, j],
                                               col_means[i][tc][t_pos],
                                               col_means[i][dc][d_pos])

    return out_dists, vec_dists

def get_color_means(data, tbeg, tend, twindow, tstep, color_window=.2,
                    time_key='SAMPLES_ON_diode',
                    targ_key='LABthetaTarget',
                    dist_key='LABthetaDist',
                    upper_key='IsUpperSample',
                    regions=None, leave_out=0,
                    norm=True, norm_neurons=True,
                    test_data=None, pops_xs=None,
                    use_cache=False):
    targ_cols = data[targ_key]
    dist_cols = data[dist_key]
    upper_samps = data[upper_key]
    u_cols = np.unique(np.concatenate(targ_cols))
    if pops_xs is not None:
        pops, xs = pops_xs
    else:
        pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                        time_zero_field=time_key,
                                        regions=regions,
                                        skl_axes=True, cache=use_cache)
    targ_dists_all = []
    vec_dists_all = []
    dist_dists_all = []
    targ_means_all = []
    dist_means_all = []
    means = []
    stds = []
    for i, pop_i in enumerate(pops):
        out = _get_leftout_color_dists(pop_i, targ_cols[i], dist_cols[i],
                                       upper_samps[i],
                                       norm=norm, u_cols=u_cols,
                                       color_window=color_window,
                                       return_norm=True)
        targ_dists, vec_dists, dist_dists, targ_means, dist_means, m, s = out
        means.append(m)
        stds.append(s)
        targ_dists_all.append(targ_dists)
        vec_dists_all.append(vec_dists)
        dist_dists_all.append(dist_dists)
        targ_means_all.append(targ_means)
        dist_means_all.append(dist_means)
    out_dists = (targ_dists_all, vec_dists_all, dist_dists_all)
    if test_data is not None:
        out = get_test_color_dists(test_data, tbeg, tend, twindow, tstep,
                                   targ_means_all, dist_means_all,
                                   time_key=time_key, dist_key=dist_key,
                                   regions=regions, norm=norm,
                                   norm_neurons=norm_neurons, m=means,
                                   s=stds, use_cache=use_cache)
        test_dists, test_vecs = out
        out_dists = out_dists + (test_dists, test_vecs)
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

def quantify_swap_loo(mdict, data, method='weighted', threshold=.5, data_key='p',
                      data_ind=1, comb_func=np.sum):
    loo_i_dict = {}
    sum_loo_dict = {}
    new_m = {}
    for (k, model) in mdict.items():
        m_copy = model.copy()
        data_col = np.expand_dims(data[data_key][:, data_ind], (0, 1))
        if method == 'weighted':
            adj_loo = m_copy.log_likelihood*data_col
        elif method == 'log_weighted':
            adj_loo = m_copy.log_likelihood + np.log(data_col)
        elif method == 'threshold':
            mask = data_col > threshold
            adj_loo = m_copy.log_likelihood*mask
        elif method == 'none':
            adj_loo = m_copy.log_likelihood
        else:
            raise IOError('unrecognized method')
        m_copy.log_likelihood = adj_loo
        new_m[k] = m_copy
    comp = az.compare(new_m)
    return comp

def spline_color(cols, num_bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    bins = np.linspace(0, 2*np.pi, num_bins+1)[:num_bins]
    
    dc = 2*np.pi/(len(bins))

    # get the nearest bin
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    dist_near = np.abs(distances).min(0)
    nearest = np.abs(distances).argmin(0)
    # see if the color is to the "left" or "right" of that bin
    sec_near = np.sign(distances[nearest,np.arange(len(cols))]+1e-8).astype(int)
    # add epsilon to handle 0
    # fill in the convex array
    alpha = np.zeros((len(bins),len(cols)))
    alpha[nearest, np.arange(len(cols))] = (dc-dist_near)/dc
    alpha[np.mod(nearest-sec_near,len(bins)), np.arange(len(cols))] = 1 - (dc-dist_near)/dc
    
    return alpha

# def spline_color(cols, num_bins):
#     '''
#     cols should be given between 0 and 2 pi, bins also
#     '''
#     bins = np.linspace(0, 2*np.pi, num_bins+1)[:num_bins]

#     dc = 2*np.pi/(len(bins))

#     diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
#     distances = np.arctan2(diffs.imag,diffs.real)
#     dist_near = (distances).max(0)
#     nearest = (distances).argmax(0)
#     alpha = np.zeros((len(bins),len(cols)))
#     alpha[nearest, np.arange(len(cols))] = (dist_near-dc)/dc
#     alpha[np.mod(nearest+1,len(bins)), np.arange(len(cols))] = 1 - (dist_near-dc)/dc
    
#     return alpha    

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

def retro_mask(data):
    bhv_retro = (data['is_one_sample_displayed'] == 0).rs_and(
        data['Block'] > 1)
    bhv_retro = bhv_retro.rs_and(data['StopCondition'] > -2)
    data_retro = data.mask(bhv_retro)
    return data_retro

def fit_animal_bhv_models(data, *args, animal_key='animal', retro_masking=True,
                          **kwargs):
    if retro_masking:
        bhv_retro = (data['is_one_sample_displayed'] == 0).rs_and(
            data['Block'] > 1)
        bhv_retro = bhv_retro.rs_and(data['StopCondition'] > -2)
        data = data.mask(bhv_retro)
    animals = np.unique(np.array(data[animal_key]))
    map_dict = {}
    full_dict = {}
    for i, animal in enumerate(animals):
        dbhv_mi = data.session_mask(data[animal_key] == animal)
        sd_i = fit_bhv_model(dbhv_mi, *args, **kwargs)
        full_dict[str(animal)] = sd_i
        map_dict.update(swa.transform_bhv_model(sd_i[0], sd_i[-1]))
    return full_dict, map_dict

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
                  arviz=mixture_arviz, adapt_delta=.9, diagnostics=True,
                  **stan_params):
    if prior_dict is None:
        prior_dict = default_prior_dict
    targs_is = data[targ_field]
    session_list = np.array(data[['animal', 'date']])
    mapping_list = []
    session_nums = np.array([], dtype=int)
    for i, x in enumerate(targs_is):
        sess = np.ones(len(x), dtype=int)*(i + 1)
        session_nums = np.concatenate((session_nums,
                                       sess))
        indices = x.index
        sess_info0 = (str(session_list[i, 0]),)*len(x)
        sess_info1 = (str(session_list[i, 1]),)*len(x)
        mapping_list = mapping_list + list(zip(indices, sess_info0,
                                               sess_info1))
    mapping_dict = {i:mapping_list[i] for i in range(len(session_nums))}
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
    if diagnostics:
        diag = ps.diagnostics.check_hmc_diagnostics(fit)
    else:
        diag = None
    fit_av = az.from_pystan(posterior=fit, **arviz)
    return fit, diag, fit_av, stan_data, mapping_dict
