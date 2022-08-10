
import numpy as np
import pickle
import pystan as ps
import arviz as az
import scipy.spatial.distance as spsd
import sklearn.manifold as skm
import sklearn.neighbors as skn
import sklearn.model_selection as skms
import sklearn.svm as skc
import sklearn.linear_model as sklm
import sklearn.preprocessing as skp
import scipy.stats as sts
import itertools as it
import statsmodels.stats.weightstats as smw
import elephant as el
import quantities as pq

import general.data_io as dio
import general.neural_analysis as na
import general.utility as u
import general.decoders as gd
import swap_errors.auxiliary as swa


def spline_decoding(data, activity='y', col_keys=('C_u',), cv=20,
                    cv_type=skms.LeaveOneOut,
                    model=sklm.Ridge):
    targ = np.concatenate(list(data[ck] for ck in col_keys), axis=1)
    pred = data[activity]
    if cv_type is not None:
        cv = cv_type()
    out = skms.cross_val_score(model(), pred, targ, cv=cv)
    return out

def decode_corr_swap_guess(data, thresh=.3, activity='y', type_key='p',
                           n_cv=20, cv_type=skms.ShuffleSplit,
                           test_prop=.1, model=skc.SVC):
    p = data[type_key]
    corr_mask = p[:, 0] > (1 - thresh)
    swap_mask = p[:, 1] > thresh
    guess_mask = p[:, 2] > thresh
    neur = data[activity]
    mask_list = (corr_mask, swap_mask, guess_mask)
    labels = ('corr', 'swap', 'guess')
    out_dict = {}
    out_dict_shuff = {}
    for (i, j) in it.combinations(range(len(mask_list)), 2):
        c1, c2 = mask_list[i], mask_list[j]
        n_c1 = neur[c1]
        l_c1 = np.zeros(len(n_c1))
        n_c2 = neur[c2]
        l_c2 = np.ones(len(n_c2))

        if len(n_c1) > len(n_c2):
            sub_inds = np.random.choice(np.arange(len(n_c1)), len(n_c2),
                                        replace=False)
            n_c1 = n_c1[sub_inds]
            l_c1 = l_c1[sub_inds]
        n = np.concatenate((n_c1, n_c2), axis=0)
        l = np.concatenate((l_c1, l_c2), axis=0)
        pipe = na.make_model_pipeline(model=model)
        cv = cv_type(n_cv, test_size=test_prop)
        out = skms.cross_val_score(pipe, n, l, cv=cv)
        out_dict[(labels[i], labels[j])] = out

        inds = np.arange(len(l))
        np.random.shuffle(inds)
        out_shuff = skms.cross_val_score(pipe, n, l[inds], cv=cv)
        out_dict_shuff[(labels[i], labels[j])] = out_shuff
    return out_dict, out_dict_shuff

def session_ll_analysis(session_dict, use_weights=False, **kwargs):
    out_ms = []
    out_sems = []
    k_list = []
    for k, (m, data) in session_dict.items():
        if use_weights:
            weights = data['p'][:, 1]
        else:
            weights = None
        out = log_likelihood_comparison(m, use_weights=weights, **kwargs)
        m_diff, sem_diff, names = out
        out_ms.append(m_diff)
        out_sems.append(sem_diff)
        k_list.append(k)
    ms_all = np.stack(out_ms, axis=0)
    sem_all = np.stack(out_sems, axis=0)
    return ms_all, sem_all, k_list, names

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
        l1 = np.mean(model_dict[k1].log_likelihood.y.to_numpy(), axis=(0, 1))
        l2 = np.mean(model_dict[k2].log_likelihood.y.to_numpy(), axis=(0, 1))
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

def _get_key_mu(posterior, cols, keys, mean=True, mask=None, inter_key=None):
    for i, k in enumerate(keys):
        arr = np.concatenate(posterior[k].to_numpy(), axis=0)
        if mask is not None:
            cols_use = cols[i, mask].T
        else:
            cols_use = cols[i].T
        dot_arr = np.dot(arr, cols_use)
        if i == 0:
            mu_arr = dot_arr
        else:
            mu_arr = mu_arr + dot_arr
    if inter_key is not None and inter_key in posterior.keys():
        inter_pt = np.expand_dims(np.concatenate(posterior[inter_key].to_numpy(),
                                                 axis=0), 2)
        mu_arr = mu_arr + inter_pt 
    if mean:
        mu_arr = np.mean(mu_arr, axis=0)
    return mu_arr.T

spatial_mu_config = ((('mu_d_u', 'mu_l'),
                      ('mu_u', 'mu_d_l')),
                     (('mu_l', 'mu_d_u'),
                      ('mu_d_l', 'mu_u')))
cue_mu_config = ((('mu_d_u', 'mu_l'),
                  ('mu_u', 'mu_d_l')),
                 (('mu_u', 'mu_d_l'),
                  ('mu_d_u', 'mu_l')))
def compute_vecs(fit_az, data, col_keys=('C_u', 'C_l'),
                 mu_configs=spatial_mu_config, cue_key='cue',
                 thin_col=1):
    cols = np.stack(list(data[ck] for ck in col_keys), axis=0)
    all_cols = np.concatenate(cols, axis=0)
    u_cols = np.unique(list(tuple(col) for col in all_cols), axis=0)[::thin_col]
    
    n_cols = len(u_cols)
    n_configs = len(mu_configs[0])
    n_units = fit_az.posterior['mu_u'].shape[2]

    vecs = np.zeros((n_cols, n_cols, n_configs, n_units))
    m1s = np.zeros_like(vecs)
    m2s = np.zeros_like(vecs)
    lens = np.zeros((n_cols, n_cols, n_configs))
    for (i1, i2) in it.product(range(n_cols), repeat=2):
        c1 = u_cols[i1]
        c2 = u_cols[i2]
        for j in range(n_configs):
            mu_keys1_j = mu_configs[0][j]
            mu_keys2_j = mu_configs[1][j]
            m1 = _get_key_mu(fit_az.posterior, (c1, c2), mu_keys1_j)
            m2 = _get_key_mu(fit_az.posterior, (c1, c2), mu_keys2_j)
            vec_i = m2 - m1
            l_i = np.sqrt(np.sum(vec_i**2))

            m1s[i1, i2, j] = m1
            m2s[i1, i2, j] = m2
            vecs[i1, i2, j] = vec_i
            lens[i1, i2, j] = l_i
    return vecs, lens, m1s, m2s 
    
def get_normalized_centroid_distance(fit_az, data, eh_key='err_hat',
                                     col_keys=('C_u', 'C_l'), 
                                     cent1_keys=((('mu_d_u', 'mu_l'),
                                                  'intercept_down'),
                                                 (('mu_u', 'mu_d_l'),
                                                  'intercept_up')),
                                     cent2_keys=((('mu_l', 'mu_d_u'),
                                                  'intercept_down'),
                                                 (('mu_d_l', 'mu_u'),
                                                  'intercept_up')),
                                     resp_key='y', cue_key='cue',
                                     p_thresh=.5, p_key='p', p_ind=1,
                                     eps=1e-3, use_cues=True,
                                     correction_field='p_spa',
                                     do_correction=False,
                                     type_key='type',
                                     trl_filt=None,
                                     col_thr=.1,
                                     p_comp=np.greater):
    cols = np.stack(list(data[ck] for ck in col_keys), axis=0)
    pp = np.concatenate(fit_az.posterior_predictive[eh_key].to_numpy(),
                        axis=0)
    if use_cues:
        cues = data[cue_key]
    else:
        cues = np.zeros(len(data[cue_key]), dtype=int)
    u_cues = np.unique(cues)
    resp = data[resp_key]
    if do_correction and correction_field in fit_az.posterior.keys():
        print('mult')
        p_mult = np.mean(fit_az.posterior[correction_field].to_numpy())
    else:
        p_mult = 1
    if p_thresh is not None:
        mask = p_comp(data[p_key][:, p_ind]*p_mult, p_thresh)
    else:
        mask = np.ones(len(data[p_key]), dtype=bool)
    if trl_filt is not None:
        if trl_filt == 'retro':
            mask = np.logical_and(mask, data[type_key] == 1)
        elif trl_filt == 'pro':
            mask = np.logical_and(mask, data[type_key] == 2)
        else:
            raise IOError('trl_filt key not recognized')
    true_arr = []
    pred_arr = []
    p_vals = []
    if col_thr is not None:
        col_dist = np.sum((cols[0]*cols[1]), axis=1)
        col_mask = col_dist < col_thr
        mask = np.logical_and(mask, col_mask)

    for i, cue in enumerate(u_cues):
        cue_mask = np.logical_and(mask, cues == cue)
        p_vals.append(data[p_key][cue_mask])
        mu1 = _get_key_mu(fit_az.posterior, cols, cent1_keys[cue][0],
                          mask=cue_mask, inter_key=cent1_keys[cue][1])
        mu2 = _get_key_mu(fit_az.posterior, cols, cent2_keys[cue][0],
                          mask=cue_mask, inter_key=cent2_keys[cue][1])
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

def _argmax_decider(ps, ind):
    mask = np.argmax(ps, axis=1) == ind
    return mask

def _plurality_decider(ps, ind, prob=1/3):
    mask = ps[:, ind] > prob
    return mask

def _diff_decider(ps, ind_pos, ind_sub, diff=0):
    mask = ps[:, ind_pos] - ind_sub[:, ind_sub] > diff
    return mask

def corr_diff(ps, diff=0):
    return _diff_decider(ps, 0, 1, diff=diff)

def swap_diff(ps, diff=0):
    return _diff_decider(ps, 1, 0, diff=diff)

def swap_plurality(ps, prob=1/3):
    return _plurality_decider(ps, 1, prob=prob)

def corr_plurality(ps, prob=1/3):
    return _plurality_decider(ps, 0, prob=prob)

def swap_argmax(ps):
    return _argmax_decider(ps, 1)
    
def corr_argmax(ps):
    return _argmax_decider(ps, 0)

def _col_diff_rad(c1, c2):
    return np.abs(u.normalize_periodic_range(c1 - c2))

def _col_diff_spline(c1, c2):
    return np.sqrt(np.sum((c1 - c2)**2, axis=1))

def convert_spline_to_rad(cu, cl):
    cols = np.unique(np.concatenate((cu, cl), axis=0), axis=0)
    if len(cols) != 64:
        print('weird cols', len(cols))
    rads = np.linspace(0, 2*np.pi, len(cols) + 1)[:-1]
    d = {tuple(x):rads[i] for i, x in enumerate(cols)}
    cu_rad = np.zeros(len(cu))
    cl_rad = np.zeros_like(cu_rad)
    for i, cu_i in enumerate(cu):
        cu_rad[i] = d[tuple(cu_i)]
        cl_rad[i] = d[tuple(cl[i])]
    return cu_rad, cl_rad

def cue_mask_dict(data_dict, cue_val, cue_key='cue',
                  mask_keys=('C_u', 'C_l', 'p', 'y', 'cue')):
    new_dict = {}
    new_dict.update(data_dict)
    mask = new_dict[cue_key] == cue_val
    for mk in mask_keys:
        new_dict[mk] = data_dict[mk][mask]
    return new_dict

def filter_nc_dis(centroid_dict, use_d1s, cond_types, use_range,
                  regions='all', d2_key='d2'):
    use_dis = []
    for d1_c in use_d1s:
        if len(list(centroid_dict[d1_c].keys())[0]) > 2:
            k_filt = lambda k: k[0] in use_range and k[1] == regions
        else:
            k_filt = lambda k: k[0] in use_range
        d1i_use = {k:v for k, v in centroid_dict[d1_c].items()
                   if k_filt(k)}
        use_dis.append(d1i_use)
    for use_cond in cond_types:
        if len(list(centroid_dict[d2_key].keys())[0]) > 2:
            k_filt = lambda k: (k[0] in use_range and k[1] == regions
                                and k[2] == use_cond)
        else:
            k_filt = lambda k: (k[0] in use_range and k[1] == use_cond)
        d2i_use = {k:v for k, v in centroid_dict[d2_key].items() 
                   if k_filt(k)}
        use_dis.append(d2i_use)
    return use_dis


def compute_centroid_diffs(centroid_dict, use_d1s=('d1_cl', 'd1_cu'),
                           session_dict=None, cond_types=('pro', 'retro'),
                           d2_key='d2', min_swaps=10, regions='all'):
    if session_dict is None:
        session_dict = dict(elmo_range=range(13),
                            waldorf_range=range(13, 24),
                            comb_range=range(24))

    ax_labels = list(use_d1s) + list(' '.join((d2_key, ct)) for ct in cond_types)
    m_labels = []
    m_out = np.zeros((len(session_dict), len(use_d1s) + len(cond_types)))
    p_out = np.zeros_like(m_out)
    for i, (r_key, use_range) in enumerate(session_dict.items()):
        m_labels.append(r_key)
        use_dis = filter_nc_dis(centroid_dict, use_d1s, cond_types,
                                use_range, regions=regions, d2_key=d2_key)
        comb_dicts = []
        for dict_i in use_dis:
            nulls_all = list(v[0] for v in dict_i.values())
            nulls = np.concatenate(nulls_all, axis=0)
            swaps_all = list(np.mean(v[1], axis=0) for v in dict_i.values())
            swaps = np.concatenate(swaps_all, axis=0)
            comb_dicts.append({'comb':(nulls, swaps)})

        for j, dict_i in enumerate(comb_dicts):
            nulls, swaps = dict_i['comb']
            m_null = np.nanmedian(nulls)
            m_swaps = np.nanmedian(swaps)
            if len(swaps) > min_swaps:
                utest = sts.mannwhitneyu(swaps, nulls, alternative='greater',
                                         nan_policy='omit')
                p_out[i, j] = utest.pvalue
            else:
                p_out[i, j] = np.nan
            m_out[i, j] = m_swaps - m_null
    return m_out, p_out

def compute_sweep_ncs(sweep_keys, run_ind,
                      folder='swap_errors/naive_centroids/',
                      use_d1s=('d1_cl', 'd1_cu'),
                      d2_key='d2', cond_types=('pro', 'retro'),
                      monkey_ranges=None, regions='all'):
    if monkey_ranges is None:
        monkey_ranges = dict(elmo_range=range(13),
                             waldorf_range=range(13, 24),
                             comb_range=range(24))
    nc_df = swa.load_nc_sweep(folder, run_ind)
    ax_vals = []
    for sk in sweep_keys:
        if list(nc_df[sk])[0] is None:
            app_vals = (None,)
        else:
            app_vals = np.unique(nc_df[sk])
        ax_vals.append(app_vals)

    param_shape = list(len(av) for av in ax_vals)
    cond_shape = [len(monkey_ranges), len(use_d1s) + len(cond_types)]
    
    m_arr = np.zeros(param_shape + cond_shape)
    p_arr = np.zeros_like(m_arr)
    for ind in u.make_array_ind_iterator(param_shape):
        mask = list(nc_df[sweep_keys[i]] == av[ind[i]]
                    for i, av in enumerate(ax_vals)
                    if av[ind[i]] is not None)
        mask = np.product(mask, axis=0).astype(bool)
        nc_masked = nc_df[mask].iloc[0]
        nc_ind = nc_masked.to_dict()
        ms, ps = compute_centroid_diffs(nc_ind, use_d1s=use_d1s,
                                        cond_types=cond_types,
                                        d2_key=d2_key,
                                        session_dict=monkey_ranges,
                                        regions=regions)
        m_arr[ind] = ms
        p_arr[ind] = ps
    return ax_vals, m_arr, p_arr

def organize_forgetting(folder, run_ind, sweep_keys=('decider_arg',),
                        **kwargs):
    f_df = swa.load_f_sweep(folder, run_ind)
    if f_df[sweep_keys[0]][0] is None:
        avs = ((None,),)
    else:
        avs = list(np.unique(f_df[sk]) for sk in sweep_keys)
    
    out_arr = np.zeros(list(len(av) for av in avs), dtype=object)

    for ind in u.make_array_ind_iterator(out_arr.shape):
        masks = []
        for i, sk in enumerate(sweep_keys):
            if avs[i][ind[i]] is None:
                masks.append(np.ones(len(f_df[sk]), dtype=bool))
            else:
                masks.append(f_df[sk] == avs[i][ind[i]])
        mask = np.product(masks, axis=0, dtype=bool)
        df_use = f_df[mask]
        out_arr[ind] = combine_forgetting(df_use, **kwargs)
    return avs, out_arr        

def combine_forgetting(f_df, 
                       include_keys=('forget_cu', 'forget_cl'),
                       merge_keys=True,
                       mid_average=True):
    
    merge_dict = {}
    for _, row in f_df.iterrows():
        for key in include_keys:
            if merge_keys:
                save_key = 'comb'
            else:
                save_key = key
            curr_dict = merge_dict.get(save_key, {})
            for inner_key, vals in row[key].items():
                (new_nulls, new_swaps,  dist_nulls, dist_swaps) = vals
                new_swaps = np.nanmean(new_swaps, axis=0)
                dist_swaps = np.nanmean(dist_swaps, axis=0)
                if mid_average:
                    new_swaps = np.nanmean(new_swaps, axis=0, keepdims=True)
                    new_nulls = np.nanmean(new_nulls, axis=0, keepdims=True)

                    dist_nulls = np.nanmean(dist_nulls, axis=0, keepdims=True)
                    dist_swaps = np.nanmean(dist_swaps, axis=0, keepdims=True)
                    
                curr_entry = curr_dict.get(inner_key)
                if curr_entry is None:
                    curr_dict[inner_key] = (new_nulls, new_swaps,
                                            dist_nulls, dist_swaps)
                else:
                    curr_nulls, curr_swaps, cd_nulls, cd_swaps = curr_entry
                    full_nulls = np.concatenate((curr_nulls, new_nulls), axis=0)
                    full_swaps = np.concatenate((curr_swaps, new_swaps), axis=0)
                    fd_nulls = np.concatenate((cd_nulls, dist_nulls), axis=0)
                    fd_swaps = np.concatenate((cd_swaps, dist_swaps), axis=0)
                    curr_dict[inner_key] = (full_nulls, full_swaps,
                                            fd_nulls, fd_swaps)
            merge_dict[save_key] = curr_dict
    return merge_dict

def _get_corr_swap_inds(ps, corr_decider, swap_decider, and_corr_mask=None,
                        and_swap_mask=None):
    corr_mask = corr_decider(ps)
    swap_mask = swap_decider(ps)
    common_mask = np.logical_and(corr_mask, swap_mask)
    corr_mask[common_mask] = False
    if and_corr_mask is not None:
        corr_mask = np.logical_and(corr_mask, and_corr_mask)
    swap_mask[common_mask] = False
    if and_swap_mask is not None:
        swap_mask = np.logical_and(swap_mask, and_swap_mask)
    corr_inds = np.where(corr_mask)[0]
    swap_inds = np.where(swap_mask)[0]
    return corr_inds, swap_inds

def naive_swapping(data_dict,
                   cu_key='up_col_rads',
                   cl_key='down_col_rads',
                   cue_key='cue',
                   flip_cue = False,
                   use_cue=True,
                   no_cue_targ='up_col_rads',
                   no_cue_dist='down_col_rads',
                   tp_key='p',
                   cue_targ=1,
                   activity_key='y',
                   swap_decider=swap_argmax,
                   corr_decider=corr_argmax,
                   col_exclude=0,
                   col_cent=np.pi,
                   cv=skms.LeaveOneOut, col_diff=_col_diff_rad,
                   kernel='rbf',
                   convert_splines=True,
                   swap_mask=None):
    if flip_cue:
        no_cue_targ = 'down_col_rads'
        no_cue_dist = 'up_col_rads'
    if not use_cue:
        c_t = np.zeros_like(data_dict[no_cue_targ])
        c_d = np.zeros_like(data_dict[no_cue_dist])
        c_t[:] = data_dict[no_cue_targ][:]
        c_d[:] = data_dict[no_cue_dist][:]
    else:
        c_t = np.zeros_like(data_dict[cu_key])
        c_d = np.zeros_like(data_dict[cl_key])
        
        c1_mask = data_dict[cue_key] == 1
        c0_mask = data_dict[cue_key] == 0

        c_t[c1_mask] = data_dict[cu_key][c1_mask]
        c_t[c0_mask] = data_dict[cl_key][c0_mask]
        
        c_d[c1_mask] = data_dict[cl_key][c1_mask]
        c_d[c0_mask] = data_dict[cu_key][c0_mask]
    if len(c_t.shape) > 1 and convert_splines:
        c_t, c_d = convert_spline_to_rad(c_t, c_d)
        
    c_dec = c_t
    c_ndec = c_d
    if len(c_dec.shape) > 1 and convert_splines:
        c_dec, c_ndec = convert_spline_to_rad(c_dec, c_ndec)

    norm_diff = u.normalize_periodic_range(c_dec - col_cent)
    color_cat = norm_diff < 0

    norm_diff_dist = u.normalize_periodic_range(c_ndec - col_cent)
    color_cat_dist = norm_diff_dist < 0
    
    corr_inds, swap_inds = _get_corr_swap_inds(data_dict[tp_key],
                                               corr_decider,
                                               swap_decider,
                                               and_swap_mask=swap_mask)
    null_score_targ = np.zeros(len(corr_inds))
    swap_score_targ = np.zeros((len(corr_inds), len(swap_inds)))
    null_score_dist = np.zeros(len(corr_inds))
    swap_score_dist = np.zeros((len(corr_inds), len(swap_inds)))
    y = data_dict[activity_key]
    assert(not np.any(np.isin(swap_inds, corr_inds)))

    cv_gen = cv()
    for i, (train_inds, test_inds) in enumerate(cv_gen.split(corr_inds)):
        corr_tr, corr_te = corr_inds[train_inds], corr_inds[test_inds]
        model = skc.SVC(kernel=kernel)
        model.fit(y[corr_tr], color_cat[corr_tr])
        null_score_targ[i] = model.score(y[corr_te], color_cat[corr_te])
        null_score_dist[i] = model.score(y[corr_te], color_cat_dist[corr_te])
        if len(swap_inds) > 0:
            swap_score_targ[i] = (model.predict(y[swap_inds])
                                  == color_cat[swap_inds])
            swap_score_dist[i] = (model.predict(y[swap_inds])
                                  == color_cat_dist[swap_inds])
        else:
            swap_score_targ[i] = np.nan
            swap_score_dist[i] = np.nan
    return null_score_targ, swap_score_targ, null_score_dist, swap_score_dist

def naive_forgetting(data_dict,
                     cue_key='cue',
                     flip_cue=False,
                     cue_targ=1,
                     **kwargs):
    if flip_cue:
        cue_targ = 0
    cue_mask = data_dict[cue_key] == cue_targ
    out = naive_swapping(data_dict, cue_key=cue_key,
                         use_cue=False,
                         flip_cue=flip_cue, swap_mask=cue_mask,
                         **kwargs)
    return out[:2]

def _compute_trl_c_dist(y, corr_tr, corr_te, tr_targ_cols, targ_col, dist_col,
                        col_thr=np.pi/4,
                        col_diff=_col_diff_rad):
    far_cols = col_diff(targ_col, dist_col) > col_thr
    if far_cols:
        null_cent_inds = corr_tr[col_diff(tr_targ_cols, targ_col) < col_thr]
        swap_cent_inds = corr_tr[col_diff(tr_targ_cols, dist_col) < col_thr]
        
        null_cent = np.mean(y[null_cent_inds], axis=0, keepdims=True)
        swap_cent = np.mean(y[swap_cent_inds], axis=0, keepdims=True)
        swap_vec = swap_cent - null_cent
        sv_len = np.sqrt(np.sum(swap_vec**2))
        sv_u = np.expand_dims(u.make_unit_vector(swap_vec), 0)

        test_activity = y[corr_te]
        dist = np.dot(sv_u, (test_activity - null_cent).T)/sv_len
    else:
        dist = np.nan
    return dist

def naive_centroids(data_dict,
                    cue_key='cue',
                    cu_key='up_col_rads',
                    cl_key='down_col_rads',
                    use_cue=True,
                    flip_cue = False,
                    no_cue_targ='up_col_rads',
                    no_cue_dist='down_col_rads',
                    tp_key='p',
                    activity_key='y',
                    swap_decider=swap_argmax,
                    corr_decider=corr_argmax,
                    col_thr=np.pi/4,
                    cv=skms.LeaveOneOut, col_diff=_col_diff_rad,
                    convert_splines=True):
    if flip_cue:
        no_cue_targ = 'down_col_rads'
        no_cue_dist = 'up_col_rads'
    if not use_cue:
        c_t = np.zeros_like(data_dict[no_cue_targ])
        c_d = np.zeros_like(data_dict[no_cue_dist])
        c_t[:] = data_dict[no_cue_targ][:]
        c_d[:] = data_dict[no_cue_dist][:]
    else:
        c_t = np.zeros_like(data_dict[cu_key])
        c_d = np.zeros_like(data_dict[cl_key])
        
        c1_mask = data_dict[cue_key] == 1
        c0_mask = data_dict[cue_key] == 0

        c_t[c1_mask] = data_dict[cu_key][c1_mask]
        c_t[c0_mask] = data_dict[cl_key][c0_mask]
        
        c_d[c1_mask] = data_dict[cl_key][c1_mask]
        c_d[c0_mask] = data_dict[cu_key][c0_mask]
    if len(c_t.shape) > 1 and convert_splines:
        c_t, c_d = convert_spline_to_rad(c_t, c_d)

    corr_inds, swap_inds = _get_corr_swap_inds(data_dict[tp_key],
                                               corr_decider,
                                               swap_decider)
    null_dists = np.zeros(len(corr_inds))
    swap_dists = np.zeros((len(corr_inds), len(swap_inds)))
    y = data_dict[activity_key]

    cv_gen = cv()
    for i, (train_inds, test_inds) in enumerate(cv_gen.split(corr_inds)):
        corr_tr, corr_te = corr_inds[train_inds], corr_inds[test_inds]
        tr_targ_cols = c_t[corr_tr]
        tr_dist_cols = c_d[corr_tr]
        targ_col = c_t[corr_te]
        dist_col = c_d[corr_te]
        null_dists[i] = _compute_trl_c_dist(y, corr_tr, corr_te, tr_targ_cols,
                                            targ_col, dist_col, col_thr=col_thr,
                                            col_diff=col_diff)
        for j, si in enumerate(swap_inds):
            targ_col, dist_col = c_t[si], c_d[si]

            swap_dists[i, j] = _compute_trl_c_dist(y, corr_tr, si, tr_targ_cols,
                                                   targ_col, dist_col,
                                                   col_thr=col_thr,
                                                   col_diff=col_diff)
    return null_dists, swap_dists

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
        m_copy.log_ood = adj_loo
        new_m[k] = m_copy
    comp = az.compare(new_m)
    return comp

def despline_color(cols, eps=.001, ret_err=False):
    u_cols = np.unique(cols, axis=0)
    num_bins = cols.shape[1]
    
    float_cols = np.linspace(0, 2*np.pi, len(u_cols) + 1)[:-1]
    o_cols = spline_color(float_cols, cols.shape[1]).T
    out = np.zeros(len(cols))
    err = np.zeros_like(out)
    for i, oc in enumerate(o_cols):
        mask = np.sum((np.expand_dims(oc, 0) - cols)**2, axis=1) 
        out[mask < eps] = float_cols[i]
        err[mask < eps] = mask[mask < eps]
    if ret_err:
        out = (out, err)
    return out

def spline_color(cols, num_bins, degree=1, use_skl=True):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    if use_skl:
        st = skp.SplineTransformer(num_bins + 2, degree=degree,
                                   include_bias=False,
                                   extrapolation='periodic')
        cols_spl = st.fit_transform(np.expand_dims(cols, 1))
        alpha = (cols_spl - np.mean(cols_spl, axis=0, keepdims=True)).T
    else:
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

def pro_mask(data):
    bhv_pro = (data['is_one_sample_displayed'] == 0).rs_and(
        data['Block'] == 1)
    bhv_pro = bhv_pro.rs_and(data['StopCondition'] > -2)
    data_pro = data.mask(bhv_pro)
    return data_pro



def fit_animal_bhv_models(data, *args, animal_key='animal', retro_masking=True,
                          pro_masking=False, **kwargs):
    complete = data['StopCondition'] > -2
    if retro_masking:
        bhv_mask = complete.rs_and(
            data['is_one_sample_displayed'] == 0).rs_and(
            data['Block'] > 1)
    if pro_masking:
        bhv_mask = complete.rs_and(
            data['is_one_sample_displayed'] == 0).rs_and(
            data['Block'] == 1)
    data = data.mask(bhv_mask)

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

def gpfa(data, tbeg=-.5, tend=1, winsize=.05,
         n_factors=8, tzf='CUE2_ON_diode'):
    out = data.get_spiketrains(tbeg, tend,
                               time_zero_field=tzf)
    pops = out
    fits = []
    for i, pop in enumerate(pops):
        pop_format = list(list(pop_j) for pop_j in pop)
        gp = el.gpfa.GPFA(bin_size=winsize*pq.s, x_dim=n_factors)
        gp.fit(pop_format)
        fits.append(gp)
    return fits, pops

# def latent_dynamics_analysis(data, tbeg=-.5, tend=2, winsize=.02,
#                              n_factors=8, max_iter=20, min_iter=10):
#     pops = data.get_populations(winsize, tbeg, tend, winsize,
#                                time_zero_field='CUE2_ON_diode')
#     pops, xs = out
#     fits = []
#     for i, pop in enumerate(pops):
#         pop_format = list({'y':pop_j.T, 'ID':j}
#                           for j, pop_j in enumerate(pop))
#         print(pop_format[0]['y'].shape)
#         fit = vlgp.fit(pop_format, n_factors=n_factors, max_iter=max_iter,
#                        min_iter=min_iter)
#         fits.append(fit)
#     return pops, fit, xs 

def compute_diff_dependence(data, targ_field='LABthetaTarget',
                            dist_field='LABthetaDist',
                            resp_field='LABthetaResp'):
    targ = np.concatenate(data[targ_field])
    dist = np.concatenate(data[dist_field])
    resp = np.concatenate(data[resp_field])
    td_diff = u.normalize_periodic_range(targ - dist)
    resp_diff = u.normalize_periodic_range(targ - resp)
    dist_diff = u.normalize_periodic_range(dist - resp)
    return td_diff, resp_diff, dist_diff

bmp = 'swap_errors/behavioral_model/corr_swap_guess.pkl'
bmp_ub = 'swap_errors/behavioral_model/csg_dirich.pkl'
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
ub_prior_dict = {'report_var_var_mean':1,
                 'report_var_var_var':3,
                 'report_var_mean_mean':.64,
                 'report_var_mean_var':1,
                 'swap_weight_mean_mean':0,
                 'swap_weight_mean_var':1}
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

def merge_models(*args):
    out_dict = {}
    for _, m in args:
        for k, v in m.items():
            v_pre = out_dict.get(k, [])
            out_dict[k] = v_pre + v
    return out_dict
