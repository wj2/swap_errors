import os
import numpy as np
import pickle

# import stan
# import pystan as ps
import arviz as az
import scipy.special as spsp
import sklearn.manifold as skm
import sklearn.model_selection as skms
import sklearn.svm as skc
import sklearn.linear_model as sklm
import sklearn.preprocessing as skp
import sklearn.pipeline as sklpipe
import scipy.stats as sts
import scipy.optimize as sopt
import scipy.signal as ssig
import itertools as it
import statsmodels.stats.weightstats as smw
import quantities as pq
import sklearn.metrics.pairwise as skmp

import general.data_io as dio
import general.neural_analysis as na
import general.stan_utility as su
import general.utility as u
import rsatoolbox as rsa

# import general.decoders as gd
import general.plotting as gpl
import swap_errors.auxiliary as swa
import pandas as pd


def decode_error(pickle, p_key="ps", p_thr=0.5, p_ind=0, **kwargs):
    def field_func(session, no_thr=False):
        t = session[p_key][:, p_ind]
        if not no_thr:
            t = t > p_thr
        return t

    return decode_pickle_func(pickle, field_func, balance_targ=True, **kwargs)


def decode_color_pickle(pickle, c_key="c_targ", c_offset=0, **kwargs):
    def field_func(session, no_thr=False):
        t = u.normalize_periodic_range(session[c_key] - c_offset)
        if not no_thr:
            t = t > 0
        return t

    return decode_pickle_func(pickle, c_key, field_func, **kwargs)


def joint_decode_color_pickle(
    pickle,
    p_thr=0.33,
    p_ind=0,
    p_key="ps",
    c_key="rc",
    **kwargs,
):
    def field_func(session, no_thr=False):
        t2 = u.normalize_periodic_range(session[c_key])
        t3 = u.normalize_periodic_range(session[c_key] - np.pi / 2)
        if not no_thr:
            t2 = t2 > 0
            t3 = t3 > 0
        targs = np.stack((t2, t3), axis=1)
        return targs

    def train_func(session):
        return session[p_key][:, p_ind] > p_thr

    def gen_func(session):
        return session[p_key][:, p_ind] < p_thr

    out = decode_pickle_func(
        pickle,
        field_func,
        train_func=train_func,
        gen_func=gen_func,
        **kwargs,
    )
    return out


def joint_decode_pickle(
    pickle,
    p_ind=0,
    p_thr=0.5,
    p_key="ps",
    c_key="c_targ",
    **kwargs,
):
    def field_func(session, no_thr=False):
        t1 = session[p_key][:, p_ind]
        t2 = u.normalize_periodic_range(session[c_key])
        t3 = u.normalize_periodic_range(session[c_key] - np.pi / 2)
        if not no_thr:
            t1 = t1 > p_thr
            t2 = t2 > 0
            t3 = t3 > 0
        targs = np.stack((t1, t2, t3), axis=1)
        return targs

    return decode_pickle_func(pickle, field_func, **kwargs)


def decode_outcome(pickle, p_key="ps", **kwargs):
    def field_func(session, no_thr=False):
        t = np.argmax(session[p_key], axis=1)
        return t

    return decode_pickle_func(pickle, field_func, return_confusion=True, **kwargs)


def decode_pickle_func(
    pickle,
    field_func,
    spk_key="spks",
    n_folds=100,
    balance_targ=False,
    model=skc.LinearSVC,
    return_confusion=False,
    train_func=None,
    gen_func=None,
    **kwargs,
):
    n_ts = list(pickle.values())[0][spk_key].shape[-1]
    perf = np.zeros((len(pickle), n_ts))
    confusion = []
    out_info = {}
    for i, (num, sess) in enumerate(pickle.items()):
        resp = sess[spk_key]
        dec_targ = field_func(sess)
        dec_cont = field_func(sess, no_thr=True)
        if balance_targ:
            rel_flat = dec_targ
            balance_rel_fields = True
        else:
            rel_flat = None
            balance_rel_fields = False
        if gen_func:
            mask_gen = gen_func(sess)
            resp_gen = resp[mask_gen]
            dec_targ_gen = dec_targ[mask_gen]
        else:
            resp_gen = None
            dec_targ_gen = None
        if train_func:
            mask_tr = train_func(sess)
            resp = resp[mask_tr]
            dec_targ = dec_targ[mask_tr]
            if rel_flat is not None:
                rel_flat = rel_flat[mask_tr]

        out = na.fold_skl_shape(
            resp,
            dec_targ,
            n_folds,
            rel_flat=rel_flat,
            model=model,
            balance_rel_fields=balance_rel_fields,
            return_projection=True,
            return_confusion=return_confusion,
            c_gen=resp_gen,
            l_gen=dec_targ_gen,
            **kwargs,
        )
        out_info[num] = {
            "test_inds": out["test_inds"],
            "test_labels": dec_cont[out["test_inds"]],
            "targs": dec_targ,
            "targs_continuous": dec_cont,
            "projection": out["projection"],
            "gen": out.get("score_gen"),
            "projections_gen": out.get("projections_gen"),
            "labels_gen": out.get("labels_gen"),
        }
        perf[i] = out["score"]
        if return_confusion:
            confusion.append(out["confusion"])
    out_dict = {
        "perf": perf,
        "info": out_info,
        "confusion": confusion,
    }
    return out_dict


def average_simplices(
    o_dict,
    plot_type="retro",
    simplex_key="p_err",
    model_path="swap_errors/dirich_avg.pkl",
    model_key="other",
    **kwargs,
):
    samps_all = []
    for k, (fd, data) in o_dict.items():
        samps_k = np.concatenate(fd[model_key].posterior[simplex_key])
        if len(samps_k.shape) > 2:
            ind = swa.get_type_ind(plot_type, data)
            samps_k = samps_k[:, ind]

        samps_all.append(samps_k)
    samps = np.stack(samps_all, axis=1)
    T, N, D = samps.shape
    stan_dict = dict(samps=samps, T=T, N=N, D=D)
    out = su.fit_model(stan_dict, model_path, arviz_convert=False, **kwargs)
    return out


def smooth_dfunc(func, wid, xs, mode="valid"):
    con = np.ones((1,) * (len(func.shape) - 1) + (wid,)) / wid
    out = ssig.convolve(func, con, mode=mode)
    new_xs = ssig.convolve(xs, np.squeeze(con), mode=mode)
    return out, new_xs


def number_decoding(
    data,
    corr_thr,
    swap_thr,
    activity="y",
    p="p",
    corr_ind=0,
    swap_ind=1,
    model=skc.SVC,
    single_str="single",
    type_str="retro",
    type_field="type",
    pre=True,
    n_folds=100,
    test_frac=0.1,
    shuffle=False,
    max_iter=1000,
):
    if pre:
        model = na.make_model_pipeline(model, pca=0.999, class_weight="balanced")
    else:
        model = model(max_iter=max_iter)
    _, single_int = swa.get_type_ind(single_str, data, return_type=True)
    _, type_int = swa.get_type_ind(type_str, data, return_type=True)

    single_mask = data[type_field] == single_int
    double_mask = data[type_field] == type_int
    corr_mask = data[p][:, corr_ind] > corr_thr
    swap_mask = data[p][:, swap_ind] > swap_thr

    double_tr_mask = np.logical_and(double_mask, corr_mask)
    double_te_mask = np.logical_and(double_mask, swap_mask)

    x_tr, y_tr = na.make_data_labels(
        data[activity][double_tr_mask],
        data[activity][single_mask],
    )
    x_swap, y_swap = na.make_data_labels(data[activity][double_te_mask])

    # splitter = skms.ShuffleSplit(n_folds, test_size=test_frac)
    splitter = na.BalancedShuffleSplit(n_folds, test_size=test_frac)
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(y_tr)
    out = skms.cross_validate(model, x_tr, y_tr, return_estimator=True, cv=splitter)
    dec_swap = np.zeros(len(out["test_score"]))
    if x_swap.shape[0] == 0:
        dec_swap[:] = np.nan
    else:
        for i, est in enumerate(out["estimator"]):
            dec_swap[i] = est.score(x_swap, y_swap)
    chance_level = np.mean(y_tr)
    chance_level = max(chance_level, 1 - chance_level)
    return out["test_score"], dec_swap, chance_level


def cue_decoding(
    data,
    corr_thr,
    swap_thr,
    activity="y",
    cue="cue",
    p="p",
    corr_ind=0,
    swap_ind=1,
    model=skc.LinearSVC,
    type_str="retro",
    type_field="type",
    pre=True,
    max_iter=5000,
    n_folds=100,
    test_frac=0.1,
):
    if pre:
        model = na.make_model_pipeline(model, pca=0.999, max_iter=max_iter)
    else:
        model = model(max_iter=max_iter)
    if data["is_joint"] == 1:
        _, type_int = swa.get_type_ind(type_str, data, return_type=True)
        mask = data[type_field] == type_int
    else:
        mask = np.ones(len(data[type_field]), dtype=bool)
    x = data[activity][mask]
    y = data[cue][mask]
    corr_mask = data[p][mask][:, corr_ind] > corr_thr
    swap_mask = data[p][mask][:, swap_ind] > swap_thr

    x_corr, y_corr = x[corr_mask], y[corr_mask]
    splitter = skms.ShuffleSplit(n_folds, test_size=test_frac)
    out = skms.cross_validate(model, x_corr, y_corr, return_estimator=True, cv=splitter)
    x_swap, y_swap = x[swap_mask], y[swap_mask]
    dec_swap = np.zeros(len(out["test_score"]))
    if x_swap.shape[0] == 0:
        dec_swap[:] = np.nan
    else:
        for i, est in enumerate(out["estimator"]):
            dec_swap[i] = est.score(x_swap, y_swap)
    return out["test_score"], dec_swap


def _session_decoding_analysis(fit_dict, corr_thr, swap_thr, func, **kwargs):
    out_dict = {}
    for k, (f, data) in fit_dict.items():
        out_dict[k] = func(data, corr_thr, swap_thr, **kwargs)
    return out_dict


def cue_decoding_swaps(*args, **kwargs):
    return _session_decoding_analysis(*args, cue_decoding, **kwargs)


def number_decoding_swaps(*args, **kwargs):
    return _session_decoding_analysis(*args, number_decoding, **kwargs)


def compare_params(d1, d2, param="p_err", use_type="retro", d1_p_ind=1, d2_p_ind=2):
    diff_distr = {}
    ps = {}
    for k, (fit_d1, _) in d1.items():
        fit_d2, data_d2 = d2[k]
        fit_d1 = fit_d1["other"]
        fit_d2 = fit_d2["other"]
        type_ind = swa.get_type_ind(use_type, data_d2)
        prob_d1 = np.concatenate(fit_d1.posterior[param])
        prob_d2 = np.concatenate(fit_d2.posterior[param])[:, type_ind]
        prob_d1 = 1 - prob_d1[:, d1_p_ind]
        prob_d2 = 1 - prob_d2[:, d2_p_ind]
        dim = min(prob_d1.shape[0], prob_d2.shape[0])
        diff = prob_d2[:dim] - prob_d1[:dim]
        p = diff < 0
        diff_distr[k] = diff
        ps[k] = p
    return diff_distr, ps


def spline_decoding(
    data,
    activity="y",
    col_keys=("C_u",),
    cv=20,
    cv_type=skms.LeaveOneOut,
    model=sklm.Ridge,
):
    targ = np.concatenate(list(data[ck] for ck in col_keys), axis=1)
    pred = data[activity]
    if cv_type is not None:
        cv = cv_type()
    out = skms.cross_val_score(model(), pred, targ, cv=cv)
    return out


def decode_corr_swap_guess(
    data,
    thresh=0.3,
    activity="y",
    type_key="p",
    n_cv=20,
    cv_type=skms.ShuffleSplit,
    test_prop=0.1,
    model=skc.SVC,
):
    p = data[type_key]
    corr_mask = p[:, 0] > (1 - thresh)
    swap_mask = p[:, 1] > thresh
    guess_mask = p[:, 2] > thresh
    neur = data[activity]
    mask_list = (corr_mask, swap_mask, guess_mask)
    labels = ("corr", "swap", "guess")
    out_dict = {}
    out_dict_shuff = {}
    for i, j in it.combinations(range(len(mask_list)), 2):
        c1, c2 = mask_list[i], mask_list[j]
        n_c1 = neur[c1]
        l_c1 = np.zeros(len(n_c1))
        n_c2 = neur[c2]
        l_c2 = np.ones(len(n_c2))

        if len(n_c1) > len(n_c2):
            sub_inds = np.random.choice(np.arange(len(n_c1)), len(n_c2), replace=False)
            n_c1 = n_c1[sub_inds]
            l_c1 = l_c1[sub_inds]
        n = np.concatenate((n_c1, n_c2), axis=0)
        l_ = np.concatenate((l_c1, l_c2), axis=0)
        pipe = na.make_model_pipeline(model=model)
        cv = cv_type(n_cv, test_size=test_prop)
        out = skms.cross_val_score(pipe, n, l_, cv=cv)
        out_dict[(labels[i], labels[j])] = out

        inds = np.arange(len(l_))
        np.random.shuffle(inds)
        out_shuff = skms.cross_val_score(pipe, n, l_[inds], cv=cv)
        out_dict_shuff[(labels[i], labels[j])] = out_shuff
    return out_dict, out_dict_shuff


def session_ll_analysis(session_dict, use_weights=False, **kwargs):
    out_ms = []
    out_sems = []
    k_list = []
    for k, (m, data) in session_dict.items():
        if use_weights:
            weights = data["p"][:, 1]
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
    for i1, i2 in prod:
        k1, k2 = keys[i1], keys[i2]
        l1 = np.mean(model_dict[k1].log_likelihood.y.to_numpy(), axis=(0, 1))
        l2 = np.mean(model_dict[k2].log_likelihood.y.to_numpy(), axis=(0, 1))
        if use_weights is not None:
            if thresh is not None:
                weights = use_weights > thresh
            else:
                weights = use_weights * (use_weights.shape[0] / np.sum(use_weights))
        else:
            weights = None
        delt = l1 - l2
        diff_model = mw(delt, weights=weights, ddof=0)
        m_diff[i1, i2] = diff_model.mean
        sem_diff[i1, i2] = diff_model.std_mean
        names[i1, i2] = (k1, k2)
    return m_diff, sem_diff, names


def _get_key_mu(
    posterior, cols, keys, mean=True, mask=None, inter_key=None, use_ind=None
):
    for i, k in enumerate(keys):
        if mask is not None:
            cols_use = cols[i, mask].T
        else:
            cols_use = cols[i].T
        if use_ind is not None:
            k = k + "_type"
            if mask is not None:
                ui = use_ind[mask]
            else:
                ui = use_ind
            cols_use = np.expand_dims(cols_use, axis=(0, 1, 2))
        else:
            cols_use = np.expand_dims(cols_use, axis=(0, 1))
        arr = np.concatenate(posterior[k].to_numpy(), axis=0)
        arr = np.expand_dims(arr, axis=-1)
        dot_arr = np.sum(arr * cols_use, axis=-2)
        if use_ind is not None:
            dot_arr_new = np.zeros((dot_arr.shape[0],) + dot_arr.shape[2:])
            da0 = dot_arr[:, 0]
            da1 = dot_arr[:, 1]
            dot_arr_new[:, :, ui == 1] = da0[..., ui == 1]
            dot_arr_new[:, :, ui == 2] = da1[..., ui == 2]
            dot_arr = dot_arr_new
        # dot_arr = np.dot(arr, cols_use)
        if i == 0:
            mu_arr = dot_arr
        else:
            mu_arr = mu_arr + dot_arr
    if inter_key is not None and inter_key in posterior.keys():
        inter_pt = np.concatenate(posterior[inter_key].to_numpy(), axis=0)
        if use_ind is not None:
            i_new = np.zeros(mu_arr.shape)
            i0 = inter_pt[:, 0]
            i1 = inter_pt[:, 1]
            i_new[:, :, ui == 1] = np.expand_dims(i0, 2)
            i_new[:, :, ui == 2] = np.expand_dims(i1, 2)
            inter_pt = i_new
        else:
            inter_pt = np.expand_dims(inter_pt, 2)
        mu_arr = mu_arr + inter_pt
    if mean:
        mu_arr = np.mean(mu_arr, axis=0).T
    else:
        mu_arr = np.swapaxes(mu_arr, 1, 2)
    return mu_arr


spatial_mu_config = (
    (("mu_d_u", "mu_l"), ("mu_u", "mu_d_l")),
    (("mu_l", "mu_d_u"), ("mu_d_l", "mu_u")),
)
cue_mu_config = (
    (("mu_d_u", "mu_l"), ("mu_u", "mu_d_l")),
    (("mu_u", "mu_d_l"), ("mu_d_u", "mu_l")),
)


def compute_vecs(
    fit_az,
    data,
    col_keys=("C_u", "C_l"),
    mu_configs=spatial_mu_config,
    cue_key="cue",
    thin_col=1,
):
    cols = np.stack(list(data[ck] for ck in col_keys), axis=0)
    all_cols = np.concatenate(cols, axis=0)
    u_cols = np.unique(list(tuple(col) for col in all_cols), axis=0)[::thin_col]

    n_cols = len(u_cols)
    n_configs = len(mu_configs[0])
    n_units = fit_az.posterior["mu_u"].shape[2]

    vecs = np.zeros((n_cols, n_cols, n_configs, n_units))
    m1s = np.zeros_like(vecs)
    m2s = np.zeros_like(vecs)
    lens = np.zeros((n_cols, n_cols, n_configs))
    for i1, i2 in it.product(range(n_cols), repeat=2):
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


def get_normalized_centroid_distance(
    fit_az,
    data,
    eh_key="err_hat",
    col_keys=("C_u", "C_l"),
    col_rads_keys=("up_col_rads", "down_col_rads"),
    col_guess_rad_key="resp_rads",
    col_key_guess="C_resp",
    cent1_keys=(
        (("mu_d_u", "mu_l"), "intercept_down"),
        (("mu_u", "mu_d_l"), "intercept_up"),
    ),
    cent2_keys=(
        (("mu_l", "mu_d_u"), "intercept_down"),
        (("mu_d_l", "mu_u"), "intercept_up"),
    ),
    resp_key="y",
    cue_key="cue",
    p_thresh=0.5,
    p_key="p",
    p_ind=1,
    eps=1e-3,
    use_cues=True,
    correction_field="p_spa",
    do_correction=False,
    type_key="type",
    trl_filt=None,
    col_thr=np.pi / 2,
    new_joint=False,
    p_comp=np.greater,
    use_resp_color=False,
):
    cols = np.stack(list(data[ck] for ck in col_keys), axis=0)
    cols_rads = np.stack(list(data[ck] for ck in col_rads_keys), axis=0)

    pp = np.concatenate(fit_az.posterior_predictive[eh_key].to_numpy(), axis=0)
    if use_cues:
        cues = data[cue_key]
    else:
        cues = np.zeros(len(data[cue_key]), dtype=int)
    u_cues = np.unique(cues)
    resp = data[resp_key]
    if do_correction and correction_field in fit_az.posterior.keys():
        print("mult")
        p_mult = np.mean(fit_az.posterior[correction_field].to_numpy())
    else:
        p_mult = 1
    if p_thresh is not None:
        mask = p_comp(data[p_key][:, p_ind] * p_mult, p_thresh)
    else:
        mask = np.ones(len(data[p_key]), dtype=bool)
    if trl_filt is not None:
        _, ti = swa.get_type_ind(trl_filt, data, return_type=True)
        mask = np.logical_and(mask, data[type_key] == ti)
    true_arr = []
    pred_arr = []
    p_vals = []
    if col_thr is not None and not use_resp_color:
        col_dist = np.abs(u.normalize_periodic_range(cols_rads[0] - cols_rads[1]))
        col_mask = col_dist > col_thr
        mask = np.logical_and(mask, col_mask)
    elif col_thr is not None and use_resp_color:
        targ_col = np.zeros(len(cols_rads[0]))
        targ_col[cues == 0] = cols_rads[1][cues == 0]
        targ_col[cues == 1] = cols_rads[0][cues == 1]

        guess_col = data[col_guess_rad_key]
        col_dist = np.abs(u.normalize_periodic_range(targ_col - guess_col))
        col_mask = col_dist > col_thr
        mask = np.logical_and(mask, col_mask)

    for i, cue in enumerate(u_cues):
        cue_mask = np.logical_and(mask, cues == cue)
        p_vals.append(data[p_key][cue_mask])
        if new_joint:
            use_ind = data[type_key]
        else:
            use_ind = None
        if use_resp_color:
            alt_cols = np.zeros_like(cols)
            alt_cols_rads = np.zeros_like(cols_rads)
            c_g = data[col_key_guess]
            c_g_rads = data[col_guess_rad_key]
            if cue == 0:
                alt_cols[0] = cols[0]
                alt_cols[1] = c_g
                alt_cols_rads[0] = cols_rads[0]
                alt_cols_rads[1] = c_g_rads
            else:
                alt_cols[0] = c_g
                alt_cols[1] = cols[1]
                alt_cols_rads[0] = c_g_rads
                alt_cols_rads[1] = cols_rads[1]
        else:
            alt_cols = cols
            alt_cols_rads = cols_rads
        mu1 = _get_key_mu(
            fit_az.posterior,
            cols,
            cent1_keys[cue][0],
            mask=cue_mask,
            inter_key=cent1_keys[cue][1],
            use_ind=use_ind,
            mean=True,
        )
        mu2 = _get_key_mu(
            fit_az.posterior,
            alt_cols,
            cent2_keys[cue][0],
            mask=cue_mask,
            inter_key=cent2_keys[cue][1],
            use_ind=use_ind,
            mean=True,
        )

        v_len = np.sqrt(np.sum((mu2 - mu1) ** 2, axis=1))
        v_len[v_len < eps] = 1
        resp_c = resp[cue_mask]
        pp_c = pp[:, cue_mask]
        true_arr_i = np.sum((resp_c - mu1) * (mu2 - mu1), axis=1) / v_len**2
        true_arr.append(true_arr_i)
        pred_arr_i = np.sum((pp_c - mu1) * (mu2 - mu1), axis=2) / v_len**2
        pred_arr.append(pred_arr_i)
        big_mask = np.abs(true_arr_i) > 10
        if np.any(big_mask):
            print(true_arr_i[big_mask])
            print(v_len[big_mask])
            print(resp_c[big_mask])
            print(data["C_u"][cue_mask][big_mask])
            print(data["C_l"][cue_mask][big_mask])
    true_arr_full = np.concatenate(true_arr, axis=0)
    pred_arr_full = np.concatenate(pred_arr, axis=1).flatten()
    p_vals_full = np.concatenate(p_vals, axis=0)
    return true_arr_full, pred_arr_full, p_vals_full


def nonlinear_dimred(
    data,
    tbeg,
    tend,
    twindow=None,
    tstep=None,
    time_key="SAMPLES_ON_diode",
    color_key="LABthetaTarget",
    regions=None,
    dim_red=skm.LocallyLinearEmbedding,
    n_components=2,
    **kwargs,
):
    if twindow is None:
        twindow = tend - tbeg
    pops, xs = data.get_populations(
        twindow, tbeg, tend, tstep, time_zero_field=time_key
    )
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


def get_pareto_k(fit, k_thresh=0.7):
    x = az.loo(fit, pointwise=True)
    k_val = x["pareto_k"]
    k_mask = k_val > k_thresh
    inds = np.where(k_mask)[0]
    trls = fit.observed_data.y[k_mask]
    return k_val, inds, trls


def get_pareto_k_dict(fit_dict, **kwargs):
    k_dict = {}
    for k, v in fit_dict.items():
        k_dict[k] = get_pareto_k(v, **kwargs)
    return k_dict


# def decode_fake_data(n_times, n_neurons, n_trials, n_colors, noise_std=0.1):
#     cols = np.linspace(0, 2 * np.pi, n_colors)
#     x = np.sin(cols)
#     y = np.cos(cols)
#     x_code_m = sts.norm(0, 10).rvs((n_neurons, 1))
#     x_code = x_code_m + sts.norm(0, 1).rvs((1, n_times))
#     y_code = x_code_m + sts.norm(0, 1).rvs((1, n_times))
#     resp_x = np.expand_dims(x_code, -1) * np.expand_dims(x, (0, 1))
#     resp_y = np.expand_dims(y_code, -1) * np.expand_dims(y, (0, 1))
#     resp_m = resp_x + resp_y
#     t_inds = np.random.choice(range(n_colors), n_trials)
#     resps = resp_m[:, :, t_inds]
#     resps = resps + sts.norm(0, noise_std).rvs(resps.shape)
#     resps = np.swapaxes(resps, 1, 2)
#     resps = np.expand_dims(resps, 1)
#     cols = cols[t_inds]
#     resps[..., 0] = sts.norm(0, noise_std).rvs(resps.shape[:-1])
#     out = na.pop_regression_stan(resps, cols, model=gd.PeriodicDecoderTF)
#     xs = np.arange(n_times)
#     return out, xs


def _get_cmean(trls, trl_cols, targ_col, all_cols, color_window=0.2, positions=None):
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
    d = np.sqrt(np.nansum((trl - m) ** 2, axis=0)) / m.shape[0]
    return d


def _get_leftout_color_dists(
    pop_i,
    targ_cols,
    dist_cols,
    upper_samp,
    splitter=skms.LeaveOneOut,
    norm=True,
    u_cols=None,
    color_window=0.2,
    norm_neurons=True,
    return_norm=False,
):
    cols_arr = np.stack((np.array(targ_cols), np.array(dist_cols)), axis=1)
    cols_pos = np.stack(
        (np.array(upper_samp), np.logical_not(np.array(upper_samp))), axis=1
    )
    if u_cols is None:
        u_cols = np.unique(cols_arr)
    if norm:
        m = np.mean(pop_i, axis=2, keepdims=True)
        s = np.std(pop_i, axis=2, keepdims=True)
        s[np.isnan(s)] = 1
        pop_i = (pop_i - m) / s
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
            targ_means[i][uc] = _get_cmean(
                pop_train,
                cols_train[:, 0],
                uc,
                u_cols,
                color_window=color_window,
                positions=pos_train[:, 0],
            )
            dist_means[i][uc] = _get_cmean(
                pop_train,
                cols_train[:, 1],
                uc,
                u_cols,
                color_window=color_window,
                positions=pos_train[:, 1],
            )
        for j in range(pop_test.shape[2]):
            tc = cols_test[j, 0]
            dc = cols_test[j, 1]
            t_pos = pos_test[j, 0]
            d_pos = pos_test[j, 1]
            targ_dists[i, j, 0] = _get_trl_dist(
                pop_test[:, 0, j], targ_means[i][tc][t_pos], norm_neurons=norm_neurons
            )
            targ_dists[i, j, 1] = _get_trl_dist(
                pop_test[:, 0, j], targ_means[i][dc][t_pos], norm_neurons=norm_neurons
            )
            targ_dists[i, j, 2] = _get_trl_dist(
                pop_test[:, 0, j], targ_means[i][tc][d_pos], norm_neurons=norm_neurons
            )
            targ_dists[i, j, 3] = _get_trl_dist(
                pop_test[:, 0, j], targ_means[i][dc][d_pos], norm_neurons=norm_neurons
            )
            vec_dists[i, j, 0] = _get_vec_dist(
                pop_test[:, 0, j], targ_means[i][tc][t_pos], targ_means[i][dc][t_pos]
            )
            vec_dists[i, j, 1] = _get_vec_dist(
                pop_test[:, 0, j], targ_means[i][tc][t_pos], targ_means[i][tc][d_pos]
            )
            vec_dists[i, j, 2] = _get_vec_dist(
                pop_test[:, 0, j], targ_means[i][tc][t_pos], targ_means[i][dc][d_pos]
            )
    out = targ_dists, vec_dists, dist_dists, targ_means, dist_means
    if return_norm and norm:
        out = out + (m, s)
    return out


def _get_vec_dist(pop, targ, dist):
    vec = targ - dist
    cent = np.nanmean(np.stack((targ, dist), axis=0), axis=0)
    mid = np.nansum(vec * cent, axis=0)
    vec_len = np.sqrt(np.nansum(vec**2, axis=0, keepdims=True))

    proj = (np.nansum(pop * vec, axis=0) - mid) / vec_len
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

        def func(x):
            np.mean(x > 0, axis=0)

        for j, dd in enumerate(diff_diff):
            out_i[j] = u.bootstrap_list(dd, func, n=n_boots, out_shape=(ed.shape[-1],))
        outs.append(out_i)
    return outs


def get_test_color_dists(
    data,
    tbeg,
    tend,
    twindow,
    tstep,
    targ_means,
    dist_means,
    time_key="SAMPLES_ON_diode",
    targ_key="LABthetaTarget",
    dist_key="LABthetaDist",
    resp_field="LABthetaResp",
    upper_key="IsUpperSample",
    regions=None,
    norm=True,
    norm_neurons=True,
    m=None,
    s=None,
    err_thr=None,
    use_cache=False,
):
    targ_cols = data[targ_key]
    dist_cols = data[dist_key]
    targ_pos = data[upper_key]
    pops, xs = data.get_populations(
        twindow,
        tbeg,
        tend,
        tstep,
        time_zero_field=time_key,
        skl_axes=True,
        regions=regions,
        cache=use_cache,
    )
    if m is None:
        m = (None,) * len(pops)
        s = (None,) * len(pops)
    out_dists = []
    out_vecs = []
    for i, pop_i in enumerate(pops):
        tc_i = np.array(targ_cols[i])
        dc_i = np.array(dist_cols[i])
        tp_i = np.array(targ_pos[i])
        dp_i = np.logical_not(np.array(targ_pos[i])).astype(int)
        out = compute_dists(
            pop_i,
            tc_i,
            dc_i,
            tp_i,
            dp_i,
            targ_means[i],
            norm_neurons=norm_neurons,
            mean=m[i],
            std=s[i],
        )
        out_dist, out_vec = out
        out_dists.append(out_dist)
        out_vecs.append(out_vec)
    return out_dists, out_vecs


def _argmax_decider(ps, ind):
    mask = np.argmax(ps, axis=1) == ind
    return mask


def _plurality_decider(ps, ind, prob=1 / 3):
    mask = ps[:, ind] > prob
    return mask


def _diff_decider(ps, ind_pos, ind_sub, diff=0):
    mask = (ps[:, ind_pos] - ind_sub[:, ind_sub]) > diff
    return mask


def corr_diff(ps, diff=0):
    return _diff_decider(ps, 0, 1, diff=diff)


def swap_diff(ps, diff=0):
    return _diff_decider(ps, 1, 0, diff=diff)


def guess_diff(ps, diff=0):
    return _diff_decider(ps, 2, 0, diff=diff)


def swap_plurality(ps, prob=1 / 3):
    return _plurality_decider(ps, 1, prob=prob)


def guess_plurality(ps, prob=1 / 3):
    return _plurality_decider(ps, 2, prob=prob)


def corr_plurality(ps, prob=1 / 3):
    return _plurality_decider(ps, 0, prob=prob)


def swap_argmax(ps):
    return _argmax_decider(ps, 1)


def guess_argmax(ps):
    return _argmax_decider(ps, 2)


def corr_argmax(ps):
    return _argmax_decider(ps, 0)


def _col_diff_rad(c1, c2):
    return np.abs(u.normalize_periodic_range(c1 - c2))


def _col_diff_spline(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2, axis=1))


def convert_spline_to_rad(cu, cl):
    cols = np.unique(np.concatenate((cu, cl), axis=0), axis=0)
    if len(cols) != 64:
        print("weird cols", len(cols))
    rads = np.linspace(0, 2 * np.pi, len(cols) + 1)[:-1]
    d = {tuple(x): rads[i] for i, x in enumerate(cols)}
    cu_rad = np.zeros(len(cu))
    cl_rad = np.zeros_like(cu_rad)
    for i, cu_i in enumerate(cu):
        cu_rad[i] = d[tuple(cu_i)]
        cl_rad[i] = d[tuple(cl[i])]
    return cu_rad, cl_rad


def cue_mask_dict(
    data_dict, cue_val, cue_key="cue", mask_keys=("C_u", "C_l", "p", "y", "cue")
):
    new_dict = {}
    new_dict.update(data_dict)
    mask = new_dict[cue_key] == cue_val
    for mk in mask_keys:
        new_dict[mk] = data_dict[mk][mask]
    return new_dict


def filter_nc_dis(
    centroid_dict, use_d1s, cond_types, use_range, regions="all", d2_key="d2"
):
    use_dis = []
    for d1_c in use_d1s:
        if len(list(centroid_dict[d1_c].keys())[0]) > 2:

            def k_filt(k):
                return k[0] in use_range and k[1] == regions

        else:

            def k_filt(k):
                return k[0] in use_range

        d1i_use = {k: v for k, v in centroid_dict[d1_c].items() if k_filt(k)}
        use_dis.append(d1i_use)
    for use_cond in cond_types:
        if len(list(centroid_dict[d2_key].keys())[0]) > 2:

            def k_filt(k):
                return k[0] in use_range and k[1] == regions and k[2] == use_cond

        else:

            def k_filt(k):
                return k[0] in use_range and k[1] == use_cond

        d2i_use = {k: v for k, v in centroid_dict[d2_key].items() if k_filt(k)}
        use_dis.append(d2i_use)
    return use_dis


def compute_centroid_diffs(
    centroid_dict,
    use_d1s=("d1_cl", "d1_cu"),
    session_dict=None,
    cond_types=("pro", "retro"),
    d2_key="d2",
    min_swaps=10,
    regions="all",
):
    if session_dict is None:
        session_dict = dict(
            elmo_range=range(13), waldorf_range=range(13, 24), comb_range=range(24)
        )

    m_labels = []
    m_out = np.zeros((len(session_dict), len(use_d1s) + len(cond_types)))
    p_out = np.zeros_like(m_out)
    for i, (r_key, use_range) in enumerate(session_dict.items()):
        m_labels.append(r_key)
        use_dis = filter_nc_dis(
            centroid_dict,
            use_d1s,
            cond_types,
            use_range,
            regions=regions,
            d2_key=d2_key,
        )
        comb_dicts = []
        for dict_i in use_dis:
            nulls_all = list(v[0] for v in dict_i.values())
            nulls = np.concatenate(nulls_all, axis=0)
            swaps_all = list(np.mean(v[1], axis=0) for v in dict_i.values())
            swaps = np.concatenate(swaps_all, axis=0)
            comb_dicts.append({"comb": (nulls, swaps)})

        for j, dict_i in enumerate(comb_dicts):
            nulls, swaps = dict_i["comb"]
            m_null = np.nanmedian(nulls)
            m_swaps = np.nanmedian(swaps)
            if len(swaps) > min_swaps:
                utest = sts.mannwhitneyu(
                    swaps, nulls, alternative="greater", nan_policy="omit"
                )
                p_out[i, j] = utest.pvalue
            else:
                p_out[i, j] = np.nan
            m_out[i, j] = m_swaps - m_null
    return m_out, p_out


def compute_sweep_ncs(
    sweep_keys,
    run_ind,
    folder="swap_errors/naive_centroids/",
    use_d1s=("d1_cl", "d1_cu"),
    d2_key="d2",
    cond_types=("pro", "retro"),
    monkey_ranges=None,
    regions="all",
    guess=False,
):
    if monkey_ranges is None:
        monkey_ranges = dict(
            elmo_range=range(13), waldorf_range=range(13, 24), comb_range=range(24)
        )
    nc_df = swa.load_nc_sweep(folder, run_ind, guess=guess)
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
        mask = list(
            nc_df[sweep_keys[i]] == av[ind[i]]
            for i, av in enumerate(ax_vals)
            if av[ind[i]] is not None
        )
        mask = np.product(mask, axis=0).astype(bool)
        nc_masked = nc_df[mask].iloc[0]
        nc_ind = nc_masked.to_dict()
        ms, ps = compute_centroid_diffs(
            nc_ind,
            use_d1s=use_d1s,
            cond_types=cond_types,
            d2_key=d2_key,
            session_dict=monkey_ranges,
            regions=regions,
        )
        m_arr[ind] = ms
        p_arr[ind] = ps
    return ax_vals, m_arr, p_arr


def organize_target_swapping(
    folder,
    run_ind,
    sweep_keys=("decider_arg", "avg_dist"),
    res_keys=("d1_cu", "d1_cl", "d2"),
    **kwargs,
):
    st_df = swa.load_fs_sweep(folder, run_ind)
    ax_vals = []
    for sk in sweep_keys:
        if list(st_df[sk])[0] is None:
            app_vals = (None,)
        else:
            app_vals = np.unique(st_df[sk])
        ax_vals.append(app_vals)

    out_arr = np.zeros(list(len(av) for av in ax_vals), dtype=object)
    for ind in u.make_array_ind_iterator(out_arr.shape):
        mask = list(
            st_df[sweep_keys[i]] == av[ind[i]]
            for i, av in enumerate(ax_vals)
            if av[ind[i]] is not None
        )
        mask = np.product(mask, axis=0).astype(bool)
        out_arr[ind] = combine_forgetting(
            st_df[mask], include_keys=res_keys, mid_average=False, merge_keys=False
        )
    return ax_vals, out_arr


def organize_forgetting_swapping(
    folder, run_ind, sweep_keys=("decider_arg",), **kwargs
):
    f_df = swa.load_fs_sweep(folder, run_ind)
    print(f_df.columns)
    print(f_df[sweep_keys[0]])

    if f_df[sweep_keys[0]][0] is None:
        avs = ((None,),)
    else:
        avs = list(np.unique(f_df[sk]) for sk in sweep_keys)

    out_arr = np.zeros(list(len(av) for av in avs), dtype=object)

    for ind in u.make_array_ind_iterator(out_arr.shape):
        masks = []
        print(out_arr.shape, avs)
        for i, sk in enumerate(sweep_keys):
            if avs[i][ind[i]] is None:
                masks.append(np.ones(len(f_df[sk]), dtype=bool))
            else:
                masks.append(f_df[sk] == avs[i][ind[i]])
        mask = np.product(masks, axis=0, dtype=bool)
        df_use = f_df[mask]
        out_arr[ind] = combine_forgetting(df_use, **kwargs)
    return avs, out_arr


def combine_forgetting(
    f_df, include_keys=("forget_cu", "forget_cl"), merge_keys=True, mid_average=True
):
    merge_dict = {}
    for _, row in f_df.iterrows():
        for key in include_keys:
            if merge_keys:
                save_key = "comb"
            else:
                save_key = key
            curr_dict = merge_dict.get(save_key, {})
            for inner_key, vals in row[key].items():
                (new_nulls, new_swaps, dist_nulls, dist_swaps) = vals
                new_swaps = np.nanmean(new_swaps, axis=0)
                dist_swaps = np.nanmean(dist_swaps, axis=0)
                if mid_average:
                    new_swaps = np.nanmean(new_swaps, axis=0, keepdims=True)
                    new_nulls = np.nanmean(new_nulls, axis=0, keepdims=True)

                    dist_nulls = np.nanmean(dist_nulls, axis=0, keepdims=True)
                    dist_swaps = np.nanmean(dist_swaps, axis=0, keepdims=True)

                curr_entry = curr_dict.get(inner_key)
                if curr_entry is None:
                    curr_dict[inner_key] = (
                        new_nulls,
                        new_swaps,
                        dist_nulls,
                        dist_swaps,
                    )
                else:
                    curr_nulls, curr_swaps, cd_nulls, cd_swaps = curr_entry
                    full_nulls = np.concatenate((curr_nulls, new_nulls), axis=0)
                    full_swaps = np.concatenate((curr_swaps, new_swaps), axis=0)
                    fd_nulls = np.concatenate((cd_nulls, dist_nulls), axis=0)
                    fd_swaps = np.concatenate((cd_swaps, dist_swaps), axis=0)
                    curr_dict[inner_key] = (full_nulls, full_swaps, fd_nulls, fd_swaps)
            merge_dict[save_key] = curr_dict
    return merge_dict


def _get_corr_swap_inds(
    ps,
    corr_decider,
    swap_decider,
    and_corr_mask=None,
    and_swap_mask=None,
    and_mask=None,
):
    corr_mask = corr_decider(ps)
    swap_mask = swap_decider(ps)
    common_mask = np.logical_and(corr_mask, swap_mask)
    corr_mask[common_mask] = False
    if and_mask is not None:
        corr_mask = np.logical_and(corr_mask, and_mask)
        swap_mask = np.logical_and(swap_mask, and_mask)
    if and_corr_mask is not None:
        corr_mask = np.logical_and(corr_mask, and_corr_mask)
    swap_mask[common_mask] = False
    if and_swap_mask is not None:
        swap_mask = np.logical_and(swap_mask, and_swap_mask)
    corr_inds = np.where(corr_mask)[0]
    swap_inds = np.where(swap_mask)[0]
    return corr_inds, swap_inds


def swap_corr_guess_rts(
    data,
    p_keys=("corr_prob", "swap_prob", "guess_prob"),
    rt_key="ReactionTime",
    decider_dict=None,
):
    if decider_dict is None:
        decider_dict = {
            "correct": corr_plurality,
            "swap": swap_plurality,
            "guess": guess_plurality,
        }

    ps = data[list(p_keys)]
    rts = data[rt_key]

    out_session_dict = {}
    for k, func in decider_dict.items():
        for i, rt_i in enumerate(rts):
            ps_i = ps[i].to_numpy()
            mask = func(ps_i)
            rt_group = rt_i[mask]
            group_collection = out_session_dict.get(k, [])
            group_collection.append(rt_group)
            out_session_dict[k] = group_collection
    return out_session_dict


def naive_swapping(
    data_dict,
    cu_key="up_col_rads",
    cl_key="down_col_rads",
    cue_key="cue",
    flip_cue=False,
    use_cue=True,
    no_cue_targ="up_col_rads",
    no_cue_dist="down_col_rads",
    tp_key="p",
    cue_targ=1,
    activity_key="y",
    swap_decider=swap_argmax,
    corr_decider=corr_argmax,
    col_exclude=0,
    col_cent=np.pi,
    cv=skms.LeaveOneOut,
    col_diff=_col_diff_rad,
    kernel="rbf",
    convert_splines=True,
    swap_mask=None,
    targeted=False,
    avg_width=np.pi / 2,
):
    if flip_cue:
        no_cue_targ = "down_col_rads"
        no_cue_dist = "up_col_rads"
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

    corr_inds, swap_inds = _get_corr_swap_inds(
        data_dict[tp_key], corr_decider, swap_decider, and_swap_mask=swap_mask
    )
    null_score_targ = np.zeros(len(corr_inds))
    swap_score_targ = np.zeros((len(corr_inds), len(swap_inds)))
    null_score_dist = np.zeros(len(corr_inds))
    swap_score_dist = np.zeros((len(corr_inds), len(swap_inds)))
    y = data_dict[activity_key]
    assert not np.any(np.isin(swap_inds, corr_inds))

    cv_gen = cv()
    for i, (train_inds, test_inds) in enumerate(cv_gen.split(corr_inds)):
        corr_tr, corr_te = corr_inds[train_inds], corr_inds[test_inds]
        model = skc.SVC(kernel=kernel)
        if targeted:
            ns_targ, ns_dist = _target_dec(
                y, c_dec, c_ndec, corr_tr, corr_te, avg_width, kernel=kernel
            )
        else:
            tr_labels = color_cat[corr_tr]
            te_labels = color_cat[corr_te]
            te_dist_labels = color_cat_dist[corr_te]
            y_tr_use = y[corr_tr]
            model.fit(y_tr_use, tr_labels)
            ns_targ = model.score(y[corr_te], te_labels)
            ns_dist = model.score(y[corr_te], te_dist_labels)
        null_score_targ[i], null_score_dist[i] = ns_targ, ns_dist

        if targeted:
            out = _target_dec(
                y, c_dec, c_ndec, corr_tr, swap_inds, avg_width, kernel=kernel
            )
        else:
            out = _non_target_swap_dec(
                model, y[swap_inds], color_cat[swap_inds], color_cat_dist[swap_inds]
            )
        swap_score_targ[i], swap_score_dist[i] = out
    return null_score_targ, swap_score_targ, null_score_dist, swap_score_dist


def make_cats(col1, col2, width, *to_label, to_mask=None):
    labeled = []
    masked = []
    if to_mask is None:
        to_mask = (None,) * len(to_label)
    for i, tl in enumerate(to_label):
        c1_mask = np.abs(u.normalize_periodic_range(tl - col1)) < width
        c2_mask = np.abs(u.normalize_periodic_range(tl - col2)) < width
        tr_mask = np.logical_xor(c1_mask, c2_mask)
        labeled.append(c2_mask[tr_mask])
        if to_mask[i] is not None:
            masked.append(to_mask[i][tr_mask])
        else:
            masked.append(None)
    return labeled, masked


def _target_dec(
    y, targ_cols, dist_cols, corr_tr, test_inds, col_width, model=skc.SVC, **kwargs
):
    score_targ = np.zeros(len(test_inds))
    score_dist = np.zeros_like(score_targ)
    for i, ti in enumerate(test_inds):
        tc, dc = targ_cols[ti], dist_cols[ti]
        if np.abs(u.normalize_periodic_range(tc - dc)) < col_width:
            score_targ[i] = np.nan
            score_dist[i] = np.nan
        else:
            out = make_cats(
                tc, dc, col_width, targ_cols[corr_tr], to_mask=(y[corr_tr],)
            )
            (tr_labels,), (y_tr,) = out
            m = model(**kwargs)
            m.fit(y_tr, tr_labels)
            score_targ[i] = m.predict([y[ti]]) == np.array([0])
            score_dist[i] = np.logical_not(score_targ[i])
    return score_targ, score_dist


def _non_target_swap_dec(model, y_use, color_cat_use, color_cat_dist_use):
    if len(y_use) > 0:
        swap_score_targ = model.predict(y_use) == color_cat_use
        swap_score_dist = model.predict(y_use) == color_cat_dist_use
    else:
        swap_score_targ = np.nan
        swap_score_dist = np.nan
    return swap_score_targ, swap_score_dist


def naive_forgetting(data_dict, cue_key="cue", flip_cue=False, cue_targ=1, **kwargs):
    if flip_cue:
        cue_targ = 0
    cue_mask = data_dict[cue_key] == cue_targ
    out = naive_swapping(
        data_dict,
        cue_key=cue_key,
        use_cue=False,
        flip_cue=flip_cue,
        swap_mask=cue_mask,
        **kwargs,
    )
    return out[:2]


def _compute_trl_c_dist(
    y,
    corr_tr,
    corr_te,
    tr_targ_cols,
    targ_col,
    dist_col,
    col_thr=np.pi / 4,
    col_diff=_col_diff_rad,
    return_centroids=True,
):
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
        dist = np.dot(sv_u, (test_activity - null_cent).T) / sv_len
    else:
        dist = np.nan
        null_cent = np.nan
        swap_cent = np.nan
    if return_centroids:
        out = (dist, null_cent, swap_cent)
    else:
        out = dist
    return out


def _subsample_categories(y_list, l_list, min_c0, min_c1, rng):
    y_all = []
    for i, y in enumerate(y_list):
        l_ = l_list[i]
        c0_inds = rng.choice(np.where(l_ == 0)[0], int(min_c0), replace=False)
        c1_inds = rng.choice(np.where(l_ == 1)[0], int(min_c1), replace=False)
        y_new = np.concatenate((y[c0_inds], y[c1_inds]), axis=0)
        l_new = np.concatenate((l_[c0_inds], l_[c1_inds]), axis=0)
        y_all.append(y_new)
    return y_all, l_new


def _pseudo_split_generator(pop_dict, n_groups=100):
    rng = np.random.default_rng()
    for i in range(n_groups):
        y_tr = []
        l_tr = []

        y_te = []
        l_te = []

        y_sw = []
        l_sw = []

        min_tr_c0 = np.inf
        min_tr_c1 = np.inf

        min_sw_c0 = np.inf
        min_sw_c1 = np.inf
        for k, data_list in pop_dict.items():
            ind = rng.choice(len(data_list))
            y_tr_k, l_tr_k = data_list[ind]["training"]
            min_tr_c1 = np.min([np.sum(l_tr_k), min_tr_c1])
            min_tr_c0 = np.min([np.sum(~l_tr_k), min_tr_c0])

            y_te_k, l_te_k = data_list[ind]["test"]

            y_sw_k, l_sw_k = data_list[ind]["swap"]
            min_sw_c1 = np.min([np.sum(l_sw_k), min_sw_c1])
            min_sw_c0 = np.min([np.sum(~l_sw_k), min_sw_c0])

            y_tr.append(y_tr_k)
            l_tr.append(l_tr_k)
            y_te.append(y_te_k)
            l_te.append(l_te_k)
            y_sw.append(y_sw_k)
            l_sw.append(l_sw_k)

        y_tr, l_tr = _subsample_categories(y_tr, l_tr, min_tr_c0, min_tr_c1, rng)
        y_sw, l_sw = _subsample_categories(y_sw, l_sw, min_sw_c0, min_sw_c1, rng)
        yield {
            "training": (np.concatenate(y_tr, axis=1), l_tr),
            "test": (np.concatenate(y_te, axis=1), l_te[0]),
            "swap": (np.concatenate(y_sw, axis=1), l_sw),
        }


def color_pseudopop(
    session_dict,
    cu_key="up_col_rads",
    cl_key="down_col_rads",
    use_cue=True,
    flip_cue=False,
    no_cue_targ="up_col_rads",
    no_cue_dist="down_col_rads",
    convert_splines=True,
    tp_key="p",
    activity_key="y",
    swap_decider=guess_argmax,
    corr_decider=corr_argmax,
    col_thr=np.pi / 4,
    cv=skms.LeaveOneOut,
    col_diff=_col_diff_rad,
    n_reps=1000,
    model=skc.SVC,
    min_swaps=1,
):
    pop_dict = {}
    for k, data_dict in session_dict.items():
        c_t, c_d = _organize_colors(
            data_dict,
            cu_key=cu_key,
            cl_key=cl_key,
            use_cue=use_cue,
            flip_cue_centroids=flip_cue,
            no_cue_targ=no_cue_targ,
            no_cue_dist=no_cue_dist,
            convert_splines=convert_splines,
        )
        corr_inds, swap_inds = _get_corr_swap_inds(
            data_dict[tp_key], corr_decider, swap_decider
        )
        y = data_dict[activity_key]

        cv_gen = cv()

        targ_col_swap = c_t[swap_inds]
        y_swap = y[swap_inds]
        for i, (train_inds, test_inds) in enumerate(cv_gen.split(corr_inds)):
            corr_tr, corr_te = corr_inds[train_inds], corr_inds[test_inds]

            tr_targ_cols = c_t[corr_tr]
            targ_col = c_t[corr_te]
            dist_col = c_d[corr_te]
            if col_diff(targ_col, dist_col) > col_thr and len(swap_inds):
                labels_tr = col_diff(tr_targ_cols, targ_col) < col_diff(
                    tr_targ_cols, dist_col
                )
                labels_te = np.array([True])
                y_tr = y[train_inds]
                y_te = y[test_inds]

                labels_swap = col_diff(targ_col_swap, targ_col) < col_diff(
                    targ_col_swap, dist_col
                )
                if (
                    np.sum(labels_swap) >= min_swaps
                    and np.sum(~labels_swap) >= min_swaps
                ):
                    set_list = pop_dict.get(k, [])
                    data_split = {
                        "training": (y_tr, labels_tr),
                        "test": (y_te, labels_te),
                        "swap": (y_swap, labels_swap),
                    }
                    set_list.append(data_split)
                    pop_dict[k] = set_list
    corr_score = np.zeros(n_reps)
    swap_score = np.zeros(n_reps)
    for i, data_split in enumerate(_pseudo_split_generator(pop_dict, n_groups=n_reps)):
        m = na.make_model_pipeline(model=model, pca=0.95)
        m.fit(*data_split["training"])

        corr_score[i] = m.score(*data_split["test"])
        swap_score[i] = m.score(*data_split["swap"])
    return corr_score, swap_score


def naive_centroids(*args, shuffle_nulls=False, shuffle_swaps=False, **kwargs):
    if shuffle_nulls or shuffle_swaps:
        out = naive_centroids_shuffle(
            *args, **kwargs, shuffle_nulls=shuffle_nulls, shuffle_swaps=shuffle_swaps
        )
        nds, sds = out[:2]
        if not shuffle_nulls:
            nds = nds[0]
        if not shuffle_swaps:
            sds = sds[0]
        out = (nds, sds) + tuple(out[2:])
    else:
        out = _naive_centroids_inner(*args, **kwargs)
    return out


def naive_centroids_shuffle(*args, n_shuffles=5, **kwargs):
    null_dists_all = []
    swap_dists_all = []
    cents_all = []
    for i in range(n_shuffles):
        nds, sds, cents = _naive_centroids_inner(*args, **kwargs)
        null_dists_all.append(nds)
        swap_dists_all.append(sds)
        cents_all.append(cents)
    out = (np.array(null_dists_all), np.array(swap_dists_all), np.array(cents_all))
    return out


def naive_guessing(
    data_dict,
    cue_key="cue",
    cu_key="up_col_rads",
    cl_key="down_col_rads",
    use_cue=True,
    flip_cue=False,
    no_cue_targ="up_col_rads",
    no_cue_dist="resp_rads",
    tp_key="p",
    guess_key="resp_rads",
    activity_key="y",
    swap_decider=guess_argmax,
    corr_decider=corr_argmax,
    col_thr=np.pi / 4,
    cv=skms.LeaveOneOut,
    col_diff=_col_diff_rad,
    convert_splines=True,
    shuffle_nulls=False,
    shuffle_swaps=False,
):
    c_t, c_d = _organize_colors(
        data_dict,
        cu_key=cu_key,
        cl_key=cl_key,
        use_cue=use_cue,
        flip_cue_guess=flip_cue,
        no_cue_targ=no_cue_targ,
        no_cue_dist=no_cue_dist,
        convert_splines=convert_splines,
    )
    # if flip_cue:
    #     no_cue_targ = "down_col_rads"
    #     no_cue_dist = "resp_rads"
    # if not use_cue:
    #     c_t = np.zeros_like(data_dict[no_cue_targ])
    #     c_d = np.zeros_like(data_dict[no_cue_dist])
    #     c_t[:] = data_dict[no_cue_targ][:]
    #     c_d[:] = data_dict[no_cue_dist][:]
    # else:
    #     c_t = np.zeros_like(data_dict[cu_key])
    #     c_d = np.zeros_like(data_dict[cl_key])

    #     c1_mask = data_dict[cue_key] == 1
    #     c0_mask = data_dict[cue_key] == 0

    #     c_t[c1_mask] = data_dict[cu_key][c1_mask]
    #     c_t[c0_mask] = data_dict[cl_key][c0_mask]

    #     c_d[c1_mask] = data_dict[guess_key][c1_mask]
    #     c_d[c0_mask] = data_dict[guess_key][c0_mask]
    # if len(c_t.shape) > 1 and convert_splines:
    #     c_t, c_d = convert_spline_to_rad(c_t, c_d)

    corr_inds, guess_inds = _get_corr_swap_inds(
        data_dict[tp_key], corr_decider, swap_decider
    )
    y = data_dict[activity_key]
    return _compute_centroid_dists(
        data_dict,
        y,
        corr_inds,
        guess_inds,
        c_t,
        c_d,
        col_thr=col_thr,
        col_diff=col_diff,
        tp_key=tp_key,
        cv=cv,
    )


def _organize_colors(
    data_dict,
    cu_key="up_col_rads",
    cl_key="down_col_rads",
    cue_key="cue",
    no_cue_targ="up_col_rads",
    no_cue_dist="down_col_rads",
    convert_splines=True,
    use_cue=True,
    flip_cue_centroids=False,
    flip_cue_guess=False,
    use_guess=True,
    guess_key="resp_rads",
):
    if flip_cue_centroids:
        no_cue_targ = "down_col_rads"
        no_cue_dist = "up_col_rads"
    elif flip_cue_guess:
        no_cue_targ = "down_col_rads"
        no_cue_dist = "resp_rads"
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

        if use_guess:
            c_d[c1_mask] = data_dict[cl_key][c1_mask]
            c_d[c0_mask] = data_dict[cu_key][c0_mask]
        else:
            c_d[c1_mask] = data_dict[guess_key][c1_mask]
            c_d[c0_mask] = data_dict[guess_key][c0_mask]
    if len(c_t.shape) > 1 and convert_splines:
        c_t, c_d = convert_spline_to_rad(c_t, c_d)
    return c_t, c_d


def _naive_centroids_inner(
    data_dict,
    cue_key="cue",
    cu_key="up_col_rads",
    cl_key="down_col_rads",
    use_cue=True,
    flip_cue=False,
    no_cue_targ="up_col_rads",
    no_cue_dist="down_col_rads",
    tp_key="p",
    activity_key="y",
    swap_decider=swap_argmax,
    corr_decider=corr_argmax,
    col_thr=np.pi / 4,
    cv=skms.LeaveOneOut,
    col_diff=_col_diff_rad,
    convert_splines=True,
    shuffle_nulls=False,
    shuffle_swaps=False,
):
    c_t, c_d = _organize_colors(
        data_dict,
        cu_key=cu_key,
        cl_key=cl_key,
        use_cue=use_cue,
        flip_cue_centroids=flip_cue,
        no_cue_targ=no_cue_targ,
        no_cue_dist=no_cue_dist,
        convert_splines=convert_splines,
    )
    # if flip_cue:
    #     no_cue_targ = "down_col_rads"
    #     no_cue_dist = "up_col_rads"
    # if not use_cue:
    #     c_t = np.zeros_like(data_dict[no_cue_targ])
    #     c_d = np.zeros_like(data_dict[no_cue_dist])
    #     c_t[:] = data_dict[no_cue_targ][:]
    #     c_d[:] = data_dict[no_cue_dist][:]
    # else:
    #     c_t = np.zeros_like(data_dict[cu_key])
    #     c_d = np.zeros_like(data_dict[cl_key])

    #     c1_mask = data_dict[cue_key] == 1
    #     c0_mask = data_dict[cue_key] == 0

    #     c_t[c1_mask] = data_dict[cu_key][c1_mask]
    #     c_t[c0_mask] = data_dict[cl_key][c0_mask]

    #     c_d[c1_mask] = data_dict[cl_key][c1_mask]
    #     c_d[c0_mask] = data_dict[cu_key][c0_mask]
    # if len(c_t.shape) > 1 and convert_splines:
    #     c_t, c_d = convert_spline_to_rad(c_t, c_d)

    corr_inds, swap_inds = _get_corr_swap_inds(
        data_dict[tp_key], corr_decider, swap_decider
    )
    y = data_dict[activity_key]
    return _compute_centroid_dists(
        data_dict,
        y,
        corr_inds,
        swap_inds,
        c_t,
        c_d,
        col_thr=col_thr,
        col_diff=col_diff,
        cv=cv,
        tp_key=tp_key,
    )


def _compute_centroid_dists(
    data_dict,
    y,
    corr_inds,
    swap_inds,
    c_t,
    c_d,
    col_thr=np.pi / 4,
    col_diff=_col_diff_rad,
    cv=skms.LeaveOneOut,
    tp_key="p",
    shuffle_swaps=False,
    shuffle_nulls=False,
):
    rng = np.random.default_rng()
    null_dists = np.zeros(len(corr_inds))
    swap_dists = np.zeros((len(corr_inds), len(swap_inds)))

    null_cent = np.zeros((len(corr_inds), y.shape[1]))
    swap_cent = np.zeros_like(null_cent)
    s_null_cent = np.zeros((len(corr_inds), len(swap_inds), y.shape[1]))
    s_swap_cent = np.zeros_like(s_null_cent)

    cv_gen = cv()
    null_ps = np.zeros((len(corr_inds), 3))
    swap_ps = data_dict[tp_key][swap_inds]
    for i, (train_inds, test_inds) in enumerate(cv_gen.split(corr_inds)):
        null_ps[i] = data_dict[tp_key][corr_inds[test_inds]]
        corr_tr, corr_te = corr_inds[train_inds], corr_inds[test_inds]
        tr_targ_cols = c_t[corr_tr]
        targ_col = c_t[corr_te]
        if shuffle_nulls:
            dist_col = rng.choice(c_d)
        dist_col = c_d[corr_te]
        out = _compute_trl_c_dist(
            y,
            corr_tr,
            corr_te,
            tr_targ_cols,
            targ_col,
            dist_col,
            col_thr=col_thr,
            col_diff=col_diff,
        )
        null_dists[i], null_cent[i], swap_cent[i] = out
        for j, si in enumerate(swap_inds):
            targ_col, dist_col = c_t[si], c_d[si]
            if shuffle_swaps:
                dist_col = rng.choice(c_d)

            out = _compute_trl_c_dist(
                y,
                corr_tr,
                si,
                tr_targ_cols,
                targ_col,
                dist_col,
                col_thr=col_thr,
                col_diff=col_diff,
            )
            swap_dists[i, j], s_null_cent[i, j], s_swap_cent[i, j] = out
    s_null_cent = np.mean(s_null_cent, axis=0)
    s_swap_cent = np.mean(s_swap_cent, axis=0)
    cents_all = ((null_cent, swap_cent), (s_null_cent, s_swap_cent))
    ps = (null_ps, swap_ps)
    return null_dists, swap_dists, cents_all, ps


def _project_preds(preds, samps, use_center=False):
    """
    Parameters
    ----------
    preds : list of array, T x N
        where T is the number of trials and N is the number of dimensions and the list
    samps : array, T x N
        where T and N are the same as above
    """
    n_preds = len(preds)
    out = np.zeros((n_preds, n_preds, len(samps)))
    for i, j in it.product(range(n_preds), repeat=2):
        if i != j:
            p_i, p_j = preds[i], preds[j]
            swap_vec = p_j - p_i
            sv_len = np.sqrt(np.sum(swap_vec**2, axis=1))
            swap_vec_u = u.make_unit_vector(swap_vec, squeeze=False)
            if use_center:
                sub_c = np.mean([p_i, p_j], axis=0)
            else:
                sub_c = p_i
            dist = swap_vec_u * (samps - sub_c)
            dist = np.sum(dist, axis=1) / sv_len

            out[i, j] = dist
    return out


def _fit_cn_lm_tc(tr_data, te_data, model=sklm.Ridge, pre_pipe=None):
    tr_coeffs, tr_y = tr_data
    n_ts = tr_y.shape[-1]

    te_coeffs, te_y = te_data

    dists = np.zeros(n_ts)
    for i in range(n_ts):
        tr_y_i = tr_y[..., i]
        y_pair_i = te_y[..., i]

        if pre_pipe is not None:
            tr_y_i = pre_pipe.fit_transform(tr_y_i)
            y_pair_i = pre_pipe.transform(y_pair_i)
        m = model()
        m.fit(tr_coeffs, tr_y_i)

        y_te_pred = m.predict(te_coeffs)
        v1 = np.diff(y_pair_i, axis=0)[0]
        v2 = np.diff(y_te_pred, axis=0)[0]
        dists[i] = v1 @ v2
    return dists


def _fit_lm_tc_model(tr_pair, null_pairs, swap_pairs, model=sklm.Ridge, pre_pipe=None):
    tr_coeffs, tr_y = tr_pair

    n_ts = tr_y.shape[-1]

    n_te_combs = len(null_pairs[0])
    n_te_trls = null_pairs[1].shape[0]
    null_proj = np.zeros((n_te_combs, n_te_combs, n_te_trls, n_ts))

    n_swap_combs = len(swap_pairs[0])
    n_swap_trls = swap_pairs[1].shape[0]
    swap_proj = np.zeros((n_swap_combs, n_swap_combs, n_swap_trls, n_ts))

    for i in range(n_ts):
        tr_y_i = tr_y[..., i]
        null_pairs_i = (null_pairs[0], null_pairs[1][..., i])
        swap_pairs_i = (swap_pairs[0], swap_pairs[1][..., i])

        if pre_pipe is not None:
            tr_y_i = pre_pipe.fit_transform(tr_y_i)
            null_pairs_i = (null_pairs_i[0], pre_pipe.transform(null_pairs_i[1]))
            swap_pairs_i = (swap_pairs_i[0], pre_pipe.transform(swap_pairs_i[1]))
        m = model()
        m.fit(tr_coeffs, tr_y_i)

        null_preds = list(m.predict(ap) for ap in null_pairs_i[0])
        null_proj[..., i] = _project_preds(null_preds, null_pairs_i[1])
        swap_preds = list(m.predict(ap) for ap in swap_pairs_i[0])
        swap_proj[..., i] = _project_preds(swap_preds, swap_pairs_i[1])
    return null_proj, swap_proj


retro_sequences = {
    "color presentation": (-0.75, 1, "SAMPLES_ON_diode", False, True),
    "pre-cue presentation": (
        -0.8,
        0,
        "CUE2_ON_diode",
        False,
        True,
    ),
    "post-cue presentation": (
        -0.5,
        0.8,
        "CUE2_ON_diode",
        True,
        True,
    ),
    "wheel presentation": (-1, 0, "WHEEL_ON_diode", True, True),
}
pro_sequences = {
    "cue presentation": (
        -0.5,
        0.5,
        "CUE1_ON_diode",
        True,
        False,
    ),
    "pre-color presentation": (-0.5, 0, "SAMPLES_ON_diode", True, False),
    "post-color presentation": (-0.5, 1.5, "SAMPLES_ON_diode", True, True),
    "wheel presentation": (-1, 0, "WHEEL_ON_diode", True, True),
}


all_regions = ("7ab", "fef", "motor", "pfc", "tpot", "v4pit")
single_region_subsets = {r: (r,) for r in all_regions}
sub_region_subsets = {
    "no_{}".format(r): tuple(x for x in all_regions if x != r) for r in all_regions
}
all_region_subset = {"all": all_regions}


def prepare_lm_tc_pops(
    *args,
    region_subsets=single_region_subsets,
    rt_thresh=None,
    out_folder="../data/swap_errors/lm_data",
    **kwargs,
):
    if rt_thresh is not None:
        add_orig = "rt{}_".format(rt_thresh)
    else:
        add_orig = ""
    for k, sub in region_subsets.items():
        retro_dict = make_lm_tc_pops(*args, regions=sub, **kwargs)
        save_lm_tc_pops(
            retro_dict, add=add_orig + "retro_{}".format(k), out_folder=out_folder
        )

        pro_dict = make_lm_tc_pops(
            *args,
            regions=sub,
            use_pro=True,
            **kwargs,
        )
        save_lm_tc_pops(
            pro_dict, add=add_orig + "pro_{}".format(k), out_folder=out_folder
        )

        single_dict = make_lm_tc_pops(
            *args,
            regions=sub,
            use_single=True,
            **kwargs,
        )
        save_lm_tc_pops(
            single_dict, add=add_orig + "single_{}".format(k), out_folder=out_folder
        )


panichello_save_keys = {
    "uc": "upper_color",
    "lc": "lower_color",
    "rc": "LABthetaResp",
    "cues_alt": "IsUpperSample",
    "saccade_angle": "ResponseTheta",
    "regions": "neur_regions",
    "ps": ("corr_prob", "swap_prob", "guess_prob"),
    "rt": "ReactionTime",
}


def make_lm_tc_pops(
    data,
    use_pro=False,
    use_single=False,
    winsize=0.5,
    tstep=0.05,
    pro_sequences=pro_sequences,
    retro_sequences=retro_sequences,
    save_keys=panichello_save_keys,
    rt_key="ReactionTime",
    rt_thresh=None,
    **kwargs,
):
    if use_pro:
        data = pro_mask(data)
        sequence = pro_sequences
    elif use_single:
        data = single_mask(data)
        sequence = retro_sequences
    else:
        data = retro_mask(data)
        sequence = retro_sequences
    if rt_thresh is not None:
        mask = data[rt_key] < rt_thresh
        data = data.mask(mask)
    pop_dict = {}
    for k, (tbeg, tend, tzf, cue_on, color_on) in sequence.items():
        spks, xs = data.get_populations(
            winsize,
            tbeg,
            tend,
            tstep,
            time_zero_field=tzf,
            **kwargs,
        )
        out_dict = {}
        for k, ref in save_keys.items():
            if u.check_list(k):
                k = list(k)
            out_dict[k] = data[k]
        if "cues_alt" in out_dict.keys() and cue_on:
            out_dict["cues"] = out_dict["cues_alt"]
        else:
            out_dict["cues"] = (None,) * len(spks)
        if "regions" in out_dict.keys():
            out_dict["regions"] = list(
                x.iloc[0].to_numpy() for x in out_dict["regions"]
            )
        out_dict["other"] = list({"xs": xs, "sequence": k} for _ in spks)
        pop_dict[k] = out_dict
    return pop_dict


def save_lm_tc_pops(
    pop_dict,
    add="retro",
    out_path="lmtc_{}_{}_{}.pkl",
    out_folder="../data/swap_errors/lm_data/",
):
    for k, pd_k in pop_dict.items():
        k_save = k.replace(" ", "-")
        n_sessions = len(pd_k["spks"])
        for i in range(n_sessions):
            path = os.path.join(out_folder, out_path.format(add, k_save, i))
            sd = {k: v[i] for k, v in pd_k.itmes()}
            pickle.dump(sd, open(path, "wb"))


def distance_lm_tc_frompickle(
    path,
    out_folder=".",
    prefix="fit_dist_",
    jobid="0000",
    **kwargs,
):
    sd = pd.read_pickle(open(path, "rb"))
    _, name = os.path.split(path)
    name, ext = os.path.splitext(name)
    new_name = prefix + name + "_{}".format(jobid) + ext
    out_path = os.path.join(out_folder, new_name)

    use_keys = ("spks", "uc", "lc", "ps", "cues")
    args = list(sd.pop(uk) for uk in use_keys)
    spks, uc, lc, ps, cues = args
    other_info = sd.pop("other")
    xs = other_info["xs"]

    dist_mat = distance_lm_tc(*args, **sd, **kwargs)

    out_dict = {
        "dist_mat": dist_mat,
        "args": args,
        "kwargs": kwargs,
        "xs": xs,
        "other": other_info,
        "sd": sd,
    }
    pickle.dump(out_dict, open(out_path, "wb"))
    return out_path, out_dict


def swap_lm_tc_frompickle(path, out_folder=".", prefix="fit_", jobid="0000", **kwargs):
    sd = pickle.load(open(path, "rb"))
    _, name = os.path.split(path)
    name, ext = os.path.splitext(name)
    new_name = prefix + name + "_{}".format(jobid) + ext
    out_path = os.path.join(out_folder, new_name)

    use_keys = ("spks", "uc", "lc", "ps", "cues")
    args = list(sd.pop(uk) for uk in use_keys)
    spks, uc, lc, ps, cues = args
    other_info = sd.pop("other")
    xs = other_info["xs"]

    null_color, swap_color = swap_lm_tc(*args, **sd, **kwargs)

    if cues is not None:
        null_cue, swap_cue = swap_cue_tc(spks, ps, cues.to_numpy(), **kwargs)
    else:
        null_cue, swap_cue = np.zeros((0, 0, len(xs))), np.zeros((0, 0, len(xs)))

    out_dict = {
        "null_color": null_color,
        "swap_color": swap_color,
        "null_cue": null_cue,
        "swap_cue": swap_cue,
        "args": args,
        "kwargs": kwargs,
        "xs": xs,
        "other": other_info,
        "sd": sd,
    }
    pickle.dump(out_dict, open(out_path, "wb"))
    return out_path, out_dict


def swap_lm_tc_null_frompickle(
    region_path,
    null_path,
    out_folder=".",
    prefix="fit_nulls_",
    jobid="0000",
    n_reps=2,
    **kwargs,
):
    sd = pickle.load(open(region_path, "rb"))
    sd_null = pickle.load(open(null_path, "rb"))
    _, name = os.path.split(region_path)
    name, ext = os.path.splitext(name)
    new_name = prefix + name + "_{}".format(jobid) + ext
    out_path = os.path.join(out_folder, new_name)

    use_keys = ("spks", "uc", "lc", "ps", "cues")
    args = list(sd.pop(uk) for uk in use_keys)
    spks, uc, lc, ps, cues = args
    other_info = sd.pop("other")
    xs = other_info["xs"]

    args_null = list(sd_null.pop(uk) for uk in use_keys)
    spks_null, uc_null, lc_null, ps_null, cues_null = args_null
    other_info = sd_null.pop("other")
    xs_null = other_info["xs"]

    subsample_neurs = spks.shape[1]
    tot_neurs = spks_null.shape[1]
    rng = np.random.default_rng()

    ncue_nulls = np.zeros((n_reps, len(xs)))
    scue_nulls = np.zeros_like(ncue_nulls)
    for i in range(n_reps):
        print(i)
        inds = rng.choice(tot_neurs, size=subsample_neurs, replace=False)
        spks_null_i = spks_null[:, inds]
        args_null_i = spks_null_i, uc_null, lc_null, ps_null, cues_null
        nc_null, sc_null = swap_lm_tc(*args_null_i, **sd_null, **kwargs)
        if i == 0:
            nc_nulls = np.zeros((n_reps,) + nc_null.shape[1:3] + (len(xs_null),))
            sc_nulls = np.zeros_like(nc_nulls)
        nc_nulls[i] = np.mean(nc_null, axis=(0, 3))
        sc_nulls[i] = np.mean(sc_null, axis=(0, 3))
        if cues is not None:
            ncue_null, scue_null = swap_cue_tc(
                spks_null_i, ps_null, cues_null.to_numpy(), **kwargs
            )
        else:
            ncue_null, scue_null = np.zeros((0, 0, len(xs))), np.zeros((0, 0, len(xs)))
        ncue_nulls[i] = np.mean(ncue_null, axis=(0, 1))
        scue_nulls[i] = np.mean(scue_null, axis=(0, 1))

    out_dict = {
        "null_color": nc_nulls,
        "swap_color": sc_nulls,
        "null_cue": ncue_nulls,
        "swap_cue": scue_nulls,
        "args": args,
        "kwargs": kwargs,
        "xs": xs,
        "other": other_info,
        "sd": sd,
    }
    pickle.dump(out_dict, open(out_path, "wb"))
    return out_path, out_dict


def fit_lm_tc_all(full_pd, **kwargs):
    outs = {}
    for k, pd_ in full_pd.items():
        outs[k] = fit_lm_tc_pop_dicts(pd_)
    return outs


def fit_lm_tc_pop_dicts(pd, use_waldorf=False, **kwargs):
    nulls = []
    swaps = []
    if use_waldorf:
        sess_range = range(13, 23)
    else:
        sess_range = range(13)
    for sess_ind in sess_range:
        spks, uc, lc, ps, cue = list(x[sess_ind] for x in pd)[:-1]
        xs = pd[-1]
        uc = uc.to_numpy()
        lc = lc.to_numpy()
        ps = ps.to_numpy()

        null_traj, swap_traj = swap_lm_tc(spks, uc, lc, ps, cues=cue, **kwargs)
        nulls.append(null_traj)
        swaps.append(np.mean(swap_traj, axis=0))

    swaps_comb = np.concatenate(swaps, axis=2)
    nulls_comb = np.concatenate(nulls, axis=0)
    nulls_comb = np.squeeze(np.swapaxes(nulls_comb, 0, 3))

    return (nulls_comb, swaps_comb), (nulls, swaps), xs


def fit_cue_tc_pop_dicts(pd, use_waldorf=False, **kwargs):
    nulls = []
    swaps = []
    if use_waldorf:
        sess_range = range(13, 23)
    else:
        sess_range = range(13)
    for sess_ind in sess_range:
        spks, _, _, ps, cue = list(x[sess_ind] for x in pd)[:-1]
        xs = pd[-1]
        ps = ps.to_numpy()

        if cue is not None:
            null_traj, swap_traj = swap_cue_tc(spks, ps, cue.to_numpy(), **kwargs)
        else:
            null_traj, swap_traj = np.zeros((0, 0, len(xs))), np.zeros((0, 0, len(xs)))
        nulls.append(null_traj)
        swaps.append(np.mean(swap_traj, axis=0))

    swaps_comb = np.concatenate(swaps, axis=0)
    nulls_comb = np.concatenate(nulls, axis=0)
    nulls_comb = np.squeeze(nulls_comb)

    return (nulls_comb, swaps_comb), (nulls, swaps), xs


def _fit_cue_tc_model(train_pair, null_pair, swap_pair, model=None):
    if model is None:
        model = skc.LinearSVC()
    tr_cues, tr_y = train_pair
    te_cues, te_y = null_pair
    swap_cues, swap_y = swap_pair

    null_flipper = np.sign(te_cues - 0.5)
    swap_flipper = np.sign(swap_cues - 0.5)

    n_ts = tr_y.shape[-1]

    n_te_trls = null_pair[1].shape[0]
    null_proj = np.zeros((n_te_trls, n_ts))

    n_swap_trls = swap_pair[1].shape[0]
    swap_proj = np.zeros((n_swap_trls, n_ts))
    for i in range(n_ts):
        model.fit(tr_y[..., i], tr_cues)
        null_proj[:, i] = null_flipper * model.decision_function(te_y[..., i])
        swap_proj[:, i] = swap_flipper * model.decision_function(swap_y[..., i])
    return null_proj, swap_proj


def swap_cue_tc(
    y,
    ps,
    cues,
    p_swap_ind=1,
    p_corr_ind=0,
    swap_decider=swap_argmax,
    corr_decider=corr_argmax,
    cv=skms.LeaveOneOut,
    model=skc.LinearSVC,
    norm=True,
    pre_pca=None,
    max_iter=2000,
    **kwargs,
):
    corr_inds, swap_inds = _get_corr_swap_inds(
        ps,
        corr_decider,
        swap_decider,
    )
    n_trls, n_neurs, n_ts = y.shape

    null_dists = np.zeros((len(corr_inds), 1, n_ts))
    swap_dists = np.zeros((len(corr_inds), len(swap_inds), n_ts))

    model = na.make_model_pipeline(
        model,
        norm=norm,
        pca=pre_pca,
        max_iter=max_iter,
    )
    swap_pair = (cues[swap_inds], y[swap_inds])

    if len(swap_inds) == 0:
        null_dists[:] = np.nan
        swap_dists[:] = np.nan
    else:
        cv_gen = cv()
        for i, (train_inds, test_inds) in enumerate(cv_gen.split(corr_inds)):
            corr_tr, corr_te = corr_inds[train_inds], corr_inds[test_inds]

            tr_labels = cues[corr_tr]
            te_labels = cues[corr_te]

            y_corr_tr = y[corr_tr]
            y_corr_te = y[corr_te]

            train_pair = (tr_labels, y_corr_tr)
            null_pair = (te_labels, y_corr_te)

            out = _fit_cue_tc_model(
                train_pair,
                null_pair,
                swap_pair,
                model=model,
            )
            null_dists[i] = out[0]
            swap_dists[i] = out[1]
    return null_dists, swap_dists


def lm_tc(
    y,
    upper_col,
    lower_col,
    ps,
    cues=None,
    p_swap_ind=1,
    p_corr_ind=0,
    spline_order=1,
    n_knots=5,
    swap_decider=swap_argmax,
    corr_decider=corr_argmax,
    col_thr=np.pi / 4,
    col_diff=_col_diff_rad,
    model=sklm.Ridge,
    single_color=False,
    norm=True,
    pre_pca=None,
):
    if cues is not None:
        targ_col = np.zeros_like(upper_col)
        dist_col = np.zeros_like(upper_col)
        targ_col[cues == 1] = upper_col[cues == 1]
        targ_col[cues == 0] = lower_col[cues == 0]
        dist_col[cues == 1] = lower_col[cues == 1]
        dist_col[cues == 0] = upper_col[cues == 0]
        upper_col = targ_col
        lower_col = dist_col

    null_colors = (upper_col, lower_col)
    if single_color:
        null_colors = null_colors[:1]
    null_coeffs, spliner = make_lm_coefficients(
        *null_colors,
        cues=cues,
        spline_knots=n_knots,
        spline_degree=spline_order,
        return_spliner=True,
    )

    swap_colors = null_colors[::-1]
    if single_color:
        swap_colors = swap_colors[:1]
    color_swap_coeffs = make_lm_coefficients(
        *swap_colors,
        cues=cues,
        spline_knots=n_knots,
        spline_degree=spline_order,
        use_spliner=spliner,
    )
    alternates = (
        null_coeffs,
        color_swap_coeffs,
    )
    if cues is not None:
        cue_swap_coeffs = make_lm_coefficients(
            *swap_colors,
            cues=1 - cues,
            spline_knots=n_knots,
            spline_degree=spline_order,
            use_spliner=spliner,
        )
        cue_swap_null_coeffs = make_lm_coefficients(
            *null_colors,
            cues=1 - cues,
            spline_knots=n_knots,
            spline_degree=spline_order,
            use_spliner=spliner,
        )
        alternates = alternates + (cue_swap_coeffs, cue_swap_null_coeffs)

    col_dist_mask = col_diff(lower_col, upper_col) > col_thr
    corr_inds, swap_inds = _get_corr_swap_inds(
        ps,
        corr_decider,
        swap_decider,
        and_mask=col_dist_mask,
    )

    pipe = na.make_model_pipeline(norm=norm, pca=pre_pca, post_norm=False)
    y_trs = pipe.fit_transform(y[corr_inds])
    m = model()
    m.fit(null_coeffs[corr_inds], y_trs)

    def resp_gen(cu, cl, cue=None):
        if not u.check_list(cu):
            cu = np.array([cu])
        if not u.check_list(cl):
            cl = np.array([cl])
        coeffs = make_lm_coefficients(cu, cl, cues=cue, use_spliner=spliner)
        return coeffs, m.predict(coeffs)

    return null_coeffs, resp_gen


def coeff_threshold(fg1, fg2, col_thr=np.pi / 4, col_diff=_col_diff_rad):
    c11, c21, cue1 = fg1
    c12, c22, cue2 = fg2

    c11 = np.expand_dims(c11, 1)
    c21 = np.expand_dims(c21, 1)
    cue1 = np.expand_dims(cue1, 1)

    c12 = np.expand_dims(c12, 0)
    c22 = np.expand_dims(c22, 0)
    cue2 = np.expand_dims(cue2, 0)

    c1_close = col_diff(c11, c12) < col_thr
    c2_close = col_diff(c21, c22) < col_thr
    same_cue = cue1 == cue2

    mask = c1_close * c2_close * same_cue
    inds = np.stack(np.where(mask), axis=1)
    mask = np.squeeze(np.diff(inds, axis=1) > 0)
    inds = inds[mask]
    return inds


def distance_lm_tc(
    y,
    upper_col,
    lower_col,
    ps,
    cues=None,
    p_swap_ind=1,
    p_corr_ind=0,
    spline_order=1,
    n_knots=5,
    corr_decider=corr_argmax,
    swap_decider=swap_argmax,
    close_coeffs_decider=coeff_threshold,
    col_thr=np.pi / 4,
    col_diff=_col_diff_rad,
    model=sklm.Ridge,
    single_color=False,
    norm=True,
    pre_pca=None,
    model_based=True,
    min_trls=2,
):
    """
    Parameters
    ----------
    y : array N x K x T
        Array where N is the number of trials, K is the number of neurons and T is the
        number of time points
    upper_col, lower_col : array, N
        Array giving upper color
    ps : array, N x 3
    cues : array, N
    """
    if cues is not None:
        targ_col = np.zeros_like(upper_col)
        dist_col = np.zeros_like(upper_col)
        targ_col[cues == 1] = upper_col[cues == 1]
        targ_col[cues == 0] = lower_col[cues == 0]
        dist_col[cues == 1] = lower_col[cues == 1]
        dist_col[cues == 0] = upper_col[cues == 0]
        upper_col = targ_col
        lower_col = dist_col
        try:
            cues = cues.to_numpy()
        except AttributeError:
            pass

    n_trls, n_neurs, n_ts = y.shape
    main_colors = (upper_col, lower_col)
    null_colors_use = main_colors
    if single_color:
        null_colors_use = main_colors[:1]
    null_coeffs, spliner = make_lm_coefficients(
        *null_colors_use,
        cues=cues,
        spline_knots=n_knots,
        spline_degree=spline_order,
        return_spliner=True,
    )

    swap_colors_use = main_colors[::-1]
    if single_color:
        swap_colors_use = swap_colors_use[:1]
    color_swap_coeffs = make_lm_coefficients(
        *swap_colors_use,
        cues=cues,
        spline_knots=n_knots,
        spline_degree=spline_order,
        use_spliner=spliner,
    )
    alternates = (
        null_coeffs,
        color_swap_coeffs,
    )
    alternates_raw = (null_colors_use + (cues,), swap_colors_use + (cues,))
    if cues is not None:
        cue_swap_coeffs = make_lm_coefficients(
            *swap_colors_use,
            cues=1 - cues,
            spline_knots=n_knots,
            spline_degree=spline_order,
            use_spliner=spliner,
        )
        cue_swap_null_coeffs = make_lm_coefficients(
            *null_colors_use,
            cues=1 - cues,
            spline_knots=n_knots,
            spline_degree=spline_order,
            use_spliner=spliner,
        )
        alternates = alternates + (cue_swap_coeffs, cue_swap_null_coeffs)
        add = (swap_colors_use + (1 - cues,), null_colors_use + (1 - cues,))
        alternates_raw = alternates_raw + add

    col_dist_mask = col_diff(*main_colors) > col_thr
    corr_inds, _ = _get_corr_swap_inds(
        ps,
        corr_decider,
        swap_decider,
        and_mask=col_dist_mask,
    )
    alternates = list(alt[corr_inds] for alt in alternates)
    alternates_raw = list(
        list(alt_i[corr_inds] for alt_i in alt) for alt in alternates_raw
    )
    y = y[corr_inds]

    n_alts = len(alternates)
    n_trls = len(corr_inds)
    dists = np.zeros((n_alts, n_alts), dtype=object)

    null_coeffs_raw = np.array(alternates_raw[0])
    trl_inds = set(np.arange(n_trls))
    for i_alt, j_alt in it.combinations(range(n_alts), 2):
        ar_i, ar_j = alternates_raw[i_alt], alternates_raw[j_alt]
        close_inds = close_coeffs_decider(ar_i, ar_j)
        dists_ij = np.zeros((len(close_inds), n_ts))
        for i, te_inds in enumerate(close_inds):
            pipe = na.make_model_pipeline(norm=norm, pca=pre_pca, tc=not model_based)
            tr_inds = np.array(list(trl_inds.difference(te_inds)))
            if model_based:
                coeff_tr = null_coeffs[tr_inds]
                y_tr = y[tr_inds]

                c_pair = null_coeffs[te_inds]
                y_te = y[te_inds]

                dists_ij[i] = _fit_cn_lm_tc(
                    (coeff_tr, y_tr), (c_pair, y_te), pre_pipe=pipe
                )
            else:
                y_tr = y[tr_inds]
                y_tr = pipe.fit_transform(y_tr)
                y_te = pipe.transform(y[te_inds])
                t0, t1 = te_inds
                t0_tr_inds = close_coeffs_decider(
                    null_coeffs_raw[:, t0 : t0 + 1],
                    null_coeffs_raw[:, tr_inds],
                )
                t1_tr_inds = close_coeffs_decider(
                    null_coeffs_raw[:, t1 : t1 + 1],
                    null_coeffs_raw[:, tr_inds],
                )
                if len(t0_tr_inds) > min_trls and len(t1_tr_inds) > min_trls:
                    y_tr0 = np.mean(y_tr[t0_tr_inds[:, 1]], axis=0)
                    y_tr1 = np.mean(y_tr[t1_tr_inds[:, 1]], axis=0)
                    y_te0 = y_te[0]
                    y_te1 = y_te[1]
                    dists_ij[i] = np.sum((y_tr0 - y_tr1) * (y_te0 - y_te1), axis=0)
                else:
                    dists_ij[i] = np.nan

        dists[i_alt, j_alt] = dists_ij
        dists[j_alt, i_alt] = dists_ij
    return dists


def make_mean_dist_mat(trl_dist_mat, xs, central=np.nanmean):
    dm_mu = np.zeros(trl_dist_mat.shape + (len(xs),))
    for session, i, j in u.make_array_ind_iterator(trl_dist_mat.shape):
        if i != j:
            dm_mu[session, i, j] = central(trl_dist_mat[session, i, j], axis=0)
    return dm_mu, xs


def swap_lm_tc(
    y,
    upper_col,
    lower_col,
    ps,
    cues=None,
    p_swap_ind=1,
    p_corr_ind=0,
    spline_order=1,
    n_knots=5,
    swap_decider=swap_argmax,
    corr_decider=corr_argmax,
    col_thr=np.pi / 4,
    cv=skms.LeaveOneOut,
    col_diff=_col_diff_rad,
    model=sklm.Ridge,
    single_color=False,
    norm=True,
    pre_pca=None,
):
    """
    Parameters
    ----------
    y : array N x K x T
        Array where N is the number of trials, K is the number of neurons and T is the
        number of time points
    upper_col, lower_col : array, N
        Array giving upper color
    ps : array, N x 3
    cues : array, N
    """
    if cues is not None:
        targ_col = np.zeros_like(upper_col)
        dist_col = np.zeros_like(upper_col)
        targ_col[cues == 1] = upper_col[cues == 1]
        targ_col[cues == 0] = lower_col[cues == 0]
        dist_col[cues == 1] = lower_col[cues == 1]
        dist_col[cues == 0] = upper_col[cues == 0]
        upper_col = targ_col
        lower_col = dist_col

    n_trls, n_neurs, n_ts = y.shape
    null_colors = (upper_col, lower_col)
    if single_color:
        null_colors = null_colors[:1]
    null_coeffs, spliner = make_lm_coefficients(
        *null_colors,
        cues=cues,
        spline_knots=n_knots,
        spline_degree=spline_order,
        return_spliner=True,
    )

    swap_colors = null_colors[::-1]
    if single_color:
        swap_colors = swap_colors[:1]
    color_swap_coeffs = make_lm_coefficients(
        *swap_colors,
        cues=cues,
        spline_knots=n_knots,
        spline_degree=spline_order,
        use_spliner=spliner,
    )
    alternates = (
        null_coeffs,
        color_swap_coeffs,
    )
    if cues is not None:
        cue_swap_coeffs = make_lm_coefficients(
            *swap_colors,
            cues=1 - cues,
            spline_knots=n_knots,
            spline_degree=spline_order,
            use_spliner=spliner,
        )
        cue_swap_null_coeffs = make_lm_coefficients(
            *null_colors,
            cues=1 - cues,
            spline_knots=n_knots,
            spline_degree=spline_order,
            use_spliner=spliner,
        )
        alternates = alternates + (cue_swap_coeffs, cue_swap_null_coeffs)

    col_dist_mask = col_diff(lower_col, upper_col) > col_thr
    corr_inds, swap_inds = _get_corr_swap_inds(
        ps,
        corr_decider,
        swap_decider,
        and_mask=col_dist_mask,
    )

    n_alts = len(alternates)
    null_dists = np.zeros((len(corr_inds), n_alts, n_alts, 1, n_ts))
    swap_dists = np.zeros((len(corr_inds), n_alts, n_alts, len(swap_inds), n_ts))

    y_swap_te = y[swap_inds]
    swap_pairs = (list(a[swap_inds] for a in alternates), y_swap_te)

    pipe = na.make_model_pipeline(norm=norm, pca=pre_pca)

    if len(swap_inds) == 0:
        null_dists[:] = np.nan
        swap_dists[:] = np.nan
    else:
        cv_gen = cv()
        for i, (train_inds, test_inds) in enumerate(cv_gen.split(corr_inds)):
            corr_tr, corr_te = corr_inds[train_inds], corr_inds[test_inds]
            tr_coeffs = null_coeffs[corr_tr]

            y_corr_tr = y[corr_tr]
            y_corr_te = y[corr_te]

            train_pair = (tr_coeffs, y_corr_tr)
            null_pairs = (list(a[corr_te] for a in alternates), y_corr_te)

            out = _fit_lm_tc_model(
                train_pair,
                null_pairs,
                swap_pairs,
                model=model,
                pre_pipe=pipe,
            )
            null_dists[i] = out[0]
            swap_dists[i] = out[1]
    return null_dists, swap_dists


def compute_dists(
    pop_test,
    trl_targs,
    trl_dists,
    targ_pos,
    dist_pos,
    col_means,
    norm_neurons=True,
    mean=None,
    std=None,
):
    if mean is not None and pop_test.shape[2] > 0:
        pop_test = (pop_test - mean) / std
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
            out_dists[i, j, 0] = _get_trl_dist(
                pop_test[:, 0, j], col_means[i][tc][t_pos], norm_neurons=norm_neurons
            )
            out_dists[i, j, 1] = _get_trl_dist(
                pop_test[:, 0, j], col_means[i][dc][t_pos], norm_neurons=norm_neurons
            )
            out_dists[i, j, 2] = _get_trl_dist(
                pop_test[:, 0, j], col_means[i][tc][d_pos], norm_neurons=norm_neurons
            )
            out_dists[i, j, 3] = _get_trl_dist(
                pop_test[:, 0, j], col_means[i][dc][d_pos], norm_neurons=norm_neurons
            )
            vec_dists[i, j, 0] = _get_vec_dist(
                pop_test[:, 0, j], col_means[i][tc][t_pos], col_means[i][dc][t_pos]
            )
            vec_dists[i, j, 1] = _get_vec_dist(
                pop_test[:, 0, j], col_means[i][tc][t_pos], col_means[i][tc][d_pos]
            )
            vec_dists[i, j, 2] = _get_vec_dist(
                pop_test[:, 0, j], col_means[i][tc][t_pos], col_means[i][dc][d_pos]
            )

    return out_dists, vec_dists


def _resample_equal_conds(resps, conds, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    u_conds, counts = np.unique(conds, return_counts=True)
    sample_count = np.min(counts)

    resps_sampled, conds_sampled = [], []
    for uc in u_conds:
        inds_all = np.where(conds == uc)[0]
        inds_use = rng.choice(inds_all, sample_count, replace=False)
        resps_sampled.append(resps[inds_use])
        conds_sampled.append(conds[inds_use])
    resps_use = np.concatenate(resps_sampled, axis=0)
    conds_use = np.concatenate(conds_sampled, axis=0)
    return resps_use, conds_use


def rdm_similarity(resps, cols, n_bins=8):
    bins = np.linspace(0, np.pi * 2, n_bins + 1)
    bin_cents = bins[:-1] + np.diff(bins)[0] / 2
    conds = np.digitize(cols, bins)
    resps, conds = _resample_equal_conds(resps, conds)
    data = rsa.data.Dataset(resps, obs_descriptors={"stimulus": conds})
    rdm = rsa.rdm.calc_rdm(data, descriptor="stimulus", noise=None, method="crossnobis")
    mat = rdm.get_matrices()[0]
    return bin_cents, mat


def negative_euclidean_distances(*args, **kwargs):
    return -skmp.euclidean_distances(*args, **kwargs)


def compute_similarity_curve(
    resps,
    cols,
    similarity_metric=negative_euclidean_distances,
):
    dists = u.normalize_periodic_range(
        np.expand_dims(cols, 1) - np.expand_dims(cols, 0)
    )
    corrs = similarity_metric(resps, resps)
    mask = np.identity(corrs.shape[0], dtype=bool)
    corrs[mask] = np.nan
    dists[mask] = np.nan
    return dists, corrs


def norm_distr_func(x, sig, norm=False):
    c_hat = np.exp(-(x**2) / (2 * sig**2))
    if norm:
        c_hat_zeroed = c_hat - np.nanmin(c_hat)
        c_hat = c_hat_zeroed / np.nanmax(c_hat_zeroed)
    return c_hat


def fit_similarity_dispersion(dists, corrs, return_func=True, **kwargs):
    d_cents, c_cents = gpl.digitize_vars(dists, corrs, ret_all_y=False, **kwargs)
    c_zeroed = c_cents - np.nanmin(c_cents)
    norm_cs = c_zeroed / np.nanmax(c_zeroed)

    def _min_func(sig):
        c_hat = norm_distr_func(d_cents, sig, norm=True)

        loss = np.nansum((norm_cs - c_hat) ** 2)
        return loss

    res = sopt.minimize(_min_func, 0.1, bounds=((0, None),))
    out = res.x
    if return_func:
        out = (res.x, (norm_cs, d_cents, norm_distr_func(d_cents, res.x, norm=True)))
    return out


def simulate_emp_err(sig, dprime, n_samps=10000, n_pts=100, norm=True):
    func_dists = np.linspace(-np.pi, np.pi, n_pts)
    sim_func = norm_distr_func(func_dists, sig, norm=norm)

    noise = sts.norm(0, 1).rvs((n_samps, n_pts))

    inds = np.argmax(np.expand_dims(sim_func * dprime, 0) + noise, axis=1)
    errs = func_dists[inds]
    return errs


def simulate_emp_errs(
    sig, min_dp=0.2, max_dp=2, n_dp=10, n_pts=100, n_samps=10000, norm=True
):
    func_dists = np.linspace(-np.pi, np.pi, n_pts)
    sim_func = norm_distr_func(func_dists, sig, norm=norm)

    frozen_noise = sts.norm(0, 1).rvs((n_samps, n_pts))

    def _min_func(dprime):
        inds = np.argmax(np.expand_dims(sim_func * dprime, 0) + frozen_noise, axis=1)
        errs = func_dists[inds]
        return errs

    dprimes = np.linspace(min_dp, max_dp, n_dp)
    errs_ds = np.stack(list(_min_func(dp) for dp in dprimes), axis=0)
    # sopt.minimize(_min_func, .1, bounds=((0, None),))
    return errs_ds


def dprime_sig_sweep(
    emp_errs,
    min_dp=0.2,
    max_dp=3,
    n_dp=25,
    min_sigma=0.1,
    max_sigma=5,
    n_sigma=20,
    n_bins=10,
):
    sigmas = np.linspace(min_sigma, max_sigma, n_sigma)
    dps = np.linspace(min_dp, max_dp, n_dp)
    hist_bins = np.linspace(-np.pi, np.pi, n_bins)
    err_hist, _ = np.histogram(emp_errs, bins=hist_bins, density=True)
    kls = np.zeros((n_sigma, n_dp))
    for i, sig in enumerate(sigmas):
        samps = simulate_emp_errs(sig, min_dp=min_dp, max_dp=max_dp, n_dp=n_dp)
        for j, samp in enumerate(samps):
            samp_hist, _ = np.histogram(samp, bins=hist_bins, density=True)
            kls[i, j] = np.sum(spsp.kl_div(err_hist, samp_hist))
    return sigmas, dps, kls


def fit_tcc_dprime(resps, targs, cols, n_pts=100, n_samps=10000, **kwargs):
    emp_errs = u.normalize_periodic_range(targs - cols)
    dists, corrs = compute_similarity_curve(resps, targs)
    sig = fit_similarity_dispersion(dists, corrs)
    print(sig)
    sig += 1

    func_dists = np.linspace(-np.pi, np.pi, n_pts)
    sim_func = norm_distr_func(func_dists, sig)

    frozen_noise = sts.norm(0, 1).rvs((n_samps, n_pts))

    def _min_func(dprime):
        inds = np.argmax(np.expand_dims(sim_func * dprime, 0) + frozen_noise, axis=1)
        errs = func_dists[inds]
        return errs

    dprimes = np.linspace(0.1, 2, 10)
    errs_ds = np.stack(list(_min_func(dp) for dp in dprimes), axis=0)
    return emp_errs, errs_ds


def get_color_means(
    data,
    tbeg,
    tend,
    twindow,
    tstep,
    color_window=0.2,
    time_key="SAMPLES_ON_diode",
    targ_key="LABthetaTarget",
    dist_key="LABthetaDist",
    upper_key="IsUpperSample",
    regions=None,
    leave_out=0,
    norm=True,
    norm_neurons=True,
    test_data=None,
    pops_xs=None,
    use_cache=False,
):
    targ_cols = data[targ_key]
    dist_cols = data[dist_key]
    upper_samps = data[upper_key]
    u_cols = np.unique(np.concatenate(targ_cols))
    if pops_xs is not None:
        pops, xs = pops_xs
    else:
        pops, xs = data.get_populations(
            twindow,
            tbeg,
            tend,
            tstep,
            time_zero_field=time_key,
            regions=regions,
            skl_axes=True,
            cache=use_cache,
        )
    targ_dists_all = []
    vec_dists_all = []
    dist_dists_all = []
    targ_means_all = []
    dist_means_all = []
    means = []
    stds = []
    for i, pop_i in enumerate(pops):
        out = _get_leftout_color_dists(
            pop_i,
            targ_cols[i],
            dist_cols[i],
            upper_samps[i],
            norm=norm,
            u_cols=u_cols,
            color_window=color_window,
            return_norm=True,
        )
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
        out = get_test_color_dists(
            test_data,
            tbeg,
            tend,
            twindow,
            tstep,
            targ_means_all,
            dist_means_all,
            time_key=time_key,
            dist_key=dist_key,
            regions=regions,
            norm=norm,
            norm_neurons=norm_neurons,
            m=means,
            s=stds,
            use_cache=use_cache,
        )
        test_dists, test_vecs = out
        out_dists = out_dists + (test_dists, test_vecs)
    out_means = (targ_means_all, dist_means_all)
    out = out_dists, out_means, xs
    return out


def fit_tccish_model(col_cents, pops, err, max_dp=5, shuffle=False):
    if shuffle:
        rng = np.random.default_rng()
        col_cents = rng.permuted(col_cents)
    sigs_emp, funcs = estimate_distance_decay(col_cents, pops)

    sigs_sweep, dps_sweep, kls = dprime_sig_sweep(err)
    inds = np.argmin((sigs_emp[None] - sigs_sweep[:, None]) ** 2, axis=0)
    dps_emp = dps_sweep[np.argmin(kls[inds], axis=1)]

    out_dict = {}
    out_dict["sweep"] = (sigs_sweep, dps_sweep, kls)
    out_dict["emp"] = (sigs_emp, dps_emp)
    out_dict["err"] = err
    out_dict["funcs"] = funcs
    return out_dict


def save_color_pseudopops_regions(
    *args,
    save_file="swap_errors/color_pseudo_data/m_{region_key}_pseudos.pkl",
    region_groups=all_region_subset,
    **kwargs,
):
    out_data = {}
    for region_key, region_list in region_groups.items():
        path = save_file.format(region_key=region_key)
        try:
            out_rk = make_all_color_pseudopops(*args, **kwargs, regions=region_list)
            out_data[region_key] = out_rk
            pickle.dump(out_rk, open(path, "wb"))
        except ValueError as e:
            print("error creating population, {rk}".format(rk=region_key))
            print(e)
    return out_data


def load_color_pseudopops_regions(
    region_key,
    folder="swap_errors/color_pseudo_data/",
    template="m_{region_key}_pseudos.pkl",
):
    path = os.path.join(folder, template.format(region_key=region_key))
    return pd.read_pickle(path)


def make_all_color_pseudopops(
    data,
    tbeg,
    tend,
    monkeys=("Elmo", "Waldorf"),
    trls=(
        "retro",
        "pro",
    ),
    filter_swap_prob=0.3,
    resamples=200,
    min_trials=10,
    cue_filter=None,
    **kwargs,
):
    out_dict = {}
    for m in monkeys:
        out_dict[m] = {}
        data_m = data.session_mask(data["animal"] == m)
        if cue_filter is not None:
            data_m = data_m.mask(data_m["IsUpperSample"] == cue_filter)
        for trl in trls:
            out = make_color_pseudopops(
                data_m,
                tbeg,
                tend,
                trl_type=trl,
                min_trials=min_trials,
                filter_swap_prob=filter_swap_prob,
                resamples=resamples,
                **kwargs,
            )
            out_dict[m][trl] = out
    return out_dict


def make_color_pseudopops(
    data,
    tbeg,
    tend,
    time_key="WHEEL_ON_diode",
    color_key="LABthetaTarget",
    error_key="err",
    trl_type="retro",
    swap_prob_key="swap_prob",
    n_col_bins=11,
    trl_ax=3,
    filter_swap_prob=None,
    **kwargs,
):
    if trl_type == "retro":
        data = retro_mask(data)
    elif trl_type == "pro":
        data = pro_mask(data)
    elif trl_type == "single":
        data = single_mask(data)
    else:
        raise IOError("trl_type {} is not recognized".format(trl_type))
    if filter_swap_prob is not None:
        data = data.mask(data[swap_prob_key] < filter_swap_prob)
    col_bins = np.linspace(0, np.pi * 2, n_col_bins + 1)
    bin_cents = col_bins[:-1] + np.diff(col_bins)[0] / 2
    errs = np.concatenate(data[error_key])
    masks = []
    for i, cb_beg in enumerate(col_bins[:-1]):
        cb_end = col_bins[i + 1]
        mask_low_i = data[color_key] >= cb_beg
        if i + 1 == len(col_bins):
            mask_i = mask_low_i
        else:
            mask_high_i = data[color_key] < cb_end
            mask_i = mask_low_i.rs_and(mask_high_i)

        masks.append(mask_i)

    xs, pops_pseudo = data.make_pseudo_pops(
        tend - tbeg,
        tbeg,
        tend,
        tend - tbeg,
        *masks,
        tzfs=(time_key,) * n_col_bins,
        shuffle_trials=True,
        **kwargs,
    )
    t_cent = (tbeg + tend) / 2
    t_ind = np.argmin((xs - t_cent) ** 2)
    cols_all = []
    for i, pp in enumerate(pops_pseudo):
        cols_all.append(bin_cents[i] * np.ones(pp.shape[trl_ax]))
    cols_all = np.concatenate(cols_all)

    resps_t = list(pp[..., t_ind] for pp in pops_pseudo)
    resps_all = np.concatenate(resps_t, axis=trl_ax)
    resps_all = np.swapaxes(np.squeeze(resps_all), 1, 2)

    return cols_all, resps_all, errs


def _preprocess_dist_pop(pop, norm=True, pre_pca=0.99, **kwargs):
    pipe = na.make_model_pipeline(norm=norm, pca=pre_pca, **kwargs)
    pop_pre = pipe.fit_transform(pop)
    return pop_pre


def estimate_distance_decay(bin_cents, pops, pre_pca=0.99, **kwargs):
    sigs = np.zeros(len(pops))
    cents_u = np.unique(bin_cents)
    n_bins = len(cents_u)
    funcs = np.zeros((len(pops), n_bins))
    fits = np.zeros_like(funcs)
    for i, pop in enumerate(pops):
        pop_i_pre = _preprocess_dist_pop(pops[i], norm=True, pre_pca=pre_pca)
        dists, corrs = compute_similarity_curve(pop_i_pre, bin_cents, **kwargs)
        sigs[i], func = fit_similarity_dispersion(
            dists, corrs, n_bins=n_bins, return_func=True
        )
        funcs[i] = func[0]
        cents = func[1]
        fits[i] = func[2]
    return sigs, (funcs, cents, fits)


def decode_color(
    data,
    tbeg,
    tend,
    twindow,
    tstep,
    time_key="SAMPLES_ON_diode",
    color_key="LABthetaTarget",
    regions=None,
    n_folds=10,
    transform_color=True,
    model=skc.SVR,
    n_jobs=-1,
    use_stan=True,
    max_pops=np.inf,
    time=True,
    pseudo=False,
    min_trials_pseudo=1,
    resample_pseudo=10,
    **kwargs,
):
    regs = data[color_key]
    if pseudo:
        regs = np.concatenate(regs)
        u_cols = np.unique(regs)
        pis = []
        ns = []
        for i, uc in enumerate(u_cols):
            mi = data[color_key] == uc
            di = data.mask(mi)
            pi, xs = di.get_populations(
                twindow, tbeg, tend, tstep, time_zero_field=time_key, skl_axes=True
            )

            ns.append(di.get_ntrls())
            pis.append(pi)
        comb_n = dio.combine_ntrls(*ns)
        new_pis = []
        cols = []
        for i, pi in enumerate(pis):
            pi = data.make_pseudopop(
                pi, comb_n, min_trials_pseudo, resample_pseudo, skl_axs=True
            )
            new_pis.append(pi)
            cols.append((u_cols[i],) * pi.shape[3])
        pops = np.concatenate(new_pis, axis=3)
        cols_cat = np.concatenate(cols)
        regs = (cols_cat,) * resample_pseudo
    else:
        pops, xs = data.get_populations(
            twindow, tbeg, tend, tstep, time_zero_field=time_key, skl_axes=True
        )
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
            out = na.pop_regression_skl(
                pop, regs_i, n_folds, mean=False, model=model, n_jobs=n_jobs, **kwargs
            )
        outs.append(out)
    return outs, xs


def quantify_swap_loo(
    mdict,
    data,
    method="weighted",
    threshold=0.5,
    data_key="p",
    data_ind=1,
    comb_func=np.sum,
):
    new_m = {}
    for k, model in mdict.items():
        m_copy = model.copy()
        data_col = np.expand_dims(data[data_key][:, data_ind], (0, 1))
        if method == "weighted":
            adj_loo = m_copy.log_likelihood * data_col
        elif method == "log_weighted":
            adj_loo = m_copy.log_likelihood + np.log(data_col)
        elif method == "threshold":
            mask = data_col > threshold
            adj_loo = m_copy.log_likelihood * mask
        elif method == "none":
            adj_loo = m_copy.log_likelihood
        else:
            raise IOError("unrecognized method")
        m_copy.log_ood = adj_loo
        new_m[k] = m_copy
    comp = az.compare(new_m)
    return comp


def despline_color(cols, eps=0.001, ret_err=False):
    u_cols = np.unique(cols, axis=0)

    float_cols = np.linspace(0, 2 * np.pi, len(u_cols) + 1)[:-1]
    o_cols = spline_color(float_cols, cols.shape[1]).T
    out = np.zeros(len(cols))
    err = np.zeros_like(out)
    for i, oc in enumerate(o_cols):
        mask = np.sum((np.expand_dims(oc, 0) - cols) ** 2, axis=1)
        out[mask < eps] = float_cols[i]
        err[mask < eps] = mask[mask < eps]
    if ret_err:
        out = (out, err)
    return out


def spline_color(cols, num_bins, degree=1, use_skl=True):
    """
    cols should be given between 0 and 2 pi, bins also
    """
    if use_skl:
        st = skp.SplineTransformer(
            num_bins + 2, degree=degree, include_bias=False, extrapolation="periodic"
        )
        cols_spl = st.fit_transform(np.expand_dims(cols, 1))
        alpha = (cols_spl - np.mean(cols_spl, axis=0, keepdims=True)).T
    else:
        bins = np.linspace(0, 2 * np.pi, num_bins + 1)[:num_bins]

        dc = 2 * np.pi / (len(bins))

        # get the nearest bin
        diffs = np.exp(1j * bins)[:, None] / np.exp(1j * cols)[None, :]
        distances = np.arctan2(diffs.imag, diffs.real)
        dist_near = np.abs(distances).min(0)
        nearest = np.abs(distances).argmin(0)
        # see if the color is to the "left" or "right" of that bin
        sec_near = np.sign(distances[nearest, np.arange(len(cols))] + 1e-8).astype(int)
        # add epsilon to handle 0
        # fill in the convex array
        alpha = np.zeros((len(bins), len(cols)))
        alpha[nearest, np.arange(len(cols))] = (dc - dist_near) / dc
        alpha[np.mod(nearest - sec_near, len(bins)), np.arange(len(cols))] = (
            1 - (dc - dist_near) / dc
        )

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


def single_neuron_color(
    data,
    tbeg,
    tend,
    twindow,
    tstep,
    time_key="SAMPLES_ON_diode",
    color_key="LABthetaTarget",
    neur_chan="neur_channels",
    neur_id="neur_ids",
    neur_region="neur_regions",
):
    pops, xs = data.get_populations(
        twindow, tbeg, tend, tstep, time_zero_field=time_key
    )
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


_sd_keys = ("T", "K", "N", "y", "cue", "C_u", "C_l", "C_resp", "p", "type", "is_joint")


def generate_fake_data_from_model(
    model,
    stan_data,
    cu="C_u",
    cl="C_l",
    use_t=True,
    make_new_dict=True,
    keep_keys=_sd_keys,
    **kwargs,
):
    use_type = len(np.unique(stan_data["type"])) > 1
    y_new = np.zeros_like(stan_data["y"])
    mp = stan_data["model_path"]
    for i, cu_i in enumerate(stan_data[cu]):
        cl_i = np.expand_dims(stan_data[cl][i], 0)
        cu_i = np.expand_dims(cu_i, 0)
        if use_type:
            t_i = stan_data["type"][i] - 1
            mu_u = model.posterior["mu_u_type"][:, :, t_i]
            mu_l = model.posterior["mu_l_type"][:, :, t_i]
        else:
            mu_u = model.posterior["mu_u"]
            mu_l = model.posterior["mu_l"]
        mu_u = np.mean(mu_u, axis=(0, 1))
        mu_l = np.mean(mu_l, axis=(0, 1))
        r_mean = np.array(np.sum(mu_u * cu_i, axis=1)) + np.array(
            np.sum(mu_l * cl_i, axis=1)
        )

        std = np.sqrt(np.mean(model.posterior["vars"], axis=(0, 1)))
        y_new[i] = sts.norm(r_mean, std).rvs()
    if make_new_dict:
        new_dict = {k: v for k, v in stan_data.items() if k in keep_keys}
        new_dict["y"] = y_new
        new_dict.update(kwargs)
    else:
        new_dict = y_new
    return new_dict, mp


def make_lm_coefficients(
    *cols,
    cues=None,
    spline_knots=4,
    spline_degree=2,
    standardize_splines=False,
    use_spliner=None,
    return_spliner=False,
):
    all_coeffs = []
    if len(cols) > 0:
        all_cols = np.expand_dims(np.concatenate(cols), 1)
        if use_spliner is not None and cues is None:
            spliner = use_spliner
        elif use_spliner is not None and cues is not None:
            spliner, cue_trs = use_spliner
        else:
            pipe = []
            pipe.append(
                skp.SplineTransformer(
                    spline_knots,
                    degree=spline_degree,
                    include_bias=True,
                    extrapolation="periodic",
                )
            )
            if standardize_splines:
                pipe.append(skp.StandardScaler())

            spliner = sklpipe.make_pipeline(*pipe)
            spliner.fit(all_cols)
        for col in cols:
            col_spl = spliner.transform(np.expand_dims(col, 1))
            all_coeffs.append(col_spl)

    elif cues is None:
        raise IOError("need to provide something! neither colors nor cues present")
    else:
        all_cols = np.zeros((len(cues), 0))
        all_coeffs.append(all_cols)
    if cues is not None and use_spliner is None:
        cue_trs = skp.StandardScaler()
        cues = cue_trs.fit_transform(np.expand_dims(cues, 1))
        all_coeffs.append(cues)
    elif cues is not None and use_spliner is not None:
        cues = cue_trs.transform(np.expand_dims(cues, 1))
        all_coeffs.append(cues)
    coeffs = np.concatenate(all_coeffs, axis=1)
    if return_spliner and cues is None:
        out = (coeffs, spliner)
    elif return_spliner and cues is not None:
        out = (coeffs, (spliner, cue_trs))
    else:
        out = coeffs
    return out


def retro_mask(data):
    bhv_retro = (data["is_one_sample_displayed"] == 0).rs_and(data["Block"] > 1)
    bhv_retro = bhv_retro.rs_and(data["StopCondition"] > -2)
    data_retro = data.mask(bhv_retro)
    return data_retro


def single_mask(data):
    bhv_single = data["is_one_sample_displayed"] == 1
    bhv_single = bhv_single.rs_and(data["StopCondition"] > -2)
    data_single = data.mask(bhv_single)
    return data_single


def pro_mask(data):
    bhv_pro = (data["is_one_sample_displayed"] == 0).rs_and(data["Block"] == 1)
    bhv_pro = bhv_pro.rs_and(data["StopCondition"] > -2)
    data_pro = data.mask(bhv_pro)
    return data_pro


def fit_animal_bhv_models(
    data, *args, animal_key="animal", retro_masking=True, pro_masking=False, **kwargs
):
    complete = data["StopCondition"] > -2
    if retro_masking:
        bhv_mask = complete.rs_and(data["is_one_sample_displayed"] == 0).rs_and(
            data["Block"] > 1
        )
    if pro_masking:
        bhv_mask = complete.rs_and(data["is_one_sample_displayed"] == 0).rs_and(
            data["Block"] == 1
        )
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


mixture_arviz = {
    "observed_data": "err",
    "log_likelihood": {"err": "log_lik"},
    "posterior_predictive": "err_hat",
    "dims": {
        "report_var": ["run_ind"],
        "swap_prob": ["run_ind"],
        "guess_prob": ["run_ind"],
    },
}


# def gpfa(data, tbeg=-0.5, tend=1, winsize=0.05, n_factors=8, tzf="CUE2_ON_diode"):
#     out = data.get_spiketrains(tbeg, tend, time_zero_field=tzf)
#     pops = out
#     fits = []
#     for i, pop in enumerate(pops):
#         pop_format = list(list(pop_j) for pop_j in pop)
#         gp = el.gpfa.GPFA(bin_size=winsize * pq.s, x_dim=n_factors)
#         gp.fit(pop_format)
#         fits.append(gp)
#     return fits, pops


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


def compute_diff_dependence(
    data,
    targ_field="LABthetaTarget",
    dist_field="LABthetaDist",
    resp_field="LABthetaResp",
):
    targ = np.concatenate(data[targ_field])
    dist = np.concatenate(data[dist_field])
    resp = np.concatenate(data[resp_field])
    td_diff = u.normalize_periodic_range(targ - dist)
    resp_diff = u.normalize_periodic_range(targ - resp)
    dist_diff = u.normalize_periodic_range(dist - resp)
    return td_diff, resp_diff, dist_diff


bmp = "swap_errors/behavioral_model/corr_swap_guess.pkl"
bmp_ub = "swap_errors/behavioral_model/csg_dirich.pkl"
default_prior_dict = {
    "report_var_var_mean": 1,
    "report_var_var_var": 3,
    "report_var_mean_mean": 0.64,
    "report_var_mean_var": 1,
    "swap_weight_var_mean": 1,
    "swap_weight_var_var": 3,
    "swap_weight_mean_mean": 0,
    "swap_weight_mean_var": 1,
    "guess_weight_var_mean": 1,
    "guess_weight_var_var": 3,
    "guess_weight_mean_mean": 0,
    "guess_weight_mean_var": 1,
}
ub_prior_dict = {
    "report_var_var_mean": 1,
    "report_var_var_var": 3,
    "report_var_mean_mean": 0.64,
    "report_var_mean_var": 1,
    "swap_weight_mean_mean": 0,
    "swap_weight_mean_var": 1,
}


def fit_bhv_model(
    data,
    model_path=bmp,
    targ_field="LABthetaTarget",
    dist_field="LABthetaDist",
    resp_field="LABthetaResp",
    prior_dict=None,
    stan_iters=2000,
    stan_chains=4,
    arviz=mixture_arviz,
    adapt_delta=0.9,
    diagnostics=True,
    **stan_params,
):
    if prior_dict is None:
        prior_dict = default_prior_dict
    targs_is = data[targ_field]
    session_list = np.array(data[["animal", "date"]])
    mapping_list = []
    session_nums = np.array([], dtype=int)
    for i, x in enumerate(targs_is):
        sess = np.ones(len(x), dtype=int) * (i + 1)
        session_nums = np.concatenate((session_nums, sess))
        indices = x.index
        sess_info0 = (str(session_list[i, 0]),) * len(x)
        sess_info1 = (str(session_list[i, 1]),) * len(x)
        mapping_list = mapping_list + list(zip(indices, sess_info0, sess_info1))
    mapping_dict = {i: mapping_list[i] for i in range(len(session_nums))}
    targs = np.concatenate(targs_is, axis=0)
    dists = np.concatenate(data[dist_field], axis=0)
    resps = np.concatenate(data[resp_field], axis=0)
    errs = u.normalize_periodic_range(targs - resps)
    dist_errs = u.normalize_periodic_range(dists - resps)
    dists_per = u.normalize_periodic_range(dists - targs)
    stan_data = dict(
        T=dist_errs.shape[0],
        S=len(targs_is),
        err=errs,
        dist_err=dist_errs,
        run_ind=session_nums,
        dist_loc=dists_per,
        **prior_dict,
    )
    control = {
        "adapt_delta": stan_params.pop("adapt_delta", 0.8),
        "max_treedepth": stan_params.pop("max_treedepth", 10),
    }
    sm = pickle.load(open(model_path, "rb"))
    fit = sm.sampling(
        data=stan_data,
        iter=stan_iters,
        chains=stan_chains,
        control=control,
        **stan_params,
    )
    if diagnostics:
        diag = az.diagnostics.check_hmc_diagnostics(fit)
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
