import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn.decomposition as skd
import sklearn.manifold as skm
import sklearn.linear_model as sklm
import sklearn.svm as sk_svm
import scipy.stats as sts
import arviz as az
import itertools as it
import seaborn as sns
import functools as ft

import general.plotting as gpl
import general.utility as u
import general.rf_models as rfm
import general.neural_analysis as na
import swap_errors.analysis as swan
import swap_errors.auxiliary as swaux


def plot_sigma_vs_emp(
    tcc_dict,
    fwid=3,
    ind=None,
    ax=None,
    color=None,
    indiv_color=(.6,)*3,
    plot_indiv_fits=False,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    emp_results = tcc_dict["emp"]
    sigs, dps = emp_results
    funcs = tcc_dict["funcs"]
    if ind is None:
        sig_m = np.median(sigs)
        ind = np.argmin((sigs - sig_m)**2)
    ax.plot(funcs[1], funcs[0].T, alpha=.2, color=indiv_color)
    if plot_indiv_fits:
        _ = ax.plot(
            funcs[1], funcs[2].T, linestyle='dashed', alpha=.1, color=indiv_color
        )
    ax.plot(funcs[1], np.mean(funcs[0], axis=0), color=color)
    gpl.plot_trace_werr(
        funcs[1],
        np.mean(funcs[2], axis=0),
        linestyle='dashed',
        color=color,
        plot_outline=True,
        ax=ax,
    )
    return ax


def visualize_rf_properties(w_distrib, n_units=2000, total_pwr=10, axs=None, skip=.2):
    if axs is None:
        fwid = 2
        f, axs = plt.subplots(1, 3, figsize=(fwid*3, fwid))

    out = rfm.get_random_uniform_w_distrib_pop(10, 100, 1, w_distrib)
    stim_distr, rf, drf, noise_distr = out

    print(np.mean(np.sum(rf(stim_distr.rvs(1000))**2, axis=1)))

    xs = np.expand_dims(np.linspace(skip, 1 - skip, 50), 1)
    reps = rf(xs)

    axs[0].hist(w_distrib.rvs(1000))

    dists, corrs = swan.compute_similarity_curve(reps, xs)

    d_flat = np.squeeze(dists).flatten()
    c_flat = np.squeeze(corrs).flatten()
    gpl.plot_scatter_average(d_flat, c_flat, ax=axs[2])

    gpl.plot_highdim_trace(reps, dim_red_mean=False, n_dim=2, ax=axs[1])
    gpl.make_yaxis_scale_bar(axs[1], label="PC 2", text_buff=.3)
    gpl.make_xaxis_scale_bar(axs[1], label="PC 1")
    gpl.clean_plot(axs[1], 0)
    gpl.clean_plot(axs[0], 0)
    axs[1].set_aspect("equal")
    return axs


def plot_tcc_results(tcc_dict, fwid=2, ind=None, faxs=None, color=None, plot_title=""):
    if faxs is None:
        faxs = plt.subplots(1, 3, figsize=(fwid*3.5, fwid))
    f, axs = faxs
    sweep_results = tcc_dict["sweep"]
    emp_results = tcc_dict["emp"]
    errs = tcc_dict["err"]

    plot_sweep(
        *sweep_results, theor_params=emp_results, ax=axs[0], fit_color=color, f=f
    )

    plot_kl_comp_distrib(*sweep_results, errs, ax=axs[1], hist_color=color)

    emp_results = tcc_dict["emp"]
    sigs, dps = emp_results
    if ind is None:
        sig_m = np.mean(sigs)
        ind = np.argmin((sigs - sig_m)**2)
    plot_comparison_distrib(sigs[ind], dps[ind], errs, hist_color=color, ax=axs[2])
    gpl.clean_plot(axs[1], 0)
    gpl.clean_plot(axs[2], 1)
    gpl.make_yaxis_scale_bar(axs[1], .1, label="density", double=False)
    axs[1].set_xlabel("error (radians)")
    axs[2].set_xlabel("error (radians)")
    axs[0].set_title(plot_title)


def compare_model_confs(param_key, *models, ax=None, fwid_h=10, fwid_v=1):
    if ax is None:
        f, ax = plt.subplots(figsize=(fwid_h, fwid_v))
    colors = {}
    wids = {}
    for i, model in enumerate(models):
        samples = np.concatenate(model.posterior[param_key], axis=0)
        samples = np.moveaxis(samples, 0, -1)
        wids[i] = []
        for j, ind in enumerate(u.make_array_ind_iterator(samples.shape[:-1])):
            sl = samples[ind]
            color = colors.get(i)
            l = gpl.plot_trace_werr(
                [j + i / (2 * len(models))],
                np.expand_dims(sl, 1),
                conf95=True,
                fill=False,
                color=color,
                ax=ax,
            )
            delt = gpl.conf95_interval(sl)
            wids[i].append(delt[0, 0] - delt[1, 0])
            colors[i] = l[0].get_color()
    return wids


def plot_trial_lls(
    m1, m2, axs=None, use_types=None, model_key="hybrid", l_key="y", fwid=3
):
    if axs is None:
        f, axs = plt.subplots(1, 3, figsize=(fwid * 3, fwid))
    (ax_scatt, ax_hist, ax_mean) = axs
    mean_diffs = []
    for k, (m1_k, sd1_k) in m1.items():
        m2_k, sd2_k = m2[k]
        if use_types is not None:
            mask1 = sd1_k["type"] == use_types[0]
            mask2 = sd2_k["type"] == use_types[1]
        else:
            mask1 = np.ones_like(sd1_k["type"], dtype=bool)
            mask2 = np.ones_like(sd2_k["type"], dtype=bool)
        ll1_k = np.concatenate(m1_k[model_key].log_likelihood[l_key].to_numpy())
        ll2_k = np.concatenate(m2_k[model_key].log_likelihood[l_key].to_numpy())
        ll1_k = ll1_k[:, mask1]
        ll2_k = ll2_k[:, mask2]
        ax_scatt.plot(np.mean(ll1_k, axis=0), np.mean(ll2_k, axis=0), "o")
        all_diffs = np.reshape(ll1_k - ll2_k, (-1,))
        ax_hist.hist(all_diffs, histtype="step", density=True)
        mean_diffs.append(np.mean(all_diffs))
    ax_mean.hist(mean_diffs)
    gpl.add_vlines(np.mean(mean_diffs), ax_mean)


def plot_decoding_comparison(
    d1_rates,
    d2_rates,
    axs=None,
    color=None,
    spread=0.05,
    lw=1,
    elinewidth=2,
    corr_thr=0.6,
    swap_thr=0.3,
    buff=0.4,
):
    if axs is None:
        f, axs = plt.subplots(1, 2)
    rng = np.random.default_rng()

    for k, (d1_corr, d1_swap) in d1_rates.items():
        d2_corr, d2_swap = d2_rates[k]
        corr_line = np.stack((d1_corr, d2_corr), axis=1)
        swap_line = np.stack((d1_swap, d2_swap), axis=1)
        off = rng.normal(0, spread)

        gpl.plot_trace_werr(
            [off, 1 + off],
            corr_line,
            fill=False,
            errorbar=True,
            ax=axs[0],
            color=color,
            error_func=gpl.std,
            lw=lw,
            elinewidth=elinewidth,
        )
        gpl.plot_trace_werr(
            [off, 1 + off],
            swap_line,
            fill=False,
            errorbar=True,
            ax=axs[1],
            color=color,
            lw=lw,
            error_func=gpl.std,
            elinewidth=elinewidth,
        )
    axs[0].set_ylabel("decoding performance\np(corr) > {}".format(corr_thr))
    axs[1].set_ylabel("decoding performance\np(swap) > {}".format(swap_thr))
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(["delay 1", "delay 2"])
    axs[0].set_xlim([-buff, 1 + buff])
    axs[1].set_xlim([-buff, 1 + buff])
    gpl.clean_plot_bottom(axs[0], keeplabels=True)
    gpl.clean_plot_bottom(axs[1], keeplabels=True)


def plot_cue_decoding(rate_dict, color=None, axs=None, n_boots=1000, x_cent=0):
    if axs is None:
        f, axs = plt.subplots(1, 2)
    diffs = []
    for k, err_tuple in rate_dict.items():
        c_err, s_err = err_tuple[:2]
        c_err = np.expand_dims(c_err, 1)
        s_err = np.expand_dims(s_err, 1)
        l = gpl.plot_trace_werr(
            c_err, s_err, ax=axs[0], points=True, error_func=gpl.std, color=color
        )
        diffs.append(np.mean(s_err) - np.mean(c_err))
        color = l[0].get_color()

    mu_diff = u.bootstrap_list(np.array(diffs), np.nanmean, n_boots)
    gpl.violinplot(
        [mu_diff],
        [x_cent],
        color=(color,),
        ax=axs[1],
        showextrema=False,
        showmedians=True,
    )
    return axs


def plot_model_probs(
    *args,
    plot_keys=("swap_prob", "guess_prob"),
    ax=None,
    sep=0.5,
    comb_func=np.median,
    colors=None,
    sub_x=-1,
    labels=("swaps", "guesses"),
    total_label="correct",
    arg_names=("Elmo", "Waldorf"),
    ms=3,
    monkey_colors=None,
):
    if colors is None:
        colors = (None,) * len(args)
    if ax is None:
        f, ax = plt.subplots(1, 1)
    cents = np.arange(0, len(plot_keys))
    n_clusters = len(args)
    swarm_full = {"x": [], "y": [], "monkey": []}
    violin_full = {"x": [], "y": [], "monkey": []}
    for i, m in enumerate(args):
        offset = (i - n_clusters / 2) * sep
        swarm_data = {"x": [], "y": []}
        violin_data = {"x": [], "y": []}
        for j, pk in enumerate(plot_keys):
            pk_sessions = comb_func(m[pk], axis=(0, 1))
            pk_full = m[pk].to_numpy().flatten()
            if j == 0:
                totals = np.zeros_like(pk_sessions)
                totals_full = np.zeros_like(pk_full)
            totals = totals + pk_sessions
            totals_full = totals_full + pk_full
            xs_full = np.ones(len(pk_full)) * (offset + cents[j])
            violin_data["x"] = np.concatenate((violin_data["x"], xs_full))
            violin_data["y"] = np.concatenate((violin_data["y"], pk_full))
            xs = np.ones(len(pk_sessions)) * (offset + cents[j])
            swarm_data["x"] = np.concatenate((swarm_data["x"], xs))
            swarm_data["y"] = np.concatenate((swarm_data["y"], pk_sessions))

        xs_full = np.ones(len(totals_full)) * (offset + sub_x)
        violin_data["x"] = np.concatenate((violin_data["x"], xs_full))
        violin_data["y"] = np.concatenate((violin_data["y"], 1 - totals_full))

        xs = np.ones(len(pk_sessions), dtype=float) * (offset + sub_x)
        swarm_data["x"] = np.concatenate((swarm_data["x"], xs))
        swarm_data["y"] = np.concatenate((swarm_data["y"], 1 - totals))
        swarm_full["x"] = np.concatenate((swarm_full["x"], swarm_data["x"]))
        swarm_full["y"] = np.concatenate((swarm_full["y"], swarm_data["y"]))
        monkey_list = [arg_names[i]] * len(swarm_data["x"])
        swarm_full["monkey"] = np.concatenate((swarm_full["monkey"], monkey_list))

        violin_full["x"] = np.concatenate((violin_full["x"], violin_data["x"]))
        violin_full["y"] = np.concatenate((violin_full["y"], violin_data["y"]))
        monkey_list = [arg_names[i]] * len(violin_data["x"])
        violin_full["monkey"] = np.concatenate((violin_full["monkey"], monkey_list))

    # sns.violinplot(data=violin_full, x='x', y='y', hue='monkey',
    #                palette=monkey_colors, ax=ax)
    l = sns.swarmplot(
        data=swarm_full,
        x="x",
        y="y",
        hue="monkey",
        palette=monkey_colors,
        ax=ax,
        size=ms,
    )
    ax.legend(frameon=False)
    ax.set_ylim([0, 1])
    ax.set_xticks([0.5, 2.5, 4.5])
    ax.set_xticklabels((total_label,) + labels)
    ax.set_ylabel("probability")
    gpl.clean_plot(ax, 0)
    return ax


default_ax_labels = ("guess probability", "swap probability", "correct probability")


def visualize_simplex(
    pts, ax=None, ax_labels=default_ax_labels, thr=0.5, grey_col=(0.7, 0.7, 0.7)
):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection="3d")
    all_pts = np.stack(list(v[1] for v in pts), axis=0)
    ax.plot(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], "o", color=grey_col)
    for i in range(all_pts.shape[1]):
        mask = all_pts[:, i] > thr
        plot_pts = all_pts[mask]
        ax.plot(
            plot_pts[:, 0],
            plot_pts[:, 1],
            plot_pts[:, 2],
            "o",
            label=ax_labels[i].split()[0],
        )

    ax.legend(frameon=False)
    ax.plot([0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0], color=grey_col)
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_zlabel(ax_labels[2])
    ax.view_init(45, 45)
    return ax


def plot_naive_centroid(
    nulls,
    swaps,
    ax=None,
    biggest_extreme=None,
    n_bins=41,
    c_color=(0.1, 0.6, 0.1),
    s_color=(0.6, 0.1, 0.1),
    p_thr=0.05,
    limit_bins=None,
    plot_pval=True,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if len(ax) > 1:
        ax_null = ax[0]
        ax_swap = ax[1]
        null_histtype = "bar"
        swap_histtype = "bar"
    else:
        ax_null = ax
        ax_swap = ax
        null_histtype = "bar"
        swap_histtype = "step"
    extreme = np.nanmax(np.abs((np.nanmin(nulls), np.nanmax(nulls))))
    if biggest_extreme is not None:
        extreme = min(biggest_extreme, extreme)
    if limit_bins is None:
        limit_bins = (-extreme, extreme + 1)
    bins = np.linspace(*limit_bins, n_bins)
    ax_null.hist(nulls, bins=bins, density=True, histtype=null_histtype, color=c_color)
    if len(swaps.shape) > 1:
        swaps = np.mean(swaps, axis=0)
    ax_swap.hist(swaps, density=True, bins=bins, histtype=swap_histtype, color=s_color)
    gpl.add_vlines([0, 1], ax_null)
    gpl.add_vlines([0, 1], ax_swap)
    m_null = np.nanmedian(nulls)
    m_swaps = np.nanmedian(swaps)

    gpl.add_vlines(m_null, ax_null, color=c_color, alpha=0.5)
    gpl.add_vlines(m_swaps, ax_swap, color=s_color, alpha=0.5)
    utest = sts.mannwhitneyu(swaps, nulls, alternative="greater", nan_policy="omit")
    diffs = u.bootstrap_diff(swaps, nulls, np.nanmedian)
    info = (diffs, utest)
    y_low, y_up = ax_null.get_ylim()
    if utest.pvalue < p_thr and plot_pval:
        ax_null.plot([m_swaps], [y_up - y_up * 0.1], "*", ms=5, color=s_color)

    return ax, info


def plot_nc_sweep(
    ax_vals, plot_arr, fwid=3, axs=None, p_thr=0.01, vmin=None, vmax=None, **kwargs
):
    if axs is None:
        n_rows, n_cols = plot_arr.shape[-2:]
        f, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * fwid, n_rows * fwid),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
    if vmin is None:
        vmin = np.nanmin(plot_arr)
    if vmax is None:
        vmax = np.nanmax(plot_arr)
    for i1, i2 in u.make_array_ind_iterator(axs.shape):
        plot_map = plot_arr[..., i1, i2]
        if plot_map.shape[0] == 1:
            plot_map = np.concatenate((plot_map, plot_map), axis=0)
            ax_vals = list(av for av in ax_vals)
            ax_vals[0] = (0, 1)
        if plot_map.shape[1] == 1:
            plot_map = np.concatenate((plot_map, plot_map), axis=1)
            ax_vals = list(av for av in ax_vals)
            ax_vals[1] = (0, 1)
        gpl.pcolormesh(
            *ax_vals, plot_map.T, ax=axs[i1, i2], vmin=vmin, vmax=vmax, **kwargs
        )


def plot_naive_centroid_dict_indiv(*dicts, axs=None, fwid=3, **kwargs):
    if axs is None:
        plot_dims = (len(dicts[0]), len(dicts))
        f, axs = plt.subplots(
            *plot_dims,
            figsize=(fwid * plot_dims[1], fwid * plot_dims[0]),
            squeeze=False,
        )
    info_dicts = []
    for i, dict_i in enumerate(dicts):
        info_dicts.append({})
        for j, (k, (null, swaps)) in enumerate(dict_i.items()):
            ax_ji, info = plot_naive_centroid(null, swaps, ax=axs[j, ..., i], **kwargs)
            info_dicts[i][k] = info
    return axs, info_dicts


def merge_session_dicts(*dicts):
    comb_dicts = []
    for dict_i in dicts:
        nulls_all = list(v[0] for v in dict_i.values())
        if len(nulls_all[0].shape) > 1:
            nulls_all = list(np.concatenate(na) for na in nulls_all)
        nulls = np.concatenate(nulls_all, axis=0)
        swaps_all = list(np.mean(v[1], axis=0) for v in dict_i.values())
        if len(swaps_all[0].shape) > 1:
            swaps_all = list(np.concatenate(sa) for sa in swaps_all)
        swaps = np.concatenate(swaps_all, axis=0)
        comb_dicts.append({"comb": (nulls, swaps)})
    return comb_dicts


def merge_centroid_dicts(*dicts):
    v = list(dicts[0].values())[0]
    n_conds = len(v)
    n_cents = len(v[0])
    out_arr = np.array((n_conds, n_cents), dtype=object)
    comb_dicts = []
    for dict_i in dicts:
        for i, j in u.make_array_ind_iterator((n_conds, n_cents)):
            c_ij_all = list(v[2][i][j] for v in dict_i.values())
            c_ij = np.concatenate(c_ij_all, axis=0)
            out_arr[i, j] = c_ij
        comb_dicts.append({"comb": out_arr})
    return comb_dicts


def plot_naive_centroid_dict_comb(*dicts, **kwargs):
    comb_dicts = merge_session_dicts(*dicts)
    return plot_naive_centroid_dict_indiv(*comb_dicts, **kwargs)


def _mean_select(*args):
    return list(np.mean(arg) for arg in args)


def _max_select(a1, *args):
    ind = np.argmax(a1)
    return [a1[ind]] + list(arg[ind] for arg in args)


def plot_target_swap_group(groups, axs=None):
    assert len(axs) == len(groups)
    for i, group in enumerate(groups):
        axs[i].plot([0, 1], [0, 1], color=(0.9, 0.9, 0.9))
        gpl.add_vlines(0.5, axs[i])
        gpl.add_hlines(0.5, axs[i])
        nulls_all = []
        swaps_all = []
        for j, (k, item) in enumerate(group.items()):
            (nulls_targ, swaps_targ, nulls_dist, swaps_dist) = item
            nulls_all.append(np.nanmean(nulls_dist))
            swaps_all.append(np.nanmean(swaps_dist))

        axs[i].plot(nulls_all, swaps_all, "o")
        wil_test = sts.wilcoxon(
            np.array(nulls_all) - np.array(swaps_all),
            alternative="greater",
            nan_policy="omit",
        )
        axs[i].set_xlim([0, 1])
        axs[i].set_ylim([0, 1])
        print(wil_test)


def plot_swap_group(groups, axs=None, select_func=_max_select):
    assert len(axs) == len(groups)
    for i, group in enumerate(groups):
        axs[i].plot([-0.5, 0.5], [-0.5, 0.5], color=(0.9, 0.9, 0.9))
        gpl.add_vlines(0, axs[i])
        gpl.add_hlines(0, axs[i])
        diffs_null = []
        diffs_swap = []
        nulls_all = []
        swaps_all = []
        for j, (k, item) in enumerate(group.items()):
            item_pt = select_func(*item)
            (nulls_targ, swaps_targ, nulls_dist, swaps_dist) = item_pt

            diffs_null.append(nulls_targ - swaps_targ)
            diffs_swap.append(nulls_targ - swaps_dist)
            nulls_all.append(swaps_targ - 0.5)
            swaps_all.append(swaps_dist - 0.5)
        axs[i].plot(nulls_all, swaps_all, "o")
        wil_test = sts.wilcoxon(
            np.array(nulls_all) - np.array(swaps_all),
            alternative="less",
            nan_policy="omit",
        )
        axs[i].set_xlim([-0.5, 0.5])
        axs[i].set_ylim([-0.5, 0.5])
        print(wil_test)


def plot_forget_group(groups, axs=None, select_func=_max_select):
    assert len(axs) == len(groups)
    for i, group in enumerate(groups):
        axs[i].plot([0.25, 1], [0.25, 1], color=(0.9, 0.9, 0.9))
        gpl.add_vlines(0.5, axs[i])
        gpl.add_hlines(0.5, axs[i])
        diffs = []
        nulls_all = []
        swaps_all = []
        for j, (k, (nulls, swaps)) in enumerate(group.items()):
            null_pt, swap_pt = select_func(nulls, swaps)

            diffs.append(null_pt - swap_pt)
            nulls_all.append(null_pt)
            swaps_all.append(swap_pt)
        axs[i].plot(nulls_all, swaps_all, "o")
        wil_test = sts.wilcoxon(diffs, alternative="greater", nan_policy="omit")
        print(wil_test)


def plot_forget_dict(*args, **kwargs):
    return plot_fs_dict(*args, plot=plot_forget_group, **kwargs)


def plot_swap_dict(*args, **kwargs):
    return plot_fs_dict(*args, plot=plot_swap_group, **kwargs)


def plot_target_swap_dict(*args, **kwargs):
    return plot_fs_dict(*args, plot=plot_target_swap_group, **kwargs)


def plot_fs_dict(
    forget_dict,
    use_keys=("forget_cu", "forget_cl"),
    session_dict=None,
    axs=None,
    fwid=3,
    regions="all",
    plot=plot_forget_group,
    cond_type=None,
):
    if session_dict is None:
        session_dict = dict(
            elmo_range=range(13), waldorf_range=range(13, 24), comb_range=range(24)
        )

    if axs is None:
        n_cols = len(use_keys)
        n_rows = len(session_dict)
        f, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fwid * n_cols, fwid * n_rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
    list(axs[0, j].set_title(k) for j, k in enumerate(use_keys))
    for i, (r_key, use_range) in enumerate(session_dict.items()):
        group = []
        for uk in use_keys:
            if cond_type is not None:

                def filt(k):
                    return k[0] in use_range and k[1] == regions and k[2] == cond_type

            else:

                def filt(k):
                    return k[0] in use_range and k[1] == regions

            use_entries = {k: v for k, v in forget_dict[uk].items() if filt(k)}
            group.append(use_entries)
        axs[i, 0].set_ylabel(r_key)
        plot(group, axs[i])
    return axs


def vec_correlation(c1, c2):
    c1_null, c1_swap = c1
    c2_null, c2_swap = c2
    c1_vu = u.make_unit_vector(c1_swap - c1_null)
    c2_vu = u.make_unit_vector(c2_swap - c2_null)
    out = np.sum(c1_vu * c2_vu, axis=1)
    return out


def config_distance(c1, c2):
    c1_null, c1_swap = c1
    c2_null, c2_swap = c2

    null_config = c1_null + c2_null
    swap_config = c1_swap + c2_swap
    dists = np.sqrt(np.sum((null_config - swap_config) ** 2, axis=1))
    c1_ax_dist = np.sqrt(np.sum((c1_null - c1_swap) ** 2, axis=1))
    c2_ax_dist = np.sqrt(np.sum((c2_null - c2_swap) ** 2, axis=1))
    return dists, c1_ax_dist, c2_ax_dist


def plot_config_differences(
    centroid_dict,
    k1="d1_cl",
    k2="d1_cu",
    session_dict=None,
    axs=None,
    fwid=3,
    biggest_extreme=2,
    cent_ind=2,
    p_ind=3,
    swap_p_ind=1,
    regions="all",
):
    if session_dict is None:
        session_dict = dict(
            elmo_range=range(13), waldorf_range=range(13, 24), comb_range=range(24)
        )

    if axs is None:
        n_cols = 4
        n_rows = len(session_dict)
        f, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fwid * n_cols, fwid * n_rows),
            sharex="col",
            sharey="col",
        )
    dists = {}
    for i, (r_key, use_range) in enumerate(session_dict.items()):
        d1, d2 = swan.filter_nc_dis(
            centroid_dict, (k1, k2), (), use_range, regions=regions
        )
        null_conf_dists = []
        swap_conf_dists = []
        n1_dists = []
        n2_dists = []
        s1_dists = []
        s2_dists = []
        vcorr_nulls = []
        vcorr_swaps = []
        vcorr_mus = []
        p_rates_nulls = []
        p_rates_swaps = []
        for k, v1 in d1.items():
            v2 = d2[k]
            null_cents1, swap_cents1 = v1[cent_ind]
            null_cents2, swap_cents2 = v2[cent_ind]
            null_ps1, swap_ps1 = v1[p_ind]
            all_ps = np.concatenate((null_ps1, swap_ps1), axis=0)
            vcorr_null = vec_correlation(null_cents1, null_cents2)
            out = config_distance(null_cents1, null_cents2)
            null_dists, n1_dist, n2_dist = out
            null_conf_dists.extend(null_dists)
            n1_dists.extend(n1_dist)
            n2_dists.extend(n2_dist)

            vcorr_swap = vec_correlation(swap_cents1, swap_cents2)
            out = config_distance(swap_cents1, swap_cents2)
            swap_dists, s1_dist, s2_dist = out
            swap_conf_dists.extend(swap_dists)
            s1_dists.extend(s1_dist)
            s2_dists.extend(s2_dist)
            vcorr_nulls.extend(vcorr_null)
            vcorr_swaps.extend(vcorr_swap)
            vcorr_mus.append(np.mean(vcorr_null))
            p_rates_nulls.extend(null_ps1[:, swap_p_ind])
            p_rates_swaps.extend(swap_ps1[:, swap_p_ind])
        axs[i, 0].hist(null_conf_dists, density=True)
        axs[i, 0].hist(swap_conf_dists, density=True, histtype="step")
        axs[i, 1].hist(n1_dists, density=True)
        axs[i, 1].hist(s1_dists, density=True, histtype="step")
        axs[i, 2].hist(n2_dists, density=True)
        axs[i, 2].hist(s2_dists, density=True, histtype="step")
        axs[i, 3].plot(vcorr_nulls, p_rates_nulls, "o")
        axs[i, 3].plot(vcorr_swaps, p_rates_swaps, "o")
        r_nulls = sts.pearsonr(vcorr_nulls, p_rates_nulls)
        r_swaps = sts.pearsonr(vcorr_swaps, p_rates_swaps)
        # axs[i, 3].hist(vcorr_swaps, density=True, histtype='step')
        print(r_key, np.mean(vcorr_nulls), np.mean(vcorr_mus))
        print(np.mean(vcorr_swaps))
        print(r_nulls, r_swaps)
        dists[r_key] = (null_conf_dists, swap_conf_dists)
    return dists


def plot_nc_epoch_corr(
    centroid_dict,
    use_d1s=("d1_cl", "d1_cu"),
    session_dict=None,
    cond_types=("retro",),
    d2_key="d2",
    axs=None,
    fwid=3,
    biggest_extreme=2,
    regions="all",
):
    if session_dict is None:
        session_dict = dict(
            elmo_range=range(13), waldorf_range=range(13, 24), comb_range=range(24)
        )

    if axs is None:
        n_cols = len(use_d1s) * len(cond_types)
        n_rows = len(session_dict)
        f, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fwid * n_cols, fwid * n_rows),
            sharex=True,
            sharey=True,
        )
    titles = list(use_d1s) + list(" ".join((d2_key, ct)) for ct in cond_types)
    list(ax.set_title(titles[i]) for i, ax in enumerate(axs[0]))
    for i, (r_key, use_range) in enumerate(session_dict.items()):
        use_dis = swan.filter_nc_dis(
            centroid_dict,
            use_d1s,
            cond_types,
            use_range,
            regions=regions,
            d2_key=d2_key,
        )
        merge_dis = merge_session_dicts(*use_dis)
        nulls_cl, swaps_cl = merge_dis[0]["comb"]
        nulls_cu, swaps_cu = merge_dis[1]["comb"]
        nulls_d2, swaps_d2 = merge_dis[2]["comb"]
        axs[i, 0].plot(nulls_cl, nulls_d2, "o")
        axs[i, 0].plot(swaps_cl, swaps_d2, "o")
        print(
            "{} cl null".format(r_key),
            sts.spearmanr(nulls_cl, nulls_d2, nan_policy="omit"),
        )
        print(
            "{} cl swap".format(r_key),
            sts.spearmanr(swaps_cl, swaps_d2, nan_policy="omit"),
        )
        axs[i, 1].plot(nulls_cu, nulls_d2, "o")
        axs[i, 1].plot(swaps_cu, swaps_d2, "o")
        print(
            "{} cu null".format(r_key),
            sts.spearmanr(nulls_cu, nulls_d2, nan_policy="omit"),
        )
        print(
            "{} cu swap".format(r_key),
            sts.spearmanr(swaps_cu, swaps_d2, nan_policy="omit"),
        )
        axs[i, 0].set_ylabel(r_key)
    return axs


def plot_all_nc_dict(
    centroid_dict,
    use_d1s=("d1_cl", "d1_cu"),
    session_dict=None,
    cond_types=("pro", "retro"),
    d2_key="d2",
    axs=None,
    fwid=3,
    biggest_extreme=2,
    regions="all",
    plot_inverted=False,
    **kwargs,
):
    if session_dict is None:
        session_dict = dict(
            elmo_range=range(13), waldorf_range=range(13, 24), comb_range=range(24)
        )

    if axs is None:
        n_cols = len(use_d1s) + len(cond_types)
        n_rows = len(session_dict)
        f, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fwid * n_cols, fwid * n_rows),
            sharex=True,
            sharey=True,
        )
    if len(axs.shape) == 2:
        axs = np.expand_dims(axs, 1)
    axs = np.expand_dims(axs, 1)
    max_ys = []

    titles = list(use_d1s) + list(" ".join((d2_key, ct)) for ct in cond_types)
    list(ax[0, 0].set_title(titles[i]) for i, ax in enumerate(axs[0].T))
    info_groups = {}
    for i, (r_key, use_range) in enumerate(session_dict.items()):
        use_dis = swan.filter_nc_dis(
            centroid_dict,
            use_d1s,
            cond_types,
            use_range,
            regions=regions,
            d2_key=d2_key,
        )
        _, info_dicts = plot_naive_centroid_dict_comb(
            *use_dis,
            biggest_extreme=biggest_extreme,
            axs=axs[i],
            **kwargs,
        )

        info_groups[r_key] = info_dicts
        axs[i, 0, 0, 0].set_ylabel(r_key)
        if plot_inverted:
            for ind in u.make_array_ind_iterator(axs[i, 0, 1].shape):
                max_ys.append(axs[i, 0, 1][ind].get_ylim()[-1])
                axs[i, 0, 1][ind].invert_yaxis()
            for ind in u.make_array_ind_iterator(axs[i, 0, 0].shape):
                max_ys.append(axs[i, 0, 0][ind].get_ylim()[-1])
    if plot_inverted:
        max_y = np.max(max_ys)
        for ind in u.make_array_ind_iterator(axs.shape):
            axs[ind].set_xlabel("")
            if ind[-1] < (axs.shape[-1] - 1):
                gpl.clean_plot_bottom(axs[ind])
            else:
                gpl.clean_plot_bottom(axs[ind], keeplabels=True)
            if axs[ind].yaxis_inverted():
                axs[ind].set_ylim([max_y, 0])
            else:
                axs[ind].set_ylim([0, max_y])
            gpl.clean_plot(axs[ind], ind[0])

    return axs, info_groups


def plot_nc_diffs(info_groups, diff_axs, colors=None, plot_key="comb"):
    if colors is None:
        colors = {}
    for i, (k, l) in enumerate(info_groups.items()):
        m_color = colors.get(k)
        for j, d in enumerate(l):
            ax = diff_axs[j]
            diffs, test = d[plot_key]
            gpl.violinplot([diffs], [i], color=[m_color], ax=ax)
    for ax in diff_axs:
        gpl.clean_plot(ax, 0)
        gpl.add_hlines(0, ax)


def _plot_simplex(pts, ax, line_grey_col=(0.6, 0.6, 0.6)):
    pts_x = pts[:, 1] - pts[:, 0]
    pts_y = pts[:, 2] - (pts[:, 0] + pts[:, 1])
    ax.plot(pts_x, pts_y, "o")
    ax.plot([-1, 0], [-1, 1], color=line_grey_col)
    ax.plot([0, 1], [1, -1], color=line_grey_col)
    ax.plot([-1, 1], [-1, -1], color=line_grey_col)
    ax.set_aspect("equal")


def plot_cumulative_simplex(
    o_dict,
    ax=None,
    fwid=3,
    model_key="other",
    simplex_key="p_err",
    plot_type="retro",
    session_avg=True,
    n_boots=1000,
    color=None,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots()

    samps_all = []
    session_means = []
    for k, (fd, data) in o_dict.items():
        samps_k = np.concatenate(fd[model_key].posterior[simplex_key])
        if len(samps_k.shape) > 2:
            ind = swaux.get_type_ind(plot_type, data)
            samps_k = samps_k[:, ind]
        samps_all.extend(samps_k)
        session_means.append(np.mean(samps_k, axis=0))
    gpl.simplefy(np.array(samps_all), ax=ax, **kwargs)
    if session_avg:
        avg_conf = u.bootstrap_array(np.array(session_means), np.mean, n=n_boots)
        gpl.simplefy(avg_conf, ax=ax, how="contours", color=color)


def plot_cumulative_simplex_1d(
    o_dict,
    ax=None,
    fwid=3,
    model_key="other",
    simplex_key="p_err",
    ref_ind=1,
    plot_type="retro",
    session_avg=True,
    n_boots=1000,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(figsize=(fwid, fwid))
    samps_all = []
    session_means = []
    for k, (fd, data) in o_dict.items():
        samps_k = np.concatenate(fd[model_key].posterior[simplex_key])
        if len(samps_k.shape) > 2:
            type_ind = swaux.get_type_ind(plot_type, data)
            samps_k = samps_k[:, type_ind]
        samps_all.extend(1 - samps_k[:, ref_ind])
        session_means.append(np.mean(1 - samps_k[:, ref_ind], axis=0))
    ax.hist(samps_all, density=True, **kwargs)
    gpl.clean_plot(ax, 0)
    if session_avg:
        avg_conf = u.bootstrap_list(np.array(session_means), np.mean, n=n_boots)
        y_val = ax.get_ylim()[-1]
        gpl.plot_trace_werr(
            np.expand_dims(avg_conf, 1),
            [y_val],
            conf95=True,
            ax=ax,
            elinewidth=3,
            fill=False,
            **kwargs,
        )


def plot_all_simplices_1d(
    o_dict,
    axs_dict=None,
    fwid=3,
    model_key="other",
    simplex_key="p_guess_err",
    type_order=("retro", "pro"),
    line_grey_col=(0.6, 0.6, 0.6),
    plot_rate=True,
    p_ind=1,
):
    if axs_dict is None:
        axs_dict = {}

    if plot_rate:
        rows = 2
    else:
        rows = 1
    for k, fit_dict in o_dict.items():
        if axs_dict.get(k) is None:
            if k[-1] == "joint":
                sec_ax = 2
            else:
                sec_ax = 1
            f, axs_k = plt.subplots(
                rows, sec_ax, figsize=(fwid * sec_ax, fwid * rows), squeeze=False
            )
            f.suptitle(k)
        for sess_ind, (fit, data) in fit_dict.items():
            simplex = np.concatenate(fit[model_key].posterior[simplex_key], axis=0)
            if len(simplex.shape) == 3:
                for i, type_ in enumerate(type_order):
                    ind = swaux.get_type_ind(type_, data)
                    pts = simplex[:, ind]
                    _, _, b = axs_k[0, i].hist(pts[:, 0])
                    axs_k[0, i].set_xlabel("{}\n{}".format(simplex_key, type_))
                    if plot_rate:
                        ax = axs_k[1, i]
                        prob = np.nanmean(data["p"], axis=0)[p_ind]
                        gpl.plot_trace_werr(
                            sess_ind,
                            prob * pts[:, 0:1],
                            ax=ax,
                            conf95=True,
                            color=b[0].get_facecolor(),
                        )
            else:
                pts = simplex
                _, _, b = axs_k[0, 0].hist(pts[:, 0])
                axs_k[0, 0].set_xlabel("{}".format(simplex_key))
                if plot_rate:
                    ax = axs_k[1, 0]
                    prob = np.mean(data["p"], axis=0)[p_ind]
                    # trls = prob*pts[:, 0:1]*len(data['p'])
                    trls = pts[:, 0:1]
                    gpl.plot_trace_werr(
                        sess_ind,
                        trls,
                        ax=ax,
                        conf95=True,
                        color=b[0].get_facecolor(),
                        lw=10,
                    )
        gpl.add_hlines(0, axs_k[1, 0])


def plot_rates(
    *o_dicts,
    simplex_key="p_err",
    ref_ind=1,
    ax=None,
    colors=None,
    lw=1.5,
    task_type="retro",
    diff=0,
):
    if colors is None:
        colors = (None,) * len(o_dicts)
    if ax is None:
        f, ax = plt.subplots()
    for i, od in enumerate(o_dicts):
        color = colors[i]
        for sess_ind, (fit, data) in od.items():
            fit = fit["other"]
            simpl = np.concatenate(fit.posterior[simplex_key])
            if len(simpl.shape) > 2:
                ind = swaux.get_type_ind(task_type, data)
                simpl = simpl[:, ind]
            pts = 1 - simpl[:, ref_ind]
            l = gpl.plot_trace_werr(
                sess_ind + diff,
                np.expand_dims(pts, 1),
                ax=ax,
                conf95=True,
                fill=False,
                elinewidth=lw,
                color=color,
            )
            color = l[0].get_color()
    gpl.add_hlines(0, ax)
    gpl.add_hlines(1, ax)


def plot_all_simplices(
    o_dict,
    axs_dict=None,
    fwid=3,
    model_key="other",
    simplex_key="p_err",
    thin=10,
    type_order=("retro", "pro"),
    line_grey_col=(0.6, 0.6, 0.6),
):
    if axs_dict is None:
        axs_dict = {}
    for k, fit_dict in o_dict.items():
        if axs_dict.get(k) is None:
            if k[-1] == "joint":
                sec_ax = 2
            else:
                sec_ax = 1
            f, axs_k = plt.subplots(1, sec_ax, figsize=(fwid * sec_ax, fwid))
        for sess_ind, (fit, data) in fit_dict.items():
            simplex = np.concatenate(fit[model_key].posterior[simplex_key], axis=0)
            if len(simplex.shape) == 3:
                for i, type_ in enumerate(type_order):
                    ind = swaux.get_type_ind(type_, data)
                    pts = simplex[::thin, ind]
                    _plot_simplex(pts, axs_k[i], line_grey_col=line_grey_col)
            else:
                pts = simplex[::thin]
                _plot_simplex(pts, axs_k, line_grey_col=line_grey_col)


def cond_func(x, field=None, split=None):
    s1 = x[field] < split
    s2 = x[field] >= split
    return s1, s2


def mask_func(x, field=None, targ=None, op=np.equal):
    mask = op(x[field], targ)
    return mask


no_cue = ft.partial(mask_func, field="IsUpperSample", targ=-1, op=np.greater)

upper_col_split = ft.partial(cond_func, field="upper_color", split=np.pi)
lower_col_split = ft.partial(cond_func, field="lower_color", split=np.pi)
targ_col_split = ft.partial(cond_func, field="LABthetaTarget", split=np.pi)
dist_col_split = ft.partial(cond_func, field="LABthetaDist", split=np.pi)

cue_split = ft.partial(cond_func, field="IsUpperSample", split=0.5)


def plot_period_units_tuning(
    data,
    date,
    neur_inds=None,
    samples_t_pair=(0, 0.5),
    cue_t_pair=(-0.5, 0),
    wheel_t_pair=(-0.5, 0),
    default_upper_color=np.array((0, 187, 62)) / 255,
    default_lower_color=np.array((41, 130, 255)) / 255,
    default_target_color=np.array((0, 187, 62)) / 255,
    default_distr_color=np.array((41, 130, 255)) / 255,
    default_cue_color=np.array((181, 144, 225)) / 255,
    axs=None,
    fwid=3,
    date_field="date",
    neur_num_field="n_neurs",
    use_retro=True,
    polar=True,
    **kwargs,
):
    if neur_inds is None:
        rd = data.session_mask(data[date_field] == date)
        n_neurs = rd[neur_num_field].iloc[0]
        neur_inds = range(n_neurs)

    base_colors_pre = (default_upper_color, default_lower_color)
    base_colors_wh = (default_target_color, default_distr_color)
    if axs is None:
        figsize = (fwid * 3, fwid * len(neur_inds))
        if polar:
            subplot_kw = {"projection": "polar"}
        else:
            subplot_kw = {}

        f, axs = plt.subplots(
            len(neur_inds),
            3,
            figsize=figsize,
            sharey="row",
            sharex="col",
            subplot_kw=subplot_kw,
        )

    if use_retro:
        samp_ind = 0
        cue_ind = 1
    else:
        samp_ind = 1
        cue_ind = 0
    wheel_ind = 2

    if not use_retro:
        base_colors_pre = (default_target_color, default_distr_color)
    else:
        base_colors_cue = base_colors_pre

    plot_single_neuron_tuning_samples(
        data,
        *samples_t_pair,
        date,
        neur_inds,
        base_colors=base_colors_pre,
        axs=axs[:, samp_ind],
        use_retro=use_retro,
        **kwargs,
    )

    plot_single_neuron_tuning_cue(
        data,
        *cue_t_pair,
        date,
        neur_inds,
        base_colors=base_colors_pre,
        axs=axs[:, cue_ind],
        use_retro=use_retro,
        **kwargs,
    )

    plot_single_neuron_tuning_wheel(
        data,
        *wheel_t_pair,
        date,
        neur_inds,
        base_colors=base_colors_wh,
        axs=axs[:, wheel_ind],
        use_retro=use_retro,
        **kwargs,
    )
    return axs


def plot_period_units_trace(
    data,
    date,
    neur_inds=None,
    samples_t_pair=(-0.5, 0.7),
    cue_t_pair=(-0.7, 0.5),
    wheel_t_pair=(-1.2, 0.2),
    plot_colors=True,
    default_upper_color=np.array((0, 187, 62)) / 255,
    default_lower_color=np.array((41, 130, 255)) / 255,
    default_target_color=np.array((0, 187, 62)) / 255,
    default_distr_color=np.array((41, 130, 255)) / 255,
    default_cue_color=np.array((181, 144, 225)) / 255,
    axs=None,
    fwid=3,
    date_field="date",
    neur_num_field="n_neurs",
    use_retro=True,
    **kwargs,
):
    if neur_inds is None:
        rd = data.session_mask(data[date_field] == date)
        n_neurs = rd[neur_num_field].iloc[0]
        neur_inds = range(n_neurs)

    if plot_colors:
        base_colors_pre = (default_upper_color, default_lower_color)
        base_colors_wh = (default_target_color, default_distr_color)
    else:
        base_colors = (default_cue_color,)
    if axs is None:
        figsize = (fwid * 3, fwid * len(neur_inds))
        f, axs = plt.subplots(
            len(neur_inds), 3, figsize=figsize, sharey="row", sharex="col"
        )

    if use_retro:
        samp_ind = 0
        cue_ind = 1
    else:
        samp_ind = 1
        cue_ind = 0
    wheel_ind = 2

    if not use_retro:
        plot_colors_cue = False
        base_colors_cue = (default_cue_color,)
        base_colors_pre = (default_target_color, default_distr_color)
    else:
        plot_colors_cue = plot_colors
        base_colors_cue = base_colors_pre

    plot_single_neuron_trace_samples(
        data,
        *samples_t_pair,
        date,
        neur_inds,
        plot_colors=plot_colors,
        base_colors=base_colors_pre,
        axs=axs[:, samp_ind],
        use_retro=use_retro,
        **kwargs,
    )

    plot_single_neuron_trace_cue(
        data,
        *cue_t_pair,
        date,
        neur_inds,
        plot_colors=plot_colors_cue,
        base_colors=base_colors_cue,
        axs=axs[:, cue_ind],
        use_retro=use_retro,
        **kwargs,
    )

    plot_single_neuron_trace_wheel(
        data,
        *wheel_t_pair,
        date,
        neur_inds,
        plot_colors=plot_colors,
        base_colors=base_colors_wh,
        axs=axs[:, wheel_ind],
        use_retro=use_retro,
        **kwargs,
    )
    for ax_row in axs:
        gpl.clean_plot(ax_row[1], 1)
        gpl.clean_plot(ax_row[2], 2)
        ax_row[0].set_ylabel("spikes/s")
    axs[-1, samp_ind].set_xlabel("time from stimuli (s)")
    axs[-1, cue_ind].set_xlabel("time from cue (s)")
    axs[-1, wheel_ind].set_xlabel("time from\nresponse wheel (s)")
    return axs


def plot_single_neuron_trace_samples(*args, plot_colors=True, use_retro=True, **kwargs):
    if plot_colors:
        if use_retro:
            cond_funcs = (upper_col_split, lower_col_split)
        else:
            cond_funcs = (targ_col_split, dist_col_split)
    else:
        cond_funcs = (cue_split,)
    if use_retro:
        background = None
    else:
        background = (-0.5, 0)

    return plot_single_neuron_trace(
        *args,
        tzf="SAMPLES_ON_diode",
        cond_funcs=cond_funcs,
        use_retro=use_retro,
        background=background,
        **kwargs,
    )


def plot_single_neuron_trace_cue(*args, use_retro=True, plot_colors=True, **kwargs):
    if plot_colors:
        cond_funcs = (targ_col_split, dist_col_split)
    else:
        cond_funcs = (cue_split,)
    if use_retro:
        tzf = "CUE2_ON_diode"
        background = (-0.5, 0)
    else:
        tzf = "CUE1_ON_diode"
        background = None
    return plot_single_neuron_trace(
        *args,
        tzf=tzf,
        cond_funcs=cond_funcs,
        use_retro=use_retro,
        background=background,
        **kwargs,
    )


def plot_single_neuron_trace_wheel(*args, plot_colors=True, **kwargs):
    if plot_colors:
        cond_funcs = (targ_col_split, dist_col_split)
    else:
        cond_funcs = (cue_split,)
    return plot_single_neuron_trace(
        *args,
        tzf="WHEEL_ON_diode",
        cond_funcs=cond_funcs,
        background=(-0.5, 0),
        **kwargs,
    )


def plot_single_neuron_tuning_samples(*args, **kwargs):
    colors = ("upper_color", "lower_color")
    return plot_single_neuron_tuning(
        *args, tzf="SAMPLES_ON_diode", colors=colors, **kwargs
    )


def plot_single_neuron_tuning_cue(*args, use_retro=True, **kwargs):
    colors = ("upper_color", "lower_color")
    if use_retro:
        tzf = "CUE2_ON_diode"
    else:
        tzf = "CUE1_ON_diode"
    return plot_single_neuron_tuning(
        *args, tzf=tzf, colors=colors, use_retro=use_retro, **kwargs
    )


def plot_single_neuron_tuning_wheel(*args, **kwargs):
    colors = ("LABthetaTarget", "LABthetaDist")
    use_cues = True
    return plot_single_neuron_tuning(
        *args, tzf="WHEEL_ON_diode", colors=colors, use_cues=use_cues, **kwargs
    )


def plot_population_toruses(
    data,
    date,
    cue_t_pair=(-0.5, 0),
    wheel_t_pair=(-0.5, 0),
    axs=None,
    fwid=3,
    date_field="date",
    use_retro=True,
    n_dims=3,
    wheel_tzf="WHEEL_ON_diode",
    cue_colors=("upper_color", "lower_color"),
    default_cue_color=None,
    wheel_colors=("LABthetaTarget", "LABthetaDist"),
    **kwargs,
):
    if axs is None:
        figsize = (fwid * 2, fwid)
        if n_dims == 3:
            subplot_kw = {"projection": "3d"}
        else:
            subplot_kw = {}

        f, axs = plt.subplots(2, 2, figsize=figsize, subplot_kw=subplot_kw)
    if use_retro:
        cue_tzf = "CUE2_ON_diode"
        plot_func_cue = plot_population_torus
    else:
        cue_tzf = "SAMPLES_ON_diode"
        plot_func_cue = plot_population_cue

    plot_func_cue(
        data,
        *cue_t_pair,
        date,
        tzf=cue_tzf,
        axs=axs[0],
        use_retro=use_retro,
        cue_color=default_cue_color,
        colors=cue_colors,
        **kwargs,
    )

    plot_population_torus(
        data,
        *wheel_t_pair,
        date,
        tzf=wheel_tzf,
        axs=axs[1],
        use_retro=use_retro,
        colors=wheel_colors,
        **kwargs,
    )


def plot_population_cue(
    data,
    start,
    end,
    session_date,
    tzf="SAMPLES_ON_diode",
    colors=("upper_color", "lower_color"),
    cue="IsUpperSample",
    use_retro=True,
    base_colors=None,
    axs=None,
    fwid=3,
    date_field="date",
    use_corr=True,
    error_field="StopCondition",
    region_key="neur_regions",
    use_cues=False,
    n_bins=6,
    spline_knots=5,
    spline_degree=2,
    n_pts=100,
    swap_color=None,
    corr_color=None,
    ms=5,
    grey_col=(0.8, 0.8, 0.8),
    eg_pt=(np.pi, 0),
    n_dims=3,
    cue_color=None,
    col_diff=0.1,
):
    if cue_color is None:
        cue_color = gpl.get_next_n_colors(1)[0]
    if axs is None:
        if n_dims == 3:
            subplot_kw = {"projection": "3d"}
        else:
            subplot_kw = {}
        f, axs = plt.subplots(
            1, 2, figsize=(2 * fwid, fwid), squeeze=False, subplot_kw=subplot_kw
        )
        axs = axs.flatten()
    data = data.session_mask(data[date_field] == session_date)
    if use_retro:
        data = swan.retro_mask(data)
    else:
        data = swan.pro_mask(data)
    if use_corr:
        mask = data[error_field] >= -1
        data = data.mask(mask)

    cues = data[cue][0]
    pop, xs = data.get_neural_activity(
        end - start, start, end, stepsize=end - start, time_zero_field=tzf
    )

    pop_use = pop[0][..., 0]
    inds = np.arange(len(pop_use))
    m_df = np.zeros(len(pop_use))
    model = na.make_model_pipeline(norm=True, pca=0.99, model=sk_svm.LinearSVC)
    for i in range(len(pop_use)):
        mask_tr = inds != i
        mask_te = inds == i

        model.fit(pop_use[mask_tr], cues[mask_tr])
        m_df[i] = model.decision_function(pop_use[mask_te])
    cue_mask = cues == 0
    cue_up_color = gpl.add_color_value(cue_color, -col_diff)
    cue_down_color = gpl.add_color_value(cue_color, col_diff)

    ys = sts.norm(0, 1).rvs(len(m_df[cue_mask]))
    axs[0].plot(m_df[cue_mask], ys, "o", color=cue_up_color, ms=1, alpha=0.3)
    axs[0].plot(np.mean(m_df[cue_mask]), [0], "o", color=cue_up_color, ms=ms)

    ys = sts.norm(0, 1).rvs(len(m_df[~cue_mask]))
    axs[0].plot(m_df[~cue_mask], ys, "o", color=cue_down_color, ms=1, alpha=0.3)
    axs[0].plot(np.mean(m_df[~cue_mask]), [0], "o", color=cue_down_color, ms=ms)

    gpl.add_vlines(0, axs[0])
    gpl.clean_plot(axs[0], 1)

    gpl.clean_3d_plot(axs[1])
    gpl.make_3d_bars(axs[1], bar_len=0, center=(-0.1, -0.1, -0.1))


def plot_population_torus(
    data,
    start,
    end,
    session_date,
    tzf="SAMPLES_ON_diode",
    colors=("upper_color", "lower_color"),
    cue="IsUpperSample",
    use_retro=True,
    base_colors=None,
    axs=None,
    fwid=3,
    date_field="date",
    use_corr=True,
    error_field="StopCondition",
    region_key="neur_regions",
    use_cues=False,
    n_bins=6,
    spline_knots=5,
    spline_degree=2,
    n_pts=100,
    swap_color=None,
    corr_color=None,
    ms=5,
    grey_col=(0.8, 0.8, 0.8),
    eg_pt=(np.pi, 0),
    n_dims=3,
    **kwargs,
):
    if base_colors is None:
        base_colors = gpl.get_next_n_colors(len(colors))
    if axs is None:
        if n_dims == 3:
            subplot_kw = {"projection": "3d"}
        else:
            subplot_kw = {}
        f, axs = plt.subplots(
            1, 2, figsize=(2 * fwid, fwid), squeeze=False, subplot_kw=subplot_kw
        )
        axs = axs.flatten()
    data = data.session_mask(data[date_field] == session_date)
    if use_retro:
        data = swan.retro_mask(data)
    else:
        data = swan.pro_mask(data)
    if use_corr:
        mask = data[error_field] >= -1
        data = data.mask(mask)

    regions = data[region_key][0].iloc[0]

    cols = list(data[col][0] for col in colors)
    if use_cues:
        cues = data[cue][0]
    else:
        cues = None
    coeffs, spliner = swan.make_lm_coefficients(
        *cols,
        cues=cues,
        spline_knots=spline_knots,
        spline_degree=spline_degree,
        return_spliner=True,
    )

    pop, xs = data.get_neural_activity(
        end - start, start, end, stepsize=end - start, time_zero_field=tzf
    )

    preproc = na.make_model_pipeline(norm=True, pca=0.99, post_norm=True)
    pop_act = preproc.fit_transform(pop[0][..., 0])
    m = sklm.Ridge()
    m.fit(coeffs, pop_act)
    m_predictive = m.predict(coeffs)

    color_tr = np.linspace(0, np.pi * 2, n_pts)
    color_const = np.zeros(n_pts)
    if use_cues:
        cues_const = np.ones(n_pts)
        eg_cues = np.ones(2)
    else:
        cues_const = None
        eg_cues = None
    coeffs_upper = swan.make_lm_coefficients(
        color_tr,
        color_const,
        cues=cues_const,
        spline_knots=spline_knots,
        spline_degree=spline_degree,
        use_spliner=spliner,
    )
    coeffs_lower = swan.make_lm_coefficients(
        color_const,
        color_tr,
        cues=cues_const,
        spline_knots=spline_knots,
        spline_degree=spline_degree,
        use_spliner=spliner,
    )

    coeffs_eg = swan.make_lm_coefficients(
        eg_pt,
        eg_pt[::-1],
        cues=eg_cues,
        spline_knots=spline_knots,
        spline_degree=spline_degree,
        use_spliner=spliner,
    )

    rep_upper = m.predict(coeffs_upper)
    rep_lower = m.predict(coeffs_lower)
    rep_eg = m.predict(coeffs_eg)

    p_upper = skd.PCA(n_dims)
    plot_upper_u = p_upper.fit_transform(rep_upper)
    gpl.plot_colored_line(*plot_upper_u.T, ax=axs[0], cmap="hsv")
    plot_lower_u = p_upper.transform(rep_lower)
    axs[0].plot(*plot_lower_u.T, color=grey_col)

    plot_egs_u = p_upper.transform(rep_eg)
    axs[0].plot(*plot_egs_u[0], "o", ms=ms, color=corr_color, zorder=10)
    axs[0].plot(*plot_egs_u[1], "o", ms=ms, color=swap_color, zorder=10)

    p_lower = skd.PCA(n_dims)

    plot_lower_l = p_lower.fit_transform(rep_lower)
    gpl.plot_colored_line(*plot_lower_l.T, ax=axs[1], cmap="hsv")
    plot_upper_l = p_lower.transform(rep_upper)
    axs[1].plot(*plot_upper_l.T, color=grey_col)

    plot_egs_l = p_lower.transform(rep_eg)
    axs[1].plot(*plot_egs_l[0], "o", ms=ms, color=corr_color, zorder=10)
    axs[1].plot(*plot_egs_l[1], "o", ms=ms, color=swap_color, zorder=10)

    gpl.clean_3d_plot(axs[0])
    gpl.make_3d_bars(axs[0], bar_len=0.5, center=(-0.5, -0.5, -0.5))
    gpl.clean_3d_plot(axs[1])
    gpl.make_3d_bars(axs[1], bar_len=0.5, center=(-0.5, -0.5, -0.5))


def plot_single_neuron_tuning(
    data,
    start,
    end,
    session_date,
    neur_inds,
    tzf="SAMPLES_ON_diode",
    colors=("upper_color", "lower_color"),
    cue="IsUpperSample",
    use_retro=True,
    base_colors=None,
    axs=None,
    fwid=3,
    date_field="date",
    use_corr=True,
    error_field="StopCondition",
    region_key="neur_regions",
    use_cues=False,
    n_bins=6,
    spline_knots=5,
    spline_degree=1,
    polar=True,
    n_pts=100,
):
    if base_colors is None:
        base_colors = gpl.get_next_n_colors(len(colors))
    if axs is None:
        n_plots = len(neur_inds)
        sl = int(np.ceil(np.sqrt(n_plots)))
        if polar:
            subplot_kw = {"projection": "polar"}
        else:
            subplot_kw = {}
        f, axs = plt.subplots(
            sl, sl, figsize=(fwid * sl, fwid * sl), squeeze=False, subplot_kw=subplot_kw
        )
        axs = axs.flatten()
    data = data.session_mask(data[date_field] == session_date)
    if use_retro:
        data = swan.retro_mask(data)
    else:
        data = swan.pro_mask(data)
    if use_corr:
        mask = data[error_field] >= -1
        data = data.mask(mask)

    regions = data[region_key][0].iloc[0]

    cols = list(data[col][0] for col in colors)
    if use_cues:
        cues = data[cue][0]
    else:
        cues = None
    coeffs = swan.make_lm_coefficients(
        *cols, cues=cues, spline_knots=spline_knots, spline_degree=spline_degree
    )

    pop, xs = data.get_neural_activity(
        end - start, start, end, stepsize=end - start, time_zero_field=tzf
    )
    for i, ni in enumerate(neur_inds):
        neur_act = pop[0][:, ni, 0]

        m = sklm.Ridge()
        m.fit(coeffs, neur_act)
        m_predictive = m.predict(coeffs)

        for j, col in enumerate(cols):
            gpl.plot_scatter_average(
                col[mask[0]],
                neur_act[mask[0]],
                ax=axs[i],
                n_bins=n_bins,
                color=base_colors[j],
                polar=polar,
            )
            gpl.plot_scatter_average(
                col[mask[0]],
                m_predictive[mask[0]],
                ax=axs[i],
                n_bins=n_bins,
                color=base_colors[j],
                linestyle="dashed",
                polar=polar,
            )

        # axs[i].set_title('{}, {}, {}'.format(session_date, ni, regions[ni]))

        if polar:
            xs = np.linspace(0, np.pi * 2, n_pts)
            y_low, y_high = axs[i].get_ylim()
            ys = np.ones(n_pts) * y_high
            gpl.plot_colored_line(xs, ys, cmap="hsv", ax=axs[i])
            axs[i].set_ylim((y_low, y_high))
            axs[i].set_xticks([0])
            axs[i].set_xticklabels([""])
            axs[i].set_rlabel_position(0)
            # axs[i].set_ylabel('spikes/s')
            axs[i].spines["polar"].set_visible(False)

    return axs


def plot_single_neuron_trace(
    data,
    start,
    end,
    session_date,
    neur_inds,
    winsize=0.2,
    winstep=0.02,
    tzf="SAMPLES_ON_diode",
    cond_funcs=(upper_col_split, lower_col_split),
    error_field="StopCondition",
    use_retro=True,
    date_field="date",
    axs=None,
    fwid=3,
    base_colors=None,
    col_diff=0.1,
    region_key="neur_regions",
    use_corr=True,
    background=None,
):
    if base_colors is None:
        base_colors = gpl.get_next_n_colors(len(cond_funcs))
    if axs is None:
        n_plots = len(neur_inds)
        sl = int(np.ceil(np.sqrt(n_plots)))
        f, axs = plt.subplots(sl, sl, figsize=(fwid * sl, fwid * sl), squeeze=False)
        axs = axs.flatten()
    data = data.session_mask(data[date_field] == session_date)
    if use_retro:
        data = swan.retro_mask(data)
    else:
        data = swan.pro_mask(data)
    if use_corr:
        mask = data[error_field] >= -1
        data = data.mask(mask)

    cond_masks = list(cf(data) for cf in cond_funcs)
    regions = data[region_key][0].iloc[0]

    pop, xs = data.get_neural_activity(
        winsize, start, end, stepsize=winstep, time_zero_field=tzf
    )
    for i, ni in enumerate(neur_inds):
        neur_act = pop[0][:, ni]

        for j, (m1, m2) in enumerate(cond_masks):
            c1 = gpl.add_color_value(base_colors[j], col_diff)
            c2 = gpl.add_color_value(base_colors[j], -col_diff)
            n1_use = neur_act[m1[0]]
            n2_use = neur_act[m2[0]]
            gpl.plot_trace_werr(xs, n1_use, ax=axs[i], color=c1)
            gpl.plot_trace_werr(xs, n2_use, ax=axs[i], color=c2)

        # axs[i].set_title('{}, {}, {}'.format(session_date, ni, regions[ni]))
        if background is not None:
            gpl.add_x_background(*background, axs[i])

    return axs


def visualize_simplex_2d(
    pts,
    ax=None,
    ax_labels=None,
    thr=0.5,
    pt_grey_col=(0.7, 0.7, 0.7),
    line_grey_col=(0.6, 0.6, 0.6),
    colors=None,
    bottom_x=0.8,
    bottom_y=-1.1,
    top_x=0.35,
    top_y=1,
    legend=False,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
        # ax.set_aspect('equal')
    if colors is None:
        colors = (None,) * pts.shape[1]
    if ax_labels is None:
        ax_labels = ("",) * pts.shape[1]
    pts_x = pts[:, 1] - pts[:, 0]
    pts_y = pts[:, 2] - (pts[:, 0] + pts[:, 1])
    ax.plot(pts_x, pts_y, "o", color=pt_grey_col, **kwargs)
    for i in range(pts.shape[1]):
        mask = pts[:, i] > thr
        ax.plot(
            pts_x[mask], pts_y[mask], "o", color=colors[i], label=ax_labels[i], **kwargs
        )
    ax.plot([-1, 0], [-1, 1], color=line_grey_col)
    ax.plot([0, 1], [1, -1], color=line_grey_col)
    ax.plot([-1, 1], [-1, -1], color=line_grey_col)
    if legend:
        ax.legend(frameon=False)
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax, 0)
    ax.text(
        bottom_x,
        bottom_y,
        ax_labels[1],
        verticalalignment="top",
        horizontalalignment="center",
    )
    ax.text(
        -bottom_x,
        bottom_y,
        ax_labels[0],
        verticalalignment="top",
        horizontalalignment="center",
    )
    ax.text(
        top_x,
        top_y,
        ax_labels[2],
        verticalalignment="top",
        horizontalalignment="center",
        rotation=-60,
    )
    return ax


def plot_posterior_predictive_dims_dict(
    models, data, dims=5, fwid=3, ks_dict=None, **kwargs
):
    f, axs = plt.subplots(len(models), dims, figsize=(dims * fwid, len(models) * fwid))
    for i, (k, v) in enumerate(models.items()):
        if ks_dict is not None:
            ks = ks_dict.get(k, None)
        else:
            ks = None
        plot_posterior_predictive_dims(v, data, dims=dims, axs=axs[i], ks=ks, **kwargs)
        axs[i, 0].set_ylabel(k)
    axs[i, -1].legend(frameon=False)


def plot_mu_hists(models, mu_key, mu2=None, axs=None, fwid=3):
    pshape = list(models.values())[0].posterior[mu_key].shape[2:]
    if axs is None:
        figsize = (fwid * pshape[1], fwid * pshape[0])
        f, axs = plt.subplots(*pshape, figsize=figsize)
    for k, m in models.items():
        mus = m.posterior[mu_key]
        for i, j in u.make_array_ind_iterator(pshape):
            mu_plot = mus[..., i, j].to_numpy().flatten()
            if mu2 is not None:
                mu2_plot = m.posterior[mu2][..., i, j].to_numpy().flatten()
                mu_plot = mu_plot - mu2_plot
            axs[i, j].hist(mu_plot, label=k, histtype="step")
            gpl.add_vlines(0, axs[i, j])
    axs[i, j].legend(frameon=False)
    return axs


def plot_posterior_predictive_dims(m, d, dims=5, axs=None, ks=None):
    if axs is None:
        f, axs = plt.subplots(5, 1)
    total_post = np.concatenate(m.posterior_predictive.err_hat, axis=0)
    total_post = np.concatenate(total_post, axis=0)
    if ks is not None:
        ks_inds = ks[1]
        d_ks = d[ks_inds]
    for i in range(dims):
        _, bins, _ = axs[i].hist(d[:, i], density=True, label="observed")
        axs[i].hist(
            total_post[:, i],
            histtype="step",
            density=True,
            label="predictive",
            linestyle="dashed",
            color="k",
            bins=bins,
        )
        if ks is not None:
            gpl.add_vlines(d_ks[:, i], axs[i])


def plot_color_means(cmeans, ax=None, dimred=3, t_ind=5, links=2):
    if ax is None:
        f = plt.figure()
        if dimred == 3:
            ax = f.add_subplot(1, 1, 1, projection="3d")
        else:
            ax = f.add_subplot(1, 1, 1)

    for i, (k, v) in enumerate(cmeans.items()):
        if i == 0:
            upper = np.zeros((len(cmeans), v.shape[1]))
            lower = np.zeros_like(upper)
        upper[i] = v[0, :, t_ind]
        lower[i] = v[1, :, t_ind]
    p = skd.PCA()
    inp_arr = np.concatenate((upper, lower), axis=0)
    neur_mask = np.logical_not(np.any(np.isnan(inp_arr), axis=0))
    inp_arr = inp_arr[:, neur_mask]
    p.fit(inp_arr)
    upper_trans = p.transform(upper[:, neur_mask])
    lower_trans = p.transform(lower[:, neur_mask])
    print(upper_trans.shape)
    ax.plot(*upper_trans[:, :dimred].T, "o")
    ax.plot(*upper_trans[:, :dimred].T)
    ax.plot(*lower_trans[:, :dimred].T, "o")
    ax.plot(*lower_trans[:, :dimred].T)
    links = np.linspace(0, len(cmeans) - 1, links, dtype=int)
    for l in links:
        up_low = np.stack((upper_trans[l, :dimred], lower_trans[l, :dimred]), axis=0)
        ax.plot(*up_low.T, color=(0.8, 0.8, 0.8))
    print(u.pr_only(p.explained_variance_ratio_))
    return ax


def _plot_mu(
    mu_format,
    trs,
    color,
    style,
    ax,
    ms=5,
    hsv_color=True,
    bckgd_color=(0.9, 0.9, 0.9),
    n_col_pts=64,
    **kwargs,
):
    plot_mu = trs(mu_format)
    plot_mu_append = np.concatenate((plot_mu, plot_mu[:, :1]), axis=1)
    if hsv_color:
        cmap = plt.get_cmap("hsv")
        n_bins = plot_mu.shape[1]
        pts = np.linspace(0, 1, n_col_pts + 1)[:-1]
        cols = pts * np.pi * 2
        col_rep = swan.spline_color(cols, n_bins)
        plot_mu_pts = np.dot(trs(mu_format), col_rep)
        pt_colors = cmap(pts)
        col = None
        ax.plot(*plot_mu_append[:3], color="k", ls=style, **kwargs)
        for i in range(plot_mu_pts.shape[1]):
            ax.plot(
                *plot_mu_pts[:3, i : i + 1],
                "o",
                color=pt_colors[i],
                ls=style,
                markersize=ms,
            )
    else:
        l = ax.plot(*plot_mu_append[:3], color=color, ls=style, **kwargs)
        col = l[0].get_color()
        ax.plot(*plot_mu_append[:3], "o", color=col, ls=style, markersize=ms)
    return ax, col


def visualize_model_collection_views(
    mdict,
    dim_red_model=skd.PCA,
    n_neighbors=16,
    dim_red=True,
    c_u=None,
    axs=None,
    fwid=3,
    kwarg_combs=None,
    mu_g_keys=(),
    use_isomap=True,
    **kwargs,
):
    if use_isomap:
        dim_red_model = skm.Isomap
        vis_kwargs = {"dim_red_model": dim_red_model, "n_neighbors": n_neighbors}
        print(vis_kwargs)
    else:
        vis_kwargs = {"dim_red_model": skd.PCA}
    if kwarg_combs is None:
        kwarg_combs = (
            {"mu_u_keys": ("mu_u",), "mu_l_keys": ("mu_l",)},
            {"mu_u_keys": ()},
            {"mu_l_keys": ()},
            {"mu_u_keys": ("mu_d_u",), "mu_l_keys": ("mu_d_l",)},
        )
    n_plots = len(kwarg_combs)
    if axs is None:
        f = plt.figure(figsize=(fwid * n_plots, fwid))
        axs = list(
            f.add_subplot(1, n_plots, i + 1, projection="3d") for i in range(n_plots)
        )
    for i, ax in enumerate(axs):
        kwarg_combs[i].update(kwargs)
        kwarg_combs[i].update(vis_kwargs)
        if i == 0:
            legend = True
        else:
            legend = False
        visualize_model_collection(
            mdict,
            dim_red=dim_red,
            c_u=c_u,
            ax=ax,
            mu_g_keys=mu_g_keys,
            legend=legend,
            **kwarg_combs[i],
        )
    return f, axs


def plot_rate_differences(
    d1, d2, param="p_err", use_type="retro", d1_p_ind=1, d2_p_ind=2, ax=None, color=None
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for k, (fit_d1, _) in d1.items():
        fit_d2, data_d2 = d2[k]
        fit_d1 = fit_d1["other"]
        fit_d2 = fit_d2["other"]
        type_ind = swaux.get_type_ind(use_type, data_d2)
        prob_d1 = np.concatenate(fit_d1.posterior[param])
        prob_d2 = np.concatenate(fit_d2.posterior[param])[:, type_ind]
        prob_d1 = 1 - prob_d1[:, d1_p_ind]
        prob_d2 = 1 - prob_d2[:, d2_p_ind]
        dim = min(prob_d1.shape[0], prob_d2.shape[0])
        diff = prob_d2[:dim] - prob_d1[:dim]
        gpl.violinplot(
            [diff], [k], color=[color], ax=ax, showmedians=True, showextrema=False
        )


def _compute_common_dimred(
    mdict,
    use_keys=(),
    convert=True,
    truncate_dim=None,
    n_cols=64,
    dim_red_model=skd.PCA,
    **kwargs,
):
    l = []
    for fit_az in mdict.values():
        for k in use_keys:
            d_ak = fit_az.posterior[k]
            if convert:
                d_ak = d_ak.to_numpy()
            m_ak = np.mean(d_ak, axis=(0, 1))
            cols = np.linspace(0, 2 * np.pi, n_cols + 1)[:-1]
            m_cols = swan.spline_color(cols, m_ak.shape[-1])
            trs_mu = np.dot(m_ak, m_cols)
            l.append(trs_mu)
    m_collection = np.concatenate(l, axis=1)
    ptrs = dim_red_model(n_components=3)
    if truncate_dim is None:
        ptrs.fit(m_collection.T)
        trs = lambda x: ptrs.transform(x.T).T
    else:
        ptrs.fit(m_collection[:truncate_dim].T)
        trs = lambda x: ptrs.transform(x[:truncate_dim].T).T
    return trs


def visualize_model_collection(
    mdict,
    dim_red=True,
    mu_u_keys=("mu_u", "mu_d_u"),
    mu_l_keys=("mu_l", "mu_d_l"),
    inter_up="i_up_type",
    inter_down="i_down_type",
    ax=None,
    common_dim_red=False,
    **kwargs,
):
    trs = None
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection="3d")
    if dim_red and common_dim_red:
        trs = _compute_common_dimred(mdict, use_keys=mu_u_keys + mu_l_keys, **kwargs)
    for k, v in mdict.items():
        out = visualize_fit_results(
            v,
            dim_red=dim_red,
            ax=ax,
            mu_u_keys=mu_u_keys,
            mu_l_keys=mu_l_keys,
            inter_up=inter_up,
            inter_down=inter_down,
            label_cl=k,
            trs=trs,
            **kwargs,
        )
        if dim_red and trs is None:
            trs = out[1]
    return ax


def get_color_reps(cols, mus, ignore_col=False, roll_ax=0, roll=False, transpose=False):
    mu_mus = np.mean(mus, axis=(0, 1))
    n_bins = mu_mus.shape[1]
    if ignore_col:
        out = np.zeros((n_bins, n_bins) + (mu_mus.shape[0],))
    else:
        out = np.zeros(cols.shape + (mu_mus.shape[0],))
    for i in range(out.shape[0]):
        if ignore_col:
            c_reps = np.identity(n_bins)
            if roll:
                c_reps = np.roll(c_reps, i, axis=roll_ax)
        else:
            col_row = cols[i]
            c_reps = swan.spline_color(col_row, mu_mus.shape[1])
        out[i] = np.dot(c_reps.T, mu_mus.T)
    if transpose:
        out = np.swapaxes(out, 0, 1)
    return out


def visualize_fit_torus(
    fit_az,
    ax=None,
    trs=None,
    eh_key="err_hat",
    dim_red=True,
    color=None,
    ms=5,
    n_cols=64,
    u_key="mu_u",
    l_key="mu_d_l",
    ignore_col=True,
    plot_surface=False,
    plot_trls=False,
    dim_red_model=skd.PCA,
    **kwargs,
):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection="3d")
    cols = np.linspace(0, 2 * np.pi, n_cols)
    col_grid_x, col_grid_y = np.meshgrid(cols, cols)
    x_reps = get_color_reps(col_grid_x, fit_az.posterior[u_key], ignore_col=ignore_col)
    y_reps = get_color_reps(
        col_grid_y, fit_az.posterior[l_key], ignore_col=ignore_col, transpose=False
    )
    tot_reps = x_reps + y_reps
    if dim_red and trs is None:
        ptrs = dim_red_model(n_components=3, **kwargs)
        con_data = np.concatenate(tot_reps, axis=0)
        ptrs.fit(con_data)
        trs = lambda x: ptrs.transform(x)
    elif trs is None:
        trs = lambda x: x
    col_r = None
    col_c = None
    all_trs = np.zeros(tot_reps.shape[:2] + (3,))
    for i in range(0, tot_reps.shape[0]):
        trs_row = trs(tot_reps[i, :])
        all_trs[i] = trs_row
        trs_row = np.concatenate((trs_row, trs_row[:1]), axis=0)
        trs_col = trs(tot_reps[:, i])
        trs_col = np.concatenate((trs_col, trs_col[:1]), axis=0)
        l = ax.plot(*trs_row.T, color=col_r)
        col_r = l[0].get_color()
        l = ax.plot(*trs_col.T, color=col_c)
        col_c = l[0].get_color()
    if plot_surface:
        all_trs = np.concatenate((all_trs, all_trs[:1]), axis=0)
        all_trs = np.concatenate((all_trs, all_trs[:, :1]), axis=1)
        ax.plot_surface(*all_trs.T)
    if plot_trls:
        pts = fit_az.posterior_predictive[eh_key]
        pts = np.median(np.concatenate(pts, axis=0), axis=0)
        pts_dr = trs(pts)
        ax.plot(*pts_dr.T, "o")
    return ax


def save_all_dists(
    loaded_data,
    p_thrs=(0, 0.2, 0.3, 0.4, 0.5),
    corr_fl="histogram",
    wheel_types=("retro", "pro"),
    new_joint=False,
    n_bins=20,
    only_keys=None,
):
    for (m, time, trl), data in loaded_data.items():
        if time == "CUE2_ON":
            types = (None,)
            mistakes = ("misbind", "guess")
            cue_time = True
        else:
            types = wheel_types
            mistakes = ("spatial", "cue", "spatial-cue", "guess")
            cue_time = False
        file_templ = "{}-" + "{}-{}-{}-".format(m, time, trl) + corr_fl + "-pthr" + "{}"
        figs_g = plot_dists(
            p_thrs,
            types,
            data,
            n_bins=n_bins,
            file_templ=file_templ,
            mistakes=mistakes,
            bin_bounds=(-1, 2),
            ret_data=False,
            new_joint=new_joint,
            cue_time=cue_time,
            only_keys=only_keys,
        )

        file_templ = "{}-" + "{}-{}-{}-less-histograms-pthr".format(m, time, trl) + "{}"
        figs_l = plot_dists(
            p_thrs,
            types,
            data,
            n_bins=n_bins,
            file_templ=file_templ,
            mistakes=mistakes,
            bin_bounds=(-1, 2),
            ret_data=False,
            p_comp=np.less,
            new_joint=new_joint,
            cue_time=cue_time,
            only_keys=only_keys,
        )
        print(figs_l)
        return figs_g, figs_l


def plot_dists(
    p_thrs,
    types,
    *args,
    fwid=3,
    mult=1.5,
    color_dict=None,
    n_bins=25,
    bin_bounds=(-1, 2),
    file_templ="{}-histograms-pthr{}",
    mistakes=("spatial", "cue"),
    ret_data=True,
    p_comp=np.greater,
    new_joint=False,
    cue_time=False,
    only_keys=None,
    axs_arr=None,
    legend=True,
    color=None,
    precomputed_data=None,
    pred_label=True,
    simple_label=False,
    legend_label="observed",
    **kwargs,
):
    figsize = (fwid * len(mistakes) * mult, fwid)
    if color_dict is None:
        spatial_color = np.array([36, 123, 160]) / 256
        hybrid_color = np.array([37, 49, 94]) / 256

        colors_dict = {"spatial": spatial_color, "hybrid": hybrid_color}

    figs = []
    out_data = {}
    for i, j in u.make_array_ind_iterator((len(p_thrs), len(types))):
        if axs_arr is None:
            f, axs = plt.subplots(
                1, len(mistakes), figsize=figsize, sharey=True, squeeze=False
            )
            axs = axs[0]
        else:
            axs = axs_arr[i, j]
            f = None
        trl_type = types[j]
        p_thr = p_thrs[i]

        for k, mistake in enumerate(mistakes):
            new_key = (mistake, trl_type, p_thr)
            if precomputed_data is not None:
                use_data = precomputed_data[new_key]
            else:
                use_data = None
            if k > 0:
                legend = False
            _, w_dat = plot_session_swap_distr_collection(
                *args,
                n_bins=n_bins,
                p_thresh=p_thr,
                colors=color_dict,
                bin_bounds=bin_bounds,
                axs=axs[k : k + 1],
                trl_filt=trl_type,
                mistake=mistake,
                p_comp=p_comp,
                new_joint=new_joint,
                cue_time=cue_time,
                only_keys=only_keys,
                legend=legend,
                color=color,
                precomputed_data=use_data,
                pred_label=pred_label,
                legend_label=legend_label,
            )

            out_data[new_key] = w_dat
            gpl.clean_plot(axs[k], k)
            if k == 0:
                if simple_label:
                    axs[k].set_ylabel("density")
                else:
                    l = r"density | $p_{swp} > " + "{}$".format(p_thr)
                    axs[k].set_ylabel(l)
            else:
                axs[k].set_ylabel("")
            axs[k].set_xlabel("prototype distance (au)")
            axs[k].set_xticks([0, 1])
            m_comps = mistake.split("-")
            if len(m_comps) > 1:
                l1, l2 = m_comps
            else:
                l1 = "correct"
                l2 = mistake
            axs[k].set_xticklabels([l1, l2], rotation=45)

        if f is not None:
            f.savefig(
                file_templ.format(trl_type, p_thr) + ".svg",
                bbox_inches="tight",
                transparent=True,
            )
            f.savefig(
                file_templ.format(trl_type, p_thr) + ".pdf",
                bbox_inches="tight",
                transparent=True,
            )
            figs.append(f)
    if ret_data:
        out = (figs, out_data)
    else:
        out = figs
    return out


def plot_sig_comparison(emp_func, bins, theor_func, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    l = ax.plot(bins, emp_func.T)
    _ = ax.plot(bins, theor_func.T, linestyle='dashed', color=l[0].get_color())
    return ax


def plot_sweep(
    sigmas,
    dps,
    kls,
    theor_params=None,
    ax=None,
    f=None,
    cmap="Greys",
    fit_color=(.6,)*3,
    opt_color="w",
    ms=5,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    m = gpl.pcolormesh(dps, sigmas, np.log(kls), ax=ax, cmap=cmap)
    f.colorbar(m)
    ax.set_ylabel('width')
    ax.set_xlabel("d'")
    xt = ax.get_xticks()
    ax.set_xticks(xt[::5])
    yt = ax.get_yticks()
    ax.set_yticks(yt[::5])
    ax.set_yticks([.1, 2, 4])
    ax.set_xticks([.2, 1, 2])
    if theor_params is not None:
        ax.plot(theor_params[1], theor_params[0], color=fit_color, marker='o', ms=ms)
    sig_inds, dp_inds = np.where(kls == np.min(kls))
    ax.plot(dps[dp_inds], sigmas[sig_inds], color=opt_color, marker='*', ms=ms)
    return ax


def plot_kl_comp_distrib(sigmas, dps, kls, errs, **kwargs):
    sig_ind, dp_ind = np.where(np.min(kls) == kls)
    sig_opt = sigmas[sig_ind]
    dp_opt = dps[dp_ind]
    labels = ("empirical", "predicted")
    return plot_comparison_distrib(sig_opt, dp_opt, errs, labels=labels, **kwargs)


def plot_comparison_distrib(
    sig_opt,
    dp_opt,
    errs,
    ax=None,
    labels=None,
    fit_color="k",
    hist_color=np.array([52, 152, 219])/255,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if labels is None:
        labels = ("empirical", "average width")
    err_theor = swan.simulate_emp_err(sig_opt, dp_opt)
    ax.hist(errs, density=True, label=labels[0], color=hist_color)
    ax.hist(
        err_theor,
        histtype="step",
        density=True,
        label=labels[1],
        color=fit_color,
        linestyle="dashed",
    )
    ax.legend(frameon=False)
    return ax


def plot_proj_p_heat(
    td, p, ax=None, p_n_bins=5, td_n_bins=10, bounds=(-1, 2), normalize_cols=True
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    comb_data = np.stack((td, p), axis=1)
    p_bins = np.linspace(0, np.max(p), p_n_bins + 1)
    td_bins = np.linspace(*bounds, td_n_bins + 1)
    out = np.histogramdd(comb_data, (td_bins, p_bins))
    hm, _ = out
    if normalize_cols:
        hm = hm / np.sum(hm, axis=0, keepdims=True)
    ax.pcolormesh(td_bins, p_bins, hm.T)


def plot_proj_p_scatter(
    td, p, ax=None, bounds=(-1, 2), color=None, eps=1e-5, n_bins=5, cent_func=np.mean
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ax.plot(td, p, "o", ms=1.5)
    gpl.add_vlines(0, ax)
    gpl.add_vlines(1, ax)

    members = np.digitize(p, np.linspace(0, np.max(p) + eps, n_bins + 1))

    for mem in np.unique(members):
        mask = mem == members
        t_pt = cent_func(td[mask])
        p_pt = cent_func(p[mask])
        l = ax.plot(t_pt, p_pt, "o", color=color)
        color = l[0].get_color()

    if bounds is not None:
        ax.set_xlim(bounds)
    return ax


def plot_cue_tc(*args, **kwargs):
    kwargs.pop("mat_inds", None)
    return plot_lm_tc(*args, use_mat_inds=False, plot_markers=False, cent=0, **kwargs)


default_label_dict = {
    "color": "samples",
    "pre-color": "samples",
    "post-color": "samples",
    "pre-cue": "cue",
    "post-cue": "cue",
}


def plot_lm_tc(
    out_dict,
    mat_ind=(0, 1),
    axs=None,
    fwid=3,
    null_color="green",
    swap_color="red",
    mat_inds=None,
    use_mat_inds=True,
    plot_markers=True,
    cent=0.5,
    key_order=None,
    null_colors=None,
    swap_colors=None,
    label_dict=default_label_dict,
):
    if key_order is None:
        key_order = out_dict.keys()
    if axs is None:
        f, axs = plt.subplots(
            1,
            len(out_dict),
            figsize=(fwid * len(out_dict), fwid),
            sharey=True,
            squeeze=False,
        )
    if mat_inds is None:
        mat_inds = (mat_ind,) * len(out_dict)
    if null_colors is None:
        null_colors = (null_color,) * len(out_dict)
    if swap_colors is None:
        swap_colors = (swap_color,) * len(out_dict)
    for i, k in enumerate(key_order):
        out = out_dict[k]
        (nc_comb, sc_comb), (nc_indiv, sc_indiv), xs = out
        if nc_comb.shape[0] > 0:
            nc_plot = list(
                np.squeeze(np.mean(nc_indiv_i, axis=0)) for nc_indiv_i in nc_indiv
            )
            if use_mat_inds:
                nc_plot = np.stack(nc_plot, axis=2)
                nc_plot = nc_plot[mat_inds[i]]
                sc_comb = sc_comb[mat_inds[i]]
            else:
                nc_plot = np.stack(nc_plot, axis=0)
            gpl.plot_trace_werr(
                xs,
                nc_plot,
                ax=axs[0, i],
                color=null_colors[i],
                central_tendency=np.nanmedian,
                error_func=gpl.std,
            )
            gpl.plot_trace_werr(
                xs,
                sc_comb,
                ax=axs[0, i],
                central_tendency=np.nanmedian,
                color=swap_colors[i],
            )

        if plot_markers:
            gpl.add_hlines(0, axs[0, i], color=null_colors[i], plot_outline=True)
            gpl.add_hlines(1, axs[0, i], color=swap_colors[i], plot_outline=True)
        gpl.add_hlines(cent, axs[0, i], linestyle="dashed")
        gpl.clean_plot(axs[0, i], i)

        axs[0, i].set_xlabel("time from {}".format(label_dict.get(k, k)))
    axs[0, 0].set_ylabel("normalized\nprojection")
    return axs


default_error_labels = ("misbinding", "color\nselection", "color\nselection")


def plot_lm_hists(
    out_dict,
    mat_ind=(0, 1),
    x_pts=(0.25, 0, -0.25),
    axs=None,
    fwid=3,
    null_color="green",
    swap_color="red",
    eps=1e-5,
    n_bins=20,
    bin_bounds=(-1, 2),
    error_labels=default_error_labels,
    mat_inds=None,
    key_order=None,
):
    if key_order is None:
        key_order = out_dict.keys()
    if axs is None:
        f, axs = plt.subplots(
            2, len(out_dict), figsize=(fwid * len(out_dict), 2 * fwid), sharex="all"
        )
    if mat_inds is None:
        mat_inds = (mat_ind,) * len(out_dict)

    bins = np.linspace(*bin_bounds, n_bins + 1)
    for i, k in enumerate(key_order):
        out = out_dict[k]
        (nc_comb, sc_comb), (nc_indiv, sc_indiv), xs = out
        t_diffs = np.abs(xs - x_pts[i])
        time_ind = np.argmin(t_diffs)
        if np.min(t_diffs) > eps:
            s = (
                "time difference not exactly zero, closest is time point {} "
                "for target {}"
            )
            print(s.format(xs[time_ind], x_pts[i]))
        nc_plot = nc_comb[mat_inds[i]][:, time_ind]
        sc_plot = sc_comb[mat_inds[i]][:, time_ind]
        axs[0, i].hist(nc_plot, bins=bins, color=null_color, density=True)
        axs[1, i].hist(sc_plot, bins=bins, color=swap_color, density=True)
        gpl.add_vlines(0, axs[0, i], color=null_color, plot_outline=True)
        gpl.add_vlines(1, axs[0, i], color=swap_color, plot_outline=True)
        gpl.add_vlines(0, axs[1, i], color=null_color, plot_outline=True)
        gpl.add_vlines(1, axs[1, i], color=swap_color, plot_outline=True)
        gpl.clean_plot_bottom(axs[0, i])
        axs[1, i].set_xticks([0, 1])
        gpl.clean_plot(axs[0, i], 0)
        gpl.clean_plot(axs[1, i], 0)
        axs[1, i].set_xticklabels(["correct", error_labels[i]], rotation=45)
        gpl.clean_plot_bottom(axs[1, i], keeplabels=True)
        axs[1, i].invert_yaxis()
        gpl.make_yaxis_scale_bar(
            axs[0, i], magnitude=0.2, double=False, label="density"
        )
        gpl.make_yaxis_scale_bar(
            axs[1, i], magnitude=0.2, double=False, label="density"
        )
        axs[0, i].set_title(k)


def plot_session_swap_distr_collection(
    session_dict,
    axs=None,
    n_bins=20,
    fwid=3,
    default_p_ind=1,
    bin_bounds=None,
    p_ind=None,
    ret_data=True,
    colors=None,
    mistake="spatial",
    new_joint=False,
    cue_time=False,
    only_keys=None,
    legend=True,
    color=None,
    precomputed_data=None,
    legend_label="observed",
    pred_label=True,
    **kwargs,
):
    if pred_label:
        pred_label = "predicted"
    else:
        pred_label = ""
    if colors is None:
        colors = {}
    if axs is None:
        n_plots = len(list(session_dict.values())[0][0])
        fsize = (fwid * n_plots, fwid)
        f, axs = plt.subplots(1, n_plots, figsize=fsize, sharex=False, sharey=False)
        if n_plots == 1:
            axs = [axs]
    true_d = {}
    pred_d = {}
    ps_d = {}
    if mistake == "spatial":
        cent_keys = dict(
            cent1_keys=(
                (("mu_d_u", "mu_l"), "i_down_type"),
                (("mu_u", "mu_d_l"), "i_up_type"),
            ),
            cent2_keys=(
                (("mu_l", "mu_d_u"), "i_down_type"),
                (("mu_d_l", "mu_u"), "i_up_type"),
            ),
        )
    elif mistake == "cue":
        ## IS THIS RIGHT? or is something weird with intercepts?
        ## CHECK THIS CODE (again, sheesh)
        cent_keys = dict(
            cent1_keys=(
                (("mu_d_u", "mu_l"), "i_down_type"),
                (("mu_u", "mu_d_l"), "i_up_type"),
            ),
            cent2_keys=(
                (("mu_u", "mu_d_l"), "i_up_type"),
                (("mu_d_u", "mu_l"), "i_down_type"),
            ),
        )
    elif mistake == "spatial-cue":
        cent_keys = dict(
            cent1_keys=(
                (("mu_l", "mu_d_u"), "i_down_type"),
                (("mu_d_l", "mu_u"), "i_up_type"),
            ),
            cent2_keys=(
                (("mu_u", "mu_d_l"), "i_up_type"),
                (("mu_d_u", "mu_l"), "i_down_type"),
            ),
        )
    elif mistake == "guess" and not cue_time:
        cent_keys = dict(
            cent1_keys=(
                (("mu_d_u", "mu_l"), "i_down_type"),
                (("mu_u", "mu_d_l"), "i_up_type"),
            ),
            cent2_keys=(
                (("mu_d_u", "mu_l"), "i_down_type"),
                (("mu_u", "mu_d_l"), "i_up_type"),
            ),
        )
        default_p_ind = 2
        cent_keys["use_resp_color"] = True
    elif mistake == "guess" and cue_time:
        cent_keys = dict(
            cent1_keys=((("mu_u", "mu_l"), None), (("mu_u", "mu_l"), None)),
            cent2_keys=((("mu_u", "mu_l"), None), (("mu_u", "mu_l"), None)),
        )
        cent_keys["use_cues"] = True
        default_p_ind = 2
        cent_keys["use_resp_color"] = True
    elif mistake == "misbind":
        cent_keys = dict(
            cent1_keys=((("mu_u", "mu_l"), None),),
            cent2_keys=((("mu_l", "mu_u"), None),),
        )
        cent_keys["use_cues"] = False
    else:
        raise IOError("unrecognized mistake type")
    if p_ind is None:
        p_ind = default_p_ind
    cent_keys.update(kwargs)
    if precomputed_data is None:
        for sn, (mdict, data) in session_dict.items():
            if only_keys is None or sn in only_keys:
                for k, faz in mdict.items():
                    out = swan.get_normalized_centroid_distance(
                        faz, data, p_ind=p_ind, new_joint=new_joint, **cent_keys
                    )
                    true, pred, ps = out
                    true_k = true_d.get(k, [])
                    true_k.append(true)
                    true_d[k] = true_k
                    pred_k = pred_d.get(k, [])
                    pred_k.append(pred)
                    pred_d[k] = pred_k
                    ps_k = ps_d.get(k, [])
                    ps_k.append(ps[:, p_ind])
                    ps_d[k] = ps_k
    else:
        true_d = precomputed_data
    if bin_bounds is not None:
        bins = np.linspace(*bin_bounds, n_bins)
    else:
        bins = n_bins
    out_data = {}
    for i, (k, td) in enumerate(true_d.items()):
        if precomputed_data is None:
            td_full = np.concatenate(td, axis=0)
            pd_full = np.concatenate(pred_d[k], axis=0)
            ps_full = np.concatenate(ps_d[k], axis=0)
        else:
            td_full, pd_full, ps_full = precomputed_data[k]
        use_color = colors.get(k)
        if use_color is None:
            use_color = color
        _, bins, _ = axs[i].hist(
            td_full, bins=bins, color=use_color, density=True, label=legend_label
        )
        axs[i].hist(
            pd_full,
            bins=bins,
            histtype="step",
            color="k",
            linestyle="dashed",
            density=True,
            label=pred_label,
        )
        gpl.add_vlines([0, 1], axs[i])
        axs[i].set_ylabel(k)
        if ret_data:
            out_data[k] = (td_full, pd_full, ps_full)
    if legend and i == 0:
        axs[i].legend(frameon=False)
    if ret_data:
        out = axs, out_data
    else:
        out = axs
    return out


def plot_color_swap_bias(
    data,
    targ="swap_prob",
    colors=("upper_color", "lower_color"),
    n_bins=10,
    axs=None,
    label="",
    fwid=3,
):
    if axs is None:
        f, axs = plt.subplots(
            1, len(colors), figsize=(fwid * len(colors), fwid), sharey=True
        )
    for i, color in enumerate(colors):
        cols = np.concatenate(data[color])
        prob = np.concatenate(data[targ])
        col_mus, (prob_mus,) = u.discretize_group(cols, prob, n=1000)

        col_mus = np.array(col_mus)
        prob_mus = np.array(prob_mus)
        gpl.plot_trace_werr(
            np.mean(col_mus, axis=1), prob_mus.T, ax=axs[i], conf95=True, label=label
        )
    return axs


def _c1_scatter(arr, ax, **kwargs):
    assert np.mean(arr[1]) > 0.99
    ax.plot(arr[0][:, 0, 0], arr[0][:, 0, 1], "o", **kwargs)


def _c1_d_hist(arr, ax, n_bins=21, **kwargs):
    assert np.mean(arr[1]) > 0.99
    bins = np.linspace(-np.pi, np.pi, n_bins)
    ax.hist(arr[0][:, 0, 1], bins=bins, **kwargs)


def _c2_scatter(arr, ax, **kwargs):
    assert np.mean(arr[1]) > 0.99
    ax.plot(arr[0][:, 0, 1], arr[0][:, 0, 0], "o", **kwargs)


def _c2_d_hist(arr, ax, n_bins=21, **kwargs):
    assert np.mean(arr[1]) > 0.99
    bins = np.linspace(-np.pi, np.pi, n_bins)
    ax.hist(arr[0][:, 0, 0], bins=bins, **kwargs)


def plot_circus_sweep_resps(
    sweep_dict, ax_params, pk_type="scatter", plot_keys=None, axs=None, fwid=1, **kwargs
):
    if pk_type == "scatter" and plot_keys is None:
        plot_keys = (("cue1", _c1_scatter), ("cue2", _c2_scatter))
    elif pk_type == "dist_hist" and plot_keys is None:
        plot_keys = (("cue1", _c1_d_hist), ("cue2", _c2_d_hist))
    elif plot_keys is None:
        raise IOError("need to supply known pk_type or plot_keys")
    conjunctions = []
    n_vals = []
    assert len(ax_params) == 2
    for ap in ax_params:
        u_vals = np.unique(sweep_dict[ap])
        conjunctions.append(u_vals)
        n_vals.append(len(u_vals))
    if axs is None:
        f, axs = plt.subplots(
            *n_vals,
            figsize=(fwid * n_vals[1], fwid * n_vals[0]),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
    for i, j in it.product(*list(range(nv) for nv in n_vals)):
        mask = np.logical_and(
            sweep_dict[ax_params[0]] == conjunctions[0][i],
            sweep_dict[ax_params[1]] == conjunctions[1][j],
        )
        inds = np.where(mask)[0]
        for ind in inds:
            for pk, func in plot_keys:
                func(sweep_dict[pk][ind], axs[i, j], **kwargs)
        if j == 0:
            axs[i, j].set_ylabel(np.round(conjunctions[0][i], 2))
        if i == n_vals[0] - 1:
            axs[i, j].set_xlabel(np.round(conjunctions[1][j], 2))
    # if i == n_vals[0] - 1 and j == 0:
    #     f.text
    #     axs[i, j].set_xlabel(ax_params[1])
    #     axs[i, j].set_ylabel(ax_params[0])
    return axs


def plot_swap_distr_collection(model_dict, data, axs=None, fwid=3, **kwargs):
    if axs is None:
        n_plots = len(model_dict)
        fsize = (fwid * n_plots, fwid)
        f, axs = plt.subplots(1, n_plots, figsize=fsize, sharex=True, sharey=True)
    for i, (k, faz) in enumerate(model_dict.items()):
        plot_swap_distr(faz, data, ax=axs[i], **kwargs)
        axs[i].set_ylabel(k)
    return axs


def plot_swap_distr(fit_az, data, ax=None, n_bins=10, **kwargs):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    out = swan.get_normalized_centroid_distance(fit_az, data, **kwargs)
    true_arr_full, pred_arr_full, _ = out
    _, bins, _ = ax.hist(true_arr_full, histtype="step", density=True, bins=n_bins)
    ax.hist(pred_arr_full, histtype="step", density=True, bins=bins)
    gpl.add_vlines([0, 1], ax)
    return ax


class MultiSessionFit:
    def __init__(self, fit_list):
        self.posterior = {}
        for k in fit_list[0].posterior.keys():
            k_list = []
            for fl in fit_list:
                add_arr = fl.posterior[k].to_numpy()
                if len(add_arr.shape) < 3:
                    add_arr = np.expand_dims(add_arr, 2)
                k_list.append(add_arr)
            self.posterior[k] = np.concatenate(k_list, axis=2)


def make_multi_session_fit(sdict):
    post_lists = {}
    for session_id, (m, _) in sdict.items():
        for mk, fit_az in m.items():
            mk_list = post_lists.get(mk, [])
            mk_list.append(fit_az)
            post_lists[mk] = mk_list
    msf = {}
    for k, pl in post_lists.items():
        msf[k] = MultiSessionFit(pl)
    return msf


def visualize_fit_results(
    fit_az,
    mu_u_keys=("mu_u", "mu_d_u"),
    mu_l_keys=("mu_l", "mu_d_l"),
    dim_red=True,
    c_u=(1, 0, 0),
    c_l=(0, 1, 0),
    c_g=(0, 0, 1),
    mu_g_keys=("mu_g",),
    trs=None,
    inter_up="intercept_up",
    inter_down="intercept_down",
    styles=("solid", "dashed"),
    ax=None,
    label_cu="",
    label_cl="",
    same_color=True,
    legend=True,
    truncate_dim=None,
    n_cols=64,
    dim_red_model=skd.PCA,
    convert=False,
    **kwargs,
):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection="3d")
    all_keys = mu_u_keys + mu_l_keys
    if dim_red and trs is None:
        l = []
        for i, ak in enumerate(all_keys):
            d_ak = fit_az.posterior[ak]
            if convert:
                d_ak = d_ak.to_numpy()
            m_ak = np.mean(d_ak, axis=(0, 1))
            cols = np.linspace(0, 2 * np.pi, n_cols + 1)[:-1]
            m_cols = swan.spline_color(cols, m_ak.shape[-1])
            trs_mu = np.dot(m_ak, m_cols)
            l.append(trs_mu)
        m_collection = np.concatenate(l, axis=1)
        ptrs = dim_red_model(n_components=3, **kwargs)
        if truncate_dim is None:
            ptrs.fit(m_collection.T)
            trs = lambda x: ptrs.transform(x.T).T
        else:
            ptrs.fit(m_collection[:truncate_dim].T)
            trs = lambda x: ptrs.transform(x[:truncate_dim].T).T
    elif trs is None:
        trs = lambda x: x
    for i, mu_k in enumerate(mu_u_keys):
        mu_ak = fit_az.posterior[mu_k]
        if convert:
            mu_ak = mu_ak.to_numpy()
        mu_format = np.mean(mu_ak, axis=(0, 1))
        if "_d_" in mu_k:
            inter_format = np.expand_dims(fit_az.posterior[inter_down], 3)
        else:
            inter_format = np.expand_dims(fit_az.posterior[inter_up], 3)
        mu_format = mu_format + np.mean(inter_format, axis=(0, 1))

        _, c_u = _plot_mu(mu_format, trs, c_u, styles[i], ax, label=label_cu)
    if same_color:
        c_l, c_g = c_u, c_u
    for i, mu_k in enumerate(mu_l_keys):
        mu_ak = fit_az.posterior[mu_k]
        if convert:
            mu_ak = mu_ak.to_numpy()
        mu_format = np.mean(mu_ak, axis=(0, 1))
        if "_d_" in mu_k:
            inter_format = np.expand_dims(fit_az.posterior[inter_up], 3)
        else:
            inter_format = np.expand_dims(fit_az.posterior[inter_down], 3)
        mu_format = mu_format + np.mean(inter_format, axis=(0, 1))

        _, c_l = _plot_mu(mu_format, trs, c_l, styles[i], ax, label=label_cl)
    for i, mu_k in enumerate(mu_g_keys):
        mu_ak = fit_az.posterior[mu_k]
        if convert:
            mu_ak = mu_ak.to_numpy()
        mu_format = np.mean(mu_ak, axis=(0, 1))
        mu_format = np.expand_dims(mu_format, 1)
        _, c_g = _plot_mu(mu_format, trs, c_g, styles[i], ax)
    if legend:
        ax.legend(frameon=False)
    return ax, trs


def plot_k_trials(
    fit_az,
    k_thresh=0.7,
    dim_red=None,
    ax=None,
    k_color=(1, 0, 0),
    reg_color=(0.5, 0.5, 0.5),
    plot_nk=False,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    x = az.loo(fit_az, pointwise=True)
    k_vals = x["pareto_k"]
    k_mask = k_vals > k_thresh
    k_trls = fit_az.observed_data.y[k_mask]
    nk_trls = fit_az.observed_data.y[np.logical_not(k_mask)]
    if dim_red is not None:
        k_trls = dim_red(k_trls.T)
        nk_trls = dim_red(nk_trls.T)
    ax.plot(*k_trls, "o", color=k_color)
    if plot_nk:
        ax.plot(*nk_trls, "o", color=reg_color)


def plot_k_distributions_dict(inp_dict, **kwargs):
    ks = list(inp_dict.keys())
    vs = list(inp_dict[k] for k in ks)
    return plot_k_distributions(vs, ks, **kwargs)


def plot_k_correlation_dict(ks_dict, data, data_label="", fwid=3):
    f, axs = plt.subplots(1, len(ks_dict), figsize=(fwid * len(ks_dict), fwid))
    for i, (k, k_vals) in enumerate(ks_dict.items()):
        plot_k_correlation(k_vals[0], data, ax=axs[i])
        axs[i].set_xlabel(k)
        axs[i].set_ylabel(data_label)
    return f, axs


def plot_k_correlation(k_vals, data, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ax.plot(k_vals, data, "o")


def plot_k_distributions(
    models, labels, k_thresh=0.7, fwid=3, sharex=True, sharey=True, compute_k=True
):
    if compute_k:
        models = list(swan.get_pareto_k(m)[0] for m in models)
    combs = list(it.combinations(range(len(models)), 2))
    side_plots = len(models) - 1
    f, axs = plt.subplots(
        side_plots,
        side_plots,
        figsize=(fwid * side_plots, fwid * side_plots),
        sharex=sharex,
        sharey=sharey,
    )
    for i, (ind1, ind2) in enumerate(combs):
        ax = axs[ind1, ind2 - 1]
        ax.plot(models[ind1], models[ind2], "o")
        gpl.add_hlines(k_thresh, ax)
        gpl.add_vlines(k_thresh, ax)
        ax.set_xlabel(labels[ind1])
        ax.set_ylabel(labels[ind2])
    return f, axs


def make_color_circle(ax=None, px=1000, r_cent=350, r_wid=100):
    if ax is None:
        f, ax = plt.subplots(1, 1)

    surface = np.zeros((px, px))
    x, y = np.meshgrid(np.arange(px), np.arange(px))
    full = np.stack((x, y), axis=2)
    cent = full - px / 2
    norm = cent / np.sqrt(np.sum(cent**2, axis=2, keepdims=True))
    vec = np.expand_dims([0, 1], axis=(0, 1))
    dist = np.sqrt(np.sum((norm - vec) ** 2, axis=2))
    sim = np.arcsin(dist / 2)
    sim[x < px / 2] = -sim[x < px / 2]
    sim = sim - np.nanmin(sim)
    sim = sim / np.nanmax(sim)

    r = np.sqrt(np.sum(cent**2, axis=2))
    mask = np.logical_or(r < r_cent - r_wid / 2, r > r_cent + r_wid / 2)
    sim[mask] = np.nan
    ax.imshow(sim, cmap="hsv")
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax, 0)


def plot_error_swap_distribs(
    data,
    err_field="err",
    dist_field="LABthetaDist",
    resp_field="LABthetaResp",
    **kwargs,
):
    errs = np.concatenate(data[err_field])
    dist_errs = np.concatenate(data[dist_field] - data[resp_field])
    dist_errs = u.normalize_periodic_range(dist_errs)
    return plot_error_swap_distribs_err(errs, dist_errs, **kwargs)


def plot_error_swap_distribs_err(
    errs,
    dist_errs,
    axs=None,
    fwid=3,
    label="",
    model_data=None,
    color=None,
    model_derr=None,
):
    if axs is None:
        fsize = (2 * fwid, fwid)
        f, axs = plt.subplots(1, 2, figsize=fsize, sharey=True, sharex=True)
    l = axs[0].hist(errs, density=True, color=color)
    if model_data is not None:
        axs[0].hist(
            model_data.flatten(),
            histtype="step",
            density=True,
            color="k",
            linestyle="dashed",
        )
    axs[1].hist(dist_errs, label=label, density=True, color=color)
    if model_derr is not None:
        m_derr = u.normalize_periodic_range(model_derr - model_data)
        axs[1].hist(
            m_derr.flatten(),
            histtype="step",
            density=True,
            color="k",
            linestyle="dashed",
        )
    axs[1].legend(frameon=False)
    axs[0].set_xlabel("error (rads)")
    axs[0].set_ylabel("density")
    axs[1].set_xlabel("distractor distance (rads)")
    gpl.clean_plot(axs[0], 0)
    gpl.clean_plot(axs[1], 1)
    return axs


def _plot_di(di, t_ind, xs, axs, n_boots=1000, buff=0.01, avg_models=False):
    x_pts = di[:, 0, 0, t_ind]
    y_pts = di[:, 0, 1, t_ind]
    diffs = di[:, :, 1] - di[:, :, 0]
    if avg_models:
        diffs = np.mean(diffs, axis=0)
    else:
        diffs = diffs[:, 0]
    diffs_b = u.bootstrap_list(
        diffs, u.mean_axis0, n=n_boots, out_shape=(diffs.shape[1],)
    )
    gpl.plot_trace_werr(xs, diffs_b, ax=axs[1], conf95=True)
    axs[0].plot(x_pts, y_pts, "o")
    bound_low = min(np.min(x_pts), np.min(y_pts)) - buff
    bound_high = max(np.max(x_pts), np.max(y_pts)) + buff
    axs[0].plot([bound_low, bound_high], [bound_low, bound_high])
    return axs


def plot_distance_correlation(
    dists, t_ind, xs=None, axs=None, fwid=2, n_boots=1000, test_dists=None
):
    if xs is None:
        xs = np.arange(dists[0].shape[-1])
    if axs is None:
        fsize = (2 * fwid, fwid * len(dists))
        f, axs = plt.subplots(len(dists), 2, figsize=fsize, squeeze=False)
    for pop_ind, di in enumerate(dists):
        _plot_di(di, t_ind, xs, axs[pop_ind], n_boots=n_boots)
        if test_dists is not None:
            if test_dists[pop_ind].shape[1] > 0:
                _plot_di(
                    test_dists[pop_ind],
                    t_ind,
                    xs,
                    axs[pop_ind],
                    n_boots=n_boots,
                    avg_models=True,
                )
        gpl.add_hlines(0, axs[pop_ind, 1])
    return axs


def plot_embedding_color(emb_pop, colors, ax=None, cm="twilight"):
    if emb_pop.shape[-1] == 1:
        symbol = "o"
    else:
        symbol = "-"
    cmap = plt.get_cmap(cm)
    colors_norm = colors / (2 * np.pi)
    color_rotation = cmap(colors_norm)
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for i, trl in enumerate(emb_pop):
        ax.plot(trl[0], trl[1], symbol, color=color_rotation[i])
    return ax


def plot_barcode(
    h0, h1, h2, col_list=("r", "g", "m", "c"), plot_percent=(99, 98, 90), large_num=100
):
    # replace the infinity bar (-1) in H0 by a really large number
    h0[np.logical_not(np.isfinite(h0))] = large_num
    # Plot the longest barcodes only
    to_plot = []
    x_lim = 0
    for curr_h, cutoff in zip([h0, h1, h2], plot_percent):
        bar_lens = curr_h[:, 1] - curr_h[:, 0]
        plot_h = curr_h[bar_lens > np.percentile(bar_lens, cutoff)]
        x_lim = max(np.max(bar_lens), x_lim)
        to_plot.append(plot_h)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            ax.plot(
                [interval[0], interval[1]], [i, i], color=col_list[curr_betti], lw=1.5
            )
        ax.set_xlim([0, x_lim])
        ax.set_xticks([0, x_lim])
        ax.set_ylim([-1, len(curr_bar)])
        ax.set_yticks([])
    return ax


def plot_single_neuron_color(
    neur_dict, xs, plot_x, filter_keys=None, axs=None, n_shuffs=100
):
    x_ind = np.argmin(np.abs(xs - plot_x))
    mi_cs = []
    for i, (k, v) in enumerate(neur_dict.items()):
        if filter_keys is not None and k[-1] in filter_keys:
            if axs is None:
                f, ax = plt.subplots(1, 1)
            else:
                ax = axs[i]
            edges = np.linspace(0, 2 * np.pi, 9)
            bin_ids = np.digitize(v[1], edges)
            jag_tr = []
            jag_tr_shuff = []
            unique_ids = np.unique(bin_ids)
            for i, bi in enumerate(unique_ids):
                jag_tr.append(v[0][bi == bin_ids])
            mi_null = np.zeros(n_shuffs)
            for j in range(n_shuffs):
                bin_ids_shuff = np.random.choice(bin_ids, len(bin_ids), replace=False)
                for i, bi in enumerate(unique_ids):
                    jag_tr_shuff.append(v[0][bi == bin_ids_shuff])
                mi_null[j] = color_index(jag_tr_shuff)

            gpl.plot_trace_werr(unique_ids, jag_tr, ax=ax, jagged=True)
            mi_c = color_index(jag_tr)
            print(mi_null)
            print("m null", np.nanmean(mi_null))
            print("s null", np.nanstd(mi_null))
            print("unnorm", mi_c)
            mi_c = (mi_c - np.nanmean(mi_null)) / np.nanstd(mi_null)
            print("final", mi_c)
            mi_cs.append(mi_c)
            ax.set_title("MI: {}; {}".format(mi_c, k[-1]))
    return np.array(mi_cs)


def color_index(frs):
    n = len(frs)
    ct = np.array(list(np.nanmean(fr) for fr in frs))
    ct_norm = ct / np.nansum(ct)
    mi_color = np.nansum(ct_norm * np.log(n * ct_norm)) / np.log(n)
    return mi_color
