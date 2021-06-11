
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import general.plotting as gpl
import general.utility as u

def _plot_di(di, t_ind, xs, axs, n_boots=1000, buff=.01, avg_models=False):
    x_pts = di[:, 0, 0, t_ind]
    y_pts = di[:, 0, 1, t_ind]
    diffs = di[:, :, 1] - di[:, :, 0]
    if avg_models:
        diffs = np.mean(diffs, axis=0)
    else:
        diffs = diffs[:, 0]
    diffs_b = u.bootstrap_list(diffs, u.mean_axis0, n=n_boots,
                               out_shape=(diffs.shape[1],))
    gpl.plot_trace_werr(xs, diffs_b, ax=axs[1], conf95=True)
    axs[0].plot(x_pts, y_pts, 'o')
    bound_low = min(np.min(x_pts), np.min(y_pts)) - buff
    bound_high = max(np.max(x_pts), np.max(y_pts)) + buff
    axs[0].plot([bound_low, bound_high], [bound_low, bound_high])
    return axs

def plot_distance_correlation(dists, t_ind, xs=None, axs=None, fwid=2,
                              n_boots=1000, test_dists=None):
    if xs is None:
        xs = np.arange(dists[0].shape[-1])
    if axs is None:
        fsize = (2*fwid, fwid*len(dists))
        f, axs = plt.subplots(len(dists), 2, figsize=fsize, squeeze=False)
    for pop_ind, di in enumerate(dists):
        _plot_di(di, t_ind, xs, axs[pop_ind], n_boots=n_boots)
        if test_dists is not None:
            _plot_di(test_dists[pop_ind], t_ind, xs, axs[pop_ind],
                     n_boots=n_boots, avg_models=True)
        gpl.add_hlines(0, axs[pop_ind, 1])
    return axs

def plot_embedding_color(emb_pop, colors, ax=None, cm='twilight'):
    if emb_pop.shape[-1] == 1:
        symbol = 'o'
    else:
        symbol = '-'
    cmap = plt.get_cmap(cm)
    colors_norm = colors/(2*np.pi)
    color_rotation = cmap(colors_norm)
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for i, trl in enumerate(emb_pop):
        ax.plot(trl[0], trl[1], symbol, color=color_rotation[i])
    return ax

def plot_barcode(h0, h1, h2, col_list=('r', 'g', 'm', 'c'),
                 plot_percent=(99, 98, 90), large_num=100):
    # replace the infinity bar (-1) in H0 by a really large number
    h0[np.logical_not(np.isfinite(h0))] = large_num
    # Plot the longest barcodes only
    to_plot = []
    x_lim = 0
    for curr_h, cutoff in zip([h0, h1, h2], plot_percent):
        bar_lens = curr_h[:,1] - curr_h[:,0]
        plot_h = curr_h[bar_lens > np.percentile(bar_lens, cutoff)]
        x_lim = max(np.max(bar_lens), x_lim)
        to_plot.append(plot_h)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
                lw=1.5)
        ax.set_xlim([0, x_lim])
        ax.set_xticks([0, x_lim])
        ax.set_ylim([-1, len(curr_bar)])
        ax.set_yticks([])
    return ax

def plot_single_neuron_color(neur_dict, xs, plot_x, filter_keys=None,
                             axs=None, n_shuffs=100):
    x_ind = np.argmin(np.abs(xs - plot_x))
    mi_cs = []
    for i, (k, v) in enumerate(neur_dict.items()):
        if filter_keys is not None and k[-1] in filter_keys:
            if axs is None:
                f, ax = plt.subplots(1, 1)
            else:
                ax = axs[i]
            edges = np.linspace(0, 2*np.pi, 9)
            bin_ids = np.digitize(v[1], edges)
            jag_tr = []
            jag_tr_shuff = []
            unique_ids = np.unique(bin_ids)
            for i, bi in enumerate(unique_ids):
                jag_tr.append(v[0][bi == bin_ids])
            mi_null = np.zeros(n_shuffs)
            for j in range(n_shuffs):
                bin_ids_shuff = np.random.choice(bin_ids, len(bin_ids),
                                                 replace=False)
                for i, bi in enumerate(unique_ids):
                    jag_tr_shuff.append(v[0][bi == bin_ids_shuff])
                mi_null[j] = color_index(jag_tr_shuff)

            gpl.plot_trace_werr(unique_ids, jag_tr, ax=ax, jagged=True)
            mi_c = color_index(jag_tr)
            print(mi_null)
            print('m null', np.nanmean(mi_null))
            print('s null', np.nanstd(mi_null))
            print('unnorm', mi_c)
            mi_c = (mi_c - np.nanmean(mi_null))/np.nanstd(mi_null)
            print('final', mi_c)
            mi_cs.append(mi_c)
            ax.set_title('MI: {}; {}'.format(mi_c, k[-1]))
    return np.array(mi_cs)

def color_index(frs):
    n = len(frs)
    ct = np.array(list(np.nanmean(fr) for fr in frs))
    ct_norm = ct/np.nansum(ct)
    mi_color = np.nansum(ct_norm*np.log(n*ct_norm))/np.log(n)
    return mi_color
