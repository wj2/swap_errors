
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
import sklearn.decomposition as skd
import sklearn.manifold as skm
import arviz as az
import itertools as it
import seaborn as sns

import general.plotting as gpl
import general.utility as u
import swap_errors.analysis as swan

def plot_model_probs(*args, plot_keys=('swap_prob', 'guess_prob'), ax=None,
                     sep=.5, comb_func=np.median, colors=None, sub_x=-1,
                     labels=('swaps', 'guesses'), total_label='correct',
                     arg_names=('Elmo', 'Waldorf'), ms=3, monkey_colors=None):
    if colors is None:
        colors = (None,)*len(args)
    if ax is None:
        f, ax = plt.subplots(1, 1)
    cents = np.arange(0, len(plot_keys))
    n_clusters = len(args)
    swarm_full = {'x':[], 'y':[], 'monkey':[]}
    violin_full = {'x':[], 'y':[], 'monkey':[]}
    for i, m in enumerate(args):
        offset = (i - n_clusters/2)*sep
        swarm_data = {'x':[], 'y':[]}
        violin_data = {'x':[], 'y':[]}
        for j, pk in enumerate(plot_keys):
            pk_sessions = comb_func(m[pk], axis=(0, 1))
            pk_full = m[pk].to_numpy().flatten()
            if j == 0:
                totals = np.zeros_like(pk_sessions)
                totals_full = np.zeros_like(pk_full)
            totals = totals + pk_sessions
            totals_full = totals_full + pk_full
            xs_full = np.ones(len(pk_full))*(offset + cents[j])
            violin_data['x'] = np.concatenate((violin_data['x'],
                                               xs_full))
            violin_data['y'] = np.concatenate((violin_data['y'],
                                               pk_full))
            xs = np.ones(len(pk_sessions))*(offset + cents[j])
            swarm_data['x'] = np.concatenate((swarm_data['x'], xs))
            swarm_data['y'] = np.concatenate((swarm_data['y'],
                                              pk_sessions))
            
        xs_full = np.ones(len(totals_full))*(offset + sub_x)
        violin_data['x'] = np.concatenate((violin_data['x'],
                                           xs_full))
        violin_data['y'] = np.concatenate((violin_data['y'],
                                           1 - totals_full))
        
        xs = np.ones(len(pk_sessions), dtype=float)*(offset + sub_x)
        swarm_data['x'] = np.concatenate((swarm_data['x'], xs))
        swarm_data['y'] = np.concatenate((swarm_data['y'],
                                          1 - totals))
        swarm_full['x'] = np.concatenate((swarm_full['x'],
                                          swarm_data['x']))
        swarm_full['y'] = np.concatenate((swarm_full['y'],
                                          swarm_data['y']))
        monkey_list = [arg_names[i]]*len(swarm_data['x'])
        swarm_full['monkey'] = np.concatenate((swarm_full['monkey'],
                                               monkey_list))
        
        violin_full['x'] = np.concatenate((violin_full['x'],
                                          violin_data['x']))
        violin_full['y'] = np.concatenate((violin_full['y'],
                                          violin_data['y']))
        monkey_list = [arg_names[i]]*len(violin_data['x'])
        violin_full['monkey'] = np.concatenate((violin_full['monkey'],
                                                monkey_list))

    # sns.violinplot(data=violin_full, x='x', y='y', hue='monkey',
    #                palette=monkey_colors, ax=ax)
    l = sns.swarmplot(data=swarm_full, x='x', y='y', hue='monkey',
                      palette=monkey_colors,
                      ax=ax, size=ms)
    ax.legend(frameon=False)
    ax.set_ylim([0, 1])
    ax.set_xticks([.5, 2.5, 4.5])
    ax.set_xticklabels((total_label,) + labels)
    ax.set_ylabel('probability')
    gpl.clean_plot(ax, 0)
    return ax

default_ax_labels = ('guess probability', 'swap probability',
                     'correct probability')
def visualize_simplex(pts, ax=None, ax_labels=default_ax_labels, thr=.5,
                      grey_col=(.7, .7, .7)):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')
    all_pts = np.stack(list(v[1] for v in pts), axis=0)
    ax.plot(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], 'o',
            color=grey_col)
    for i in range(all_pts.shape[1]):
        mask = all_pts[:, i] > thr
        plot_pts = all_pts[mask]
        ax.plot(plot_pts[:, 0], plot_pts[:, 1], plot_pts[:, 2], 'o',
                label=ax_labels[i].split()[0])

    ax.legend(frameon=False)
    ax.plot([0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0], color=grey_col)
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_zlabel(ax_labels[2])
    ax.view_init(45, 45)
    return ax

def visualize_simplex_2d(pts, ax=None, ax_labels=None, thr=.5,
                         pt_grey_col=(.7, .7, .7),
                         line_grey_col=(.6, .6, .6),
                         colors=None, bottom_x=.8,
                         bottom_y=-1.1, top_x=.35, top_y=1,
                         legend=False):
    if colors is None:
        colors = (None,)*pts.shape[1]
    if ax_labels is None:
        ax_labels = ('',)*pts.shape[1]
    pts_x = pts[:, 1] - pts[:, 0]
    pts_y = pts[:, 2] - (pts[:, 0] + pts[:, 1])
    ax.plot(pts_x, pts_y, 'o', color=pt_grey_col)
    for i in range(pts.shape[1]):
        mask = pts[:, i] > thr
        ax.plot(pts_x[mask], pts_y[mask], 'o', color=colors[i],
                label=ax_labels[i])
    ax.plot([-1, 0], [-1, 1], color=line_grey_col)
    ax.plot([0, 1], [1, -1], color=line_grey_col)
    ax.plot([-1, 1], [-1, -1], color=line_grey_col)
    if legend:
        ax.legend(frameon=False)
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax, 0)
    ax.text(bottom_x, bottom_y, ax_labels[1], verticalalignment='top',
            horizontalalignment='center')
    ax.text(-bottom_x, bottom_y, ax_labels[0], verticalalignment='top',
            horizontalalignment='center')
    ax.text(top_x, top_y, ax_labels[2], verticalalignment='top',
            horizontalalignment='center', rotation=-60)
    return ax

def plot_posterior_predictive_dims_dict(models, data, dims=5, fwid=3,
                                        ks_dict=None, **kwargs):
    f, axs = plt.subplots(len(models), dims,
                          figsize=(dims*fwid, len(models)*fwid))
    for i, (k, v) in enumerate(models.items()):
        if ks_dict is not None:
            ks = ks_dict.get(k, None)
        else:
            ks = None
        plot_posterior_predictive_dims(v, data, dims=dims, axs=axs[i],
                                       ks=ks, **kwargs)
        axs[i, 0].set_ylabel(k)
    axs[i, -1].legend(frameon=False)
    
def plot_mu_hists(models, mu_key, mu2=None, axs=None, fwid=3):
    pshape = list(models.values())[0].posterior[mu_key].shape[2:]
    if axs is None:
        figsize = (fwid*pshape[1], fwid*pshape[0])
        f, axs = plt.subplots(*pshape, figsize=figsize)
    for k, m in models.items():
        mus = m.posterior[mu_key]
        for (i, j) in u.make_array_ind_iterator(pshape):
            mu_plot = mus[..., i, j].to_numpy().flatten()
            if mu2 is not None:
                mu2_plot = m.posterior[mu2][..., i, j].to_numpy().flatten()
                mu_plot = mu_plot - mu2_plot
            axs[i, j].hist(mu_plot, label=k,
                           histtype='step')
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
        _, bins, _ = axs[i].hist(d[:, i], density=True,
                    label='observed')
        axs[i].hist(total_post[:, i], histtype='step', density=True,
                    label='predictive', linestyle='dashed', color='k',
                    bins=bins)
        if ks is not None:
            gpl.add_vlines(d_ks[:, i], axs[i])
        
        
def plot_color_means(cmeans, ax=None, dimred=3, t_ind=5,
                     links=2):
    if ax is None:
        f = plt.figure()
        if dimred == 3:
            ax = f.add_subplot(1, 1, 1,  projection='3d')
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
    ax.plot(*upper_trans[:, :dimred].T, 'o')
    ax.plot(*upper_trans[:, :dimred].T)
    ax.plot(*lower_trans[:, :dimred].T, 'o')
    ax.plot(*lower_trans[:, :dimred].T)
    links = np.linspace(0, len(cmeans) - 1, links, dtype=int)
    for l in links:
        up_low = np.stack((upper_trans[l, :dimred],
                           lower_trans[l, :dimred]),
                          axis=0)
        ax.plot(*up_low.T, color=(.8, .8, .8))
    print(u.pr_only(p.explained_variance_ratio_))
    return ax         

def _plot_mu(mu_format, trs, color, style, ax, ms=5, hsv_color=True,
             bckgd_color=(.9, .9, .9), n_col_pts=64, **kwargs):
    plot_mu = trs(mu_format)
    plot_mu_append = np.concatenate((plot_mu, plot_mu[:, :1]), axis=1)
    if hsv_color:
        cmap = plt.get_cmap('hsv')
        n_bins = plot_mu.shape[1]
        pts = np.linspace(0, 1, n_col_pts + 1)[:-1]
        cols = pts*np.pi*2
        col_rep = swan.spline_color(cols, n_bins)
        plot_mu_pts = np.dot(trs(mu_format), col_rep)
        pt_colors = cmap(pts)
        col = None
        ax.plot(*plot_mu_append[:3], color=bckgd_color, ls=style,
                **kwargs)
        for i in range(plot_mu_pts.shape[1]):
            ax.plot(*plot_mu_pts[:3, i:i+1], 'o', color=pt_colors[i], ls=style,
                    markersize=ms)
    else:    
        l = ax.plot(*plot_mu_append[:3], color=color, ls=style, **kwargs)
        col = l[0].get_color()
        ax.plot(*plot_mu_append[:3], 'o', color=col, ls=style, markersize=ms)
    return ax, col

def visualize_model_collection_views(mdict, dim_red_model=skd.PCA,
                                     n_neighbors=16, dim_red=True,
                                     c_u=None, axs=None, fwid=3,
                                     kwarg_combs=None, mu_g_keys=(),
                                     use_isomap=True, **kwargs):
    if use_isomap:
        dim_red_model = skm.Isomap
        vis_kwargs = {'dim_red_model':dim_red_model,
                      'n_neighbors':n_neighbors}
    else:
        vis_kwargs = {'dim_red_model':skd.PCA}
    if kwarg_combs is None:
        kwarg_combs = ({'mu_u_keys':('mu_u',), 'mu_l_keys':('mu_l',)},
                       {'mu_u_keys':()},
                       {'mu_l_keys':()},
                       {'mu_u_keys':('mu_d_u',), 'mu_l_keys':('mu_d_l',)},)
    n_plots = len(kwarg_combs)
    if axs is None:
        f = plt.figure(figsize=(fwid*n_plots, fwid))
        axs = list(f.add_subplot(1, n_plots, i + 1, projection='3d')
                   for i in range(n_plots))
    for i, ax in enumerate(axs):
        kwarg_combs[i].update(kwargs)
        kwarg_combs[i].update(vis_kwargs)
        if i == 0:
            legend = True
        else:
            legend = False
        visualize_model_collection(mdict, dim_red=dim_red,
                                   c_u=c_u, ax=ax, mu_g_keys=mu_g_keys,
                                   legend=legend,
                                   **kwarg_combs[i])
    return f, axs
    
    

def visualize_model_collection(mdict, dim_red=True,
                               ax=None, **kwargs):
    trs = None
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')
    for k, v in mdict.items():
        out = visualize_fit_results(v, dim_red=dim_red, ax=ax,
                                    label_cl=k, trs=trs, **kwargs)
        if dim_red and trs is None:
            trs = out[1]
    return ax

def get_color_reps(cols, mus, ignore_col=False, roll_ax=0,
                   roll=False, transpose=False):
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

def visualize_fit_torus(fit_az, ax=None, trs=None, eh_key='err_hat',
                        dim_red=True, color=None, ms=5, n_cols=64,
                        u_key='mu_u', l_key='mu_d_l', ignore_col=True,
                        plot_surface=False, plot_trls=False,
                        dim_red_model=skd.PCA, **kwargs):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')
    cols = np.linspace(0, 2*np.pi, n_cols)
    col_grid_x, col_grid_y = np.meshgrid(cols, cols)
    x_reps = get_color_reps(col_grid_x, fit_az.posterior[u_key],
                            ignore_col=ignore_col)
    y_reps = get_color_reps(col_grid_y, fit_az.posterior[l_key],
                            ignore_col=ignore_col, transpose=False)
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
        ax.plot(*pts_dr.T, 'o')
    return ax

def plot_session_swap_distr_collection(session_dict, axs=None, n_bins=20,
                                       fwid=3, p_ind=1, bin_bounds=None,
                                       ret_data=True, colors=None,
                                       **kwargs):
    if colors is None:
        colors = {}
    if axs is None:
        n_plots = len(list(session_dict.values())[0][0])
        fsize = (fwid*n_plots, fwid)
        f, axs = plt.subplots(1, n_plots, figsize=fsize,
                              sharex=False, sharey=False)
        if n_plots == 1:
            axs = [axs]
    true_d = {}
    pred_d = {}
    ps_d = {}
    for (sn, (mdict, data)) in session_dict.items():
        for (k, faz) in mdict.items():
            out = swan.get_normalized_centroid_distance(faz, data, p_ind=p_ind,
                                                        **kwargs)
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
    if bin_bounds is not None:
        bins = np.linspace(*bin_bounds, n_bins)
    else:
        bins = n_bins
    out_data = {}
    for i, (k, td) in enumerate(true_d.items()):
        td_full = np.concatenate(td, axis=0)
        pd_full = np.concatenate(pred_d[k], axis=0)
        ps_full = np.concatenate(ps_d[k], axis=0)
        # print(swd.dip(td_full))
        color = colors.get(k)
        _, bins, _ = axs[i].hist(td_full, bins=bins, color=color,
                                    density=True, label='observed')
        axs[i].hist(pd_full, bins=bins, histtype='step', color='k',
                    linestyle='dashed',
                    density=True, label='predicted')
        gpl.add_vlines([0, 1], axs[i])
        axs[i].set_ylabel(k)
        if ret_data:
            out_data[k] = td_full
    axs[i].legend(frameon=False)
    if ret_data:
        out = axs, out_data
    else:
        out = axs
    return out
            
def plot_swap_distr_collection(model_dict, data, axs=None,
                               fwid=3, **kwargs):
    if axs is None:
        n_plots = len(model_dict)
        fsize = (fwid*n_plots, fwid)
        f, axs = plt.subplots(1, n_plots, figsize=fsize,
                              sharex=True, sharey=True)
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
    _, bins, _ = ax.hist(true_arr_full, histtype='step', density=True,
                         bins=n_bins)
    ax.hist(pred_arr_full, histtype='step', density=True,
            bins=bins)
    gpl.add_vlines([0, 1], ax)
    return ax

class MultiSessionFit():
    
    def __init__(self, fit_list):
        self.posterior = {}
        for k in fit_list[0].posterior.keys():
            k_list = []
            for fl in fit_list:
                add_arr = fl.posterior[k]
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

def visualize_fit_results(fit_az, mu_u_keys=('mu_u', 'mu_d_u'),
                          mu_l_keys=('mu_l', 'mu_d_l'), dim_red=True,
                          c_u=(1, 0, 0), c_l=(0, 1, 0), c_g=(0, 0, 1),
                          mu_g_keys=('mu_g',), trs=None,
                          styles=('solid', 'dashed'), ax=None,
                          label_cu='', label_cl='', same_color=True,
                          legend=True, truncate_dim=None,
                          n_cols=64, dim_red_model=skd.PCA, **kwargs):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')
    all_keys = mu_u_keys + mu_l_keys
    if dim_red and trs is None:
        l = []
        for i, ak in enumerate(all_keys):
            d_ak = fit_az.posterior[ak]
            m_ak = np.mean(d_ak, axis=(0, 1))
            cols = np.linspace(0, 2*np.pi, n_cols + 1)[:-1]
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
        mu_format = np.mean(fit_az.posterior[mu_k], axis=(0, 1))
        _, c_u = _plot_mu(mu_format, trs, c_u, styles[i], ax,
                          label=label_cu)
    if same_color:
        c_l, c_g = c_u, c_u
    for i, mu_k in enumerate(mu_l_keys):
        mu_format = np.mean(fit_az.posterior[mu_k], axis=(0, 1))
        _, c_l = _plot_mu(mu_format, trs, c_l, styles[i], ax,
                          label=label_cl)
    for i, mu_k in enumerate(mu_g_keys):
        mu_format = np.mean(fit_az.posterior[mu_k], axis=(0, 1))
        mu_format = np.expand_dims(mu_format, 1)
        _, c_g = _plot_mu(mu_format, trs, c_g, styles[i], ax)
    if legend:
        ax.legend(frameon=False)
    return ax, trs

def plot_k_trials(fit_az, k_thresh=.7, dim_red=None, ax=None,
                  k_color=(1, 0, 0), reg_color=(.5, .5, .5),
                  plot_nk=False):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    x = az.loo(fit_az, pointwise=True)
    k_vals = x['pareto_k']
    k_mask = k_vals > k_thresh
    k_trls = fit_az.observed_data.y[k_mask]
    nk_trls = fit_az.observed_data.y[np.logical_not(k_mask)]
    if dim_red is not None:
        k_trls = dim_red(k_trls.T)
        nk_trls = dim_red(nk_trls.T)
    ax.plot(*k_trls, 'o', color=k_color)
    if plot_nk:
        ax.plot(*nk_trls, 'o', color=reg_color)

def plot_k_distributions_dict(inp_dict, **kwargs):
    ks = list(inp_dict.keys())
    vs = list(inp_dict[k] for k in ks)
    return plot_k_distributions(vs, ks, **kwargs)

def plot_k_correlation_dict(ks_dict, data, data_label='', fwid=3):
    f, axs = plt.subplots(1, len(ks_dict), figsize=(fwid*len(ks_dict), fwid))
    for i, (k, k_vals) in enumerate(ks_dict.items()):
        plot_k_correlation(k_vals[0], data, ax=axs[i])
        axs[i].set_xlabel(k)
        axs[i].set_ylabel(data_label)
    return f, axs

def plot_k_correlation(k_vals, data, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ax.plot(k_vals, data, 'o')    

def plot_k_distributions(models, labels, k_thresh=.7, fwid=3, sharex=True,
                         sharey=True, compute_k=True):
    if compute_k:
        models = list(swan.get_pareto_k(m)[0] for m in models)
    combs = list(it.combinations(range(len(models)), 2))
    side_plots = len(models) - 1
    f, axs = plt.subplots(side_plots, side_plots,
                          figsize=(fwid*side_plots, fwid*side_plots),
                          sharex=sharex, sharey=sharey)
    for i, (ind1, ind2) in enumerate(combs):
        ax = axs[ind1, ind2 - 1]
        ax.plot(models[ind1], models[ind2], 'o')
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
    cent = full - px/2
    norm = cent/np.sqrt(np.sum(cent**2, axis=2, keepdims=True))
    vec = np.expand_dims([0, 1], axis=(0, 1))
    dist = np.sqrt(np.sum((norm - vec)**2, axis=2))
    sim = np.arcsin(dist/2) 
    sim[x < px/2] = -sim[x < px/2]
    sim = sim - np.nanmin(sim)
    sim = sim/np.nanmax(sim)

    r = np.sqrt(np.sum(cent**2, axis=2))
    mask = np.logical_or(r < r_cent - r_wid/2, r > r_cent + r_wid/2)
    sim[mask] = np.nan
    ax.imshow(sim, cmap='hsv')
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax, 0)

def plot_error_swap_distribs(data, err_field='err', dist_field='LABthetaDist',
                             resp_field='LABthetaResp', **kwargs):
    errs = np.concatenate(data[err_field])
    dist_errs = np.concatenate(data[dist_field] - data[resp_field])
    dist_errs = u.normalize_periodic_range(dist_errs)
    return plot_error_swap_distribs_err(errs, dist_errs, **kwargs)

def plot_error_swap_distribs_err(errs, dist_errs, axs=None, fwid=3,
                                 label='', model_data=None, color=None,
                                 model_derr=None):
    if axs is None:
        fsize = (2*fwid, fwid)
        f, axs = plt.subplots(1, 2, figsize=fsize, sharey=True,
                              sharex=True)
    l = axs[0].hist(errs,  density=True, color=color)
    if model_data is not None:
        axs[0].hist(model_data.flatten(), histtype='step', density=True,
                    color='k', linestyle='dashed')
    axs[1].hist(dist_errs, label=label, density=True,
                color=color)
    if model_derr is not None:
        m_derr = u.normalize_periodic_range(model_derr - model_data)
        axs[1].hist(m_derr.flatten(), histtype='step', density=True,
                    color='k', linestyle='dashed')
    axs[1].legend(frameon=False)
    axs[0].set_xlabel('error (rads)')
    axs[0].set_ylabel('density')
    axs[1].set_xlabel('distractor distance (rads)')
    gpl.clean_plot(axs[0], 0)
    gpl.clean_plot(axs[1], 1)
    return axs

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
            if test_dists[pop_ind].shape[1] > 0:
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
