import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import sklearn.decomposition as skd
import torch
import scipy.special as sps

import general.plotting as gpl
import general.paper_utilities as pu
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.figures as swf
import swap_errors.theory as swt
import swap_errors.tcc_model as stcc
import swap_errors.visualization as swv
from . import neural_similarity as sns
import sklearn.preprocessing as skp

config_path = "swap_errors/swap_errors/figures_kernel.conf"


class KernelFigure(swf.SwapErrorFigure):
    def load_pickles(self, time="wheel-presentation", task="retro"):
        out_full = {}
        for use_monkey in self.monkeys:
            out_pickles, xs = swa.load_pop_pickles(
                time=time, monkey=use_monkey, task=task
            )
            out_full[use_monkey] = out_pickles, xs
        return out_full


class MKernelFigure(swf.SwapErrorFigure):
    def load_pickles(self, time="wheel-presentation", task="retro"):
        out_full = {}
        for use_monkey in self.monkeys:
            out_pickles, xs = swa.load_motoaki_pickles(
                time=time, monkey=use_monkey, task=task
            )
            out_full[use_monkey] = out_pickles, xs
        return out_full


class SchematicFigure(KernelFigure):
    def __init__(self, fig_key="schematic", colors=swf.colors, **kwargs):
        fsize = (5, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        mixture_gs = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 60, 0, 35, 10, 5)
        mixture_ax = self.get_axs(
            mixture_gs,
            squeeze=True,
            sharey="all",
        )
        gss["panel_mixture"] = mixture_ax

        geom_gs = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 60, 45, 80, 10, 5)
        geom_ax = self.get_axs(
            geom_gs,
            squeeze=True,
            sharey="all",
        )
        gss["panel_geom_distrib"] = geom_ax

        geom_gs = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 60, 65, 100, 10, 5)
        geom_ax = self.get_axs(
            geom_gs,
            squeeze=True,
            all_3d=True,
        )
        gss["panel_geom_rep"] = geom_ax

        bhv_gs = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 32, 40, 100, 10, 5)
        bhv_axs = self.get_axs(bhv_gs, squeeze=True, sharey="all", sharex="all")
        gss["panel_bhv"] = bhv_axs
        self.gss = gss

    def panel_bhv(self, task="retro"):
        key = "panel_bhv"
        axs = self.gss[key]
        data = self.load_pickles(task=task)

        n_bins = self.params.getint("n_bins")
        tb = self.params.getfloat("text_buff")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        for i, monkey in enumerate(self.monkeys):
            errs = []
            for sess_data in data[monkey][0].values():
                err = u.normalize_periodic_range(sess_data["rc"] - sess_data["c_targ"])
                errs.extend(err)
            axs[i].hist(
                errs,
                bins=bins,
                density=True,
                color=self.monkey_colors[monkey],
                label="Monkey {}".format(monkey[0]),
            )
            _make_xaxis_pi(axs[i])
            gpl.clean_plot(axs[i], 0)
            axs[i].set_xlabel("response error")
            if i == 0:
                label = "density"
            else:
                label = ""
                
            gpl.make_yaxis_scale_bar(
                axs[i], double=False, magnitude=0.1, label=label, text_buff=tb
            )

    def panel_mixture(self):
        key = "panel_mixture"
        axs = self.gss[key]

        pts = np.linspace(-np.pi, np.pi, 100)
        sig = 0.5
        gaussian = sts.norm(0, sig).pdf(pts)
        guess_weights = (0.5, 0.2)
        guess_color = self.params.getcolor("guess_color")
        gaussian_color = self.params.getcolor("correct_color")
        distr_color = self.params.getcolor("elmo_color")
        for i, guess_weight in enumerate(guess_weights):
            guess = np.ones_like(pts) * 1 / (2 * np.pi)

            ax = axs[i]
            ax.plot(pts, gaussian, color=gaussian_color, ls="dashed", label="encoded")
            ax.plot(pts, guess, color=guess_color, ls="dashed", label="not encoded")
            ax.plot(
                pts,
                guess * guess_weight + gaussian * (1 - guess_weight),
                color=distr_color,
                label="combined distribution",
            )
            gpl.clean_plot(ax, 0)
            ax.set_xlabel("response error")
            gpl.make_yaxis_scale_bar(
                ax, 0.2, double=False, label="density", text_buff=0.24
            )
            ax.legend(frameon=False)
            _make_xaxis_pi(ax)

    def panel_geom_rep(self):
        key = "panel_geom_rep"
        axs = self.gss[key]

        cmap = plt.get_cmap("hsv")
        pwr = 1
        wids = self.params.getlist("widths", typefunc=float)
        for i, wid in enumerate(wids):
            ax = axs[i]
            gpm = swt.GPKernelPopulation(pwr, wid)
            use_reps = gpm.reps[30:70]
            plot_reps = skd.PCA(3).fit_transform(use_reps)
            gpl.plot_colored_line(
                *plot_reps.T, ax=ax, cmap=cmap, color_bounds=(0.3, 0.7)
            )
            gpl.clean_3d_plot(ax)
            gpl.make_3d_bars(ax, bar_len=0.5)
            ax.view_init(33, 45)

    def panel_geom_distrib(self):
        key = "panel_geom_distrib"
        axs = self.gss[key]

        wids = self.params.getlist("widths", typefunc=float)
        snr = self.params.getfloat("snr")
        n_samps = self.params.getint("n_samps")
        cm = plt.get_cmap(self.params.get("cmap"))
        colors = cm(np.linspace(0.3, 0.9, len(wids)))

        for i, wid in enumerate(wids):
            gpm = swt.GPKernelPopulation(snr, wid)
            xs, dec_samps = gpm.simulate_decoding(n_samps=n_samps)
            diffs, kern = gpm.theoretical_kernel()
            kern_prob = np.exp(kern) / np.sum(np.exp(kern))
            kern_prob = kern_prob / (diffs[1] - diffs[0])
            axs[i].plot(diffs, kern_prob, color=colors[i])
            gpl.clean_plot(axs[i], 0)
            gpl.make_yaxis_scale_bar(
                axs[i],
                magnitude=0.2,
                double=False,
                label="density",
                text_buff=0.28,
            )
            _make_xaxis_pi(axs[i])
            axs[i].set_xlabel("response error")


class TheoryFigure(KernelFigure):
    def __init__(self, fig_key="theory", colors=swf.colors, **kwargs):
        fsize = (6, 2.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        theory_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100, 0, 50, 5, 8)
        axs = self.get_axs(theory_grid, sharex=True, sharey=True)
        gss["panel_theory"] = axs
        sim_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100, 60, 100, 10, 8)
        axs = self.get_axs(sim_grid, sharex="all", squeeze=True)
        gss["panel_simulation"] = axs
        self.gss = gss

    def get_models(self, **kwargs):
        pwr = kwargs.get("pwr", self.params.getfloat("pwr"))
        wid = kwargs.get("wid", self.params.getfloat("wid"))
        if self.data.get(("models", pwr, wid)) is None:
            rf_model = swt.RFKernelPopulation(pwr, wid)
            gp_model = swt.GPKernelPopulation(pwr, wid)
            self.data[("models", pwr, wid)] = (rf_model, gp_model)
        return self.data[("models", pwr, wid)]

    def panel_simulation(self):
        key = "panel_simulation"
        axs = self.gss[key]

        single_stim = self.params.getfloat("single_stim")
        n_dec_samps = self.params.getint("plot_n_resps")
        pwrs = self.params.getlist("sim_pwrs", typefunc=float)
        n_bins = self.params.getint("n_bins")
        n_samps = self.params.getint("n_samps")
        color_gp = self.params.getcolor("color_gp")

        eps = 1e-10
        bins = np.linspace(-np.pi - eps, np.pi + eps, n_bins)
        for i, pwr in enumerate(pwrs):
            ax_dist, ax_samps = axs[:, i]
            model_gp = self.get_models(pwr=pwr)[-1]
            xs, dec_samps = model_gp.sample_dec_gp(
                # single_stim=single_stim,
                n_samps=n_samps,
            )
            for j in range(n_dec_samps):
                ds = dec_samps[j]
                ax_samps.plot(xs, ds, color=(0.7,) * 3)
                ind = np.argmax(ds)
                ax_samps.plot(xs[ind], ds[ind], "o", color="k")
            true, est = model_gp.simulate_decoding(
                single_stim=single_stim, n_samps=n_samps
            )

            ax_dist.hist(est[:, 0], bins=bins, density=True, color=color_gp)
            ax_dist.hist(
                xs[np.argmax(dec_samps, axis=1)],
                histtype="step",
                bins=bins,
                density=True,
                color="k",
                linestyle="dashed",
            )
            ax_dist.set_xlabel("error")
            ax_samps.set_xlabel("stimulus\ndifference")
            ax_samps.set_ylabel("similarity")
            ax_dist.set_ylabel("density")
            gpl.clean_plot(ax_dist, i)
            gpl.clean_plot(ax_samps, 0)

    def panel_theory(self):
        key = "panel_theory"
        axs_sn = self.gss[key]

        n_sns = self.params.getint("plot_n_resps")
        models = self.get_models()
        colors = (
            self.params.getcolor("color_rf"),
            self.params.getcolor("color_gp"),
        )
        plot_theor = (
            False,
            True,
        )
        for i, model in enumerate(models):
            xs = model.stim[:, 0]
            ys = model.reps[:, :n_sns]
            axs_sn[i, 0].plot(xs, ys, color=colors[i])
            gpl.clean_plot(axs_sn[i, 0], 0)
            axs_sn[i, 0].set_ylabel("unit response (au)")
            if i < len(models) - 1:
                gpl.clean_plot_bottom(axs_sn[i, 0])
                gpl.clean_plot_bottom(axs_sn[i, 1])
            else:
                axs_sn[i, 1].set_xlabel("stimulus\ndifference")
                axs_sn[i, 0].set_xlabel("stimulus value")
            diffs, kernel = model.empirical_kernel()
            gpl.plot_scatter_average(diffs, kernel, ax=axs_sn[i, 1], color=colors[i])
            if plot_theor[i]:
                diffs, kernel = model.theoretical_kernel()
                axs_sn[i, 1].plot(diffs, kernel, color="k", ls="dashed")
            axs_sn[i, 1].set_ylabel("similarity")


def _cat_keys(pickles, *keys):
    out = []
    for k in keys:
        k_cat = np.concatenate(list(v[k] for v in pickles.values()))
        out.append(k_cat)
    return out


def _kl_div_quant(err, bins, n=1000):
    out = np.zeros((n, len(bins) - 1))
    rng = np.random.default_rng()
    for i in range(n):
        inds = rng.choice(len(err), len(err))
        out[i], _ = np.histogram(err[inds], bins=bins, density=True)
    return out


def _get_constrained_wid_fit(pickles, wids, n_bins=20, **kwargs):
    out_all = {}
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    out_uncons = {}
    for m, (data, xs) in pickles.items():
        out_all[m] = {}
        targ, dist, resp = _cat_keys(data, "c_targ", "c_dist", "rc")
        out_uncons[m] = stcc.fit_tcc_swap_model(
            targ, dist, resp, **kwargs
        )

        for m2, wid in wids.items():
            fixed_params = {"width": torch.tensor(wid[0])}
            out = stcc.fit_tcc_swap_model(
                targ, dist, resp, fixed_params=fixed_params, **kwargs
            )
            ts = out["targs"]
            rs_pred = out["predictive_samples"].detach().numpy()
            rs = out["resps"]

            errs = u.normalize_periodic_range(ts - rs)
            errs_wid = u.normalize_periodic_range(ts[None] - rs_pred).flatten()

            prob_bhv = _kl_div_quant(errs, bins)
            prob_wid = _kl_div_quant(errs_wid, bins)
            kl = np.sum(sps.kl_div(prob_bhv, prob_wid), axis=1)
            out_all[m][m2] = (kl, errs, errs_wid, out)
    return out_uncons, out_all


def _estimate_behavioral_wids(pickles, **kwargs):
    out_all = {}
    for m, (data, xs) in pickles.items():
        targ, dist, resp = _cat_keys(data, "c_targ", "c_dist", "rc")

        out = stcc.fit_tcc_swap_model(targ, dist, resp, **kwargs)
        out_all[m] = np.mean(out["samples"]["width"])
    return out_all


def _estimate_neural_wids(pickles, n_bins=25, use_gp=False, ks_dict=None, **kwargs):
    out_all = {}
    for m, (data, xs) in pickles.items():
        if ks_dict is None:
            kern = sns.AverageKernelMap(data, xs, n_bins=n_bins, **kwargs)
            ks, bcs = kern.get_kernel(use_gp=use_gp)
        else:
            ks, bcs = ks_dict[m][0]
        k_avg = np.mean(ks, axis=0)
        k_avg = k_avg - np.mean(k_avg)
        k_avg = k_avg / np.std(k_avg)

        wids = np.linspace(0.2, 8, 1000)
        sub = swt.vm_kernel(bcs[None], wids[:, None])
        ind = np.argmin(np.sum((sub - k_avg[None]) ** 2, axis=1))
        m_wid = wids[ind]
        out_all[m] = m_wid, sub, ind, k_avg, bcs
    return out_all



def _estimate_neural_func(pickles, n_bins=25, **kwargs):
    out_all = {}
    def wrap_func(func, mu, std, dtype=torch.float):
        def new_func(x):
            x = torch.tensor(x, dtype=dtype)
            y = np.mean(func(x), axis=1)
            return (y - mu) / std
        return new_func

    xs_norm = np.linspace(-np.pi, np.pi, 100)
    for m, (data, xs) in pickles.items():
        kern = sns.AverageKernelMap(data, xs, n_bins=n_bins, **kwargs)
        f = kern.get_func()
        k_norm = np.mean(f(torch.tensor(xs_norm, dtype=torch.float)), axis=1)
        kf = wrap_func(f, np.mean(k_norm), np.std(k_norm))
        
        out_all[m] = kf
    return out_all


def _make_xaxis_pi(ax):
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])


def _make_yaxis_pi(ax):
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])


class ExpAverageFigure(KernelFigure):
    def __init__(self, fig_key="exp-avg", colors=swf.colors, **kwargs):
        fsize = (8, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        single_unit_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 20, 0, 25, 5, 8)
        axs_e = self.get_axs(single_unit_grid, sharex=True, sharey=True)

        single_unit_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 25, 45, 0, 25, 5, 8)
        axs_w = self.get_axs(single_unit_grid, sharex=True, sharey=True)
        gss["panel_neuron_responses"] = (axs_e.flatten(), axs_w.flatten())

        kernel_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 42, 65, 100, 10, 8)
        axs = self.get_axs(kernel_grid, sharey="vertical", sharex="all", squeeze=True)
        gss["panel_bhv"] = axs[:, 1]
        gss["panel_kernel"] = axs.T

        full_kernel = pu.make_mxn_gridspec(self.gs, 2, 3, 50, 90, 30, 100, 8, 5)
        full_kernel_axs = self.get_axs(
            full_kernel,
            squeeze=True,
            sharex="all",
            sharey="all",
        )
        gss["panel_full_kernel"] = full_kernel_axs[:, 0]
        gss["panel_full_bhv_kernel"] = full_kernel_axs[:, 2]

        full_quant_ax = self.get_axs((self.gs[93:, 70:],), squeeze=False)[0, 0]
        gss["panel_full_bhv_prediction"] = (full_kernel_axs[:, 1], full_quant_ax)

        plane_ax = self.get_axs((self.gs[55:80, 0:25],), squeeze=False)[0, 0]
        quant_ax = self.get_axs((self.gs[93:, :25],), squeeze=False)[0, 0]
        gss["panel_bhv_plane"] = ((plane_ax, plane_ax), quant_ax)

        self.gss = gss

    def panel_neuron_responses(self, task="retro"):
        key = "panel_neuron_responses"
        axs_e, axs_w = self.gss[key]
        neuron_info_e = (
            ("Elmo", 0, 40),
            ("Elmo", 4, 15),
            ("Elmo", 5, 96),
            ("Elmo", 6, 145),
        )
        self._plot_neuron_responses(neuron_info_e, axs_e, task)

        neuron_info_w = (
            ("Waldorf", 13, 47),
            ("Waldorf", 20, 99),
            ("Waldorf", 14, 63),
            ("Waldorf", 18, 6),
        )
        self._plot_neuron_responses(neuron_info_w, axs_w, task)

    def panel_full_bhv_kernel(self, refit=False):
        key = "panel_full_bhv_kernel"
        axs = self.gss[key]
        task = self.params.get("task")

        for i, m in enumerate(self.monkeys):
            self._plot_full_bhv_kernel(m, axs[i], task)

    def _plot_full_bhv_kernel(self, monkey, ax, task, n_bins=30):
        tb = self.params.getfloat("text_buff")

        data = self.load_pickles(task=task)
        neurs, xs = data[monkey]

        swv.plot_target_response_scatter(neurs, n_plot_bins=n_bins, ax=ax)
        swv.make_colored_scale_bars(ax, tb=tb)
        ax.set_aspect("equal")
        gpl.clean_plot(ax, 0)

    def panel_full_kernel(self, refit=False):
        key = "panel_full_kernel"
        axs = self.gss[key]

        task = self.params.get("task")
        for i, m in enumerate(self.monkeys):
            self._plot_full_average_kernel(m, axs[i], task, refit=refit)

    def _plot_full_average_kernel(
        self, monkey, ax, task, n_bins=30, lw_color=3, refit=False
    ):
        key = "full-kernel_{}-{}".format(monkey, task)
        target_x = self.params.getfloat("target_x")
        p_thr = self.params.getfloat("p_thr")
        tb = self.params.getfloat("text_buff")
        cmap = "Grays"

        data = self.load_pickles(task=task)
        neurs, xs = data[monkey]

        if self.data.get(key) is None or refit:
            out_all = sns.compute_continuous_distance_matrix(
                neurs,
                xs,
                x_targ=target_x,
            )
            kern_map = []
            for out in out_all.values():
                mask = out["ps"][:, 0] > p_thr
                dists = out["dists"][mask][:, mask]
                c1 = out["c1"][mask][:, mask]
                c2 = out["c2"][mask][:, mask]

                d_flat = dists.flatten()
                c1_flat = c1.flatten()
                c2_flat = c2.flatten()
                d_flat, c1_flat, c2_flat = u.filter_nan(d_flat, c1_flat, c2_flat, ind=0)
                cs = np.stack((c1_flat, c2_flat), axis=1)
                c_pred = np.linspace(-np.pi, np.pi, n_bins)
                c1s_pred, c2s_pred = np.meshgrid(c_pred, c_pred)
                cs_pred = np.stack((c1s_pred.flatten(), c2s_pred.flatten()), axis=1)
                preds = sns._fit_svgpr(cs, d_flat, cs_pred, num_epochs=15)
                kern_map_i = np.reshape(preds, (n_bins, n_bins))
                kern_map.append(kern_map_i)
            kern_map = np.mean(kern_map, axis=0)
            self.data[key] = (c_pred, kern_map)
        c_pred, kern_map = self.data[key]
        v_extrem = np.abs(np.max(kern_map))
        m = gpl.pcolormesh(
            c_pred, c_pred, kern_map, ax=ax, vmin=-v_extrem, vmax=v_extrem, cmap=cmap
        )
        self.f.colorbar(m, ax=ax, label="similarity (au)")
        swv.make_colored_scale_bars(ax, tb=tb, lw=lw_color)
        ax.set_aspect("equal")
        gpl.clean_plot(ax, 0)

    def _plot_session_kernel(
        self, session, monkey, ax, task, n_bins=30, lw_color=3, refit=False
    ):
        key = "kernel-{}-{}-{}".format(session, monkey, task)
        target_x = self.params.getfloat("target_x")
        p_thr = self.params.getfloat("p_thr")
        tb = self.params.getfloat("text_buff")

        data = self.load_pickles(task=task)
        neurs, xs = data[monkey]

        if self.data.get(key) is None or refit:
            use_data = {0: neurs[session]}
            out = sns.compute_continuous_distance_matrix(
                use_data,
                xs,
                x_targ=target_x,
            )[0]
            mask = out["ps"][:, 0] > p_thr
            dists = out["dists"][mask][:, mask]
            c1 = out["c1"][mask][:, mask]
            c2 = out["c2"][mask][:, mask]

            d_flat = dists.flatten()
            c1_flat = c1.flatten()
            c2_flat = c2.flatten()
            d_flat, c1_flat, c2_flat = u.filter_nan(d_flat, c1_flat, c2_flat, ind=0)
            cs = np.stack((c1_flat, c2_flat), axis=1)
            c_pred = np.linspace(-np.pi, np.pi, n_bins)
            c1s_pred, c2s_pred = np.meshgrid(c_pred, c_pred)
            cs_pred = np.stack((c1s_pred.flatten(), c2s_pred.flatten()), axis=1)
            preds = sns._fit_svgpr(cs, d_flat, cs_pred)
            kern_map = np.reshape(preds, (n_bins, n_bins))
            self.data[key] = (c_pred, kern_map)
        c_pred, kern_map = self.data[key]
        v_extrem = np.abs(np.max(kern_map))
        m = gpl.pcolormesh(
            c_pred, c_pred, kern_map, ax=ax, vmin=-v_extrem, vmax=v_extrem, cmap="bwr"
        )
        self.f.colorbar(m, ax=ax, label="similarity (au)")
        ax.set_aspect("equal")
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
        ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
        gpl.make_xaxis_scale_bar(ax, np.pi, label="target color", text_buff=tb * 0.5)
        y_ext, _ = ax.get_ylim()
        gpl.plot_colored_line(
            c_pred,
            np.ones_like(c_pred) * y_ext,
            cmap="hsv",
            ax=ax,
            lw=lw_color,
        )
        gpl.make_yaxis_scale_bar(ax, np.pi, label="target color", text_buff=tb * 0.5)
        x_ext, _ = ax.get_xlim()
        gpl.plot_colored_line(
            np.ones_like(c_pred) * x_ext,
            c_pred,
            cmap="hsv",
            ax=ax,
            label="target_color",
            lw=lw_color,
        )
        gpl.clean_plot(ax, 0)

    def _plot_neuron_responses(self, neuron_info, axs, task):
        target_x = self.params.getfloat("target_x")
        p_thr = self.params.getfloat("p_thr")
        data = self.load_pickles(task=task)
        for i, (monkey, session, dim) in enumerate(neuron_info):
            neurs, xs = data[monkey]
            t_ind = np.argmin((xs - target_x) ** 2)
            resps = neurs[session]["spks"][:, dim, t_ind]
            colors = neurs[session]["c_targ"]
            ps = neurs[session]["ps"]
            mask = ps[:, 0] > p_thr
            (x, y), _ = gpl.plot_gp_scatter_average(
                colors[mask] - np.pi,
                resps[mask],
                ax=axs[i],
                return_xy=True,
                color=(0.8, 0.8, 0.8),
            )
            gpl.plot_colored_line(x, np.mean(y, axis=0), cmap="hsv", ax=axs[i])
            axs[i].set_xticks([-np.pi, 0, np.pi])
            axs[i].set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

            axs[i].set_ylabel("spikes/s")
            axs[i].set_xlabel("target color")

    def panel_bhv(self, task="retro"):
        key = "panel_bhv"
        axs = self.gss[key]
        data = self.load_pickles(task=task)

        n_bins = self.params.getint("n_bins")
        tb = self.params.getfloat("text_buff")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        for i, monkey in enumerate(self.monkeys):
            errs = []
            for sess_data in data[monkey][0].values():
                err = u.normalize_periodic_range(sess_data["rc"] - sess_data["c_targ"])
                errs.extend(err)
            axs[i].hist(
                errs,
                bins=bins,
                density=True,
                color=self.monkey_colors[monkey],
                label="Monkey {}".format(monkey[0]),
            )
            _make_xaxis_pi(axs[i])
            gpl.clean_plot(axs[i], 0)
        axs[-1].set_xlabel("response error")
        gpl.make_yaxis_scale_bar(
            axs[0], double=False, magnitude=0.1, label="density", text_buff=tb
        )
        gpl.make_yaxis_scale_bar(
            axs[1], double=False, magnitude=0.1, label="density", text_buff=tb
        )

    def _get_full_average_kernel_func(self, neurs, xs, monkey, refit=False, **kwargs):
        key = "full-kfunc-{}".format(monkey)
        if self.data.get(key) is None or refit:
            kfunc = sns.fit_full_average_kernel(neurs, xs, **kwargs)
            self.data[key] = kfunc
        return self.data[key]

    def panel_full_bhv_prediction(self, refit_kfunc=False, refit=False):
        key = "panel_full_bhv_prediction"
        axs, ax_quant = self.gss[key]

        lr = self.params.getfloat("lr")
        n_steps = self.params.getint("n_steps")
        n_bins = self.params.getint("n_bins")
        p_thr = self.params.getfloat("p_thr")
        tb = self.params.getfloat("text_buff")

        if self.data.get(key) is None or refit:
            data = self.load_pickles(task="retro")

            bhv_fits = {}
            kernel_funcs = {}
            use_wids = self._estimate_neural_wids(data)
            for monkey in self.monkeys:
                kernel_funcs[monkey] = self._get_full_average_kernel_func(
                    *data[monkey],
                    monkey,
                    p_thr=p_thr,
                    n_bins=n_bins,
                    refit=refit_kfunc,
                )
            for monkey in self.monkeys:
                targs_all = []
                dists_all = []
                resps_all = []
                for sess_data in data[monkey][0].values():
                    targs_all.extend(sess_data["c_targ"])
                    dists_all.extend(sess_data["c_dist"])
                    resps_all.extend(sess_data["rc"])

                bhv_fits[monkey] = {}
                for m2 in self.monkeys:
                    bhv_fits[monkey][m2] = stcc.fit_tcc_kernel_model(
                        np.array(targs_all),
                        np.array(dists_all),
                        np.array(resps_all),
                        kernel_funcs[m2],
                        fixed_params={"width": use_wids[monkey][0]},
                        lr=lr,
                        n_steps=n_steps,
                    )
            self.data[key] = bhv_fits
        bhv_fits = self.data[key]
        for i, (monkey, res) in enumerate(bhv_fits.items()):
            pred = res[monkey]["predictive_samples"].detach().numpy().T
            targ = res[monkey]["targs"]
            targ = np.repeat(targ[:, None], pred.shape[1], axis=1)
            swv.plot_target_response_scatter_direct(
                targ.flatten(), pred.flatten(), ax=axs[i]
            )
            swv.make_colored_scale_bars(axs[i], tb=tb * 1.5, y_label="response color")
            axs[i].set_aspect("equal")
            gpl.clean_plot(axs[i], 0)

            for m2 in self.monkeys:
                kernel_use = res[m2]["samples"]["kernel_mix"][:, 1]
                gpl.violinplot(
                    [kernel_use],
                    [i],
                    color=[self.monkey_colors[m2]],
                    ax=ax_quant,
                    vert=False,
                )
        gpl.clean_plot(ax_quant, 1, ticks=False)
        ax_quant.invert_yaxis()
        gpl.make_xaxis_scale_bar(
            ax_quant,
            magnitude=0.1,
            double=False,
            label="kernel weight",
            text_buff=0.45,
        )
        ax_quant.set_yticks([0, 1])
        ax_quant.set_yticklabels(["E", "W"])

    def panel_kernel(self, refit=False, task="retro"):
        key = "panel_kernel"
        axs_kern, axs_bhv = self.gss[key]
        data = self.load_pickles(task=task)

        lr = self.params.getfloat("lr")
        n_steps = self.params.getint("n_steps")
        n_bins = self.params.getint("n_bins")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        p_thr = self.params.getfloat("p_thr")
        tb = self.params.getfloat("text_buff")

        if self.data.get(key) is None or refit:
            bhv_fits = {}
            kern_fits = {}
            for i, monkey in enumerate(self.monkeys):
                targs_all = []
                dists_all = []
                resps_all = []
                for sess_data in data[monkey][0].values():
                    targs_all.extend(sess_data["c_targ"])
                    dists_all.extend(sess_data["c_dist"])
                    resps_all.extend(sess_data["rc"])

                if task == "single":
                    bhv_fits[monkey] = stcc.fit_tcc_model(
                        np.array(targs_all),
                        np.array(resps_all),
                        lr=lr,
                        n_steps=n_steps,
                    )
                else:
                    bhv_fits[monkey] = stcc.fit_tcc_swap_model(
                        np.array(targs_all),
                        np.array(dists_all),
                        np.array(resps_all),
                        lr=lr,
                        n_steps=n_steps,
                    )
                out = sns.compute_continuous_distance_matrix(
                    *data[monkey],
                    color_key=self.params.get("use_color"),
                )
                kern_fits[monkey] = sns.compute_continuous_distance_masks(
                    out, p_thr=p_thr, n_bins=n_bins
                )

            self.data[key] = bhv_fits, kern_fits
        bhv_fits, kern_fits = self.data[key]
        for i, (monkey, fit) in enumerate(bhv_fits.items()):
            pred_errs = u.normalize_periodic_range(
                fit["predictive_samples"] - np.expand_dims(fit["targs"], 0)
            ).flatten()
            axs_bhv[i].hist(
                pred_errs,
                bins=bins,
                histtype="step",
                density=True,
                color="k",
                linestyle="dashed",
                label="behavioral model",
            )
            axs_bhv[i].legend(frameon=False)

            use_bs = np.expand_dims(bins, 0)
            wids = np.expand_dims(fit["samples"]["width"], 1)
            funcs = swt.vm_kernel(use_bs, wids)
            color = self.monkey_colors[monkey]
            func_mu = np.mean(funcs, axis=0)
            fmin = np.min(func_mu)
            fmax = np.max(func_mu)
            gpl.plot_trace_werr(
                bins,
                (funcs - fmin) / (fmax - fmin),
                ax=axs_kern[i],
                conf95=True,
                color="k",
                linestyle="dashed",
                label="kernel estimated\nfrom behavior",
            )

            ys, xs = kern_fits[monkey][0]
            ys_mu = np.nanmean(ys, axis=0, keepdims=True)
            scaler = skp.MinMaxScaler().fit(ys_mu.T)
            ys_scale = np.squeeze(
                np.array(list(scaler.transform(y_i.reshape((-1, 1))) for y_i in ys))
            )
            gpl.plot_trace_werr(
                xs, ys_scale, ax=axs_kern[i], color=color, label="neural kernel"
            )
            gpl.clean_plot(axs_kern[i], 0)
            _make_xaxis_pi(axs_kern[i])

        axs_kern[-1].set_xlabel("difference from\ntarget")
        axs_bhv[-1].set_xlabel("response error")

        gpl.make_yaxis_scale_bar(
            axs_kern[0],
            double=False,
            magnitude=0.2,
            label="similarity (au)",
            text_buff=tb,
        )
        gpl.make_yaxis_scale_bar(
            axs_kern[1],
            double=False,
            magnitude=0.2,
            label="similarity (au)",
            text_buff=tb,
        )

    def panel_bhv_plane(self, refit=False):
        key = "panel_bhv_plane"
        axs, _ = self.gss[key]

        tasks = ("retro",)  # "pro", "single")
        keys = {
            "retro": ("c_targ", "c_dist", "rc"),
            "pro": ("c_targ", "c_dist", "rc"),
            "single": ("c_targ", "rc"),
        }
        models = {
            "retro": stcc.fit_corr_guess_swap_model,
            "pro": stcc.fit_corr_guess_swap_model,
            "single": stcc.fit_corr_guess_model,
        }
        kwargs = {}

        if self.data.get(key) is None or refit:
            fits = {}
            for task in tasks:
                data_task = self.load_pickles(task=task)
                fits[task] = {}
                for m, (data_task_m, _) in data_task.items():
                    cats_task = _cat_keys(data_task_m, *keys[task])
                    model = models[task]
                    fits[task][m] = model(*cats_task, **kwargs)
            self.data[key] = fits

        fits = self.data[key]
        cms = {m: gpl.make_linear_cmap(col) for m, col in self.monkey_colors.items()}
        for i, task in enumerate(tasks):
            fit_task = fits[task]
            for j, (m, fit_tm) in enumerate(fit_task.items()):
                sigmas = fit_tm["samples"]["sigma"]
                rates = fit_tm["samples"]["resp_rate"]
                guess_rates = rates[:, -1] / (rates[:, 0] + rates[:, -1])
                color = cms[m]((i + 1) / len(tasks))
                axs[j].plot(sigmas, guess_rates, "o", color=color, ms=1)

    def _estimate_neural_wids(self, data, panel_key="panel_kernel", **kwargs):
        if self.data.get(panel_key) is not None:
            _, ks_dict = self.data[panel_key]
        else:
            ks_dict = None
        return _estimate_neural_wids(data, ks_dict=ks_dict, **kwargs)

    def panel_bhv_plane_theory_func(self, refit=False):
        key_data = "panel_bhv_plane_theory"
        key_ax = "panel_bhv_plane"
        axs, ax_quant = self.gss[key_ax]

        snr_theory = self.params.getlist("snr_theory_linspace", typefunc=float)
        n_snr_theory = self.params.getint("n_snr")
        snrs = np.linspace(*snr_theory, n_snr_theory)
        n_samps = self.params.getint("n_samps")
        n_trls = self.params.getint("n_theory_trials")
        n_reps = self.params.getint("n_reps")

        if self.data.get(key_data) is None or refit:
            data = self.load_pickles()
            use_funcs = _estimate_neural_func(data)
            out = {}
            for m, func in use_funcs.items():
                sigmas = np.zeros((len(snrs), n_reps, n_samps))
                guess_rates = np.zeros_like(sigmas)
                for i, snr in enumerate(snrs):
                    for j in range(n_reps):
                        skt = swt.SDKTFunction(snr, func)
                        true, dec = skt.simulate_decoding(n_samps=n_trls)
                        out_fit = stcc.fit_corr_guess_model(true, dec, n_samps=n_samps)
                        sigmas[i, j] = out_fit["samples"]["sigma"]
                        guess_rates[i, j] = out_fit["samples"]["resp_rate"][:, -1]
                out[m] = (sigmas, guess_rates)
            self.data[key_data] = use_funcs, out

        use_wids, out = self.data[key_data]
        for i, (m, (sig_m, guess_m)) in enumerate(out.items()):
            gpl.plot_trace_werr(
                np.mean(sig_m, axis=-1).T,
                np.mean(guess_m, axis=-1).T,
                conf95=True,
                ax=axs[i],
                fill=False,
                color=self.monkey_colors[m],
            )
            axs[i].set_yscale("log")
            axs[i].set_xlabel("response variance")
            axs[i].set_ylabel("threshold error rate")

        gpl.clean_plot(ax_quant, 1, ticks=False)
        ax_quant.invert_yaxis()
        gpl.make_xaxis_scale_bar(
            ax_quant,
            magnitude=0.03,
            double=False,
            label="divergence",
            text_buff=0.45,
        )
        ax_quant.set_yticks([0, 1])
        ax_quant.set_yticklabels(["E", "W"])

    def panel_bhv_plane_theory(self, refit=False):
        key_data = "panel_bhv_plane_theory"
        key_ax = "panel_bhv_plane"
        axs, ax_quant = self.gss[key_ax]

        snr_theory = self.params.getlist("snr_theory_linspace", typefunc=float)
        n_snr_theory = self.params.getint("n_snr")
        snrs = np.linspace(*snr_theory, n_snr_theory)
        n_samps = self.params.getint("n_samps")
        n_trls = self.params.getint("n_theory_trials")
        n_reps = self.params.getint("n_reps")

        if self.data.get(key_data) is None or refit:
            data = self.load_pickles()
            use_wids = self._estimate_neural_wids(data)
            wid_fit = _get_constrained_wid_fit(data, use_wids)
            out = {}
            for m, wid in use_wids.items():
                sigmas = np.zeros((len(snrs), n_reps, n_samps))
                guess_rates = np.zeros_like(sigmas)
                for i, snr in enumerate(snrs):
                    for j in range(n_reps):
                        skt = swt.SimplifiedDiscreteKernelTheory(snr, wid[0])
                        true, dec = skt.simulate_decoding(n_samps=n_trls)
                        out_fit = stcc.fit_corr_guess_model(true, dec, n_samps=n_samps)
                        sigmas[i, j] = out_fit["samples"]["sigma"]
                        guess_rates[i, j] = out_fit["samples"]["resp_rate"][:, -1]
                out[m] = (sigmas, guess_rates)
            self.data[key_data] = use_wids, out, wid_fit

        use_wids, out, wid_fit = self.data[key_data]
        for i, (m, (sig_m, guess_m)) in enumerate(out.items()):
            gpl.plot_trace_werr(
                np.mean(sig_m, axis=-1).T,
                np.mean(guess_m, axis=-1).T,
                conf95=True,
                ax=axs[i],
                fill=False,
                color=self.monkey_colors[m],
            )
            axs[i].set_yscale("log")
            axs[i].set_xlabel("response variance")
            axs[i].set_ylabel("threshold error rate")

            for m2 in self.monkeys:
                gpl.violinplot(
                    [wid_fit[m][m2]],
                    [i],
                    ax=ax_quant,
                    vert=False,
                    color=[self.monkey_colors[m2]],
                )
        gpl.clean_plot(ax_quant, 1, ticks=False)
        ax_quant.invert_yaxis()
        gpl.make_xaxis_scale_bar(
            ax_quant,
            magnitude=0.03,
            double=False,
            label="divergence",
            text_buff=0.45,
        )
        ax_quant.set_yticks([0, 1])
        ax_quant.set_yticklabels(["E", "W"])


class GeometryChangeFigure(KernelFigure):
    def __init__(self, fig_key="kernel-change", colors=swf.colors, **kwargs):
        fsize = (6, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        geoms_grid = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 0, 50, 10, 10)
        geoms_axs = self.get_axs(geoms_grid, sharex="all", sharey="columns")
        gss["panel_geoms"] = geoms_axs

        gss["panel_plane"] = self.get_axs((self.gs[25:75, 60:],), squeeze=False)[0, 0]
        self.gss = gss

    def panel_plane(self):
        key = "panel_plane"
        ax = self.gss[key]
        wids = self.params.getlist("widths", typefunc=float)
        snr_range = self.params.getlist("snr_range", typefunc=float)
        n_snr = self.params.getint("n_snr")
        snrs = np.linspace(*snr_range, n_snr)
        cm = plt.get_cmap(self.params.get("cmap"))
        colors = cm(np.linspace(0.3, 0.9, len(wids)))

        for i, wid in enumerate(wids):
            le = swt.local_err(snrs, wid)
            tb = swt.threshold_prob(snrs, wid)
            gpl.plot_trace_werr(le, tb, color=colors[i], ax=ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("local error")
        ax.set_ylabel("threshold error rate")

    def panel_geoms(self):
        key = "panel_geoms"
        axs = self.gss[key]
        wids = self.params.getlist("widths", typefunc=float)
        snr = self.params.getfloat("snr")
        n_samps = self.params.getint("n_samps")
        cm = plt.get_cmap(self.params.get("cmap"))
        colors = cm(np.linspace(0.3, 0.9, len(wids)))

        for i, wid in enumerate(wids):
            gpm = swt.GPKernelPopulation(snr, wid)
            xs, dec_samps = gpm.simulate_decoding(n_samps=n_samps)
            errs = u.normalize_periodic_range(xs - dec_samps)
            diffs, kern = gpm.theoretical_kernel()
            axs[i, 0].plot(diffs, kern, color=colors[i])
            axs[i, 1].hist(errs, density=True, color=colors[i], bins=20)
            gpl.clean_plot(axs[i, 0], 0)
            gpl.clean_plot(axs[i, 1], 0)
            gpl.make_yaxis_scale_bar(
                axs[i, 0],
                magnitude=0.2,
                double=False,
                label="similarity",
                text_buff=0.28,
            )
            gpl.make_yaxis_scale_bar(
                axs[i, 1],
                magnitude=0.1,
                double=False,
                label="density",
                text_buff=0.28,
            )
            _make_xaxis_pi(axs[i, 0])
            _make_xaxis_pi(axs[i, 1])
            axs[i, 0].set_xlabel(r"$\Delta$ target")
            axs[i, 1].set_xlabel(r"$\Delta$ target")


class CombinedSingleTrialFigure(KernelFigure):
    def __init__(
        self, fig_key="combined-single-trial", task="retro", colors=swf.colors, **kwargs
    ):
        fsize = (8.5, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        self.task = task

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        td_ax = self.get_axs((self.gs[:15, :17],), squeeze=False)[0, 0]
        gss["panel_trial_division"] = td_ax

        schem_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 25, 45, 0, 17, 7, 2)
        axs = self.get_axs(schem_grid, sharex="all", sharey="all", squeeze=True)
        gss["panel_early_schematic_random"] = axs[0]
        gss["panel_early_schematic_kernel"] = axs[1]

        kernels_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 45, 25, 47, 10, 2)
        axs = self.get_axs(
            kernels_grid, sharey="horizontal", sharex="all", squeeze=True
        )
        gss["panel_early_kernels"] = axs.T

        schem_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 25, 45, 53, 70, 7, 2)
        axs = self.get_axs(schem_grid, sharex="all", sharey="all", squeeze=True)
        gss["panel_late_schematic_random"] = axs[0]
        gss["panel_late_schematic_kernel"] = axs[1]

        kernels_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 45, 78, 100, 10, 2)
        axs = self.get_axs(
            kernels_grid, sharey="horizontal", sharex="all", squeeze=True
        )
        gss["panel_late_kernels"] = axs.T

        tc_grid = pu.make_mxn_gridspec(self.gs, 2, 3, 60, 100, 40, 100, 13, 3)
        tc_axs = self.get_axs(
            tc_grid,
            sharey="horizontal",
            sharex="vertical",
        )
        gss["panel_tc"] = tc_axs

        self.gss = gss

    def panel_trial_division(self):
        key = "panel_trial_division"
        ax = self.gss[key]

        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")
        gray_color = self.params.getcolor("null_color")
        p_thr = self.params.getfloat("p_thr")
        n_bins = self.params.getint("n_bins_hist")
        eg_monkey = "Waldorf"

        swv.plot_corr_guess_histogram(
            self.load_pickles()[eg_monkey][0],
            ax=ax,
            p_thr=p_thr,
            corr_color=c_color,
            guess_color=g_color,
            other_color=gray_color,
            bins=n_bins,
        )
        _make_xaxis_pi(ax)

    def panel_early_schematic_kernel(self):
        key = "panel_early_schematic_kernel"
        ax_targ, ax_resp = self.gss[key]

        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")

        n_bins = self.params.getint("n_bins")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        func = np.exp(-(bins**2) / 2)
        unif = np.ones_like(bins) * 0.5
        ax_targ.plot(bins, func, color=c_color)
        ax_targ.plot(bins, func + 0.06, color=g_color)
        ax_resp.plot(bins, func, color=c_color)
        ax_resp.plot(bins, unif, color=g_color)
        gpl.clean_plot(ax_resp, 1)
        gpl.clean_plot(ax_targ, 1)
        ax_targ.set_xlabel(r"$\Delta$ target")
        ax_resp.set_xlabel(r"$\Delta$ response")
        gpl.make_yaxis_scale_bar(
            ax_targ, magnitude=0.4, double=False, label="similarity"
        )
        _make_xaxis_pi(ax_resp)
        _make_xaxis_pi(ax_targ)

    def panel_early_schematic_random(self):
        key = "panel_early_schematic_random"
        ax_targ, ax_resp = self.gss[key]
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")

        n_bins = self.params.getint("n_bins")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        func = np.exp(-(bins**2) / 2)
        unif = np.ones_like(bins) * 0.5
        ax_targ.plot(bins, func, color=c_color)
        ax_targ.plot(bins, unif, color=g_color)
        ax_resp.plot(bins, func, color=c_color)
        ax_resp.plot(bins, unif, color=g_color)
        gpl.clean_plot(ax_resp, 1)
        gpl.clean_plot(ax_targ, 1)
        gpl.make_yaxis_scale_bar(
            ax_targ, magnitude=0.4, double=False, label="similarity"
        )
        _make_xaxis_pi(ax_resp)
        _make_xaxis_pi(ax_targ)

    def panel_late_schematic_kernel(self):
        key = "panel_late_schematic_kernel"
        ax_targ, ax_resp = self.gss[key]

        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")

        n_bins = self.params.getint("n_bins")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        func = np.exp(-(bins**2) / 2)
        unif = np.ones_like(bins) * 0.5
        ax_targ.plot(bins, func, color=c_color)
        ax_targ.plot(bins, unif, color=g_color)
        ax_resp.plot(bins, func, color=c_color)
        ax_resp.plot(bins, func + 0.06, color=g_color)
        gpl.clean_plot(ax_resp, 1)
        gpl.clean_plot(ax_targ, 1)
        ax_targ.set_xlabel(r"$\Delta$ target")
        ax_resp.set_xlabel(r"$\Delta$ response")
        gpl.make_yaxis_scale_bar(
            ax_targ, magnitude=0.4, double=False, label="similarity"
        )
        _make_xaxis_pi(ax_resp)
        _make_xaxis_pi(ax_targ)

    def panel_late_schematic_random(self):
        key = "panel_late_schematic_random"
        ax_targ, ax_resp = self.gss[key]
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")

        n_bins = self.params.getint("n_bins")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        func = np.exp(-(bins**2) / 2)
        unif = np.ones_like(bins) * 0.5
        ax_targ.plot(bins, func, color=c_color)
        ax_targ.plot(bins, unif, color=g_color)
        ax_resp.plot(bins, func, color=c_color)
        ax_resp.plot(bins, unif, color=g_color)
        gpl.clean_plot(ax_resp, 1)
        gpl.clean_plot(ax_targ, 1)
        gpl.make_yaxis_scale_bar(
            ax_targ, magnitude=0.4, double=False, label="similarity"
        )
        _make_xaxis_pi(ax_resp)
        _make_xaxis_pi(ax_targ)

    def panel_late_kernels(self, recompute=False):
        key = "panel_late_kernels"
        axs = self.gss[key]

        time_key = self.params.get("late_time_event")
        time_targ = self.params.getfloat("late_time_target")
        same_cue = None

        self._kernel_panel(
            time_key, time_targ, same_cue, axs, key, recompute=recompute, labels=False
        )

    def panel_early_kernels(self, recompute=False):
        key = "panel_early_kernels"
        axs = self.gss[key]

        time_key = self.params.get("early_time_event")
        time_targ = self.params.getfloat("early_time_target")
        same_cue = (False, True)

        self._kernel_panel(
            time_key, time_targ, same_cue, axs, key, recompute=recompute, labels=False
        )

    def _kernel_panel(
        self,
        time_key,
        time_targ,
        same_cue,
        axs,
        key,
        recompute=False,
        labels=True,
    ):
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")
        use_gp = self.params.getboolean("use_gp")
        colors = (c_color, g_color)

        n_bins = self.params.getint("n_bins")
        p_thr = self.params.getfloat("p_thr")
        color_keys = ("c_targ", "rc")
        plot_inds = {0: "correct", 2: "guess"}
        data = self.load_pickles(time=time_key)
        if self.data.get(key) is None or recompute:
            kernels = {}
            for i, monkey in enumerate(self.monkeys):
                data_m = data[monkey]
                kernels[monkey] = {}
                for j, ck in enumerate(color_keys):
                    if same_cue is None:
                        out = sns.compute_continuous_distance_matrix(
                            *data_m,
                            color_key=ck,
                            x_targ=time_targ,
                        )

                        mask_dict = sns.compute_continuous_distance_masks(
                            out,
                            p_thr=p_thr,
                            n_bins=n_bins,
                            use_gp=use_gp,
                        )
                    else:
                        mask_dict = {}
                        for pi in plot_inds.keys():
                            k, bs = sns.make_cued_kernel_map(
                                *data_m,
                                time_targ,
                                col_ind=pi,
                                second_color_key=ck,
                                use_gp=use_gp,
                                same_cue=same_cue[j],
                                p_thr=p_thr,
                                n_bins=n_bins,
                            )
                            mask_dict[pi] = (k, bs)
                    kernels[monkey][ck] = mask_dict
            self.data[key] = kernels

        kernels = self.data[key]
        for i, monkey in enumerate(self.monkeys):
            for j, ck in enumerate(color_keys):
                mask_dict = kernels[monkey][ck]
                ax = axs[j, i]
                if j == 0:
                    if i == len(self.monkeys) - 1:
                        ax.set_xlabel(r"$\Delta$ target")
                    ax.set_ylabel("similarity")
                else:
                    if i == len(self.monkeys) - 1:
                        ax.set_xlabel(r"$\Delta$ response")
                gpl.clean_plot(ax, j)
                for k, (ind, label) in enumerate(plot_inds.items()):
                    ys, xs = mask_dict[ind]
                    if labels and i == 0 and j == 0:
                        label_use = label
                    else:
                        label_use = ""
                    gpl.plot_trace_werr(xs, ys, ax=ax, label=label_use, color=colors[k])
                    _make_xaxis_pi(ax)

    def panel_tc(self, refit=False):
        key = "panel_tc"
        axs = self.gss[key]
        t_keys = self.params.getlist("tc_times")
        n_bins = self.params.getint("n_bins")
        p_thr = self.params.getfloat("p_thr")
        use_gp = False

        if self.data.get(key) is None or refit:
            out = {}
            for i, time in enumerate(t_keys):
                out[time] = {}
                data = self.load_pickles(task=self.task, time=time)
                for j, m in enumerate(self.monkeys):
                    neur, xs = data[m]
                    k_c_tc, bs = sns.make_cued_kernel_map_tc(
                        neur, xs, p_thr=p_thr, n_bins=n_bins, use_gp=use_gp
                    )
                    k_g_tc, bs = sns.make_cued_kernel_map_tc(
                        neur,
                        xs,
                        col_ind=2,
                        use_gp=use_gp,
                        n_bins=n_bins,
                        p_thr=p_thr,
                    )
                    k_rc_tc, bs = sns.make_cued_kernel_map_tc(
                        neur,
                        xs,
                        col_ind=2,
                        second_color_key="rc",
                        use_gp=use_gp,
                        n_bins=n_bins,
                        p_thr=p_thr,
                        same_cue=True,
                    )
                    max_ind = np.argmin(np.abs(bs))
                    min_ind = np.argmax(np.abs(bs))
                    c_tc = k_c_tc[:, max_ind] - k_c_tc[:, min_ind]
                    g_tc = k_g_tc[:, max_ind] - k_g_tc[:, min_ind]
                    rc_tc = k_rc_tc[:, max_ind] - k_rc_tc[:, min_ind]
                    out[time][m] = ((c_tc, g_tc, rc_tc), xs)
            self.data[key] = out
        kernels = self.data[key]
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")
        s_color = self.params.getcolor("swap_color")
        t_strs = ("color onset", "cue onset", "response period")

        colors = (c_color, g_color, s_color)
        for i, time in enumerate(t_keys):
            for j, m in enumerate(self.monkeys):
                ks, xs = kernels[time][m]
                for k, k_ijk in enumerate(ks):
                    gpl.plot_trace_werr(xs, k_ijk, ax=axs[j, i], color=colors[k])
                gpl.add_hlines(0, axs[j, i])
                gpl.clean_plot(axs[j, i], i)
                if j == 1:
                    axs[j, i].set_xlabel("time from\n{}".format(t_strs[i]))
        axs[0, 0].set_ylabel("signal strength (au)")
        axs[1, 0].set_ylabel("signal strength (au)")


class CosyneSingleTrialFigure(KernelFigure):
    def __init__(self, fig_key="single-trial-late", colors=swf.colors, **kwargs):
        fsize = (4.5, 2.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        schem_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 50, 100, 0, 35, 13, 4)
        axs = self.get_axs(schem_grid, sharex="all", sharey="all", squeeze=True)
        gss["panel_schematic_random"] = axs[0]
        gss["panel_schematic_kernel"] = axs[1]

        kernels_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100, 50, 100, 13, 5)
        axs = self.get_axs(
            kernels_grid, sharey="horizontal", sharex="all", squeeze=True
        )
        gss["panel_kernels"] = axs.T
        self.gss = gss

    def panel_schematic_kernel(self):
        key = "panel_schematic_kernel"
        ax_targ, ax_resp = self.gss[key]

        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")

        n_bins = self.params.getint("n_bins")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        func = np.exp(-(bins**2) / 2)
        unif = np.ones_like(bins) * 0.5
        ax_targ.plot(bins, func, color=c_color)
        ax_targ.plot(bins, unif, color=g_color)
        ax_resp.plot(bins, func, color=c_color)
        ax_resp.plot(bins, func + 0.06, color=g_color)
        gpl.clean_plot(ax_resp, 1)
        gpl.clean_plot(ax_targ, 1)
        ax_targ.set_xlabel("difference from\ntarget")
        ax_resp.set_xlabel("difference from\nresponse")
        gpl.make_yaxis_scale_bar(
            ax_targ, magnitude=0.4, double=False, label="similarity"
        )

    def panel_schematic_random(self):
        key = "panel_schematic_random"
        ax_targ, ax_resp = self.gss[key]
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")

        n_bins = self.params.getint("n_bins")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        func = np.exp(-(bins**2) / 2)
        unif = np.ones_like(bins) * 0.5
        ax_targ.plot(bins, func, color=c_color)
        ax_targ.plot(bins, unif, color=g_color)
        ax_resp.plot(bins, func, color=c_color)
        ax_resp.plot(bins, unif, color=g_color)
        gpl.clean_plot(ax_resp, 1)
        gpl.clean_plot(ax_targ, 1)
        gpl.make_yaxis_scale_bar(
            ax_targ, magnitude=0.4, double=False, label="similarity"
        )

    def panel_kernels(self, recompute=False):
        key = "panel_kernels"
        axs = self.gss[key]
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")
        time_key = self.params.get("time_event")
        time_targ = self.params.getfloat("time_target")
        colors = (c_color, g_color)

        n_bins = self.params.getint("n_bins")
        p_thr = self.params.getfloat("p_thr")
        color_keys = ("c_targ", "rc")
        plot_inds = {0: "correct", 2: "guess"}
        data = self.load_pickles(time=time_key)
        if self.data.get(key) is None or recompute:
            kernels = {}
            for i, monkey in enumerate(self.monkeys):
                data_m = data[monkey]
                kernels[monkey] = {}
                for j, ck in enumerate(color_keys):
                    out = sns.compute_continuous_distance_matrix(
                        *data_m,
                        color_key=ck,
                        x_targ=time_targ,
                    )

                    mask_dict = sns.compute_continuous_distance_masks(
                        out,
                        p_thr=p_thr,
                        n_bins=n_bins,
                        use_gp=False,
                    )
                    kernels[monkey][ck] = mask_dict
            self.data[key] = kernels

        kernels = self.data[key]
        for i, monkey in enumerate(self.monkeys):
            for j, ck in enumerate(color_keys):
                mask_dict = kernels[monkey][ck]
                ax = axs[j, i]
                if j == 0:
                    if i == len(self.monkeys) - 1:
                        ax.set_xlabel("difference from\ntarget")
                    ax.set_ylabel("similarity")
                else:
                    if i == len(self.monkeys) - 1:
                        ax.set_xlabel("difference from\nresponse")
                gpl.clean_plot(ax, j)
                for k, (ind, label) in enumerate(plot_inds.items()):
                    ys, xs = mask_dict[ind]
                    gpl.plot_trace_werr(xs, ys, ax=ax, label=label, color=colors[k])


class CosyneSingleTrialEarlyFigure(CosyneSingleTrialFigure):
    def __init__(self, fig_key="single-trial-early", **kwargs):
        super().__init__(fig_key=fig_key)

    def panel_kernels(self, recompute=False):
        key = "panel_kernels"
        axs = self.gss[key]
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")
        time_key = self.params.get("time_event")
        time_targ = self.params.getfloat("time_target")
        colors = (c_color, g_color)

        n_bins = self.params.getint("n_bins")
        p_thr = self.params.getfloat("p_thr")
        color_keys = ("c_targ", "rc")
        same_cue = (False, True)
        plot_inds = {0: "correct", 2: "guess"}
        data = self.load_pickles(time=time_key)
        if self.data.get(key) is None or recompute:
            kernels = {}
            for i, monkey in enumerate(self.monkeys):
                data_m = data[monkey]
                kernels[monkey] = {}
                for j, ck in enumerate(color_keys):
                    mask_dict = {}
                    for pi in plot_inds.keys():
                        k, bs = sns.make_cued_kernel_map(
                            *data_m,
                            time_targ,
                            col_ind=pi,
                            second_color_key=ck,
                            use_gp=False,
                            same_cue=same_cue[j],
                            p_thr=p_thr,
                            n_bins=n_bins,
                        )
                        mask_dict[pi] = (k, bs)
                    kernels[monkey][ck] = mask_dict
            self.data[key] = kernels

        kernels = self.data[key]
        for i, monkey in enumerate(self.monkeys):
            for j, ck in enumerate(color_keys):
                mask_dict = kernels[monkey][ck]
                ax = axs[j, i]
                if j == 0:
                    if i == len(self.monkeys) - 1:
                        ax.set_xlabel("difference from\ntarget")
                    ax.set_ylabel("similarity")
                else:
                    if i == len(self.monkeys) - 1:
                        ax.set_xlabel("difference from\nresponse")
                gpl.clean_plot(ax, j)
                for k, (ind, label) in enumerate(plot_inds.items()):
                    ys, xs = mask_dict[ind]
                    gpl.plot_trace_werr(xs, ys, ax=ax, label=label, color=colors[k])


class SingleTrialFigure(KernelFigure):
    def __init__(
        self,
        task="retro",
        fig_key="retro-single",
        t_keys=("wheel-presentation",),
        colors=swf.colors,
        fwid=1,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.task = task
        self.t_keys = t_keys
        n_ts = len(t_keys)
        n_cs = len(params.getlist("c1_prewheel"))

        fsize = (fwid * (2 * n_cs), n_ts * fwid)

        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        n_cs = len(self.params.getlist("c1_prewheel"))
        n_ts = len(self.t_keys)

        vspace = 4
        hspace = 3
        kernels_grid_m1 = pu.make_mxn_gridspec(
            self.gs, n_ts, n_cs, 0, 100, 0, 47, vspace, hspace
        )
        kernels_grid_m2 = pu.make_mxn_gridspec(
            self.gs,
            n_ts,
            n_cs,
            0,
            100,
            53,
            100,
            vspace,
            hspace,
        )
        axs_m1 = self.get_axs(kernels_grid_m1, sharey="all", sharex="all", squeeze=True)
        axs_m2 = self.get_axs(kernels_grid_m2, sharey="all", sharex="all", squeeze=True)
        for i, k in enumerate(self.t_keys):
            gss[k] = np.stack((axs_m1[i], axs_m2[i]), axis=0)

        self.gss = gss

    def _plot_kernels(
        self,
        time,
        c1s,
        axs,
        c2s=None,
        cues=None,
        x_targ=-0.25,
        add_x_label=False,
        **kwargs,
    ):
        out_full = self.load_pickles(time=time, task=self.task)
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")
        s_color = self.params.getcolor("swap_color")
        n_bins = self.params.getint("n_bins")
        p_thr = self.params.getfloat("p_thr")
        inds = self.params.getlist("plot_inds", typefunc=int)
        x_labels = self.params.getlist("c1_labels")
        use_gp = self.params.getboolean("use_gp")
        colors = (c_color, s_color, g_color)
        if cues is None:
            cues = (None,) * len(c1s)
        if c2s is None:
            c2s = (None,) * len(c1s)

        for i, (monkey, (data, xs)) in enumerate(out_full.items()):
            axs[i]
            for j, c1 in enumerate(c1s):
                if i == 0 and j == 0:
                    labels = swv.ind_labels
                else:
                    labels = ("",) * len(swv.ind_labels)
                swv.plot_kernel_targ(
                    data,
                    xs,
                    c1,
                    c2=c2s[j],
                    cue_only=cues[j],
                    ax=axs[i, j],
                    inds=inds,
                    n_bins=n_bins,
                    colors=colors,
                    labels=labels,
                    x_targ=x_targ,
                    p_thr=p_thr,
                    use_gp=use_gp,
                    **kwargs,
                )
                gpl.clean_plot(axs[i, j], j)
                if add_x_label:
                    xl = r"$\Delta$ {}".format(x_labels[j])
                    axs[i, j].set_xlabel(xl)
                if j == 0:
                    axs[i, j].set_ylabel("similarity (au)")


class RetroCuedTC(KernelFigure):
    def __init__(
        self, fig_key="retro-tc", task="retro", colors=swf.colors, fwid=1.5, **kwargs
    ):
        t_keys = (
            "color-presentation",
            "pre-cue-presentation",
            "post-cue-presentation",
            "wheel-presentation",
        )
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.task = task
        self.t_keys = t_keys
        n_ts = len(t_keys)

        fsize = (fwid * n_ts, 2 * fwid)

        self.fig_key = fig_key

        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        n_ts = len(self.t_keys)

        vspace = 8
        hspace = 2
        tc_grid = pu.make_mxn_gridspec(self.gs, 2, n_ts, 0, 100, 0, 100, vspace, hspace)
        axs = self.get_axs(
            tc_grid, sharey="horizontal", sharex="vertical", squeeze=True
        )
        gss["panel_tc"] = axs

        self.gss = gss

    def panel_tc(self, refit=False):
        key = "panel_tc"
        axs = self.gss[key]

        if self.data.get(key) is None or refit:
            out = {}
            for i, time in enumerate(self.t_keys):
                out[time] = {}
                data = self.load_pickles(task=self.task, time=time)
                for j, m in enumerate(self.monkeys):
                    neur, xs = data[m]
                    k_c_tc, bs = sns.make_cued_kernel_map_tc(neur, xs, use_gp=False)
                    k_g_tc, bs = sns.make_cued_kernel_map_tc(
                        neur, xs, col_ind=2, use_gp=False
                    )
                    k_rc_tc, bs = sns.make_cued_kernel_map_tc(
                        neur,
                        xs,
                        col_ind=2,
                        second_color_key="rc",
                        use_gp=False,
                        same_cue=True,
                    )
                    max_ind = np.argmin(np.abs(bs))
                    min_ind = np.argmax(np.abs(bs))
                    c_tc = k_c_tc[:, max_ind] - k_c_tc[:, min_ind]
                    g_tc = k_g_tc[:, max_ind] - k_g_tc[:, min_ind]
                    rc_tc = k_rc_tc[:, max_ind] - k_rc_tc[:, min_ind]
                    out[time][m] = ((c_tc, g_tc, rc_tc), xs)
            self.data[key] = out
        kernels = self.data[key]
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")
        s_color = self.params.getcolor("swap_color")
        t_strs = ("color onset", "cue onset", "cue onset", "response period")

        colors = (c_color, g_color, s_color)
        for i, time in enumerate(self.t_keys):
            for j, m in enumerate(self.monkeys):
                ks, xs = kernels[time][m]
                for k, k_ijk in enumerate(ks):
                    gpl.plot_trace_werr(xs, k_ijk, ax=axs[j, i], color=colors[k])
                gpl.add_hlines(0, axs[j, i])
                gpl.clean_plot(axs[j, i], i)
                if j == 1:
                    axs[j, i].set_xlabel("time from\n{}".format(t_strs[i]))
        axs[0, 0].set_ylabel("signal strength (au)")
        axs[1, 0].set_ylabel("signal strength (au)")


class RetroSingleTrialFigure(SingleTrialFigure):
    def __init__(self, fig_key="retro-single", task="retro", **kwargs):
        t_keys = (
            "panel_kernels_postcolor",
            "panel_kernels_precue",
            "panel_kernels_postcue",
            "panel_kernels_prewheel",
        )

        super().__init__(fig_key=fig_key, t_keys=t_keys, task=task, **kwargs)

    def panel_kernels_postcolor(self):
        key = "panel_kernels_postcolor"
        axs = self.gss[key]

        time = self.params.get("postcolor_time")
        c1s = self.params.getlist("c1_postcolor")
        cues = self.params.getlist("cues_postcolor", typefunc=int)
        if u.check_list(cues):
            cues = list(cue if cue >= 0 else None for cue in cues)
        x_targ = self.params.getfloat("postcolor_x_targ")
        self._plot_kernels(time, c1s, axs, cues=cues, x_targ=x_targ)

    def panel_kernels_precue(self):
        key = "panel_kernels_precue"
        axs = self.gss[key]

        time = self.params.get("precue_time")
        c1s = self.params.getlist("c1_precue")
        cues = self.params.getlist("cues_precue", typefunc=int)
        if u.check_list(cues):
            cues = list(cue if cue >= 0 else None for cue in cues)
        x_targ = self.params.getfloat("precue_x_targ")
        self._plot_kernels(time, c1s, axs, cues=cues, x_targ=x_targ)

    def panel_kernels_postcue(self):
        key = "panel_kernels_postcue"
        axs = self.gss[key]

        time = self.params.get("postcue_time")
        c1s = self.params.getlist("c1_postcue")
        cues = self.params.getlist("cues_postcue", typefunc=int)
        if u.check_list(cues):
            cues = list(cue if cue >= 0 else None for cue in cues)
        x_targ = self.params.getfloat("postcolor_x_targ")
        self._plot_kernels(time, c1s, axs, cues=cues, x_targ=x_targ)

    def panel_kernels_prewheel(self):
        key = "panel_kernels_prewheel"
        axs = self.gss[key]

        time = self.params.get("prewheel_time")
        c1s = self.params.getlist("c1_prewheel")
        x_targ = self.params.getfloat("prewheel_x_targ")
        self._plot_kernels(time, c1s, axs, x_targ=x_targ, add_x_label=True)


class ProSingleTrialFigure(SingleTrialFigure):
    def __init__(self, fig_key="pro-single", task="pro", **kwargs):
        t_keys = (
            "panel_kernels_postcolor",
            "panel_kernels_prewheel",
        )

        super().__init__(fig_key=fig_key, t_keys=t_keys, task=task)

    def panel_kernels_postcolor(self):
        key = "panel_kernels_postcolor"
        axs = self.gss[key]

        time = self.params.get("postcolor_time")
        c1s = self.params.getlist("c1_postcolor")
        cues = self.params.getlist("cues_postcolor", typefunc=int)
        if u.check_list(cues):
            cues = list(cue if cue >= 0 else None for cue in cues)
        x_targ = self.params.getfloat("postcolor_x_targ")
        self._plot_kernels(time, c1s, axs, cues=cues, x_targ=x_targ)

    def panel_kernels_prewheel(self):
        key = "panel_kernels_prewheel"
        axs = self.gss[key]

        time = self.params.get("prewheel_time")
        c1s = self.params.getlist("c1_prewheel")
        x_targ = self.params.getfloat("prewheel_x_targ")
        self._plot_kernels(time, c1s, axs, x_targ=x_targ, add_x_label=True)
