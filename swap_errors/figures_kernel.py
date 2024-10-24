import numpy as np

import general.plotting as gpl
import general.paper_utilities as pu
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.figures as swf
import swap_errors.theory as swt
import swap_errors.tcc_model as stcc
import swap_errors.neural_similarity as sns
import general.torch.simple as gts
import sklearn.preprocessing as skp

config_path = "swap_errors/figures_kernel.conf"


class KernelFigure(swf.SwapErrorFigure):
    def load_pickles(self, time="wheel-presentation", task="retro"):
        out_full = {}
        for use_monkey in self.monkeys:
            out_pickles, xs = swa.load_pop_pickles(
                time=time, monkey=use_monkey, task=task
            )
            out_full[use_monkey] = out_pickles, xs
        return out_full


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


class ExpAverageFigure(KernelFigure):
    def __init__(self, fig_key="exp-avg", colors=swf.colors, **kwargs):
        fsize = (6.5, 4.25)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        single_unit_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 50, 100, 0, 40, 5, 8)
        axs = self.get_axs(single_unit_grid, sharex=True, sharey=True)
        gss["panel_neuron_responses"] = axs

        kernel_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100, 50, 100, 10, 4)
        axs = self.get_axs(kernel_grid, sharey="horizontal", sharex="all", squeeze=True)
        gss["panel_bhv"] = axs[0]
        gss["panel_kernel"] = axs
        self.gss = gss

    def panel_neuron_responses(self):
        key = "panel_neuron_responses"
        axs = self.gss[key].flatten()

        target_x = self.params.getfloat("target_x")
        p_thr = self.params.getfloat("p_thr")
        data = self.load_pickles()
        neuron_info = (
            ("Elmo", 8, 150),
            ("Elmo", 4, 15),
            ("Waldorf", 14, 63),
            ("Waldorf", 18, 6),
        )
        for i, (monkey, session, dim) in enumerate(neuron_info):
            neurs, xs = data[monkey]
            t_ind = np.argmin((xs - target_x) ** 2)
            resps = neurs[session]["spks"][:, dim, t_ind]
            colors = neurs[session]["c_targ"]
            ps = neurs[session]["ps"]
            mask = ps[:, 0] > p_thr
            _, (x, y) = gpl.plot_scatter_average(
                colors[mask],
                resps[mask],
                ax=axs[i],
                return_xy=True,
                color=(0.8, 0.8, 0.8),
            )
            gpl.plot_colored_line(x, y, cmap="hsv", ax=axs[i])
            axs[i].set_xticks([])
            axs[i].set_ylabel("spikes/s")
            axs[i].set_xlabel("target color")

    def panel_bhv(self):
        key = "panel_bhv"
        axs = self.gss[key]
        data = self.load_pickles()

        n_bins = self.params.getint("n_bins")
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
            gpl.clean_plot(axs[i], i)
            axs[i].set_xlabel("response error")
        gpl.make_yaxis_scale_bar(
            axs[0], double=False, magnitude=0.1, label="density", text_buff=0.3
        )

    def panel_kernel(self, refit=False):
        key = "panel_kernel"
        axs_bhv, axs_kern = self.gss[key]
        data = self.load_pickles()

        lr = self.params.getfloat("lr")
        n_steps = self.params.getint("n_steps")
        n_bins = self.params.getint("n_bins")
        bins = np.linspace(-np.pi, np.pi, n_bins)
        p_thr = self.params.getfloat("p_thr")

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

                bhv_fits[monkey] = stcc.fit_tcc_swap_model(
                    np.array(targs_all),
                    np.array(dists_all),
                    np.array(resps_all),
                    lr=lr,
                    n_steps=n_steps,
                )
                kern_fits[monkey] = sns.compute_continuous_distance_matrix(
                    *data[monkey],
                    color_key=self.params.get("use_color"),
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
            funcs = np.exp(-(use_bs**2) / wids)
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

            kf_m = kern_fits[monkey]
            ys, xs = sns.compute_continuous_distance_masks(
                kf_m, p_thr=p_thr, n_bins=n_bins
            )[0]
            ys_mu = np.nanmean(ys, axis=0, keepdims=True)
            scaler = skp.MinMaxScaler().fit(ys_mu.T)
            ys_scale = np.squeeze(
                np.array(list(scaler.transform(y_i.reshape((-1, 1))) for y_i in ys))
            )
            gpl.plot_trace_werr(
                xs, ys_scale, ax=axs_kern[i], color=color, label="neural kernel"
            )
            axs_kern[i].set_xlabel("stimulus difference")
            gpl.clean_plot(axs_kern[i], i)
        gpl.make_yaxis_scale_bar(
            axs_kern[0],
            double=False,
            magnitude=0.2,
            label="similarity (au)",
            text_buff=0.3,
        )


class SingleTrialFigure(KernelFigure):
    def __init__(self, fig_key="single-trial", colors=swf.colors, **kwargs):
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
        axs = self.get_axs(kernels_grid, sharey="horizontal", sharex="all", squeeze=True)
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
        ax_targ.set_xlabel("target color\ndifference")
        ax_resp.set_xlabel("response color\ndifference")
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

    def panel_kernels(self):
        key = "panel_kernels"
        axs = self.gss[key]
        c_color = self.params.getcolor("correct_color")
        g_color = self.params.getcolor("guess_color")
        colors = (c_color, g_color)

        n_bins = self.params.getint("n_bins")
        p_thr = self.params.getfloat("p_thr")
        color_keys = ("c_targ", "rc")
        plot_inds = {0: "correct", 2: "guess"}
        data = self.load_pickles()
        for i, monkey in enumerate(self.monkeys):
            data_m = data[monkey]
            for j, ck in enumerate(color_keys):
                out = sns.compute_continuous_distance_matrix(
                    *data_m,
                    color_key=ck,
                )

                mask_dict = sns.compute_continuous_distance_masks(
                    out, p_thr=p_thr, n_bins=n_bins
                )
                ax = axs[j, i]
                if j == 0:
                    if i == len(self.monkeys) - 1:
                        ax.set_xlabel("target color\ndifference")
                    ax.set_ylabel("similarity")
                else:
                    if i == len(self.monkeys) - 1:
                        ax.set_xlabel("response color\ndifference")
                gpl.clean_plot(ax, j)
                for k, (ind, label) in enumerate(plot_inds.items()):
                    ys, xs = mask_dict[ind]
                    gpl.plot_trace_werr(xs, ys, ax=ax, label=label, color=colors[k])
