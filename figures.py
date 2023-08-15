import numpy as np
import pickle

import general.plotting as gpl
import general.paper_utilities as pu
import general.utility as u
import general.data_io as gio
import swap_errors.visualization as swv
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan

config_path = "swap_errors/figures.conf"

colors = (
    np.array(
        [
            (127, 205, 187),
            (65, 182, 196),
            (29, 145, 192),
            (34, 94, 168),
            (37, 52, 148),
            (8, 29, 88),
        ]
    )
    / 256
)


def _color_mask(data, thr=np.pi, color="upper_color"):
    cols = gio.ResultSequence(
        u.normalize_periodic_range(data[color] - thr, convert_array=False)
    )
    mask_c1 = cols > 0
    mask_c2 = cols <= 0
    return mask_c1, mask_c2


def _upper_color_mask(data, thr=np.pi):
    return _color_mask(data, thr=thr, color="upper_color")


def _lower_color_mask(data, thr=0):
    return _color_mask(data, thr=thr, color="lower_color")


def _cue_mask(data):
    mask_c1 = data["IsUpperSample"] == 1
    mask_c2 = data["IsUpperSample"] == 0
    return mask_c1, mask_c2


def _target_color_mask(data, thr=np.pi):
    upper_cat, _ = _color_mask(data, thr=thr, color="upper_color")
    lower_cat, _ = _color_mask(data, thr=thr, color="lower_color")
    filt_mask = upper_cat.rs_xor(lower_cat)
    mask_c1, mask_c2 = _color_mask(data, thr=thr, color="LABthetaTarget")
    mask_c1 = mask_c1.rs_and(filt_mask)
    mask_c2 = mask_c2.rs_and(filt_mask)
    return mask_c1, mask_c2


class SwapErrorFigure(pu.Figure):
    def _make_color_dict(self, ks):
        return self._make_param_dict(ks)

    def _make_param_dict(self, ks, add="_color", func=None):
        if func is None:
            func = self.params.getcolor
        color_dict = {}
        for k in ks:
            color_dict[k] = func(k + add)
        return color_dict

    @property
    def monkeys(self):
        return (self.params.get("monkey1"), self.params.get("monkey2"))

    @property
    def monkey_names(self):
        return self._make_param_dict(self.monkeys, add="_name", func=self.params.get)

    @property
    def monkey_colors(self):
        return self._make_color_dict(self.monkeys)

    @property
    def bhv_outcomes(self):
        return (
            self.params.get("bhv_outcome1"),
            self.params.get("bhv_outcome2"),
            self.params.get("bhv_outcome3"),
        )

    @property
    def bhv_colors(self):
        return self._make_color_dict(self.bhv_outcomes)


class BehaviorFigure(SwapErrorFigure):
    def __init__(self, fig_key="bhv", colors=colors, **kwargs):
        fsize = (8, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        task_schem_grid = self.gs[:, :25]
        gss["panel_task_schematic"] = self.get_axs((task_schem_grid,))

        err_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100, 35, 50, 10, 2)
        gss["panel_err_distribution"] = self.get_axs(err_grid, sharex=True, sharey=True)

        model_schem_grid = self.gs[:, 55:75]
        gss["panel_model_schematic"] = self.get_axs((model_schem_grid,))

        r_simp_gs = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 100, 80, 100, 10, 0)
        r_simp_3d = np.zeros_like(r_simp_gs, dtype=bool)
        r_simp_3d[1] = False
        err_rate_ax, simp_ax = self.get_axs(r_simp_gs, plot_3ds=r_simp_3d)

        gss["panel_err_rates"] = err_rate_ax[0]
        gss["panel_trial_simplex"] = simp_ax[0]

        self.gss = gss

    def _get_bhv_model(self):
        bhv_model_out = self.data.get("bhv_model_out")
        if bhv_model_out is None:
            bhv_model_out = pickle.load(open(self.params.get("bhv_model_path"), "rb"))
            self.data["bhv_model_out"] = bhv_model_out
        return bhv_model_out

    def panel_err_distribution(self):
        key = "panel_err_distribution"
        axs = self.gss[key]

        m_dict = self._get_bhv_model()
        monkeys = self.monkeys
        m_colors = self.monkey_colors
        for i, mi in enumerate(monkeys):
            model_err = m_dict[mi][0].posterior_predictive["err_hat"].to_numpy()
            err = m_dict[mi][1]["err"]
            dist_err = m_dict[mi][1]["dist_err"]
            swv.plot_error_swap_distribs_err(
                err,
                dist_err,
                axs=axs[i],
                model_data=model_err,
                model_derr=dist_err,
                label=self.monkey_names[mi],
                color=m_colors[mi],
            )
            for ax_ij in axs[i]:
                ax_ij.set_xticks([-np.pi, 0, np.pi])
                ax_ij.set_xlabel("")
                ax_ij.set_xticklabels([])
        axs[i, 0].set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axs[i, 1].set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axs[i, 0].set_xlabel("error")
        axs[i, 1].set_xlabel("distractor\ndistance")

    def panel_err_rates(self):
        key = "panel_err_rates"
        ax = self.gss[key]

        m_dict = self._get_bhv_model()
        models = []
        cols = []
        names = []
        monkey_colors = {}
        for m in self.monkeys:
            model_m = m_dict[m][0].posterior
            models.append(model_m)
            names.append(self.monkey_names[m])
            monkey_colors[self.monkey_names[m]] = self.monkey_colors[m]
            self._save_trial_type_stats(model_m, self.monkey_names[m])
        swv.plot_model_probs(
            *models, colors=cols, ax=ax, arg_names=names, monkey_colors=monkey_colors
        )

    def _save_trial_type_stats(self, m, monkey):
        swm = m["swap_weight_mean"].to_numpy()
        gwm = m["guess_weight_mean"].to_numpy()
        o_sum = 1 + np.exp(swm) + np.exp(gwm)
        swap_prob = np.concatenate(np.exp(swm) / o_sum, axis=0)
        guess_prob = np.concatenate(np.exp(gwm) / o_sum, axis=0)
        corr_prob = 1 - (swap_prob + guess_prob)
        corr_int = u.conf_interval(corr_prob, withmean=True)
        guess_int = u.conf_interval(guess_prob, withmean=True)
        swap_int = u.conf_interval(swap_prob, withmean=True)
        s = """{monkey}:
        correct probability = \SIrange{{{clow:.2f}}}{{{chigh:.2f}}}{{}},
        swap probability = \SIrange{{{slow:.2f}}}{{{shigh:.2f}}}{{}},
        guess probability = \SIrange{{{glow:.2f}}}{{{ghigh:.2f}}}{{}}"""
        s = s.format(monkey=monkey,
                     clow=corr_int[1, 0],
                     chigh=corr_int[0, 0],
                     slow=swap_int[1, 0],
                     shigh=swap_int[0, 0],
                     glow=guess_int[1, 0],
                     ghigh=guess_int[0, 0])
        mname = monkey.replace(" ", "_")
        self.save_stats_string(s, "trial-types_{}".format(mname))

    def panel_trial_simplex(self):
        key = "panel_trial_simplex"
        ax = self.gss[key]

        m = self.params.get("simplex_monkey")
        session_date = self.params.get("simplex_session")
        if self.data.get(key) is None:
            simpl = pickle.load(open(self.params.get("bhv_simplex_path"), "rb"))
            self.data[key] = simpl
        simplex = self.data[key]
        simplex_labels = self.bhv_outcomes
        simplex_colors = tuple(self.bhv_colors[k] for k in simplex_labels)
        sd = simplex[(m, session_date)]
        sd_arr = np.stack(list(v[1] for v in sd), axis=0)
        swv.visualize_simplex_2d(
            sd_arr, ax=ax, colors=simplex_colors, ax_labels=simplex_labels
        )


class SingleUnitFigure(SwapErrorFigure):
    def __init__(self, fig_key="single_neurons", colors=colors, **kwargs):
        fsize = (7.5, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        n_egs = len(self.params.getlist("neur_inds"))

        # brain_schem_grid = self.gs[:33, :33]
        # gss['panel_brain_schem'] = self.get_axs((brain_schem_grid,))

        offset = 10
        vert = 40
        wid = 60
        v_spacing = 3
        h_spacing = 3
        su_retro_tr_grids = pu.make_mxn_gridspec(
            self.gs, n_egs, 3, offset, offset + vert, 0, wid, v_spacing, h_spacing
        )
        su_tr_axs = self.get_axs(su_retro_tr_grids, sharex=True, sharey="horizontal")
        su_tune_axs = np.zeros_like(su_tr_axs, dtype=object)
        for ind in u.make_array_ind_iterator(su_tr_axs.shape):
            if ind[1] == 0:
                s_y = None
            else:
                s_y = su_tune_axs[ind[0], 0]
            su_tune_axs[ind] = su_tr_axs[ind].inset_axes(
                (0.6, 0.6, 0.4, 0.4),
                polar=True,
                sharey=s_y,
            )
        gss["panel_su_retro_examples"] = (su_tr_axs, su_tune_axs)

        offset = 60
        su_pro_tr_grids = pu.make_mxn_gridspec(
            self.gs, n_egs, 3, offset, offset + vert, 0, wid, v_spacing, h_spacing
        )

        su_tr_axs = self.get_axs(su_pro_tr_grids, sharex=True, sharey="horizontal")
        su_tune_axs = np.zeros_like(su_tr_axs, dtype=object)
        for ind in u.make_array_ind_iterator(su_tr_axs.shape):
            if ind[1] == 0:
                s_y = None
            else:
                s_y = su_tune_axs[ind[0], 0]
            su_tune_axs[ind] = su_tr_axs[ind].inset_axes(
                (0.6, 0.6, 0.4, 0.4),
                polar=True,
                sharey=s_y,
            )
        gss["panel_su_pro_examples"] = (su_tr_axs, su_tune_axs)

        pop_retro_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 50, 60, 100, 2, 2)
        su_axs = self.get_axs(pop_retro_grids, all_3d=True)
        gss["panel_pop_retro_examples"] = su_axs

        pop_pro_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 50, 100, 60, 100, 2, 2)
        plot_3ds = np.ones((2, 2), dtype=bool)
        plot_3ds[0, 0] = False
        su_axs = self.get_axs(pop_pro_grids, plot_3ds=plot_3ds)
        gss["panel_pop_pro_examples"] = su_axs

        self.gss = gss

    def _get_experimental_data(self):
        if self.exp_data is None:
            max_files = np.inf
            df = "../data/swap_errors/"
            data = gio.Dataset.from_readfunc(
                swa.load_buschman_data,
                df,
                max_files=max_files,
                seconds=True,
                load_bhv_model="../data/swap_errors/bhv_model.pkl",
                spks_template=swa.busch_spks_templ_mua,
            )
            self.exp_data = data
        return self.exp_data

    def _su_examples(self, axs, use_retro=True, plot_colors=True):
        tr_axs, tune_axs = axs

        data = self._get_experimental_data()
        date = self.params.get("eg_date")
        neur_inds = self.params.getlist("neur_inds", typefunc=int)

        upper_color = self.params.getcolor("upper_color")
        lower_color = self.params.getcolor("lower_color")
        target_color = self.params.getcolor("target_color")
        distr_color = self.params.getcolor("distr_color")

        cue_color = self.params.getcolor("cue_color")

        _ = swv.plot_period_units_trace(
            data,
            date,
            neur_inds,
            plot_colors=plot_colors,
            use_retro=use_retro,
            axs=tr_axs,
            default_upper_color=upper_color,
            default_lower_color=lower_color,
            default_target_color=target_color,
            default_distr_color=distr_color,
            default_cue_color=cue_color,
        )
        _ = swv.plot_period_units_tuning(
            data,
            date,
            neur_inds,
            use_retro=use_retro,
            axs=tune_axs,
            default_upper_color=upper_color,
            default_lower_color=lower_color,
            default_target_color=target_color,
            default_distr_color=distr_color,
            default_cue_color=cue_color,
        )

    def _pop_examples(self, axs, use_retro=True, plot_colors=True, **kwargs):
        data = self._get_experimental_data()
        date = self.params.get("eg_date")

        corr_color = self.params.getcolor("correct_color")
        swap_color = self.params.getcolor("swap_color")
        swv.plot_population_toruses(
            data,
            date,
            use_retro=use_retro,
            corr_color=corr_color,
            swap_color=swap_color,
            axs=axs,
            **kwargs
        )

    def panel_su_retro_examples(self):
        key = "panel_su_retro_examples"
        axs = self.gss[key]
        self._su_examples(axs)

    def panel_su_pro_examples(self):
        key = "panel_su_pro_examples"
        axs = self.gss[key]
        self._su_examples(axs, use_retro=False)

    def panel_pop_retro_examples(self):
        key = "panel_pop_retro_examples"
        axs = self.gss[key]
        self._pop_examples(axs)

        axs[0, 0].view_init(40, 30)
        axs[0, 1].view_init(50, 20)
        axs[1, 0].view_init(50, 10)
        axs[1, 1].view_init(50, 10)

    def panel_pop_pro_examples(self):
        key = "panel_pop_pro_examples"
        axs = self.gss[key]

        cue_color = self.params.getcolor("cue_color")
        self._pop_examples(axs, use_retro=False, default_cue_color=cue_color)

        axs[0, 1].view_init(20, 30)
        axs[1, 0].view_init(50, 30)
        axs[1, 1].view_init(50, 30)


class LMFigure(SwapErrorFigure):
    def load_all_runs(self, reload_data=False):
        run_inds = tuple(self.params.getlist("run_inds"))
        folder = self.params.get("lm_folder")
        if self.data.get((run_inds, folder)) is None or reload_data:
            out = swa.load_lm_results(run_inds, folder)
            self.data[(run_inds, folder)] = out
        return self.data[(run_inds, folder)]

    def _plot_lm_dict(self, *args, **kwargs):
        return self._plot_cue_dict(
            *args, **kwargs, plot_func=swv.plot_lm_tc, set_ticks=True
        )

    def _plot_cue_dict(
            self,
            trial_type,
            data_dict,
            colors_null,
            colors_alt,
            mat_inds_all,
            axs,
            plot_func=swv.plot_cue_tc,
            set_ticks=False,
    ):
        key_order = {
            'pro': ('cue', 'pre-color', 'post-color', 'wheel'),
            'retro': ('color', 'pre-cue', 'post-cue', 'wheel'),
        }

        e_name = self.params.get("Elmo_name")
        w_name = self.params.get("Waldorf_name")

        m_names = (e_name, w_name)
        monkeys = ("Elmo", "Wald")
        for i, m in enumerate(monkeys):
            fj = len(mat_inds_all)*i
            for j, mat_inds in enumerate(mat_inds_all):
                ind = fj + j
                plot_func(
                    data_dict[m][trial_type],
                    key_order=key_order[trial_type],
                    axs=np.expand_dims(axs[ind], 0),
                    null_colors=colors_null[j],
                    swap_colors=colors_alt[j],
                    mat_inds=mat_inds,
                )
                if ind < len(axs) - 1:
                    list(gpl.clean_plot_bottom(ax) for ax in axs[ind])
                    list(ax.set_xlabel("") for ax in axs[ind])
            if set_ticks:
                axs[ind, 0].set_yticks([0, .5, 1])
                axs[ind, 0].set_ylim([0, 1])
            self._save_cv_stats(
                data_dict[m][trial_type], m_names[i],
            )
        return axs

    def _save_cv_stats(self, data, monkey):
        if self.trial_type == "pro":
            stat_dict = {
                ("cue", "correct", "none"): (.25, (0, 1)),
                ("cue", "correct", "cue interpretation error"): (.25, (0, 1)),
                ("pre-color", "correct", "none"): (-.25, (0, 1)),
                ("pre-color", "correct", "cue interpretation error"): (-.25, (0, 1)),
                ("post-color", "correct", "misbinding error"): (.25, (0, 1)),
                ("post-color", "correct", "cue selection error"): (.25, (0, 2)),
                ("wheel", "correct", "misbinding error"): (-.25, (0, 1)),
                ("wheel", "correct", "cue selection error"): (-.25, (0, 2)),
            }
        else:
            stat_dict = {
                ("color", "correct", "misbind"): (.25, (0, 1)),
                ("color", "correct", "none"): (.25, (0, 1)),
                ("pre-cue", "correct", "misbind"): (-.25, (0, 1)),
                ("pre-cue", "correct", "none"): (-.25, (0, 1)),
                ("post-cue", "correct", "color selection error"): (.25, (0, 1)),
                ("post-cue", "correct", "cue interpretation error"): (.25, (0, 2)),
                ("wheel", "correct", "color selection error"): (-.25, (0, 1)),
                ("wheel", "correct", "cue interpretation error"): (-.25, (0, 2)),
            }
            
        for (k, t1, t2), (t_pt, pt) in stat_dict.items():
            t2_save = t2.split(" ")[0]
            (nc_comb, sc_comb), (nc_indiv, sc_indiv), xs = data[k]
            t_ind = np.argmin((xs - t_pt)**2)
            nc_comb = nc_comb[..., t_ind]
            sc_comb = sc_comb[..., t_ind]
            if len(nc_comb.shape) == 1 or nc_comb.shape[0] == 0:
                l_proto = nc_comb
                r_proto = sc_comb
                t1_t2_str = "cue-dec"
                diff_str = "correct rather than incorrect cue"
            else:
                r_proto = nc_comb[pt]
                l_proto = sc_comb[pt]
                t1_t2_str = "{}-{}".format(t1, t2_save)
                diff_str = "{type2} than {type1} prototype".format(type2=t2, type1=t1)

            if nc_comb.shape[0] != 0:
                diffs = u.bootstrap_diff(l_proto, r_proto)
                high, low = u.conf_interval(diffs, withmean=True)[:, 0]
                diff_range = u.format_sirange(high, low)
            
                s = "{monkey}: {diffs} closer to the {diff_str}"
                s = s.format(monkey=monkey, diffs=diff_range, diff_str=diff_str)
                cv_name = "cv_{}_{}_{}_{}".format(
                    self.trial_type, monkey.replace(" ", "_"), k, t1_t2_str
                )
                self.save_stats_string(s, cv_name)
    
    def make_gss(self):
        gss = {}

        horiz_gap = 5
        vert_gap = 10

        vert_pt = int(200/3)
        
        lm_gs = pu.make_mxn_gridspec(
            self.gs, 4, 4, 0, vert_pt - vert_gap/2, 0, 100, 5, horiz_gap
        )
        lm_axs = self.get_axs(lm_gs, sharey="all", sharex="vertical")

        cue_gs = pu.make_mxn_gridspec(
            self.gs, 2, 4, vert_pt + vert_gap/2, 100, 0, 100, 5, horiz_gap
        )
        cue_axs = self.get_axs(cue_gs, sharey="horizontal", sharex="vertical")

        gss["panel_color_cue"] = (lm_axs, cue_axs)

        self.gss = gss
        
    def panel_color_cue(self):
        key = "panel_color_cue"
        color_axs, cue_axs = self.gss[key]
        fd, color_dict, cue_dict = self.load_all_runs()

        colors_null, colors_alt = self.get_lm_plot_colors()        
        mat_inds_lm = (((0, 1), (0, 1), (0, 1), (0, 1)),
                       ((0, 1), (0, 1), (0, 2), (0, 2)))

        self._plot_lm_dict(
            self.trial_type,
            color_dict,
            colors_null,
            colors_alt,
            mat_inds_lm,
            color_axs,
        )

        colors_null, colors_alt = self.get_cue_plot_colors()        
        mat_inds_cue = (((0, 1), (0, 1), (0, 1), (0, 1)),)
        self._plot_cue_dict(
            self.trial_type,
            cue_dict,
            colors_null,
            colors_alt,
            mat_inds_cue,
            cue_axs,
        )


class NeuronNumFigure(SwapErrorFigure):
    def __init__(self, fig_key="nnf", colors=colors, **kwargs):
        fsize = (4, 7)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        self.gss = gss

    def save_neuron_nums(self, data=None):
        if data is None:
            data = self._get_experimental_data()
        regions = data["neur_regions"]

        region_counts = {}
        total_counts = []
        for session in regions:
            sess_regions = session.iloc[0]
            rs, cs = np.unique(sess_regions, return_counts=True)
            for i, r in enumerate(rs):
                r_i = region_counts.get(r, [])
                r_i.append(cs[i])
                region_counts[r] = r_i
            total_counts.append(len(sess_regions))
        label_dict = {
            "7ab": "posterior parietal cortex",
            "motor": "motor cortex",
            "pfc": "prefrontal cortex",
            "v4pit": "V4 and posterior IT",
        }
        s_base = "{label}: \SIrange{{{min_num}}}{{{max_num}}}{{}} units"
        s_list = list(
            s_base.format(label=l,
                          min_num=np.min(region_counts[k]),
                          max_num=np.max(region_counts[k]))
            for k, l in label_dict.items()
        )
        full_str = "; ".join(s_list)
        self.save_stats_string(full_str, "neuron_numbers_regions")

        s_total = "combined: \SIrange{{{min_num}}}{{{max_num}}}{{}} units"
        s_total = s_total.format(min_num=np.min(total_counts),
                                 max_num=np.max(total_counts))
        self.save_stats_string(s_total, "neuron_numbers_total")
        

class RetroLMFigure(LMFigure):
    def __init__(self, fig_key="retro_lm", colors=colors, **kwargs):
        fsize = (4, 7)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        self.trial_type = "retro"
        super().__init__(fsize, params, colors=colors, **kwargs)

    def get_cue_plot_colors(self):
        corr_color = self.params.getcolor("correct_color")
        cue_interp_color = self.params.getcolor("cue_interp_color")
        
        colors_alt = ((cue_interp_color, cue_interp_color,
                       cue_interp_color, cue_interp_color),)
        colors_null = ((corr_color, corr_color,
                        corr_color, corr_color),)
        return colors_null, colors_alt

    def get_lm_plot_colors(self):
        corr_color = self.params.getcolor("correct_color")
        color11 = self.params.getcolor("misbinding_color")
        color21 = self.params.getcolor("selection_color")
        color22 = self.params.getcolor("cue_interp_color")
        color31 = self.params.getcolor("selection_color")
        color32 = self.params.getcolor("cue_interp_color")
        colors_alt = ((color11, color11, color21, color31),
                      (color11, color11, color22, color32))
        colors_null = ((corr_color, corr_color, corr_color, corr_color),
                       (corr_color, corr_color, corr_color, corr_color))
        return colors_null, colors_alt


class ProLMFigure(LMFigure):
    def __init__(self, fig_key="pro_lm", colors=colors, **kwargs):
        fsize = (4, 7)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        self.trial_type = "pro"
        super().__init__(fsize, params, colors=colors, **kwargs)

    def get_cue_plot_colors(self):
        corr_color = self.params.getcolor("correct_color")
        cue_interp_color = self.params.getcolor("cue_interp_color")
        cue_sel_color = self.params.getcolor("selection_color")
        
        colors_alt = ((cue_interp_color, cue_interp_color,
                       cue_sel_color, cue_sel_color),)
        colors_null = ((corr_color, corr_color, corr_color, corr_color),)
        return colors_null, colors_alt

    def get_lm_plot_colors(self):
        corr_color = self.params.getcolor("correct_color")
        color11 = self.params.getcolor("cue_interp_color")
        color21 = self.params.getcolor("misbinding_color")
        color22 = self.params.getcolor("selection_color")
        color31 = self.params.getcolor("misbinding_color")
        color32 = self.params.getcolor("selection_color")
        
        colors_alt = ((color11, color11, color21, color31),
                      (color11, color11, color22, color32))
        colors_null = ((corr_color, corr_color, corr_color, corr_color),
                       (corr_color, corr_color, corr_color, corr_color))
        return colors_null, colors_alt

    
class ModelBasedFigure(SwapErrorFigure):
    def _get_d1_pro_fits(self):
        n_colors = self.params.get("d1_load_n_colors")
        spline_order = self.params.get("d1_load_spline_order")
        session_split = self.params.getint("session_split")
        total_sessions = self.params.getint("total_sessions")

        t_beg = self.params.getfloat("d1_t_beg")
        t_end = self.params.getfloat("d1_t_end")

        e_name = self.params.get("Elmo_name")
        w_name = self.params.get("Waldorf_name")

        impute = self.params.getboolean("d1_impute")

        elmo_sessions = range(session_split)
        elmo_data = swa.load_pro_d1_stan_data(
            session_range=elmo_sessions,
            n_colors=n_colors,
            spline_order=spline_order,
            start=t_beg,
            end=t_end,
            impute=impute,
        )

        wald_sessions = range(session_split, total_sessions)
        wald_data = swa.load_pro_d1_stan_data(
            session_range=wald_sessions,
            n_colors=n_colors,
            spline_order=spline_order,
            start=t_beg,
            end=t_end,
            impute=impute,
        )

        full_dict = {
            (e_name, "pro d1", "joint"): elmo_data,
            (w_name, "pro d1", "joint"): wald_data,
        }
        return full_dict, elmo_data, wald_data

    def _save_kind_diff_stats(self, fits, monkey, task, 
                              pos_ind=0, neg_ind=1,
                              type1="color selection",
                              type2="cue interpretation"):
        kind_diffs = []
        nz = 0
        for k, (fit, data) in fits.items():
            probs = np.concatenate(fit['other'].posterior["p_err"])
            type_ind = swa.get_type_ind(task, data)
            probs = probs[:, type_ind]
            diff = probs[:, pos_ind] - probs[:, neg_ind]
            interv = u.conf_interval(diff, withmean=True)
            nz += np.all(interv > 0)
            kind_diffs.append(np.mean(diff))
        avg_diffs = u.bootstrap_list(np.array(kind_diffs), np.nanmean)
        ad_interv = u.conf_interval(avg_diffs, withmean=True)[:, 0]
        avg_diffs_str = u.format_sirange(*ad_interv)
        s = ("{monkey}: {avg_diffs} greater probability of "
             "{type1} than {type2} errors")
        s = s.format(monkey=monkey, avg_diffs=avg_diffs_str,
                     type1=type1, type2=type2)
        mname = monkey.replace(" ", "_")
        self.save_stats_string(s, "kind-diff_{}_{}".format(task, mname))
    
    def _save_rate_stats(self, fits, monkey, delay, task, t_ind=True,
                         thresh=.1, use_ind=-1):
        no_zero = 0
        probs_all = []
        for k, (fit, data) in fits.items():
            probs = np.concatenate(fit['other'].posterior["p_err"])
            if t_ind:
                type_ind = swa.get_type_ind(task, data)
                probs = probs[:, type_ind]
            probs = 1 - probs[:, use_ind]
            probs_all.append(np.mean(probs))
            interv = u.conf_interval(probs, withmean=True)
            no_zero += np.all(interv > thresh)
        ps_all = u.bootstrap_list(np.array(probs_all), np.nanmean)
        ps_range = u.format_sirange(*u.conf_interval(ps_all, withmean=True)[:, 0])
        mname = monkey.replace(" ", "_")
        s1 = "{monkey}: {ps_range} average probability of error"
        s1 = s1.format(monkey=monkey, ps_range=ps_range)
        self.save_stats_string(s1, "range-rate_{}_{}_{}".format(task, delay, mname))
        
        s2 = "{monkey}: significantly greater than {thresh} in {nz}/{tot} sessions"
        s2 = s2.format(monkey=monkey, nz=no_zero, thresh=thresh, tot=len(fits))
        self.save_stats_string(s2, "nz-rate_{}_{}_{}".format(task, delay, mname))

    def _save_rate_diff_stats(self, diff_bootstrap, monkey, task):
        high, low = u.conf_interval(diff_bootstrap, withmean=True)[:, 0]
        diff = u.format_sirange(high, low)
        s = "{monkey}: {diff} greater in delay 2 than delay 1"
        s = s.format(monkey=monkey, diff=diff)
        mname = monkey.replace(" ", "_")
        self.save_stats_string(s, "nz-rate_{}_{}".format(task, mname))

    def _save_decoding_rates(self, rates, monkey, key, task):
        diffs = []
        for k, err_tuple in rates.items():
            c_err, s_err = err_tuple[:2]
            diffs.append(np.mean(s_err) - np.mean(c_err))
        mu_diff = u.bootstrap_list(np.array(diffs), np.nanmean)
        dec_diff = u.format_sirange(*u.conf_interval(mu_diff, withmean=True)[:, 0])

        s = "{monkey}: {diff} difference in decoding performance, swap - correct"
        s = s.format(monkey=monkey, diff=dec_diff)
        mname = monkey.replace(" ", "_")

        self.save_stats_string(s, "dec-diff_{}_{}_{}".format(mname, key, task))

    def _save_monkey_dec_diff(self, m1_rates, m2_rates, key, task, m1="Monkey E",
                              m2="W", m_name=None, t_ind=0, suffix=""):
        m1_mus = []
        for k, err_tuple in m1_rates.items():
            c_err = err_tuple[t_ind]
            m1_mus.append(np.mean(c_err))
        m2_mus = []
        for k, err_tuple in m2_rates.items():
            c_err = err_tuple[t_ind]
            m2_mus.append(np.mean(c_err))
        diffs = u.bootstrap_diff(np.array(m1_mus), np.array(m2_mus))
        s = "{diff} higher cue decoding performance{suffix} in {m1} than {m2}"
        if m_name is not None:
            m_f = m_name.replace(" ", "_")
            key = "_".join((key, m_f, str(t_ind)))
            s = "{}: ".format(m_name) + s
        diff_str = u.format_sirange(*u.conf_interval(diffs, withmean=True)[:, 0])
        s = s.format(diff=diff_str, m1=m1, m2=m2, suffix=suffix)
        self.save_stats_string(s, "dec-mdiff_{}_{}".format(key, task))
        
    def get_model_dict(self, ri, period):
        if self.data.get((ri, period)) is None:
            self.data[(ri, period)] = self._get_model_dict(ri, period)
        return self.data[(ri, period)]

    def _panel_decoding(
        self,
        e_fits,
        w_fits,
        dec_ax,
        diff_ax,
        key=None,
        refit=False,
        type_str="retro",
        corr_ind=0,
        swap_ind=1,
        func=swan.cue_decoding_swaps,
    ):
        corr_thr = self.params.getfloat("corr_thr")
        swap_thr = self.params.getfloat("swap_thr")

        if self.data.get(key) is None or refit:
            e_rates = func(
                e_fits,
                corr_thr,
                swap_thr,
                type_str=type_str,
                corr_ind=corr_ind,
                swap_ind=swap_ind,
            )
            w_rates = func(
                w_fits,
                corr_thr,
                swap_thr,
                type_str=type_str,
                corr_ind=corr_ind,
                swap_ind=swap_ind,
            )
            self.data[key] = (e_rates, w_rates)
        e_rates, w_rates = self.data[key]

        self._save_decoding_rates(e_rates, "Monkey E", key, type_str)
        self._save_decoding_rates(w_rates, "Monkey W", key, type_str)
        self._save_monkey_dec_diff(e_rates, w_rates, key, type_str)
        swv.plot_cue_decoding(
            e_rates, axs=(dec_ax, diff_ax), color=self.monkey_colors["Elmo"]
        )
        swv.plot_cue_decoding(
            w_rates,
            axs=(dec_ax, diff_ax),
            color=self.monkey_colors["Waldorf"],
            x_cent=1,
        )
        gpl.clean_plot(diff_ax, 0)
        diff_ax.set_xticks([0, 1])
        diff_ax.set_xticklabels(["E", "W"])
        gpl.clean_plot_bottom(diff_ax, keeplabels=True)
        diff_ax.set_xticks([0, 1])

        dec_ax.plot(
            [0.4, 1],
            [0.4, 1],
            ls="dashed",
            color="k",
        )
        gpl.add_hlines(0, diff_ax)
        gpl.add_hlines(0.5, dec_ax)
        gpl.add_vlines(0.5, dec_ax)
        dec_ax.set_aspect("equal")
        names = ("corr", "swap", "guess")
        dec_ax.set_xlabel(
            "decoding performance\np({}) > {}".format(
                names[corr_ind],
                corr_thr,
            )
        )
        dec_ax.set_ylabel(
            "decoding performance\np({}) > {}".format(
                names[swap_ind],
                swap_thr,
            )
        )
        diff_ax.set_ylabel(
            r"$\Delta$"
            + " performance\n({} - {})".format(names[swap_ind], names[corr_ind])
        )
        diff_ax.set_yticks([0, -0.1])

    def get_monkey_d1d2_colors(self):
        e_color = self.monkey_colors["Elmo"]
        w_color = self.monkey_colors["Waldorf"]

        col_diff = self.params.getfloat("d1_d2_color_diff")
        e_d1_color = gpl.add_color_value(e_color, -col_diff)
        e_d2_color = gpl.add_color_value(e_color, col_diff)
        w_d1_color = gpl.add_color_value(w_color, -col_diff)
        w_d2_color = gpl.add_color_value(w_color, col_diff)
        return (e_d1_color, e_d2_color), (w_d1_color, w_d2_color)

    def _get_model_dict(self, ri, period):
        n_colors = self.params.get("n_colors")
        spline_order = self.params.get("spline_order")
        session_split = self.params.getint("session_split")
        total_sessions = self.params.getint("total_sessions")
        fp = self.params.get("model_fit_folder")
        templ = self.params.get("model_fit_template")

        e_name = self.params.get("Elmo_name")
        w_name = self.params.get("Waldorf_name")

        elmo_sessions = range(session_split)
        elmo_fits = swa.load_o_fits(
            ri,
            sess_inds=elmo_sessions,
            load_data=True,
            n_colors=n_colors,
            period=period,
            fit_templ=templ,
            folder=fp,
        )

        wald_sessions = range(session_split, total_sessions)
        wald_fits = swa.load_o_fits(
            ri,
            sess_inds=wald_sessions,
            load_data=True,
            n_colors=n_colors,
            period=period,
            fit_templ=templ,
            folder=fp,
        )
        full_dict = {
            (e_name, period, "joint"): elmo_fits,
            (w_name, period, "joint"): wald_fits,
        }
        return full_dict, elmo_fits, wald_fits

    def get_d1_fits(self, runind_name="d1_runind", period="CUE2_ON"):
        ri = self.params.get(runind_name)
        out = self.get_model_dict(ri, period)
        return out

    def get_d2_fits(self, runind_name="d2_runind"):
        ri = self.params.get(runind_name)
        period = "WHEEL_ON"
        out = self.get_model_dict(ri, period)
        return out

    def plot_ppc_groups(
        self,
        types,
        mistakes,
        axs,
        *m_dicts,
        collapse_correct_error=False,
        plot_inverted=True,
        new_joint=False,
        precomputed_data=None,
        cue_time=False
    ):
        p_thr = self.params.getfloat("model_plot_pthr")
        n_bins = self.params.getint("model_plot_n_bins")

        corr_color = self.params.getcolor("correct_color")
        swap_color = self.params.getcolor("swap_color")

        swap_thr = self.params.getfloat("ppc_thr")

        out_data = {}
        max_ys = []
        for i, md in enumerate(m_dicts):
            if collapse_correct_error:
                corr_ind = i
                swap_ind = i
                i_beg = 0
                i_end = len(mistakes)
            elif plot_inverted:
                corr_ind = i * 2
                swap_ind = i * 2 + 1
                i_beg = 0
                i_end = len(mistakes)
            else:
                corr_ind = 1
                swap_ind = 0
                i_beg = i * len(mistakes)
                i_end = (i + 1) * len(mistakes)
            swap_ax_arr = np.expand_dims(axs[swap_ind, i_beg:i_end], (0, 1))
            if precomputed_data is None:
                swap_precomp = None
                corr_precomp = None
            else:
                swap_precomp, corr_precomp = precomputed_data[i]
            _, swap_data = swv.plot_dists(
                (swap_thr,),
                types,
                md,
                n_bins=n_bins,
                mistakes=mistakes,
                ret_data=True,
                p_comp=np.greater,
                axs_arr=swap_ax_arr,
                new_joint=new_joint,
                color=swap_color,
                precomputed_data=swap_precomp,
                cue_time=cue_time,
                simple_label=True,
                legend_label="p(swap) > {}".format(swap_thr),
            )

            corr_ax_arr = np.expand_dims(axs[corr_ind, i_beg:i_end], (0, 1))
            _, corr_data = swv.plot_dists(
                (swap_thr,),
                types,
                md,
                n_bins=n_bins,
                mistakes=mistakes,
                ret_data=True,
                p_comp=np.less,
                axs_arr=corr_ax_arr,
                new_joint=new_joint,
                legend=True,
                pred_label=False,
                color=corr_color,
                precomputed_data=corr_precomp,
                cue_time=cue_time,
                simple_label=True,
                legend_label="p(swap) < {}".format(swap_thr),
            )
            out_data[i] = (swap_data, corr_data)
            if plot_inverted:
                for ind in u.make_array_ind_iterator(swap_ax_arr.shape):
                    max_ys.append(swap_ax_arr[ind].get_ylim()[-1])
                    swap_ax_arr[ind].invert_yaxis()
                for ind in u.make_array_ind_iterator(corr_ax_arr.shape):
                    max_ys.append(corr_ax_arr[ind].get_ylim()[-1])
        max_y = np.max(max_ys)
        for ind in u.make_array_ind_iterator(axs.shape):
            axs[ind].set_xlabel("")
            if ind[0] < (len(axs) - 1):
                gpl.clean_plot_bottom(axs[ind])
            else:
                gpl.clean_plot_bottom(axs[ind], keeplabels=True)
            if axs[ind].yaxis_inverted():
                axs[ind].set_ylim([max_y, 0])
            else:
                axs[ind].set_ylim([0, max_y])

        return out_data

    def _panel_d2(self, type_str="retro", **kwargs):
        key = "panel_d2"
        ppc_axs, posterior_axs, sess_ax = self.gss[key]

        # (_, e_d2_color), (_, w_d2_color) = self.get_monkey_d1d2_colors()
        e_d2_color = self.params.getcolor("elmo_color")
        w_d2_color = self.params.getcolor("waldorf_color")

        full_dict, elmo_fits, wald_fits = self.get_d2_fits()
        self._save_rate_stats(elmo_fits, "Monkey E", "d2", type_str)
        self._save_rate_stats(wald_fits, "Monkey W", "d2", type_str)
        self._save_kind_diff_stats(elmo_fits, "Monkey E", type_str, **kwargs)
        self._save_kind_diff_stats(wald_fits, "Monkey W", type_str, **kwargs)

        swv.plot_cumulative_simplex(
            elmo_fits, ax=posterior_axs[0], color=e_d2_color, plot_type=type_str
        )
        swv.plot_cumulative_simplex(
            wald_fits, ax=posterior_axs[1], color=w_d2_color, plot_type=type_str
        )

        types = (type_str,)
        mistakes = (
            "spatial",
            "cue",
            "spatial-cue",
        )
        out = self.plot_ppc_groups(
            types,
            mistakes,
            ppc_axs,
            elmo_fits,
            wald_fits,
            new_joint=True,
            precomputed_data=self.data.get(key),
        )

        self.data[key] = out
        self.data["d2_ppc_pts"] = out


class ProSwapFigure(ModelBasedFigure):
    def __init__(self, fig_key="pro_swap", colors=colors, **kwargs):
        fsize = (8.5, 5.2)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        lb = 70
        gap = 10
        ppc_wid = 12
        # delay 1

        dec_scatter_grids = pu.make_mxn_gridspec(self.gs, 2, 1, 0, lb, 0, 15, 5, 2)
        dec_scatter_axs = self.get_axs(
            dec_scatter_grids, sharex="all", sharey="all", squeeze=True
        )
        dec_diff_grids = pu.make_mxn_gridspec(self.gs, 2, 1, 0, lb, 26, 32, 8, 2)
        dec_diff_axs = self.get_axs(
            dec_diff_grids, sharex="all", sharey="all", squeeze=True
        )

        gss["panel_d1"] = (dec_scatter_axs[0], dec_diff_axs[0])
        gss["panel_d2_decoding"] = (dec_scatter_axs[1], dec_diff_axs[1])

        dec_comparison_grid = pu.make_mxn_gridspec(
            self.gs, 1, 2, lb + gap, 100, 0, 40, 5, 10
        )
        gss["panel_decoding_comparison"] = self.get_axs(
            dec_comparison_grid, sharex="all", sharey="all", squeeze=True
        )

        # delay 2
        delt = 100 - ppc_wid * 3 - 4
        ppc_grids = pu.make_mxn_gridspec(self.gs, 4, 3, 0, lb, delt, 100, 2, 2)
        d2_ppc_axs = self.get_axs(ppc_grids)

        post_grids = pu.make_mxn_gridspec(
            self.gs,
            2,
            1,
            0,
            lb,
            34,
            52,
            2,
            4,
        )
        d2_post_axs = self.get_axs(post_grids, squeeze=True, sharey="all")

        gss["panel_d2"] = (d2_ppc_axs, d2_post_axs, None)

        self.gss = gss

    def get_d1_fits(self, reload_=False):
        if self.data.get("pro_d1") is None or reload_:
            self.data["pro_d1"] = self._get_d1_pro_fits()
        return self.data["pro_d1"]

    def panel_d1(self, refit=False, reload_=False):
        key = "panel_d1"
        dec_ax, diff_ax = self.gss[key]
        _, e_fits, w_fits = self.get_d1_fits(reload_=reload_)

        self._panel_decoding(
            e_fits,
            w_fits,
            dec_ax,
            diff_ax,
            key=key,
            type_str="pro",
            refit=(refit or reload_),
        )
        diff_ax.set_yticks([0, -0.2, -0.4])

    def panel_d2_decoding(self):
        key = "panel_d2_decoding"
        axs = self.gss[key]
        _, e_fits, w_fits = self.get_d2_fits()

        self._panel_decoding(e_fits, w_fits, *axs, key=key, type_str="pro")
        axs[1].set_yticks([0, -0.2, -0.4])

    def panel_d2(self):
        self._panel_d2(type_str="pro", pos_ind=1, neg_ind=0,
                       type1="cue selection",
                       type2="misbinding")

    def panel_decoding_comparison(self):
        key = "panel_decoding_comparison"
        axs = self.gss[key]

        d1_e_rates, d1_w_rates = self.data.get("panel_d1")
        d2_e_rates, d2_w_rates = self.data.get("panel_d2_decoding")

        corr_thr = self.params.getfloat("corr_thr")
        swap_thr = self.params.getfloat("swap_thr")

        e_color = self.monkey_colors["Elmo"]
        w_color = self.monkey_colors["Waldorf"]

        self._save_monkey_dec_diff(d1_e_rates, d2_e_rates, key, "pro", m1="delay 1",
                                   m2="2", m_name="Monkey E",
                                   suffix=" on correct trials")
        self._save_monkey_dec_diff(d1_e_rates, d2_e_rates, key, "pro", m1="delay 1",
                                   m2="2", t_ind=1, m_name="Monkey E",
                                   suffix=" on swap trials")
        self._save_monkey_dec_diff(d1_w_rates, d2_w_rates, key, "pro", m1="delay 1",
                                   m2="2", m_name="Monkey W",
                                   suffix=" on correct trials")
        self._save_monkey_dec_diff(d1_w_rates, d2_w_rates, key, "pro", m1="delay 1",
                                   m2="2", t_ind=1, m_name="Monkey W",
                                   suffix=" on swap trials")
        swv.plot_decoding_comparison(
            d1_e_rates,
            d2_e_rates,
            axs=axs,
            color=e_color,
            corr_thr=corr_thr,
            swap_thr=swap_thr,
        )
        swv.plot_decoding_comparison(
            d1_w_rates,
            d2_w_rates,
            axs=axs,
            color=w_color,
            corr_thr=corr_thr,
            swap_thr=swap_thr,
        )
        gpl.add_hlines(0.5, axs[0])
        gpl.add_hlines(0.5, axs[1])
        axs[0].set_yticks([0, 0.5, 1])
        axs[1].set_yticks([0, 0.5, 1])


class GuessFigure(ModelBasedFigure):
    def __init__(self, fig_key="guess", colors=colors, **kwargs):
        fsize = (8.5, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        lb = 45
        wb = 50
        gap = 10
        post_wid = 12
        scatt_wid = 15
        diff_wid = 8

        # delay 1
        post_grids = pu.make_mxn_gridspec(
            self.gs,
            2,
            2,
            0,
            lb,
            40 + post_wid - (post_wid * 2 + 5),
            40 + post_wid,
            8,
            5,
        )
        d1d2_post_axs = self.get_axs(post_grids, squeeze=True, sharey="all")

        gss["panel_retro_d1"] = d1d2_post_axs[:, 0]

        scatter_grid = self.gs[0:lb, 65 : 65 + scatt_wid]
        diff_grid = self.gs[0:lb, 100 - diff_wid :]
        dec_axs = self.get_axs(
            (
                scatter_grid,
                diff_grid,
            ),
            squeeze=True,
        )
        gss["panel_retro_d2"] = (d1d2_post_axs[:, 1], dec_axs)

        scatter_grid = self.gs[lb + gap : 100, 0:scatt_wid]
        diff_grid = self.gs[lb + gap : 100, 34 - diff_wid : 34]
        dec_axs = self.get_axs(
            (
                scatter_grid,
                diff_grid,
            ),
            squeeze=True,
        )
        gss["panel_pro_d1"] = dec_axs

        post_grids = pu.make_mxn_gridspec(
            self.gs,
            2,
            1,
            lb + gap,
            100,
            40,
            40 + post_wid,
            8,
            5,
        )
        d2_post_axs = self.get_axs(post_grids, squeeze=True, sharey="all")

        scatter_grid = self.gs[lb + gap : 100, 65 : 65 + scatt_wid]
        diff_grid = self.gs[lb + gap : 100, 100 - diff_wid :]
        dec_axs = self.get_axs(
            (
                scatter_grid,
                diff_grid,
            ),
            squeeze=True,
        )
        gss["panel_pro_d2"] = (d2_post_axs, dec_axs)

        self.gss = gss

    def panel_retro_d1(self, recompute=False):
        key = "panel_retro_d1"
        posterior_axs = self.gss[key]

        e_d1_color = self.params.getcolor("elmo_color")
        w_d1_color = self.params.getcolor("waldorf_color")

        full_dict, elmo_fits, wald_fits = self.get_d1_fits(
            runind_name="d1_retro_runind"
        )
        swv.plot_cumulative_simplex_1d(
            elmo_fits,
            ax=posterior_axs[0],
            color=e_d1_color,
            simplex_key="p_guess_err",
        )
        swv.plot_cumulative_simplex_1d(
            wald_fits,
            ax=posterior_axs[1],
            color=w_d1_color,
            simplex_key="p_guess_err",
        )
        posterior_axs[0].set_ylabel("density")
        posterior_axs[1].set_ylabel("density")
        posterior_axs[1].set_xlabel("p(guess)")

        # types = (None,)
        # mistakes = ('guess',)
        # if recompute:
        #     precomp = None
        # else:
        #     precomp = self.data.get(key)
        # out = self.plot_ppc_groups(types, mistakes, ppc_axs,
        #                            elmo_fits, wald_fits,
        #                            precomputed_data=precomp,
        #                            cue_time=True)
        # self.data[key] = out
        # self.data['d1_retro_ppc_pts'] = out

    def panel_retro_d2(self, recompute=False, refit=False):
        key = "panel_retro_d2"
        posterior_axs, dec_axs = self.gss[key]

        e_d1_color = self.params.getcolor("elmo_color")
        w_d1_color = self.params.getcolor("waldorf_color")

        full_dict, elmo_fits, wald_fits = self.get_d2_fits()
        swv.plot_cumulative_simplex_1d(
            elmo_fits,
            ax=posterior_axs[0],
            color=e_d1_color,
            simplex_key="p_guess_err",
        )
        swv.plot_cumulative_simplex_1d(
            wald_fits,
            ax=posterior_axs[1],
            color=w_d1_color,
            simplex_key="p_guess_err",
        )
        posterior_axs[0].set_ylabel("density")
        posterior_axs[1].set_ylabel("density")
        posterior_axs[1].set_xlabel("p(guess)")

        self._panel_decoding(
            elmo_fits,
            wald_fits,
            *dec_axs,
            key=key,
            type_str="retro",
            swap_ind=2,
            refit=refit
        )
        dec_axs[1].set_yticks([0, -0.2, -0.4])

    def panel_pro_d1(self, refit=False):
        key = "panel_pro_d1"

        dec_ax, diff_ax = self.gss[key]
        _, e_fits, w_fits = self._get_d1_pro_fits()

        self._panel_decoding(
            e_fits,
            w_fits,
            dec_ax,
            diff_ax,
            key=key,
            type_str="pro",
            swap_ind=2,
            refit=refit,
        )
        diff_ax.set_yticks([0, -0.2, -0.4])

    def panel_pro_d2(self, recompute=False, refit=False):
        key = "panel_pro_d2"
        posterior_axs, dec_axs = self.gss[key]

        e_d1_color = self.params.getcolor("elmo_color")
        w_d1_color = self.params.getcolor("waldorf_color")

        full_dict, elmo_fits, wald_fits = self.get_d2_fits()
        swv.plot_cumulative_simplex_1d(
            elmo_fits,
            ax=posterior_axs[0],
            color=e_d1_color,
            plot_type="pro",
            simplex_key="p_guess_err",
        )
        swv.plot_cumulative_simplex_1d(
            wald_fits,
            ax=posterior_axs[1],
            color=w_d1_color,
            plot_type="pro",
            simplex_key="p_guess_err",
        )
        posterior_axs[0].set_ylabel("density")
        posterior_axs[1].set_ylabel("density")
        posterior_axs[1].set_xlabel("p(guess)")

        self._panel_decoding(
            elmo_fits,
            wald_fits,
            *dec_axs,
            key=key,
            type_str="pro",
            swap_ind=2,
            refit=refit
        )
        dec_axs[1].set_yticks([0, -0.2, -0.4])


class RetroSwapFigure(ModelBasedFigure):
    def __init__(self, fig_key="retro_swap", colors=colors, **kwargs):
        fsize = (8.5, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        lb = 60
        gap = 10
        ppc_wid = 12
        # delay 1
        ppc_grids = pu.make_mxn_gridspec(self.gs, 4, 1, 0, lb, 20, 20 + ppc_wid, 2, 2)
        d1_ppc_axs = self.get_axs(
            ppc_grids,
        )

        post_grids = pu.make_mxn_gridspec(
            self.gs,
            2,
            1,
            5,
            lb - 10,
            0,
            10,
            8,
            4,
        )
        d1_post_axs = self.get_axs(post_grids, squeeze=True, sharey="all")

        avg_grids = pu.make_mxn_gridspec(self.gs, 1, 1, lb, 100, 0, 30, 2, 2)
        sess_bars_grid = self.gs[lb + gap : 100, 0:20]
        sess_diff_grid = self.gs[lb + gap : 100, 28:35]
        sess_ax, sess_diff_ax = self.get_axs(
            (
                sess_bars_grid,
                sess_diff_grid,
            ),
            squeeze=True,
            sharey="all",
        )
        gss["panel_d1"] = (d1_ppc_axs, d1_post_axs, sess_ax)

        # delay 2
        delt = 100 - ppc_wid * 3 - 4
        ppc_grids = pu.make_mxn_gridspec(self.gs, 4, 3, 0, lb, delt, 100, 2, 2)
        d2_ppc_axs = self.get_axs(ppc_grids)

        post_grids = pu.make_mxn_gridspec(
            self.gs,
            2,
            1,
            0,
            lb,
            35,
            50,
            2,
            4,
        )
        d2_post_axs = self.get_axs(post_grids, squeeze=True, sharey="all")

        gss["panel_d2"] = (d2_ppc_axs, d2_post_axs, sess_ax)

        dec_scatter_grid = self.gs[lb + gap : 100, 70:85]
        dec_diff_grid = self.gs[lb + gap + 3 : 98, 95:100]
        gss["panel_decoding"] = self.get_axs(
            (
                dec_scatter_grid,
                dec_diff_grid,
            ),
            squeeze=True,
        )

        corr_gs = self.gs[lb + gap : 100, 43:60]
        gss["panel_corr"] = self.get_axs((corr_gs,))[0, 0]

        gss["panel_rate_differences"] = (sess_ax, sess_diff_ax)

        self.gss = gss

    def get_monkey_d1d2_colors(self):
        e_color = self.monkey_colors["Elmo"]
        w_color = self.monkey_colors["Waldorf"]

        col_diff = self.params.getfloat("d1_d2_color_diff")
        e_d1_color = gpl.add_color_value(e_color, -col_diff)
        e_d2_color = gpl.add_color_value(e_color, col_diff)
        w_d1_color = gpl.add_color_value(w_color, -col_diff)
        w_d2_color = gpl.add_color_value(w_color, col_diff)
        return (e_d1_color, e_d2_color), (w_d1_color, w_d2_color)

    def panel_rate_differences(self):
        key = "panel_rate_differences"
        sess_ax, sess_diff_ax = self.gss[key]

        n_boots = self.params.getint("n_boots")
        e_color = self.params.getcolor("elmo_color")
        w_color = self.params.getcolor("waldorf_color")

        out = self.get_monkey_d1d2_colors()
        (e_d1_color, e_d2_color), (w_d1_color, w_d2_color) = out

        _, e_d1_fits, w_d1_fits = self.get_d1_fits()
        _, e_d2_fits, w_d2_fits = self.get_d2_fits()

        sess_ax.set_xlabel("sessions")
        sess_diff_ax.set_ylabel("average difference")
        sess_ax.set_ylabel("swap rate difference\n(delay 2 - delay 1)")
        sess_diff_ax.set_xticks([0, 1])
        sess_diff_ax.set_xticklabels(["E", "W"])

        # swv.plot_rates(e_d1_fits, w_d1_fits, ax=sess_ax,
        #                colors=(e_d1_color, w_d1_color),
        #                diff=-.1)
        # swv.plot_rates(e_d2_fits, w_d2_fits, ax=sess_ax, ref_ind=2,
        #                colors=(e_d2_color, w_d2_color),
        #                diff=.1)

        self._save_rate_stats(e_d1_fits, "Monkey E", "d1", "retro", t_ind=False)
        self._save_rate_stats(w_d1_fits, "Monkey W", "d1", "retro", t_ind=False)
        swv.plot_rate_differences(e_d1_fits, e_d2_fits, ax=sess_ax, color=e_color)
        swv.plot_rate_differences(w_d1_fits, w_d2_fits, ax=sess_ax, color=w_color)
        gpl.clean_plot(sess_ax, 0)
        gpl.add_hlines(0, sess_ax)
        gpl.clean_plot_bottom(sess_ax, keeplabels=True)
        gpl.clean_plot_bottom(sess_diff_ax, keeplabels=True)

        e_diffs, e_ps = swan.compare_params(e_d1_fits, e_d2_fits)
        w_diffs, w_ps = swan.compare_params(w_d1_fits, w_d2_fits)

        e_diffs_full = np.array(list(np.mean(v) for v in e_diffs.values()))
        e_diffs_boot = u.bootstrap_list(e_diffs_full, np.nanmean, n_boots)
        w_diffs_full = np.array(list(np.mean(v) for v in w_diffs.values()))
        w_diffs_boot = u.bootstrap_list(w_diffs_full, np.nanmean, n_boots)

        self._save_rate_diff_stats(e_diffs_boot, "Monkey E", "retro")
        self._save_rate_diff_stats(w_diffs_boot, "Monkey W", "retro")
        gpl.violinplot(
            [e_diffs_boot],
            [0],
            ax=sess_diff_ax,
            showextrema=False,
            showmedians=True,
            color=(e_color,),
        )
        gpl.violinplot(
            [w_diffs_boot],
            [1],
            ax=sess_diff_ax,
            showextrema=False,
            showmedians=True,
            color=(w_color,),
        )
        gpl.add_hlines(0, sess_diff_ax)

        gpl.clean_plot(sess_diff_ax, 0)

    def panel_corr(self, task_type="retro"):
        key = "panel_corr"
        ax = self.gss[key]

        d1_key = "d1_ppc_pts"
        d2_key = "d2_ppc_pts"

        if self.data.get(d1_key) is None or self.data.get(d2_key) is None:
            raise IOError(
                "the delay1 and delay2 panels must be run before this "
                "panel since they generate important data for this "
                "plot"
            )
        d1_dict = self.data.get(d1_key)
        d2_dict = self.data.get(d2_key)

        p_thr = self.params.getfloat("model_plot_pthr")
        pt_ms = 1

        d1_swap_key = ("misbind", None, p_thr)
        d2_spatial_key = ("spatial", task_type, p_thr)
        d2_cue_key = ("cue", task_type, p_thr)

        colors = (self.monkey_colors["Elmo"], self.monkey_colors["Waldorf"])
        for i, (k, (d1_swap, _)) in enumerate(d1_dict.items()):
            d2_swap, _ = d2_dict[k]
            d1_pts, _, d1_ps = d1_swap[d1_swap_key]["other"]
            d2_spatial_pts, _, d2_spatial_ps = d2_swap[d2_spatial_key]["other"]
            d2_cue_pts, _, d2_cue_ps = d2_swap[d2_spatial_key]["other"]
            d1_inds = np.argsort(d1_ps)
            d2_spatial_inds = np.argsort(d2_spatial_ps)
            d2_cue_inds = np.argsort(d2_cue_ps)
            d1_ps_sort = d1_ps[d1_inds]
            d2_ps_sort = d2_spatial_ps[d2_spatial_inds]
            assert np.all(d1_ps_sort == d2_ps_sort)

            d1_pts = d1_pts[d1_inds]
            d2_spatial_pts = d2_spatial_pts[d2_spatial_inds]
            d2_cue_pts = d2_cue_pts[d2_cue_inds]

            ax.plot(d1_pts, d2_spatial_pts, "o", ms=pt_ms, color=colors[i])
            np.corrcoef(d1_pts, d2_spatial_pts)
        ax.set_aspect("equal")
        gpl.clean_plot(ax, 0)
        ax.set_xlabel("projection in delay 1")
        ax.set_ylabel("projection in delay 2")

    def panel_decoding(self, refit=False):
        key = "panel_decoding"
        dec_ax, diff_ax = self.gss[key]
        _, e_fits, w_fits = self.get_d2_fits()

        self._panel_decoding(e_fits, w_fits, dec_ax, diff_ax, key=key)
        diff_ax.set_xticks([0, -0.2, -0.4])

    def panel_d2(self):
        self._panel_d2()

    def panel_d1(self):
        key = "panel_d1"
        ppc_axs, posterior_axs, sess_ax = self.gss[key]

        # (e_d1_color, _), (w_d1_color, _) = self.get_monkey_d1d2_colors()
        e_d1_color = self.params.getcolor("elmo_color")
        w_d1_color = self.params.getcolor("waldorf_color")

        full_dict, elmo_fits, wald_fits = self.get_d1_fits()
        swv.plot_cumulative_simplex_1d(elmo_fits, ax=posterior_axs[0], color=e_d1_color)
        swv.plot_cumulative_simplex_1d(wald_fits, ax=posterior_axs[1], color=w_d1_color)
        posterior_axs[0].set_ylabel("density")
        posterior_axs[1].set_ylabel("density")
        posterior_axs[1].set_xlabel("p(misbinding)")

        types = (None,)
        mistakes = ("misbind",)
        precomp = self.data.get(key)
        out = self.plot_ppc_groups(
            types, mistakes, ppc_axs, elmo_fits, wald_fits, precomputed_data=precomp
        )
        self.data[key] = out
        self.data["d1_ppc_pts"] = out


class NaiveSwapFigure(SwapErrorFigure):
    def __init__(self, fig_key='naive_swapping', colors=colors, **kwargs):
        fsize = (6, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        
        dist_grids = pu.make_mxn_gridspec(
            self.gs,
            8, 3,
            0, 100,
            20, 80,
            2, 4,
        )
        dist_axs = self.get_axs(dist_grids, squeeze=True, sharex="all")

        gap = 5
        diff_grids = pu.make_mxn_gridspec(
            self.gs,
            4, 1,
            gap, 100 - 2*gap,
            90, 100,
            gap*2, 4,
        )
        diff_axs = self.get_axs(diff_grids, squeeze=True, sharey="all")
        gss['panel_dists'] = (dist_axs, diff_axs)

        self.gss = gss

    def panel_dists(self):
        key = 'panel_dists'
        dist_axs, diff_axs = self.gss[key]

        nc_run = self.params.get('nc_run')
        folder = self.params.get('nc_folder')
        regions = self.params.get('regions')

        null_color = self.params.getcolor('correct_color')
        swap_color = self.params.getcolor('swap_color')

        c_dict = swa.load_naive_results(nc_run, folder)

        dr_axs = np.reshape(dist_axs, (int(dist_axs.shape[0]/2), 2, -1))
        dr_axs = np.swapaxes(dr_axs, 0, 2)

        _, info_groups = swv.plot_all_nc_dict(
            c_dict,
            regions=regions,
            axs=dr_axs,
            plot_inverted=True,
            c_color=null_color,
            s_color=swap_color,
            limit_bins=(-1, 2)
        )

        colors = {
            'elmo_range':self.params.getcolor('elmo_color'),
            'waldorf_range':self.params.getcolor('waldorf_color'),
        }
        swv.plot_nc_diffs(info_groups, diff_axs, colors=colors)
        for ax in diff_axs[:-1]:
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['', '', ''])
        diff_axs[-1].set_xticks([0, 1, 2])
        diff_axs[-1].set_xticklabels(['E', 'W', 'combined'], rotation=45)


class ForgettingFigure(ModelBasedFigure):
    def __init__(self, fig_key="forget", colors=colors, **kwargs):
        fsize = (6, 2)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        lb = 60
        gap = 10

        scatter_wid = 25
        diff_wid = 8
        gap = 10

        # retro
        scatter_grid = self.gs[:, 0:scatter_wid]
        diff_grid = self.gs[:, scatter_wid + gap : scatter_wid + gap + diff_wid]
        dec_axs = self.get_axs(
            (
                scatter_grid,
                diff_grid,
            ),
            squeeze=True,
        )
        gss["panel_retro"] = dec_axs

        # pro
        scatter_grid = self.gs[:, 55 : 55 + scatter_wid]
        diff_grid = self.gs[
            :, 55 + scatter_wid + gap : 55 + scatter_wid + gap + diff_wid
        ]
        dec_axs = self.get_axs(
            (
                scatter_grid,
                diff_grid,
            ),
            squeeze=True,
        )
        gss["panel_pro"] = dec_axs

        self.gss = gss

    def panel_retro(self, refit=False):
        key = "panel_retro"
        axs = self.gss[key]

        corr_thr = self.params.getfloat("corr_thr")
        swap_thr = self.params.getfloat("swap_thr")

        e_color = self.monkey_colors["Elmo"]
        w_color = self.monkey_colors["Waldorf"]

        _, e_fits, w_fits = self.get_d2_fits()
        self._panel_decoding(
            e_fits,
            w_fits,
            *axs,
            key=key,
            refit=refit,
            type_str="retro",
            func=swan.number_decoding_swaps
        )
        axs[1].set_yticks([0, 0.1])

    def panel_pro(self, refit=False):
        key = "panel_pro"
        axs = self.gss[key]

        corr_thr = self.params.getfloat("corr_thr")
        swap_thr = self.params.getfloat("swap_thr")

        e_color = self.monkey_colors["Elmo"]
        w_color = self.monkey_colors["Waldorf"]

        _, e_fits, w_fits = self.get_d2_fits()
        self._panel_decoding(
            e_fits,
            w_fits,
            *axs,
            key=key,
            refit=refit,
            type_str="pro",
            func=swan.number_decoding_swaps
        )
        axs[1].set_yticks([0, 0.1])


class EphysIntroFigure(SwapErrorFigure):
    def __init__(self, fig_key="ephys", colors=colors, **kwargs):
        fsize = (6, 7)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.exp_data = None
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        brain_schem_grid = self.gs[:33, :33]
        gss["panel_brain_schem"] = self.get_axs((brain_schem_grid,))

        su_grids = pu.make_mxn_gridspec(self.gs, 4, 2, 0, 33, 40, 100, 2, 2)
        su_axs = self.get_axs(su_grids, sharex=True, sharey=True)
        gss["panel_single_neuron_examples"] = su_axs

        dec_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 40, 100, 0, 100, 10, 10)
        dec_axs = self.get_axs(dec_grids, sharey=True)
        gss["panel_dec"] = dec_axs

        self.gss = gss

    def _get_experimental_data(self):
        if self.exp_data is None:
            max_files = np.inf
            df = "../data/swap_errors/"
            data = gio.Dataset.from_readfunc(
                swa.load_buschman_data,
                df,
                max_files=max_files,
                seconds=True,
                load_bhv_model="../data/swap_errors/bhv_model.pkl",
                spks_template=swa.busch_spks_templ_mua,
            )
            self.exp_data = data
        return self.exp_data

    def panel_brain_schem(self):
        pass

    def panel_single_neuron_examples(self):
        key = "panel_single_neuron_examples"
        axs = self.gss[key]

        if self.data.get(key) is None:
            exp_data = self._get_experimental_data()

    def _decode_pseudopop(
        self, data_m, field_str, type_str, gen_field_str, mask_func=_upper_color_mask
    ):
        tbeg = self.params.getfloat(field_str + "_beg")
        tend = self.params.getfloat(field_str + "_end")
        twindow = self.params.getfloat("window")
        tstep = self.params.getfloat("step")

        tzf = self.params.get("{}_{}_timekey".format(field_str, type_str))
        gen_tzf = self.params.get("{}_{}_timekey".format(gen_field_str, type_str))

        min_trials = self.params.getint("min_trials")
        pre_pca = self.params.getfloat("pre_pca")
        repl_nan = self.params.getboolean("repl_nan")
        resample_pseudo = self.params.getint("resample_pseudo")
        pseudo = True
        dec_less = False
        n_folds = self.params.getint("n_folds")
        collapse_time = self.params.getboolean("collapse_time")
        dec_beg = self.params.getfloat("dec_beg")
        dec_end = self.params.getfloat("dec_end")

        mask_c1, mask_c2 = mask_func(data_m)
        out = data_m.decode_masks(
            mask_c1,
            mask_c2,
            twindow,
            tbeg,
            tend,
            tstep,
            resample_pseudo=resample_pseudo,
            time_zero_field=tzf,
            n_folds=n_folds,
            pseudo=pseudo,
            repl_nan=repl_nan,
            min_trials_pseudo=min_trials,
            pre_pca=pre_pca,
            dec_less=dec_less,
            collapse_time=collapse_time,
            dec_beg=dec_beg,
            dec_end=dec_end,
            decode_m1=mask_c1,
            decode_m2=mask_c2,
            decode_tzf=gen_tzf,
        )
        return out

    def panel_dec(self):
        key = "panel_dec"
        axs = self.gss[key]

        if self.data.get(key) is None:
            data_use = self._get_experimental_data()

            mask = data_use["StopCondition"] == 1
            mask = mask.rs_and(data_use["is_one_sample_displayed"] == 0)

            retro_mask = mask.rs_and(data_use["Block"] > 1)
            data_retro = data_use.mask(retro_mask)

            pro_mask = mask.rs_and(data_use["Block"] == 1)
            data_pro = data_use.mask(pro_mask)

            mask_funcs = {
                "upper color": _upper_color_mask,
                "lower color": _lower_color_mask,
                "cue": _cue_mask,
                "target color": _target_color_mask,
            }
            retro_d1 = {}
            retro_d2 = {}
            pro_d1 = {}
            pro_d2 = {}
            for label, mf in mask_funcs.items():
                retro_d1[label] = self._decode_pseudopop(
                    data_retro, "delay1", "retro", "delay2", mf
                )
                retro_d2[label] = self._decode_pseudopop(
                    data_retro, "delay2", "retro", "delay1", mf
                )
                pro_d1[label] = self._decode_pseudopop(
                    data_pro, "delay1", "pro", "delay2", mf
                )
                pro_d2[label] = self._decode_pseudopop(
                    data_pro, "delay2", "pro", "delay1", mf
                )
            self.data[key] = ((retro_d1, retro_d2), (pro_d1, pro_d2))
            # self.data[key] = ((retro_d1,),
            #                   ())
        decs = self.data[key]

        titles = (("retrospective", ""), ("prospective", ""))
        y_labels = (("decoding performance", ""), ("decoding performance", ""))
        x_labels = (
            ("time from stimuli", "time from cue"),
            ("time from cue", "time from stimuli"),
        )
        plot_dec = (
            (("upper color", "lower color"), ("target color", "cue")),
            (("cue",), ("target color")),
        )
        plot_gen = (((), ("upper color", "lower color")), ((), ("cue",)))
        for i, j in u.make_array_ind_iterator((2, 2)):
            ax = axs[i, j]
            for key, (dec, xs, gen) in decs[i][j].items():
                if key in plot_dec[i][j]:
                    dec_avg = np.mean(dec, axis=1)
                    gpl.plot_trace_werr(xs, dec_avg, ax=ax, label=key, conf95=True)
                if key in plot_gen[i][j]:
                    gen_avg = np.mean(gen, axis=1)
                    gpl.plot_trace_werr(xs, gen_avg, ax=ax, label=key, conf95=True)

            ax.set_xlabel(x_labels[i][j])
            ax.set_ylabel(y_labels[i][j])
            ax.set_title(titles[i][j], loc="left")
            gpl.clean_plot(ax, j)
            gpl.add_hlines(0.5, ax)
