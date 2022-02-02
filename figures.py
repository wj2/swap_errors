
import numpy as np
import scipy.stats as sts
import functools as ft
import pickle

import general.plotting as gpl
import general.plotting_styles as gps
import general.paper_utilities as pu
import general.utility as u
import swap_errors.visualization as swv

config_path = 'swap_errors/figures.conf'

colors = np.array([(127,205,187),
                   (65,182,196),
                   (29,145,192),
                   (34,94,168),
                   (37,52,148),
                   (8,29,88)])/256

class SwapErrorFigure(pu.Figure):

    def _make_color_dict(self, ks):
        return self._make_param_dict(ks)

    def _make_param_dict(self, ks, add='_color', func=None):
        if func is None:
            func = self.params.getcolor
        color_dict = {}
        for k in ks:
            color_dict[k] = func(k + add)
        return color_dict

    @property
    def monkeys(self):
        return (self.params.get('monkey1'),
                self.params.get('monkey2'))

    @property
    def monkey_names(self):
        return self._make_param_dict(self.monkeys, add='_name',
                                     func=self.params.get)

    @property
    def monkey_colors(self):
        return self._make_color_dict(self.monkeys)

    @property
    def bhv_outcomes(self):
        return (self.params.get('bhv_outcome1'),
                self.params.get('bhv_outcome2'),
                self.params.get('bhv_outcome3'))

    @property
    def bhv_colors(self):
        return self._make_color_dict(self.bhv_outcomes)
    
class BehaviorFigure(SwapErrorFigure):

    def __init__(self, fig_key='bhv', colors=colors, **kwargs):
        fsize = (8, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        task_schem_grid = self.gs[:, :25]
        gss['panel_task_schematic'] = self.get_axs((task_schem_grid,))
        
        err_grid = pu.make_mxn_gridspec(self.gs, 2, 2,
                                        0, 100, 35, 50,
                                        10, 2)
        gss['panel_err_distribution'] = self.get_axs(err_grid, sharex=True,
                                               sharey=True)

        model_schem_grid = self.gs[:, 55:75]
        gss['panel_model_schematic'] = self.get_axs((model_schem_grid,))

        r_simp_gs = pu.make_mxn_gridspec(self.gs, 2, 1,
                                         0, 100, 80, 100,
                                         10, 0)
        r_simp_3d = np.zeros_like(r_simp_gs, dtype=bool)
        r_simp_3d[1] = False
        err_rate_ax, simp_ax = self.get_axs(r_simp_gs, plot_3ds=r_simp_3d)

        gss['panel_err_rates'] = err_rate_ax[0]
        gss['panel_trial_simplex'] = simp_ax[0]
        
        self.gss = gss

    def _get_bhv_model(self):
        bhv_model_out = self.data.get('bhv_model_out')
        if bhv_model_out is None:
            bhv_model_out = pickle.load(open(self.params.get('bhv_model_path'),
                                             'rb'))
            self.data['bhv_model_out'] = bhv_model_out
        return bhv_model_out        
        
    def panel_err_distribution(self):
        key = 'panel_err_distribution'
        axs = self.gss[key]
        
        m_dict = self._get_bhv_model()
        monkeys = self.monkeys
        m_colors = self.monkey_colors
        for i, mi in enumerate(monkeys):
            model_err = m_dict[mi][0].posterior_predictive['err_hat'].to_numpy()
            err = m_dict[mi][1]['err']
            dist_err = m_dict[mi][1]['dist_err']
            swv.plot_error_swap_distribs_err(err, dist_err, axs=axs[i],
                                             model_data=model_err,
                                             model_derr=dist_err,
                                             label=self.monkey_names[mi],
                                             color=m_colors[mi])
            for ax_ij in axs[i]:
                ax_ij.set_xticks([-np.pi, 0, np.pi])
                ax_ij.set_xlabel('')
                ax_ij.set_xticklabels([])
        axs[i, 0].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        axs[i, 1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        axs[i, 0].set_xlabel('error')
        axs[i, 1].set_xlabel('distractor\ndistance')

    def panel_err_rates(self):
        key = 'panel_err_rates'
        ax = self.gss[key]

        m_dict = self._get_bhv_model()
        models = []
        cols = []
        names = []
        monkey_colors = {}
        for m in self.monkeys:
            models.append(m_dict[m][0].posterior)
            names.append(self.monkey_names[m])
            monkey_colors[self.monkey_names[m]] = self.monkey_colors[m]
        swv.plot_model_probs(*models, colors=cols, ax=ax,
                             arg_names=names,
                             monkey_colors=monkey_colors)

    def panel_trial_simplex(self):
        key = 'panel_trial_simplex'
        ax = self.gss[key]

        m = self.params.get('simplex_monkey')
        session_date = self.params.get('simplex_session')
        if self.data.get(key) is None:
            simpl = pickle.load(open(self.params.get('bhv_simplex_path'), 'rb'))
            self.data[key] = simpl
        simplex = self.data[key]
        simplex_labels = self.bhv_outcomes
        simplex_colors = tuple(self.bhv_colors[k] for k in simplex_labels)
        sd = simplex[(m, session_date)]
        sd_arr = np.stack(list(v[1] for v in sd), axis=0)
        swv.visualize_simplex_2d(sd_arr, ax=ax, colors=simplex_colors,
                                 ax_labels=simplex_labels)
        
