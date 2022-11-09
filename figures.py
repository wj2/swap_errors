
import numpy as np
import scipy.stats as sts
import functools as ft
import pickle

import general.plotting as gpl
import general.plotting_styles as gps
import general.paper_utilities as pu
import general.utility as u
import general.data_io as gio
import swap_errors.visualization as swv
import swap_errors.auxiliary as swa

config_path = 'swap_errors/figures.conf'

colors = np.array([(127,205,187),
                   (65,182,196),
                   (29,145,192),
                   (34,94,168),
                   (37,52,148),
                   (8,29,88)])/256

def _color_mask(data, thr=np.pi, color='upper_color'):
    cols = gio.ResultSequence(u.normalize_periodic_range(data[color] - thr,
                                                         convert_array=False))
    mask_c1 = cols > 0
    mask_c2 = cols <= 0
    return mask_c1, mask_c2

def _upper_color_mask(data, thr=np.pi):
    return _color_mask(data, thr=thr, color='upper_color')

def _lower_color_mask(data, thr=0):
    return _color_mask(data, thr=thr, color='lower_color')

def _cue_mask(data):
    mask_c1 = data['IsUpperSample'] == 1
    mask_c2 = data['IsUpperSample'] == 0
    return mask_c1, mask_c2

def _target_color_mask(data, thr=np.pi):
    upper_cat, _ = _color_mask(data, thr=thr, color='upper_color')
    lower_cat, _ = _color_mask(data, thr=thr, color='lower_color')
    filt_mask = upper_cat.rs_xor(lower_cat)
    mask_c1, mask_c2 =  _color_mask(data, thr=thr, color='LABthetaTarget')
    mask_c1 = mask_c1.rs_and(filt_mask)
    mask_c2 = mask_c2.rs_and(filt_mask)
    return mask_c1, mask_c2

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
        
class EphysIntroFigure(SwapErrorFigure):

    def __init__(self, fig_key='ephys', colors=colors, **kwargs):
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
        gss['panel_brain_schem'] = self.get_axs((brain_schem_grid,))

        su_grids = pu.make_mxn_gridspec(self.gs, 4, 2,
                                        0, 33, 40, 100,
                                        2, 2)
        su_axs = self.get_axs(su_grids, sharex=True, sharey=True)
        gss['panel_single_neuron_examples'] = su_axs

        
        dec_grids = pu.make_mxn_gridspec(self.gs, 2, 2,
                                         40, 100, 0, 100,
                                         10, 10)
        dec_axs = self.get_axs(dec_grids, sharey=True)
        gss['panel_dec'] = dec_axs
        
        self.gss = gss

    def _get_experimental_data(self):
        if self.exp_data is None:
            max_files = np.inf
            df = '../data/swap_errors/'
            data = gio.Dataset.from_readfunc(
                swa.load_buschman_data,
                df,
                max_files=max_files,
                seconds=True, 
                load_bhv_model='../data/swap_errors/bhv_model.pkl',
                spks_template=swa.busch_spks_templ_mua
            )
            self.exp_data = data
        return self.exp_data
        
    def panel_brain_schem(self):
        pass

    def panel_single_neuron_examples(self):
        key = 'panel_single_neuron_examples'
        axs = self.gss[key]

        if self.data.get(key) is None:
            exp_data = self._get_experimental_data()

    def _decode_pseudopop(self,
                          data_m,
                          field_str,
                          type_str,
                          gen_field_str,
                          mask_func=_upper_color_mask):
        tbeg = self.params.getfloat(field_str + '_beg')
        tend = self.params.getfloat(field_str + '_end')
        twindow = self.params.getfloat('window')
        tstep = self.params.getfloat('step')

        tzf = self.params.get('{}_{}_timekey'.format(field_str, type_str))
        gen_tzf = self.params.get('{}_{}_timekey'.format(gen_field_str,
                                                         type_str))
        
        min_trials = self.params.getint('min_trials')
        pre_pca = self.params.getfloat('pre_pca')
        repl_nan = self.params.getboolean('repl_nan')
        resample_pseudo = self.params.getint('resample_pseudo')
        pseudo = True
        dec_less = False
        n_folds = self.params.getint('n_folds')
        collapse_time = self.params.getboolean('collapse_time')
        dec_beg = self.params.getfloat('dec_beg')
        dec_end = self.params.getfloat('dec_end')

        mask_c1, mask_c2 = mask_func(data_m)
        out = data_m.decode_masks(mask_c1, mask_c2, twindow, tbeg, tend, tstep,
                                  resample_pseudo=resample_pseudo,
                                  time_zero_field=tzf, n_folds=n_folds,
                                  pseudo=pseudo, repl_nan=repl_nan, 
                                  min_trials_pseudo=min_trials,
                                  pre_pca=pre_pca, dec_less=dec_less,
                                  collapse_time=collapse_time,
                                  dec_beg=dec_beg, dec_end=dec_end,
                                  decode_m1=mask_c1, decode_m2=mask_c2,
                                  decode_tzf=gen_tzf)
        return out

    def panel_dec(self):
        key = 'panel_dec'
        axs = self.gss[key]

        if self.data.get(key) is None:
            data_use = self._get_experimental_data()

            mask = data_use['StopCondition'] == 1
            mask = mask.rs_and(data_use['is_one_sample_displayed'] == 0)

            retro_mask = mask.rs_and(data_use['Block'] > 1)
            data_retro = data_use.mask(retro_mask)

            pro_mask = mask.rs_and(data_use['Block'] == 1)
            data_pro = data_use.mask(pro_mask)

            mask_funcs = {
                'upper color':_upper_color_mask,
                'lower color':_lower_color_mask,
                'cue':_cue_mask,
                'target color':_target_color_mask,
            }
            retro_d1 = {}
            retro_d2 = {}
            pro_d1 = {}
            pro_d2 = {}
            for label, mf in mask_funcs.items():
                retro_d1[label] = self._decode_pseudopop(data_retro, 'delay1',
                                                         'retro', 'delay2', mf)
                retro_d2[label] = self._decode_pseudopop(data_retro, 'delay2',
                                                         'retro', 'delay1', mf)
                pro_d1[label] = self._decode_pseudopop(data_pro, 'delay1',
                                                       'pro', 'delay2', mf)
                pro_d2[label] = self._decode_pseudopop(data_pro, 'delay2',
                                                       'pro', 'delay1', mf)
            self.data[key] = ((retro_d1, retro_d2),
                              (pro_d1, pro_d2))
            # self.data[key] = ((retro_d1,),
            #                   ())
        decs = self.data[key]

        titles = (('retrospective', ''),
                  ('prospective', ''))
        y_labels = (('decoding performance', ''),
                    ('decoding performance', ''))
        x_labels = (('time from stimuli', 'time from cue'),
                    ('time from cue', 'time from stimuli'))
        plot_dec = ((('upper color', 'lower color'), ('target color', 'cue')),
                    (('cue',), ('target color')))
        plot_gen = (((), ('upper color', 'lower color')),
                    ((), ('cue',)))
        for (i, j) in u.make_array_ind_iterator((2, 2)):
            ax = axs[i, j]
            for (key, (dec, xs, gen)) in decs[i][j].items():
                if key in plot_dec[i][j]:
                    dec_avg = np.mean(dec, axis=1)
                    gpl.plot_trace_werr(xs, dec_avg, ax=ax, label=key,
                                        conf95=True)
                if key in plot_gen[i][j]:
                    gen_avg = np.mean(gen, axis=1)
                    gpl.plot_trace_werr(xs, gen_avg, ax=ax, label=key,
                                        conf95=True)
                    
                
            ax.set_xlabel(x_labels[i][j])
            ax.set_ylabel(y_labels[i][j])
            ax.set_title(titles[i][j], loc='left')
            gpl.clean_plot(ax, j)
            gpl.add_hlines(.5, ax)

        
