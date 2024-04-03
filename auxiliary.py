
import numpy as np
import scipy.io as sio
import pandas as pd
import os
import re
import pickle
import arviz as az
import collections as c
import itertools as it

import general.utility as u
import general.neural_analysis as na
import general.data_io as gio

guess_model_names = ('arviz_fit_null_guess_model.pkl',
                     'arviz_fit_cue_mistake_model.pkl',
                     'arviz_fit_spatial_errors_model.pkl')
guess_model_keys = ('lin g', 'cue g', 'spatial g')

wh_default_model_names = ('arviz_fit_null_hierarchical_model.pkl',
                          'arviz_fit_spatial_error_hierarchical_model.pkl',
                          'arviz_fit_cue_error_hierarchical_model.pkl',
                          'arviz_fit_super_hybrid_error_hierarchical_model.pkl')
wh_default_model_keys = ('lin', 'spatial', 'cue', 'hybrid')
wh_default_model_names = ('arviz_fit_spatial_error_hierarchical_model.pkl',
                          'arviz_fit_cue_error_hierarchical_model.pkl',
                          'arviz_fit_super_hybrid_error_hierarchical_model.pkl')
wh_default_model_keys = ('spatial', 'cue', 'hybrid')


cue_default_model_names = ('arviz_fit_null_precue_model.pkl',
                           'arviz_fit_spatial_error_precue_model.pkl',
                           'arviz_fit_hybrid_error_precue_model.pkl')
cue_default_model_keys = ('lin', 'spatial', 'hybrid')

model_folder_template = ('swap_errors/neural_model_fits/{num_cols}_colors/'
                         'sess_{session_num}/{time_period}/{time_bin}/'
                         'pca_0.95_before/impute_True/interpolated_knots/')

o_fit_templ = 'fit_spline{n_colors}_sess{sess_ind}_{period}_{run_ind}{ext}'
o_fp = '../results/swap_errors/fits/'
naive_template = '[0-9\\-_:\\.]*\\.pkl'


lm_template = ("fit_(nulls_)?lmtc_(?P<trial_type>pro|retro)_[a-z0-9_]*"
               "(?P<timing>cue|color|wheel|pre-cue|post-cue|pre-color|post-color)"
               "\\-presentation_"
               "(?P<session_ind>[0-9]+)_(?P<jobid>{jobids})\\.pkl")

lm_dist_template = (
    "fit_dist_lmtc_(?P<trial_type>pro|retro)_[a-z0-9_]*"
    "(?P<timing>cue|color|wheel|pre-cue|post-cue|pre-color|post-color)"
    "\\-presentation_"
    "(?P<session_ind>[0-9]+)_(?P<jobid>{jobids})\\.pkl"
)

lm_targ_template = ("fit_nulls_lmtc_{trl_type}_no_{region}"
                    "_{timing}"
                    "\\-presentation_"
                    "(?P<session_ind>[0-9]+)_(?P<jobid>[0-9]+)\\.pkl")

session_to_monkey_dict = {
    k: "Elmo" for k in range(13)
}
session_to_monkey_dict.update({k: "Waldorf" for k in range(13, 23)})


def make_lm_lists(
    *args, **kwargs
):
    regions = ("pfc", "fef", "7ab", "motor", "tpot", "v4pit")
    times = ("pre-cue", "pre-color", "wheel")
    use_tt = {"pre-cue": "retro", "pre-color": "pro", "wheel": "pro"}
    for r in regions:
        for t in times:
            s = list_lm_runinds(
                *args, **kwargs, region=r, timing=t, trl_type=use_tt[t]
            )
            full_s = "{}_null_{} = {}".format(r, t, s)
            print(full_s)

def list_lm_runinds(
    folder, templ=lm_targ_template, region="pfc", trl_type="pro", timing="pre-cue",
):
    fls = os.listdir(folder)
    inds = {}
    use_templ = templ.format(region=region, trl_type=trl_type, timing=timing)
    for fl in fls:
        m = re.match(use_templ, fl)
        if m is not None:
            si = m.group("session_ind")
            l = inds.get(si, [])
            ji = m.group("jobid")
            l.append(ji)
            inds[si] = l

    inds = list(inds.values())[0]
    s = ", ".join(inds)
    return s


def load_motoaki_mat(path, dis_key="disN_block", col_key="disC_block", bhv_key="Error"):
    x = sio.loadmat(path)
    n_sessions = x[dis_key].shape[1]
    n_blocks = x[dis_key][0, -1].shape[1]
    n_cols = x[dis_key][0, -1][0, -1].shape[1]

    out_mat = np.zeros((n_sessions, n_blocks, n_cols))
    out_mat[:] = np.nan
    col_list = x[col_key][0, -1][0, -1][0]
    errors = {}
    for i in range(n_sessions):
        session_ds = x[dis_key][0, i]
        session_cs = x[col_key][0, i]

        n_blocks_i = x[dis_key][0, i].shape[1]
        for j in range(n_blocks_i):
            err_list = errors.get(j, [])
            err_ij = x[bhv_key][0, i][0, j]
            err_list.append(err_ij)
            errors[j] = err_list

            cols_ij = session_cs[0, j][0]
            dist_ij = session_ds[0, j][0]
            if len(cols_ij) < len(col_list):
                mask = np.isin(col_list, cols_ij)
            else:
                mask = np.ones_like(col_list, dtype=bool)
            out_mat[i, j][mask] = dist_ij
    return out_mat, col_list, errors


def load_lm_dist_results(runind, folder="swap_errors/lm_fits/", templ=lm_dist_template):
    use_template = templ.format(jobids=runind)
    loaded_runs = {}
    
    for fl, fl_info, data_fl in u.load_folder_regex_generator(folder, use_template):
        monkey = session_to_monkey_dict[int(fl_info["session_ind"])]
        k = (monkey, fl_info["trial_type"], fl_info["timing"])
        mat_list, xs = loaded_runs.get(k, ([], None))
        mat_list.append(data_fl["dist_mat"])
        xs = data_fl["xs"]
        loaded_runs[k] = (mat_list, xs)
    out = {k: (np.stack(m, axis=0), xs) for k, (m, xs) in loaded_runs.items()}
    return out


def load_lm_results(runinds, folder='swap_errors/lms/',
                    templ=lm_template, swap_mean=True):
    jobids = '|'.join(runinds)
    templ = templ.format(jobids=jobids)
    fls = os.listdir(folder)
    full_dict = {}
    for fl in fls:
        m = re.match(templ, fl)
        if m is not None:
            timing = m.group("timing")
            trial_type = m.group("trial_type")
            session_ind = int(m.group("session_ind"))
            out = pd.read_pickle(open(os.path.join(folder, fl), 'rb'))
            null_col_fl = out["null_color"]
            null_cue_fl = out["null_cue"]
            if swap_mean:
                swap_col_fl = np.mean(out["swap_color"], axis=0)
                swap_cue_fl = np.mean(out["swap_cue"], axis=0)
            else:
                swap_col_fl = out["swap_color"]
                swap_cue_fl = out["swap_cue"]

            xs = out["xs"]

            if session_ind in range(13):
                monkey = "Elmo"
            else:
                monkey = "Wald"

            tt_dict = full_dict.get(trial_type, {})
            tt_timing_dict = tt_dict.get(timing, {})

            (null_col, swap_col), (null_cue, swap_cue), _ = tt_timing_dict.get(
                monkey, (([], []), ([], []), None)
            )
            null_col.append(null_col_fl)
            swap_col.append(swap_col_fl)
            null_cue.append(np.squeeze(null_cue_fl))
            swap_cue.append(swap_cue_fl)

            tt_timing_dict[monkey] = (
                (null_col, swap_col),
                (null_cue, swap_cue),
                xs
            )
            tt_dict[timing] = tt_timing_dict
            full_dict[trial_type] = tt_dict

    # print(out.keys())
    # print(out["kwargs"])
    cue_dict = {}
    color_dict = {}
    for trial_type, tt_dict in full_dict.items():
        for timing, monkey_dict in tt_dict.items():
            for monkey, (colors, cues, xs) in monkey_dict.items():
                nulls, swaps = colors
                nulls_comb = np.concatenate(nulls, axis=0)
                if swap_mean:
                    nulls_comb = np.squeeze(np.swapaxes(nulls_comb, 0, 3))
                    swaps_comb = np.concatenate(swaps, axis=2)
                else:
                    swaps_comb = np.concatenate(swaps, axis=0)
                colors = (nulls_comb, swaps_comb), (nulls, swaps), xs
                cd = color_dict.get(monkey, {})
                cd_tt = cd.get(trial_type, {})
                cd_tt[timing] = colors
                cd[trial_type] = cd_tt
                color_dict[monkey] = cd

                nulls, swaps = cues
                swaps_comb = np.concatenate(swaps, axis=0)
                nulls_comb = np.concatenate(nulls, axis=0)
                nulls_comb = np.squeeze(nulls_comb)
                cues = (nulls_comb, swaps_comb), (nulls, swaps), xs
                cd = cue_dict.get(monkey, {})
                cd_tt = cd.get(trial_type, {})
                cd_tt[timing] = cues
                cd[trial_type] = cd_tt
                cue_dict[monkey] = cd

                monkey_dict[monkey] = (colors, cues)
            tt_dict[timing] = monkey_dict
        full_dict[trial_type] = tt_dict
    return full_dict, color_dict, cue_dict


def load_naive_results(nc_run, folder='swap_errors/naive_centroids/',
                       templ=naive_template):
    path = os.path.join(folder, nc_run)
    if not os.path.isfile(path):
        template = nc_run + naive_template
        fls = os.listdir(folder)
        matches = list(fl for fl in fls if re.match(template, fl) is not None)
        if len(matches) > 0:
            path = os.path.join(folder, matches[0])
        else:
            raise IOError('no matching file found')
    c_dict = pd.read_pickle(open(path, 'rb'))
    return c_dict


def get_type_ind(type_, data, use_default=None, return_type=False):
    if use_default is None:
        use_default = {'retro':1, 'pro':2}
    use_dict = data.get('type_conv', use_default)
    type_int = use_dict[type_]
    out = type_int - 1
    if return_type:
        out = (out, type_int)
    return out


def print_error_details(fit, Rhat=True, eps=.05, divergence=True, **kwargs):
    if not Rhat:
        rh = az.rhat(fit)
        eps = .05
        for k in rh.keys():
            v = rh[k].to_numpy()
            m1 = v > 1 + eps
            m2 = v < 1 - eps
            mask = np.logical_or(m1, m2)

            if np.any(mask):
                vals = v[mask]
                print('rhat in {}\nvals are {}'.format(k, vals))
    if not divergence:
        mu_div = np.mean(fit.sample_stats['diverging'].to_numpy())
        if mu_div > 0:
            print('div\npercent is {}'.format(mu_div))

def load_o_fits(run_ind, n_colors=5, sess_inds=range(23),
                period='WHEEL_ON', load_data=False, fit_templ=o_fit_templ,
                folder=o_fp):
    out_dict = {}
    for ind in sess_inds:
        az_ind = fit_templ.format(n_colors=n_colors, sess_ind=ind,
                                  period=period,
                                  ext='_az.nc',
                                  run_ind=run_ind)
        az_fp = os.path.join(folder, az_ind)
        if os.path.isfile(az_fp):
            fit = az.from_netcdf(az_fp)
            out_dict[ind] = ({'other':fit},)
            if load_data:
                d_ind = fit_templ.format(n_colors=n_colors, sess_ind=ind,
                                         period=period,
                                         ext='.pkl',
                                         run_ind=run_ind)
                d_fp = os.path.join(folder, d_ind)
                data = pd.read_pickle(open(d_fp, 'rb'))
            for k, v in data['diags'].items():
                if not v:
                    print('session {ind} has {k} warning'.format(ind=ind,
                                                                 k=k))
            print_error_details(fit, **data['diags'])
            fit_data = data.pop('data')
            data.update(fit_data)
            data['type_conv'] = dict(zip(data['type_str'], data['type']))
            out_dict[ind] = out_dict[ind] + (data,)
        else:
            print('no file found for session {}'.format(ind))
    return out_dict
    

def load_x_sweep(folder, run_ind, template, guess=False):
    fls = os.listdir(folder)
    if guess:
        g_str = 'guess_'
    else:
        g_str = ''
    template = template.format(run_ind=run_ind, guess=g_str)
        
    x_list = []
    for fl in fls:
        m = re.match(template, fl)
        if m is not None:
            decider = m.group('decider')
            out_fl = pd.read_pickle(open(os.path.join(folder, fl), 'rb'))
            args = out_fl.pop('args')
            out_fl.update(vars(args))
            x_list.append(out_fl)
    return pd.DataFrame(x_list)

nc_sweep_pattern = 'nc_{guess}(?P<decider>[a-zA-Z]+)_[0-9]+_{run_ind}_[0-9_\\-:.]+\\.pkl'
def load_nc_sweep(folder, run_ind,
                  template=nc_sweep_pattern,
                  **kwargs):
    return load_x_sweep(folder, run_ind, template, **kwargs)
    
fs_sweep_pattern = ('[fst]+_(?P<decider>[a-zA-Z]+)_[0-9]+_{run_ind}_[0-9_\\-:.]+'
                    '\\.pkl')
def load_fs_sweep(folder, run_ind,
                 template=fs_sweep_pattern):
    return load_x_sweep(folder, run_ind, template)

circus_sweep_pattern = 'r_[a-zA-Z_0-9]*[0-9]+_[0-9-._:]+\\.pkl'
def load_circus_sweep(folder, swept_keys, store_keys=('cue1', 'cue2'),
                      template=circus_sweep_pattern):
    fls = os.listdir(folder)
    keep_keys = swept_keys + store_keys
    out = {key:[] for key in keep_keys}
    out['conj'] = []
    for fl in fls:        
        m = re.match(template, fl)
        if m is not None:
            data = pd.read_pickle(open(os.path.join(folder, fl), 'rb'))
            for kk in keep_keys:
                out[kk].append(data[kk])
            out['conj'] = {sk:out[sk] for sk in swept_keys}
    return out

cluster_naive_d1_path_templ = (
    '/burg/theory/users/ma3811/assignment_errors/5_colors/'
    'sess_{}/CUE2_ON_diode/-0.5-0.0-0.5_0.5/pca_0.95_before/'
    'impute_True/spline1_knots/{}/{}/stan_data.pkl')
cluster_naive_d1_manual_path_templ = (
    '/burg/theory/users/ma3811/assignment_errors/manual/5_colors/'
    'sess_{}/CUE2_ON_diode/-0.5-0.0-0.5_0.5/pca_0.95_before/'
    'impute_True/spline1_knots/{}/{}/stan_data.pkl')
cluster_naive_d1_pro_path_templ = (
    '/burg/theory/users/ma3811/assignment_errors/{manual}{n_colors}_colors/'
    'sess_{sess_num}/SAMPLES_ON_diode/{start}-{end}-{diff}_{diff}/pca_0.95_before/'
    'impute_{impute}/spline{spline_order}_knots/{region}/{trl_type}/stan_data.pkl')
cluster_naive_d1_format_options = {
    'sessions':range(0, 23),
    'region':('all', 'frontal', 'posterior'),
    'trl_type':('retro',)}
cluster_naive_d1_format_options_noregions = {
    'sessions':range(0, 23),
    'region':('all',),
    'trl_type':('retro',)}


cluster_naive_d2_path_templ = (
    '/burg/theory/users/ma3811/assignment_errors/5_colors/'
    'sess_{}/WHEEL_ON_diode/-0.5-0.0-0.5_0.5/pca_0.95_before/'
    'impute_True/spline1_knots/{}/{}/stan_data.pkl')
cluster_naive_d2_manual_path_templ = (
    '/burg/theory/users/ma3811/assignment_errors/manual/5_colors/'
    'sess_{}/WHEEL_ON_diode/-0.5-0.0-0.5_0.5/pca_0.95_before/'
    'impute_True/spline1_knots/{}/{}/stan_data.pkl')
cluster_naive_d2_format_options = {
    'sessions':range(0, 23),
    'region':('all', 'frontal', 'posterior'),
    'trl_type':('retro', 'pro')}
cluster_naive_d2_format_options_noregions = {
    'sessions':range(0, 23),
    'region':('all',),
    'trl_type':('retro', 'pro')}

def load_pro_d1_stan_data(n_colors=5, spline_order=1, region='all',
                          trl_type='pro', session_range=None,
                          templ=cluster_naive_d1_pro_path_templ,
                          use_manual=True,
                          impute=True,
                          start=-0.2, end=.3,
                          **kwargs):
    if use_manual:
        manual = 'manual/'
    else:
        manual = ''
    if session_range is None:
        session_range = range(0, 23)
    diff = end - start
    out_dict = {}
    for sr in session_range:
        path = templ.format(n_colors=n_colors, spline_order=spline_order,
                            region=region, trl_type=trl_type,
                            sess_num=sr, start=start, end=end,
                            diff=diff, manual=manual,
                            impute=impute,
                            **kwargs)
        sd = pd.read_pickle(open(path, 'rb'))
        out_dict[sr] = (None, sd)
    return out_dict

def load_files_ma_folders(file_template, **format_options):
    format_options = c.OrderedDict(format_options)
    all_read = {}
    for prod in it.product(*format_options.values()):
        fp = file_template.format(*prod)
        m = pd.read_pickle(open(fp, 'rb'))
        all_read[prod] = m
    return all_read

def session_df(file_template, keys, **format_options):
    all_keys = tuple(format_options.keys()) + tuple(keys)
    m_dict = {k:[] for k in all_keys}
    m_dict['dims'] = []
    format_options = c.OrderedDict(format_options)
    for prod in it.product(*format_options.values()):
        fp = file_template.format(*prod)
        m = pd.read_pickle(open(fp, 'rb'))
        all_vals = prod + tuple(m[k] for k in keys)

        head, _ = os.path.split(fp)
        dp = os.path.join(head, 'stan_data.pkl')
        data = pd.read_pickle(open(dp, 'rb'))
        m_dict['dims'].append(data['y'].shape[1])
    
        for i, k in enumerate(all_keys):
            m_dict[k].append(all_vals[i])
    return pd.DataFrame(m_dict)        

def load_many_sessions(session_nums, num_cols, time_period, time_bin,
                       **kwargs):
    if time_period == 'CUE2_ON_diode':
        k = dict(no_guess_model_names=cue_default_model_names,
                 no_guess_model_keys=cue_default_model_keys)
        k.update(kwargs)
    elif time_period == 'WHEEL_ON_diode':
        k = dict(no_guess_model_names=wh_default_model_names,
                 no_guess_model_keys=wh_default_model_keys)
        k.update(kwargs)        
    else:
        k = kwargs
    session_dict = {}
    for sn in session_nums:
        session_dict[sn] = load_model_fits_templ(num_cols, sn, time_period,
                                                 time_bin, **k)
    return session_dict

def load_model_fits_templ(num_cols, session_num, time_period, time_bin,
                          template=model_folder_template, **kwargs):
    folder = template.format(num_cols=num_cols, session_num=session_num,
                             time_period=time_period, time_bin=time_bin)
    return load_model_fits(folder, **kwargs)

def load_model_fits(folder, guess_model_names=cue_default_model_names,
                    no_guess_model_names=wh_default_model_names,
                    guess_model_keys=cue_default_model_keys,
                    no_guess_model_keys=wh_default_model_keys,
                    data_name='stan_data.pkl', load_guess=False,
                    load_no_guess=True):
    load_names = ()
    load_keys = ()
    if load_guess:
        load_names = load_names + guess_model_names
        load_keys = load_keys + guess_model_keys
    if load_no_guess:
        load_names = load_names + no_guess_model_names
        load_keys = load_keys + no_guess_model_keys
    model_dict = {}
    for i, ln in enumerate(load_names):
        path = os.path.join(folder, ln)
        m = pd.read_pickle(open(path, 'rb'))
        model_dict[load_keys[i]] = m
    data_path = os.path.join(folder, data_name)
    data = pd.read_pickle(open(data_path, 'rb'))
    return model_dict, data        

busch_bhv_fields = ('StopCondition', 'ReactionTime', 'Block',
                    'is_one_sample_displayed', 'IsUpperSample',
                    'TargetTheta', 'DistTheta', 'ResponseTheta',
                    'LABthetaTarget', 'LABthetaDist', 'LABthetaResp',
                    'CueDelay', 'CueDelay2', 'CueRespDelay', 'FIXATE_ON_diode',
                    'CUE1_ON_diode', 'SAMPLES_ON_diode', 'CUE2_ON_diode',
                    'WHEEL_ON_diode')
def load_bhv_data(fl, flname='bhv.mat', const_fields=('Date', 'Monkey'),
                  extract_fields=busch_bhv_fields, add_color=True,
                  add_err=True):
    bhv = sio.loadmat(os.path.join(fl, flname))['bhv']
    const_dict = {cf:np.squeeze(bhv[cf][0,0]) for cf in const_fields}
    trl_dict = {}
    for tf in extract_fields:
        elements = bhv['Trials'][0,0][tf][0]
        for i, el in enumerate(elements):
            if len(el) == 0:
                elements[i] = np.array([[np.nan]])
        trl_dict[tf] = np.squeeze(np.stack(elements, axis=0))
    if add_color:
        targ_color = trl_dict['LABthetaTarget']
        dist_color = trl_dict['LABthetaDist']
        upper_col = np.zeros(len(targ_color))
        lower_col = np.zeros_like(upper_col)
        upper_mask = trl_dict['IsUpperSample'] == 1
        n_upper_mask = np.logical_not(upper_mask)
        upper_col[upper_mask] = targ_color[upper_mask]
        upper_col[n_upper_mask] = dist_color[n_upper_mask]
        lower_col[upper_mask] = dist_color[upper_mask]
        lower_col[n_upper_mask] = targ_color[n_upper_mask]
        trl_dict['upper_color'] = upper_col
        trl_dict['lower_color'] = lower_col
    if add_err:
        err = u.normalize_periodic_range(trl_dict['LABthetaTarget']
                                         - trl_dict['LABthetaResp'])
        trl_dict['err'] = err
    return const_dict, trl_dict

busch_spks_templ_unsrt = 'selWM_001_chan([0-9]+)_4sd\\.mat'
busch_spks_templ_mua = 'selWM_001_chan([0-9]+)_4sd-srt-mua\\.mat'
busch_spks_templ = 'selWM_001_chan([0-9]+)_4sd-srt\\.mat'
def load_spikes_data(folder, templ=busch_spks_templ):
    fls = os.listdir(folder)
    chan_all, ids_all, ts_all = [], [], []
    for fl in fls:
        m = re.match(templ, fl)
        if m is not None:
            spks = sio.loadmat(os.path.join(folder, fl))
            ts = np.squeeze(spks['ts'])
            ids = np.squeeze(spks['id'])
            chan = np.ones(len(ids))*int(m.group(1))
            chan_all = np.concatenate((chan_all, chan))
            ids_all = np.concatenate((ids_all, ids))
            ts_all = np.concatenate((ts_all, ts))
    return chan_all, ids_all, ts_all

def split_spks_bhv(chan, ids, ts, beg_ts, end_ts, extra):
    unique_neurs = np.unique(list(zip(chan, ids)), axis=0)
    spks_cont = np.zeros((len(beg_ts), len(unique_neurs)), dtype=object)
    for i, bt in enumerate(beg_ts):
        bt = np.squeeze(bt)
        et = np.squeeze(end_ts[i])
        mask = np.logical_and(ts > bt - extra, ts < et + extra)
        chan_m, ids_m, ts_m = chan[mask], ids[mask], ts[mask]
        for j, un in enumerate(unique_neurs):
            mask_n = np.logical_and(chan_m == un[0], ids_m == un[1])
            spks_cont[i, j] = ts_m[mask_n]
    return spks_cont, unique_neurs

def merge_dicts(sd_primary, **sess_dicts):
    new_dict = {}
    for k, (model_pr, data_pr) in sd_primary.items():
        model_dict_k = {}
        model_dict_k.update(model_pr)
        for add_k, sess_dict in sess_dicts.items():
            new_model_dict = sess_dict[k][0]
            for mk, model in new_model_dict.items():
                model_dict_k[mk + ' ' + add_k] = model
        new_dict[k] = (model_dict_k, data_pr)
    return new_dict

def load_label_data(labelpath, unique_neurs, templ='([0-9a-zA-Z]+)\\.txt'):
    fls = os.listdir(labelpath)
    region_labels = np.zeros(len(unique_neurs), dtype=object)
    for fl in fls:
        m = re.match(templ, fl)
        if m is not None and (m.group(1) != 'frontal'
                              and m.group(1) != 'posterior'):
            with open(os.path.join(labelpath, fl), 'rb') as chans:
                c = chans.read().decode().split('\n')
                c_filt = list(float(ci) for ci in c if len(ci) > 0)
                mask = np.isin(unique_neurs[:, 0], c_filt)
                region_labels[mask] = m.group(1)
    return region_labels

def transform_bhv_model(fit, mapping_dict, transform_prob=True, take_mean=True):
    probs = fit['outcome_lps']
    if transform_prob:
        tot = np.sum(np.exp(probs), axis=2, keepdims=True)
        probs = np.exp(probs)/tot
    session_dict = {}
    for i in range(probs.shape[1]):
        pi = probs[:, i]
        if take_mean:
            pi = np.mean(pi, axis=0)
        key = mapping_dict[i]
        ind = key[0]
        session_ident = key[1:]
        l = session_dict.get(session_ident, [])
        l.append((ind, pi))
        session_dict[session_ident] = l
    return session_dict

def load_buschman_data(folder, template='[0-9]{2}[01][0-9][0123][0-9]',
                       bhv_sub='bhv', spikes_sub='spikes', label_sub='labels',
                       spks_template=busch_spks_templ,
                       trl_beg_field='FIXATE_ON_diode',
                       trl_end_field='WHEEL_ON_diode', extra_time=1,
                       max_files=np.inf, load_bhv_model=None):
    fls = os.listdir(folder)
    counter = 0
    dates, expers, monkeys, datas = [], [], [], []
    n_neurs = []
    if load_bhv_model is not None:
        bhv_model = pd.read_pickle(open(load_bhv_model, 'rb'))
    for fl in fls:
        m = re.match(template, fl)
        if m is not None:
            run_data, trl_data = load_bhv_data(os.path.join(folder, fl,
                                                            bhv_sub))
            if load_bhv_model is not None:
                key = (str(run_data['Monkey']), str(run_data['Date']))
                dat = bhv_model[key]
                n_trls = len(trl_data['LABthetaTarget'])
                store = np.zeros((n_trls, dat[0][1].shape[-1]))
                store[...] = np.nan
                for ind, data in dat:
                    store[ind] = data
                trl_data['guess_prob'] = store[:, 0]
                trl_data['swap_prob'] = store[:, 1]
                trl_data['corr_prob'] = store[:, 2]
            
            spks_path = os.path.join(folder, fl, spikes_sub)
            chans, ids, ts = load_spikes_data(spks_path, templ=spks_template)
            spk_split, neurs = split_spks_bhv(chans, ids, ts,
                                              trl_data[trl_beg_field],
                                              trl_data[trl_end_field],
                                              extra_time)
            data_lab = load_label_data(os.path.join(folder, fl, label_sub),
                                       neurs)
            n_trls = spk_split.shape[0]
            d_dict = {}
            d_dict.update(trl_data)
            d_dict.update((('spikeTimes', list(n for n in spk_split)),
                           ('neur_channels', (neurs[:, 0],)*n_trls),
                           ('neur_ids', (neurs[:, 1],)*n_trls),
                           ('neur_regions', (data_lab,)*n_trls)))
            df = pd.DataFrame(data=d_dict)
            datas.append(df)
            dates.append(run_data['Date'])
            monkeys.append(run_data['Monkey'])
            expers.append('WMrecall')
            n_neurs.append(len(d_dict['neur_ids'][0]))
            counter = counter + 1
        if counter >= max_files:
            break
    super_dict = dict(date=dates, experiment=expers, animal=monkeys,
                      data=datas, n_neurs=n_neurs)
    return super_dict
