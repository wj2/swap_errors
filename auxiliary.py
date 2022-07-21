
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

sweep_pattern = 'r_[a-zA-Z_0-9]*[0-9]+_[0-9-._:]+\.pkl'
def load_circus_sweep(folder, swept_keys, store_keys=('cue1', 'cue2'),
                      template=sweep_pattern):
    fls = os.listdir(folder)
    keep_keys = swept_keys + store_keys
    out = {key:[] for key in keep_keys}
    out['conj'] = []
    for fl in fls:        
        m = re.match(template, fl)
        if m is not None:
            data = pickle.load(open(os.path.join(folder, fl), 'rb'))
            for kk in keep_keys:
                out[kk].append(data[kk])
            out['conj'] = {sk:out[sk] for sk in swept_keys}
    return out

cluster_naive_d1_path_templ = (
    '/burg/theory/users/ma3811/assignment_errors/5_colors/'
    'sess_{}/CUE2_ON_diode/-0.5-0.0-0.5_0.5/pca_0.95_before/'
    'impute_True/spline1_knots/all/{}/stan_data.pkl')
cluster_naive_d1_format_options = {
    'sessions':range(0, 23),
    'trl_type':('retro',)}

cluster_naive_d2_path_templ = (
    '/burg/theory/users/ma3811/assignment_errors/5_colors/'
    'sess_{}/WHEEL_ON_diode/-0.5-0.0-0.5_0.5/pca_0.95_before/'
    'impute_True/spline1_knots/all/{}/stan_data.pkl')
cluster_naive_d2_format_options = {
    'sessions':range(0, 23),
    'trl_type':('retro', 'pro')}

def load_files_ma_folders(file_template, **format_options):
    format_options = c.OrderedDict(format_options)
    all_read = {}
    for prod in it.product(*format_options.values()):
        fp = file_template.format(*prod)
        m = pickle.load(open(fp, 'rb'))
        all_read[prod] = m
    return all_read

def session_df(file_template, keys, **format_options):
    all_keys = tuple(format_options.keys()) + tuple(keys)
    m_dict = {k:[] for k in all_keys}
    m_dict['dims'] = []
    format_options = c.OrderedDict(format_options)
    for prod in it.product(*format_options.values()):
        fp = file_template.format(*prod)
        m = pickle.load(open(fp, 'rb'))
        all_vals = prod + tuple(m[k] for k in keys)

        head, _ = os.path.split(fp)
        dp = os.path.join(head, 'stan_data.pkl')
        data = pickle.load(open(dp, 'rb'))
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
        m = pickle.load(open(path, 'rb'))
        model_dict[load_keys[i]] = m
    data_path = os.path.join(folder, data_name)
    data = pickle.load(open(data_path, 'rb'))
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

busch_spks_templ_unsrt = 'selWM_001_chan([0-9]+)_4sd\.mat'
busch_spks_templ_mua = 'selWM_001_chan([0-9]+)_4sd-srt-mua\.mat'
busch_spks_templ = 'selWM_001_chan([0-9]+)_4sd-srt\.mat'
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

def load_label_data(labelpath, unique_neurs, templ='([0-9a-zA-Z]+)\.txt'):
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
        bhv_model = pickle.load(open(load_bhv_model, 'rb'))
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

