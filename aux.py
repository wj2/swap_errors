
import numpy as np
import scipy.io as sio
import pandas as pd
import os
import re
from sklearn import svm

import general.utility as u
import general.neural_analysis as na
import general.data_io as gio


busch_bhv_fields = ('StopCondition', 'ReactionTime', 'Block',
                    'is_one_sample_displayed', 'IsUpperSample',
                    'TargetTheta', 'DistTheta', 'ResponseTheta',
                    'CueDelay', 'CueDelay2', 'CueRespDelay', 'FIXATE_ON_diode',
                    'CUE1_ON_diode', 'SAMPLES_ON_diode', 'CUE2_ON_diode',
                    'WHEEL_ON_diode')
def load_bhv_data(fl, flname='bhv.mat', const_fields=('Date', 'Monkey'),
                  extract_fields=busch_bhv_fields):
    bhv = sio.loadmat(os.path.join(fl, flname))['bhv']
    const_dict = {cf:np.squeeze(bhv[cf][0,0]) for cf in const_fields}
    trl_dict = {}
    for tf in extract_fields:
        elements = bhv['Trials'][0,0][tf][0]
        for i, el in enumerate(elements):
            if len(el) == 0:
                elements[i] = np.array([[np.nan]])
        trl_dict[tf] = np.squeeze(np.stack(elements, axis=0))
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
        if np.isnan(et):
            et = bt + extra
        mask = np.logical_and(ts > bt - extra, ts < et + extra)
        chan_m, ids_m, ts_m = chan[mask], ids[mask], ts[mask]
        for j, un in enumerate(unique_neurs):
            mask_n = np.logical_and(chan_m == un[0], ids_m == un[1])
            spks_cont[i, j] = ts_m[mask_n]
    return spks_cont, unique_neurs

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

def load_buschman_data(folder, template='[0-9]{2}[01][0-9][0123][0-9]',
                       bhv_sub='bhv', spikes_sub='spikes', label_sub='labels',
                       spks_template=busch_spks_templ,
                       trl_beg_field='FIXATE_ON_diode',
                       trl_end_field='WHEEL_ON_diode', extra_time=1,
                       max_files=np.inf):
    fls = os.listdir(folder)
    counter = 0
    super_df = pd.DataFrame(columns=('experiment', 'animal', 'date', 'data'))
    dates, expers, monkeys, datas = [], [], [], []
    for fl in fls:
        m = re.match(template, fl)
        if m is not None:
            run_data, trl_data = load_bhv_data(os.path.join(folder, fl,
                                                            bhv_sub))
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
            counter = counter + 1
        if counter >= max_files:
            break
    super_dict = dict(date=dates, experiment=expers, animal=monkeys,
                      data=datas)
    return super_dict

