
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle

JEFF_CODE_DIR = '/Users/wjj/Dropbox/research/analysis/repler/src/'
MATTEO_CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
if os.path.isdir(JEFF_CODE_DIR):
    CODE_DIR = JEFF_CODE_DIR
elif os.path.isdir(MATTEO_CODE_DIR):
    CODE_DIR = MATTEO_CODE_DIR
else:
    CODE_DIR = ''
    
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from matplotlib import colors as mpc
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import scipy.stats as sts
import scipy.linalg as la
import scipy.optimize as sopt

# import umap
from cycler import cycler

# my code
import students
import assistants
import experiments as exp
import util
import tasks
import plotting as tplt
import anime as ani

# jeff code
import swap_errors.analysis as swan
import general.plotting as gpl

class Task(object):
    
    def __init__(self, num_cols, T_inp1, T_inp2, T_resp, T_tot, go_cue=False):
        
        self.num_col = num_cols
        
        self.T_inp1 = T_inp1
        self.T_inp2 = T_inp2
        self.T_resp = T_resp
        self.T_tot = T_tot
        
        self.go_cue = go_cue
        
    def generate_data(self, n_seq, *seq_args, **seq_kwargs):
        
        upcol, downcol, cue = self.generate_colors(n_seq)
        
        inps, outs, train_mask = self.generate_sequences(upcol, downcol, cue,
                                                         *seq_args, **seq_kwargs)
        
        return inps, outs, upcol, downcol, cue, train_mask
        
    
    def generate_colors(self, n_seq):
        
        upcol = np.random.choice(np.linspace(0,2*np.pi, self.num_col), n_seq)
        downcol = np.random.choice(np.linspace(0,2*np.pi, self.num_col), n_seq)
            
        cue = np.random.choice([-1.,1.], n_seq) 
        
        return upcol, downcol, cue
        
    def generate_sequences(self, upcol, downcol, cue, jitter=3, inp_noise=0.0,
                           dyn_noise=0.0, new_T=None, retro_only=False,
                           pro_only=False, net_size=None, present_len=1,
                           report_cue=True, report_uncue_color=True):
        
        T_inp1 = self.T_inp1
        T_inp2 = self.T_inp2
        T_resp = self.T_resp
        if new_T is None:
            T = self.T_tot
        else:
            T = new_T
        n_seq = len(upcol)
        ndat = n_seq

        if net_size is None and dyn_noise > 0:
            raise IOError('cannot do dynamics noise without providing the net '
                          'size')
        elif dyn_noise > 0:
            net_inp = net_size
        else:
            net_inp = 0
            
        
        col_inp = np.stack([np.cos(upcol), np.sin(upcol), np.cos(downcol),
                            np.sin(downcol)]).T + np.random.randn(n_seq,4)*inp_noise
        
        cuecol = np.where(cue>0, upcol, downcol)
        uncuecol = np.where(cue>0, downcol, upcol)
        
        cue += np.random.randn(n_seq)*inp_noise
        inps = np.zeros((n_seq,T, col_inp.shape[1] + net_inp +
                         1 + 1*self.go_cue))
        
        t_stim1 = np.random.choice(range(T_inp1 - jitter, T_inp1 + jitter + 1),
                                   (ndat, 1))
        t_stim2 = np.random.choice(range(T_inp2 - jitter, T_inp2 + jitter + 1),
                                   (ndat, 1))
        t_targ = np.random.choice(range(T_resp - jitter, T_resp + jitter + 1),
                                  (ndat, 1))
        
        
        t_stim1 = t_stim1 + np.arange(present_len).reshape((1, -1))
        t_stim2 = t_stim2 + np.arange(present_len).reshape((1, -1))

        comb_cue_t = np.concatenate((t_stim2[:n_seq//2], t_stim1[n_seq//2:]))
        comb_col_t = np.concatenate((t_stim1[:n_seq//2], t_stim2[n_seq//2:]))

        t_stim1 = np.concatenate(t_stim1.T)
        t_stim2 = np.concatenate(t_stim2.T)
        comb_cue_t = np.concatenate(comb_cue_t.T)
        comb_col_t = np.concatenate(comb_col_t.T)
        
        retro_cue = t_stim2
        retro_col = t_stim1
        pro_cue = t_stim1
        pro_col = t_stim2
        
        seq_inds = np.tile(np.arange(n_seq), present_len)
        col_inp = np.tile(col_inp, (present_len, 1))
        cue_rep = np.tile(cue, present_len)
        if retro_only:
            inps[seq_inds, retro_col, :4] = col_inp # retro
            inps[seq_inds, retro_cue, 4] = cue_rep
        elif pro_only:
            inps[seq_inds, pro_cue, 4] = cue_rep # pro
            inps[seq_inds, pro_col, :4] = col_inp
        else:
            inps[seq_inds, comb_cue_t, 4] = cue_rep
            inps[seq_inds, comb_col_t, :4] = col_inp
            
        inps[:,:,5:5+net_inp] = np.random.randn(n_seq, T, net_inp)*dyn_noise
        train_mask = np.zeros(inps.shape[2], dtype=bool)
        train_mask[4:4+net_inp] = True


        report_list = (np.cos(cuecol), np.sin(cuecol))
        if report_uncue_color:
            report_list = report_list + (np.cos(uncuecol), np.sin(uncuecol))
        if report_cue:
            report_list = report_list + (cue,)
        outs = np.stack(report_list)
        
        outputs = np.zeros((T, n_seq, outs.shape[0]))
        
        if self.go_cue:
            for i, targ_i in enumerate(np.squeeze(t_targ)):
                inps[i, targ_i:, -1] = 1
                outputs[targ_i:, i, :] = outs[:, i]
        
        # inps = np.zeros((ndat,T,1))
        # inps[np.arange(ndat),t_stim, 0] = cue
        
        # outs = np.stack([np.cos(cuecol), np.sin(cuecol), np.cos(uncuecol), np.sin(uncuecol)])
        # outs = np.stack([np.cos(cuecol), np.sin(cuecol)])

        return inps, outputs, train_mask

def plot_loss_summary(loss_a, tr_corr, tr_swap, val_corr, val_swap, fwid=5):
    f, (ax_l, ax_ang, ax_val) = plt.subplots(1, 3, figsize=(fwid*3, fwid),)
    ax_l.plot(loss_a)
    xs_full = np.arange(len(tr_corr))
    xs = np.arange(len(val_corr))
    gpl.plot_trace_werr(xs_full, np.abs(np.array(tr_corr).T), ax=ax_ang)
    gpl.plot_trace_werr(xs_full, np.abs(np.array(tr_swap).T), ax=ax_ang)
    gpl.plot_trace_werr(xs, np.abs(np.array(val_corr).T), ax=ax_val)
    gpl.plot_trace_werr(xs, np.abs(np.array(val_swap).T), ax=ax_val)
    
    gpl.add_hlines(np.pi/2, ax_ang)
    gpl.add_hlines(np.pi/2, ax_val)
    
def fit_ring(neurs, col1, col2, n_bins=32, acc=None, model=sklm.Ridge,
             acc_thr=.8, **kwargs):
    m = model(**kwargs)
    col1_spl = swan.spline_color(col1, n_bins).T
    col2_spl = swan.spline_color(col2, n_bins).T
    x = np.concatenate((col1_spl, col2_spl), axis=1)
    if acc is not None:
        acc_mask = np.abs(acc) < acc_thr
        x = x[acc_mask]
        neurs = neurs[acc_mask]
    m.fit(x, neurs)
    score = m.score(x, neurs)
    return m.coef_, score

def decode_trls(neurs, ring_coeffs, **kwargs):
    out_cols = np.zeros((len(neurs), 2))
    for i, neur_trl in enumerate(neurs):
        c1, c2, _ = decode_colors(neur_trl, ring_coeffs, **kwargs)
        out_cols[i] = c1, c2
    return out_cols

def decode_colors(neur_trl, ring_coeffs, **kwargs):
    nbins = int(ring_coeffs.shape[1]/2)
    
    def func(cols):
        col_vecs = list(swan.spline_color(np.array([col]), nbins)
                       for col in cols)
        col_vec = np.concatenate(col_vecs, axis=0)
        out = np.dot(ring_coeffs, col_vec)
        loss = np.sum((out - neur_trl)**2)
        return loss
    
    res = sopt.minimize(func, (np.pi, np.pi), bounds=((0, 2*np.pi),)*2,
                         **kwargs)
    return res.x[0], res.x[1], res
    
def make_training_data(t_inp1, t_inp2, t_resp, total_t, jitter=3, ndat=2000,
                       n_cols=64, train_noise=0, train_z_noise=.1,
                       go_cue=True, net_size=None, make_val_set=False,
                       val_frac=.2, **kwargs):
    task = Task(n_cols, t_inp1, t_inp2, t_resp, total_t, go_cue=go_cue)

    out  = task.generate_data(ndat, jitter, train_noise, train_z_noise,
                              net_size=net_size, **kwargs)
    inps, outs, upcol, downcol, cue, train_inp_mask = out
    
    cuecol = np.where(cue>0, upcol, downcol)
    uncuecol = np.where(cue>0, downcol, upcol)

    inputs = torch.tensor(inps)
    outputs = torch.tensor(outs)
    components = (upcol, downcol, cue, cuecol, uncuecol)
    td_out = (inputs, outputs, train_inp_mask, components)
    if make_val_set:
        out  = task.generate_data(int(ndat*val_frac), jitter, train_noise,
                                  train_z_noise, net_size=net_size, **kwargs)
        inps, outs, upcol, downcol, cue, _ = out
        cuecol = np.where(cue>0, upcol, downcol)
        uncuecol = np.where(cue>0, downcol, upcol)
        val_inputs = torch.tensor(inps)
        val_outputs = torch.tensor(outs)
        val_components = (upcol, downcol, cue, cuecol, uncuecol)
        val_set = (val_inputs, val_outputs, val_components)

        td_out = td_out + (val_set,)
    return td_out

def make_task_rnn(inputs, outputs, net_size, basis=None, train_mask=None,
                  **kwargs):
    n_in = inputs.shape[-1]
    n_out = outputs.shape[-1]
    net = TaskRNN(n_in, net_size, n_out, basis=basis, train_mask=train_mask,
                  **kwargs)
    return net

class TaskRNN:

    def __init__(self, inp_dim, rnn_dim, out_dim, basis=None, train_mask=None,
                 activity_reg=0):
        if train_mask is None:
            train_mask = np.ones(inp_dim, dtype=bool)
        self.dec = nn.Linear(rnn_dim, out_dim, bias=True)
        self.rnn = nn.RNN(inp_dim, rnn_dim, 1, nonlinearity='relu')
        self.net = students.GenericRNN(self.rnn, students.GausId(out_dim),
                                       decoder=self.dec,
                                       z_dist=students.GausId(rnn_dim),
                                       beta=activity_reg)
        
        if basis is not None:
            with torch.no_grad():
                inp_w = np.append(basis[:,n_out:n_out+n_in-1],
                                  np.ones((N,1))/np.sqrt(N), axis=-1)
                net.rnn.weight_ih_l0.copy_(torch.tensor(inp_w).float())
                net.rnn.weight_ih_l0.requires_grad = False
        with torch.no_grad():
            weight_mask = torch.tensor(np.ones_like(self.net.rnn.weight_ih_l0))
            weight_mask[:, train_mask] = 0
            ident_weights = torch.tensor(np.identity(rnn_dim), dtype=torch.float)
            self.net.rnn.weight_ih_l0[:, train_mask] = ident_weights
            self.net.rnn.weight_ih_l0.register_hook(
                lambda grad: grad.mul_(weight_mask))
 
    def _pre_proc(self, inp, transpose=False):
        out = inp.float()
        if transpose:
            out = out.transpose(0, 1)
        return out
                        
    def fit(self, inputs, outputs, components, lr=1e-3, batch_size=200,
            shuffle=True, n_epochs=2500, validation_set=None):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        dset = torch.utils.data.TensorDataset(
            self._pre_proc(inputs),
            self._pre_proc(outputs, transpose=True))

        dl = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                         shuffle=shuffle)

        init_params = list(self.net.parameters())
        train_loss = []
        train_corr, train_swap = [], []
        val_corr, val_swap = [], []
        for epoch in tqdm(range(n_epochs)):
            loss = self.net.grad_step(dl, optimizer, init_state=False,
                                      only_final=False)
            train_loss.append(loss)
            tr_out = self.eval_net(inputs)
            tr_cue_col, tr_uncue_col = components[-2:]
            out = compute_diffs(tr_out, tr_cue_col, tr_uncue_col)
            _, (tr_corr_diff, tr_swap_diff) = out
            train_corr.append(tr_corr_diff)
            train_swap.append(tr_swap_diff)
            if validation_set is not None and np.mod(epoch, 10) == 0:
                val_inp, val_out, val_comp = validation_set
                val_cue_col, val_uncue_col = val_comp[-2:]
                net_out = self.eval_net(val_inp)

                out = compute_diffs(net_out, val_cue_col, val_uncue_col)
                _, (val_corr_diff, val_swap_diff) = out
                val_corr.append(val_corr_diff)
                val_swap.append(val_swap_diff)
        out = (train_loss, train_corr, train_swap)
        if validation_set is not None:
            out = out + (val_corr, val_swap)
        return out

    def eval_net(self, inputs):
        inputs = self._pre_proc(inputs, transpose=True)
        outputs = self.net(inputs)
        return outputs

    def plot_response_scatter(self, inputs, fwid=3):
        out = self.eval_net(inputs)

        _plot_response_scatter(out)
    
    def plot_response_hist(self, inputs, *targs, **kwargs):
        out = self.eval_net(inputs)
        _plot_response_hist(out, *targs, **kwargs)

def _plot_response_scatter(outs, fwid=3):
    f, (ax_cue, ax_uncue) = plt.subplots(1, 2, figsize=(2*fwid, fwid),
                                         sharex=True, sharey=True)
    outs = outs[0].detach().numpy()
    ax_cue.plot(outs[:, 0], outs[:, 1], 'o')
    ax_uncue.plot(outs[:, 2], outs[:, 3], 'o')
        
def _plot_response_hist(outs, *targs, fwid=3, n_bins=20, add_zero=True):
    if add_zero:
        targs = targs + (np.zeros_like(targs[0]),)
    theta, diffs = compute_diffs(outs, *targs)

    n_targs = len(targs)
    f, axs = plt.subplots(1, n_targs, figsize=(n_targs*fwid, fwid),
                          sharey=True)
    bins = np.linspace(-np.pi, np.pi, n_bins)
    for i, diff in enumerate(diffs):
        axs[i].hist(diff, density=True, bins=bins)
            
def compute_diffs(outputs, *cols):
    theta = np.arctan2(outputs[0][:,1].detach(),
                       outputs[0][:,0].detach()).numpy()
    diffs = []
    for col in cols:
        diff = np.arctan2(np.exp(col*1j - theta*1j).imag,
                          np.exp(col*1j - theta*1j).real)
        diffs.append(diff)
    return theta, diffs

        
