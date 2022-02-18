
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
        
        upcol = np.random.choice(np.linspace(0,2*np.pi,N_cols), n_seq)
        downcol = np.random.choice(np.linspace(0,2*np.pi,N_cols), n_seq)
            
        cue = np.random.choice([-1.,1.], n_seq) 
        
        return upcol, downcol, cue
        
    def generate_sequences(self, upcol, downcol, cue, jitter=3, inp_noise=0.0,
                           dyn_noise=0.0, new_T=None, retro_only=False,
                           pro_only=False):
        
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
            
        
        col_inp = np.stack([np.cos(upcol), np.sin(upcol), np.cos(downcol), np.sin(downcol)]).T + np.random.randn(n_seq,4)*inp_noise
        
        cuecol = np.where(cue>0, upcol, downcol)
        uncuecol = np.where(cue>0, downcol, upcol)
        
        cue += np.random.randn(n_seq)*inp_noise
        inps = np.zeros((n_seq,T, col_inp.shape[1] + net_inp + 1*self.go_cue))
        
        if jitter>0:
            t_stim1 = np.random.choice(range(T_inp1 - jitter, T_inp1 + jitter), ndat)
            t_stim2 = np.random.choice(range(T_inp2 - jitter, T_inp2 + jitter), ndat)
            t_targ = np.random.choice(range(T_resp - jitter, T_resp + jitter), ndat)
        else:
            t_stim1 = np.ones(n_seq, dtype=int)*(T_inp1)
            t_stim2 = np.ones(n_seq, dtype=int)*(T_inp2)
            t_targ = np.ones(n_seq, dtype=int)*(T_resp)
        
        if retro_only:
            inps[np.arange(n_seq),t_stim1,:4] = col_inp # retro
            inps[np.arange(n_seq),t_stim2, 4] = cue
        elif pro_only:
            inps[np.arange(n_seq),t_stim1,4] = cue # pro
            inps[np.arange(n_seq),t_stim2, :4] = col_inp
        else:
            inps[np.arange(n_seq//2),t_stim1[:n_seq//2],:4] = col_inp[:n_seq//2,:] # retro
            inps[np.arange(n_seq//2),t_stim2[:n_seq//2], 4] = cue[:n_seq//2]
            
            inps[np.arange(n_seq//2, n_seq),t_stim1[n_seq//2:],4] = cue[n_seq//2:] # pro
            inps[np.arange(n_seq//2, n_seq),t_stim2[n_seq//2:], :4] = col_inp[n_seq//2:,:]
        
        inps[:,:,4:4+net_inp] = np.random.randn(n_seq, T, net_inp)*dyn_noise
        train_mask = np.zeros(inps.shape[2], dtype=bool)
        train_mask[4:4+net_inp] = True
        
        if self.go_cue:
            inps[np.arange(n_seq), t_targ, -1] = 1
        
        # inps = np.zeros((ndat,T,1))
        # inps[np.arange(ndat),t_stim, 0] = cue
        
        outs = np.concatenate([np.stack([np.cos(cuecol), np.sin(cuecol),
                                         np.cos(uncuecol), np.sin(uncuecol)]),
                               cue[None,:]], axis=0)
        # outs = np.stack([np.cos(cuecol), np.sin(cuecol), np.cos(uncuecol), np.sin(uncuecol)])
        # outs = np.stack([np.cos(cuecol), np.sin(cuecol)])

        outputs = np.zeros((T, n_seq, outs.shape[0]))
        outputs[t_targ,np.arange(n_seq),:] = outs.T
        outputs = np.cumsum(outputs, axis=0)

        return inps, outputs, train_mask
 
def make_training_data(t_inp1, t_inp2, t_resp, total_t, jitter=3, ndat=2000,
                       n_cols=64, train_noise=0, train_z_noise=.1,
                       go_cue=True, net_size=None):
    task = Task(n_cols, t_inp1, t_inp2, t_resp, total_t, go_cue=go_cue)

    out  = task.generate_data(ndat, jitter, train_noise, train_z_noise,
                              net_size=net_size)
    inps, outs, upcol, downcol, cue, train_inp_mask = out
    
    cuecol = np.where(cue>0, upcol, downcol)
    uncuecol = np.where(cue>0, downcol, upcol)

    inputs = torch.tensor(inps)
    outputs = torch.tensor(outs)
    components = (upcol, downcol, cue, cuecol, uncuecol)
    
    return inputs, outputs, train_inp_mask, components

def make_task_rnn(inputs, outputs, net_size, basis=None, train_mask=None):
    n_in = inputs.shape[-1]
    n_out = outputs.shape[-1]
    net = TaskRNN(n_in, net_size, n_out, basis=basis, train_mask=train_mask)
    return net

class TaskRNN:

    def __init__(self, inp_dim, rnn_dim, out_dim, basis=None, train_mask=None):
        if train_mask is None:
            train_mask = np.ones(inp_dim, dtype=bool)
        self.dec = nn.Linear(rnn_dim, out_dim, bias=True)
        self.rnn = nn.RNN(inp_dim, rnn_dim, 1, nonlinearity='relu')
        self.net = students.GenericRNN(self.rnn, students.GausId(out_dim),
                                       decoder=self.dec,
                                       z_dist=students.GausId(rnn_dim),
                                       beta=0)
        
        if basis is not None:
            # with torch.no_grad():
            #     dec.weight.copy_( torch.tensor(basis[:,:outs.shape[0]].T).float())
            #     dec.weight.requires_grad = False
            # with torch.no_grad():
            #     net.rnn.inp2hid.weight.copy_(torch.tensor(basis[:,:5]).float())
            #     net.rnn.inp2hid.weight.requires_grad = False
            #     net.rnn.inp2hid.bias.requires_grad = False
            with torch.no_grad():
                inp_w = np.append(basis[:,n_out:n_out+n_in-1],
                                  np.ones((N,1))/np.sqrt(N), axis=-1)
                net.rnn.weight_ih_l0.copy_(torch.tensor(inp_w).float())
                net.rnn.weight_ih_l0.requires_grad = False
                # net.rnn.bias_ih_l0.requires_grad = False
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
        
                
    def fit(self, inputs, outputs, lr=1e-3, batch_size=200, shuffle=True,
            n_epochs=2500):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        dset = torch.utils.data.TensorDataset(
            self._pre_proc(inputs),
            self._pre_proc(outputs, transpose=True))

        dl = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                         shuffle=shuffle)

        init_params = list(self.net.parameters())
        train_loss = []
        scats = []
        for epoch in tqdm(range(n_epochs)):
            loss = self.net.grad_step(dl, optimizer, init_state=False,
                                      only_final=False)
            print(loss)
            train_loss.append(loss)
            
        return train_loss

    def eval_net(self, inputs):
        inputs = self._pre_proc(inputs, transpose=True)
        outputs = self.net(inputs)
        return outputs

    def compute_diffs(self, outputs, *cols):
        theta = np.arctan2(outputs[0][:,1].detach(),
                           outputs[0][:,0].detach()).numpy()
        diffs = []
        for col in cols:
            diff = np.arctan2(np.exp(col*1j - theta*1j).imag,
                              np.exp(col*1j - theta*1j).real)
            diffs.append(diff)
        return theta, diffs
