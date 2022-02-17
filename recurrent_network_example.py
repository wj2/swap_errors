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

#%%

class Task(object):
    def __init__(self, num_cols, T_inp1, T_inp2, T_resp, T_tot, go_cue=False):
        
        self.num_col = num_cols
        
        self.T_inp1 = T_inp1
        self.T_inp2 = T_inp2
        self.T_resp = T_resp
        self.T_tot = T_tot
        
        self.go_cue = go_cue
        
    def generate_data(self, n_seq, jitter=3, inp_noise=0.0, dyn_noise=0.0,
                      **kwargs):
        
        upcol, downcol, cue = self.generate_colors(n_seq)
        
        inps, outs, train_mask = self.generate_sequences(
            upcol, downcol, cue, jitter, inp_noise, dyn_noise, **kwargs)
        
        return inps, outs, upcol, downcol, cue, train_mask
        
    
    def generate_colors(self, n_seq):
        
        upcol = np.random.choice(np.linspace(0,2*np.pi, self.num_col), n_seq)
        downcol = np.random.choice(np.linspace(0,2*np.pi, self.num_col), n_seq)
            
        cue = np.random.choice([-1.,1.], n_seq) 
        
        return upcol, downcol, cue
        
    def generate_sequences(self, upcol, downcol, cue, jitter=3, inp_noise=0.0,
                           dyn_noise=0.0, new_T=None, net_size=None):
        ndat = upcol.shape[0]
        T_inp1 = self.T_inp1
        T_inp2 = self.T_inp2
        T_resp = self.T_resp
        if new_T is None:
            T = self.T_tot
        else:
            T = new_T
        n_seq = len(upcol)

        if net_size is None and dyn_noise > 0:
            raise IOError('cannot do dynamics noise without providing the net '
                          'size')
        elif dyn_noise > 0:
            net_inp = net_size
        else:
            net_inp = 0
            
        
        col_inp = np.stack([np.cos(upcol), np.sin(upcol), np.cos(downcol),
                            np.sin(downcol)]).T
        col_inp = col_inp + np.random.randn(n_seq,4)*inp_noise
        
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

#%%

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

if __name__ == '__main__':
 
    # TODO: linear reg to get color reps, track over time
    N = 100 # network size
    
    go_cue = True
    # go_cue = False

    T_inp1 = 5
    T_inp2 = 15
    T_resp = 25
    T = 35

    jitter = 3 # range of input jitter (second input arrives at time t +/- jitter)

    ndat = 2000

    N_cols = 64

    train_noise = 0.0
    train_z_noise = 0.1

    inputs, outputs, components = make_training_data(
        T_inp1, T_inp2, T_resp, T, jitter=jitter, ndat=ndat, n_cols=N_cols,
        train_noise=train_noise, train_z_noise=train_z_noise, go_cue=go_cue)
    basis = la.qr( np.random.randn(N,N))[0]
    inps, outs, upcol, downcol, cue, cuecol, uncuecol = components
    
    #%%

    # try constraining the recurrent weights to be rotational (?) riemannian SGD
    # try combinations of train and test noise
    # input and output representations
    # "reasonable amount of noise" -- choose based on response noise
    
    n_epoch = 5000

    swap_prob = 0.15

    z_prior = None
    # z_prior = students.GausId(N)

    dec = nn.Linear(N, n_out, bias=True)
    # with torch.no_grad():
    #     dec.weight.copy_( torch.tensor(basis[:,:outs.shape[0]].T).float())
    #     dec.weight.requires_grad = False

    rnn = nn.RNN(n_in, N, 1, nonlinearity='relu')
    
    # net = students.GenericRNN(rnn, students.GausId(outs.shape[0]), fix_decoder=True, decoder=dec, z_dist=z_prior)
    net = students.GenericRNN(rnn, students.GausId(n_out), decoder=dec, z_dist=students.GausId(N), beta=0)
    # net = students.GenericRNN(rnn, students.GausId(outs.shape[0]), fix_decoder=False)


    # with torch.no_grad():
    #     net.rnn.inp2hid.weight.copy_(torch.tensor(basis[:,:5]).float())
    #     net.rnn.inp2hid.weight.requires_grad = False
    #     net.rnn.inp2hid.bias.requires_grad = False
    with torch.no_grad():
        inp_w = np.append(basis[:,n_out:n_out+n_in-1], np.ones((N,1))/np.sqrt(N), axis=-1)
        net.rnn.weight_ih_l0.copy_(torch.tensor(inp_w).float())
        net.rnn.weight_ih_l0.requires_grad = False
        # net.rnn.bias_ih_l0.requires_grad = False


    n_trn = int(0.8*ndat)   
    trn = np.random.choice(ndat,ndat,replace=False)
    tst = np.setdiff1d(range(ndat),trn)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    dset = torch.utils.data.TensorDataset(inputs[trn,...].float(),
                                          outputs[:,trn,:].float().transpose(0,1))

    # dset = torch.utils.data.TensorDataset(inputs[trn,:,:].float(),
    #                                       outputs[trn,:].float(),
    #                                       torch.tensor(init_torus[:,trn]).float().T)

    dl = torch.utils.data.DataLoader(dset, batch_size=200, shuffle=True)
    n_compute = np.min([len(tst),100])

    init_params = list(net.parameters())
    
    train_loss = []
    scats = []
    for epoch in tqdm(range(n_epoch)):
        loss = net.grad_step(dl, optimizer, init_state=False, only_final=False)
    
        train_loss.append(loss)
    
        # z_retro = net(torch.tensor(fake_inputs).transpose(1,0).float())
        
        # scats.append([z_retro[0][:,0].detach(),z_retro[0][:,1].detach()])
    
        # running_loss = 0
        # for i, (inp, out) in enumerate(dl):
        #     optimizer.zero_grad()
        
        #     hid = net.init_hidden(inp.size(0))
        #     pred, _ = net(inp.transpose(0,1), hid)
        
        #     # swap = np.where(np.random.rand(len(out))<swap_prob)[0]
        #     # out[swap][...,[0,1,2,3]] = out[swap][...,[2,3,0,1]]
        
        #     loss = -net.obs.distr(pred).log_prob(out[:,-1,:]).mean()
        
        #     running_loss += loss.item()
        
        #     loss.backward()
        #     optimizer.step()
    
        # train_loss.append(running_loss/(i+1))

    #%%
    test_z_noise = 0.0
    test_inp_noise = 0.0

    # fake_inputs = np.zeros((ndat,T,6))
    # fake_inputs[:,0, :4] = col_inp + np.random.randn(ndat,4)*test_inp_noise
    # fake_inputs[:,T//2, 4] = cue
    # fake_inputs[:,:,5] = np.random.randn(ndat,T)*test_z_noise
    fake_inputs, _ = task.generate_sequences(upcol, downcol, cue, jitter=0, inp_noise=test_inp_noise, dyn_noise=test_z_noise)

    z_retro = net(torch.tensor(fake_inputs).transpose(1,0).float())

    plt.subplot(121)
    theta = np.arctan2(z_retro[0][:,1].detach(),z_retro[0][:,0].detach()).numpy()
    diff = np.arctan2(np.exp(cuecol*1j - theta*1j).imag, np.exp(cuecol*1j - theta*1j).real)
    
    plt.hist(diff, bins=50, alpha=0.5, density=True)

    swap_diff = np.arctan2(np.exp(uncuecol*1j - theta*1j).imag, np.exp(uncuecol*1j - theta*1j).real)
    plt.hist(swap_diff, bins=50, alpha=0.5, density=True)

    plt.legend(['Error to cue', 'Error to swap'])

    plt.subplot(122)
    plt.scatter(z_retro[0][:,0].detach(),z_retro[0][:,1].detach(),c=cuecol,cmap='hsv')

    # %%
    cbins = np.unique(cuecol)

    up_cue_circ = np.array([z_retro[1][:,(upcol==c)&(cue>0),:].detach().numpy().mean(1) for c in cbins])
    down_cue_circ = np.array([z_retro[1][:,(downcol==c)&(cue<0),:].detach().numpy().mean(1) for c in cbins])

    up_uncue_circ = np.array([z_retro[1][:,(upcol==c)&(cue<0),:].detach().numpy().mean(1) for c in cbins])
    down_unue_circ = np.array([z_retro[1][:,(downcol==c)&(cue>0),:].detach().numpy().mean(1) for c in cbins])

    cue_circs = np.concatenate([up_cue_circ, down_cue_circ], axis=0)
    up_circs = np.concatenate([up_cue_circ, up_uncue_circ], axis=0)

    # plot_circs = cue_circs
    plot_circs = up_circs

    U, S = util.pca(plot_circs[:,:task.T_inp,:].reshape((-1,100)).T)
    # U = dec.weight.detach().numpy()[[0,1,4],:].T

    proj_z = plot_circs@U[:,:3]

    ani.ScatterAnime3D(proj_z[...,0],proj_z[...,1],proj_z[...,2], c=np.tile(cbins,2), cmap='hsv',
                       rotation_period=100, after_period=50, view_period=20).save(SAVE_DIR+'temp.mp4', fps=10)

    #%%
    fake_inputs = np.zeros((ndat,T,6))
    fake_inputs[:,0, :4] = col_inp
    fake_inputs[:,T//2, 4] = cue


    z_retro = net(torch.tensor(fake_inputs).transpose(1,0).float())
    
    fake_inputs = np.zeros((ndat,T,6))
    fake_inputs[:,0, 4] = cue
    fake_inputs[:,T//2, :4] = col_inp


    z_pro = net(torch.tensor(fake_inputs).transpose(1,0).float())
    
    U1, l1 = util.pca(z_retro[1].detach().numpy())
    U2, l2 = util.pca(z_pro[1].detach().numpy())
    
    U1V1 = np.einsum('tki,fkj->tfij',U1,U1)
    U2V2 = np.einsum('tki,fkj->tfij',U2,U2)
    U2V1 = np.einsum('tki,fkj->tfij',U2,U1)

    rr = np.einsum('tfij,tj->tfi',U1V1**2, l1)
    pr = np.einsum('tfij,tj->tfi',U2V1**2, l1)
    pp = np.einsum('tfij,tj->tfi',U2V2**2, l2)

    #%%
    k=8

    plt.subplot(2,2,1)
    plt.imshow(rr[...,:k].sum(-1)/l1[:,:k].sum(-1)[:,None], clim=[0,1])
    plt.plot([T//2, T//2], plt.ylim(), 'k--')
    plt.plot(plt.xlim(), [T//2, T//2], 'k--')

    plt.subplot(2,2,3)
    plt.imshow(pr[...,:k].sum(-1)/l1[:,:k].sum(-1)[:,None], clim=[0,1])
    plt.plot([T//2, T//2], plt.ylim(),'k--')
    plt.plot(plt.xlim(), [T//2, T//2], 'k--')

    plt.subplot(2,2,4)
    plt.imshow(pp[...,:k].sum(-1)/l2[:,:k].sum(-1)[:,None], clim=[0,1])
    plt.plot([T//2, T//2], plt.ylim() ,'k--')
    plt.plot(plt.xlim(), [T//2, T//2], 'k--')


    #%%
    fake_inputs = np.zeros((ndat,T,6))
    fake_inputs[:,0, :4] = col_inp
    fake_inputs[:,T//2, 4] = cue


    # z = net.transparent_forward(torch.tensor(fake_inputs).transpose(1,0).float(),
    #                             hidden=torch.tensor(init_torus.T[None,:,:]).float())[1].detach().numpy().transpose((1,0,2))
    z = net(torch.tensor(fake_inputs).transpose(1,0).float())[1].detach().numpy().transpose((2,0,1))


    clf = svm.LinearSVC()
    
    cue_dec = []
    ovlp = []
    pcs = []
    eigs = []
    for t in range(T):
    
        U_all, l_all = util.pca(z[:,t,:])
        pcs.append(U_all)
        eigs.append(l_all)
    
        avg_up_cue = np.array([z[:,t,(upcol==c)&(cue>0)].mean(1) for c in np.unique(upcol)])
        avg_down_cue = np.array([z[:,t,(downcol==c)&(cue<0)].mean(1) for c in np.unique(upcol)])
    
        avg_up_uncue = np.array([z[:,t,(upcol==c)&(cue<0)].mean(1) for c in np.unique(upcol)])
        avg_down_uncue = np.array([z[:,t,(downcol==c)&(cue>0)].mean(1) for c in np.unique(upcol)])
    
        U, mwa = util.pca(z[:,t,cue>0])
        V, mwa = util.pca(z[:,t,cue<0])
    
        clf.fit(z[:,t,:].T, cue>0)
        cue_dec.append(clf.score(z[:,t,:].T,cue>0))
    
        # plt.plot([(U[:,:k].T@z[:,t,cue<0]).var(1).sum(0)/z[:,t,cue<0].var(1).sum(0) for k in range(N)])
        plt.plot(np.cumsum((mwa[:,None]*(U.T@V)**2).sum(0))/np.sum(mwa))
    
        U, mwa = util.pca(avg_up_cue.T)
        V, mwa = util.pca(avg_down_cue.T)
        ovlp.append(np.sum((mwa[None,:]*(U[:,:2].T@V)**2).sum(0))/np.sum(mwa))
    
    
#%%







