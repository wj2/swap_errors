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

#%%

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
    
    N_cols = 32

    train_noise = 0.0
    train_z_noise = 0.01

    basis = la.qr( np.random.randn(N,N))[0]

    task = Task(N_cols, T_inp1, T_inp2, T_resp, T, go_cue=go_cue)

    inps, outs, upcol, downcol, cue = task.generate_data(ndat, jitter, train_noise, train_z_noise)

    n_in = inps.shape[-1]
    inps = np.concatenate([inps, np.random.randn(ndat, T, N)*train_z_noise], axis=-1)


    cuecol = np.where(cue>0, upcol, downcol)
    uncuecol = np.where(cue>0, downcol, upcol)

    n_out = outs.shape[-1]

    inputs = torch.tensor(inps)
    outputs = torch.tensor(outs)

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
    with torch.no_grad():
        dec.weight.copy_( torch.tensor(basis[:,:n_out].T).float())
        dec.weight.requires_grad = False

    rnn = nn.RNN(n_in+N, N, 1, nonlinearity='relu')

    # net = students.GenericRNN(rnn, students.GausId(outs.shape[0]), fix_decoder=True, decoder=dec, z_dist=z_prior)
    net = students.GenericRNN(rnn, students.GausId(n_out), decoder=dec, z_dist=students.GausId(N), beta=0)
    # net = students.GenericRNN(rnn, students.GausId(outs.shape[0]), fix_decoder=False)
    
    # with torch.no_grad():
    #     net.rnn.inp2hid.weight.copy_(torch.tensor(basis[:,:5]).float())
    #     net.rnn.inp2hid.weight.requires_grad = False
    #     net.rnn.inp2hid.bias.requires_grad = False
    with torch.no_grad():
        inp_w = np.append(basis[:,n_out:n_out+n_in], np.eye(N), axis=-1)
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

    fake_inputs, _ = task.generate_sequences(upcol, downcol, cue, jitter=0, 
                                             inp_noise=test_inp_noise, dyn_noise=test_z_noise, retro_only=True)
    fake_inputs = np.concatenate([fake_inputs, np.random.randn(ndat, T, N)*test_z_noise], axis=-1)

    train_loss = []
    scat = []
    swap_scat = []
    for epoch in tqdm(range(n_epoch)):
        loss = net.grad_step(dl, optimizer, init_state=False, only_final=False)
    
        train_loss.append(loss)
        
        if not np.mod(epoch, 10):
            z_retro = net(torch.tensor(fake_inputs).transpose(1,0).float())
            
            theta = np.arctan2(z_retro[0][:,1].detach(),z_retro[0][:,0].detach()).numpy()
            diff = np.arctan2(np.exp(cuecol*1j - theta*1j).imag, np.exp(cuecol*1j - theta*1j).real)
            
            scat.append(diff)
            
            swap_diff = np.arctan2(np.exp(uncuecol*1j - theta*1j).imag, np.exp(uncuecol*1j - theta*1j).real)
            swap_scat.append(swap_diff)

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
    test_z_noise = 0.01
    test_inp_noise = 0.0

    # fake_inputs = np.zeros((ndat,T,6))
    # fake_inputs[:,0, :4] = col_inp + np.random.randn(ndat,4)*test_inp_noise
    # fake_inputs[:,T//2, 4] = cue
    # fake_inputs[:,:,5] = np.random.randn(ndat,T)*test_z_noise
    fake_inputs, _ = task.generate_sequences(upcol, downcol, cue, jitter=0, 
                                             inp_noise=test_inp_noise, dyn_noise=test_z_noise, retro_only=True)
    fake_inputs = np.concatenate([fake_inputs, np.random.randn(ndat, T, N)*test_z_noise], axis=-1)

    
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

    plot_circs = cue_circs
    # plot_circs = up_circs

    U, S = util.pca(plot_circs[:,task.T_inp2:task.T_resp,:].reshape((-1,N)).T)
    # U = dec.weight.detach().numpy()[[0,1,4],:].T
    # U = wa['estimator'][0].coef_

    proj_z = plot_circs@U[:,:3]
    # proj_z = util.pca_reduce(wa['estimator'][0].coef_@plot_circs.reshape((-1,100)).T, num_comp=3)
    # proj_z = proj_z.reshape((128, 35, 3))
    
    ani.ScatterAnime3D(proj_z[...,0],proj_z[...,1],proj_z[...,2], c=np.tile(cbins,2), cmap='hsv',
                       rotation_period=100, after_period=50, view_period=20).save(SAVE_DIR+'temp.mp4', fps=10)

    #%%

    
    #%%
    fake_inputs, _ = task.generate_sequences(upcol, downcol, cue, jitter=0, 
                                         inp_noise=0, dyn_noise=0, retro_only=True)
    
    z_retro = net(torch.tensor(fake_inputs).transpose(1,0).float())

    fake_inputs, _ = task.generate_sequences(upcol, downcol, cue, jitter=0, 
                                         inp_noise=0, dyn_noise=0, pro_only=True)

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
    k=3

    plt.subplot(2,2,1)
    plt.imshow(rr[...,:k].sum(-1)/l1[:,:k].sum(-1)[:,None], clim=[0,1], cmap='binary')
    plt.plot([task.T_inp1, task.T_inp1], plt.ylim(), 'k--')
    plt.plot([task.T_inp2, task.T_inp2], plt.ylim(), 'k-.')
    plt.plot([task.T_resp, task.T_resp], plt.ylim(), 'k:')
    plt.plot(plt.xlim(), [task.T_inp1, task.T_inp1], 'k--')
    plt.plot(plt.xlim(), [task.T_inp2, task.T_inp2], 'k-.')
    plt.plot(plt.xlim(), [task.T_resp, task.T_resp], 'k:')

    plt.subplot(2,2,3)
    plt.imshow(pr[...,:k].sum(-1)/l1[:,:k].sum(-1)[:,None], clim=[0,1], cmap='binary')
    plt.plot([task.T_inp1, task.T_inp1], plt.ylim(), 'k--')
    plt.plot([task.T_inp2, task.T_inp2], plt.ylim(), 'k-.')
    plt.plot([task.T_resp, task.T_resp], plt.ylim(), 'k:')
    plt.plot(plt.xlim(), [task.T_inp1, task.T_inp1], 'k--')
    plt.plot(plt.xlim(), [task.T_inp2, task.T_inp2], 'k-.')
    plt.plot(plt.xlim(), [task.T_resp, task.T_resp], 'k:')

    plt.subplot(2,2,4)
    plt.imshow(pp[...,:k].sum(-1)/l2[:,:k].sum(-1)[:,None], clim=[0,1], cmap='binary')
    plt.plot([task.T_inp1, task.T_inp1], plt.ylim(), 'k--')
    plt.plot([task.T_inp2, task.T_inp2], plt.ylim(), 'k-.')
    plt.plot([task.T_resp, task.T_resp], plt.ylim(), 'k:')
    plt.plot(plt.xlim(), [task.T_inp1, task.T_inp1], 'k--')
    plt.plot(plt.xlim(), [task.T_inp2, task.T_inp2], 'k-.')
    plt.plot(plt.xlim(), [task.T_resp, task.T_resp], 'k:')


    #%%
    fake_inputs, _ = task.generate_sequences(upcol, downcol, cue, jitter=0, 
                                         inp_noise=0, dyn_noise=0, retro_only=True)


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







