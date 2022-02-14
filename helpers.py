import socket
import os
import sys

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx


def convexify(cols, bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    dc = 2*np.pi/(len(bins))
    
    # get the nearest bin
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    dist_near = np.abs(distances).min(0)
    nearest = np.abs(distances).argmin(0)
    # see if the color is to the "left" or "right" of that bin
    sec_near = np.sign(distances[nearest,np.arange(len(cols))]+1e-8).astype(int) # add epsilon to handle 0
    # fill in the convex array
    alpha = np.zeros((len(bins),len(cols)))
    alpha[nearest, np.arange(len(cols))] = (dc-dist_near)/dc
    alpha[np.mod(nearest-sec_near,len(bins)), np.arange(len(cols))] = 1 - (dc-dist_near)/dc
    
    return alpha

def softmax_cols(cols, bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    num_bins = len(bins)
    dc = 2*np.pi/num_bins
    
    # get the nearest bin
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    alpha = np.exp(np.abs(distances))/np.exp(np.abs(distances)).sum(0)
    
    return alpha


def box_conv(X, len_filt):
    '''
    Convolves X with a square filter, for all possible padding offsets
    '''
    T = X.shape[1]
    N = X.shape[0]
    
    f = np.eye(T+len_filt,T)
    f[np.arange(T)+len_filt,np.arange(T)] = -1
    filt = np.cumsum(f,0)
    
    x_pad = np.stack([np.concatenate([np.zeros((N,len_filt-i)), X, np.zeros((N,i))],axis=1) for i in range(len_filt+1)])
    filted = x_pad@filt
    
    return filted

def folder_hierarchy(dset_info):
    """
    dset info should have at least the following fields:
        session, tzf, tbeg, tend, twindow, tstep, num_bins
    """
    FOLDS = ('/{num_bins}_colors/'
        'sess_{session}/{tzf}/'
        '{tbeg}-{tend}-{twindow}_{tstep}/'
        'pca_{pca_thrs}_{do_pca}/'
        'impute_{impute_nan}/'
        '{color_weights}_knots/'
        '{regions}/')
    if dset_info['shuffle_probs']:
        FOLDS += 'shuffled/'
    
    return FOLDS.format(**dset_info)


# def file_extension():
# 	return 0