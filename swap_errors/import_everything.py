
CODE_DIR = '/home/kelarion/github/'
SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'

import socket
import os
import sys
import pickle as pkl

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx
from tqdm import tqdm
import pystan as ps
import arviz as az

sys.path.append(CODE_DIR)
sys.path.append(CODE_DIR+'jeffcode/')
import general.data_io as gio
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan
# import swap_errors.visualization as swv
import general.neural_analysis as na
import general.plotting as gpl
import general.stan_utility as su

import helpers as hlp

sys.path.append(CODE_DIR+'repler/src/')
import util