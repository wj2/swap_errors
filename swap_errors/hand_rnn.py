
import functools as ft
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.integrate as spi

import general.plotting as gpl

class LowRankRNN:

    def __init__(self, n_units, transfer_func=np.tanh, **weight_kwargs):
        self.n_units = n_units
        self.f = transfer_func
        out = self.make_weights(n_units, **weight_kwargs)
        self.w, self.v_inp, self.v_out = out

    def simulate_n_trials(self, n_trls, t_len, inits=None, delta_t=.1,
                          **kwargs):
        if inits is None:
            inits = np.array(list(get_g_vec(self.n_units, 0, 1)
                                  for _ in range(n_trls)))
        n_ts = int(t_len/delta_t)
        out_all = np.zeros((n_trls, n_ts, self.n_units))
        out_inp = {i:np.zeros((n_trls, len(self.v_inp[i]), n_ts))
                   for i in self.v_inp.keys()}
        out_out = {i:np.zeros((n_trls, len(self.v_inp[i]), n_ts))
                   for i in self.v_out.keys()}
        for i in range(n_trls):
            ts, out = self.integrate(t_len, init=inits[i], delta_t=delta_t,
                                     **kwargs)
            inp_p = self.compute_inp_overlap(out)
            out_p = self.compute_out_overlap(out)
            for j in inp_p.keys():
                out_inp[j][i] = inp_p[j]
                out_out[j][i] = out_p[j]
            out_all[i] = out
        return ts, out_all, out_inp, out_out, inits

    def make_weights(self, n_units, **kwargs):
        r, v_inp, v_out = self.choose_vectors(n_units, **kwargs)
        w_mat = self._combine_weights(r, v_inp, v_out)
        return w_mat, v_inp, v_out

    def _combine_weights(self, disorder, v_inp, v_out):
        m = 0
        for i in v_inp.keys():
            gen = (np.outer(*x) for x in zip(v_inp[i], v_out[i]))
            m_i = np.sum(list(gen), axis=0) / self.n_units
            m = m + m_i
        w_mat = disorder + m
        return w_mat
    
    def integrate(self, t_len, delta_t=.1, init=None, inp=0):
        if init is None:
            init = np.zeros(self.n_units)
        ts = np.linspace(0, t_len, int(t_len/delta_t))
        func = ft.partial(_integ_eq, w_mat=self.w, inp=inp, func=self.f)
        out = spi.odeint(func, init, ts)
        return ts, out

    def _compute_vec_overlap(self, z, vecs):
        overlaps = np.zeros((len(vecs), z.shape[0]))
        for i, vec in enumerate(vecs):
            overlaps[i] = np.dot(self.f(z), vec) / self.n_units
        return overlaps
    
    def compute_inp_overlap(self, z):
        return {i:self._compute_vec_overlap(z, v_inp_i)
                for i, v_inp_i in self.v_inp.items()}

    def compute_out_overlap(self, z):
        return {i:self._compute_vec_overlap(z, v_out_i)
                for i, v_out_i in self.v_out.items()}

def plot_out_trls(vec_proj, axs=None, fwid=5):
    if axs is None:
        figsize = (fwid*len(vec_proj), fwid)
        f, axs = plt.subplots(1, len(vec_proj),
                              figsize=figsize, squeeze=False)
        axs = axs[0]
    for k, proj in vec_proj.items():
        for act_j in proj:
            gpl.plot_colored_line(act_j[0], act_j[1], 
                                  func=gpl.line_speed_func,
                                  ax=axs[k])
            axs[k].plot(act_j[0, 0], act_j[1, 0], 'o', color='k',
                        markersize=5)
        axs[k].set_aspect('equal')
    return axs

def _integ_eq(r, t, w_mat, inp, func):
    drdt = -r + np.dot(w_mat, func(r)) + inp
    return drdt
    
def make_random_weights(n):
    w = sts.norm(0, np.sqrt(1/n)).rvs((n, n))
    return w

def get_g_vec(n, mean, std):
    vec = sts.norm(mean, std).rvs(n)
    return vec

def make_ring_weights(n_units, g, rho, w_std):
    y1 = get_g_vec(n_units, 0, 1)
    y2 = get_g_vec(n_units, 0, 1)
        
    x1 = get_g_vec(n_units, 0, 1)
    x2 = get_g_vec(n_units, 0, 1)
    x3 = get_g_vec(n_units, 0, 1)
    x4 = get_g_vec(n_units, 0, 1)

    m1 = np.sqrt(w_std**2 - rho**2)*x1 + rho*y1
    m2 = np.sqrt(w_std**2 - rho**2)*x2 + rho*y2
    
    n1 = np.sqrt(w_std**2 - rho**2)*x3 + rho*y1
    n2 = np.sqrt(w_std**2 - rho**2)*x4 + rho*y2

    return (n1, n2), (m1, m2)

class NRingRNN(LowRankRNN):

    def __init__(self, n_rings, *args, **kwargs):
        super().__init__(*args, n_rings=n_rings, **kwargs)
    
    def choose_vectors(self, n_units, g=1., rho=1.6, w_std=2., n_rings=1):
        r = g*make_random_weights(n_units)
        vd_in = {}
        vd_out = {}
        for i in range(n_rings):
            v_inp_i, v_out_i = make_ring_weights(n_units, g=g, rho=rho,
                                                 w_std=w_std)
            vd_in[i] = v_inp_i
            vd_out[i] = v_out_i
        return r, vd_in, vd_out

    
        
