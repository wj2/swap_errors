
import numpy as np
import matplotlib.pyplot as plt

import general.utility as u
import general.plotting as gpl
import ring_attractor.ring as ring

default_wf_params = (-10, 7, 0)

class FourRingModel:

    def __init__(self, n_units, tau=10, transfer_func=ring.relu_transfer,
                 weight_func=ring.cosine_weightfunc, r_inhib=1,
                 tf_params=(1,), r_beta=None,
                 r_excit=1.5, i_ws=1, wf_params=default_wf_params,
                 **kwargs):
        self.n_neurs = n_units

        r_u = ring.RingAttractor(n_units, tau, transfer_func, tf_params,
                                 weight_func=weight_func,
                                 wf_params=wf_params,
                                 divide_wm=True, **kwargs)
        r_l = ring.RingAttractor(n_units, tau, transfer_func, tf_params,
                                 weight_func=weight_func,
                                 wf_params=wf_params,
                                 divide_wm=True, **kwargs)
        r_t = ring.RingAttractor(n_units, tau, transfer_func, tf_params,
                                 weight_func=weight_func,
                                 wf_params=wf_params,
                                 divide_wm=True, **kwargs)
        r_d = ring.RingAttractor(n_units, tau, transfer_func, tf_params,
                                 weight_func=weight_func,
                                 wf_params=wf_params,
                                 divide_wm=True, **kwargs)
        
        self.rings = dict(r_u=r_u, r_l=r_l, r_t=r_t, r_d=r_d)
        self.thetas = r_u.thetas

        out = self.make_ring_connectivity(n_units, weight_scale=r_inhib)
        w_ul, w_lu, w_ut, w_ud = out[:4]
        w_lt, w_ld, w_td, w_dt = out[4:]

        r_inhib_corr = r_inhib/n_units
        r_excit_corr = r_excit/n_units
        if r_beta is None:
            r_beta = 0# -r_inhib_corr

        # p1 = np.random.choice(n_units, int(n_units/2), replace=False)
        p1 = np.arange(n_units)[::2]
        mask = np.isin(np.arange(n_units), p1).reshape((-1, 1))
        c1_weights = np.identity(n_units)*r_excit_corr*mask
        c2_weights = np.identity(n_units)*r_excit_corr*np.logical_not(mask)

        c1_rev_weights = np.roll(c1_weights, 1, 0)
        c2_rev_weights = np.roll(c2_weights, 1, 0)
        
        self.cue_mask = np.squeeze(mask)
        
        w_ut = -np.ones((n_units, n_units))*r_inhib_corr + c1_weights
        w_lt = -np.ones((n_units, n_units))*r_inhib_corr + c2_weights
        w_tu = -np.ones((n_units, n_units))*r_inhib_corr - c2_rev_weights
        w_tl = -np.ones((n_units, n_units))*r_inhib_corr - c1_rev_weights

        w_ud = -np.ones((n_units, n_units))*r_inhib_corr + c2_weights
        w_ld = -np.ones((n_units, n_units))*r_inhib_corr + c1_weights
        w_du = -np.ones((n_units, n_units))*r_inhib_corr - c1_rev_weights
        w_dl = -np.ones((n_units, n_units))*r_inhib_corr - c2_rev_weights
        
        w_dt = np.zeros((n_units, n_units))
        w_td = np.zeros((n_units, n_units))
        w_ul = np.zeros((n_units, n_units))
        w_lu = np.zeros((n_units, n_units))

        self.ring_weights = dict(r_u=((w_lu, r_l),
                                      (w_tu, r_t),
                                      (w_du, r_d)),
                                 r_l=((w_ul, r_u),
                                      (w_tl, r_t),
                                      (w_dl, r_d)),
                                 r_t=((w_ut, r_u),
                                      (w_lt, r_l),
                                      (w_dt, r_d)),
                                 r_d=((w_ud, r_u),
                                      (w_ld, r_l),
                                      (w_td, r_t)))
        # self.ring_weights = dict()

        out = self.make_input_connectivity(n_units, weight_scale=i_ws)
        w_iu, w_il, w_id, w_it = out
        self.input_weights = dict(r_u=(w_iu,),
                                  r_l=(w_il,),
                                  r_t=(w_it,),
                                  r_d=(w_id,))
        self.initialize_network()

    def compute_bump_statistics(self):
        return self.rings['r_u'].compute_bump_statistics()

    def compute_bump_statistics_empirical(self, **kwargs):
        return self.rings['r_u'].compute_bump_statistics_empirical(**kwargs)

    def compute_pc(self, cue_mag):
        return self.rings['r_u'].compute_pc(cue_mag)

    def compute_pc_empirical(self, cue_mag):
        return self.rings['r_u'].compute_pc_empirical(cue_mag)

    def estimate_noise(self, **kwargs):
        return self.rings['r_u'].estimate_noise(**kwargs)

    def estimate_pc_noise(self, cue_mag, **kwargs):
        return self.rings['r_u'].estimate_pc_noise(cue_mag, **kwargs)
        
    def make_ring_connectivity(self, n_units, weight_scale=1, conn_n=8):
        out = (np.identity(n_units)*weight_scale/self.n_neurs,)*conn_n
        return out

    def make_input_connectivity(self, n_units, weight_scale=1):
        return self.make_ring_connectivity(n_units,
                                           weight_scale=n_units*weight_scale,
                                           conn_n=4)

    def initialize_network(self, init_scale=.01):
        self._curr_time = 0
        for r in self.rings.values():
            r.initialize_network(init_scale=init_scale)
        
    def iterate_step(self, drive, dt, dynamics_type='noiseless',
                     dynamics_f=True):
        self._curr_time += dt
        for rk, r in self.rings.items():
            if dynamics_type == 'noiseless':
                dynamics = r.dynamics_noiseless_f
            else:
                dynamics = r.dynamics_poisson_f
            inp_r = 0            
            for i, (w, r_inp) in enumerate(self.ring_weights.get(rk, [])):
                wr_inp = np.dot(w, r_inp.get_output(dynamics_f=dynamics_f))
                inp_r += wr_inp
            for w in self.input_weights.get(rk, []):
                inp_r += np.dot(w, drive.get(rk, np.zeros(w.shape[1])))
            r.iterate_step(inp_r, dt, dynamics)

    def state(self, dynamics_f=True):
        return {k:r.get_output(dynamics_f=dynamics_f)
                for k, r in self.rings.items()}

    def foci(self):
        return {k:r.focus() for k, r in self.rings.items()}
            
    def integrate_until(self, t_end, dt, keep=10, dynamics_type='noiseless',
                        drive_dict=None, ):
        
        if drive_dict is None:
            drive_func = lambda t, curr, dt: 0
            drive_dict = {k:drive_func for k in self.rings.keys()}
        j = 0
        out = ring.get_steps_and_containers(self._curr_time, t_end, dt, keep,
                                            self.n_neurs,
                                            dict_keys=self.rings.keys())
        steps_to_end, steps_to_keep, interval = out[:3]
        trace, focus, store_drive, time = out[3:]
        for i in range(steps_to_end):
            drive_out = {k:df(self._curr_time, self.state(), dt)
                         for k, df in drive_dict.items()}
            self.iterate_step(drive_out, dt, dynamics_type)
            if (i % interval) == 0 and j < steps_to_keep:
                trace = add_dict_elements(trace, self.state(), j)
                focus = add_dict_elements(focus, self.foci(), j)
                store_drive = add_dict_elements(store_drive, drive_out, j)
                time = add_dict_elements(time, self._curr_time, j,
                                         ne_dict=False)
                j = j + 1
        return trace, focus, store_drive, time


def make_drivers(frm, c_u=None, c_l=None, col_width=.2, stim_start=200,
                 stim_dur=200, stim_mag=15, cue_start=800, cue_dur=600,
                 cue_mag=20, gen_mag=20, use_cue1=None, struct_cue=False):
    rng = np.random.default_rng()
    if c_u is None:
        c_u = rng.uniform(0, 2*np.pi)
    if c_l is None:
        c_l = rng.uniform(0, 2*np.pi)
    d_u = ring.step_drive_function_creator(frm.thetas, c_u, col_width, stim_mag, 
                                           stim_start, stim_start + stim_dur)
    d_l = ring.step_drive_function_creator(frm.thetas, c_l, col_width, stim_mag, 
                                           stim_start, stim_start + stim_dur)

    if use_cue1 is None:
        use_cue1 = bool(rng.integers(0, 1))
    if use_cue1:
        cue_mask = frm.cue_mask
    else:
        cue_mask = np.logical_not(frm.cue_mask)
    # cue_mask = np.mod(cue_mask + (rng.uniform(0, 1, len(cue_mask)) < .05), 2)

    if struct_cue:
        cue_pop_mask = cue_mask
    else:
        cue_pop_mask = None
    cue = ring.step_drive_function_creator(frm.thetas, 0, 2*np.pi, 
                                           gen_mag, cue_start,
                                           cue_start + cue_dur,
                                           pop_mask=cue_pop_mask)

    cue_opp = ring.step_drive_function_creator(frm.thetas, 0, 2*np.pi, 
                                               cue_mag, cue_start,
                                               cue_start + cue_dur,
                                               pop_mask=np.logical_not(cue_mask))

    drive_dict = {'r_u':d_u - cue_opp, 'r_l':d_l - cue_opp,
                  'r_t':cue, 'r_d':cue}
    return drive_dict
    
def simulate_trials(frm, c_u_set=None, c_l_set=None, n_trls=200,
                    dynamics_type='poisson', total_time=1500, dt=1,
                    **driver_kwargs):
    summary_full = np.zeros((n_trls, 2, 2))
    for i in range(n_trls):
        rng = np.random.default_rng()
        if c_u_set is None:
            c_u = rng.uniform(0, 2*np.pi)
        else:
            c_u = c_u_set
        if c_l_set is None:
            c_l = rng.uniform(0, 2*np.pi)
        else:
            c_l = c_l_set

        drive_dict = make_drivers(frm, c_u, c_l, **driver_kwargs)
        
        frm.initialize_network()
        out = frm.integrate_until(total_time, dt, drive_dict=drive_dict,
                                  dynamics_type=dynamics_type)
        activity, focus, drive, time = out
        n_ts = activity['r_u'].shape[0]
        if i == 0:
            activity_full = np.zeros((n_trls, 2, 2, n_ts, frm.n_neurs))
        
        activity_full[i, 0, 0, :, :] = activity['r_u']
        activity_full[i, 0, 1, :, :] = activity['r_l']
        activity_full[i, 1, 0, :, :] = activity['r_t']
        activity_full[i, 1, 1, :, :] = activity['r_d']
        summary = summarize_trial(focus, c_u, c_l)
        summary_full[i] = summary
    return summary_full, activity_full

def plot_resp_hists(summary, axs=None, n_bins=11, **kwargs):
    if axs is None:
        f, axs = plt.subplots(*summary.shape[1:], sharex=True, sharey=True)
    bins = np.linspace(-np.pi, np.pi, n_bins)
    for (i, j) in u.make_array_ind_iterator(summary.shape[1:]):
        axs[i, j].hist(summary[:, i, j], bins=bins, density=True, **kwargs)
    return axs

def summarize_trial(focus, c_u, c_l, pop_keys=('r_t', 'r_d'), t_ind=-1):
    n_pops = len(pop_keys)
    out = np.zeros((n_pops, n_pops))
    cols = (c_u, c_l)
    for (i, j) in u.make_array_ind_iterator(out.shape):
        pk = pop_keys[i]
        foci = focus[pk][t_ind]
        out[i, j] = u.normalize_periodic_range(foci - cols[j])
    return out
    
def add_dict_elements(struct, new_elements, ind, ne_dict=True):
    for key, arr in struct.items():
        if ne_dict:
            ne = new_elements.get(key, np.nan)
        else:
            ne = new_elements
        arr[ind] = ne
    return struct

def plot_activity(activity, fwid=3, axs=None, use_all_max=True,
                  set_min=0):
    if axs is None:
        f, axs = plt.subplots(2, 2, figsize=(fwid*2, fwid*2))
    axs_flat = np.reshape(axs, -1)
    if use_all_max:
        all_max = np.max(list(np.max(v) for v in activity.values()))
    else:
        all_max = None
    for i, (k, act) in enumerate(activity.items()):
        axs_flat[i].imshow(act, vmin=set_min, vmax=all_max)
        axs_flat[i].set_title(k)

def plot_max_trace(time, activity, fwid=3, axs=None, vlines=None):
    if axs is None:
        f, axs = plt.subplots(2, 2, figsize=(fwid*2, fwid*2),
                              sharex=True, sharey=False)
    axs_flat = np.reshape(axs, -1)
    for i, (k, act) in enumerate(activity.items()):        
        axs_flat[i].plot(time[k], np.max(activity[k], axis=1), label=k)
        axs_flat[i].set_title(k)
        if vlines is not None:
            list(gpl.add_vlines(vl, axs_flat[i]) for vl in vlines)
