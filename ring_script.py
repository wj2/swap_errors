import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
from datetime import datetime

import general.utility as u
import swap_errors.circus as swc

def create_parser():
    parser = argparse.ArgumentParser(description='fit several modularizers')
    parser.add_argument('-o', '--output_file',
                        default='swap_errors/model_sims/circus_{}.pkl',
                        type=str,
                        help='folder to save the output in')
    parser.add_argument('--ring_dim', default=200, type=int,
                        help='units in each ring')
    parser.add_argument('--wf_params', default=(-5, 7, 0), type=float,
                        nargs=3)
    parser.add_argument('--integ_dt', default=1, type=float)
    parser.add_argument('--total_time', default=1800, type=float)
    parser.add_argument('--gen_mag_delta', default=0, type=float)
    parser.add_argument('--stim_mag', default=10, type=float)
    parser.add_argument('--bias', default=10, type=float)
    parser.add_argument('--cue_mag', default=5, type=float)
    parser.add_argument('--alpha_beta_ratio', default=1, type=float)
    parser.add_argument('--beta', default=10, type=float)
    parser.add_argument('--n_trls', default=500, type=int)
    parser.add_argument('--dynamics_type', default='poisson', type=str)
    parser.add_argument('--config_path', default=None, type=str)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.config_path is not None:
        config_dict = pickle.load(open(args.config_path, 'rb'))
        args = u.merge_params_dict(args, config_dict)

    args.date = datetime.now()

    wf_params = args.wf_params
    n_units = args.ring_dim
    dt = args.integ_dt
    total_time = args.total_time
    gen_mag_delt = args.gen_mag_delta
    stim_mag = args.stim_mag
    bias = args.bias
    cue_mag = args.cue_mag
    alpha_beta_ratio = args.alpha_beta_ratio
    beta = args.beta
    alpha = alpha_beta_ratio*beta
    n_trls = args.n_trls
    dynamics_type = args.dynamics_type

    frm = swc.FourRingModel(n_units, r_inhib=beta, r_excit=alpha, bias=bias,
                            wf_params=wf_params)
    bump_stats = frm.compute_bump_statistics_empirical()
    cue_stats = frm.compute_pc_empirical(cue_mag)
    gen_mag = 2*cue_stats[-1]*beta/n_units - bias + gen_mag_delt

    targ_theta_u = None
    targ_theta_l = None
    out_cue1, act_c1 = swc.simulate_trials(frm, targ_theta_u, targ_theta_l,
                                           n_trls=n_trls,
                                           dynamics_type=dynamics_type,
                                           total_time=total_time,
                                           use_cue1=True, cue_mag=cue_mag,
                                           gen_mag=gen_mag,
                                           stim_mag=stim_mag)
    out_cue2, act_c2 = swc.simulate_trials(frm, targ_theta_u, targ_theta_l,
                                           n_trls=n_trls,
                                           dynamics_type=dynamics_type,
                                           gen_mag=gen_mag, 
                                           total_time=total_time,
                                           use_cue1=False, cue_mag=cue_mag,
                                           stim_mag=stim_mag)
    
    out_file = args.output_file.format(args.date).replace(' ', '_')
    out_dict = vars(args)
    out_dict['cue1'] = (out_cue1, act_c1)
    out_dict['cue2'] = (out_cue2, act_c2)
    pickle.dump(out_dict, open(out_file, 'wb'))    
