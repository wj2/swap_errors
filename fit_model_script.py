
import argparse
import os
import datetime
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft

import general.utility as u
import general.stan_utility as su

def create_parser():
    parser = argparse.ArgumentParser(description='fit several autoencoders')
    parser.add_argument('-o', '--output_folder',
                        default='../results/swap_errors/fits/', type=str,
                        help='folder to save the output in')
    default_outname = ('fit_spline{num_colors}_sess{sess_ind}_{period}_'
                       '{jobid}.pkl')
    parser.add_argument('--output_name', default=default_outname)
    parser.add_argument('--model_path',
                        default='swap_errors/ushh_dh_t_inter_model.pkl',
                        type=str,
                        help='stan model to use')
    default_data = ('/burg/theory/users/ma3811/assignment_errors/'
                    '{num_colors}_colors/sess_{sess_ind}/{period}_diode/'
                    '-0.5-0.0-0.5_0.5/'
                    'pca_0.95_before/impute_True/spline1_knots/all/'
                    '{trl_type}/stan_data.pkl')
    parser.add_argument('--use_trl_types', default=('pro', 'retro'),
                        type=str, nargs='+')
    parser.add_argument('--use_joint_data', default=False, action='store_true')
    parser.add_argument('--jobid', default='0000', type=str)    
    parser.add_argument('--data_path', default=default_data, type=str)
    parser.add_argument('--num_colors', default=5, type=int)
    parser.add_argument('--period', default='WHEEL_ON', type=str)
    parser.add_argument('--sess_ind', default=0, type=int)
    parser.add_argument('--n_iter', default=500, type=int)
    parser.add_argument('--n_chains', default=4, type=int)
    return parser

default_add_keys = ('y', 'C_u', 'C_l', 'cue', 'p', 'up_col_rads',
                    'down_col_rads')
def add_key(key, fd, kd):
    if fd.get(key) is None:
        out = kd[key]
    else:
        out = np.concatenate((fd[key], kd[key]))
    return out

def merge_data(data_d, noerr_types=('single',), add_keys=default_add_keys):
    all_keys = set(data_d.keys())
    noerr_types = set(noerr_types)
    first_keys = list(all_keys.difference(noerr_types))
    last_keys = list(noerr_types.intersection(all_keys))
    ordered_keys = first_keys + last_keys
    full_dict = {'is_joint':1}
    extra_dict = {}
    key_ind = 1
    for key in ordered_keys:
        dk = data_d[key]
        full_dict['T'] = full_dict.get('T', 0) + dk['T']
        full_dict['N'] = dk['N']
        full_dict['K'] = dk['K']
        full_dict['type'] = np.concatenate((full_dict.get('type', []),
                                            np.ones(dk['T'], dtype=int)*key_ind))
        extra_dict['type_str'] = extra_dict.get('type_str', ()) + (key,)*dk['T']
        model_error = (np.ones(dk['T'])*(key not in noerr_types)).astype(int)
        full_dict['model_error'] = (full_dict.get('model_error', ())
                                    + (model_error,)*dk['T'])
        for ak in default_add_keys:
            full_dict[ak] = add_key(ak, full_dict, dk)
        key_ind = key_ind + 1
    full_dict['type'] = full_dict['type'].astype(int)
    return full_dict, extra_dict

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    data_path = args.data_path
    if args.use_joint_data:
        data_path = data_path.format(num_colors=args.num_colors,
                                     sess_ind=args.sess_ind,
                                     period=args.period,
                                     trl_type='joint')
        data = pickle.load(open(data_path, 'rb'))
        extra_data = {}
    else:
        data_unmerged = {}
        for i, utt in enumerate(args.use_trl_types):
            data_path = data_path.format(num_colors=args.num_colors,
                                         sess_ind=args.sess_ind,
                                         period=args.period,
                                         trl_type=utt)
            data_i = pickle.load(open(data_path, 'rb'))
            data_unmerged[utt] = data_i
        data, extra_data = merge_data(data_unmerged)
    model = pickle.load(open(args.model_path, 'rb'))

    n_iter = args.n_iter
    n_chains = args.n_chains

    fit, fit_az, diag = su.fit_model(data, args.model_path, iter=n_iter, 
                                     chains=n_chains)
    out_name = args.output_name.format(num_colors=args.num_colors,
                                       sess_ind=args.sess_ind,
                                       period=args.period,
                                       jobid=args.jobid)
    out_path = os.path.join(args.output_folder,
                            out_name)
    now = datetime.datetime.now()
    data.update(extra_data)
    out_struct = {
        'model_fit':fit_az,
        'diags':diag,
        'fit_time':now,
        'data':data,
    }
    
    pickle.dump(out_struct, open(out_path, 'wb'))

    
