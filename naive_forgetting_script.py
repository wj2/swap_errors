import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
from datetime import datetime

import general.utility as u
import swap_errors.auxiliary as swaux
import swap_errors.analysis as swan

def create_parser():
    parser = argparse.ArgumentParser(description='fit several modularizers')
    parser.add_argument('-o', '--output_file',
                        default='swap_errors/naive_forgetting/forget_{}.pkl',
                        type=str,
                        help='folder to save the output in')
    parser.add_argument('--decider', default='argmax', type=str)
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('--file_templ_d1', default=None, type=str)
    parser.add_argument('--file_templ_d2', default=None, type=str)
    parser.add_argument('--local_test', default=False, action='store_true')
    parser.add_argument('--forget_kernel', default='rbf', type=str)
    parser.add_argument('--decider_arg', default=None, type=float)
    return parser

decider_dict = {'argmax':(swan.corr_argmax, swan.swap_argmax),
                'plurality':(swan.corr_plurality, swan.swap_plurality),
                'diff':(swan.corr_diff, swan.swap_diff)}

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    if args.config_path is not None:
        config_dict = pickle.load(open(args.config_path, 'rb'))
        args = u.merge_params_dict(args, config_dict)

    corr_decider, swap_decider = decider_dict[args.decider]
    if args.decider_arg is not None:
        corr_decider = lambda x: corr_decider(x, args.decider_arg)
        swap_decider = lambda x: swap_decider(x, args.decider_arg)
    args.date = datetime.now()
    if not args.local_test and args.file_templ_d1 is None:
        file_templ_d1 = swaux.cluster_naive_d1_path_templ
        form_opts_d1 = swaux.cluster_naive_d1_format_options
    elif args.local_test:
        file_templ_d1 = 'swap_errors/test_sessions/retro_{}/stan_data.pkl'
        form_opts_d1 = {'test_type':(13,)}
    else:
        file_templ_d1, form_opts_d1 = pickle.load(open(args.file_templ_d1, 'rb'))

    sessions_d1 = swaux.load_files_ma_folders(file_templ_d1, **form_opts_d1)
    out_forget_cu = {}
    out_forget_cl = {}
    for k, d_dict in sessions_d1.items():
        out_f_cu = swan.naive_forgetting(d_dict, flip_cue=False,
                                         swap_decider=swap_decider,
                                         corr_decider=corr_decider,
                                         kernel=args.forget_kernel)
        out_forget_cu[k] = out_f_cu
        out_f_cl = swan.naive_forgetting(d_dict, flip_cue=True,
                                         swap_decider=swap_decider,
                                         corr_decider=corr_decider,
                                         kernel=args.forget_kernel)
        out_forget_cl[k] = out_f_cl
        
    out_dict = {'args':args, 'forget_cu':out_forget_cu,
                'forget_cl':out_forget_cl}
    fname = args.output_file.format(args.date)
    fname = fname.replace(' ', '_')
    pickle.dump(out_dict, open(fname, 'wb'))
