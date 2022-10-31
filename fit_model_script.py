
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
                        default='~/results/swap_errors/fits/', type=str,
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
                    'joint/stan_data.pkl')
    parser.add_argument('--jobid', default='-1', type=str)    
    parser.add_argument('--data_path', default=default_data, type=str)
    parser.add_argument('--num_colors', default=5, type=int)
    parser.add_argument('--period', default='WHEEL_ON', type=str)
    parser.add_argument('--sess_ind', default=0, type=int)
    parser.add_argument('--n_iter', default=500, type=int)
    parser.add_argument('--n_chains', default=4, type=int)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    data_path = args.data_path
    data_path = data_path.format(num_colors=args.num_colors,
                                 sess_ind=args.sess_ind,
                                 period=args.period)
    data = pickle.load(open(data_path, 'rb'))

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
    pickle.dump((fit_az, diag, now), open(out_path, 'wb'))

    
