import argparse
import numpy as np
import pickle
from datetime import datetime

import general.utility as u
import swap_errors.auxiliary as swaux
import swap_errors.analysis as swan


def create_parser():
    parser = argparse.ArgumentParser(description="fit several modularizers")
    parser.add_argument(
        "-o",
        "--output_file",
        default="swap_errors/naive_centroids/cents_{}.pkl",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument("--decider", default="argmax", type=str)
    parser.add_argument("--avg_dist", default=np.pi / 4, type=float)
    parser.add_argument("--config_path", default=None, type=str)
    parser.add_argument("--file_templ_d1", default=None, type=str)
    parser.add_argument("--file_templ_d2", default=None, type=str)
    parser.add_argument("--local_test", default=False, action="store_true")
    parser.add_argument("--decider_arg", default=None, type=float)
    parser.add_argument("--shuffle_swaps", default=False, action="store_true")
    parser.add_argument("--shuffle_nulls", default=False, action="store_true")
    parser.add_argument("--use_guesses", default=False, action="store_true")
    parser.add_argument("--no_regions", default=False, action="store_true")
    parser.add_argument("--use_manual", default=False, action="store_true")
    parser.add_argument("--no_imputation", default=False, action="store_true")
    return parser


decider_dict = {
    "argmax": (swan.corr_argmax, swan.swap_argmax),
    "plurality": (swan.corr_plurality, swan.swap_plurality),
    "diff": (swan.corr_diff, swan.swap_diff),
    "argmax_guess": (swan.corr_argmax, swan.guess_argmax),
    "plurality_guess": (swan.corr_plurality, swan.guess_plurality),
    "diff_guess": (swan.corr_diff, swan.guess_diff),
}

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if args.config_path is not None:
        config_dict = pickle.load(open(args.config_path, "rb"))
        args = u.merge_params_dict(args, config_dict)

    if args.use_guesses:
        centroid_func = swan.naive_guessing
        decider_str = args.decider + "_guess"
    else:
        centroid_func = swan.naive_centroids
        decider_str = args.decider

    corr_decider_pl, swap_decider_pl = decider_dict[decider_str]
    if args.decider_arg is not None:

        def corr_decider(x):
            return corr_decider_pl(x, args.decider_arg)

        def swap_decider(x):
            return swap_decider_pl(x, args.decider_arg)

    else:
        corr_decider = corr_decider_pl
        swap_decider = swap_decider_pl
    args.date = datetime.now()
    if not args.local_test and args.file_templ_d1 is None:
        if args.use_manual:
            file_templ_d1 = swaux.cluster_naive_d1_manual_path_templ
        else:
            file_templ_d1 = swaux.cluster_naive_d1_path_templ
        if args.no_regions:
            form_opts_d1 = swaux.cluster_naive_d1_format_options_noregions
        else:
            form_opts_d1 = swaux.cluster_naive_d1_format_options
    elif args.local_test:
        file_templ_d1 = "swap_errors/test_sessions/retro_{}/stan_data.pkl"
        form_opts_d1 = {"test_type": (14,)}
    else:
        file_templ_d1, form_opts_d1 = pickle.load(open(args.file_templ_d1, "rb"))

    if args.no_imputation:
        file_templ_d1 = file_templ_d1.replace("impute_True", "impute_False")

    sessions_d1 = swaux.load_files_ma_folders(file_templ_d1, **form_opts_d1)
    out_d1_cu = {}
    out_d1_cl = {}
    for k, d_dict in sessions_d1.items():
        out_cu = centroid_func(
            d_dict,
            use_cue=False,
            swap_decider=swap_decider,
            corr_decider=corr_decider,
            col_thr=args.avg_dist,
            shuffle_nulls=args.shuffle_nulls,
            shuffle_swaps=args.shuffle_swaps,
        )
        out_d1_cu[k] = out_cu
        out_cl = centroid_func(
            d_dict,
            use_cue=False,
            flip_cue=True,
            swap_decider=swap_decider,
            corr_decider=corr_decider,
            col_thr=args.avg_dist,
            shuffle_nulls=args.shuffle_nulls,
            shuffle_swaps=args.shuffle_swaps,
        )
        out_d1_cl[k] = out_cl

    if not args.local_test and args.file_templ_d2 is None:
        if args.use_manual:
            file_templ_d2 = swaux.cluster_naive_d2_manual_path_templ
        else:
            file_templ_d2 = swaux.cluster_naive_d2_path_templ
        if args.no_regions:
            form_opts_d2 = swaux.cluster_naive_d2_format_options_noregions
        else:
            form_opts_d2 = swaux.cluster_naive_d2_format_options

    elif args.local_test:
        file_templ_d2 = "swap_errors/test_sessions/retro_{}/stan_data.pkl"
        form_opts_d2 = {"test_type": (9,)}
    else:
        file_templ_d2, form_opts_d2 = pickle.load(open(args.file_templ_d2, "rb"))

    if args.no_imputation:
        file_templ_d2 = file_templ_d2.replace("impute_True", "impute_False")

    sessions_d2 = swaux.load_files_ma_folders(file_templ_d2, **form_opts_d2)
    out_d2 = {}
    for k, d_dict in sessions_d2.items():
        out = centroid_func(
            d_dict,
            swap_decider=swap_decider,
            corr_decider=corr_decider,
            col_thr=args.avg_dist,
            shuffle_nulls=args.shuffle_nulls,
            shuffle_swaps=args.shuffle_swaps,
        )
        out_d2[k] = out

    out_dict = {"args": args, "d1_cu": out_d1_cu, "d1_cl": out_d1_cl, "d2": out_d2}
    fname = args.output_file.format(args.date)
    fname = fname.replace(" ", "_")
    pickle.dump(out_dict, open(fname, "wb"))
