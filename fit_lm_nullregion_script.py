import argparse
import functools as ft

import swap_errors.analysis as swan


def create_parser():
    parser = argparse.ArgumentParser(description="fit linear models")
    parser.add_argument(
        "data_file",
        type=str,
    )
    parser.add_argument("--region_str", default="no_pfc", type=str)
    parser.add_argument(
        "-o",
        "--output_folder",
        default="../results/swap_errors/lms/",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument("--spline_order", default=1, type=int)
    parser.add_argument("--num_knots", default=5, type=int)
    parser.add_argument("--use_threshold", default=None, type=float)
    parser.add_argument("--pre_pca", default=None, type=float)
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--single_color", default=False, action='store_true')
    parser.add_argument("--n_reps", default=10, type=int)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    kwargs = {}
    if args.use_threshold is not None or args.use_threshold > 0:
        swap_decider = ft.partial(swan.swap_plurality, prob=args.use_threshold)
        kwargs["swap_decider"] = swap_decider
        corr_decider = ft.partial(swan.corr_plurality, prob=args.use_threshold)
        kwargs["corr_decider"] = corr_decider

    use_region = "_{}".format(args.region_str)
    region_file = args.data_file.format(region=use_region)
    
    use_null = "_all"
    null_file = args.data_file.format(region=use_null)
        
    swan.swap_lm_tc_null_frompickle(
        region_file,
        null_file,
        out_folder=args.output_folder,
        spline_order=args.spline_order,
        n_knots=args.num_knots,
        pre_pca=args.pre_pca,
        jobid=args.jobid,
        single_color=args.single_color,
        n_reps=args.n_reps,
        **kwargs,
    )
