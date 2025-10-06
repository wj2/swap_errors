import numpy as np
import argparse

import general.data_io as gio
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan


def create_parser():
    parser = argparse.ArgumentParser(
        description="make reduced dataset from Buschman data"
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default="../data/swap_errors/lm_data",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument(
        "--max_files", default=np.inf, type=float, help="number of files to load"
    )
    parser.add_argument(
        "--data_folder", default="../data/swap_errors/", help="folder with datasets"
    )
    parser.add_argument("--bhv_model", help="{data_folder}/bhv_model-pr.pkl")
    parser.add_argument("--motoaki", action="store_true", default=False)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    max_files = args.max_files
    df = args.data_folder
    print(df)
    if args.motoaki:
        func = swa.load_motoaki_data
        kwargs = {}
    else:
        func = swa.load_buschman_data
        bhv_file = args.bhv_model.format(data_folder=df)
        kwargs = {"load_bhv_model": bhv_file}
    data = gio.Dataset.from_readfunc(
        func,
        df,
        max_files=max_files,
        seconds=True,
        spks_template=swa.busch_spks_templ_mua,
        **kwargs,
    )
    swan.prepare_lm_tc_pops(
        data,
        region_subsets=swan.all_region_subset,
        out_folder=args.out_folder,
        motoaki_data=args.motoaki,
    )
