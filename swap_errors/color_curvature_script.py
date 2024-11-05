import pickle
import os
import argparse

import swap_errors.analysis as swan


def create_parser():
    parser = argparse.ArgumentParser(description="fit several autoencoders")
    parser.add_argument(
        "data_file",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default="../results/swap_errors/color_curvature/",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument("--jobid", default="0000", type=str)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    core_str = os.path.splitext(os.path.split(args.data_file)[1])[0]
    file_name = "fit_{input}_{jobid}.pkl".format(input=core_str, jobid=args.jobid)
    output_path = os.path.join(args.output_folder, file_name)

    input_data = pickle.load(open(args.data_file, "rb"))

    results_dict = {}
    for m, m_dict in input_data.items():
        results_dict[m] = {}
        for t, t_m_data in m_dict.items():
            out_tcc = swan.fit_tccish_model(*t_m_data)
            results_dict[m][t] = out_tcc, t_m_data[2]
    pickle.dump(results_dict, open(output_path, "wb"))
