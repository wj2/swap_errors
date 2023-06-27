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
    parser = argparse.ArgumentParser(description="fit several autoencoders")
    parser.add_argument(
        "-o",
        "--output_folder",
        default="../results/swap_errors/fits/",
        type=str,
        help="folder to save the output in",
    )
    default_outname = "fit_spline{num_colors}_sess{sess_ind}_{period}_" "{jobid}.pkl"
    parser.add_argument("--output_name", default=default_outname)
    parser.add_argument(
        "--model_path",
        default="swap_errors/ushh_dh_t_inter_model.pkl",
        type=str,
        help="stan model to use",
    )
    default_data = (
        "/burg/theory/users/ma3811/assignment_errors/"
        "{num_colors}_colors/sess_{sess_ind}/{period}_diode/"
        "{time}/"
        "pca_0.95_before/impute_{use_impute}"
        "/spline{spline_order}_knots/{use_regions}/"
        "{trl_type}/stan_data.pkl"
    )
    parser.add_argument(
        "--use_trl_types", default=("retro", "pro"), type=str, nargs="+"
    )
    parser.add_argument("--use_regions", default="all")
    parser.add_argument("--no_imputation", default=False, action="store_true")
    parser.add_argument("--spline_order", default=1, type=int)
    parser.add_argument("--use_joint_data", default=False, action="store_true")
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--data_path", default=default_data, type=str)
    parser.add_argument("--num_colors", default=5, type=int)
    parser.add_argument("--period", default="WHEEL_ON", type=str)
    parser.add_argument("--sess_ind", default=0, type=int)
    parser.add_argument("--n_iter", default=500, type=int)
    parser.add_argument("--n_chains", default=4, type=int)
    parser.add_argument("--fit_guesses", default=False, action="store_true")
    parser.add_argument("--fit_delay1", default=False, action="store_true")
    parser.add_argument("--fit_samples", default=False, action="store_true")
    parser.add_argument("--use_manual", default=False, action="store_true")
    parser.add_argument("--use_time", default="-0.5-0.0-0.5_0.5", type=str)

    parser.add_argument("--prior_alpha", default=1, type=float)
    parser.add_argument("--prior_std", default=10, type=float)
    parser.add_argument("--prior_gamma_alpha", default=1, type=float)
    parser.add_argument("--prior_gamma_beta", default=.5, type=float)
    parser.add_argument("--prior_dirichlet", default=None, type=float, nargs='+')

    return parser


default_add_keys = ("y", "C_u", "C_l", "cue", "p", "C_resp")
default_extra_keys = ("up_col_rads", "down_col_rads", "resp_rads")


def add_key(key, fd, kd):
    if fd.get(key) is None:
        out = kd.get(key)
    else:
        out = np.concatenate((fd[key], kd.get(key)))
    return out


def merge_data(
    data_d,
    noerr_types=("single",),
    add_keys=default_add_keys,
    extra_add_keys=default_extra_keys,
):
    all_keys = set(data_d.keys())
    noerr_types = set(noerr_types)
    first_keys = list(all_keys.difference(noerr_types))
    last_keys = list(noerr_types.intersection(all_keys))
    ordered_keys = first_keys + last_keys
    full_dict = {"is_joint": 1}
    extra_dict = {}
    key_ind = 1
    for key in ordered_keys:
        dk = data_d[key]
        full_dict["T"] = full_dict.get("T", 0) + dk["T"]
        full_dict["N"] = dk["N"]
        full_dict["K"] = dk["K"]
        full_dict["type"] = np.concatenate(
            (full_dict.get("type", []), np.ones(dk["T"], dtype=int) * key_ind)
        )
        extra_dict["type_str"] = extra_dict.get("type_str", ()) + (key,) * dk["T"]
        model_error = (np.ones(dk["T"]) * (key not in noerr_types)).astype(int)
        full_dict["model_error"] = (
            full_dict.get("model_error", ()) + (model_error,) * dk["T"]
        )
        for ak in add_keys:
            full_dict[ak] = add_key(ak, full_dict, dk)
        for eak in extra_add_keys:
            extra_dict[eak] = add_key(eak, extra_dict, dk)
        key_ind = key_ind + 1
    full_dict["model_error"] = np.concatenate(full_dict["model_error"])
    full_dict["type"] = full_dict["type"].astype(int)
    print(full_dict["type"], key)
    print(len(full_dict["y"]), len(extra_dict["up_col_rads"]))
    return full_dict, extra_dict


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    data_path_templ = args.data_path
    if args.use_manual:
        data_path_templ = (
            "/burg/theory/users/ma3811/assignment_errors/manual/"
            "{num_colors}_colors/sess_{sess_ind}/{period}_diode/"
            "{time}/"
            "pca_0.95_before/impute_{use_impute}"
            "/spline{spline_order}_knots/all/"
            "{trl_type}/stan_data.pkl"
        )

    use_impute = not args.no_imputation
    if args.fit_delay1:
        args.period = "CUE2_ON"
        args.use_trl_types = ("retro",)
        args.model_path = "swap_errors/ushh_d1_t_model.pkl"
    elif args.fit_samples:
        args.period = "SAMPLES_ON"
        args.use_trl_types = ("retro",)
        args.model_path = "swap_errors/ushh_d1_t_model.pkl"
        args.use_time = "0-0.5-0.5_0.5"
        args.fit_delay1 = True
    # start, end, step = args.use_time
    time = args.use_time
    # "{:.1f}_{:.1f}_{:.1f}".format(start, end, step)
    if args.use_joint_data:
        data_path = data_path_templ.format(
            num_colors=args.num_colors,
            sess_ind=args.sess_ind,
            period=args.period,
            trl_type="joint",
            use_impute=use_impute,
            spline_order=args.spline_order,
            use_regions=args.use_regions,
            time=time,
        )
        data = pickle.load(open(data_path, "rb"))
        data["C_resp"] = data["resp_spl"]
        extra_data = {}
    else:
        data_unmerged = {}
        for i, utt in enumerate(args.use_trl_types):
            data_path = data_path_templ.format(
                num_colors=args.num_colors,
                sess_ind=args.sess_ind,
                period=args.period,
                trl_type=utt,
                use_impute=use_impute,
                spline_order=args.spline_order,
                use_regions=args.use_regions,
                time=time,
            )
            data_i = pickle.load(open(data_path, "rb"))
            data_i["C_resp"] = data_i["resp_spl"]
            data_unmerged[utt] = data_i
        data, extra_data = merge_data(data_unmerged)
    model = pickle.load(open(args.model_path, "rb"))

    n_iter = args.n_iter
    n_chains = args.n_chains

    model_dict = {
        # fit_guesses, fit_delay1, use_single
        (True, False, False): "swap_errors/ushh_dh_guess_t_model.pkl",
        (False, False, False): "swap_errors/ushh_dh_t_inter_model.pkl",
        (True, True, False): "swap_errors/ushh_d1_guess_t_model.pkl",
        (False, True, False): "swap_errors/ushh_d1_t_model.pkl",
        (True, False, True): "swap_errors/ushh_sdh_guess_t_model.pkl",
        (False, False, True): "swap_errors/ushh_sdh_t_inter_model.pkl",
    }

    model_path = model_dict[
        args.fit_guesses, args.fit_delay1, "single" in args.use_trl_types
    ]
    print(model_path)
    data["prior_alpha"] = args.prior_alpha
    data['swap_prior'] = (.5, 1)
    data['guess_prior'] = (.5, 1)
    data["prior_std"] = args.prior_std
    data["prior_g_alpha"] = args.prior_gamma_alpha
    data["prior_g_beta"] = args.prior_gamma_beta

    fit, fit_az, diag = su.fit_model(data, model_path, iter=n_iter, chains=n_chains)
    out_name = args.output_name.format(
        num_colors=args.num_colors,
        sess_ind=args.sess_ind,
        period=args.period,
        jobid=args.jobid,
    )
    out_path = os.path.join(args.output_folder, out_name)
    out_root, _ = os.path.splitext(out_path)
    out_az_path = out_root + "_az.nc"
    now = datetime.datetime.now()
    data.update(extra_data)
    out_struct = {
        "model_fit_path": out_az_path,
        "diags": diag,
        "fit_time": now,
        "data": data,
        "model_path": model_path,
        "args": args,
    }
    pickle.dump(out_struct, open(out_path, "wb"))
    fit_az.to_netcdf(out_az_path)
