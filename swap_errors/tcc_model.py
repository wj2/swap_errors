import numpy as np
import pyro
import torch
import sklearn.preprocessing as skp

import general.pyro_utility as gpu
import pyro.distributions as distribs
from pyro.ops.indexing import Vindex
import general.utility as u


def combine_pickles(data, targ_key="c_targ", dist_key="c_dist", resp_key="rc"):
    targs = np.concatenate(list(d_k[targ_key] for d_k in data.values()))
    dists = np.concatenate(list(d_k[dist_key] for d_k in data.values()))
    resps = np.concatenate(list(d_k[resp_key] for d_k in data.values()))
    return targs, dists, resps


@pyro.infer.config_enumerate
def constrained_multi_func_swap_model(
    targ_funcs,
    dist_funcs,
    targ_funcs_spec,
    dist_funcs_spec,
    resp_inds=None,
    hn_width=10,
    alpha=0.5,
):
    snr = pyro.sample(
        "snr",
        distribs.HalfNormal(hn_width),
    )
    swap_rate = pyro.sample(
        "swap_rate", distribs.Dirichlet(torch.tensor([alpha, alpha]))
    )
    spec_fraction = pyro.sample(
        "blend_fraction",
        distribs.Dirichlet(torch.tensor([alpha, alpha])),
    )
    funcs_avg = torch.stack((targ_funcs, dist_funcs), dim=1)
    funcs_spec = torch.stack((targ_funcs_spec, dist_funcs_spec), dim=1)
    funcs = funcs_avg * spec_fraction[0] + funcs_spec * spec_fraction[1]

    funcs_scale = snr * funcs
    denom = torch.sum(torch.exp(funcs_scale), dim=2, keepdims=True)
    probs = torch.exp(funcs_scale) / denom

    with pyro.plate("data", len(probs)) as ind:
        swap = pyro.sample("swap", distribs.Categorical(swap_rate))
        use_probs = Vindex(probs)[ind, swap]
        targ_distr = distribs.Categorical(use_probs)
        out = pyro.sample("obs", targ_distr, obs=resp_inds)
        return out


@pyro.infer.config_enumerate
def constrained_func_swap_model(
    targ_funcs, dist_funcs, resp_inds=None, hn_width=10, alpha=0.5
):
    snr = pyro.sample(
        "snr",
        distribs.HalfNormal(hn_width),
    )
    swap_rate = pyro.sample(
        "swap_rate", distribs.Dirichlet(torch.tensor([alpha, alpha]))
    )
    funcs = torch.stack((targ_funcs, dist_funcs), dim=1)

    funcs_scale = snr * funcs
    denom = torch.sum(torch.exp(funcs_scale), dim=2, keepdims=True)
    probs = torch.exp(funcs_scale) / denom

    with pyro.plate("data", len(probs)) as ind:
        swap = pyro.sample("swap", distribs.Categorical(swap_rate))
        use_probs = Vindex(probs)[ind, swap]
        targ_distr = distribs.Categorical(use_probs)
        out = pyro.sample("obs", targ_distr, obs=resp_inds)
        return out


def _rescale_kernels(kernels):
    p = skp.MinMaxScaler()
    kernels = p.fit_transform(kernels.T).T
    return kernels


def fit_func_neuron_swap_model(
    targs,
    dists,
    resps,
    gpkernel,
    n_bins=10,
    bounds=(-np.pi, np.pi),
    n_samps=500,
    rescale_kernel=False,
    use_resp_as_targ=False,
    **kwargs,
):
    nan_mask = ~np.isnan(resps)
    targs = targs[nan_mask]
    dists = dists[nan_mask]
    resps = resps[nan_mask]
    if use_resp_as_targ:
        targs = resps

    bins = np.linspace(*bounds, n_bins + 1)
    resp_inds = torch.tensor(np.digitize(resps, bins) - 1)
    bin_cents = bins[:-1] + np.diff(bins)[0] / 2

    u_cols, targ_inds = np.unique(targs, return_inverse=True)
    col_kern = np.zeros((len(u_cols), len(bin_cents)))
    for i, uc in enumerate(u_cols):
        bcs_i = u.normalize_periodic_range(bin_cents - uc)
        _, col_kern[i] = gpkernel.get_average_kernel(bcs_i, rescale=rescale_kernel)
    targ_funcs = col_kern[targ_inds]

    u_cols_d, dist_inds = np.unique(dists, return_inverse=True)
    assert np.all(u_cols_d == u_cols)
    dist_funcs = col_kern[dist_inds]        
        
    _, targ_func_spec = gpkernel.get_conditioned_kernel(
        bin_cents, u.normalize_periodic_range(targs), rescale=rescale_kernel,
    )
    targ_func_spec = np.squeeze(targ_func_spec)
    _, dist_func_spec = gpkernel.get_conditioned_kernel(
        bin_cents, u.normalize_periodic_range(dists), rescale=rescale_kernel,
    )
    dist_func_spec = np.squeeze(dist_func_spec)

    inp = (
        torch.tensor(targ_funcs, dtype=torch.float),
        torch.tensor(dist_funcs, dtype=torch.float),
        torch.tensor(targ_func_spec, dtype=torch.float),
        torch.tensor(dist_func_spec, dtype=torch.float),
    )
    targ = (resp_inds,)

    print("preliminaries")
    loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    out = gpu.fit_model(
        inp,
        targ,
        constrained_multi_func_swap_model,
        loss=loss,
        block_vars=["swap"],
        **kwargs,
    )
    preds = gpu.sample_fit_model(
        inp,
        out["model"],
        out["guide"],
        n_samps=n_samps,
    )
    out["predictive_samples"] = bin_cents[preds]
    out["targs"] = targs
    out["dists"] = dists
    out["targ_funcs"] = targ_funcs
    out["dist_funcs"] = dist_funcs
    out["targ_funcs_spec"] = targ_func_spec
    out["dist_funcs_spec"] = dist_func_spec

    out["bin_cents"] = bin_cents
    out["resps"] = resps
    return out



def fit_func_swap_model(
    targs,
    dists,
    resps,
    model,
    spec_model=None,
    n_bins=10,
    bounds=(-np.pi, np.pi),
    n_samps=500,
    circularize_funcs=True,
    **kwargs,
):
    nan_mask = ~np.isnan(resps)
    targs = targs[nan_mask]
    dists = dists[nan_mask]
    resps = resps[nan_mask]

    bins = np.linspace(*bounds, n_bins + 1)
    resp_inds = torch.tensor(np.digitize(resps, bins) - 1)
    bin_cents = bins[:-1] + np.diff(bins)[0] / 2

    bcs = np.expand_dims(bin_cents, 0)
    targ_func_diffs = u.normalize_periodic_range(np.expand_dims(targs, 1) - bcs)
    targ_fds_flat = targ_func_diffs.flatten()
    dist_func_diffs = u.normalize_periodic_range(np.expand_dims(dists, 1) - bcs)
    dist_fds_flat = dist_func_diffs.flatten()

    if circularize_funcs:
        targ_fds_flat = np.stack((np.sin(targ_fds_flat), np.cos(targ_fds_flat)), axis=1)
        dist_fds_flat = np.stack((np.sin(dist_fds_flat), np.cos(dist_fds_flat)), axis=1)

    targ_funcs, _ = model(torch.tensor(targ_fds_flat), noiseless=True)
    targ_funcs = targ_funcs.reshape((targ_func_diffs.shape)).type(torch.float)
    dist_funcs, _ = model(torch.tensor(dist_fds_flat), noiseless=True)
    dist_funcs = dist_funcs.reshape((dist_func_diffs.shape)).type(torch.float)

    norm = torch.max(targ_funcs)
    targ_funcs = -targ_funcs / norm
    dist_funcs = -dist_funcs / norm

    inp = (targ_funcs, dist_funcs)
    targ = (resp_inds,)

    if spec_model is not None:
        pyro_model = constrained_multi_func_swap_model

        targs_arr, bcs_arr = np.meshgrid(bcs, targs)
        targ_spec_arr = np.stack((targs_arr, bcs_arr), axis=0)
        targ_spec_arr_flat = np.reshape(targ_spec_arr, (2, -1)).T

        dist_arr, bcs_arr = np.meshgrid(bcs, targs)
        dist_spec_arr = np.stack((dist_arr, bcs_arr), axis=0)
        dist_spec_arr_flat = np.reshape(dist_spec_arr, (2, -1)).T
        if circularize_funcs:
            targ_spec_arr_flat = np.stack(
                (
                    np.sin(targ_spec_arr_flat[:, 0]),
                    np.cos(targ_spec_arr_flat[:, 0]),
                    np.sin(targ_spec_arr_flat[:, 1]),
                    np.cos(targ_spec_arr_flat[:, 1]),
                ),
                axis=1,
            )
            dist_spec_arr_flat = np.stack(
                (
                    np.sin(dist_spec_arr_flat[:, 0]),
                    np.cos(dist_spec_arr_flat[:, 0]),
                    np.sin(dist_spec_arr_flat[:, 1]),
                    np.cos(dist_spec_arr_flat[:, 1]),
                ),
                axis=1,
            )
        tf_spec, _ = spec_model(torch.tensor(targ_spec_arr_flat), noiseless=True)
        tf_spec = tf_spec.reshape((targ_spec_arr.shape[1:])).type(torch.float)

        df_spec, _ = spec_model(torch.tensor(dist_spec_arr_flat), noiseless=True)
        df_spec = df_spec.reshape((dist_spec_arr.shape[1:])).type(torch.float)

        norm = torch.max(tf_spec)
        tf_spec = -tf_spec / norm
        df_spec = -df_spec / norm

        inp = inp + (tf_spec, df_spec)
    else:
        pyro_model = constrained_func_swap_model

    loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    out = gpu.fit_model(
        inp,
        targ,
        pyro_model,
        loss=loss,
        block_vars=["swap"],
        **kwargs,
    )
    preds = gpu.sample_fit_model(
        inp,
        out["model"],
        out["guide"],
        n_samps=n_samps,
    )
    out["predictive_samples"] = bin_cents[preds]
    out["targs"] = targs
    out["dists"] = dists
    out["targ_funcs"] = targ_funcs.detach().numpy()
    out["dist_funcs"] = dist_funcs.detach().numpy()
    if spec_model is not None:
        out["targ_funcs_spec"] = tf_spec.detach().numpy()
        out["dist_funcs_spec"] = df_spec.detach().numpy()

    out["bin_cents"] = bin_cents
    out["resps"] = resps
    return out


@pyro.infer.config_enumerate
def tcc_swap_model(
    targ,
    dist,
    bin_cents,
    resp_inds=None,
    hn_width=10,
    alpha=0.5,
):
    width = pyro.sample(
        "width",
        distribs.HalfNormal(hn_width),
    )
    snr = pyro.sample(
        "snr",
        distribs.HalfNormal(hn_width),
    )
    swap_rate = pyro.sample(
        "swap_rate", distribs.Dirichlet(torch.tensor([alpha, alpha]))
    )
    colors = torch.stack((targ, dist), dim=1)
    colors = torch.unsqueeze(colors, -1)
    bin_cents = torch.unsqueeze(bin_cents, 0)
    bin_cents = torch.unsqueeze(bin_cents, 0)

    targs = snr * torch.exp(torch.cos(bin_cents - colors) / width)
    denom = torch.sum(torch.exp(targs), dim=2, keepdims=True)
    probs = torch.exp(targs) / denom

    with pyro.plate("data", len(targ)) as ind:
        swap = pyro.sample("swap", distribs.Categorical(swap_rate))
        use_probs = Vindex(probs)[ind, swap]
        targ_distr = distribs.Categorical(use_probs)
        out = pyro.sample("obs", targ_distr, obs=resp_inds)
        return out


def fit_tcc_swap_model(
    targs,
    dists,
    resps,
    n_bins=10,
    bounds=(-np.pi, np.pi),
    n_samps=500,
    **kwargs,
):
    nan_mask = ~np.isnan(resps)
    targs = targs[nan_mask]
    dists = dists[nan_mask]
    resps = resps[nan_mask]
    bins = np.linspace(*bounds, n_bins + 1)
    resp_inds = torch.tensor(np.digitize(resps, bins) - 1)
    bin_cents = torch.tensor(bins[:-1] + np.diff(bins)[0] / 2, dtype=torch.float)
    targs = torch.tensor(targs, dtype=torch.float)
    dists = torch.tensor(dists, dtype=torch.float)

    loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    out = gpu.fit_model(
        (targs, dists, bin_cents),
        (resp_inds,),
        tcc_swap_model,
        loss=loss,
        block_vars=["swap"],
        **kwargs,
    )
    preds = gpu.sample_fit_model(
        (
            targs,
            dists,
            bin_cents,
        ),
        out["model"],
        out["guide"],
        n_samps=n_samps,
    )
    out["predictive_samples"] = bin_cents[preds]
    out["targs"] = targs.detach().numpy()
    out["dists"] = dists.detach().numpy()
    out["resps"] = resps
    return out


def tcc_model(bin_cents, err_inds=None, dist_err_inds=None, n_samples=100):
    width = pyro.sample(
        "width",
        distribs.HalfNormal(10),
    )
    snr = pyro.sample(
        "snr",
        distribs.HalfNormal(10),
    )
    func = snr * torch.exp(-(bin_cents**2) / (2 * width**2))
    probs = torch.exp(func) / torch.sum(torch.exp(func))
    if err_inds is None:
        n_data = n_samples
    else:
        n_data = len(err_inds)
    with pyro.plate("data", n_data):
        targ_distr = distribs.Categorical(probs)
        return pyro.sample("obs", targ_distr, obs=err_inds)


def fit_tcc_model(
    errs,
    dist_errs,
    n_bins=10,
    bounds=(-np.pi, np.pi),
    n_samps=500,
    **kwargs,
):
    bins = np.linspace(*bounds, n_bins + 1)
    bin_cents = torch.tensor(bins[:-1] + np.diff(bins)[0] / 2)
    err_inds = torch.tensor(np.digitize(errs, bins) - 1)
    dist_err_inds = torch.tensor(np.digitize(dist_errs, bins) - 1)

    out = gpu.fit_model((bin_cents,), (err_inds, dist_err_inds), tcc_model, **kwargs)

    preds = gpu.sample_fit_model(
        (bin_cents,),
        out["model"],
        out["guide"],
        n_samps=n_samps,
    )

    out["predictive_samples"] = preds
    return out
