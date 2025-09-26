import numpy as np
import pyro
import torch
import sklearn.preprocessing as skp
import functools as ft

import general.plotting as gpl
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
        bin_cents,
        u.normalize_periodic_range(targs),
        rescale=rescale_kernel,
    )
    targ_func_spec = np.squeeze(targ_func_spec)
    _, dist_func_spec = gpkernel.get_conditioned_kernel(
        bin_cents,
        u.normalize_periodic_range(dists),
        rescale=rescale_kernel,
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


def normalize_periodic(x):
    normed = 2 * torch.arcsin(torch.sin((x) / 2))
    return normed


class NormalGuessMixture(distribs.Distribution):
    def __init__(
        self,
        corr_mu,
        sigma,
        probs,
        bounds=(-torch.pi, torch.pi),
        vonmises=True,
    ):
        if vonmises:
            self.resp_distr = distribs.VonMises(torch.zeros_like(corr_mu), 1 / sigma)
        else:
            self.resp_distr = distribs.Normal(torch.zeros_like(corr_mu), sigma)
        self.corr_mu = corr_mu
        b0 = torch.ones_like(corr_mu) * bounds[0]
        self.guess_distr = distribs.Uniform(b0, bounds[1])
        self.categ_distr = distribs.Categorical(probs)
        self.cat_log_probs = torch.log(probs)

    @property
    def batch_shape(self):
        return self.guess_distr.batch_shape

    @property
    def event_shape(self):
        return self.guess_distr.event_shape

    def sample(self, *args, **kwargs):
        corr_samps = self.corr_mu + self.resp_distr.sample(*args, **kwargs)
        guess_samps = self.guess_distr.sample(*args, **kwargs)

        cat_samps = self.categ_distr.sample(corr_samps.shape, *args, **kwargs)
        distr_samps = torch.stack((corr_samps, guess_samps), axis=0)
        samps = distr_samps[cat_samps, torch.arange(guess_samps.shape[0])]
        samps = normalize_periodic(samps)
        return samps

    def log_prob(self, x, *args, **kwargs):
        norm_corr_x = normalize_periodic(x - self.corr_mu)
        corr_lp = self.resp_distr.log_prob(norm_corr_x, *args, **kwargs)
        guess_lp = self.guess_distr.log_prob(x, *args, **kwargs)

        grouped = torch.stack(
            (
                corr_lp + self.cat_log_probs[0],
                guess_lp + self.cat_log_probs[1],
            ),
            axis=0,
        )
        return torch.logsumexp(grouped, 0)


class NormalSwapGuessMixture(distribs.Distribution):
    def __init__(
        self,
        corr_mu,
        swap_mu,
        sigma,
        probs,
        bounds=(-torch.pi, torch.pi),
        vonmises=False,
    ):
        if vonmises:
            self.resp_distr = distribs.VonMises(torch.zeros_like(corr_mu), 1 / sigma)
        else:
            self.resp_distr = distribs.Normal(torch.zeros_like(corr_mu), sigma)
        self.corr_mu = corr_mu
        self.swap_mu = swap_mu
        b0 = torch.ones_like(swap_mu) * bounds[0]
        self.guess_distr = distribs.Uniform(b0, bounds[1])
        self.categ_distr = distribs.Categorical(probs)
        self.cat_log_probs = torch.log(probs)

    @property
    def batch_shape(self):
        return self.guess_distr.batch_shape

    @property
    def event_shape(self):
        return self.guess_distr.event_shape

    def sample(self, *args, **kwargs):
        corr_samps = self.corr_mu + self.resp_distr.sample(*args, **kwargs)
        swap_samps = self.swap_mu + self.resp_distr.sample(*args, **kwargs)
        guess_samps = self.guess_distr.sample(*args, **kwargs)

        cat_samps = self.categ_distr.sample(corr_samps.shape, *args, **kwargs)
        distr_samps = torch.stack((corr_samps, swap_samps, guess_samps), axis=0)
        samps = distr_samps[cat_samps, torch.arange(guess_samps.shape[0])]
        samps = normalize_periodic(samps)
        return samps

    def log_prob(self, x, *args, **kwargs):
        norm_corr_x = normalize_periodic(x - self.corr_mu)
        norm_swap_x = normalize_periodic(x - self.swap_mu)
        norm_x = normalize_periodic(x)
        corr_lp = self.resp_distr.log_prob(norm_corr_x, *args, **kwargs)
        swap_lp = self.resp_distr.log_prob(norm_swap_x, *args, **kwargs)
        guess_lp = self.guess_distr.log_prob(norm_x, *args, **kwargs)

        grouped = torch.stack(
            (
                corr_lp + self.cat_log_probs[0],
                swap_lp + self.cat_log_probs[1],
                guess_lp + self.cat_log_probs[2],
            ),
            axis=0,
        )
        return torch.logsumexp(grouped, 0)


def corr_guess_model(
    targ,
    resps=None,
    hn_width=10,
    alpha=0.5,
    bounds=(-np.pi, np.pi),
):
    sigma = pyro.sample(
        "sigma",
        distribs.HalfNormal(hn_width),
    )
    resp_rate = pyro.sample(
        "resp_rate", distribs.Dirichlet(torch.tensor([alpha, alpha]))
    )

    with pyro.plate("data", len(targ)):
        out = pyro.sample(
            "obs",
            NormalGuessMixture(targ, sigma, resp_rate),
            obs=resps,
        )
        return out


def corr_guess_swap_model(
    targ,
    dist,
    resps=None,
    hn_width=10,
    alpha=0.5,
):
    sigma = pyro.sample(
        "sigma",
        distribs.HalfNormal(hn_width),
    )
    resp_rate = pyro.sample(
        "resp_rate", distribs.Dirichlet(torch.tensor([alpha, alpha, alpha]))
    )

    with pyro.plate("data", len(targ)):
        out = pyro.sample(
            "obs",
            NormalSwapGuessMixture(targ, dist, sigma, resp_rate),
            obs=resps,
        )
        return out


def fit_corr_guess_model(
    targs,
    resps,
    n_samps=500,
    **kwargs,
):
    nan_mask = ~np.isnan(resps)
    targs = u.normalize_periodic_range(targs[nan_mask])
    resps = u.normalize_periodic_range(resps[nan_mask])
    targs = torch.tensor(targs, dtype=torch.float)
    resps = torch.tensor(resps, dtype=torch.float)

    loss = pyro.infer.Trace_ELBO()
    out = gpu.fit_model(
        (targs,),
        (resps,),
        corr_guess_model,
        loss=loss,
        block_vars=["kind"],
        **kwargs,
    )
    preds = gpu.sample_fit_model(
        (targs,),
        out["model"],
        out["guide"],
        n_samps=n_samps,
    )
    out["predictive_samples"] = preds
    out["targs"] = targs.detach().numpy()
    out["resps"] = resps.detach().numpy()
    return out


def fit_corr_guess_swap_model(
    targs,
    dists,
    resps,
    n_samps=500,
    **kwargs,
):
    nan_mask = ~np.isnan(resps)
    targs = u.normalize_periodic_range(targs[nan_mask])
    dists = u.normalize_periodic_range(dists[nan_mask])
    resps = u.normalize_periodic_range(resps[nan_mask])
    targs = torch.tensor(targs, dtype=torch.float)
    dists = torch.tensor(dists, dtype=torch.float)
    resps = torch.tensor(resps, dtype=torch.float)

    loss = pyro.infer.Trace_ELBO()
    out = gpu.fit_model(
        (targs, dists),
        (resps,),
        corr_guess_swap_model,
        loss=loss,
        block_vars=["kind"],
        **kwargs,
    )
    preds = gpu.sample_fit_model(
        (
            targs,
            dists,
        ),
        out["model"],
        out["guide"],
        n_samps=n_samps,
    )
    out["predictive_samples"] = preds
    out["targs"] = targs.detach().numpy()
    out["dists"] = dists.detach().numpy()
    out["resps"] = resps.detach().numpy()
    return out


@pyro.infer.config_enumerate
def tcc_swap_model(
    targ,
    dist,
    bin_cents,
    resp_inds=None,
    hn_width=10,
    alpha=0.5,
    width=None,
    swap_rate=None,
    snr=None,
):
    if width is None:
        width = pyro.sample(
            "width",
            distribs.HalfNormal(hn_width),
        )
    if snr is None:
        snr = pyro.sample(
            "snr",
            distribs.HalfNormal(hn_width),
        )
    if swap_rate is None:
        swap_rate = pyro.sample(
            "swap_rate", distribs.Dirichlet(torch.tensor([alpha, alpha]))
        )
    colors = torch.stack((targ, dist), dim=1)
    colors = torch.unsqueeze(colors, -1)
    bin_cents = torch.unsqueeze(bin_cents, 0)
    bin_cents = torch.unsqueeze(bin_cents, 0)

    targs = snr * vm_kernel(bin_cents - colors, width)
    # gaussian doesn't work
    # targs = snr * torch.exp(-(bin_cents - colors)**2 / (2 * width ** 2))
    denom = torch.sum(torch.exp(targs), dim=2, keepdims=True)
    probs = torch.exp(targs) / denom

    with pyro.plate("data", len(targ)) as ind:
        swap = pyro.sample("swap", distribs.Categorical(swap_rate))
        use_probs = Vindex(probs)[ind, swap]
        targ_distr = distribs.Categorical(use_probs)
        out = pyro.sample("obs", targ_distr, obs=resp_inds)
        return out


@pyro.infer.config_enumerate
def tcc_kernel_model(
    targ,
    dist,
    targ_kfuncs,
    dist_kfuncs,
    bin_cents,
    resp_inds=None,
    hn_width=10,
    alpha_mix=0.5,
    alpha_swap=0.5,
    kernel_mix=None,
    width=1,
    swap_rate=None,
    snr=None,
):
    """
    targ : array_like
        list of target colors (N)
    dist : array_like
        list of distractor colors
    targ_kfuncs : array_like
        array of kernels corresponding to every target color (N x len(bin_cents))
    dist_kfuncs : array_like
        array of kernels corresponding to every distractor color (N x len(bin_cents))
    bin_cents : array_like
        list of centers of bins, should span the range of targ and dist
    resp_inds : array_like, optional
        indices of the participant response for every trial in terms of the
        bin_cents (N)
    """
    if kernel_mix is None:
        kernel_mix = pyro.sample(
            "kernel_mix", distribs.Dirichlet(torch.tensor([alpha_mix, alpha_mix]))
        )
    if snr is None:
        snr = pyro.sample(
            "snr",
            distribs.HalfNormal(hn_width),
        )
    if swap_rate is None:
        swap_rate = pyro.sample(
            "swap_rate", distribs.Dirichlet(torch.tensor([alpha_swap, alpha_swap]))
        )
    colors = torch.stack((targ, dist), dim=1)
    colors = torch.unsqueeze(colors, -1)
    bin_cents = torch.reshape(bin_cents, (1, 1, -1))
    kernel_comb = torch.stack((targ_kfuncs, dist_kfuncs), dim=1)

    targs = snr * (
        kernel_mix[0] * vm_kernel(bin_cents - colors, width)
        + kernel_mix[1] * kernel_comb
    )
    # gaussian doesn't work
    # targs = snr * torch.exp(-(bin_cents - colors)**2 / (2 * width ** 2))
    denom = torch.sum(torch.exp(targs), dim=2, keepdims=True)
    probs = torch.exp(targs) / denom

    with pyro.plate("data", len(targ)) as ind:
        swap = pyro.sample("swap", distribs.Categorical(swap_rate))
        use_probs = Vindex(probs)[ind, swap]
        targ_distr = distribs.Categorical(use_probs)
        out = pyro.sample("obs", targ_distr, obs=resp_inds)
        return out


# def vm_kernel2(x, wid):
#     k = 1 / wid
#     num = torch.exp(k * torch.cos(x)) - torch.special.bessel_j0(k)
#     denom = torch.exp(k) - torch.special.bessel_j0(k)
#     return num / denom


# def vm_kernel(x, wid):
#     k = 1 / wid
#     num = torch.exp(k * torch.cos(x))
#     denom = torch.exp(k)
#     return num / denom


def vm_kernel(x, wid):
    k = 1 / wid
    num = torch.exp(k * torch.cos(x)) - torch.special.i0(k)
    denom = torch.sqrt(torch.special.i0(2 * k) - torch.special.i0(k) ** 2)
    return num / denom


def tcc_ll(
    targ,
    bin_cents,
    resp_inds,
    snr,
    wid,
):
    colors = torch.unsqueeze(targ, -1)
    bin_cents = torch.unsqueeze(bin_cents, 0)

    targs = snr * vm_kernel(bin_cents - colors, wid)
    denom = torch.sum(torch.exp(targs), dim=1, keepdims=True)
    probs = torch.exp(targs) / denom
    ll = distribs.Categorical(probs).log_prob(resp_inds)
    return np.mean(ll.detach().numpy())


def tcc_model(
    targ,
    bin_cents,
    resp_inds=None,
    hn_width=10,
    alpha=0.5,
    width=None,
    snr=None,
):
    if width is None:
        width = pyro.sample(
            "width",
            distribs.HalfNormal(hn_width),
        )
    if snr is None:
        snr = pyro.sample(
            "snr",
            distribs.HalfNormal(hn_width),
        )
    colors = torch.unsqueeze(targ, -1)
    bin_cents = torch.unsqueeze(bin_cents, 0)

    targs = snr * vm_kernel(bin_cents - colors, width)
    denom = torch.sum(torch.exp(targs), dim=1, keepdims=True)
    probs = torch.exp(targs) / denom

    with pyro.plate("data", len(targ)):
        targ_distr = distribs.Categorical(probs)
        out = pyro.sample("obs", targ_distr, obs=resp_inds)
        return out


def _norm_kfuncs(kf):
    mu = np.mean(kf, axis=1, keepdims=True)
    std = np.std(kf, axis=1, keepdims=True)
    kf_norm = (kf - mu) / std
    return kf_norm


def fit_tcc_kernel_model(
    targs,
    dists,
    resps,
    kernel_func,
    n_bins=101,
    bounds=(-np.pi, np.pi),
    n_samps=500,
    fixed_params=None,
    model=tcc_kernel_model,
    **kwargs,
):
    nan_mask = ~np.isnan(resps)
    targs = u.normalize_periodic_range(targs[nan_mask])
    dists = u.normalize_periodic_range(dists[nan_mask])
    resps = resps[nan_mask]
    bins = np.linspace(*bounds, n_bins + 1)
    resp_inds = torch.tensor(np.digitize(resps, bins) - 1)
    bin_cents = torch.tensor(bins[:-1] + np.diff(bins)[0] / 2, dtype=torch.float)

    targs_grid, bc_grid = np.meshgrid(targs, bin_cents)
    targ_pts = np.stack((targs_grid.flatten(), bc_grid.flatten()), axis=1)
    targ_kfuncs = kernel_func(targ_pts, mean=True).reshape(targs_grid.shape).T
    dists_grid, bc_grid = np.meshgrid(dists, bin_cents)
    dist_pts = np.stack((dists_grid.flatten(), bc_grid.flatten()), axis=1)
    dist_kfuncs = kernel_func(dist_pts, mean=True).reshape(dists_grid.shape).T

    targ_kfuncs = _norm_kfuncs(targ_kfuncs)
    dist_kfuncs = _norm_kfuncs(dist_kfuncs)

    targs = torch.tensor(targs, dtype=torch.float)
    dists = torch.tensor(dists, dtype=torch.float)
    targ_kfuncs = torch.tensor(targ_kfuncs, dtype=torch.float)
    dist_kfuncs = torch.tensor(dist_kfuncs, dtype=torch.float)
    if fixed_params is None:
        fixed_params = {}
    else:
        fixed_params = {
            k: torch.tensor(v, dtype=torch.float) for k, v in fixed_params.items()
        }
    model = ft.partial(model, **fixed_params)

    loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    out = gpu.fit_model(
        (targs, dists, targ_kfuncs, dist_kfuncs, bin_cents),
        (resp_inds,),
        model,
        loss=loss,
        block_vars=["swap"],
        **kwargs,
    )
    preds = gpu.sample_fit_model(
        (
            targs,
            dists,
            targ_kfuncs,
            dist_kfuncs,
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


def fit_tcc_swap_model(
    targs,
    dists,
    resps,
    n_bins=101,
    bounds=(-np.pi, np.pi),
    n_samps=500,
    fixed_params=None,
    model=tcc_swap_model,
    **kwargs,
):
    nan_mask = ~np.isnan(resps)
    targs = u.normalize_periodic_range(targs[nan_mask])
    dists = u.normalize_periodic_range(dists[nan_mask])
    resps = resps[nan_mask]
    bins = np.linspace(*bounds, n_bins + 1)
    resp_inds = torch.tensor(np.digitize(resps, bins) - 1)
    bin_cents = torch.tensor(bins[:-1] + np.diff(bins)[0] / 2, dtype=torch.float)
    targs = torch.tensor(targs, dtype=torch.float)
    dists = torch.tensor(dists, dtype=torch.float)
    if fixed_params is None:
        fixed_params = {}
    model = ft.partial(model, **fixed_params)

    loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    out = gpu.fit_model(
        (targs, dists, bin_cents),
        (resp_inds,),
        model,
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


def fit_tcc_model(
    targs,
    resps,
    n_bins=101,
    bounds=(-np.pi, np.pi),
    n_samps=500,
    fixed_params=None,
    model=tcc_model,
    **kwargs,
):
    nan_mask = ~np.isnan(resps)
    targs = u.normalize_periodic_range(targs[nan_mask])
    resps = u.normalize_periodic_range(resps[nan_mask])
    bins = np.linspace(*bounds, n_bins + 1)
    resp_inds = torch.tensor(np.digitize(resps, bins) - 1)
    bin_cents = torch.tensor(bins[:-1] + np.diff(bins)[0] / 2, dtype=torch.float)
    targs = torch.tensor(targs, dtype=torch.float)
    if fixed_params is None:
        fixed_params = {}
    model = ft.partial(model, **fixed_params)

    loss = pyro.infer.Trace_ELBO()
    out = gpu.fit_model(
        (targs, bin_cents),
        (resp_inds,),
        model,
        loss=loss,
        block_vars=["swap"],
        **kwargs,
    )
    preds = gpu.sample_fit_model(
        (
            targs,
            bin_cents,
        ),
        out["model"],
        out["guide"],
        n_samps=n_samps,
    )
    out["samples"]["width"] = out["samples"]["width"]
    out["predictive_samples"] = bin_cents[preds]
    out["targs"] = targs.detach().numpy()
    out["resps"] = resps
    return out


@gpl.ax_adder()
def plot_predictive_distribution(
    model_fit,
    resp_key="resps",
    samp_key="predictive_samples",
    targ_key="targs",
    only_model=False,
    ax=None,
    **kwargs,
):
    err_model = model_fit[samp_key] - model_fit[targ_key][None]
    err_true = model_fit[resp_key] - model_fit[targ_key]
    err_model = u.normalize_periodic_range(err_model).flatten()
    err_true = u.normalize_periodic_range(err_true).flatten()
    if not only_model:
        _, bins, _ = ax.hist(err_true, density=True, **kwargs)
        kwargs["bins"] = bins
    return ax.hist(err_model, histtype="step", density=True, **kwargs)
