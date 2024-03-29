
import numpy as np
import scipy.stats as sts
import scipy.special as spsp


def fit_dprime_pts(
    col_rep_dists,
    col_diffs,
    bhv_errs,
    renorm_dists=True,
    n_bins=10,
    min_d=0,
    max_d=5,
    n_ds=1000,
    filt=True,
    n_samps=10000,
):
    """
    Fit the dprime that best fits the behavioral response distribution given
    a pointwise similarity function.

    Parameters
    ----------
    col_rep_dists : array_like (C,)
        The ordered similarity function, giving the representational distance
        between color 0 and each other color. 
    col_diffs : arrray_like (C,)
        The x-axis for the similarity function, should be the same length as
        col_rep_dists.
    bhv_errs : array_like, (T,)
        Array of all behavioral errors, should have the same range as col_diffs.
    """
    if renorm_dists:
        crd_zeroed = col_rep_dists - np.nanmin(col_rep_dists)
        col_rep_dists = crd_zeroed/np.nanmax(crd_zeroed)
    if n_bins is None:
        n_bins = len(col_diffs)

    if filt:
        bounds = (np.min(col_diffs), np.max(col_diffs))
    else:
        bounds = (-np.pi, np.pi)
    if n_samps is None:
        n_samps = len(bhv_errs)
    hist_bins = np.linspace(*bounds, n_bins)
    bhv_hist = np.histogram(bhv_errs, bins=hist_bins, density=True)[0]
    frozen_noise = sts.norm(0, 1).rvs((n_samps, len(col_rep_dists)))
    def _min_func(dprime):
        e_dists = np.exp(col_rep_dists*dprime)
        probs = e_dists/np.sum(e_dists)
        inds = np.argmax(np.expand_dims(col_rep_dists*dprime, 0) + frozen_noise,
                         axis=1)
        errs = col_diffs[inds]
        pred_hist = np.histogram(errs, bins=hist_bins, density=True)[0]
        kl = np.sum(spsp.kl_div(pred_hist, bhv_hist))
        return kl

    dps = dprimes = np.linspace(min_d, max_d, n_ds)
    kls = list(_min_func(dp) for dp in dps)
    best_dp = dps[np.argmin(kls)]

    return best_dp


def sample_pts_dprime(
    col_rep_dists,
    col_diffs,
    dprime,
    renorm_dists=True,
    n_bins=10,
    n_samps=10000,
):
    if renorm_dists:
        crd_zeroed = col_rep_dists - np.nanmin(col_rep_dists)
        col_rep_dists = crd_zeroed/np.nanmax(crd_zeroed)
    if n_bins is None:
        n_bins = len(col_diffs)
    noise = sts.norm(0, 1).rvs((n_samps, len(col_rep_dists)))
    pred_reps = np.expand_dims(dprime*col_rep_dists, 0) + noise
    pred_resps = col_diffs[np.argmax(pred_reps, axis=1)]
    return pred_resps


def fit_and_sample_dprime_pts(
    col_rep_dists, col_diffs, bhv_errs, n_bins=20, n_samps=10000, **kwargs,
):
    """
    Fit and sample from the dprime that best fits the behavioral response
    distribution given a pointwise similarity function.

    Parameters
    ----------
    col_rep_dists : array_like (C,)
        The ordered similarity function, giving the representational distance
        between color 0 and each other color. 
    col_diffs : arrray_like (C,)
        The x-axis for the similarity function, should be the same length as
        col_rep_dists.
    bhv_errs : array_like, (T,)
        Array of all behavioral errors, should have the same range as col_diffs.
    """
    dp_fit = fit_dprime_pts(
        col_rep_dists, col_diffs, bhv_errs, n_bins=n_bins, **kwargs
    )
    samps = sample_pts_dprime(
        col_rep_dists, col_diffs, dp_fit, n_bins=n_bins, n_samps=n_samps,
    )
    return dp_fit, samps
