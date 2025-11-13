import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import scipy.special as sps
import scipy.stats as sts
import sklearn.neighbors as skn
import sklearn.gaussian_process as skgp

import general.rf_models as rfm
import general.utility as u
import general.plotting as gpl


def local_err(snr, wid):
    orig = (wid / snr) ** 2

    comb = np.min([orig, np.ones_like(snr) * wid ** 2, threshold_err(snr, wid)], axis=0)
    return comb


def threshold_prob(snr, wid):
    n = (2 * np.pi) / (2 * wid)
    dist = sts.norm(0, 1).cdf(-np.sqrt(2) * snr / 2)

    prod = n * dist
    return np.min((prod, np.ones_like(prod)), axis=0)


def threshold_err(snr, wid):
    return np.ones_like(snr) * np.ones_like(wid) * (1 / 3) * np.pi**2


def mse(snr, wid):
    p = threshold_prob(snr, wid)
    t_err = threshold_err(snr, wid)
    l_err = local_err(snr, wid)
    return (1 - p) * l_err + p * t_err


def optimal_w(snrs, w_bounds=(.1, 10), n_ws=5000):
    ws = np.expand_dims(np.linspace(*w_bounds, n_ws), 0)
    snrs = np.expand_dims(snrs, 1)
    errs = mse(snrs, ws)
    opt_ws = ws[0][np.argmin(errs, axis=1)]
    return opt_ws, np.min(errs, axis=1)


def vm_kernel(x, wid):
    k = 1 / wid
    num = np.exp(k * np.cos(x)) - sps.i0(k)
    denom = np.sqrt(sps.i0(2 * k) - sps.i0(k)**2)
    return num / denom
    

class SimplifiedDiscreteKernelTheory:
    def __init__(
        self,
        snr,
        wid,
        n_bins=1001,
        bounds=(-np.pi, np.pi),
    ):
        self.snr = snr
        self.wid = wid
        self.rng = np.random.default_rng()
        bins = np.linspace(*bounds, n_bins + 1)[:-1]
        self.bin_cents = bins + np.diff(bins)[0] / 2

    def kern_func(self, x):
        return vm_kernel(x, self.wid)

    def simulate_decoding(self, n_samps=1000, **kwargs):
        samps = self.rng.choice(self.bin_cents, size=n_samps)
        diffs = self.bin_cents[None] - samps[:, None]
        mu = self.snr * self.kern_func(diffs)
        probs = np.exp(mu) / np.sum(np.exp(mu), axis=1, keepdims=True)
        choices = np.zeros_like(samps)
        for i, prob in enumerate(probs):
            choices[i] = self.rng.choice(
                self.bin_cents, size=1, p=prob
            )
        return samps, choices


class SDKTFunction(SimplifiedDiscreteKernelTheory):
    def __init__(self, snr, func, **kwargs):
        self.func = func
        super().__init__(snr, None, **kwargs)
        
    def kern_func(self, x):
        return self.func(x)


class SimplifiedKernelTheory:
    def __init__(
        self, snr, wid, n_bins=101, bounds=(-np.pi, np.pi), correlated_noise=False
    ):
        self.snr = snr
        self.wid = wid
        self.rng = np.random.default_rng()
        bins = np.linspace(*bounds, n_bins + 1)[:-1]
        self.bin_cents = bins + np.diff(bins)[0] / 2
        self.correlated_noise = correlated_noise

    def kern_func(self, x):
        return np.exp(np.cos(x) / self.wid)

    def simulate_decoding(self, n_samps=1000, **kwargs):
        samps = self.rng.choice(self.bin_cents, size=n_samps)
        mu = self.snr * self.kern_func(self.bin_cents[None] - samps[:, None])

        bc_diffs = self.bin_cents[None] - self.bin_cents[:, None]
        cov = self.kern_func(bc_diffs)
        if self.correlated_noise:
            noise = sts.multivariate_normal(
                np.zeros_like(self.bin_cents), cov, allow_singular=True
            ).rvs(mu.shape[0])
        else:
            noise = self.rng.normal(0, 1, size=mu.shape)

        trls = mu + noise
        return samps, self.bin_cents[np.argmax(trls, axis=1)]


class KernelPopulation:
    def __init__(
        self,
        pwr,
        wid,
        n_units=1000,
        stim_pts=101,
        stim_range=(-np.pi, np.pi),
        make_periodic=False,
        **kwargs,
    ):
        self.stim_range = stim_range
        self.stim = np.expand_dims(np.linspace(*stim_range, stim_pts), 1)
        self.rng = np.random.default_rng()
        if make_periodic:
            self.stim = u.radian_to_sincos(self.stim)
        self.reps = self.make_pop(self.stim, pwr, wid, n_units)
        self.decoder = skn.KNeighborsRegressor(**kwargs)
        self.decoder.fit(self.reps, self.stim)
        self.wid = wid
        self.n_units = n_units

    def sample_reps(self, n_samps=1000, single_stim=None, add_noise=False, noise=1):
        if single_stim is not None:
            ind = np.argmin(np.sum((self.stim - single_stim) ** 2, axis=1))
            inds = np.array((ind,) * n_samps)
        else:
            inds = self.rng.choice(len(self.stim), size=n_samps)
        use_stim = self.stim[inds]
        use_reps = self.reps[inds]
        if add_noise:
            use_reps = use_reps + self.rng.normal(0, noise, size=use_reps.shape)
        return inds, use_stim, use_reps

    def empirical_kernel(self):
        kernel = self.reps @ self.reps.T
        diffs = np.expand_dims(self.stim, 1) - np.expand_dims(self.stim, 0)

        return diffs.flatten(), kernel.flatten()

    def simulate_decoding(self, **kwargs):
        inds, true_stim, use_reps = self.sample_reps(add_noise=True, **kwargs)
        dec_stim = self.decoder.predict(use_reps)
        return true_stim, dec_stim

    def sample_dec_gp(self, **kwargs):
        inds, true_stim, use_reps = self.sample_reps(add_noise=True, **kwargs)
        dec_samps = use_reps @ self.reps.T
        return np.squeeze(self.stim), dec_samps
    

class PeriodicGPKernelPopulation(KernelPopulation):
    def make_pop(self, stim, pwr, wid, n_units):
        p = pwr / n_units
        kernel = skgp.kernels.ConstantKernel(p) * skgp.kernels.RBF(length_scale=wid)
        self.kernel = kernel
        gp = skgp.GaussianProcessRegressor(kernel=kernel)
        stim_sc = np.squeeze(u.radian_to_sincos(stim))
        y = gp.sample_y(stim_sc, n_samples=n_units, random_state=None)
        return y

    def theoretical_kernel(self, n_bins=100):
        s_diffs = u.radian_to_sincos(np.linspace(*self.stim_range, n_bins))
        k = self.n_units * self.kernel(s_diffs, np.zeros((1, 1)))
        return s_diffs, np.squeeze(k)

    def sample_dec_gp(self, n_samps=1000, noise_sigma=1, **kwargs):
        s_diffs, mu = self.theoretical_kernel(**kwargs)
        cov = self.n_units * noise_sigma**2 * self.kernel(s_diffs, s_diffs)
        pre = self.kernel([[0]], [[0]]) + noise_sigma**2
        post = (
            self.kernel(s_diffs, s_diffs)
            + self.kernel(s_diffs, np.zeros((1, 1)))
            * self.kernel(s_diffs, np.zeros((1, 1))).T
        )
        cov = self.n_units * pre * post
        gp = sts.multivariate_normal(mu, cov, allow_singular=True)
        dec_samps = gp.rvs(n_samps)
        return s_diffs, dec_samps    


class GPKernelPopulation(KernelPopulation):
    def make_pop(self, stim, pwr, wid, n_units):
        p = pwr / n_units
        kernel = skgp.kernels.ConstantKernel(p) * skgp.kernels.RBF(length_scale=wid)
        self.kernel = kernel
        gp = skgp.GaussianProcessRegressor(kernel=kernel)
        y = gp.sample_y(stim, n_samples=n_units, random_state=None)
        return y

    def theoretical_kernel(self, n_bins=100):
        s_diffs = np.expand_dims(np.linspace(*self.stim_range, n_bins), 1)
        k = self.n_units * self.kernel(s_diffs, np.zeros((1, 1)))
        return s_diffs, np.squeeze(k)

    def sample_dec_gp(self, n_samps=1000, noise_sigma=1, **kwargs):
        s_diffs, mu = self.theoretical_kernel(**kwargs)
        cov = self.n_units * noise_sigma**2 * self.kernel(s_diffs, s_diffs)
        pre = self.kernel([[0]], [[0]]) + noise_sigma**2
        post = (
            self.kernel(s_diffs, s_diffs)
            + self.kernel(s_diffs, np.zeros((1, 1)))
            * self.kernel(s_diffs, np.zeros((1, 1))).T
        )
        cov = self.n_units * pre * post
        gp = sts.multivariate_normal(mu, cov, allow_singular=True)
        dec_samps = gp.rvs(n_samps)
        return s_diffs, dec_samps


class RFKernelPopulation(KernelPopulation):
    def make_pop(self, stim, pwr, wid, n_units):
        cents = self.rng.choice(stim, size=n_units)
        wids = np.ones_like(cents) * wid**2
        (r, _), (_, _, scale, _) = rfm.make_gaussian_vector_rf(
            cents,
            wids,
            pwr,
            0,
            titrate_samps=stim,
            return_params=True,
        )
        self.scale = scale
        y = r(stim)
        return y

    def theoretical_kernel(self, n_bins=100):
        range_ = np.abs(self.stim[-1] - self.stim[0])
        s_diffs = np.linspace(-range_, range_, n_bins)

        k = analytic_kernel(s_diffs, self.scale, self.wid, self.n_units)
        k = np.squeeze(k)
        return s_diffs, k


def analytic_kernel_periodic(delta, scale, wid, n_units=100):
    pre = (wid * scale**2) / (2 * np.sqrt(np.pi))
    a = np.exp(-(delta**2) / (4 * wid**2))
    b, c = (
        sps.erf((delta + 2 * np.pi) / (2 * wid)),
        sps.erf((delta - 2 * np.pi) / (2 * wid)),
    )
    return n_units * pre * a * (b - c)


def analytic_kernel(delta, scale, wid, n_units=100):
    pre = (wid * scale**2) / (2 * np.sqrt(np.pi))
    a = np.exp(-(delta**2) / (4 * wid**2))
    pi = np.pi
    spi = np.sqrt(pi)
    b, c, d, e, f, g, h, i, j, k = (
        1 / 4 * wid,
        spi * (2 * pi - delta) * np.exp(delta**2 / (4 * wid**2)),
        sps.erf((delta * np.pi) / (2 * wid)),
        sps.erf((delta - 2 * np.pi) / (2 * wid)),
        2 * wid * (np.exp(pi * (delta - pi) / wid**2) - 1),
        1 / 4 * wid * np.exp(-pi * (delta + pi) / wid**2),
        np.sqrt(2) * (delta + 2 * pi) * np.exp((delta + 2 * pi) ** 2 / (4 * wid**2)),
        sps.erf((delta + 2 * pi) / (2 * wid)),
        sps.erf(delta / (2 * wid)),
        2 * wid * (np.exp(pi * (delta + pi) / wid**2) - 1),
    )
    second = b * (c * (d - e) + f) + g * (h * (i - j) - k)
    return n_units * pre * a * second


def angle_decoding(
    train_range=(-np.pi, np.pi),
    test_range=None,
    train_offset=(0, 0),
    test_offset=None,
    n_samps=10000,
    model=skn.KNeighborsRegressor,
    pwr=2,
    **kwargs,
):
    train_range = (train_range[0], train_range[1] - train_range[0])
    if test_range is None:
        test_range = train_range
    else:
        test_range = (test_range[0], test_range[1] - test_range[0])
    train_offset = np.array(train_offset)
    if len(train_offset.shape) == 1:
        train_offset = np.expand_dims(train_offset, 0)
    if test_offset is None:
        test_offset = train_offset
    elif len(test_offset.shape) == 1:
        test_offset = np.expand_dims(test_offset, 0)

    rs_tr = sts.uniform(*train_range).rvs(n_samps)
    rs_te = sts.uniform(*test_range).rvs(n_samps)

    samps_tr = (
        pwr * u.radian_to_sincos(rs_tr)
        + train_offset
        + sts.norm(0, 1).rvs((n_samps, 2))
    )
    m = model(**kwargs)
    m.fit(samps_tr, rs_tr)
    samps_te = (
        pwr * u.radian_to_sincos(rs_te) + test_offset + sts.norm(0, 1).rvs((n_samps, 2))
    )
    rs_te_est = m.predict(samps_te)
    err = u.normalize_periodic_range(rs_te_est - rs_te)
    return err, (rs_te, rs_te_est)


def make_resp_function(theta, offset=0):
    def r(x, y):
        tx = np.array((np.cos(x), np.sin(x)))
        ty = np.array((np.cos(y + offset), np.sin(y + offset)))
        mat1 = [[1, 0, 0, 0], [0, 1, 0, 0]]
        mat2 = [
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, np.cos(theta), 0, np.sin(theta)],
        ]

        return tx @ mat1 + ty @ mat2

    return r


def make_kernel_map(r, n_samps=11, cent=(0, 0)):
    dists = np.zeros((n_samps, n_samps))
    cols = np.linspace(-np.pi, np.pi, n_samps)

    for i, j in it.product(range(n_samps), repeat=2):
        d_ij = np.sum((r(*cent) - r(cols[i], cols[j])) ** 2)
        dists[i, j] = d_ij
    return (cols, cols, dists)


def sample_errs(r, n_trials=1000, noise=1, **kwargs):
    errs = np.zeros(n_trials)
    dist_errs = np.zeros(n_trials)
    rng = np.random.default_rng()
    targs = np.zeros(n_trials)
    dists = rng.uniform(-np.pi, np.pi, size=n_trials)
    for i in range(n_trials):
        xs, ys, km = make_kernel_map(r, cent=(targs[i], dists[i]), **kwargs)
        eta = rng.normal(0, noise, size=km.shape)
        km = km + eta
        x_pos, _ = np.where(km == np.min(km))
        r_i = xs[x_pos][0]
        errs[i] = r_i
        dist_errs[i] = r_i - dists[i]
    dist_errs = u.normalize_periodic_range(dist_errs)
    return errs, dist_errs


def make_distr_prop(noise, n_trials=2500, n_props=10, offset=0, **kwargs):
    angles = np.linspace(0, np.pi / 2, n_props)
    errs_all = np.zeros((n_props, n_trials))
    dist_errs_all = np.zeros((n_props, n_trials))
    for i, prop in enumerate(angles):
        r = make_resp_function(angles[i], offset=offset)
        errs_all[i], dist_errs_all[i] = sample_errs(
            r, noise=noise, n_trials=n_trials, **kwargs
        )
    return errs_all, dist_errs_all, angles


def plot_distr_prop(errs, d_errs, angles=None, axs=None, fwid=3):
    if axs is None:
        f, axs = plt.subplots(
            len(errs),
            2,
            figsize=(fwid * 2, fwid * len(errs)),
            sharex=True,
        )
    if angles is None:
        labels = ("",) * len(errs)
    else:
        labels = list("deg = {:.2f}".format(ang) for ang in angles)
    bins = np.linspace(-np.pi, np.pi, 12)
    for i, errs_i in enumerate(errs):
        ax1, ax2 = axs[i]
        ax1.hist(errs_i, bins=bins, density=True)
        ax2.hist(d_errs[i], bins=bins, density=True)
        ax1.set_title(labels[i])
        gpl.clean_plot(ax1, 0)
        gpl.clean_plot(ax2, 1)
        gpl.make_yaxis_scale_bar(ax1, 0.1, double=False)
    ax1.set_xlabel("error (target difference)")
    ax2.set_xlabel("distractor difference")
    return axs
