import numpy as np
import pyro
import torch
import imblearn.under_sampling as imb_us
import tqdm
from torch.utils.data import TensorDataset, DataLoader

import scipy.stats as sts
import sklearn.preprocessing as skp
import sklearn.metrics.pairwise as skmp
import pyro.contrib.gp as gp
import general.pyro_utility as gpu
import general.neural_analysis as na
import general.utility as u
import pyro.distributions as distribs
import gpytorch as gpt
import gpflow
import logging
import general.plotting as gpl


class SVGPRFlow:
    def __init__(
        self,
        kernel=None,
        opt=None,
        model=gpflow.models.SVGP,
        likelihood=None,
    ):
        if kernel is None:
            kernel = gpflow.kernels.SquaredExponential()
        self.kernel = kernel
        if opt is None:
            opt = gpflow.optimizers.Scipy()
        self.opt = opt
        self.model = model


def similarity_model(
    c_diff_ind,
    n_diffs,
    similarities=None,
    hn_width=10,
):
    sig = pyro.sample("sig", distribs.HalfNormal(hn_width))
    mu = pyro.sample("mu", distribs.Normal(torch.zeros(n_diffs), hn_width))
    with pyro.plate("data", len(c_diff_ind)):
        targ_distr = distribs.Normal(mu[c_diff_ind], sig)
        print("distr", targ_distr)
        print("mu", mu[c_diff_ind].shape)
        out = pyro.sample("obs", targ_distr, obs=similarities)
        print("sample", out.shape)
        return out


def fit_similarity_model(c_diff_ind, diff_cents, similarities, n_samps=500, **kwargs):
    c_diff_flat = c_diff_ind.flatten()
    sim_flat = similarities.flatten()
    mask = ~np.isnan(sim_flat)
    d_mu = np.nanmean(sim_flat)
    c_diff_flat = torch.tensor(c_diff_flat[mask], dtype=torch.int)
    sim_flat = torch.tensor(sim_flat[mask], dtype=torch.float) - d_mu

    loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
    inp = (c_diff_flat, len(diff_cents))
    out = gpu.fit_model(
        inp,
        (sim_flat,),
        similarity_model,
        loss=loss,
        **kwargs,
    )
    preds = gpu.sample_fit_model(
        inp,
        out["model"],
        out["guide"],
        n_samps=n_samps,
    )
    out["predictive_samples"] = preds
    return out


def fit_all_gps(c_diff_ind, diff_cents, similarities, c1, single_ls=True, **kwargs):
    out_gps = {}
    cols = np.unique(c1)
    if single_ls:
        fit_all = fit_similarity_gp(c_diff_ind, diff_cents, similarities, **kwargs)
        use_ls = fit_all["lengthscales"][-1]
        kwargs["ls"] = use_ls
        kwargs["fix_params"] = ("kernel.lengthscale_unconstrained",)
    for i, col_ind in enumerate(cols):
        color = diff_cents[col_ind - 1]
        fit_i = fit_similarity_gp(
            c_diff_ind, diff_cents, similarities, c1=c1, targ_c1=color, **kwargs
        )
        out_gps[color] = fit_i
    return out_gps


def _fit_similarity_gp(
    diff_flat,
    sim_flat,
    var=0.2,
    ls=1,
    noise=1,
    lr=0.005,
    smoke_test=False,
    num_steps=2000,
    c_sep_kernel=False,
    fix_params=(),
    max_inducing=11,
    inducing=None,
    c1_flat=None,
    c2_flat=None,
    circularize_dists=True,
):
    pyro.clear_param_store()
    if inducing is None:
        inducing = torch.tensor(diff_flat)
    X = torch.tensor(diff_flat)
    if len(X.shape) == 1:
        X = X.unsqueeze(1)
    sim_flat = torch.tensor(sim_flat)

    if len(inducing) > max_inducing:
        inducing = torch.tensor(np.linspace(min(inducing), max(inducing), max_inducing))
        inducing = inducing.unsqueeze(1)
    else:
        inducing = X
    if circularize_dists:
        X = torch.concatenate((torch.sin(X), torch.cos(X)), axis=1)
        inducing = torch.concatenate((torch.sin(inducing), torch.cos(inducing)), axis=1)
    kernel = gp.kernels.RBF(
        input_dim=X.shape[1],
        variance=torch.tensor(var),
        lengthscale=torch.tensor(ls),
        active_dims=list(range(X.shape[1])),
    )

    if c_sep_kernel:
        c1_flat = torch.tensor(c1_flat)
        k_c1 = gp.kernels.RBF(
            input_dim=2,
            variance=torch.tensor(var),
            lengthscale=torch.tensor(ls),
            active_dims=list(range(2)),
        )
        c1_sc = torch.tensor(np.stack((np.sin(c1_flat), np.cos(c1_flat)), axis=1))

        c2_flat = torch.tensor(c2_flat)
        k_c2 = gp.kernels.RBF(
            input_dim=2,
            variance=torch.tensor(var),
            lengthscale=torch.tensor(ls),
            active_dims=list(range(-2, 0)),
        )
        c2_sc = torch.tensor(np.stack((np.sin(c2_flat), np.cos(c2_flat)), axis=1))

        kernel = gp.kernels.Product(kern0=k_c1, kern1=k_c2)
        X = torch.concatenate((c1_sc, c2_sc), axis=1)
        if X.shape[0] > max_inducing:
            # inducing is original inducing vars x new vars
            new_inducing = np.linspace(
                -np.pi, np.pi - 2 * np.pi / max_inducing, max_inducing
            )
            new_inducing = np.stack(
                (np.sin(new_inducing), np.cos(new_inducing)), axis=1
            )
            pre_inds, new_inds = np.meshgrid(
                np.arange(inducing.shape[0]), np.arange(new_inducing.shape[0])
            )
            pre_col = new_inducing[pre_inds.flatten()]
            new_col = new_inducing[new_inds.flatten()]
            inducing = torch.tensor(np.concatenate((pre_col, new_col), axis=1))
        else:
            inducing = X
    print(X.shape, sim_flat.shape, inducing.shape)
    gpr = gp.models.SparseGPRegression(
        X, sim_flat.squeeze(), kernel, inducing, noise=torch.tensor(noise)
    )

    param_list = list(p for name, p in gpr.named_parameters() if name not in fix_params)
    optimizer = torch.optim.Adam(param_list, lr=lr)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    variances = []
    lengthscales = []
    noises = []
    use_steps = num_steps if not smoke_test else 2
    for i in range(use_steps):
        noises.append(gpr.noise.item())
        if c_sep_kernel:
            variances.append(gpr.kernel.kern0.variance.item())
            lengthscales.append(
                (
                    gpr.kernel.kern0.lengthscale.item(),
                    gpr.kernel.kern1.lengthscale.item(),
                )
            )
        else:
            variances.append(gpr.kernel.variance.item())
            lengthscales.append(gpr.kernel.lengthscale.item())
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        print(i, loss.item())
        optimizer.step()

        losses.append(loss.item())
    out_dict = {
        "X": X,
        "y": sim_flat,
        "variances": np.array(variances),
        "noises": np.array(noises),
        "lengthscales": np.array(lengthscales),
        "losses": np.array(losses),
        "inducing": inducing,
        "model": gpr,
    }
    return out_dict


def _flatten_and_preprocess(c_diff, similarities, c1, c2, norm=True):
    c_diff_flat = c_diff.flatten()
    sim_flat = similarities.flatten()
    mask = ~np.isnan(sim_flat)
    c_diff_flat = c_diff_flat[mask]
    sim_flat = np.expand_dims(sim_flat[mask], 1)
    if norm:
        sim_flat = skp.StandardScaler().fit_transform(sim_flat)
    if c1 is not None:
        c1 = c1.flatten()[mask]
    if c2 is not None:
        c2 = c2.flatten()[mask]
    return c_diff_flat, sim_flat, c1, c2


def fit_similarity_gp_combined(
    data_dict,
    xs,
    pre_pca=0.95,
    norm=True,
    p_thr=0.6,
    c1_targ=None,
    c1_targ_wid=np.pi / 2,
    **kwargs,
):
    cd_all = []
    sims_all = []
    c1_all = []
    c2_all = []
    for data_use in data_dict.values():
        dists, c_diffs, c1, c2 = prepare_data_continuous(
            data_use["spks"],
            data_use["c_targ"],
            ps=data_use["ps"],
            xs=xs,
            pre_pca=pre_pca,
            norm=False,
            p_thr=p_thr,
        )
        c_diff_flat, sim_flat, c1_flat, c2_flat = _flatten_and_preprocess(
            c_diffs, dists, c1, c2, norm=norm
        )
        cd_all.append(c_diff_flat)
        sims_all.append(sim_flat)
        c1_all.append(c1_flat)
        c2_all.append(c2_flat)
    cd_all = np.concatenate(cd_all)
    sims_all = np.concatenate(sims_all)
    c1_all = np.concatenate(c1_all)
    c2_all = np.concatenate(c2_all)
    if c1_targ is not None:
        mask = np.abs(u.normalize_periodic_range(c1_all - c1_targ)) <= (c1_targ_wid / 2)
        cd_all = cd_all[mask]
        sims_all = sims_all[mask]
        c1_all = c1_all[mask]
        c2_all = c2_all[mask]
    return _fit_similarity_gp(
        cd_all,
        sims_all,
        c1_flat=c1_all,
        c2_flat=c2_all,
        **kwargs,
    )


def fit_similarity_gp(
    c_diff,
    similarities,
    c1=None,
    norm=True,
    **kwargs,
):
    c_diff_flat, sim_flat, c1 = _flatten_and_preprocess(
        c_diff, similarities, c1=c1, norm=norm
    )
    return _fit_similarity_gp(
        c_diff_flat,
        sim_flat,
        c1_flat=c1,
        **kwargs,
    )


def fit_similarity_gp_pseudo(
    c_diff_ind,
    diff_cents,
    similarities,
    c1=None,
    targ_c1=None,
    **kwargs,
):
    c_diff_flat = c_diff_ind.flatten()
    sim_flat = similarities.flatten()
    mask = ~np.isnan(sim_flat)
    d_mu = np.nanmean(sim_flat)
    d_std = np.nanstd(sim_flat)
    sim_flat = sim_flat[mask]
    c_diff_flat = c_diff_flat[mask]
    c_dists = diff_cents[c_diff_flat]
    diff_cents = diff_cents

    sim_flat = (sim_flat - d_mu) / d_std
    if c1 is not None:
        c1 = c1.flatten()[mask]
    if c1 is not None and targ_c1 is not None:
        c_mask = diff_cents[c1 - 1] == targ_c1
        sim_flat = sim_flat[c_mask]
        c_dists = c_dists[c_mask]
    return _fit_similarity_gp(
        c_diff_flat, sim_flat, c1_flat=c1, inducing=diff_cents, **kwargs
    )


def similarity_model_hierarchical(
    c_diff_ind,
    c1_ind,
    c2_ind,
    diff_cents,
    n_diffs,
    n_col,
    similarities=None,
    hn_width=10,
):
    sig_o = pyro.sample("sig_o", distribs.HalfNormal(hn_width))
    mu_o = pyro.sample("mu_o", distribs.Normal(torch.zeros(n_diffs), hn_width))
    sig = pyro.sample("sig", distribs.HalfNormal(hn_width))
    with pyro.plate("colors", n_col):
        sub_mu = pyro.sample("mu_sub", distribs.Normal(mu_o, sig_o))
    with pyro.plate("data", len(c_diff_ind)):
        td1 = distribs.Normal(sub_mu[c1_ind, c_diff_ind], sig)
        x = pyro.sample(
            "obs",
            td1,
            obs=similarities,
        )
        td2 = distribs.Normal(sub_mu[c2_ind, -c_diff_ind], sig)
        y = pyro.sample(
            "obs",
            td2,
            obs=similarities,
        )
        return x, y


def fit_single_neuron_gps(
    pickles, xs, norm=True, pca=None, color_key="c_targ", **kwargs
):
    out_models = {}
    out_full = {}
    for key, pickle in pickles.items():
        resps = pickle["spks"]
        colors = pickle[color_key]
        ps = pickle["ps"]
        resps, colors = prepare_data_neuron_fits(
            resps,
            colors,
            xs=xs,
            ps=ps,
            norm=norm,
            pca=pca,
        )
        m_key = _fit_single_neuron_gps(resps, colors, **kwargs)
        out_models[key] = m_key["model"]
        out_full[key] = m_key
    return out_models, out_full


def prepare_data_neuron_fits(
    resps, colors, ps=None, xs=None, ps_thr=0.6, p_ind=0, x_targ=-0.25, **kwargs
):
    if xs is not None:
        ind = np.argmin((xs - x_targ) ** 2)
        resps = resps[..., ind]
    if ps is not None:
        mask = ps[:, p_ind] > ps_thr
        resps = resps[mask]
        colors = colors[mask]
    pipe = na.make_model_pipeline(**kwargs)
    if len(pipe.steps) > 0:
        resps = pipe.fit_transform(resps)
    return resps, colors


def _fit_single_neuron_gps(
    resps,
    colors,
    circularize_color=True,
    max_inducing=10,
    individual_gps=True,
    **kwargs,
):
    if colors.shape[0] > max_inducing:
        inducing = np.linspace(-np.pi, np.pi, max_inducing + 1)[::-1].copy()
    else:
        inducing = colors
    if circularize_color:
        colors = np.stack((np.sin(colors), np.cos(colors)), axis=1)
        inducing = np.stack((np.sin(inducing), np.cos(inducing)), axis=1)
    else:
        colors = np.expand_dims(colors, 1)
        inducing = np.expand_dims(inducing, 1)

    if individual_gps:
        out = _fit_indiv_gps(colors, resps, inducing, **kwargs)
    else:
        out = _fit_joint_gp(colors, resps, inducing, **kwargs)
    return out


def _fit_indiv_gps(
    colors,
    resps,
    inducing,
    noise=1,
    var=0.2,
    ls=1,
    fix_params=(),
    lr=0.005,
    num_steps=2000,
    show_logging=False,
    smoke_test=False,
):
    variances = np.zeros((resps.shape[1], num_steps))
    noises = np.zeros_like(variances)
    lengthscales = np.zeros_like(variances)
    losses = np.zeros_like(variances)
    models = np.zeros(resps.shape[1], dtype=object)
    for j in range(resps.shape[1]):
        pyro.clear_param_store()
        resps_j = torch.tensor(resps[:, j])
        colors_j = torch.tensor(colors)

        kernel = gp.kernels.RBF(
            input_dim=colors.shape[1],
            variance=torch.tensor(var),
            lengthscale=torch.tensor(ls),
        )

        gpr = gp.models.SparseGPRegression(
            colors_j,
            resps_j,
            kernel,
            torch.tensor(inducing),
            noise=torch.tensor(noise),
        )
        param_list = list(
            p for name, p in gpr.named_parameters() if name not in fix_params
        )
        optimizer = torch.optim.Adam(param_list, lr=lr)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        use_steps = num_steps if not smoke_test else 2
        for i in range(use_steps):
            noises[j, i] = gpr.noise.item()
            variances[j, i] = gpr.kernel.variance.item()
            lengthscales[j, i] = gpr.kernel.lengthscale.item()
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
            losses[j, i] = loss.item()
            if i % 100 == 0 and show_logging:
                logging.info("{}. Elbo loss: {}".format(j, loss))
        models[j] = gpr
    out_dict = {
        "X": colors,
        "y": resps,
        "variances": np.array(variances),
        "noises": np.array(noises),
        "lengthscales": np.array(lengthscales),
        "losses": np.array(losses),
        "inducing": inducing,
        "model": models,
    }
    return out_dict


class GPWrapper:
    def __init__(self, train_x, train_y, gp):
        self.gp = gp
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = gp.likelihood_store
        self.eval_mode = False
        self.n_dim = train_y.shape[1]
        self.losses = None

    def train(self):
        self.gp.train()
        self.likelihood.train()
        self.eval_mode = False

    def eval(self):
        self.gp.eval()
        self.likelihood.eval()
        self.eval_mode = True

    def train_model(
        self,
        num_steps=2000,
        lr=0.005,
        show_logging=False,
        smoke_test=False,
        fix_params=(),
    ):
        self.train()
        param_list = list(
            p for name, p in self.gp.named_parameters() if name not in fix_params
        )
        optimizer = torch.optim.Adam(param_list, lr=lr)
        loss_fn = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
        losses = []
        use_steps = num_steps if not smoke_test else 2
        for i in range(use_steps):
            optimizer.zero_grad()
            output = self.gp(self.train_x)
            loss = -loss_fn(output, self.train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 100 == 0 and show_logging:
                logging.info("Elbo loss: {}".format(loss))
        self.losses = losses
        self.eval()

    def __call__(self, inp, **kwargs):
        if not self.eval_mode:
            self.eval()
            switch = True
        else:
            switch = False
        mu, cov = _mu_cov_gp(self.gp, self.likelihood, inp)
        if switch:
            self.train()
        return mu, cov


def _mu_cov_gp(gp, likelihood, inp):
    with torch.no_grad():
        predictions = likelihood(gp(inp))
        mu = predictions.mean
        cov = predictions.variance
    return mu, cov


class MultitaskGPModel(gpt.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=None, task_rank=1):
        num_tasks = train_y.shape[1]
        if likelihood is None:
            likelihood = gpt.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_tasks
            )
        super().__init__(train_x, train_y, likelihood)
        self.likelihood_store = likelihood
        self.num_tasks = num_tasks
        self.mean_module = gpt.means.MultitaskMean(
            gpt.means.ConstantMean(),
            num_tasks=num_tasks,
        )
        self.covar_module = gpt.kernels.MultitaskKernel(
            gpt.kernels.RBFKernel(),
            num_tasks=num_tasks,
            rank=task_rank,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def _fit_joint_gp(
    colors,
    resps,
    inducing=None,
    noise=1,
    var=0.2,
    ls=1,
    **kwargs,
):
    torch.set_default_dtype(torch.double)
    resps = torch.tensor(resps)
    colors = torch.tensor(colors)

    gpr = MultitaskGPModel(colors, resps)
    gp_wrapper = GPWrapper(colors, resps, gpr)
    gp_wrapper.train_model(**kwargs)

    out_dict = {
        "X": colors.detach().numpy(),
        "y": resps.detach().numpy(),
        "model": gp_wrapper,
    }
    return out_dict


class CombinedGP:
    def __init__(self, gp_list):
        self.gp_list = gp_list
        self.n_dim = len(gp_list)

    def __call__(self, X, full_cov=False, **kwargs):
        def _f(gp, X, **kwargs):
            return gp(X, **kwargs)

        mu, cov = u.non_parallel_stack(
            _f,
            self.gp_list,
            X,
            **kwargs,
            n_outs=2,
            stack_func=torch.stack,
            n_jobs=1,
        )
        return mu.T, cov.T


def _combine_model_dict_gps(md):
    new_dict = {}
    for k, v in md.items():
        new_dict[k] = CombinedGP(v)
    return new_dict


class CombinedSingleNeuronGPKernel:
    def __init__(
        self,
        model_dict,
    ):
        if u.check_list(list(model_dict.values())[0]):
            model_dict = _combine_model_dict_gps(model_dict)
        self.model_dict = model_dict
        self.avg_values = {}
        self.cov_values = {}

    def get_average_kernel(self, bin_cents, rescale=True, **kwargs):
        kernel = np.zeros((len(bin_cents), len(bin_cents)))
        for i, cond_color in enumerate(bin_cents):
            bc_i = u.normalize_periodic_range(bin_cents + cond_color)
            _, k_i = self.get_conditioned_kernel(
                bc_i, cond_color, **kwargs, rescale=False
            )
            kernel[i] = np.squeeze(np.mean(k_i, axis=0))
        kernel = np.mean(kernel, axis=0)
        if rescale:
            scaler = skp.StandardScaler()
            kernel = np.squeeze(scaler.fit_transform(np.expand_dims(kernel, 1)))
        return bin_cents, kernel

    def get_full_kernel(self, bin_cents, **kwargs):
        return self.get_conditioned_kernel(bin_cents, bin_cents, **kwargs)

    def get_model_values(
        self, key, model, vals, noiseless=True, full_cov=False, **kwargs
    ):
        avg_vals = self.avg_values.get(key, {})
        cov_vals = self.cov_values.get(key, {})
        second_dim = model.n_dim
        vals_nt = vals.detach().numpy()
        if noiseless and not full_cov:
            mu_outs = np.array(
                list(
                    avg_vals.get(tuple(v), np.ones(second_dim) * np.nan)
                    for v in vals_nt
                )
            )
            cov_outs = np.array(
                list(
                    cov_vals.get(tuple(v), np.ones(second_dim) * np.nan)
                    for v in vals_nt
                )
            )
            mask = np.any(np.isnan(mu_outs), axis=1)
            use_vals = vals[mask]
            use_vals_nt = vals_nt[mask]
        else:
            mask = np.ones(len(vals), dtype=bool)
            use_vals = vals
            use_vals_nt = vals_nt
            mu_outs = np.zeros((len(vals), second_dim))
            cov_outs = np.zeros((len(vals), second_dim))
        if len(use_vals) > 0:
            use_mu, use_cov = model(
                use_vals, noiseless=noiseless, full_cov=full_cov, **kwargs
            )
            mu_outs[mask] = use_mu.detach()
            cov_outs[mask] = use_cov.detach()
            for i, um in enumerate(use_mu.detach()):
                avg_vals[tuple(use_vals_nt[i])] = um
                cov_vals[tuple(use_vals_nt[i])] = use_cov[i].detach()
        self.avg_values[key] = avg_vals
        self.cov_values[key] = cov_vals
        return mu_outs, cov_outs

    def get_conditioned_kernel(
        self,
        bin_cents,
        conditions,
        n_samps=500,
        circularize_color=True,
        rescale=True,
        noiseless=True,
    ):
        if not u.check_list(conditions):
            conditions = np.array([conditions])
        if circularize_color:
            use_bin_cents = np.stack((np.sin(bin_cents), np.cos(bin_cents)), axis=1)
            use_conditions = np.stack((np.sin(conditions), np.cos(conditions)), axis=1)
        else:
            use_bin_cents = np.expand_dims(bin_cents, 1)
            use_conditions = np.expand_dims(conditions, 1)
        if noiseless:
            n_samps = 1
        dist_mat = np.zeros((n_samps, len(conditions), len(bin_cents)))
        for k, model in self.model_dict.items():
            mu_b, cov_b = self.get_model_values(
                k,
                model,
                torch.tensor(use_bin_cents),
                noiseless=noiseless,
                full_cov=False,
            )

            mu_c, cov_c = self.get_model_values(
                k,
                model,
                torch.tensor(use_conditions),
                noiseless=noiseless,
                full_cov=False,
            )
            if not noiseless:
                reps_b = np.stack(
                    list(
                        sts.multivariate_normal(mu_b[i], cov_b[i]).rvs(n_samps)
                        for i in range(mu_b.shape[0])
                    ),
                    axis=2,
                )
                reps_c = np.stack(
                    list(
                        sts.multivariate_normal(mu_c[i], cov_c[i]).rvs(n_samps)
                        for i in range(mu_c.shape[0])
                    ),
                    axis=-1,
                )
                if len(reps_c.shape) == 2:
                    reps_c = np.expand_dims(reps_c, 1)
                use_cov = 0
            else:
                use_cov = np.sum(
                    np.expand_dims(cov_b, 1) + np.expand_dims(cov_c, 0), axis=-1
                ).T

                reps_b = np.expand_dims(mu_b, 0)
                reps_c = np.expand_dims(mu_c, 0)

            for i in range(reps_b.shape[0]):
                dist_i = skmp.euclidean_distances(reps_c[i], reps_b[i]) ** 2
                dist_mat[i] += dist_i + use_cov

        full_kernel = -np.sqrt(dist_mat)
        if rescale:
            new_kernel = np.zeros_like(full_kernel)
            for i in range(full_kernel.shape[0]):
                scaler = skp.StandardScaler()
                new_kernel[i] = scaler.fit_transform(full_kernel[i].T).T
            full_kernel = new_kernel
        return bin_cents, full_kernel


def sample_gp_kernel(
    model_dict,
    bin_cents=None,
    n_bins=20,
    noiseless=True,
    n_samps=100,
    circularize_color=True,
    rescale=True,
):
    if bin_cents is None:
        offset = np.pi / n_bins
        bin_cents = np.linspace(-np.pi + offset, np.pi - offset, n_bins)
    if circularize_color:
        use_bin_cents = np.stack((np.sin(bin_cents), np.cos(bin_cents)), axis=1)
    else:
        use_bin_cents = np.expand_dims(bin_cents, 1)
    if noiseless:
        n_samps = 1
    dist_mat = np.zeros((n_samps, len(bin_cents), len(bin_cents)))
    for k, model in model_dict.items():
        mu, cov = model(torch.tensor(use_bin_cents), noiseless=noiseless)
        mu = mu.detach().numpy()
        cov = cov.detach().numpy()
        if not noiseless:
            reps1 = np.stack(
                list(
                    sts.multivariate_normal(mu[i], cov[i]).rvs(n_samps)
                    for i in range(mu.shape[0])
                ),
                axis=2,
            )
            reps2 = np.stack(
                list(
                    sts.multivariate_normal(mu[i], cov[i]).rvs(n_samps)
                    for i in range(mu.shape[0])
                ),
                axis=2,
            )
        else:
            cov = cov.T
            use_cov = np.sum(np.expand_dims(cov, 1) + np.expand_dims(cov, 0), axis=-1)
            use_cov = np.mean(use_cov)

            reps1 = np.expand_dims(mu.T, 0)
            reps2 = np.expand_dims(mu.T, 0)
        for i in range(reps1.shape[0]):
            dist_i = skmp.euclidean_distances(reps1[i], reps2[i]) ** 2
            dist_mat[i] += dist_i + use_cov
    full_kernel = np.sqrt(dist_mat)
    avg_kernel = _compute_average_kernel(
        bin_cents, np.mean(full_kernel, axis=0), rescale=rescale
    )
    if rescale:
        new_kernel = np.zeros_like(full_kernel)
        for i in range(full_kernel.shape[0]):
            scaler = skp.MinMaxScaler()
            new_kernel[i] = scaler.fit_transform(-full_kernel[i].T).T
        full_kernel = new_kernel
    return bin_cents, full_kernel, avg_kernel, dist_mat


def _compute_average_kernel(bins, full_kernel, rescale=True):
    c_diffs = u.normalize_periodic_range(
        np.expand_dims(bins, 1) - np.expand_dims(bins, 0)
    )

    c_diffs = np.round(c_diffs, decimals=4)
    uc_all = np.unique(c_diffs)
    kernel = np.zeros(len(uc_all))
    for i, uc in enumerate(uc_all):
        kernel[i] = np.mean(full_kernel[c_diffs == uc])
    if rescale:
        scaler = skp.MinMaxScaler()
        kernel = np.squeeze(scaler.fit_transform(np.expand_dims(-kernel, 1)))
    return kernel


def make_kernel_map_tc(pickles, xs, *args, n_bins=5, two_dims=True, **kwargs):
    n_sessions = len(pickles)
    n_ts = len(xs)
    if two_dims:
        out_arr = np.zeros((n_sessions, n_bins, n_bins, n_ts))
    else:
        out_arr = np.zeros((n_sessions, n_bins, n_ts))
    for i, x_i in enumerate(xs):
        out_arr[..., i], bins = make_kernel_map(
            pickles, xs, *args, n_bins=n_bins, **kwargs, two_dims=two_dims, x_targ=x_i
        )
    return out_arr, bins


class AverageKernelMap:
    def __init__(self, pickles, xs, p2=None, **kwargs):
        if p2 is None:
            p2 = {}
        self.kernels = list(
            CrossKernelMap(pickle, xs, p2=p2.get(k), **kwargs)
            for k, pickle in pickles.items()
        )

    def get_kernel(self, **kwargs):
        kern_out = []
        for k in self.kernels:
            k_i, bins = k.get_kernel(**kwargs)
            kern_out.append(k_i)
        kern_out = np.stack(kern_out, axis=0)
        return kern_out, bins

    def get_resampled_kernel(self, *args, **kwargs):
        kern_out = []
        for k in self.kernels:
            k_i, bins = k.get_resampled_kernel(*args, **kwargs)
            kern_out.append(k_i)
        kern_out = np.stack(kern_out, axis=0)
        return kern_out, bins


class AGPRegression(gpt.models.ApproximateGP):
    def __init__(self, inducing_points, periodic=False):
        if periodic:
            inducing_points = self._make_periodic(inducing_points)
        self.periodic = periodic
        variational_distribution = gpt.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpt.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(AGPRegression, self).__init__(variational_strategy)
        self.mean_module = gpt.means.ConstantMean()
        self.covar_module = gpt.kernels.ScaleKernel(gpt.kernels.RBFKernel())

    def _make_periodic(self, x):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, -1)
        x = list(
            torch.stack((torch.cos(x[:, i]), torch.sin(x[:, i])), axis=1)
            for i in range(x.shape[1])
        )
        return torch.concatenate(x, axis=1)

    def forward(self, x):
        if self.periodic:
            x = self._make_periodic(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, **kwargs):
        if self.periodic:
            x = self._make_periodic(x)
        return super().__call__(x, **kwargs)


def _fit_svgpr(
    X,
    y,
    x_preds,
    smoke_test=False,
    batch_size=2000,
    num_epochs=8,
    num_inducing=500,
    lr=0.01,
    return_model=False,
    **kwargs,
):
    X_nan_mask = np.isnan(X)
    if len(X_nan_mask.shape) > 1:
        X_nan_mask = np.sum(X_nan_mask, axis=1) > 0
    y_nan_mask = np.isnan(y)
    nan_mask = np.logical_not(np.logical_or(X_nan_mask, y_nan_mask))

    X = X[nan_mask]
    y = y[nan_mask]
    rng = np.random.default_rng()
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    x_preds = torch.tensor(x_preds, dtype=torch.float)
    if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()
    inds = rng.choice(len(X), num_inducing, axis=0)
    pts = X[inds]
    model = AGPRegression(pts, periodic=True)
    likelihood = gpt.likelihoods.GaussianLikelihood()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    num_epochs = 1 if smoke_test else num_epochs

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
    )

    mll = gpt.mlls.VariationalELBO(likelihood, model, num_data=y.size(0))
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model(x_preds).mean.cpu().numpy()
    out = preds
    if return_model:
        out = (preds, model)
    return out


def fit_full_average_kernel(
    neurs,
    xs,
    target_x=-0.25,
    p_thr=0.4,
    num_epochs=15,
    n_bins=20,
):
    out_all = compute_continuous_distance_matrix(
        neurs,
        xs,
        x_targ=target_x,
    )
    kern_models = []
    for out in out_all.values():
        mask = out["ps"][:, 0] > p_thr
        dists = out["dists"][mask][:, mask]
        c1 = out["c1"][mask][:, mask]
        c2 = out["c2"][mask][:, mask]

        d_flat = dists.flatten()
        c1_flat = c1.flatten()
        c2_flat = c2.flatten()
        d_flat, c1_flat, c2_flat = u.filter_nan(d_flat, c1_flat, c2_flat, ind=0)
        cs = np.stack((c1_flat, c2_flat), axis=1)
        c_pred = np.linspace(-np.pi, np.pi, n_bins)
        c1s_pred, c2s_pred = np.meshgrid(c_pred, c_pred)
        cs_pred = np.stack((c1s_pred.flatten(), c2s_pred.flatten()), axis=1)
        _, kern_model_i = _fit_svgpr(
            cs, d_flat, cs_pred, num_epochs=num_epochs, return_model=True
        )
        kern_models.append(kern_model_i)

    def func(X, mean=False):
        preds = []
        X = torch.tensor(X, dtype=torch.float)
        with torch.no_grad():
            for model in kern_models:
                preds.append(model(X).mean.cpu().numpy())
        out = np.stack(preds, axis=0)
        if mean:
            out = np.mean(out, axis=0)
        return out
    return func


class KernelMap:
    def __init__(self, pickle, xs, ps_thr=0.6, bin_range=(-np.pi, np.pi), n_bins=5):
        self.pickle = pickle
        self.xs = xs
        self.ps_thr = ps_thr
        self.bin_range = bin_range
        self.n_bins = n_bins

    def compute_distance_matrix(self, **kwargs):
        info = compute_continuous_distance_matrix(
            {0: self.pickle},
            self.xs,
            **kwargs,
        )[0]
        return info

    def _compute_kernel(
        self,
        dists,
        c1,
        c2,
        row_mask,
        col_mask,
        two_dims=False,
        use_gp=True,
    ):
        dist_mat = dists[row_mask][:, col_mask].flatten()
        c1_mat = u.normalize_periodic_range(
            c1[row_mask][:, col_mask].flatten(),
        )
        c2_mat = u.normalize_periodic_range(
            c2[row_mask][:, col_mask].flatten(),
        )
        mask = ~np.isnan(dist_mat)
        dist_mat = dist_mat[mask]
        c1_mat = c1_mat[mask]
        c2_mat = c2_mat[mask]

        if two_dims:
            c_group = np.stack((c1_mat, c2_mat), axis=1)
        else:
            c_group = np.expand_dims(u.normalize_periodic_range(c2_mat - c1_mat), 1)
        arr_raw, bins = np.histogramdd(
            c_group,
            bins=self.n_bins,
            range=(self.bin_range,) * c_group.shape[1],
            weights=dist_mat,
        )
        arr_count, bins = np.histogramdd(
            c_group,
            bins=self.n_bins,
            range=(self.bin_range,) * c_group.shape[1],
        )
        arr = arr_raw / arr_count
        bins = np.array(list(b[:-1] + np.diff(b)[0] / 2 for b in bins))
        if use_gp:
            arr = _fit_svgpr(c_group, dist_mat, bins.T)
        if not two_dims:
            bins = bins[0]
        return arr, bins

    def _kernel_masks(
        self,
        dist_info,
        row_ind=0,
        col_ind=0,
        cue_only=None,
        col_cue=None,
        row_cue=None,
    ):
        if np.all(np.isnan(dist_info["ps"])):
            ps = np.zeros_like(dist_info["ps"])
            ps[:, 0] = 1
        else:
            ps = dist_info["ps"]
        row_mask = ps[:, row_ind] > self.ps_thr
        col_mask = ps[:, col_ind] > self.ps_thr
        if cue_only is not None:
            add_mask = dist_info["cues"] == cue_only
            row_mask = np.logical_and(row_mask, add_mask)
            col_mask = np.logical_and(col_mask, add_mask)
        if col_cue is not None:
            add_mask = dist_info["cues"] == col_cue
            col_mask = np.logical_and(col_mask, add_mask)
        if row_cue is not None:
            add_mask = dist_info["cues"] == row_cue
            row_mask = np.logical_and(row_mask, add_mask)
        return row_mask, col_mask

    def get_kernel(
        self,
        row_ind=0,
        col_ind=0,
        two_dims=False,
        cue_only=None,
        col_cue=None,
        row_cue=None,
        use_gp=True,
        **kwargs,
    ):
        dist_info = self.compute_distance_matrix(**kwargs)
        dists_all = dist_info["dists"]
        c1s = dist_info["c1"]
        c2s = dist_info["c2"]
        row_mask, col_mask = self._kernel_masks(
            dist_info,
            row_ind=row_ind,
            col_ind=col_ind,
            cue_only=cue_only,
            col_cue=col_cue,
            row_cue=row_cue,
        )
        return self._compute_kernel(
            dists_all,
            c1s,
            c2s,
            row_mask,
            col_mask,
            two_dims=two_dims,
            use_gp=use_gp,
        )

    def get_resampled_kernel(
        self,
        extra_key,
        row_ind=0,
        col_ind=0,
        cue_only=None,
        n_resamples=100,
        resample_bins=5,
        two_dims=False,
        **kwargs,
    ):
        dist_info = self.compute_distance_matrix(extra_key=extra_key, **kwargs)
        dists_all = dist_info["dists"]
        c1s = dist_info["c1"]
        c2s = dist_info["c2"]
        row_mask, col_mask = self._kernel_masks(
            dist_info, row_ind=row_ind, col_ind=col_ind, cue_only=cue_only
        )
        sample_x = dist_info["extra"]
        amts, x, sample_x_binned = sts.binned_statistic(
            sample_x,
            np.ones_like(sample_x),
            statistic="sum",
            bins=resample_bins,
        )
        inds_all = np.arange(len(dists_all)).reshape((-1, 1))
        kerns = []
        for i in range(n_resamples):
            rus = imb_us.RandomUnderSampler()
            inds_rs, _ = rus.fit_resample(inds_all, sample_x_binned)
            inds_rs = np.squeeze(inds_rs)
            dists_rs, c1_rs, c2_rs = _index_matrix(inds_rs, dists_all, c1s, c2s)
            dists_rs = dists_all[inds_rs][:, inds_rs]
            rm_rs = row_mask[inds_rs]
            cm_rs = col_mask[inds_rs]
            kern_i, bin_cents = self._compute_kernel(
                dists_rs,
                c1_rs,
                c2_rs,
                rm_rs,
                cm_rs,
                two_dims=two_dims,
            )
            kerns.append(kern_i)
        kerns = np.stack(kerns, axis=0)
        return kerns, bin_cents


def _index_matrix(inds, *mats):
    return list(x[inds][:, inds] for x in mats)


class CrossKernelMap(KernelMap):
    def __init__(
        self,
        p1,
        xs1,
        p2=None,
        xs2=None,
        ps_thr=0.6,
        bin_range=(-np.pi, np.pi),
        n_bins=5,
    ):
        self.p1 = p1
        self.xs1 = xs1
        if p2 is None:
            p2 = p1
        if xs2 is None:
            xs2 = xs1
        self.p2 = p2
        self.xs2 = xs2
        self.ps_thr = ps_thr
        self.bin_range = bin_range
        self.n_bins = n_bins

    def compute_distance_matrix(self, **kwargs):
        info = compute_continuous_distance_matrix(
            {0: self.p1},
            self.xs1,
            data_dict2={0: self.p2},
            xs2=self.xs2,
            **kwargs,
        )[0]
        return info


def make_cued_kernel_map_tc(
    pickles,
    xs,
    n_bins=5,
    two_dims=False,
    **kwargs,
):
    if two_dims:
        shape = (len(pickles), n_bins, n_bins, len(xs))
    else:
        shape = (len(pickles), n_bins, len(xs))
    kern_tc = np.zeros(shape)
    for i, x in enumerate(xs):
        kern_tc[..., i], bs = make_cued_kernel_map(
            pickles, xs, x, n_bins=n_bins, two_dims=two_dims, **kwargs
        )
    return kern_tc, bs


def make_cued_kernel_map(
    pickles,
    xs,
    x_t,
    p_thr=0.3,
    row_ind=0,
    col_ind=0,
    n_bins=5,
    bin_range=(-np.pi, np.pi),
    two_dims=False,
    same_cue=False,
    **kwargs,
):
    akm = AverageKernelMap(
        pickles, xs, n_bins=n_bins, ps_thr=p_thr, bin_range=bin_range
    )
    if same_cue:
        upper_row_cue = 1
        lower_row_cue = 0
    else:
        upper_row_cue = None
        lower_row_cue = None
    kern_upper, bs = akm.get_kernel(
        x_targ=x_t,
        row_ind=row_ind,
        col_ind=col_ind,
        color_key="uc",
        two_dims=two_dims,
        col_cue=1,
        row_cue=upper_row_cue,
        **kwargs,
    )
    kern_lower, bs = akm.get_kernel(
        x_targ=x_t,
        row_ind=row_ind,
        col_ind=col_ind,
        color_key="lc",
        two_dims=two_dims,
        col_cue=0,
        row_cue=lower_row_cue,
        **kwargs,
    )
    kern_comb = np.stack((kern_upper, kern_lower), axis=0)
    kern = np.mean(kern_comb, axis=0)
    return kern, bs


def make_kernel_map(
    pickles,
    xs,
    c1,
    c2=None,
    p2=None,
    xs2=None,
    n_bins=5,
    p_thr=0.3,
    row_ind=0,  # doesn't matter which one changes, yields same result
    col_ind=0,
    cue_only=None,
    bin_range=(-np.pi, np.pi),
    two_dims=True,
    **kwargs,
):
    akm = AverageKernelMap(
        pickles, xs, p2=p2, xs2=xs2, n_bins=n_bins, ps_thr=p_thr, bin_range=bin_range
    )
    out = akm.get_kernel(
        row_ind=row_ind,
        col_ind=col_ind,
        color_key=c1,
        second_color_key=c2,
        two_dims=two_dims,
        cue_only=cue_only,
        **kwargs,
    )
    return out


def compute_continuous_distance_matrix(
    data_dict,
    xs,
    color_key="rc",
    second_color_key=None,
    data_dict2=None,
    xs2=None,
    x_targ2=None,
    ps_key="ps",
    spk_key="spks",
    cue_key="cues_alt",
    extra_key=None,
    regions=None,
    region_key="regions",
    **kwargs,
):
    out_dict = {}
    for sess, sess_dict in data_dict.items():
        if data_dict2 is not None:
            sess_dict2 = data_dict2[sess]
            if xs2 is None:
                raise IOError(
                    "a second dictionary is supplied but no second xs are given"
                )
        else:
            sess_dict2 = sess_dict
            xs2 = xs
        resps1 = sess_dict[spk_key]
        resps2 = sess_dict2[spk_key]
        if second_color_key is None:
            second_color_key = color_key
        if regions is not None:
            dim_mask1 = np.isin(sess_dict[region_key], regions)
            resps1 = resps1[:, dim_mask1]
            dim_mask2 = np.isin(sess_dict2[region_key], regions)
            resps2 = resps2[:, dim_mask2]
        colors1 = sess_dict[color_key]
        colors2 = sess_dict2[second_color_key]
        ps = sess_dict[ps_key]
        dists, c_diffs, c1, c2 = prepare_data_continuous(
            resps1,
            resps2,
            colors1,
            colors2,
            xs=xs,
            xs2=xs2,
            **kwargs,
        )
        out_dict[sess] = {
            "dists": dists,
            "c_diffs": c_diffs,
            "c1": c1,
            "c2": c2,
            "ps": ps,
            "cues": sess_dict[cue_key],
        }
        if extra_key is not None:
            out_dict[sess]["extra"] = sess_dict[extra_key]
    return out_dict


def session_average_kernel(
    mask_dict, stim_bounds=(-np.pi, np.pi), use_gp=True, n_bins=10, **kwargs
):
    out_dict = {}
    for ind, (rd_sess, cd_sess) in mask_dict.items():
        kernels = np.zeros((len(rd_sess), n_bins))
        for i, rd in enumerate(rd_sess):
            cd = cd_sess[i]
            min_, max_ = stim_bounds
            xs, ys = gpl.digitize_vars(
                cd,
                rd,
                n_bins=n_bins,
                use_max=max_,
                use_min=min_,
                ret_all_y=False,
                cent_func=np.nanmean,
            )
            if use_gp:
                ys = _fit_svgpr(cd, rd, xs.T)
            kernels[i] = ys
        out_dict[ind] = (kernels, xs)
    return out_dict


def compute_continuous_distance_masks(
    dist_dict, p_thr=0.4, average_kernel=True, **kwargs
):
    mask_dict = {}
    for sess, sess_dist_dict in dist_dict.items():
        dists = sess_dist_dict["dists"]
        cds = sess_dist_dict["c_diffs"]
        ps = sess_dist_dict["ps"]
        if np.all(np.isnan(ps)):
            ps = np.ones_like(ps)
        corr_mask = ps[:, 0] > p_thr
        for i in range(ps.shape[1]):
            rd_i, cd_i = mask_dict.get(i, ([], []))
            use_mask = ps[:, i] > p_thr
            m_dists = dists[use_mask][:, corr_mask]
            m_cds = cds[use_mask][:, corr_mask]
            rd_i.append(m_dists.flatten())
            cd_i.append(m_cds.flatten())
            mask_dict[i] = (rd_i, cd_i)
    if average_kernel:
        mask_dict = session_average_kernel(mask_dict, **kwargs)
    else:
        mask_dict = {
            k: (np.concatenate(v1), np.concatenate(v2))
            for k, (v1, v2) in mask_dict.items()
        }
    return mask_dict


def prepare_data_continuous(
    resps1,
    resps2,
    c1,
    c2,
    ps=None,
    ps2=None,
    xs=None,
    xs2=None,
    ps_thr=0.6,
    p_ind=0,
    x_targ=-0.25,
    x2_targ=None,
    regions=None,
    mask_ident=True,
    warn_x_diff=0.1,
    **kwargs,
):
    if x2_targ is None:
        x2_targ = x_targ
    if xs is not None:
        ind = np.argmin((xs - x_targ) ** 2)
        resps1 = resps1[..., ind]
        if np.abs(xs[ind] - x_targ) > warn_x_diff:
            print(
                "large difference in target and actual x value: {}, {}".format(
                    x_targ, xs[ind]
                )
            )
    if xs2 is not None:
        ind = np.argmin((xs2 - x2_targ) ** 2)
        resps2 = resps2[..., ind]
        if np.abs(xs2[ind] - x2_targ) > warn_x_diff:
            print(
                "large difference in target and actual x value: {}, {}".format(
                    x2_targ, xs2[ind]
                )
            )
    if ps2 is None:
        ps2 = ps
    if ps is not None:
        mask = ps[:, p_ind] > ps_thr
        resps1 = resps1[mask]
        c1 = c1[mask]
    if ps2 is not None:
        mask = ps2[:, p_ind] > ps_thr
        resps2 = resps2[mask]
        c2 = c2[mask]
    pipe = na.make_model_pipeline(**kwargs)
    resps_comb = np.concatenate((resps1, resps2), axis=0)
    if len(pipe.steps) > 0:
        pipe.fit(resps_comb)
        resps1 = pipe.transform(resps1)
        resps2 = pipe.transform(resps2)
    dists = resps1 @ resps2.T
    if mask_ident:
        ident_mask = np.identity(len(resps1), dtype=bool)
        dists[ident_mask] = np.nan

    c_diffs = u.normalize_periodic_range(np.expand_dims(c2, 0) - np.expand_dims(c1, 1))
    c1 = np.repeat(np.expand_dims(c1, 1), len(c1), 1)
    c2 = np.repeat(np.expand_dims(c2, 0), len(c2), 0)
    return dists, c_diffs, c1, c2


def prepare_data_discrete(resps, colors, xs=None, x_targ=-0.25, **kwargs):
    if xs is not None:
        ind = np.argmin((xs - x_targ) ** 2)
        resps = resps[..., ind]
    pipe = na.make_model_pipeline(**kwargs)
    resps = pipe.fit_transform(resps)
    dists = skmp.euclidean_distances(resps)

    ident_mask = np.identity(len(resps), dtype=bool)
    dists[ident_mask] = np.nan
    color_diffs = u.normalize_periodic_range(
        np.expand_dims(colors, 0) - np.expand_dims(colors, 1)
    )
    colors = u.normalize_periodic_range(colors)
    col_cents, colors_inds = np.unique(colors, return_inverse=True)
    c1_inds = np.repeat(np.expand_dims(colors_inds, 0), len(colors_inds), 0)
    c2_inds = np.repeat(np.expand_dims(colors_inds, 1), len(colors_inds), 1)

    bin_cents, c_diff_inds = np.unique(color_diffs, return_inverse=True)
    return dists, bin_cents, c_diff_inds, c1_inds, c2_inds
