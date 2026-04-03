import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.contrib.cevae as ce
import sklearn.base as skb
import sklearn.model_selection as skms

import general.neural_analysis as na
import general.utility as u
import general.torch.feedforward as gtf
import general.torch.any as gta


class KernelDistribution(dist.Distribution):
    def __init__(self, mu, sigma):
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu)
        self.mu = mu
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma)
        self.sigma = sigma
        self.dist = dist.Normal(self.mu, self.sigma)
        mu_comb = self.mu @ self.mu.T
        sig_comb = self.sigma @ self.sigma.T
        self.comb_dist = dist.Normal(mu_comb.flatten(), sig_comb.flatten())

    def batch_shape(self):
        return self.dist.batch_shape

    def event_shape(self):
        return self.dist.event_shape

    def sample(self, *args, **kwargs):
        samp = self.dist.sample(*args, **kwargs)
        return samp @ samp.T

    def log_prob(self, kern, **kwargs):
        eye = ~torch.eye(len(kern), dtype=bool)
        lps = self.comb_dist.log_prob(kern.flatten(), **kwargs)
        return torch.sum(torch.reshape(lps, (len(kern), len(kern))) * eye, axis=1)


class EigenfunctionModel(pyro.nn.PyroModule):
    def __init__(
        self,
        inp_dim,
        k,
        layer_dims=(500, 100),
        probe_bounds=(-np.pi, np.pi),
        n_probe=1000,
        net_kwargs=None,
    ):
        super().__init__()
        sizes = (inp_dim,) + tuple(layer_dims) + (k,)
        self.func_net = ce.DiagNormalNet(list(sizes))
        self.k = k

    def func_dist(self, x):
        loc, scale = self.func_net(x)
        return dist.Normal(loc, scale).to_event(1)

    def forward(self, x, x_samp=None, size=None):
        if size is None:
            size = len(x)
        if x_samp is not None:
            mu_targ = x_samp @ x_samp.T
        else:
            mu_targ = None
        with pyro.plate("data", size):
            loc, scale = self.func_net(x)
            k_dist = KernelDistribution(loc, scale)
            k = pyro.sample("k", k_dist, obs=mu_targ)
        return k


class EigenfunctionGuide(pyro.nn.PyroModule):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x, x_samp=None, size=None):
        pass


def estimate_n_funcs(
    model,
    X,
    y,
    n_funcs_range=(1, 50),
    resource="num_epochs",
    min_resources=30,
    max_resources=150,
    return_search=False,
    **kwargs,
):
    n_funcs_candidates = np.arange(*n_funcs_range)
    model_gs = skms.HalvingGridSearchCV(
        model(n_funcs_candidates[0], **kwargs),
        {"k": n_funcs_candidates},
        refit=True,
        resource=resource,
        min_resources=min_resources,
        max_resources=max_resources,
    )
    model_gs.fit(X, y)
    if return_search:
        out = model_gs.best_estimator_, model_gs
    else:
        out = model_gs.best_estimator_
    return out


class ProbabilisticEigenfunctionEstimator(skb.BaseEstimator):
    def __init__(
        self,
        k,
        periodic=True,
        layer_dims=(500, 100),
        net_kwargs=None,
        learning_rate=1e-4,
        num_epochs=100,
        batch_size=50,
    ):
        super().__init__()
        self.k = k
        self.periodic = periodic
        self.layer_dims = layer_dims
        if net_kwargs is None:
            net_kwargs = {}
        self.net_kwargs = net_kwargs
        self.use_optimizer = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.fitted = False

    def __sklearn_is_fitted__(self):
        return self.fitted

    @property
    def inp_dim(self):
        return 2 if self.periodic else 1

    def _model_y(self, y):
        if len(y.shape) == 1:
            y = np.expand_dims(y, -1)
        if self.periodic:
            y_use = np.squeeze(u.radian_to_sincos(y))
        else:
            y_use = y
        return y_use

    def t(self, x):
        return torch.tensor(x, dtype=torch.float)

    def n(self, x):
        return x.detach().cpu().numpy()

    def fit(self, rs, color):
        pyro.clear_param_store()
        c = self._model_y(color)
        adam_params = {"lr": self.learning_rate}
        optimizer = pyro.optim.Adam(adam_params)

        self.model = EigenfunctionModel(self.inp_dim, self.k, self.layer_dims)
        self.guide = EigenfunctionGuide(self.k)
        c = self.t(c)
        rs = self.t(rs)
        loader = gta.batch_generator(c, rs, batch_size=self.batch_size)

        loss = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, loss=loss)
        loss_record = np.zeros((self.num_epochs, len(loader)))
        for i in range(self.num_epochs):
            for j, (c_i, rs_i) in enumerate(loader):
                loss = svi.step(c, rs)
                loss_record[i, j] = loss
        fit_record = {
            "loss": loss_record,
        }
        self.fit_record = fit_record
        self.fitted = True
        return self

    def transform(self, color):
        c = self.t(self._model_y(color))
        return self.n(self.model.func_net(c)[0])

    def compute_kernel(self, color):
        r_c = self.transform(color)
        return r_c @ r_c.T

    def score(self, rs, color):
        color = self._model_y(color)
        mu, sig = self.model.func_net(self.t(color))
        k_dist = KernelDistribution(mu, sig)
        r_kernel = self.t(rs @ rs.T)
        return self.n(torch.mean(k_dist.log_prob(r_kernel)))


class NNEigenfunctionEstimator(gta.GenericModule, gta.GenericTrainingLoop):
    def __init__(
        self,
        k,
        lam=0.5,
        periodic=True,
        layer_dims=(500, 100),
        probe_bounds=(-np.pi, np.pi),
        n_probe=1000,
        net_kwargs=None,
        weight_decay=0,
        dropout=0.1,
    ):
        super().__init__()
        self.k = k
        self.periodic = periodic
        self.layer_dims = layer_dims
        self.probe_points = np.linspace(*probe_bounds, n_probe)
        if net_kwargs is None:
            net_kwargs = {}
        self.net_kwargs = net_kwargs
        self.orthonormal_lambda = lam
        self.use_optimizer = None
        self.weight_decay = weight_decay
        self.dropout = dropout

    def _model_y(self, y):
        if len(y.shape) == 1:
            y = np.expand_dims(y, -1)
        if self.periodic:
            y_use = np.squeeze(u.radian_to_sincos(y))
        else:
            y_use = y
        return y_use

    def compute_kernel(self, pts):
        pts = self._model_y(pts)
        pts = self.net(self._setup_outsider(pts))
        k_est = pts * self.pos_trs(self.evs)[None] @ pts.T
        return self._make_numpy(k_est)

    def get_estimated_kernel(self):
        return self.compute_kernel(self.probe_points)

    def _make_func(self, inp_dim):
        eig_net = gtf.make_feedforward_network(
            inp_dim,
            self.layer_dims,
            self.k,
            dtype=torch.float,
            dropout=self.dropout,
            **self.net_kwargs,
        )[0]
        evs = nn.Parameter(
            torch.exp(torch.randn(self.k, dtype=torch.float)).to(self.device)
        )
        return eig_net.to(self.device), evs

    def _make_numpy(self, x):
        return x.detach().cpu().numpy()

    @property
    def eigenvalues(self):
        return self.pos_trs(self.evs)

    def fit(self, X, y, **kwargs):
        """X is N x D, y is N x 1"""
        y_inp = self._model_y(y)
        probe_inp = self._model_y(self.probe_points)
        self.net, self.evs = self._make_func(y_inp.shape[-1])
        self.pos_trs = nn.Softplus()

        return super().fit(
            y_inp,
            X,
            probe_inp=self._setup_outsider(probe_inp),
            orthonormal_lambda=self.orthonormal_lambda,
            optim_kwargs={"weight_decay": self.weight_decay},
            **kwargs,
        )

    def setup_optimizer(self, optim, lr, lr_stepping=False, lr_patience=3, **kwargs):
        use_optimizer = optim(self.parameters(), lr=lr, **kwargs)
        if lr_stepping:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                use_optimizer,
                patience=lr_patience,
            )
        else:
            lr_scheduler = None
        return use_optimizer, lr_scheduler

    def get_loss(self, X, y, loss_func, probe_inp=None, orthonormal_lambda=0):
        y_hat = self.net(X)
        k_targ = y @ y.T
        mask = 1 - torch.eye(len(y), dtype=torch.float).to(self.device)
        k_est = y_hat * self.pos_trs(self.evs)[None] @ y_hat.T
        loss = loss_func(k_est * mask, k_targ * mask) / y.shape[-1] ** 2
        eig = self.net(probe_inp)
        k_eye = torch.eye(self.k, dtype=torch.float).to(self.device)
        on_loss = torch.mean((eig.T @ eig / len(probe_inp) - k_eye) ** 2)
        loss = loss + orthonormal_lambda * on_loss
        return loss


def estimate_average_kernel(
    pops,
    xs,
    x_targ=-0.25,
    p_thr=.3,
    p_ind=0,
    ps_key="ps",
    spk_key="spks",
    color_key="rc",
    n_funcs=8,
    norm=True,
    model=ProbabilisticEigenfunctionEstimator,
    estimate_n=True,
    **kwargs,
):
    t_ind = np.argmin(np.abs(xs - x_targ))
    k_funcs = []
    losses = []
    for pop in pops.values():
        pipe = na.make_model_pipeline(norm=True, pca=None)
        mask = pop[ps_key][:, p_ind] > p_thr
        X = pipe.fit_transform(pop[spk_key][mask, ..., t_ind])
        y = pop[color_key][mask]
        if estimate_n:
            est = estimate_n_funcs(model, X, y, **kwargs)
        else:
            est = model(n_funcs, **kwargs)
            est = est.fit(X, y)
        losses.append(est.fit_record["loss"])
        k_funcs.append(est.compute_kernel)

    def func(pts, mean=False, meanfunc=np.mean):
        ks = np.stack(list(kf(pts) for kf in k_funcs), axis=0)
        if mean:
            ks = meanfunc(ks, axis=0)
        return ks

    return func
