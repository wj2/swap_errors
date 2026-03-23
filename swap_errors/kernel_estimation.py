import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.contrib.cevae as ce

import general.neural_analysis as na
import general.utility as u
import general.torch.feedforward as gtf
import general.torch.any as gta


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
            size = x.shape[0] * (x.shape[0] - 1)
        eye = ~torch.eye(len(x), dtype=torch.bool)
        if x_samp is not None:
            mu_targ = x_samp @ x_samp.T
            mu_targ = mu_targ[eye]
        else:
            mu_targ = None
        ev_rate = pyro.sample("ev_rate", dist.HalfNormal(1))
        ev_dist = dist.HalfNormal(ev_rate * torch.ones(self.k)).to_event(1)
        evs = pyro.sample(
            "evs",
            ev_dist,
        )
        with pyro.plate("data", size):
            loc, scale = self.func_net(x)
            diag = torch.diag(evs)
            mu = loc @ diag @ loc.T
            scales = scale @ scale.T

            mu = mu[eye]
            scales = scales[eye]

            k = pyro.sample("k", dist.Normal(mu, scales), obs=mu_targ)
        return k


class EigenfunctionGuide(pyro.nn.PyroModule):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x, x_samp=None, size=None):
        evr_loc = pyro.param(
            "evr_loc",
            torch.tensor(1.0),
            constraint=pyro.distributions.constraints.positive,
        )
        evr_scale = pyro.param(
            "evr_scale",
            torch.tensor(.05),
            constraint=pyro.distributions.constraints.positive,
        )
        pyro.sample("ev_rate", dist.Normal(evr_loc, evr_scale))

        evs_loc = pyro.param(
            "evs_loc",
            torch.ones(self.k),
            constraint=pyro.distributions.constraints.positive,
        )
        evs_scale = pyro.param(
            "evs_scale",
            torch.ones(self.k) * .05,
            constraint=pyro.distributions.constraints.positive,
        )
        pyro.sample("evs", dist.Normal(evs_loc, evs_scale).to_event(1))


class ProbabilisticEigenfunctionEstimator:
    def __init__(
        self,
        k,
        periodic=True,
        layer_dims=(500, 100),
        probe_bounds=(-np.pi, np.pi),
        n_probe=1000,
        net_kwargs=None,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.k = k
        self.periodic = periodic
        self.layer_dims = layer_dims
        self.probe_points = np.linspace(*probe_bounds, n_probe)
        if net_kwargs is None:
            net_kwargs = {}
        self.net_kwargs = net_kwargs
        self.use_optimizer = None
        self.learning_rate = learning_rate

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

    def fit(self, color, rs, n_epochs=100):
        pyro.clear_param_store()
        c = self._model_y(color)
        adam_params = {"lr": self.learning_rate}
        optimizer = pyro.optim.Adam(adam_params)

        self.model = EigenfunctionModel(self.inp_dim, self.k, self.layer_dims)
        self.guide = EigenfunctionGuide(self.k)
        c = self.t(c)
        rs = self.t(rs)
        loader = gta.batch_generator(c, rs)

        loss = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, loss=loss)
        loss_record = np.zeros((n_epochs, len(loader)))
        for i in range(n_epochs):
            for j, (c_i, rs_i) in enumerate(loader):
                loss = svi.step(c, rs)
                loss_record[i, j] = loss
        return loss_record

    def transform(self, color):
        c = self.t(self._model_y(color))
        return self.n(self.model.func_net(c)[0])


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
    spk_key="spks",
    color_key="c_targ",
    n_funcs=8,
    weight_decay=1e-10,
    norm=True,
    **kwargs,
):
    t_ind = np.argmin(np.abs(xs - x_targ))
    k_funcs = []
    losses = []
    for pop in pops.values():
        pipe = na.make_model_pipeline(norm=True, pca=None)
        X = pipe.fit_transform(pops[0]["spks"][..., t_ind])
        y = pops[0]["c_targ"]
        m = NNEigenfunctionEstimator(n_funcs, weight_decay=weight_decay)
        out_p = m.fit(X, y, **kwargs)
        losses.append(out_p["loss"])
        k_funcs.append(m.compute_kernel)

    def func(pts, mean=False, meanfunc=np.mean):
        ks = np.stack(list(kf(pts) for kf in k_funcs), axis=0)
        if mean:
            ks = meanfunc(ks, axis=0)
        return ks

    return func
