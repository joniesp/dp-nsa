"""Deep baselines of the paper.

Unsupervised: Deep SVDD (soft-boundary), Contrastive AE, Deep MS-SVDD.
Supervised / semi-supervised: TabResNet, Deep SAD, DevNet.

Every wrapper takes a `seed` (applied to random, numpy and torch) and a
`device` (None = CUDA when available). `score` returns an anomaly score
(higher = more anomalous).

Also: effective_rank / spectral_diagnostics, the spectral diagnostics of the
latent covariance reported with exp1 and exp2.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device=None):
    """None -> automatic; an explicit value is respected."""
    if device is None:
        return torch.device(DEVICE)
    return torch.device(device)


# =============================================================================
# 1. Deep SVDD (Ruff et al., ICML 2018), soft-boundary
# =============================================================================
class DeepSVDDNetwork(nn.Module):
    def __init__(self, d_in, d_lat=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Linear(64, d_lat),
        )

    def forward(self, x):
        return self.net(x)


class DeepSVDDWrapper:
    def __init__(self, d_in, d_lat=32, epochs=50, lr=1e-3, bs=1024, nu=0.1,
                 device=None, seed=1337):
        self.d_in, self.d_lat = d_in, d_lat
        self.epochs, self.lr, self.bs, self.nu = epochs, lr, bs, nu
        self.device = _resolve_device(device)
        self.seed = seed
        self.net, self.c, self.R = None, None, None

    def fit(self, X, y=None):
        _set_seed(self.seed)
        dev = self.device
        self.net = DeepSVDDNetwork(self.d_in, self.d_lat).to(dev)

        Xt = torch.tensor(X, dtype=torch.float32).to(dev)
        self.net.eval()
        with torch.no_grad():
            idx = torch.randperm(len(Xt), device=dev)[:min(1000, len(Xt))]
            self.c = self.net(Xt[idx]).mean(dim=0).detach()

        self.R = torch.tensor(0.0, device=dev, requires_grad=True)
        opt = torch.optim.Adam(list(self.net.parameters()) + [self.R],
                               lr=self.lr, weight_decay=1e-6)

        bs = min(self.bs, max(32, len(X) // 4))
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=bs, shuffle=True, drop_last=False)

        self.net.train()
        for _ in range(self.epochs):
            for (bx,) in loader:
                if bx.size(0) < 2:      # BatchNorm needs more than one sample
                    continue
                bx = bx.to(dev)
                opt.zero_grad()
                z = self.net(bx)
                dist = torch.sum((z - self.c) ** 2, dim=1)
                scores = dist - self.R ** 2
                loss = self.R ** 2 + (1 / self.nu) * torch.clamp(scores, min=0).mean()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X):
        self.net.eval()
        dev = self.device
        out = []
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=self.bs, shuffle=False)
        for (bx,) in loader:
            z = self.net(bx.to(dev))
            out.append((torch.sum((z - self.c) ** 2, dim=1) - self.R ** 2).cpu().numpy())
        return np.concatenate(out)


# =============================================================================
# 2. Contrastive autoencoder
# =============================================================================
class ContrastiveAEWrapper:
    def __init__(self, d_in, d_lat=32, epochs=50, lr=1e-3, bs=1024, margin=1.0,
                 device=None, seed=1337):
        self.d_in, self.d_lat = d_in, d_lat
        self.epochs, self.lr, self.bs, self.margin = epochs, lr, bs, margin
        self.device = _resolve_device(device)
        self.seed = seed
        self.model = None

    def fit(self, X, y=None):
        _set_seed(self.seed)
        dev = self.device
        self.model = nn.Sequential(
            nn.Linear(self.d_in, 128), nn.ReLU(),
            nn.Linear(128, self.d_lat), nn.ReLU(),
            nn.Linear(self.d_lat, 128), nn.ReLU(),
            nn.Linear(128, self.d_in),
        ).to(dev)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        bs = min(self.bs, max(32, len(X) // 4))
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=bs, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for (bx,) in loader:
                bx = bx.to(dev)
                b_neg = bx[torch.randperm(bx.size(0), device=dev)]
                loss_mse = F.mse_loss(self.model(bx), bx)
                z_pos = self.model[0:3](bx)
                z_neg = self.model[0:3](b_neg)
                dist_neg = F.pairwise_distance(z_pos, z_neg, p=2)
                loss_con = torch.clamp(self.margin - dist_neg, min=0.0).mean()
                loss = loss_mse + 0.1 * loss_con
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        dev = self.device
        out = []
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=self.bs, shuffle=False)
        for (bx,) in loader:
            bx = bx.to(dev)
            out.append(torch.mean((bx - self.model(bx)) ** 2, dim=1).cpu().numpy())
        return np.concatenate(out)


# =============================================================================
# 3. Supervised tabular ResNet
# =============================================================================
class _SimpleTabResNet(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.head = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.layer1(x))
        h2 = self.relu(self.layer2(h1))
        return self.head(h1 + h2)          # skip connection


class TabResNetWrapper:
    def __init__(self, input_dim, hidden_dim=128, epochs=50, batch_size=256,
                 lr=1e-3, device=None, seed=1337):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.device = _resolve_device(device)
        self.seed = seed
        _set_seed(seed)
        self.model = _SimpleTabResNet(input_dim, hidden_dim).to(self.device)

    def fit(self, X_train, y_train):
        _set_seed(self.seed)
        dev = self.device
        n_neg, n_pos = int(np.sum(y_train == 0)), int(np.sum(y_train == 1))
        pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0]).to(dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        bs = min(self.batch_size, max(16, len(X_train) // 4))
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=bs, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(dev), by.to(dev)
                opt.zero_grad()
                loss = criterion(self.model(bx).squeeze(-1), by)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X_test):
        self.model.eval()
        Xt = torch.FloatTensor(X_test).to(self.device)
        return torch.sigmoid(self.model(Xt)).squeeze(-1).cpu().numpy()

    def score(self, X):
        return self.predict_proba(X)


# =============================================================================
# 4. Deep SAD (Ruff et al., ICLR 2020)
# =============================================================================
class DeepSADWrapper:
    """Deep SAD with the autoencoder pre-training of the original paper
    (pretrain_epochs > 0), which initialises the encoder and the centre c."""

    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, epochs=50,
                 batch_size=256, lr=1e-3, eta=1.0, pretrain_epochs=30,
                 device=None, seed=1337):
        self.epochs, self.batch_size, self.lr, self.eta = epochs, batch_size, lr, eta
        self.pretrain_epochs = pretrain_epochs
        self.device = _resolve_device(device)
        self.seed = seed
        self.input_dim, self.hidden_dim, self.latent_dim = input_dim, hidden_dim, latent_dim
        _set_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        ).to(self.device)
        self.c = None

    def _pretrain(self, loader):
        opt = torch.optim.Adam(list(self.encoder.parameters()) +
                               list(self.decoder.parameters()), lr=self.lr)
        self.encoder.train(); self.decoder.train()
        for _ in range(self.pretrain_epochs):
            for bx, _y in loader:
                bx = bx.to(self.device)
                opt.zero_grad()
                loss = F.mse_loss(self.decoder(self.encoder(bx)), bx)
                loss.backward()
                opt.step()

    @torch.no_grad()
    def _init_center(self, loader, eps=0.1):
        self.encoder.eval()
        zs = [self.encoder(bx.to(self.device)) for bx, _y in loader]
        c = torch.cat(zs, dim=0).mean(dim=0)
        # keep the coordinates of c away from 0, as in the original, to avoid a trivial collapse
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.c = c.detach()

    def fit(self, X_train, y_train):
        _set_seed(self.seed)
        bs = min(self.batch_size, max(16, len(X_train) // 4))
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=bs, shuffle=True)

        if self.pretrain_epochs > 0:
            self._pretrain(loader)
        self._init_center(loader)

        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.encoder.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                z = self.encoder(bx)
                dist = torch.sum((z - self.c) ** 2, dim=1)
                loss_n = torch.where(by == 0, dist, torch.zeros_like(dist))
                loss_a = torch.where(by == 1,
                                     self.eta * ((dist + 1e-6) ** -1),
                                     torch.zeros_like(dist))
                loss = torch.mean(loss_n + loss_a)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X_test):
        self.encoder.eval()
        Xt = torch.FloatTensor(X_test).to(self.device)
        return torch.sum((self.encoder(Xt) - self.c) ** 2, dim=1).cpu().numpy()

    def predict_proba(self, X_test):   # alias of score (a distance, not a probability)
        return self.score(X_test)


# =============================================================================
# 5. DevNet (Pang et al., KDD 2019)
# =============================================================================
class DevNetWrapper:
    """DevNet with the deviation loss against a standard Gaussian reference:
        dev(x) = (phi(x) - mu_R) / sigma_R
        L = (1 - y) |dev| + y * max(0, a - dev)
    """

    def __init__(self, input_dim, hidden_dim=128, epochs=50, batch_size=256,
                 lr=1e-3, confidence_margin=5.0, n_ref=5000,
                 device=None, seed=1337):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.a, self.n_ref = confidence_margin, n_ref
        self.device = _resolve_device(device)
        self.seed = seed
        _set_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def fit(self, X_train, y_train):
        _set_seed(self.seed)
        dev = self.device
        opt = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        bs = min(self.batch_size, max(16, len(X_train) // 4))
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=bs, shuffle=True)

        self.network.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(dev), by.to(dev)
                opt.zero_grad()
                phi = self.network(bx).squeeze(-1)
                ref = torch.randn(self.n_ref, device=dev)
                dev_score = (phi - ref.mean()) / (ref.std() + 1e-6)
                loss = torch.mean(
                    (1 - by) * torch.abs(dev_score) +
                    by * torch.clamp(self.a - dev_score, min=0)
                )
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X_test):
        self.network.eval()
        Xt = torch.FloatTensor(X_test).to(self.device)
        return self.network(Xt).squeeze(-1).cpu().numpy()

    def predict_proba(self, X_test):   # alias of score
        return self.score(X_test)


# =============================================================================
# 6. Deep Multi-Sphere SVDD (Ghafoori and Leckie, SDM 2020)
# =============================================================================
class DeepMSSVDDWrapper:
    """Deep SVDD with K hyperspheres instead of one. Unsupervised: it uses no
    anomaly labels, so it is the counterpart of the lambda=0 variant of DP-NSA.

    1. Autoencoder pre-training, as in Deep SVDD and Deep SAD.
    2. K centres by k-means on the latent projection of the normal data.
    3. Minimise the squared distance to the nearest centre; the centres are
       re-placed every 10 epochs, since the encoder moves.
    4. Spheres with fewer than prune_frac of the samples are pruned.
    5. Score: squared distance to the nearest centre.
    """

    def __init__(self, input_dim, n_spheres=5, hidden_dim=128, latent_dim=32,
                 epochs=50, pretrain_epochs=30, batch_size=1024, lr=1e-3,
                 prune_frac=0.01, device=None, seed=1337):
        self.d_in = input_dim
        self.K = n_spheres
        self.h = hidden_dim
        self.d_lat = latent_dim
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.bs = batch_size
        self.lr = lr
        self.prune_frac = prune_frac
        self.seed = seed
        self.device = _resolve_device(device)

        _set_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        ).to(self.device)
        self.centers = None

    # ------------------------------------------------------------------
    def _pretrain(self, loader):
        opt = torch.optim.Adam(list(self.encoder.parameters()) +
                               list(self.decoder.parameters()), lr=self.lr)
        self.encoder.train(); self.decoder.train()
        for _ in range(self.pretrain_epochs):
            for (bx,) in loader:
                bx = bx.to(self.device)
                opt.zero_grad()
                F.mse_loss(self.decoder(self.encoder(bx)), bx).backward()
                opt.step()

    @torch.no_grad()
    def _init_centers(self, X):
        from sklearn.cluster import KMeans
        self.encoder.eval()
        Z = self.encoder(torch.tensor(X, dtype=torch.float32).to(self.device))
        Zc = Z.cpu().numpy()
        k = max(1, min(self.K, len(Zc) // 3))
        km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(Zc)
        self.centers = torch.tensor(km.cluster_centers_, dtype=torch.float32,
                                    device=self.device)

    @torch.no_grad()
    def _prune(self, X):
        if self.centers.shape[0] <= 1:
            return
        self.encoder.eval()
        Z = self.encoder(torch.tensor(X, dtype=torch.float32).to(self.device))
        assign = torch.cdist(Z, self.centers).argmin(dim=1)
        counts = torch.bincount(assign, minlength=self.centers.shape[0])
        keep = counts >= max(3, int(self.prune_frac * len(Z)))
        if keep.sum() >= 1 and keep.sum() < self.centers.shape[0]:
            self.centers = self.centers[keep]

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """If y is given it is only used to keep the normal samples, as in Deep SVDD."""
        _set_seed(self.seed)
        Xn = X[y == 0] if y is not None else X

        bs = min(self.bs, max(32, len(Xn) // 4))
        loader = DataLoader(TensorDataset(torch.tensor(Xn, dtype=torch.float32)),
                            batch_size=bs, shuffle=True)

        if self.pretrain_epochs > 0:
            self._pretrain(loader)
        self._init_centers(Xn)

        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.encoder.train()
        for ep in range(self.epochs):
            for (bx,) in loader:
                bx = bx.to(self.device)
                opt.zero_grad()
                z = self.encoder(bx)
                d2 = torch.cdist(z, self.centers) ** 2
                d2.min(dim=1)[0].mean().backward()
                opt.step()
            if (ep + 1) % 10 == 0:
                self._init_centers(Xn)
                self._prune(Xn)
        self._prune(Xn)
        return self

    @torch.no_grad()
    def score(self, X):
        self.encoder.eval()
        out = []
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=self.bs, shuffle=False)
        for (bx,) in loader:
            z = self.encoder(bx.to(self.device))
            d2 = torch.cdist(z, self.centers) ** 2
            out.append(d2.min(dim=1)[0].cpu().numpy())
        return np.concatenate(out)

    @property
    def n_spheres_final(self):
        """Spheres left after pruning (recorded in the results CSV)."""
        return 0 if self.centers is None else int(self.centers.shape[0])


# =============================================================================
# Spectral diagnostics of the latent covariance
# =============================================================================
def effective_rank(S, eps=1e-12):
    """Participation ratio of the spectrum, r_eff = (sum l_i)^2 / sum l_i^2, in [1, d].

    It equals d when the variance is spread evenly and tends to 1 when it
    concentrates in one direction; unlike counting eigenvalues above a
    tolerance, it needs no threshold and is continuous.
    """
    if isinstance(S, torch.Tensor):
        ev = torch.linalg.eigvalsh(S.double()).clamp(min=0).cpu().numpy()
    else:
        ev = np.linalg.eigvalsh(np.asarray(S, dtype=np.float64)).clip(min=0)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 ** 2 / s2) if s2 > eps else 1.0


def spectral_diagnostics(ensemble):
    """Effective rank of the self covariances of a fitted repertoire.

    Returns r_eff_mean, r_eff_min, d_lat, r_ratio = r_eff_mean / d_lat and
    cond_mean (mean condition number). A low r_ratio means the projection
    leaves most latent directions without variance, so the covariance is
    singular by construction rather than for lack of samples.
    """
    r_eff, conds, d_lat = [], [], None
    for m in getattr(ensemble, "models", [ensemble]):
        comps = getattr(m, "params", {}).get("P0", [])
        for c in comps:
            Sinv = c["Sinv"]
            d_lat = Sinv.shape[0]
            try:
                S = torch.linalg.inv(Sinv.double())
                r_eff.append(effective_rank(S))
                ev = torch.linalg.eigvalsh(S).clamp(min=1e-30)
                conds.append(float(ev.max() / ev.min()))
            except Exception:
                continue
    if not r_eff:
        return dict(r_eff_mean=np.nan, r_eff_min=np.nan, d_lat=np.nan,
                    r_ratio=np.nan, cond_mean=np.nan)
    return dict(
        r_eff_mean=float(np.mean(r_eff)),
        r_eff_min=float(np.min(r_eff)),
        d_lat=int(d_lat),
        r_ratio=float(np.mean(r_eff) / d_lat),
        cond_mean=float(np.mean(conds)) if conds else np.nan,
    )
