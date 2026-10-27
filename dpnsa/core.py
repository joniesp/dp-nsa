"""Single DP-NSA clone: a ResNet encoder trained so that the self class falls
inside a set of Mahalanobis ellipsoids and the non-self class falls outside.

score_mode
    'self_only'      s = d(z, P0)              (Eq. 9)
    'discriminative' s = d(z, P0) - d(z, P1)   (headline DP-NSA rows of exp1)
loss_mode
    'paper'          L = L_comp + lambda * L_rep   (Eq. 5)
    'extended'       also compacts the anomaly class around its own ellipsoids
With loss_mode='paper' and lambda_rep=0 the model is strictly one-class.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

import torch.multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass

SEED = 1337

# Can be forced with the environment variable DPNSA_DEVICE.
DEVICE = os.environ.get(
    "DPNSA_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seeds(s=SEED):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Geometry
# =============================================================================
@torch.no_grad()
def kcenter_indices_torch(X: torch.Tensor, k: int, seed=SEED) -> torch.Tensor:
    n = X.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=X.device)
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed)
    centers = [torch.randint(0, n, (1,), generator=gen, device=X.device).item()]
    dists = torch.cdist(X, X[centers[0]].unsqueeze(0)).squeeze(1)
    for _ in range(1, min(k, n)):
        new_c = torch.argmax(dists).item()
        centers.append(new_c)
        d_new = torch.cdist(X, X[new_c].unsqueeze(0)).squeeze(1)
        dists = torch.minimum(dists, d_new)
    return torch.tensor(centers, dtype=torch.long, device=X.device)


def _safe_quantile(t: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile fails above ~16M elements: subsample in that case."""
    if t.numel() == 0:
        return torch.tensor(0.0, device=t.device)
    if t.numel() > 15_000_000:
        idx = torch.randperm(t.numel(), device=t.device)[:1_000_000]
        t = t[idx]
    return torch.quantile(t, q)


@torch.no_grad()
def fit_multi_ellipsoids_optimized(Z, Y, k=1, q=0.95, alpha=0.1, seed=SEED,
                                   fit_anomaly=True):
    """Fit up to k shrunk-covariance ellipsoids per class (P0: self, P1: non-self).

    Each ellipsoid has radius R = q-quantile of the Mahalanobis distances of its
    members. fit_anomaly=False skips P1, which neither loss_mode='paper' nor
    score_mode='self_only' needs.
    """
    Z0_all, Z1_all = Z[Y == 0], Z[Y == 1]

    def fit_class_gmm_torch(Z_class, k_curr):
        components = []
        n_samples = Z_class.size(0)
        if n_samples < 3:
            return components
        k_real = max(1, min(k_curr, n_samples // 3))

        if k_real <= 1:
            labels = torch.zeros(n_samples, device=Z.device, dtype=torch.long)
        else:
            try:
                C = Z_class[kcenter_indices_torch(Z_class, k_real, seed=seed)]
                labels = torch.cdist(Z_class, C).argmin(dim=1)
            except Exception:
                labels = torch.zeros(n_samples, device=Z.device, dtype=torch.long)

        d_dim = Z_class.size(1)
        eye = torch.eye(d_dim, device=Z.device)
        for l in torch.unique(labels):
            Zc = Z_class[labels == l]
            if Zc.size(0) <= 2:
                continue
            S = torch.cov(Zc.T)
            if S.ndim == 0:
                S = S.reshape(1, 1)
            tr = torch.trace(S) / d_dim
            S_reg = (1 - alpha) * S + alpha * tr * eye + 1e-6 * eye
            try:
                Sinv = torch.linalg.inv(S_reg)
            except Exception:
                continue
            mu = Zc.mean(dim=0)
            diff = Zc - mu
            d_maha = torch.sqrt(torch.clamp((diff @ Sinv * diff).sum(1), min=0))
            components.append({'mu': mu, 'Sinv': Sinv, 'R': _safe_quantile(d_maha, q)})
        return components

    P0 = fit_class_gmm_torch(Z0_all, k)
    P1 = fit_class_gmm_torch(Z1_all, k) if fit_anomaly else []
    return {'P0': P0, 'P1': P1}


def dist_to_set_optimized(Z, components):
    """Signed distance to a set of ellipsoids: min over components of d_Mahalanobis - R."""
    if len(components) == 0:
        return torch.zeros(Z.size(0), device=Z.device)
    dists = []
    for comp in components:
        diff = Z - comp['mu']
        d = torch.sqrt(torch.clamp((diff @ comp['Sinv'] * diff).sum(1), min=0))
        dists.append(d - comp['R'])
    return torch.stack(dists, dim=1).min(dim=1)[0]


class FeatureBankEMA:
    """Bank of the most recent latent vectors of one class (memory, Section 3)."""

    def __init__(self, d_lat, max_size=4096, momentum=0.9, device=DEVICE):
        self.Z = torch.empty(0, d_lat, device=device)
        self.m = momentum
        self.max_size = max_size

    @torch.no_grad()
    def update(self, Z_new):
        if Z_new.numel() == 0:
            return
        Z_new = Z_new.detach()
        self.Z = Z_new[-self.max_size:] if self.Z.numel() == 0 \
            else torch.cat([self.Z, Z_new], dim=0)[-self.max_size:]

    def sample(self):
        return self.Z


# =============================================================================
# Model
# =============================================================================
class NSARepV2Ellipsoid:
    """One clone of the repertoire. See the module docstring for the modes."""

    def __init__(self, encoder, k=1, lambda_rep=5.0, use_memory=True,
                 q=0.95, alpha=0.1, epochs=30, bs=1024, lr=1e-3, seed=SEED,
                 device=DEVICE, score_mode='self_only', loss_mode='paper',
                 margin=0.1):
        self.enc = encoder
        self.k = k
        self.lambda_rep = lambda_rep
        self.use_memory = use_memory
        self.q, self.alpha = q, alpha
        self.epochs, self.bs, self.lr = epochs, bs, lr
        self.seed, self.dev = seed, device
        self.score_mode = score_mode
        self.loss_mode = loss_mode
        self.margin = margin
        self.params = None
        self.c = None
        # P1 is only needed if one of the two modes uses it
        self._need_P1 = (score_mode == 'discriminative') or (loss_mode == 'extended')

    # ---------------------------------------------------------------- loss
    def _compute_loss(self, Z, yb, P):
        dev = Z.device
        z0, z1 = Z[yb == 0], Z[yb == 1]
        m = self.margin
        loss = torch.zeros((), device=dev)

        # L_comp: pull self inside P0
        if z0.numel() > 0:
            loss = loss + F.relu(dist_to_set_optimized(z0, P['P0']) + m).mean()

        # L_rep: push non-self outside P0
        if z1.numel() > 0 and self.lambda_rep > 0:
            s1_out = dist_to_set_optimized(z1, P['P0'])
            loss = loss + self.lambda_rep * F.relu(-s1_out + m).mean()

        # 'extended' only: also compact the anomaly class (two-class discriminant)
        if self.loss_mode == 'extended':
            if z0.numel() > 0 and len(P['P1']) > 0:
                loss = loss + F.relu(-dist_to_set_optimized(z0, P['P1']) + m).mean()
            if z1.numel() > 0 and len(P['P1']) > 0:
                loss = loss + F.relu(dist_to_set_optimized(z1, P['P1']) + m).mean()

        return loss

    # ----------------------------------------------------------------- fit
    def fit(self, X, y):
        set_seeds(self.seed)
        dev = torch.device(self.dev)
        self.enc = self.enc.to(dev)
        opt = optim.Adam(self.enc.parameters(), lr=self.lr, weight_decay=1e-4)

        X_t = torch.from_numpy(X).float()
        Y_t = torch.from_numpy(y).long()
        ds = TensorDataset(X_t, Y_t)
        n_samples = len(X)

        # Adaptive batch size, so that small datasets still get several batches
        bs_eff = int(min(self.bs, max(32, n_samples // 4)))
        drop_last = n_samples >= 2 * bs_eff
        num_workers = 2 if n_samples >= 50000 else 0

        # Class-balanced sampling
        classes, counts = np.unique(y, return_counts=True)
        freq = {int(c): counts[i] for i, c in enumerate(classes)}
        w = np.array([1.0 / max(freq.get(int(t), 1), 1) for t in y], dtype=np.float32)
        sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w),
                                        replacement=True)

        dl = DataLoader(ds, batch_size=bs_eff, sampler=sampler, drop_last=drop_last,
                        num_workers=num_workers, pin_memory=(dev.type == 'cuda'),
                        persistent_workers=False)
        assert len(dl) > 0, f"Empty DataLoader: n={n_samples}, bs={bs_eff}"

        bank0 = FeatureBankEMA(self.enc.d_lat, device=dev) if self.use_memory else None
        bank1 = FeatureBankEMA(self.enc.d_lat, device=dev) if self.use_memory else None

        # The ellipsoids are refitted every 20 iterations, on the memory banks
        # (use_memory) or on the current batch.
        P, it = None, 0
        self.enc.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                opt.zero_grad()
                Z = self.enc(xb)

                if self.use_memory:
                    with torch.no_grad():
                        bank0.update(Z[yb == 0])
                        bank1.update(Z[yb == 1])
                        if (P is None) or (it % 20 == 0):
                            Z0b, Z1b = bank0.sample(), bank1.sample()
                            Zb = torch.cat([Z0b, Z1b], dim=0)
                            Yb = torch.cat([
                                torch.zeros(Z0b.size(0), dtype=torch.long, device=dev),
                                torch.ones(Z1b.size(0), dtype=torch.long, device=dev)
                            ])
                            if Zb.size(0) > 10:
                                P = fit_multi_ellipsoids_optimized(
                                    Zb, Yb, k=self.k, q=self.q, alpha=self.alpha,
                                    seed=self.seed, fit_anomaly=self._need_P1)
                else:
                    if ((P is None) or (it % 20 == 0)) and Z.size(0) > 10:
                        with torch.no_grad():
                            P = fit_multi_ellipsoids_optimized(
                                Z.detach(), yb, k=self.k, q=self.q, alpha=self.alpha,
                                seed=self.seed, fit_anomaly=self._need_P1)

                if P is None or len(P['P0']) == 0:
                    loss = ((Z - Z.mean(0)) ** 2).sum(1).mean()   # warm-up
                else:
                    loss = self._compute_loss(Z, yb, P)

                loss.backward()
                opt.step()
                it += 1

        # Final geometry on the whole training set
        with torch.no_grad():
            self.enc.eval()
            Xt = torch.from_numpy(X).float().to(dev)
            yt = torch.from_numpy(y).long().to(dev)
            Z = self.enc(Xt)
            self.params = fit_multi_ellipsoids_optimized(
                Z, yt, k=self.k, q=self.q, alpha=self.alpha, seed=self.seed,
                fit_anomaly=True)   # always fitted, so that both score modes are available
            self.c = self.params['P0'][0]['mu'] if self.params['P0'] else Z.mean(dim=0)

        return self

    # --------------------------------------------------------------- score
    @torch.no_grad()
    def score(self, X, mode=None):
        """Anomaly score (higher = more anomalous)."""
        mode = mode or self.score_mode
        dev = torch.device(self.dev)
        self.enc.eval()
        Xt = torch.from_numpy(X).float()
        dl = DataLoader(TensorDataset(Xt), batch_size=self.bs * 4, shuffle=False,
                        num_workers=(2 if len(X) > 50000 else 0),
                        pin_memory=(dev.type == 'cuda'))
        P = self.params
        scores = []
        for (xb,) in dl:
            Z = self.enc(xb.to(dev, non_blocking=True))
            s0 = dist_to_set_optimized(Z, P['P0'])
            if mode == 'self_only' or len(P['P1']) == 0:
                s = s0
            else:
                s = s0 - dist_to_set_optimized(Z, P['P1'])
            scores.append(s.cpu().numpy())
        return np.concatenate(scores)
