"""DP-NSA: the polyclonal repertoire (Section 3 of the paper).

M independent clones (NSARepV2Ellipsoid), each with its own seed and,
optionally, its own number of ellipsoids k; the anomaly score is the mean of
the clone scores (Eq. 8).

    NSARepV23Ensemble(d_in, n_estimators=5, k=3)              # same k in every clone
    NSARepV23Ensemble(d_in, n_estimators=5, k=[1, 3, 5, 1, 3])  # one k per clone
"""

import numpy as np

from dpnsa.core import (
    NSARepV2Ellipsoid, set_seeds, DEVICE
)
from dpnsa.encoders import ResNetTabularEncoder


class NSARepV23Ensemble:
    """
    Parameters
    ----------
    k : int | list[int]
        int        -> same number of ellipsoids in every clone.
        list/tuple -> one k per clone; cycled if shorter than n_estimators.
    score_mode : {'self_only', 'discriminative'}
        'self_only'      -> s = d(z, P0)            (Eq. 9)
        'discriminative' -> s = d(z, P0) - d(z, P1)
    loss_mode : {'paper', 'extended'}
        'paper'    -> L = L_comp + lambda * L_rep   (Eq. 5)
        'extended' -> also compacts the anomaly class
    device : defaults to CUDA when available (or $DPNSA_DEVICE).
    """

    def __init__(self, d_in, d_lat=32,
                 n_estimators=5,
                 k=3,
                 lambda_rep=5.0,
                 use_memory=True,
                 alpha=0.1,
                 verbose=False,
                 n_blocks=4, d_hidden=128, dropout=0.2,
                 epochs=30, q=0.95, lr=1e-3, bs=1024,
                 seed_start=1337,
                 score_mode='self_only',
                 loss_mode='paper',
                 device=None):

        self.n_estimators = n_estimators
        self.seed_start = seed_start
        self.verbose = verbose
        self.score_mode = score_mode
        self.loss_mode = loss_mode
        self.device = device or DEVICE

        if isinstance(k, (list, tuple, np.ndarray)):
            if len(k) == 0:
                raise ValueError("k cannot be an empty list.")
            self.k_per_clone = [int(k[i % len(k)]) for i in range(n_estimators)]
            self.k_heterogeneous = len(set(self.k_per_clone)) > 1
        else:
            self.k_per_clone = [int(k)] * n_estimators
            self.k_heterogeneous = False

        self.model_args = {
            "d_in": d_in, "d_lat": d_lat,
            "lambda_rep": lambda_rep, "use_memory": use_memory, "alpha": alpha,
            "n_blocks": n_blocks, "d_hidden": d_hidden, "dropout": dropout,
            "epochs": epochs, "q": q, "lr": lr, "bs": bs,
        }
        self.models = []

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        """X: float32 array (N, d_in); y: 0 = self (normal), 1 = non-self (anomaly)."""
        self.models = []
        for i in range(self.n_estimators):
            seed = self.seed_start + i
            k_i = self.k_per_clone[i]

            if self.verbose:
                print(f"  [Clone {i+1}/{self.n_estimators}] seed={seed} k={k_i} "
                      f"device={self.device} score={self.score_mode} "
                      f"loss={self.loss_mode}")

            set_seeds(seed)
            encoder = ResNetTabularEncoder(
                d_in=self.model_args["d_in"],
                d_lat=self.model_args["d_lat"],
                n_blocks=self.model_args["n_blocks"],
                d_hidden=self.model_args["d_hidden"],
                dropout=self.model_args["dropout"],
            )

            model = NSARepV2Ellipsoid(
                encoder=encoder,
                k=k_i,
                lambda_rep=self.model_args["lambda_rep"],
                use_memory=self.model_args["use_memory"],
                alpha=self.model_args["alpha"],
                q=self.model_args["q"],
                epochs=self.model_args["epochs"],
                lr=self.model_args["lr"],
                bs=self.model_args["bs"],
                seed=seed,
                device=self.device,
                score_mode=self.score_mode,
                loss_mode=self.loss_mode,
            )
            model.fit(X, y)
            self.models.append(model)
        return self

    # ---------------------------------------------------------------- score
    def score(self, X, mode=None):
        """Polyclonal consensus: mean of the clone scores (Eq. 8)."""
        if not self.models:
            raise RuntimeError("The ensemble is not fitted (call fit first).")
        return np.mean(
            np.stack([m.score(X, mode=mode) for m in self.models], axis=0), axis=0)

    def score_per_clone(self, X, mode=None):
        """(M, N) matrix with the score of each clone."""
        if not self.models:
            raise RuntimeError("The ensemble is not fitted (call fit first).")
        return np.stack([m.score(X, mode=mode) for m in self.models], axis=0)

    # ----------------------------------------------------------------- info
    def describe(self):
        return {
            "n_estimators": self.n_estimators,
            "k_per_clone": self.k_per_clone,
            "k_heterogeneous": self.k_heterogeneous,
            "score_mode": self.score_mode,
            "loss_mode": self.loss_mode,
            "lambda_rep": self.model_args["lambda_rep"],
            "use_memory": self.model_args["use_memory"],
            "device": str(self.device),
        }
