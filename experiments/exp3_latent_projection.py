"""Experiment 3: t-SNE projection of the latent space (Figure 10).

For each dataset, a repertoire with the configuration of exp1 (M = 5,
self-only score, loss 'paper') and the modal K chosen in exp1 is fitted; the
test split (stratified subsample with a fixed seed) is projected with the
encoder of the first clone together with its ellipsoid centres.
Panel (a): classes and centres. Panel (b): the anomaly score of the model.
Output: PNG and vector PDF in <results>/plots_tsne/.

    EXP3_DATASETS=Cardio,Mammography,Musk,Satimage python experiments/exp3_latent_projection.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dpnsa.paths import RUNS_DIR, resultado  # noqa: E402
RESULTS_DIR = RUNS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
PLOT_DIR = os.path.join(RESULTS_DIR, "plots_tsne")
os.makedirs(PLOT_DIR, exist_ok=True)
EXP1_CSV = resultado("exp1_benchmark_results.csv")

from dpnsa.datasets import (
    prepare_annthyroid, prepare_arrhythmia, prepare_cardiotocography,
    prepare_ecg5000, prepare_ionosphere, prepare_mammography,
    prepare_musk, prepare_satimage, prepare_shuttle, prepare_creditcard
)
from dpnsa.ensemble import NSARepV23Ensemble

CREDIT_PATH = os.path.join(project_root, "data/creditcard.csv")
SEED = 42
MAX_SAMPLES = 5000
M_ENS = 5

ALL_DATASETS = {
    "Annthyroid": prepare_annthyroid,
    "Arrhythmia": prepare_arrhythmia,
    "Cardio": prepare_cardiotocography,
    "ECG5000": prepare_ecg5000,
    "Ionosphere": prepare_ionosphere,
    "Mammography": prepare_mammography,
    "Musk": prepare_musk,
    "Satimage": prepare_satimage,
    "Shuttle": prepare_shuttle,
    "CreditCard": lambda seed=SEED: prepare_creditcard(CREDIT_PATH, seed=seed),
}

# Datasets of Figure 10
DEFAULT_DS = ["Cardio", "Satimage", "Musk", "Mammography"]

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150,
})


def load_k_per_dataset():
    """Modal K chosen on calibration in exp1 (K = 1 if exp1 is not available)."""
    if not os.path.exists(EXP1_CSV):
        print("[WARNING] exp1_benchmark_results.csv not found -> K=1")
        return {}
    df = pd.read_csv(EXP1_CSV)
    sub = df[df.Method == "DP-NSA (self-only)"].dropna(subset=["K"])
    if sub.empty:
        return {}
    return sub.groupby("Dataset")["K"].agg(lambda x: int(x.mode().iloc[0])).to_dict()


def stratified_subsample(X, y, max_n, seed=SEED):
    """Stratified subsample with a fixed seed (at least 150 anomalies when available)."""
    if len(X) <= max_n:
        return X, y, np.arange(len(X))
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n1 = min(len(i1), max(int(max_n * len(i1) / len(y)), min(len(i1), 150)))
    n0 = min(len(i0), max_n - n1)
    sel = np.concatenate([rng.choice(i0, n0, replace=False),
                          rng.choice(i1, n1, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel], sel


def plot_dataset(ds_name, loader, k_ds):
    print(f"\n--- {ds_name} (K={k_ds}) ---")
    Xtr, _, Xte, ytr, _, yte = loader(seed=SEED)
    d_in = Xtr.shape[1]

    ens = NSARepV23Ensemble(
        d_in, n_estimators=M_ENS, k=k_ds, seed_start=SEED, verbose=False,
        score_mode='self_only', loss_mode='paper').fit(Xtr, ytr)

    X_viz, y_viz, _ = stratified_subsample(Xte, yte, MAX_SAMPLES)
    print(f"    plotting {len(X_viz)} samples ({int(y_viz.sum())} anomalies)")

    scores = ens.score(X_viz)

    # Latent projection of the first clone (t-SNE of one space, not of the consensus)
    det = ens.models[0]
    enc = det.enc
    enc.eval()
    dev = next(enc.parameters()).device
    with torch.no_grad():
        z = enc(torch.tensor(X_viz, dtype=torch.float32).to(dev))
        if isinstance(z, tuple):
            z = z[0]
        z_np = z.cpu().numpy()
        # self ellipsoid centres, projected together with the data
        mus = np.stack([c['mu'].cpu().numpy() for c in det.params['P0']], axis=0)

    print("    computing t-SNE...")
    combined = np.vstack([z_np, mus])
    perp = min(30, max(5, len(combined) // 4))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perp,
                init='pca', learning_rate='auto')
    emb = tsne.fit_transform(combined)
    z2, mu2 = emb[:len(z_np)], emb[len(z_np):]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) classes and centres
    ax = axes[0]
    ax.scatter(z2[y_viz == 0, 0], z2[y_viz == 0, 1], c='dodgerblue', alpha=0.35,
               s=14, edgecolors='none', label='Normal (Self)')
    ax.scatter(z2[y_viz == 1, 0], z2[y_viz == 1, 1], c='crimson', alpha=0.85,
               s=26, marker='x', linewidths=1.1, label='Anomaly (Non-Self)')
    ax.scatter(mu2[:, 0], mu2[:, 1], c='black', s=190, marker='*',
               edgecolors='white', linewidths=1.2, zorder=5,
               label=f'Detector centroids (K={k_ds})')
    ax.set_title(f"(a) {ds_name}: latent projection")
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # (b) anomaly score of the model
    ax = axes[1]
    vmin, vmax = np.percentile(scores, [2, 98])
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=scores, cmap='RdYlBu_r', s=16,
                    alpha=0.85, edgecolors='none', vmin=vmin, vmax=vmax)
    ax.scatter(mu2[:, 0], mu2[:, 1], c='black', s=190, marker='*',
               edgecolors='white', linewidths=1.2, zorder=5)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Anomaly score  $S(x)$', fontsize=10)
    ax.set_title("(b) Geometric anomaly score")
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(PLOT_DIR, f"tsne_{ds_name}.{ext}"),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [OK] tsne_{ds_name}.png / .pdf")


def run():
    ds_env = os.environ.get("EXP3_DATASETS", "")
    names = [d.strip() for d in ds_env.split(",") if d.strip()] or DEFAULT_DS
    K_BY_DS = load_k_per_dataset()

    print(">>> EXP-3: LATENT SPACE PROJECTION <<<")
    print(f"    output -> {PLOT_DIR}")
    print(f"    datasets={names}")

    for n in names:
        if n not in ALL_DATASETS:
            print(f"[SKIP] unknown dataset: {n}")
            continue
        try:
            plot_dataset(n, ALL_DATASETS[n], K_BY_DS.get(n, 1))
        except Exception as e:
            import traceback
            print(f"    [ERROR] {n}: {e}")
            traceback.print_exc()

    print("\n>>> DONE <<<")


if __name__ == "__main__":
    run()
