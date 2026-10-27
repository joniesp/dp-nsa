"""Experiment 5: granularity of the repertoire (Figure 7).

Part A, K x M grid: K in {1, 3, 5, 10} x M in {1, 3, 5, 10}.
Part B, heterogeneous repertoire: clones with different K against
homogeneous repertoires of the same size M = 5:

    Homog_K1     K = [1, 1, 1, 1, 1]
    Homog_K3     K = [3, 3, 3, 3, 3]
    Homog_K5     K = [5, 5, 5, 5, 5]
    Heterog      K = [1, 3, 5, 1, 3]
    Heterog_wide K = [1, 2, 3, 5, 10]

This experiment describes the response surface on the test split; it does
not select hyperparameters (that is done on calibration in exp1).

    python experiments/exp5_granularity.py
    EXP5_PARTE=A python experiments/exp5_granularity.py      # grid only
    EXP5_PARTE=B python experiments/exp5_granularity.py      # heterogeneous only
    EXP5_DATASETS=CreditCard,Shuttle python experiments/exp5_granularity.py
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dpnsa.paths import RUNS_DIR  # noqa: E402
RESULTS_DIR = RUNS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

from dpnsa.datasets import (
    prepare_creditcard, prepare_annthyroid, prepare_arrhythmia, prepare_mammography,
    prepare_shuttle, prepare_satimage, prepare_musk, prepare_ionosphere,
    prepare_cardiotocography, prepare_ecg5000
)
from dpnsa.ensemble import NSARepV23Ensemble

CREDIT_PATH = os.path.join(project_root, "data/creditcard.csv")
SEEDS = [int(s) for s in os.environ.get("DPNSA_SEEDS", "100,101,102,103,104").split(",")]
CUDA = torch.cuda.is_available()

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
    "CreditCard": lambda seed: prepare_creditcard(CREDIT_PATH, seed=seed),
}

# Datasets of Figure 7
DEFAULT_DS = ["CreditCard", "Mammography", "Arrhythmia", "Shuttle"]

K_GRID = [1, 3, 5, 10]
M_GRID = [1, 3, 5, 10]

REPERTORIES = {
    "Homog_K1":     [1, 1, 1, 1, 1],
    "Homog_K3":     [3, 3, 3, 3, 3],
    "Homog_K5":     [5, 5, 5, 5, 5],
    "Heterog":      [1, 3, 5, 1, 3],
    "Heterog_wide": [1, 2, 3, 5, 10],
}


def _eval(model, Xte, yte):
    s = model.score(Xte)
    return roc_auc_score(yte, s), average_precision_score(yte, s)


def check_heterogeneous_support(d_in=10):
    """True if the ensemble accepts one K per clone."""
    try:
        m = NSARepV23Ensemble(d_in, n_estimators=3, k=[1, 3, 5])
        return hasattr(m, "k_per_clone") and m.k_per_clone == [1, 3, 5]
    except Exception:
        return False


def parte_A(datasets, records):
    print("\n" + "=" * 60)
    print(" PART A — K x M grid")
    print("=" * 60)
    for ds_name in datasets:
        loader = ALL_DATASETS[ds_name]
        print(f"\n=== {ds_name} ===")
        splits = {s: loader(seed=s) for s in SEEDS}
        d_in = splits[SEEDS[0]][0].shape[1]

        for K in K_GRID:
            for M in M_GRID:
                aucs, prs = [], []
                for s in tqdm(SEEDS, desc=f"  K={K} M={M}", leave=False):
                    Xtr, _, Xte, ytr, _, yte = splits[s]
                    try:
                        model = NSARepV23Ensemble(
                            d_in, n_estimators=M, k=K, seed_start=s,
                            verbose=False, score_mode='self_only',
                            loss_mode='paper').fit(Xtr, ytr)
                        auc, pr = _eval(model, Xte, yte)
                        records.append(dict(
                            Parte="A", Dataset=ds_name, Config=f"K{K}_M{M}",
                            K=K, M=M, Repertorio="", Seed=s,
                            AUROC=auc, AUPRC=pr, Error=""))
                        aucs.append(auc); prs.append(pr)
                    except Exception as e:
                        print(f"    [ERROR] {ds_name} K={K} M={M} seed={s}: {e}")
                        records.append(dict(
                            Parte="A", Dataset=ds_name, Config=f"K{K}_M{M}",
                            K=K, M=M, Repertorio="", Seed=s,
                            AUROC=np.nan, AUPRC=np.nan,
                            Error=f"{type(e).__name__}: {e}"))
                if aucs:
                    print(f"    K={K:2d} M={M:2d} | AUC {np.mean(aucs):.4f} "
                          f"| PR {np.mean(prs):.4f}±{np.std(prs):.4f}")
                pd.DataFrame(records).to_csv(
                    os.path.join(RESULTS_DIR, "exp5_sensibilidad_results.csv"),
                    index=False)


def parte_B(datasets, records):
    print("\n" + "=" * 60)
    print(" PART B — heterogeneous repertoire (M=5)")
    print("=" * 60)
    for ds_name in datasets:
        loader = ALL_DATASETS[ds_name]
        print(f"\n=== {ds_name} ===")
        splits = {s: loader(seed=s) for s in SEEDS}
        d_in = splits[SEEDS[0]][0].shape[1]

        for rep_name, k_list in REPERTORIES.items():
            aucs, prs = [], []
            for s in tqdm(SEEDS, desc=f"  {rep_name}", leave=False):
                Xtr, _, Xte, ytr, _, yte = splits[s]
                try:
                    model = NSARepV23Ensemble(
                        d_in, n_estimators=len(k_list), k=k_list, seed_start=s,
                        verbose=False, score_mode='self_only',
                        loss_mode='paper').fit(Xtr, ytr)
                    auc, pr = _eval(model, Xte, yte)
                    records.append(dict(
                        Parte="B", Dataset=ds_name, Config=rep_name,
                        K=str(k_list), M=len(k_list), Repertorio=rep_name, Seed=s,
                        AUROC=auc, AUPRC=pr, Error=""))
                    aucs.append(auc); prs.append(pr)
                except Exception as e:
                    print(f"    [ERROR] {ds_name}/{rep_name}/seed={s}: {e}")
                    traceback.print_exc()
                    records.append(dict(
                        Parte="B", Dataset=ds_name, Config=rep_name,
                        K=str(k_list), M=len(k_list), Repertorio=rep_name, Seed=s,
                        AUROC=np.nan, AUPRC=np.nan,
                        Error=f"{type(e).__name__}: {e}"))
            if aucs:
                print(f"    {rep_name:14s} | AUC {np.mean(aucs):.4f} "
                      f"| PR {np.mean(prs):.4f}±{np.std(prs):.4f}")
            pd.DataFrame(records).to_csv(
                os.path.join(RESULTS_DIR, "exp5_sensibilidad_results.csv"),
                index=False)


def run():
    parte = os.environ.get("EXP5_PARTE", "AB").upper()
    ds_env = os.environ.get("EXP5_DATASETS", "")
    datasets = [d.strip() for d in ds_env.split(",") if d.strip()] or DEFAULT_DS
    for d in datasets:
        if d not in ALL_DATASETS:
            raise SystemExit(f"Unknown dataset: {d}. Options: {list(ALL_DATASETS)}")

    print(">>> EXP-5: GRANULARITY <<<")
    print(f"    device={'cuda' if CUDA else 'cpu'}  parts={parte}")
    print(f"    datasets={datasets}")

    het_ok = check_heterogeneous_support()
    if "B" in parte and not het_ok:
        print("\n[PART B SKIPPED] The ensemble does not accept a list of K.")
        print("   Check dpnsa/ensemble.py.")
        parte = parte.replace("B", "")
        if not parte:
            return

    records = []
    if "A" in parte:
        parte_A(datasets, records)
    if "B" in parte:
        parte_B(datasets, records)

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "exp5_sensibilidad_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to: {out}")

    ok = df.dropna(subset=["AUPRC"])

    if "A" in parte and (ok.Parte == "A").any():
        a = ok[ok.Parte == "A"]
        print("\n=== PART A — mean AUPRC (rows K, columns M) ===")
        for ds in a.Dataset.unique():
            print(f"\n{ds}")
            print(a[a.Dataset == ds].pivot_table(
                values="AUPRC", index="K", columns="M", aggfunc="mean").round(4).to_string())
        print("\n=== PART A — std across seeds (rows K, columns M) ===")
        for ds in a.Dataset.unique():
            print(f"\n{ds}")
            print(a[a.Dataset == ds].pivot_table(
                values="AUPRC", index="K", columns="M", aggfunc="std").round(4).to_string())

    if "B" in parte and (ok.Parte == "B").any():
        b = ok[ok.Parte == "B"]
        piv = b.groupby(['Dataset', 'Repertorio'])['AUPRC'].mean().unstack()
        print("\n=== PART B — mean AUPRC per repertoire ===")
        print(piv.round(4).to_string())
        if "Heterog" in piv.columns:
            print("\n=== Heterog minus homogeneous ===")
            for c in [c for c in piv.columns if c.startswith("Homog")]:
                d = piv["Heterog"] - piv[c]
                print(f"  vs {c:12s} W{(d>0).sum()}-L{(d<0).sum()}  "
                      f"mean_diff={d.mean():+.4f}")


if __name__ == "__main__":
    run()