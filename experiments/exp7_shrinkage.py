"""Experiment 7: sensitivity to the shrinkage coefficient alpha (Figure 8).

alpha in {0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9} crossed with
K in {1, 3, 5}. alpha = 0 keeps only the ridge term (control); alpha -> 1
tends to a sphere, the isotropic assumption of Deep SVDD. For every
configuration the mean condition number of the covariances and the number of
fitted components are recorded, to tell a drop caused by poor
regularisation from one caused by lost detectors.

Default datasets, chosen for their N/d regime: Arrhythmia (~1), Musk (~11),
Mammography (~1118), CreditCard (~5893).

    EXP7_DATASETS=Arrhythmia,Musk python experiments/exp7_shrinkage.py
    EXP7_ALPHAS=0.0,0.1,0.5 EXP7_KS=1,3 python experiments/exp7_shrinkage.py
    python experiments/exp7_shrinkage.py --merge      # after per-dataset runs (EXP7_TAG)
"""

import os
import sys
import glob
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
M_ENS = int(os.environ.get("DPNSA_M", 5))
TAG = os.environ.get("EXP7_TAG", "all")
CUDA = torch.cuda.is_available()

ALPHAS = [float(a) for a in os.environ.get(
    "EXP7_ALPHAS", "0.0,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9").split(",")]
KS = [int(k) for k in os.environ.get("EXP7_KS", "1,3,5").split(",")]

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

DEFAULT_DS = ["Arrhythmia", "Musk", "Mammography", "CreditCard"]


def geometry_diagnostics(ensemble):
    """Mean condition number of the self covariances and mean number of fitted components."""
    conds, n_comp = [], []
    for model in ensemble.models:
        comps = model.params.get('P0', [])
        n_comp.append(len(comps))
        for c in comps:
            try:
                # Sinv is the precision matrix: same condition number as Sigma
                s = torch.linalg.svdvals(c['Sinv'])
                conds.append(float(s.max() / (s.min() + 1e-12)))
            except Exception:
                pass
    return (float(np.mean(conds)) if conds else np.nan,
            float(np.mean(n_comp)) if n_comp else np.nan)


def run():
    ds_env = os.environ.get("EXP7_DATASETS", "")
    names = [d.strip() for d in ds_env.split(",") if d.strip()] or DEFAULT_DS
    for n in names:
        if n not in ALL_DATASETS:
            raise SystemExit(f"Unknown dataset: {n}. Options: {list(ALL_DATASETS)}")

    out_csv = os.path.join(RESULTS_DIR, f"exp7_shrinkage_{TAG}.csv")
    records = []

    print(">>> EXP-7: SHRINKAGE SENSITIVITY <<<")
    print(f"    device={'cuda' if CUDA else 'cpu'}  tag={TAG}  M={M_ENS}")
    print(f"    alphas={ALPHAS}")
    print(f"    K={KS}")
    print(f"    datasets={names}\n")

    for ds_name in names:
        print(f"\n=== {ds_name} " + "=" * 45)
        try:
            splits = {s: ALL_DATASETS[ds_name](seed=s) for s in SEEDS}
        except Exception as e:
            print(f"    [LOAD ERROR] {e}")
            traceback.print_exc()
            continue

        d_in = splits[SEEDS[0]][0].shape[1]
        n_tr = len(splits[SEEDS[0]][3])
        print(f"    d={d_in}  N_train={n_tr}  N/d={n_tr/d_in:.1f}")

        for K in KS:
            for alpha in ALPHAS:
                aucs, prs, conds, ncomps = [], [], [], []
                for s in tqdm(SEEDS, desc=f"    K={K} a={alpha}", leave=False):
                    Xtr, _, Xte, ytr, _, yte = splits[s]
                    try:
                        ens = NSARepV23Ensemble(
                            d_in, n_estimators=M_ENS, k=K, alpha=alpha,
                            seed_start=s, verbose=False,
                            score_mode='self_only', loss_mode='paper').fit(Xtr, ytr)
                        score = ens.score(Xte)
                        auc = roc_auc_score(yte, score)
                        pr = average_precision_score(yte, score)
                        cond, ncomp = geometry_diagnostics(ens)

                        records.append(dict(
                            Dataset=ds_name, K=K, alpha=alpha, Seed=s,
                            AUROC=auc, AUPRC=pr, cond_number=cond,
                            n_components=ncomp, d=d_in, N_train=n_tr,
                            N_over_d=round(n_tr / d_in, 2), Error=""))
                        aucs.append(auc); prs.append(pr)
                        if not np.isnan(cond):
                            conds.append(cond)
                        ncomps.append(ncomp)

                    except Exception as e:
                        print(f"      [ERROR] {ds_name}/K={K}/a={alpha}/seed={s}: {e}")
                        records.append(dict(
                            Dataset=ds_name, K=K, alpha=alpha, Seed=s,
                            AUROC=np.nan, AUPRC=np.nan, cond_number=np.nan,
                            n_components=np.nan, d=d_in, N_train=n_tr,
                            N_over_d=round(n_tr / d_in, 2),
                            Error=f"{type(e).__name__}: {e}"))

                if aucs:
                    c_str = f"{np.mean(conds):.1e}" if conds else "n/a"
                    print(f"    K={K}  alpha={alpha:<5} | AUC {np.mean(aucs):.4f} "
                          f"| PR {np.mean(prs):.4f}+-{np.std(prs):.4f} "
                          f"| cond {c_str} | comps {np.mean(ncomps):.1f}")

                pd.DataFrame(records).to_csv(out_csv, index=False)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")
    resumen(df)


def resumen(df):
    """Console summary, printed at the end of a run and after --merge."""
    ok = df.dropna(subset=["AUPRC"])
    if ok.empty:
        print("No valid results.")
        return

    for ds in sorted(ok.Dataset.unique()):
        sub = ok[ok.Dataset == ds]
        print(f"\n=== {ds}  (N/d = {sub.N_over_d.iloc[0]}) ===")
        print("Mean AUPRC (rows K, columns alpha):")
        print(sub.pivot_table(values="AUPRC", index="K", columns="alpha",
                              aggfunc="mean").round(4).to_string())
        print("Std across seeds:")
        print(sub.pivot_table(values="AUPRC", index="K", columns="alpha",
                              aggfunc="std").round(4).to_string())

    print("\n=== Range of AUPRC across alpha ===")
    print("(how much the choice of alpha matters in each regime)")
    rows = []
    for (ds, K), g in ok.groupby(["Dataset", "K"]):
        m = g.groupby("alpha")["AUPRC"].mean()
        rows.append(dict(Dataset=ds, K=K, N_over_d=g.N_over_d.iloc[0],
                         best_alpha=float(m.idxmax()), best_AUPRC=round(m.max(), 4),
                         worst_AUPRC=round(m.min(), 4),
                         range=round(m.max() - m.min(), 4),
                         AUPRC_at_0p1=round(float(m.get(0.1, np.nan)), 4)))
    t = pd.DataFrame(rows).sort_values(["Dataset", "K"])
    print(t.to_string(index=False))

    print("\n=== Cost of fixing alpha = 0.1 against the best per configuration ===")
    t["gap_vs_best"] = (t["best_AUPRC"] - t["AUPRC_at_0p1"]).round(4)
    print(t[["Dataset", "K", "best_alpha", "gap_vs_best"]].to_string(index=False))
    print(f"\n  Mean loss from fixing alpha=0.1: {t['gap_vs_best'].mean():+.4f} AUPRC")


def merge():
    """Merge the per-dataset CSVs into exp7_shrinkage_all.csv."""
    files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "exp7_shrinkage_*.csv"))
             if not f.endswith("exp7_shrinkage_all.csv")]
    if not files:
        print("No partial files to merge.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = os.path.join(RESULTS_DIR, "exp7_shrinkage_all.csv")
    df.to_csv(out, index=False)
    print(f"Merged {len(files)} files -> {out}  ({len(df)} rows)")
    resumen(df)


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        run()