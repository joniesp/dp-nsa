"""Experiment 6: label scarcity (Figure 4; input of the trend test in exp9).

The number of labelled anomalies in the training split is reduced to
100, 50, 25, 10, 5 and 1 % (at least EXP6_MIN_ANOM = 5). Discarded anomalies
are removed, not relabelled as normal, so that label scarcity is not
confused with label noise. K is chosen on calibration within each fraction,
using only the supervision available at that fraction. Methods that use no
anomaly labels are fitted once and repeated across fractions. The number of
non-self components actually fitted (n_P1) is recorded.

Environment variables: EXP6_DATASETS, EXP6_TAG, EXP6_MIN_ANOM, DPNSA_SEEDS,
DPNSA_M, DPNSA_K_GRID, DPNSA_M_SELECT, MSSVDD_K. Partial runs are merged
with --merge.
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
from baselines.shallow import (
    IsolationForestWrapper, OCSVMWrapper, XGBoostBalancedWrapper
)
from baselines.deep import (
    DeepSVDDWrapper, TabResNetWrapper, DeepSADWrapper, DevNetWrapper,
    DeepMSSVDDWrapper
)

CREDIT_PATH = os.path.join(project_root, "data/creditcard.csv")
SEEDS = [int(s) for s in os.environ.get("DPNSA_SEEDS", "100,101,102,103,104").split(",")]
FRACTIONS = [1.0, 0.5, 0.25, 0.10, 0.05, 0.01]
MIN_ANOM = int(os.environ.get("EXP6_MIN_ANOM", 5))
M_ENS = int(os.environ.get("DPNSA_M", 5))
K_GRID = [int(x) for x in os.environ.get("DPNSA_K_GRID", "1,3,5").split(",")]
M_SELECT = int(os.environ.get("DPNSA_M_SELECT", 2))
MS_SPHERES = int(os.environ.get("MSSVDD_K", 5))
CUDA = torch.cuda.is_available()

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))

TAG = os.environ.get("EXP6_TAG", "all")
DS_FILTER = [d.strip() for d in os.environ.get("EXP6_DATASETS", "").split(",")
             if d.strip()]
OUT_CSV = os.path.join(RESULTS_DIR, f"exp6_escasez_{TAG}.csv")

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

TAG_MAIN = "DP-NSA"
TAG_SELF = "DP-NSA (self-only)"
TAG_ONEC = "DP-NSA (lambda=0)"

SUPERVISED = ["DP-NSA", "XGBoost", "TabularResNet", "DeepSAD", "DevNet"]
INVARIANT = [TAG_ONEC, "Deep SVDD", "Deep MS-SVDD", "iForest", "OCSVM"]


def subsample_anomalies(ytr, frac, seed):
    """Mask of the training rows to keep: all normal rows and a seeded subset of anomalies."""
    idx_norm = np.where(ytr == 0)[0]
    idx_anom = np.where(ytr == 1)[0]
    n_keep = max(MIN_ANOM, int(round(len(idx_anom) * frac)))
    n_keep = min(n_keep, len(idx_anom))
    rng = np.random.RandomState(seed)
    keep = rng.choice(idx_anom, size=n_keep, replace=False)
    mask = np.zeros(len(ytr), dtype=bool)
    mask[idx_norm] = True
    mask[keep] = True
    return mask, n_keep


def select_k(d_in, Xtr, ytr, Xcal, ycal, seed):
    """Same procedure as in exp1, called on the subsampled training set."""
    best_k, best_ap = K_GRID[0], -np.inf
    for k in K_GRID:
        try:
            m = NSARepV23Ensemble(d_in, n_estimators=M_SELECT, k=k, seed_start=seed,
                                  verbose=False, score_mode='self_only',
                                  loss_mode='paper').fit(Xtr, ytr)
            ap = average_precision_score(ycal, m.score(Xcal, mode='self_only'))
        except Exception:
            ap = -np.inf
        if ap > best_ap:
            best_ap, best_k = ap, k
    return best_k


def build(name, d_in, seed, k):
    if name == TAG_ONEC:
        return NSARepV23Ensemble(d_in, n_estimators=M_ENS, k=k, lambda_rep=0.0,
                                 seed_start=seed, verbose=False,
                                 score_mode='self_only', loss_mode='paper')
    if name == "XGBoost":       return XGBoostBalancedWrapper(random_state=seed)
    if name == "TabularResNet": return TabResNetWrapper(input_dim=d_in, seed=seed)
    if name == "DeepSAD":       return DeepSADWrapper(input_dim=d_in, seed=seed)
    if name == "DevNet":        return DevNetWrapper(input_dim=d_in, seed=seed)
    if name == "Deep SVDD":     return DeepSVDDWrapper(d_in, seed=seed)
    if name == "Deep MS-SVDD":  return DeepMSSVDDWrapper(d_in, n_spheres=MS_SPHERES,
                                                         seed=seed)
    if name == "iForest":       return IsolationForestWrapper(random_state=seed)
    if name == "OCSVM":         return OCSVMWrapper(nu=0.1)
    raise ValueError(name)


def fit_model(name, model, X, y):
    if name in (TAG_ONEC, "XGBoost", "TabularResNet", "DeepSAD", "DevNet"):
        model.fit(X, y)
    elif name in ("Deep SVDD", "Deep MS-SVDD"):
        model.fit(X[y == 0])
    else:
        model.fit(X)
    return model


def infer(model, X):
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        return raw[:, 1] if getattr(raw, "ndim", 1) == 2 else raw
    return model.score(X)


def mean_n_P1(ens):
    """Mean number of non-self components fitted per clone."""
    try:
        return float(np.mean([len(m.params['P1']) for m in ens.models]))
    except Exception:
        return np.nan


def run():
    datasets = ({k: v for k, v in ALL_DATASETS.items() if k in DS_FILTER}
                if DS_FILTER else ALL_DATASETS)
    desconocidos = [d for d in DS_FILTER if d not in ALL_DATASETS]
    if desconocidos:
        raise SystemExit(f"Unknown datasets: {desconocidos}")

    records = []
    print(">>> EXP-6: LABEL SCARCITY <<<")
    print(f"    device={'cuda' if CUDA else 'cpu'}  tag={TAG}  M={M_ENS}")
    print("    K chosen on calibration within each fraction")
    print(f"    fractions={FRACTIONS}  floor={MIN_ANOM} anomalies")
    print(f"    datasets={list(datasets)}")
    print(f"    output -> {OUT_CSV}\n")

    for ds_name, loader in datasets.items():
        print(f"\n=== {ds_name} " + "=" * 45)
        try:
            splits = {s: loader(seed=s) for s in SEEDS}
        except Exception as e:
            print(f"    [LOAD ERROR] {e}")
            traceback.print_exc()
            continue

        d_in = splits[SEEDS[0]][0].shape[1]
        n_anom_full = int(splits[SEEDS[0]][3].sum())
        print(f"    d_in={d_in}  training anomalies (100%)={n_anom_full}")

        # ------------------------------------------- methods without anomaly labels
        print("    [invariant]")
        inv = {}
        for m_name in INVARIANT:
            aucs, prs = [], []
            for s in SEEDS:
                Xtr, Xcal, Xte, ytr, ycal, yte = splits[s]
                try:
                    k_use = (select_k(d_in, Xtr, ytr, Xcal, ycal, s)
                             if m_name == TAG_ONEC else 1)
                    model = fit_model(m_name, build(m_name, d_in, s, k_use), Xtr, ytr)
                    sc = infer(model, Xte)
                    a, p = roc_auc_score(yte, sc), average_precision_score(yte, sc)
                    inv.setdefault(m_name, []).append((s, a, p))
                    aucs.append(a); prs.append(p)
                except Exception as e:
                    print(f"      [ERROR] {m_name}/seed={s}: {e}")
                    inv.setdefault(m_name, []).append((s, np.nan, np.nan))
            if aucs:
                print(f"      {m_name:22s} | AUC {np.mean(aucs):.4f} "
                      f"| PR {np.mean(prs):.4f}")

        for m_name, vals in inv.items():
            for frac in FRACTIONS:
                for (s, a, p) in vals:
                    records.append(dict(
                        Dataset=ds_name, Method=m_name, Frac=frac,
                        n_anom_train=n_anom_full, Seed=s, AUROC=a, AUPRC=p,
                        K=np.nan, n_P1=np.nan, d_in=d_in, invariant=True, Error=""))

        # -------------------------------- methods that depend on the fraction
        for frac in FRACTIONS:
            n_kept = None
            for m_name in SUPERVISED:
                aucs, prs, ks, n_p1s, prs_s = [], [], [], [], []
                for s in tqdm(SEEDS, desc=f"    frac={frac:<5} {m_name}", leave=False):
                    Xtr, Xcal, Xte, ytr, ycal, yte = splits[s]
                    try:
                        mask, n_kept = subsample_anomalies(ytr, frac, seed=s)
                        Xs, ys = Xtr[mask], ytr[mask]

                        if m_name == "DP-NSA":
                            k_use = select_k(d_in, Xs, ys, Xcal, ycal, s)
                            ks.append(k_use)
                            model = NSARepV23Ensemble(
                                d_in, n_estimators=M_ENS, k=k_use, seed_start=s,
                                verbose=False, score_mode='self_only',
                                loss_mode='paper').fit(Xs, ys)
                            n_p1 = mean_n_P1(model)
                            n_p1s.append(n_p1)

                            for tag, mode in ((TAG_MAIN, 'discriminative'),
                                              (TAG_SELF, 'self_only')):
                                sc = model.score(Xte, mode=mode)
                                a = roc_auc_score(yte, sc)
                                p = average_precision_score(yte, sc)
                                records.append(dict(
                                    Dataset=ds_name, Method=tag, Frac=frac,
                                    n_anom_train=n_kept, Seed=s, AUROC=a, AUPRC=p,
                                    K=k_use, n_P1=n_p1, d_in=d_in,
                                    invariant=False, Error=""))
                                if tag == TAG_MAIN:
                                    aucs.append(a); prs.append(p)
                                else:
                                    prs_s.append(p)
                        else:
                            model = fit_model(m_name, build(m_name, d_in, s, 1), Xs, ys)
                            sc = infer(model, Xte)
                            a = roc_auc_score(yte, sc)
                            p = average_precision_score(yte, sc)
                            records.append(dict(
                                Dataset=ds_name, Method=m_name, Frac=frac,
                                n_anom_train=n_kept, Seed=s, AUROC=a, AUPRC=p,
                                K=np.nan, n_P1=np.nan, d_in=d_in,
                                invariant=False, Error=""))
                            aucs.append(a); prs.append(p)

                    except Exception as e:
                        print(f"      [ERROR] {ds_name}/{m_name}/frac={frac}/"
                              f"seed={s}: {e}")
                        traceback.print_exc()
                        records.append(dict(
                            Dataset=ds_name, Method=m_name, Frac=frac,
                            n_anom_train=n_kept, Seed=s, AUROC=np.nan, AUPRC=np.nan,
                            K=np.nan, n_P1=np.nan, d_in=d_in, invariant=False,
                            Error=f"{type(e).__name__}: {e}"))

                if aucs:
                    extra = ""
                    if prs_s:
                        extra = (f" || self-only PR {np.mean(prs_s):.4f}"
                                 f" | K={ks} | n_P1={np.mean(n_p1s):.1f}")
                    print(f"    frac={frac:<5} n={n_kept:<5} {m_name:15s} | "
                          f"AUC {np.mean(aucs):.4f} | PR {np.mean(prs):.4f}"
                          f"+-{np.std(prs):.4f}{extra}")

            pd.DataFrame(records).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to: {OUT_CSV}")
    n_err = int((df.Error.fillna("") != "").sum())
    if n_err:
        print(f"[WARNING] {n_err} rows with errors.")
    resumen(df)


def resumen(df):
    """Console summary, printed at the end of a run and after --merge."""
    ok = df.dropna(subset=["AUPRC"])
    if ok.empty:
        print("No valid results.")
        return

    order = [TAG_MAIN, TAG_SELF, TAG_ONEC, "XGBoost", "TabularResNet",
             "DeepSAD", "DevNet", "Deep SVDD", "Deep MS-SVDD", "iForest", "OCSVM"]
    cols = [c for c in order if c in ok.Method.unique()]
    piv = ok.groupby(["Frac", "Method"])["AUPRC"].mean().unstack()[cols]
    fr = sorted(piv.index, reverse=True)
    piv = piv.loc[fr]

    print("\n=== Mean AUPRC per fraction ===")
    print(piv.round(4).to_string())

    print("\n=== Degradation from 100% to 1% ===")
    for c in cols:
        hi, lo = piv.loc[1.0, c], piv.loc[0.01, c]
        print(f"  {c:22s} {hi:.4f} -> {lo:.4f}   "
              f"delta={lo-hi:+.4f}  ({100*(lo-hi)/hi:+.1f}%)")

    if TAG_SELF in cols:
        print("\n=== Discriminative rule vs self-only ===")
        for f in fr:
            s = ok[ok.Frac == f].groupby(["Dataset", "Method"])["AUPRC"].mean().unstack()
            d = s[TAG_MAIN] - s[TAG_SELF]
            print(f"  {f:<5} diff={d.mean():+.4f}  "
                  f"W{int((d>0).sum())}-L{int((d<0).sum())}")

    if "n_P1" in ok.columns:
        p1 = ok[ok.Method == TAG_MAIN].groupby("Frac")["n_P1"].mean()
        if not p1.isna().all():
            print("\n=== Non-self components fitted ===")
            print(p1.loc[fr].round(2).to_string())

    kk = ok[ok.Method == TAG_MAIN].dropna(subset=["K"])
    if not kk.empty:
        print("\n=== K chosen per fraction ===")
        print(kk.groupby("Frac")["K"].agg(
            lambda x: dict(sorted(x.astype(int).value_counts().items()))
        ).loc[fr].to_string())

    n = ok[~ok.invariant].groupby(["Dataset", "Frac"])["n_anom_train"].first().unstack()
    print("\n=== Training anomalies ===")
    print(n[fr].astype("Int64").to_string())


def merge():
    """Merge the partial CSVs of a parallel run."""
    files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "exp6_escasez_*.csv"))
             if not f.endswith("exp6_escasez_results.csv")]
    if not files:
        print("No partial files to merge.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = os.path.join(RESULTS_DIR, "exp6_escasez_results.csv")
    df.to_csv(out, index=False)
    print(f"Merged {len(files)} files -> {out}  ({len(df)} rows)")
    faltan = set(ALL_DATASETS) - set(df.Dataset.unique())
    if faltan:
        print(f"[WARNING] missing datasets: {sorted(faltan)}")
    resumen(df)


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        run()