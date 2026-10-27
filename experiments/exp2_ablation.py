"""Experiment 2: ablation study (Table 4, Figures 5 and 6).

Block 1, mechanisms: every variant uses the K chosen on the calibration split
for each dataset and seed, with the same procedure as exp1, so the ablation
measures the configuration that the method actually deploys. The K is chosen
once per seed and shared by all variants, so the comparison is paired.

Block 2, anisotropy, with K = 1 fixed: with one component, a shrinkage close
to 1 turns the ellipsoid into a sphere whose radius follows the trace, so
comparing it with the tuned shrinkage isolates the contribution of the
learned covariance. The protocol is recorded in the k_protocol column.

The spectral diagnostics are recorded for every variant, as in exp1.

Environment variables: EXP2_DATASETS, EXP2_TAG, DPNSA_SEEDS, DPNSA_M,
DPNSA_LAMBDA, DPNSA_ALPHA, DPNSA_ALPHA_SPHERE, DPNSA_K_GRID, DPNSA_M_SELECT.
Partial runs are merged with --merge.
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))

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
from baselines.deep import spectral_diagnostics

CREDIT_PATH = os.path.join(project_root, "data/creditcard.csv")
SEEDS = [int(s) for s in os.environ.get("DPNSA_SEEDS", "100,101,102,103,104").split(",")]
M_FULL = int(os.environ.get("DPNSA_M", 5))
LAMBDA = float(os.environ.get("DPNSA_LAMBDA", 5.0))
ALPHA = float(os.environ.get("DPNSA_ALPHA", 0.1))
ALPHA_SPHERE = float(os.environ.get("DPNSA_ALPHA_SPHERE", 0.99))
K_GRID = [int(x) for x in os.environ.get("DPNSA_K_GRID", "1,3,5").split(",")]
M_SELECT = int(os.environ.get("DPNSA_M_SELECT", 2))
CUDA = torch.cuda.is_available()

TAG = os.environ.get("EXP2_TAG", "all")
DS_FILTER = [d.strip() for d in os.environ.get("EXP2_DATASETS", "").split(",")
             if d.strip()]
OUT_CSV = os.path.join(RESULTS_DIR, f"exp2_ablacion_{TAG}.csv")

DATASETS = {
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

# Block 1: mechanisms, K chosen on calibration
# name -> (loss_mode, lambda, score_mode, use_memory, M, alpha)
VARIANTS = {
    "Full":            ("paper",    LAMBDA, "discriminative", True,  M_FULL, ALPHA),
    "SelfOnly":        ("paper",    LAMBDA, "self_only",      True,  M_FULL, ALPHA),
    "OneClass":        ("paper",    0.0,    "self_only",      True,  M_FULL, ALPHA),
    "NoMemory":        ("paper",    LAMBDA, "discriminative", False, M_FULL, ALPHA),
    "Monoclonal":      ("paper",    LAMBDA, "discriminative", True,  1,      ALPHA),
    "NoRepMonoclonal": ("paper",    0.0,    "self_only",      True,  1,      ALPHA),
}

# Block 2: anisotropy, K = 1 fixed (with K > 1 the comparison would mix the
# effect of the shape with that of the granularity)
GEOMETRY = {
    "Anisotropic_K1": ("paper", LAMBDA, "discriminative", True, M_FULL, ALPHA),
    "Isotropic_K1":   ("paper", LAMBDA, "discriminative", True, M_FULL, ALPHA_SPHERE),
}


def _sync():
    if CUDA:
        torch.cuda.synchronize()


def select_k(d_in, Xtr, ytr, Xcal, ycal, seed):
    """Same procedure as in exp1 (copied so that each experiment runs on its own)."""
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


def _run_block(block, ds_name, splits, d_in, k_by_seed, k_protocol, records):
    """Run a block of variants on one dataset."""
    for v_name, (loss_mode, lam, score_mode, use_mem, M, alpha) in block.items():
        aucs, prs, ks = [], [], []
        for s in tqdm(SEEDS, desc=f"    {v_name}", leave=False):
            Xtr, Xcal, Xte, ytr, ycal, yte = splits[s]
            k_use = 1 if k_protocol == "fixed_1" else k_by_seed[s]
            try:
                _sync()
                t0 = time.time()
                model = NSARepV23Ensemble(
                    d_in, n_estimators=M, k=k_use, lambda_rep=lam, alpha=alpha,
                    use_memory=use_mem, seed_start=s, verbose=False,
                    score_mode=score_mode, loss_mode=loss_mode).fit(Xtr, ytr)
                _sync()
                t_fit = time.time() - t0

                spec = spectral_diagnostics(model)
                score = model.score(Xte)
                auc = roc_auc_score(yte, score)
                pr = average_precision_score(yte, score)

                records.append(dict(
                    Dataset=ds_name, Variant=v_name, Seed=s,
                    AUROC=auc, AUPRC=pr, K=k_use, M=M, alpha=alpha,
                    k_protocol=k_protocol, loss_mode=loss_mode,
                    lambda_rep=lam, score_mode=score_mode, use_memory=use_mem,
                    fit_time_s=round(t_fit, 2), d_in=d_in, n_test=len(yte),
                    rho_test=float(yte.mean()), Error="", **spec))
                aucs.append(auc); prs.append(pr); ks.append(k_use)

            except Exception as e:
                print(f"    [ERROR] {ds_name}/{v_name}/seed={s}: {e}")
                traceback.print_exc()
                records.append(dict(
                    Dataset=ds_name, Variant=v_name, Seed=s,
                    AUROC=np.nan, AUPRC=np.nan, K=k_use, M=M, alpha=alpha,
                    k_protocol=k_protocol, loss_mode=loss_mode,
                    lambda_rep=lam, score_mode=score_mode, use_memory=use_mem,
                    fit_time_s=np.nan, d_in=d_in, n_test=len(yte),
                    rho_test=float(yte.mean()),
                    Error=f"{type(e).__name__}: {e}",
                    r_eff_mean=np.nan, r_eff_min=np.nan, d_lat=np.nan,
                    r_ratio=np.nan, cond_mean=np.nan))

        if aucs:
            print(f"    {v_name:16s} | AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                  f"| PR {np.mean(prs):.4f}±{np.std(prs):.4f} | K={ks}")

        pd.DataFrame(records).to_csv(OUT_CSV, index=False)


def run():
    records = []

    datasets = ({k: v for k, v in DATASETS.items() if k in DS_FILTER}
                if DS_FILTER else DATASETS)
    desconocidos = [d for d in DS_FILTER if d not in DATASETS]
    if desconocidos:
        raise SystemExit(f"Unknown datasets: {desconocidos}")

    print(">>> EXP-2: ABLATION <<<")
    print(f"    device={'cuda' if CUDA else 'cpu'}  M={M_FULL}  lambda={LAMBDA}")
    print(f"    K chosen on calibration among {K_GRID}")
    print(f"    geometry block with K=1 fixed, spherical alpha={ALPHA_SPHERE}")
    print(f"    tag={TAG}  datasets={list(datasets)}")
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

        print("    [K selection on calibration]")
        k_by_seed = {}
        for s in SEEDS:
            Xtr, Xcal, _, ytr, ycal, _ = splits[s]
            k_by_seed[s] = select_k(d_in, Xtr, ytr, Xcal, ycal, s)
        print(f"    d_in={d_in}  K per seed: {list(k_by_seed.values())}")

        print("    [block 1: mechanisms]")
        _run_block(VARIANTS, ds_name, splits, d_in, k_by_seed, "calibrated", records)

        print("    [block 2: geometry, K=1 fixed]")
        _run_block(GEOMETRY, ds_name, splits, d_in, k_by_seed, "fixed_1", records)

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to: {OUT_CSV}")

    n_err = int((df.Error.fillna("") != "").sum())
    if n_err:
        print(f"[WARNING] {n_err} rows with errors.")

    resumen(df)


def resumen(df):
    """Console summary with the paired tests (the paper's are in analysis/stat_tests.py)."""
    ok = df.dropna(subset=["AUPRC"])
    if ok.empty:
        print("No valid results.")
        return

    b1 = ok[ok.k_protocol == "calibrated"]
    piv = b1.groupby(['Dataset', 'Variant'])['AUPRC'].mean().unstack()
    print("\n=== BLOCK 1: mean AUPRC per variant ===")
    print(piv.round(4).to_string())
    if 'Full' in piv.columns:
        print("\n=== Difference to Full ===")
        print(piv.sub(piv['Full'], axis=0).drop(columns=['Full']).round(4).to_string())

    try:
        from scipy.stats import wilcoxon, friedmanchisquare

        def wil(a, b):
            try:
                return float(wilcoxon(a, b, zero_method='zsplit').pvalue)
            except Exception:
                return np.nan

        def holm(p):
            p = np.asarray(p, float)
            m = len(p)
            o = np.argsort(p)
            adj = np.empty(m)
            r = 0.
            for k, i in enumerate(o):
                r = max(r, (m - k) * p[i])
                adj[i] = min(1., r)
            return adj

        if len(piv) >= 3 and 'Full' in piv.columns:
            st, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
            print(f"\nFriedman: chi2={st:.2f}  p={p:.3e}  (df={len(piv.columns) - 1})")
            rivals = [c for c in piv.columns if c != 'Full']
            padj = holm([wil(piv[c], piv['Full']) for c in rivals])
            print(f"\n=== Wilcoxon vs Full, Holm over {len(rivals)} comparisons ===")
            for c, pa in sorted(zip(rivals, padj), key=lambda x: x[1]):
                d = piv[c] - piv['Full']
                print(f"  {c:16s} W{int((d > 0).sum())}-L{int((d < 0).sum())}  "
                      f"diff={d.mean():+.4f}  p_holm={pa:.4f}")
    except Exception as e:
        print(f"[scipy not available: {e}]")

    b2 = ok[ok.k_protocol == "fixed_1"]
    if not b2.empty:
        g = b2.groupby(['Dataset', 'Variant'])['AUPRC'].mean().unstack()
        print("\n=== BLOCK 2: anisotropy (K=1 fixed) ===")
        print(g.round(4).to_string())
        if {"Anisotropic_K1", "Isotropic_K1"} <= set(g.columns):
            d = g["Anisotropic_K1"] - g["Isotropic_K1"]
            print(f"\nContribution of anisotropy: mean={d.mean():+.4f}  "
                  f"W{int((d > 0).sum())}-L{int((d < 0).sum())}")
            try:
                from scipy.stats import wilcoxon
                pv = wilcoxon(g['Anisotropic_K1'], g['Isotropic_K1'],
                              zero_method='zsplit').pvalue
                print(f"  Wilcoxon p={pv:.4f}")
            except Exception:
                pass
            print("\n  Positive: the learned covariance adds to the sphere.")
            print("  Close to zero: it does not.")
            print("")

    sp = ok[ok.Variant == "Full"].groupby("Dataset")[
        ["d_in", "d_lat", "r_eff_mean", "r_ratio"]].mean()
    if not sp.empty and not sp.r_ratio.isna().all():
        print("\n=== Effective rank of the latent covariance ===")
        print(sp.round(2).to_string())


def merge():
    """Merge the partial CSVs of a parallel run."""
    import glob
    files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "exp2_ablacion_*.csv"))
             if not f.endswith("exp2_ablacion_results.csv")]
    if not files:
        print("No partial files to merge.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = os.path.join(RESULTS_DIR, "exp2_ablacion_results.csv")
    df.to_csv(out, index=False)
    print(f"Merged {len(files)} files -> {out}  ({len(df)} rows)")

    faltan = set(DATASETS) - set(df.Dataset.unique())
    if faltan:
        print(f"[WARNING] missing datasets: {sorted(faltan)}")
    resumen(df)


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        run()
