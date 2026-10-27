"""Experiment 1: benchmark against the baselines (Table 3).

Ten datasets x five seeds (100-104); each seed changes the train/calibration/
test split as well as the initialisation. For DP-NSA the number of ellipsoids
K is chosen on the calibration split (never on test), and both scoring rules
are evaluated from the same fitted model: "DP-NSA" (discriminative rule, the
headline method) and "DP-NSA (self-only)". Latency is measured per sample,
with CUDA synchronisation. The spectral diagnostics of the latent covariance
(effective rank) are recorded for every DP-NSA fit. Errors are recorded in
the 'Error' column instead of being hidden.

Environment variables: EXP1_DATASETS (comma-separated subset), EXP1_TAG
(name of the partial CSV), DPNSA_SEEDS, DPNSA_K_GRID, DPNSA_M,
DPNSA_M_SELECT, MSSVDD_K. Partial runs are merged with --merge.
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
from baselines.shallow import (
    IsolationForestWrapper, OCSVMWrapper, XGBoostBalancedWrapper
)
from baselines.deep import (
    DeepSVDDWrapper, ContrastiveAEWrapper,
    TabResNetWrapper, DeepSADWrapper, DevNetWrapper,
    DeepMSSVDDWrapper, spectral_diagnostics
)

CREDIT_PATH = os.path.join(project_root, "data/creditcard.csv")
SEEDS = [int(s) for s in os.environ.get("DPNSA_SEEDS", "100,101,102,103,104").split(",")]

K_GRID = [int(x) for x in os.environ.get("DPNSA_K_GRID", "1,3,5").split(",")]
N_ESTIMATORS = int(os.environ.get("DPNSA_M", 5))
M_SELECT = int(os.environ.get("DPNSA_M_SELECT", 2))
MS_SPHERES = int(os.environ.get("MSSVDD_K", 5))
CUDA = torch.cuda.is_available()

# Parallel runs: each process takes a subset of datasets and writes its own CSV.
TAG = os.environ.get("EXP1_TAG", "all")
DS_FILTER = [d.strip() for d in os.environ.get("EXP1_DATASETS", "").split(",")
             if d.strip()]
OUT_CSV = os.path.join(RESULTS_DIR, f"exp1_benchmark_{TAG}.csv")

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

SUPERVISED = {"DP-NSA", "XGBoost", "TabularResNet", "DeepSAD", "DevNet"}
# methods fitted on the normal class only
NORMAL_ONLY = {"Deep SVDD", "Deep MS-SVDD"}

# Method names in the CSV: the headline method uses the discriminative rule
TAG_MAIN = "DP-NSA"
TAG_SELF = "DP-NSA (self-only)"


def _sync():
    if CUDA:
        torch.cuda.synchronize()


def _timed(fn, n):
    """Latency in ms per sample, with CUDA synchronisation."""
    _sync()
    t0 = time.time()
    out = fn()
    _sync()
    return out, (time.time() - t0) * 1000.0 / max(n, 1)


def _infer(model, X):
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        return raw[:, 1] if getattr(raw, "ndim", 1) == 2 else raw
    return model.score(X)


def _empty_spec():
    return dict(r_eff_mean=np.nan, r_eff_min=np.nan, d_lat=np.nan,
                r_ratio=np.nan, cond_mean=np.nan)


def select_k(d_in, Xtr, ytr, Xcal, ycal, seed):
    """Choose K maximising AUPRC on the calibration split.

    The search uses the self-only rule, which does not need P1 and is
    therefore cheaper.
    """
    best_k, best_ap, trace = K_GRID[0], -np.inf, {}
    for k in K_GRID:
        try:
            m = NSARepV23Ensemble(d_in, n_estimators=M_SELECT, k=k, seed_start=seed,
                                  verbose=False, score_mode='self_only',
                                  loss_mode='paper').fit(Xtr, ytr)
            ap = average_precision_score(ycal, m.score(Xcal, mode='self_only'))
        except Exception as e:
            print(f"      [K={k}] failed: {type(e).__name__}: {e}")
            ap = -np.inf
        trace[k] = None if ap == -np.inf else round(float(ap), 4)
        if ap > best_ap:
            best_ap, best_k = ap, k
    return best_k, trace


def run_benchmark():
    records = []

    datasets = ({k: v for k, v in DATASETS.items() if k in DS_FILTER}
                if DS_FILTER else DATASETS)
    desconocidos = [d for d in DS_FILTER if d not in DATASETS]
    if desconocidos:
        raise SystemExit(f"Unknown datasets: {desconocidos}")

    print(">>> EXP-1: BENCHMARK <<<")
    print(f"    device={'cuda' if CUDA else 'cpu'}  K_grid={K_GRID}  M={N_ESTIMATORS}")
    print(f"    Deep MS-SVDD with K={MS_SPHERES} initial spheres")
    print(f"    tag={TAG}  datasets={list(datasets)}")
    print(f"    output -> {OUT_CSV}\n")

    for ds_name, loader in datasets.items():
        print(f"\n=== {ds_name} " + "=" * 45)
        try:
            splits = {s: loader(seed=s) for s in SEEDS}
        except Exception as e:
            print(f"    [LOAD ERROR] {ds_name}: {e}")
            traceback.print_exc()
            continue

        d_in = splits[SEEDS[0]][0].shape[1]
        print(f"    d_in={d_in}  rho_train={splits[SEEDS[0]][3].mean():.4f}")

        models = {
            "DP-NSA":        None,   # handled separately (K selection)
            "Deep SVDD":     lambda s: DeepSVDDWrapper(d_in, seed=s),
            "Deep MS-SVDD":  lambda s: DeepMSSVDDWrapper(
                                 d_in, n_spheres=MS_SPHERES, seed=s),
            "ContrastiveAE": lambda s: ContrastiveAEWrapper(d_in, seed=s),
            "iForest":       lambda s: IsolationForestWrapper(random_state=s),
            "OCSVM":         lambda s: OCSVMWrapper(nu=0.1),
            "XGBoost":       lambda s: XGBoostBalancedWrapper(random_state=s),
            "TabularResNet": lambda s: TabResNetWrapper(input_dim=d_in, seed=s),
            "DeepSAD":       lambda s: DeepSADWrapper(input_dim=d_in, seed=s),
            "DevNet":        lambda s: DevNetWrapper(input_dim=d_in, seed=s),
        }

        for m_name, factory in models.items():
            aucs, prs, lats, ks, spheres = [], [], [], [], []

            for s in tqdm(SEEDS, desc=f"  {m_name}", leave=False):
                Xtr, Xcal, Xte, ytr, ycal, yte = splits[s]
                try:
                    if CUDA:
                        torch.cuda.reset_peak_memory_stats()

                    # ------------------------------------------------ DP-NSA
                    if m_name == "DP-NSA":
                        best_k, ktrace = select_k(d_in, Xtr, ytr, Xcal, ycal, s)
                        ks.append(best_k)
                        model = NSARepV23Ensemble(
                            d_in, n_estimators=N_ESTIMATORS, k=best_k, seed_start=s,
                            verbose=False, score_mode='self_only',
                            loss_mode='paper').fit(Xtr, ytr)

                        spec = spectral_diagnostics(model)

                        s_self, lat = _timed(
                            lambda: model.score(Xte, mode='self_only'), len(Xte))
                        s_disc, lat_d = _timed(
                            lambda: model.score(Xte, mode='discriminative'), len(Xte))
                        mem = torch.cuda.max_memory_allocated() / 1024**2 if CUDA else 0.0

                        for tag, sc, la in ((TAG_MAIN, s_disc, lat_d),
                                            (TAG_SELF, s_self, lat)):
                            auc = roc_auc_score(yte, sc)
                            pr = average_precision_score(yte, sc)
                            records.append(dict(
                                Dataset=ds_name, Method=tag, Seed=s,
                                AUROC=auc, AUPRC=pr, Latency_ms_per_sample=la,
                                Memory_MB=mem, K=best_k, M=N_ESTIMATORS,
                                K_trace=str(ktrace), n_spheres=np.nan,
                                d_in=d_in, n_test=len(yte),
                                rho_test=float(yte.mean()), Error="", **spec))
                            if tag == TAG_MAIN:
                                aucs.append(auc); prs.append(pr); lats.append(la)
                        continue

                    # --------------------------------------------- baselines
                    model = factory(s)
                    if m_name in SUPERVISED:
                        model.fit(Xtr, ytr)
                    elif m_name in NORMAL_ONLY:
                        model.fit(Xtr[ytr == 0])
                    else:
                        model.fit(Xtr)

                    score, lat = _timed(lambda: _infer(model, Xte), len(Xte))
                    mem = torch.cuda.max_memory_allocated() / 1024**2 if CUDA else 0.0
                    auc = roc_auc_score(yte, score)
                    pr = average_precision_score(yte, score)

                    # spheres left after pruning (Deep MS-SVDD)
                    nsp = getattr(model, "n_spheres_final", np.nan)
                    if m_name == "Deep MS-SVDD" and not np.isnan(nsp):
                        spheres.append(nsp)

                    records.append(dict(
                        Dataset=ds_name, Method=m_name, Seed=s,
                        AUROC=auc, AUPRC=pr, Latency_ms_per_sample=lat,
                        Memory_MB=mem, K=None, M=None, K_trace="",
                        n_spheres=nsp, d_in=d_in, n_test=len(yte),
                        rho_test=float(yte.mean()), Error="", **_empty_spec()))
                    aucs.append(auc); prs.append(pr); lats.append(lat)

                except Exception as e:
                    print(f"    [ERROR] {ds_name}/{m_name}/seed={s}: {e}")
                    traceback.print_exc()
                    records.append(dict(
                        Dataset=ds_name, Method=m_name, Seed=s,
                        AUROC=np.nan, AUPRC=np.nan, Latency_ms_per_sample=np.nan,
                        Memory_MB=np.nan, K=None, M=None, K_trace="",
                        n_spheres=np.nan, d_in=d_in, n_test=len(yte),
                        rho_test=float(yte.mean()),
                        Error=f"{type(e).__name__}: {e}", **_empty_spec()))

            if aucs:
                extra = ""
                if ks:
                    extra += f" | K: {ks}"
                if spheres:
                    extra += f" | spheres: {np.mean(spheres):.1f}"
                print(f"    {m_name:15s} | AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                      f"| PR {np.mean(prs):.4f}±{np.std(prs):.4f} "
                      f"| {np.mean(lats):.4f} ms/sample{extra}")

            pd.DataFrame(records).to_csv(
                os.path.join(RESULTS_DIR, "exp1_benchmark_results.csv"), index=False)

        # spectral summary of the dataset, to follow the run
        sp = pd.DataFrame(records)
        sp = sp[(sp.Dataset == ds_name) & (sp.Method == TAG_MAIN)]
        if not sp.empty and not sp.r_ratio.isna().all():
            print(f"    [spectrum] r_eff={sp.r_eff_mean.mean():.1f} of "
                  f"d_lat={int(sp.d_lat.iloc[0])}  "
                  f"ratio={sp.r_ratio.mean():.2f}  (d_in={d_in})")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)

    n_err = int((df.Error.fillna("") != "").sum())
    print(f"\nSaved to: {OUT_CSV}")
    if n_err:
        print(f"[WARNING] {n_err} rows with errors. See the 'Error' column.")

    resumen(df)


def resumen(df):
    """Console summary, printed at the end of a run and after --merge."""
    ok = df.dropna(subset=["AUPRC"])
    if ok.empty:
        print("No valid results.")
        return

    piv = ok.groupby(['Dataset', 'Method'])['AUPRC'].mean().unstack()
    print("\n--- Mean rank (AUPRC, 1 = best) ---")
    print(piv.rank(axis=1, ascending=False).mean().sort_values().round(2).to_string())

    print("\n--- Effective rank of the latent covariance ---")
    sp = ok[ok.Method == TAG_MAIN].groupby("Dataset")[
        ["d_in", "d_lat", "r_eff_mean", "r_ratio", "cond_mean"]].mean()
    print(sp.round(2).to_string())
    print("\n  r_ratio close to 1: the covariance fills the latent space.")
    print("  low r_ratio: the matrix is singular by construction, not for")
    print("  lack of samples.")
    print("")

    ms = ok[ok.Method == "Deep MS-SVDD"]
    if not ms.empty and not ms.n_spheres.isna().all():
        print("\n--- Deep MS-SVDD: spheres after pruning ---")
        print(ms.groupby("Dataset")["n_spheres"].mean().round(1).to_string())


def merge():
    """Merge the partial CSVs of a parallel run."""
    import glob
    files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "exp1_benchmark_*.csv"))
             if not f.endswith("exp1_benchmark_results.csv")]
    if not files:
        print("No partial files to merge.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = os.path.join(RESULTS_DIR, "exp1_benchmark_results.csv")
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
        run_benchmark()
