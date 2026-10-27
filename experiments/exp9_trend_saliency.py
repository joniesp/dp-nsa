"""Experiment 9: trend test over the label budget, and saliency control (Table 5).

    python experiments/exp9_trend_saliency.py --parte tendencia   # part A only
    python experiments/exp9_trend_saliency.py --parte saliencia   # part B only
    python experiments/exp9_trend_saliency.py                     # both

Part A, trend (seconds, CPU, reads exp6_escasez_results.csv). Page's L test
for ordered alternatives, per method, with datasets as blocks and label
fractions as treatments: is performance monotonically increasing with the
label budget? Then a bootstrap over datasets (10,000 resamples) of the
difference in slope (AUPRC per decade of budget) between DP-NSA and each
other method, i.e. whether DP-NSA degrades less.

Part B, saliency control (minutes, GPU, Cardiotocography). Whether the
attribution of the geometric score to the clinical markers is specific to
DP-NSA or shared by any model trained on the data, comparing on the same
split and with the same normalisation:

    DP-NSA          gradient of the geometric score
    Tabular ResNet  gradient of the probability of the positive class
    XGBoost         SHAP values (gain importance if shap is not installed)
"""

import os
import sys
import argparse
import traceback

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dpnsa.paths import RUNS_DIR, resultado  # noqa: E402
RESULTS_DIR = RUNS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [int(s) for s in os.environ.get("DPNSA_SEEDS", "100,101,102,103,104").split(",")]


# ==========================================================================
# Part A: trend
# ==========================================================================
def page_trend(matriz):
    """Page's L test for ordered alternatives (normal approximation).

    matriz: rows = blocks (datasets); columns = treatments in the order of the
    alternative, from the one expected lowest to the one expected highest.
    Returns (L, z, p).
    """
    X = np.asarray(matriz, dtype=float)
    n, k = X.shape
    # ranks within each block, ascending
    R = np.apply_along_axis(
        lambda f: pd.Series(f).rank(method="average").values, 1, X)
    pesos = np.arange(1, k + 1)
    L = float((R * pesos).sum())

    mu = n * k * (k + 1) ** 2 / 4.0
    var = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144.0
    z = (L - mu) / np.sqrt(var)
    from scipy.stats import norm
    return L, z, float(1 - norm.cdf(z))


def pendiente_log(fracciones, valores):
    """Slope of AUPRC against log10 of the label fraction (the fractions are geometric)."""
    x = np.log10(np.asarray(fracciones, dtype=float))
    y = np.asarray(valores, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def parte_tendencia():
    csv = resultado("exp6_escasez_results.csv")
    if not os.path.exists(csv):
        print(f"[SKIPPED] {csv} not found")
        return

    ok = pd.read_csv(csv).dropna(subset=["AUPRC"])
    ok = ok[~ok.invariant]                       # methods without labels have no trend
    fracs = sorted(ok.Frac.unique())             # ascending: 0.01 ... 1.0
    metodos = [m for m in ["DP-NSA", "DP-NSA (self-only)", "XGBoost",
                           "TabularResNet", "DeepSAD", "DevNet"]
               if m in ok.Method.unique()]

    print("\n" + "=" * 74)
    print(" PART A — TREND OVER THE LABEL BUDGET")
    print("=" * 74)
    print("\nPage's L test. The alternative is that performance INCREASES with")
    print("the budget: columns go from the smallest to the largest fraction.")
    print("Each dataset is a block.\n")

    filas = []
    for m in metodos:
        piv = (ok[ok.Method == m]
               .groupby(["Dataset", "Frac"])["AUPRC"].mean().unstack())
        piv = piv[fracs].dropna()
        if len(piv) < 3:
            print(f"  {m:22s} not enough blocks")
            continue

        L, z, p = page_trend(piv.values)
        pend = np.mean([pendiente_log(fracs, fila) for fila in piv.values])
        rango = piv[fracs[-1]].mean() - piv[fracs[0]].mean()

        filas.append(dict(Method=m, n_bloques=len(piv), L=round(L, 1),
                          z=round(z, 2), p=p, pendiente=round(pend, 4),
                          delta=round(rango, 4)))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {m:22s} L={L:7.1f}  z={z:5.2f}  p={p:.2e} {sig:3s}  "
              f"slope={pend:+.4f}")

    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(RESULTS_DIR, "exp9_tendencia.csv"), index=False)

    print("\n  The slope is the change in AUPRC per order of magnitude of")
    print("  label budget: the larger in absolute value, the more the method")
    print("  depends on the number of labels.\n")

    # Difference in slope against DP-NSA: bootstrap over the datasets (blocks)
    if "DP-NSA" not in metodos:
        return
    print("=" * 74)
    print(" Difference in slope against DP-NSA, bootstrap over the")
    print(" datasets, 10000 resamples")
    print("=" * 74 + "\n")

    rng = np.random.RandomState(0)
    base = (ok[ok.Method == "DP-NSA"]
            .groupby(["Dataset", "Frac"])["AUPRC"].mean().unstack()[fracs])

    for m in [x for x in metodos if x != "DP-NSA"]:
        otro = (ok[ok.Method == m]
                .groupby(["Dataset", "Frac"])["AUPRC"].mean().unstack()[fracs])
        comun = base.index.intersection(otro.index)
        b, o = base.loc[comun], otro.loc[comun]

        difs = []
        for _ in range(10000):
            idx = rng.randint(0, len(comun), len(comun))
            pb = np.mean([pendiente_log(fracs, b.values[i]) for i in idx])
            po = np.mean([pendiente_log(fracs, o.values[i]) for i in idx])
            difs.append(pb - po)
        difs = np.array(difs)
        lo, hi = np.percentile(difs, [2.5, 97.5])
        # two-sided: share of resamples on the less frequent side of zero
        p_boot = 2 * min((difs > 0).mean(), (difs < 0).mean())

        print(f"  vs {m:22s} diff={difs.mean():+.4f}  "
              f"CI95=[{lo:+.4f}, {hi:+.4f}]  p={p_boot:.4f}")

    print("\n  A negative difference means DP-NSA has a SMALLER slope, i.e. it")
    print("  degrades less. A CI that excludes zero establishes the")
    print("  interaction.")


# ==========================================================================
# Part B: saliency control
# ==========================================================================
def _saliencia_dpnsa(model, X, device):
    """|gradient x input| of the geometric score of the first clone, normalised per sample."""
    import torch
    det = model.models[0]
    enc = det.enc
    enc.eval()
    comps = det.params["P0"]
    out = []
    for fila in X:
        x = torch.tensor(fila, dtype=torch.float32, device=device).unsqueeze(0)
        x.requires_grad_(True)
        enc.zero_grad(set_to_none=True)
        z = enc(x)
        if isinstance(z, tuple):
            z = z[0]
        ds = []
        for c in comps:
            diff = z - c["mu"]
            d2 = (diff @ c["Sinv"] * diff).sum(dim=1)
            ds.append(torch.sqrt(torch.clamp(d2, min=1e-12)) - c["R"])
        torch.stack(ds, dim=1).min(dim=1)[0].sum().backward()
        a = np.abs(x.grad.detach().cpu().numpy()[0] * fila)
        out.append(a / (a.max() + 1e-12))
    return np.array(out)


def _saliencia_resnet(model, X, device):
    """|gradient x input| of the output of the ResNet, normalised per sample."""
    import torch
    net = getattr(model, "model", getattr(model, "net", None))
    if net is None:
        return None
    net.eval()
    out = []
    for fila in X:
        x = torch.tensor(fila, dtype=torch.float32, device=device).unsqueeze(0)
        x.requires_grad_(True)
        net.zero_grad(set_to_none=True)
        y = net(x)
        y = y[0] if isinstance(y, tuple) else y
        y.sum().backward()
        a = np.abs(x.grad.detach().cpu().numpy()[0] * fila)
        out.append(a / (a.max() + 1e-12))
    return np.array(out)


def _saliencia_xgb(model, X, nombres):
    """SHAP values; without shap, the gain importance (global, not per sample)."""
    booster = getattr(model, "model", getattr(model, "clf", None))
    try:
        import shap
        expl = shap.TreeExplainer(booster)
        v = expl.shap_values(X)
        if isinstance(v, list):
            v = v[1]
        a = np.abs(v)
        return a / (a.max(axis=1, keepdims=True) + 1e-12), "SHAP"
    except Exception:
        try:
            imp = booster.get_booster().get_score(importance_type="gain")
        except Exception:
            return None, None
        v = np.array([imp.get(f"f{i}", 0.0) for i in range(len(nombres))])
        v = v / (v.max() + 1e-12)
        return np.tile(v, (len(X), 1)), "gain"


def parte_saliencia(dataset="Cardio", n_muestras=100):
    import torch
    from dpnsa.datasets import prepare_cardiotocography
    from dpnsa.ensemble import NSARepV23Ensemble
    from baselines.shallow import XGBoostBalancedWrapper
    from baselines.deep import TabResNetWrapper

    print("\n" + "=" * 74)
    print(" PART B — SALIENCY CONTROL")
    print("=" * 74)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xcal, Xte, ytr, ycal, yte = prepare_cardiotocography(seed=100)
    d_in = Xtr.shape[1]

    # real feature names, when available
    ruta = os.path.join(project_root, "data", "cardiotocography", "X.csv")
    if os.path.exists(ruta):
        cols = pd.read_csv(ruta, nrows=1).columns.tolist()
        nombres = cols if len(cols) == d_in else [f"V{i}" for i in range(d_in)]
    else:
        nombres = [f"V{i}" for i in range(d_in)]

    rng = np.random.RandomState(100)
    idx_a = rng.choice(np.where(yte == 1)[0],
                       size=min(n_muestras, int(yte.sum())), replace=False)
    idx_n = rng.choice(np.where(yte == 0)[0],
                       size=min(n_muestras, int((yte == 0).sum())), replace=False)
    print(f"    {len(idx_a)} anomalies and {len(idx_n)} normal samples from the "
          f"test split\n")

    tablas = {}

    # --- DP-NSA -------------------------------------------------------
    try:
        m = NSARepV23Ensemble(d_in, n_estimators=5, k=1, seed_start=100,
                              verbose=False, score_mode='self_only',
                              loss_mode='paper').fit(Xtr, ytr)
        A = _saliencia_dpnsa(m, Xte[idx_a], device)
        N = _saliencia_dpnsa(m, Xte[idx_n], device)
        tablas["DP-NSA"] = A.mean(0) - N.mean(0)
        print("    [OK] DP-NSA, gradient of the geometric score")
    except Exception as e:
        print(f"    [ERROR] DP-NSA: {e}")
        traceback.print_exc()

    # --- Tabular ResNet -----------------------------------------------
    try:
        m = TabResNetWrapper(input_dim=d_in, seed=100).fit(Xtr, ytr)
        A = _saliencia_resnet(m, Xte[idx_a], device)
        N = _saliencia_resnet(m, Xte[idx_n], device)
        if A is not None:
            tablas["Tabular ResNet"] = A.mean(0) - N.mean(0)
            print("    [OK] Tabular ResNet, gradient of the probability")
        else:
            print("    [SKIPPED] Tabular ResNet: inner module not found")
    except Exception as e:
        print(f"    [ERROR] Tabular ResNet: {e}")

    # --- XGBoost ------------------------------------------------------
    try:
        m = XGBoostBalancedWrapper(random_state=100).fit(Xtr, ytr)
        SA, tipo = _saliencia_xgb(m, Xte[idx_a], nombres)
        SN, _ = _saliencia_xgb(m, Xte[idx_n], nombres)
        if SA is not None:
            tablas[f"XGBoost ({tipo})"] = SA.mean(0) - SN.mean(0)
            print(f"    [OK] XGBoost, {tipo}")
            if tipo == "gain":
                print("         The gain importance is GLOBAL: it does not")
                print("         distinguish anomalies from normal samples, so")
                print("         its difference is zero by construction.")
                print("         Install shap for a valid comparison.")
    except Exception as e:
        print(f"    [ERROR] XGBoost: {e}")

    if not tablas:
        print("\n    No method produced attributions.")
        return

    tab = pd.DataFrame(tablas, index=nombres)
    tab["rank_DPNSA"] = tab["DP-NSA"].rank(ascending=False).astype(int)
    tab = tab.sort_values("DP-NSA", ascending=False)
    tab.round(4).to_csv(os.path.join(RESULTS_DIR, "exp9_saliencia_control.csv"))

    print("\n=== Attribution difference, anomalies minus normal ===")
    print(tab.round(4).to_string())

    # --- agreement ------------------------------------------------------
    print("\n=== Agreement between methods ===")
    from itertools import combinations
    from scipy.stats import spearmanr
    ms = [c for c in tab.columns if c != "rank_DPNSA"]
    for a, b in combinations(ms, 2):
        rho, p = spearmanr(tab[a], tab[b])
        top_a = set(tab[a].nlargest(5).index)
        top_b = set(tab[b].nlargest(5).index)
        print(f"  {a:20s} vs {b:20s}  rho={rho:+.3f} (p={p:.3f})  "
              f"shared in top-5: {len(top_a & top_b)}/5")

    clinicas = [v for v in ("ASTV", "DP", "ALTV") if v in tab.index]
    if clinicas:
        print(f"\n=== Rank of the clinical markers {clinicas} ===")
        for m in ms:
            r = tab[m].rank(ascending=False)
            print(f"  {m:20s} " +
                  "  ".join(f"{v}={int(r[v])}" for v in clinicas))
        print("\n  If all three methods rank the markers at the top, the")
        print("  attribution is shared by the data; if only DP-NSA does,")
        print("  it is specific to the geometric score.")
        print("")
        print("")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", choices=["tendencia", "saliencia", "ambas"],
                    default="ambas")
    ap.add_argument("--n", type=int, default=100,
                    help="samples per class in the saliency control")
    args = ap.parse_args()

    if args.parte in ("tendencia", "ambas"):
        try:
            parte_tendencia()
        except Exception as e:
            print(f"[ERROR] trend: {type(e).__name__}: {e}")
            traceback.print_exc()

    if args.parte in ("saliencia", "ambas"):
        try:
            parte_saliencia(n_muestras=args.n)
        except Exception as e:
            print(f"[ERROR] saliency: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\nOutput in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
