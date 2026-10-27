"""Figures 4-9 of the paper (Section 5), generated from the result CSVs only.

    Figure_4  label scarcity: curves and relative degradation       (exp6)
    Figure_5  contribution of the repulsion term, per dataset        (exp2)
    Figure_6  contribution of the learned covariance (anisotropy)    (exp2)
    Figure_7  granularity and repertoire size                        (exp5)
    Figure_8  shrinkage sensitivity and conditioning                 (exp7)
    Figure_9  effective rank of the latent covariance                (exp1)

Figure 10 (latent projection) is produced by experiments/exp3_latent_projection.py.
Input: $DPNSA_PAPER_RESULTS (default results/paper). Output: $DPNSA_OUT/figures.

    python analysis/make_figures.py
    python analysis/make_figures.py --fig 3,6      # internal numbers 1-6 = Figures 4-9
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.environ.get("DPNSA_PAPER_RESULTS", os.path.join(ROOT, "results", "paper"))
FIGDIR = os.path.join(os.environ.get("DPNSA_OUT", os.path.join(ROOT, "paper")), "figures")
os.makedirs(FIGDIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.8",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})

# Colour, marker and line width are redundant so that the figures remain
# readable in greyscale.
STYLE = {
    "DP-NSA":             dict(color="#0d3b66", marker="o", lw=2.6, ms=6,   z=10),
    "DP-NSA (self-only)": dict(color="#3d8ec9", marker="s", lw=2.0, ms=5,   z=9),
    "DP-NSA (lambda=0)":  dict(color="#7fb3d5", marker=None, lw=1.4, z=3),
    "XGBoost":            dict(color="#b03a2e", marker="^", lw=1.8, ms=4.5, z=7),
    "TabularResNet":      dict(color="#d68910", marker="v", lw=1.6, ms=4.5, z=6),
    "DeepSAD":            dict(color="#6c8c3b", marker="D", lw=1.6, ms=4,   z=6),
    "DevNet":             dict(color="#7d3c98", marker="P", lw=1.6, ms=5,   z=5),
    "Deep SVDD":          dict(color="#566573", marker=None, lw=1.4, z=3),
    "Deep MS-SVDD":       dict(color="#85929e", marker=None, lw=1.4, z=3),
    "iForest":            dict(color="#909497", marker=None, lw=1.4, z=3),
    "OCSVM":              dict(color="#aeb6bf", marker=None, lw=1.4, z=3),
}

LABEL = {"DP-NSA (lambda=0)": "DP-NSA ($\\lambda{=}0$)"}

DS_ORDER = ["CreditCard", "Satimage", "Mammography", "Musk", "Shuttle",
            "Annthyroid", "Arrhythmia", "Cardio", "Ionosphere", "ECG5000"]


def load(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    if not files:
        raise FileNotFoundError(f"{pattern} not found in {RESULTS}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def save(fig, n, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"Figure_{n + 3}_{name}.{ext}"), dpi=400)
    print(f"  [OK] Figure_{n + 3}_{name}.pdf / .png")
    plt.close(fig)


def st(m):
    d = dict(STYLE.get(m, dict(color="#888888", marker="o", lw=1.4, ms=4)))
    d.pop("z", None)
    return d


def lab(m):
    return LABEL.get(m, m)


# ==========================================================================
# Figure 4 (internal 1): label scarcity
# ==========================================================================
def figura_1():
    ok = load("exp6_escasez_results.csv").dropna(subset=["AUPRC"])
    piv = ok.groupby(["Frac", "Method"])["AUPRC"].mean().unstack()
    fr = sorted(piv.index)
    x = [f * 100 for f in fr]

    sup = [m for m in ["DP-NSA", "DP-NSA (self-only)", "XGBoost",
                       "TabularResNet", "DeepSAD", "DevNet"] if m in piv.columns]
    inv = [m for m in ["DP-NSA (lambda=0)", "Deep SVDD", "Deep MS-SVDD",
                       "iForest", "OCSVM"] if m in piv.columns]

    lo_v = min(piv[c].min() for c in sup + inv)
    hi_v = max(piv[c].max() for c in sup + inv)
    YMIN, YMAX = lo_v - 0.04, hi_v + 0.06

    fig = plt.figure(figsize=(12.0, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylim(YMIN, YMAX)

    cross = None
    for i in range(len(fr) - 1):
        a = piv.loc[fr[i], "DP-NSA"] - piv.loc[fr[i], "XGBoost"]
        b = piv.loc[fr[i + 1], "DP-NSA"] - piv.loc[fr[i + 1], "XGBoost"]
        if a > 0 >= b:
            cross = x[i] * 1.35
    if cross:
        ax.axvspan(x[0], cross, color="#0d3b66", alpha=0.05, zorder=1)

    ref = sorted([(piv.loc[fr[0], m], m) for m in inv])
    sep = 0.030 * (YMAX - YMIN)
    lab_y = []
    for v, m in ref:
        lab_y.append(v if not lab_y else max(v, lab_y[-1] + sep))
    for (v, m), yy in zip(ref, lab_y):
        ax.axhline(v, color=STYLE[m]["color"], ls=(0, (4, 3)), lw=1.2,
                   zorder=2, alpha=0.9)
        ax.annotate(lab(m), xy=(0.995, yy), xycoords=("axes fraction", "data"),
                    va="center", ha="right", fontsize=7.5,
                    color=STYLE[m]["color"],
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2),
                    zorder=6)

    for m in sup:
        ax.plot(x, [piv.loc[f, m] for f in fr], **st(m), label=lab(m),
                zorder=STYLE[m]["z"], markeredgecolor="white",
                markeredgewidth=0.6)

    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(["1", "5", "10", "25", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Labelled anomalies retained for training (%)")
    ax.set_ylabel("AUPRC, mean over 10 datasets")
    ax.set_title("(a) Performance under decreasing supervision", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=2.0,
              bbox_to_anchor=(0.015, 0.995))

    ax = fig.add_subplot(gs[0, 1])
    rel = {m: 100 * (piv.loc[0.01, m] - piv.loc[1.0, m]) / piv.loc[1.0, m]
           for m in sup}
    s = pd.Series(rel).sort_values(ascending=False)
    y = np.arange(len(s))

    ax.barh(y, s.values, color=[STYLE[m]["color"] for m in s.index],
            edgecolor="white", lw=0.8, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([lab(m) for m in s.index])
    ax.axvline(0, color="0.2", lw=1.0, zorder=4)

    for i, (m, v) in enumerate(s.items()):
        if abs(v) > 6:
            ax.text(v + 1.0, i, f"{v:.1f}%", va="center", ha="left",
                    fontsize=8.5, fontweight="bold", color="white")
        else:
            ax.text(v - 1.0, i, f"{v:.1f}%", va="center", ha="right",
                    fontsize=8.5, fontweight="bold", color=STYLE[m]["color"])
        ax.text(1.5, i, f"{piv.loc[1.0, m]:.3f}$\\to${piv.loc[0.01, m]:.3f}",
                va="center", ha="left", fontsize=8, color="0.35")

    ax.set_xlim(min(s.values) * 1.12, 30)
    ax.set_xlabel("Relative change in AUPRC (%)")
    ax.set_title("(b) Degradation from 100% to 1% of labels", loc="left",
                 fontweight="bold")
    ax.grid(axis="y", visible=False)

    save(fig, 1, "label_scarcity")


# ==========================================================================
# Figures 5 and 6 (internal 2, 3): shift between two variants
# ==========================================================================
def _dumbbell(piv, col_a, col_b, etiqueta_a, etiqueta_b, titulo, num, nombre):
    """Dumbbell plot: the segment length is the contribution of the mechanism,
    and the absolute values stay readable on the same axis."""
    eff = (piv[col_b] - piv[col_a]).sort_values()
    ds = list(eff.index)
    y = np.arange(len(ds))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, d in enumerate(ds):
        a, b = piv.loc[d, col_a], piv.loc[d, col_b]
        col = "#b03a2e" if b < a else "#0d3b66"
        ax.plot([a, b], [i, i], "-", color=col, lw=3.0,
                solid_capstyle="round", zorder=3, alpha=0.85)
        ax.plot(a, i, "o", ms=7, mfc="white", mec="0.35", mew=1.5, zorder=5)
        ax.plot(b, i, "o", ms=7, color=col, mec="white", mew=1.2, zorder=6)
        ax.text(1.045, i, f"{b - a:+.3f}", va="center", ha="left",
                fontsize=8.5, fontweight="bold", color=col)

    ax.set_yticks(y)
    ax.set_yticklabels(ds)
    ax.set_xlabel("AUPRC")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(ds) - 0.4)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", visible=False)
    ax.annotate("$\\Delta$", xy=(1.045, len(ds) - 0.4), ha="left", va="bottom",
                fontsize=9, fontweight="bold", color="0.3")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec="0.35",
               mew=1.5, label=etiqueta_a),
        Line2D([], [], marker="o", ls="", ms=7, color="#0d3b66",
               label=etiqueta_b)],
        loc="upper left", bbox_to_anchor=(0.30, 0.30), framealpha=0.95)
    ax.set_title(titulo, loc="left", fontweight="bold")

    save(fig, num, nombre)


def figura_2():
    df = load("exp2_ablacion_results.csv")
    if "k_protocol" in df.columns:
        df = df[df.k_protocol == "calibrated"]
    piv = df.groupby(["Dataset", "Variant"])["AUPRC"].mean().unstack()
    _dumbbell(piv, "OneClass", "Full",
              "Without repulsion ($\\lambda{=}0$)", "Full model",
              "Contribution of the repulsion term $\\mathcal{L}_{rep}$",
              2, "repulsion_effect")


def figura_3():
    """Spherical against ellipsoidal boundary with a single component (K = 1)."""
    df = load("exp2_ablacion_results.csv")
    if "k_protocol" not in df.columns:
        raise FileNotFoundError("exp2 has no k_protocol column: geometry "
                                "block missing")
    b = df[df.k_protocol == "fixed_1"]
    piv = b.groupby(["Dataset", "Variant"])["AUPRC"].mean().unstack()
    if not {"Anisotropic_K1", "Isotropic_K1"} <= set(piv.columns):
        raise FileNotFoundError("geometry variants missing")
    _dumbbell(piv, "Isotropic_K1", "Anisotropic_K1",
              "Spherical boundary ($\\alpha\\to1$)", "Anisotropic ellipsoid",
              "Contribution of the learned covariance ($K=1$)",
              3, "anisotropy_effect")


# ==========================================================================
# Figure 7 (internal 4): granularity and repertoire
# ==========================================================================
def figura_4():
    ok = load("exp5_sensibilidad_results.csv").dropna(subset=["AUPRC"])
    a = ok[ok.Parte == "A"].copy()
    for c in ("K", "M", "AUPRC"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.dropna(subset=["K", "M", "AUPRC"])
    a["K"] = a["K"].astype(int)
    a["M"] = a["M"].astype(int)

    Ks = sorted(int(k) for k in a["K"].unique())
    Ms = sorted(int(m) for m in a["M"].unique())
    Mmin, Mmax = min(Ms), max(Ms)
    ds_list = [d for d in DS_ORDER if d in a.Dataset.unique()] or \
              sorted(a.Dataset.unique())
    ds_col = {"CreditCard": "#0d3b66", "Mammography": "#b03a2e",
              "Shuttle": "#6c8c3b", "Arrhythmia": "#d68910"}
    ds_mk = {"CreditCard": "o", "Mammography": "s",
             "Shuttle": "^", "Arrhythmia": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))

    ax = axes[0]
    for ds in ds_list:
        s = a[(a.Dataset == ds) & (a.M == Mmax)].groupby("K")["AUPRC"].mean()
        if 1 not in s.index:
            continue
        base = float(s.loc[1])
        ax.plot([int(k) for k in s.index],
                [100.0 * float(v) / base for v in s.values],
                marker=ds_mk.get(ds, "o"), ms=5.5, lw=1.8,
                color=ds_col.get(ds, "gray"), label=ds,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(100, color="0.3", ls=(0, (4, 3)), lw=1.1)
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 50, 70, 100])
    ax.set_yticklabels(["20", "30", "50", "70", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Granularity $K$ (number of ellipsoids)")
    ax.set_ylabel("AUPRC relative to $K=1$ (%, log scale)")
    ax.set_title("(a) Higher granularity rarely helps", loc="left",
                 fontweight="bold")
    ax.legend(loc="lower left")

    ax = axes[1]
    sd = a.groupby(["Dataset", "K", "M"])["AUPRC"].std().reset_index()
    w = 0.20
    xs = np.arange(len(ds_list))
    kc = {1: "#bcd4e6", 3: "#7fb3d5", 5: "#3d8ec9", 10: "#0d3b66"}
    for j, K in enumerate(Ks):
        vals = []
        for ds in ds_list:
            s = sd[(sd.Dataset == ds) & (sd.K == K)].set_index("M")["AUPRC"]
            vals.append(s.loc[Mmin] / s.loc[Mmax]
                        if (Mmax in s.index and s.loc[Mmax] > 0) else np.nan)
        ax.bar(xs + (j - 1.5) * w, vals, width=w * 0.9, color=kc.get(K, "gray"),
               edgecolor="white", lw=0.7, label=f"$K={K}$", zorder=3)
        for xi, v in zip(xs + (j - 1.5) * w, vals):
            if not np.isnan(v) and v > 1.6:
                ax.text(xi, v + 0.08, f"{v:.1f}$\\times$", ha="center",
                        fontsize=7.5, fontweight="bold", color=kc.get(K, "gray"))
    ax.axhline(1.0, color="0.25", lw=1.2, zorder=4)
    ax.annotate("no effect", xy=(-0.42, 1.10), ha="left", va="bottom",
                fontsize=8, style="italic", color="0.35")
    ax.set_xticks(xs)
    ax.set_xticklabels(ds_list)
    ax.set_ylabel("SD reduction: $\\mathrm{SD}(M{=}%d)/\\mathrm{SD}(M{=}%d)$"
                  % (Mmin, Mmax))
    ax.set_ylim(0, 6.0)
    ax.set_title("(b) Variance control is conditional", loc="left",
                 fontweight="bold")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", title="Granularity", title_fontsize=8.5, ncol=2)

    fig.tight_layout()
    save(fig, 4, "granularity_repertoire")


# ==========================================================================
# Figure 8 (internal 5): shrinkage sensitivity
# ==========================================================================
def figura_5():
    ok = load("exp7_shrinkage_*.csv").dropna(subset=["AUPRC"])
    ds_list = ok.groupby("Dataset")["N_over_d"].first().sort_values().index.tolist()
    n = len(ds_list)
    kcol = {1: "#0d3b66", 3: "#d68910", 5: "#6c8c3b", 10: "#7d3c98"}
    kmk = {1: "o", 3: "s", 5: "D", 10: "P"}

    fig, axes = plt.subplots(2, n, figsize=(3.25 * n, 5.9), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.0],
                                          "hspace": 0.16, "wspace": 0.30})
    if n == 1:
        axes = axes.reshape(2, 1)

    cmin = ok.cond_number.min() * 0.4
    cmax = ok.cond_number.max() * 3.0

    for j, dsn in enumerate(ds_list):
        s = ok[ok.Dataset == dsn]
        nd = s.N_over_d.iloc[0]

        ax = axes[0, j]
        ymin, ymax = [], []
        for K, g in s.groupby("K"):
            m = g.groupby("alpha")["AUPRC"].mean()
            e = g.groupby("alpha")["AUPRC"].std()
            ax.plot(m.index, m.values, marker=kmk.get(K, "o"), ms=4.5, lw=1.8,
                    color=kcol.get(K, "gray"), label=f"$K={K}$",
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
            ax.fill_between(m.index, m - e, m + e, color=kcol.get(K, "gray"),
                            alpha=0.10, lw=0, zorder=2)
            ymin.append((m - e).min())
            ymax.append((m + e).max())

        lo, hi = min(ymin), max(ymax)
        pad = (hi - lo) * 0.18
        ax.set_ylim(lo - pad, hi + pad * 2.2)

        used = []
        for K, g in s.groupby("K"):
            m = g.groupby("alpha")["AUPRC"].mean()
            bx, by = float(m.idxmax()), float(m.max())
            off = 0.0
            while any(abs(bx - ux) < 0.06 and abs(by + off - uy) < (hi - lo) * 0.07
                      for ux, uy in used):
                off += (hi - lo) * 0.075
            ax.plot([bx], [by + off], marker="*", ms=11, color=kcol.get(K, "gray"),
                    markeredgecolor="white", markeredgewidth=0.8, zorder=7,
                    clip_on=False)
            used.append((bx, by + off))

        ax.axvline(0.1, color="0.3", ls=(0, (4, 3)), lw=1.1, zorder=1)
        ax.set_xlim(-0.06, 0.96)
        nd_txt = f"{nd:.0f}" if nd >= 10 else f"{nd:.1f}"
        ax.set_title(f"{dsn}\n$N/d_{{in}} = {nd_txt}$", fontsize=10)
        if j == 0:
            ax.set_ylabel("AUPRC")
            ax.legend(loc="lower right", handlelength=1.6)

        ax = axes[1, j]
        for K, g in s.groupby("K"):
            c = g.groupby("alpha")["cond_number"].mean()
            ax.plot(c.index, c.values, marker=kmk.get(K, "o"), ms=3.8, lw=1.6,
                    color=kcol.get(K, "gray"), markeredgecolor="white",
                    markeredgewidth=0.5)
        ax.axvline(0.1, color="0.3", ls=(0, (4, 3)), lw=1.1)
        ax.set_yscale("log")
        ax.set_ylim(cmin, cmax)
        ax.set_xlim(-0.06, 0.96)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xlabel("Shrinkage coefficient $\\alpha$")
        if j == 0:
            ax.set_ylabel("Condition number of $\\hat{\\Sigma}_k$")
        else:
            ax.tick_params(labelleft=False)

    axes[0, 0].annotate("default $\\alpha$", xy=(0.1, axes[0, 0].get_ylim()[1]),
                        xytext=(3, -3), textcoords="offset points",
                        fontsize=7.5, color="0.35", va="top", ha="left")
    fig.legend(handles=[Line2D([], [], marker="*", ls="", ms=11, color="0.4",
                               markeredgecolor="white")],
               labels=["optimum per configuration"], loc="lower center",
               bbox_to_anchor=(0.5, -0.035), frameon=False)

    save(fig, 5, "shrinkage_sensitivity")


# ==========================================================================
# Figure 9 (internal 6): effective rank of the latent covariance
# ==========================================================================
def figura_6():
    """Effective rank against input dimensionality, both on a log scale. Points
    on the diagonal would mean the effective rank follows the input
    dimensionality."""
    df = load("exp1_benchmark_results.csv")
    sp = df[df.Method == "DP-NSA"].groupby("Dataset")[
        ["d_in", "d_lat", "r_eff_mean", "r_ratio"]].mean().dropna()
    if sp.empty:
        raise FileNotFoundError("exp1 has no spectral diagnostics")

    d_lat = int(sp.d_lat.iloc[0])
    orden = [d for d in DS_ORDER if d in sp.index] or list(sp.index)
    sp = sp.loc[orden]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.axhline(d_lat, color="0.35", ls=(0, (4, 3)), lw=1.3, zorder=2)
    ax.annotate(f"latent dimension $d_{{lat}} = {d_lat}$",
                xy=(0.99, d_lat), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color="0.35")

    lo = min(sp.d_in.min(), sp.r_eff_mean.min()) * 0.6
    hi = max(sp.d_in.max(), d_lat) * 1.6
    ax.plot([lo, hi], [lo, hi], color="0.75", ls=":", lw=1.2, zorder=1)
    ax.annotate("$r_{\\mathrm{eff}} = d_{in}$", xy=(hi * 0.5, hi * 0.5),
                fontsize=8.5, color="0.6", rotation=30,
                ha="center", va="bottom")

    ax.scatter(sp.d_in, sp.r_eff_mean, s=70, color="#0d3b66",
               edgecolor="white", lw=1.0, zorder=5)

    # Fixed label offsets (several datasets share almost the same d_in)
    OFF = {"Mammography": (-4, -17), "Annthyroid": (2, 11),
           "Shuttle": (30, -5), "Cardio": (0, -17),
           "CreditCard": (20, 8), "Ionosphere": (-16, -15),
           "Satimage": (12, 10), "ECG5000": (-6, -17),
           "Musk": (0, 11), "Arrhythmia": (0, 11)}
    for ds, r in sp.iterrows():
        ax.annotate(ds, xy=(r.d_in, r.r_eff_mean),
                    xytext=OFF.get(ds, (0, 10)), textcoords="offset points",
                    ha="center", fontsize=8, color="0.3")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Input dimensionality $d_{in}$ (log scale)")
    ax.set_ylabel("Effective rank of $\\hat{\\Sigma}$ (log scale)")
    ax.set_title("Effective rank of the latent covariance", loc="left",
                 fontweight="bold")

    save(fig, 6, "effective_rank")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=str, default="all", help="1..6 (Figures 4..9) or 'all'")
    args = ap.parse_args()

    print(f"Results : {RESULTS}")
    print(f"Figures : {FIGDIR}\n")

    figs = {"1": figura_1, "2": figura_2, "3": figura_3,
            "4": figura_4, "5": figura_5, "6": figura_6}
    sel = list(figs) if args.fig == "all" else [f.strip() for f in args.fig.split(",")]

    for f in sel:
        if f not in figs:
            print(f"[SKIP] unknown figure: {f}")
            continue
        try:
            figs[f]()
        except FileNotFoundError as err:
            print(f"[SKIPPED] Figure {f}: {err}")
        except Exception as err:
            import traceback
            print(f"[ERROR] Figure {f}: {type(err).__name__}: {err}")
            traceback.print_exc()


if __name__ == "__main__":
    main()