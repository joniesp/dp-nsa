"""Dataset loading, label conventions and splits (Section 4.1 of the paper).

- Each loader documents how the binary anomaly label is derived from the
  original labels, following ODDS where the dataset is part of it.
- verify_all() checks N, d and contamination rate against EXPECTED; Table 1
  is generated from the data with build_table1().
- prepare_*(seed=...) changes the train/calibration/test split with the seed,
  so the five repetitions differ in the split as well as in the initialisation.
- The ODDS subsampling (Musk, Satimage-2) uses a fixed seed, independent of
  the split, so the dataset itself is identical in every repetition.
- Imputation, scaling and the choice of features use the training split only.
- Unrecognised labels raise an error instead of being silently mapped.
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

SEED = 1337                 # default split seed
SUBSAMPLE_SEED = 20240101   # fixed seed of the ODDS subsampling (never changes)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# =============================================================================
# Reference values (ODDS, http://odds.cs.stonybrook.edu/)
# =============================================================================
# N and d are indicative (d may differ after dropping constant columns);
# rho is the value that must be reproduced.
EXPECTED = {
    "annthyroid":        dict(N=7200,   d=6,   rho=0.0742),
    "arrhythmia":        dict(N=452,    d=274, rho=0.1460),
    "cardiotocography":  dict(N=2126,   d=21,  rho=0.2220),
    "ecg5000":           dict(N=5000,   d=140, rho=0.4160),
    "ionosphere":        dict(N=351,    d=33,  rho=0.3590),
    "mammography":       dict(N=11183,  d=6,   rho=0.0232),
    "musk":              dict(N=3062,   d=166, rho=0.0317),
    "satimage":          dict(N=5803,   d=36,  rho=0.0122),
    "shuttle":           dict(N=49097,  d=9,   rho=0.0715),
    "creditcard":        dict(N=284807, d=29,  rho=0.0017),
}

# ECG5000 is not part of ODDS. The standard UCR convention is used:
# class 1 = normal (58.4%), classes 2-5 = anomaly (41.6%).
# load_ecg5000_df(subsample_anomalies_to=n) gives an imbalanced variant.


# =============================================================================
# Splits
# =============================================================================
def make_splits(df: pd.DataFrame, test_frac=0.2, calib_frac=0.2, seed=SEED):
    """Stratified train/calibration/test splits; `seed` changes the partition."""
    y = df["Class"].values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(sss1.split(np.zeros(len(y)), y))
    df_trval = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    y_trval = df_trval["Class"].values
    calib_frac_adj = calib_frac / (1.0 - test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=calib_frac_adj, random_state=seed)
    tr_idx2, cal_idx = next(sss2.split(np.zeros(len(y_trval)), y_trval))
    df_tr = df_trval.iloc[tr_idx2].reset_index(drop=True)
    df_cal = df_trval.iloc[cal_idx].reset_index(drop=True)

    return df_tr, df_cal, df_te


# =============================================================================
# Preprocessing
# =============================================================================
class TabularPreprocessor:
    """Median imputation + MinMax scaling, both fitted on the training split only."""

    def __init__(self, label_col="Class"):
        self.label_col = label_col
        self.imputer = None
        self.scaler = None
        self.keep_cols = None

    def fit(self, df_tr: pd.DataFrame):
        X = df_tr.drop(columns=[self.label_col])
        self.keep_cols = X.columns.tolist()

        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X)
        self.scaler = MinMaxScaler().fit(X_imp)
        return self

    def transform(self, df: pd.DataFrame):
        X = df.drop(columns=[self.label_col])[self.keep_cols]
        X_imp = self.imputer.transform(X)
        Xs = self.scaler.transform(X_imp).astype(np.float32)
        y = df[self.label_col].astype(int).values
        return Xs, np.zeros((len(df), 0), dtype=np.int64), y


class CreditPreprocessor(TabularPreprocessor):
    """CreditCard: no missing values; same scaling."""
    pass


# =============================================================================
# Helpers
# =============================================================================
def _read_Xy(name: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    ddir = os.path.join(DATA_DIR, name.lower())
    X = pd.read_csv(os.path.join(ddir, "X.csv"))
    y = pd.read_csv(os.path.join(ddir, "y.csv")).iloc[:, 0]
    meta = {}
    mp = os.path.join(ddir, "meta.json")
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return X, y, meta


# Columns that must never be used as features (label leakage).
LEAK_COLS = {
    "class", "nsp", "target", "label", "y", "outcome",
    # CLASS dummies in the UCI CTG file
    "a", "b", "c", "d", "e", "ad", "de", "ld", "fs", "susp",
}


def _drop_leak_columns(X: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Drop label columns disguised as features."""
    bad = [c for c in X.columns if c.strip().lower() in LEAK_COLS]
    if bad:
        print(f"   [LEAK] {dataset}: dropped label columns from X -> {bad}")
        X = X.drop(columns=bad)
    return X


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in "OUSb":
            df[col] = df[col].replace({"?": np.nan, "NA": np.nan, "NaN": np.nan, "": np.nan})
            df[col] = pd.to_numeric(df[col], errors="coerce")
    all_nan = df.columns[df.isna().all()]
    if len(all_nan):
        df = df.drop(columns=list(all_nan))
    return df


def _to_numeric_labels(y: pd.Series) -> pd.Series:
    if y.dtype.kind in "OUS":
        y = pd.to_numeric(y, errors="coerce")
    if y.isna().any():
        raise ValueError(f"Labels not convertible to numbers: {y[y.isna()].index[:5].tolist()}")
    return y.astype(int)


def _assemble(X: pd.DataFrame, y_bin: np.ndarray) -> pd.DataFrame:
    df = X.copy()
    df["Class"] = np.asarray(y_bin).astype(int)
    return df


def _odds_subsample(df: pd.DataFrame, n_anomalies: int, seed=SUBSAMPLE_SEED) -> pd.DataFrame:
    """Subsample the anomaly class to `n_anomalies`, as ODDS does (fixed seed)."""
    rng = np.random.RandomState(seed)
    idx_norm = df.index[df["Class"] == 0].to_numpy()
    idx_anom = df.index[df["Class"] == 1].to_numpy()
    if len(idx_anom) <= n_anomalies:
        return df.reset_index(drop=True)
    keep = rng.choice(idx_anom, size=n_anomalies, replace=False)
    out = df.loc[np.concatenate([idx_norm, keep])].sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


# =============================================================================
# Per-dataset loaders
# =============================================================================

def load_credit_csv(path: str) -> pd.DataFrame:
    """Kaggle/ULB Credit Card Fraud: drop Time, log1p(Amount). -> 0.17%"""
    df = pd.read_csv(path)
    assert "Class" in df.columns
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    if "Amount" in df.columns:
        df["Amount"] = np.log1p(df["Amount"].astype(float))
    return df


def load_arrhythmia_df() -> pd.DataFrame:
    """ODDS: classes {3,4,5,7,8,9,14,15} are anomalies, the rest normal. -> 66/452 = 14.6%"""
    X, y, _ = _read_Xy("arrhythmia")
    y = _to_numeric_labels(y)
    ODDS_OUTLIER_CLASSES = {3, 4, 5, 7, 8, 9, 14, 15}
    y_bin = y.isin(ODDS_OUTLIER_CLASSES).astype(int)
    X = _drop_leak_columns(_coerce_numeric_df(X), "arrhythmia")
    return _assemble(X, y_bin)


def load_mammography_df() -> pd.DataFrame:
    """OpenML 310: -1 normal, 1 anomaly. -> 2.32%"""
    X, y, _ = _read_Xy("mammography")
    y = _to_numeric_labels(y)
    uniq = set(y.unique().tolist())
    if uniq <= {-1, 1}:
        y_bin = (y == 1).astype(int)
    elif uniq <= {0, 1}:
        y_bin = y.astype(int)
    else:
        raise ValueError(f"Mammography: unexpected labels {sorted(uniq)}")
    X = _drop_leak_columns(_coerce_numeric_df(X), "mammography")
    return _assemble(X, y_bin)


def load_shuttle_df() -> pd.DataFrame:
    """ODDS: class 1 normal, classes {2,3,5,6,7} anomaly, class 4 dropped. -> 3511/49097 = 7.15%"""
    X, y, _ = _read_Xy("shuttle")
    y = _to_numeric_labels(y)
    X = _drop_leak_columns(_coerce_numeric_df(X), "shuttle")
    keep = (y != 4).values
    X, y = X.loc[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    y_bin = (y != 1).astype(int)
    return _assemble(X, y_bin)


def load_satimage_df() -> pd.DataFrame:
    """ODDS satimage-2: class 2 (cotton crop) is the anomaly, subsampled to 71. -> 71/5803 = 1.22%"""
    X, y, _ = _read_Xy("satimage")
    y = _to_numeric_labels(y)
    X = _drop_leak_columns(_coerce_numeric_df(X), "satimage")
    y_bin = (y == 2).astype(int)
    df = _assemble(X, y_bin)
    return _odds_subsample(df, n_anomalies=71)


def load_musk_df() -> pd.DataFrame:
    """ODDS: Musk v2 non-musk molecules {153,152,145,143} normal, musk {213,211}
    anomaly, subsampled to 97. -> 97/3062 = 3.17%

    If y.csv only holds the binary musk/non-musk label, the ODDS ratio is
    reproduced by subsampling instead of the exact molecule selection.
    """
    X, y, _ = _read_Xy("musk")
    y = _to_numeric_labels(y)
    X = _drop_leak_columns(_coerce_numeric_df(X), "musk")
    uniq = set(y.unique().tolist())
    if uniq <= {0, 1}:
        print("   [WARNING] Musk: y.csv is binary; the ODDS ratio is reproduced by "
              "subsampling, not the exact molecule selection.")
        y_bin = y.astype(int)
    else:
        y_bin = y.isin({213, 211}).astype(int)
    df = _assemble(X, y_bin)
    # ODDS has 2965 normal and 97 anomalous samples: subsample both sides if needed
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    idx_n = df.index[df["Class"] == 0].to_numpy()
    if len(idx_n) > 2965:
        idx_n = rng.choice(idx_n, size=2965, replace=False)
    idx_a = df.index[df["Class"] == 1].to_numpy()
    if len(idx_a) > 97:
        idx_a = rng.choice(idx_a, size=97, replace=False)
    out = df.loc[np.concatenate([idx_n, idx_a])].sample(frac=1.0, random_state=SUBSAMPLE_SEED)
    return out.reset_index(drop=True)


def load_ionosphere_df() -> pd.DataFrame:
    """'g' (good) normal, 'b' (bad) anomaly. -> 35.9%"""
    X, y, _ = _read_Xy("ionosphere")
    if y.dtype.kind in "OUS":
        y_bin = (y.astype(str).str.lower().str.strip() != "g").astype(int)
    else:
        raise ValueError("Ionosphere: expected 'g'/'b' labels.")
    X = _drop_leak_columns(_coerce_numeric_df(X), "ionosphere")
    return _assemble(X, y_bin)


def load_cardiotocography_df() -> pd.DataFrame:
    """UCI CTG with the NSP label (1 Normal, 2 Suspect, 3 Pathologic), not CLASS.
    NSP != 1 is the anomaly. -> 471/2126 = 22.2%

    X.csv must not contain NSP, CLASS or the CLASS dummies (A, B, ..., SUSP).
    """
    X, y, _ = _read_Xy("cardiotocography")
    y = _to_numeric_labels(y)
    uniq = sorted(y.unique().tolist())
    if len(uniq) > 3:
        raise ValueError(
            f"Cardiotocography: y.csv has {len(uniq)} classes {uniq[:12]}. "
            "It looks like the CLASS column (morphologic pattern, 1-10). "
            "NSP is required (1=Normal, 2=Suspect, 3=Pathologic). "
            "Rebuild y.csv from the NSP column of the UCI CTG.xls file."
        )
    X = _drop_leak_columns(_coerce_numeric_df(X), "cardiotocography")
    y_bin = (y != 1).astype(int)
    return _assemble(X, y_bin)


def load_ecg5000_df(subsample_anomalies_to: Optional[int] = None) -> pd.DataFrame:
    """UCR ECG5000: class 1 normal (58.4%), classes 2-5 anomaly (41.6%)."""
    X, y, _ = _read_Xy("ecg5000")
    y = _to_numeric_labels(y)
    X = _drop_leak_columns(_coerce_numeric_df(X), "ecg5000")
    y_bin = (y != 1).astype(int)
    df = _assemble(X, y_bin)
    if subsample_anomalies_to is not None:
        df = _odds_subsample(df, n_anomalies=subsample_anomalies_to)
    return df


def load_annthyroid_df() -> pd.DataFrame:
    """Annthyroid: 'normal'/'anomaly' or numeric labels (minority = anomaly). -> 7.42%"""
    X, y, _ = _read_Xy("annthyroid")
    if y.dtype.kind in "OUS":
        ys = y.astype(str).str.lower().str.strip()
        y_bin = ys.map({"normal": 0, "anomaly": 1})
        if y_bin.isna().any():
            raise ValueError(
                f"Annthyroid: unrecognised labels -> {sorted(ys[y_bin.isna()].unique())[:5]}"
            )
    else:
        y = _to_numeric_labels(y)
        uniq = set(y.unique().tolist())
        if uniq <= {0, 1}:
            y_bin = y
        else:
            y_bin = (y != y.mode().iloc[0]).astype(int)
    X = _drop_leak_columns(_coerce_numeric_df(X), "annthyroid")
    return _assemble(X, y_bin.astype(int))


# =============================================================================
# Public API: (Xtr, Xcal, Xte, ytr, ycal, yte), features scaled to [0, 1]
# =============================================================================
def _prepare_generic(load_df_fn, seed=SEED, verbose=True):
    df = load_df_fn()
    feats = df.drop(columns=["Class"])
    nun = feats.nunique(dropna=True)
    df = df[nun.index[nun > 1].tolist() + ["Class"]]
    df_tr, df_cal, df_te = make_splits(df, seed=seed)
    pp = TabularPreprocessor().fit(df_tr)
    Xtr, _, ytr = pp.transform(df_tr)
    Xcal, _, ycal = pp.transform(df_cal)
    Xte, _, yte = pp.transform(df_te)
    if verbose:
        print(f"{load_df_fn.__name__}: N={len(df)} d={Xtr.shape[1]} "
              f"rho={df['Class'].mean():.4f} | Tr={len(ytr)} Cal={len(ycal)} Te={len(yte)} "
              f"| seed_split={seed}")
    return Xtr, Xcal, Xte, ytr, ycal, yte


def prepare_creditcard(csv_path: str, seed=SEED, verbose=True):
    df = load_credit_csv(csv_path)
    df_tr, df_cal, df_te = make_splits(df, seed=seed)
    pp = CreditPreprocessor().fit(df_tr)
    Xtr, _, ytr = pp.transform(df_tr)
    Xcal, _, ycal = pp.transform(df_cal)
    Xte, _, yte = pp.transform(df_te)
    if verbose:
        print(f"CreditCard: N={len(df)} d={Xtr.shape[1]} rho={df['Class'].mean():.4f} "
              f"| Tr={len(ytr)} Cal={len(ycal)} Te={len(yte)} | seed_split={seed}")
    return Xtr, Xcal, Xte, ytr, ycal, yte


def prepare_arrhythmia(seed=SEED):        return _prepare_generic(load_arrhythmia_df, seed)
def prepare_mammography(seed=SEED):       return _prepare_generic(load_mammography_df, seed)
def prepare_shuttle(seed=SEED):           return _prepare_generic(load_shuttle_df, seed)
def prepare_satimage(seed=SEED):          return _prepare_generic(load_satimage_df, seed)
def prepare_musk(seed=SEED):              return _prepare_generic(load_musk_df, seed)
def prepare_ionosphere(seed=SEED):        return _prepare_generic(load_ionosphere_df, seed)
def prepare_cardiotocography(seed=SEED):  return _prepare_generic(load_cardiotocography_df, seed)
def prepare_ecg5000(seed=SEED):           return _prepare_generic(load_ecg5000_df, seed)
def prepare_annthyroid(seed=SEED):        return _prepare_generic(load_annthyroid_df, seed)


# =============================================================================
# Checks and Table 1
# =============================================================================
LOADERS = {
    "annthyroid": load_annthyroid_df,
    "arrhythmia": load_arrhythmia_df,
    "cardiotocography": load_cardiotocography_df,
    "ecg5000": load_ecg5000_df,
    "ionosphere": load_ionosphere_df,
    "mammography": load_mammography_df,
    "musk": load_musk_df,
    "satimage": load_satimage_df,
    "shuttle": load_shuttle_df,
}


def build_table1(credit_path: Optional[str] = None) -> pd.DataFrame:
    """Table 1, computed from the data."""
    rows = []
    for name, fn in LOADERS.items():
        try:
            df = fn()
            rows.append(dict(Dataset=name, N=len(df), d=df.shape[1] - 1,
                             n_anom=int(df["Class"].sum()),
                             rho_pct=round(100 * df["Class"].mean(), 3)))
        except Exception as e:
            rows.append(dict(Dataset=name, N=None, d=None, n_anom=None,
                             rho_pct=None, error=f"{type(e).__name__}: {e}"))
    if credit_path and os.path.exists(credit_path):
        df = load_credit_csv(credit_path)
        rows.append(dict(Dataset="creditcard", N=len(df), d=df.shape[1] - 1,
                         n_anom=int(df["Class"].sum()),
                         rho_pct=round(100 * df["Class"].mean(), 3)))
    return pd.DataFrame(rows)


def verify_all(credit_path: Optional[str] = None, tol=0.01) -> pd.DataFrame:
    """Compare the real N/d/rho against EXPECTED ('status' = OK or what deviates)."""
    t = build_table1(credit_path)
    out = []
    for _, r in t.iterrows():
        exp = EXPECTED.get(r["Dataset"], {})
        if r.get("rho_pct") is None:
            out.append({**r.to_dict(), "status": "ERROR"})
            continue
        drho = abs(r["rho_pct"] / 100 - exp.get("rho", np.nan))
        ok_rho = drho <= tol
        ok_N = (exp.get("N") is None) or (abs(r["N"] - exp["N"]) <= max(5, 0.01 * exp["N"]))
        estado = "OK" if (ok_rho and ok_N) else (
            f"{'rho' if not ok_rho else ''}{' N' if not ok_N else ''} deviates")
        out.append({**r.to_dict(), "rho_expected": exp.get("rho"),
                    "N_expected": exp.get("N"), "status": estado})
    return pd.DataFrame(out)


if __name__ == "__main__":
    cp = os.path.join(DATA_DIR, "creditcard.csv")
    print(verify_all(cp).to_string(index=False))
