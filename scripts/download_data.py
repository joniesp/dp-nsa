#!/usr/bin/env python3
"""Download the datasets of the paper into data/<name>/ (X.csv, y.csv, meta.json).

    pip install openml ucimlrepo
    python scripts/download_data.py              # into data/
    python scripts/download_data.py --out DIR    # elsewhere
    python scripts/check_data.py                 # then check them

The files keep the ORIGINAL labels; dpnsa/datasets.py applies the anomaly
conventions. Existing datasets are skipped. Credit Card Fraud is not
downloaded automatically: it requires a Kaggle account (see data/README.md).
"""
import argparse
import json
import os
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OPENML = {
    "annthyroid": 40886,
    "arrhythmia": 1017,
    "mammography": 310,
    "musk": 300,
    "cardiotocography": 1466,
    "ecg5000": 1471,
}
UCI = {
    "shuttle": 148,
    "satimage": 146,
    "ionosphere": 52,
}


def from_openml(did, attempts=3):
    import openml
    last = None
    for i in range(attempts):
        try:
            ds = openml.datasets.get_dataset(did, download_data=True)
            target = ds.default_target_attribute
            X, y, _, _ = ds.get_data(target=target, dataset_format="dataframe")
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            meta = {"source": "openml", "did": int(ds.dataset_id), "name": ds.name,
                    "version": int(ds.version), "target": target,
                    "url": f"https://www.openml.org/d/{ds.dataset_id}",
                    "shape": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}}
            return X, y, meta
        except Exception as e:
            last = e
            print(f"  [WARN] OpenML {did}, attempt {i + 1}/{attempts}: {e}")
            time.sleep(1.0)
    raise RuntimeError(f"OpenML {did} failed: {last}")


def from_uci(uid):
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=uid)
    X, y = ds.data.features.copy(), ds.data.targets.copy()
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    meta = {"source": "uci", "uci_id": int(uid), "name": ds.metadata.name,
            "shape": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}}
    return X, y, meta


def save(out, name, X, y, meta):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    X.to_csv(os.path.join(d, "X.csv"), index=False)
    y.to_frame(name="target").to_csv(os.path.join(d, "y.csv"), index=False)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {name}: X {X.shape}, y {y.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    out = ap.parse_args().out
    jobs = [(n, lambda i=i: from_openml(i)) for n, i in OPENML.items()] + \
           [(n, lambda i=i: from_uci(i)) for n, i in UCI.items()]
    fallos = 0
    for name, fetch in jobs:
        if os.path.exists(os.path.join(out, name, "X.csv")):
            print(f"  [SKIP] {name} already present")
            continue
        try:
            save(out, name, *fetch())
        except Exception as e:
            fallos += 1
            print(f"  [ERROR] {name}: {e}")
    if not os.path.exists(os.path.join(out, "creditcard.csv")):
        print("\n  Credit Card Fraud: download creditcard.csv from "
              "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into", out)
    raise SystemExit(1 if fallos else 0)


if __name__ == "__main__":
    main()
