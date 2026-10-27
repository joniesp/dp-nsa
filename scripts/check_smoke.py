#!/usr/bin/env python
"""Check the smoke test: no errors in the logs, and every experiment wrote a
CSV whose columns match the published one. The experiments catch exceptions
per dataset and keep going, so a clean exit code alone proves nothing."""
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS, SMOKE, PAPER = ROOT / "logs" / "smoke", ROOT / "results" / "smoke", ROOT / "results" / "paper"
# experiment -> (prefix of its output CSV, published file to compare columns with)
ESPERADO = {
    "exp1": ("exp1_", "exp1_benchmark_results.csv"),
    "exp2": ("exp2_", "exp2_ablacion_results.csv"),
    "exp5": ("exp5_", "exp5_sensibilidad_results.csv"),
    "exp6": ("exp6_", "exp6_escasez_results.csv"),
    "exp7": ("exp7_", "exp7_shrinkage_Musk.csv"),
    "exp9": ("exp9_saliencia", "exp9_saliencia_control.csv"),
}
ERROR = re.compile(r"Traceback|\[ERROR\]|OutOfMemory|OpenBLAS : Program is Terminated")

fallos = []
for log in sorted(LOGS.glob("*.log")):
    texto = log.read_text(errors="replace")
    if ERROR.search(texto):
        fallos.append(f"{log.name}: error in log")
for exp, (pref, pub) in ESPERADO.items():
    csvs = sorted(SMOKE.glob(f"{pref}*.csv"))
    if not csvs:
        fallos.append(f"{exp}: no CSV written")
        continue
    esperadas = set(pd.read_csv(PAPER / pub, nrows=1).columns)
    for c in csvs:
        cols = set(pd.read_csv(c, nrows=1).columns)
        if cols != esperadas:
            fallos.append(f"{exp}: {c.name} columns differ: missing {sorted(esperadas - cols)}, "
                          f"extra {sorted(cols - esperadas)}")
if not list((SMOKE / "plots_tsne").glob("*.png")):
    fallos.append("exp3: no latent projection plot")

print("\n".join(fallos) if fallos else "SMOKE TEST PASSED: every experiment ran and wrote the expected columns")
sys.exit(1 if fallos else 0)
