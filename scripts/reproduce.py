#!/usr/bin/env python
"""Regenerate every table, figure and in-text number of the paper from the
per-seed results in results/paper/ (no GPU, about a minute).

    python scripts/reproduce.py                 # writes to paper/
    DPNSA_PAPER_RESULTS=results/runs python scripts/reproduce.py   # from your own runs

Figure 10 (latent projection) needs the trained model: experiments/exp3_latent_projection.py.
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PASOS = ["make_tables.py", "make_figures.py"]

if __name__ == "__main__":
    fallos = 0
    for s in PASOS:
        print(f"== {s}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "analysis" / s)], cwd=ROOT,
                           env=dict(os.environ, MPLBACKEND="Agg"))
        fallos += r.returncode != 0
    print(f"\nOutput in {os.environ.get('DPNSA_OUT', ROOT / 'paper')}")
    sys.exit(1 if fallos else 0)
