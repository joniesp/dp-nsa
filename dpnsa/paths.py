"""Repository paths, in one place."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
# Per-seed results released with the paper (never overwritten).
PAPER_RESULTS = os.path.join(ROOT, "results", "paper")
# Where re-runs of the experiments write.
RUNS_DIR = os.environ.get("DPNSA_RESULTS", os.path.join(ROOT, "results", "runs"))


def resultado(nombre):
    """An input CSV: the one from a local re-run if it exists, otherwise the released one."""
    for d in (RUNS_DIR, PAPER_RESULTS):
        p = os.path.join(d, nombre)
        if os.path.exists(p):
            return p
    return os.path.join(PAPER_RESULTS, nombre)
