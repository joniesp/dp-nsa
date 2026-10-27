"""Level 1: the tables, figures and in-text numbers of the paper are regenerated
from the released per-seed results and compared against the manuscript.

Runs on CPU in about a minute; no GPU, no PyTorch, no datasets needed. The
dataset-characteristic numbers (Table 1 and the dpN/dpDim/... macros) are
checked only when the datasets have been downloaded (see data/README.md).
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference"
HAY_DATOS = (ROOT / "data" / "creditcard.csv").exists()
# Macros computed from the raw datasets rather than from the results.
MACROS_DATOS = re.compile(r"^dp(N|Dim|NAnom|NoverD|Rho)[A-Z]")
FIGURAS = [(4, "label_scarcity"), (5, "repulsion_effect"), (6, "anisotropy_effect"),
           (7, "granularity_repertoire"), (8, "shrinkage_sensitivity"), (9, "effective_rank")]


def lineas(p):
    texto = p.read_text(encoding="utf-8").replace("\r", "")
    return [l for l in texto.splitlines() if l.strip() and not l.startswith("%")]


def macros(p):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", p.read_text(encoding="utf-8")))


def correr(script, out):
    env = dict(os.environ, DPNSA_OUT=str(out), OPENBLAS_NUM_THREADS="4", MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(ROOT / "analysis" / script)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


@pytest.fixture(scope="module")
def tablas(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper")
    correr("make_tables.py", out)
    return out


@pytest.mark.parametrize("tabla", ["tabla3_benchmark.tex", "tabla4_ablacion.tex", "tabla5_saliencia.tex"])
def test_table_matches_manuscript(tablas, tabla):
    assert lineas(tablas / tabla) == lineas(REF / tabla)


def test_table1_matches_manuscript(tablas):
    if not HAY_DATOS:
        pytest.skip("datasets not downloaded: Table 1 is computed from the raw data")
    assert lineas(tablas / "tabla1_datasets.tex") == lineas(REF / "tabla1_datasets.tex")


def test_in_text_numbers_match_manuscript(tablas):
    ref, gen = macros(REF / "cifras.tex"), macros(tablas / "cifras.tex")
    distintos = {k: (ref.get(k), gen[k]) for k in gen if gen[k] != ref.get(k)}
    assert not distintos, f"{len(distintos)} macros differ, e.g. {list(distintos.items())[:5]}"
    faltan = [k for k in ref if k not in gen and (HAY_DATOS or not MACROS_DATOS.match(k))]
    assert not faltan, f"missing macros: {faltan[:10]}"


def test_figures_are_generated_with_paper_numbering(tmp_path):
    correr("make_figures.py", tmp_path)
    hechas = sorted(p.name for p in (tmp_path / "figures").glob("*.pdf"))
    assert hechas == sorted(f"Figure_{n}_{x}.pdf" for n, x in FIGURAS)
