"""Level 2 (static): every module compiles and the lightweight modules import
without PyTorch. With PyTorch installed, the model and baselines also import
and the model fits and scores a small synthetic problem deterministically."""
import importlib
import pathlib
import py_compile

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAQUETES = ["dpnsa", "baselines", "experiments", "analysis", "scripts"]


def test_every_module_compiles():
    ficheros = [p for d in PAQUETES for p in (ROOT / d).rglob("*.py")]
    assert ficheros
    for p in ficheros:
        py_compile.compile(str(p), doraise=True)


@pytest.mark.parametrize("modulo", ["dpnsa", "dpnsa.paths", "dpnsa.datasets"])
def test_light_modules_import_without_torch(modulo):
    importlib.import_module(modulo)


def test_model_is_deterministic_on_synthetic_data():
    torch = pytest.importorskip("torch")
    from dpnsa.ensemble import NSARepV23Ensemble

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8)).astype(np.float32)
    y = (rng.random(300) < 0.1).astype(int)
    X[y == 1] += 3.0

    def ajustar():
        torch.manual_seed(0)
        m = NSARepV23Ensemble(d_in=8, n_estimators=2, k=2, epochs=2, device="cpu", seed_start=7)
        m.fit(X, y)
        return np.asarray(m.score(X))

    a, b = ajustar(), ajustar()
    assert a.shape == (300,)
    np.testing.assert_array_equal(a, b)
    assert np.mean(a[y == 1]) > np.mean(a[y == 0]), "anomalies should score higher"
