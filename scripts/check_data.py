#!/usr/bin/env python3
"""Check the downloaded datasets before running any experiment.

1) Every file listed in data/SHA256SUMS is present and byte-identical to the
   one used for the paper.
2) After the label conventions of dpnsa/datasets.py, every dataset has the
   size and contamination rate of Table 1.

    python scripts/check_data.py
Exit code 0 only if both checks pass.
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


fallos = 0
print("== 1) checksums (data/SHA256SUMS)")
for linea in open(os.path.join(DATA, "SHA256SUMS"), encoding="utf-8"):
    esperado, rel = linea.split(maxsplit=1)
    rel = rel.strip().lstrip("*").removeprefix("./")
    p = os.path.join(DATA, rel)
    if not os.path.exists(p):
        estado = "MISSING"
    elif sha256(p) != esperado:
        estado = "DIFFERENT"
    else:
        estado = "ok"
    fallos += estado != "ok"
    print(f"  {estado:9s} {rel}")

print("\n== 2) size and contamination after the label conventions (Table 1)")
from dpnsa.datasets import verify_all  # noqa: E402

t = verify_all(os.path.join(DATA, "creditcard.csv"))
print(t.to_string(index=False))
fallos += int((t["status"] != "OK").sum())
print("\nDATA OK" if not fallos else f"\n{fallos} PROBLEMS: see above")
sys.exit(1 if fallos else 0)
