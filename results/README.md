# Results

`paper/` holds the per-seed results behind every table and figure of the paper: one row per dataset, method, configuration and seed (100–104). They are the input of `scripts/reproduce.py` and the reference of the tests.

Re-running an experiment writes to `runs/` (or to `$DPNSA_RESULTS`), never to `paper/`.
