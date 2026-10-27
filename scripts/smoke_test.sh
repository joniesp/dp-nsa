#!/usr/bin/env bash
# ==========================================================================
# Level 4: end-to-end smoke test of every experiment on a small dataset
# (Ionosphere), one seed, writing to results/smoke/ (published results are
# never touched). About 15 minutes on one GPU.
#
#   scripts/smoke_test.sh
#   GPU_ID=1 scripts/smoke_test.sh
#
# Then scripts/check_smoke.py checks the logs for errors and that every
# experiment wrote a CSV with the same columns as the published one.
# ==========================================================================
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}"
export DPNSA_RESULTS=results/smoke DPNSA_SEEDS=100
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
T="${THREADS:-8}"
export OPENBLAS_NUM_THREADS=$T OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T MPLBACKEND=Agg
rm -rf results/smoke logs/smoke && mkdir -p results/smoke logs/smoke

run () {  # run <name> <env...> -- <script> [args]
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "== $name"
  env "${envs[@]}" "$PY" -u "$@" > "logs/smoke/$name.log" 2>&1
  echo "   exit $?  (logs/smoke/$name.log)"
}

run exp1 EXP1_DATASETS=Ionosphere EXP1_TAG=smoke -- experiments/exp1_benchmark.py
run exp2 EXP2_DATASETS=Ionosphere EXP2_TAG=smoke -- experiments/exp2_ablation.py
run exp3 EXP3_DATASETS=Ionosphere -- experiments/exp3_latent_projection.py
run exp5 EXP5_DATASETS=Ionosphere EXP5_PARTE=AB -- experiments/exp5_granularity.py
run exp6 EXP6_DATASETS=Ionosphere EXP6_TAG=smoke -- experiments/exp6_label_scarcity.py
run exp7 EXP7_DATASETS=Ionosphere EXP7_TAG=smoke EXP7_KS=1 EXP7_ALPHAS=0.1,0.9 -- experiments/exp7_shrinkage.py
run exp9 DUMMY=1 -- experiments/exp9_trend_saliency.py --parte saliencia --n 20

"$PY" scripts/check_smoke.py
