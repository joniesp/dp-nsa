#!/usr/bin/env bash
# ==========================================================================
# Launch one experiment in parallel over groups of datasets, one process per
# group, in the background, with one log per group.
#
#   scripts/run_experiment.sh exp1 "CreditCard" "Shuttle" "Mammography,Annthyroid"
#   GPU_ID=0 THREADS=12 scripts/run_experiment.sh exp6 "Musk" "Satimage"
#
# Experiments: exp1 exp2 exp5 exp6 exp7 (the groups used for the paper are
# listed in README.md). When every group has finished, merge the per-group
# files with:  python experiments/<script>.py --merge
#
# Notes for many-core machines:
#  - BLAS threads are capped per process (THREADS). Some OpenBLAS builds abort
#    ("too many memory regions") when several processes each try to use all
#    cores.
#  - CUDA_DEVICE_ORDER=PCI_BUS_ID makes GPU_ID match the numbering of nvidia-smi.
# ==========================================================================
set -u
cd "$(dirname "$0")/.."

EXP="${1:?usage: run_experiment.sh <exp1|exp2|exp5|exp6|exp7> <group> [<group> ...]}"
shift
case "$EXP" in
  exp1) SCRIPT=experiments/exp1_benchmark.py ;;
  exp2) SCRIPT=experiments/exp2_ablation.py ;;
  exp5) SCRIPT=experiments/exp5_granularity.py ;;
  exp6) SCRIPT=experiments/exp6_label_scarcity.py ;;
  exp7) SCRIPT=experiments/exp7_shrinkage.py ;;
  *) echo "unknown experiment: $EXP"; exit 1 ;;
esac
[ "$#" -ge 1 ] || { echo "give at least one dataset group"; exit 1; }

PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}"
GPU="${GPU_ID:-0}"
THREADS="${THREADS:-12}"
PRE=$(echo "$EXP" | tr a-z A-Z)
mkdir -p logs

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" -c "import torch; print('GPU ->', torch.cuda.get_device_name(0))" || exit 1

for G in "$@"; do
  T="${G//,/_}"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
      OPENBLAS_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      NUMEXPR_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS" \
      "${PRE}_DATASETS=$G" "${PRE}_TAG=$T" \
      nohup "$PY" -u "$SCRIPT" > "logs/${EXP}_${T}.log" 2>&1 &
  echo "  launched $EXP $T (PID $!)  ->  logs/${EXP}_${T}.log"
done
echo
echo "Follow:   tail -n 20 logs/${EXP}_*.log"
echo "Errors:   grep -l 'Traceback\\|\\[ERROR\\]\\|OutOfMemory' logs/${EXP}_*.log"
echo "Merge:    $PY $SCRIPT --merge"
