#!/usr/bin/env bash
# Closed-loop CT search with current sim_setup.pkl (Qu on).
# 2 parallel trials, until Ctrl+C. Does not overwrite C_cl_bd.
# Log: ../data/hpo_results/C_cl_qu/trials.tsv
set -euo pipefail
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export HPO_N_JOBS=2
export HPO_BLAS_THREADS=1

echo "=== $(date)  C closed-loop search Qu (2 jobs, unlimited, block_diag=True) ==="
echo "Stop with Ctrl+C. Results: ../data/hpo_results/C_cl_qu/trials.tsv"
python hpo_koopman_C_cl.py --n-jobs 2 --blas-threads 1 --no-final-retrain
echo "=== $(date)  closed-loop search stopped ==="
