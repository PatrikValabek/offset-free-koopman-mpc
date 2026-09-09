#!/usr/bin/env bash
# Closed-loop CT map search on 8 CPUs: even and odd nz in [4, 32].
# Objective = CT notebook OF (not identification MAE).
# Log: ../data/hpo_results/C_cl/trials.tsv
set -euo pipefail
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export HPO_N_JOBS=8
export HPO_BLAS_THREADS=1

echo "=== $(date)  C closed-loop map search (8 jobs, nz=4..32 int) ==="
python hpo_koopman_C_cl.py --n-trials 48 --n-jobs 8 --blas-threads 1

echo "=== $(date)  closed-loop map search finished ==="
