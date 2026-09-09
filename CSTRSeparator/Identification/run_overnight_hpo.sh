#!/usr/bin/env bash
# Full overnight HPO rerun: C + noC + comparison.
# Logs to CSTRSeparator/data/hpo_results/overnight_run.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
RESULTS_DIR="${DATA_DIR}/hpo_results"
LOG="${RESULTS_DIR}/overnight_run.log"

mkdir -p "${RESULTS_DIR}"

exec > >(tee -a "${LOG}") 2>&1

echo "========================================"
echo "Overnight HPO rerun started: $(date)"
echo "========================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kmpc

cd "${SCRIPT_DIR}"

echo ""
echo "--- Stage ALL: Koopman C (matrix_C=True) ---"
python hpo_koopman_C.py --stage all --n-trials 40 --top-k 3 --seeds 0 1 2

echo ""
echo "--- Stage ALL: Koopman noC (matrix_C=False) ---"
python hpo_koopman_noC.py --stage all --n-trials 40 --top-k 3 --seeds 0 1 2

echo ""
echo "--- Comparison (SIPPY cached) ---"
conda activate sippy
python compare_koopman_sippy.py --skip-sippy

echo ""
echo "========================================"
echo "Overnight HPO rerun finished: $(date)"
echo "========================================"
echo "Results:"
echo "  C meta:    ${DATA_DIR}/model_cstr_separator_C_optimized_meta.json"
echo "  noC meta:  ${DATA_DIR}/model_cstr_separator_noC_optimized_meta.json"
echo "  plot:      ${RESULTS_DIR}/comparison_plot.png"
echo "  summary:   ${RESULTS_DIR}/comparison_summary.json"
echo "  log:       ${LOG}"
