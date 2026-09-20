#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "${ROOT}/.." && pwd)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE_ROOT}/training_outputs/bips_reproduction_7b}
export ROOT WORKSPACE_ROOT RUN_ROOT

"${ROOT}/run_bips_reproduction_7b_stage1.sh"
RUN_ROOT="${RUN_ROOT}" "${ROOT}/tools/merge_final_checkpoint.sh" stage1
"${ROOT}/run_bips_reproduction_7b_stage2.sh"
RUN_ROOT="${RUN_ROOT}" "${ROOT}/tools/merge_final_checkpoint.sh" stage2

echo "BiPS 7B reproduction completed. Final checkpoint records:"
cat "${RUN_ROOT}/stage1/FINAL_CHECKPOINT.txt"
cat "${RUN_ROOT}/stage2/FINAL_CHECKPOINT.txt"
