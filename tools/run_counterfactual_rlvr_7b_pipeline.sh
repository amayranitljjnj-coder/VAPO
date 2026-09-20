#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "${ROOT}/.." && pwd)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE_ROOT}/training_outputs/counterfactual_rlvr_7b}
export ROOT
export WORKSPACE_ROOT
export RUN_ROOT

"${ROOT}/run_counterfactual_rlvr_7b_stage1.sh" calibrate
"${ROOT}/run_counterfactual_rlvr_7b_stage1.sh" train
"${ROOT}/tools/merge_final_checkpoint.sh" stage1
"${ROOT}/run_counterfactual_rlvr_7b_stage2.sh" calibrate
"${ROOT}/run_counterfactual_rlvr_7b_stage2.sh" train
"${ROOT}/tools/merge_final_checkpoint.sh" stage2

echo "Both stages completed. Final checkpoint records:"
cat "${RUN_ROOT}/stage1/FINAL_CHECKPOINT.txt"
cat "${RUN_ROOT}/stage2/FINAL_CHECKPOINT.txt"
