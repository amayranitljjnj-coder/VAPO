#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 stage1|stage2" >&2; exit 2; }
STAGE=$1
[[ "${STAGE}" == stage1 || "${STAGE}" == stage2 ]] || exit 2
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "${ROOT}/.." && pwd)}
BIPS_ROOT=${BIPS_ROOT:-${ROOT}/vendor/BiPS}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE_ROOT}/training_outputs/counterfactual_rlvr_7b}
CONDA_ENV=${CONDA_ENV:-bips}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/${STAGE}}
TRACKER=${OUTPUT_DIR}/checkpoints/latest_checkpointed_iteration.txt

[[ -s "${TRACKER}" ]] || { echo "Missing checkpoint tracker: ${TRACKER}" >&2; exit 1; }
FINAL_STEP=$(tr -d '[:space:]' <"${TRACKER}")
[[ "${FINAL_STEP}" =~ ^[0-9]+$ ]] || { echo "Invalid final step: ${FINAL_STEP}" >&2; exit 1; }
FINAL_ACTOR=${OUTPUT_DIR}/checkpoints/global_step_${FINAL_STEP}/actor
TARGET=${OUTPUT_DIR}/merged_hf
TEMP=${TARGET}.incomplete
[[ -d "${FINAL_ACTOR}/huggingface" ]] || { echo "Incomplete actor checkpoint: ${FINAL_ACTOR}" >&2; exit 1; }

rm -rf -- "${TEMP}"
mkdir -p "${TEMP}"
conda run --no-capture-output -n "${CONDA_ENV}" env \
  PYTHONPATH="${BIPS_ROOT}:${PYTHONPATH:-}" \
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -m verl.model_merger merge \
    --backend fsdp \
    --use_cpu_initialization \
    --local_dir "${FINAL_ACTOR}" \
    --target_dir "${TEMP}"

[[ -s "${TEMP}/config.json" ]] || { echo "Merged model has no config.json" >&2; exit 1; }
compgen -G "${TEMP}/model*.safetensors" >/dev/null || {
  echo "Merged model has no safetensors" >&2
  exit 1
}
rm -rf -- "${TARGET}"
mv -- "${TEMP}" "${TARGET}"
printf 'stage=%s\nfinal_step=%s\nfsdp_checkpoint=%s\nmerged_model=%s\n' \
  "${STAGE}" "${FINAL_STEP}" "${FINAL_ACTOR}" "${TARGET}" | tee "${OUTPUT_DIR}/FINAL_CHECKPOINT.txt"
