#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 stage1|stage2 calibrate|train [Hydra overrides ...]" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
STAGE=$1
MODE=$2
shift 2
[[ "${STAGE}" == stage1 || "${STAGE}" == stage2 ]] || usage
[[ "${MODE}" == calibrate || "${MODE}" == train ]] || usage

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "${ROOT}/.." && pwd)}
BIPS_ROOT=${BIPS_ROOT:-${ROOT}/vendor/BiPS}
DATA_ROOT=${DATA_ROOT:-${WORKSPACE_ROOT}/training_data/Counterfactual_RLVR_ECD_7B_clean_v1}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-${WORKSPACE_ROOT}/models/Qwen2.5-VL-7B-Instruct}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE_ROOT}/training_outputs/counterfactual_rlvr_7b}
REWARD_FILE=${REWARD_FILE:-${ROOT}/counterfactual_rlvr/reward_accuracy.py}
CALIBRATION_DIR=${CALIBRATION_DIR:-${RUN_ROOT}/calibration}

CUDA_IDS=${CUDA_IDS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a CUDA_ID_ARRAY <<<"${CUDA_IDS}"
N_GPU=${N_GPU:-${#CUDA_ID_ARRAY[@]}}
CONDA_ENV=${CONDA_ENV:-bips}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
DATA_SEED=${DATA_SEED:-42}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.52}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-50}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
CHECKPOINT_CONTENTS=${CHECKPOINT_CONTENTS:-'["model","optimizer","extra"]'}
LOG_RATIO_CLIP=${LOG_RATIO_CLIP:-20.0}
CALIBRATION_QUANTILE=${CALIBRATION_QUANTILE:-0.5}
CALIBRATION_SAMPLE_TARGET=${CALIBRATION_SAMPLE_TARGET:-2048}
MIN_AVAILABLE_MEMORY_GIB=${MIN_AVAILABLE_MEMORY_GIB:-200}
MIN_AVAILABLE_DISK_GIB=${MIN_AVAILABLE_DISK_GIB:-350}
MIN_FREE_GPU_MEMORY_MIB=${MIN_FREE_GPU_MEMORY_MIB:-60000}

if [[ "${STAGE}" == stage1 ]]; then
  TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/stage1_train.parquet}
  VAL_FILE=${VAL_FILE:-${DATA_ROOT}/stage1_val.parquet}
  MODEL_PATH=${MODEL_PATH:-${BASE_MODEL_PATH}}
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage1}
  CALIBRATION_FILE=${CALIBRATION_FILE:-${CALIBRATION_DIR}/stage1_base_kl.json}
  PRIMARY_IMAGE_KEY=images_pres
  COUNTERPART_IMAGE_KEY=images
  TRAINER_KL_FLAG=algorithm.use_kl_cons=true
  ACTOR_KL_FLAG=actor_rollout_ref.actor.use_kl_cons=false
  OTHER_TRAINER_KL_FLAG=algorithm.use_kl_sep=false
  ORIGINAL_LOG_PROB_KEY=cons_log_probs
  PROJECT_NAME=counterfactual_rlvr_7b_stage1
else
  TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/stage2_train.parquet}
  VAL_FILE=${VAL_FILE:-${DATA_ROOT}/stage2_val.parquet}
  MODEL_PATH=${MODEL_PATH:-${RUN_ROOT}/stage1/merged_hf}
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage2}
  CALIBRATION_FILE=${CALIBRATION_FILE:-${CALIBRATION_DIR}/stage2_stage1_ckpt_kl.json}
  PRIMARY_IMAGE_KEY=images_abl
  COUNTERPART_IMAGE_KEY=images
  TRAINER_KL_FLAG=algorithm.use_kl_sep=true
  ACTOR_KL_FLAG=actor_rollout_ref.actor.use_kl_sep=false
  OTHER_TRAINER_KL_FLAG=algorithm.use_kl_cons=false
  ORIGINAL_LOG_PROB_KEY=sep_log_probs
  PROJECT_NAME=counterfactual_rlvr_7b_stage2
fi

for path in "${BIPS_ROOT}/verl/trainer/main_ppo.py" "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}" "${REWARD_FILE}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
[[ "${N_GPU}" -eq "${#CUDA_ID_ARRAY[@]}" ]] || {
  echo "N_GPU=${N_GPU} does not match CUDA_IDS=${CUDA_IDS}" >&2
  exit 1
}

available_memory_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
(( available_memory_kib >= MIN_AVAILABLE_MEMORY_GIB * 1024 * 1024 )) || {
  echo "Insufficient host memory: require ${MIN_AVAILABLE_MEMORY_GIB} GiB available" >&2
  exit 1
}
available_disk_kib=$(df --output=avail "${ROOT}" | tail -n 1 | tr -d '[:space:]')
(( available_disk_kib >= MIN_AVAILABLE_DISK_GIB * 1024 * 1024 )) || {
  echo "Insufficient disk space: require ${MIN_AVAILABLE_DISK_GIB} GiB available" >&2
  exit 1
}

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }
for gpu_id in "${CUDA_ID_ARRAY[@]}"; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d '[:space:]')
  [[ "${free_mib}" =~ ^[0-9]+$ ]] || { echo "Cannot read GPU ${gpu_id} free memory" >&2; exit 1; }
  (( free_mib >= MIN_FREE_GPU_MEMORY_MIB )) || {
    echo "GPU ${gpu_id} has ${free_mib} MiB free; require ${MIN_FREE_GPU_MEMORY_MIB} MiB" >&2
    exit 1
  }
done

mkdir -p "${OUTPUT_DIR}"/{checkpoints,logs,tensorboard,hf_cache} "${CALIBRATION_DIR}"
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/counterfactual_rlvr_7b_${STAGE}_${USER}}
mkdir -p "${RAY_TMPDIR}"
exec > >(tee -a "${OUTPUT_DIR}/logs/${MODE}.log") 2>&1

if [[ "${MODE}" == calibrate ]]; then
  EFFECTIVE_ROLLOUT_N=1
  EFFECTIVE_TOTAL_EPOCHS=1
  EFFECTIVE_MAX_STEPS=1
  EFFECTIVE_SAVE_FREQ=-1
  VAL_BEFORE_TRAIN=false
  TEST_FREQ=-1
  USE_REFERENCE_KL=false
  CALIBRATION_ONLY=true
  NONLINEAR_PARAMETER_OVERRIDE=()
else
  EFFECTIVE_ROLLOUT_N=${ROLLOUT_N}
  EFFECTIVE_TOTAL_EPOCHS=${TOTAL_EPOCHS}
  EFFECTIVE_MAX_STEPS=${MAX_STEPS}
  EFFECTIVE_SAVE_FREQ=${SAVE_FREQ}
  VAL_BEFORE_TRAIN=true
  TEST_FREQ=10
  USE_REFERENCE_KL=true
  CALIBRATION_ONLY=false
  [[ -s "${CALIBRATION_FILE}" ]] || {
    echo "Missing calibration artifact: ${CALIBRATION_FILE}. Run ${STAGE} calibrate first." >&2
    exit 1
  }
  if [[ "${STAGE}" == stage1 ]]; then
    TAU1=${TAU1:-$(python3 "${ROOT}/counterfactual_rlvr/read_calibration.py" "${CALIBRATION_FILE}" tau1)}
    NONLINEAR_PARAMETER_OVERRIDE=(algorithm.counterfactual_reward.tau="${TAU1}")
  else
    ALPHA2=${ALPHA2:-$(python3 "${ROOT}/counterfactual_rlvr/read_calibration.py" "${CALIBRATION_FILE}" alpha2)}
    NONLINEAR_PARAMETER_OVERRIDE=(algorithm.counterfactual_reward.alpha="${ALPHA2}")
  fi
fi

date -Is
echo "stage=${STAGE} mode=${MODE} model=${MODEL_PATH}"
echo "train=${TRAIN_FILE} val=${VAL_FILE} output=${OUTPUT_DIR}"
echo "cuda_ids=${CUDA_IDS} calibration=${CALIBRATION_FILE}"

cd "${BIPS_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_IDS}"
export PYTHONPATH="${BIPS_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${OUTPUT_DIR}/hf_cache"
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
export RAY_TMPDIR
export RAY_memory_usage_threshold=0.97
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export MIN_PIXELS=$((128 * 128))
export MAX_PIXELS=$((2048 * 2048))
export VLLM_WORKER_MULTIPROC_METHOD=spawn

conda run --no-capture-output -n "${CONDA_ENV}" python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  "${TRAINER_KL_FLAG}" \
  "${OTHER_TRAINER_KL_FLAG}" \
  algorithm.counterfactual_reward.enabled=true \
  algorithm.counterfactual_reward.stage="${STAGE}" \
  algorithm.counterfactual_reward.original_log_prob_key="${ORIGINAL_LOG_PROB_KEY}" \
  algorithm.counterfactual_reward.coefficient=0.3 \
  algorithm.counterfactual_reward.power=2.0 \
  algorithm.counterfactual_reward.log_ratio_clip="${LOG_RATIO_CLIP}" \
  algorithm.counterfactual_reward.calibration_only="${CALIBRATION_ONLY}" \
  algorithm.counterfactual_reward.calibration_quantile="${CALIBRATION_QUANTILE}" \
  algorithm.counterfactual_reward.calibration_sample_target="${CALIBRATION_SAMPLE_TARGET}" \
  algorithm.counterfactual_reward.calibration_output="${CALIBRATION_FILE}" \
  "${NONLINEAR_PARAMETER_OVERRIDE[@]}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.image_key="${PRIMARY_IMAGE_KEY}" \
  data.image_pres_key="$([[ "${STAGE}" == stage1 ]] && echo "${COUNTERPART_IMAGE_KEY}" || echo null)" \
  data.image_abl_key="$([[ "${STAGE}" == stage2 ]] && echo "${COUNTERPART_IMAGE_KEY}" || echo null)" \
  data.return_multi_modal_inputs=true \
  data.seed="${DATA_SEED}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length=4096 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=true \
  data.filter_overlong_prompts_workers=16 \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.nccl_timeout=3600 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.freeze_vision_tower=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.use_kl_loss="${USE_REFERENCE_KL}" \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_CONTENTS}" \
  actor_rollout_ref.actor.checkpoint.load_contents="${CHECKPOINT_CONTENTS}" \
  "${ACTOR_KL_FLAG}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.85 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.n="${EFFECTIVE_ROLLOUT_N}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enforce_eager=false \
  actor_rollout_ref.rollout.free_cache_engine=true \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  custom_reward_function.path="${REWARD_FILE}" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node="${N_GPU}" \
  trainer.nnodes=1 \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${EFFECTIVE_SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.log_val_generations=3 \
  ++trainer.rollout_generation_chunks=4 \
  ++trainer.log_prob_chunks=4 \
  ++trainer.actor_update_chunks=4 \
  trainer.total_epochs="${EFFECTIVE_TOTAL_EPOCHS}" \
  trainer.total_training_steps="${EFFECTIVE_MAX_STEPS}" \
  trainer.resume_mode=auto \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${PROJECT_NAME}_seed42" \
  "$@"
