#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 stage1|stage2 [Hydra overrides ...]" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
STAGE=$1
shift
[[ "${STAGE}" == stage1 || "${STAGE}" == stage2 ]] || usage

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "${ROOT}/.." && pwd)}
BIPS_ROOT=${BIPS_ROOT:-${ROOT}/vendor/BiPS}
DATA_ROOT=${DATA_ROOT:-${WORKSPACE_ROOT}/training_data/BiPS_ECD_3B_filtered_v3}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-${WORKSPACE_ROOT}/models/Qwen2.5-VL-7B-Instruct}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE_ROOT}/training_outputs/bips_reproduction_7b}
REWARD_FILE=${REWARD_FILE:-${ROOT}/bips_reproduction/reward_paper.py}

CUDA_IDS=${CUDA_IDS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a CUDA_ID_ARRAY <<<"${CUDA_IDS}"
N_GPU=${N_GPU:-${#CUDA_ID_ARRAY[@]}}
CONDA_ENV=${CONDA_ENV:-bips}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
DATA_SEED=${DATA_SEED:-42}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-50}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
CHECKPOINT_CONTENTS=${CHECKPOINT_CONTENTS:-'["model","optimizer","extra"]'}
MIN_AVAILABLE_MEMORY_GIB=${MIN_AVAILABLE_MEMORY_GIB:-200}
MIN_AVAILABLE_DISK_GIB=${MIN_AVAILABLE_DISK_GIB:-350}
MIN_FREE_GPU_MEMORY_MIB=${MIN_FREE_GPU_MEMORY_MIB:-60000}

if [[ "${STAGE}" == stage1 ]]; then
  TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/stage1_train.parquet}
  VAL_FILE=${VAL_FILE:-${DATA_ROOT}/stage1_val.parquet}
  MODEL_PATH=${MODEL_PATH:-${BASE_MODEL_PATH}}
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage1}
  IMAGE_PRES_KEY=images_pres
  IMAGE_ABL_KEY=null
  USE_KL_CONS=true
  USE_KL_SEP=false
  PROJECT_NAME=bips_reproduction_7b_stage1
else
  TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/stage2_train.parquet}
  VAL_FILE=${VAL_FILE:-${DATA_ROOT}/stage2_val.parquet}
  MODEL_PATH=${MODEL_PATH:-${RUN_ROOT}/stage1/merged_hf}
  OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage2}
  IMAGE_PRES_KEY=null
  IMAGE_ABL_KEY=images_abl
  USE_KL_CONS=false
  USE_KL_SEP=true
  PROJECT_NAME=bips_reproduction_7b_stage2
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
available_disk_kib=$(df --output=avail "${WORKSPACE_ROOT}" | tail -n 1 | tr -d '[:space:]')
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

mkdir -p "${OUTPUT_DIR}"/{checkpoints,logs,tensorboard,hf_cache}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/bips_reproduction_7b_${STAGE}_${USER}}
mkdir -p "${RAY_TMPDIR}"
exec > >(tee -a "${OUTPUT_DIR}/logs/train.log") 2>&1

date -Is
echo "experiment=bips_reproduction stage=${STAGE} model=${MODEL_PATH}"
echo "train=${TRAIN_FILE} val=${VAL_FILE} output=${OUTPUT_DIR}"
echo "cuda_ids=${CUDA_IDS} use_kl_cons=${USE_KL_CONS} use_kl_sep=${USE_KL_SEP}"

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
  algorithm.use_kl_cons="${USE_KL_CONS}" \
  algorithm.use_kl_sep="${USE_KL_SEP}" \
  algorithm.counterfactual_reward.enabled=false \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.image_key=images \
  data.image_pres_key="${IMAGE_PRES_KEY}" \
  data.image_abl_key="${IMAGE_ABL_KEY}" \
  data.return_multi_modal_inputs=true \
  data.seed="${DATA_SEED}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=true \
  data.filter_overlong_prompts_workers=32 \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.nccl_timeout=3600 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.freeze_vision_tower=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_CONTENTS}" \
  actor_rollout_ref.actor.checkpoint.load_contents="${CHECKPOINT_CONTENTS}" \
  actor_rollout_ref.actor.use_kl_cons="${USE_KL_CONS}" \
  actor_rollout_ref.actor.kl_cons_penalty_type=low_var_kl \
  actor_rollout_ref.actor.kl_cons_coef=0.01 \
  actor_rollout_ref.actor.kl_cons_clip=1.0 \
  actor_rollout_ref.actor.use_kl_sep="${USE_KL_SEP}" \
  actor_rollout_ref.actor.kl_sep_penalty_type=low_var_kl \
  actor_rollout_ref.actor.kl_sep_coef=0.02 \
  actor_rollout_ref.actor.kl_sep_clip=0.2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.85 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enforce_eager=false \
  actor_rollout_ref.rollout.free_cache_engine=true \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  custom_reward_function.path="${REWARD_FILE}" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node="${N_GPU}" \
  trainer.nnodes=1 \
  trainer.val_before_train=true \
  trainer.test_freq=10 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.log_val_generations=3 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${MAX_STEPS}" \
  trainer.resume_mode=auto \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${PROJECT_NAME}_seed42" \
  "$@"
