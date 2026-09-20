# Instructions for coding agents

## Project objective

This repository contains two independent Qwen2.5-VL-7B full-parameter RLVR
experiments with different immutable prepared datasets:

1. `bips_reproduction_7b`: reproduce the existing BiPS two-stage method with
   the historical 3B-filtered data.
2. `counterfactual_rlvr_7b`: run the new detached nonlinear KL-reward method.

Read both `docs/BIPS_7B_REPRODUCTION.md` and
`docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md` before launching training.

## Non-negotiable data rule

Do not regenerate, edit, reshuffle, filter, or rewrite either training dataset.
Use the track-specific sibling directory:

```text
BiPS reproduction:     ../training_data/BiPS_ECD_3B_filtered_v3/
Counterfactual RLVR:   ../training_data/Counterfactual_RLVR_ECD_7B_clean_v1/
```

Never point one track at the other track's dataset. Run
`python3 tools/verify_training_package.py` before training. It validates the
fixed counts and SHA-256 of all eight Parquet files.

Because both the method and dataset differ between tracks, report their final
benchmark comparison as two configured experiments, not as a method-only
controlled ablation.

## Required stage lineage

- BiPS Stage 1 starts from the original Qwen2.5-VL-7B-Instruct model.
- BiPS Stage 2 starts only from the final BiPS Stage-1 `merged_hf`.
- Counterfactual RLVR Stage 1 starts from the original model and first
  calibrates `tau1`.
- Counterfactual RLVR Stage 2 starts only from the final Counterfactual RLVR
  Stage-1 `merged_hf` and first calibrates `alpha2`.
- Never initialize either track from the other track's checkpoint.

## Setup

```bash
./tools/setup_bips_counterfactual_rlvr.sh
pip install -e vendor/BiPS
python3 tools/verify_training_package.py
```

The base model default is `../models/Qwen2.5-VL-7B-Instruct`. Obtain it from
the official Hugging Face repository or the project's GitHub Release before
training.

## Launch commands

BiPS reproduction:

```bash
CUDA_IDS=0,1,2,3,4,5,6,7 N_GPU=8 \
  ./tools/run_bips_reproduction_7b_pipeline.sh
```

Counterfactual RLVR:

```bash
CUDA_IDS=0,1,2,3,4,5,6,7 N_GPU=8 \
  ./tools/run_counterfactual_rlvr_7b_pipeline.sh
```

When asked to execute the complete comparison and no order is specified, run
BiPS first and Counterfactual RLVR second. Run them sequentially, never on the
same GPUs at the same time.

## Resource and checkpoint requirements

Use 8 GPUs with approximately 80GB memory each, at least 200GiB available host
RAM, and at least 350GiB free workspace disk. The launchers enforce these
thresholds. Do not lower a guard merely to make a busy machine accept a job;
first identify other GPU and host-memory users.

Both stages of both tracks must retain:

- the final resumable FSDP checkpoint (`model`, `optimizer`, `extra`);
- a merged Hugging Face model;
- `FINAL_CHECKPOINT.txt`;
- logs and TensorBoard output.

Outputs are isolated under:

```text
../training_outputs/bips_reproduction_7b/
../training_outputs/counterfactual_rlvr_7b/
```

Do not delete a Stage-1 final checkpoint after Stage 2 starts.

## Method boundaries

For the BiPS reproduction, keep `algorithm.counterfactual_reward.enabled=false`,
the official accuracy-only reward (`format_score=0.0`), Stage-1 consistency loss,
and Stage-2 separation loss. Do not add `tau1` or `alpha2`.

For Counterfactual RLVR, keep BiPS actor consistency/separation loss disabled.
Use accuracy-only reward and the detached reverse-k3 sequence-mean multiplier
defined in the training document.
