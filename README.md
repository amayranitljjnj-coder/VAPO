# Counterfactual RLVR for Qwen2.5-VL-7B

This repository packages the reproducible two-stage **Counterfactual RLVR** experiment built from BiPS and VeRL. It trains all Qwen2.5-VL-7B-Instruct parameters, calibrates each nonlinear policy-difference reward from a measured KL distribution, and preserves the final resumable FSDP checkpoint plus merged Hugging Face weights for both stages.

It also contains a **Qwen2.5-VL-7B BiPS reproduction** using the earlier Qwen2.5-VL-3B-filtered ECD dataset.

The two experiment tracks are independent:

| Track | Dataset | Method | Complete command | Output |
|---|---|---|---|---|
| BiPS reproduction | [`BiPS_ECD_3B_filtered_v3`](https://huggingface.co/datasets/amayranitljjnj/bips-ecd-3b-filtered-v3) | Stage-1 consistency loss + Stage-2 separation loss | `./tools/run_bips_reproduction_7b_pipeline.sh` | `../training_outputs/bips_reproduction_7b/` |
| Counterfactual RLVR | [`Counterfactual_RLVR_ECD_7B_clean_v1`](https://huggingface.co/datasets/amayranitljjnj/counterfactual-rlvr-ecd-7b-clean-v1) | Detached nonlinear KL-modulated accuracy reward | `./tools/run_counterfactual_rlvr_7b_pipeline.sh` | `../training_outputs/counterfactual_rlvr_7b/` |

## Resources

### Code

GitHub repository:

https://github.com/amayranitljjnj-coder/VAPO

### Training Datasets

Counterfactual RLVR dataset:

https://huggingface.co/datasets/amayranitljjnj/counterfactual-rlvr-ecd-7b-clean-v1

BiPS reproduction dataset:

https://huggingface.co/datasets/amayranitljjnj/bips-ecd-3b-filtered-v3

### Base Model

Qwen2.5-VL-7B-Instruct:

https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

---

## Documentation

Read [`docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md`](docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md) and [`docs/BIPS_7B_REPRODUCTION.md`](docs/BIPS_7B_REPRODUCTION.md) before running the experiments.

Publishing instructions are in [`docs/PUBLISHING.md`](docs/PUBLISHING.md).

Coding agents should also follow [`AGENTS.md`](AGENTS.md).

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/amayranitljjnj-coder/VAPO.git
cd VAPO
```

### 2. Prepare BiPS / VeRL

```bash
./tools/setup_bips_counterfactual_rlvr.sh
```

The setup script clones the required BiPS repository, checks out the fixed base commit, and applies the patches included in this repository.

### 3. Prepare the base model

If using the GitHub Release model package prepared for this repository:

```bash
./tools/download_qwen25vl_7b_github_release.sh amayranitljjnj-coder/VAPO
```

The expected model directory is:

```text
../models/Qwen2.5-VL-7B-Instruct/
```

Alternatively, users may prepare `Qwen/Qwen2.5-VL-7B-Instruct` themselves and place it at the same path.

### 4. Download the Counterfactual RLVR dataset

```bash
HF_ENDPOINT=https://huggingface.co hf download \
  amayranitljjnj/counterfactual-rlvr-ecd-7b-clean-v1 \
  --repo-type dataset \
  --local-dir ../training_data/Counterfactual_RLVR_ECD_7B_clean_v1
```

### 5. Download the BiPS reproduction dataset

```bash
HF_ENDPOINT=https://huggingface.co hf download \
  amayranitljjnj/bips-ecd-3b-filtered-v3 \
  --repo-type dataset \
  --local-dir ../training_data/BiPS_ECD_3B_filtered_v3
```

The resulting workspace layout should be approximately:

```text
workspace/
├── VAPO/
├── models/
│   └── Qwen2.5-VL-7B-Instruct/
├── training_data/
│   ├── Counterfactual_RLVR_ECD_7B_clean_v1/
│   └── BiPS_ECD_3B_filtered_v3/
└── training_outputs/
```

### 6. Verify the training package

```bash
python3 tools/verify_training_package.py
```

---

## Training

### Fast Counterfactual RLVR pilot

Before committing to the five-epoch recipe, run the documented
five-step-per-stage smoke test and then the full-data, one-epoch pilot. Both
use `ROLLOUT_N=4`, a 1,024-token response limit, 512 calibration sequences,
and only final-step validation/checkpointing.

The smoke test exercises Stage-1 calibration/training/merge and Stage-2
calibration/training/merge. Smoke and pilot runs must use different dedicated
`RUN_ROOT` paths, and the one-epoch pilot starts again from the base model.

Complete commands, acceptance gates, runtime projection, and evaluation rules
are in the
[`Fast pilot` section](docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md#8-前期快速实验配置半天至一天目标)
of the training guide. The half-day to one-day duration is a target to verify
on the actual machine.

### BiPS Reproduction

Run the complete two-stage BiPS reproduction:

```bash
CUDA_IDS=0,1,2,3,4,5,6,7 \
N_GPU=8 \
./tools/run_bips_reproduction_7b_pipeline.sh
```

The outputs are written to:

```text
../training_outputs/bips_reproduction_7b/
```

For stage-by-stage execution, see:

[`docs/BIPS_7B_REPRODUCTION.md`](docs/BIPS_7B_REPRODUCTION.md)

---

### Counterfactual RLVR

Run the complete two-stage Counterfactual RLVR pipeline:

```bash
CUDA_IDS=0,1,2,3,4,5,6,7 \
N_GPU=8 \
./tools/run_counterfactual_rlvr_7b_pipeline.sh
```

The outputs are written to:

```text
../training_outputs/counterfactual_rlvr_7b/
```

For stage-by-stage execution and reward calibration details, see:

[`docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md`](docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md)

---

## Experiment Organization

The code repository, both training datasets, base model, and training outputs use sibling directories.

Each four-Parquet training dataset is published independently on Hugging Face:

- [`amayranitljjnj/counterfactual-rlvr-ecd-7b-clean-v1`](https://huggingface.co/datasets/amayranitljjnj/counterfactual-rlvr-ecd-7b-clean-v1)
- [`amayranitljjnj/bips-ecd-3b-filtered-v3`](https://huggingface.co/datasets/amayranitljjnj/bips-ecd-3b-filtered-v3)

For a complete comparison, we recommend running the BiPS reproduction first and Counterfactual RLVR second.

Do not run two 7B full-parameter training jobs on the same GPU resources concurrently.

---

## Reproducibility Anchors

- **BiPS base commit:** `41d3a60a558c6901a8eeb4d6fbb365dc48379ce6`
- **Base model:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **BiPS data:** `BiPS_ECD_3B_filtered_v3`
  - Stage 1: 10,781 rows
  - Stage 2: 14,860 rows
- **Counterfactual RLVR data:** `Counterfactual_RLVR_ECD_7B_clean_v1`
  - Stage 1: 12,219 rows
  - Stage 2: 12,630 rows
- **Counterfactual RLVR Stage 1:** preserved-view rollout, `p=2`, `lambda1=0.3`
- **Counterfactual RLVR Stage 2:** Stage-1 final checkpoint initialization, ablated-view rollout, `lambda2=0.3`
- **`tau1` and `alpha2`:** median half-saturation calibration over 2,048 sampled sequences, recorded as JSON artifacts
- **Dataset split seed:** `20260919`, split by ECD image id

---

## License

Apache-2.0 applies to the BiPS/VeRL patch and Qwen2.5-VL model components included or referenced by this project.

Dataset reuse remains subject to the licenses and terms of the original source benchmarks. See the dataset cards and provenance files in the corresponding Hugging Face dataset repositories for details.
