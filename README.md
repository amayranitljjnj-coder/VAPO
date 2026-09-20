# Counterfactual RLVR for Qwen2.5-VL-7B

This repository packages the reproducible two-stage Counterfactual RLVR
experiment built from BiPS and VeRL. It trains all Qwen2.5-VL-7B-Instruct
parameters, calibrates each nonlinear policy-difference reward from a measured
KL distribution, and preserves the final resumable FSDP checkpoint plus merged
Hugging Face weights for both stages.

It also contains a Qwen2.5-VL-7B BiPS reproduction using the earlier
Qwen2.5-VL-3B-filtered ECD dataset. The two experiment tracks are independent:

| Track | Dataset | Method | Complete command | Output |
|---|---|---|---|---|
| BiPS reproduction | `BiPS_ECD_3B_filtered_v3` | Stage-1 consistency loss + Stage-2 separation loss | `./tools/run_bips_reproduction_7b_pipeline.sh` | `../training_outputs/bips_reproduction_7b/` |
| Counterfactual RLVR | `Counterfactual_RLVR_ECD_7B_clean_v1` | Detached nonlinear KL-modulated accuracy reward | `./tools/run_counterfactual_rlvr_7b_pipeline.sh` | `../training_outputs/counterfactual_rlvr_7b/` |

Read [`docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md`](docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md)
and [`docs/BIPS_7B_REPRODUCTION.md`](docs/BIPS_7B_REPRODUCTION.md) before
running. Publishing instructions are in [`docs/PUBLISHING.md`](docs/PUBLISHING.md).
Coding agents must also follow [`AGENTS.md`](AGENTS.md). The setup path
is:

```bash
./tools/setup_bips_counterfactual_rlvr.sh
./tools/download_qwen25vl_7b_github_release.sh OWNER/THIS_REPOSITORY
HF_ENDPOINT=https://huggingface.co hf download OWNER/COUNTERFACTUAL_DATASET_REPOSITORY \
  --repo-type dataset \
  --local-dir ../training_data/Counterfactual_RLVR_ECD_7B_clean_v1
HF_ENDPOINT=https://huggingface.co hf download OWNER/BIPS_3B_FILTERED_DATASET_REPOSITORY \
  --repo-type dataset \
  --local-dir ../training_data/BiPS_ECD_3B_filtered_v3
python3 tools/verify_training_package.py
```

Then launch the requested track. For a complete comparison, run BiPS
first and Counterfactual RLVR second; do not run two 7B full-parameter jobs on
the same resources concurrently.

The code repository, both training datasets, base model, and training outputs use
sibling directories. Each four-Parquet dataset is published in its own Hugging
Face dataset repository. The original model is distributed as verified,
sub-2-GiB GitHub Release assets.

## Reproducibility anchors

- BiPS base commit: `41d3a60a558c6901a8eeb4d6fbb365dc48379ce6`
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- BiPS data: `BiPS_ECD_3B_filtered_v3`, 10,781 Stage-1 and 14,860 Stage-2 rows
- Counterfactual RLVR data: `Counterfactual_RLVR_ECD_7B_clean_v1`, 12,219
  Stage-1 and 12,630 Stage-2 rows
- Counterfactual RLVR Stage 1: preserved-view rollout, `p=2`, `lambda1=0.3`
- Counterfactual RLVR Stage 2: Stage-1 final checkpoint initialization, ablated-view rollout,
  `lambda2=0.3`
- `tau1` and `alpha2`: median half-saturation calibration over 2,048 sampled
  sequences, recorded as JSON artifacts
- Dataset split seed: `20260919`, split by ECD image id

Apache-2.0 applies to the BiPS/VeRL patch and Qwen2.5-VL model components.
Dataset reuse remains subject to the terms of its source benchmarks; see the
dataset card and provenance file.
