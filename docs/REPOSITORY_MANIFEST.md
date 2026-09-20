# GitHub 仓库文件清单

本文件列出 GitHub 代码仓库实际跟踪的全部文件及用途。原始模型和训练 Parquet
通过独立的发布渠道提供，不进入普通 Git 历史。

## 根目录

| 文件 | 用途 |
|---|---|
| `.gitattributes` | Git 对补丁文件的属性设置。 |
| `.gitignore` | 排除模型、数据、输出、缓存和密钥。 |
| `AGENTS.md` | coding agent 必读的实验边界、执行顺序、资源与 checkpoint 规则。 |
| `LICENSE` | 训练代码的 Apache-2.0 许可证。 |
| `README.md` | 仓库首页、实验摘要和最短运行流程。 |
| `requirements-training.txt` | 已验证的训练环境核心版本。 |
| `requirements-publish.txt` | Hugging Face/Xet 发布依赖。 |
| `run_counterfactual_rlvr_7b.sh` | 两阶段共用训练入口，包含资源检查、Hydra 配置和全量微调参数。 |
| `run_counterfactual_rlvr_7b_stage1.sh` | Stage 1 preserved-view 校准/训练入口。 |
| `run_counterfactual_rlvr_7b_stage2.sh` | Stage 2 ablated-view 校准/训练入口。 |
| `run_bips_reproduction_7b.sh` | BiPS 7B 两阶段共用训练入口。 |
| `run_bips_reproduction_7b_stage1.sh` | BiPS Stage 1 consistency 训练入口。 |
| `run_bips_reproduction_7b_stage2.sh` | BiPS Stage 2 separation 训练入口。 |

## `counterfactual_rlvr/`

| 文件 | 用途 |
|---|---|
| `README.md` | reward 子模块说明。 |
| `reward_accuracy.py` | A-E boxed-answer 二值 accuracy reward；错误答案严格为零。 |
| `read_calibration.py` | 从校准 JSON 读取 `tau1` 或 `alpha2`。 |
| `Qwen2.5-VL-7B-Instruct.sha256` | 原始 7B 模型 16 个实质文件的 SHA-256。 |
| `model_files.txt` | 模型发布和恢复时应包含的文件清单。 |

## `docs/`

| 文件 | 用途 |
|---|---|
| `COUNTERFACTUAL_RLVR_7B_TRAINING.md` | 完整中文训练文档：KL 审计、reward、校准规则、硬件、数据、训练与 checkpoint。 |
| `REPOSITORY_MANIFEST.md` | 当前文件清单。 |
| `BIPS_7B_REPRODUCTION.md` | 3B 难度过滤数据上的 7B 全参数 BiPS 方法复刻说明。 |
| `PUBLISHING.md` | GitHub 代码/模型 Release 与两个 Hugging Face 数据仓库的发布流程。 |

## `bips_reproduction/`

| 文件 | 用途 |
|---|---|
| `README.md` | BiPS baseline 轨道入口说明。 |
| `reward_paper.py` | 与 BiPS 官方 `utils/reward.py` 等价的纯 accuracy rule reward。 |

## 独立训练数据目录

训练数据不位于代码仓库内。两条实验轨道使用不同的物理目录：

```text
Counterfactual RLVR:
/mnt/sdc_data/huaming/VAPO/training_data/Counterfactual_RLVR_ECD_7B_clean_v1/

BiPS reproduction:
/mnt/sdc_data/huaming/VAPO/training_data/BiPS_ECD_3B_filtered_v3/
```

Counterfactual RLVR 数据规模：

```text
stage1_train.parquet   11,622 rows
stage1_val.parquet        597 rows
stage2_train.parquet   12,007 rows
stage2_val.parquet        623 rows
```

BiPS 3B 难度过滤数据规模：

```text
stage1_train.parquet   10,257 rows
stage1_val.parquet        524 rows
stage2_train.parquet   14,136 rows
stage2_val.parquet        724 rows
```

每个目录均包含 dataset card、`PROVENANCE.md`、`PARQUET_SHA256SUMS` 和
`split_manifest.json`。Counterfactual RLVR 数据另外包含
`row_manifest.jsonl` 和 `excluded_samples.jsonl`，并由数据
发布脚本分别上传到 Hugging Face 数据仓库。

## `patches/`

| 文件 | 用途 |
|---|---|
| `bips_counterfactual_rlvr.patch` | 在固定 BiPS commit 上加入 Counterfactual RLVR 与内存/checkpoint 修复。 |

补丁实际修改或增加：

```text
tests/trainer/ppo/test_counterfactual_reward.py
verl/trainer/config/algorithm.py
verl/trainer/config/bips_trainer.yaml
verl/trainer/ppo/counterfactual_reward.py
verl/trainer/ppo/ray_trainer.py
verl/utils/checkpoint/fsdp_checkpoint_manager.py
verl/workers/actor/dp_actor.py
verl/workers/fsdp_workers.py
```

`tools/setup_bips_counterfactual_rlvr.sh` 会 clone 固定 commit，再自动应用此补丁，
因此 GitHub 仓库不需要嵌入完整 BiPS 副本。

## `licenses/`

| 文件 | 用途 |
|---|---|
| `Qwen2.5-VL-7B-Instruct-APACHE-2.0.txt` | 随模型 Release 提供的 Qwen2.5-VL 许可证文本。 |

## `tools/`

| 文件 | 用途 |
|---|---|
| `setup_bips_counterfactual_rlvr.sh` | clone 固定 BiPS commit 并应用补丁。 |
| `run_counterfactual_rlvr_7b_pipeline.sh` | 顺序执行 Stage 1 校准/训练/合并和 Stage 2 校准/训练/合并。 |
| `run_bips_reproduction_7b_pipeline.sh` | 顺序执行 BiPS Stage 1/Stage 2 训练与合并。 |
| `merge_final_checkpoint.sh` | 将指定阶段最终 FSDP checkpoint 合并为 Hugging Face 权重。 |
| `verify_training_package.py` | 校验脚本、样本量和四个 Parquet SHA-256。 |
| `publish_github_training_repo.sh` | 创建并推送 GitHub 代码仓库。 |
| `publish_hf_training_dataset.py` | 创建并上传 Hugging Face 数据仓库。 |
| `publish_qwen25vl_7b_github_release.sh` | 校验并将 7B 模型作为小于 2GiB 的 Release 分卷发布。 |
| `download_qwen25vl_7b_github_release.sh` | 下载、校验和重组模型 Release。 |

## 明确不进入 GitHub 普通 Git 历史的内容

```text
../models/                      原始模型
../training_data/               大型训练数据
../training_outputs/            checkpoint、日志和 TensorBoard
vendor/BiPS/                    本地修改工作树（由固定 commit + patch 重建）
configs/api_keys.env            API 凭证
.env / *.key / *.pem            其他凭证
```

其他用户需要的实质内容仍然完整可获得：代码来自 GitHub clone，数据来自 Hugging
Face，原始模型可来自官方 `Qwen/Qwen2.5-VL-7B-Instruct` 或本项目的 GitHub
Release。
