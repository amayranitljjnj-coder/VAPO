# Qwen2.5-VL-7B 两阶段 Counterfactual RLVR 训练说明

## 1. 目标与固定控制变量

本实验从 `Qwen2.5-VL-7B-Instruct` 开始，执行两阶段、全参数 FSDP 微调。
Stage 1 使用 evidence-preserved 图像；Stage 2 必须从 Stage 1 的最终合并权重
初始化，并使用 evidence-ablated 图像。两阶段都使用同一题目、选项格式、
accuracy 判分器、GRPO 超参数和常规 reference-policy KL 稳定项；阶段之间只改变
输入视图、GT 和已定义的非线性 counterfactual reward。

数据已经定稿，训练过程不得改写或重新生成数据：

| 阶段 | train | validation | 主输入列 | 原图对照列 | 正确答案语义 |
|---|---:|---:|---|---|---|
| Stage 1 | 11,622 | 597 | `images_pres` | `images` | 原任务答案 |
| Stage 2 | 12,007 | 623 | `images_abl` | `images` | IDK / cannot determine |

数据按 ECD `image_id` 切分，train/validation 无图像泄漏。总计保留 14,140 个
unique samples；10,709 个样本跨两阶段配对。完整统计、文件哈希和排除原因位于
`split_manifest.json`、`row_manifest.jsonl` 与 `PROVENANCE.md`。

推荐的物理目录布局如下，代码、数据、模型和训练输出互不混放：

```text
WORKSPACE_ROOT/
├── VAPO_program/       # GitHub 代码仓库
├── training_data/      # Hugging Face 训练数据
├── models/             # 原始 Qwen2.5-VL-7B-Instruct
└── training_outputs/   # 两阶段 checkpoint、日志和 merged 权重
```

## 2. 现有 BiPS KL 的代码审计

审计对象为 BiPS commit
`41d3a60a558c6901a8eeb4d6fbb365dc48379ce6` 及服务器上已经用于 3B 复现的
修改。相关位置是 `verl/trainer/ppo/ray_trainer.py`、
`verl/workers/actor/dp_actor.py` 和 `verl/trainer/ppo/core_algos.py`。

1. **比较的 policy**：同一个可训练 actor 在两种图像条件下的策略。旧流程先在
   原图生成 response 并计算原图 log-prob，再把相同 prompt/response 的视觉输入
   换成 preserved 或 ablated 图，计算 `cons_log_probs` / `sep_log_probs`。它不是
   两个独立模型；常规 GRPO reference policy 是另一条独立的 KL 路径。
2. **方向**：旧 `low_var_kl` 接收 `logprob=pi_original`、
   `ref_logprob=pi_edited`，且 token 来自 original rollout，因此其期望方向为
   `D_KL(pi_original || pi_edited)`。
3. **分布范围**：它没有读取完整 vocabulary distribution。它只使用实际 rollout
   token 的两个 log-prob，通过 Schulman k3 / `low_var_kl` 估计 KL。
4. **聚合层级**：先产生逐 token 值，再由 actor loss 的 `agg_loss` 聚合；默认是
   masked token mean，随后成为 micro-batch/batch loss，而不是先为每条 sequence
   产生等权 scalar。
5. **sequence sum/mean**：旧路径没有独立 sequence-level KL；全局 token mean 会
   让较长 response 在 batch 中占更大权重。
6. **是否非负**：`low_var_kl` 使用 `r - log(r) - 1`，理论上逐点非负；代码还对
   log-ratio 和最终值做数值裁剪。若切换为 `kl/k1`，逐样本值则不保证非负。
7. **是否 detach 为 scalar reward**：不是。edited-view log-prob 是预计算常量，
   但 original-view 当前 log-prob 保留梯度；KL 作为正/负 actor loss 项参与反传，
   没有 detach，也没有调制 rule reward。

因此旧实现虽然方向可以对应论文 BiPS，但不满足本实验。它的 rollout 输入是原图，
此前本项目 3B wrapper 的 reward 为 `0.9 accuracy + 0.1 format`（BiPS 官方 7B recipe
实际使用纯 accuracy；独立 baseline 轨道已按官方实现），并且策略差异是梯度 loss，
而不是已定义的 detached scalar reward。

## 3. 本实验采用的最小改动

训练数据保持不变。数据加载时交换主视图和对照视图：Stage 1 用
`image_key=images_pres, image_pres_key=images`；Stage 2 用
`image_key=images_abl, image_abl_key=images`。因此 response 和 accuracy 都确实来自
当前阶段视图，额外 forward 只负责在相同 prompt、prefix 和 response token 上计算
原图策略 log-prob。

设当前阶段视图策略为 `q=pi_view`，原图策略为 `p=pi_original`。token 从 `q`
采样。对每个有效 response token 使用 reverse-k3：

```text
d = clip(log p(a|s) - log q(a|s), -20, 20)
r = exp(d)
K_token = r*d - r + 1
```

在固定 prefix 上，`E_(a~q)[K_token] = D_KL(p || q)`；`K_token` 逐点非负。
实现按每条 response 的有效 token 求 mean，再显式 `detach`。这仍是 rollout-token
采样估计，不是完整 vocabulary KL；它避免了 7B VLM 保存
`batch × response × vocabulary` logits 的巨大显存代价，同时修正了采样分布下的
KL 方向。log-ratio 裁剪只用于数值稳定，会对极端比率带来有限偏差。

得到的 sequence scalar 记为 `D`。accuracy reward 只接受最终
`\\boxed{A}` 到 `\\boxed{E}`，正确为 1，错误为 0；不再混入 format 分数。

Stage 1 固定为：

```text
f1(D) = (D/tau1)^2 / (1 + (D/tau1)^2)
R1 = Racc * (1 - 0.3*f1(D))
```

Stage 2 固定为：

```text
f2(D) = 1 - exp(-alpha2*D)
R2 = Racc * (1 + 0.3*f2(D))
```

错误答案始终为零；Stage 1 正确样本 reward 位于 `[0.7, 1]`，Stage 2 正确样本
reward 位于 `[1, 1.3]`。

## 4. tau1 与 alpha2 的训练前校准

两个参数都不手工填写。校准模式使用与正式训练相同的主视图和原图 forward，
但 `rollout.n=1`、不做 optimizer step、不保存 checkpoint。默认从训练集收集
2,048 条 sequence KL，记录 q10/q25/q50/q75/q90/q95/q99、mean、std、min、max。

固定选择规则是让经验中位数成为两种映射的半饱和点：

```text
tau1 = median(D_pres)
alpha2 = ln(2) / median(D_abl)
```

因此 Stage 1 在 `D=tau1` 时 `f1=0.5`；Stage 2 在经验中位数处 `f2=0.5`。
Stage 1 使用原始 7B 模型校准；Stage 2 必须使用 Stage 1 最终 checkpoint 校准。
结果写入代码仓库外的独立训练输出目录：

```text
../training_outputs/counterfactual_rlvr_7b/calibration/stage1_base_kl.json
../training_outputs/counterfactual_rlvr_7b/calibration/stage2_stage1_ckpt_kl.json
```

## 5. 环境与硬件

已验证的软件核心版本为 Python 3.10.20、PyTorch 2.6.0+cu124、Transformers
4.55.2、vLLM 0.8.4、Ray 2.56.1、VeRL 0.6.1。先创建兼容 CUDA 的环境，再执行：

```bash
git clone YOUR_GITHUB_REPOSITORY VAPO_program
cd VAPO_program
./tools/setup_bips_counterfactual_rlvr.sh
pip install -e vendor/BiPS
pip install -r requirements-training.txt
```

推荐 8 张 80GB GPU、至少 256GB host RAM，并为模型、数据、两阶段 FSDP
checkpoint、optimizer state 和 merged weights 预留至少 350GB 可用磁盘。脚本默认在
启动前检查每张 GPU 至少有 60,000 MiB 空闲、host RAM 至少 200GB、磁盘至少
350GB；任何一项不足都会退出，避免与其他用户任务争抢内存。使用不同硬件时可通过
环境变量调整，但全参数 7B 训练不能把 LoRA 当作等价替代。

## 6. 获取模型和数据

模型 release 下载脚本会取得全部小于 2GiB 的分卷，校验分卷 SHA-256，重组 tar，
再逐文件校验原始模型：

```bash
./tools/download_qwen25vl_7b_github_release.sh OWNER/REPOSITORY
```

数据从 Hugging Face dataset repo 下载，不能运行数据 builder：

```bash
HF_ENDPOINT=https://huggingface.co hf download OWNER/DATASET_REPOSITORY \
  --repo-type dataset \
  --local-dir ../training_data/Counterfactual_RLVR_ECD_7B_clean_v1
(cd ../training_data/Counterfactual_RLVR_ECD_7B_clean_v1 && \
  sha256sum -c PARQUET_SHA256SUMS)
```

最终目录必须含四个 Parquet、dataset card、provenance、split manifest 和 row
manifest。

## 7. 执行完整两阶段训练

默认 8 GPU 配置直接运行：

```bash
CUDA_IDS=0,1,2,3,4,5,6,7 \
N_GPU=8 \
./tools/run_counterfactual_rlvr_7b_pipeline.sh
```

流水线严格按以下顺序执行：

1. 用 base 7B 在 Stage 1 preserved 数据上校准 `tau1`；
2. Stage 1 全参数 GRPO；
3. 保留最终 resumable FSDP checkpoint，并合并为 Hugging Face 权重；
4. 用 Stage 1 merged checkpoint 在 Stage 2 ablated 数据上校准 `alpha2`；
5. Stage 2 全参数 GRPO；
6. 保留最终 resumable FSDP checkpoint，并合并为 Hugging Face 权重。

也可以逐步执行：

```bash
./run_counterfactual_rlvr_7b_stage1.sh calibrate
./run_counterfactual_rlvr_7b_stage1.sh train
./tools/merge_final_checkpoint.sh stage1
./run_counterfactual_rlvr_7b_stage2.sh calibrate
./run_counterfactual_rlvr_7b_stage2.sh train
./tools/merge_final_checkpoint.sh stage2
```

默认训练控制变量：5 epochs、train batch 256、GRPO rollout `n=8`、temperature
0.85、learning rate `1e-6`、data shuffle seed 42、response length 2048、full vision tower training、
gradient checkpointing、FSDP 无 CPU parameter/optimizer offload。常规 frozen-reference
KL loss 保持 `coef=0.01`，它用于 GRPO 稳定，不代替 counterfactual reward。

## 8. checkpoint、恢复与结果判定

每 50 steps 保存一次，最多保留两个训练 checkpoint；无论间隔如何，VeRL 在最后
一步强制保存。每个 checkpoint 保存 `model + optimizer + extra`，因此可以恢复完整
训练状态。两个阶段使用独立目录，Stage 2 不会覆盖 Stage 1：

```text
../training_outputs/counterfactual_rlvr_7b/stage1/checkpoints/global_step_N/actor/
../training_outputs/counterfactual_rlvr_7b/stage1/merged_hf/
../training_outputs/counterfactual_rlvr_7b/stage1/FINAL_CHECKPOINT.txt

../training_outputs/counterfactual_rlvr_7b/stage2/checkpoints/global_step_M/actor/
../training_outputs/counterfactual_rlvr_7b/stage2/merged_hf/
../training_outputs/counterfactual_rlvr_7b/stage2/FINAL_CHECKPOINT.txt
```

`resume_mode=auto` 会读取每个阶段自己的
`latest_checkpointed_iteration.txt`。只有 `FINAL_CHECKPOINT.txt` 存在、FSDP actor
目录完整、merged 目录含 config 和 safetensors，才视为阶段完成。

TensorBoard 指标包括 accuracy、最终 reward、`counterfactual/kl_*`、
`counterfactual/reward_scale_mean`、常规 actor KL、entropy、GRPO loss、gradient norm
和吞吐。校准 JSON、完整启动命令、Git commit、数据 manifest hash 应与最终模型一起
归档。

## 9. 发布维护

数据发布使用可恢复的 Hugging Face Xet upload：

```bash
pip install -r requirements-publish.txt
hf auth login
python tools/publish_hf_training_dataset.py OWNER/DATASET_REPOSITORY --public
```

代码 push 后，模型作为 GitHub Release 发布：

```bash
gh auth login
./tools/publish_qwen25vl_7b_github_release.sh OWNER/REPOSITORY
```

发布脚本先验证本地模型 hash，临时流式生成 1,900MiB 分卷，上传后删除临时分卷，
不会在项目目录复制一套 16GB 权重。模型上游为
`Qwen/Qwen2.5-VL-7B-Instruct`，Apache-2.0 license 文件随 release 一起分发。
