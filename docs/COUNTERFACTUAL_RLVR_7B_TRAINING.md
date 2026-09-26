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
HF_ENDPOINT=https://huggingface.co hf download amayranitljjnj/counterfactual-rlvr-ecd-7b-clean-v1 \
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

## 8. 前期快速实验配置（半天至一天目标）

在运行默认 5 epochs、`rollout.n=8` 的正式实验之前，可以先用下面的完整数据快速
配置筛选 reward 与训练方向。目标是在文档所述 8×80GB GPU 上将单次完整两阶段实验
压缩至约半天到一天；实际耗时仍取决于 GPU 型号、共享负载、I/O 和生成长度分布，
必须记录实测 wall-clock，不能把时间目标表述成保证。

快速实验只改变计算预算，不改变数据、reward 定义、两阶段顺序或 checkpoint 血缘：

| 项目 | 快速实验值 | 正式实验默认值 |
|---|---:|---:|
| 数据 | 完整 Stage-1/Stage-2 训练集 | 完整训练集 |
| epochs | `1` | `5` |
| GRPO rollout `n` | `4` | `8` |
| maximum response length | `1024` | `2048` |
| 每阶段校准序列数 | `512` | `2048` |
| 训练前验证 | 关闭 | 开启 |
| 训练中验证/checkpoint | 不做周期性操作，仅最后一步 | 每 10/50 steps |
| 输出目录 | 独立 pilot `RUN_ROOT` | `counterfactual_rlvr_7b` |

### 8.1 两阶段冒烟实验

完整一轮 pilot 前，先用每阶段 5 个全局训练 step 跑通同一配置的关键路径。冒烟仍然
读取原始完整 Parquet；`MAX_STEPS=5` 只提前终止训练器，不创建子集、不改写数据。
冒烟的目标是验证配置与工程链路，因此 5 steps 后的 accuracy 是否提升不作为通过条件。

先完成静态校验和 reward 单元测试：

```bash
python3 tools/verify_training_package.py
./tools/setup_bips_counterfactual_rlvr.sh
(cd vendor/BiPS && python3 -m pytest -q tests/trainer/ppo/test_counterfactual_reward.py)
```

然后从 `VAPO_program` 执行完整两阶段冒烟。每次冒烟都使用新的目录后缀，避免
`resume_mode=auto` 读到旧结果：

```bash
(
SMOKE_WORKSPACE_ROOT=$(cd .. && pwd)
export RUN_ROOT="${SMOKE_WORKSPACE_ROOT}/training_outputs/counterfactual_rlvr_7b_smoke_001"
export TOTAL_EPOCHS=1
export ROLLOUT_N=4
export CALIBRATION_SAMPLE_TARGET=512
export MAX_STEPS=5
export SAVE_FREQ=1000000

SMOKE_DATA_OVERRIDES=(
  data.max_response_length=1024
)
SMOKE_TRAIN_OVERRIDES=(
  data.max_response_length=1024
  trainer.val_before_train=false
  trainer.test_freq=1000000
)

./run_counterfactual_rlvr_7b_stage1.sh calibrate "${SMOKE_DATA_OVERRIDES[@]}"
python3 counterfactual_rlvr/read_calibration.py \
  "${RUN_ROOT}/calibration/stage1_base_kl.json" tau1
./run_counterfactual_rlvr_7b_stage1.sh train "${SMOKE_TRAIN_OVERRIDES[@]}"
./tools/merge_final_checkpoint.sh stage1

./run_counterfactual_rlvr_7b_stage2.sh calibrate "${SMOKE_DATA_OVERRIDES[@]}"
python3 counterfactual_rlvr/read_calibration.py \
  "${RUN_ROOT}/calibration/stage2_stage1_ckpt_kl.json" alpha2
./run_counterfactual_rlvr_7b_stage2.sh train "${SMOKE_TRAIN_OVERRIDES[@]}"
./tools/merge_final_checkpoint.sh stage2

test -s "${RUN_ROOT}/stage1/FINAL_CHECKPOINT.txt"
test -s "${RUN_ROOT}/stage1/merged_hf/config.json"
test -s "${RUN_ROOT}/stage2/FINAL_CHECKPOINT.txt"
test -s "${RUN_ROOT}/stage2/merged_hf/config.json"
)
```

冒烟通过需要同时满足：

1. 数据校验、reward 测试、两次校准、两阶段训练和两次模型合并全部正常退出；
2. 两份校准 JSON 中选择的半饱和 KL、均值、标准差和分位数均为有限值，`tau1`、
   `alpha2` 为正；
3. 日志没有 OOM、NaN/Inf、持续异常的 gradient norm，Stage 1/Stage 2 的
   `counterfactual/kl_*`、reward scale、accuracy 和 GRPO loss 均有记录；
4. Stage 1 正确样本的 reward scale 位于 `[0.7,1]`，Stage 2 位于 `[1,1.3]`，
   错误答案 reward 保持为零；
5. 两个阶段都只在最后一步验证和保存，FSDP checkpoint、merged model 与
   `FINAL_CHECKPOINT.txt` 完整；Stage 2 日志中的模型路径属于本次 smoke 的
   `stage1/merged_hf`；
6. 检查 response-length/clip 指标和生成样例。如果触及 1024-token 上限的比例超过
   10%，至少人工复查 20 条；若大量输出缺少最终答案，先调整长度预算再运行 pilot；
7. 用去掉首步预热后的平均 step 时间分别估算两个阶段完整一轮的耗时，加上校准、
   末步验证、保存、合并与统一评测，再乘 `1.3` 余量后不超过 24 小时。

五步训练的准确率波动很大，不能用它证明方法有效或无效。冒烟产出的 checkpoint、
`tau1` 和 `alpha2` 只服务于本次链路检查；完整 pilot 必须使用新的 `RUN_ROOT` 从 base
重新校准和训练，Stage 2 也必须基于 pilot 自己的 Stage-1 最终模型重新校准。

### 8.2 完整一轮快速实验

完整快速实验不要覆盖 `TRAIN_FILE`、`DATA_ROOT` 或 `MAX_STEPS`，这样仍然使用完整训练数据并由
`TOTAL_EPOCHS=1` 控制一轮训练。快速实验也必须先校准 Stage 1，Stage 1 训练后合并
最终权重，再用该 Stage-1 `merged_hf` 校准和训练 Stage 2。

从 `VAPO_program` 执行：

```bash
PILOT_WORKSPACE_ROOT=$(cd .. && pwd)
export RUN_ROOT="${PILOT_WORKSPACE_ROOT}/training_outputs/counterfactual_rlvr_7b_pilot_1e_r4_1024"
export TOTAL_EPOCHS=1
export ROLLOUT_N=4
export CALIBRATION_SAMPLE_TARGET=512

# 必须保持为正数；取远大于单轮总步数的值，使 checkpoint 只在末步保存。
export SAVE_FREQ=1000000

PILOT_DATA_OVERRIDES=(
  data.max_response_length=1024
)
PILOT_TRAIN_OVERRIDES=(
  data.max_response_length=1024
  trainer.val_before_train=false
  trainer.test_freq=1000000
)

./run_counterfactual_rlvr_7b_stage1.sh calibrate "${PILOT_DATA_OVERRIDES[@]}"
./run_counterfactual_rlvr_7b_stage1.sh train "${PILOT_TRAIN_OVERRIDES[@]}"
./tools/merge_final_checkpoint.sh stage1

./run_counterfactual_rlvr_7b_stage2.sh calibrate "${PILOT_DATA_OVERRIDES[@]}"
./run_counterfactual_rlvr_7b_stage2.sh train "${PILOT_TRAIN_OVERRIDES[@]}"
./tools/merge_final_checkpoint.sh stage2
```

这里不能把 `trainer.test_freq` 或 `trainer.save_freq` 设为 `-1`：在当前 VeRL 逻辑中，
负数会同时禁用最后一步的验证或保存。使用远大于总步数的正数时，周期条件不会触发，
但 `is_last_step` 仍会在末步各执行一次。`trainer.val_before_train=false` 另外关闭训练前
验证。上述 `RUN_ROOT` 是绝对路径，并与正式实验目录隔离，因此 `resume_mode=auto` 只会
在本次 pilot 内恢复；开始一个全新的 pilot 时应换一个新的目录名。

### 8.3 快速实验的结果判定

快速配置的目标是筛选，而不是预先保证提升。至少保留下列同协议结果：

1. 在同一冻结评测集和同一生成配置下，比较 base 与 Stage-1 `merged_hf`，Stage 1
   应获得正向且有实际意义的提升；
2. 在同一冻结评测集和同一生成配置下，比较 Stage-1 与 Stage-2 `merged_hf`，Stage 2
   应进一步提升目标指标，同时检查原任务能力没有不可接受的退化；
3. 不直接用 Stage-1 validation accuracy 与 Stage-2 validation accuracy 的数值大小证明
   “Stage 2 优于 Stage 1”，因为两个阶段的图像视图和 GT 语义不同；必须让待比较的
   checkpoint 运行同一套评测；
4. 单次正向结果只能作为筛选信号。若要声称“稳定提升”，应对入选配置补做多个随机
   种子或独立重复，并报告均值、离散程度、完整命令、校准 JSON 和 wall-clock。

如果 pilot 未达到上述两级提升，不应仅凭缩短后的训练结果否定正式配置；应先检查
Stage-1/Stage-2 各自的 reward、KL、response length 截断率及验证样例，再决定调整
训练预算或 reward 超参数。

## 9. checkpoint、恢复与结果判定

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

快速实验使用第 8 节的独立输出目录，并把周期保存频率设为远大于总步数的正数，因此
只保留最后一步 checkpoint。正式实验仍使用本节默认的每 50 steps 保存策略。

## 10. 发布维护

数据发布使用可恢复的 Hugging Face Xet upload：

```bash
pip install -r requirements-publish.txt
hf auth login
python tools/publish_hf_training_dataset.py \
  amayranitljjnj/counterfactual-rlvr-ecd-7b-clean-v1 \
  --folder ../training_data/Counterfactual_RLVR_ECD_7B_clean_v1 \
  --public
```

代码 push 后，模型作为 GitHub Release 发布：

```bash
gh auth login
./tools/publish_qwen25vl_7b_github_release.sh OWNER/REPOSITORY
```

发布脚本先验证本地模型 hash，临时流式生成 1,900MiB 分卷，上传后删除临时分卷，
不会在项目目录复制一套 16GB 权重。模型上游为
`Qwen/Qwen2.5-VL-7B-Instruct`，Apache-2.0 license 文件随 release 一起分发。
