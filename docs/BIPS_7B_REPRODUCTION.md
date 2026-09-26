# Qwen2.5-VL-7B BiPS 复刻实验

## 1. 实验目标

本轨道使用 `Qwen2.5-VL-7B-Instruct` 全参数训练，但数据切换为此前由
`Qwen2.5-VL-3B-Instruct` 八次 rollout 做难度过滤的版本，并按 BiPS 官方 7B recipe
运行双阶段优化。它与新的 Counterfactual RLVR 轨道使用不同的数据目录。

这里复刻的是 **BiPS 方法与训练协议**。数据生成中曾以 DeepSeek V4 Pro 代替论文的
GPT-5-mini，并以 3B 模型代替 7B 模型做难度过滤，因此它仍不是对论文数据和最终数字的
逐样本复现。报告结果时应写作
“BiPS 7B reproduction on the ECD 3B-filtered v3 data”。

禁止重新生成、改写、重新过滤或重新划分数据。数据目录固定为：

```text
../training_data/BiPS_ECD_3B_filtered_v3/
```

| 阶段 | train | validation | rollout 主图 | 对照图 | GT |
|---|---:|---:|---|---|---|
| Stage 1 | 10,257 | 524 | `images` | `images_pres` | 原始 A-D 任务答案 |
| Stage 2 | 14,136 | 724 | `images` | `images_abl` | 原始 A-D 任务答案 |

总规模为 Stage 1 10,781、Stage 2 14,860。该版本没有新增 IDK 选项，也没有将
Stage-2 GT 改成 IDK；这正是此前 BiPS 复现使用的 3B 难度过滤数据语义。

因此 BiPS 与 Counterfactual RLVR 的最终 benchmark 对比同时包含“训练方法”和“训练
数据”两个变量，不能表述成只改变 reward 的严格单变量消融。

## 2. BiPS reward 与 policy-difference loss

两阶段 rule reward 严格采用仓库内 BiPS 官方入口
`vendor/BiPS/utils/reward.py` 的定义：

```text
R_BiPS = R_accuracy
```

`R_accuracy` 检查最终 boxed option letter。官方实现调用
`geo3k.compute_score(..., format_score=0.0)`；错误答案不会因格式正确获得 0.1 分。
本轨道的等价封装位于 `bips_reproduction/reward_paper.py`。此前本项目 3B 脚本使用过
`0.9 accuracy + 0.1 format`，那是历史复现设置，不用于这次 7B BiPS 轨道。

BiPS policy difference 不作为 detached scalar reward。它在 actor update 中保留当前
policy log-prob 的梯度，使用 rollout token 上的 `low_var_kl`（k3）并加入 policy
loss。它不是完整 vocabulary KL。

### Stage 1：consistency

1. 使用原图 `images` 生成 rollout；
2. 对相同 response token 补算 preserved 图 `images_pres` 的 log-prob；
3. 计算 `low_var_kl(pi_original || pi_preserved)`；
4. 只在 rule reward 大于等于 0.5 的正确样本上启用 consistency mask；
5. 使用 `kl_cons_coef=0.01`、`kl_cons_clip=1.0`，将 consistency loss 加到 actor loss，
   从而缩小原图与 preserved 图策略差异。

### Stage 2：separation

1. 从 BiPS Stage-1 最终 merged checkpoint 初始化；
2. 使用原图 `images` 生成 rollout；
3. 对相同 response token 补算 ablated 图 `images_abl` 的 log-prob；
4. 使用 `kl_sep_coef=0.02`、`kl_sep_clip=0.2`；
5. 从 actor loss 中减去 separation loss，从而增大原图与 ablated 图策略差异。

两阶段仍保留标准 GRPO frozen-reference KL loss：`coef=0.01`、
`type=low_var_kl`。Counterfactual RLVR 的 `tau1`、`alpha2`、reverse-k3 scalar 和
非线性 reward 在本轨道全部关闭。

## 3. BiPS 官方训练参数

优化超参数按仓库内 BiPS 官方 7B recipe
`vendor/BiPS/recipe/bips/train_stage{1,2}.sh` 设置：

```text
base model                 Qwen2.5-VL-7B-Instruct
full parameter training    true（vision tower 也训练）
epochs                     5
train batch                256
PPO mini-batch             64
rollout n                  8
temperature                0.85
top_p                      1.0
learning rate              1e-6
data shuffle seed          42
PPO micro-batch/GPU        4
rollout/ref logprob batch  32/GPU
rollout GPU utilization    0.7
max prompt/response        4096 / 4096
gradient checkpointing     true
FSDP CPU offload           false
checkpoint contents        model + optimizer + extra
```

`top_p=1.0`、`top_k=-1` 沿用 BiPS/verl 默认值。与官方 recipe 相比，本脚本只增加
资源前置检查、离线模型路径、固定的数据 shuffle seed、完整可恢复 checkpoint、
TensorBoard 日志和独立输出目录；这些改动不改变 reward 或 actor loss 定义。

## 4. 执行方式

首次使用先构建 patched BiPS 工作树：

```bash
cd VAPO_program
./tools/setup_bips_counterfactual_rlvr.sh
pip install -e vendor/BiPS
```

从独立的 Hugging Face 数据仓库下载本轨道数据并核验：

```bash
HF_ENDPOINT=https://huggingface.co hf download amayranitljjnj/bips-ecd-3b-filtered-v3 \
  --repo-type dataset \
  --local-dir ../training_data/BiPS_ECD_3B_filtered_v3
(cd ../training_data/BiPS_ECD_3B_filtered_v3 && \
  sha256sum -c PARQUET_SHA256SUMS)
```

完整两阶段执行：

```bash
CUDA_IDS=0,1,2,3,4,5,6,7 \
N_GPU=8 \
./tools/run_bips_reproduction_7b_pipeline.sh
```

逐阶段执行：

```bash
./run_bips_reproduction_7b_stage1.sh
RUN_ROOT=../training_outputs/bips_reproduction_7b \
  ./tools/merge_final_checkpoint.sh stage1

./run_bips_reproduction_7b_stage2.sh
RUN_ROOT=../training_outputs/bips_reproduction_7b \
  ./tools/merge_final_checkpoint.sh stage2
```

Stage 2 脚本默认读取：

```text
../training_outputs/bips_reproduction_7b/stage1/merged_hf
```

如果该目录不存在，Stage 2 会直接退出，不会错误地使用 base 模型或
Counterfactual RLVR 的 Stage-1 checkpoint。

## 5. checkpoint 与输出隔离

BiPS 轨道输出固定为：

```text
../training_outputs/bips_reproduction_7b/stage1/
├── checkpoints/global_step_N/actor/
├── merged_hf/
└── FINAL_CHECKPOINT.txt

../training_outputs/bips_reproduction_7b/stage2/
├── checkpoints/global_step_M/actor/
├── merged_hf/
└── FINAL_CHECKPOINT.txt
```

每 50 steps 保存一次，最多保留三个 checkpoint；最后一步强制保存。最终 FSDP
checkpoint 保存 model、optimizer 和 extra state，两个阶段都另外生成 merged HF
权重。`resume_mode=auto` 只在本轨道自己的阶段目录中恢复。

新的 Counterfactual RLVR 输出位于独立的：

```text
../training_outputs/counterfactual_rlvr_7b/
```

禁止在两条轨道之间复用 `RUN_ROOT`。

## 6. 对比实验记录

最终对比至少保留：

- base model、两套数据各自的 manifest/hash 和代码 Git commit；
- BiPS Stage-1/Stage-2 最终 FSDP checkpoint 与 merged 权重；
- Counterfactual RLVR Stage-1/Stage-2 最终 FSDP checkpoint 与 merged 权重；
- 两条轨道的 TensorBoard、完整命令和最终验证指标；
- Counterfactual RLVR 的两份 calibration JSON。

推荐先完成 BiPS baseline，再执行 Counterfactual RLVR。两条 7B 全参数训练不能并行
占用同一组 GPU，也不能在 host 可用内存或磁盘低于脚本阈值时强制启动。

发布本轨道数据时使用：

```bash
python tools/publish_hf_training_dataset.py \
  amayranitljjnj/bips-ecd-3b-filtered-v3 \
  --folder ../training_data/BiPS_ECD_3B_filtered_v3 \
  --public
```
