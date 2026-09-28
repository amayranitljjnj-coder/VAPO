# Pilot 后评测与独立 Stage-2 混合训练计划

**状态：实验计划，待执行，尚未得到结果。** 本文不报告任何已完成的八基准评测、
IDK 诊断或混合训练结果，也不预设混合训练一定有效。当前 Stage-2 验证准确率
`0.979` 不作为方法有效的证据；它不能代替独立基准和成对视图上的检验。
暂不设计单独的数据集质量消融。

本计划针对 Counterfactual RLVR pilot，reward 定义沿用
[Counterfactual RLVR 训练说明](COUNTERFACTUAL_RLVR_7B_TRAINING.md)。
不改变 [BiPS 复刻实验](BIPS_7B_REPRODUCTION.md) 的方法或数据。

## 1. 固定模型与执行顺序

先登记四个评测对象的精确路径、checkpoint step、权重哈希和来源：

- **base**：原始 `Qwen2.5-VL-7B-Instruct`。
- **pilot Stage-1**：本次 Counterfactual RLVR pilot 的 Stage-1 checkpoint。
- **pilot Stage-2**：同一 pilot 从上述 Stage-1 初始化得到的 Stage-2 checkpoint。
- **BiPS 7B**：本仓库独立 BiPS 7B reproduction 轨道的最终 Stage-2 模型，
  保留其 Stage-1 → Stage-2 的 checkpoint lineage 与数据版本记录。

不将 pilot 名称自动映射为其他完整训练运行的 checkpoint，不跨轨道使用权重。
完成上述模型的八基准评测及冻结成对测试集诊断后，再从 **pilot Stage-1 checkpoint**
启动独立的 Stage-2 混合训练；不能从 pilot Stage-2 或 BiPS checkpoint 启动。
本文仅记录安排，不启动训练或生成派生数据。

## 2. 八基准评测

**基准与评测规则应尽可能对应 BiPS 的评测基准和协议。** 执行前核对 BiPS 论文、
官方评测代码及配置，登记来源版本，逐项确认下列八基准的任务版本、split、
题目范围、prompt、图像预处理、解码参数、随机种子、答案提取和判分规则。
对可取得的 BiPS 设置优先沿用；无法对应或必须调整的部分，明确记录差异及原因。

若 BiPS 未提供某项基准或某项协议细节，由我们在评测前自行决断并冻结配置，
记录选择依据。不能根据某个 checkpoint 的结果事后调整规则。
协议可按基准适配，但同一基准的所有 checkpoint 必须使用相同数据、题目列表及
评测规则，包括 base、pilot Stage-1、pilot Stage-2、BiPS 7B 和未来混合训练模型。
固定失败、空输出和无法解析答案的计分方式，并计入总题数。

**无论最终采用哪些基准和规则，都必须在同一协议下实际评测 BiPS 7B 模型，
与其他 checkpoint 的结果一起对比。** 不能用 BiPS 论文中的数字代替本次统一评测；
若基准或规则后续发生变更，应对所有参与比较的 checkpoint 重新评测，
不能将不同协议的结果混在同一对比中。

下表逐项填写准确率（%），模型间差值以百分点（pp）报告；
“待评测”是占位标记，不是零分或测量结果。

| 基准 | base | pilot Stage-1 | pilot Stage-2 | BiPS 7B | Stage-1 − base | Stage-2 − base | Stage-2 − Stage-1 |
|---|---|---|---|---|---|---|---|
| CharXiv | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| ChartQAPro | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| ChartMuseum | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| EvoChart | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| MathVista | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| MathVision | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| MathVerse-VO | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| MMStar | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |

最终报告还须逐基准列出 `BiPS 7B − base`、`pilot Stage-1 − BiPS 7B`、
`pilot Stage-2 − BiPS 7B` 的差值；若混合训练模型进入八基准复评，
将其准确率及相对 base、pilot Stage-1、pilot Stage-2、BiPS 7B 的差值一并列出。
BiPS 与 Counterfactual RLVR 使用不同方法和不同训练数据，因此共同对比应表述为
**两个配置实验的比较**，不能作为方法单变量的受控消融。

保留所有逐题预测，至少记录 benchmark/split、题目 ID、模型标识、GT、原始输出、
提取答案、是否正确及运行错误。归档评测配置、样本清单、模型哈希和代码 commit，
使差值能按同一题目逐项复核。

## 3. IDK 退化诊断

选取并冻结 **100–200 道未参与训练的题目**。每题构造一对视图：

- **可回答视图**：原图保留充分证据，GT 为具体答案。
- **移除视图**：移除回答该题所必需的关键证据，GT 为 IDK。

同一对视图保持题目与选项一致，两者均包含 IDK 选项。人工核验原图确实可回答，
移除图无法由其他残留证据推得答案，并记录编辑区域、GT 和核验依据。
冻结后不按任何模型表现替换题目或改变 GT；测试题及两个视图均不得用于训练。

检查候选样本与两条现有轨道的训练图表无重合：核对来源图表 ID、图像文件哈希，
并检查重编码、裁剪、编辑版本等近重复图表。将同源图表视为重合，不能仅凭
题目文本不同认定独立；保存去重审计记录和最终样本 manifest/hash。
混合训练派生集也须通过同样的隔离检查。

base、pilot Stage-1、pilot Stage-2 和 BiPS 7B 在同一冻结集合及同一协议上均评测。
BiPS 未提供的成对视图诊断规则也应事先冻结，并统一用于后续混合训练 checkpoint。
设冻结题目数为 `N`，报告：

| 指标 | 定义（分母均为 N） |
|---|---|
| 可回答视图准确率 | 可回答视图预测具体 GT 的题数 / N |
| 可回答视图误选 IDK 率 | 可回答视图预测 IDK 的题数 / N |
| 移除视图 IDK 正确率 | 移除视图预测 IDK 的题数 / N |
| 成对正确率 | 同一题可回答视图答对具体 GT，且移除视图答对 IDK 的题数 / N |

保留按 `pair_id` 关联的两视图逐题预测、原始输出与判分。报告每个模型的四项指标、
模型间差值及计数；不能用两个视图的平均准确率代替成对正确率。

## 4. 独立的 Stage-2 混合训练

完成上述评测和诊断后，建立独立实验，从 **pilot Stage-1 checkpoint** 初始化。
在每个训练批次按约 **1:1** 混合以下两类样本，记录实际数量与比例：

| 样本来源 | rollout 输入 | GT | 选项 | reward |
|---|---|---|---|---|
| Stage-1 训练样本 | 证据保留视图 `images_pres` | 具体答案 | 包含 IDK | 原 Stage-1 的 `R1` |
| Stage-2 训练样本 | 证据移除视图 `images_abl` | IDK | 包含 IDK | 原 Stage-2 的 `R2` |

每条样本按自己的 GT 做 accuracy 判分，不能将混合批次统一判为 IDK。
按样本类型分别计算原有 reward，不将两种公式平均或统一替换为 `R2`：

```text
R1 = Racc * (1 - 0.3 * (D/tau1)^2 / (1 + (D/tau1)^2))
R2 = Racc * (1 + 0.3 * (1 - exp(-alpha2*D)))
```

`D` 沿用原方法的 detached reverse-k3 sequence mean，分别比较当前保留/移除视图
与原图策略；错误答案 reward 为零，不加入格式分。复用并归档 pilot 原 `R1`、
`R2` 对应的 `tau1`、`alpha2` 校准记录，保持其定义和参数。
BiPS actor consistency/separation loss 继续关闭。

先进行短程训练，在启动前登记步数预算、检查点间隔及判断阈值。
用冻结的成对测试集与 pilot Stage-1、pilot Stage-2 结果比较，检查可回答视图的
误拒答是否下降，同时缺证据视图的 IDK 正确率是否保留，并联合检查可回答准确率和
成对正确率。报告实际差值及不确定性，不能只凭单个指标决定有效。
依据事先登记的标准决定是否延长训练、复跑八基准；若不满足条件，记录结果并停止
延长。反复用于这一决策的冻结集合应在报告中注明其模型选择用途。

## 5. 数据、脚本与输出隔离

新实验仅从原训练 split 只读派生独立数据，不改写、重新过滤或重新划分已发布数据。
仅在新派生目录中记录样本类型、GT、来源行 ID 和混合采样配置。
建议使用以下独立路径（待创建）：

```text
../training_data/Counterfactual_RLVR_Pilot_Stage2_Mix_v1/
../evaluation_data/Post_Pilot_IDK_Pairs_v1/
../training_outputs/post_pilot_evaluation/<run_id>/
../training_outputs/counterfactual_rlvr_pilot_stage2_mix/<run_id>/
```

不覆盖已发布的 `BiPS_ECD_3B_filtered_v3`、`Counterfactual_RLVR_ECD_7B_clean_v1`，
不修改原训练脚本，不覆盖或删除任何已有 checkpoint。未来实现混合训练时使用
独立入口和配置；不得直接修改原 Stage-2 launcher 的行为。
派生数据需保留来源、计数及哈希，训练前运行 `python3 tools/verify_training_package.py`
核验原始八个 Parquet，并单独核验派生数据和冻结测试集。

新训练沿用仓库的资源检查要求，不降低 guard；独立保存最终可恢复 FSDP
checkpoint（`model`、`optimizer`、`extra`）、`merged_hf`、`FINAL_CHECKPOINT.txt`、
日志、TensorBoard、校准记录和完整配置。运行目录不得指向现有 pilot 输出，
恢复训练也只能读取新实验自身的状态。
