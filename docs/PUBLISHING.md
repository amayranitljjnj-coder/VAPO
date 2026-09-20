# GitHub 与 Hugging Face 发布流程

## 1. 发布内容与流量

| 目标 | 内容 | 本地规模 | 发布位置 |
|---|---|---:|---|
| GitHub repository | 训练代码、补丁、文档、校验清单 | 小于 1 MiB | 普通 Git 历史 |
| GitHub Release（可选） | 原始 Qwen2.5-VL-7B-Instruct | 约 16 GiB | 分成小于 2 GiB 的 Release assets |
| Hugging Face dataset 1 | Counterfactual RLVR 7B-filtered IDK 数据 | 4.260 GB | 独立 dataset repo |
| Hugging Face dataset 2 | BiPS 3B-filtered v3 数据 | 5.316 GB | 独立 dataset repo |

上传两套训练数据约消耗 9.58 GB 上行流量；如果同时上传模型 Release，总上行流量约
25–26 GB，另有少量协议开销。模型也可以让训练方直接从官方
`Qwen/Qwen2.5-VL-7B-Instruct` 下载，从而省去本服务器约 16 GiB 的模型上传流量。

## 2. 发布前校验

```bash
cd /mnt/sdc_data/huaming/VAPO/VAPO_program
python3 tools/verify_training_package.py
```

该命令必须显示八个 Parquet 的计数与 SHA-256 全部有效。

## 3. 登录 GitHub

服务器当前使用 GitHub CLI。由账号持有人在终端执行交互式登录：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth status
```

访问令牌或密码只应输入 CLI 登录提示，不写入项目文件。

## 4. 发布 GitHub 代码仓库

目标仓库必须尚不存在，或者已经存在但没有任何 branch。下面示例创建公开仓库；将
`public` 改为 `private` 可创建私有仓库。

```bash
cd /mnt/sdc_data/huaming/VAPO/VAPO_program
./tools/publish_github_training_repo.sh GITHUB_USER/REPOSITORY public
```

脚本会校验两套数据、设置当前仓库的 GitHub noreply identity、创建首个 commit、创建
remote 并推送 `main`。训练数据、模型和 checkpoint 不会进入普通 Git 历史。

## 5. 发布或省略 7B 模型 Release

如需把当前原始模型随项目发布：

```bash
cd /mnt/sdc_data/huaming/VAPO/VAPO_program
./tools/publish_qwen25vl_7b_github_release.sh GITHUB_USER/REPOSITORY
```

脚本先验证模型 SHA-256，再在
`../training_outputs/publish_tmp` 临时生成小于 2 GiB 的分卷并上传。成功后临时分卷自动
删除，原模型保持在 `../models/Qwen2.5-VL-7B-Instruct`。

如果不发布模型 Release，训练方可直接执行：

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir ../models/Qwen2.5-VL-7B-Instruct
```

## 6. 登录 Hugging Face

`hf` CLI 安装在本机 `bips` 环境中：

```bash
export HF_ENDPOINT=https://huggingface.co
/home/huaming/miniconda3/envs/bips/bin/hf auth login
/home/huaming/miniconda3/envs/bips/bin/hf auth whoami
```

本服务器的 `~/.bashrc` 默认设置了 `HF_ENDPOINT=https://hf-mirror.com`。镜像跨域重定向
会移除认证头，因此登录和上传必须显式切换到官方 endpoint。Token 需要具有创建和写入
仓库的权限。数据发布脚本内部也固定使用官方 endpoint。

## 7. 发布两个数据仓库

以下命令创建公开 dataset repo。删除 `--public` 时脚本默认创建私有仓库。

```bash
cd /mnt/sdc_data/huaming/VAPO/VAPO_program

/home/huaming/miniconda3/envs/bips/bin/python \
  tools/publish_hf_training_dataset.py \
  HF_USER/counterfactual-rlvr-ecd-7b-clean-v1 \
  --folder ../training_data/Counterfactual_RLVR_ECD_7B_clean_v1 \
  --public

/home/huaming/miniconda3/envs/bips/bin/python \
  tools/publish_hf_training_dataset.py \
  HF_USER/bips-ecd-3b-filtered-v3 \
  --folder ../training_data/BiPS_ECD_3B_filtered_v3 \
  --public
```

## 8. 训练方的下载命令

```bash
git clone https://github.com/GITHUB_USER/REPOSITORY.git VAPO_program
cd VAPO_program

./tools/download_qwen25vl_7b_github_release.sh GITHUB_USER/REPOSITORY

HF_ENDPOINT=https://huggingface.co \
/home/huaming/miniconda3/envs/bips/bin/hf download \
  HF_USER/counterfactual-rlvr-ecd-7b-clean-v1 \
  --repo-type dataset \
  --local-dir ../training_data/Counterfactual_RLVR_ECD_7B_clean_v1

HF_ENDPOINT=https://huggingface.co \
/home/huaming/miniconda3/envs/bips/bin/hf download \
  HF_USER/bips-ecd-3b-filtered-v3 \
  --repo-type dataset \
  --local-dir ../training_data/BiPS_ECD_3B_filtered_v3

python3 tools/verify_training_package.py
```

如果模型不发布到 GitHub Release，将模型下载命令替换成第 5 节的官方 Hugging Face
下载命令。
