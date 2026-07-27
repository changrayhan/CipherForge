# BioTriplex Baseline 测试报告

## 目录

1. [测试环境](#1-测试环境)
2. [测试流程](#2-测试流程)
3. [模型与参数配置](#3-模型与参数配置)
4. [数据集信息](#4-数据集信息)
5. [测试结果汇总](#5-测试结果汇总)
6. [数据文件说明](#6-数据文件说明)
7. [复现测试方法](#7-复现测试方法)

---

## 1. 测试环境

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GPU (CUDA 可用) |
| 显存 | 足够加载 Llama-3.1-8B-Instruct 模型 |
| 内存 | 建议 32GB+ |

### 1.2 软件环境

| 项目 | 版本/配置 |
|------|-----------|
| Python | 3.x |
| PyTorch | 支持 CUDA |
| PyOT (BFV) | 同态加密库 |
| transformers | 模型加载 |
| peft | LoRA 微调 |
| tqdm | 进度条 |

### 1.3 路径配置

| 变量 | 路径 |
|------|------|
| 基础模型 | `/root/autodl-tmp/hf_cache/Llama-3-1-8B-I` |
| 数据集 | `/root/autodl-tmp/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/` |
| BFV 缓存 | `/root/autodl-tmp/slg-bfv-cache` |
| 代码根目录 | `/root/autodl-tmp/SLG-HE-PIR` |

---

## 2. 测试流程

### 2.1 任务概述

本测试包含两个 BioTriplex 任务：

| 任务 | 任务类型 | 说明 |
|------|----------|------|
| **Task A** | 分类 (Classification) | 基因-疾病关系分类，7 分类任务 |
| **Task B** | 生成 (Generation) | NER 实体识别，生成 JSON 格式结果 |

### 2.2 执行顺序

```
Task A (分类) → Task B (NER生成)
```

两个任务按顺序执行，Task A 完成后再执行 Task B。

### 2.3 执行脚本

| 脚本 | 功能 |
|------|------|
| `scripts/biotriplex_classification_genrel.sh` | 执行 Task A 分类任务 |
| `scripts/biotriplex_generation_ner.sh` | 执行 Task B NER 生成任务 |
| `scripts/biotriplex_run_all.sh` | 一键执行上述两个任务 |

---

## 3. 模型与参数配置

### 3.1 基础模型

- **模型名称**: Llama-3.1-8B-Instruct
- **模型路径**: `/root/autodl-tmp/hf_cache/Llama-3-1-8B-I`
- **类型**: 因果语言模型 (Causal LM)

### 3.2 LoRA 微调配置

两个任务共享相同的 LoRA 配置：

| 参数 | 值 |
|------|-----|
| LoRA Rank (r) | 8 |
| LoRA Alpha | 16 (或 32) |
| LoRA Dropout | 0.05 |
| Target Modules | `q_proj`, `v_proj` |
| PEFT Type | LORA |

### 3.3 训练超参数对比

| 参数 | Task A (分类) | Task B (NER) |
|------|---------------|--------------|
| Max Epochs | 6 | 10 |
| Batch Size | 1 | 1 |
| Max Seq Length | 10000 | 10000 |
| Learning Rate | 1e-4 | 1e-4 |
| Weight Decay | 0.0 | 0.2 |
| Warmup Steps | 200 | 200 |
| Gradient Clip Norm | 1.0 | 1.0 |
| Chunk Tokens | 3072 | 1536 |
| Use Chunked Pipeline | True | True |
| General Relations | True | - |
| Seed | 42 | 42 |

### 3.4 优化器配置

- **优化器**: AdamW
- **学习率调度**: 线性 warmup 后余弦衰减
- **Gradient Clipping**: 1.0

---

## 4. 数据集信息

### 4.1 数据集来源

- **数据集名称**: BioTriplex
- **预处理版本**: Preprocessed BioTriplex
- **数据路径**: `/root/autodl-tmp/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/`

### 4.2 任务数据规格

| 任务 | 数据划分 | 样本数 |
|------|----------|--------|
| Task A (分类) | Test | 213 |
| Task B (NER) | Test | 174 |

### 4.3 任务标签说明

**Task A - 关系分类 (7 类)**:

| 标签 | 说明 |
|------|------|
| pathological | 病理关系 |
| modulatory | 调节关系 |
| expression change | 表达变化 |
| diagnosis | 诊断关系 |
| therapy | 治疗关系 |
| no relation | 无关系 |
| relation undefined | 关系未定义 |

**Task B - NER 实体类型 (3 类)**:

| 标签 | 说明 |
|------|------|
| GENE | 基因实体 |
| DISEASE | 疾病实体 |
| RELATION | 关系实体 |

---

## 5. 测试结果汇总

### 5.1 Task A - 分类任务结果

**基础指标**:

| 指标 | 值 |
|------|-----|
| Micro Accuracy | 0.5775 |
| Micro F1 | 0.5775 |
| Macro F1 | 0.4094 |
| Weighted F1 | 0.5714 |
| Macro AUC | 0.8722 |
| Micro AUC | 0.8750 |

**分类型指标**:

| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| pathological | 0.5625 | 0.5625 | 0.5625 | 48 |
| modulatory | 0.1818 | 0.4615 | 0.2609 | 13 |
| expression change | 0.8654 | 0.6081 | 0.7143 | 74 |
| diagnosis | 0.6441 | 0.8837 | 0.7451 | 43 |
| therapy | 0.5385 | 0.6364 | 0.5833 | 11 |
| no relation | 0.0 | 0.0 | 0.0 | 1 |
| relation undefined | 0.0 | 0.0 | 0.0 | 23 |

### 5.2 Task B - NER 生成任务结果

**基础指标**:

| 指标 | 值 |
|------|-----|
| Macro F1 | 0.4131 |
| Weighted F1 | 0.5516 |
| Micro Precision | 0.8904 |
| Micro Recall | 0.4236 |
| Micro F1 | 0.5741 |

**分类型指标**:

| 类别 | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|--------|-----|-----|-----|-----|
| GENE | 0.9013 | 0.4774 | 0.6242 | 402 | 44 | 440 |
| DISEASE | 0.8673 | 0.4766 | 0.6151 | 183 | 28 | 201 |
| RELATION | 0.0 | 0.0 | 0.0 | 0 | 0 | 155 |

### 5.3 结果分析

1. **Task A (分类)**:
   - 整体准确率 57.75%，表现一般
   - "diagnosis" 和 "expression change" 类别 F1 较高 (>0.7)
   - "no relation" 和 "relation undefined" 类别完全无法识别

2. **Task B (NER)**:
   - GENE 和 DISEASE 实体识别精确率高 (>86%)，但召回率偏低 (~48%)
   - RELATION 关系完全无法识别 (F1=0)
   - 解析失败率较高：79/174 (约 45%)

---

## 6. 数据文件说明

### 6.1 目录结构

```
test-data/baseline-test-data/
├── cls-base-test-data/     # Task A 分类任务数据
│   ├── checkpoints/        # 检查点目录
│   │   ├── adapter_model.safetensors    # LoRA 适配器权重
│   │   ├── adapter_config.json          # 适配器配置
│   │   ├── README.md                    # 检查点说明
│   │   └── metrics_data_*.json         # 训练指标
│   └── logs/             # 日志目录
│       ├── genrel_*_evaluate_metrics.json  # 评估指标
│       ├── infer_outputs_*.json            # 推理输出
│       └── train_*.log                     # 训练日志
│
└── NER-base-test-data/    # Task B NER生成任务数据
    ├── checkpoints/      # 检查点目录
    │   ├── adapter_model.safetensors    # LoRA 适配器权重
    │   ├── adapter_config.json          # 适配器配置
    │   ├── README.md                    # 检查点说明
    │   └── metrics_data_*.json         # 训练指标
    └── logs/             # 日志目录
        ├── ner_evaluate_metrics.json    # 评估指标
        ├── infer_outputs_*.json         # 推理输出
        └── train_*.log                 # 训练日志
```

### 6.2 关键文件说明

| 文件 | 说明 |
|------|------|
| `adapter_model.safetensors` | LoRA 适配器权重，可用于后续推理部署 |
| `adapter_config.json` | 适配器配置，记录 LoRA 参数 |
| `*_evaluate_metrics.json` | 评估结果指标 (JSON 格式) |
| `infer_outputs_*.json` | 模型推理输出 |
| `train_*.log` | 训练过程日志 |

### 6.3 评估指标文件格式

**分类任务** (`genrel_*_evaluate_metrics.json`):
```json
{
  "task": "classification_genrel_qa",
  "n_samples": 213,
  "micro_accuracy": 0.577,
  "macro_f1": 0.409,
  "per_class": { ... }
}
```

**NER任务** (`ner_evaluate_metrics.json`):
```json
{
  "task": "NER (Span-level Exact-match)",
  "n_common_doc_keys": 174,
  "metrics": {
    "macro_f1": 0.413,
    "weighted_f1": 0.552
  },
  "per_class_metrics": { ... }
}
```

---

## 7. 复现测试方法

### 7.1 环境准备

#### 7.1.1 基础环境

```bash
# 克隆项目
git clone <repo_url>
cd SLG-HE-PIR

# 安装依赖
pip install torch transformers peft pyyaml tqdm

# 确保 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"
```

#### 7.1.2 准备模型和数据

```bash
# 下载 Llama-3.1-8B-Instruct 模型到指定路径
# 模型路径: /root/autodl-tmp/hf_cache/Llama-3-1-8B-I

# 准备 BioTriplex 数据集
# 数据路径: /root/autodl-tmp/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/
```

### 7.2 完整测试流程

#### 7.2.1 一键运行全部任务

```bash
cd /root/autodl-tmp/SLG-HE-PIR

# 执行完整流程 (Task A + Task B)
bash scripts/biotriplex_run_all.sh
```

#### 7.2.2 分步运行

**Step 1: 运行 Task A (分类任务)**

```bash
cd /root/autodl-tmp/SLG-HE-PIR

bash scripts/biotriplex_classification_genrel.sh
```

**Step 2: 运行 Task B (NER生成任务)**

```bash
cd /root/autodl-tmp/SLG-HE-PIR

bash scripts/biotriplex_generation_ner.sh
```

### 7.3 核心参数说明

执行 `biotriplex_finetune.py` 的完整参数列表：

```bash
python src/scripts/biotriplex_finetune.py \
    --task_type [classification|generation] \    # 任务类型
    --stage all \                                  # 训练+评估
    --data_path <PATH> \                           # 数据集路径
    --hf_model <PATH> \                            # 基础模型路径
    --bfv_cache_dir <PATH> \                       # BFV 缓存目录
    --output_dir <PATH> \                         # 输出目录 (检查点)
    --log_dir <PATH> \                            # 日志目录
    --adapter_dir <PATH> \                        # 适配器目录
    --max_epochs <INT> \                          # 最大训练轮数
    --batch_size <INT> \                          # 批大小
    --max_seq_length <INT> \                      # 最大序列长度
    --learning_rate <FLOAT> \                      # 学习率
    --weight_decay <FLOAT> \                      # 权重衰减
    --warmup_steps <INT> \                        # Warmup 步数
    --gradient_clip_norm <FLOAT> \                # 梯度裁剪阈值
    --lora_rank <INT> \                           # LoRA rank
    --lora_alpha <INT> \                          # LoRA alpha
    --lora_dropout <FLOAT> \                     # LoRA dropout
    --use_chunked_pipeline <BOOL> \               # 是否使用分块流水线
    --chunk_tokens <INT> \                        # 分块 token 数
    --seed <INT> \                                # 随机种子
    --log_freq <INT> \                            # 日志打印频率
    --save_freq <INT> \                           # 检查点保存频率
    --do_test_eval                                # 是否在测试集评估
```

### 7.4 关键参数对照表

| 参数 | Task A 值 | Task B 值 | 说明 |
|------|-----------|-----------|------|
| `--task_type` | `classification` | `generation` | 任务类型 |
| `--max_epochs` | 6 | 10 | 训练轮数 |
| `--weight_decay` | 0.0 | 0.2 | 权重衰减 |
| `--chunk_tokens` | 3072 | 1536 | 分块大小 |
| `--general_relations` | `True` | - | 是否使用通用关系 |

### 7.5 输出日志查看

```bash
# 查看 Task A 训练日志
cat baseline/classification_genrel/logs/train_*.log

# 查看 Task A 评估指标
cat baseline/classification_genrel/logs/genrel_*_evaluate_metrics.json

# 查看 Task B 训练日志
cat baseline/generation_ner/logs/train_*.log

# 查看 Task B 评估指标
cat baseline/generation_ner/logs/ner_evaluate_metrics.json
```

### 7.6 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| OOM (显存不足) | 序列长度过长 | 减小 `--max_seq_length` |
| OOM | Batch size 过大 | 减小 `--batch_size` |
| 模型加载失败 | 模型路径错误 | 检查 `--hf_model` 路径 |
| 数据找不到 | 数据路径错误 | 检查 `--data_path` 路径 |
| 评估结果为空 | 未加 `--do_test_eval` | 添加该参数 |

### 7.7 注意事项

1. **顺序依赖**: Task A 和 Task B 可独立运行，无数据依赖
2. **显存需求**: 建议至少 24GB 显存
3. **运行时间**: Task A 约 1-2 小时，Task B 约 40 分钟
4. **随机种子**: 默认使用 seed=42 确保可复现性

---

## 附录: adapter_config.json 完整内容

两个任务的 adapter_config.json 相同：

```json
{
  "peft_type": "LORA",
  "base_model_name_or_path": "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I",
  "task_type": "CAUSAL_LM",
  "r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "bias": "none",
  "inference_mode": true
}
```

---

*文档生成时间: 2026-07-22*
*测试执行时间: 2026-07-22 08:37 ~ 12:48 (UTC+8)*
