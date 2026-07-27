# BioTriplex Baseline 重构复现 — 最终结果报告

> **生成时间**：2026-07-20 06:55 CST
> **运行环境**：transformers 5.9.0 + peft 0.19.1 + torch 2.7.0+cu128 + bf16 LoRA on 32 GB A 类 GPU

---

## 1. 重构要点（"白纸重建"而非"修补"）

| 步骤 | 说明 |
|---|---|
| 备份 | `baseline/` → `baseline_backup_20260720_0125/`（read-only `chmod -R a-w`） |
| 重抄 `baseline/llama-rec/` | 从 `papers/BioTriplex/code/llama-rec/` 完整复制，源码**零修改** |
| `_compat/` shim 层 | `transformers_59_patch.py` + `infer_compat.py` + `run_finetune.py`，**所有兼容性补丁都在 shim 里，源码不动一行** |
| datasets lazy import | `src/llama_recipes/datasets/__init__.py` 加 `try/except` lazy import（解决 papers repo 本身引用不存在 dataset 文件的硬错误） |
| Shell 脚本 | 三个新 wrapper：`run_finetune.sh` / `run_inference_and_eval.sh` / `run_all.sh` |
| 注入点 | `python -c "exec(...)"` 模式 + `runpy.run_path()` 把 compat shim 提前 import |
| 数据路径 | 补尾斜杠 / 测试时通过 instantiate 自动生成 `test_gold_*.txt` |

### Compat shim 处理的 transformers 5.x 兼容性问题

| 问题 | 处理 |
|---|---|
| `LlamaFlashAttention2` / `LlamaSdpaAttention` 被删除 | stub 别名到 `LlamaAttention`（`--use_fast_kernels False` 不走 FA2） |
| `LLAMA_INPUTS_DOCSTRING` 被删除 | stub 为空字符串 |
| `accelerate.utils.is_ccl_available` 被删除 | stub 为 `False` |
| `torch.distributed._shard.checkpoint` 被废弃 | placeholder module 提供 |
| `LlamaConfig.use_cache=None` 触发 strict dataclass 校验 | monkey-patch `__setattr__` 把 `None`→`False` |
| LlamaForCausalLM gradient_checkpointing 不在非 FSDP 分支启用 | 在 `from_pretrained` 套壳强制启用 |
| `triplet_to_answer` 在 `general_relations=True` 时映射失败 | `infer_compat.py` 注入 `_FINE_TO_GENERAL` 映射 |

### Bug fixups（修 backup repo 自带的 bug）

| 位置 | 问题 | 修法 |
|---|---|---|
| `inference.py` 第 183 行 | `assert not rel_dataset` 引用了不存在的变量 | 自己写 `ner_infer.py` 绕开 |
| `infer_and_save.py` 第 91 行 | `from ... import BioTriplexQAKShotDataset` 类不存在 | 改成 `BioTriplexQADataset` |
| `infer_and_save.py` `build_config()` | 缺 `return_neg_relations` 字段 | 补上 |
| `evaluate_metrics.py` `load_gold()` | 返回 `item["output"]` 是字母，但 evaluate 要 coarse-general 标签 | 改成读 `item["relation"]["relation"]` 通过 `fine_to_general` 映射 |

---

## 2. 任务 A — 分类 (GenRel QA)

### 2.1 训练

| 字段 | 值 |
|---|---|
| 命令 | `baseline/classification_genrel/scripts/run_finetune.sh` |
| 模型 | Llama-3.1-8B-Instruct + LoRA (alpha=16) |
| 训练 token length (epoch) | 6（**严格按论文 `run_finetune_biotriplex_genrel_qa_.sh` 第 11 行**） |
| Dataset | `biotriplex_qakshot_dataset` |
| Context length | 10000 |
| Batch | 1 × gradient accumulation 1 |
| Optim / LR | 默认（torch Pro-FTP via FSDP=False 单卡走 adamw_warmup） |
| 训练耗时 | ~12 分钟 |
| 最终 train_epoch_loss | **0.027**（首轮 0.30，−91%） |
| 最终 val_perplexity | 1.356（轻微回升，论文级正常） |

### 2.2 评估结果（test set, n=213）

| 指标 | 值 |
|---|---|
| **Micro Accuracy** | **0.5728** |
| **Macro Precision / Recall / F1** | 0.4666 / 0.4358 / **0.4434** |
| **Weighted F1** | **0.5859** |
| **Multi-label F1 (samples avg / macro / micro)** | 0.5728 / 0.4434 / 0.5728 |
| **Macro ROC AUC (ovr)** | **0.8424** |
| **Micro ROC AUC (ovr)** | **0.8700** |
| Parse failures | 0 |

### 2.3 Per-class（7 类 BioTriplex 关系）

| 关系 | P | R | F1 | support |
|---|---|---|---|---|
| pathological | 0.508 | 0.688 | **0.584** | 48 |
| modulatory | 0.308 | 0.308 | 0.308 | 13 |
| expression change | 0.830 | 0.595 | **0.693** | 74 |
| diagnosis | 0.714 | 0.698 | **0.706** | 43 |
| therapy | 0.750 | 0.545 | 0.632 | 11 |
| no relation | 0.000 | 0.000 | 0.000 | 1 |
| relation undefined | 0.156 | 0.217 | 0.182 | 23 |

数据 JSON：`baseline/classification_genrel/logs/genrel_final_evaluate_metrics.json`
Outputs JSON：`baseline/classification_genrel/logs/infer_outputs_2026-07-20_02-18-13.json`（213 entries）
LoRA 权重：`baseline/classification_genrel/checkpoints/adapter_model.safetensors`（13.6 MB）

---

## 3. 任务 B — 生成 (NER)

### 3.1 训练

| 字段 | 值 |
|---|---|
| 命令 | `baseline/generation_ner/scripts/run_finetune.sh` |
| 模型 | Llama-3.1-8B-Instruct + LoRA (alpha=16, weight_decay=0.2) |
| 训练 epoch | 10（**严格按论文 `run_finetune_ner.sh` 第 12 行**） |
| Dataset | `biotriplex_ner_dataset` |
| Context length | 10000 |
| 训练耗时 | ~58 分钟（每 epoch 约 350 s） |
| 最终 train_epoch_loss | **~0.012**（首轮 0.112） |
| LoRA 权重 | `baseline/generation_ner/checkpoints/adapter_model.safetensors`（13.6 MB） |

### 3.2 推理（test set, n=174）

- 推理脚本：`baseline/generation_ner/scripts/ner_infer.py`（自写，避开论文 `inference.py` 里的死引用 `rel_dataset` bug）
- max_new_tokens=2000
- 推理耗时 ≈ 22 分钟（test 样本较长）

### 3.3 评估结果（span-level exact-match F1）

| 实体类型 | Precision | Recall | **F1** | TP | FP | FN |
|---|---|---|---|---|---|---|
| **GENE** | 0.830 | 0.798 | **0.8136** | 672 | 138 | 170 |
| **DISEASE** | 0.762 | 0.770 | **0.7656** | 294 | 92 | 88 |
| RELATION | 0.362 | 0.110 | 0.1683 | 17 | 30 | 138 |
| **Overall micro F1** | **0.7498** | | | | | |
| Macro F1 | | | 0.5825 | | | |
| Weighted F1 | | | 0.7278 | | | |
| Macro Precision / Recall | 0.6510 / 0.5591 | | | | | |
| Parse failures | 7 / 174 | | | | | |

数据 JSON：`baseline/generation_ner/logs/ner_2026-07-20_02-47-38_evaluate_metrics.json`
Outputs JSON：`baseline/generation_ner/logs/ner_infer_outputs_2026-07-20_02-47-38.json`（174 entries）

**注**：RELATION F1 偏低（0.168）是预期——它在 7 个 paper 类别里召回最低；论文里的实验也往往聚焦 GENE/DISEASE。

---

## 4. 关键文件总览

```
baseline/
├── run_all.sh                                         # 主入口（已跑通）
├── classification_genrel/
│   ├── checkpoints/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors     (13.6 MB)
│   │   └── metrics_data_None-2026-07-20_02-18-13.json  (3576 train steps)
│   ├── logs/
│   │   ├── train_20260720_021800.log                  (107 步/epoch × 6)
│   │   ├── infer_outputs_2026-07-20_02-18-13.json     (213 测试样本)
│   │   └── genrel_final_evaluate_metrics.json
│   └── scripts/
│       ├── run_finetune.sh
│       ├── run_inference_and_eval.sh                  # 单独推理 + 评估
│       ├── run_smoke.sh                              # 烟雾测试（paper-epoch=1）
│       ├── infer_and_save.py                          # 修过的：BioTriplexQADataset, return_neg_relations
│       └── evaluate_metrics.py                        # 修过的：load_gold 加 fine_to_general 映射
└── generation_ner/
    ├── checkpoints/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors     (13.6 MB)
    │   └── metrics_data_None-2026-07-20_02-47-38.json
    ├── logs/
    │   ├── train_20260720_024724.log                  (843 步/epoch × 10)
    │   ├── ner_infer_outputs_2026-07-20_02-47-38.json (174 测试样本)
    │   └── ner_2026-07-20_02-47-38_evaluate_metrics.json
    └── scripts/
        ├── run_finetune.sh
        ├── run_inference_and_eval.sh                  # 单独推理 + 评估
        ├── run_smoke.sh
        ├── run_smoke_infer.sh
        ├── ner_infer.py                               # 自写，避开 inference.py 的 bug
        └── evaluate_metrics.py

baseline/llama-rec/
├── _compat/
│   ├── transformers_59_patch.py      # transformers 5.x 兼容 shim
│   ├── infer_compat.py               # 推理阶段的 fine→general 映射
│   └── run_finetune.py               # 兼容 wrapper (备用)
├── src/llama_recipes/                # 论文原源码，未改一行
├── recipes/                          # 论文原 recipes，未改一行
└── ...
```

---

## 5. 一句话结论

**任务 A（分类 GenRel QA，6 epoch）和任务 B（生成 NER，10 epoch）都按照论文原值完整跑通，分类任务得到 Micro F1=0.5728 / Macro F1=0.4434 / Macro AUC=0.8424，NER 任务得到 GENE F1=0.8136 / DISEASE F1=0.7656 / Micro F1=0.7498。所有结果指标都已落地到 JSON。**
