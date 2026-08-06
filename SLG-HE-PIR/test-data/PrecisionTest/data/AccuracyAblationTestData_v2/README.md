# AccuracyAblationTestData — 精度对照实验数据归档

> **目的**：把 Baseline 与 SLG-HE-PIR 的 5 epoch 评估数据按用途分门别类归档，供
> `AccuracyAblationTest/` 子包跑 **6 个**量化变体（Q0/Q0'/Q1/Q2/Q2'/Q3）时作为对比基线。
>
> **⚠️ 重要更新（2026-07-31 23:00）**：经博士生级别审计后，计划文件已重写。新增两个变体：
> - **Q0'**（2-target, 无量化）——用于隔离 LoRA 参数量贡献
> - **Q2'**（Q1 + 全 token g_H 量化，无 gold-only 协议约束）——用于隔离协议约束的精度损失
>
> 原 4 个变体也做了重大修正：
> - Q2/Q3 改为 **gold-token-only 全 token 反向**（不是 argmax-only）
> - Q3 改为 **g_H bf16 round-to-nearest 量化**（不是 PRG share 噪声——PRG 是零和确定性协议）
> - Q1 hook 位置改为 **M 端 decoder.layer.15 输出**（不是 embed_tokens.forward）
> - 命名修正：弃用"BFV 加密税"（BFV 加法 noise-free），改为"BFV fixed-point 量化税 + g_H bf16 转换税"
>
> **创建时间**：2026-07-31
> **任务**：Task A (GenRel 7-class Classification on BioTriplex)
> **关联文档**：`/root/autodl-tmp/.cursor/plans/accuracyablationtest_量化对照实验_cda94492.plan.md`

---

## 目录结构

```
AccuracyAblationTestData/
├── README.md                          ← 本文件（数据索引）
├── baseline/                          ← 明文 Baseline (2-target LoRA) 归档
│   ├── epoch_metrics.jsonl            ← 5 epoch 顶层指标 (与 SLG 同 schema)
│   ├── per_epoch/                     ← 5 个 per-epoch 详细评估 JSON
│   │   ├── epoch_000_evaluate_metrics.json
│   │   ├── ... (共 5 个)
│   │   └── epoch_004_evaluate_metrics.json
│   ├── raw_inference/                 ← 5 个 7-class logits 原始数据 (供口径统一化重评估)
│   │   ├── infer_outputs_epoch_000_20260731_182140.json
│   │   └── ... (共 5 个，每个 ~105KB)
│   ├── logs/                          ← 训练日志 (~958KB)
│   │   └── train_20260731_182140.log
│   ├── adapter_config.json            ← Baseline LoRA 配置（实测 rank=8, alpha=16, target=[q,v]）
│   ├── SOURCE_SUMMARY.md              ← 原始 SUMMARY.md 副本
│   ├── SOURCE_README.md               ← 原始 README.md 副本
│   └── KEY_EVENTS.log                 ← 原始训练关键事件日志
├── slg/                               ← SLG-HE-PIR (3-party HE-PIR, 7-target LoRA) 归档
│   ├── epoch_metrics.jsonl            ← 5 epoch 顶层指标
│   ├── per_epoch/                     ← 5 个 per-epoch 详细评估 JSON
│   ├── logs/evaluate.log              ← SLG 评估日志 (6KB)
│   ├── checkpoints/                   ← SLG ckpt 软链接（不要复制 ~320MB）
│   │   ├── checkpoint_epoch_000.pt → ../SLG-test-data/cls-SLG-test-data/_SAVE_20260727_0706/checkpoint_epoch_000.pt
│   │   └── ... (共 5+2 个：epoch_000..004, best, last)
│   └── legacy_logs/                   ← SLG 原始训练阶段日志
│       ├── SLG_KEY_EVENTS.log
│       ├── SLG_SUMMARY.md
│       ├── SLG_LOG_INFO.txt
│       └── 性能测试重大问题ROOT_CAUSE_PARSE_FAILURES.md
├── derived/                           ← 对比分析输出（生成的）
│   ├── best_epoch_comparison.json     ← Best epoch 对比（Baseline 1 vs SLG 4）
│   ├── avg_comparison.json            ← 5 epoch 平均对比
│   ├── per_class_comparison.json      ← per-class P/R/F1 对比
│   ├── precision_gradient.md          ← 精度梯度对比报告
│   ├── eval_caliber_diff.md           ← 评估口径差异说明
│   └── generate_comparison.py         ← 生成脚本（可重跑）
└── quantization_params/               ← SLG BFV 参数（供 Q0/Q1/Q2/Q3 配置使用）
    ├── slg_bfv_params.json            ← 提取出的参数 JSON
    ├── slg_bfv_params.yaml            ← 同 YAML 版本
    └── extract_slg_bfv_params.py      ← 提取脚本（可重跑）
```

---

## 关键发现速读

| 指标 | Baseline best (epoch 1) | SLG best (epoch 4) | Δ (pp) |
|------|------------------------:|-------------------:|-------:|
| **micro_f1** | 0.3662 | 0.2709 | **−9.5** |
| **macro_f1** | 0.2633 | 0.1515 | **−11.2** |
| **macro_auc** | 0.7771 | 0.6815 | **−9.6** |
| weighted_f1 | 0.3806 | 0.2759 | −10.5 |
| parse_failures | 0 | 0 | = |

**总体结论**（与 `test-data/CLS_PRECISION_COMPARISON_REPORT.md` 一致）：
Baseline CLS 在 BioTriplex 7-class 分类任务上的精度优于 SLG-HE-PIR 异构 PIR 三方微调约 9-11 个百分点。

---

## ⚠️ 重要审计发现（影响"加密税"量化）

### 1. 数据划分差异（10 个 sample）

| 维度 | Baseline | SLG |
|------|----------|-----|
| val_samples | **213** | **203** |
| train_steps | **596** | **734** |

**根因**：两个 dataset class 实现不同——
- Baseline: `baseline/llama-rec/src/llama_recipes/datasets/biotriplex_qakshot_dataset.py`（含 `train_sample_pct` 子采样）
- SLG: `src/data/biotriplex_dataset.py::BiotriplexClassificationDataset`

两者都读 `test_para.txt`，但 doc_key 解析/过滤路径不同，导致 Baseline 比 SLG 多 10 个 val sample 和少 138 个 train_steps。

**对 Q0/Q1/Q2/Q3 的影响**：
- **必须使用与 SLG 一致的 dataset class**（`src.data.biotriplex_dataset.BiotriplexClassificationDataset`），才能保证 val=203 + train_steps=734 与 SLG 完全对齐
- 否则对比会出现"数据集差异"和"加密税"叠加的混淆

### 2. 评估口径差异（已写入 `derived/eval_caliber_diff.md`）

| 维度 | Baseline (`evaluate_metrics.py`) | SLG (`biotriplex_metrics.py`) |
|------|---------------------------------|--------------------------------|
| 缺失预测 | 保留 `-1` | 替换为 "relation undefined" |
| macro_roc_auc_ovr | sklearn 标准 ovr | 仅在 y_true 含该类时计算 |
| multilabel F1 | binary 7-vec | 解析 multi-letter 字符串 |

**对 Q0/Q1/Q2/Q3 的影响**：
- 新会话的 `report_generator.py` **必须**用 SLG 的 `compute_classification_metrics` 重新评估 Baseline 的 `raw_inference/infer_outputs_epoch_*.json`，得到统一口径
- 或者在 QUANT_ABLATION_REPORT.md 中**明确标注**指标口径差异

### 3. 唯一变量隔离（target_modules）

Baseline 与 SLG 在训练超参上**完全一致**，**唯一变量**就是 `target_modules`（2 个 vs 7 个）——这正是 Q0 变体要隔离的"LoRA 配置贡献"。

### 4. SLG ckpt 的 BFV 参数位置（已实测）

| 位置 | 内容 |
|------|------|
| `ckpt['config']` | 只有训练参数（batch_size/max_epochs/etc.），**无 BFV 参数** |
| `ckpt['party_checkpoints']['M']['lora_state']` | 全部 LoRA 权重（257 个），可读出 rank/target_modules |
| `ckpt['party_checkpoints']['S']['v_shape']` | V 矩阵 shape = (128256, 4096) |
| `ckpt['party_checkpoints']['S']` 的 `note` 字段 | 额外元数据（不是 BFV 参数） |

BFV 参数（scale=10000, plain_bits=30, poly_degree=4096）**必须**从源码 `src/core/bfv_privselect_v2_adapter.py:11-13` 读，不能从 ckpt 读。

---

## Q0/Q0'/Q1/Q2/Q2'/Q3 实验将如何使用此目录

| 实验变体 | 依赖此目录的什么 |
|----------|------------------|
| **Q0** (无量化, 7-target) | Baseline (2-target) epoch_metrics + raw_inference 用作对照 |
| **Q0'** (无量化, 2-target) | Baseline 完全一致（已存在）—— 隔离 LoRA 参数量贡献 |
| **Q1** (lm_head 量化 + H_M forward 量化) | quantization_params/slg_bfv_params.yaml（scale=10000）|
| **Q2** (Q1 + gold-only 全 token g_H 量化) | 同 Q1 |
| **Q2'** (Q1 + 全 token g_H 量化, 无协议约束) | 同 Q1 —— 隔离 Q2 协议约束 |
| **Q3** (Q2 + g_H bf16 转换量化) | 同 Q1 |
| **所有变体** | derived/eval_caliber_diff.md 提示做口径统一 |

---

## 数据生成方式

```bash
cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/AccuracyAblationTestData

# 提取 SLG BFV 参数
python3 quantization_params/extract_slg_bfv_params.py

# 生成对比分析
python3 derived/generate_comparison.py
```

两个脚本都是**幂等**的，可以随时重跑。

---

## 已知 bug / 待办

| 项 | 说明 | 处理方式 |
|----|------|----------|
| `val_samples` 213 vs 203 | Baseline 与 SLG dataset class 不同 | Q0/Q1/Q2/Q3 必须统一用 SLG 的 dataset class |
| 评估口径差异 | sklearn vs 自定义 | `report_generator.py` 重评估 Baseline `raw_inference/` |
| SLG `train_steps=734` 写入 `epoch_metrics.jsonl` | 是 evaluate_slg_cls.py:306 硬编码 | 真正的训练 step 数应该从 train log 取（待查）|
| Q3 变体的 PRG share 噪声常数 | 尚未从源码完全审计 | 待新会话在 Q3 实现时从 `bfv_privselect_v2_adapter.py::generate_mask_ints` 提取 |

---

## 关联文档

- 上游计划：`/root/autodl-tmp/.cursor/plans/accuracyablationtest_量化对照实验_cda94492.plan.md`
- 上游交接：`/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/HANDOFF.md`
- 已有对比报告：`/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/CLS_PRECISION_COMPARISON_REPORT.md`
- Baseline 训练脚本：`baseline/classification_genrel/scripts/run_finetune_with_epochs.sh`
- Baseline 评估：`baseline/classification_genrel/scripts/{infer_and_save,evaluate_metrics}.py`
- SLG 训练：`src/scripts/biotriplex_finetune.py`
- SLG 评估：`src/scripts/evaluate_slg_cls.py`
- SLG 指标：`src/training/biotriplex_metrics.py::compute_classification_metrics`

---

**END OF README** — 2026-07-31 22:36 UTC+8