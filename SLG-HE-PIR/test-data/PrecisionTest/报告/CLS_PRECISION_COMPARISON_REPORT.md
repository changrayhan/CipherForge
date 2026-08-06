# Task A (GenRel 7-class CLS) 精度对比报告
## Baseline CLS vs SLG-HE-PIR — BioTriplex Classification 微调精度测试

**报告日期**: 2026-07-31
**任务**: BioTriplex GenRel QA 7-class classification (a-g)
**数据集**: `/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/`
**测试样本数**: 213 (Baseline) / 203 (SLG-HE-PIR, 数据划分差异)
**评估方式**: 7-class projection on last-token logits (与 Baseline `infer_and_save.py` 对齐)
**测试集 (split)**: test

---

## 1. 实验设置对齐

| 参数 | Baseline CLS | SLG-HE-PIR |
|------|--------------|-------------|
| Base model | Llama-3-1-8B-I | Llama-3-1-8B-I |
| LoRA rank | 8 | 8 |
| LoRA alpha | 16 | 16 |
| LoRA dropout | 0.05 | 0.05 |
| Target modules | q_proj,k_proj,v_proj,o_proj | q,k,v,o,gate,up,down |
| Learning rate | 1e-4 | 1e-4 |
| Weight decay | 0.0 | 0.0 |
| Batch size | 1 | 1 |
| Epochs | 5 | 5 (评估 5 个独立 checkpoint) |
| Context length | 10000 | 10000 (eval=4096) |
| Dataset | biotriplex_classification | biotriplex_classification |
| 7-class projection | ✅ (baseline infer_and_save.py) | ✅ (修复后 PartyS._classify_from_logits) |

**重要**: SLG-HE-PIR 的训练 loss 不可直接对比 — 它是 `g_H` 范数代理 (~28160)，不是 CE loss。**唯一有效对比指标是 val 评估指标**。

---

## 2. 5 Epoch 评估指标对比

### 2.1 Baseline CLS (Plaintext LoRA, 5 epochs)

| Epoch | micro_f1 | macro_f1 | macro_auc | weighted_f1 | parse_fail |
|-------|----------|----------|-----------|-------------|------------|
| 0 | 0.2300 | 0.1640 | **0.7989** | 0.2104 | 0/213 |
| 1 | **0.3662** | **0.2633** | 0.7771 | **0.3213** | 0/213 |
| 2 | 0.3052 | 0.2343 | 0.7998 | 0.2718 | 0/213 |
| 3 | 0.3333 | 0.2368 | 0.7703 | 0.2965 | 0/213 |
| 4 | 0.2676 | 0.1869 | 0.7786 | 0.2143 | 0/213 |

**Baseline 最佳 epoch**: **Epoch 1** (micro_f1=0.3662, macro_f1=0.2633)

### 2.2 SLG-HE-PIR (3-Party HE-PIR LoRA, 5 epochs)

| Epoch | micro_f1 | macro_f1 | macro_auc | weighted_f1 | parse_fail |
|-------|----------|----------|-----------|-------------|------------|
| 0 | 0.2611 | 0.1444 | 0.6796 | 0.2191 | **0/203** |
| 1 | **0.2709** | 0.1492 | 0.6790 | 0.2268 | 0/203 |
| 2 | 0.2611 | 0.1438 | 0.6764 | 0.2191 | 0/203 |
| 3 | 0.2611 | 0.1453 | 0.6784 | 0.2191 | 0/203 |
| 4 | 0.2709 | **0.1515** | **0.6815** | **0.2268** | 0/203 |

**SLG-HE-PIR 最佳 epoch**: **Epoch 4** (micro_f1=0.2709, macro_f1=0.1515, macro_auc=0.6815)

### 2.3 5 Epoch 平均指标对比

| 指标 | Baseline | SLG-HE-PIR | Δ (Baseline - SLG) |
|------|----------|------------|---------------------|
| avg micro_f1 | 0.3005 | 0.2650 | **+0.0355** |
| avg macro_f1 | 0.2171 | 0.1468 | **+0.0703** |
| avg macro_auc | 0.7849 | 0.6790 | **+0.1059** |
| avg weighted_f1 | 0.2629 | 0.2222 | **+0.0407** |

### 2.4 最佳 Epoch 对比

| 指标 | Baseline (Epoch 1) | SLG-HE-PIR (Epoch 4) | Δ (Baseline - SLG) |
|------|--------------------|----------------------|---------------------|
| **micro_f1** | **0.3662** | 0.2709 | **+0.0953** |
| **macro_f1** | **0.2633** | 0.1515 | **+0.1118** |
| **macro_auc** | **0.7771** | 0.6815 | **+0.0956** |
| weighted_f1 | 0.3213 | 0.2268 | +0.0945 |
| parse_failures | 0 | 0 | = |

---

## 3. 关键发现

### 3.1 Baseline CLS 优势

1. **micro_f1 +9.5%**: Baseline 在主指标上比 SLG-HE-PIR 高出 9.5 个百分点
2. **macro_f1 +11.2%**: Baseline 在类别均衡指标上优势更大（+11.2pp）
3. **macro_auc +9.6%**: Baseline 的排序质量明显更好
4. **收敛更快**: Baseline 在 Epoch 1 即达到最佳，SLG 在 Epoch 4

### 3.2 SLG-HE-PIR 的弱项

1. **macro_f1 显著低**: 0.15 vs 0.26 — 提示长尾类别（modulatory, therapy）F1 接近 0
2. **macro_auc 接近 0.68**: 说明 logits 排序质量低于 Baseline（0.78+）
3. **训练不稳定**: 5 个 epoch 指标几乎不变（micro_f1 ∈ [0.2611, 0.2709]），收敛饱和

### 3.3 共同点

1. **parse_failures 全部为 0**: 7-class projection 修复对两个方案都正常工作
2. **测试集 0.30/0.27**: 在 GenRel 7-class 任务上精度都在 30% 以下，符合 BioTriplex QA 任务本身的难度

---

## 4. 修复摘要

### 4.1 Baseline CLS 修复 (`scripts/run_finetune_with_epochs.sh`)

**Bug 1**: `PREV_CKPT` 被设置为目录而非 checkpoint 文件
- 位置: 第 255 行 `PREV_CKPT="${OUTPUT_DIR}"` → 应为 `"${CKPT_FILE}"`
- 影响: 第二个 epoch 起 `--from_peft_checkpoint` 传错路径，整个 epoch 循环中断
- 修复: 改为 `"${CKPT_FILE}"`（后调整为 `"${OUTPUT_DIR}"` 因为 PEFT 需要目录路径含 `adapter_config.json`）

**Bug 2**: `KEY_EVENTS.log` echo 命令 bash 语法错误
- 位置: 第 252 行 `$(5 + (EPOCH % 2) * 30)` → 缺少 `$((...))` 外层括号
- 影响: `set -uo pipefail` 下脚本在 epoch 循环内意外退出
- 修复: 改为简单 `$(date '+...')` 格式

### 4.2 SLG-HE-PIR 修复 (`src/parties/party_s.py`)

**Bug**: `generate_predictions` 把 LM logits `[B, S, V=128256]` 当成 `[B, num_classes=7]` 取整句 argmax
- 影响: 100% parse failures → 所有 val 指标为 0
- 修复: 实现 `task_type=='classification'` 分支，使用 7-class projection on last non-pad position
- 代码: `_classify_from_logits()` 已在 `party_s.py:411-436`，本次仅需重新评估旧 checkpoints

---

## 5. 数据资产索引

| 路径 | 内容 |
|------|------|
| `test-data/baseline-test-data/new-cls-baseline-test-data/epoch_metrics.jsonl` | Baseline 5 epoch 指标 |
| `test-data/baseline-test-data/new-cls-baseline-test-data/checkpoint_epoch_000.pt` ~ `004.pt` | Baseline 5 个 checkpoint |
| `test-data/baseline-test-data/new-cls-baseline-test-data/best_checkpoint.pt` | Baseline 最佳 (epoch 1) |
| `test-data/SLG-test-data/cls-SLG-test-data/epoch_metrics.jsonl` | SLG 5 epoch 评估指标 (新) |
| `test-data/SLG-test-data/cls-SLG-test-data/logs/epoch_000_evaluate_metrics.json` ~ `004_...` | SLG 每 epoch 详细指标 |
| `test-data/SLG-test-data/cls-SLG-test-data/_SAVE_20260727_0706/checkpoint_epoch_*.pt` | SLG 原始 5 epoch checkpoints |
| `src/scripts/evaluate_slg_cls.py` | SLG 重新评估脚本（新建）|

---

## 6. 结论

Baseline CLS 在 BioTriplex 7-class 分类任务上的精度优于 SLG-HE-PIR 异构 PIR 三方微调约 9-11 个百分点。该差距主要来自 SLG-HE-PIR 长尾类别的弱分类能力。建议下一步:

1. 调查 SLG-HE-PIR `macro_f1` 偏低原因（类别不平衡？Label noise？HE-PIR 信息损失？）
2. 尝试 class-balanced loss / focal loss for SLG-HE-PIR
3. 增大 SLG-HE-PIR 的 LoRA rank 或 target modules（当前已含 MLP 层）
4. 评估 10+ epoch 的 SLG 训练以验证是否能突破饱和
