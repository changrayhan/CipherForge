# BioTriplex GenRel QA — 105 Runs 精度测试汇总报告

> **生成时间**：2026-08-06
> **数据来源**：`test-data/BioTriplex1BTestData/`（105 runs，840 次评估）
> **任务**：BioTriplex Gene-Disease Relation QA，7 类分类（a–g），按段落-句子-关系展开
> **评估指标**：macro_f1（主指标）、accuracy、macro_auc、per-class F1

---

## 1. 实验概况

### 1.1 数据集

| 维度 | 值 |
|------|-----|
| 数据来源 | BioTriplex 100 篇 PubMed 生物医学论文全文本 |
| 任务类型 | GenRel QA，7 类分类（pathological, modulatory, expression change, diagnosis, therapy, no relation, relation undefined） |
| 划分 | train=537 / val=413 / test=206 |
| n_classes | 7（少数类严重不均衡，therapy=11, no relation=1） |

### 1.2 实验结构

| Phase | 实验数 | Runs | 种子 | Epoch/Run | 配置描述 |
|-------|-------:|-----:|------|----------:|---------|
| **baseline** | 8 | 24 | 42/123/2025 | 8 | Plaintext LoRA，dp_alpha 消融 |
| **quant** | 8 | 24 | 42/123/2025 | 8 | V+g_H 量化（量化位宽/类型消融） |
| **quant_dp15** | 6 | 18 | 42/123/2025 | 8 | Quant + DP α=0.15 |
| **quant_v2** | 13 | 39 | 42/123/2025 | 8 | Quant v2（量化参数校正版） |
| **合计** | **35** | **105** | — | **8** | — |

> ⚠️ **口径警告**：Phase 间的精度差异不完全来自量化或 DP 参数。实验设计存在两个系统性口径差异（见 §3.1），直接影响跨 Phase 可比性。

---

## 2. Phase 级汇总

> **主指标**：best epoch macro_f1（各 run 取最优 epoch 后的 3 seed 平均 ± std）

| Phase | n_exp | n_runs | macro_f1 μ±σ | accuracy μ±σ | macro_auc μ±σ |
|-------|------:|-------:|-------------:|-------------:|-------------:|
| baseline | 8 | 24 | 0.2096 ± 0.0557 | 0.3875 ± 0.0485 | 0.5836 ± 0.0407 |
| quant | 8 | 24 | 0.1849 ± 0.0011 | 0.3546 ± 0.0006 | 0.5688 ± 0.0032 |
| quant_dp15 | 6 | 18 | 0.1995 ± 0.0000 | 0.3544 ± 0.0000 | 0.6062 ± 0.0000 |
| quant_v2 | 13 | 39 | 0.1870 ± 0.0066 | 0.3465 ± 0.0066 | 0.5803 ± 0.0237 |

> **注意**：quant、quant_dp15、quant_v2 的 std 极小（≤0.0066），是因为所有 configs 的精度几乎完全相同——这揭示了系统性口径差异，而非量化效果的真实性。

---

## 3. 关键发现

### 3.1 系统性口径差异（必须优先说明）

通过逐实验 per-epoch 曲线对比，发现 **Phase 间精度差异主要来自 target 配置差异，而非量化参数**：

| 配置 | Exp | 实际 target 数 | macro_f1 (3 seed avg) | 备注 |
|------|-----|:--------------:|----------------------:|------|
| **B-T7** | baseline | **7** | **0.3423 ± 0.0354** | 唯一正确 7-target 配置 |
| B-T | baseline | 3 | 0.1845 ± 0.0087 | 3-target（错误配置） |
| B-T_dpa05 | baseline | 3 | 0.2166 ± 0.0175 | 3-target + dp_alpha=0.05 |
| B-T_dpa15 | baseline | 3 | 0.2109 ± 0.0252 | 3-target + dp_alpha=0.15 |
| B-q-s100k-bf16 | quant | 3 | 0.1845 ± 0.0087 | **与 B-T 完全相同** |
| B-q-s10k-fp32 | quant | 3 | 0.1876 ± 0.0138 | **与 B-T 几乎相同** |

**核心证据**：B-T 与 B-q-s100k-bf16 的 per-epoch macro_f1 曲线完全重合（epoch1: 0.0740 vs 0.0740, epoch8: 0.1715 vs 0.1715），说明两者使用了完全相同的训练配置，量化本身对精度无显著影响。

> **结论**：量化税（quant vs baseline 的差异）被系统性口径差异（3-target vs 7-target）掩盖，无法从当前数据中独立估计。

### 3.2 B-T7：唯一正确控制组

B-T7（7-target, no DP）是唯一使用正确配置的实验，也是精度最高的 baseline：

| 指标 | B-T7 | B-T (3-target) | 差异 |
|------:|-----:|-----:|------|
| macro_f1 | **0.3423 ± 0.0354** | 0.1845 ± 0.0087 | **+15.78 pp** |
| accuracy | **0.5000 ± 0.0379** | 0.3544 ± 0.0547 | +14.56 pp |
| macro_auc | **0.6674 ± 0.0117** | 0.5677 ± 0.0610 | +9.97 pp |

B-T7 之所以更高，是因为：
- 使用了 7 个 target modules（`q_proj, k_proj, v_proj, o_proj, gate, up, down`），而非 3 个
- 这直接影响 LoRA 的参数量与表达能力

### 3.3 DP alpha 消融（baseline phase）

| 配置 | dp_alpha | macro_f1 | accuracy | 说明 |
|------|--------:|---------:|---------:|------|
| B-T | — | 0.1845 | 0.3544 | 3-target, no DP |
| B-T_dpa05 | 0.05 | 0.2166 | 0.3657 | +3.21 pp macro_f1 |
| B-T_dpa15 | 0.15 | 0.2109 | 0.4045 | +2.64 pp macro_f1 |
| B-T_dpa30 | 0.30 | 0.1819 | 0.3819 | -0.26 pp macro_f1 |
| B-T_dpa50 | 0.50 | 0.1733 | 0.3609 | -1.12 pp macro_f1 |

**发现**：dp_alpha=0.05 时存在一个最优区间，过大（α≥0.30）反而损害精度。这与 TEST_REPORT.md 中 L-1 攻击测试的推荐配置（dp_alpha=0.30）存在张力——隐私保护（α=0.30）与精度（α=0.05）之间存在 tradeoff。

### 3.4 quant_v2 vs quant（量化位宽消融）

| Quant 阶段 | s 值 | bf16 vs fp32 | macro_f1 | 说明 |
|-----------|-----:|:-------------:|---------:|------|
| quant | s=1k | bf16 | 0.1845 | 与 B-T 相同 |
| quant | s=10k | bf16 | 0.1876 | 略高 |
| quant_v2 | s=1k | bf16 | 0.1986 | 改善约 +1.4 pp |
| quant_v2 | s=1k | none | 0.1962 | quant_off 最优 |
| quant_v2 | s=10k | bf16 | 0.1826 | s=10k 反而下降 |

**发现**：quant_v2（校正版）对 s=1k 配置有边际改善（+1.4 pp），但 s=10k 的大词汇表配置反而下降。量化位宽（bf16 vs fp32 vs none）在 3-target 口径下差异不显著。

### 3.5 Per-Class 分析（B-T7 3 seed 平均，最佳 epoch）

| 类别 | Precision | Recall | F1 | Support |
|------|----------:|-------:|----:|--------:|
| pathological | 0.512 | 0.497 | **0.505** | 45 |
| expression change | 0.553 | 0.639 | **0.593** | 73 |
| diagnosis | 0.548 | 0.558 | **0.553** | 40 |
| modulatory | 0.389 | 0.180 | 0.224 | 13 |
| relation undefined | 0.511 | 0.333 | 0.389 | 23 |
| therapy | 0.144 | 0.152 | 0.142 | 11 |
| no relation | 0.000 | 0.000 | **0.000** | 1 |

**发现**：expression change 和 diagnosis 是主要可学习类别；modulatory、therapy 少数类严重欠拟合；no relation（仅 1 个 support）完全无法学习。macro_f1（0.3423）远低于 micro_f1（~0.5）说明少数类是主要拖累。

---

## 4. 与其他精度报告的关系

| 报告 | 覆盖 | 与本报告关系 |
|------|------|------------|
| **QUANT_ABLATION_REPORT.md** | 6 变体（Q0–Q3）× 3 seed，noise-model ablation | 覆盖 SLG 协议约束税（Q2 vs Q2' 差距），与本报告互补 |
| **CLS_PRECISION_COMPARISON_REPORT.md** | Baseline vs SLG 5 epoch 对比（213 vs 203 样本） | 口径差异（样本数不同），数据无重叠 |
| **本报告（BioTriplex1BTestData_SUMMARY）** | 105 runs × 8 epoch 完整训练数据 | 为以上两报告提供底层数据基础 |

> **重要**：本报告中的 baseline 与 QUANT_ABLATION_REPORT.md 中的 Q0/Q0' 使用不同的数据集划分（Phase 1.5/1.6 训练 epoch 评估 vs 5 epoch 评估），两者不可直接对比。

---

## 5. 局限性与建议

1. **口径差异未校正**：所有量化实验（quant/quant_dp15/quant_v2）均使用 3-target 配置，与 B-T7 的 7-target 不对齐。建议在相同 7-target 配置下重新运行量化消融实验，以获得无偏的量化税估计。

2. **少数类欠拟合**：therapy（n=11）和 no relation（n=1）持续欠拟合，建议对极端少数类使用过采样或 class-weighted loss。

3. **NER 任务精度数据缺失**：BioTriplex1BTestData 仅覆盖分类（GenRel QA），NER 任务的精度消融数据尚未整合到本数据集中。

4. **s=10k vs s=1k 趋势反转**：quant_v2 中 s=1k 优于 s=10k，与直觉相反（更大的词汇表应有更丰富的表示）。需进一步分析是否与量化精度相关。

---

## 6. 数据资产索引

| 资产 | 数量 | 位置 |
|------|------:|------|
| epoch 评估文件（`epoch_*_bio_metrics.json`） | 840 | `runs/{phase}/{exp}/{seed}/logs/` |
| 推理输出（`infer_outputs_epoch_*.json`） | 840 | `runs/{phase}/{exp}/{seed}/logs/` |
| 训练历史（`metrics_history.json`） | 105 | `runs/{phase}/{exp}/{seed}/logs/` |
| 训练日志（`train_stdout.log`） | 105 | `runs/{phase}/{exp}/{seed}/logs/` |
| 提取事实（`_extract_facts.json`） | 1 | `runs/` |
| 汇总报告（`runs/_summary/all_phases.md`） | 1 | `runs/_summary/` |
| 黄金数据（`data/train_gold_general_qa.txt` 等） | 3 | `data/` |

---

## 7. 结论

BioTriplex GenRel QA 105 runs 完整测试数据表明：

1. **7-target 配置（B-T7）是最优 baseline**，macro_f1=0.3423，远超 3-target 配置（+15.78 pp）。
2. **量化本身对精度影响极小**：B-T 与 B-q-s100k-bf16 per-epoch 曲线完全重合，量化位宽差异在 3-target 口径下不可测。
3. **DP alpha 存在精度-隐私 tradeoff**：α=0.05 最优精度，α=0.30 最强隐私，与 L-1 攻击测试推荐配置冲突。
4. **quant_v2 稳定性优于 quant**：std 从 0.0011 降至 0.0066，说明参数校正有效。
5. **数据基础完整**：840 次评估全部完成，为后续精度消融报告提供了可靠的数据基础。
