# AccuracyAblationTest 设计文档索引

> 本目录包含 AccuracyAblationTest 精度损失定量分解方案的**完整设计文档**。

---

## 文档结构

### 📋 [`code_audit_findings.md`](./code_audit_findings.md) — 代码审计报告

**内容**：在设计新方案前，对现有 AccuracyAblationTest 全部代码 + Baseline/SLG 真实训练/推理代码 + 真实数据进行的**穷尽审计**。

**关键发现**（摘要）：

1. **现有实现是 "Logits-Level Noise Injection"**，不是训练时真实量化
2. **三方样本数不一致**：168 / 203 / 213
3. **evaluator 不一致**：sklearn vs 自定义 per-class
4. **`_remap_doc_keys` 有逻辑 bug**：`or True` 永远为真
5. **per-class 分布完全不吻合**：Q3 模拟还能预测 diagnosis/therapy，SLG 真实完全失败
6. **Q0 和 Q0' 完全相同**：LoRA 配置差异未模拟

---

### 📋 [`root_cause_analysis.md`](./root_cause_analysis.md) — SLG diagnosis/therapy F1=0 根因分析

**内容**：深挖 SLG 真实训练数据下 diagnosis/therapy 类 F1=0 的根本原因，确认是否属于任务评估失配。

**核心结论**：**不是任务评估失配，是模型 mode collapse**。

**3 个根因（按严重程度排序）**：

| 根因 | 严重度 | 真实精度损失贡献 | 证据 |
|------|--------|-----------------|------|
| **R3** SLG 模型 mode collapse → 67% 预测为 modulatory | **关键** | ~10pp (diagnosis/therapy 完全失败) | y_pred_distribution 严重倾斜（modulatory=137/203, diagnosis=0/203） |
| **R1** SLG 数据集类过滤 negative 样本不一致 | 中 | ~0.5pp | baseline 213 vs SLG 203，差 24 个 negative 样本 |
| **R2** baseline/SLG 实体处理边界 case 不一致 | 中 | ~0.3pp | positive 样本膨胀 +14 个（diagnosis +8, pathological +3, expression +3）|

**任务评估失配的复核结论**：
- ✅ `n_parse_failures=0`，证明 2026-07-26 修复后评估器无解析失败
- ✅ `pred_logits` 有值，compute_classification_metrics 拿到正确 7 维 logits
- ✅ baseline 在 diagnosis/therapy 上 F1>0（0.485/0.254），证明评估器正确

**关键洞察**：
- Q3 noise model 在 aggregate macro_f1 上"拟合成功"是**统计巧合**
- Q3 noise model **无法复现 mode collapse**（这是 LoRA + 协议交互的产物）
- 当前 noise-model ablation **不能用作协议税的机制分解**

**数据集类差异的代码证据**：
- baseline `correct_relation_char_index` (line 610-681)：**不根据 return_neg_relations 过滤**
- SLG `_correct_relation_char_index` (line 272-360)：**主动过滤** negative 样本

---

### 📋 [`proposal.md`](./proposal.md) — 新方案设计

**内容**：基于代码审计，设计**四阶段精度损失分解方案**：
- 阶段 A：口径统一（样本对齐 + evaluator 对齐）
- 阶段 B：精度损失分解的核心对比矩阵
- 阶段 C：加密税的精确分离（真实 V/协议/g_H 模拟）
- 阶段 D：累积梯度分解（最终输出表）

**核心设计思路**：

```
最终目标：分解 "Baseline (macro_f1=0.2633) → SLG (macro_f1=0.1515) = -11.18pp" 中各税占比

分解方法：
1. 先修 bug：_remap_doc_keys + evaluator 对齐
2. 真实模拟：在 baseline logits 上做真实 V 量化 / gold-only 协议 / g_H 量化
3. 对比：真实模拟 vs noise model 拟合 vs SLG 真实
4. 输出：累积梯度分解表 + 残差分析

关键决策点（D1-D4）需用户确认：
- D1: 是否重训 baseline?
- D2: gold-only 协议的真实模拟方式?
- D3: 报告粒度?
- D4: 是否修正 Q0/Q0' LoRA 配置差异?
```

**预期结论**（基于代码审计推断）：
- 当前 noise model 的 σ 比真实协议 σ 大 100-1000 倍
- noise model 能拟合 aggregate macro_f1 但 per-class 分布完全错位
- SLG 在 diagnosis/therapy 失败是 noise model 完全未捕获的协议税
- **协议税在精度损失中占比最大（约 50%）**，加密税占比 < 5%

---

## 关键数据（基于真实数据）

### Baseline 真实训练（epoch_metrics.jsonl, val_samples=213）

| epoch | macro_f1 | macro_auc (sklearn) | diagnosis_F1 | therapy_F1 |
|-------|----------|---------------------|--------------|------------|
| 0 | 0.1640 | 0.7989 | 0.128 | 0.286 |
| **1 (best)** | **0.2633** | 0.7771 | **0.485** | **0.254** |
| 2 | 0.2343 | 0.7998 | 0.387 | 0.353 |
| 3 | 0.2368 | 0.7703 | 0.523 | 0.320 |
| 4 | 0.1869 | 0.7786 | 0.291 | 0.267 |

### SLG 真实训练（epoch_metrics.jsonl, val_samples=203）

| epoch | macro_f1 | macro_auc (custom) | diagnosis_F1 | therapy_F1 |
|-------|----------|---------------------|--------------|------------|
| 0 | 0.1444 | 0.6796 | **0.000** | **0.000** |
| 1 | 0.1492 | 0.6790 | **0.000** | **0.000** |
| 2 | 0.1438 | 0.6764 | **0.000** | **0.000** |
| 3 | 0.1453 | 0.6784 | **0.000** | **0.000** |
| **4 (best)** | **0.1515** | 0.6815 | **0.000** | **0.000** |

### Q0-Q3 模拟（已生成但有 bug, n_samples=168）

| variant | macro_f1 | diagnosis_F1 | therapy_F1 |
|---------|----------|--------------|------------|
| Q0' (无噪声) | 0.2292 | 0.103 | 0.293 |
| Q0 (无噪声) | 0.2292 | 0.103 | 0.293 |
| Q1 (V quant noise) | 0.2254 | ? | ? |
| Q2' (g_H noise) | 0.2215 | ? | ? |
| Q2 (+protocol noise) | 0.1722 | ? | ? |
| Q3 (+bf16 noise) | 0.1495 | 0.185 | 0.167 |

---

## 待办

- [ ] 用户确认 D1-D4 决策
- [ ] 实施 P0.1: 修复 `_remap_doc_keys` bug
- [ ] 实施 P0.2: 写 `unified_evaluator.py`
- [ ] 实施 P0.3: 用统一 evaluator 重评所有数据
- [ ] 实施 P1.1-1.4: 真实 V / gold-only / g_H / bf16 模拟
- [ ] 实施 P2: 累积梯度报告生成
- [ ] 实施 P3: 最终报告整合

---

## 联系方式

设计者：基于 2026-08-01 代码审计
预期实施时间：20-25 小时（完整 P1.x）或 8-10 小时（仅 P0 + P2）