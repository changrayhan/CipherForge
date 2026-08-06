# AccuracyAblationTest 代码审计与数据真实性报告

> **审计日期**：2026-08-01
> **审计范围**：现有 AccuracyAblationTest 全部代码 + Baseline/SLG 真实训练/推理代码 + 真实数据
> **目的**：在设计"加密/量化/加噪/gold-only 精度损失定量分解"方案前，必须先理解现有实现**实际做了什么**、**与 SLG 真实数据是否吻合**、**有哪些 bug/局限**

---

## 1. 现有实验框架的本质

### 1.1 当前实现是 "Logits-Level Noise Injection"，不是训练时真实量化

`accuracy_ablation/quant_hooks.py:155-209` 的 `inject_quant_noise()` 函数：

```python
def inject_quant_noise(logits: np.ndarray, spec: QuantNoiseSpec, seed: int, ...) -> np.ndarray:
    """对单个样本的 7 维 logits 注入量化税噪声。"""
    if spec.variant in ("Q0", "Q0'"):
        return logits   # 短路：Q0/Q0' 完全无噪声
    rng = np.random.default_rng(seed)
    out = logits.astype(np.float64).copy()
    if spec.v_round_sigma > 0:
        out += rng.normal(0, spec.v_round_sigma, size=out.shape)   # 仅 7 维 logits 加噪
    ...
```

**这意味着**：
- 所有量化税通过**手工选择的 N(0, σ) 噪声**模拟
- σ 值在 `quant_hooks.py:115-146` 的 `make_spec()` 中**硬编码**（v_round_sigma=0.5, g_h_int_sigma=0.5, g_h_bf16_sigma=1.5）
- **没有真实地修改 V 矩阵、没有真实地修改 a_t、没有真实地修改 g_H**

**这是当前实验的根本缺陷**：报告里宣称的"Q1=V 量化税"、"Q2=协议约束税"是**通过 logits-level 噪声**模拟的，不是真实 SLG 训练中的对应操作。

### 1.2 应用流程

`accuracy_ablation/eval_replay.py:144-209` 的 `replay_variant()`：
1. 读取 baseline 的 `infer_outputs_epoch_*.json`（每 epoch 一个）
2. 用 `_remap_doc_keys` 把 baseline 的 `_relation_<fine>` doc_key 映射到 gold 的 `_rel_<coarse>` 格式
3. 调用 `apply_variant_to_infer_outputs()` 对每个 doc_key 的 7 维 logits 加噪
4. 调用 baseline 的 `evaluate_metrics.py` 计算指标

**关键限制**：**baseline 的 logits 在训练时已经收敛**，加噪后的精度损失**仅反映"对单次推理的 logits 加这种分布的噪声会损失多少精度"**，**不反映"训练时累积的精度损失"**。

---

## 2. 数据划分与 doc_key 映射问题（关键 bug）

### 2.1 三方样本量不一致

| 数据源 | val_samples | 来源 | Evaluator |
|--------|------------|------|-----------|
| Baseline epoch_metrics.jsonl | **213** | baseline 重训 | sklearn `roc_auc_score(..., average='macro', multi_class='ovr')` |
| Q0/Q1/Q2/Q3 模拟 | **168** | baseline infer_outputs + `_remap_doc_keys` | 同 baseline |
| SLG epoch_metrics.jsonl | **203** | SLG 重训 | 自定义 per-class `roc_auc_score` |

### 2.2 doc_key 格式差异

| 来源 | doc_key 格式 | 例子 |
|------|------------|------|
| Baseline (biotriplex_qakshot_dataset) | `_relation_<fine_relation>` | `..._gene_X_disease_Y_relation_decreased_expression` |
| SLG (biotriplex_dataset) | `_rel_<coarse_relation>` | `..._gene_X_disease_Y_rel_expression_change` |
| Gold (test_gold_general_qa.txt) | `_rel_<coarse_relation>` | 与 SLG 相同 |

### 2.3 _remap_doc_keys 的 bug

`accuracy_ablation/eval_replay.py:212-250`：

```python
def _remap_doc_keys(infer_outputs: dict, base_to_full: dict[str, str]) -> tuple[dict, dict]:
    out = {}
    mapping = {}
    for base_dk, entry in infer_outputs.items():
        if "_relation_" in base_dk:
            base_part = base_dk.rsplit("_relation_", 1)[0]
            fine_rel = base_dk.rsplit("_relation_", 1)[1]
            coarse = FINE_TO_GENERAL.get(fine_rel.lower().strip())
            if coarse:
                candidate = f"{base_part}_rel_{coarse.replace(' ', '_')}"
                if candidate in [k for k in base_to_full.values()]:
                    # ⚠️ Bug: 这里 candidate 是 base_part + "_rel_" + coarse
                    # 但 base_to_full 是 {base: full_doc_key} 字典
                    # 实际检查 candidate 是否在 base_to_full.values() 中
                    # 而 base_to_full.values() 中的 key 是 gold 的 doc_key (含 _rel_)
                    # 应该能匹配
                    if candidate in infer_outputs or True:  # ⚠️ 这个 "or True" 永远为真
                        out[candidate] = entry
                        mapping[base_dk] = candidate
                        continue
        out[base_dk] = entry  # ⚠️ fallback：直接保留原 doc_key
        mapping[base_dk] = base_dk
    return out, mapping
```

**实际统计**：
- baseline infer_outputs: 213 entries
- 通过 `_remap_doc_keys` 成功映射: 189 (手动模拟)
- Q0/Q1/Q2/Q3 模拟最终评估样本: **168**
- gold 文件总样本: **182**

**为什么 189 → 168**：
- 189 个 candidate 确实在 gold 中（通过 `candidate in [k for k in base_to_full.values()]`）
- 但 168 是 `evaluate_metrics.py` 实际评估的样本数（common doc_keys 交集）
- 推测：有 21 个 candidate 在 gold 但 baseline infer_outputs 没有对应 logits（或反之）

**关键证据**：
```
baseline 总数: 213
成功映射到 gold: 189 (在 _remap_doc_keys 中)
gold 总数: 182
gold 中未在 baseline 出现的: 0
最终 evaluate_metrics 评估: 168
```

**结论**：**现有 Q0/Q1/Q2/Q3 模拟使用的 168 个样本 ≠ baseline epoch_metrics.jsonl 的 213 个样本**。这意味着 Q0/Q1/Q2/Q3 的精度不能直接与 baseline epoch_metrics.jsonl 对比。

---

## 3. 评估口径不一致（影响 macro_auc 对比）

### 3.1 Baseline/Q0-Q3 用 sklearn macro-ovr

`baseline/classification_genrel/scripts/evaluate_metrics.py:272-275`：

```python
macro_auc_ovr = roc_auc_score(
    y_true_bin, y_score, average="macro", multi_class="ovr"
)
```

**问题**：对所有 7 类（含 0-support 类）都计算 AUC，0-support 类因为 y_true 全 0 会触发 sklearn `ValueError`，被 `try/except` 兜底为 None。但 `n=168` 时实际包含哪些类？

### 3.2 SLG 用自定义 per-class 循环

`src/training/biotriplex_metrics.py:259-275`：

```python
per_class_auc = np.full(n_classes, np.nan, dtype=np.float64)
for k in labels_range:
    yt_k = (y_t == k).astype(np.int32)
    if yt_k.sum() == 0:
        continue   # 跳过 0-support 类
    if yt_k.sum() == len(yt_k):
        continue   # 跳过全 1 类
    try:
        per_class_auc[k] = float(roc_auc_score(yt_k, y_score[:, k]))
    except ValueError:
        per_class_auc[k] = np.nan
valid_aucs = per_class_auc[~np.isnan(per_class_auc)]
if valid_aucs.size > 0:
    macro_roc_auc = float(np.mean(valid_aucs))
```

**差异**：SLG 跳过 0-support 类（no relation, relation undefined），baseline/Q0-Q3 不跳过。

### 3.3 实际数据中 macro_auc 的可比性

| 指标 | Baseline (real, last 3 epoch avg) | SLG (real, last 3 epoch avg) | Q0 (noise sim, 3 seed avg) | Q3 (full noise sim, 3 seed avg) |
|------|--------------------------------|------------------------------|---------------------------|--------------------------------|
| macro_auc | ~0.78 | ~0.68 | ~0.76 | ~0.61 |
| 计算方式 | sklearn macro-ovr (含 0-support) | per-class 跳过 0-support | sklearn macro-ovr | sklearn macro-ovr |

**问题**：
- SLG 的 macro_auc=0.68 实际是 5 类（pathological, modulatory, expression change, diagnosis, therapy）的平均
- Baseline 的 macro_auc=0.78 是 7 类（含全 0 的 no relation 和 relation undefined，AUC=0.5）的平均
- **这两个 macro_auc 数值上不可比**

---

## 4. 现有 Noise Spec 的设计缺陷

### 4.1 σ 值是手工选择的，不是从协议代码推出的

`accuracy_ablation/quant_hooks.py:107-114`：

```python
"""Sigma 标定（基于 epoch 1 baseline macro_f1=0.2885 的 sweep）：
  σ=0.0 → 0.2885
  σ=0.5 → 0.2888 (≈ 无影响)
  σ=1.0 → 0.2343 (-5.4pp)
  σ=1.5 → 0.2246 (-6.4pp)
  σ=2.0 → 0.2079 (-8.1pp)
  σ=3.0 → 0.1886 (-10.0pp)
"""

if variant == "Q1":
    return QuantNoiseSpec(variant="Q1", v_round_sigma=0.5)
if variant == "Q2'":
    return QuantNoiseSpec(variant="Q2'", v_round_sigma=0.5, g_h_int_sigma=0.5)
if variant == "Q2":
    return QuantNoiseSpec(variant="Q2", v_round_sigma=0.5, g_h_int_sigma=0.5, protocol_constraint=True)
if variant == "Q3":
    return QuantNoiseSpec(variant="Q3", v_round_sigma=0.5, g_h_int_sigma=0.5, g_h_bf16_sigma=1.5, protocol_constraint=True)
```

**问题**：
1. **σ 值是 sweep 出来的，目的是让最终 macro_f1 接近 SLG 真实值**（Q3=0.1495 vs SLG=0.1515）
2. **不是从协议数学推导的**——例如 V 矩阵 fixed-point 量化的真实 σ 应该约 √hidden·1/(2·scale) ≈ 0.0032，而不是 sweep 出的 0.5
3. **σ=0.5 对 7 维 logits 来说相当于 50% 噪声**——这远超任何真实量化噪声

### 4.2 Q2 protocol_constraint 的实现是 heuristic

`accuracy_ablation/quant_hooks.py:196-203`：

```python
if spec.g_h_int_sigma > 0:
    out += rng.normal(0, spec.g_h_int_sigma, size=out.shape)
    if spec.protocol_constraint:
        if gold_ids is not None:
            gid = int(gold_ids if np.isscalar(gold_ids) else gold_ids.flatten()[0])
            if 0 <= gid < len(out):
                out[gid] += rng.uniform(-1.5, 0.5)  # ⚠️ 这个 [-1.5, +0.5] 是手工设计的
```

**问题**：
- `out[gid] += rng.uniform(-1.5, 0.5)` 是手工选择的参数，目的是让 protocol 税 = 4.93pp
- 但真实 SLG 中 gold-only 协议的精度损失**不通过"在 gold 位置加噪声"实现**——而是通过 `g_H = a_t - V_gold`（范数 ≈ √hidden·0.01 ≈ 0.64）作为上游梯度反传到所有 token 位置
- 当前 noise model **没有真实模拟这个机制**

### 4.3 Q0/Q0' 完全相同（LoRA 参数量差异未模拟）

`accuracy_ablation/quant_hooks.py:115-118`：

```python
if variant == "Q0":
    return QuantNoiseSpec(variant="Q0")
if variant == "Q0'":
    return QuantNoiseSpec(variant="Q0'")
```

**Q0 和 Q0' 都是无噪声**，差异仅在 `variant` 字段。

`apply_variant_to_infer_outputs()` 对 Q0 和 Q0' 都直接 `return logits`（无操作）。

**实际效果**：
- Q0' 应该是"baseline 配置（2-target LoRA）"
- Q0 应该是"SLG 配置（7-target LoRA）"
- 但两者都用 baseline infer_outputs 的 logits，**没有真正的 LoRA 配置差异**

**这意味着**：Q0' → Q0 的 0.00pp 差异**不是因为 2-target vs 7-target LoRA 配置相同**（实际它们配置不同），而是**因为根本没模拟**。

---

## 5. SLG 真实数据 vs Q3 模拟数据的 per-class 差异（最关键的发现）

### 5.1 SLG 真实 per_class (epoch 4 best)

```
pathological            : F1=0.6250, support=51
modulatory              : F1=0.1467, support=13
expression change       : F1=0.2887, support=77
diagnosis               : F1=0.0000, support=51  ← 彻底失败
therapy                 : F1=0.0000, support=11  ← 彻底失败
no relation             : F1=0.0000, support=0
relation undefined      : F1=0.0000, support=0
```

### 5.2 Q3 模拟 per_class (seed=42, epoch=0)

```
pathological            : F1=0.2921, support=43
modulatory              : F1=0.0286, support=10
expression change       : F1=0.1481, support=71
diagnosis               : F1=0.1852, support=36  ← 还能预测对
therapy                 : F1=0.1667, support=8   ← 还能预测对
no relation             : F1=0.0000, support=0
relation undefined      : F1=0.0000, support=0
```

### 5.3 关键差异

| Class | SLG 真实 | Q3 模拟 | 差异 | 解释 |
|-------|---------|--------|------|------|
| pathological | 0.625 | 0.292 | -0.33 | Q3 noise 过度压低 |
| modulatory | 0.147 | 0.029 | -0.12 | Q3 noise 过度压低 |
| expression change | 0.289 | 0.148 | -0.14 | Q3 noise 过度压低 |
| **diagnosis** | **0.000** | 0.185 | **+0.185** | **SLG 真实彻底失败，但 Q3 模拟还有** |
| **therapy** | **0.000** | 0.167 | **+0.167** | **SLG 真实彻底失败，但 Q3 模拟还有** |

**结论**：
- **aggregate macro_f1 上**：Q3=0.1495 ≈ SLG=0.1515，看似吻合
- **per-class 分布上**：**完全不同**！SLG 在 diagnosis/therapy 上彻底失败，但 Q3 模拟没有复现这个失败

### 5.4 为什么 SLG 在 diagnosis/therapy 失败？

需要深入分析协议代码，但当前实现的 noise model 没有捕获这个机制。这说明：
- **Q3 的 noise model 是不完整的**
- **诊断 diagnosis/therapy 失败的真正原因可能是**：SLG 训练时 gold_ids 索引错配（sample 顺序不一致），或 U 端 12 层 transformer 累积误差，或 BFV 多项式模运算影响

---

## 6. 现有代码的 bug 清单

### 6.1 `_remap_doc_keys` 逻辑错误

`accuracy_ablation/eval_replay.py:243`：

```python
if candidate in infer_outputs or True:
```

**永远为真的条件**，应删除。

### 6.2 doc_key 格式边界处理缺失

当 baseline doc_key 是 `_relation_relation_undefined`（fine_rel=`relation undefined`）时，转换结果为 `_rel_relation_undefined`，这与 gold 的格式冲突。

实际数据中：
- baseline 有 24 个 `_relation_relation_undefined` 格式（来自 fine relation "relation undefined"）
- gold 中 `relation_undefined` 类的 doc_key 应该是从原始 fine 关系来的，但 gold 没有这些 doc_key

### 6.3 gold 文件仅 182 条 vs baseline 213 条

差距 31 条。可能原因：
- baseline `biotriplex_qakshot_dataset` 把多个 fine relation 都展开为独立样本
- gold `test_gold_general_qa.txt` 只保留 coarse relation 的一份

**修复方向**：在 baseline infer_outputs 上重新跑 `_remap_doc_keys`，使输出样本集与 baseline epoch_metrics.jsonl 的 213 个样本对齐。

### 6.4 evaluator 路径不一致

| 数据 | evaluator | macro_auc 计算 | 0-support 类处理 |
|------|----------|----------------|-----------------|
| Baseline | `evaluate_metrics.py` | sklearn macro-ovr | 含 0-support |
| Q0/Q1/Q2/Q3 | `evaluate_metrics.py` | sklearn macro-ovr | 含 0-support |
| SLG | `biotriplex_metrics.compute_classification_metrics` | per-class 跳过 | 跳过 0-support |

**修复方向**：在 `report_generator.py` 中统一用 `compute_classification_metrics` 重新评估 baseline 和 Q0-Q3 的 `infer_outputs_epoch_*.json`，得到统一口径。

---

## 7. 当前实验的根本局限

### 7.1 三层叠加的近似问题

| 层 | 真实协议操作 | 当前模拟方法 | 失真来源 |
|----|------------|------------|---------|
| V 矩阵量化 | `lm_head.weight = round(W * 10000) / 10000`（一次性） | 推理后 logits += N(0, 0.5) | σ 值与真实量化步长不对应；没有修改模型参数 |
| a_t 量化 | S 端 share 时 int64 round | 未单独模拟（合并到 V 量化税） | 数学上独立，但 noise model 合并 |
| g_H int64 量化 | M 端解密后 int64 round | 推理后 logits += N(0, 0.5) | 同上 |
| gold-only 协议 | `g_H = a_t - V_gold` 作为上游梯度 | logits[gold] += U(-1.5, +0.5) | **完全不同机制**：真实是 gradient flow，当前是 post-hoc logits perturbation |
| g_H bf16 转换 | `g_H.bfloat16()` 在 M 端 | 推理后 logits += N(0, 1.5) | σ 值 1.5 对应 bf16 step 1/256 但量纲不对 |

### 7.2 当前 noise model 的拟合优度

报告里 Q3 vs SLG 的差距 = 0.1495 vs 0.1515 = **-0.0020pp（Q3 比 SLG 略好）**。

但这个"吻合"是**aggregate macro_f1 上的拟合**，**per-class 分布完全不吻合**。

报告里的"Q3 与真实 SLG 的差 = 未建模精度残差 = 0.26pp"是**用 aggregate macro_f1 的差算出来的**，不能说明 noise model 真实地捕获了 SLG 协议的精度损失分布。

---

## 8. 数据真实性总结

| 维度 | 真实度 | 说明 |
|------|--------|------|
| Baseline 真实训练 | ✓ 真实数据 | epoch_metrics.jsonl + infer_outputs_epoch_*.json 都来自实际训练 |
| SLG 真实训练 | ✓ 真实数据 | epoch_metrics.jsonl 来自实际训练 |
| Q0/Q1/Q2/Q3 模拟 | ⚠️ 部分真实 | 基于 baseline 真实 logits，**加手工选择的 N(0, σ) 噪声** |
| 协议税的精确分解 | ❌ **不精确** | 当前 noise model 是拟合 SLG aggregate macro_f1 的结果，不是真实协议机制的模拟 |
| Per-class 分布 | ❌ **不吻合** | Q3 在 diagnosis/therapy 还能预测对，SLG 真实完全失败 |

---

## 9. 设计新方案前必须解决的问题

1. **样本数对齐**：168 (Q0-Q3 模拟) vs 213 (baseline 真实) vs 203 (SLG 真实) → 必须统一为同一份 203 样本
2. **evaluator 对齐**：sklearn macro-ovr vs 自定义 per-class 跳过 → 必须统一为同一 evaluator
3. **noise model 真实性**：当前 σ 是手工 sweep 出来的 → 必须从协议数学推导真实 σ
4. **Q0/Q0' 区分**：当前完全无差异 → 必须真正用不同 LoRA 配置重训（或至少模拟差异）
5. **gold-only 机制**：当前是 post-hoc 噪声 → 必须改为 gradient-level hook 或真实重训

---

## 10. 现有代码的可重用部分

| 模块 | 可重用度 | 备注 |
|------|---------|------|
| `quant_hooks.py` 中 `QuantNoiseSpec` dataclass | ✓ 可重用 | 字段设计合理（v_round_sigma, g_h_int_sigma, g_h_bf16_sigma, protocol_constraint） |
| `quant_config.py` 中 `QuantConfig` | ✓ 可重用 | yaml 配置读取逻辑 |
| `report_generator.py` 中 markdown 渲染 | ✓ 可重用 | 表格格式化、CSS 等 |
| `report_generator.py` 中 CI 计算 | ✓ 可重用 | `statistics.stdev`, t 分布等 |
| `eval_replay.py` 中 `_run_evaluate_metrics` | ✓ 可重用 | 调用 baseline evaluate_metrics.py 子进程 |
| `eval_replay.py` 中 `load_gold_map` | ✓ 可重用 | gold JSONL 解析 |
| `eval_replay.py` 中 `_remap_doc_keys` | ⚠️ 需修复 | 当前有逻辑 bug，需重新实现 |
| `quant_hooks.py` 中 `inject_quant_noise` | ⚠️ 仅作 fallback | 当新 noise model 不可用时，可作为基线 |

---

## 11. 设计新方案的关键决策点

在写新方案前需要确认：

1. **重训 vs 不重训**：
   - 选项 A：完全不重训，仅改进 noise model + 修复样本对齐
   - 选项 B：在 baseline 训练循环中注入 V 量化 + g_H 量化 hook（半真实）
   - 选项 C：在 baseline 上用 7-target LoRA 重训 Q0（对比 Q0'），其它变体不重训
   
2. **gold-only 协议如何模拟**：
   - 选项 A：保持当前 post-hoc 噪声，但用真实梯度范数推导 σ
   - 选项 B：在 baseline training 中 monkey-patch CE loss，改用 `g_H = a_t - V_gold` 路径
   - 选项 C：保持协议设计层面分析（不模拟训练）

3. **如何对比 Q0/Q1/Q2/Q3 与 SLG**：
   - 选项 A：使用同一份 203 样本（SLG 口径）
   - 选项 B：使用同一份 213 样本（baseline 口径）
   - 选项 C：使用同一份 baseline 推理时实际评估的样本（约 168/189）
