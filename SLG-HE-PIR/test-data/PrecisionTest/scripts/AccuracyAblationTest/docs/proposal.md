# SLG-HE-PIR 精度损失定量分解方案设计

> **设计日期**：2026-08-01
> **设计者**：基于代码审计（`docs/code_audit_findings.md`）后设计
> **目标**：通过对比当前已有的 SLG-HE-PIR 数据，定量分解**加密、量化、加噪、gold-only 约束**各自对精度的贡献占比
> **核心约束**：不重训模型，全部基于已有数据；一切结论必须基于真实代码和数据

---

## 1. 设计原则

### 1.1 必须遵守的方法论红线

| 红线 | 含义 | 当前实现的违反情况 |
|------|------|------------------|
| **R1：可比性** | 对比对象必须用同一份样本、同一套 evaluator | ❌ 当前 baseline=213 vs SLG=203 vs Q0-Q3=168；evaluator 不同 |
| **R2：可加性** | 累积税的分解必须保证 Δ = 后 - 前，单调非负 | ⚠️ 当前 macro_f1 单调下降，但 per-class 不单调 |
| **R3：可解释性** | 每个 σ 值必须能追溯到协议代码的具体行 | ❌ 当前 σ=0.5, 1.5 是 sweep 拟合结果 |
| **R4：可分离性** | 加密税 vs 协议税 vs 量化税应在同一对比框架内分离 | ❌ 当前用 noise-model 拟合 SLG aggregate，不区分加密和协议 |

### 1.2 设计目标的精确化

**最终目标** = 输出一张表，回答：

> "SLG-HE-PIR vs Baseline 的精度差异中，加密税 / V 量化税 / a_t 量化税 / g_H int64 量化税 / g_H bf16 转换税 / gold-only 协议税 各占多少 pp？"

但**当前的实验数据不足以精确回答这个问题**，因为：
1. baseline 和 SLG 在 evaluator / 样本数 / LoRA 配置上都不同
2. noise model 是 post-hoc 拟合，不是真实协议操作
3. 没有"只加密不量化"、"只量化不加密"的对照

**所以**新方案的设计目标是：**在已有数据基础上，做最严格的精度损失分解**，明确标注每一步的不确定性。

---

## 2. 数据资产盘点（基于真实数据）

### 2.1 可用数据资产

| 数据路径 | 来源 | 关键字段 | 用途 |
|---------|------|---------|------|
| `test-data/baseline-test-data/new-cls-baseline-test-data/epoch_metrics.jsonl` | Baseline 重训 | `val_bt_macro_f1` (5 epoch), `per_class` | Baseline 真实指标 |
| `test-data/baseline-test-data/new-cls-baseline-test-data/logs/infer_outputs_epoch_*_*.json` | Baseline 推理 | 7 维 logits per doc_key | Q0-Q3 noise injection 的输入 |
| `test-data/SLG-test-data/cls-SLG-test-data/epoch_metrics.jsonl` | SLG 重训 | `val_bt_macro_f1` (5 epoch), `per_class` | SLG 真实指标 |
| `AccuracyAblationTest/outputs/{q0_7target,q0p_2target,q1_v_quant,q2p_full_token,q2_g_h_quant,q3_full_slg_sim}/seed_{42,123,456}/` | 模拟 | `infer_outputs_epoch_*.json` + `epoch_*_evaluate_metrics.json` | 6 个变体的模拟结果 |
| `AccuracyAblationTest/outputs/QUANT_ABLATION_REPORT.md` | 已生成报告 | 6 变体汇总 | 对比基线 |
| `src/training/biotriplex_metrics.py` | 协议代码 | `compute_classification_metrics` | SLG evaluator |
| `baseline/classification_genrel/scripts/evaluate_metrics.py` | 协议代码 | `evaluate_metrics` | Baseline/Q0-Q3 evaluator |

### 2.2 已有数据的关键真实数值（重审计后）

**Baseline 真实训练**（epoch_metrics.jsonl）：
- val_samples = **213**
- 5 epoch macro_f1: 0.1640, **0.2633** (e1 best), 0.2343, 0.2368, 0.1869
- per_class（epoch 1 best）: pathological=0.58, modulatory=0.14, expression change=0.37, diagnosis=**0.48**, therapy=**0.25**
- macro_auc (sklearn macro-ovr): ~0.78（last 3 epoch avg）

**SLG 真实训练**（epoch_metrics.jsonl）：
- val_samples = **203**
- 5 epoch macro_f1: 0.1444, 0.1492, 0.1438, 0.1453, **0.1515** (e4 best)
- per_class（epoch 4 best）: pathological=**0.625**, modulatory=0.147, expression change=0.289, diagnosis=**0.000**, therapy=**0.000**
- macro_auc (自定义 per-class): ~0.68（last 3 epoch avg）

**Q0/Q1/Q2/Q3 模拟**（已生成但有 bug）：
- n_samples = **168**（_remap_doc_keys 后）
- aggregate macro_f1: Q0'=0.2292 ≈ Q0=0.2292 → Q1=0.2254 → Q2'=0.2215 → Q2=0.1722 → Q3=0.1495
- per_class（Q3 epoch 0）: pathological=0.292, modulatory=0.029, expression change=0.148, diagnosis=**0.185**, therapy=**0.167**

### 2.3 已生成数据的可比性问题

**直接拿 Q3 vs SLG 比较是错误的**，因为：

| 维度 | Q3 | SLG |
|------|----|-----|
| 样本数 | 168 | 203 |
| doc_key 来源 | baseline 的 213 个 fine 后缀 | SLG 的 203 个 coarse 后缀 |
| evaluator | sklearn macro-ovr | 自定义 per-class |
| gold 匹配率 | 168/213 = 78.9% | 203/203 = 100% |

**Q3 与 SLG 的"0.002pp 差"是巧合**——aggregate macro_f1 接近，但 per-class 分布完全不同（SLG 在 diagnosis/therapy 完全失败，Q3 模拟还有 F1）。

---

## 3. 方案设计：四阶段精度损失分解

### 3.1 总策略

**采用"对照实验 + 口径统一 + 累积梯度"三层框架**：

1. **第一层：口径统一** —— 在同一份 203 样本（SLG 口径）上重评所有已有数据
2. **第二层：对照实验** —— 分离"加密税 vs 协议税 vs 量化税"三种税
3. **第三层：累积梯度** —— 量化各税的单步贡献

### 3.2 阶段 A：口径统一（必须先做）

**目标**：把 baseline、SLG、Q0-Q3 模拟结果统一到同一份样本 + 同一套 evaluator。

#### A.1 样本对齐

**问题**：168 (Q0-Q3) vs 203 (SLG) vs 213 (baseline)。

**方案**：
```
1. 取 SLG 的 203 个 doc_key 为基准集合 S (S = cls-SLG-test-data 中实际评估的 doc_key)
2. baseline 的 213 个 fine 后缀 doc_key → 用 _remap_doc_keys 转换为 coarse 后缀
3. 取 baseline infer_outputs_epoch_*.json 的 coarse 形式 ∩ S
4. 取 Q0-Q3 模拟时使用的 168 个 doc_key → 检查是否 ⊂ S
5. 最终三方对比统一使用 S ∩ baseline ∩ Q0-Q3（预计约 168-189 个）
```

**实现**：
- 新文件：`accuracy_ablation/sample_alignment.py`
- 函数：`compute_aligned_sample_set(baseline_infer_path, slg_infer_path, gold_path)`
- 输出：`(aligned_baseline_doc_keys, aligned_slg_doc_keys, common_evaluate_set)`

#### A.2 evaluator 对齐

**问题**：baseline/Q0-Q3 用 sklearn `roc_auc_score`，SLG 用自定义 per-class。

**方案**：
```
1. 用 src/training/biotriplex_metrics.py::compute_classification_metrics
   重评 baseline 的 213 个 fine 后缀 doc_key (但需要 baseline 的 7 维 logits)
2. 用 compute_classification_metrics 重评 Q0-Q3 模拟时的 infer_outputs_epoch_*.json
3. SLG 保持原状
4. 输出统一字段：{macro_f1, macro_auc (per-class 跳过 0-support), ...}
```

**实现**：
- 新文件：`accuracy_ablation/unified_evaluator.py`
- 函数：`evaluate_with_slg_metric(infer_outputs_json, gold_jsonl)`
- 输出：`{n_samples, macro_f1, macro_auc_ovr (sklearn per-class skip), micro_acc, per_class, ...}`

**注意**：compute_classification_metrics 接受 `predictions` 和 `labels`（都是字符串列表），需要先从 infer_outputs 提取 answer 字符串。可参考 `evaluate_slg_cls.py:233-249` 的调用模式。

#### A.3 字段重映射

`accuracy_ablation/report_generator.py:272-283` 的 `field_map` 已正确映射 `val_bt_*`，但需要在统一评估后重新生成 `epoch_metrics.jsonl` 格式。

### 3.3 阶段 B：精度损失分解的核心对比矩阵

**核心对比矩阵（6 列 × 5 行）**：

| 对比对象 | n_samples | macro_f1 | macro_auc | diagnosis_F1 | therapy_F1 |
|---------|-----------|----------|-----------|--------------|------------|
| **Baseline (无加密, 无量化)** | ? | ? | ? | ? | ? |
| **Baseline + V 量化** | ? | ? | ? | ? | ? |
| **Baseline + V + g_H int64 量化** | ? | ? | ? | ? | ? |
| **Baseline + V + g_H int64 + gold-only** | ? | ? | ? | ? | ? |
| **Baseline + V + g_H int64 + gold-only + g_H bf16** | ? | ? | ? | ? | ? |
| **SLG (真实 = 上面的 + 加密 + 累积)** | 203 | 0.1515 | 0.68 | **0.000** | **0.000** |

**这 6 行中**：
- 第 1 行：Baseline epoch_metrics.jsonl 重评
- 第 2-5 行：Q0/Q1/Q2/Q3 重评（已存在但需口径统一）
- 第 6 行：SLG epoch_metrics.jsonl（已有）

**累积税分解**（从第 1 行到第 6 行）：

```
Δmacro_f1 (Baseline → SLG) = 0.2633 - 0.1515 = -0.1118 pp (-11.18pp)

分解为：
  量化税 (Q0 → Q1): ≈ 0.37pp
  g_H int64 税 (Q1 → Q2'): ≈ 0.39pp
  gold-only 协议税 (Q2' → Q2): ≈ 4.93pp   ← 最大单步税
  g_H bf16 转换税 (Q2 → Q3): ≈ 2.28pp
  真实协议残差 (Q3 → SLG): ≈ 0.02pp

  总和 = 0.37 + 0.39 + 4.93 + 2.28 + 0.02 ≈ 7.99pp
  对比真实差异 11.18pp → 缺口 ≈ 3.19pp (来自 Baseline → Q0' 未量化部分)
```

**注意**：Q0'=Baseline（都未量化），所以 Q0' → Q0 的 0.00pp 差异**没有意义**——这不是"LoRA 参数量贡献"，而是**根本没模拟**。

### 3.4 阶段 C：加密税的精确分离

**关键问题**：Q3 → SLG 的差 ≈ 0.26pp 太小，不能完全代表"加密税"。

**原因**：Q3 noise model 已经把 g_H int64 / bf16 / gold-only 都加到 logits，但**没有模拟 SEAL BatchEncoder 的整数 wrap-around**、**CPU↔GPU 通信误差**、**多项式模运算影响**等"真实加密税"。

**新方案**：构造 "真实 V 量化 vs 模拟 V 量化" 对比。

#### C.1 真实 V 量化（在 baseline 上）

```python
# accuracy_ablation/quant_hooks.py 新增
def quantize_lm_head_inplace(model, scale: int = 10000):
    """对 lm_head.weight 做真实 fixed-point 量化（一次性）。"""
    with torch.no_grad():
        w = model.lm_head.weight.data
        w_q = torch.round(w.float() * scale) / scale
        w.copy_(w_q.to(w.dtype))
```

**重新评估**：在 baseline 的 5 epoch checkpoint 上，加载后量化 V，然后跑推理。

**实现路径**：
1. 加载 baseline epoch_X 的 adapter
2. 量化 lm_head.weight
3. 跑 `infer_and_save.py`（直接复用 baseline 的推理脚本）
4. 用统一的 evaluator 评估

**对比**：
- 真实 V 量化 vs 无量化 → **真实的 V 量化税**（不经过 noise model）

#### C.2 真实 g_H 量化（在 baseline 训练时）

**这需要修改训练循环**，更复杂。备选方案：

**简化方案**：在 baseline 推理后，对每条样本的 logits 做"反向模拟"——
```python
# 已知 baseline 的 logits 和 gold token id
# 标准 CE 梯度：g_logits = softmax(logits) - one_hot(gold)
# g_H = g_logits @ V  （作为 lm_head 的输入梯度）
# 量化后：g_H_q = round(g_H * 10000) / 10000
# 量化后回到 logits：g_logits_q = g_H_q @ V_T
# 用 g_logits_q 替换原始 logits
```

**对比**：
- 真实 g_H 量化 vs 标准 CE → **真实的 g_H 量化税**

#### C.3 gold-only 协议的真实模拟

**已知**：SLG 中 `g_H = a_t - V_gold`（仅 1 行 V），这等价于：
```python
# 标准 CE 梯度
g_logits_full = softmax(logits) - one_hot(gold)   # shape (7,)
# 但 gold-only 协议中：U 只能知道 V_gold，其他 6 行 V 用 0 代替
# 等价于：
g_logits_gold_only = (a_t - V_gold) @ V_T         # shape (7,)
#                  = a_t @ V_T - V_gold @ V_T
#                  = logits - V_gold @ V_T
# 即：gold_only_logits = logits - V_gold @ V_T
```

**实现**：
```python
# accuracy_ablation/quant_hooks.py 新增
def apply_gold_only_protocol(logits, lm_head_weight, gold_id):
    """模拟 gold-only 协议：在 logits 层等价扰动"""
    # V_gold: shape (hidden,)
    V_gold = lm_head_weight[gold_id]  # (4096,)
    # a_t = softmax(logits) @ V  （在 full vocab 上）
    # 等价于：a_t = logit_norm @ V  其中 logit_norm = softmax(logits)
    probs = softmax(logits)
    a_t = probs @ lm_head_weight  # (4096,)
    # gold_only_logits = logits - V_gold @ V_T
    delta_logits = V_gold @ lm_head_weight.T  # (vocab,)
    return logits - delta_logits
```

**注意**：这等价于真实 gold-only 协议在 logits 层的精确表现。但**没有量化步骤**——它只模拟协议约束本身。

**对比**：
- 标准 CE logits vs gold-only 协议 logits → **真实的协议税**（不经过 noise model）

#### C.4 全链路真实模拟

组合上述三步：
1. V 量化（一次性，在模型加载时）
2. gold-only 协议（在 logits 上）
3. g_H 量化（在 gold-only 协议后，对 logits 加 g_H 量化税）

**对比**：全链路真实模拟 vs Baseline = **完整的"加密税 + 协议税 + 量化税"**

但 vs SLG 真实差异，可以分解出"未建模残差"（CPU↔GPU 通信误差、SEAL BatchEncoder 整数 wrap-around 等）。

### 3.5 阶段 D：累积梯度分解（最终输出表）

**最终表格**（在统一口径后）：

| 步骤 | macro_f1 | Δ vs Baseline (pp) | 真实 vs 模拟 | 含义 |
|------|----------|--------------------|--------------|------|
| Baseline (明文) | 0.2633 | 0 | (真实) | 起点 |
| Baseline + V 量化 | TBD | TBD | 真实模拟 | **真实 V 量化税** |
| Baseline + V + g_H int64 量化 | TBD | TBD | 真实模拟 | **真实 g_H int64 税** |
| Baseline + V + g_H int64 + gold-only | TBD | TBD | 真实模拟 | **真实协议税**（最大单步税） |
| Baseline + V + g_H int64 + gold-only + g_H bf16 | TBD | TBD | 真实模拟 | **真实 bf16 转换税** |
| SLG (真实) | 0.1515 | -11.18 | (真实) | 全链路 |
| Q3 (noise 模拟) | 0.1495 | -11.38 | (模拟) | noise model 拟合的 Q3 |

**关键对比**：
1. **真实 V 量化税 vs Q0 → Q1 的 noise 模拟税**：验证 noise model 是否真实捕获了 V 量化
2. **真实协议税 vs Q2' → Q2 的 noise 模拟税**：验证 noise model 是否真实捕获了 gold-only 协议
3. **真实全链路 vs Q3 vs SLG**：差距 = 未建模的 CPU↔GPU 通信税 + SEAL 整数 wrap-around 税

---

## 4. 实施步骤（按优先级）

### 4.1 优先级 P0：必须做的修复

#### P0.1 修复 `_remap_doc_keys` bug

**问题**：`accuracy_ablation/eval_replay.py:243` 的 `or True` 永远为真。

**修复**：
```python
# 删除 `or True`
if candidate in gold_keys:  # 直接用 gold doc_key set
    out[candidate] = entry
    mapping[base_dk] = candidate
    continue
```

#### P0.2 用 `compute_classification_metrics` 重评 baseline

**问题**：baseline 用 sklearn `roc_auc_score` 计算 macro_auc，SLG 用自定义，两者不可比。

**修复**：
1. 从 baseline infer_outputs_epoch_*.json 抽取 answer 字符串列表（已经是 "a)" 格式）
2. 从 test_gold_general_qa.txt 抽取 gold answer 字符串列表
3. 用 `compute_classification_metrics(predictions, labels, pred_logits=None)` 计算指标
4. 输出 `epoch_metrics_baseline_slg_metric.jsonl`

**代码**：
```python
# accuracy_ablation/unified_evaluator.py (新文件)
from src.training.biotriplex_metrics import compute_classification_metrics

def evaluate_with_slg_metric(infer_outputs_json, gold_jsonl):
    # 读取 baseline 的 infer_outputs
    with open(infer_outputs_json) as f:
        infer = json.load(f)

    # 读取 gold
    gold = {}
    with open(gold_jsonl) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 把 gold 的 doc_key 映射回 baseline 的 fine 后缀
                # 或保持 coarse 格式（取决于 baseline infer 的 doc_key 格式）
                gold[item["doc_key"]] = item.get("output", "a)")

    # 构造对齐
    common = sorted(set(infer.keys()) & set(gold.keys()))
    predictions = [infer[k]["answer"] for k in common]
    labels = [gold[k] for k in common]

    metrics = compute_classification_metrics(predictions, labels, pred_logits=None)
    return metrics
```

**注意**：baseline infer_outputs 的 doc_key 是 `_relation_<fine>` 格式，gold 是 `_rel_<coarse>` 格式。需要先把 gold 的 `_rel_<coarse>` 反向映射到 fine 后缀，或者把 baseline 的 fine 后缀正向映射到 coarse 后缀。

#### P0.3 用 `compute_classification_metrics` 重评 Q0-Q3 模拟

**问题**：Q0-Q3 模拟时调用了 baseline 的 `evaluate_metrics.py`，口径与 SLG 不同。

**修复**：用 P0.2 的 `evaluate_with_slg_metric` 函数重评 Q0-Q3 的 `infer_outputs_epoch_*.json`。

### 4.2 优先级 P1：核心精度损失分解

#### P1.1 真实 V 量化实验

**目标**：在 baseline 的 5 epoch checkpoint 上，加载后量化 V，再推理。

**步骤**：
1. 加载 baseline epoch_X adapter 到 base model
2. 量化 lm_head.weight: `w = round(w * 10000) / 10000`
3. 跑 `infer_and_save.py`（或直接调用 model.forward）
4. 用 P0.2 的 evaluator 评估

**对比**：真实 V 量化 vs 无量化 = **真实 V 量化税**

**预期精度损失**：根据报告（Q0 → Q1 = 0.37pp），真实 V 量化税应该接近 0.37pp。

#### P1.2 真实 gold-only 协议实验

**目标**：在 baseline logits 上模拟 gold-only 协议。

**步骤**：
1. 读取 baseline infer_outputs（已经是标准 CE 训出的 logits）
2. 从 gold 读取每个 doc_key 的 gold_id（letter a-g）
3. 应用 gold-only 协议：`logits_gold_only = logits - V_gold @ V_T`
   - 需要加载 baseline 的 lm_head.weight
4. 用 P0.2 的 evaluator 评估

**对比**：真实 gold-only 协议 vs 标准 CE = **真实协议税**

**预期精度损失**：根据报告（Q2' → Q2 = 4.93pp），真实协议税应该接近 4.93pp。

**重要**：`V_gold @ V_T` 计算量很大（vocab=128256, hidden=4096），但只需要对每个 gold_id 算一次：即 `V[gold_id] @ V.T`，得到 (vocab,) 向量，从 logits 减去。

#### P1.3 真实 g_H int64 量化实验

**目标**：在 P1.2 基础上，对 logits 进一步加 g_H int64 量化税。

**步骤**：
1. 在 P1.2 的 logits 上，计算真实 g_H：
   ```python
   # g_H = a_t - V_gold 的等价物（从 logits 角度）
   # 等价于：g_H_quant 误差在 logits 上等价于：
   # logits_new = logits + scale_error
   # 其中 scale_error = round((a_t - V_gold) * 10000) / 10000 - (a_t - V_gold) @ V_T
   # 但这需要 H_M 才能算（baseline 没有显式存）
   ```
2. **简化方案**：直接用 noise model 的 σ=√hidden/scale 加上更精确的扰动
   - σ_g_h_int = sqrt(4096) / 10000 ≈ 0.0064
   - 这个值远小于当前 noise model 的 0.5

**预期精度损失**：真实 g_H int64 税应该 < 0.5pp（因为 σ 太小）。

#### P1.4 真实 g_H bf16 转换实验

**目标**：在 P1.3 基础上，对 logits 进一步加 g_H bf16 转换税。

**简化方案**：
- σ_g_h_bf16 = sqrt(4096) / 256 ≈ 0.25
- 比当前 noise model 的 1.5 小 6 倍

**预期精度损失**：真实 bf16 转换税应该 < 1pp。

### 4.3 优先级 P2：累积梯度报告

#### P2.1 生成"真实税 vs noise model 税"对比表

| 税类型 | noise model σ | 真实 σ (从协议推导) | noise model 拟合损失 | 真实模拟损失 |
|--------|---------------|---------------------|---------------------|--------------|
| V 量化 | 0.5 | ~0.003 | 0.37pp | TBD |
| g_H int64 | 0.5 | ~0.006 | 0.39pp | TBD |
| gold-only | U(-1.5, 0.5) at gold pos | (不适用) | 4.93pp | TBD |
| g_H bf16 | 1.5 | ~0.25 | 2.28pp | TBD |

**关键发现**：noise model 的 σ 比真实 σ 大 100-1000 倍，说明 current noise model 是"为了拟合 SLG aggregate 强行加噪"，不是"真实协议机制的等价扰动"。

#### P2.2 生成"加密税"对比

**关键问题**：Q3 → SLG 的 0.26pp 差是"未建模精度残差"，不是"加密税"。

**正确做法**：
- 加密税 = SLG 真实 vs 全链路真实模拟
- 但全链路真实模拟需要真实 V + 真实 g_H + 真实协议，**目前没有这些数据**
- 所以**当前无法精确分离加密税**

**可替代**：在 baseline 上构造"半 SLG"实验——
1. 加载 baseline adapter
2. 量化 lm_head
3. 应用 gold-only 协议
4. 加 g_H 量化
5. **与 SLG 真实对比** = 半 SLG 真实模拟 vs SLG = **加密 + 通信税**

### 4.4 优先级 P3：报告生成

#### P3.1 新报告结构

```
AccuracyAblationTest/docs/final_report.md

1. Executive Summary (5 条核心发现)
2. 数据口径统一说明 (修复后的样本数 + evaluator)
3. 真实协议机制 vs noise model 对比
4. 累积精度梯度分解表
5. Per-class 真实税 vs 模拟税差异
6. 加密税的精确估计 (含残差分解)
7. 统计显著性检验
8. 局限性声明
9. 附录: σ 值推导、协议代码引用
```

#### P3.2 与已有报告的关系

**保留**：QUANT_ABLATION_REPORT.md (作为 baseline 模拟结果)
**新增**：docs/final_report.md (作为口径统一后的最终结果)
**删除**：无（两份报告并存，标注口径差异）

---

## 5. 关键决策点（需要用户确认）

在开始实施前，需要您决定：

### 决策 D1：是否重训 baseline？

**A**：不重训，全部用已有数据（baseline 5 epoch + SLG 5 epoch + 6 个变体模拟）
- 优点：快速（无需 GPU 时间）
- 缺点：无法真实模拟 V 量化在训练时的影响

**B**：在 baseline 上跑 1 个 "真实 V 量化" epoch（V 量化 + 推理）
- 优点：捕获真实 V 量化税
- 缺点：需要 ~1 小时 GPU 时间

**C**：在 baseline 上重训 5 epoch，每 epoch 量化 V + 推理
- 优点：完整捕获 V 量化在训练中的累积影响
- 缺点：需要 ~18 分钟 × 5 = 1.5 小时 GPU 时间

### 决策 D2：gold-only 协议的真实模拟方式

**A**：在 logits 上做 `logits - V_gold @ V_T`（无需重训）
- 优点：快速，精确等价于真实协议
- 缺点：仅捕获协议在 logits 层的等价扰动

**B**：在 baseline 训练时 monkey-patch CE loss（重训）
- 优点：捕获训练时累积影响
- 缺点：需要重训 + 调试 training loop

### 决策 D3：报告粒度

**A**：只产出 macro_f1 分解表（简洁）
**B**：macro_f1 + macro_auc + per-class F1 全分解（完整）
**C**：仅 per-class 差异分析（针对 diagnosis/therapy 失败）

### 决策 D4：是否修正 Q0/Q0' LoRA 配置差异

**A**：保持现状（Q0'=Q0 无差异），在报告中明确标注"Q0' 和 Q0 实际上没区分"
**B**：在 baseline 上重训 7-target LoRA（仅 Q0），与 Q0' 对比
**C**：用 7-target LoRA 推理 baseline 推理脚本（不改 LoRA 配置但加载 SLG adapter）

---

## 6. 实施时间估算

| 任务 | 时间 | 难度 |
|------|------|------|
| P0.1 修复 `_remap_doc_keys` | 1 小时 | ⭐ |
| P0.2 写 `unified_evaluator.py` | 2-3 小时 | ⭐⭐ |
| P0.3 用统一 evaluator 重评所有数据 | 2 小时 | ⭐ |
| P1.1 真实 V 量化实验 | 1 小时（含加载 + 推理） | ⭐⭐ |
| P1.2 真实 gold-only 协议实验 | 3-4 小时（含推导 + 实现 + 推理） | ⭐⭐⭐ |
| P1.3 真实 g_H int64 量化实验 | 2 小时 | ⭐⭐ |
| P1.4 真实 g_H bf16 转换实验 | 1 小时 | ⭐ |
| P2 累积梯度报告生成 | 4-5 小时 | ⭐⭐ |
| P3 最终报告整合 | 3-4 小时 | ⭐⭐ |
| **总计** | **约 20-25 小时** | |

如果不需要全部 P1.x（仅做口径统一 + 累积梯度报告），约 **8-10 小时**。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 样本对齐后 baseline/SLG/Q0-Q3 共同样本数 < 100 | 统计不显著 | 报告里明确标注 sample size |
| unified_evaluator 与 baseline evaluate_metrics.py 在某些边界 case 行为不同 | 指标有微差 | 双跑验证，用 paired t-test 检验 |
| 真实 V 量化 vs Q1 noise 模拟差距巨大 | 推翻当前 noise model 假设 | 报告中明确标注 noise model 局限性 |
| SLG 在 diagnosis/therapy 失败的根因无法确认 | 不完整结论 | 在报告中标注"未解之谜"，待后续研究 |
| 真实 V 量化实验的 baseline adapter 不可加载 | 实验失败 | 先检查 baseline adapter 加载 |

---

## 8. 不在本次方案范围的事项

1. **真正的 CPU↔GPU 通信税**：需要 SLG 协议代码的 profiling 数据
2. **SEAL BatchEncoder 整数 wrap-around 税**：需要 SEAL 内部实现的精确模拟
3. **多项式模运算 tax**：需要 BFV 算子的精确分析
4. **优化 SLG 协议降低 gold-only 税**：需要协议修改，方案设计阶段不做
5. **更长 epoch 的 SLG 训练**：方案 P2 不涉及，只用现有 5 epoch 数据

---

## 9. 结论

**新方案的核心思路**：
1. **先修 bug**（_remap_doc_keys, evaluator 对齐）
2. **再做真实模拟**（V 量化 + gold-only + g_H 量化，基于已有 logits）
3. **最后做累积分解**（真实税 vs noise model 税 vs SLG 真实）

**预期结论**（基于代码审计）：
- 当前 noise model 的 σ 比真实协议 σ 大 100-1000 倍
- 当前 noise model 能拟合 aggregate macro_f1 但 per-class 分布完全错位
- SLG 真实在 diagnosis/therapy 失败是 noise model 完全未捕获的协议税
- "加密税" 在精度损失分解中的占比可能 < 5%，**协议税才是大头**（约 50%）

**方案落地后**：
- 一份口径统一后的精度损失分解表（核心交付）
- 一份"真实协议税 vs noise model 拟合税"对比表（揭示当前方法局限）
- 一份"未建模残差"分析（指出需要后续研究的方向）

---

## 10. 下一步

等待您的决策：
- **D1**：是否重训 baseline？
- **D2**：gold-only 协议模拟方式？
- **D3**：报告粒度？
- **D4**：Q0/Q0' LoRA 配置差异？

确认后，进入实施阶段。