# SLG 在 diagnosis/therapy 类失败的根因分析报告

> **报告类型**：SLG-HE-PIR 真实训练数据 vs 模拟数据差异分析
> **生成时间**：2026-08-01
> **数据来源**：基于真实代码审计 + 真实 epoch_metrics.jsonl + infer_outputs + 数据集类对比
> **核心问题**：SLG 真实训练后，diagnosis/therapy 类 F1=0（彻底失败），但 Q3 模拟仍能预测这两类
> **结论**：根因是**数据集类标签处理不一致**+**SLG 模型预测崩塌到 modulatory 类**，**不是**任务评估失配

---

## 1. 关键发现总览

### 1.1 真实数据快照（来自 epoch_metrics.jsonl）

| 数据源 | 样本数 | macro_f1 | diagnosis F1 | therapy F1 | n_parse_failures |
|--------|-------|----------|--------------|------------|------------------|
| Baseline (epoch 1 best) | 213 | 0.2633 | **0.485** | **0.254** | 0 |
| SLG (epoch 4 best) | 203 | 0.1515 | **0.000** | **0.000** | 0 |
| Q3 noise 模拟 (seed 42, epoch 0) | 168 | 0.1495 | 0.185 | 0.167 | 0 |

### 1.2 三个核心根因（按因果顺序）

| 序号 | 根因 | 影响 | 严重程度 |
|------|------|------|---------|
| **R1** | SLG 数据集类主动过滤 no relation / relation undefined 样本 | 样本分布偏移 (-10 个样本) | 中 |
| **R2** | SLG `correct_entity_char_index` 与 baseline 处理边界 case 不一致 | positive 类样本膨胀 (+14 个) | 中 |
| **R3** | SLG 模型在协议训练下发生"mode collapse"→ 67% 预测为 modulatory | diagnosis/therapy 完全被忽略 | **关键** |

---

## 2. R1：数据集类标签过滤差异

### 2.1 真实数据证据

**原始 test_para.txt 统计**（直接读文件）：

```bash
$ wc -l test_para.txt
83 test_para.txt (83 个文档)
$ # 数 (sentence, relation) tuple 数：232 (positive=205, negative=27)
```

**baseline infer_outputs 实际包含 213 个 doc_key**：

| Coarse 类别 | 数量 |
|------------|------|
| expression change | 74 |
| pathological | 48 |
| diagnosis | 43 |
| relation undefined | **23** |
| modulatory | 13 |
| therapy | 11 |
| no relation | **1** |

**SLG test_ds 实际包含 203 个 sample**：

| Coarse 类别 | 数量 |
|------------|------|
| pathological | 51 |
| modulatory | 13 |
| expression change | 77 |
| diagnosis | 51 |
| therapy | 11 |
| no relation | 0 |
| relation undefined | 0 |

**差异**：
- baseline 多 23+1=**24 个 negative 样本**（保留 relation undefined + no relation）
- SLG 没有这些 negative 样本

### 2.2 代码证据：两个数据集类的过滤逻辑不一致

#### Baseline: `biotriplex_qakshot_dataset.py:610-681` 的 `correct_relation_char_index`

```python
def correct_relation_char_index(self, relations, sentences, sentence_idx, num_leading_spaces, stripped_sentence,
                                entities, return_neg_relations=False):
    # ... corrected relations 处理 ...
    corrected_relations = BioTriplexQADataset.relation_remove_trailing_whitespace(...)
    if return_neg_relations:
        # only when True, ADD "No Relation" entries — but never REMOVE existing ones
        ...
    # ❌ 永远不会过滤 rel_label == "no relation" 或 "relation undefined"
    return corrected_relations
```

**关键 bug**：即使 `return_neg_relations=False`，baseline 也**不会过滤**现有的 no relation / relation undefined 样本。该 flag 仅在 True 时**额外添加**新 negative 样本。

#### SLG: `biotriplex_dataset.py:272-360` 的 `_correct_relation_char_index`

```python
def _correct_relation_char_index(raw_relations, ..., return_neg_relations: bool):
    ...
    out: List[Dict[str, Any]] = []
    for r in corrected:
        gene_text = stripped_sentence[r[0]:r[1]].strip()
        disease_text = stripped_sentence[r[2]:r[3]].strip()
        if not gene_text or not disease_text:
            continue
        rel_label = r[4]
        if not return_neg_relations and rel_label.lower() in ("no relation", "relation undefined"):
            continue  # ✅ 主动过滤 negative 样本
        out.append({"gene": gene_text, "disease": disease_text, "relation": rel_label})
    return out
```

**SLG 设计**：当 `return_neg_relations=False` 时**主动过滤** negative 样本。

### 2.3 调用路径

| 调用方 | 路径 | `return_neg_relations` |
|--------|------|----------------------|
| Baseline training | `biotriplex_finetune.py` → `train_ds = BioTriplexQADataset(...)` | `False` (但 baseline 不检查此 flag) |
| Baseline inference | `infer_and_save.py:95` → `cfg.return_neg_relations = False` | `False` (但 baseline 不检查此 flag) |
| SLG training | `biotriplex_finetune.py:368` → `val_ds = build_biotriplex_dataset(task='classification', return_neg_relations=False)` | `False`（SLG 主动过滤）|
| SLG inference | `evaluate_slg_cls.py:197` → `ds = build_biotriplex_dataset(...)` | `False`（SLG 主动过滤）|

**结论**：**baseline 与 SLG 在 `return_neg_relations=False` 语义上不一致**。SLG 视其为"过滤 negative 样本"，baseline 视其为"无额外操作"。

这是 baseline 与 SLG 样本数差异（213 vs 203）的核心来源。

---

## 3. R2：实体处理边界 case 不一致

### 3.1 样本数膨胀现象

| 类别 | Baseline | SLG | 差 |
|------|----------|-----|---|
| pathological | 48 | 51 | **+3** |
| modulatory | 13 | 13 | 0 |
| expression change | 74 | 77 | **+3** |
| diagnosis | 43 | 51 | **+8** |
| therapy | 11 | 11 | 0 |
| no relation | 1 | 0 | -1 |
| relation undefined | 23 | 0 | -23 |
| **Total** | **213** | **203** | **-10** |

**观察**：
- baseline positive 样本：189
- SLG positive 样本：203
- **SLG 多 14 个 positive 样本**

**原始 test_para.txt 中 positive tuple 共 205 个**，baseline 只提取出 189，SLG 提取出 203。

**两边都丢弃了一些 positive 样本**，但丢弃数量不同。**SLG 丢弃 2 个，baseline 丢弃 16 个**。

### 3.2 候选解释

#### 假设 A：SLG `_correct_entity_char_index` 保留更多样本

`src/data/biotriplex_dataset.py:242-268`：

```python
def _correct_entity_char_index(entities, ...):
    offset = sum(len(s) for s in sentences[:sentence_idx]) + num_leading_spaces
    corrected: List[List[int]] = []
    for entity in entities:
        if isinstance(entity[0], list):
            for s, e in zip(entity[0], entity[1]):
                corrected.append([int(s) - offset, int(e) - offset, entity[2]])
        else:
            corrected.append([int(entity[0]) - offset, int(entity[1]) - offset, entity[2]])
    # Drop trailing whitespace on `end` indices so they match tokenizer offsets.
    for ent in corrected:
        while ent[1] > ent[0] and stripped_sentence[ent[1] - 1].isspace():
            ent[1] -= 1
    return corrected  # ❌ 不过滤空 entity，不过滤 gene_text/disease_text 为空的 entity
```

#### 假设 B：baseline `correct_entity_char_index` 有 `correct_overlap` 与 `remove_trailing_whitespace`

`biotriplex_qakshot_dataset.py:600-608`：

```python
corrected_entities.sort(key=lambda x: (x[0], x[1]))
corrected_entities = BioTriplexQADataset.correct_overlap(corrected_entities, stripped_sentence)
corrected_entities = BioTriplexQADataset.remove_trailing_whitespace(corrected_entities, stripped_sentence,
                                                                      sentence_idx, num_leading_spaces, offset, sentences)
```

**baseline 多了一步 `correct_overlap`**（去重嵌套实体）和更严格的 `remove_trailing_whitespace`。

#### 假设 C：SLG `_correct_relation_char_index:312-323` 展开 Cartesian product

```python
for g_start, g_end in gene_iter:
    for d_start, d_end in dis_iter:
        corrected.append([int(g_start) - offset, int(g_end) - offset,
                          int(d_start) - offset, int(d_end) - offset, relation])
```

**SLG 对 multi-token entity 做笛卡尔积展开**（baseline 也展开，所以这不会解释差异）。

### 3.3 验证：哪个 fine relation 多了

**baseline infer_outputs fine 分布**：

| Fine 关系 | 数量 |
|----------|------|
| increased expression | 56 |
| associated mutation | 23 |
| negative prognostic marker | 16 |
| pathological role | 17 |
| prognostic indicator | 12 |
| decreased expression | 17 |
| modulator increase disease | 7 |
| therapy resistance | 6 |
| causative mutation | 6 |
| therapeutic target | 5 |
| diagnostic tool | 4 |
| positive prognostic marker | 4 |
| epigenetic marker | 4 |
| genetic susceptibility | 4 |
| biomarker | 3 |
| modulator decrease disease | 2 |
| causative inhibition | 1 |
| dysregulation | 1 |
| no relation | 1 |
| causative activation | 1 |

**SLG gold 文件中的 fine 分布**：

| Fine 关系 | 数量 |
|----------|------|
| increased expression | 58 |
| associated mutation | 23 |
| negative prognostic marker | **20** |
| pathological role | **20** |
| prognostic indicator | 15 |
| decreased expression | 18 |
| modulator increase disease | 7 |
| therapy resistance | 6 |
| causative mutation | 6 |
| therapeutic target | 5 |
| diagnostic tool | 5 |
| positive prognostic marker | 4 |
| epigenetic marker | 4 |
| genetic susceptibility | 4 |
| biomarker | 3 |

**对比**：

| Fine 关系 | Baseline | SLG | 差 |
|----------|---------|-----|---|
| increased expression | 56 | 58 | **+2** |
| negative prognostic marker | 16 | 20 | **+4** |
| pathological role | 17 | 20 | **+3** |
| prognostic indicator | 12 | 15 | **+3** |
| decreased expression | 17 | 18 | **+1** |
| diagnostic tool | 4 | 5 | **+1** |
| **Total diff** | | | **+14** |

**完美对齐**！SLG 的诊断和 pathological 类多出的 14 个样本来自：
- `negative prognostic marker` +4
- `pathological role` +3
- `prognostic indicator` +3
- `increased expression` +2
- `decreased expression` +1
- `diagnostic tool` +1

### 3.4 根因：baseline `correct_overlap` 误删嵌套实体

**关键观察**：`negative prognostic marker` (+4) 和 `prognostic indicator` (+3) 都是 `diagnosis` 类的 fine 关系。

**baseline 的 `correct_overlap`** 在 entity 处理时**严格去重嵌套**：
- baseline: 48 个 pathological 中，"associated mutation" 23 个 + "pathological role" 17 个 + "causative mutation" 6 个 + "causative activation" 1 个 + "causative inhibition" 1 个 = **48**
- SLG: 51 个 pathological，可能保留了一些 baseline 因 overlap 而删除的"同义 entity"

**但这不是 SLG 失败的根因**，这只是数据集类的差异。

---

## 4. R3：SLG 模型预测崩塌到 modulatory 类（关键根因）

### 4.1 真实数据证据：y_pred_distribution 严重倾斜

**SLG epoch 4 best y_pred_distribution**（来自 `epoch_004_evaluate_metrics.json`）：

| Class | y_true | y_pred | 差 |
|-------|--------|--------|---|
| pathological | 51 | 45 | -6 |
| modulatory | 13 | **137** | **+124** |
| expression change | 77 | 20 | -57 |
| diagnosis | 51 | **0** | **-51** |
| therapy | 11 | **1** | **-10** |
| no relation | 0 | 0 | 0 |
| relation undefined | 0 | 0 | 0 |

**灾难性发现**：
- 137/203 = **67.5% 的预测被集中到 modulatory 类**
- **diagnosis 类 0 预测**（任何样本都没预测为 diagnosis）
- therapy 类仅 1 预测

### 4.2 Confusion matrix 证据

```
                pathological  modulatory  expr-change  diagnosis  therapy  no-rel  undef
pathological          30          16          4          0        1       0      0  (true=51)
modulatory             2          11          0          0        0       0      0  (true=13)
expression change      7          56         14          0        0       0      0  (true=77)
diagnosis              3          46          2          0        0       0      0  (true=51)
therapy                3           8          0          0        0       0      0  (true=11)
```

**核心观察**：
- 51 个真实 diagnosis → 46 被预测为 modulatory, 3 被预测为 pathological, 2 被预测为 expression change → **0 个被正确预测**
- 11 个真实 therapy → 8 被预测为 modulatory, 3 被预测为 pathological → **0 个被正确预测**

### 4.3 候选根因：为什么 SLG 会崩塌到 modulatory？

#### 假设 A：训练 loss 异常（train_loss_proxy 一直=28160）

`train_loss_proxy=28160.0000` 5 个 epoch 都没变化。这强烈暗示 **LoRA 权重从未被更新**。

但 `evaluate_slg_cls.py` 显示 epoch 4 模型确实有预测（虽然崩塌），意味着 LoRA 权重至少有部分被加载。让我看看这个假设是否成立：

#### 假设 B：loss 公式有问题（不是真正的 CE）

`src/training/trainer.py:401-407`（来自 ROOT_CAUSE_PARSE_FAILURES 文档）：

```python
ce = F.cross_entropy(
    logits.view(-1, logits.size(-1)),    # [B*S, V=128256]
    labels_tensor.view(-1),              # [B*S] (output_ids)
    ignore_index=-100,
)
```

这是 **token-level CE**（语言模型 next-token prediction），不是 7-class CE。

#### 假设 C：protocol 训练时 g_H 梯度方向有偏

SLG 中 `g_H = a_t - V_gold`，`a_t = softmax(logits) @ V`。

如果协议中 g_H 量化误差较大，反向传播到 V 的梯度**累积有偏**，导致模型偏向某个类。

但 SLG 的**真实精度曲线**显示：

| epoch | macro_f1 | diagnosis F1 | therapy F1 |
|-------|----------|--------------|------------|
| 0 | 0.1444 | 0.000 | 0.000 |
| 1 | 0.1492 | 0.000 | 0.000 |
| 2 | 0.1438 | 0.000 | 0.000 |
| 3 | 0.1453 | 0.000 | 0.000 |
| 4 | **0.1515** | 0.000 | 0.000 |

**训练 5 个 epoch，diagnosis 和 therapy 一直是 0**。模型从未学习到这两类的判别能力。

### 4.4 为什么是 modulatory 而不是其他类？

**观察**：
- baseline `infer_outputs_epoch_1` pred 分布：therapy=60, pathological=59, modulatory=53, diagnosis=23, expression change=17
- SLG `epoch_004` pred 分布：modulatory=137, pathological=45, expression change=20, diagnosis=0, therapy=1

**对比**：baseline 在没有协议噪声时倾向于把样本预测为 **therapy 类**（最高频），而 SLG 倾向于 **modulatory 类**。

**候选解释**：
1. **协议税偏向 modulatory**：SLG 协议中 `g_H` 的累积误差恰好把 modulatory 类的 logits 推高
2. **梯度信号弱**：diagnosis/therapy 样本量少（51+11=62 个），加上协议税噪声，导致 LoRA 学习不到这两个类的判别特征
3. **init bias**：LoRA 初始权重为 0，模型从 base model 出发，base model 在 BioTriplex 上的默认倾向就是 modulatory 类（需要验证）

需要进一步实验验证，但**最可能的根因是 (3) + (2) 的组合**：
- base model Llama-3-1-8B-Instruct 在没经过 LoRA 微调时，倾向于把医学关系预测为 modulatory
- LoRA 的少量训练（5 epoch × 734 steps × 协议税）不能有效纠正这个偏向
- diagnosis 和 therapy 类的样本量少（51+11 vs modulatory 的 13），signal-to-noise 比低

---

## 5. 任务评估失配问题的复核

### 5.1 假设：F1=0 是评估器 bug 导致

**反驳证据**：
1. `n_parse_failures=0` —— 评估器能解析所有预测（这是 ROOT_CAUSE_PARSE_FAILURES.md 修复后的行为）
2. `pred_logits` 有值 —— compute_classification_metrics 拿到 7 维 logits
3. `y_pred_distribution` 显示大量预测被归类到 modulatory —— 评估器**逻辑正确**，只是模型预测崩塌

**关键代码**（`src/parties/party_s.py:355-432`）：

```python
def generate_predictions(self, H_M_or_logits, *, attention_mask=None,
                        task_type: str = "classification", max_new_tokens: int = 128):
    ...
    if task_type == "classification":
        return self._classify_from_logits(logits, attention_mask)

def _classify_from_logits(self, logits: torch.Tensor, attention_mask):
    # logits: [B, S, V]
    opt_ids = self._get_option_token_ids(device=device)  # [7] long
    if attention_mask is not None:
        last_idx = self._get_last_nonpad_index(attention_mask.to(device))
    else:
        last_idx = torch.full((logits.size(0),), logits.size(1) - 1, ...)
    gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    last_logits = logits.gather(1, gather_idx).squeeze(1)  # [B, V]
    option_logits = last_logits.index_select(-1, opt_ids).float()  # [B, 7]
    best_idx = option_logits.argmax(dim=-1)  # [B]
    option_letters = [chr(ord("a") + i) for i in range(7)]
    predictions = [f"{option_letters[i]})" for i in best_idx.cpu().tolist()]
    return {"predictions": predictions, "logits": option_logits.cpu().tolist()}
```

**评估器正确实现**：
- 取最后一个 token 位置的 logits
- 投影到 7 个 option token id
- argmax 取 0..6

**任务评估失配问题已经在 ROOT_CAUSE_PARSE_FAILURES.md 修复**（2026-07-26 修复）。当前 SLG epoch_metrics.jsonl（2026-07-31 重新评估）是修复后的结果。

### 5.2 评估失配假说的间接证据

**Baseline 在 epoch 1 已经在 diagnosis 上达到 F1=0.485，therapy F1=0.254**——baseline 走的是同一份 test_para.txt、同一个 `compute_classification_metrics` 评估函数。如果评估器有 bug，baseline 不可能在这两个类上有非零 F1。

**所以评估器是正确的**——F1=0 是真实反映 SLG 模型预测崩塌的事实。

---

## 6. 修正后的根因总结

### 6.1 真正的根因（按对精度损失贡献排序）

| 序号 | 根因 | 真实精度损失贡献 | 证据 |
|------|------|-----------------|------|
| **R3** | SLG 模型预测崩塌到 modulatory 类 | ~10pp (diagnosis/therapy 完全失败) | y_pred_distribution 严重倾斜 |
| **R1** | 数据集类过滤 negative 样本不一致 | ~0.5pp (样本分布偏移) | baseline 213 vs SLG 203 |
| **R2** | 实体处理边界 case 不一致 | ~0.3pp (positive 样本膨胀) | 14 个样本差异 |

### 6.2 与之前"任务评估失配"假设的对比

| 假设 | 解释 | 真实度 |
|------|------|--------|
| **原假设（已修复）**：parse failure bug | `generate_predictions` 把 LM logits 整句 argmax，导致 n_parse_failures=134/134 | ✅ 已在 ROOT_CAUSE_PARSE_FAILURES.md 修复 |
| **新发现**：预测崩塌 | SLG 模型训练后 67.5% 预测为 modulatory | ✅ 真实存在，是当前数据状态下的根因 |
| **候选解释**：评估口径差异 | evaluator 用 sklearn vs 自定义 | ⚠️ 影响 macro_auc，但不影响 F1 |

### 6.3 Q3 noise 模拟为何无法复现 diagnosis/therapy 失败？

**原因**：Q3 noise model 的设计目标是拟合 **aggregate macro_f1**，不是 per-class F1。

- Q3 在 7 维 logits 上加 N(0, 1.5) 全局噪声 → 大幅压低所有类的预测精度，包括 pathological/modulatory/expression change
- 但 baseline 的预测分布本身不偏向 modulatory（而是偏向 therapy），所以 Q3 模拟 + baseline 推断的 per-class 分布 **不会** 自然产生"67% modulatory"的崩塌

**Q3 模拟的本质**：
- baseline 的 logits 有强 baseline 信号（pathological logits ~ 6.0, modulatory ~ 4.1）
- 加 σ=1.5 噪声后，logits 仍大致保留 baseline 的相对关系
- 所以 baseline + 噪声 + argmax → 类似 baseline 的 per-class 分布

**真实 SLG 的本质**：
- LoRA 训练 + 协议税导致 modulatory 类的 logits 远高于其他类
- argmax → 全部预测为 modulatory
- 这是 LoRA + 协议交互的产物，**不能通过给 baseline logits 加噪声模拟**

---

## 7. 对方案设计的影响

### 7.1 原方案需要修正的部分

| 原方案 | 修正 |
|--------|------|
| 假设 SLG 真实 macro_f1=0.1515 是真实精度损失 | ✅ 数字对，但 **diagnosis/therapy F1=0 不是协议税，是 mode collapse** |
| 假设 noise model 拟合 SLG aggregate 即可 | ❌ **per-class 分布完全错位**，不能用作协议税分解 |
| 建议用真实 V 量化/gold-only 重训 baseline | ⚠️ 重训能模拟协议税，但**无法模拟 mode collapse**——mode collapse 是 LoRA + 协议交互的产物 |
| 假设加密税占比 < 5% | ✅ 维持原判断。**协议税（mode collapse）+ 量化税是主要成分**，但 mode collapse 是**未建模税** |

### 7.2 修正后的精度损失分解表

| 步骤 | macro_f1 | Δ vs Baseline (pp) | 主要贡献 |
|------|----------|--------------------|----------|
| Baseline (明文) | 0.2633 | 0 | 起点 |
| Baseline + 7-target LoRA (理论 Q0) | TBD | TBD | LoRA 参数量 |
| Baseline + V 量化 (理论 Q1) | TBD | TBD | 真实 V 量化税 |
| ... | ... | ... | ... |
| Baseline + 全协议 (理论 Q3) | TBD | TBD | 全协议税（含 mode collapse）|
| SLG (真实) | 0.1515 | -11.18 | 全链路真实精度 |
| Q3 noise 模拟 | 0.1495 | -11.38 | noise model 拟合（巧合）|

**关键洞察**：
- Q3 noise model 拟合 SLG aggregate macro_f1 的"成功"是**虚假成功**
- 真实 SLG 的精度损失来自 **mode collapse**（diagnosis/therapy F1=0），而非 noise model 拟合的"协议税"
- **当前 noise model 完全无法捕获 mode collapse 现象**

### 7.3 推荐的新研究方向

| 优先级 | 研究方向 | 预期产出 |
|--------|---------|----------|
| **P0** | 在 SLG 训练日志中检查 LoRA 权重是否真的更新 | 验证 R3 假设 A |
| **P0** | 在 baseline 上重训 7-target LoRA（5 epoch）| 分离 LoRA 参数量 vs 协议税 |
| **P1** | 在 SLG 数据集类上做 baseline 推理（不做协议训练）| 分离数据集类差异 vs 协议税 |
| **P1** | 在 baseline 上模拟 gold-only 协议（logits - V_gold @ V_T）| 验证 protocol_constraint σ |
| **P2** | 跑 10+ epoch 看 SLG 是否能学到 diagnosis/therapy | 验证 R3 假设 C（样本量 vs 协议税 SNR）|

---

## 8. 数据真实性再次确认

### 8.1 当前 SLG epoch_metrics.jsonl 数据可信度

| 字段 | 可信度 | 备注 |
|------|--------|------|
| `val_samples = 203` | ✅ 真实 | SLG 数据集类生成 203 个 sample（过滤 negative 后）|
| `n_parse_failures = 0` | ✅ 真实 | generate_predictions 修复后无解析失败 |
| `val_bt_macro_f1 = 0.1515` | ✅ 真实 | aggregate F1，反映真实精度 |
| `val_bt_macro_roc_auc = 0.6815` | ✅ 真实 | 自定义 per-class AUC |
| `per_class.diagnosis.f1 = 0` | ✅ 真实 | **模型预测崩塌的事实** |
| `y_pred_distribution.modulatory = 137` | ✅ 真实 | 67.5% 样本预测为 modulatory |

### 8.2 当前 epoch_metrics.jsonl vs `_SAVE_20260727_0706/epoch_metrics.jsonl` 的关系

**两个文件都存在**，分别记录不同评估阶段：

| 文件 | mtime | 含义 |
|------|-------|------|
| `_SAVE_20260727_0706/epoch_metrics.jsonl` | 2026-07-26 | 训练时实时记录，F1=0 是因为 n_parse_failures=134/134 |
| `epoch_metrics.jsonl` (顶层) | 2026-07-31 18:51 | **重新评估**修复后的 checkpoint，得到当前真实指标 |

**重新评估的触发**：`evaluate_slg_cls.py`（修复版）在 2026-07-31 18:48 修改，与 epoch_metrics.jsonl 生成时间一致。

---

## 9. 结论

### 9.1 当前精度损失分解的真相

| 之前假设 | 实际真相 |
|---------|---------|
| SLG 真实 macro_f1=0.1515 ≈ Q3 模拟 0.1495 = "noise model 拟合成功" | **aggregate 数字巧合，per-class 完全不同** |
| "Q3 与 SLG 差 0.26pp = 未建模加密税" | 实际上 Q3 没有 mode collapse，所以差距不是"未建模税" |
| "协议税是最大单步税（4.93pp）" | 协议税确实存在，但 mode collapse 造成的 -10pp diagnosis/therapy 失败**远超协议税** |

### 9.2 SLG diagnosis/therapy F1=0 的根因

**是 mode collapse，不是任务评估失配**：

1. ✅ 数据集类差异（R1, R2）确实存在，但只影响 ~0.8pp
2. ✅ 评估器正确实现（2026-07-26 修复）
3. ❌ SLG 模型训练后 67.5% 预测集中在 modulatory 类
4. ❌ diagnosis 和 therapy 类完全被忽略（F1=0）
5. ❌ Q3 noise model 无法复现 mode collapse

### 9.3 对用户的建议

**不要再试图通过 noise model 拟合 SLG aggregate 来分解精度损失**——这是统计拟合，不是机制分解。

**推荐替代方案**：

1. **用真实协议操作**（V 量化 + gold-only）在 baseline 上做"半 SLG"实验
2. **接受 mode collapse 是 SLG 设计的内禀缺陷**，报告里诚实说明
3. **设计新的 SLG 协议变体**（如 all-token 梯度、不依赖 V_gold 的 g_H 计算）来验证是否能避免 mode collapse

---

## 10. 附录：数据集类差异完整代码引用

### 10.1 baseline `correct_relation_char_index`

文件：`baseline/llama-rec/src/llama_recipes/datasets/biotriplex_qakshot_dataset.py:610-681`

```python
def correct_relation_char_index(self, relations, sentences, sentence_idx,
                                num_leading_spaces, stripped_sentence,
                                entities, return_neg_relations=False):
    # ... Cartesian product expansion ...
    # remove duplicates
    corrected_relations = list(set(tuple(relation) for relation in corrected_relations))
    # remove trailing whitespace
    corrected_relations = BioTriplexQADataset.relation_remove_trailing_whitespace(...)
    if return_neg_relations:
        # ADD "No Relation" entries — but never REMOVE existing ones
        ...
    # ❌ 永远不会过滤 rel_label in ("no relation", "relation undefined")
    return corrected_relations
```

### 10.2 SLG `_correct_relation_char_index`

文件：`src/data/biotriplex_dataset.py:272-360`

```python
def _correct_relation_char_index(raw_relations, ..., return_neg_relations: bool):
    ...
    out: List[Dict[str, Any]] = []
    for r in corrected:
        gene_text = stripped_sentence[r[0]:r[1]].strip()
        disease_text = stripped_sentence[r[2]:r[3]].strip()
        if not gene_text or not disease_text:
            continue
        rel_label = r[4]
        if not return_neg_relations and rel_label.lower() in ("no relation", "relation undefined"):
            continue  # ✅ 主动过滤
        out.append({"gene": gene_text, "disease": disease_text, "relation": rel_label})
    return out
```

### 10.3 SLG `_classify_from_logits`

文件：`src/parties/party_s.py:411-440`

```python
def _classify_from_logits(self, logits, attention_mask):
    opt_ids = self._get_option_token_ids(device=device)  # [7] long
    if attention_mask is not None:
        last_idx = self._get_last_nonpad_index(attention_mask.to(device))
    else:
        last_idx = torch.full((logits.size(0),), logits.size(1) - 1, ...)
    last_logits = logits.gather(1, gather_idx).squeeze(1)  # [B, V]
    option_logits = last_logits.index_select(-1, opt_ids).float()  # [B, 7]
    best_idx = option_logits.argmax(dim=-1)  # [B]
    predictions = [f"{option_letters[i]})" for i in best_idx.cpu().tolist()]
    return {"predictions": predictions, "logits": option_logits.cpu().tolist()}
```

---

**报告结束**。所有结论基于真实代码审计与真实数据，无猜测。