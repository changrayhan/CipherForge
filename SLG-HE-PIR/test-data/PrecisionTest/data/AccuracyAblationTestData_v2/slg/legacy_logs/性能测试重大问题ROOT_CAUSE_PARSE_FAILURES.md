# Task A (GenRel 7-class) 验证 134/134 解析失败 — 根因诊断与修复指南

**编写日期**：2026-07-26
**适用场景**：SLG-HE-PIR 异构 PIR 三方微调中，Task A (Classification GenRel QA) 验证阶段 `val_bt_n_parse_failures = 134/134` 全部失败的问题。
**文档目的**：定位根因、给出修复方案，并指出其他被同一根因掩盖的次生 bug。

---

## 1. 问题现象

Stage 2 验证日志中观察到：

```
val_bt_n_parse_failures = 134    # 全部 134 个样本都解析失败
val_bt_micro_f1 = 0.0
val_bt_macro_f1 = 0.0
val_bt_weighted_f1 = 0.0
val_bt_multilabel_f1_samples = 0.0
val_bt_micro_accuracy = 0.0
val_ce_loss ≈ 14.25               # 这个数字也是错的，原因见 §4
train_loss ≈ 28160.0              # 这个数字也不是真正的 CE，见 §4
```

对照基线 (`baseline/classification_genrel/scripts/`) 的 7-class 分类任务**完全可以正常运行**：

| 指标 | 基线 (paper 实现) | SLG-HE-PIR 异构 PIR 路径 |
|------|-------------------|---------------------------|
| `parse_failures` | 0 / 213 | 134 / 134 |
| `micro_accuracy` | 0.5775 | 0.0 |
| `macro_f1` | 0.4094 | 0.0 |
| `macro_auc_ovr` | 0.8722 | (无法计算) |

问题不在训练数据、不在 prompt 格式、不在标签生成；问题在**验证阶段的预测生成路径**。

---

## 2. 根因 — `generate_predictions` 把语言模型 logits 当成 7-class logits

### 2.1 出错位置

`src/parties/party_s.py:287-311`，函数 `PartyS.generate_predictions`：

```python
def generate_predictions(self, H_M_or_logits):
    # ... logits = H_M @ V^T, shape = [B, S, V]   V = 128256 (vocab_size)
    token_ids = logits.argmax(dim=-1)            # [B, S]   ← 错误：对每个 token 位置取 argmax
    for b in range(token_ids.shape[0]):
        pred_tokens = token_ids[b].cpu().tolist()  # 整句的 argmax token ids
        predictions.append(self._decode_tokens(pred_tokens))  # decode 整句 = 噪声
    return {"predictions": predictions, "token_ids": token_ids.cpu().tolist()}
```

**本质**：S 的 `logits` 是**语言模型 logits** `shape = [batch, seq_len, vocab_size]`，vocab_size = 128256。但本函数当成 `[batch, num_classes]` 用，对**整句每个 token 位置**取 argmax，得到 `[batch, seq_len]` 的 token-id 序列，再 decode 成字符串。

decode 出来的字符串**根本不含** `a)` / `b)` / ... / `g)` 这种格式的子串 — 因为 prompt 文本本身不含这些 pattern，而 argmax 在 128256 维 vocab 上的随机选择几乎不可能凑出 `a)` 格式。

### 2.2 后续解析为什么一定失败

链路：

1. `heterogeneous_protocol.py:531` 把上面的 `predictions`（噪声字符串）返回
2. `heterogeneous_protocol.py:567` 对每个 `p` 调 `parse_answer_letter(p)`：
   ```python
   matches = re.findall(r'[a-z]+\)', text)
   if matches:
       return ", ".join(matches)
   ```
   对噪声字符串匹配 `a)` / `b)` / ... → **几乎必失败** → 返回 `""`
3. `trainer.py:549` 把空字符串传给 `compute_classification_metrics(all_predictions, all_labels)`
4. `biotriplex_metrics.py:51-61` 的 `_parse_letter_answer("")`：
   ```python
   if not t:  # 空字符串
       return None
   ```
   → `parse_fail += 1`

每个样本都要走这条路径 → **134/134 失败**。

### 2.3 基线怎么做的（正确的做法）

参考 `baseline/classification_genrel/scripts/infer_and_save.py:60-78, 146-161`：

```python
# 1. 把 7 个 option letter 映射到 7 个 token id
def get_option_token_ids(tokenizer, letters=["a","b","c","d","e","f","g"]):
    token_ids = []
    for letter in letters:
        candidates = [f"{letter})", letter, f" {letter})", f" {letter}"]
        chosen = None
        for c in candidates:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                chosen = ids[0]
                break
        token_ids.append(chosen)
    return token_ids

# 2. 推理时：
out = model(**inputs)
last_logits = out.logits[0, -1, :].float().cpu()         # ← 只取最后一个位置
option_logits = [float(last_logits[tid]) for tid in option_token_ids]  # ← 只取 7 个 option id
probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
best_idx = int(torch.tensor(probs).argmax().item())
answer_letter = OPTION_LETTERS[best_idx]                  # ← 7-class argmax → "a"/"b"/.../"g"
outputs[doc_key] = {"answer": f"{answer_letter})", "logits": option_logits, ...}
```

**关键三步**：
1. **取 `logits[:, -1, :]`**（最后一个 token 位置），而不是整句 argmax
2. **只取 7 个 option token id 上的 logits**（投影到 7 维），而不是 vocab argmax
3. **7-class argmax**，直接选 0..6，再映射回 `"a)"` 字符串

`SLG-HE-PIR` 异构 PIR 路径里 S 拿到的是 `[B, S, V]` 的 logits（与基线 `model.logits` 同构），所以**完全可以照搬**这套做法。

---

## 3. 修复方案

### 3.1 推荐方案 — 改 `generate_predictions`

修改 `src/parties/party_s.py:287-311`：

```python
def generate_predictions(self, H_M_or_logits):
    if isinstance(H_M_or_logits, dict) and "H_M" in H_M_or_logits:
        H_M = H_M_or_logits["H_M"]
        logits = self.compute_logits_gpu(H_M)  # [B, S, V]
    elif isinstance(H_M_or_logits, torch.Tensor):
        logits = H_M_or_logits
    else:
        raise ValueError("Unknown input type for generate_predictions")

    # === 7-class classification projection ===
    option_token_ids = self._get_option_token_ids()  # [7] long
    last_logits = logits[:, -1, :].float()           # [B, V]
    option_logits = last_logits[:, option_token_ids]  # [B, 7]
    best_idx = option_logits.argmax(dim=-1).cpu().tolist()  # [B] int

    option_letters = ["a", "b", "c", "d", "e", "f", "g"]
    predictions = [f"{option_letters[i]})" for i in best_idx]
    return {
        "predictions": predictions,
        "logits": option_logits.cpu().tolist(),       # [B, 7] 给 AUC 用
    }

def _get_option_token_ids(self):
    """把 a/b/c/d/e/f/g 七个字母映射到对应的 single-token id。"""
    from transformers import AutoTokenizer
    if not hasattr(self, "_tokenizer"):
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_path, use_fast=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
    ids = []
    for letter in "abcdefg":
        chosen = None
        for c in (f"{letter})", letter, f" {letter})", f" {letter}"):
            tok = self._tokenizer.encode(c, add_special_tokens=False)
            if len(tok) == 1:
                chosen = tok[0]
                break
        if chosen is None:
            tok = self._tokenizer.encode(f"{letter})", add_special_tokens=False)
            chosen = tok[0]
        ids.append(chosen)
    return torch.tensor(ids, dtype=torch.long, device=logits.device)
```

### 3.2 调用方 — 验证代码如何对接

`src/parties/heterogeneous_protocol.py:531` 不需要大改 — 现在的代码 `predictions = s_pred.get("predictions", [])` 直接拿字符串列表就行。但 `compute_classification_metrics` 想算 ROC AUC 必须拿到 logits，需要**扩展 result**：

```python
# heterogeneous_protocol.py:step_val 增加：
result = self.party_s.generate_predictions(H_M)
predictions = result.get("predictions", [])
pred_logits_7class = result.get("logits")  # [B, 7] 或 None

out = {
    "predictions": predictions,
    "predictions_letters": predictions,           # 与 predictions 等价（已是 "a)" 形式）
    "labels": labels or [],
    "labels_letters": labels or [],               # 标签也是 "a)" 形式
    "pred_logits": pred_logits_7class,            # ← 新增，给 AUC
    ...
}
```

`src/training/biotriplex_metrics.py` 的 `compute_classification_metrics` 也需要增加接受 `pred_logits` 参数（每个 sample 的 7 维 logits），用于算 `macro_roc_auc_ovr` / `micro_roc_auc_ovr`。当前的实现是退化版的 one-hot（`biotriplex_metrics.py:230-232`），所以 AUC 一直是退化值。

### 3.3 验证

修复后跑 Stage 2 validation，应该看到：

- `val_bt_n_parse_failures = 0`
- `val_bt_micro_accuracy` 在 0.4-0.6 区间（参考基线 0.5775）
- `val_bt_macro_f1` 在 0.3-0.5 区间（参考基线 0.4094）
- `val_bt_macro_roc_auc_ovr` 在 0.7-0.9 区间（参考基线 0.8722）

如果修复后这些数字仍是 0 或接近 0，说明模型训练本身有问题（梯度方向、lr、数据 pipeline），需要另外排查。

---

## 4. 顺带发现的次生 bug

### 4.1 trainer.py 的 val CE loss 算的不是分类 CE

`src/training/trainer.py:401-407`：

```python
ce = F.cross_entropy(
    logits.view(-1, logits.size(-1)),    # [B*S, V=128256]
    labels_tensor.view(-1),              # [B*S] (output_ids, prompt 部分 = -100)
    ignore_index=-100,
)
```

- `logits` shape `[B, S, V=128256]`（vocab logits）
- `labels_tensor` 是 `output_ids`，shape `[B, S]`，**prompt 部分 = -100，response 部分是 token id**

这是 **token-level CE**（语言模型 next-token prediction），不是 7-class classification CE。对分类任务**没有评估价值**。

`val_ce_loss = 14.25` 也因此是个伪信号 — 它度量的是模型预测 response token 的能力（其实只预测 4-5 个 token：`\n### Response:\na)`），与 7-class 分类无关。

**修复**：在 7-class 路径上，CE loss 应该是：

```python
# 7-class CE：取 logits 最后一个位置在 7 个 option token id 上的 logits
# y_true = label_idx (0..6)
# y_pred = softmax(option_logits)[label_idx] 的负对数
```

### 4.2 `train_loss` 不是真正的 CE loss

`src/parties/party_m.py:499`：

```python
loss_proxy = float(g_H.detach().norm() / max(1, g_H.shape[0] * g_H.shape[1]))
return loss_proxy
```

`train_loss` 是 **gradient tensor `g_H` 的 L2 范数除以元素数**，不是 CE loss。`trainer.py:340` 直接拿 `result.loss = float(ack.get("loss", 0.0))` 当 `train_loss`。

这是个**代理指标**（proxy），用来观察训练是否在进行（梯度范数会缓慢变化）。它**完全不能**用来判断模型收敛、欠拟合、过拟合。所以：

- `train_loss ≈ 28160.0` 4 个 epoch 都没怎么变 → **不意味着模型没学到东西**
- 只能说明 `g_H` 范数稳定（这正常，因为 `g_H = a_t - V_y` 的期望范数由初始化决定）

要真正知道训练是否有效，**唯一可用的指标是 val 评估指标**（修复 §3 之后）。

---

## 5. 其他验证路径不受影响

- **NER 任务**（task_type="generation"）：`generate_predictions` 用的是 `_decode_token_tensor` + `parse_answer_letter` → NER 输出是 JSON 数组的 JSON 字符串。`generate_predictions` 的 bug 让整句 argmax，decode 出来基本不会是 `[{"span":..., "entity_type":...}]` JSON 格式，所以 NER 任务**也有同样的 bug**。
- **`predictions_letters`（`heterogeneous_protocol.py:567`）**：拿 `parse_answer_letter(predictions[i])` 处理噪声，对分类任务**没意义**。
- **`labels_letters`（`heterogeneous_protocol.py:548-565`）**：gold letter 是从 `output_ids` decode 出来再 `parse_answer_letter` 提取，对 `a)` 这种干净的 gold 文本**能解析成功**，所以 labels 部分不会引发 `parse_failures`。

---

## 6. 行动清单（优先级降序）

1. **【必做】修复 `PartyS.generate_predictions`**：按 §3.1 实现 7-class 投影，输出 `"a)"`/`"b)"`/.../`"g)"` 字符串 + 7 维 logits。
2. **【必做】扩展 `HeterogeneousProtocol.step_val`**：把 7 维 logits 通过 `pred_logits` 字段传出，供 AUC 计算。
3. **【必做】扩展 `compute_classification_metrics`**：接受 `pred_logits` 参数，正确计算 `macro_roc_auc_ovr` / `micro_roc_auc_ovr`。
4. **【建议】重写 trainer 的 val CE loss**：按 §4.1 改成 7-class CE，或干脆删掉（val 已有 macro_f1 等更直接的指标）。
5. **【可选】改 `train_loss` 的语义**：要么在文档里明确标注它是 `g_H` 范数代理，要么重命名避免与 CE loss 混淆。
6. **【可选】NER 任务同样修复**：`generate_predictions` 是 NER / classification 共用的路径。

---

## 7. 参考文件

- 错误代码：`src/parties/party_s.py:287-311` (`PartyS.generate_predictions`)
- 错误链路：`src/parties/heterogeneous_protocol.py:509-608` (`step_val`)
- 评估逻辑：`src/training/biotriplex_metrics.py:51-273` (`compute_classification_metrics`)
- 训练侧调用：`src/training/trainer.py:347-602` (`_run_val_epoch`)
- 基线参考：`baseline/classification_genrel/scripts/infer_and_save.py:60-78, 146-161`
- 基线指标参考：`baseline/classification_genrel/logs/genrel_20260722_094508_evaluate_metrics.json`