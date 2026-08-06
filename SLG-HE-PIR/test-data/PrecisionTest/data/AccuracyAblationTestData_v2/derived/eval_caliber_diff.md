# Evaluation-caliber differences: Baseline vs SLG

Baseline uses `baseline/classification_genrel/scripts/evaluate_metrics.py` (sklearn-backed).
SLG uses `src/training/biotriplex_metrics.py::compute_classification_metrics`.

| 维度 | Baseline (`evaluate_metrics.py`) | SLG (`biotriplex_metrics.py`) |
|------|----------------------------------|--------------------------------|
| 缺失预测处理 | 保留 `pred_idx=-1`，最后 y_pred_safe 替换为 0 | 替换为 "relation undefined" (idx=6) |
| Multilabel F1 输入 | 把 single-label 转成 binary 7-vec | 解析 multi-letter 字符串 `j),o)` |
| macro_roc_auc_ovr | sklearn 标准 ovr（包含无 support 的类） | 仅在 y_true 含该类时计算，取有效类均值 |
| ROC AUC 的 y_score | softmax over 7-class logits | softmax over 7-class logits（相同路径）|
| Per-class P/R/F1 | 自实现的 TP/FP/FN + zero_division | sklearn `precision_score` + `f1_score` |
| Confusion matrix | sklearn 完整 | sklearn 完整 |

**结论**:
1. parse_failures 都为 0 时 (本次数据)，缺失预测处理差异**不会影响最终指标**（因为没有失败样本）
2. macro_auc 计算路径**不同**：SLG 排除 y_true 中没有的类——本次测试集中 "no relation" 和 "relation undefined" 在 SLG 全部 support=0，所以 SLG 实际只在 5 个类上算 macro_auc，而 Baseline 在 7 个类上算（其中两个为 0）→ **SLG 的 macro_auc 比 Baseline 的更"乐观"**
3. multilabel F1 的解析路径**相同**（因为生成为单字母）

## 建议

为了获得**统一口径**的 Baseline vs SLG 对比，新会话的 `report_generator.py` 必须：

```python
from src.training.biotriplex_metrics import compute_classification_metrics
# 从 baseline raw_inference/infer_outputs_epoch_XXX.json 抽 answer + logits
# 重新计算，统一口径
```

或者在 QUANT_ABLATION_REPORT.md 中**明确标注** "本报告中的 Baseline 指标来自 sklearn 口径，SLG 指标来自自定义口径，两者不可直接相减"。
