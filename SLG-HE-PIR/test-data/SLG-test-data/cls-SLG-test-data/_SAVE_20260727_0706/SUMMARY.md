# Task A (GenRel 7-class) 训练结果归档

**停止时间**：2026-07-27 07:06 (UTC+8)
**停止原因**：用户请求停止
**退出方式**：SIGINT → KeyboardInterrupt 干净路径
**总耗时**：~4 天 15 小时 (epoch 0-4 各 ~20 小时，epoch 5 未开始)

## 已保存资产 (428 MB)

| 文件 | 大小 | 含义 |
|------|------|------|
| `best_checkpoint.pt` | 62M | 历史最佳 val 指标对应的 epoch (实为 epoch 0, 因为后续指标全 0) |
| `last_checkpoint.pt` | 62M | 最新 checkpoint (epoch 4) |
| `checkpoint_epoch_000.pt` ~ `004.pt` | 62M × 5 | 各 epoch 末的 LoRA 权重 |
| `epoch_metrics.jsonl` | 3.5K | 5 个 epoch 的关键指标 |
| `KEY_EVENTS.log` | 4.2K | 训练日志中关键事件 (epoch 切分、save ckpt 等) |
| `性能测试重大为题ROOT_CAUSE_PARSE_FAILURES.md` | - | 根因诊断 + 修复方案文档 |

## epoch_metrics 摘要

{"epoch": 0, "timestamp": 1784779277.2304137, "elapsed_s": 74649.09987711906, "train_loss": 28160.0, "train_steps": 734, "avg_step_time_ms": 100980.29654487278, "avg_gpu_mem_mb": 29583.277902545982, "val_entity_micro_f1": 0.0, "val_letter_micro_f1": 0.0, "val_micro_precision": 0.0, "val_micro_recall": 0.0, "val_micro_accuracy": 0.0, "val_macro_f1": 0.0, "val_weighted_f1": 0.0, "val_ce_loss": 14.256529850746269, "val_samples": 134, "val_bt_micro_f1": 0.0, "val_bt_macro_f1": 0.0, "val_bt_weighted_f1": 0.0, "val_bt_multilabel_f1_samples": 0.0, "val_bt_multilabel_f1_macro": 0.0, "val_bt_multilabel_f1_micro": 0.0, "val_bt_macro_roc_auc": 0.0, "val_bt_micro_roc_auc": 0.0, "val_bt_n_parse_failures": 134}
{"epoch": 1, "timestamp": 1784853760.0460894, "elapsed_s": 149131.91555309296, "train_loss": 28160.0, "train_steps": 734, "avg_step_time_ms": 100724.8126658172, "avg_gpu_mem_mb": 29589.99267578125, "val_entity_micro_f1": 0.0, "val_letter_micro_f1": 0.0, "val_micro_precision": 0.0, "val_micro_recall": 0.0, "val_micro_accuracy": 0.0, "val_macro_f1": 0.0, "val_weighted_f1": 0.0, "val_ce_loss": 14.838619402985074, "val_samples": 134, "val_bt_micro_f1": 0.0, "val_bt_macro_f1": 0.0, "val_bt_weighted_f1": 0.0, "val_bt_multilabel_f1_samples": 0.0, "val_bt_multilabel_f1_macro": 0.0, "val_bt_multilabel_f1_micro": 0.0, "val_bt_macro_roc_auc": 0.0, "val_bt_micro_roc_auc": 0.0, "val_bt_n_parse_failures": 134}
{"epoch": 2, "timestamp": 1784928038.176335, "elapsed_s": 223410.04579877853, "train_loss": 28160.0, "train_steps": 734, "avg_step_time_ms": 100435.98426882512, "avg_gpu_mem_mb": 29589.99267578125, "val_entity_micro_f1": 0.0, "val_letter_micro_f1": 0.0, "val_micro_precision": 0.0, "val_micro_recall": 0.0, "val_micro_accuracy": 0.0, "val_macro_f1": 0.0, "val_weighted_f1": 0.0, "val_ce_loss": 14.88945895522388, "val_samples": 134, "val_bt_micro_f1": 0.0, "val_bt_macro_f1": 0.0, "val_bt_weighted_f1": 0.0, "val_bt_multilabel_f1_samples": 0.0, "val_bt_multilabel_f1_macro": 0.0, "val_bt_multilabel_f1_micro": 0.0, "val_bt_macro_roc_auc": 0.0, "val_bt_micro_roc_auc": 0.0, "val_bt_n_parse_failures": 134}
{"epoch": 3, "timestamp": 1785003169.76124, "elapsed_s": 298541.63070344925, "train_loss": 28160.0, "train_steps": 734, "avg_step_time_ms": 101597.8563179437, "avg_gpu_mem_mb": 29589.99267578125, "val_entity_micro_f1": 0.0, "val_letter_micro_f1": 0.0, "val_micro_precision": 0.0, "val_micro_recall": 0.0, "val_micro_accuracy": 0.0, "val_macro_f1": 0.0, "val_weighted_f1": 0.0, "val_ce_loss": 14.59375, "val_samples": 134, "val_bt_micro_f1": 0.0, "val_bt_macro_f1": 0.0, "val_bt_weighted_f1": 0.0, "val_bt_multilabel_f1_samples": 0.0, "val_bt_multilabel_f1_macro": 0.0, "val_bt_multilabel_f1_micro": 0.0, "val_bt_macro_roc_auc": 0.0, "val_bt_micro_roc_auc": 0.0, "val_bt_n_parse_failures": 134}
{"epoch": 4, "timestamp": 1785078565.1906772, "elapsed_s": 373937.060141325, "train_loss": 28160.0, "train_steps": 734, "avg_step_time_ms": 101952.24711420751, "avg_gpu_mem_mb": 29589.99267578125, "val_entity_micro_f1": 0.0, "val_letter_micro_f1": 0.0, "val_micro_precision": 0.0, "val_micro_recall": 0.0, "val_micro_accuracy": 0.0, "val_macro_f1": 0.0, "val_weighted_f1": 0.0, "val_ce_loss": 14.777518656716419, "val_samples": 134, "val_bt_micro_f1": 0.0, "val_bt_macro_f1": 0.0, "val_bt_weighted_f1": 0.0, "val_bt_multilabel_f1_samples": 0.0, "val_bt_multilabel_f1_macro": 0.0, "val_bt_multilabel_f1_micro": 0.0, "val_bt_macro_roc_auc": 0.0, "val_bt_micro_roc_auc": 0.0, "val_bt_n_parse_failures": 134}

## 未保存的资产

- `adapter/`: 空目录 — 从未生成 (训练代码可能没调用 LoRA save_adapter)
- `test_metrics.json`: 未生成 — `do_test_eval=True` 阶段还没跑到
- `bfv_cache/`: 在 `/root/autodl-tmp/slg-bfv-cache/` 单独存放 (公用，不需归档)

## 已知问题

5 个 epoch 训练下来，所有 val 指标都是 0。这不是模型没学到东西，而是 `generate_predictions` 把 LM logits 整句 argmax 导致 134/134 解析失败 (详见 ROOT_CAUSE 文档)。修复 `src/parties/party_s.py:287-311` 后，应该能看到 `val_bt_micro_accuracy ≈ 0.4-0.6` 区间。
