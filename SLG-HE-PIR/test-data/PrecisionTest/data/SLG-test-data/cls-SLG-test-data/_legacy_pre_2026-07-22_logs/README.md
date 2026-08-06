# _legacy_pre_2026-07-22_logs

本目录收录 2026-07-21 的早期 BioTriplex 分类微调实验日志，已从仓库外的 `/root/autodl-tmp/slg-biotriplex-classification/logs/` 迁入。

## 文件清单

| 文件模式 | 数量 | 说明 |
|---|---:|---|
| `biotriplex_classification_2epochs_*.log` | 5 | 早期分类任务 2 epoch 微调日志（1784619166 / 1784619238 / 1784619600 / 1784619712 / 1784619844） |
| `biotriplex_finetune_1_*.log` | 4 | 早期 BioTriplex 微调 v1 版本日志（1784616072 / 1784616402 / 1784617212 / 1784617374） |
| `biotriplex_finetune_2epochs_1784619128.log` | 1 | 早期 2 epoch 微调日志 |
| `biotriplex_finetune_all_*.log` | 5 | 早期 BioTriplex 完整微调日志（1784619167 / 1784619240 / 1784619602 / 1784619714 / 1784619846） |
| `step_profiles.jsonl` | 1 | 早期 step profile 性能采样数据（73 KB） |

合计 16 个文件。

## 上下文

这些是 2026-07-21 当天多次启动 BioTriplex 分类微调实验产生的日志，使用了 `slg-biotriplex-classification/` 这一临时项目目录。其中的 adapter、checkpoint 目录迁移时已清空，仅保留日志和 step profile 用于历史回溯。

## 与当前产物的关系

| 时间 | 路径 | 关系 |
|---|---|---|
| 2026-07-21 | `_legacy_pre_2026-07-22_logs/`（本目录） | 早期微调实验 |
| 2026-07-22 起 | `cls-SLG-test-data/logs/` | 重新组织后的微调产物 |
| 2026-07-27 07:06 | `cls-SLG-test-data/_SAVE_20260727_0706/` | 当前最新分类测试快照 |

`STEP_PROFILE`、`LOSS`、`METRIC` 字段语义在三个时间段保持一致，可直接对比。