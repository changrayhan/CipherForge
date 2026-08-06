# _legacy_work_logs

本目录收录 2026-07-19 ~ 2026-07-20 早期 F1 评估实验的工作残留日志，已从仓库外的 `/root/autodl-tmp/_work/` 迁入。

## 文件清单

| 文件 | 说明 |
|---|---|
| `evaluate_f1.log` | 第一轮 F1 评估日志（2026-07-20 00:20） |
| `evaluate_f1_v2.log` | 第二轮 F1 评估日志（2026-07-20 00:24） |
| `wait_then_run_f1.sh` | "等待资源空闲后启动 F1 评估"的 shell 包装脚本 |
| `wait_then_run_f1.log` | 上述脚本的执行日志 |
| `wait_then_run_f1.nohup` | 上述脚本的 nohup 输出 |
| `monitor.pid` | 监控进程的 PID 文件残留（2026-07-19 23:46） |

## 上下文

这些文件来自项目最早期一轮"等待 GPU 资源空闲再跑 F1"实验的调度残留，路径硬编码于早期的 AutoDL 临时工作目录中。仓库整理时统一归入 `test-data/_legacy_work_logs/`，保留以备审计，但不再被任何生产脚本引用。

## 时间线

- 2026-07-19 23:46 — `monitor.pid` 写入
- 2026-07-20 00:20 — `wait_then_run_f1.sh` / `evaluate_f1.log` 写入
- 2026-07-20 00:24 — `evaluate_f1_v2.log` 写入
- 2026-07-20 08:30 — `wait_then_run_f1.sh` 最后修改（副本归档入此处）
- 2026-07-28 — 迁入本目录