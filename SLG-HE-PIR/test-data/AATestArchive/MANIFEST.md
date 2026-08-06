# AATestArchive 归档清单

> **生成日期**：2026-08-06 22:15 (UTC+8)
> **最后更新**：2026-08-06 22:39（删除重复备份 `test-data/AccuracyAblationTestData/`，节省 2.9 GB）
> **目的**：把报告 v9（`/root/autodl-tmp/精度测试摘要.md`）**未引用**的所有代码/数据归档到本目录，使 `test-data/BioTriplex1BTestData/` 仅保留 v9 引用的 105 runs 数据与必要代码。
> **操作方式**：cp -a（拷贝，源文件暂时保留，等用户指示决定是否删除）。
> **可逆性**：归档操作完全可逆——源文件未改动，可任意回滚。

---

## 一、归档规模

- **文件总数**：2,219
- **目录总数**：432
- **磁盘用量**：4,182,821,557 bytes ≈ **3.99 GB**
- **归档根**：`/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/test-data/AATestArchive`

---

## 二、归档目录结构

```
AATestArchive/
├── MANIFEST.md                                    # 本文件
├── BioTriplex1BTestData/
│   ├── scripts/                                   # 19 个 runner/备份脚本
│   │   ├── bio_baseline_trainer.py.phase15.bak    # Phase 1.5 历史备份 (20K)
│   │   ├── _bio_internal_runner.sh
│   │   ├── _bio_monitor.sh
│   │   ├── _rerun_bt_ab_no_beta.sh
│   │   ├── bio_phase15B_runner.sh
│   │   ├── bio_phase15C_runner.sh
│   │   ├── bio_phase16_runner.sh
│   │   ├── bio_phase1C_runner.sh
│   │   ├── bio_phase1D_runner.sh
│   │   ├── bio_phase1E_runner.sh
│   │   ├── bio_quant_runner.sh
│   │   ├── bio_slg_runner.sh
│   │   ├── chained_worker.sh
│   │   ├── launch_bio_after_trec.sh
│   │   ├── monitor_phase_runs.sh
│   │   ├── run_bio_full_background.sh
│   │   ├── run_bio_one_experiment.sh
│   │   ├── slg_metrics_adapter.py
│   │   ├── wait_for_trec_then_cleanup.sh
│   │   └── worker.sh
│   └── runs/
│       ├── _helpers/                              # 18 个报告构建/监控脚本与日志
│       │   ├── _appendix.md (80K)
│       │   ├── _appendix_body.md (80K)
│       │   ├── _appendix_body_clean.md (80K)
│       │   ├── _build_v9.py
│       │   ├── _dump_summary.py
│       │   ├── _dump_v9.py
│       │   ├── _dump_v9.txt (16K)
│       │   ├── _extract_facts.py
│       │   ├── _make_appendix.py
│       │   ├── _monitor.sh
│       │   ├── _monitor.log (282K)
│       │   ├── _monitor_nohup.log
│       │   ├── _phase16_daemon.sh
│       │   ├── _phase16_daemon.log (37K)
│       │   ├── _phase16_daemon.startup.log
│       │   ├── _phase16_watcher.sh
│       │   ├── _phase16_watcher.log
│       │   └── _phase16_watcher.startup.log
│       ├── _failed_phases/                        # 6 个失败/未完成 phase 目录
│       │   ├── baseline_extra_seeds/              # 18 runs, 0 epoch, 强停
│       │   ├── cumulative/                        # 12 runs, 1 partial (3 epoch)
│       │   ├── dp_alpha_scan/                     # 空目录（仅有 _runner_logs）
│       │   ├── fullstack_baseline/                # 3 runs, 0 epoch, 强停
│       │   ├── slg/                               # 1 partial run, seed_42 step 322 中断
│       │   └── _backup_phase1.5_20260801_1936/    # 早期备份 (611M)
│       └── _per_phase_runner_logs/                # 8 phase × {_runner_logs, _summary, _worker1.*}
└── test-data/
    └── AccuracyAblationTestData/                  # 独立的早期对照实验（2.9G）
        ├── README.md
        ├── baseline/                              # 早期 baseline (5 epoch, 1.5M)
        ├── slg/                                   # 早期 SLG (5 epoch, 80K)
        ├── runs/
        │   ├── v2/                                # baseline 9×3=27 runs × 106M = 2.8G
        │   │   ├── baseline/                      # 含 adapter_model.safetensors + ckpts
        │   │   ├── ablation/                      # 8 cfg × 3 seeds = 24 runs
        │   │   ├── bugfix/                        # 修复版数据
        │   │   ├── configs/
        │   │   ├── logs/
        │   │   ├── outputs/
        │   │   └── (其他辅助文件)
        │   └── slg/
        │       └── SLG-fixed_seed42/              # SLG 早期 fix 版本
        ├── derived/                               # 对比分析结果 (6 文件)
        ├── quantization_params/                   # SLG 参数提取 (3 文件)
        └── _archive/                              # 早期被杀的 smoke test
```

---

## 三、报告 v9 引用的路径（保留，未归档）

| # | 路径 | 用途 |
|---|------|------|
| R1 | `src/training/biotriplex_metrics.py` | §2.1 / §8 引用，metrics 计算器 |
| R2 | `test-data/BioTriplex1BTestData/scripts/bio_baseline_trainer.py` | §8 引用 |
| R3 | `test-data/BioTriplex1BTestData/scripts/bio_baseline_trainer_v2.py` | §8 引用 |
| R4 | `test-data/BioTriplex1BTestData/scripts/bio_evaluator.py` | 隐式（_extract_facts 调用） |
| R5 | `test-data/BioTriplex1BTestData/scripts/bio_summarize.py` | 数据汇总 |
| R6 | `test-data/BioTriplex1BTestData/scripts/bio_resplit.py` | 数据划分 |
| R7 | `test-data/BioTriplex1BTestData/data/` | train/val/test para + gold + split_manifest |
| R8 | `test-data/BioTriplex1BTestData/PHASE_1.5_TREND_ANALYSIS.md` | Phase 1.5 分析 |
| R9 | `test-data/BioTriplex1BTestData/runs/baseline/` | §8 + §3.1 引用 |
| R10 | `test-data/BioTriplex1BTestData/runs/quant/` | §8 + §3.1 引用 |
| R11 | `test-data/BioTriplex1BTestData/runs/quant_dp15/` | §8 + §3.1 引用 |
| R12 | `test-data/BioTriplex1BTestData/runs/quant_v2/` | §8 + §3.1 引用 |
| R13 | `test-data/BioTriplex1BTestData/runs/_summary/all_phases.{csv,md}` | §8 引用 |
| R14 | `test-data/BioTriplex1BTestData/runs/_extract_facts.json` | 数据池（隐式） |

---

## 四、二次核查结果

### 4.1 完整性核查（归档 vs 源）
| 项 | 期望 | 实际 | 状态 |
|----|------|------|------|
| `scripts/` 归档文件数 | 19+1=20 | 20 | ✅ |
| `runs/_helpers/` 归档文件数 | 18 | 18 | ✅ |
| `runs/_failed_phases/` 目录数 | 6 | 6 | ✅ |
| `runs/_per_phase_runner_logs/` 目录数 | 8 phase | 8 | ✅ |
| `AccuracyAblationTestData/` 文件数 | 1384 | 1384 | ✅ |
| `AccuracyAblationTestData/` `diff -rq` 无差异 | true | true | ✅ |

### 4.2 报告引用完整性核查（105 runs 全部留在源）
| Phase | 期望 epoch files | 实际 | 状态 |
|-------|------------------|------|------|
| baseline | 24×8=192 | 192 | ✅ |
| quant | 24×8=192 | 192 | ✅ |
| quant_dp15 | 18×8=144 | 144 | ✅ |
| quant_v2 | 39×8=312 | 312 | ✅ |
| **合计** | **840** | **840** | ✅ |
| `_summary/all_phases.csv` | 存在 | 存在 | ✅ |
| `_summary/all_phases.md` | 存在 | 存在 | ✅ |
| `_extract_facts.json` | 存在 | 存在 | ✅ |

### 4.3 隔离核查（归档中不应含 v9 引用项的核心数据）
| 检查项 | 期望 | 实际 | 状态 |
|--------|------|------|------|
| 归档 `baseline/` 含 `epoch_xxx_bio_metrics.json` | 0 | 0 | ✅ |
| 归档 `quant/` 含 `epoch_xxx_bio_metrics.json` | 0 | 0 | ✅ |
| 归档 `quant_dp15/` 含 `epoch_xxx_bio_metrics.json` | 0 | 0 | ✅ |
| 归档 `quant_v2/` 含 `epoch_xxx_bio_metrics.json` | 0 | 0 | ✅ |
| 归档 `_summary/all_phases.{csv,md}` | 不存在 | 不存在 | ✅ |

---

## 五、用户授权的归档范围（Q1_A）

| 来源路径 | 类别 | 大小 |
|---------|------|------|
| `BioTriplex1BTestData/scripts/` 下 19 个辅助脚本 | runner/备份 | 132 KB |
| `BioTriplex1BTestData/scripts/bio_baseline_trainer.py.phase15.bak` | 备份 | 20 KB |
| `BioTriplex1BTestData/runs/` 下 18 个辅助构建/监控文件 | 工具/日志 | 672 KB |
| `BioTriplex1BTestData/runs/_backup_phase1.5_20260801_1936/` | 早期备份 | 611 MB |
| `BioTriplex1BTestData/runs/baseline_extra_seeds/` | 失败 phase | 232 KB |
| `BioTriplex1BTestData/runs/cumulative/` | 失败 phase | 796 KB |
| `BioTriplex1BTestData/runs/dp_alpha_scan/` | 空 phase | 0 |
| `BioTriplex1BTestData/runs/fullstack_baseline/` | 失败 phase | 60 KB |
| `BioTriplex1BTestData/runs/slg/` | 失败 phase | 480 KB |
| `BioTriplex1BTestData/runs/{phase}/_runner_logs/` × 8 | 调度痕迹 | 144 KB |
| `BioTriplex1BTestData/runs/{phase}/_summary/` × 8（注意：顶层 _summary/ 保留） | phase 级汇总 | 76 KB |
| `BioTriplex1BTestData/runs/quant_dp15/_worker1.{sh,log}` | 调度脚本 | 8 KB |
| `test-data/AccuracyAblationTestData/` 全部 | 独立实验 | 2.9 GB |
| **合计** | — | **~3.99 GB** |

---

## 六、操作记录

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-06 22:13 | 创建归档子目录 | ✅ |
| 2026-08-06 22:13 | 归档 scripts/ 下 19+1 文件 | ✅ 20/20 verified |
| 2026-08-06 22:13 | 归档 runs/_helpers/ 下 18 文件 | ✅ 18/18 verified |
| 2026-08-06 22:14 | 归档 runs/_failed_phases/ 下 6 目录 | ✅ 6/6 verified (含 611M 早期备份) |
| 2026-08-06 22:14 | 归档 runs/_per_phase_runner_logs/ | ✅ 8 phase, 40+ 文件 |
| 2026-08-06 22:15 | 归档 AccuracyAblationTestData/ (2.9G) | ✅ 1384/1384 files, diff -rq 无差异 |
| 2026-08-06 22:15 | 二次核查 105 runs 数据完整性 | ✅ 840/840 epoch files 在源 |
| 2026-08-06 22:15 | 二次核查归档隔离性 | ✅ 4 个 phase 归档中无 bio_metrics |

---

## 七、待用户决策

| 问题 | 当前状态 |
|------|---------|
| 是否删除源文件（`rm` 备份归档后的 105 runs 之外的所有文件）？ | **未执行**——源文件全部保留，等用户指示 |

