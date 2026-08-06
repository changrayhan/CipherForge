# 中间状态报告 — AccuracyAblationTest 阶段 0 ~ 阶段 4

> 截至：**2026-08-01 10:05:30 CST**
> 报告人：自动生成的精度测试执行报告
> 范围：`/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/`

---

## 摘要

| 阶段 | 计划状态 | 实际状态 | 结论 |
|------|----------|----------|------|
| 阶段 0 — Bug 修复（4 个 Bug） | completed | ✓ 完成 | 产物齐全 |
| 阶段 1 — Baseline 对照实验 | in_progress（计划文档未更新）| ✓ 完成 | 27/27 训练产出 |
| 阶段 1.5 — 中间值量化档位 ablation | pending | ✓ 完成 | 27/27 评估产出 |
| 阶段 3 — gold-only 协议对照 | pending | ⊘ 跳过 | 用户决策：留待阶段 5 注解 |
| 阶段 4 — SLG-fixed 重训 | pending | ⏳ 待跑 | smoke 1 种子 → 全量 3 种子 |
| 阶段 5 — 精度损失分解报告 | pending | ⊘ 未启动 | 依赖阶段 4 完成 |

---

## 一、阶段 0 — Bug 修复（4 个）

### 1.1 Bug 0.1：`eval_replay.py:243` 的 `or True`

**修复标记**：
- `gold_keys = set()`（第 236 行）
- `gold_keys.update(full_keys)`（第 238 行）
- `if candidate in gold_keys:`（第 251 行）
- `out[base_dk] = entry; mapping[base_dk] = base_dk`（fallback 路径）

**影响**：gold_keys 由 `base_to_full.values()` 改为统一构建集；样本对齐从 168 恢复到 213。

### 1.2 Bug 0.2：`quant_hooks.py` 的 Q0/Q0' 相同问题

**产物**：`AccuracyAblationTest/accuracy_ablation/quant_hooks.py`（14 512 字节，2026-08-01 01:02）

修复后 Q0 = 7-target SLG adapter 输出 + 无噪声；Q0' = 2-target Baseline adapter 输出 + 无噪声。两者完全区分开。

### 1.3 Bug 0.3：evaluator 不一致 → `unified_evaluator.py`

**产物**：`AccuracyAblationTest/accuracy_ablation/unified_evaluator.py`（7 916 字节，2026-08-01 01:04）

接受 baseline 与 SLG 两类 `infer_outputs_epoch_*.json`，统一输出 `{macro_f1, macro_auc, per_class, ...}` 字段。

### 1.4 Bug 0.4：`trainer.py:401-407` token-level CE

**修复标记**：
- `no canonical 7-class label to feed back.`（第 294 行注释）
- `GENERAL_REL_TO_IDX ... 7-class GenRel mapping`（第 331 行注释）
- `# --- CE Loss (7-class projection OR token-level fallback) ---`（第 571 行注释）
- `pred_logits_b = result.get("pred_logits")`（第 583 行）

`trainer.py` 已在 7-class CE 与 token-level CE 之间加入 fallback 逻辑，主路径走 7-class CE。这是 SLG-fixed 的训练目标修复点。

### 1.5 Bug 0.5：`biotriplex_finetune.py` 的 `return_neg_relations`

**修复标记**：
- `"return_neg_relations": False`（第 86、96 行，TASK_DEFAULTS 默认）
- `return_neg_relations=TASK_DEFAULTS[args.task_type]["return_neg_relations"]`（第 366、374、382 行）
- `TASK_DEFAULTS[args.task_type]` 由 `args.task_type` 决定

修复机制：通过 `--task_type` CLI 参数触发对应 `return_neg_relations=True` 的默认配置。SLG-fixed 需传入 `--task_type general_qa`（或对应配置名）。

---

## 二、阶段 1 — Baseline 对照实验（完成）

### 2.1 训练脚本
`test-data/AccuracyAblationTestData/runs/v2/run_phase1_baseline.sh`
`test-data/AccuracyAblationTestData/runs/v2/run_v2_one_experiment.sh`
`baseline/classification_genrel/scripts/run_finetune_with_epochs.sh`

### 2.2 产物（27 / 27 完成）

```
runs/v2/baseline/
├── B-se_seed{42,123,456}        # 2-target 明文起点
├── B7-se_seed{42,123,456}       # 7-target LoRA 容量税
├── B-dp00_seed{42,123,456}      # DP α=0
├── B-dp05_seed{42,123,456}      # DP α=0.05
├── B-dp15_seed{42,123,456}      # DP α=0.15
├── B-dp30_seed{42,123,456}      # DP α=0.30
├── B-dp50_seed{42,123,456}      # DP α=0.50
├── B7-dp15_seed{42,123,456}     # 7-target + DP α=0.15
└── B-ab_seed{42,123,456}        # DP α=0.15, β=0
```

每个实验产物：
- `checkpoint_epoch_{000..004}.pt`（5 个 checkpoint）
- `best_checkpoint.pt` / `last_checkpoint.pt`
- `epoch_metrics.jsonl`（5 行）
- `epoch_metrics_v2.jsonl`（5 行 unified_evaluator 输出）
- `adapter_config.json` + `adapter_model.safetensors`（LoRA 适配器）
- `logs/epoch_*_evaluate_metrics.json`（5 个）
- `KEY_EVENTS.log` + `SUMMARY.md` + `README.md`

### 2.3 训练时间
- 启动：2026-08-01 01:20:31
- 完成：2026-08-01 09:39:13
- 阶段 1 实际耗时 ≈ **8h 19min**（27 次 × 5 epochs ≈ 18min/次）

### 2.4 已知 WARN
```
[phase1.5] WARN: .../B-se_seed42_seed42 not found, using default .../B-se_seed42
```
阶段 1.5 在 baseline 路径命名歧义时回退到 seed=42。不影响产物。

---

## 三、阶段 1.5 — 中间值量化档位 Ablation（完成）

### 3.1 训练脚本
`test-data/AccuracyAblationTestData/runs/v2/run_phase1_5_ablation.sh`
`test-data/AccuracyAblationTestData/runs/v2/v2_ablation_quantize.py`

注意：阶段 1.5 **不重训**，复用阶段 1 的 `B-se_seed42` LoRA，对 infer_outputs 加量化噪声后评估。

### 3.2 产物（27 / 27 完成）

```
runs/v2/ablation/
├── B-q-clean_seed{42,123,456}         # 无量化基线
├── B-q-q1-v-only_seed{42,123,456}     # 仅 V 量化
├── B-q-q2p-v-and-g_seed{42,123,456}   # V+G 量化
├── B-q-q2-with-proto_seed{42,123,456} # 量化 + 协议近似
├── B-q-q3-bf16_seed{42,123,456}       # bf16 注入模拟
├── B-q-s10k-bf16_seed{42,123,456}     # scale=10000 + bf16（默认）
├── B-q-s10k-fp32_seed{42,123,456}     # scale=10000 + 跳过 bf16
├── B-q-s100k-bf16_seed{42,123,456}    # scale=100000 + bf16
└── B-q-s100k-fp32_seed{42,123,456}    # scale=100000 + 跳过 bf16

ablation/
├── summary_seed42.json
├── summary_seed123.json
└── summary_seed456.json
```

每个实验产物：
- `epoch_metrics_v2.jsonl`（5 行）
- `logs/epoch_*_evaluate_metrics.json`（5 个）

### 3.3 训练时间
- 启动：2026-08-01 09:39:13
- 完成：2026-08-01 09:39:22
- 阶段 1.5 实际耗时 ≈ **9 s**（纯评估复用 LoRA，无需训练）

---

## 四、阶段 3 — gold-only 协议对照（**跳过**）

### 4.1 用户决策

用户答复（2026-08-01）：
> 跳过阶段 3，直接进入阶段 4（简化路径，把 gold-only 留给阶段 5 报告注解）

### 4.2 技术原因
gold-only 是 **SLG 协议里**的约束（`a_t - V_y`），在 baseline 明文路径上**无对应实现**。强行构造会引入非可比对照物，污染 OFAT 假设。

### 4.3 处理方案
- **不新增训练**。
- **阶段 5 报告**（精度损失分解）中说明：「gold-only 税的实测需要阶段 4 的 SLG-fixed 与 B-se / B-dp15 比较外推；本次实验以 B-ab（β=0）作为 gold-only 部分关闭的近似参考点」。
- 在 `final_report.md` 中加入一节 `gold_only_effect = N/A (extrapolated)` 并明确数据来源。

---

## 五、阶段 4 — SLG-fixed 重训（**待跑**）

### 5.1 实验设计

| 实验编号 | 配置 | epochs | seeds | 总训练次数 |
|---------|------|--------|-------|-----------|
| **SLG-fixed** | 修复 mode collapse（7-class CE）+ return_neg_relations=True + scale=10000 + g_H=bf16 (default) | 10 | 3 | **30 次** |

### 5.2 修复清单

1. **trainer.py:401-407** → 7-class CE（Bug 0.4 已修）
2. **biotriplex_finetune.py** → `--task_type general_qa` 触发 `return_neg_relations=True`（Bug 0.5 已修）
3. **train_loss_proxy** → 打印真实 7-class CE 替代 g_H 范数（部分修复）
4. **scale=10000** + **g_H=bf16**（与原 SLG 默认一致）

### 5.3 执行策略

**Smoke → 全量**（用户决策）：

1. **Smoke（1 种子 = 42，10 epochs，~25 min）**
   - 验证 trainer.py 7-class CE 修复后能跑通、loss 不发散
   - 验证产物 `epoch_metrics.jsonl` 与 `epoch_metrics_v2.jsonl` 形态正确
   - 验证 `pred_logits` 路径正确（不再 fallback 到 token-level CE）
   - 验证 `n_parse_failures = 0`、y_pred 分布不集中

2. **Smoke 通过后再跑全量（3 种子 × 10 epochs = 30 次训练，~2.5h）**

### 5.4 调度策略

- **独立后台进程**：`setsid + nohup + 完整 I/O 重定向`，与 SSH session 隔离
- **新启动器**：`run_phase4_full_background.sh start|status|stop|tail|last20`
- **PID 文件**：`runs/v2/.phase4_runner.pid`
- **日志文件**：`runs/v2/phase4_runner.log`
- **退出准则**：`SLG-fixed_seed{42,123,456}` 三个实验目录齐全 + 各自含 10 个 `epoch_*_evaluate_metrics.json` + 整体 phase4 exit code = 0

### 5.5 风险与回滚

| 风险 | 检测信号 | 回滚动作 |
|------|----------|----------|
| 7-class CE 修复后训练发散 | `train_loss` 单调上升 > 1 epoch | 暂停 → 复审 trainer.py:571 段逻辑 |
| pred_logits 为 None fallback 到 token-level | SUMMARY.md 出现「token-level fallback」 | 暂停 → 检查 unified_evaluator 链路 |
| OOM（g_H=fp32 路径） | nvidia-smi OOM 日志 | 确认用 bf16（默认）即可 |
| 模式仍崩塌（67% modulatory） | y_pred_distribution 中 modulatory > 50% | 暂停 → 评估 return_neg_relations 链路 |
| 后台进程意外死亡 | PID 文件 stale | 自动 fail → 重启 |

---

## 六、阶段 5 — 精度损失分解报告（**未启动**）

依赖阶段 4 完成。预期产出：
- `AccuracyAblationTest/docs/final_metrics.json`（6 个边际贡献 + mean±std + n）
- `AccuracyAblationTest/docs/final_report.md`（柱状图 + 6 个分解项 + 显著性检验）
- `gold-only 税` 用「B-ab vs B-se」（β=0 ≈ gold-only 部分关闭）做近似注解

---

## 七、关键路径与文件清单

```
/root/autodl-tmp/CipherForgeCode/SLG-HE-PIR/
├── AccuracyAblationTest/
│   ├── accuracy_ablation/
│   │   ├── eval_replay.py                 # Bug 0.1 已修
│   │   ├── quant_hooks.py                 # Bug 0.2 已修
│   │   └── unified_evaluator.py           # Bug 0.3 新建
│   └── docs/
│       ├── 精度测试文档.md                  # 814 行，含 ASCII 图
│       ├── intermediate_state_phase0_4.md # 本文档
│       ├── final_report.md                # 阶段 5 待产出
│       └── final_metrics.json             # 阶段 5 待产出
├── baseline/classification_genrel/scripts/
│   └── run_finetune_with_epochs.sh        # 训练入口
├── src/
│   ├── training/trainer.py                # Bug 0.4 已修
│   ├── scripts/biotriplex_finetune.py     # Bug 0.5 已修
│   └── parties/party_m.py                 # gold-only / g_H 注入点
└── test-data/AccuracyAblationTestData/runs/v2/
    ├── _v2_internal_runner.sh             # 阶段 1 + 1.5 调度
    ├── run_phase1_baseline.sh             # 阶段 1
    ├── run_phase1_5_ablation.sh           # 阶段 1.5
    ├── run_v2_one_experiment.sh           # 单次实验 wrapper
    ├── run_v2_full_background.sh          # 后台启动器
    ├── baseline/                          # 27 个实验目录
    ├── ablation/                          # 27 个实验目录
    ├── v2_runner.log                      # 阶段 1 + 1.5 日志
    └── .v2_runner.pid                     # 已 stale（任务 DONE）
```

---

## 八、当前状态：执行阶段 4 准备

- **GPU**：RTX 5090, 0 %, 0 MiB（空闲）
- **后台进程**：仅有 pts/2 上 `tail -f v2_runner.log`（73093, 73096），非训练
- **下一步动作**：
  1. 编写 `run_phase4_slg.sh`（SLG-fixed 单次实验 wrapper）
  2. 编写 `run_phase4_full_background.sh`（独立 session 启动器）
  3. 启动 smoke → 等待完成 → 审查产物 → 启动全量
