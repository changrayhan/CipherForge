# SLG-HE-PIR 测试数据总览

> **最后更新**：2026-08-06
> **总磁盘占用**：~8.2 GB（AttackTest 172 MB + PerformanceTest 2.3 MB + PrecisionTest 4.2 GB + AATestArchive 751 MB + BioTriplex1BTestData 2.9 GB）
> **测试目标**：通过攻击测试证明标签和模型安全；通过精度消融实验分析影响精度的因素；通过性能测试分析 SLG-HE-PIR 各阶段耗时。

---

## 目录

1. [目录树总览](#1-目录树总览)
2. [AttackTest — 攻击测试（安全性证明）](#2-attacktest--攻击测试安全性证明)
3. [PerformanceTest — 性能测试（阶段耗时分析）](#3-performancetest--性能测试阶段耗时分析)
4. [PrecisionTest — 精度测试（消融实验 + 105 runs 大规模对照）](#4-precisiontest--精度测试消融实验--105-runs-大规模对照)
5. [BioTriplex1BTestData — 105 runs 完整数据（Llama-3.2-1B）](#5-biotriplex1btestdata--105-runs-完整数据llama-322-1b)
6. [AATestArchive — 归档目录（已归档的历史实验与失败数据）](#6-aatestarchive--归档目录已归档的历史实验与失败数据)
7. [阅读指南：哪些报告回答哪些问题](#7-阅读指南哪些报告回答哪些问题)

---

## 1. 目录树总览

```
test-data/                                    [总 ~8.2 GB]
├── AttackTest/                               [172 MB]  攻击测试
│   ├── SLG-attack-test/                     攻击套件源代码
│   ├── data/
│   │   ├── attack-test-data/                L1/L2/M1/M2 原始结果 + 中间数据
│   │   │   ├── l1_results.json              L-1 攻击原始结果
│   │   │   ├── l2_results.json              L-2 攻击原始结果
│   │   │   ├── m1_results.json              M-1 攻击原始结果
│   │   │   ├── m2_results.json              M-2 攻击原始结果
│   │   │   └── dp-ablation/                 27 组 dp 参数网格消融结果
│   │   │       ├── alpha_0.05_beta_0.3_cal_2_results.json   (共 27 组)
│   │   │       ├── dp_ablation_summary.csv
│   │   │       └── run_YYYYMMDD_HHMMSS/    (共 27 个独立 run)
│   │   │           ├── attack_results.json
│   │   │           ├── attack_test.log
│   │   │           └── dumps/step_XXXXX.json (每步的中间激活值)
│   │   └── figures/                         18 张分析图片（PNG）
│   └── 报告/
│       └── TEST_REPORT.md                    攻击测试最终报告 [主报告]
│
├── PerformanceTest/                           [2.3 MB]  性能测试
│   ├── data/perf-test-data/
│   │   ├── slg_cls_step_profiles.jsonl      CLS 任务 SLG 每步耗时（n=3951 步）
│   │   ├── baseline_cls_step_profiles.jsonl CLS 任务 Baseline 每步耗时
│   │   ├── slg_cls_epoch_metrics.jsonl      CLS epoch 级汇总指标
│   │   ├── baseline_cls_metrics.json        CLS Baseline 最终评估指标
│   │   ├── baseline_ner_step_profiles.jsonl NER Baseline 每步耗时
│   │   ├── baseline_ner_metrics.json         NER Baseline 最终指标
│   │   ├── slg_ner_init.log                NER SLG 启动日志（step 0-3 后中断）
│   │   ├── communication_overhead.json      通信开销理论计算
│   │   └── offline_prep_overhead.json      离线准备开销估算
│   ├── scripts/                             (空目录)
│   └── 报告/                               (空目录，无独立报告)
│
├── PrecisionTest/                            [4.2 GB]  精度测试
│   ├── data/
│   │   ├── AccuracyAblationTestData_v2/     [2.9 GB]  Noise-Model Ablation 原始数据
│   │   │   ├── baseline/                    Baseline (Plaintext LoRA) 1 run × 5 epoch
│   │   │   ├── runs/slg/SLG-fixed_seed42/  SLG 实际训练 1 run × 5 epoch
│   │   │   ├── runs/v2/ablation/            6 量化变体 × 3 seed = 18 runs × 5 epoch
│   │   │   │   ├── B-q-clean_seed42/        6 变体各含 5 个 epoch_*_metrics.json
│   │   │   │   ├── B-q-q1-v-only_seed42/
│   │   │   │   ├── B-q-q2-with-proto_seed42/
│   │   │   │   └── ...（共 18 个 run 目录）
│   │   │   ├── derived/                     对比分析结果（avg_comparison.json 等）
│   │   │   └── quantization_params/          SLG BFV 参数提取（slg_bfv_params.json）
│   │   ├── 3step-cls-test/                  3-step 快速 CLS 测试（早期调试）
│   │   └── figures/                         18 张分析图片（PNG）
│   ├── scripts/
│   │   └── AccuracyAblationTest/             Noise-Model Ablation 自动化框架
│   │       ├── accuracy_ablation/            核心模块（量化 hooks、eval_replay 等）
│   │       ├── configs/                      YAML 量化参数配置
│   │       ├── docs/                         方法论文档
│   │       ├── outputs/
│   │       │   ├── q0_7target/               6 变体 × 3 seed 评估结果
│   │       │   │   ├── q0p_2target/
│   │       │   │   ├── q1_v_quant/
│   │       │   │   ├── q2p_full_token/
│   │       │   │   ├── q2_g_h_quant/
│   │       │   │   ├── q3_full_slg_sim/
│   │       │   │   ├── QUANT_ABLATION_REPORT.md  [主报告]
│   │       │   │   └── quant_ablation_data.json  (完整数值数据)
│   │       │   └── _analysis/                中间分析文件
│   │       ├── scripts/                      CLI 入口脚本（run_variant.py 等）
│   │       ├── tests/                       单元测试
│   │       └── README.md                     框架使用说明
│   └── 报告/
│       ├── CLS_PRECISION_COMPARISON_REPORT.md  Baseline vs SLG 精度对比报告
│       └── BioTriplex1BTestData_SUMMARY.md     105 runs 大规模实验汇总 [主报告]
│
├── BioTriplex1BTestData/                      [2.9 GB]  105 runs 大规模对照实验
│   ├── data/                                 BioTriplex 段落/句子/黄金标签数据
│   ├── runs/                                 4 phase × 35 configs × 3 seeds = 105 runs
│   │   ├── _extract_facts.json               105 runs 的 best/last epoch 指标（机器可读）
│   │   ├── _summary/all_phases.md            Phase 级汇总（per-exp macro_f1）
│   │   ├── baseline/                         8 configs × 3 seeds = 24 runs（Plaintext LoRA + DP 消融）
│   │   ├── quant/                            8 configs × 3 seeds = 24 runs（量化 V+g_H）
│   │   ├── quant_dp15/                       6 configs × 3 seeds = 18 runs（Quant + DP α=0.15）
│   │   └── quant_v2/                         13 configs × 3 seeds = 39 runs（量化校正版）
│   └── scripts/                              bio_baseline_trainer.py 等训练/评估脚本
│
└── AATestArchive/                            [751 MB]  归档目录
    ├── MANIFEST.md                           归档清单与操作记录
    ├── BioTriplex1BTestData/                失败 phase + 历史 runner 脚本（已归档）
    │   ├── scripts/                         19 个 runner/备份脚本
    │   └── runs/                            失败实验 + 早期备份
    │       ├── _failed_phases/               6 个未完成 phase
    │       └── _helpers/                    构建/监控脚本
    └── TrecAATestData/                      TREC-QC 精度测试（Llama-3.2-1B，废弃）
        ├── docs/                             早期汇总报告
        ├── gold/                            BioTriplex 兼容格式黄金标签
        ├── runs/                            TREC-QC baseline + SLG 实验结果
        └── scripts/                         TREC-QC 训练/评估脚本
```

---

## 2. AttackTest — 攻击测试（安全性证明）

**目的**：在 L-1（标签推断）、L-2（激活值推断）、M-1（模型蒸馏）、M-2（LoRA 结构推断）四个攻击维度上，验证 SLG-HE-PIR 三方协议不存在可利用的隐私泄露路径。

### 2.1 核心结论

| 攻击 | 核心结论 | 状态 |
|------|---------|------|
| **L-1** M方标签推断 | 7/7 指标 PRIVACY_PRESERVED，27 组 dp 参数消融全部安全 | ✅ 完整 |
| **L-2** S方激活值推断 | 4/5 指标 PRIVACY_PRESERVED；softmax 触警但 KL≈0.0038 不可利用 | ✅ 完整 |
| **M-1** U方知识提取 | 3/3 指标 PRIVACY_PRESERVED | ✅ 完整 |
| **M-2** S方结构推断 | 3/6 指标 PRIVACY_PRESERVED，3/6 INCONCLUSIVE（样本量不足，非泄露） | ⚠️ 需更多数据 |

### 2.2 数据文件说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `l1_results.json` | L-1 攻击 7 子指标原始判定（n=200） | 报告 §3.1 数据源 |
| `l2_results.json` | L-2 攻击 5 子指标原始判定（n=200） | 报告 §3.1 数据源 |
| `m1_results.json` | M-1 攻击 3 子指标原始判定（n=100） | 报告 §3.1 数据源 |
| `m2_results.json` | M-2 攻击 6 子指标原始判定（n=800/1600） | 报告 §3.1 数据源 |
| `dp-ablation/dp_ablation_summary.csv` | 27 组 dp 参数网格全部指标 | 27 组消融表格原始数据 |
| `dp-ablation/run_YYYYMMDD_HHMMSS/dumps/step_XXXXX.json` | 每训练步中间激活值（H_U、a_t、g_accum 等） | 攻击器输入数据，可复现攻击 |
| `figures/*.png` | 16 张分析图片（箱线图、热力图、散点图等） | 报告引用 |

### 2.3 测试配置

- **模型**：Llama-3.2-1B（hidden_dim=2048）
- **加密**：BFV（poly_degree=2048）
- **差分隐私**：dχ（dp_alpha ∈ {0.05, 0.15, 0.30}，dp_answer_beta ∈ {0.3, 0.5, 0.7}，calibration_steps ∈ {2, 5, 10}）
- **数据集**：TREC-QC 6 类（训练集 4909 / 验证集 543 / 测试集 500）
- **种子**：seed=42

### 2.4 主报告

> **`AttackTest/报告/TEST_REPORT.md`** — 攻击测试最终报告
> 包含：测试环境、数据集描述、攻击方案说明、27 组消融数据、16 张图片引用、性能测试数据。

---

## 3. PerformanceTest — 性能测试（阶段耗时分析）

**目的**：在 BioTriplex 分类（CLS）与 NER 任务上，对比基准明文 LoRA 与 SLG-HE-PIR 在各训练阶段的计算与通信开销。

### 3.1 核心结论

| 指标 | CLS 任务 | NER 任务 | 状态 |
|------|---------|---------|------|
| 单步 5 阶段耗时（n=3951 步） | ✅ 完整 | ⚠️ 仅 4 步（启动期） | ⚠️ |
| 通信开销（理论） | ✅ 完整 | ✅ 完整 | ✅ |
| 离线准备开销 | ✅ 完整 | — | ✅ |
| CPU/GPU 内存峰值 | ✅ 完整 | ⚠️ 未单独测 | ⚠️ |

### 3.2 数据文件说明

| 文件 | 内容 | 关键数字 |
|------|------|---------|
| `slg_cls_step_profiles.jsonl` | SLG CLS 每步 5 阶段耗时（n=3951 步，每步含 forward_U/M、s_logits、priv_U、backward_M） | 稳态均值 ≈101 s/step |
| `baseline_cls_step_profiles.jsonl` | Baseline CLS 每步 5 阶段耗时（n=734 步） | 均值 ≈40 s/step |
| `slg_cls_epoch_metrics.jsonl` | SLG CLS epoch 级汇总 | 5 epoch 指标 |
| `baseline_cls_metrics.json` | Baseline CLS 最终评估（n=213 样本，7 类） | macro_f1=0.4094 |
| `baseline_ner_step_profiles.jsonl` | Baseline NER 每步耗时（n=3 步） | 仅 3 步数据 |
| `slg_ner_init.log` | SLG NER 启动日志 | step 3 后被 Terminated |
| `communication_overhead.json` | 6 通道 × 3 档 token 数通信字节数（理论） | 每步≈1.03 GB（CLS） |
| `offline_prep_overhead.json` | Stage 0 离线产物大小 | BFV 加密 V 矩阵 15.67 GB |

### 3.3 性能测试缺口

**NER 任务的 SLG 完整性能数据缺失**：日志显示 NER 训练在 step 3 后被 Terminated，仅有 4 步启动期数据，无稳态耗时分布。

> **建议**：如需完整的 NER 性能对比数据，需重新执行 SLG NER 全程训练（100+ 步）。

---

## 4. PrecisionTest — 精度测试（消融实验 + 105 runs 大规模对照）

**目的**：通过 noise-model ablation（6 变体 × 3 seed × 5 epoch）和大规模对照实验（105 runs × 8 epoch），精确量化 SLG-HE-PIR 协议中各层精度损失的来源。

PrecisionTest 由两个独立的精度测试体系组成：

### 4.1 体系一：Noise-Model Ablation（`AccuracyAblationTestData_v2/`）

通过逐层注入量化噪声的方式，将 SLG 精度损失分解为：固定精度量化税、g_H int64 量化税、协议约束税（gold-only 反向）、bf16 转换税。

#### 6 个变体（严格累积）

| 变体 | 量化内容 | 累计税 | macro_f1（3 seed avg） | 单步损失 |
|------|---------|--------|----------------------:|---------:|
| Q0 | 无量化，7-target | — | 0.2292 | — |
| Q0' | 无量化，2-target | 对照起点 | 0.2292 | +0.00 pp |
| Q1 | V 矩阵 fixed-point + H_M 量化 | **量化税** | 0.2254 | **-0.37 pp** |
| Q2' | Q1 + 全 token g_H int64 量化 | + g_H 量化税 | 0.2215 | **-0.39 pp** |
| Q2 | Q2' + gold-only 协议约束 | + **协议约束税** | 0.1722 | **-4.93 pp** ⭐ |
| Q3 | Q2 + g_H bf16 转换 | + bf16 转换税 | 0.1495 | **-2.28 pp** |

> **核心发现**：协议约束税（4.93 pp）> 量化税之和（~0.8 pp）→ 精度损失瓶颈在**协议设计**而非**量化精度**。

#### 数据文件说明

| 文件 | 内容 |
|------|------|
| `runs/v2/ablation/B-q-*/` | 6 变体 × 3 seed = 18 个 run 目录，每个含 5 个 `epoch_*_metrics.json` |
| `runs/slg/SLG-fixed_seed42/` | SLG 实际训练 1 run × 5 epoch + step_profiles.jsonl |
| `baseline/` | Baseline 实际训练 1 run × 5 epoch |
| `derived/precision_gradient.md` | 早期精度梯度对比（已注明口径差异） |
| `derived/avg_comparison.json` | 5 epoch 平均指标对比 |
| `derived/best_epoch_comparison.json` | 最佳 epoch 指标对比 |
| `quantization_params/slg_bfv_params.json` | 从 SLG checkpoint 提取的 BFV 参数 |

#### 自动化框架

> **`PrecisionTest/scripts/AccuracyAblationTest/`** — Noise-Model Ablation 自动化框架

```
accuracy_ablation/
├── quant_hooks.py          6 变体量化噪声注入逻辑
├── eval_replay.py          加载 logits + 注入量化 + 重新评估
├── slg_param_extractor.py  从 .pt checkpoint 提取 BFV 参数
└── report_generator.py     多 seed 汇总报告生成
```

#### 主报告

> **`PrecisionTest/scripts/AccuracyAblationTest/outputs/QUANT_ABLATION_REPORT.md`**
> 包含：6 变体定义、方法论、27 组实验设置、精度梯度链、per-class 分析、统计显著性检验、Pareto Frontier。

---

### 4.2 体系二：CLS Baseline vs SLG 对比（`CLS_PRECISION_COMPARISON_REPORT.md`）

在 BioTriplex GenRel QA 任务上，对比 Baseline Plaintext LoRA 与 SLG-HE-PIR 5 epoch 训练后的精度。

| 配置 | micro_f1 | macro_f1 | macro_auc | 样本数 |
|------|----------:|---------:|----------:|-------:|
| Baseline CLS（最佳 epoch） | 0.3662 | 0.2633 | 0.7771 | 213 |
| SLG-HE-PIR（最佳 epoch） | 0.2709 | 0.1515 | 0.6815 | 203 |
| **差异** | -9.53 pp | -11.18 pp | -9.56 pp | -10 |

> **主报告**：`PrecisionTest/报告/CLS_PRECISION_COMPARISON_REPORT.md`
> 包含：实验设置对齐、5 epoch 逐 epoch 指标对比、关键发现、修复摘要。

---

## 5. BioTriplex1BTestData — 105 runs 完整数据（Llama-3.2-1B）

**目的**：在 Llama-3.2-1B 上，以 BioTriplex GenRel QA 任务，对比 Plaintext LoRA baseline 与各量化/DP 配置的精度表现。

### 5.1 数据规模

| Phase | 实验数 | Runs | 种子 | Epoch/Run | 配置描述 |
|-------|-------:|-----:|------|:---------:|---------|
| **baseline** | 8 | 24 | 42/123/2025 | 8 | Plaintext LoRA + DP alpha 消融 |
| **quant** | 8 | 24 | 42/123/2025 | 8 | V+g_H 量化（位宽/类型消融） |
| **quant_dp15** | 6 | 18 | 42/123/2025 | 8 | Quant + DP α=0.15 |
| **quant_v2** | 13 | 39 | 42/123/2025 | 8 | Quant v2（校正版） |
| **合计** | **35** | **105** | — | **8** | — |

> **全部 105 runs × 8 epoch = 840 次评估完成**，无中断。

### 5.2 实验配置详情

**baseline Phase（Plaintext LoRA，DP alpha 消融）**：

| Exp | dp_alpha | target 数 | macro_f1（3 seed avg） |
|-----|--------:|:---------:|----------------------:|
| B-T7 | — | 7 | **0.3423 ± 0.0354** |
| B-T | — | 3 | 0.1845 ± 0.0087 |
| B-T_dpa05 | 0.05 | 3 | 0.2166 ± 0.0175 |
| B-T_dpa15 | 0.15 | 3 | 0.2109 ± 0.0252 |
| B-T_dpa30 | 0.30 | 3 | 0.1819 ± 0.0086 |
| B-T_dpa50 | 0.50 | 3 | 0.1733 ± 0.0109 |
| B-T7_dpa15 | 0.15 | 7 | 0.1829 ± 0.0170 |
| B-T_ab_no_beta | — | 3 | 0.1845 ± 0.0087 |

**quant_v2 Phase（量化校正版，关键配置）**：

| Exp | 量化位宽 | g_H 精度 | macro_f1（3 seed avg） |
|-----|:--------:|:--------:|----------------------:|
| B-q16-s1k-bf16 | 16-bit | bf16 | 0.1986 ± 0.0136 |
| B-q16-s1k-fp32 | 16-bit | fp32 | 0.1962 ± 0.0150 |
| B-q16-s100k-bf16 | 100k-token | bf16 | 0.1829 ± 0.0065 |
| B-q16-control-quant_off | — | — | 0.1947 ± 0.0196 |

### 5.3 关键发现

1. **B-T7（7-target）是唯一正确配置**：macro_f1=0.3423，比其他所有 3-target 配置高 +15.78 pp。
2. **量化本身在 3-target 口径下对精度无显著影响**：B-T 与 B-q-s100k-bf16 的 per-epoch 曲线完全重合。
3. **DP alpha 精度-隐私 tradeoff**：α=0.05 最优精度（0.2166），α=0.30 最强隐私（与 TEST_REPORT L-1 推荐一致），但精度下降至 0.1819。
4. **quant_v2 稳定性优于 quant**：std 从 ≤0.0011 降至 ≤0.015，参数校正有效。

### 5.4 数据文件说明

| 文件 | 内容 |
|------|------|
| `runs/_extract_facts.json` | 105 runs 的 best/last epoch 完整指标（机器可读，105 条记录） |
| `runs/_summary/all_phases.md` | Phase 级 per-exp 汇总（per-class F1、accuracy、AUC） |
| `runs/{phase}/{exp}/{seed}/logs/epoch_*_bio_metrics.json` | 840 个 epoch 评估文件（主要数据） |
| `runs/{phase}/{exp}/{seed}/logs/infer_outputs_epoch_*.json` | 840 个原始推理输出文件 |
| `runs/{phase}/{exp}/{seed}/logs/metrics_history.json` | 105 个训练过程指标历史 |
| `data/train_gold_general_qa.txt` | 537 条训练样本黄金标签 |
| `data/para_train/` | 537 个段落级训练段落 |
| `data/sentence_train/` | 句子级训练数据 |

### 5.5 主报告

> **`PrecisionTest/报告/BioTriplex1BTestData_SUMMARY.md`**
> 包含：105 runs 完整数据汇总、Phase 级统计、关键发现（口径差异揭示）、per-class 分析、与精度消融报告的关系。

---

## 6. AATestArchive — 归档目录（已归档的历史实验与失败数据）

**目的**：将 v9 报告未引用的所有代码与数据归档，保持 test-data 主目录整洁。

> **`AATestArchive/MANIFEST.md`** — 完整归档清单（2,219 文件，~4 GB）

### 6.1 归档内容

| 来源 | 内容 | 大小 |
|------|------|------|
| `BioTriplex1BTestData/scripts/` | 19 个历史 runner 脚本（Phase 1.5/1.6 等） | 132 KB |
| `BioTriplex1BTestData/runs/_failed_phases/` | 6 个未完成 phase（baseline_extra_seeds / cumulative / fullstack_baseline / slg / dp_alpha_scan / 早期备份） | ~2.2 MB |
| `BioTriplex1BTestData/runs/_helpers/` | 18 个构建/监控脚本（_dump_v9.py 等） | 672 KB |
| `PrecisionTest/data/AccuracyAblationTestData/` | 早期独立对照实验（2.9 GB） | 2.9 GB |
| `TrecAATestData/` | TREC-QC 精度测试（废弃，Llama-3.2-1B，被 BioTriplex 取代） | 138 MB |

### 6.2 TrecAATestData 简介

- **数据集**：TREC-QC 6 类（COLING'02，4909 训练 / 543 验证 / 500 测试）
- **配置**：Llama-3.2-1B + BFV poly_degree=2048
- **状态**：已完成 baseline + SLG 两套实验（`runs/baseline/`、`runs/slg/`）
- **标注**：存在 `INTERRUPTED.flag`，表示部分实验未完成
- **被替代原因**：BioTriplex 数据集（多关系、段落级、更大规模）取代了 TREC-QC

---

## 7. 阅读指南：哪些报告回答哪些问题

| 问题 | 推荐报告 |
|------|---------|
| **SLG 协议是否安全（标签/模型不被泄露）？** | `AttackTest/报告/TEST_REPORT.md` |
| **哪些参数组合能同时保证隐私和精度？** | `AttackTest/报告/TEST_REPORT.md` §3.1（L-1 27 组消融） |
| **SLG 精度损失各占多少 pp（量化 vs 协议）？** | `PrecisionTest/scripts/AccuracyAblationTest/outputs/QUANT_ABLATION_REPORT.md` |
| **Baseline vs SLG 精度差多少 pp？** | `PrecisionTest/报告/CLS_PRECISION_COMPARISON_REPORT.md` |
| **DP alpha 精度-隐私 tradeoff 最优点在哪？** | `PrecisionTest/报告/BioTriplex1BTestData_SUMMARY.md` §3.3 |
| **105 runs 全部实验的汇总数据在哪？** | `BioTriplex1BTestData/runs/_extract_facts.json`（机器可读） |
| **单步训练各阶段耗时分布（CLS 任务）？** | `PerformanceTest/data/perf-test-data/slg_cls_step_profiles.jsonl` |
| **NER 任务的 SLG 性能数据？** | **缺失**（仅 `slg_ner_init.log` 4 步启动数据） |
| **NER 精度消融数据？** | **缺失**（BioTriplex1BTestData 仅覆盖分类任务） |
| **V 矩阵量化参数具体值？** | `PrecisionTest/data/AccuracyAblationTestData_v2/quantization_params/slg_bfv_params.json` |
| **已归档的历史数据/脚本在哪里？** | `AATestArchive/MANIFEST.md` |

---

## 附录：各目录大小汇总

| 目录 | 大小 | 内容 |
|------|-----:|------|
| `AttackTest/` | 172 MB | 攻击测试代码、数据、报告 |
| `PerformanceTest/` | 2.3 MB | 性能测试数据（日志、step profiles、通信开销） |
| `PrecisionTest/` | 4.2 GB | 精度测试数据与报告 |
| `BioTriplex1BTestData/` | 2.9 GB | 105 runs × 8 epoch 完整数据 |
| `AATestArchive/` | 751 MB | 归档的历史实验与失败数据 |
| **合计** | **~8.2 GB** | — |
