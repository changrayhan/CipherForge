# TREC-QC 精度测试目录

TREC-QC 6-class 短文本分类精度测试 (Llama-3.2-1B)。

## 目录结构

```
TrecAATestData/
├── README.md                          (本文件)
├── docs/                              报告与汇总
│   ├── trec_summary.md               (阶段 1+4 完成后的最终报告)
│   ├── trec_summary.json             (机器可读汇总)
│   └── migration_notes.md            (迁移方案笔记)
├── gold/                              BioTriplex 兼容格式的金标准
│   ├── test_gold_general_qa.txt      (500 样本)
│   ├── val_gold_general_qa.txt       (543 样本)
│   └── train_gold_general_qa.txt     (4909 样本)
├── runs/
│   ├── baseline/                      阶段 1 (PyTorch plaintext)
│   │   ├── B-T_se_2target/seed42/    (LoRA q,v)
│   │   ├── B-T7_se_7target/seed42/   (LoRA 7 modules)
│   │   ├── B-T_dpa00/seed42/         (DP alpha=0)
│   │   ├── B-T_dpa05/seed42/         (DP alpha=0.05)
│   │   ├── B-T_dpa15/seed42/         (DP alpha=0.15)
│   │   ├── B-T_dpa30/seed42/         (DP alpha=0.30)
│   │   ├── B-T_dpa50/seed42/         (DP alpha=0.50)
│   │   ├── B-T7_dpa15/seed42/        (7-target + DP alpha=0.15)
│   │   ├── B-T_ab_no_beta/seed42/    (alpha=0.15, beta=0)
│   │   └── _runner_logs/
│   └── slg/                           阶段 4 (SLG 加密)
│       └── SLG-T_dpa15/seed<seed>/   (SLG-fixed, DP alpha=0.15)
└── scripts/
    ├── trec_gold.py                   生成 gold 文件
    ├── trec_baseline_trainer.py       阶段 1 训练 (PyTorch + bf16 + LoRA)
    ├── trec_evaluator.py              6-class TREC-QC 评估器
    ├── trec_summarize.py              汇总所有实验输出
    ├── _trec_internal_runner.sh       内部调度
    ├── run_trec_full_background.sh    后台启动器
    ├── run_trec_baseline.sh           阶段 1 入口
    ├── run_trec_slg.sh                阶段 4 入口
    └── run_trec_one_experiment.sh     单实验调试入口
```

## 任务定义

### 数据集
TREC-QC (Li & Roth, COLING'02) — 6 个粗粒度类别的问题分类:
- 0=DESC, 1=ENTY, 2=ABBR, 3=HUM, 4=NUM, 5=LOC
- 总样本 5,952 (train 4909 + val 543 + test 500)
- 文本长度 avg=51 chars / max=196 chars

### 模型
Llama-3.2-1B (`models--unsloth--Llama-3.2-1B`)
- hidden_size=2048, num_hidden_layers=16, num_kv_heads=8
- vocab_size=128256 (config), 实际 token 词表 128000 (Llama-3 tokenizer)

### 训练配置
- LoRA rank=8, alpha=16, dropout=0.05
- bf16, batch=8, lr=1e-4, weight_decay=0
- max_seq_length=256 (TREC-QC 文本极短)
- 5 epochs (baseline), 10 epochs (SLG)

## 实验矩阵

### 阶段 1 (Baseline 对照实验)
9 个实验 × 3 个 seeds × 5 epochs:

| 实验 | LoRA target | DP α | DP β | 目的 |
|---|---|---|---|---|
| B-T_se_2target | q_proj,v_proj | 0.0 | 0.5 | 起点 baseline |
| B-T7_se_7target | 7 modules | 0.0 | 0.5 | LoRA 容量 |
| B-T_dpa00 | q_proj,v_proj | 0.00 | 0.5 | DP 基线 |
| B-T_dpa05 | q_proj,v_proj | 0.05 | 0.5 | DP 中等 |
| B-T_dpa15 | q_proj,v_proj | 0.15 | 0.5 | **SLG 默认 DP** |
| B-T_dpa30 | q_proj,v_proj | 0.30 | 0.5 | DP 强 |
| B-T_dpa50 | q_proj,v_proj | 0.50 | 0.5 | DP 极强 |
| B-T7_dpa15 | 7 modules | 0.15 | 0.5 | 容量+DP |
| B-T_ab_no_beta | q_proj,v_proj | 0.15 | 0.0 | ablation: 答案 token 比例 |

### 阶段 4 (SLG-Fixed 加密)
1 个实验 × 3 个 seeds × 10 epochs:
- SLG-T_dpa15: 重建 BFV DB d=2048, 6-class CE

## 使用方式

### 一键启动 (后台)
```bash
cd /root/autodl-tmp/CipherForgeCode/SLG-HE-PIR
# 阶段 1 (baseline)
bash test-data/TrecAATestData/scripts/run_trec_baseline.sh start
# 阶段 4 (SLG)
bash test-data/TrecAATestData/scripts/run_trec_slg.sh start
```

### 检查状态
```bash
bash test-data/TrecAATestData/scripts/run_trec_full_background.sh status
```

### 汇总报告
```bash
python3 test-data/TrecAATestData/scripts/trec_summarize.py \
    --runs_root test-data/TrecAATestData/runs \
    --output_dir test-data/TrecAATestData/docs
```

### 单实验调试
```bash
bash test-data/TrecAATestData/scripts/run_trec_one_experiment.sh B-T_dpa15 "q_proj,v_proj" 0.15 0.5 42
```

## 与 BioTriplex 阶段的对比

| 维度 | BioTriplex | TREC-QC |
|---|---|---|
| 类别数 | 7 (含 no relation) | 6 (clean labels) |
| 文本长度 | avg 1020 chars | avg 51 chars |
| 类别分布 | 长尾 67/51/13/11/26/77/1 | DESC/ENTY/ABBR/HUM/NUM/LOC 各 9-138 |
| 训练时间/epoch | ~3h (SLG) | ~5 min (baseline) / ~10 min (SLG) |
| 模型 | Llama-3.1-8B | Llama-3.2-1B |

## 迁移变更摘要

仅对 BioTriplex 阶段代码做了 **3 处最小改动**:

1. `src/data/biotriplex_dataset.py`: 新增 `TRECQADataset` 类 (180 行)
2. `src/scripts/biotriplex_finetune.py`: `task_type` 增加 "trec-qc" 选项 + TASK_DEFAULTS 条目 (10 行)
3. `src/parties/party_s.py` + `heterogeneous_protocol.py`: 传递 task_type 让 S shard 用 6 vs 7 字母 alphabet (15 行)

所有其他调度、评估、汇总逻辑全部通过 `test-data/TrecAATestData/scripts/*.sh + *.py` 实现，**完全不动** BioTriplex 的精度测试脚本。

## 限制

- ABBR 类测试样本仅 9 个，macro F1 估计不稳定
- 单 token letter projection 依赖 Llama tokenizer 将 `a)`/`b)`/... 编码为单 token；不成功时使用第一 token 作为 fallback (token id 64~69)