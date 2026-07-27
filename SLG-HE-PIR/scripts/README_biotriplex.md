# BioTriplex 微调脚本 (SLG-HE-PIR)

本文档描述在 SLG-HE-PIR 三方隐私保护 LoRA 微调框架下，复刻 BioTriplex 论文两项真实任务的脚本。

所有参数与 [`docs/BIOTRIPLEX_FINETUNE_README.md`](../docs/BIOTRIPLEX_FINETUNE_README.md) 保持一致。

---

## 任务概览

| 任务 | 类型 | 输入文件 | Epoch | LoRA r/α/dropout | 学习率 | Weight Decay | 评估指标 |
|---|---|---|---|---|---|---|---|
| A — GenRel QA | 7 类分类 | `train/val/test_para.txt` | 6 | 8/16/0.05 | 1e-4 | 0.0 | multi-label F1、Macro F1、Macro ROC AUC |
| B — NER JSON | 生成 | `train/val/test_shorter.txt` | 10 | 8/16/0.05 | 1e-4 | 0.2 | 3 类 span-level P/R/F1 + Macro/Weighted/Micro |

---

## 目录结构（新增文件）

```
SLG-HE-PIR/
├── docs/BIOTRIPLEX_FINETUNE_README.md        # 原始任务说明（不变）
├── baseline/classification_genrel/...        # 原 baseline 脚本（不变）
├── baseline/generation_ner/...               # 原 baseline 脚本（不变）
├── scripts/
│   ├── biotriplex_classification_genrel.sh    # 任务 A 入口
│   ├── biotriplex_generation_ner.sh          # 任务 B 入口
│   ├── biotriplex_run_all.sh                 # A→B 串行总入口
│   └── README_biotriplex.md                  # 本文件
└── src/
    ├── data/biotriplex_dataset.py            # 两类 BioTriplex Dataset
    ├── scripts/
    │   ├── biotriplex_finetune.py            # 主编排（Stage 0/1/2）
    │   └── evaluate_biotriplex.py            # Stage 2 评估器
    ├── parties/heterogeneous_protocol.py      # （已扩展）step_val/step_test 多带 doc_keys
    └── training/
        ├── trainer.py                        # （已扩展）支持 task_type="generation" 时计算 NER 指标
        └── biotriplex_metrics.py             # NER / 分类指标计算（被 evaluate_biotriplex.py 复用）
```

---

## 运行方法

### 方式一：一键运行两个任务

```bash
cd /root/autodl-tmp/SLG-HE-PIR
bash scripts/biotriplex_run_all.sh
```

该脚本会：
1. 调用 `scripts/biotriplex_classification_genrel.sh`（任务 A）
2. 调用 `scripts/biotriplex_generation_ner.sh`（任务 B）

### 方式二：单独运行某个任务

```bash
cd /root/autodl-tmp/SLG-HE-PIR

# 任务 A — GenRel QA 分类（6 epoch）
bash scripts/biotriplex_classification_genrel.sh

# 任务 B — NER 生成（10 epoch）
bash scripts/biotriplex_generation_ner.sh
```

### 方式三：手动调用底层 Python CLI

```bash
cd /root/autodl-tmp/SLG-HE-PIR

# 完整 Stage 0+1+2
python src/scripts/biotriplex_finetune.py \
    --task_type classification \
    --stage all \
    --data_path "/root/autodl-tmp/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/" \
    --output_dir /root/autodl-tmp/SLG-HE-PIR/baseline/classification_genrel/checkpoints \
    --max_epochs 6 \
    --batch_size 1

# 只跑 Stage 2 评估（基于已有 best_checkpoint.pt + adapter）
python src/scripts/biotriplex_finetune.py \
    --task_type generation \
    --stage 2 \
    --data_path "/root/autodl-tmp/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/" \
    --output_dir /root/autodl-tmp/SLG-HE-PIR/baseline/generation_ner/checkpoints
```

### 方式四：直接调用 Stage 2 评估器

```bash
python src/scripts/evaluate_biotriplex.py \
    --task_type generation \
    --adapter_dir /root/autodl-tmp/SLG-HE-PIR/baseline/generation_ner/adapter \
    --data_path "/root/autodl-tmp/SLG-HE-PIR/datasets/botriplex/Preprocessed BioTriplex/" \
    --output_dir /root/autodl-tmp/SLG-HE-PIR/baseline/generation_ner/logs
```

---

## Stage 流水线（每个任务）

| Stage | 内容 | 入口 | 产物 |
|---|---|---|---|
| **Stage 0** | BFV Encrypted DB 构建（一次性）+ S3PIR hints | `--stage 0` 或 `all` | `${BFV_CACHE_DIR}/` 下 `.enc` DB + hints |
| **Stage 1** | 三方隐私 LoRA 微调（U/M/S） | `--stage 1` 或 `all` | `checkpoints/best_checkpoint.pt` + `adapter/`（PEFT 格式） |
| **Stage 2** | 合并 LoRA + base model，明文前向 + 计算指标 | `--stage 2` 或 `all` | `logs/infer_outputs_<TS>.json` + `logs/{genrel|ner}_<TS>_evaluate_metrics.json` |

Stage 2 在合并 LoRA 之后做的是**普通的前向推理**（无 PIR、无 BFV）——这一步脱离了训练时的三方隐私结构，因为 Base Model + LoRA Adapter 已经在 U 一侧，与 [`docs/SLG_HE_PIR_USAGE.md`](../docs/SLG_HE_PIR_USAGE.md) 文档 §6.1 的「Stage 2 测试评估」描述完全一致。

---

## 产物路径

任务 A — `baseline/classification_genrel/`：

```
baseline/classification_genrel/
├── checkpoints/
│   └── best_checkpoint.pt                # Stage 1 输出（party 检查点）
├── adapter/
│   ├── adapter_config.json               # Stage 1 末尾导出的 PEFT adapter
│   └── adapter_model.safetensors
└── logs/
    ├── train_<TS>.log                    # Stage 0/1 训练日志
    ├── biotriplex_finetune_<stage>_<TS>.log
    ├── infer_outputs_<TS>.json           # Stage 2 推理输出
    ├── evaluate_<TS>.log
    └── genrel_<TS>_evaluate_metrics.json # ★ 最终结果
```

任务 B — `baseline/generation_ner/`：

```
baseline/generation_ner/
├── checkpoints/
│   └── best_checkpoint.pt
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── logs/
    ├── train_<TS>.log
    ├── biotriplex_finetune_<stage>_<TS>.log
    ├── infer_outputs_<TS>.json
    ├── evaluate_<TS>.log
    └── ner_<TS>_evaluate_metrics.json    # ★ 最终结果
```

---

## `evaluate_metrics.json` 字段说明

### 任务 A（`genrel_<TS>_evaluate_metrics.json`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `task` | str | `"GenRel QA (7-class Classification)"` |
| `n_samples` | int | 测试样本数 |
| `n_parse_failures` | int | 模型输出无法解析的样本数 |
| `metrics.micro_accuracy` | float | sklearn `accuracy_score` |
| `metrics.macro_precision/recall/f1` | float | sklearn macro average |
| `metrics.weighted_f1` | float | sklearn weighted F1 |
| `metrics.micro_f1` | float | sklearn micro F1 |
| `metrics.multilabel_f1_samples/macro/micro` | float | 多标签 F1 |
| `metrics.macro_roc_auc_ovr` | float | ROC AUC OvR |
| `metrics.micro_roc_auc_ovr` | float | ROC AUC OvR 微平均 |
| `per_class_metrics.<rel>` | dict | 每个 relation 的 P/R/F1/support |
| `y_true_distribution` | dict | gold 各类样本数 |
| `y_pred_distribution` | dict | pred 各类样本数 |
| `confusion_matrix` | list | 7×7 混淆矩阵 |

### 任务 B（`ner_<TS>_evaluate_metrics.json`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `task` | str | `"NER (Span-level Exact-match)"` |
| `n_common_doc_keys` | int | 有 gold 匹配的样本数 |
| `n_parse_failures` | int | JSON 解析失败数 |
| `metrics.macro_f1` | float | 3 类 macro F1 |
| `metrics.weighted_f1` | float | 加权 F1 |
| `metrics.overall_micro_precision/recall/f1` | float | 三类合并 micro |
| `per_class_metrics.GENE/DISEASE/RELATION` | dict | 每类 P/R/F1/tp/fp/fn |
| `per_class_parse_failures` | dict | 每类解析失败计数 |

---

## 与 baseline 的差异

| 项目 | baseline (`llama-rec`) | SLG-HE-PIR (本脚本) |
|---|---|---|
| 训练运行时 | 单进程标准 LoRA | 三方隐私协议（U/M/S） |
| 推理 + 评估 | 标准 generate + 自定义 eval 脚本 | 同上（Stage 2 阶段剥离 PIR） |
| LoRA 权重 | `PeftModel.save_pretrained` | 同左（Stage 1 末尾导出） |
| `evaluate_metrics.json` 字段 | baseline 自定义 | 同 README §1.4 / §2.4 字段一致 |
| BFV Enc DB | 不需要 | Stage 0 构建一次（≈ 几分钟，可复用） |

---

## 参数对齐（README §1.5 / §2.5）

| 维度 | README | 本脚本 |
|---|---|---|
| `task_type` | "classification" / "generation" | 同左 |
| `num_epochs` | 6 / 10 | `--max_epochs` 默认 6 / 10 |
| `batch_size` | 1 | `--batch_size 1` |
| `batching_strategy` | "padding" | 数据集已实现 padding |
| `context_length` | 10000 | `--max_seq_length 10000`（脚本默认） |
| `use_peft` | True | LoRA r=8 / α=16 / dropout=0.05 |
| `lr` | 1e-4 | `--learning_rate 1e-4` |
| `weight_decay` | 0.0 / 0.2 | `--weight_decay 0.0` (A) / `0.2` (B) |
| `gamma` | 0.85 | lr_scheduler cosine_with_warmup 中间衰减 |
| `gradient_clipping` | 1.0 | `--gradient_clip_norm 1.0` |
| `seed` | 42 | `--seed 42` |
| `general_relations` | True | 数据集固定 |
| `return_neg_relations` | False | 数据集固定 |
| `upweight_minority_class` | False | 数据集固定 |

---

## 调试小贴士

1. **快速冒烟**：用 `--eval_max_samples 10` 把评估限制在前 10 个样本上：
   ```bash
   python src/scripts/evaluate_biotriplex.py \
       --task_type classification \
       --adapter_dir <adapter_dir> \
       --data_path <DATA_PATH> \
       --output_dir <OUT_DIR> \
       --max_eval_samples 10
   ```

2. **分阶段跑**：先 `--stage 0` 跑完数据库构建，再 `--stage 1` 训练，最后 `--stage 2` 评估。

3. **跳过 DB 重建**：如果 ${BFV_CACHE_DIR} 已经存在，可以 `--skip_db --skip_hints`。

4. **OOM 排查**：
   - `max_seq_length` 10000 对 8B 模型可能太大；可降至 4096 / 3072；
   - `chunk_tokens` 3072 是显存友好的默认值。

5. **日志位置**：
   - 主入口日志：`baseline/<task>/logs/train_<TS>.log`
   - 评估日志：`baseline/<task>/logs/evaluate_<TS>.log`

---

## 常见问题

**Q: 为什么 Stage 2 不走 PIR？**
A: 训练结束后 LoRA adapter + base model 都在 U 一侧合并。要评估验证集/测试集，必须先把模型实例化成完整形式，再做标准前向——这一步没有 PIR 的需求。这与 SLG-HE-PIR 设计文档一致。

**Q: 评估时 `evaluate_metrics.json` 里的 ROC AUC 怎么算？**
A: 任务 A 的 `evaluate_biotriplex.py` 在 `a)/b)/.../g)` 七个 token id 上取 last-token logits 做 softmax，得到 7 维概率向量；ROC AUC OvR 通过 `sklearn.metrics.roc_auc_score(..., multi_class='ovr')` 计算。

**Q: NER 任务的 gold 文件在哪？**
A: 数据集初始化时会自动写到 `${DATA_PATH}/test_gold_ner.txt`、`${DATA_PATH}/val_gold_ner.txt` 等。`evaluate_biotriplex.py` 通过 `doc_key` 对齐预测与 gold 实体。

---

## 重新评估 / 恢复训练

若已经跑过 Stage 1 但只想重跑 Stage 2：

```bash
# 单独跑 Stage 2（使用已有 checkpoint + adapter）
python src/scripts/biotriplex_finetune.py \
    --task_type classification \
    --stage 2 \
    --data_path "<DATA_PATH>" \
    --output_dir baseline/classification_genrel/checkpoints
```

从 `last_checkpoint.pt` 恢复训练：

```bash
# 修改 trainer.py 的 _load_checkpoint 调用，或在 config 里加 --resume
# (当前脚本默认从 best_checkpoint.pt 恢复)
```

---

## 相关链接

- 任务规格：[`docs/BIOTRIPLEX_FINETUNE_README.md`](../docs/BIOTRIPLEX_FINETUNE_README.md)
- 框架说明：[`docs/SLG_HE_PIR_USAGE.md`](../docs/SLG_HE_PIR_USAGE.md)
- 系统设计：[`docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md`](../docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md)
- 原 baseline 入口：[`baseline/run_all.sh`](../baseline/run_all.sh)