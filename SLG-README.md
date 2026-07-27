# CipherForge(原SLG-HE-PIR)

> **隐私保护大模型 LoRA 微调框架** —— 单 GPU 主机模拟三方（User U / Model M / Server S）协作微调，基于 BFV 同态加密 + S3PIR 私有信息检索 + dχ 差分隐私实现梯度和数据隐私保护，同时保持与标准明文微调相近的训练效果。

---

## 项目特性

- **协议层**：单 GPU 进程内装配 U/M/S 三方 + 三个独立 CPU 密码学进程池（`HeterogeneousProtocol`）
- **密码学层**：BFV 同态加密（seal-python）、S3PIR Hint 表、按 `y_t` 直接 mmap 取行的 PIR（Design-2）
- **隐私层**：U→M 切分层上的 dχ 多元 Laplace 噪声 + 标签条件 CTI + H15 私有化器
- **模型层**：Llama-3.1-8B-Instruct 切分为 U（前层）/ M（后层 + LoRA）/ S（`lm_head`），支持 safetensors 共享加载
- **优化**：FlashAttention2、SageAttention2++ (INT8)、SageAttention3 (FP4)、Gradient Checkpointing、DeepSpeed ZeRO、Chunked Pipeline
- **任务**：BioTriplex 7 类 GenRel QA 分类、NER 生成式、通用 TREC-QC 6 类分类

## 技术栈

| 组件 | 选型 |
|---|---|
| 基础模型 | Llama-3.1-8B-Instruct |
| 微调方法 | LoRA（rank=8，注入 Q/K/V/O + Gate/Up/Down 7 个投影层） |
| 加密方案 | BFV（seal-python） |
| PIR 方案 | S3PIR（Hint-table 骨架 + 密文 mmap DB） |
| 差分隐私 | dχ (多元 Laplace) + CTI 标签条件 + H15 |
| 注意力 | FlashAttention2、SageAttention2++/3 |
| 训练优化 | Gradient Checkpointing、DeepSpeed ZeRO 1/2/3、Chunked Pipeline |
| 数据集 | BioTriplex（classification + generation）、TREC-QC |

---

## 目录结构

```
SLG-HE-PIR/
├── SLG-README.md                 # 本文件
├── configs/                      # 配置 dataclass（Llama-BioTriplex）
├── src/                          # 框架核心源代码
│   ├── core/                     # 密码学与隐私原语层
│   ├── model/                    # 模型切分与适配
│   ├── parties/                  # 三方协议、运行时、通信
│   │   └── crypto_workers/       # U/M/S 密码学 CPU worker 池
│   ├── data/                     # 数据集与 prompt 适配
│   ├── training/                 # 训练循环、指标、checkpoint
│   ├── scripts/                  # CLI 入口（biotriplex_finetune / evaluate / build_*）
│   ├── attacks/                  # 协议级攻击与安全审计
│   │   └── test_doubles/         # 攻击测试替身（mock party / bus / recorder）
│   ├── audit/                    # 离线隐私审计
│   └── utils/                    # DeepSpeed、metrics、statistics
├── scripts/                      # 顶层生产脚本与功能性测试套件
│   ├── biotriplex_*.sh           # 任务 A/B 全套启动脚本
│   ├── regen_peft_adapter.py     # 从 best_checkpoint 重建 PEFT adapter
│   ├── render_test_report_figures.py  # 渲染 TEST_REPORT.md 17 幅图
│   ├── README_biotriplex.md      # 上述脚本的使用说明
│   └── function-tests/           # 18 个功能性测试 + README
├── tests/                        # 单元/集成测试
│   ├── test_party_s_classification.py
│   ├── test_prg_vectorization.py
│   ├── test_3step_cls.py         # 3-step 分类验证
│   ├── env/                      # 环境/依赖验证（FlashAttention）
│   ├── data-analysis/            # 一次性数据分析脚本
│   └── dp-tests/                 # DP 机制单元测试（8 个）
├── docs/                         # 项目文档（16 个 .md/.svg）
├── test-data/                    # 测试数据与运行产物
│   ├── TEST_REPORT.md            # 主测试报告
│   ├── 3step-cls-test/           # 3-step 分类测试产物
│   ├── SLG-test-data/            # SLG 真实微调产物
│   ├── attack-test-data/         # 攻击测试产物
│   ├── baseline-test-data/       # baseline 微调产物
│   ├── perf-test-data/           # 性能基准产物
│   ├── _legacy_work_logs/        # 旧 _work/ 日志（7-19/7-20）
│   ├── SLG-test-data/cls-SLG-test-data/_legacy_pre_2026-07-22_logs/
│   └── figures/                  # TEST_REPORT 图表
├── baseline/                     # 明文 LoRA 微调基线
│   ├── classification_genrel/    # 任务 A 产物
│   ├── generation_ner/           # 任务 B 产物
│   ├── docs/                     # 基线测试报告与图表
│   ├── logs/                     # baseline 全套运行日志
│   └── llama-rec/                # Meta llama-recipes 子模块
├── SLG-attack-test/              # 攻击测试套件（独立子项目）
├── datasets/                     # BioTriplex + TREC-QC 数据
├── papers/                       # 论文参考（BioTriplex.pdf 等）
├── S3PIR/                        # S3PIR 上游 C++ 实现
├── checkpoints/                  # 训练默认输出（占位）
└── logs/                         # 训练默认输出（占位）
```

---

## 代码文件作用

### `src/core/` —— 密码学与隐私原语层

| 文件 | 作用 |
|---|---|
| `bfv_privselect_v2_adapter.py` | BFV 主实现：建立 SEAL 上下文、定点编码、密钥管理、加密 `lm_head` 数据库、密文序列化、PRG 掩码份额。是 Stage 0 加密库与 Stage 1 U/M/S 密码学路径的底座。 |
| `dchi_privacy.py` | 实现 U→M 切分层上的 dχ 差分隐私：多元 Laplace 噪声、激活范数校准、标签条件 CTI、逐步审计记录。被 `PartyU` 挂载、`Trainer` 写审计日志。 |
| `s3pir_hints.py` | 定义 `HintTable`，负责 S3PIR 分区、主/备 hint 骨架、查询索引构造及 JSON 缓存读写。被 `build_s3pir_hints.py`、`HeterogeneousProtocol`、`CryptoSWorker` 使用。 |
| `key_remapping.py` | 修复 PEFT/LoRA checkpoint 键名与 `.default` adapter 后缀不一致，处理 Conv1D/Linear 权重方向。被 Stage 2 模型加载与 `biotriplex_finetune.py` 导出 PEFT adapter 调用。 |
| `protocol_he_pir.py` | 较早的 SLG-BPL-Lite RSA-KEM + AES-GCM 协议封装（**历史实现**），不在当前 BFV/异构训练主链上。 |

### `src/model/` —— 模型切分与适配层

| 文件 | 作用 |
|---|---|
| `model_splitting.py` | 将 HF Causal LM 切成 U 的 embedding/前层、M 的后层+norm+LoRA、S 的 `lm_head`；实现 safetensors 共享加载、FlashAttention/SageAttention、梯度检查点等内存优化。被 `party_u/m/s.py` 调用。 |

### `src/parties/` —— 三方协议、运行时、通信层

| 文件 | 作用 | 当前定位 |
|---|---|---|
| `party_u.py` | U 持有输入和标签，执行 embedding/前半模型得到 `H_U`，可在 U→M 边界注入 dχ 噪声；调用 `CryptoU` 对 S3PIR 密文加 PRG mask。 | 当前主链核心 |
| `party_m.py` | M 执行后半模型+LoRA，持有或委托 `CryptoM` 使用 `sk_M` 解密带掩码密文；组合 S 的 share 得到 `a_t - V_y`，仅更新 M 侧 LoRA。 | 当前主链核心 |
| `party_s.py` | S 持有冻结的 `V = lm_head.weight`，计算 logits 和 `a_t = softmax(z) V`，委托 `CryptoS` 生成 `s_share` 和读取加密行（Design-2 直接按 `y_t` mmap 取行）。 | 当前主链核心 |
| `heterogeneous_protocol.py` | **当前主运行时**：单 GPU 进程内装配 U/M/S，并创建三个 CPU CryptoWorker 池；向 `Trainer` 暴露 `step_train(_chunked)`、`step_val`、checkpoint、shutdown。 | **当前推荐入口** |
| `fusion_protocol.py` | U/M/S 完全放在同一进程直接调用，复刻 IPC 协议接口，主要用于节省多 CUDA context 内存。 | 已被 `HeterogeneousProtocol` 取代 |
| `transport.py` | 定义统一 `MessageBus` 协议、内存条件变量总线 `InProcessBus` 和多进程队列总线 `QueueBus`，处理按 peer/tag/step 路由及旧消息清理。 | 通信抽象层 |
| `ipc_protocol.py` | 废弃兼容模块，将历史 `IPCProtocol` 名称重导出为 `LegacyIPCStub`。 | **不可作为当前推荐入口** |
| `wire.py` | 定义训练步结果 `StepResult` 和阶段性能记录器 `StepProfiler`，被异构运行时与旧 IPC 共同使用。 | 公共协议数据类型 |
| `legacy_ipc_stub.py` | 保留三独立 spawn 进程 U/M/S 的历史实现和队列消息循环，用于多主机预演、边界审计、向后兼容。 | 非默认运行路径 |

### `src/parties/crypto_workers/` —— CPU 密码学进程池

| 文件 | 作用 | 隐私边界 |
|---|---|---|
| `base.py` | 提供 worker 进程初始化、按 PID 缓存 SEAL 状态和统一请求分发函数。 | 不持有固定角色秘密 |
| `crypto_u.py` | 对 S 返回的 `Enc(-V_y)` 同态加入 U/S 共享 PRG 生成的 `R_t`，得到 `Enc(-V_y + R_t)`。 | 仅持有公钥和 PRG seed |
| `crypto_m.py` | 批量解密 U 发来的掩码密文，输出定点解码后的 `masked_arr`。 | **唯一应持有 `sk_M` 的 CPU worker** |
| `crypto_s.py` | 生成 `s_share = a_t - R_t`，并从 mmap 加密数据库读取 `Enc(-V_y)` 组成响应。 | 持有公钥、PRG seed、只读加密库 |
| `pool.py` | 为单一 worker 类型创建长生命周期 spawn 进程池，提供同步/异步提交和关闭接口。 | 被 `HeterogeneousProtocol` 实例化为 U/M/S 三个池 |

### `src/data/` —— 数据集与 prompt 适配层

| 文件 | 作用 |
|---|---|
| `biotriplex_dataset.py` | 面向当前 BioTriplex 基线格式，支持 GenRel 7 类分类和 NER JSON 生成；构建 Llama chat prompt、标签、实体和 gold 文件。被 `biotriplex_finetune.py`、`evaluate_biotriplex.py` 使用。 |
| `dataset.py` | 较通用/旧版 JSONL 数据加载器、Llama tokenizer wrapper、基础 `BioTriplexQADataset` 及答案/实体解析工具。被通用 `finetune.py` 和 `Trainer` 部分解析逻辑使用。 |

### `src/training/` —— 训练循环、指标、checkpoint 层

| 文件 | 作用 |
|---|---|
| `trainer.py` | 协议无关的高层 epoch/step 循环：DataLoader、调用 `step_train(_chunked)`/`step_val`、指标、早停、checkpoint、测试与 dχ 审计。 |
| `biotriplex_metrics.py` | 计算 BioTriplex 分类的多类/多标签 F1、AUC，以及 NER 的实体级每类与宏/微平均指标。 |
| `checkpoint.py` | `CheckpointManager` 管理 U/M/S 联合 checkpoint、best checkpoint 和旧 checkpoint 清理。 |
| `evaluation.py` | 从联合 checkpoint 恢复 LoRA、修复键名并加载标准 HF 模型，在无 PIR/BFV 情况下执行测试集生成和通用实体指标。 |

### `src/scripts/` —— CLI/流水线入口层

| 文件 | 作用 |
|---|---|
| `biotriplex_finetune.py` | BioTriplex 专用总入口，支持 Stage 0/1/2/all；应用分类/生成任务默认超参数，构建异构协议、运行 Trainer、导出标准 PEFT adapter，并调用专用评估脚本。 |
| `evaluate_biotriplex.py` | BioTriplex 专用 Stage 2：加载 base model + PEFT adapter，分类任务取 a–g logits，生成任务输出 NER JSON，并生成基线兼容指标文件。 |
| `finetune.py` | 通用/旧数据格式总入口，内置 `Config` 并支持 JSON override；同样运行 Stage 0/1/2，但数据与评估指标不针对当前 BioTriplex baseline 格式。 |
| `build_encrypted_db.py` | Stage 0 Step 1：读取 HF `lm_head.weight`，调用 BFV backend 加密每一行并保存密文数据库、公钥和元数据。 |
| `build_s3pir_hints.py` | Stage 0 Step 2：检查已存在的 BFV 数据库，构建 S3PIR 分区/hint 骨架和分区元数据。**当前实现中的 parity 计算仍是简化骨架**，完整实现中应计算实际 parities。 |

### `src/attacks/` —— 协议级攻击与安全审计层

| 文件 | 作用 |
|---|---|
| `L1A_separation.py` | 测试 M 能否从 `s_share` 与解密后的 `masked_arr` 中单独分离 `V_y` 或 PRG 掩码。 |
| `L1C_dlg_inversion.py` | DLG/TAG 风格梯度输入反演评估占位。 |
| `L2_cutgrad.py` | 对 `g_accum` 实施 Free/Prior/Oracle/`H_M` 辅助的聚类或标签推断，计算 agreement、ARI。 |
| `L3B_pir_bytes.py` | 从 PIR/parity 字节直方图和哈希特征训练随机森林，测试字节流是否泄漏标签/索引。 |
| `L4A0_hu_inversion.py` | U→M 明文切分激活 `H_U` 的 smashed-data inversion 评估框架，关注 dχ 加噪后的重构率。 |
| `L4A_hm_inversion.py` | 更深层 `H_M` 的输入 token 反演框架占位。 |
| `L4B_tag_inversion.py` | 梯度匹配的 TAG/DLG 路径尝试恢复输入占位。 |
| `L5_s_inversion.py` | 从 S 可见的 `H_M` 和公开 backbone 尝试输入恢复。 |
| `L6_long_term.py` | 跨 step/epoch 做窗口均值、PCA、K-Means、ARI 和自相关，检测长期训练导致的隐私退化。 |
| `M1_u_extract.py` | 模拟 U 获得 logits 后通过知识蒸馏提取 M 模型。 |
| `M2_s_detect.py` | 测试 S 能否通过 `H_M` 的 Jacobian 秩或 MMD 推断 M 的层数、LoRA rank 等结构。 |
| `M3_lora_internals.py` | M 主机内部的 LoRA 梯度/参数轨迹审计，不属于普通外部攻击者能力。 |
| `M4_mia.py` | 基于 loss 阈值及 shadow-model 设想执行 Membership Inference，报告 AUC、TPR/FPR。 |
| `M5_v_infer.py` | 测试从多步梯度、mock logits、PRG 输出和残余 mask 推断 S 的 `V` 矩阵。 |
| `P1_bfv_security.py` | 审计 BFV 参数安全级、密文随机性、噪声预算、明密文相关性和 PlainModulus 一致性。 |
| `P2_prg.py` | 审计 PRG 均值、方差、自相关、唯一性、seed 长度和 seed 泄漏风险。 |
| `P3_pir.py` | 审计 PIR 查询不可区分性、hint 完整性和加密库布局；明确报告 Design-2 中 S 直接知道 `y_t` 的问题。 |
| `P4_P13_system.py` | 汇总时序、OOM、pickle、临时文件、checkpoint、日志、DeepSpeed、攻击 dump、hint 确定性等系统级风险。 |

### `src/attacks/test_doubles/` —— 攻击测试替身

| 文件 | 作用 |
|---|---|
| `attack_test_bus.py` | 为攻击实验提供三条独立多进程 Queue、消息录制及可选恶意中间人。 |
| `malicious_bus.py` | 实现窃听、篡改、重放、冒充等 MITM 行为，支持恶意 pickle 注入测试。 |
| `mock_party.py` | U/M/S 测试替身抽象基类，严格校验每方允许观察的字段并分发恶意行为。 |
| `mock_party_u.py` | U 侧替身；可选开启真实协议不存在的 logits 回流，以支持模型提取攻击。 |
| `mock_party_m.py` | M 侧替身；收集 `g_H`、`H_M`、标签等攻击观测数据。 |
| `mock_party_s.py` | S 侧替身；记录 `H_M`、PIR query、argmax 和标签等观测。 |
| `allowed_keys.py` | 定义物理隔离下 U/M/S 的合法可见字段，提供严格检查和静默过滤。 |
| `wire_recorder.py` | 将跨边界消息写为 JSONL 元数据和二进制 payload，附 SHA-256 完整性摘要。 |

### `src/audit/` —— 离线隐私审计层

| 文件 | 作用 |
|---|---|
| `lia_h15_audit.py` | 读取训练生成的 `dp_audit.jsonl`，汇总 dχ 激活率、校准次数、η 和噪声范数，输出 JSON 与 Markdown 报告。 |

### `src/utils/` —— 通用运行与实验工具层

| 文件 | 作用 |
|---|---|
| `deepspeed_zero.py` | 生成 DeepSpeed ZeRO 1/2/3 配置并封装 M shard 的 ZeRO 初始化、checkpoint 等辅助逻辑。 |
| `metrics.py` | 提供训练计数器、timer、命名指标序列、均值/标准差/min/max 汇总。 |
| `statistics.py` | 提供描述统计、95% CI、t-test、bootstrap CI、Mann–Whitney U 和分布比较。 |

---

## 启动指引

### 前置准备

1. **基础模型**：在 HuggingFace 缓存目录准备好 `Llama-3.1-8B-Instruct`（参 `hf_cache/Llama-3-1-8B-I/`）。
2. **数据集**：在 `datasets/botriplex/Preprocessed BioTriplex/` 准备分类（GenRel 7 类）与生成（NER JSON）格式数据。
3. **BFV 缓存目录**：创建独立的 BFV 缓存目录（如 `slg-bfv-cache/`），用于存放密文数据库与 S3PIR hints。
4. **Python 依赖**：seal-python、PyTorch、Transformers、PEFT、safetensors、FlashAttention2（可选）。

### Stage 0：构建加密数据库与 S3PIR hints

```bash
cd SLG-HE-PIR

# Stage 0 Step 1：构建 BFV 加密 V 矩阵数据库
python -m src.scripts.build_encrypted_db \
  --model_path "$HF_MODEL" \
  --cache_dir "$BFV_CACHE" \
  --vocab_size 128256 \
  --hidden_dim 4096 \
  --poly_degree 4096 \
  --plain_bits 30 \
  --scale 10000

# Stage 0 Step 2：构建 S3PIR hint table（必须先完成 Step 1）
python -m src.scripts.build_s3pir_hints \
  --cache_dir "$BFV_CACHE" \
  --n_entries 128256 \
  --lam 80
```

成功标志：`$BFV_CACHE` 下出现 `bfv_meta.json`、`bfv_ct_db_*.bin`、`bfv_pk.bin`、`hint_table.json`。

### Stage 1：BioTriplex 三方隐私保护微调

#### 分类任务（GenRel QA，6 epoch）

```bash
python -m src.scripts.biotriplex_finetune \
  --task_type classification \
  --stage all \
  --data_path "$DATA_PATH" \
  --hf_model "$HF_MODEL" \
  --bfv_cache_dir "$BFV_CACHE" \
  --output_dir "$OUTPUT_DIR/classification_genrel" \
  --max_epochs 6 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --u_layers 16 \
  --m_layers 16
```

#### 生成任务（NER，10 epoch）

```bash
python -m src.scripts.biotriplex_finetune \
  --task_type generation \
  --stage all \
  --data_path "$DATA_PATH" \
  --hf_model "$HF_MODEL" \
  --bfv_cache_dir "$BFV_CACHE" \
  --output_dir "$OUTPUT_DIR/generation_ner" \
  --max_epochs 10 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.2 \
  --u_layers 16 \
  --m_layers 16
```

### Stage 2：纯明文评估

```bash
python -m src.scripts.evaluate_biotriplex \
  --task_type classification \
  --base_model "$HF_MODEL" \
  --adapter_dir "$OUTPUT_DIR/classification_genrel/adapter" \
  --data_path "$DATA_PATH" \
  --split test \
  --output_dir "$OUTPUT_DIR/classification_genrel/logs"
```

成功标志：`$OUTPUT_DIR/<task>/logs/` 下出现 `*_evaluate_metrics.json`、`infer_outputs_*.json`。

### 顶层 Shell 一键启动

`scripts/biotriplex_run_all.sh` 串联任务 A → B；`scripts/biotriplex_classification_genrel.sh`、`scripts/biotriplex_generation_ner.sh` 单独运行其一。详见 [scripts/README_biotriplex.md](scripts/README_biotriplex.md)。

---

## 测试与基线

- **单元/集成测试**：`pytest tests/`（含 `tests/dp-tests/` 8 个 DP 机制测试）
- **功能性测试**：`python -m scripts.function_tests.<module>`（18 个脚本，参 [scripts/function-tests/README.md](scripts/function-tests/README.md)）
- **环境验证**：`python tests/env/test_flash_attention.py`、`python tests/env/monitor_flash_attn.py`
- **基线对比**：`bash scripts/biotriplex_run_all.sh`（参 [baseline/docs/BIOTRIPLEX_BASELINE_TEST_REPORT.md](baseline/docs/BIOTRIPLEX_BASELINE_TEST_REPORT.md)）
- **攻击测试**：见独立子项目 [SLG-attack-test/README.md](SLG-attack-test/README.md) 与 [SLG-attack-test/run_attack_suite.py](SLG-attack-test/run_attack_suite.py)
- **测试数据**：见 [test-data/TEST_REPORT.md](test-data/TEST_REPORT.md)

---

## 关键调用链

### 初始化链

```
biotriplex_finetune.run_stage1
  → BFVPrivSelectV2Backend(...)        # 建立 BFV 上下文与缓存 DB
  → _drop_secret_key()                  # 主进程不保留 sk_M
  → HeterogeneousProtocol(...)          # 构造当前主运行时
  → PartyU / PartyM / PartyS            # 加载 U/M/S 模型分片
  → model_splitting.py                  # 各方加载对应层
  → CryptoWorkerPool(U/M/S)             # 创建三个 CPU worker 进程池
  → Trainer(protocol, ...)              # Trainer 只看协议公共接口
```

### 单步训练链

```
Trainer._run_epoch
  → HeterogeneousProtocol.step_train_chunked
  → PartyU.forward_train(batch)             # H_U = U-embeds(x)
  → PartyM.forward(H_U)                     # H_M = M-decoder(H_U)
  → PartyS.process_logits_dispatch          # z = H_M V^T, a_t = softmax(z) V
  → CryptoSWorker                           # 生成 s_share, 读 Enc(-V_y)
  → PartyU.privselect_and_recover_dispatch  # S3PIR response
  → CryptoUWorker                           # Enc(-V_y) + R_t → Enc(-V_y+R_t)
  → PartyM.backward_and_update_dispatch
  → CryptoMWorker                           # sk_M 解密 → masked_arr
  → PartyM                                  # masked_arr + s_share = a_t - V_y
  → StepResult → Trainer
```

---

## 注意事项

1. **Design-2 S3PIR 查询隐藏失效**：`CryptoSWorker` 当前直接通过 `y_t` 读取 `Enc(-V_y)`，`P3_pir.py` 将其判为查询隐藏失效。**目前未实现严格真 S3PIR**，仅作 Design-2 占位。

2. **配置默认值不一致**：
   - `configs/llama_biotriplex_he_pir.py` 中 `LlamaBioTriplexConfig` 默认 `u_layers=0, m_layers=32`
   - `biotriplex_finetune.py` CLI 默认 `u_layers=16, m_layers=16`
   二者**当前不一致**。CLI 推荐 16/16，README 文档须区分。

3. **Stage 2 是纯明文标准 forward**，无 PIR/BFV（这是设计行为，不是漏调用）。

4. **历史协议实现**：`fusion_protocol.py`、`ipc_protocol.py`、`legacy_ipc_stub.py`、`protocol_he_pir.py` 为旧版/单进程方案，**不要作为新入口**。当前推荐 `HeterogeneousProtocol`。

5. **攻击模块依赖外部包**：顶层攻击类依赖外部 `SLG_attack_test` 包，部分 GPU 反演攻击仍是实验 stub。

6. **`max_seq_length` 默认较大**：默认 10000 显著增加 logits 与 attention 显存。建议先用较短长度或 `--max_train_steps` 做 smoke test。

7. **`test-data/` 数据巨大**（多数百 MB～GB 级），首次 clone 后单独同步。可考虑 Git LFS。

---

## 引用文档

- 系统文档：[docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md](docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md)
- 使用文档：[docs/SLG_HE_PIR_USAGE.md](docs/SLG_HE_PIR_USAGE.md)
- 项目文档：[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)
- BioTriplex 微调指南：[baseline/docs/BIOTRIPLEX_BASELINE_TEST_REPORT.md](baseline/docs/BIOTRIPLEX_BASELINE_TEST_REPORT.md)
- 攻击测试方案：[docs/攻击类测试方案.md](docs/攻击类测试方案.md)
- DP 机制迁移参考：[docs/DP机制-迁移参考.md](docs/DP机制-迁移参考.md)
- 主测试报告：[test-data/TEST_REPORT.md](test-data/TEST_REPORT.md)
- 攻击测试套件：[SLG-attack-test/README.md](SLG-attack-test/README.md)

## License

本仓库沿用 BioTriplex 与 Llama-3.1 的原始 License；具体条款参 [hf_cache/Llama-3-1-8B-I/LICENSE](hf_cache/Llama-3-1-8B-I/LICENSE) 与 [papers/BioTriplex/LICENSE](papers/BioTriplex/LICENSE)。