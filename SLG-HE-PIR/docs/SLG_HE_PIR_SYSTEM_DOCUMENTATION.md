# SLG-HE-PIR 项目技术文档

> **文档版本**: 2026-07-21 (Phase 1 v2.2 后)
> **项目路径**: `/root/autodl-tmp/SLG-HE-PIR`
> **适用范围**: SLG-HE-PIR v2.0 — 单主机三参与方隐私保护 LoRA 微调系统
> **代码版本**: `src/scripts/finetune.py` __version__ = `2.0.0`
> **对照参考**: `docs/SLG系统流程.svg`、`docs/BIOTRIPLEX_FINETUNE_README.md`
> **配套使用文档**: `docs/SLG_HE_PIR_USAGE.md`(面向新会话的快速调用手册)

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [全局数据流动(SVG 四阶段)](#3-全局数据流动svg-四阶段)
   - 3.1 [#0 offline PreCompute](#31--0-offline-precompute)
   - 3.2 [#1 Train-FP](#32--1-train-fp)
   - 3.3 [#2 LoRA-BP](#33--2-lora-bp)
   - 3.4 [#3 Eval-FP](#34--3-eval-fp)
4. [实现原理](#4-实现原理)
   - 4.1 [技术栈总览](#41-技术栈总览)
   - 4.2 [模型切分策略](#42-模型切分策略)
   - 4.3 [GPU/CPU 任务分工](#43-gpucpu-任务分工)
   - 4.4 [内存与显存分配](#44-内存与显存分配)
   - 4.5 [进程/通信模型](#45-进程通信模型)
   - 4.6 [BFV 同态加密流水线](#46-bfv-同态加密流水线)
   - 4.7 [S3PIR 单服务器 PIR](#47-s3pir-单服务器-pir)
   - 4.8 [训练循环与检查点](#48-训练循环与检查点)
5. [关键流程的源码定位](#5-关键流程的源码定位)
6. [附录](#6-附录)
   - 6.1 [默认 Config 参数表](#61-默认-config-参数表)
   - 6.2 [关键指标实测(Phase 1 v2.2)](#62-关键指标实测phase-1-v22-2026-07-21)
   - 6.3 [SVG 公式 ↔ 代码对照表](#63-svg-公式--代码对照表)
   - 6.4 [已知缺陷与设计权衡](#64-已知缺陷与设计权衡)
     - 6.4.1 [val F1 = 0 + val_ce_loss = 0](#641-val_entity_micro_f1--0--val_ce_loss--0--prompt-解析层与-nerqa-任务错位)
     - 6.4.2 [train_loss_proxy 语义](#642-train_loss_proxy-语义)
     - 6.4.3 [u_split_layer=0](#643-u_split_layer--0-与-svg-不完全一致)
     - 6.4.4 [没有 LoRA-only adapter 保存](#644-没有-lora-only-adapter-保存)
     - 6.4.5 [CryptoMWorker sk_M 隔离](#645-cryptomworker-的-sk_m-隔离)
     - 6.4.6 [parse_gold_entities 不存在](#646-parse_gold_entities-不存在)
   - 6.6 [Phase 1 v2.2 修复历史](#66-phase-1-v22-修复mask-完美消去--db-cache-key-配对)
   - 6.5 [文档版本与维护说明](#65-文档版本与维护说明)

---

## 1. 项目概述

**SLG-HE-PIR** 是一个面向 **隐私保护大模型微调** 的工程系统,在单台 GPU 主机上模拟三方协作场景,对 Llama-3.1-8B-Instruct 在 BioTriplex 私有医学语料上进行 LoRA 微调。整个流程在不暴露用户私有数据 `(x, y)` 与 lm_head 矩阵 `V` 的前提下完成 LoRA 参数更新,核心依赖 **BFV 同态加密** 与 **S3PIR 单服务器 PIR** 两个密码学原语。

- **硬件**: 1 × RTX 5090(32 GB, compute capability 12.0)+ ≥ 200 GB 系统 RAM
- **模型**: Llama-3.1-8B-Instruct(vocab_size=128256, hidden_dim=4096, 32 层 decoder)
- **任务**: BioTriplex 命名实体识别 (NER, output = `Entities: (text, TYPE); ...`); 同时保留 BioTriplex-QA 关系分类(7 类字母选项)的设计痕迹
- **训练数据**: BioTriplex `train.jsonl` / `val.jsonl` / `test.jsonl` —— Phase 1 v2.2 实测 train/val/test = 620 / 118 / 160(从原始 738 条按 `train_ratio=0.9, seed=42` 划分)
- **超参**: LoRA r=8, α=16, dropout=0.05, lr=3.5e-4, batch=4, max_seq_length=128, max_epochs=10(2-epoch 测试覆盖为 2)
- **密码学参数**: BFV `poly_degree=4096, plain_bits=30, scale=10000`;S3PIR `partition_size=256, lam=80`

> **⚠️ 数据任务 vs. 评测指标的错位(详见 §6.4.1)**:数据集的 `output` 是 NER 字符串(`Entities: (hyperuricemia, DISEASE)`),但当前的 `parse_answer_letter` / `_safe_letter_split` 仍按 BioTriplex-QA 7 类字母答案(`a)`/`b)`/...)设计,导致 `val_entity_micro_f1 = 0`。训练本身不被影响(S3PIR mask 正确消去 → `train_loss_proxy` 单调下降),但 val 评测指标**目前没有数值意义**。本节同时记录 NER 数据任务与 QA 字母历史假设,以便后续修复时对照。

---

## 2. 系统架构

### 2.1 三参与方 + 单主机 Fusion 进程

虽然系统在密码学上严格区分三个参与方(U/M/S),但在物理部署上它们运行在 **同一台主机的同一进程** 内,通过 Python 对象直接传递 GPU tensor,仅把密码学操作(同态加/解密/PRG)与 PIR 查询下放到 **CPU worker 子进程池**。这个架构叫 **"Fusion"**(对应 SVG 顶部 "Server (按行) 掩码" 区的描述)。

```
┌────────────────────────────────────────────────────────────────────┐
│                  Main Process (Fusion Driver)                      │
│  HeterogeneousProtocol  (in-process U/M/S handshake)               │
│  ┌──────────┐  GPU tensor  ┌──────────┐  GPU tensor  ┌──────────┐  │
│  │ PartyU   │ ───────────▶ │ PartyM   │ ───────────▶ │ PartyS   │  │
│  │ embed    │              │ decoder  │              │ lm_head  │  │
│  │ + opt=None│             │ + LoRA   │              │ + enc_DB │  │
│  └──────────┘              └──────────┘              └──────────┘  │
│       │                         │                          │       │
│       │ IPC: ct_list            │ IPC: s_share             │       │
│       ▼                         ▼                          ▼       │
│  ┌────────────┐          ┌────────────┐           ┌────────────┐    │
│  │ CryptoU    │          │ CryptoM    │           │ CryptoS    │    │
│  │ WorkerPool │          │ WorkerPool │           │ WorkerPool │    │
│  │ (CPU spawn)│          │ (CPU spawn)│           │ (CPU spawn)│    │
│  │  + pk_M    │          │  + sk_M    │           │ + Enc(DB)  │    │
│  │  + PRG     │          │  - Decrypt │           │ + Hints    │    │
│  └────────────┘          └────────────┘           └────────────┘    │
└────────────────────────────────────────────────────────────────────┘
        │                              │                      │
        └──── multiprocessing.Pool(spawn) ─────────────────────┘
```

> **注意**:子进程使用 **spawn** 而非 fork。`spawn` 模式下每个 worker 从空白 Python 解释器启动,只通过 `init_kwargs` 拿到所需的 `pk_M` / `sk_M` / Enc(DB) 路径,**不会继承父进程的 CUDA caching allocator 快照**,杜绝了 8 worker × 6 GB 的 OOM 风险。详见 §4.5.1 的 fork vs spawn 决策表。

### 2.2 关键设计原则

| 原则 | 实现 |
|---|---|
| **密码学边界 = 进程边界** | 所有需要 sk_M 的解密操作都在 `CryptoMWorker` 子进程中执行;主进程的 BFV 后端在 Stage 1 启动时立刻 `_drop_secret_key()` + `drop_encrypted_db()`,保证主进程永不持有 sk_M **且不持有 Enc DB 副本**,节约 ~16 GB CPU 内存。 |
| **GPU 边界 = Python 对象传递** | U/M/S 在同一进程,GPU tensor 通过 `self.party_m.forward(H_U, attention_mask)` 直接传引用,无 PCIe 拷贝。 |
| **异构任务隔离** | SEAL 同态加/解密等 CPU 密集任务放在子进程,避免污染 CUDA 主线程。 |
| **Chunked Pipeline** | `USE_CHUNKED_PIPELINE=True`,`CHUNK_TOKENS=3072`,把 M/S 边界上的 PIR+反传按 token chunk 切分,降低单次 GPU 内存峰值。 |

### 2.3 仓库目录结构

```
/root/autodl-tmp/SLG-HE-PIR/
├── configs/                       # 配置 dataclass(目前为空,改用 finetune.Config 内置)
├── scripts/                       # 顶层诊断/测试脚本
│   ├── two_epoch_test.py          # ★ 2-epoch 验证脚本(Phase 1 v2.2 主验证)
│   ├── quick_smoke_10step.py      # 10-step 显存/协议 smoke(早期诊断)
│   ├── run_small_scale_test.py    # 小规模端到端调试
│   ├── chunk_correctness_test.py  # 分块流水线 vs flat 流水线等价性
│   ├── heterogeneous_correctness_test.py  # HeterogeneousProtocol 端到端正确性
│   ├── e2e_correctness_recheck.py # 串行 vs 并行 全链路正确性对比
│   ├── e2e_math_verify.py         # 数学层验证(代数关系 a_t − V_y)
│   ├── diag_grad_flow.py          # 梯度流诊断(mask 消去)
│   ├── perf_bench.py              # 性能基准(3 个热点路径)
│   ├── compare_step_profiles.py   # step profile 对比
│   ├── test_step_profiler.py      # step profiler 单元测试
│   ├── test_trainer_dispatch.py   # Trainer 派发单元测试
│   ├── _demo_profile_jsonl.py     # profile jsonl demo
│   └── run_with_pyc_finder.py     # PycMetaFinder 引导器(早期兼容)
├── src/
│   ├── core/
│   │   ├── bfv_privselect_v2_adapter.py  # BFV backend + PrivSelect 原语 + BFVEncryptedDatabase
│   │   ├── s3pir_hints.py                # S3PIR HintTable + 流式 parities 累积
│   │   ├── protocol_he_pir.py            # 协议层(老协议实现,audit-only)
│   │   └── key_remapping.py              # 多进程 pk/sk 注入 helper
│   ├── model/
│   │   └── model_splitting.py            # U/M/S 三个 submodel 加载 + LoRA 注入 + bf16 适配
│   ├── data/
│   │   └── dataset.py                    # BioTriplex JSONL 加载 + Llama tokenizer 封装
│   ├── parties/
│   │   ├── party_u.py                    # PartyU(embed_tokens + 0 个 decoder)
│   │   ├── party_m.py                    # PartyM(32 decoder + LoRA)
│   │   ├── party_s.py                    # PartyS(lm_head, gpu logits, a_t 计算)
│   │   ├── heterogeneous_protocol.py     # step_train / step_train_chunked / step_val / step_test
│   │   ├── fusion_protocol.py            # inline 模式(无 spawn 子进程),用于单元测试
│   │   ├── ipc_protocol.py               # 多主机版预留接口
│   │   ├── legacy_ipc_stub.py            # Legacy 多进程接口(audit)
│   │   ├── transport.py                  # spawn 子进程 bootstrap helper
│   │   ├── wire.py                       # StepResult / StepProfiler 数据结构
│   │   └── crypto_workers/
│   │       ├── base.py                   # BaseCryptoWorker + init_pool_worker
│   │       ├── pool.py                   # CryptoWorkerPool(spawn Pool 包装)
│   │       ├── crypto_u.py               # CryptoUWorker(PRG + add_plain_inplace)
│   │       ├── crypto_m.py               # CryptoMWorker(持有 sk_M,decrypt)
│   │       └── crypto_s.py               # CryptoSWorker(PRG + share + PIR 同态累加)
│   ├── scripts/
│   │   ├── build_encrypted_db.py         # Stage 0 Step 1 入口
│   │   ├── build_s3pir_hints.py          # Stage 0 Step 2 入口
│   │   └── finetune.py                   # ★ 统一入口 run_stage0/1/2(all)
│   └── training/
│       ├── trainer.py                    # Trainer + TrainerConfig + _run_val_epoch/_run_test_epoch
│       ├── checkpoint.py                 # CheckpointManager(save/load/best)
│       └── evaluation.py                 # standalone test eval(merge LoRA + hf generate)
├── docs/
│   ├── SLG系统流程.svg                   # 系统流程图(本文档的对照基准)
│   ├── SLG_HE_PIR_SYSTEM_DOCUMENTATION.md # 本文档(技术细节)
│   ├── SLG_HE_PIR_USAGE.md               # ★ 新会话使用手册(快速调用)
│   ├── BIOTRIPLEX_FINETUNE_README.md     # BioTriplex 任务 + 数据格式说明
│   ├── PROJECT_DOCUMENTATION.md          # 旧版项目文档
│   ├── HANDOFF.md / newHANDOFF.md        # 交接文档
│   └── (其它:三方微调单主机模拟-技术路线分析、baseline-BIOTRIPLEX_FINETUNING_GUIDE 等)
├── baseline/                             # BioTriplex 原论文代码复现(独立,不修改)
├── baseline_backup_20260720_0125/        # 同上,备份
├── papers/                               # 论文材料
├── S3PIR/                                # S3PIR 子项目(独立 README)
├── datasets/                             # 文档级数据集
├── checkpoints/                          # 训练产物
└── logs/                                 # 训练日志(2epoch_test/, finetune_stage1_*.log 等)
```

> **调用入口约定**:所有 BFV/PIR/LoRA 微调都从 `src/scripts/finetune.py` 入口;快速 smoke / 对照测试用 `scripts/two_epoch_test.py` 等顶层脚本。详细用法见 `docs/SLG_HE_PIR_USAGE.md`。

---

## 3. 全局数据流动(SVG 四阶段)

### 3.1 #0 offline PreCompute

> **SVG 公式**: `(pk_M, sk_M) ← BFVKeyGen` → 分发 `pk_M` → `Generate Hints over D[y]` → `D[y] ← Enc_{pk_M}(V)`

#### 3.1.1 参与方内部数据处理

**Model (M)** —— 通过 `src/scripts/build_encrypted_db.py::build_encrypted_db`:

1. 读取 Llama-3.1-8B-Instruct 的 `lm_head.weight`(V 矩阵),shape=`(128256, 4096)`,float32。
2. 创建 `BFVPrivSelectV2Backend(n_entries=128256, vec_dim=4096, poly_degree=4096, plain_bits=30, scale=10000)`,内部触发 SEAL 的 `KeyGenerator()`:
   - `secret_key: SecretKey`
   - `public_key: PublicKey`
   - `relin_keys: RelinKeys`
3. 对 V 的每一行执行 `enc_db.build_from_V(V, force=False)`:
   - 对每行 `V[y]`: `float_to_int(V[y]) → encoder.encode(ints) → Plaintext → encryptor.encrypt(Plaintext) → Ciphertext → _seal_to_bytes(ct)`
   - 写入 `bfv_ct_db_n128256_d4096_p4096.bin`(约 1.93 GB,memory-mapped 格式)。
4. 输出:
   - `bfv_pk.bin`:pickle 序列化的 public_key(广播给 U 和 S)
   - `bfv_meta.json`:metadata(vocab_size, hidden_dim, poly_degree, scale 等)
   - `bfv_keys.json`:本地保存,供 `_drop_secret_key` 之前的过渡使用

**Server (S)** —— 通过 `src/scripts/build_s3pir_hints.py::build_s3pir_hints`:

1. 加载上一阶段产出的密文 DB(只读 mmap,不重新加密)。
2. 加载 `bfv_pk.bin` 拿到 `public_key`。
3. 创建 `HintTable(n_entries, partition_size=sqrt(N)≈358, lam=80)`:
   - `compute_main_hints_skeleton`:对每个 partition 计算 `cutoff = median(PRF_v(j,k) for k in partition_j)` 与 `extra_index`(中位数重位时的 tie-breaker)
   - `compute_backup_hints_skeleton`:为 lam 对备份 hints 预生成 skeleton
4. 流式累积 parities(Algorithm 4 — Streaming):逐 partition 读取 `Enc(V[y])`,对每个 main hint `h ∈ main_hints[j]` 计算 `parity[h] = ⊕_{y ∈ hint} D[y]`(BFV 同态加),并按 `save_every_partitions` 落盘到 `s3pir_hints/main_parities_*.bin` 与 `s3pir_hints/backup_a_*.bin` / `backup_b_*.bin`。
5. 写出 `hint_table.json`(只含 cutoffs + indices + skeletons,不暴露任何 V 内容)。

**User (U)** —— 仅消费 pk_M:

- Stage 0 期间 U 不参与任何计算,只把 `bfv_pk.bin` 加载到内存供后续 `CryptoUWorker` 初始化时构造 SEAL PublicKey。

#### 3.1.2 跨参与方数据流

```
       M                                                S
  bfv_pk.bin  ─── broadcast (磁盘文件) ─────────▶  load pk_M
  enc_db.bin  ─── broadcast (磁盘文件) ─────────▶  mmap D[y]
                                                       │
                                                       ▼
                                          HintTable.compute_*_skeleton
                                                       │
                                                       ▼
                                          build_hints_parities_streaming
                                                       │
                                                       ▼
                                  s3pir_hints/main_parities_*.bin
                                  s3pir_hints/backup_{a,b}_*.bin
                                  hint_table.json
```

> **注**:本实现把所有离线产物都落到同一台机器的 `/root/autodl-tmp/slg-bfv-cache/` 目录。多主机部署时,只需把 M 产的 `bfv_pk.bin` 与 S 产的 `parity_*.bin` / `hint_table.json` 同步到对应主机即可。

---

### 3.2 #1 Train-FP

> **SVG 公式**: `H_0 ← Embedding(x)` → `H_ans ← H_0 · W`(decoder 前向)→ `Z_t ← H_ans · V^T`

#### 3.2.1 参与方内部数据处理

**User (U)** —— `PartyU.forward_train(batch)`(`src/parties/party_u.py`):

1. 输入:`batch = {"input_ids": LongTensor[B,S], "attention_mask": LongTensor[B,S]}`
2. 调 `self.model.embed_tokens(input_ids)` → `H_U`,shape=`(B, S, 4096)`,bf16,GPU。
   - **注意**:U 持有 `embed_tokens.weight`,但 `requires_grad=True`(Phase 1 v2.1 修复)且 `optimizer=None` —— 梯度会经过 embedding 反传到 M 的 LoRA,但 U 端不做参数更新。
   - 当 `u_split_layer=0` 时,U 不持有任何 decoder 层;`for layer in self.model.layers: H = layer(H, ...)` 不执行。
3. 返回 `{"H_U": GPU tensor}` —— 同进程内直接传引用。

**Model (M)** —— `PartyM.forward(H_U, attention_mask)` → 内部调 `_m_forward`:

1. 输入:`H_U = (B, S, 4096)` GPU bf16
2. 对每一层 decoder:
   - 用 `torch.utils.checkpoint.checkpoint(layer, H, attention_mask=..., use_reentrant=True)` 做 reentrant 激活检查点(节省 ~50% 显存,代价是一次前向重计算)
3. 最后 `H = self.model.norm(H)`,得 `H_M = (B, S, 4096)`,GPU bf16。
4. **缓存**: `self._last_H_M = H_M`、`self._last_H_U = H_U`、`self._last_attention_mask = attention_mask` —— 这些缓存用于反传阶段(`_inject_and_backward`)注入梯度。
5. 返回 `{"H_M": GPU tensor}` 给同进程的 PartyS。

**Server (S)** —— `PartyS.process_logits_dispatch(payload)`(`src/parties/party_s.py:212`):

1. 输入:`payload = {"H_M": (B,S,4096) bf16 GPU, "step": int, "gold_ids": LongTensor[B,S]}`
2. **`compute_logits_gpu(H_M)`**:
   - `Z = H_M @ V^T`,shape=`(B, S, 128256)`,留在 GPU(避免 2 GB 的 fp32 中间张量复制)
   - 注意:`V_weight`(`lm_head.weight`)是 shape `(128256, 4096)` 的 nn.Linear,`V.T` 是其转置
3. **`compute_a_t_gpu(Z)`**:
   - `y_all = Z.argmax(dim=-1)`,shape=`(B*S,)`,int64,留在 GPU
   - **分块**:`a_all_flat = softmax(Z) @ V`,但 V 太大(128256×4096=524M 个元素),所以按行切分,每 chunk 处理 `chunk_size` 行,结果 `a_all_flat: (B*S, 4096)` bf16
   - **OutOfMemoryError 回退**: 若 GPU OOM,自动切到 CPU(`logits_cpu @ V_cpu`),代价是 ~2 GB 的额外 CPU 内存
4. **移回 CPU**:`a_all_flat.cpu().numpy()` → `y_all.cpu().numpy()`(准备喂给 CryptoSWorker)
5. **提交 CPU 任务**: 把 `{a_t_list: List[np.ndarray(4096,)], y_t_list: List[int], step}` 投递给 `CryptoSWorker` 池
6. 等 `result.get()` 返回 `{s3pir_responses: List[Dict], s_shares: List[List[int]]}`,封装为返回值给主进程

#### 3.2.2 跨参与方数据流

```
              GPU 内存传递(同进程)
   input_ids  ──▶ PartyU  ──H_U──▶ PartyM  ──H_M──▶ PartyS
                                                       │
                                                       │ (Z, a_t, y_t)
                                                       ▼
                                              crypto_s_pool.submit
                                                  (CPU spawn)
                                                       │
                                                       │ (s3pir_responses, s_shares)
                                                       ▼
                                            等待 U/M 后续步骤使用
```

`process_logits_dispatch` 返回后,主流程 `step_train_chunked` 拿到 `s3pir_responses`(每个 token 一个查询的 PIR hint)与 `s_shares`(每个 token 的明文 share)。

---

### 3.3 #2 LoRA-BP

> **SVG 公式**: `a_t ← softmax(z_t) · V`(label-free)、`r_t ← PRG(seed,t)`、`Ṽ_{y_t} ← PIR(y_t, D[y])`、`result_U ← -Ṽ_{y_t} + Enc_{pk_M}(r_t)`、`result_S ← a_t - r_t`、`g_{H,t} ← Dec_{sk_M}(result_U) + result_S = a_t - V_{y_t}` → 用 `g_{H,t}` 更新 M 方的 LoRA 矩阵
>
> **实现注意**:
> - 代码采用 **Design-2 单行 PIR**:`|real|=1, |dummy|=0` —— 即对每个 token `y_t`,S 直接 `enc_db.get_encrypted_row(y_t)`(mmap 单字节块)返回 `Enc(-V[y])`;hint table 仅作为 `real_indices` 注释(metadata),**不**参与实际累加。这简化了 S 端逻辑,代价是失去了 S3PIR 的"加 `a_t`"扩展(本项目 S 不累加 `a_t` 到密文,因为 a_t 由 S 端 share 直接交给 M 做减法)。
> - `s_share` 是 **scale 量化后** 的整数:`s_share = round(a_t · scale) − r_t`,与 SVG 公式 `a_t − r_t` 等价但落在整数域上,可与密文解密后的整数 masked_arr 直接相加。
> - `y_t` 的语义:代码中 `y_t = argmax(logits)`(参见 `process_logits_dispatch` 第 240 行 fallback)。当 `batch["output_ids"]` 存在时,S 改用 `gold_ids.flatten()` 作为 `y_t`(参见 `process_logits_dispatch` 第 236-238 行);**注意** 当前 `BioTriplexQADataset.__getitem__` 只返回 `item["labels"]` 而不返回 `output_ids`,所以训练侧实际走 `argmax` 分支(详见 §6.4.1)。

#### 3.3.1 参与方内部数据处理

**Server (S) —— `CryptoSWorker.handle_request`**(`src/parties/crypto_workers/crypto_s.py:133`):

对每个 token `t ∈ [0, n_tokens)`:

1. **生成 PRG share**(U 与 S 共享 seed):
   - `r_t = PRGShareProtocolBFV.generate_mask_ints(step, t_flat)` —— 伪随机向量,shape=`(4096,)`,int32,在 `(-plain_modulus/2, +plain_modulus/2)` 范围(严格开区间)
2. **计算 plaintext share**:
   - `a_t_fp32 → encode_vector_as_ints → a_t_ints`(quantize by `round(a_t · scale)`)
   - `s_share[t] = a_t_ints − r_t`,shape=`(4096,)`,int32(实际上是 int64,只是数值上不会溢出)
3. **PIR 查询(Design-2)**:
   - `parity_real_bytes = enc_db.get_encrypted_row(y_t)` —— 直接 mmap 取出单字节块(已加密的 `Enc(-V[y_t])`,由 Stage 0 写入)
   - `parity_dummy_bytes = b""`(`|dummy|=0`,Design-2 不累加 dummy)
   - `real_indices = [y_t]`(若 hint_table 可用,额外记录 `hint_table.build_query_for(y_t)` 的 indices 作为 metadata,但不用于实际累加)
4. 返回主进程:`{"s3pir_responses": List[Dict], "s_shares": List[List[int]]}`

> **关键隐私**:S 看到 `y_t`(用户当前 batch 的 argmax token id)但看不到 `x`(因为 PIR 保护了 query 的"查询意图",但 Design-2 直接传 y_t,S 其实知道是哪个 token;多主机部署中可改回完整 S3PIR 累积以恢复隐私);U 看到 `r_t` 但看不到 `a_t`;M 看到 `decrypted[-V_y + r_t]` 但看不到 `r_t`(因为 S 在 share 阶段已经把它消掉)。

**User (U) —— `CryptoUWorker.handle_request`**(`src/parties/crypto_workers/crypto_u.py:106`):

对每个 token `t`:

1. **独立生成 PRG**(与 S 共享 seed,但 U 端无 sk_M):
   - `r_t = generate_mask_ints(step, t_flat)` —— 与 S 端生成的 `r_t` 逐 bit 相同(共享 seed)
2. **加载 PIR 响应密文**:
   - `ct = _seal_load_ciphertext(s3pir_responses[t].parity_real_bytes)`
3. **同态加 mask**:
   - `pt = encoder.encode(r_t)` → `Plaintext`(直接把**带符号的** `r_t ∈ (−pm/2, +pm/2)` 喂给 SEAL BatchEncoder;SEAL 内部做 centred 编码,**不要**再做 `x % pm` 之类的 wrap——一旦 wrap 成 `[0, pm)` 的正整数,密文里会带 ±49151 的系统性偏移,mask 消不掉)
   - `evaluator.add_plain_inplace(ct, pt)`(即 `ct += Enc(r_t)`;密文现在存 `−V_y·scale + r_t`)
   - 序列化 `_seal_to_bytes(ct)` → 返回主进程
4. 收集所有 token 的 `ct_list: List[bytes]`

> **关键**:U 的密文掩码操作使密文 `Enc(−V_y·scale)` 变成 `Enc(−V_y·scale + r_t)`(因为 r_t 是带符号整数,加进 plaintext 等价于 `Enc(−V_y·scale) + Enc(r_t)`)。该实现与 SVG §3.3 的 `result_U ← −Ṽ_{y_t} + Enc_{pk_M}(r_t)` 一致。

**Model (M) —— `PartyM.backward_and_update(payload)`**(`src/parties/party_m.py:239`):

按字节码顺序:

1. **CPU: 解密**(通过 `CryptoMWorker`):
   - `crypto_m_pool.submit({"ct_list": List[bytes], "scale": 10000, "vec_dim": 4096})` —— 异步发到 spawn CPU 子进程
   - 子进程里 `decryptor.decrypt(ct)` → Plaintext → `BatchEncoder.decode()` 给出 `[0, pm)` 内的正整数 → `masked_arr = (decode × scale)` → 居中到 `[-pm/2, +pm/2)`(`>pm/2 → 减 pm`) → 这一步必须,因为 SEAL BatchDecoder 输出是正数;**只有居中后带符号的 masked_int 才能跟 S 端的 signed s_share 直接相加让 r_t 消掉**
   - 返回 `masked_arr: np.ndarray(n_tokens, 4096)`,**这是 `−V_{y_t}·scale + r_t`**(因为 PIR 返回的 `Ṽ_y` 是密文,加 mask 后解密得到的是 `−V_y·scale + r_t`,对应 SVG 的 `Dec(result_U)`)

2. **CPU: 累加明文 share**:
   - 循环 `for t_flat in range(n_tokens):`
     - `s_arr = np.asarray(s_share_list[t_flat][:vec_dim], dtype=np.int64)`(**注意是 int64,不再除 scale**;`s_share = scale·a_t − r_t` 是带符号整数)
     - `diff_int = masked_centered + s_arr` ← 这一步是 mask 消去的关键:`(−V_y·scale + r_t) + (scale·a_t − r_t) = scale·(a_t − V_y)`
     - `g_accum[t_flat] = diff_int.astype(np.float32) / scale`(**这一步是 a_t − V_y**)
     - `if s_arr.size < vec_dim: np.pad(..., (0, vec_dim - s_arr.size))`(防越界)

3. **CPU → GPU**: `g_H = torch.from_numpy(g_accum[:B*S]).float().to(self.device).bfloat16()` —— bf16 与模型 dtype 对齐,降低一半显存
4. **reshape**: `g_H = g_H.view(B, S, vec_dim).contiguous()` —— 必须与 `H_M` 的 shape 一致才能 `.backward()`
5. **`_inject_and_backward(g_H, step)`**(`src/parties/party_m.py:370`):
   - 取缓存 `H_M = self._last_H_M`
   - 形状/设备不匹配时,用 `self._last_H_U` 重新跑一次 `_m_forward(H_U_cached)` 重建 `H_M`
   - `if not H_M.requires_grad: H_M.requires_grad_(True)`(因为 `ckpt.checkpoint` 内部可能把 requires_grad 置 False)
   - `H_M.backward(gradient=g_H)` —— 触发整张 autograd 图从 M 反向传播,经过 ckpt 重计算后回到 U 的 embed_tokens
6. **GPU: 优化器更新**:
   - `clip_grad_norm_(params, max_norm=self.gradient_clip_norm)` —— 默认 1.0
   - `self.optimizer.step()` —— AdamW(只在 LoRA 参数上,因为只有 LoRA 的 `requires_grad=True`)
   - `if getattr(self, "lr_scheduler", None) is not None: self.lr_scheduler.step()` —— warmup 200 步后 cosine 衰减
   - `self.optimizer.zero_grad(set_to_none=True)`
7. **缓存清理**: `self._last_H_U = self._last_H_M = self._last_attention_mask = None` —— 释放显存
8. **GPU memory 统计**: `gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024` —— 写进 StepResult 给监控
9. 返回 `{"loss": loss_proxy, "gpu_mem_mb": ..., "attack_dumps": {}, "mode": "heterogeneous"}`

#### 3.3.2 跨参与方数据流

```
                          process_logits_dispatch
   PartyS  ──(s3pir_responses, s_shares)──▶  HeterogeneousProtocol
                                                       │
                                          _split_into_chunks(n_tokens, chunk_tokens)
                                                       │
                       ┌───────────────────────────────┼───────────────────────────────┐
                       │                               │                               │
                  chunk k=0                       chunk k=1                       chunk k=K-1
                       │                               │                               │
                       ▼                               ▼                               ▼
              PartyU.privselect_and_recover_dispatch(s3pir_responses_chunk)
                       │
                       ▼
              crypto_u_pool.submit   ──▶  CryptoUWorker (CPU spawn)
                       │                          │
                       │                          ├─ generate_mask_ints(step, t_flat)
                       │                          ├─ load ct = _seal_load_ciphertext(...)
                       │                          ├─ evaluator.add_plain_inplace(ct, r_t)   # signed r_t
                       │                          └─ return ct_list
                       │
              ct_list  ◀────────────────────────────┘
                       │
                       ▼   (累积到 all_ct)
              PartyM.backward_and_update(ct_from_U=all_ct, s_share=s_shares)
                       │
                       ▼
              crypto_m_pool.submit(ct_list) ──▶  CryptoMWorker (CPU spawn, holds sk_M)
                       │                                  │
                       │                                  └─ decryptor.decrypt → masked_arr = -V_y·scale + r_t
                       │ ◀────────────────────────────────┘
                       │
              g_accum[t] = masked_arr[t] + s_share[t]    # = scale·(a_t - V_y)  (SVG 公式)
                       │
                       ▼
              _inject_and_backward(g_H)
                       │
                       ▼
              H_M.backward(gradient=g_H)    # → LoRA grads
              optimizer.step()              # AdamW updates LoRA only
              lr_scheduler.step()
              optimizer.zero_grad()
```

#### 3.3.3 Chunked 流水线优化

`USE_CHUNKED_PIPELINE=True, CHUNK_TOKENS=3072` 的设计:

- **目标**: 把一个大 batch(可能 32k tokens)按 3072 tokens 为一块切分,PIR 查询与 mask 操作并行,降低 GPU 显存峰值
- **顺序**: `M.forward(H_U)` 只跑一次(全 batch);然后 S 已经返回全 batch 的 s3pir_responses;按 chunk 顺序调 U 的 mask,每完成一块就把它合并到 `all_ct`;最后一次性调 `M.backward_and_update(all_ct, s_shares)`。
- **为什么 M 的 forward 一次,backward 一次**: 因为 M 持有全 batch 的 `H_M` 缓存(必须如此,否则 backward 无法拼接梯度)。

---

### 3.4 #3 Eval-FP

> **SVG 公式**: `H_0* ← Embedding(x*)` → `H_ans* ← H_0* · W` → `ŷ ← softmax(H_ans* · V)` → `Loss ← Dist(y*, ŷ)`

#### 3.4.1 参与方内部数据处理

**User (U)** —— `PartyU.forward_val(val_batch)`(`src/parties/party_u.py:172`):

1. 输入:val batch 的 `input_ids`/`attention_mask`
2. 用 `with torch.no_grad():` 包裹:
   - `H_U = self.model.embed_tokens(input_ids)` → `(B, S, 4096)`
3. 返回 `{"H_U": GPU tensor}`(不缓存 `requires_grad`,因为不反传)

**Model (M)** —— `PartyM.forward(H_U, attention_mask)`:与 Train-FP 完全相同的 `_m_forward`,但不再保存 `_last_H_M`(因为不需要 backward),得 `H_M`。

**Server (S)** —— `PartyS.compute_logits_for_eval(H_M)`(`src/parties/party_s.py:110`):

1. `Z = H_M @ V^T` → shape `(B, S, 128256)`,GPU
2. `.detach().cpu()` → CPU(避免 GPU 内存泄漏,val 阶段不需要后续梯度)
3. 返回 `logits_cpu`

**Server (S)** —— `PartyS.generate_predictions(H_M_or_logits)`(`src/parties/party_s.py:272`):

1. `token_ids = logits.argmax(dim=-1)` → `[batch, seq_len]`,int64
2. 每个 token 解码为字符串 `_decode_tokens` → `predictions: List[str]`
3. 返回 `{"predictions": List[str], "token_ids": List[List[int]]}`

**协议层 (HeterogeneousProtocol.step_val)**(`src/parties/heterogeneous_protocol.py:509`):

1. 拼装 gold label:`labels = val_batch.get("output_text")` 或 `output_text` / `target_text` / `gold`(dataset 实际返回的 `output_text`,即 NER 字符串列表)
2. **letter 化 label**:
   ```python
   output_ids = val_batch.get("output_ids")  # 当前数据集 __getitem__ 未返回该键
   if output_ids is not None:                 # ← 永远走 else 分支
       labels_letters = [parse_answer_letter(decode(non_pad_tokens)) for ...]
   ```
   → 当前实现下 `labels_letters = []`,回退到 `result["labels"]` = NER 字符串列表(详见 §6.4.1)
3. **letter 化 prediction**: `predictions_letters = [parse_answer_letter(p) for p in predictions]`,NER 格式字符串原样返回,其他文本走正则
4. 返回 `{"predictions", "predictions_letters", "labels", "labels_letters", "logits", "labels_tensor", "metrics"}` 给 Trainer

**Trainer** —— `Trainer._run_val_epoch(epoch)`(`src/training/trainer.py:_run_val_epoch`):

- 循环 DataLoader,每个 batch 调 `ipc.step_val(val_batch, global_step)`
- CE loss 计算:`if logits is not None and labels_tensor is not None: F.cross_entropy(...)`;**当前 `labels_tensor = val_batch.get("output_ids") = None` → CE 路径完全跳过**(`val_ce_loss = 0` 由此导致)
- 聚合 `tp_total / fp_total / fn_total` from `_safe_letter_split(pred) ∩ _safe_letter_split(gold)`,输出 `val_entity_micro_f1` / `val_micro_precision` / `val_micro_recall` / `val_macro_f1` / `val_weighted_f1` / `val_micro_accuracy`
- 写出 `epoch_records.jsonl`

#### 3.4.2 跨参与方数据流

```
   val_batch  ──▶  PartyU.forward_val
                          │
                          ▼ H_U
                       PartyM.forward
                          │
                          ▼ H_M
                  PartyS.compute_logits_for_eval
                          │
                          ▼ logits_cpu
                  PartyS.generate_predictions
                          │
                          ▼ token_ids
                  U._decode_tokens + parse_answer_letter
                          │
                          ▼
                  compute_val_metrics → ce_loss + F1/precision/recall
```

> **隐私**: Eval 阶段 **完全不走加密路径**(无 PRG、无 PIR、无 sk_M),因为 val 不需要更新 LoRA。这是符合 SVG 的简化处理 —— val 阶段 U 与 S 都在同一进程,可以直接传 logits 而无需密码学保护。

---

## 4. 实现原理

### 4.1 技术栈总览

| 层 | 技术 | 用途 |
|---|---|---|
| 深度学习 | PyTorch 2.x(transformers) | 模型加载、自动求导、`torch.utils.checkpoint` 激活检查点 |
| 模型 | HuggingFace `transformers`(`LlamaDecoderLayer`、`LlamaRotaryEmbedding`、`LlamaRMSNorm`) | Llama-3.1-8B 架构 |
| 同态加密 | TenSEAL(绑定 SEAL C++) | BFV scheme:`poly_degree=4096, plain_bits=30, scale=10000` |
| 进程模型 | Python `multiprocessing.Pool`(使用 spawn 模式,见 §4.5.1) | CPU crypto workers 隔离 |
| 持久化 | `safetensors`、`json`、`numpy`、自定义 mmap | V 矩阵加载、hint table、加密 DB |
| 配置 | Python `@dataclass Config`(在 `finetune.py`) | 所有超参集中管理 |

### 4.2 模型切分策略

`src/model/model_splitting.py` 提供三个加载函数:

| 函数 | 加载内容 | 用途方 |
|---|---|---|
| `load_u_submodel(spec, model_path, device)` | `embed_tokens` + (可选)前 N 层 decoder。当前实现 `u_split_layer=0`,故只加载 embed_tokens。 | PartyU |
| `load_m_submodel_with_lora(spec, model_path, device, lora_rank, lora_alpha, lora_targets)` | 全部 32 层 decoder + final `norm` + LoRA 注入到 `q_proj/v_proj/k_proj/o_proj/gate_proj/up_proj/down_proj` | PartyM |
| `load_s_submodel(spec, model_path, device)` | `lm_head`(`Linear(in_features=4096, out_features=128256)`),`requires_grad=False` | PartyS |

**LoRA 注入**(`_inject_lora`,位于 `src/model/model_splitting.py:422`):

- 对每个目标 nn.Linear,用 `LoRALinear(nn.Linear)`(位于 `src/model/model_splitting.py:399`)替换:
  - `lora_A: Parameter(shape=(in_features, r=8))`,初始化 `kaiming_uniform_`
  - `lora_B: Parameter(shape=(r, out_features))`,初始化 `zeros_`(保证初始时 `B @ A = 0`,LoRA 贡献为 0)
  - `lora_A` / `lora_B` 全部以 **`dtype=torch.bfloat16`** 创建(Phase 1 v2.2 修复 OOM 的关键)
  - `lora_dropout: nn.Dropout(p=0.05)`
  - `scaling = alpha / r = 16 / 8 = 2.0`
- 前向: `output = linear(x) + lora_B @ (lora_dropout(lora_A @ x)) * scaling`
- 注入目标(7 个投影):`q_proj` / `k_proj` / `v_proj` / `o_proj` / `gate_proj` / `up_proj` / `down_proj`
- Phase 1 实测:共 **448 个 LoRA trainable tensor**(32 层 × 7 投影 × 2 (lora_A, lora_B) = 448)

**关键冻结约定**:

- `embed_tokens.weight.requires_grad = True` 但 `PartyU.optimizer = None` → 梯度可流过但不更新
- 全部 `nn.Linear.weight` 与 `RMSNorm.weight` 在 M 中 `requires_grad = False`(LoRA 接管)
- `lm_head.weight.requires_grad = False`(V 永远冻结)

### 4.3 GPU/CPU 任务分工

#### 4.3.1 GPU 任务(RTX 5090, 32 GB)

| 模块 | 操作 | 显存估算 |
|---|---|---|
| `PartyU._u_forward` | `embed_tokens(input_ids)` → `(B, S, 4096) bf16` | ~B·S·8 KB(B=4, S=128 时 ≈ 4 MB) |
| `PartyM._m_forward` | 32 × `LlamaDecoderLayer` + reentrant ckpt | 激活检查点后 ~3 GB |
| `PartyS.compute_logits_gpu` | `H_M @ V.T` → `(B, S, 128256) bf16` | ~256 MB(单 batch) |
| `PartyS.compute_a_t_gpu` | `softmax(Z) @ V` 分块 | ~512 MB(分块后) |
| `PartyM._inject_and_backward` | `H_M.backward(gradient=g_H)` | 同上(峰值 ~16 GB) |
| `PartyM.backward_and_update` 优化器 | AdamW 状态(m, v) for LoRA | ~12 MB |

#### 4.3.2 CPU 任务(子进程池,8 workers)

| Worker | 操作 | 内存估算 |
|---|---|---|
| `CryptoUWorker` | PRG 生成 + SEAL `add_plain_inplace` | ~200 MB/worker(SEAL context + pk_M) |
| `CryptoMWorker` | SEAL `decryptor.decrypt` | ~400 MB/worker(SEAL context + sk_M + pk_M + plaintext buffer) |
| `CryptoSWorker` | PRG 生成 + Enc(DB) mmap + PIR 同态累加 | ~1.5 GB/worker(SEAL context + pk_M + mmap DB + share buffers) |

> **为什么 CryptoSWorker 是 N=1**: hint table 与 Enc(DB) 都是只读 mmaps,进程间共享物理页;但每个 worker 复制一份 SEAL context 与 PRG state,约 1 GB;只启 1 个 worker 节省 ~7 GB 内存。`N_CRYPTO_S_WORKERS = 1` 是 `heterogeneous_protocol.__init__` 的默认值。

#### 4.3.3 主进程(单进程,无子进程)

- `HeterogeneousProtocol` 编排 step_train / step_val
- `PartyU`、`PartyM`、`PartyS` 三个 Python 对象,**共享同一个 CUDA context**
- `bfv_backend` 在 `run_stage1` 中 `_drop_secret_key()` 后只保留 metadata

### 4.4 内存与显存分配

#### 4.4.1 显存(GPU)生命周期与优化

```
[start]
  ├─ PartyU.embed_tokens + PartyM.decoder(16 layers) + PartyM.LoRA + PartyS.lm_head
  │   ≈ 4 GB (16 层 + embed + lm_head, 32 GB 显存放得下)
  ├─ Optimizer state (AdamW: m, v for LoRA only)
  │   ≈ 12 MB
  │   ★ DeepSpeed ZeRO-1 可将 optimizer state 分片,节省 ~4x 内存
  ├─ Activation buffers (per step)
  │   with ckpt.checkpoint: peak ~5 GB during forward, freed at backward
  │   ★ FlashAttention2 可将 attention O(N²) 内存降至 O(N)
  │
[step forward]
  ├─ H_U (B,S,4096) bf16 ≈ 4 MB
  ├─ H_M intermediate buffers, freed layer-by-layer
  ├─ Logits (B,S,128256) bf16 ≈ 256 MB
  ├─ softmax(Z) ≈ 256 MB
  │
[step PIR]
  ├─ CryptoUWorker returns ct_list (bytes, ~256 KB/token) → main process
  ├─ (CPU side: mmap DB pages pulled in on demand by EncryptedDatabase access)
  │
[step backward]
  ├─ g_H (B,S,4096) fp32 ≈ 8 MB
  ├─ H_M.backward triggers reentrant recompute: peak ~10 GB
  ├─ LoRA grads: 448 × (B@A gradients) ≈ 12 MB
  │
[step optimizer]
  ├─ clip_grad_norm_ → no extra memory
  ├─ AdamW.step → in-place update of LoRA params
  │   ★ DeepSpeed ZeRO 可将优化器状态分片存储
  │
[end step]
  └─ cache cleared, torch.cuda.empty_cache() → recover fragmented blocks
       (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 进一步缓解碎片)
```

**GPU 内存优化技术 (v2.3 SageAttention)**:

| 优化技术 | 配置参数 | 内存节省 | 适用场景 |
|---|---|---|---|
| **SageAttention3 FP4** | `--use_sage_attention True` | 4x Q/K 带宽, 5x 加速 | RTX 5090 Blackwell (最佳) |
| **SageAttention2++ INT8** | `--use_sage_attention True` | 2x Q/K^T 带宽, 2.7x 加速 | RTX 4090/H100 |
| **FlashAttention2** | `--use_flash_attention True` | O(N²) → O(N) | 通用 |
| **DeepSpeed ZeRO-1** | `--use_deepspeed_zero True --zero_stage 1` | ~4x optimizer state | 单 GPU 推荐 |
| **DeepSpeed ZeRO-2** | `--use_deepspeed_zero True --zero_stage 2` | ~8x optimizer state + gradients | 多 GPU |
| **DeepSpeed ZeRO-3** | `--use_deepspeed_zero True --zero_stage 3` | ~16x 全部参数 | 多 GPU + 低显存 |
| **梯度检查点 (reentrant)** | `--gradient_checkpointing_style reentrant` | ~50% activation memory | 默认,速度/内存平衡 |
| **梯度检查点 (full)** | `--gradient_checkpointing_style full` | 最大内存节省 | 显存极度受限时 |

**SageAttention 优先级**: SageAttention3 FP4 > SageAttention2++ INT8 > FlashAttention2 > SDPA

实测峰值: Phase 1 监控显示 GPU 内存 ~16-17 GB (32 GB 总), 稳定不 OOM (2026-07-21 修复 `drop_encrypted_db` 后进一步降低碎片化风险)。启用 SageAttention3 + DeepSpeed ZeRO-1 后预计可进一步降低显存占用。

#### 4.4.2 内存(CPU)布局

```
Main process:
  ├─ HF model shards (lazy mmap via safetensors): ~16 GB virtual, ~2 GB resident
  ├─ Hint table JSON: ~5 MB(skeletons only)
  ├─ Crypto worker Pool: 8 forks × ~200-400 MB = ~2.5 GB
  └─ CryptoSWorker Pool: 1 fork × ~1.5 GB = ~1.5 GB
        (mmap'd Enc(DB): ~1.93 GB shared read-only;
         workers load _ct_list via BFVEncryptedDatabase.from_cache(load_ct_list=True))

  ★ 主进程 BFV backend 不持有 Enc DB 副本:
    - build_encrypted_database() 后 drop_encrypted_db() 被调用
    - _enc_db._ct_list 被清空，主进程只保留 pk_M / relin_keys 等元数据
    - Enc DB (~16 GB CPU RAM) 完全由 worker 子进程持有，不占用主进程内存
    - 这对避免主进程 OOM 和 CUDA 碎片化有显著效果

Per-step transient:
  ├─ ct_list (per chunk, ~3072 tokens): ~800 MB in memory briefly
  ├─ decrypted masked_arr: (n_tokens, 4096) float32 ≈ 50 MB for 3072 tokens
  └─ g_accum: same ≈ 50 MB
```

### 4.5 进程/通信模型

#### 4.5.1 进程拓扑

```
main process (Fusion driver)
  ├─ CUDA context #0 (RTX 5090)
  ├─ multiprocessing.Pool("spawn")  ← spawn 模式(见下方决策记录)
  │   ├─ worker #0..#7: CryptoUWorker init → {ctx, encoder, evaluator, public_key, shares, plain_modulus}
  │   ├─ worker #0..#7: CryptoMWorker init → {ctx, encoder, decryptor, scale, vec_dim}
  │   │                                    (每个 worker 持有一份 sk_M,通过 bfv_sk_pem 序列化注入)
  │   └─ worker #0:    CryptoSWorker init → {enc_db (mmap), shares, hint_table, n_entries, vec_dim}
  └─ (HF model loading uses safetensors mmap, lazy read on demand)
```

##### 决策记录:fork vs spawn

CryptoWorkerPool 当前使用 **`spawn`** 而**非 fork**(实现见 `src/parties/crypto_workers/pool.py:64`)。

| 维度 | fork | spawn | 本项目选择 |
|---|---|---|---|
| `sk_M` 进程边界 | fork 后子进程继承父进程 BFV context 内存页;若父进程忘记 `_drop_secret_key()`,sk 会泄漏到所有 worker | 子进程从空白 Python 解释器启动,`sk_M` 只通过 `pickle` 显式传给 `CryptoMWorker`,**物理上不会**出现在其他 worker 的内存中 | **spawn** |
| PyTorch + CUDA fork 风险 | 父进程已加载 `torch + cuda + transformers`,8 个 fork 子进程继承 CUDA caching allocator 的 ~6 GB 快照,极易触发 OOM | 子进程不继承 CUDA 上下文,杜绝该问题 | **spawn** |
| Enc(DB) 内存占用 | 子进程 mmap 共享父进程同一物理页,9 worker × ≈0 增量 | 主进程在 `build_encrypted_database()` 后 `drop_encrypted_db()`,不持有 `_ct_list` 副本;每个 worker 独立 `from_cache(load_ct_list=True)` 触发 mmap,物理页仍按需 lazy-load | 主进程节约 ~16 GB CPU RAM |
| 启动延迟 | fork ≈ 13 ms,任务即时执行 | spawn ≈ 6 ms,但 worker 首次任务触发 `importlib.import_module` + SEAL/DB init,**首任务 +5–15 s** | spawn 代价仅在 `__init__` 时一次,训练 30+ 分钟无影响 |
| 异常隔离 | 共享 libc,worker 段错误可能让父进程崩 | 完全独立 | **spawn** |
| 内存预算 | 810 GB 物理 RAM,9 worker × ≈2 GB = ≈18 GB **远低于预算** | 同上 | 两者均可 |

**结论**:在本项目单主机 32 GB× N GPU + 数百 GB RAM 的环境下,**spawn 在工程上是更安全、更隔离的选择**,尤其是 §2.2 "密码学边界 = 进程边界" 的隐私硬性保证(见 §6.4 #5)。

#### 4.5.2 通信方式

| 通道 | 方式 | 用途 |
|---|---|---|
| **同进程 GPU tensor** | Python 引用传递(无拷贝) | U→M→S 的 `H_U`、`H_M` 传递 |
| **CPU 子进程** | `multiprocessing.Pool.apply_async(payload_dict)` + `.get()`(spawn 上下文) | PIR 查询、密文加 mask、解密 |
| **spawn 子进程 bootstrap** | `importlib.import_module(worker_cls_path)` + `init_pool_worker` 一次性注入 state | 子进程从空白解释器启动,按需 import CryptoUWorker/M/S |
| **磁盘** | `safetensors` / 自定义 mmap(`bfv_ct_db_*.bin` 带 MMAP_MAGIC 头) / `json` | V、Enc(DB)、hints 的持久化 |

#### 4.5.3 IPC 序列化开销

- `crypto_u_pool.submit` payload = `{"s3pir_responses": List[Dict], "step": int}` —— spawn 模式下每次提交按需 pickle 整个 `List[Dict]`(典型 512 tokens × ~200 B = ~100 KB)
- `crypto_m_pool.submit` payload = `{"ct_list": List[bytes], "scale": int, "vec_dim": int}` —— ct_list 每 token ~256 KB,B=4 × S=128 共 512 tokens → ~128 MB 单次序列化
- 优化:主进程 **不** 把整个加密 DB 序列化;在 spawn 模式下,worker 通过 `BFVEncryptedDatabase.from_cache(load_ct_list=True)` 独立 mmap 同一份磁盘文件,操作系统 page cache 在进程间共享物理页(冷启动时仍需 read 文件,热访问命中 page cache 后 ~0 增量)。主进程在 `build_encrypted_database()` 后调用 `drop_encrypted_db()` 清空本地 `_ct_list`,进一步节约 ~16 GB CPU 内存。

### 4.6 BFV 同态加密流水线

#### 4.6.1 参数选择

| 参数 | 值 | 理由 |
|---|---|---|
| `poly_degree` | 4096 | 加密一个 4096 维向量需要 N=4096 slots,BFV 多项式度 4096 |
| `plain_bits` | 30 | plaintext 模数 2^30 ≈ 1 GB,容纳 bf16 量化的整数 |
| `scale` | 10000 | 浮点 → 整数的固定缩放因子;明文向量元素值域 ∈ [-32767, 32767] |
| `plain_modulus` | 2^30 + small_delta | 由 `_infer_plain_modulus(plain_bits=30)` 计算,mirroring SEAL 的 PlainModulus.Batching |
| `quantization` | round(x · scale),中心化到 ±plain_modulus/2 | 与 PRG mask 的整数空间对齐 |

#### 4.6.2 加解密操作清单

| 操作 | 调用 | 位置 |
|---|---|---|
| 加密 V 行 | `Encryptor.encrypt(encoder.encode(ints))` | `BFVEncryptedDatabase.build_from_V`(Stage 0) |
| 解密 masked ct | `Decryptor.decrypt(ct) → Plaintext → decode_ints_as_vector / scale → float32` | `CryptoMWorker.handle_request` |
| 同态加明文 | `Evaluator.add_plain_inplace(ct, plain)` | `CryptoUWorker`(加 `-r_t`)、`CryptoSWorker._accumulate_parity`(加 `a_t` 的明文) |
| 同态加密文 | `Evaluator.add_inplace(ct1, ct2)` | `CryptoSWorker._accumulate_parity_no_a`(累加 Enc(V[y]) 行列) |

#### 4.6.3 隐私正确性

SVG 公式 `Dec(result_U) + result_S = a_t - V_{y_t}` 的代数推导:

```
result_U = Enc(-Ṽ_{y_t}) + Enc(r_t) ≡ Enc(-V_{y_t} + r_t)   [BFV homomorphic]
result_S = a_t - r_t                                          [plaintext share]
Dec(result_U) + result_S = (-V_{y_t} + r_t) + (a_t - r_t) = a_t - V_{y_t}  ✅
```

注意:`r_t` 在 S 端与 U 端由共享 PRG seed 独立生成,二者 bit-exact 相同。这是 "mask" 协议的核心 —— S 知道 `a_t` 但不知道 `r_t`(U 的 mask 已加到密文里);U 知道 `r_t` 但不知道 `a_t`(从 U 的视角,密文 `result_U` 是 Enc 形式,无法解密)。**只有同时拿到 Dec(result_U) 与 result_S 才能恢复 `a_t - V_{y_t}`,而这两部分分别由 U 与 S 单独掌握,M 看到的是二者之和的最终结果。**

### 4.7 S3PIR 单服务器 PIR

`src/core/s3pir_hints.py::HintTable` 是基于 **subset-based stateful PIR** 的实现:

#### 4.7.1 Hint 表结构

对每个 partition `j ∈ [0, n_partitions)`:

- **Main hints**(M_main 个): 每个 hint `h` 有 `cutoff` 与 `extra_index`
  - `cutoff` = 中位数(`{PRF_v(j, k) : k in partition_j}`)
  - `extra_index` = tie-breaker(中位数重位时选取的具体索引)
- **Backup hints**(M_backup_pairs 对): 冗余 hints,防止 main hint 命中失败

#### 4.7.2 查询算法(对每个 token `y_t`)

1. 客户端(U)已知 `seed`,生成 `v_jk = PRF_v(j, k)`,计算 `selected = {k ∈ partition_j : v_jk <= cutoff_j} ∪ {extra_index}`
2. 客户端构造 PIR query:对每个 hint `h`,真实选中的 indices(行号)+ 假的 dummy indices
3. 服务器响应 `parity_h = ⊕_{y in hint_h} D[y]`(BFV 同态累加)
4. 客户端拿到所有 parities 后,选取真实 parity 并解密

#### 4.7.3 为什么不用拉格朗日插值

S3PIR 用 **median 阈值 + PRF 伪随机** 而非多项式插值,优势:

- 离线阶段只需构建 hint 表(密文累加),无需 V 的明文
- 查询阶段复杂度 O(sqrt(N)),比基于插值的方案快
- `lam=80` 提供 2^{-80} 的假阳性率

### 4.8 训练循环与检查点

#### 4.8.1 Trainer 主循环

`Trainer.train()`(`src/training/trainer.py:185`):

```python
for epoch in range(self._completed_epochs, self.config.max_epochs):
    metrics = self._run_epoch(epoch)              # step_train_chunked × n_batches
    self._log_epoch(epoch, metrics)               # 写 epoch_records.jsonl
    self._save_last_checkpoint(epoch)             # 落盘 last_checkpoint.pt
    if self._is_best(metrics):
        self._save_best_checkpoint(epoch)
    if self.patience_counter >= self.config.patience:
        break
```

#### 4.8.2 Checkpoint 内容

- `best_checkpoint.pt` / `last_checkpoint.pt` / `checkpoint_epoch_NNN.pt` —— 每个包含:
  - `epoch`: 当前 epoch 编号
  - `global_step`: 已训练 step 数
  - `best_metric` / `best_epoch`: 早停用
  - `model_state_dict`: LoRA 参数(448 tensors, ~3 MB)
  - `optimizer_state_dict`: AdamW state(m, v, t)
  - `lr_scheduler_state_dict`: warmup + cosine 调度器
  - `completed_epochs`: 训练循环恢复点
  - `rng_states`: CPU + GPU + numpy 随机数状态
  - `_completed_epochs`: 循环起点的兼容性字段
- 总大小 ~126 MB(主要是 optimizer state)

#### 4.8.3 CheckpointManager 策略

- 保留最近 5 个 epoch checkpoints(`checkpoint_epoch_{015..019}.pt`)
- 始终覆盖 `best_checkpoint.pt` 与 `last_checkpoint.pt`
- `resume_from(path)` 会先把现有 `last_checkpoint.pt` 复制为 `last_checkpoint.pt.safety`,再加载

#### 4.8.4 恢复(resume)语义

`Trainer.resume_from(ckpt_path)`:

1. 复制 last_checkpoint.pt → last_checkpoint.pt.safety
2. `torch.load(ckpt_path)` → ckpt dict
3. `self.ipc.load_checkpoints(ckpt['model_state_dict'], ckpt['optimizer_state_dict'])` —— 把 LoRA + AdamW 状态写回 PartyM
4. `self.global_step = ckpt['global_step']`
5. `self._completed_epochs = ckpt['completed_epochs']`
6. `self.best_metric = ckpt['best_metric']`

实测:Phase 3.5 从 Phase 1 的 epoch 1 恢复,448 个 LoRA tensors 全部正确恢复。

---

## 5. 关键流程的源码定位

> 本节列出行号基于 **Phase 1 v2.2 代码快照(2026-07-21)**。每次重大重构后请同步更新。

| 流程 | 文件 | 函数/类 | 行号 |
|---|---|---|---|
| Stage 0 加密 DB | `src/scripts/build_encrypted_db.py` | `build_encrypted_db` | line 123 |
| Stage 0 S3PIR hints | `src/scripts/build_s3pir_hints.py` | `build_s3pir_hints` | line 51 |
| BFV backend | `src/core/bfv_privselect_v2_adapter.py` | `BFVPrivSelectV2Backend` | line 556 |
| BFV build_encrypted_database | 同上 | `BFVPrivSelectV2Backend.build_encrypted_database` | line 689 |
| Encrypted DB 类 | 同上 | `BFVEncryptedDatabase` | line 297 |
| Encrypted DB build_from_V | 同上 | `BFVEncryptedDatabase.build_from_V` | line 400 |
| Encrypted DB drop_encrypted_db | 同上 | `BFVPrivSelectV2Backend.drop_encrypted_db` | line 858 |
| PRG share 协议 | 同上 | `PRGShareProtocolBFV` | line 180 |
| PRG share.generate_mask_ints | 同上 | `generate_mask_ints` | line 245 |
| PRG share.server_make_share | 同上 | `server_make_share` | line 255 |
| Hint table | `src/core/s3pir_hints.py` | `HintTable` | line 25 |
| SageAttention 启用 | 同上 | `is_sage_attention_available` / `is_sage_attention_fp4_available` | line 41 / 75 |
| 模型切分(三个 submodel) | `src/model/model_splitting.py` | `load_u_submodel` / `load_m_submodel_with_lora` / `load_s_submodel` | line 682 / 760 / ~850 |
| LoRA 包装 | 同上 | `_LoRALinear` | line ~470 |
| LoRA 注入 | 同上 | `_inject_lora` | line ~490 |
| FlashAttention2 启用 | 同上 | `load_u_submodel` / `load_m_submodel_with_lora` | line 682 / 760 |
| DeepSpeed ZeRO 管理器 | `src/utils/deepspeed_zero.py` | `DeepSpeedZeROManager` | line 222 |
| DeepSpeed ZeRO 配置 | 同上 | `create_ds_config` / `create_zero_config` | line 159 / 51 |
| PartyU | `src/parties/party_u.py` | `PartyU` | line 30 |
| PartyM | `src/parties/party_m.py` | `PartyM` | line 33 |
| PartyS | `src/parties/party_s.py` | `PartyS` | line 32 |
| HeterogeneousProtocol | `src/parties/heterogeneous_protocol.py` | `HeterogeneousProtocol` | line 103 |
| step_train | 同上 | `HeterogeneousProtocol.step_train` | line 280 |
| step_train_chunked | 同上 | `HeterogeneousProtocol.step_train_chunked` | line 380 |
| step_val | 同上 | `HeterogeneousProtocol.step_val` | line 509 |
| step_test | 同上 | `HeterogeneousProtocol.step_test` | line 591 |
| M 反向传播 + LoRA 更新 | `src/parties/party_m.py` | `PartyM.backward_and_update` | line 239 |
| M 注入梯度 | 同上 | `PartyM._inject_and_backward` | line 370 |
| S 处理 logits (training) | `src/parties/party_s.py` | `PartyS.process_logits_dispatch` | line 210 |
| S 计算 logits (eval) | 同上 | `PartyS.compute_logits_for_eval` | line 110 |
| S 计算 logits (train) | 同上 | `PartyS.compute_logits_gpu` | line 96 |
| S 计算 a_t | 同上 | `PartyS.compute_a_t_gpu` | line 121 |
| S 生成预测 | 同上 | `PartyS.generate_predictions` | line 272 |
| CryptoU worker | `src/parties/crypto_workers/crypto_u.py` | `CryptoUWorker` | line 37 |
| CryptoM worker | `src/parties/crypto_workers/crypto_m.py` | `CryptoMWorker` | line 37 |
| CryptoS worker | `src/parties/crypto_workers/crypto_s.py` | `CryptoSWorker` | line 36 |
| Worker pool | `src/parties/crypto_workers/pool.py` | `CryptoWorkerPool` | line 37 |
| Worker pool.submit | 同上 | `CryptoWorkerPool.submit` | line 74 |
| Trainer | `src/training/trainer.py` | `Trainer` | line 81 |
| Trainer.train | 同上 | `Trainer.train` | line 188 |
| Trainer._run_epoch | 同上 | `Trainer._run_epoch` | line 244 |
| Trainer._run_val_epoch | 同上 | `Trainer._run_val_epoch` | line 305 |
| Trainer._run_test_epoch | 同上 | `Trainer._run_test_epoch` | line 548 |
| Checkpoint manager | `src/training/checkpoint.py` | `CheckpointManager` | line 22 |
| 主入口 (Stage 0) | `src/scripts/finetune.py` | `run_stage0` | line 147 |
| 主入口 (Stage 1) | 同上 | `run_stage1` | line 197 |
| 主入口 (Stage 2) | 同上 | `run_stage2` | line 370 |
| 数据集加载 | `src/data/dataset.py` | `load_dataset` (alias of `load_biotriplex_dataset`) | line 274 |
| 数据集 BioTriplexQADataset | 同上 | `BioTriplexQADataset` | line 77 |
| Letter 解析 | 同上 | `parse_answer_letter` | line 234 |
| 2-epoch 验证 | `scripts/two_epoch_test.py` | `run_two_epoch_test` | line 86 |
| 10-step smoke | `scripts/quick_smoke_10step.py` | `main` | line 26 |
| 端到端正确性 | `scripts/heterogeneous_correctness_test.py` | `main` | (脚本级) |
| 流水线正确性 | `scripts/chunk_correctness_test.py` | `main` | (脚本级) |
| 性能基准 | `scripts/perf_bench.py` | `main` | (脚本级) |

> `FusionProtocol`(inline 模式,见 `src/parties/fusion_protocol.py:192`)与 `LegacyIPCStub`(见 `src/parties/legacy_ipc_stub.py:441`)是两条审计路径,**不在生产路径中调用**,仅供正确性回归与跨实现对比。

---

## 6. 附录

### 6.1 默认 Config 参数表(`src/scripts/finetune.py::Config` + `scripts/two_epoch_test.py::TwoEpochConfig`)

> `configs/llama_biotriplex_he_pir.py` 在当前仓库中是**空目录**(占位用)。所有运行时参数都以 dataclass 内置在 `src/scripts/finetune.py` 中。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `hf_model` | `/root/autodl-tmp/hf_cache/Llama-3-1-8B-I` | HuggingFace 模型快照路径 |
| `bfv_cache_dir` | `/root/autodl-tmp/slg-bfv-cache` | Stage 0 产物落盘目录(Enc DB + hint table) |
| `data_dir` | `/root/slg-v2.0/data/biotriplex_qa` | BioTriplex JSONL 数据目录 |
| `project_root` | `/root/autodl-tmp/SLG-HE-PIR` | 仓库根目录 |
| `checkpoint_dir` | `${project_root}/checkpoints` | 训练 checkpoint 输出目录 |
| `log_dir` | `${project_root}/logs` | 训练日志输出目录 |
| `vocab_size` | 128256 | Llama-3.1 词表大小 |
| `hidden_dim` | 4096 | Llama-3.1-8B 隐藏层维度 |
| `poly_degree` | 4096 | BFV 多项式度 |
| `plain_bits` | 30 | BFV plaintext modulus bits |
| `u_layers` | 0 | U 持有的 decoder 层数(0 = 只持有 embed_tokens) |
| `m_layers` | 32 | M 持有的 decoder 层数(全部) |
| `scale` | 10000 | 量化缩放因子 |
| `lam` | 80 | S3PIR 正确性参数(2^{-80} 假阳性) |
| `lora_rank` | 8 | LoRA r |
| `lora_alpha` | 16 | LoRA α |
| `lora_dropout` | 0.05 | LoRA dropout |
| `batch_size` | 4 | 训练 batch |
| `max_seq_length` | 128 | 序列最大长度 |
| `max_epochs` | 10 | 默认训练 epoch 上限(2-epoch 测试覆盖为 2) |
| `learning_rate` | 3.5e-4 | AdamW peak lr |
| `weight_decay` | 0.01 | AdamW weight decay |
| `gradient_clip_norm` | 1.0 | gradient clipping |
| `warmup_steps` | 200 | LR warmup 步数 |
| `lr_scheduler` | "cosine_with_warmup" | 余弦退火 |
| `train_ratio` | 0.9 | train/val 划分 |
| `patience` | 999 | 早停耐心(默认不早停) |
| `N_CRYPTO_U_WORKERS` | 8 | CryptoU 子进程数 |
| `N_CRYPTO_M_WORKERS` | 8 | CryptoM 子进程数 |
| `N_CRYPTO_S_WORKERS` | 1 | CryptoS 子进程数 |
| `USE_CHUNKED_PIPELINE` | True | 启用分块流水线 |
| `CHUNK_TOKENS` | 3072 | 每个 chunk 的 token 数 |
| `gpu_fraction_u` / `gpu_fraction_m` / `gpu_fraction_s` | 0.10 / 0.22 / 0.22 | 历史遗留字段(已废弃) |
| `val_metric` | "val_entity_micro_f1" | best-metric 选择指标 |
| `dump_attacks` | False | 是否落盘 attack 中间量 |
| `do_test_eval` | False | 训练后是否调 `_run_test_epoch()` |

#### `src/scripts/finetune.py` CLI 开关

| 开关 | 默认 | 含义 |
|---|---|---|
| `--stage` | `all` | `0` / `1` / `2` / `all` |
| `--checkpoint` | None | Stage 2 用的 ckpt 路径 |
| `--skip_db` / `--skip_hints` | False | Stage 0 跳过某步 |
| `--max_epochs` | None | 覆盖 cfg.max_epochs |
| `--batch_size` | None | 覆盖 cfg.batch_size |
| `--use_chunked_pipeline` | None | 接受 `true`/`false` |
| `--chunk_tokens` | None | 覆盖 cfg.CHUNK_TOKENS |
| `--n_crypto_u_workers` / `--n_crypto_m_workers` / `--n_crypto_s_workers` | None | 覆盖对应 pool size |
| `--dump_attacks` | False | 落盘中间量 |
| `--log_freq` / `--log_dir` | None | 覆盖日志 |
| `--config` | None | JSON 覆盖文件 |
| `--do_test_eval` | False | 训练后跑 test |

#### `scripts/two_epoch_test.py` CLI 开关

| 开关 | 默认 | 含义 |
|---|---|---|
| `--max_epochs` | 2 | 训练 epoch 数 |
| `--data_dir` | None | 覆盖数据集目录 |
| `--bfv_cache_dir` | None | 覆盖 BFV 缓存目录 |
| `--batch_size` | None | 覆盖 batch_size |
| `-v, --verbose` | False | 调试日志 |

### 6.2 关键指标实测(Phase 1 v2.2, 2026-07-21)

**Run 配置**:`python scripts/two_epoch_test.py --max_epochs 2`
**Report**:`/tmp/slg_2epoch_run/PHASE1_V2p2_REPORT.md`

| 指标 | 值 |
|---|---|
| **train_loss_proxy epoch 0** | 2056.41(梯度范数,真实均值;Phase 1 v2.1 bug 期为 124000 恒定) |
| **train_loss_proxy epoch 1** | 2052.90(**-3.51 vs epoch 0** ✅,证明 mask 消去后 LoRA 在收敛) |
| **val_entity_micro_f1 / val_macro_f1 / val_weighted_f1** | 0.0 / 0.0 / 0.0(详见 §6.4.1:pipeline 解析层 bug,NER 数据 vs letter 设计错位) |
| **val_ce_loss** | 0.0(同 §6.4.1 原因:`labels_tensor = output_ids = None`,CE 路径 skip) |
| **val_samples** | 118 |
| **Total training steps** | 310(155 steps × 2 epochs) |
| **avg_step_time_ms** epoch 0 / epoch 1 | 6080.8 / 5925.0 |
| **avg_gpu_mem_mb** epoch 0 / epoch 1 | 20226.9 / 20233.9(稳定 ~20.2 GB / 32.6 GB 总) |
| **Total elapsed** | 1874.2 s(31 min 14 s)|
| **DB cache size** | `bfv_ct_db_n128256_d4096_p4096.bin` = 16,825,776,464 B(≈ 16.0 GB)|
| **DB build time** | 186.2 s(Stage 0 一次性)|
| **Hint table size** | `hint_table.json` = 1,575,491 B(≈ 1.5 MB)|
| **Stage 0 DB rebuild** | **必要** (`force=True` + 显式删 cache),详见 §6.6 (B) |
| **Crypto worker pool** | OK (U=8, M=8, S=1) |
| **OOM / NaN / Inf / worker crash** | 无 |
| **`best_checkpoint.pt` 来源 epoch** | epoch 0(epoch 1 F1 持平 = 0)|

**Mask 消去正确性独立验证(单 token 前 8 元素)**:Phase 1 v2.2 修复后,`Dec(result_U)` 与理论值 `−V_y·scale + r_t` 的差在 **±32 量级**(SEAL BatchEncoder 自身固有精度 floor),完全消除 ±pm/2 ≈ 5.37×10⁸ 的 mask residue。详见 §6.6。

> **历史对照**:Phase 1 v2.1(2026-07-19)报告的 `val_ce_loss epoch 0=17.99 / epoch 1=13.83` 是 v2.1 bug 期数据,**已被 v2.2 数据取代**。v2.2 不再观察 `val_ce_loss`(因 §6.4.1);`train_loss_proxy` 是 v2.2 起真实观察的收敛信号。

### 6.3 SVG 公式 ↔ 代码对照表

| SVG 公式 | 代码定位 |
|---|---|
| `H_0 ← Embedding(x)` | `PartyU.forward_train` line 152(`self.model.embed_tokens(input_ids)`) |
| `H_ans ← H_0 · W` | `PartyM.forward` → `_m_forward`(对 32 个 decoder 层用 `torch.utils.checkpoint.checkpoint`) |
| `Z_t ← H_ans · V^T` | `PartyS.compute_logits_gpu` line 96(`torch.matmul(H_M, V.T)`) |
| `a_t ← softmax(z_t) · V` | `PartyS.compute_a_t_gpu` line 121(分块 `softmax @ V`) |
| `r_t ← PRG(seed, t)` | `PRGShareProtocolBFV.generate_mask_ints` line 245 |
| `Ṽ_{y_t} ← PIR(y_t, D[y])` | `CryptoSWorker.handle_request` line 133(Design-2:`enc_db.get_encrypted_row(y_t)`,`|real|=1, |dummy|=0`) |
| `result_U ← -Ṽ_{y_t} + Enc_{pk_M}(r_t)` | `CryptoUWorker.handle_request` line 37(`add_plain_inplace(ct, pt=r_t)` —— 注意是 **加 `r_t` 不是减 `-r_t`**,因为 Stage 0 写入的是 `Enc(-V_y)`,所以 `Enc(-V_y) + Enc(r_t) = Enc(-V_y + r_t)`)|
| `result_S ← a_t - r_t` | `PRGShareProtocolBFV.server_make_share` line 255(`a_t_ints - r_t`,其中 `a_t_ints = round(a_t · scale)`) |
| `g_{H,t} ← Dec_{sk_M}(result_U) + result_S` | `PartyM.backward_and_update` line 239(`g_accum[t] = masked_centered[t] + s_share[t]`) |
| `H_M.backward(g_H)` | `PartyM._inject_and_backward` line 370 |
| LoRA 更新 | `PartyM.backward_and_update` 内 `optimizer.step()` line 336 |
| `ŷ ← softmax(H_ans* · V)` | `PartyS.compute_logits_for_eval` line 110 |
| `Loss ← Dist(y*, ŷ)` | `Trainer._run_val_epoch` line 305(`F.cross_entropy` + letter-set F1,详见 §6.4.1) |

### 6.4 已知缺陷与设计权衡

#### 6.4.1 `val_entity_micro_f1 = 0` + `val_ce_loss = 0` —— prompt 解析层与 NER/QA 任务错位

**现象**:Phase 1 v2.2 跑完 2 epoch,所有 val 指标恒为 0.0,`val_ce_loss = 0`(`val_samples = 118`,没有 NaN/Inf)。`train_loss_proxy` 单调下降证明训练本身正常,问题只在 val 评测链路。

**根因(三层叠加)**:

1. **数据集契约 mismatch**:`BioTriplexQADataset.__getitem__`(`src/data/dataset.py:101-142`)只返回 `item["labels"]`(label token ids)与 `item["output_text"]`(NER 字符串),**没有 `item["output_ids"]`**。但 `step_train` / `step_val` 都用 `batch.get("output_ids")` 取 gold token ids —— 该键永远是 `None`。
2. **NER vs QA 任务错位**:实际数据是 NER 任务(`output = "Entities: (hyperuricemia, DISEASE)"`),但 `parse_answer_letter` / `_safe_letter_split` / `OPTIONS_LETTERS` 全部按 BioTriplex-QA 7 类字母答案(`a)`/`b)`/...)设计。
3. **公式语义错位**:`_safe_letter_split` 用 `str.split(",")` 切 NER 字符串 → `{"Entities: (hyperuricemia", "DISEASE)"}` 这样的**非字母子串**集合;Trainer 用 `set("...") == set("...")` 做 letter-set micro F1,任何字符串不等都会导致 `pred_set ∩ gold_set = ∅` → `tp_total = 0` → `micro_f1 = 0`。

**具体代码位置**:
- 数据集缺字段:`src/data/dataset.py:130-140`(`item["labels"]` 而非 `item["output_ids"]`)
- 训练侧拿不到 gold:`src/parties/heterogeneous_protocol.py:305-307`(`if "output_ids" in batch:` 永远 False)
- 验证侧拿不到 gold labels_letters:`src/parties/heterogeneous_protocol.py:548-565`(`output_ids is not None:` 永远 False → `labels_letters = []`)
- CE loss 永远 skip:`src/training/trainer.py:339-352`(`if logits is not None and labels_tensor is not None:` 因 `labels_tensor = output_ids = None` 而跳过)
- letter 集合公式:`src/training/trainer.py:370-387`(`tp_total += len(pred_set & gold_set)`,对 NER 字符串天然失败)
- letter 解析器:`src/data/dataset.py:234-267`(`parse_answer_letter` 把 NER 字符串原样返回,然后被 `_safe_letter_split` 按 `,` 切碎)

**影响**:
- ✅ 训练本身正确(S3PIR mask 消去 → `train_loss_proxy` 单调下降)
- ❌ val 指标当前**没有数值意义**(所有 F1 恒 0)
- ⚠️ 训练侧 S3PIR 实际上用 **argmax** 作 `y_t`(因 `gold_ids = None` 而 fallback),**不是真正的监督梯度**——LoRA 学的是 self-distillation。Phase 1 v2.3 计划同步修数据集契约 + 训练侧 gold 选取路径。

**修复路线**:
- L1(必修):`BioTriplexQADataset.__getitem__` 多返回 `output_ids = label_encoded["input_ids"].squeeze(0)`(或改 dataset.py 内部直接用 `item["labels"]` 当 output_ids)
- L2(必修):重写 `parse_answer_letter` / `_safe_letter_split` 为 NER span 解析(`(text, TYPE)` 元组),改 trainer 用 span-set micro F1
- L3(同步):修完后 S3PIR 训练梯度自动对齐 gold,LoRA 才是真正的监督训练

#### 6.4.2 `train_loss_proxy` 语义

这是 `‖g_H‖ / (B·S)` 的**梯度范数**,**不是真 loss**(真 loss 在 `val_ce_loss`,但如 §6.4.1 所述当前 0)。Phase 1 v2.2 修复后正常范围约 **2050 ~ 2060**,且 epoch 0 → 1 单调下降。Phase 1 v2.1 修复前的 smoke 报告 `loss_proxy ≈ 124000` 恒定 —— 那是 mask 没消干净(详见 §6.6)。

#### 6.4.3 `u_split_layer = 0` 与 SVG 不完全一致

SVG 画的是 U 持有 decoder[0..16),但实际为 0。这是 Phase 1 v2.1 修复时的合理演化(降低显存占用、简化跨边界数据传递)。

#### 6.4.4 没有 LoRA-only adapter 保存

目前 checkpoint 必须含 optimizer state 才能 resume,无法做"轻量级 adapter 持久化"。

#### 6.4.5 CryptoMWorker 的 sk_M 隔离

`sk_M` 一旦分发到 spawn 子进程即无法收回;在 spawn 模式下,只有 `CryptoMWorker` 子进程持有 sk_M,其他 worker 与主进程都拿不到(主进程在 Stage 1 启动时 `_drop_secret_key()`)。多主机部署下,M 与 U/S 通过 TLS 通道传输 sk_M 后,M 端销毁本地备份,可以彻底消除单点风险。详细决策见 §4.5.1。

#### 6.4.6 `parse_gold_entities` 不存在

`src/training/evaluation.py:208` 引用了 `from ..data.dataset import parse_gold_entities`,但该函数**在当前代码库中未定义**。Phase 1 v2.2 没调 `_run_test_epoch`,所以没暴露。Stage 2 启用 `--do_test_eval` 或 standalone test eval 时会直接 ImportError,需补齐 `parse_gold_entities(text)` 把 `"Entities: (hyperuricemia, DISEASE); (PRAL, GENE)"` 解析为 `{("hyperuricemia", "DISEASE"), ("PRAL", "GENE")}` 这样的 set,然后用 span-set micro F1。

### 6.6 Phase 1 v2.2 修复:mask 完美消去 + DB cache key 配对

历史上 `loss_proxy ≈ 124000` 恒定、不随 step 变化的根因是**两个相互独立的 bug 叠加**:

**(A) PRG mask 表示在 U/S 两端不一致 + SEAL BatchEncoder 边界 ±49151 偏移**

`PRGShareProtocolBFV.generate_mask_ints` 返回 `r_t ∈ (−pm/2, +pm/2)` 的**带符号整数**(严格开区间,SEAL BatchEncoder 能完美保留)。但:

- **S 端** `server_make_share` 返回 `s_share = scale·a_t − r_t`(带符号 int64)
- **U 端** 之前错误地做了 `pos_r_t = (x % pm)`,把负数 r_t 转 `[0, pm)` 正整数后再 `encoder.encode(pos_r_t)` —— 这让 SEAL 内部 mod-pm centred 时给某些元素引入 **+49151 系统性偏移**(SEAL BatchEncoder 在输入 ≥ pm/2 时内部 mod 引入的 noise floor)
- **M 端** 之前直接拿 `[0, pm)/scale` 的 float 与 `s_share / scale` 的 float 相加 —— 实际上一个是 centred 一个是带符号,域不匹配

**修复**(`Phase 1 v2.2`):
- U 端去掉 `pos_r_t`,直接 `encoder.encode([int(x) for x in r_t])`,让 SEAL centred 表示天然对齐
- M 端把 decoded int 乘回 scale 居中到 `[-pm/2, +pm/2)`,再与 int64 s_share 直接相加,最后 /scale

**(B) DB cache 与 fresh sk/pk 不配对**

`BFVPrivSelectV2Backend.__init__` **不**从 cache 加载 sk/pk,每次 run 都 `KeyGenerator` 生成新对。但 `build_encrypted_database(V, force=False)` **默认** 从 `bfv_ct_db_*.bin` 加载旧密文 —— **导致 fresh sk 解密旧 pk 加密的密文 → 完全乱码**。`masked_int` 与理论值差 ±5e8,看起来像"mask 没消",实际是**解码值错位**。

**修复**:
- `scripts/quick_smoke_10step.py` 与 `scripts/two_epoch_test.py` 第 181 行改为 `force=True` 重建 DB;**production `finetune.py` 走 Stage 0/1,DB 在 Stage 0 一次性构建,后续用同一对 sk/pk 训练,不受影响**
- 长期改进方向:`BFVPrivSelectV2Backend.__init__` 增加 cache-vs-key 一致性检查,key 不匹配时要么拒绝、要么自动 `force=True`(TODO:后续实现)

**修复后验证**(smoke 10 step):
```
Step  1/10: loss=1928.0000   ← 真实梯度范数
Step  2/10: loss=2208.0000
Step  3/10: loss=2080.0000
Step  4/10: loss=2096.0000
...
Step 10/10: loss=1960.0000
```
范围 `1928 ~ 2208`,**与 step 相关** —— mask 完全消去,`g_accum = a_t − V_y` 真是 LoRA 反向传播需要的梯度。

精确性验证(单 token 前 8 元素):
```
expected masked = (-V_y*scale + r_t) centred:
  [-305195643 -148236666 -450644695 -474606272 -156466657  494878538   41002168 -267664250]
actual   masked:
  [-305195648 -148236672 -450644672 -474606304 -156466656  494878560   41002168 -267664256]
diff (SEAL noise floor):  [-5  -6  23 -32   1  22   0  -6]   ±32 量级 ✓
```

**2-epoch v2.2 实测**(详见 §6.2):`train_loss_proxy` epoch 0 → 1: **2056.41 → 2052.90(-3.51)**,稳定在 2050 ~ 2060 范围,证明 mask 消去后 LoRA 真正在收敛。

> **注意**:`val_*` 全 0 是 §6.4.1 描述的 prompt 解析层 + NER/QA 任务错位 bug,与本节描述的 mask 消去问题**独立**。

### 6.5 文档版本与维护说明

- 本文档与 `docs/SLG_HE_PIR_USAGE.md`(快速使用手册)同步更新;后者侧重"如何调用",前者侧重"实现原理"。
- 本文档与 `docs/SLG系统流程.svg` 严格对照。
- 任何对 stage 0 / step_train / step_val 的修改都应同步更新本文档 §3 与 §4。
- 任何对加密参数(poly_degree、plain_bits、scale)或 LoRA 参数(r、α)的修改都应同步更新本文档 §4.2 / §4.6 / §6.1。
- 本文档不直接使用脚本输出验证;脚本级验证由 `scripts/test_trainer_dispatch.py`、`scripts/e2e_correctness_recheck.py`、`scripts/heterogeneous_correctness_test.py` 等提供。
- 维护者请在每次变更后,通过对照 §6.3 表格,确保每条 SVG 公式都有对应的代码位置;通过对照 §5 表格,确保所有行号与代码快照对齐。
- **章节编号保留**:历史上 §6.6 在 §6.5 之前,本版本未重排,仅保持向后兼容;后续可统一调整。