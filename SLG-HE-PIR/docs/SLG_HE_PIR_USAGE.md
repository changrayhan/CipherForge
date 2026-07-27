# SLG-HE-PIR 使用文档

> **版本**: 2026-07-21
> **面向**: 新会话中的助手 / 用户 / 协作者
> **目标**: 在 5 分钟内理解 SLG-HE-PIR 是什么、怎么跑、如何排错
> **配套文档**: [`SLG_HE_PIR_SYSTEM_DOCUMENTATION.md`](./SLG_HE_PIR_SYSTEM_DOCUMENTATION.md) — 实现原理的"长篇"对照文档(密码学细节、数据流、源码定位)

---

## 0. 30 秒 TL;DR

| 维度 | 值 |
|---|---|
| **是什么** | 在单台 GPU 主机上,模拟三方协作(用户 U / 模型 M / 服务器 S),对 Llama-3.1-8B-Instruct 做 **隐私保护 LoRA 微调** |
| **保护什么** | 用户私有 `(x, y)` 不暴露给 M;lm_head 矩阵 `V` 不暴露给 U |
| **密码学原语** | BFV 同态加密(`poly_degree=4096, plain_bits=30, scale=10000`)+ S3PIR 单服务器 PIR(`partition_size=256, lam=80`)+ PRG 共享掩码 |
| **当前状态** | Phase 1 v2.2 — 2-epoch 训练可跑通,`train_loss_proxy` 单调下降(-3.51),但 `val_*` 全 0(详见 §6.4.1) |
| **主入口** | `python src/scripts/finetune.py --stage {0|1|2|all}` |
| **快速测试** | `python scripts/two_epoch_test.py --max_epochs 2` |
| **硬件** | 1 × RTX 5090(32 GB)+ ≥ 200 GB RAM |

---

## 1. 项目是什么 & 不是什么

### 1.1 它解决的问题

第三方机构(医院 / 金融机构)持有私有医学语料 `(x, y)`,**不想**把数据交给大模型拥有者(M);大模型拥有者也**不想**把 `lm_head`(`V`)给医院。SLG-HE-PIR 在这种双方不互信的场景下,通过密码学手段让 LoRA 微调"秘密地"发生 —— 训练结束后双方都拿到了自己想要的(医院拿到适配过的 LoRA,M 拿到适配过的 LoRA),**任何一方都没有看到对方的明文数据**。

### 1.2 它不解决的问题

- ❌ **不是生产系统**:单机模拟三方,密码学边界通过 `multiprocessing.Pool(spawn)` 子进程隔离,**不适用于真正多主机场景**(但 `IPCProtocol` / `LegacyIPCStub` 接口已预留)。
- ❌ **不是 LoRA 训练最优解**:只是证明隐私路径下 LoRA 能收敛。`val_*` 评测指标当前是 0(详见 §6.4.1)。
- ❌ **不是新算法**:BFV + S3PIR 都是已知原语,本项目把它们工程化。

### 1.3 关键术语对照表

| 术语 | 含义 | 在代码里 |
|---|---|---|
| U / User | 持有私有 `(x, y)`、embed_tokens | `PartyU`、`src/parties/party_u.py` |
| M / Model | 持有 decoder 权重 + LoRA、需要解密 + 反向传播 | `PartyM`、`src/parties/party_m.py` |
| S / Server | 持有 `V`(lm_head)+ Enc(DB)+ hint table | `PartyS`、`src/parties/party_s.py` |
| V / lm_head | shape `(128256, 4096)` 的 LLM 输出投影 | HuggingFace `lm_head.weight` |
| D[y] / Enc(-V[y]) | S 端的密文 V 矩阵第 y 行 | `BFVEncryptedDatabase` |
| r_t | U 与 S 共享 PRG 种子生成的伪随机掩码向量 | `PRGShareProtocolBFV.generate_mask_ints` |
| a_t | `softmax(z_t) @ V`,S 端拿到的明文 share 前置量 | `PartyS.compute_a_t_gpu` |
| Ẽ[V_y] / Enc(-V[y]) | PIR 查询的密文响应 | `CryptoSWorker.handle_request` |
| g_H / train_loss_proxy | `‖g_H‖ / (B·S)` 梯度范数(**不是真 loss**) | `PartyM.backward_and_update` |

---

## 2. 环境准备

### 2.1 硬件 & 系统

- **GPU**: ≥ 1 张 RTX 5090(32 GB, compute capability 12.0);实测显存占用 ~17 GB,留 15 GB 余量
- **RAM**: ≥ 200 GB(CryptoWorker spawn 子进程 + BFV Enc DB ~16 GB CPU RAM + mmap)
- **磁盘**: ≥ 60 GB(`/root/autodl-tmp/slg-bfv-cache/` 占用 ~16 GB Enc DB + checkpoint ~128 MB × N)
- **OS**: Linux(本项目在 `linux 5.15.0-94-generic` 上验证)
- **Python**: ≥ 3.10(测试用 `python 3.10+`)

### 2.2 软件依赖

```bash
# 必需
pip install torch transformers safetensors pyyaml

# 必需(BFV 后端)
pip install tenseal          # TenSEAL 包装 SEAL C++;版本需支持 poly_degree=4096

# 必需(其它)
pip install numpy           # ≥ 1.20
```

### 2.3 模型快照

Llama-3.1-8B-Instruct 的 HuggingFace 快照必须预下载到本地:

```bash
# 推荐路径(与 Config 默认对齐)
/root/autodl-tmp/hf_cache/Llama-3-1-8B-I/

# 必须包含
# - config.json
# - model.safetensors.index.json
# - model-00001-of-00002.safetensors
# - model-00002-of-00002.safetensors  (含 lm_head)
# - tokenizer.json
# - tokenizer_config.json
```

> **强烈建议**:在断网/离线环境运行前设 `export HF_HUB_OFFLINE=1`,避免 transformers 自动重连 hub。

### 2.4 数据集

```bash
# 默认路径(可改)
/root/slg-v2.0/data/biotriplex_qa/
├── train.jsonl          # 738 条原始,9:1 划分后 train=620
├── val.jsonl            # 同上,val=118
└── test.jsonl           # 160 条
```

每条 JSONL 行形如:

```json
{"id": "...", "input": "...text...", "question": "...NER prompt...", "output": "Entities: (hyperuricemia, DISEASE)"}
```

### 2.5 BFV 缓存目录

```bash
/root/autodl-tmp/slg-bfv-cache/    # Config.bfv_cache_dir 默认
├── bfv_ct_db_n128256_d4096_p4096.bin   # Enc(-V[y]) ~16 GB, Stage 0 产出
├── bfv_pk.bin                          # Stage 0 产出
├── bfv_meta.json                       # Stage 0 产出
├── bfv_keys.json                       # Stage 0 产出 (无 sk,仅 metadata)
└── s3pir_hints/
    └── hint_table.json                 # Stage 0 产出
```

**重要**:每次**新的** `BFVPrivSelectV2Backend` 都生成新的 `sk/pk` 对;若想让密文 DB 可被解,Stage 1 启动时必须用与 Stage 0 同一对 `sk/pk`(详见 §6.4.6 / §6.6)。

---

## 3. 调用流程

### 3.1 标准三阶段流程(推荐)

```bash
cd /root/autodl-tmp/SLG-HE-PIR

# Stage 0: 一次性离线准备(约 3 min)
python src/scripts/finetune.py --stage 0
# 产出: bfv_ct_db_*.bin, bfv_pk.bin, bfv_meta.json, hint_table.json

# Stage 1: 在线训练(取决于 max_epochs; 2 epoch ≈ 31 min)
python src/scripts/finetune.py --stage 1 --max_epochs 10
# 产出: logs/finetune_stage1_*.log, checkpoints/checkpoint_epoch_*.pt

# Stage 2: 加载 best checkpoint 做 test 评测(当前会触发 §6.4.6 ImportError)
python src/scripts/finetune.py --stage 2 --checkpoint checkpoints/best_checkpoint.pt
```

### 3.2 完整流程(Stage 0 → 1 → 2)

```bash
python src/scripts/finetune.py --stage all --max_epochs 10 --do_test_eval
```

### 3.3 快速 smoke 测试(2 epoch,推荐用于首次验证)

```bash
# 复用 two_epoch_test 的初始化路径,但只跑 2 epoch
python scripts/two_epoch_test.py --max_epochs 2 --batch_size 4
# 默认 force=True 重建 DB(避免 §6.6 描述的 cache key 错位)
```

### 3.4 10-step smoke(诊断显存/协议/参数流向,~1-2 min)

```bash
python scripts/quick_smoke_10step.py
```

### 3.5 单 batch 端到端正确性

```bash
python scripts/heterogeneous_correctness_test.py
# 验证 mask 消去 + LoRA 收敛 + 不 OOM + sk_M 不在主进程
```

### 3.6 流水线 vs flat 路径正确性

```bash
python scripts/chunk_correctness_test.py
# 验证 USE_CHUNKED_PIPELINE 路径与 flat 路径 bit-exact 等价
```

### 3.7 性能基准(3 个热点路径)

```bash
python scripts/perf_bench.py
# 输出 SERIAL vs PARALLEL 表
```

---

## 4. CLI 速查表

### 4.1 `src/scripts/finetune.py`

```bash
python src/scripts/finetune.py \
    --stage {0|1|2|all}             # 默认 all
    --max_epochs 10                 # 训练 epoch 数
    --batch_size 4                  # 训练 batch size
    --use_chunked_pipeline true     # true / false
    --chunk_tokens 3072             # 每个 PIR+反传 chunk 的 token 数
    --n_crypto_u_workers 8          # U 子进程数
    --n_crypto_m_workers 8          # M 子进程数
    --n_crypto_s_workers 1          # S 子进程数
    --skip_db                       # Stage 0 跳过加密 DB 构建
    --skip_hints                    # Stage 0 跳过 hint 构建
    --checkpoint checkpoints/best_checkpoint.pt  # Stage 2 用
    --do_test_eval                  # Stage 1 训练完后跑 test
    --dump_attacks                  # 落盘 attack 中间量
    --log_freq 10                   # 日志频率
    --log_dir logs                   # 日志目录
    --config override.json           # JSON 覆盖(覆盖 Config 默认值)
```

#### 4.1.1 JSON 覆盖示例(`override.json`)

```json
{
  "max_epochs": 2,
  "batch_size": 4,
  "bfv_cache_dir": "/tmp/slg-bfv-cache-test",
  "data_dir": "/root/slg-v2.0/data/biotriplex_qa",
  "USE_CHUNKED_PIPELINE": true,
  "CHUNK_TOKENS": 3072,
  "do_test_eval": false,
  "dump_attacks": false
}
```

### 4.2 `scripts/two_epoch_test.py`

```bash
python scripts/two_epoch_test.py \
    --max_epochs 2                  # 默认 2
    --data_dir /root/slg-v2.0/data/biotriplex_qa
    --bfv_cache_dir /root/autodl-tmp/slg-bfv-cache
    --batch_size 4
    --verbose / -v                  # 调试日志
```

### 4.4 `src/scripts/biotriplex_finetune.py`

```bash
python src/scripts/biotriplex_finetune.py \
    --task_type {classification|generation}  # 必选：任务类型
    --stage {0|1|2|all}             # 默认 all
    --data_path "/path/to/Preprocessed BioTriplex/"  # 数据路径
    --output_dir ./output            # 输出目录
    --max_epochs 6                  # 分类默认 6；生成默认 10
    --batch_size 1                  # 推荐 1
    --max_seq_length 2048           # 根据显存调整

    # === GPU 内存优化 (v2.3, SageAttention3 支持) ===
    --use_flash_attention True      # FlashAttention2 (默认开启)
    --use_sage_attention True       # SageAttention2++/3 (默认开启, RTX 5090 推荐)
    --use_deepspeed_zero True       # DeepSpeed ZeRO (默认开启)
    --zero_stage 1                  # ZeRO-1 单 GPU 推荐
    --gradient_checkpointing_style reentrant  # reentrant 或 full

    # === 其他参数 ===
    --skip_db                       # 跳过 Stage 0 加密 DB 构建
    --skip_hints                    # 跳过 Stage 0 hint 构建
    --log_freq 10                  # 日志频率
    --save_freq 1                  # checkpoint 保存频率
    --do_test_eval                  # 训练后运行测试评估
```

### 4.3 其它脚本(无 CLI 或简单 CLI)

| 脚本 | 用途 | 调用 |
|---|---|---|
| `scripts/quick_smoke_10step.py` | 10-step 显存/协议 smoke | `python scripts/quick_smoke_10step.py` |
| `scripts/heterogeneous_correctness_test.py` | 单 batch 端到端正确性 | `python scripts/heterogeneous_correctness_test.py` |
| `scripts/chunk_correctness_test.py` | chunked vs flat 正确性 | `python scripts/chunk_correctness_test.py` |
| `scripts/e2e_correctness_recheck.py` | 串行 vs 并行 全链路 | `python scripts/e2e_correctness_recheck.py` |
| `scripts/e2e_math_verify.py` | 代数关系验证 | `python scripts/e2e_math_verify.py` |
| `scripts/diag_grad_flow.py` | 梯度流诊断 | `python scripts/diag_grad_flow.py` |
| `scripts/perf_bench.py` | 3 个热点性能基准 | `python scripts/perf_bench.py` |
| `scripts/compare_step_profiles.py` | step profile 对比 | `python scripts/compare_step_profiles.py --help` |
| `scripts/test_step_profiler.py` | profiler 单元测试 | `python scripts/test_step_profiler.py` |
| `scripts/test_trainer_dispatch.py` | Trainer 派发测试 | `python scripts/test_trainer_dispatch.py` |

---

## 5. 关键参数说明

### 5.1 必看(改错就崩)

| 参数 | 默认值 | 含义 | 改错后果 |
|---|---|---|---|
| `vocab_size` | 128256 | Llama 词表大小,决定 Enc DB 行数 | DB 大小不匹配 |
| `hidden_dim` | 4096 | V 矩阵列数 = Enc DB 每行维度 | V 加载报错 |
| `poly_degree` | 4096 | BFV 多项式度,**必须 ≥ hidden_dim** | SEAL 上下文失败 |
| `plain_bits` | 30 | BFV plaintext modulus bits | 大值溢出,小值精度差 |
| `scale` | 10000 | 量化缩放因子,**必须与 `server_make_share` 内一致** | mask 消不掉 |
| `hf_model` | `/root/autodl-tmp/hf_cache/Llama-3-1-8B-I` | HF 快照路径 | 找不到模型 |
| `bfv_cache_dir` | `/root/autodl-tmp/slg-bfv-cache` | Stage 0 产物落盘目录 | cache 找不到 |

### 5.2 性能调优(可改)

| 参数 | 默认 | 含义 | 调优建议 |
|---|---|---|---|
| `batch_size` | 4 | 训练 batch | 减小到 2 或 1 可省显存;增大到 8 更快但 OOM 风险高 |
| `max_seq_length` | 128 | 最大序列长度 | 减小到 64 可省显存;增大到 256 适合长文本 |
| `USE_CHUNKED_PIPELINE` | True | 启用分块流水线 | 大 batch 时必须 True |
| `CHUNK_TOKENS` | 3072 | 每 chunk token 数 | 减小可省显存,增大可加速 |
| `N_CRYPTO_U_WORKERS` | 8 | U 子进程数 | 增到 16 可加速 U 端;但 RAM 用量翻倍 |
| `N_CRYPTO_M_WORKERS` | 8 | M 子进程数 | 同上;受 RAM 限制 |
| `N_CRYPTO_S_WORKERS` | 1 | S 子进程数 | **不要改**;hint table 与 DB mmap 是只读 |
| `learning_rate` | 3.5e-4 | AdamW peak lr | 保守调小到 1e-4 |
| `warmup_steps` | 200 | LR warmup 步数 | 长训练可增到 500 |

### 5.3 GPU 内存优化 (v2.3 SageAttention)

| 参数 | 默认 | 含义 | 调优建议 |
|---|---|---|---|
| `use_sage_attention` | True | 启用 SageAttention (INT8/FP4 量化注意力) | RTX 5090 推荐开启，自动选择 SageAttention3 FP4 |
| `use_flash_attention` | True | 启用 FlashAttention2 (O(N) 内存) | 默认开启；如有问题可设为 False 回退到 SDPA |
| `use_deepspeed_zero` | True | 启用 DeepSpeed ZeRO 优化器状态分片 | 单 GPU 推荐 ZeRO-1；多 GPU 可用 ZeRO-2/3 |
| `zero_stage` | 1 | ZeRO 阶段：1=优化器状态，2=+梯度，3=+参数 | 单 GPU 用 1；多 GPU 可用 2 或 3 |
| `gradient_checkpointing_style` | "reentrant" | 梯度检查点风格 | "reentrant" 节省 ~50% 激活内存；"full" 内存最低但最慢 |

#### SageAttention 优先级

```
1. SageAttention3 FP4 (RTX 5090 Blackwell) — 5x 速度提升，75% 显存降低
2. SageAttention2++ INT8 — 2.7x 速度提升，50% 显存降低
3. FlashAttention2 — O(N) 内存
4. SDPA — fallback
```

> **注意**: SageAttention3 需要 CUDA 12.8+ 和 PyTorch 2.3+。不支持的硬件会自动降级。

### 5.4 诊断用(默认 False / 0)

| 参数 | 默认 | 含义 |
|---|---|---|
| `dump_attacks` | False | 落盘中间量到 `logs/attack_dumps/`(每 token 几十 KB) |
| `log_freq` | 10 | 每 N step 打一条 log |
| `do_test_eval` | False | Stage 1 训练完后调 `_run_test_epoch()` |
| `val_metric` | "val_entity_micro_f1" | best-metric 选择指标 |

---

## 6. 输出物清单

每次 `python src/scripts/finetune.py --stage 1` 跑完会在以下位置产生:

```
${log_dir}/                             # 默认 /root/autodl-tmp/SLG-HE-PIR/logs/
├── finetune_stage1_<timestamp>.log      # 主日志(INFO+)
├── epoch_metrics.jsonl                  # 每个 epoch 一行 JSON
└── attack_dumps/                        # 仅当 --dump_attacks
    └── step_<N>_t_<K>.npz             # 单 token 中间量

${checkpoint_dir}/                      # 默认 /root/autodl-tmp/SLG-HE-PIR/checkpoints/
├── checkpoint_epoch_000.pt            # ~128 MB,epoch 0 LoRA + AdamW state
├── checkpoint_epoch_001.pt            # ~128 MB,epoch 1 ...
├── last_checkpoint.pt                  # 最新 epoch 的 alias
└── best_checkpoint.pt                  # 最佳 val_metric 对应 epoch
```

`epoch_metrics.jsonl` 示例:

```json
{"epoch": 0, "train_loss": 2056.41, "val_ce_loss": 0.0, "val_entity_micro_f1": 0.0, "val_macro_f1": 0.0, ...}
{"epoch": 1, "train_loss": 2052.90, "val_ce_loss": 0.0, "val_entity_micro_f1": 0.0, "val_macro_f1": 0.0, ...}
```

---

## 7. 常见错误 & 排错

### 7.1 `RuntimeError: incompatible version`(SEAL 加载 Enc DB 失败)

**触发**:`CryptoMWorker` 解密旧 `bfv_ct_db_*.bin` 时报错。

**原因**:**key-parity bug**(`§6.6`)。当前 run 生成的 `sk/pk` 与 cache 中的 Enc DB 用的 `sk/pk` **不是同一对**,解密出来是乱码。

**解决**:
```bash
# 1. 删旧 cache
rm -rf /root/autodl-tmp/slg-bfv-cache/

# 2. 重新跑 Stage 0
python src/scripts/finetune.py --stage 0

# 3. 用同一对 sk/pk 跑 Stage 1(默认行为;production 路径不需要 force)
python src/scripts/finetune.py --stage 1
```

**两 epoch test 已经默认 `force=True`**(见 `scripts/two_epoch_test.py:181`),所以该脚本不会触发此问题。

### 7.2 `CUDA out of memory`(训练中 OOM)

**触发**:Phase 1 早期(未修复 `drop_encrypted_db` 时);修复后稳定在 ~17 GB。

**常见原因与解决**:
- `batch_size` 太大 → 改为 2 或 1
- `max_seq_length` 太大 → 改为 64 或 128
- `CHUNK_TOKENS` 太小(导致 chunk 数过多、碎片化) → 改为 1536 或 2048
- 显存碎片化 → 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 确保走 `step_train_chunked`(默认);eval-FP 路径显存占用更低

**GPU 内存优化选项**:
- 启用 SageAttention: `--use_sage_attention True` (默认开启，RTX 5090 自动使用 FP4，5x 加速)
- 启用 FlashAttention2: `--use_flash_attention True` (默认开启，O(N) 内存)
- 启用 DeepSpeed ZeRO: `--use_deepspeed_zero True` (默认开启，需安装 deepspeed)
- 梯度检查点风格: `--gradient_checkpointing_style full` (最低内存)

> **2026-07-21 内存优化**:SageAttention3 (FP4) 集成完成，可在 RTX 5090 上提供 5x 加速和 75% 显存降低。

> **2026-07-21 内存优化**:BFV Encrypted DB 现在只加载到 CPU worker 子进程中。主进程 `BFVPrivSelectV2Backend` 在 `build_encrypted_database()` 后调用 `drop_encrypted_db()` 释放 ~16 GB CPU 内存,不再持有 `_ct_list`(`List[bytes]`)副本。Workers 通过 `BFVEncryptedDatabase.from_cache(..., load_ct_list=True)` 独立持有自己的 mmap 副本。

### 7.3 `loss_proxy = 124000` 恒定(Phase 1 v2.1 状态)

**触发**:mask 没消干净;`train_loss_proxy` 是 ∑|masked_diff| 而非真梯度。

**原因与解决**:这是 **Phase 1 v2.1 bug 期数据**,v2.2 已修复。详见 `docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md` §6.6。如果还看到:

1. 确认 `scripts/two_epoch_test.py:181` 是 `force=True`(不是 `force=False`)
2. 确认 `src/core/bfv_privselect_v2_adapter.py::server_make_share` 是 `s_share = round(a_t · scale) − r_t`
3. 确认 `src/parties/crypto_workers/crypto_u.py::handle_request` 没有 `pos_r_t = (x % pm)`
4. 确认 `src/parties/party_m.py::backward_and_update` 第 296-302 行有 `masked_centered` 居中

### 7.4 `val_entity_micro_f1 = 0` + `val_ce_loss = 0`

**触发**:Phase 1 v2.2 已知缺陷,详见 `docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md` §6.4.1。

**短期**:val 指标没意义,只看 `train_loss_proxy` 单调下降判断训练收敛。

**长期修复**:改 `src/data/dataset.py::BioTriplexQADataset.__getitem__` 多返回 `output_ids` + 改 `parse_answer_letter` 为 NER span 解析。

### 7.5 `ImportError: cannot import name 'parse_gold_entities'`

**触发**:`python src/scripts/finetune.py --stage 2 --do_test_eval`。

**原因**:`src/training/evaluation.py:208` 引用了 `src/data/dataset.py` 中**不存在的**函数 `parse_gold_entities`。详见 §6.4.6。

**解决**:补齐 `parse_gold_entities(text)` 把 `"Entities: (hyperuricemia, DISEASE); (PRAL, GENE)"` 解析为 `{("hyperuricemia", "DISEASE"), ("PRAL", "GENE")}`(尚未实现)。

### 7.6 Stage 0 跑完后没有任何产物

**可能原因**:
- `bfv_cache_dir` 路径不存在或不可写 → `mkdir -p /root/autodl-tmp/slg-bfv-cache/`
- 没等到 build 完成(约 3 min)→ 看日志 `=== Stage 0 Step 1 Complete ===`
- `force=False` 且 cache 已存在 → 脚本会跳过;若想强制重建,设 `force=True`

### 7.7 GPU `expandable_segments` 警告

**触发**:`UserWarning: expandable_segments not supported on this platform`。

**原因**:CUDA < 11.4 不支持 `expandable_segments`。

**解决**:`export PYTORCH_CUDA_ALLOC_CONF=`(留空)或升级 CUDA ≥ 11.4。

---

## 8. 理解"为什么"(快速参考)

| 问题 | 简短答案 | 详细位置 |
|---|---|---|
| 为什么 U/M/S 在同一进程? | 单机模拟三方 + GPU tensor 零拷贝;多 spawn CPU 子进程隔离 sk_M | §2.1, §4.5 |
| 为什么用 spawn 而非 fork? | spawn 不会继承 CUDA caching allocator(避免 8 worker × 6 GB OOM) | §4.5.1 决策表 |
| 为什么 V 矩阵要加密? | S 不能让 U 看到 V 全文;但需要 U 通过 PIR 查单行 → 必须 Enc 单行 | §1, §3.1 |
| 为什么 S 端 share `a_t - r_t`? | 让 M 解密 U 的密文 + 加 s_share 时消去 r_t,得到纯梯度 `a_t - V_y` | §3.3, §4.6.3 |
| 为什么 LoRA 在 bf16 训练? | 32 GB 显存放不下 fp32 LoRA + 32 层 decoder + 激活;bf16 减半 | §4.4, §4.2 |
| 为什么 `train_loss_proxy` 不是真 loss? | 训练没有显式 label;LoRA 是被 S3PIR "隐式监督"的;`val_ce_loss` 才是真 loss(但当前 = 0) | §6.4.1, §6.4.2 |
| 为什么 val 指标全是 0? | 数据是 NER 但评测层按 letter-QA 设计,且数据集缺 `output_ids` 字段 | §6.4.1 |
| 为什么强制 `force=True` 重建 DB? | 否则 fresh sk 解旧 pk 加密的密文 → 乱码 → mask 看起来没消 | §6.6 (B) |

---

## 9. 新会话的"接力"姿势

如果你接手这个项目,按以下顺序读:

1. **本文档**(USAGE.md, 5 min)— 知道怎么跑、什么参数、常见错误
2. **`docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md` §1-§3**(15 min)— 知道数据流怎么走
3. **跑一次 `python scripts/two_epoch_test.py --max_epochs 2`**(30 min,带检查)— 验证环境
4. **读 §4.5、§4.6、§4.7**(30 min)— 知道密码学层细节
5. **读 `docs/HANDOFF.md` / `docs/newHANDOFF.md`**(15 min)— 知道历史决策与已知坑
6. **读 `docs/BIOTRIPLEX_FINETUNE_README.md`**(可选, 10 min)— 知道任务和数据

如果中途有疑问,**先看**:
- `train_loss_proxy` 行为异常 → §6.6
- `val_* = 0` → §6.4.1
- OOM → §7.2
- SEAL 错误 → §7.1
- 想改评测 → §6.4.1 修复路线
- 想加新 party 或协议 → §2.2 设计原则 + `IPCProtocol` 接口

如果还是不懂,直接看代码:
- `src/scripts/finetune.py` — 主入口
- `src/parties/heterogeneous_protocol.py` — 协议编排
- `src/parties/party_m.py::backward_and_update` — 反向传播 + LoRA 更新(密码学最密集)
- `src/core/bfv_privselect_v2_adapter.py::BFVPrivSelectV2Backend` — BFV backend

---

## 10. 附:产物的可复现性

每次跑完整流程后,以下文件应**保留**以便复现:

```
/root/autodl-tmp/slg-bfv-cache/                    # Stage 0 产物(16 GB)
├── bfv_ct_db_n128256_d4096_p4096.bin
├── bfv_pk.bin
├── bfv_meta.json
├── bfv_keys.json
└── s3pir_hints/hint_table.json

/root/autodl-tmp/SLG-HE-PIR/checkpoints/          # Stage 1 产物
├── best_checkpoint.pt
├── last_checkpoint.pt
└── checkpoint_epoch_*.pt

/root/autodl-tmp/SLG-HE-PIR/logs/                 # 日志
├── finetune_stage1_*.log
└── epoch_metrics.jsonl
```

删除任一会导致 Stage 1 失败或无法恢复训练。
