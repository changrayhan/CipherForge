# SLG-HE-PIR 功能测试套件

## 目录结构

```
scripts/
└── function-tests/
    ├── __init__.py                  # 包初始化（库存清单）
    ├── README.md                    # 本文档
    ├── e2e_math_verify.py           # 端到端数学验证
    ├── e2e_correctness_recheck.py   # 串行 vs 并行正确性
    ├── chunk_correctness_test.py    # Chunked vs flat 正确性
    ├── perf_bench.py                # 完整规模性能基准
    ├── bench_optimizations.py       # 加密流水线微基准
    ├── heterogeneous_correctness_test.py  # v2.0 异构运行时测试
    ├── run_small_scale_test.py      # 小规模 7 项烟雾测试
    ├── two_epoch_test.py            # 2 epoch 收敛验证
    ├── quick_smoke_10step.py        # 10 step 显存烟雾测试
    ├── diag_grad_flow.py            # 梯度流诊断
    ├── test_step_profiler.py       # StepProfiler 单元测试
    ├── test_trainer_dispatch.py     # Trainer 分发逻辑测试
    ├── compare_step_profiles.py     # Flat vs Chunked JSONL A/B 对比
    ├── diff_step_profiles.py        # Task A 优化速度报告
    ├── _demo_profile_jsonl.py      # 合成 JSONL 数据（Dashboard 开发用）
    └── run_with_pyc_finder.py      # 缺失 .py 源码时的启动器
```

## 运行方式

所有脚本均可从仓库根目录执行：

```bash
# 方式 1：作为模块运行（推荐）
python -m scripts.function_tests.<script_name> [args...]

# 方式 2：直接执行
python scripts/function-tests/<script_name>.py [args...]

# 方式 3：从 scripts/function-tests/ 内运行
cd scripts/function-tests
python <script_name>.py [args...]
```

## 脚本分类

### 一、端到端数学/正确性验证

| 脚本 | 功能 | 关键输出 |
|------|------|---------|
| `e2e_math_verify.py` | Design-2 协议端到端数学验证（N=64） | 梯度恢复误差 < 0.5，隐私边界检查 |
| `e2e_correctness_recheck.py` | S/U/M 三段热路径串行 vs 并行一致性 | 掩码密文字节级相同，梯度误差 < 1e-2 |
| `chunk_correctness_test.py` | Chunked pipeline vs flat pipeline 正确性 | 4 种 chunk size 逐字节验证 |

### 二、性能基准

| 脚本 | 功能 | 关键输出 |
|------|------|---------|
| `perf_bench.py` | 全规模（N=128256，24,576 tokens/step）性能 | S/U/M 各段 speedup 表格 |
| `bench_optimizations.py` | 5 步加密流水线微基准 | PRG/tmpfs/encoder 各步 ms/call |

### 三、运行时测试

| 脚本 | 功能 | 关键输出 |
|------|------|---------|
| `heterogeneous_correctness_test.py` | v2.0 异构运行时完整测试 | 6 项 CHECK pass/fail |
| `run_small_scale_test.py` | 小规模（N=1024）7 组件流水线 | 7 项 TEST pass/fail |

### 四、训练主流程

| 脚本 | 功能 | 关键输出 |
|------|------|---------|
| `two_epoch_test.py` | 2 epoch 收敛验证 | loss↓ / F1↑ 收敛判据 |
| `quick_smoke_10step.py` | 10 step 显存烟雾测试 | peak GPU memory，< 32GB |
| `diag_grad_flow.py` | 1 step 梯度流诊断 | grad norm 日志，H_U/H_M requires_grad |

### 五、Step Profiler / Trainer 测试

| 脚本 | 功能 | 关键输出 |
|------|------|---------|
| `test_step_profiler.py` | StepProfiler 5 项单元测试 | pass/fail |
| `test_trainer_dispatch.py` | Trainer 分发逻辑测试 | flat/chunked 路由正确性 |
| `compare_step_profiles.py` | 两份 JSONL A/B 对比报告 | `compare.md` + 柱状图 |
| `diff_step_profiles.py` | Task A 优化速度报告 | keep/partial/marginal/revert 决策 |

### 六、工具

| 脚本 | 功能 |
|------|------|
| `_demo_profile_jsonl.py` | 生成合成 flat/chunked JSONL（Dashboard 开发） |
| `run_with_pyc_finder.py` | .py 源码缺失时从 .pyc 启动 |

## 快速运行指南

```bash
# 最快的正确性验证（~1 分钟）
python scripts/function-tests/e2e_math_verify.py

# 小规模烟雾测试（~2 分钟）
python scripts/function-tests/run_small_scale_test.py

# 10 step 显存测试（~10 分钟，取决于硬件）
python scripts/function-tests/quick_smoke_10step.py

# 异构运行时端到端（~10 分钟）
python scripts/function-tests/heterogeneous_correctness_test.py

# 完整 2 epoch 收敛测试（~1 小时）
python scripts/function-tests/two_epoch_test.py --max_epochs 2

# 全规模性能基准（需要 GPU，约 30 分钟）
python scripts/function-tests/perf_bench.py

# Step Profiler 单元测试（< 1 分钟）
python scripts/function-tests/test_step_profiler.py

# Trainer 分发测试（< 1 分钟）
python scripts/function-tests/test_trainer_dispatch.py

# 梯度流诊断
python scripts/function-tests/diag_grad_flow.py --batch-size 4 --max-length 128
```
