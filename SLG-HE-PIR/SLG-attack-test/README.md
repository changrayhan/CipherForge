# SLG-HE-PIR Attack Test Suite (v2.0)

Security attack test suite for the SLG-HE-PIR heterogeneous PIR training protocol.
Tests four core attack surfaces based on the updated TEST_REPORT.md.

**协议版本：** SLG-HE-PIR v2.0（含 BFV 加密层 + S3PIR 检索）
**测试模式：** 静态分析 + 数据录制 + 统计分析（GPU-free 测试优先）
**数据集：** TREC-QC（6 类粗粒度分类）

---

## 4 Core Attacks (Based on TEST_REPORT.md)

| ID | 名称 | 威胁方 | 目标数据 | 严重度 |
|----|------|--------|----------|--------|
| **L-1** | M方梯度标签推断攻击 | M（诚实但好奇） | `g_{H,t} = a_t - V_y` | HIGH |
| **L-2** | S方激活值标签推断攻击 | S（诚实但好奇） | `a_t`, `result_S` | MEDIUM |
| **M-1** | U方模型推断攻击 (评估阶段) | U（诚实但好奇） | S's predictions during evaluation | HIGH |
| **M-2** | S方隐藏层反演攻击 | S（诚实但好奇） | `Z_t = H_M @ V^T` | MEDIUM |

**协议数据边界说明：**
- U 在整个协议流程中**不持有** `H_0* ~ H_ans*`，因此基于中间层表示的模型推断攻击不适用于 U 作为攻击者的场景
- M-1 为**评估阶段攻击**：评估阶段 U 可以收集 S 返回的预测结果，训练替代模型

---

## 目录结构

```
SLG-attack-test/
├── README.md
├── requirements.txt
├── run_attack_suite.py            # 统一入口
├── attacks/
│   ├── __init__.py
│   ├── base.py                   # BaseAttack 基类
│   ├── L1_gradient_inference.py  # L-1: M方梯度标签推断
│   ├── L2_activation_inference.py # L-2: S方激活值标签推断
│   ├── M1_logits_distillation.py  # M-1: U方Logits蒸馏 (条件)
│   └── M2_hidden_inversion.py     # M-2: S方隐藏层反演
├── protocol/
│   └── attack_protocol_wrapper.py # 协议劫持层
├── evaluation/
│   ├── metrics.py                # 评估指标
│   └── reporter.py               # 结果报告
├── config/
│   └── attack_config.py          # 配置管理
└── data/
    └── trecqc_dataset.py        # TREC-QC 数据适配器
```

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/changrayhan/hCode/SLG-HE-PIR-code/SLG-HE-PIR
source llmTest/bin/activate
pip install -r SLG-attack-test/requirements.txt
```

### 2. 运行完整攻击套件

```bash
python SLG-attack-test/run_attack_suite.py \
    --attacks L1,L2,M1,M2 \
    --n_steps 20 \
    --hf_model /home/changrayhan/hCode/SLG-HE-PIR-code/hf_cache/Llama-3-1-8B-I \
    --data_dir /home/changrayhan/hCode/SLG-HE-PIR-code/SLG-HE-PIR/datasets/trec-qc \
    --output_dir SLG-attack-test/results
```

### 3. 仅运行标签推断攻击

```bash
python SLG-attack-test/run_attack_suite.py --attacks L1,L2
```

### 4. 仅运行模型推断攻击

```bash
python SLG-attack-test/run_attack_suite.py --attacks M1,M2
```

### 5. 自定义参数

```bash
# 自定义 L-1 置换检验强度
python SLG-attack-test/run_attack_suite.py \
    --attacks L1 \
    --l1_n_permutations 10000 \
    --l1_alpha 0.05

# 启用 M-1（需协议扩展支持 logits 回传）
python SLG-attack-test/run_attack_suite.py --attacks M1 --m1_logits_available
```

---

## 攻击详解

### L-1: M方梯度标签推断攻击

**威胁模型：** M（诚实但好奇）

**手中数据：** `g_{H,t} = a_t - V_y`（BFV 解密后）

**攻击方法：**
1. 拦截协议反向传播中的 `g_accum` 梯度向量
2. 构建梯度矩阵 $G \in \mathbb{R}^{N \times d}$
3. 五项统计检验：K-Means ARI、1-NN 一致率、Cosine AUC、置换检验、梯度幅度 ANOVA

**评测指标：**

| 指标 | 泄露判定标准 | 期望（PRG有效） |
|------|-------------|-----------------|
| K-Means ARI | > 0.1 | ARI ≈ 0 |
| 1-NN 一致率 | > 16.7% + 2σ | ≈ 16.7% |
| Cosine AUC | > 0.5 + 2σ | ≈ 0.5 |
| 置换检验 p-value | < 0.05 | > 0.05 |
| 梯度幅度 ANOVA p-value | < 0.05 | > 0.05 |

### L-2: S方激活值标签推断攻击

**威胁模型：** S（诚实但好奇）

**手中数据：** `a_t = softmax(Z) @ V`、`result_S = scale·a_t - r_t`、`V`

**攻击方法：**
1. 按真实标签分组，计算 `a_t` 的统计特性
2. KL 散度分析：各类别 `a_t` 分布与均匀分布的偏离
3. Chi-sq 检验：类别条件分布显著性

**评测指标：**

| 指标 | 泄露判定标准 | 期望（协议安全） |
|------|-------------|-------------------|
| 类间均值差异 ANOVA p-value | < 0.05 | > 0.05 |
| 类间范数差异 ANOVA p-value | < 0.05 | > 0.05 |
| KL 散度（各类 vs 均匀） | > 0.1 | ≈ 0 |

**关键说明：** S 持有 V 是协议设计的显式假设（S 没有 V 无法完成前向传播）

### M-1: U方模型推断攻击（评估阶段）

**威胁模型：** U（诚实但好奇）

**手中数据：** S 在评估阶段返回的预测结果

**攻击方法：**
1. 收集 S 的评估预测（预测标签、置信度）
2. 预测一致性分析：分析置信度分布
3. 替代模型训练：使用收集的预测数据训练替代模型
4. 信息泄露分析：检测预测模式中的信息泄露

**评测指标：**

| 指标 | 泄露判定标准 | 期望（协议安全） |
|------|-------------|-------------------|
| 置信度方差 | > 0.1 | ≈ 0 |
| 预测多样性 | > 0.7 | 高多样性 |
| 替代模型准确率提升 | > 10% vs 基线 | 无显著提升 |

**关键说明：** 如果 S 只返回最终预测标签，U 很难从中提取模型知识；如果返回置信度或 top-k 预测，则存在潜在风险

### M-2: S方隐藏层反演攻击

**威胁模型：** S（诚实但好奇）

**手中数据：** `Z_t = H_M @ V^T`、`V`

**攻击方法：**
1. 矩阵反演：`H_M ≈ Z @ V^T`（伪逆）
2. 维度分析：检测低秩结构（LoRA 特征）
3. 重建误差：`||H_M - Z @ V^+||_F / ||H_M||_F`

**评测指标：**

| 指标 | 泄露判定标准 | 期望（协议安全） |
|------|-------------|-------------------|
| 重建误差 | < 0.5 则泄露 | > 0.5 |
| H_M 秩估计 | 若接近 LoRA rank=8 则泄露 | 应大于 8 |

**关键说明：** V 通常是满秩的，而 `H_M` 维度与 `V^T` 维度不匹配，直接反演不可行

---

## 输出格式

```
SLG-attack-test/results/
├── attack_results.json     # 所有 AttackVerdict 的 JSON
├── run_metadata.json      # 运行配置
└── {attack_id}/
    ├── gradient_matrix.npy # (N, 4096) 梯度矩阵
    ├── activation_matrix.npy # (N, 4096) 激活矩阵
    ├── label_array.npy    # (N,) 标签数组
    ├── singular_values.npy # SVD 奇异值
    └── metadata.json      # 元信息
```

### AttackVerdict Schema

```json
{
  "attack_id": "L1",
  "sub_attack": "kmeans_ari",
  "metric": "adjusted_rand_index",
  "value": 0.023,
  "chance_level": 0.0,
  "p_value": 0.31,
  "n_samples": 80,
  "verdict": "PRIVACY_PRESERVED",
  "notes": "ARI near 0: gradient space does not encode labels"
}
```

---

## 设计原则

1. **不修改 src/**：所有劫持通过 `AttackProtocolWrapper` 从外部包装
2. **GPU-free 优先**：可使用合成数据测试；真实攻击需要 GPU + BFV + 模型
3. **最小步数**：20 步 × batch_size=4 = 80 样本，对 6 类分类足够做统计功效分析
4. **条件激活**：M-1 仅在协议支持 logits 回传时适用

---

## 扩展攻击套件

新增攻击只需继承 `BaseAttack` 并在 `run_attack_suite.py` 注册：

```python
from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict

class L7CustomAttack(BaseAttack):
    ATTACK_ID = "L7"
    ATTACK_NAME = "Custom Attack"

    def run(self) -> list:
        return [AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="custom_metric",
            value=0.5,
            verdict="PRIVACY_PRESERVED",
            notes="...",
        )]
```
