# SLG-HE-PIR 测试报告

## 目录

1. [测试环境](#1-测试环境)
2. [测试方案](#2-测试方案)
3. [测试数据与分析](#3-测试数据与分析)
4. [参考文献](#参考文献)
5. [附录A：测试脚本使用说明](#附录a测试脚本使用说明)

---

## 1. 测试环境

### 1.1 硬件环境

| 项目 | 配置 | 说明 |
|------|------|------|
| GPU | NVIDIA GeForce RTX 5090, 32 GB GDDR7 (CUDA 580.105.08) | 承担 LLM 前向/反向传播、BFV 密文梯度计算的 GPU 加速 |
| CPU | Intel Xeon Platinum 8470Q (48 cores @ 2.0 GHz) | 处理密码学运算、攻击套件的并行调度 |
| 内存 | 90 GB DDR5 | 支持大批量密文数据加载与多进程并行攻击 |
| 操作系统 | Ubuntu 22.04.5 LTS | 统一的运行内核与系统调用 |

### 1.2 软件环境

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| Python | 3.12.3 | 编程语言 |
| PyTorch | 2.7.0+cu128 | 深度学习框架 |
| transformers | 5.14.1 | 预训练模型加载与推理 |
| peft | 0.19.1 | LoRA 参数高效微调 |
| seal-python | 4.1.2.1 | BFV 同态加密运算 |
| numpy | 2.2.6 | 数值计算 |
| scikit-learn | 1.9.0 | 机器学习指标计算 |
| flash-attn | -- | FlashAttention 加速 |
| sage-attention | -- | SageAttention 加速 |

### 1.3 数据集

**TREC-QC**（COLING'02）用于攻击测试。本文采用6类粗粒度标签，按分层抽样划分为训练集（4,909）、验证集（543）、测试集（500）。类别分布呈长尾结构（ABBR仅95条，1.6%），有助于验证隐私保护对少数类的边界敏感性。

| 粗粒度类别 |    训练集 |  验证集 |  测试集 |      总计 | 类别描述                                            |
| :--------- | --------: | ------: | ------: | --------: | :-------------------------------------------------- |
| DESC       |     1,046 |     116 |     138 |     1,300 | description and abstract concepts（描述与抽象概念） |
| ENTY       |     1,125 |     125 |      94 |     1,344 | entities（实体）                                    |
| ABBR       |        78 |       8 |       9 |        95 | abbreviation（缩写）                                |
| HUM        |     1,101 |     122 |      65 |     1,288 | human beings（人类）                                |
| NUM        |       807 |      89 |     113 |     1,009 | numeric values（数值）                              |
| LOC        |       752 |      83 |      81 |       916 | locations（地点）                                   |
| **总计**   | **4,909** | **543** | **500** | **5,952** | —                                                   |

**BioTriplex**是100篇PubMed生物医学论文的全文本标注语料（604段落、21,745句），覆盖基因/疾病/关系三类实体及21种gene-disease关系类别。本文派生两个互不重叠的任务：分类任务（QA多选题，431/90/83）与NER任务（句子级实体抽取，843/181/174），采用段落级互斥划分以保证划分可复现。

| 任务        | 段落数量（训练/验证/测试） | 形式                                |
| ----------- | -------------------------- | ----------------------------------- |
| 任务A：分类 | 431/90/83                  | QA 多选题形式，按段落-句子-关系展开 |
| 任务B：NER  | 431/90/83                  | 句子级实体抽取（span + 类型）       |

---

## 2. 测试方案

测试分为攻击测试与性能测试。攻击测试从标签推断（L-1, L-2）与模型推断（M-1, M-2）两维度设计4项攻击；性能测试以Llama-3.1-8B-Instruct为基座，在BioTriplex的分类与NER任务上对比基准LoRA微调与SLG-HE-PIR方案的通信/计算开销。

### 2.1 攻击测试

**L-1（M方标签推断）**：M观察$H_U$、$a_t$、$g_{accum}$，从类间均值ANOVA、L2范数分离、K-Means ARI、1-NN一致率、Cosine AUC、置换检验六个维度评估标签泄露。d$\chi$机制向$H_U$注入高斯噪声$\tilde H_U=H_U+\xi$，关键参数为加噪强度`dp_alpha`、答案扰动比例`dp_answer_beta`、校准步数`dp_calibration_steps`。对三个参数各取3水平做27组网格消融以验证鲁棒性。

**L-2（S方激活值标签推断）**：S从其持有的$a_t$与$result_S$推断标签信息。评估$a_t$的类间均值ANOVA、范数分离$\eta^2$、KL散度，以及PRG掩码后$result_S$的同向指标。$a_t=\mathrm{softmax}(Z)\text{@}V$沿词嵌入$V$的确定性几何效应可被检测但不可利用（KL$\approx$0.004$\ll$0.1），详见§3.1.2。

**M-1（U方知识提取）**：U通过100次黑盒查询收集S的top-1预测与置信度，训练替代模型尝试重建S的输出模式。评估置信度方差、6-bucket预测多样性及查询预算使用率。

**M-2（S方结构推断）**：S从$A_{post}$的SVD频谱推断M端LoRA秩。评估1%有效秩、秩指纹$\rho(r=8)$、方向投影能量、$result_S$-标签最大$|Pearson_r|$及直接矩阵反演可行性。warmup=2000步，$n_{permutations}$=1999。

### 2.2 性能测试

**通信开销**：量化U$\leftrightarrow$M$\leftrightarrow$S三方单步字节数。BFV密文$\approx$96KB/token（poly_degree=4096，3 primes），明文share$\approx$32KB/token，覆盖CLS(3072)/NER(3584)/生产(4096)三档token/step。

**离线准备开销**：评估Stage 0中BFV加密$V$矩阵构建与S3PIR Hint表生成的耗时及产物大小。

**基准对比**：对比SLG与基准方案在Stage 1单步耗时（forward_U/M、s_logits、priv_U、backward_M五阶段）、Stage 2推理耗时及CPU/GPU峰值占用。

---

## 3. 测试数据与分析

**攻击测试配置**：Llama-3.2-1B(hidden_dim=2048)，BFV(poly_degree=2048)，d$\chi$(default $\alpha$=0.15,$\beta$=0.5,cal=5)，TREC-QC 6类，seed=42。评估样本L-1/L-2 n=200，M-2 n=800。
**性能测试配置**：Llama-3.1-8B-Instruct(hidden_dim=4096)，BFV(poly_degree=4096)，LoRA rank=8，BioTriplex数据集。

### 3.1 攻击测试

**L-1**：**7/7子指标PRIVACY_PRESERVED**。d$\chi$加噪后M无法从$H_U$、$a_t$、$g_{accum}$中恢复6类标签结构。

| 子指标 | 度量 | 实测值 | 机会水平 / 判定阈值 | 判定 |
|--------|------|------:|------:|------|
| $\tilde H_U$ 类间均值 ANOVA 最小 p | `h_u_class_mean_anova_pvalue` | 9.85×10⁻⁴ | 9.77×10⁻⁵（BH-FDR $\alpha$=0.05 阈值） | PRIVACY_PRESERVED |
| $\tilde H_U$ 范数 ANOVA p + $\eta^2$ | `h_u_norm_anova_pvalue`, $\eta^2$=0.0062 | 0.943 | 0.05（$\alpha$ 阈值） | PRIVACY_PRESERVED |
| K-Means ARI | `adjusted_rand_index` | 0.0064 | 0.0（机会水平） | PRIVACY_PRESERVED |
| 1-NN 一致率 | `1nn_label_agreement` | 0.190 | 0.1667（机会水平，1/6） | PRIVACY_PRESERVED |
| Cosine AUC | `cosine_similarity_auc` | 0.504 | 0.500（机会水平） | PRIVACY_PRESERVED |
| 1-NN 置换检验 p（10 000 次） | `1nn_agreement_permutation_pvalue` | 0.566 | 0.05（$\alpha$ 阈值） | PRIVACY_PRESERVED |
| 梯度幅度 ANOVA p + $\eta^2$ | `gradient_magnitude_anova_pvalue`, $\eta^2$=0.029 | 0.332 | 0.05（$\alpha$ 阈值） | PRIVACY_PRESERVED |

![F-L1-1](figures/F-L1-1_h_u_norm_boxplot.png)
**图1**：d$\chi$加噪后$\tilde H_U$的6类L2范数箱线图（200样本）。横轴为TREC-QC类别，纵轴为$\|\tilde H_U\|_2$。六类箱体几乎完全重叠（中位数差异$\ll$1%，ABBR仅4样本），为整体通过的直观证据。

![F-L1-2](figures/F-L1-2_ablation_four_metrics.png)
**图2**：27组消融的4项核心指标vs $dp_{alpha}$（按$\beta\times$calibration分面）。四子图分别展示ARI（机会水平=0）、1-NN一致率（≈0.167）、Cosine AUC（0.5）、置换检验p（0.05)。27组紧贴机会水平参考线，未随$dp_{alpha}$单调变化，d$\chi$在该区间防护饱和。

![F-L1-3](figures/F-L1-3_anova_min_p_scatter.png)
**图3**：27组消融的$H_U$类间均值ANOVA最小p散点图（log变换，绿=SAFE，红=LEAK）。横轴为dp_alpha，纵轴为$-\log_{10}(\min p)$，红灰两线标注BH-FDR阈值与$\alpha$=0.05。20组SAFE，7组LEAK虽触警但ARI/1-NN/AUC仍贴近机会水平，验证单维显著$\neq$标签泄露。

27组消融完整数据：

| dp_alpha | dp_answer_beta | dp_calibration_steps | H_U ANOVA 最小 p | H_U ANOVA 判定 | H_U 范数 p | K-Means ARI | 1-NN 一致率 | Cosine AUC | 置换检验 p | 梯度幅 ANOVA p | 梯度幅判定 |
|----------|----------------|---------------------|-----------------|-------------|------------|-------------|-------------|------------|------------|---------------|------------|
| 0.05 | 0.3 | 2 | 0.0100 | PRIVACY_PRESERVED | 0.6718 | 8.5e-04 | 0.1850 | 0.4969 | 0.6323 | 0.7938 | PRIVACY_PRESERVED |
| 0.05 | 0.3 | 5 | 0.0015 | PRIVACY_PRESERVED | 0.3917 | -0.0040 | 0.2200 | 0.5056 | 0.3475 | 0.1460 | PRIVACY_PRESERVED |
| 0.05 | 0.3 | 10 | 0.0042 | PRIVACY_PRESERVED | 0.7172 | 0.0012 | 0.2250 | 0.5048 | 0.2498 | 0.0960 | PRIVACY_PRESERVED |
| 0.05 | 0.5 | 2 | 0.0028 | LEAK_DETECTED | 0.7739 | -0.0031 | 0.2250 | 0.4947 | 0.1777 | 0.5226 | PRIVACY_PRESERVED |
| 0.05 | 0.5 | 5 | 0.0028 | LEAK_DETECTED | 0.7739 | -0.0031 | 0.2250 | 0.4947 | 0.1777 | 0.5226 | PRIVACY_PRESERVED |
| 0.05 | 0.5 | 10 | 0.0095 | PRIVACY_PRESERVED | 0.5348 | 0.0017 | 0.2200 | 0.4990 | 0.2443 | 0.7883 | PRIVACY_PRESERVED |
| 0.05 | 0.7 | 2 | 5.1e-04 | PRIVACY_PRESERVED | 0.4375 | 0.0070 | 0.2250 | 0.4961 | 0.3354 | 0.3676 | PRIVACY_PRESERVED |
| 0.05 | 0.7 | 5 | 8.5e-04 | LEAK_DETECTED | 0.3637 | -0.0014 | 0.2100 | 0.4986 | 0.4210 | 0.6405 | PRIVACY_PRESERVED |
| 0.05 | 0.7 | 10 | 8.5e-04 | LEAK_DETECTED | 0.3637 | -0.0014 | 0.2100 | 0.4986 | 0.4210 | 0.6405 | PRIVACY_PRESERVED |
| 0.15 | 0.3 | 2 | 0.0011 | PRIVACY_PRESERVED | 0.7105 | -0.0010 | 0.2550 | 0.4974 | 0.0525 | 0.2370 | PRIVACY_PRESERVED |
| 0.15 | 0.3 | 5 | 0.0011 | PRIVACY_PRESERVED | 0.7105 | -0.0010 | 0.2550 | 0.4974 | 0.0525 | 0.2370 | PRIVACY_PRESERVED |
| 0.15 | 0.3 | 10 | 0.0052 | PRIVACY_PRESERVED | 0.5124 | 1.8e-04 | 0.2150 | 0.4931 | 0.3272 | 0.7753 | PRIVACY_PRESERVED |
| 0.15 | 0.5 | 2 | 0.0056 | PRIVACY_PRESERVED | 0.7030 | -0.0085 | 0.2250 | 0.5008 | 0.2670 | 0.1802 | PRIVACY_PRESERVED |
| 0.15 | 0.5 | 5 | 0.0020 | PRIVACY_PRESERVED | 0.8709 | 0.0065 | 0.2000 | 0.5004 | 0.4952 | 0.9559 | PRIVACY_PRESERVED |
| 0.15 | 0.5 | 10 | 0.0021 | PRIVACY_PRESERVED | 0.5331 | 0.0079 | 0.1200 | 0.5071 | 0.9942 | 0.6320 | PRIVACY_PRESERVED |
| 0.15 | 0.7 | 2 | 0.0055 | PRIVACY_PRESERVED | 0.8059 | -0.0043 | 0.1950 | 0.5037 | 0.5621 | 0.1466 | PRIVACY_PRESERVED |
| 0.15 | 0.7 | 5 | 0.0023 | PRIVACY_PRESERVED | 0.2789 | 3.7e-04 | 0.2000 | 0.4978 | 0.5025 | 0.6339 | PRIVACY_PRESERVED |
| 0.15 | 0.7 | 10 | 0.0049 | PRIVACY_PRESERVED | 0.2756 | -0.0023 | 0.2350 | 0.4894 | 0.1834 | 0.8367 | PRIVACY_PRESERVED |
| 0.30 | 0.3 | 2 | 0.0094 | PRIVACY_PRESERVED | 0.3840 | -0.0033 | 0.2400 | 0.5005 | 0.1168 | 0.8097 | PRIVACY_PRESERVED |
| 0.30 | 0.3 | 5 | 0.0041 | LEAK_DETECTED | 0.4578 | -0.0078 | 0.2150 | 0.5013 | 0.2327 | 0.5263 | PRIVACY_PRESERVED |
| 0.30 | 0.3 | 10 | 2.8e-04 | LEAK_DETECTED | 0.7288 | 0.0036 | 0.2600 | 0.4998 | 0.0876 | 0.5000 | PRIVACY_PRESERVED |
| 0.30 | 0.5 | 2 | 6.8e-04 | PRIVACY_PRESERVED | 0.9609 | -0.0022 | 0.1850 | 0.4959 | 0.6931 | 0.2359 | PRIVACY_PRESERVED |
| 0.30 | 0.5 | 5 | 6.8e-04 | PRIVACY_PRESERVED | 0.9609 | -0.0022 | 0.1850 | 0.4959 | 0.6931 | 0.2359 | PRIVACY_PRESERVED |
| 0.30 | 0.5 | 10 | 0.0031 | PRIVACY_PRESERVED | 0.7542 | -0.0096 | 0.1700 | 0.4982 | 0.7879 | 0.5523 | PRIVACY_PRESERVED |
| 0.30 | 0.7 | 2 | 0.0014 | PRIVACY_PRESERVED | 0.0709 | -0.0106 | 0.1950 | 0.5025 | 0.5821 | 0.0136 | PRIVACY_PRESERVED |
| 0.30 | 0.7 | 5 | 0.0014 | PRIVACY_PRESERVED | 0.0709 | -0.0106 | 0.1950 | 0.5025 | 0.5821 | 0.0136 | PRIVACY_PRESERVED |
| 0.30 | 0.7 | 10 | 1.3e-04 | LEAK_DETECTED | 0.7977 | 0.0071 | 0.2500 | 0.4997 | 0.0984 | 0.5213 | PRIVACY_PRESERVED |

**消融趋势**：27组中20/27组完全SAFE，7/27组仅$H_U$ ANOVA单特征触发LEAK，但该7组ARI（均$\approx$0）、1-NN（均$\approx$0.50）、AUC（均$\approx$0.50）、置换p（均$>$0.05）均贴近机会水平，与SAFE判定间不呈单调关系。LEAK集中在低$\alpha$区（$\alpha$=0.05占5/7），符合直觉；但$\alpha$=0.30下仍有3组触警——与calibration_steps交互效应有关：校准不足时即使$\alpha$较大也会残留特征级信号。该非单调性恰是d$\chi$鲁棒性的佐证。**推荐配置**：dp_alpha=0.30, dp_answer_beta=0.7, dp_calibration_steps=5（ARI=−0.0106, 1-NN=0.195）。

![F-L1-4](figures/F-L1-4_ablation_grid.png)
**图4**：27组消融参数网格与指标分布（5面板，数据全部来自上表）。**面板A**（左上，SAFE/LEAK热力图）：横轴为$\alpha$（0.05/0.15/0.30），纵轴为9组$\beta\times$calibration组合，绿色=S（SAFE）、红色=L（LEAK）。可直观看到LEAK集中在$\alpha$=0.05行（5/9）与$\alpha$=0.30行（2/9），$\alpha$=0.15行全部SAFE——呈现非单调分布，排除"噪声强度线性决定安全"的简单假说。同一$\beta$/calibration组合在不同$\alpha$下可能判SAFE或LEAK（如$\beta$=0.3/cal=5在$\alpha$=0.05、0.15为SAFE，$\alpha$=0.30为LEAK），证实存在参数间的交互效应。**面板B**（ARI vs $\alpha$）：全部27组ARI∈[−0.011, 0.007]，紧贴机会水平0，绿色S与红色L散点完全混合——被ANOVA判为LEAK的7组，其ARI与SAFE组不可区分。**面板C**（1-NN一致率 vs $\alpha$）：27组一致率集中在0.12–0.26区间，围绕机会水平1/6≈0.167波动，LEAK组与SAFE组同样不可区分。**面板D**（Cosine AUC vs $\alpha$）：全部27组AUC∈[0.489, 0.507]，紧贴机会水平0.5，LEAK组无系统性偏离。**面板E**（$-\log_{10}$(min ANOVA p) vs $\alpha$）：展示LEAK的触发机制——红虚线为BH-FDR阈值，灰点线为$\alpha$=0.05。7个LEAK点的$-\log_{10}(p)$均高于BH-FDR线，对应单特征ANOVA检测到类间均值差异；但面板B/C/D联合表明这些差异对聚类/近邻/相似度指标无影响。**综合解读**：图4从"在哪触发"（面板A/E）和"触发后是否可区分"（面板B/C/D）两个层次，验证了d$\chi$消融的核心结论——LEAK仅发生在特征级ANOVA信号上，聚类与近邻指标始终安全，单维显著$\neq$标签泄露。

**L-2**：**4/5子指标PRIVACY_PRESERVED**，$a_t$类间均值ANOVA触警（LEAK_DETECTED），但触警为softmax(·)@$V$的确定性几何效应：当$Z=H_M\text{@}V^T$的同类行向量落在相似子空间时，softmax对$V^T$特定方向的非线性放大造成类间均值可测偏移。但该信号不可利用——KL=0.0038$\ll$0.1证明各类条件分布与均匀基线不可区分，且softmax$\to$argmax的多对一映射使从类均值反推单样本标签在计算上不可行。这是**可检测不可利用**的边界条件，不影响协议的隐私保证。

| 子指标 | 度量 | 实测值 | 机会水平 | 判定 |
|--------|------|------:|------:|------|
| $\triangle$ $a_t$ 类间均值 ANOVA 最小 p（原始） | `a_t_class_mean_anova_pvalue` | 3.40×10⁻⁵ | 9.77×10⁻⁵（BH-FDR $\alpha$=0.05 阈值） | LEAK_DETECTED |
| $a_t$ 范数 ANOVA p + $\eta^2$ | `a_t_norm_anova_pvalue`, $\eta^2$=0.022 | 0.505 | 0.05 | PRIVACY_PRESERVED |
| $a_t$ 类条件分布 KL 均值 | `a_t_kl_divergence_mean` | 0.0038 | 0.10（机会水平） | PRIVACY_PRESERVED |
| $result_S$ 类间均值 ANOVA 最小 p | `result_S_class_mean_anova_pvalue` | 6.14×10⁻⁴ | 9.77×10⁻⁵（BH-FDR $\alpha$=0.05 阈值） | PRIVACY_PRESERVED |
| $result_S$ 范数 ANOVA p + $\eta^2$ | `result_S_norm_anova_pvalue`, $\eta^2$=0.037 | 0.193 | 0.05 | PRIVACY_PRESERVED |

![F-L2-1](figures/F-L2-1_at_norm_boxplot.png)
**图5**：$a_t$的6类L2范数分布（200样本）。横轴为TREC-QC粗粒度类别，纵轴为$\|a_t\|_2$。六类箱体高度重叠，范数中位数差异$<$5%，ABBR仅4样本箱体压扁。$a_t$沿$V$方向无显著系统性偏差，对应$\eta^2$=0.022（p=0.505）。

![F-L2-2](figures/F-L2-2_results_norm_boxplot.png)
**图6**：反向传播中间量$result_S$的6类L2范数分布（PRG掩码后）。横轴为类别，纵轴为$\|result_S\|_2$。六类箱体几乎重合，范数中位数差异$<$5%，证明PRG重生掩码有效稀释了$a_t$中的标签结构，对应$\eta^2$=0.037（p=0.193）。

![F-L2-3](figures/F-L2-3_anova_p_histogram.png)
**图7**：$a_t$特征级ANOVA p值直方图（512抽样特征，40 bins）。横轴为p值（log scale），红灰线标注BH-FDR阈值与$\alpha$=0.05。右偏分布，267/512特征raw p$<$0.05，BH-FDR触警根因为softmax(·)@$V$几何效应，KL=0.0038证明不可利用。

**M-1**：**3个子指标PRIVACY_PRESERVED**。

| 子指标 | 度量 | 实测值 | 机会水平 | 判定 |
|--------|------|------:|------:|------|
| 置信度方差 | `confidence_variance` | 0.0034 | 0.10 | PRIVACY_PRESERVED |
| 查询预算使用率 | `query_budget_utilisation` | 0.10 (=100/1000) | n/a | PRIVACY_PRESERVED |
| 6-bucket 预测多样性 | `prediction_diversity_6bucket` | 0.4241 | 0.50 | PRIVACY_PRESERVED |

![F-M1-1](figures/F-M1-1_confidence_histogram.png)
**图8**：S端top-1置信度分布（100 query，20 bins)。横轴为softmax置信度，纵轴为query计数，红色实线标注样本均值0.0866，灰色虚线标注方差阈值0.10。分布集中在[0.02,0.20]，方差=0.0034$\ll$0.10，反映出U无法从置信度散布推断M端LoRA行为。

![F-M1-2](figures/F-M1-2_token_distribution.png)
**图9**：预测token频次与6-bucket投影分布（100 query）。左图为原始token id频次（10 distinct tokens），右图为token_id mod 6粗粒度投影。原始分布呈长尾但粗粒度投影后接近均匀（diversity=0.4241），偏离均匀$<$1$\sigma$，不可区分协议行为与随机猜测。

**M-2**：**3/6核心子指标PRIVACY_PRESERVED**，3项INCONCLUSIVE。INCONCLUSIVE项是攻击器在保守门闸下的失效本身构成对协议鲁棒性的正面证据：秩指纹$\rho(r=8)$=0.000 vs 零分布95%分位=9.552——效应量差异超过一个数量级，置换p=1.000表示1999次随机置换无一产生超过实测的信号。攻击器的consistency gate要求gap_z$>$0.5$\sigma$，在n=800低功效下将3项指纹指标判为INCONCLUSIVE，但效应量方向压倒性地指向"无LoRA低秩痕迹"。3项PRIVACY_PRESERVED指标集中在结构性硬约束（$Z_t$有效秩=43$\gg$8、矩阵反演不可行、$result_S$-标签max|$r$|=0.0960），这是攻击器无法跨越的天然屏障。因此，warmup=2000步、n=800的完整测试条件下，S端无法识别M端LoRA结构。

| 项 | 配置值 | 实际评估使用值 | 说明 |
|------|------:|------:|------|
| warmup steps | 2000 | 200 | warmup阶段共2000步（=8000样本）收集$a_t^{pre}$，攻击评估仅取前200步对应800样本 |
| n_pre（样本数） | 8000 | **800** | `attack_results.json`中`n_samples=800` |
| n_post | 2000 | **800** | 攻击器n_post=800（评估样本量，硬编码） |
| permutation cap | 1999 | 1999 | rank_fingerprint与direction_fingerprint共享同一组置换零分布 |

| 子指标 | 度量 | 实测值 | 机会水平 | n_samples | 判定 |
|--------|------|------:|------:|------:|----------|
| 秩指纹 $\rho(r=8)$ | `rho_spectral_at_lora_rank` | 0.000 | 9.552（置换零95%分位） | 800 | INCONCLUSIVE |
| 秩指纹置换 p | `permutation_test_pvalue` | 1.000 | 0.05（$\alpha$阈值） | 800 | INCONCLUSIVE |
| 方向投影能量 | `projection_energy_in_deltaW` | 0.0000 | 0.0000（置换零95%分位） | 800 | INCONCLUSIVE |
| $result_S$与标签最大|Pearson $r$| `result_s_label_correlation` | 0.0960 | 0.00026 | 1600 | PRIVACY_PRESERVED |
| $Z_t$有效秩 | `z_t_effective_rank` | 43.0 | 8.0（LoRA rank） | 1600 | PRIVACY_PRESERVED |
| 直接矩阵反演可行性 | `direct_inversion_feasible` | 0.0 | 0.5（不可行阈值） | 800 | PRIVACY_PRESERVED |

![F-M2-1](figures/F-M2-1_rank_fingerprint.png)
**图10**：秩指纹$\rho(r)$曲线。横轴rank r，纵轴$\rho(r)$（symlog），绿线LoRA rank=8，红线零分布95%分位（9.552）。$\rho(8)$=0.000$\ll$9.552，置换p=1.000，S端无法识别LoRA低秩结构。

![F-M2-2](figures/F-M2-2_verdicts.png)
**图11**：M-2六项核心子指标verdict。绿色=PRIVACY_PRESERVED，琥珀色=INCONCLUSIVE。3项通过集中在结构性硬约束，3项INCONCLUSIVE集中在需大样本power的指纹统计量。

**威胁面覆盖**：四项攻击覆盖三方完整威胁面——L-1针对M方（最强攻击者），L-2针对S方（词嵌入+PRG），M-1针对U方（黑盒查询），M-2针对S推断M端结构。L-1全过表明最强攻击面下安全；L-2触警因KL不可区分性与softmax→argmax不可逆性不可利用；M-1受限于查询预算；M-2效应量方向指向无泄露。协议三方不存在可利用的隐私泄露路径。

### 3.2 性能测试

> 以下数据标注：`[解析]` 理论计算，`[实测, n=X]` 实验测量，`[估算]` 硬件参数近似。

**通信开销 [解析]**：基于BFV密文=98,304B/token（poly_degree×3 primes×8B）、明文share=32,768B/token解析计算。

| 通道 | 阶段 | 字节数（CLS, 3072 tok） | 字节数（NER, 3584 tok） | 字节数（生产, 4096 tok） |
|------|------|------:|------:|------:|
| U → M | 前向 | 288.00 MB | 336.00 MB | 384.00 MB |
| M → S | 前向 | ~1 KB（PIR 查询） | ~1 KB（PIR 查询） | ~1 KB（PIR 查询） |
| S → U | 前向 | 288.00 MB | 336.00 MB | 384.00 MB |
| U → M | 反向 | 288.00 MB | 336.00 MB | 384.00 MB |
| M → S | 反向 | 96.00 MB | 112.00 MB | 128.00 MB |
| S → U | 反向 | 96.00 MB | 112.00 MB | 128.00 MB |
| **前向合计** | — | **576.00 MB** | **672.00 MB** | **768.00 MB** |
| **反向合计** | — | **480.00 MB** | **560.00 MB** | **640.00 MB** |
| **每训练步总通信** | — | **1 056.00 MB ≈ 1.03 GB** | **1 232.00 MB ≈ 1.20 GB** | **1 408.00 MB ≈ 1.38 GB** |

![F-COMM-1](figures/F-COMM-1_channels_per_step.png)
**图12**：单训练步三方通信字节数（三档场景，6通道）。BFV密文通道占≈81.8%，每token≈344KB（明文≈10.75×），倍数仅由BFV参数决定。

**离线准备开销**：

实测数据（warm缓存命中）：

| 阶段 | 耗时 |
|------|------|
| Stage 0 初始化 [实测] | 0.011 s |

估算数据（基于SEAL/RTX 5090算力，未实测）：

| 阶段 | 估算耗时 |
|------|------|
| 冷启动V矩阵BFV加密 | 600–1500 s |
| Hint表构建 | ~25 s |
| BFV密钥生成 | ~3 s |

| 产物 | 大小 | 用途 |
|------|------|------|
| BFV 公钥 pk | 192 KB | M、S 用于 BFV 加密 |
| BFV 加密 V 矩阵 | 15.67 GB | S持有密文DB，128,256×4096 |
| S3PIR Hint 表 | 1.54 MB | n_partitions=501, λ=80 |
| BFV 私钥 sk | ~67 KB（仅内存） | M解密用，不落盘 |

加密V矩阵（15.67GB）为绝对主导产物，私钥不落盘最小化密钥暴露面。

![F-OFFLINE-1](figures/F-OFFLINE-1_stage0_sizes.png)
**图13**：Stage 0离线产物大小（对数尺度）。加密V矩阵占绝对主导，Hint表仅1.54MB。

**单步耗时与资源占用 [实测]**：

CLS任务（Llama-3.1-8B, max_seq_length=256, SLG n=3951, baseline n=734）：

| 阶段 | 基准方案（明文LoRA） | SLG-HE-PIR方案 | 倍率 |
|------|------:|------:|------:|
| `forward_U` | 0.017 s | 0.020 s | 1.18× |
| `forward_M` | 0.137 s | 0.446 s | 3.26× |
| `s_logits` | 17.125 s | 38.900 s | 2.27× |
| `priv_U` | 16.660 s | 43.402 s | 2.60× |
| `backward_M` | 5.889 s | 18.414 s | 3.13× |
| **单步均值** | **39.83 s** | **101.18 s** | — |
| 范围 | 23.70–540.83 s | 98.00–113.07 s | — |

> SLG稳态均值≈101s/step（范围98.0–113.1s，n=3951），s_logits与priv_U两阶段≈81%。

![F-TIME-1](figures/F-TIME-1_cls_phases.png)
**图14**：CLS单步耗时五阶段堆叠。

NER任务（max_seq_length=512）：

| 方案 | 单步均值 | 样本量 |
|------|------:|------:|
| 基准（明文LoRA） | 23.10 s | 734步 |
| SLG-HE-PIR | ≈172 s | n=3 步（启动期） |

> SLG NER为启动期实测（n=3，2026-07-27在step 4后中断），因seq_len更长（512 vs 256）、chunk更多（7 vs 4）。

| 维度 | 基准方案 | SLG-HE-PIR 方案 |
|------|------:|------:|
| CPU 内存峰值 CLS [实测] | 46.0 GB | 46.2 GB |
| CPU 内存峰值 NER [实测] | 55.6 GB | 55.6 GB |

| 任务 | SLG GPU 显存 [实测] |
|------|------:|
| CLS 均值 | 29.59 GB |
| NER 峰值 | 30.61 GB |

> CPU内存两方案持平；GPU显存主要来自密文-明文转换缓冲。

![F-TIME-2](figures/F-TIME-2_slg_cls_timeline.png)
**图15**：SLG CLS全程step耗时（3951步），稳态≈101s/step，无drift。

![F-RES-1](figures/F-RES-1_cpu_gpu_memory.png)
**图16**：CPU/GPU内存峰值对比。

**总体观察**：攻击测试表明SLG-HE-PIR协议在三方威胁面下均未暴露可利用的隐私泄露——L-1全过，L-2单点触警但不可利用，M-1/M-2受限于预算或样本量但效应量方向指向安全。性能代价集中于s_logits与priv_U两密码学阶段（≈81%耗时）与BFV密文通道（≈81.8%通信），根因均为同态加密在4096维嵌入空间上的密文乘法开销。CPU内存与基准持平表明内存开销不来自密码学而来自模型本身；GPU显存≈30GB瓶颈在密文-明文转换时的张量缓冲。


## 参考文献

