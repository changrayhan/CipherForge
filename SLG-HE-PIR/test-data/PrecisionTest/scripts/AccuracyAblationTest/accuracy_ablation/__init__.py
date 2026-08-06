"""AccuracyAblationTest — 量化对照实验子包。

通过在已训练的 LoRA adapter logits 上注入 SLG 协议中各层精度损失，
精确量化 SLG-HE-PIR 相比明文 Baseline 的精度梯度来源。

变体定义：
  Q0   无量化, 7-target
  Q0'  无量化, 2-target
  Q1   lm_head 量化 + H_M forward 量化 (V 矩阵 fixed-point 量化税)
  Q2'  Q1 + 全 token g_H 量化 (无协议约束对照)
  Q2   Q1 + gold-token-only 全 token g_H 量化 (协议约束 + g_H 量化税)
  Q3   Q2 + g_H 转 bf16 的 round-to-nearest 量化 (bf16 转换税)
"""

__version__ = "0.1.0"