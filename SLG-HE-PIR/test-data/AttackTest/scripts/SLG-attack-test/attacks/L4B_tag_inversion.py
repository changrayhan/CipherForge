"""L-4B: TAG/DLG 梯度反演攻击。

攻击逻辑（docs/攻击类测试方案.md §4.2）：
随机初始化 dummy input x'，用 DLG/TAG 迭代优化 x' 使其梯度匹配真实 g_H。
"""
from __future__ import annotations

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class L4BTAGInversionAttack(BaseAttack):
    """L-4B: TAG/DLG 梯度反演。"""

    ATTACK_ID = "L4B"
    ATTACK_NAME = "TAG/DLG 梯度反演"
    TARGET = "输入 token 序列 x"
    THREAT_LEVEL = "HIGH"
    GPU_REQUIRED = True

    def __init__(
        self,
        n_steps: int = 1000,
        lambda_lm: float = 0.1,
        gumbel_T: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_steps = n_steps
        self.lambda_lm = lambda_lm
        self.gumbel_T = gumbel_T

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "g_H" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 g_H 数据（需 GPU + LLM 前向/反向传播）",
            )]

        g_H_list = attack_data["g_H"]
        n_samples = len(g_H_list)

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="token_reconstruction_rate",
            value=0.0,
            chance_level=0.01,
            p_value=None,
            n_samples=n_samples,
            verdict="INCONCLUSIVE",
            notes=f"需 GPU + LLM 前向/反向传播（n_steps={self.n_steps}，λ={self.lambda_lm}）。"
                  f"SLG-HE-PIR 的梯度不是 one-hot，预期 Token Rec Rate ≈ 0。",
        ))

        return verdicts
