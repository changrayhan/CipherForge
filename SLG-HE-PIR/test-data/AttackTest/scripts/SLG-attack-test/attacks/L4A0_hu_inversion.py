"""L-4A-0: H_U Smashed-Data Inversion（最强路径）。

攻击逻辑（docs/攻击类测试方案.md §4.2）：
M 物理可观测 H_U（前向 U→M 明文 GPU 张量），比 H_M 更接近 token embedding。
在公开语料上预训练 inverter 网络，从 H_U 反演 token 序列。

隐私边界：H_U 经 dχ 加噪（M 只看到加噪后结果），
攻击者反演精度受 dχ EP 预算限制。
"""
from __future__ import annotations

from attacks.base import BaseAttack
from evaluation.metrics import AttackVerdict


class L4A0HUInversionAttack(BaseAttack):
    """L-4A-0: H_U Smashed-Data Inversion。"""

    ATTACK_ID = "L4A0"
    ATTACK_NAME = "H_U Smashed-Data Inversion"
    TARGET = "输入 token 序列 x"
    THREAT_LEVEL = "CRITICAL"
    GPU_REQUIRED = True

    def __init__(
        self,
        inversion_model_type: str = "transformer_6L_4096H",
        pretrain_data: str = "wikitext-103",
        pretrain_epochs: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = inversion_model_type
        self.pretrain_data = pretrain_data
        self.pretrain_epochs = pretrain_epochs

    def run(self) -> list:
        verdicts = []
        attack_data = self.load_attack_data()

        if "H_U" not in attack_data:
            return [AttackVerdict(
                attack_id=self.ATTACK_ID,
                metric="availability",
                value=0.0,
                chance_level=0.0,
                p_value=None,
                n_samples=0,
                verdict="INCONCLUSIVE",
                notes="缺少 H_U 数据（需 GPU 训练 inverter）",
            )]

        H_U_list = attack_data["H_U"]
        n_samples = len(H_U_list)

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            metric="token_reconstruction_rate",
            value=0.0,
            chance_level=0.01,
            p_value=None,
            n_samples=n_samples,
            verdict="INCONCLUSIVE",
            notes=f"需 GPU 训练 inverter ({self.model_type})，"
                  f"在 {self.pretrain_data} 上预训练 {self.pretrain_epochs} epochs。"
                  f"预期 Token Rec Rate > L-4A（5-30%），因 H_U 离 token embedding 更近。",
        ))

        return verdicts
