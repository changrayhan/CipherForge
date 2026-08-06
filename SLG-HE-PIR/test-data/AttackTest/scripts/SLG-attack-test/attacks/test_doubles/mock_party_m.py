"""MockPartyM — Model 方测试替身。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from .mock_party import MockParty, MockBehavior


class MockPartyM(MockParty):
    """Model 方（M）测试替身。

    物理可观测（§0.1）：
        H_U（明文 GPU tensor），H_M, g_H, masked_arr, s_share, sk_M, ct_list
    物理不可观测：
        prg_seed（仅 U/S 共享）, r_t 单独值, V 矩阵, a_t 单独值
    """

    NAME: Literal["M"] = "M"

    def __init__(
        self,
        output_dir: Path | str,
        behavior: MockBehavior = "honest",
        bfv_sk_pem: str | None = None,
    ):
        super().__init__(output_dir=output_dir, behavior=behavior)
        self.bfv_sk_pem = bfv_sk_pem
        # 收集的 g_accum（用于 L-1B / L-2 / L-4 等攻击）
        self.collected_g_accum: list = []
        self.collected_H_M: list = []
        self.collected_labels: list = []

    def _on_receive_honest(self, payload: dict) -> None:
        """
        诚实 M 的处理逻辑：
        - 接收 U 发来的 ct_list（已加 mask 的密文）
        - 接收 S 发来的 s_share = scale·a_t - r_t
        - 解密 ct_list 得到 masked_arr = -V_y·scale + r_t
        - 组合：g_H = (masked_arr + s_share) / scale ≈ a_t - V_y
        - 注入 autograd，执行反向传播 + AdamW 更新 LoRA
        """
        # 记录 g_accum（用于 L-1B 攻击分析）
        if "g_H" in payload or "g_accum" in payload:
            g_val = payload.get("g_H") or payload.get("g_accum")
            self.collected_g_accum.append(g_val)

        if "H_M" in payload:
            self.collected_H_M.append(payload["H_M"])

        if "labels" in payload:
            self.collected_labels.extend(payload["labels"])

    def get_attack_data(self) -> dict:
        """导出已收集的攻击数据（供 L-1B/L-2/L-4 使用）。"""
        return {
            "g_accum": self.collected_g_accum,
            "H_M": self.collected_H_M,
            "labels": self.collected_labels,
        }
