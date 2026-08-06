"""MockPartyS — Server 方测试替身。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from .mock_party import MockParty, MockBehavior


class MockPartyS(MockParty):
    """Server 方（S）测试替身。

    物理可观测（§0.1）：
        H_M, V, logits, a_t, s_share, prg_seed,
        parity_real_bytes（PIR 响应字节）,
        permutation_bit, real_indices
    物理不可观测：
        sk_M, -V_y 密文原内容, pk_M 之外的金钥
    """

    NAME: Literal["S"] = "S"

    def __init__(
        self,
        output_dir: Path | str,
        behavior: MockBehavior = "honest",
        V_matrix_path: str | None = None,
    ):
        super().__init__(output_dir=output_dir, behavior=behavior)
        self.V_matrix_path = V_matrix_path
        # 收集的 PIR query/response（用于 L-3B 攻击）
        self.collected_queries: list = []
        self.collected_argmax_preds: list = []
        self.collected_labels: list = []

    def _on_receive_honest(self, payload: dict) -> None:
        """
        诚实 S 的处理逻辑：
        - 接收 M 发来的 H_M（GPU tensor）
        - 自算 logits = H_M @ V^T
        - 自算 a_t = softmax(logits) @ V
        - 自算 s_share = scale·a_t - r_t，发回 M
        - PIR 查询路径（Design-2）：直接 mmap 读 y_t 行
        """
        if "H_M" in payload:
            H_M = payload["H_M"]
            # S 自算 argmax（用于 L-3A / L-3B）
            # 实际 logits 计算需要 V 矩阵，这里仅记录元数据
            self.collected_queries.append({
                "step": payload.get("step", -1),
                "has_H_M": H_M is not None,
            })

        if "argmax_pred" in payload:
            self.collected_argmax_preds.append(payload["argmax_pred"])

        if "labels" in payload:
            self.collected_labels.extend(payload["labels"])

    def get_attack_data(self) -> dict:
        """导出已收集的攻击数据（供 L-3A/L-3B 使用）。"""
        return {
            "queries": self.collected_queries,
            "argmax_preds": self.collected_argmax_preds,
            "labels": self.collected_labels,
        }
