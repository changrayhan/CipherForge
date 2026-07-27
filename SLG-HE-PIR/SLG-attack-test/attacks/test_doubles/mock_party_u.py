"""MockPartyU — User 方测试替身。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from .mock_party import MockParty, MockBehavior


class MockPartyU(MockParty):
    """User 方（U）测试替身。

    物理可观测（§0.1）：
        input_ids, output_ids, H_U, ct_list, prg_seed, a_t
    物理不可观测：
        sk_M, M 后 16 层权重, LoRA, V 矩阵
    """

    NAME: Literal["U"] = "U"

    def __init__(
        self,
        output_dir: Path | str,
        behavior: MockBehavior = "honest",
        logits回流: bool = False,
    ):
        super().__init__(output_dir=output_dir, behavior=behavior)
        self.logits回流 = logits回流

    def _on_receive_honest(self, payload: dict) -> None:
        """
        诚实 U 的处理逻辑：
        - 接收 M 发来的 a_t（PIR 响应恢复后的 softmax 概率向量）
        - 不解密、不持有 sk_M
        """
        # U 在诚实模式下只需要记录，不主动处理
        pass

    def send_embed_and_ct(
        self,
        input_ids: list[int],
        output_ids: list[int],
        prg_seed: bytes,
        step: int,
    ) -> dict:
        """
        U 发送给 M 的消息构建。

        Returns:
            符合 ALLOWED_KEYS_U 的 payload
        """
        payload = {
            "step": step,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "prg_seed": prg_seed,
        }
        return self.sends(payload, step=step)
