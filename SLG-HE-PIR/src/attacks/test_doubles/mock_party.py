"""MockParty — 独立进程测试替身基类。

每个 MockParty 运行在独立的 spawn 子进程中，严格按 ALLOWED_KEYS 过滤
进出的 payload，任何越界字段触发 AssertionError。
"""
from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

from .allowed_keys import validate_allowed_keys, ALLOWED_KEYS
from .wire_recorder import WireRecorder


# ─── MockParty 行为类型 ────────────────────────────────────────────────────────

MockBehavior = Literal["honest", "eavesdrop", "tamper", "replay", "impersonate"]


# ─── MockParty 基类 ───────────────────────────────────────────────────────────

class MockParty(ABC):
    """
    测试替身基类。

    设计原则（docs/攻击类测试方案.md §0.1）：
    - 每个 MockParty 严格持有该方应有的字段（ALLOWED_KEYS）
    - 任何越界字段在 receives() 入口处触发 AssertionError
    - 默认行为 = "honest"（按真实协议逻辑处理）
    - 其他行为用于攻击模拟
    """

    NAME: Literal["U", "M", "S"]  # 子类必须设置
    _mp_ctx: mp.context.SpawnContext | None = None

    def __init__(
        self,
        output_dir: Path | str,
        behavior: MockBehavior = "honest",
        log_wire: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.behavior = behavior
        self.log_wire = log_wire

        # WireRecorder 仅在 spawn 子进程内使用
        self._recorder: WireRecorder | None = None
        if self.log_wire:
            self._recorder = WireRecorder(self.output_dir, self.NAME)

    # ── 入口校验 ───────────────────────────────────────────────────────────

    def receives(self, payload: dict) -> None:
        """
        物理隔离视角：过滤掉不该看的数据。

        入口处自动校验 ALLOWED_KEYS，越界字段触发 AssertionError。
        """
        peer = self.NAME  # type: ignore[attr-defined]
        received_keys = set(payload.keys())

        # 强制校验：任何越界字段立即报错
        validate_allowed_keys(peer, received_keys)

        # 录制到日志
        if self._recorder is not None:
            self._recorder.record(
                tag="receive",
                step=payload.get("step", -1),
                direction="in",
                payload=payload,
            )

        # 行为分发
        if self.behavior == "honest":
            self._on_receive_honest(payload)
        elif self.behavior == "eavesdrop":
            self._on_receive_eavesdrop(payload)
        elif self.behavior == "tamper":
            self._on_receive_tamper(payload)
        elif self.behavior == "replay":
            self._on_receive_replay(payload)
        elif self.behavior == "impersonate":
            self._on_receive_impersonate(payload)

    def sends(self, payload: dict, step: int = -1) -> dict:
        """
        发送前校验：确保发出的字段在自身 ALLOWED_KEYS 内。
        """
        peer = self.NAME  # type: ignore[attr-defined]
        outgoing_keys = set(payload.keys())
        allowed = ALLOWED_KEYS.get(peer, frozenset())
        forbidden = outgoing_keys - allowed
        assert not forbidden, (
            f"[{peer}] tried to send forbidden fields: {forbidden}"
        )

        if self._recorder is not None:
            self._recorder.record(
                tag="send",
                step=step,
                direction="out",
                payload=payload,
            )
        return payload

    # ── 行为钩子（子类实现） ───────────────────────────────────────────────

    @abstractmethod
    def _on_receive_honest(self, payload: dict) -> None:
        """诚实行为：按协议正常处理。子类必须实现。"""
        raise NotImplementedError

    def _on_receive_eavesdrop(self, payload: dict) -> None:
        """窃听行为：记录但不修改。"""
        pass

    def _on_receive_tamper(self, payload: dict) -> None:
        """篡改行为：子类可覆盖以实现具体篡改逻辑。"""
        pass

    def _on_receive_replay(self, payload: dict) -> None:
        """重放行为：子类可覆盖以实现具体重放逻辑。"""
        pass

    def _on_receive_impersonate(self, payload: dict) -> None:
        """冒充行为：子类可覆盖以实现具体冒充逻辑。"""
        pass


# ─── 工厂函数 ────────────────────────────────────────────────────────────────

def make_mock_party(
    party_name: Literal["U", "M", "S"],
    output_dir: Path | str,
    behavior: MockBehavior = "honest",
) -> MockParty:
    """
    工厂函数：按 party_name 返回对应 MockParty 实例。

    Args:
        party_name: "U" | "M" | "S"
        output_dir: WireRecorder 输出目录
        behavior:  "honest" | "eavesdrop" | "tamper" | "replay" | "impersonate"

    Returns:
        MockPartyU | MockPartyM | MockPartyS 实例
    """
    if party_name == "U":
        from .mock_party_u import MockPartyU
        return MockPartyU(output_dir=output_dir, behavior=behavior)
    elif party_name == "M":
        from .mock_party_m import MockPartyM
        return MockPartyM(output_dir=output_dir, behavior=behavior)
    elif party_name == "S":
        from .mock_party_s import MockPartyS
        return MockPartyS(output_dir=output_dir, behavior=behavior)
    else:
        raise ValueError(f"Unknown party: {party_name}")
