"""AttackTestBus — 用 multiprocessing.Queue 替代 InProcessBus 的测试总线。

设计原则：
- 用 spawn 模式创建独立子进程（避免 CUDA 缓存冲突）
- 每条总线（U↔M, M↔S, U↔S）独立 Queue
- 支持各 party 的 WireRecorder 录制
- 可嵌入 MaliciousBus 进行中间人攻击
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Literal, Optional

from .wire_recorder import WireRecorder


# ─── AttackTestBus ─────────────────────────────────────────────────────────────

class AttackTestBus:
    """
    多进程测试总线，替代 InProcessBus。

    总线拓扑：
        U ←──────────────→ M
        │                 │
        └────→ S ←───────┘

    三条独立 Queue：
        - u_m_queue: U ↔ M 通信
        - m_s_queue: M ↔ S 通信
        - u_s_queue: U ↔ S 通信（PRG seed 专用）

    可选嵌入 MaliciousBus 进行中间人攻击。

    设计要点：
        - 使用 spawn 模式（避免 fork 继承 CUDA 缓存）
        - 所有 GPU tensor 在主进程处理，Queue 仅传 bytes/numpy
        - WireRecorder 录制每条总线消息
    """

    def __init__(
        self,
        output_dir: Path | str,
        embed_malicious: bool = False,
        malicious_mode: str = "eavesdrop",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ctx = mp.get_context("spawn")

        # 三条独立 Queue
        self.u_m_queue: mp.Queue = self._ctx.Queue()
        self.m_s_queue: mp.Queue = self._ctx.Queue()
        self.u_s_queue: mp.Queue = self._ctx.Queue()

        # 三条 WireRecorder
        self._recorders = {
            "U": WireRecorder(self.output_dir, "U"),
            "M": WireRecorder(self.output_dir, "M"),
            "S": WireRecorder(self.output_dir, "S"),
            "BUS": WireRecorder(self.output_dir, "BUS"),
        }

        # MaliciousBus（可选嵌入）
        self._malicious = None
        if embed_malicious:
            from .malicious_bus import MaliciousBus
            self._malicious = MaliciousBus(
                output_dir=self.output_dir / "malicious",
                mode=malicious_mode,
            )

    # ── 上下文管理器 ───────────────────────────────────────────────────────

    def __enter__(self) -> AttackTestBus:
        for rec in self._recorders.values():
            rec.__enter__()
        if self._malicious is not None:
            self._malicious.__enter__()
        return self

    def __exit__(self, *_) -> None:
        if self._malicious is not None:
            self._malicious.__exit__(*_)
        for rec in self._recorders.values():
            rec.__exit__(*_)

    # ── 发送接口 ──────────────────────────────────────────────────────────

    def send_u_to_m(self, payload: bytes) -> None:
        """U → M 发送密文 ct_list（payload 是 pickle bytes）。"""
        self.u_m_queue.put(payload)
        self._recorders["BUS"].record(
            tag="U→M", step=-1, direction="out", payload=payload
        )

    def send_m_to_u(self, payload: bytes) -> None:
        """M → U 发送（通常为空，U 不需要从 M 接收）"""
        self.u_m_queue.put(payload)

    def send_m_to_s(self, payload: bytes) -> None:
        """M → S 发送 H_M（GPU tensor → numpy → bytes）。"""
        self.m_s_queue.put(payload)
        self._recorders["BUS"].record(
            tag="M→S", step=-1, direction="out", payload=payload
        )

    def send_s_to_m(self, payload: bytes) -> None:
        """S → M 发送 s_share。"""
        self.m_s_queue.put(payload)
        self._recorders["BUS"].record(
            tag="S→M", step=-1, direction="out", payload=payload
        )

    def send_u_to_s(self, payload: bytes) -> None:
        """U → S 发送 PRG seed（仅 U/S 共享）。"""
        self.u_s_queue.put(payload)
        self._recorders["BUS"].record(
            tag="U→S", step=-1, direction="out", payload=payload
        )

    def send_s_to_u(self, payload: bytes) -> None:
        """S → U 发送 PIR 响应。"""
        self.u_s_queue.put(payload)
        self._recorders["BUS"].record(
            tag="S→U", step=-1, direction="out", payload=payload
        )

    # ── 接收接口 ──────────────────────────────────────────────────────────

    def recv_u_from_m(self, timeout: float = 10.0) -> bytes:
        return self.u_m_queue.get(timeout=timeout)

    def recv_m_from_u(self, timeout: float = 10.0) -> bytes:
        return self.u_m_queue.get(timeout=timeout)

    def recv_m_from_s(self, timeout: float = 10.0) -> bytes:
        return self.m_s_queue.get(timeout=timeout)

    def recv_s_from_m(self, timeout: float = 10.0) -> bytes:
        return self.m_s_queue.get(timeout=timeout)

    def recv_u_from_s(self, timeout: float = 10.0) -> bytes:
        return self.u_s_queue.get(timeout=timeout)

    def recv_s_from_u(self, timeout: float = 10.0) -> bytes:
        return self.u_s_queue.get(timeout=timeout)

    # ── 中间人接口 ────────────────────────────────────────────────────────

    def mitm_u_m(self, payload: bytes) -> bytes:
        """U↔M 链路中间人。"""
        if self._malicious is not None:
            return self._malicious.intercept("U", "M", payload)
        return payload

    def mitm_m_s(self, payload: bytes) -> bytes:
        """M↔S 链路中间人。"""
        if self._malicious is not None:
            return self._malicious.intercept("M", "S", payload)
        return payload

    def mitm_u_s(self, payload: bytes) -> bytes:
        """U↔S 链路中间人（P-6 攻击目标链路）。"""
        if self._malicious is not None:
            return self._malicious.intercept("U", "S", payload)
        return payload
