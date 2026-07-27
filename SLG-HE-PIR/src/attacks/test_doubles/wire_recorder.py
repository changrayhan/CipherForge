"""WireRecord — 攻击测试字节流录制格式。

每条跨进程边界的消息都记录为一个 WireRecord，落地到：
- JSONL 索引文件（WireRecord 元数据）
- .bin 文件（大字段 payload）
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Any


# ─── WireRecord 数据类 ────────────────────────────────────────────────────────

@dataclass
class WireRecord:
    """单条总线消息记录。"""
    timestamp: float          # time.time() 秒级时间戳
    peer: Literal["U", "M", "S"]   # 消息所属的参与方
    tag: str                 # "ct_list" | "s_share" | "H_M" | "g_H" | ...
    step: int                # 训练步数
    direction: Literal["in", "out"] # 进入或离开该 party
    payload_sha256: str      # payload_bytes 的 SHA-256（完整性校验）
    payload_path: str        # 大字段 .bin 文件路径，空串表示 payload 在内存中未持久化

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> WireRecord:
        d = json.loads(s)
        return WireRecord(**d)

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


# ─── WireRecorder ────────────────────────────────────────────────────────────

class WireRecorder:
    """录制总线消息到 .bin + JSONL 文件。"""

    def __init__(
        self,
        output_dir: Path | str,
        peer: Literal["U", "M", "S"],
        max_file_size_mb: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.peer = peer
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.jsonl_path = self.output_dir / f"wire_{peer}.jsonl"
        self.bin_path = self.output_dir / f"wire_{peer}.bin"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._bin_fp: Optional[object] = None   # 二进制写句柄
        self._bin_offset = 0                     # 当前 .bin 写入偏移量
        self._jsonl_fp: Optional[object] = None # JSONL 写句柄

    # ── 上下文管理器 ───────────────────────────────────────────────────────

    def __enter__(self) -> WireRecorder:
        self._bin_fp = open(self.bin_path, "ab")
        self._jsonl_fp = open(self.jsonl_path, "a", encoding="utf-8")
        return self

    def __exit__(self, *_) -> None:
        if self._bin_fp is not None:
            self._bin_fp.close()
        if self._jsonl_fp is not None:
            self._jsonl_fp.close()
        self._bin_fp = None
        self._jsonl_fp = None

    # ── 录制接口 ───────────────────────────────────────────────────────────

    def record(
        self,
        tag: str,
        step: int,
        direction: Literal["in", "out"],
        payload: bytes | str | Any,
    ) -> WireRecord:
        """
        将一条消息写入录制流。

        Args:
            tag:        消息类型标签
            step:       训练步数
            direction:  "in" = 接收， "out" = 发送
            payload:    原始字节或可 pickle 的 Python 对象

        Returns:
            WireRecord 元数据（payload 已写入 .bin，record 指向其偏移）
        """
        if self._bin_fp is None or self._jsonl_fp is None:
            raise RuntimeError("WireRecorder must be used as a context manager")

        # 序列化 payload
        if isinstance(payload, bytes):
            data_bytes = payload
        elif isinstance(payload, str):
            data_bytes = payload.encode("utf-8")
        else:
            import pickle
            data_bytes = pickle.dumps(payload)

        sha256 = WireRecord.compute_sha256(data_bytes)

        # 写入 .bin（滚动：如果当前文件超过 max_file_size，开新文件）
        offset = self._write_bin_rolling(data_bytes)

        record = WireRecord(
            timestamp=time.time(),
            peer=self.peer,
            tag=tag,
            step=step,
            direction=direction,
            payload_sha256=sha256,
            payload_path=str(self.bin_path),
        )

        self._jsonl_fp.write(record.to_json() + "\n")
        self._jsonl_fp.flush()
        return record

    def _write_bin_rolling(self, data: bytes) -> int:
        """写入 .bin，返回写入偏移量。文件过大时截断并从头写。"""
        if self._bin_fp is None:
            raise RuntimeError("Binary file not open")

        # 检查是否需要滚动
        if self._bin_offset + len(data) > self.max_file_size:
            self._bin_fp.close()
            self._bin_path.unlink(missing_ok=True)
            self._bin_fp = open(self.bin_path, "ab")
            self._bin_offset = 0

        offset = self._bin_offset
        self._bin_fp.write(len(data).to_bytes(8, "little"))  # 长度头
        self._bin_fp.write(data)
        self._bin_fp.flush()
        self._bin_offset += 8 + len(data)
        return offset

    # ── 读取接口（用于攻击脚本） ───────────────────────────────────────────

    @classmethod
    def load_wire_records(cls, jsonl_path: Path | str) -> list[WireRecord]:
        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(WireRecord.from_json(line))
        return records

    @classmethod
    def load_payload_at(cls, bin_path: Path | str, offset: int) -> bytes:
        """从 .bin 文件指定偏移读取一条 payload。"""
        with open(bin_path, "rb") as f:
            f.seek(offset)
            length = int.from_bytes(f.read(8), "little")
            return f.read(length)
