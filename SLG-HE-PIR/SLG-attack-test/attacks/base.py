"""Base attack interface — all attack modules inherit from BaseAttack."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from evaluation.metrics import AttackVerdict


class BaseAttack(ABC):
    """Abstract base class for all attack modules in the SLG-HE-PIR suite.

    Subclasses must implement:
      - ``run()`` → List[AttackVerdict]
      - ``ATTACK_ID`` (str): unique identifier, e.g. "L1A", "L3A", "P6"
      - ``ATTACK_NAME`` (str): human-readable name
      - ``TARGET`` (str): what the attack targets
      - ``THREAT_MODEL`` (str): threat model description

    Optional lifecycle methods:
      - ``prepare(protocol, cfg)`` — called before training (data collection setup)
      - ``collect(protocol, batch, result)`` — called after each step
      - ``finalise()`` — called after all steps (post-processing)
    """

    ATTACK_ID: str = "BASE"
    ATTACK_NAME: str = "Base Attack"
    TARGET: str = ""
    THREAT_MODEL: str = ""

    def __init__(self, **kwargs):
        self._verdicts: List[AttackVerdict] = []
        self._prepared = False
        # Allow subclasses to receive output_dir via kwargs without crashing
        self.output_dir: Optional[str] = kwargs.get("output_dir")

    @abstractmethod
    def run(self) -> List[AttackVerdict]:
        """Run the attack and return a list of verdicts."""
        ...

    def prepare(self, protocol: Any, cfg: Any) -> None:
        """Prepare the attack (e.g. set up hooks, configure protocol)."""
        self._prepared = True

    def collect(self, step_result: Any) -> None:
        """Collect data from one step result (called after each step)."""
        pass

    def finalise(self) -> List[AttackVerdict]:
        """Post-processing after all steps collected. Returns verdicts."""
        return self.run()

    @property
    def verdicts(self) -> List[AttackVerdict]:
        return self._verdicts

    def add_verdict(self, verdict: AttackVerdict) -> None:
        self._verdicts.append(verdict)

    # ── 数据加载接口 ─────────────────────────────────────────────────────────

    def load_attack_data(self) -> dict:
        """
        从 `output_dir/attack_data.json` 加载攻击数据字典。

        由统一入口在 run_attack_suite.py 中调用，每个 attack 脚本
        可通过 `self.load_attack_data()` 获取之前收集的数据。

        Returns:
            dict，键如 "g_accum", "labels", "H_M", "parity_bytes", "prg_outputs" 等
        """
        # 默认从当前工作目录或上一级目录查找
        candidates = [
            Path("attack_data.json"),
            Path("SLG-attack-test/results/attack_data.json"),
            Path("../attack_data.json"),
            Path("../SLG-attack-test/results/attack_data.json"),
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return {}

    def save_verdicts(self, output_path: Path | str) -> None:
        """将 verdicts 列表序列化为 JSON。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([v.to_dict() for v in self._verdicts], f, ensure_ascii=False, indent=2)

