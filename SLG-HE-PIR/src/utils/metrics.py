"""Metrics collection utilities for SLG-HE-PIR training."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["MetricsCollector", "MetricsSummary"]


@dataclass
class MetricsSummary:
    """Aggregated metrics for a named measurement series."""

    name: str
    values: List[float] = field(default_factory=list)
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    _min: float = float("inf")
    _max: float = float("-inf")

    def add(self, value: float) -> None:
        self.values.append(value)
        self.count += 1
        self.sum += value
        self.sum_sq += value * value
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        variance = (self.sum_sq - self.sum * self.sum / self.count) / (self.count - 1)
        return variance**0.5

    @property
    def min(self) -> float:
        return self._min if self.count else 0.0

    @property
    def max(self) -> float:
        return self._max if self.count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }


class MetricsCollector:
    """Thread-safe metrics accumulator for training runs.

    Tracks counters, timers, and arbitrary named metric series.

    Example
    -------
    >>> mc = MetricsCollector()
    >>> mc.increment("steps")
    >>> mc.start_timer("step")
    >>> mc.record("loss", 0.23)
    >>> mc.stop_timer("step")
    >>> mc.summarize()
    """

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, float] = {}
        self._metrics: Dict[str, MetricsSummary] = defaultdict(
            lambda: MetricsSummary(name="")
        )
        self._epoch_starts: Dict[str, float] = {}
        self._epoch_metrics: Dict[str, List[float]] = defaultdict(list)

    # ── Counters ──────────────────────────────────────────────────────────────
    def increment(self, name: str, delta: int = 1) -> None:
        self._counters[name] += delta

    def get_counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    # ── Timers ────────────────────────────────────────────────────────────────
    def start_timer(self, name: str) -> None:
        self._timers[name] = time.time()

    def stop_timer(self, name: str) -> Optional[float]:
        if name not in self._timers:
            return None
        elapsed = time.time() - self._timers[name]
        self.record(f"{name}_duration", elapsed)
        del self._timers[name]
        return elapsed

    def get_timer(self, name: str) -> Optional[float]:
        started = self._timers.get(name)
        if started is None:
            return None
        return time.time() - started

    # ── Named metric series ───────────────────────────────────────────────────
    def record(self, name: str, value: float) -> None:
        ms = self._metrics[name]
        if not ms.name:
            ms.name = name
        ms.add(value)
        self._epoch_metrics[name].append(value)

    def get(self, name: str) -> List[float]:
        return list(self._metrics.get(name, MetricsSummary(name=name)).values)

    def get_summary(self, name: str) -> MetricsSummary:
        return self._metrics.get(name, MetricsSummary(name=name))

    # ── Epoch tracking ────────────────────────────────────────────────────────
    def start_epoch(self, epoch: int) -> None:
        self._epoch_starts["epoch"] = time.time()
        self._epoch_starts["epoch_id"] = epoch

    def stop_epoch(self, epoch: int) -> Optional[float]:
        key = "epoch"
        if key not in self._epoch_starts:
            return None
        elapsed = time.time() - self._epoch_starts[key]
        self.record(f"epoch_{epoch}_duration", elapsed)
        return elapsed

    def epoch_metrics(self, name: str) -> List[float]:
        return list(self._epoch_metrics.get(name, []))

    def clear_epoch_metrics(self) -> None:
        self._epoch_metrics.clear()

    # ── Serialization ─────────────────────────────────────────────────────────
    def summarize(self) -> Dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "metrics": {name: ms.to_dict() for name, ms in self._metrics.items()},
        }

    def reset(self) -> None:
        self._counters.clear()
        self._timers.clear()
        self._metrics.clear()
        self._epoch_metrics.clear()
        self._epoch_starts.clear()
