"""
Message-bus abstraction for SLG-HE-PIR v2.0 heterogeneous architecture.

Provides a single transport interface that both the active heterogeneous
runtime (in-process) and the legacy three-process IPC stub (queue-based) plug
into, so the protocol layer never has to know which transport it is using.

Two concrete implementations:

  * ``InProcessBus`` — used by ``HeterogeneousProtocol``. Same Python process
    as the GPU Fusion driver; communication is just dict lookup with a
    Condition variable for blocking receives.

  * ``QueueBus`` — used by ``LegacyIPCStub``. Underlying transport is
    ``multiprocessing.Queue``. Stale-reply filtering by ``(peer, step)`` is
    done by ``recv`` itself, mirroring what the legacy IPCProtocol used to
    do via ``_recv_with_drain``.

The bus never enforces the 10-message protocol surface — that's the protocol
class's job. The bus only guarantees:

  * in-order delivery (per peer) within a single step
  * stale-reply drain (matches by ``step``)
  * per-peer FIFO
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _Envelope:
    """Single message envelope passed over a bus."""
    peer: str
    tag: str
    payload: Any
    step: int

    def matches(self, expected_tag: str, expected_step: int) -> bool:
        return self.tag == expected_tag and self.step == expected_step


class MessageBus(Protocol):
    """The single interface every protocol-class depends on.

    Subclasses must implement :meth:`send` and :meth:`recv`. The protocol
    layer uses ``peer`` to route (e.g. ``"U"``, ``"M"``, ``"S"``); ``tag`` is
    the wire-level message label (e.g. ``"FORWARD"``, ``"H_U"``); ``step`` is
    the global training step (used for stale-reply drain).
    """

    def send(self, peer: str, tag: str, payload: Any, step: int) -> None: ...

    def recv(
        self,
        peer: str,
        tag: str,
        step: int,
        timeout: float = 120.0,
    ) -> Optional[Any]: ...

    def close(self) -> None: ...


# =============================================================================
#  In-process bus
# =============================================================================
class InProcessBus:
    """Single-process bus using a ``dict`` + ``Condition``.

    Used by ``HeterogeneousProtocol``. The GPU Fusion driver writes messages
    via :meth:`send` (synchronous, no copy for Python objects) and reads via
    :meth:`recv` (blocks on a Condition variable until the matching
    ``(peer, tag, step)`` arrives).

    Stale-reply handling: any message with ``step < expected_step`` for the
    same peer is drained silently. Messages with ``step > expected_step`` are
    stashed for future receives.
    """

    def __init__(self) -> None:
        # Map (peer, step) -> list of envelopes (tag, payload). Multiple
        # envelopes with the same (peer, step) are kept in arrival order so
        # that successive recv() calls can drain them.
        self._pending: Dict[Tuple[str, int], list] = {}
        self._cv = threading.Condition(self._pending_lock())
        self._closed = False

    class _pending_lock:
        """Trivial lockable wrapper that supports ``Condition``'s context API."""

        def __init__(self) -> None:
            self._lock = threading.Lock()

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *exc):
            self._lock.release()

        def acquire(self):
            self._lock.acquire()

        def release(self):
            self._lock.release()

    def send(self, peer: str, tag: str, payload: Any, step: int) -> None:
        env = _Envelope(peer=peer, tag=tag, payload=payload, step=step)
        with self._cv:
            self._pending.setdefault((peer, step), []).append(env)
            self._cv.notify_all()

    def recv(
        self,
        peer: str,
        tag: str,
        step: int,
        timeout: float = 120.0,
    ) -> Optional[Any]:
        """Block until a matching ``(tag, step)`` envelope arrives for ``peer``.

        Stale envelopes (same peer, smaller step) are dropped silently.
        Returns ``None`` on timeout or after ``close``.
        """
        deadline = time.time() + timeout
        with self._cv:
            while True:
                if self._closed:
                    return None
                key = (peer, step)
                queue = self._pending.get(key, [])
                for idx, env in enumerate(queue):
                    if env.tag == tag:
                        # Pop and return.
                        queue.pop(idx)
                        if not queue:
                            del self._pending[key]
                        return env.payload
                # Drain stale envelopes for this peer (any step < expected).
                for stale_step in list(self._pending.keys()):
                    sp, ss = stale_step
                    if sp == peer and ss < step:
                        del self._pending[stale_step]
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=min(remaining, 1.0))

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._pending.clear()
            self._cv.notify_all()


# =============================================================================
#  Multi-process queue bus (legacy IPC stub)
# =============================================================================
class QueueBus:
    """mp.Queue-backed bus used by ``LegacyIPCStub``.

    Each peer has an outbound queue (driver → worker) and an inbound queue
    (worker → driver). Stale-reply filtering mirrors what
    ``IPCProtocol._recv_with_drain`` did: messages with the wrong tag or with
    ``msg_step < expected_step`` are dropped.
    """

    def __init__(
        self,
        ctx: Optional[mp.context.BaseContext] = None,
        queue_pairs: Optional[Dict[str, Tuple[Any, Any]]] = None,
    ) -> None:
        """Two construction modes:

        * ``queue_pairs`` provided — wrap pre-existing queues (used by
          ``LegacyIPCStub`` so it can reuse the queues it built for the
          worker processes).
        * otherwise — build six brand-new queues under ``mp.get_context("spawn")``.
        """
        self._ctx = ctx or mp.get_context("spawn")
        if queue_pairs is None:
            queue_pairs = {
                "U": (self._ctx.Queue(), self._ctx.Queue()),
                "M": (self._ctx.Queue(), self._ctx.Queue()),
                "S": (self._ctx.Queue(), self._ctx.Queue()),
            }
        self._queues: Dict[str, Tuple[Any, Any]] = queue_pairs
        # Per-peer stale-reply bookkeeping.
        self._stale_lock = threading.Lock()
        self._stale: Dict[str, list] = {"U": [], "M": [], "S": []}

    # The "driver side" calls send/recv against outbound/inbound queues.
    # queue_pairs structure: {"U": (to_U, from_U), "M": (to_M, from_M), ...}

    def send(self, peer: str, tag: str, payload: Any, step: int) -> None:
        to_peer, _ = self._queues[peer]
        to_peer.put((tag, payload, step), timeout=10.0)

    def recv(
        self,
        peer: str,
        tag: str,
        step: int,
        timeout: float = 120.0,
    ) -> Optional[Any]:
        _, from_peer = self._queues[peer]
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = from_peer.get(timeout=min(deadline - time.time(), 1.0))
            except Exception:
                continue
            if not isinstance(msg, tuple) or len(msg) < 3:
                continue
            msg_tag, payload, msg_step = msg[0], msg[1], msg[2]
            if msg_tag == tag and msg_step == step:
                if isinstance(payload, dict) and "error" in payload:
                    logger.warning(
                        "peer %s returned error at step %d: %s",
                        peer, step, payload["error"],
                    )
                return payload if isinstance(payload, dict) else {}
            with self._stale_lock:
                if msg_step >= 0 and msg_step < step:
                    self._stale[peer].append((msg_step, msg_tag, payload))
        logger.error("timeout waiting for %s from %s at step %d", tag, peer, step)
        return {"error": f"timeout: {tag}"}

    def get_queue_pairs(self) -> Dict[str, Tuple[Any, Any]]:
        """Return raw ``(outbound, inbound)`` tuples per peer — for spawning
        worker processes that bind to these queues."""
        return self._queues

    def close(self) -> None:
        # Queues are owned by the caller (legacy stub); do not close them here.
        return


__all__ = ["MessageBus", "InProcessBus", "QueueBus"]
