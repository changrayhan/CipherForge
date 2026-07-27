"""
Legacy three-process IPC runtime — preserved for backwards-compatibility.

The active runtime is now ``HeterogeneousProtocol`` (see
:mod:`heterogeneous_protocol`). This module exposes the legacy
``LegacyIPCStub`` class plus the worker entry-point functions it spawns. The
stub class is intentionally **not invoked** by ``finetune.py`` — it is kept
around for two purposes:

  1. Multi-host deployment preview. When deploying to three physical hosts,
     swap the in-process bus for an RPC bus and reuse ``LegacyIPCStub``'s
     surface unchanged.
  2. Cryptographic boundary audit. Researchers who want to verify that
     ``sk_M`` truly never crosses a process boundary can temporarily run
     ``LegacyIPCStub`` to do an apples-to-apples comparison against the
     heterogeneous runtime.

Note: this module is *not* the active code path. ``finetune.py` instantiates
``HeterogeneousProtocol`` directly. The legacy stub remains here for the
audit/deployment-preview use cases listed above.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .wire import StepResult, StepProfiler

__all__ = ["LegacyIPCStub", "_worker_U_entry", "_worker_M_entry", "_worker_S_entry"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Worker entry points — same as the old IPCProtocol worker functions.
#  These run inside the spawned processes and never return; the parent talks
#  to them through ``mp.Queue`` pairs built by LegacyIPCStub.
# --------------------------------------------------------------------------- #
def _worker_U_entry(
    model_path: str,
    bfv_pk_pem: bytes,
    prg_seed: bytes,
    hint_table,
    config: dict,
    queue_to_U,
    queue_from_U,
    queue_to_M,
    queue_from_M,
    queue_to_S,
    queue_from_S,
) -> None:
    """Party U (User) worker process."""
    import logging as _logging
    import os as _os
    _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _logging.basicConfig(level=_logging.INFO, format="[U] %(asctime)s %(message)s")
    from .party_u import PartyU
    import multiprocessing.synchronize as _mp_sync
    _mp_sync.SemLock._cleanup = staticmethod(lambda name: None)

    try:
        worker = PartyU(
            model_path=model_path,
            bfv_pk_pem=bfv_pk_pem,
            prg_seed=prg_seed,
            hint_table=hint_table,
            config=config,
        )
    except Exception as e:
        _logging.getLogger("worker_U").error("Worker U init failed: %s", e)
        import traceback; traceback.print_exc()
        queue_from_U.put(("ERROR", {"error": str(e)}, -1))
        return

    while True:
        try:
            cmd = queue_to_U.get(timeout=300)
        except Exception:
            continue
        if not cmd or not isinstance(cmd, tuple):
            continue
        tag = cmd[0]
        if tag == "STOP":
            break
        elif tag == "FORWARD":
            batch, step = cmd[1], cmd[2]
            try:
                result = worker.forward_train(batch)
                queue_from_U.put(("H_U", result, step))
            except Exception as e:
                queue_from_U.put(("H_U", {"error": str(e)}, step))
        elif tag == "S3PIR_RESP":
            s3pir_msg, step = cmd[1], cmd[2]
            try:
                result = worker.privselect_and_recover_dispatch(s3pir_msg)
                queue_from_U.put(("G_H_MASKED", {"g_H_masked": result}, step))
            except Exception as e:
                queue_from_U.put(("G_H_MASKED", {"error": str(e)}, step))
        elif tag == "S3PIR_RESP_CHUNK":
            s3pir_msg, step, chunk_id, n_chunks = (
                cmd[1], cmd[2], cmd[3] if len(cmd) > 3 else 0, cmd[4] if len(cmd) > 4 else 1,
            )
            try:
                result = worker.privselect_and_recover_dispatch(s3pir_msg)
                queue_from_U.put(
                    ("G_H_MASKED_CHUNK",
                     {"g_H_masked": result, "chunk_id": chunk_id, "n_chunks": n_chunks},
                     step)
                )
            except Exception as e:
                queue_from_U.put(
                    ("G_H_MASKED_CHUNK",
                     {"error": str(e), "chunk_id": chunk_id, "n_chunks": n_chunks},
                     step)
                )
        elif tag == "VAL_PRED":
            pred_msg, step = cmd[1], cmd[2]
            try:
                metrics = worker.compute_val_metrics(pred_msg)
                queue_from_U.put(("VAL_METRICS", {"metrics": metrics}, step))
            except Exception as e:
                queue_from_U.put(("VAL_METRICS", {"error": str(e)}, step))
        elif tag == "SAVE":
            try:
                ckpt = worker.save_checkpoint()
                queue_from_U.put(("CHECKPOINT", ckpt, -1))
            except Exception as e:
                queue_from_U.put(("CHECKPOINT", {"error": str(e)}, -1))


def _worker_M_entry(
    model_path: str,
    bfv_sk_pem: bytes,
    bfv_pk_pem: bytes,
    config: dict,
    queue_to_M,
    queue_from_M,
    queue_to_U,
    queue_from_U,
    queue_to_S,
    queue_from_S,
) -> None:
    """Party M (Model) worker process."""
    import logging as _logging
    import os as _os
    _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    _logging.basicConfig(level=_logging.INFO, format="[M] %(asctime)s %(message)s")
    from .party_m import PartyM
    import multiprocessing.synchronize as _mp_sync
    _mp_sync.SemLock._cleanup = staticmethod(lambda name: None)

    try:
        worker = PartyM(
            model_path=model_path,
            bfv_sk_pem=bfv_sk_pem,
            bfv_pk_pem=bfv_pk_pem,
            config=config,
        )
    except Exception as e:
        _logging.getLogger("worker_M").error("Worker M init failed: %s", e)
        import traceback; traceback.print_exc()
        queue_from_M.put(("ERROR", {"error": str(e)}, -1))
        return

    while True:
        try:
            cmd = queue_to_M.get(timeout=300)
        except Exception:
            continue
        if not cmd or not isinstance(cmd, tuple):
            continue
        tag = cmd[0]
        if tag == "STOP":
            break
        elif tag == "H_U":
            payload, step = cmd[1], cmd[2]
            try:
                # payload is either {"H_U": tensor, ...} or just the tensor
                H_U = payload.get("H_U") if isinstance(payload, dict) else payload
                attention_mask = payload.get("attention_mask") if isinstance(payload, dict) else None
                result = worker.forward(H_U, attention_mask=attention_mask)
                queue_from_M.put(("LOGITS", result, step))
            except Exception as e:
                queue_from_M.put(("LOGITS", {"error": str(e)}, step))
        elif tag == "VAL_H_U":
            payload, step = cmd[1], cmd[2]
            try:
                H_U = payload.get("H_U") if isinstance(payload, dict) else payload
                attention_mask = payload.get("attention_mask") if isinstance(payload, dict) else None
                result = worker.forward(H_U, attention_mask=attention_mask)
                queue_from_M.put(("VAL_LOGITS", result, step))
            except Exception as e:
                queue_from_M.put(("VAL_LOGITS", {"error": str(e)}, step))
        elif tag == "INJECT_GRAD":
            payload, step = cmd[1], cmd[2]
            try:
                ack = worker.backward_and_update_dispatch(payload)
                queue_from_M.put(("STEP_ACK", ack, step))
            except Exception as e:
                queue_from_M.put(("STEP_ACK", {"error": str(e)}, step))
        elif tag == "SAVE":
            try:
                ckpt = worker.save_checkpoint()
                queue_from_M.put(("CHECKPOINT", ckpt, -1))
            except Exception as e:
                queue_from_M.put(("CHECKPOINT", {"error": str(e)}, -1))


def _worker_S_entry(
    lm_head_path: str,
    bfv_pk_pem: bytes,
    prg_seed: bytes,
    hint_table,
    config: dict,
    queue_to_S,
    queue_from_S,
    queue_to_U,
    queue_from_U,
    queue_to_M,
    queue_from_M,
) -> None:
    """Party S (Server) worker process."""
    import os as _os
    _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from .party_s import PartyS
    import multiprocessing.synchronize as _mp_sync
    _mp_sync.SemLock._cleanup = staticmethod(lambda name: None)

    from src.core.bfv_privselect_v2_adapter import (
        BFVEncryptedDatabase,
        BFVPrivSelectV2Backend,
        create_bfv_context,
    )

    bfv_backend_local = BFVPrivSelectV2Backend(
        n_entries=int(config["vocab_size"]),
        vec_dim=int(config.get("hidden_dim", 4096)),
        cache_dir=str(config.get("bfv_cache_dir", "/root/autodl-tmp/slg-bfv-cache")),
        poly_degree=int(config.get("poly_degree", 4096)),
        plain_bits=int(config.get("plain_bits", 30)),
        scale=int(config.get("scale", 10_000)),
    )
    ctx = create_bfv_context(
        poly_degree=int(config.get("poly_degree", 4096)),
        plain_bits=int(config.get("plain_bits", 30)),
    )
    bfv_backend_local.enc_db = BFVEncryptedDatabase(
        ctx=ctx,
        n_entries=int(config["vocab_size"]),
        vec_dim=int(config.get("hidden_dim", 4096)),
        public_key=bfv_backend_local.public_key,
        cache_path=bfv_backend_local._cache_path,
    )
    bfv_backend_local.enc_db._ct_db = (
        BFVEncryptedDatabase._load_cache_mmap(bfv_backend_local._cache_path)
    )

    worker = PartyS(
        lm_head_path=lm_head_path,
        bfv_pk_pem=bfv_pk_pem,
        prg_seed=prg_seed,
        bfv_backend=bfv_backend_local,
        hint_table=hint_table,
        config=config,
    )

    while True:
        try:
            cmd = queue_to_S.get(timeout=300)
        except Exception:
            continue
        if not cmd or not isinstance(cmd, tuple):
            continue
        tag = cmd[0]
        if tag == "STOP":
            break
        elif tag == "COMPUTE_LOGITS":
            H_M, step = cmd[1], cmd[2]
            try:
                result = worker.process_logits_dispatch({"H_M": H_M, "step": step})
                queue_from_S.put(("S3PIR_RESP", result, step))
            except Exception as e:
                queue_from_S.put(("S3PIR_RESP", {"error": str(e)}, step))
        elif tag == "VAL_LOGITS":
            logits, step = cmd[1], cmd[2]
            try:
                predictions = worker.generate_predictions(logits)
                queue_from_S.put(("VAL_PRED", {"predictions": predictions}, step))
            except Exception as e:
                queue_from_S.put(("VAL_PRED", {"error": str(e)}, step))
        elif tag == "SAVE":
            try:
                ckpt = worker.save_checkpoint()
                queue_from_S.put(("CHECKPOINT", ckpt, -1))
            except Exception as e:
                queue_from_S.put(("CHECKPOINT", {"error": str(e)}, -1))


# --------------------------------------------------------------------------- #
#  LegacyIPCStub
# --------------------------------------------------------------------------- #
class LegacyIPCStub:
    """Three-process IPC runtime kept for audit / multi-host preview.

    See module docstring for rationale. This class is **not** instantiated by
    ``finetune.py``; the active runtime is :class:`HeterogeneousProtocol`.

    Public API mirrors :class:`HeterogeneousProtocol`:
      * ``step_train(batch, global_step) -> StepResult``
      * ``step_train_chunked(batch, global_step, chunk_tokens) -> StepResult``
      * ``step_val(val_batch, global_step) -> dict``
      * ``gather_checkpoints() -> dict``
      * ``shutdown() -> None``
      * ``profiler`` (StepProfiler or None)
    """

    def __init__(
        self,
        u_submodel_path: str,
        m_submodel_path: str,
        s_lm_head_path: str,
        bfv_backend,
        hint_table,
        bfv_sk_pem: bytes,
        bfv_pk_pem: bytes,
        prg_seed: bytes,
        config: Dict,
    ):
        import warnings
        warnings.warn(
            "LegacyIPCStub is preserved for audit/multi-host preview only. "
            "The active runtime is HeterogeneousProtocol; do NOT use this in "
            "production training.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.config = config
        self.bfv_backend = bfv_backend
        self.hint_table = hint_table
        self.bfv_sk_pem = bfv_sk_pem
        self.bfv_pk_pem = bfv_pk_pem
        self.prg_seed = prg_seed

        self._alive = True
        self._stale_reply_lock = threading.Lock()
        self._stale_replies: Dict[str, List] = {"H_U": [], "LOGITS": [], "S3PIR_RESP": [],
                                                 "G_H_MASKED": [], "STEP_ACK": [],
                                                 "VAL_LOGITS": [], "VAL_PRED": [],
                                                 "VAL_METRICS": [], "CHECKPOINT": []}

        self.profiler: Optional[StepProfiler] = None
        if bool(self.config.get("ENABLE_STEP_PROFILING", True)):
            log_dir = self.config.get("LOG_DIR")
            self.profiler = StepProfiler(log_dir=log_dir)

        self.ctx = mp.get_context("spawn")
        self.queue_to_U = self.ctx.Queue()
        self.queue_from_U = self.ctx.Queue()
        self.queue_to_M = self.ctx.Queue()
        self.queue_from_M = self.ctx.Queue()
        self.queue_to_S = self.ctx.Queue()
        self.queue_from_S = self.ctx.Queue()

        self.proc_U = self.ctx.Process(
            target=_worker_U_entry,
            args=(
                u_submodel_path, bfv_pk_pem, prg_seed, hint_table, self.config,
                self.queue_to_U, self.queue_from_U,
                self.queue_to_M, self.queue_from_M,
                self.queue_to_S, self.queue_from_S,
            ),
            daemon=True,
        )
        self.proc_M = self.ctx.Process(
            target=_worker_M_entry,
            args=(
                m_submodel_path, bfv_sk_pem, bfv_pk_pem, self.config,
                self.queue_to_M, self.queue_from_M,
                self.queue_to_U, self.queue_from_U,
                self.queue_to_S, self.queue_from_S,
            ),
            daemon=True,
        )
        self.proc_S = self.ctx.Process(
            target=_worker_S_entry,
            args=(
                s_lm_head_path, bfv_pk_pem, prg_seed, hint_table, self.config,
                self.queue_to_S, self.queue_from_S,
                self.queue_to_U, self.queue_from_U,
                self.queue_to_M, self.queue_from_M,
            ),
            daemon=True,
        )

        self.proc_U.start()
        self.proc_M.start()
        self.proc_S.start()
        logger.info(
            "LegacyIPCStub workers started: U=%d M=%d S=%d",
            self.proc_U.pid, self.proc_M.pid, self.proc_S.pid,
        )

    # ------------------------------------------------------------------ #
    #  Stale-reply drain
    # ------------------------------------------------------------------ #
    def _recv_with_drain(self, queue, expected_tag: str, step: int, timeout: float = 120.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = queue.get(timeout=min(deadline - time.time(), 1.0))
            except Exception:
                continue
            if not isinstance(msg, tuple) or len(msg) < 3:
                continue
            tag, payload, msg_step = msg[0], msg[1], msg[2] if len(msg) > 2 else -1
            if tag == expected_tag and msg_step == step:
                if isinstance(payload, dict) and "error" in payload:
                    logger.warning("Worker %s returned error at step %d: %s",
                                   expected_tag, step, payload["error"])
                return payload if isinstance(payload, dict) else {}
            with self._stale_reply_lock:
                if msg_step >= 0 and msg_step < step:
                    self._stale_replies.setdefault(expected_tag, []).append((msg_step, payload))
        logger.error("Timeout waiting for %s at step %d", expected_tag, step)
        return {"error": f"timeout: {expected_tag}"}

    # ------------------------------------------------------------------ #
    #  Flat step
    # ------------------------------------------------------------------ #
    def step_train(self, batch, global_step) -> StepResult:
        t0 = time.time()
        self.queue_to_U.put(("FORWARD", batch, global_step))
        H_U_msg = self._recv_with_drain(self.queue_from_U, "H_U", global_step)
        self.queue_to_M.put(("H_U", H_U_msg, global_step))
        logits_msg = self._recv_with_drain(self.queue_from_M, "LOGITS", global_step)
        H_M = logits_msg.get("H_M") if isinstance(logits_msg, dict) else logits_msg
        self.queue_to_S.put(("COMPUTE_LOGITS", H_M, global_step))
        s3pir_msg = self._recv_with_drain(self.queue_from_S, "S3PIR_RESP", global_step)
        self.queue_to_U.put(("S3PIR_RESP", s3pir_msg, global_step))
        g_H_msg = self._recv_with_drain(self.queue_from_U, "G_H_MASKED", global_step)
        g_H_inner = g_H_msg.get("g_H_masked", g_H_msg)
        ct_list = g_H_inner.get("ct_list", []) if isinstance(g_H_inner, dict) else []
        s_shares = s3pir_msg.get("s_shares", [])
        if isinstance(batch, dict) and "input_ids" in batch:
            expected_shape = (
                int(batch["input_ids"].shape[0]),
                int(batch["input_ids"].shape[1]),
            )
        else:
            expected_shape = None
        self.queue_to_M.put(
            ("INJECT_GRAD",
             {"ct_from_U": ct_list, "s_share": s_shares,
              "step": global_step, "expected_shape": expected_shape},
             global_step),
        )
        ack = self._recv_with_drain(self.queue_from_M, "STEP_ACK", global_step)
        return StepResult(
            step=global_step,
            loss=float(ack.get("loss", 0.0)),
            gpu_mem_mb=float(ack.get("gpu_mem_mb", 0.0)),
            step_time_ms=(time.time() - t0) * 1000,
            attack_dumps=ack.get("attack_dumps", {}),
            n_chunks=1,
        )

    # ------------------------------------------------------------------ #
    #  Chunked step — same shape as HeterogeneousProtocol
    # ------------------------------------------------------------------ #
    def step_train_chunked(self, batch, global_step, chunk_tokens: int = 3072) -> StepResult:
        t0 = time.time()
        self.queue_to_U.put(("FORWARD", batch, global_step))
        H_U_msg = self._recv_with_drain(self.queue_from_U, "H_U", global_step)
        self.queue_to_M.put(("H_U", H_U_msg, global_step))
        logits_msg = self._recv_with_drain(self.queue_from_M, "LOGITS", global_step)
        H_M = logits_msg.get("H_M") if isinstance(logits_msg, dict) else logits_msg
        self.queue_to_S.put(("COMPUTE_LOGITS", H_M, global_step))
        s3pir_msg = self._recv_with_drain(self.queue_from_S, "S3PIR_RESP", global_step)
        s3pir_responses = s3pir_msg.get("s3pir_responses", [])
        s_shares = s3pir_msg.get("s_shares", [])
        n_tokens = len(s3pir_responses)
        # Single-chunk fast path
        if chunk_tokens <= 0 or chunk_tokens >= n_tokens:
            n_chunks = 1
            self.queue_to_U.put(("S3PIR_RESP", s3pir_msg, global_step))
            g_H_msg = self._recv_with_drain(self.queue_from_U, "G_H_MASKED", global_step)
            g_H_inner = g_H_msg.get("g_H_masked", g_H_msg)
            all_ct = g_H_inner.get("ct_list", []) if isinstance(g_H_inner, dict) else []
        else:
            n_chunks = (n_tokens + chunk_tokens - 1) // chunk_tokens
            all_ct = []
            for c_idx in range(n_chunks):
                c_start = c_idx * chunk_tokens
                c_end = min(c_start + chunk_tokens, n_tokens)
                chunk_resp = s3pir_responses[c_start:c_end]
                self.queue_to_U.put(
                    ("S3PIR_RESP_CHUNK", {"s3pir_responses": chunk_resp, "step": global_step},
                     global_step, c_idx, n_chunks),
                )
                chunk_msg = self._recv_with_drain(self.queue_from_U, "G_H_MASKED_CHUNK", global_step)
                inner = chunk_msg.get("g_H_masked", chunk_msg)
                ct_chunk = inner.get("ct_list", []) if isinstance(inner, dict) else []
                all_ct.extend(ct_chunk)
        if isinstance(batch, dict) and "input_ids" in batch:
            expected_shape = (
                int(batch["input_ids"].shape[0]),
                int(batch["input_ids"].shape[1]),
            )
        else:
            expected_shape = None
        self.queue_to_M.put(
            ("INJECT_GRAD",
             {"ct_from_U": all_ct, "s_share": s_shares,
              "step": global_step, "expected_shape": expected_shape},
             global_step),
        )
        ack = self._recv_with_drain(self.queue_from_M, "STEP_ACK", global_step)
        return StepResult(
            step=global_step,
            loss=float(ack.get("loss", 0.0)),
            gpu_mem_mb=float(ack.get("gpu_mem_mb", 0.0)),
            step_time_ms=(time.time() - t0) * 1000,
            attack_dumps=ack.get("attack_dumps", {}),
            n_chunks=n_chunks,
        )

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def step_val(self, val_batch, global_step) -> Dict:
        # NOTE: this path used to call worker.forward_val which never existed
        # on the rewritten PartyU. We use forward_train (no_grad → same effect)
        # via the legacy FORWARD tag.
        self.queue_to_U.put(("FORWARD", val_batch, global_step))
        H_U_msg = self._recv_with_drain(self.queue_from_U, "H_U", global_step)
        self.queue_to_M.put(("H_U", H_U_msg, global_step))
        logits_msg = self._recv_with_drain(self.queue_from_M, "LOGITS", global_step)
        H_M = logits_msg.get("H_M") if isinstance(logits_msg, dict) else logits_msg
        self.queue_to_S.put(("VAL_LOGITS", H_M, global_step))
        pred_msg = self._recv_with_drain(self.queue_from_S, "VAL_PRED", global_step)
        predictions = pred_msg.get("predictions", []) if isinstance(pred_msg, dict) else []
        labels = (val_batch.get("output_text") if isinstance(val_batch, dict) else None) \
            or (val_batch.get("labels") if isinstance(val_batch, dict) else None) \
            or (val_batch.get("target_text") if isinstance(val_batch, dict) else None)
        if isinstance(labels, (list, tuple)):
            labels = list(labels)
        self.queue_to_U.put(("VAL_PRED", {"predictions": predictions, "labels": labels or []}, global_step))
        metrics_msg = self._recv_with_drain(self.queue_from_U, "VAL_METRICS", global_step)
        return metrics_msg.get("metrics", {}) if isinstance(metrics_msg, dict) else {}

    # ------------------------------------------------------------------ #
    #  Checkpoint + shutdown
    # ------------------------------------------------------------------ #
    def gather_checkpoints(self) -> Dict:
        self.queue_to_U.put(("SAVE", None, -1))
        self.queue_to_M.put(("SAVE", None, -1))
        self.queue_to_S.put(("SAVE", None, -1))
        return {
            "U": self._recv_with_drain(self.queue_from_U, "CHECKPOINT", -1),
            "M": self._recv_with_drain(self.queue_from_M, "CHECKPOINT", -1),
            "S": self._recv_with_drain(self.queue_from_S, "CHECKPOINT", -1),
        }

    def shutdown(self) -> None:
        self._alive = False
        for q in [self.queue_to_U, self.queue_to_M, self.queue_to_S]:
            try:
                q.put(("STOP", None, -1), timeout=1.0)
            except Exception:
                pass
        for proc in [self.proc_U, self.proc_M, self.proc_S]:
            try:
                if proc.is_alive():
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.terminate()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()