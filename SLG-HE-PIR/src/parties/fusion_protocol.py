"""
In-process fusion protocol for SLG-HE-PIR v2.0.

Drops the spawn-three-workers model (and its per-process CUDA Context) by
co-locating PartyU/PartyM/PartyS in a single Python process. All 10 messages
of the training protocol become direct in-process function calls — no IPC,
no pickle, no per-step ``CUDA_VISIBLE_DEVICES`` reset, no SemLock patch.

Why this exists
---------------
The original ``IPCProtocol`` ``spawn``-mode design (see ``ipc_protocol.py``)
was introduced to dodge the well-known pitfall where a fork child inherits
the parent's CUDA driver state via ``CachingAllocator`` — leading to the
inability to use the GPU from workers. ``spawn`` fixes that, but creates a
new problem for resource-constrained demo machines: each worker instantiates
its own ``torch.cuda`` context (~1.2 GB / context), pushing the U/M/S
weights + activations past the 32 GB ceiling on RTX 5090.

For a demo / prototype deployment the three-way process split is overkill.
Production deployments will run each party on its own physical host — the
privacy boundary there is enforced by the network topology, not by Python
processes. So at the demo layer we collapse the three workers into a
single ``FusionProtocol`` instance whose role is the same as
``IPCProtocol``: drive a Trainer through ``step_train`` /
``step_train_chunked`` / ``step_val`` / ``gather_checkpoints`` /
``shutdown``.

A single BFV ``backend`` instance is shared across all three party objects;
the M-side gets ``secret_key`` re-attached on its copy in
``PartyM._setup_bfv`` (matching spawn semantics). See ``PartyS._setup_bfv``
for the analogous sanitizer on the S side.

DRY with ``IPCProtocol``
------------------------
The 10-message training protocol is unchanged; only the transport
changes from ``mp.Queue.put`` → Python return values. Every method here
mirrors the signature and return shape of ``IPCProtocol`` so the existing
``Trainer`` consumes ``FusionProtocol`` with zero changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .ipc_protocol import StepResult, StepProfiler
from .party_u import PartyU
from .party_m import PartyM
from .party_s import PartyS

__all__ = ["FusionProtocol"]

logger = logging.getLogger(__name__)


@dataclass
class _ChunkedSplit:
    """Logical chunk boundaries across the ``(B * S)`` token axis."""
    starts: List[int]
    ends: List[int]

    @property
    def n_chunks(self) -> int:
        return len(self.starts)


def _split_into_chunks(n_tokens: int, chunk_tokens: int) -> _ChunkedSplit:
    """Mirror ``IPCProtocol.step_train_chunked``: single chunk if
    ``chunk_tokens >= n_tokens`` or ``chunk_tokens <= 0``."""
    if chunk_tokens <= 0 or chunk_tokens >= n_tokens:
        return _ChunkedSplit(starts=[0], ends=[n_tokens])
    starts: List[int] = []
    ends: List[int] = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk_tokens, n_tokens)
        starts.append(start)
        ends.append(end)
        start = end
    return _ChunkedSplit(starts=starts, ends=ends)


class FusionProtocol:
    """In-process coordinator that drives the three party objects directly.

    Mirrors the public surface of ``IPCProtocol``:

      * ``step_train(batch, global_step) -> StepResult``
      * ``step_train_chunked(batch, global_step, chunk_tokens=...) -> StepResult``
      * ``step_val(val_batch, global_step) -> dict``
      * ``gather_checkpoints() -> dict``
      * ``shutdown() -> None``
      * ``broadcast_stop() -> None``
      * ``profiler`` (StepProfiler or None)

    Args:
        u_submodel_path / m_submodel_path / s_lm_head_path: ``model_path``
            strings passed to each ``load_*_submodel`` call.
        bfv_backend: passed through to ``PartyS`` (the only party that
            currently accepts an externally-injected backend). ``PartyU``
            and ``PartyM`` each instantiate their own ``BFVPrivSelectV2Backend``
            in their ``_setup_bfv`` — this matches the spawn-worker design
            where each worker independently holds its own SEAL context.
            The three independent SEAL contexts add ~200 MB of C++ heap
            each; we trade that off against clearer isolation of the
            privacy boundary (only ``PartyM`` ever holds ``sk_M``).
        bfv_sk_pem / bfv_pk_pem / prg_seed: bytes forwarded to the
            per-party ctor (sk_pem is only used by PartyM, which loads
            sk_M into its own backend copy).
        hint_table: shared ``HintTable`` instance.
        config: the ``worker_config`` dict copied from ``finetune.py``.
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
        config: "dict",
    ) -> None:
        self.config = config
        self.bfv_backend = bfv_backend
        self.hint_table = hint_table
        self.bfv_sk_pem = bfv_sk_pem
        self.bfv_pk_pem = bfv_pk_pem
        self.prg_seed = prg_seed

        # ------------------------------------------------------------------
        # Step profiler — same surface as ``IPCProtocol``, so that
        # ``Trainer.step_callback`` reading ``ipc.profiler.recent_steps``
        # still works.
        # ------------------------------------------------------------------
        self.profiler: Optional[StepProfiler] = None
        if bool(self.config.get("ENABLE_STEP_PROFILING", True)):
            log_dir = self.config.get("LOG_DIR")
            self.profiler = StepProfiler(log_dir=log_dir)

        logger.info(
            "FusionProtocol starting in-process U/M/S"
        )

        # ------------------------------------------------------------------
        # Construct U / M / S. Each Party's ``__init__`` loads its own
        # submodel into the shared CUDA context. Loading is sequential
        # (no overlap), but PyTorch's caching allocator reclaims the
        # transient CPU→GPU transfer buffers between calls.
        # ------------------------------------------------------------------
        t_init = time.time()
        logger.info("[Fusion] constructing PartyU (embed + decoder[0:16))...")
        self.party_u = PartyU(
            model_path=u_submodel_path,
            bfv_pk_pem=bfv_pk_pem,
            prg_seed=prg_seed,
            hint_table=hint_table,
            config=config,
        )

        logger.info("[Fusion] constructing PartyM (decoder[16:32) + LoRA + sk_M)...")
        self.party_m = PartyM(
            model_path=m_submodel_path,
            bfv_sk_pem=bfv_sk_pem,
            bfv_pk_pem=bfv_pk_pem,
            config=config,
        )

        logger.info("[Fusion] constructing PartyS (V + mmap DB)...")
        self.party_s = PartyS(
            lm_head_path=s_lm_head_path,
            bfv_pk_pem=bfv_pk_pem,
            prg_seed=prg_seed,
            bfv_backend=bfv_backend,
            hint_table=hint_table,
            config=config,
        )
        logger.info(
            "[Fusion] U/M/S ready in %.1fs (single CUDA context shared)",
            time.time() - t_init,
        )

    # ------------------------------------------------------------------ #
    #  Flat (one-shot) training step
    # ------------------------------------------------------------------ #
    def step_train(self, batch: dict, global_step: int) -> StepResult:
        """Execute one training step in-place.

        Mirrors ``IPCProtocol.step_train``: U forward → M forward →
        S process_logits → U privselect → M backward.

        Returns:
            ``StepResult`` identical to the IPC version (loss, gpu_mem_mb,
            step_time_ms). For flat mode ``n_chunks == 1``.
        """
        t0 = time.time()
        prof = self.profiler

        # === [1-2] U forward ===
        if prof:
            prof.begin_phase("forward_U_wait_send")
        if prof:
            prof.end_phase("forward_U_wait_send")
            t_forward_u = time.time()
        u_result = self.party_u.forward_train(batch)
        H_U = u_result["H_U"]
        if prof:
            self._add_phase(prof, "forward_U", t_forward_u)

        # === [3-4] M forward ===
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
        if prof:
            t_forward_m = time.time()
        m_result = self.party_m.forward(H_U, attention_mask=attention_mask)
        H_M = m_result["H_M"]
        if prof:
            self._add_phase(prof, "forward_M", t_forward_m)

        # === [5-6] S process_logits (single shot) ===
        if prof:
            t_s_logits = time.time()
        s_result = self.party_s.process_logits_dispatch({
            "H_M": H_M,
            "step": global_step,
        })
        if prof:
            self._add_phase(prof, "s_logits", t_s_logits)

        # === [7-8] U privselect ===
        if prof:
            t_priv_u = time.time()
        u_priv = self.party_u.privselect_and_recover_dispatch({
            "s3pir_responses": s_result["s3pir_responses"],
            "step": global_step,
        })
        ct_list = u_priv.get("ct_list") or []
        if prof:
            self._add_phase(prof, "priv_U", t_priv_u)

        # === [9-10] M backward + LoRA step ===
        if isinstance(batch, dict) and "input_ids" in batch:
            expected_shape = (
                int(batch["input_ids"].shape[0]),
                int(batch["input_ids"].shape[1]),
            )
        else:
            expected_shape = None

        if prof:
            t_backward_m = time.time()
        ack = self.party_m.backward_and_update({
            "ct_from_U": ct_list,
            "s_share": s_result.get("s_shares") or [],
            "step": global_step,
            "expected_shape": expected_shape,
        })
        if prof:
            self._add_phase(prof, "backward_M", t_backward_m)

        step_time_ms = (time.time() - t0) * 1000

        if prof:
            prof.end_step(
                step=global_step,
                n_tokens=H_U.shape[0] * H_U.shape[1] if H_U is not None else 0,
                n_chunks=1,
                step_time_ms=step_time_ms,
                extra={"mode": "flat"},
            )

        return StepResult(
            step=global_step,
            loss=float(ack.get("loss", 0.0)),
            gpu_mem_mb=float(ack.get("gpu_mem_mb", 0.0)),
            step_time_ms=step_time_ms,
            attack_dumps=ack.get("attack_dumps", {}),
            n_chunks=1,
        )

    # ------------------------------------------------------------------ #
    #  Chunked training step (S once → U per-chunk → M once)
    # ------------------------------------------------------------------ #
    def step_train_chunked(
        self,
        batch: dict,
        global_step: int,
        chunk_tokens: int = 3072,
    ) -> StepResult:
        """Same shape as ``IPCProtocol.step_train_chunked`` but in-process.

        The chunking exists to allow U-side add_mask to overlap with M-side
        decrypt in production: chunk i's M-decrypt can run while chunk i+1
        U-add_mask runs. **In FusionProtocol there is no separate U or M
        process**, so they share the Python GIL and cannot physically
        overlap. The chunking is still useful because:

          * U's ``privselect_and_recover_dispatch`` calls the parallel pool
            (``U_N_WORKERS``). Splitting into chunks lets each pool chunk
            release its SEAL ctx between chunks.
          * The \"roll a few partial steps\" semantic lets the profiler
            emit per-chunk timings useful for diagnosing per-chunk cost.
          * The Trainer / ``step_callback`` API contract stays identical
            to the IPC version, enabling one-pathway testing.

        Args:
            batch: same dict as ``step_train``.
            global_step: same.
            chunk_tokens: tokens per chunk. ``n_chunks = ceil(n_tokens / chunk_tokens)``.
        """
        t0 = time.time()
        prof = self.profiler

        # === Step A: U forward ===
        if prof:
            t_forward_u = time.time()
        u_result = self.party_u.forward_train(batch)
        H_U = u_result["H_U"]
        if prof:
            self._add_phase(prof, "forward_U", t_forward_u)

        # === Step B: M forward ===
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
        if prof:
            t_forward_m = time.time()
        m_result = self.party_m.forward(H_U, attention_mask=attention_mask)
        H_M = m_result["H_M"]
        if prof:
            self._add_phase(prof, "forward_M", t_forward_m)

        # === Step C: S process_logits (one-shot — no per-chunk S work) ===
        if prof:
            t_s_logits = time.time()
        s_result = self.party_s.process_logits_dispatch({
            "H_M": H_M,
            "step": global_step,
        })
        if prof:
            self._add_phase(prof, "s_logits", t_s_logits)

        s3pir_responses = s_result.get("s3pir_responses") or []
        s_shares = s_result.get("s_shares") or []
        n_tokens = len(s3pir_responses)
        if n_tokens == 0:
            raise RuntimeError(
                f"step {global_step}: S returned 0 s3pir_responses"
            )

        # === Step D: chunk split ===
        split = _split_into_chunks(n_tokens, chunk_tokens)
        n_chunks = split.n_chunks
        logger.info(
            "step_train_chunked: step=%d, n_tokens=%d, K=%d chunks of ~%d tokens",
            global_step, n_tokens, n_chunks, chunk_tokens,
        )

        # === Step E: stream chunks through U ===
        chunk_ct_lists: List[List[bytes]] = [[] for _ in range(n_chunks)]
        for chunk_id, (c_start, c_end) in enumerate(zip(split.starts, split.ends)):
            chunk_resp = s3pir_responses[c_start:c_end]
            t_chunk_u0 = time.time()
            chunk_reply = self.party_u.privselect_and_recover_dispatch({
                "s3pir_responses": chunk_resp,
                "step": global_step,
            })
            t_chunk_u_dt = (time.time() - t_chunk_u0) * 1000
            if prof:
                prof.record_chunk("U", t_chunk_u_dt)
            ct_list = (chunk_reply.get("ct_list") or [])
            chunk_ct_lists[chunk_id] = list(ct_list)

        # Stitch.
        all_ct: List[bytes] = []
        for chunk_ct in chunk_ct_lists:
            all_ct.extend(chunk_ct)
        if len(all_ct) != n_tokens:
            raise RuntimeError(
                f"step {global_step}: chunked U produced {len(all_ct)} cts, "
                f"expected {n_tokens}"
            )

        # === Step F: M backward + LoRA step ===
        if isinstance(batch, dict) and "input_ids" in batch:
            expected_shape = (
                int(batch["input_ids"].shape[0]),
                int(batch["input_ids"].shape[1]),
            )
        else:
            expected_shape = None

        if prof:
            t_backward_m = time.time()
        ack = self.party_m.backward_and_update({
            "ct_from_U": all_ct,
            "s_share": s_shares,
            "step": global_step,
            "expected_shape": expected_shape,
        })
        if prof:
            self._add_phase(prof, "backward_M", t_backward_m)
            # Aggregate priv_U as sum of per-chunk round-trips, matching
            # ``IPCProtocol`` semantics.
            priv_u_ms = sum(prof._chunk_u_times)
            prof.phases["priv_U"] = priv_u_ms
            if "priv_U" not in prof.order:
                prof.order.append("priv_U")

        step_time_ms = (time.time() - t0) * 1000

        if prof:
            prof.end_step(
                step=global_step,
                n_tokens=n_tokens,
                n_chunks=n_chunks,
                step_time_ms=step_time_ms,
                extra={"mode": "chunked"},
            )

        return StepResult(
            step=global_step,
            loss=float(ack.get("loss", 0.0)),
            gpu_mem_mb=float(ack.get("gpu_mem_mb", 0.0)),
            step_time_ms=step_time_ms,
            attack_dumps=ack.get("attack_dumps", {}),
            n_chunks=n_chunks,
        )

    # ------------------------------------------------------------------ #
    #  Validation step
    # ------------------------------------------------------------------ #
    def step_val(self, val_batch: dict, global_step: int) -> Dict[str, Any]:
        """Validation forward pass via PIR-style flow, returns dict of
        metric-friendly fields (``predictions`` / ``labels``).

        Note: in production the IPC version splits this across three
        workers (U → M → S → U). Here we run it inline.

        In Design-2 the val pass still goes U→M→S, but S uses
        ``generate_predictions`` (no S3PIR parity) since labels are
        public at validation time. We use ``forward_train`` for U
        because ``PartyU.forward_val`` does not exist (it would be a
        duplicate of ``forward_train`` for the in-process flow).
        """
        # U forward (same as training forward — backward-free).
        u_result = self.party_u.forward_train(val_batch)
        H_U = u_result["H_U"]

        # M forward — produces both H_M and logits (M needs V's shape, but
        # in fusion mode M has no V matrix; only S has). The current M
        # forward returns H_M only, so we then ask S to compute logits.
        attention_mask = (
            val_batch.get("attention_mask") if isinstance(val_batch, dict) else None
        )
        m_result = self.party_m.forward(H_U, attention_mask=attention_mask)
        H_M = m_result["H_M"]

        # S computes predictions via standard forward (no privacy needed at val).
        s_pred = self.party_s.generate_predictions({"H_M": H_M})
        predictions = s_pred.get("predictions") or []

        # Labels are read off the batch in main-thread (Trainer consumes them).
        labels: List[str] = []
        if isinstance(val_batch, dict):
            label_field = val_batch.get("label")
            if label_field is not None:
                if isinstance(label_field, torch.Tensor):
                    labels = [
                        str(x) for x in label_field.reshape(-1).tolist()
                    ]
                elif isinstance(label_field, list):
                    labels = [str(x) for x in label_field]
                else:
                    labels = [str(label_field)]
            else:
                # Fall back to output_text (val/test datasets expose it).
                output_text = val_batch.get("output_text")
                if output_text is not None:
                    if isinstance(output_text, list):
                        labels = [str(x) for x in output_text]
                    else:
                        labels = [str(output_text)]

        return {"predictions": predictions, "labels": labels}

    # ------------------------------------------------------------------ #
    #  Control plane
    # ------------------------------------------------------------------ #
    def broadcast_stop(self) -> None:
        """Logical no-op in fusion mode — there is no worker to signal.

        Retained to keep the API symmetric with ``IPCProtocol``.
        """
        logger.info("FusionProtocol.broadcast_stop — no-op (in-process)")

    def gather_checkpoints(self) -> Dict[str, Any]:
        """Collect checkpoints from all three party objects directly.

        Matches the ``IPCProtocol.gather_checkpoints`` shape:
        ``{"U": ..., "M": ..., "S": ...}``.
        """
        return {
            "U": self.party_u.save_checkpoint(),
            "M": self.party_m.save_checkpoint(),
            "S": self.party_s.save_checkpoint(),
        }

    def shutdown(self) -> None:
        """Release CUDA / Python resources.

        Detaches party objects (which hold CUDA tensors) and empties the
        caching allocator to give back memory to subsequent processes /
        other tests on the same machine.
        """
        try:
            self.broadcast_stop()
        except Exception:
            pass
        for attr in ("party_u", "party_m", "party_s"):
            try:
                getattr(self, attr, None)
            except Exception:
                pass
        self.party_u = None  # type: ignore[assignment]
        self.party_m = None  # type: ignore[assignment]
        self.party_s = None  # type: ignore[assignment]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("FusionProtocol.shutdown complete")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_phase(prof: StepProfiler, name: str, t0: float) -> None:
        """Mirror the ``IPCProtocol`` profiler convention: each phase is the
        delta from its ``t0`` start, accumulated into ``prof.phases[name]``
        with the order recorded the first time the phase is seen."""
        dt_ms = (time.time() - t0) * 1000
        prof.phases[name] = prof.phases.get(name, 0.0) + dt_ms
        if name not in prof.order:
            prof.order.append(name)
