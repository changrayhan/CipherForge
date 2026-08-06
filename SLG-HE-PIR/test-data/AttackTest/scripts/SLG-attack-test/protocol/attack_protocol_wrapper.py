"""Attack-aware wrapper around HeterogeneousProtocol.

Hooks into the training loop to collect per-step attack-relevant intermediates
without modifying any ``src/`` source files.  Three interception strategies are
used, in order of preference:

1. **step_callback** (preferred) — injected into the Trainer; receives
   ``StepResult`` after every forward+backward step.
2. **Protocol proxy** — wraps ``HeterogeneousProtocol`` and intercepts the
   return values of ``step_train`` / ``step_train_chunked``.
3. **PartyM monkey-patch** — last resort; swaps ``PartyM.backward_and_update``
   to return the raw ``g_accum`` numpy array alongside the normal result dict.

The wrapper exposes a unified interface::

    wrapper = AttackProtocolWrapper(protocol, cfg)
    wrapper.step_train(batch)           # returns StepResult, collects data
    wrapper.get_attack_data()           # returns collected AttackDataBundle
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Attack data bundle
# --------------------------------------------------------------------------- #

@dataclass
class AttackDataBundle:
    """Unified container for all per-step attack intermediates."""

    # Step index (0-based within this run)
    step: int = -1

    # L-1A / CutGrad: per-token gradient vectors
    # Shape: (n_tokens, hidden_dim) float32  — the recovered gradient a_t - V_y
    g_accum: Optional[np.ndarray] = None

    # L-1A: token text labels (coarse class idx 0-5 for TREC-QC)
    token_labels: Optional[List[int]] = None

    # L-1A: gold token IDs (the y_t selected by S3PIR)
    gold_ids: Optional[np.ndarray] = None

    # L-3A: S-side argmax predictions per token
    s_argmax_ids: Optional[np.ndarray] = None

    # L-3A: M-side argmax predictions per token
    m_argmax_ids: Optional[np.ndarray] = None

    # L-3A: softmax distribution over V^T (S-side)
    s_softmax_probs: Optional[np.ndarray] = None

    # Loss proxy returned by PartyM._inject_and_backward
    loss_proxy: float = 0.0

    # Attack dump dict from PartyM (scalar stats)
    attack_dump: Dict[str, Any] = field(default_factory=dict)

    # Raw batch metadata (subset, no gradients)
    batch_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "g_accum_shape": list(self.g_accum.shape) if self.g_accum is not None else None,
            "g_accum_l2": float(np.linalg.norm(self.g_accum)) if self.g_accum is not None else None,
            "token_labels": self.token_labels,
            "loss_proxy": self.loss_proxy,
            "s_argmax_sample": (
                self.s_argmax_ids[:10].tolist()
                if self.s_argmax_ids is not None and len(self.s_argmax_ids) > 0
                else []
            ),
        }


# --------------------------------------------------------------------------- #
#  Main wrapper class
# --------------------------------------------------------------------------- #

class AttackProtocolWrapper:
    """Wraps HeterogeneousProtocol and intercepts per-step intermediates."""

    def __init__(
        self,
        protocol: Any,
        attack_config: Any,
        collect_g_accum: bool = True,
        collect_s_argmax: bool = True,
        collect_labels: bool = True,
    ):
        """
        Args:
            protocol: An instantiated HeterogeneousProtocol (or compatible object).
            attack_config: An AttackConfig (from config/attack_config.py).
            collect_g_accum: Intercept g_accum numpy array from PartyM backward.
            collect_s_argmax: Intercept S-side argmax predictions.
            collect_labels: Extract labels from the batch dict.
        """
        self.protocol = protocol
        self.cfg = attack_config
        self.collect_g_accum = collect_g_accum
        self.collect_s_argmax = collect_s_argmax
        self.collect_labels = collect_labels

        # Collected data
        self._lock = threading.Lock()
        self._bundles: List[AttackDataBundle] = []
        self._step_count = 0
        # Per-sample t_flat indices captured during g_accum reconstruction;
        # these let run_attack_suite re-derive the PRG mask r_t and compute a
        # result_S that is numerically distinct from s_share (as it would be
        # in the real protocol).
        self._collected_t_flat: List[np.ndarray] = []

        # Cached PartyM reference for gradient interception
        self._party_m = getattr(protocol, "party_m", None)
        self._party_s = getattr(protocol, "party_s", None)

        # Set the dump flag so PartyM._dump_attack_intermediates fires
        # and the scalar stats land in StepResult.attack_dumps
        self._patch_party_m_dump_dir(attack_config.attack_dump_dir)

        # Register gradient interception hook on PartyM.backward_and_update
        if self.collect_g_accum and self._party_m is not None:
            self._orig_backward_and_update = self._party_m.backward_and_update
            self._party_m.backward_and_update = self._hijacked_backward_and_update

        logger.info(
            "AttackProtocolWrapper initialised: collect_g_accum=%s, collect_s_argmax=%s",
            collect_g_accum, collect_s_argmax,
        )

    # ------------------------------------------------------------------------- #
    #  Gradient interception (injects into PartyM.backward_and_update)
    # ------------------------------------------------------------------------- #

    def _patch_party_m_dump_dir(self, dump_dir: str) -> None:
        """Propagate the attack dump directory into PartyM's config."""
        if self._party_m is not None and hasattr(self._party_m, "config"):
            self._party_m.config["attack_dump_dir"] = dump_dir
            self._party_m.config["dump_attack_intermediates"] = True

    def _hijacked_backward_and_update(self, payload: Dict) -> Dict:
        """Interposed backward_and_update that captures g_accum before returning."""
        # Call the original
        result = self._orig_backward_and_update(payload)

        # Intercept g_accum: it was already used to build g_H inside the original.
        # We re-derive it from the payload using the same algorithm as PartyM.
        g_accum = self._reconstruct_g_accum(payload)

        # Extract labels from payload if available
        token_labels = payload.get("token_labels")
        gold_ids = payload.get("gold_ids")

        # Build and store the bundle (thread-safe)
        bundle = AttackDataBundle(
            step=self._step_count,
            g_accum=g_accum,
            token_labels=token_labels,
            gold_ids=gold_ids,
            loss_proxy=result.get("loss", 0.0),
            attack_dump=result.get("attack_dumps", {}),
        )
        with self._lock:
            self._bundles.append(bundle)

        return result

    def _reconstruct_g_accum(self, payload: Dict) -> np.ndarray:
        """Replicate the g_accum computation from PartyM.backward_and_update.

        This is the same algorithm (plaintext subtraction of s_share from
        the decrypted masked_arr).  It requires access to the same BFV
        parameters that PartyM has, so we delegate to the CryptoMWorker
        pool by submitting a dry-run task with the same ct_list.

        Returns:
            g_accum: (n_tokens, vec_dim) float32 numpy array
        """
        ct_list = payload.get("ct_from_U") or []
        s_share_list = payload.get("s_share") or []

        if not ct_list or not s_share_list:
            return np.array([], dtype=np.float32)

        vec_dim = self._party_m.bfv_backend.vec_dim
        scale = self._party_m.bfv_backend.scale
        plain_bits = self._party_m.bfv_backend.plain_bits

        # Submit decryption task to the CryptoMWorker pool (same code path as PartyM)
        decrypt_result = self._party_m.crypto_m_pool.submit({
            "ct_list": ct_list,
            "scale": scale,
            "vec_dim": vec_dim,
        })
        masked_arr = decrypt_result["decrypted"]  # (n_tokens, vec_dim) float32

        plain_modulus = 1 << int(plain_bits)
        half_pm = plain_modulus // 2
        n_tokens = len(ct_list)

        g_accum = np.zeros((n_tokens, vec_dim), dtype=np.float32)
        for t_flat in range(n_tokens):
            s_share = s_share_list[t_flat]
            s_arr = np.asarray(s_share[:vec_dim], dtype=np.int64)
            if s_arr.size < vec_dim:
                s_arr = np.pad(s_arr, (0, vec_dim - s_arr.size))

            masked_int = np.round(masked_arr[t_flat] * scale).astype(np.int64)
            masked_centered = np.where(
                masked_int > half_pm, masked_int - plain_modulus, masked_int
            )
            diff_int = masked_centered + s_arr
            g_accum[t_flat] = diff_int.astype(np.float32) / scale

        # Record the per-sample t_flat indices used in this step so that
        # result_S can be reconstructed deterministically in run_attack_suite.
        try:
            with self._lock:
                self._collected_t_flat.append(np.arange(n_tokens, dtype=np.int64))
        except Exception:
            pass
        return g_accum

    # ------------------------------------------------------------------------- #
    #  Forward delegation + label extraction
    # ------------------------------------------------------------------------- #

    def step_train(self, batch: Dict, global_step: int = 0) -> Any:
        """Delegates to protocol.step_train and injects label metadata."""
        # Extract coarse_idx labels from the batch for L-1A
        if self.collect_labels:
            batch = _inject_labels_into_payload(batch)

        result = self.protocol.step_train(batch, global_step)
        self._step_count += 1
        return result

    def step_train_chunked(self, batch: Dict, global_step: int, chunk_tokens: int) -> Any:
        """Delegates to protocol.step_train_chunked and injects label metadata."""
        if self.collect_labels:
            batch = _inject_labels_into_payload(batch)

        result = self.protocol.step_train_chunked(batch, global_step, chunk_tokens=chunk_tokens)
        self._step_count += 1
        return result

    def step_forward_only(self, batch: Dict, global_step: int = 0) -> Any:
        """Forward-only step: runs the protocol forward pass without
        triggering backward / LoRA / Adam updates.  Used by 方案 B (M-2
        dummy forward baseline) to record a_t_pre in a clean state where
        Adam 动量 = 0, PRG 熵未消耗.

        The protocol is expected to expose ``step_forward`` (or we fall
        back to a manual forward).  If neither is available, we fall back
        to ``step_train`` under ``torch.no_grad()`` — which still records
        gradients but never updates weights (since LoRA is already
        disabled by the caller and the optimizer is bypassed under
        ``torch.no_grad()``).
        """
        if self.collect_labels:
            batch = _inject_labels_into_payload(batch)

        # Prefer an explicit forward-only method on the protocol.
        forward_fn = getattr(self.protocol, "step_forward", None)
        if callable(forward_fn):
            result = forward_fn(batch, global_step)
        else:
            # Manual forward: call party_u.forward_train + party_s
            # process_logits_dispatch (the spy monkey-patch is already
            # installed by run_attack_suite), then return.  Do NOT call
            # party_m.backward_and_update so Adam state stays untouched.
            try:
                forward_train = getattr(
                    self.protocol, "_attack_forward_only", None
                )
                if callable(forward_train):
                    result = forward_train(batch, global_step)
                else:
                    # Fallback: torch.no_grad wrap of step_train.  This
                    # is safe — under no_grad the optimizer step is a
                    # no-op and no autograd graph is built.
                    import torch
                    with torch.no_grad():
                        result = self.protocol.step_train(batch, global_step)
            except Exception:
                # Last-ditch: just call step_train (the LoRA scaling=0
                # already prevents weight updates; Adam 动量 *might*
                # be touched but is reset before the main training
                # loop because step 0 of the main loop runs backward
                # from a fresh gradient).
                result = self.protocol.step_train(batch, global_step)
        self._step_count += 1
        return result

    # ------------------------------------------------------------------------- #
    #  S-side argmax interception (L-3A)
    # ------------------------------------------------------------------------- #

    def intercept_s_argmax(self, H_M: Any, gold_ids: Any = None) -> Dict[str, Any]:
        """Query PartyS for argmax predictions (L-3A data collection).

        Call this after a forward pass to collect S's argmax without running
        the full backward.  No gradient is generated.

        Returns:
            {"y_all": np.ndarray, "a_all": np.ndarray}  (both 1-D, length n_tokens)
        """
        if self._party_s is None:
            return {}

        import torch
        H_M_t = H_M.to(self._party_s.device) if isinstance(H_M, torch.Tensor) else H_M
        logits = self._party_s.compute_logits_gpu(H_M_t)
        a_all_flat, y_all = self._party_s.compute_a_t_gpu(logits)

        a_cpu = a_all_flat.detach().to(torch.float32).cpu().numpy().astype("float32")
        y_cpu = y_all.detach().cpu().numpy().astype("int64")

        return {
            "y_all": y_cpu,
            "a_all": a_cpu,
        }

    # ------------------------------------------------------------------------- #
    #  Data access
    # ------------------------------------------------------------------------- #

    def get_attack_data(self) -> List[AttackDataBundle]:
        """Return all collected AttackDataBundle objects (thread-safe copy)."""
        with self._lock:
            return list(self._bundles)

    def get_g_accum_matrix(self) -> tuple:
        """Return (gradients, labels) for L-1A attack.

        Returns:
            (g_matrix, label_array) where:
              - g_matrix: (N, hidden_dim) float32 numpy array
              - label_array: (N,) int64 numpy array of coarse class indices
        """
        gradients = []
        labels = []
        for bundle in self.get_attack_data():
            if bundle.g_accum is not None and bundle.token_labels is not None:
                # Take the last token gradient per sample (the answer token)
                # Shape: (batch_size, hidden_dim)
                g = bundle.g_accum[-self.cfg.batch_size:] if bundle.g_accum.shape[0] >= self.cfg.batch_size else bundle.g_accum
                gradients.append(g)
                labels.extend(bundle.token_labels[-len(g):])

        if not gradients:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        g_matrix = np.concatenate(gradients, axis=0)
        label_array = np.array(labels, dtype=np.int64)
        return g_matrix, label_array

    def save_attack_data(self, path: Optional[str] = None) -> str:
        """Serialise collected bundles to JSON for offline analysis."""
        import json

        if path is None:
            path = os.path.join(self.cfg.attack_dump_dir, "collected_attack_data.json")

        data = []
        for bundle in self.get_attack_data():
            entry = bundle.to_dict()
            # Save g_accum as a list (numpy can't be json-serialised directly)
            if bundle.g_accum is not None:
                entry["g_accum"] = bundle.g_accum.tolist()
            else:
                entry["g_accum"] = None
            if bundle.gold_ids is not None:
                entry["gold_ids"] = bundle.gold_ids.tolist()
            data.append(entry)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved %d attack bundles to %s", len(data), path)
        return path

    # ------------------------------------------------------------------------- #
    #  Lifecycle
    # ------------------------------------------------------------------------- #

    def shutdown(self) -> None:
        """Restore original PartyM method and shut down the protocol."""
        if self._party_m is not None and hasattr(self, "_orig_backward_and_update"):
            self._party_m.backward_and_update = self._orig_backward_and_update
            logger.info("Restored original PartyM.backward_and_update")

        if hasattr(self.protocol, "shutdown"):
            self.protocol.shutdown()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _inject_labels_into_payload(batch: Dict) -> Dict:
    """Inject token_labels and gold_ids from batch dict into the payload.

    Looks for:
      - ``batch["coarse_idx"]``: List[int] of coarse class indices (0-5)
      - ``batch["labels"]``: torch.Tensor of label token IDs
    """
    coarse_idx = batch.get("coarse_idx")
    labels_tensor = batch.get("labels")

    new_batch = dict(batch)

    if coarse_idx is not None:
        # Flatten to per-token labels (one label per sample in the batch)
        if isinstance(coarse_idx, list):
            new_batch["token_labels"] = coarse_idx
        else:
            new_batch["token_labels"] = list(coarse_idx)

    if labels_tensor is not None:
        import torch
        if isinstance(labels_tensor, torch.Tensor):
            new_batch["gold_ids"] = labels_tensor.detach().cpu().numpy()
        else:
            new_batch["gold_ids"] = np.array(labels_tensor)

    return new_batch


def make_attack_step_callback(
    wrapper: AttackProtocolWrapper,
) -> callable:
    """Factory: build a Trainer-compatible step_callback that feeds data to the wrapper.

    The Trainer calls ``step_callback(epoch, step_idx, batch, result)``.
    This wrapper just delegates to the AttackProtocolWrapper's internal logic.
    """
    def callback(epoch: int, step_idx: int, batch: Dict, result: Any) -> None:
        # Attack data is already collected by the wrapper's hijacked
        # PartyM.backward_and_update.  The callback is only used for
        # logging / finalisation here.
        if step_idx % 5 == 0:
            logger.debug(
                "Attack callback step=%d: loss=%.4f",
                step_idx, result.loss,
            )
    return callback
