#!/usr/bin/env python3
"""
1-step gradient-flow diagnostic.

Reuses the same initialization pipeline as ``two_epoch_test.py`` (BFV cache,
hint table, HeterogeneousProtocol, Trainer) but:

  * Monkey-patches ``PartyM.backward_and_update`` so it prints every trainable
    parameter's ``.grad`` (norm or None) and ``H_M.requires_grad`` after the
    backward pass.
  * Stops after the very first training step; does not run the full training
    loop, no checkpoints written, no val epoch.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 \
        python -m scripts.function_tests.diag_grad_flow \
            --batch-size 4 --max-length 128 --lam 10

Output:
    /tmp/diag_grad_flow.log

Expected (under the Stage 0 hypothesis — gradient flow killed by
``H_U.detach()``):

    [DIAG-grad] trainable_with_grad=<small>, total_grad_norm=<small-but-non-zero>
                 trainable_with_grad=None=<almost all>
    [DIAG-tensor] H_M.requires_grad=True H_U.requires_grad=False

After the Stage 1 fix (remove ``.detach()``), the expected output flips:

    [DIAG-grad] trainable_with_grad=<large>, total_grad_norm=<large>
                 trainable_with_grad=None=0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow running from repo root.
ROOT = Path("/root/autodl-tmp/SLG-HE-PIR")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Reuse helpers from the existing driver so we don't drift.
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.function_tests import two_epoch_test  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("/tmp/diag_grad_flow.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("diag_grad_flow")

DIAG_LOG = Path("/tmp/diag_grad_flow.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1-step gradient-flow diagnostic")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lam", type=int, default=10)
    p.add_argument("--process-mode", choices=["fusion", "legacy"],
                   default="fusion")
    return p.parse_args()


def patch_party_m_with_diag() -> None:
    """Monkey-patch PartyM.backward_and_update to dump grad info after each step."""
    from src.parties import party_m as _party_m_mod  # noqa: E402

    orig = _party_m_mod.PartyM.backward_and_update

    def patched(self, payload):
        result = orig(self, payload)

        n_with = 0
        n_none = 0
        total_norm = 0.0
        per_layer_summary = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                n_none += 1
            else:
                gnorm = float(p.grad.norm().item())
                n_with += 1
                total_norm += gnorm
                # Aggregate per-layer for compactness.
                if "." in n:
                    layer_idx = n.split(".")[1] if n.split(".")[0] == "layers" else "?"
                    per_layer_summary.setdefault(
                        layer_idx, {"with": 0, "none": 0}
                    )["with"] += 1
                if n_with <= 5:
                    logger.info("[DIAG-grad-sample] %s norm=%.4e",
                                n, gnorm)
        H_M = getattr(self, "_last_H_M", None)
        H_M_req = bool(H_M.requires_grad) if H_M is not None else None
        H_U = getattr(self, "_last_H_U", None)
        H_U_req = bool(H_U.requires_grad) if H_U is not None else None

        logger.info(
            "[DIAG-grad] trainable_with_grad=%d trainable_with_gradNone=%d "
            "total_grad_norm=%.4e",
            n_with, n_none, total_norm,
        )
        logger.info(
            "[DIAG-tensor] H_M.requires_grad=%s H_U.requires_grad=%s",
            H_M_req, H_U_req,
        )
        # Per-layer roll-up (key layers first/last few only).
        sorted_keys = sorted(per_layer_summary.keys(),
                            key=lambda x: (x == "?", x))
        shown = sorted_keys[:3] + (["..."] if len(sorted_keys) > 6 else []) \
                + sorted_keys[-3:]
        for k in shown:
            if k == "...":
                logger.info("[DIAG-grad-per-layer] ...")
            else:
                logger.info("[DIAG-grad-per-layer] layer=%s %s",
                            k, per_layer_summary[k])

        # U-side
        try:
            u = self.__class__  # diagnostic, not used
        except Exception:
            pass

        # Cross-check U-side params
        try:
            from src.parties.heterogeneous_protocol import (
                HeterogeneousProtocol,  # noqa: F401
            )
            ipc = self._ipc_for_diag if hasattr(self, "_ipc_for_diag") else None
        except Exception:
            ipc = None

        return result

    _party_m_mod.PartyM.backward_and_update = patched


def attach_u_grad_dump(ipc) -> None:
    """Also dump U-side trainable param grads after every step."""
    from src.parties import party_u as _pu_mod  # noqa: E402

    orig = _pu_mod.PartyU.forward_train

    def patched_u(self, batch):
        result = orig(self, batch)
        # After forward, dump U's trainable params state (their .grad will
        # only be populated after backward — we re-check after M.backward
        # by hooking PartyU.step_optimizer instead).
        n = 0
        for pname, p in self.model.named_parameters():
            if p.requires_grad:
                n += 1
        logger.info("[DIAG-u-forward] U trainable params=%d", n)
        return result

    _pu_mod.PartyU.forward_train = patched_u

    orig_step = _pu_mod.PartyU.step_optimizer
    def patched_step(self):
        n_with = 0
        n_none = 0
        for pname, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                n_none += 1
            else:
                n_with += 1
                gnorm = float(p.grad.norm().item())
                logger.info("[DIAG-u-grad-sample] %s norm=%.4e", pname, gnorm)
        logger.info("[DIAG-u-step_opt] with_grad=%d none_grad=%d", n_with, n_none)
        return orig_step(self)
    _pu_mod.PartyU.step_optimizer = patched_step


class StopAfterFirstStep(Exception):
    """Sentinel to bail out of Trainer.train() after one step."""


def main() -> None:
    args = parse_args()
    logger.info("==== diag_grad_flow starting ====")
    logger.info("args=%s", vars(args))

    patch_party_m_with_diag()

    # Build cfg same way two_epoch_test does.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lam", type=int, default=10)
    parser.add_argument("--process-mode", default="fusion")
    parser.add_argument("--no-purge", dest="no_purge", action="store_true")
    parser.add_argument("--no-build-db", dest="no_build_db", action="store_true")
    parser.add_argument("--no-build-hints", dest="no_build_hints",
                        action="store_true")
    parser.set_defaults(no_purge=False)
    ns = parser.parse_args([])  # ignore argv
    ns.batch_size = args.batch_size
    ns.max_length = args.max_length
    ns.epochs = 1
    ns.lam = args.lam
    ns.process_mode = args.process_mode
    ns.no_purge = True
    cfg = two_epoch_test.make_fast_cfg(ns)
    cfg.max_epochs = 1
    cfg.log_freq = 1
    cfg.save_freq = 999

    # Hijack Trainer.train to stop after the first step via callback.
    from src.training import trainer as _trainer_mod

    orig_train = _trainer_mod.Trainer.train
    first_step_seen = {"v": False}

    def patched_train(self):
        # Wire a step callback that raises after the first step.
        def _on_first_step(epoch, step_idx, batch, result):
            first_step_seen["v"] = True
            logger.info("[DIAG] first step completed — stopping.")
            raise StopAfterFirstStep()

        prev_cb = self.step_callback
        def cb(epoch, step_idx, batch, result):
            if prev_cb is not None:
                try:
                    prev_cb(epoch, step_idx, batch, result)
                except Exception:
                    pass
            _on_first_step(epoch, step_idx, batch, result)
        self.step_callback = cb

        try:
            return orig_train(self)
        except StopAfterFirstStep:
            return {
                "best_metric": float("nan"),
                "best_epoch": -1,
                "total_steps": 1,
                "elapsed_s": time.time() - self.start_time,
                "metrics_path": "",
                "stopped_early": True,
            }

    _trainer_mod.Trainer.train = patched_train

    # Run.
    results = two_epoch_test.run_stage1_with_hooks(cfg, resume_from=None)

    logger.info("==== diag_grad_flow done ====")
    logger.info("results.total_steps=%s", results.get("n_steps"))
    logger.info("log written to %s", DIAG_LOG)


if __name__ == "__main__":
    main()
