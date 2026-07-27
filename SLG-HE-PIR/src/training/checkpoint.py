"""
Checkpoint management for SLG-HE-PIR v2.0.

Handles saving and loading of U/M/S party checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

__all__ = ["CheckpointManager", "load_best_checkpoint", "save_checkpoint"]

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint lifecycle: save, load, best tracking."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save(
        self,
        epoch: int,
        metrics: Dict,
        party_checkpoints: Dict,
        is_best: bool = False,
    ) -> str:
        """Save a checkpoint.

        Args:
            epoch: epoch number
            metrics: dict of metrics
            party_checkpoints: {"U": ..., "M": ..., "S": ...}
            is_best: whether this is the best checkpoint so far

        Returns:
            path to saved checkpoint
        """
        ckpt = {
            "epoch": epoch,
            "metrics": metrics,
            "party_checkpoints": party_checkpoints,
        }

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(ckpt, path)
        logger.info("Checkpoint saved: %s", path)

        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(ckpt, best_path)
            logger.info("Best checkpoint updated: %s", best_path)

        self._prune_old_checkpoints()
        return str(path)

    def load(self, path: str) -> Dict:
        """Load a checkpoint from disk."""
        logger.info("Loading checkpoint: %s", path)
        return torch.load(path, map_location="cpu")

    def load_best(self) -> Dict:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")
        return self.load(str(best_path))

    def list_checkpoints(self) -> List[Dict]:
        """List all saved checkpoints."""
        checkpoints = []
        for p in sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt")):
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                checkpoints.append({
                    "path": str(p),
                    "epoch": ckpt.get("epoch", -1),
                    "metrics": ckpt.get("metrics", {}),
                })
            except Exception as e:
                logger.warning("Failed to load %s: %s", p, e)
        return checkpoints

    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        best = self.checkpoint_dir / "best_checkpoint.pt"
        for p in checkpoints[self.max_checkpoints:]:
            if p != best:
                p.unlink()
                logger.debug("Pruned old checkpoint: %s", p)


def load_best_checkpoint(checkpoint_dir: str) -> Dict:
    """Convenience function to load the best checkpoint."""
    manager = CheckpointManager(checkpoint_dir)
    return manager.load_best()


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    metrics: Dict,
    party_checkpoints: Dict,
    is_best: bool = False,
) -> str:
    """Convenience function to save a checkpoint."""
    manager = CheckpointManager(checkpoint_dir)
    return manager.save(epoch, metrics, party_checkpoints, is_best=is_best)
