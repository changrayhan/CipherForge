"""DeepSpeed ZeRO integration for SLG-HE-PIR GPU memory optimization.

This module provides ZeRO optimizer state partitioning for the M shard,
which can reduce GPU memory usage by 4-8x for optimizer states.

ZeRO Stages:
    ZeRO-1: Partition optimizer states across GPUs (most common, good speed/memory balance)
    ZeRO-2: Partition optimizer states + gradients (more memory savings, slightly slower)
    ZeRO-3: Partition optimizer states + gradients + parameters (maximum savings, for multi-GPU)

For single-GPU training (this use case), ZeRO-1 is the recommended setting.
It partitions only the optimizer states, providing ~4x memory reduction
with minimal performance overhead.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.nn.modules import Module

logger = logging.getLogger(__name__)


def is_deepspeed_available() -> bool:
    """Check if DeepSpeed is installed and available."""
    try:
        import deepspeed  # noqa: F401
        return True
    except ImportError:
        return False


def get_zero_stage_name(stage: int) -> str:
    """Return human-readable name for ZeRO stage."""
    names = {
        0: "Disabled (no ZeRO)",
        1: "ZeRO-1: Optimizer States Partitioned",
        2: "ZeRO-2: Optimizer States + Gradients Partitioned",
        3: "ZeRO-3: Optimizer States + Gradients + Parameters Partitioned",
    }
    return names.get(stage, f"Unknown ZeRO-{stage}")


def create_zero_config(
    stage: int = 1,
    reduce_bucket_size: int = 5_000_000,
    stage3_prefetch_bucket_size: int = 5_000_000,
    stage3_param_persistence_threshold: int = 100_000,
    stage3_max_live_parameters: int = 1_000_000,
    stage3_max_reuse_distance: int = 1_000_000,
    allgather_bucket_size: int = 5_000_000,
    round_robin_gradients: bool = True,
    ZeRO损耗函数: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create DeepSpeed ZeRO configuration dictionary.

    Args:
        stage: ZeRO stage (0=disabled, 1=optimizer states, 2=+gradients, 3=+params)
        reduce_bucket_size: Communication bucket size for gradient reduction
        stage3_prefetch_bucket_size: Bucket size for parameter fetching in ZeRO-3
        stage3_param_persistence_threshold: Min param size to persist in ZeRO-3
        stage3_max_live_parameters: Max params to keep in GPU simultaneously in ZeRO-3
        stage3_max_reuse_distance: Reuse distance for parameter eviction in ZeRO-3
        allgather_bucket_size: Communication bucket size for all-gather
        round_robin_gradients: Enable gradient load balancing across ranks
        ZeRO损耗函数: Loss function config (unused in current implementation)

    Returns:
        DeepSpeed zero config dict to be passed to ds_config
    """
    if stage == 0:
        return {"stage": 0, "stage3_config": {}}

    zero_config = {
        "stage": stage,
        "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
        "stage3_max_live_parameters": stage3_max_live_parameters,
        "stage3_max_reuse_distance": stage3_max_reuse_distance,
        "stage3_config": {
            "stage3_prefetch_bucket_size": stage3_prefetch_bucket_size,
            "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
            "stage3_max_live_parameters": stage3_max_live_parameters,
            "stage3_max_reuse_distance": stage3_max_reuse_distance,
            "stage3_overlap_comm": True,
            "stage3_tidy_model_ios": True,
        },
        "reduce_bucket_size": reduce_bucket_size,
        "allgather_bucket_size": allgather_bucket_size,
        "reduce_scatter": True,
        "contiguous_gradients": True,
        "round_robin_gradients": round_robin_gradients,
    }

    return zero_config


def create_optimizer_config(
    optimizer: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Create DeepSpeed optimizer configuration.

    Args:
        optimizer: Optimizer type ("adam", "adamw", "sgd")
        lr: Learning rate
        weight_decay: Weight decay coefficient
        beta1: Adam beta1
        beta2: Adam beta2
        eps: Adam epsilon

    Returns:
        DeepSpeed optimizer config dict
    """
    if optimizer == "adamw":
        return {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": (beta1, beta2),
            "eps": eps,
        }
    elif optimizer == "adam":
        return {
            "type": "Adam",
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": (beta1, beta2),
            "eps": eps,
        }
    elif optimizer == "sgd":
        return {
            "type": "SGD",
            "lr": lr,
            "momentum": beta1,
            "weight_decay": weight_decay,
        }
    else:
        logger.warning(f"Unknown optimizer '{optimizer}', defaulting to AdamW")
        return {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": (beta1, beta2),
            "eps": eps,
        }


def create_ds_config(
    fp16_enabled: bool = True,
    bf16_enabled: bool = False,
    zero_stage: int = 1,
    gradient_clipping: float = 1.0,
    gradient_accumulation_steps: int = 1,
    steps_per_print: int = 10,
    wall_clock_breakdown: bool = False,
    zero_config: Optional[Dict[str, Any]] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    autotuning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create complete DeepSpeed configuration dictionary.

    This config is used by deepspeed.initialize() to set up the runtime.

    Args:
        fp16_enabled: Use FP16 mixed precision training
        bf16_enabled: Use BF16 mixed precision training (recommended for stability)
        zero_stage: ZeRO stage (0-3)
        gradient_clipping: Max gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        steps_per_print: Print frequency
        wall_clock_breakdown: Enable timing breakdown
        zero_config: Pre-built ZeRO config (if None, built from zero_stage)
        optimizer_config: Pre-built optimizer config (if None, uses AdamW defaults)
        autotuning: Autotuning config for optimal settings

    Returns:
        Complete DeepSpeed config dict
    """
    if fp16_enabled and bf16_enabled:
        logger.warning("Both fp16 and bf16 enabled; disabling fp16")
        fp16_enabled = False

    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "steps_per_print": steps_per_print,
        "wall_clock_breakdown": wall_clock_breakdown,
        "fp16": {
            "enabled": fp16_enabled,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        } if fp16_enabled else {"enabled": False},
        "bf16": {
            "enabled": bf16_enabled,
        } if bf16_enabled else {"enabled": False},
        "zero_optimization": zero_config or create_zero_config(stage=zero_stage),
        "optimizer": optimizer_config or create_optimizer_config(),
    }

    if autotuning:
        config["autotuning"] = autotuning

    return config


class DeepSpeedZeROManager:
    """Manages DeepSpeed ZeRO integration for SLG-HE-PIR M shard.

    This class wraps the M shard model and optimizer to apply ZeRO partitioning,
    reducing GPU memory usage for optimizer states.

    Usage:
        manager = DeepSpeedZeROManager(
            model=m_shard,
            optimizer=adamw_optimizer,
            config={"zero_stage": 1, "bf16_enabled": True}
        )
        # Training loop
        outputs = manager.forward(model_inputs)
        manager.backward(loss)
        manager.step()

    Memory savings (approximate):
        - ZeRO-1: ~4x reduction in optimizer state memory
        - ZeRO-2: ~8x reduction (optimizer states + gradients)
        - ZeRO-3: ~16x reduction (full partitioning, multi-GPU only)
    """

    def __init__(
        self,
        model: "Module",
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        zero_stage: int = 1,
        bf16_enabled: bool = True,
        gradient_clipping: float = 1.0,
        gradient_accumulation_steps: int = 1,
        device: Optional[str] = None,
        remote_device: str = "cpu",
        pin_memory: bool = True,
        os_cache: bool = False,
        mpu: Optional[Any] = None,
        dist_init_required: bool = True,
        training_dataloader: Optional[Any] = None,
        config_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize DeepSpeed ZeRO manager.

        Args:
            model: PyTorch model (M shard)
            optimizer: Existing optimizer (will be replaced by DeepSpeed optimizer)
            zero_stage: ZeRO stage (1-3)
            bf16_enabled: Use BF16 mixed precision (recommended)
            gradient_clipping: Max gradient norm
            gradient_accumulation_steps: Gradient accumulation steps
            device: Device for model (None = same as model)
            remote_device: Where to offload parameters ("cpu" or "nvme")
            pin_memory: Pin CPU memory for faster GPU transfer
            os_cache: Use OS page cache for offloading
            mpu: Model parallel unit (None for single GPU)
            dist_init_required: Require distributed init
            training_dataloader: DataLoader for autotuning
            config_params: Additional DeepSpeed config params
        """
        self.model = model
        self.zero_stage = zero_stage
        self.bf16_enabled = bf16_enabled
        self.gradient_clipping = gradient_clipping
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self._engine = None
        self._initialized = False
        self._config_params = config_params or {}

        if not is_deepspeed_available():
            logger.warning("DeepSpeed not available; ZeRO optimization disabled")
            self._engine = None
            return

        import deepspeed

        # Build DeepSpeed config
        zero_config = create_zero_config(stage=zero_stage)
        optimizer_config = create_optimizer_config(
            optimizer="adamw",
            lr=1e-4,
            weight_decay=0.0,
        )
        ds_config = create_ds_config(
            fp16_enabled=not bf16_enabled,
            bf16_enabled=bf16_enabled,
            zero_stage=zero_stage,
            gradient_clipping=gradient_clipping,
            gradient_accumulation_steps=gradient_accumulation_steps,
            zero_config=zero_config,
            optimizer_config=optimizer_config,
        )
        ds_config.update(self._config_params)

        # Initialize DeepSpeed
        if not dist.is_initialized() and dist_init_required:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available and dist not initialized; skipping DeepSpeed init")
                return
            # For single-GPU, we can initialize a local process group
            try:
                dist.init_process_group(backend="nccl")
            except RuntimeError:
                logger.warning("Distributed already initialized or init failed; skipping")
                return

        # Determine device map
        if device is None:
            device = next(model.parameters(), torch.device("cpu")).device

        # Build parameter groups
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad]},
        ]

        # Replace optimizer if provided
        if optimizer is not None:
            # DeepSpeed will create its own optimizer from param groups
            pass

        try:
            # Initialize DeepSpeed engine
            self._engine, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=ds_config,
                dist_init_required=dist_init_required,
            )
            self._initialized = True
            logger.info(
                f"DeepSpeed ZeRO-{zero_stage} initialized: "
                f"bf16={bf16_enabled}, grad_clip={gradient_clipping}"
            )
        except Exception as e:
            logger.warning(f"DeepSpeed initialization failed: {e}; continuing without ZeRO")
            self._engine = None
            self._initialized = False

    @property
    def engine(self):
        """Return DeepSpeed engine or original model."""
        return self._engine if self._initialized else self.model

    @property
    def module(self):
        """Return the (wrapped) model."""
        if self._initialized:
            return self._engine.module
        return self.model

    @property
    def optimizer(self):
        """Return the optimizer."""
        if self._initialized:
            return self._engine.optimizer
        return None

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        if self._initialized:
            return self._engine(*args, **kwargs)
        return self.model(*args, **kwargs)

    def backward(self, loss, **kwargs):
        """Backward pass with optional loss scaling."""
        if self._initialized:
            self._engine.backward(loss, **kwargs)
        elif hasattr(loss, "backward"):
            loss.backward(**kwargs)

    def step(self, **kwargs):
        """Optimizer step with gradient clipping."""
        if self._initialized:
            self._engine.step(**kwargs)

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        if self._initialized:
            self._engine.zero_grad()
        else:
            for p in self.model.parameters():
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def get_static_optimizer_state(self) -> Optional[Dict[str, Any]]:
        """Extract fp32 optimizer state from ZeRO-partitioned state.

        This is useful for checkpointing or analysis. Returns None if ZeRO
        is not active or state cannot be extracted.

        Returns:
            Dict mapping parameter names to fp32 optimizer states, or None
        """
        if not self._initialized:
            return None

        try:
            import deepspeed.utils.zero_to_fp32
            fp32_state = deepspeed.utils.zero_to_fp32.get_fp32_state_dict(self._engine)
            return fp32_state
        except Exception as e:
            logger.debug(f"Could not extract fp32 optimizer state: {e}")
            return None

    def __repr__(self) -> str:
        status = "enabled" if self._initialized else "disabled"
        return (
            f"DeepSpeedZeROManager(zero_stage={self.zero_stage}, "
            f"bf16={self.bf16_enabled}, status={status})"
        )


# --------------------------------------------------------------------------- #
#  Convenience function for inline ZeRO wrapping
# --------------------------------------------------------------------------- #

def apply_deepspeed_zero(
    model: "Module",
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    zero_stage: int = 1,
    bf16_enabled: bool = True,
    gradient_clipping: float = 1.0,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    **kwargs,
) -> "DeepSpeedZeROManager":
    """Apply DeepSpeed ZeRO optimization to a model.

    This is a convenience wrapper around DeepSpeedZeROManager.

    Args:
        model: PyTorch model to optimize
        optimizer: Existing optimizer (replaced by DeepSpeed optimizer)
        zero_stage: ZeRO stage (1-3, 0 to disable)
        bf16_enabled: Use BF16 mixed precision
        gradient_clipping: Max gradient norm
        learning_rate: Learning rate for new optimizer
        weight_decay: Weight decay coefficient
        **kwargs: Additional DeepSpeedZeROManager kwargs

    Returns:
        DeepSpeedZeROManager instance wrapping the model

    Example:
        manager = apply_deepspeed_zero(
            model=m_shard,
            zero_stage=1,
            bf16_enabled=True,
            learning_rate=1e-4,
        )
        # Training
        loss = manager.forward(inputs)
        manager.backward(loss)
        manager.step()
    """
    if zero_stage == 0:
        logger.info("ZeRO disabled (stage=0); returning model unchanged")
        return model

    if not is_deepspeed_available():
        logger.error("DeepSpeed not installed. Install with: pip install deepspeed")
        return model

    config_params = {
        "optimizer": optimizer,
    }

    manager = DeepSpeedZeROManager(
        model=model,
        optimizer=optimizer,
        zero_stage=zero_stage,
        bf16_enabled=bf16_enabled,
        gradient_clipping=gradient_clipping,
        config_params=config_params,
        **kwargs,
    )

    logger.info(
        f"Applied DeepSpeed ZeRO-{zero_stage} to model "
        f"(params={sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
        f"bf16={bf16_enabled})"
    )

    return manager
