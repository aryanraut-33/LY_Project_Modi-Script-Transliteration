"""
==============================================================================
utils.py — Shared Utility Functions for MoScNet Pipeline
==============================================================================

PURPOSE:
    Provides reusable helper functions that every other module depends on:
      • Reproducibility  — set_seed() locks all random number generators
      • Device detection — get_device() auto-detects CUDA / MPS / CPU
      • Checkpointing    — save/load model + optimizer state to/from disk
      • Logging          — consistent log format across training & evaluation
      • Diagnostics      — count_parameters() for quick model size checks

WHERE IT IS USED:
    - train.py       → set_seed(), setup_logging(), save_checkpoint(),
                       get_device(), count_parameters()
    - evaluate.py    → load_checkpoint(), get_device()
    - Any module     → logging via the configured logger

HOW IT INTERACTS WITH OTHERS:
    This file has NO dependencies on any other project file. It is a pure
    utility layer that sits beneath the entire pipeline. All other modules
    import from here but never the reverse.

    Dependency graph:
        utils.py  ←──  config.py (only for checkpoint metadata)
            ↑
        train.py, evaluate.py, teacher_model.py, student_model.py, ...
==============================================================================
"""

import torch
import random
import numpy as np
import os
import logging
from typing import Dict, Any, Optional


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    This locks:
      - Python's `random` module
      - NumPy's random generator
      - PyTorch's CPU and CUDA generators
      - cuDNN's deterministic mode (may slow down training slightly)

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """
    Auto-detect the best available compute device.

    Priority: CUDA GPU > Apple MPS > CPU.

    Returns:
        Device string usable with torch.device().
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_logging(log_dir: str, name: str = "MoScNet") -> logging.Logger:
    """
    Configure logging to both file and console.

    Creates the log directory if it doesn't exist. Returns a named logger
    that writes timestamps, severity levels, and messages.

    Args:
        log_dir: Directory to write the log file into.
        name:    Logger name (appears in log lines).

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "train.log")
        )
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_bleu: float,
    path: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint to disk.

    Stores the model weights, optimizer state, current epoch, validation
    BLEU score, and any extra metadata (e.g., student config, loss weights).

    Args:
        model:          The model to save (student or teacher).
        optimizer:      The optimizer whose state to save.
        epoch:          Current epoch number.
        val_bleu:       Best validation BLEU score so far.
        path:           Full file path for the checkpoint (e.g., './checkpoints/best.pt').
        extra_metadata: Optional dict with additional info to store.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_bleu": val_bleu,
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint from disk.

    Restores model weights and optionally optimizer state. Returns a dict
    with 'epoch' and 'val_bleu' for resuming training.

    Args:
        path:      Path to the saved checkpoint file.
        model:     The model to load weights into (must match architecture).
        optimizer: Optional optimizer to restore state into.

    Returns:
        Dict with keys 'epoch', 'val_bleu', and optionally 'metadata'.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint["epoch"],
        "val_bleu": checkpoint["val_bleu"],
        "metadata": checkpoint.get("metadata", {}),
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.

    Useful for verifying student model sizes against the paper:
      - MoScNet-S:  ~15M
      - MoScNet-M:  ~63M
      - MoScNet-L:  ~177M
      - MoScNet-XL: ~429M

    Args:
        model: Any nn.Module.

    Returns:
        Dict with 'total' and 'trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
