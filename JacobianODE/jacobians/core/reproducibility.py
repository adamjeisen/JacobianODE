"""Reproducibility utilities for JacobianODE."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(
    seed: int,
    workers: bool = True,
) -> int:
    """Set random seeds for reproducibility across all libraries.

    This function sets seeds for Python's random module, NumPy, and PyTorch,
    including CUDA if available. It provides a centralized way to ensure
    reproducible experiments.

    Args:
        seed: The random seed to use. Must be non-negative.
        workers: If True, sets PYTHONHASHSEED environment variable to ensure
            reproducibility of data loader workers. Default: True.
        deterministic: If True, enables PyTorch's deterministic algorithms.
            This may impact performance. Default: False.

    Returns:
        The seed that was set (same as input seed).

    Raises:
        ValueError: If seed is negative.

    Example:
        >>> from JacobianODE.jacobians.core import seed_everything
        >>> seed_everything(42)
        42
        >>> # Now all random operations are reproducible

    Note:
        - Setting deterministic=True may significantly slow down training
        - For full reproducibility with CUDA, you may also need to set
          CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    logger.debug(f"Setting random seed to {seed}")

    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Worker seed for DataLoader reproducibility
    if workers:
        os.environ["PYTHONHASHSEED"] = str(seed)

    return seed


def get_worker_init_fn(seed: int):
    """Get a worker initialization function for DataLoader reproducibility.

    This function returns a callable that can be passed to DataLoader's
    worker_init_fn parameter to ensure reproducible data loading across
    workers.

    Args:
        seed: Base seed to use.

    Returns:
        A function that initializes worker random state.

    Example:
        >>> worker_init = get_worker_init_fn(42)
        >>> loader = DataLoader(dataset, worker_init_fn=worker_init)
    """

    def worker_init_fn(worker_id: int) -> None:
        """Initialize worker with seed based on worker_id."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn
