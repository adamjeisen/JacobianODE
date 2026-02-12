"""DataLoader creation for JacobianODE."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .splitting import generate_train_and_test_sets

logger = logging.getLogger(__name__)


def create_dataloaders(
    cfg: DictConfig,
    values: Union[np.ndarray, torch.Tensor, "TimeSeriesData"],
    verbose: bool = False,
    num_workers: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create PyTorch DataLoaders for training, validation, and testing.

    Generates DataLoader objects for continuous trajectory data, handling both
    training and validation sets.

    Args:
        cfg: Configuration object containing dataloader parameters.
        values: Input data to create dataloaders from. Shape: (n_traj, time, dim).
            Also accepts a :class:`~.types.TimeSeriesData` instance (uses its ``.values``).
        verbose: Whether to print progress information. Defaults to False.
        num_workers: Number of worker processes for data loading. Defaults to 2.
        persistent_workers: Keep workers alive between epochs. Defaults to True.
        pin_memory: Pin memory for faster GPU transfer. Defaults to True.

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader, trajs) where:
            - train_dataloader: DataLoader for training data
            - val_dataloader: DataLoader for validation data
            - test_dataloader: DataLoader for test data
            - trajs: Dictionary containing trajectory information

    Example:
        >>> train_dl, val_dl, test_dl, trajs = create_dataloaders(cfg, values)
        >>> for batch in train_dl:
        ...     # batch shape: (batch_size, seq_length, dim)
        ...     pass

    Note:
        The function filters cfg.data.train_test_params to only include
        valid parameters for generate_train_and_test_sets.
    """
    # Unwrap TimeSeriesData if provided
    from .types import TimeSeriesData
    if isinstance(values, TimeSeriesData):
        values = values.values

    cfg.data.train_test_params.verbose = verbose

    # Filter to only valid parameters
    valid_keys = set(inspect.signature(generate_train_and_test_sets).parameters.keys())
    del_keys = [
        key
        for key in cfg.data.train_test_params.keys()
        if key not in valid_keys
    ]
    for key in del_keys:
        del cfg.data.train_test_params[key]

    train_dataset, val_dataset, test_dataset, trajs = generate_train_and_test_sets(
        values, **cfg.data.train_test_params
    )

    batch_size = cfg.training.batch_size

    # Create continuous trajectory dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    if verbose:
        logger.info(f"Created dataloaders:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Val: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        logger.info(f"  Batch size: {batch_size}")

    return train_dataloader, val_dataloader, test_dataloader, trajs
