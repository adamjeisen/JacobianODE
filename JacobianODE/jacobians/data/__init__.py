"""Data utilities for JacobianODE: trajectory generation, processing, and dataloaders."""

from __future__ import annotations

from .trajectory import make_trajectories, make_dysts_trajectories
from .processing import postprocess_data, normalize_data, PostprocessResult
from .dataloaders import create_dataloaders
from .splitting import generate_train_and_test_sets, TimeSeriesDataset, embed_signal_torch
from .filtering import filter_data
from .types import TimeSeriesData

__all__ = [
    "make_trajectories",
    "make_dysts_trajectories",
    "postprocess_data",
    "PostprocessResult",
    "normalize_data",
    "create_dataloaders",
    "generate_train_and_test_sets",
    "TimeSeriesDataset",
    "embed_signal_torch",
    "filter_data",
    "TimeSeriesData",
]
