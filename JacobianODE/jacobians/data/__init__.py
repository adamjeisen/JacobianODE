"""Data utilities for JacobianODE: trajectory generation, processing, and dataloaders."""

from __future__ import annotations

from .trajectory import make_trajectories, make_dysts_trajectories
from .processing import (
    postprocess_data,
    postprocess_per_condition,
    normalize_data,
    PostprocessResult,
    PostprocessPerConditionResult,
)
from .dataloaders import create_dataloaders
from .splitting import generate_train_and_test_sets, TimeSeriesDataset, embed_signal_torch, collate_with_optional_condition
from .filtering import filter_data
from .ragged import (
    delay_embed_ragged,
    pad_trajs_to_max,
    sliding_windows,
    split_balanced_by_timepoints,
    truncate_chronological_balanced,
)
from .types import TimeSeriesData

__all__ = [
    "make_trajectories",
    "make_dysts_trajectories",
    "postprocess_data",
    "postprocess_per_condition",
    "PostprocessResult",
    "PostprocessPerConditionResult",
    "normalize_data",
    "create_dataloaders",
    "generate_train_and_test_sets",
    "TimeSeriesDataset",
    "embed_signal_torch",
    "collate_with_optional_condition",
    "filter_data",
    "TimeSeriesData",
    "pad_trajs_to_max",
    "sliding_windows",
    "delay_embed_ragged",
    "split_balanced_by_timepoints",
    "truncate_chronological_balanced",
]
