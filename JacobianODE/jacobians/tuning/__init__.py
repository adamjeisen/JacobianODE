"""Hyperparameter tuning for JacobianODE: criteria, selection, and sweep orchestration."""

from __future__ import annotations

from .criteria import DiagnosticMetrics, diagnostics_from_wandb
from .selection import SelectionResult, select_best_model
from .sweep import (
    DEFAULT_LAMBDA_LOOP_VALUES,
    DiscoveredSweep,
    SweepResult,
    discover_sweep_runs,
    run_sweep,
    select_best_from_sweep,
    select_from_wandb_runs,
)

__all__ = [
    "DEFAULT_LAMBDA_LOOP_VALUES",
    "DiagnosticMetrics",
    "DiscoveredSweep",
    "SelectionResult",
    "SweepResult",
    "diagnostics_from_wandb",
    "discover_sweep_runs",
    "run_sweep",
    "select_best_from_sweep",
    "select_best_model",
    "select_from_wandb_runs",
]
