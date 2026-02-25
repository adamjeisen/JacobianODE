"""JacobianODE jacobians module.

This module provides utilities for training and using Jacobian-based ODE models.

Public API:
    Core:
        - load_config: Load configuration with optional overrides
        - initialize_config: Initialize and complete configuration setup
        - seed_everything: Set random seeds for reproducibility
        - in_ipython: Check if running in IPython/Jupyter

    Data:
        - make_trajectories: Generate trajectories for training
        - make_dysts_trajectories: Generate dynamical systems trajectories
        - postprocess_data: Apply noise and filtering to data
        - normalize_data: Normalize data (z-score)
        - create_dataloaders: Create PyTorch DataLoaders

    Training:
        - make_model: Create and initialize the model
        - train_model: Train the model using PyTorch Lightning
        - log_training_info: Log training setup information
        - setup_wandb: Set up Weights & Biases logging

    Checkpoints:
        - load_run: Load a previous training run
        - load_checkpoint: Load a specific checkpoint
        - get_all_checkpoints: Get all checkpoints for a run
        - reverse_wandb_run: Reverse engineer a W&B run (legacy)

    Tuning:
        - run_sweep: Run lambda_loop hyperparameter sweep (notebook mode)
        - select_from_wandb_runs: Select best model from W&B runs (post-hoc)
        - select_best_model: Physics-informed model selection
        - DiagnosticMetrics: Dataclass of per-model diagnostic metrics
        - SelectionResult: Dataclass of selection outcome
        - SweepResult: Dataclass of full sweep result
        - DEFAULT_LAMBDA_LOOP_VALUES: Default lambda grid

    Custom Data:
        - CustomDatasetLoader: Loader for custom time series data
        - load_timeseries_data: Load data from TimeSeriesData .npz files (includes dt)
        - load_from_numpy: Load data from numpy files
        - load_from_pickle: Load data from pickle files
        - load_from_array: Load data from numpy arrays
        - create_custom_loader: Create a callable loader
        - validate_data_shape: Validate data has correct shape

Example (standard usage):
    >>> from JacobianODE.jacobians import load_config, initialize_config, seed_everything
    >>> cfg = load_config()
    >>> cfg = initialize_config(cfg)
    >>> seed_everything(42)

Example (in-memory data in notebooks):
    >>> import numpy as np
    >>> from JacobianODE.jacobians import (
    ...     load_config, initialize_config, make_trajectories,
    ...     postprocess_data, create_dataloaders
    ... )
    >>>
    >>> # Your data: shape (n_trials, n_timepoints, n_dimensions)
    >>> my_data = np.random.randn(32, 1000, 3)
    >>> dt = 0.01
    >>>
    >>> # Load config for custom data
    >>> cfg = load_config(overrides=["data=custom"])
    >>> cfg = initialize_config(cfg, data_dim=my_data.shape[-1])
    >>>
    >>> # Pass data directly to make_trajectories (no file I/O needed)
    >>> eq, sol, dt = make_trajectories(cfg, data=my_data, dt=dt)
    >>> values = postprocess_data(cfg, sol["values"])
    >>> train_dl, val_dl, test_dl, trajs = create_dataloaders(cfg, values)
"""

from __future__ import annotations

# Re-export from submodules for convenience
from .checkpoints.legacy import reverse_wandb_config

# Custom data utilities
from .custom_data import (
    CustomDatasetLoader,
    load_timeseries_data,
    load_from_numpy,
    load_from_pickle,
    load_from_array,
    create_custom_loader,
    validate_data_shape,
)

# Core utilities
from .core import (
    load_config,
    initialize_config,
    seed_everything,
    in_ipython,
)

# Data utilities
from .data import (
    make_trajectories,
    make_dysts_trajectories,
    postprocess_data,
    normalize_data,
    create_dataloaders,
    TimeSeriesData,
)

# Training utilities
from .training import (
    make_model,
    train_model,
    log_training_info,
    setup_wandb,
)

# Checkpoint utilities
from .checkpoints import (
    load_run,
    load_checkpoint,
    get_all_checkpoints,
    reverse_wandb_run,
)

# Tuning utilities
from .tuning import (
    run_sweep,
    select_from_wandb_runs,
    select_best_model,
    DiagnosticMetrics,
    SelectionResult,
    SweepResult,
    DEFAULT_LAMBDA_LOOP_VALUES,
)

# Metrics
from .metrics import (
    mase,
    mse,
    mae,
    r2_score,
)
__all__ = [
    # Legacy
    "reverse_wandb_config",
    # Custom data
    "CustomDatasetLoader",
    "load_timeseries_data",
    "load_from_numpy",
    "load_from_pickle",
    "load_from_array",
    "create_custom_loader",
    "validate_data_shape",
    # Core
    "load_config",
    "initialize_config",
    "seed_everything",
    "in_ipython",
    # Data
    "make_trajectories",
    "make_dysts_trajectories",
    "postprocess_data",
    "normalize_data",
    "create_dataloaders",
    "TimeSeriesData",
    # Training
    "make_model",
    "train_model",
    "log_training_info",
    "setup_wandb",
    # Checkpoints
    "load_run",
    "load_checkpoint",
    "get_all_checkpoints",
    "reverse_wandb_run",
    # Tuning
    "run_sweep",
    "select_from_wandb_runs",
    "select_best_model",
    "DiagnosticMetrics",
    "SelectionResult",
    "SweepResult",
    "DEFAULT_LAMBDA_LOOP_VALUES",
]
