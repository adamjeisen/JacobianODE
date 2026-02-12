"""Sweep orchestration for hyperparameter tuning.

Two modes:
  1. Notebook mode (``run_sweep``): train models from scratch for each lambda.
  2. Post-hoc W&B mode (``select_from_wandb_runs``): load already-trained runs.

Uses existing ``make_model``, ``train_model``, ``load_run``, ``load_checkpoint``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .criteria import (
    DiagnosticMetrics,
    compute_all_diagnostics,
    compute_persistence_baseline,
)
from .selection import SelectionResult, select_best_model

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA_LOOP_VALUES: List[float] = [
    0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
]


@dataclass
class SweepResult:
    """Result of a hyperparameter sweep.

    Attributes:
        selection: The SelectionResult from model selection.
        all_diagnostics: DiagnosticMetrics for each lambda value.
        lambda_values: The lambda_loop values that were swept.
        persistence_baseline: The persistence baseline used for C1.
        best_lit_model: The best Lightning model (only in notebook mode).
        run_ids: W&B run IDs (only in post-hoc mode).
    """

    selection: SelectionResult
    all_diagnostics: List[DiagnosticMetrics]
    lambda_values: List[float]
    persistence_baseline: float
    best_lit_model: Optional[Any] = None
    run_ids: Optional[List[str]] = None


def run_sweep(
    cfg: DictConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_trajs: torch.Tensor,
    dt: float,
    n_dims: int,
    lambda_values: Optional[List[float]] = None,
    n_batches: int = 100,
    eigenvalue_threshold: float = 0.001,
    use_loop_closure: bool = True,
    eq: Optional[Any] = None,
    project: str = "tuning-sweep",
    entity: Optional[str] = None,
    mu: float = 0.0,
    sigma: float = 1.0,
    verbose: bool = False,
) -> SweepResult:
    """Run a full hyperparameter sweep over lambda_loop values (notebook mode).

    For each lambda value: deep-copies cfg, sets ``loop_closure_weight``,
    creates model, trains, loads best checkpoint, and computes diagnostics.

    Args:
        cfg: Base Hydra config (will be deep-copied per lambda).
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        train_trajs: Training trajectories tensor for persistence baseline.
        dt: Time step of the data.
        n_dims: State-space dimensionality.
        lambda_values: Lambda values to sweep (default: DEFAULT_LAMBDA_LOOP_VALUES).
        n_batches: Number of validation batches for diagnostics.
        eigenvalue_threshold: Threshold for eigenvalue criterion.
        use_loop_closure: Whether to apply loop closure criterion.
        eq: Equation object (passed to make_model).
        project: W&B project name for logging.
        entity: W&B entity for logging.
        mu: Data mean for normalization.
        sigma: Data std for normalization.
        verbose: Whether to print progress.

    Returns:
        SweepResult with selection, diagnostics, and best model.
    """
    from ..training import make_model, train_model

    if lambda_values is None:
        lambda_values = list(DEFAULT_LAMBDA_LOOP_VALUES)

    persistence = compute_persistence_baseline(train_trajs)
    all_diagnostics: List[DiagnosticMetrics] = []
    lit_models: List[Any] = []

    for lam in lambda_values:
        if verbose:
            logger.info(f"Training with loop_closure_weight={lam}")

        # Deep-copy config and set lambda
        sweep_cfg = copy.deepcopy(cfg)
        OmegaConf.set_struct(sweep_cfg, False)
        sweep_cfg.training.lightning.loop_closure_weight = lam
        OmegaConf.set_struct(sweep_cfg, True)

        # Create and train model
        lit_model = make_model(sweep_cfg, dt, eq=eq, mu=mu, sigma=sigma, verbose=verbose)
        name = f"sweep_lambda_{lam}"
        train_model(
            sweep_cfg,
            lit_model,
            train_dataloader,
            val_dataloader,
            name=name,
            project=project,
            entity=entity,
        )

        # Evaluate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lit_model.eval()
        lit_model = lit_model.to(device)

        metrics = compute_all_diagnostics(
            lit_model, val_dataloader, dt,
            n_batches=n_batches, use_loop_closure=use_loop_closure,
        )
        all_diagnostics.append(metrics)
        lit_models.append(lit_model)

        if verbose:
            logger.info(f"  lambda={lam}: {metrics}")

        # Free GPU memory
        lit_model.cpu()
        torch.cuda.empty_cache()

    selection = select_best_model(
        all_diagnostics,
        persistence,
        n_dims,
        eigenvalue_threshold=eigenvalue_threshold,
        use_loop_closure=use_loop_closure,
    )

    best_model = None
    if selection.best_index is not None:
        best_model = lit_models[selection.best_index]

    return SweepResult(
        selection=selection,
        all_diagnostics=all_diagnostics,
        lambda_values=lambda_values,
        persistence_baseline=persistence,
        best_lit_model=best_model,
    )


def select_from_wandb_runs(
    run_ids: List[str],
    project: str,
    train_trajs: torch.Tensor,
    dt: float,
    n_dims: int,
    n_batches: int = 100,
    eigenvalue_threshold: float = 0.001,
    use_loop_closure: bool = True,
    lambda_values: Optional[List[float]] = None,
    verbose: bool = False,
) -> SweepResult:
    """Select the best model from already-trained W&B runs (post-hoc mode).

    For each run: calls ``load_run`` and ``load_checkpoint``, then computes
    diagnostics and runs the selection algorithm.

    Args:
        run_ids: List of W&B run IDs to evaluate.
        project: W&B project name.
        train_trajs: Training trajectories tensor for persistence baseline.
        dt: Time step of the data.
        n_dims: State-space dimensionality.
        n_batches: Number of validation batches for diagnostics.
        eigenvalue_threshold: Threshold for eigenvalue criterion.
        use_loop_closure: Whether to apply loop closure criterion.
        lambda_values: Lambda values corresponding to each run (for reporting).
        verbose: Whether to print progress.

    Returns:
        SweepResult with selection, diagnostics, and run IDs.
    """
    from ..checkpoints import load_run, load_checkpoint

    persistence = compute_persistence_baseline(train_trajs)
    all_diagnostics: List[DiagnosticMetrics] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_dataloader = None
    for i, run_id in enumerate(run_ids):
        if verbose:
            logger.info(f"Loading run {run_id} ({i+1}/{len(run_ids)})")

        # First run: generate data; subsequent runs reuse it
        generate_data = (i == 0)
        run_obj, run_cfg, eq, run_dt, values, train_dl, val_dl, test_dl, trajs, lit_model = load_run(
            project, run_id=run_id, generate_data=generate_data, dt=dt, verbose=verbose,
        )
        if val_dataloader is None:
            val_dataloader = val_dl

        load_checkpoint(run_obj, run_cfg, lit_model, verbose=verbose)

        lit_model.eval()
        lit_model = lit_model.to(device)

        metrics = compute_all_diagnostics(
            lit_model, val_dataloader, dt,
            n_batches=n_batches, use_loop_closure=use_loop_closure,
        )
        all_diagnostics.append(metrics)

        if verbose:
            logger.info(f"  run={run_id}: {metrics}")

        lit_model.cpu()
        torch.cuda.empty_cache()

    selection = select_best_model(
        all_diagnostics,
        persistence,
        n_dims,
        eigenvalue_threshold=eigenvalue_threshold,
        use_loop_closure=use_loop_closure,
    )

    if lambda_values is None:
        lambda_values = list(range(len(run_ids)))

    return SweepResult(
        selection=selection,
        all_diagnostics=all_diagnostics,
        lambda_values=lambda_values,
        persistence_baseline=persistence,
        run_ids=run_ids,
    )
