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
        best_lit_model: The best Lightning model (only in notebook mode).
        run_ids: W&B run IDs (only in post-hoc mode).
    """

    selection: SelectionResult
    all_diagnostics: List[DiagnosticMetrics]
    lambda_values: List[float]
    best_lit_model: Optional[Any] = None
    run_ids: Optional[List[str]] = None

    @property
    def diagnostics(self) -> List[DiagnosticMetrics]:
        """Alias for all_diagnostics (e.g. sweep_result.diagnostics)."""
        return self.all_diagnostics


def run_sweep(
    cfg: DictConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
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
        best_lit_model=best_model,
    )


def _diagnostics_cache_path(save_dir: str, run_id: str, n_batches: int) -> str:
    """Path to per-run cached diagnostics. Enables reuse across sweeps."""
    import os
    cache_dir = os.path.join(save_dir, "diagnostics_cache")
    return os.path.join(cache_dir, f"{run_id}_n{n_batches}.json")


def select_from_wandb_runs(
    run_ids: List[str],
    project: str,
    dt: float,
    n_dims: int,
    n_batches: int = 100,
    eigenvalue_threshold: float = 0.001,
    use_loop_closure: bool = True,
    lambda_values: Optional[List[float]] = None,
    save_dir: Optional[str] = None,
    verbose: bool = False,
) -> SweepResult:
    """Select the best model from already-trained W&B runs (post-hoc mode).

    For each run: calls ``load_run`` and ``load_checkpoint``, then computes
    diagnostics and runs the selection algorithm.

    Args:
        run_ids: List of W&B run IDs to evaluate.
        project: W&B project name.
        dt: Time step of the data.
        n_dims: State-space dimensionality.
        n_batches: Number of validation batches for diagnostics.
        eigenvalue_threshold: Threshold for eigenvalue criterion.
        use_loop_closure: Whether to apply loop closure criterion.
        lambda_values: Lambda values corresponding to each run (for reporting).
        save_dir: If set, cache diagnostics per run in save_dir/diagnostics_cache/
            so the same run can be reused across sweeps without recomputing.
        verbose: Whether to print progress.

    Returns:
        SweepResult with selection, diagnostics, and run IDs.
    """
    import json
    import os

    from ..checkpoints import load_run, load_checkpoint

    all_diagnostics: List[DiagnosticMetrics] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dataloader = None
    data_generated = False

    for i, run_id in enumerate(run_ids):
        # Per-run cache: reuse diagnostics when same run appears in multiple sweeps
        if save_dir:
            cache_path = _diagnostics_cache_path(save_dir, run_id, n_batches)
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    data = json.load(f)
                metrics = DiagnosticMetrics(
                    one_step_mase=data["one_step_mase"],
                    loop_closure_loss=data.get("loop_closure_loss"),
                    fast_eigenvalue_fraction=data["fast_eigenvalue_fraction"],
                    trajectory_val_loss=data["trajectory_val_loss"],
                )
                all_diagnostics.append(metrics)
                if verbose:
                    msg = f"  run={run_id}: {metrics} (from cache, n_batches={n_batches})"
                    logger.info(msg)
                    print(msg, flush=True)
                continue

        if verbose:
            msg = f"Loading run {run_id} ({i+1}/{len(run_ids)})"
            logger.info(msg)
            print(msg, flush=True)

        generate_data = not data_generated
        run_obj, run_cfg, eq, run_dt, values, train_dl, val_dl, test_dl, trajs, lit_model = load_run(
            project,
            run_id=run_id,
            save_dir=save_dir,
            generate_data=generate_data,
            dt=dt,
            verbose=verbose,
        )
        if val_dataloader is None:
            val_dataloader = val_dl
        if generate_data:
            data_generated = True

        load_checkpoint(
            run_obj, run_cfg, lit_model, save_dir=save_dir, verbose=verbose
        )

        lit_model.eval()
        lit_model = lit_model.to(device)

        if verbose:
            print(f"  Computing diagnostics ({n_batches} batches)...", flush=True)
        metrics = compute_all_diagnostics(
            lit_model, val_dataloader, dt,
            n_batches=n_batches, use_loop_closure=use_loop_closure,
            verbose=verbose,
        )
        all_diagnostics.append(metrics)

        if save_dir:
            cache_path = _diagnostics_cache_path(save_dir, run_id, n_batches)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "n_batches": n_batches,
                        "one_step_mase": metrics.one_step_mase,
                        "loop_closure_loss": metrics.loop_closure_loss,
                        "fast_eigenvalue_fraction": metrics.fast_eigenvalue_fraction,
                        "trajectory_val_loss": metrics.trajectory_val_loss,
                    },
                    f,
                    indent=2,
                )
            if verbose:
                print(f"  Cached to {cache_path}", flush=True)

        if verbose:
            msg = f"  run={run_id}: {metrics}"
            logger.info(msg)
            print(msg, flush=True)

        lit_model.cpu()
        torch.cuda.empty_cache()

    selection = select_best_model(
        all_diagnostics,
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
        run_ids=run_ids,
    )
