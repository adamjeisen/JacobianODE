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
    diagnostics_from_wandb,
)
from .ranking import RankingMethod
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
    n_latent: Optional[int] = None,
    ranking_method: RankingMethod = "pareto_knee",
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
        ranking_method: How to rank survivors and pick the best model.

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
        loop_closure_n_dims=n_latent,
        ranking_method=ranking_method,
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
    n_latent: Optional[int] = None,
    ranking_method: RankingMethod = "pareto_knee",
) -> SweepResult:
    """Select the best model from already-trained W&B runs (post-hoc mode).

    For each run, diagnostics are resolved in priority order:
      1. JSON file cache (``save_dir/diagnostics_cache/``)
      2. W&B history at the best-checkpoint epoch (no model loading needed)
      3. Full ``load_run`` + ``load_checkpoint`` + ``compute_all_diagnostics``

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
        n_latent: Latent dimension of the encoder. For latent models
            (LitLatentJacobianODE), C2 uses sqrt(n_latent) instead of sqrt(n_dims).
            When None, uses n_dims for C2.

    Returns:
        SweepResult with selection, diagnostics, and run IDs.
    """
    import json
    import os

    import wandb as _wandb

    from ..checkpoints import load_run, load_checkpoint

    all_diagnostics: List[DiagnosticMetrics] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dataloader = None
    data_generated = False
    api = _wandb.Api(timeout=90)

    for i, run_id in enumerate(run_ids):
        # --- Priority 1: JSON file cache ---
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

        # --- Priority 2: W&B history at best epoch (no model loading) ---
        api_run = api.run(f"{project}/{run_id}")
        wandb_metrics = diagnostics_from_wandb(api_run)
        if wandb_metrics is not None:
            all_diagnostics.append(wandb_metrics)
            # Persist to JSON cache for future calls
            if save_dir:
                cache_path = _diagnostics_cache_path(save_dir, run_id, n_batches)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(
                        {
                            "run_id": run_id,
                            "n_batches": n_batches,
                            "source": "wandb_history",
                            "one_step_mase": wandb_metrics.one_step_mase,
                            "loop_closure_loss": wandb_metrics.loop_closure_loss,
                            "fast_eigenvalue_fraction": wandb_metrics.fast_eigenvalue_fraction,
                            "trajectory_val_loss": wandb_metrics.trajectory_val_loss,
                        },
                        f,
                        indent=2,
                    )
            if verbose:
                msg = f"  run={run_id}: {wandb_metrics} (from W&B history)"
                logger.info(msg)
                print(msg, flush=True)
            continue

        # --- Priority 3: Full model loading + compute_all_diagnostics ---
        if verbose:
            msg = f"Loading run {run_id} ({i+1}/{len(run_ids)})"
            logger.info(msg)
            print(msg, flush=True)

        generate_data = not data_generated
        run_obj, run_cfg, eq, run_dt, values, train_dl, val_dl, test_dl, trajs, lit_model = load_run(
            project,
            run_id=run_id,
            run=api_run,
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
                        "source": "compute_all_diagnostics",
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
        loop_closure_n_dims=n_latent,
        ranking_method=ranking_method,
    )

    if lambda_values is None:
        lambda_values = list(range(len(run_ids)))

    return SweepResult(
        selection=selection,
        all_diagnostics=all_diagnostics,
        lambda_values=lambda_values,
        run_ids=run_ids,
    )


# ---------------------------------------------------------------------------
# Sweep discovery helpers
# ---------------------------------------------------------------------------


def _is_jac_ode_run(run) -> bool:
    """Return True if this W&B run has an encoder (i.e. is a JacobianODE run)."""
    return "model" in run.config and "encoder" in run.config.get("model", {})


def _get_loop_closure_weight(run) -> Optional[float]:
    """Robustly extract ``loop_closure_weight`` from a W&B run config."""
    try:
        val = run.config.get("training", {}).get("lightning", {}).get("loop_closure_weight")
        if val is not None:
            return float(val)
    except (TypeError, AttributeError):
        pass
    for key in (
        "training.lightning.loop_closure_weight",
        "training/lightning/loop_closure_weight",
    ):
        if hasattr(run.config, "get") and key in run.config:
            return float(run.config[key])
    return None


def _get_tangent_entropy_weight(run) -> float:
    """Robustly extract ``tangent_entropy_weight`` from a W&B run config."""
    try:
        val = run.config.get("training", {}).get("lightning", {}).get("tangent_entropy_weight")
        if val is not None:
            return float(val)
    except (TypeError, AttributeError):
        pass
    for key in (
        "training.lightning.tangent_entropy_weight",
        "training/lightning/tangent_entropy_weight",
    ):
        if hasattr(run.config, "get") and key in run.config:
            return float(run.config[key])
    return 0.0


def _get_kl_dyn_weight(run) -> float:
    """Robustly extract ``kl_dyn_weight`` from a W&B run config."""
    try:
        val = run.config.get("training", {}).get("lightning", {}).get("kl_dyn_weight")
        if val is not None:
            return float(val)
    except (TypeError, AttributeError):
        pass
    for key in (
        "training.lightning.kl_dyn_weight",
        "training/lightning/kl_dyn_weight",
    ):
        if hasattr(run.config, "get") and key in run.config:
            return float(run.config[key])
    return 0.0


@dataclass
class DiscoveredSweep:
    """Result of :func:`discover_sweep_runs`.

    Attributes:
        run_ids: W&B run IDs sorted by ``loop_closure_weight``.
        lambdas: Corresponding ``loop_closure_weight`` values.
        tangent_entropy_weights: Corresponding ``tangent_entropy_weight`` values.
        kl_dyn_weights: Corresponding ``kl_dyn_weight`` values.
        crashed_ids: IDs of crashed/failed JacobianODE runs (kept or deleted).
    """

    run_ids: List[str]
    lambdas: List[float]
    tangent_entropy_weights: List[float]
    kl_dyn_weights: List[float]
    crashed_ids: List[str] = field(default_factory=list)


def discover_sweep_runs(
    wandb_entity: str,
    wandb_project: str,
    wandb_group: Optional[str] = None,
    *,
    delete_crashed: bool = False,
    verbose: bool = False,
) -> DiscoveredSweep:
    """Query W&B for finished JacobianODE runs in a project (optionally filtered by group).

    Filters to finished runs that have an encoder config and a
    ``loop_closure_weight``, then returns them sorted by that weight.

    Parameters
    ----------
    wandb_entity : str
        W&B entity (team/user).
    wandb_project : str
        W&B project name (without entity prefix).
    wandb_group : str, optional
        W&B group name to filter on.  If ``None``, all runs in the project
        are considered.
    delete_crashed : bool
        If True, delete crashed/failed runs from W&B.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    DiscoveredSweep
        Sorted run IDs, lambda values, tangent-entropy weights, and
        kl-dyn weights.
    """
    import wandb as _wandb

    api = _wandb.Api(timeout=90)
    project_path = f"{wandb_entity}/{wandb_project}"
    run_filters = {"group": wandb_group} if wandb_group else None
    all_runs = api.runs(project_path, filters=run_filters)

    if verbose:
        msg = f"Found {len(all_runs)} total runs in {project_path}"
        if wandb_group:
            msg += f" (group={wandb_group})"
        print(msg)
        print("All runs (state, loop_closure_weight, tangent_entropy_weight, kl_dyn_weight):")
        for r in all_runs:
            lc = _get_loop_closure_weight(r)
            te = _get_tangent_entropy_weight(r)
            kd = _get_kl_dyn_weight(r)
            print(f"  {r.id}: state={r.state}, lc={lc}, te={te}, kl_dyn={kd}")
        print()

    # Handle crashed/failed runs
    crashed_ids: List[str] = []
    for run in all_runs:
        if run.state in ("crashed", "failed") and _is_jac_ode_run(run):
            lam = _get_loop_closure_weight(run)
            if delete_crashed:
                if verbose:
                    print(f"CRASHED: run_id={run.id} (lc={lam}) — deleting from W&B")
                run.delete()
                crashed_ids.append(run.id)
            elif verbose:
                print(f"CRASHED: run_id={run.id} (lc={lam}) — keeping")

    # Collect finished sweep runs
    run_ids: List[str] = []
    lambdas: List[float] = []
    te_weights: List[float] = []
    kd_weights: List[float] = []

    for run in all_runs:
        if run.state != "finished" or not _is_jac_ode_run(run):
            continue
        if run.id in crashed_ids:
            continue
        lc = _get_loop_closure_weight(run)
        if lc is not None:
            run_ids.append(run.id)
            lambdas.append(lc)
            te_weights.append(_get_tangent_entropy_weight(run))
            kd_weights.append(_get_kl_dyn_weight(run))

    # Sort by lambda
    sorted_tuples = sorted(zip(lambdas, te_weights, kd_weights, run_ids))
    lambdas = [t[0] for t in sorted_tuples]
    te_weights = [t[1] for t in sorted_tuples]
    kd_weights = [t[2] for t in sorted_tuples]
    run_ids = [t[3] for t in sorted_tuples]

    if verbose:
        print(f"Found {len(run_ids)} finished sweep runs:")
        for lam, te, kd, rid in zip(lambdas, te_weights, kd_weights, run_ids):
            print(
                f"  loop_closure_weight={lam}, tangent_entropy_weight={te}, "
                f"kl_dyn_weight={kd} -> run_id={rid}"
            )

    return DiscoveredSweep(
        run_ids=run_ids,
        lambdas=lambdas,
        tangent_entropy_weights=te_weights,
        kl_dyn_weights=kd_weights,
        crashed_ids=crashed_ids,
    )


def select_best_from_sweep(
    wandb_entity: str,
    wandb_project: str,
    save_dir: str,
    *,
    wandb_group: Optional[str] = None,
    n_batches: int = 100,
    eigenvalue_threshold: float = 0.001,
    use_loop_closure: bool = True,
    delete_crashed: bool = False,
    verbose: bool = False,
    ranking_method: RankingMethod = "pareto_knee",
) -> tuple[str, SweepResult, DiscoveredSweep]:
    """Discover sweep runs and select the best one.

    Convenience wrapper that calls :func:`discover_sweep_runs` then
    :func:`select_from_wandb_runs`.

    Parameters
    ----------
    wandb_entity : str
        W&B entity (team/user).
    wandb_project : str
        W&B project name (without entity prefix).
    save_dir : str
        Directory for checkpoints and diagnostics cache.
    wandb_group : str, optional
        W&B group name to filter on.  If ``None``, all runs in the project
        are considered.
    n_batches : int
        Number of validation batches for diagnostics.
    eigenvalue_threshold : float
        Threshold for the eigenvalue criterion.
    use_loop_closure : bool
        Whether to apply the loop-closure criterion.
    delete_crashed : bool
        If True, delete crashed/failed runs from W&B.
    verbose : bool
        Print progress.

    Returns
    -------
    best_run_id : str
        The run ID of the selected best model.
    sweep_result : SweepResult
        Full sweep result with diagnostics and selection.
    discovered : DiscoveredSweep
        The discovered runs (for further inspection).

    Raises
    ------
    RuntimeError
        If no finished sweep runs are found or selection fails.
    """
    from ..checkpoints import load_run

    discovered = discover_sweep_runs(
        wandb_entity, wandb_project, wandb_group,
        delete_crashed=delete_crashed, verbose=verbose,
    )

    if not discovered.run_ids:
        group_msg = f" group={wandb_group}" if wandb_group else ""
        raise RuntimeError(
            f"No finished JacobianODE sweep runs found in "
            f"{wandb_entity}/{wandb_project}{group_msg}"
        )

    # Load one run to determine data dimensionality and n_latent
    project_path = f"{wandb_entity}/{wandb_project}"
    _run0, _cfg0, _eq0, _dt0, _values0, _, _, _, _, _ = load_run(
        project_path,
        run_id=discovered.run_ids[0],
        save_dir=save_dir,
        generate_data=True,
        verbose=False,
    )
    # Compute delay-embedded dimensionality (not raw pre-embedding dims).
    _delay_params = _cfg0.data.train_test_params.delay_embedding_params
    _n_delays = int(_delay_params.n_delays)
    if _delay_params.observed_indices == "all":
        n_dims = int(_values0.shape[-1]) * _n_delays
    else:
        n_dims = len(_delay_params.observed_indices) * _n_delays
    n_latent = OmegaConf.select(_cfg0, "model.encoder.n_latent", default=None)
    if n_latent is None:
        n_latent = n_dims
    dt = _dt0

    if verbose:
        n_target_dims = OmegaConf.select(_cfg0, "model.n_target_dims", default=None)
        n_dyn = n_target_dims if n_target_dims is not None else n_latent
        print(f"n_dims={n_dims}, n_latent={n_latent}, n_dyn={n_dyn}, dt={dt:.4f}")

    sweep_result = select_from_wandb_runs(
        run_ids=discovered.run_ids,
        project=project_path,
        dt=dt,
        n_dims=n_dims,
        n_batches=n_batches,
        eigenvalue_threshold=eigenvalue_threshold,
        use_loop_closure=use_loop_closure,
        lambda_values=discovered.lambdas,
        save_dir=save_dir,
        verbose=verbose,
        n_latent=n_latent,
        ranking_method=ranking_method,
    )

    result = sweep_result.selection
    if result.best_index is None:
        raise RuntimeError("Model selection failed: no model passed all criteria.")

    best_run_id = discovered.run_ids[result.best_index]

    if verbose:
        idx = result.best_index
        print(f"\nRanking method:           {ranking_method}")
        print(f"Best run ID:              {best_run_id}")
        print(f"Best loop_closure_weight: {discovered.lambdas[idx]}")
        print(f"Best tangent_entropy_weight: {discovered.tangent_entropy_weights[idx]}")
        print(f"Best kl_dyn_weight:       {discovered.kl_dyn_weights[idx]}")
        print(f"Best traj loss:           {result.best_metrics.trajectory_val_loss:.6f}")
        print(f"Criteria applied: {result.criteria_applied}")
        print(f"Surviving: {len(result.surviving_indices)} / {len(sweep_result.all_diagnostics)}")

    return best_run_id, sweep_result, discovered
