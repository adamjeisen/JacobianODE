"""Checkpoint loading utilities for JacobianODE."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from ..core.reproducibility import seed_everything
from ..data.dataloaders import create_dataloaders
from ..data.processing import postprocess_data
from ..data.trajectory import make_trajectories
from ..lightning_base import LitBase
from ..training.model_factory import make_model
from .legacy import reverse_wandb_run

logger = logging.getLogger(__name__)

# Cutoff date for legacy run handling (January 31st 2025 at 2pm EST)
LEGACY_CUTOFF_DATE = datetime(2025, 1, 31, 14, 0, tzinfo=ZoneInfo("America/New_York"))


def load_run(
    project: str,
    run_id: Optional[str] = None,
    run: Optional[Any] = None,
    save_dir: Optional[str] = None,
    no_noise: bool = False,
    generate_data: bool = True,
    dt: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Any, Any, Any, Optional[float], Any, Any, Any, Any, Any, Any]:
    """Load a previous training run and its associated data.

    Handles loading of both recent and legacy runs, including model checkpoints,
    configuration, and trajectory data.

    Args:
        project: W&B project name.
        run_id: ID of the run to load. Defaults to None.
        run: W&B run object. Defaults to None.
        save_dir: Directory containing saved data. Defaults to None.
        no_noise: Whether to disable noise in data generation. Defaults to False.
        generate_data: Whether to generate new trajectory data. Defaults to True.
        dt: Time step size. Defaults to None.
        verbose: Whether to print progress information. Defaults to False.

    Returns:
        Tuple of (run, cfg, eq, dt, values, train_dataloader, val_dataloader,
        test_dataloader, trajs, lit_model) containing all components of the loaded run.

    Raises:
        ValueError: If both run_id and run are None.

    Example:
        >>> run, cfg, eq, dt, values, train_dl, val_dl, test_dl, trajs, lit_model = \\
        ...     load_run("my-project", run_id="abc123")
        >>> lit_model.eval()
    """
    # Get run object
    if run is None:
        api = wandb.Api(timeout=30)
        run = api.run(f"{project}/{run_id}")
    elif run_id is None:
        raise ValueError("run_id and run cannot both be None")

    # Check run date to determine handling method
    utc_dt = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    est_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))

    if verbose:
        logger.info(f"Run created at {est_dt} EST")
        logger.info(f"Is after Jan 31 2025 2pm EST? {est_dt > LEGACY_CUTOFF_DATE}")

    if est_dt > LEGACY_CUTOFF_DATE:
        return _load_recent_run(
            run, project, save_dir, no_noise, generate_data, dt, verbose
        )
    else:
        return _load_legacy_run(
            run, project, save_dir, no_noise, generate_data, dt, verbose
        )


def _load_recent_run(
    run: Any,
    project: str,
    save_dir: Optional[str],
    no_noise: bool,
    generate_data: bool,
    dt: Optional[float],
    verbose: bool,
) -> Tuple[Any, Any, Any, Optional[float], Any, Any, Any, Any, Any, Any]:
    """Load a run created after the legacy cutoff date."""
    if verbose:
        logger.info("Date is after Jan 31 2025 2pm EST")

    cfg = OmegaConf.create(run.config)

    # Handle missing config keys
    if "use_deriv_net" not in cfg.training:
        cfg.training.use_deriv_net = False

    if save_dir is None:
        save_dir = cfg.training.logger.save_dir

    # Set seeds for reproducibility
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number)

    if no_noise:
        cfg.data.postprocessing.obs_noise = 0

    if generate_data:
        if verbose:
            logger.info("Making trajectories")
        eq, sol, dt = make_trajectories(cfg, save_dir=save_dir, verbose=verbose)
        if verbose:
            logger.info("Done making trajectories")
            logger.info(f"obs_noise: {cfg.data.postprocessing.obs_noise}")
            logger.info(f"seq_length: {cfg.data.train_test_params.seq_length}")
    else:
        eq = None
        sol = None
        dt = None if dt is None else float(dt)

    # Clean up lightning config
    del_keys = []
    for key in cfg.training.lightning.keys():
        if key not in LitBase.__init__.__code__.co_varnames and key != "_target_":
            del_keys.append(key)
    for key in del_keys:
        del cfg.training.lightning[key]

    if generate_data:
        # Postprocess data
        values = postprocess_data(cfg, sol["values"])
        # Create train and test sets
        train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
            cfg, values
        )
    else:
        values = None
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None
        trajs = None

    # Make model
    if "NeuralODE" in cfg.model.params._target_:
        cfg.model.params.dt = float(dt)

    if cfg.data.train_test_params.delay_embedding_params.n_delays > 1:
        lit_model = make_model(cfg, dt, eq=None, save_dir=save_dir, verbose=verbose)
    else:
        lit_model = make_model(
            cfg, dt, eq=eq, project=project, save_dir=save_dir, verbose=verbose
        )

    return (
        run,
        cfg,
        eq,
        dt,
        values,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        trajs,
        lit_model,
    )


def _load_legacy_run(
    run: Any,
    project: str,
    save_dir: Optional[str],
    no_noise: bool,
    generate_data: bool,
    dt: Optional[float],
    verbose: bool,
) -> Tuple[Any, Any, Any, Optional[float], Any, Any, Any, Any, Any, Any]:
    """Load a run created before the legacy cutoff date."""
    if verbose:
        logger.info("Date is before Jan 31 2025 2pm EST")

    ret_dict = reverse_wandb_run(run, return_data=True, save_dir=save_dir, checkpoint=None)
    cfg = ret_dict["cfg"]
    lit_model = ret_dict["lit_model"]
    eq = ret_dict["eq"]
    values = ret_dict["values"]
    dt = ret_dict["dt"]

    # Set seeds for reproducibility
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number)

    if generate_data:
        eq, sol, dt = make_trajectories(cfg)
        train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
            cfg, values
        )
    else:
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None
        trajs = None

    return (
        run,
        cfg,
        eq,
        dt,
        values,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        trajs,
        lit_model,
    )


def get_all_checkpoints(
    run: Any,
    cfg: Any,
    save_dir: Optional[str] = None,
) -> Tuple[List[str], str]:
    """Get all available checkpoint files for a run.

    Args:
        run: W&B run object.
        cfg: Configuration object.
        save_dir: Directory containing checkpoints. Defaults to None.

    Returns:
        Tuple of (checkpoint_files, checkpoint_dir) where:
            - checkpoint_files: List of checkpoint filenames sorted by epoch
            - checkpoint_dir: Directory containing the checkpoints

    Example:
        >>> checkpoints, ckpt_dir = get_all_checkpoints(run, cfg)
        >>> print(f"Found {len(checkpoints)} checkpoints")
    """
    if save_dir is None:
        save_dir = (
            run.config["save_dir"]
            if "save_dir" in run.config
            else cfg.training.logger.save_dir
        )

    checkpoint_dir = os.path.join(save_dir, run.project, run.id, "checkpoints")
    checkpoint_files = os.listdir(checkpoint_dir)

    # Sort by epoch number
    checkpoint_files = sorted(
        checkpoint_files, key=lambda x: int(x.split("=")[1].split("-")[0])
    )

    return checkpoint_files, checkpoint_dir


def load_checkpoint(
    run: Any,
    cfg: Any,
    lit_model: Any,
    save_dir: Optional[str] = None,
    epoch: Optional[int] = None,
    loss_key: str = "mean_val_loss",
    verbose: bool = False,
) -> None:
    """Load a specific checkpoint for a model.

    Can load either the best checkpoint (based on validation loss) or a specific epoch.

    Args:
        run: W&B run object.
        cfg: Configuration object.
        lit_model: Model to load checkpoint into.
        save_dir: Directory containing checkpoints. Defaults to None.
        epoch: Specific epoch to load. If None, loads best checkpoint. Defaults to None.
        loss_key: Key to use for finding best checkpoint. Defaults to 'mean_val_loss'.
        verbose: Whether to print progress information. Defaults to False.

    Example:
        >>> load_checkpoint(run, cfg, lit_model, epoch=10, verbose=True)
        Loading checkpoint from epoch 10
    """
    if verbose:
        logger.info(f"Loading checkpoint from {save_dir}")

    checkpoint_files, checkpoint_dir = get_all_checkpoints(run, cfg, save_dir)

    if verbose:
        epochs = [int(f.split("=")[1].split("-")[0]) for f in checkpoint_files]
        logger.info(f"Checkpoint epochs: {epochs}")

    if epoch is None:
        # Pick the checkpoint with minimum validation loss
        mean_val_losses = [
            {"epoch": h["epoch"], "mean_val_loss": h[loss_key]}
            for h in run.scan_history()
            if loss_key in h and h[loss_key] is not None
        ]

        # Fallback to alternate key name
        if len(mean_val_losses) == 0:
            mean_val_losses = [
                {"epoch": h["epoch"], "mean_val_loss": h["mean val loss"]}
                for h in run.scan_history()
                if "mean val loss" in h and h["mean val loss"] is not None
            ]

        epoch = mean_val_losses[
            np.argmin([mvl["mean_val_loss"] for mvl in mean_val_losses])
        ]["epoch"]
        checkpoint = [f for f in checkpoint_files if f.startswith(f"epoch={epoch}-")][0]
    else:
        checkpoint = [f for f in checkpoint_files if f.startswith(f"epoch={epoch}-")][0]

    if verbose:
        logger.info(f"Loading checkpoint from epoch {epoch}")

    # Load checkpoint
    checkpoint_data = torch.load(
        os.path.join(checkpoint_dir, checkpoint),
        weights_only=False,
        map_location="cpu",
        mmap=True,
    )
    loaded_state = checkpoint_data["state_dict"]

    # Handle uncertainty parameters if needed
    if (
        "use_uncertainty" in cfg.training.lightning
        and cfg.training.lightning.use_uncertainty
    ):
        if "logvar_trajectory" not in loaded_state:
            loaded_state["logvar_trajectory"] = torch.tensor(0)
        if "logvar_reverse" not in loaded_state:
            loaded_state["logvar_reverse"] = torch.tensor(0)
        if "logvar_loop_closure" not in loaded_state:
            loaded_state["logvar_loop_closure"] = torch.tensor(0)
        if (
            "logvar_lipschitz" not in loaded_state
            and "lipschitz" in cfg.model.params
            and cfg.model.params.lipschitz
        ):
            loaded_state["logvar_lipschitz"] = torch.tensor(0)

    lit_model.load_state_dict(loaded_state)
    lit_model.eval()

    # Cleanup
    del loaded_state
    torch.cuda.empty_cache()
