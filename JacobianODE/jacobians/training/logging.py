"""Logging utilities for JacobianODE training."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def make_run_info(cfg):
    if cfg.data.data_type == 'dysts':
        data_cls = cfg.data.flow._target_.split('.')[-1]
    elif cfg.data.data_type == 'wmtask':
        data_cls = 'WMTask'
    elif cfg.data.data_type == 'custom':
        data_cls = cfg.data.get('name', 'custom')
    else:
        data_cls = cfg.data.data_type

    # Create name tuple
    name = tuple([
        f"{key}_{value}"
        for key, value in cfg.model.params.items()
        if value is not None and key not in ['_target_', '_partial_', 'embedder_kwargs', 'input_dim', 'output_dim']
    ])
    if 'deriv_params' in cfg.model and cfg.model.deriv_params is not None:
        name = name + tuple([f"{key}_{value}" for key, value in cfg.model.deriv_params.items() if value is not None and key not in ['_target_', '_partial_', 'embedder_kwargs', 'input_dim', 'output_dim']])
    name = name + tuple([f"{key}_{value}" for key, value in cfg.training.items() if key in ['batch_size', 'save_top_k']])
    name = name + tuple([f"{key}_{value:.4f}" if key in ['obs_noise_scale', 'obs_noise_scale_validation'] else f"{key}_{value}" for key, value in cfg.training.lightning.items() if value is not None and key not in ['_target_', 'eq']])
    name = (cfg.model.params._target_.split('.')[-1],) + name
    name = name + tuple([f"{key}_{value:.4f}" if key in ['obs_noise'] else f"{key}_{value}" for key, value in cfg.data.postprocessing.items() if key not in ('noise_scale_factor', 'mu', 'sigma')])
    name = name + tuple([f"{key}_{value}" for key, value in cfg.data.train_test_params.items() if key in ("n_delays")])
    name = name + tuple([f"{key}_{value}" for key, value in cfg.data.train_test_params.items() if key in ("seq_length")])
    if 'trajectory_params' in cfg.data:
        name = name + tuple([f"{key}_{value}" for key, value in cfg.data.trajectory_params.items() if key in ("n_periods", "pts_per_period", "standardize", "noise", "num_ics")])
    name = "__".join(name)

    project = (data_cls,'JacobianODE')
    project = "__".join(project)

    return name, project


def log_training_info(
    train_dataloader: DataLoader,
    trajs: Dict[str, Any],
    lit_model: Any,
    log: Optional[logging.Logger] = None,
) -> None:
    """Log information about the training setup.

    Prints or logs information about the training data size, model parameters,
    and other relevant training details.

    Args:
        train_dataloader: Training data loader.
        trajs: Dictionary containing trajectory information. Must have 'train_trajs'
            with a 'sequence' tensor.
        lit_model: The PyTorch Lightning model.
        log: Logger object for output. If None, uses print. Defaults to None.

    Example:
        >>> log_training_info(train_dl, trajs, lit_model, log=logger)
        Number of training trajectory examples: 10.000k
        ...
    """
    num_train_examples = len(train_dataloader.dataset.sequence)
    num_traj_points = (
        trajs["train_trajs"].sequence.shape[0] * trajs["train_trajs"].sequence.shape[1]
    )
    num_train_data_points = num_traj_points * trajs["train_trajs"].sequence.shape[2]
    total_params = sum(p.numel() for p in lit_model.parameters())

    messages = [
        f"Number of training trajectory examples: {num_train_examples / 1000:.3f}k",
        f"Number of training trajectory points: {num_traj_points / 1000:.3f}k",
        f"Number of training data points: {num_train_data_points / 1000:.3f}k",
        f"Total number of model parameters: {total_params / 1000:.3f}k",
    ]

    for msg in messages:
        if log is not None:
            log.info(msg)
        else:
            logger.info(msg)


def setup_wandb(
    cfg: DictConfig,
    trajs: Dict[str, Any],
    log: Optional[logging.Logger] = None,
    raw_values_to_use_for_noise: Optional[Union[np.ndarray, torch.Tensor]] = None,
    scale_noise: bool = True,
    prompt_entity: bool = False,
) -> Tuple[str, str, Optional[str]]:
    """Set up Weights & Biases logging for the training run.

    Configures W&B logging, including noise scaling and run naming.
    Handles versioning of run names to avoid conflicts.

    Args:
        cfg: Configuration object containing W&B parameters.
        trajs: Dictionary containing trajectory information. Must contain
            'train_trajs' with sequence of shape (n_traj, time_steps, n_dim).
        log: Logger object for output. Defaults to None.
        raw_values_to_use_for_noise: Alternative raw values to use for noise scaling.
            Defaults to None.
        scale_noise: Whether to scale noise. Defaults to True.
        prompt_entity: Whether to prompt for entity/team name instead of using
            config/env. Defaults to False.

    Returns:
        Tuple of (name, project, entity) where:
            - name: The W&B run name (may include version suffix)
            - project: The W&B project name
            - entity: The W&B entity/team name (or None to use default)

    Example:
        >>> name, project, entity = setup_wandb(cfg, trajs)
        >>> print(f"Starting run: {name} in {project}")
    """
    # Generate run name and project from config
    name, project = make_run_info(cfg)

    if cfg.get("wandb_project"):
        logger.info(
            f"Overriding auto-generated project '{project}' with "
            f"wandb_project='{cfg.wandb_project}'"
        )
        project = cfg.wandb_project

    # Handle entity/team name
    entity = _resolve_entity(log, prompt_entity)

    # Check for existing runs and increment version if needed
    name = _deduplicate_run_name(name, project, entity, log)

    return name, project, entity


def _resolve_entity(
    log: Optional[logging.Logger],
    prompt_entity: bool,
) -> Optional[str]:
    """Resolve the W&B entity from environment or user input."""
    entity = None

    should_prompt = prompt_entity or os.environ.get("WANDB_PROMPT_ENTITY", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if should_prompt:
        env_entity = os.environ.get("WANDB_ENTITY", None)
        suggested_entity = (
            env_entity if env_entity else "default (your personal account)"
        )

        if env_entity:
            msg = f"WANDB_ENTITY environment variable is set to: {env_entity}"
            if log is not None:
                log.info(msg)
            else:
                logger.info(msg)

        user_entity = input(
            f"Enter W&B team/entity name (default: {suggested_entity}, press Enter for None): "
        ).strip()

        if user_entity:
            entity = user_entity
            msg = f"Using entity: {entity}"
            if log is not None:
                log.info(msg)
            else:
                logger.info(msg)
        elif env_entity:
            entity = env_entity
    else:
        # Use environment variable if set, otherwise None (wandb will use default)
        entity = os.environ.get("WANDB_ENTITY", None)

    return entity


def _deduplicate_run_name(
    name: str,
    project: str,
    entity: Optional[str],
    log: Optional[logging.Logger],
) -> str:
    """Check for existing runs and add version suffix if needed.

    Skips the check when running under SLURM (sweep context). The public
    wandb API call (``api.runs(project_path)``) lists every run in the
    project and is not cached — when a sweep launches N tasks
    simultaneously, that's N parallel full-project listings, which reliably
    trips the ``429 Too Many Requests`` limit on ``api.wandb.ai/graphql``.
    Observed at the ``wandb_init``-adjacent startup path, where a 429 can
    cascade into wandb never attaching and the run never appearing (see
    vae_kl run_idx 22). Wandb already assigns globally unique run IDs, so
    duplicated *names* are harmless; the dedup is only cosmetic for
    interactive/notebook users.
    """
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID"):
        msg = (
            f"[wandb-dedup] skipping name-dedup under SLURM; "
            f"run names are unique-by-wandb-id, not by display name"
        )
        if log is not None:
            log.info(msg)
        else:
            logger.info(msg)
        return name

    project_path = f"{entity}/{project}" if entity else project

    try:
        api = wandb.Api()
        runs = api.runs(project_path)
        found_run = True
        version = 1
        base_name = name

        while found_run:
            found_run = False
            for run in runs:
                if run.name == name:
                    found_run = True
                    msg = f"Run {name} already exists, incrementing version"
                    if log is not None:
                        log.info(msg)
                    else:
                        logger.info(msg)
                    break
            if found_run:
                version += 1
                name = f"{base_name}_v{version}"
    except ValueError:
        msg = f"Project {project_path} does not exist!"
        if log is not None:
            log.warning(msg)
        else:
            logger.warning(msg)
    except Exception as e:
        # Network / auth / quota failures must not crash training.
        # Wandb assigns globally-unique run IDs anyway; name dedup is cosmetic.
        msg = (
            f"[wandb-dedup] skipping name-dedup; wandb.Api() unavailable "
            f"({type(e).__name__}: {e})"
        )
        if log is not None:
            log.warning(msg)
        else:
            logger.warning(msg)

    return name
