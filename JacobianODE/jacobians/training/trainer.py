"""Training utilities for JacobianODE."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Union

import torch
import lightning as L
import wandb
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..core.types import in_ipython
from ..lightning_base import OptunaPruneCallback, OptunaProgressCallback, OptunaConstrainedProgressCallback, PercentEarlyStopping

logger = logging.getLogger(__name__)


class _WandbFlushCallback(L.Callback):
    """Flush W&B metrics to the server at the end of each validation epoch.

    By default W&B syncs on a background thread whose interval may be too long
    for cluster environments with restricted network access.  Calling
    ``experiment.log({})`` forces the background sender to drain its queue,
    giving live metric streaming without waiting for ``wandb.finish()``.
    """

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logger_ = trainer.logger
        if logger_ is not None and hasattr(logger_, "experiment"):
            logger_.experiment.log({})


def train_model(
    cfg: DictConfig,
    lit_model: L.LightningModule,
    train_dataloaders: Union[DataLoader, List[DataLoader]],
    val_dataloaders: Union[DataLoader, List[DataLoader]],
    name: Optional[str] = None,
    extra_callbacks: Optional[List[L.Callback]] = None,
) -> L.Trainer:
    """Train the model using PyTorch Lightning.

    Sets up the training environment including callbacks, logging, and training
    parameters, then executes the training process.

    Args:
        cfg: Configuration object containing training parameters.
        lit_model: The PyTorch Lightning model to train.
        train_dataloaders: Training data loader(s).
        val_dataloaders: Validation data loader(s).
        name: Name of the training run. Defaults to None.
        project: W&B project name. Defaults to None.
        entity: W&B entity/team name. Defaults to None.
        group: W&B group name to organize runs within the project. Defaults to None.

    Returns:
        The Lightning Trainer instance after training completes.

    Example:
        >>> trainer = train_model(cfg, lit_model, train_dl, val_dl, name="run1", project="my-project")
        >>> # Training complete, model checkpoints saved

    Note:
        This function will call wandb.finish() after training completes.
        The training strategy is automatically set to 'ddp_notebook' if running
        in a Jupyter environment.
    """
    entity  = cfg.wandb_entity
    project = cfg.wandb_project
    group   = cfg.wandb_group
    name    = name or cfg.get("run_name", None)

    logger_kwargs = {
        "name": name,
        "project": project,
        "entity": entity,
        "group": group,
    }

    experiment_logger = instantiate(cfg.training.logger, **logger_kwargs)

    # Only update the logger config in the main process (rank 0)
    if os.getenv("LOCAL_RANK") == "0" or os.getenv("LOCAL_RANK") is None:
        experiment_logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True)
        )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.model_checkpoint.monitor,
        save_top_k=cfg.training.model_checkpoint.save_top_k,
        mode=cfg.training.model_checkpoint.mode,
    )

    traj_checkpoint = ModelCheckpoint(
        monitor="trajectory val_loss",
        save_top_k=1,
        mode="min",
        filename="traj-best-{epoch:02d}-{trajectory val_loss:.4f}",
    )

    if cfg.training.early_stopping.early_stopping_mode == "percent_thresh":
        early_stopping_callback = PercentEarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.early_stopping_patience,
            mode=cfg.training.early_stopping.mode,
            percent_thresh=cfg.training.early_stopping.percent_thresh,
            min_epochs=cfg.training.early_stopping.get("min_epochs", 0),
        )
    else:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.early_stopping_patience,
            mode=cfg.training.early_stopping.mode,
        )

    callbacks = [checkpoint_callback, traj_checkpoint, early_stopping_callback, _WandbFlushCallback()]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # Optuna pruning callback (only when study_name + storage are configured)
    optuna_prune_epoch = cfg.training.get("optuna_prune_epoch", None)
    optuna_study_name = cfg.get("optuna_study_name", None)
    optuna_storage = cfg.get("optuna_storage", None)
    if optuna_prune_epoch is not None and optuna_study_name and optuna_storage:
        optuna_constraint_metric = cfg.training.get("optuna_constraint_metric", None)
        optuna_constraint_threshold = cfg.training.get("optuna_constraint_threshold", None)
        prune_cb = OptunaPruneCallback(
            prune_epoch=int(optuna_prune_epoch),
            monitor="trajectory val_loss",
            study_name=optuna_study_name,
            storage=optuna_storage,
            min_completed=int(cfg.training.get("optuna_prune_min_completed", 5)),
            quantile=float(cfg.training.get("optuna_prune_quantile", 0.5)),
            constraint_metric=optuna_constraint_metric,
            constraint_threshold=float(optuna_constraint_threshold) if optuna_constraint_threshold is not None else None,
        )
        callbacks.append(prune_cb)
        logger.info(
            f"Optuna pruning enabled: epoch={optuna_prune_epoch}, "
            f"quantile={prune_cb.quantile}, min_completed={prune_cb.min_completed}"
            + (f", constraint: {optuna_constraint_metric} <= {optuna_constraint_threshold}" if optuna_constraint_metric else "")
        )

    # Optuna progress callback — write best-so-far loss to DB every validation epoch
    # so timed-out trials still have their results recorded.
    if optuna_study_name and optuna_storage:
        optuna_constraint_metric = cfg.training.get("optuna_constraint_metric", None)
        optuna_constraint_threshold = cfg.training.get("optuna_constraint_threshold", None)

        if optuna_constraint_metric and optuna_constraint_threshold is not None:
            progress_cb = OptunaConstrainedProgressCallback(
                monitor="trajectory val_loss",
                constraint_metric=optuna_constraint_metric,
                constraint_threshold=float(optuna_constraint_threshold),
                study_name=optuna_study_name,
                storage=optuna_storage,
            )
            callbacks.append(progress_cb)
            logger.info(
                f"Optuna constrained progress tracking enabled "
                f"(constraint: {optuna_constraint_metric} <= {optuna_constraint_threshold})"
            )
        else:
            progress_cb = OptunaProgressCallback(
                monitor="trajectory val_loss",
                study_name=optuna_study_name,
                storage=optuna_storage,
            )
            callbacks.append(progress_cb)
            logger.info("Optuna progress tracking enabled (best_so_far written to DB each epoch)")

    # Choose DDP strategy based on environment and number of GPUs.
    # DDP requires forking, which fails if CUDA is already initialized
    # (common in notebooks). Single-GPU doesn't benefit from DDP anyway.
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus <= 1:
        strategy = "auto"
        devices = "auto"
    elif in_ipython():
        strategy = "ddp_notebook"
        devices = "auto"
    else:
        strategy = "ddp"
        devices = "auto"

    # Extract gradient clipping parameters from the lightning model
    gradient_clip_val = lit_model.gradient_clip_val
    gradient_clip_algorithm = lit_model.gradient_clip_algorithm

    # Create trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=experiment_logger,
        log_every_n_steps=10,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        **cfg.training.trainer_params,
        devices=devices,
        strategy=strategy,
    )

    logger.info(f"Starting training run: {name}")
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
    )

    wandb.finish()
    logger.info(f"Training complete for run: {name}")

    return trainer
