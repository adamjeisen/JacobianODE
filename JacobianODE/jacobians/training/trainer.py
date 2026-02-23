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
from ..lightning_base import PercentEarlyStopping

logger = logging.getLogger(__name__)


def train_model(
    cfg: DictConfig,
    lit_model: L.LightningModule,
    train_dataloaders: Union[DataLoader, List[DataLoader]],
    val_dataloaders: Union[DataLoader, List[DataLoader]],
    name: str,
    project: str,
    entity: Optional[str] = None,
) -> None:
    """Train the model using PyTorch Lightning.

    Sets up the training environment including callbacks, logging, and training
    parameters, then executes the training process.

    Args:
        cfg: Configuration object containing training parameters.
        lit_model: The PyTorch Lightning model to train.
        train_dataloaders: Training data loader(s).
        val_dataloaders: Validation data loader(s).
        name: Name of the training run.
        project: W&B project name.
        entity: W&B entity/team name. Defaults to None.

    Example:
        >>> train_model(cfg, lit_model, train_dl, val_dl, name="run1", project="my-project")
        >>> # Training complete, model checkpoints saved

    Note:
        This function will call wandb.finish() after training completes.
        The training strategy is automatically set to 'ddp_notebook' if running
        in a Jupyter environment.
    """
    # Set up logger
    logger_kwargs = {
        "name": name,
        "project": project,
    }
    if entity is not None:
        logger_kwargs["entity"] = entity

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
        )
    else:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.early_stopping_patience,
            mode=cfg.training.early_stopping.mode,
        )

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
        callbacks=[checkpoint_callback, traj_checkpoint, early_stopping_callback],
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
