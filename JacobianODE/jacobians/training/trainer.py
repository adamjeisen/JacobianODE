"""Training utilities for JacobianODE."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
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


def _resume_state_dir(cfg: DictConfig) -> Optional[Path]:
    """Deterministic per-job directory for preempt-safe checkpoints.

    SLURM preserves SLURM_ARRAY_JOB_ID / SLURM_ARRAY_TASK_ID / SLURM_JOB_ID
    across ``REQUEUE`` (which is the preemption behaviour on ou_bcs_low and
    mit_preemptable). Keying the resume dir on those IDs gives us a path
    that's stable across preempt cycles, so a fresh process launched by
    SLURM can find the last.ckpt from the previous attempt.

    Returns ``None`` when we're not running under SLURM or when
    ``logger_save_dirs`` is unset — in those cases resume is disabled and
    the training path is unchanged.
    """
    base_dir = cfg.training.get("logger_save_dirs")
    if not base_dir:
        return None
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    if array_job and array_task:
        key = f"{array_job}_{array_task}"
    elif job_id:
        key = job_id
    else:
        return None
    return Path(base_dir) / "_resume_state" / key


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

    # ----------------------------------------------------------------- #
    # Preempt-safe resume state
    # ----------------------------------------------------------------- #
    # If we've been requeued by SLURM after a preempt, look for the
    # last.ckpt + wandb_run_id saved by the previous attempt and wire
    # them into trainer.fit / WandbLogger so training continues instead
    # of restarting from epoch 0.
    resume_dir = _resume_state_dir(cfg)
    ckpt_path_resume: Optional[str] = None
    wandb_resume_id: Optional[str] = None
    if resume_dir is not None:
        resume_dir.mkdir(parents=True, exist_ok=True)
        last_file = resume_dir / "last.ckpt"
        if last_file.is_file():
            ckpt_path_resume = str(last_file)
            id_file = resume_dir / "wandb_run_id.txt"
            if id_file.is_file():
                wandb_resume_id = id_file.read_text().strip() or None
            logger.info(
                f"[resume] Found {last_file}; resuming training from it"
                + (f" (wandb id={wandb_resume_id})" if wandb_resume_id else "")
            )

    logger_kwargs = {
        "name": name,
        "project": project,
        "entity": entity,
        "group": group,
    }
    if wandb_resume_id:
        # Continue the same wandb run so the step counter stays monotonic.
        logger_kwargs["id"] = wandb_resume_id
        logger_kwargs["resume"] = "allow"

    experiment_logger = instantiate(cfg.training.logger, **logger_kwargs)

    # Persist the wandb run id on first start so future requeues continue
    # the same run rather than creating a new one each cycle.
    if resume_dir is not None:
        try:
            run_id = experiment_logger.experiment.id
            if run_id:
                (resume_dir / "wandb_run_id.txt").write_text(str(run_id))
        except Exception as e:
            logger.warning(f"[resume] Could not persist wandb run id: {e}")

    # Only update the logger config in the main process (rank 0)
    if os.getenv("LOCAL_RANK") == "0" or os.getenv("LOCAL_RANK") is None:
        experiment_logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True)
        )

    # Set up callbacks — "best" checkpoints track top-k by metric as before;
    # the new "last" checkpoint enables preempt-safe resume (see resume_dir).
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

    resume_cbs: List[L.Callback] = []
    if resume_dir is not None:
        # save_top_k=0 with save_last=True => only last.ckpt is maintained in
        # this dir. Best-by-metric checkpoints continue to live at the default
        # (WandbLogger-managed) path unchanged.
        resume_cbs.append(
            ModelCheckpoint(
                dirpath=str(resume_dir),
                save_last=True,
                save_top_k=0,
                filename="resume",  # ignored when save_top_k=0
            )
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
    callbacks.extend(resume_cbs)
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
        ckpt_path=ckpt_path_resume,
    )

    # trainer.fit returning normally means training finished (early stop or
    # max_epochs). A SLURM preempt would have SIGTERM'd us before reaching
    # here, so it's safe to wipe the resume dir — we're done and don't
    # want stale last.ckpt files lingering on disk.
    if resume_dir is not None and resume_dir.is_dir():
        try:
            shutil.rmtree(str(resume_dir))
        except Exception as e:
            logger.warning(f"[resume] Could not clean up {resume_dir}: {e}")

    wandb.finish()
    logger.info(f"Training complete for run: {name}")

    return trainer
