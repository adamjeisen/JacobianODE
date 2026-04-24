"""Training utilities for JacobianODE."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import sys
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
from ..lightning_base import (
    OptunaPruneCallback,
    OptunaProgressCallback,
    OptunaConstrainedProgressCallback,
    PercentEarlyStopping,
    ShadowPercentEarlyStoppingCheckpoint,
)

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


def _install_walltime_shutdown_handler() -> None:
    """On SIGUSR2, finish wandb cleanly and exit.

    submitit sends SIGUSR2 shortly before SLURM walltime expires. Its default
    handler raises UncompletedJobError, which crashes Python without calling
    wandb.finish(). The wandb run then lingers in state "running" for ~24h
    (until wandb's heartbeat timeout kicks in), so the sweep monitor never
    sees a terminal state and the sentinel can't fire.

    Replacing submitit's handler with this one ensures the run reaches a
    clean "finished" state in wandb as soon as walltime hits, which
    classify_wandb_run recognises as done_finished. The best-by-metric
    checkpoint from ModelCheckpoint is already on disk at that point, so
    downstream analysis has what it needs. We deliberately do NOT try to
    requeue — consistent with the "stop and analyze on walltime" policy.
    """
    def _handler(signum, frame):
        try:
            logger.warning(
                f"[walltime] Received {signal.Signals(signum).name}; "
                "finishing wandb and exiting cleanly."
            )
        except Exception:
            pass
        try:
            wandb.finish(exit_code=0)
        except Exception:
            pass
        sys.exit(0)

    try:
        signal.signal(signal.SIGUSR2, _handler)
    except (ValueError, OSError) as e:
        # Can't set signal handler outside main thread; not fatal.
        logger.debug(f"Could not install SIGUSR2 handler: {e}")


DONE_MARKER_NAME = "done.marker"


def read_done_marker(resume_dir: Optional[Path]) -> Optional[str]:
    """If this slot already finished, return the prior wandb id; else None.

    The marker is written by ``train_model`` on a clean trainer.fit() return.
    SLURM preserves SLURM_ARRAY_JOB_ID/TASK_ID across requeue, so a re-launch
    of the same array slot lands in the same ``resume_dir`` and sees the
    marker. We use it to short-circuit the second attempt instead of
    starting a fresh wandb run / retraining from scratch.
    """
    if resume_dir is None:
        return None
    marker = resume_dir / DONE_MARKER_NAME
    if not marker.is_file():
        return None
    try:
        return marker.read_text().strip() or ""
    except Exception:
        return ""


def _write_done_marker(resume_dir: Path, wandb_run_id: Optional[str]) -> None:
    """Atomically write the done marker. Best-effort — never raise."""
    marker = resume_dir / DONE_MARKER_NAME
    try:
        tmp = marker.with_suffix(marker.suffix + ".tmp")
        tmp.write_text(str(wandb_run_id or ""))
        os.replace(str(tmp), str(marker))
    except Exception as e:
        logger.warning(f"[resume] Could not write done marker {marker}: {e}")


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
    # Replace submitit's SIGUSR2 handler with our own so walltime ends the
    # run cleanly (wandb.finish + sys.exit) instead of raising mid-training.
    _install_walltime_shutdown_handler()

    entity  = cfg.wandb_entity
    project = cfg.wandb_project
    group   = cfg.wandb_group
    name    = name or cfg.get("run_name", None)

    # ----------------------------------------------------------------- #
    # Done-marker short-circuit
    # ----------------------------------------------------------------- #
    # If SLURM relaunches an array slot whose previous attempt already
    # completed cleanly (e.g. preempt-cycle that fired *after* trainer.fit
    # returned), we don't want to spin up a fresh wandb run and retrain
    # from scratch. The done marker is written below on a clean fit return,
    # and contains the prior wandb_run_id for traceability.
    resume_dir = _resume_state_dir(cfg)
    prior_id = read_done_marker(resume_dir)
    if prior_id is not None:
        logger.info(
            f"[resume] {resume_dir / DONE_MARKER_NAME} present (prior wandb "
            f"run id={prior_id!r}); this array slot already finished. "
            "Exiting without retraining."
        )
        sys.exit(0)

    # ----------------------------------------------------------------- #
    # Preempt-safe resume state
    # ----------------------------------------------------------------- #
    # If we've been requeued by SLURM after a preempt, look for the
    # last.ckpt + wandb_run_id saved by the previous attempt and wire
    # them into trainer.fit / WandbLogger so training continues instead
    # of restarting from epoch 0.
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

    # Only update the logger config in the main process (rank 0).
    # allow_val_change=True is required when resuming a wandb run because
    # config.update() otherwise rejects any diff between the existing config
    # (from the original start) and the re-serialized config — including
    # spurious float-precision differences like sigma=13.26897638365872 vs
    # 13.268976383658721 that arise from OmegaConf re-resolution. Without
    # this, preempt→requeue crashed the training script on restart and
    # killed the entire sweep's ability to benefit from REQUEUE.
    if os.getenv("LOCAL_RANK") == "0" or os.getenv("LOCAL_RANK") is None:
        experiment_logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True),
            allow_val_change=True,
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
        # Checkpoint at every epoch, keep only one file (the most recent),
        # and mirror it as last.ckpt for resume. In Lightning ModelCheckpoint,
        # save_last is triggered "whenever a checkpoint file gets saved", so
        # it requires save_top_k > 0 to actually fire — save_top_k=0 silently
        # disables the whole thing. monitor=None means "save on time, not on
        # metric".
        resume_cbs.append(
            ModelCheckpoint(
                dirpath=str(resume_dir),
                save_last=True,
                save_top_k=1,
                monitor=None,
                every_n_epochs=1,
                filename="resume-{epoch}",
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

    # Optional: shadow checkpoint that freezes at a simulated smaller-
    # patience ES-trigger. Primary ES still controls when training stops;
    # this callback writes a parallel best-so-far checkpoint frozen at the
    # point a hypothetical shadow-patience ES would have triggered. Useful
    # for comparing models trained with different ES patience settings
    # without running separate sweeps.
    shadow_patience = cfg.training.early_stopping.get("shadow_patience", None)
    if shadow_patience is not None:
        callbacks.append(
            ShadowPercentEarlyStoppingCheckpoint(
                monitor=cfg.training.early_stopping.monitor,
                shadow_patience=int(shadow_patience),
                percent_thresh=cfg.training.early_stopping.get("percent_thresh", 0.01),
                min_epochs=cfg.training.early_stopping.get("min_epochs", 0),
                filename=f"es{int(shadow_patience)}-best",
            )
        )

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
    # max_epochs). Drop the heavy artifacts (last.ckpt, anything > 1 KB)
    # but leave a tiny ``done.marker`` containing the wandb_run_id behind.
    # If SLURM later requeues this same array slot — which can happen on
    # preempt-style partitions like ou_bcs_low even after a clean exit —
    # the next train_model call sees the marker and exits immediately
    # instead of redoing all the work and creating a duplicate wandb run.
    if resume_dir is not None and resume_dir.is_dir():
        finished_run_id: Optional[str] = None
        try:
            finished_run_id = experiment_logger.experiment.id
        except Exception:
            pass
        # Keep marker (and wandb_run_id.txt for debug), drop everything else.
        for p in resume_dir.iterdir():
            if p.name in (DONE_MARKER_NAME, "wandb_run_id.txt"):
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(str(p))
                else:
                    p.unlink()
            except Exception as e:
                logger.warning(f"[resume] Could not clean up {p}: {e}")
        _write_done_marker(resume_dir, finished_run_id)

    wandb.finish()
    logger.info(f"Training complete for run: {name}")

    return trainer
