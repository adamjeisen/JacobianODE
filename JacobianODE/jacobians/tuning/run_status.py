"""Utilities for determining whether a W&B run is effectively complete.

A run counts as "done" if any of the following hold:

1. ``run.state == "finished"`` — the trainer exited cleanly.
2. The run's validation loss history shows it had converged (would have
   early-stopped) before it was killed.
3. The run's wall-clock duration is within ``walltime_tolerance_min`` of
   the SLURM ``timeout_min``, indicating a scheduler kill rather than a
   real error.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional


def _to_float(val) -> float:
    """Safely coerce a value to float, returning NaN on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def meets_early_stopping_criterion(
    run,
    monitor: str = "mean val loss",
    patience: int = 5,
    percent_thresh: float = 0.01,
    min_epochs: int = 10,
) -> tuple[bool, Optional[int]]:
    """Check if a run's loss history shows it had converged before it ended.

    This catches runs that were killed (SLURM timeout, crash) AFTER the loss
    had already plateaued — i.e., they would have early-stopped if given more
    time.

    Returns ``(met_criterion, last_epoch)``.
    """
    try:
        history = run.history(keys=[monitor, "epoch"], pandas=True)
    except Exception:
        return False, None

    if history.empty or monitor not in history.columns:
        return False, None

    history = history.dropna(subset=[monitor])
    if len(history) < 2:
        return False, None

    losses = [_to_float(v) for v in history[monitor].tolist()]
    epochs = (
        [_to_float(v) for v in history["epoch"].tolist()]
        if "epoch" in history.columns
        else list(range(len(losses)))
    )
    last_epoch = int(epochs[-1]) if epochs else None

    # Don't consider convergence before min_epochs
    start_idx = 0
    for i, e in enumerate(epochs):
        if e >= min_epochs:
            start_idx = i
            break
    else:
        return False, last_epoch

    # Replay the PercentEarlyStopping logic
    wait_count = 0
    prev_loss = losses[start_idx]
    for i in range(start_idx + 1, len(losses)):
        current = losses[i]
        if not math.isfinite(current):
            wait_count += 1
            if wait_count >= patience:
                return True, last_epoch
            continue
        if not math.isfinite(prev_loss):
            prev_loss = current
            wait_count = 0
            continue

        if prev_loss > current:
            pct_improvement = (prev_loss - current) / prev_loss
            if pct_improvement < percent_thresh:
                wait_count += 1
            else:
                wait_count = 0
        else:
            wait_count += 1

        prev_loss = current
        if wait_count >= patience:
            return True, last_epoch

    return False, last_epoch


def hit_slurm_walltime(
    run,
    timeout_min: float,
    tolerance_min: float = 5.0,
) -> bool:
    """Check if a crashed/failed run's wall-clock duration is within
    ``tolerance_min`` minutes of the SLURM timeout, indicating it was
    killed by the job scheduler rather than a real error.
    """
    try:
        created = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))
        heartbeat = run.heartbeat_at or run.updated_at
        ended = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
        duration_min = (ended - created).total_seconds() / 60.0
        return duration_min >= (timeout_min - tolerance_min)
    except Exception:
        return False


def is_run_effectively_done(
    run,
    *,
    monitor: str = "mean val loss",
    patience: int = 5,
    percent_thresh: float = 0.01,
    min_epochs: int = 10,
    slurm_timeout_min: Optional[float] = None,
    walltime_tolerance_min: float = 5.0,
) -> bool:
    """Return ``True`` if a W&B run should be considered complete.

    Checks (in order):
    1. ``run.state == "finished"``
    2. Loss had converged (early stopping criterion met)
    3. Run hit the SLURM walltime

    Parameters
    ----------
    run : wandb.apis.public.Run
        A W&B run object.
    monitor : str
        Metric name to check for convergence.
    patience : int
        Number of epochs without sufficient improvement.
    percent_thresh : float
        Minimum fractional improvement to reset patience.
    min_epochs : int
        Minimum epochs before early stopping can trigger.
    slurm_timeout_min : float, optional
        SLURM job timeout in minutes.  If ``None``, walltime check is skipped.
    walltime_tolerance_min : float
        A run within this many minutes of the timeout is considered killed
        by the scheduler.
    """
    if run.state == "finished":
        return True

    if run.state not in ("crashed", "failed"):
        return False

    converged, _ = meets_early_stopping_criterion(
        run,
        monitor=monitor,
        patience=patience,
        percent_thresh=percent_thresh,
        min_epochs=min_epochs,
    )
    if converged:
        return True

    if slurm_timeout_min is not None:
        if hit_slurm_walltime(run, slurm_timeout_min, tolerance_min=walltime_tolerance_min):
            return True

    return False
