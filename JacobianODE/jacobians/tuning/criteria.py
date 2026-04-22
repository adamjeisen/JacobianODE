"""Pure metric computation and exclusion predicates for model selection.

Implements the physics-informed model selection criteria from Appendix D.8.8:
  C1 (one-step MASE): model must beat persistence (MASE < 1)
  C2 (loop closure): loop closure loss must be below sqrt(n) where n is the
  loop-closure space dimension (n_latent for latent models, n_dims otherwise)
  C3 (eigenvalue): fraction of fast eigenvalues must be below threshold

Reference: checkpoints/wandb_utils.py lines 210-240, 266, 276-320.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler


@dataclass
class DiagnosticMetrics:
    """Diagnostic metrics computed for a single trained model.

    Attributes:
        one_step_mase: Mean one-step MASE (teacher-forced).  A value < 1
            means the model beats the persistence baseline.
        loop_closure_loss: Mean loop closure MSE (None for NeuralODE).
        fast_eigenvalue_fraction: Fraction of eigenvalues with real part < -1/dt.
        trajectory_val_loss: Trajectory validation loss (used for final ranking).
    """

    one_step_mase: float
    loop_closure_loss: Optional[float]
    fast_eigenvalue_fraction: float
    trajectory_val_loss: float


def compute_all_diagnostics(
    lit_model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    dt: float,
    n_batches: int = 100,
    use_loop_closure: bool = True,
    verbose: bool = False,
    seed: int = 42,
) -> DiagnosticMetrics:
    """Compute all diagnostic metrics over randomly sampled validation batches.

    Batches are selected via a seeded ``RandomSampler`` so results are
    reproducible yet not biased towards the first portion of the dataset.

    Args:
        lit_model: Lightning model (already on device, in eval mode).
        val_dataloader: Validation data loader.
        dt: Time step of the data.
        n_batches: Maximum number of batches to evaluate.
        use_loop_closure: Whether to compute loop closure loss.
        verbose: Whether to show a progress bar.
        seed: Fixed seed for the random sampler (ensures reproducibility).

    Returns:
        DiagnosticMetrics with all fields populated.
    """
    device = next(lit_model.parameters()).device

    # Build a randomized dataloader with a fixed seed so the same batches
    # are selected every time, regardless of external random state.
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(val_dataloader.dataset, generator=generator)
    rand_dl = DataLoader(
        val_dataloader.dataset,
        batch_size=val_dataloader.batch_size,
        sampler=sampler,
        num_workers=val_dataloader.num_workers,
        pin_memory=val_dataloader.pin_memory,
    )

    # Encoder-only runs have no trained dynamics MLP, so trajectory rollout
    # and loop closure return meaningless values. Replace the dynamics-based
    # metrics with reconstruction-based ones so selection ranks correctly.
    if getattr(lit_model, "encoder_only_mode", False):
        recon_losses = []
        with torch.no_grad():
            iterator = tqdm(total=n_batches, disable=not verbose)
            for i, batch in enumerate(rand_dl):
                if i >= n_batches:
                    break
                batch = batch.to(device)
                # Same path training + val use: encode, _reconstruction_loss
                # handles the split + zero-pad + decode internally.
                z_full = lit_model.encode_trajectory(batch)
                recon_losses.append(
                    float(lit_model._reconstruction_loss(batch, z_full=z_full).item())
                )
                iterator.update(1)
            iterator.close()
        n = len(recon_losses)
        return DiagnosticMetrics(
            one_step_mase=0.0,              # not applicable; keeps C1 filter happy
            loop_closure_loss=None,         # not applicable
            fast_eigenvalue_fraction=0.0,   # not applicable
            trajectory_val_loss=(sum(recon_losses) / n) if n > 0 else float("inf"),
        )

    total_model_mae = 0.0
    total_persistence_mae = 0.0
    loop_closure_losses = []
    num_eigs_too_fast = 0
    total_eigs = 0
    trajectory_losses = []
    threshold = -1.0 / dt
    n_mase_batches = 0

    with torch.no_grad():
        iterator = tqdm(total=n_batches, disable=not verbose)
        for i, batch in enumerate(rand_dl):
            if i >= n_batches:
                break
            batch = batch.to(device)
            lit_model.dt = dt

            # One-step MASE (teacher-forced) — accumulate numerator/denominator
            # for ratio-of-means (avoids inflation from small-denominator windows).
            traj_ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=1)
            total_model_mae += traj_ret["metric_vals"]["model_mae"].float().item()
            total_persistence_mae += traj_ret["metric_vals"]["persistence_mae"].float().item()
            n_mase_batches += 1

            # Trajectory val loss (free-running)
            traj_free_ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0)
            trajectory_losses.append(traj_free_ret["loss"].float().item())

            # For latent models (LitLatentJacobianODE), the Jacobian MLP and
            # loop closure operate in latent space, not observation space.
            # Encode the batch first so compute_jacobians and
            # loop_closure_model_step receive latent vectors, matching what
            # validation_step does during training.
            if hasattr(lit_model, 'encode_trajectory'):
                z_for_eval = lit_model.encode_trajectory(batch)
                # For coupling encoders, Jacobian MLP and loop closure
                # operate on the dynamic subspace only.
                if hasattr(lit_model, '_split_latent'):
                    z_for_eval, _ = lit_model._split_latent(z_for_eval)
            else:
                z_for_eval = batch

            # Eigenvalue fraction
            pred_jacs = lit_model.compute_jacobians(z_for_eval)
            eigs_real = torch.linalg.eigvals(pred_jacs).real.flatten()
            num_eigs_too_fast += torch.sum(eigs_real <= threshold).float().item()
            total_eigs += len(eigs_real)

            # Loop closure
            if use_loop_closure:
                lc_ret = lit_model.loop_closure_model_step(z_for_eval)
                loop_closure_losses.append(lc_ret["metric_vals"]["mse"].float().item())
            iterator.update(1)

        iterator.close()

    n = n_mase_batches
    # Compute MASE as ratio-of-means (not mean-of-ratios) to avoid
    # inflation from windows with small persistence denominators.
    avg_model_mae = total_model_mae / n if n > 0 else float("inf")
    avg_persistence_mae = total_persistence_mae / n if n > 0 else 1e-8
    return DiagnosticMetrics(
        one_step_mase=avg_model_mae / avg_persistence_mae if n > 0 else float("inf"),
        loop_closure_loss=(
            sum(loop_closure_losses) / len(loop_closure_losses)
            if loop_closure_losses
            else None
        ),
        fast_eigenvalue_fraction=(
            num_eigs_too_fast / total_eigs if total_eigs > 0 else 0.0
        ),
        trajectory_val_loss=sum(trajectory_losses) / n if n > 0 else float("inf"),
    )


# ---------------------------------------------------------------------------
# Boolean predicates (pure, model-free)
# ---------------------------------------------------------------------------


def fails_one_step_criterion(metrics: DiagnosticMetrics) -> bool:
    """C1: Does the model fail the one-step MASE criterion?

    A model fails if its one-step MASE exceeds 1.0, meaning it is worse
    than the persistence baseline (predicting the previous value).

    Args:
        metrics: Diagnostic metrics for the model.

    Returns:
        True if the model FAILS (should be excluded).
    """
    return metrics.one_step_mase > 1.0


def fails_loop_closure_criterion(
    metrics: DiagnosticMetrics, loop_closure_n_dims: int
) -> bool:
    """C2: Does the model fail the loop closure criterion?

    A model fails if its loop closure loss exceeds sqrt(loop_closure_n_dims).
    For latent models (LitLatentJacobianODE), loop closure is computed in latent
    space, so use n_latent. For non-latent models, use the state-space dimension.
    Models without loop closure loss (e.g., NeuralODE) always pass.

    Args:
        metrics: Diagnostic metrics for the model.
        loop_closure_n_dims: Dimensionality of the space where loop closure
            is computed (n_latent for latent models, n_dims otherwise).

    Returns:
        True if the model FAILS (should be excluded).
    """
    if metrics.loop_closure_loss is None:
        return False
    return metrics.loop_closure_loss > math.sqrt(loop_closure_n_dims)


def fails_eigenvalue_criterion(
    metrics: DiagnosticMetrics, eigenvalue_threshold: float = 0.001
) -> bool:
    """C3: Does the model fail the eigenvalue criterion?

    A model fails if its fast eigenvalue fraction exceeds the threshold.

    Args:
        metrics: Diagnostic metrics for the model.
        eigenvalue_threshold: Maximum allowed fraction of fast eigenvalues.

    Returns:
        True if the model FAILS (should be excluded).
    """
    return metrics.fast_eigenvalue_fraction > eigenvalue_threshold


# ---------------------------------------------------------------------------
# W&B history-based diagnostics (avoids loading model checkpoints)
# ---------------------------------------------------------------------------


def diagnostics_from_wandb(
    run,
    monitor: str = "trajectory val_loss",
) -> Optional[DiagnosticMetrics]:
    """Extract diagnostic metrics from W&B history at the best-checkpoint epoch.

    Looks up the epoch with the lowest ``monitor`` metric and reads
    ``val/one_step_mase``, ``val/loop_closure_loss``,
    ``val/fast_eigenvalue_fraction``, and the trajectory val loss from that row.

    Returns ``None`` if any required metric is missing from the W&B history
    (e.g., for runs trained before these metrics were logged).

    Args:
        run: A ``wandb.apis.public.Run`` object (from ``wandb.Api().run(...)``).
        monitor: The metric name whose minimum identifies the best epoch.
            Defaults to ``"trajectory val_loss"`` (the traj checkpoint monitor).

    Returns:
        DiagnosticMetrics populated from the W&B history, or None if the
        required metrics are not available.
    """
    import pandas as pd

    try:
        hist = run.history(samples=500_000, pandas=True)
    except Exception:
        return None

    if hist is None or hist.empty or monitor not in hist.columns:
        return None

    # Encoder-only runs don't log val/one_step_mase etc. (no dynamics to
    # teacher-force) AND their "trajectory val_loss" was the recon+KL
    # composite prior to the fix, which misleads best_traj_loss selection.
    # Detect and route to recon-based ranking directly from wandb history.
    is_encoder_only = bool(
        (run.config.get("model") or {}).get("encoder_only_mode", False)
    )
    if is_encoder_only:
        if "val/recon_loss" not in hist.columns:
            return None
        m_recon = pd.to_numeric(hist["val/recon_loss"], errors="coerce")
        valid_r = m_recon.notna()
        if not valid_r.any():
            return None
        best_idx = m_recon.loc[valid_r].idxmin()
        return DiagnosticMetrics(
            one_step_mase=0.0,
            loop_closure_loss=None,
            fast_eigenvalue_fraction=0.0,
            trajectory_val_loss=float(m_recon.loc[best_idx]),
        )

    m = pd.to_numeric(hist[monitor], errors="coerce")
    valid = m.notna()
    if not valid.any():
        return None

    best_idx = m.loc[valid].idxmin()
    row = hist.loc[best_idx]

    # one_step_mase is the metric that was previously not logged — if it's
    # missing, this is an old run and we must fall back to full computation.
    if "val/one_step_mase" not in row.index or pd.isna(row.get("val/one_step_mase")):
        return None

    loop_closure_loss: Optional[float] = None
    if "val/loop_closure_loss" in row.index and pd.notna(row.get("val/loop_closure_loss")):
        loop_closure_loss = float(row["val/loop_closure_loss"])

    fast_eig_frac = 0.0
    if "val/fast_eigenvalue_fraction" in row.index and pd.notna(row.get("val/fast_eigenvalue_fraction")):
        fast_eig_frac = float(row["val/fast_eigenvalue_fraction"])

    return DiagnosticMetrics(
        one_step_mase=float(row["val/one_step_mase"]),
        loop_closure_loss=loop_closure_loss,
        fast_eigenvalue_fraction=fast_eig_frac,
        trajectory_val_loss=float(m.loc[best_idx]),
    )
