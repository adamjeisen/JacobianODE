"""Pure metric computation and exclusion predicates for model selection.

Implements the physics-informed model selection criteria from Appendix D.8.8:
  C1 (one-step MASE): model must beat persistence (MASE < 1)
  C2 (loop closure): loop closure loss must be below sqrt(n_dims)
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

    one_step_mases = []
    loop_closure_losses = []
    num_eigs_too_fast = 0
    total_eigs = 0
    trajectory_losses = []
    threshold = -1.0 / dt

    with torch.no_grad():
        for i, batch in tqdm(enumerate(rand_dl), total=n_batches, disable=not verbose):
            if i >= n_batches:
                break
            batch = batch.to(device)
            lit_model.dt = dt

            # One-step MASE (teacher-forced)
            traj_ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=1)
            one_step_mases.append(traj_ret["metric_vals"]["mase"].float().item())

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

    n = len(one_step_mases)
    return DiagnosticMetrics(
        one_step_mase=sum(one_step_mases) / n if n > 0 else float("inf"),
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
    metrics: DiagnosticMetrics, n_dims: int
) -> bool:
    """C2: Does the model fail the loop closure criterion?

    A model fails if its loop closure loss exceeds sqrt(n_dims).
    Models without loop closure loss (e.g., NeuralODE) always pass.

    Args:
        metrics: Diagnostic metrics for the model.
        n_dims: Dimensionality of the state space.

    Returns:
        True if the model FAILS (should be excluded).
    """
    if metrics.loop_closure_loss is None:
        return False
    return metrics.loop_closure_loss > math.sqrt(n_dims)


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
