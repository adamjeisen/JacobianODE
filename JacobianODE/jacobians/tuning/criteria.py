"""Pure metric computation and exclusion predicates for model selection.

Implements the physics-informed model selection criteria from Appendix D.8.8:
  C1 (one-step error): model must beat persistence baseline
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

from ..metrics import normalized_mse


@dataclass
class DiagnosticMetrics:
    """Diagnostic metrics computed for a single trained model.

    Attributes:
        one_step_error: Mean one-step prediction normalized MSE (teacher-forced).
        loop_closure_loss: Mean loop closure MSE (None for NeuralODE).
        fast_eigenvalue_fraction: Fraction of eigenvalues with real part < -1/dt.
        trajectory_val_loss: Trajectory validation loss (used for final ranking).
    """

    one_step_error: float
    loop_closure_loss: Optional[float]
    fast_eigenvalue_fraction: float
    trajectory_val_loss: float


def compute_persistence_baseline(data: torch.Tensor, normalize: bool = False) -> float:
    """Compute persistence baseline using normalized MSE.

    Matches the scale of trajectory_model_step, which returns normalized_mse.
    Uses the same per-dimension variance normalization so the baseline is
    directly comparable to model one-step errors.

    Args:
        train_trajs: Training trajectories, shape (..., T, D).

    Returns:
        Scalar persistence baseline value (normalized MSE scale).
    """
    pred = data[..., :-1, :]    # x_{t-1}
    target = data[..., 1:, :]   # x_t
    if normalize:
        return normalized_mse(pred, target).item()
    else:
        return (target - pred).pow(2).mean().item()


def compute_one_step_error(
    lit_model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    n_batches: int = 100,
    verbose: bool = False,
) -> float:
    """Compute mean one-step prediction error over validation data.

    Calls ``lit_model.trajectory_model_step(batch, alpha_teacher_forcing=1)``
    to get teacher-forced one-step loss.

    Args:
        lit_model: Lightning model (already on device, in eval mode).
        val_dataloader: Validation data loader.
        n_batches: Maximum number of batches to evaluate.

    Returns:
        Mean one-step prediction error.
    """
    device = next(lit_model.parameters()).device
    errors = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=n_batches, disable=not verbose):
            if i >= n_batches:
                break
            batch = batch.to(device)
            ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=1)
            errors.append(ret["loss"].float().item())
    return sum(errors) / len(errors) if errors else float("inf")


# def compute_fast_eigenvalue_fraction(
#     lit_model: torch.nn.Module,
#     val_dataloader: torch.utils.data.DataLoader,
#     dt: float,
#     n_batches: int = 100,
# ) -> float:
#     """Compute fraction of eigenvalues with real part < -1/dt.

#     These correspond to dynamics that are 'too fast' relative to the sampling
#     rate and suggest numerical instability.

#     Args:
#         lit_model: Lightning model (already on device, in eval mode).
#         val_dataloader: Validation data loader.
#         dt: Time step of the data.
#         n_batches: Maximum number of batches to evaluate.

#     Returns:
#         Fraction of eigenvalues that are too fast.
#     """
#     device = next(lit_model.parameters()).device
#     num_too_fast = 0
#     total_eigs = 0
#     threshold = -1.0 / dt
#     with torch.no_grad():
#         for i, batch in enumerate(val_dataloader):
#             if i >= n_batches:
#                 break
#             batch = batch.to(device)
#             lit_model.dt = dt
#             pred_jacs = lit_model.compute_jacobians(batch)
#             eigs_real = torch.linalg.eigvals(pred_jacs).real.flatten()
#             num_too_fast += torch.sum(eigs_real <= threshold).float().item()
#             total_eigs += len(eigs_real)
#     return num_too_fast / total_eigs if total_eigs > 0 else 0.0


# def compute_val_loop_closure_loss(
#     lit_model: torch.nn.Module,
#     val_dataloader: torch.utils.data.DataLoader,
#     n_batches: int = 100,
# ) -> float:
#     """Compute mean loop closure MSE over validation data.

#     Calls ``lit_model.loop_closure_model_step(batch)`` to get loop closure loss.

#     Args:
#         lit_model: Lightning model (already on device, in eval mode).
#         val_dataloader: Validation data loader.
#         n_batches: Maximum number of batches to evaluate.

#     Returns:
#         Mean loop closure MSE.
#     """
#     device = next(lit_model.parameters()).device
#     losses = []
#     with torch.no_grad():
#         for i, batch in enumerate(val_dataloader):
#             if i >= n_batches:
#                 break
#             batch = batch.to(device)
#             ret = lit_model.loop_closure_model_step(batch)
#             losses.append(ret["mse"].float().item())
#     return sum(losses) / len(losses) if losses else float("inf")


def compute_all_diagnostics(
    lit_model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    dt: float,
    n_batches: int = 100,
    use_loop_closure: bool = True,
    verbose: bool = False,
) -> DiagnosticMetrics:
    """Compute all diagnostic metrics in a single pass over validation data.

    Args:
        lit_model: Lightning model (already on device, in eval mode).
        val_dataloader: Validation data loader.
        dt: Time step of the data.
        n_batches: Maximum number of batches to evaluate.
        use_loop_closure: Whether to compute loop closure loss.

    Returns:
        DiagnosticMetrics with all fields populated.
    """
    device = next(lit_model.parameters()).device
    one_step_errors = []
    loop_closure_losses = []
    num_eigs_too_fast = 0
    total_eigs = 0
    trajectory_losses = []
    threshold = -1.0 / dt

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=n_batches, disable=not verbose):
            if i >= n_batches:
                break
            batch = batch.to(device)
            lit_model.dt = dt

            # One-step error (teacher-forced)
            traj_ret = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=1)
            print(traj_ret["metric_vals"]["mase"])
            one_step_errors.append(traj_ret["loss"].float().item())

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

    n = len(one_step_errors)
    return DiagnosticMetrics(
        one_step_error=sum(one_step_errors) / n if n > 0 else float("inf"),
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


def fails_one_step_criterion(
    metrics: DiagnosticMetrics, persistence_baseline: float
) -> bool:
    """C1: Does the model fail the one-step error criterion?

    A model fails if its one-step prediction error exceeds the persistence
    baseline (predicting the previous value).

    Args:
        metrics: Diagnostic metrics for the model.
        persistence_baseline: Persistence baseline MSE.

    Returns:
        True if the model FAILS (should be excluded).
    """
    return metrics.one_step_error > persistence_baseline


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
