"""Data processing utilities for JacobianODE."""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

from .filtering import filter_data

logger = logging.getLogger(__name__)


class PostprocessResult(NamedTuple):
    """Result of postprocess_data containing processed values and metadata."""
    values: Union[np.ndarray, torch.Tensor]
    mu: float
    sigma: float
    noise_scale_factor: float


def postprocess_data(
    cfg: DictConfig,
    raw_values: Union[np.ndarray, torch.Tensor],
    raw_values_to_use_for_noise: Optional[Union[np.ndarray, torch.Tensor]] = None,
    scale_noise: bool = True,
    dt: Optional[float] = None,
) -> PostprocessResult:
    """Post-process trajectory data by adding noise, filtering, and normalizing.

    Applies observation noise (optionally scaled by data magnitude), optional
    filtering, and optional z-score normalization.  The config is NOT mutated;
    noise percentages remain as the user specified them.

    Args:
        cfg: Configuration object containing postprocessing parameters.
            Uses ``cfg.data.postprocessing.obs_noise`` (noise percentage),
            ``cfg.data.postprocessing.normalize`` (whether to z-score normalize),
            and filter settings.
        raw_values: Raw trajectory values of shape ``(n_traj, time_steps, n_dim)``.
        raw_values_to_use_for_noise: Alternative raw values to use for noise
            scaling.  Defaults to None (uses *raw_values*).
        scale_noise: Whether to scale noise based on data magnitude.
            Defaults to True.
        dt: Time step for filtering.  Required if ``filter_data`` is True.
            Defaults to None.

    Returns:
        PostprocessResult namedtuple with fields:
            - values: Processed trajectory values (same type as input).
            - mu: Mean used for normalization (0.0 if normalize=False).
            - sigma: Std dev used for normalization (1.0 if normalize=False).
            - noise_scale_factor: Factor by which noise percentages were
              multiplied to get absolute noise levels.

    Note:
        The noise level is scaled by the average L2 norm of the data divided by
        ``sqrt(n_dim)`` to make it dimension-independent.
    """
    obs_noise_pct = cfg.data.postprocessing.obs_noise

    # Compute noise scale factor from data magnitude
    noise_scale_factor = 1.0
    if scale_noise and obs_noise_pct > 0:
        if raw_values_to_use_for_noise is None:
            noise_ref = raw_values
        else:
            noise_ref = raw_values_to_use_for_noise

        noise_scale_factor = float(
            np.linalg.norm(noise_ref, axis=-1).mean() / np.sqrt(noise_ref.shape[-1])
        )

    # Absolute noise level = percentage * scale factor
    obs_noise_abs = obs_noise_pct * noise_scale_factor

    values = raw_values.copy() if isinstance(raw_values, np.ndarray) else raw_values.clone()

    if obs_noise_abs > 0:
        if isinstance(values, torch.Tensor):
            values = values + torch.randn_like(values) * obs_noise_abs
        else:
            values = values + np.random.normal(0, obs_noise_abs, values.shape)

    if cfg.data.postprocessing.filter_data:
        if dt is None:
            logger.warning(
                "dt not provided for filtering, using default behavior. "
                "Pass dt parameter for correct filtering."
            )
        values_filtered = np.zeros(values.shape)
        for traj_num in range(values.shape[0]):
            values_filtered[traj_num] = filter_data(
                values[traj_num],
                low_pass=cfg.data.postprocessing.low_pass,
                high_pass=cfg.data.postprocessing.high_pass,
                dt=dt,
            )
        values = values_filtered

    # Normalization
    if cfg.data.postprocessing.normalize:
        values, mu, sigma = normalize_data(values)
    else:
        mu = 0.0
        sigma = 1.0

    return PostprocessResult(
        values=values,
        mu=mu,
        sigma=sigma,
        noise_scale_factor=noise_scale_factor,
    )


def normalize_data(
    values: Union[np.ndarray, torch.Tensor],
) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
    """Normalize data by subtracting mean and dividing by standard deviation.

    Performs z-score normalization on the input data. This is useful for
    stabilizing training when data has varying scales.

    Args:
        values: Input data to normalize. Can be numpy array or torch tensor.

    Returns:
        Tuple of (normalized_values, mu, sigma) where:
            - normalized_values: The normalized data (same type as input)
            - mu: Mean of the original data
            - sigma: Standard deviation of the original data

    Example:
        >>> normalized, mu, sigma = normalize_data(values)
        >>> # To denormalize: original = normalized * sigma + mu
        >>> reconstructed = normalized * sigma + mu

    Note:
        The mean and sigma are computed over the entire array (global statistics).
        Store mu and sigma to denormalize predictions later.
    """
    if isinstance(values, torch.Tensor):
        mu = float(values.mean())
        sigma = float(values.std())
        normalized = (values - mu) / sigma
    else:
        mu = float(values.mean())
        sigma = float(values.std())
        normalized = (values - mu) / sigma

    logger.debug(f"Normalized data: mu={mu:.4f}, sigma={sigma:.4f}")
    return normalized, mu, sigma


def denormalize_data(
    values: Union[np.ndarray, torch.Tensor],
    mu: float,
    sigma: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Denormalize data using stored mean and standard deviation.

    Reverses the z-score normalization performed by normalize_data.

    Args:
        values: Normalized data to denormalize.
        mu: Mean used for normalization.
        sigma: Standard deviation used for normalization.

    Returns:
        Denormalized data in original scale.

    Example:
        >>> normalized, mu, sigma = normalize_data(original)
        >>> # ... use normalized data ...
        >>> recovered = denormalize_data(normalized, mu, sigma)
        >>> np.allclose(original, recovered)  # True
    """
    return values * sigma + mu
