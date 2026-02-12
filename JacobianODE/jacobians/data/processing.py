"""Data processing utilities for JacobianODE."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

from .filtering import filter_data

logger = logging.getLogger(__name__)


def postprocess_data(
    cfg: DictConfig,
    raw_values: Union[np.ndarray, torch.Tensor],
    raw_values_to_use_for_noise: Optional[Union[np.ndarray, torch.Tensor]] = None,
    scale_noise: bool = True,
    dt: Optional[float] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Post-process trajectory data by adding noise and/or filtering.

    Applies observation noise and optional filtering to the trajectory data.
    Noise can be scaled based on the data magnitude.

    Args:
        cfg: Configuration object containing postprocessing parameters.
        raw_values: Raw trajectory values - must be of shape (n_traj, time_steps, n_dim).
        raw_values_to_use_for_noise: Alternative raw values to use for noise scaling.
            Defaults to None.
        scale_noise: Whether to scale noise based on data magnitude. Defaults to True.
        dt: Time step for filtering. Required if filter_data is True. Defaults to None.

    Returns:
        Processed trajectory values with same type as input.

    Example:
        >>> values = postprocess_data(cfg, sol['values'])
        >>> print(f"Processed values shape: {values.shape}")

    Note:
        The noise level is scaled by the average L2 norm of the data divided by
        sqrt(n_dim) to make it dimension-independent.
    """
    obs_noise = cfg.data.postprocessing.obs_noise

    if scale_noise:
        if raw_values_to_use_for_noise is None:
            noise_ref = raw_values
        else:
            noise_ref = raw_values_to_use_for_noise

        noise_scale_factor = float(
            np.linalg.norm(noise_ref, axis=-1).mean() / np.sqrt(noise_ref.shape[-1])
        )
        # Scale noise by average norm per dimension
        obs_noise = obs_noise * noise_scale_factor
        cfg.data.postprocessing.obs_noise = obs_noise
        cfg.training.lightning.obs_noise_scale *= noise_scale_factor

    values = raw_values.copy() if isinstance(raw_values, np.ndarray) else raw_values.clone()

    if obs_noise > 0:
        if isinstance(values, torch.Tensor):
            values = values + torch.randn_like(values) * obs_noise
        else:
            values = values + np.random.normal(0, obs_noise, values.shape)

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

    return values


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
