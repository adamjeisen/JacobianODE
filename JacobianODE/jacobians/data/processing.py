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


class PostprocessPerConditionResult(NamedTuple):
    """Result of postprocess_per_condition.

    ``values``, ``mu``, ``sigma``, and ``noise_scale_factor`` mirror
    :class:`PostprocessResult` so callers that ignore per-source detail
    can read them directly. Per-source detail lives in the ``*_per_source``
    lists, one entry per unique source_id (in ascending order, matching
    ``source_ids``).
    """
    values: Union[np.ndarray, torch.Tensor]
    mu: float
    sigma: float
    noise_scale_factor: float
    mu_per_source: list[float]
    sigma_per_source: list[float]
    noise_scale_factor_per_source: list[float]
    source_ids: list[int]


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


def postprocess_per_condition(
    cfg: DictConfig,
    raw_values: Union[np.ndarray, torch.Tensor],
    source_id: Union[np.ndarray, "list", torch.Tensor],
    raw_values_to_use_for_noise: Optional[Union[np.ndarray, torch.Tensor]] = None,
    scale_noise: bool = True,
    dt: Optional[float] = None,
) -> PostprocessPerConditionResult:
    """Per-source postprocess: noise + filter + normalize, INDEPENDENTLY per
    unique value of ``source_id``.

    For each unique ``s`` in ``source_id``, calls
    :func:`postprocess_data` on ``raw_values[mask]`` (where
    ``mask = source_id == s``), then stitches the results back into a
    single output tensor preserving original row order. Each source ends
    up with its own grand-mean center and noise_scale_factor — useful
    when concatenating trajectories from multiple variants of the same
    underlying system (e.g., the same network at different training
    checkpoints) so the model sees both conditions normalized to roughly
    comparable magnitudes regardless of the underlying scale.

    The top-level scalar ``mu`` / ``sigma`` / ``noise_scale_factor`` are
    the mean across sources — kept for back-compat with downstream code
    (analytics, load_run) that reads single scalars. Per-source detail
    is in the ``*_per_source`` lists.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing postprocessing parameters
        (passed through unchanged to :func:`postprocess_data`).
    raw_values : np.ndarray or torch.Tensor
        Raw trajectory values of shape ``(n_traj, time_steps, n_dim)``.
    source_id : 1-D array-like of int, length ``n_traj``
        Per-trajectory source identifier. Trajectories sharing a source_id
        are postprocessed together as one source.
    raw_values_to_use_for_noise : np.ndarray or torch.Tensor, optional
        Alternative raw values to use for noise scaling (passed through
        per-source). If provided, must have the same first-axis length as
        ``raw_values``. Defaults to None (uses ``raw_values`` for noise).
    scale_noise : bool, optional
        Whether to scale noise based on data magnitude. Defaults to True.
    dt : float, optional
        Time step for filtering. Required if ``filter_data`` is True.

    Returns
    -------
    PostprocessPerConditionResult
        NamedTuple with the per-source postprocessed values, the global
        scalar summaries, and per-source mu/sigma/noise_scale_factor lists.

    Notes
    -----
    The implementation matches the inline behavior previously in
    ``run_jacobians.train()`` (extracted as a callable in the
    ``train_from_arrays`` library API refactor). Output dtype follows
    :func:`postprocess_data`'s behavior — float64 when noise is added
    (np.random.normal upcasts), original dtype otherwise.
    """
    # Coerce source_id to a numpy array so we can do mask operations.
    src_ids_arr = np.asarray(source_id)
    unique_src = np.unique(src_ids_arr)

    # Allocate output container once with the postprocess output dtype
    # (float64 — matches the noisy postprocess output type from
    # np.random.normal). For torch inputs, allocate a torch tensor of
    # matching float64 dtype on the same device.
    if isinstance(raw_values, np.ndarray):
        values = np.empty_like(raw_values, dtype=np.float64)
    else:
        values = torch.empty_like(raw_values, dtype=torch.float64)

    per_source_mu: list[float] = []
    per_source_sigma: list[float] = []
    per_source_nsf: list[float] = []
    source_ids: list[int] = []
    for s in unique_src:
        mask = src_ids_arr == s
        sub_raw = raw_values[mask]
        sub_noise_ref = (
            raw_values_to_use_for_noise[mask]
            if raw_values_to_use_for_noise is not None
            else None
        )
        sub_result = postprocess_data(
            cfg,
            sub_raw,
            raw_values_to_use_for_noise=sub_noise_ref,
            scale_noise=scale_noise,
            dt=dt,
        )
        # Numpy assignment auto-upcasts source → destination dtype.
        # Torch does not — explicitly cast to the destination dtype to
        # avoid the "Index put requires the source and destination dtypes
        # match" RuntimeError when input is torch.float32 but `values` is
        # torch.float64. Inline behavior in run_jacobians.py was implicit
        # numpy-only; this preserves it for both numpy and torch inputs.
        if isinstance(values, np.ndarray):
            values[mask] = sub_result.values
        else:
            values[mask] = sub_result.values.to(values.dtype)
        per_source_mu.append(float(sub_result.mu))
        per_source_sigma.append(float(sub_result.sigma))
        per_source_nsf.append(float(sub_result.noise_scale_factor))
        source_ids.append(int(s))

    # Top-level scalars: mean across sources (back-compat for downstream
    # code that reads single mu/sigma/noise_scale_factor).
    mu = float(np.mean(per_source_mu))
    sigma = float(np.mean(per_source_sigma))
    noise_scale_factor = float(np.mean(per_source_nsf))

    return PostprocessPerConditionResult(
        values=values,
        mu=mu,
        sigma=sigma,
        noise_scale_factor=noise_scale_factor,
        mu_per_source=per_source_mu,
        sigma_per_source=per_source_sigma,
        noise_scale_factor_per_source=per_source_nsf,
        source_ids=source_ids,
    )
