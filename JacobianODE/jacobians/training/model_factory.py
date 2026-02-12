"""Model factory for JacobianODE."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def make_model(
    cfg: DictConfig,
    dt: float,
    eq: Optional[Any] = None,
    project: Optional[str] = None,
    x0: Optional[Union[np.ndarray, torch.Tensor]] = None,
    save_dir: Optional[str] = None,
    mu: float = 0.0,
    sigma: float = 1.0,
    verbose: bool = False,
) -> Any:
    """Create and initialize the model for training.

    Instantiates the main model and optional derivative model, handling various
    model types and pretrained model loading.

    Args:
        cfg: Configuration object containing model parameters.
        dt: Time step size for integration.
        eq: Equation/model object for dynamical systems. Defaults to None.
        project: W&B project name for loading pretrained models. Defaults to None.
        x0: Initial state to use as base point for model. Defaults to None.
        save_dir: Directory to save model checkpoints. Defaults to None.
        mu: Mean for normalization. Defaults to 0.0.
        sigma: Standard deviation for normalization. Defaults to 1.0.
        verbose: Whether to print progress information. Defaults to False.

    Returns:
        The PyTorch Lightning model (lit_model).

    Example:
        >>> lit_model = make_model(cfg, dt, eq=eq, mu=mu, sigma=sigma)
        >>> # Model is ready for training
    """
    # Instantiate the Jacobian model from config
    jac_model = instantiate(cfg.model.params)

    # Instantiate the Lightning model wrapper
    lit_model = instantiate(
        cfg.training.lightning,
        model=jac_model,
        dt=dt,
        save_dir=cfg.training.logger.save_dir if save_dir is None else save_dir,
        base_pt_init=x0,
        mu=mu,
        sigma=sigma,
    )

    # Attach the equation object for Jacobian computation during validation
    lit_model.eq = eq

    if verbose:
        total_params = sum(p.numel() for p in lit_model.parameters())
        logger.info(f"Created model with {total_params:,} parameters")

    return lit_model
