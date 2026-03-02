"""Configuration loading for the encoder-only module.

Provides a notebook-friendly ``load_encoder_config`` that mirrors the
``load_config`` / ``initialize_config`` pattern from the main jacobians module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_encoder_config(
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load an encoder-only Hydra config.

    Parameters
    ----------
    config_name : str
        Name of the top-level config file (without extension), e.g. ``"config"``.
    overrides : list of str, optional
        Hydra override strings, e.g.::

            overrides=[
                "model=ssm",
                "++training.lightning.fnn_weight=0.01",
                "++data.trajectory_params.num_ics=16",
            ]

    Returns
    -------
    DictConfig
        Loaded configuration.

    Examples
    --------
    >>> from JacobianODE.encoder_only.config import load_encoder_config
    >>> cfg = load_encoder_config(overrides=["model=transformer"])
    >>> cfg.model.encoder.n_latent
    10
    """
    if overrides is None:
        overrides = []

    # config_path is relative to *this* file's directory → encoder_only/conf/
    with hydra.initialize(version_base="1.3", config_path="conf"):
        cfg = hydra.compose(config_name=config_name, overrides=list(overrides))

    logger.debug(f"Loaded encoder config '{config_name}' with {len(overrides)} overrides")
    return cfg
