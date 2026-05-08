"""Model factory for JacobianODE."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

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
    noise_scale_factor: float = 1.0,
    generalized_variance: Optional[float] = None,
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
        noise_scale_factor: Factor to scale noise percentages by data magnitude.
            Defaults to 1.0.
        verbose: Whether to print progress information. Defaults to False.

    Returns:
        The PyTorch Lightning model (lit_model).

    Example:
        >>> lit_model = make_model(cfg, dt, eq=eq, mu=mu, sigma=sigma)
        >>> # Model is ready for training
    """
    # Instantiate the Jacobian model from config
    jac_model = instantiate(cfg.model.params)

    # Optional: per-source dynamics. Builds one dynamics MLP per source
    # (count = len(section_condition_values)) and wraps them so each
    # sample is hard-routed to its source's sub-MLP. See
    # JacobianODE.models.per_source_dynamics for the rationale (regime
    # imbalance: awake-vs-anesthesia LFP, etc.).
    use_per_source = bool(OmegaConf.select(
        cfg, "model.per_source_dynamics", default=False
    ))
    if use_per_source:
        section_vals_raw = OmegaConf.select(
            cfg, "model.section_condition_values", default=None
        )
        if section_vals_raw is None:
            raise ValueError(
                "model.per_source_dynamics=True requires "
                "model.section_condition_values (one float per source)."
            )
        section_vals = list(OmegaConf.to_container(section_vals_raw, resolve=True))
        if len(section_vals) < 2:
            raise ValueError(
                f"model.section_condition_values must have ≥2 entries when "
                f"per_source_dynamics=True (got {section_vals})."
            )
        from JacobianODE.models.per_source_dynamics import PerSourceDynamicsMLP
        n_sources = len(section_vals)
        # Build n-1 additional copies (we already have one in jac_model).
        sub_mlps = [jac_model] + [
            instantiate(cfg.model.params) for _ in range(n_sources - 1)
        ]
        jac_model = PerSourceDynamicsMLP(sub_mlps, section_vals)
        if verbose:
            logger.info(
                f"per-source dynamics: {n_sources} sub-MLPs, routing "
                f"by nearest-neighbor in {section_vals}"
            )

    # Build extra kwargs for the Lightning model
    extra_kwargs = {}

    # If the config has an encoder section, instantiate it.
    if "encoder" in cfg.model:
        encoder = instantiate(cfg.model.encoder)
        # DirectSumCouplingEncoder exposes n_target_dims (sum of per-area k's).
        # Downstream _split_latent slices z[..., :cfg.model.n_target_dims], so
        # the encoder's total dyn-dim must match the model-level setting —
        # otherwise the dynamic subspace and the slicer silently disagree.
        _enc_k = getattr(encoder, "n_target_dims", None)
        if _enc_k is not None and "n_target_dims" in cfg.model \
                and cfg.model.n_target_dims is not None \
                and int(cfg.model.n_target_dims) != int(_enc_k):
            raise ValueError(
                f"model.n_target_dims ({cfg.model.n_target_dims}) must equal "
                f"the encoder's n_target_dims ({_enc_k}, = sum of "
                f"n_target_dims_per_block). Fix one of them in the YAML."
            )
        extra_kwargs["encoder"] = encoder
        if "prediction_steps" in cfg.model:
            extra_kwargs["prediction_steps"] = cfg.model.prediction_steps
        # Pass optional latent-model attributes stored in cfg.model so that
        # load_run correctly reconstructs the architecture without manual
        # attribute assignment after the fact.
        if "encoder_warmup_epochs" in cfg.model:
            extra_kwargs["encoder_warmup_epochs"] = cfg.model.encoder_warmup_epochs
        if "dynamics_warmup_epochs" in cfg.model:
            extra_kwargs["dynamics_warmup_epochs"] = cfg.model.dynamics_warmup_epochs
        if "jac_window_stride" in cfg.model and cfg.model.jac_window_stride is not None:
            extra_kwargs["jac_window_stride"] = cfg.model.jac_window_stride
        if "decode_only_recent" in cfg.model:
            extra_kwargs["decode_only_recent"] = cfg.model.decode_only_recent
        if "n_target_dims" in cfg.model and cfg.model.n_target_dims is not None:
            extra_kwargs["n_target_dims"] = cfg.model.n_target_dims
        if "n_recent_dims" in cfg.model and cfg.model.n_recent_dims is not None:
            extra_kwargs["n_recent_dims"] = cfg.model.n_recent_dims
        if "use_vae" in cfg.model:
            extra_kwargs["use_vae"] = cfg.model.use_vae
        if "trajectory_loss_most_recent" in cfg.model:
            extra_kwargs["trajectory_loss_most_recent"] = cfg.model.trajectory_loss_most_recent
        if "decoded_only_pred_loss" in cfg.model:
            extra_kwargs["decoded_only_pred_loss"] = cfg.model.decoded_only_pred_loss
        if "encoder_only_mode" in cfg.model:
            extra_kwargs["encoder_only_mode"] = cfg.model.encoder_only_mode
        if "vae_sample_all_losses" in cfg.model:
            extra_kwargs["vae_sample_all_losses"] = cfg.model.vae_sample_all_losses
        if "kl_warmup_epochs" in cfg.model:
            extra_kwargs["kl_warmup_epochs"] = cfg.model.kl_warmup_epochs
        if "geometric_noise" in cfg.model:
            extra_kwargs["geometric_noise"] = OmegaConf.to_container(
                cfg.model.geometric_noise, resolve=True
            )
        if "jacobian_noise" in cfg.model:
            extra_kwargs["jacobian_noise"] = OmegaConf.to_container(
                cfg.model.jacobian_noise, resolve=True
            )
        # kl_null_weight / kl_dyn_weight are set via cfg.training.lightning
        # (the single source of truth). Model YAML values are kept as
        # documentation defaults only — do NOT pass them here, otherwise
        # they override the training config via extra_kwargs precedence.

    # Instantiate the Lightning model wrapper
    lit_model = instantiate(
        cfg.training.lightning,
        model=jac_model,
        dt=dt,
        save_dir=cfg.training.logger.save_dir if save_dir is None else save_dir,
        base_pt_init=x0,
        mu=mu,
        sigma=sigma,
        noise_scale_factor=noise_scale_factor,
        generalized_variance=generalized_variance,
        **extra_kwargs,
    )

    # Attach the equation object for Jacobian computation during validation
    lit_model.eq = eq

    if verbose:
        total_params = sum(p.numel() for p in lit_model.parameters())
        logger.info(f"Created model with {total_params:,} parameters")

    return lit_model
