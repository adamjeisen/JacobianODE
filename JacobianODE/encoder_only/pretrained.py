"""Utilities for loading pre-trained encoders and using them with JacobianODE.

Provides
--------
PretrainedEncoderAdapter
    Wraps a pre-trained encoder + decoder to match the
    ``SequenceAutoencoder``-like interface expected by
    ``LitLatentJacobianODE`` (i.e. ``encode()`` / ``decode()`` /
    ``context_margin`` / ``n_latent``).

load_pretrained_encoder
    Loads a pre-trained ``LitEncoderDecoder`` from a W&B run and returns
    a ``PretrainedEncoderAdapter`` ready for downstream JacobianODE training.

load_pretrained_jac_run
    Loads a ``LitLatentJacobianODE`` run that used a pretrained encoder,
    reconstructing the full model architecture from the W&B config and
    loading the best checkpoint.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class PretrainedEncoderAdapter(nn.Module):
    """Adapter that makes a pre-trained encoder + decoder look like a
    ``SequenceAutoencoder`` for ``LitLatentJacobianODE``.

    ``LitLatentJacobianODE.encode_trajectory`` expects the encoder to have:

    * ``encode(x) -> z``  — maps ``(B, T, D_obs)`` to ``(B, T, n_latent)``
    * ``decode(z) -> x_hat`` — maps latent back to observation space
    * ``context_margin`` (int) — initial timesteps to drop
    * ``n_latent`` (int)

    The pre-trained ``LitEncoderDecoder`` stores:

    * ``encoder`` — a raw sequence encoder (Transformer / SSM / TCN)
    * ``same_state_decoder`` — an ``nn.Sequential`` MLP (z_t → x_t)

    This adapter bridges the two interfaces.

    Parameters
    ----------
    encoder : nn.Module
        The raw sequence encoder from the pre-trained model.
    decoder : nn.Module
        The same-state decoder from the pre-trained model.
    context_margin : int
        Number of initial timesteps the sequence encoder's output lacks
        sufficient causal context (dropped by ``LitLatentJacobianODE``).
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        context_margin: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.context_margin = context_margin

    @property
    def n_latent(self) -> int:
        return self.encoder.n_latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space.

        Parameters
        ----------
        x : torch.Tensor
            Observations of shape ``(B, T, D_obs)``.

        Returns
        -------
        torch.Tensor
            Latent trajectory ``(B, T, n_latent)``.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to observation space.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors of shape ``(B, T, n_latent)`` or
            ``(B, n_latent)``.

        Returns
        -------
        torch.Tensor
            Decoded observations.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def clone(self) -> "PretrainedEncoderAdapter":
        """Return a deep copy (useful for sweep runs with unfrozen encoder)."""
        return copy.deepcopy(self)


def load_pretrained_encoder(
    project: str,
    run_id: str,
    save_dir: Optional[str] = None,
    freeze: bool = True,
    verbose: bool = False,
) -> Tuple[PretrainedEncoderAdapter, DictConfig, Any]:
    """Load a pre-trained encoder from a W&B run.

    Parameters
    ----------
    project : str
        W&B project path (e.g. ``"JacobianODE/Lorenz__EncoderOnly"``).
    run_id : str
        W&B run ID.
    save_dir : str, optional
        Directory containing saved checkpoints.
    freeze : bool
        If ``True``, freeze the adapter so only the downstream Jacobian
        MLP is trainable.
    verbose : bool
        Print loading progress.

    Returns
    -------
    adapter : PretrainedEncoderAdapter
        Encoder + decoder adapter with ``encode()`` / ``decode()`` interface.
    cfg : DictConfig
        Configuration from the pre-trained run.
    run : wandb.Run
        The W&B API run object.

    Raises
    ------
    ValueError
        If the loaded model does not have a ``same_state_decoder``.
    """
    from ..jacobians.checkpoints import load_run

    run, cfg, eq, dt, values, *_, lit_model = load_run(
        project=project,
        run_id=run_id,
        save_dir=save_dir,
        generate_data=False,
        verbose=verbose,
    )

    # Validate: need same_state_decoder for obs-space decoding
    if not (
        hasattr(lit_model, "same_state_decoder")
        and getattr(lit_model, "use_same_state_decoder", False)
    ):
        raise ValueError(
            "Pre-trained model must have use_same_state_decoder=True "
            "to provide obs-space decoding for JacobianODE training."
        )

    encoder = lit_model.encoder
    decoder = lit_model.same_state_decoder
    context_margin = getattr(lit_model, "context_margin", 0)

    adapter = PretrainedEncoderAdapter(encoder, decoder, context_margin)

    if freeze:
        adapter.requires_grad_(False)

    if verbose:
        n_params = sum(p.numel() for p in adapter.parameters())
        n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        logger.info(
            f"Loaded encoder: {type(encoder).__name__}, "
            f"n_latent={adapter.n_latent}, "
            f"context_margin={context_margin}, "
            f"params={n_params:,} ({n_trainable:,} trainable)"
        )

    return adapter, cfg, run


# ---------------------------------------------------------------
# Loading a JacobianODE run that used a pretrained encoder
# ---------------------------------------------------------------

@dataclass
class PretrainedJacRunResult:
    """All objects needed for evaluation of a pretrained-encoder JacobianODE run."""

    run: Any
    cfg: DictConfig
    eq: Any
    dt: float
    values: Any
    mu: float
    sigma: float
    noise_scale_factor: float
    train_dl: Any
    val_dl: Any
    test_dl: Any
    trajs: Dict[str, Any]
    test_trajs_full: Optional[Any]
    adapter: PretrainedEncoderAdapter
    lit_model: Any
    n_obs: int


def _build_decoder_from_config(
    n_latent: int,
    n_obs: int,
    hidden_dim: int = 128,
    n_layers: int = 2,
) -> nn.Sequential:
    """Reconstruct the same-state decoder MLP architecture.

    Mirrors ``LitEncoderDecoder._build_decoder``: Linear → GELU pairs
    followed by a final Linear projection.
    """
    layers: list[nn.Module] = []
    in_dim = n_latent
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, n_obs))
    return nn.Sequential(*layers)


def load_pretrained_jac_run(
    project: str,
    run_id: str,
    save_dir: Optional[str] = None,
    entity: str = "JacobianODE",
    verbose: bool = True,
) -> PretrainedJacRunResult:
    """Load a ``LitLatentJacobianODE`` run that used a pretrained encoder.

    Reconstructs the full pipeline from the W&B run config:
    encoder adapter, Jacobian MLP, data, and checkpoint weights.

    Parameters
    ----------
    project : str
        W&B project name (e.g. ``"Lorenz__PretrainedEncoderJacODE"``).
    run_id : str
        W&B run ID.
    save_dir : str, optional
        Local directory containing checkpoints. If ``None``, read from
        the run config.
    entity : str
        W&B entity (team/user). Defaults to ``"JacobianODE"``.
    verbose : bool
        Print progress.

    Returns
    -------
    PretrainedJacRunResult
        Dataclass with all objects needed for evaluation.
    """
    import wandb
    from hydra.utils import instantiate

    from ..jacobians.core.reproducibility import seed_everything
    from ..jacobians.data.dataloaders import create_dataloaders
    from ..jacobians.data.processing import postprocess_data
    from ..jacobians.data.trajectory import make_trajectories

    # ---- Step 1: load W&B run + config ----
    api = wandb.Api(timeout=90)
    run = api.run(f"{entity}/{project}/{run_id}")
    cfg = OmegaConf.create(run.config)

    if save_dir is None:
        save_dir = cfg.training.logger.save_dir

    if verbose:
        print(f"Loaded config from run {run.name} (id={run.id})")

    # ---- Step 2: generate data ----
    seed_everything(cfg.data.flow.random_state)
    eq, sol, dt = make_trajectories(cfg, verbose=verbose)

    result = postprocess_data(cfg, sol["values"])
    values = result.values
    mu = result.mu
    sigma = result.sigma
    noise_scale_factor = result.noise_scale_factor

    train_dl, val_dl, test_dl, trajs = create_dataloaders(
        cfg, values, verbose=verbose, return_full_obs=True,
    )
    n_obs = trajs["train_trajs"].sequence.shape[-1]
    test_trajs_full = trajs.get("test_trajs_full")
    if test_trajs_full is not None:
        test_trajs_full = test_trajs_full.sequence

    if verbose:
        print(f"n_obs = {n_obs}, test_trajs_full = "
              f"{test_trajs_full.shape if test_trajs_full is not None else 'N/A'}")

    # ---- Step 3: load checkpoint state dict and infer architecture ----
    # The run config's model.encoder may contain Hydra *template* defaults
    # that don't match the actual pretrained encoder (e.g. d_model=64 in
    # config vs d_model=128 in the actual encoder).  We infer the true
    # architecture from the checkpoint weight shapes.
    import os

    from ..jacobians.checkpoints import get_all_checkpoints

    ckpt_files, ckpt_dir = get_all_checkpoints(run, cfg, save_dir)
    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    loaded_state = ckpt_data["state_dict"]

    enc_cfg = cfg.model.encoder
    context_margin = int(enc_cfg.get("context_margin", 0))

    # Infer encoder dimensions from checkpoint weights
    d_model = loaded_state["encoder.encoder.input_proj.weight"].shape[0]
    d_state = loaded_state["encoder.encoder.layers.0.nu_log"].shape[1]
    n_latent = loaded_state["encoder.encoder.to_latent.weight"].shape[0]
    ffn_dim = loaded_state["encoder.encoder.layers.0.ffn_gate.weight"].shape[0]
    ffn_expand = ffn_dim // d_model
    n_layers = 0
    while f"encoder.encoder.layers.{n_layers}.D" in loaded_state:
        n_layers += 1

    # Override config with inferred values so instantiate builds the
    # correct architecture
    enc_cfg_override = OmegaConf.create(dict(enc_cfg))
    enc_cfg_override.d_model = d_model
    enc_cfg_override.d_state = d_state
    enc_cfg_override.n_latent = n_latent
    enc_cfg_override.n_layers = n_layers
    enc_cfg_override.ffn_expand = ffn_expand

    seq_autoencoder = instantiate(enc_cfg_override, n_input=n_obs)
    encoder = seq_autoencoder.encoder  # raw SSM/Transformer/TCN

    # Infer decoder dimensions from checkpoint weights
    decoder_keys = sorted(
        k for k in loaded_state if k.startswith("encoder.decoder.")
    )
    # Count Linear layers (weight keys) to get n_layers
    weight_keys = [k for k in decoder_keys if k.endswith(".weight")]
    decoder_n_layers = len(weight_keys) - 1  # last one is output projection
    decoder_hidden = loaded_state[weight_keys[0]].shape[0]

    decoder = _build_decoder_from_config(n_latent, n_obs, decoder_hidden, decoder_n_layers)

    adapter = PretrainedEncoderAdapter(encoder, decoder, context_margin)

    if verbose:
        print(f"Encoder: {type(encoder).__name__}, d_model={d_model}, "
              f"n_latent={n_latent}, n_layers={n_layers}, "
              f"context_margin={context_margin}")

    # ---- Step 4: build LitLatentJacobianODE ----
    jac_model = instantiate(cfg.model.params)

    lit_model = instantiate(
        cfg.training.lightning,
        model=jac_model,
        encoder=adapter,
        dt=dt,
        save_dir=save_dir,
        mu=float(mu),
        sigma=float(sigma),
        noise_scale_factor=float(noise_scale_factor),
        prediction_steps=int(cfg.model.prediction_steps),
    )

    # ---- Step 5: load checkpoint ----
    # Use strict=False for buffers like pos_enc.pe that may not be in the
    # checkpoint (registered buffers are not always saved).
    lit_model.load_state_dict(loaded_state, strict=False)
    lit_model.eval()

    if verbose:
        total = sum(p.numel() for p in lit_model.parameters())
        trainable = sum(p.numel() for p in lit_model.parameters() if p.requires_grad)
        print(f"Model loaded: {total:,} params ({trainable:,} trainable)")

    return PretrainedJacRunResult(
        run=run,
        cfg=cfg,
        eq=eq,
        dt=dt,
        values=values,
        mu=mu,
        sigma=sigma,
        noise_scale_factor=noise_scale_factor,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        trajs=trajs,
        test_trajs_full=test_trajs_full,
        adapter=adapter,
        lit_model=lit_model,
        n_obs=n_obs,
    )
