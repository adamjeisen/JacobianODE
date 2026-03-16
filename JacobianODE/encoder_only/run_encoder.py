"""Training script for encoder-only representation learning.

Entry point for training a sequence encoder (Transformer / SSM / TCN)
without the downstream JacobianODE, using Hydra for configuration management.

Typical usage
-------------
# Single run (from the encoder_only/ directory)
python -m JacobianODE.encoder_only.run_encoder model=transformer

# Multirun sweep (grid over fnn_weight)
python -m JacobianODE.encoder_only.run_encoder --multirun \\
    ++training.lightning.fnn_weight=0,0.001,0.01,0.1
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
import traceback

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, read_write

from ..jacobians.core import seed_everything
from ..jacobians.data import create_dataloaders, make_trajectories, postprocess_data
from ..jacobians.training import train_model
from ..jacobians.training.logging import (
    _deduplicate_run_name,
    _resolve_entity,
    log_training_info,
)

log = logging.getLogger("EncoderLogger")


def _make_run_name(cfg: DictConfig) -> str:
    """Derive a human-readable W&B run name from the encoder config."""
    # Data class
    if cfg.data.data_type == "dysts":
        data_cls = cfg.data.flow._target_.split(".")[-1]
    elif cfg.data.data_type in ("custom", "wmtask"):
        data_cls = cfg.data.get("name", cfg.data.data_type)
    else:
        data_cls = cfg.data.data_type

    # Encoder type (last component of _target_)
    enc_target = cfg.model.encoder._target_.split(".")[-1]

    # Key model params
    n_latent = cfg.model.encoder.n_latent
    parts = [
        data_cls,
        enc_target,
        f"n_latent_{n_latent}",
    ]

    # Decoder modes
    if cfg.model.get("use_same_state_decoder", True):
        parts.append("same")
    if cfg.model.get("use_next_state_decoder", False):
        parts.append("next")

    # Regularisation weights (only non-zero)
    for key in ("fnn_weight", "amplification_weight", "decov_weight", "jacobian_nuclear_weight"):
        val = cfg.training.lightning.get(key, 0.0)
        if val and val != 0.0:
            parts.append(f"{key}_{val:.4g}")

    # Run number for replicate identification
    parts.append(f"run_{cfg.training.run_number}")

    return "__".join(parts)


def _make_project(cfg: DictConfig) -> str:
    if cfg.data.data_type == "dysts":
        data_cls = cfg.data.flow._target_.split(".")[-1]
    elif cfg.data.data_type in ("custom", "wmtask"):
        data_cls = cfg.data.get("name", cfg.data.data_type)
    else:
        data_cls = cfg.data.data_type
    return f"{data_cls}__EncoderOnly"


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def train_encoder(cfg: DictConfig) -> None:
    """Hydra entry point: runs encoder training with explicit traceback on error."""
    try:
        _train_encoder_impl(cfg)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def _train_encoder_impl(cfg: DictConfig) -> None:
    """Internal implementation of encoder training. See train_encoder for docs."""
    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    log.info("Starting encoder-only training")
    torch.set_float32_matmul_precision("high")
    log.info(f"GPUs available: {torch.cuda.device_count()}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # NOTE: Do NOT call initialize_config() here — that function is designed
    # for the JacobianODE pipeline and overwrites the Lightning _target_ to
    # LitLatentJacobianODE, breaking encoder-only training.

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------
    seed_everything(cfg.data.flow.random_state)

    # Stagger concurrent SLURM jobs to avoid NFS contention.
    # Use a non-seeded source so seed_everything() doesn't make all jobs identical.
    _rng = random.Random(os.getpid() ^ int(time.time() * 1000))
    jitter = _rng.uniform(0, 3.0)
    log.info(f"NFS jitter: sleeping {jitter:.2f}s before data loading (pid={os.getpid()})")
    time.sleep(jitter)

    log.info("Calling make_trajectories...")
    eq, sol, dt = make_trajectories(cfg)
    log.info("make_trajectories returned")
    values_raw = sol["values"]

    log.info("Calling postprocess_data...")
    result = postprocess_data(cfg, values_raw)
    values = result.values
    mu, sigma = result.mu, result.sigma
    noise_scale_factor = result.noise_scale_factor

    # Store postprocessing metadata on config (logged to W&B / load_run)
    cfg.data.postprocessing.noise_scale_factor = noise_scale_factor
    cfg.data.postprocessing.mu = mu
    cfg.data.postprocessing.sigma = sigma

    # ------------------------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------------------------
    log.info("Creating dataloaders...")
    train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
        cfg, values
    )

    # Infer observation dimension from training data
    n_obs: int = trajs["train_trajs"].sequence.shape[-1]
    log.info(f"Observation dimension: {n_obs}")

    # ------------------------------------------------------------------
    # WANDB
    # ------------------------------------------------------------------
    log.info("Setting up W&B...")
    prompt_entity = cfg.wandb_entity is None
    name = _make_run_name(cfg)
    project = _make_project(cfg)

    if cfg.get("wandb_project"):
        project = cfg.wandb_project

    entity = _resolve_entity(log, prompt_entity)
    name = _deduplicate_run_name(name, project, entity, log)

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    log.info("Creating model...")
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number + 1)

    # Instantiate encoder, injecting runtime n_input
    encoder = instantiate(cfg.model.encoder, n_input=n_obs)

    # Collect model-level kwargs from cfg.model
    model_kwargs = dict(
        encoder=encoder,
        n_obs=n_obs,
        context_margin=int(cfg.model.get("context_margin", 0)),
        next_state_burn_in=int(cfg.model.get("next_state_burn_in", 0)),
        use_same_state_decoder=bool(cfg.model.get("use_same_state_decoder", True)),
        use_next_state_decoder=bool(cfg.model.get("use_next_state_decoder", False)),
        decoder_hidden_dim=int(cfg.model.get("decoder_hidden_dim", 128)),
        decoder_n_layers=int(cfg.model.get("decoder_n_layers", 2)),
        k_steps_ahead=int(cfg.model.get("k_steps_ahead", 1)),
        n_obs_pred=cfg.model.get("n_obs_pred", None),
    )

    # Instantiate Lightning model; cfg.training.lightning provides _target_ + training HPs
    lit_model = instantiate(cfg.training.lightning, **model_kwargs)

    log_training_info(train_dataloader, trajs, lit_model, log=log)

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    # When both decoders are disabled, early stopping / checkpointing should
    # monitor a regularization metric. Default to val/amplification_loss.
    use_same = bool(cfg.model.get("use_same_state_decoder", True))
    use_next = bool(cfg.model.get("use_next_state_decoder", False))
    if not use_same and not use_next:
        with read_write(cfg):
            cfg.training.early_stopping.monitor = "val/amplification_loss"
            cfg.training.model_checkpoint.monitor = "val/amplification_loss"
        log.info(
            "Both decoders disabled; monitoring val/amplification_loss for "
            "early stopping and checkpointing"
        )

    log.info("Starting training...")
    wandb_group = cfg.get("wandb_group") or None

    train_model(
        cfg,
        lit_model,
        train_dataloader,
        val_dataloader,
        name,
        project,
        entity=entity,
        group=wandb_group,
    )


if __name__ == "__main__":
    try:
        train_encoder()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
