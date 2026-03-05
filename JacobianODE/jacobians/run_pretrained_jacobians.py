"""Training script for JacobianODE with a pre-trained encoder.

This script provides a Hydra entry point that:
1. Loads a pre-trained encoder from a W&B checkpoint
2. Generates data (matching the encoder's data config)
3. Builds a LitLatentJacobianODE with the pretrained encoder
4. Trains the model

Designed for use with ``hydra --multirun`` + SLURM for parallel sweeps.
"""

from __future__ import annotations

import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .core import initialize_config, seed_everything
from .data import make_trajectories, postprocess_data, create_dataloaders
from .training import train_model, log_training_info

log = logging.getLogger("PretrainedJacobianLogger")


@hydra.main(version_base="1.3", config_path="conf", config_name="pretrained_config")
def train_pretrained_jacobians(cfg: DictConfig) -> None:
    """Train a JacobianODE model using a pre-trained encoder."""
    # ----------------------------------------
    # INITIAL SETUP
    # ----------------------------------------
    log.info("Starting Pretrained-Encoder JacobianODE training")

    torch.set_float32_matmul_precision("high")
    num_gpus = torch.cuda.device_count()
    log.info(f"Number of available GPUs: {num_gpus}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize configuration (non-mutating)
    cfg = initialize_config(cfg)

    # ----------------------------------------
    # LOAD PRE-TRAINED ENCODER
    # ----------------------------------------
    from ..encoder_only.pretrained import load_pretrained_encoder

    adapter, encoder_cfg, encoder_run = load_pretrained_encoder(
        project=cfg.pretrained_encoder.project,
        run_id=cfg.pretrained_encoder.run_id,
        save_dir=cfg.pretrained_encoder.get("save_dir"),
        freeze=cfg.pretrained_encoder.get("freeze", True),
        verbose=True,
    )

    log.info(
        f"Loaded pretrained encoder: {type(adapter.encoder).__name__}, "
        f"n_latent={adapter.n_latent}, "
        f"context_margin={adapter.context_margin}, "
        f"frozen={cfg.pretrained_encoder.get('freeze', True)}"
    )

    # ----------------------------------------
    # GENERATE DATA
    # ----------------------------------------
    seed_everything(cfg.data.flow.random_state)

    eq, sol, dt = make_trajectories(cfg)
    values_raw = sol["values"]

    # ----------------------------------------
    # POSTPROCESS DATA
    # ----------------------------------------
    result = postprocess_data(cfg, values_raw)
    values = result.values
    mu = result.mu
    sigma = result.sigma
    noise_scale_factor = result.noise_scale_factor

    cfg.data.postprocessing.noise_scale_factor = noise_scale_factor
    cfg.data.postprocessing.mu = mu
    cfg.data.postprocessing.sigma = sigma

    # ----------------------------------------
    # CREATE DATALOADERS
    # ----------------------------------------
    train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
        cfg, values
    )

    # ----------------------------------------
    # SET UP WANDB
    # ----------------------------------------
    entity = cfg.wandb_entity

    # Auto-generate project name if not specified
    data_cls = cfg.data.flow._target_.split(".")[-1]
    project = cfg.wandb_project or f"{data_cls}__PretrainedEncoderJacODE"

    # Build a descriptive run name
    encoder_type = type(adapter.encoder).__name__
    n_latent = adapter.n_latent
    freeze_tag = "frozen" if cfg.pretrained_encoder.get("freeze", True) else "unfrozen"
    lc_weight = cfg.training.lightning.loop_closure_weight
    pred_steps = cfg.model.prediction_steps

    name = (
        f"{data_cls}__{encoder_type}__n{n_latent}__{freeze_tag}"
        f"__lc{lc_weight}__pred{pred_steps}"
    )

    # ----------------------------------------
    # MAKE MODEL
    # ----------------------------------------
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number + 1)

    jac_model = instantiate(cfg.model.params)

    extra_kwargs = {}
    if "prediction_steps" in cfg.model:
        extra_kwargs["prediction_steps"] = cfg.model.prediction_steps
    if "encoder_warmup_epochs" in cfg.model:
        extra_kwargs["encoder_warmup_epochs"] = cfg.model.encoder_warmup_epochs
    if cfg.model.get("jac_window_stride") is not None:
        extra_kwargs["jac_window_stride"] = cfg.model.jac_window_stride

    lit_model = instantiate(
        cfg.training.lightning,
        model=jac_model,
        encoder=adapter,
        dt=dt,
        save_dir=cfg.training.logger.save_dir,
        mu=float(mu),
        sigma=float(sigma),
        noise_scale_factor=float(noise_scale_factor),
        **extra_kwargs,
    )

    lit_model.eq = eq

    log_training_info(train_dataloader, trajs, lit_model, log=log)

    # ----------------------------------------
    # TRAIN MODEL
    # ----------------------------------------
    train_model(
        cfg, lit_model, train_dataloader, val_dataloader,
        name, project, entity=entity,
    )


if __name__ == "__main__":
    train_pretrained_jacobians()
