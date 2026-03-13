"""Main training script for JacobianODE.

This script provides the entry point for training Jacobian-based ODE models
using Hydra for configuration management.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
import traceback

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Import from new modular structure
from .core import in_ipython, initialize_config, seed_everything
from .data import make_trajectories, postprocess_data, create_dataloaders
from .training import make_model, train_model, log_training_info, setup_wandb

# Set up logging
log = logging.getLogger("JacobianLogger")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def train_jacobians(cfg: DictConfig) -> None:
    """Train a JacobianODE model.

    This function orchestrates the complete training pipeline:
    1. Initial setup (GPU, logging, config)
    2. Data generation and preprocessing
    3. DataLoader creation
    4. W&B setup
    5. Model creation
    6. Training

    Args:
        cfg: Hydra configuration object containing all training parameters.
    """
    # ----------------------------------------
    # DIAGNOSTIC: Capture crashes and exceptions
    # ----------------------------------------
    # Faulthandler dumps Python traceback on SIGSEGV, SIGFPE, etc.
    diag_dir = os.getcwd()
    try:
        _fh_file = open(os.path.join(diag_dir, "faulthandler_dump.txt"), "w")
        faulthandler.enable(file=_fh_file, all_threads=True)
    except OSError:
        faulthandler.enable(all_threads=True)  # fallback to stderr

    try:
        _run_training(cfg)
    except Exception as e:
        err_path = os.path.join(diag_dir, "error_traceback.txt")
        with open(err_path, "w") as f:
            traceback.print_exc(file=f)
        log.error(f"Fatal error (see {err_path}): {e}", exc_info=True)
        raise


def _run_training(cfg: DictConfig) -> None:
    """Inner training logic (separated for diagnostic try/except)."""
    # ----------------------------------------
    # INITIAL SETUP
    # ----------------------------------------
    log = logging.getLogger("JacobianLogger")
    log.info("Starting JacobianODE training")

    torch.set_float32_matmul_precision("high")
    num_gpus = torch.cuda.device_count()

    log.info(f"Number of available GPUs: {num_gpus}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize configuration (non-mutating)
    log.info("Initializing config...")
    cfg = initialize_config(cfg)

    # ----------------------------------------
    # GENERATE DATA
    # ----------------------------------------
    # Set seeds for reproducibility
    seed_everything(cfg.data.flow.random_state)

    log.info("Generating trajectories (this may take several minutes if cache is cold)...")
    log.info("Calling make_trajectories...")
    eq, sol, dt = make_trajectories(cfg, verbose=True)
    log.info("make_trajectories returned")
    values_raw = sol["values"]

    # Select which solution to use for noise scaling
    raw_values_noise = None
    if (
        cfg.data.data_type == "wmtask"
        and cfg.data.dataset_loader.model_to_load != "final"
    ):
        temp_cfg = cfg.copy()
        temp_cfg.data.dataset_loader.model_to_load = "final"
        _, sol_noise, _ = make_trajectories(temp_cfg)
        raw_values_noise = sol_noise["values"]

    # ----------------------------------------
    # POSTPROCESS DATA
    # ----------------------------------------
    result = postprocess_data(cfg, values_raw, raw_values_to_use_for_noise=raw_values_noise)
    values = result.values
    mu = result.mu
    sigma = result.sigma
    noise_scale_factor = result.noise_scale_factor

    # Store postprocessing metadata on config so it gets logged to W&B.
    # This allows load_run to correctly reconstruct the data pipeline.
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
    prompt_entity = cfg.wandb_entity is None
    name, project, entity = setup_wandb(
        cfg,
        trajs,
        raw_values_to_use_for_noise=raw_values_noise,
        prompt_entity=prompt_entity,
    )

    # ----------------------------------------
    # MAKE MODEL
    # ----------------------------------------
    # Re-seed with run_number offset for model initialization
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number + 1)

    if "NeuralODE" in cfg.model.params._target_:
        cfg.model.params.dt = float(dt)

    if cfg.training.lightning.use_base_deriv_pt:
        x0 = trajs["train_trajs"].sequence.mean(dim=(0, 1))
    else:
        x0 = None

    if cfg.data.train_test_params.delay_embedding_params.n_delays > 1:
        lit_model = make_model(cfg, dt, eq=None, project=project, mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor, verbose=True)
    else:
        lit_model = make_model(cfg, dt, eq=eq, project=project, x0=x0, mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor, verbose=True)

    # Log training information
    log_training_info(train_dataloader, trajs, lit_model, log=log)

    # ----------------------------------------
    # TRAIN MODEL
    # ----------------------------------------
    wandb_group = cfg.get("wandb_group") or None
    train_model(cfg, lit_model, train_dataloader, val_dataloader, name, project, entity=entity, group=wandb_group)


if __name__ == "__main__":
    train_jacobians()
