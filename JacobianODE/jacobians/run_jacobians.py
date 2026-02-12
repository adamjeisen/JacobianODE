"""Main training script for JacobianODE.

This script provides the entry point for training Jacobian-based ODE models
using Hydra for configuration management.
"""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Import from new modular structure
from .core import in_ipython, initialize_config, seed_everything
from .data import make_trajectories, postprocess_data, normalize_data, create_dataloaders
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
    # INITIAL SETUP
    # ----------------------------------------
    log.info("Starting JacobianODE training")

    torch.set_float32_matmul_precision("high")
    num_gpus = torch.cuda.device_count()

    log.info(f"Number of available GPUs: {num_gpus}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize configuration (non-mutating)
    cfg = initialize_config(cfg)

    # ----------------------------------------
    # GENERATE DATA
    # ----------------------------------------
    # Set seeds for reproducibility
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number)

    eq, sol, dt = make_trajectories(cfg)
    values_raw = sol["values"]

    # Select which solution to use for noise scaling
    raw_values_noise = None
    if (
        cfg.data.data_type == "wmtask"
        and cfg.data.trajectory_params.model_to_load != "final"
    ):
        temp_cfg = cfg.copy()
        temp_cfg.data.trajectory_params.model_to_load = "final"
        _, sol_noise, _ = make_trajectories(temp_cfg)
        raw_values_noise = sol_noise["values"]

    # ----------------------------------------
    # POSTPROCESS DATA
    # ----------------------------------------
    values = postprocess_data(cfg, values_raw, raw_values_to_use_for_noise=raw_values_noise)

    if cfg.data.postprocessing.normalize:
        values, mu, sigma = normalize_data(values)
    else:
        mu = 0
        sigma = 1

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
    if "NeuralODE" in cfg.model.params._target_:
        cfg.model.params.dt = float(dt)

    if cfg.training.lightning.use_base_deriv_pt:
        x0 = trajs["train_trajs"].sequence.mean(dim=(0, 1))
    else:
        x0 = None

    # Re-seed with run_number offset for model initialization
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number)

    if cfg.data.train_test_params.delay_embedding_params.n_delays > 1:
        lit_model = make_model(cfg, dt, eq=None, project=project, mu=mu, sigma=sigma, verbose=True)
    else:
        lit_model = make_model(cfg, dt, eq=eq, project=project, x0=x0, mu=mu, sigma=sigma, verbose=True)

    # Log training information
    log_training_info(train_dataloader, trajs, lit_model, log=log)

    # ----------------------------------------
    # TRAIN MODEL
    # ----------------------------------------
    train_model(cfg, lit_model, train_dataloader, val_dataloader, name, project, entity=entity)


if __name__ == "__main__":
    train_jacobians()
