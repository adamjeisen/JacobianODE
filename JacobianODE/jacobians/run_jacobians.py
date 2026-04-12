"""Main training script for JacobianODE.

This script provides the entry point for training Jacobian-based ODE models
using Hydra for configuration management.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import random
import sys
import time
import traceback
from typing import Optional

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
def train_jacobians(cfg: DictConfig) -> Optional[float]:
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

    Returns:
        The best trajectory validation loss (used as the Optuna objective when
        running with ``hydra/sweeper=optuna``).  Ignored by other sweepers.
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
        return _run_training(cfg)
    except Exception as e:
        err_path = os.path.join(diag_dir, "error_traceback.txt")
        with open(err_path, "w") as f:
            traceback.print_exc(file=f)
        log.error(f"Fatal error (see {err_path}): {e}", exc_info=True)
        raise


def _run_training(cfg: DictConfig) -> float:
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

    # Stagger concurrent SLURM jobs to avoid NFS contention.
    # Use a non-seeded source so seed_everything() doesn't make all jobs identical.
    _rng = random.Random(os.getpid() ^ int(time.time() * 1000))
    _jitter = _rng.uniform(0, 3.0)
    log.info(f"NFS jitter: sleeping {_jitter:.2f}s before data loading (pid={os.getpid()})")
    time.sleep(_jitter)

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

    # Precompute generalized variance (det(Cov)^(1/D)) for the
    # 'generalized_normalized_mse' loss function. Volume-preserving under
    # additive-coupling encoders. Cheap (one-time O(D^3) eigendecomposition).
    from .metrics import compute_generalized_variance
    generalized_variance = compute_generalized_variance(values)
    log.info(f"Precomputed generalized variance det(Cov)^(1/D) = {generalized_variance:.6g}")
    cfg.data.postprocessing.generalized_variance = generalized_variance

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
        lit_model = make_model(cfg, dt, eq=None, project=project, mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor, generalized_variance=generalized_variance, verbose=True)
    else:
        lit_model = make_model(cfg, dt, eq=eq, project=project, x0=x0, mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor, generalized_variance=generalized_variance, verbose=True)

    # Store SLURM timeout in the app config so it gets logged to W&B.
    # This allows downstream tools (run_analytics, discover_sweep_runs) to
    # determine if a crashed run hit the walltime without reading local files.
    try:
        from hydra.core.hydra_config import HydraConfig
        hcfg = HydraConfig.get()
        timeout_min = OmegaConf.select(hcfg.cfg, "hydra.launcher.timeout_min", default=None)
        if timeout_min is not None:
            OmegaConf.set_struct(cfg, False)
            cfg.slurm_timeout_min = int(timeout_min)
            OmegaConf.set_struct(cfg, True)
    except Exception:
        pass  # Not running under Hydra launcher (e.g. local / notebook)

    # Log training information
    log_training_info(train_dataloader, trajs, lit_model, log=log)

    # ----------------------------------------
    # TRAIN MODEL
    # ----------------------------------------
    trainer = train_model(cfg, lit_model, train_dataloader, val_dataloader, name=name)

    # Return objective for Optuna (silently ignored by non-sweep runs).
    # The OptunaCoordinator reads best_so_far from the DB instead, but
    # this return value is still used by Hydra's Optuna sweeper if active.
    best = trainer.callback_metrics.get("trajectory val_loss")
    if best is not None:
        return float(best)
    best = trainer.callback_metrics.get("mean val loss")
    return float(best) if best is not None else float("inf")


if __name__ == "__main__":
    train_jacobians()
