"""Main training script for JacobianODE.

This script provides the entry point for training Jacobian-based ODE models
using Hydra for configuration management.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import random
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
    """Hydra-decorated CLI shim for :func:`train`."""
    return train(cfg)


def train(cfg: DictConfig) -> Optional[float]:
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
        Returns ``None`` if the run short-circuits on a done-marker.
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


def _run_training(cfg: DictConfig) -> Optional[float]:
    """Inner training logic (separated for diagnostic try/except).

    Library consumers should call :func:`train` instead — it adds the
    faulthandler / error-traceback diagnostics around this function.
    """
    # ----------------------------------------
    # INITIAL SETUP
    # ----------------------------------------
    log = logging.getLogger("JacobianLogger")
    log.info("Starting JacobianODE training")

    # Done-marker short-circuit (full short-circuit, before data loading).
    # Mirrors the check in train_model — duplicated here so a relaunch of an
    # already-finished SLURM array slot exits before paying any data /
    # checkpoint-loading cost. train_model still has the check too as a
    # defense-in-depth backstop for callers that bypass run_jacobians.
    from .training.trainer import _resume_state_dir, read_done_marker, DONE_MARKER_NAME
    _resume_dir = _resume_state_dir(cfg)
    _prior_id = read_done_marker(_resume_dir)
    if _prior_id is not None:
        log.info(
            f"[resume] {_resume_dir / DONE_MARKER_NAME} present (prior wandb "
            f"run id={_prior_id!r}); slot already finished. Exiting."
        )
        return None

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

    # Select which solution to use for noise scaling. The standard wmtask
    # loader path scales noise by the FINAL checkpoint's trajectory norm
    # so noise is comparable across model_to_load choices. Combined loader
    # has no single model_to_load — `sources` is a list — so this branch
    # is skipped and noise is scaled by the concatenated trajectories
    # themselves (which is already the desired behavior: shared scale
    # across conditions).
    raw_values_noise = None
    if (
        cfg.data.data_type == "wmtask"
        and OmegaConf.select(cfg, "data.dataset_loader.model_to_load") is not None
        and cfg.data.dataset_loader.model_to_load != "final"
    ):
        temp_cfg = cfg.copy()
        temp_cfg.data.dataset_loader.model_to_load = "final"
        _, sol_noise, _ = make_trajectories(temp_cfg)
        raw_values_noise = sol_noise["values"]

    # ----------------------------------------
    # POSTPROCESS DATA
    # ----------------------------------------
    # Optional per-condition normalization. When enabled AND the combined
    # loader emitted source_id (one integer per trajectory), postprocess
    # each source's trajectories independently — its own grand-mean center
    # (single scalar across dims, NOT per-dim), its own noise_scale_factor.
    # Each source then has its own observable scale; the model sees both
    # conditions normalized to roughly comparable magnitudes regardless of
    # whether the underlying network produced different output scales.
    # Stitched back into a single `values` tensor preserving original order.
    # Default (across-condition) path is unchanged.
    _normalize_per_condition = bool(
        OmegaConf.select(cfg, "data.postprocessing.normalize_per_condition", default=False)
    )
    _src_ids = sol.get("source_id") if isinstance(sol, dict) else None
    if _normalize_per_condition and _src_ids is not None:
        from .data.processing import postprocess_per_condition
        import numpy as _np
        src_ids_arr = _np.asarray(_src_ids)
        log.info(
            f"per-condition normalization: {len(_np.unique(src_ids_arr))} sources, "
            f"sizes={[int((src_ids_arr == s).sum()) for s in _np.unique(src_ids_arr)]}"
        )
        pc_result = postprocess_per_condition(
            cfg, values_raw, source_id=src_ids_arr,
            raw_values_to_use_for_noise=raw_values_noise,
        )
        values = pc_result.values
        mu = pc_result.mu
        sigma = pc_result.sigma
        noise_scale_factor = pc_result.noise_scale_factor
        for i, s in enumerate(pc_result.source_ids):
            log.info(
                f"  source {s}: mu={pc_result.mu_per_source[i]:.6g}, "
                f"sigma={pc_result.sigma_per_source[i]:.6g}, "
                f"noise_scale_factor={pc_result.noise_scale_factor_per_source[i]:.6g}"
            )
        OmegaConf.update(cfg, "data.postprocessing.mu_per_source", pc_result.mu_per_source, force_add=True)
        OmegaConf.update(cfg, "data.postprocessing.sigma_per_source", pc_result.sigma_per_source, force_add=True)
        OmegaConf.update(cfg, "data.postprocessing.noise_scale_factor_per_source", pc_result.noise_scale_factor_per_source, force_add=True)
    else:
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
    # 'generalized_normalized_mse' loss function.
    # Slice to the "most recent observable" (first n_recent_dims of the
    # delay embedding) so the denominator matches the dims the obs-space
    # losses actually see when reconstruction_mode='most_recent'.
    # For fully-observed data (n_recent_dims == D) this is unchanged.
    from .metrics import compute_generalized_variance
    _n_recent = OmegaConf.select(cfg, "model.n_recent_dims", default=None)
    if _n_recent is not None and _n_recent < values.shape[-1]:
        _values_for_gv = values[..., :_n_recent]
        log.info(f"Computing obs-space gen. variance over first {_n_recent} of {values.shape[-1]} dims (most recent observable)")
    else:
        _values_for_gv = values
    generalized_variance = compute_generalized_variance(_values_for_gv)
    log.info(f"Precomputed obs-space generalized variance det(Cov)^(1/D) = {generalized_variance:.6g}")
    OmegaConf.update(cfg, "data.postprocessing.generalized_variance", generalized_variance, force_add=True)

    # ----------------------------------------
    # CREATE DATALOADERS
    # ----------------------------------------
    # Combined-loader path: sol carries per-trajectory `condition` and
    # `source_id`. Forward both into create_dataloaders so each split
    # carries its slice of conditions and the per-source split is balanced
    # across train/val/test. Both default to None for single-source
    # loaders (the standard path).
    condition = sol.get("condition")
    split_groups = sol.get("source_id")
    train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
        cfg, values, condition=condition, split_groups=split_groups,
    )

    # ----------------------------------------
    # PCA on noisy training delay embeddings
    # ----------------------------------------
    # PCA / FNN auto-dim of the dynamic subspace.
    #   model.n_target_var_threshold (PCA): pick smallest k with cum_var[k-1] >= threshold.
    #   model.n_target_dim_method='fnn'   : use whitened-PCA-FNN k=1 stop-at-min instead.
    # ----------------------------------------
    from .training.autodim import infer_n_target_dims
    autodim_result = infer_n_target_dims(cfg, trajs["train_trajs"].sequence)
    if autodim_result is not None:
        # Write back chosen dims + diagnostic stats. (The function is pure
        # by design — it doesn't mutate cfg, so the caller controls
        # which fields land where.)
        cfg.model.n_target_dims = autodim_result.n_target_dims_total
        cfg.model.params.input_dim = autodim_result.n_target_dims_total
        cfg.model.params.output_dim = autodim_result.n_target_dims_total ** 2
        if autodim_result.is_direct_sum:
            cfg.model.encoder.n_target_dims_per_block = list(
                autodim_result.n_target_dims_per_block
            )
            if autodim_result.method == "fnn":
                OmegaConf.update(
                    cfg, "model.n_target_dims_per_block_fnn_auto",
                    list(autodim_result.n_target_dims_per_block), force_add=True,
                )
            else:  # pca
                OmegaConf.update(
                    cfg, "model.n_target_dims_per_block_pca_auto",
                    list(autodim_result.n_target_dims_per_block), force_add=True,
                )
                OmegaConf.update(
                    cfg, "model.n_target_dims_per_block_pca_cum_var",
                    list(autodim_result.pca_cum_var), force_add=True,
                )
        else:
            if autodim_result.method == "fnn":
                OmegaConf.update(
                    cfg, "model.n_target_dims_fnn_auto",
                    autodim_result.n_target_dims_total, force_add=True,
                )
            else:  # pca
                OmegaConf.update(
                    cfg, "model.n_target_dims_pca_auto",
                    autodim_result.n_target_dims_total, force_add=True,
                )
                OmegaConf.update(
                    cfg, "model.n_target_dims_pca_cum_var",
                    float(autodim_result.pca_cum_var), force_add=True,
                )

    # ----------------------------------------
    # SET UP WANDB
    # ----------------------------------------
    wandb_disabled = bool(OmegaConf.select(cfg, "wandb.disabled", default=False))
    if wandb_disabled:
        log.info("wandb.disabled=true; skipping setup_wandb (no W&B Api / init).")
        from .training.logging import make_run_info
        name, project = make_run_info(cfg)
        if cfg.get("wandb_project"):
            project = cfg.wandb_project
        entity = None
    else:
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
