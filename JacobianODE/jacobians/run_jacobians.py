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
        sys.exit(0)

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
    train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
        cfg, values
    )

    # ----------------------------------------
    # PCA on noisy training delay embeddings
    # ----------------------------------------
    # Two opt-in features share the same PCA decomposition:
    #   (1) model.n_target_var_threshold → auto-pick n_target_dims as the
    #       smallest k such that cum_var[k-1] >= threshold.
    #   (2) model.encoder.init_pca_basis → initialize the encoder's final
    #       layer so that, at init, z = V @ x where V is the full PCA basis
    #       (top-k PCs become z_dyn; lower PCs become z_null).
    # If either is set, run the eigh once and use the results for both.
    _n_target_var_thresh = OmegaConf.select(
        cfg, "model.n_target_var_threshold", default=None
    )
    _init_pca_basis = OmegaConf.select(
        cfg, "model.encoder.init_pca_basis", default=False
    )
    # DirectSumCouplingEncoder partitions the input axis into N subsystems via
    # area_indices. With n_target_var_threshold set, we do PCA *per area* (each
    # area's input data is decomposed independently) and pick n_target_dims for
    # each area as the smallest k that captures the threshold of that area's
    # variance. The total n_target_dims is the sum across areas.
    _is_direct_sum = (
        str(OmegaConf.select(cfg, "model.encoder._target_", default=""))
        .endswith("DirectSumCouplingEncoder")
    )
    pca_basis_tensor = None
    if _is_direct_sum and _n_target_var_thresh is not None:
        if _init_pca_basis:
            raise ValueError(
                "init_pca_basis is not supported for DirectSumCouplingEncoder "
                "(would need per-area PCA bases threaded through model_factory). "
                "Set encoder.init_pca_basis=false."
            )
        train_seq = trajs["train_trajs"].sequence  # (N_traj, T, D_embed)
        flat = train_seq.reshape(-1, train_seq.shape[-1]).to(torch.float64)
        area_indices = OmegaConf.to_container(
            cfg.model.encoder.area_indices, resolve=True
        )
        n_target_per_block = []
        cum_at_pick = []
        for i, idxs in enumerate(area_indices):
            x_area = flat[:, idxs]
            x_area = x_area - x_area.mean(dim=0, keepdim=True)
            cov_a = (x_area.T @ x_area) / (x_area.shape[0] - 1)
            eigvals_a = torch.linalg.eigvalsh(cov_a).flip(0).clamp_min(0.0)
            explained_a = eigvals_a / eigvals_a.sum()
            cum_var_a = explained_a.cumsum(0)
            n_k = int((cum_var_a >= _n_target_var_thresh).float().argmax().item()) + 1
            n_target_per_block.append(n_k)
            cum_at_pick.append(float(cum_var_a[n_k - 1].item()))
            log.info(
                f"  area {i}: input_dims={len(idxs)}, n_target_dims={n_k}, "
                f"cum_var_at_pick={cum_at_pick[-1]:.4f}"
            )
        total = int(sum(n_target_per_block))
        log.info(
            f"DirectSum PCA-auto: n_target_dims_per_block={n_target_per_block} "
            f"(total={total}, threshold={_n_target_var_thresh})"
        )
        cfg.model.encoder.n_target_dims_per_block = list(n_target_per_block)
        cfg.model.n_target_dims = total
        cfg.model.params.input_dim = total
        cfg.model.params.output_dim = total ** 2
        OmegaConf.update(
            cfg, "model.n_target_dims_per_block_pca_auto",
            list(n_target_per_block), force_add=True,
        )
        OmegaConf.update(
            cfg, "model.n_target_dims_per_block_pca_cum_var",
            list(cum_at_pick), force_add=True,
        )
    elif _n_target_var_thresh is not None or _init_pca_basis:
        train_seq = trajs["train_trajs"].sequence  # (N_traj, T, D_embed)
        flat = train_seq.reshape(-1, train_seq.shape[-1]).to(torch.float64)
        flat -= flat.mean(dim=0, keepdim=True)
        cov = (flat.T @ flat) / (flat.shape[0] - 1)
        # eigh returns ascending eigvals + matching eigvecs as columns.
        eigvals_asc, eigvecs_asc = torch.linalg.eigh(cov)
        eigvals = eigvals_asc.flip(0).clamp_min(0.0)
        # PCA basis V: rows = PCs in variance-descending order.
        # eigvecs_asc[:, i] is the i-th eigvec; transpose puts PCs as rows,
        # then flip so descending.
        V = eigvecs_asc.T.flip(0).contiguous()  # (D, D)
        explained = eigvals / eigvals.sum()
        cum_var = explained.cumsum(0)
        _cum = [f"{v:.4f}" for v in cum_var[: min(10, len(cum_var))].tolist()]
        _exp = [f"{v:.4f}" for v in explained[: min(10, len(explained))].tolist()]
        log.info(
            f"PCA on training delay embeddings: D_embed={flat.shape[-1]}, "
            f"N_samples={flat.shape[0]}"
        )
        log.info(f"  explained variance (first 10): {_exp}")
        log.info(f"  cumulative variance (first 10): {_cum}")

        if _n_target_var_thresh is not None:
            n_target = int((cum_var >= _n_target_var_thresh).float().argmax().item()) + 1
            log.info(
                f"PCA-auto n_target_dims: threshold={_n_target_var_thresh}, "
                f"chose n_target_dims={n_target}"
            )
            cfg.model.n_target_dims = n_target
            cfg.model.params.input_dim = n_target
            cfg.model.params.output_dim = n_target ** 2
            OmegaConf.update(cfg, "model.n_target_dims_pca_auto", n_target, force_add=True)
            OmegaConf.update(
                cfg, "model.n_target_dims_pca_cum_var",
                float(cum_var[n_target - 1].item()),
                force_add=True,
            )

        if _init_pca_basis:
            pca_basis_tensor = V.float()
            log.info(
                f"PCA-basis encoder init enabled: V shape={tuple(pca_basis_tensor.shape)}, "
                f"orthogonality residual ||V V^T - I||_F = "
                f"{torch.linalg.norm(V @ V.T - torch.eye(V.shape[0], dtype=torch.float64)).item():.3g}"
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
        lit_model = make_model(cfg, dt, eq=None, project=project, mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor, generalized_variance=generalized_variance, verbose=True, pca_basis=pca_basis_tensor)
    else:
        lit_model = make_model(cfg, dt, eq=eq, project=project, x0=x0, mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor, generalized_variance=generalized_variance, verbose=True, pca_basis=pca_basis_tensor)

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
