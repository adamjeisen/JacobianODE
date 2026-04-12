"""Checkpoint loading utilities for JacobianODE."""

from __future__ import annotations

import importlib
import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from ..core.reproducibility import seed_everything
from ..data.dataloaders import create_dataloaders
from ..data.processing import postprocess_data
from ..data.trajectory import make_trajectories
from ..lightning_base import LitBase
from ..training.model_factory import make_model
from .legacy import reverse_wandb_run

# Lazy import to avoid circular import when encoder_only loads from jacobians
def _make_encoder_model(cfg, n_obs, save_dir, verbose):
    from hydra.utils import instantiate
    encoder = instantiate(cfg.model.encoder, n_input=n_obs)
    # n_obs_pred: for next_state_decoder; when using delay embedding, should be raw obs
    # dim (before embedding), not full delay-embedded dim (n_obs)
    n_obs_pred = cfg.model.get("n_obs_pred")
    if n_obs_pred is None:
        delay_params = cfg.data.train_test_params.get("delay_embedding_params") or {}
        n_delays = int(delay_params.get("n_delays", 1))
        if n_delays > 1:
            obs_indices = delay_params.get("observed_indices", [0])
            if obs_indices == "all":
                n_obs_pred = n_obs // n_delays  # raw dim from delay embedding
            else:
                n_obs_pred = len(obs_indices)
        else:
            n_obs_pred = n_obs
    else:
        n_obs_pred = int(n_obs_pred)
    k_steps_ahead = int(cfg.model.get("k_steps_ahead", 1))
    lit_model = instantiate(
        cfg.training.lightning,
        encoder=encoder,
        n_obs=n_obs,
        n_obs_pred=n_obs_pred,
        k_steps_ahead=k_steps_ahead,
        context_margin=int(cfg.model.get("context_margin", 0)),
        next_state_burn_in=int(cfg.model.get("next_state_burn_in", 0)),
        use_same_state_decoder=bool(cfg.model.get("use_same_state_decoder", True)),
        use_next_state_decoder=bool(cfg.model.get("use_next_state_decoder", False)),
        decoder_hidden_dim=int(cfg.model.get("decoder_hidden_dim", 128)),
        decoder_n_layers=int(cfg.model.get("decoder_n_layers", 2)),
    )
    if verbose:
        total_params = sum(p.numel() for p in lit_model.parameters())
        logger.info(f"Created encoder-only model with {total_params:,} parameters")
    return lit_model

logger = logging.getLogger(__name__)

# Cutoff date for legacy run handling (January 31st 2025 at 2pm EST)
LEGACY_CUTOFF_DATE = datetime(2025, 1, 31, 14, 0, tzinfo=ZoneInfo("America/New_York"))

# Cache for pretrained encoder adapter when loading multiple pretrained-encoder runs
# (e.g. select_from_wandb_runs loads the same encoder 6 times; cache avoids redundant W&B/disk I/O)
_PRETRAINED_ADAPTER_CACHE: Dict[Tuple[str, ...], Any] = {}


def load_run(
    project: str,
    run_id: Optional[str] = None,
    run: Optional[Any] = None,
    save_dir: Optional[str] = None,
    no_noise: bool = False,
    generate_data: bool = True,
    dt: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Any, Any, Any, Optional[float], Any, Any, Any, Any, Any, Any]:
    """Load a previous training run and its associated data.

    Handles loading of both recent and legacy runs, including model checkpoints,
    configuration, and trajectory data.

    Args:
        project: W&B project name.
        run_id: ID of the run to load. Defaults to None.
        run: W&B run object. Defaults to None.
        save_dir: Directory containing saved data. Defaults to None.
        no_noise: Whether to disable noise in data generation. Defaults to False.
        generate_data: Whether to generate new trajectory data. Defaults to True.
        dt: Time step size. Defaults to None.
        verbose: Whether to print progress information. Defaults to False.

    Returns:
        Tuple of (run, cfg, eq, dt, values, train_dataloader, val_dataloader,
        test_dataloader, trajs, lit_model) containing all components of the loaded run.

    Raises:
        ValueError: If both run_id and run are None.

    Example:
        >>> run, cfg, eq, dt, values, train_dl, val_dl, test_dl, trajs, lit_model = \\
        ...     load_run("my-project", run_id="abc123")
        >>> lit_model.eval()
    """
    # Get run object
    if run is None:
        api = wandb.Api(timeout=90)
        run = api.run(f"{project}/{run_id}")
    elif run_id is None:
        raise ValueError("run_id and run cannot both be None")

    # Check run date to determine handling method
    utc_dt = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    est_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))

    if verbose:
        logger.info(f"Run created at {est_dt} EST")
        logger.info(f"Is after Jan 31 2025 2pm EST? {est_dt > LEGACY_CUTOFF_DATE}")

    if est_dt > LEGACY_CUTOFF_DATE:
        return _load_recent_run(
            run, project, save_dir, no_noise, generate_data, dt, verbose
        )
    else:
        return _load_legacy_run(
            run, project, save_dir, no_noise, generate_data, dt, verbose
        )


def _load_recent_run(
    run: Any,
    project: str,
    save_dir: Optional[str],
    no_noise: bool,
    generate_data: bool,
    dt: Optional[float],
    verbose: bool,
) -> Tuple[Any, Any, Any, Optional[float], Any, Any, Any, Any, Any, Any]:
    """Load a run created after the legacy cutoff date."""
    if verbose:
        logger.info("Date is after Jan 31 2025 2pm EST")

    cfg = OmegaConf.create(run.config)

    # Handle missing config keys
    if "use_deriv_net" not in cfg.training:
        cfg.training.use_deriv_net = False

    if save_dir is None:
        save_dir = cfg.training.logger.save_dir

    # Set seeds for reproducibility
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number)

    if no_noise:
        cfg.data.postprocessing.obs_noise = 0

    if generate_data:
        if verbose:
            logger.info("Making trajectories")
        eq, sol, dt = make_trajectories(cfg, save_dir=save_dir, verbose=verbose)
        if verbose:
            logger.info("Done making trajectories")
            logger.info(f"obs_noise: {cfg.data.postprocessing.obs_noise}")
            logger.info(f"seq_length: {cfg.data.train_test_params.seq_length}")
    else:
        eq = None
        sol = None
        dt = None if dt is None else float(dt)

    # Clean up lightning config: remove keys not accepted by the target class.
    # Use the union of LitBase and the actual target class's __init__ variables
    # so that subclass-specific parameters (e.g. reconstruction_loss_weight in
    # LitLatentJacobianODE) are not incorrectly stripped.
    valid_lightning_vars = set(LitBase.__init__.__code__.co_varnames)
    target_str = str(cfg.training.lightning.get("_target_", ""))
    if target_str:
        try:
            module_path, class_name = target_str.rsplit(".", 1)
            target_module = importlib.import_module(module_path)
            target_cls = getattr(target_module, class_name)
            if target_cls is not LitBase:
                valid_lightning_vars |= set(target_cls.__init__.__code__.co_varnames)
        except (ImportError, AttributeError, ValueError):
            pass

    del_keys = []
    for key in cfg.training.lightning.keys():
        if key not in valid_lightning_vars and key != "_target_":
            del_keys.append(key)
    for key in del_keys:
        del cfg.training.lightning[key]

    # Detect whether this is a new-style run (noise stored as percentage)
    # or old-style (noise already scaled in config by postprocess_data).
    is_new_style = "noise_scale_factor" in cfg.data.postprocessing

    if generate_data:
        if is_new_style:
            # New style: obs_noise is a percentage, recompute scale factor.
            result = postprocess_data(cfg, sol["values"], scale_noise=True)
        else:
            # Old style: obs_noise was already scaled in-place before logging.
            # Pass scale_noise=False to avoid double-scaling.
            result = postprocess_data(cfg, sol["values"], scale_noise=False)
        values = result.values
        mu = result.mu
        sigma = result.sigma
        noise_scale_factor = result.noise_scale_factor
        # Recompute the generalized variance if the run used gennMSE loss.
        # (run_jacobians.py slices to n_recent_dims before computing it.)
        generalized_variance = None
        loss_func = cfg.training.lightning.get("loss_func", "mse")
        if loss_func == "generalized_normalized_mse":
            from ..metrics import compute_generalized_variance
            _n_recent = OmegaConf.select(cfg, "model.n_recent_dims", default=None)
            if _n_recent is not None and _n_recent < values.shape[-1]:
                _values_for_gv = values[..., :_n_recent]
            else:
                _values_for_gv = values
            generalized_variance = compute_generalized_variance(_values_for_gv)
        # Create train and test sets
        return_full = "LitEncoderDecoder" in str(cfg.training.lightning.get("_target_", ""))
        train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
            cfg, values, return_full_obs=return_full
        )
    else:
        values = None
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None
        trajs = None
        # Use stored values from config if available, else defaults
        if is_new_style:
            mu = cfg.data.postprocessing.get("mu", 0.0)
            sigma = cfg.data.postprocessing.get("sigma", 1.0)
            noise_scale_factor = cfg.data.postprocessing.noise_scale_factor
        else:
            mu = 0.0
            sigma = 1.0
            noise_scale_factor = 1.0
        # Pull generalized_variance from stored postprocessing if present
        # (written by run_jacobians.py at training time).
        generalized_variance = cfg.data.postprocessing.get("generalized_variance", None)

    # Make model
    target_str = str(cfg.training.lightning.get("_target_", ""))
    is_encoder_only = "LitEncoderDecoder" in target_str
    pretrained_enc = cfg.get("pretrained_encoder") or {}
    is_eigentime_delay = (
        not is_encoder_only
        and "LitLatentJacobianODE" in target_str
        and pretrained_enc.get("encoder_type") == "eigentime_delay"
    )
    is_pretrained_jac_run = (
        not is_encoder_only
        and not is_eigentime_delay
        and "LitLatentJacobianODE" in target_str
        and pretrained_enc.get("project")
        and pretrained_enc.get("run_id")
    )

    if is_encoder_only:
        if trajs is not None:
            n_obs = trajs["train_trajs"].sequence.shape[-1]
        else:
            # Infer from config when no data generated (e.g. generate_data=False)
            delay_params = cfg.data.train_test_params.delay_embedding_params
            n_delays = int(delay_params.get("n_delays", 1))
            obs_indices = delay_params.get("observed_indices", [0])
            if obs_indices == "all":
                # "all" is a string; len("all") = 3 would be wrong. Use model.n_obs_pred
                # or data.flow.dim instead.
                n_obs = cfg.model.get("n_obs_pred")
                if n_obs is None and cfg.data.get("flow") and hasattr(cfg.data.flow, "dim"):
                    n_obs = cfg.data.flow.dim
                if n_obs is None:
                    raise ValueError(
                        "Cannot infer n_obs when observed_indices='all' and generate_data=False. "
                        "Set model.n_obs_pred or data.flow.dim in config."
                    )
                n_obs = int(n_obs)
            else:
                n_obs = n_delays * len(obs_indices)
        lit_model = _make_encoder_model(cfg, n_obs, save_dir, verbose)
        load_checkpoint(
            run, cfg, lit_model, save_dir=save_dir,
            loss_key="mean val loss", verbose=verbose,
        )
    elif is_eigentime_delay:
        # Build ETD adapter from training data (same as run_pretrained_jacobians does).
        # Cache the adapter so repeated load_run calls (e.g. select_from_wandb_runs)
        # don't fail when generate_data=False (train_dataloader=None).
        from hydra.utils import instantiate
        from ...encoder_only.pretrained import create_eigentime_delay_adapter

        etd_cfg = pretrained_enc.get("eigentime", {})
        cache_key = (
            "eigentime_delay",
            bool(etd_cfg.get("use_pca", True)),
            etd_cfg.get("n_components"),
            float(etd_cfg.get("variance_threshold", 0.99)),
        )
        if cache_key not in _PRETRAINED_ADAPTER_CACHE:
            if train_dataloader is None:
                raise RuntimeError(
                    "Cannot build EigentimeDelayAdapter without training data. "
                    "Ensure generate_data=True on the first load_run call."
                )
            if verbose:
                print("Building eigentime delay adapter (once per sweep)...", flush=True)
            adapter = create_eigentime_delay_adapter(
                train_dataloader,
                use_pca=etd_cfg.get("use_pca", True),
                n_components=etd_cfg.get("n_components", None),
                variance_threshold=etd_cfg.get("variance_threshold", 0.99),
                verbose=verbose,
            )
            _PRETRAINED_ADAPTER_CACHE[cache_key] = adapter
        else:
            if verbose:
                print("Using cached eigentime delay adapter.", flush=True)
        adapter = _PRETRAINED_ADAPTER_CACHE[cache_key]

        jac_model = instantiate(cfg.model.params)
        extra_kwargs = {}
        if "prediction_steps" in cfg.model:
            extra_kwargs["prediction_steps"] = cfg.model.prediction_steps
        if "encoder_warmup_epochs" in cfg.model:
            extra_kwargs["encoder_warmup_epochs"] = cfg.model.encoder_warmup_epochs
        if "dynamics_warmup_epochs" in cfg.model:
            extra_kwargs["dynamics_warmup_epochs"] = cfg.model.dynamics_warmup_epochs
        if cfg.model.get("jac_window_stride") is not None:
            extra_kwargs["jac_window_stride"] = cfg.model.jac_window_stride

        lit_model = instantiate(
            cfg.training.lightning,
            model=jac_model,
            encoder=adapter,
            dt=dt,
            save_dir=save_dir,
            mu=float(mu),
            sigma=float(sigma),
            noise_scale_factor=float(noise_scale_factor),
            generalized_variance=generalized_variance,
            **extra_kwargs,
        )
        lit_model.eq = eq
    elif is_pretrained_jac_run:
        # Pretrained-encoder runs use PretrainedEncoderAdapter (decoder is nn.Sequential),
        # not the latent_ssm build_ssm (which uses StepDecoder with .net). Build the model
        # the same way run_pretrained_jacobians does so checkpoint keys match.
        # Cache the adapter when loading multiple runs from the same sweep (same encoder).
        from hydra.utils import instantiate
        from ...encoder_only.pretrained import load_pretrained_encoder

        cache_key = (
            str(pretrained_enc.project),
            str(pretrained_enc.run_id),
            str(pretrained_enc.get("save_dir") or ""),
            bool(pretrained_enc.get("freeze", True)),
        )
        if cache_key not in _PRETRAINED_ADAPTER_CACHE:
            if verbose:
                print("Loading pretrained encoder (once per sweep)...", flush=True)
            adapter, _, _ = load_pretrained_encoder(
                project=cache_key[0],
                run_id=cache_key[1],
                save_dir=pretrained_enc.get("save_dir") or None,
                freeze=cache_key[3],
                verbose=verbose,
            )
            _PRETRAINED_ADAPTER_CACHE[cache_key] = adapter
        else:
            if verbose:
                logger.info("Using cached pretrained encoder adapter")
                print("Using cached pretrained encoder.", flush=True)
        adapter = _PRETRAINED_ADAPTER_CACHE[cache_key]

        jac_model = instantiate(cfg.model.params)
        extra_kwargs = {}
        if "prediction_steps" in cfg.model:
            extra_kwargs["prediction_steps"] = cfg.model.prediction_steps
        if "encoder_warmup_epochs" in cfg.model:
            extra_kwargs["encoder_warmup_epochs"] = cfg.model.encoder_warmup_epochs
        if "dynamics_warmup_epochs" in cfg.model:
            extra_kwargs["dynamics_warmup_epochs"] = cfg.model.dynamics_warmup_epochs
        if cfg.model.get("jac_window_stride") is not None:
            extra_kwargs["jac_window_stride"] = cfg.model.jac_window_stride

        lit_model = instantiate(
            cfg.training.lightning,
            model=jac_model,
            encoder=adapter,
            dt=dt,
            save_dir=save_dir,
            mu=float(mu),
            sigma=float(sigma),
            noise_scale_factor=float(noise_scale_factor),
            generalized_variance=generalized_variance,
            **extra_kwargs,
        )
        lit_model.eq = eq
    else:
        if "params" in cfg.model and "NeuralODE" in str(cfg.model.params.get("_target_", "")):
            cfg.model.params.dt = float(dt)

        if cfg.data.train_test_params.delay_embedding_params.n_delays > 1:
            lit_model = make_model(
                cfg, dt, eq=None, save_dir=save_dir,
                mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor,
                generalized_variance=generalized_variance,
                verbose=verbose,
            )
        else:
            lit_model = make_model(
                cfg, dt, eq=eq, project=project, save_dir=save_dir,
                mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor,
                generalized_variance=generalized_variance,
                verbose=verbose,
            )

    return (
        run,
        cfg,
        eq,
        dt,
        values,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        trajs,
        lit_model,
    )


def _load_legacy_run(
    run: Any,
    project: str,
    save_dir: Optional[str],
    no_noise: bool,
    generate_data: bool,
    dt: Optional[float],
    verbose: bool,
) -> Tuple[Any, Any, Any, Optional[float], Any, Any, Any, Any, Any, Any]:
    """Load a run created before the legacy cutoff date."""
    if verbose:
        logger.info("Date is before Jan 31 2025 2pm EST")

    ret_dict = reverse_wandb_run(run, return_data=True, save_dir=save_dir, checkpoint=None)
    cfg = ret_dict["cfg"]
    lit_model = ret_dict["lit_model"]
    eq = ret_dict["eq"]
    values = ret_dict["values"]
    dt = ret_dict["dt"]

    # Set seeds for reproducibility
    seed_everything(cfg.data.flow.random_state + cfg.training.run_number)

    if generate_data:
        eq, sol, dt = make_trajectories(cfg)
        train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
            cfg, values
        )
    else:
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None
        trajs = None

    return (
        run,
        cfg,
        eq,
        dt,
        values,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        trajs,
        lit_model,
    )


def get_all_checkpoints(
    run: Any,
    cfg: Any,
    save_dir: Optional[str] = None,
) -> Tuple[List[str], str]:
    """Get all available checkpoint files for a run.

    Args:
        run: W&B run object.
        cfg: Configuration object.
        save_dir: Directory containing checkpoints. Defaults to None.

    Returns:
        Tuple of (checkpoint_files, checkpoint_dir) where:
            - checkpoint_files: List of checkpoint filenames sorted by epoch
            - checkpoint_dir: Directory containing the checkpoints

    Example:
        >>> checkpoints, ckpt_dir = get_all_checkpoints(run, cfg)
        >>> print(f"Found {len(checkpoints)} checkpoints")
    """
    if save_dir is None:
        save_dir = (
            run.config["save_dir"]
            if "save_dir" in run.config
            else cfg.training.logger.save_dir
        )

    checkpoint_dir = os.path.join(save_dir, run.project, run.id, "checkpoints")
    checkpoint_files = os.listdir(checkpoint_dir)

    # Sort by epoch number
    checkpoint_files = sorted(
        checkpoint_files, key=lambda x: int(x.split("=")[1].split("-")[0])
    )

    return checkpoint_files, checkpoint_dir


def load_checkpoint(
    run: Any,
    cfg: Any,
    lit_model: Any,
    save_dir: Optional[str] = None,
    epoch: Optional[int] = None,
    loss_key: str = "mean_val_loss",
    verbose: bool = False,
) -> None:
    """Load a specific checkpoint for a model.

    Can load either the best checkpoint (based on validation loss) or a specific epoch.

    Args:
        run: W&B run object.
        cfg: Configuration object.
        lit_model: Model to load checkpoint into.
        save_dir: Directory containing checkpoints. Defaults to None.
        epoch: Specific epoch to load. If None, loads best checkpoint. Defaults to None.
        loss_key: Key to use for finding best checkpoint. Defaults to 'mean_val_loss'.
        verbose: Whether to print progress information. Defaults to False.

    Example:
        >>> load_checkpoint(run, cfg, lit_model, epoch=10, verbose=True)
        Loading checkpoint from epoch 10
    """
    if verbose:
        logger.info(f"Loading checkpoint from {save_dir}")

    checkpoint_files, checkpoint_dir = get_all_checkpoints(run, cfg, save_dir)

    if verbose:
        epochs = [int(f.split("=")[1].split("-")[0]) for f in checkpoint_files]
        logger.info(f"Checkpoint epochs: {epochs}")

    epoch_explicitly_requested = epoch is not None

    if epoch is None:
        # Pick the checkpoint with minimum validation loss
        mean_val_losses = [
            {"epoch": h["epoch"], "mean_val_loss": h[loss_key]}
            for h in run.scan_history()
            if loss_key in h and h[loss_key] is not None
        ]

        # Fallback to alternate key name
        if len(mean_val_losses) == 0:
            mean_val_losses = [
                {"epoch": h["epoch"], "mean_val_loss": h["mean val loss"]}
                for h in run.scan_history()
                if "mean val loss" in h and h["mean val loss"] is not None
            ]

        epoch = mean_val_losses[
            np.argmin([mvl["mean_val_loss"] for mvl in mean_val_losses])
        ]["epoch"]

    # W&B may return epoch as float (e.g. 4.0); checkpoint filenames use int (epoch=4-)
    epoch_int = int(epoch)
    epoch_matches = [f for f in checkpoint_files if f.startswith(f"epoch={epoch_int}-")]
    if epoch_matches:
        checkpoint = epoch_matches[0]
    elif epoch_explicitly_requested:
        available_epochs = sorted(
            int(f.split("=")[1].split("-")[0])
            for f in checkpoint_files
            if f.startswith("epoch=")
        )
        raise FileNotFoundError(
            f"No checkpoint found for epoch {epoch_int} in {checkpoint_dir}. "
            f"Available epochs: {available_epochs}."
        )
    else:
        # Fallback: best epoch from history may not have a saved checkpoint (e.g. save_top_k=1
        # replaced it). Use the checkpoint with the highest epoch among those available.
        epoch_files = [f for f in checkpoint_files if f.startswith("epoch=")]
        if not epoch_files:
            raise FileNotFoundError(
                f"No checkpoint files matching epoch=* found in {checkpoint_dir}. "
                f"Available: {checkpoint_files}. "
                f"Best epoch from W&B was {epoch}."
            )
        checkpoint = epoch_files[-1]
        if verbose:
            logger.warning(
                f"No checkpoint for best epoch {epoch}; using {checkpoint} instead."
            )

    if verbose:
        logger.info(f"Loading checkpoint from epoch {epoch}")
        print(f"Loading checkpoint {checkpoint}...", flush=True)

    # Load checkpoint
    checkpoint_data = torch.load(
        os.path.join(checkpoint_dir, checkpoint),
        weights_only=False,
        map_location="cpu",
        mmap=True,
    )
    loaded_state = checkpoint_data["state_dict"]

    # Handle uncertainty parameters if needed
    if (
        "use_uncertainty" in cfg.training.lightning
        and cfg.training.lightning.use_uncertainty
    ):
        if "logvar_trajectory" not in loaded_state:
            loaded_state["logvar_trajectory"] = torch.tensor(0)
        if "logvar_reverse" not in loaded_state:
            loaded_state["logvar_reverse"] = torch.tensor(0)
        if "logvar_loop_closure" not in loaded_state:
            loaded_state["logvar_loop_closure"] = torch.tensor(0)
        if (
            "logvar_lipschitz" not in loaded_state
            and "lipschitz" in cfg.model.params
            and cfg.model.params.lipschitz
        ):
            loaded_state["logvar_lipschitz"] = torch.tensor(0)

    # Backward compatibility: checkpoints saved before the latent_criterion
    # split (gennMSE with gen_variance_mode) won't have latent_criterion.denom.
    # If criterion.denom is present but latent_criterion.denom is not, mirror it.
    if (
        "criterion.denom" in loaded_state
        and "latent_criterion.denom" not in loaded_state
        and hasattr(lit_model, "latent_criterion")
        and hasattr(lit_model.latent_criterion, "denom")
    ):
        loaded_state["latent_criterion.denom"] = loaded_state["criterion.denom"].clone()

    lit_model.load_state_dict(loaded_state)
    lit_model.eval()

    # Cleanup
    del loaded_state
    torch.cuda.empty_cache()
