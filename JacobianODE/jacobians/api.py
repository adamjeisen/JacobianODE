"""High-level public API for training a JacobianODE model from in-memory
arrays.

Provides :func:`train_from_arrays` — a flat-kwargs entry point for
downstream consumers (notebooks, MindControl, future labs/datasets)
that bypasses the Hydra-driven :func:`JacobianODE.jacobians.run_jacobians.train`
orchestration. It composes a config from JacobianODE-side defaults +
user overrides, then calls the same primitive functions
(:func:`make_trajectories`, :func:`postprocess_data` /
:func:`postprocess_per_condition`, :func:`create_dataloaders`,
:func:`infer_n_target_dims`, :func:`make_model`, :func:`train_model`)
as the canonical pipeline — DRY by design.

Quickstart
----------
.. code-block:: python

    from JacobianODE import train_from_arrays
    lit_model, result = train_from_arrays(
        values, dt,
        condition=cond, source_id=src,
        n_delays=10, seq_length=45, seq_spacing=15,
        encoder='latent_direct_sum_coupling',
        encoder_kwargs={'hidden_dim': 1024, 'use_cayley_perms': True},
        n_target_var_threshold=0.99,
        prediction_steps=30,
        normalize_per_condition=True,
        n_epochs=5, batch_size=16,
        save_dir=Path('./ckpts'),
        wandb_disabled=True,
    )

See :func:`train_from_arrays`'s docstring for the full kwarg list.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class TrainingResult(NamedTuple):
    """Bundle of artifacts returned by :func:`train_from_arrays`.

    Attributes
    ----------
    trainer : pytorch_lightning.Trainer
        The trainer instance after .fit() completes. Holds
        callback_metrics (loss histories, best metrics) and references
        to checkpoint callbacks.
    cfg : DictConfig
        The fully-resolved config used for this run. Useful for
        reproducibility (every default + override that landed in the
        run is captured here).
    trajs : dict
        The trajectories dict from :func:`create_dataloaders`. Includes
        ``train_trajs`` / ``val_trajs`` / ``test_trajs`` (TimeSeriesDatasets)
        and per-trajectory ``*_condition`` arrays when conditioning was
        used.
    train_dataloader : DataLoader
    val_dataloader : DataLoader
    test_dataloader : DataLoader
    best_ckpt_path : str or None
        Path to the best checkpoint by ``monitor`` metric, or None if no
        checkpoints were saved.
    mu : float
    sigma : float
        Postprocessing scalars (mean across sources for the per-condition
        path). Useful for denormalizing predictions later.
    autodim_result : InferNTargetDimsResult or None
        The PCA / FNN autodim result if autodim ran, else None.
    """
    trainer: Any
    cfg: DictConfig
    trajs: dict
    train_dataloader: Any
    val_dataloader: Any
    test_dataloader: Any
    best_ckpt_path: Optional[str]
    mu: float
    sigma: float
    autodim_result: Optional[Any]


def train_from_arrays(
    values: Union[np.ndarray, torch.Tensor],
    dt: float,
    *,
    # ---------------------- per-trajectory metadata ----------------------
    condition: Optional[Union[np.ndarray, torch.Tensor]] = None,
    source_id: Optional[Union[np.ndarray, torch.Tensor]] = None,
    area_indices: Optional[list[list[int]]] = None,

    # ----------------------- data preprocessing -------------------------
    n_delays: int = 1,
    delay_spacing: int = 1,
    observed_indices: Union[str, list[int]] = "all",
    seq_length: Optional[int] = None,
    seq_spacing: int = 1,
    train_percent: float = 0.7,
    test_percent: float = 0.15,
    split_by: str = "trajectory",

    # ----------------------- normalization & noise ----------------------
    obs_noise: float = 0.0,
    normalize: bool = True,
    normalize_per_condition: bool = False,
    filter_data: bool = False,
    low_pass: Optional[float] = None,
    high_pass: Optional[float] = None,

    # -------------------------------- model ----------------------------
    encoder: str = "latent_direct_sum_coupling",
    encoder_kwargs: Optional[dict] = None,
    dynamics_kwargs: Optional[dict] = None,
    n_target_dims: Optional[int] = None,
    n_target_var_threshold: Optional[float] = None,
    n_target_dim_method: str = "pca",
    prediction_steps: int = 30,
    condition_dim: Optional[int] = None,

    # ------------------------------ training ---------------------------
    lightning_kwargs: Optional[dict] = None,
    n_epochs: int = 100,
    batch_size: int = 16,
    limit_train_batches: Optional[int] = None,
    limit_val_batches: Optional[int] = None,
    early_stopping_kwargs: Optional[dict] = None,
    accelerator: str = "auto",
    devices: Union[int, str] = "auto",
    seed: int = 42,

    # ------------------------------- output ----------------------------
    save_dir: Optional[Union[Path, str]] = None,
    wandb_disabled: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_run_name: Optional[str] = None,

    # ------------------------- pretrained init -------------------------
    init_from_pretrained: Optional[Union[str, Path]] = None,
    init_from_pretrained_strict: bool = True,

    # ------------------------- escape hatch + misc --------------------
    cfg_overrides: Optional[dict] = None,
    log_level: str = "INFO",
    verbose: bool = False,
) -> tuple[Any, TrainingResult]:
    """Train a JacobianODE model from in-memory arrays.

    High-level public API for downstream consumers. Composes a config
    from JacobianODE-side defaults + your overrides, then calls the
    same canonical pipeline as ``train(cfg)`` (postprocess →
    create_dataloaders → infer_n_target_dims → make_model →
    train_model). No Hydra invocation, no SLURM-side-effects, no
    automatic wandb (default-off). Single fresh training run; for
    sweeps or multi-stage protocols, use the Hydra-driven
    :func:`train` entrypoint instead.

    Parameters
    ----------
    values : np.ndarray or torch.Tensor of shape ``(N_traj, T, D)``
        Pre-built raw, **uniform-length** trajectories. If your data is
        ragged (variable per-trajectory length), pre-slide via
        :func:`JacobianODE.jacobians.data.ragged.sliding_windows` before
        calling. ``T`` is the time axis BEFORE delay embedding;
        delay-embed will reduce it by ``n_delays - 1``.
    dt : float
        Time step between observations (seconds typically).
    condition : np.ndarray or torch.Tensor of shape ``(N_traj, condition_dim)``, optional
        Per-trajectory condition vector. When provided, the encoder
        and dynamics MLP receive it as a per-sample input (for
        multi-condition training). When None, no conditioning.
    source_id : np.ndarray of shape ``(N_traj,)``, optional
        Per-trajectory integer source ID. Used for two purposes:
        (a) split-balancing (each source is split independently into
        train/val/test) when ``split_by='trajectory'``, and
        (b) per-condition normalization when
        ``normalize_per_condition=True``. When None, no source-aware
        behavior.
    area_indices : list[list[int]], optional
        For DirectSum encoders only: per-area indices into the **raw
        obs space** (positions in [0, D)). Will be auto-extended for
        delay embedding via
        :func:`JacobianODE.jacobians.core.config.extend_area_indices_for_delay_embedding`.
        Required when ``encoder`` is a DirectSum variant; ignored
        otherwise.

    n_delays : int, default 1
        Number of delay copies for delay embedding. ``n_delays=1`` =
        no delay embedding.
    delay_spacing : int, default 1
        Time-step gap between consecutive delay copies.
    observed_indices : str or list[int], default ``'all'``
        Partial-obs filter applied BEFORE delay embedding: ``'all'``
        keeps every dim, ``'random'`` requires
        ``cfg.data.train_test_params.delay_embedding_params.n_observed``,
        a list selects specific raw indices.
    seq_length : int, optional
        Length of training sub-sequences (in delay-embedded time).
        Required when the post-delay-embed time axis is longer than
        what fits in memory per batch. Defaults to the full length
        (post-delay-embed) if None.
    seq_spacing : int, default 1
        Stride between sub-sequence starts (sliding window). Defaults
        to 1 (dense sliding); set to ``seq_length`` for non-overlapping
        tiling.
    train_percent : float, default 0.7
    test_percent : float, default 0.15
        Train/test fractions. Validation = ``1 - train - test``.
    split_by : str, default ``'trajectory'``
        ``'trajectory'`` or ``'time'``. With ``source_id`` set,
        ``'trajectory'`` ALSO balances sources across splits.

    obs_noise : float, default 0.0
        Additive observation noise (percentage of data magnitude;
        scaled by mean L2 norm of the data). Set > 0 for noise
        injection during training.
    normalize : bool, default True
        Z-score normalize the data before delay embedding.
    normalize_per_condition : bool, default False
        When True AND ``source_id`` is provided, normalize each
        source's trajectories independently (per-source mu / sigma).
        Useful when concatenating trajectories from different
        source distributions with different scales.
    filter_data : bool, default False
        If True, apply a Butterworth bandpass to the raw values via
        :func:`JacobianODE.jacobians.data.filtering.filter_data`.
    low_pass, high_pass : float, optional
        Filter cutoffs (Hz). Required if ``filter_data=True``.

    encoder : str, default ``'latent_direct_sum_coupling'``
        Name of the model YAML in ``conf/model/``. Common choices:
        ``'latent_additive_coupling'`` (single-area encoder),
        ``'latent_direct_sum_coupling'`` (multi-area encoder, requires
        ``area_indices``), ``'latent_spline_coupling'`` (spline
        coupling).
    encoder_kwargs : dict, optional
        Overrides on the ``cfg.model.encoder.*`` block. Examples:
        ``{'hidden_dim': 1024, 'use_cayley_perms': True,
        'n_coupling_layers': 8, 'final_perm_identity': True}``.
    dynamics_kwargs : dict, optional
        Overrides on the dynamics MLP (``cfg.model.params.*``).
        Examples: ``{'hidden_dim': [256, 1024, 2048, 2048],
        'num_layers': 4, 'activation': 'silu'}``.
    n_target_dims : int, optional
        Fixed dynamic-subspace dimension. When None, inferred at
        runtime via PCA / FNN (controlled by
        ``n_target_var_threshold`` / ``n_target_dim_method``).
    n_target_var_threshold : float, optional
        PCA cumulative-variance threshold (e.g. 0.99). Only used when
        ``n_target_dim_method='pca'`` AND ``n_target_dims is None``.
    n_target_dim_method : str, default ``'pca'``
        ``'pca'`` or ``'fnn'``. Method for autodim selection.
    prediction_steps : int, default 30
        Number of timesteps the JacobianODE rolls forward in the
        trajectory loss.
    condition_dim : int, optional
        Conditioning vector dim. Auto-inferred from ``condition.shape[-1]``
        if not provided.

    lightning_kwargs : dict, optional
        Overrides on ``cfg.training.lightning.*``. Examples:
        ``{'loss_func': 'mse', 'loop_closure_weight': 1e-4,
        'reconstruction_mode': 'uniform'}``.
    n_epochs : int, default 100
        Maximum number of training epochs. Early stopping may end
        training sooner.
    batch_size : int, default 16
    limit_train_batches : int, optional
        Cap on the number of training batches per epoch (passed to
        Lightning Trainer). When None, falls through to the cfg
        default (200 in JacobianODE's training.yaml).
    limit_val_batches : int, optional
        Cap on the number of validation batches per epoch. When None,
        falls through to the cfg default (50 in JacobianODE's
        training.yaml).
    early_stopping_kwargs : dict, optional
        Overrides on ``cfg.training.early_stopping.*``. Examples:
        ``{'early_stopping_patience': 5, 'percent_thresh': 0.01,
        'min_epochs': 10}``.
    accelerator : str, default ``'auto'``
        Lightning Trainer accelerator. ``'auto'`` picks GPU if
        available, else CPU.
    devices : int or str, default ``'auto'``
        Lightning Trainer devices. ``'auto'`` picks all available.
    seed : int, default 42
        Global RNG seed for reproducibility.

    save_dir : Path or str, optional
        Directory for Lightning checkpoints. If None, Lightning uses
        its default (``./lightning_logs``).
    wandb_disabled : bool, default True
        When True, no W&B logger is created (Lightning runs without
        an experiment logger). Default is OFF for notebook use.
    wandb_entity, wandb_project, wandb_group, wandb_run_name : str, optional
        W&B configuration. Used only when ``wandb_disabled=False``.

    init_from_pretrained : str or Path, optional
        Warm-start the model weights from a previous training run.
        Accepts: a path to a Lightning ``.ckpt`` file, OR a wandb
        run id (resolved to its checkpoint via
        :func:`JacobianODE.jacobians.checkpoints.loader`). When None,
        fresh random initialization.
    init_from_pretrained_strict : bool, default True
        Strict state-dict loading. When False, missing / unexpected
        keys are tolerated (use only when intentionally fine-tuning a
        differently-shaped encoder).

    cfg_overrides : dict, optional
        Catch-all flat-dotted-key dict applied LAST, after all
        kwargs. Use for any rare cfg knob not exposed as a kwarg.
        Example: ``{'training.lightning.k_scale': 2.0,
        'data.postprocessing.high_pass': 30}``.
    log_level : str, default ``'INFO'``
    verbose : bool, default False

    Returns
    -------
    (lit_model, result) : tuple
        ``lit_model`` is the trained ``LitLatentJacobianODE``.
        ``result`` is a :class:`TrainingResult` NamedTuple with the
        trainer, dataloaders, ckpt path, postprocessing scalars, and
        the resolved cfg.

    Raises
    ------
    ValueError
        For shape / dim mismatches between ``values``, ``condition``,
        ``source_id``, and ``area_indices``.
    FileNotFoundError
        If ``init_from_pretrained`` points at a non-existent ckpt.

    See Also
    --------
    JacobianODE.jacobians.run_jacobians.train : the Hydra-driven
        canonical training entrypoint (used for sweeps + SLURM-coordinated
        multi-stage runs).
    JacobianODE.jacobians.data.ragged.sliding_windows : helper to convert
        ragged variable-length trajectories into uniform-length sub-windows
        before passing here.
    """
    # Install the same crash diagnostics that train(cfg) provides.
    faulthandler.enable()

    logging.basicConfig(level=log_level)

    try:
        return _train_from_arrays_inner(
            values=values, dt=dt,
            condition=condition, source_id=source_id, area_indices=area_indices,
            n_delays=n_delays, delay_spacing=delay_spacing,
            observed_indices=observed_indices,
            seq_length=seq_length, seq_spacing=seq_spacing,
            train_percent=train_percent, test_percent=test_percent,
            split_by=split_by,
            obs_noise=obs_noise, normalize=normalize,
            normalize_per_condition=normalize_per_condition,
            filter_data=filter_data, low_pass=low_pass, high_pass=high_pass,
            encoder=encoder, encoder_kwargs=encoder_kwargs or {},
            dynamics_kwargs=dynamics_kwargs or {},
            n_target_dims=n_target_dims,
            n_target_var_threshold=n_target_var_threshold,
            n_target_dim_method=n_target_dim_method,
            prediction_steps=prediction_steps,
            condition_dim=condition_dim,
            lightning_kwargs=lightning_kwargs or {},
            n_epochs=n_epochs, batch_size=batch_size,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            early_stopping_kwargs=early_stopping_kwargs or {},
            accelerator=accelerator, devices=devices, seed=seed,
            save_dir=save_dir,
            wandb_disabled=wandb_disabled,
            wandb_entity=wandb_entity, wandb_project=wandb_project,
            wandb_group=wandb_group, wandb_run_name=wandb_run_name,
            init_from_pretrained=init_from_pretrained,
            init_from_pretrained_strict=init_from_pretrained_strict,
            cfg_overrides=cfg_overrides or {},
            verbose=verbose,
        )
    except Exception:
        # Mirror train(cfg)'s error_traceback diagnostics: write a
        # local file so the traceback survives even if the caller
        # swallowed the exception.
        try:
            tb_path = Path("error_traceback.txt")
            tb_path.write_text(traceback.format_exc())
            logger.error(f"train_from_arrays raised; traceback written to {tb_path.resolve()}")
        except Exception:
            pass
        raise


def _train_from_arrays_inner(
    values, dt, *,
    condition, source_id, area_indices,
    n_delays, delay_spacing, observed_indices,
    seq_length, seq_spacing, train_percent, test_percent, split_by,
    obs_noise, normalize, normalize_per_condition,
    filter_data, low_pass, high_pass,
    encoder, encoder_kwargs, dynamics_kwargs,
    n_target_dims, n_target_var_threshold, n_target_dim_method,
    prediction_steps, condition_dim,
    lightning_kwargs, n_epochs, batch_size,
    limit_train_batches, limit_val_batches, early_stopping_kwargs,
    accelerator, devices, seed,
    save_dir, wandb_disabled,
    wandb_entity, wandb_project, wandb_group, wandb_run_name,
    init_from_pretrained, init_from_pretrained_strict,
    cfg_overrides, verbose,
) -> tuple[Any, TrainingResult]:
    """Inner implementation; outer wrapper handles the
    error_traceback.txt + faulthandler diagnostics."""
    from .core.reproducibility import seed_everything
    from .core.config import extend_area_indices_for_delay_embedding
    from .data import (
        make_trajectories, postprocess_data, postprocess_per_condition,
        create_dataloaders,
    )
    from .training.autodim import infer_n_target_dims
    from .training.model_factory import make_model
    from .training.trainer import train_model
    from .training.logging import log_training_info
    from .metrics import compute_generalized_variance

    log = logging.getLogger("JacobianODE.train_from_arrays")

    # ---- Validate inputs --------------------------------------------------
    values_arr = np.asarray(values) if not isinstance(values, np.ndarray) else values
    if values_arr.ndim != 3:
        raise ValueError(
            f"values must be 3-D (N_traj, T, D), got shape {values_arr.shape}"
        )
    n_traj, t_axis, n_features = values_arr.shape

    if condition is not None:
        cond_arr = np.asarray(condition)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr.reshape(-1, 1)
        if cond_arr.ndim != 2:
            raise ValueError(
                f"condition must be 1-D (N_traj,) or 2-D "
                f"(N_traj, condition_dim); got shape {cond_arr.shape}"
            )
        if cond_arr.shape[0] != n_traj:
            raise ValueError(
                f"condition.shape[0]={cond_arr.shape[0]} != "
                f"values.shape[0]={n_traj}"
            )
        if condition_dim is None:
            condition_dim = int(cond_arr.shape[-1])
        elif condition_dim != cond_arr.shape[-1]:
            raise ValueError(
                f"condition_dim={condition_dim} disagrees with "
                f"condition.shape[-1]={cond_arr.shape[-1]}"
            )
    else:
        cond_arr = None

    if source_id is not None:
        src_arr = np.asarray(source_id)
        if src_arr.ndim != 1:
            raise ValueError(
                f"source_id must be 1-D, got shape {src_arr.shape}"
            )
        if src_arr.shape[0] != n_traj:
            raise ValueError(
                f"source_id.shape[0]={src_arr.shape[0]} != "
                f"values.shape[0]={n_traj}"
            )
    else:
        src_arr = None

    if normalize_per_condition and src_arr is None:
        raise ValueError(
            "normalize_per_condition=True requires source_id to be provided"
        )

    # ---- Build cfg --------------------------------------------------------
    cfg = _compose_cfg(
        encoder=encoder,
        n_features=n_features,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        n_delays=n_delays, delay_spacing=delay_spacing,
        observed_indices=observed_indices,
        seq_length=seq_length, seq_spacing=seq_spacing,
        train_percent=train_percent, test_percent=test_percent,
        split_by=split_by,
        obs_noise=obs_noise, normalize=normalize,
        normalize_per_condition=normalize_per_condition,
        filter_data=filter_data, low_pass=low_pass, high_pass=high_pass,
        encoder_kwargs=encoder_kwargs, dynamics_kwargs=dynamics_kwargs,
        n_target_dims=n_target_dims,
        n_target_var_threshold=n_target_var_threshold,
        n_target_dim_method=n_target_dim_method,
        prediction_steps=prediction_steps,
        condition_dim=condition_dim,
        lightning_kwargs=lightning_kwargs,
        n_epochs=n_epochs, batch_size=batch_size,
        early_stopping_kwargs=early_stopping_kwargs,
        accelerator=accelerator, devices=devices, seed=seed,
        save_dir=save_dir,
        wandb_disabled=wandb_disabled,
        wandb_entity=wandb_entity, wandb_project=wandb_project,
        wandb_group=wandb_group, wandb_run_name=wandb_run_name,
        cfg_overrides=cfg_overrides,
    )

    # ---- Resolve area_indices into delay-embedded space -------------------
    if area_indices is not None:
        de_indices = extend_area_indices_for_delay_embedding(
            area_indices, n_delays=n_delays, n_features=n_features,
        )
        OmegaConf.update(cfg, "model.encoder.area_indices", de_indices, force_add=True)
        log.info(
            f"area_indices: extended {len(area_indices)} areas from raw "
            f"{n_features}-D to delay-embedded {n_features * n_delays}-D"
        )

    # ---- Initialize cfg (resolves auto-dim sentinels, sets lightning._target_)
    # Don't pass data_dim — let initialize_config use the custom branch
    # which correctly multiplies cfg.data.flow.dim by n_delays. (Passing
    # data_dim short-circuits to dim=data_dim with no n_delays multiply.)
    from .core.config import initialize_config
    cfg = initialize_config(cfg)

    # ---- Seed ------------------------------------------------------------
    seed_everything(seed)

    # ---- Inject in-memory data via make_trajectories(data=) ---------------
    eq, sol, dt_returned = make_trajectories(cfg, data=values_arr, dt=dt, verbose=verbose)
    assert dt_returned == dt, f"dt mismatch: passed {dt}, got {dt_returned}"

    # Augment sol with condition + source_id (make_trajectories doesn't do this
    # for the data= injection path).
    if cond_arr is not None:
        sol["condition"] = cond_arr.astype(np.float32)
    if src_arr is not None:
        sol["source_id"] = src_arr.astype(np.int64)
        # Combined-loader pattern: source_eqs is a list of None for each unique source.
        n_sources = int(np.unique(src_arr).shape[0])
        sol["source_eqs"] = [None] * n_sources

    # ---- Postprocess (per-condition or single) ----------------------------
    raw_values_noise = None  # No alternate noise reference in the API path.
    if normalize_per_condition and src_arr is not None:
        log.info(
            f"per-condition normalization: {len(np.unique(src_arr))} sources, "
            f"sizes={[int((src_arr == s).sum()) for s in np.unique(src_arr)]}"
        )
        pc_result = postprocess_per_condition(
            cfg, sol["values"], source_id=src_arr,
            raw_values_to_use_for_noise=raw_values_noise,
        )
        values_norm = pc_result.values
        mu = pc_result.mu
        sigma = pc_result.sigma
        nsf = pc_result.noise_scale_factor
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
        result = postprocess_data(cfg, sol["values"], raw_values_to_use_for_noise=raw_values_noise)
        values_norm = result.values
        mu = result.mu
        sigma = result.sigma
        nsf = result.noise_scale_factor

    cfg.data.postprocessing.noise_scale_factor = nsf
    cfg.data.postprocessing.mu = mu
    cfg.data.postprocessing.sigma = sigma

    # ---- Generalized variance for loss-scaling denominator ----------------
    n_recent = OmegaConf.select(cfg, "model.n_recent_dims", default=None)
    if n_recent is not None and n_recent < values_norm.shape[-1]:
        gv_input = values_norm[..., :n_recent]
    else:
        gv_input = values_norm
    gv = compute_generalized_variance(gv_input)
    OmegaConf.update(cfg, "data.postprocessing.generalized_variance", gv, force_add=True)
    log.info(f"Generalized variance det(Cov)^(1/D) = {gv:.6g}")

    # ---- Create dataloaders (delay-embed + split + batch) -----------------
    train_dl, val_dl, test_dl, trajs = create_dataloaders(
        cfg, values_norm, condition=cond_arr, split_groups=src_arr, verbose=verbose,
    )

    # ---- Autodim PCA / FNN ------------------------------------------------
    autodim_result = None
    if n_target_dims is None:
        autodim_result = infer_n_target_dims(cfg, trajs["train_trajs"].sequence)
        if autodim_result is not None:
            cfg.model.n_target_dims = autodim_result.n_target_dims_total
            cfg.model.params.input_dim = autodim_result.n_target_dims_total
            cfg.model.params.output_dim = autodim_result.n_target_dims_total ** 2
            if autodim_result.is_direct_sum:
                cfg.model.encoder.n_target_dims_per_block = list(
                    autodim_result.n_target_dims_per_block
                )

    # ---- Build run name (used by trainer + checkpoint dir) ----------------
    if wandb_run_name is not None:
        name = wandb_run_name
    else:
        try:
            from .training.logging import make_run_info
            name, derived_project = make_run_info(cfg)
            if not cfg.get("wandb_project"):
                cfg.wandb_project = derived_project
        except Exception as e:
            log.warning(f"make_run_info failed: {e}; using fallback name")
            name = "train_from_arrays"

    # ---- Build model ------------------------------------------------------
    lit_model = make_model(
        cfg, dt, eq=None,
        mu=mu, sigma=sigma,
        noise_scale_factor=nsf,
        generalized_variance=gv,
        verbose=verbose,
    )

    # ---- Optionally load pretrained weights -------------------------------
    if init_from_pretrained is not None:
        _load_pretrained_into_litmodel(
            lit_model, init_from_pretrained,
            strict=init_from_pretrained_strict,
        )

    log_training_info(train_dl, trajs, lit_model, log=log)

    # ---- Train -----------------------------------------------------------
    trainer = train_model(cfg, lit_model, train_dl, val_dl, name=name)

    # ---- Build TrainingResult --------------------------------------------
    best_ckpt = None
    for cb in getattr(trainer, "callbacks", []):
        # ModelCheckpoint exposes best_model_path
        if hasattr(cb, "best_model_path") and cb.best_model_path:
            best_ckpt = cb.best_model_path
            break

    result = TrainingResult(
        trainer=trainer,
        cfg=cfg,
        trajs=trajs,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
        best_ckpt_path=best_ckpt,
        mu=mu,
        sigma=sigma,
        autodim_result=autodim_result,
    )
    return lit_model, result


def _compose_cfg(
    *, encoder, n_features,
    n_delays, delay_spacing, observed_indices,
    seq_length, seq_spacing, train_percent, test_percent, split_by,
    obs_noise, normalize, normalize_per_condition,
    filter_data, low_pass, high_pass,
    encoder_kwargs, dynamics_kwargs,
    n_target_dims, n_target_var_threshold, n_target_dim_method,
    prediction_steps, condition_dim,
    lightning_kwargs, n_epochs, batch_size,
    limit_train_batches, limit_val_batches, early_stopping_kwargs,
    accelerator, devices, seed,
    save_dir, wandb_disabled,
    wandb_entity, wandb_project, wandb_group, wandb_run_name,
    cfg_overrides,
) -> DictConfig:
    """Compose a JacobianODE cfg using `data=custom`, the chosen encoder
    YAML, and the kwargs as overrides. Returns a fully-populated cfg
    that downstream pipeline functions can consume.
    """
    import JacobianODE
    conf_dir = Path(JacobianODE.__file__).parent / "jacobians" / "conf"

    # Use `++` for the "add or override" prefix on every field — most are
    # already in the default cfg but a few (n_target_dim_method, accelerator,
    # devices, normalize_per_condition) are read with OmegaConf.select(default=)
    # and may not exist statically. `++` is permissive to both cases.
    overrides: list[str] = [
        "data=custom",
        f"model={encoder}",
        f"++data.flow.dim={n_features}",
        f"++data.flow.random_state={seed}",
        f"++data.train_test_params.delay_embedding_params.n_delays={n_delays}",
        f"++data.train_test_params.delay_embedding_params.delay_spacing={delay_spacing}",
        f"++data.train_test_params.train_percent={train_percent}",
        f"++data.train_test_params.test_percent={test_percent}",
        f"++data.train_test_params.split_by={split_by}",
        f"++data.train_test_params.seq_spacing={seq_spacing}",
        f"++data.postprocessing.obs_noise={obs_noise}",
        f"++data.postprocessing.normalize={str(normalize).lower()}",
        f"++data.postprocessing.normalize_per_condition={str(normalize_per_condition).lower()}",
        f"++data.postprocessing.filter_data={str(filter_data).lower()}",
        f"++training.batch_size={batch_size}",
        f"++training.trainer_params.max_epochs={n_epochs}",
        # limit_train_batches / limit_val_batches: cap batches per
        # epoch (passed to Lightning Trainer). When None, fall through
        # to the cfg default (200 / 50 from training.yaml).
        *([f"++training.trainer_params.limit_train_batches={limit_train_batches}"]
          if limit_train_batches is not None else []),
        *([f"++training.trainer_params.limit_val_batches={limit_val_batches}"]
          if limit_val_batches is not None else []),
        # NB: accelerator / devices intentionally NOT passed via
        # cfg.training.trainer_params — train_model() unpacks
        # trainer_params then passes devices=... explicitly, so adding
        # them to trainer_params triggers TypeError "got multiple
        # values for keyword argument 'devices'". For now we accept the
        # train_model() auto-detection (devices="auto", strategy
        # picked based on environment). Users wanting explicit CPU /
        # multi-GPU control can override via cfg_overrides today; a
        # future refactor of train_model could expose these cleanly.
        f"++wandb.disabled={str(wandb_disabled).lower()}",
        f"++model.prediction_steps={prediction_steps}",
        f"++model.n_target_dim_method={n_target_dim_method}",
    ]

    if observed_indices != "all":
        if isinstance(observed_indices, str):
            overrides.append(
                f"++data.train_test_params.delay_embedding_params.observed_indices={observed_indices}"
            )
        else:
            overrides.append(
                "++data.train_test_params.delay_embedding_params.observed_indices="
                + "[" + ",".join(str(i) for i in observed_indices) + "]"
            )

    if seq_length is not None:
        overrides.append(f"++data.train_test_params.seq_length={seq_length}")
    if low_pass is not None:
        overrides.append(f"++data.postprocessing.low_pass={low_pass}")
    if high_pass is not None:
        overrides.append(f"++data.postprocessing.high_pass={high_pass}")
    if n_target_dims is not None:
        overrides.append(f"++model.n_target_dims={n_target_dims}")
    if n_target_var_threshold is not None:
        overrides.append(f"++model.n_target_var_threshold={n_target_var_threshold}")
    if condition_dim is not None:
        overrides.append(f"++model.encoder.condition_dim={condition_dim}")
        overrides.append(f"++model.params.condition_dim={condition_dim}")
    if save_dir is not None:
        overrides.append(f"++training.logger.save_dir={save_dir}")
    if wandb_entity is not None:
        overrides.append(f"++wandb_entity={wandb_entity}")
    if wandb_project is not None:
        overrides.append(f"++wandb_project={wandb_project}")
    if wandb_group is not None:
        overrides.append(f"++wandb_group={wandb_group}")

    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    # Apply nested overrides via OmegaConf.update (kwargs flat dicts that
    # set many fields under one prefix). Each kwarg is its own dict whose
    # keys are the leaf names under the appropriate cfg prefix.
    _apply_nested(cfg, "model.encoder", encoder_kwargs)
    _apply_nested(cfg, "model.params", dynamics_kwargs)
    _apply_nested(cfg, "training.lightning", lightning_kwargs)
    _apply_nested(cfg, "training.early_stopping", early_stopping_kwargs)

    # Catch-all flat-dotted-key overrides applied LAST.
    for k, v in (cfg_overrides or {}).items():
        OmegaConf.update(cfg, k, v, force_add=True)

    return cfg


def _apply_nested(cfg: DictConfig, prefix: str, leaf_dict: dict) -> None:
    """Apply each ``leaf_dict[key] = value`` as
    ``OmegaConf.update(cfg, f'{prefix}.{key}', value, force_add=True)``."""
    if not leaf_dict:
        return
    for k, v in leaf_dict.items():
        OmegaConf.update(cfg, f"{prefix}.{k}", v, force_add=True)


def _load_pretrained_into_litmodel(
    lit_model: Any,
    init_from_pretrained: Union[str, Path],
    strict: bool = True,
) -> None:
    """Load a pretrained Lightning checkpoint into a freshly-built lit_model.

    Accepts either:
    - A filesystem path to a ``.ckpt`` file
    - A wandb run id (resolves to the run's checkpoint via the existing
      :mod:`JacobianODE.jacobians.checkpoints.loader` machinery)

    For wandb run ids: requires the wandb run's checkpoints to be reachable
    on disk under the standard JacobianODE ckpt dir (see
    :func:`JacobianODE.jacobians.checkpoints.loader.load_run`). Wandb-stored
    ckpts (``log_model=True``) are not supported here.
    """
    candidate = Path(str(init_from_pretrained))
    if candidate.is_file() and candidate.suffix in (".ckpt", ".pt", ".pth"):
        ckpt_path = candidate
        logger.info(f"[pretrained] Loading from local file: {ckpt_path}")
    else:
        # Treat as a wandb run id and resolve via the loader.
        try:
            from .checkpoints.loader import load_run
        except Exception as e:
            raise FileNotFoundError(
                f"init_from_pretrained={init_from_pretrained!r} is not a file path, "
                f"and the checkpoints loader is not importable: {e}"
            )
        # load_run typically takes (entity, project, run_id) or just run_id.
        # We only have a run id here, so the loader's signature must accept it
        # directly. If your project / entity differs from wandb defaults, this
        # may need extending — flagged as a TODO in the API plan.
        raise NotImplementedError(
            "init_from_pretrained as a wandb run id is not yet wired up. "
            "For now, pass an explicit path to a .ckpt file. "
            "(TODO: thread (entity, project) through to checkpoints.loader.load_run.)"
        )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    missing, unexpected = lit_model.load_state_dict(sd, strict=strict)
    if missing:
        logger.warning(f"[pretrained] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"[pretrained] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    logger.info(f"[pretrained] state_dict loaded into lit_model (strict={strict})")
