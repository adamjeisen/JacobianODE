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
    lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
    post_filter_downsample: int = 1,
    pre_pca_per_area: bool = False,
    pre_pca_var_threshold: float = 0.99,
    whiten_after_pre_pca: bool = False,

    # -------------------------------- model ----------------------------
    encoder: str = "latent_direct_sum_coupling",
    encoder_kwargs: Optional[dict] = None,
    dynamics_kwargs: Optional[dict] = None,
    n_target_dims: Optional[int] = None,
    n_target_var_threshold: Optional[float] = None,
    n_target_dim_method: str = "pca",
    prediction_steps: int = 30,
    condition_dim: Optional[int] = None,
    n_dynamics_per_source: int = 1,
    section_condition_values: Optional[list] = None,

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
        Pre-built raw trajectories. ``T`` is the time axis BEFORE delay
        embedding; delay-embed will reduce it by ``n_delays - 1``.

        Two input modes:

        * **Uniform** (``lengths is None``): every trajectory must be
          fully populated with valid samples. The standard pipeline
          (delay-embed → split → stride) is used.
        * **Ragged** (``lengths is not None``): each ``values[i]``
          contains ``lengths[i]`` valid samples followed by NaN padding
          (the convention from
          :func:`JacobianODE.jacobians.data.ragged.pad_trajs_to_max`).
          A different pipeline runs:
          per-trajectory delay-embed → per-condition timepoint-balanced
          trajectory-level split → stride within each split. This
          eliminates the leakage path where pre-strided sub-windows
          from the same parent trajectory end up in different splits.
          Requires ``seq_length``.
    dt : float
        Time step between observations (seconds typically).
    lengths : np.ndarray or torch.Tensor of shape ``(N_traj,)``, optional
        Per-trajectory valid sample count. When provided, triggers the
        ragged pipeline (see ``values``). When None (default), every
        trajectory is treated as fully valid.
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
        common_kwargs = dict(
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
            post_filter_downsample=post_filter_downsample,
            pre_pca_per_area=pre_pca_per_area,
            pre_pca_var_threshold=pre_pca_var_threshold,
            whiten_after_pre_pca=whiten_after_pre_pca,
            n_dynamics_per_source=n_dynamics_per_source,
            section_condition_values=section_condition_values,
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
        if lengths is not None:
            return _train_from_ragged_arrays_inner(lengths=lengths, **common_kwargs)
        return _train_from_arrays_inner(**common_kwargs)
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
    post_filter_downsample,
    pre_pca_per_area, pre_pca_var_threshold,
    whiten_after_pre_pca,
    n_dynamics_per_source, section_condition_values,
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
        post_filter_downsample=post_filter_downsample,
        pre_pca_per_area=pre_pca_per_area,
        pre_pca_var_threshold=pre_pca_var_threshold,
        whiten_after_pre_pca=whiten_after_pre_pca,
        n_dynamics_per_source=n_dynamics_per_source,
        section_condition_values=section_condition_values,
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


def _train_from_ragged_arrays_inner(
    values, dt, *, lengths,
    condition, source_id, area_indices,
    n_delays, delay_spacing, observed_indices,
    seq_length, seq_spacing, train_percent, test_percent, split_by,
    obs_noise, normalize, normalize_per_condition,
    filter_data, low_pass, high_pass,
    post_filter_downsample,
    pre_pca_per_area, pre_pca_var_threshold,
    whiten_after_pre_pca,
    n_dynamics_per_source, section_condition_values,
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
    """Ragged input path: per-trajectory delay-embed → per-condition
    timepoint-balanced trajectory split → stride within each split.

    Bypasses :func:`create_dataloaders` (which expects uniform input)
    and builds the per-split datasets / dataloaders directly. The split
    is at the parent-trajectory level, so sub-windows from the same
    parent never appear in different splits. Per-condition split also
    keeps the source mix balanced across train/val/test.
    """
    from torch.utils.data import DataLoader, Subset

    from .core.config import (
        extend_area_indices_for_delay_embedding,
        initialize_config,
    )
    from .core.reproducibility import seed_everything
    from .data.ragged import (
        delay_embed_ragged,
        sliding_windows,
        split_balanced_by_timepoints,
    )
    from .data.splitting import (
        TimeSeriesDataset,
        collate_with_optional_condition,
    )
    from .training.autodim import infer_n_target_dims
    from .training.logging import log_training_info
    from .training.model_factory import make_model
    from .training.trainer import train_model
    from .metrics import compute_generalized_variance

    log = logging.getLogger("JacobianODE.train_from_arrays")

    if filter_data and (low_pass is None and high_pass is None):
        raise ValueError(
            "filter_data=True requires at least one of low_pass / high_pass."
        )

    # ---- Validate inputs --------------------------------------------------
    if isinstance(values, np.ndarray):
        padded = torch.from_numpy(values)
    elif isinstance(values, torch.Tensor):
        padded = values
    else:
        padded = torch.as_tensor(values)
    if padded.ndim != 3:
        raise ValueError(
            f"values must be 3-D (N_traj, T, D), got shape {tuple(padded.shape)}"
        )
    if isinstance(lengths, torch.Tensor):
        lengths_t = lengths.to(torch.long).detach().cpu()
    else:
        lengths_t = torch.as_tensor(np.asarray(lengths), dtype=torch.long)
    if lengths_t.shape != (padded.shape[0],):
        raise ValueError(
            f"lengths shape {tuple(lengths_t.shape)} != "
            f"(N_traj={padded.shape[0]},)"
        )

    n_traj, t_max, n_features = padded.shape
    log.info(
        f"ragged input: N_traj={n_traj}, t_max={t_max}, D={n_features}, "
        f"lengths range [{int(lengths_t.min())}, {int(lengths_t.max())}], "
        f"total_valid_timepoints={int(lengths_t.sum())}"
    )

    if seq_length is None:
        raise ValueError(
            "seq_length is required on the ragged path (used to stride within "
            "each split after delay-embedding)."
        )

    if condition is not None:
        cond_arr = np.asarray(condition)
        if cond_arr.ndim == 1:
            cond_arr = cond_arr.reshape(-1, 1)
        if cond_arr.ndim != 2 or cond_arr.shape[0] != n_traj:
            raise ValueError(
                f"condition must be 1-D (N_traj,) or 2-D (N_traj, condition_dim); "
                f"got shape {cond_arr.shape}"
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
        if src_arr.ndim != 1 or src_arr.shape[0] != n_traj:
            raise ValueError(
                f"source_id must be 1-D length N_traj; got shape {src_arr.shape}"
            )
    else:
        src_arr = None

    if normalize_per_condition and src_arr is None:
        raise ValueError(
            "normalize_per_condition=True requires source_id to be provided"
        )

    # Use a single source for the per-condition split when no source_id is set,
    # so the same balanced-split function still applies.
    split_src = src_arr if src_arr is not None else np.zeros(n_traj, dtype=np.int64)

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
        post_filter_downsample=post_filter_downsample,
        pre_pca_per_area=pre_pca_per_area,
        pre_pca_var_threshold=pre_pca_var_threshold,
        whiten_after_pre_pca=whiten_after_pre_pca,
        n_dynamics_per_source=n_dynamics_per_source,
        section_condition_values=section_condition_values,
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

    if area_indices is not None:
        de_indices = extend_area_indices_for_delay_embedding(
            area_indices, n_delays=n_delays, n_features=n_features,
        )
        OmegaConf.update(cfg, "model.encoder.area_indices", de_indices, force_add=True)
        log.info(
            f"area_indices: extended {len(area_indices)} areas from raw "
            f"{n_features}-D to delay-embedded {n_features * n_delays}-D"
        )

    cfg = initialize_config(cfg)
    seed_everything(seed)

    # ---- Optional NaN-aware filtfilt on the per-trajectory valid region --
    # ORDER: filter BEFORE normalization. Reason: a low-pass filter
    # changes the variance distribution (typically removes high-freq
    # content with ~uniform variance across dims), and we want sigma to
    # reflect what the encoder actually sees. Filtering after normalize
    # would make sigma reflect the un-filtered (wider-bandwidth) signal,
    # so the post-filter input would have non-unit variance again.
    #
    # The ragged-padded shape (NaN-padded between trajectories) breaks
    # naive scipy.signal.filtfilt at trajectory boundaries — NaNs would
    # propagate through every output sample. We loop per-trajectory and
    # only filter the valid prefix [:lengths[i]].
    if filter_data:
        from .data.filtering import filter_data as _filt
        log.info(
            f"filter_data=True (low_pass={low_pass}, high_pass={high_pass}) — "
            f"applying per-trajectory zero-phase Butterworth (filtfilt) "
            f"BEFORE normalization..."
        )
        padded_filt = padded.clone().to(torch.float32)
        n_filtered = 0
        for i in range(padded.shape[0]):
            L_i = int(lengths_t[i].item())
            if L_i <= 8:    # below filter init transient — leave as-is
                continue
            chunk = padded[i, :L_i].cpu().numpy()
            chunk_filt = _filt(
                chunk, low_pass=low_pass, high_pass=high_pass,
                dt=dt, bidirectional=True,    # filtfilt
            )
            padded_filt[i, :L_i] = torch.from_numpy(
                np.ascontiguousarray(chunk_filt)
            ).to(padded_filt.dtype)
            n_filtered += 1
        padded = padded_filt
        log.info(f"  filtered {n_filtered}/{padded.shape[0]} trajectories")
        OmegaConf.update(cfg, "data.postprocessing.filter_data", True, force_add=True)
        if low_pass is not None:
            OmegaConf.update(cfg, "data.postprocessing.low_pass", float(low_pass), force_add=True)
        if high_pass is not None:
            OmegaConf.update(cfg, "data.postprocessing.high_pass", float(high_pass), force_add=True)

    # ---- Optional stride decimation AFTER filter, BEFORE normalize -------
    # Pairs with filter_data: filter the high-frequency content out, then
    # decimate by stride. Caller is responsible for picking a low_pass
    # below the new Nyquist (=fs/(2 × post_filter_downsample)) to avoid
    # aliasing — we don't auto-lower the cutoff.
    #
    # Updates the local dt + lengths_t so downstream sees the lower
    # effective rate. cfg.data.postprocessing.post_filter_downsample is
    # saved for inference round-tripping.
    if post_filter_downsample is not None and post_filter_downsample > 1:
        if not filter_data:
            log.warning(
                f"post_filter_downsample={post_filter_downsample} but "
                f"filter_data=False — pure stride decimation will alias."
            )
        log.info(
            f"post-filter decimation by {post_filter_downsample} "
            f"(stride sampling — assumes filtering already band-limited)..."
        )
        # Stride along time axis. ceil(L / N) = (L + N - 1) // N.
        padded = padded[:, ::post_filter_downsample, :].contiguous()
        new_lengths = (lengths_t + post_filter_downsample - 1) // post_filter_downsample
        lengths_t = new_lengths.to(torch.long)
        new_dt = float(dt) * post_filter_downsample
        log.info(
            f"  effective dt = {new_dt:.6g} s (= {1/new_dt:.0f} Hz), "
            f"new t_max = {padded.shape[1]}, "
            f"valid lengths range [{int(lengths_t.min())}, {int(lengths_t.max())}]"
        )
        dt = new_dt
        OmegaConf.update(
            cfg, "data.postprocessing.post_filter_downsample",
            int(post_filter_downsample), force_add=True,
        )

    # ---- NaN-aware per-source noise + normalize on (filtered) padded -----
    padded_norm, mu_per_source, sigma_per_source, nsf_per_source, src_ids_for_norm = (
        _normalize_ragged(
            padded, lengths_t,
            split_src if normalize_per_condition else np.zeros(n_traj, dtype=np.int64),
            normalize=normalize, obs_noise=obs_noise,
        )
    )
    mu = float(np.mean(mu_per_source))
    sigma = float(np.mean(sigma_per_source))
    nsf = float(np.mean(nsf_per_source))

    if normalize_per_condition:
        log.info(
            f"per-condition normalization: {len(src_ids_for_norm)} sources"
        )
        for s, m, sg, n in zip(src_ids_for_norm, mu_per_source, sigma_per_source, nsf_per_source):
            log.info(
                f"  source {s}: mu={m:.6g}, sigma={sg:.6g}, noise_scale_factor={n:.6g}"
            )
        OmegaConf.update(cfg, "data.postprocessing.mu_per_source", list(mu_per_source), force_add=True)
        OmegaConf.update(cfg, "data.postprocessing.sigma_per_source", list(sigma_per_source), force_add=True)
        OmegaConf.update(cfg, "data.postprocessing.noise_scale_factor_per_source", list(nsf_per_source), force_add=True)
    else:
        log.info(f"global normalization: mu={mu:.6g}, sigma={sigma:.6g}, noise_scale_factor={nsf:.6g}")

    cfg.data.postprocessing.noise_scale_factor = nsf
    cfg.data.postprocessing.mu = mu
    cfg.data.postprocessing.sigma = sigma

    # ---- Optional per-area PCA reduction BEFORE delay-embed --------------
    # Shrinks each area's channel count to its 99%-variance components,
    # which usually drops the second-stage autodim by 30-50% on neural
    # data (per the diagnostic in mindcontrol/diagnostics/per_area_pca_pipeline).
    if pre_pca_per_area:
        from .data.per_area_pca import fit_per_area_pca, apply_per_area_pca
        if area_indices is None:
            raise ValueError(
                "pre_pca_per_area=True requires area_indices to be set "
                "(it operates per-area)."
            )
        log.info(
            f"per-area pre-delay-embed PCA at threshold "
            f"{pre_pca_var_threshold:.4f}..."
        )
        pca_fit = fit_per_area_pca(
            padded_norm, lengths_t, area_indices,
            threshold=pre_pca_var_threshold,
        )
        for ai, pca in enumerate(pca_fit.per_area):
            log.info(
                f"  area {ai}: n_in={pca.n_in:3d} -> n_out={pca.k:3d} "
                f"(cum_var={pca.cum_var:.4f})"
            )
        log.info(
            f"  total: {sum(p.n_in for p in pca_fit.per_area)} → {pca_fit.n_total_out} "
            f"({pca_fit.n_total_out / sum(p.n_in for p in pca_fit.per_area):.1%} of raw)"
        )
        padded_norm = apply_per_area_pca(padded_norm, lengths_t, pca_fit)

        # ---- Optional per-dim whitening after pre-PCA ----------------------
        # When True, divides each output dim by its own std so every kept
        # PC has var ≈ 1 going into delay-embed. Trade-off vs the
        # default arithmetic-mean rescale below: whitening flattens the
        # PCA's eigenvalue hierarchy (low-eigenvalue PCs get amplified
        # to unit scale), which equalizes per-dim gradient pressure on
        # the encoder but loses the natural variance ordering. The
        # downstream arith-mean rescale becomes a no-op when whitening
        # is on (mean per-dim var is exactly 1).
        if whiten_after_pre_pca:
            log.info(
                f"whiten_after_pre_pca=True — per-dim z-score after pre-PCA"
            )
            if normalize_per_condition:
                src_ids_w = src_ids_for_norm
                split_src_w = np.asarray(split_src)
            else:
                src_ids_w = [0]
                split_src_w = np.zeros(padded_norm.shape[0], dtype=np.int64)
            sigma_whiten_per_source: list[list[float]] = []
            for s in src_ids_w:
                traj_idxs = np.where(split_src_w == s)[0]
                valid_chunks = []
                for i in traj_idxs:
                    L_i = int(lengths_t[i].item())
                    if L_i > 0:
                        valid_chunks.append(padded_norm[i, :L_i])
                if not valid_chunks:
                    sigma_whiten_per_source.append(
                        [1.0] * padded_norm.shape[-1]
                    )
                    continue
                valid_concat = torch.cat(valid_chunks, dim=0).to(torch.float64)
                per_dim_std = (
                    valid_concat.std(dim=0).clamp_min(1e-8).to(padded_norm.dtype)
                )
                sigma_whiten_per_source.append([float(x) for x in per_dim_std])
                for i in traj_idxs:
                    L_i = int(lengths_t[i].item())
                    if L_i > 0:
                        padded_norm[i, :L_i] = padded_norm[i, :L_i] / per_dim_std
            std_min = min(min(s) for s in sigma_whiten_per_source)
            std_max = max(max(s) for s in sigma_whiten_per_source)
            log.info(
                f"  per-source per-dim std range across all sources/dims: "
                f"[{std_min:.4g}, {std_max:.4g}]"
            )
            OmegaConf.update(
                cfg, "data.postprocessing.whiten_after_pre_pca", True,
                force_add=True,
            )
            OmegaConf.update(
                cfg, "data.postprocessing.sigma_whiten_per_source",
                sigma_whiten_per_source, force_add=True,
            )

        # ---- Restore arithmetic-mean unit variance after pre-PCA --------
        # Per-area PCA concentrates total per-area variance (≈ n_in_area
        # after the upstream global z-score) into the kept k_i dims. Mean
        # per-dim variance after pre-PCA is therefore (n_in / k) × the
        # input mean — for our LFP setup ~13×, which inflates the MSE
        # recon floor from 0.01 (= 1 - second-stage threshold) to ~0.13.
        # Restore arithmetic-mean E[Var[d,d]] = 1 with one global scalar
        # per source (mirroring _normalize_ragged's convention) so the
        # encoder sees unit-mean-variance input regardless of the pre-PCA
        # threshold. Saves the scale into cfg for inference round-tripping.
        sigma_post_pca_per_source: list[float] = []
        if normalize_per_condition:
            src_ids_iter = src_ids_for_norm
            split_src_arr = np.asarray(split_src)
        else:
            src_ids_iter = [0]
            split_src_arr = np.zeros(padded_norm.shape[0], dtype=np.int64)
        for s in src_ids_iter:
            traj_idxs = np.where(split_src_arr == s)[0]
            valid_chunks = []
            for i in traj_idxs:
                L_i = int(lengths_t[i].item())
                if L_i > 0:
                    valid_chunks.append(padded_norm[i, :L_i])
            if not valid_chunks:
                sigma_post_pca_per_source.append(1.0)
                continue
            valid_concat = torch.cat(valid_chunks, dim=0)   # (sum_L, n_total_out)
            sigma_s = float(valid_concat.std().item())      # global scalar per source
            sigma_post_pca_per_source.append(sigma_s)
            for i in traj_idxs:
                L_i = int(lengths_t[i].item())
                if L_i > 0:
                    padded_norm[i, :L_i] = padded_norm[i, :L_i] / sigma_s
        log.info(
            f"  post-pre-PCA arith-mean rescale: "
            f"sigma_per_source={sigma_post_pca_per_source}"
        )
        OmegaConf.update(
            cfg, "data.postprocessing.sigma_post_pca_per_source",
            list(sigma_post_pca_per_source), force_add=True,
        )
        OmegaConf.update(
            cfg, "data.postprocessing.sigma_post_pca",
            float(np.mean(sigma_post_pca_per_source)), force_add=True,
        )

        # Local n_features (raw channel count → PCA component count).
        n_features = pca_fit.n_total_out
        area_indices = pca_fit.area_indices_out
        # cfg.data.flow.dim has already been multiplied by n_delays in
        # initialize_config (custom-data branch). Update it to the
        # post-PCA value × n_delays so make_model builds the encoder
        # against the right input dim.
        cfg.data.flow.dim = n_features * n_delays
        # Re-extend area_indices for delay embedding now that they live
        # in the PCA-reduced space.
        de_indices = extend_area_indices_for_delay_embedding(
            area_indices, n_delays=n_delays, n_features=n_features,
        )
        OmegaConf.update(cfg, "model.encoder.area_indices", de_indices, force_add=True)
        # NOTE: We previously also wrote cfg.model.encoder.dim and
        # cfg.model.params.dim here, but neither MLP nor
        # DirectSumCouplingEncoder accepts a `dim` kwarg — the encoder
        # gets its size from area_indices, the dynamics MLP from
        # input_dim/output_dim set later by the autodim logic. Adding
        # `dim` would later raise "unexpected keyword argument 'dim'"
        # at make_model. Just don't.
        # Persist the PCA components in the cfg under postprocessing so
        # downstream loading + roundtrip can reconstruct them. We store
        # as tensor lists since OmegaConf handles those.
        OmegaConf.update(
            cfg, "data.postprocessing.pre_pca_per_area",
            {
                "threshold": pre_pca_var_threshold,
                "n_total_out": pca_fit.n_total_out,
                "area_indices_in": pca_fit.area_indices_in,
                "area_indices_out": pca_fit.area_indices_out,
                "cum_var": [p.cum_var for p in pca_fit.per_area],
                "k_per_area": [p.k for p in pca_fit.per_area],
            },
            force_add=True,
        )

    # ---- Per-trajectory delay embed --------------------------------------
    de_padded, de_lengths = delay_embed_ragged(
        padded_norm, lengths_t, n_delays=n_delays, delay_spacing=delay_spacing,
    )
    de_padded = de_padded.to(torch.float32)
    log.info(
        f"delay-embedded ragged: {tuple(de_padded.shape)}, "
        f"lengths range [{int(de_lengths.min())}, {int(de_lengths.max())}], "
        f"total_de_timepoints={int(de_lengths.sum())}"
    )

    # ---- Per-condition timepoint-balanced trajectory split ---------------
    log.info("per-condition timepoint-balanced trajectory split:")
    train_idx, val_idx, test_idx = split_balanced_by_timepoints(
        de_lengths, split_src,
        train_percent=train_percent, test_percent=test_percent,
        seed=seed, log=log,
    )
    log.info(
        f"split sizes (trajectories): train={len(train_idx)}, "
        f"val={len(val_idx)}, test={len(test_idx)}"
    )

    # ---- Stride within each split → uniform sub-sequences ----------------
    train_seq = sliding_windows(
        de_padded[train_idx], de_lengths[train_idx], seq_length, seq_spacing,
    )
    val_seq = sliding_windows(
        de_padded[val_idx], de_lengths[val_idx], seq_length, seq_spacing,
    )
    test_seq = sliding_windows(
        de_padded[test_idx], de_lengths[test_idx], seq_length, seq_spacing,
    )
    log.info(
        f"sub-sequences (seq_length={seq_length}, seq_spacing={seq_spacing}): "
        f"train={train_seq.shape[0]}, val={val_seq.shape[0]}, test={test_seq.shape[0]}"
    )
    if min(train_seq.shape[0], val_seq.shape[0], test_seq.shape[0]) == 0:
        raise ValueError(
            f"empty split after striding (train={train_seq.shape[0]}, "
            f"val={val_seq.shape[0]}, test={test_seq.shape[0]}); seq_length="
            f"{seq_length} may be too long relative to per-trajectory "
            f"de_lengths {de_lengths.tolist()}"
        )

    # Defensive: sliding_windows respects lengths so no NaN should leak in.
    for name, seq in [("train", train_seq), ("val", val_seq), ("test", test_seq)]:
        if torch.isnan(seq).any():
            raise RuntimeError(
                f"NaN detected in {name} sub-sequences after sliding_windows; "
                f"this indicates a bug in delay_embed_ragged / sliding_windows."
            )

    # ---- Per-window condition (tile parent's condition by # windows) ------
    train_cond_per_seq = val_cond_per_seq = test_cond_per_seq = None
    if cond_arr is not None:
        train_cond_per_seq = _tile_condition_per_window(
            cond_arr[train_idx], de_lengths[train_idx], seq_length, seq_spacing,
        )
        val_cond_per_seq = _tile_condition_per_window(
            cond_arr[val_idx], de_lengths[val_idx], seq_length, seq_spacing,
        )
        test_cond_per_seq = _tile_condition_per_window(
            cond_arr[test_idx], de_lengths[test_idx], seq_length, seq_spacing,
        )
        # Sanity: per-window cond count must match sub-sequence count
        for name, seq, cond_seq in [
            ("train", train_seq, train_cond_per_seq),
            ("val", val_seq, val_cond_per_seq),
            ("test", test_seq, test_cond_per_seq),
        ]:
            if cond_seq.shape[0] != seq.shape[0]:
                raise RuntimeError(
                    f"{name} condition tile count {cond_seq.shape[0]} != "
                    f"sub-sequence count {seq.shape[0]} (bug in tiling)"
                )

    train_cond_t = (
        torch.as_tensor(train_cond_per_seq, dtype=torch.float32)
        if train_cond_per_seq is not None else None
    )
    val_cond_t = (
        torch.as_tensor(val_cond_per_seq, dtype=torch.float32)
        if val_cond_per_seq is not None else None
    )
    test_cond_t = (
        torch.as_tensor(test_cond_per_seq, dtype=torch.float32)
        if test_cond_per_seq is not None else None
    )

    train_dataset = TimeSeriesDataset(train_seq, condition=train_cond_t)
    val_dataset = TimeSeriesDataset(val_seq, condition=val_cond_t)
    test_dataset = TimeSeriesDataset(test_seq, condition=test_cond_t)

    # ---- Build trajs dict (delay-embedded NaN-padded full trajs per split)
    trajs: dict = {
        "train_trajs": train_dataset,
        "val_trajs": val_dataset,
        "test_trajs": test_dataset,
        "train_inds": train_idx,
        "val_inds": val_idx,
        "test_inds": test_idx,
        "train_trajs_full": TimeSeriesDataset(de_padded[train_idx]),
        "val_trajs_full": TimeSeriesDataset(de_padded[val_idx]),
        "test_trajs_full": TimeSeriesDataset(de_padded[test_idx]),
        "train_trajs_full_lengths": de_lengths[train_idx],
        "val_trajs_full_lengths": de_lengths[val_idx],
        "test_trajs_full_lengths": de_lengths[test_idx],
    }
    if cond_arr is not None:
        trajs["train_condition"] = cond_arr[train_idx]
        trajs["val_condition"] = cond_arr[val_idx]
        trajs["test_condition"] = cond_arr[test_idx]

    # ---- Generalized variance (loss-scaling denom) ------------------------
    n_recent = OmegaConf.select(cfg, "model.n_recent_dims", default=None)
    if n_recent is not None and n_recent < train_seq.shape[-1]:
        gv_input = train_seq[..., :n_recent]
    else:
        gv_input = train_seq
    gv = compute_generalized_variance(gv_input)
    OmegaConf.update(cfg, "data.postprocessing.generalized_variance", gv, force_add=True)
    log.info(f"Generalized variance det(Cov)^(1/D) = {gv:.6g}")

    # ---- DataLoaders -----------------------------------------------------
    collate_fn = collate_with_optional_condition if cond_arr is not None else None
    num_workers = 2
    persistent_workers = True
    pin_memory = True
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_seed = int(cfg.data.flow.random_state)
    val_perm = torch.randperm(
        len(val_dataset), generator=torch.Generator().manual_seed(val_seed)
    )
    val_dataset_perm = Subset(val_dataset, val_perm.tolist())
    val_dl = DataLoader(
        val_dataset_perm,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    # ---- Autodim PCA / FNN ------------------------------------------------
    autodim_result = None
    if n_target_dims is None:
        autodim_result = infer_n_target_dims(cfg, train_seq)
        if autodim_result is not None:
            cfg.model.n_target_dims = autodim_result.n_target_dims_total
            cfg.model.params.input_dim = autodim_result.n_target_dims_total
            cfg.model.params.output_dim = autodim_result.n_target_dims_total ** 2
            if autodim_result.is_direct_sum:
                cfg.model.encoder.n_target_dims_per_block = list(
                    autodim_result.n_target_dims_per_block
                )

    # ---- Run name --------------------------------------------------------
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

    if init_from_pretrained is not None:
        _load_pretrained_into_litmodel(
            lit_model, init_from_pretrained, strict=init_from_pretrained_strict,
        )

    log_training_info(train_dl, trajs, lit_model, log=log)

    # ---- Train -----------------------------------------------------------
    trainer = train_model(cfg, lit_model, train_dl, val_dl, name=name)

    best_ckpt = None
    for cb in getattr(trainer, "callbacks", []):
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


def _normalize_ragged(
    padded: torch.Tensor,
    lengths: torch.Tensor,
    source_id: np.ndarray,
    *,
    normalize: bool,
    obs_noise: float,
    scale_noise: bool = True,
) -> tuple[torch.Tensor, list[float], list[float], list[float], list[int]]:
    """NaN-aware per-source obs-noise + z-score on a ragged padded tensor.

    For each unique value in ``source_id``, computes
    ``noise_scale_factor`` and ``(mu, sigma)`` from VALID samples only
    (i.e. ``padded[i, :lengths[i]]`` for trajectories ``i`` in that
    source), adds Gaussian noise of std ``obs_noise * nsf`` to the
    valid samples (when ``obs_noise > 0``), then z-score normalizes.
    NaN-padded entries stay NaN.

    To get global (single-source) behavior, pass ``source_id`` as a
    constant array; the function computes a single ``(mu, sigma, nsf)``.
    """
    src_arr = np.asarray(source_id)
    out = padded.clone().to(torch.float32)
    per_mu: list[float] = []
    per_sigma: list[float] = []
    per_nsf: list[float] = []
    src_ids: list[int] = []
    for s in np.unique(src_arr):
        traj_idxs = np.where(src_arr == s)[0]
        # Concatenate all valid samples for this source for stat computation.
        valid_chunks = []
        for i in traj_idxs:
            L_i = int(lengths[i].item())
            if L_i > 0:
                valid_chunks.append(padded[i, :L_i].to(torch.float32))
        if not valid_chunks:
            raise ValueError(
                f"_normalize_ragged: source {s!r} has zero valid samples"
            )
        valid_concat = torch.cat(valid_chunks, dim=0)  # (T_valid, D)

        if scale_noise and obs_noise > 0:
            nsf = float(
                torch.linalg.norm(valid_concat, dim=-1).mean().item()
                / np.sqrt(valid_concat.shape[-1])
            )
        else:
            nsf = 1.0

        # Optionally inject noise into each trajectory's valid region.
        # Recompute mu/sigma POST-noise (matches postprocess_data
        # convention which normalizes after adding noise).
        if obs_noise > 0:
            std = obs_noise * nsf
            for i in traj_idxs:
                L_i = int(lengths[i].item())
                if L_i > 0:
                    out[i, :L_i] = padded[i, :L_i].to(torch.float32) + (
                        torch.randn(L_i, padded.shape[-1], dtype=torch.float32) * std
                    )
            # Recollect post-noise valid samples for stats.
            valid_chunks = [out[i, :int(lengths[i].item())] for i in traj_idxs]
            valid_concat = torch.cat(valid_chunks, dim=0)

        if normalize:
            mu = float(valid_concat.mean().item())
            sigma = float(valid_concat.std().item())
        else:
            mu = 0.0
            sigma = 1.0

        # Apply z-score to valid samples only.
        if normalize:
            for i in traj_idxs:
                L_i = int(lengths[i].item())
                if L_i > 0:
                    if obs_noise > 0:
                        # noise already in `out`
                        out[i, :L_i] = (out[i, :L_i] - mu) / sigma
                    else:
                        out[i, :L_i] = (padded[i, :L_i].to(torch.float32) - mu) / sigma

        per_mu.append(mu)
        per_sigma.append(sigma)
        per_nsf.append(nsf)
        src_ids.append(int(s))

    return out, per_mu, per_sigma, per_nsf, src_ids


def _tile_condition_per_window(
    cond_for_split: np.ndarray,
    de_lengths_for_split: torch.Tensor,
    seq_length: int,
    seq_spacing: int,
) -> np.ndarray:
    """For each parent trajectory in a split, repeat its condition vector
    once per sub-window that :func:`sliding_windows` will emit for that
    trajectory. Used to align the per-window conditions in the
    TimeSeriesDataset with the windows that ``sliding_windows`` produces
    in trajectory-then-window order.

    Parameters
    ----------
    cond_for_split : np.ndarray of shape ``(N_split, condition_dim)``
        Per-parent-trajectory condition vectors for one split (already
        indexed by ``train_idx`` / ``val_idx`` / ``test_idx``).
    de_lengths_for_split : torch.Tensor of shape ``(N_split,)``
        Per-parent-trajectory delay-embedded valid lengths for one
        split.
    seq_length, seq_spacing : int
        Same values passed to :func:`sliding_windows`.

    Returns
    -------
    np.ndarray of shape ``(K_total_windows, condition_dim)``
        Tiled per-window conditions in the same order
        :func:`sliding_windows` emits them.
    """
    cond = np.asarray(cond_for_split)
    if cond.ndim == 1:
        cond = cond.reshape(-1, 1)
    if isinstance(de_lengths_for_split, torch.Tensor):
        de_lengths_arr = de_lengths_for_split.detach().cpu().numpy().astype(np.int64)
    else:
        de_lengths_arr = np.asarray(de_lengths_for_split, dtype=np.int64)
    if cond.shape[0] != de_lengths_arr.shape[0]:
        raise ValueError(
            f"_tile_condition_per_window: cond shape {cond.shape} disagrees "
            f"with de_lengths shape {de_lengths_arr.shape}"
        )
    pieces = []
    for i, L_i in enumerate(de_lengths_arr):
        if L_i < seq_length:
            continue
        n_windows = (int(L_i) - seq_length) // seq_spacing + 1
        if n_windows > 0:
            pieces.append(np.repeat(cond[i:i + 1], n_windows, axis=0))
    if not pieces:
        return np.zeros((0, cond.shape[1]), dtype=cond.dtype)
    return np.concatenate(pieces, axis=0)


def _compose_cfg(
    *, encoder, n_features,
    n_delays, delay_spacing, observed_indices,
    seq_length, seq_spacing, train_percent, test_percent, split_by,
    obs_noise, normalize, normalize_per_condition,
    filter_data, low_pass, high_pass,
    post_filter_downsample,
    pre_pca_per_area, pre_pca_var_threshold,
    whiten_after_pre_pca,
    n_dynamics_per_source, section_condition_values,
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
    if n_dynamics_per_source is not None and int(n_dynamics_per_source) > 1:
        if section_condition_values is None:
            raise ValueError(
                "n_dynamics_per_source > 1 requires section_condition_values "
                "(one float per sub-MLP)."
            )
        if len(section_condition_values) != int(n_dynamics_per_source):
            raise ValueError(
                f"n_dynamics_per_source={n_dynamics_per_source} but "
                f"section_condition_values has length {len(section_condition_values)} — "
                f"they must match."
            )
        overrides.append(f"++model.n_dynamics_per_source={int(n_dynamics_per_source)}")
        overrides.append(
            "++model.section_condition_values="
            + "[" + ",".join(repr(float(v)) for v in section_condition_values) + "]"
        )
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
