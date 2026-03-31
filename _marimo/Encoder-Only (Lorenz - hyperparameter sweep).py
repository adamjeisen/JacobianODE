import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Encoder-Only — Hyperparameter Sweep (Lorenz)

    Performs a **Hydra grid sweep** over encoder-only regularisation parameters
    using SLURM for parallel training.

    **Pipeline:**
    1. Set hyperparameters in clearly labelled cells (Sections 1–2)
    2. Build Hydra config (single source of truth for local + sweep)
    3. Generate data locally for post-hoc diagnostics
    4. Check W&B for already-completed runs
    5. Launch Hydra `--multirun` sweep via SLURM (one job per parameter combination)
    6. Collect results and visualise

    > **Wait for all SLURM jobs to finish** before running Section 7.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _():
    import itertools
    import subprocess
    import sys, os

    REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import wandb
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from JacobianODE.jacobians.core import seed_everything
    from JacobianODE.jacobians.data import make_trajectories, postprocess_data, create_dataloaders
    from JacobianODE.jacobians.training import train_model
    from JacobianODE.encoder_only.config import load_encoder_config

    torch.set_float32_matmul_precision('high')
    print(f'PyTorch {torch.__version__}  |  GPUs: {torch.cuda.device_count()}')
    return (
        OmegaConf,
        create_dataloaders,
        itertools,
        load_encoder_config,
        make_trajectories,
        pd,
        plt,
        postprocess_data,
        seed_everything,
        subprocess,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Hyperparameters
    """)
    return


@app.cell
def _():
    # ----------------------------------------------------------------
    # Paths and W&B settings
    # ----------------------------------------------------------------
    SAVE_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/encoder_runs"
    WANDB_ENTITY = "JacobianODE"
    WANDB_PROJECT = None  # set to None for auto-generated name
    WANDB_PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_PROJECT else None
    return SAVE_DIR, WANDB_ENTITY, WANDB_PROJECT


@app.cell
def _():
    # ----------------------------------------------------------------
    # Data hyperparameters
    # ----------------------------------------------------------------
    NUM_ICS = 32
    N_PERIODS = 12
    PTS_PER_PERIOD = 100
    # SEQ_LENGTH = 50
    SEQ_LENGTH = 100
    # OBS_NOISE = 0.05
    OBS_NOISE = 0.01

    # Partial observation (delay embedding)
    # OBSERVED_INDICES = [0, 1, 2]   # Observe ALL three Lorenz dimensions; use [0] for x-only
    OBSERVED_INDICES = [0]
    # N_DELAYS         = 1           # Number of delays for delay embedding (1 = no embedding)
    # N_DELAYS         = 43
    DELAY_SPACING    = 1           # Spacing between delays when N_DELAYS > 1
    # N_DELAYS = 100
    N_DELAYS = 25
    # DELAY_SPACING = 10
    return (
        DELAY_SPACING,
        NUM_ICS,
        N_DELAYS,
        N_PERIODS,
        OBSERVED_INDICES,
        OBS_NOISE,
        PTS_PER_PERIOD,
        SEQ_LENGTH,
    )


@app.cell
def _(OBSERVED_INDICES):
    # ----------------------------------------------------------------
    # Encoder / architecture hyperparameters
    # ----------------------------------------------------------------
    # MODEL = 'ssm'              # SSM (LRU) sequence encoder
    # MODEL = 'transformer'      # Transformer sequence encoder
    # MODEL = 'tcn'              # TCN sequence encoder
    # MODEL = 'coupling'         # AffineCouplingEncoder (invertible, dim-preserving)
    MODEL = 'spline_coupling'    # CouplingEncoder with rational quadratic splines + ActNorm

    assert MODEL in ('ssm', 'transformer', 'tcn', 'coupling', 'spline_coupling'), f"Invalid MODEL: {MODEL}"
    IS_COUPLING = MODEL in ('coupling', 'spline_coupling')

    # ============================================================
    # Sequence encoder settings (ssm / transformer / tcn)
    # ============================================================
    if not IS_COUPLING:
        N_LATENT = 10

        # SSM (LRU) parameters -- used when MODEL='ssm'
        SSM_D_MODEL    = 128
        SSM_D_STATE    = 64
        SSM_N_LAYERS   = 3
        SSM_R_MIN      = 0.9
        SSM_R_MAX      = 0.999
        SSM_FFN_EXPAND = 4
        SSM_DROPOUT    = 0.1

        # Transformer parameters -- used when MODEL='transformer'
        TRANSFORMER_D_MODEL       = 64
        TRANSFORMER_N_HEADS       = 4
        TRANSFORMER_N_LAYERS      = 3
        TRANSFORMER_DIM_FEEDFORWARD = 128
        TRANSFORMER_DROPOUT       = 0.1

        # TCN parameters -- used when MODEL='tcn'
        TCN_N_CHANNELS = 64
        TCN_KERNEL_SIZE = 7
        TCN_N_LAYERS   = 4
        TCN_DROPOUT    = 0.1

        # Decoder head (shared across sequence architectures)
        USE_SAME_STATE_DECODER = True
        USE_NEXT_STATE_DECODER = True
        DECODER_HIDDEN_DIM    = 128
        DECODER_N_LAYERS      = 2
        NEXT_STATE_BURN_IN    = 0

        # Multi-step ahead prediction
        K_STEPS_AHEAD = 10
        N_OBS_PRED = len(OBSERVED_INDICES)

    # ============================================================
    # Coupling flow encoder settings (coupling / spline_coupling)
    # Self-supervised: encode → split → zero-pad → inverse decode → MSE
    # ============================================================
    if IS_COUPLING:
        # ---- Architecture ----
        N_COUPLING_LAYERS   = 8      # number of coupling stacks
        COUPLING_HIDDEN_DIM = 128    # conditioner MLP width
        N_HIDDEN_LAYERS     = 2      # conditioner MLP depth
        ZERO_INIT           = True   # identity initialization trick
        PERMUTATION_SEED    = 42     # base seed for fixed random permutations
        USE_LOFT            = False  # LOFT layer after coupling blocks
        LOFT_TAU            = 100.0  # LOFT threshold

        # Affine coupling only (ignored for spline_coupling)
        SCALE_ACTIVATION    = "tanh"       # bounds log-scale via tanh
        SCALE_CLAMP         = 3.0          # max |log_s|
        CLAMP_TYPE          = "symmetric"  # 'symmetric' | 'asymmetric'
        ALPHA_POS           = 0.1          # asymmetric clamp: expansion bound
        ALPHA_NEG           = 2.0          # asymmetric clamp: compression bound

        # Spline coupling only (ignored for affine coupling)
        NUM_BINS            = 8      # rational quadratic spline segments
        TAIL_BOUND          = 3.0    # linear tails outside [-B, B]
        USE_ACTNORM         = True   # ActNorm between coupling layers

        # ---- Subspace splitting ----
        N_TARGET_DIMS        = 3           # dynamic subspace (e.g. 3 for Lorenz)
        KL_NULL_WEIGHT = 1.0               # null-space MSE penalty (structural constraint)
        KL_DYN_WEIGHT = 0.0                # dynamic subspace KL divergence (smoothness regularizer)
        RECONSTRUCTION_MODE  = "most_recent"  # 'uniform' | 'harmonic' | 'most_recent'

        # ---- VAE settings ----
        USE_VAE                = True     # VAE reparameterization on dynamic subspace
        KL_WARMUP_EPOCHS       = 0        # 0 = fixed weight, >0 = linear ramp

    print(f"Encoder: {MODEL}" + (f" (n_target_dims={N_TARGET_DIMS})" if IS_COUPLING else f" (n_latent={N_LATENT})"))
    return (
        ALPHA_NEG,
        ALPHA_POS,
        CLAMP_TYPE,
        COUPLING_HIDDEN_DIM,
        DECODER_HIDDEN_DIM,
        DECODER_N_LAYERS,
        IS_COUPLING,
        KL_DYN_WEIGHT,
        KL_NULL_WEIGHT,
        KL_WARMUP_EPOCHS,
        K_STEPS_AHEAD,
        LOFT_TAU,
        MODEL,
        NEXT_STATE_BURN_IN,
        NUM_BINS,
        N_COUPLING_LAYERS,
        N_HIDDEN_LAYERS,
        N_LATENT,
        N_OBS_PRED,
        N_TARGET_DIMS,
        PERMUTATION_SEED,
        RECONSTRUCTION_MODE,
        SCALE_ACTIVATION,
        SCALE_CLAMP,
        SSM_DROPOUT,
        SSM_D_MODEL,
        SSM_D_STATE,
        SSM_FFN_EXPAND,
        SSM_N_LAYERS,
        SSM_R_MAX,
        SSM_R_MIN,
        TAIL_BOUND,
        TCN_DROPOUT,
        TCN_KERNEL_SIZE,
        TCN_N_CHANNELS,
        TCN_N_LAYERS,
        TRANSFORMER_DIM_FEEDFORWARD,
        TRANSFORMER_DROPOUT,
        TRANSFORMER_D_MODEL,
        TRANSFORMER_N_HEADS,
        TRANSFORMER_N_LAYERS,
        USE_ACTNORM,
        USE_LOFT,
        USE_NEXT_STATE_DECODER,
        USE_SAME_STATE_DECODER,
        USE_VAE,
        ZERO_INIT,
    )


@app.cell
def _():
    # ----------------------------------------------------------------
    # Training hyperparameters
    # ----------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 150
    LIMIT_TRAIN_BATCHES = 200
    EARLY_STOPPING_PATIENCE = 2
    PERCENT_THRESH = 0.01
    return (
        BATCH_SIZE,
        EARLY_STOPPING_PATIENCE,
        LEARNING_RATE,
        LIMIT_TRAIN_BATCHES,
        MAX_EPOCHS,
        PERCENT_THRESH,
    )


@app.cell
def _(IS_COUPLING):
    if IS_COUPLING:
        SWEEP_PARAMS = {'model.kl_null_weight': [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
    else:
        SWEEP_PARAMS = {'training.lightning.fnn_weight': [0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'training.lightning.fnn_normalize': [True], 'training.lightning.fnn_elementwise_regularization': [True], 'training.lightning.amplification_weight': [0.0], 'training.lightning.decov_weight': [0.0]}
    n_combos = 1
    for _vals in SWEEP_PARAMS.values():
        n_combos = n_combos * len(_vals)
    print(f'Sweep grid: {n_combos} total combinations')
    for k, v in SWEEP_PARAMS.items():
        print(f'  {k}: {v}')
    return (SWEEP_PARAMS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Build Base Config

    Build the Hydra overrides list — this is the **single source of truth** used both
    for local data generation and as the template for the sweep command.
    """)
    return


@app.cell
def _(
    ALPHA_NEG,
    ALPHA_POS,
    BATCH_SIZE,
    CLAMP_TYPE,
    COUPLING_HIDDEN_DIM,
    DECODER_HIDDEN_DIM,
    DECODER_N_LAYERS,
    DELAY_SPACING,
    EARLY_STOPPING_PATIENCE,
    IS_COUPLING,
    KL_DYN_WEIGHT,
    KL_NULL_WEIGHT,
    KL_WARMUP_EPOCHS,
    K_STEPS_AHEAD,
    LEARNING_RATE,
    LIMIT_TRAIN_BATCHES,
    LOFT_TAU,
    MAX_EPOCHS,
    MODEL,
    NEXT_STATE_BURN_IN,
    NUM_BINS,
    NUM_ICS,
    N_COUPLING_LAYERS,
    N_DELAYS,
    N_HIDDEN_LAYERS,
    N_LATENT,
    N_OBS_PRED,
    N_PERIODS,
    N_TARGET_DIMS,
    OBSERVED_INDICES,
    OBS_NOISE,
    OmegaConf,
    PERCENT_THRESH,
    PERMUTATION_SEED,
    PTS_PER_PERIOD,
    RECONSTRUCTION_MODE,
    SAVE_DIR,
    SCALE_ACTIVATION,
    SCALE_CLAMP,
    SEQ_LENGTH,
    SSM_DROPOUT,
    SSM_D_MODEL,
    SSM_D_STATE,
    SSM_FFN_EXPAND,
    SSM_N_LAYERS,
    SSM_R_MAX,
    SSM_R_MIN,
    TAIL_BOUND,
    TCN_DROPOUT,
    TCN_KERNEL_SIZE,
    TCN_N_CHANNELS,
    TCN_N_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_N_LAYERS,
    USE_ACTNORM,
    USE_LOFT,
    USE_NEXT_STATE_DECODER,
    USE_SAME_STATE_DECODER,
    USE_VAE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    ZERO_INIT,
    load_encoder_config,
):
    # Format list-valued overrides without spaces so they double as valid Hydra CLI args
    _obs_idx_str = str(list(OBSERVED_INDICES)).replace(' ', '')
    overrides = [f'model={MODEL}']
    if IS_COUPLING:
        _n_raw_obs = len(OBSERVED_INDICES)  # --- Model ---
        _n_input = N_DELAYS * _n_raw_obs
        overrides.extend([f'++model.encoder.n_input={_n_input}', f'++model.encoder.n_coupling_layers={N_COUPLING_LAYERS}', f'++model.encoder.hidden_dim={COUPLING_HIDDEN_DIM}', f'++model.encoder.n_hidden_layers={N_HIDDEN_LAYERS}', f'++model.encoder.zero_init={str(ZERO_INIT).lower()}', f'++model.encoder.permutation_seed={PERMUTATION_SEED}', f'++model.encoder.use_loft={str(USE_LOFT).lower()}', f'++model.encoder.loft_tau={LOFT_TAU}'])
        if MODEL == 'coupling':
    # ============================================================
    # Coupling flow encoder overrides
            overrides.extend([f'++model.encoder.scale_activation={SCALE_ACTIVATION}', f'++model.encoder.scale_clamp={SCALE_CLAMP}', f'++model.encoder.clamp_type={CLAMP_TYPE}', f'++model.encoder.alpha_pos={ALPHA_POS}', f'++model.encoder.alpha_neg={ALPHA_NEG}'])
        elif MODEL == 'spline_coupling':
            overrides.extend([f'++model.encoder.num_bins={NUM_BINS}', f'++model.encoder.tail_bound={TAIL_BOUND}', f'++model.encoder.use_actnorm={str(USE_ACTNORM).lower()}'])
        overrides.extend([f'++model.n_target_dims={N_TARGET_DIMS}', f'++model.kl_null_weight={KL_NULL_WEIGHT}', f'++model.kl_dyn_weight={KL_DYN_WEIGHT}', f'++model.reconstruction_mode={RECONSTRUCTION_MODE}', f'++model.use_vae={str(USE_VAE).lower()}', f'++model.kl_warmup_epochs={KL_WARMUP_EPOCHS}'])
    else:
        overrides.extend([f'++model.encoder.n_latent={N_LATENT}', f'++model.use_same_state_decoder={str(USE_SAME_STATE_DECODER).lower()}', f'++model.use_next_state_decoder={str(USE_NEXT_STATE_DECODER).lower()}', f'++model.decoder_hidden_dim={DECODER_HIDDEN_DIM}', f'++model.decoder_n_layers={DECODER_N_LAYERS}', f'++model.next_state_burn_in={NEXT_STATE_BURN_IN}', f'++model.k_steps_ahead={K_STEPS_AHEAD}', f'++model.n_obs_pred={N_OBS_PRED}'])
        if MODEL == 'ssm':
            overrides.extend([f'++model.encoder.d_model={SSM_D_MODEL}', f'++model.encoder.d_state={SSM_D_STATE}', f'++model.encoder.n_layers={SSM_N_LAYERS}', f'++model.encoder.r_min={SSM_R_MIN}', f'++model.encoder.r_max={SSM_R_MAX}', f'++model.encoder.ffn_expand={SSM_FFN_EXPAND}', f'++model.encoder.dropout={SSM_DROPOUT}'])
        elif MODEL == 'transformer':
            overrides.extend([f'++model.encoder.d_model={TRANSFORMER_D_MODEL}', f'++model.encoder.n_heads={TRANSFORMER_N_HEADS}', f'++model.encoder.n_layers={TRANSFORMER_N_LAYERS}', f'++model.encoder.dim_feedforward={TRANSFORMER_DIM_FEEDFORWARD}', f'++model.encoder.dropout={TRANSFORMER_DROPOUT}'])
        elif MODEL == 'tcn':
            overrides.extend([f'++model.encoder.n_channels={TCN_N_CHANNELS}', f'++model.encoder.kernel_size={TCN_KERNEL_SIZE}', f'++model.encoder.n_layers={TCN_N_LAYERS}', f'++model.encoder.dropout={TCN_DROPOUT}'])
    overrides.extend([f'++training.batch_size={BATCH_SIZE}', f'++training.lightning.optimizer_kwargs.lr={LEARNING_RATE}', f'++training.trainer_params.max_epochs={MAX_EPOCHS}', f'++training.trainer_params.limit_train_batches={LIMIT_TRAIN_BATCHES}', f'++training.early_stopping.early_stopping_patience={EARLY_STOPPING_PATIENCE}', f'++training.early_stopping.percent_thresh={PERCENT_THRESH}', f'++data.trajectory_params.num_ics={NUM_ICS}', f'++data.trajectory_params.n_periods={N_PERIODS}', f'++data.trajectory_params.pts_per_period={PTS_PER_PERIOD}', f'++data.train_test_params.seq_length={SEQ_LENGTH}', f'++data.postprocessing.obs_noise={OBS_NOISE}', f'++data.train_test_params.delay_embedding_params.observed_indices={_obs_idx_str}', f'++data.train_test_params.delay_embedding_params.n_delays={N_DELAYS}', f'++data.train_test_params.delay_embedding_params.delay_spacing={DELAY_SPACING}', f'++training.logger.save_dir={SAVE_DIR}'])
    cfg = load_encoder_config(overrides=overrides)
    if WANDB_PROJECT is None:
        data_cls = cfg.data.flow._target_.split('.')[-1]
        if IS_COUPLING:
            WANDB_PROJECT_1 = f'{data_cls}__{MODEL}__EncoderOnly'
        else:
            WANDB_PROJECT_1 = f'{data_cls}__EncoderOnly'
        WANDB_PROJECT_PATH_1 = f'{WANDB_ENTITY}/{WANDB_PROJECT_1}'
    print(f'W&B project: {WANDB_PROJECT_PATH_1}')
    enc_target = cfg.model.encoder._target_.split('.')[-1]
    print(f'Encoder: {enc_target}')
    if IS_COUPLING:
        print(f'  n_target_dims={N_TARGET_DIMS}, kl_null={KL_NULL_WEIGHT}, kl_dyn={KL_DYN_WEIGHT}, recon_mode={RECONSTRUCTION_MODE}')
        if USE_VAE:
            print(f'  VAE: enabled (kl_warmup={KL_WARMUP_EPOCHS})')
    else:
        print(f'  n_latent={cfg.model.encoder.n_latent}, d_model={cfg.model.encoder.d_model}')
    # Sequence encoder overrides (ssm / transformer / tcn)
    # Shared overrides (training, data, paths)
    # Auto-generate project name if not set
    print(OmegaConf.to_yaml(cfg))  # --- Training ---  # --- Data ---  # --- Partial observation / delay embedding ---  # --- Save directory ---
    return WANDB_PROJECT_1, WANDB_PROJECT_PATH_1, cfg, overrides


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Generate Data

    Generate trajectories once locally. The same data pipeline runs inside each
    SLURM job (deterministic via fixed seed), but we also need it here for
    post-hoc diagnostics.
    """)
    return


@app.cell
def _(cfg, make_trajectories, seed_everything):
    seed_everything(cfg.data.flow.random_state)
    eq, sol, dt = make_trajectories(cfg)
    print(f"Full trajectory shape: {sol['values'].shape}")
    print(f"Time step dt = {dt:.4f}")
    return (sol,)


@app.cell
def _(cfg, create_dataloaders, postprocess_data, sol):
    result = postprocess_data(cfg, sol['values'])
    values = result.values
    cfg.data.postprocessing.noise_scale_factor = result.noise_scale_factor
    cfg.data.postprocessing.mu = result.mu
    cfg.data.postprocessing.sigma = result.sigma

    train_dl, val_dl, test_dl, trajs = create_dataloaders(cfg, values)
    n_obs = trajs['train_trajs'].sequence.shape[-1]

    print(f"n_obs = {n_obs}")
    print(f"Train: {len(train_dl.dataset)} sequences")
    print(f"Val:   {len(val_dl.dataset)} sequences")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Check W&B for Completed Runs

    Query W&B for runs that match the current sweep parameter combinations.
    Only the remaining (not-yet-completed) combinations will be launched.
    """)
    return


@app.cell
def _(SWEEP_PARAMS, WANDB_PROJECT_PATH_1, itertools, pd, wandb):
    def _get_sweep_param_from_run(run, key):
        """Extract a sweep parameter value from a W&B run config.

        Keys are dot-separated Hydra paths like 'training.lightning.fnn_weight'.
        """
        parts = _key.split('.')
        val = run.config
        for p in parts:
            if not isinstance(val, dict) or p not in val:
                return None
            val = val[p]
        return val

    def _combo_matches_run(combo, run):
        """Check if a parameter combination matches a finished W&B run."""
        for _key, target_val in _combo.items():
            run_val = _get_sweep_param_from_run(run, _key)
            if run_val is None:
                return False
            if isinstance(target_val, bool):
                if bool(run_val) != target_val:
                    return False
            elif isinstance(target_val, (int, float)):
                if abs(float(run_val) - float(target_val)) > 1e-10:
                    return False
            elif str(run_val) != str(target_val):
                return False
        return True

    def _checkpoint_monitor_for_run(run):
        """Same monitor string as Lightning ModelCheckpoint (from saved W&B config if present)."""
        try:
            cfg = run.config
            if isinstance(cfg, dict):
                tr = cfg.get('training')
                if isinstance(tr, dict):
                    mc = tr.get('model_checkpoint')
                    if isinstance(mc, dict) and mc.get('monitor'):
                        return str(mc['monitor'])
        except Exception:
            pass
        return 'mean val loss'

    def _checkpoint_mode_for_run(run):
        try:
            cfg = run.config
            if isinstance(cfg, dict):
                tr = cfg.get('training')
                if isinstance(tr, dict):
                    mc = tr.get('model_checkpoint')
                    if isinstance(mc, dict) and mc.get('mode'):
                        return str(mc['mode'])
        except Exception:
            pass
        return 'min'

    def _metrics_at_best_checkpoint_epoch(run, is_coupling):
        """Validation metrics at the epoch that minimized ModelCheckpoint.monitor.

        Default monitor is ``mean val loss`` (``encoder_only/conf/training/training.yaml``).
        W&B ``run.summary`` defaults to the *last* logged step; this function takes metrics
        from the *best* epoch so they match the saved checkpoint.
        """
        monitor = _checkpoint_monitor_for_run(run)
        mode = _checkpoint_mode_for_run(run)
        try:
            hist = run.history(samples=500000, pandas=True)
        except Exception:
            return None
        if hist is None or hist.empty or monitor not in hist.columns:
            return None
        m = pd.to_numeric(hist[monitor], errors='coerce')
        valid = m.notna()
        if not valid.any():
            return None
        sub = m.loc[valid]
        idx = sub.idxmax() if mode == 'max' else sub.idxmin()
        row = hist.loc[idx]

        def _cell(name):
            if name not in row.index:
                return float('nan')
            v = row[name]
            if pd.isna(v):
                return float('nan')
            try:
                return float(v)
            except (TypeError, ValueError):
                return float('nan')
        ep = _cell('epoch')
        out = {'best_epoch': ep}
        if is_coupling:
            out['val_recon_loss'] = _cell('val/recon_loss')
            out['val_kl_null_loss'] = _cell('val/kl_null_loss')
            out['val_kl_dyn_loss'] = _cell('val/kl_dyn_loss')
            out['val_kl_total'] = _cell('val/kl_total_loss')
            out['val_total_loss'] = _cell('val/total_loss')
        else:
            out['val_same_loss'] = _cell('val/same_state_loss')
            out['val_next_loss'] = _cell('val/next_state_loss')
            out['val_fnn_loss'] = _cell('val/fnn_loss')
            out['val_amp_loss'] = _cell('val/amplification_loss')
            out['latent_util'] = _cell('val/latent_utilization')
            out['val_total_loss'] = _cell('val/total_loss')
        return out
    sweep_keys = list(SWEEP_PARAMS.keys())
    sweep_value_lists = [SWEEP_PARAMS[k] for k in sweep_keys]
    all_combos = [dict(zip(sweep_keys, _vals)) for _vals in itertools.product(*sweep_value_lists)]
    _api = wandb.Api()
    try:
        existing_runs = _api.runs(WANDB_PROJECT_PATH_1)
        print(f'Found {len(existing_runs)} total runs in {WANDB_PROJECT_PATH_1}')
    except Exception as e:
        print(f'Could not query project (may not exist yet): {e}')
        existing_runs = []
    finished_runs = [r for r in existing_runs if r.state == 'finished']
    already_done = []
    remaining_combos = []
    for _combo in all_combos:
        if any((_combo_matches_run(_combo, r) for r in finished_runs)):
            already_done.append(_combo)
        else:
            remaining_combos.append(_combo)
    print(f'\nAlready completed: {len(already_done)} / {len(all_combos)}')
    print(f'Remaining to run:  {len(remaining_combos)}')
    return all_combos, remaining_combos, sweep_keys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Launch Hydra Grid Sweep

    Uses Hydra `--multirun` with the SLURM launcher to submit one job per
    parameter combination. The `overrides` list from Section 2 is the
    **single source of truth** — the sweep command simply appends the swept
    parameters (comma-separated) and W&B/SLURM settings.

    > **Wait for all SLURM jobs to finish** before proceeding to Section 7.
    """)
    return


@app.cell
def _(
    SWEEP_PARAMS,
    WANDB_ENTITY,
    WANDB_PROJECT_1,
    all_combos,
    overrides,
    remaining_combos,
):
    if remaining_combos:
        _swept_keys = set(SWEEP_PARAMS.keys())

        def _is_swept(ov):
            for k in _swept_keys:
                bare = ov.lstrip('+')
                if bare.startswith(k + '='):
                    return True
            return False
        fixed_overrides = [ov for ov in overrides if not _is_swept(ov)]
        n_remaining = len(remaining_combos)
        use_subset_sweep = n_remaining < len(all_combos)
        if use_subset_sweep:
            sweep_cmds = []
            for _combo in remaining_combos:
                combo_overrides = [f'++{_key}={(str(val).lower() if isinstance(val, bool) else val)}' for _key, val in _combo.items()]
                cmd_overrides = fixed_overrides + combo_overrides + [f'wandb_entity={WANDB_ENTITY}', f'wandb_project={WANDB_PROJECT_1}', 'slurm=default']
                sweep_cmds.append('python -m JacobianODE.encoder_only.run_encoder --multirun ' + ' '.join(cmd_overrides))
        else:
            sweep_param_overrides = []
            for _key, _vals in SWEEP_PARAMS.items():
                val_str = ','.join((str(v).lower() if isinstance(v, bool) else str(v) for v in _vals))
                sweep_param_overrides.append(f'++{_key}={val_str}')
            cmd_overrides = fixed_overrides + sweep_param_overrides + [f'wandb_entity={WANDB_ENTITY}', f'wandb_project={WANDB_PROJECT_1}', 'slurm=default']
            sweep_cmds = ['python -m JacobianODE.encoder_only.run_encoder --multirun ' + ' '.join(cmd_overrides)]
        sweep_type = 'subset (1 cmd per combo)' if use_subset_sweep else 'grid'
        print(f'Sweep: {n_remaining} jobs ({sweep_type})')
        print(f'\nFirst command:\n{sweep_cmds[0][:300]}...')
    else:
        sweep_cmds = []
        print('No sweep to launch -- all runs already completed.')
    return (sweep_cmds,)


@app.cell
def _(subprocess, sweep_cmds):
    # Launch the sweep (submits SLURM jobs).
    if sweep_cmds:
        if len(sweep_cmds) > 1:
            print(f"Starting {len(sweep_cmds)} commands in parallel...")
        procs = [subprocess.Popen(cmd, shell=True) for cmd in sweep_cmds]
        for i, p in enumerate(procs):
            p.wait()
            if p.returncode != 0 and len(procs) > 1:
                print(f"Command {i + 1}/{len(procs)} exited with code {p.returncode}")
        print("Sweep complete.")
    else:
        print("No sweep to launch -- all runs already completed.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Collect W&B Runs

    After all jobs finish, re-query W&B to verify all runs completed successfully.
    Crashed/failed runs are flagged.
    """)
    return


@app.cell
def _(WANDB_PROJECT_PATH_1, all_combos, wandb):
    _api = wandb.Api()
    all_runs = _api.runs(WANDB_PROJECT_PATH_1)
    print(f'Found {len(all_runs)} total runs in {WANDB_PROJECT_PATH_1}')
    for run in all_runs:
        if run.state in ('crashed', 'failed'):
            print(f'  CRASHED/FAILED: {run.id} ({run.name}) — state={run.state}')
    finished_runs_1 = [r for r in all_runs if r.state == 'finished']
    completed_combos = []
    missing_combos = []
    for _combo in all_combos:
        if any((_combo_matches_run(_combo, r) for r in finished_runs_1)):
            completed_combos.append(_combo)
        else:
            missing_combos.append(_combo)
    print(f'\nCompleted: {len(completed_combos)} / {len(all_combos)}')
    if missing_combos:
        print(f'MISSING {len(missing_combos)} combinations — re-run Section 5 to retry.')
        for _combo in missing_combos[:5]:
            print(f'  {_combo}')
        if len(missing_combos) > 5:
            print(f'  ... and {len(missing_combos) - 5} more')
    else:
        print('All combinations completed successfully!')
    return (finished_runs_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Analyse Sweep Results

    Validation metrics below are taken from the **best-checkpoint epoch** (minimum `mean val loss`, same monitor as `ModelCheckpoint` in `encoder_only/conf/training/training.yaml`), not the last epoch — so they align with the weights W&B saved. If history is unavailable, the cell falls back to `run.summary` (typically last epoch).
    """)
    return


@app.cell
def _(IS_COUPLING, all_combos, finished_runs_1, pd, sweep_keys):
    records = []
    for r in finished_runs_1:
        if not any((_combo_matches_run(_combo, r) for _combo in all_combos)):
            continue
        rec = {}
        for _key in sweep_keys:
            short_name = _key.split('.')[-1]
            rec[short_name] = _get_sweep_param_from_run(r, _key)
        m_best = _metrics_at_best_checkpoint_epoch(r, IS_COUPLING)
        if m_best is not None:
            rec.update(m_best)
        else:
            rec['best_epoch'] = float('nan')
            if IS_COUPLING:
                rec['val_recon_loss'] = r.summary.get('val/recon_loss', float('nan'))
                rec['val_kl_null_loss'] = r.summary.get('val/kl_null_loss', float('nan'))
                rec['val_kl_dyn_loss'] = r.summary.get('val/kl_dyn_loss', float('nan'))
                rec['val_kl_total'] = r.summary.get('val/kl_total_loss', float('nan'))
                rec['val_total_loss'] = r.summary.get('val/total_loss', r.summary.get('mean val loss', float('nan')))
            else:
                rec['val_same_loss'] = r.summary.get('val/same_state_loss', float('nan'))
                rec['val_next_loss'] = r.summary.get('val/next_state_loss', float('nan'))
                rec['val_fnn_loss'] = r.summary.get('val/fnn_loss', float('nan'))
                rec['val_amp_loss'] = r.summary.get('val/amplification_loss', float('nan'))
                rec['latent_util'] = r.summary.get('val/latent_utilization', float('nan'))
                rec['val_total_loss'] = r.summary.get('val/total_loss', float('nan'))
        rec['run_name'] = r.name
        rec['run_id'] = r.id
        records.append(rec)
    _sort_col = 'val_recon_loss' if IS_COUPLING else 'val_same_loss'
    df = pd.DataFrame(records)
    _metric_cols = ('best_epoch', 'val_recon_loss', 'val_kl_null_loss', 'val_kl_dyn_loss', 'val_kl_total', 'val_total_loss') if IS_COUPLING else ('best_epoch', 'val_same_loss', 'val_next_loss', 'val_fnn_loss', 'val_amp_loss', 'latent_util', 'val_total_loss')
    for c in _metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for _key in sweep_keys:
        c = _key.split('.')[-1]
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values(_sort_col)
    print(f'{len(df)} finished runs in results')
    df.head(10)
    return (df,)


@app.cell
def _(IS_COUPLING, df, plt):
    # ---- Scatter: sweep param vs primary loss ----
    _fig, _ax = plt.subplots(figsize=(7, 5))
    if IS_COUPLING:
        _x_col = 'kl_null_weight'
        _y_col = 'val_recon_loss'
        _sc = _ax.scatter(df[_x_col], df[_y_col], s=60, alpha=0.8, edgecolors='k', linewidths=0.4)
        _ax.set_xlabel('kl_null_weight')
        _ax.set_ylabel('val/recon_loss')
        _ax.set_title('Sweep: KL weight vs. reconstruction loss')
    else:
        _x_col = 'fnn_weight'
        _y_col = 'val_same_loss'
        _sc = _ax.scatter(df[_x_col], df[_y_col], c=df['amplification_weight'], cmap='viridis', s=60, alpha=0.8, edgecolors='k', linewidths=0.4)
        plt.colorbar(_sc, ax=_ax, label='amplification_weight')
        _ax.set_xlabel('fnn_weight')
        _ax.set_ylabel('val/same_state_loss')
        _ax.set_title('Sweep: regularisation vs. reconstruction loss')
    _ax.set_xscale('symlog', linthresh=0.0001)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(IS_COUPLING, df, plt):
    # ---- Scatter: sweep param vs secondary loss ----
    _fig, _ax = plt.subplots(figsize=(7, 5))
    if IS_COUPLING:
        _x_col = 'kl_null_weight'
        _y_col = 'val_kl_null_loss'
        _sc = _ax.scatter(df[_x_col], df[_y_col], s=60, alpha=0.8, edgecolors='k', linewidths=0.4)
        _ax.set_xlabel('kl_null_weight')
        _ax.set_ylabel('val/kl_null_loss')
        _ax.set_title('Sweep: KL weight vs. null-space loss')
    else:
        _x_col = 'fnn_weight'
        _y_col = 'val_next_loss'
        _sc = _ax.scatter(df[_x_col], df[_y_col], c=df['amplification_weight'], cmap='viridis', s=60, alpha=0.8, edgecolors='k', linewidths=0.4)
        plt.colorbar(_sc, ax=_ax, label='amplification_weight')
        _ax.set_xlabel('fnn_weight')
        _ax.set_ylabel('val/next_state_loss')
        _ax.set_title('Sweep: regularisation vs. next-state loss')
    _ax.set_xscale('symlog', linthresh=0.0001)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(IS_COUPLING, df, plt):
    # ---- Loss distribution across sweep runs ----
    _fig, _ax = plt.subplots(figsize=(6, 4))
    if IS_COUPLING:
        _col = 'val_recon_loss'
        _ax.hist(df[_col].dropna(), bins=20, color='steelblue', edgecolor='white')
        _ax.set_xlabel('val/recon_loss')
        _ax.set_title('Reconstruction loss across sweep runs')
    else:
        _col = 'latent_util'
        _ax.hist(df[_col].dropna(), bins=20, color='steelblue', edgecolor='white')
        _ax.set_xlabel('latent utilisation (entropy-based, 0–1)')
        _ax.set_title('Latent utilisation across sweep runs')
    _ax.set_ylabel('count')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(IS_COUPLING, df):
    # ---- Best run summary ----
    _sort_col = 'val_recon_loss' if IS_COUPLING else 'val_same_loss'
    best = df.sort_values(_sort_col).iloc[0]
    print(f'Best run (lowest {_sort_col}):')
    for _col in df.columns:
        print(f'  {_col:40s}: {best[_col]}')
    return


if __name__ == "__main__":
    app.run()
