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
    # Hyperparameter Sweep — Optuna (Latent JacobianODE)

    This notebook launches a **Bayesian hyperparameter sweep** for the Latent JacobianODE model
    using [Optuna](https://optuna.org/) via the `hydra-optuna-sweeper` plugin.

    **How it differs from the grid sweep notebook:**
    - Instead of exhaustive grid search (`itertools.product`), Optuna uses **TPE (Tree-structured Parzen Estimators)** to sample promising hyperparameter regions.
    - Fewer trials needed to find good configurations (sample-efficient).
    - Search spaces are continuous ranges (log-uniform, uniform) rather than discrete lists.
    - Optional SQLite storage for **resumable** sweeps across SLURM failures.

    **Workflow:**
    1. Configure fixed params + Optuna search space (Sections 1–4)
    2. Build and launch the `--multirun sweeper=optuna` command (Section 5)
    3. Analyze results via Optuna study + W&B (Section 6)
    4. Apply post-hoc physics selection (Section 7)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Imports
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
    from __future__ import annotations

    import os
    import subprocess
    from pathlib import Path

    import wandb

    # Ensure we're in the repo root
    REPO_ROOT = Path(os.environ.get("JACOBIANODE_ROOT", ".." )).resolve()
    os.chdir(REPO_ROOT)
    print(f"Working directory: {os.getcwd()}")
    return Path, REPO_ROOT, os, subprocess


@app.cell
def _():
    # ----------------------------------------------------------------
    # Paths and W&B settings
    # ----------------------------------------------------------------
    SAVE_DIR     = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs"
    WANDB_ENTITY = "JacobianODE"
    return (SAVE_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Training Mode & Encoder Settings

    Set `MODE` to `"from_scratch"` or `"pretrained"`.
    """)
    return


@app.cell
def _():
    # ============================================================
    # Training mode
    # ============================================================
    MODE = "from_scratch"  # <-- CHANGE THIS

    assert MODE in ("from_scratch", "pretrained"), f"Invalid MODE: {MODE}"

    # Encoder architecture (from_scratch only)
    ENCODER_TYPE = "spline_coupling"  # "mlp" | "coupling" | "spline_coupling"

    print(f"Training mode: {MODE}, Encoder: {ENCODER_TYPE}")
    return ENCODER_TYPE, MODE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data Settings
    """)
    return


@app.cell
def _():
    # ============================================================
    # Data source
    # ============================================================
    DATA_SOURCE = "dysts"  # "dysts" | "wmtask"

    # ---- dysts settings ----
    FLOW_TARGET   = "JacobianODE.dysts_sim.flows.Lorenz"
    N_PERIODS     = 12
    PTS_PER_PERIOD = 100
    NUM_ICS       = 32
    OBS_NOISE     = 0.01
    NORMALIZE     = True
    # OBSERVED_INDICES = "all"  # list of ints or "all"
    OBSERVED_INDICES = [0]

    # ---- Delay embedding ----
    N_DELAYS      = 100
    DELAY_SPACING = 1

    # ---- wmtask settings (only if DATA_SOURCE == "wmtask") ----
    WMTASK_PROJECT = "WMSelectionTask__cue_time_0.1__response_time_0.25__enforce_fixation_False"
    WMTASK_NAME = "BiologicalRNN__cue_time_0.1__learning_rate_0.0005__max_epochs_42__N1_64__N2_64__tau_0.05__dt_0.02__eig_lower_bound_0.1__init_mode_random"
    WMTASK_MODEL_TO_LOAD = "final"
    WMTASK_DATALOADER = "all"
    WMTASK_TRAJ_WINDOW = "delay2"   # 'delay2' or 'full'
    WMTASK_DIM = 128                # N1 + N2 = 64 + 64
    return (
        DATA_SOURCE,
        DELAY_SPACING,
        FLOW_TARGET,
        NORMALIZE,
        NUM_ICS,
        N_DELAYS,
        N_PERIODS,
        OBSERVED_INDICES,
        OBS_NOISE,
        PTS_PER_PERIOD,
        WMTASK_DATALOADER,
        WMTASK_DIM,
        WMTASK_MODEL_TO_LOAD,
        WMTASK_NAME,
        WMTASK_PROJECT,
        WMTASK_TRAJ_WINDOW,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Encoder-Specific Settings (from_scratch)
    """)
    return


@app.cell
def _(ENCODER_TYPE, MODE):
    # ============================================================
    # From-scratch encoder architecture
    # ============================================================
    if MODE == "from_scratch":
        # Latent dimension
        RECONSTRUCTION_MODE      = "most_recent" # 'uniform' | 'harmonic' | 'most_recent'

        # Encoder warmup
        ENCODER_WARMUP_EPOCHS  = 5
        DYNAMICS_WARMUP_EPOCHS = 0

        # Learnable loss weights
        LEARN_R2_WEIGHT           = False
        LEARN_LOOP_CLOSURE_WEIGHT = False
        LEARN_FNN_WEIGHT          = False
        LEARN_JAC_CONS_WEIGHT     = False
        LEARN_JAC_NORM_WEIGHT     = False
        LOG_VAR_INIT              = 0.0

        DECODE_ONLY_RECENT = False

        if ENCODER_TYPE == "mlp":
            N_LATENT = 3
            MLP_HIDDEN_DIM = 128
            MLP_N_LAYERS   = 4
            MLP_DROPOUT    = 0.0

            # Decoder
            DECODER_HIDDEN     = 128
            DECODER_LAYERS     = 3

        elif ENCODER_TYPE in ("coupling", "spline_coupling"):
            N_LATENT = None  # computed from data dim in the config-building cell
            N_TARGET_DIMS       = 3  

            # ---- Architecture ----
            N_COUPLING_LAYERS   = 8
            COUPLING_HIDDEN_DIM = 128
            N_HIDDEN_LAYERS     = 2
            ZERO_INIT           = True   # identity initialization trick
            PERMUTATION_SEED    = 42     # base seed for fixed random permutations
            USE_LOFT            = False  # LOFT layer after coupling blocks
            LOFT_TAU            = 100.0  # LOFT threshold

            # ---- VAE settings ----
            USE_VAE                = True    # VAE reparameterization on dynamic subspace
            VAE_SAMPLE_ALL_LOSSES  = False    # when False, only recon uses sampled z_dyn
            KL_WARMUP_EPOCHS       = 0       # 0 = fixed weight, >0 = linear ramp
            KL_NULL_WEIGHT      = "null"
            KL_DYN_WEIGHT       = 0.0001  # default if not swept

            # ---- Tangent entropy loss ----
            TANGENT_ENTROPY_WEIGHT   = 0.0
            TANGENT_ENTROPY_MODE     = "quadratic"
            TANGENT_ENTROPY_N_SAMPLES = 1024

            if ENCODER_TYPE == "coupling":
                # Coupling-specific
                SCALE_ACTIVATION    = "tanh" # bounds log-scale via tanh(·)
                SCALE_CLAMP         = 3.0    # max |log_s|, ~20× scaling range
                # ---- Numerical stability (Andrade 2024, arXiv:2402.16408) ----
                CLAMP_TYPE          = "symmetric"  # 'symmetric' | 'asymmetric'
                ALPHA_POS           = 0.1          # asymmetric clamp: expansion bound
                ALPHA_NEG           = 2.0          # asymmetric clamp: compression bound

            # Spline-coupling-specific
            if ENCODER_TYPE == "spline_coupling":
                NUM_BINS    = 8
                TAIL_BOUND  = 3.0
                USE_ACTNORM = True
    return (
        ALPHA_NEG,
        ALPHA_POS,
        CLAMP_TYPE,
        COUPLING_HIDDEN_DIM,
        DECODER_HIDDEN,
        DECODER_LAYERS,
        DECODE_ONLY_RECENT,
        DYNAMICS_WARMUP_EPOCHS,
        ENCODER_WARMUP_EPOCHS,
        KL_NULL_WEIGHT,
        KL_WARMUP_EPOCHS,
        LEARN_FNN_WEIGHT,
        LEARN_JAC_CONS_WEIGHT,
        LEARN_JAC_NORM_WEIGHT,
        LEARN_LOOP_CLOSURE_WEIGHT,
        LEARN_R2_WEIGHT,
        LOFT_TAU,
        LOG_VAR_INIT,
        MLP_DROPOUT,
        MLP_HIDDEN_DIM,
        MLP_N_LAYERS,
        NUM_BINS,
        N_COUPLING_LAYERS,
        N_HIDDEN_LAYERS,
        N_LATENT,
        N_TARGET_DIMS,
        PERMUTATION_SEED,
        RECONSTRUCTION_MODE,
        SCALE_ACTIVATION,
        SCALE_CLAMP,
        TAIL_BOUND,
        TANGENT_ENTROPY_MODE,
        TANGENT_ENTROPY_N_SAMPLES,
        TANGENT_ENTROPY_WEIGHT,
        USE_ACTNORM,
        USE_LOFT,
        USE_VAE,
        VAE_SAMPLE_ALL_LOSSES,
        ZERO_INIT,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Optuna Search Space & Sweep Settings

    Define which hyperparameters Optuna should optimize and their ranges.

    **Search space types:**
    - `float` with `log: True` — log-uniform sampling (good for loss weights, learning rates)
    - `float` with `log: False` — uniform sampling
    - `int` — uniform integer sampling
    - `categorical` with `choices: [...]` — discrete choices
    """)
    return


@app.cell
def _(N_TARGET_DIMS):
    # ============================================================
    # Optuna sweep configuration
    # ============================================================

    # Number of trials and parallelism
    N_TRIALS         = 40    # total Optuna trials
    N_PARALLEL_JOBS  = 10     # concurrent GPU worker jobs
    N_STARTUP_TRIALS = 10    # random trials before TPE kicks in
    SAMPLER_SEED     = 42

    # ---- Early pruning ----
    # At OPTUNA_PRUNE_EPOCH, each worker reads the SQLite study and compares
    # its trajectory val_loss against completed trials.  If worse than the
    # configured quantile, training stops early (saving ~90% of that trial's
    # GPU time).  Set to None to disable.
    OPTUNA_PRUNE_EPOCH         = 15    # epoch to evaluate (None = no pruning)
    OPTUNA_PRUNE_MIN_COMPLETED = 5     # need this many finished trials first
    OPTUNA_PRUNE_QUANTILE      = 0.5   # prune if worse than median

    # ---- Constrained optimization ----
    # When enabled, trials that violate the constraint are considered infeasible.
    # A feasible result always beats an infeasible one (regardless of loss).
    # Set to None to disable (use unconstrained progress tracking).
    OPTUNA_CONSTRAINT_METRIC    = "val/loop_closure_loss"  # metric to check feasibility
    OPTUNA_CONSTRAINT_THRESHOLD = N_TARGET_DIMS ** 0.5     # sqrt(n_dynamic_dims)

    # ---- Search space ----
    # Keys must match Hydra config paths.
    # Each entry is passed to hydra.sweeper.search_space.
    SEARCH_SPACE = {
        "training.lightning.loop_closure_weight": {
            "type": "float", "low": 1e-7, "high": 1.0, "log": True,
        },
        "training.lightning.kl_dyn_weight": {
            "type": "float", "low": 1e-7, "high": 1.0, "log": True,
        },
        # ---- Uncomment to add more params to the search ----
        "training.lightning.optimizer_kwargs.lr": {
            "type": "float", "low": 1e-6, "high": 1e-3, "log": True,
        },
        # "training.lightning.fnn_weight": {
        #     "type": "float", "low": 0.0, "high": 1.0, "log": False,
        # },
        # "training.lightning.tangent_entropy_weight": {
        #     "type": "float", "low": 1e-4, "high": 1e-1, "log": True,
        # },
    }

    print(f"Optuna sweep: {N_TRIALS} trials, {N_PARALLEL_JOBS} parallel jobs")
    print(f"Pruning: epoch={OPTUNA_PRUNE_EPOCH}, quantile={OPTUNA_PRUNE_QUANTILE}, min_completed={OPTUNA_PRUNE_MIN_COMPLETED}")
    if OPTUNA_CONSTRAINT_METRIC:
        print(f"Constraint: {OPTUNA_CONSTRAINT_METRIC} <= {OPTUNA_CONSTRAINT_THRESHOLD:.4f}")
    else:
        print("Constraint: disabled (unconstrained)")
    print(f"Search space ({len(SEARCH_SPACE)} params):")
    for k, v in SEARCH_SPACE.items():
        print(f"  {k}: {v}")
    return (
        N_PARALLEL_JOBS,
        N_STARTUP_TRIALS,
        N_TRIALS,
        OPTUNA_CONSTRAINT_METRIC,
        OPTUNA_CONSTRAINT_THRESHOLD,
        OPTUNA_PRUNE_EPOCH,
        OPTUNA_PRUNE_MIN_COMPLETED,
        OPTUNA_PRUNE_QUANTILE,
        SAMPLER_SEED,
        SEARCH_SPACE,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Fixed Parameters & Overrides

    These parameters are held constant across all Optuna trials.
    """)
    return


@app.cell
def _(MODE, SEARCH_SPACE):
    # ============================================================
    # Fixed training params (not swept by Optuna)
    # ============================================================
    PREDICTION_STEPS = 30

    FIXED_PARAMS = {
        "model.prediction_steps":                                    PREDICTION_STEPS,
        "training.batch_size":                                       32 if MODE == "from_scratch" else 16,
        "training.trainer_params.max_epochs":                        150 if MODE == "from_scratch" else 200,
        "training.trainer_params.limit_train_batches":               200,
        "training.trainer_params.limit_val_batches":                 50,
        "training.trainer_params.accumulate_grad_batches":           4,
        "training.lightning.optimizer_kwargs.lr":                    1e-4,
        "training.lightning.optimizer_kwargs.weight_decay":          1e-4,
        "training.lightning.scheduler_type":                         "cosine",
        "training.lightning.reconstruction_loss_weight":             1.0,
        "training.lightning.latent_prediction_loss_weight":          1.0,
        "training.lightning.jac_consistency_weight":                 0.0,
        "training.lightning.fnn_normalize":                          True,
        "training.lightning.fnn_elementwise_regularization":         True,
        "training.lightning.fnn_use_pca":                            True,
        "training.lightning.fnn_n_samples":                          1024,
        "training.lightning.jacobianODEint_kwargs.traj_init_steps":  15,
        "training.lightning.jacobianODEint_kwargs.interp_pts":       4,
        "training.lightning.jacobianODEint_kwargs.inner_N":          20,
        "training.lightning.jacobianODEint_kwargs.inner_path":       "line",
        "training.early_stopping.early_stopping_patience":           5,
        "training.early_stopping.min_epochs":                        10,
        "training.early_stopping.percent_thresh":                    0.01,
        "training.model_checkpoint.save_top_k":                      1,
    }

    # Remove swept params from FIXED_PARAMS if they overlap
    for key in SEARCH_SPACE:
        FIXED_PARAMS.pop(key, None)

    print(f"{len(FIXED_PARAMS)} fixed params, {len(SEARCH_SPACE)} swept params")
    return FIXED_PARAMS, PREDICTION_STEPS


@app.cell
def _(
    DATA_SOURCE,
    DELAY_SPACING,
    ENCODER_TYPE,
    ENCODER_WARMUP_EPOCHS,
    MODE,
    NORMALIZE,
    N_DELAYS,
    N_LATENT,
    N_TARGET_DIMS,
    N_TRIALS,
    OBSERVED_INDICES,
    Path,
    SAVE_DIR,
    SEARCH_SPACE,
):
    # ============================================================
    # W&B settings
    WANDB_ENTITY_1 = 'JacobianODE'
    _obs_idx_label = ''.join((str(i) for i in OBSERVED_INDICES)) if OBSERVED_INDICES != 'all' else 'all'
    _data_prefix = 'Lorenz' if DATA_SOURCE == 'dysts' else 'WMTask'
    # Auto-generate project name (matches grid sweep convention)
    _warmup = ENCODER_WARMUP_EPOCHS if MODE == 'from_scratch' else 'pt'
    if MODE == 'from_scratch':
        if ENCODER_TYPE in ('coupling', 'spline_coupling'):
            _latent_label = f'T{N_TARGET_DIMS}'
            _enc_label = 'spline_coupling' if ENCODER_TYPE == 'spline_coupling' else 'coupling'
        else:
            _latent_label = f'L{N_LATENT}'
            _enc_label = 'MLP'
        WANDB_PROJECT = f'{_data_prefix}_IND{_obs_idx_label}_N{N_DELAYS}_D{DELAY_SPACING}_Norm{NORMALIZE}_{_latent_label}__{_enc_label}__JacobianODE'
    else:
        WANDB_PROJECT = f'{_data_prefix}_Pretrained_L{N_LATENT}__JacobianODE'
    WANDB_PROJECT_PATH = f'{WANDB_ENTITY_1}/{WANDB_PROJECT}'
    from datetime import datetime
    _datetime_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _swept_keys_short = '_'.join((k.split('.')[-1] for k in SEARCH_SPACE))
    WANDB_GROUP = f'optuna_{MODE}_{_enc_label}_{_swept_keys_short}_n{N_TRIALS}_{_datetime_tag}'
    STUDY_NAME = WANDB_GROUP
    # W&B group name with datetime tag
    OPTUNA_SAVE_DIR = Path(SAVE_DIR) / 'optuna'
    OPTUNA_STORAGE = f'sqlite:///{OPTUNA_SAVE_DIR}/optuna_{STUDY_NAME}.db'
    print(f'W&B project:    {WANDB_PROJECT_PATH}')
    print(f'W&B group:      {WANDB_GROUP}')
    print(f'Optuna study:   {STUDY_NAME}')
    # Auto-generated study name (matches W&B group)
    # Study persistence — SQLite DB so the sweep survives coordinator restarts
    # and completed trials are never lost. Set to None only for quick local tests.
    print(f'Optuna storage: {OPTUNA_STORAGE}')
    return (
        OPTUNA_SAVE_DIR,
        OPTUNA_STORAGE,
        STUDY_NAME,
        WANDB_ENTITY_1,
        WANDB_GROUP,
        WANDB_PROJECT,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Build & Preview Command
    """)
    return


@app.cell
def _(
    ALPHA_NEG,
    ALPHA_POS,
    CLAMP_TYPE,
    COUPLING_HIDDEN_DIM,
    DATA_SOURCE,
    DECODER_HIDDEN,
    DECODER_LAYERS,
    DECODE_ONLY_RECENT,
    DELAY_SPACING,
    DYNAMICS_WARMUP_EPOCHS,
    ENCODER_TYPE,
    ENCODER_WARMUP_EPOCHS,
    FIXED_PARAMS,
    FLOW_TARGET,
    KL_NULL_WEIGHT,
    KL_WARMUP_EPOCHS,
    LEARN_FNN_WEIGHT,
    LEARN_JAC_CONS_WEIGHT,
    LEARN_JAC_NORM_WEIGHT,
    LEARN_LOOP_CLOSURE_WEIGHT,
    LEARN_R2_WEIGHT,
    LOFT_TAU,
    LOG_VAR_INIT,
    MLP_DROPOUT,
    MLP_HIDDEN_DIM,
    MLP_N_LAYERS,
    MODE,
    NORMALIZE,
    NUM_BINS,
    NUM_ICS,
    N_COUPLING_LAYERS,
    N_DELAYS,
    N_HIDDEN_LAYERS,
    N_LATENT,
    N_PERIODS,
    N_TARGET_DIMS,
    OBSERVED_INDICES,
    OBS_NOISE,
    OPTUNA_CONSTRAINT_METRIC,
    OPTUNA_CONSTRAINT_THRESHOLD,
    OPTUNA_PRUNE_EPOCH,
    OPTUNA_PRUNE_MIN_COMPLETED,
    OPTUNA_PRUNE_QUANTILE,
    OPTUNA_STORAGE,
    PERMUTATION_SEED,
    PREDICTION_STEPS,
    PTS_PER_PERIOD,
    RECONSTRUCTION_MODE,
    SAVE_DIR,
    SCALE_ACTIVATION,
    SCALE_CLAMP,
    STUDY_NAME,
    TAIL_BOUND,
    TANGENT_ENTROPY_MODE,
    TANGENT_ENTROPY_N_SAMPLES,
    TANGENT_ENTROPY_WEIGHT,
    USE_ACTNORM,
    USE_LOFT,
    USE_VAE,
    VAE_SAMPLE_ALL_LOSSES,
    WANDB_ENTITY_1,
    WANDB_GROUP,
    WANDB_PROJECT,
    WMTASK_DATALOADER,
    WMTASK_DIM,
    WMTASK_MODEL_TO_LOAD,
    WMTASK_NAME,
    WMTASK_PROJECT,
    WMTASK_TRAJ_WINDOW,
    ZERO_INIT,
):
    # ============================================================
    # Build overrides
    def _fmt(v):
        """Format a value for Hydra CLI override."""
        if isinstance(v, bool):
            return str(v).lower()
        if v is None:
            return 'null'
        return str(v)
    overrides = [f'{k}={_fmt(v)}' for k, v in FIXED_PARAMS.items()]
    JAC_HIDDEN_DIMS = [128, 128]
    JAC_NUM_LAYERS = 2
    JAC_ACTIVATION = 'silu'
    # Fixed param overrides
    JAC_WINDOW_STRIDE = 5 if MODE == 'from_scratch' else PREDICTION_STEPS
    _hidden_dim_str = '[' + ','.join((str(d) for d in JAC_HIDDEN_DIMS)) + ']'
    # Jacobian MLP settings
    overrides.append(f'model.params.hidden_dim={_hidden_dim_str}')
    overrides.append(f'model.params.num_layers={JAC_NUM_LAYERS}')
    overrides.append(f'model.params.activation={JAC_ACTIVATION}')
    TRAJ_INIT_STEPS = FIXED_PARAMS.get('training.lightning.jacobianODEint_kwargs.traj_init_steps', 15)
    # No quotes around list — coordinator uses single-run mode, not --multirun
    JAC_WINDOW = TRAJ_INIT_STEPS + PREDICTION_STEPS
    SEQ_LENGTH = JAC_WINDOW
    if DATA_SOURCE == 'dysts':
        _obs_idx_str = '[' + ','.join((str(i) for i in OBSERVED_INDICES)) + ']' if OBSERVED_INDICES != 'all' else 'all'
        overrides = overrides + ['data=dysts', f'data.flow._target_={FLOW_TARGET}', f'data.trajectory_params.n_periods={N_PERIODS}', f'data.trajectory_params.pts_per_period={PTS_PER_PERIOD}', f'data.trajectory_params.num_ics={NUM_ICS}', f'data.postprocessing.obs_noise={OBS_NOISE}', f'data.postprocessing.normalize={NORMALIZE}', f'data.train_test_params.delay_embedding_params.observed_indices={_obs_idx_str}', f'data.train_test_params.delay_embedding_params.n_delays={N_DELAYS}', f'data.train_test_params.delay_embedding_params.delay_spacing={DELAY_SPACING}', f'data.train_test_params.seq_length={SEQ_LENGTH}']
    # Sequence length (matches grid sweep)
    elif DATA_SOURCE == 'wmtask':
        overrides = overrides + ['data=wmtask', f'data.dataset_loader.project={WMTASK_PROJECT}', f'data.dataset_loader.name={WMTASK_NAME}', f'data.dataset_loader.model_to_load={WMTASK_MODEL_TO_LOAD}', f'data.dataset_loader.dataloader_to_use={WMTASK_DATALOADER}', f'data.dataset_loader.traj_window={WMTASK_TRAJ_WINDOW}', f'data.flow.dim={WMTASK_DIM}']
    if MODE == 'from_scratch':
        _n_raw_obs = len(OBSERVED_INDICES) if OBSERVED_INDICES != 'all' else 3 if DATA_SOURCE == 'dysts' else WMTASK_DIM
        _n_input = N_DELAYS * _n_raw_obs
        if ENCODER_TYPE == 'mlp':
    # ---- Data overrides ----
            overrides = overrides + ['model=latent_mlp', f'model.encoder.n_input={_n_input}', f'model.encoder.n_latent={N_LATENT}', f'model.encoder.hidden_dim={MLP_HIDDEN_DIM}', f'model.encoder.n_layers={MLP_N_LAYERS}', f'model.encoder.dropout={MLP_DROPOUT}', f'model.encoder.decoder_hidden={DECODER_HIDDEN}', f'model.encoder.decoder_layers={DECODER_LAYERS}', f'model.encoder.decoder_n_output={(_n_raw_obs if DECODE_ONLY_RECENT else 'null')}', 'model.encoder.context_margin=0']
        elif ENCODER_TYPE == 'coupling':  # No quotes around list — coordinator uses single-run mode, not --multirun
            overrides = overrides + ['model=latent_coupling', f'model.encoder.n_input={_n_input}', f'model.encoder.n_coupling_layers={N_COUPLING_LAYERS}', f'model.encoder.hidden_dim={COUPLING_HIDDEN_DIM}', f'model.encoder.n_hidden_layers={N_HIDDEN_LAYERS}', f'model.encoder.scale_activation={SCALE_ACTIVATION}', f'model.encoder.scale_clamp={SCALE_CLAMP}', f'model.encoder.zero_init={_fmt(ZERO_INIT)}', f'model.encoder.permutation_seed={PERMUTATION_SEED}', f'model.encoder.clamp_type={CLAMP_TYPE}', f'model.encoder.alpha_pos={ALPHA_POS}', f'model.encoder.alpha_neg={ALPHA_NEG}', f'model.encoder.use_loft={_fmt(USE_LOFT)}', f'model.encoder.loft_tau={LOFT_TAU}', f'model.n_target_dims={_fmt(N_TARGET_DIMS)}', f'model.use_vae={_fmt(USE_VAE)}', f'model.vae_sample_all_losses={_fmt(VAE_SAMPLE_ALL_LOSSES)}', f'model.kl_warmup_epochs={KL_WARMUP_EPOCHS}', f'training.lightning.kl_null_weight={KL_NULL_WEIGHT}', f'training.lightning.reconstruction_mode={RECONSTRUCTION_MODE}', f'training.lightning.tangent_entropy_weight={TANGENT_ENTROPY_WEIGHT}', f'training.lightning.tangent_entropy_mode={TANGENT_ENTROPY_MODE}', f'training.lightning.tangent_entropy_n_samples={TANGENT_ENTROPY_N_SAMPLES}']
        elif ENCODER_TYPE == 'spline_coupling':
            overrides = overrides + ['model=latent_spline_coupling', f'model.encoder.n_input={_n_input}', f'model.encoder.n_coupling_layers={N_COUPLING_LAYERS}', f'model.encoder.hidden_dim={COUPLING_HIDDEN_DIM}', f'model.encoder.n_hidden_layers={N_HIDDEN_LAYERS}', f'model.encoder.num_bins={NUM_BINS}', f'model.encoder.tail_bound={TAIL_BOUND}', f'model.encoder.use_actnorm={_fmt(USE_ACTNORM)}', f'model.encoder.zero_init={_fmt(ZERO_INIT)}', f'model.encoder.permutation_seed={PERMUTATION_SEED}', f'model.encoder.use_loft={_fmt(USE_LOFT)}', f'model.encoder.loft_tau={LOFT_TAU}', f'model.n_target_dims={_fmt(N_TARGET_DIMS)}', f'model.use_vae={_fmt(USE_VAE)}', f'model.vae_sample_all_losses={_fmt(VAE_SAMPLE_ALL_LOSSES)}', f'model.kl_warmup_epochs={KL_WARMUP_EPOCHS}', f'training.lightning.kl_null_weight={KL_NULL_WEIGHT}', f'training.lightning.reconstruction_mode={RECONSTRUCTION_MODE}', f'training.lightning.tangent_entropy_weight={TANGENT_ENTROPY_WEIGHT}', f'training.lightning.tangent_entropy_mode={TANGENT_ENTROPY_MODE}', f'training.lightning.tangent_entropy_n_samples={TANGENT_ENTROPY_N_SAMPLES}']
        overrides = overrides + [f'model.decode_only_recent={_fmt(DECODE_ONLY_RECENT)}', f'model.jac_window_stride={JAC_WINDOW_STRIDE}', f'model.encoder_warmup_epochs={ENCODER_WARMUP_EPOCHS}', f'model.dynamics_warmup_epochs={DYNAMICS_WARMUP_EPOCHS}', f'training.lightning.learn_r2_weight={_fmt(LEARN_R2_WEIGHT)}', f'training.lightning.learn_loop_closure_weight={_fmt(LEARN_LOOP_CLOSURE_WEIGHT)}', f'training.lightning.learn_fnn_weight={_fmt(LEARN_FNN_WEIGHT)}', f'training.lightning.learn_jac_cons_weight={_fmt(LEARN_JAC_CONS_WEIGHT)}', f'training.lightning.learn_jac_norm_weight={_fmt(LEARN_JAC_NORM_WEIGHT)}', f'training.lightning.log_var_init={LOG_VAR_INIT}', f'training.logger.save_dir={SAVE_DIR}']
    if OPTUNA_PRUNE_EPOCH is not None:
        overrides = overrides + [f'training.optuna_prune_epoch={OPTUNA_PRUNE_EPOCH}', f'training.optuna_prune_min_completed={OPTUNA_PRUNE_MIN_COMPLETED}', f'training.optuna_prune_quantile={OPTUNA_PRUNE_QUANTILE}', f'optuna_study_name={STUDY_NAME}', f'optuna_storage={OPTUNA_STORAGE}']
    if OPTUNA_CONSTRAINT_METRIC is not None:
        overrides = overrides + [f'training.optuna_constraint_metric={OPTUNA_CONSTRAINT_METRIC}', f'training.optuna_constraint_threshold={OPTUNA_CONSTRAINT_THRESHOLD}']
    overrides = overrides + [f'wandb_entity={WANDB_ENTITY_1}', f'wandb_project={WANDB_PROJECT}', f'wandb_group={WANDB_GROUP}', 'slurm=default']
    # ---- Encoder overrides (from_scratch) ----
    # ---- Optuna pruning ----
    # ---- Optuna constrained optimization ----
    # ---- W&B + SLURM ----
    print(f'Total fixed overrides: {len(overrides)}')  # Compute n_input (n_delays × n_raw_obs)  # Common from-scratch overrides
    return (overrides,)


@app.cell
def _(
    MODE,
    N_PARALLEL_JOBS,
    N_STARTUP_TRIALS,
    N_TRIALS,
    OPTUNA_CONSTRAINT_METRIC,
    OPTUNA_CONSTRAINT_THRESHOLD,
    OPTUNA_SAVE_DIR,
    OPTUNA_STORAGE,
    REPO_ROOT,
    SAMPLER_SEED,
    SEARCH_SPACE,
    STUDY_NAME,
    os,
    overrides,
):
    # ============================================================
    # Build the coordinator Python script
    ENTRY_POINT = 'python -m JacobianODE.jacobians.run_jacobians' if MODE == 'from_scratch' else 'python -m JacobianODE.jacobians.run_pretrained_jacobians'
    #
    # Instead of Hydra --multirun (which batches jobs and blocks),
    # we use OptunaCoordinator for true backfilling: as soon as one
    # SLURM job finishes, a new trial is immediately queued.
    _coordinator_py = f'\nimport logging, sys\nlogging.basicConfig(\n    level=logging.INFO,\n    format="[%(asctime)s] %(message)s",\n    datefmt="%Y-%m-%d %H:%M:%S",\n    stream=sys.stdout,\n)\n\nfrom JacobianODE.jacobians.training.optuna_coordinator import OptunaCoordinator\n\ncoordinator = OptunaCoordinator(\n    fixed_overrides={overrides!r},\n    search_space={SEARCH_SPACE!r},\n    study_name="{STUDY_NAME}",\n    storage="{OPTUNA_STORAGE}",\n    entry_point="{ENTRY_POINT}",\n    repo_root="{REPO_ROOT}",\n    sampler_seed={SAMPLER_SEED},\n    n_startup_trials={N_STARTUP_TRIALS},\n    # SLURM worker params\n    slurm_partition="ou_bcs_normal",\n    slurm_gpus_per_node=1,\n    slurm_cpus_per_task=4,\n    slurm_mem_gb=16,\n    slurm_timeout_min=180,\n    slurm_exclude="node4000",\n    poll_interval=30.0,\n    # Constraint\n    constraint_metric={(f'"{OPTUNA_CONSTRAINT_METRIC}"' if OPTUNA_CONSTRAINT_METRIC else 'None')},\n    constraint_threshold={(OPTUNA_CONSTRAINT_THRESHOLD if OPTUNA_CONSTRAINT_METRIC else 'None')},\n)\n\ncoordinator.run(n_trials={N_TRIALS}, n_parallel={N_PARALLEL_JOBS})\n'
    _coordinator_py_path = OPTUNA_SAVE_DIR / f'coordinator_{STUDY_NAME}.py'
    os.makedirs(OPTUNA_SAVE_DIR, exist_ok=True)
    with open(_coordinator_py_path, 'w') as _f:
        _f.write(_coordinator_py)
    print(f'Coordinator script: {_coordinator_py_path}')
    print(f'\nWill run {N_TRIALS} trials, {N_PARALLEL_JOBS} parallel (true backfilling)')
    # Build the coordinator Python code (will be written to disk and
    # executed by the SLURM coordinator job).
    print(f'Entry point: {ENTRY_POINT}')
    print(f'Search space: {list(SEARCH_SPACE.keys())}')
    if OPTUNA_CONSTRAINT_METRIC:
        print(f'Constraint: {OPTUNA_CONSTRAINT_METRIC} <= {OPTUNA_CONSTRAINT_THRESHOLD:.4f}')
    print('to change to optuna directory:')
    # Write to disk
    print(f'cd {OPTUNA_SAVE_DIR}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Launch Sweep

    The notebook submits a **coordinator SLURM job** (CPU-only, 24 h) that:
    1. Uses `OptunaCoordinator` with **true backfilling** — as soon as one GPU job finishes, a new trial is immediately queued
    2. Maintains `N_PARALLEL_JOBS` concurrent GPU workers at all times
    3. Reads results from the shared SQLite DB (written by training callbacks)
    4. Supports resuming — re-run with the same `STUDY_NAME` to continue
    """)
    return


@app.cell
def _(OPTUNA_SAVE_DIR, REPO_ROOT, STUDY_NAME):
    # ============================================================
    # Build coordinator SLURM script
    # ============================================================
    COORDINATOR_TIME   = "24:00:00"   # 24 h — enough for N_TRIALS × 3 h / N_PARALLEL_JOBS
    COORDINATOR_PARTITION = "ou_bcs_normal"
    COORDINATOR_MEM    = "8G"
    COORDINATOR_CPUS   = 2

    _script_path = OPTUNA_SAVE_DIR / f"optuna_coordinator_{STUDY_NAME}.sh"
    _log_path    = OPTUNA_SAVE_DIR / f"optuna_coordinator_{STUDY_NAME}_%j.log"

    coordinator_script = f"""#!/bin/bash
    #SBATCH --job-name=optuna_{STUDY_NAME[:30]}
    #SBATCH --partition={COORDINATOR_PARTITION}
    #SBATCH --time={COORDINATOR_TIME}
    #SBATCH --mem={COORDINATOR_MEM}
    #SBATCH --cpus-per-task={COORDINATOR_CPUS}
    #SBATCH --gpus=0
    #SBATCH --output={_log_path}

    # ---- Environment setup ----
    cd {REPO_ROOT}
    source .venv/bin/activate

    # ---- Run Optuna coordinator (submits GPU worker jobs via submitit) ----
    python {_coordinator_py_path}
    """

    print(f"Coordinator script: {_script_path}")
    print(f"Log file:           {_log_path}")
    print(f"\n--- Script preview ---")
    print(coordinator_script)
    return (coordinator_script,)


@app.cell
def _(Path, coordinator_script, os, subprocess):
    # ============================================================
    # Submit coordinator job
    DRY_RUN = False
    # DRY_RUN = True  # <-- Set to False to actually submit
    os.makedirs(Path(_script_path).parent, exist_ok=True)
    with open(_script_path, 'w') as _f:
    # Write script to disk
        _f.write(coordinator_script)
    os.chmod(_script_path, 493)
    if DRY_RUN:
        print('DRY RUN — set DRY_RUN = False to submit.')
        print(f'\nTo submit manually:\n  sbatch {_script_path}')
    else:
        result = subprocess.run(['sbatch', str(_script_path)], capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            print(f'ERROR: {result.stderr.strip()}')
        else:
            print(f'\nCoordinator submitted. Monitor with:')
            print(f'  squeue -u $USER')
            print(f'  tail -f {str(_log_path).replace('%j', '*')}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Analyze Optuna Results

    After the sweep completes, load the Optuna study to inspect trial results,
    parameter importances, and optimization history.
    """)
    return


@app.cell
def _():
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
    )

    return (
        optuna,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    )


@app.cell
def _(OPTUNA_STORAGE, STUDY_NAME, optuna):
    # Load study (requires OPTUNA_STORAGE to have been set, or find the auto-generated db)
    if OPTUNA_STORAGE is not None:
        study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_STORAGE)
    else:
        # Without persistent storage, results are only in W&B.
        # Use the W&B-based analysis in the next section instead.
        print("No persistent storage configured — skipping Optuna study analysis.")
        print("Set OPTUNA_STORAGE = 'sqlite:///optuna_sweep.db' for study persistence.")
        study = None
    return (study,)


@app.cell
def _(display, study):
    if study is not None:
        print(f"Study: {study.study_name}")
        print(f"Completed trials: {len(study.trials)}")
        print(f"Best trial:")
        print(f"  Value (trajectory val_loss): {study.best_value:.6f}")
        print(f"  Params: {study.best_trial.params}")
        print(f"\nTop 5 trials:")
        df = study.trials_dataframe().sort_values("value")
        display(df.head())
    return


@app.cell
def _(plot_optimization_history, study):
    if study is not None:
        # Optimization history: shows convergence over trials
        plot_optimization_history(study)
    return


@app.cell
def _(plot_param_importances, study):
    if study is not None:
        # Parameter importances: which params matter most
        plot_param_importances(study)
    return


@app.cell
def _(plot_slice, study):
    if study is not None:
        # Slice plot: objective vs each param
        plot_slice(study)
    return


@app.cell
def _(plot_parallel_coordinate, study):
    if study is not None:
        # Parallel coordinate plot: see param interactions
        plot_parallel_coordinate(study)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Post-Hoc Physics Selection

    Apply the standard physics-based criteria (C1–C3) to the Optuna-optimized runs.
    This reuses the existing selection pipeline from `JacobianODE.jacobians.tuning`.
    """)
    return


@app.cell
def _():
    from JacobianODE.jacobians.tuning import select_best_from_sweep

    return (select_best_from_sweep,)


@app.cell
def _(WANDB_ENTITY_1, WANDB_GROUP, WANDB_PROJECT, select_best_from_sweep):
    best_run_id, sweep_result, discovered = select_best_from_sweep(wandb_entity=WANDB_ENTITY_1, wandb_project=WANDB_PROJECT, save_dir='./sweep_results', wandb_group=WANDB_GROUP, verbose=True)
    if best_run_id:
        print(f'\nBest run: {best_run_id}')
    else:
        print('\nNo runs passed all physics criteria.')
    return


if __name__ == "__main__":
    app.run()
