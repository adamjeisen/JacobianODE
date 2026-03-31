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
    # Encoder-Only (Pointwise MLP) — Hyperparameter Sweep

    Performs a **Hydra grid sweep** over encoder-only regularisation parameters
    using SLURM for parallel training.

    **Supports two data sources** (set `DATA_SOURCE` in Section 1):
    - `"dysts"` — synthetic trajectories from dynamical systems (e.g. Lorenz)
    - `"wmtask"` — hidden-state trajectories from a trained working memory RNN

    **Key difference from the SSM/Transformer sweep:** The encoder is a **pointwise
    MLP** — it applies the same MLP independently at each timestep with **no
    temporal context**. All temporal information comes from the delay embedding
    in the input. This avoids the double-smoothing problem where an SSM's
    recurrent state integration compounds with the delay embedding's implicit
    temporal averaging, compressing the Lyapunov spectrum toward zero.

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
        np,
        os,
        pd,
        plt,
        postprocess_data,
        seed_everything,
        subprocess,
        torch,
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
    return SAVE_DIR, WANDB_ENTITY


@app.cell
def _(np):
    # ----------------------------------------------------------------
    # Data source: "dysts" or "wmtask"
    # ----------------------------------------------------------------
    # DATA_SOURCE = "wmtask"   # <-- change to "wmtask" for WM task RNN data
    DATA_SOURCE = "dysts"

    # ----------------------------------------------------------------
    # dysts-specific hyperparameters (ignored when DATA_SOURCE="wmtask")
    # ----------------------------------------------------------------
    NUM_ICS = 32
    N_PERIODS = 12
    PTS_PER_PERIOD = 100

    # ----------------------------------------------------------------
    # wmtask-specific hyperparameters (ignored when DATA_SOURCE="dysts")
    # ----------------------------------------------------------------
    WMTASK_PROJECT = "WMSelectionTask__cue_time_0.1__response_time_0.25__enforce_fixation_False"
    WMTASK_NAME = "BiologicalRNN__cue_time_0.1__learning_rate_0.0005__max_epochs_42__N1_64__N2_64__tau_0.05__dt_0.02__eig_lower_bound_0.1__init_mode_random"
    WMTASK_MODEL_TO_LOAD = "final"
    WMTASK_DATALOADER = "all"
    WMTASK_TRAJ_WINDOW = "delay2"   # 'delay2' or 'full'
    WMTASK_DIM = 128                # N1 + N2 = 64 + 64

    # ----------------------------------------------------------------
    # Shared data hyperparameters
    # ----------------------------------------------------------------

    OBS_NOISE = 0.01
    NORMALIZE = True

    # Partial observation (delay embedding)
    # The MLP encoder has NO temporal context — all history comes from the
    # delay embedding.  We use a generous embedding (43 delays, spacing=1)
    # to ensure Takens-style reconstruction of the attractor geometry.
    # OBSERVED_INDICES = [0]
    # N_DELAYS = 43
    # # DELAY_SPACING = 1
    # N_DELAYS = 5
    # DELAY_SPACING = 10

    # OBSERVED_INDICES = [0, 1, 2]
    # N_DELAYS = 1
    # DELAY_SPACING = 1

    USE_PARTIAL_OBS = True
    N_PARTIAL_OBS = 42
    PARTIAL_OBS_SEED = 42
    if DATA_SOURCE == "dysts":
        # OBSERVED_INDICES = [0, 1 ,2]
        OBSERVED_INDICES = [0]
        # N_DELAYS = 25
        N_DELAYS = 100
        # N_DELAYS = 11
        # N_DELAYS = 70
        # N_DELAYS = 1
        DELAY_SPACING = 1
        # N_DELAYS = 4
        # DELAY_SPACING = 11
        SEQ_LENGTH = 100
    else:
        # wmtask: observe all hidden dims by default; set to a list for partial obs
        # OBSERVED_INDICES = 'all'
        if USE_PARTIAL_OBS:
            np.random.seed(PARTIAL_OBS_SEED)
            OBSERVED_INDICES = sorted(np.random.choice(WMTASK_DIM, N_PARTIAL_OBS, replace=False).tolist())
        else:
            OBSERVED_INDICES = 'all'
        N_DELAYS = 1
        DELAY_SPACING = 1
        SEQ_LENGTH = 49
    return (
        DATA_SOURCE,
        DELAY_SPACING,
        NORMALIZE,
        NUM_ICS,
        N_DELAYS,
        N_PARTIAL_OBS,
        N_PERIODS,
        OBSERVED_INDICES,
        OBS_NOISE,
        PARTIAL_OBS_SEED,
        PTS_PER_PERIOD,
        SEQ_LENGTH,
        USE_PARTIAL_OBS,
        WMTASK_DATALOADER,
        WMTASK_DIM,
        WMTASK_MODEL_TO_LOAD,
        WMTASK_NAME,
        WMTASK_PROJECT,
        WMTASK_TRAJ_WINDOW,
    )


@app.cell
def _(DATA_SOURCE, OBSERVED_INDICES, WMTASK_DIM):
    # ----------------------------------------------------------------
    # Encoder / architecture hyperparameters — Pointwise MLP
    # ----------------------------------------------------------------
    MODEL = 'mlp'   # pointwise MLP (no temporal context)

    if DATA_SOURCE == "dysts":
        # N_LATENT = 3
        # N_LATENT = 4
        # N_LATENT = 7
        # N_LATENT = 6
        # N_LATENT = 25
        N_LATENT = 100
        # N_LATENT = 11
    else:
        # N_LATENT = 128
        N_LATENT = 13

    # MLP parameters
    MLP_HIDDEN_DIM = 256
    MLP_N_LAYERS   = 3
    MLP_DROPOUT    = 0.1

    # Decoder head (shared across architectures)
    # USE_SAME_STATE_DECODER = True
    USE_SAME_STATE_DECODER = False
    USE_NEXT_STATE_DECODER = True
    # USE_NEXT_STATE_DECODER = False
    DECODER_HIDDEN_DIM    = 128
    DECODER_N_LAYERS      = 2
    NEXT_STATE_BURN_IN    = 0

    # Multi-step ahead prediction:
    # K_STEPS_AHEAD=k means the next-state decoder jointly predicts [x_{t+1}, ..., x_{t+k}],
    # which forces the latent to encode the full dynamical state rather than collapsing.
    K_STEPS_AHEAD = 10

    # Raw observation dimension for next-state targets.
    # When delay embedding is active (N_DELAYS > 1), set this to len(OBSERVED_INDICES)
    # so that targets are un-embedded observations: [x_{t+1}, ..., x_{t+k}] without redundancy.
    if OBSERVED_INDICES == 'all':
        N_OBS_PRED = WMTASK_DIM  # full hidden state
    else:
        N_OBS_PRED = len(OBSERVED_INDICES)  # = raw obs dim, independent of N_DELAYS
    return (
        DECODER_HIDDEN_DIM,
        DECODER_N_LAYERS,
        K_STEPS_AHEAD,
        MLP_DROPOUT,
        MLP_HIDDEN_DIM,
        MLP_N_LAYERS,
        MODEL,
        NEXT_STATE_BURN_IN,
        N_LATENT,
        N_OBS_PRED,
        USE_NEXT_STATE_DECODER,
        USE_SAME_STATE_DECODER,
    )


@app.cell
def _(
    DATA_SOURCE,
    DELAY_SPACING,
    NORMALIZE,
    N_DELAYS,
    N_PARTIAL_OBS,
    OBSERVED_INDICES,
    PARTIAL_OBS_SEED,
    USE_PARTIAL_OBS,
    WANDB_ENTITY,
):
    if DATA_SOURCE == 'dysts':
        _obs_label = ''.join((str(_i) for _i in OBSERVED_INDICES))
        WANDB_PROJECT = f'Lorenz_IND{_obs_label}_N{N_DELAYS}_D{DELAY_SPACING}_Norm{NORMALIZE}__EncoderOnly'  # WANDB_PROJECT = f"Lorenz_IND{_obs_label}_N{N_DELAYS}_D{DELAY_SPACING}_Norm{NORMALIZE}_L{N_LATENT}__EncoderOnly"
    elif DATA_SOURCE == 'wmtask':
        if OBSERVED_INDICES == 'all':
            _obs_label = 'all'
        elif USE_PARTIAL_OBS:
            _obs_label = f'Partial_{N_PARTIAL_OBS}_seed{PARTIAL_OBS_SEED}'
        else:
            _obs_label = ''.join((str(_i) for _i in OBSERVED_INDICES))  # _obs_label = ''.join(str(i) for i in OBSERVED_INDICES)
        WANDB_PROJECT = f'WMTask_IND{_obs_label}_N{N_DELAYS}_D{DELAY_SPACING}_Norm{NORMALIZE}__EncoderOnly'
    WANDB_PROJECT_PATH = f'{WANDB_ENTITY}/{WANDB_PROJECT}'  # WANDB_PROJECT = f"WMTask_IND{_obs_label}_N{N_DELAYS}_D{DELAY_SPACING}_Norm{NORMALIZE}_L{N_LATENT}__EncoderOnly"
    return (WANDB_PROJECT,)


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

    FNN_USE_PCA = True
    FNN_N_SAMPLES = 1024
    FNN_NORMALIZE = True
    FNN_ELEMENTWISE_REGULARIZATION = True
    # FNN_SPARSIFY = False
    FNN_SPARSIFY = True

    JACOBIAN_NUCLEAR_WEIGHT = 0.0   # Nuclear norm of encoder Jacobian (mean singular values)
    # JACOBIAN_NUCLEAR_N_SAMPLES = 256  # Max points for Jacobian loss; null in config = use all
    JACOBIAN_NUCLEAR_N_SAMPLES = None

    TANGENT_ENTROPY_N_SAMPLES = 1024  # Max points for tangent entropy Jacobian computation
    # TANGENT_ENTROPY_N_SAMPLES = 128
    TANGENT_ENTROPY_MODE = "quadratic"  # 'shannon', 'quadratic', or 'renyi_half'
    return (
        BATCH_SIZE,
        EARLY_STOPPING_PATIENCE,
        FNN_ELEMENTWISE_REGULARIZATION,
        FNN_NORMALIZE,
        FNN_N_SAMPLES,
        FNN_SPARSIFY,
        FNN_USE_PCA,
        JACOBIAN_NUCLEAR_N_SAMPLES,
        LEARNING_RATE,
        LIMIT_TRAIN_BATCHES,
        MAX_EPOCHS,
        PERCENT_THRESH,
        TANGENT_ENTROPY_MODE,
        TANGENT_ENTROPY_N_SAMPLES,
    )


@app.cell
def _():
    SWEEP_PARAMS = {'training.lightning.fnn_weight': [0.0], 'training.lightning.amplification_weight': [1.0], 'training.lightning.decov_weight': [0.0], 'training.lightning.jacobian_nuclear_weight': [0.0], 'training.lightning.tangent_entropy_weight': [0.0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
    n_combos = 1
    for _vals in SWEEP_PARAMS.values():
        n_combos = n_combos * len(_vals)
    print(f'Sweep grid: {n_combos} total combinations')
    return SWEEP_PARAMS, n_combos


@app.cell
def _(N_LATENT, TANGENT_ENTROPY_N_SAMPLES):
    # Optional: W&B group to separate different experiments within the same project.
    # Set to a string (e.g. "fnn_sweep_2026", "sweep_v2") to group this sweep's runs.
    # When set, sweep jobs log to this group and post-hoc evaluation filters by it.
    # Set to None to log runs without a group.
    # WANDB_GROUP = None  # e.g. "fnn_sweep_2026-03" or "sweep_v2"
    # WANDB_GROUP = f"MLP_L_{N_LATENT}_AMP_LOSS_NO_RECON_JAC_NUC_SWEEP"
    # WANDB_GROUP = f"MLP_L_{N_LATENT}_FNN_SWEEP_SPARSIFY_{FNN_SPARSIFY}"
    WANDB_GROUP = f"MLP_L_{N_LATENT}_TANGENT_ENTROPY_SWEEP_N_SAMPLES_{TANGENT_ENTROPY_N_SAMPLES}_NOSAME"
    print(f"W&B group: {WANDB_GROUP}")
    return (WANDB_GROUP,)


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
    BATCH_SIZE,
    DATA_SOURCE,
    DECODER_HIDDEN_DIM,
    DECODER_N_LAYERS,
    DELAY_SPACING,
    EARLY_STOPPING_PATIENCE,
    FNN_ELEMENTWISE_REGULARIZATION,
    FNN_NORMALIZE,
    FNN_N_SAMPLES,
    FNN_SPARSIFY,
    FNN_USE_PCA,
    JACOBIAN_NUCLEAR_N_SAMPLES,
    K_STEPS_AHEAD,
    LEARNING_RATE,
    LIMIT_TRAIN_BATCHES,
    MAX_EPOCHS,
    MLP_DROPOUT,
    MLP_HIDDEN_DIM,
    MLP_N_LAYERS,
    MODEL,
    NEXT_STATE_BURN_IN,
    NORMALIZE,
    NUM_ICS,
    N_DELAYS,
    N_LATENT,
    N_OBS_PRED,
    N_PERIODS,
    OBSERVED_INDICES,
    OBS_NOISE,
    OmegaConf,
    PERCENT_THRESH,
    PTS_PER_PERIOD,
    SAVE_DIR,
    SEQ_LENGTH,
    SSM_DROPOUT,
    SSM_D_MODEL,
    SSM_D_STATE,
    SSM_FFN_EXPAND,
    SSM_N_LAYERS,
    SSM_R_MAX,
    SSM_R_MIN,
    TANGENT_ENTROPY_MODE,
    TANGENT_ENTROPY_N_SAMPLES,
    TCN_DROPOUT,
    TCN_KERNEL_SIZE,
    TCN_N_CHANNELS,
    TCN_N_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_N_LAYERS,
    USE_NEXT_STATE_DECODER,
    USE_SAME_STATE_DECODER,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WMTASK_DATALOADER,
    WMTASK_DIM,
    WMTASK_MODEL_TO_LOAD,
    WMTASK_NAME,
    WMTASK_PROJECT,
    WMTASK_TRAJ_WINDOW,
    load_encoder_config,
):
    # Format list-valued overrides without spaces so they double as valid Hydra CLI args
    if OBSERVED_INDICES == 'all':
        _obs_idx_str = 'all'
    else:
        _obs_idx_str = str(list(OBSERVED_INDICES)).replace(' ', '')
    overrides = [f'model={MODEL}', f'++model.encoder.n_latent={N_LATENT}', f'++model.use_same_state_decoder={str(USE_SAME_STATE_DECODER).lower()}', f'++model.use_next_state_decoder={str(USE_NEXT_STATE_DECODER).lower()}', f'++model.decoder_hidden_dim={DECODER_HIDDEN_DIM}', f'++model.decoder_n_layers={DECODER_N_LAYERS}', f'++model.next_state_burn_in={NEXT_STATE_BURN_IN}', f'++model.k_steps_ahead={K_STEPS_AHEAD}', f'++model.n_obs_pred={N_OBS_PRED}']
    if MODEL == 'mlp':
        overrides.extend([f'++model.encoder.hidden_dim={MLP_HIDDEN_DIM}', f'++model.encoder.n_layers={MLP_N_LAYERS}', f'++model.encoder.dropout={MLP_DROPOUT}'])  # --- Model ---
    elif MODEL == 'ssm':
        overrides.extend([f'++model.encoder.d_model={SSM_D_MODEL}', f'++model.encoder.d_state={SSM_D_STATE}', f'++model.encoder.n_layers={SSM_N_LAYERS}', f'++model.encoder.r_min={SSM_R_MIN}', f'++model.encoder.r_max={SSM_R_MAX}', f'++model.encoder.ffn_expand={SSM_FFN_EXPAND}', f'++model.encoder.dropout={SSM_DROPOUT}'])
    elif MODEL == 'transformer':
        overrides.extend([f'++model.encoder.d_model={TRANSFORMER_D_MODEL}', f'++model.encoder.n_heads={TRANSFORMER_N_HEADS}', f'++model.encoder.n_layers={TRANSFORMER_N_LAYERS}', f'++model.encoder.dim_feedforward={TRANSFORMER_DIM_FEEDFORWARD}', f'++model.encoder.dropout={TRANSFORMER_DROPOUT}'])
    elif MODEL == 'tcn':
        overrides.extend([f'++model.encoder.n_channels={TCN_N_CHANNELS}', f'++model.encoder.kernel_size={TCN_KERNEL_SIZE}', f'++model.encoder.n_layers={TCN_N_LAYERS}', f'++model.encoder.dropout={TCN_DROPOUT}'])
    overrides.extend([f'++training.batch_size={BATCH_SIZE}', f'++training.lightning.optimizer_kwargs.lr={LEARNING_RATE}', f'++training.lightning.fnn_use_pca={FNN_USE_PCA}', f'++training.lightning.fnn_n_samples={FNN_N_SAMPLES}', f'++training.lightning.fnn_normalize={FNN_NORMALIZE}', f'++training.lightning.fnn_elementwise_regularization={FNN_ELEMENTWISE_REGULARIZATION}', f'++training.lightning.fnn_sparsify={FNN_SPARSIFY}', f'++training.lightning.jacobian_nuclear_n_samples={JACOBIAN_NUCLEAR_N_SAMPLES}', f'++training.lightning.tangent_entropy_n_samples={TANGENT_ENTROPY_N_SAMPLES}', f'++training.lightning.tangent_entropy_mode={TANGENT_ENTROPY_MODE}', f'++training.trainer_params.max_epochs={MAX_EPOCHS}', f'++training.trainer_params.limit_train_batches={LIMIT_TRAIN_BATCHES}', f'++training.early_stopping.early_stopping_patience={EARLY_STOPPING_PATIENCE}', f'++training.early_stopping.percent_thresh={PERCENT_THRESH}', f'++data.train_test_params.seq_length={SEQ_LENGTH}', f'++data.postprocessing.obs_noise={OBS_NOISE}', f'++data.postprocessing.normalize={NORMALIZE}', f'++data.train_test_params.delay_embedding_params.observed_indices={_obs_idx_str}', f'++data.train_test_params.delay_embedding_params.n_delays={N_DELAYS}', f'++data.train_test_params.delay_embedding_params.delay_spacing={DELAY_SPACING}', f'++training.logger.save_dir={SAVE_DIR}', '++dirs_precreated=true'])
    if DATA_SOURCE == 'dysts':
        overrides.extend([f'++data.trajectory_params.num_ics={NUM_ICS}', f'++data.trajectory_params.n_periods={N_PERIODS}', f'++data.trajectory_params.pts_per_period={PTS_PER_PERIOD}'])
    elif DATA_SOURCE == 'wmtask':
        overrides.extend(['data=wmtask', f'++data.dataset_loader.project={WMTASK_PROJECT}', f'++data.dataset_loader.name={WMTASK_NAME}', f'++data.dataset_loader.model_to_load={WMTASK_MODEL_TO_LOAD}', f'++data.dataset_loader.dataloader_to_use={WMTASK_DATALOADER}', f'++data.dataset_loader.traj_window={WMTASK_TRAJ_WINDOW}', f'++data.flow.dim={WMTASK_DIM}'])
    # Architecture-specific encoder overrides
    cfg = load_encoder_config(overrides=overrides)
    if WANDB_PROJECT is None:
        if DATA_SOURCE == 'dysts':
            data_cls = cfg.data.flow._target_.split('.')[-1]
        else:
            data_cls = cfg.data.get('name', DATA_SOURCE)
        WANDB_PROJECT_1 = f'{data_cls}__EncoderOnly'
        WANDB_PROJECT_PATH_1 = f'{WANDB_ENTITY}/{WANDB_PROJECT_1}'
    print(f'W&B project: {WANDB_PROJECT_PATH_1}')
    encoder_cls = cfg.model.encoder._target_.split('.')[-1]
    print(f'Encoder: {encoder_cls}')
    if hasattr(cfg.model.encoder, 'hidden_dim'):
        print(f'  n_latent={cfg.model.encoder.n_latent}, hidden_dim={cfg.model.encoder.hidden_dim}, n_layers={cfg.model.encoder.n_layers}')
    else:
        print(f'  n_latent={cfg.model.encoder.n_latent}, d_model={cfg.model.encoder.d_model}')
    if OBSERVED_INDICES == 'all':
        _n_raw_obs = WMTASK_DIM
    else:
        _n_raw_obs = len(OBSERVED_INDICES)
    print(f'  Input dim (delay embedded): N_DELAYS={N_DELAYS} x n_raw_obs={_n_raw_obs} = {N_DELAYS * _n_raw_obs}')
    # Data-source-specific overrides
    # Auto-generate project name if not set
    print(OmegaConf.to_yaml(cfg))  # --- Training ---  # f"++training.lightning.jacobian_nuclear_weight={JACOBIAN_NUCLEAR_WEIGHT}",  # --- Shared data ---  # --- Partial observation / delay embedding ---  # --- Save directory ---  # --- Skip makedirs in SLURM jobs (dirs pre-created by notebook) ---
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
    _n_obs = trajs['train_trajs'].sequence.shape[-1]
    print(f'n_obs = {_n_obs}')
    print(f'Train: {len(train_dl.dataset)} sequences')
    print(f'Val:   {len(val_dl.dataset)} sequences')
    return (trajs,)


@app.cell
def _(np, plt, trajs):
    train_trajs = trajs['train_trajs'].sequence
    train_trajs = train_trajs.reshape(-1, train_trajs.shape[-1])
    # compute variance explained of each principal component
    from sklearn.decomposition import PCA
    pca = PCA(n_components=train_trajs.shape[1])
    pca.fit(train_trajs)
    variance_explained = pca.explained_variance_ratio_
    _cumulative_variance = np.cumsum(variance_explained)
    dim_99 = np.argmax(_cumulative_variance >= 0.99) + 1
    # Find dimension explaining at least 99% variance
    plt.plot(_cumulative_variance)
    plt.axhline(0.99, color='red', linestyle='--', label='99% Variance')
    plt.axvline(dim_99 - 1, color='green', linestyle='--', label=f'{dim_99} components')
    plt.xlabel('Number of principal components')
    plt.ylabel('Cumulative variance explained')
    plt.title('Cumulative variance explained by principal components')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 WMTask RNN Weight Matrix Analysis (wmtask only)

    Load the source RNN from [wmtask](https://github.com/adamjeisen/wmtask) and plot singular values and eigenvalues of its recurrent weight matrix `W_hh`.
    """)
    return


@app.cell
def _(DATA_SOURCE, WMTASK_MODEL_TO_LOAD, WMTASK_NAME, WMTASK_PROJECT, np, plt):
    if DATA_SOURCE == 'wmtask':
        from wmtask.loading import load_wmtask_model
        wmtask_model, wmtask_params = load_wmtask_model(WMTASK_PROJECT, WMTASK_NAME, model_to_load=WMTASK_MODEL_TO_LOAD)
        W_hh = wmtask_model.W_hh.detach().cpu().numpy()
        _U, s, Vh = np.linalg.svd(W_hh)
        eigvals = np.linalg.eigvals(W_hh)
        _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].semilogy(range(1, len(s) + 1), s, 'o-', markersize=4)
        axes[0].set_xlabel('Index')  # Singular values
        axes[0].set_ylabel('Singular value')
        axes[0].set_title('Singular values of W_hh')  # Eigenvalues (may be complex)
        axes[0].grid(True, alpha=0.3)
        axes[1].scatter(eigvals.real, eigvals.imag, alpha=0.7, s=20)
        axes[1].axhline(0, color='k', linewidth=0.5)
        axes[1].axvline(0, color='k', linewidth=0.5)
        axes[1].set_xlabel('Re(λ)')  # Singular values (sorted descending, already from SVD)
        axes[1].set_ylabel('Im(λ)')
        axes[1].set_title('Eigenvalues of W_hh')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:  # Eigenvalues in complex plane
        print("Skipped: RNN weight analysis only applies when DATA_SOURCE='wmtask'")
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
def _(SWEEP_PARAMS, WANDB_GROUP, WANDB_PROJECT_PATH_1, itertools, wandb):
    def _get_sweep_param_from_run(run, key):
        """Extract a sweep parameter value from a W&B run config.

        Keys are dot-separated Hydra paths like 'training.lightning.fnn_weight'.
        """
        parts = _key.split('.')
        _val = run.config
        for p in parts:
            if not isinstance(_val, dict) or p not in _val:
                return None
            _val = _val[p]
        return _val

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
    sweep_keys = list(SWEEP_PARAMS.keys())
    sweep_value_lists = [SWEEP_PARAMS[k] for k in sweep_keys]
    all_combos = [dict(zip(sweep_keys, _vals)) for _vals in itertools.product(*sweep_value_lists)]
    _api = wandb.Api()
    try:
        _run_filters = {'group': WANDB_GROUP} if WANDB_GROUP else None
        existing_runs = _api.runs(WANDB_PROJECT_PATH_1, filters=_run_filters)
        _msg = f'Found {len(existing_runs)} total runs in {WANDB_PROJECT_PATH_1}'
        if WANDB_GROUP:
            _msg = _msg + f' (group={WANDB_GROUP})'
        print(_msg)
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
    WANDB_GROUP,
    WANDB_PROJECT_1,
    n_combos,
    overrides,
    remaining_combos,
):
    ENTRY_POINT = 'python -m JacobianODE.encoder_only.run_encoder'
    if remaining_combos:
        n_remaining = len(remaining_combos)
        use_subset_sweep = n_remaining < n_combos
        wandb_overrides = [f'wandb_entity={WANDB_ENTITY}', f'wandb_project={WANDB_PROJECT_1}', 'slurm=default']
        if WANDB_GROUP:
            wandb_overrides.append(f'wandb_group={WANDB_GROUP}')
        if use_subset_sweep:
            sweep_cmds = []
            for _combo in remaining_combos:
                combo_overrides = [f'++{_key}={(str(_val).lower() if isinstance(_val, bool) else _val)}' for _key, _val in _combo.items()]
                sweep_overrides = overrides.copy() + combo_overrides + wandb_overrides
                sweep_cmds.append(f'{ENTRY_POINT} --multirun ' + ' '.join(sweep_overrides))
            print(f'Sweep command ({n_remaining} job(s), subset):')
            for _i, cmd in enumerate(sweep_cmds):
                print(f'  [{_i + 1}] {cmd[:120]}...')
        else:
            sweep_overrides = overrides.copy()
            for _key, _vals in SWEEP_PARAMS.items():
                val_str = ','.join((str(v).lower() if isinstance(v, bool) else str(v) for v in _vals))
                sweep_overrides.append(f'++{_key}={val_str}')
            sweep_overrides = sweep_overrides + wandb_overrides
            sweep_cmd = f'{ENTRY_POINT} --multirun ' + ' '.join(sweep_overrides)
            sweep_cmds = [sweep_cmd]
            print(f'Sweep command ({n_combos} jobs):')
            print(sweep_cmd)
    if not remaining_combos:
        sweep_cmd = None
        sweep_cmds = []
        print('No sweep to launch — all runs already completed.')
    return (sweep_cmds,)


@app.cell
def _(SAVE_DIR, os):
    # Pre-create directories so SLURM jobs skip os.makedirs entirely (avoids NFS hangs).
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "dysts_data"), exist_ok=True)
    print(f"Ensured directories exist:\n  {SAVE_DIR}\n  {os.path.join(SAVE_DIR, 'dysts_data')}")
    return


@app.cell
def _(subprocess, sweep_cmds):
    # Launch the sweep (submits SLURM jobs).
    if sweep_cmds:
        if len(sweep_cmds) > 1:
            print(f'Starting {len(sweep_cmds)} commands in parallel...')
        procs = [subprocess.Popen(cmd, shell=True) for cmd in sweep_cmds]
        for _i, p in enumerate(procs):
            p.wait()
            if p.returncode != 0 and len(procs) > 1:
                print(f'Command {_i + 1}/{len(procs)} exited with code {p.returncode}')
        print('Sweep complete.')
    else:
        print('No sweep to launch — all runs already completed.')
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
def _(WANDB_GROUP, WANDB_PROJECT_PATH_1, all_combos, wandb):
    _api = wandb.Api()
    _run_filters = {'group': WANDB_GROUP} if WANDB_GROUP else None
    all_runs = _api.runs(WANDB_PROJECT_PATH_1, filters=_run_filters)
    _msg = f'Found {len(all_runs)} total runs in {WANDB_PROJECT_PATH_1}'
    if WANDB_GROUP:
        _msg = _msg + f' (group={WANDB_GROUP})'
    print(_msg)
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
    """)
    return


@app.cell
def _(all_combos, finished_runs_1, pd, sweep_keys):
    records = []
    for r in finished_runs_1:
        if not any((_combo_matches_run(_combo, r) for _combo in all_combos)):
            continue
        rec = {}
        for _key in sweep_keys:
            short_name = _key.split('.')[-1]
            rec[short_name] = _get_sweep_param_from_run(r, _key)
        rec['val_same_loss'] = r.summary.get('val/same_state_loss', float('nan'))
        rec['val_next_loss'] = r.summary.get('val/next_state_loss', float('nan'))
        rec['val_fnn_loss'] = r.summary.get('val/fnn_loss', float('nan'))
        rec['val_amp_loss'] = r.summary.get('val/amplification_loss', float('nan'))
        rec['latent_util'] = r.summary.get('val/latent_utilization', float('nan'))
        rec['val_total_loss'] = r.summary.get('val/total_loss', float('nan'))
        rec['run_name'] = r.name
        rec['run_id'] = r.id
        records.append(rec)
    df = pd.DataFrame(records).sort_values('val_next_loss')
    print(f'{len(df)} finished runs in results')
    df.head(10)
    return (df,)


@app.cell
def _(df, plt):
    # ---- Scatter: fnn_weight vs val_same_state_loss, coloured by amplification_weight ----
    _fig, _ax = plt.subplots(figsize=(7, 5))
    sc = _ax.scatter(df['tangent_entropy_weight'], df['val_same_loss'], c='C0', label='val/same_state_loss', s=60, alpha=0.8, edgecolors='k', linewidths=0.4)
    sc = _ax.scatter(df['tangent_entropy_weight'], df['val_next_loss'], c='C1', label='val/next_state_loss', s=60, alpha=0.8, edgecolors='k', linewidths=0.4)  # df['fnn_weight'],
    plt.legend()  # df['n_latent'],
    _ax.set_xscale('symlog', linthresh=0.0001)
    _ax.set_yscale('log')
    _ax.set_xlabel('tangent_entropy_weight')
    _ax.set_ylabel('val/same_state_loss')
    _ax.set_title('Sweep: regularisation vs. reconstruction loss')
    plt.tight_layout()
    # plt.colorbar(sc, ax=ax, label='fnn_weight')
    # ax.set_xlabel('fnn_weight')
    # ax.set_xlabel('n_latent')
    plt.show()  # df['fnn_weight'],  # df['n_latent'],
    return


@app.cell
def _(df, plt):
    # ---- Latent utilisation distribution ----
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.hist(df['latent_util'].dropna(), bins=20, color='steelblue', edgecolor='white')
    _ax.set_xlabel('latent utilisation (entropy-based, 0–1)')
    _ax.set_ylabel('count')
    _ax.set_title('Latent utilisation across sweep runs')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df):
    # ---- Best run summary ----
    best = df.iloc[0]
    # best = df[df['fnn_weight'] == 1e-1].iloc[0]
    print('Best run (lowest val/same_state_loss):')
    for _col in df.columns:
        print(f'  {_col:40s}: {best[_col]}')
    return (best,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Tangent Space Variance (Best Encoder)

    Compute the variance explained by each dimension of the encoder tangent space for the best run (lowest val/same_state_loss), following the methodology from the Tangent Spaces notebook.
    """)
    return


@app.cell
def _(SAVE_DIR, WANDB_PROJECT_PATH_1, best, torch, trajs):
    from JacobianODE.encoder_only.pretrained import load_pretrained_encoder
    _SORT = True
    _adapter, _, _ = load_pretrained_encoder(project=WANDB_PROJECT_PATH_1, run_id=best['run_id'], save_dir=SAVE_DIR, freeze=True, verbose=True, require_same_state_decoder=False)
    _adapter.eval()
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _adapter.to(_device)
    _train_seq = trajs['train_trajs'].sequence.to(_device)
    _n_obs = _train_seq.shape[-1]
    _n_latent = _adapter.n_latent

    def _encode_point(x):
        """Map single observation (D,) to latent (n_latent,)."""
        return _adapter.encoder(x.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    _x_t = _train_seq[:, :-1, :].reshape(-1, _n_obs)
    print(f'x_t.shape: {_x_t.shape}')
    _x_next = _train_seq[:, 1:, :].reshape(-1, _n_obs)
    _N_diff = min(5000, _x_t.shape[0])
    _x_t = _x_t[:_N_diff]
    _x_next = _x_next[:_N_diff]
    print(f'x_t.shape after reshape: {_x_t.shape}, using {_N_diff} consecutive pairs')
    with torch.no_grad():
        _z_t = _adapter.encoder(_x_t.unsqueeze(1)).squeeze(1)
        _z_next = _adapter.encoder(_x_next.unsqueeze(1)).squeeze(1)
        _dz = _z_next - _z_t
        print(f'dz.shape: {_dz.shape}')
        jacobians_t = torch.func.vmap(torch.func.jacfwd(_encode_point))(_x_t)
        _U, _S, _V = torch.linalg.svd(jacobians_t, full_matrices=True)
    print(f'dz.shape: {_dz.shape}')
    print(f'U.shape: {_U.shape}')
    _tangent_projection = (_dz.unsqueeze(1) @ _U).squeeze(1)
    with torch.no_grad():
        _squared_projections = _tangent_projection ** 2
        if _SORT:
            squared_projections_sorted, _ = torch.sort(_squared_projections, dim=-1, descending=True)
            _variance_per_dim = squared_projections_sorted.mean(dim=0)
        else:
            _variance_per_dim = _squared_projections.mean(dim=0)
        _total_variance = _variance_per_dim.sum()
        variance_explained_1 = _variance_per_dim / _total_variance
        _cumulative_variance = torch.cumsum(variance_explained_1, dim=0)
    print(f'\nTangent space variance (best encoder, fnn_weight={best.get('fnn_weight', 'N/A')}):')
    print(f'{'Dimension':<10} | {'Variance Explained':<20} | {'Cumulative'}')
    print('-' * 45)
    for _i in range(len(variance_explained_1)):
        _val = variance_explained_1[_i].item() * 100
        _cum = _cumulative_variance[_i].item() * 100
        print(f'Dim {_i + 1:<5} | {_val:>17.4f}% | {_cum:>10.4f}%')
    return jacobians_t, load_pretrained_encoder, variance_explained_1


@app.cell
def _(np, plt, variance_explained_1):
    plt.scatter(np.arange(len(variance_explained_1)) + 1, variance_explained_1.cpu().numpy())
    plt.yscale('log')
    plt.show()
    return


@app.cell
def _(jacobians_t, np, plt, torch):
    import torch.nn.functional as F
    from scipy.optimize import curve_fit
    avg_column_norms = jacobians_t.norm(p=2, dim=-2).mean(dim=0).cpu()

    def exp_decay_with_floor(x, a, b, c):
        return a * np.exp(-b * x) + c
    delay_number = np.arange(jacobians_t.shape[-1]) + 1
    norms_np = avg_column_norms.numpy()
    # 1. Define the exponential decay model with a constant floor
    initial_guesses = [norms_np[0], 0.1, np.min(norms_np)]
    popt, _ = curve_fit(exp_decay_with_floor, delay_number, norms_np, p0=initial_guesses)
    A_fit, b_fit, C_fit = popt
    predicted_values = exp_decay_with_floor(delay_number, A_fit, b_fit, C_fit)
    residuals = norms_np - predicted_values
    std_noise = np.std(residuals)
    # 2. Fit the curve to the data
    # p0 provides initial guesses for [A, b, C] to help the solver converge quickly
    principled_noise_floor = C_fit + 2 * std_noise
    # initial_guesses = [norms_np[0], 1, np.min(norms_np)]
    principled_baseline = C_fit
    clean_norms = F.relu(torch.tensor(norms_np - principled_baseline))
    # Extract the fitted parameters
    energies = clean_norms ** 2
    reweighted_energies = energies / (energies.sum() + 1e-08)
    # # 3. Calculate the noise variance from the residuals
    # # The fit gives us the expected value. The residuals tell us the noise variance.
    sorted_energies, sorted_indices = torch.sort(reweighted_energies, descending=True)
    cutoff_idx = np.argmax(torch.cumsum(sorted_energies, dim=0).numpy() > 0.99) + 1
    used_indices = sorted_indices[:cutoff_idx]
    unused_indices = sorted_indices[cutoff_idx:]
    # Principled noise floor: The fitted asymptote + 2 standard deviations
    plt.plot(delay_number, predicted_values, 'k--', label='Fitted Exponential', alpha=0.5)
    plt.axhline(principled_baseline, color='g', linestyle='--', label='Fitted Asymptote')
    # 3. Principled baseline: Just the fitted asymptote
    plt.scatter(delay_number[used_indices], norms_np[used_indices], c='magenta', label='Used', zorder=5)
    plt.scatter(delay_number[unused_indices], norms_np[unused_indices], c='blue', label='Unused', zorder=5)
    # 4. Apply the subtraction and clamp to 0
    plt.xlabel('Delay #')
    plt.ylabel('Average Jacobian Column Norm')
    # 5. Calculate Energy and apply the 99% threshold
    plt.legend()
    # --- Plotting ---
    # plt.axhline(principled_noise_floor, color='r', linestyle=':', label='Calculated Noise Floor')
    plt.show()
    return


@app.cell
def _(
    SAVE_DIR,
    WANDB_PROJECT_PATH_1,
    best,
    load_pretrained_encoder,
    torch,
    trajs,
):
    _SORT = True
    K_NEIGHBORS = 10
    _adapter, _, _ = load_pretrained_encoder(project=WANDB_PROJECT_PATH_1, run_id=best['run_id'], save_dir=SAVE_DIR, freeze=True, verbose=True)
    _adapter.eval()
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _adapter.to(_device)
    _train_seq = trajs['train_trajs'].sequence.to(_device)
    _n_obs = _train_seq.shape[-1]
    _n_latent = _adapter.n_latent

    def _encode_point(x):
        """Map single observation (D,) to latent (n_latent,)."""
        return _adapter.encoder(x.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    _x_t = _train_seq.reshape(-1, _n_obs)
    print(f'Total x_t.shape: {_x_t.shape}')
    _N_diff = min(10000, _x_t.shape[0])
    _x_t = _x_t[:_N_diff]
    print(f'x_t.shape after reshape: {_x_t.shape}, finding {K_NEIGHBORS} spatial neighbors')
    with torch.no_grad():
        _z_t = _adapter.encoder(_x_t.unsqueeze(1)).squeeze(1)
        dists = torch.cdist(_z_t, _z_t)
        _, indices = torch.topk(dists, K_NEIGHBORS + 1, largest=False, dim=1)
        neighbor_idx = indices[:, 1:]
        z_neighbors = _z_t[neighbor_idx]
        _dz = z_neighbors - _z_t.unsqueeze(1)
        print(f'dz.shape (spatial perturbations): {_dz.shape}')
        jacobians_t_1 = torch.func.vmap(torch.func.jacfwd(_encode_point))(_x_t)
        _U, _S, _V = torch.linalg.svd(jacobians_t_1, full_matrices=True)
    print(f'U.shape: {_U.shape}')
    _tangent_projection = _dz @ _U
    with torch.no_grad():
        _squared_projections = _tangent_projection ** 2
        local_variance = _squared_projections.mean(dim=1)
        if _SORT:
            local_variance_sorted, _ = torch.sort(local_variance, dim=-1, descending=True)
            _variance_per_dim = local_variance_sorted.mean(dim=0)
        else:
            _variance_per_dim = local_variance.mean(dim=0)
        _total_variance = _variance_per_dim.sum()
        variance_explained_2 = _variance_per_dim / _total_variance
        _cumulative_variance = torch.cumsum(variance_explained_2, dim=0)
    print(f'\nTangent space spatial variance (best encoder, fnn_weight={best.get('fnn_weight', 'N/A')}):')
    print(f'{'Dimension':<10} | {'Variance Explained':<20} | {'Cumulative'}')
    print('-' * 45)
    for _i in range(len(variance_explained_2)):
        _val = variance_explained_2[_i].item() * 100
        _cum = _cumulative_variance[_i].item() * 100
        print(f'Dim {_i + 1:<5} | {_val:>17.4f}% | {_cum:>10.4f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Going by next state
    """)
    return


@app.cell
def _(df):
    best_1 = df.loc[df['val_next_loss'].idxmin()]
    print('Best run (lowest val/next_state_loss):')
    for _col in df.columns:
        print(f'  {_col:40s}: {best_1[_col]}')
    return (best_1,)


@app.cell
def _(
    SAVE_DIR,
    WANDB_PROJECT_PATH_1,
    best_1,
    load_pretrained_encoder,
    torch,
    trajs,
):
    _adapter, _, _ = load_pretrained_encoder(project=WANDB_PROJECT_PATH_1, run_id=best_1['run_id'], save_dir=SAVE_DIR, freeze=True, verbose=True)
    _adapter.eval()
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _adapter.to(_device)
    _train_seq = trajs['train_trajs'].sequence.to(_device)
    _n_obs = _train_seq.shape[-1]
    _n_latent = _adapter.n_latent

    def _encode_point(x):
        """Map single observation (D,) to latent (n_latent,)."""
        return _adapter.encoder(x.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    _x_t = _train_seq[:, :-1, :].reshape(-1, _n_obs)
    print(f'x_t.shape: {_x_t.shape}')
    _x_next = _train_seq[:, 1:, :].reshape(-1, _n_obs)
    _N_diff = min(2000, _x_t.shape[0])
    _x_t = _x_t[:_N_diff]
    _x_next = _x_next[:_N_diff]
    print(f'x_t.shape after reshape: {_x_t.shape}, using {_N_diff} consecutive pairs')
    with torch.no_grad():
        _z_t = _adapter.encoder(_x_t.unsqueeze(1)).squeeze(1)
        _z_next = _adapter.encoder(_x_next.unsqueeze(1)).squeeze(1)
        _dz = _z_next - _z_t
        print(f'dz.shape: {_dz.shape}')
        jacobians_t_2 = torch.func.vmap(torch.func.jacfwd(_encode_point))(_x_t)
        _U, _S, _V = torch.linalg.svd(jacobians_t_2, full_matrices=True)
    print(f'dz.shape: {_dz.shape}')
    print(f'U.shape: {_U.shape}')
    _tangent_projection = (_dz.unsqueeze(1) @ _U).squeeze(1)
    with torch.no_grad():
        _squared_projections = _tangent_projection ** 2
        _variance_per_dim = _squared_projections.mean(dim=0)
        _total_variance = _variance_per_dim.sum()
        variance_explained_3 = _variance_per_dim / _total_variance
        _cumulative_variance = torch.cumsum(variance_explained_3, dim=0)
    print(f'\nTangent space variance (best encoder, fnn_weight={best_1.get('fnn_weight', 'N/A')}):')
    print(f'{'Dimension':<10} | {'Variance Explained':<20} | {'Cumulative'}')
    print('-' * 45)
    for _i in range(len(variance_explained_3)):
        _val = variance_explained_3[_i].item() * 100
        _cum = _cumulative_variance[_i].item() * 100
        print(f'Dim {_i + 1:<5} | {_val:>17.4f}% | {_cum:>10.4f}%')
    return


@app.cell
def _():
    # # -----------------------------------------------------------------------------
    # # Variance explained by each active/inactive dimension of the input space
    # # -----------------------------------------------------------------------------
    # # Requires: df, best, trajs, WANDB_PROJECT_PATH, SAVE_DIR from Sections 3 and 7

    # from JacobianODE.encoder_only.pretrained import load_pretrained_encoder

    # # Load the best encoder (lowest val/same_state_loss)
    # adapter, _, _ = load_pretrained_encoder(
    #     project=WANDB_PROJECT_PATH,
    #     run_id=best["run_id"],
    #     save_dir=SAVE_DIR,
    #     freeze=True,
    #     verbose=True,
    # )
    # adapter.eval()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # adapter.to(device)

    # # Training sequences: (n_sequences, seq_length, n_obs)
    # train_seq = trajs["train_trajs"].sequence.to(device)
    # n_obs = train_seq.shape[-1]
    # n_latent = adapter.n_latent


    # def encode_point(x):
    #     """Map single observation (D,) to latent (n_latent,)."""
    #     return adapter.encoder(x.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


    # # Consecutive pairs for dx
    # x_t = train_seq[:, :-1, :].reshape(-1, n_obs)
    # print(f"x_t.shape: {x_t.shape}")
    # x_next = train_seq[:, 1:, :].reshape(-1, n_obs)
    # N_diff = min(2000, x_t.shape[0])
    # x_t = x_t[:N_diff]
    # x_next = x_next[:N_diff]
    # print(f"x_t.shape after reshape: {x_t.shape}, using {N_diff} consecutive pairs")

    # with torch.no_grad():
    #     # Calculate the step in the observation space
    #     dx = x_next - x_t
    #     print(f"dx.shape: {dx.shape}")
    
    #     jacobians_t = torch.func.vmap(torch.func.jacfwd(encode_point))(x_t)
    #     # Note: PyTorch svd returns U, S, V^T. The third output is V^T.
    #     U, S, V_T = torch.linalg.svd(jacobians_t, full_matrices=True)

    # # Project dx onto the right singular vectors (the true V matrix)
    # # We transpose V_T to get V, where the columns are the right singular vectors.
    # V_true = V_T.transpose(-1, -2)
    # input_projection = (dx.unsqueeze(1) @ V_true).squeeze(1)

    # # Variance explained per dimension in the input space
    # with torch.no_grad():
    #     squared_projections = input_projection**2
    #     variance_per_dim = squared_projections.mean(dim=0)
    #     total_variance = variance_per_dim.sum()
    #     variance_explained = variance_per_dim / total_variance
    #     cumulative_variance = torch.cumsum(variance_explained, dim=0)

    # print(f"\nInput space variance (best encoder, fnn_weight={best.get('fnn_weight', 'N/A')}):")
    # print(f"{'Dimension':<10} | {'Variance Explained':<20} | {'Cumulative'}")
    # print("-" * 45)
    # for i in range(len(variance_explained)):
    #     val = variance_explained[i].item() * 100
    #     cum = cumulative_variance[i].item() * 100
    #     print(f"Dim {i+1:<5} | {val:>17.4f}% | {cum:>10.4f}%")
    return


if __name__ == "__main__":
    app.run()
