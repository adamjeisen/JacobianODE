import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameter Sweep — Latent JacobianODE

    This notebook launches a hyperparameter sweep for the **Latent JacobianODE** model.
    It supports two training modes:

    - **`from_scratch`**: Encoder and JacobianODE are trained jointly from random initialization.
    - **`pretrained`**: A pre-trained encoder (loaded from W&B) is used; only the JacobianODE is trained (encoder optionally fine-tuned).

    **Supports two data sources** (set `DATA_SOURCE` in Section 1 for from_scratch; inherited from encoder for pretrained):
    - `"dysts"` — synthetic trajectories from dynamical systems (e.g. Lorenz)
    - `"wmtask"` — hidden-state trajectories from a trained working memory RNN

    **Workflow:**
    1. Select the training mode and configure hyperparameters (Sections 1–3)
    2. Build the Hydra config and generate data (Sections 4–5)
    3. Launch the Hydra `--multirun` sweep via SLURM (Section 6)
    4. After training completes, use the companion **Sweep Analytics** notebook for model selection and diagnostics.
    """)
    return


@app.cell
def _():
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    from omegaconf import OmegaConf
    import os
    import subprocess
    import sys
    import torch
    from tqdm.auto import tqdm
    import wandb

    from JacobianODE.jacobians import (
        load_config,
        initialize_config,
        seed_everything,
        make_trajectories,
        postprocess_data,
        create_dataloaders,
    )
    from JacobianODE.jacobians.tuning import DEFAULT_LAMBDA_LOOP_VALUES

    torch.set_float32_matmul_precision('high')
    return (
        create_dataloaders,
        initialize_config,
        itertools,
        load_config,
        make_trajectories,
        np,
        os,
        plt,
        postprocess_data,
        seed_everything,
        subprocess,
        sys,
        torch,
        tqdm,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Training Mode and Data Source

    Set `MODE` to `"from_scratch"` or `"pretrained"`.

    - **`from_scratch`**: Define encoder architecture below (Section 2a). Set `DATA_SOURCE` to `"dysts"` or `"wmtask"` in the Paths cell.
    - **`pretrained`**: Specify the W&B project and run ID of the pre-trained encoder (Section 2b). Data source is inherited from the encoder config.
    """)
    return


@app.cell
def _():
    # ----------------------------------------------------------------
    # Training mode: "from_scratch" or "pretrained"
    # ----------------------------------------------------------------
    MODE = "from_scratch"  # <-- CHANGE THIS

    assert MODE in ("from_scratch", "pretrained"), f"Invalid MODE: {MODE}"

    # ----------------------------------------------------------------
    # Encoder type (only relevant for from_scratch mode):
    #   "mlp"                         — pointwise MLP (dimension-reducing, existing behavior)
    #   "coupling"                    — AffineCouplingEncoder (dimension-preserving, invertible)
    #   "spline_coupling"             — CouplingEncoder with rational quadratic splines + ActNorm
    #   "spline_autoregressive"       — SplineAutoregressiveEncoder (A-RQS, arXiv:2302.12024)
    #   "cubic_rational_coupling"     — CouplingEncoder with cubic rational bijections (Gerdes & Cheng 2026)
    #   "sinh_coupling"               — CouplingEncoder with sinh conjugation bijections
    #   "cubic_conjugation_coupling"  — CouplingEncoder with cubic conjugation bijections
    #   "cubic_rational_autoregressive"    — AnalyticAutoregressiveEncoder with cubic rational
    #   "sinh_autoregressive"              — AnalyticAutoregressiveEncoder with sinh conjugation
    #   "cubic_conjugation_autoregressive" — AnalyticAutoregressiveEncoder with cubic conjugation
    # ----------------------------------------------------------------
    # ENCODER_TYPE = "mlp"  # <-- CHANGE THIS
    # ENCODER_TYPE = "coupling"
    ENCODER_TYPE = "spline_coupling"
    # ENCODER_TYPE = "spline_autoregressive"
    # ENCODER_TYPE = "cubic_rational_coupling"
    # ENCODER_TYPE = "sinh_coupling"
    # ENCODER_TYPE = "cubic_conjugation_coupling"
    # ENCODER_TYPE = "cubic_rational_autoregressive"
    # ENCODER_TYPE = "sinh_autoregressive"
    # ENCODER_TYPE = "cubic_conjugation_autoregressive"

    # Convenience sets for condition checks
    _ANALYTIC_COUPLING_TYPES = ("cubic_rational_coupling", "sinh_coupling", "cubic_conjugation_coupling")
    _ANALYTIC_AR_TYPES = ("cubic_rational_autoregressive", "sinh_autoregressive", "cubic_conjugation_autoregressive")
    _ALL_COUPLING_TYPES = ("coupling", "spline_coupling") + _ANALYTIC_COUPLING_TYPES
    _ALL_AR_TYPES = ("spline_autoregressive",) + _ANALYTIC_AR_TYPES
    ALL_INVERTIBLE_TYPES = _ALL_COUPLING_TYPES + _ALL_AR_TYPES

    _VALID_TYPES = ("mlp",) + ALL_INVERTIBLE_TYPES
    assert ENCODER_TYPE in _VALID_TYPES, f"Invalid ENCODER_TYPE: {ENCODER_TYPE}"
    print(f"Training mode: {MODE}, Encoder: {ENCODER_TYPE}")
    return ALL_INVERTIBLE_TYPES, ENCODER_TYPE, MODE


@app.cell
def _(os):
    # ----------------------------------------------------------------
    # Paths and W&B settings
    # ----------------------------------------------------------------
    SAVE_DIR     = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs"
    WANDB_ENTITY = "JacobianODE"

    # ----------------------------------------------------------------
    # Data source: "dysts" or "wmtask" (from_scratch only; pretrained inherits from encoder)
    # ----------------------------------------------------------------
    # DATA_SOURCE = "dysts"   # <-- change to "wmtask" for WM task RNN data
    DATA_SOURCE = "wmtask"

    # ----------------------------------------------------------------
    # wmtask-specific hyperparameters (ignored when DATA_SOURCE="dysts")
    # ----------------------------------------------------------------
    WMTASK_PROJECT = "WMSelectionTask__cue_time_0.1__response_time_0.25__enforce_fixation_False"
    WMTASK_NAME = "BiologicalRNN__cue_time_0.1__learning_rate_0.0005__max_epochs_42__N1_64__N2_64__tau_0.05__dt_0.02__eig_lower_bound_0.1__init_mode_random"
    WMTASK_MODEL_TO_LOAD = "final"
    WMTASK_DATALOADER = "all"
    WMTASK_TRAJ_WINDOW = "delay2"   # 'delay2' or 'full'
    WMTASK_DIM = 128                # N1 + N2 = 64 + 64

    os.makedirs(SAVE_DIR, exist_ok=True)
    if DATA_SOURCE == "dysts":
        data_dir = f"{SAVE_DIR}/dysts_data"
        os.makedirs(data_dir, exist_ok=True)
    return (
        DATA_SOURCE,
        SAVE_DIR,
        WANDB_ENTITY,
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
    ## 2. Data Hyperparameters
    """)
    return


@app.cell
def _(DATA_SOURCE):
    # ----------------------------------------------------------------
    # Data hyperparameters
    # ----------------------------------------------------------------
    OBS_NOISE        = 0.05
    NORMALIZE        = True

    if DATA_SOURCE == "dysts":
        # OBSERVED_INDICES = [0, 1, 2]
        OBSERVED_INDICES = [0]
        N_PERIODS      = 12
        PTS_PER_PERIOD = 100
        NUM_ICS        = 32
    else:
        # Partial observation: "random" picks N_OBSERVED dims at random,
        # "all" uses all dims, or pass an explicit list of indices.
        OBSERVED_INDICES = 'all'
        # OBSERVED_INDICES = "random"
        # N_OBSERVED       = 12        # number of dims (only used when "random")
        PARTIAL_OBS_SEED = None      # None → uses flow.random_state (default 42)

    # N_DELAYS will be determined by amplification loss analysis (Section 4).
    # DELAY_SPACING is fixed at 1.
    DELAY_SPACING = 1

    # Known Lorenz Lyapunov exponents (dysts only; sigma=10, rho=28, beta=8/3)
    TRUE_LYAPUNOV = [0.91, 0.0, -14.57] if DATA_SOURCE == "dysts" else None

    print(f"DATA_SOURCE: {DATA_SOURCE}")
    if DATA_SOURCE == "dysts":
        print(f"Data: NUM_ICS={NUM_ICS}, N_PERIODS={N_PERIODS}, PTS_PER_PERIOD={PTS_PER_PERIOD}")
    print(f"  OBS_NOISE={OBS_NOISE}, NORMALIZE={NORMALIZE}")
    print(f"  OBSERVED_INDICES={OBSERVED_INDICES}, DELAY_SPACING={DELAY_SPACING}")
    return (
        DELAY_SPACING,
        NORMALIZE,
        NUM_ICS,
        N_PERIODS,
        OBSERVED_INDICES,
        OBS_NOISE,
        PARTIAL_OBS_SEED,
        PTS_PER_PERIOD,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Load Data (for delay embedding analysis)

    Load trajectories with `n_delays=1` (no delay embedding yet) so we can analyze
    the raw observed signal and determine the optimal number of delays.
    """)
    return


@app.cell
def _(
    DATA_SOURCE,
    NORMALIZE,
    NUM_ICS,
    N_OBSERVED,
    N_PERIODS,
    OBSERVED_INDICES,
    OBS_NOISE,
    PARTIAL_OBS_SEED,
    PTS_PER_PERIOD,
    SAVE_DIR,
    WMTASK_DATALOADER,
    WMTASK_DIM,
    WMTASK_MODEL_TO_LOAD,
    WMTASK_NAME,
    WMTASK_PROJECT,
    WMTASK_TRAJ_WINDOW,
    initialize_config,
    load_config,
    make_trajectories,
    seed_everything,
):
    _obs_idx_str = str(list(OBSERVED_INDICES)).replace(' ', '') if isinstance(OBSERVED_INDICES, list) else OBSERVED_INDICES
    _data_overrides = [f'data={DATA_SOURCE}', f'data.postprocessing.obs_noise={OBS_NOISE}', f'data.postprocessing.normalize={str(NORMALIZE).lower()}', f'data.train_test_params.delay_embedding_params.observed_indices={_obs_idx_str}', 'data.train_test_params.delay_embedding_params.n_delays=1', 'data.train_test_params.delay_embedding_params.delay_spacing=1', f'training.logger.save_dir={SAVE_DIR}']
    if OBSERVED_INDICES == 'random':
        _data_overrides.append(f'data.train_test_params.delay_embedding_params.n_observed={N_OBSERVED}')
        if PARTIAL_OBS_SEED is not None:
            _data_overrides.append(f'data.train_test_params.delay_embedding_params.partial_obs_seed={PARTIAL_OBS_SEED}')
    if DATA_SOURCE == 'dysts':
        _data_overrides = _data_overrides + [f'data.trajectory_params.num_ics={NUM_ICS}', f'data.trajectory_params.n_periods={N_PERIODS}', f'data.trajectory_params.pts_per_period={PTS_PER_PERIOD}']
    else:
        _data_overrides = _data_overrides + [f'++data.dataset_loader.project={WMTASK_PROJECT}', f'++data.dataset_loader.name={WMTASK_NAME}', f'++data.dataset_loader.model_to_load={WMTASK_MODEL_TO_LOAD}', f'++data.dataset_loader.dataloader_to_use={WMTASK_DATALOADER}', f'++data.dataset_loader.traj_window={WMTASK_TRAJ_WINDOW}', f'++data.flow.dim={WMTASK_DIM}']
    cfg_raw = load_config(overrides=_data_overrides)
    cfg_raw = initialize_config(cfg_raw)
    seed_everything(cfg_raw.data.flow.random_state)
    _resolved_obs = cfg_raw.data.train_test_params.delay_embedding_params.observed_indices
    print(f'Resolved OBSERVED_INDICES: {_resolved_obs}')
    _eq, sol, _dt = make_trajectories(cfg_raw, verbose=True)
    print(f'\nFull trajectory shape: {sol['values'].shape}')
    print(f'Time step dt = {_dt:.4f}')
    return cfg_raw, sol


@app.cell
def _(cfg_raw, create_dataloaders, postprocess_data, sol):
    result_raw = postprocess_data(cfg_raw, sol['values'])
    _values = result_raw.values
    _train_dl, _val_dl, _test_dl, _trajs = create_dataloaders(cfg_raw, _values, verbose=True, return_full_obs=True)
    OBSERVED_INDICES_1 = cfg_raw.data.train_test_params.delay_embedding_params.observed_indices
    if OBSERVED_INDICES_1 != 'all':
        OBSERVED_INDICES_1 = list(OBSERVED_INDICES_1)
    print(f'\nResolved OBSERVED_INDICES: {OBSERVED_INDICES_1}')
    x_obs = _trajs['train_trajs'].sequence
    n_obs_dims = x_obs.shape[-1]
    print(f'Training data for delay analysis: {x_obs.shape}')
    print(f'  n_observed_dims = {n_obs_dims}, T = {x_obs.shape[1]}')
    return OBSERVED_INDICES_1, n_obs_dims, x_obs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Amplification Loss: Choose N_DELAYS

    Sweep over candidate `n_delays` values (with `delay_spacing=1`) and compute the
    noise amplification loss for each. The optimal number of delays minimizes
    amplification — where the delay embedding best preserves local geometry.
    """)
    return


@app.cell
def _(DELAY_SPACING, n_obs_dims, np, torch, tqdm, x_obs):
    from JacobianODE.fnn import loss_amplification
    from JacobianODE.jacobians.data import embed_signal_torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------------------------------------------
    # Sweep settings
    # ----------------------------------------------------------------
    MAX_N_DELAYS = 5          # maximum number of delays to try
    MAX_POINTS   = 50000       # subsample if total points exceed this (OOM safety)
    AMP_MAX_T    = 30          # look-ahead steps for amplification loss
    AMP_N_NEIGHBORS = 10

    # Determine max feasible delays (need at least half the time steps remaining)
    max_feasible = x_obs.shape[1] // 2
    N_DELAYS_VALS = np.arange(1, min(MAX_N_DELAYS, max_feasible) + 1)

    amplification_losses = {}

    for n_d in tqdm(N_DELAYS_VALS, desc="Amplification loss sweep"):
        x_embedded = embed_signal_torch(x_obs, n_d, DELAY_SPACING)
        if x_embedded.shape[1] < AMP_MAX_T + AMP_N_NEIGHBORS:
            break  # not enough time steps left

        x_data = x_embedded[..., :n_obs_dims]

        # Flatten and subsample to avoid OOM
        x_emb_flat = x_embedded.reshape(-1, x_embedded.shape[-1])
        x_dat_flat = x_data.reshape(-1, x_data.shape[-1])

        if x_emb_flat.shape[0] > MAX_POINTS:
            idx = torch.randperm(x_emb_flat.shape[0])[:MAX_POINTS]
            x_emb_flat = x_emb_flat[idx]
            x_dat_flat = x_dat_flat[idx]

        # Reshape to (1, N, D) for loss_amplification
        x_emb_in = x_emb_flat.unsqueeze(0).to(device)
        x_dat_in = x_dat_flat.unsqueeze(0).to(device)

        with torch.no_grad():
            loss = loss_amplification(
                x_emb_in, x_dat_in,
                max_T=AMP_MAX_T, n_neighbors=AMP_N_NEIGHBORS, normalize=True,
            )
        amplification_losses[n_d] = loss.item()

    print(f"Computed amplification loss for {len(amplification_losses)} delay values")
    return amplification_losses, embed_signal_torch


@app.cell
def _(amplification_losses, np, plt):
    # ----------------------------------------------------------------
    # Plot amplification loss vs number of delays
    n_delays_arr = np.array(list(amplification_losses.keys()))
    amp_loss_arr = np.array(list(amplification_losses.values()))
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(n_delays_arr, amp_loss_arr, 'o-', markersize=4)
    _ax.set_xlabel('Number of delays')
    _ax.set_ylabel('Amplification loss')
    _ax.set_title('Amplification Loss vs. Number of Delays')
    best_idx = np.argmin(amp_loss_arr)
    # ax.set_yscale("log")
    N_DELAYS = int(n_delays_arr[best_idx])
    _ax.axvline(N_DELAYS, color='red', linestyle='--', alpha=0.7, label=f'Best: n_delays={N_DELAYS}')
    # override
    N_DELAYS = 1
    # Auto-select: argmin
    _ax.legend()
    plt.tight_layout()
    plt.show()
    if N_DELAYS != int(n_delays_arr[best_idx]):
        print(f'\nSelected N_DELAYS = {N_DELAYS} (amp loss = {amp_loss_arr[best_idx]:.5g}) [OVERRIDDEN - optimal is {int(n_delays_arr[best_idx])}]')
        print('WARNING: N_DELAYS does not match automatically selected optimal value. N_DELAYS OVERRIDEN.')
    else:
        print(f'\nSelected N_DELAYS = {N_DELAYS} (amp loss = {amp_loss_arr[best_idx]:.5g})')
    print('Override N_DELAYS here if desired.')
    return (N_DELAYS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. PCA on Delay-Embedded Data: Choose N_TARGET_DIMS

    Embed the training data with the chosen `N_DELAYS` and compute PCA to determine
    how many principal components capture 99% of variance. This guides the choice
    of `N_TARGET_DIMS` (for coupling encoders) or `N_LATENT` (for MLP encoders).
    """)
    return


@app.cell
def _(DELAY_SPACING, N_DELAYS, embed_signal_torch, np, plt, x_obs):
    from sklearn.decomposition import PCA
    x_delay_embedded = embed_signal_torch(x_obs, N_DELAYS, DELAY_SPACING)
    x_de_flat = x_delay_embedded.reshape(-1, x_delay_embedded.shape[-1]).numpy()
    print(f'Delay-embedded shape: {x_delay_embedded.shape}')
    print(f'  → flattened for PCA: {x_de_flat.shape}')
    n_components = min(x_de_flat.shape[0], x_de_flat.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(x_de_flat)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    thresh = 0.99
    dim_99 = int(np.argmax(cumulative_variance >= thresh) + 1)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance)
    _ax.axhline(thresh, color='red', linestyle='--', label=f'{thresh:.0%} variance')
    _ax.axvline(dim_99, color='green', linestyle='--', label=f'{dim_99} components')
    _ax.set_xlabel('Number of principal components')
    _ax.set_ylabel('Cumulative variance explained')
    _ax.set_title(f'PCA on Delay-Embedded Data (n_delays={N_DELAYS})')
    _ax.legend()
    plt.tight_layout()
    plt.show()
    print(f'\n{dim_99} PCs capture {thresh:.0%} of variance in the delay-embedded data.')
    print(f'Suggested N_TARGET_DIMS = {dim_99}')
    return (dim_99,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6a. From-Scratch Encoder Settings

    *Skip this section if `MODE = "pretrained"`.*
    """)
    return


@app.cell
def _(ENCODER_TYPE, MODE, dim_99):
    # ================================================================
    # From-scratch encoder settings (skip if MODE == "pretrained")
    # ================================================================
    if MODE == "from_scratch":

        if ENCODER_TYPE == "mlp":
            # --- MLP encoder (pointwise: same MLP at every timestep) ---
            MLP_HIDDEN_DIM = 256
            MLP_N_LAYERS   = 3
            MLP_DROPOUT    = 0.1

            # Decoder MLP (pointwise: z_t -> x_hat_t)
            DECODER_HIDDEN = 128
            DECODER_LAYERS = 2

            # Decode only the most recent delay (first N columns of delay embedding)?
            # When False, the decoder reconstructs the full delay embedding.
            # When True, the decoder only reconstructs the most recent delay block
            # (i.e. the raw observed dimensions, not the full n_delays * n_obs embedding).
            # DECODE_ONLY_RECENT = False
            DECODE_ONLY_RECENT = True

        elif ENCODER_TYPE == "coupling":
            # --- Affine Coupling Flow encoder (dimension-preserving, invertible) ---
            N_COUPLING_LAYERS   = 8      # number of affine coupling stacks
            COUPLING_HIDDEN_DIM = 128    # conditioner MLP width
            N_HIDDEN_LAYERS     = 2      # conditioner MLP depth
            SCALE_ACTIVATION    = "tanh" # bounds log-scale via tanh(·)
            SCALE_CLAMP         = 3.0    # max |log_s|, ~20× scaling range
            ZERO_INIT           = True   # identity initialization trick
            PERMUTATION_SEED    = 42      # base seed for fixed random permutations

            # ---- Numerical stability (Andrade 2024, arXiv:2402.16408) ----
            CLAMP_TYPE          = "symmetric"  # 'symmetric' | 'asymmetric'
            ALPHA_POS           = 0.1          # asymmetric clamp: expansion bound
            ALPHA_NEG           = 2.0          # asymmetric clamp: compression bound
            USE_LOFT            = False        # LOFT layer after coupling blocks
            LOFT_TAU            = 100.0        # LOFT threshold

            # Coupling has an exact inverse → no separate decoder needed
            DECODE_ONLY_RECENT = False

            # ---- Subspace splitting ----
            # N_TARGET_DIMS        = 3          # dynamic subspace (e.g. 3 for Lorenz)
            N_TARGET_DIMS        = dim_99          
            # N_TARGET_DIMS       = 6          # dynamic subspace (e.g. 3 for Lorenz)
            # N_TARGET_DIMS       = 7
            # KL_NULL_WEIGHT       = 1.0        # null-space MSE penalty (structural constraint)
            # KL_DYN_WEIGHT        = 0.0        # dynamic subspace KL divergence (smoothness regularizer)
            # KL_NULL_WEIGHT = 0.0
            RECONSTRUCTION_MODE  = "most_recent"  # 'uniform' | 'harmonic' | 'most_recent'

            # ---- VAE settings ----
            USE_VAE                = False    # VAE reparameterization on dynamic subspace
            VAE_SAMPLE_ALL_LOSSES  = False    # when False, only recon uses sampled z_dyn
            KL_WARMUP_EPOCHS       = 0       # 0 = fixed weight, >0 = linear ramp

            # ---- Tangent space entropy regularizer ----
            TANGENT_ENTROPY_WEIGHT    = 0.0
            # TANGENT_ENTROPY_WEIGHT    = None # sweep it
            TANGENT_ENTROPY_MODE      = "quadratic"  # 'shannon' | 'quadratic' | 'renyi_half'
            TANGENT_ENTROPY_N_SAMPLES = 256

        elif ENCODER_TYPE == "spline_coupling":
            # --- Spline Coupling Flow encoder (rational quadratic splines + ActNorm) ---
            N_COUPLING_LAYERS   = 8      # number of spline coupling stacks
            COUPLING_HIDDEN_DIM = 128    # conditioner MLP width
            N_HIDDEN_LAYERS     = 2      # conditioner MLP depth
            NUM_BINS            = 8      # rational quadratic spline segments
            TAIL_BOUND          = 3.0    # linear tails outside [-B, B]
            USE_ACTNORM         = True   # ActNorm between coupling layers
            ZERO_INIT           = True   # identity initialization trick
            PERMUTATION_SEED    = 42     # base seed for fixed random permutations
            USE_LOFT            = False  # LOFT layer after coupling blocks
            LOFT_TAU            = 100.0  # LOFT threshold

            # Coupling has an exact inverse → no separate decoder needed
            DECODE_ONLY_RECENT = False

            # ---- Subspace splitting ----
            # N_TARGET_DIMS        = 3          # dynamic subspace (e.g. 3 for Lorenz)
            # N_TARGET_DIMS        = 7
            # N_TARGET_DIMS        = 17
            # N_TARGET_DIMS        = dim_99
            N_TARGET_DIMS = 128
            # KL_NULL_WEIGHT     = 1.0        # null-space MSE penalty (structural constraint)
            KL_NULL_WEIGHT       = "null"
            KL_DYN_WEIGHT        = 0.0001        # dynamic subspace KL divergence (smoothness regularizer)
            RECONSTRUCTION_MODE  = "most_recent"  # 'uniform' | 'harmonic' | 'most_recent'

            # ---- VAE settings ----
            USE_VAE                = True    # VAE reparameterization on dynamic subspace
            VAE_SAMPLE_ALL_LOSSES  = False    # when False, only recon uses sampled z_dyn
            KL_WARMUP_EPOCHS       = 0       # 0 = fixed weight, >0 = linear ramp

            # ---- Tangent space entropy regularizer ----
            TANGENT_ENTROPY_WEIGHT    = 0.0
            TANGENT_ENTROPY_MODE      = "quadratic"  # 'shannon' | 'quadratic' | 'renyi_half'
            TANGENT_ENTROPY_N_SAMPLES = 256

        elif ENCODER_TYPE in _ANALYTIC_COUPLING_TYPES:
            # --- Analytic Coupling Flow encoder (Gerdes & Cheng 2026, arXiv:2601.10774) ---
            # C-infinity bijections with exact closed-form inverse, fewer params than RQS.
            N_COUPLING_LAYERS   = 8      # number of analytic coupling stacks
            COUPLING_HIDDEN_DIM = 128    # conditioner MLP width
            N_HIDDEN_LAYERS     = 2      # conditioner MLP depth
            USE_ACTNORM         = True   # ActNorm between coupling layers
            ZERO_INIT           = True   # identity initialization trick
            PERMUTATION_SEED    = 42     # base seed for fixed random permutations
            USE_LOFT            = False  # LOFT layer after coupling blocks
            LOFT_TAU            = 100.0  # LOFT threshold

            # Map ENCODER_TYPE → coupling_type for Hydra config
            _ANALYTIC_COUPLING_MAP = {
                "cubic_rational_coupling":     "cubic_rational",
                "sinh_coupling":               "sinh",
                "cubic_conjugation_coupling":  "cubic_conjugation",
            }
            COUPLING_TYPE = _ANALYTIC_COUPLING_MAP[ENCODER_TYPE]

            # Coupling has an exact inverse → no separate decoder needed
            DECODE_ONLY_RECENT = False

            # ---- Subspace splitting ----
            N_TARGET_DIMS        = dim_99
            KL_NULL_WEIGHT       = "null"
            KL_DYN_WEIGHT        = 0.0001
            RECONSTRUCTION_MODE  = "most_recent"  # 'uniform' | 'harmonic' | 'most_recent'

            # ---- VAE settings ----
            USE_VAE                = True
            VAE_SAMPLE_ALL_LOSSES  = False
            KL_WARMUP_EPOCHS       = 0

            # ---- Tangent space entropy regularizer ----
            TANGENT_ENTROPY_WEIGHT    = 0.0
            TANGENT_ENTROPY_MODE      = "quadratic"  # 'shannon' | 'quadratic' | 'renyi_half'
            TANGENT_ENTROPY_N_SAMPLES = 256

        elif ENCODER_TYPE == "spline_autoregressive":
            # --- Spline Autoregressive encoder (A-RQS, Coccaro et al. arXiv:2302.12024) ---
            # More stable & accurate than coupling splines at higher dimensions.
            # Note: decode is sequential O(D) per layer (MAF-style); encode is parallel via MADE.
            N_AR_LAYERS         = 8      # number of autoregressive layers
            AR_HIDDEN_DIM       = 128    # MADE hidden width
            N_HIDDEN_LAYERS     = 2      # MADE depth
            NUM_BINS            = 8      # rational quadratic spline segments
            TAIL_BOUND          = 3.0    # linear tails outside [-B, B]
            USE_ACTNORM         = True   # ActNorm between autoregressive layers
            ZERO_INIT           = True   # identity initialization trick
            PERMUTATION_SEED    = 42     # base seed for inter-layer permutations
            USE_LOFT            = False  # LOFT layer after all AR layers
            LOFT_TAU            = 100.0  # LOFT threshold

            # Autoregressive flow has an exact inverse → no separate decoder needed
            DECODE_ONLY_RECENT = False

            # ---- Subspace splitting ----
            N_TARGET_DIMS        = dim_99
            KL_NULL_WEIGHT       = "null"
            KL_DYN_WEIGHT        = 0.0001
            RECONSTRUCTION_MODE  = "most_recent"  # 'uniform' | 'harmonic' | 'most_recent'

            # ---- VAE settings ----
            USE_VAE                = True
            VAE_SAMPLE_ALL_LOSSES  = False
            KL_WARMUP_EPOCHS       = 0

            # ---- Tangent space entropy regularizer ----
            TANGENT_ENTROPY_WEIGHT    = 0.0
            TANGENT_ENTROPY_MODE      = "quadratic"  # 'shannon' | 'quadratic' | 'renyi_half'
            TANGENT_ENTROPY_N_SAMPLES = 256

        elif ENCODER_TYPE in _ANALYTIC_AR_TYPES:
            # --- Analytic Autoregressive encoder (Gerdes & Cheng 2026, arXiv:2601.10774) ---
            # Autoregressive architecture with analytic bijections (parallel encode, sequential decode).
            N_AR_LAYERS         = 8      # number of autoregressive layers
            AR_HIDDEN_DIM       = 128    # MADE hidden width
            N_HIDDEN_LAYERS     = 2      # MADE depth
            USE_ACTNORM         = True   # ActNorm between autoregressive layers
            ZERO_INIT           = True   # identity initialization trick
            PERMUTATION_SEED    = 42     # base seed for inter-layer permutations
            USE_LOFT            = False  # LOFT layer after all AR layers
            LOFT_TAU            = 100.0  # LOFT threshold

            # Map ENCODER_TYPE → bijection_type for Hydra config
            _ANALYTIC_AR_MAP = {
                "cubic_rational_autoregressive":     "cubic_rational",
                "sinh_autoregressive":               "sinh",
                "cubic_conjugation_autoregressive":  "cubic_conjugation",
            }
            BIJECTION_TYPE = _ANALYTIC_AR_MAP[ENCODER_TYPE]

            # Autoregressive flow has an exact inverse → no separate decoder needed
            DECODE_ONLY_RECENT = False

            # ---- Subspace splitting ----
            N_TARGET_DIMS        = dim_99
            KL_NULL_WEIGHT       = "null"
            KL_DYN_WEIGHT        = 0.0001
            RECONSTRUCTION_MODE  = "most_recent"  # 'uniform' | 'harmonic' | 'most_recent'

            # ---- VAE settings ----
            USE_VAE                = True
            VAE_SAMPLE_ALL_LOSSES  = False
            KL_WARMUP_EPOCHS       = 0

            # ---- Tangent space entropy regularizer ----
            TANGENT_ENTROPY_WEIGHT    = 0.0
            TANGENT_ENTROPY_MODE      = "quadratic"  # 'shannon' | 'quadratic' | 'renyi_half'
            TANGENT_ENTROPY_N_SAMPLES = 256

        ENCODER_WARMUP_EPOCHS = 5
        DYNAMICS_WARMUP_EPOCHS = 0
        # DYNAMICS_WARMUP_EPOCHS = 5
    return (
        ALPHA_NEG,
        ALPHA_POS,
        AR_HIDDEN_DIM,
        BIJECTION_TYPE,
        CLAMP_TYPE,
        COUPLING_HIDDEN_DIM,
        COUPLING_TYPE,
        DECODER_HIDDEN,
        DECODER_LAYERS,
        DECODE_ONLY_RECENT,
        DYNAMICS_WARMUP_EPOCHS,
        ENCODER_WARMUP_EPOCHS,
        KL_DYN_WEIGHT,
        KL_NULL_WEIGHT,
        KL_WARMUP_EPOCHS,
        LOFT_TAU,
        MLP_DROPOUT,
        MLP_HIDDEN_DIM,
        MLP_N_LAYERS,
        NUM_BINS,
        N_AR_LAYERS,
        N_COUPLING_LAYERS,
        N_HIDDEN_LAYERS,
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
    ## 6b. Pre-trained Encoder Settings

    *Skip this section if `MODE = "from_scratch"`.*
    """)
    return


@app.cell
def _(MODE):
    # ================================================================
    # Pre-trained encoder settings (skip if MODE == "from_scratch")
    # ================================================================
    if MODE == "pretrained":
        from JacobianODE.encoder_only.pretrained import load_pretrained_encoder

        ENCODER_SAVE_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/encoder_runs"

        # Encoder source on W&B
        ENCODER_PROJECT = "Lorenz_MLPFULL_Normed_L6__EncoderOnly"
        ENCODER_RUN_ID  = "vc98usdl"

        FREEZE_ENCODER = False

        adapter, encoder_cfg, encoder_run = load_pretrained_encoder(
            project=ENCODER_PROJECT,
            run_id=ENCODER_RUN_ID,
            save_dir=ENCODER_SAVE_DIR,
            freeze=FREEZE_ENCODER,
            verbose=True,
        )

        print(f"Encoder type:    {type(adapter.encoder).__name__}")
        print(f"n_latent:        {adapter.n_latent}")
        print(f"context_margin:  {adapter.context_margin}")
        print(f"Frozen:          {FREEZE_ENCODER}")
    return FREEZE_ENCODER, adapter, encoder_cfg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Hyperparameters
    """)
    return


@app.cell
def _(
    ALL_INVERTIBLE_TYPES,
    DATA_SOURCE,
    ENCODER_TYPE,
    MODE,
    N_TARGET_DIMS,
    adapter,
):
    # ----------------------------------------------------------------
    # Latent space
    # ----------------------------------------------------------------
    if MODE == "pretrained":
        N_LATENT = adapter.n_latent
    elif ENCODER_TYPE in ALL_INVERTIBLE_TYPES:
        # For coupling/autoregressive encoders, n_latent = n_input (set at runtime from data).
        # The Jacobian MLP operates on N_TARGET_DIMS, not the full n_latent.
        N_LATENT = None  # computed from data dim in the config-building cell
    else:
        if DATA_SOURCE == "dysts":
            # N_LATENT = 3
            N_LATENT = 7
        else:
            N_LATENT = 7

    # ----------------------------------------------------------------
    # Jacobian MLP
    # ----------------------------------------------------------------
    JAC_HIDDEN_DIM = [256, 1024, 2048, 2048]
    JAC_NUM_LAYERS = 4
    JAC_ACTIVATION = 'silu'

    N_EIGVAL_JACOBIANS = 128

    _display_latent = N_LATENT if N_LATENT is not None else f"n_input (coupling, n_target_dims={N_TARGET_DIMS})"
    print(f"N_LATENT: {_display_latent}")
    print(f"Jacobian MLP: {JAC_NUM_LAYERS} layers, hidden_dim={JAC_HIDDEN_DIM}")
    return (
        JAC_ACTIVATION,
        JAC_HIDDEN_DIM,
        JAC_NUM_LAYERS,
        N_EIGVAL_JACOBIANS,
        N_LATENT,
    )


@app.cell
def _(
    ALL_INVERTIBLE_TYPES,
    ENCODER_TYPE,
    FREEZE_ENCODER,
    MODE,
    N_DELAYS,
    x_obs,
):
    # ============================================================
    # Hyperparameters
    TRAJ_INIT_STEPS = 15
    # Rules:
    #   single value    → fixed override (same for every run)
    #   list of values  → swept (one run per combination)
    #
    # Mode-specific params that are not worth sweeping
    # (learn_* weights, noise injection) remain as plain Python
    # vars in the sections below.  To sweep them, add them here
    # and remove them from those sections.
    PREDICTION_STEPS = min((30, x_obs.shape[1] - N_DELAYS - TRAJ_INIT_STEPS))
    if PREDICTION_STEPS < 30:
        print(f'WARNING: Prediction steps truncated to {PREDICTION_STEPS} due to data length')
    PARAMS = {
        'model.prediction_steps': PREDICTION_STEPS,
        'training.batch_size': 32 if MODE == 'from_scratch' else 16,
        'training.trainer_params.max_epochs': 150 if MODE == 'from_scratch' else 200,
        'training.trainer_params.limit_train_batches': 200,
        'training.trainer_params.limit_val_batches': 50,
        'training.trainer_params.accumulate_grad_batches': 4,
        'training.lightning.optimizer_kwargs.lr': 0.0001,
        'training.lightning.optimizer_kwargs.weight_decay': 0.0001,
        'training.lightning.scheduler_type': 'cosine',
        'training.lightning.reconstruction_loss_weight':
            1.0 if MODE == 'from_scratch'
            else 0.0 if MODE == 'pretrained' and FREEZE_ENCODER
            else 1.0,
        'training.lightning.latent_prediction_loss_weight': 1.0,
        'training.lightning.jac_consistency_weight': 0.0,
        'training.lightning.fnn_normalize': True,
        'training.lightning.fnn_elementwise_regularization': True,
        'training.lightning.fnn_use_pca': True,
        'training.lightning.fnn_n_samples': 1024,
        'training.lightning.jacobianODEint_kwargs.traj_init_steps': TRAJ_INIT_STEPS,
        'training.lightning.jacobianODEint_kwargs.interp_pts': 4,
        'training.lightning.jacobianODEint_kwargs.inner_N': 20,
        'training.lightning.jacobianODEint_kwargs.inner_path': 'line',
        'training.early_stopping.early_stopping_patience': 5,
        'training.early_stopping.min_epochs': 10,
        'training.early_stopping.percent_thresh': 0.01,
        'training.model_checkpoint.save_top_k': 1,
        'training.lightning.loop_closure_weight': [
            0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10
        ],
        'training.lightning.fnn_weight': [0],
    }
    if ENCODER_TYPE in ALL_INVERTIBLE_TYPES:
        PARAMS['training.lightning.kl_dyn_weight'] = [0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    if MODE == 'from_scratch':
        LEARN_R2_WEIGHT = False  # ---- JacobianODE model ----
        LEARN_LOOP_CLOSURE_WEIGHT = False
        LEARN_FNN_WEIGHT = False  # "model.prediction_steps":                                     10,
        LEARN_JAC_CONS_WEIGHT = False
        LEARN_JAC_NORM_WEIGHT = False  # ---- Training ----
        LOG_VAR_INIT = 'auto'
    if MODE == 'pretrained':
        OBS_NOISE_SCALE = 0.0
        LATENT_NOISE_SCALE = 0.05
        LATENT_NOISE_PER_STEP = True  # "training.trainer_params.accumulate_grad_batches":            1 if MODE == "from_scratch" else 4,
    # Optional coupling / autoregressive sweep params (uncomment as needed)
    # Optional spline-specific sweep params (uncomment as needed)
    # if ENCODER_TYPE in ("spline_coupling", "spline_autoregressive"):
    #     PARAMS["model.encoder.num_bins"]   = [4, 8, 16]
    #     PARAMS["model.encoder.tail_bound"] = [3.0, 5.0]
    # Optional spline-autoregressive-specific sweep params (uncomment as needed)
    # if ENCODER_TYPE == "spline_autoregressive":
    #     PARAMS["model.encoder.n_layers"]   = [4, 8]
    #     PARAMS["model.encoder.hidden_dim"] = [64, 128, 256]
    # ---- From-scratch-only params (add to PARAMS above to sweep) ----
    # ---- Pretrained-only params (add to PARAMS above to sweep) ----
        PRECOMPUTE_LATENT_NOISE_FACTOR = True  # ---- Loss weights ----  # ---- FNN regularization ----  # ---- JacobianODEint ----  # ---- Early stopping ----  # ---- Checkpoint saving ----  # "training.model_checkpoint.save_top_k": -1,  # ============================================================  # SWEEP PARAMETERS  (list of values = swept)  # ============================================================  # "training.lightning.loop_closure_weight": [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],  # "training.lightning.loop_closure_weight": [1e-5],  # PARAMS["training.lightning.kl_dyn_weight"]  = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]  # PARAMS["training.lightning.kl_dyn_weight"]  = [0.0001]  # PARAMS["training.lightning.kl_null_weight"] = [None, 1.0]  # PARAMS["model.kl_warmup_epochs"]                        = [0, 10]  # PARAMS["training.lightning.tangent_entropy_weight"]     = [0, 1e-3, 1e-2]  # PARAMS["training.lightning.reconstruction_mode"]        = ["uniform", "most_recent"]
    return (
        LEARN_FNN_WEIGHT,
        LEARN_JAC_CONS_WEIGHT,
        LEARN_JAC_NORM_WEIGHT,
        LEARN_LOOP_CLOSURE_WEIGHT,
        LEARN_R2_WEIGHT,
        LOG_VAR_INIT,
        PARAMS,
        PREDICTION_STEPS,
        TRAJ_INIT_STEPS,
    )


@app.cell
def _(MODE, PARAMS, PREDICTION_STEPS, itertools):
    # ============================================================
    # Auto-split PARAMS → fixed overrides + sweep combinations
    FIXED_PARAMS = {_k: _v for _k, _v in PARAMS.items() if not isinstance(_v, list)}
    SWEEP_PARAMS = {_k: _v for _k, _v in PARAMS.items() if isinstance(_v, list)}
    # Any param whose value is a list is swept; everything else is fixed.

    def _param(key, default=None):
        """Return the scalar value for a PARAMS key (first element if list)."""
    # ------------------------------------------------------------
    # Convenience scalar vars needed by downstream cells
        _v = PARAMS.get(key, default)
        return _v[0] if isinstance(_v, list) else _v
    BATCH_SIZE = _param('training.batch_size', 32)
    JAC_WINDOW_STRIDE = 5 if MODE == 'from_scratch' else PREDICTION_STEPS
    sweep_keys = list(SWEEP_PARAMS.keys())
    all_sweep_combos = [dict(zip(sweep_keys, vals)) for vals in itertools.product(*SWEEP_PARAMS.values())]
    n_sweep_combos = len(all_sweep_combos)
    # PREDICTION_STEPS  = _param("model.prediction_steps", 30)
    # TRAJ_INIT_STEPS   = _param("training.lightning.jacobianODEint_kwargs.traj_init_steps", 15)
    print(f'Sweep grid: {n_sweep_combos} total combinations')
    for _k, _v in SWEEP_PARAMS.items():
    # Build sweep combos (cartesian product over all swept params)
        print(f'  {_k}: {_v}')
    return (
        BATCH_SIZE,
        FIXED_PARAMS,
        JAC_WINDOW_STRIDE,
        SWEEP_PARAMS,
        all_sweep_combos,
        n_sweep_combos,
    )


@app.cell
def _(
    ALL_INVERTIBLE_TYPES,
    DATA_SOURCE,
    DECODE_ONLY_RECENT,
    DELAY_SPACING,
    DYNAMICS_WARMUP_EPOCHS,
    ENCODER_TYPE,
    ENCODER_WARMUP_EPOCHS,
    KL_DYN_WEIGHT,
    KL_NULL_WEIGHT,
    MODE,
    NORMALIZE,
    N_DELAYS,
    N_LATENT,
    N_TARGET_DIMS,
    OBSERVED_INDICES_1,
    SWEEP_PARAMS,
    TANGENT_ENTROPY_WEIGHT,
    USE_VAE,
    WANDB_ENTITY,
):
    _obs_idx_label = ''.join((str(i) for i in OBSERVED_INDICES_1)) if OBSERVED_INDICES_1 != 'all' else 'all'
    _data_prefix = 'Lorenz' if DATA_SOURCE == 'dysts' else 'WMTask'
    _warmup = ENCODER_WARMUP_EPOCHS if MODE == 'from_scratch' else 'pt'
    _dyn_warmup = DYNAMICS_WARMUP_EPOCHS if MODE == 'from_scratch' else 0
    if MODE == 'from_scratch':
        if ENCODER_TYPE in ALL_INVERTIBLE_TYPES:
            _latent_label = f'T{N_TARGET_DIMS}'
            _enc_label = {'coupling': 'coupling', 'spline_coupling': 'spline_coupling', 'spline_autoregressive': 'spline_ar', 'cubic_rational_coupling': 'cubic_rat_coupling', 'sinh_coupling': 'sinh_coupling', 'cubic_conjugation_coupling': 'cubic_conj_coupling', 'cubic_rational_autoregressive': 'cubic_rat_ar', 'sinh_autoregressive': 'sinh_ar', 'cubic_conjugation_autoregressive': 'cubic_conj_ar'}[ENCODER_TYPE]
        else:
            _latent_label = f'L{N_LATENT}'
            _enc_label = 'MLP'
        WANDB_PROJECT = f'{_data_prefix}_IND{_obs_idx_label}_N{N_DELAYS}_D{DELAY_SPACING}_Norm{NORMALIZE}_{_latent_label}__{_enc_label}__JacobianODE'
    else:
        WANDB_PROJECT = None
        if WANDB_PROJECT is None:
            WANDB_PROJECT = f'{_data_prefix}_Pretrained_L{N_LATENT}__JacobianODE'
    WANDB_PROJECT_PATH = f'{WANDB_ENTITY}/{WANDB_PROJECT}'
    if ENCODER_TYPE in ALL_INVERTIBLE_TYPES:
        kld_vals = '_'.join((str(_v) for _v in SWEEP_PARAMS['training.lightning.kl_dyn_weight'])) if 'training.lightning.kl_dyn_weight' in SWEEP_PARAMS else KL_DYN_WEIGHT
        kln_vals = '_'.join((str(_v) for _v in SWEEP_PARAMS['training.lightning.kl_null_weight'])) if 'training.lightning.kl_null_weight' in SWEEP_PARAMS else KL_NULL_WEIGHT
        _lc_vals = SWEEP_PARAMS.get('training.lightning.loop_closure_weight', [0])
        WANDB_GROUP = f'sweep_{MODE}_{_enc_label}_lc_{len(_lc_vals)}vals__klN{kln_vals}__klD{kld_vals}__te{TANGENT_ENTROPY_WEIGHT}_enc_warmup_{_warmup}_vae{str(USE_VAE).lower()}'
    else:
        _lc_vals = SWEEP_PARAMS.get('training.lightning.loop_closure_weight', [0])
        _fnn_vals = SWEEP_PARAMS.get('training.lightning.fnn_weight', [0])
        if len(_fnn_vals) > 1:
            WANDB_GROUP = f'sweep_{MODE}_lc+fnn_enc_warmup_{_warmup}_dyn_warmup_{_dyn_warmup}'
        else:
            WANDB_GROUP = f'sweep_{MODE}_lc_only_fnn{_fnn_vals[0]}_enc_warmup_{_warmup}_dyn_warmup_{_dyn_warmup}_recent_{DECODE_ONLY_RECENT}'
    print(f'W&B project: {WANDB_PROJECT_PATH}')
    print(f'W&B group:   {WANDB_GROUP}')
    return WANDB_GROUP, WANDB_PROJECT, WANDB_PROJECT_PATH


@app.cell
def _(BATCH_SIZE, JAC_WINDOW_STRIDE, PREDICTION_STEPS, TRAJ_INIT_STEPS):
    # ----------------------------------------------------------------
    # Sequence length calculation
    # ----------------------------------------------------------------
    JAC_WINDOW = TRAJ_INIT_STEPS + PREDICTION_STEPS
    SEQ_LENGTH = JAC_WINDOW

    n_sub_windows = (SEQ_LENGTH - JAC_WINDOW) // JAC_WINDOW_STRIDE + 1
    print(f"JacobianODE window:      {JAC_WINDOW} = {TRAJ_INIT_STEPS} init + {PREDICTION_STEPS} pred")
    print(f"Dataset sequence length: {SEQ_LENGTH}")
    print(f"Sub-windows per batch:   {n_sub_windows}")
    print(f"Effective samples/step:  {BATCH_SIZE * n_sub_windows}")
    return (SEQ_LENGTH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Config Setup

    Build the base Hydra config. This config is used for local data generation and as the
    template for the sweep command.
    """)
    return


@app.cell
def _(
    ALL_INVERTIBLE_TYPES,
    ALPHA_NEG,
    ALPHA_POS,
    AR_HIDDEN_DIM,
    BIJECTION_TYPE,
    CLAMP_TYPE,
    COUPLING_HIDDEN_DIM,
    COUPLING_TYPE,
    DATA_SOURCE,
    DECODER_HIDDEN,
    DECODER_LAYERS,
    DECODE_ONLY_RECENT,
    DELAY_SPACING,
    DYNAMICS_WARMUP_EPOCHS,
    ENCODER_TYPE,
    ENCODER_WARMUP_EPOCHS,
    FIXED_PARAMS,
    FREEZE_ENCODER,
    JAC_ACTIVATION,
    JAC_HIDDEN_DIM,
    JAC_NUM_LAYERS,
    JAC_WINDOW_STRIDE,
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
    N_AR_LAYERS,
    N_COUPLING_LAYERS,
    N_DELAYS,
    N_EIGVAL_JACOBIANS,
    N_HIDDEN_LAYERS,
    N_LATENT,
    N_PERIODS,
    N_TARGET_DIMS,
    OBS_NOISE,
    PARAMS,
    PERMUTATION_SEED,
    PTS_PER_PERIOD,
    RECONSTRUCTION_MODE,
    SAVE_DIR,
    SCALE_ACTIVATION,
    SCALE_CLAMP,
    SEQ_LENGTH,
    TAIL_BOUND,
    TANGENT_ENTROPY_MODE,
    TANGENT_ENTROPY_N_SAMPLES,
    TANGENT_ENTROPY_WEIGHT,
    USE_ACTNORM,
    USE_LOFT,
    USE_VAE,
    VAE_SAMPLE_ALL_LOSSES,
    WANDB_PROJECT_PATH,
    WMTASK_DATALOADER,
    WMTASK_DIM,
    WMTASK_MODEL_TO_LOAD,
    WMTASK_NAME,
    WMTASK_PROJECT,
    WMTASK_TRAJ_WINDOW,
    ZERO_INIT,
    adapter,
    all_sweep_combos,
    cfg_raw,
    encoder_cfg,
    initialize_config,
    load_config,
):
    _hidden_dim_str = str(JAC_HIDDEN_DIM).replace(' ', '')
    OBSERVED_INDICES_2 = cfg_raw.data.train_test_params.delay_embedding_params.observed_indices
    if OBSERVED_INDICES_2 != 'all':
        OBSERVED_INDICES_2 = list(OBSERVED_INDICES_2)
    _obs_idx_str = str(list(OBSERVED_INDICES_2)).replace(' ', '') if OBSERVED_INDICES_2 != 'all' else 'all'

    def _fmt(val):
        if val is None:
            return 'null'
        if isinstance(val, bool):
            return str(val).lower()
        return str(val)

    # Gather basic overrides
    overrides = [
        f'{_k}={_fmt(_v)}'
        for _k, _v in FIXED_PARAMS.items()
    ]
    for _k, _v in all_sweep_combos[0].items():
        overrides.append(f'{_k}={_fmt(_v)}')

    overrides.append(f'model.params.hidden_dim={_hidden_dim_str}')

    if 'model.params.num_layers' not in PARAMS:
        overrides.append(f'model.params.num_layers={JAC_NUM_LAYERS}')
    if 'model.params.activation' not in PARAMS:
        overrides.append(f'model.params.activation={JAC_ACTIVATION}')

    # Main dataset and logging
    overrides += [
        f'data.postprocessing.obs_noise={OBS_NOISE}',
        f'data.postprocessing.normalize={NORMALIZE}',
        f'data.train_test_params.delay_embedding_params.observed_indices={_obs_idx_str}',
        f'data.train_test_params.delay_embedding_params.n_delays={N_DELAYS}',
        f'data.train_test_params.delay_embedding_params.delay_spacing={DELAY_SPACING}',
        f'data.train_test_params.seq_length={SEQ_LENGTH}',
        f'training.logger.save_dir={SAVE_DIR}',
    ]

    overrides += [
        'training.lightning.loop_closure_training=True',
        'training.lightning.trajectory_training=True',
        'training.lightning.alpha_teacher_forcing=1',
        'training.lightning.teacher_forcing_annealing=True',
        'training.lightning.gamma_teacher_forcing=0.999',
    ]

    if DATA_SOURCE == 'dysts':
        overrides += [
            'data=dysts',
            'data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz',
            f'data.trajectory_params.n_periods={N_PERIODS}',
            f'data.trajectory_params.pts_per_period={PTS_PER_PERIOD}',
            f'data.trajectory_params.num_ics={NUM_ICS}',
        ]
    else:
        overrides += [
            'data=wmtask',
            f'data.dataset_loader.project={WMTASK_PROJECT}',
            f'data.dataset_loader.name={WMTASK_NAME}',
            f'data.dataset_loader.model_to_load={WMTASK_MODEL_TO_LOAD}',
            f'data.dataset_loader.dataloader_to_use={WMTASK_DATALOADER}',
            f'data.dataset_loader.traj_window={WMTASK_TRAJ_WINDOW}',
            f'data.flow.dim={WMTASK_DIM}',
        ]

    if MODE == 'from_scratch':
        _n_raw_obs = (
            len(OBSERVED_INDICES_2)
            if OBSERVED_INDICES_2 != 'all'
            else 3 if DATA_SOURCE == 'dysts' else WMTASK_DIM
        )
        _n_input = N_DELAYS * _n_raw_obs
        if ENCODER_TYPE == 'mlp':
            overrides += [
                'model=latent_mlp',
                f'model.encoder.n_input={_n_input}',
                f'model.encoder.n_latent={N_LATENT}',
                f'model.encoder.hidden_dim={MLP_HIDDEN_DIM}',
                f'model.encoder.n_layers={MLP_N_LAYERS}',
                f'model.encoder.dropout={MLP_DROPOUT}',
                f'model.encoder.decoder_hidden={DECODER_HIDDEN}',
                f'model.encoder.decoder_layers={DECODER_LAYERS}',
                f'model.encoder.decoder_n_output={_n_raw_obs if DECODE_ONLY_RECENT else "null"}',
                'model.encoder.context_margin=0',
            ]
        elif ENCODER_TYPE == 'coupling':
            overrides += [
                'model=latent_coupling',
                f'model.encoder.n_input={_n_input}',
                f'model.encoder.n_coupling_layers={N_COUPLING_LAYERS}',
                f'model.encoder.hidden_dim={COUPLING_HIDDEN_DIM}',
                f'model.encoder.n_hidden_layers={N_HIDDEN_LAYERS}',
                f'model.encoder.scale_activation={SCALE_ACTIVATION}',
                f'model.encoder.scale_clamp={SCALE_CLAMP}',
                f'model.encoder.zero_init={ZERO_INIT}',
                f'model.encoder.permutation_seed={PERMUTATION_SEED}',
                f'model.encoder.clamp_type={CLAMP_TYPE}',
                f'model.encoder.alpha_pos={ALPHA_POS}',
                f'model.encoder.alpha_neg={ALPHA_NEG}',
                f'model.encoder.use_loft={USE_LOFT}',
                f'model.encoder.loft_tau={LOFT_TAU}',
                f'model.n_target_dims={N_TARGET_DIMS}',
                f'model.use_vae={USE_VAE}',
                f'model.vae_sample_all_losses={VAE_SAMPLE_ALL_LOSSES}',
                f'model.kl_warmup_epochs={KL_WARMUP_EPOCHS}',
                f'training.lightning.kl_null_weight={KL_NULL_WEIGHT}',
                f'training.lightning.reconstruction_mode={RECONSTRUCTION_MODE}',
                f'training.lightning.tangent_entropy_weight={TANGENT_ENTROPY_WEIGHT}',
                f'training.lightning.tangent_entropy_mode={TANGENT_ENTROPY_MODE}',
                f'training.lightning.tangent_entropy_n_samples={TANGENT_ENTROPY_N_SAMPLES}',
            ]
        elif ENCODER_TYPE == 'spline_coupling':
            overrides += [
                'model=latent_spline_coupling',
                f'model.encoder.n_input={_n_input}',
                f'model.encoder.n_coupling_layers={N_COUPLING_LAYERS}',
                f'model.encoder.hidden_dim={COUPLING_HIDDEN_DIM}',
                f'model.encoder.n_hidden_layers={N_HIDDEN_LAYERS}',
                f'model.encoder.num_bins={NUM_BINS}',
                f'model.encoder.tail_bound={TAIL_BOUND}',
                f'model.encoder.use_actnorm={str(USE_ACTNORM).lower()}',
                f'model.encoder.zero_init={str(ZERO_INIT).lower()}',
                f'model.encoder.permutation_seed={PERMUTATION_SEED}',
                f'model.encoder.use_loft={str(USE_LOFT).lower()}',
                f'model.encoder.loft_tau={LOFT_TAU}',
                f'model.n_target_dims={N_TARGET_DIMS}',
                f'model.use_vae={USE_VAE}',
                f'model.vae_sample_all_losses={VAE_SAMPLE_ALL_LOSSES}',
                f'model.kl_warmup_epochs={KL_WARMUP_EPOCHS}',
                f'training.lightning.kl_null_weight={KL_NULL_WEIGHT}',
                f'training.lightning.reconstruction_mode={RECONSTRUCTION_MODE}',
                f'training.lightning.n_eigval_jacobians={N_EIGVAL_JACOBIANS}',
                f'training.lightning.tangent_entropy_weight={TANGENT_ENTROPY_WEIGHT}',
                f'training.lightning.tangent_entropy_mode={TANGENT_ENTROPY_MODE}',
                f'training.lightning.tangent_entropy_n_samples={TANGENT_ENTROPY_N_SAMPLES}',
            ]
        elif ENCODER_TYPE in _ANALYTIC_COUPLING_TYPES:
            _model_cfg = f'latent_{COUPLING_TYPE}_coupling'
            overrides += [
                f'model={_model_cfg}',
                f'model.encoder.n_input={_n_input}',
                f'model.encoder.n_coupling_layers={N_COUPLING_LAYERS}',
                f'model.encoder.hidden_dim={COUPLING_HIDDEN_DIM}',
                f'model.encoder.n_hidden_layers={N_HIDDEN_LAYERS}',
                f'model.encoder.use_actnorm={str(USE_ACTNORM).lower()}',
                f'model.encoder.zero_init={str(ZERO_INIT).lower()}',
                f'model.encoder.permutation_seed={PERMUTATION_SEED}',
                f'model.encoder.use_loft={str(USE_LOFT).lower()}',
                f'model.encoder.loft_tau={LOFT_TAU}',
                f'model.n_target_dims={N_TARGET_DIMS}',
                f'model.use_vae={USE_VAE}',
                f'model.vae_sample_all_losses={VAE_SAMPLE_ALL_LOSSES}',
                f'model.kl_warmup_epochs={KL_WARMUP_EPOCHS}',
                f'training.lightning.kl_null_weight={KL_NULL_WEIGHT}',
                f'training.lightning.reconstruction_mode={RECONSTRUCTION_MODE}',
                f'training.lightning.tangent_entropy_weight={TANGENT_ENTROPY_WEIGHT}',
                f'training.lightning.tangent_entropy_mode={TANGENT_ENTROPY_MODE}',
                f'training.lightning.tangent_entropy_n_samples={TANGENT_ENTROPY_N_SAMPLES}',
            ]
        elif ENCODER_TYPE == 'spline_autoregressive':
            overrides += [
                'model=latent_spline_autoregressive',
                f'model.encoder.n_input={_n_input}',
                f'model.encoder.n_layers={N_AR_LAYERS}',
                f'model.encoder.hidden_dim={AR_HIDDEN_DIM}',
                f'model.encoder.n_hidden_layers={N_HIDDEN_LAYERS}',
                f'model.encoder.num_bins={NUM_BINS}',
                f'model.encoder.tail_bound={TAIL_BOUND}',
                f'model.encoder.use_actnorm={str(USE_ACTNORM).lower()}',
                f'model.encoder.zero_init={str(ZERO_INIT).lower()}',
                f'model.encoder.permutation_seed={PERMUTATION_SEED}',
                f'model.encoder.use_loft={str(USE_LOFT).lower()}',
                f'model.encoder.loft_tau={LOFT_TAU}',
                f'model.n_target_dims={N_TARGET_DIMS}',
                f'model.use_vae={USE_VAE}',
                f'model.vae_sample_all_losses={VAE_SAMPLE_ALL_LOSSES}',
                f'model.kl_warmup_epochs={KL_WARMUP_EPOCHS}',
                f'training.lightning.kl_null_weight={KL_NULL_WEIGHT}',
                f'training.lightning.reconstruction_mode={RECONSTRUCTION_MODE}',
                f'training.lightning.tangent_entropy_weight={TANGENT_ENTROPY_WEIGHT}',
                f'training.lightning.tangent_entropy_mode={TANGENT_ENTROPY_MODE}',
                f'training.lightning.tangent_entropy_n_samples={TANGENT_ENTROPY_N_SAMPLES}',
            ]
        elif ENCODER_TYPE in _ANALYTIC_AR_TYPES:
            _model_cfg = f'latent_{BIJECTION_TYPE}_autoregressive'
            overrides += [
                f'model={_model_cfg}',
                f'model.encoder.n_input={_n_input}',
                f'model.encoder.n_layers={N_AR_LAYERS}',
                f'model.encoder.hidden_dim={AR_HIDDEN_DIM}',
                f'model.encoder.n_hidden_layers={N_HIDDEN_LAYERS}',
                f'model.encoder.use_actnorm={str(USE_ACTNORM).lower()}',
                f'model.encoder.zero_init={str(ZERO_INIT).lower()}',
                f'model.encoder.permutation_seed={PERMUTATION_SEED}',
                f'model.encoder.use_loft={str(USE_LOFT).lower()}',
                f'model.encoder.loft_tau={LOFT_TAU}',
                f'model.n_target_dims={N_TARGET_DIMS}',
                f'model.use_vae={USE_VAE}',
                f'model.vae_sample_all_losses={VAE_SAMPLE_ALL_LOSSES}',
                f'model.kl_warmup_epochs={KL_WARMUP_EPOCHS}',
                f'training.lightning.kl_null_weight={KL_NULL_WEIGHT}',
                f'training.lightning.reconstruction_mode={RECONSTRUCTION_MODE}',
                f'training.lightning.tangent_entropy_weight={TANGENT_ENTROPY_WEIGHT}',
                f'training.lightning.tangent_entropy_mode={TANGENT_ENTROPY_MODE}',
                f'training.lightning.tangent_entropy_n_samples={TANGENT_ENTROPY_N_SAMPLES}',
            ]
        overrides += [
            f'model.decode_only_recent={DECODE_ONLY_RECENT}',
            f'model.jac_window_stride={JAC_WINDOW_STRIDE}',
            f'model.encoder_warmup_epochs={ENCODER_WARMUP_EPOCHS}',
            f'model.dynamics_warmup_epochs={DYNAMICS_WARMUP_EPOCHS}',
            f'training.lightning.learn_r2_weight={LEARN_R2_WEIGHT}',
            f'training.lightning.learn_loop_closure_weight={LEARN_LOOP_CLOSURE_WEIGHT}',
            f'training.lightning.learn_fnn_weight={LEARN_FNN_WEIGHT}',
            f'training.lightning.learn_jac_cons_weight={LEARN_JAC_CONS_WEIGHT}',
            f'training.lightning.learn_jac_norm_weight={LEARN_JAC_NORM_WEIGHT}',
            f'training.lightning.log_var_init={LOG_VAR_INIT}',
            f'training.logger_save_dirs={SAVE_DIR}',
        ]
        config_name = 'config'
    elif MODE == 'pretrained':
        _enc_cls = type(adapter.encoder).__name__
        if 'MLP' in _enc_cls:
            MODEL_CONFIG = 'latent_mlp'
        elif 'SSM' in _enc_cls or 'LRU' in _enc_cls:
            MODEL_CONFIG = 'latent_ssm'
        elif 'Transformer' in _enc_cls:
            MODEL_CONFIG = 'latent_transformer'
        elif 'TCN' in _enc_cls:
            MODEL_CONFIG = 'latent_tcn'
        elif 'AnalyticAutoregressive' in _enc_cls:
            _bijtype = getattr(adapter.encoder, 'bijection_type', 'cubic_rational')
            MODEL_CONFIG = f'latent_{_bijtype}_autoregressive'
        elif 'SplineAutoregressive' in _enc_cls:
            MODEL_CONFIG = 'latent_spline_autoregressive'
        elif 'Coupling' in _enc_cls or 'Affine' in _enc_cls:
            _ctype = getattr(adapter.encoder, 'coupling_type', None)
            if _ctype and _ctype in ('cubic_rational', 'sinh', 'cubic_conjugation'):
                MODEL_CONFIG = f'latent_{_ctype}_coupling'
            else:
                MODEL_CONFIG = 'latent_coupling'
        else:
            MODEL_CONFIG = 'latent_ssm'
        _enc = encoder_cfg.model.encoder
        _encoder_overrides = [
            f'model.encoder.n_latent={N_LATENT}'
        ]
        if MODEL_CONFIG == 'latent_ssm':
            _encoder_overrides += [
                f'model.encoder.d_model={int(_enc.get("d_model", 64))}',
                f'model.encoder.d_state={int(_enc.get("d_state", 64))}',
                f'model.encoder.n_layers={int(_enc.get("n_layers", 3))}',
                f'model.encoder.ffn_expand={int(_enc.get("ffn_expand", 2))}',
                f'model.encoder.r_min={float(_enc.get("r_min", 0.0))}',
                f'model.encoder.r_max={float(_enc.get("r_max", 0.99))}',
                f'model.encoder.dropout={float(_enc.get("dropout", 0.1))}',
                f'model.encoder.use_positional_encoding={bool(_enc.get("use_positional_encoding", True))}',
                f'model.encoder.positional_encoding_type={_enc.get("positional_encoding_type", "sinusoidal")}',
                f'model.encoder.decoder_hidden={int(_enc.get("decoder_hidden", 128))}',
                f'model.encoder.decoder_layers={int(_enc.get("decoder_layers", 2))}',
                f'model.encoder.context_margin={adapter.context_margin}',
            ]
        elif MODEL_CONFIG == 'latent_transformer':
            _encoder_overrides += [
                f'model.encoder.d_model={int(_enc.get("d_model", 64))}',
                f'model.encoder.n_heads={int(_enc.get("n_heads", 4))}',
                f'model.encoder.n_layers={int(_enc.get("n_layers", 3))}',
                f'model.encoder.dim_feedforward={int(_enc.get("dim_feedforward", 128))}',
                f'model.encoder.dropout={float(_enc.get("dropout", 0.1))}',
                f'model.encoder.decoder_hidden={int(_enc.get("decoder_hidden", 128))}',
                f'model.encoder.decoder_layers={int(_enc.get("decoder_layers", 2))}',
                f'model.encoder.context_margin={adapter.context_margin}',
            ]
        elif MODEL_CONFIG == 'latent_tcn':
            _encoder_overrides += [
                f'model.encoder.n_channels={int(_enc.get("n_channels", 64))}',
                f'model.encoder.kernel_size={int(_enc.get("kernel_size", 7))}',
                f'model.encoder.n_layers={int(_enc.get("n_layers", 4))}',
                f'model.encoder.dropout={float(_enc.get("dropout", 0.1))}',
                f'model.encoder.decoder_hidden={int(_enc.get("decoder_hidden", 128))}',
                f'model.encoder.decoder_layers={int(_enc.get("decoder_layers", 2))}',
                f'model.encoder.context_margin={adapter.context_margin}',
            ]
        elif MODEL_CONFIG == 'latent_coupling':
            _encoder_overrides = [
                f'model.encoder.n_input={_enc.get("n_input", "null")}',
                f'model.encoder.n_coupling_layers={int(_enc.get("n_coupling_layers", 8))}',
                f'model.encoder.hidden_dim={int(_enc.get("hidden_dim", 128))}',
                f'model.encoder.n_hidden_layers={int(_enc.get("n_hidden_layers", 2))}',
                f'model.encoder.scale_activation={_enc.get("scale_activation", "tanh")}',
                f'model.encoder.scale_clamp={float(_enc.get("scale_clamp", 3.0))}',
                f'model.encoder.zero_init={bool(_enc.get("zero_init", True))}',
                f'model.encoder.permutation_seed={int(_enc.get("permutation_seed", 0))}',
                f'model.encoder.clamp_type={_enc.get("clamp_type", "symmetric")}',
                f'model.encoder.alpha_pos={float(_enc.get("alpha_pos", 0.1))}',
                f'model.encoder.alpha_neg={float(_enc.get("alpha_neg", 2.0))}',
            ]
        elif MODEL_CONFIG == 'latent_spline_autoregressive':
            _encoder_overrides = [
                f'model.encoder.n_input={_enc.get("n_input", "null")}',
                f'model.encoder.n_layers={int(_enc.get("n_layers", 8))}',
                f'model.encoder.hidden_dim={int(_enc.get("hidden_dim", 128))}',
                f'model.encoder.n_hidden_layers={int(_enc.get("n_hidden_layers", 2))}',
                f'model.encoder.num_bins={int(_enc.get("num_bins", 8))}',
                f'model.encoder.tail_bound={float(_enc.get("tail_bound", 3.0))}',
                f'model.encoder.use_actnorm={bool(_enc.get("use_actnorm", True))}',
                f'model.encoder.zero_init={bool(_enc.get("zero_init", True))}',
                f'model.encoder.permutation_seed={int(_enc.get("permutation_seed", 0))}',
            ]
        elif (
            MODEL_CONFIG.startswith('latent_')
            and MODEL_CONFIG.endswith('_autoregressive')
            and (MODEL_CONFIG != 'latent_spline_autoregressive')
        ):
            _encoder_overrides = [
                f'model.encoder.n_input={_enc.get("n_input", "null")}',
                f'model.encoder.n_layers={int(_enc.get("n_layers", 8))}',
                f'model.encoder.hidden_dim={int(_enc.get("hidden_dim", 128))}',
                f'model.encoder.n_hidden_layers={int(_enc.get("n_hidden_layers", 2))}',
                f'model.encoder.use_actnorm={bool(_enc.get("use_actnorm", True))}',
                f'model.encoder.zero_init={bool(_enc.get("zero_init", True))}',
                f'model.encoder.permutation_seed={int(_enc.get("permutation_seed", 0))}',
            ]
        elif (
            MODEL_CONFIG.startswith('latent_')
            and MODEL_CONFIG.endswith('_coupling')
            and (MODEL_CONFIG not in ('latent_coupling', 'latent_spline_coupling'))
        ):
            _encoder_overrides = [
                f'model.encoder.n_input={_enc.get("n_input", "null")}',
                f'model.encoder.n_coupling_layers={int(_enc.get("n_coupling_layers", 8))}',
                f'model.encoder.hidden_dim={int(_enc.get("hidden_dim", 128))}',
                f'model.encoder.n_hidden_layers={int(_enc.get("n_hidden_layers", 2))}',
                f'model.encoder.use_actnorm={bool(_enc.get("use_actnorm", True))}',
                f'model.encoder.zero_init={bool(_enc.get("zero_init", True))}',
                f'model.encoder.permutation_seed={int(_enc.get("permutation_seed", 0))}',
            ]
        else:
            _encoder_overrides += [
                f'model.encoder.hidden_dim={int(_enc.get("hidden_dim", 256))}',
                f'model.encoder.n_layers={int(_enc.get("n_layers", 3))}',
                f'model.encoder.dropout={float(_enc.get("dropout", 0.1))}',
                f'model.encoder.decoder_hidden={int(_enc.get("decoder_hidden", 128))}',
                f'model.encoder.decoder_layers={int(_enc.get("decoder_layers", 2))}',
            ]
        overrides += (
            [
                f'model={MODEL_CONFIG}',
                f'model.encoder.n_input={(_n_input if MODE == "from_scratch" else int(_enc.get("n_input", 0)))}',
            ]
            + _encoder_overrides
            + [
                f'model.jac_window_stride={JAC_WINDOW_STRIDE}',
                f'training.lightning.freeze_encoder={FREEZE_ENCODER}',
            ]
        )
        config_name = 'config'

    cfg = load_config(config_name=config_name, overrides=overrides)
    cfg = initialize_config(cfg)
    print(f'\nConfig built successfully.')
    print(f'  Lightning target: {cfg.training.lightning._target_}')
    print(f'  Jacobian MLP: input_dim={cfg.model.params.input_dim}, output_dim={cfg.model.params.output_dim}')
    if 'encoder' in cfg.model:
        print(f'  Encoder: n_input={cfg.model.encoder.n_input}')
        if ENCODER_TYPE in ALL_INVERTIBLE_TYPES:
            print(f'  n_target_dims={cfg.model.get("n_target_dims", "N/A")}')
    if hasattr(cfg.model, 'use_vae') and cfg.model.use_vae:
        print(f'  VAE: enabled')
    print(f'  N_DELAYS={N_DELAYS}, DELAY_SPACING={DELAY_SPACING}')
    print(f'  OBSERVED_INDICES: {_obs_idx_str[:60]}{("..." if len(_obs_idx_str) > 60 else "")}')
    print(f'  W&B project: {WANDB_PROJECT_PATH}')
    return cfg, overrides


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Generate Data

    Generate trajectories, apply postprocessing (noise + normalization), and build dataloaders
    with the final delay embedding parameters.
    """)
    return


@app.cell
def _(cfg, make_trajectories, seed_everything):
    seed_everything(cfg.data.flow.random_state)
    _eq, sol_1, _dt = make_trajectories(cfg, verbose=True)
    print(f'\nFull trajectory shape: {sol_1['values'].shape}')
    print(f'Time step dt = {_dt:.4f}')
    return (sol_1,)


@app.cell
def _(
    ALL_INVERTIBLE_TYPES,
    ENCODER_TYPE,
    N_LATENT,
    N_TARGET_DIMS,
    cfg,
    create_dataloaders,
    np,
    postprocess_data,
    sol_1,
):
    result = postprocess_data(cfg, sol_1['values'])
    _values = result.values
    mu = result.mu
    sigma = result.sigma
    noise_scale_factor = result.noise_scale_factor
    cfg.data.postprocessing.mu = float(mu)
    cfg.data.postprocessing.sigma = float(sigma)
    cfg.data.postprocessing.noise_scale_factor = float(noise_scale_factor)
    _train_dl, _val_dl, _test_dl, _trajs = create_dataloaders(cfg, _values, verbose=True, return_full_obs=True)
    n_obs = _trajs['train_trajs'].sequence.shape[-1]
    n_dims = _values.shape[-1]
    _n_dyn = (
        N_TARGET_DIMS
        if ENCODER_TYPE in ALL_INVERTIBLE_TYPES
        else (N_LATENT if N_LATENT is not None else N_TARGET_DIMS)
    )
    print(f'\nn_obs = {n_obs}, n_dims = {n_dims}')
    print(f'C2 loop-closure threshold: sqrt(n_dyn) = {np.sqrt(_n_dyn):.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Launch Sweep

    Uses Hydra `--multirun` with the SLURM launcher. Checks W&B for already-completed
    runs and only launches jobs for remaining parameter combinations.

    > **Wait for all SLURM jobs to finish** before running the companion Analytics notebook.
    """)
    return


@app.cell
def _():
    def get_sweep_param_from_run(run, key):
        """Extract a sweep parameter value from a W&B run config."""
        parts = key.split('.')
        val = run.config
        for p in parts:
            if not isinstance(val, dict) or p not in val:
                return None
            val = val[p]
        return val

    def combo_matches_run(combo, run):
        """Check if a parameter combination matches a finished W&B run."""
        for key, target_val in combo.items():
            run_val = get_sweep_param_from_run(run, key)
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

    def is_jac_ode_run(run):
        """Return True if this run has an encoder (i.e. is a JacobianODE run)."""
        return 'model' in run.config and 'encoder' in run.config.get('model', {})

    return combo_matches_run, is_jac_ode_run


@app.cell
def _(
    WANDB_GROUP,
    WANDB_PROJECT_PATH,
    all_sweep_combos,
    combo_matches_run,
    is_jac_ode_run,
    wandb,
):
    api = wandb.Api()
    try:
        run_filters = {'group': WANDB_GROUP} if WANDB_GROUP else None
        existing_runs = api.runs(WANDB_PROJECT_PATH, filters=run_filters)
        msg = f'Found {len(existing_runs)} total runs in {WANDB_PROJECT_PATH}'
        if WANDB_GROUP:
            msg = msg + f' (group={WANDB_GROUP})'
        print(msg)
    except Exception as e:
        print(f'Could not query project (may not exist yet): {e}')
        existing_runs = []
    finished_runs = [r for r in existing_runs if r.state == 'finished' and is_jac_ode_run(r)]
    already_done = []
    remaining_combos = []
    for combo in all_sweep_combos:
        if any((combo_matches_run(combo, r) for r in finished_runs)):
            already_done.append(combo)
            run_match = next((r for r in finished_runs if combo_matches_run(combo, r)))
            print(f'  Already done: {combo} (run_id={run_match.id})')
        else:
            remaining_combos.append(combo)
    if already_done:
        print(f'\nSkipping {len(already_done)} already-completed combinations')
    if remaining_combos:
        print(f'Need to run {len(remaining_combos)} combinations')
    else:
        print('\nAll parameter combinations already have finished runs -- no sweep needed!')
    return (remaining_combos,)


@app.cell
def _(
    MODE,
    SWEEP_PARAMS,
    WANDB_ENTITY,
    WANDB_GROUP,
    WANDB_PROJECT,
    n_sweep_combos,
    overrides,
    remaining_combos,
    sys,
):
    # ----------------------------------------------------------------
    # Build the Hydra --multirun sweep command(s)
    ENTRY_POINT = f'{sys.executable} -m JacobianODE.jacobians.run_jacobians' if MODE == 'from_scratch' else f'{sys.executable} -m JacobianODE.jacobians.run_pretrained_jacobians'

    def hydra_val(val):
        """Format a Python value for a Hydra command-line override."""
        if val is None:
            return 'null'
        if isinstance(val, bool):
            return str(val).lower()
        return val
    if remaining_combos:
        _swept_keys = set(SWEEP_PARAMS.keys())

        def _is_swept(ov):
            for _k in _swept_keys:
                if ov.strip().startswith(_k + '='):
                    return True
            return False  # Base overrides: exclude keys that are being swept
        fixed_overrides = [ov for ov in overrides if not _is_swept(ov)]
        n_remaining = len(remaining_combos)
        use_subset_sweep = n_remaining < n_sweep_combos
        if use_subset_sweep:
            sweep_cmds = []
            for _combo in remaining_combos:
                combo_overrides = [f'{key}={hydra_val(val)}' for key, val in _combo.items()]
                sweep_overrides = fixed_overrides + combo_overrides + [f'wandb_entity={WANDB_ENTITY}', f'wandb_project={WANDB_PROJECT}', 'slurm=default']
                if WANDB_GROUP:
                    sweep_overrides.append(f'wandb_group={WANDB_GROUP}')
                sweep_cmds.append(f'{ENTRY_POINT} --multirun ' + ' '.join(sweep_overrides))
        else:
            sweep_param_overrides = []  # One command per remaining combo
            for key, vals in SWEEP_PARAMS.items():
                val_str = ','.join((str(hydra_val(_v)) for _v in vals))
                sweep_param_overrides.append(f'{key}={val_str}')
            sweep_overrides = fixed_overrides + sweep_param_overrides + [f'wandb_entity={WANDB_ENTITY}', f'wandb_project={WANDB_PROJECT}', 'slurm=default']
            if WANDB_GROUP:
                sweep_overrides.append(f'wandb_group={WANDB_GROUP}')
            sweep_cmds = [f'{ENTRY_POINT} --multirun ' + ' '.join(sweep_overrides)]
        sweep_type = 'subset (1 cmd per combo)' if use_subset_sweep else 'grid'
        print(f'Sweep: {n_remaining} jobs ({sweep_type})')
        print(f'\nFirst command:\n{sweep_cmds[0][:200]}...')
    else:
        sweep_cmds = []
        print('No sweep to launch -- all runs already completed.')  # Full grid sweep: comma-separated values
    return ENTRY_POINT, sweep_cmds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Timing analysis

    | Label | What it measures |
    |---|---|
    | `train/1.encode_tf+compute_jacs` | No-grad encode + compute Jacobians for teacher forcing update |
    | `train/2.trajectory_step` | Full `trajectory_model_step` (encode → ODE → decode → loss) |
    | `train/3.encode_recon` | Second encode pass shared by recon, FNN, loop closure, KL |
    | `train/4.loop_closure` | Loop closure step in latent space |
    | `train/5.reconstruction_loss` | decode(encode(x)) ≈ x loss |
    | `train/6.fnn_loss` | False nearest-neighbor regularizer |
    | `train/7.jac_consistency` | Jacobian consistency loss ($‖e^{Jdt} dz_t − dz_{t+1}‖²)$|
    | `train/8.tangent_entropy` | Tangent space entropy via autograd Jacobians (0 if weight=0) |
    | `traj/1.encode` | Encoder forward pass inside `trajectory_model_step` |
    | `traj/2.window_gather` | Sub-window parameter computation |
    | `traj/2b.window_stack` | Python loop to gather and stack latent sub-windows |
    | `traj/3.jacobianODEint` | JacobianODE integration (likely dominant cost) |
    | `traj/4.decode` | Crop predicted latent + decoder forward pass |
    | `traj/5.obs_targets` | Extract observation-space targets from raw batch |
    | `traj/6.loss_and_metrics` | Obs-space loss + MASE/R²/MAE metrics |
    | `val/1.trajectory_step` | Full `trajectory_model_step` at validation (no noise, fixed α) |
    | `val/2.encode` | Second encode pass for loop closure / recon / eigval diagnostics |
    | `val/3.loop_closure` | Loop closure step |
    | `val/4.reconstruction_loss` | Reconstruction loss (no grad) |
    | `val/5.one_step_traj` | **Second full trajectory pass** with α=1 for MASE diagnostic |
    | `val/6.eigvals` | `compute_jacobians` + `torch.linalg.eigvals` on all B×T Jacobians |
    | `val/7.log_metrics` | W&B / Lightning metric logging |
    """)
    return


@app.cell
def _(
    ENTRY_POINT,
    SWEEP_PARAMS,
    WANDB_ENTITY,
    WANDB_GROUP,
    WANDB_PROJECT,
    all_sweep_combos,
    overrides,
    subprocess,
):
    # ----------------------------------------------------------------
    # Run one config locally (no SLURM) — exceptions show in notebook
    RUN_DIAG_LOCAL = False
    # Set to True to run; uses all_sweep_combos[DIAG_COMBO_IX]
    DIAG_COMBO_IX = 0  # <-- Set True to diagnose

    def _hydra_val(val):
        if val is None:
            return 'null'
        if isinstance(val, bool):
            return str(val).lower()
        return val

    if RUN_DIAG_LOCAL:
        _swept_keys = set(SWEEP_PARAMS.keys())

        def _is_swept(ov):
            for _k in _swept_keys:
                if ov.strip().startswith(_k + '='):
                    return True
            return False
        _fixed = [ov for ov in overrides if not _is_swept(ov)]
        _combo = all_sweep_combos[DIAG_COMBO_IX]
        _combo_ov = [f'{_k}={_hydra_val(_v)}' for _k, _v in _combo.items()]
        _local_overrides = _fixed + _combo_ov + [f'wandb_entity={WANDB_ENTITY}', f'wandb_project={WANDB_PROJECT}', 'slurm=none']
        if WANDB_GROUP:
            _local_overrides.append(f'wandb_group={WANDB_GROUP}')
        _ENTRY_POINT = ENTRY_POINT
        _cmd = f'{_ENTRY_POINT} ' + ' '.join(_local_overrides)
        print('Running locally (no SLURM). Any exception will appear below:\n')  # No SLURM launcher — runs in-process (Hydra default)
        print(_cmd[:300] + '...\n')
        subprocess.run(_cmd, shell=True)
    else:
        print('RUN_DIAG_LOCAL=False. Set to True and re-run to diagnose one config locally.')
    return


@app.cell
def _(WANDB_GROUP, WANDB_PROJECT):
    print(f"WANDB_PROJECT = \"{WANDB_PROJECT}\"")
    print(f"WANDB_GROUP = \"{WANDB_GROUP}\"")
    return


@app.cell
def _(subprocess, sweep_cmds):
    # Launch the sweep (submits SLURM jobs)
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
    ## Next Steps

    Once all SLURM jobs have finished, use the companion
    **Sweep Analytics (Latent JacobianODE)** notebook to:

    1. Collect results from W&B
    2. Apply physics-informed model selection (C1/C2/C3)
    3. Load and diagnose the best model

    Set the same `WANDB_PROJECT` and `WANDB_GROUP` in that notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
