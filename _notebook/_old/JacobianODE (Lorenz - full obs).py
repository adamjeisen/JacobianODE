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
    # JacobianODE — Fully Observed 3D Lorenz

    This notebook tests the **vanilla JacobianODE** model (no latent encoder) on the **fully observed** Lorenz system:
    - **Input**: Fully observed 3D Lorenz system (all 3 coordinates visible)
    - **Model**: MLP that directly predicts Jacobians from state vectors
    - **No encoder/decoder** — the Jacobian MLP operates directly on observation space

    ## Two Validation Tests

    1. **Lyapunov Exponent Test** — Predicted Jacobians → QR-decomposition Lyapunov exponents
       should match the known Lorenz spectrum λ ≈ (+0.91, 0.0, −14.57).

    2. **Jacobian Test** — The predicted Jacobians should match the true analytical Lorenz Jacobian:

       $$J_{\mathrm{pred}}(x) \approx J_{\mathrm{true}}(x)$$

       where
       $$
       J_{\mathrm{true}}(x) = \begin{pmatrix}
       -\sigma & \sigma & 0 \\
       \rho - x_3 & -1 & -x_1 \\
       x_2 & x_1 & -\beta
       \end{pmatrix}
       $$
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
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm

    return OmegaConf, np, plt, torch, tqdm


@app.cell
def _():
    from JacobianODE.jacobians import (
        load_config,
        initialize_config,
        seed_everything,
        make_trajectories,
        postprocess_data,
        create_dataloaders,
        make_model,
        train_model,
    )

    return (
        create_dataloaders,
        initialize_config,
        load_config,
        make_model,
        make_trajectories,
        postprocess_data,
        seed_everything,
        train_model,
    )


@app.cell
def _(seed_everything):
    seed_everything(42)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Hyperparameters

    ### Data
    We simulate standard Lorenz (σ=10, ρ=28, β=8/3) and observe **all three** dimensions
    with **no observation noise** — the cleanest possible test.
    """)
    return


@app.cell
def _():
    # ----------------------------------------------------------------
    # Data hyperparameters
    # ----------------------------------------------------------------
    OBS_NOISE      = 0.05           # Small noise
    OBSERVED_INDICES = [0, 1, 2]  # Observe ALL three Lorenz dimensions
    N_PERIODS      = 15            # Lorenz periods to simulate
    PTS_PER_PERIOD = 100           # Temporal resolution (dt ≈ 0.015)
    NUM_ICS        = 32            # Independent initial conditions

    # Known Lorenz Lyapunov exponents (σ=10, ρ=28, β=8/3)
    TRUE_LYAPUNOV  = [0.91, 0.0, -14.57]

    SAVE_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/vanilla_jac_runs"
    return (
        NUM_ICS,
        N_PERIODS,
        OBSERVED_INDICES,
        OBS_NOISE,
        PTS_PER_PERIOD,
        SAVE_DIR,
        TRUE_LYAPUNOV,
    )


@app.cell
def _():
    # ----------------------------------------------------------------
    # Jacobian MLP hyperparameters
    # ----------------------------------------------------------------
    # Input dim = 3 (full obs), Output dim = 3^2 = 9.
    JAC_HIDDEN_DIM = [512, 512, 512]
    JAC_NUM_LAYERS  = 3
    JAC_ACTIVATION  = 'silu'
    return JAC_ACTIVATION, JAC_HIDDEN_DIM, JAC_NUM_LAYERS


@app.cell
def _():
    # ----------------------------------------------------------------
    # Training hyperparameters
    # ----------------------------------------------------------------
    BATCH_SIZE               = 32
    MAX_EPOCHS               = 150
    LIMIT_TRAIN_BATCHES      = 200   # steps per epoch
    LIMIT_VAL_BATCHES        = 50
    ACCUMULATE_GRAD_BATCHES  = 1
    LEARNING_RATE            = 1e-4
    WEIGHT_DECAY             = 1e-4

    TRAJ_INIT_STEPS          = 15   # Init steps for JacobianODEint
    LOOP_CLOSURE_WEIGHT      = 0
    INTERP_PTS               = 4
    INNER_N                  = 20

    EARLY_STOPPING_PATIENCE  = 5
    return (
        ACCUMULATE_GRAD_BATCHES,
        BATCH_SIZE,
        EARLY_STOPPING_PATIENCE,
        INNER_N,
        INTERP_PTS,
        LEARNING_RATE,
        LIMIT_TRAIN_BATCHES,
        LIMIT_VAL_BATCHES,
        LOOP_CLOSURE_WEIGHT,
        MAX_EPOCHS,
        TRAJ_INIT_STEPS,
        WEIGHT_DECAY,
    )


@app.cell
def _(TRAJ_INIT_STEPS):
    # ----------------------------------------------------------------
    # Sequence length
    # ----------------------------------------------------------------
    # For vanilla JacobianODE, the sequence length is the full trajectory
    # window used for training. The model needs TRAJ_INIT_STEPS for
    # initialization, then predicts the remaining steps.
    SEQ_LENGTH = 25

    print(f"Dataset sequence length: {SEQ_LENGTH}")
    print(f"Init steps:             {TRAJ_INIT_STEPS}")
    print(f"Prediction steps:       {SEQ_LENGTH - TRAJ_INIT_STEPS}")
    return (SEQ_LENGTH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Config Setup

    We use the default `model=mlp` config (no encoder). The Jacobian MLP maps
    directly from the 3D state to 9 Jacobian entries.
    """)
    return


@app.cell
def _(
    ACCUMULATE_GRAD_BATCHES,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    INNER_N,
    INTERP_PTS,
    JAC_ACTIVATION,
    JAC_HIDDEN_DIM,
    JAC_NUM_LAYERS,
    LEARNING_RATE,
    LIMIT_TRAIN_BATCHES,
    LIMIT_VAL_BATCHES,
    LOOP_CLOSURE_WEIGHT,
    MAX_EPOCHS,
    NUM_ICS,
    N_PERIODS,
    OBSERVED_INDICES,
    OBS_NOISE,
    PTS_PER_PERIOD,
    SAVE_DIR,
    SEQ_LENGTH,
    TRAJ_INIT_STEPS,
    WEIGHT_DECAY,
    initialize_config,
    load_config,
):
    overrides = [
        # --- Model: vanilla MLP (no encoder) ---
        "model=mlp",
        f"model.params.hidden_dim={JAC_HIDDEN_DIM}",
        f"model.params.num_layers={JAC_NUM_LAYERS}",
        f"model.params.activation={JAC_ACTIVATION}",

        # --- Data: fully observed Lorenz ---
        "data=dysts",
        "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
        f"data.trajectory_params.n_periods={N_PERIODS}",
        f"data.trajectory_params.pts_per_period={PTS_PER_PERIOD}",
        f"data.trajectory_params.num_ics={NUM_ICS}",
        f"data.postprocessing.obs_noise={OBS_NOISE}",
        "data.postprocessing.normalize=true",
        f"data.train_test_params.delay_embedding_params.observed_indices={list(OBSERVED_INDICES)}",
        "data.train_test_params.delay_embedding_params.n_delays=1",

        # --- Sequence length ---
        f"data.train_test_params.seq_length={SEQ_LENGTH}",

        # --- Save directory ---
        f"training.logger.save_dir={SAVE_DIR}",
        f"training.logger_save_dirs={SAVE_DIR}",

        # --- Training hyperparameters ---
        f"training.batch_size={BATCH_SIZE}",
        f"training.lightning.optimizer_kwargs.lr={LEARNING_RATE}",
        f"training.lightning.optimizer_kwargs.weight_decay={WEIGHT_DECAY}",
        f"training.lightning.loop_closure_weight={LOOP_CLOSURE_WEIGHT}",
        "training.lightning.loop_closure_training=True",
        f"training.lightning.jacobianODEint_kwargs.traj_init_steps={TRAJ_INIT_STEPS}",
        f"training.lightning.jacobianODEint_kwargs.interp_pts={INTERP_PTS}",
        f"training.lightning.jacobianODEint_kwargs.inner_N={INNER_N}",
        f"training.trainer_params.max_epochs={MAX_EPOCHS}",
        f"training.trainer_params.limit_train_batches={LIMIT_TRAIN_BATCHES}",
        f"training.trainer_params.limit_val_batches={LIMIT_VAL_BATCHES}",
        f"training.trainer_params.accumulate_grad_batches={ACCUMULATE_GRAD_BATCHES}",
        f"training.early_stopping.early_stopping_patience={EARLY_STOPPING_PATIENCE}",
    ]

    cfg = load_config(overrides=overrides)
    cfg = initialize_config(cfg)
    return (cfg,)


@app.cell
def _(OmegaConf, cfg):
    print(OmegaConf.to_yaml(cfg))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Generate Trajectories and Create DataLoaders
    """)
    return


@app.cell
def _(cfg, make_trajectories):
    eq, sol, dt = make_trajectories(cfg, verbose=True)
    print(f"\nFull trajectory shape: {sol['values'].shape}")
    print(f"Time step dt = {dt:.4f}")
    return dt, eq, sol


@app.cell
def _(cfg, postprocess_data, sol):
    # Postprocess: add noise (scaled by data magnitude) and z-score normalize.
    # postprocess_data returns a PostprocessResult namedtuple with
    # (values, mu, sigma, noise_scale_factor).
    _result = postprocess_data(cfg, sol['values'])
    values = _result.values
    mu = _result.mu
    sigma = _result.sigma
    noise_scale_factor = _result.noise_scale_factor
    print(f'Postprocessed shape: {values.shape}')
    print(f'Normalization: mu={mu:.4f}, sigma={sigma:.4f}')
    print(f'Noise scale factor: {noise_scale_factor:.4f}')
    return mu, noise_scale_factor, sigma, values


@app.cell
def _(cfg, create_dataloaders, values):
    train_dl, val_dl, test_dl, trajs = create_dataloaders(cfg, values, verbose=True)

    sample_batch = next(iter(train_dl))
    print(f"\nBatch shape: {sample_batch.shape}")
    print(f"  (B={sample_batch.shape[0]}, T={sample_batch.shape[1]}, D={sample_batch.shape[2]})")
    return sample_batch, train_dl, val_dl


@app.cell
def _(SEQ_LENGTH, plt, sample_batch, sol, values):
    _fig = plt.figure(figsize=(14, 4))
    ax1 = _fig.add_subplot(131, projection='3d')
    # 3D attractor (raw)
    traj0 = sol['values'][0]
    ax1.plot(traj0[:, 0], traj0[:, 1], traj0[:, 2], lw=0.4, alpha=0.7)
    ax1.set_title('Lorenz Attractor (raw)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    # Normalized attractor
    _ax2 = _fig.add_subplot(132, projection='3d')
    traj_n = values[0]  # normalized, shape (T, 3)
    _ax2.plot(traj_n[:, 0], traj_n[:, 1], traj_n[:, 2], lw=0.4, alpha=0.7, color='orange')
    _ax2.set_title('Lorenz Attractor (normalized)')
    _ax2.set_xlabel("x'")
    _ax2.set_ylabel("y'")
    # Sample training sequences
    _ax2.set_zlabel("z'")
    ax3 = _fig.add_subplot(133)
    for dim, color, label in zip(range(3), ['C0', 'C1', 'C2'], ['x', 'y', 'z']):
        ax3.plot(sample_batch[0, :, dim].numpy(), color=color, lw=1.0, label=label)
    ax3.set_xlabel('t (window)')
    ax3.set_ylabel('normalized value')
    ax3.set_title(f'Sample Training Window (len={SEQ_LENGTH})')
    ax3.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Create the Vanilla JacobianODE Model
    """)
    return


@app.cell
def _(TRUE_LYAPUNOV, cfg, dt, eq, make_model, mu, noise_scale_factor, sigma):
    lit_model = make_model(cfg, dt=dt, eq=eq, mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor, verbose=True)

    print(f"\nModel type:         {type(lit_model).__name__}")
    print(f"Jacobian MLP:       {type(lit_model.model).__name__}")
    print(f"Direct prediction:  {lit_model.direct}")
    print(f"dt:                 {lit_model.dt:.4f}")
    print(f"True Lyapunov:      {TRUE_LYAPUNOV}")

    jac_params = sum(p.numel() for p in lit_model.model.parameters())
    print(f"\nParameters — Jacobian MLP: {jac_params:,}")
    return (lit_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Sanity Check: Forward Pass
    """)
    return


@app.cell
def _(lit_model, sample_batch, torch):
    lit_model.eval()
    with torch.no_grad():
        jacs = lit_model.compute_jacobians(sample_batch)  # 1. Predict Jacobians directly from observations
        print(f'Jacobians shape:       {jacs.shape}')  # (B, T, 3, 3)
        _result = lit_model.trajectory_model_step(sample_batch)
        print(f'\nTrajectory step loss:  {_result['loss'].item():.6f}')
        print(f'Metrics:               {_result['metric_vals']}')  # 2. Full trajectory step
    lit_model.train()
    loss = lit_model.training_step(sample_batch, batch_idx=0)
    loss.backward()
    jac_grad = max((p.grad.abs().max().item() for p in lit_model.model.parameters() if p.grad is not None))
    # Quick gradient check
    print(f'\nTraining loss:         {loss.item():.6f}')
    print(f'Max Jacobian MLP grad: {jac_grad:.2e}')
    lit_model.zero_grad()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Train the Model
    """)
    return


@app.cell
def _():
    import os, wandb
    try:
        wandb.finish(quiet=True)
    except Exception:
        pass
    return


@app.cell
def _(
    cfg,
    dt,
    eq,
    make_model,
    mu,
    noise_scale_factor,
    sigma,
    train_dl,
    train_model,
    val_dl,
):
    # Re-create model to reset any gradient state from the sanity check above.
    lit_model_1 = make_model(cfg, dt=dt, eq=eq, mu=mu, sigma=sigma, noise_scale_factor=noise_scale_factor, verbose=False)
    RUN_NAME = f'vanilla_jac_lorenz_3d_full'
    train_model(cfg=cfg, lit_model=lit_model_1, train_dataloaders=train_dl, val_dataloaders=val_dl, name=RUN_NAME, project='VanillaLorenz__JacobianODE', entity='JacobianODE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. Load Model and Test
    """)
    return


@app.cell
def _():
    run_id = 'FILL_IN'
    return (run_id,)


@app.cell
def _(SAVE_DIR, create_dataloaders, run_id):
    # ----------------------------------------------------------------
    # Load the trained run from Weights & Biases
    from JacobianODE.jacobians import load_run, load_checkpoint
    # load_run now handles postprocessing (noise + normalization) internally
    # and returns already-processed values.
    PROJECT_PATH = f'JacobianODE/VanillaLorenz__JacobianODE'
    run, cfg_1, eq_1, dt_1, values_1, _, _, _, _, lit_model_2 = load_run(PROJECT_PATH, run_id=run_id, save_dir=SAVE_DIR, generate_data=True, verbose=True)
    train_dl_1, val_dl_1, test_dl_1, trajs_1 = create_dataloaders(cfg_1, values_1, verbose=True)
    # W&B path: entity/project
    # ---- Step 1: rebuild architecture + trajectory data (postprocessed) ----
    # ---- Step 2: dataloaders from postprocessed data ----
    # ---- Step 3: load best checkpoint ----
    load_checkpoint(run, cfg_1, lit_model_2, save_dir=SAVE_DIR, verbose=True)
    return dt_1, lit_model_2, test_dl_1, trajs_1, val_dl_1


@app.cell
def _(np, test_dl_1):
    sample_batch_1 = test_dl_1.dataset.sequence[np.random.choice(np.arange(len(test_dl_1.dataset.sequence)), size=(128,), replace=False)]
    return (sample_batch_1,)


@app.cell
def _(dt_1, lit_model_2, sample_batch_1, torch, trajs_1):
    # PROCESS SAMPLE BATCH
    with torch.no_grad():
        traj_ret = lit_model_2.trajectory_model_step(sample_batch_1, alpha_teacher_forcing=0.0)
        targets_full = sample_batch_1
        outputs = traj_ret['outputs']
    test_trajs = trajs_1['test_trajs'].sequence
    # PROCESS TEST TRAJS
    with torch.no_grad():
        jacs_full = lit_model_2.compute_jacobians(test_trajs)
        from JacobianODE.models.latent_jacobian import LitLatentJacobianODE
        lyaps_full = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_full, dt_1)  # Compute Lyapunov exponents
    return LitLatentJacobianODE, lyaps_full, outputs, targets_full


@app.cell
def _(lyaps_full):
    lyaps_full.mean(axis=0)
    return


@app.cell
def _(lit_model_2, outputs, plt, targets_full):
    from JacobianODE.jacobians.metrics import r2_score
    dim_names = ['x', 'y', 'z']
    traj_init_steps = lit_model_2.jacobianODEint_kwargs.get('traj_init_steps', 15)
    pred_portion = outputs[:, traj_init_steps:, :]
    true_portion = targets_full[:, traj_init_steps:, :]
    _plot_ind = ((true_portion - pred_portion) ** 2).mean(axis=(1, 2)).argmax()
    for _i in range(3):
        plt.plot(true_portion[_plot_ind, :, _i].numpy(), c=f'C{_i}', label=f'true {dim_names[_i]}')
        plt.plot(pred_portion[_plot_ind, :, _i].numpy(), c=f'C{_i}', linestyle='--', label=f'predicted {dim_names[_i]}')
    plt.legend()
    _r2_val = r2_score(true_portion.reshape(-1, 3), pred_portion.reshape(-1, 3))
    plt.title(f'True vs. Predicted (worst case)\n$R^2 = {_r2_val:.4f}$')
    plt.show()
    return dim_names, pred_portion, r2_score, true_portion


@app.cell
def _(dim_names, plt, pred_portion, r2_score, true_portion):
    # Plot predictions vs true (best case)
    _plot_ind = ((true_portion - pred_portion) ** 2).mean(axis=(1, 2)).argmin()
    for _i in range(3):
        plt.plot(true_portion[_plot_ind, :, _i].numpy(), c=f'C{_i}', label=f'true {dim_names[_i]}')
        plt.plot(pred_portion[_plot_ind, :, _i].numpy(), c=f'C{_i}', linestyle='--', label=f'predicted {dim_names[_i]}')
    plt.legend()
    _r2_val = r2_score(true_portion.reshape(-1, 3), pred_portion.reshape(-1, 3))
    plt.title(f'True vs. Predicted (best case)\n$R^2 = {_r2_val:.4f}$')
    plt.show()
    return


@app.cell
def _(lit_model_2, val_dl_1):
    from JacobianODE.jacobians.tuning.criteria import compute_persistence_baseline, compute_one_step_error
    lit_model_3 = lit_model_2.to('cpu')
    one_step_error = compute_one_step_error(lit_model_3, val_dl_1, n_batches=10, verbose=True)
    lit_model_3.to('cpu')
    return compute_one_step_error, compute_persistence_baseline, lit_model_3


@app.cell
def _(compute_one_step_error, lit_model_3, val_dl_1):
    lit_model_4 = lit_model_3.to('cuda')
    one_step_error_1 = compute_one_step_error(lit_model_4, val_dl_1, n_batches=30, verbose=True)
    lit_model_4 = lit_model_4.to('cpu')
    return lit_model_4, one_step_error_1


@app.cell
def _(one_step_error_1):
    one_step_error_1
    return


@app.cell
def _(compute_persistence_baseline, trajs_1):
    compute_persistence_baseline(trajs_1['train_trajs'].sequence, normalize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Test 1 — Lyapunov Exponent Comparison

    We compute the Jacobians from the learned Jacobian MLP directly on validation
    trajectories and then run the standard QR-decomposition Lyapunov algorithm.

    Because the true state is **3-dimensional** and fully observed, all three Lyapunov
    exponents should match the Lorenz spectrum.

    Known Lorenz values: **λ₁ ≈ +0.91, λ₂ ≈ 0.0, λ₃ ≈ −14.57**
    """)
    return


@app.cell
def _(lit_model_4, torch, tqdm, val_dl_1):
    lit_model_4.eval()
    all_obs = []
    # Collect all validation data
    with torch.no_grad():
        n_iters = 50
        iterator = tqdm(total=n_iters)
        for batch in val_dl_1:
            all_obs.append(batch)
            iterator.update(1)
            if iterator.n >= n_iters:
                break
        iterator.close()
    obs_all = torch.cat(all_obs, dim=0)
    print(f'Validation data bank: {obs_all.shape}')  # (N_val, T, 3)
    return (obs_all,)


@app.cell
def _(LitLatentJacobianODE, TRUE_LYAPUNOV, dt_1, lit_model_4, obs_all, torch):
    N_TRAJ_LYAP = min(30, obs_all.shape[0])
    obs_long = obs_all[:N_TRAJ_LYAP].reshape(-1, 3).unsqueeze(0)
    with torch.no_grad():
        jacs_long = lit_model_4.compute_jacobians(obs_long)[0]
        pred_lyap = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_long, dt_1)
    print('Predicted Lyapunov exponents:')
    for _i, le in enumerate(pred_lyap):
        print(f'  λ_{_i + 1} = {le.item():+.4f}')
    print('\nTrue Lorenz Lyapunov exponents:')
    for _i, le in enumerate(TRUE_LYAPUNOV):
        print(f'  λ_{_i + 1} = {le:+.4f}')
    true_t = torch.tensor(TRUE_LYAPUNOV)
    lyap_mse = ((pred_lyap - true_t) ** 2).mean().item()
    print(f'\nLyapunov MSE: {lyap_mse:.4f}')
    return lyap_mse, pred_lyap


@app.cell
def _(TRUE_LYAPUNOV, lyap_mse, np, plt, pred_lyap):
    N_LATENT = 3  # dimension
    _fig, _ax = plt.subplots(figsize=(6, 4))
    x = np.arange(N_LATENT)
    pred_np = pred_lyap.numpy()
    true_np = np.array(TRUE_LYAPUNOV)
    _ax.bar(x - 0.18, pred_np, width=0.35, label='Predicted', alpha=0.8, color='C0')
    _ax.bar(x + 0.18, true_np, width=0.35, label='True (Lorenz)', alpha=0.8, color='C1')
    _ax.axhline(0, color='k', lw=0.7, ls='--')
    _ax.set_xticks(x)
    _ax.set_xticklabels([f'λ_{_i + 1}' for _i in range(N_LATENT)])
    _ax.set_ylabel('Lyapunov exponent')
    _ax.set_title(f'Lyapunov Spectrum  (MSE={lyap_mse:.4f})')
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Test 2 — Predicted Jacobian vs True Lorenz Jacobian

    We verify that the learned Jacobian MLP recovers the true analytical Lorenz Jacobian
    directly (no encoder/decoder composition needed).

    $$J_{\mathrm{pred}}(x) \approx J_{\mathrm{true}}(x)$$

    **Lorenz Jacobian** at $x = (x_1, x_2, x_3)$ (in normalized coords = same as raw since normalization is a global scalar):
    $$
    J_{\mathrm{true}}(x) = \begin{pmatrix}
    -\sigma & \sigma & 0 \\
    \rho - x_3 & -1 & -x_1 \\
    x_2 & x_1 & -\beta
    \end{pmatrix}
    $$
    """)
    return


@app.cell
def _(torch):
    # ---------------------------------------------------------------
    # Helper: analytical Lorenz Jacobian
    # Note: Since normalization is x_norm = (x - mu)/sigma (global scalar),
    # J_lorenz(x_norm) == J_lorenz(x_raw).  Pass denormalized x.
    # ---------------------------------------------------------------
    def lorenz_jacobian(x_raw, sigma_lz=10.0, rho=28.0, beta=8.0/3.0):
        """Analytical Lorenz Jacobian at x_raw = (x1, x2, x3)."""
        x1, x2, x3 = float(x_raw[0]), float(x_raw[1]), float(x_raw[2])
        return torch.tensor([
            [-sigma_lz, sigma_lz,    0.0 ],
            [rho - x3,  -1.0,       -x1  ],
            [x2,         x1,        -beta],
        ], dtype=torch.float32)

    return (lorenz_jacobian,)


@app.cell
def _(lit_model_4, lorenz_jacobian, mu, sigma, torch, tqdm, trajs_1):
    # ---------------------------------------------------------------
    # Compute predicted vs true Jacobians on a test trajectory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit_model_5 = lit_model_4.to(device)
    lit_model_5.eval()
    test_seq = trajs_1['test_trajs'].sequence[:1].to(device)
    T_test = test_seq.shape[1]
    # Use one long test sequence

    def to_device(x, device):
        if torch.is_tensor(x):
    # --- sigma and mu may be float or tensor; ensure they're tensors on device
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)
    sigma_device = to_device(sigma, device)
    mu_device = to_device(mu, device)
    STEP = 2
    T_START = 5
    t_indices = range(T_START, T_test - 1, STEP)
    with torch.no_grad():
    # Evaluate every STEP-th timestep
        jacs_pred_all = lit_model_5.compute_jacobians(test_seq)[0]
    J_preds = []
    J_trues = []
    for t_idx in tqdm(t_indices):
    # Compute predicted Jacobians for the full sequence
        x_t_n = test_seq[0, t_idx].detach()
        J_pred = jacs_pred_all[t_idx].cpu()  # (T, 3, 3)
        x_raw = x_t_n * sigma_device + mu_device
        J_true = lorenz_jacobian(x_raw.cpu())
        J_preds.append(J_pred)
        J_trues.append(J_true)
    J_preds = torch.stack(J_preds)
    J_trues = torch.stack(J_trues)
    print(f'Computed {len(t_indices)} Jacobian comparisons')  # --- Predicted Jacobian (directly from MLP) ---  # (3, 3)  # --- True Lorenz Jacobian (in raw/denormalized coords) ---  # denormalize  # (N_t, 3, 3)
    return J_preds, J_trues, t_indices


@app.cell
def _(J_preds, J_trues, np, r2_score, torch):
    # ---------------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------------
    frob_errors = torch.norm(J_preds - J_trues, dim=(-2, -1))   # (N_t,)
    frob_true   = torch.norm(J_trues,           dim=(-2, -1))
    rel_errors  = frob_errors / (frob_true + 1e-8)

    # Pearson correlation across all (i,j) entries and timesteps
    pred_flat = J_preds.reshape(-1).numpy()
    true_flat = J_trues.reshape(-1).numpy()
    corr = float(np.corrcoef(pred_flat, true_flat)[0, 1])
    r2_jac = r2_score(torch.tensor(true_flat).unsqueeze(0), torch.tensor(pred_flat).unsqueeze(0))

    print("Jacobian test results")
    print("=" * 38)
    print(f"Frobenius error  — mean: {frob_errors.mean():.4f}  |  std: {frob_errors.std():.4f}")
    print(f"Relative error   — mean: {rel_errors.mean():.4f}  |  std: {rel_errors.std():.4f}")
    print(f"Entry correlation:       {corr:.4f}")
    print(f"R^2:                     {r2_jac:.4f}")

    # Per-entry breakdown
    print("\nPer-entry mean absolute error:")
    mae_matrix = (J_preds - J_trues).abs().mean(dim=0).numpy()
    print(np.round(mae_matrix, 4))

    print("\nMean predicted Jacobian:")
    print(np.round(J_preds.mean(dim=0).numpy(), 3))

    print("\nMean true Lorenz Jacobian:")
    print(np.round(J_trues.mean(dim=0).numpy(), 3))
    return corr, frob_errors, pred_flat, true_flat


@app.cell
def _(corr, frob_errors, plt, pred_flat, t_indices, true_flat):
    # ---------------------------------------------------------------
    # Scatter plot: predicted vs true entries  (all timesteps, all 9 entries)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _ax = _axes[0]
    _ax.scatter(true_flat, pred_flat, s=2, alpha=0.3, color='C0', rasterized=True)
    # --- Scatter plot ---
    lim = max(abs(true_flat).max(), abs(pred_flat).max()) * 1.05
    _ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.0, label='ideal')
    _ax.set_xlabel('$J_{\\mathrm{true}}$ entries')
    _ax.set_ylabel('$J_{\\mathrm{pred}}$ entries')
    _ax.set_title(f'Predicted vs True (r = {corr:.3f})')
    _ax.legend()
    _ax2 = _axes[1]
    _ax2.plot(list(t_indices), frob_errors.numpy(), lw=1.0, alpha=0.8)
    _ax2.axhline(frob_errors.mean().item(), color='r', ls='--', lw=1.0, label=f'mean = {frob_errors.mean():.3f}')
    # --- Frobenius error over time ---
    _ax2.set_xlabel('Timestep in test window')
    _ax2.set_ylabel('Frobenius error')
    _ax2.set_title('$\\|J_{\\mathrm{pred}} - J_{\\mathrm{true}}\\|_F$')
    _ax2.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(J_preds, J_trues, np, plt):
    # ---------------------------------------------------------------
    # Per-entry (i,j) scatter:  one subplot per Jacobian entry
    _fig, _axes = plt.subplots(3, 3, figsize=(11, 10))
    row_labels = ['x', 'y', 'z']
    col_labels = ['x', 'y', 'z']
    for _i in range(3):
        for j in range(3):
            _ax = _axes[_i, j]
            p = J_preds[:, _i, j].numpy()
            t = J_trues[:, _i, j].numpy()
            r = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-08 and np.std(t) > 1e-08 else float('nan')
            _ax.scatter(t, p, s=4, alpha=0.5, color='C0', rasterized=True)
            mn, mx = (min(t.min(), p.min()), max(t.max(), p.max()))
            pad = (mx - mn) * 0.05 if mx > mn else 0.1
            _ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], 'k--', lw=0.8)
            _ax.set_title(f'({row_labels[_i]},{col_labels[j]})  r={r:.2f}')
            _ax.set_xlabel('true')
            _ax.set_ylabel('pred')
    _fig.suptitle('Predicted $J_{\\mathrm{pred}}$ vs $J_{\\mathrm{true}}$ — per entry', y=1.01)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(J_preds, J_trues, frob_errors, plt, t_indices):
    # ---------------------------------------------------------------
    # Heatmap comparison at a single representative timestep
    mid = len(t_indices) // 2
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.5))
    vabs = max(J_trues[mid].abs().max().item(), J_preds[mid].abs().max().item())
    im0 = _axes[0].imshow(J_trues[mid].numpy(), vmin=-vabs, vmax=vabs, cmap='RdBu_r')
    im1 = _axes[1].imshow(J_preds[mid].numpy(), vmin=-vabs, vmax=vabs, cmap='RdBu_r')
    diff = (J_preds[mid] - J_trues[mid]).numpy()
    im2 = _axes[2].imshow(diff, cmap='RdBu_r', vmin=-abs(diff).max(), vmax=abs(diff).max())
    for _ax, im, title in zip(_axes, [im0, im1, im2], ['$J_{\\mathrm{true}}$', '$J_{\\mathrm{pred}}$', 'Difference']):
        _ax.set_title(title, fontsize=12)
        _ax.set_xticks(range(3))
        _ax.set_yticks(range(3))
        _ax.set_xticklabels(['x', 'y', 'z'])
        _ax.set_yticklabels(['x', 'y', 'z'])
        plt.colorbar(im, ax=_ax, fraction=0.046, pad=0.04)
    t_shown = list(t_indices)[mid]
    _fig.suptitle(f'Jacobian heatmaps at t={t_shown}  (Frob. error = {frob_errors[mid]:.4f})', y=1.03)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Summary

    | Test | Metric | Target | Result |
    |------|--------|--------|--------|
    | **Lyapunov exponents** | MSE vs true | < 1.0 | `lyap_mse` above |
    | **Predicted Jacobian** | Entry correlation | > 0.9 | `corr` above |
    | **Predicted Jacobian** | Mean relative Frob. error | < 0.3 | `rel_errors.mean()` above |

    The vanilla JacobianODE directly learns the Jacobian field in observation space.
    Since the system is fully observed (3D → 3D), no encoder/decoder is needed.

    ### Hyperparameter Reference

    | Category | Parameter | Value | Notes |
    |----------|-----------|-------|-------|
    | **System** | Observed dims | 3/3 | Fully observed |
    | **JacMLP** | `hidden_dim` | [512,512,512] | 3D system |
    | **JacMLP** | `activation` | silu | |
    | **JacODE** | `traj_init_steps` | 15 | Init steps |
    | **Batch** | `SEQ_LENGTH` | 50 | Window length |
    | **Batch** | `BATCH_SIZE` | 32 | Batch size |
    """)
    return


if __name__ == "__main__":
    app.run()
