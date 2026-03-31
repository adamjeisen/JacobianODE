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
    # Sweep Analytics — Latent JacobianODE

    Post-training diagnostics and model selection for a JacobianODE hyperparameter sweep.

    **Inputs:** W&B project (and optional group) containing completed sweep runs.

    **Workflow:**
    1. Collect W&B runs and apply physics-informed model selection (C1/C2/C3)
    2. Visualize selection criteria
    3. Load the best model
    4. Run diagnostics: reconstruction quality, prediction quality, Lyapunov exponents,
       latent space analysis, smoothness, Jacobian diagnostics, FNN loss, and more.
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
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    from omegaconf import OmegaConf
    import os
    import pandas as pd
    import torch
    from torch.autograd.functional import jacobian as autograd_jacobian
    from torch.utils.data import RandomSampler, DataLoader
    from tqdm.auto import tqdm
    import wandb

    from JacobianODE.jacobians import (
        load_config,
        initialize_config,
        seed_everything,
        make_trajectories,
        postprocess_data,
        create_dataloaders,
        load_run,
        load_checkpoint,
        select_best_model,
        DiagnosticMetrics,
    )
    from JacobianODE.jacobians.metrics import r2_score, normalized_mse as nmse_fn, mase
    from JacobianODE.jacobians.tuning import select_from_wandb_runs
    from JacobianODE.jacobians.lightning_base import loop_closure
    from JacobianODE.models.latent_jacobian import LitLatentJacobianODE
    from JacobianODE.fnn import loss_false, loss_amplification

    torch.set_float32_matmul_precision('high')
    return (
        DataLoader,
        LitLatentJacobianODE,
        OmegaConf,
        RandomSampler,
        create_dataloaders,
        load_checkpoint,
        load_run,
        loss_amplification,
        mase,
        math,
        nmse_fn,
        np,
        plt,
        r2_score,
        select_from_wandb_runs,
        torch,
        tqdm,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Settings

    Specify the W&B project, optional group, and save directory for checkpoints.
    """)
    return


@app.cell
def _():
    # ================================================================
    # W&B project and group (EDIT THESE)
    # ================================================================
    WANDB_ENTITY  = "JacobianODE"
    # WANDB_PROJECT = "Lorenz_IND012_N1_D1_NormTrue_L3__JacobianODE"  # <-- CHANGE THIS
    # WANDB_GROUP   = "MLP_lc_sweep_enc_warmup_5"  # Set to a string to filter by group, or None for all runs

    # WANDB_PROJECT = "Lorenz_IND012_N1_D1_NormTrue_L6__JacobianODE"
    # WANDB_GROUP   = "sweep_from_scratch_lc_only_enc_warmup_5"
    # WANDB_GROUP   = "sweep_from_scratch_lc_only_fnn1e-5_enc_warmup_5"

    # WANDB_PROJECT = "Lorenz_IND0_N25_D1_NormTrue_L3__JacobianODE"
    # WANDB_GROUP   = "sweep_from_scratch_lc_only_fnn0_enc_warmup_5"

    # WANDB_PROJECT   = "WMTask_INDall_N1_D1_NormTrue_L7__JacobianODE"
    # WANDB_GROUP     = "sweep_from_scratch_lc_only_fnn0_enc_warmup_5"

    # WANDB_PROJECT   = "Lorenz_IND0_N100_D1_NormTrue_L5__JacobianODE"
    # WANDB_GROUP     = "sweep_from_scratch_lc_only_fnn0_enc_warmup_5"

    # WANDB_PROJECT   = "Lorenz_IND0_N4_D11_NormTrue_L3__JacobianODE"
    # WANDB_GROUP     = "sweep_from_scratch_lc_only_fnn0_enc_warmup_5"

    # WANDB_PROJECT   = "Lorenz_IND0_N25_D1_NormTrue_L3__JacobianODE"
    # WANDB_GROUP     = "sweep_from_scratch_lc_only_fnn0_enc_warmup_5_recent_True"

    # WANDB_PROJECT   = "Lorenz_IND0_N25_D1_NormTrue_L3__JacobianODE"
    # WANDB_GROUP     = "sweep_from_scratch_lc_only_fnn0_enc_warmup_0_recent_True"

    # WANDB_PROJECT   = "Lorenz_IND012_N1_D1_NormTrue_T3__Coupling__JacobianODE"
    # WANDB_GROUP     = "sweep_from_scratch_Coupling_lc+zp1.0_enc_warmup_5"

    # WANDB_PROJECT   = "Lorenz_IND012_N2_D1_NormTrue_T3__Coupling__JacobianODE"
    # WANDB_GROUP     = "sweep_from_scratch_Coupling_lc+zp1.0_enc_warmup_5"

    # WANDB_PROJECT = "Lorenz_IND012_N2_D1_NormTrue_T6__Coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_Coupling_lc+zp1.0_enc_warmup_5"

    # WANDB_PROJECT = "Lorenz_IND012_N2_D1_NormTrue_T6__Coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_Coupling_lc_9vals__zp1.0__tenc0.0_enc_warmup_5"

    # # bad initialization
    # WANDB_PROJECT = "Lorenz_IND0_N25_D1_NormTrue_T3__Coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_coupling_lc_9vals__zp1.0__tenc0.0_enc_warmup_5"

    # # better initialization
    # WANDB_PROJECT = "Lorenz_IND0_N25_D1_NormTrue_T3__coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_coupling_lc_9vals__zp1.0__te0.0_enc_warmup_5"

    # # no Z-null penalty
    # WANDB_PROJECT = "Lorenz_IND0_N25_D1_NormTrue_T3__coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_coupling_lc_9vals__zp0.0__te0.0_enc_warmup_5"

    # # no Z-null penalty, full obs
    # WANDB_PROJECT = "Lorenz_IND012_N2_D1_NormTrue_T3__coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_coupling_lc_9vals__zp0.0__te0.0_enc_warmup_5"

    # # more stable training, 2 epoch stopping
    # WANDB_PROJECT = "Lorenz_IND0_N25_D1_NormTrue_T3__coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_coupling_lc_9vals__zp1.0__te0.0_enc_warmup_5_stab"

    # # spline coupling w/ full obs (N2)
    # WANDB_PROJECT = "Lorenz_IND012_N2_D1_NormTrue_T3__spline_coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_spline_coupling_lc_9vals__zp1.0__te0.0_enc_warmup_5_stab"

    # # spline coupling w/ partial obs (N25)
    # WANDB_PROJECT = "Lorenz_IND0_N25_D1_NormTrue_T3__spline_coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_spline_coupling_lc_9vals__zp1.0__te0.0_enc_warmup_5_stab"

    # # spline coupling w/ partial obs (N100)
    # WANDB_PROJECT = "Lorenz_IND0_N100_D1_NormTrue_T3__spline_coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_spline_coupling_lc_9vals__kl1.0__te0.0_enc_warmup_5_vaefalse"

    # # spline coupling w/ partial obs (N25), vae KL=0.0001
    # WANDB_PROJECT = "Lorenz_IND0_N25_D1_NormTrue_T3__spline_coupling__JacobianODE"
    # WANDB_GROUP = "sweep_from_scratch_spline_coupling_lc_9vals__kl0.0001__te0.0_enc_warmup_5_vaetrue"

    # spline coupling w/ partial obs (N25), vae KLD=0.0001, KLN = 1.0
    WANDB_PROJECT = "Lorenz_IND0_N25_D1_NormTrue_T3__spline_coupling__JacobianODE"
    WANDB_GROUP = "sweep_from_scratch_spline_coupling_lc_9vals__klN1.0__klD0.0001__te0.0_enc_warmup_5_vaetrue"


    WANDB_PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

    # ================================================================
    # Checkpoint save directory
    # ================================================================
    SAVE_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs"

    # ================================================================
    # Known true Lyapunov exponents (for Lorenz: sigma=10, rho=28, beta=8/3)
    # Set to None for wmtask — will be overridden after loading first run (Section 3)
    # ================================================================
    TRUE_LYAPUNOV = [0.91, 0.0, -14.57]  # None for wmtask (overridden from config)

    print(f"W&B project: {WANDB_PROJECT_PATH}")
    print(f"W&B group:   {WANDB_GROUP}")
    print(f"Save dir:    {SAVE_DIR}")
    return SAVE_DIR, WANDB_GROUP, WANDB_PROJECT, WANDB_PROJECT_PATH


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Collect W&B Runs
    """)
    return


@app.cell
def _(WANDB_GROUP, WANDB_PROJECT_PATH, wandb):
    api = wandb.Api()
    try:
        run_filters = {'group': WANDB_GROUP} if WANDB_GROUP else None
        all_runs = api.runs(WANDB_PROJECT_PATH, filters=run_filters)
        msg = f'Found {len(all_runs)} total runs in {WANDB_PROJECT_PATH}'
        if WANDB_GROUP:
            msg = msg + f' (group={WANDB_GROUP})'
        print(msg)
    except Exception as e:
        print(f'Could not query project: {e}')
        all_runs = []
    return (all_runs,)


@app.cell
def _(all_runs):
    def _is_jac_ode_run(run):
        """Return True if this run has an encoder (i.e. is a JacobianODE run)."""
        return 'model' in _run.config and 'encoder' in _run.config.get('model', {})

    def _get_loop_closure_weight(run):
        """Robustly extract loop_closure_weight from W&B run config (handles nested and flat formats)."""
        try:
            val = _run.config.get('training', {}).get('lightning', {}).get('loop_closure_weight')  # Nested: config["training"]["lightning"]["loop_closure_weight"]
            if val is not None:
                return float(val)
        except (TypeError, AttributeError):
            pass
        for key in ('training.lightning.loop_closure_weight', 'training/lightning/loop_closure_weight'):
            if hasattr(_run.config, 'get') and key in _run.config:
                return float(_run.config[key])  # Flat keys (W&B may use different separators)
        return None

    def _get_tangent_entropy_weight(run):
        """Robustly extract tangent_entropy_weight from W&B run config."""
        try:
            val = _run.config.get('training', {}).get('lightning', {}).get('tangent_entropy_weight')
            if val is not None:
                return float(val)
        except (TypeError, AttributeError):
            pass
        for key in ('training.lightning.tangent_entropy_weight', 'training/lightning/tangent_entropy_weight'):
            if hasattr(_run.config, 'get') and key in _run.config:
                return float(_run.config[key])
        return 0.0
    DELETE_CRASHED_RUNS = False
    print('All runs (state, loop_closure_weight, tangent_entropy_weight):')
    for r in all_runs:
        lc = _get_loop_closure_weight(r)
        _te = _get_tangent_entropy_weight(r)
        lc_str = str(lc) if lc is not None else '?'
    # Set to True to delete crashed runs from W&B (default False — keeps them for debugging, e.g. lc=0)
        print(f'  {r.id}: state={r.state}, lc={lc_str}, te={_te}')
    print()
    # Diagnostic: list all runs (helps debug why lc=0 may not show — e.g. crashed runs were being deleted)
    crashed_ids = []
    for _run in all_runs:
        if _run.state in ('crashed', 'failed') and _is_jac_ode_run(_run):
            _lam = _get_loop_closure_weight(_run)
            if DELETE_CRASHED_RUNS:
                print(f'CRASHED: run_id={_run.id} (loop_closure_weight={_lam}) — deleting from W&B')
                _run.delete()
                crashed_ids.append(_run.id)
            else:
                print(f'CRASHED: run_id={_run.id} (loop_closure_weight={_lam}) — keeping for debugging (set DELETE_CRASHED_RUNS=True to delete)')
    if crashed_ids:
        print(f'\nDeleted {len(crashed_ids)} crashed run(s). Re-query if needed.')
    elif any((r.state in ('crashed', 'failed') and _is_jac_ode_run(r) for r in all_runs)):
        print('\nCrashed runs kept (DELETE_CRASHED_RUNS=False). Check run logs for debugging.')
    else:
        print('No crashed JacobianODE runs found.')
    return (crashed_ids,)


@app.cell
def _(all_runs, crashed_ids):
    # ----------------------------------------------------------------
    # Build sorted list of finished sweep runs
    sweep_run_ids = []
    sweep_lambdas = []
    sweep_te_weights = []
    for _run in all_runs:
        if _run.state != 'finished' or not _is_jac_ode_run(_run):
            continue
        if _run.id in crashed_ids:
            continue
        lc_weight = _get_loop_closure_weight(_run)
        te_weight = _get_tangent_entropy_weight(_run)
        if lc_weight is not None:
            sweep_run_ids.append(_run.id)
            sweep_lambdas.append(lc_weight)
            sweep_te_weights.append(te_weight)
    sorted_pairs = sorted(zip(sweep_lambdas, sweep_te_weights, sweep_run_ids))
    sweep_lambdas = [p[0] for p in sorted_pairs]
    sweep_te_weights = [p[1] for p in sorted_pairs]
    # Sort by lambda value
    sweep_run_ids = [p[2] for p in sorted_pairs]
    print(f'\nFound {len(sweep_run_ids)} finished sweep runs:')
    for _lam, _te, rid in zip(sweep_lambdas, sweep_te_weights, sweep_run_ids):
        print(f'  loop_closure_weight={_lam}, tangent_entropy_weight={_te} -> run_id={rid}')
    return sweep_lambdas, sweep_run_ids, sweep_te_weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compute Diagnostics and Select Best Model

    Uses the tuning module's `select_from_wandb_runs` for physics-informed selection:

    - **C1 (one-step MASE)**: model must beat persistence baseline (MASE < 1)
    - **C2 (loop closure)**: loop closure loss must be below `sqrt(n_latent)`
    - **C3 (eigenvalue)**: fraction of fast eigenvalues must be below threshold
    """)
    return


@app.cell
def _(OmegaConf, SAVE_DIR, WANDB_PROJECT_PATH, load_run, np, sweep_run_ids):
    _run0, _cfg0, _eq0, _dt0, _values0, _, _, _, _, _ = load_run(WANDB_PROJECT_PATH, run_id=sweep_run_ids[0], save_dir=SAVE_DIR, generate_data=True, verbose=False)
    n_dims = _values0.shape[-1]
    n_latent = OmegaConf.select(_cfg0, 'model.encoder.n_latent', default=None)
    if n_latent is None:
        n_latent = n_dims
    dt = _dt0
    _n_target_dims = OmegaConf.select(_cfg0, 'model.n_target_dims', default=None)
    _n_dyn = _n_target_dims if _n_target_dims is not None else n_latent
    DATA_SOURCE = OmegaConf.select(_cfg0, 'data.data_type', default='dysts')
    if DATA_SOURCE != 'dysts':
        TRUE_LYAPUNOV_1 = None
    print(f'n_dims = {n_dims}, n_latent = {n_latent}, n_dyn = {_n_dyn}, dt = {dt:.4f}, data = {DATA_SOURCE}')
    print(f'C2 loop-closure threshold: sqrt(n_dyn) = {np.sqrt(_n_dyn):.4f}')
    return TRUE_LYAPUNOV_1, dt, n_dims, n_latent


@app.cell
def _(OmegaConf, n_latent):
    # ----------------------------------------------------------------
    # Detect encoder type from model config
    import torch.nn.functional as F
    # For coupling encoders, n_target_dims is set in the model config.
    # For standard encoders (MLP, SSM, etc.), n_target_dims is None.
    n_target_dims = OmegaConf.select(_cfg0, 'model.n_target_dims', default=None)
    IS_COUPLING = n_target_dims is not None
    N_DYN = n_target_dims if IS_COUPLING else n_latent
    if IS_COUPLING:
        print(f'Coupling encoder detected: n_target_dims={n_target_dims}, full n_latent={n_latent}')
    else:
        print(f'Standard encoder: n_latent={n_latent}')

    def get_z_dyn(z):
        """Extract dynamic subspace from full latent encoding.

        No-op for standard (MLP/SSM/Transformer/TCN) encoders.
        For coupling encoders, returns z[..., :n_target_dims].
        """
        if n_target_dims is not None:
            return _z[..., :n_target_dims]
        return _z

    def get_z_null(z):
        """Extract null subspace (penalized toward zero).

        Returns None for standard encoders.
        """
        if n_target_dims is not None:
            return _z[..., n_target_dims:]
        return None

    return F, IS_COUPLING, get_z_dyn, get_z_null


@app.cell
def _(
    SAVE_DIR,
    WANDB_PROJECT_PATH,
    dt,
    n_dims,
    n_latent,
    select_from_wandb_runs,
    sweep_lambdas,
    sweep_run_ids,
    sweep_te_weights,
):
    sweep_result = select_from_wandb_runs(
        run_ids=sweep_run_ids,
        project=WANDB_PROJECT_PATH,
        dt=dt,
        n_dims=n_dims,
        n_batches=100,
        eigenvalue_threshold=0.001,
        use_loop_closure=True,
        lambda_values=sweep_lambdas,
        save_dir=SAVE_DIR,
        verbose=True,
        n_latent=n_latent,
    )

    result = sweep_result.selection
    all_diagnostics = sweep_result.diagnostics

    print("\n" + "=" * 60)
    print("MODEL SELECTION RESULT")
    print("=" * 60)
    print(f"Best lambda loop closure:    {sweep_lambdas[result.best_index]}")
    print(f"Best lambda tangent entropy: {sweep_te_weights[result.best_index]}")
    print(f"Best run ID:    {sweep_run_ids[result.best_index]}")
    print(f"Best traj loss: {result.best_metrics.trajectory_val_loss:.6f}")
    print(f"\nCriteria applied: {result.criteria_applied}")
    print(f"Surviving models: {len(result.surviving_indices)} / {len(all_diagnostics)}")
    return all_diagnostics, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Visualize Selection
    """)
    return


@app.cell
def _(all_diagnostics, n_latent, np, plt, result, sweep_lambdas):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))
    one_step_mases = [m.one_step_mase for m in all_diagnostics]
    loop_closure_losses = [m.loop_closure_loss for m in all_diagnostics]
    eig_fracs = [m.fast_eigenvalue_fraction for m in all_diagnostics]
    traj_losses = [m.trajectory_val_loss for m in all_diagnostics]
    colors = ['tab:green' if _i in result.surviving_indices else 'tab:red' for _i in range(len(all_diagnostics))]
    x_labels = [str(v) for v in sweep_lambdas]
    x_pos = range(len(sweep_lambdas))
    _axes[0, 0].bar(x_pos, one_step_mases, color=colors)
    if result.best_index is not None:
        _axes[0, 0].bar(result.best_index, one_step_mases[result.best_index], color='gold', edgecolor='black', linewidth=2, label='Selected')
    _axes[0, 0].axhline(y=1.0, color='k', linestyle='--', lw=1, label='persistence')
    # Panel 1: One-step MASE
    _axes[0, 0].set_xticks(x_pos)
    _axes[0, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    _axes[0, 0].set_ylabel('MASE')
    _axes[0, 0].set_title('C1: One-step MASE')
    _axes[0, 0].set_yscale('log')
    _axes[0, 0].legend(fontsize=8)
    _axes[0, 1].bar(x_pos, loop_closure_losses, color=colors)
    if result.best_index is not None:
        _axes[0, 1].bar(result.best_index, loop_closure_losses[result.best_index], color='gold', edgecolor='black', linewidth=2, label='Selected')
    _axes[0, 1].axhline(y=np.sqrt(n_latent), color='k', linestyle='--', lw=1, label=f'sqrt(n_latent)={np.sqrt(n_latent):.2f}')
    # Panel 2: Loop closure loss
    _axes[0, 1].set_xticks(x_pos)
    _axes[0, 1].set_xticklabels(x_labels, rotation=45, ha='right')
    _axes[0, 1].set_ylabel('Loop closure loss')
    _axes[0, 1].set_title('C2: Loop Closure')
    _axes[0, 1].legend(fontsize=8)
    _axes[0, 1].set_yscale('log')
    _axes[1, 0].bar(x_pos, eig_fracs, color=colors)
    if result.best_index is not None:
        _axes[1, 0].bar(result.best_index, eig_fracs[result.best_index], color='gold', edgecolor='black', linewidth=2, label='Selected')
    _axes[1, 0].set_xticks(x_pos)
    # Panel 3: Fast eigenvalue fraction
    _axes[1, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    _axes[1, 0].set_ylabel('Fraction fast eigenvalues')
    _axes[1, 0].set_title('C3: Eigenvalue Fraction')
    _axes[1, 0].legend(fontsize=8)
    _axes[1, 1].bar(x_pos, traj_losses, color=colors)
    if result.best_index is not None:
        _axes[1, 1].bar(result.best_index, traj_losses[result.best_index], color='gold', edgecolor='black', linewidth=2, label='Selected')
    _axes[1, 1].set_xticks(x_pos)
    # Panel 4: Trajectory val loss (selection target)
    _axes[1, 1].set_xticklabels(x_labels, rotation=45, ha='right')
    _axes[1, 1].set_ylabel('Trajectory val loss')
    _axes[1, 1].set_title('Trajectory Loss (selection target)')
    _axes[1, 1].legend(fontsize=8)
    _axes[1, 1].set_yscale('log')
    for _ax in _axes.flat:
        _ax.set_xlabel('loop_closure_weight')
    _fig.suptitle('Sweep Model Selection\n(green = passes all criteria; gold = selected best)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Load the Best Model

    Load the selected model's architecture, data, and checkpoint from W&B.
    """)
    return


@app.cell
def _(
    SAVE_DIR,
    TRUE_LYAPUNOV_1,
    WANDB_PROJECT_PATH,
    create_dataloaders,
    load_checkpoint,
    load_run,
    sweep_lambdas,
    sweep_run_ids,
    torch,
):
    idx = 4
    best_run_id = sweep_run_ids[idx]
    best_lambda = sweep_lambdas[idx]
    print(f'Loading model: run_id={best_run_id}, loop_closure_weight={best_lambda}')
    _run, best_cfg, eq, dt_1, values, _, _, _, _, best_lit_model = load_run(WANDB_PROJECT_PATH, run_id=best_run_id, save_dir=SAVE_DIR, generate_data=True, verbose=True)
    train_dl, val_dl, test_dl, trajs = create_dataloaders(best_cfg, values, verbose=True, return_full_obs=True)
    test_trajs_full = trajs['test_trajs_full'].sequence
    load_checkpoint(_run, best_cfg, best_lit_model, save_dir=SAVE_DIR, verbose=True)
    if TRUE_LYAPUNOV_1 is not None:
        best_lit_model.true_lyapunov_exponents = torch.tensor(TRUE_LYAPUNOV_1, dtype=torch.float32)
    best_lit_model.eval()
    mu = best_cfg.data.postprocessing.mu
    sigma = best_cfg.data.postprocessing.sigma
    OBSERVED_INDICES = best_cfg.data.train_test_params.delay_embedding_params.observed_indices
    N_DELAYS = best_cfg.data.train_test_params.delay_embedding_params.n_delays
    DELAY_SPACING = best_cfg.data.train_test_params.delay_embedding_params.delay_spacing
    PREDICTION_STEPS = best_lit_model.prediction_steps
    n_dims_1 = values.shape[-1]
    n_obs = trajs['train_trajs'].sequence.shape[-1]
    print(f'\nModel type: {type(best_lit_model).__name__}')
    print(f'Total params: {sum((p.numel() for p in best_lit_model.parameters())):,}')
    print(f'Test trajectories (full obs): {test_trajs_full.shape}')
    return (
        DELAY_SPACING,
        N_DELAYS,
        OBSERVED_INDICES,
        best_cfg,
        best_lambda,
        best_lit_model,
        dt_1,
        eq,
        mu,
        sigma,
        test_dl,
        test_trajs_full,
        trajs,
        val_dl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Reconstruction Quality

    Encode-then-decode (no prediction) to check autoencoder quality.
    """)
    return


@app.cell
def _(
    DELAY_SPACING,
    F,
    IS_COUPLING,
    N_DELAYS,
    OBSERVED_INDICES,
    best_lit_model,
    get_z_dyn,
    get_z_null,
    np,
    plt,
    r2_score,
    test_dl,
    test_trajs_full,
    torch,
    trajs,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit_model = best_lit_model.to(device)
    lit_model.eval()
    if hasattr(lit_model.encoder, 'decoder'):
        _decoder_n_out = lit_model.encoder.decoder.n_output
    else:
        _decoder_n_out = lit_model.encoder.n_latent
    _time_offset = (N_DELAYS - 1) * DELAY_SPACING
    with torch.no_grad():
        _batch = test_dl.dataset.sequence[:16].to(device)
        _z = lit_model.encode_trajectory(_batch)
        _recon = lit_model.decode_trajectory(_z)
        margin = getattr(lit_model.encoder, 'context_margin', 0)
        targets = _batch[:, margin:] if margin > 0 else _batch
        targets = targets[..., :_decoder_n_out]
        recon_mse = ((_recon - targets) ** 2).mean().item()
        recon_nmse = recon_mse / targets.var().item()
        print(f'Reconstruction nMSE: {recon_nmse:.6f}')
        if getattr(lit_model, 'decode_only_recent', False):
            print(f'  (decode_only_recent=True, decoder outputs {_decoder_n_out} of {_batch.shape[-1]} dims)')
    if IS_COUPLING:
        with torch.no_grad():
            z_check = lit_model.encode_trajectory(_batch)
            x_roundtrip = lit_model.decode_trajectory(z_check)
            inv_err = F.mse_loss(x_roundtrip, _batch).item()
            z_null = get_z_null(z_check)
            null_rms = z_null.pow(2).mean().sqrt().item() if z_null is not None and z_null.numel() > 0 else 0
            print(f'Inverse consistency MSE (encode→decode): {inv_err:.6e}')
            print(f'Null subspace RMS: {null_rms:.6e}')
    test_trajs_obs = trajs['test_trajs'].sequence
    with torch.no_grad():
        latent_traj_full = lit_model.encode_trajectory(test_trajs_obs.to(device))
        if IS_COUPLING:
            latent_traj_full = get_z_dyn(latent_traj_full)
            if z_null is not None:
                print(f'padding with {z_null.shape[-1]} zeros')
            else:
                print('no null space, not padding')
            latent_traj_full = lit_model._pad_to_full_dim(latent_traj_full)
        traj_decoded_full = lit_model.decode_trajectory(latent_traj_full)
    latent_traj_full = latent_traj_full.cpu()
    traj_decoded_full = traj_decoded_full.cpu()
    test_trajs_full_aligned = test_trajs_full[:, _time_offset:]
    if getattr(lit_model, 'decode_only_recent', False):
        if OBSERVED_INDICES != 'all':
            plot_targets = test_trajs_full_aligned[..., OBSERVED_INDICES]
        else:
            plot_targets = test_trajs_full_aligned
    else:
        plot_targets = test_trajs_obs
    if lit_model.reconstruction_mode == 'most_recent':
        plot_dims = np.arange(lit_model.n_recent_dims)
    else:
        n_dims_2 = min(10, plot_targets.shape[-1], traj_decoded_full.shape[-1])
        plot_dims = np.arange(n_dims_2)
    T_plot = min(plot_targets.shape[1], traj_decoded_full.shape[1])
    for _i in plot_dims:
        plt.plot(plot_targets[0, :T_plot, _i].numpy(), c=f'C{_i}', label=f'true dim {_i}')
    for _i in plot_dims:
        plt.plot(traj_decoded_full[0, :T_plot, _i].numpy(), c=f'C{_i}', linestyle='--', label=f'decoded dim {_i}')
    r2_val = r2_score(plot_targets[:, :T_plot, plot_dims].reshape(-1, len(plot_dims)), traj_decoded_full[:, :T_plot, plot_dims].reshape(-1, len(plot_dims)))
    plt.title(f'True vs. Decoded (no prediction)\n$R^2 = {r2_val:.4f}$')
    plt.legend()
    plt.show()
    return device, latent_traj_full, lit_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Prediction Quality

    Teacher-forced (one-step) and free-running predictions on validation data.
    """)
    return


@app.cell
def _(DataLoader, RandomSampler, device, lit_model, torch, tqdm, val_dl):
    model_maes_forced = []
    model_maes_free = []
    persistence_maes = []
    n_batches = 10
    RAND_SEED = 42
    _generator = torch.Generator().manual_seed(RAND_SEED)
    dl = val_dl
    num_samples = n_batches * dl.batch_size
    rand_sampler = RandomSampler(dl.dataset, num_samples=num_samples, replacement=False, generator=_generator)
    rand_dl = DataLoader(dl.dataset, batch_size=dl.batch_size, sampler=rand_sampler, num_workers=dl.num_workers, pin_memory=getattr(dl, 'pin_memory', False))
    for _i, _batch in enumerate(tqdm(rand_dl, desc='Evaluating', total=n_batches)):
        if _i >= n_batches:
            break
        _batch = _batch.to(device)
        with torch.no_grad():
            ret_forced = lit_model.trajectory_model_step(_batch, alpha_teacher_forcing=1, return_decoded=True)
            ret_free = lit_model.trajectory_model_step(_batch, alpha_teacher_forcing=0, return_decoded=True)
        model_maes_forced.append(ret_forced['metric_vals']['model_mae'].item())
        model_maes_free.append(ret_free['metric_vals']['model_mae'].item())
        persistence_maes.append(ret_free['metric_vals']['persistence_mae'].item())
    forced_mase = torch.tensor(model_maes_forced).mean() / torch.tensor(persistence_maes).mean()
    free_mase = torch.tensor(model_maes_free).mean() / torch.tensor(persistence_maes).mean()
    print(f'Teacher-forced MASE: {forced_mase:.4f}')
    print(f'Free-running MASE:   {free_mase:.4f}')
    print(f'Mean persistence MAE: {torch.tensor(persistence_maes).mean():.6f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Latent Space Analysis — Utilization, PCA Embeddings
    """)
    return


@app.cell
def _(
    IS_COUPLING,
    device,
    get_z_dyn,
    get_z_null,
    latent_traj_full,
    lit_model,
    math,
    np,
    plt,
    test_trajs_full,
    torch,
    tqdm,
    trajs,
):
    from sklearn.decomposition import PCA

    def participation_ratio(X):
        X_flat = X.reshape(-1, X.shape[-1])
        cov = np.cov(X_flat, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        return eigvals.sum() ** 2 / ((eigvals ** 2).sum() + 1e-12)
    print(f'test_trajs_full.shape: {test_trajs_full.shape}')
    print(f'latent_traj_full.shape: {latent_traj_full.shape}')
    pr_true = participation_ratio(test_trajs_full)
    pr_latent = participation_ratio(latent_traj_full)
    use_train = True
    # Encode full trajectories
    traj_key = 'train_trajs' if use_train else 'test_trajs'
    full_obs_key = 'train_trajs_full' if use_train else 'test_trajs_full'
    label_prefix = 'Train' if use_train else 'Test'
    trajs_obs = trajs[traj_key].sequence
    trajs_full_arr = np.asarray(trajs[full_obs_key].sequence)
    all_latents = []
    with torch.no_grad():
        for _i in tqdm(range(0, trajs_obs.shape[0], 8)):
            _x = torch.as_tensor(trajs_obs[_i:_i + 8]).float().to(device)
            _z = lit_model.encode_trajectory(_x)
            all_latents.append(_z.cpu())
    Z = torch.cat(all_latents, dim=0).numpy()
    Z_dyn = get_z_dyn(torch.from_numpy(Z)).numpy()
    T_latent = Z.shape[1]
    X_true = trajs_full_arr[:, -T_latent:, :]
    print(f'Z shape: {Z.shape}, Z_dyn shape: {Z_dyn.shape}, X_true shape: {X_true.shape}')
    # Extract dynamic subspace for latent-space analysis
    Z_flat = Z_dyn.reshape(-1, Z_dyn.shape[-1])
    X_true_flat = X_true.reshape(-1, X_true.shape[-1])
    dim_var = Z_flat.var(axis=0)
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.bar(range(len(dim_var)), dim_var / dim_var.sum(), color='steelblue')
    _ax.set_xlabel('latent dimension (dynamic)')
    # Use Z_dyn for all latent-space analysis (PCA, utilization, etc.)
    _ax.set_ylabel('fractional variance')
    _ax.set_title(f'Latent dimension utilisation (D_dyn={Z_dyn.shape[-1]})')
    plt.tight_layout()
    # Per-dimension variance (latent utilization) — on dynamic subspace
    plt.show()
    p = dim_var / dim_var.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    _n_dyn = Z_flat.shape[-1]
    utilization = entropy / math.log(_n_dyn) if _n_dyn > 1 else 1.0
    print(f'Entropy-based utilization: {utilization:.3f}  (1.0 = uniform)')
    if IS_COUPLING:
        Z_null = get_z_null(torch.from_numpy(Z)).numpy()
        null_rms_per_t = np.sqrt((Z_null ** 2).mean(axis=(0, 2)))
        plt.figure(figsize=(10, 3))
        plt.plot(null_rms_per_t)
        plt.xlabel('Time step')
        plt.ylabel('Null subspace RMS')
    # Null subspace monitoring (coupling only)
        plt.title('Null Subspace Magnitude Over Time')
        plt.tight_layout()
        plt.show()
        print(f'Null subspace mean RMS: {np.sqrt((Z_null ** 2).mean()):.6e}')
    return PCA, X_true_flat, Z_flat, label_prefix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Lyapunov Exponent Comparison
    """)
    return


@app.cell
def _(
    LitLatentJacobianODE,
    TRUE_LYAPUNOV_1,
    device,
    dt_1,
    get_z_dyn,
    lit_model,
    test_dl,
    torch,
):
    def kaplan_yorke_dim(lyap_exps):
        """
        Compute the Kaplan-Yorke (Lyapunov) dimension D_KY given the Lyapunov exponents.
        D_KY = k + (sum_{i=1}^k lambda_i) / |lambda_{k+1}|,
        where k is the largest integer such that sum_{i=1}^k lambda_i > 0,
        and lambda_i are sorted in decreasing order.

        Handles batches of shape [B, N] and outputs [B].
        (Torch version)
        """
        lyap_exps = torch.as_tensor(lyap_exps)
        if lyap_exps.ndim == 1:
            lyap_exps = lyap_exps.unsqueeze(0)
        _B, N = lyap_exps.shape
        lyap_sorted, _ = torch.sort(lyap_exps, descending=True, dim=1)
        cumsum = torch.cumsum(lyap_sorted, dim=1)
        mask_pos = cumsum > 0
        k_idx = mask_pos.sum(dim=1) - 1
        D_KY = torch.zeros(_B, dtype=lyap_exps.dtype, device=lyap_exps.device)
        for b in range(_B):
            k = k_idx[b].item()
            if k < 0:
                D_KY[b] = 0.0
                continue
            sum_k = cumsum[b, k]
            if k + 1 < N and lyap_sorted[b, k + 1] != 0:
                D_KY[b] = k + 1 + sum_k / torch.abs(lyap_sorted[b, k + 1])
            else:
                D_KY[b] = float(N)
        return D_KY if D_KY.shape[0] > 1 else D_KY[0]
    lit_model.eval()
    lit_model_1 = lit_model.to(device)
    _N_SAMPLE = 128
    with torch.no_grad():
        _traj_full = torch.as_tensor(test_dl.dataset.sequence).float().to(device)
        n_test_trajs = _traj_full.shape[0]
        _T_full = _traj_full.shape[1]
        print(f'Using {n_test_trajs} full-length test trajectories, T={_T_full} each')
        _z_full = lit_model_1.encode_trajectory(_traj_full)
        _z_for_jac = get_z_dyn(_z_full)
        _generator = torch.Generator().manual_seed(0)
        _z_for_jac = _z_for_jac[torch.randperm(_z_for_jac.shape[0], generator=_generator)[:_N_SAMPLE]]
        _jacs = lit_model_1.compute_jacobians(_z_for_jac)
        all_pred_lyap = LitLatentJacobianODE.compute_lyapunov_exponents(_jacs, dt_1)
        pred_lyap = all_pred_lyap.mean(dim=0)
        pred_lyap_std = all_pred_lyap.std(dim=0)
    print(f'\nPredicted Lyapunov exponents (mean +/- std over {_N_SAMPLE} trajs):')
    for _i, (_le, _std) in enumerate(zip(pred_lyap, pred_lyap_std)):
        print(f'  lambda_{_i + 1} = {_le.item():+.4f} +/- {_std.item():.4f}')
    print(f'\nTrue Lorenz: {TRUE_LYAPUNOV_1}')
    print(f'Kaplan-Yorke dimension: {kaplan_yorke_dim(all_pred_lyap).mean():.3f} +/- {kaplan_yorke_dim(all_pred_lyap).std():.3f}')
    return kaplan_yorke_dim, lit_model_1


@app.cell
def _(
    LitLatentJacobianODE,
    TRUE_LYAPUNOV_1,
    device,
    dt_1,
    get_z_dyn,
    kaplan_yorke_dim,
    lit_model_1,
    test_dl,
    torch,
):
    T_BURN = 400
    T_DROP = 100
    lit_model_1.eval()
    lit_model_2 = lit_model_1.to(device)
    _N_SAMPLE = 128
    with torch.no_grad():
        _traj_full = torch.as_tensor(test_dl.dataset.sequence).float().to(device)
        n_test_trajs_1 = _traj_full.shape[0]
        _T_full = _traj_full.shape[1]
        print(f'Using {n_test_trajs_1} full-length test trajectories, T={_T_full} each')
        _z_full = lit_model_2.encode_trajectory(_traj_full)
        _z_for_jac = get_z_dyn(_z_full)
        _generator = torch.Generator().manual_seed(0)
        _z_for_jac = _z_for_jac[torch.randperm(_z_for_jac.shape[0], generator=_generator)[:_N_SAMPLE]]
        _B, T_true, D_dyn = _z_for_jac.shape
        z_padded = torch.cat([_z_for_jac, torch.zeros(_B, T_BURN, D_dyn, device=device)], dim=1)
        print(f'Running artificial burn-in for {T_BURN} steps...')
        from JacobianODE.jacobians.jacobianODE import JacobianODEint
        jacobian_odeint = JacobianODEint(lit_model_2.compute_jacobians, dt_1)
        z_combined = jacobian_odeint.generate_dynamics(z_padded, traj_init_steps=T_true, alpha_teacher_forcing=0.0, fast_mode=True, verbose=True, interp_pts=4, inner_N=20)
        print(f'Combined trajectory length: {z_combined.shape[1]} (True: {T_true}, Burn: {T_BURN}, Drop: {T_DROP}, Using: {T_true + T_BURN - T_DROP})')
        _jacs = lit_model_2.compute_jacobians(z_combined)
        all_pred_lyap_1 = LitLatentJacobianODE.compute_lyapunov_exponents(_jacs[:, T_DROP:], dt_1)
        pred_lyap_1 = all_pred_lyap_1.mean(dim=0)
        pred_lyap_std_1 = all_pred_lyap_1.std(dim=0)
    print(f'\nPredicted Lyapunov exponents (mean +/- std over {_N_SAMPLE} trajs):')
    for _i, (_le, _std) in enumerate(zip(pred_lyap_1, pred_lyap_std_1)):
        print(f'  lambda_{_i + 1} = {_le.item():+.4f} +/- {_std.item():.4f}')
    print(f'\nTrue Lorenz: {TRUE_LYAPUNOV_1}')
    print(f'Kaplan-Yorke dimension: {kaplan_yorke_dim(all_pred_lyap_1).mean():.3f} +/- {kaplan_yorke_dim(all_pred_lyap_1).std():.3f}')
    return (lit_model_2,)


@app.cell
def _(
    LitLatentJacobianODE,
    TRUE_LYAPUNOV_1,
    device,
    dt_1,
    get_z_dyn,
    lit_model_2,
    torch,
    trajs,
):
    lit_model_2.eval()
    lit_model_3 = lit_model_2.to(device)
    with torch.no_grad():
        _traj_full = torch.as_tensor(trajs['test_trajs'].sequence).float().to(device)
        n_test_trajs_2 = _traj_full.shape[0]
        _T_full = _traj_full.shape[1]
        print(f'Using {n_test_trajs_2} full-length test trajectories, T={_T_full} each')
        _z_full = lit_model_3.encode_trajectory(_traj_full)
        _z_for_jac = get_z_dyn(_z_full)
        all_pred_lyap_2 = []
        for _i in range(n_test_trajs_2):
            jacs_i = lit_model_3.compute_jacobians(_z_for_jac[_i:_i + 1])[0]
            _le_i = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_i.cpu(), dt_1)
            all_pred_lyap_2.append(_le_i)
            if _i < 3:
                print(f'  Traj {_i}: {_le_i.numpy()}')
        all_pred_lyap_2 = torch.stack(all_pred_lyap_2)
        pred_lyap_2 = all_pred_lyap_2.mean(dim=0)
        pred_lyap_std_2 = all_pred_lyap_2.std(dim=0)
    print(f'\nPredicted Lyapunov exponents (mean +/- std over {n_test_trajs_2} trajs):')
    for _i, (_le, _std) in enumerate(zip(pred_lyap_2, pred_lyap_std_2)):
        print(f'  lambda_{_i + 1} = {_le.item():+.4f} +/- {_std.item():.4f}')
    print(f'\nTrue Lorenz: {TRUE_LYAPUNOV_1}')
    return (
        all_pred_lyap_2,
        lit_model_3,
        n_test_trajs_2,
        pred_lyap_2,
        pred_lyap_std_2,
    )


@app.cell
def _(
    LitLatentJacobianODE,
    TRUE_LYAPUNOV_1,
    best_cfg,
    dt_1,
    eq,
    n_test_trajs_2,
    np,
    pred_lyap_2,
    pred_lyap_std_2,
    torch,
    trajs,
):
    mu_val = best_cfg.data.postprocessing.mu
    sigma_norm = best_cfg.data.postprocessing.sigma
    if eq is not None:
        if 'test_trajs_full' in trajs:
            traj_full_np = trajs['test_trajs_full'].sequence
        else:
            traj_full_np = trajs['test_trajs'].sequence
        traj_raw = np.asarray(traj_full_np) * sigma_norm + mu_val
        all_empirical_lyap = []
        for _i in range(n_test_trajs_2):
            traj_i = traj_raw[_i]
            if hasattr(eq, 'model'):
                traj_i = torch.as_tensor(traj_i).float()
            jacs_np = eq.jac(traj_i, t=0)
            jacs_t = torch.as_tensor(jacs_np).float()
            _le_i = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_t, dt_1)
            all_empirical_lyap.append(_le_i)
        all_empirical_lyap = torch.stack(all_empirical_lyap)
        empirical_lyap = all_empirical_lyap.mean(dim=0)
        empirical_lyap_std = all_empirical_lyap.std(dim=0)
        emp_np = empirical_lyap.numpy()
        emp_std_np = empirical_lyap_std.numpy()
        print(f'Empirical true Lyapunov exponents (mean +/- std):')
        for _i, (_le, _std) in enumerate(zip(empirical_lyap, empirical_lyap_std)):
            print(f'  lambda_{_i + 1} = {_le.item():+.4f} +/- {_std.item():.4f}')
    else:
        emp_np = np.array([])
        emp_std_np = np.array([])
        print('(Skipped empirical Lyapunov: eq is None. For wmtask, ensure the package returns (eq, sol, dt) with WMTaskEq; reinstall from the updated wmtask repo if needed.)')
    pred_np = pred_lyap_2.numpy()
    pred_std_np = pred_lyap_std_2.numpy()
    n_true = len(TRUE_LYAPUNOV_1) if TRUE_LYAPUNOV_1 else 0
    n_plot = max(len(pred_np), len(emp_np), n_true)
    x_idx = np.arange(n_plot)
    bar_w = 0.25
    return (
        all_empirical_lyap,
        bar_w,
        emp_np,
        emp_std_np,
        pred_np,
        pred_std_np,
        x_idx,
    )


@app.cell
def _(
    TRUE_LYAPUNOV_1,
    all_empirical_lyap,
    all_pred_lyap_2,
    kaplan_yorke_dim,
    np,
    plt,
    torch,
):
    ky_pred = kaplan_yorke_dim(all_pred_lyap_2)
    ky_emp = kaplan_yorke_dim(all_empirical_lyap)
    ky_pred_np = np.atleast_1d(ky_pred.numpy() if torch.is_tensor(ky_pred) else np.array(ky_pred))
    ky_emp_np = np.atleast_1d(ky_emp.numpy() if torch.is_tensor(ky_emp) else np.array(ky_emp))
    if TRUE_LYAPUNOV_1 is not None:
        lyap_t = np.sort(TRUE_LYAPUNOV_1)[::-1]
        cum = np.cumsum(lyap_t)
        k = np.where(cum > 0)[0][-1] if np.any(cum > 0) else -1
        ky_theory = k + 1 + cum[k] / np.abs(lyap_t[k + 1]) if k >= 0 and k + 1 < len(lyap_t) else None
    else:
        ky_theory = None
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _ax = _axes[0]
    _ax.scatter(ky_pred_np, ky_emp_np, s=120, c=np.arange(len(ky_pred_np)), cmap='viridis', edgecolors='black', linewidths=1.5, zorder=3)
    lim_lo = min(ky_pred_np.min(), ky_emp_np.min()) - 0.02
    lim_hi = max(ky_pred_np.max(), ky_emp_np.max()) + 0.02
    _ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=1.5, label='y = x (perfect agreement)', alpha=0.7)
    if ky_theory is not None:
        _ax.axhline(ky_theory, color='#e74c3c', ls=':', lw=1.5, label=f'Theoretical ≈ {ky_theory:.3f}')
        _ax.axvline(ky_theory, color='#e74c3c', ls=':', lw=1.5)
    _ax.set_xlabel('Predicted D_KY (model)', fontsize=11)
    _ax.set_ylabel('Empirical D_KY (true Jacobian)', fontsize=11)
    _ax.set_title('Kaplan–Yorke dimension: Model vs Ground Truth', fontsize=12)
    _ax.set_aspect('equal')
    _ax.legend(loc='lower right', fontsize=9)
    _ax.set_xlim(lim_lo, lim_hi)
    _ax.set_ylim(lim_lo, lim_hi)
    _ax.grid(True, alpha=0.3)
    _ax = _axes[1]
    _x = np.arange(len(ky_pred_np))
    width = 0.35
    bars1 = _ax.bar(_x - width / 2, ky_pred_np, width, label='Predicted', color='#3498db', alpha=0.9)
    bars2 = _ax.bar(_x + width / 2, ky_emp_np, width, label='Empirical', color='#2ecc71', alpha=0.9)
    if ky_theory is not None:
        _ax.axhline(ky_theory, color='#e74c3c', ls='--', lw=1.5, label=f'Theoretical ≈ {ky_theory:.3f}')
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f'Traj {_i}' for _i in range(len(ky_pred_np))])
    _ax.set_ylabel('Kaplan–Yorke dimension')
    _ax.set_title('Per-trajectory D_KY comparison')
    _ax.legend(loc='upper right', fontsize=9)
    _ax.grid(True, alpha=0.3, axis='y')
    mean_pred, std_pred = (ky_pred_np.mean(), ky_pred_np.std())
    mean_emp, std_emp = (ky_emp_np.mean(), ky_emp_np.std())
    _fig.suptitle(f'Predicted: {mean_pred:.4f} ± {std_pred:.4f}   |   Empirical: {mean_emp:.4f} ± {std_emp:.4f}' + (f'   |   Theoretical: {ky_theory:.3f}' if ky_theory else ''), fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()
    return ky_emp_np, ky_pred_np


@app.cell
def _(
    TRUE_LYAPUNOV_1,
    bar_w,
    best_lambda,
    emp_np,
    emp_std_np,
    plt,
    pred_np,
    pred_std_np,
    x_idx,
):
    n_lyaps = min(10, len(pred_np))
    n_lyaps = min(n_lyaps, len(emp_np))
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.bar(x_idx[:n_lyaps] - bar_w, pred_np[:n_lyaps], width=bar_w, yerr=pred_std_np[:n_lyaps], capsize=3, label='Predicted (model)', alpha=0.8)
    if len(emp_np) > 0:
        _ax.bar(x_idx[:n_lyaps] - bar_w, emp_np[:n_lyaps], width=bar_w, yerr=emp_std_np[:n_lyaps], capsize=3, label='Empirical (true Jacobian)', alpha=0.8)
    if TRUE_LYAPUNOV_1:
        _ax.bar(x_idx[:n_lyaps] + bar_w, TRUE_LYAPUNOV_1[:n_lyaps], width=bar_w, label='Literature', alpha=0.8)
    _ax.axhline(y=0, color='k', linestyle='--', lw=0.5)
    _ax.set_xticks(x_idx[:n_lyaps])
    _ax.set_xlabel('Exponent index')
    _ax.set_ylabel('Lyapunov exponent')
    _ax.set_title(f'Lyapunov Spectrum (loop_closure_weight={best_lambda})')
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(PCA, X_true_flat, Z_flat, ky_emp_np, ky_pred_np, label_prefix, plt):
    # Compute mean Kaplan–Yorke dimensions for latents and true state
    mean_ky_latent = ky_pred_np.mean()
    mean_ky_true = ky_emp_np.mean()
    print(f'Mean Kaplan–Yorke dim — Latents: {mean_ky_latent:.3f}  |  True state ({label_prefix}): {mean_ky_true:.3f}')
    pca_latent = PCA(n_components=2).fit(Z_flat)
    pca_true = PCA(n_components=2).fit(X_true_flat)
    Z_pc = pca_latent.transform(Z_flat)
    X_true_pc = pca_true.transform(X_true_flat)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    _axes[0].scatter(Z_pc[:, 0], Z_pc[:, 1], s=1, alpha=0.4, c=Z_pc[:, 0], cmap='viridis')
    _axes[0].set_xlabel('PC 1')
    _axes[0].set_ylabel('PC 2')
    _axes[0].set_title(f'Latents\nmean D_KY={mean_ky_latent:.3f}')
    _axes[1].scatter(X_true_pc[:, 0], X_true_pc[:, 1], s=1, alpha=0.4, c=X_true_pc[:, 0], cmap='viridis')
    _axes[1].set_xlabel('PC 1')
    _axes[1].set_ylabel('PC 2')
    _axes[1].set_title(f'True state ({label_prefix})\nmean D_KY={mean_ky_true:.3f}')
    _fig.suptitle('PC 1 x PC 2', fontsize=12)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Generated Predictions (Latent & Observation Space)

    Roll out the model (no teacher forcing) and compare latent + observation predictions to ground truth.
    """)
    return


@app.cell
def _(device, lit_model_3, mu, nmse_fn, np, plt, sigma, test_dl, torch):
    n_test = test_dl.dataset.sequence.shape[0]
    win_idx = np.random.randint(0, n_test)
    traj_obs = torch.as_tensor(test_dl.dataset.sequence[win_idx:win_idx + 1]).float().to(device)
    with torch.no_grad():
        result_dict = lit_model_3.trajectory_model_step(traj_obs, alpha_teacher_forcing=0.0, obs_noise_scale=0, return_decoded=True)
    _z_pred_full = result_dict['outputs']
    decoded_pred = result_dict['decoded']
    obs_targets = result_dict['targets']
    batch_nmse = nmse_fn(obs_targets, decoded_pred).item()
    print(f'Batch nMSE: {batch_nmse:.6f}')
    with torch.no_grad():
        z_true_full = lit_model_3.encode_trajectory(traj_obs)
    traj_init_steps = lit_model_3.jacobianODEint_kwargs.get('traj_init_steps', 15)
    jac_window_len = traj_init_steps + lit_model_3.prediction_steps
    z_pred_win = _z_pred_full[0].detach().cpu().numpy()
    z_true_win = z_true_full[0, :jac_window_len].detach().cpu().numpy()
    obs_pred_0 = decoded_pred[0].detach().cpu().numpy()
    obs_true_0 = obs_targets[0].detach().cpu().numpy()
    n_latent_1 = z_pred_win.shape[-1]
    _n_cols = min(5, n_latent_1)
    _n_rows = (n_latent_1 + _n_cols - 1) // _n_cols
    _fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(4 * _n_cols, 3 * _n_rows), squeeze=False)
    t_steps = np.arange(jac_window_len)
    for _i in range(n_latent_1):
        _ax = _axes[_i // _n_cols, _i % _n_cols]
        _ax.plot(t_steps, z_true_win[:, _i], 'k-', lw=1.5, label='True (encoded)')
        _ax.plot(t_steps, z_pred_win[:, _i], 'r--', lw=1.5, label='Predicted')
        _ax.axvline(x=traj_init_steps, color='gray', ls=':', lw=1)
        _ax.set_title(f'$z_{{{_i}}}$')
        if _i == 0:
            _ax.legend(fontsize=8)
    for _i in range(n_latent_1, _n_rows * _n_cols):
        _axes[_i // _n_cols, _i % _n_cols].set_visible(False)
    _fig.suptitle('Latent Space: True vs Predicted (no teacher forcing)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    dim_labels = ['x', 'y', 'z']
    t_pred = np.arange(obs_pred_0.shape[0])
    _D_obs = obs_pred_0.shape[-1]
    obs_pred_phys = obs_pred_0 * sigma + mu
    obs_true_phys = obs_true_0 * sigma + mu
    _fig, _axes_p = plt.subplots(1, min(_D_obs, 3), figsize=(5 * min(_D_obs, 3), 4), squeeze=False)
    for _d in range(min(_D_obs, 3)):
        _ax = _axes_p[0, _d]
        _label = dim_labels[_d] if _d < len(dim_labels) else f'dim {_d}'
        _ax.plot(t_pred, obs_true_phys[:, _d], 'k-', lw=1.5, label='True')
        _ax.plot(t_pred, obs_pred_phys[:, _d], 'r--', lw=1.5, label='Predicted')
        _ax.set_xlabel('Prediction step')
        _ax.set_ylabel(f'{_label} (physical)')
        _ax.set_title(_label)
        _ax.legend(fontsize=8)
    _fig.suptitle('Observation Space: True vs Predicted (no teacher forcing)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    return (dim_labels,)


@app.cell
def _(device, eq, get_z_dyn, lit_model_3, plt, torch, trajs):
    if hasattr(eq, 'model'):
        traj_long = trajs['test_trajs'].sequence[[0]].to(device)
    else:
        traj_long = trajs['test_trajs'].sequence[[0]].to(device)[:, 100:]
    with torch.no_grad():
        rd_long = lit_model_3.trajectory_model_step(traj_long, alpha_teacher_forcing=0.0, obs_noise_scale=0, return_decoded=True, strided=False)
        encoded = lit_model_3.encode_trajectory(traj_long)
        _recon = lit_model_3.decode_trajectory(encoded)
        _z_pred_full = lit_model_3._pad_to_full_dim(rd_long['outputs'])
        decoded_v2 = lit_model_3.decode_trajectory(_z_pred_full)
    _fig, _axs = plt.subplots(2, 1, figsize=(8, 6))
    _axs[0].plot(traj_long[0, :, 0].cpu(), label='True Observation')
    _axs[0].plot(decoded_v2[0, :, 0].cpu(), label='Predicted Observation')
    _axs[0].axvline(15, color='k', linestyle='--', alpha=0.7, label='Prediction Start')
    _axs[0].set_title('Observation Space')
    _axs[0].set_ylabel('Obs dim 0')
    _axs[1].plot(get_z_dyn(encoded)[0, :, 0].cpu(), label='Encoded Latent', linestyle='--')
    _axs[1].plot(rd_long['outputs'][0, :, 0].cpu(), label='Predicted Latent')
    _axs[1].axvline(15, color='k', linestyle='--', alpha=0.7, label='Prediction Start')
    _axs[1].set_title('Latent Space')
    _axs[1].set_ylabel('Latent dim 0')
    _axs[1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Prediction Diagnostics — Per-Window nMSE & One-Step vs Free-Running

    Compute per-window normalized MSE across test trajectories, then compare
    one-step (teacher-forced) vs free-running MASE.
    """)
    return


@app.cell
def _(device, lit_model_3, np, torch, trajs):
    traj_init_steps_1 = lit_model_3.jacobianODEint_kwargs.get('traj_init_steps', 15)
    jac_window_len_1 = traj_init_steps_1 + lit_model_3.prediction_steps
    stride = lit_model_3.jac_window_stride
    N_TEST_TRAJS = min(3, trajs['test_trajs'].sequence.shape[0])
    all_decoded_pred = []
    all_obs_targets = []
    all_z_pred = []
    all_z_true_windows = []
    all_per_window_nmse = []
    for t_idx in range(N_TEST_TRAJS):
        traj_obs_i = torch.as_tensor(trajs['test_trajs'].sequence[t_idx:t_idx + 1]).float().to(device)
        with torch.no_grad():
            rd = lit_model_3.trajectory_model_step(traj_obs_i, alpha_teacher_forcing=0.0, obs_noise_scale=0, return_decoded=True)
            z_true_i = lit_model_3.encode_trajectory(traj_obs_i)
        dec_pred = rd['decoded'].cpu()
        obs_tgt = rd['targets'].cpu()
        z_pred_i = rd['outputs'].cpu()
        N_win = dec_pred.shape[0]
        mean_var = obs_tgt.reshape(-1, obs_tgt.shape[-1]).var(dim=0).mean().clamp(min=1e-08)
        for w in range(N_win):
            w_mse = (dec_pred[w] - obs_tgt[w]).pow(2).mean()
            all_per_window_nmse.append((w_mse / mean_var).item())
        z_true_np = z_true_i[0].cpu()
        T_prime = z_true_np.shape[0]
        n_windows = max(1, (T_prime - jac_window_len_1) // stride + 1)
        for w in range(n_windows):
            start = w * stride
            if start + jac_window_len_1 <= T_prime:
                all_z_true_windows.append(z_true_np[start:start + jac_window_len_1].numpy())
        all_decoded_pred.append(dec_pred.numpy())
        all_obs_targets.append(obs_tgt.numpy())
        all_z_pred.append(z_pred_i.numpy())
    all_decoded_pred = np.concatenate(all_decoded_pred, axis=0)
    all_obs_targets = np.concatenate(all_obs_targets, axis=0)
    all_z_pred = np.concatenate(all_z_pred, axis=0)
    all_z_true_windows = np.array(all_z_true_windows)
    all_per_window_nmse = np.array(all_per_window_nmse)
    print(f'Total windows: {len(all_per_window_nmse)}')
    print(f'nMSE — min: {all_per_window_nmse.min():.6f}, median: {np.median(all_per_window_nmse):.6f}, mean: {all_per_window_nmse.mean():.6f}, max: {all_per_window_nmse.max():.6f}')
    return (
        all_decoded_pred,
        all_obs_targets,
        all_per_window_nmse,
        all_z_pred,
        all_z_true_windows,
        jac_window_len_1,
        traj_init_steps_1,
    )


@app.cell
def _(all_per_window_nmse, np, plt):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.plot(all_per_window_nmse, 'k-', lw=0.8, alpha=0.7)
    _ax.axhline(np.median(all_per_window_nmse), color='C0', ls='--', lw=1.5, label=f'median = {np.median(all_per_window_nmse):.4f}')
    _ax.axhline(all_per_window_nmse.mean(), color='C1', ls='--', lw=1.5, label=f'mean = {all_per_window_nmse.mean():.4f}')
    _median_idx = np.argsort(all_per_window_nmse)[len(all_per_window_nmse) // 2]
    _ax.scatter([0], [all_per_window_nmse[0]], c='red', s=50, zorder=5, label=f'window 0')
    _ax.scatter([_median_idx], [all_per_window_nmse[_median_idx]], c='C0', s=50, zorder=5, marker='D', label=f'median window ({_median_idx})')
    _ax.set_xlabel('Window index')
    _ax.set_ylabel('nMSE')
    _ax.set_title(f'Per-window nMSE ({len(all_per_window_nmse)} windows)')
    _ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    # # Compare one-step vs free-running MASE (same method as diagnostics: val_dl + ratio-of-means)
    # # Using mase() on test_trajs gave wrong results; diagnostics use model_mae/persistence_mae
    # # from trajectory_model_step over validation batches. n_batches=100 matches compute_all_diagnostics.
    # n_mase_batches = 100
    # RAND_SEED = 42
    # generator = torch.Generator().manual_seed(RAND_SEED)
    # rand_sampler = RandomSampler(val_dl.dataset, generator=generator)
    # rand_dl = DataLoader(
    #     val_dl.dataset, batch_size=val_dl.batch_size, sampler=rand_sampler,
    #     num_workers=val_dl.num_workers, pin_memory=getattr(val_dl, 'pin_memory', False),
    # )

    # model_maes_onestep = []
    # model_maes_freerun = []
    # persistence_maes = []

    # for i, batch in enumerate(tqdm(rand_dl, desc="MASE", total=n_mase_batches)):
    #     if i >= n_mase_batches:
    #         break
    #     batch = batch.to(device)
    #     with torch.no_grad():
    #         ret_onestep = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=1.0)
    #         ret_free = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0.0)
    #     model_maes_onestep.append(ret_onestep['metric_vals']['model_mae'].item())
    #     model_maes_freerun.append(ret_free['metric_vals']['model_mae'].item())
    #     persistence_maes.append(ret_onestep['metric_vals']['persistence_mae'].item())

    # avg_model_mae_onestep = np.mean(model_maes_onestep)
    # avg_model_mae_freerun = np.mean(model_maes_freerun)
    # avg_persistence = np.mean(persistence_maes)
    # mase_onestep = avg_model_mae_onestep / avg_persistence
    # mase_freerun = avg_model_mae_freerun / avg_persistence

    # print("=" * 60)
    # print("One-step vs Free-running MASE")
    # print("=" * 60)
    # print(f"  MASE (one-step, alpha=1): {mase_onestep:.4f}  [matches diagnostics C1]")
    # print(f"  MASE (free-run, alpha=0): {mase_freerun:.4f}")
    return


@app.cell
def _(
    all_decoded_pred,
    all_obs_targets,
    all_per_window_nmse,
    all_z_pred,
    all_z_true_windows,
    dim_labels,
    jac_window_len_1,
    mase,
    mu,
    np,
    plt,
    sigma,
    traj_init_steps_1,
):
    _median_idx = int(np.argsort(all_per_window_nmse)[len(all_per_window_nmse) // 2])
    print(f'Median-loss window: index={_median_idx}, nMSE={all_per_window_nmse[_median_idx]:.6f}')
    z_pred_med = all_z_pred[_median_idx]
    z_true_med = all_z_true_windows[_median_idx]
    obs_pred_med = all_decoded_pred[_median_idx]
    obs_true_med = all_obs_targets[_median_idx]
    n_latent_2 = z_pred_med.shape[-1]
    _n_cols = min(5, n_latent_2)
    _n_rows = (n_latent_2 + _n_cols - 1) // _n_cols
    t_steps_med = np.arange(jac_window_len_1)
    _fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(4 * _n_cols, 3 * _n_rows), squeeze=False)
    for _i in range(n_latent_2):
        _ax = _axes[_i // _n_cols, _i % _n_cols]
        _ax.plot(t_steps_med, z_true_med[:, _i], 'k-', lw=1.5, label='True (encoded)')
        _ax.plot(t_steps_med, z_pred_med[:, _i], 'r--', lw=1.5, label='Predicted')
        _ax.axvline(x=traj_init_steps_1, color='gray', ls=':', lw=1)
        _ax.set_title(f'$z_{{{_i}}}$')
        if _i == 0:
            _ax.legend(fontsize=8)
    for _i in range(n_latent_2, _n_rows * _n_cols):
        _axes[_i // _n_cols, _i % _n_cols].set_visible(False)
    _fig.suptitle(f'Latent Space: Median-loss window (nMSE={all_per_window_nmse[_median_idx]:.4f})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    obs_pred_phys_med = obs_pred_med * sigma + mu
    obs_true_phys_med = obs_true_med * sigma + mu
    D_obs_med = obs_pred_med.shape[-1]
    t_pred_med = np.arange(obs_pred_med.shape[0])
    _fig, _axes_p = plt.subplots(1, min(D_obs_med, 3), figsize=(5 * min(D_obs_med, 3), 4), squeeze=False)
    for _d in range(min(D_obs_med, 3)):
        _ax = _axes_p[0, _d]
        _label = dim_labels[_d] if _d < len(dim_labels) else f'dim {_d}'
        _ax.plot(t_pred_med, obs_true_phys_med[:, _d], 'k-', lw=1.5, label='True')
        _ax.plot(t_pred_med, obs_pred_phys_med[:, _d], 'r--', lw=1.5, label='Predicted')
        _ax.set_xlabel('Prediction step')
        _ax.set_ylabel(f'{_label}')
        _ax.set_title(_label)
        _ax.legend(fontsize=8)
    window_mase = mase(obs_true_phys_med, obs_pred_phys_med)
    _fig.suptitle(f'Physical Space: Median-loss window (MASE={window_mase:.4f})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Amplification Loss
    """)
    return


@app.cell
def _(device, lit_model_3, loss_amplification, np, plt, torch, trajs):
    N_TRAJS_AMP = 64
    N_NEIGHBORS = 10
    MAX_T = 10
    seq_length = 45

    def extract_sequences(x, seq_length):
        """
        Extract non-overlapping sequences of length seq_length from input tensor x.
        x: Tensor of shape (B, T, D)
        Returns: Tensor of shape (N_seq, seq_length, D)
        """
        _B, T, D = _x.shape
        n_seqs_per_traj = T - seq_length + 1
        seqs = []
        for b in range(_B):
            for t0 in range(n_seqs_per_traj):
                seqs.append(_x[b, t0:t0 + seq_length])
        return torch.stack(seqs, dim=0) if len(seqs) > 0 else torch.empty(0, seq_length, D)
    x_de = extract_sequences(trajs['test_trajs'].sequence, seq_length)
    x_orig = extract_sequences(trajs['test_trajs_full'].sequence[:, -trajs['test_trajs'].sequence.shape[1]:], seq_length)
    rng_amp = np.random.default_rng(42)
    B_amp = x_de.shape[0]
    idx_amp = rng_amp.choice(B_amp, min(N_TRAJS_AMP, B_amp), replace=False)
    X_de_sampled = x_de[torch.from_numpy(idx_amp)].to(device)
    X_orig_sampled = x_orig[torch.from_numpy(idx_amp)].to(device)
    X_latent = lit_model_3.encode_trajectory(X_de_sampled)
    with torch.no_grad():
        amp_loss_true = loss_amplification(X_de_sampled, X_orig_sampled[..., [0]], n_neighbors=N_NEIGHBORS, max_T=MAX_T, normalize=True).item()
        amp_loss_latent = loss_amplification(X_latent, X_orig_sampled[..., [0]], n_neighbors=N_NEIGHBORS, max_T=MAX_T, normalize=True).item()
    print(f'Amplification loss -- True state: {amp_loss_true:.6f}')
    print(f'Amplification loss -- Latent:     {amp_loss_latent:.6f}')
    labels_amp = [f'True state\n(D={X_orig_sampled.shape[-1]})', f'Latent\n(D={X_de_sampled.shape[-1]})']
    values_amp = [amp_loss_true, amp_loss_latent]
    _fig, _ax = plt.subplots(figsize=(5, 4))
    bars = _ax.bar(labels_amp, values_amp, color=['C0', 'C1'], alpha=0.7)
    _ax.set_ylabel('Amplification loss')
    _ax.set_title('Noise Amplification Loss')
    for bar, val in zip(bars, values_amp):
        _ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(device, lit_model_3, np, torch, trajs):
    _N_SAMPLE = 512
    test_trajs_all = trajs['test_trajs'].sequence
    test_pts = test_trajs_all.reshape(-1, test_trajs_all.shape[-1])
    num_pts = test_pts.shape[0]
    rng_jac = np.random.default_rng(123)
    if num_pts < _N_SAMPLE:
        sample_idx = np.arange(num_pts)
    else:
        sample_idx = rng_jac.choice(num_pts, _N_SAMPLE, replace=False)
    test_pts_sampled = test_pts[sample_idx]
    from torch.func import vmap, jacrev, jacfwd
    _enc = lit_model_3.encoder
    _enc.eval()
    _traj = torch.as_tensor(test_trajs_all, dtype=torch.float32, device=device)
    _D_obs = _traj.shape[-1]
    _probe = torch.zeros(1, _D_obs, device=device, dtype=_traj.dtype)
    with torch.no_grad():
        try:
            _enc.encode(_probe)
            _enc_accepts_flat = True
        except Exception:
            _enc_accepts_flat = False
    if hasattr(_enc, 'time_window'):
        _w = _enc.time_window
        if _traj.shape[1] < _w:
            raise ValueError(f'T={_traj.shape[1]} < time_window={_w}; cannot form windows')
        _windows = _traj.unfold(1, _w, 1).permute(0, 1, 3, 2).reshape(-1, _w, _D_obs)
        num_windows = _windows.shape[0]
        if num_windows < _N_SAMPLE:
            window_sample_idx = np.arange(num_windows)
        else:
            window_sample_idx = rng_jac.choice(num_windows, _N_SAMPLE, replace=False)
        _windows_sampled = _windows[torch.from_numpy(window_sample_idx).to(_windows.device)]

        def _encode_one(x):
            return _enc.encode(_x.unsqueeze(0)).squeeze(0)

        def _decode_one(z):
            return _enc.decode(_z.unsqueeze(0)).squeeze(0)
        _z = _enc.encode(_windows_sampled)
        encoder_jacobian = vmap(jacrev(_encode_one))(_windows_sampled)
        decoder_jacobian = vmap(jacfwd(_decode_one))(_z)
    elif _enc_accepts_flat:
        _x_flat = torch.as_tensor(test_pts_sampled, dtype=torch.float32, device=device)

        def _encode_one(x):
            return _enc.encode(_x.unsqueeze(0)).squeeze(0)

        def _decode_one(z):
            return _enc.decode(_z.unsqueeze(0)).squeeze(0)
        _z = _enc.encode(_x_flat)
        encoder_jacobian = vmap(jacrev(_encode_one))(_x_flat)
        decoder_jacobian = vmap(jacfwd(_decode_one))(_z)
    else:
        _B, T, D = _traj.shape
        num_traj = _B
        if num_traj < _N_SAMPLE:
            traj_sample_idx = np.arange(num_traj)
        else:
            traj_sample_idx = rng_jac.choice(num_traj, _N_SAMPLE, replace=False)
        _traj_sampled = _traj[torch.from_numpy(traj_sample_idx).to(_traj.device)]

        def _encode_traj(x):
            return lit_model_3.encode_trajectory(_x.unsqueeze(0)).squeeze(0)

        def _decode_traj(z):
            return lit_model_3.decode_trajectory(_z.unsqueeze(0)).squeeze(0)
        _z = lit_model_3.encode_trajectory(_traj_sampled)
        encoder_jacobian = vmap(jacrev(_encode_traj))(_traj_sampled)
        decoder_jacobian = vmap(jacfwd(_decode_traj))(_z)
    print(f'encoder_jacobian shape: {tuple(encoder_jacobian.shape)}')
    print(f'decoder_jacobian shape: {tuple(decoder_jacobian.shape)}')
    return decoder_jacobian, encoder_jacobian


@app.cell
def _(decoder_jacobian, encoder_jacobian, np, plt):
    _fig, _axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _axs[0].plot(decoder_jacobian.norm(dim=1).mean(dim=0).detach().cpu().numpy() / np.sqrt(decoder_jacobian.shape[1]))
    _axs[0].set_title('Decoder Jacobian Column-wise Norm Mean Over Samples\n(Normalized by sqrt(delay dimension))')
    _axs[0].set_ylabel('Mean Decoder Jacobian Norm')
    _axs[1].plot(encoder_jacobian.norm(dim=-1).mean(dim=0).detach().cpu().numpy() / np.sqrt(encoder_jacobian.shape[-1]))
    _axs[1].set_title('Encoder Jacobian Row-wise Norm Mean Over Samples\n(Normalized by sqrt(delay dimension))')
    _axs[1].set_ylabel('Mean Encoder Jacobian Norm')
    _axs[1].set_xlabel('Latent/Output Dimension (index)')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 21. Export HTML
    """)
    return


@app.cell
def _(WANDB_GROUP, WANDB_PROJECT, best_lambda):
    import subprocess
    from datetime import datetime

    now_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_file = f"reports/{now_str} - {WANDB_PROJECT}_{WANDB_GROUP}_lambda_{best_lambda}"
    cmd = f"uv run jupyter nbconvert --to html --output '{output_file}' '/home/eisenaj/code/JacobianODE/_jupyter/Sweep Analytics (Latent JacobianODE).ipynb'"
    # Uncomment to export:
    subprocess.run(cmd, shell=True)
    # print(f"Export command:\n{cmd}")
    return


if __name__ == "__main__":
    app.run()
