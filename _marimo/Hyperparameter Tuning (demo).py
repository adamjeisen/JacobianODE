import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameter Tuning Demo: Loop Closure Weight

    This notebook demonstrates the physics-informed model selection procedure
    for choosing `lambda_loop` (loop closure weight) as described in Appendix D.8.8.

    **Workflow:**
    1. Load Lorenz data from file
    2. Launch a Hydra grid sweep over `loop_closure_weight` values (SLURM jobs)
    3. After training completes, use the `tuning` module to select the best model
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
    import os

    # FILL THESE IN FOR YOUR OWN USE
    save_dir = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning"
    wandb_entity = "JacobianODE"
    return os, save_dir, wandb_entity


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Lorenz Data
    """)
    return


@app.cell
def _(os, save_dir):
    from JacobianODE.jacobians import TimeSeriesData

    curr_dir = os.getcwd()
    data_dir = os.path.join(curr_dir, 'example_data')
    os.makedirs(data_dir, exist_ok=True)
    obs_noise = 0.01
    file_path = os.path.join(data_dir, f'lorenz_data_obs_noise_{obs_noise}.npz')

    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        custom_data = TimeSeriesData.load(file_path)
    else:
        # Generate Lorenz data if it doesn't exist yet
        from omegaconf import OmegaConf
        from JacobianODE.jacobians import (
            load_config,
            initialize_config,
            make_trajectories,
            seed_everything,
        )

        overrides = [
            "data=dysts",
            "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
            f"data.postprocessing.obs_noise={obs_noise}",
            f"training.logger.save_dir={save_dir}",
        ]
        cfg = load_config(overrides=overrides)
        cfg = initialize_config(cfg)
        seed_everything(cfg.data.flow.random_state + cfg.training.run_number)
        eq, sol, dt = make_trajectories(cfg)

        custom_data = TimeSeriesData(
            values=sol['values'],
            dt=dt,
            metadata=OmegaConf.to_container(cfg.data, resolve=True),
        )
        custom_data.save(file_path)

    print(f"Data shape: {custom_data.shape}  (trials={custom_data.n_trials}, time={custom_data.n_timepoints}, dims={custom_data.n_dims})")
    print(f"dt: {custom_data.dt}")
    return (
        custom_data,
        file_path,
        initialize_config,
        load_config,
        make_trajectories,
        seed_everything,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Launch Hydra Grid Sweep over `loop_closure_weight`

    We use Hydra's `--multirun` mode with the SLURM launcher to submit one job
    per `loop_closure_weight` value. Each job trains a full model and logs to W&B.

    **The cell below generates and runs the sweep command.** After launching,
    wait for all SLURM jobs to finish before proceeding to step 3.

    > **NOTE**: This code assumes all hyperparameters other than `loop_closure_weight` are fixed. If you want to run multiple sweeps on different sets of hyperparameters, the code will need to be modified. Specifically, the code will need to check whether *your intended hyperparameters* have already been run, and it will subsequently need to pool all the runs specifically containing those hyperparameters. Right now, it only checks for runs with the intended `loop_closure_weight`.
    """)
    return


@app.cell
def _(custom_data, file_path, save_dir, wandb_entity):
    from JacobianODE.jacobians.tuning import DEFAULT_LAMBDA_LOOP_VALUES

    # Lambda values to sweep (from the tuning module defaults)
    # lambda_values = DEFAULT_LAMBDA_LOOP_VALUES
    lambda_values = [0]
    print(f"Sweep values: {lambda_values}")

    # Format for Hydra comma-separated sweep
    lambda_str = ",".join(str(v) for v in lambda_values)

    # Build the sweep command
    sweep_cmd = (
        f"python -m JacobianODE.jacobians.run_jacobians --multirun"
        f" data=custom"
        f" data.name=LorenzCustomv2"
        f" data.postprocessing.obs_noise=0",
        f"data.postprocessing.normalize=True", # ***************
        f" data.dataset_loader._target_=JacobianODE.jacobians.custom_data.load_timeseries_data"
        f" +data.dataset_loader.file_path={file_path}"
        f" data.flow.dim={custom_data.n_dims}"
        f" training.lightning.loop_closure_weight={lambda_str}"
        f" training.logger.save_dir={save_dir}"
        f" wandb_entity={wandb_entity}"
        f" slurm=default"
    )

    print("\nSweep command:")
    print(sweep_cmd)
    return lambda_values, sweep_cmd


@app.cell
def _(custom_data, file_path, lambda_values, save_dir, wandb_entity):
    import wandb
    project = 'LorenzCustomv2__JacobianODE'
    # Check which lambda values already have completed runs on W&B
    project_path = f'{wandb_entity}/{project}' if wandb_entity else project
    api = wandb.Api()
    try:
        existing_runs = api.runs(project_path)
        print(f'Found {len(existing_runs)} existing runs in {project_path}')
    except Exception as e:
        print(f'Could not query project (may not exist yet): {e}')
        existing_runs = []
    already_run = []
    remaining_lambdas = []
    for lam in lambda_values:
    # Identify lambda values that already have finished runs
        matching = [run for run in existing_runs if 'training' in run.config and 'lightning' in run.config['training'] and (abs(run.config['training']['lightning'].get('loop_closure_weight', -1) - lam) < 1e-10) and (run.state == 'finished')]
        if matching:
            already_run.append(lam)
            print(f'  lambda={lam}: already completed (run_id={matching[0].id})')
        else:
            remaining_lambdas.append(lam)
    if already_run:
        print(f'\nSkipping {len(already_run)} already-completed values: {already_run}')
    if remaining_lambdas:
        lambda_str_1 = ','.join((str(v) for v in remaining_lambdas))
        sweep_cmd_1 = f'python -m JacobianODE.jacobians.run_jacobians --multirun data=custom data.name=LorenzCustom data.postprocessing.obs_noise=0 data.dataset_loader._target_=JacobianODE.jacobians.custom_data.load_timeseries_data +data.dataset_loader.file_path={file_path} data.flow.dim={custom_data.n_dims} training.lightning.loop_closure_weight={lambda_str_1} training.logger.save_dir={save_dir} wandb_entity={wandb_entity} slurm=default'
        print(f'\nUpdated sweep command ({len(remaining_lambdas)} remaining):')
        print(sweep_cmd_1)
    else:
        sweep_cmd_1 = None
        print('\nAll lambda values already have finished runs -- no sweep needed!')  # Rebuild sweep command with only the remaining lambda values
    return (wandb,)


@app.cell
def _(subprocess, sweep_cmd):
    # Launch the sweep (submits SLURM jobs)
    # Uncomment the line below to actually run:
    if sweep_cmd is not None:
        #! {sweep_cmd}
        subprocess.call([str(sweep_cmd)])
    else:
        print("No sweep to launch — all runs already completed.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Select the Best Model (Post-hoc)

    After all sweep jobs finish, collect the W&B run IDs and use the tuning module
    to apply the physics-informed selection criteria:

    - **C1 (one-step error)**: model must beat persistence baseline
    - **C2 (loop closure)**: loop closure loss must be below `sqrt(n_dims)`
    - **C3 (eigenvalue)**: fraction of fast eigenvalues must be below threshold

    With relaxation rules when criteria conflict.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3a. Collect W&B Run IDs

    Find the runs from the sweep. You can get these from the W&B UI or
    use the W&B API to query by project and sweep parameters.
    """)
    return


@app.cell
def _(wandb, wandb_entity):
    project_1 = 'LorenzCustom__JacobianODE'
    project_path_1 = f'{wandb_entity}/{project_1}' if wandb_entity else project_1
    # Query W&B for the sweep runs
    api_1 = wandb.Api()
    try:
        all_runs = api_1.runs(project_path_1)
        print(f'Found {len(all_runs)} total runs in {project_path_1}')
    except Exception as e:
        print(f'Could not query project: {e}')
        all_runs = []
    return all_runs, project_path_1


@app.cell
def _(all_runs, lambda_values):
    # Check for crashed/failed runs and delete them from W&B
    crashed_lambdas = []
    for run in all_runs:
        if run.state in ('crashed', 'failed'):
            lam_1 = None
            if 'training' in run.config and 'lightning' in run.config['training']:
                lam_1 = run.config['training']['lightning'].get('loop_closure_weight', None)
            print(f'CRASHED: run_id={run.id} (lambda={lam_1}) -- deleting from W&B')
            run.delete()
            if lam_1 is not None:
                crashed_lambdas.append(lam_1)
    if crashed_lambdas:
        needs_rerun = []
        for lam_1 in crashed_lambdas:
            has_finished = any((run for run in all_runs if run.state == 'finished' and 'training' in run.config and ('lightning' in run.config['training']) and (abs(run.config['training']['lightning'].get('loop_closure_weight', -1) - lam_1) < 1e-10)))  # Check which crashed lambdas still have a finished run available
            if not has_finished:
                needs_rerun.append(lam_1)
        print(f'\nDeleted {len(crashed_lambdas)} crashed run(s).')
        if needs_rerun:
            print(f'Re-run the sweep to retry lambda={needs_rerun}')
        else:
            print('All crashed lambdas have other finished runs -- no re-runs needed.')
    else:
        print('No crashed runs found.')
    sweep_run_ids = []
    sweep_lambdas = []
    for lam_1 in lambda_values:
        matching_1 = [run for run in all_runs if 'training' in run.config and 'lightning' in run.config['training'] and (abs(run.config['training']['lightning'].get('loop_closure_weight', -1) - lam_1) < 1e-10) and (run.state == 'finished')]
        if matching_1:
            best_run = matching_1[0]
            sweep_run_ids.append(best_run.id)
            sweep_lambdas.append(lam_1)
            print(f'lambda={lam_1}: run_id={best_run.id} ({best_run.name})')
        else:
            print(f'lambda={lam_1}: NO MATCHING RUN FOUND')
    # Match runs to lambda values by their loop_closure_weight config
    print(f'\nFound {len(sweep_run_ids)} / {len(lambda_values)} sweep runs')  # Take the most recent run for this lambda
    return sweep_lambdas, sweep_run_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3b. Run Model Selection

    Load each checkpoint, compute diagnostic metrics, and apply the selection algorithm.
    """)
    return


@app.cell
def _(
    custom_data,
    initialize_config,
    load_config,
    make_trajectories,
    save_dir,
    seed_everything,
    wandb_entity,
):
    import torch
    import numpy as np
    from JacobianODE.jacobians import postprocess_data, create_dataloaders, load_run, load_checkpoint
    from JacobianODE.jacobians.tuning.criteria import compute_all_diagnostics, DiagnosticMetrics
    from JacobianODE.jacobians.tuning.selection import select_best_model
    common_overrides = ['data=custom', 'data.name=LorenzCustom', 'data.postprocessing.obs_noise=0', 'data.postprocessing.normalize=True', f'training.logger.save_dir={save_dir}', f'wandb_entity={wandb_entity}']
    cfg_1 = load_config(overrides=common_overrides)
    cfg_1 = initialize_config(cfg_1, data_dim=custom_data.n_dims)
    seed_everything(cfg_1.data.flow.random_state + cfg_1.training.run_number)
    eq_1, sol_1, dt_1 = make_trajectories(cfg_1, data=custom_data.values, dt=custom_data.dt)
    result = postprocess_data(cfg_1, sol_1['values'])
    values = result.values
    train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(cfg_1, values)
    train_trajs = trajs['train_trajs'].sequence
    n_dims = custom_data.n_dims
    print(f'Data dimensionality: {n_dims}')
    print(f'sqrt(n_dims) = {np.sqrt(n_dims):.4f}  (C2 threshold)')
    # ----------------------------------------
    # Prepare data for evaluation
    # Persistence baseline from training data
    print(f'dt = {dt_1:.6f}')
    return (
        compute_all_diagnostics,
        dt_1,
        load_checkpoint,
        load_run,
        n_dims,
        np,
        select_best_model,
        torch,
        train_trajs,
        trajs,
    )


@app.cell
def _(train_trajs):
    import matplotlib.pyplot as plt
    plt.plot(train_trajs[0])
    return (plt,)


@app.cell
def _(
    compute_all_diagnostics,
    dt_1,
    load_checkpoint,
    load_run,
    project_path_1,
    save_dir,
    sweep_lambdas,
    sweep_run_ids,
    torch,
):
    # ----------------------------------------
    # Load each model and compute diagnostics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_batches = 100
    all_diagnostics = []
    for i, run_id in enumerate(sweep_run_ids):
        print(f'\n[{i + 1}/{len(sweep_run_ids)}] Loading run {run_id} (lambda={sweep_lambdas[i]})')
        generate_data = i == 0
        run_obj, run_cfg, _, _, _, _, val_dl, _, _, lit_model = load_run(project_path_1, run_id=run_id, save_dir=save_dir, generate_data=generate_data, dt=dt_1, verbose=False)
        if generate_data:
            eval_val_dl = val_dl  # Load run and checkpoint
        load_checkpoint(run_obj, run_cfg, lit_model, save_dir=save_dir, verbose=False)  # Only generate data for first run
        lit_model.eval()
        lit_model = lit_model.to(device)
        metrics = compute_all_diagnostics(lit_model, eval_val_dl, dt_1, n_batches=n_batches, use_loop_closure=True)
        all_diagnostics.append(metrics)
        print(f'  one_step_error:          {metrics.one_step_error:.6f}')
        print(f'  loop_closure_loss:       {metrics.loop_closure_loss:.6f}')  # Reuse for all models
        print(f'  fast_eigenvalue_fraction: {metrics.fast_eigenvalue_fraction:.6f}')
        print(f'  trajectory_val_loss:     {metrics.trajectory_val_loss:.6f}')
        lit_model.cpu()
        torch.cuda.empty_cache()  # Compute all diagnostic metrics  # Free GPU memory
    return (all_diagnostics,)


@app.cell
def _(
    all_diagnostics,
    n_dims,
    persistence,
    select_best_model,
    sweep_lambdas,
    sweep_run_ids,
):
    # ----------------------------------------
    # Run model selection
    result_1 = select_best_model(all_diagnostics, persistence_baseline=persistence, n_dims=n_dims, eigenvalue_threshold=0.001, use_loop_closure=True)
    print('=' * 60)
    print('MODEL SELECTION RESULT')
    print('=' * 60)
    print(f'Best lambda:    {sweep_lambdas[result_1.best_index]}')
    print(f'Best run ID:    {sweep_run_ids[result_1.best_index]}')
    print(f'Best traj loss: {result_1.best_metrics.trajectory_val_loss:.6f}')
    print(f'\nCriteria applied (after relaxation): {result_1.criteria_applied}')
    print(f'Surviving models: {len(result_1.surviving_indices)} / {len(all_diagnostics)}')
    print(f'Surviving lambda values: {[sweep_lambdas[i] for i in result_1.surviving_indices]}')
    print('\nExclusion details (before relaxation):')
    for criterion, excluded in result_1.exclusion_details.items():
        if excluded:
            excluded_lambdas = [sweep_lambdas[i] for i in excluded]
            print(f'  {criterion}: excluded lambdas {excluded_lambdas}')
        else:
            print(f'  {criterion}: no exclusions')
    return (result_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c. Visualize Selection
    """)
    return


@app.cell
def _(all_diagnostics, n_dims, np, persistence, plt, result_1, sweep_lambdas):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    one_step_errors = [m.one_step_error for m in all_diagnostics]
    loop_closure_losses = [m.loop_closure_loss for m in all_diagnostics]
    eig_fracs = [m.fast_eigenvalue_fraction for m in all_diagnostics]
    # Extract metrics arrays
    traj_losses = [m.trajectory_val_loss for m in all_diagnostics]
    colors = ['tab:green' if i in result_1.surviving_indices else 'tab:red' for i in range(len(all_diagnostics))]
    x_labels = [str(v) for v in sweep_lambdas]
    x_pos = range(len(sweep_lambdas))
    axes[0, 0].bar(x_pos, one_step_errors, color=colors)
    # Color: green for survivors, red for excluded, star for best
    axes[0, 0].axhline(y=persistence, color='k', linestyle='--', label=f'Persistence baseline ({persistence:.4f})')
    axes[0, 0].set_ylabel('One-step error')
    axes[0, 0].set_title('C1: One-step Error')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0, 0].legend(fontsize=8)
    # Panel 1: One-step error vs persistence baseline (C1)
    axes[0, 0].set_yscale('log')
    axes[0, 1].bar(x_pos, loop_closure_losses, color=colors)
    axes[0, 1].axhline(y=np.sqrt(n_dims), color='k', linestyle='--', label=f'sqrt(n_dims) = {np.sqrt(n_dims):.2f}')
    axes[0, 1].set_ylabel('Loop closure loss')
    axes[0, 1].set_title('C2: Loop Closure Loss')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_yscale('log')
    # Panel 2: Loop closure loss vs sqrt(n_dims) (C2)
    axes[1, 0].bar(x_pos, eig_fracs, color=colors)
    axes[1, 0].axhline(y=0.001, color='k', linestyle='--', label='Threshold (0.001)')
    axes[1, 0].set_ylabel('Fast eigenvalue fraction')
    axes[1, 0].set_title('C3: Eigenvalue Criterion')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1, 0].set_xlabel('loop_closure_weight')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(x_pos, traj_losses, color=colors)
    # Panel 3: Fast eigenvalue fraction (C3)
    if result_1.best_index is not None:
        axes[1, 1].bar(result_1.best_index, traj_losses[result_1.best_index], color='gold', edgecolor='black', linewidth=2, label='Selected')
    axes[1, 1].set_ylabel('Trajectory val loss')
    axes[1, 1].set_title('Trajectory Val Loss (selection target)')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1, 1].set_xlabel('loop_closure_weight')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_yscale('log')
    # Panel 4: Trajectory val loss (selection target)
    fig.suptitle('Hyperparameter Sweep: Loop Closure Weight Selection', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3d. Load the Best Model for Downstream Use
    """)
    return


@app.cell
def _(
    load_checkpoint,
    load_run,
    project_path_1,
    result_1,
    save_dir,
    sweep_lambdas,
    sweep_run_ids,
):
    # Load the selected best model
    best_run_id = sweep_run_ids[result_1.best_index]
    best_lambda = sweep_lambdas[result_1.best_index]
    print(f'Loading best model: run_id={best_run_id}, lambda={best_lambda}')
    run_obj_1, run_cfg_1, eq_2, _, _, _, _, _, _, lit_model_1 = load_run(project_path_1, run_id=best_run_id, save_dir=save_dir, no_noise=True, generate_data=True, verbose=True)
    load_checkpoint(run_obj_1, run_cfg_1, lit_model_1, save_dir=save_dir, verbose=True)
    lit_model_1.eval()
    print(f'\nBest model loaded and ready for analysis.')
    print(lit_model_1)
    return eq_2, lit_model_1


@app.cell
def _(dt_1, eq_2, lit_model_1, np, plt, torch, trajs):
    # '%autoreload 2' command supported automatically in marimo
    from sklearn.metrics import r2_score
    from JacobianODE.jacobians.analysis import compute_lyaps
    from JacobianODE.dysts_sim.flows import Lorenz
    eq_true = Lorenz() if eq_2 is None else eq_2
    device_1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit_model_2 = lit_model_1.to(device_1)
    test_batch = trajs['test_trajs'].sequence.to(device_1)
    # True dynamics for Jacobian comparison (Lorenz generates this custom data)
    with torch.no_grad():
        jacs_pred = torch.stack([lit_model_2.compute_jacobians(test_batch[i].type(lit_model_2.dtype)) for i in range(test_batch.shape[0])]).cpu()
    # 2. Get test data and move to GPU

    def _to_numpy(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    x_orig = (test_batch.cpu().numpy() * _to_numpy(lit_model_2.sigma) + _to_numpy(lit_model_2.mu)).astype(np.float64)
    # 3. Predicted Jacobians (model expects normalized input)
    n_trials, n_time, n_dim = x_orig.shape
    t_indices = np.arange(n_time) * dt_1
    jacs_true_np = eq_true.jac(x_orig, t_indices)
    jacs_true = torch.from_numpy(jacs_true_np).float()
    print(f'Overall R2 score: {r2_score(jacs_pred.flatten(), jacs_true.flatten())}')
    jac_error = (jacs_pred - jacs_true).abs()
    # 4. True Jacobians (denormalize for eq: x_orig = x_norm * sigma + mu)
    error_per_point = jac_error.mean(dim=(2, 3)).numpy()
    fig_1, axs = plt.subplots(1, 2, figsize=(12, 5))
    sc = axs[0].scatter(x_orig[:, :, 0].flatten(), x_orig[:, :, 1].flatten(), c=error_per_point.flatten(), s=1, cmap='viridis')
    cb = fig_1.colorbar(sc, ax=axs[0], label='Mean absolute Jacobian error')
    axs[0].set_title('Mean Abs Jacobian Error')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    lyaps_pred = compute_lyaps(torch.linalg.matrix_exp(jacs_pred * dt_1), dt=dt_1)
    lyaps_true = compute_lyaps(torch.linalg.matrix_exp(jacs_true * dt_1), dt=dt_1)
    lyap_pred_mean = lyaps_pred.mean(dim=0)
    lyap_pred_sem = lyaps_pred.std(dim=0) / np.sqrt(lyaps_pred.shape[0])
    # 5. Compute error and plot
    lyap_true_mean = lyaps_true.mean(dim=0)
    lyap_true_sem = lyaps_true.std(dim=0) / np.sqrt(lyaps_true.shape[0])
    x_vals = np.arange(1, lyap_pred_mean.shape[0] + 1)
    axs[1].errorbar(x_vals, lyap_pred_mean, yerr=lyap_pred_sem, fmt='.', label='pred', markersize=12, capsize=0)
    axs[1].errorbar(x_vals, lyap_true_mean, yerr=lyap_true_sem, fmt='.', label='true', markersize=12, capsize=0)
    axs[1].set_xlabel('Lyapunov Exponent #')
    # First subplot: scatter of mean absolute Jacobian error
    axs[1].set_ylabel('Lyapunov Exponent')
    axs[1].set_title('Predicted vs True Lyapunov Exponents')
    axs[1].legend()
    plt.tight_layout()
    # Second subplot: Lyapunov spectrum
    # make the size of the points bigger
    # add error bars on the estimated lyaps (standard error of the mean)
    # capsize=0 removes horizontal caps so error bars are purely vertical (along y)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
