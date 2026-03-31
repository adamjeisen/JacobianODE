import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    return


@app.cell
def _():
    import os

    return (os,)


@app.cell
def _():
    # FILL THIS IN FOR YOUR OWN USE
    save_dir = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning"
    wandb_entity = "JacobianODE"
    return save_dir, wandb_entity


@app.cell
def _():
    data_loading_method = "from_file"
    # data_loading_method = "in_memory"
    return (data_loading_method,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom Data Loader Demo

    This notebook demonstrates how to use custom data arrays with the JacobianODE framework.

    **Data Format**: Custom data must be in the format **Trials x Timepoints x Variables**
    - Trials: Number of independent trajectories/trials
    - Timepoints: Number of time steps in each trajectory
    - Variables: Number of dimensions/variables observed at each time point
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Make Custom Data
    """)
    return


@app.cell
def _(os, save_dir):
    # '%autoreload 2' command supported automatically in marimo
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
        # Generate Lorenz data
        from omegaconf import OmegaConf
        from JacobianODE.jacobians import (
            load_config,
            initialize_config,
            make_trajectories,
            seed_everything,
        )

        # load the config
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
    print(f"Metadata: {custom_data.metadata}")
    return (
        custom_data,
        file_path,
        initialize_config,
        load_config,
        make_trajectories,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run JacobianODE with custom data
    """)
    return


@app.cell
def _(
    custom_data,
    data_loading_method,
    file_path,
    initialize_config,
    load_config,
    make_trajectories,
    save_dir,
    wandb_entity,
):
    from JacobianODE.jacobians import postprocess_data, normalize_data
    common_overrides = ['data=custom', 'data.name=LorenzCustom', 'data.postprocessing.obs_noise=0', 'training.lightning.loop_closure_weight=0.001', f'training.logger.save_dir={save_dir}', f'wandb_entity={wandb_entity}']
    if data_loading_method == 'from_file':
        cfg_1 = load_config(overrides=common_overrides, custom_dataset_loader='JacobianODE.jacobians.custom_data.load_timeseries_data', custom_dataset_loader_kwargs={'file_path': file_path}, data_dim=custom_data.n_dims)
        cfg_1 = initialize_config(cfg_1)
        eq_1, sol_1, dt_1 = make_trajectories(cfg_1)
    else:
        cfg_1 = load_config(overrides=common_overrides)
    # Common overrides for both loading methods
        cfg_1 = initialize_config(cfg_1, data_dim=custom_data.n_dims)
        eq_1, sol_1, dt_1 = make_trajectories(cfg_1, data=custom_data.values, dt=custom_data.dt)  # Load from a TimeSeriesData .npz file (dt is read from the file automatically)  # In-memory: pass data directly to make_trajectories (no file I/O needed)
    return cfg_1, dt_1, eq_1, normalize_data, postprocess_data, sol_1


@app.cell
def _(cfg_1, dt_1, normalize_data, postprocess_data, sol_1):
    values_raw = sol_1['values']
    print(f'Loaded data shape: {values_raw.shape}')  # Should be (Trials, Timepoints, Variables)
    raw_values_noise = None
    # For custom data, we typically don't have alternative noise data
    values = postprocess_data(cfg_1, values_raw, raw_values_noise, dt=dt_1)
    if cfg_1.data.postprocessing.normalize:
    # ----------------------------------------
    # POSTPROCESS DATA
        print(f'Normalizing data')
    # Postprocess data (adds noise, filtering, etc. based on config)
        values, mu, sigma = normalize_data(values)
    else:
        print(f'Not normalizing data')
        mu = 0
        sigma = 1
    print(f'Postprocessed data shape: {values.shape}')
    return mu, sigma, values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At this point, the array `values` is an np.ndarray of shape (Trials, Timepoints, Variables) and is ready to be used for training.

    The data is now ready for:
    1. Creating dataloaders: `create_dataloaders(cfg, values)`
    2. Training: `train_jacobians(cfg)`
    """)
    return


@app.cell
def _(cfg_1, dt_1, eq_1, mu, sigma, values):
    from JacobianODE.jacobians import create_dataloaders, setup_wandb, make_model, log_training_info, train_model
    _train_dataloader, _val_dataloader, _test_dataloader, trajs = create_dataloaders(cfg_1, values)
    name, _project, entity = setup_wandb(cfg_1, trajs, prompt_entity=False)
    lit_model = make_model(cfg_1, dt_1, eq=eq_1, project=_project, mu=mu, sigma=sigma, verbose=True)
    log_training_info(_train_dataloader, trajs, lit_model)
    train_model(cfg_1, lit_model, _train_dataloader, _val_dataloader, name, _project, entity=entity)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing with Custom Data
    """)
    return


@app.cell
def _(save_dir, wandb_entity):
    run_id = 'kya4cr5p'
    _project = 'LorenzCustom__JacobianODE'
    project_path = f'{wandb_entity}/{_project}' if wandb_entity else _project
    from JacobianODE.jacobians import load_run, load_checkpoint
    run, cfg_2, eq_2, dt_2, values_1, _train_dataloader, _val_dataloader, _test_dataloader, trajs_1, lit_model_1 = load_run(project_path, run_id=run_id, save_dir=save_dir, no_noise=True, generate_data=True, verbose=True)
    load_checkpoint(run, cfg_2, lit_model_1, save_dir=save_dir, verbose=True)
    lit_model_1.eval()
    return dt_2, eq_2, lit_model_1, trajs_1


@app.cell
def _(dt_2, eq_2, lit_model_1, trajs_1):
    # '%autoreload 2' command supported automatically in marimo
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from JacobianODE.jacobians.analysis import compute_lyaps
    from JacobianODE.dysts_sim.flows import Lorenz
    eq_true = Lorenz() if eq_2 is None else eq_2
    # True dynamics for Jacobian comparison (Lorenz generates this custom data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit_model_2 = lit_model_1.to(device)
    # 2. Get test data and move to GPU
    test_batch = trajs_1['test_trajs'].sequence.to(device)
    with torch.no_grad():
        jacs_pred = torch.stack([lit_model_2.compute_jacobians(test_batch[i].type(lit_model_2.dtype)) for i in range(test_batch.shape[0])]).cpu()

    # 3. Predicted Jacobians (model expects normalized input)
    def _to_numpy(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    x_orig = (test_batch.cpu().numpy() * _to_numpy(lit_model_2.sigma) + _to_numpy(lit_model_2.mu)).astype(np.float64)
    n_trials, n_time, n_dim = x_orig.shape
    t_indices = np.arange(n_time) * dt_2
    jacs_true_np = eq_true.jac(x_orig, t_indices)
    # 4. True Jacobians (denormalize for eq: x_orig = x_norm * sigma + mu)
    jacs_true = torch.from_numpy(jacs_true_np).float()
    print(f'Overall R2 score: {r2_score(jacs_pred.flatten(), jacs_true.flatten())}')
    jac_error = (jacs_pred - jacs_true).abs()
    error_per_point = jac_error.mean(dim=(2, 3)).numpy()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sc = axs[0].scatter(x_orig[:, :, 0].flatten(), x_orig[:, :, 1].flatten(), c=error_per_point.flatten(), s=1, cmap='viridis')
    cb = fig.colorbar(sc, ax=axs[0], label='Mean absolute Jacobian error')
    axs[0].set_title('Mean Abs Jacobian Error')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    lyaps_pred = compute_lyaps(torch.linalg.matrix_exp(jacs_pred * dt_2), dt=dt_2)
    # 5. Compute error and plot
    lyaps_true = compute_lyaps(torch.linalg.matrix_exp(jacs_true * dt_2), dt=dt_2)
    lyap_pred_mean = lyaps_pred.mean(dim=0)
    lyap_pred_sem = lyaps_pred.std(dim=0) / np.sqrt(lyaps_pred.shape[0])
    lyap_true_mean = lyaps_true.mean(dim=0)
    lyap_true_sem = lyaps_true.std(dim=0) / np.sqrt(lyaps_true.shape[0])
    x_vals = np.arange(1, lyap_pred_mean.shape[0] + 1)
    # First subplot: scatter of mean absolute Jacobian error
    axs[1].errorbar(x_vals, lyap_pred_mean, yerr=lyap_pred_sem, fmt='.', label='pred', markersize=12, capsize=0)
    axs[1].errorbar(x_vals, lyap_true_mean, yerr=lyap_true_sem, fmt='.', label='true', markersize=12, capsize=0)
    axs[1].set_xlabel('Lyapunov Exponent #')
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
