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
    # Sequence Encoder Comparison: Lorenz Attractor from Partial Observations

    This notebook compares several sequence autoencoder architectures for
    reconstructing the Lorenz attractor from **partial observations** (only
    the $x$ coordinate).

    Instead of requiring a hand-tuned delay embedding, each architecture
    takes a raw $(T, D)$ window and encodes it into an $N$-dimensional latent
    space, which is then decoded to **reconstruct the full input window**.

    **Architectures tested:**
    1. Transformer (no positional encoding)
    2. Transformer (with positional encoding)
    3. State Space Model / S4 (no positional encoding)
    4. State Space Model / S4 (with positional encoding)
    5. Temporal Convolutional Network (TCN)
    6. TCN + Spatial Convolutions

    All models use **FNN regularization** on the $N$-dimensional latent.
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
    import matplotlib.pyplot as plt
    import torch
    import time
    from collections import OrderedDict

    from JacobianODE.dysts_sim.flows import Lorenz
    from JacobianODE.fnn import (
        TransformerEmbedding,
        SSMEmbedding,
        TCNEmbedding,
        TCNSpatialEmbedding,
        FNN,
        compute_variances,
        compute_s_dim,
    )

    plt.rcParams.update({
        'figure.figsize': (12, 4),
        'font.size': 11,
    })

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')
    return (
        FNN,
        Lorenz,
        OrderedDict,
        SSMEmbedding,
        TCNEmbedding,
        TCNSpatialEmbedding,
        TransformerEmbedding,
        compute_variances,
        np,
        plt,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Lorenz Data
    """)
    return


@app.cell
def _(Lorenz, np):
    # Generate Lorenz trajectories
    lorenz = Lorenz()
    data_dict = lorenz.make_trajectory(
        n_periods=12,
        pts_per_period=100,
        num_ics=16,
        new_ic_mode='random',
        traj_offset_sd=0.2,
        return_times=True,
    )

    full_data = data_dict['values']  # (num_ics, T, 3)
    dt = data_dict['dt']
    print(f'Full data shape: {full_data.shape}, dt = {dt:.4f}')

    # Add small observation noise
    obs_noise = 0.01
    np.random.seed(42)
    full_data_noisy = full_data + obs_noise * np.random.randn(*full_data.shape)

    # Partial observation: only x coordinate (D=1)
    partial_data = full_data_noisy[:, :, [0]]  # (num_ics, T, 1)
    print(f'Partial observation shape: {partial_data.shape}')

    # Train / test split by initial conditions
    n_train = 12
    train_data = partial_data[:n_train]
    test_data = partial_data[n_train:]
    train_full = full_data_noisy[:n_train]
    test_full = full_data_noisy[n_train:]
    print(f'Train: {train_data.shape}, Test: {test_data.shape}')
    return (
        full_data_noisy,
        n_train,
        test_data,
        test_full,
        train_data,
        train_full,
    )


@app.cell
def _(n_train, plt, train_data, train_full):
    # Visualize the Lorenz attractor and partial observation
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    for _i in range(min(4, n_train)):
    # 3D projection (x vs z)
        _axes[0].plot(train_full[_i, :, 0], train_full[_i, :, 2], alpha=0.5, lw=0.5)
    _axes[0].set_xlabel('x')
    _axes[0].set_ylabel('z')
    _axes[0].set_title('Lorenz attractor (x vs z)')
    for _i in range(min(4, n_train)):
    # Partial observation time series
        _axes[1].plot(train_data[_i, :200, 0], alpha=0.7, lw=0.8)
    _axes[1].set_xlabel('Time step')
    _axes[1].set_ylabel('x')
    _axes[1].set_title('Partial observation (x only, first 200 steps)')
    for _i in range(min(4, n_train)):
    # x-y phase space
        _axes[2].plot(train_full[_i, :, 0], train_full[_i, :, 1], alpha=0.5, lw=0.5)
    _axes[2].set_xlabel('x')
    _axes[2].set_ylabel('y')
    _axes[2].set_title('Lorenz attractor (x vs y)')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Define Architectures and Hyperparameters
    """)
    return


@app.cell
def _(
    FNN,
    OrderedDict,
    SSMEmbedding,
    TCNEmbedding,
    TCNSpatialEmbedding,
    TransformerEmbedding,
):
    # Shared hyperparameters
    N_LATENT = 10          # Latent dimension (Lorenz has 3 dims, but we give extra capacity)
    TIME_WINDOW = 64      # Input sequence length
    N_FEATURES = 1        # Partial observation dimension (x only)
    FNN_STRENGTH = 0.1   # FNN regularization strength
    # TRAIN_STEPS = 200     # Training epochs
    TRAIN_STEPS = 50        # Training epochs
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    SEED = 42

    fnn_reg = FNN(FNN_STRENGTH)

    # Define all model configurations
    model_configs = OrderedDict([
        ('Transformer (no PE)', {
            'class': TransformerEmbedding,
            'kwargs': dict(
                use_positional_encoding=False,
                d_model=64, n_heads=4, n_layers=3,
                dim_feedforward=128, dropout=0.1,
            ),
        }),
        ('Transformer (PE)', {
            'class': TransformerEmbedding,
            'kwargs': dict(
                use_positional_encoding=True,
                d_model=64, n_heads=4, n_layers=3,
                dim_feedforward=128, dropout=0.1,
            ),
        }),
        ('SSM (no PE)', {
            'class': SSMEmbedding,
            'kwargs': dict(
                use_positional_encoding=False,
                d_model=64, d_state=64, n_layers=3, dropout=0.1,
            ),
        }),
        ('SSM (PE)', {
            'class': SSMEmbedding,
            'kwargs': dict(
                use_positional_encoding=True,
                d_model=64, d_state=64, n_layers=3, dropout=0.1,
            ),
        }),
        ('TCN', {
            'class': TCNEmbedding,
            'kwargs': dict(
                n_channels=64, kernel_size=7, n_layers=4, dropout=0.1,
            ),
        }),
        ('TCN + Spatial', {
            'class': TCNSpatialEmbedding,
            'kwargs': dict(
                n_channels=64, kernel_size_temporal=7,
                kernel_size_spatial=3, n_layers=4, dropout=0.1,
            ),
        }),
    ])

    print(f'{len(model_configs)} architectures to compare')
    return (
        BATCH_SIZE,
        FNN_STRENGTH,
        LEARNING_RATE,
        N_FEATURES,
        N_LATENT,
        SEED,
        TIME_WINDOW,
        TRAIN_STEPS,
        model_configs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Train All Models
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    FNN,
    FNN_STRENGTH,
    LEARNING_RATE,
    N_FEATURES,
    N_LATENT,
    OrderedDict,
    SEED,
    TIME_WINDOW,
    TRAIN_STEPS,
    compute_variances,
    model_configs,
    np,
    test_data,
    time,
    train_data,
):
    results = OrderedDict()
    for _name, _cfg in model_configs.items():
        print(f'\n{'=' * 60}')
        print(f'Training: {_name}')
        print(f'{'=' * 60}')
        model = _cfg['class'](n_latent=N_LATENT, time_window=TIME_WINDOW, n_features=N_FEATURES, random_state=SEED, latent_regularizer=FNN(FNN_STRENGTH), **_cfg['kwargs'])
        t0 = time.time()
        model.fit(train_data, train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, verbose=1, optimizer='adamw')
        train_time = time.time() - t0
        _recon = model.reconstruct(test_data)
        _latent = model.transform(test_data)
        from JacobianODE.fnn.sequence_models import sliding_windows
        _test_std = model._standardize(test_data)
        _x_true = sliding_windows(_test_std, TIME_WINDOW)
        _mse = float(np.mean((_recon - _x_true) ** 2))
        _latent_flat = _latent.reshape(-1, _latent.shape[-1])
        _norm_var = compute_variances(_latent_flat, normalize=True)
        results[_name] = {'model': model, 'mse': _mse, 'norm_var': _norm_var, 'latent': _latent, 'recon': _recon, 'x_true': _x_true, 'train_time': train_time, 'n_params': model.count_parameters(), 'history': model.train_history}
        print(f'  Test Recon MSE: {_mse:.6f}')
        print(f'  Params:   {model.count_parameters():,}')
        print(f'  Time:     {train_time:.1f}s')
    print(f'\nAll {len(results)} models trained.')  # Evaluate reconstruction on test set  # (n_windows, TIME_WINDOW, 1)  # (n_windows, TIME_WINDOW, N_LATENT)  # Compute reconstruction MSE against standardized input windows  # Flatten per-timestep latent for variance analysis: (n_windows, T, D') -> (n_windows*T, D')
    return results, sliding_windows


@app.cell
def _(np, plt, results):
    _model_names = list(results.keys())
    _n_models = len(_model_names)
    n_examples = 6
    _n_windows = results[_model_names[0]]['x_true'].shape[0]
    if _n_windows < n_examples:
        example_idxs = np.arange(_n_windows)
    else:
        example_idxs = np.linspace(0, _n_windows - 1, n_examples, dtype=int)
    time_axis = np.arange(results[_model_names[0]]['x_true'].shape[1])
    _fig, _axes = plt.subplots(_n_models, n_examples, figsize=(18, 13), sharex='col', sharey='all')
    if _axes.ndim == 1:
        _axes = _axes.reshape(_n_models, n_examples)
    for row, _name in enumerate(_model_names):
        _recon = results[_name]['recon']
        _x_true = results[_name]['x_true']
        if _recon.shape[-1] == 1:
            _recon = _recon.squeeze(-1)
        if _x_true.shape[-1] == 1:
            _x_true = _x_true.squeeze(-1)
        for col, _idx in enumerate(example_idxs):
            _ax = _axes[row, col]
            _ax.plot(time_axis, _x_true[_idx], label='True', color='black', linewidth=1.5, alpha=0.8)
            _ax.plot(time_axis, _recon[_idx], label='Recon', color='#e74c3c', linewidth=1.2, alpha=0.8)
            if row == 0:
                _ax.set_title(f'Window {_idx}', fontsize=11)
            if col == 0:
                _ax.set_ylabel(_name, fontsize=11)
            if row == _n_models - 1:
                _ax.set_xlabel('timestep')
            _ax.spines['top'].set_visible(False)
            _ax.spines['right'].set_visible(False)
            if row == 0 and col == 0:
                _ax.legend(frameon=False, fontsize=9, loc='upper right')
    plt.tight_layout(h_pad=1.7)
    plt.suptitle('Reconstruction: Each row is a model, each col is a test window', fontsize=15, y=1.03)
    plt.show()
    return


@app.cell
def _(test_full):
    test_full.reshape(-1, 3)
    return


@app.cell
def _(test_data):
    test_data.shape
    return


@app.cell
def _(results, test_data):
    results["Transformer (no PE)"]['model'].transform(test_data).shape
    return


@app.cell
def _(plt, results):
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D
    _model_names = list(results.keys())
    # List of model names to match the order of your plots/results
    _n_models = len(_model_names)
    assert _n_models == 6, 'Expected 6 models!'
    _fig = plt.figure(figsize=(18, 14))
    for _i, _name in enumerate(_model_names):
        _latent = results[_name]['latent']
        _latent_flat = _latent.reshape(-1, _latent.shape[-1])
        _pca = PCA(n_components=3)  # shape: (n_windows, T, D_latent)
        lat_3d = _pca.fit_transform(_latent_flat)
        _ax = _fig.add_subplot(3, 2, _i + 1, projection='3d')
        _ax.scatter(lat_3d[:, 0], lat_3d[:, 1], lat_3d[:, 2], s=0.5, alpha=0.3, c='tab:blue')  # Fit PCA on this model's latents
        _ax.set_title(f'{_name}\nLatent Space (PCA 3D)', fontsize=12)
        _ax.set_xlabel('PC1')
        _ax.set_ylabel('PC2')
        _ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.suptitle('3D PCA of Latent Spaces Across All Models', fontsize=18, y=1.04)
    plt.show()
    return (PCA,)


@app.cell
def _(np):
    def smoothness_relative(x, normalize_dim=True, normalize_time=True, reduction='mean', eps=1e-12):
        """
        Relative smoothness for (B, T, D).

        normalize_dim:
            Divide by D to avoid scaling with dimension.

        normalize_time:
            Average over time instead of summing.

        reduction:
            "mean", "sum", or "none"
        """
        x = np.asarray(x)
        B, T, D = x.shape
        diffs = x[:, 1:, :] - x[:, :-1, :]
        num = np.sum(diffs ** 2, axis=2)
        den = np.sum(x ** 2, axis=2)
        if normalize_dim:
            num = num / D
            den = den / D
        if normalize_time:
            num = num.mean(axis=1)
            den = den.mean(axis=1)
        else:  # (B, T-1, D)
            num = num.sum(axis=1)
            den = den.sum(axis=1)  # Euclidean squared norms
        smooth = num / (den + eps)  # (B, T-1)
        if reduction == 'mean':  # (B, T)
            return smooth.mean()
        elif reduction == 'sum':
            return smooth.sum()
        else:
            return smooth

    return (smoothness_relative,)


@app.cell
def _(TIME_WINDOW, sliding_windows, smoothness_relative, test_full):
    smoothness_relative(sliding_windows(test_full, TIME_WINDOW))
    return


@app.cell
def _(np, plt, results):
    _model_names = list(results.keys())
    _n_models = len(_model_names)
    _n_windows = 6
    _fig, _axes = plt.subplots(nrows=6, ncols=6, figsize=(18, 18), sharex='col', sharey='row')
    num_available_windows = min([results[m]['latent'].shape[0] for m in _model_names])
    rand_idx = np.random.choice(num_available_windows, size=6, replace=False)
    for _i, model_1 in enumerate(_model_names):
        latents = results[model_1]['latent']
        for j, _idx in enumerate(rand_idx):
            _ax = _axes[_i, j]
            for d in range(latents.shape[-1]):
                _ax.plot(latents[_idx, :, d], alpha=0.7)
            if _i == 0:
                _ax.set_title(f'Window idx {_idx}')
            if j == 0:
                _ax.set_ylabel(model_1)
            _ax.tick_params(labelbottom=_i == 5, labelleft=j == 0)
    for _i in range(len(_model_names), 6):
        for j in range(6):
            _axes[_i, j].axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    TIME_WINDOW,
    plt,
    results,
    sliding_windows,
    smoothness_relative,
    test_full,
):
    # Compute relative smoothness for each model
    _model_names = list(results.keys())
    smoothness_values = [smoothness_relative(results[_name]['latent']) for _name in _model_names]
    test_smoothness = smoothness_relative(sliding_windows(test_full, TIME_WINDOW))
    # Get relative smoothness for test_full sliding windowed
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _bars = _ax.bar(range(len(_model_names)), smoothness_values, color='#3498db', edgecolor='white', linewidth=0.8)
    # Bar plot
    _ax.axhline(test_smoothness, color='red', linestyle='--', label='Test Full (Sliding Windowed)')
    _ax.set_xticks(range(len(_model_names)))
    _ax.set_xticklabels(_model_names, rotation=30, ha='right')
    _ax.set_ylabel('Relative Smoothness')
    _ax.set_title('Relative Smoothness of Latents (test set)')
    for _bar, value in zip(_bars, smoothness_values):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height(), f'{value:.2g}', ha='center', va='bottom', fontsize=9)
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Results Comparison
    """)
    return


@app.cell
def _(results):
    # Print summary table
    print(f'{'Model':25s} {'Recon MSE':>12s} {'Params':>10s} {'Train (s)':>10s}')
    print('-' * 60)
    for _name, _r in results.items():
        print(f'{_name:25s} {_r['mse']:12.6f} {_r['n_params']:10,} {_r['train_time']:10.1f}')
    best_name = min(results, key=lambda k: results[k]['mse'])
    # Identify best model
    print(f'\nBest model: {best_name} (Recon MSE = {results[best_name]['mse']:.6f})')
    return (best_name,)


@app.cell
def _(best_name, plt, results):
    # Bar chart of test reconstruction MSE
    _names = list(results.keys())
    mses = [results[n]['mse'] for n in _names]
    colors = ['#2ecc71' if n == best_name else '#3498db' for n in _names]
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _bars = _ax.bar(range(len(_names)), mses, color=colors, edgecolor='white', linewidth=0.8)
    _ax.set_xticks(range(len(_names)))
    _ax.set_xticklabels(_names, rotation=30, ha='right')
    _ax.set_ylabel('Test MSE (window reconstruction)')
    _ax.set_title('Architecture Comparison: Window Reconstruction on Lorenz (x only)')
    for _bar, mse_val in zip(_bars, mses):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height(), f'{mse_val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Training Curves
    """)
    return


@app.cell
def _(plt, results):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    for _name, _r in results.items():
        _axes[0].plot(_r['history']['recon_loss'], label=_name, alpha=0.8)
        _axes[1].plot(_r['history']['reg_loss'], label=_name, alpha=0.8)
    _axes[0].set_xlabel('Epoch')
    _axes[0].set_ylabel('Reconstruction Loss (MSE)')
    _axes[0].set_title('Reconstruction Loss')
    _axes[0].legend(fontsize=8)
    _axes[0].set_yscale('log')
    _axes[1].set_xlabel('Epoch')
    _axes[1].set_ylabel('FNN Regularization Loss')
    _axes[1].set_title('FNN Regularization Loss')
    _axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Latent Space Analysis
    """)
    return


@app.cell
def _(N_LATENT, compute_variances, full_data_noisy, np, plt, results):
    # Normalized variance spectrum of latent dimensions
    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _name, _r in results.items():
        _ax.plot(range(1, len(_r['norm_var']) + 1), _r['norm_var'], 'o-', label=_name, alpha=0.8)
    var_true = compute_variances(full_data_noisy.reshape(-1, 3), normalize=True)
    var_true = np.pad(var_true, (0, N_LATENT - len(var_true)), mode='constant', constant_values=0)
    # Add normalized variance of the full 3D Lorenz system for reference|
    # Compute variance over the dataset (assume shape (n_samples, 3))
    _ax.plot(range(1, N_LATENT + 1), var_true, '-', color='k', marker='s', markersize=7, label='True 3D Lorenz (xyz)', alpha=0.8)
    _ax.set_xlabel('Latent Dimension')
    _ax.set_ylabel('Normalized Variance')
    _ax.set_title('Latent Variance Spectrum (FNN-regularized)')
    _ax.legend(fontsize=8)
    _ax.set_xticks(range(1, N_LATENT + 1))
    plt.tight_layout()
    # ax.set_yscale('log')
    plt.show()
    return (var_true,)


@app.cell
def _(PCA, plt, results):
    _n_models = len(results)
    _fig, _axes = plt.subplots(1, _n_models, figsize=(4 * _n_models, 4))
    if _n_models == 1:
        _axes = [_axes]
    for _ax, (_name, _r) in zip(_axes, results.items()):
        lat = _r['latent']
        lat_flat = lat.reshape(-1, lat.shape[-1])
        _pca = PCA(n_components=2)
        lat_2d = _pca.fit_transform(lat_flat)
        _ax.scatter(lat_2d[:, 0], lat_2d[:, 1], s=0.5, alpha=0.3, c='tab:blue')
        _ax.set_title(_name, fontsize=9)
        _ax.set_xlabel('PC1')
        _ax.set_ylabel('PC2')
    _fig.suptitle('Latent Space (PCA projection)', y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Reconstruction Quality
    """)
    return


@app.cell
def _(TIME_WINDOW, np, plt, results):
    # Reconstructed vs true (using middle time step of each window)
    _n_models = len(results)
    _fig, _axes = plt.subplots(1, _n_models, figsize=(4 * _n_models, 4))
    if _n_models == 1:
        _axes = [_axes]
    for _ax, (_name, _r) in zip(_axes, results.items()):
        mid = TIME_WINDOW // 2
        recon_mid = _r['recon'][:, mid, 0]  # Compare the middle time step of each window
        true_mid = _r['x_true'][:, mid, 0]
        n_plot = min(2000, len(recon_mid))
        _idx = np.random.choice(len(recon_mid), n_plot, replace=False)
        _ax.scatter(true_mid[_idx], recon_mid[_idx], s=1, alpha=0.3)  # Subsample for clarity
        lims = [true_mid.min(), true_mid.max()]
        _ax.plot(lims, lims, 'r--', lw=0.8, alpha=0.5)
        _ax.set_title(f'{_name}\nMSE={_r['mse']:.4f}', fontsize=9)
        _ax.set_xlabel('True')
        _ax.set_ylabel('Reconstructed')
        _ax.set_aspect('equal')
    _fig.suptitle('Window Reconstruction: True vs Reconstructed (mid-window step)', y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(TIME_WINDOW, plt, results, test_data):
    _fig, _axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)
    test_single = test_data[0]
    t_plot = 200
    for _ax, (_name, _r) in zip(_axes, results.items()):
        model_2 = _r['model']
        recon_single = model_2.reconstruct(test_single)
        test_std_local = model_2._standardize(test_single)
        recon_last = recon_single[:t_plot, -1, 0]
        true_vals = test_std_local[TIME_WINDOW - 1:TIME_WINDOW - 1 + t_plot, 0]
        _ax.plot(true_vals, 'k-', lw=1, label='True', alpha=0.8)
        _ax.plot(recon_last, '--', lw=1, label='Reconstructed (last step)', alpha=0.8)
        _ax.set_ylabel('x (std)')
        _ax.set_title(_name, fontsize=10)
        _ax.legend(loc='upper right', fontsize=8)
    _axes[-1].set_xlabel('Time step')
    _fig.suptitle('Window Reconstruction on Test Trajectory (last step of each window)', y=1.01)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Autoregressive Trajectory Prediction
    """)
    return


@app.cell
def _(TIME_WINDOW, plt, results, test_data):
    N_ROLLOUT = 300
    test_traj = test_data[0]
    _fig, _axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)
    for _ax, (_name, _r) in zip(_axes, results.items()):
        model_3 = _r['model']
        seed = test_traj[:TIME_WINDOW + 50]
        pred_traj = model_3.predict_trajectory(seed, n_steps=N_ROLLOUT)
        true_traj = test_traj[TIME_WINDOW + 50:TIME_WINDOW + 50 + N_ROLLOUT, 0]
        n_compare = min(len(true_traj), len(pred_traj))
        _ax.plot(true_traj[:n_compare], 'k-', lw=1, label='True', alpha=0.8)
        _ax.plot(pred_traj[:n_compare, 0], '--', lw=1, label='Autoregressive', alpha=0.8)
        _ax.set_ylabel('x')
        _ax.set_title(_name, fontsize=10)
        _ax.legend(loc='upper right', fontsize=8)
    _axes[-1].set_xlabel('Time step')
    _fig.suptitle(f'Autoregressive Rollout ({N_ROLLOUT} steps)', y=1.01)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. FNN Regularization Sweep (Best Architecture)

    Sweep the FNN regularization strength on the best-performing architecture
    to find the optimal trade-off between prediction accuracy and latent
    structure.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    FNN,
    LEARNING_RATE,
    N_FEATURES,
    N_LATENT,
    SEED,
    TIME_WINDOW,
    TRAIN_STEPS,
    best_name,
    compute_variances,
    model_configs,
    np,
    sliding_windows,
    test_data,
    train_data,
):
    best_cfg = model_configs[best_name]
    lambda_values = [0.001]
    sweep_results = {'lambda': [], 'mse': [], 'norm_var': []}
    for _lam in lambda_values:
        print(f'  lambda={_lam} ...', end=' ')
        reg = FNN(_lam) if _lam > 0 else None
        model_4 = best_cfg['class'](n_latent=N_LATENT, time_window=TIME_WINDOW, n_features=N_FEATURES, random_state=SEED, latent_regularizer=reg, **best_cfg['kwargs'])
        model_4.fit(train_data, train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, verbose=1, optimizer='adamw')
        _recon = model_4.reconstruct(test_data)
        _test_std = model_4._standardize(test_data)
        _x_true = sliding_windows(_test_std, TIME_WINDOW)
        _mse = float(np.mean((_recon - _x_true) ** 2))
        _latent = model_4.transform(test_data)
        _latent_flat = _latent.reshape(-1, _latent.shape[-1])
        _norm_var = compute_variances(_latent_flat, normalize=True)
        sweep_results['lambda'].append(_lam)
        sweep_results['mse'].append(_mse)
        sweep_results['norm_var'].append(_norm_var)
        print(f'Recon MSE={_mse:.6f}')
    print('Sweep complete.')
    return model_4, sweep_results


@app.cell
def _(N_LATENT, best_name, plt, sweep_results, var_true):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    x_labels = [str(v) for v in sweep_results['lambda']]
    # MSE vs lambda
    _axes[0].plot(x_labels, sweep_results['mse'], 'o-', color='tab:blue')
    _axes[0].set_xlabel('FNN $\\lambda$')
    _axes[0].set_ylabel('Test MSE')
    _axes[0].set_title(f'FNN Sweep: {best_name}')
    _axes[0].tick_params(axis='x', rotation=45)
    for _lam, nv in zip(sweep_results['lambda'], sweep_results['norm_var']):
        _axes[1].plot(range(1, len(nv) + 1), nv, 'o-', label=f'$\\lambda$={_lam}', alpha=0.8)
    # Variance spectrum vs lambda
    _axes[1].plot(range(1, N_LATENT + 1), var_true, '-', color='k', marker='s', markersize=7, label='True 3D Lorenz (xyz)', alpha=0.8)
    _axes[1].set_xlabel('Latent Dimension')
    # plot the true variance spectrum
    _axes[1].set_ylabel('Normalized Variance')
    _axes[1].set_title('Latent Variance Spectrum')
    _axes[1].set_yscale('log')
    _axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    # import loss amplification
    from JacobianODE.fnn import loss_amplification

    return (loss_amplification,)


@app.cell
def _(full_data_noisy, loss_amplification, n_train, torch, train_data):
    loss_amplification(torch.from_numpy(full_data_noisy[:n_train]), torch.from_numpy(train_data), n_neighbors=2, max_T=5, normalize=True)
    return


@app.cell
def _(loss_amplification, torch, train_data):
    loss_amplification(torch.from_numpy(train_data), torch.from_numpy(train_data), n_neighbors=2, max_T=5, normalize=True)
    return


@app.cell
def _(loss_amplification, model_4, torch, train_data):
    latent_train = model_4.transform(train_data)
    # Flatten per-timestep: (n_windows, T, D') -> (n_windows*T, D')
    latent_train_flat = latent_train.reshape(-1, latent_train.shape[-1])
    loss_amplification(torch.from_numpy(latent_train_flat), torch.from_numpy(train_data), n_neighbors=2, max_T=5, normalize=True)
    return


@app.cell
def _(loss_amplification, model_4, test_data, torch):
    latent_test = model_4.transform(test_data)
    latent_test_flat = latent_test.reshape(-1, latent_test.shape[-1])
    loss_amplification(torch.from_numpy(latent_test_flat), torch.from_numpy(test_data), n_neighbors=2, max_T=5, normalize=True)
    return


@app.cell
def _(PCA, model_4, plt, train_data):
    laten_train = model_4.transform(train_data)
    laten_train_flat = laten_train.reshape(-1, laten_train.shape[-1])
    _pca = PCA(n_components=2)
    laten_train_2d = _pca.fit_transform(laten_train_flat)
    plt.scatter(laten_train_2d[:, 0], laten_train_2d[:, 1], s=0.5, alpha=0.3, c='tab:blue')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Partial Observations with D=2

    Repeat the comparison observing both $x$ and $y$ to see how extra
    information affects each architecture.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    FNN,
    FNN_STRENGTH,
    LEARNING_RATE,
    N_LATENT,
    OrderedDict,
    SEED,
    TIME_WINDOW,
    TRAIN_STEPS,
    full_data_noisy,
    model_configs,
    n_train,
    np,
    results,
    sliding_windows,
):
    partial_data_2d = full_data_noisy[:, :, :2]
    train_data_2d = partial_data_2d[:n_train]
    test_data_2d = partial_data_2d[n_train:]
    results_2d = OrderedDict()
    for _name, _cfg in model_configs.items():
        print(f'Training: {_name} (D=2) ...', end=' ')
        model_5 = _cfg['class'](n_latent=N_LATENT, time_window=TIME_WINDOW, n_features=2, random_state=SEED, latent_regularizer=FNN(FNN_STRENGTH), **_cfg['kwargs'])
        model_5.fit(train_data_2d, train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, verbose=0, optimizer='adamw')
        _recon = model_5.reconstruct(test_data_2d)
        _test_std = model_5._standardize(test_data_2d)
        _x_true = sliding_windows(_test_std, TIME_WINDOW)
        _mse = float(np.mean((_recon - _x_true) ** 2))
        results_2d[_name] = {'mse': _mse, 'n_params': model_5.count_parameters()}
        print(f'Recon MSE={_mse:.6f}')
    print()
    print(f'{'Model':25s} {'MSE (D=1)':>12s} {'MSE (D=2)':>12s}')
    print('-' * 52)
    for _name in results.keys():
        print(f'{_name:25s} {results[_name]['mse']:12.6f} {results_2d[_name]['mse']:12.6f}')
    return (results_2d,)


@app.cell
def _(np, plt, results, results_2d):
    # Side-by-side bar chart: D=1 vs D=2
    _names = list(results.keys())
    mse_1d = [results[n]['mse'] for n in _names]
    mse_2d = [results_2d[n]['mse'] for n in _names]
    x = np.arange(len(_names))
    width = 0.35
    _fig, _ax = plt.subplots(figsize=(12, 5))
    _ax.bar(x - width / 2, mse_1d, width, label='D=1 (x only)', color='#3498db')
    _ax.bar(x + width / 2, mse_2d, width, label='D=2 (x, y)', color='#e74c3c')
    _ax.set_xticks(x)
    _ax.set_xticklabels(_names, rotation=30, ha='right')
    _ax.set_ylabel('Test MSE')
    _ax.set_title('Architecture Comparison: D=1 vs D=2 Partial Observations')
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Summary
    """)
    return


@app.cell
def _(
    FNN_STRENGTH,
    LEARNING_RATE,
    N_LATENT,
    TIME_WINDOW,
    TRAIN_STEPS,
    results,
    results_2d,
):
    print('=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)
    print(f'\nTask: Window reconstruction of Lorenz system from partial observations')
    print(f'Latent dim: {N_LATENT}, Window: {TIME_WINDOW}, FNN lambda: {FNN_STRENGTH}')
    print(f'Training: {TRAIN_STEPS} epochs, lr={LEARNING_RATE}\n')
    print(f'{'Model':25s} {'MSE (D=1)':>12s} {'MSE (D=2)':>12s} {'Params':>10s} {'Time (s)':>10s}')
    print('-' * 72)
    for _name in results.keys():
        print(f'{_name:25s} {results[_name]['mse']:12.6f} {results_2d[_name]['mse']:12.6f} {results[_name]['n_params']:10,} {results[_name]['train_time']:10.1f}')
    best_1d = min(results, key=lambda k: results[k]['mse'])
    best_2d = min(results_2d, key=lambda k: results_2d[k]['mse'])
    print(f'\nBest (D=1): {best_1d}  (Recon MSE={results[best_1d]['mse']:.6f})')
    print(f'Best (D=2): {best_2d}  (Recon MSE={results_2d[best_2d]['mse']:.6f})')
    return


if __name__ == "__main__":
    app.run()
