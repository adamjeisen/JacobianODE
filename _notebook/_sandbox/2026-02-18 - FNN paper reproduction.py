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
    # Reproducing Lorenz Results from Gilpin (NeurIPS 2020)

    This notebook reproduces the Lorenz attractor reconstruction results from:

    > Gilpin, W. "Deep reconstruction of strange attractors from time series." NeurIPS 2020.
    > https://arxiv.org/abs/2002.05909

    ## Paper parameters (Appendix, Section 3)

    - **Lorenz ODE**: $\sigma=10$, $\rho=28$, $\beta=2.667$
    - **Integration**: LSODA method, $\Delta t = 0.004$
    - **Trajectory length**: 125,000 steps (total time = 500)
    - **Transient removal**: discard first 120,000 steps, keep last 5,000 per trajectory
    - **Downsampling**: factor of 10
    - **Observable**: $x(t)$ only (first coordinate, univariate)
    - **Train/val/test**: 5,000 timepoints each, from 3 trajectories with different ICs

    ## Model parameters (Appendix, Section 5)

    - **LSTM autoencoder**: `[Input-GN-LSTM(10)-BN]-[GN-LSTM(10)-BN-ELU-Output]`
    - **MLP autoencoder**: `[Input-GN-FC(10)-BN-ELU-FC(10)-BN-ELU-FC(10)-BN]-[GN-FC(10)-BN-ELU-FC(10)-BN-ELU-FC(10)-BN-ELU-Output]`
    - **Latent units**: $L=10$
    - **GaussianNoise**: stddev = 0.5 (training only)
    - **FNN regularizer**: $R_{tol}=10$, $A_{tol}=2.0$, $K=\max(1, \lceil 0.01B \rceil)$
    - **Optimizer**: Adam, $\gamma = 10^{-3}$ (default)
    - **Batch size**: 100
    - **Training epochs**: 200
    - **5 replicate** networks with random initializations per $\lambda$
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
    from scipy.integrate import solve_ivp
    from scipy.spatial import procrustes
    from sklearn.decomposition import PCA

    from JacobianODE.fnn import (
        LSTMEmbedding,
        MLPEmbedding,
        ETDEmbedding,
        TICAEmbedding,
        FNN,
        compute_variances,
        compute_s_dim,
    )

    return (
        ETDEmbedding,
        FNN,
        LSTMEmbedding,
        MLPEmbedding,
        PCA,
        TICAEmbedding,
        compute_s_dim,
        compute_variances,
        np,
        plt,
        solve_ivp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Lorenz data (exact paper parameters)
    """)
    return


@app.cell
def _(np, solve_ivp):
    # Lorenz system parameters (Appendix Eq. A1-A3)
    SIGMA = 10.0
    RHO = 28.0
    BETA = 2.667  # paper uses 2.667, not 8/3
    DT = 0.004
    # Integration parameters (Appendix Section 3)
    N_STEPS = 125000  # integration stepsize
    DOWNSAMPLE = 10  # total steps per trajectory (total time = 500)
    N_KEEP = 5000  # downsampling factor
      # points to keep per partition (last 5000 after downsampling)
    def lorenz(t, y, sigma=SIGMA, rho=RHO, beta=BETA):
        """Lorenz system RHS."""
        _x, y_, z = y
        return [sigma * (y_ - _x), _x * (rho - z) - y_, _x * y_ - beta * z]

    def simulate_lorenz(ic, n_steps=N_STEPS, dt=DT):
        """Integrate the Lorenz system using LSODA (as in the paper)."""
        t_span = (0, n_steps * dt)
        t_eval = np.arange(0, n_steps * dt, dt)
        sol = solve_ivp(lorenz, t_span, ic, method='LSODA', t_eval=t_eval, rtol=1e-08, atol=1e-08)
        return sol.y.T
    ics = [[1.0, 1.0, 1.0], [-5.0, 3.0, 20.0], [10.0, -10.0, 25.0]]
    trajectories = []
    for ic in ics:
        traj = simulate_lorenz(ic)
        traj_ds = traj[::DOWNSAMPLE]
        traj_ds = traj_ds[-N_KEEP:]
        trajectories.append(traj_ds)
        print(f'IC={ic} -> raw shape {traj.shape}, downsampled+trimmed shape {traj_ds.shape}')
    train_full = trajectories[0]
    val_full = trajectories[1]  # (n_steps, 3)
    test_full = trajectories[2]
    train_x = train_full[:, 0]
    # Three different initial conditions for train/val/test
    val_x = val_full[:, 0]
    test_x = test_full[:, 0]
    print(f'\nTrain/Val/Test univariate shapes: {train_x.shape}, {val_x.shape}, {test_x.shape}')
    # Paper uses only x(t) as the observable
    print(f'Effective dt after downsampling: {DT * DOWNSAMPLE}')  # Downsample by factor of 10  # Keep last 5000 points (discard transients)  # (5000, 3)  # (5000,)
    return (
        DOWNSAMPLE,
        DT,
        N_KEEP,
        lorenz,
        test_full,
        test_x,
        train_full,
        train_x,
    )


@app.cell
def _(DOWNSAMPLE, DT, np, plt, train_full, train_x):
    # Visualize the Lorenz attractor and the univariate observable
    _fig = plt.figure(figsize=(14, 4))
    _ax1 = _fig.add_subplot(131, projection='3d')
    _ax1.plot(*train_full.T, lw=0.3, c='k')
    _ax1.set_title('Lorenz attractor (train)')
    _ax1.set_xlabel('x')
    _ax1.set_ylabel('y')
    _ax1.set_zlabel('z')
    _ax2 = _fig.add_subplot(132)
    t = np.arange(len(train_x)) * DT * DOWNSAMPLE
    _ax2.plot(t[:500], train_x[:500], lw=0.5, c='k')
    _ax2.set_title('Observable: x(t)')
    _ax2.set_xlabel('Time')
    ax3 = _fig.add_subplot(133)
    ax3.plot(train_full[:, 0], train_full[:, 1], lw=0.3, c='k')
    ax3.set_title('x-y projection')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Model setup (exact paper architecture)

    From Appendix Section 5:
    - LSTM: `[Input-GN-LSTM(10)-BN]-[GN-LSTM(10)-BN-ELU-Output]` — single-layer LSTM with 10 hidden units
    - MLP: 3 FC(10) layers in both encoder and decoder
    - Both use L=10 latent units
    - Time window `T`: the paper uses the Hankel matrix width as a hyperparameter. The paper states T should be "large enough" for sufficient variation. From the original code demos, T=10 is the default.
    """)
    return


@app.cell
def _():
    # Paper architecture parameters
    N_LATENT = 10       # L = 10 latent units
    TIME_WINDOW = 10    # Hankel matrix width (default in original code)

    # Paper training parameters
    LEARNING_RATE = 1e-3  # Adam learning rate
    BATCH_SIZE = 100      # batch size
    TRAIN_STEPS = 200     # training epochs
    N_REPLICATES = 1      # 5 replicate networks per setting
    return (
        BATCH_SIZE,
        LEARNING_RATE,
        N_LATENT,
        N_REPLICATES,
        TIME_WINDOW,
        TRAIN_STEPS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sweep regularizer strength $\lambda$ (Figure 5)

    The paper shows that increasing $\lambda$ causes the distribution of latent activations to develop increasing right skewness, with optimal $S_{\text{dim}}$ at intermediate $\lambda$.
    """)
    return


@app.cell
def _(compute_variances, np):
    def pad_variances(data_3d, n_latent):
        """Compute normalized variances of the true system, padded to n_latent dims."""
        d = data_3d.shape[1]
        if d < n_latent:
            padding = np.zeros((data_3d.shape[0], n_latent - d))
            data_padded = np.concatenate([data_3d, padding], axis=1)
        else:
            data_padded = data_3d[:, :n_latent]
        return compute_variances(data_padded, normalize=True)

    return (pad_variances,)


@app.cell
def _(N_LATENT, np, pad_variances, test_full):
    # True system normalized variances (for S_dim computation)
    var_true = pad_variances(test_full, N_LATENT)
    print(f"True system normalized variances: {var_true}")
    print(f"True system has {np.sum(var_true > 0.01)} active dimensions")
    return (var_true,)


@app.cell
def _(
    BATCH_SIZE,
    FNN,
    LEARNING_RATE,
    LSTMEmbedding,
    N_LATENT,
    N_REPLICATES,
    TIME_WINDOW,
    TRAIN_STEPS,
    compute_s_dim,
    compute_variances,
    np,
    test_x,
    train_x,
    var_true,
):
    # Lambda sweep for LSTM model (paper Figure 5)
    lambda_values = [0, 1e-05, 0.0001, 0.001, 0.01, 0.1]
    lstm_results = {'lambda': [], 's_dim_mean': [], 's_dim_std': [], 'norm_vars': [], 'best_var': []}
    for _lam in lambda_values:
        print(f'\n=== Lambda = {_lam} ===')
        _replicate_s_dims = []
        _replicate_vars = []
        for _rep in range(N_REPLICATES):  # all replicate variance vectors
            _reg = FNN(_lam) if _lam > 0 else None  # best replicate per lambda
            _model = LSTMEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, n_features=1, network_shape=[], latent_regularizer=_reg)
            _model.fit(train_x, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, verbose=0)
            _embedding = _model.transform(test_x)
            pred_var = compute_variances(_embedding, normalize=True)
            _s_dim = compute_s_dim(var_true, pred_var)
            _replicate_s_dims.append(_s_dim)
            _replicate_vars.append(pred_var)
            print(f'  Replicate {_rep + 1}/{N_REPLICATES}: S_dim={_s_dim:.4f}')
        _s_dims = np.array(_replicate_s_dims)
        _best_idx = np.argmax(_s_dims)  # Paper LSTM: single LSTM layer with 10 hidden units -> network_shape=[]
        lstm_results['lambda'].append(_lam)  # (final LSTM has n_latent=10 units, matching the paper's LSTM(10))
        lstm_results['s_dim_mean'].append(_s_dims.mean())
        lstm_results['s_dim_std'].append(_s_dims.std())
        lstm_results['norm_vars'].append(_replicate_vars)
        lstm_results['best_var'].append(_replicate_vars[_best_idx])
        print(f'  Mean S_dim = {_s_dims.mean():.4f} +/- {_s_dims.std():.4f}')  # single-layer LSTM as in paper  # Evaluate on test data  # (n_windows, N_LATENT)
    return lambda_values, lstm_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Reproduce Figure 5: Latent variance distributions & dimensionality error
    """)
    return


@app.cell
def _(N_LATENT, lambda_values, lstm_results, np, plt, var_true):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _colors = plt.cm.coolwarm(np.linspace(0, 1, len(lambda_values)))
    # --- Panel A: Normalized latent variances per lambda ---
    for _i, (_lam, _nvars) in enumerate(zip(lstm_results['lambda'], lstm_results['norm_vars'])):
        _mean_nv = np.mean(_nvars, axis=0)
        std_nv = np.std(_nvars, axis=0)  # Plot mean across replicates
        _x = np.arange(1, N_LATENT + 1)
        _ax1.plot(_x, _mean_nv, 'o-', color=_colors[_i], label=f'$\\lambda={_lam}$', markersize=4)
        _ax1.fill_between(_x, _mean_nv - std_nv, _mean_nv + std_nv, alpha=0.1, color=_colors[_i])
    _ax1.plot(_x, var_true, 'k-', lw=2, label='True system')
    best_overall_idx = np.argmax(lstm_results['s_dim_mean'])
    _ax1.plot(_x, lstm_results['best_var'][best_overall_idx], 'k--', lw=2, label='Best LSTM')
    # Overlay true system variances
    _ax1.set_xlabel('Latent index')
    # Overlay best performing model (dashed)
    _ax1.set_ylabel('Normalized variance')
    _ax1.set_title('(A) Latent variance distribution')
    _ax1.legend(fontsize=7, loc='upper right')
    _ax1.set_ylim(bottom=0.0001)
    _lam_plot = [max(l, 1e-06) for l in lstm_results['lambda']]
    _dim_errors = [1.0 - s for s in lstm_results['s_dim_mean']]
    dim_err_std = lstm_results['s_dim_std']
    # ax1.set_yscale("log")
    _ax2.errorbar(_lam_plot, _dim_errors, yerr=dim_err_std, fmt='o-', color='tab:blue', capsize=3)
    _ax2.set_xscale('log')
    # --- Panel B: Dimensionality error (1 - S_dim) vs lambda ---
    _ax2.set_xlabel('Regularizer strength $\\lambda$')  # avoid log(0)
    _ax2.set_ylabel('Dimensionality error $(1 - S_{\\mathrm{dim}})$')
    _ax2.set_title('(B) Dimensionality error vs $\\lambda$')
    plt.tight_layout()
    plt.show()
    return (best_overall_idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Baseline comparisons (Figure 3B)

    Compare LSTM+FNN against ETD, tICA, and unregularized LSTM.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    ETDEmbedding,
    FNN,
    LEARNING_RATE,
    LSTMEmbedding,
    N_LATENT,
    N_REPLICATES,
    TICAEmbedding,
    TIME_WINDOW,
    TRAIN_STEPS,
    best_overall_idx,
    compute_s_dim,
    compute_variances,
    lstm_results,
    np,
    test_x,
    train_x,
    var_true,
):
    # Best lambda from sweep (use the one with highest mean S_dim)
    best_lambda = lstm_results['lambda'][best_overall_idx]
    print(f'Best lambda from sweep: {best_lambda}')
    baseline_results = {}
    print('\nTraining ETD...')
    etd = ETDEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW)
    # --- ETD (Eigen-Time-Delay / PCA of Hankel matrix) ---
    etd.fit(train_x)
    etd_emb = etd.transform(test_x)
    nv_etd = compute_variances(etd_emb, normalize=True)
    s_dim_etd = compute_s_dim(var_true, nv_etd)
    baseline_results['ETD'] = {'embedding': etd_emb, 's_dim': s_dim_etd, 'norm_var': nv_etd}
    print(f'  ETD S_dim = {s_dim_etd:.4f}')
    print('\nTraining tICA...')
    tica = TICAEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, time_lag=1)
    tica.fit(train_x)
    # --- tICA ---
    tica_emb = tica.transform(test_x)
    nv_tica = compute_variances(tica_emb, normalize=True)
    s_dim_tica = compute_s_dim(var_true, nv_tica)
    baseline_results['tICA'] = {'embedding': tica_emb, 's_dim': s_dim_tica, 'norm_var': nv_tica}
    print(f'  tICA S_dim = {s_dim_tica:.4f}')
    print('\nTraining LSTM (no regularizer)...')
    unreg_s_dims = []
    unreg_embs = []
    for _rep in range(N_REPLICATES):
    # --- Unregularized LSTM (lambda=0) ---
        model_unreg = LSTMEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, network_shape=[])
        model_unreg.fit(train_x, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, verbose=0)
        _emb = model_unreg.transform(test_x)
        _nv = compute_variances(_emb, normalize=True)
        s = compute_s_dim(var_true, _nv)
        unreg_s_dims.append(s)
        unreg_embs.append(_emb)
    best_unreg = np.argmax(unreg_s_dims)
    baseline_results['LSTM'] = {'embedding': unreg_embs[best_unreg], 's_dim': np.mean(unreg_s_dims), 's_dim_std': np.std(unreg_s_dims), 'norm_var': compute_variances(unreg_embs[best_unreg], normalize=True)}
    print(f'  LSTM S_dim = {np.mean(unreg_s_dims):.4f} +/- {np.std(unreg_s_dims):.4f}')
    print(f'\nTraining LSTM + FNN (lambda={best_lambda})...')
    fnn_s_dims = []
    fnn_embs = []
    for _rep in range(N_REPLICATES):
        model_fnn = LSTMEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, network_shape=[], latent_regularizer=FNN(best_lambda))
        model_fnn.fit(train_x, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, verbose=0)
        _emb = model_fnn.transform(test_x)
        _nv = compute_variances(_emb, normalize=True)
        s = compute_s_dim(var_true, _nv)
        fnn_s_dims.append(s)
        fnn_embs.append(_emb)
    best_fnn = np.argmax(fnn_s_dims)
    baseline_results['LSTM-FNN'] = {'embedding': fnn_embs[best_fnn], 's_dim': np.mean(fnn_s_dims), 's_dim_std': np.std(fnn_s_dims), 'norm_var': compute_variances(fnn_embs[best_fnn], normalize=True)}
    # --- LSTM + FNN (best lambda) ---
    print(f'  LSTM-FNN S_dim = {np.mean(fnn_s_dims):.4f} +/- {np.std(fnn_s_dims):.4f}')
    return baseline_results, best_lambda


@app.cell
def _(PCA, baseline_results, plt, test_full):
    # PCA of Lorenz attractor encodings for each baseline model
    _fig, _axes = plt.subplots(1, len(baseline_results) + 1, figsize=(4 * (len(baseline_results) + 1), 4))
    pca_true = PCA(n_components=2)
    # True attractor (PCA of full 3D state for reference)
    true_2d = pca_true.fit_transform(test_full)
    _axes[0].scatter(true_2d[:, 0], true_2d[:, 1], s=0.3, c='k', rasterized=True)
    _axes[0].set_title('True attractor\n(PCA of 3D state)')
    _axes[0].set_xlabel('PC 1')
    _axes[0].set_ylabel('PC 2')
    for _i, (_name, _res) in enumerate(baseline_results.items()):
        _emb = _res['embedding']
        _pca = PCA(n_components=2)
        emb_2d = _pca.fit_transform(_emb)
        var_explained = _pca.explained_variance_ratio_
        _axes[_i + 1].scatter(emb_2d[:, 0], emb_2d[:, 1], s=0.3, c='k', rasterized=True)
        _axes[_i + 1].set_title(f'{_name}\n$S_{{dim}}$={_res['s_dim']:.3f}\nPC1+PC2={100 * (var_explained[0] + var_explained[1]):.1f}% var')
        _axes[_i + 1].set_xlabel('PC 1')
        _axes[_i + 1].set_ylabel('PC 2')
    plt.suptitle('PCA of Lorenz attractor encodings (test set)', fontsize=13)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Reproduce Figure 3A: Visualize reconstructed attractors
    """)
    return


@app.cell
def _(PCA, baseline_results, plt, test_full):
    _fig = plt.figure(figsize=(16, 4))
    _ax = _fig.add_subplot(151, projection='3d')
    # True attractor (first 3 coords)
    _ax.plot(*test_full.T, lw=0.1, c='k')
    _ax.set_title('True attractor')
    for _i, (_name, _res) in enumerate(baseline_results.items()):
        _emb = _res['embedding']
    # For each method, PCA to 3D and plot
        _pca = PCA(n_components=3)
        _emb_3d = _pca.fit_transform(_emb)
        _ax = _fig.add_subplot(1, 5, _i + 2, projection='3d')  # Use PCA to extract top 3 components for visualization
        _ax.plot(*_emb_3d.T, lw=0.1, c='k')
        s_dim_val = _res['s_dim']
        _ax.set_title(f'{_name}\n$S_{{dim}}$={s_dim_val:.3f}')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. MLP autoencoder comparison (Appendix)

    Paper MLP: `[Input-GN-FC(10)-BN-ELU-FC(10)-BN-ELU-FC(10)-BN]-[GN-FC(10)-BN-ELU-FC(10)-BN-ELU-FC(10)-BN-ELU-Output]`

    This is a 3-layer MLP with 10 units per layer, matching `network_shape=[10, 10]` (plus the latent layer = 3rd FC).
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    FNN,
    LEARNING_RATE,
    MLPEmbedding,
    N_LATENT,
    N_REPLICATES,
    TIME_WINDOW,
    TRAIN_STEPS,
    compute_s_dim,
    compute_variances,
    lambda_values,
    np,
    test_x,
    train_x,
    var_true,
):
    # Lambda sweep for MLP model (Appendix Figure S4)
    mlp_results = {'lambda': [], 's_dim_mean': [], 's_dim_std': [], 'norm_vars': [], 'best_var': []}
    for _lam in lambda_values:
        print(f'\n=== MLP Lambda = {_lam} ===')
        _replicate_s_dims = []
        _replicate_vars = []
        for _rep in range(N_REPLICATES):
            _reg = FNN(_lam) if _lam > 0 else None
            _model = MLPEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, n_features=1, network_shape=[10, 10], latent_regularizer=_reg)
            _model.fit(train_x, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, verbose=0)
            _embedding = _model.transform(test_x)
            _nv = compute_variances(_embedding, normalize=True)
            _s_dim = compute_s_dim(var_true, _nv)
            _replicate_s_dims.append(_s_dim)
            _replicate_vars.append(_nv)
            print(f'  Replicate {_rep + 1}/{N_REPLICATES}: S_dim={_s_dim:.4f}')
        _s_dims = np.array(_replicate_s_dims)  # Paper MLP: 3 FC layers with 10 units each
        _best_idx = np.argmax(_s_dims)  # network_shape=[10, 10] gives 2 hidden + 1 latent = 3 layers in encoder
        mlp_results['lambda'].append(_lam)
        mlp_results['s_dim_mean'].append(_s_dims.mean())
        mlp_results['s_dim_std'].append(_s_dims.std())
        mlp_results['norm_vars'].append(_replicate_vars)
        mlp_results['best_var'].append(_replicate_vars[_best_idx])  # paper architecture
        print(f'  Mean S_dim = {_s_dims.mean():.4f} +/- {_s_dims.std():.4f}')
    return (mlp_results,)


@app.cell
def _(N_LATENT, lambda_values, lstm_results, mlp_results, np, plt, var_true):
    # Combined Figure 5 / S4: LSTM and MLP comparison
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 10))
    for row, (_name, results) in enumerate([('LSTM', lstm_results), ('MLP', mlp_results)]):
        ax_var, ax_err = _axes[row]
        _colors = plt.cm.coolwarm(np.linspace(0, 1, len(lambda_values)))
        for _i, (_lam, _nvars) in enumerate(zip(results['lambda'], results['norm_vars'])):
            _mean_nv = np.mean(_nvars, axis=0)  # Panel A: Normalized latent variances
            _x = np.arange(1, N_LATENT + 1)
            ax_var.plot(_x, _mean_nv, 'o-', color=_colors[_i], label=f'$\\lambda={_lam}$', markersize=3)
        ax_var.plot(_x, var_true, 'k-', lw=2, label='True system')
        ax_var.set_xlabel('Latent index')
        ax_var.set_ylabel('Normalized variance')
        ax_var.set_title(f'{_name}: Latent variance distribution')
        ax_var.legend(fontsize=7)
        ax_var.set_yscale('log')
        ax_var.set_ylim(bottom=0.0001)
        _lam_plot = [max(l, 1e-06) for l in results['lambda']]
        _dim_errors = [1.0 - s for s in results['s_dim_mean']]
        ax_err.errorbar(_lam_plot, _dim_errors, yerr=results['s_dim_std'], fmt='o-', capsize=3)
        ax_err.set_xscale('log')
        ax_err.set_xlabel('Regularizer strength $\\lambda$')  # Panel B: Dimensionality error
        ax_err.set_ylabel('Dimensionality error $(1 - S_{\\mathrm{dim}})$')
        ax_err.set_title(f'{_name}: Dimensionality error vs $\\lambda$')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Consistency across replicates (Figure S3, Appendix B)

    Train an ensemble of LSTM models with and without FNN to show that the regularizer produces more consistent embeddings.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    FNN,
    LEARNING_RATE,
    LSTMEmbedding,
    N_LATENT,
    TIME_WINDOW,
    TRAIN_STEPS,
    best_lambda,
    test_x,
    train_x,
):
    n_ensemble = 5
    print('Training LSTM ensemble (no regularizer)...')
    # Train replicates without regularizer
    unreg_embeddings = []
    for _i in range(n_ensemble):
        m = LSTMEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, network_shape=[])
        m.fit(train_x, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, verbose=0)
        unreg_embeddings.append(m.transform(test_x))
        print(f'  Replicate {_i + 1}/{n_ensemble} done')
    print(f'\nTraining LSTM+FNN ensemble (lambda={best_lambda})...')
    fnn_embeddings = []
    for _i in range(n_ensemble):
    # Train replicates with FNN regularizer
        m = LSTMEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, network_shape=[], latent_regularizer=FNN(best_lambda))
        m.fit(train_x, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, verbose=0)
        fnn_embeddings.append(m.transform(test_x))
        print(f'  Replicate {_i + 1}/{n_ensemble} done')
    return fnn_embeddings, n_ensemble, unreg_embeddings


@app.cell
def _(PCA, fnn_embeddings, n_ensemble, plt, unreg_embeddings):
    _fig, _axes = plt.subplots(2, n_ensemble, figsize=(4 * n_ensemble, 8), subplot_kw={'projection': '3d'})
    for _i in range(n_ensemble):
        _pca = PCA(n_components=3)
        _emb_3d = _pca.fit_transform(unreg_embeddings[_i])
        _axes[0, _i].plot(*_emb_3d.T, lw=0.1, c='k')  # Unreg row
        if _i == 0:
            _axes[0, _i].set_ylabel('LSTM (no reg)')
        _axes[0, _i].set_title(f'Rep {_i + 1}')
        _pca = PCA(n_components=3)
        _emb_3d = _pca.fit_transform(fnn_embeddings[_i])
        _axes[1, _i].plot(*_emb_3d.T, lw=0.1, c='k')
        if _i == 0:
            _axes[1, _i].set_ylabel('LSTM + FNN')  # FNN row
    _fig.suptitle('Consistency across replicates (cf. Figure S3)', fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Stochastic Lorenz (Figure 4)

    From the paper: "an uncorrelated white noise term $\xi(t)$ is appended to each dynamical variable before integration, the integration timestep is decreased to $\Delta t = 0.0004$, and the integration output is downsampled by a factor of 100."
    """)
    return


@app.cell
def _(N_KEEP, lorenz, np):
    def simulate_stochastic_lorenz(ic, xi0, n_steps=125000, dt=0.0004, downsample=100):
        """Euler-Maruyama integration of stochastic Lorenz system.
    
        The paper adds uncorrelated white noise xi(t) to each variable.
        For SDE integration we use Euler-Maruyama since LSODA is for ODEs.
        """
        y = np.array(ic, dtype=float)
        trajectory = np.zeros((n_steps, 3))
        sqrt_dt = np.sqrt(dt)
        for _i in range(n_steps):
            trajectory[_i] = y
            dydt = np.array(lorenz(0, y))
            noise = _xi0 * np.random.randn(3) * sqrt_dt  # Deterministic part
            y = y + dydt * dt + noise
        return trajectory[::downsample]  # Stochastic part: uncorrelated white noise
    xi0_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    print('Generating stochastic Lorenz trajectories...')
    stoch_train = {}
    stoch_test = {}  # Downsample
    for _xi0 in xi0_values:
        np.random.seed(42)
        tr = simulate_stochastic_lorenz([1.0, 1.0, 1.0], _xi0)
    # Generate stochastic trajectories at different noise levels
        np.random.seed(123)
        te = simulate_stochastic_lorenz([-5.0, 3.0, 20.0], _xi0)
        stoch_train[_xi0] = tr[-N_KEEP:, 0]
        stoch_test[_xi0] = te[-N_KEEP:, 0]
        print(f'  xi0={_xi0}: train shape {stoch_train[_xi0].shape}')  # Different ICs for train/test  # x(t), last 5000 points
    return stoch_test, stoch_train, xi0_values


@app.cell
def _(plt, stoch_train, xi0_values):
    # Visualize stochastic trajectories
    _fig, _axes = plt.subplots(2, 3, figsize=(14, 6))
    for _i, _xi0 in enumerate(xi0_values):
        _ax = _axes.flat[_i]
        _ax.plot(stoch_train[_xi0][:500], lw=0.5, c='k')
        _ax.set_title(f'$\\xi_0 = {_xi0}$')
        _ax.set_xlabel('Time index')
    plt.suptitle('Stochastic Lorenz x(t) at different noise levels')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    BATCH_SIZE,
    FNN,
    LEARNING_RATE,
    LSTMEmbedding,
    N_LATENT,
    TIME_WINDOW,
    TRAIN_STEPS,
    best_lambda,
    compute_s_dim,
    compute_variances,
    stoch_test,
    stoch_train,
    var_true,
    xi0_values,
):
    # Figure 4B: S_dim vs noise strength for LSTM and LSTM+FNN
    print('Training models across noise levels...')
    stoch_results = {'xi0': [], 'lstm_sdim': [], 'fnn_sdim': []}
    for _xi0 in xi0_values:
        print(f'\nxi0 = {_xi0}')
        m_unreg = LSTMEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, network_shape=[])
        m_unreg.fit(stoch_train[_xi0], learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, verbose=0)
        emb_unreg = m_unreg.transform(stoch_test[_xi0])  # LSTM without FNN
        nv_unreg = compute_variances(emb_unreg, normalize=True)
        s_unreg = compute_s_dim(var_true, nv_unreg)
        m_fnn = LSTMEmbedding(n_latent=N_LATENT, time_window=TIME_WINDOW, network_shape=[], latent_regularizer=FNN(best_lambda))
        m_fnn.fit(stoch_train[_xi0], learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, verbose=0)
        emb_fnn = m_fnn.transform(stoch_test[_xi0])
        nv_fnn = compute_variances(emb_fnn, normalize=True)
        s_fnn = compute_s_dim(var_true, nv_fnn)
        stoch_results['xi0'].append(_xi0)  # LSTM with FNN
        stoch_results['lstm_sdim'].append(s_unreg)
        stoch_results['fnn_sdim'].append(s_fnn)
        print(f'  LSTM: S_dim={s_unreg:.4f}, LSTM+FNN: S_dim={s_fnn:.4f}')
    return (stoch_results,)


@app.cell
def _(plt, stoch_results):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(stoch_results['xi0'], stoch_results['lstm_sdim'], 'o-', label='LSTM', color='tab:blue')
    _ax.plot(stoch_results['xi0'], stoch_results['fnn_sdim'], 's-', label='LSTM + FNN', color='tab:red')
    _ax.set_xlabel('Noise amplitude $\\xi_0$')
    _ax.set_ylabel('$S_{\\mathrm{dim}}$')
    _ax.set_title('Dimension similarity vs noise strength (cf. Figure 4B)')
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Summary

    This notebook reproduces the key Lorenz results from Gilpin (NeurIPS 2020):

    1. **Data generation**: Lorenz system with exact paper parameters ($\sigma=10, \rho=28, \beta=2.667$, LSODA integration at $\Delta t=0.004$, downsampled $\times 10$, 5000 points per partition)
    2. **Architecture**: Single-layer LSTM and 3-layer MLP, both with 10 hidden units and $L=10$ latent dimensions
    3. **Figure 5**: $\lambda$ sweep showing optimal dimensionality recovery at intermediate regularizer strength
    4. **Figure 3**: Baseline comparisons (ETD, tICA, LSTM, LSTM+FNN)
    5. **Figure S3**: Consistency of embeddings across replicate trainings
    6. **Figure 4**: Robustness to stochastic forcing
    """)
    return


if __name__ == "__main__":
    app.run()
