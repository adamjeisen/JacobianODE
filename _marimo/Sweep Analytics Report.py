import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from JacobianODE.jacobians.run_analytics import run_analytics

    return (run_analytics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analytics Report
    """)
    return


@app.cell
def _():
    WANDB_ENTITY = "JacobianODE"

    # WANDB_PROJECT = "Lorenz_IND0_N100_D1_NormTrue_T3__spline_coupling__JacobianODE"
    # # WANDB_GROUP = "sweep_from_scratch_spline_coupling_lc_9vals__klNNone_1.0__klD0_0.0001_0.001_0.01_0.1_1__te0.0_enc_warmup_5_vaetrue" # 108 runs
    # WANDB_GROUP = "sweep_from_scratch_spline_coupling_lc_9vals__klNnull__klD0_0.0001_0.001_0.01_0.1_1__te0.0_enc_warmup_5_vaetrue" # 54 runs (lc v kl_dyn_weight, pred_steps = 10)

    # WANDB_PROJECT = "Lorenz_IND0_N100_D1_NormTrue_T7__spline_coupling__JacobianODE"
    # WANDB_GROUP = None

    # WANDB_PROJECT = "WMTask_INDall_N1_D1_NormTrue_T17__spline_coupling__JacobianODE"
    # WANDB_GROUP = None

    # WANDB_PROJECT = "WMTask_IND1926273140555669819498104_N10_D1_NormTrue_T10__spline_coupling__JacobianODE"
    # WANDB_GROUP = None

    WANDB_PROJECT = "WMTask_INDall_N1_D1_NormTrue_T128__JacobianODE"
    # WANDB_GROUP = None
    # WANDB_GROUP = "spline_coupling__sweep_lc_x_kl_dyn_vae_sample_all_losses"
    WANDB_GROUP = "mlp_diffeo__sweep_lc_x_kl_dyn_vae_sample_all_losses"

    SAVE_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs"
    # TRUE_LYAPUNOV = [0.91, 0.0, -14.57]  # None for wmtask (overridden from config)a\
    TRUE_LYAPUNOV = None
    return SAVE_DIR, TRUE_LYAPUNOV, WANDB_ENTITY, WANDB_GROUP, WANDB_PROJECT


@app.cell
def _(
    SAVE_DIR,
    TRUE_LYAPUNOV,
    WANDB_ENTITY,
    WANDB_GROUP,
    WANDB_PROJECT,
    run_analytics,
):
    # Implementation is typed as dict | tuple | None; marimo/pyright need a runtime
    # narrow before unpacking when return_model=True (always a 3-tuple then).
    _analytics_out = run_analytics(
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        save_dir=SAVE_DIR,
        wandb_group=WANDB_GROUP,
        true_lyapunov=TRUE_LYAPUNOV,
        lyapunov_burn_in_steps=1000,
        lyapunov_burn_in_drop=200,
        output=['html'],
        output_dir='reports',
        return_model=True,
    )
    assert isinstance(_analytics_out, tuple) and len(_analytics_out) == 3
    _result, lit_model, run_id = _analytics_out
    return lit_model, run_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tangent Space Stuff
    """)
    return


@app.cell
def _(SAVE_DIR, TRUE_LYAPUNOV, WANDB_ENTITY, WANDB_PROJECT, lit_model, run_id):
    import torch
    from JacobianODE.jacobians import load_run, create_dataloaders, load_checkpoint
    wandb_project_path = f'{WANDB_ENTITY}/{WANDB_PROJECT}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run, cfg, eq, dt, values, _, _, _, _, _ = load_run(wandb_project_path, run_id=run_id, save_dir=SAVE_DIR, generate_data=True, verbose=True)
    train_dl, val_dl, test_dl, trajs = create_dataloaders(cfg, values, verbose=True, return_full_obs=True)
    # Load run config, equation, data, and model scaffold
    load_checkpoint(run, cfg, lit_model, save_dir=SAVE_DIR, verbose=True)
    if TRUE_LYAPUNOV is not None:
        lit_model.true_lyapunov_exponents = torch.tensor(TRUE_LYAPUNOV, dtype=torch.float32)
    lit_model_2 = lit_model.to(device)
    lit_model_2.eval()
    print(f'\nrun_id: {run_id}')
    print(f'device: {device}')
    print(f'cfg.model: {(cfg.model._target_ if hasattr(cfg.model, '_target_') else type(cfg.model))}')
    # Create dataloaders (with full obs for analysis)
    print(f'data shape: {values.shape}')
    print(f'n_latent: {lit_model_2.encoder.n_latent}')
    # Load best checkpoint into the model
    print(f'train/val/test batches: {len(train_dl)}/{len(val_dl)}/{len(test_dl)}')
    return device, dt, lit_model_2, torch, trajs


@app.cell
def _(device, dt, lit_model_2, torch, trajs):
    import numpy as np

    def compute_lyapunov_vectors(jacs: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Gram-Schmidt orthonormal Lyapunov vectors via QR decomposition.

        Performs the same forward QR iteration as the Lyapunov exponent computation,
        but returns the orthonormal Q matrices at every timestep.

        Parameters
        ----------
        jacs : torch.Tensor
            Jacobian matrices along trajectories, shape ``(B, T, D, D)`` or ``(T, D, D)``.
        dt : float
            Time step between successive Jacobians.

        Returns
        -------
        Q_all : torch.Tensor
            Orthonormal Lyapunov vectors at each timestep, shape ``(B, T, D, D)``
            (or ``(T, D, D)`` if input was unbatched).
            ``Q_all[..., t, :, i]`` is the i-th Lyapunov vector at time t.
        exponents : torch.Tensor
            Running-average Lyapunov exponents at each timestep, shape ``(B, T, D)``
            (or ``(T, D)``). The final slice ``exponents[..., -1, :]`` gives the
            converged exponents (equivalent to ``compute_lyapunov_exponents``).
        """
        unbatched = jacs.ndim == 3
        if unbatched:
            jacs = jacs.unsqueeze(0)
        B, T, D, _ = jacs.shape
        Q = torch.eye(D, dtype=jacs.dtype, device=jacs.device).expand(B, -1, -1).clone()
        log_diag_sum = torch.zeros(B, D, dtype=jacs.dtype, device=jacs.device)
        Q_all = torch.empty(B, T, D, D, dtype=jacs.dtype, device=jacs.device)
        exponents_all = torch.empty(B, T, D, dtype=jacs.dtype, device=jacs.device)
        for _t in range(T):
            M = torch.linalg.matrix_exp(jacs[:, _t] * dt)
            Z = M @ Q
            Q, R = torch.linalg.qr(Z)
            diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
            signs = torch.sign(diag_R)
            signs[signs == 0] = 1.0
            Q = Q * signs.unsqueeze(-2)
            log_diag_sum = log_diag_sum + torch.log(torch.abs(diag_R))
            Q_all[:, _t] = Q
            exponents_all[:, _t] = log_diag_sum / ((_t + 1) * dt)
        sort_idx = exponents_all[:, -1].argsort(dim=-1, descending=True)
        Q_all = Q_all.gather(-1, sort_idx[:, None, None, :].expand_as(Q_all))
        exponents_all = exponents_all.gather(-1, sort_idx[:, None, :].expand_as(exponents_all))
        if unbatched:
            Q_all = Q_all.squeeze(0)
            exponents_all = exponents_all.squeeze(0)
        return (Q_all, exponents_all)
    n_target_dims = lit_model_2.n_target_dims
    with torch.no_grad():
        traj_seq = torch.as_tensor(trajs['test_trajs'].sequence).float().to(device)
        print(f'Test trajectories shape: {traj_seq.shape}')
        z_seq = lit_model_2.encode_trajectory(traj_seq)
        z_dyn = z_seq[..., :n_target_dims] if n_target_dims is not None else z_seq
        print(f'Latent dynamic shape:    {z_dyn.shape}')
        N_traj = z_dyn.shape[0]
        Q_list, exp_list = ([], [])
        for _i in range(N_traj):
            jacs_i = lit_model_2.compute_jacobians(z_dyn[_i:_i + 1])
            Q_i, exp_i = compute_lyapunov_vectors(jacs_i, dt)
            Q_list.append(Q_i.cpu())
            exp_list.append(exp_i.cpu())
            le_final = exp_i[0, -1].cpu().numpy()
            print(f'  Traj {_i}: λ = {le_final}')
        lyapunov_vectors = torch.cat(Q_list, dim=0)
        running_exponents = torch.cat(exp_list, dim=0)
    print(f'\nLyapunov vectors shape:    {lyapunov_vectors.shape}')
    print(f'Running exponents shape:   {running_exponents.shape}')
    print(f'Final exponents (traj 0):  {running_exponents[0, -1].numpy()}')
    return N_traj, lyapunov_vectors, n_target_dims, np, traj_seq, z_dyn


@app.cell
def _(lyapunov_vectors, torch, z_dyn):
    # Project dz onto Lyapunov vectors at each point
    # z_dyn: (3, 1101, 7), lyapunov_vectors: (3, 1101, 7, 7)

    # Compute dz = z_{t+1} - z_t, shape (3, 1100, 7)
    dz = z_dyn[:, 1:, :] - z_dyn[:, :-1, :]

    # Trim Lyapunov vectors to match (use vectors at time t, not t+1)
    Q_t = lyapunov_vectors[:, :-1, :, :]  # (3, 1100, 7, 7)

    # Project: for each point, dot dz with each Lyapunov vector column
    # dz: (3, 1100, 7) -> (3, 1100, 7, 1)
    # Q_t: (3, 1100, 7, 7)
    # projection[..., i] = dz · Q_t[:, :, :, i]
    projections = torch.einsum("btd, btdi -> bti", dz.cpu(), Q_t)  # (3, 1100, 7)

    print(f"dz shape:          {dz.shape}")
    print(f"Q_t shape:         {Q_t.shape}")
    print(f"projections shape: {projections.shape}")
    print(f"\nTraj 0, t=0 projections: {projections[0, 0].numpy()}")
    print(f"Traj 0, t=0 |dz|:       {dz[0, 0].cpu().norm().item():.6f}")
    print(f"Traj 0, t=0 sum(proj²): {(projections[0, 0] ** 2).sum().item():.6f}  (should match |dz|²)")
    return (projections,)


@app.cell
def _(
    N_traj,
    device,
    dt,
    lit_model_2,
    n_target_dims,
    np,
    torch,
    traj_seq,
    z_dyn,
):
    import matplotlib.pyplot as plt
    n_target = n_target_dims
    D_obs = traj_seq.shape[-1]
    T = z_dyn.shape[1]
    print('Computing encoder Jacobians ...')
    enc_jacs = []
    for _b in range(N_traj):
        rows = []
        for j in range(n_target):
            x = traj_seq[_b:_b + 1].to(device).detach().requires_grad_(True)
            z_out = lit_model_2.encode_trajectory(x)
            z_out[0, :T, j].sum().backward()
            rows.append(x.grad[0, :T].cpu())
        J_b = torch.stack(rows, dim=1)
        enc_jacs.append(J_b)
        print(f'  Traj {_b}: done')
    enc_jacs = torch.stack(enc_jacs)
    print(f'Encoder Jacobians shape: {enc_jacs.shape}')
    print('\nComputing SVD ...')
    U_enc, S_enc, _ = torch.linalg.svd(enc_jacs, full_matrices=False)
    print('Singular values (mean ± std across all points):')
    for _i in range(n_target):
        print(f'  σ_{_i + 1} = {S_enc[:, :, _i].mean():.4f} ± {S_enc[:, :, _i].std():.4f}')
    print('\nAligning singular vectors via Procrustes ...')
    for _b in range(N_traj):
        for _t in range(1, T):
            M = U_enc[_b, _t - 1].T @ U_enc[_b, _t]
            Vp, _, Wp = torch.linalg.svd(M)
            R = Wp.T @ Vp.T
            U_enc[_b, _t] = U_enc[_b, _t] @ R
    S_mean = S_enc.mean(dim=(0, 1))
    S_std = S_enc.std(dim=(0, 1))
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
    _ax = _axes[0]
    _ax.bar(range(1, n_target + 1), S_mean.numpy(), yerr=S_std.numpy(), capsize=4)
    _ax.set_xlabel('Index')
    _ax.set_ylabel('σ (log scale)')
    _ax.set_title('Mean encoder Jacobian singular values')
    _ax.set_yscale('log')
    _ax = _axes[1]
    for _i in range(n_target):
        _ax.plot(S_enc[0, :, _i].numpy(), label=f'σ_{_i + 1}', alpha=0.8)
    _ax.set_xlabel('Time step')
    _ax.set_ylabel('σ (log scale)')
    _ax.set_title('Singular values along trajectory 0')
    _ax.legend(fontsize=8)
    _ax.set_yscale('log')
    var_explained = S_mean ** 2 / (S_mean ** 2).sum()
    cum_var = torch.cumsum(var_explained, dim=0)
    _ax = _axes[2]
    _ax.bar(range(1, n_target + 1), cum_var.numpy())
    _ax.axhline(y=0.99, color='r', linestyle='--', label='99%')
    _ax.set_xlabel('Number of dimensions')
    _ax.set_ylabel('Cumulative variance explained')
    _ax.set_title('Intrinsic dimensionality (encoder Jacobian)')
    _ax.legend()
    plt.tight_layout()
    plt.show()
    log_S_mean = torch.log(S_mean)
    gaps = log_S_mean[:-1] - log_S_mean[1:]
    print('\nSingular value gaps (in log space):')
    for _i in range(len(gaps)):
        eps_i = torch.sqrt(S_mean[_i] * S_mean[_i + 1]).item()
        print(f'  gap {_i + 1}→{_i + 2}: {gaps[_i]:.4f}  (ε = {eps_i:.4f})')
    print('\nSoft weight profiles for different ε values:')
    print(f'  {'ε':>8s}', end='')
    for _i in range(n_target):
        print(f'  {'w_' + str(_i + 1):>8s}', end='')
    print()
    for k in range(n_target - 1):
        eps_k = torch.sqrt(S_mean[k] * S_mean[k + 1]).item()
        w_k = S_mean ** 2 / (S_mean ** 2 + eps_k ** 2)
        print(f'  {eps_k:8.4f}', end='')
        for _i in range(n_target):
            print(f'  {w_k[_i]:8.4f}', end='')
        print(f'  ← gap {k + 1}→{k + 2}')
    gap_idx = min(2, len(gaps) - 1)
    eps = torch.sqrt(S_mean[gap_idx] * S_mean[gap_idx + 1]).item()
    print(f'\nUsing ε = {eps:.6f} (gap {gap_idx + 1}→{gap_idx + 2})')
    W = S_enc ** 2 / (S_enc ** 2 + eps ** 2)
    W_sqrt = W.sqrt()
    print(f'Mean soft weights: {W.mean(dim=(0, 1)).numpy()}')
    U_dot = torch.zeros_like(U_enc)
    U_dot[:, 1:-1] = (U_enc[:, 2:] - U_enc[:, :-2]) / (2 * dt)
    U_dot[:, 0] = (U_enc[:, 1] - U_enc[:, 0]) / dt
    U_dot[:, -1] = (U_enc[:, -1] - U_enc[:, -2]) / dt
    skew_errs = []
    for t_check in [100, 300, 500, 700, 900]:
        geom_sample = torch.einsum('btji,btjk->btik', U_dot[:, t_check:t_check + 1], U_enc[:, t_check:t_check + 1])
        skew_err = (geom_sample + geom_sample.transpose(-2, -1)).abs().max().item()
        skew_errs.append(skew_err)
    print(f'\nU̇ᵀU skew-symmetry check (max |U̇ᵀU + UᵀU̇|) at t=100,300,500,700,900:')
    print(f'  {skew_errs}')
    print(f'  Mean: {np.mean(skew_errs):.6f}')
    print('\nComputing projected Jacobians ...')
    with torch.no_grad():
        jacs_list = []
        for _b in range(N_traj):
            J_b = lit_model_2.compute_jacobians(z_dyn[_b:_b + 1]).cpu()[0]
            jacs_list.append(J_b)
        jacs_dyn = torch.stack(jacs_list)
    J_rotated = torch.einsum('btji,btjk,btkl->btil', U_enc, jacs_dyn, U_enc)
    geom_correction = torch.einsum('btji,btjk->btik', U_dot, U_enc)
    J_tangent = J_rotated + geom_correction
    J_projected = J_tangent * W_sqrt.unsqueeze(-1) * W_sqrt.unsqueeze(-2)
    print(f'\nResults:')
    print(f'  U_enc         {U_enc.shape}        — encoder SVD basis at each point')
    print(f'  S_enc         {S_enc.shape}     — singular values (manifold dimension spectrum)')
    print(f'  W             {W.shape}     — soft weights per direction')
    print(f'  J_tangent     {J_tangent.shape}  — full tangent Jacobian (UᵀJU + U̇ᵀU)')
    print(f'  J_projected   {J_projected.shape}  — soft-projected Jacobian')
    print(f'  geom_correction {geom_correction.shape} — U̇ᵀU connection term')
    return (
        J_projected,
        J_tangent,
        S_enc,
        U_dot,
        U_enc,
        enc_jacs,
        jacs_dyn,
        n_target,
        plt,
    )


@app.cell
def _(J_projected, J_tangent, N_traj, TRUE_LYAPUNOV, dt, jacs_dyn, n_target):
    from JacobianODE.models.latent_jacobian import LitLatentJacobianODE
    print('Computing Lyapunov exponents from J_projected ...')
    lyap_projected = LitLatentJacobianODE.compute_lyapunov_exponents(J_projected, dt)
    print('\nSoft-projected Lyapunov exponents per trajectory:')
    for _b in range(N_traj):
        _le = lyap_projected[_b].numpy()
        print(f'  Traj {_b}: {_le}')
    lyap_proj_mean = lyap_projected.mean(dim=0)
    lyap_proj_std = lyap_projected.std(dim=0)
    print(f'\nMean ± std:')
    for _i in range(n_target):
        print(f'  λ_{_i + 1} = {lyap_proj_mean[_i]:+.4f} ± {lyap_proj_std[_i]:.4f}')
    _lyap_tangent = LitLatentJacobianODE.compute_lyapunov_exponents(J_tangent, dt)
    _lyap_raw = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_dyn, dt)
    print('\n--- Comparison ---')
    print(f'{'':>24s} {'Projected':>12s} {'Tangent (no W)':>14s} {'Raw dynamics':>14s}' + (f' {'True':>10s}' if TRUE_LYAPUNOV else ''))
    for _i in range(n_target):
        _row = f'  λ_{_i + 1}:                '
        _row = _row + f'{lyap_proj_mean[_i]:+10.4f}  '
        _row = _row + f'{_lyap_tangent.mean(0)[_i]:+12.4f}  '
        _row = _row + f'{_lyap_raw.mean(0)[_i]:+12.4f}  '
        if TRUE_LYAPUNOV and _i < len(TRUE_LYAPUNOV):
            _row = _row + f'{TRUE_LYAPUNOV[_i]:+8.4f}'
        print(_row)
    return (LitLatentJacobianODE,)


@app.cell
def _(
    J_tangent,
    LitLatentJacobianODE,
    N_traj,
    TRUE_LYAPUNOV,
    U_enc,
    dt,
    jacs_dyn,
    n_target,
    torch,
    z_dyn,
):
    dz_latent = z_dyn[:, 1:, :] - z_dyn[:, :-1, :]
    U_t = U_enc[:, :-1]
    dz_proj = torch.einsum('btd, btdi -> bti', dz_latent.cpu(), U_t)
    var_per_dir = dz_proj.var(dim=(0, 1))
    frac_var = var_per_dir / var_per_dir.sum()
    print('dz variance in encoder SVD basis:')
    for _i in range(n_target):
        print(f'  dir {_i + 1}: var = {var_per_dir[_i]:.6f}  ({frac_var[_i] * 100:.2f}%)')
    print(f'  cumulative: {torch.cumsum(frac_var, dim=0).numpy()}')
    W_dz = var_per_dir / var_per_dir.max()
    W_dz_sqrt = W_dz.sqrt()
    print(f'\nSoft weights (max-normalized dz variance): {W_dz.numpy()}')
    print(f'Sum of weights: {W_dz.sum():.4f}  (base case = {n_target})')
    J_projected_dz = J_tangent * W_dz_sqrt.unsqueeze(-1) * W_dz_sqrt.unsqueeze(-2)
    print('\nComputing Lyapunov exponents ...')
    _lyap_proj_dz = LitLatentJacobianODE.compute_lyapunov_exponents(J_projected_dz, dt)
    _lyap_tangent = LitLatentJacobianODE.compute_lyapunov_exponents(J_tangent, dt)
    _lyap_raw = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_dyn, dt)
    print(f'\n{'':>6s} {'dz-weighted':>12s} {'Tangent (W=I)':>14s} {'Raw dynamics':>14s}' + (f' {'True':>10s}' if TRUE_LYAPUNOV else ''))
    for _i in range(n_target):
        _row = f'  λ_{_i + 1}: '
        _row = _row + f'{_lyap_proj_dz.mean(0)[_i]:+10.4f}  '
        _row = _row + f'{_lyap_tangent.mean(0)[_i]:+12.4f}  '
        _row = _row + f'{_lyap_raw.mean(0)[_i]:+12.4f}  '
        if TRUE_LYAPUNOV and _i < len(TRUE_LYAPUNOV):
            _row = _row + f'{TRUE_LYAPUNOV[_i]:+8.4f}'
        print(_row)
    print(f'\ndz-weighted Lyapunov exponents per trajectory:')
    for _b in range(N_traj):
        print(f'  Traj {_b}: {_lyap_proj_dz[_b].numpy()}')
    return (J_projected_dz,)


@app.cell
def _(
    J_projected_dz,
    J_tangent,
    LitLatentJacobianODE,
    N_traj,
    S_enc,
    TRUE_LYAPUNOV,
    dt,
    jacs_dyn,
    n_target,
):
    W_svd = S_enc ** 2 / S_enc[:, :, 0:1] ** 2
    W_svd_sqrt = W_svd.sqrt()
    print('SVD weights (mean across all points):')
    W_svd_mean = W_svd.mean(dim=(0, 1))
    for _i in range(n_target):
        print(f'  w_{_i + 1} = {W_svd_mean[_i]:.4f}')
    print(f'Sum of weights: {W_svd_mean.sum():.4f}  (base case = {n_target})')
    J_projected_svd = J_tangent * W_svd_sqrt.unsqueeze(-1) * W_svd_sqrt.unsqueeze(-2)
    print('\nComputing Lyapunov exponents ...')
    lyap_svd = LitLatentJacobianODE.compute_lyapunov_exponents(J_projected_svd, dt)
    _lyap_tangent = LitLatentJacobianODE.compute_lyapunov_exponents(J_tangent, dt)
    _lyap_raw = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_dyn, dt)
    _lyap_proj_dz = LitLatentJacobianODE.compute_lyapunov_exponents(J_projected_dz, dt)
    print(f'\n{'':>6s} {'SVD-weighted':>12s} {'dz-weighted':>12s} {'Tangent (W=I)':>14s} {'Raw dynamics':>14s}' + (f' {'True':>10s}' if TRUE_LYAPUNOV else ''))
    for _i in range(n_target):
        _row = f'  λ_{_i + 1}: '
        _row = _row + f'{lyap_svd.mean(0)[_i]:+10.4f}  '
        _row = _row + f'{_lyap_proj_dz.mean(0)[_i]:+10.4f}  '
        _row = _row + f'{_lyap_tangent.mean(0)[_i]:+12.4f}  '
        _row = _row + f'{_lyap_raw.mean(0)[_i]:+12.4f}  '
        if TRUE_LYAPUNOV and _i < len(TRUE_LYAPUNOV):
            _row = _row + f'{TRUE_LYAPUNOV[_i]:+8.4f}'
        print(_row)
    print(f'\nSVD-weighted Lyapunov exponents per trajectory:')
    for _b in range(N_traj):
        print(f'  Traj {_b}: {lyap_svd[_b].numpy()}')
    return


@app.cell
def _(
    LitLatentJacobianODE,
    TRUE_LYAPUNOV,
    U_dot,
    U_enc,
    dt,
    jacs_dyn,
    n_target,
    plt,
    torch,
):
    print('Computing tangent-space Lyapunov exponents for d = 1..{} ...\n'.format(n_target))
    lyap_by_d = {}
    for _d in range(1, n_target + 1):
        _U_d = U_enc[..., :_d]
        Ud_d = U_dot[..., :_d]
        _J_proj = torch.einsum('btji,btjk,btkl->btil', _U_d, jacs_dyn, _U_d)
        geom = torch.einsum('btji,btjk->btik', Ud_d, _U_d)
        J_bar = _J_proj + geom
        _lyap_d = LitLatentJacobianODE.compute_lyapunov_exponents(J_bar, dt)
        lyap_by_d[_d] = _lyap_d
        _mean = _lyap_d.mean(dim=0).numpy()
        _std = _lyap_d.std(dim=0).numpy()
        _exps_str = '  '.join((f'{m:+.4f}±{s:.4f}' for m, s in zip(_mean, _std)))
        print(f'  d={_d}: {_exps_str}')
    print(f'\n{'d':>3s}', end='')
    for _i in range(n_target):
        print(f'  {'λ_' + str(_i + 1):>12s}', end='')
    print(f'  {'True':>10s}' if TRUE_LYAPUNOV else '')
    for _d in range(1, n_target + 1):
        _row = f'  {_d}'
        _mean = lyap_by_d[_d].mean(dim=0)
        for _i in range(_d):
            _row = _row + f'  {_mean[_i]:+10.4f}  '
        for _i in range(_d, n_target):
            _row = _row + f'  {'—':>10s}  '
        if TRUE_LYAPUNOV:
            _row = _row + '  |'
            for _i in range(min(_d, len(TRUE_LYAPUNOV))):
                _row = _row + f' {TRUE_LYAPUNOV[_i]:+.4f}'
        print(_row)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _ax = _axes[0]
    for _i in range(n_target):
        _ds = list(range(_i + 1, n_target + 1))
        _means = [lyap_by_d[_d].mean(0)[_i].item() for _d in _ds]
        _stds = [lyap_by_d[_d].std(0)[_i].item() for _d in _ds]
        _ax.errorbar(_ds, _means, yerr=_stds, marker='o', markersize=4, capsize=3, label=f'λ_{_i + 1}')
    if TRUE_LYAPUNOV:
        for _i, _le in enumerate(TRUE_LYAPUNOV):
            _ax.axhline(y=_le, color=f'C{_i}', linestyle='--', alpha=0.5)
    _ax.set_xlabel('Projection dimension d')
    _ax.set_ylabel('Lyapunov exponent')
    _ax.set_title('Tangent-space Lyapunov spectrum vs projection dimension')
    _ax.legend(fontsize=8)
    _ax.set_xticks(range(1, n_target + 1))
    _ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    _ax = _axes[1]
    for _i in range(min(3, n_target)):
        _ds = list(range(_i + 1, n_target + 1))
        _means = [lyap_by_d[_d].mean(0)[_i].item() for _d in _ds]
        _stds = [lyap_by_d[_d].std(0)[_i].item() for _d in _ds]
        _ax.errorbar(_ds, _means, yerr=_stds, marker='o', markersize=5, capsize=3, label=f'λ_{_i + 1}')
    if TRUE_LYAPUNOV:
        for _i in range(min(3, len(TRUE_LYAPUNOV))):
            _ax.axhline(y=TRUE_LYAPUNOV[_i], color=f'C{_i}', linestyle='--', alpha=0.5, label=f'true λ_{_i + 1} = {TRUE_LYAPUNOV[_i]}')
    _ax.set_xlabel('Projection dimension d')
    _ax.set_ylabel('Lyapunov exponent')
    _ax.set_title('Top 3 exponents vs d (dashed = true Lorenz)')
    _ax.legend(fontsize=8)
    _ax.set_xticks(range(1, n_target + 1))
    _ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    LitLatentJacobianODE,
    TRUE_LYAPUNOV,
    dt,
    enc_jacs,
    jacs_dyn,
    n_target,
    np,
    plt,
    torch,
):
    from scipy.linalg import schur as scipy_schur, rsf2csf
    print('Recomputing raw SVD (no Procrustes) ...')
    U_raw, S_raw, _ = torch.linalg.svd(enc_jacs, full_matrices=False)
    print('\n=== Approach 1: Raw SVD projection (U_d^T J U_d, no geometric correction) ===\n')
    lyap_svd_proj = {}
    for _d in range(1, n_target + 1):
        _U_d = U_raw[..., :_d]
        _J_proj = torch.einsum('btji,btjk,btkl->btil', _U_d, jacs_dyn, _U_d)
        _lyap_d = LitLatentJacobianODE.compute_lyapunov_exponents(_J_proj, dt)
        lyap_svd_proj[_d] = _lyap_d
        _mean = _lyap_d.mean(dim=0).numpy()
        _std = _lyap_d.std(dim=0).numpy()
        _exps_str = '  '.join((f'{m:+.4f}±{s:.4f}' for m, s in zip(_mean, _std)))
        print(f'  d={_d}: {_exps_str}')
    print('\n=== Approach 2: Schur decomposition of J(t) ===\n')
    B_n, T_n = jacs_dyn.shape[:2]
    J_schur_blocks = {}
    jacs_np = jacs_dyn.numpy()
    schur_T = np.empty_like(jacs_np)
    print('  Computing ordered Schur decompositions ...')
    for _b in range(B_n):
        for _t in range(T_n):
            J_pt = jacs_np[_b, _t]
            T_s, Q_s = scipy_schur(J_pt, output='real', sort=lambda r, i: -r)
            schur_T[_b, _t] = T_s
    schur_T_torch = torch.from_numpy(schur_T).float()
    lyap_schur = {}
    for _d in range(1, n_target + 1):
        T_block = schur_T_torch[:, :, :_d, :_d]
        _lyap_d = LitLatentJacobianODE.compute_lyapunov_exponents(T_block, dt)
        lyap_schur[_d] = _lyap_d
        _mean = _lyap_d.mean(dim=0).numpy()
        _std = _lyap_d.std(dim=0).numpy()
        _exps_str = '  '.join((f'{m:+.4f}±{s:.4f}' for m, s in zip(_mean, _std)))
        print(f'  d={_d}: {_exps_str}')
    print(f'\n{'d':>3s}  {'--- SVD projection ---':>40s}  {'--- Schur block ---':>40s}' + ('  True' if TRUE_LYAPUNOV else ''))
    for _d in range(1, n_target + 1):
        svd_mean = lyap_svd_proj[_d].mean(0)
        sch_mean = lyap_schur[_d].mean(0)
        svd_str = '  '.join((f'{svd_mean[_i]:+.4f}' for _i in range(_d)))
        sch_str = '  '.join((f'{sch_mean[_i]:+.4f}' for _i in range(_d)))
        true_str = ''
        if TRUE_LYAPUNOV:
            true_str = '  | ' + '  '.join((f'{TRUE_LYAPUNOV[_i]:+.4f}' for _i in range(min(_d, len(TRUE_LYAPUNOV)))))
        print(f'  {_d}  {svd_str:>40s}  {sch_str:>40s}{true_str}')
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    for _ax, data, title in [(_axes[0], lyap_svd_proj, 'SVD projection (U_d^T J U_d)'), (_axes[1], lyap_schur, 'Schur block (top-left d×d of T)')]:
        for _i in range(n_target):
            _ds = list(range(_i + 1, n_target + 1))
            _means = [data[_d].mean(0)[_i].item() for _d in _ds]
            _stds = [data[_d].std(0)[_i].item() for _d in _ds]
            _ax.errorbar(_ds, _means, yerr=_stds, marker='o', markersize=4, capsize=3, label=f'λ_{_i + 1}')
        if TRUE_LYAPUNOV:
            for _i, _le in enumerate(TRUE_LYAPUNOV):
                _ax.axhline(y=_le, color=f'C{_i}', linestyle='--', alpha=0.4)
        _ax.set_xlabel('Projection dimension d')
        _ax.set_ylabel('Lyapunov exponent')
        _ax.set_title(title)
        _ax.legend(fontsize=7)
        _ax.set_xticks(range(1, n_target + 1))
        _ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(projections):
    squared_projections = (projections**2)
    return (squared_projections,)


@app.cell
def _(squared_projections, torch):
    percent_var = squared_projections/squared_projections.sum(axis=-1, keepdims=True)
    torch.cumsum(percent_var.mean(axis=(0, 1)), dim=0)
    return (percent_var,)


@app.cell
def _(percent_var, plt):
    all_pts_percent_var = percent_var.reshape(-1, percent_var.shape[-1])
    plt.figure(figsize=(10, 6))
    plt.boxplot([all_pts_percent_var[:, _i].detach().cpu().numpy() for _i in range(all_pts_percent_var.shape[1])])
    plt.xlabel('Lyapunov Vector Index')
    plt.ylabel('Fraction of |dz|² captured')
    plt.title('Distribution of |dz|² Fraction per Lyapunov Direction')
    plt.show()
    return


@app.cell
def _(squared_projections, torch):
    percent_overall_var = squared_projections/squared_projections.sum()
    torch.cumsum(percent_overall_var.sum(axis=(0, 1)), dim=0)
    return (percent_overall_var,)


@app.cell
def _(percent_overall_var, plt):
    all_pts_percent_overall_var = percent_overall_var.reshape(-1, percent_overall_var.shape[-1])
    plt.figure(figsize=(10, 6))
    plt.boxplot([all_pts_percent_overall_var[:, _i].detach().cpu().numpy() for _i in range(all_pts_percent_overall_var.shape[1])])
    plt.xlabel('Lyapunov Vector Index')
    plt.ylabel('Fraction of |dz|² captured')
    plt.title('Distribution of |dz|² Fraction per Lyapunov Direction')
    plt.yscale('log')
    plt.show()
    return


@app.cell
def _(projections, torch):
    squared_projections_1 = projections ** 2  # (3, 1100, 7)                                                              
    variance_per_direction = squared_projections_1.sum(dim=(0, 1))  # (7,)
    fraction_per_direction = variance_per_direction / variance_per_direction.sum()
    cumulative = torch.cumsum(fraction_per_direction, dim=0)
    cumulative
    return (fraction_per_direction,)


@app.cell
def _(plt, projections, torch):
    _svd_vals = torch.linalg.svdvals(projections.reshape(-1, projections.shape[-1]))
    plt.scatter(range(len(_svd_vals)), _svd_vals.cpu().numpy())
    return


@app.cell
def _(plt, projections, torch):
    _svd_vals = torch.linalg.svdvals(projections.reshape(-1, projections.shape[-1]))
    plt.scatter(range(len(_svd_vals)), _svd_vals.cpu().numpy())
    return


@app.cell
def _(fraction_per_direction, plt):
    plt.scatter(range(len(fraction_per_direction)), fraction_per_direction)
    plt.xlabel("Lyapunov Vector Index")
    plt.ylabel("Fraction of |dz|² captured")
    plt.title("Distribution of |dz|² Fraction per Lyapunov Direction")
    plt.yscale("log")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
