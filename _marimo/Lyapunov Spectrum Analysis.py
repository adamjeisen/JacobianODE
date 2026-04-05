import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lyapunov Spectrum Analysis — Sweep Group

    Compute empirical and predicted Lyapunov spectra for every run in a W&B sweep group.

    **Workflow:**
    1. Select W&B project / group
    2. For each run: load best-validation model, compute Lyapunov spectra on test trajectories
    3. Optionally compute observed-space Jacobians via $J_E^{-1} J_\text{latent} J_E$
    4. Cache results per-run, generate HTML reports
    """)
    return


@app.cell
def _():
    import base64
    import json
    import traceback
    from datetime import datetime
    from io import BytesIO
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import OmegaConf

    from JacobianODE.jacobians.checkpoints.loader import load_run
    from JacobianODE.jacobians.metrics import r2_score
    from JacobianODE.jacobians.run_analytics import plot_lyapunov_spectrum
    from JacobianODE.jacobians.tuning.sweep import discover_sweep_runs
    from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

    torch.set_float32_matmul_precision("high")
    return (
        BytesIO,
        LitLatentJacobianODE,
        OmegaConf,
        Path,
        base64,
        datetime,
        discover_sweep_runs,
        json,
        load_run,
        np,
        pd,
        plot_lyapunov_spectrum,
        plt,
        r2_score,
        torch,
        traceback,
    )


@app.cell(hide_code=True)
def _(mo):
    WANDB_PROJECT = "WMTask_INDall_N1_D1_NormTrue_T128__JacobianODE"
    # WANDB_GROUP = None
    WANDB_GROUP = "spline_coupling__sweep_lc_x_kl_dyn_vae_sample_all_losses"

    wb_entity = mo.ui.text(
        value="JacobianODE", label="W&B Entity", full_width=True
    )
    wb_project = mo.ui.text(
        value=WANDB_PROJECT,
        label="W&B Project",
        full_width=True,
    )
    wb_group = mo.ui.text(
        value=WANDB_GROUP,
        label="W&B Group (leave empty for all)",
        full_width=True,
    )
    save_dir_input = mo.ui.text(
        value="/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs",
        label="Checkpoint Save Dir",
        full_width=True,
    )
    true_lyap_input = mo.ui.text(
        # value="0.91, 0.0, -14.57",
        value="",
        label="True Lyapunov exponents (comma-separated, or empty)",
        full_width=True,
    )
    obs_jac_switch = mo.ui.switch(
        value=True, label="Compute observed-space Jacobians (J_E⁻¹ J_t J_E)"
    )
    force_recompute_switch = mo.ui.switch(
        value=False, label="Force recompute (ignore cache)"
    )

    mo.vstack(
        [
            mo.md("## Configuration"),
            wb_entity,
            wb_project,
            wb_group,
            save_dir_input,
            true_lyap_input,
            mo.hstack([obs_jac_switch, force_recompute_switch]),
        ]
    )
    return (
        force_recompute_switch,
        obs_jac_switch,
        save_dir_input,
        true_lyap_input,
        wb_entity,
        wb_group,
        wb_project,
    )


@app.cell
def _(
    force_recompute_switch,
    mo,
    obs_jac_switch,
    save_dir_input,
    true_lyap_input,
    wb_entity,
    wb_group,
    wb_project,
):
    ENTITY = wb_entity.value.strip()
    PROJECT = wb_project.value.strip()
    GROUP = wb_group.value.strip() or None
    SAVE_DIR = save_dir_input.value.strip()
    COMPUTE_OBS_JAC = obs_jac_switch.value
    FORCE_RECOMPUTE = force_recompute_switch.value

    lyap_str = true_lyap_input.value.strip() if true_lyap_input.value else ""
    TRUE_LYAPUNOV = (
        [float(x.strip()) for x in lyap_str.split(",") if x.strip()]
        if lyap_str
        else None
    )

    PROJECT_PATH = f"{ENTITY}/{PROJECT}"

    mo.md(
        f"**Config:** `{PROJECT_PATH}` | group=`{GROUP}` | obs_jac={COMPUTE_OBS_JAC} | "
        f"true_lyap={TRUE_LYAPUNOV}"
    )
    return (
        COMPUTE_OBS_JAC,
        ENTITY,
        FORCE_RECOMPUTE,
        GROUP,
        PROJECT,
        PROJECT_PATH,
        SAVE_DIR,
        TRUE_LYAPUNOV,
    )


@app.cell
def _(ENTITY, GROUP, PROJECT, discover_sweep_runs, mo):
    discovered = discover_sweep_runs(
        ENTITY, PROJECT, wandb_group=GROUP, verbose=True
    )
    mo.md(
        f"**Found {len(discovered.run_ids)} runs.** "
        f"IDs: {', '.join(discovered.run_ids[:10])}"
        + ("..." if len(discovered.run_ids) > 10 else "")
    )
    return (discovered,)


@app.cell
def _(ENTITY, PROJECT, discovered, mo):
    # ---- Fetch W&B validation metrics for each run ----
    import wandb as _wandb

    _api = _wandb.Api(timeout=90)
    wandb_val_metrics = {}
    for _run_id in discovered.run_ids:
        try:
            _run = _api.run(f"{ENTITY}/{PROJECT}/{_run_id}")
            _summary = _run.summary
            wandb_val_metrics[_run_id] = {
                "trajectory_val_loss": _summary.get("trajectory val_loss"),
                "val_loop_closure_loss": _summary.get("val/loop_closure_loss"),
                "mean_val_loss": _summary.get("mean val loss"),
            }
        except Exception as _e:
            wandb_val_metrics[_run_id] = {
                "trajectory_val_loss": None,
                "val_loop_closure_loss": None,
                "mean_val_loss": None,
            }

    mo.md(f"**Fetched W&B validation metrics for {len(wandb_val_metrics)} runs.**")
    return (wandb_val_metrics,)


@app.cell
def _(BytesIO, base64, json, np, plt, r2_score, torch):
    # ---------- utility functions ----------

    def z_dyn_slice(z, n_target_dims):
        if n_target_dims is not None:
            return z[..., :n_target_dims]
        return z

    def compute_encoder_jacobian(lit_model, traj_obs, n_target_dims, device):
        """Compute encoder Jacobian J_E at each point along a trajectory.

        Args:
            lit_model: trained LitLatentJacobianODE
            traj_obs: (T, D_obs) tensor in observation space
            n_target_dims: int or None
            device: torch device

        Returns:
            J_E: (T', D_dyn, D_obs) tensor
        """
        encoder = lit_model.encoder
        margin = getattr(encoder, "context_margin", 0)

        if hasattr(encoder, "time_window"):
            w = encoder.time_window
            D_raw = traj_obs.shape[-1]
            T_avail = traj_obs.shape[0] - w + 1
            windows = traj_obs.unfold(0, w, 1).permute(0, 2, 1)  # (T', w, D_raw)
            x_flat = windows.reshape(T_avail, w * D_raw)

            def encode_point(x_pt):
                z = encoder.encode(x_pt.reshape(1, w, D_raw)).squeeze(0)
                return z[:n_target_dims] if n_target_dims is not None else z
        else:
            T_prime = traj_obs.shape[0] - margin
            x_flat = traj_obs[margin : margin + T_prime]  # (T', D_obs)

            def encode_point(x_pt):
                z = encoder.encode(x_pt.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                return z[:n_target_dims] if n_target_dims is not None else z

        D_obs = x_flat.shape[-1]
        D_dyn = n_target_dims if n_target_dims is not None else D_obs

        if D_dyn <= D_obs:
            jac_fn = torch.func.jacrev(encode_point)
        else:
            jac_fn = torch.func.jacfwd(encode_point)

        was_training = encoder.training
        encoder.eval()
        try:
            # Process in chunks to avoid OOM
            chunk_size = 64
            J_chunks = []
            for ci in range(0, x_flat.shape[0], chunk_size):
                chunk = x_flat[ci : ci + chunk_size]
                J_chunk = torch.func.vmap(jac_fn)(chunk)  # (chunk, D_dyn, D_obs)
                J_chunks.append(J_chunk)
            J_E = torch.cat(J_chunks, dim=0)
        finally:
            if was_training:
                encoder.train()

        return J_E  # (T', D_dyn, D_obs)

    def compute_observed_space_jacobians(lit_model, traj_single, n_target_dims, device):
        """Compute J_obs = J_E_pinv @ J_latent @ J_E for a single trajectory.

        Args:
            lit_model: trained LitLatentJacobianODE
            traj_single: (T, D_obs) tensor
            n_target_dims: int or None
            device: torch device

        Returns:
            J_obs: (T', D_obs, D_obs) tensor
        """
        traj = traj_single.unsqueeze(0).to(device)  # (1, T, D_obs)

        # 1. Encode -> z_dyn
        z_full = lit_model.encode_trajectory(traj)  # (1, T', D_latent)
        z_dyn = z_dyn_slice(z_full, n_target_dims)
        T_prime = z_dyn.shape[1]

        # 2. Latent Jacobians
        J_latent = lit_model.compute_jacobians(z_dyn)[0]  # (T', D_dyn, D_dyn)

        # 3. Encoder Jacobian
        # Align: J_E should have the same T' as J_latent
        J_E = compute_encoder_jacobian(
            lit_model, traj_single, n_target_dims, device
        )  # (T'', D_dyn, D_obs)
        # Trim to match (they should already match but be safe)
        T_min = min(J_E.shape[0], J_latent.shape[0])
        J_E = J_E[:T_min]
        J_latent = J_latent[:T_min]

        # 4. Pseudo-inverse
        J_E_pinv = torch.linalg.pinv(J_E)  # (T', D_obs, D_dyn)

        # 5. Assemble: J_obs = J_E_pinv @ J_latent @ J_E
        J_obs = J_E_pinv @ J_latent @ J_E  # (T', D_obs, D_obs)

        return J_obs

    def save_results(path, data):
        """Save results dict to JSON, converting numpy arrays to lists."""
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(convert_for_json(data), f, indent=2)

    def load_cached(path):
        """Load cached JSON results, or return None if missing/corrupt."""
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if data:
                    return data
            except (json.JSONDecodeError, ValueError):
                path.unlink()  # remove corrupt cache file
        return None

    def fig_to_base64(fig):
        """Convert matplotlib figure to base64-encoded PNG string."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    def spectrum_comparison(pred_le, emp_le):
        """Compute correlation and R^2 between predicted and empirical LE spectra."""
        n = min(len(pred_le), len(emp_le))
        if n == 0:
            return float("nan"), float("nan")
        p = np.asarray(pred_le[:n], dtype=np.float64)
        e = np.asarray(emp_le[:n], dtype=np.float64)
        corr = float(np.corrcoef(p, e)[0, 1]) if n > 1 else float("nan")
        r2 = float(r2_score(torch.tensor(e), torch.tensor(p)))
        return corr, r2

    def spectrum_mse(pred_le, emp_le):
        """MSE between predicted and empirical spectra (first min(len) exponents)."""
        n = min(len(pred_le), len(emp_le))
        if n == 0:
            return float("nan")
        p = np.asarray(pred_le[:n], dtype=np.float64)
        e = np.asarray(emp_le[:n], dtype=np.float64)
        return float(np.mean((p - e) ** 2))

    def lyap_to_timescale(le_array):
        """Convert Lyapunov exponents to timescales: 1/|lambda|.

        Timescales de-emphasise large negative exponents (fast-decaying modes)
        and better reflect dynamically relevant time horizons.
        """
        le = np.asarray(le_array, dtype=np.float64)
        with np.errstate(divide="ignore"):
            ts = 1.0 / np.abs(le)
        ts[~np.isfinite(ts)] = np.nan  # lambda=0 -> nan timescale
        return ts

    return (
        compute_observed_space_jacobians,
        fig_to_base64,
        load_cached,
        lyap_to_timescale,
        save_results,
        spectrum_comparison,
        spectrum_mse,
        z_dyn_slice,
    )


@app.cell
def _(
    COMPUTE_OBS_JAC,
    FORCE_RECOMPUTE,
    LitLatentJacobianODE,
    OmegaConf,
    PROJECT_PATH,
    Path,
    SAVE_DIR,
    compute_observed_space_jacobians,
    datetime,
    discovered,
    load_cached,
    load_run,
    mo,
    np,
    r2_score,
    save_results,
    spectrum_comparison,
    torch,
    traceback,
    z_dyn_slice,
):
    # ---- Main processing loop ----
    _repo_root = Path(__file__).resolve().parent.parent
    results_dir = _repo_root / "_marimo" / "results" / "lyapunov" / PROJECT_PATH.split("/")[-1]
    if discovered.run_ids:
        # Use group name in path if available
        _group_name = getattr(discovered, "_group", None)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    # Shared data (loaded once from first run)
    _eq_shared = None
    _trajs_shared = None
    _dt_shared = None
    _cfg_shared = None
    _data_loaded = False

    with mo.status.spinner(title="Processing runs...") as _status:
        for _ri, _run_id in enumerate(discovered.run_ids):
            _status.update(
                title=f"Processing run {_ri + 1}/{len(discovered.run_ids)}: {_run_id}"
            )

            # Check cache
            _cache_path = results_dir / f"{_run_id}.json"
            if not FORCE_RECOMPUTE:
                _cached = load_cached(_cache_path)
                if _cached is not None:
                    all_results[_run_id] = _cached
                    print(f"  [{_ri+1}/{len(discovered.run_ids)}] {_run_id}: loaded from cache")
                    continue

            try:
                # Load model
                (
                    _run_obj,
                    _cfg,
                    _eq,
                    _dt,
                    _values,
                    _train_dl,
                    _val_dl,
                    _test_dl,
                    _trajs,
                    _lit_model,
                ) = load_run(
                    PROJECT_PATH,
                    run_id=_run_id,
                    save_dir=SAVE_DIR,
                    generate_data=(not _data_loaded),
                    verbose=False,
                )

                if not _data_loaded:
                    _eq_shared = _eq
                    _trajs_shared = _trajs
                    _dt_shared = _dt
                    _cfg_shared = _cfg
                    _data_loaded = True

                # Ensure best checkpoint is loaded (load_run may not load it
                # for standard JacobianODE runs in the non-encoder-only path).
                from JacobianODE.jacobians.checkpoints.loader import load_checkpoint
                try:
                    load_checkpoint(
                        _run_obj, _cfg, _lit_model,
                        save_dir=SAVE_DIR, verbose=False,
                    )
                except Exception:
                    pass  # checkpoint already loaded by load_run

                _lit_model.eval()
                _lit_model = _lit_model.to(device)

                # Extract config
                _n_target_dims = OmegaConf.select(_cfg, "model.n_target_dims", default=None)
                if _n_target_dims is not None:
                    _n_target_dims = int(_n_target_dims)
                _lc_weight = float(OmegaConf.select(_cfg, "training.lightning.loop_closure_weight", default=0.0))
                _kl_dyn = float(OmegaConf.select(_cfg, "training.lightning.kl_dyn_weight", default=0.0))

                # Normalization params
                _mu = np.asarray(OmegaConf.select(_cfg_shared, "data.postprocessing.mu", default=0.0))
                _sigma = np.asarray(OmegaConf.select(_cfg_shared, "data.postprocessing.sigma", default=1.0))

                # Test trajectories
                if "test_trajs_full" in _trajs_shared:
                    _test_seq = _trajs_shared["test_trajs_full"].sequence
                else:
                    _test_seq = _trajs_shared["test_trajs"].sequence
                _n_test = _test_seq.shape[0]

                # Delay embedding info
                _delay_params = OmegaConf.select(_cfg_shared, "data.train_test_params.delay_embedding_params", default=None)
                _n_delays = int(_delay_params.get("n_delays", 1)) if _delay_params else 1

                # Result dict
                _result = {
                    "run_id": _run_id,
                    "config": {
                        "loop_closure_weight": _lc_weight,
                        "kl_dyn_weight": _kl_dyn,
                        "n_target_dims": _n_target_dims,
                    },
                    "pred_lyap_per_traj": [],
                    "emp_lyap_per_traj": [],
                    "spectrum_r2_per_traj": [],
                    "spectrum_corr_per_traj": [],
                    "obs_jac_lyap_per_traj": [],
                    "jac_r2_per_traj": [],
                    "timestamp": datetime.now().isoformat(),
                }

                # -- Process each test trajectory --
                with torch.no_grad():
                    # Encode all test trajectories at once (they share the encoder)
                    _traj_t = torch.as_tensor(_test_seq).float().to(device)
                    _z_full_all = _lit_model.encode_trajectory(_traj_t)
                    z_dyn_slice_all = z_dyn_slice(_z_full_all, _n_target_dims)

                    for _ti in range(_n_test):
                        # -- Predicted Lyapunov (latent space) --
                        _jacs_pred = _lit_model.compute_jacobians(
                            z_dyn_slice_all[_ti : _ti + 1]
                        )[0]  # (T', D_dyn, D_dyn)
                        _pred_le = LitLatentJacobianODE.compute_lyapunov_exponents(
                            _jacs_pred.cpu(), _dt_shared
                        )

                        # -- Empirical Lyapunov (from eq.jac) --
                        _traj_raw = np.asarray(_test_seq[_ti]) * _sigma + _mu
                        if hasattr(_eq_shared, "model"):
                            _traj_raw_input = torch.as_tensor(_traj_raw).float()
                        else:
                            _traj_raw_input = _traj_raw
                        _jacs_true = _eq_shared.jac(_traj_raw_input, t=0)
                        _jacs_true_t = torch.as_tensor(_jacs_true).float()
                        _emp_le = LitLatentJacobianODE.compute_lyapunov_exponents(
                            _jacs_true_t, _dt_shared
                        )

                        # -- Spectrum comparison --
                        _corr, _r2 = spectrum_comparison(
                            _pred_le.numpy(), _emp_le.numpy()
                        )

                        _result["pred_lyap_per_traj"].append(_pred_le.numpy().tolist())
                        _result["emp_lyap_per_traj"].append(_emp_le.numpy().tolist())
                        _result["spectrum_r2_per_traj"].append(_r2)
                        _result["spectrum_corr_per_traj"].append(_corr)

                        # -- Optional: observed-space Jacobian --
                        if COMPUTE_OBS_JAC:
                            try:
                                _J_obs = compute_observed_space_jacobians(
                                    _lit_model,
                                    _traj_t[_ti],
                                    _n_target_dims,
                                    device,
                                )
                                _obs_le = LitLatentJacobianODE.compute_lyapunov_exponents(
                                    _J_obs.cpu(), _dt_shared
                                )
                                _result["obs_jac_lyap_per_traj"].append(
                                    _obs_le.numpy().tolist()
                                )

                                # Raw Jacobian R^2 only valid when n_delays == 1
                                if _n_delays == 1:
                                    _T_cmp = min(_J_obs.shape[0], _jacs_true_t.shape[0])
                                    _jac_r2 = float(
                                        r2_score(
                                            _jacs_true_t[:_T_cmp].reshape(-1),
                                            _J_obs[:_T_cmp].cpu().reshape(-1),
                                        )
                                    )
                                    _result["jac_r2_per_traj"].append(_jac_r2)
                            except Exception as _obs_err:
                                print(f"    Obs-space Jacobian failed for traj {_ti}: {_obs_err}")

                    if _ti < 3 or _ti == _n_test - 1:
                        print(
                            f"    Traj {_ti}: pred_LE={np.array(_result['pred_lyap_per_traj'][-1])[:3]}, "
                            f"r2={_result['spectrum_r2_per_traj'][-1]:.4f}"
                        )

                # -- Aggregate stats --
                _pred_arr = np.array(_result["pred_lyap_per_traj"])
                _emp_arr = np.array(_result["emp_lyap_per_traj"])
                _result["pred_lyap_mean"] = _pred_arr.mean(axis=0).tolist()
                _result["pred_lyap_std"] = _pred_arr.std(axis=0).tolist()
                _result["emp_lyap_mean"] = _emp_arr.mean(axis=0).tolist()
                _result["emp_lyap_std"] = _emp_arr.std(axis=0).tolist()
                _result["mean_spectrum_r2"] = float(np.nanmean(_result["spectrum_r2_per_traj"]))
                _result["mean_spectrum_corr"] = float(np.nanmean(_result["spectrum_corr_per_traj"]))

                if _result["obs_jac_lyap_per_traj"]:
                    _obs_arr = np.array(_result["obs_jac_lyap_per_traj"])
                    _result["obs_jac_lyap_mean"] = _obs_arr.mean(axis=0).tolist()
                    _result["obs_jac_lyap_std"] = _obs_arr.std(axis=0).tolist()
                if _result["jac_r2_per_traj"]:
                    _result["mean_jac_r2"] = float(np.nanmean(_result["jac_r2_per_traj"]))

                # Save to cache
                save_results(_cache_path, _result)
                all_results[_run_id] = _result

                print(
                    f"  [{_ri+1}/{len(discovered.run_ids)}] {_run_id}: "
                    f"mean_r2={_result['mean_spectrum_r2']:.4f}, "
                    f"mean_corr={_result['mean_spectrum_corr']:.4f}"
                )

                # Cleanup
                _lit_model.cpu()
                del _lit_model
                torch.cuda.empty_cache()

            except Exception as _err:
                all_results[_run_id] = {
                    "run_id": _run_id,
                    "error": str(_err),
                    "traceback": traceback.format_exc(),
                }
                print(
                    f"  [{_ri+1}/{len(discovered.run_ids)}] {_run_id}: ERROR - {_err}"
                )

    mo.md(f"**Done.** Processed {len(all_results)} runs. Results cached in `{results_dir}`.")
    return all_results, results_dir


@app.cell
def _(
    all_results,
    discovered,
    lyap_to_timescale,
    mo,
    np,
    pd,
    spectrum_comparison,
    spectrum_mse,
    wandb_val_metrics,
):
    # ---- Summary table ----
    # Per-trajectory mean R²/MSE and R²/MSE of mean spectra, at full / top-10 / top-5
    # In both exponent space and timescale space (1/|lambda|).

    def partial_metric_per_traj(pred_per_traj, emp_per_traj, k, metric_fn):
        """Mean of per-trajectory metric using only first k exponents."""
        vals = []
        for p, e in zip(pred_per_traj, emp_per_traj):
            pa, ea = np.array(p), np.array(e)
            n = min(len(pa), len(ea), k)
            if n == 0:
                continue
            vals.append(metric_fn(pa[:n], ea[:n]))
        return float(np.nanmean(vals)) if vals else None

    def r2_fn(p, e):
        return spectrum_comparison(p, e)[1]

    def ts_r2_fn(p, e):
        return spectrum_comparison(lyap_to_timescale(p), lyap_to_timescale(e))[1]

    def mse_fn(p, e):
        return spectrum_mse(p, e)

    def ts_mse_fn(p, e):
        return spectrum_mse(lyap_to_timescale(p), lyap_to_timescale(e))

    _rows = []
    for _run_id in discovered.run_ids:
        _r = all_results.get(_run_id, {})
        _wm = wandb_val_metrics.get(_run_id, {})

        if "error" in _r:
            _rows.append({"run_id": _run_id, "error": _r.get("error", "unknown")})
            continue

        _cfg = _r.get("config", {})
        _pred_mean = np.array(_r.get("pred_lyap_mean", []))
        _emp_mean = np.array(_r.get("emp_lyap_mean", []))
        _pred_per = _r.get("pred_lyap_per_traj", [])
        _emp_per = _r.get("emp_lyap_per_traj", [])
        _n_le = min(len(_pred_mean), len(_emp_mean))

        # Timescale versions of mean spectra
        _pred_ts_mean = lyap_to_timescale(_pred_mean)
        _emp_ts_mean = lyap_to_timescale(_emp_mean)

        _row = {
            "run_id": _run_id,
            "lambda_lc": _cfg.get("loop_closure_weight"),
            "kl_dyn": _cfg.get("kl_dyn_weight"),
            "traj_val_loss": _wm.get("trajectory_val_loss"),
            "val_lc_loss": _wm.get("val_loop_closure_loss"),
        }

        # --- Exponent-space metrics ---
        # R² of mean spectra
        _, _row["ms_r2_full"] = spectrum_comparison(_pred_mean, _emp_mean)
        # MSE of mean spectra
        _row["ms_mse_full"] = spectrum_mse(_pred_mean, _emp_mean)
        # Per-traj mean R² and MSE
        _row["pt_r2_full"] = _r.get("mean_spectrum_r2")
        _row["pt_mse_full"] = partial_metric_per_traj(_pred_per, _emp_per, _n_le, mse_fn)

        if _n_le >= 5:
            _row["ms_r2_5"] = spectrum_comparison(_pred_mean[:5], _emp_mean[:5])[1]
            _row["ms_mse_5"] = spectrum_mse(_pred_mean[:5], _emp_mean[:5])
            _row["pt_r2_5"] = partial_metric_per_traj(_pred_per, _emp_per, 5, r2_fn)
            _row["pt_mse_5"] = partial_metric_per_traj(_pred_per, _emp_per, 5, mse_fn)
        if _n_le >= 10:
            _row["ms_r2_10"] = spectrum_comparison(_pred_mean[:10], _emp_mean[:10])[1]
            _row["ms_mse_10"] = spectrum_mse(_pred_mean[:10], _emp_mean[:10])
            _row["pt_r2_10"] = partial_metric_per_traj(_pred_per, _emp_per, 10, r2_fn)
            _row["pt_mse_10"] = partial_metric_per_traj(_pred_per, _emp_per, 10, mse_fn)

        # --- Timescale-space metrics ---
        _, _row["ts_ms_r2_full"] = spectrum_comparison(_pred_ts_mean, _emp_ts_mean)
        _row["ts_ms_mse_full"] = spectrum_mse(_pred_ts_mean, _emp_ts_mean)
        _row["ts_pt_r2_full"] = partial_metric_per_traj(_pred_per, _emp_per, _n_le, ts_r2_fn)
        _row["ts_pt_mse_full"] = partial_metric_per_traj(_pred_per, _emp_per, _n_le, ts_mse_fn)

        if _n_le >= 5:
            _row["ts_ms_r2_5"] = spectrum_comparison(_pred_ts_mean[:5], _emp_ts_mean[:5])[1]
            _row["ts_ms_mse_5"] = spectrum_mse(_pred_ts_mean[:5], _emp_ts_mean[:5])
            _row["ts_pt_r2_5"] = partial_metric_per_traj(_pred_per, _emp_per, 5, ts_r2_fn)
            _row["ts_pt_mse_5"] = partial_metric_per_traj(_pred_per, _emp_per, 5, ts_mse_fn)
        if _n_le >= 10:
            _row["ts_ms_r2_10"] = spectrum_comparison(_pred_ts_mean[:10], _emp_ts_mean[:10])[1]
            _row["ts_ms_mse_10"] = spectrum_mse(_pred_ts_mean[:10], _emp_ts_mean[:10])
            _row["ts_pt_r2_10"] = partial_metric_per_traj(_pred_per, _emp_per, 10, ts_r2_fn)
            _row["ts_pt_mse_10"] = partial_metric_per_traj(_pred_per, _emp_per, 10, ts_mse_fn)

        # --- Jacobian metrics ---
        _row["mean_jac_r2"] = _r.get("mean_jac_r2")
        _jac_r2_per = _r.get("jac_r2_per_traj", [])
        if _jac_r2_per:
            _jac_mses = []
            # jac_r2_per_traj stores R² values; compute MSE from raw Jacobians isn't
            # cached, so we store the mean Jacobian MSE from R² via the identity
            # MSE = Var(y_true) * (1 - R²) — but we don't have Var cached either.
            # Instead, note mean_jac_r2 is already the Jacobian R². MSE would require
            # re-loading the model. So we just include R² for Jacobians.
            pass
        _row["mean_jac_mse"] = None  # would need recomputation; not cached

        _row["pred_LE_1"] = float(_pred_mean[0]) if len(_pred_mean) > 0 else None
        _row["emp_LE_1"] = float(_emp_mean[0]) if len(_emp_mean) > 0 else None
        _row["error"] = None
        _rows.append(_row)

    summary_df = pd.DataFrame(_rows)
    mo.ui.table(summary_df, label="Lyapunov Spectrum Summary")
    return (summary_df,)


@app.cell
def _(TRUE_LYAPUNOV, all_results, discovered, mo, np, plot_lyapunov_spectrum):
    # ---- Per-run Lyapunov spectrum plots ----
    _figs = {}
    for _run_id in discovered.run_ids:
        _r = all_results.get(_run_id, {})
        if "error" in _r or "pred_lyap_mean" not in _r:
            continue

        _pred_mean = np.array(_r["pred_lyap_mean"])
        _pred_std = np.array(_r["pred_lyap_std"])
        _emp_mean = np.array(_r["emp_lyap_mean"])
        _emp_std = np.array(_r["emp_lyap_std"])
        _lc = _r.get("config", {}).get("loop_closure_weight")

        _result = plot_lyapunov_spectrum(
            _pred_mean,
            _pred_std,
            _emp_mean,
            _emp_std,
            true_lyapunov=TRUE_LYAPUNOV,
            loop_closure_weight=_lc,
        )
        if isinstance(_result, list):
            _fig = _result[0]
        else:
            _fig = _result
        _fig.suptitle(f"Run {_run_id} (lc={_lc})", fontsize=10, y=1.02)
        _figs[_run_id] = _fig

    lyap_figs = _figs

    if lyap_figs:
        mo.md(f"**{len(lyap_figs)} Lyapunov spectrum plots generated.**")
    else:
        mo.md("No successful runs to plot.")
    return (lyap_figs,)


@app.cell
def _(all_results, discovered, mo, np, plt):
    # ---- Cross-run comparison plots ----
    _lcs = []
    _r2s = []
    _pred_le1s = []
    _emp_le1s = []
    _labels = []

    for _run_id in discovered.run_ids:
        _r = all_results.get(_run_id, {})
        if "error" in _r or "mean_spectrum_r2" not in _r:
            continue
        _cfg = _r.get("config", {})
        _lcs.append(_cfg.get("loop_closure_weight", 0))
        _r2s.append(_r["mean_spectrum_r2"])
        _pred = _r.get("pred_lyap_mean", [])
        _emp = _r.get("emp_lyap_mean", [])
        _pred_le1s.append(_pred[0] if _pred else float("nan"))
        _emp_le1s.append(_emp[0] if _emp else float("nan"))
        _labels.append(_run_id[:8])

    if _lcs:
        cross_fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Largest LE comparison
        _ax1.scatter(_emp_le1s, _pred_le1s, c=np.log10(np.array(_lcs) + 1e-10), cmap="viridis", s=60)
        _lim = [
            min(min(_emp_le1s), min(_pred_le1s)) - 0.1,
            max(max(_emp_le1s), max(_pred_le1s)) + 0.1,
        ]
        _ax1.plot(_lim, _lim, "k--", alpha=0.3, label="y=x")
        _ax1.set_xlabel("Empirical LE_1")
        _ax1.set_ylabel("Predicted LE_1")
        _ax1.set_title("Largest Lyapunov Exponent")
        _ax1.legend()

        # Right: R^2 vs loop_closure_weight
        _ax2.scatter(_lcs, _r2s, s=60, c="steelblue")
        _ax2.set_xscale("symlog", linthresh=1e-7)
        _ax2.set_xlabel("loop_closure_weight")
        _ax2.set_ylabel("Mean Spectrum R^2")
        _ax2.set_title("Spectrum R^2 vs Loop Closure Weight")
        _ax2.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)

        plt.tight_layout()
        mo.as_html(cross_fig)
    else:
        cross_fig = None
        mo.md("No runs to compare.")
    return (cross_fig,)


@app.cell
def _(all_results, mo, np, plt, summary_df):
    # ---- Scatter plots: R²/MSE metrics vs val losses ----
    # Three spaces: exponent R², exponent MSE, timescale R², timescale MSE
    # Three x-axes: traj val loss, LC val loss, traj val loss (feasible only)

    _df = summary_df.dropna(subset=["lambda_lc"]).copy()

    # Add n_target_dims from cached results config
    _df["n_target_dims"] = _df["run_id"].map(
        lambda rid: all_results.get(rid, {}).get("config", {}).get("n_target_dims")
    )

    def collect_available(col_label_pairs, df):
        """Filter to columns that exist and have data."""
        return [(c, l) for c, l in col_label_pairs if c in df.columns and df[c].notna().any()]

    # -- Exponent-space R² columns --
    _exp_r2_cols = collect_available([
        ("pt_r2_5", "Per-Traj R² (top 5)"),
        ("pt_r2_10", "Per-Traj R² (top 10)"),
        ("pt_r2_full", "Per-Traj R² (full)"),
        ("ms_r2_5", "Mean-Spec R² (top 5)"),
        ("ms_r2_10", "Mean-Spec R² (top 10)"),
        ("ms_r2_full", "Mean-Spec R² (full)"),
        ("mean_jac_r2", "Jacobian R²"),
    ], _df)

    # -- Exponent-space MSE columns --
    _exp_mse_cols = collect_available([
        ("pt_mse_5", "Per-Traj MSE (top 5)"),
        ("pt_mse_10", "Per-Traj MSE (top 10)"),
        ("pt_mse_full", "Per-Traj MSE (full)"),
        ("ms_mse_5", "Mean-Spec MSE (top 5)"),
        ("ms_mse_10", "Mean-Spec MSE (top 10)"),
        ("ms_mse_full", "Mean-Spec MSE (full)"),
    ], _df)

    # -- Timescale-space R² columns --
    _ts_r2_cols = collect_available([
        ("ts_pt_r2_5", "TS Per-Traj R² (top 5)"),
        ("ts_pt_r2_10", "TS Per-Traj R² (top 10)"),
        ("ts_pt_r2_full", "TS Per-Traj R² (full)"),
        ("ts_ms_r2_5", "TS Mean-Spec R² (top 5)"),
        ("ts_ms_r2_10", "TS Mean-Spec R² (top 10)"),
        ("ts_ms_r2_full", "TS Mean-Spec R² (full)"),
    ], _df)

    # -- Timescale-space MSE columns --
    _ts_mse_cols = collect_available([
        ("ts_pt_mse_5", "TS Per-Traj MSE (top 5)"),
        ("ts_pt_mse_10", "TS Per-Traj MSE (top 10)"),
        ("ts_pt_mse_full", "TS Per-Traj MSE (full)"),
        ("ts_ms_mse_5", "TS Mean-Spec MSE (top 5)"),
        ("ts_ms_mse_10", "TS Mean-Spec MSE (top 10)"),
        ("ts_ms_mse_full", "TS Mean-Spec MSE (full)"),
    ], _df)

    def make_scatter_grid(df, x_col, x_label, title_prefix, metric_cols,
                          y_label="R²", y_clip=(-15, 1.1), log_x=False, log_y=False):
        """Scatter subplots: each metric vs x_col."""
        _n = len(metric_cols)
        if _n == 0:
            return None
        _ncols = min(4, _n)
        _nrows = (_n + _ncols - 1) // _ncols
        fig, axes = plt.subplots(_nrows, _ncols, figsize=(5 * _ncols, 4 * _nrows), squeeze=False)

        for _idx, (_col, _label) in enumerate(metric_cols):
            _ax = axes[_idx // _ncols][_idx % _ncols]
            _mask = df[_col].notna() & df[x_col].notna()
            _x = df.loc[_mask, x_col].values
            _y = df.loc[_mask, _col].values
            if not log_y:
                _y = np.clip(_y, y_clip[0], y_clip[1])

            _ax.scatter(_x, _y, s=25, alpha=0.7, c="steelblue", edgecolors="k", linewidths=0.3)
            if not log_y:
                _ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)
                _ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
                _ax.set_ylim(y_clip)
            if log_x:
                _ax.set_xscale("log")
            if log_y:
                _ax.set_yscale("log")
            _ax.set_xlabel(x_label, fontsize=9)
            _ax.set_ylabel(y_label, fontsize=9)
            _ax.set_title(_label, fontsize=10)

        for _idx2 in range(_n, _nrows * _ncols):
            axes[_idx2 // _ncols][_idx2 % _ncols].set_visible(False)

        fig.suptitle(title_prefix, fontsize=13, y=1.02)
        plt.tight_layout()
        return fig

    scatter_figs = {}

    if _df.empty:
        mo.md("No data for scatter plots.")
    else:
        # Build list of (metric_group_name, cols, y_label, y_clip, log_y)
        _metric_groups = [
            ("exp_r2", _exp_r2_cols, "Exponent R²", "R²", (-15, 1.1), False),
            ("exp_mse", _exp_mse_cols, "Exponent MSE", "MSE", None, True),
            ("ts_r2", _ts_r2_cols, "Timescale R²", "R²", (-15, 1.1), False),
            ("ts_mse", _ts_mse_cols, "Timescale MSE", "MSE", None, True),
        ]

        # 3 x-axis groups
        _x_groups = []

        # Group 1: all runs vs traj val loss
        _df1 = _df.dropna(subset=["traj_val_loss"])
        if not _df1.empty:
            _x_groups.append(("traj_vloss", _df1, "traj_val_loss", "trajectory val loss", "all runs"))

        # Group 2: all runs vs LC val loss
        _df2 = _df.dropna(subset=["val_lc_loss"])
        if not _df2.empty:
            _x_groups.append(("lc_vloss", _df2, "val_lc_loss", "loop closure val loss", "all runs"))

        # Group 3: feasible runs vs traj val loss
        _df3 = _df.dropna(subset=["traj_val_loss", "val_lc_loss", "n_target_dims"]).copy()
        if not _df3.empty:
            _df3 = _df3[_df3["val_lc_loss"] <= np.sqrt(_df3["n_target_dims"])]
            if not _df3.empty:
                _x_groups.append(("traj_vloss_feasible", _df3, "traj_val_loss", "trajectory val loss",
                                  f"feasible (n={len(_df3)})"))

        for _xkey, _xdf, _xcol, _xlabel, _xsuffix in _x_groups:
            for _mkey, _mcols, _mtitle, _ylabel, _yclip, _logy in _metric_groups:
                if not _mcols:
                    continue
                _fig = make_scatter_grid(
                    _xdf, _xcol, _xlabel,
                    f"{_mtitle} vs {_xlabel} ({_xsuffix})",
                    _mcols, y_label=_ylabel, y_clip=_yclip, log_x=True, log_y=_logy,
                )
                if _fig is not None:
                    scatter_figs[f"{_mkey}_vs_{_xkey}"] = _fig

        mo.md(f"**Generated {len(scatter_figs)} scatter plot grids.**")
    return (scatter_figs,)


@app.cell
def _(
    ENTITY,
    GROUP,
    PROJECT,
    Path,
    all_results,
    cross_fig,
    datetime,
    discovered,
    fig_to_base64,
    lyap_figs,
    mo,
    results_dir,
    scatter_figs,
    summary_df,
):
    # ---- HTML report generation ----

    _CSS = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 24px 32px; background: #fff; color: #222; }
    h1   { font-size: 1.6em; margin-bottom: 0.15em; }
    h2   { font-size: 1.15em; color: #444; border-bottom: 2px solid #e0e0e0;
            padding-bottom: 4px; margin-top: 2.2em; }
    pre  { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
            padding: 12px 16px; font-size: 12.5px; line-height: 1.6;
            overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
    .figure { margin: 18px 0; text-align: center; }
    img  { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }
    .subtitle { color: #666; font-size: 0.9em; margin-bottom: 2em; }
    table { border-collapse: collapse; margin: 12px 0; }
    th, td { border: 1px solid #e0e0e0; padding: 6px 12px; text-align: left; font-size: 13px; }
    th { background: #f6f8fa; }
    """

    def make_html_report(title, subtitle, sections):
        """Build a self-contained HTML report.

        sections: list of (heading, content_html) tuples
        """
        body_parts = []
        for heading, content in sections:
            body_parts.append(f"<h2>{heading}</h2>\n{content}")
        body = "\n".join(body_parts)
        return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <title>{title}</title>
      <style>{_CSS}</style>
    </head>
    <body>
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
    {body}
    </body>
    </html>"""

    def fig_html(fig_or_b64):
        if isinstance(fig_or_b64, str):
            b64 = fig_or_b64
        else:
            b64 = fig_to_base64(fig_or_b64)
        return f'<div class="figure"><img src="data:image/png;base64,{b64}"/></div>'

    def table_html(df):
        return df.to_html(index=False, na_rep="--", float_format=lambda x: f"{x:.4f}")

    # --- Per-run HTML reports ---
    _per_run_dir = results_dir
    for _run_id in discovered.run_ids:
        _r = all_results.get(_run_id, {})
        if "error" in _r:
            continue
        _cfg = _r.get("config", {})
        _sections = []

        # Config section
        _config_lines = "\n".join(f"{k}: {v}" for k, v in _cfg.items())
        _sections.append(("Configuration", f"<pre>{_config_lines}</pre>"))

        # Lyapunov spectrum
        _lyap_lines = []
        _pred_mean = _r.get("pred_lyap_mean", [])
        _pred_std = _r.get("pred_lyap_std", [])
        _emp_mean = _r.get("emp_lyap_mean", [])
        _emp_std = _r.get("emp_lyap_std", [])
        _lyap_lines.append("Predicted (mean +/- std):")
        for _i, (_le, _sd) in enumerate(zip(_pred_mean, _pred_std)):
            _lyap_lines.append(f"  LE_{_i+1} = {_le:+.4f} +/- {_sd:.4f}")
        _lyap_lines.append("Empirical (mean +/- std):")
        for _i, (_le, _sd) in enumerate(zip(_emp_mean, _emp_std)):
            _lyap_lines.append(f"  LE_{_i+1} = {_le:+.4f} +/- {_sd:.4f}")
        _lyap_lines.append(f"Mean R^2:   {_r.get('mean_spectrum_r2', float('nan')):.4f}")
        _lyap_lines.append(f"Mean corr:  {_r.get('mean_spectrum_corr', float('nan')):.4f}")
        _content = f"<pre>{'chr(10)'.join(_lyap_lines)}</pre>"
        # Fix: use actual newlines
        _content = "<pre>" + "\n".join(_lyap_lines) + "</pre>"
        if _run_id in lyap_figs:
            _content += fig_html(fig_to_base64(lyap_figs[_run_id]))
        _sections.append(("Lyapunov Spectrum", _content))

        # Obs-space Jacobian (if computed)
        if _r.get("obs_jac_lyap_mean"):
            _obs_lines = ["Observed-space Jacobian Lyapunov (mean +/- std):"]
            for _i, (_le, _sd) in enumerate(
                zip(_r["obs_jac_lyap_mean"], _r.get("obs_jac_lyap_std", []))
            ):
                _obs_lines.append(f"  LE_{_i+1} = {_le:+.4f} +/- {_sd:.4f}")
            if "mean_jac_r2" in _r:
                _obs_lines.append(f"Mean Jacobian R^2: {_r['mean_jac_r2']:.4f}")
            _sections.append(
                ("Observed-Space Jacobian", "<pre>" + "\n".join(_obs_lines) + "</pre>")
            )

        _html = make_html_report(
            f"Lyapunov Report - {_run_id}",
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Project: {PROJECT} | Group: {GROUP or '(none)'}",
            _sections,
        )
        (_per_run_dir / f"{_run_id}.html").write_text(_html, encoding="utf-8")

    # --- Aggregate HTML report ---
    _repo_root = Path(__file__).resolve().parent.parent
    _report_dir = _repo_root / "_marimo" / "reports"
    _report_dir.mkdir(parents=True, exist_ok=True)
    _timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _agg_path = _report_dir / f"{_timestamp} - Lyapunov_{PROJECT}_{GROUP or 'all'}.html"

    _agg_sections = []

    # Summary table (sorted by full mean-spectrum R² descending)
    _sorted_df = summary_df.sort_values("ms_r2_full", ascending=False, na_position="last")
    _agg_sections.append(("Summary", table_html(_sorted_df)))

    # Cross-run plot
    if cross_fig is not None:
        _agg_sections.append(("Cross-Run Comparison", fig_html(fig_to_base64(cross_fig))))

    # Scatter plots (all generated grids, in sorted key order)
    for _key in sorted(scatter_figs.keys()):
        _fig = scatter_figs[_key]
        if _fig is not None:
            _agg_sections.append((_key.replace("_", " ").title(), fig_html(fig_to_base64(_fig))))

    # Per-run spectra
    for _run_id in discovered.run_ids:
        if _run_id in lyap_figs:
            _r = all_results.get(_run_id, {})
            _r2 = _r.get("mean_spectrum_r2", float("nan"))
            _agg_sections.append(
                (
                    f"Run {_run_id} (R^2={_r2:.4f})",
                    fig_html(fig_to_base64(lyap_figs[_run_id])),
                )
            )

    _agg_html = make_html_report(
        f"Lyapunov Spectrum Analysis - {PROJECT}",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Entity: {ENTITY} | Project: {PROJECT} | Group: {GROUP or '(none)'} | "
        f"Runs: {len(discovered.run_ids)}",
        _agg_sections,
    )
    _agg_path.write_text(_agg_html, encoding="utf-8")

    mo.md(
        f"**Reports saved.**\n\n"
        f"- Per-run HTML: `{_per_run_dir}/`\n"
        f"- Aggregate: `{_agg_path}`"
    )
    return


@app.cell
def _(summary_df):
    summary_df.columns
    return


@app.cell
def _(summary_df):
    summary_df.loc[summary_df['traj_val_loss'].argsort()]
    return


@app.cell
def _(np, plt, summary_df):
    valid_runs = summary_df['val_lc_loss'] <= np.sqrt(0.5)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    thresh = 3
    # ms_r2_5 (already handled by original code for axs[0][0])
    valid_runs_ax0 = valid_runs & ((1 - summary_df['ms_r2_5']) < thresh)
    axs[0][0].scatter(summary_df['traj_val_loss'][valid_runs_ax0], 1 - summary_df['ms_r2_5'][valid_runs_ax0])
    min_traj_val_idx_0 = summary_df['traj_val_loss'][valid_runs_ax0].idxmin()
    axs[0][0].scatter(
        summary_df.loc[min_traj_val_idx_0, 'traj_val_loss'], 
        1 - summary_df.loc[min_traj_val_idx_0, 'ms_r2_5'],
        color='magenta', s=100, label='Min Traj Val Loss', edgecolor='k', zorder=10, alpha=0.7
    )
    min_nmse_idx_0 = (1 - summary_df['ms_r2_5'][valid_runs_ax0]).idxmin()
    axs[0][0].scatter(
        summary_df.loc[min_nmse_idx_0, 'traj_val_loss'], 
        1 - summary_df.loc[min_nmse_idx_0, 'ms_r2_5'],
        color='orange', s=100, label='Min NMSE', edgecolor='k', zorder=10, alpha=0.7
    )
    axs[0][0].set_xscale('log')
    axs[0][0].set_yscale('log')
    axs[0][0].set_xlabel('Trajectory Validation Loss')
    axs[0][0].set_ylabel('Mean Spectrum\nNormalized MSE\n(first 5 exponents)')
    axs[0][0].legend()

    # ms_r2_full
    valid_runs_ax1 = valid_runs & ((1 - summary_df['ms_r2_full']) < thresh)
    axs[0][1].scatter(summary_df['traj_val_loss'][valid_runs_ax1], 1 - summary_df['ms_r2_full'][valid_runs_ax1])
    min_traj_val_idx_1 = summary_df['traj_val_loss'][valid_runs_ax1].idxmin()
    axs[0][1].scatter(
        summary_df.loc[min_traj_val_idx_1, 'traj_val_loss'], 
        1 - summary_df.loc[min_traj_val_idx_1, 'ms_r2_full'], 
        color='magenta', s=100, label='Min Traj Val Loss', edgecolor='k', zorder=10, alpha=0.7
    )
    min_nmse_idx_1 = (1 - summary_df['ms_r2_full'][valid_runs_ax1]).idxmin()
    axs[0][1].scatter(
        summary_df.loc[min_nmse_idx_1, 'traj_val_loss'], 
        1 - summary_df.loc[min_nmse_idx_1, 'ms_r2_full'],
        color='orange', s=100, label='Min NMSE', edgecolor='k', zorder=10, alpha=0.7
    )
    axs[0][1].set_xscale('log')
    axs[0][1].set_yscale('log')
    axs[0][1].set_xlabel('Trajectory Validation Loss')
    axs[0][1].set_ylabel('Mean Spectrum\nNormalized MSE\n(full spectrum)')
    axs[0][1].legend()

    # ts_ms_r2_5
    valid_runs_ax2 = valid_runs & ((1 - summary_df['ts_ms_r2_5']) < thresh)
    axs[1][0].scatter(summary_df['traj_val_loss'][valid_runs_ax2], 1 - summary_df['ts_ms_r2_5'][valid_runs_ax2])
    min_traj_val_idx_2 = summary_df['traj_val_loss'][valid_runs_ax2].idxmin()
    axs[1][0].scatter(
        summary_df.loc[min_traj_val_idx_2, 'traj_val_loss'], 
        1 - summary_df.loc[min_traj_val_idx_2, 'ts_ms_r2_5'], 
        color='magenta', s=100, label='Min Traj Val Loss', edgecolor='k', zorder=10, alpha=0.7
    )
    min_nmse_idx_2 = (1 - summary_df['ts_ms_r2_5'][valid_runs_ax2]).idxmin()
    axs[1][0].scatter(
        summary_df.loc[min_nmse_idx_2, 'traj_val_loss'], 
        1 - summary_df.loc[min_nmse_idx_2, 'ts_ms_r2_5'],
        color='orange', s=100, label='Min NMSE', edgecolor='k', zorder=10, alpha=0.7
    )
    axs[1][0].set_xscale('log')
    axs[1][0].set_yscale('log')
    axs[1][0].set_xlabel('Trajectory Validation Loss')
    axs[1][0].set_ylabel('Time Series Mean Spectrum\nNormalized MSE\n(first 5 exponents)')
    axs[1][0].legend()

    # ts_ms_r2_full
    valid_runs_ax3 = valid_runs & ((1 - summary_df['ts_ms_r2_full']) < thresh)
    axs[1][1].scatter(summary_df['traj_val_loss'][valid_runs_ax3], 1 - summary_df['ts_ms_r2_full'][valid_runs_ax3])
    min_traj_val_idx_3 = summary_df['traj_val_loss'][valid_runs_ax3].idxmin()
    axs[1][1].scatter(
        summary_df.loc[min_traj_val_idx_3, 'traj_val_loss'], 
        1 - summary_df.loc[min_traj_val_idx_3, 'ts_ms_r2_full'], 
        color='magenta', s=100, label='Min Traj Val Loss', edgecolor='k', zorder=10, alpha=0.7
    )
    min_nmse_idx_3 = (1 - summary_df['ts_ms_r2_full'][valid_runs_ax3]).idxmin()
    axs[1][1].scatter(
        summary_df.loc[min_nmse_idx_3, 'traj_val_loss'], 
        1 - summary_df.loc[min_nmse_idx_3, 'ts_ms_r2_full'],
        color='orange', s=100, label='Min NMSE', edgecolor='k', zorder=10, alpha=0.7
    )
    axs[1][1].set_xscale('log')
    axs[1][1].set_yscale('log')
    axs[1][1].set_xlabel('Trajectory Validation Loss')
    axs[1][1].set_ylabel('Time Series Mean Spectrum\nNormalized MSE\n(full spectrum)')
    axs[1][1].legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, plt, summary_df):
    # Cross-metric NMSE scatter (log–log): magenta = argmin x NMSE, orange = argmin y NMSE
    # (Unique names: marimo allows each variable to be defined in only one cell.)
    valid_runs_nmse_cross = summary_df["val_lc_loss"] <= np.sqrt(128)
    fig_nmse_cross, axs_nmse_cross = plt.subplots(2, 2, figsize=(12, 10))
    thresh_nmse_cross = 4
    hl_alpha_nmse_cross = 0.7

    def _nmse_pair(ax, x_col, y_col, x_lbl, y_lbl):
        x_nmse = 1 - summary_df[x_col]
        y_nmse = 1 - summary_df[y_col]
        m = (
            valid_runs_nmse_cross
            & x_nmse.notna()
            & y_nmse.notna()
            & (x_nmse < thresh_nmse_cross)
            & (y_nmse < thresh_nmse_cross)
        )
        ax.scatter(x_nmse[m], y_nmse[m], alpha=0.5)
        if not m.any():
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)
            return
        min_x_idx = x_nmse[m].idxmin()
        min_y_idx = y_nmse[m].idxmin()
        ax.scatter(
            x_nmse.loc[min_x_idx],
            y_nmse.loc[min_x_idx],
            color="magenta",
            s=100,
            label="Min x-axis NMSE",
            edgecolor="k",
            zorder=10,
            alpha=hl_alpha_nmse_cross,
        )
        ax.scatter(
            x_nmse.loc[min_y_idx],
            y_nmse.loc[min_y_idx],
            color="orange",
            s=100,
            label="Min y-axis NMSE",
            edgecolor="k",
            zorder=10,
            alpha=hl_alpha_nmse_cross,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.legend()

    _nmse_pair(
        axs_nmse_cross[0][0],
        "ms_r2_5",
        "ms_r2_full",
        "NMSE (mean spectrum, first 5)",
        "NMSE (mean spectrum, full)",
    )
    _nmse_pair(
        axs_nmse_cross[0][1],
        "ts_ms_r2_5",
        "ts_ms_r2_full",
        "NMSE (TS mean spectrum, first 5)",
        "NMSE (TS mean spectrum, full)",
    )
    _nmse_pair(
        axs_nmse_cross[1][0],
        "ms_r2_5",
        "mean_jac_r2",
        "NMSE (mean spectrum, first 5)",
        "NMSE (mean Jacobian)",
    )
    _nmse_pair(
        axs_nmse_cross[1][1],
        "ms_r2_full",
        "mean_jac_r2",
        "NMSE (mean spectrum, full)",
        "NMSE (mean Jacobian)",
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(summary_df):
    summary_df.loc[summary_df['ts_ms_r2_full'].idxmax()]
    return


@app.cell
def _(summary_df):
    summary_df.loc[summary_df['ts_ms_r2_5'].idxmin()]
    return


@app.cell
def _(mo, summary_df):
    # ---- Identify top 5 runs by ms_r2_5 and ms_r2_full ----
    _valid = summary_df.dropna(subset=["ms_r2_5", "ms_r2_full"]).copy()

    top5_by_r2_5 = _valid.nlargest(5, "ms_r2_5")["run_id"].tolist()
    top5_by_r2_full = _valid.nlargest(5, "ms_r2_full")["run_id"].tolist()

    mo.md(
        f"## Jacobian Consistency Loss Comparison\n\n"
        f"**Top 5 by ms_r2_5 (best first-5 exponents):** {top5_by_r2_5}\n\n"
        f"**Top 5 by ms_r2_full (best full spectrum):** {top5_by_r2_full}\n\n"
        f"**Overlap:** {set(top5_by_r2_5) & set(top5_by_r2_full) or 'none'}"
    )
    return top5_by_r2_5, top5_by_r2_full


@app.cell
def _(
    OmegaConf,
    PROJECT_PATH,
    SAVE_DIR,
    load_run,
    mo,
    np,
    top5_by_r2_5,
    top5_by_r2_full,
    torch,
    z_dyn_slice,
):
    # ---- Load models and compute jac consistency loss on test data ----
    _all_run_ids = list(dict.fromkeys(top5_by_r2_5 + top5_by_r2_full))  # unique, order-preserving
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _data_loaded = False
    _trajs_shared = None
    _dt_shared = None
    _cfg_shared = None

    jac_cons_results = {}

    with mo.status.spinner(title="Computing jac consistency losses...") as _status:
        for _i, _run_id in enumerate(_all_run_ids):
            _status.update(
                title=f"Jac consistency: {_i+1}/{len(_all_run_ids)} ({_run_id})"
            )
            try:
                (
                    _run_obj, _cfg, _eq, _dt_val, _values,
                    _train_dl, _val_dl, _test_dl, _trajs_val, _lit_model,
                ) = load_run(
                    PROJECT_PATH,
                    run_id=_run_id,
                    save_dir=SAVE_DIR,
                    generate_data=(not _data_loaded),
                    verbose=False,
                )

                if not _data_loaded:
                    _trajs_shared = _trajs_val
                    _dt_shared = _dt_val
                    _cfg_shared = _cfg
                    _data_loaded = True

                # Ensure best checkpoint is loaded
                import JacobianODE.jacobians.checkpoints.loader as _ckpt_loader
                try:
                    _ckpt_loader.load_checkpoint(
                        _run_obj, _cfg, _lit_model,
                        save_dir=SAVE_DIR, verbose=False,
                    )
                except Exception:
                    pass

                _lit_model.eval()
                _lit_model = _lit_model.to(_device)

                _n_target_dims = OmegaConf.select(
                    _cfg, "model.n_target_dims", default=None
                )
                if _n_target_dims is not None:
                    _n_target_dims = int(_n_target_dims)

                # Test trajectories
                if "test_trajs_full" in _trajs_shared:
                    _test_seq = _trajs_shared["test_trajs_full"].sequence
                else:
                    _test_seq = _trajs_shared["test_trajs"].sequence

                _traj_t = torch.as_tensor(_test_seq).float().to(_device)

                # Use shared dt (self.dt may be None if not injected during load)
                _dt_run = _lit_model.dt if _lit_model.dt is not None else _dt_shared

                with torch.no_grad():
                    _z_full = _lit_model.encode_trajectory(_traj_t)
                    _z_dyn = z_dyn_slice(_z_full, _n_target_dims)
                    _jacs = _lit_model.compute_jacobians(_z_dyn)

                    # Per-trajectory jac consistency loss (inline to avoid self.dt=None)
                    # ||expm(J*dt) @ dz_t - dz_{t+1}||^2 / var(z)
                    _losses = []
                    for _ti in range(_z_dyn.shape[0]):
                        _z_s = _z_dyn[_ti : _ti + 1]   # (1, T, D)
                        _j_s = _jacs[_ti : _ti + 1]     # (1, T, D, D)
                        _J_exp = torch.matrix_exp(_j_s[:, :-2] * _dt_run)
                        _vel = _z_s[:, 1:] - _z_s[:, :-1]
                        _vel_pred = (_J_exp @ _vel[:, :-1].unsqueeze(-1)).squeeze(-1)
                        _loss = (_vel[:, 1:] - _vel_pred).pow(2).mean() / _z_s.var()
                        _losses.append(_loss.item())

                    jac_cons_results[_run_id] = {
                        "mean": float(np.mean(_losses)),
                        "std": float(np.std(_losses)),
                        "per_traj": _losses,
                    }

                print(
                    f"  [{_i+1}/{len(_all_run_ids)}] {_run_id}: "
                    f"jac_cons = {jac_cons_results[_run_id]['mean']:.6f} "
                    f"+/- {jac_cons_results[_run_id]['std']:.6f}"
                )

                _lit_model.cpu()
                del _lit_model
                torch.cuda.empty_cache()

            except Exception as _e:
                import traceback as _tb
                print(f"  Error for {_run_id}: {_e}\n{_tb.format_exc()}")
                jac_cons_results[_run_id] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "per_traj": [],
                }

    mo.md(f"**Computed jac consistency loss for {len(jac_cons_results)} runs.**")
    return (jac_cons_results,)


@app.cell
def _(
    jac_cons_results,
    mo,
    np,
    pd,
    plt,
    summary_df,
    top5_by_r2_5,
    top5_by_r2_full,
):
    # ---- Compare jac consistency loss: best-first-5 vs best-full-spectrum ----
    _df = summary_df.set_index("run_id")
    _all_ids = list(dict.fromkeys(top5_by_r2_5 + top5_by_r2_full))

    # --- Build comparison table ---
    _rows = []
    for _rid in _all_ids:
        _r = jac_cons_results.get(_rid, {})
        _in_5 = _rid in top5_by_r2_5
        _in_full = _rid in top5_by_r2_full
        _rows.append({
            "run_id": _rid[:8],
            "group": (
                "both" if (_in_5 and _in_full)
                else "best_r2_5" if _in_5
                else "best_r2_full"
            ),
            "ms_r2_5": _df.loc[_rid, "ms_r2_5"] if _rid in _df.index else None,
            "ms_r2_full": _df.loc[_rid, "ms_r2_full"] if _rid in _df.index else None,
            "jac_cons_mean": _r.get("mean"),
            "jac_cons_std": _r.get("std"),
            "lambda_lc": _df.loc[_rid, "lambda_lc"] if _rid in _df.index else None,
            "kl_dyn": _df.loc[_rid, "kl_dyn"] if _rid in _df.index else None,
            "traj_val_loss": _df.loc[_rid, "traj_val_loss"] if _rid in _df.index else None,
        })
    jac_cons_table = pd.DataFrame(_rows).sort_values("group")

    # --- Plots ---
    from matplotlib.patches import Patch

    fig_jc, (ax_box, ax_scatter) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: box plots of per-trajectory jac consistency loss
    data_5 = [
        jac_cons_results.get(rid, {}).get("per_traj", []) for rid in top5_by_r2_5
    ]
    data_full = [
        jac_cons_results.get(rid, {}).get("per_traj", []) for rid in top5_by_r2_full
    ]

    pos_5 = list(range(len(top5_by_r2_5)))
    pos_full = list(range(
        len(top5_by_r2_5) + 1,
        len(top5_by_r2_5) + 1 + len(top5_by_r2_full),
    ))

    if any(d for d in data_5):
        bp5 = ax_box.boxplot(
            data_5, positions=pos_5, widths=0.6, patch_artist=True
        )
        for patch in bp5["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
    if any(d for d in data_full):
        bpf = ax_box.boxplot(
            data_full, positions=pos_full, widths=0.6, patch_artist=True
        )
        for patch in bpf["boxes"]:
            patch.set_facecolor("coral")
            patch.set_alpha(0.7)

    ax_box.set_xticks(pos_5 + pos_full)
    ax_box.set_xticklabels(
        [rid[:6] for rid in top5_by_r2_5]
        + [rid[:6] for rid in top5_by_r2_full],
        fontsize=7, rotation=45,
    )
    ax_box.set_ylabel("Jac Consistency Loss")
    ax_box.set_title("Per-Trajectory Jac Consistency Loss")
    ax_box.set_yscale("log")
    ax_box.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.7, label="Best ms_r2_5 (top-5 exponents)"),
        Patch(facecolor="coral", alpha=0.7, label="Best ms_r2_full (full spectrum)"),
    ], fontsize=8)

    # Right: jac_cons_mean vs full-spectrum NMSE, colored by group
    for _rid in _all_ids:
        _r = jac_cons_results.get(_rid, {})
        _jc = _r.get("mean", float("nan"))
        _nmse_full = (
            1 - _df.loc[_rid, "ms_r2_full"]
            if _rid in _df.index else float("nan")
        )
        _nmse_5 = (
            1 - _df.loc[_rid, "ms_r2_5"]
            if _rid in _df.index else float("nan")
        )
        _in_5 = _rid in top5_by_r2_5
        _in_full = _rid in top5_by_r2_full
        _color = (
            "purple" if (_in_5 and _in_full)
            else "steelblue" if _in_5
            else "coral"
        )
        ax_scatter.scatter(
            _nmse_full, _jc, c=_color, marker="o", s=80,
            alpha=0.8, edgecolors="k", linewidths=0.5,
        )
        # Annotate with short run id
        ax_scatter.annotate(
            _rid[:6], (_nmse_full, _jc),
            fontsize=6, alpha=0.7,
            xytext=(4, 4), textcoords="offset points",
        )

    ax_scatter.set_xlabel("Full Spectrum NMSE (1 - R²)")
    ax_scatter.set_ylabel("Jac Consistency Loss (mean)")
    ax_scatter.set_title("Jac Consistency vs Full Spectrum NMSE")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.legend(handles=[
        Patch(facecolor="steelblue", label="Best ms_r2_5 only"),
        Patch(facecolor="coral", label="Best ms_r2_full only"),
        Patch(facecolor="purple", label="Both"),
    ], fontsize=8)

    plt.tight_layout()

    # Group-level summary
    _g5_means = [
        jac_cons_results.get(rid, {}).get("mean", float("nan"))
        for rid in top5_by_r2_5
    ]
    _gf_means = [
        jac_cons_results.get(rid, {}).get("mean", float("nan"))
        for rid in top5_by_r2_full
    ]

    mo.vstack([
        mo.md(
            f"**Group means:**  "
            f"Best-r2-5 = {np.nanmean(_g5_means):.6f},  "
            f"Best-r2-full = {np.nanmean(_gf_means):.6f}  "
            f"(ratio = {np.nanmean(_g5_means) / np.nanmean(_gf_means):.2f}x)"
        ),
        mo.as_html(fig_jc),
        mo.ui.table(jac_cons_table, label="Jac Consistency Comparison"),
    ])
    return


if __name__ == "__main__":
    app.run()
