"""Compute and save Lyapunov-spectrum raw arrays for the best run of a sweep.

Outputs a single ``lyapunov.npz`` with three named arrays (each shape
``(n_samples, n_lyaps)``) plus metadata. ``plot.py`` consumes this file
to render the paper figure.

Selection: ``best_traj_loss`` ranking via
:func:`JacobianODE.jacobians.tuning.selection.select_best_model`, which
applies C1 (one-step MASE) + C2 (loop closure norm) as hard filters
before picking the lowest trajectory val_loss.

The three series, matching the existing report's Lyapunov bar chart:

* ``pred_batch_burnin`` — predicted spectrum from model-integrated rollout
  on ``n_batch`` sampled windowed trajectories, each prepended with
  ``burn_in_steps`` zero-padded burn-in and dropping the first
  ``burn_in_drop`` of those when computing exponents.
* ``pred_full``        — predicted spectrum from the *model's* Jacobian
  evaluated at TRUE test-trajectory points (one spectrum per full-length
  test trajectory).
* ``empirical_full``   — spectrum from the analytical / ground-truth
  Jacobian (``eq.jac``) at the same true test-trajectory points. Only
  populated when the system has a known analytical J (Lorenz, RNN, etc.);
  omitted from the npz otherwise.

Usage::

    python -m JacobianODE.figures.lyapunov.eval \\
        --group <group_name> \\
        --out ~/Documents/paper-figures/<group>/lyapunov/

GPU is used if available. For Lorenz-scale systems (D=3) Pascal is fine.
"""
from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger("figures.lyapunov.eval")


# ---------------------------------------------------------------------------
# Run selection (best_traj_loss with C1 + C2 filters)
# ---------------------------------------------------------------------------
def _cfg_get(cfg: dict, dotted: str, default=None):
    """OmegaConf-free dotted-key getter for wandb config dicts."""
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def find_best_run_id(group: str, project: str) -> str:
    """Return the wandb run-id of the best run in ``group`` via
    ``ranking_method='best_traj_loss'`` with C1 + C2 hard filters
    (one-step MASE + loop closure norm).
    """
    from JacobianODE.jacobians.tuning.criteria import diagnostics_from_wandb
    from JacobianODE.jacobians.tuning.selection import select_best_model
    import wandb

    api = wandb.Api(timeout=90)
    runs = list(api.runs(f"JacobianODE/{project}", filters={"group": group}))
    if not runs:
        raise SystemExit(f"no wandb runs found in group {group!r}")

    # n_dims (state-space) and n_latent are required by select_best_model
    # for the C2 (loop closure) sqrt(n) threshold. Read from the first
    # run's config — these are sweep-invariant so any run will do.
    cfg0 = runs[0].config
    n_dims = _cfg_get(cfg0, "model.encoder.n_input")
    n_latent = _cfg_get(cfg0, "model.n_target_dims")
    if n_dims is None:
        raise SystemExit(
            f"could not infer n_dims from runs[0].config for group {group!r}; "
            f"set --n-dims explicitly"
        )

    diagnostics = [diagnostics_from_wandb(r) for r in runs]
    sel = select_best_model(
        diagnostics,
        n_dims=int(n_dims),
        loop_closure_n_dims=int(n_latent) if n_latent is not None else None,
        ranking_method="best_traj_loss",
        use_loop_closure=True,
    )
    if sel.best_index is None:
        raise SystemExit(
            f"select_best_model returned no chosen run for group {group!r}. "
            f"Exclusions: {sel.exclusion_details}"
        )
    chosen = runs[sel.best_index]
    chosen_metrics = diagnostics[sel.best_index]
    logger.info(
        f"chosen run: id={chosen.id} name={chosen.name!r} "
        f"traj_val_loss={chosen_metrics.trajectory_val_loss:.4g}  "
        f"(n_dims={n_dims}, n_latent={n_latent})"
    )
    return chosen.id


# ---------------------------------------------------------------------------
# Lyapunov computations
# ---------------------------------------------------------------------------
def compute_pred_batch_burnin(
    lit_model, z_for_jac: torch.Tensor, dt: float, *,
    n_batch: int = 128, burn_in_steps: int = 400, burn_in_drop: int = 100,
    device: torch.device, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Predicted Lyapunov via model-integrated rollout with burn-in.

    ``z_for_jac`` is the full set of latent-space windowed trajectories
    (shape ``(N, T, D)``). We sample ``n_batch`` of them, prepend
    ``burn_in_steps`` zeros so the integrator has a "warm-up" stretch
    that's discarded later (``burn_in_drop`` initial Jacobians dropped
    before QR averaging), then compute Lyapunov per-trajectory.

    Returns a CPU tensor ``(n_batch, D)``.
    """
    from JacobianODE.jacobians.jacobianODE import JacobianODEint
    from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

    if generator is None:
        generator = torch.Generator().manual_seed(0)

    N, T_true, D = z_for_jac.shape
    n_take = min(n_batch, N)
    perm = torch.randperm(N, generator=generator)[:n_take]
    z_sampled = z_for_jac[perm].to(device)

    # Pad with zeros for burn-in (the integrator computes Jacobians on this
    # full padded sequence, then we drop the first burn_in_drop)
    z_padded = torch.cat(
        [z_sampled, torch.zeros(n_take, burn_in_steps, D, device=device)],
        dim=1,
    )
    jacobian_odeint = JacobianODEint(lit_model.compute_jacobians, dt)
    with torch.no_grad():
        z_combined = jacobian_odeint.generate_dynamics(z_padded)
    # Compute Jacobians at every step of the (padded + predicted) sequence,
    # then drop the first burn_in_drop before QR. Matches what run_analytics
    # does internally for the batch+burn-in path.
    jacs = lit_model.compute_jacobians(z_combined[:, burn_in_drop:])
    lyap = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt)
    return lyap.cpu()


def compute_pred_full(
    lit_model, z_for_jac_full: torch.Tensor, dt: float, *,
    device: torch.device, chunk_size: int = 64,
) -> torch.Tensor:
    """Predicted Lyapunov: model J at TRUE test-trajectory points.

    Returns ``(n_traj, D)``.
    """
    from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

    n_full = z_for_jac_full.shape[0]
    out: list[torch.Tensor] = []
    for ci in tqdm(range(0, n_full, chunk_size), desc="pred-full Lyap"):
        chunk = z_for_jac_full[ci:ci + chunk_size].to(device)
        with torch.no_grad():
            jacs = lit_model.compute_jacobians(chunk)
        le = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt)
        out.append(le.cpu())
    return torch.cat(out, dim=0)


def compute_empirical_full(
    eq: Any, traj_raw: np.ndarray, dt: float, *,
    device: torch.device, chunk_size: int = 64,
) -> torch.Tensor | None:
    """Empirical (true J) Lyapunov along the same true test trajectories.

    ``traj_raw`` is the un-normalized data the equation expects. Tries
    both the torch and numpy branches of ``eq.jac`` — uses the torch
    branch when ``eq`` has a ``.model`` attribute (e.g. RNN-based dyn).

    Returns ``(n_traj, D)`` or ``None`` if ``eq`` doesn't support jac.
    """
    from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

    if eq is None or not hasattr(eq, "jac"):
        return None

    if hasattr(eq, "model"):
        # Torch path (RNN-like): jac broadcasts over batch + time
        sub_t = torch.as_tensor(traj_raw).float().to(device)
        n_sub = sub_t.shape[0]
        out: list[torch.Tensor] = []
        for ci in tqdm(range(0, n_sub, chunk_size), desc="empirical Lyap"):
            chunk = sub_t[ci:ci + chunk_size]
            jacs = eq.jac(chunk, t=0)
            le = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt)
            out.append(le.cpu())
        return torch.cat(out, dim=0)

    # NumPy path (dysts equations like Lorenz): per-trajectory loop
    out_np: list[np.ndarray] = []
    for i in tqdm(range(traj_raw.shape[0]), desc="empirical Lyap (np)"):
        traj_i = traj_raw[i]
        try:
            jacs_np = eq.jac(traj_i, t=0)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"eq.jac failed on trajectory {i}: {e}")
            return None
        jacs_t = torch.as_tensor(jacs_np).float()
        le = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_t, dt)
        out_np.append(le.numpy())
    return torch.as_tensor(np.stack(out_np))


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------
def _z_dyn(z: torch.Tensor, n_target_dims: int | None) -> torch.Tensor:
    if n_target_dims is None or n_target_dims >= z.shape[-1]:
        return z
    return z[..., :n_target_dims]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--group", required=True, help="wandb group name")
    ap.add_argument("--project", default="Lorenz_INDall_N1_D1_NormTrue_T3__JacobianODE",
                    help="wandb project name (under JacobianODE entity)")
    ap.add_argument("--run-id", default=None,
                    help="Explicit run id; defaults to best_traj_loss selection")
    ap.add_argument("--save-dir", default=None,
                    help="Override save_dir for load_run (where checkpoints live)")
    ap.add_argument("--n-batch", type=int, default=128)
    ap.add_argument("--burnin", type=int, default=400)
    ap.add_argument("--burnin-drop", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--skip-empirical", action="store_true",
                    help="Skip the empirical (true J) series even if eq.jac is available")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # 1. Select best run if not specified
    run_id = args.run_id or find_best_run_id(args.group, args.project)

    # 2. Load run (cfg, eq, dt, trajs, lit_model) + checkpoint
    from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
    run, cfg, eq, dt, values, train_dl, val_dl, test_dl, trajs, lit_model = load_run(
        args.project, run_id=run_id, save_dir=args.save_dir, verbose=True,
    )
    load_checkpoint(run, cfg, lit_model, save_dir=args.save_dir, verbose=True)
    lit_model = lit_model.to(device).eval()

    # 3. Determine latent vs non-latent + n_target_dims
    is_latent = hasattr(lit_model, "encoder") and lit_model.encoder is not None
    n_target_dims = getattr(lit_model, "n_target_dims", None)

    # 4. Build full-length test trajectories in z_for_jac space
    traj_seq_full = torch.as_tensor(trajs["test_trajs"].sequence).float().to(device)
    with torch.no_grad():
        z_seq_full = lit_model.encode_trajectory(traj_seq_full) if is_latent else traj_seq_full
    z_for_jac_full = _z_dyn(z_seq_full, n_target_dims)

    # 5. Build batched windowed trajectories (for the batch+burnin path)
    traj_batched_t = torch.as_tensor(test_dl.dataset.sequence).float().to(device)
    with torch.no_grad():
        if is_latent:
            _chunks = []
            for ci in range(0, traj_batched_t.shape[0], 64):
                _chunks.append(lit_model.encode_trajectory(traj_batched_t[ci:ci + 64]))
            z_batched_t = torch.cat(_chunks, dim=0)
        else:
            z_batched_t = traj_batched_t
    z_for_jac = _z_dyn(z_batched_t, n_target_dims)

    # 6. Compute the three series
    logger.info(f"pred_batch_burnin: n_batch={args.n_batch} burnin={args.burnin} drop={args.burnin_drop}")
    pred_batch = compute_pred_batch_burnin(
        lit_model, z_for_jac, dt,
        n_batch=args.n_batch, burn_in_steps=args.burnin, burn_in_drop=args.burnin_drop,
        device=device,
    )
    logger.info(f"  done: shape={tuple(pred_batch.shape)}")

    logger.info(f"pred_full: n_traj={z_for_jac_full.shape[0]} T={z_for_jac_full.shape[1]}")
    pred_full = compute_pred_full(lit_model, z_for_jac_full, dt, device=device)
    logger.info(f"  done: shape={tuple(pred_full.shape)}")

    empirical_full = None
    if not args.skip_empirical:
        # Empirical uses un-normalized data the equation expects
        mu = cfg.data.postprocessing.mu
        sigma = cfg.data.postprocessing.sigma
        if "test_trajs_full" in trajs:
            traj_full_np = trajs["test_trajs_full"].sequence
        else:
            traj_full_np = trajs["test_trajs"].sequence
        traj_raw = np.asarray(traj_full_np) * sigma + mu
        logger.info(f"empirical_full: n_traj={traj_raw.shape[0]} T={traj_raw.shape[1]}")
        empirical_full = compute_empirical_full(eq, traj_raw, dt, device=device)
        if empirical_full is not None:
            logger.info(f"  done: shape={tuple(empirical_full.shape)}")
        else:
            logger.info("  empirical_full unavailable for this eq; omitting")

    # 7. Save
    npz_path = args.out / "lyapunov.npz"
    to_save: dict[str, np.ndarray | str | float] = {
        "pred_batch_burnin": pred_batch.detach().cpu().numpy(),
        "pred_full": pred_full.detach().cpu().numpy(),
        "group": args.group,
        "run_id": run_id,
        "n_lyaps": pred_full.shape[-1],
        "dt": float(dt) if dt is not None else float("nan"),
        "n_batch": args.n_batch,
        "burnin_steps": args.burnin,
        "burnin_drop": args.burnin_drop,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if empirical_full is not None:
        to_save["empirical_full"] = empirical_full.detach().cpu().numpy()
    # Hyperparameters from config — useful for the plot title
    try:
        to_save["loop_closure_weight"] = float(cfg.training.lightning.loop_closure_weight)
    except Exception:
        pass
    try:
        to_save["obs_noise_scale"] = float(cfg.training.lightning.obs_noise_scale)
    except Exception:
        pass

    np.savez(npz_path, **to_save)
    logger.info(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
