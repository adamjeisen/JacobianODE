"""Per-dim noise propagation + encoder-Jacobian SVD diagnostic.

For each (wandb_entity, project, group) tuple, picks the best run (lowest
``best_traj_loss``), loads its primary best.ckpt, and reports two metrics:

(1) **Per-dim noise amplification.** ``a_d = std(z_noisy[:,d] - z_clean[:,d]) / σ``
    where σ is the training obs_noise (added iid in obs space). For a
    volume-preserving encoder, geometric-mean(a_d) ≈ 1; if some dims have
    a_d ≪ 1 while others are ≫ 1, the encoder is routing noise away from
    the "quiet" dims (which the dynamics MLP then over-fits as a flow
    that's too contractive in z → suppressed Lyapunov negative tail).

(2) **Encoder Jacobian SVD.** SVD of ``∂z/∂x`` along a sample of test
    points. Reports singular-value distribution (median + 5/95 percentiles
    across the sample) and the per-point condition number κ = σ_max/σ_min.
    Volume preservation pins prod(σ) = 1, but spread can be wild — large κ
    means heavy local metric distortion, which decouples z-space dynamics
    from x-space dynamics and lets the dynamics MLP fit a different (often
    over-attractive) Lyapunov spectrum than what the data implies.

Both diagnostics overlay multiple groups on the same axes for direct
comparison (e.g. DirectSum vs. vanilla full coupling encoder).

Usage:
    python -m JacobianODE.jacobians.diagnostics.encoder_noise_propagation \\
        --wandb-entity JacobianODE \\
        --wandb-project WMTask_identity_encoder_verification \\
        --groups <group1>,<group2> \\
        --save-dir /orcd/.../latent_jac_runs \\
        --output-dir /orcd/.../diagnostics
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _fmt_param_value(v):
    if v is None:
        return "?"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == 0:
        return "0"
    if abs(f) < 1e-2 or abs(f) >= 1e3:
        return f"{f:.0e}"
    return f"{f:g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--groups", required=True,
                        help="Comma-separated wandb group names")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-dir", required=True,
                        help="Path to the jacobian-reports repo root (used to "
                             "look up <project>/<group>/metrics.json for the "
                             "chosen-best run id; fallback if --run-ids omitted)")
    parser.add_argument("--run-ids", default=None,
                        help="Comma-separated explicit run ids to use, one per "
                             "group (overrides metrics.json lookup)")
    parser.add_argument("--n-trajectories", type=int, default=16,
                        help="Test trajectories used for both diagnostics")
    parser.add_argument("--n-jac-samples", type=int, default=512,
                        help="Random sample of (B*T) points for the SVD pass")
    parser.add_argument("--noise-multiplier", type=float, default=1.0,
                        help="Scale the training obs_noise by this (default 1x)")
    parser.add_argument("--tag", default=None,
                        help="Optional output filename suffix")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    explicit_run_ids = (
        [s.strip() for s in args.run_ids.split(",")] if args.run_ids else [None] * len(groups)
    )
    if len(explicit_run_ids) != len(groups):
        raise ValueError("--run-ids must have one entry per --groups entry")
    reports_dir = Path(args.reports_dir)
    logger.info(f"groups: {groups}")

    results: list[dict] = []  # per-group dict

    for group, run_id_override in zip(groups, explicit_run_ids):
        logger.info(f"=== {group} ===")

        # Resolve best run: explicit override > metrics.json overall_chosen_run
        if run_id_override:
            best_run_id = run_id_override
            loss_val = None
            logger.info(f"  using explicit run_id={best_run_id}")
        else:
            metrics_path = reports_dir / args.wandb_project / group / "metrics.json"
            if not metrics_path.is_file():
                logger.warning(f"  metrics.json not found at {metrics_path}; skipping")
                continue
            metrics = json.loads(metrics_path.read_text())
            chosen = metrics.get("metrics_summary", {}).get("overall_chosen_run") or {}
            best_run_id = chosen.get("run_id")
            loss_val = chosen.get("best_traj_loss")
            if not best_run_id:
                logger.warning(f"  no overall_chosen_run in {metrics_path}; skipping")
                continue
            logger.info(f"  chosen run: {best_run_id}  best_traj_loss={loss_val}")

        ckpt_base = save_dir / args.wandb_project
        if not (ckpt_base / best_run_id / "checkpoints").is_dir():
            logger.warning(f"  no checkpoint dir for {best_run_id}; skipping")
            continue

        try:
            loaded = load_run(
                f"{args.wandb_entity}/{args.wandb_project}",
                run_id=best_run_id, save_dir=str(save_dir),
                generate_data=True, verbose=False, return_full_obs=False,
            )
            run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded
            load_checkpoint(run_obj, cfg, lit_model, save_dir=str(save_dir),
                            verbose=False)
            lit_model = lit_model.to(device).eval()

            if not hasattr(lit_model, "encoder"):
                logger.warning(f"  {best_run_id} has no encoder (vanilla MLP?); skipping")
                continue

            sigma = float(cfg.data.postprocessing.get("obs_noise", 0.0))
            sigma_eff = sigma * args.noise_multiplier
            if sigma_eff <= 0:
                # Fall back to a small probe noise so the diagnostic still
                # produces something even for clean-trained runs.
                sigma_eff = 0.01
                logger.info(f"  obs_noise=0; using probe σ={sigma_eff}")

            seq = trajs.get("train_trajs", trajs["test_trajs"]).sequence
            x_clean = seq[: args.n_trajectories].to(device).float()  # (B, T, D)
            torch.manual_seed(0)
            noise = sigma_eff * torch.randn_like(x_clean)

            # ---- (1) per-dim noise propagation ----
            with torch.no_grad():
                z_clean = lit_model.encoder.encode(x_clean)
                z_noisy = lit_model.encoder.encode(x_clean + noise)
            z_diff = (z_noisy - z_clean).reshape(-1, z_clean.shape[-1])
            x_diff = noise.reshape(-1, x_clean.shape[-1])
            z_std_per_dim = z_diff.std(dim=0).cpu().numpy()
            x_std_per_dim = x_diff.std(dim=0).cpu().numpy()
            x_std_mean = float(x_std_per_dim.mean())
            amp_per_dim = z_std_per_dim / max(x_std_mean, 1e-12)
            amp_sorted = np.sort(amp_per_dim)[::-1]
            log_geom_mean = float(np.exp(np.log(np.maximum(amp_per_dim, 1e-30)).mean()))

            # ---- (2) encoder Jacobian SVD ----
            x_flat = x_clean.reshape(-1, x_clean.shape[-1])
            n_obs = x_flat.shape[-1]
            n_dyn = getattr(lit_model, "n_target_dims", None) or n_obs
            n_jac = min(args.n_jac_samples, x_flat.shape[0])
            sample_idx = torch.randperm(x_flat.shape[0], device=device)[:n_jac]
            x_sample = x_flat[sample_idx]
            J = lit_model._encoder_jacobian_at(x_sample, n_dyn, n_obs)  # (M, n_dyn, n_obs)
            with torch.no_grad():
                sv = torch.linalg.svdvals(J)  # (M, min(n_dyn, n_obs))
            sv_np = sv.cpu().numpy()
            cond = sv_np[:, 0] / np.maximum(sv_np[:, -1], 1e-30)
            sv_med = np.median(sv_np, axis=0)
            sv_lo = np.percentile(sv_np, 5, axis=0)
            sv_hi = np.percentile(sv_np, 95, axis=0)
            log_det = np.log(np.maximum(sv_np, 1e-30)).sum(axis=-1)  # (M,)

            results.append({
                "group": group,
                "run_id": best_run_id,
                "best_traj_loss": loss_val,
                "sigma_obs": sigma,
                "sigma_eff": sigma_eff,
                "n_obs": n_obs,
                "n_dyn": n_dyn,
                "amp_per_dim_sorted": amp_sorted.tolist(),
                "amp_geom_mean": log_geom_mean,
                "x_noise_std_mean": x_std_mean,
                "z_noise_std_min": float(z_std_per_dim.min()),
                "z_noise_std_max": float(z_std_per_dim.max()),
                "sv_median": sv_med.tolist(),
                "sv_p05": sv_lo.tolist(),
                "sv_p95": sv_hi.tolist(),
                "cond_median": float(np.median(cond)),
                "cond_p95": float(np.percentile(cond, 95)),
                "log_det_median": float(np.median(log_det)),
                "log_det_p95": float(np.percentile(log_det, 95)),
            })
            logger.info(
                f"    a_d range: [{amp_sorted[-1]:.3g}, {amp_sorted[0]:.3g}]  "
                f"geo-mean={log_geom_mean:.3f}  "
                f"κ_median={np.median(cond):.2f}  "
                f"log|J|_median={np.median(log_det):+.3f}"
            )
        except Exception as e:
            logger.exception(f"  FAILED for {best_run_id}: {e}")
        finally:
            try:
                del lit_model
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        logger.error("No groups produced results.")
        return 1

    # ---- Plot: 2 cols (noise amp, SVD spectrum), one row across all groups ----
    fig, (ax_amp, ax_sv) = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = plt.colormaps["tab10"]

    for i, r in enumerate(results):
        color = cmap(i % 10)
        label = f"{r['group'][:48]}…" if len(r["group"]) > 48 else r["group"]
        label += f"\n{r['run_id']}  σ={r['sigma_eff']:.3g}"

        x = np.arange(len(r["amp_per_dim_sorted"]))
        ax_amp.plot(x, r["amp_per_dim_sorted"], "o-", color=color, ms=2.5, lw=1.0,
                    label=label, alpha=0.85)

        sv_x = np.arange(len(r["sv_median"]))
        ax_sv.fill_between(sv_x, r["sv_p05"], r["sv_p95"],
                           color=color, alpha=0.15)
        ax_sv.plot(sv_x, r["sv_median"], "-", color=color, lw=1.4,
                   label=f"{r['group'][:32]}…  κ̃={r['cond_median']:.2g}"
                         if len(r['group']) > 32 else
                         f"{r['group']}  κ̃={r['cond_median']:.2g}")

    ax_amp.axhline(1.0, color="k", ls="--", lw=0.7, alpha=0.5,
                   label="VP geometric-mean target = 1")
    ax_amp.set_yscale("log")
    ax_amp.set_xlabel("z-dim (sorted by noise amplification, desc)")
    ax_amp.set_ylabel(r"$\mathrm{std}(\Delta z_d)\,/\,\sigma$")
    ax_amp.set_title("Per-dim noise propagation through encoder")
    ax_amp.grid(True, alpha=0.3, which="both")
    ax_amp.legend(fontsize=7, loc="best")

    ax_sv.set_yscale("log")
    ax_sv.set_xlabel("singular-value index")
    ax_sv.set_ylabel(r"$\sigma_i(J_{enc})$  (median across sample)")
    ax_sv.set_title("Encoder Jacobian singular-value spectrum (5/95% band)")
    ax_sv.grid(True, alpha=0.3, which="both")
    ax_sv.legend(fontsize=7, loc="best")

    fig.suptitle("Encoder noise + Jacobian SVD diagnostic", y=1.01)
    fig.tight_layout()

    tag = args.tag or "_".join(r["group"][:24] for r in results)
    base = f"encoder_noise_svd_{tag}"
    fig_path = output_dir / f"{base}.png"
    json_path = output_dir / f"{base}.json"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    json_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved → {fig_path}")
    logger.info(f"Saved → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
