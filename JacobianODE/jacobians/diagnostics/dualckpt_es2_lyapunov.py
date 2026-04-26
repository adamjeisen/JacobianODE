"""Lyapunov spectra from the patience-2 sidecar (`es2-best.ckpt`) for a
ShadowPercentEarlyStoppingCheckpoint sweep.

The standard per-run Lyapunov diagnostic loads each run's primary best.ckpt
(saved by Lightning's ModelCheckpoint, picked by best val loss). For dualckpt
sweeps where ShadowPercentEarlyStoppingCheckpoint also writes `es2-best.ckpt`
at the state a patience=2 ES would have triggered, this script computes the
same overlay against the sidecar — useful for "did patience=5 over-train into
a different Lyapunov regime?" comparisons.

Usage:
    python -m JacobianODE.jacobians.diagnostics.dualckpt_es2_lyapunov \\
        --wandb-entity JacobianODE \\
        --wandb-project WMTask_identity_encoder_verification \\
        --group <sweep_group> \\
        --save-dir /orcd/.../latent_jac_runs \\
        --output-dir /orcd/.../diagnostics \\
        --sidecar es2-best.ckpt
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--save-dir", required=True,
                        help="Base dir containing <project>/<run_id>/checkpoints/")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write the .png + .json")
    parser.add_argument("--sidecar", default="es2-best.ckpt",
                        help="Sidecar checkpoint filename to load (default: es2-best.ckpt)")
    parser.add_argument("--n-sample-trajectories", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--swept-key",
                        default="training.lightning.loop_closure_weight",
                        help="Config path used to color per-run curves")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import wandb as _wandb

    from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
    from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, sidecar={args.sidecar}")

    api = _wandb.Api()
    raw_runs = list(api.runs(
        f"{args.wandb_entity}/{args.wandb_project}",
        filters={"group": args.group},
    ))
    ckpt_base = save_dir / args.wandb_project
    runs = []
    for r in raw_runs:
        ckpt_dir = ckpt_base / r.id / "checkpoints"
        if (ckpt_dir / args.sidecar).is_file():
            runs.append(r)
        else:
            logger.warning(f"  skipping {r.id}: missing {args.sidecar} in {ckpt_dir}")
    logger.info(f"Found {len(runs)} run(s) with {args.sidecar} (out of {len(raw_runs)})")
    if not runs:
        logger.error("No runs with sidecar checkpoint; nothing to do.")
        return 1

    per_run: dict[str, dict] = {}
    swept_vals: dict[str, float | str | None] = {}

    for i, run in enumerate(runs):
        rid = run.id
        try:
            loaded = load_run(
                f"{args.wandb_entity}/{args.wandb_project}",
                run_id=rid, save_dir=str(save_dir),
                generate_data=True, verbose=False, return_full_obs=False,
            )
            run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded

            load_checkpoint(
                run_obj, cfg, lit_model, save_dir=str(save_dir),
                checkpoint_filename=args.sidecar, verbose=False,
            )
            lit_model = lit_model.to(device).eval()

            if trajs is None or "test_trajs" not in trajs:
                raise RuntimeError(f"No trajectories from load_run for {rid}")
            seq = trajs.get("train_trajs", trajs["test_trajs"]).sequence
            test_trajs_this = seq[: args.n_sample_trajectories].to(device)

            is_vanilla = not hasattr(lit_model, "encode_trajectory")
            lambdas = []
            with torch.no_grad():
                for start in range(0, test_trajs_this.shape[0], args.chunk_size):
                    chunk = test_trajs_this[start : start + args.chunk_size]
                    if is_vanilla:
                        jacs = lit_model.compute_jacobians(chunk)
                    else:
                        z_full = lit_model.encode_trajectory(chunk)
                        mu_dyn, _ = lit_model._split_latent(z_full)
                        jacs = lit_model.compute_jacobians(mu_dyn)
                        del z_full, mu_dyn
                    lams = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt)
                    lambdas.append(lams.detach().cpu())
                    del jacs

            lambda_per_traj = torch.cat(lambdas, dim=0).numpy()
            lambda_mean = lambda_per_traj.mean(axis=0)
            cum = np.cumsum(lambda_mean)
            k_star = int(np.searchsorted(-cum, 0))
            ky = (float(k_star + cum[k_star - 1] / max(abs(lambda_mean[k_star]), 1e-12))
                  if 0 < k_star < len(lambda_mean) else None)

            cfg_dict = dict(run.config)
            swept = cfg_dict
            for part in args.swept_key.split("."):
                swept = swept.get(part, {}) if isinstance(swept, dict) else None
            swept_vals[rid] = swept if not isinstance(swept, dict) else None

            per_run[rid] = {
                "lambda_spectrum": lambda_mean.tolist(),
                "lambda_max": float(lambda_mean.max()),
                "lambda_sum": float(lambda_mean.sum()),
                "kaplan_yorke_dim": ky,
                "swept_value": swept_vals[rid],
                "error": None,
            }
            logger.info(
                f"  [{i + 1}/{len(runs)}] {rid}  "
                f"{args.swept_key.split('.')[-1]}={swept_vals[rid]}  "
                f"λ_max={per_run[rid]['lambda_max']:.4f}  Σλ={per_run[rid]['lambda_sum']:.2f}"
            )
        except Exception as e:
            per_run[rid] = {"error": f"{type(e).__name__}: {e}"}
            logger.warning(f"  [{i + 1}/{len(runs)}] {rid} FAILED: {e}")
        finally:
            try:
                del lit_model
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Plot ----
    success = [(rid, d) for rid, d in per_run.items() if d.get("error") is None]
    if not success:
        logger.error("No successful runs; skipping plot.")
        (output_dir / f"{args.group}_es2_lyapunov.json").write_text(
            json.dumps({"sidecar": args.sidecar, "per_run": per_run}, indent=2)
        )
        return 1

    # Color by swept value (numeric → log-norm; else categorical)
    swept_numeric = []
    for rid, d in success:
        v = d.get("swept_value")
        try:
            swept_numeric.append(float(v))
        except (TypeError, ValueError):
            swept_numeric.append(None)
    use_numeric = all(v is not None for v in swept_numeric)
    if use_numeric:
        positives = [v for v in swept_numeric if v > 0]
        vmin = min(positives) if positives else 1e-12
        vmax = max(positives) if positives else 1.0
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (rid, d), v in zip(success, swept_numeric):
        spec = np.asarray(d["lambda_spectrum"])
        if use_numeric and v is not None and v > 0:
            color = cmap(norm(v))
        else:
            color = "C0"
        ax.plot(spec, "o-", color=color, lw=1.0, ms=3, alpha=0.85)
    ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("index (sorted desc)")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title(
        f"Lyapunov spectra from {args.sidecar} ({len(success)} runs)\n"
        f"group: {args.group}"
    )
    ax.grid(True, alpha=0.3)
    if use_numeric:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(args.swept_key.split(".")[-1])
    fig.tight_layout()

    fig_path = output_dir / f"{args.group}_es2_lyapunov.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    logger.info(f"Saved figure → {fig_path}")

    json_path = output_dir / f"{args.group}_es2_lyapunov.json"
    json_path.write_text(json.dumps({
        "sidecar": args.sidecar,
        "group": args.group,
        "swept_key": args.swept_key,
        "per_run": per_run,
    }, indent=2))
    logger.info(f"Saved data → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
