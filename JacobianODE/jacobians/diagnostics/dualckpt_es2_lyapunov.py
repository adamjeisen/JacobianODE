"""Lyapunov spectra from the patience-2 sidecar (`es2-best.ckpt`) for a
ShadowPercentEarlyStoppingCheckpoint sweep.

Per-run subplot grid: each panel shows one run's predicted Lyapunov spectrum
(blue, from es2-best.ckpt) overlaid on the empirical ground-truth spectrum
(black, from eq.jac on the raw trajectories). Title carries run_idx, run_id,
swept hyperparameters, and best traj_loss for direct comparison across the
sweep.

Usage:
    python -m JacobianODE.jacobians.diagnostics.dualckpt_es2_lyapunov \\
        --wandb-entity JacobianODE \\
        --wandb-project WMTask_identity_encoder_verification \\
        --group <sweep_group> \\
        --save-dir /orcd/.../latent_jac_runs \\
        --output-dir /orcd/.../diagnostics \\
        --sidecar es2-best.ckpt \\
        [--sentinel /orcd/.../sweeps/{done,processed,failed}/<group>.done.json]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def _nested_get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


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
    parser.add_argument("--group", required=True)
    parser.add_argument("--save-dir", required=True,
                        help="Base dir containing <project>/<run_id>/checkpoints/")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write the .png + .json")
    parser.add_argument("--sentinel", default=None,
                        help="Path to <group>.done.json sentinel (for run_idx + "
                             "swept-key auto-detection). If omitted, the script "
                             "searches /orcd/.../sweeps/{done,processed,failed}/.")
    parser.add_argument("--sweeps-dir", default="/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps",
                        help="Used only when --sentinel is omitted.")
    parser.add_argument("--sidecar", default="es2-best.ckpt",
                        help="Sidecar checkpoint filename (default: es2-best.ckpt)")
    parser.add_argument("--n-sample-trajectories", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--swept-keys", default=None,
                        help="Comma-separated config paths to display in titles. "
                             "If omitted, derived from sentinel's overrides_template.")
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--loss-key", default="best_traj_loss",
                        help="wandb summary key to display in subplot titles")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import wandb as _wandb

    from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
    from JacobianODE.jacobians.tuning.analyze_sweep import _compute_empirical_lyapunov
    from JacobianODE.jacobians.tuning.monitor import match_run_to_idx
    from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)

    # ---- Locate sentinel for run_idx + swept-key resolution ----
    sentinel_path: Path | None = None
    if args.sentinel:
        sentinel_path = Path(args.sentinel)
    else:
        for sub in ("done", "processed", "failed"):
            cand = Path(args.sweeps_dir) / sub / f"{args.group}.done.json"
            if cand.is_file():
                sentinel_path = cand
                break
    expected_snapshot: dict = {}
    if sentinel_path and sentinel_path.is_file():
        sentinel_doc = json.loads(sentinel_path.read_text())
        expected_snapshot = sentinel_doc.get("expected_snapshot", {})
        logger.info(f"sentinel: {sentinel_path}")
    else:
        logger.warning("No sentinel found; run_idx + auto swept-key disabled.")

    # ---- Determine swept keys (CLI > sentinel.overrides_template > default) ----
    if args.swept_keys:
        swept_keys = [s.strip() for s in args.swept_keys.split(",") if s.strip()]
    else:
        swept_keys = []
        tpl = expected_snapshot.get("hydra", {}).get("overrides_template", []) or []
        for ov in tpl:
            if "=" not in ov:
                continue
            k, v = ov.split("=", 1)
            if "," in v and k not in ("experiment",) and not k.startswith("hydra."):
                swept_keys.append(k)
        if not swept_keys:
            swept_keys = ["training.lightning.loop_closure_weight"]
    logger.info(f"swept_keys: {swept_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, sidecar={args.sidecar}")

    # ---- List runs ----
    api = _wandb.Api()
    raw_runs = list(api.runs(
        f"{args.wandb_entity}/{args.wandb_project}",
        filters={"group": args.group},
    ))
    ckpt_base = save_dir / args.wandb_project
    runs = []
    for r in raw_runs:
        if (ckpt_base / r.id / "checkpoints" / args.sidecar).is_file():
            runs.append(r)
        else:
            logger.warning(f"  skipping {r.id}: missing {args.sidecar}")
    logger.info(f"Found {len(runs)} run(s) with {args.sidecar} (out of {len(raw_runs)})")
    if not runs:
        logger.error("No runs with sidecar checkpoint; nothing to do.")
        return 1

    # ---- Compute predicted spectra + cache empirical from first run ----
    per_run: dict[str, dict] = {}
    empirical_mean: np.ndarray | None = None

    resolved_runs = expected_snapshot.get("hydra", {}).get("resolved_runs", []) or []

    for i, run in enumerate(runs):
        rid = run.id
        try:
            loaded = load_run(
                f"{args.wandb_entity}/{args.wandb_project}",
                run_id=rid, save_dir=str(save_dir),
                generate_data=True, verbose=False,
                return_full_obs=(empirical_mean is None),
            )
            run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded

            if empirical_mean is None and eq is not None and trajs is not None:
                mu_val = cfg.data.postprocessing.get("mu", 0.0)
                sigma_val = cfg.data.postprocessing.get("sigma", 1.0)
                if "train_trajs_full" in trajs:
                    traj_for_emp = trajs["train_trajs_full"].sequence
                elif values is not None:
                    traj_for_emp = values
                else:
                    traj_for_emp = (trajs.get("test_trajs_full")
                                    or trajs["test_trajs"]).sequence
                logger.info("Computing empirical Lyapunov spectrum (one-time)...")
                em_mean, _ = _compute_empirical_lyapunov(
                    eq, traj_for_emp, dt, mu_val, sigma_val, device,
                )
                if em_mean is not None:
                    empirical_mean = em_mean
                    logger.info(f"  empirical: λ₁={em_mean[0]:.4f}  λ_min={em_mean[-1]:.4f}  "
                                f"Σλ={em_mean.sum():.3f}  D={len(em_mean)}")

            load_checkpoint(
                run_obj, cfg, lit_model, save_dir=str(save_dir),
                checkpoint_filename=args.sidecar, verbose=False,
            )
            lit_model = lit_model.to(device).eval()

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

            cfg_dict = dict(run.config)
            run_idx = match_run_to_idx(cfg_dict, resolved_runs) if resolved_runs else None
            swept_vals = {k: _nested_get(cfg_dict, k) for k in swept_keys}
            traj_loss = run.summary.get(args.loss_key)

            per_run[rid] = {
                "run_idx": run_idx,
                "lambda_spectrum": lambda_mean.tolist(),
                "lambda_max": float(lambda_mean.max()),
                "lambda_sum": float(lambda_mean.sum()),
                "swept_values": swept_vals,
                "traj_loss": float(traj_loss) if traj_loss is not None else None,
                "error": None,
            }
            logger.info(
                f"  [{i + 1}/{len(runs)}] {rid}  idx={run_idx}  "
                + "  ".join(f"{k.split('.')[-1]}={_fmt_param_value(v)}"
                            for k, v in swept_vals.items())
                + f"  λ_max={per_run[rid]['lambda_max']:.4f}"
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

    # ---- Plot per-run grid ----
    success = [(rid, d) for rid, d in per_run.items() if d.get("error") is None]
    if not success:
        logger.error("No successful runs; saving JSON only.")
        (output_dir / f"{args.group}_es2_lyapunov.json").write_text(
            json.dumps({"sidecar": args.sidecar, "per_run": per_run}, indent=2)
        )
        return 1

    # Sort by run_idx (None last)
    success.sort(key=lambda rd: (rd[1]["run_idx"] is None, rd[1]["run_idx"] or 0))
    n = len(success)
    ncols = max(1, args.ncols)
    nrows = math.ceil(n / ncols)

    # Y-range: union of empirical + all predicted, for honest comparison
    all_vals: list[float] = []
    if empirical_mean is not None:
        all_vals.extend(float(v) for v in empirical_mean)
    for _, d in success:
        all_vals.extend(d["lambda_spectrum"])
    ymin = math.floor(min(all_vals) - 1)
    ymax = math.ceil(max(all_vals) + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                              sharex=True, sharey=True, squeeze=False)
    for ax_i, (rid, d) in enumerate(success):
        ax = axes[ax_i // ncols][ax_i % ncols]
        spec = np.asarray(d["lambda_spectrum"])
        x = np.arange(len(spec))
        if empirical_mean is not None:
            ax.plot(np.arange(len(empirical_mean)), empirical_mean,
                    "k-", lw=1.0, label="empirical")
        ax.plot(x, spec, "o-", color="C0", lw=1.0, ms=3, label="pred")
        ax.axhline(0, color="k", ls=":", lw=0.5, alpha=0.5)
        ax.set_ylim(ymin, ymax)

        title_bits = [f"idx={d['run_idx']}" if d['run_idx'] is not None else f"idx=?"]
        title_bits.append(rid)
        for k, v in d["swept_values"].items():
            title_bits.append(f"{k.split('.')[-1]}={_fmt_param_value(v)}")
        line1 = "  ".join(title_bits)
        line2 = (f"traj_loss={d['traj_loss']:.4f}"
                 if d['traj_loss'] is not None else "traj_loss=?")
        ax.set_title(f"{line1}\n{line2}", fontsize=8)
        ax.grid(True, alpha=0.25)
        if ax_i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide trailing empty axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"Lyapunov spectra from {args.sidecar}  ·  {n} runs  ·  group: {args.group}",
        y=1.0,
    )
    fig.tight_layout()

    fig_path = output_dir / f"{args.group}_es2_lyapunov_per_run.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    logger.info(f"Saved figure → {fig_path}")

    json_path = output_dir / f"{args.group}_es2_lyapunov.json"
    json_path.write_text(json.dumps({
        "sidecar": args.sidecar,
        "group": args.group,
        "swept_keys": swept_keys,
        "empirical_lyapunov": (empirical_mean.tolist()
                               if empirical_mean is not None else None),
        "per_run": per_run,
    }, indent=2))
    logger.info(f"Saved data → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
