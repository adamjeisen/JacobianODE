"""Side-by-side dynamics-MLP block-Jacobian comparison across groups.

For each --groups entry, picks the chosen-best run from the sweep's
analysis metrics.json, loads its primary best.ckpt, computes the dynamics
MLP Jacobian ``J_dyn(z) = ∂f_dyn/∂z`` along a sample of test trajectories
(chunked, vmap'd jacrev/jacfwd), then averages |J_dyn| over (batch, time)
to produce a per-group ``D × D`` mean-|Jacobian| heatmap. Splits each
into 4 area-blocks (vis-vis, vis-cog, cog-vis, cog-cog) and reports the
block Frobenius and mean-|.| norms.

Plot:
    Row per group, 3 columns:
      [0] full mean-|J_dyn| heatmap with area boundary lines.
      [1] block-norm bar chart (Frobenius and mean-|.|, both shown).
      [2] same heatmap, log-scaled colour, for off-diagonal visibility.

Usage:
    python -m JacobianODE.jacobians.diagnostics.block_jacobian_compare \\
        --wandb-entity JacobianODE \\
        --wandb-project WMTask_identity_encoder_verification \\
        --groups <g1>,<g2> \\
        --save-dir /orcd/.../latent_jac_runs \\
        --reports-dir /orcd/.../jacobian-reports \\
        --output-dir /orcd/.../diagnostics \\
        --area-split 64    # WMTask: visual = 0..63, cognitive = 64..127
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
    parser.add_argument("--groups", required=True,
                        help="Comma-separated wandb group names")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-ids", default=None,
                        help="Comma-separated explicit run ids (overrides metrics.json lookup)")
    parser.add_argument("--area-split", type=int, default=64,
                        help="Index where block 1 (visual) ends and block 2 (cognitive) starts")
    parser.add_argument("--n-trajectories", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--tag", default=None)
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
    reports_dir = Path(args.reports_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    overrides = ([s.strip() for s in args.run_ids.split(",")]
                 if args.run_ids else [None] * len(groups))
    if len(overrides) != len(groups):
        raise ValueError("--run-ids must have one entry per --groups entry")

    results: list[dict] = []

    for group, run_override in zip(groups, overrides):
        logger.info(f"=== {group} ===")
        if run_override:
            best_id = run_override
            traj_loss = None
        else:
            mp = reports_dir / args.wandb_project / group / "metrics.json"
            if not mp.is_file():
                logger.warning(f"  no metrics.json at {mp}; skipping")
                continue
            metrics = json.loads(mp.read_text())
            chosen = metrics.get("metrics_summary", {}).get("overall_chosen_run") or {}
            best_id = chosen.get("run_id")
            traj_loss = chosen.get("best_traj_loss")
            if not best_id:
                logger.warning(f"  no overall_chosen_run; skipping")
                continue
            logger.info(f"  chosen: {best_id}  best_traj_loss={traj_loss}")

        try:
            loaded = load_run(
                f"{args.wandb_entity}/{args.wandb_project}",
                run_id=best_id, save_dir=str(save_dir),
                generate_data=True, verbose=False, return_full_obs=False,
            )
            run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded
            load_checkpoint(run_obj, cfg, lit_model, save_dir=str(save_dir),
                            verbose=False)
            lit_model = lit_model.to(device).eval()

            seq = trajs.get("train_trajs", trajs["test_trajs"]).sequence
            test_batch = seq[: args.n_trajectories].to(device)

            # Encode (or pass through if no encoder) to get z trajectories.
            if hasattr(lit_model, "encode_trajectory"):
                with torch.no_grad():
                    z_full = lit_model.encode_trajectory(test_batch)
                    if hasattr(lit_model, "_split_latent"):
                        z_dyn, _ = lit_model._split_latent(z_full)
                    else:
                        z_dyn = z_full
            else:
                z_dyn = test_batch  # vanilla: dynamics on x directly

            D = z_dyn.shape[-1]
            logger.info(f"  z_dyn shape: {tuple(z_dyn.shape)}, D={D}")

            jacs_acc = torch.zeros(D, D, device=device, dtype=torch.float64)
            n_added = 0
            with torch.no_grad():
                for start in range(0, z_dyn.shape[0], args.chunk_size):
                    chunk = z_dyn[start : start + args.chunk_size]  # (b, T, D)
                    j = lit_model.compute_jacobians(chunk)  # (b, T, D, D)
                    j_abs = j.abs().to(torch.float64)
                    jacs_acc += j_abs.reshape(-1, D, D).sum(dim=0)
                    n_added += j_abs.shape[0] * j_abs.shape[1]
                    del j, j_abs
            mean_abs_J = (jacs_acc / max(n_added, 1)).cpu().numpy()  # (D, D)
            logger.info(f"  averaged over {n_added} (batch*time) Jacobians")

            split = args.area_split
            assert 0 < split < D, f"--area-split {split} not in (0, {D})"
            blocks = {
                "vv": mean_abs_J[:split, :split],
                "vc": mean_abs_J[:split, split:],
                "cv": mean_abs_J[split:, :split],
                "cc": mean_abs_J[split:, split:],
            }
            block_stats = {}
            for name, b in blocks.items():
                block_stats[name] = {
                    "mean_abs": float(b.mean()),
                    "frobenius": float(np.linalg.norm(b)),
                    "shape": list(b.shape),
                }
            results.append({
                "group": group,
                "run_id": best_id,
                "best_traj_loss": traj_loss,
                "D": int(D),
                "split": split,
                "mean_abs_J": mean_abs_J,           # (D, D) numpy
                "blocks": block_stats,
            })
            logger.info(
                f"  block mean|J|:  "
                f"vv={block_stats['vv']['mean_abs']:.3e}  "
                f"vc={block_stats['vc']['mean_abs']:.3e}  "
                f"cv={block_stats['cv']['mean_abs']:.3e}  "
                f"cc={block_stats['cc']['mean_abs']:.3e}"
            )
        except Exception as e:
            logger.exception(f"  FAILED for {best_id}: {e}")
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

    # ---- Plot ----
    n_g = len(results)
    fig, axes = plt.subplots(n_g, 3, figsize=(13, 4.0 * n_g),
                              gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]},
                              squeeze=False)
    # Shared colour scale across both heatmap columns and across groups
    # so absolute magnitudes are visually comparable.
    vmax_lin = max(r["mean_abs_J"].max() for r in results)
    vmax_log = vmax_lin
    vmin_log = max(min(r["mean_abs_J"][r["mean_abs_J"] > 0].min() for r in results), 1e-8)

    for i, r in enumerate(results):
        ax_lin, ax_bar, ax_log = axes[i]
        D = r["D"]; split = r["split"]; M = r["mean_abs_J"]
        title_label = f"{r['group'][:48]}…" if len(r["group"]) > 48 else r["group"]
        title_label += f"  ·  {r['run_id']}"
        if r["best_traj_loss"] is not None:
            title_label += f"  ·  traj={r['best_traj_loss']:.4f}"

        # [0] linear heatmap
        im = ax_lin.imshow(M, cmap="viridis", aspect="equal",
                           vmin=0, vmax=vmax_lin)
        ax_lin.axhline(split - 0.5, color="white", lw=0.7, alpha=0.8)
        ax_lin.axvline(split - 0.5, color="white", lw=0.7, alpha=0.8)
        ax_lin.set_xlabel("source dim (input)")
        ax_lin.set_ylabel("target dim (output)")
        ax_lin.set_title(f"{title_label}\nmean|J_dyn|  (linear)", fontsize=9)
        plt.colorbar(im, ax=ax_lin, fraction=0.046, pad=0.04)

        # [1] block-norm bar chart
        names = ["vv", "vc", "cv", "cc"]
        mean_vals = [r["blocks"][n]["mean_abs"] for n in names]
        frob_vals = [r["blocks"][n]["frobenius"] for n in names]
        x = np.arange(4)
        ax_bar.bar(x - 0.2, mean_vals, 0.4, label="mean |J|", color="C0")
        ax_bar2 = ax_bar.twinx()
        ax_bar2.bar(x + 0.2, frob_vals, 0.4, label="Frobenius", color="C1")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(["vis→vis", "cog→vis\n(vc)", "vis→cog\n(cv)", "cog→cog"], fontsize=8)
        ax_bar.set_ylabel("mean|J|", color="C0")
        ax_bar2.set_ylabel("Frobenius", color="C1")
        ax_bar.tick_params(axis='y', labelcolor='C0')
        ax_bar2.tick_params(axis='y', labelcolor='C1')
        ax_bar.set_title("block norms", fontsize=9)
        ax_bar.grid(True, axis="y", alpha=0.3)

        # [2] log-scaled heatmap (better for spotting tiny off-diagonal blocks)
        from matplotlib.colors import LogNorm
        M_safe = np.maximum(M, vmin_log)
        im2 = ax_log.imshow(M_safe, cmap="viridis", aspect="equal",
                            norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
        ax_log.axhline(split - 0.5, color="white", lw=0.7, alpha=0.8)
        ax_log.axvline(split - 0.5, color="white", lw=0.7, alpha=0.8)
        ax_log.set_xlabel("source dim (input)")
        ax_log.set_title(f"mean|J_dyn|  (log scale)", fontsize=9)
        plt.colorbar(im2, ax=ax_log, fraction=0.046, pad=0.04)

    fig.suptitle("Dynamics-MLP block-Jacobian comparison", y=1.0)
    fig.tight_layout()

    tag = args.tag or "_".join(r["group"][:24] for r in results)
    base = f"block_jacobian_{tag}"
    fig_path = output_dir / f"{base}.png"
    json_path = output_dir / f"{base}.json"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    # Strip the heavy mean_abs_J array from JSON
    json_payload = []
    for r in results:
        json_payload.append({
            "group": r["group"], "run_id": r["run_id"],
            "best_traj_loss": r["best_traj_loss"],
            "D": r["D"], "split": r["split"],
            "blocks": r["blocks"],
        })
    json_path.write_text(json.dumps(json_payload, indent=2))
    logger.info(f"Saved → {fig_path}")
    logger.info(f"Saved → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
