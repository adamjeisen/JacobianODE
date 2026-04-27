"""∂L/∂J_f at the dynamics-MLP output, per area-block, per loss type.

For each (group, stage ∈ {init, trained}, loss_type ∈ {obs, latent}):

1. Set up the model. ``stage='trained'`` loads the chosen-best ckpt;
   ``stage='init'`` rebuilds the encoder from cfg with the original training
   seed (the dynamics MLP itself isn't used for this diagnostic, only
   indirectly — see step 2 — but for "init" it remains the freshly-built
   one so that the J_f outputs reflect random-init MLP behavior).
2. Monkey-patch ``lit_model.compute_jacobians`` to capture the produced
   J_f tensors (the actual MLP output, reshaped to (..., D, D)).
3. Call ``lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0.5)``
   — the same code path training uses, so ``loss`` is the obs-space rollout
   loss and ``metric_vals['latent_pred_loss']`` is the latent-space loss.
4. ``∂loss/∂J_f`` via ``torch.autograd.grad(loss, captured)``. Concatenate
   across calls, take abs-mean over (batch, time). Result: a (D, D) matrix
   of per-J-entry gradient magnitudes. Split into 4 area-blocks.

The point: this is the gradient the optimizer would apply to whatever
makes up J_f — it tells us how strongly each block of J_f is being
"pulled" by the loss. If DirectSum's block-diagonal encoder/decoder
zero out the cross-area gradient channels, we'd see vc/cv blocks
strictly smaller than monolithic.

Usage:
    python -m JacobianODE.jacobians.diagnostics.dynamics_gradient_signal \\
        --wandb-entity JacobianODE \\
        --wandb-project WMTask_identity_encoder_verification \\
        --groups <g_directsum>,<g_mono> \\
        --save-dir /orcd/.../latent_jac_runs \\
        --reports-dir /orcd/.../jacobian-reports \\
        --output-dir /orcd/.../diagnostics \\
        --area-split 64
"""
from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _block_norms(M, split):
    return {
        "vv": float(M[:split, :split].abs().mean().item()),
        "vc": float(M[:split, split:].abs().mean().item()),
        "cv": float(M[split:, :split].abs().mean().item()),
        "cc": float(M[split:, split:].abs().mean().item()),
        "vv_max": float(M[:split, :split].abs().max().item()),
        "vc_max": float(M[:split, split:].abs().max().item()),
        "cv_max": float(M[split:, :split].abs().max().item()),
        "cc_max": float(M[split:, split:].abs().max().item()),
    }


def _instantiate_encoder(cfg, n_input, instantiate_fn):
    """Build a fresh encoder, passing n_input only if the encoder class accepts it."""
    target = cfg.model.encoder.get("_target_", "")
    if target:
        mod_path, cls_name = target.rsplit(".", 1)
        encoder_cls = getattr(importlib.import_module(mod_path), cls_name)
        sig = inspect.signature(encoder_cls.__init__)
        if "n_input" in sig.parameters:
            return instantiate_fn(cfg.model.encoder, n_input=n_input)
    return instantiate_fn(cfg.model.encoder)


def _measure_grad_on_J(lit_model, batch, alpha):
    """Returns ((D, D) obs-grad, (D, D) latent-grad) — abs-mean per entry."""
    import torch

    captured: list = []
    orig = lit_model.compute_jacobians

    def patched(*a, **kw):
        j = orig(*a, **kw)
        captured.append(j)
        return j

    lit_model.compute_jacobians = patched
    try:
        # Use the same training code path. alpha=0.5 is partial teacher forcing.
        result = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=alpha)
    finally:
        lit_model.compute_jacobians = orig

    obs_loss = result["loss"]
    lat_loss = result["metric_vals"]["latent_pred_loss"]

    g_obs = torch.autograd.grad(
        obs_loss, captured, retain_graph=True, allow_unused=True,
    )
    g_lat = torch.autograd.grad(
        lat_loss, captured, retain_graph=False, allow_unused=True,
    )

    def agg(grads):
        # Each g has shape (..., D, D). Stack last-2 dims, abs-mean over the rest.
        flat = []
        for g in grads:
            if g is None:
                continue
            flat.append(g.reshape(-1, g.shape[-2], g.shape[-1]).abs())
        if not flat:
            return None
        cat = torch.cat(flat, dim=0)
        return cat.mean(dim=0)  # (D, D)

    return agg(g_obs), agg(g_lat)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--area-split", type=int, default=64)
    parser.add_argument("--n-trajectories", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Teacher-forcing alpha for trajectory_model_step (0..1)")
    parser.add_argument("--stages", default="init,trained")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from hydra.utils import instantiate

    from JacobianODE.jacobians.checkpoints.loader import load_run
    from JacobianODE.jacobians.core.reproducibility import seed_everything

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)
    reports_dir = Path(args.reports_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    logger.info(f"groups={groups}  stages={stages}  alpha={args.alpha}  device={device}")

    rows: list[dict] = []

    for group in groups:
        mp = reports_dir / args.wandb_project / group / "metrics.json"
        if not mp.is_file():
            logger.warning(f"  no metrics.json at {mp}; skipping {group}")
            continue
        chosen = json.loads(mp.read_text()).get("metrics_summary", {}).get("overall_chosen_run") or {}
        run_id = chosen.get("run_id")
        if not run_id:
            logger.warning(f"  no overall_chosen_run for {group}; skipping")
            continue
        logger.info(f"=== {group}  chosen={run_id} ===")

        for stage in stages:
            try:
                loaded = load_run(
                    f"{args.wandb_entity}/{args.wandb_project}",
                    run_id=run_id, save_dir=str(save_dir),
                    generate_data=True, verbose=False, return_full_obs=False,
                )
                run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded
                # load_run loads ckpt by default; for "init" rebuild encoder
                if stage == "init":
                    seed = (cfg.data.flow.random_state
                            + cfg.training.run_number + 1)
                    seed_everything(seed)
                    n_input = trajs["train_trajs"].sequence.shape[-1]
                    fresh_encoder = _instantiate_encoder(cfg, n_input, instantiate)
                    lit_model.encoder = fresh_encoder
                lit_model = lit_model.to(device).train()  # train mode for full graph
                # Make sure ALL params allow grad (we don't actually update them,
                # but the autograd graph needs them).
                for p in lit_model.parameters():
                    p.requires_grad_(True)

                seq = trajs.get("train_trajs", trajs["test_trajs"]).sequence
                batch = seq[: args.n_trajectories].to(device).float()

                G_obs, G_lat = _measure_grad_on_J(lit_model, batch, alpha=args.alpha)
                if G_obs is None and G_lat is None:
                    logger.warning(f"  stage={stage}: no captured Jacobians")
                    continue

                bn_obs = _block_norms(G_obs, args.area_split) if G_obs is not None else None
                bn_lat = _block_norms(G_lat, args.area_split) if G_lat is not None else None
                if bn_obs is not None:
                    rows.append(dict(group=group, run_id=run_id, stage=stage,
                                     loss_type="obs", block_norms=bn_obs,
                                     n_dyn=int(G_obs.shape[-1])))
                    logger.info(
                        f"  {stage} obs:    "
                        f"vv={bn_obs['vv']:.3e}  vc={bn_obs['vc']:.3e}  "
                        f"cv={bn_obs['cv']:.3e}  cc={bn_obs['cc']:.3e}  "
                        f"(max: vv={bn_obs['vv_max']:.2e}, cc={bn_obs['cc_max']:.2e})"
                    )
                if bn_lat is not None:
                    rows.append(dict(group=group, run_id=run_id, stage=stage,
                                     loss_type="latent", block_norms=bn_lat,
                                     n_dyn=int(G_lat.shape[-1])))
                    logger.info(
                        f"  {stage} latent: "
                        f"vv={bn_lat['vv']:.3e}  vc={bn_lat['vc']:.3e}  "
                        f"cv={bn_lat['cv']:.3e}  cc={bn_lat['cc']:.3e}"
                    )
            except Exception as e:
                logger.exception(f"  {stage} FAILED for {group}: {e}")
            finally:
                try:
                    del lit_model
                except NameError:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not rows:
        logger.error("No results.")
        return 1

    # Plot: 2 stages x 2 losses, bars per block per group
    n_rows = len(stages); n_cols = 2
    loss_types = ["obs", "latent"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.0 * n_rows),
                              squeeze=False, sharey=False)
    blocks = ["vv", "vc", "cv", "cc"]
    block_labels = ["vis→vis", "cog→vis", "vis→cog", "cog→cog"]
    cmap = plt.colormaps["tab10"]
    for r_i, stage in enumerate(stages):
        for c_i, loss_type in enumerate(loss_types):
            ax = axes[r_i][c_i]
            cell = [r for r in rows if r["stage"] == stage and r["loss_type"] == loss_type]
            if not cell:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{stage} · {loss_type}", fontsize=10)
                continue
            x = np.arange(len(blocks))
            width = 0.8 / max(len(cell), 1)
            for g_i, r in enumerate(cell):
                vals = [r["block_norms"][b] for b in blocks]
                offset = (g_i - (len(cell) - 1) / 2) * width
                lbl = r["group"][:30] + ("…" if len(r["group"]) > 30 else "")
                ax.bar(x + offset, vals, width, color=cmap(g_i % 10), label=lbl)
            ax.set_xticks(x)
            ax.set_xticklabels(block_labels, fontsize=9)
            ax.set_yscale("log")
            ax.set_ylabel(r"mean $|\partial L / \partial J_f|$")
            ax.set_title(f"stage = {stage}  ·  loss = {loss_type}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3, which="both")
            if r_i == 0 and c_i == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"∂L/∂J_f (dynamics-MLP output) per area-block — measured via "
        f"trajectory_model_step (α={args.alpha})",
        y=1.01,
    )
    fig.tight_layout()

    tag = args.tag or "_".join(g[:24] for g in groups)
    base = f"dynamics_gradient_signal_{tag}"
    fig_path = output_dir / f"{base}.png"
    json_path = output_dir / f"{base}.json"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    json_path.write_text(json.dumps(rows, indent=2))
    logger.info(f"Saved → {fig_path}")
    logger.info(f"Saved → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
