"""Gradient signal arriving at the dynamics MLP, per area-block, per loss type.

For each (group, stage ∈ {init, trained}, loss_type ∈ {obs, latent}):
  - Load the model (init = fresh from factory, trained = chosen-best ckpt).
  - Sample a batch of test trajectories x[t], x[t+1].
  - Treat z' = E(x[t+1])[..., :n_dyn] as a leaf variable (the "target" of the
    dynamics MLP), backprop the chosen loss into z', and use the outer-product
    decomposition

        ∂L / ∂J_f[i, j]  ≈  E_{batch, time}[ (∂L/∂z'[..., i]) * z[..., j] ]

    to get the (D × D) "gradient-on-effective-Jacobian" — i.e. the signal the
    optimizer would push the dynamics MLP's effective J towards if it could
    perturb J directly. Block-split this matrix into vv / vc / cv / cc and
    report mean |.| per block.

This isolates the encoder/decoder gradient-flow from the MLP architecture
itself: the MLP isn't actually run during the diagnostic. Only the
encoder/decoder shape the gradient-on-J pattern.

Why "init" is uninformative: with zero_init=true the additive coupling
encoder is exact identity at init, so J_E = J_D = I and the per-block
gradient-on-J is identical across DirectSum and monolithic. Differences
only emerge after the encoder trains away from identity.

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
    }


def _compute_grad_on_J(lit_model, x_input, x_target, *, loss_type, n_dyn):
    """Returns (D × D) ∂L/∂J_f via outer-product decomposition.

    x_input  : (B, T, D_obs)  observations at t
    x_target : (B, T, D_obs)  observations at t+1
    loss_type: 'obs' (decode then MSE in obs space) or 'latent' (MSE in z space)
    n_dyn    : dyn-subspace size (so we can pad with the actual encoded null)
    """
    import torch

    device = x_input.device
    has_enc = hasattr(lit_model, "encoder") and lit_model.encoder is not None

    # --- Encode the input to get z (and its null part for obs-loss padding) ---
    if has_enc:
        with torch.no_grad():
            z_full_in = lit_model.encoder.encode(x_input)
            # z_full = [z_dyn || z_null] with dyn first.
            z_in_dyn = z_full_in[..., :n_dyn]
            z_in_null = z_full_in[..., n_dyn:]
    else:
        z_in_dyn = x_input
        z_in_null = None

    # --- Treat z' (the MLP output) as a leaf, equal to the encoded target ---
    if has_enc:
        with torch.no_grad():
            z_full_tgt = lit_model.encoder.encode(x_target)
            z_tgt_dyn = z_full_tgt[..., :n_dyn]
            z_tgt_null = z_full_tgt[..., n_dyn:]
    else:
        z_tgt_dyn = x_target
        z_tgt_null = None

    z_prime = z_tgt_dyn.detach().clone().requires_grad_(True)

    # --- Build the loss ---
    if loss_type == "latent":
        # MSE between MLP output and encoded target (no decoder).
        loss = ((z_prime - z_tgt_dyn) ** 2).mean()
    elif loss_type == "obs":
        if not has_enc:
            # Vanilla: no decoder, "obs loss" == MSE in obs space directly.
            loss = ((z_prime - x_target) ** 2).mean()
        else:
            # Pad z' with the actual encoded null subspace, then decode.
            if z_tgt_null is not None and z_tgt_null.shape[-1] > 0:
                z_full_pred = torch.cat([z_prime, z_tgt_null], dim=-1)
            else:
                z_full_pred = z_prime
            x_pred = lit_model.encoder.decode(z_full_pred)
            loss = ((x_pred - x_target) ** 2).mean()
    else:
        raise ValueError(loss_type)

    # NOTE: z_prime ≡ z_tgt_dyn at the linearization point, so ∂L/∂z' is
    # the gradient *direction*, not a residual. For obs loss this isolates
    # the decoder pull-back; for latent it's exactly zero (z_prime − z_tgt = 0).
    # That's a feature of the diagnostic: latent loss has trivially zero
    # gradient at the target, so we offset z_prime slightly so the gradient
    # lives along a representative direction. Done by re-anchoring at z_in_dyn:
    # what would the gradient be if we predicted z_in_dyn instead of z_tgt_dyn?
    # That makes z_prime − z_tgt = z_in − z_tgt (= the per-step latent
    # increment), which is the realistic prediction error.

    # Override: re-do with z_prime = z_in_dyn so loss has nontrivial gradient.
    z_prime = z_in_dyn.detach().clone().requires_grad_(True)
    if loss_type == "latent":
        loss = ((z_prime - z_tgt_dyn) ** 2).mean()
    elif loss_type == "obs":
        if not has_enc:
            loss = ((z_prime - x_target) ** 2).mean()
        else:
            if z_in_null is not None and z_in_null.shape[-1] > 0:
                z_full_pred = torch.cat([z_prime, z_in_null], dim=-1)
            else:
                z_full_pred = z_prime
            x_pred = lit_model.encoder.decode(z_full_pred)
            loss = ((x_pred - x_target) ** 2).mean()

    grad_zp = torch.autograd.grad(loss, z_prime)[0]  # (B, T, D)

    # Outer-product into (D × D): grad-on-J[i, j] = E[ grad_zp[..., i] * z_in_dyn[..., j] ]
    B, T, D = z_in_dyn.shape
    flat_g = grad_zp.reshape(-1, D)
    flat_z = z_in_dyn.reshape(-1, D)
    grad_J = (flat_g.unsqueeze(2) * flat_z.unsqueeze(1)).mean(dim=0)  # (D, D)
    return grad_J.detach()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--groups", required=True,
                        help="Comma-separated wandb group names")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--area-split", type=int, default=64)
    parser.add_argument("--n-trajectories", type=int, default=16)
    parser.add_argument("--stages", default="init,trained")
    parser.add_argument("--loss-types", default="obs,latent")
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
    loss_types = [s.strip() for s in args.loss_types.split(",") if s.strip()]
    logger.info(f"groups={groups}  stages={stages}  losses={loss_types}  device={device}")

    rows: list[dict] = []

    for group in groups:
        # Pick chosen-best run id from the published metrics.json
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
                # NOTE: load_run always calls load_checkpoint internally, so
                # lit_model now has the TRAINED encoder. For "init" stage we
                # swap the encoder for a fresh one built from the same cfg
                # (re-seeded so the random init matches what training would
                # have started from). Only the encoder/decoder are used by
                # this diagnostic — the dynamics MLP is never run — so we
                # don't need to rebuild anything else.
                if stage == "init":
                    seed = (cfg.data.flow.random_state
                            + cfg.training.run_number + 1)
                    seed_everything(seed)
                    n_input = trajs["train_trajs"].sequence.shape[-1]
                    fresh_encoder = instantiate(cfg.model.encoder, n_input=n_input)
                    lit_model.encoder = fresh_encoder
                lit_model = lit_model.to(device).eval()

                n_dyn = getattr(lit_model, "n_target_dims", None)
                if n_dyn is None:
                    n_dyn = trajs["train_trajs"].sequence.shape[-1]
                logger.info(f"  stage={stage}  n_dyn={n_dyn}")

                seq = trajs.get("train_trajs", trajs["test_trajs"]).sequence
                batch = seq[: args.n_trajectories].to(device).float()
                x_in, x_tgt = batch[..., :-1, :], batch[..., 1:, :]

                for loss_type in loss_types:
                    has_enc = hasattr(lit_model, "encoder") and lit_model.encoder is not None
                    if loss_type == "latent" and not has_enc:
                        logger.info(f"    skip latent loss (no encoder)")
                        continue
                    grad_J = _compute_grad_on_J(
                        lit_model, x_in, x_tgt,
                        loss_type=loss_type, n_dyn=int(n_dyn),
                    )
                    bn = _block_norms(grad_J, args.area_split)
                    logger.info(
                        f"    loss={loss_type}  "
                        f"vv={bn['vv']:.3e}  vc={bn['vc']:.3e}  "
                        f"cv={bn['cv']:.3e}  cc={bn['cc']:.3e}"
                    )
                    rows.append({
                        "group": group, "run_id": run_id,
                        "stage": stage, "loss_type": loss_type,
                        "block_norms": bn,
                        "n_dyn": int(n_dyn),
                    })
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

    # ---- Plot: 2 rows (stages) × 2 cols (loss types), bars per block per group ----
    n_rows = len(stages); n_cols = len(loss_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.0 * n_rows),
                              squeeze=False, sharey=False)
    blocks = ["vv", "vc", "cv", "cc"]
    block_labels = ["vis→vis", "cog→vis", "vis→cog", "cog→cog"]
    cmap = plt.colormaps["tab10"]
    for r_i, stage in enumerate(stages):
        for c_i, loss_type in enumerate(loss_types):
            ax = axes[r_i][c_i]
            cell_rows = [row for row in rows
                         if row["stage"] == stage and row["loss_type"] == loss_type]
            if not cell_rows:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{stage} · {loss_type}", fontsize=10)
                continue
            x = np.arange(len(blocks))
            width = 0.8 / max(len(cell_rows), 1)
            for g_i, row in enumerate(cell_rows):
                vals = [row["block_norms"][b] for b in blocks]
                offset = (g_i - (len(cell_rows) - 1) / 2) * width
                color = cmap(g_i % 10)
                lbl = row["group"][:30] + ("…" if len(row["group"]) > 30 else "")
                ax.bar(x + offset, vals, width, color=color, label=lbl)
            ax.set_xticks(x)
            ax.set_xticklabels(block_labels, fontsize=9)
            ax.set_yscale("log")
            ax.set_ylabel(r"mean $|\partial L / \partial J_f|$")
            ax.set_title(f"stage = {stage}  ·  loss = {loss_type}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3, which="both")
            if r_i == 0 and c_i == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Gradient signal at the dynamics MLP, per area-block — outer-product "
        "decomposition\nE[ ∂L/∂z'[i] · z[j] ]; off-diagonals = signal asking the MLP "
        "to learn cross-area structure",
        y=1.02,
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
