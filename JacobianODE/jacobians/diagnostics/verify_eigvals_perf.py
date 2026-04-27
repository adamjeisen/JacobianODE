"""Verify the eigvals performance gap between DS and MO trained Jacobians.

Loads each trained model, computes a small batch of dynamics Jacobians,
times torch.linalg.eigvals on each set in isolation. Confirms or refutes
the claim that MO Jacobians are 20× slower to eigendecompose than DS's.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--run-ids", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--n-jacobians", type=int, default=8,
                        help="Matches model.n_eigval_jacobians (default 8)")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Repeat each eigvals call N times for stable timing")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import time
    import torch
    import numpy as np
    from JacobianODE.jacobians.checkpoints.loader import load_run

    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    groups = [g.strip() for g in args.groups.split(",")]
    run_ids = [s.strip() for s in args.run_ids.split(",")]

    all_jacs = {}
    all_results = {}

    for group, rid in zip(groups, run_ids):
        logger.info(f"=== {group}  rid={rid} ===")
        loaded = load_run(
            f"{args.wandb_entity}/{args.wandb_project}",
            run_id=rid, save_dir=str(save_dir),
            generate_data=True, verbose=False, return_full_obs=False,
        )
        run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded
        lit_model = lit_model.to(device).eval()

        seq = trajs.get("train_trajs", trajs["test_trajs"]).sequence
        batch = seq[:16].to(device).float()

        # Replicate the exact validation eigvals path: encode → split → compute_jacobians → subsample
        with torch.no_grad():
            z_full = lit_model.encode_trajectory(batch)
            z_dyn, _ = lit_model._split_latent(z_full)
            pred_jacs = lit_model.compute_jacobians(z_dyn)  # (B, T, D, D)
        B, T, D, _ = pred_jacs.shape
        jacs_flat = pred_jacs.reshape(B * T, D, D)
        torch.manual_seed(0)
        idx = torch.randperm(B * T, device=device)[: args.n_jacobians]
        jacs_sample = jacs_flat[idx]
        finite_mask = torch.isfinite(jacs_sample).all(dim=-1).all(dim=-1)
        jacs = jacs_sample[finite_mask]
        logger.info(f"  jacs shape: {tuple(jacs.shape)}, dtype={jacs.dtype}")

        # Conditioning + matrix-property summary (helps interpret the gap)
        with torch.no_grad():
            sv = torch.linalg.svdvals(jacs)
            cond = sv[:, 0] / sv[:, -1].clamp(min=1e-30)
            spectral_norm = sv[:, 0]
            frob_norm = jacs.flatten(1).norm(dim=1)
            non_normality = frob_norm / spectral_norm  # ~1 normal, >>1 non-normal
            logger.info(
                f"  cond:    median={cond.median().item():.2e}  "
                f"max={cond.max().item():.2e}"
            )
            logger.info(
                f"  σ_max:   median={spectral_norm.median().item():.3f}"
            )
            logger.info(
                f"  ||J||_F / ||J||_2 (non-normality):  "
                f"median={non_normality.median().item():.3f}"
            )

        # Time eigvals: warmup + N trials. Synchronize before/after each.
        for _ in range(3):
            with torch.no_grad():
                _ = torch.linalg.eigvals(jacs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(args.n_trials):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                eigs = torch.linalg.eigvals(jacs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times = np.asarray(times)
        logger.info(
            f"  eigvals time over {args.n_trials} trials: "
            f"median={1e3*np.median(times):.1f} ms  "
            f"mean={1e3*times.mean():.1f} ms  "
            f"min={1e3*times.min():.1f} ms"
        )

        # Eigenvalue distribution (real parts, sorted)
        with torch.no_grad():
            eigs = torch.linalg.eigvals(jacs)
            real = eigs.real.flatten().cpu().numpy()
        logger.info(
            f"  Re(λ) distribution: min={real.min():.2f}  max={real.max():.2f}  "
            f"|Im(λ)|>0 fraction: {float((eigs.imag.abs() > 1e-6).float().mean().item()):.2f}"
        )

        all_results[group] = {
            "run_id": rid,
            "jacs_shape": list(jacs.shape),
            "eigvals_ms_median": float(1e3 * np.median(times)),
            "eigvals_ms_mean": float(1e3 * times.mean()),
            "cond_median": float(cond.median().item()),
            "cond_max": float(cond.max().item()),
            "spectral_norm_median": float(spectral_norm.median().item()),
            "non_normality_median": float(non_normality.median().item()),
            "lambda_real_min": float(real.min()),
            "lambda_real_max": float(real.max()),
        }
        del lit_model
        torch.cuda.empty_cache()

    out_path = output_dir / "verify_eigvals_perf.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info(f"Saved → {out_path}")

    if len(all_results) >= 2:
        keys = list(all_results.keys())
        t1 = all_results[keys[0]]["eigvals_ms_median"]
        t2 = all_results[keys[1]]["eigvals_ms_median"]
        logger.info(f"\n=== ratio {keys[1][:30]} / {keys[0][:30]} ===")
        logger.info(f"  eigvals: {t1:.1f} ms  vs  {t2:.1f} ms  ->  ratio = {t2/t1 if t1>0 else float('inf'):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
