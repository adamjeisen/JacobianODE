"""Wall-clock profile of validation_step phases with EXPLICIT cuda.synchronize.

Lightning's SimpleProfiler + cProfile mis-attributed validation time to
torch.linalg.eigvals — but a focused micro-benchmark showed eigvals is
the same speed for DS and MO. The real bottleneck is hidden in async
GPU execution: whichever call happens to force a sync gets blamed.

This script wraps each validation phase (encode, trajectory step, loop
closure, reconstruction, one-step trajectory, eigvals, etc.) with
torch.cuda.synchronize() before AND after — so wall time per phase
reflects actual GPU compute for that phase, not pending queue drain.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def synced(label, sink, device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize()
    sink[label].append(time.perf_counter() - t0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--run-ids", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--n-iters", type=int, default=10,
                        help="N validation_step iterations to time")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import torch
    import numpy as np
    import torch.nn.functional as F

    from JacobianODE.jacobians.checkpoints.loader import load_run

    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    groups = [g.strip() for g in args.groups.split(",")]
    run_ids = [s.strip() for s in args.run_ids.split(",")]

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

        # Replicate validation_step phases with explicit syncs.
        # Mirrors LitLatentJacobianODE.validation_step around line 1860+.
        sink: dict[str, list[float]] = defaultdict(list)
        n_eigval = int(getattr(lit_model, "n_eigval_jacobians", 8) or 8)

        def one_val_iter():
            kw = dict(
                alpha_teacher_forcing=getattr(lit_model, "alpha_validation", 1),
                obs_noise_scale=0,

                reconstruction_mode="most_recent",
            )
            with synced("val/1.trajectory_step", sink, device):
                vt = lit_model.trajectory_model_step(batch, 0, 0, **kw)
            with synced("val/2.encode", sink, device):
                z_full = lit_model.encode_trajectory(batch)
                z_dyn, z_null = lit_model._split_latent(z_full)
                z_dyn_s, _, _ = lit_model._vae_reparameterize(z_dyn)
            with synced("val/3.loop_closure", sink, device):
                _ = lit_model.loop_closure_model_step(z_dyn_s, 0, 0)
            with synced("val/4.recon_loss", sink, device):
                if lit_model.reconstruction_loss_weight > 0:
                    with torch.no_grad():
                        _ = lit_model._reconstruction_loss(batch, z_full=z_full)
            with synced("val/5.one_step_traj", sink, device):
                with torch.no_grad():
                    _ = lit_model.trajectory_model_step(
                        batch, 0, 0, alpha_teacher_forcing=1,
                        obs_noise_scale=0,
                    )
            with synced("val/6.eigvals", sink, device):
                with torch.no_grad():
                    pred_jacs = lit_model.compute_jacobians(z_dyn_s)
                    B, T, D, _ = pred_jacs.shape
                    flat = pred_jacs.reshape(B * T, D, D)
                    if n_eigval < B * T:
                        idx = torch.randperm(B * T, device=device)[:n_eigval]
                        flat = flat[idx]
                    fm = torch.isfinite(flat).all(dim=-1).all(dim=-1)
                    flat = flat[fm]
                    if flat.numel() > 0:
                        _ = torch.linalg.eigvals(flat).real.flatten()

        for _ in range(args.n_warmup):
            one_val_iter()
        for k in list(sink.keys()):
            sink[k].clear()

        for _ in range(args.n_iters):
            one_val_iter()

        stats = {}
        for k, vs in sink.items():
            arr = np.asarray(vs)
            stats[k] = {
                "mean_ms": float(1e3 * arr.mean()),
                "median_ms": float(1e3 * np.median(arr)),
                "p95_ms": float(1e3 * np.percentile(arr, 95)),
            }
        total_mean = sum(s["mean_ms"] for s in stats.values())
        logger.info(f"  total mean: {total_mean:.1f} ms/val_iter")
        for k in sorted(stats.keys()):
            s = stats[k]
            pct = 100 * s["mean_ms"] / total_mean
            logger.info(f"    {k:>22}: {s['mean_ms']:>7.1f} ms (p95 {s['p95_ms']:>7.1f})  "
                        f"{pct:>5.1f}%")

        all_results[group] = {"run_id": rid, "n_iters": args.n_iters,
                              "stats": stats, "total_mean_ms": total_mean}
        del lit_model
        torch.cuda.empty_cache()

    # Side-by-side ratio
    if len(all_results) >= 2:
        keys = list(all_results.keys())
        s1 = all_results[keys[0]]["stats"]
        s2 = all_results[keys[1]]["stats"]
        logger.info(f"\n=== ratio {keys[1][:30]} / {keys[0][:30]} ===")
        for k in sorted(set(s1.keys()) | set(s2.keys())):
            t1 = s1.get(k, {}).get("mean_ms", 0.0)
            t2 = s2.get(k, {}).get("mean_ms", 0.0)
            r = (t2 / t1) if t1 > 0 else float("inf")
            logger.info(f"  {k:>22}: {t1:>7.1f}  vs  {t2:>7.1f}  ms   ratio={r:.2f}x")

    out = output_dir / "sync_profile_val.json"
    out.write_text(json.dumps(all_results, indent=2))
    logger.info(f"Saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
