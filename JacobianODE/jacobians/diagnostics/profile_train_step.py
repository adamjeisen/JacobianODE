"""Profile per-component training-step time for DS vs MO encoders.

Builds both models fresh from cfg, runs N warmup + N timed iterations of
trajectory_model_step on a fixed batch, and breaks down where time goes
(encoder.encode, encode_trajectory, MLP.forward via compute_jacobians,
JacobianODEint integration, decoder, loss + backward).

Hypothesis under test: monolithic was reported as ~4.6x slower per epoch
than DirectSum in production runs. Verify and find the bottleneck.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def cuda_timer(label, sink, device):
    """CUDA-event timer that records elapsed ms into sink[label]."""
    import torch
    if device.type == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        yield
        e.record()
        torch.cuda.synchronize(device)
        sink.setdefault(label, []).append(s.elapsed_time(e))
    else:
        t0 = time.perf_counter()
        yield
        sink.setdefault(label, []).append((time.perf_counter() - t0) * 1000.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--groups", required=True,
                        help="Comma-separated wandb group names")
    parser.add_argument("--run-ids", required=True,
                        help="Comma-separated run ids matching --groups")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="DS_vs_MO_profile")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import numpy as np
    import torch
    from JacobianODE.jacobians.checkpoints.loader import load_run

    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    groups = [g.strip() for g in args.groups.split(",")]
    run_ids = [s.strip() for s in args.run_ids.split(",")]
    if len(groups) != len(run_ids):
        raise ValueError("--groups and --run-ids must have same length")

    results = {}

    for group, rid in zip(groups, run_ids):
        logger.info(f"=== {group}  rid={rid} ===")
        loaded = load_run(
            f"{args.wandb_entity}/{args.wandb_project}",
            run_id=rid, save_dir=str(save_dir),
            generate_data=True, verbose=False, return_full_obs=False,
        )
        run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded
        lit_model = lit_model.to(device).train()
        for p in lit_model.parameters():
            p.requires_grad_(True)

        seq = trajs.get("train_trajs", trajs["test_trajs"]).sequence
        batch = seq[: args.batch_size].to(device).float()

        # Wrap key methods so we can measure individual phases
        sink: dict[str, list[float]] = {}
        orig_encode_traj = lit_model.encode_trajectory
        orig_compute_jacs = lit_model.compute_jacobians
        orig_decode_traj = lit_model.decode_trajectory
        orig_recon = lit_model._reconstruction_loss
        orig_loop_close = getattr(lit_model, "loop_closure_model_step", None)
        orig_traj_step = lit_model.trajectory_model_step

        def wrapped_encode_traj(*a, **kw):
            with cuda_timer("encode_trajectory", sink, device):
                return orig_encode_traj(*a, **kw)

        def wrapped_compute_jacs(*a, **kw):
            with cuda_timer("compute_jacobians", sink, device):
                return orig_compute_jacs(*a, **kw)

        def wrapped_decode_traj(*a, **kw):
            with cuda_timer("decode_trajectory", sink, device):
                return orig_decode_traj(*a, **kw)

        def wrapped_recon(*a, **kw):
            with cuda_timer("recon_loss", sink, device):
                return orig_recon(*a, **kw)

        def wrapped_loop_close(*a, **kw):
            with cuda_timer("loop_closure_step", sink, device):
                return orig_loop_close(*a, **kw)

        def wrapped_traj_step(*a, **kw):
            with cuda_timer("trajectory_model_step", sink, device):
                return orig_traj_step(*a, **kw)

        lit_model.encode_trajectory = wrapped_encode_traj
        lit_model.compute_jacobians = wrapped_compute_jacs
        lit_model.decode_trajectory = wrapped_decode_traj
        lit_model._reconstruction_loss = wrapped_recon
        if orig_loop_close is not None:
            lit_model.loop_closure_model_step = wrapped_loop_close
        lit_model.trajectory_model_step = wrapped_traj_step

        optimizer = torch.optim.Adam(lit_model.parameters(), lr=1e-4)

        def _do_train_step():
            """Run the FULL Lightning training_step + backward."""
            optimizer.zero_grad(set_to_none=True)
            loss = lit_model.training_step(batch, batch_idx=0)
            if isinstance(loss, dict):
                loss = loss.get("loss", loss)
            loss.backward()

        # Warmup
        for _ in range(args.n_warmup):
            _do_train_step()
        sink.clear()

        # Timed iterations — full training_step
        for _ in range(args.n_iters):
            optimizer.zero_grad(set_to_none=True)
            with cuda_timer("TOTAL_iter", sink, device):
                with cuda_timer("training_step_forward", sink, device):
                    loss = lit_model.training_step(batch, batch_idx=0)
                    if isinstance(loss, dict):
                        loss = loss.get("loss", loss)
                with cuda_timer("backward", sink, device):
                    loss.backward()

        # Restore
        lit_model.encode_trajectory = orig_encode_traj
        lit_model.compute_jacobians = orig_compute_jacs
        lit_model.decode_trajectory = orig_decode_traj
        lit_model._reconstruction_loss = orig_recon
        if orig_loop_close is not None:
            lit_model.loop_closure_model_step = orig_loop_close
        lit_model.trajectory_model_step = orig_traj_step

        stats = {}
        for k, vs in sink.items():
            arr = np.asarray(vs)
            stats[k] = {
                "n_calls_per_iter": float(arr.size / args.n_iters),
                "mean_ms_per_call": float(arr.mean()),
                "total_ms_per_iter": float(arr.sum() / args.n_iters),
                "p95_ms_per_call": float(np.percentile(arr, 95)),
            }
        results[group] = {"run_id": rid, "stats": stats,
                          "iter_total_ms": stats.get("TOTAL_iter", {}).get("total_ms_per_iter")}

        logger.info(f"  --- {group} ---")
        for k in ("TOTAL_iter", "training_step_forward", "trajectory_model_step",
                  "loop_closure_step", "recon_loss", "encode_trajectory",
                  "compute_jacobians", "decode_trajectory", "backward"):
            if k in stats:
                s = stats[k]
                logger.info(
                    f"    {k:>22}: {s['total_ms_per_iter']:>8.1f} ms/iter   "
                    f"({s['n_calls_per_iter']:>4.0f} calls × {s['mean_ms_per_call']:>6.1f} ms)"
                )

        del lit_model
        torch.cuda.empty_cache()

    # ---- Side-by-side comparison ----
    if len(results) >= 2:
        keys = list(results.keys())
        g1, g2 = keys[0], keys[1]
        s1, s2 = results[g1]["stats"], results[g2]["stats"]
        logger.info(f"\n=== ratio {g2[:30]} / {g1[:30]} ===")
        all_keys = sorted(set(s1.keys()) | set(s2.keys()))
        for k in all_keys:
            t1 = s1.get(k, {}).get("total_ms_per_iter", 0)
            t2 = s2.get(k, {}).get("total_ms_per_iter", 0)
            ratio = t2 / t1 if t1 > 0 else float("nan")
            logger.info(f"  {k:>22}: {t1:>7.1f}  vs  {t2:>7.1f}  ms/iter   ratio={ratio:.2f}x")

    out_path = output_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
