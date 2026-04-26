"""FNN-autodim sweep on the project's dysts-generated training data.

Sweeps (obs_noise × n_delays) on Lorenz delay embeddings, runs whitened
PCA-FNN k=1 stop-at-min for each cell, and saves an overlay figure +
JSON. Two stability modes:

    --mode single    one call to fnn_dim_estimate at --n-samples (default 10000).
    --mode median    --n-trials calls at --n-samples each, median dim returned.

GPU-friendly: pass --device cuda --dtype float32 to fit n_samples ~= 10000
on an H100 (the O(N^2 * D) pdist tensor is ~40GB at N=10000, D=100, fp32).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True,
                        help="Hydra experiment name (loaded with default overrides)")
    parser.add_argument("--mode", choices=("single", "median"), required=True)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Median mode only: number of independent trials")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default=None,
                        help="Optional filename suffix; defaults to mode/n params")
    parser.add_argument("--noise-levels", default="0.0,0.01,0.05",
                        help="Comma-separated obs_noise values to sweep")
    parser.add_argument("--n-delays-list", default="3,5,10,15,20,30,45,60,75,85,100",
                        help="Comma-separated n_delays values to sweep")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from JacobianODE.fnn.dim_estimator import fnn_dim_estimate
    from JacobianODE.jacobians.data import (
        make_trajectories, postprocess_data, create_dataloaders,
    )

    NOISE_LEVELS = [float(x) for x in args.noise_levels.split(",")]
    NDELAYS = [int(x) for x in args.n_delays_list.split(",")]
    DTYPE = torch.float32 if args.dtype == "float32" else torch.float64
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        logger.warning("--device=cuda requested but CUDA not available; falling back to CPU")
        DEVICE = torch.device("cpu")
    logger.info(f"device={DEVICE}, dtype={DTYPE}, mode={args.mode}, "
                f"n_samples={args.n_samples}, n_trials={args.n_trials if args.mode=='median' else 1}")

    # Bootstrap base cfg once.
    GlobalHydra.instance().clear()
    conf_dir = str(
        Path(__file__).resolve().parents[2] / "jacobians" / "conf"
    )
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        base_cfg = compose(
            config_name="config",
            overrides=[f"experiment={args.experiment}"],
        )

    logger.info("Generating dysts trajectories (one-time)...")
    eq, sol, dt = make_trajectories(base_cfg, verbose=False)
    values_raw = sol["values"]
    raw_shape = tuple(values_raw.shape) if hasattr(values_raw, "shape") else None
    logger.info(f"  raw shape: {raw_shape}, dt={dt}")

    rows = []
    for noise in NOISE_LEVELS:
        for nd in NDELAYS:
            cfg = base_cfg.copy()
            OmegaConf.update(cfg, "data.postprocessing.obs_noise", float(noise))
            OmegaConf.update(cfg,
                "data.train_test_params.delay_embedding_params.n_delays", int(nd))
            result = postprocess_data(cfg, values_raw)
            cfg.data.postprocessing.noise_scale_factor = result.noise_scale_factor
            cfg.data.postprocessing.mu = result.mu
            cfg.data.postprocessing.sigma = result.sigma
            _, _, _, trajs = create_dataloaders(cfg, result.values)

            seq = trajs["train_trajs"].sequence
            flat = seq.reshape(-1, seq.shape[-1])
            flat_t = flat.to(dtype=DTYPE, device=DEVICE)

            if args.mode == "single":
                d = int(fnn_dim_estimate(
                    flat_t, threshold=args.threshold,
                    n_samples=args.n_samples, k=args.k,
                    device=DEVICE, dtype=DTYPE,
                ))
                row = {"noise": noise, "n_delays": nd, "N": int(flat.shape[0]),
                       "D": int(flat.shape[-1]), "fnn_dim": d}
            else:
                trials = []
                for t in range(args.n_trials):
                    torch.manual_seed(t)  # control the subsample inside loss_false
                    if DEVICE.type == "cuda":
                        torch.cuda.manual_seed_all(t)
                    trials.append(int(fnn_dim_estimate(
                        flat_t, threshold=args.threshold,
                        n_samples=args.n_samples, k=args.k,
                        device=DEVICE, dtype=DTYPE,
                    )))
                med = int(round(statistics.median(trials)))
                row = {"noise": noise, "n_delays": nd, "N": int(flat.shape[0]),
                       "D": int(flat.shape[-1]),
                       "fnn_dim": med, "trials": trials}
            rows.append(row)
            logger.info(
                f"  noise={noise}  n_delays={nd:>3}  D={int(flat.shape[-1]):>3}  "
                f"fnn={row['fnn_dim']}"
                + (f"  (trials={row.get('trials')})" if 'trials' in row else "")
            )

    # ---- Plot ----
    fig, axes = plt.subplots(1, len(NOISE_LEVELS), figsize=(5 * len(NOISE_LEVELS), 4.5),
                              sharex=True, sharey=True)
    if len(NOISE_LEVELS) == 1:
        axes = [axes]
    for ax, noise in zip(axes, NOISE_LEVELS):
        rs = [r for r in rows if r["noise"] == noise]
        xs = [r["n_delays"] for r in rs]
        ys = [r["fnn_dim"] for r in rs]
        ax.plot(xs, xs, "k-", lw=0.6, alpha=0.4, label="n_delays")
        ax.plot(xs, ys, "o-", color="C3", lw=1.4, ms=5, label=f"fnn_dim ({args.mode})")
        if "trials" in rs[0]:
            for r in rs:
                ax.plot([r["n_delays"]] * len(r["trials"]), r["trials"],
                        "x", color="C3", ms=4, alpha=0.4)
        ax.axhline(3, color="gray", ls=":", lw=0.8, label="Lorenz intrinsic = 3")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_delays"); ax.set_ylabel("fnn dim")
        ax.set_title(f"obs_noise = {noise}")
        ax.grid(True, alpha=0.3, which="both")
        if noise == NOISE_LEVELS[0]:
            ax.legend(fontsize=8, loc="upper left")

    title = (f"FNN-autodim sweep ({args.experiment})  ·  mode={args.mode}  "
             f"·  n_samples={args.n_samples}"
             + (f"  ·  median of {args.n_trials} trials" if args.mode == "median" else ""))
    fig.suptitle(title, y=1.01)
    fig.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (
        f"single_n{args.n_samples}" if args.mode == "single"
        else f"median{args.n_trials}_n{args.n_samples}"
    )
    base = f"fnn_autodim_dysts_{args.experiment}_{tag}"
    fig_path = output_dir / f"{base}.png"
    json_path = output_dir / f"{base}.json"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    json_path.write_text(json.dumps({
        "experiment": args.experiment,
        "mode": args.mode,
        "n_samples": args.n_samples,
        "n_trials": args.n_trials if args.mode == "median" else 1,
        "threshold": args.threshold,
        "k": args.k,
        "dtype": args.dtype,
        "noise_levels": NOISE_LEVELS,
        "n_delays_list": NDELAYS,
        "rows": rows,
    }, indent=2))
    logger.info(f"Saved → {fig_path}")
    logger.info(f"Saved → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
