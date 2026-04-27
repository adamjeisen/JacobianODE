"""Production-faithful per-phase profile of a few training+val batches.

Per-iter compute (forward / backward / loop_closure / reconstruction) was
shown to be identical between DirectSum and Monolithic encoders by the
isolated `profile_train_step` script — yet wandb production runtime
shows MO ~4.4× slower per epoch than DS. The gap therefore lives outside
the per-iter compute path: somewhere in Lightning's Trainer loop,
DataLoader, callbacks, optimizer.step, or validation.

This script runs the *real* Lightning Trainer with the production cfg
(loaded from a wandb run id), with both SimpleProfiler and
AdvancedProfiler attached, for a small but representative number of
train + val batches. Output: per-phase wall-time breakdown (Lightning's
internal categories) plus a cProfile dump of all Python frames.

Usage:
    python -m JacobianODE.jacobians.diagnostics.production_profile \\
        --wandb-entity JacobianODE \\
        --wandb-project WMTask_identity_encoder_verification \\
        --groups <g_ds>,<g_mo> --run-ids <rid_ds>,<rid_mo> \\
        --save-dir /orcd/.../latent_jac_runs \\
        --output-dir /orcd/.../diagnostics \\
        --train-batches 20 --val-batches 5 --tag prod_profile
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--tag", default="prod_profile")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import lightning.pytorch as L
    from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

    from JacobianODE.jacobians.checkpoints.loader import load_run

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)

    groups = [g.strip() for g in args.groups.split(",")]
    run_ids = [s.strip() for s in args.run_ids.split(",")]
    if len(groups) != len(run_ids):
        raise ValueError("--groups and --run-ids must have equal length")

    summary = {}

    for group, rid in zip(groups, run_ids):
        logger.info(f"=== {group}  run={rid} ===")
        prof_dir = output_dir / f"{args.tag}_{group[:40]}"
        prof_dir.mkdir(parents=True, exist_ok=True)

        loaded = load_run(
            f"{args.wandb_entity}/{args.wandb_project}",
            run_id=rid, save_dir=str(save_dir),
            generate_data=True, verbose=False, return_full_obs=False,
        )
        run_obj, cfg, eq, dt, values, train_dl, val_dl, test_dl, trajs, lit_model = loaded
        # load_checkpoint already happened inside load_run; we use the
        # trained model so we exercise the same code paths production hit.

        accum = int(cfg.training.trainer_params.get("accumulate_grad_batches", 1))
        logger.info(f"  cfg: accumulate_grad_batches={accum}, "
                    f"train_batches={args.train_batches}, val_batches={args.val_batches}")

        # Two profilers in series: SimpleProfiler gives the Lightning-phase
        # breakdown ("run_training_epoch", "training_step", "backward",
        # "optimizer_step_with_closure", "validation_step", etc.).
        # AdvancedProfiler runs cProfile under the hood — heavyweight but
        # gives every Python frame (useful for tracking down stragglers).
        simple = SimpleProfiler(dirpath=str(prof_dir), filename="simple")
        advanced = AdvancedProfiler(dirpath=str(prof_dir), filename="advanced")

        for tag, prof in [("simple", simple), ("advanced", advanced)]:
            logger.info(f"  --- profiler={tag} ---")
            trainer = L.Trainer(
                max_epochs=1,
                limit_train_batches=args.train_batches,
                limit_val_batches=args.val_batches,
                accumulate_grad_batches=accum,
                accelerator="gpu",
                devices=1,
                profiler=prof,
                logger=False,             # don't talk to wandb in profile
                enable_checkpointing=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
            )
            trainer.fit(lit_model, train_dataloaders=train_dl,
                        val_dataloaders=val_dl)

        # Read SimpleProfiler txt output (Lightning writes <name>-fit-...txt)
        simple_txts = sorted(prof_dir.glob("simple*.txt"))
        if simple_txts:
            logger.info(f"  SimpleProfiler output: {simple_txts[-1]}")
            text = simple_txts[-1].read_text()
            # Surface a few high-level lines in the log
            for line in text.splitlines():
                if any(s in line for s in ("Action", "------", "run_training_epoch",
                                            "run_validation_epoch", "training_step",
                                            "validation_step", "backward",
                                            "optimizer_step", "get_train_batch",
                                            "_TrainingEpochLoop", "_ValidationEpochLoop")):
                    logger.info(f"    {line}")
            summary[group] = {"run_id": rid,
                              "simple_profile_path": str(simple_txts[-1])}

        del lit_model, trainer, train_dl, val_dl, test_dl, trajs
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved summary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
