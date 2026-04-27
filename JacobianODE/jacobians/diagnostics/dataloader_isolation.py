"""Isolate the DS-vs-MO validation slowdown to (a) DataLoader workers or
(b) something deeper inside Lightning.

For each (group, dl_mode) combo runs a short Trainer.fit with SimpleProfiler
and reports validation_step wall time:
  dl_mode=production: keep cfg's num_workers + pin_memory (matches real runs)
  dl_mode=minimal:    rebuild val_dl with num_workers=0, pin_memory=False

If the 4× gap vanishes in `minimal`, it's DataLoader worker IPC.
If it persists, the cause is in Lightning's val hooks themselves.
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
    parser.add_argument("--train-batches", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import lightning.pytorch as L
    from lightning.pytorch.profilers import SimpleProfiler
    from torch.utils.data import DataLoader

    from JacobianODE.jacobians.checkpoints.loader import load_run

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)

    groups = [g.strip() for g in args.groups.split(",")]
    run_ids = [s.strip() for s in args.run_ids.split(",")]
    summary = {}

    for group, rid in zip(groups, run_ids):
        logger.info(f"=== {group}  rid={rid} ===")
        loaded = load_run(
            f"{args.wandb_entity}/{args.wandb_project}",
            run_id=rid, save_dir=str(save_dir),
            generate_data=True, verbose=False, return_full_obs=False,
        )
        run_obj, cfg, eq, dt, values, train_dl, val_dl, test_dl, trajs, lit_model = loaded

        # Inspect production DataLoader settings (for the log)
        prod_workers = getattr(val_dl, "num_workers", "?")
        prod_pin = getattr(val_dl, "pin_memory", "?")
        prod_persist = getattr(val_dl, "persistent_workers", "?")
        logger.info(f"  cfg val_dl: num_workers={prod_workers}  "
                    f"pin_memory={prod_pin}  persistent_workers={prod_persist}")

        # Build a minimal DataLoader from the same dataset
        minimal_val_dl = DataLoader(
            val_dl.dataset,
            batch_size=val_dl.batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            persistent_workers=False,
        )
        minimal_train_dl = DataLoader(
            train_dl.dataset,
            batch_size=train_dl.batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            persistent_workers=False,
        )

        accum = int(cfg.training.trainer_params.get("accumulate_grad_batches", 1))
        group_summary = {"run_id": rid, "modes": {}}

        for mode, t_dl, v_dl in [("production", train_dl, val_dl),
                                  ("minimal", minimal_train_dl, minimal_val_dl)]:
            logger.info(f"  --- mode={mode} ---")
            sp = SimpleProfiler(dirpath=str(output_dir),
                                filename=f"dl_iso_{group[:30]}_{mode}")
            trainer = L.Trainer(
                max_epochs=1,
                limit_train_batches=args.train_batches,
                limit_val_batches=args.val_batches,
                accumulate_grad_batches=accum,
                accelerator="gpu",
                devices=1,
                profiler=sp,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
            )
            trainer.fit(lit_model, train_dataloaders=t_dl, val_dataloaders=v_dl)

            # Extract validation_step time from the SimpleProfiler dict
            recorded = sp.recorded_durations
            val_durs = recorded.get(
                "[Strategy]SingleDeviceStrategy.validation_step", []
            )
            train_durs = recorded.get(
                "[Strategy]SingleDeviceStrategy.training_step", []
            )
            val_total = sum(val_durs) if val_durs else 0.0
            train_total = sum(train_durs) if train_durs else 0.0
            n_val = len(val_durs)
            n_train = len(train_durs)
            logger.info(
                f"    val: {val_total:.2f} sec total  "
                f"({n_val} calls × {val_total/max(n_val,1):.3f} sec)   "
                f"train: {train_total:.2f} sec ({n_train} × "
                f"{train_total/max(n_train,1):.3f})"
            )
            group_summary["modes"][mode] = {
                "n_val": n_val, "val_total_sec": float(val_total),
                "val_per_call_sec": float(val_total / max(n_val, 1)),
                "n_train": n_train, "train_total_sec": float(train_total),
                "train_per_call_sec": float(train_total / max(n_train, 1)),
            }

        summary[group] = group_summary
        del lit_model, trainer
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-group comparison
    if len(summary) >= 2:
        keys = list(summary.keys())
        logger.info(f"\n=== val_per_call ratio {keys[1][:30]} / {keys[0][:30]} ===")
        for mode in ("production", "minimal"):
            t1 = summary[keys[0]]["modes"][mode]["val_per_call_sec"]
            t2 = summary[keys[1]]["modes"][mode]["val_per_call_sec"]
            r = t2 / t1 if t1 > 0 else float("inf")
            logger.info(f"  {mode}: {t1:.3f}s vs {t2:.3f}s  ratio={r:.2f}x")

    out = output_dir / "dataloader_isolation.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
