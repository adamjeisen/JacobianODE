"""Run a training config and capture per-step (step, loss, walltime) trace.

Bypasses @hydra.main: uses hydra.compose() so we can inject the
PerStepTimingCallback into train_model without modifying the trainer.

Usage:
    uv run --no-sync python /path/to/run_with_callback.py \
        <output_json> <config_name> <max_steps> <batch_size> <seed> <run_dir> [hydra overrides...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the perf_callback module importable when the script is run by path.
sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    if len(sys.argv) < 7:
        print(__doc__)
        return 2

    output_json = os.path.abspath(sys.argv[1])
    config_name = sys.argv[2]
    max_steps = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    seed = int(sys.argv[5])
    run_dir = os.path.abspath(sys.argv[6])
    extra = sys.argv[7:]

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    from hydra import initialize_config_dir, compose

    overrides = [
        f"experiment={config_name}",
        f"+training.trainer_params.max_steps={max_steps}",
        "training.trainer_params.limit_train_batches=200",
        "training.trainer_params.limit_val_batches=10",
        f"training.batch_size={batch_size}",
        f"training.logger_save_dirs={run_dir}/lightning",
        f"training.logger.save_dir={run_dir}/lightning",
        f"data.flow.random_state={seed}",
    ] + list(extra)

    conf_dir = "/workspaces/JacobianODE/JacobianODE/jacobians/conf"
    with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=overrides)

    from JacobianODE.jacobians.training import trainer as trainer_mod
    from JacobianODE.jacobians import training as training_pkg
    import JacobianODE.jacobians.run_jacobians as rj
    from perf_callback import PerStepTimingCallback

    timing_cb = PerStepTimingCallback(output_json)
    orig_train_model = trainer_mod.train_model

    def patched_train_model(cfg, lit_model, train_dataloaders, val_dataloaders, name=None, extra_callbacks=None):
        cbs = list(extra_callbacks or [])
        cbs.append(timing_cb)
        return orig_train_model(
            cfg,
            lit_model,
            train_dataloaders,
            val_dataloaders,
            name=name,
            extra_callbacks=cbs,
        )

    trainer_mod.train_model = patched_train_model
    training_pkg.train_model = patched_train_model
    rj.train_model = patched_train_model

    from JacobianODE.jacobians.run_jacobians import _run_training
    _run_training(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
