"""Profile a few training steps with torch.profiler.

Loads a config, runs N profiled steps, prints top hot ops by self CUDA time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: profile_step.py <config> <max_steps> <run_dir> [n_active=10]")
        return 2
    config_name = sys.argv[1]
    max_steps = int(sys.argv[2])
    run_dir = os.path.abspath(sys.argv[3])

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    from hydra import initialize_config_dir, compose

    case_bs = 4 if "wmtask" in config_name else 8
    overrides = [
        f"experiment={config_name}",
        f"+training.trainer_params.max_steps={max_steps}",
        "training.trainer_params.limit_train_batches=200",
        "training.trainer_params.limit_val_batches=10",
        f"training.batch_size={case_bs}",
        f"training.logger_save_dirs={run_dir}/lightning",
        f"training.logger.save_dir={run_dir}/lightning",
        "data.flow.random_state=42",
        f"hydra.run.dir={run_dir}/hydra",
    ]

    conf_dir = "/workspaces/JacobianODE/JacobianODE/jacobians/conf"
    with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=overrides)

    import lightning as L
    import torch
    from torch.profiler import profile, ProfilerActivity, schedule

    # Inject a profiling callback that runs torch.profiler over a few steps
    class ProfilerCallback(L.Callback):
        def __init__(self):
            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=2, warmup=2, active=5, repeat=1),
                record_shapes=False,
                with_stack=False,
            )

        def on_train_start(self, trainer, pl_module):
            self.prof.__enter__()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.prof.step()

        def on_train_end(self, trainer, pl_module):
            self.prof.__exit__(None, None, None)
            print("\n=== Top 30 by self_cuda_time_total ===")
            print(self.prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=30))
            print("\n=== Top 20 by cpu_time_total ===")
            print(self.prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=20))

    from JacobianODE.jacobians.training import trainer as trainer_mod
    from JacobianODE.jacobians import training as training_pkg
    import JacobianODE.jacobians.run_jacobians as rj

    cb = ProfilerCallback()
    orig = trainer_mod.train_model

    def patched(cfg, lit_model, train_dataloaders, val_dataloaders, name=None, extra_callbacks=None):
        cbs = list(extra_callbacks or [])
        cbs.append(cb)
        return orig(cfg, lit_model, train_dataloaders, val_dataloaders, name=name, extra_callbacks=cbs)

    trainer_mod.train_model = patched
    training_pkg.train_model = patched
    rj.train_model = patched

    from JacobianODE.jacobians.run_jacobians import _run_training
    _run_training(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
