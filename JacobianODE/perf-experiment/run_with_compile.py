"""Variant of run_with_callback.py that applies torch.compile to the
inner Jacobian MLP and encoder before training starts.

Same arguments as run_with_callback.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
    import torch

    timing_cb = PerStepTimingCallback(output_json)
    orig_train_model = trainer_mod.train_model

    compile_mode = os.environ.get("TC_MODE", "default")
    compile_dynamic = os.environ.get("TC_DYNAMIC", "1") == "1"
    compile_targets = os.environ.get("TC_TARGETS", "model,encoder").split(",")
    compile_backend = os.environ.get("TC_BACKEND", "")  # "" = use default

    def patched_train_model(cfg, lit_model, train_dataloaders, val_dataloaders, name=None, extra_callbacks=None):
        # Wrap inner nn.Modules with torch.compile.
        # We set the attribute so that lit_model.model(x) goes through the compiled
        # wrapper. (torch.compile returns an OptimizedModule that subclasses
        # nn.Module, so it's drop-in.)
        for tgt in compile_targets:
            tgt = tgt.strip()
            if not tgt:
                continue
            mod = getattr(lit_model, tgt, None)
            if mod is None or not isinstance(mod, torch.nn.Module):
                print(f"[compile] skip {tgt}: not an nn.Module")
                continue
            kwargs = dict(dynamic=compile_dynamic)
            if compile_backend:
                kwargs["backend"] = compile_backend
            else:
                kwargs["mode"] = compile_mode
            print(f"[compile] wrapping lit_model.{tgt} with torch.compile({kwargs})")
            compiled = torch.compile(mod, **kwargs)
            setattr(lit_model, tgt, compiled)

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
