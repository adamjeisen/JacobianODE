# perf-experiment — baseline measurements

Pre-flight baseline establishing that the sandbox can launch and run
training for both target configs, and giving sandbox-claude starting
wall-clock numbers to compare against.

## Wall-clock (max_steps=20, seed=42)

| Config | Run 1 | Run 2 | Mean | Per-step (avg) |
|---|---|---|---|---|
| Lorenz `lorenz_partial_additive_splitmode_p30_obsnoise001_nd75_init15_autodim__lc_sweep` | 66.15s | 64.33s | 65.2s | ~3.26s/step |
| wmtask `wmtask_direct_sum_additive_splitmode_p30_perareaautodim__lc_x_obsnoisescale_sweep` | 110.87s | 113.21s | 112.0s | ~5.60s/step |

Per-step is averaged across all 20 steps including process startup, hydra
config resolution, data loading, model init, and CUDA warmup. **The startup
overhead is non-trivial relative to 20 steps.** To get a clean
per-step-after-warmup number, run at two different `max_steps` values and
take the slope:

```
t(max_steps=N) ≈ t_setup + N * t_per_step_steady
```

## How to reproduce

From the host:

```bash
cd ~/Documents/code/JacobianODE
devcontainer exec --workspace-folder . bash -lc '
  cd /workspaces/JacobianODE
  bash JacobianODE/perf-experiment/run_baseline.sh <config_name> <seed> <max_steps>
'
```

The wrapper (`run_baseline.sh`) does two passes per config and writes
each pass to `/tmp/perf-exp/<config>/{run1,run2}/`.

## Sandbox-specific constraints discovered during pre-flight

These were not documented before; they shape what works in this environment.

1. **GPU memory: 4 GB total, ~3.6 GB usable.** GTX 1050 Ti is small. The
   default `batch_size: 32` (Lorenz) / `16` (wmtask) OOMs immediately. Use
   `batch_size=8` (Lorenz) or `batch_size=4` (wmtask) for measurement.
   This is a setup constraint, not a perf-improvement target — keep the
   reduced batch size for ALL measurements so they're comparable to
   baseline.

2. **Engaging mount is read-only.** Override the YAML's `training.logger_save_dirs`
   and `training.logger.save_dir` to `/tmp/perf-exp/...` (writable, ephemeral)
   to avoid `Read-only file system` errors when Lightning tries to write
   logs/checkpoints.

3. **Wandb: use `WANDB_MODE=offline`.** Offline mode preserves the wandb
   code path's timing characteristics (so you're measuring what you'd see
   in a real run) but skips network sync.

4. **Hydra override quirks.**
   - `+training.trainer_params.max_steps=N` (the `+` is needed because
     `max_steps` isn't a pre-existing key in `training.yaml`, only
     `max_epochs` is).
   - `training.batch_size=N` (no `+`, exists in base).
   - `training.logger_save_dirs=...` (no `+`, brought in by experiment override).
   - `data.flow.random_state=N` (this is the `seed_everything` source).

5. **Bind-mount handles are captured at container-create.** If you remount
   sshfs (e.g. change mount options), the container's bind-mounts go stale
   with `Transport endpoint is not connected`. Recreate the container with
   `docker rm -f <id>; devcontainer up --workspace-folder .`.

6. **The required overrides bundle** (use this as a starting point for any
   timing run):

   ```
   experiment=<config>
   +training.trainer_params.max_steps=<N>
   training.trainer_params.limit_train_batches=200    # plenty for max_steps
   training.trainer_params.limit_val_batches=10
   training.batch_size=<8 lorenz | 4 wmtask>
   training.logger_save_dirs=/tmp/perf-exp/<config>/<runid>/lightning
   training.logger.save_dir=/tmp/perf-exp/<config>/<runid>/lightning
   data.flow.random_state=<seed>
   hydra.run.dir=/tmp/perf-exp/<config>/<runid>/hydra
   ```

   Plus env: `WANDB_MODE=offline`.

## What pre-flight did NOT verify

- **Loss reproducibility across two runs.** The wandb-history.jsonl file
  isn't generated in offline mode; wandb writes a binary `.wandb` file that
  isn't easily parsed. Building a per-step `(step, loss)` callback or
  alternative metrics-extraction is part of the 8h sandbox task — it's the
  prerequisite for the regression-test gate.

- **Per-step time after warmup.** Only end-to-end wall-clock is measured.
  Sandbox-claude should add a Lightning callback that records per-step
  walltime and writes to a JSON sidecar, then compute mean ± std after
  dropping the first ~5 warmup steps.

## Trajectory.py lockfile fix

`trajectory.py:make_dysts_trajectories` originally opened a write-mode
lockfile before checking the cache, which crashed on the read-only engaging
mount. Fixed in commit `f126db62` on `latent-JacobianODE` (now also on
this `perf-experiment` branch): `os.access(data_save_dir, os.W_OK)` gates
lock acquisition so cache-reads on a ro fs work. No effect on engaging-side
training (RW there → original lock path).
