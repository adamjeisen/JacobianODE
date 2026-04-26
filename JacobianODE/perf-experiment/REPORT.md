# perf-experiment — REPORT

Live log of optimization attempts. Updated after every attempt.

## Run metadata

- Branch: `perf-experiment-claude`
- Start: 2026-04-26 22:13 UTC
- Sandbox GPU: NVIDIA GeForce GTX 1050 Ti (4 GB) — Pascal, no native fp16
- Measurement schedule: `max_steps=25` for fast iteration, `max_steps=80` for
  attempts where compile / cudnn warmup costs need to amortize. Warmup steps
  dropped from per-step mean: **first 5 steps**.
- Per-step trace via `JacobianODE/perf-experiment/perf_callback.py`
  (`PerStepTimingCallback`), wired in by
  `JacobianODE/perf-experiment/run_with_callback.py` (uses `hydra.compose`
  to bypass `@hydra.main` and inject the callback into `train_model`).
- Regression-test gate: `np.allclose(loss_new, loss_baseline, atol=1e-4, rtol=0)`
  on per-step training losses.
- Override bundle: as documented in `TASK.md` / `baseline.md`. Batch size
  fixed at 8 (Lorenz) / 4 (wmtask) due to the 4 GB GPU.
- `WANDB_MODE=offline` for all runs.

### Configs under test

| Config | Per-step baseline (max_steps=20, incl warmup, end-to-end wall-clock) |
|---|---|
| `lorenz_partial_additive_splitmode_p30_obsnoise001_nd75_init15_autodim__lc_sweep` | ~3.26 s/step |
| `wmtask_direct_sum_additive_splitmode_p30_perareaautodim__lc_x_obsnoisescale_sweep` | ~5.60 s/step |

End-to-end wall-clock includes process startup, hydra config resolution,
data loading, model init, and CUDA warmup. The numbers below are
**per-step steady-state** (warmup-5 dropped) measured by the callback.

## Baselines (per-step steady-state)

Two seed=42 runs per config, max_steps=25, reading per-step records via
`PerStepTimingCallback`. Note: wmtask uses `accumulate_grad_batches=4`,
so its 25 optimizer steps correspond to **100 forward/backward batches**;
Lorenz has accumulate=1, so 25 steps = 25 batches. The callback fires
per-batch (`on_train_batch_*`), so the per-step times below are
**per-batch** times. Reproducibility is bit-exact (max |Δloss|=0).

| Config | Per-batch (mean ± std, warmup-5 dropped) | Reproducibility (run1 vs run2) |
|---|---|---|
| `lorenz_partial_additive_splitmode_p30_obsnoise001_nd75_init15_autodim__lc_sweep` | 1.063 s ± 0.104 (run1), 1.056 s ± 0.083 (run2) | max \|Δloss\| = 0.000e+00, allclose@1e-4 ✓ |
| `wmtask_direct_sum_additive_splitmode_p30_perareaautodim__lc_x_obsnoisescale_sweep` | 0.967 s ± 0.030 (run1), 0.959 s ± 0.029 (run2) | max \|Δloss\| = 0.000e+00, allclose@1e-4 ✓ |

Traces:
- `runs/baseline-lorenz-seed42-run{1,2}.json`
- `runs/baseline-wmtask-seed42-run{1,2}.json`

For attempts below, the **baseline reference** is `run1` of each config.
Run-to-run timing variance is ~1% so report a candidate as a real win
only if delta is meaningfully larger than that.

## Attempt log

_To be populated as attempts run._

## Summary

_To be written at end-of-window._
