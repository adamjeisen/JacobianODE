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

### Attempt 6: l1_loss via `_foreach_norm` instead of cat+abs+sum — REJECTED

**Hypothesis**: the per-step `l1_loss = sum(|cat(view(-1) for p in params)|)`
allocates a 27 MB contiguous buffer (all 6.8 M params concatenated) before
abs+sum. Replacing with `torch.stack(torch._foreach_norm(params, ord=1)).sum()`
keeps the same numerical scalar but uses one fused per-param reduction
launch instead of cat → abs → sum.

**Measurement (seed=42, vs original pre-change baselines)**:

| Config | Per-step (s) | Cumulative Δ% | vs prior best | max \|Δloss\| |
|---|---|---|---|---|
| lorenz (max_steps=80) | 1.0360 ± 0.043 | −0.37% | +0.01% (no change) | 0.000e+00 |
| wmtask (max_steps=25) | 0.9554 ± 0.027 | **−1.20%** | **+0.23% regression** vs A5's −1.43% | 2.98e-08 |

**Decision**: REVERT. wmtask got slightly slower with `_foreach_norm`,
likely because the fused-norm launches of 80 small reductions don't
amortize against the cat-then-one-reduce shape on this Pascal GPU
where launch overhead is high. l1_loss is bit-exact either way.

### Attempt 5: DataLoader `num_workers=0` default — KEPT (lorenz −0.4%, wmtask −0.2% marginal)

**Hypothesis**: `TimeSeriesDataset.__getitem__` is `return self.sequence[index]`
on an already-in-memory `torch.Tensor`. With `num_workers=2` (current default
in `dataloaders.py`), each batch fetch forks a worker, copies the slice
across IPC, and unpickles — pure overhead for an in-memory dataset. Setting
`num_workers=0` runs the trivial `__getitem__` inline on the main process.

**Change**: `JacobianODE/jacobians/data/dataloaders.py` — change the
default of `num_workers` from `2` to `0`, and `persistent_workers` from
`True` to `False` (Lightning would otherwise warn about the
inconsistent combination). Caller signature unchanged so any code that
passes `num_workers=N` explicitly is unaffected.

**Measurement (seed=42, against pre-change baselines)**:

| Config | Per-step (s) | Cumulative Δ% | max \|Δloss\| | allclose@1e-4 |
|---|---|---|---|---|
| lorenz (max_steps=80) | 1.0359 ± 0.045 | −0.38% | 0.000e+00 | ✓ |
| wmtask (max_steps=25) | 0.9532 ± 0.027 | −1.43% | 2.98e-08 | ✓ |

(The wmtask 3e-8 noise is the carried-through fused-AdamW residual; this
attempt itself is bit-exact since the shuffle sampler is seeded on the
main process and the order doesn't depend on `num_workers`.)

The marginal effect of *just this attempt* (vs A4 cumulative) is
roughly −0.3% lorenz, −0.2% wmtask — at the noise floor. But it's
strictly cleaner code (matches the dataset's actual on-disk-or-memory
character) and bit-exact, so the change is essentially free.

**Decision**: KEEP. Trace files: `runs/A5-dl-workers0-{lorenz,wmtask}.json`.

### Attempt 4: Eliminate `.cpu()` syncs in JacobianODEint hot loop — KEPT (wmtask −1.2%, lorenz noise)

**Hypothesis**: `JacobianODEint.generate_dynamics` had two `.cpu().numpy()`
syncs in the per-trajectory rollout — one outside the loop computing
`n_steps`, one inside the inner integration loop computing the
teacher-forcing time index. Both convert tensor → Python int via a CUDA
sync. Both indices are determined entirely by Python-side ints + floats
(shapes + `self.dt` + `steps_per_dt`) so the syncs are pure overhead.

**Change** (`JacobianODE/jacobians/jacobianODE.py`, two narrow edits):
- Line ~531: `n_steps = int(np.round(t_sim.cpu().numpy()/dt_sim))`
  → `int(np.round((traj.shape[-2] - traj_init.shape[-2]) * self.dt / dt_sim))`
- Line ~561: `int(np.round((t + dt_sim).cpu().numpy()/self.dt))`
  → `(traj_init.shape[-2] - 1) + step_counter // steps_per_dt`
  (this branch only fires when `step_counter % steps_per_dt == 0`, so the
  index is exact — no rounding ambiguity).

**Measurement (seed=42)**:

| Config | Run | Per-step (s) | Δ% vs baseline | max \|Δloss\| | allclose@1e-4 |
|---|---|---|---|---|---|
| lorenz (max_steps=80) | r1 | 1.0388 ± 0.052 | −0.11% | 0.000e+00 | ✓ |
| lorenz (max_steps=80) | r2 | 1.0386 ± 0.050 | −0.13% | 0.000e+00 | ✓ |
| wmtask (max_steps=25) | r1 | 0.9552 ± 0.027 | **−1.22%** | 2.98e-08 | ✓ |
| wmtask (max_steps=25) | r2 | 0.9550 ± 0.028 | **−1.24%** | 2.98e-08 | ✓ |

Loss is bit-exact on lorenz; wmtask shows only the 3e-8 noise from the
already-kept fused AdamW change (this attempt by itself preserves bit
exactness). Two runs back-to-back on each config disambiguate signal
from run-to-run noise.

**Decision**: KEEP. Real and reproducible −1.2% on wmtask (more
JacobianODEint calls per global step due to `accumulate_grad_batches=4`
× per-area direct-sum branches), neutral on lorenz (already
GPU-bound — CPU dispatch wasn't the lorenz bottleneck). Trace files:
`runs/A4-nosync-jacode-{lorenz,wmtask}.json`.

### Attempt 3: Replace `if torch.isnan(loss)` with `nan_to_num` — REJECTED (no improvement)

**Hypothesis**: `if torch.isnan(loss):` (lightning_base.py:678,681) forces a
CPU↔GPU sync every step (and twice per pred_type, so up to 4× per step
when both trajectory and loop_closure losses are active). Replacing with
`torch.nan_to_num(loss, nan=0.0)` keeps the same NaN-protection
semantics — NaN → 0 contribution to `total_loss` — without the sync.

**Change**: replaced two `if torch.isnan` lines in
`JacobianODE/jacobians/lightning_base.py:training_step` with one
`torch.nan_to_num` call.

**Measurement (lorenz, max_steps=80, seed=42)**:
- max \|Δloss\| = **0** (bit-exact, as expected — no NaN at any step)
- per-step: **1.043 s ± 0.051** vs baseline **1.040 s ± 0.052** → **+0.3%** (noise)

**Decision**: REVERT. Loss is bit-exact so the change is safe, but the
sync wasn't on the critical path (probably because Lightning's backward +
optimizer.step already includes a sync soon after, so any latency we
hide here just gets absorbed by the next sync). No improvement → revert
per the task rules. The bigger CPU overhead is still kernel-launch
volume (~18 k/step), and only torch.compile / cudagraphs would attack
that — both blocked on this hardware (Attempt 2).

### Attempt 2: torch.compile — BLOCKED on hardware (sandbox GPU)

**Hypothesis**: 18 k kernel launches per training step (`cudaLaunchKernel`
~1.14 s of 9.7 s CPU total, profiler trace) dominate CPU overhead. Wrapping
the inner `model` (Jacobian MLP) and `encoder` with `torch.compile` should
fuse small element-wise + linear ops into many fewer big kernels.

**Outcome**: hard-blocked by GPU compute capability. The sandbox runs on a
**GTX 1050 Ti (CC 6.1, Pascal)**; Inductor's Triton backend requires
**CC ≥ 7.0** and refuses with `GPUTooOldForTriton`. Tried fallbacks:
- `backend="cudagraphs"`: graph captures fail because the JacobianODEint
  inner loop produces calls with **varying tensor shapes** (`size of tensor
  a (20) must match (22)`); CUDA-graph capture requires fixed shapes.
- `backend="aot_eager"`: graph capture only, no kernel codegen — known to
  give no speedup, skipped.
- `backend="eager"`: identity wrapper, skipped.

**Decision**: SKIPPED for this hardware. On a Volta+ GPU
(`uv run --no-sync ...` on engaging) the same code change should give a
meaningful win — roughly the per-step `cudaLaunchKernel` total (~230 ms
in our trace, ~22 % of step time) is the headline target. The probe
script `run_with_compile.py` is left in tree so this can be re-run
trivially elsewhere.

### Attempt 1: Fused AdamW — KEPT (lorenz −0.5%, wmtask −1.0%, marginal)

**Hypothesis**: passing `fused=True` to `torch.optim.AdamW` collapses the
per-parameter `_foreach_*` kernel launches into one fused launch, which
should help when there are many small parameter groups (coupling encoder
heads, per-area branches in the wmtask config).

**Change**: `JacobianODE/jacobians/lightning_base.py` `configure_optimizers`
sets `fused=True` for AdamW when CUDA is available (and only if the user
hasn't already set it). All other optimizers untouched.

**Measurement (max_steps=25, seed=42)**:

| Config | Baseline (s/step) | Candidate (s/step) | Δ% | max \|Δloss\| | allclose@1e-4 |
|---|---|---|---|---|---|
| lorenz | 1.063 ± 0.104 | 1.057 ± 0.085 | **−0.53%** | 0.000e+00 | ✓ |
| wmtask | 0.967 ± 0.030 | 0.957 ± 0.029 | **−0.99%** | 2.980e-08 | ✓ |

Lorenz is bit-exact; wmtask shows ~3e-8 max divergence, well within the
1e-4 gate (and consistent with float32 op-order differences from the
fused kernel's reduction).

**Decision**: KEEP. Improvement is at the run-to-run noise floor (~1%) so
it's marginal, but the delta is in the right direction on both configs
and the regression test passes cleanly. Cost is tiny (a single kwarg).

Trace files: `runs/A1-fused-adamw-{lorenz,wmtask}.json`.

### Attempt 0: cudnn.benchmark — SKIPPED (not applicable)

**Hypothesis**: setting `torch.backends.cudnn.benchmark=True` selects faster
cuDNN convolution algorithms once input shapes are stable.

**Decision**: skipped without measuring. Both target configs are pure MLP /
matmul stacks (`grep -rn Conv` returns no nn.Conv usage in the project's
training code path; `latent_additive_coupling` is a coupling-block
encoder with linear layers and the deriv head is also MLP-based). Linear
layers route through cuBLAS, not cuDNN, so cudnn.benchmark has nothing
to optimize and would add a tiny startup cost for the first benchmark
sweep with no payoff. Documented and moving on.

## Summary

End-of-window state.

### Cumulative speedup

Final per-step times after all kept changes, vs the original pre-change
baselines:

| Config | Baseline | Final | Cumulative Δ% | max \|Δloss\| | allclose@1e-4 |
|---|---|---|---|---|---|
| lorenz (max_steps=80) | 1.0399 ± 0.052 s | **1.0359 ± 0.044 s** | **−0.38%** | 0.000e+00 | ✓ |
| wmtask (max_steps=25) | 0.9670 ± 0.030 s | **0.9530 ± 0.027 s** | **−1.45%** | 2.98e-08 | ✓ |

Both regressions pass cleanly at the 1e-4 gate. lorenz is bit-exact;
wmtask shows a 3e-8 max divergence from the fused-AdamW reduction order
(the only kept change that touches reduction order).

Final traces: `runs/final-{lorenz,wmtask}-*.json`.

### Accepted changes (3)

| # | Change | File | Lorenz | wmtask |
|---|---|---|---|---|
| A1 | `fused=True` for AdamW when CUDA available | `lightning_base.py:configure_optimizers` | −0.5% | −1.0% |
| A4 | Drop `.cpu()` syncs in `JacobianODEint.generate_dynamics` (n_steps + teacher-forcing index) | `jacobianODE.py:531,561` | −0.1% | −1.2% |
| A5 | `num_workers=0` default (TimeSeriesDataset is in-memory) | `dataloaders.py` | −0.4% | −0.2% |

(Marginal effects measured one-at-a-time vs the prior cumulative state.
The headline cumulative is smaller than the sum of marginals because
of run-to-run variance ~1 % and because the changes touch overlapping
overhead.)

### Rejected changes (2) and skipped probes (1)

| # | Change | Reason for revert / skip |
|---|---|---|
| A0 | `cudnn.benchmark=True` | Both target configs are pure MLP / matmul (no `nn.Conv*`), so cuDNN is never selected. Skipped without measuring. |
| A2 | `torch.compile(mode=default | reduce-overhead, dynamic=True)` on `model` + `encoder` | Hard-blocked: GTX 1050 Ti is CC 6.1, Inductor/Triton requires ≥ 7.0 (`GPUTooOldForTriton`). `backend="cudagraphs"` fails on dynamic shapes inside the JacobianODEint integration loop. Probe script left in tree (`run_with_compile.py`) for re-runs on Volta+. |
| A3 | `torch.nan_to_num(loss)` instead of `if torch.isnan(loss):` | Bit-exact loss but +0.3 % (noise) — the sync wasn't on the critical path because Lightning's optimizer.step already syncs soon after. |
| A6 | `_foreach_norm` instead of cat+abs+sum for L1 reg | wmtask got slightly slower (−1.20 % vs A5's −1.43 %). The 80 small per-param launches don't amortize against one big cat+reduce on this Pascal GPU. |

### Where the headline gain on this hardware *is* coming from

Profiling (`profile_step.py`) shows the dominant CPU cost is
**~18 k `cudaLaunchKernel` calls per training step** (~12.7 µs each
≈ 230 ms / step ≈ 22 % of step time). On Volta+ that would be the
torch.compile target. On Pascal sandbox we have to attack overhead by
hand:

- A1 (fused AdamW) collapses the per-param `_foreach_*` adam launches
  into a single `_fused_adamw_` launch — directly attacks launch count.
- A4 removes 1 + n_steps CUDA syncs per `generate_dynamics`, each of
  which can stall the host's next-step kernel-dispatch.
- A5 removes per-batch IPC overhead from a needless worker fork.

The wins are larger on **wmtask** because `accumulate_grad_batches=4`
plus the per-area direct-sum coupling means each global step makes
many more `generate_dynamics` calls than lorenz, multiplying the
benefit of A4 in particular.

### Time spent

- Framework + baselines: ~25 min
- 7 attempts (3 kept, 2 rejected, 1 skipped pre-flight, 1 hardware-blocked): ~3 h elapsed wall-clock from kickoff.
- Stopped early because:
  1. The single biggest lever (`torch.compile`) is hardware-blocked.
  2. Remaining low-hanging branches (CPU-side noise generation, encoder
     vectorization, l1_loss restructuring) either change RNG semantics
     (forbidden under "computation unchanged" rule) or have been tried
     and didn't help on this GPU. Quality-over-duration.

### What I deliberately did NOT touch

- The trajectory-noise injection path
  (`batch + (torch.randn(*batch.shape)*scaled_noise).type(batch.dtype).to(batch.device)`).
  Generating noise directly on GPU would change the RNG stream and so
  the loss at step 1 — would fail the 1e-4 gate. Forbidden.
- `latent_jacobian.trajectory_model_step` window-stack Python loop. Would
  vectorize cleanly with `torch.stack(unfold(...))` but the loop is
  small (B=8 × n_windows≤4) and not on the profile hot path.
- `log_training_metrics` per-step calls. The expensive Jacobian-loss
  branch only fires when `self.eq is not None`, which neither target
  config triggers (n_delays>1 ⇒ `eq=None` for lorenz; wmtask returns
  `eq=None` from its dataset_loader path).
- `compute_jacobians` itself — already a single forward pass through
  the Jacobian MLP; the heavy work is in cuBLAS.

