# Task: speed up JacobianODE training without changing behavior

You are running unattended in the Claude Code sandbox on a separate branch
(`perf-experiment-claude`). Your goal is to reduce per-training-step wall-clock
time for two specific training configs **without changing the effective
computation**. You have an 8-hour budget. Read this whole file before doing
anything.

## Goal

For each of the two configs below, find optimizations that reduce per-step
wall-clock training time while keeping the loss trajectory identical (within
tolerance) for fixed-seed runs. Report findings and per-attempt deltas to
`JacobianODE/perf-experiment/REPORT.md`, updating it after every attempt.

## "Computation unchanged" — definition (option (a))

The user explicitly chose this level of equivalence:

> Behavioral equivalence: same loss curve within tolerance (~1e-4 relative or
> absolute, whichever is tighter) over N steps with fixed seed.

Concretely: for any optimization to be **accepted**, two same-seed training
runs (one with the optimization, one without) must produce per-step training
losses that match within `1e-4`. Any optimization that fails this is
**rejected** and reverted.

### Allowed (consistent with option (a))
- `torch.compile` (model wrap)
- `torch.backends.cudnn.benchmark = True` (kernel selection variation OK if
  loss matches within tolerance)
- Fused optimizers (`AdamW(fused=True)`)
- `channels_last` memory format (probably no-op here, conv-light)
- DataLoader tuning: `num_workers`, `pin_memory`, `persistent_workers`,
  `prefetch_factor`
- `non_blocking=True` H2D transfers
- Removing redundant CPU↔GPU sync points (e.g. `.item()` calls inside hot
  loops, dropping unnecessary `.cpu()` round-trips)
- Setup-time wins (eliminating redundant config resolution, lazy imports,
  etc.) — these don't show up in steady-state per-step timing but they
  reduce total walltime
- Caching/memoization of work that doesn't depend on the training step

### Forbidden (would change computation under (a))
- Changing `batch_size`, `learning_rate`, `optimizer` choice, loss
  formulation, `accumulate_grad_batches`, `gradient_clip_val`
- Mixed precision / autocast (fp16 / bf16) — Pascal GPU has no native fp16
  support anyway, this would be a regression *and* break numerics
- Changing the model architecture, layer counts, or any cfg.model.* values
- Skipping or reordering training steps
- Reducing the number of train/val batches per epoch (changes what the
  optimizer sees)
- Anything that requires updating dependencies via `uv add` (not in scope —
  stay within the existing locked env)

### When in doubt
**Prefer NOT making the change** unless you can show the regression test
passes. The user's stated default is "boring conservative option" — apply it.

## Hard prerequisites — do these first

### 1. Read `baseline.md`

It documents:
- Baseline wall-clock for both configs (pre-flight numbers, max_steps=20)
- Sandbox-specific constraints (4 GB GPU → batch_size=8 Lorenz / 4 wmtask,
  read-only engaging mount, hydra override quirks, FUSE remount staleness)
- The full hydra override bundle that works
- The trajectory.py read-only-mount lockfile fix that's already applied

### 2. Build a per-step measurement framework

Pre-flight only captured wall-clock + max_steps. You need per-step data
to enforce the regression test. Specifically:

- Write a Lightning callback (e.g. in
  `JacobianODE/perf-experiment/perf_callback.py`) that records `(step,
  loss, walltime_seconds)` for every training step and writes to a JSON
  sidecar at the end of training.
- Wire it into the training run via `train_model`'s existing
  `extra_callbacks: Optional[List[L.Callback]]` parameter
  (`JacobianODE/jacobians/training/trainer.py:153`). You can do this by
  writing a small Python wrapper script in `JacobianODE/perf-experiment/`
  that imports `_run_training` and calls it with the callback added —
  bypassing the `@hydra.main` decorator via `hydra.compose()`.
- Keep `WANDB_MODE=offline` so wandb's code path runs (preserves timing
  representativeness) but skips network sync. Don't try to extract
  loss from the wandb binary file — it's painful.

### 3. Re-establish baselines with this framework

Run each config twice with `seed=42` and capture the per-step trace.
Verify the two runs match within `1e-4` per-step — that proves seed
determinism works on this hardware. If they don't match, fix that
*before* attempting any optimization (it could be cudnn nondeterminism;
set `torch.use_deterministic_algorithms(True)` and reseed).

Save baseline traces as
`JacobianODE/perf-experiment/runs/baseline-<config>-seed42-{run1,run2}.json`.

### 4. Decide on the measurement schedule

For each subsequent attempt, run with `max_steps` large enough that
per-step time is stable (drop the first ~5 warmup steps from the average).
Likely `max_steps=50–80` is enough. Pick a number, document it in
REPORT.md, and stick with it across all attempts so timings are
comparable.

## Process for each optimization attempt

1. **Hypothesis** — write 1–2 sentences in REPORT.md naming the change and
   the reason you expect a speedup.
2. **Implement** — make the change. Prefer narrow, targeted edits (the
   user's CLAUDE.md is explicit about this).
3. **Measure** — run with the new code, fixed seed, capture per-step
   trace.
4. **Regression check** — compare per-step loss against the baseline
   trace. Use `numpy.allclose(losses_new, losses_baseline, atol=1e-4)`
   or equivalent.
5. **Decision**:
   - **Pass + faster**: commit the change. REPORT.md gets:
     ```
     ### Attempt N: <description> — KEPT (-X% per-step time)
     ...
     ```
   - **Pass but no speedup (or slower)**: revert. REPORT.md notes "no
     improvement" and moves on.
   - **Fail (loss diverged)**: revert. REPORT.md notes the divergence
     magnitude and what the change touched. Do NOT try to "fix" the
     divergence by relaxing the tolerance — the tolerance is a hard gate.
6. **Commit** — every accepted change is its own commit on
   `perf-experiment-claude`. Commit message: `perf: <short description> —
   <delta>%`. If reverted, no commit (just REPORT.md update).

## Configs to optimize

Both are pre-existing experiment YAMLs. Don't modify the YAMLs themselves
— use hydra CLI overrides, OR (if a setup change like enabling
`cudnn.benchmark` is needed) make the code change at the relevant call
site, not in the experiment YAML.

| Config | Per-step baseline (incl warmup, max_steps=20) |
|---|---|
| `lorenz_partial_additive_splitmode_p30_obsnoise001_nd75_init15_autodim__lc_sweep` | ~3.26 s |
| `wmtask_direct_sum_additive_splitmode_p30_perareaautodim__lc_x_obsnoisescale_sweep` | ~5.60 s |

### Override bundle (use this for every measurement run)

```
experiment=<config>
+training.trainer_params.max_steps=<N>          # the + is required
training.trainer_params.limit_train_batches=200
training.trainer_params.limit_val_batches=10
training.batch_size=<8 lorenz | 4 wmtask>       # GPU memory constraint
training.logger_save_dirs=/tmp/perf-exp/<config>/<runid>/lightning
training.logger.save_dir=/tmp/perf-exp/<config>/<runid>/lightning
data.flow.random_state=42
hydra.run.dir=/tmp/perf-exp/<config>/<runid>/hydra
```

Plus env: `WANDB_MODE=offline`.
Plus shell: `uv run --no-sync ...` (the cu118 guard hook blocks bare `uv run`).

## Seed ideas to try (not exhaustive, not prescriptive)

In rough order of "likely cheap, likely safe":

1. DataLoader: `num_workers > 0` (currently default 0?), `pin_memory=True`,
   `persistent_workers=True`. Check what's in `cfg.training` and the
   dataloader factory at `JacobianODE/jacobians/data/`.
2. `torch.backends.cudnn.benchmark = True` (set at process start for
   fixed input shapes).
3. Fused AdamW: pass `fused=True` to the optimizer. Look up where
   AdamW is constructed.
4. `non_blocking=True` on `.to(device)` calls — but only useful with
   `pin_memory`.
5. `torch.compile(model)` — the big one if it works. May need
   `dynamic=True` for variable-length inputs. Compilation cost amortizes
   over many steps so report measured at `max_steps=80+` to capture
   amortized savings.
6. Look for redundant `.item()` / `.cpu()` calls in the training step
   (they force CPU-GPU sync). `latent_jacobian.py` is a likely
   suspect since it does Jacobian computation per step.
7. `cudnn.deterministic = False` (default in PyTorch ≥ 1.x is False, but
   verify). Note: if your baselines pass regression with `deterministic
   = True`, leaving it True is fine; just don't set it True if it isn't.

Don't feel bound to this list. If you spot a hot path in profiling,
attack it directly. `torch.profiler` is available.

## Time budget

- 8 hours total.
- First ~30 min: prereqs (read baseline.md, build callback, re-baseline).
- 30 min to 7h30 min: optimization attempts.
- Last 30 min: write final summary section in REPORT.md, ensure all
  accepted changes are committed and pushed.

If you hit Anthropic usage limits before the 8h window closes, stop
gracefully: write current state to REPORT.md, commit, push. The next
session can resume from REPORT.md.

## Reporting

`REPORT.md` is your primary output and lives at
`JacobianODE/perf-experiment/REPORT.md`. It must include, in order:

1. **Run metadata** — start time, config of the measurement runs (max_steps,
   batch_size, seed, etc.), the GPU, hardware notes.
2. **Baselines** — per-config per-step time (mean ± std after dropping
   warmup), with paths to the baseline traces.
3. **Attempt log** — chronological. Each entry:
   - Hypothesis
   - Change (file paths + summary)
   - Measurement (per-step time delta, regression test result)
   - Decision (kept / rejected / no-improvement)
4. **Summary** (written at end-of-window) — total cumulative speedup per
   config, list of accepted changes, list of rejected changes with one-line
   reasons.

Update REPORT.md *after every attempt*, not at the end. The user wants to
read it while you're still running.

## Commits

- One commit per accepted change. Message: `perf: <description> — <delta>%`.
- One commit at start with the per-step measurement framework (callback +
  wrapper script + baselines).
- A final commit at the end with the summary section in REPORT.md.
- All commits go on `perf-experiment-claude`. Push periodically (every
  hour or so) so the user can see incremental progress.

## Resilience

- **Don't get stuck on one attempt.** If something doesn't work after 30
  min of debugging, revert and move on. Note the failure in REPORT.md.
- **Don't widen the tolerance** to make a failing attempt pass.
- **Don't change the configs** to make timing easier (that's gaming the
  metric, not optimizing).
- **If you hit a sandbox limitation** (e.g. another OOM at a smaller
  batch size, FUSE issue, etc.) — write it up in REPORT.md and skip
  attempts that hit it.
- **Don't `uv add` new packages.** Stay in the locked env.
- **The `bare uv run` cu118 guard hook is active**: always use `uv run
  --no-sync ...`. Hook will block you otherwise.

## Definition of done

At t=8h:
- `REPORT.md` has a final "Summary" section
- All accepted changes are committed on `perf-experiment-claude`
- Branch is pushed to `origin/perf-experiment-claude`
- Working tree is clean (no uncommitted scratch)

If you finish early (find no more wins), stop early. Quality > duration.
