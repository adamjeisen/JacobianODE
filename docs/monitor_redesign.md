# Monitor redesign: stateless classification + orphan cleanup + idempotent re-analysis

## Problem

The sweep-automation control plane has accumulated compounding bugs whose
root cause is not any single piece — it's the architecture. Observable
symptoms we've hit:

- "9 terminal · 6 running" on a single group: `state.json` says a sweep
  is fully done, but SLURM is still processing tasks for it.
- Reports with missing figures: analysis ran while some wandb runs had no
  checkpoint dir, aborted partway.
- Preempt + requeue double-schedules: SLURM's `REQUEUE` mode and
  `monitor.resubmit_run` both try to recover, creating parallel jobs on
  different partitions.
- Sweeps never re-analyze: once the sentinel fires, a single analysis pass
  is frozen — later-completing runs never get reflected in the report.
- Orphan SLURM jobs churn after a sweep is declared complete.

The pattern behind these is three architectural mistakes:

1. **Sticky terminal classification.** Once
   `state.runs[k].terminal = True`, the bit never flips back. Makes sense
   for clean training, wrong for preempt-requeue cycles where wandb state
   oscillates `running → finished → running → finished`.
2. **One-shot sentinel/analysis.** Analysis runs exactly once per
   sentinel. Later data never triggers a re-analysis.
3. **Two parallel retry mechanisms.** SLURM's `REQUEUE` and
   `monitor.resubmit_run` both fire on preempt, creating duplicate jobs.

## Goals

- Kill the sticky-classification bug at the source.
- Stop the double-schedule at the source.
- Make analysis catch up with reality when new data arrives.
- Clean up orphan SLURM jobs when a sweep is declared complete.

## Principles

- **Classification is a pure function** of
  `(wandb state, SLURM state, checkpoint-dir existence)`, recomputed every
  monitor cycle. `state.runs[k]` becomes a display/debug cache, not an
  authoritative record.
- **SLURM REQUEUE + our resume infra is the only retry mechanism.**
  `monitor.resubmit_run` is deleted.
- **Orphan SLURM tasks are scancelled** when a sweep is declared complete.
- **Analysis re-runs automatically** on state change, idempotently.

## Phases

### Phase 1 — Stateless classification (core fix)

Rewrite the run-idx classification in `monitor.py` as a pure function:

```python
def classify(k, wandb_runs_matched, slurm_states, ckpt_base, expected, resolved_run):
    # Returns one of: pending, running, done_finished,
    # done_early_stopped, done_walltime, failed.
```

Decision order per `run_idx`:

- No wandb runs and no alive SLURM task → **pending**.
- Any alive SLURM task on `{array_id}_{k}` → **running**.
- Best wandb run is `finished` and ckpt dir exists → **done_finished**.
- wandb `crashed/failed` and convergence criterion met → **done_early_stopped**.
- wandb `crashed/failed` and walltime hit → **done_walltime**.
- Otherwise → **failed** (no automatic retry).

`state.json` schema stays backwards-compatible but the semantics change:
`classification` is always recomputed from scratch each cycle. `terminal`
is derived (`True` iff `classification.startswith("done_")`), not
persisted independently.

Sentinel condition becomes: **all `run_idx` in `done_*` AND no alive
SLURM task for any `{array_id}_{k}`**.

### Phase 2 — Orphan scancel on sentinel fire

When the sentinel fires, enumerate any SLURM tasks still alive for this
group's array(s) (from `expected["slurm_arrays"]`) and `scancel` them
before writing the `done.json`. This stops redundant training cycles and
prevents subsequent wandb state oscillation.

Phases 1 and 2 together form the first landing — they're tightly coupled.

### Phase 3 — Idempotent re-analysis (later)

After each analysis completes, record a "data fingerprint" —
`{run_id: best_ckpt_mtime}` — alongside the processed sentinel. Each
cycle, recompute the current fingerprint for the group. If it has grown
(new runs completed, new checkpoints saved) relative to the recorded
fingerprint, move the sentinel `processed/` → `done/` so the controller
re-dispatches analysis.

Debounce: only re-trigger if the fingerprint is stable for ≥1 cycle
(avoids re-running mid-save).

Defer until we've lived with Phases 1+2 and confirmed the lack of
automatic re-analysis is actually painful.

### Phase 4 — Retire `monitor.resubmit_run`

Delete the function and its call site in `monitor.py`. Drop the `attempts`
counter. Rely entirely on SLURM REQUEUE + our checkpoint-resume infra for
preempt recovery. If SLURM doesn't requeue a genuinely-failed task, it
stays `failed` and analysis skips it with a warning.

Defer until Phases 1+2 have run clean for at least one sweep cycle in the
wild.

## Out of scope

These stay as they are — they're correct, just downstream of the broken
control flow:

- Checkpoint resume (`save_last=True`, `wandb_run_id.txt`,
  `trainer.fit(ckpt_path=...)`).
- Extend-not-replace `expected.json` for re-submissions.
- SIGUSR2 walltime shutdown handler.
- `allow_val_change=True` on wandb `config.update`.
- `select_best_from_sweep` and `compute_per_run_lyapunov` filters for
  runs with missing checkpoint dirs.

## Risks

- **Downstream readers of `state.runs[k].terminal`.** Anywhere the old
  sticky bit was used authoritatively needs auditing. Grep and migrate.
- **Fingerprint false positives (Phase 3).** Checkpoint mtimes can bump
  for reasons other than new training progress (filesystem touches).
  Mitigation: per-cycle debounce + only trigger on checkpoint *count*
  increase, not mtime change alone.
- **Removing `resubmit_run` (Phase 4) eliminates a safety net.** Cases
  SLURM REQUEUE doesn't catch (genuine non-preempt failures) will stay
  `failed`. Accept this explicitly; analyze_sweep already handles
  per-run failures gracefully.

## Implementation sequence (Phases 1+2 specifically)

1. Audit `state.runs[k].terminal` readers in the codebase.
2. Extract `classify()` as a pure function; unit-test against synthetic
   `(wandb state, SLURM state, checkpoint)` triples.
3. Replace the sticky-promotion loop in `monitor.check_sweep` with a
   call to `classify()` for each `run_idx`.
4. Rewrite sentinel condition to use fresh classifications.
5. Add orphan-scancel just before `write_sentinel`.
6. Roll out; observe one sweep cycle end-to-end; iterate.
