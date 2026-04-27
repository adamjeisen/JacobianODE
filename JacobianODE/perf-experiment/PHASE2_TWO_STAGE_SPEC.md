# Spec: Two-stage sweep protocol

**For**: another Claude (or human) implementing on `latent-JacobianODE`.

**Goal**: Cut sweep wall-clock by ~50 % by training all configs to a short
"shortlist" budget (Stage A), culling the bottom 50 %, and resuming the top
50 % to convergence (Stage B). Empirically validated on 71 historical sweeps
(see `JacobianODE/perf-experiment/analysis/two_stage_out/summary.md`):

| Stage A epoch K | Cull bottom | top-1 survives | top-3 median survive |
|---|---|---|---|
| 20 | 50 % (default) | **65/71 = 92 %** | 3.0/3 |
| 20 | 33 % (safer) | **71/71 = 100 %** | 3.0/3 |

Default proposal: **K=20, top-50 % cull**. The remaining 8 % failure mode
is "the eventual top-1 was in the bottom half at K=20 and got culled" —
acceptable given that across 71 historical sweeps, even when the top-1
was killed it was usually replaced by a near-tie config (top-3 still all
survived in the median case). Make `--cull-fraction` a CLI flag so users
can tune.

This spec deliberately scopes implementation to **one new Python module +
two ~10-line edits to existing files**. No changes to engaging-controller,
no new state machine, no new SLURM dependency graph. The cost is one
manual `python -m ... two_stage_cull` invocation between Stage A and
Stage B; the benefit is that each stage is a regular existing-shape sweep.

## Protocol (user-facing)

```bash
# Stage A: regular sweep with reduced max_epochs.
j-submit lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep \
    training.trainer_params.max_epochs=20 \
    wandb_group=lorenz_..._lc_sweep__stage_a

# Wait for Stage A to finish (engaging-controller does its usual thing —
# the sweep finishes, gets its analysis sentinel, etc.).

# Cull + dispatch Stage B:
python -m JacobianODE.jacobians.tuning.two_stage_cull \
    --group lorenz_..._lc_sweep__stage_a \
    --full-max-epochs 200 \
    --cull-fraction 0.5

# That command:
#   1. queries W&B for the Stage A group's runs
#   2. ranks by best-so-far trajectory val_loss
#   3. picks survivors (top 50%)
#   4. for each survivor, finds its Stage A last.ckpt
#   5. emits one j-submit instruction per survivor with explicit overrides
#      + +training.ckpt_path=<survivor's stage-a last.ckpt>
#   6. logs which configs were culled vs kept (writes a survivors.json
#      next to the Stage A analysis report)
```

Each Stage B instruction is a regular instruction YAML — engaging-controller
treats them like any other sweep submission. Stage B's runs share
`wandb_group=..._stage_b` so analysis pipeline produces one Stage-B report
covering only survivors.

## Required code changes

Three localized changes. All on `latent-JacobianODE`.

### 1. Stable per-cell checkpoint path (`JacobianODE/jacobians/training/trainer.py`)

The existing `_resume_state_dir(cfg)` keys on `SLURM_ARRAY_JOB_ID +
SLURM_ARRAY_TASK_ID`, which differs between Stage A and Stage B SLURM
arrays. Add a complementary stable path that keys on the *swept config
cell*, so Stage B can find Stage A's checkpoint regardless of SLURM IDs.

```python
def _two_stage_ckpt_path(cfg) -> Optional[Path]:
    """Stable last-ckpt path that survives across SLURM arrays.

    Keyed on (wandb_group, hash-of-swept-config). Identical config in a
    different array hits the same path → Stage B resumes from Stage A.
    """
    base = cfg.training.get("logger_save_dirs")
    group = cfg.get("wandb_group")
    if not base or not group:
        return None
    cell_key = _compute_cell_key(cfg)
    return Path(base).parent / "two_stage_ckpts" / group / cell_key / "last.ckpt"


def _compute_cell_key(cfg) -> str:
    """Hash of the cfg fields that differ across cells in a sweep grid.
    Robust to harmless config diffs (timestamps, SLURM IDs, etc.).

    Implementation: read prepare_sweep's `expected.json` if present (it
    has `resolved_runs[i].overrides` for each cell — perfect cell key).
    Fallback: hash the entire config minus a known blocklist of keys.
    """
    # See implementation notes below.
    ...
```

**Save**: in `train_model`, after `trainer.fit(...)` returns cleanly, copy
the saved `last.ckpt` from `resume_dir` (existing) into the two-stage path
*if `wandb_group` ends in `__stage_a`* (or any condition the controller
sets — see Decision 4 below). Optionally make this conditional on a new
`cfg.training.save_two_stage_ckpt` flag if you'd rather not auto-copy on
every Stage A.

### 2. `cfg.training.ckpt_path` Hydra override (`run_jacobians.py` + `trainer.py`)

Currently `train_model` derives `ckpt_path_resume` only from
`_resume_state_dir(cfg)`. Add precedence:

```python
# In train_model, before the existing resume block:
ckpt_path_resume: Optional[str] = cfg.training.get("ckpt_path", None)
# Existing logic only fires if the explicit override didn't already set it.
if ckpt_path_resume is None and resume_dir is not None:
    last_file = resume_dir / "last.ckpt"
    ...
```

This is the path Stage B uses. The cull tool emits
`+training.ckpt_path=/abs/path/to/cell-key/last.ckpt` per survivor.
The `+` is needed because `ckpt_path` isn't in `training.yaml` by default —
do NOT add it to the YAML default (keeps the field empty / opt-in).

### 3. Cull tool (`JacobianODE/jacobians/tuning/two_stage_cull.py`, new file)

```python
"""Pick survivors from a Stage-A sweep group + dispatch Stage B."""

import argparse, json, subprocess, hashlib
from pathlib import Path
import wandb

# Where Stage-A last.ckpt files live; must match _two_stage_ckpt_path.
TWO_STAGE_ROOT = Path("/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps/two_stage_ckpts")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="Stage-A wandb_group")
    ap.add_argument("--project", default=None,
                    help="auto-detected from one run if omitted")
    ap.add_argument("--cull-fraction", type=float, default=0.5)
    ap.add_argument("--full-max-epochs", type=int, required=True,
                    help="max_epochs for Stage B (typically the original)")
    ap.add_argument("--metric", default="trajectory val_loss")
    ap.add_argument("--stage-b-suffix", default="__stage_b")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = wandb.Api()
    runs = list(_get_runs(api, args.group, args.project))
    survivors = _pick_survivors(runs, args.metric, args.cull_fraction)
    print(f"culled {len(runs) - len(survivors)} / {len(runs)} runs")

    stage_b_group = args.group + args.stage_b_suffix
    survivors_meta = []
    for r in survivors:
        # Reconstruct the swept overrides for this cell.
        overrides = _swept_overrides_from_config(r.config)
        # Locate the Stage A ckpt by recomputing the cell key.
        cell_key = _compute_cell_key_from_overrides(r.config, args.group)
        ckpt = TWO_STAGE_ROOT / args.group / cell_key / "last.ckpt"
        if not ckpt.is_file():
            print(f"WARN: no ckpt at {ckpt}; skipping {r.id}")
            continue
        # Build Stage-B overrides
        ov = list(overrides) + [
            f"training.trainer_params.max_epochs={args.full_max_epochs}",
            f"wandb_group={stage_b_group}",
            f"+training.ckpt_path={ckpt}",
        ]
        survivors_meta.append({"run_id": r.id, "overrides": ov, "ckpt": str(ckpt)})

    # Write survivors.json (audit trail)
    out = Path(f"/tmp/two_stage_survivors_{args.group}.json")
    out.write_text(json.dumps({"group": args.group, "survivors": survivors_meta}, indent=2))
    print(f"wrote {out}")

    if args.dry_run:
        print("(dry-run: no j-submit fired)")
        return

    # Dispatch one j-submit per survivor. Each is a regular sweep submission
    # with the per-cell overrides flattened — no sweep_grid mutation needed.
    exp = _experiment_from_run(survivors[0])
    for s in survivors_meta:
        cmd = ["j-submit", exp] + s["overrides"]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)


def _pick_survivors(runs, metric, cull_fraction):
    # Best-so-far metric per run from history
    scored = []
    for r in runs:
        best = float("inf")
        try:
            for row in r.scan_history(keys=[metric]):
                v = row.get(metric)
                if v is None:
                    continue
                v = float(v)
                if v < best:
                    best = v
        except Exception:
            continue
        if best != float("inf"):
            scored.append((best, r))
    scored.sort(key=lambda kv: kv[0])
    keep_n = max(1, int(round(len(scored) * (1 - cull_fraction))))
    return [r for _, r in scored[:keep_n]]


def _swept_overrides_from_config(cfg) -> list[str]:
    """Reconstruct the override CLI args that produced this run's swept-cell.
    Read from cfg.sweep_grid (resolved at run time); the resolved values
    are the ones passed via Hydra override."""
    sg = cfg.get("sweep_grid", {}) or {}
    out = []
    for key, _ in sg.items():
        v = _resolve_dotted(cfg, key)
        if v is not None:
            out.append(f"{key}={v}")
    return out


def _compute_cell_key_from_overrides(cfg, group):
    """Same hash that trainer.py:_compute_cell_key produces."""
    sg = cfg.get("sweep_grid", {}) or {}
    pairs = sorted([(k, _resolve_dotted(cfg, k)) for k in sg.keys()])
    blob = json.dumps(pairs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _resolve_dotted(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def _get_runs(api, group, project):
    if project is None:
        # Pick from any run in the group (uses default page, then narrows)
        for p in api.projects(entity="JacobianODE"):
            try:
                rr = list(api.runs(f"JacobianODE/{p.name}", filters={"group": group}, per_page=1))
                if rr:
                    project = p.name
                    break
            except Exception:
                continue
    return api.runs(f"JacobianODE/{project}", filters={"group": group}, per_page=200)


def _experiment_from_run(run):
    return run.config.get("metadata", {}).get("experiment") or run.group  # best-effort


if __name__ == "__main__":
    main()
```

**Key invariant**: `_compute_cell_key_from_overrides` (in cull tool) and
`_compute_cell_key` (in trainer) must produce the same hash for the same
swept-cell config. Test this with a unit test on a saved expected.json.

## Open decisions (read these before coding)

1. **Cull metric**: default `trajectory val_loss`, since that's what the
   71-group simulation used. **Should this be the user's choice?**
   Some users may want to cull on Jacobian R² or `val/recon_loss`.
   *Recommendation*: support `--metric` flag, default `trajectory val_loss`.

2. **Cull fraction**: default 0.5 (50 % cull). The 92 % top-1 survival
   in the simulation was on this setting. **Are you OK with the 8 %
   "lost the top-1" failure mode?** If not, default to 0.33.
   *Recommendation*: parameterize, default 0.5, document the trade-off.

3. **Stage A budget K**: pin to 20 epochs, or scale with the original
   `max_epochs`? Sweeps with `max_epochs=200` may benefit from K=30;
   sweeps with `max_epochs=80` could use K=15.
   *Recommendation*: `--stage-a-epochs N` with default 20.

4. **Auto-save Stage A ckpt vs explicit flag**: should *every* run save
   to the two-stage path, or only when `wandb_group` ends in `__stage_a`
   (or `cfg.training.save_two_stage_ckpt=true`)?
   - Auto: simpler for users. Disk cost: 1 last.ckpt per run, already
     created by ModelCheckpoint, just an extra cp.
   - Flag: explicit, keeps non-stage-a sweeps from leaving extra files.
   *Recommendation*: auto-save. The `last.ckpt` already exists; we're
   just symlinking/copying it to a deterministic name. Cleanup tool can
   prune `two_stage_ckpts/<group>/` when the analysis report is finalized.

5. **Cell key derivation**: hash of `sweep_grid` resolved values, or read
   `expected.json`'s `resolved_runs[i].overrides`?
   - Hash from `cfg.sweep_grid`: works at runtime without filesystem
     access; portable. Risk: must be deterministic across processes.
   - Read from `expected.json`: ground truth. Risk: requires controller-
     written file to exist when train_model runs, and a way to map
     `run_idx` → cell.
   *Recommendation*: hash from `cfg.sweep_grid` — simpler, portable.
   Verify with a unit test.

6. **Resume vs restart for survivors**: spec assumes resume from Stage A
   ckpt. **Verify the Lightning ckpt preserves**:
   - optimizer state (Adam moments)
   - LR scheduler state (cosine epoch counter)
   - alpha_teacher_forcing (it's a Python float on the LightningModule —
     does `on_save_checkpoint` save it? quick check: lightning_base.py)
   If alpha doesn't survive the ckpt, the schedule restarts at 1.0
   in Stage B, which is wrong. *Action*: write a test that loads a
   ckpt and verifies `pl_module.alpha_teacher_forcing` is preserved.

7. **What if Stage A run failed?** The cull tool should skip runs in
   non-`finished` state, and skip survivors whose `last.ckpt` doesn't
   exist (with a printed warning). The `survivors.json` should record
   skip reasons.

## Files to touch

```
JacobianODE/jacobians/training/trainer.py            ← +_two_stage_ckpt_path,
                                                        +cfg.training.ckpt_path read
JacobianODE/jacobians/run_jacobians.py               ← (no change if trainer.py
                                                        owns the cfg.training.ckpt_path read)
JacobianODE/jacobians/tuning/two_stage_cull.py       ← NEW
JacobianODE/jacobians/tuning/__init__.py             ← export the new module
JacobianODE/jacobians/conf/training/training.yaml    ← (no change — ckpt_path is +-only)
```

No changes needed to:
- `bin/engaging-controller`
- `bin/j-submit` (calls existing one)
- `tuning/prepare_sweep.py`, `tuning/monitor.py`, `tuning/render_report.py`
- experiment YAMLs

## Testing

### Unit tests
- `_compute_cell_key` is deterministic across processes (run twice, get same hash).
- `_compute_cell_key` produces different hashes for different swept-grid values.
- Round-trip: cell key from runtime cfg matches cell key from cull tool's
  reconstruction-from-W&B-config.

### Integration test (no engaging compute needed)
- Use the cached `lorenz_current` group (`perf-experiment/analysis/cache/lorenz_current__configs.pkl.gz`).
- Run the cull tool's `_pick_survivors` against it.
- Verify it picks the same set the simulation predicted at K=20, top-50 %
  cull (cf. `analysis/two_stage_out/all_results.csv`).

### Smoke test on engaging
- Submit a tiny Stage A: 5 cells (override sweep_grid), `max_epochs=10`,
  `limit_train_batches=20`. Should finish in ~5 minutes.
- Run cull tool with `--cull-fraction 0.5` → 2-3 survivors expected.
- Verify Stage B instructions are written to `instructions/pending/` and
  picked up by engaging-controller within 5 min.
- Verify survivor runs in Stage B start from a non-zero loss (i.e.
  resumed from ckpt, not from scratch).

### Validation on a real sweep
- Pick one of the existing sweep configs (e.g., `lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep`).
- Run the two-stage protocol: Stage A at K=20, cull 0.5, Stage B at the
  full `max_epochs=200`.
- Compare against the cached single-pass sweep on the three quality axes:
  - Final `trajectory val_loss` of survivors should match the cached
    sweep's same-cell within run-to-run noise (~5–10 %).
  - Lyapunov spectrum of the chosen run should match the cached chosen
    run within typical variance (NOT the regression we saw with the
    Step 1 alpha-schedule change — that was a different lever).
  - `loop_closure_weight` winner per `n_delays` bin should match in 2/3
    or 3/3 bins (this is what the simulation predicted).
- Total wall-clock should be ~50 % of the cached single-pass sweep.

## Validation history (why we trust the protocol)

- 71 historical sweep groups, all `JacobianODE` entity, last 30 days.
- Methodology in `JacobianODE/perf-experiment/analysis/two_stage_sim.py`.
- Aggregate results in `JacobianODE/perf-experiment/analysis/two_stage_out/summary.md`.
- Per-group breakdown in `analysis/two_stage_out/all_results.csv`.
- The 92 % / 100 % top-1 survival numbers are direct retrospective replays
  on actual sweep history, not Monte Carlo simulations — they reflect
  what *would have happened* if these sweeps had been run two-stage.

## What this spec deliberately doesn't do

- **Doesn't change the alpha schedule.** That was Phase 1 Step 1; we
  validated it on engaging and it costs Lyapunov-spectrum quality
  (`PHASE1_HYPOTHESES.md`). Reverted.
- **Doesn't add a controller-side state machine.** Manual cull-tool
  invocation between stages keeps blast radius small.
- **Doesn't auto-dependency Stage B SLURM jobs on Stage A.** User runs
  cull tool when Stage A's analysis report exists.
- **Doesn't touch the per-step training code.** Pure orchestration.
