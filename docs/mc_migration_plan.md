# MC sweep migration to mit_normal_gpu — design + handoff

**Goal**: extend `engaging-controller`'s `process_migrations` so it auto-migrates
PENDING cells from MindControl mc-sweeps onto `mit_normal_gpu` (currently
JacobianODE-native Hydra sweeps only). Same global QOS budget (4).

## Current state of the world

### JacobianODE-native side (works today)

- `mc_sbatch`-equivalent (`bin/jsweep`) writes `<SWEEPS_DIR>/active/<group>.expected.json` with:
  - `wandb.group`, `git.repo_dir`
  - `hydra.resolved_runs[*]` — per-cell `experiment` + `overrides`
  - `slurm_arrays: {run_idx_str: task_id_str}` populated per-cell
  - `migrate_to: "mit_normal_gpu"` (opt-in; absence means migration skips the sweep)
- A separate `<group>.state.json` is written by `monitor` each cycle, with
  per-cell `slurm_job_ids`, `last_slurm_state`, `classification`, `migrated`.
- `engaging-controller` calls `process_migrations(SWEEPS_DIR)` after monitor.
- `process_sweep_migrations` reads `expected.json`, requires `migrate_to`,
  loads `state.json`, picks PENDING cells via `_select_candidates`, then
  `migrate_one` per cell:
  1. journal "scancel_initiated"
  2. `scancel <bare_jobid>`, poll until gone
  3. `submit_cell(expected, run_idx, MIT_NORMAL_GPU, sweeps_dir, ...)` — builds
     a Hydra invocation `cd <repo> && OPENBLAS=4 uv run python -m
     JacobianODE.jacobians.run_jacobians experiment=<exp> <overrides>`
  4. journal "submitted_pending_expected_update"
  5. atomic update `expected.json` slurm_arrays[run_idx] = new_task; partition_per_cell[run_idx] = "mit_normal_gpu"
  6. stage state_patch with `migrated: 1`
  7. journal "complete"

### MindControl side (today)

- `mc_sbatch.py` writes `<MC_SWEEPS_DIR>/<group>/expected.json` (NOT in `active/`).
  Format is different:
  - `wandb_group`, `wandb_project`, `wandb_entity` (flat, not nested under `wandb`)
  - `slurm_array_id` (single string for the whole array, NOT per-cell mapping)
  - `instruction_path` (path to instruction.json, used by `mindcontrol.sweep_cell`)
  - `cells: [{cell_index, cell_params, wandb_run_name}]`
  - No `migrate_to` field (no opt-in)
- **No monitor → no `state.json`**.
- `process_migrations(SWEEPS_DIR)` is called only once (with the JacobianODE dir),
  so MC sweeps are invisible to migration.

## Format differences to bridge

| field                   | JacobianODE                                | MC                                  |
|-------------------------|--------------------------------------------|-------------------------------------|
| location                | `SWEEPS_DIR/active/<group>.expected.json`  | `MC_SWEEPS_DIR/<group>/expected.json` |
| wandb info              | nested `wandb.{group,project,entity}`      | flat `wandb_{group,project,entity}` |
| per-cell SLURM mapping  | `slurm_arrays: {run_idx: task_id}`         | implicit — derive from `slurm_array_id` + `cell_index` |
| per-cell run command    | `hydra.resolved_runs[i].{experiment, overrides}` → run_jacobians invocation | always `python -m mindcontrol.sweep_cell --instruction-path X --cell-index i` |
| per-cell partition      | `partition_per_cell: {run_idx: name}` (added on migration) | needs to be added |
| migrate_to opt-in       | top-level `migrate_to: "mit_normal_gpu"`   | currently nothing |
| `kind` discriminator    | absent (implicitly Hydra)                  | `kind: "mindcontrol"` in instruction; should also live in expected.json |
| state.json              | written by monitor every cycle             | **does not exist** — no MC monitor |

## Design

### A. mc_sbatch.py changes (most localized)

After the array submit, write expected.json with these *additional* fields so
it's migration-ready:

```python
expected = {
    "kind": "mc",                                    # NEW: discriminator
    "migrate_to": slurm.get("migrate_to"),           # NEW: opt-in (string or None)
    "wandb": {                                       # NEW: nested
        "group": wandb_group,
        "project": inst.get("wandb_project"),
        "entity": inst.get("wandb_entity"),
    },
    "slurm": {                                       # NEW: nested with full spec
        "partition": partition,
        "gres": gres,
        "cpus_per_task": cpus,
        "mem": mem,
        "timeout_min": timeout_min,
        "exclude_nodes": exclude_nodes,
        "account": account,
        "qos": qos,
    },
    "slurm_arrays": {                                # NEW: per-cell mapping
        str(i): f"{array_id}_{i}" for i in range(n_cells)
    },
    "partition_per_cell": {},                        # NEW: empty initially; migration populates
    "mc": {                                          # NEW: MC-specific deeplink
        "instruction_path": str(inst_path),
        "mc_repo": str(MC_REPO),
        "uv": UV,
    },
    # ... legacy MC fields preserved for back-compat
    "wandb_group": wandb_group,
    "wandb_project": inst.get("wandb_project"),
    ... (etc)
}
```

Also write a *symlink-or-copy* into `MC_SWEEPS_DIR/active/<group>.expected.json`
so `process_migrations` can find it via the standard "list active/" iteration.
Symlink is cheaper but copy is more crash-safe; lean toward copy.

### B. submit_cell.py: add `submit_mc_cell`

Parallel to `submit_cell`, takes the same signature, dispatches based on `kind`:

```python
def submit_mc_cell(
    expected: dict, run_idx: int, partition: PartitionSpec,
    sweeps_dir: Path, job_name_prefix: str = "mc_migrated",
) -> str:
    mc = expected["mc"]
    inst_path = mc["instruction_path"]
    mc_repo = mc["mc_repo"]
    uv = mc.get("uv", UV_PATH)
    group = expected["wandb"]["group"]

    wrap_cmd = (
        f"cd {mc_repo} && OPENBLAS_NUM_THREADS=4 HDF5_USE_FILE_LOCKING=FALSE "
        f"{uv} run --no-sync python -m mindcontrol.sweep_cell "
        f"--instruction-path {inst_path} --cell-index {run_idx}"
    )
    log_dir = sweeps_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"{job_name_prefix}_{group}_r{run_idx}"
    sbatch_args = build_sbatch_args(partition, job_name, log_dir, wrap_cmd)
    result = subprocess.run(sbatch_args, check=True, capture_output=True, text=True)
    jid = result.stdout.strip()
    logger.info(f"submit_mc_cell {group}/r{run_idx} -> SLURM {jid} on {partition.partition}")
    return jid
```

### C. migration.py dispatch on `kind`

In `migrate_one`, replace the direct `submit_cell(...)` call with:

```python
if expected.get("kind") == "mc":
    new_task = submit_mc_cell(expected, run_idx, MIT_NORMAL_GPU, sweeps_dir, ...)
else:
    new_task = submit_cell(expected, run_idx, MIT_NORMAL_GPU, sweeps_dir, ...)
```

### D. State.json bypass for MC

MC has no monitor, so `state.json` doesn't exist. Two options:

- **Option D1 (quickest)**: in `process_sweep_migrations`, when `kind == "mc"`,
  skip the state.json existence check. Build a synthetic `state` dict by
  querying squeue directly for each cell's task_id and synthesizing
  `state["runs"][k] = {"slurm_job_ids": [task_id], "last_slurm_state": <squeue_state>}`.

- **Option D2 (more aligned)**: write a tiny "mc_monitor" stub in the
  controller cycle that does this synthesis once per cycle and writes a
  state.json. More code, but keeps the migration's contract identical.

Lean D1 — minimal new code, no new monitor.

### E. engaging-controller changes

Add a second call after the JacobianODE migration call:

```python
mig_summary_jodes = process_migrations(SWEEPS_DIR)
mig_summary_mc = process_migrations(MC_SWEEPS_DIR)
```

`process_migrations` already handles "no active dir" gracefully.

### F. MC YAML opt-in syntax

```yaml
slurm:
  partition: ou_bcs_normal
  migrate_to: mit_normal_gpu     # NEW: opt-in for migration
```

`mc_sbatch.py` reads `slurm.migrate_to` and threads it into `expected.json`.

## Status (as of handoff)

- [x] Analysis + design captured here
- [ ] mc_sbatch.py: write migration-compatible `expected.json` + place in active/
- [ ] cell_submit.py: add `submit_mc_cell`
- [ ] migration.py: dispatch on `kind` in `migrate_one`
- [ ] migration.py: D1 state synthesis for MC sweeps
- [ ] engaging-controller: second `process_migrations` call
- [ ] Add `migrate_to: mit_normal_gpu` to the in-flight 42-cell LC sweep YAML
- [ ] Test on a small MC sweep before deploying

## Things to verify tomorrow

1. **The `slurm_array_id_TASK` bare jobid scancel pattern** works for an MC array
   (currently `submit_cell` derives bare jobid as `task.split("_")[0]` — should
   work for `13770000_17` → `13770000`, scanceling the WHOLE array though).
   Actually this is the wrong behavior for MC — we want to scancel ONE task
   in the array (`scancel 13770000_17`), not the whole array. Fix by sending
   the full task_id (don't strip suffix) to scancel.

2. **MC array partial-cancel** — verify SLURM lets you `scancel 13770000_17`
   (one task) without affecting the rest of the array. (It does, but worth
   re-confirming.)

3. **expected.json atomic update** when the file is in `MC_SWEEPS_DIR/active/`
   AND the original `MC_SWEEPS_DIR/<group>/expected.json` — keep both in sync,
   or have the active/ one be the source of truth.

4. **Concurrency budget** stays at MIT_BUDGET=4 across all sweeps (correct;
   the budget is computed globally before either call, but if we call
   `process_migrations` twice independently, the second call will see the
   freshly-migrated cells and skip them. So global budget is preserved
   *naturally* via squeue counts. Verify.)

5. **state_patch sidecar** is written under `<sweeps_dir>/active/<group>.state_patch.json`.
   For MC, there's no monitor to consume this file. Either: don't write it,
   or write it for future-proofing (when an MC monitor exists).

## Risk surface

- **Crash mid-migration leaves MC sweep in inconsistent state**: the journal +
  audit_journals startup recovery should still work because they only inspect
  the journal file. They might need a `kind`-aware path for cleanup though.
- **scancel of one array task** might accidentally kill the whole array if we
  use bare jobid. Need to use the full `array_id_task_idx` for MC.
