"""JacobianODE sweep monitor daemon.

Runs periodically (cron, every ~10 min) on engaging. Each cycle:

1. Acquires an exclusive ``flock`` in ``$SWEEPS_DIR/active/monitor.lock`` to
   prevent two monitor processes from stomping on each other.
2. Enumerates active sweeps: ``$SWEEPS_DIR/active/*.expected.json`` that
   don't yet have a corresponding ``$SWEEPS_DIR/done/<group>.done.json``
   or ``$SWEEPS_DIR/processed/<group>.done.json``.
3. For each such sweep:
   a. Loads the expected.json (written by ``engaging-submit``) and the
      existing state.json (if any).
   b. Queries the W&B API for every run tagged with the sweep's group.
   c. Matches each W&B run to a ``run_idx`` in the expected sweep grid by
      comparing override values against the run's logged config.
   d. Classifies each ``run_idx`` as running / done_* / failed_*.
   e. Resubmits runs classified as ``failed_retrying`` via ``sbatch``, up
      to the sweep's retry cap.
   f. Writes a new ``state.json`` atomically.
   g. If every ``run_idx`` is terminal and the sweep has been running
      longer than the minimum-elapsed guard, writes the ``done.json``
      sentinel atomically into ``$SWEEPS_DIR/done/``.

Usage:
    SWEEPS_DIR=/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps \\
        python -m JacobianODE.jacobians.tuning.monitor
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_status import (
    hit_slurm_walltime,
    meets_early_stopping_criterion,
)


logger = logging.getLogger("JacobianODE.sweep_monitor")


DEFAULT_SWEEPS_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps"

TERMINAL_CLASSIFICATIONS = {
    "done_finished", "done_early_stopped", "done_walltime", "failed_exhausted",
}

# SLURM job states we treat as "not yet terminated" — if an array task is in
# any of these, the corresponding run_idx classifies as `running` (even if
# wandb briefly flips to crashed/failed during a preempt+requeue).
ALIVE_SLURM_STATES = (
    "PENDING", "RUNNING", "CONFIGURING", "REQUEUED", "SUSPENDED",
)

# Hysteresis: a slot must miss this many consecutive monitor cycles
# (squeue reporting no alive task) before it can be reclassified into the
# failed_retrying lifecycle. squeue queries occasionally return empty
# transiently (scheduler load, NFS hiccup); requiring N consecutive
# misses prevents false-positive ghost retries while still catching real
# failures within ~N · cron_interval (default 2 · 5min = 10min).
DEFAULT_MISSING_CYCLES_THRESHOLD = 2


def _slurm_task_id(slurm_arrays: dict, k: str) -> str | None:
    """Return the SLURM task_id for a run_idx, or None if not recorded.

    Two stored formats are supported (the new format was introduced to fix
    the EXTEND-path misalignment, where run_idx was offset from the new
    array's task indices):
      * legacy: slurm_arrays[k] = "<array_id>"           → task_id = "<array_id>_<k>"
      * new:    slurm_arrays[k] = "<array_id>_<task_idx>" (already a task_id) → use as-is

    SLURM array_ids from sbatch are pure numeric, so the presence of "_"
    unambiguously identifies the new format.
    """
    val = slurm_arrays.get(k)
    if val is None:
        return None
    return val if "_" in val else f"{val}_{k}"


# ---------------------------------------------------------------------------
# Atomic JSON IO
# ---------------------------------------------------------------------------

def iso_now() -> str:
    """ISO-8601 UTC timestamp without microseconds, ending in 'Z'."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def atomic_write_json(path: Path, doc: dict) -> None:
    """Atomically write ``doc`` as JSON to ``path`` via tmp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Run / SLURM state queries
# ---------------------------------------------------------------------------

def query_wandb_runs(entity: str, project: str, group: str):
    """Return a list of wandb.apis.public.Run in the given group."""
    import wandb
    api = wandb.Api()
    project_path = f"{entity}/{project}"
    return list(api.runs(project_path, filters={"group": group}))


def query_squeue_states(user: str = "eisenaj") -> dict[str, str]:
    """Return dict of SLURM job_id -> current state string.

    ``-r`` (``--array``) expands compressed array specs into per-task rows.
    Without it, a pending array like ``12172560_[0-8%9]`` is returned as a
    single key with brackets, and per-task lookups in ``classify_run_idx``
    (e.g. ``slurm_states.get('12172560_0')``) all miss — which mis-
    classifies every run_idx as ``pending`` and (pre-ever_alive guard)
    stampedes ``resubmit_run`` on every subsequent monitor cycle.
    """
    try:
        output = subprocess.check_output(
            ["squeue", "-u", user, "-r", "-h", "-o", "%i|%T"], text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    result: dict[str, str] = {}
    for line in output.strip().split("\n"):
        if "|" not in line:
            continue
        jid, state = line.split("|", 1)
        jid = jid.strip()
        state = state.strip()
        result[jid] = state
        # SLURM compresses single-task arrays (size 1) to just the bare
        # array_id without the `_0` suffix — even with -r. Downstream
        # lookups (`_slurm_task_id` constructs `<array_id>_<task_idx>`)
        # would miss for those single-task arrays. Synthesize the
        # canonical task_id form so per-slot lookups succeed.
        if "_" not in jid:
            result[f"{jid}_0"] = state
    return result


# ---------------------------------------------------------------------------
# Matching wandb runs to expected run_idx
# ---------------------------------------------------------------------------

def _coerce(s: str):
    """Coerce a string override value to Python for comparison."""
    s = s.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "none"):
        return None
    try:
        f = float(s)
        if f.is_integer() and "." not in s and "e" not in s.lower():
            return int(f)
        return f
    except ValueError:
        return s


def _get_nested(d: Any, key: str, default=None):
    """Look up ``d['a.b.c']`` as ``d['a']['b']['c']`` (works on dicts)."""
    cur = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _values_match(wanted, actual) -> bool:
    """Compare two values, allowing numeric fuzz."""
    if actual is None:
        return wanted is None
    if isinstance(wanted, float) or isinstance(actual, float):
        try:
            w = float(wanted); a = float(actual)
            return abs(w - a) <= 1e-9 * max(abs(w), abs(a), 1.0)
        except (TypeError, ValueError):
            return False
    return wanted == actual


def _is_matchable_override(ov: str) -> bool:
    """An override is matchable against a wandb run's config if it targets
    a training-visible key. Hydra consumes ``hydra.*`` and ``experiment=...``
    before the config reaches the training loop, so those keys aren't
    present in the wandb config and would cause spurious non-matches."""
    if "=" not in ov:
        return False
    key = ov.split("=", 1)[0]
    if key.startswith("hydra."):
        return False
    if key == "experiment":
        return False
    return True


def match_run_to_idx(wandb_config: dict, resolved_runs: list) -> int | None:
    """Return the run_idx whose overrides match this run's config, or None.

    Ambiguous matches (more than one) return None — we'd rather skip than
    mislabel. Most override sets are uniquely identifying so this is rare.

    Overrides on keys that hydra-consumes-and-strips before config reaches
    the training loop (``hydra.*``, ``experiment=...``) are ignored — those
    don't appear in the wandb run config and would cause spurious
    non-matches for every run.

    Trivial-sweep shortcut: if a single resolved_run has NO matchable
    overrides (e.g. ``sweep_grid: {}`` with launcher-only overrides),
    there's nothing to disambiguate — return its run_idx unconditionally.

    But when called with a one-element ``resolved_runs`` list that DOES
    contain matchable overrides (as ``classify_run_idx`` does, once per
    run_idx), we MUST fall through to the real override-based match.
    Returning the run_idx unconditionally in that case causes every wandb
    run to falsely match every run_idx — this was a latent bug that made
    every non-alive-SLURM run_idx classify as ``done_finished`` whenever
    any wandb run in the group had state=finished.
    """
    if len(resolved_runs) == 1 and not any(
        _is_matchable_override(ov) for ov in resolved_runs[0].get("overrides", [])
    ):
        return resolved_runs[0]["run_idx"]

    # (run_idx, specificity) for each matching resolved_run; specificity is
    # the number of matchable overrides the resolved_run constrains. When
    # expected.json was extended with a finer sweep axis (e.g. obs_noise),
    # old runs have fewer overrides than new runs — without a specificity
    # tiebreak a new wandb run would match both and be dropped as ambiguous.
    matches: list[tuple[int, int]] = []
    for r in resolved_runs:
        filtered_overrides = [ov for ov in r["overrides"] if _is_matchable_override(ov)]
        if not filtered_overrides:
            # All overrides were launcher/experiment — can't disambiguate.
            continue
        if all(
            _values_match(
                _coerce(ov.split("=", 1)[1]),
                _get_nested(wandb_config, ov.split("=", 1)[0]),
            )
            for ov in filtered_overrides
        ):
            matches.append((r["run_idx"], len(filtered_overrides)))
    if not matches:
        return None
    max_spec = max(s for _, s in matches)
    top = [idx for idx, s in matches if s == max_spec]
    return top[0] if len(top) == 1 else None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_wandb_run(run, slurm_timeout_min: float | None) -> str:
    """Map a wandb run's state to our closed-set classification.

    Returns one of: running, done_finished, done_early_stopped, done_walltime,
    failed. The caller decides failed_retrying vs failed_exhausted based on
    attempts vs retry cap.
    """
    state = (run.state or "").lower()
    if state == "finished":
        return "done_finished"
    if state == "running":
        return "running"
    # Crashed/failed — see if it would have stopped on its own
    try:
        met, _ = meets_early_stopping_criterion(run)
        if met:
            return "done_early_stopped"
    except Exception as e:
        logger.debug(f"early_stopping_criterion check failed: {e}")
    if slurm_timeout_min is not None:
        try:
            if hit_slurm_walltime(run, timeout_min=slurm_timeout_min):
                return "done_walltime"
        except Exception as e:
            logger.debug(f"hit_slurm_walltime check failed: {e}")
    return "failed"


# ---------------------------------------------------------------------------
# Stateless run_idx classification
# ---------------------------------------------------------------------------

def _ckpt_base(expected: dict) -> Path | None:
    """Where Lightning wrote checkpoints for this sweep, or None if
    unknown. Format: ``{logger_save_dirs}/{wandb_project}``."""
    save_dir = (
        expected.get("training", {}).get("logger_save_dirs")
        or expected.get("training", {}).get("logger", {}).get("save_dir")
    )
    proj = expected.get("wandb", {}).get("project")
    if not save_dir or not proj:
        return None
    return Path(save_dir) / proj


def classify_run_idx(
    run_idx: int,
    resolved_run: dict,
    wandb_runs: list,
    slurm_states: dict[str, str],
    ckpt_base: Path | None,
    slurm_arrays: dict[str, str],
    slurm_timeout_min: float | None,
) -> tuple[str, list[str], str | None]:
    """Pure classification of a single ``run_idx`` from current observables.

    This is the core of the stateless control-plane rewrite: instead of
    accumulating ``terminal=True`` across cycles and letting it stick
    regardless of later evidence, each cycle recomputes a run_idx's
    classification from (wandb runs, SLURM queue, checkpoint existence).

    Decision order:
      1. If any SLURM array task ``{array_id}_{run_idx}`` is in an alive
         state (PENDING/RUNNING/…) → ``running``. This wins over any
         transient wandb state, so a preempt's brief crashed-window
         doesn't mis-classify the run.
      2. If no wandb runs are matched yet → ``pending``.
      3. If any matched run has state=finished AND a checkpoint dir →
         ``done_finished``.
      4. If any matched run (that has a checkpoint dir) meets the
         early-stopping criterion → ``done_early_stopped``.
      5. If any matched run (that has a checkpoint dir) hit SLURM walltime
         → ``done_walltime``.
      6. Otherwise → ``failed`` (no alive SLURM task, no satisfied
         completion criterion).

    Returns
    -------
    (classification, matched_wandb_run_ids, last_wandb_state_seen)
    """
    k = str(run_idx)
    matched = []
    for r in wandb_runs:
        try:
            cfg = dict(r.config)
        except Exception:
            continue
        if match_run_to_idx(cfg, [resolved_run]) == run_idx:
            matched.append(r)

    # (1) Alive SLURM array task — wins over any wandb state.
    task_id = _slurm_task_id(slurm_arrays, k)
    if task_id is not None:
        if slurm_states.get(task_id) in ALIVE_SLURM_STATES:
            last = (matched[-1].state if matched else None)
            return ("running", [r.id for r in matched], last)

    if not matched:
        return ("pending", [], None)

    # Sort matched by heartbeat_at (or updated_at) descending so we look at
    # the most recent attempt first when picking the "best" classification.
    def _sort_key(r):
        return r.heartbeat_at or r.updated_at or ""
    matched_sorted = sorted(matched, key=_sort_key, reverse=True)

    def _has_ckpt(r) -> bool:
        if ckpt_base is None:
            return True  # can't verify; don't penalise
        return (ckpt_base / r.id / "checkpoints").is_dir()

    ckpt_runs = [r for r in matched_sorted if _has_ckpt(r)]

    # (3) Best wandb run is finished AND has a checkpoint dir.
    for r in ckpt_runs:
        if (r.state or "").lower() == "finished":
            return ("done_finished", [r.id for r in matched], r.state)

    # (4) Converged.
    for r in ckpt_runs:
        try:
            converged, _ = meets_early_stopping_criterion(r)
            if converged:
                return ("done_early_stopped", [r.id for r in matched], r.state)
        except Exception:
            continue

    # (5) Hit walltime.
    if slurm_timeout_min is not None:
        for r in ckpt_runs:
            try:
                if hit_slurm_walltime(r, timeout_min=slurm_timeout_min):
                    return ("done_walltime", [r.id for r in matched], r.state)
            except Exception:
                continue

    # (6) Failed — has wandb evidence but none of the done_* criteria met.
    last = matched_sorted[0].state
    return ("failed", [r.id for r in matched], last)


# ---------------------------------------------------------------------------
# Resubmission
# ---------------------------------------------------------------------------

def resubmit_run(expected: dict, run_idx: int, sweeps_dir: Path) -> str:
    """Submit a retry for ``run_idx`` via direct sbatch. Returns the new job
    ID, or empty string on failure.

    Thin wrapper around ``cell_submit.submit_cell``: reads the partition
    spec from the sweep's ``expected.slurm`` block (preserved on initial
    submission OR updated by migration to the migrated partition), then
    delegates the sbatch construction. Migrated cells therefore retry on
    mit_normal_gpu rather than falling back to ou_bcs_normal.
    """
    from .cell_submit import spec_from_expected_slurm, submit_cell
    group = expected["wandb"]["group"]
    partition = spec_from_expected_slurm(expected.get("slurm") or {})
    try:
        return submit_cell(
            expected, run_idx, partition, sweeps_dir,
            job_name_prefix="jacobian_retry",
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            f"sbatch failed for {group}/run_idx={run_idx}: "
            f"{(e.stderr or '').strip()}"
        )
        return ""


# ---------------------------------------------------------------------------
# Main per-sweep logic
# ---------------------------------------------------------------------------

def load_or_init_state(sweeps_dir: Path, group: str, expected: dict) -> dict:
    path = sweeps_dir / "active" / f"{group}.state.json"
    if path.is_file():
        return load_json(path)
    return {
        "schema_version": 1,
        "group": group,
        "last_updated": iso_now(),
        "monitor_cycle": 0,
        "runs": {},
        "summary": {},
    }


def check_sweep(expected_path: Path, sweeps_dir: Path) -> None:
    """One cycle of monitoring for a single sweep.

    - Loads expected.json + current state.json.
    - Queries wandb + squeue.
    - Updates state and (maybe) writes the done sentinel.
    """
    expected = load_json(expected_path)
    group = expected["wandb"]["group"]

    done_path = sweeps_dir / "done" / f"{group}.done.json"
    processed_path = sweeps_dir / "processed" / f"{group}.done.json"
    failed_path = sweeps_dir / "failed" / f"{group}.done.json"
    # Don't re-create a sentinel that's already been handed off (done/ or
    # processed/) or permanently shelved by the analysis agent (failed/).
    if done_path.is_file() or processed_path.is_file() or failed_path.is_file():
        return

    state = load_or_init_state(sweeps_dir, group, expected)
    state["monitor_cycle"] = state.get("monitor_cycle", 0) + 1
    state["last_updated"] = iso_now()

    # Seed state.runs from expected if this is the first cycle
    resolved = expected["hydra"]["resolved_runs"]
    for r in resolved:
        k = str(r["run_idx"])
        if k not in state["runs"]:
            state["runs"][k] = {
                "wandb_run_ids": [],
                "slurm_job_ids": [],
                "attempts": 0,
                "last_wandb_state": None,
                "last_slurm_state": None,
                "classification": "pending_resubmit",
                "terminal": False,
            }

    # Prune any state entries whose run_idx was removed from expected.json
    # (happens when a sweep is trimmed mid-flight — e.g. cancelling a subset
    # of the grid). Without this, the summary keeps counting ghost runs and
    # the dashboard disagrees with reality.
    expected_keys = {str(r["run_idx"]) for r in resolved}
    stale_keys = [k for k in state["runs"] if k not in expected_keys]
    for k in stale_keys:
        state["runs"].pop(k)
    if stale_keys:
        logger.info(
            f"{group}: pruned {len(stale_keys)} stale run_idx entries "
            f"no longer in expected.json: {stale_keys}"
        )

    try:
        wandb_runs = query_wandb_runs(
            expected["wandb"]["entity"],
            expected["wandb"]["project"],
            group,
        )
    except Exception as e:
        logger.error(f"wandb query failed for group={group}: {e}")
        return

    slurm_states = query_squeue_states()
    timeout_min = expected["slurm"].get("timeout_min", 180)
    slurm_arrays = expected.get("slurm_arrays") or {}
    ckpt_base = _ckpt_base(expected)
    retry_cap = expected.get("retry", {}).get("cap_per_run", 2)
    missing_threshold = expected.get("retry", {}).get(
        "missing_cycles_threshold", DEFAULT_MISSING_CYCLES_THRESHOLD,
    )

    # STATELESS classification: recompute each run_idx's classification from
    # scratch every cycle. This replaces the previous sticky-terminal logic
    # where `entry["terminal"] = True` once-set-never-unset — that model
    # broke badly on preempt+requeue cycles where wandb state oscillates.
    for r in resolved:
        k = str(r["run_idx"])
        entry = state["runs"].setdefault(k, {
            "wandb_run_ids": [],
            "slurm_job_ids": [],
            "attempts": 0,
            "last_wandb_state": None,
            "last_slurm_state": None,
            "classification": "pending",
            "terminal": False,
            "consecutive_missing_cycles": 0,
        })

        cls, wids, last_wb = classify_run_idx(
            run_idx=r["run_idx"],
            resolved_run=r,
            wandb_runs=wandb_runs,
            slurm_states=slurm_states,
            ckpt_base=ckpt_base,
            slurm_arrays=slurm_arrays,
            slurm_timeout_min=timeout_min,
        )

        # Track whether we've *ever* observed this slot's array task alive
        # in squeue. Guards the "pending → failed_retrying" reclass below:
        # only slots we watched alive and then go missing (crash, 429,
        # srun step error, OOM before wandb.init) should be retried. A
        # slot that's never been seen alive is almost certainly either
        # (a) still about to be dispatched or (b) the squeue query
        # returned a transient empty result.
        task_id = _slurm_task_id(slurm_arrays, k)
        if task_id is not None:
            if slurm_states.get(task_id) in ALIVE_SLURM_STATES:
                entry["ever_alive"] = True

        # Hysteresis on "task is missing from squeue" — a single missed
        # squeue snapshot can be a transient (scheduler load, NFS hiccup)
        # and should NOT immediately escalate the slot into the retry
        # lifecycle. Increment a counter on each consecutive miss; reset
        # to 0 the moment the task is observed alive again. Reclass to
        # failed_retrying only after `missing_threshold` consecutive misses.
        # This catches the false-positive ghost-retry mode where the
        # original SLURM array task is still RUNNING but squeue briefly
        # didn't return it.
        if task_id is not None:
            if slurm_states.get(task_id) in ALIVE_SLURM_STATES:
                entry["consecutive_missing_cycles"] = 0
            else:
                entry["consecutive_missing_cycles"] = (
                    entry.get("consecutive_missing_cycles", 0) + 1
                )
        missing_confirmed = (
            task_id is not None
            and entry.get("consecutive_missing_cycles", 0) >= missing_threshold
        )

        # Translate "failed" into the retry lifecycle (failed_retrying vs
        # failed_exhausted). Gated on hysteresis: a single squeue miss is
        # not enough to declare failure when there's no completion
        # criterion met yet. (If task_id is None — slot was never even
        # tracked in slurm_arrays — we have no SLURM evidence either way
        # and fall back to the unconditional behavior.)
        if cls == "failed":
            if task_id is not None and not missing_confirmed:
                # Treat as still-running for now; await more evidence.
                cls = "running"
            elif len(wids) > retry_cap:
                cls = "failed_exhausted"
            else:
                cls = "failed_retrying"

        # A "pending" slot that was previously dispatched, has been seen
        # alive in at least one prior cycle, and is no longer alive,
        # without any wandb evidence, failed *before* wandb.init —
        # e.g. srun step error, wandb API 429 at startup, OOM before
        # training began. Route it into the retry lifecycle. Also gated
        # on the same hysteresis.
        if (
            cls == "pending"
            and state["monitor_cycle"] > 1
            and entry.get("ever_alive")
            and missing_confirmed
        ):
            cls = (
                "failed_exhausted" if entry.get("attempts", 0) >= retry_cap
                else "failed_retrying"
            )

        entry["wandb_run_ids"] = wids
        entry["attempts"] = len(wids)
        entry["last_wandb_state"] = last_wb
        entry["classification"] = cls
        entry["terminal"] = cls in TERMINAL_CLASSIFICATIONS

    # Fold in SLURM states for monitor-retry jobs we've tracked (still used by
    # the resubmit loop below to decide whether a retry is already queued).
    for entry in state["runs"].values():
        for jid in entry["slurm_job_ids"]:
            if jid in slurm_states:
                entry["last_slurm_state"] = slurm_states[jid]

    # Resubmit anything currently marked failed_retrying and not already queued.
    # Note: with the stateless classify_run_idx, the alive-array-task check
    # is redundant (a run_idx with an alive array task classifies as
    # "running", never reaching "failed_retrying"). Keeping the guard for
    # defense-in-depth until Phase 4 retires this retry path entirely.
    for k, entry in state["runs"].items():
        if entry["classification"] != "failed_retrying":
            continue
        if any(slurm_states.get(jid) in ALIVE_SLURM_STATES
               for jid in entry["slurm_job_ids"]):
            continue
        orig_task = _slurm_task_id(slurm_arrays, k)
        if orig_task is not None:
            if slurm_states.get(orig_task) in ALIVE_SLURM_STATES:
                logger.info(
                    f"{expected['wandb']['group']}/run_idx={k}: original SLURM "
                    f"task {orig_task} still alive — skipping monitor retry"
                )
                continue
        new_jid = resubmit_run(expected, int(k), sweeps_dir)
        if new_jid:
            entry["slurm_job_ids"].append(new_jid)

    # Compute summary
    runs = state["runs"]
    state["summary"] = {
        "terminal": sum(1 for e in runs.values() if e["terminal"]),
        "in_flight": sum(1 for e in runs.values() if not e["terminal"]),
        "failed_exhausted": sum(
            1 for e in runs.values() if e["classification"] == "failed_exhausted"
        ),
        "effectively_done": sum(
            1 for e in runs.values() if e["classification"].startswith("done_")
        ),
    }

    # Write state.json atomically
    state_path = sweeps_dir / "active" / f"{group}.state.json"
    atomic_write_json(state_path, state)

    # Sentinel check: all terminal AND elapsed time guard met
    all_terminal = (
        len(runs) == expected["expected_run_count"]
        and all(e["terminal"] for e in runs.values())
    )
    if not all_terminal:
        return

    launched_at = datetime.fromisoformat(
        expected["launched_at"].replace("Z", "+00:00")
    )
    elapsed = (datetime.now(timezone.utc) - launched_at).total_seconds()
    min_elapsed = expected.get("retry", {}).get("min_elapsed_before_done_sec", 600)
    if elapsed < min_elapsed:
        logger.info(
            f"{group}: all terminal but only {elapsed:.0f}s since launch "
            f"(guard = {min_elapsed}s); waiting."
        )
        return

    # Orphan cleanup: sentinel is about to fire, so any SLURM array task
    # still alive for this group is wasted compute (training continues on a
    # wandb run the sweep already considers terminal). Scancel them before
    # writing done.json so the orphan doesn't keep flipping wandb state and
    # confusing downstream analysis.
    orphan_ids = []
    for k in slurm_arrays:
        task = _slurm_task_id(slurm_arrays, k)
        if task is not None and slurm_states.get(task) in ALIVE_SLURM_STATES:
            orphan_ids.append(task)
    # Also pick up any monitor-retry jobs still queued/running.
    for entry in runs.values():
        for jid in entry.get("slurm_job_ids", []):
            if slurm_states.get(jid) in ALIVE_SLURM_STATES:
                orphan_ids.append(jid)
    if orphan_ids:
        logger.info(
            f"{group}: scancelling {len(orphan_ids)} orphan SLURM task(s) "
            f"before sentinel: {orphan_ids}"
        )
        try:
            subprocess.run(["scancel", *orphan_ids], check=False)
        except Exception as e:
            logger.warning(f"orphan scancel failed: {e}")

    # Write done.json atomically
    done_doc = {
        "schema_version": 1,
        "group": group,
        "completed_at": iso_now(),
        "outcome": (
            "complete_clean"
            if state["summary"]["failed_exhausted"] == 0
            else "complete_with_failures"
        ),
        "expected_snapshot": expected,
        "final_state_snapshot": state,
        "successful_run_ids": [
            rid
            for e in runs.values()
            if e["classification"].startswith("done_")
            for rid in e["wandb_run_ids"]
        ],
        "failed_run_indices": [
            int(k) for k, e in runs.items()
            if e["classification"] == "failed_exhausted"
        ],
        "multirun_dir": expected["hydra"].get("multirun_dir"),
    }
    atomic_write_json(done_path, done_doc)
    logger.info(f"Wrote sentinel: {done_path}")


# ---------------------------------------------------------------------------
# Lock + main
# ---------------------------------------------------------------------------

def acquire_lock(sweeps_dir: Path):
    lock_path = sweeps_dir / "active" / "monitor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        return None
    return fd


def install_cron(sweeps_dir: Path, interval_min: int = 10) -> int:
    """Install a crontab entry that runs this monitor every ``interval_min``
    minutes. Intended to be invoked once from engaging.
    """
    uv_bin = os.path.expanduser("~/.local/bin/uv")
    # Prefer the repo that contains this file — if we're running from
    # /home/eisenaj/code/JacobianODE, that's the repo root.
    repo_root = Path(__file__).resolve().parents[3]
    log_path = sweeps_dir / "logs" / "monitor.log"
    line = (
        f"*/{interval_min} * * * * "
        f"SWEEPS_DIR={sweeps_dir} {uv_bin} run --no-sync --project {repo_root} "
        f"python -m JacobianODE.jacobians.tuning.monitor "
        f">> {log_path} 2>&1"
    )
    # Get current crontab (empty if none)
    try:
        current = subprocess.check_output(["crontab", "-l"], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        current = ""
    marker = "JacobianODE.jacobians.tuning.monitor"
    kept = [ln for ln in current.splitlines() if marker not in ln]
    new_cron = "\n".join(kept + [line]) + "\n"
    p = subprocess.run(["crontab", "-"], input=new_cron, text=True)
    if p.returncode != 0:
        print("ERROR: failed to install crontab", file=sys.stderr)
        return 2
    print(f"Installed crontab entry (every {interval_min} min):")
    print(f"  {line}")
    print(f"Logs will go to: {log_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweeps-dir", default=None,
        help=f"Override $SWEEPS_DIR (default: {DEFAULT_SWEEPS_DIR})",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--install-cron", action="store_true",
        help="Install a crontab entry for this monitor and exit (run once on engaging).",
    )
    parser.add_argument(
        "--cron-interval", type=int, default=10,
        help="Interval in minutes for the installed cron entry (default: 10).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sweeps_dir = Path(
        args.sweeps_dir or os.environ.get("SWEEPS_DIR") or DEFAULT_SWEEPS_DIR
    )
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("active", "done", "processed", "logs"):
        (sweeps_dir / sub).mkdir(parents=True, exist_ok=True)

    if args.install_cron:
        return install_cron(sweeps_dir, interval_min=args.cron_interval)

    lock = acquire_lock(sweeps_dir)
    if lock is None:
        logger.info("Another monitor cycle is running; exiting.")
        return 0

    try:
        expected_files = sorted(
            (sweeps_dir / "active").glob("*.expected.json")
        )
        logger.info(f"Monitor cycle: {len(expected_files)} active sweep(s)")
        for path in expected_files:
            try:
                check_sweep(path, sweeps_dir)
            except Exception:
                logger.exception(f"Error processing {path}")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
