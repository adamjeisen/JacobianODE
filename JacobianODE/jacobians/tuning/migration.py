"""Dynamic mit_normal_gpu ↔ ou_bcs_normal cell migration.

Runs as a step in the engaging-controller cycle, AFTER the monitor cycle.
Per active migrate-eligible sweep:
  - Counts our running/pending jobs on mit_normal_gpu (QOS budget = 4).
  - Picks PENDING ou_bcs_normal cells with clean state (≤ 1 SLURM job, not
    yet migrated) and migrates them: scancel old + sbatch new + atomic
    expected.json update.

Safety architecture:
  * Monitor runs BEFORE migration in the same controller cycle, so when
    the next cycle's monitor runs, expected.json reflects the new task IDs
    and squeue lookups succeed (no spurious retries).
  * Every state-changing step writes to a per-sweep journal first, THEN
    acts. ``audit_journals`` runs at controller startup and reconciles
    any half-finished migrations from a prior crash.
  * scancel BEFORE sbatch — worst case is a missing run (recovered by
    monitor's standard retry), never a duplicate.
  * Atomic expected.json mutation via .tmp + os.replace.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .cell_submit import (
    MIT_NORMAL_GPU,
    OU_BCS_NORMAL,
    PartitionSpec,
    submit_cell,
)

logger = logging.getLogger(__name__)


# QOS-imposed concurrency limit for mit_amf_advanced_gpu.
MIT_BUDGET = 4

# How long to poll for a scancel'd job to actually leave squeue.
SCANCEL_POLL_TIMEOUT_S = 15
SCANCEL_POLL_INTERVAL_S = 1.0

# SLURM states that indicate the task is currently allocated/queued
# (i.e. NOT eligible for migration — only PENDING is).
ALIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "REQUEUED", "SUSPENDED"}


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# squeue queries
# ---------------------------------------------------------------------------

def _slurm_user() -> str:
    """The user we run as on engaging."""
    return os.environ.get("USER", "eisenaj")


def count_running_mit_jobs() -> int:
    """Count *our* jobs currently in any alive state on mit_normal_gpu.

    Used to compute migration budget: ``budget = MIT_BUDGET - count``.
    Counts both PENDING and RUNNING because the QOS limit is enforced
    on submitted-jobs, not just running ones — submitting a 5th when 4
    are already PENDING gets rejected (or queued indefinitely).
    """
    try:
        out = subprocess.run(
            ["squeue", "-u", _slurm_user(), "-h",
             "-p", "mit_normal_gpu",
             "--account", "mit_amf_advanced_gpu",
             "-o", "%i"],
            check=False, capture_output=True, text=True, timeout=15,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return MIT_BUDGET  # conservative: assume full, skip migration
    return len([line for line in out.splitlines() if line.strip()])


def _squeue_state(task_id: str) -> Optional[str]:
    """Return the SLURM state (e.g. 'PENDING', 'RUNNING') of a single
    task_id, or None if not in queue. Handles both bare jobids and
    array-task syntax."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", task_id, "-o", "%T"],
            check=False, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out if out else None


def _poll_gone(task_id: str, timeout_s: float = SCANCEL_POLL_TIMEOUT_S) -> bool:
    """Wait until ``task_id`` is no longer alive in squeue. Returns True
    on success, False on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = _squeue_state(task_id)
        if state is None or state not in ALIVE_STATES:
            return True
        time.sleep(SCANCEL_POLL_INTERVAL_S)
    return False


# ---------------------------------------------------------------------------
# Atomic expected.json update
# ---------------------------------------------------------------------------

def _atomic_update_json(path: Path, mutator) -> None:
    """Read JSON, apply mutator (in place), write to .tmp, rename. Raises
    on any IO failure so the caller can recover (scancel orphan)."""
    doc = json.loads(path.read_text())
    mutator(doc)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(str(tmp), str(path))


def _apply_migration(expected: dict, run_idx: int, new_task_id: str,
                     new_partition: PartitionSpec) -> None:
    """In-place mutation: update slurm_arrays + slurm partition + per-cell
    partition map + bump migration counter."""
    expected.setdefault("slurm_arrays", {})[str(run_idx)] = new_task_id
    # Update top-level slurm so monitor.resubmit_run uses the migrated
    # partition for any future retries (Q2 decision: cap=1, retries inherit).
    slurm = expected.setdefault("slurm", {})
    slurm["partition"] = new_partition.partition
    slurm["account"] = new_partition.account
    slurm["qos"] = new_partition.qos
    slurm["gres"] = new_partition.gres
    # Per-cell partition map so subsequent migration cycles know which
    # cells are still on ou_bcs_normal vs already on mit. This is the
    # canonical "where is each cell" record.
    pcp = expected.setdefault("partition_per_cell", {})
    pcp[str(run_idx)] = new_partition.partition
    # Audit counter
    mig = expected.setdefault("migrations", {})
    mig[str(run_idx)] = mig.get(str(run_idx), 0) + 1


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def _journal_path(sweeps_dir: Path, group: str) -> Path:
    return sweeps_dir / "active" / f"{group}.migration.journal.jsonl"


def _journal_append(path: Path, entry: dict) -> None:
    """Append one JSON line. fsync for crash-safety on the journal entry
    itself (the journal is the recovery mechanism — must survive SIGKILL)."""
    entry = {**entry, "at": utc_iso()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _journal_read_all(path: Path) -> list[dict]:
    """Read all entries. Returns [] if file doesn't exist."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"[journal] skipping malformed line in {path}")
    return out


def _journal_latest_per_run(entries: list[dict]) -> dict[int, dict]:
    """Reduce a journal to the latest entry per run_idx."""
    latest: dict[int, dict] = {}
    for e in entries:
        ri = e.get("run_idx")
        if ri is None:
            continue
        latest[int(ri)] = e
    return latest


# ---------------------------------------------------------------------------
# Per-cell migration
# ---------------------------------------------------------------------------

def _is_clean_for_migration(state_runs: dict, run_idx: int) -> tuple[bool, str]:
    """Check whether a cell is in a clean enough state to migrate.
    Returns (eligible, reason_if_not). Inspects state.json's per-cell record.
    """
    s = state_runs.get(str(run_idx))
    if s is None:
        return False, "no_state_entry"
    # Already migrated (cap = 1 per cell)
    if s.get("migrated", 0) >= 1:
        return False, "already_migrated"
    # More than one slurm_job_id = monitor retry already in flight
    if len(s.get("slurm_job_ids", [])) > 1:
        return False, "monitor_retry_in_flight"
    # Classification must not be terminal
    cls = s.get("classification") or ""
    if cls.startswith("done_") or cls == "failed_exhausted":
        return False, f"terminal_classification_{cls}"
    # Last seen SLURM state should be PENDING (PD) when known.
    # ``None`` is allowed because monitor's last_slurm_state is only
    # populated when squeue has been queried for this exact task_id; for
    # cells that have never been migrated/retried, this can be None even
    # when the task IS pending in squeue. ``migrate_one`` re-checks via
    # real-time _squeue_state before acting, so allowing None here is
    # safe — worst case is we attempt a migration that aborts cleanly
    # because the cell is no longer PD.
    last_state = (s.get("last_slurm_state") or "").upper()
    if last_state and last_state not in {"PENDING", "PD"}:
        return False, f"not_pending_{last_state}"
    return True, ""


def migrate_one(
    group: str,
    run_idx: int,
    expected_path: Path,
    expected: dict,
    state_runs: dict,
    sweeps_dir: Path,
) -> bool:
    """Atomically migrate one cell from ou_bcs_normal to mit_normal_gpu.

    Returns True on confirmed migration (expected.json updated, new task
    alive in squeue or at least submitted), False otherwise.

    The function is idempotent at journal-entry granularity: if it crashes
    mid-execution, ``audit_journals`` reconciles on next controller startup.
    """
    journal = _journal_path(sweeps_dir, group)
    old_task = expected.get("slurm_arrays", {}).get(str(run_idx))
    if not old_task:
        logger.warning(f"[migrate skip] {group}/r{run_idx}: no slurm_arrays entry")
        return False

    # Re-check liveness against squeue (state.json may be stale)
    live = _squeue_state(old_task)
    if live not in {"PENDING", "PD"}:
        logger.info(f"[migrate skip] {group}/r{run_idx}: state={live!r} not PENDING")
        return False

    # 1. Journal scancel intent
    _journal_append(journal, {
        "run_idx": run_idx, "old_task": old_task,
        "stage": "scancel_initiated",
    })

    # 2. scancel old (use bare jobid for arrays — splits "12345_3" -> "12345")
    bare_jobid = old_task.split("_")[0]
    try:
        subprocess.run(
            ["scancel", bare_jobid], check=True,
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"[migrate fail] {group}/r{run_idx} scancel failed: {e}")
        _journal_append(journal, {
            "run_idx": run_idx, "stage": "abort_scancel_error",
            "error": str(e)[:200],
        })
        return False

    if not _poll_gone(old_task):
        logger.warning(
            f"[migrate fail] {group}/r{run_idx} scancel didn't clear "
            f"{old_task} within {SCANCEL_POLL_TIMEOUT_S}s — aborting"
        )
        _journal_append(journal, {
            "run_idx": run_idx, "stage": "abort_scancel_stuck",
        })
        return False

    # 3. sbatch new task on mit
    try:
        new_task = submit_cell(
            expected, run_idx, MIT_NORMAL_GPU, sweeps_dir,
            job_name_prefix="jacobian_migrated",
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            f"[migrate fail] {group}/r{run_idx} sbatch on mit failed: "
            f"{(e.stderr or '').strip()[:200]}. Cell will be picked up by "
            f"monitor retry path on ou_bcs_normal."
        )
        _journal_append(journal, {
            "run_idx": run_idx, "stage": "abort_sbatch_failed",
            "error": (e.stderr or str(e))[:200],
        })
        return False

    # 4. Journal "submitted_pending_expected_update" — KEY recovery checkpoint.
    # If we crash between here and the expected.json update, audit_journals
    # will see this entry, find that expected.json doesn't have new_task,
    # and scancel the orphan.
    _journal_append(journal, {
        "run_idx": run_idx, "old_task": old_task, "new_task": new_task,
        "stage": "submitted_pending_expected_update",
    })

    # 5. Atomic expected.json update
    try:
        _atomic_update_json(
            expected_path,
            lambda doc: _apply_migration(doc, run_idx, new_task, MIT_NORMAL_GPU),
        )
    except Exception as e:
        logger.error(
            f"[migrate ORPHAN] {group}/r{run_idx} expected.json update "
            f"FAILED ({e}); scanceling orphan {new_task}"
        )
        try:
            subprocess.run(["scancel", new_task], check=False, timeout=10)
        except Exception as sc:
            logger.error(f"[migrate ORPHAN] orphan scancel ALSO failed: {sc}")
            # journal entry stays; startup audit will retry
        _journal_append(journal, {
            "run_idx": run_idx, "stage": "orphan_scanceled",
        })
        return False

    # 6. Mark migration done in state-patch sidecar (monitor merges on next read)
    _stage_state_patch(sweeps_dir, group, run_idx, {"migrated": 1})

    _journal_append(journal, {
        "run_idx": run_idx, "old_task": old_task, "new_task": new_task,
        "stage": "complete",
    })
    logger.info(
        f"[migrate OK] {group}/r{run_idx}: {old_task} -> {new_task} (mit_normal_gpu)"
    )
    return True


# ---------------------------------------------------------------------------
# state.json sidecar (single-writer discipline: only monitor writes
# state.json; migration writes a sidecar that monitor merges next cycle)
# ---------------------------------------------------------------------------

def _state_patch_path(sweeps_dir: Path, group: str) -> Path:
    return sweeps_dir / "active" / f"{group}.state_patch.json"


def _stage_state_patch(sweeps_dir: Path, group: str, run_idx: int,
                       fields: dict) -> None:
    """Stage a per-run patch for monitor to merge into state.json next cycle."""
    p = _state_patch_path(sweeps_dir, group)
    doc = {}
    if p.is_file():
        try:
            doc = json.loads(p.read_text())
        except Exception:
            doc = {}
    doc.setdefault(str(run_idx), {}).update(fields)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(str(tmp), str(p))


def consume_state_patches(sweeps_dir: Path, group: str) -> dict:
    """Called by monitor: returns the staged patches and deletes the file.

    Returns {} if no patch exists. Patch shape: ``{run_idx_str: {field: value}}``.
    Monitor should merge into its state["runs"][k] before classification.
    """
    p = _state_patch_path(sweeps_dir, group)
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text())
    except Exception:
        logger.warning(f"[migration] couldn't parse state patch {p}")
        doc = {}
    try:
        p.unlink()
    except Exception:
        pass
    return doc


# ---------------------------------------------------------------------------
# Per-sweep migration loop
# ---------------------------------------------------------------------------

def _select_candidates(
    expected: dict, state: dict, max_count: int,
) -> list[int]:
    """Pick up to ``max_count`` PENDING ou_bcs_normal cells eligible for
    migration. Order: ascending run_idx (deterministic, FIFO-ish)."""
    state_runs = state.get("runs", {}) or {}
    slurm_arrays = expected.get("slurm_arrays") or {}
    partition_per_cell = expected.get("partition_per_cell") or {}
    sweep_partition = (expected.get("slurm") or {}).get("partition", "ou_bcs_normal")

    candidates = []
    # Sort run_idx numerically (slurm_arrays keys are strings)
    for k in sorted(slurm_arrays.keys(), key=lambda s: int(s)):
        run_idx = int(k)
        # Skip cells that aren't on ou_bcs_normal
        cell_partition = partition_per_cell.get(k, sweep_partition)
        if cell_partition != "ou_bcs_normal":
            continue
        eligible, _ = _is_clean_for_migration(state_runs, run_idx)
        if eligible:
            candidates.append(run_idx)
        if len(candidates) >= max_count:
            break
    return candidates


def process_sweep_migrations(
    expected_path: Path, sweeps_dir: Path, budget: int,
) -> tuple[int, int]:
    """Run migration for one sweep, given a precomputed budget. Returns
    (migrated_count, attempted_count). Skips if migrate_to absent."""
    try:
        expected = json.loads(expected_path.read_text())
    except Exception as e:
        logger.warning(f"[migration] couldn't parse {expected_path}: {e}")
        return 0, 0

    if not expected.get("migrate_to"):
        return 0, 0
    group = expected.get("wandb", {}).get("group")
    if not group:
        return 0, 0

    state_path = expected_path.with_name(f"{group}.state.json")
    if not state_path.is_file():
        # No monitor state yet → cannot tell which cells are PENDING. Skip
        # this cycle — monitor will write state.json shortly.
        return 0, 0
    try:
        state = json.loads(state_path.read_text())
    except Exception as e:
        logger.warning(f"[migration {group}] state.json parse fail: {e}")
        return 0, 0

    candidates = _select_candidates(expected, state, max_count=budget)
    if not candidates:
        return 0, 0

    migrated = 0
    for run_idx in candidates:
        # Re-read expected.json each time so concurrent updates are visible
        # (defensive — controller flock means we're single-writer, but the
        # in-memory copy gets stale after our own _apply_migration writes).
        expected = json.loads(expected_path.read_text())
        ok = migrate_one(
            group, run_idx, expected_path, expected,
            state.get("runs", {}), sweeps_dir,
        )
        if ok:
            migrated += 1
    return migrated, len(candidates)


def process_migrations(sweeps_dir: Path) -> dict:
    """Top-level entry point: iterate active migrate-eligible sweeps and
    perform any eligible migrations. Returns summary counts for logging.

    Called from the controller cycle, AFTER monitor and BEFORE dispatch
    of analyses. Budget is global (4 GPUs across all sweeps); FIFO by
    sweep filename order.
    """
    summary = {"sweeps_checked": 0, "migrated": 0, "attempted": 0}
    active_dir = sweeps_dir / "active"
    if not active_dir.is_dir():
        return summary

    initial_count = count_running_mit_jobs()
    budget = max(0, MIT_BUDGET - initial_count)
    if budget == 0:
        logger.info(
            f"[migration] mit_normal_gpu at QOS limit "
            f"({initial_count}/{MIT_BUDGET}), nothing to migrate"
        )
        return summary

    expected_files = sorted(p for p in active_dir.iterdir()
                            if p.name.endswith(".expected.json"))
    for ep in expected_files:
        if budget == 0:
            break
        summary["sweeps_checked"] += 1
        m, a = process_sweep_migrations(ep, sweeps_dir, budget)
        summary["migrated"] += m
        summary["attempted"] += a
        budget = max(0, budget - m)
    if summary["migrated"] or summary["attempted"]:
        logger.info(
            f"[migration] cycle done: {summary['migrated']} migrated / "
            f"{summary['attempted']} attempted across "
            f"{summary['sweeps_checked']} sweeps"
        )
    return summary


# ---------------------------------------------------------------------------
# Startup recovery — audit unfinished journal entries
# ---------------------------------------------------------------------------

def _audit_one_journal(journal_path: Path, sweeps_dir: Path) -> dict:
    """For one sweep's journal, find any run_idx whose latest stage is
    'submitted_pending_expected_update' (= dangerous in-flight) and
    reconcile by checking expected.json + squeue.

    Returns counts dict for logging.
    """
    counts = {"orphans_scanceled": 0, "completed": 0, "skipped": 0}
    entries = _journal_read_all(journal_path)
    if not entries:
        return counts
    latest = _journal_latest_per_run(entries)

    group = journal_path.name.replace(".migration.journal.jsonl", "")
    expected_path = sweeps_dir / "active" / f"{group}.expected.json"
    if not expected_path.is_file():
        return counts
    try:
        expected = json.loads(expected_path.read_text())
    except Exception:
        return counts
    slurm_arrays = expected.get("slurm_arrays", {})

    for run_idx, entry in latest.items():
        if entry.get("stage") != "submitted_pending_expected_update":
            continue  # not in dangerous state
        new_task = entry.get("new_task")
        if not new_task:
            counts["skipped"] += 1
            continue

        recorded = slurm_arrays.get(str(run_idx))
        if recorded == new_task:
            # expected.json was updated successfully before crash, just
            # never journaled "complete". Mark complete now.
            _journal_append(journal_path, {
                "run_idx": run_idx, "stage": "complete_via_audit",
                "new_task": new_task,
            })
            counts["completed"] += 1
            continue

        # expected.json doesn't know about new_task — it's an orphan
        state = _squeue_state(new_task)
        if state in ALIVE_STATES:
            logger.warning(
                f"[audit] orphan task {new_task} alive on mit for "
                f"{group}/r{run_idx}; scanceling"
            )
            try:
                subprocess.run(["scancel", new_task], check=False, timeout=10)
            except Exception as e:
                logger.error(f"[audit] orphan scancel failed: {e}")
            _journal_append(journal_path, {
                "run_idx": run_idx, "stage": "orphan_scanceled_by_audit",
                "new_task": new_task,
            })
            counts["orphans_scanceled"] += 1
        else:
            # Orphan already gone (timed out before audit, or never
            # scheduled). Monitor retry will bring the cell back.
            _journal_append(journal_path, {
                "run_idx": run_idx, "stage": "orphan_already_gone",
                "new_task": new_task,
            })
            counts["completed"] += 1
    return counts


def audit_journals(sweeps_dir: Path) -> dict:
    """Scan all active sweep journals at controller startup.

    Reconciles half-finished migrations from a prior controller crash.
    Should be called BEFORE the first cycle runs anything else.
    """
    summary = {"orphans_scanceled": 0, "completed": 0, "skipped": 0}
    active_dir = sweeps_dir / "active"
    if not active_dir.is_dir():
        return summary
    for jp in sorted(active_dir.glob("*.migration.journal.jsonl")):
        c = _audit_one_journal(jp, sweeps_dir)
        for k, v in c.items():
            summary[k] += v
    if any(summary.values()):
        logger.info(f"[audit] startup summary: {summary}")
    return summary
