"""Tests for the monitor's missing-cycles hysteresis.

A single squeue snapshot can transiently miss an alive array task
(scheduler load, NFS hiccup) — without hysteresis this triggered
false-positive ghost retries even when the original SLURM job was still
running fine. The hysteresis requires N consecutive misses before the
slot is escalated into the failed_retrying lifecycle.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from JacobianODE.jacobians.tuning import monitor as mon


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep_paths(tmp_path):
    """Build the sweeps_dir layout check_sweep expects + write expected.json."""
    sweeps_dir = tmp_path / "sweeps"
    (sweeps_dir / "active").mkdir(parents=True)
    (sweeps_dir / "done").mkdir()
    (sweeps_dir / "processed").mkdir()
    (sweeps_dir / "failed").mkdir()

    expected = {
        "wandb": {"entity": "e", "project": "p", "group": "g"},
        "experiment_metadata": {},
        "hydra": {
            "experiments": ["g"],
            "overrides_template": [],
            "resolved_runs": [
                {"run_idx": 0, "experiment": "g", "overrides": []},
            ],
        },
        "expected_run_count": 1,
        "slurm": {"timeout_min": 180},
        "slurm_arrays": {"0": "9999"},  # task_id will be "9999_0"
        "retry": {
            "cap_per_run": 2,
            "min_elapsed_before_done_sec": 0,
            "missing_cycles_threshold": 2,
        },
        "launched_at": "2026-01-01T00:00:00Z",
    }
    expected_path = sweeps_dir / "active" / "g.expected.json"
    expected_path.write_text(json.dumps(expected))

    return {
        "sweeps_dir": sweeps_dir,
        "expected_path": expected_path,
        "state_path": sweeps_dir / "active" / "g.state.json",
    }


def _wandb_run(run_id, state="running", config=None):
    class _R:
        def __init__(self):
            self.id = run_id
            self.state = state
            self.config = config or {"experiment": "g"}
            self.heartbeat_at = "2026-01-01T00:01:00Z"
            self.updated_at = "2026-01-01T00:01:00Z"
    return _R()


def _seed_state(state_path: Path, run0_overrides: dict):
    """Write a state.json so check_sweep starts from a known per-slot state."""
    base = {
        "wandb_run_ids": [],
        "slurm_job_ids": [],
        "attempts": 0,
        "last_wandb_state": None,
        "last_slurm_state": None,
        "classification": "pending",
        "terminal": False,
        "consecutive_missing_cycles": 0,
    }
    base.update(run0_overrides)
    state = {
        "schema_version": 1,
        "group": "g",
        "monitor_cycle": run0_overrides.pop("_monitor_cycle", 1),
        "runs": {"0": base},
        "summary": {},
    }
    state_path.write_text(json.dumps(state))


def _drive_cycle(sweep_paths, squeue_states, wandb_runs=None):
    """Run one check_sweep cycle with mocked external queries."""
    if wandb_runs is None:
        wandb_runs = []
    with (
        patch.object(mon, "query_wandb_runs", return_value=wandb_runs),
        patch.object(mon, "query_squeue_states", return_value=squeue_states),
        # Don't actually resubmit anything in tests (we only assert
        # classification, not the resubmit path).
        patch.object(mon, "resubmit_run", return_value=""),
    ):
        mon.check_sweep(sweep_paths["expected_path"], sweep_paths["sweeps_dir"])
    return json.loads(sweep_paths["state_path"].read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_alive_task_resets_counter(sweep_paths):
    """Counter resets to 0 when task is alive in squeue."""
    _seed_state(sweep_paths["state_path"], {
        "consecutive_missing_cycles": 5,  # pre-existing high count
        "_monitor_cycle": 2,
    })

    state = _drive_cycle(sweep_paths, {"9999_0": "RUNNING"})

    assert state["runs"]["0"]["consecutive_missing_cycles"] == 0
    assert state["runs"]["0"]["classification"] == "running"


def test_single_miss_does_not_retry(sweep_paths):
    """A single missed squeue snapshot must NOT trigger failed_retrying.

    Load-bearing: original SLURM task is alive but squeue briefly
    returned empty. Pre-hysteresis this would have queued a ghost retry.
    """
    _seed_state(sweep_paths["state_path"], {
        "wandb_run_ids": ["wid_a"],
        "attempts": 1,
        "last_wandb_state": "running",
        "last_slurm_state": "RUNNING",
        "classification": "running",
        "ever_alive": True,
        "consecutive_missing_cycles": 0,
        "_monitor_cycle": 5,
    })

    state = _drive_cycle(
        sweep_paths, squeue_states={},
        wandb_runs=[_wandb_run("wid_a", state="running")],
    )

    assert state["runs"]["0"]["consecutive_missing_cycles"] == 1
    assert state["runs"]["0"]["classification"] != "failed_retrying", (
        f"single miss should not reclassify; got "
        f"{state['runs']['0']['classification']}"
    )


def test_n_consecutive_misses_triggers_retry(sweep_paths):
    """After N consecutive misses (default 2), reclassify."""
    _seed_state(sweep_paths["state_path"], {
        "wandb_run_ids": ["wid_a"],
        "attempts": 1,
        "last_wandb_state": "running",
        "last_slurm_state": "RUNNING",
        "classification": "running",
        "ever_alive": True,
        "consecutive_missing_cycles": 1,  # already missed once before
        "_monitor_cycle": 5,
    })

    state = _drive_cycle(
        sweep_paths, squeue_states={},
        wandb_runs=[_wandb_run("wid_a", state="running")],
    )

    assert state["runs"]["0"]["consecutive_missing_cycles"] == 2
    assert state["runs"]["0"]["classification"] == "failed_retrying"


def test_alive_after_one_miss_resets(sweep_paths):
    """A miss followed by an alive sighting resets the counter to 0."""
    _seed_state(sweep_paths["state_path"], {
        "wandb_run_ids": ["wid_a"],
        "attempts": 1,
        "last_wandb_state": "running",
        "last_slurm_state": "RUNNING",
        "classification": "running",
        "ever_alive": True,
        "consecutive_missing_cycles": 1,
        "_monitor_cycle": 5,
    })

    state = _drive_cycle(
        sweep_paths, squeue_states={"9999_0": "RUNNING"},
        wandb_runs=[_wandb_run("wid_a", state="running")],
    )

    assert state["runs"]["0"]["consecutive_missing_cycles"] == 0
    assert state["runs"]["0"]["classification"] == "running"


def test_single_task_array_bare_id_treated_as_alive(sweep_paths):
    """SLURM compresses size-1 arrays to bare array_id (no _0 suffix).

    Bug: query_squeue_states populated only the bare key, but
    _slurm_task_id constructed '<array_id>_0', so the lookup always
    missed and the slot was classified as failed every cycle —
    defeating the hysteresis. Fix: synthesize the canonical _0 key
    from any bare array_id reported by squeue.
    """
    _seed_state(sweep_paths["state_path"], {
        "wandb_run_ids": ["wid_a"], "attempts": 1,
        "last_wandb_state": "running", "last_slurm_state": "RUNNING",
        "classification": "running", "ever_alive": True,
        "consecutive_missing_cycles": 0,
        "_monitor_cycle": 2,
    })

    # squeue returns ONLY the bare array_id (no _0) — what SLURM does
    # for single-task arrays even with the -r flag.
    from JacobianODE.jacobians.tuning import monitor as mon

    # Patch query_squeue_states to return that real-world output. Then
    # check_sweep should still classify the slot as running because the
    # bare-id-as-alias logic kicks in inside query_squeue_states.
    # Here we go through the patched version (the synthesizer is built
    # into query_squeue_states) by emulating the synthesized output:
    state = _drive_cycle(
        sweep_paths,
        squeue_states={"9999": "RUNNING", "9999_0": "RUNNING"},
        wandb_runs=[_wandb_run("wid_a")],
    )

    assert state["runs"]["0"]["consecutive_missing_cycles"] == 0
    assert state["runs"]["0"]["classification"] == "running"


def test_query_squeue_states_synthesizes_bare_array_id_alias(monkeypatch):
    """query_squeue_states must add a `<array_id>_0` alias for any bare
    array_id reported by squeue (the single-task-array case)."""
    import subprocess
    from JacobianODE.jacobians.tuning import monitor as mon

    fake_output = (
        "12637732|RUNNING\n"      # bare (single-task array)
        "12345_0|RUNNING\n"       # already in canonical form
        "12345_1|PENDING\n"
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: fake_output)
    states = mon.query_squeue_states()
    # Bare array_id present
    assert states["12637732"] == "RUNNING"
    # Synthesized alias present with same state
    assert states["12637732_0"] == "RUNNING"
    # Canonical-form entries unchanged
    assert states["12345_0"] == "RUNNING"
    assert states["12345_1"] == "PENDING"
    # Don't synthesize an alias for already-canonical entries
    assert "12345" not in states
