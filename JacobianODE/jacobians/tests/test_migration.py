"""Unit tests for migration.py — candidate selection, journal recovery,
atomic update, budget, and orphan cleanup.

squeue / scancel / sbatch are mocked. The tests focus on the state-
machine logic and crash-recovery semantics, not on the SLURM behavior
itself.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from JacobianODE.jacobians.tuning.migration import (
    _apply_migration,
    _atomic_update_json,
    _audit_one_journal,
    _is_clean_for_migration,
    _journal_append,
    _journal_path,
    _journal_read_all,
    _select_candidates,
    audit_journals,
    consume_state_patches,
    count_running_mit_jobs,
    migrate_one,
    process_migrations,
)
from JacobianODE.jacobians.tuning.cell_submit import MIT_NORMAL_GPU


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_expected(group="g", n_cells=4, partition="ou_bcs_normal"):
    return {
        "wandb": {"group": group, "project": "test"},
        "git": {"repo_dir": "/repo"},
        "slurm": {"partition": partition, "gres": "gpu:1",
                  "cpus_per_task": 4, "mem": "16GB", "timeout_min": 180},
        "hydra": {"resolved_runs": [
            {"run_idx": i, "experiment": "exp_a",
             "overrides": [f"lc=1e-{i}"]}
            for i in range(n_cells)
        ]},
        "slurm_arrays": {str(i): f"1000_{i}" for i in range(n_cells)},
        "migrate_to": {"partition": "mit_normal_gpu"},
    }


def _make_state(n_cells=4, default_state="PENDING"):
    return {
        "monitor_cycle": 5,
        "runs": {
            str(i): {
                "slurm_job_ids": [f"1000_{i}"],
                "last_slurm_state": default_state,
                "classification": "pending",
                "consecutive_missing_cycles": 0,
                "ever_alive": True,
                "attempts": 1,
            }
            for i in range(n_cells)
        },
    }


# ---------------------------------------------------------------------------
# _is_clean_for_migration
# ---------------------------------------------------------------------------

class TestIsCleanForMigration:
    def test_pending_clean_cell_eligible(self):
        state = _make_state()
        eligible, _ = _is_clean_for_migration(state["runs"], 0)
        assert eligible

    def test_already_migrated_skipped(self):
        state = _make_state()
        state["runs"]["0"]["migrated"] = 1
        eligible, reason = _is_clean_for_migration(state["runs"], 0)
        assert not eligible
        assert "already_migrated" in reason

    def test_multiple_slurm_jobs_skipped(self):
        """Cell with 2+ slurm_job_ids = monitor retry already in flight."""
        state = _make_state()
        state["runs"]["0"]["slurm_job_ids"] = ["1000_0", "9999"]
        eligible, reason = _is_clean_for_migration(state["runs"], 0)
        assert not eligible
        assert "monitor_retry_in_flight" in reason

    def test_running_cell_skipped(self):
        """Only PENDING cells are eligible — RUNNING means we'd interrupt work."""
        state = _make_state(default_state="RUNNING")
        eligible, reason = _is_clean_for_migration(state["runs"], 0)
        assert not eligible
        assert "not_pending" in reason

    def test_none_last_state_eligible(self):
        """last_slurm_state==None is permitted (not all monitor versions
        populate it for the canonical task). migrate_one re-checks via
        real-time squeue before acting."""
        state = _make_state()
        state["runs"]["0"]["last_slurm_state"] = None
        eligible, _ = _is_clean_for_migration(state["runs"], 0)
        assert eligible

    def test_terminal_classification_skipped(self):
        for cls in ("done_finished", "done_walltime", "failed_exhausted"):
            state = _make_state()
            state["runs"]["0"]["classification"] = cls
            eligible, reason = _is_clean_for_migration(state["runs"], 0)
            assert not eligible, f"failed for cls={cls}"

    def test_missing_state_entry_skipped(self):
        eligible, reason = _is_clean_for_migration({}, 0)
        assert not eligible
        assert reason == "no_state_entry"


# ---------------------------------------------------------------------------
# _select_candidates
# ---------------------------------------------------------------------------

class TestSelectCandidates:
    def test_returns_eligible_cells_in_run_idx_order(self):
        e = _make_expected(n_cells=4)
        s = _make_state(n_cells=4)
        cands = _select_candidates(e, s, max_count=10)
        assert cands == [0, 1, 2, 3]

    def test_respects_max_count(self):
        e = _make_expected(n_cells=10)
        s = _make_state(n_cells=10)
        cands = _select_candidates(e, s, max_count=3)
        assert len(cands) == 3
        assert cands == [0, 1, 2]

    def test_filters_by_per_cell_partition(self):
        """If partition_per_cell records that cell 0 is on mit (already
        migrated), it's not a candidate."""
        e = _make_expected(n_cells=4)
        e["partition_per_cell"] = {"0": "mit_normal_gpu",
                                    "1": "ou_bcs_normal"}
        s = _make_state(n_cells=4)
        cands = _select_candidates(e, s, max_count=10)
        assert 0 not in cands
        assert 1 in cands

    def test_filters_by_clean_state(self):
        e = _make_expected(n_cells=4)
        s = _make_state(n_cells=4)
        s["runs"]["1"]["migrated"] = 1
        s["runs"]["2"]["last_slurm_state"] = "RUNNING"
        cands = _select_candidates(e, s, max_count=10)
        assert cands == [0, 3]


# ---------------------------------------------------------------------------
# Atomic update
# ---------------------------------------------------------------------------

class TestAtomicUpdate:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "expected.json"
        p.write_text(json.dumps({"a": 1}))
        _atomic_update_json(p, lambda d: d.update(b=2))
        loaded = json.loads(p.read_text())
        assert loaded == {"a": 1, "b": 2}

    def test_no_partial_write_on_mutator_failure(self, tmp_path):
        """If the mutator raises, the file must be untouched."""
        p = tmp_path / "expected.json"
        p.write_text(json.dumps({"a": 1}))

        def bad_mutator(d):
            d["b"] = 2
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            _atomic_update_json(p, bad_mutator)
        assert json.loads(p.read_text()) == {"a": 1}


class TestApplyMigration:
    def test_updates_slurm_arrays_and_partition_per_cell(self):
        """_apply_migration must update slurm_arrays + partition_per_cell
        + migrations counter ONLY. Top-level expected.slurm must NOT
        change — it's read by monitor.resubmit_run for every cell, and
        leaking one migrated cell's partition into the rest causes
        ALL retries to land on mit (saturating QOS for unrelated cells).
        """
        e = _make_expected(n_cells=4)
        original_slurm = dict(e["slurm"])
        _apply_migration(e, 2, "9999", MIT_NORMAL_GPU)
        assert e["slurm_arrays"]["2"] == "9999"
        assert e["partition_per_cell"]["2"] == "mit_normal_gpu"
        # Top-level slurm UNCHANGED — per-cell partition lives in
        # partition_per_cell, NOT here.
        assert e["slurm"] == original_slurm
        # Migration counter
        assert e["migrations"]["2"] == 1

    def test_double_migration_increments(self):
        e = _make_expected(n_cells=4)
        _apply_migration(e, 0, "9999", MIT_NORMAL_GPU)
        _apply_migration(e, 0, "10000", MIT_NORMAL_GPU)
        assert e["migrations"]["0"] == 2


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

class TestJournal:
    def test_append_and_read_round_trip(self, tmp_path):
        p = tmp_path / "g.migration.journal.jsonl"
        _journal_append(p, {"run_idx": 0, "stage": "a"})
        _journal_append(p, {"run_idx": 0, "stage": "b"})
        _journal_append(p, {"run_idx": 1, "stage": "c"})
        entries = _journal_read_all(p)
        assert len(entries) == 3
        assert all("at" in e for e in entries)  # timestamp injected

    def test_read_all_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "g.migration.journal.jsonl"
        p.write_text(
            json.dumps({"run_idx": 0, "stage": "ok"}) + "\n"
            + "garbage\n"
            + json.dumps({"run_idx": 1, "stage": "ok2"}) + "\n"
        )
        assert len(_journal_read_all(p)) == 2

    def test_returns_empty_when_missing(self, tmp_path):
        assert _journal_read_all(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# Journal audit (controller startup recovery)
# ---------------------------------------------------------------------------

class TestAuditOneJournal:
    def _setup(self, tmp_path):
        active = tmp_path / "active"
        active.mkdir()
        expected = _make_expected(n_cells=4)
        (active / "g.expected.json").write_text(json.dumps(expected))
        return tmp_path

    def test_completed_migration_marked_complete(self, tmp_path):
        """expected.json shows new_task → migration finished cleanly,
        just never journaled. Should be no-op (mark complete)."""
        sweeps = self._setup(tmp_path)
        # Simulate: expected.json updated to new_task=9999 for run 0
        ep = sweeps / "active" / "g.expected.json"
        doc = json.loads(ep.read_text())
        doc["slurm_arrays"]["0"] = "9999"
        ep.write_text(json.dumps(doc))

        jp = sweeps / "active" / "g.migration.journal.jsonl"
        _journal_append(jp, {
            "run_idx": 0, "stage": "submitted_pending_expected_update",
            "old_task": "1000_0", "new_task": "9999",
        })
        with patch("subprocess.run") as mock_run:
            counts = _audit_one_journal(jp, sweeps)
        assert counts["completed"] == 1
        assert counts["orphans_scanceled"] == 0
        # No scancel called
        scancel_calls = [c for c in mock_run.call_args_list
                          if "scancel" in c[0][0][0]]
        assert len(scancel_calls) == 0

    def test_orphan_alive_scanceled(self, tmp_path):
        """expected.json doesn't have new_task and squeue says it's alive
        → orphan, must scancel."""
        sweeps = self._setup(tmp_path)
        # expected.json still has old task, NOT new_task
        jp = sweeps / "active" / "g.migration.journal.jsonl"
        _journal_append(jp, {
            "run_idx": 0, "stage": "submitted_pending_expected_update",
            "old_task": "1000_0", "new_task": "9999",
        })
        # First mock: squeue says alive. Second: scancel succeeds.
        squeue_response = MagicMock(stdout="PENDING\n", returncode=0)
        scancel_response = MagicMock(stdout="", returncode=0)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [squeue_response, scancel_response]
            counts = _audit_one_journal(jp, sweeps)
        assert counts["orphans_scanceled"] == 1
        # Verify one of the calls was a scancel
        scancel_calls = [c for c in mock_run.call_args_list
                          if c[0][0][0] == "scancel"]
        assert len(scancel_calls) == 1

    def test_orphan_already_gone_no_action(self, tmp_path):
        """If the orphan is no longer in squeue, nothing to do."""
        sweeps = self._setup(tmp_path)
        jp = sweeps / "active" / "g.migration.journal.jsonl"
        _journal_append(jp, {
            "run_idx": 0, "stage": "submitted_pending_expected_update",
            "old_task": "1000_0", "new_task": "9999",
        })
        # squeue returns empty
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            counts = _audit_one_journal(jp, sweeps)
        assert counts["completed"] == 1
        assert counts["orphans_scanceled"] == 0

    def test_complete_stage_ignored(self, tmp_path):
        """Already-complete entries don't trigger any action."""
        sweeps = self._setup(tmp_path)
        jp = sweeps / "active" / "g.migration.journal.jsonl"
        _journal_append(jp, {
            "run_idx": 0, "stage": "complete",
            "old_task": "1000_0", "new_task": "9999",
        })
        with patch("subprocess.run") as mock_run:
            counts = _audit_one_journal(jp, sweeps)
        assert counts["orphans_scanceled"] == 0
        assert counts["completed"] == 0  # no work done
        assert mock_run.call_count == 0


# ---------------------------------------------------------------------------
# count_running_mit_jobs
# ---------------------------------------------------------------------------

class TestCountRunningMitJobs:
    def test_counts_lines(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345\n12346\n12347\n", returncode=0,
            )
            assert count_running_mit_jobs() == 3

    def test_empty_returns_zero(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            assert count_running_mit_jobs() == 0

    def test_squeue_missing_returns_full_budget(self):
        """If squeue is missing (or times out), be conservative and skip
        migration this cycle."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("squeue not found")
            assert count_running_mit_jobs() == 4  # MIT_BUDGET


# ---------------------------------------------------------------------------
# State-patch sidecar
# ---------------------------------------------------------------------------

class TestStatePatch:
    def test_consume_returns_and_deletes(self, tmp_path):
        active = tmp_path / "active"
        active.mkdir()
        # Write a sidecar
        (active / "g.state_patch.json").write_text(
            json.dumps({"0": {"migrated": 1}, "2": {"migrated": 1}})
        )
        out = consume_state_patches(tmp_path, "g")
        assert out == {"0": {"migrated": 1}, "2": {"migrated": 1}}
        # Sidecar should be gone after consumption
        assert not (active / "g.state_patch.json").is_file()

    def test_consume_missing_returns_empty(self, tmp_path):
        (tmp_path / "active").mkdir()
        assert consume_state_patches(tmp_path, "g") == {}


# ---------------------------------------------------------------------------
# migrate_one — end-to-end happy path & failure modes
# ---------------------------------------------------------------------------

class TestMigrateOne:
    def _setup_sweep(self, tmp_path):
        active = tmp_path / "active"
        active.mkdir()
        expected = _make_expected(n_cells=4)
        ep = active / "g.expected.json"
        ep.write_text(json.dumps(expected))
        return tmp_path, ep, expected

    def test_happy_path(self, tmp_path):
        sweeps, ep, expected = self._setup_sweep(tmp_path)
        state_runs = _make_state()["runs"]

        def fake_subprocess_run(argv, **kw):
            cmd = argv[0]
            if cmd == "squeue":
                # Initial liveness check + post-scancel polls
                return MagicMock(stdout="", returncode=0)
            if cmd == "scancel":
                return MagicMock(stdout="", returncode=0)
            if cmd == "sbatch":
                return MagicMock(stdout="9999\n", returncode=0)
            raise AssertionError(f"unexpected: {argv}")

        with patch("subprocess.run", side_effect=fake_subprocess_run), \
             patch("JacobianODE.jacobians.tuning.migration._squeue_state",
                   return_value="PENDING"), \
             patch("JacobianODE.jacobians.tuning.migration._poll_gone",
                   return_value=True):
            ok = migrate_one("g", 0, ep, expected, state_runs, sweeps)
        assert ok
        # expected.json updated
        new_doc = json.loads(ep.read_text())
        assert new_doc["slurm_arrays"]["0"] == "9999"
        assert new_doc["partition_per_cell"]["0"] == "mit_normal_gpu"
        # journal has the complete entry
        journal = _journal_read_all(_journal_path(sweeps, "g"))
        stages = [e["stage"] for e in journal]
        assert "complete" in stages

    def test_aborts_when_no_longer_pending(self, tmp_path):
        """Race: state.json said PENDING, but by the time migrate_one runs
        the cell already started — must abort cleanly."""
        sweeps, ep, expected = self._setup_sweep(tmp_path)
        state_runs = _make_state()["runs"]
        with patch("JacobianODE.jacobians.tuning.migration._squeue_state",
                   return_value="RUNNING"):
            ok = migrate_one("g", 0, ep, expected, state_runs, sweeps)
        assert not ok
        # No scancel issued
        new_doc = json.loads(ep.read_text())
        assert new_doc["slurm_arrays"]["0"] == "1000_0"  # unchanged

    def test_sbatch_failure_aborts_cleanly(self, tmp_path):
        """scancel succeeded but sbatch on mit failed → cell is now NOT
        in slurm. We log and return False; monitor retry brings cell back
        via expected.slurm (still ou_bcs_normal at this point)."""
        sweeps, ep, expected = self._setup_sweep(tmp_path)
        state_runs = _make_state()["runs"]

        def fake_run(argv, **kw):
            cmd = argv[0]
            if cmd == "scancel":
                return MagicMock(stdout="", returncode=0)
            if cmd == "sbatch":
                raise subprocess.CalledProcessError(
                    1, argv, stderr="sbatch failed",
                )
            return MagicMock(stdout="", returncode=0)

        with patch("subprocess.run", side_effect=fake_run), \
             patch("JacobianODE.jacobians.tuning.migration._squeue_state",
                   return_value="PENDING"), \
             patch("JacobianODE.jacobians.tuning.migration._poll_gone",
                   return_value=True):
            ok = migrate_one("g", 0, ep, expected, state_runs, sweeps)
        assert not ok
        # expected.json unchanged (no _apply_migration called)
        new_doc = json.loads(ep.read_text())
        assert new_doc["slurm_arrays"]["0"] == "1000_0"
        # Journal records the abort
        journal = _journal_read_all(_journal_path(sweeps, "g"))
        stages = [e["stage"] for e in journal]
        assert "abort_sbatch_failed" in stages
        assert "complete" not in stages

    def test_expected_update_failure_scancels_orphan(self, tmp_path):
        """sbatch succeeded but expected.json write failed → must scancel
        the new task to prevent duplicate next cycle."""
        sweeps, ep, expected = self._setup_sweep(tmp_path)
        state_runs = _make_state()["runs"]

        sbatch_calls = []
        scancel_calls = []

        def fake_run(argv, **kw):
            if argv[0] == "scancel":
                scancel_calls.append(argv)
                return MagicMock(stdout="", returncode=0)
            if argv[0] == "sbatch":
                sbatch_calls.append(argv)
                return MagicMock(stdout="9999\n", returncode=0)
            return MagicMock(stdout="", returncode=0)

        with patch("subprocess.run", side_effect=fake_run), \
             patch("JacobianODE.jacobians.tuning.migration._squeue_state",
                   return_value="PENDING"), \
             patch("JacobianODE.jacobians.tuning.migration._poll_gone",
                   return_value=True), \
             patch("JacobianODE.jacobians.tuning.migration._atomic_update_json",
                   side_effect=OSError("disk full")):
            ok = migrate_one("g", 0, ep, expected, state_runs, sweeps)
        assert not ok
        # Two scancels: original + orphan
        assert len(scancel_calls) == 2
        # Orphan scancel was for the new task ID
        assert "9999" in scancel_calls[-1]
        # Journal records the orphan
        journal = _journal_read_all(_journal_path(sweeps, "g"))
        stages = [e["stage"] for e in journal]
        assert "orphan_scanceled" in stages


# ---------------------------------------------------------------------------
# process_migrations — top-level
# ---------------------------------------------------------------------------

class TestProcessMigrations:
    def test_skips_sweep_without_migrate_to(self, tmp_path):
        active = tmp_path / "active"
        active.mkdir()
        e = _make_expected(n_cells=4)
        del e["migrate_to"]
        (active / "g.expected.json").write_text(json.dumps(e))
        with patch("JacobianODE.jacobians.tuning.migration."
                   "count_running_mit_jobs", return_value=0):
            summary = process_migrations(tmp_path)
        # We did check the sweep but skipped it — counted but no migrations
        assert summary["migrated"] == 0

    def test_skips_when_budget_zero(self, tmp_path):
        active = tmp_path / "active"
        active.mkdir()
        (active / "g.expected.json").write_text(
            json.dumps(_make_expected(n_cells=4))
        )
        with patch("JacobianODE.jacobians.tuning.migration."
                   "count_running_mit_jobs", return_value=4):
            summary = process_migrations(tmp_path)
        assert summary["sweeps_checked"] == 0  # bailed before scanning
        assert summary["migrated"] == 0

    def test_no_state_file_skips(self, tmp_path):
        """Sweep with migrate_to but no state.json yet (monitor hasn't
        run) → skip this cycle, wait for next."""
        active = tmp_path / "active"
        active.mkdir()
        (active / "g.expected.json").write_text(
            json.dumps(_make_expected(n_cells=4))
        )
        # No state.json
        with patch("JacobianODE.jacobians.tuning.migration."
                   "count_running_mit_jobs", return_value=0):
            summary = process_migrations(tmp_path)
        assert summary["migrated"] == 0


# ---------------------------------------------------------------------------
# audit_journals (top-level)
# ---------------------------------------------------------------------------

class TestAuditJournals:
    def test_no_journals_returns_zero_summary(self, tmp_path):
        (tmp_path / "active").mkdir()
        summary = audit_journals(tmp_path)
        assert summary == {
            "orphans_scanceled": 0, "completed": 0, "skipped": 0,
        }

    def test_iterates_multiple_groups(self, tmp_path):
        active = tmp_path / "active"
        active.mkdir()
        # Two groups, both with a clean (no-op) journal entry
        for grp in ("g1", "g2"):
            (active / f"{grp}.expected.json").write_text(
                json.dumps(_make_expected(group=grp, n_cells=2))
            )
            jp = active / f"{grp}.migration.journal.jsonl"
            _journal_append(jp, {
                "run_idx": 0, "stage": "complete",
                "old_task": "1000_0", "new_task": "9999",
            })
        summary = audit_journals(tmp_path)
        # Both already complete → nothing scanceled, nothing reconciled
        assert summary == {
            "orphans_scanceled": 0, "completed": 0, "skipped": 0,
        }
