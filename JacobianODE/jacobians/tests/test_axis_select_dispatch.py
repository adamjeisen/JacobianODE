"""Tests for axis_select_dispatch — the scout → grid chain step.

Mirrors the test conventions of test_two_stage_cull (Mock wandb run
class, reuses the same metric scan_history shape).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from JacobianODE.jacobians.tuning.axis_select_dispatch import (
    DEFAULT_METRIC,
    bucket_and_rank,
    chain_dispatched_marker,
    main as axis_select_main,
    select_top_k,
    _stringify_axis_value,
)


# ---------------------------------------------------------------------------
# MockRun: stand-in for a wandb Run with config + scan_history
# ---------------------------------------------------------------------------

class _MockRun:
    def __init__(self, run_id, state, history, config):
        self.id = run_id
        self.state = state
        self._history = history
        self.config = config

    def scan_history(self, keys):
        for row in self._history:
            yield {k: row.get(k) for k in keys}


def _make_runs():
    """7 runs across 4 distinct n_delays values, with varying loss + states.

    n_delays=2: one finished run, best=0.50 → bucket best 0.50
    n_delays=4: two finished runs, best=0.10 → bucket best 0.10  ← winner
    n_delays=6: two finished runs, best=0.30 → bucket best 0.30
    n_delays=8: one CRASHED run with valid loss=0.01 → INCLUDED, bucket best 0.01
    n_delays=9: one RUNNING run (no terminal loss yet) → skipped

    Exercises the policy: any non-``running`` state is eligible if it has
    a recorded metric. Crashed cells (typically SLURM-timeout-killed)
    contribute their best-recorded loss; in-flight cells are skipped.
    """
    cfg = lambda nd: {
        "data": {"train_test_params": {
            "delay_embedding_params": {"n_delays": nd}}}}
    return [
        _MockRun("a", "finished", [{DEFAULT_METRIC: 0.50}], cfg(2)),
        _MockRun("b", "finished", [{DEFAULT_METRIC: 0.40},
                                    {DEFAULT_METRIC: 0.10}], cfg(4)),
        _MockRun("c", "finished", [{DEFAULT_METRIC: 0.20}], cfg(4)),
        _MockRun("d", "finished", [{DEFAULT_METRIC: 0.30}], cfg(6)),
        _MockRun("e", "finished", [{DEFAULT_METRIC: 0.45}], cfg(6)),
        _MockRun("crashed", "crashed", [{DEFAULT_METRIC: 0.01}], cfg(8)),
        _MockRun("running", "running", [{DEFAULT_METRIC: 0.05}], cfg(9)),
    ]


# ---------------------------------------------------------------------------
# bucket_and_rank
# ---------------------------------------------------------------------------

class TestBucketAndRank:
    def test_buckets_by_axis_value(self):
        runs = _make_runs()
        buckets, audit = bucket_and_rank(
            runs,
            axis="data.train_test_params.delay_embedding_params.n_delays",
            metric=DEFAULT_METRIC,
        )
        # Four live buckets (2, 4, 6, 8 — crashed n_delays=8 is included);
        # only the running n_delays=9 is skipped.
        assert set(buckets) == {"2", "4", "6", "8"}

    def test_per_bucket_best_metric(self):
        runs = _make_runs()
        buckets, _ = bucket_and_rank(
            runs,
            axis="data.train_test_params.delay_embedding_params.n_delays",
            metric=DEFAULT_METRIC,
        )
        assert buckets["2"]["best_metric"] == pytest.approx(0.50)
        assert buckets["4"]["best_metric"] == pytest.approx(0.10)
        assert buckets["6"]["best_metric"] == pytest.approx(0.30)
        # Crashed run with valid loss is included — bucket reflects it.
        assert buckets["8"]["best_metric"] == pytest.approx(0.01)
        # n_runs counts all live (non-skipped) runs in each bucket.
        assert buckets["4"]["n_runs"] == 2
        assert buckets["6"]["n_runs"] == 2
        assert buckets["2"]["n_runs"] == 1
        assert buckets["8"]["n_runs"] == 1

    def test_crashed_run_with_valid_loss_is_kept(self):
        runs = _make_runs()
        _, audit = bucket_and_rank(
            runs,
            axis="data.train_test_params.delay_embedding_params.n_delays",
            metric=DEFAULT_METRIC,
        )
        crashed = next(a for a in audit if a["run_id"] == "crashed")
        assert crashed["best_metric"] == pytest.approx(0.01)
        assert crashed["skip_reason"] is None

    def test_running_run_is_skipped(self):
        runs = _make_runs()
        _, audit = bucket_and_rank(
            runs,
            axis="data.train_test_params.delay_embedding_params.n_delays",
            metric=DEFAULT_METRIC,
        )
        running = next(a for a in audit if a["run_id"] == "running")
        assert running["best_metric"] is None
        assert running["skip_reason"] == "state=running"

    def test_run_with_missing_axis_is_skipped(self):
        runs = [_MockRun("noaxis", "finished",
                         [{DEFAULT_METRIC: 0.1}],
                         config={"unrelated": "config"})]
        buckets, audit = bucket_and_rank(
            runs,
            axis="data.train_test_params.delay_embedding_params.n_delays",
            metric=DEFAULT_METRIC,
        )
        assert buckets == {}
        assert audit[0]["skip_reason"].startswith("no_value_at_axis=")

    def test_run_with_no_metric_is_skipped(self):
        cfg = {"data": {"train_test_params": {
            "delay_embedding_params": {"n_delays": 4}}}}
        runs = [_MockRun("nometric", "finished", [{}, {}], cfg)]
        buckets, audit = bucket_and_rank(
            runs,
            axis="data.train_test_params.delay_embedding_params.n_delays",
            metric=DEFAULT_METRIC,
        )
        assert buckets == {}
        assert audit[0]["skip_reason"].startswith("no_value_for_")


# ---------------------------------------------------------------------------
# select_top_k
# ---------------------------------------------------------------------------

class TestSelectTopK:
    def test_picks_lowest_metric_first(self):
        buckets = {"2": {"best_metric": 0.50, "best_run_id": "a", "n_runs": 1},
                   "4": {"best_metric": 0.10, "best_run_id": "b", "n_runs": 2},
                   "6": {"best_metric": 0.30, "best_run_id": "d", "n_runs": 2}}
        chosen = select_top_k(buckets, top_k=2)
        assert chosen == ["4", "6"]

    def test_top_k_equals_bucket_count_returns_all(self):
        buckets = {"2": {"best_metric": 0.50, "best_run_id": "a", "n_runs": 1},
                   "4": {"best_metric": 0.10, "best_run_id": "b", "n_runs": 2}}
        chosen = select_top_k(buckets, top_k=2)
        assert set(chosen) == {"2", "4"}
        # Order is by metric: 4 (0.10) before 2 (0.50)
        assert chosen == ["4", "2"]

    def test_ties_broken_by_axis_value(self):
        buckets = {"6": {"best_metric": 0.10, "best_run_id": "x", "n_runs": 1},
                   "4": {"best_metric": 0.10, "best_run_id": "y", "n_runs": 1}}
        chosen = select_top_k(buckets, top_k=2)
        # Same metric → string-sort of axis-value key; "4" < "6"
        assert chosen == ["4", "6"]


# ---------------------------------------------------------------------------
# _stringify_axis_value
# ---------------------------------------------------------------------------

class TestStringifyAxisValue:
    def test_int_passes_through(self):
        assert _stringify_axis_value("6") == "6"

    def test_int_valued_float_strips_decimal(self):
        assert _stringify_axis_value("6.0") == "6"

    def test_real_float_preserved(self):
        assert _stringify_axis_value("0.05") == "0.05"

    def test_non_numeric_passes_through(self):
        assert _stringify_axis_value("uniform") == "uniform"


# ---------------------------------------------------------------------------
# chain_dispatched_marker
# ---------------------------------------------------------------------------

class TestChainDispatchedMarker:
    def test_path_construction(self, tmp_path):
        p = chain_dispatched_marker("foo_bar__sweep", sweeps_dir=tmp_path)
        assert p == tmp_path / "active" / "foo_bar__sweep.chain_dispatched"


# ---------------------------------------------------------------------------
# CLI dry-run end-to-end (mocks wandb.Api)
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_audit_no_sentinel(self, tmp_path, monkeypatch):
        """--dry-run produces an audit JSON with the expected chosen
        axis values and DOES NOT write the sentinel."""
        runs = _make_runs()

        # Mock wandb.Api so we don't hit the network. The dispatcher does:
        #   import wandb; api = wandb.Api()
        #   api.runs(...)
        # We provide a fake module + class that returns our mock runs.
        import sys, types
        fake_wandb = types.ModuleType("wandb")

        class _FakeAPI:
            def projects(self, entity):
                # Caller falls through if --project isn't given; we DO give it.
                return []
            def runs(self, project_path, filters=None, **kw):
                return runs

        fake_wandb.Api = lambda: _FakeAPI()
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        sweeps_dir = tmp_path / "sweeps"
        (sweeps_dir / "active").mkdir(parents=True)
        audit_out = tmp_path / "audit.json"

        rc = axis_select_main([
            "--group", "scout_group",
            "--axis", "data.train_test_params.delay_embedding_params.n_delays",
            "--top-k", "2",
            "--metric", DEFAULT_METRIC,
            "--next-experiment", "next_grid_yaml",
            "--project", "WMTask_test",
            "--sweeps-dir", str(sweeps_dir),
            "--audit-out", str(audit_out),
            "--dry-run",
            "--log-level", "WARNING",
        ])
        assert rc == 0

        doc = json.loads(audit_out.read_text())
        # Top-2 of {2: 0.50, 4: 0.10, 6: 0.30, 8: 0.01 (crashed)} = [8, 4].
        # Crashed run with valid loss outranks the finished n_delays=4 cell.
        assert doc["chosen_axis_values"] == ["8", "4"]
        assert doc["next_experiment"] == "next_grid_yaml"
        # No sentinel because dry-run
        assert not chain_dispatched_marker("scout_group", sweeps_dir).exists()

    def test_dry_run_with_two_stage_block_in_audit(self, tmp_path, monkeypatch):
        """--next-two-stage-json round-trips into the audit doc."""
        runs = _make_runs()
        import sys, types
        fake_wandb = types.ModuleType("wandb")
        class _FakeAPI:
            def projects(self, entity): return []
            def runs(self, *a, **kw): return runs
        fake_wandb.Api = lambda: _FakeAPI()
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        sweeps_dir = tmp_path / "sweeps"
        (sweeps_dir / "active").mkdir(parents=True)
        audit_out = tmp_path / "audit.json"

        rc = axis_select_main([
            "--group", "scout_group",
            "--axis", "data.train_test_params.delay_embedding_params.n_delays",
            "--top-k", "2",
            "--next-experiment", "next_grid",
            "--next-two-stage-json", '{"stage_a_epochs":20,"full_max_epochs":100}',
            "--next-migrate-to-json", '{"partition":"mit_normal_gpu"}',
            "--project", "WMTask_test",
            "--sweeps-dir", str(sweeps_dir),
            "--audit-out", str(audit_out),
            "--dry-run",
            "--log-level", "WARNING",
        ])
        assert rc == 0
        doc = json.loads(audit_out.read_text())
        assert doc["next_two_stage"] == {"stage_a_epochs": 20, "full_max_epochs": 100}
        assert doc["next_migrate_to"] == {"partition": "mit_normal_gpu"}

    def test_two_stage_chain_injects_stage_a_overrides(self, tmp_path, monkeypatch):
        """When --next-two-stage-json is set, the dispatcher MUST inject
        ``training.trainer_params.max_epochs=<stage_a_epochs>`` and a
        ``wandb_group=<exp>_<ts>__stage_a`` override into the next
        instruction. Without these, the next sweep trains to the YAML's
        max_epochs (typically the FULL budget) and
        engaging-controller's _maybe_dispatch_stage_b skips the cull
        because the wandb_group lacks the __stage_a suffix.

        Regression test for the chain-dispatch bug observed
        2026-05-05 where Stage A grids ran to walltime instead of
        stage_a_epochs.
        """
        runs = _make_runs()
        import sys, types
        fake_wandb = types.ModuleType("wandb")
        class _FakeAPI:
            def projects(self, entity): return []
            def runs(self, *a, **kw): return runs
        fake_wandb.Api = lambda: _FakeAPI()
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Capture _write_and_push_instruction's call args instead of
        # actually pushing to the jacobian-reports repo.
        captured = {}

        def _fake_write_and_push(**kwargs):
            captured.update(kwargs)

        from JacobianODE.jacobians.tuning import two_stage_cull
        monkeypatch.setattr(
            two_stage_cull, "_write_and_push_instruction", _fake_write_and_push
        )

        sweeps_dir = tmp_path / "sweeps"
        (sweeps_dir / "active").mkdir(parents=True)
        audit_out = tmp_path / "audit.json"

        rc = axis_select_main([
            "--group", "scout_group",
            "--axis", "data.train_test_params.delay_embedding_params.n_delays",
            "--top-k", "2",
            "--next-experiment", "my_grid_exp",
            "--next-two-stage-json",
            '{"stage_a_epochs":20,"full_max_epochs":200,"cull_fraction":0.5}',
            "--project", "WMTask_test",
            "--sweeps-dir", str(sweeps_dir),
            "--audit-out", str(audit_out),
            "--log-level", "WARNING",
        ])
        assert rc == 0, "axis_select_main should succeed"

        overrides = captured["overrides"]
        # The axis-select override is always present
        axis_overrides = [
            o for o in overrides
            if o.startswith("data.train_test_params.delay_embedding_params.n_delays=")
        ]
        assert len(axis_overrides) == 1, f"expected one axis override, got {overrides}"

        # The Stage A epoch cap MUST be injected
        max_epoch_overrides = [
            o for o in overrides
            if o == "training.trainer_params.max_epochs=20"
        ]
        assert len(max_epoch_overrides) == 1, (
            f"expected `training.trainer_params.max_epochs=20` override "
            f"to be injected when next_two_stage is set; overrides={overrides}"
        )

        # The wandb_group override MUST end in __stage_a (required by
        # _maybe_dispatch_stage_b in engaging-controller)
        wb_group_overrides = [o for o in overrides if o.startswith("wandb_group=")]
        assert len(wb_group_overrides) == 1, (
            f"expected one wandb_group override; overrides={overrides}"
        )
        wb_group_value = wb_group_overrides[0].split("=", 1)[1]
        assert wb_group_value.startswith("my_grid_exp_"), (
            f"wandb_group should start with the experiment name; got {wb_group_value}"
        )
        assert wb_group_value.endswith("__stage_a"), (
            f"wandb_group MUST end in __stage_a (required by trainer.py "
            f"and _maybe_dispatch_stage_b); got {wb_group_value}"
        )

        # two_stage block + experiment name preserved
        assert captured["experiment"] == "my_grid_exp"
        assert captured["two_stage"] == {
            "stage_a_epochs": 20,
            "full_max_epochs": 200,
            "cull_fraction": 0.5,
        }
        assert captured["kind"] == "chain"

    def test_chain_without_two_stage_does_not_inject_overrides(
        self, tmp_path, monkeypatch
    ):
        """When --next-two-stage-json is NOT set, the dispatcher must
        NOT inject Stage A overrides — the chain is just a single-stage
        scout → grid handoff and the next sweep should use the YAML's
        configured max_epochs / wandb_group as-is.
        """
        runs = _make_runs()
        import sys, types
        fake_wandb = types.ModuleType("wandb")
        class _FakeAPI:
            def projects(self, entity): return []
            def runs(self, *a, **kw): return runs
        fake_wandb.Api = lambda: _FakeAPI()
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        captured = {}
        def _fake_write_and_push(**kwargs):
            captured.update(kwargs)

        from JacobianODE.jacobians.tuning import two_stage_cull
        monkeypatch.setattr(
            two_stage_cull, "_write_and_push_instruction", _fake_write_and_push
        )

        sweeps_dir = tmp_path / "sweeps"
        (sweeps_dir / "active").mkdir(parents=True)
        audit_out = tmp_path / "audit.json"

        rc = axis_select_main([
            "--group", "scout_group",
            "--axis", "data.train_test_params.delay_embedding_params.n_delays",
            "--top-k", "2",
            "--next-experiment", "my_grid_exp",
            "--project", "WMTask_test",
            "--sweeps-dir", str(sweeps_dir),
            "--audit-out", str(audit_out),
            "--log-level", "WARNING",
        ])
        assert rc == 0
        overrides = captured["overrides"]
        assert not any(
            o.startswith("training.trainer_params.max_epochs=") for o in overrides
        ), f"max_epochs should NOT be overridden without next_two_stage; got {overrides}"
        assert not any(
            o.startswith("wandb_group=") for o in overrides
        ), f"wandb_group should NOT be overridden without next_two_stage; got {overrides}"

    def test_two_stage_missing_stage_a_epochs_aborts(self, tmp_path, monkeypatch):
        """If --next-two-stage-json is set but lacks `stage_a_epochs`, the
        dispatcher must abort (exit non-zero) — running with no Stage A
        cap would silently train to the YAML's max_epochs."""
        runs = _make_runs()
        import sys, types
        fake_wandb = types.ModuleType("wandb")
        class _FakeAPI:
            def projects(self, entity): return []
            def runs(self, *a, **kw): return runs
        fake_wandb.Api = lambda: _FakeAPI()
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        # Track whether _write_and_push_instruction got called (it must NOT)
        write_called = []
        from JacobianODE.jacobians.tuning import two_stage_cull
        monkeypatch.setattr(
            two_stage_cull, "_write_and_push_instruction",
            lambda **kw: write_called.append(kw),
        )

        sweeps_dir = tmp_path / "sweeps"
        (sweeps_dir / "active").mkdir(parents=True)

        rc = axis_select_main([
            "--group", "scout_group",
            "--axis", "data.train_test_params.delay_embedding_params.n_delays",
            "--top-k", "2",
            "--next-experiment", "my_grid_exp",
            "--next-two-stage-json", '{"full_max_epochs":200}',
            "--project", "WMTask_test",
            "--sweeps-dir", str(sweeps_dir),
            "--log-level", "WARNING",
        ])
        assert rc != 0, "expected non-zero exit when stage_a_epochs missing"
        assert write_called == [], (
            "_write_and_push_instruction must NOT be called when "
            "stage_a_epochs is missing"
        )


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_existing_sentinel_skips_dispatch(self, tmp_path, monkeypatch):
        """A pre-existing chain_dispatched sentinel makes the dispatcher
        a no-op (exit 0) without touching wandb."""
        sweeps_dir = tmp_path / "sweeps"
        (sweeps_dir / "active").mkdir(parents=True)
        sentinel = sweeps_dir / "active" / "scout_group.chain_dispatched"
        sentinel.write_text("{}")

        # If the dispatcher reaches wandb.Api(), this will fail loudly:
        import sys, types
        fake_wandb = types.ModuleType("wandb")
        def _bad_api(): raise RuntimeError("must not be called when sentinel exists")
        fake_wandb.Api = _bad_api
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        rc = axis_select_main([
            "--group", "scout_group",
            "--axis", "x.y.z",
            "--top-k", "1",
            "--next-experiment", "any",
            "--sweeps-dir", str(sweeps_dir),
            "--log-level", "WARNING",
        ])
        assert rc == 0
        assert sentinel.exists()  # untouched
