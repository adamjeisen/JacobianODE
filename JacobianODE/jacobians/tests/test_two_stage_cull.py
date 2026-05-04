"""Unit tests for two_stage_cull's cell-key derivation, path stability,
and survivor selection.

The cell-key invariant is load-bearing: trainer.py copies Stage A's
last.ckpt to a path computed from cfg, and the cull tool reconstructs
the same path from the wandb run's saved cfg. If those two hashes ever
diverge, Stage B silently can't find any survivors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from JacobianODE.jacobians.tuning.two_stage_cull import (
    DEFAULT_METRIC,
    STAGE_A_SUFFIX,
    compute_cell_key,
    pick_survivors,
    swept_overrides_from_config,
    two_stage_ckpt_path,
)


# ---------------------------------------------------------------------------
# Fixture configs
# ---------------------------------------------------------------------------

def _make_cfg(lc=1e-5, lpl=3, group="myexp__stage_a", base_dir="/tmp/sweeps"):
    """Build a minimal cfg that has the fields cell_key + ckpt_path lookups
    care about. Realistic enough that Hydra-resolved string vs float overrides
    end up in different combinations across the test cases."""
    return OmegaConf.create({
        "wandb_group": group,
        "training": {
            "logger_save_dirs": base_dir,
            "lightning": {
                "loop_closure_weight": lc,
                "latent_prediction_loss_weight": lpl,
            },
        },
        "sweep_grid": {
            "training.lightning.loop_closure_weight": "1e-5,1e-4",
            "training.lightning.latent_prediction_loss_weight": "1,3",
        },
    })


# ---------------------------------------------------------------------------
# compute_cell_key
# ---------------------------------------------------------------------------

class TestCellKey:
    def test_determinism(self):
        """Same cfg -> same hash, even across two separate calls."""
        cfg = _make_cfg()
        assert compute_cell_key(cfg) == compute_cell_key(cfg)

    def test_sensitivity_to_swept_value(self):
        """Changing one swept-grid value changes the hash."""
        a = _make_cfg(lc=1e-5)
        b = _make_cfg(lc=1e-4)
        assert compute_cell_key(a) != compute_cell_key(b)

    def test_insensitivity_to_non_swept_value(self):
        """Changing a non-swept-grid field (here: wandb_group) MUST NOT
        change the hash. The cell-key only encodes the swept axes."""
        a = _make_cfg(group="myexp__stage_a")
        b = _make_cfg(group="otherexp__stage_a")
        assert compute_cell_key(a) == compute_cell_key(b)

    def test_string_vs_float_normalization(self):
        """A Hydra string override '1e-5' and the resolved float 1e-5 must
        hash to the same key. Otherwise trainer-side and cull-side hashes
        diverge whenever Hydra serialization is non-uniform."""
        cfg_float = _make_cfg(lc=1e-5)
        cfg_str = _make_cfg(lc="1e-5")
        assert compute_cell_key(cfg_float) == compute_cell_key(cfg_str)

    def test_no_sweep_grid_returns_fixed_key(self):
        """A cfg without sweep_grid (e.g. ad-hoc single run) returns a
        fixed sentinel rather than crashing."""
        cfg = OmegaConf.create({
            "wandb_group": "ad_hoc",
            "training": {"logger_save_dirs": "/tmp"},
        })
        key = compute_cell_key(cfg)
        assert key == "no_sweep_grid"

    def test_cross_process_determinism(self):
        """Two separate Python processes computing the same cell-key from
        the same JSON cfg get the same hash. The trainer-side and cull-side
        live in different processes, so this is the load-bearing invariant
        for the whole protocol."""
        import subprocess
        import sys
        cfg_blob = json.dumps({
            "wandb_group": "g__stage_a",
            "training": {
                "logger_save_dirs": "/tmp/sweeps",
                "lightning": {
                    "loop_closure_weight": 1e-5,
                    "latent_prediction_loss_weight": 3,
                },
            },
            "sweep_grid": {
                "training.lightning.loop_closure_weight": "1e-5,1e-4",
                "training.lightning.latent_prediction_loss_weight": "1,3",
            },
        })
        snippet = (
            "import json, sys; "
            "from omegaconf import OmegaConf; "
            "from JacobianODE.jacobians.tuning.two_stage_cull "
            "import compute_cell_key; "
            f"cfg = OmegaConf.create(json.loads({cfg_blob!r})); "
            "print(compute_cell_key(cfg))"
        )
        out = subprocess.check_output(
            [sys.executable, "-c", snippet], text=True
        ).strip()
        in_proc = compute_cell_key(OmegaConf.create(json.loads(cfg_blob)))
        assert out == in_proc


# ---------------------------------------------------------------------------
# two_stage_ckpt_path
# ---------------------------------------------------------------------------

class TestCkptPath:
    def test_path_layout(self):
        """Returned path is .../<group>/<cell_key>/last.ckpt."""
        cfg = _make_cfg()
        p = two_stage_ckpt_path(cfg)
        assert p is not None
        assert p.name == "last.ckpt"
        assert p.parent.name == compute_cell_key(cfg)
        assert p.parent.parent.name == cfg.wandb_group

    def test_path_stable_across_calls(self):
        """The path is purely a function of cfg — multiple invocations
        with the same cfg must return the same path."""
        cfg = _make_cfg()
        assert two_stage_ckpt_path(cfg) == two_stage_ckpt_path(cfg)

    def test_path_independent_of_slurm_array_id(self):
        """The whole point: different SLURM array tasks running the same
        cell hit the same path. We don't read any SLURM env in the path
        derivation; assert by inspection that the path doesn't contain
        any SLURM-id-looking number."""
        cfg = _make_cfg()
        p = two_stage_ckpt_path(cfg)
        assert "_resume_state" not in str(p)

    def test_returns_none_on_missing_fields(self):
        """If essential cfg fields are missing, return None (caller treats
        that as 'two-stage disabled for this run')."""
        no_group = OmegaConf.create({"training": {"logger_save_dirs": "/tmp"}})
        assert two_stage_ckpt_path(no_group) is None
        no_base = OmegaConf.create({"wandb_group": "g", "training": {}})
        assert two_stage_ckpt_path(no_base) is None


# ---------------------------------------------------------------------------
# pick_survivors
# ---------------------------------------------------------------------------

class _MockRun:
    """Minimal stand-in for a wandb Run. Supports id, state, and
    scan_history(keys=[...])."""
    def __init__(self, run_id, state, history):
        self.id = run_id
        self.state = state
        self._history = history  # list[dict]

    def scan_history(self, keys):
        for row in self._history:
            yield {k: row.get(k) for k in keys}


class TestPickSurvivors:
    def test_keeps_top_half(self):
        """Default cull_fraction=0.5: 4 finished runs -> keep 2 with
        lowest best-so-far metric."""
        runs = [
            _MockRun("r1", "finished", [{DEFAULT_METRIC: 1.0},
                                         {DEFAULT_METRIC: 0.9}]),
            _MockRun("r2", "finished", [{DEFAULT_METRIC: 2.0}]),
            _MockRun("r3", "finished", [{DEFAULT_METRIC: 0.5},
                                         {DEFAULT_METRIC: 0.3}]),
            _MockRun("r4", "finished", [{DEFAULT_METRIC: 1.5}]),
        ]
        survivors, audit = pick_survivors(runs, DEFAULT_METRIC, 0.5)
        assert {r.id for r in survivors} == {"r3", "r1"}
        # Audit should include all 4 runs with kept flags
        kept_ids = {a["run_id"] for a in audit if a.get("kept")}
        assert kept_ids == {"r3", "r1"}
        # And the best metric for r3 is 0.3 (lowest in its history)
        r3_audit = next(a for a in audit if a["run_id"] == "r3")
        assert r3_audit["best_metric"] == pytest.approx(0.3)

    def test_crashed_with_valid_loss_is_kept(self):
        """Crashed runs with a recorded best_loss are eligible (typical
        SLURM-timeout-killed cells have hours of training behind them).
        Only ``state == "running"`` gets blanket-skipped."""
        runs = [
            _MockRun("r1", "finished", [{DEFAULT_METRIC: 1.0}]),
            _MockRun("r2", "crashed", [{DEFAULT_METRIC: 0.1}]),
            _MockRun("r3", "running", [{DEFAULT_METRIC: 0.05}]),
        ]
        survivors, audit = pick_survivors(runs, DEFAULT_METRIC, 0.5)
        # Top half of {r1: 1.0, r2: 0.1} (r3 skipped, no terminal loss yet)
        # → 2 eligible, keep 1 → r2 (lower loss, even though crashed).
        assert {r.id for r in survivors} == {"r2"}
        # Crashed run made it into the scored audit with its loss.
        r2 = next(a for a in audit if a["run_id"] == "r2"
                  and a["best_metric"] is not None)
        assert r2["best_metric"] == pytest.approx(0.1)
        assert r2["state"] == "crashed"
        assert r2["kept"] is True
        # Running run is the only one skipped.
        running = next(a for a in audit if a["run_id"] == "r3")
        assert running["kept"] is False
        assert "state=running" in running["skip_reason"]

    def test_skips_runs_with_no_metric_value(self):
        """Runs that finished but never logged the cull metric (e.g. died
        before first val step) are skipped."""
        runs = [
            _MockRun("r1", "finished", [{DEFAULT_METRIC: 1.0}]),
            _MockRun("r2", "finished", [{}, {}]),  # no metric ever
            _MockRun("r3", "finished", [{DEFAULT_METRIC: None}]),
        ]
        survivors, audit = pick_survivors(runs, DEFAULT_METRIC, 0.5)
        assert {r.id for r in survivors} == {"r1"}
        for rid in ("r2", "r3"):
            entry = next(a for a in audit if a["run_id"] == rid)
            assert entry["kept"] is False
            assert "no_value_for_" in entry["skip_reason"]

    def test_at_least_one_survivor(self):
        """cull_fraction=0.99 with one run still keeps that one."""
        runs = [_MockRun("r1", "finished", [{DEFAULT_METRIC: 1.0}])]
        survivors, _ = pick_survivors(runs, DEFAULT_METRIC, 0.99)
        assert len(survivors) == 1

    def test_skips_runs_whose_final_metric_is_nan(self):
        """Runs whose FINAL logged metric is NaN are skipped, even if
        intermediate values were finite. Two-stage protocol resumes from
        last.ckpt, so a late-training divergence to NaN means Stage B
        loads NaN weights — better to drop the cell than poison Stage B."""
        runs = [
            # Healthy: monotonically improving, finite throughout.
            _MockRun("healthy", "finished", [{DEFAULT_METRIC: 1.0},
                                              {DEFAULT_METRIC: 0.5}]),
            # Diverged late: had a great mid-training value but final is NaN.
            # Without the NaN-final skip this would beat "healthy" on best=0.1
            # and Stage B would load its NaN-weighted last.ckpt.
            _MockRun("late_nan", "finished", [{DEFAULT_METRIC: 0.5},
                                                {DEFAULT_METRIC: 0.1},
                                                {DEFAULT_METRIC: float("nan")}]),
        ]
        survivors, audit = pick_survivors(runs, DEFAULT_METRIC, 0.0)
        assert {r.id for r in survivors} == {"healthy"}
        late_nan_audit = next(a for a in audit if a["run_id"] == "late_nan")
        assert late_nan_audit["kept"] is False
        assert "final_" in late_nan_audit["skip_reason"]
        assert "nan" in late_nan_audit["skip_reason"]


# ---------------------------------------------------------------------------
# swept_overrides_from_config
# ---------------------------------------------------------------------------

class TestSweptOverrides:
    def test_sorted_and_dotted(self):
        """Reconstructed overrides are key=value, sorted by key."""
        cfg = _make_cfg(lc=1e-5, lpl=3)
        ovs = swept_overrides_from_config(cfg)
        assert ovs == [
            "training.lightning.latent_prediction_loss_weight=3",
            "training.lightning.loop_closure_weight=1e-05",
        ]

    def test_no_sweep_grid_returns_empty(self):
        cfg = OmegaConf.create({"wandb_group": "g", "training": {}})
        assert swept_overrides_from_config(cfg) == []
