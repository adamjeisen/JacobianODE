"""Tests for the tuning module: criteria predicates and model selection."""

from __future__ import annotations

import math

import pytest
import torch

from JacobianODE.jacobians.tuning.criteria import (
    DiagnosticMetrics,
    compute_persistence_baseline,
    fails_eigenvalue_criterion,
    fails_loop_closure_criterion,
    fails_one_step_criterion,
)
from JacobianODE.jacobians.tuning.selection import SelectionResult, select_best_model


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------


class TestPersistenceBaseline:
    def test_constant_trajectory(self):
        """Constant trajectory -> baseline is 0."""
        trajs = torch.ones(2, 10, 3)
        assert compute_persistence_baseline(trajs) == pytest.approx(0.0)

    def test_known_value(self):
        """Known linear trajectory -> known baseline."""
        # shape (1, 4, 1): values [0, 1, 2, 3]
        trajs = torch.arange(4).float().reshape(1, 4, 1)
        # Differences are all 1, squared -> 1, mean -> 1
        assert compute_persistence_baseline(trajs) == pytest.approx(1.0)

    def test_multidim(self):
        """Multi-dimensional trajectory."""
        trajs = torch.zeros(1, 3, 2)
        trajs[0, 0, :] = 0
        trajs[0, 1, :] = 1
        trajs[0, 2, :] = 3
        # diffs: [1, 1], [2, 2]; squared: [1, 1], [4, 4]; mean = 10/4 = 2.5
        assert compute_persistence_baseline(trajs) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Criterion predicates
# ---------------------------------------------------------------------------


class TestFailsOneStepCriterion:
    def test_passes_when_below_baseline(self):
        m = DiagnosticMetrics(one_step_error=0.5, loop_closure_loss=0.1,
                              fast_eigenvalue_fraction=0.0, trajectory_val_loss=0.3)
        assert not fails_one_step_criterion(m, persistence_baseline=1.0)

    def test_fails_when_above_baseline(self):
        m = DiagnosticMetrics(one_step_error=1.5, loop_closure_loss=0.1,
                              fast_eigenvalue_fraction=0.0, trajectory_val_loss=0.3)
        assert fails_one_step_criterion(m, persistence_baseline=1.0)

    def test_fails_when_equal(self):
        """Equal to baseline still passes (not strictly greater)."""
        m = DiagnosticMetrics(one_step_error=1.0, loop_closure_loss=0.1,
                              fast_eigenvalue_fraction=0.0, trajectory_val_loss=0.3)
        assert not fails_one_step_criterion(m, persistence_baseline=1.0)


class TestFailsLoopClosureCriterion:
    def test_passes_when_below_sqrt_dim(self):
        m = DiagnosticMetrics(one_step_error=0.1, loop_closure_loss=1.0,
                              fast_eigenvalue_fraction=0.0, trajectory_val_loss=0.3)
        assert not fails_loop_closure_criterion(m, n_dims=4)  # sqrt(4) = 2

    def test_fails_when_above_sqrt_dim(self):
        m = DiagnosticMetrics(one_step_error=0.1, loop_closure_loss=3.0,
                              fast_eigenvalue_fraction=0.0, trajectory_val_loss=0.3)
        assert fails_loop_closure_criterion(m, n_dims=4)

    def test_none_loop_closure_always_passes(self):
        """NeuralODE mode: no loop closure -> always passes."""
        m = DiagnosticMetrics(one_step_error=0.1, loop_closure_loss=None,
                              fast_eigenvalue_fraction=0.0, trajectory_val_loss=0.3)
        assert not fails_loop_closure_criterion(m, n_dims=4)


class TestFailsEigenvalueCriterion:
    def test_passes_when_below_threshold(self):
        m = DiagnosticMetrics(one_step_error=0.1, loop_closure_loss=0.1,
                              fast_eigenvalue_fraction=0.0005, trajectory_val_loss=0.3)
        assert not fails_eigenvalue_criterion(m, eigenvalue_threshold=0.001)

    def test_fails_when_above_threshold(self):
        m = DiagnosticMetrics(one_step_error=0.1, loop_closure_loss=0.1,
                              fast_eigenvalue_fraction=0.01, trajectory_val_loss=0.3)
        assert fails_eigenvalue_criterion(m, eigenvalue_threshold=0.001)

    def test_equal_to_threshold_passes(self):
        m = DiagnosticMetrics(one_step_error=0.1, loop_closure_loss=0.1,
                              fast_eigenvalue_fraction=0.001, trajectory_val_loss=0.3)
        assert not fails_eigenvalue_criterion(m, eigenvalue_threshold=0.001)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


class TestSelectBestModel:
    """Tests for select_best_model with synthetic DiagnosticMetrics."""

    @staticmethod
    def _make(one_step=0.5, loop=0.5, eig=0.0, traj=1.0):
        return DiagnosticMetrics(
            one_step_error=one_step,
            loop_closure_loss=loop,
            fast_eigenvalue_fraction=eig,
            trajectory_val_loss=traj,
        )

    def test_basic_selection_lowest_traj_loss_wins(self):
        """All pass criteria -> lowest trajectory_val_loss wins."""
        candidates = [
            self._make(traj=2.0),
            self._make(traj=1.0),
            self._make(traj=3.0),
        ]
        result = select_best_model(candidates, persistence_baseline=10.0, n_dims=4)
        assert result.best_index == 1
        assert result.best_metrics.trajectory_val_loss == 1.0
        assert len(result.surviving_indices) == 3

    def test_exclusion_by_c1(self):
        """Model excluded by C1 (one-step error too high)."""
        candidates = [
            self._make(one_step=0.5, traj=1.0),  # passes C1
            self._make(one_step=2.0, traj=0.5),  # fails C1 (but lower traj)
        ]
        result = select_best_model(candidates, persistence_baseline=1.0, n_dims=100)
        assert result.best_index == 0
        assert 1 in result.exclusion_details["C1"]

    def test_exclusion_by_c2(self):
        """Model excluded by C2 (loop closure too high)."""
        candidates = [
            self._make(loop=1.0, traj=2.0),   # passes C2 (sqrt(4)=2)
            self._make(loop=5.0, traj=0.5),   # fails C2
        ]
        result = select_best_model(candidates, persistence_baseline=10.0, n_dims=4)
        assert result.best_index == 0
        assert 1 in result.exclusion_details["C2"]

    def test_exclusion_by_c3(self):
        """Model excluded by C3 (too many fast eigenvalues)."""
        candidates = [
            self._make(eig=0.0005, traj=2.0),  # passes C3
            self._make(eig=0.01, traj=0.5),    # fails C3
        ]
        result = select_best_model(candidates, persistence_baseline=10.0, n_dims=100)
        assert result.best_index == 0
        assert 1 in result.exclusion_details["C3"]

    def test_relaxation_c2_discarded(self):
        """Relaxation: no C2-passer also passes C1 -> discard C2.

        Model 0: fails C1, passes C2  -> excluded by C1
        Model 1: passes C1, fails C2  -> would be excluded by C2
        Without relaxation, no survivor passes both C1 and C2.
        After relaxation: C2 discarded, model 1 survives.
        """
        candidates = [
            self._make(one_step=5.0, loop=0.5, traj=0.5),   # fails C1
            self._make(one_step=0.5, loop=10.0, traj=1.0),  # fails C2
        ]
        result = select_best_model(candidates, persistence_baseline=1.0, n_dims=4)
        assert result.best_index == 1
        assert "C2" not in result.criteria_applied
        assert "C1" in result.criteria_applied

    def test_relaxation_c1_discarded(self):
        """Relaxation: no C3-passer also passes C1 -> discard C1.

        All models fail C1, but some pass C3. Since no C3-passer passes C1,
        C1 is relaxed.
        """
        candidates = [
            self._make(one_step=5.0, eig=0.0, traj=2.0),   # fails C1, passes C3
            self._make(one_step=5.0, eig=0.0, traj=1.0),   # fails C1, passes C3
            self._make(one_step=5.0, eig=0.05, traj=0.5),  # fails C1, fails C3
        ]
        result = select_best_model(candidates, persistence_baseline=1.0, n_dims=100)
        assert "C1" not in result.criteria_applied
        # Model 2 fails C3, so best among {0, 1} is 1 (lower traj)
        assert result.best_index == 1

    def test_use_loop_closure_false(self):
        """NeuralODE mode: C2 not applied."""
        candidates = [
            self._make(loop=100.0, traj=2.0),  # would fail C2 if applied
            self._make(loop=0.1, traj=3.0),
        ]
        result = select_best_model(
            candidates, persistence_baseline=10.0, n_dims=4, use_loop_closure=False,
        )
        assert result.best_index == 0  # lowest traj, C2 not applied
        assert "C2" not in result.criteria_applied

    def test_empty_candidates(self):
        """Empty candidates list."""
        result = select_best_model([], persistence_baseline=1.0, n_dims=3)
        assert result.best_index is None
        assert result.best_metrics is None
        assert result.surviving_indices == []

    def test_single_candidate(self):
        """Single candidate always selected."""
        candidates = [self._make(traj=5.0)]
        result = select_best_model(candidates, persistence_baseline=10.0, n_dims=4)
        assert result.best_index == 0

    def test_all_fail_all_criteria(self):
        """When all models fail everything, fallback to all candidates."""
        candidates = [
            self._make(one_step=10.0, loop=100.0, eig=0.5, traj=3.0),
            self._make(one_step=10.0, loop=100.0, eig=0.5, traj=1.0),
        ]
        result = select_best_model(candidates, persistence_baseline=1.0, n_dims=1)
        # After relaxation of C1 (no C3-passer passes C1), and C2 (no C2-passer
        # passes C1), all that remains is C3 which all fail -> fallback
        assert result.best_index == 1  # lowest traj
        assert len(result.surviving_indices) == 2

    def test_criteria_applied_tracks_correctly(self):
        """Verify criteria_applied reflects actual enforcement."""
        candidates = [
            self._make(one_step=0.1, loop=0.1, eig=0.0, traj=1.0),
        ]
        result = select_best_model(candidates, persistence_baseline=10.0, n_dims=4)
        assert "C1" in result.criteria_applied
        assert "C2" in result.criteria_applied
        assert "C3" in result.criteria_applied
