"""Focused unit tests for `_terminal_block_summary` SEM plumbing.

The full gramian pipeline is hard to test without instantiating a
LitModel + DirectSum encoder + per-condition source eqs. This file
narrowly tests the (mean, sem) reduction logic by mocking
`compute_all_gramians` to return synthetic per-batch spectra with
known eigenvalues, then asserting that `_terminal_block_summary`
returns the right (mean, sem) tuple per (kind, stat).
"""
from __future__ import annotations

import math
import sys
import types
from typing import Any

import pytest
import torch

from JacobianODE.jacobians.run_analytics import _terminal_block_summary


def _make_mock_gram_module(spec_r: torch.Tensor, spec_c: torch.Tensor, spec_o: torch.Tensor):
    """Build a fake gram module whose compute_all_gramians returns the
    given (r, c, o) spectra regardless of A, B, C, kwargs."""
    mod = types.SimpleNamespace()

    def compute_all_gramians(A, B, C, **kwargs):
        # Match the real signature: returns ((G_r, G_c, G_o), (spec_r, spec_c, spec_o))
        return (None, None, None), (spec_r, spec_c, spec_o)

    mod.compute_all_gramians = compute_all_gramians
    return mod


class TestTerminalBlockSummary:
    def test_returns_tuples_with_mean_and_sem(self):
        """Every returned value must be a (mean, sem) tuple of floats."""
        # 4 trajectories (batch dim), 3 eigenvalues per spectrum.
        spec_r = torch.zeros(4, 3)
        spec_c = torch.zeros(4, 3)
        spec_o = torch.zeros(4, 3)
        gm = _make_mock_gram_module(spec_r, spec_c, spec_o)

        out = _terminal_block_summary(
            torch.zeros(4, 5, 3, 3),  # A
            torch.zeros(4, 5, 3, 3),  # B
            torch.zeros(4, 5, 3, 3),  # C
            dt=1.0,
            gramian_module=gm,
        )

        expected_keys = {
            ("reach", "log_trace"), ("reach", "log_min"),
            ("ctrl",  "log_trace"), ("ctrl",  "log_min"),
            ("obs",   "log_trace"), ("obs",   "log_min"),
        }
        assert set(out.keys()) == expected_keys
        for k, v in out.items():
            assert isinstance(v, tuple), f"{k} is not a tuple: {v}"
            assert len(v) == 2, f"{k} tuple has len {len(v)}, expected 2"
            assert all(isinstance(x, float) for x in v), f"{k} contains non-float: {v}"

    def test_mean_log_trace_correct(self):
        """log_trace = logsumexp(log_eigs) per traj. With known eigs we can
        check exactly."""
        # 4 trajectories, each with log-eigs [0, 0, 0] → logsumexp = log(3).
        # So per-traj log_trace = log(3) for all 4. Mean = log(3), SEM = 0.
        spec = torch.zeros(4, 3)  # log-eigs of zero (= eigs of 1)
        gm = _make_mock_gram_module(spec, spec, spec)
        out = _terminal_block_summary(
            torch.zeros(4, 5, 3, 3), torch.zeros(4, 5, 3, 3), torch.zeros(4, 5, 3, 3),
            dt=1.0, gramian_module=gm,
        )
        mean, sem = out[("reach", "log_trace")]
        assert math.isclose(mean, math.log(3), abs_tol=1e-6)
        assert math.isclose(sem, 0.0, abs_tol=1e-6)

    def test_sem_nonzero_when_per_traj_varies(self):
        """When per-traj log-trace varies, SEM should be > 0 and equal
        std(log_trace, ddof=1) / sqrt(N)."""
        # 4 trajectories with distinct per-traj log_trace values:
        # logsumexp([a, a, a]) = log(3) + a. So if we pick a-values that
        # vary across batch, log_trace varies the same way.
        a_vals = torch.tensor([0.0, 1.0, 2.0, 3.0])
        spec = a_vals.unsqueeze(-1).expand(4, 3).contiguous()  # (4, 3)
        gm = _make_mock_gram_module(spec, spec, spec)
        out = _terminal_block_summary(
            torch.zeros(4, 5, 3, 3), torch.zeros(4, 5, 3, 3), torch.zeros(4, 5, 3, 3),
            dt=1.0, gramian_module=gm,
        )
        mean, sem = out[("reach", "log_trace")]
        # Per-traj log_trace = log(3) + a_vals
        per_traj = math.log(3) + a_vals
        expected_mean = float(per_traj.mean())
        expected_sem = float(per_traj.std(unbiased=True) / math.sqrt(4))
        assert math.isclose(mean, expected_mean, abs_tol=1e-6)
        assert math.isclose(sem, expected_sem, abs_tol=1e-6)
        assert sem > 0

    def test_log_min_takes_last_eigenvalue(self):
        """log_min uses spec[..., -1] (preserving existing convention).
        Per-traj log_min = last column of spec; mean across batch."""
        # Distinct per-traj last eigenvalues
        spec = torch.tensor([
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 7.0],
            [0.0, 0.0, 9.0],
            [0.0, 0.0, 11.0],
        ])
        gm = _make_mock_gram_module(spec, spec, spec)
        out = _terminal_block_summary(
            torch.zeros(4, 5, 3, 3), torch.zeros(4, 5, 3, 3), torch.zeros(4, 5, 3, 3),
            dt=1.0, gramian_module=gm,
        )
        mean, sem = out[("reach", "log_min")]
        per_traj = torch.tensor([5.0, 7.0, 9.0, 11.0])
        assert math.isclose(mean, float(per_traj.mean()), abs_tol=1e-6)
        assert math.isclose(sem, float(per_traj.std(unbiased=True) / math.sqrt(4)), abs_tol=1e-6)

    def test_n_one_does_not_nan(self):
        """Single-element batch: SEM should be 0.0, not NaN."""
        spec = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        gm = _make_mock_gram_module(spec, spec, spec)
        out = _terminal_block_summary(
            torch.zeros(1, 5, 3, 3), torch.zeros(1, 5, 3, 3), torch.zeros(1, 5, 3, 3),
            dt=1.0, gramian_module=gm,
        )
        for k, (mean, sem) in out.items():
            assert math.isfinite(mean), f"{k}: mean is {mean}"
            assert sem == 0.0, f"{k}: sem={sem}, expected 0.0 for N=1"
