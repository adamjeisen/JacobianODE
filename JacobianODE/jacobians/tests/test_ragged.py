"""Tests for ``ragged.pad_trajs_to_max`` and ``ragged.sliding_windows``."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from JacobianODE.jacobians.data.ragged import (
    delay_embed_ragged,
    pad_trajs_to_max,
    sliding_windows,
    split_balanced_by_timepoints,
    truncate_chronological_balanced,
)
from JacobianODE.jacobians.data.splitting import embed_signal_torch


# ---------------------------------------------------------------------------
# pad_trajs_to_max
# ---------------------------------------------------------------------------

class TestPadTrajsToMax:
    def test_mixed_lengths(self):
        """Mixed lengths → pads to max, fills tail with NaN, lengths
        recorded correctly."""
        a = torch.arange(6, dtype=torch.float32).reshape(3, 2)  # (3, 2)
        b = torch.arange(10, dtype=torch.float32).reshape(5, 2)  # (5, 2)
        c = torch.arange(2, dtype=torch.float32).reshape(1, 2)  # (1, 2)

        padded, lengths = pad_trajs_to_max([a, b, c])
        assert padded.shape == (3, 5, 2)
        assert lengths.tolist() == [3, 5, 1]
        # Valid regions match the inputs
        assert torch.equal(padded[0, :3], a)
        assert torch.equal(padded[1, :5], b)
        assert torch.equal(padded[2, :1], c)
        # Padded regions are NaN
        assert torch.isnan(padded[0, 3:]).all()
        assert torch.isnan(padded[2, 1:]).all()
        # No NaN in valid regions
        assert not torch.isnan(padded[0, :3]).any()
        assert not torch.isnan(padded[1, :5]).any()
        assert not torch.isnan(padded[2, :1]).any()

    def test_uniform_lengths_no_padding(self):
        """All trajectories the same length → no padding needed, no NaN."""
        a = torch.ones(4, 3)
        b = torch.zeros(4, 3)
        padded, lengths = pad_trajs_to_max([a, b])
        assert padded.shape == (2, 4, 3)
        assert lengths.tolist() == [4, 4]
        assert not torch.isnan(padded).any()

    def test_single_trajectory(self):
        a = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        padded, lengths = pad_trajs_to_max([a])
        assert padded.shape == (1, 4, 2)
        assert lengths.tolist() == [4]
        assert torch.equal(padded[0], a)

    def test_numpy_input_accepted(self):
        a = np.arange(6, dtype=np.float32).reshape(3, 2)
        b = np.arange(10, dtype=np.float32).reshape(5, 2)
        padded, lengths = pad_trajs_to_max([a, b])
        assert padded.shape == (2, 5, 2)
        assert lengths.tolist() == [3, 5]
        assert torch.equal(padded[0, :3], torch.as_tensor(a))

    def test_custom_fill_value(self):
        """Non-default fill_value (e.g. 0) gets used for the padding region."""
        a = torch.ones(2, 3)
        b = torch.ones(4, 3)
        padded, _ = pad_trajs_to_max([a, b], fill_value=0.0)
        assert (padded[0, 2:] == 0).all()
        assert not torch.isnan(padded).any()

    def test_float64_input_promotes(self):
        a = torch.ones(2, 3, dtype=torch.float64)
        b = torch.ones(3, 3, dtype=torch.float64)
        padded, _ = pad_trajs_to_max([a, b])
        assert padded.dtype == torch.float64

    def test_int_input_upcast_to_float32(self):
        """Integer inputs get upcast to float32 because NaN needs a float
        dtype."""
        a = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        b = torch.arange(9, dtype=torch.int64).reshape(3, 3)
        padded, _ = pad_trajs_to_max([a, b])
        assert padded.dtype == torch.float32
        assert torch.isnan(padded[0, 2:]).all()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pad_trajs_to_max([])

    def test_wrong_ndim_raises(self):
        a = torch.zeros(3, 2, 4)  # 3-D, not 2-D
        with pytest.raises(ValueError, match="2-D"):
            pad_trajs_to_max([a])

    def test_mismatched_D_raises(self):
        a = torch.zeros(3, 2)
        b = torch.zeros(4, 5)
        with pytest.raises(ValueError, match="trailing dim"):
            pad_trajs_to_max([a, b])


# ---------------------------------------------------------------------------
# sliding_windows
# ---------------------------------------------------------------------------

class TestSlidingWindows:
    def _build(self, lengths_list: list[int], D: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper: build a padded tensor + lengths from a list of T_i values.
        Each trajectory is filled with its own row index so we can detect
        cross-contamination between trajectories.
        """
        trajs = []
        for i, T in enumerate(lengths_list):
            # Use distinct values per (i, t) so windows are uniquely identifiable
            t = torch.arange(T * D, dtype=torch.float32).reshape(T, D) + 1000 * i
            trajs.append(t)
        return pad_trajs_to_max(trajs)

    def test_non_overlapping_windows(self):
        """stride == seq_length → tile (no overlap, no skip)."""
        padded, lengths = self._build([6, 4, 2])  # max_T=6
        windows = sliding_windows(padded, lengths, seq_length=2, stride=2)
        # T=6 → 3 windows; T=4 → 2 windows; T=2 → 1 window. Total = 6.
        assert windows.shape == (6, 2, 2)
        # No NaN in any window
        assert not torch.isnan(windows).any()

    def test_overlapping_windows_count(self):
        """stride < seq_length → overlapping. Count formula:
        floor((T - seq_length) / stride) + 1."""
        padded, lengths = self._build([10])
        # T=10, seq=4, stride=2 → (10-4)/2 + 1 = 4 windows
        windows = sliding_windows(padded, lengths, seq_length=4, stride=2)
        assert windows.shape == (4, 4, 2)

    def test_window_content_matches_source(self):
        """Sub-window k of trajectory i exactly equals padded[i, k*stride:k*stride+seq_length]."""
        padded, lengths = self._build([8, 5], D=3)
        seq, stride = 3, 2
        windows = sliding_windows(padded, lengths, seq_length=seq, stride=stride)
        # Build the expected windows manually
        expected = []
        for i, T_i in enumerate(lengths.tolist()):
            n = (T_i - seq) // stride + 1 if T_i >= seq else 0
            for k in range(n):
                expected.append(padded[i, k * stride : k * stride + seq])
        expected_t = torch.stack(expected, dim=0)
        assert windows.shape == expected_t.shape
        assert torch.equal(windows, expected_t)

    def test_short_trajectory_skipped(self):
        """Trajectories shorter than seq_length contribute no windows."""
        padded, lengths = self._build([3, 10, 2])
        windows = sliding_windows(padded, lengths, seq_length=5, stride=5)
        # Only T=10 contributes: (10-5)/5 + 1 = 2 windows. T=3 and T=2 skipped.
        assert windows.shape == (2, 5, 2)

    def test_no_window_contains_nan(self):
        """Windows must never overlap the padded NaN region."""
        padded, lengths = self._build([4, 7, 9])  # max_T = 9, lots of NaN tails
        for seq, stride in [(2, 1), (3, 3), (5, 2), (7, 4)]:
            windows = sliding_windows(padded, lengths, seq_length=seq, stride=stride)
            assert not torch.isnan(windows).any(), f"NaN in windows for seq={seq} stride={stride}"

    def test_all_too_short_returns_empty(self):
        """All trajectories shorter than seq_length → empty (K=0) tensor of correct shape."""
        padded, lengths = self._build([2, 3])
        windows = sliding_windows(padded, lengths, seq_length=5, stride=5)
        assert windows.shape == (0, 5, 2)
        assert windows.dtype == padded.dtype

    def test_stride_one_dense(self):
        """stride=1 → all possible starting positions."""
        padded, lengths = self._build([5])
        windows = sliding_windows(padded, lengths, seq_length=3, stride=1)
        # T=5, seq=3, stride=1 → 5-3+1 = 3 windows
        assert windows.shape == (3, 3, 2)

    def test_seq_equals_T_one_window(self):
        """seq_length == T → exactly one window."""
        padded, lengths = self._build([5])
        windows = sliding_windows(padded, lengths, seq_length=5, stride=5)
        assert windows.shape == (1, 5, 2)

    def test_invalid_seq_length_raises(self):
        padded, lengths = self._build([5])
        with pytest.raises(ValueError, match="seq_length"):
            sliding_windows(padded, lengths, seq_length=0, stride=1)

    def test_invalid_stride_raises(self):
        padded, lengths = self._build([5])
        with pytest.raises(ValueError, match="stride"):
            sliding_windows(padded, lengths, seq_length=2, stride=0)

    def test_lengths_shape_mismatch_raises(self):
        padded, _ = self._build([5, 5])
        with pytest.raises(ValueError, match="lengths"):
            sliding_windows(padded, torch.tensor([5]), seq_length=2, stride=2)

    def test_padded_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            sliding_windows(torch.zeros(3, 4), torch.tensor([4, 4, 4]),
                            seq_length=2, stride=2)


# ---------------------------------------------------------------------------
# truncate_chronological_balanced
# ---------------------------------------------------------------------------

def _make_lengths(lengths: list[int], D: int = 2) -> list[torch.Tensor]:
    """Helper: produce dummy tensors with the given first-axis lengths.
    Each trajectory is filled with distinct values so we can detect
    accidental cross-trajectory contamination in tests."""
    return [
        torch.arange(T * D, dtype=torch.float32).reshape(T, D) + 1000 * i
        for i, T in enumerate(lengths)
    ]


class TestTruncateChronologicalBalanced:
    def test_no_truncation_total_equal_target(self):
        """total == T_target → return all unchanged."""
        trajs = _make_lengths([4, 5, 3])  # total = 12
        out = truncate_chronological_balanced(trajs, T_target=12, min_length=3)
        assert [t.shape[0] for t in out] == [4, 5, 3]
        for a, b in zip(out, trajs):
            assert torch.equal(a, b)

    def test_total_under_target_raises(self):
        trajs = _make_lengths([3, 4])
        with pytest.raises(ValueError, match="total samples"):
            truncate_chronological_balanced(trajs, T_target=20, min_length=3)

    def test_T_target_zero_returns_empty(self):
        trajs = _make_lengths([3, 4])
        assert truncate_chronological_balanced(trajs, T_target=0, min_length=3) == []

    def test_negative_T_target_raises(self):
        trajs = _make_lengths([3, 4])
        with pytest.raises(ValueError, match="T_target"):
            truncate_chronological_balanced(trajs, T_target=-1, min_length=3)

    def test_min_length_zero_raises(self):
        trajs = _make_lengths([3, 4])
        with pytest.raises(ValueError, match="min_length"):
            truncate_chronological_balanced(trajs, T_target=3, min_length=0)

    def test_exact_cumsum_match_no_partial(self):
        """Cumsum exactly hits T_target → no partial taken."""
        trajs = _make_lengths([4, 5, 3, 6])  # cumsums: 4, 9, 12, 18
        out = truncate_chronological_balanced(trajs, T_target=9, min_length=3)
        assert [t.shape[0] for t in out] == [4, 5]
        for a, b in zip(out, trajs[:2]):
            assert torch.equal(a, b)

    def test_clean_partial_case_a(self):
        """delta >= min_length → take first delta samples of t_{j+1}."""
        # T_target=12, lengths [4, 5, 6]: cumsum 4, 9. delta = 12 - 9 = 3 = min_length.
        trajs = _make_lengths([4, 5, 6])
        out = truncate_chronological_balanced(trajs, T_target=12, min_length=3)
        assert [t.shape[0] for t in out] == [4, 5, 3]
        # The partial must be the first `delta` samples of t_2
        assert torch.equal(out[2], trajs[2][:3])
        assert sum(t.shape[0] for t in out) == 12

    def test_stub_case_b_even_split(self):
        """Worked example from the design discussion: lengths
        [5, 4, 6, 4], T_target=16, MIN_LEN=3.
        Cumsum: 5, 9, 15. delta = 16 - 15 = 1 < min_length.
        Budget = L_j + delta = 6 + 1 = 7. Even split (with earlier+1
        on odd): L_j_kept = 4, L_curr_kept = 3. Result: [5, 4, 4, 3].
        """
        trajs = _make_lengths([5, 4, 6, 4])
        out = truncate_chronological_balanced(trajs, T_target=16, min_length=3)
        assert [t.shape[0] for t in out] == [5, 4, 4, 3]
        assert sum(t.shape[0] for t in out) == 16
        # First two unchanged; last two are partials
        assert torch.equal(out[0], trajs[0])
        assert torch.equal(out[1], trajs[1])
        assert torch.equal(out[2], trajs[2][:4])
        assert torch.equal(out[3], trajs[3][:3])

    def test_stub_case_b_even_split_at_minimum(self):
        """budget == 2*min_length → even split = (min_length, min_length)."""
        # lengths [3, 5], T_target=5: cumsum 3, 8. delta = 5 - 3 = 2 < min_length=3.
        # Budget = L_j + delta = 3 + 2 = 5. Wait that's odd.
        # Let me use cleaner numbers: lengths [4, 5], T_target=6, min_length=3.
        # Cumsum 4, 9. delta = 6 - 4 = 2 < 3. Budget = 4 + 2 = 6 = 2*min_length.
        # Even split: L_j_kept = ceil(6/2)=3, L_curr_kept = 3. Result: [3, 3].
        trajs = _make_lengths([4, 5])
        out = truncate_chronological_balanced(trajs, T_target=6, min_length=3)
        assert [t.shape[0] for t in out] == [3, 3]
        assert sum(t.shape[0] for t in out) == 6

    def test_stub_case_b_odd_budget_earlier_gets_extra(self):
        """Odd budget → earlier trajectory gets the +1."""
        # lengths [5, 5], T_target=7, min_length=3.
        # Cumsum: 5. delta = 7 - 5 = 2 < min_length. Budget = 5 + 2 = 7.
        # Even split: L_j_kept = ceil(7/2) = 4, L_curr_kept = floor(7/2) = 3.
        trajs = _make_lengths([5, 5])
        out = truncate_chronological_balanced(trajs, T_target=7, min_length=3)
        assert [t.shape[0] for t in out] == [4, 3]
        assert sum(t.shape[0] for t in out) == 7

    def test_stub_case_b_clamp_to_L_curr(self):
        """L_curr too short for the even split → clamp L_curr_kept to
        L_curr, absorb overflow on L_j_kept."""
        # lengths [10, 4, 5], T_target=11, min_length=3.
        # Cumsum: 10. delta = 11 - 10 = 1 < min_length. j = 0.
        # Budget = L_0 + delta = 10 + 1 = 11.
        # Even split (rounded earlier+1): L_j_kept = ceil(11/2) = 6, L_curr_kept = 5.
        # But L_curr (= L_1) = 4 → clamp L_curr_kept = 4, L_j_kept = 11 - 4 = 7.
        # Both ≥ min_length ✓.
        trajs = _make_lengths([10, 4, 5])
        out = truncate_chronological_balanced(trajs, T_target=11, min_length=3)
        assert [t.shape[0] for t in out] == [7, 4]
        assert sum(t.shape[0] for t in out) == 11

    def test_stub_case_b_clamp_to_L_j(self):
        """L_j too short for the even split → clamp L_j_kept to L_j,
        absorb overflow on L_curr_kept."""
        # lengths [3, 8, 5], T_target=4, min_length=3.
        # Cumsum: 3. delta = 4 - 3 = 1 < min_length. j = 0.
        # Budget = L_0 + delta = 3 + 1 = 4. budget < 2*min_length = 6 → infeasible.
        # Need a feasible scenario: lengths [4, 10], T_target=8, min_length=3.
        # Cumsum: 4. delta = 8 - 4 = 4 >= min_length → CASE A, not stub. Bad test.
        # Try: lengths [4, 10, 5], T_target=6, min_length=3.
        # Cumsum: 4. delta = 6 - 4 = 2 < min_length. Budget = L_0 + delta = 4 + 2 = 6.
        # Even split: L_j_kept = ceil(6/2)=3, L_curr_kept=3. No clamp needed.
        # Try: lengths [4, 10], T_target=10, min_length=3.
        # Cumsum: 4. delta = 6 < min_length? No, 6 >= 3, so CASE A. Bad.
        # The L_j-clamp case requires ceil(budget/2) > L_j. Need L_j small but
        # delta close to L_j too. Smallest L_j allowed = min_length = 3 (input
        # constraint). Then budget = 3 + delta where delta < 3, so budget < 6 = 2*min.
        # → always infeasible. So L_j-clamp is unreachable in valid input regimes.
        # Skip this test; covered implicitly by infeasibility check.
        pytest.skip("L_j clamp is unreachable when input trajectories all >= min_length")

    def test_stub_case_b_infeasible_raises(self):
        """budget < 2*min_length → raise."""
        # lengths [3, 5], T_target=4, min_length=3.
        # Cumsum: 3. delta = 4 - 3 = 1 < min_length. j = 0.
        # Budget = L_0 + delta = 3 + 1 = 4 < 2*min_length = 6 → raise.
        trajs = _make_lengths([3, 5])
        with pytest.raises(ValueError, match="budget"):
            truncate_chronological_balanced(trajs, T_target=4, min_length=3)

    def test_no_previous_to_split_with_raises(self):
        """If j = -1 (first trajectory itself would be a stub partial),
        raise — no t_j to redistribute with."""
        # lengths [10], T_target=2, min_length=3.
        # Cumsum: 0 (loop exits with j=-1 because L_0=10 > T_target=2).
        # delta = 2. j < 0. → raise.
        trajs = _make_lengths([10])
        with pytest.raises(ValueError, match="no previous"):
            truncate_chronological_balanced(trajs, T_target=2, min_length=3)

    def test_total_samples_invariant(self):
        """Across many random scenarios, the output total samples equals
        T_target whenever the function returns successfully."""
        import random

        rng = random.Random(0)
        passed = 0
        for _ in range(200):
            K = rng.randint(2, 10)
            lengths = [rng.randint(3, 15) for _ in range(K)]  # all >= min_length
            total = sum(lengths)
            T_target = rng.randint(3, total)
            trajs = _make_lengths(lengths)
            try:
                out = truncate_chronological_balanced(trajs, T_target=T_target, min_length=3)
            except ValueError:
                continue  # infeasible scenarios are fine
            assert sum(t.shape[0] for t in out) == T_target
            assert all(t.shape[0] >= 3 for t in out)
            passed += 1
        # Sanity: most random scenarios should be feasible
        assert passed > 100, f"only {passed}/200 passed; truncation is failing too often"

    def test_numpy_input_returns_numpy(self):
        """Numpy input → numpy output (slicing preserves type)."""
        trajs = [
            np.arange(8, dtype=np.float32).reshape(4, 2),
            np.arange(10, dtype=np.float32).reshape(5, 2),
            np.arange(12, dtype=np.float32).reshape(6, 2),
        ]
        out = truncate_chronological_balanced(trajs, T_target=12, min_length=3)
        assert all(isinstance(t, np.ndarray) for t in out)
        assert sum(t.shape[0] for t in out) == 12

    # ------------------------------------------------------------------
    # Two-threshold mode: min_kept_length < min_length
    # ------------------------------------------------------------------

    def test_two_threshold_redistribute_below_min_length(self):
        """When min_kept_length < min_length: redistribute is triggered
        by `delta < min_length` (the desired floor), but the resulting
        split halves can be as small as min_kept_length (the absolute
        floor). Reproduces the user-data scenario from 2026-05-05 where
        single-threshold mode (min_length=min_kept_length=3000)
        infeasibly required budget >= 6000 for two trajectories near
        the 3-second floor."""
        # Trajectories all just above min_kept_length=1500 but very
        # close to min_length=3000. Total = 9700; T_target = 8123.
        # Cumsum: 3113, 6700, 10287. j=1 (cumsum=6700). delta = 1423 < min_length=3000
        # → redistribute. Budget = L_j + delta = 3587 + 1423 = 5010.
        # Single-threshold (min=3000): need budget >= 6000 → INFEASIBLE.
        # Two-threshold (min_kept=1500): need budget >= 3000 → FEASIBLE.
        # Even split: L_j_kept = ceil(5010/2) = 2505, L_curr_kept = 2505.
        # Both >= 1500 ✓; both above 1500 floor; below 3000 ideal.
        trajs = _make_lengths([3113, 3587, 3000])
        # First, single-threshold mode raises (sanity check):
        with pytest.raises(ValueError, match="2 \\* min_kept_length = 6000"):
            truncate_chronological_balanced(trajs, T_target=8123, min_length=3000)
        # Two-threshold mode succeeds:
        out = truncate_chronological_balanced(
            trajs, T_target=8123, min_length=3000, min_kept_length=1500
        )
        assert sum(t.shape[0] for t in out) == 8123
        # Result: t_0 fully (3113), t_1 trimmed (2505), t_2 partial (2505)
        assert [t.shape[0] for t in out] == [3113, 2505, 2505]
        assert all(t.shape[0] >= 1500 for t in out)

    def test_two_threshold_clean_partial_unaffected(self):
        """If naive partial >= min_length, take the clean partial — no
        redistribution invoked, even when min_kept_length < min_length."""
        trajs = _make_lengths([5, 5, 5])
        # T_target=12, cumsum 5, 10. delta = 2 < min_length=3? → redistribute.
        # Use T_target=13: cumsum 5, 10. delta = 3 = min_length → CASE A.
        out = truncate_chronological_balanced(
            trajs, T_target=13, min_length=3, min_kept_length=1
        )
        assert [t.shape[0] for t in out] == [5, 5, 3]
        assert sum(t.shape[0] for t in out) == 13

    def test_two_threshold_default_back_compat(self):
        """When min_kept_length is omitted, behavior is identical to the
        single-threshold case (min_kept_length defaults to min_length)."""
        trajs = _make_lengths([5, 4, 6, 4])
        out_default = truncate_chronological_balanced(trajs, T_target=16, min_length=3)
        out_explicit = truncate_chronological_balanced(
            trajs, T_target=16, min_length=3, min_kept_length=3
        )
        assert [t.shape[0] for t in out_default] == [t.shape[0] for t in out_explicit]
        for a, b in zip(out_default, out_explicit):
            assert torch.equal(a, b)

    def test_two_threshold_invalid_kept_greater_than_length_raises(self):
        """min_kept_length must be <= min_length."""
        trajs = _make_lengths([5, 5])
        with pytest.raises(ValueError, match="min_kept_length"):
            truncate_chronological_balanced(
                trajs, T_target=8, min_length=3, min_kept_length=5
            )

    def test_two_threshold_min_kept_length_zero_raises(self):
        trajs = _make_lengths([5, 5])
        with pytest.raises(ValueError, match="min_kept_length"):
            truncate_chronological_balanced(
                trajs, T_target=8, min_length=3, min_kept_length=0
            )

    def test_two_threshold_still_infeasible_raises(self):
        """If budget < 2 * min_kept_length even with the lower floor,
        still raise. Tighter trajectories than test_two_threshold_redistribute
        but with min_kept=1500: lengths [1700, 1700], T_target=2200,
        delta = 500. Budget = 1700 + 500 = 2200 < 2*1500=3000 → infeasible."""
        # But first: lengths must be >= min_length input or the upstream
        # filter would have dropped them. Skip the realism check; just
        # show the API behavior.
        trajs = _make_lengths([1700, 1700])
        with pytest.raises(ValueError, match="min_kept_length"):
            truncate_chronological_balanced(
                trajs, T_target=2200, min_length=1000, min_kept_length=1500
            )


# ---------------------------------------------------------------------------
# delay_embed_ragged
# ---------------------------------------------------------------------------

class TestDelayEmbedRagged:
    def test_per_trajectory_matches_direct_call(self):
        """delay_embed_ragged on each trajectory must equal embed_signal_torch
        applied to that trajectory's valid slice — no mixing across the
        NaN boundary."""
        torch.manual_seed(0)
        trajs = [torch.randn(10, 3), torch.randn(15, 3), torch.randn(7, 3)]
        padded, lengths = pad_trajs_to_max(trajs)
        n_delays, delay_spacing = 4, 1
        de_padded, de_lengths = delay_embed_ragged(
            padded, lengths, n_delays=n_delays, delay_spacing=delay_spacing
        )
        # Lengths reduced by (n_delays - 1) per trajectory.
        assert de_lengths.tolist() == [10 - 3, 15 - 3, 7 - 3]
        # Per-trajectory delay-embed matches direct call.
        for i in range(len(trajs)):
            L_i = int(lengths[i].item())
            L_i_de = int(de_lengths[i].item())
            direct = embed_signal_torch(trajs[i][:L_i], n_delays, delay_spacing)
            torch.testing.assert_close(
                de_padded[i, :L_i_de], direct.to(de_padded.dtype),
                rtol=1e-6, atol=1e-6,
            )
            # Beyond valid length: NaN.
            if L_i_de < de_padded.shape[1]:
                assert torch.isnan(de_padded[i, L_i_de:]).all()

    def test_n_delays_one_is_no_op(self):
        torch.manual_seed(0)
        trajs = [torch.randn(5, 2), torch.randn(8, 2)]
        padded, lengths = pad_trajs_to_max(trajs)
        de, dl = delay_embed_ragged(padded, lengths, n_delays=1)
        torch.testing.assert_close(de, padded.clone(), equal_nan=True)
        assert dl.tolist() == lengths.tolist()

    def test_short_trajectories_get_zero_de_length(self):
        """Trajectories shorter than the embedding window should report
        de_length=0 and contribute no valid samples."""
        trajs = [torch.randn(2, 3), torch.randn(10, 3)]
        padded, lengths = pad_trajs_to_max(trajs)
        n_delays = 4  # window covers 4 samples → traj 0 (length 2) is too short
        de, dl = delay_embed_ragged(padded, lengths, n_delays=n_delays)
        assert dl.tolist() == [0, 7]
        # Traj 0 entirely NaN.
        assert torch.isnan(de[0]).all()

    def test_delay_spacing(self):
        """delay_spacing > 1 reduces lengths by (n_delays - 1) * spacing."""
        torch.manual_seed(0)
        trajs = [torch.randn(20, 2)]
        padded, lengths = pad_trajs_to_max(trajs)
        de, dl = delay_embed_ragged(padded, lengths, n_delays=3, delay_spacing=2)
        assert dl.tolist() == [20 - 4]


# ---------------------------------------------------------------------------
# split_balanced_by_timepoints
# ---------------------------------------------------------------------------

class TestSplitBalancedByTimepoints:
    def test_per_condition_balance(self):
        """Each condition is split independently → roughly the requested
        per-condition fractions in TIMEPOINT terms."""
        # 10 trajs cond 0 with widely-varying lengths totalling 1000;
        # 10 trajs cond 1 totalling 500.
        rng = np.random.RandomState(0)
        ls0 = rng.randint(50, 200, size=10)
        ls0 = (ls0 * (1000 / ls0.sum())).round().astype(np.int64)
        ls1 = rng.randint(20, 100, size=10)
        ls1 = (ls1 * (500 / ls1.sum())).round().astype(np.int64)
        lengths = np.concatenate([ls0, ls1])
        source_id = np.array([0] * 10 + [1] * 10)
        train_idx, val_idx, test_idx = split_balanced_by_timepoints(
            lengths, source_id, train_percent=0.7, test_percent=0.15, seed=0,
        )
        assert sorted(np.concatenate([train_idx, val_idx, test_idx]).tolist()) == list(range(20))
        for s in [0, 1]:
            cond_total = lengths[source_id == s].sum()
            tr = lengths[np.intersect1d(train_idx, np.where(source_id == s)[0])].sum()
            va = lengths[np.intersect1d(val_idx, np.where(source_id == s)[0])].sum()
            te = lengths[np.intersect1d(test_idx, np.where(source_id == s)[0])].sum()
            assert abs(tr / cond_total - 0.7) < 0.15
            assert abs(va / cond_total - 0.15) < 0.15
            assert abs(te / cond_total - 0.15) < 0.15

    def test_determinism(self):
        ls = np.full(20, 10, dtype=np.int64)
        src = np.zeros(20, dtype=np.int64)
        a = split_balanced_by_timepoints(ls, src, 0.7, 0.15, seed=0)
        b = split_balanced_by_timepoints(ls, src, 0.7, 0.15, seed=0)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)

    def test_too_few_trajectories_per_source_raises(self):
        with pytest.raises(ValueError, match="need at least 3"):
            split_balanced_by_timepoints(
                np.array([10, 10]), np.array([0, 0]), 0.7, 0.15,
            )

    def test_invalid_percent_raises(self):
        ls = np.full(10, 5, dtype=np.int64)
        src = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="must be in"):
            split_balanced_by_timepoints(ls, src, train_percent=0.7, test_percent=0.5)

    def test_no_leakage_at_trajectory_level(self):
        """Each parent trajectory appears in exactly one of train/val/test."""
        ls = np.full(30, 50, dtype=np.int64)
        src = np.zeros(30, dtype=np.int64)
        train_idx, val_idx, test_idx = split_balanced_by_timepoints(
            ls, src, 0.7, 0.15, seed=42,
        )
        union = np.concatenate([train_idx, val_idx, test_idx])
        assert len(union) == len(set(union.tolist())) == 30
