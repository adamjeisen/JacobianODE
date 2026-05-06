"""Tests for ``postprocess_per_condition`` — the extracted per-source
postprocessing helper.

The function was extracted from inline logic in
``run_jacobians.train()`` (`run_jacobians.py:166-208`) so both
``train(cfg)`` and the new ``train_from_arrays(...)`` API can call the
same code path. These tests verify equivalence with the inline behavior
+ correctness on synthetic multi-source data.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from JacobianODE.jacobians.data.processing import (
    postprocess_data,
    postprocess_per_condition,
    PostprocessPerConditionResult,
)


def _make_cfg(normalize: bool = True, obs_noise: float = 0.0):
    return OmegaConf.create({
        "data": {
            "postprocessing": {
                "obs_noise": obs_noise,
                "filter_data": False,
                "normalize": normalize,
            },
        },
    })


# ---------------------------------------------------------------------------
# Equivalence with inline behavior
# ---------------------------------------------------------------------------

class TestEquivalenceWithInline:
    """Verify the extracted function produces byte-identical results to the
    original inline loop in run_jacobians.py:166-208 (modulo equality).
    """

    def test_numpy_input_matches_inline(self):
        """Numpy input + zero noise + normalize → must match inline loop
        bitwise (no random component)."""
        rng = np.random.default_rng(0)
        src0 = rng.standard_normal(size=(5, 10, 4)).astype(np.float64)
        src1 = rng.standard_normal(size=(7, 10, 4)).astype(np.float64) + 100
        values_raw = np.concatenate([src0, src1], axis=0)
        src_ids = np.concatenate([np.zeros(5), np.ones(7)]).astype(np.int64)

        cfg = _make_cfg(normalize=True, obs_noise=0.0)

        # --- Inline reference (literal copy of run_jacobians.py:174-205) ---
        ref_values = np.empty_like(values_raw, dtype=np.float64)
        ref_mu_per_source = []
        ref_sigma_per_source = []
        ref_nsf_per_source = []
        for s in np.unique(src_ids):
            mask = src_ids == s
            sub_raw = values_raw[mask]
            sub_result = postprocess_data(cfg, sub_raw)
            ref_values[mask] = sub_result.values
            ref_mu_per_source.append(float(sub_result.mu))
            ref_sigma_per_source.append(float(sub_result.sigma))
            ref_nsf_per_source.append(float(sub_result.noise_scale_factor))
        ref_mu = float(np.mean(ref_mu_per_source))
        ref_sigma = float(np.mean(ref_sigma_per_source))
        ref_nsf = float(np.mean(ref_nsf_per_source))

        # --- New function ---
        result = postprocess_per_condition(cfg, values_raw, source_id=src_ids)

        # Bitwise equality on the values (no random noise, deterministic).
        assert np.array_equal(result.values, ref_values)
        # Float equality on the scalars (computed via the same operations).
        assert result.mu == ref_mu
        assert result.sigma == ref_sigma
        assert result.noise_scale_factor == ref_nsf
        assert result.mu_per_source == ref_mu_per_source
        assert result.sigma_per_source == ref_sigma_per_source
        assert result.noise_scale_factor_per_source == ref_nsf_per_source
        # source_ids comes back as Python ints in ascending order
        assert result.source_ids == [0, 1]

    def test_torch_input_dtype_promotion(self):
        """Torch tensor input → output is torch.float64 (matches the
        inline `torch.empty_like(values_raw, dtype=torch.float64)`)."""
        torch.manual_seed(0)
        src0 = torch.randn(3, 6, 2, dtype=torch.float32)
        src1 = torch.randn(4, 6, 2, dtype=torch.float32) + 50
        values_raw = torch.cat([src0, src1], dim=0)
        src_ids = np.array([0, 0, 0, 1, 1, 1, 1])

        cfg = _make_cfg(normalize=True, obs_noise=0.0)
        result = postprocess_per_condition(cfg, values_raw, source_id=src_ids)

        assert isinstance(result.values, torch.Tensor)
        assert result.values.dtype == torch.float64
        assert result.values.shape == values_raw.shape

    def test_three_sources(self):
        """Algorithm scales beyond 2 sources."""
        rng = np.random.default_rng(2)
        src0 = rng.standard_normal(size=(4, 5, 3)).astype(np.float64)
        src1 = rng.standard_normal(size=(3, 5, 3)).astype(np.float64) + 50
        src2 = rng.standard_normal(size=(5, 5, 3)).astype(np.float64) + 200
        values_raw = np.concatenate([src0, src1, src2], axis=0)
        src_ids = np.concatenate([
            np.zeros(4), np.ones(3), 2 * np.ones(5),
        ]).astype(np.int64)

        cfg = _make_cfg(normalize=True, obs_noise=0.0)
        result = postprocess_per_condition(cfg, values_raw, source_id=src_ids)

        assert len(result.mu_per_source) == 3
        assert result.source_ids == [0, 1, 2]
        # Per-source means should match raw means (since obs_noise=0)
        assert result.mu_per_source[0] == pytest.approx(src0.mean(), abs=1e-9)
        assert result.mu_per_source[1] == pytest.approx(src1.mean(), abs=1e-9)
        assert result.mu_per_source[2] == pytest.approx(src2.mean(), abs=1e-9)


# ---------------------------------------------------------------------------
# Behavior correctness
# ---------------------------------------------------------------------------

class TestBehaviorCorrectness:
    def test_per_source_centering_independent(self):
        """Each source's normalized chunk should have mean ≈ 0 — NOT
        offset by half the inter-source distance like pooled centering
        would produce."""
        rng = np.random.default_rng(3)
        src0 = rng.standard_normal(size=(5, 10, 4)).astype(np.float64)
        src1 = rng.standard_normal(size=(7, 10, 4)).astype(np.float64) + 100
        values_raw = np.concatenate([src0, src1], axis=0)
        src_ids = np.concatenate([np.zeros(5), np.ones(7)]).astype(np.int64)

        cfg = _make_cfg(normalize=True, obs_noise=0.0)
        result = postprocess_per_condition(cfg, values_raw, source_id=src_ids)

        # Each source's chunk should be centered at zero
        assert abs(result.values[src_ids == 0].mean()) < 1e-9
        assert abs(result.values[src_ids == 1].mean()) < 1e-9
        # Per-source mu should be ~0 and ~100
        assert result.mu_per_source[0] == pytest.approx(src0.mean(), abs=1e-9)
        assert result.mu_per_source[1] == pytest.approx(src1.mean(), abs=1e-9)

    def test_top_level_mu_is_mean_across_sources(self):
        """The top-level scalar mu/sigma/noise_scale_factor are the mean
        across sources — for back-compat with downstream code."""
        rng = np.random.default_rng(4)
        src0 = rng.standard_normal(size=(4, 6, 2)).astype(np.float64) + 10
        src1 = rng.standard_normal(size=(4, 6, 2)).astype(np.float64) + 30
        values_raw = np.concatenate([src0, src1], axis=0)
        src_ids = np.concatenate([np.zeros(4), np.ones(4)]).astype(np.int64)

        cfg = _make_cfg(normalize=True, obs_noise=0.0)
        result = postprocess_per_condition(cfg, values_raw, source_id=src_ids)

        # mu = mean of [src0.mean(), src1.mean()] ≈ mean of [10, 30] = 20
        expected_mu = float(np.mean([src0.mean(), src1.mean()]))
        assert result.mu == pytest.approx(expected_mu, abs=1e-9)

    def test_raw_values_to_use_for_noise_passed_per_source(self):
        """When `raw_values_to_use_for_noise` is provided, it's also
        sliced per-source (matches inline behavior at run_jacobians:187)."""
        rng = np.random.default_rng(5)
        values_raw = rng.standard_normal(size=(6, 5, 3)).astype(np.float64)
        # Distinct noise reference: source 0 small magnitude, source 1 large
        noise_ref = np.zeros((6, 5, 3))
        noise_ref[:3] = 0.01 * rng.standard_normal(size=(3, 5, 3))   # src 0
        noise_ref[3:] = 100.0 * rng.standard_normal(size=(3, 5, 3))  # src 1
        src_ids = np.array([0, 0, 0, 1, 1, 1])

        cfg = _make_cfg(normalize=False, obs_noise=0.05)
        result = postprocess_per_condition(
            cfg, values_raw, source_id=src_ids,
            raw_values_to_use_for_noise=noise_ref,
        )

        # Per-source noise_scale_factor differs by orders of magnitude
        # (src 1's noise ref is ~10000x larger).
        assert (
            result.noise_scale_factor_per_source[1]
            > 100 * result.noise_scale_factor_per_source[0]
        )

    def test_preserves_row_order(self):
        """Output rows must be in the same order as input (per-source
        results stitched back via mask, not concatenated by sort)."""
        # Interleaved source ids (not sorted) — test that ordering is preserved.
        rng = np.random.default_rng(42)
        values_raw = rng.standard_normal((8, 3, 2)).astype(np.float64) + 100
        # source_id alternates 0, 1, 0, 1, ... → after per-source norm,
        # rows 0/2/4/6 (src 0) should be centered to src 0's mean,
        # rows 1/3/5/7 (src 1) to src 1's mean.
        # Add a constant offset to src 1 so we can detect mis-ordering.
        src_ids = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        values_raw[src_ids == 1] += 1000.0

        cfg = _make_cfg(normalize=True, obs_noise=0.0)
        result = postprocess_per_condition(cfg, values_raw, source_id=src_ids)

        # If row order is preserved, the normalized rows for src 1 (rows
        # 1, 3, 5, 7) should be centered at zero (after their own mean
        # subtraction), NOT shifted by 1000.
        for i in [1, 3, 5, 7]:
            assert abs(result.values[i].mean()) < 1.0, (
                f"row {i} (src 1) not properly centered — row order broken"
            )
        for i in [0, 2, 4, 6]:
            assert abs(result.values[i].mean()) < 1.0, (
                f"row {i} (src 0) not properly centered — row order broken"
            )


class TestEdgeCases:
    def test_single_source(self):
        """source_id with all-same value → equivalent to a single
        postprocess_data call."""
        values_raw = np.random.RandomState(0).randn(5, 4, 3).astype(np.float64)
        src_ids = np.zeros(5, dtype=np.int64)
        cfg = _make_cfg(normalize=True, obs_noise=0.0)

        pc = postprocess_per_condition(cfg, values_raw, source_id=src_ids)
        ref = postprocess_data(cfg, values_raw)

        assert np.array_equal(pc.values, ref.values)
        assert pc.mu == ref.mu
        assert pc.sigma == ref.sigma

    def test_int_source_ids(self):
        """source_id can be int64 / int32 / Python list — all handled."""
        values_raw = np.random.RandomState(0).randn(4, 3, 2).astype(np.float64)
        for src_ids in [
            np.array([0, 0, 1, 1], dtype=np.int64),
            np.array([0, 0, 1, 1], dtype=np.int32),
            [0, 0, 1, 1],
        ]:
            cfg = _make_cfg(normalize=True, obs_noise=0.0)
            result = postprocess_per_condition(cfg, values_raw, source_id=src_ids)
            assert result.source_ids == [0, 1]
