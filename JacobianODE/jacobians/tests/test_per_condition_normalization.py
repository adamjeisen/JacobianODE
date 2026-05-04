"""Smoke test for per-condition postprocessing (normalize_per_condition).

Trains nothing — just walks the postprocess branch in run_jacobians by
calling its logic on a synthetic combined-source `sol` and verifies that
each source's grand mean is removed independently (NOT the pooled mean)
and that per-source mu/sigma are recoverable.
"""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from JacobianODE.jacobians.data.processing import postprocess_data


def _make_cfg(normalize: bool, normalize_per_condition: bool):
    """Minimal cfg with the postprocessing fields postprocess_data reads."""
    return OmegaConf.create({
        "data": {
            "postprocessing": {
                "obs_noise": 0.0,                    # zero-out noise to make math exact
                "filter_data": False,
                "normalize": normalize,
                "normalize_per_condition": normalize_per_condition,
            },
        },
    })


def test_per_source_grand_mean_centering_independent():
    """Calling postprocess_data per-source produces per-source grand-mean
    centering that differs from the pooled grand mean.

    Source 0 has values centered around 0; source 1 has values centered
    around 100. The pooled grand mean is ~50 — if we centered using that,
    neither source ends up at zero. Per-source centering puts each one
    at zero individually."""
    rng = np.random.default_rng(0)
    src0 = rng.standard_normal(size=(5, 10, 4)).astype(np.float64)        # ~N(0,1)
    src1 = rng.standard_normal(size=(7, 10, 4)).astype(np.float64) + 100  # ~N(100,1)
    pooled = np.concatenate([src0, src1], axis=0)
    src_ids = np.concatenate([np.zeros(5), np.ones(7)]).astype(np.int64)

    cfg = _make_cfg(normalize=True, normalize_per_condition=True)

    # Mirror run_jacobians' per-condition branch directly.
    out = np.empty_like(pooled)
    mus = []
    sigmas = []
    for s in np.unique(src_ids):
        mask = src_ids == s
        sub = pooled[mask]
        result = postprocess_data(cfg, sub)
        out[mask] = result.values
        mus.append(result.mu)
        sigmas.append(result.sigma)

    # Per-source grand means recovered.
    assert mus[0] == pytest.approx(src0.mean(), abs=1e-9)
    assert mus[1] == pytest.approx(src1.mean(), abs=1e-9)
    # The two means differ by ~100 (the synthetic offset).
    assert mus[1] - mus[0] == pytest.approx(100.0, abs=0.5)

    # Each source's normalized chunk should now have mean ≈ 0 (per-source
    # grand mean) — NOT the pooled mean (which would leave each source
    # offset by ±50 from zero).
    assert abs(out[src_ids == 0].mean()) < 1e-9
    assert abs(out[src_ids == 1].mean()) < 1e-9


def test_pooled_centering_still_offset_per_source():
    """Sanity check the comparison case: when we center on POOLED data
    (the across-condition path), each source's normalized chunk is offset
    from zero by half the inter-source distance."""
    rng = np.random.default_rng(1)
    src0 = rng.standard_normal(size=(5, 10, 4)).astype(np.float64)
    src1 = rng.standard_normal(size=(7, 10, 4)).astype(np.float64) + 100
    pooled = np.concatenate([src0, src1], axis=0)

    cfg = _make_cfg(normalize=True, normalize_per_condition=False)
    result = postprocess_data(cfg, pooled)
    out = result.values

    src0_mean_after = out[:5].mean()
    src1_mean_after = out[5:].mean()
    # Source 0 ends up ≈ -50/sigma; source 1 ≈ +50/sigma (sign matters).
    assert src0_mean_after < 0
    assert src1_mean_after > 0
    # And they straddle zero, equal magnitudes (modulo group-size weight).
    assert abs(src0_mean_after + src1_mean_after) < abs(src0_mean_after - src1_mean_after)
