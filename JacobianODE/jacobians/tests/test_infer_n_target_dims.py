"""Tests for ``infer_n_target_dims`` — the extracted PCA/FNN-autodim helper.

Verifies equivalence with the inline behavior in
``run_jacobians.py:284-402`` for all four branches:
- PCA single-encoder
- PCA DirectSum (per-area)
- FNN single-encoder
- FNN DirectSum (per-area)

Plus the no-op case (PCA with no var_threshold → returns None).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from JacobianODE.jacobians.training.autodim import (
    infer_n_target_dims,
    InferNTargetDimsResult,
)


def _make_cfg(
    method: str = "pca",
    var_threshold: float | None = None,
    fnn_threshold: float = 0.01,
    encoder_target: str = "JacobianODE.fnn.coupling_flows.CouplingEncoder",
    area_indices: list | None = None,
):
    encoder = {"_target_": encoder_target}
    if area_indices is not None:
        encoder["area_indices"] = area_indices
    return OmegaConf.create({
        "model": {
            "n_target_dim_method": method,
            "n_target_var_threshold": var_threshold,
            "n_target_fnn_threshold": fnn_threshold,
            "encoder": encoder,
        },
    })


def _low_rank_data(n_traj=20, T=50, d_embed=10, intrinsic_dim=3, seed=0):
    """Generate (N_traj, T, D) data with a known low intrinsic dimension.
    First `intrinsic_dim` PCs carry essentially all variance."""
    rng = np.random.default_rng(seed)
    # Random (B*T, intrinsic) latent + random projection to D_embed
    latent = rng.standard_normal((n_traj * T, intrinsic_dim)) * 10
    proj = rng.standard_normal((intrinsic_dim, d_embed))
    flat = latent @ proj
    # Add tiny noise on the orthogonal complement so PCA can still see "all dims"
    flat += rng.standard_normal(flat.shape) * 0.001
    return torch.from_numpy(flat.reshape(n_traj, T, d_embed))


# ---------------------------------------------------------------------------
# PCA single-encoder
# ---------------------------------------------------------------------------

class TestPCASingleEncoder:
    def test_returns_None_when_no_var_threshold(self):
        """method='pca' (default) AND var_threshold=None → no autodim."""
        cfg = _make_cfg(method="pca", var_threshold=None)
        seq = _low_rank_data()
        assert infer_n_target_dims(cfg, seq) is None

    def test_picks_intrinsic_dim_for_low_rank_data(self):
        """For data with known intrinsic dim 3 + tiny noise, PCA at
        threshold=0.99 should pick close to 3."""
        cfg = _make_cfg(method="pca", var_threshold=0.99)
        seq = _low_rank_data(intrinsic_dim=3, d_embed=10)
        result = infer_n_target_dims(cfg, seq)
        assert result is not None
        assert isinstance(result, InferNTargetDimsResult)
        assert result.method == "pca"
        assert result.is_direct_sum is False
        assert result.n_target_dims_per_block is None
        assert result.n_target_dims_total == 3
        assert isinstance(result.pca_cum_var, float)
        assert result.pca_cum_var >= 0.99

    def test_higher_threshold_picks_more_dims(self):
        """Threshold 0.999 should pick at least as many dims as 0.99."""
        seq = _low_rank_data(intrinsic_dim=3, d_embed=10)
        cfg_99 = _make_cfg(method="pca", var_threshold=0.99)
        cfg_999 = _make_cfg(method="pca", var_threshold=0.999)
        n_99 = infer_n_target_dims(cfg_99, seq).n_target_dims_total
        n_999 = infer_n_target_dims(cfg_999, seq).n_target_dims_total
        assert n_999 >= n_99

    def test_byte_equivalence_with_inline(self):
        """Compute the inline version manually, compare to the helper."""
        seq = _low_rank_data(intrinsic_dim=3, d_embed=10, seed=42)
        var_threshold = 0.95

        # --- Inline reference (literal copy of run_jacobians.py:352-384) ---
        flat = seq.reshape(-1, seq.shape[-1]).to(torch.float64)
        flat_centered = flat - flat.mean(dim=0, keepdim=True)
        cov = (flat_centered.T @ flat_centered) / (flat_centered.shape[0] - 1)
        eigvals_asc, _ = torch.linalg.eigh(cov)
        eigvals = eigvals_asc.flip(0).clamp_min(0.0)
        explained = eigvals / eigvals.sum()
        cum_var = explained.cumsum(0)
        ref_n_target = int((cum_var >= var_threshold).float().argmax().item()) + 1
        ref_cum_var = float(cum_var[ref_n_target - 1].item())

        # --- Helper ---
        cfg = _make_cfg(method="pca", var_threshold=var_threshold)
        result = infer_n_target_dims(cfg, seq)
        assert result.n_target_dims_total == ref_n_target
        assert result.pca_cum_var == ref_cum_var


# ---------------------------------------------------------------------------
# PCA DirectSum (per-area)
# ---------------------------------------------------------------------------

class TestPCADirectSum:
    def test_per_area_picks_independently(self):
        """For DirectSum with 2 areas, each area's intrinsic dim is found
        independently and summed."""
        # Build (N_traj, T, D=10) with 2 areas:
        #   area 0 = [0..4], intrinsic_dim 2
        #   area 1 = [5..9], intrinsic_dim 3
        rng = np.random.default_rng(0)
        n_traj, T = 20, 30
        a0 = rng.standard_normal((n_traj * T, 2)) @ rng.standard_normal((2, 5))
        a1 = rng.standard_normal((n_traj * T, 3)) @ rng.standard_normal((3, 5))
        flat = np.concatenate([a0, a1], axis=1)
        flat += rng.standard_normal(flat.shape) * 0.001
        seq = torch.from_numpy(flat.reshape(n_traj, T, 10))

        cfg = _make_cfg(
            method="pca", var_threshold=0.99,
            encoder_target="JacobianODE.fnn.coupling_flows.DirectSumCouplingEncoder",
            area_indices=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        )
        result = infer_n_target_dims(cfg, seq)
        assert result is not None
        assert result.is_direct_sum is True
        assert result.method == "pca"
        assert result.n_target_dims_per_block == [2, 3]
        assert result.n_target_dims_total == 5
        assert isinstance(result.pca_cum_var, list)
        assert len(result.pca_cum_var) == 2

    def test_returns_none_no_var_threshold(self):
        """DirectSum with no var_threshold → still no-op."""
        cfg = _make_cfg(
            method="pca", var_threshold=None,
            encoder_target="JacobianODE.fnn.coupling_flows.DirectSumCouplingEncoder",
            area_indices=[[0, 1, 2], [3, 4, 5]],
        )
        seq = _low_rank_data(d_embed=6)
        assert infer_n_target_dims(cfg, seq) is None


# ---------------------------------------------------------------------------
# FNN paths (just smoke tests — FNN is harder to predict exactly without
# the algorithm specifics)
# ---------------------------------------------------------------------------

class TestFNNPaths:
    def test_fnn_single_encoder_runs(self):
        cfg = _make_cfg(method="fnn", fnn_threshold=0.01)
        seq = _low_rank_data(intrinsic_dim=2, d_embed=8, seed=10)
        result = infer_n_target_dims(cfg, seq)
        assert result is not None
        assert result.method == "fnn"
        assert result.is_direct_sum is False
        assert isinstance(result.n_target_dims_total, int)
        assert result.n_target_dims_total >= 1
        assert result.n_target_dims_per_block is None
        assert result.pca_cum_var is None

    def test_fnn_direct_sum_runs(self):
        rng = np.random.default_rng(0)
        seq = torch.from_numpy(rng.standard_normal((20, 30, 8)))
        cfg = _make_cfg(
            method="fnn", fnn_threshold=0.05,
            encoder_target="JacobianODE.fnn.coupling_flows.DirectSumCouplingEncoder",
            area_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        )
        result = infer_n_target_dims(cfg, seq)
        assert result is not None
        assert result.method == "fnn"
        assert result.is_direct_sum is True
        assert len(result.n_target_dims_per_block) == 2
        assert result.n_target_dims_total == sum(result.n_target_dims_per_block)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_method_raises(self):
        cfg = _make_cfg(method="bogus", var_threshold=0.99)
        seq = _low_rank_data()
        with pytest.raises(ValueError, match="must be 'pca' or 'fnn'"):
            infer_n_target_dims(cfg, seq)

    def test_pure_does_not_mutate_cfg(self):
        """Helper must not write to cfg.model.n_target_dims (caller's job)."""
        cfg = _make_cfg(method="pca", var_threshold=0.99)
        seq = _low_rank_data(intrinsic_dim=3, d_embed=10)
        # Snapshot cfg before
        before = OmegaConf.to_yaml(cfg)
        infer_n_target_dims(cfg, seq)
        after = OmegaConf.to_yaml(cfg)
        assert before == after, "infer_n_target_dims must NOT mutate cfg"
