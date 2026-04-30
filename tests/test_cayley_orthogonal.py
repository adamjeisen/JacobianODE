"""Tests for CayleyOrthogonal layer + use_cayley_perms plumbing.

CayleyOrthogonal parameterises a learnable orthogonal mixing layer via
the Cayley transform Q = (I - A)(I + A)^{-1}, A skew-symmetric. It's a
drop-in replacement for FixedPermutation in coupling-flow encoders that
preserves volume (det Q = +1) and is exactly invertible (Q^{-1} = Q^T).

Tests:
  - identity at init (U=0)
  - orthogonality (Q Q^T = I) for arbitrary skew A
  - det Q = +1 (rotation, not reflection)
  - exact round-trip via inverse
  - gradient flows through the parameter U
  - plumbing into CouplingEncoder + DirectSumCouplingEncoder
  - use_cayley_perms=True is identity-at-init equivalent to
    use_cayley_perms=False under zero_init + final_perm_identity
  - mutually exclusive with init_pca_basis
"""
from __future__ import annotations

import pytest
import torch

from JacobianODE.fnn.coupling_flows import (
    CayleyOrthogonal,
    CouplingEncoder,
    DirectSumCouplingEncoder,
    FixedPermutation,
)


# ---------- CayleyOrthogonal layer ---------------------------------------


class TestCayleyOrthogonalLayer:
    def test_identity_at_init(self):
        layer = CayleyOrthogonal(dim=8)
        x = torch.randn(3, 8)
        y = layer(x)
        assert torch.allclose(y, x, atol=1e-6), \
            f"At init (U=0), Cayley should be identity; got max diff {(y-x).abs().max():.2e}"

    def test_orthogonality_under_perturbation(self):
        torch.manual_seed(0)
        layer = CayleyOrthogonal(dim=10)
        layer.U.data = torch.randn_like(layer.U.data) * 0.5
        Q = layer._orthogonal()
        I = torch.eye(10)
        err = (Q @ Q.T - I).abs().max().item()
        assert err < 1e-5, f"Q should be orthogonal; ||Q Q^T - I||_inf = {err:.2e}"

    def test_det_is_plus_one(self):
        torch.manual_seed(1)
        layer = CayleyOrthogonal(dim=6)
        layer.U.data = torch.randn_like(layer.U.data) * 0.3
        Q = layer._orthogonal()
        det = torch.linalg.det(Q).item()
        assert abs(det - 1.0) < 1e-5, \
            f"Cayley produces SO(n) (det=+1, no reflection); got {det:.6f}"

    def test_round_trip(self):
        torch.manual_seed(2)
        layer = CayleyOrthogonal(dim=7)
        layer.U.data = torch.randn_like(layer.U.data) * 0.4
        x = torch.randn(4, 7)
        y = layer(x)
        x_back = layer.inverse(y)
        err = (x - x_back).abs().max().item()
        assert err < 1e-5, f"Round-trip should be exact; got max diff {err:.2e}"

    def test_gradient_flows(self):
        torch.manual_seed(3)
        layer = CayleyOrthogonal(dim=5)
        layer.U.data = torch.randn_like(layer.U.data) * 0.1
        x = torch.randn(2, 5)
        loss = layer(x).pow(2).mean()
        loss.backward()
        assert layer.U.grad is not None
        assert layer.U.grad.abs().sum() > 0, "Gradient should flow through U"

    def test_diagonal_and_lower_triangle_of_U_dont_matter(self):
        """Only the strict upper triangle of U enters via U_strict - U_strict^T."""
        torch.manual_seed(4)
        layer1 = CayleyOrthogonal(dim=6)
        torch.manual_seed(4)
        layer2 = CayleyOrthogonal(dim=6)
        # Set both with the same upper triangle, different diag/lower.
        upper = torch.randn(6, 6) * 0.3
        layer1.U.data = torch.triu(upper, diagonal=1)
        layer2.U.data = torch.triu(upper, diagonal=1) + torch.tril(torch.randn(6, 6))
        Q1 = layer1._orthogonal()
        Q2 = layer2._orthogonal()
        assert torch.allclose(Q1, Q2, atol=1e-6), \
            "Diagonal and lower triangle of U should not affect Q"


# ---------- CouplingEncoder integration -----------------------------------


class TestCouplingEncoderCayleyPlumbing:
    def test_use_cayley_perms_replaces_fixed_permutation(self):
        enc = CouplingEncoder(
            n_input=8, n_coupling_layers=4, coupling_type="additive",
            hidden_dim=16, n_hidden_layers=2,
            use_cayley_perms=True,
        )
        for p in enc.permutations:
            assert isinstance(p, CayleyOrthogonal), \
                f"With use_cayley_perms, permutations should be CayleyOrthogonal; got {type(p).__name__}"

    def test_use_cayley_perms_false_keeps_fixed_permutation(self):
        enc = CouplingEncoder(
            n_input=8, n_coupling_layers=4, coupling_type="additive",
            hidden_dim=16, n_hidden_layers=2,
            use_cayley_perms=False,
        )
        for p in enc.permutations:
            assert isinstance(p, FixedPermutation)

    def test_identity_at_init_with_zero_init_and_final_perm(self):
        """With zero_init=True + final_perm_identity=True, both
        encoder variants should map x -> x exactly at init.

        For Cayley: inter-layer mixers start at identity (U=0), and
        final_perm = identity, so the whole stack is identity.
        For Fixed: random perms compose into some R; final_perm = R^{-1}
        unwinds them, so the whole stack is identity.
        """
        n_input = 10
        x = torch.randn(3, n_input)

        torch.manual_seed(7)
        enc_fixed = CouplingEncoder(
            n_input=n_input, n_coupling_layers=4, coupling_type="additive",
            hidden_dim=16, n_hidden_layers=2,
            zero_init=True, near_identity_std=0.0,
            final_perm_identity=True, use_cayley_perms=False,
        )
        torch.manual_seed(7)
        enc_cayley = CouplingEncoder(
            n_input=n_input, n_coupling_layers=4, coupling_type="additive",
            hidden_dim=16, n_hidden_layers=2,
            zero_init=True, near_identity_std=0.0,
            final_perm_identity=True, use_cayley_perms=True,
        )

        z_fixed = enc_fixed(x)
        z_cayley = enc_cayley(x)
        assert torch.allclose(z_fixed, x, atol=1e-6), \
            f"fixed-perm encoder not identity at init; max diff {(z_fixed-x).abs().max():.2e}"
        assert torch.allclose(z_cayley, x, atol=1e-6), \
            f"cayley-perm encoder not identity at init; max diff {(z_cayley-x).abs().max():.2e}"

    def test_round_trip_under_perturbed_cayley(self):
        torch.manual_seed(0)
        enc = CouplingEncoder(
            n_input=12, n_coupling_layers=3, coupling_type="additive",
            hidden_dim=16, n_hidden_layers=2,
            zero_init=True, near_identity_std=1e-2,
            final_perm_identity=True, use_cayley_perms=True,
        )
        # Perturb the Cayley parameters
        for p in enc.permutations:
            assert isinstance(p, CayleyOrthogonal)
            p.U.data = torch.randn_like(p.U.data) * 0.2
        x = torch.randn(2, 12)
        z = enc(x)
        x_back = enc.inverse(z)
        err = (x - x_back).abs().max().item()
        assert err < 1e-4, f"Round-trip not exact under Cayley perturbation; got {err:.2e}"

    def test_init_pca_basis_with_cayley_raises(self):
        with pytest.raises(ValueError, match="init_pca_basis"):
            CouplingEncoder(
                n_input=8, n_coupling_layers=4, coupling_type="additive",
                hidden_dim=16, n_hidden_layers=2,
                init_pca_basis=True, pca_basis=torch.eye(8),
                use_cayley_perms=True,
            )


# ---------- DirectSumCouplingEncoder integration --------------------------


class TestDirectSumCayleyPlumbing:
    def test_propagates_to_subblocks(self):
        ds = DirectSumCouplingEncoder(
            area_indices=[[0,1,2,3], [4,5,6,7]],
            n_target_dims_per_block=[2, 2],
            n_coupling_layers=3, coupling_type="additive",
            hidden_dim=16, n_hidden_layers=2,
            use_cayley_perms=True,
        )
        for block in ds.blocks:
            for p in block.permutations:
                assert isinstance(p, CayleyOrthogonal), \
                    "DirectSum should propagate use_cayley_perms to sub-encoders"

    def test_round_trip(self):
        torch.manual_seed(0)
        ds = DirectSumCouplingEncoder(
            area_indices=[[0,1,2,3], [4,5,6,7]],
            n_target_dims_per_block=[2, 2],
            n_coupling_layers=3, coupling_type="additive",
            hidden_dim=16, n_hidden_layers=2,
            zero_init=True, near_identity_std=1e-2,
            final_perm_identity=True,
            use_cayley_perms=True,
        )
        # Perturb Cayley params across all sub-encoders
        for block in ds.blocks:
            for p in block.permutations:
                p.U.data = torch.randn_like(p.U.data) * 0.15
        x = torch.randn(2, 8)
        z = ds(x)
        x_back = ds.inverse(z)
        err = (x - x_back).abs().max().item()
        assert err < 1e-4, f"DirectSum round-trip not exact; got {err:.2e}"
