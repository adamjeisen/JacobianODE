"""Tests for DirectSumCouplingEncoder.

The direct-sum encoder composes N independent CouplingEncoders (one per
subsystem) with a fixed reorder to place every area's dynamic subspace at
the front of the combined latent. Key invariants:

1. encode / decode round-trip (the #1 correctness risk — the decoder's
   "un-group" must exactly invert the encoder's "group").
2. Encoder Jacobian is block-diagonal across areas (the whole point).
3. Volume preservation (|det J| = 1) — each sub-encoder is volume-
   preserving, and the reorder/scatter is a permutation.
4. Dyn-first layout: z[..., :sum(k_i)] equals the concat of per-area dyn
   parts — so LitLatentJacobianODE._split_latent slices the right place.
5. Shared MLP hyperparameters across all sub-encoders.
6. Identity-at-init when final_perm_identity=True + zero_init=True.
7. Partition-violation raises (no silent corruption).

All tests run on small shapes for fast iteration (seconds, not minutes).
"""
from __future__ import annotations

import pytest
import torch

from JacobianODE.fnn.coupling_flows import (
    CouplingEncoder,
    DirectSumCouplingEncoder,
)


def _make(
    area_indices=None,
    n_target_dims_per_block=None,
    coupling_type="additive",
    n_coupling_layers=4,
    hidden_dim=16,
    n_hidden_layers=2,
    final_perm_identity=False,
    zero_init=True,
    permutation_seed_base=0,
):
    if area_indices is None:
        # Contiguous two-area default: [0..4], [5..11]
        area_indices = [list(range(5)), list(range(5, 12))]
    if n_target_dims_per_block is None:
        n_target_dims_per_block = [2, 3]
    return DirectSumCouplingEncoder(
        area_indices=area_indices,
        n_target_dims_per_block=n_target_dims_per_block,
        coupling_type=coupling_type,
        n_coupling_layers=n_coupling_layers,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        final_perm_identity=final_perm_identity,
        zero_init=zero_init,
        permutation_seed_base=permutation_seed_base,
    )


# ---------------------------------------------------------------------------
# 1. Round-trip on contiguous partition
# ---------------------------------------------------------------------------
def test_roundtrip_contiguous():
    torch.manual_seed(0)
    enc = _make()
    enc.eval()
    x = torch.randn(4, enc.n_latent)
    z = enc.encode(x)
    x_recon = enc.decode(z)
    z_recon = enc.encode(x_recon)
    assert torch.allclose(x, x_recon, atol=1e-5), \
        "decode(encode(x)) must equal x"
    assert torch.allclose(z, z_recon, atol=1e-5), \
        "encode(decode(z)) must equal z"


# ---------------------------------------------------------------------------
# 2. Round-trip with non-contiguous (interleaved) indices — stresses
#    the gather/scatter path
# ---------------------------------------------------------------------------
def test_roundtrip_noncontiguous_interleaved():
    torch.manual_seed(1)
    # Odd indices vs even indices — two areas of 4 each.
    enc = _make(
        area_indices=[[0, 2, 4, 6], [1, 3, 5, 7]],
        n_target_dims_per_block=[2, 2],
    )
    enc.eval()
    x = torch.randn(3, 8)
    z = enc.encode(x)
    x_recon = enc.decode(z)
    assert torch.allclose(x, x_recon, atol=1e-5)
    # Check that each area's encoded output ONLY depends on its own input
    # indices (cross-area Jacobian zero — stronger test in test_block_diagonal).
    # Here just a functional check: perturb an "even" input, confirm only the
    # even-area parts of z change.
    x2 = x.clone()
    x2[..., 0] += 1.0  # perturb area 0 (even indices)
    z2 = enc.encode(x2)
    dz = (z2 - z).abs()
    # Output layout: [dyn_0 (2), dyn_1 (2), null_0 (2), null_1 (2)]
    dyn_0 = dz[..., 0:2]; dyn_1 = dz[..., 2:4]
    null_0 = dz[..., 4:6]; null_1 = dz[..., 6:8]
    area_0_output = torch.cat([dyn_0, null_0], dim=-1)
    area_1_output = torch.cat([dyn_1, null_1], dim=-1)
    assert area_0_output.max() > 1e-4, \
        "perturbing area 0 should change area 0's output"
    assert area_1_output.max() < 1e-6, \
        "perturbing area 0 must NOT change area 1's output (block-diagonal)"


# ---------------------------------------------------------------------------
# 3. Block-diagonal encoder Jacobian
# ---------------------------------------------------------------------------
def test_block_diagonal_jacobian():
    """dz_{area_a}/dx_{area_b} = 0 for a != b.

    Respecting the fact that *output* dims are laid out dyn-first-then-null
    across areas, and *input* dims are in the original x-index order, we
    build per-area output and input masks and assert cross-block Jacobian
    entries are zero.
    """
    torch.manual_seed(2)
    # Deliberately non-contiguous to make sure the test uses the index map,
    # not contiguous-slice assumptions.
    area_indices = [[0, 3, 5], [1, 2, 4, 6]]
    k = [1, 2]
    enc = _make(area_indices=area_indices, n_target_dims_per_block=k)
    enc.eval()
    D = enc.n_latent  # 7
    x0 = torch.randn(D)
    J = torch.autograd.functional.jacobian(lambda v: enc.encode(v), x0)
    # J[i_out, j_in] where i_out is in the dyn-first-then-null layout,
    # j_in is in the original x layout.

    # Output-dim → area_id: dyn_0 first (size k[0]), then dyn_1 (size k[1]),
    # then null_0 (size |area_0| - k[0]), then null_1.
    out_area = []
    for a, ka in enumerate(k):
        out_area.extend([a] * ka)
    for a, (ka, ia) in enumerate(zip(k, area_indices)):
        out_area.extend([a] * (len(ia) - ka))
    out_area = torch.tensor(out_area)  # len D

    # Input-dim → area_id: index lookup.
    in_area = torch.empty(D, dtype=torch.long)
    for a, idxs in enumerate(area_indices):
        in_area[idxs] = a

    # Cross-block entries: out_area != in_area → should be zero.
    cross_mask = out_area.unsqueeze(1) != in_area.unsqueeze(0)
    cross_J = J[cross_mask]
    assert cross_J.abs().max() < 1e-6, (
        f"cross-area Jacobian entries must be zero, got max |J_ij|="
        f"{cross_J.abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 4. Volume preservation — |det J| = 1
# ---------------------------------------------------------------------------
def test_volume_preservation():
    torch.manual_seed(3)
    enc = _make(coupling_type="additive")  # additive coupling is VP
    enc.eval()
    x0 = torch.randn(enc.n_latent).double()
    enc_double = enc.double()
    J = torch.autograd.functional.jacobian(
        lambda v: enc_double.encode(v), x0
    )
    sign, logabsdet = torch.linalg.slogdet(J)
    assert abs(sign.item()) == 1, "determinant sign should be ±1"
    assert abs(logabsdet.item()) < 1e-4, (
        f"log |det J| should be ~0 for volume-preserving flow, "
        f"got {logabsdet.item():.2e}"
    )


# ---------------------------------------------------------------------------
# 5. Dyn-first layout invariant
# ---------------------------------------------------------------------------
def test_dyn_first_layout():
    """z[..., :n_target_dims_total] equals the per-area dyn parts concatenated."""
    torch.manual_seed(4)
    area_indices = [[0, 1, 2], [3, 4, 5, 6]]
    k = [1, 2]
    enc = _make(area_indices=area_indices, n_target_dims_per_block=k)
    enc.eval()
    x = torch.randn(2, enc.n_latent)
    z = enc.encode(x)
    # Reconstruct per-area expected dyn parts by encoding each area
    # separately through the sub-encoder.
    x_vis = x.index_select(-1, enc._area_idx(0))
    x_cog = x.index_select(-1, enc._area_idx(1))
    z_vis = enc.blocks[0](x_vis)
    z_cog = enc.blocks[1](x_cog)
    expected_dyn = torch.cat([z_vis[..., :k[0]], z_cog[..., :k[1]]], dim=-1)
    assert torch.allclose(
        z[..., :enc.n_target_dims], expected_dyn, atol=1e-6
    ), "z[..., :n_target_dims] must be the per-area dyn-parts concatenated"


# ---------------------------------------------------------------------------
# 6. Shared MLP hyperparameters across sub-encoders
# ---------------------------------------------------------------------------
def test_shared_mlp_hyperparams():
    """All sub-encoders must share conditioner MLP structure — different
    n_input / different permutation_seed, but identical hidden_dim /
    n_hidden_layers / n_coupling_layers / coupling_type."""
    # Note: areas have DIFFERENT sizes to make sure they're indeed distinct
    # sub-encoders but with same MLP hyperparameters.
    enc = _make(
        area_indices=[[0, 1, 2, 3], [4, 5, 6, 7, 8]],  # sizes 4, 5 — different
        n_target_dims_per_block=[2, 2],
        coupling_type="additive",
        n_coupling_layers=4,
        hidden_dim=16,
        n_hidden_layers=2,
    )
    blk0, blk1 = enc.blocks[0], enc.blocks[1]
    assert blk0._coupling_type == blk1._coupling_type
    assert len(blk0.coupling_layers) == len(blk1.coupling_layers)
    # Both have additive coupling; check each layer's MLP dims match via a
    # parameter-shape spot check.
    for lay0, lay1 in zip(blk0.coupling_layers, blk1.coupling_layers):
        params0 = [p.shape for p in lay0.parameters()]
        params1 = [p.shape for p in lay1.parameters()]
        # Hidden-layer shapes must match (middle params); only the first
        # layer's input dim and last layer's output dim should differ due
        # to different n_input.
        hidden0 = [s for s in params0 if s == torch.Size([16, 16])]
        hidden1 = [s for s in params1 if s == torch.Size([16, 16])]
        assert len(hidden0) == len(hidden1) and len(hidden0) > 0, (
            "shared MLP hidden widths must match across sub-encoders"
        )


# ---------------------------------------------------------------------------
# 7. Identity at init
# ---------------------------------------------------------------------------
def test_identity_at_init():
    """With final_perm_identity=True + zero_init=True, each sub-encoder is
    the identity at init, so encode(x) at init is x gathered into the
    dyn-first layout — i.e. for contiguous areas, encode(x) is a known
    permutation of x."""
    torch.manual_seed(5)
    # Contiguous case for clean comparison.
    area_indices = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # two areas, size 5 each
    k = [2, 3]
    enc = _make(
        area_indices=area_indices,
        n_target_dims_per_block=k,
        final_perm_identity=True,
        zero_init=True,
    )
    enc.eval()
    x = torch.randn(4, enc.n_latent)
    z = enc.encode(x)
    # Each sub-encoder should be identity at init given final_perm_identity
    # + zero_init; the top-level reorder then produces:
    #   z = [x_area0[:k0], x_area1[:k1], x_area0[k0:], x_area1[k1:]]
    a0_idx, a1_idx = enc._area_idx(0), enc._area_idx(1)
    x_a0 = x.index_select(-1, a0_idx)
    x_a1 = x.index_select(-1, a1_idx)
    expected = torch.cat([
        x_a0[..., :k[0]], x_a1[..., :k[1]],
        x_a0[..., k[0]:], x_a1[..., k[1]:],
    ], dim=-1)
    assert torch.allclose(z, expected, atol=1e-5), (
        "at init with final_perm_identity + zero_init, encoder should be the "
        "identity-up-to-reorder"
    )


# ---------------------------------------------------------------------------
# 8. Partition-violation raises on construction
# ---------------------------------------------------------------------------
def test_partition_violation_raises():
    with pytest.raises(ValueError, match="partition"):
        # Duplicate index 0
        _make(area_indices=[[0, 1], [0, 2]], n_target_dims_per_block=[1, 1])
    with pytest.raises(ValueError, match="partition"):
        # Gap — index 2 missing
        _make(area_indices=[[0], [1, 3]], n_target_dims_per_block=[0, 1])
    with pytest.raises(ValueError, match="n_target_dims_per_block"):
        # k > n for area 0
        _make(area_indices=[[0, 1], [2, 3]], n_target_dims_per_block=[5, 1])
    with pytest.raises(ValueError, match="at least one area"):
        DirectSumCouplingEncoder(area_indices=[], n_target_dims_per_block=[])


# ---------------------------------------------------------------------------
# 9. N-ary generalization (3 areas)
# ---------------------------------------------------------------------------
def test_three_areas_roundtrip():
    torch.manual_seed(6)
    enc = _make(
        area_indices=[[0, 1], [2, 3, 4], [5, 6, 7, 8]],
        n_target_dims_per_block=[1, 2, 2],
    )
    enc.eval()
    x = torch.randn(3, enc.n_latent)
    z = enc.encode(x)
    x_recon = enc.decode(z)
    assert torch.allclose(x, x_recon, atol=1e-5)
    assert enc.n_target_dims == 5
    assert enc.n_latent == 9
