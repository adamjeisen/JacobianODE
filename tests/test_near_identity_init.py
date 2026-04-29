"""Tests for near_identity_std plumbing in coupling_flows.

near_identity_std controls how the LAST conditioner layer is initialised
when zero_init=True:
  - 0.0 (default): strict zeros — exact identity-at-init, hidden weights
    have zero gradient on step 1 (they unstick on step 2 once the last
    layer has moved off zero).
  - >0.0: weight ~ N(0, std), bias = 0 — approximately the identity at
    init (output magnitude ~ std × ‖hidden activations‖); hidden weights
    receive non-zero gradient from step 1.

This test pins:
  1. Default behavior is unchanged (strict zeros).
  2. near_identity_std > 0 produces a non-zero last weight at the right scale.
  3. The bias is always zero under zero_init.
  4. The encoder is approximately the identity at init for small std.
  5. Plumbing reaches CouplingEncoder, AdditiveCouplingLayer, MADE,
     SplineAutoregressiveLayer, etc.
  6. Hidden-layer first-step gradients are non-zero with near_identity_std>0
     and zero with strict zero_init (the design motivation).
"""
from __future__ import annotations

import math

import pytest
import torch

from JacobianODE.fnn.coupling_flows import (
    AdditiveCouplingLayer,
    AffineCouplingLayer,
    AnalyticAutoregressiveEncoder,
    CouplingEncoder,
    DirectSumCouplingEncoder,
    MADE,
    SplineAutoregressiveEncoder,
    _build_conditioner,
)


# ---------- _build_conditioner --------------------------------------------


def test_default_strict_zero_init():
    """Default near_identity_std=0 → exact zeros (current behaviour)."""
    torch.manual_seed(0)
    cond = _build_conditioner(
        input_dim=8, output_dim=4, hidden_dim=16, n_hidden_layers=2
    )
    last = cond[-1]
    assert (last.weight == 0).all()
    assert (last.bias == 0).all()


def test_near_identity_init_scale():
    """near_identity_std=σ → last weight ~ N(0, σ); bias still zero."""
    torch.manual_seed(0)
    cond = _build_conditioner(
        input_dim=64,
        output_dim=64,
        hidden_dim=256,
        n_hidden_layers=2,
        near_identity_std=1e-3,
    )
    last = cond[-1]
    assert not (last.weight == 0).all()
    # Empirical std should be close to target with this many params
    obs_std = float(last.weight.std())
    assert math.isclose(obs_std, 1e-3, rel_tol=0.15), f"got {obs_std}"
    assert (last.bias == 0).all()


def test_near_identity_ignored_when_zero_init_false():
    """If zero_init=False, near_identity_std has no effect — PyTorch default init."""
    torch.manual_seed(0)
    cond = _build_conditioner(
        input_dim=8,
        output_dim=4,
        hidden_dim=16,
        n_hidden_layers=2,
        zero_init=False,
        near_identity_std=1e-3,
    )
    last = cond[-1]
    # PyTorch default Linear init: weight ~ uniform(-sqrt(1/in), sqrt(1/in)),
    # NOT N(0, 1e-3). std should be much larger.
    assert float(last.weight.std()) > 1e-2


# ---------- per-class plumbing --------------------------------------------


@pytest.mark.parametrize("std", [0.0, 1e-3])
def test_additive_layer_near_identity_plumbing(std):
    torch.manual_seed(0)
    layer = AdditiveCouplingLayer(
        dim=12,
        hidden_dim=64,
        n_hidden_layers=2,
        near_identity_std=std,
    )
    last = layer.conditioner[-1]
    assert (last.bias == 0).all()
    if std == 0.0:
        assert (last.weight == 0).all()
    else:
        assert not (last.weight == 0).all()


@pytest.mark.parametrize("std", [0.0, 1e-3])
def test_affine_layer_near_identity_plumbing(std):
    torch.manual_seed(0)
    layer = AffineCouplingLayer(
        dim=12,
        hidden_dim=64,
        n_hidden_layers=2,
        near_identity_std=std,
    )
    last = layer.conditioner[-1]
    assert (last.bias == 0).all()
    if std == 0.0:
        assert (last.weight == 0).all()
    else:
        assert not (last.weight == 0).all()


@pytest.mark.parametrize("coupling_type", ["additive", "affine", "spline"])
def test_coupling_encoder_near_identity_plumbing(coupling_type):
    torch.manual_seed(0)
    enc = CouplingEncoder(
        n_input=12,
        n_coupling_layers=3,
        coupling_type=coupling_type,
        hidden_dim=32,
        n_hidden_layers=2,
        near_identity_std=1e-3,
    )
    for layer in enc.coupling_layers:
        last = layer.conditioner[-1]
        assert not (last.weight == 0).all(), f"{coupling_type}: last weight is zero"
        assert (last.bias == 0).all()


def test_directsum_encoder_near_identity_plumbing():
    torch.manual_seed(0)
    enc = DirectSumCouplingEncoder(
        area_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        n_target_dims_per_block=[2, 2],
        n_coupling_layers=2,
        coupling_type="additive",
        hidden_dim=32,
        n_hidden_layers=2,
        near_identity_std=1e-3,
    )
    for sub in enc.blocks:
        for layer in sub.coupling_layers:
            last = layer.conditioner[-1]
            assert not (last.weight == 0).all()
            assert (last.bias == 0).all()


def test_made_near_identity_plumbing():
    torch.manual_seed(0)
    made = MADE(
        n_features=8,
        hidden_dim=32,
        n_hidden_layers=2,
        output_dim_per_input=4,
        near_identity_std=1e-3,
    )
    last = made.layers[-1].linear
    assert not (last.weight == 0).all()
    assert (last.bias == 0).all()


def test_spline_ar_encoder_near_identity_plumbing():
    torch.manual_seed(0)
    enc = SplineAutoregressiveEncoder(
        n_input=8,
        n_layers=2,
        hidden_dim=32,
        n_hidden_layers=2,
        num_bins=4,
        use_actnorm=False,
        near_identity_std=1e-3,
    )
    for ar in enc.ar_layers:
        last = ar.made.layers[-1].linear
        assert not (last.weight == 0).all()


def test_analytic_ar_encoder_near_identity_plumbing():
    torch.manual_seed(0)
    enc = AnalyticAutoregressiveEncoder(
        n_input=8,
        n_layers=2,
        hidden_dim=32,
        n_hidden_layers=2,
        use_actnorm=False,
        near_identity_std=1e-3,
    )
    for ar in enc.ar_layers:
        last = ar.made.layers[-1].linear
        assert not (last.weight == 0).all()


# ---------- behavioral checks ---------------------------------------------


def test_encoder_approximately_identity_at_init():
    """With small near_identity_std + final_perm_identity, encoder ≈ identity."""
    torch.manual_seed(0)
    enc = CouplingEncoder(
        n_input=16,
        n_coupling_layers=4,
        coupling_type="additive",
        hidden_dim=32,
        n_hidden_layers=2,
        near_identity_std=1e-3,
        final_perm_identity=True,
    )
    enc.eval()
    x = torch.randn(8, 16)
    z = enc(x)
    # Pointwise deviation should be << 1 with σ=1e-3 over 4 stacked layers.
    # Each layer adds ~σ × ‖h‖ ~ σ × sqrt(hidden_dim) ≈ 1e-3 × ~6 ~ 0.006
    # Stacked 4 layers → max deviation O(0.05).
    deviation = (z - x).abs().max().item()
    assert deviation < 0.1, f"encoder too far from identity at init: max |z-x| = {deviation}"
    # And not zero — that would mean we accidentally reverted to strict zero_init
    assert deviation > 1e-5, f"encoder is exactly identity, near-id init not active"


def test_strict_zero_init_first_step_kills_hidden_gradient():
    """Design motivation: strict zero_init → hidden weights get zero gradient
    on step 1 because they backprop through W_last = 0."""
    torch.manual_seed(0)
    layer = AdditiveCouplingLayer(
        dim=8,
        hidden_dim=16,
        n_hidden_layers=2,
        near_identity_std=0.0,  # strict zero_init
    )
    x = torch.randn(4, 8, requires_grad=False)
    y, _ = layer(x)
    loss = y.pow(2).sum()
    loss.backward()
    last = layer.conditioner[-1]
    first = layer.conditioner[0]  # first hidden Linear
    # Last-layer weight: NON-zero gradient (h ≠ 0, ∂L/∂y_b ≠ 0)
    assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0
    # First hidden layer: gradient flows back through W_last = 0 → ZERO
    assert first.weight.grad is not None
    assert first.weight.grad.abs().sum().item() == 0.0


def test_near_identity_init_gives_first_step_hidden_gradient():
    """near_identity_std > 0 → hidden weights receive non-zero gradient on step 1."""
    torch.manual_seed(0)
    layer = AdditiveCouplingLayer(
        dim=8,
        hidden_dim=16,
        n_hidden_layers=2,
        near_identity_std=1e-3,
    )
    x = torch.randn(4, 8, requires_grad=False)
    y, _ = layer(x)
    loss = y.pow(2).sum()
    loss.backward()
    first = layer.conditioner[0]
    # Now first hidden layer DOES get gradient because W_last is non-zero
    grad_norm = first.weight.grad.abs().sum().item()
    assert grad_norm > 0.0, f"first hidden layer still gets zero gradient: {grad_norm}"
