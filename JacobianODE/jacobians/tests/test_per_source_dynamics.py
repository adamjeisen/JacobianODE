"""Unit tests for PerSourceDynamicsMLP routing wrapper."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from JacobianODE.models.per_source_dynamics import PerSourceDynamicsMLP


class _IdMLPPlusBias(nn.Module):
    """Tiny stand-in for a dynamics MLP: returns x + bias, ignores c.

    Lets us tell sub-MLPs apart by their (deterministic) output.
    """

    def __init__(self, bias: float, dim: int) -> None:
        super().__init__()
        self.register_buffer("bias", torch.full((dim,), float(bias)))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return x + self.bias


def test_routing_nearest_neighbor():
    """c[..., 0] closest to section_values[s] dispatches to mlps[s]."""
    dim = 3
    mlps = [_IdMLPPlusBias(0.0, dim), _IdMLPPlusBias(100.0, dim)]
    wrapper = PerSourceDynamicsMLP(mlps, section_condition_values=[0.0, 1.0])

    x = torch.zeros(4, dim)
    # c = [0.1, 0.2, 0.7, 0.9] → nearest to {0.0, 1.0} → [0, 0, 1, 1]
    c = torch.tensor([[0.1], [0.2], [0.7], [0.9]])
    y = wrapper(x, c)

    expected = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0],
    ])
    torch.testing.assert_close(y, expected)


def test_routing_uses_only_first_condition_dim():
    """Routing should ignore c[..., 1:]."""
    dim = 2
    mlps = [_IdMLPPlusBias(-1.0, dim), _IdMLPPlusBias(7.0, dim)]
    wrapper = PerSourceDynamicsMLP(mlps, section_condition_values=[0.0, 5.0])

    x = torch.zeros(3, dim)
    # c[..., 0] = [0.1, 4.9, 5.5]; c[..., 1] is junk that shouldn't affect routing.
    c = torch.tensor([[0.1, 999.0], [4.9, -42.0], [5.5, 0.0]])
    y = wrapper(x, c)

    expected = torch.tensor([
        [-1.0, -1.0],
        [7.0, 7.0],
        [7.0, 7.0],
    ])
    torch.testing.assert_close(y, expected)


def test_intermediate_time_dim_preserved():
    """Wrapper must handle (B, T, D) inputs, not just (B, D)."""
    dim = 4
    mlps = [_IdMLPPlusBias(0.0, dim), _IdMLPPlusBias(10.0, dim)]
    wrapper = PerSourceDynamicsMLP(mlps, section_condition_values=[0.0, 1.0])

    x = torch.zeros(3, 5, dim)  # (B=3, T=5, D=4)
    c = torch.tensor([[0.0], [1.0], [1.0]])
    y = wrapper(x, c)

    assert y.shape == (3, 5, dim)
    torch.testing.assert_close(y[0], torch.zeros(5, dim))
    torch.testing.assert_close(y[1], torch.full((5, dim), 10.0))
    torch.testing.assert_close(y[2], torch.full((5, dim), 10.0))


def test_independent_parameters():
    """Sub-MLPs must hold independent params (separate gradients)."""
    dim = 2
    real_mlps = [
        nn.Sequential(nn.Linear(dim + 1, dim)),  # +1 for c[..., 0]
        nn.Sequential(nn.Linear(dim + 1, dim)),
    ]

    class _Cat(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, c):
            xc = torch.cat([x, c[..., :1]], dim=-1)
            return self.m(xc)

    wrapper = PerSourceDynamicsMLP(
        [_Cat(m) for m in real_mlps], section_condition_values=[0.0, 1.0]
    )

    # Total parameter count = 2 × (Linear(3, 2)) = 2 × (3*2 + 2) = 16
    assert sum(p.numel() for p in wrapper.parameters()) == 16

    # Hit only sub-MLP 1: only its grads should be non-None / non-zero.
    x = torch.randn(4, dim, requires_grad=False)
    c = torch.full((4, 1), 1.0)  # all routed to s=1
    y = wrapper(x, c).sum()
    y.backward()

    g0 = real_mlps[0][0].weight.grad
    g1 = real_mlps[1][0].weight.grad
    assert g0 is None or g0.abs().sum().item() == 0.0
    assert g1 is not None and g1.abs().sum().item() > 0.0


def test_construction_validation():
    """Misconfigured input must raise."""
    dim = 2
    with pytest.raises(ValueError, match="must agree in length"):
        PerSourceDynamicsMLP(
            [_IdMLPPlusBias(0.0, dim), _IdMLPPlusBias(1.0, dim)],
            section_condition_values=[0.0],
        )
    with pytest.raises(ValueError, match="needs ≥2"):
        PerSourceDynamicsMLP(
            [_IdMLPPlusBias(0.0, dim)],
            section_condition_values=[0.0],
        )


def test_no_samples_to_one_branch_is_okay():
    """If no sample routes to a sub-MLP, wrapper should still run."""
    dim = 2
    mlps = [_IdMLPPlusBias(0.0, dim), _IdMLPPlusBias(50.0, dim)]
    wrapper = PerSourceDynamicsMLP(mlps, section_condition_values=[0.0, 1.0])
    x = torch.zeros(3, dim)
    c = torch.zeros(3, 1)  # all routed to s=0
    y = wrapper(x, c)
    torch.testing.assert_close(y, torch.zeros(3, dim))
