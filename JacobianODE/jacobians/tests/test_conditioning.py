"""Tests for the optional per-sample conditioning argument ``c``.

Covers the encoder side (CouplingEncoder + DirectSumCouplingEncoder), the
dynamics MLP, and end-to-end LitLatentJacobianODE plumbing. Two invariants
matter most:

  (i) ``c=None`` preserves the exact unconditioned behavior — old configs
      / checkpoints / tests don't notice the new argument exists.
  (ii) When the model is built with ``condition_dim > 0`` and ``c`` is
       passed, encode/decode round-trips exactly with the same ``c`` and
       the model's outputs vary meaningfully with ``c``.
"""

import pytest
import torch

from JacobianODE.fnn.coupling_flows import CouplingEncoder, DirectSumCouplingEncoder
from JacobianODE.models.mlp import MLP
from JacobianODE.models.latent_jacobian import LitLatentJacobianODE
from JacobianODE.jacobians.data.splitting import (
    TimeSeriesDataset,
    collate_with_optional_condition,
)


def _seed(s=0):
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# CouplingEncoder
# ---------------------------------------------------------------------------

class TestCouplingEncoderConditioning:
    def test_unconditioned_c_none_matches_no_arg(self):
        _seed(0)
        e = CouplingEncoder(
            n_input=8, n_coupling_layers=2, hidden_dim=16, n_hidden_layers=1,
            zero_init=False, near_identity_std=0.1, final_perm_identity=True,
        )
        x = torch.randn(3, 5, 8)
        assert torch.allclose(e.encode(x), e.encode(x, None))
        assert torch.allclose(e.decode(e.encode(x)), e.decode(e.encode(x), None))

    def test_unconditioned_roundtrip_exact(self):
        _seed(0)
        e = CouplingEncoder(
            n_input=8, n_coupling_layers=4, hidden_dim=16, n_hidden_layers=1,
            zero_init=False, near_identity_std=0.1, final_perm_identity=True,
        )
        x = torch.randn(3, 5, 8)
        assert torch.allclose(e.decode(e.encode(x)), x, atol=1e-5)

    def test_conditioned_roundtrip_exact_with_same_c(self):
        _seed(0)
        e = CouplingEncoder(
            n_input=8, n_coupling_layers=4, hidden_dim=16, n_hidden_layers=1,
            zero_init=True, near_identity_std=1e-3, condition_dim=2,
            final_perm_identity=True,
        )
        x = torch.randn(3, 5, 8)
        c = torch.randn(3, 2)
        assert torch.allclose(e.decode(e.encode(x, c), c), x, atol=1e-5)

    def test_conditioned_output_varies_with_c(self):
        _seed(0)
        e = CouplingEncoder(
            n_input=8, n_coupling_layers=2, hidden_dim=16, n_hidden_layers=1,
            zero_init=True, near_identity_std=1e-2, condition_dim=2,
            final_perm_identity=True,
        )
        x = torch.randn(3, 5, 8)
        c1 = torch.zeros(3, 2)
        c2 = torch.ones(3, 2) * 5.0
        z1 = e.encode(x, c1)
        z2 = e.encode(x, c2)
        # Cayley/perm/identity-init guarantees the *first* coupling-layer
        # output of x_a is just x_a; so z[..., :split_dim] may match across
        # c1 and c2 when n_coupling_layers is small. The full z, however,
        # mixes through subsequent layers — overall difference must be > 0.
        assert (z1 - z2).abs().mean().item() > 1e-4

    def test_conditioned_raises_when_c_missing(self):
        e = CouplingEncoder(
            n_input=4, n_coupling_layers=1, hidden_dim=8, n_hidden_layers=1,
            condition_dim=2,
        )
        x = torch.randn(2, 4)
        with pytest.raises(ValueError, match="condition_dim"):
            e.encode(x)


# ---------------------------------------------------------------------------
# DirectSumCouplingEncoder
# ---------------------------------------------------------------------------

class TestDirectSumConditioning:
    def test_unconditioned_back_compat(self):
        _seed(0)
        ds = DirectSumCouplingEncoder(
            area_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
            n_target_dims_per_block=[2, 2],
            n_coupling_layers=2, hidden_dim=8, n_hidden_layers=1,
        )
        x = torch.randn(3, 5, 8)
        assert torch.allclose(ds.decode(ds.encode(x)), x, atol=1e-5)

    def test_conditioned_roundtrip_exact(self):
        _seed(0)
        ds = DirectSumCouplingEncoder(
            area_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
            n_target_dims_per_block=[2, 2],
            n_coupling_layers=2, hidden_dim=8, n_hidden_layers=1,
            zero_init=True, near_identity_std=1e-3, condition_dim=2,
            final_perm_identity=True,
        )
        x = torch.randn(3, 5, 8)
        c = torch.randn(3, 2)
        assert torch.allclose(ds.decode(ds.encode(x, c), c), x, atol=1e-5)

    def test_shared_condition_across_areas(self):
        """Both sub-encoders must receive the SAME c (shared conditioning)."""
        _seed(0)
        ds = DirectSumCouplingEncoder(
            area_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
            n_target_dims_per_block=[2, 2],
            n_coupling_layers=2, hidden_dim=8, n_hidden_layers=1,
            zero_init=True, near_identity_std=1e-2, condition_dim=2,
        )
        x = torch.randn(3, 5, 8)
        c1 = torch.zeros(3, 2)
        c2 = torch.ones(3, 2) * 3.0
        z1 = ds.encode(x, c1)
        z2 = ds.encode(x, c2)
        # Both areas should respond to c — so the non-area-mixed output
        # differs across both halves of the latent dim.
        assert (z1[..., :4] - z2[..., :4]).abs().mean().item() > 1e-4
        assert (z1[..., 4:] - z2[..., 4:]).abs().mean().item() > 1e-4


# ---------------------------------------------------------------------------
# MLP (dynamics)
# ---------------------------------------------------------------------------

class TestMLPConditioning:
    def test_unconditioned_c_none_matches_no_arg(self):
        _seed(0)
        m = MLP(input_dim=4, hidden_dim=[8], num_layers=1, output_dim=8,
                residuals=False, dropout=0.0, activation='silu')
        x = torch.randn(3, 5, 4)
        assert torch.allclose(m(x), m(x, None))

    def test_conditioned_first_layer_input_dim(self):
        m = MLP(input_dim=4, hidden_dim=[8], num_layers=1, output_dim=8,
                condition_dim=3)
        # First layer must accept input_dim + condition_dim = 7 features.
        first_linear = next(layer for layer in m.layers
                            if isinstance(layer, torch.nn.Linear))
        assert first_linear.in_features == 4 + 3

    def test_conditioned_output_varies_with_c(self):
        _seed(0)
        m = MLP(input_dim=4, hidden_dim=[16], num_layers=1, output_dim=8,
                condition_dim=2, activation='silu')
        x = torch.randn(3, 5, 4)
        c1 = torch.zeros(3, 2)
        c2 = torch.ones(3, 2) * 5.0
        assert (m(x, c1) - m(x, c2)).abs().mean().item() > 1e-3

    def test_conditioned_raises_when_c_missing(self):
        m = MLP(input_dim=4, hidden_dim=[8], num_layers=1, output_dim=8,
                condition_dim=2)
        with pytest.raises(ValueError, match="condition_dim"):
            m(torch.randn(2, 4))


# ---------------------------------------------------------------------------
# LitLatentJacobianODE end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def conditioned_lit():
    _seed(0)
    D = 4
    n_target_dims = 2
    cdim = 1
    enc = CouplingEncoder(
        n_input=D, n_coupling_layers=2, hidden_dim=8, n_hidden_layers=1,
        zero_init=True, near_identity_std=1e-3, condition_dim=cdim,
        final_perm_identity=True,
    )
    mlp = MLP(
        input_dim=n_target_dims, hidden_dim=[8], num_layers=1,
        output_dim=n_target_dims ** 2, condition_dim=cdim,
        residuals=False, dropout=0.0, activation='silu',
    )
    lit = LitLatentJacobianODE(
        model=mlp, encoder=enc, dt=1.0,
        n_target_dims=n_target_dims, prediction_steps=4,
        loop_closure_training=False, trajectory_training=True,
        optimizer_kwargs={'lr': 1e-4},
        n_delays=1, obs_dim=D,
        use_scheduler=False,
        jacobianODEint_kwargs={'traj_init_steps': 4},
    )
    return lit, D, n_target_dims, cdim


class TestLitConditioned:
    def test_encode_decode_roundtrip_with_c(self, conditioned_lit):
        lit, D, _, cdim = conditioned_lit
        batch = torch.randn(2, 16, D)
        c = torch.randn(2, cdim)
        z = lit.encode_trajectory(batch, c)
        x = lit.decode_trajectory(z, c)
        assert torch.allclose(x, batch, atol=1e-4)

    def test_training_step_unpacks_tuple(self, conditioned_lit):
        lit, D, _, cdim = conditioned_lit
        batch = torch.randn(2, 16, D)
        c = torch.randn(2, cdim)
        loss = lit.training_step((batch, c), batch_idx=0)
        assert torch.is_tensor(loss) and loss.dim() == 0
        assert torch.isfinite(loss)

    def test_training_step_loss_varies_with_c(self, conditioned_lit):
        """Same batch, different c → different loss (model uses c)."""
        lit, D, _, cdim = conditioned_lit
        torch.manual_seed(42)
        batch = torch.randn(2, 16, D)
        c1 = torch.zeros(2, cdim)
        c2 = torch.ones(2, cdim) * 5.0
        loss1 = lit.training_step((batch, c1), batch_idx=0).item()
        loss2 = lit.training_step((batch, c2), batch_idx=0).item()
        assert abs(loss1 - loss2) > 1e-4

    def test_validation_step_unpacks_tuple(self, conditioned_lit):
        lit, D, _, cdim = conditioned_lit
        batch = torch.randn(2, 16, D)
        c = torch.randn(2, cdim)
        loss = lit.validation_step((batch, c), batch_idx=0, log_metrics=False)
        assert torch.is_tensor(loss)


# ---------------------------------------------------------------------------
# Data pipeline (dataset + collate)
# ---------------------------------------------------------------------------

class TestDataPipeline:
    def test_dataset_no_condition_returns_tensor(self):
        ds = TimeSeriesDataset(torch.randn(5, 10, 3))
        item = ds[0]
        assert torch.is_tensor(item)

    def test_dataset_with_condition_returns_tuple(self):
        seqs = torch.randn(5, 10, 3)
        conds = torch.randn(5, 1)
        ds = TimeSeriesDataset(seqs, condition=conds)
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2
        assert torch.is_tensor(item[0]) and torch.is_tensor(item[1])

    def test_dataset_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            TimeSeriesDataset(torch.randn(5, 10, 3), condition=torch.randn(4, 1))

    def test_collate_tensor_path_matches_default(self):
        items = [torch.randn(10, 3) for _ in range(4)]
        out = collate_with_optional_condition(items)
        assert torch.is_tensor(out) and out.shape == (4, 10, 3)

    def test_collate_tuple_path_returns_pair(self):
        items = [(torch.randn(10, 3), torch.randn(1)) for _ in range(4)]
        out = collate_with_optional_condition(items)
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == (4, 10, 3)
        assert out[1].shape == (4, 1)
