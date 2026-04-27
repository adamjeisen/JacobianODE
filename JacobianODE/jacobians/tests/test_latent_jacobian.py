"""Tests for LitLatentJacobianODE end-to-end latent Jacobian model."""

import pytest
import torch
import torch.nn as nn

from JacobianODE.models.mlp import MLP
from JacobianODE.models.latent_jacobian import LitLatentJacobianODE
from JacobianODE.fnn.networks import MLPAutoencoder
from JacobianODE.fnn.sequence_networks import SequenceAutoencoder


@pytest.fixture
def small_latent_model():
    """Create a small LitLatentJacobianODE for testing."""
    n_latent = 3
    time_window = 10
    n_features = 2

    encoder = MLPAutoencoder(
        n_latent=n_latent,
        time_window=time_window,
        n_features=n_features,
        network_shape=[16, 16],
    )

    jac_model = MLP(
        input_dim=n_latent,
        hidden_dim=[16, 16],
        num_layers=2,
        output_dim=n_latent * n_latent,
        activation='silu',
    )

    lit_model = LitLatentJacobianODE(
        model=jac_model,
        encoder=encoder,
        prediction_steps=3,
        dt=0.1,
        direct=True,
        loop_closure_training=True,
        loop_closure_weight=1.0,
        trajectory_training=True,
        teacher_forcing_annealing=False,
        jacobianODEint_kwargs={
            'traj_init_steps': 5,
            'inner_path': 'line',
            'inner_N': 2,
            'interp_pts': 2,
        },
    )
    return lit_model


@pytest.fixture
def sample_obs_batch():
    """Create a sample observation batch.

    Shape: (B=2, T=30, D_obs=2) — long enough for:
    - time_window=10 encoder → T' = 30 - 10 + 1 = 21
    - jac_window = traj_init_steps(5) + prediction_steps(3) = 8
    """
    torch.manual_seed(0)
    return torch.randn(2, 30, 2)


class TestEncodeDecodeTrajectory:
    """Test encoder/decoder trajectory abstractions."""

    def test_encode_shape(self, small_latent_model, sample_obs_batch):
        z = small_latent_model.encode_trajectory(sample_obs_batch)
        # T' = 30 - 10 + 1 = 21
        assert z.shape == (2, 21, 3)

    def test_decode_shape(self, small_latent_model):
        z = torch.randn(2, 5, 3)
        decoded = small_latent_model.decode_trajectory(z)
        # Each latent vector decodes to (time_window, n_features)
        assert decoded.shape == (2, 5, 10, 2)

    def test_encode_decode_roundtrip(self, small_latent_model, sample_obs_batch):
        """Verify encode then decode produces output with correct shapes."""
        z = small_latent_model.encode_trajectory(sample_obs_batch)
        decoded = small_latent_model.decode_trajectory(z)
        assert decoded.shape == (2, 21, 10, 2)


class TestComputeJacobians:
    """Test Jacobian computation in latent space."""

    def test_jacobian_shape(self, small_latent_model):
        z = torch.randn(2, 10, 3)
        jacs = small_latent_model.compute_jacobians(z)
        assert jacs.shape == (2, 10, 3, 3)

    def test_jacobian_single_point(self, small_latent_model):
        z = torch.randn(3)
        jacs = small_latent_model.compute_jacobians(z)
        assert jacs.shape == (3, 3)


class TestLyapunovExponents:
    """Test Lyapunov exponent computation."""

    def test_shape(self):
        jacs = torch.randn(50, 3, 3) * 0.1
        exponents = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt=0.1)
        assert exponents.shape == (3,)

    def test_sorted_descending(self):
        jacs = torch.randn(100, 4, 4) * 0.1
        exponents = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt=0.1)
        for i in range(len(exponents) - 1):
            assert exponents[i] >= exponents[i + 1]

    def test_identity_jacobians(self):
        """For J=0 (identity propagator), Lyapunov exponents should be ~0."""
        jacs = torch.zeros(100, 3, 3)
        exponents = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt=0.1)
        assert torch.allclose(exponents, torch.zeros(3), atol=1e-5)


class TestTrajectoryModelStep:
    """Test the trajectory_model_step method."""

    def test_returns_loss(self, small_latent_model, sample_obs_batch):
        small_latent_model.eval()
        result = small_latent_model.trajectory_model_step(sample_obs_batch)
        assert 'loss' in result
        assert 'metric_vals' in result
        assert 'outputs' in result
        assert result['loss'].dim() == 0  # scalar
        assert torch.isfinite(result['loss'])

    def test_too_short_trajectory_raises(self, small_latent_model):
        """Trajectory too short for encoder + JacobianODE window."""
        short_batch = torch.randn(2, 12, 2)  # T'=3, need 8
        with pytest.raises(ValueError, match="shorter than required"):
            small_latent_model.trajectory_model_step(short_batch)


class TestTrainingStep:
    """Test the full training step."""

    def test_training_step_runs(self, small_latent_model, sample_obs_batch):
        small_latent_model.train()
        loss = small_latent_model.training_step(sample_obs_batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_gradients_flow(self, small_latent_model, sample_obs_batch):
        """Verify gradients flow through encoder, Jacobian model, and decoder."""
        small_latent_model.train()
        loss = small_latent_model.training_step(sample_obs_batch, batch_idx=0)
        loss.backward()

        # Check encoder has gradients
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_latent_model.encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should receive gradients"

        # Check Jacobian model has gradients
        model_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_latent_model.model.parameters()
        )
        assert model_has_grad, "Jacobian model should receive gradients"


class TestValidationStep:
    """Test the validation step."""

    def test_validation_step_runs(self, small_latent_model, sample_obs_batch):
        small_latent_model.eval()
        with torch.no_grad():
            loss = small_latent_model.validation_step(
                sample_obs_batch, batch_idx=0, log_metrics=False
            )
        assert torch.isfinite(loss)


class TestConfigIntegration:
    """Test config/factory integration."""

    def test_initialize_config_latent(self):
        """Test that initialize_config handles the encoder key."""
        from JacobianODE.jacobians.core.config import load_config, initialize_config

        cfg = load_config(overrides=["model=latent_mlp", "data=custom"])
        cfg = initialize_config(cfg, data_dim=1)

        # Jacobian MLP should be sized for latent space
        assert cfg.model.params.input_dim == 10  # n_latent
        assert cfg.model.params.output_dim == 100  # n_latent^2

        # Lightning target should be LitLatentJacobianODE
        assert "LitLatentJacobianODE" in cfg.training.lightning._target_

    def test_make_model_latent(self):
        """Test that make_model creates a LitLatentJacobianODE."""
        from JacobianODE.jacobians.core.config import load_config, initialize_config
        from JacobianODE.jacobians.training.model_factory import make_model

        cfg = load_config(overrides=["model=latent_mlp", "data=custom"])
        cfg = initialize_config(cfg, data_dim=1)
        lit_model = make_model(cfg, dt=0.1)

        assert isinstance(lit_model, LitLatentJacobianODE)
        assert hasattr(lit_model, 'encoder')
        assert isinstance(lit_model.encoder, (MLPAutoencoder, SequenceAutoencoder))
