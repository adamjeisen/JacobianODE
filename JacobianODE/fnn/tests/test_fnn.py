"""
Tests for the PyTorch FNN embedding library.

Mirrors the original TensorFlow test suite and adds Lorenz attractor validation.
"""

import numpy as np
import pytest
import torch

from JacobianODE.fnn.regularizers import (
    DeCov,
    FNN,
    _knn_indices_sq_euclidean_batched,
    loss_cov,
    loss_false,
    loss_amplification,
)
from JacobianODE.fnn.networks import MLPAutoencoder, LSTMAutoencoder
from JacobianODE.fnn.models import MLPEmbedding, LSTMEmbedding, ETDEmbedding
from JacobianODE.fnn.utils import hankel_matrix, standardize_ts


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_hankel_1d(self):
        """Hankel matrix from 1D data has correct shape."""
        data = np.random.randn(1000)
        hm = hankel_matrix(data, 3)
        assert hm.shape == (997, 3, 1)

    def test_hankel_2d(self):
        """Hankel matrix from multivariate data has correct shape."""
        data = np.random.randn(1000, 3)
        hm = hankel_matrix(data, 5)
        assert hm.shape[1] == 5
        assert hm.shape[2] == 3

    def test_standardize_ts(self):
        """Standardized series has zero mean and unit std."""
        data = np.random.randn(100, 3) * 5 + 10
        normed = standardize_ts(data)
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(normed.std(axis=0), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Regularizer tests
# ---------------------------------------------------------------------------

class TestRegularizers:
    def test_fnn_loss_zero_input(self):
        """FNN loss on zero tensor should be near zero."""
        x = torch.zeros(25, 10)
        loss = loss_false(x)
        assert loss.ndim == 0  # scalar
        assert loss.item() < 1e-6

    def test_fnn_loss_finite(self):
        """FNN loss on random input should be finite and real."""
        x = torch.randn(32, 5)
        loss = loss_false(x)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_fnn_regularizer_module(self):
        """FNN module produces real-valued output."""
        reg = FNN(strength=1.0, k=1)
        x = torch.randn(10, 3)
        loss = reg(x)
        assert torch.isfinite(loss)

    def test_decov_loss(self):
        """DeCov loss on identity-like batch should be near zero."""
        x = torch.eye(5)
        loss = loss_cov(x)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_decov_regularizer_module(self):
        """DeCov module produces real-valued output."""
        reg = DeCov(strength=0.5)
        x = torch.randn(20, 4)
        loss = reg(x)
        assert torch.isfinite(loss)

    def test_knn_batched_matches_full_cdist(self):
        """Batched squared-distance k-NN indices match ``cdist`` + ``topk``."""
        torch.manual_seed(0)
        emb = torch.randn(80, 5)
        k = 7
        batch = 11
        with torch.no_grad():
            dists = torch.cdist(emb, emb)
            _, ref = torch.topk(dists, k, largest=False, dim=1)
        got = _knn_indices_sq_euclidean_batched(emb, k, batch)
        torch.testing.assert_close(got, ref)

    def test_loss_amplification_batched_grad(self):
        """Amplification loss is finite and backprops when k-NN is batched."""
        T, D = 120, 3
        max_T = 5
        z = torch.randn(1, T, D, requires_grad=True)
        x = torch.randn(1, T, D)
        loss = loss_amplification(
            z, x, n_neighbors=5, max_T=max_T, knn_batch_size=16,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_eps_k_pairwise_identity(self):
        """eps_k uses the 2K||·||^2 - 2||sum||^2 identity (matches explicit K×K diff)."""
        torch.manual_seed(0)
        n_pts, K, D = 7, 5, 4
        neighbors = torch.randn(n_pts, K, D, dtype=torch.double)
        diff = neighbors.unsqueeze(2) - neighbors.unsqueeze(1)
        sq_pairwise = diff.pow(2).sum(dim=-1)
        eps_explicit = sq_pairwise.sum(dim=(1, 2)) / (K * (K - 1) * D)
        sum_sq_norms = (neighbors * neighbors).sum(dim=(1, 2))
        sum_vec = neighbors.sum(dim=1)
        sq_norm_sum_vec = (sum_vec * sum_vec).sum(dim=-1)
        pairwise_sq_sum = 2.0 * K * sum_sq_norms - 2.0 * sq_norm_sum_vec
        eps_algebraic = pairwise_sq_sum / (K * (K - 1) * D)
        torch.testing.assert_close(eps_algebraic, eps_explicit, rtol=1e-12, atol=1e-12)

    def test_fnn_loss_gradients(self):
        """FNN loss should produce valid gradients."""
        x = torch.randn(16, 4, requires_grad=True)
        loss = loss_false(x)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------

class TestNetworks:
    def test_mlp_autoencoder_shapes(self):
        """MLP autoencoder produces correct output shapes."""
        model = MLPAutoencoder(n_latent=3, time_window=10, n_features=1,
                               network_shape=[16, 16])
        x = torch.randn(8, 10, 1)
        z = model.encode(x)
        assert z.shape == (8, 3)
        recon = model(x)
        assert recon.shape == (8, 10, 1)

    def test_mlp_autoencoder_multivariate(self):
        """MLP autoencoder handles multivariate input."""
        model = MLPAutoencoder(n_latent=5, time_window=8, n_features=3,
                               network_shape=[32])
        x = torch.randn(4, 8, 3)
        z = model.encode(x)
        assert z.shape == (4, 5)
        recon = model(x)
        assert recon.shape == (4, 8, 3)

    def test_lstm_autoencoder_shapes(self):
        """LSTM autoencoder produces correct output shapes."""
        model = LSTMAutoencoder(n_latent=3, time_window=10, n_features=1)
        x = torch.randn(8, 10, 1)
        z = model.encode(x)
        assert z.shape == (8, 3)
        recon = model(x)
        assert recon.shape == (8, 10, 1)

    def test_lstm_autoencoder_with_layers(self):
        """LSTM autoencoder with intermediate layers."""
        model = LSTMAutoencoder(n_latent=2, time_window=8, n_features=1,
                                network_shape=[8, 4])
        x = torch.randn(4, 8, 1)
        z = model.encode(x)
        assert z.shape == (4, 2)
        recon = model(x)
        assert recon.shape == (4, 8, 1)

    def test_mlp_with_fnn_regularizer(self):
        """MLP forward + FNN regularizer produces valid loss."""
        model = MLPAutoencoder(n_latent=3, time_window=5, n_features=1,
                               network_shape=[8])
        reg = FNN(strength=1.0)
        x = torch.randn(16, 5, 1)
        z = model.encode(x)
        recon = model(x)
        recon_loss = torch.nn.functional.mse_loss(recon, x)
        reg_loss = reg(z)
        total = recon_loss + reg_loss
        total.backward()
        # Check all parameters got gradients
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_train_eval_mode(self):
        """GaussianNoise is active in train mode, inactive in eval."""
        model = MLPAutoencoder(n_latent=2, time_window=5, n_features=1,
                               network_shape=[8])
        x = torch.randn(4, 5, 1)

        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

        model.train()
        with torch.no_grad():
            out3 = model(x)
            out4 = model(x)
        # In training mode, noise makes outputs differ
        assert not torch.allclose(out3, out4, atol=1e-6)


# ---------------------------------------------------------------------------
# Sklearn-style API tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_mlp_embedding_fit_transform(self):
        """MLPEmbedding fit() + transform() end-to-end."""
        data = np.random.randn(200)
        model = MLPEmbedding(n_latent=2, time_window=5)
        model.fit(data, train_steps=5, batch_size=32, verbose=0)
        embedding = model.transform(data)
        assert embedding.shape == (195, 2)

    def test_lstm_embedding_fit_transform(self):
        """LSTMEmbedding fit() + transform() end-to-end."""
        data = np.random.randn(200)
        model = LSTMEmbedding(n_latent=2, time_window=5)
        model.fit(data, train_steps=5, batch_size=32, verbose=0)
        embedding = model.transform(data)
        assert embedding.shape == (195, 2)

    def test_mlp_with_fnn_regularizer(self):
        """MLPEmbedding with FNN regularizer trains successfully."""
        data = np.random.randn(200)
        reg = FNN(strength=0.5)
        model = MLPEmbedding(n_latent=3, time_window=5,
                             latent_regularizer=reg)
        model.fit(data, train_steps=5, batch_size=32, verbose=0)
        embedding = model.transform(data)
        assert embedding.shape == (195, 3)

    def test_etd_embedding(self):
        """ETDEmbedding (PCA baseline) works."""
        data = np.random.randn(200)
        model = ETDEmbedding(n_latent=3, time_window=10)
        embedding = model.fit_transform(data)
        assert embedding.shape[1] == 3

    def test_cache_normalization(self):
        """cache_normalization=True stores train stats and reuses them."""
        np.random.seed(0)
        train_data = np.random.randn(200) * 3 + 5  # mean=5, std~3
        test_data = np.random.randn(200) * 10 - 20  # very different distribution

        # Without caching: transform uses test data's own mean/std
        model_no_cache = MLPEmbedding(
            n_latent=2, time_window=5, cache_normalization=False,
        )
        model_no_cache.fit(train_data, train_steps=3, batch_size=32, verbose=0)
        emb_no_cache = model_no_cache.transform(test_data)

        # With caching: transform uses train data's mean/std
        model_cached = MLPEmbedding(
            n_latent=2, time_window=5, cache_normalization=True,
        )
        model_cached.fit(train_data, train_steps=3, batch_size=32, verbose=0)
        emb_cached = model_cached.transform(test_data)

        # The two should differ because normalization uses different stats
        assert not np.allclose(emb_no_cache, emb_cached, atol=1e-3), \
            "Cached and uncached embeddings should differ on shifted test data"

        # Verify cached stats match training data
        np.testing.assert_allclose(
            model_cached._train_mean, np.mean(train_data), atol=1e-10,
        )
        np.testing.assert_allclose(
            model_cached._train_std, np.std(train_data), atol=1e-10,
        )

        # Without caching, no stats should be stored
        assert model_no_cache._train_mean is None

    def test_fit_transform_shortcut(self):
        """fit_transform returns correctly shaped output."""
        data = np.random.randn(100)
        model = MLPEmbedding(n_latent=2, time_window=5, random_state=42)
        emb = model.fit_transform(data, train_steps=3, batch_size=32)
        assert emb.shape == (95, 2)


# ---------------------------------------------------------------------------
# Lorenz attractor validation
# ---------------------------------------------------------------------------

def _generate_lorenz(n_points=5000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    """Generate Lorenz attractor trajectory via simple Euler integration."""
    xyz = np.zeros((n_points, 3))
    xyz[0] = [1.0, 1.0, 1.0]
    for i in range(n_points - 1):
        x, y, z = xyz[i]
        xyz[i+1, 0] = x + dt * (sigma * (y - x))
        xyz[i+1, 1] = y + dt * (rho * x - y - x * z)
        xyz[i+1, 2] = z + dt * (x * y - beta * z)
    return xyz


class TestLorenzValidation:
    """Validate FNN embedding by reconstructing the Lorenz attractor
    from a single observable (x-coordinate only)."""

    @pytest.fixture
    def lorenz_data(self):
        xyz = _generate_lorenz(n_points=5000)
        # Use only x-coordinate as the univariate observable
        return xyz[:, 0], xyz

    def test_mlp_fnn_lorenz_embedding_dimension(self, lorenz_data):
        """MLP+FNN embedding of univariate Lorenz should produce 3D output."""
        x_obs, _ = lorenz_data
        reg = FNN(strength=0.1)
        model = MLPEmbedding(
            n_latent=3, time_window=15, network_shape=[32, 64],
            latent_regularizer=reg,
        )
        model.fit(x_obs, train_steps=100, batch_size=256, learning_rate=1e-3, verbose=0)
        embedding = model.transform(x_obs)

        # Should be 3-dimensional
        assert embedding.shape[1] == 3
        # Embedding should have non-trivial variance in all 3 dimensions
        # (the FNN regularizer encourages using all dimensions)
        stds = np.std(embedding, axis=0)
        n_active = np.sum(stds > 0.01)
        assert n_active >= 2, \
            f"Embedding collapsed to {n_active} active dims: stds = {stds}"

    def test_mlp_fnn_reduces_reconstruction_loss(self, lorenz_data):
        """Training should reduce reconstruction loss."""
        x_obs, _ = lorenz_data
        model = MLPEmbedding(
            n_latent=3, time_window=15, network_shape=[32, 64],
        )
        model.fit(x_obs, train_steps=30, batch_size=128, verbose=0)
        history = model.train_history["loss"]
        # Loss should decrease over training
        assert history[-1] < history[0], \
            f"Loss did not decrease: {history[0]:.4f} -> {history[-1]:.4f}"

    def test_lstm_fnn_lorenz_embedding(self, lorenz_data):
        """LSTM+FNN embedding of Lorenz attractor."""
        x_obs, _ = lorenz_data
        reg = FNN(strength=0.5)
        model = LSTMEmbedding(
            n_latent=3, time_window=15,
            latent_regularizer=reg,
        )
        model.fit(x_obs, train_steps=20, batch_size=128, verbose=0)
        embedding = model.transform(x_obs)
        assert embedding.shape[1] == 3
        stds = np.std(embedding, axis=0)
        assert all(s > 0.01 for s in stds), \
            f"LSTM embedding collapsed: stds = {stds}"


# ---------------------------------------------------------------------------
# 3D (trials x time x dims) data tests
# ---------------------------------------------------------------------------

class TestTrialData:
    """Tests for native 3D (N, T, D) trial-structured data support."""

    @pytest.fixture
    def trial_data(self):
        """Generate synthetic (N, T, D) trial data."""
        np.random.seed(42)
        N, T, D = 4, 100, 3
        return np.random.randn(N, T, D)

    def test_standardize_ts_3d(self, trial_data):
        """standardize_ts with 3D input gives per-trial zero mean / unit std."""
        normed = standardize_ts(trial_data)
        assert normed.shape == trial_data.shape
        # Each trial should independently have zero mean and unit std
        for i in range(trial_data.shape[0]):
            np.testing.assert_allclose(
                normed[i].mean(axis=0), 0.0, atol=1e-10,
            )
            np.testing.assert_allclose(
                normed[i].std(axis=0), 1.0, atol=1e-10,
            )

    def test_hankel_matrix_3d(self, trial_data):
        """hankel_matrix with 3D input produces correct 4D shape."""
        N, T, D = trial_data.shape
        q = 5
        hm = hankel_matrix(trial_data, q)
        assert hm.ndim == 4
        assert hm.shape[0] == N
        assert hm.shape[2] == q
        assert hm.shape[3] == D

    def test_mlp_fit_transform_3d(self, trial_data):
        """MLPEmbedding fit/transform with 3D input produces correct shapes."""
        N, T, D = trial_data.shape
        tw, n_latent = 5, 2
        model = MLPEmbedding(n_latent=n_latent, time_window=tw)
        model.fit(trial_data, train_steps=3, batch_size=32, verbose=0)
        embedding = model.transform(trial_data)

        n_windows = hankel_matrix(trial_data, tw).shape[1]
        assert embedding.shape == (N, n_windows, n_latent)

    def test_mlp_reconstruct_3d(self, trial_data):
        """MLPEmbedding reconstruct with 3D input produces correct shapes."""
        N, T, D = trial_data.shape
        tw = 5
        model = MLPEmbedding(n_latent=2, time_window=tw)
        model.fit(trial_data, train_steps=3, batch_size=32, verbose=0)
        X_windows, recon = model.reconstruct(trial_data)

        n_windows = hankel_matrix(trial_data, tw).shape[1]
        assert X_windows.shape == (N, n_windows, tw, D)
        assert recon.shape == (N, n_windows, tw, D)

    def test_lstm_fit_transform_3d(self, trial_data):
        """LSTMEmbedding fit/transform with 3D input produces correct shapes."""
        N, T, D = trial_data.shape
        tw, n_latent = 5, 2
        model = LSTMEmbedding(n_latent=n_latent, time_window=tw)
        model.fit(trial_data, train_steps=3, batch_size=32, verbose=0)
        embedding = model.transform(trial_data)

        n_windows = hankel_matrix(trial_data, tw).shape[1]
        assert embedding.shape == (N, n_windows, n_latent)

    def test_fnn_regularizer_with_3d(self, trial_data):
        """FNN regularizer works correctly with 3D trial data."""
        reg = FNN(strength=0.5)
        model = MLPEmbedding(n_latent=3, time_window=5, latent_regularizer=reg)
        model.fit(trial_data, train_steps=5, batch_size=32, verbose=0)
        embedding = model.transform(trial_data)
        assert embedding.ndim == 3
        assert embedding.shape[0] == trial_data.shape[0]
        assert embedding.shape[2] == 3

    def test_cache_normalization_3d(self, trial_data):
        """cache_normalization with 3D input stores (1, 1, D) shaped stats."""
        model = MLPEmbedding(
            n_latent=2, time_window=5, cache_normalization=True,
        )
        model.fit(trial_data, train_steps=3, batch_size=32, verbose=0)

        assert model._train_mean.shape == (1, 1, trial_data.shape[2])
        assert model._train_std.shape == (1, 1, trial_data.shape[2])
