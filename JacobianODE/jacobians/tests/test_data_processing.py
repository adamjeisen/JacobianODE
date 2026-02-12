"""Tests for data processing utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..data.processing import normalize_data, denormalize_data


class TestNormalizeData:
    """Tests for normalize_data function."""

    def test_normalize_numpy_array(self, sample_trajectory_data):
        """Test normalizing a numpy array."""
        normalized, mu, sigma = normalize_data(sample_trajectory_data)

        # Check that normalization produces approximately zero mean and unit std
        assert abs(normalized.mean()) < 1e-6
        assert abs(normalized.std() - 1.0) < 1e-6

        # Check that mu and sigma are correct
        assert abs(mu - sample_trajectory_data.mean()) < 1e-6
        assert abs(sigma - sample_trajectory_data.std()) < 1e-6

    def test_normalize_torch_tensor(self, sample_trajectory_tensor):
        """Test normalizing a PyTorch tensor."""
        normalized, mu, sigma = normalize_data(sample_trajectory_tensor)

        # Check that output is still a tensor
        assert isinstance(normalized, torch.Tensor)

        # Check that normalization produces approximately zero mean and unit std
        assert abs(normalized.mean().item()) < 1e-6
        assert abs(normalized.std().item() - 1.0) < 1e-6

    def test_normalize_returns_float_stats(self, sample_trajectory_data):
        """Test that mu and sigma are Python floats."""
        _, mu, sigma = normalize_data(sample_trajectory_data)

        assert isinstance(mu, float)
        assert isinstance(sigma, float)

    def test_normalize_preserves_shape(self, sample_trajectory_data):
        """Test that normalization preserves data shape."""
        normalized, _, _ = normalize_data(sample_trajectory_data)

        assert normalized.shape == sample_trajectory_data.shape


class TestDenormalizeData:
    """Tests for denormalize_data function."""

    def test_denormalize_recovers_original(self, sample_trajectory_data):
        """Test that denormalization recovers original data."""
        normalized, mu, sigma = normalize_data(sample_trajectory_data)
        recovered = denormalize_data(normalized, mu, sigma)

        np.testing.assert_allclose(recovered, sample_trajectory_data, rtol=1e-6, atol=1e-15)

    def test_denormalize_torch_tensor(self, sample_trajectory_tensor):
        """Test denormalizing a PyTorch tensor."""
        normalized, mu, sigma = normalize_data(sample_trajectory_tensor)
        recovered = denormalize_data(normalized, mu, sigma)

        assert isinstance(recovered, torch.Tensor)
        torch.testing.assert_close(recovered, sample_trajectory_tensor)


class TestCustomDataValidation:
    """Tests for custom data validation."""

    def test_validate_correct_shape(self):
        """Test that valid data passes validation."""
        from ..custom_data import validate_data_shape

        data = np.random.randn(10, 100, 5)
        # Should not raise
        validate_data_shape(data, "test data")

    def test_validate_wrong_dimensions(self):
        """Test that wrong number of dimensions raises error."""
        from ..custom_data import validate_data_shape

        data_2d = np.random.randn(10, 100)
        with pytest.raises(ValueError, match="3-dimensional"):
            validate_data_shape(data_2d, "test data")

        data_4d = np.random.randn(10, 100, 5, 2)
        with pytest.raises(ValueError, match="3-dimensional"):
            validate_data_shape(data_4d, "test data")

    def test_validate_insufficient_trials(self):
        """Test that zero trials raises error."""
        from ..custom_data import validate_data_shape

        data = np.random.randn(0, 100, 5)
        with pytest.raises(ValueError, match="at least 1 trial"):
            validate_data_shape(data, "test data")

    def test_validate_insufficient_time_points(self):
        """Test that less than 2 time points raises error."""
        from ..custom_data import validate_data_shape

        data = np.random.randn(10, 1, 5)
        with pytest.raises(ValueError, match="at least 2 time points"):
            validate_data_shape(data, "test data")


class TestCustomDatasetLoader:
    """Tests for CustomDatasetLoader class."""

    def test_load_from_numpy(self, temp_data_file, sample_dt):
        """Test loading data from a numpy file."""
        from ..custom_data import CustomDatasetLoader

        loader = CustomDatasetLoader(file_path=str(temp_data_file), dt=sample_dt)
        sol, dt = loader()

        assert "values" in sol
        assert sol["values"].shape == (4, 100, 3)
        assert dt == sample_dt

    def test_load_from_npz(self, temp_npz_file, sample_dt):
        """Test loading data from a npz file."""
        from ..custom_data import CustomDatasetLoader

        loader = CustomDatasetLoader(file_path=str(temp_npz_file), dt=sample_dt)
        sol, dt = loader()

        assert "values" in sol
        assert sol["values"].shape == (4, 100, 3)

    def test_load_from_pickle(self, temp_pickle_file, sample_dt):
        """Test loading data from a pickle file."""
        from ..custom_data import CustomDatasetLoader

        loader = CustomDatasetLoader(file_path=str(temp_pickle_file), dt=sample_dt)
        sol, dt = loader()

        assert "values" in sol
        assert sol["values"].shape == (4, 100, 3)

    def test_load_from_array(self, sample_trajectory_data, sample_dt):
        """Test loading data directly from an array."""
        from ..custom_data import CustomDatasetLoader

        loader = CustomDatasetLoader(data=sample_trajectory_data, dt=sample_dt)
        sol, dt = loader()

        assert "values" in sol
        np.testing.assert_array_equal(sol["values"], sample_trajectory_data)
        assert dt == sample_dt

    def test_missing_dt_raises(self, sample_trajectory_data):
        """Test that missing dt raises error."""
        from ..custom_data import CustomDatasetLoader

        with pytest.raises(ValueError, match="Must provide 'dt'"):
            CustomDatasetLoader(data=sample_trajectory_data, dt=None)

    def test_missing_data_raises(self, sample_dt):
        """Test that missing data and file_path raises error."""
        from ..custom_data import CustomDatasetLoader

        with pytest.raises(ValueError, match="Must provide either"):
            CustomDatasetLoader(dt=sample_dt)

    def test_both_data_and_file_raises(self, sample_trajectory_data, temp_data_file, sample_dt):
        """Test that providing both data and file_path raises error."""
        from ..custom_data import CustomDatasetLoader

        with pytest.raises(ValueError, match="Cannot provide both"):
            CustomDatasetLoader(
                data=sample_trajectory_data,
                file_path=str(temp_data_file),
                dt=sample_dt,
            )


