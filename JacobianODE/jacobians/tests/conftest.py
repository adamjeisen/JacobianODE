"""Pytest fixtures for JacobianODE tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_trajectory_data() -> np.ndarray:
    """Generate sample trajectory data for testing.

    Returns:
        Numpy array of shape (n_trials, n_time, n_dim) = (4, 100, 3)
    """
    np.random.seed(42)
    n_trials = 4
    n_time = 100
    n_dim = 3

    # Generate smooth trajectories (simple oscillations)
    t = np.linspace(0, 10, n_time)
    data = np.zeros((n_trials, n_time, n_dim))

    for i in range(n_trials):
        freq = 0.5 + 0.1 * i
        phase = i * np.pi / 4
        data[i, :, 0] = np.sin(freq * t + phase)
        data[i, :, 1] = np.cos(freq * t + phase)
        data[i, :, 2] = np.sin(2 * freq * t + phase)

    return data


@pytest.fixture
def sample_trajectory_tensor(sample_trajectory_data: np.ndarray) -> torch.Tensor:
    """Convert sample trajectory data to PyTorch tensor.

    Args:
        sample_trajectory_data: Numpy array of trajectory data.

    Returns:
        PyTorch tensor of the same data.
    """
    return torch.from_numpy(sample_trajectory_data).float()


@pytest.fixture
def sample_dt() -> float:
    """Return a sample time step.

    Returns:
        Time step value of 0.1.
    """
    return 0.1


@pytest.fixture
def temp_data_file(tmp_path, sample_trajectory_data):
    """Create a temporary numpy file with sample data.

    Args:
        tmp_path: Pytest fixture for temporary directory.
        sample_trajectory_data: Sample trajectory data.

    Returns:
        Path to the temporary file.
    """
    file_path = tmp_path / "test_data.npy"
    np.save(file_path, sample_trajectory_data)
    return file_path


@pytest.fixture
def temp_npz_file(tmp_path, sample_trajectory_data):
    """Create a temporary npz file with sample data.

    Args:
        tmp_path: Pytest fixture for temporary directory.
        sample_trajectory_data: Sample trajectory data.

    Returns:
        Path to the temporary file.
    """
    file_path = tmp_path / "test_data.npz"
    np.savez(file_path, values=sample_trajectory_data, other_data=np.zeros(10))
    return file_path


@pytest.fixture
def temp_pickle_file(tmp_path, sample_trajectory_data):
    """Create a temporary pickle file with sample data.

    Args:
        tmp_path: Pytest fixture for temporary directory.
        sample_trajectory_data: Sample trajectory data.

    Returns:
        Path to the temporary file.
    """
    import pickle

    file_path = tmp_path / "test_data.pkl"
    with open(file_path, "wb") as f:
        pickle.dump({"values": sample_trajectory_data, "dt": 0.1}, f)
    return file_path
