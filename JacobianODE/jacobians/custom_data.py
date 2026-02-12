"""
Custom data loading utilities for JacobianODE.

This module provides tools for loading custom time series data in the format
(trials x time x dimensions) for use with JacobianODE training.

Example usage:
    # CLI - From a numpy file:
    python run_jacobians.py data=custom \\
        data.dataset_loader._target_=JacobianODE.jacobians.custom_data.load_from_numpy \\
        +data.dataset_loader.file_path=/path/to/data.npy \\
        +data.dataset_loader.dt=0.01 \\
        data.flow.dim=YOUR_DIM

    # CLI - From a pickle file:
    python run_jacobians.py data=custom \\
        data.dataset_loader._target_=JacobianODE.jacobians.custom_data.load_from_pickle \\
        +data.dataset_loader.file_path=/path/to/data.pkl \\
        +data.dataset_loader.dt=0.01 \\
        data.flow.dim=YOUR_DIM

    # Python - In-memory array (RECOMMENDED for notebooks):
    from JacobianODE.jacobians import load_config, initialize_config, make_trajectories

    my_data = np.random.randn(32, 1000, 3)  # (trials, time, dims)
    dt = 0.01

    cfg = load_config(overrides=["data=custom"])
    cfg = initialize_config(cfg, data_dim=my_data.shape[-1])
    eq, sol, dt = make_trajectories(cfg, data=my_data, dt=dt)

    # Python - File-based loading from a TimeSeriesData .npz file (dt is read from file):
    cfg = load_config(
        custom_dataset_loader="JacobianODE.jacobians.custom_data.load_timeseries_data",
        custom_dataset_loader_kwargs={"file_path": "/path/to/data.npz"},
        data_dim=50
    )
"""

import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def validate_data_shape(data: np.ndarray, name: str = "data") -> None:
    """Validate that data has the expected shape (trials x time x dimensions).

    Args:
        data: Input array to validate
        name: Name to use in error messages

    Raises:
        ValueError: If data does not have exactly 3 dimensions or has invalid shape
    """
    if data.ndim != 3:
        raise ValueError(
            f"{name} must be 3-dimensional (trials x time x dimensions), "
            f"got shape {data.shape} with {data.ndim} dimensions"
        )

    n_trials, n_time, n_dims = data.shape

    if n_trials < 1:
        raise ValueError(f"{name} must have at least 1 trial, got {n_trials}")
    if n_time < 2:
        raise ValueError(f"{name} must have at least 2 time points, got {n_time}")
    if n_dims < 1:
        raise ValueError(f"{name} must have at least 1 dimension, got {n_dims}")


class CustomDatasetLoader:
    """Loader for custom time series data.

    This class provides a flexible interface for loading custom data that can be
    used with the JacobianODE training pipeline. Data should be in the format
    (trials x time x dimensions).

    Args:
        data: The time series data array of shape (trials, time, dimensions).
            Can be a numpy array or path to a file.
        dt: The time step between observations. Required.
        file_path: Path to a data file (numpy .npy, .npz, or pickle .pkl).
            Mutually exclusive with `data` parameter.
        data_key: If loading from .npz or pickle dict, the key to extract data from.
            Defaults to 'values' or 'data'.

    Example:
        # Direct array input:
        loader = CustomDatasetLoader(data=my_array, dt=0.01)
        sol, dt = loader()

        # From file:
        loader = CustomDatasetLoader(file_path="data.npy", dt=0.01)
        sol, dt = loader()
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        file_path: Optional[str] = None,
        data_key: Optional[str] = None,
    ):
        if data is None and file_path is None:
            raise ValueError("Must provide either 'data' or 'file_path'")
        if data is not None and file_path is not None:
            raise ValueError("Cannot provide both 'data' and 'file_path'")
        if dt is None:
            raise ValueError("Must provide 'dt' (time step between observations)")

        self.dt = float(dt)
        self.data_key = data_key

        if data is not None:
            self._data = np.asarray(data)
        else:
            self._data = None
            self.file_path = Path(file_path)
            if not self.file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

    def _load_from_file(self) -> np.ndarray:
        """Load data from file."""
        suffix = self.file_path.suffix.lower()

        if suffix == '.npy':
            return np.load(self.file_path)

        elif suffix == '.npz':
            npz_data = np.load(self.file_path)
            keys = list(npz_data.keys())

            if self.data_key is not None:
                if self.data_key not in keys:
                    raise KeyError(f"Key '{self.data_key}' not found in npz file. Available keys: {keys}")
                return npz_data[self.data_key]

            # Try common key names
            for key in ['values', 'data', 'trajectories', 'x', 'X']:
                if key in keys:
                    return npz_data[key]

            # Fall back to first key
            if len(keys) == 1:
                return npz_data[keys[0]]

            raise KeyError(
                f"Could not find data in npz file. Available keys: {keys}. "
                f"Please specify 'data_key' parameter."
            )

        elif suffix in ['.pkl', '.pickle']:
            with open(self.file_path, 'rb') as f:
                pkl_data = pickle.load(f)

            if isinstance(pkl_data, np.ndarray):
                return pkl_data

            if isinstance(pkl_data, dict):
                if self.data_key is not None:
                    if self.data_key not in pkl_data:
                        raise KeyError(f"Key '{self.data_key}' not found in pickle file. Available keys: {list(pkl_data.keys())}")
                    return np.asarray(pkl_data[self.data_key])

                # Try common key names
                for key in ['values', 'data', 'trajectories', 'x', 'X']:
                    if key in pkl_data:
                        return np.asarray(pkl_data[key])

                raise KeyError(
                    f"Could not find data in pickle file. Available keys: {list(pkl_data.keys())}. "
                    f"Please specify 'data_key' parameter."
                )

            raise TypeError(f"Pickle file must contain numpy array or dict, got {type(pkl_data)}")

        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: .npy, .npz, .pkl, .pickle")

    def __call__(self) -> Tuple[Dict[str, np.ndarray], float]:
        """Load and return the data.

        Returns:
            Tuple of (sol, dt) where:
                - sol: Dictionary with 'values' key containing data of shape (trials, time, dims)
                - dt: Time step between observations
        """
        if self._data is None:
            data = self._load_from_file()
        else:
            data = self._data

        validate_data_shape(data, "Custom data")

        sol = {'values': data}
        return sol, self.dt


def load_from_numpy(
    file_path: str,
    dt: float,
    data_key: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Load custom data from a numpy file (.npy or .npz).

    This is a convenience function for use with Hydra configuration.

    Args:
        file_path: Path to the numpy file
        dt: Time step between observations
        data_key: For .npz files, the key to extract data from

    Returns:
        Tuple of (sol, dt) where sol contains 'values' key with the data

    Example config override:
        data.dataset_loader._target_=JacobianODE.jacobians.custom_data.load_from_numpy
        +data.dataset_loader.file_path=/path/to/data.npy
        +data.dataset_loader.dt=0.01
    """
    loader = CustomDatasetLoader(file_path=file_path, dt=dt, data_key=data_key)
    return loader()


def load_timeseries_data(
    file_path: str,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Load custom data from a TimeSeriesData .npz file.

    This loads files saved by ``TimeSeriesData.save()``, which bundle
    ``values``, ``dt``, and ``metadata`` into a single .npz file.
    Since ``dt`` is stored in the file, it does not need to be passed
    separately.

    Args:
        file_path: Path to the .npz file saved by TimeSeriesData.save()

    Returns:
        Tuple of (sol, dt) where sol contains 'values' key with the data

    Example config override:
        data.dataset_loader._target_=JacobianODE.jacobians.custom_data.load_timeseries_data
        +data.dataset_loader.file_path=/path/to/data.npz
    """
    from .data.types import TimeSeriesData

    ts_data = TimeSeriesData.load(file_path)
    validate_data_shape(ts_data.values, "TimeSeriesData")
    sol = {"values": ts_data.values}
    return sol, ts_data.dt


def load_from_pickle(
    file_path: str,
    dt: float,
    data_key: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Load custom data from a pickle file.

    This is a convenience function for use with Hydra configuration.

    Args:
        file_path: Path to the pickle file
        dt: Time step between observations
        data_key: If the pickle contains a dict, the key to extract data from

    Returns:
        Tuple of (sol, dt) where sol contains 'values' key with the data

    Example config override:
        data.dataset_loader._target_=JacobianODE.jacobians.custom_data.load_from_pickle
        +data.dataset_loader.file_path=/path/to/data.pkl
        +data.dataset_loader.dt=0.01
    """
    loader = CustomDatasetLoader(file_path=file_path, dt=dt, data_key=data_key)
    return loader()


def load_from_array(
    data: np.ndarray,
    dt: float,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Load custom data from a numpy array.

    This is useful for programmatic usage when you already have data in memory.

    Args:
        data: Numpy array of shape (trials, time, dimensions)
        dt: Time step between observations

    Returns:
        Tuple of (sol, dt) where sol contains 'values' key with the data

    Example:
        from JacobianODE.jacobians.custom_data import load_config,load_from_array

        # Your data: shape (n_trials, n_timepoints, n_dimensions)
        my_data = np.random.randn(32, 1000, 50)

        # Create a loader function
        def my_loader():
            return load_from_array(my_data, dt=0.01)

        # Load config with custom loader
        cfg = load_config(
            custom_dataset_loader="__main__.my_loader",
            overrides=["data.flow.dim=50"]
        )
    """
    loader = CustomDatasetLoader(data=data, dt=dt)
    return loader()


def create_custom_loader(
    data: np.ndarray,
    dt: float,
) -> callable:
    """Create a callable loader for custom data.

    This is useful when you want to create a loader function that can be
    passed to load_config().

    Args:
        data: Numpy array of shape (trials, time, dimensions)
        dt: Time step between observations

    Returns:
        A callable that returns (sol, dt) when called

    Example:
        from JacobianODE.jacobians.custom_data import create_custom_loader
        from JacobianODE.jacobians import load_config

        my_data = np.random.randn(32, 1000, 50)
        loader = create_custom_loader(my_data, dt=0.01)

        # The loader can now be registered and used with Hydra
    """
    return CustomDatasetLoader(data=data, dt=dt)
