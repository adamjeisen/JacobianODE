"""Type definitions and utilities for JacobianODE."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig


# Type aliases for common data types
ArrayLike = Union[np.ndarray, torch.Tensor]
TrajectoryData = NDArray[np.floating]
JacobianMatrix = Union[NDArray[np.floating], torch.Tensor]

# Configuration type
Config = DictConfig

# Model types
T = TypeVar("T")


@runtime_checkable
class JacobianModel(Protocol):
    """Protocol for models that compute Jacobians."""

    def compute_jacobians(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian matrices at given points.

        Args:
            x: Input tensor of shape (batch, time, dim) or (time, dim)

        Returns:
            Jacobian matrices of shape (..., dim, dim)
        """
        ...


@runtime_checkable
class DynamicalSystem(Protocol):
    """Protocol for dynamical system objects."""

    def jac(
        self, x: ArrayLike, t: ArrayLike, *args: Any, **kwargs: Any
    ) -> ArrayLike:
        """Compute the Jacobian of the system at given points.

        Args:
            x: State variables
            t: Time points
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Jacobian matrices
        """
        ...

    def make_trajectory(self, **kwargs: Any) -> Dict[str, Any]:
        """Generate trajectory data.

        Args:
            **kwargs: Trajectory generation parameters

        Returns:
            Dictionary containing trajectory data with 'values' key
        """
        ...


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for custom data loaders."""

    def __call__(self) -> Tuple[Dict[str, np.ndarray], float]:
        """Load and return data.

        Returns:
            Tuple of (sol, dt) where sol contains 'values' key with trajectory data
        """
        ...


def in_ipython() -> bool:
    """Check if running in an IPython/Jupyter environment.

    Returns:
        True if running in IPython/Jupyter, False otherwise.

    Example:
        >>> if in_ipython():
        ...     # Use notebook-specific settings
        ...     strategy = 'ddp_notebook'
    """
    try:
        # get_ipython is injected into the namespace by IPython
        get_ipython  # type: ignore[name-defined]
        return True
    except NameError:
        return False
