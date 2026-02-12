"""Time series data container for JacobianODE."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from ..custom_data import validate_data_shape


@dataclass
class TimeSeriesData:
    """Lightweight container for time series data.

    Bundles values, time step, and optional metadata into a single typed object
    with save/load support for reproducibility.

    Args:
        values: Array of shape ``(trials, time, dims)``.
        dt: Time step between observations.
        metadata: Free-form metadata dict. Conventional keys:
            ``source`` – where the data came from (e.g. ``"lorenz"``, ``"custom"``),
            ``noise_level`` – noise added during postprocessing,
            ``random_seed`` – seed used for trajectory generation,
            ``config`` – snapshot of the Hydra config used.

    Example:
        >>> import numpy as np
        >>> from JacobianODE.jacobians.data.types import TimeSeriesData
        >>> ts = TimeSeriesData(
        ...     values=np.random.randn(4, 100, 3),
        ...     dt=0.01,
        ...     metadata={"source": "lorenz"},
        ... )
        >>> ts.shape
        (4, 100, 3)
    """

    values: np.ndarray
    dt: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.dt = float(self.dt)
        validate_data_shape(self.values, "TimeSeriesData.values")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the values array ``(trials, time, dims)``."""
        return self.values.shape

    @property
    def n_trials(self) -> int:
        return self.values.shape[0]

    @property
    def n_timepoints(self) -> int:
        return self.values.shape[1]

    @property
    def n_dims(self) -> int:
        return self.values.shape[2]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save to an ``.npz`` file.

        The file stores ``values``, ``dt`` (as a 0-d array), and ``metadata``
        (JSON-encoded string).
        """
        np.savez(
            path,
            values=self.values,
            dt=np.array(self.dt),
            metadata=json.dumps(self.metadata),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TimeSeriesData":
        """Load a ``TimeSeriesData`` from a ``.npz`` file written by :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        return cls(
            values=data["values"],
            dt=float(data["dt"]),
            metadata=json.loads(str(data["metadata"])),
        )
