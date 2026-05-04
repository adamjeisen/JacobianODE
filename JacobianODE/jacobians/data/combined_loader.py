"""Combined-source dataset loader for multi-condition training.

Concatenates trajectories from N sub-loaders (each producing (eq, sol, dt))
into a single (eq, sol, dt) triple where ``sol`` carries the standard
``values`` plus a per-trajectory ``condition`` tensor and a per-trajectory
``source_id`` integer. Downstream code uses ``condition`` to thread a
conditioning vector through the encoder + dynamics MLP, and uses
``source_id`` to split each source's trajectories independently so the
condition mix is balanced across train / val / test.

Use case: training one JacobianODE model on trajectories from multiple
variants of the same underlying system (e.g., the same WMTask network at
two different training checkpoints) with a per-batch condition tag
distinguishing them. Normalization (postprocess_data) and noise-scale
calibration are computed on the concatenated data so the two conditions
share the same observable scale.

Hydra usage (with _partial_ for inner loaders):

    data:
      data_type: wmtask
      dataset_loader:
        _target_: JacobianODE.jacobians.data.combined_loader.load_combined_for_jacobianode
        sources:
          - loader:
              _target_: wmtask.trajectories.load_wmtask_for_jacobianode
              _partial_: true
              project: ...
              name: ...
              model_to_load: 1
            condition: [-1.0]
          - loader:
              _target_: wmtask.trajectories.load_wmtask_for_jacobianode
              _partial_: true
              project: ...
              name: ...
              model_to_load: final
            condition: [1.0]
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def load_combined_for_jacobianode(
    sources: Sequence[dict],
    dt_atol: float = 1e-9,
) -> tuple[Any, dict, float]:
    """Load and concatenate trajectories from multiple sub-loaders.

    Each entry in ``sources`` is a dict with two keys:

    - ``loader`` : zero-arg callable returning ``(eq, sol, dt)`` (typically
      a ``functools.partial`` produced by Hydra ``_partial_: true``).
    - ``condition`` : sequence of floats, the condition vector that will be
      attached to every trajectory loaded by this source. All sources must
      have the same condition_dim.

    Args:
        sources: list of source dicts (see above). Order matters: source_id
            for source ``i`` is just ``i``.
        dt_atol: absolute tolerance for the dt-mismatch check.

    Returns:
        ``(eq, sol, dt)`` where ``sol`` is a dict with keys:

        - ``values`` : concatenated trajectories, shape ``(n_total, T, D)``.
        - ``condition`` : per-trajectory condition, shape
          ``(n_total, condition_dim)``, dtype float32.
        - ``source_id`` : per-trajectory source index in ``[0, n_sources)``,
          shape ``(n_total,)``, dtype int64. Downstream splitting groups
          on this so each source is split independently into train/val/test
          (= conditions balanced across splits).
        - ``source_eqs`` : list of the per-source ``eq`` objects.

        ``eq`` is the first source's ``eq`` (the combined dataset has no
        single ground-truth dynamics; analytical Jacobian comparisons need
        ``source_eqs`` instead).

    Raises:
        ValueError: if no sources are given, condition_dim differs across
            sources, dt differs across sources by more than ``dt_atol``, or
            per-trajectory shape ``(T, D)`` differs across sources.
    """
    if not sources:
        raise ValueError("load_combined_for_jacobianode: sources is empty")

    all_values: list[np.ndarray] = []
    all_conditions: list[np.ndarray] = []
    all_source_ids: list[np.ndarray] = []
    all_eq: list[Any] = []

    dt_ref: float | None = None
    shape_ref: tuple | None = None
    cond_dim_ref: int | None = None

    for src_idx, src in enumerate(sources):
        loader = src["loader"]
        cond_value = np.asarray(src["condition"], dtype=np.float32)
        if cond_value.ndim != 1:
            raise ValueError(
                f"source {src_idx}: condition must be 1-D, got shape "
                f"{cond_value.shape}"
            )
        if cond_dim_ref is None:
            cond_dim_ref = cond_value.shape[0]
        elif cond_value.shape[0] != cond_dim_ref:
            raise ValueError(
                f"source {src_idx}: condition_dim={cond_value.shape[0]} "
                f"differs from source 0's condition_dim={cond_dim_ref}"
            )

        out = loader()
        if len(out) == 2:
            sol, dt = out
            eq = None
        elif len(out) == 3:
            eq, sol, dt = out
        else:
            raise ValueError(
                f"source {src_idx}: loader returned {len(out)}-tuple, "
                "expected (sol, dt) or (eq, sol, dt)"
            )

        values = np.asarray(sol["values"])
        if values.ndim != 3:
            raise ValueError(
                f"source {src_idx}: values must be 3-D (n_traj, T, D), "
                f"got shape {values.shape}"
            )
        per_traj_shape = values.shape[1:]
        if shape_ref is None:
            shape_ref = per_traj_shape
        elif per_traj_shape != shape_ref:
            raise ValueError(
                f"source {src_idx}: per-trajectory shape {per_traj_shape} "
                f"differs from source 0's shape {shape_ref}"
            )

        if dt_ref is None:
            dt_ref = float(dt)
        elif abs(float(dt) - dt_ref) > dt_atol:
            raise ValueError(
                f"source {src_idx}: dt={dt} differs from source 0's "
                f"dt={dt_ref} by more than atol={dt_atol}"
            )

        n_traj = values.shape[0]
        cond_tensor = np.tile(cond_value[None, :], (n_traj, 1))
        source_id = np.full((n_traj,), src_idx, dtype=np.int64)

        all_values.append(values)
        all_conditions.append(cond_tensor)
        all_source_ids.append(source_id)
        all_eq.append(eq)

    sol_combined = {
        "values": np.concatenate(all_values, axis=0),
        "condition": np.concatenate(all_conditions, axis=0),
        "source_id": np.concatenate(all_source_ids, axis=0),
        "source_eqs": all_eq,
    }
    return all_eq[0], sol_combined, dt_ref
