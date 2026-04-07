"""Ranking strategies for selecting the best model from a pool of survivors.

Each function takes a list of survivor indices and the full candidate list,
and returns the index of the best model.

Available methods:
  - ``best_traj_loss``: Lowest trajectory validation loss (original default).
  - ``pareto_knee``: Knee of the Pareto front in (loop_closure_loss, traj_loss)
    space, detected via maximum curvature in log-log space.
  - ``geo_rank``: Lowest geometric mean of ordinal ranks on traj_loss and
    loop_closure_loss.
  - ``minimax_rank``: Lowest max (worst-case) ordinal rank.
  - ``geo_log_score``: Geometric mean of log-normalized scores (preserves
    relative magnitudes, unlike ordinal ranks).
  - ``minimax_log_score``: Max of log-normalized scores (most balanced).
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Sequence

import numpy as np

from .criteria import DiagnosticMetrics

RankingMethod = Literal[
    "best_traj_loss",
    "pareto_knee",
    "geo_rank",
    "minimax_rank",
    "geo_log_score",
    "minimax_log_score",
]

ALL_RANKING_METHODS: list[RankingMethod] = [
    "best_traj_loss",
    "pareto_knee",
    "geo_rank",
    "minimax_rank",
    "geo_log_score",
    "minimax_log_score",
]


def rank_survivors(
    survivors: List[int],
    candidates: Sequence[DiagnosticMetrics],
    method: RankingMethod = "pareto_knee",
) -> int:
    """Select the best model index from survivors using the given ranking method.

    Parameters
    ----------
    survivors : list of int
        Indices into *candidates* that passed hard criteria (C1, C3).
    candidates : sequence of DiagnosticMetrics
        Full list of candidate diagnostics.
    method : RankingMethod
        Which ranking strategy to use.

    Returns
    -------
    int
        Index (into *candidates*) of the selected best model.
    """
    if len(survivors) == 0:
        raise ValueError("No survivors to rank.")
    if len(survivors) == 1:
        return survivors[0]

    # Extract metrics for survivors
    traj = np.array([candidates[i].trajectory_val_loss for i in survivors])
    lc = np.array([
        candidates[i].loop_closure_loss if candidates[i].loop_closure_loss is not None
        else np.nan
        for i in survivors
    ])

    # Filter to valid (non-NaN traj loss, non-NaN lc loss for methods that need it)
    finite_traj = np.isfinite(traj)

    if method == "best_traj_loss":
        return _best_traj_loss(survivors, traj, finite_traj)

    # All other methods require loop closure loss
    valid = finite_traj & np.isfinite(lc)
    if not valid.any():
        # Fall back to best traj loss if no valid LC data
        return _best_traj_loss(survivors, traj, finite_traj)

    if method == "pareto_knee":
        return _pareto_knee(survivors, traj, lc, valid)
    elif method == "geo_rank":
        return _geo_rank(survivors, traj, lc, valid)
    elif method == "minimax_rank":
        return _minimax_rank(survivors, traj, lc, valid)
    elif method == "geo_log_score":
        return _geo_log_score(survivors, traj, lc, valid)
    elif method == "minimax_log_score":
        return _minimax_log_score(survivors, traj, lc, valid)
    else:
        raise ValueError(f"Unknown ranking method: {method!r}")


def _best_traj_loss(
    survivors: List[int],
    traj: np.ndarray,
    finite: np.ndarray,
) -> int:
    """Pick the survivor with lowest trajectory validation loss."""
    valid_idx = np.where(finite)[0]
    if len(valid_idx) == 0:
        return survivors[0]
    best_local = valid_idx[np.argmin(traj[valid_idx])]
    return survivors[best_local]


def _pareto_knee(
    survivors: List[int],
    traj: np.ndarray,
    lc: np.ndarray,
    valid: np.ndarray,
) -> int:
    """Find the knee of the Pareto front in (lc_loss, traj_loss) space."""
    valid_idx = np.where(valid)[0]
    traj_v = traj[valid_idx]
    lc_v = lc[valid_idx]

    # Identify Pareto-optimal points (lower is better on both)
    pareto_mask = np.zeros(len(valid_idx), dtype=bool)
    for i in range(len(valid_idx)):
        dominated = False
        for j in range(len(valid_idx)):
            if i == j:
                continue
            if traj_v[j] <= traj_v[i] and lc_v[j] <= lc_v[i]:
                if traj_v[j] < traj_v[i] or lc_v[j] < lc_v[i]:
                    dominated = True
                    break
        if not dominated:
            pareto_mask[i] = True

    pareto_local = np.where(pareto_mask)[0]
    if len(pareto_local) == 0:
        # Shouldn't happen, but fall back
        return _best_traj_loss(survivors, traj, np.isfinite(traj))

    # Sort Pareto points by lc_loss
    order = pareto_local[np.argsort(lc_v[pareto_local])]

    if len(order) < 3:
        # Too few points for curvature — pick the one with best geometric
        # mean of log-normalized scores among Pareto points
        return _geo_log_score_from_arrays(
            [survivors[valid_idx[i]] for i in order],
            traj_v[order],
            lc_v[order],
        )

    # Knee = max curvature in log-log space
    x_log = np.log10(lc_v[order])
    y_log = np.log10(traj_v[order])

    x_n = (x_log - x_log.min()) / (x_log.max() - x_log.min() + 1e-12)
    y_n = (y_log - y_log.min()) / (y_log.max() - y_log.min() + 1e-12)

    dx = np.diff(x_n)
    dy = np.diff(y_n)
    curvature = np.abs(dx[:-1] * dy[1:] - dx[1:] * dy[:-1]) / (
        np.sqrt(dx[:-1]**2 + dy[:-1]**2) * np.sqrt(dx[1:]**2 + dy[1:]**2) + 1e-12
    )

    knee_local = np.argmax(curvature) + 1  # interior point index
    knee_in_valid = order[knee_local]
    return survivors[valid_idx[knee_in_valid]]


def _geo_log_score_from_arrays(
    survivor_subset: List[int],
    traj_v: np.ndarray,
    lc_v: np.ndarray,
) -> int:
    """Helper: pick best by geometric mean of log-normalized scores."""
    log_traj = np.log10(traj_v)
    log_lc = np.log10(lc_v)
    s_traj = (log_traj - log_traj.min()) / (log_traj.max() - log_traj.min() + 1e-12)
    s_lc = (log_lc - log_lc.min()) / (log_lc.max() - log_lc.min() + 1e-12)
    scores = np.sqrt(s_traj * s_lc)
    return survivor_subset[int(np.argmin(scores))]


def _apply_ranks(values: np.ndarray) -> np.ndarray:
    """Compute ordinal ranks (1 = best/lowest), handling ties with average."""
    from scipy.stats import rankdata
    return rankdata(values, method="average")


def _geo_rank(
    survivors: List[int],
    traj: np.ndarray,
    lc: np.ndarray,
    valid: np.ndarray,
) -> int:
    """Pick the survivor with lowest geometric mean of ordinal ranks."""
    valid_idx = np.where(valid)[0]
    traj_ranks = _apply_ranks(traj[valid_idx])
    lc_ranks = _apply_ranks(lc[valid_idx])
    geo = np.sqrt(traj_ranks * lc_ranks)
    best_local = valid_idx[np.argmin(geo)]
    return survivors[best_local]


def _minimax_rank(
    survivors: List[int],
    traj: np.ndarray,
    lc: np.ndarray,
    valid: np.ndarray,
) -> int:
    """Pick the survivor with lowest max (worst-case) ordinal rank."""
    valid_idx = np.where(valid)[0]
    traj_ranks = _apply_ranks(traj[valid_idx])
    lc_ranks = _apply_ranks(lc[valid_idx])
    mm = np.maximum(traj_ranks, lc_ranks)
    best_local = valid_idx[np.argmin(mm)]
    return survivors[best_local]


def _geo_log_score(
    survivors: List[int],
    traj: np.ndarray,
    lc: np.ndarray,
    valid: np.ndarray,
) -> int:
    """Pick the survivor with lowest geometric mean of log-normalized scores."""
    valid_idx = np.where(valid)[0]
    return _geo_log_score_from_arrays(
        [survivors[i] for i in valid_idx],
        traj[valid_idx],
        lc[valid_idx],
    )


def _minimax_log_score(
    survivors: List[int],
    traj: np.ndarray,
    lc: np.ndarray,
    valid: np.ndarray,
) -> int:
    """Pick the survivor with lowest max of log-normalized scores."""
    valid_idx = np.where(valid)[0]
    traj_v = traj[valid_idx]
    lc_v = lc[valid_idx]

    log_traj = np.log10(traj_v)
    log_lc = np.log10(lc_v)
    s_traj = (log_traj - log_traj.min()) / (log_traj.max() - log_traj.min() + 1e-12)
    s_lc = (log_lc - log_lc.min()) / (log_lc.max() - log_lc.min() + 1e-12)
    mm = np.maximum(s_traj, s_lc)
    best_local = valid_idx[np.argmin(mm)]
    return survivors[best_local]
