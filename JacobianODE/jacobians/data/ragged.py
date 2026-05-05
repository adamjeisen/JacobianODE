"""Ragged-trajectory utilities: NaN-padded storage + sliding-window batching.

Used when full-length trajectories have variable length (e.g. the
resting-state windows extracted from neural recordings — each is 3-15
seconds long). The convention:

    *_trajs_full          : (N, max_T, D), NaN-padded
    *_trajs_full_lengths  : (N,)         , int valid length per trajectory

Consumers that treat each trajectory independently must slice
``[:lengths[i]]`` first; consumers that need uniform-length sub-windows
for training use ``sliding_windows`` to materialize them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def pad_trajs_to_max(
    trajs: Sequence[torch.Tensor | np.ndarray],
    fill_value: float = float("nan"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length trajectories to a common length.

    Parameters
    ----------
    trajs : sequence of array-like, each ``(T_i, D)``
        Variable-length trajectories. All must share the trailing
        dimension ``D``.
    fill_value : float, optional
        Padding fill value (default NaN). Use NaN to surface bugs in
        any consumer that forgets to honor ``lengths``.

    Returns
    -------
    padded : torch.Tensor of shape ``(N, max_T, D)``
        Stacked trajectories padded along axis 1 with ``fill_value``.
        Output dtype is float32 (or the upcast of the input dtypes if
        any input is float64) — NaN padding requires a float dtype.
    lengths : torch.Tensor of shape ``(N,)``, dtype int64
        Valid length per trajectory (number of non-padded samples
        along axis 1).

    Raises
    ------
    ValueError
        If ``trajs`` is empty, any element is not 2-D, or trailing
        dimensions disagree.
    """
    if len(trajs) == 0:
        raise ValueError("pad_trajs_to_max: trajs is empty")

    tensors = [
        t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
        for t in trajs
    ]

    if any(t.ndim != 2 for t in tensors):
        bad_shapes = [tuple(t.shape) for t in tensors]
        raise ValueError(
            f"pad_trajs_to_max: every trajectory must be 2-D (T_i, D); "
            f"got shapes {bad_shapes}"
        )
    D = tensors[0].shape[-1]
    if any(t.shape[-1] != D for t in tensors):
        raise ValueError(
            f"pad_trajs_to_max: every trajectory must have the same "
            f"trailing dim D; got {[t.shape[-1] for t in tensors]}"
        )

    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    max_T = int(lengths.max().item())

    # Choose a float output dtype so NaN can be represented.
    if any(t.dtype == torch.float64 for t in tensors):
        out_dtype = torch.float64
    else:
        out_dtype = torch.float32

    padded = torch.full(
        (len(tensors), max_T, D),
        fill_value=fill_value,
        dtype=out_dtype,
    )
    for i, t in enumerate(tensors):
        padded[i, : t.shape[0]] = t.to(out_dtype)

    return padded, lengths


def truncate_chronological_balanced(
    trajs: Sequence[torch.Tensor | np.ndarray],
    T_target: int,
    min_length: int,
):
    """Trim a chronologically-ordered list of trajectories so that the
    total sample count equals ``T_target`` exactly, redistributing the
    trim across the last two kept trajectories if needed to keep every
    kept trajectory at least ``min_length`` samples long.

    Algorithm (chronological, with stub avoidance):

    1. Walk trajectories chronologically, accumulating samples into
       ``S = sum(L_1..L_j)`` while ``S + L_{j+1} <= T_target``. After
       the loop, ``S_j <= T_target < S_j + L_{j+1}``.
    2. ``delta = T_target - S_j`` is what we'd take from ``t_{j+1}``.
       If ``delta == 0``, return ``t_1..t_j`` as-is.
    3. **Case A** (``delta >= min_length``): take ``t_{j+1}[:delta]``
       as a clean partial. Return ``t_1..t_j + partial``.
    4. **Case B** (``0 < delta < min_length``, stub case): redistribute
       across ``t_j`` AND ``t_{j+1}``. The combined budget is
       ``budget = L_j + delta = T_target - S_{j-1}``. Split into
       ``(L_j', L_{j+1}')`` summing to ``budget`` with both
       ``>= min_length``. Even split, ceiling-rounded toward the
       earlier trajectory on odd budgets (``L_j' = ceil(budget/2)``,
       ``L_{j+1}' = floor(budget/2)``); clamp to upper bounds
       ``L_j' <= L_j``, ``L_{j+1}' <= L_{j+1}`` and absorb overflow on
       the other side. If ``budget < 2 * min_length``, raise — there's
       no feasible split.

    Parameters
    ----------
    trajs : sequence of array-like, each ``(T_i, ...)``
        Trajectories in chronological order. The first axis length is
        the sample count along time; trailing dimensions are arbitrary.
    T_target : int
        Desired total sample count after truncation.
    min_length : int
        Minimum allowed sample count per output trajectory.

    Returns
    -------
    list
        Truncated trajectories whose sample counts sum to exactly
        ``T_target``. Each entry has length ``>= min_length``. Same
        array-like type as the input (slicing preserves the type).

    Raises
    ------
    ValueError
        If ``T_target < 0``, ``min_length <= 0``, total samples in
        ``trajs`` is less than ``T_target``, or the redistribution is
        infeasible because the budget for the last two kept
        trajectories is ``< 2 * min_length``.
    """
    if T_target < 0:
        raise ValueError(f"T_target must be >= 0; got {T_target}")
    if min_length <= 0:
        raise ValueError(f"min_length must be > 0; got {min_length}")

    if T_target == 0:
        return []

    total = sum(int(t.shape[0]) for t in trajs)
    if total < T_target:
        raise ValueError(
            f"truncate_chronological_balanced: total samples {total} < T_target "
            f"{T_target}; cannot reach the target by truncation"
        )
    if total == T_target:
        return list(trajs)

    # Walk chronologically; j = index of last fully-kept trajectory
    cumsum = 0
    j = -1
    for i, t in enumerate(trajs):
        L_i = int(t.shape[0])
        if cumsum + L_i <= T_target:
            cumsum += L_i
            j = i
        else:
            break

    delta = T_target - cumsum  # 0 <= delta < L_{j+1}

    if delta == 0:
        # Cumsum exactly hit T_target — no partial needed
        return list(trajs[: j + 1])

    # There must be a t_{j+1} since total > T_target and we exited the loop
    # without consuming all trajectories.
    next_idx = j + 1
    L_curr = int(trajs[next_idx].shape[0])

    # Case A: clean partial
    if delta >= min_length:
        kept = list(trajs[: j + 1])
        kept.append(trajs[next_idx][:delta])
        return kept

    # Case B: stub case. Need to redistribute over t_j and t_{j+1}.
    if j < 0:
        raise ValueError(
            f"truncate_chronological_balanced: would take {delta} samples "
            f"(< min_length={min_length}) from the only candidate trajectory; "
            f"no previous trajectory available to split with."
        )

    L_j = int(trajs[j].shape[0])
    budget = L_j + delta  # samples assigned to t_j and t_{j+1} together

    if budget < 2 * min_length:
        raise ValueError(
            f"truncate_chronological_balanced: budget {budget} samples for the last "
            f"two kept trajectories (#{j} length {L_j}, #{next_idx} length {L_curr}) "
            f"is < 2 * min_length = {2 * min_length}; cannot avoid a stub."
        )

    # Even split, with the earlier trajectory getting the extra sample on odd
    # budgets (deterministic tiebreak).
    L_j_kept = (budget + 1) // 2
    L_curr_kept = budget // 2

    # Clamp to upper bounds (can't keep more than what's there).
    if L_j_kept > L_j:
        L_j_kept = L_j
        L_curr_kept = budget - L_j_kept
    elif L_curr_kept > L_curr:
        L_curr_kept = L_curr
        L_j_kept = budget - L_curr_kept

    # Defensive: feasibility should be guaranteed by the budget >= 2*min_length
    # check above plus L_j + L_curr >= budget (since budget = L_j + delta and
    # delta < L_curr). Validate anyway in case of arithmetic surprises.
    if L_j_kept < min_length or L_curr_kept < min_length:
        raise ValueError(
            f"truncate_chronological_balanced: post-clamp split "
            f"({L_j_kept}, {L_curr_kept}) violates min_length={min_length}; "
            f"budget={budget}, L_j={L_j}, L_curr={L_curr}"
        )

    kept = list(trajs[:j])
    kept.append(trajs[j][:L_j_kept])
    kept.append(trajs[next_idx][:L_curr_kept])
    return kept


def sliding_windows(
    padded: torch.Tensor,
    lengths: torch.Tensor,
    seq_length: int,
    stride: int,
) -> torch.Tensor:
    """Extract uniform sub-trajectories from a NaN-padded ragged tensor.

    For each trajectory of length ``T_i``, emits sub-windows starting at
    indices ``0, stride, 2*stride, ...`` as long as
    ``start + seq_length <= T_i`` (so no window straddles the padded
    region). Trajectories shorter than ``seq_length`` contribute zero
    windows.

    Parameters
    ----------
    padded : torch.Tensor of shape ``(N, max_T, D)``
        NaN-padded trajectories from :func:`pad_trajs_to_max`.
    lengths : torch.Tensor of shape ``(N,)``
        Valid length per trajectory.
    seq_length : int
        Sub-window length.
    stride : int
        Step between consecutive sub-window starts within a trajectory.

    Returns
    -------
    windows : torch.Tensor of shape ``(K, seq_length, D)``
        Stacked sub-windows.
        ``K = sum_i max(0, floor((T_i - seq_length) / stride) + 1)``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``seq_length`` / ``stride`` are
        not positive.
    """
    if padded.ndim != 3:
        raise ValueError(
            f"sliding_windows: padded must be 3-D (N, max_T, D); "
            f"got shape {tuple(padded.shape)}"
        )
    if lengths.ndim != 1 or lengths.shape[0] != padded.shape[0]:
        raise ValueError(
            f"sliding_windows: lengths must be 1-D matching padded.shape[0]; "
            f"got lengths shape {tuple(lengths.shape)}, "
            f"padded shape {tuple(padded.shape)}"
        )
    if seq_length <= 0:
        raise ValueError(f"sliding_windows: seq_length must be > 0; got {seq_length}")
    if stride <= 0:
        raise ValueError(f"sliding_windows: stride must be > 0; got {stride}")

    N, _, D = padded.shape
    chunks: list[torch.Tensor] = []
    for i in range(N):
        T_i = int(lengths[i].item())
        if T_i < seq_length:
            continue
        valid = padded[i, :T_i]  # (T_i, D)
        # ``unfold`` returns (n_windows, D, seq_length); permute to
        # (n_windows, seq_length, D).
        unfolded = valid.unfold(dimension=0, size=seq_length, step=stride)
        unfolded = unfolded.permute(0, 2, 1).contiguous()
        chunks.append(unfolded)

    if not chunks:
        return torch.empty(0, seq_length, D, dtype=padded.dtype)
    return torch.cat(chunks, dim=0)
