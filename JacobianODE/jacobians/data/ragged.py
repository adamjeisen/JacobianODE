"""Ragged-trajectory utilities: NaN-padded storage + sliding-window batching.

Used when full-length trajectories have variable length (e.g. the
resting-state windows extracted from neural recordings — each is 3-15
seconds long). The convention:

    *_trajs_full          : (N, max_T, D), NaN-padded
    *_trajs_full_lengths  : (N,)         , int valid length per trajectory

Consumers that treat each trajectory independently must slice
``[:lengths[i]]`` first; consumers that need uniform-length sub-windows
for training use ``sliding_windows`` to materialize them.

Higher-level orchestration helpers:

* :func:`delay_embed_ragged`: per-trajectory delay embedding (the
  alternative to "stack first then embed" — used when trajectories have
  different valid lengths and we want to avoid mixing across the
  NaN boundary).
* :func:`split_balanced_by_timepoints`: per-condition trajectory-level
  train/val/test split that matches *timepoint counts* (not trajectory
  counts) to the requested fractions. Avoids leakage of sub-windows from
  the same parent trajectory across splits.
"""
from __future__ import annotations

import logging
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
    min_kept_length: int | None = None,
):
    """Trim a chronologically-ordered list of trajectories so that the
    total sample count equals ``T_target`` exactly, redistributing the
    trim across the last two kept trajectories if the naive partial
    would fall below ``min_length``.

    Two thresholds:

    * ``min_length``: the *desired* minimum trajectory length AND the
      trigger for redistribution. If the naive partial of the next
      trajectory would land below this, we attempt to redistribute.
    * ``min_kept_length``: the *absolute floor* for any kept trajectory
      after redistribution. Defaults to ``min_length`` (single-threshold
      behavior). When the data has many trajectories close to
      ``min_length`` already, the 2-way redistribute often can't keep
      both halves at ``min_length``; setting ``min_kept_length`` lower
      than ``min_length`` lets the redistributed pair drop below the
      *desired* min while staying above the *hard* floor.

    Algorithm (chronological, with stub avoidance):

    1. Walk trajectories chronologically, accumulating samples into
       ``S = sum(L_1..L_j)`` while ``S + L_{j+1} <= T_target``. After
       the loop, ``S_j <= T_target < S_j + L_{j+1}``.
    2. ``delta = T_target - S_j`` is what we'd take from ``t_{j+1}``.
       If ``delta == 0``, return ``t_1..t_j`` as-is.
    3. **Case A** (``delta >= min_length``): take ``t_{j+1}[:delta]``
       as a clean partial. Return ``t_1..t_j + partial``.
    4. **Case B** (``0 < delta < min_length``, redistribute case):
       redistribute across ``t_j`` AND ``t_{j+1}``. The combined budget
       is ``budget = L_j + delta = T_target - S_{j-1}``. Split into
       ``(L_j', L_{j+1}')`` summing to ``budget`` with both
       ``>= min_kept_length``. Even split, ceiling-rounded toward the
       earlier trajectory on odd budgets
       (``L_j' = ceil(budget/2)``, ``L_{j+1}' = floor(budget/2)``);
       clamp to upper bounds (``L_j' <= L_j``, ``L_{j+1}' <= L_{j+1}``)
       and absorb overflow on the other side. If
       ``budget < 2 * min_kept_length``, raise — there's no feasible
       split that keeps both above the absolute floor.

    Parameters
    ----------
    trajs : sequence of array-like, each ``(T_i, ...)``
        Trajectories in chronological order. The first axis length is
        the sample count along time; trailing dimensions are arbitrary.
    T_target : int
        Desired total sample count after truncation.
    min_length : int
        Desired minimum length per kept trajectory AND the trigger for
        redistribution (a clean partial is taken whenever
        ``delta >= min_length``).
    min_kept_length : int, optional
        Absolute floor for any kept trajectory after redistribution.
        Defaults to ``min_length`` (single-threshold mode). Must be
        ``> 0`` and ``<= min_length``.

    Returns
    -------
    list
        Truncated trajectories whose sample counts sum to exactly
        ``T_target``. Each entry has length ``>= min_kept_length``.
        Same array-like type as the input (slicing preserves the
        type).

    Raises
    ------
    ValueError
        If ``T_target < 0``; ``min_length <= 0`` or
        ``min_kept_length <= 0``; ``min_kept_length > min_length``;
        total samples in ``trajs`` is less than ``T_target``; or the
        redistribution budget is ``< 2 * min_kept_length``.
    """
    if T_target < 0:
        raise ValueError(f"T_target must be >= 0; got {T_target}")
    if min_length <= 0:
        raise ValueError(f"min_length must be > 0; got {min_length}")
    if min_kept_length is None:
        min_kept_length = min_length
    if min_kept_length <= 0:
        raise ValueError(f"min_kept_length must be > 0; got {min_kept_length}")
    if min_kept_length > min_length:
        raise ValueError(
            f"min_kept_length ({min_kept_length}) must be <= min_length ({min_length})"
        )

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

    if budget < 2 * min_kept_length:
        raise ValueError(
            f"truncate_chronological_balanced: budget {budget} samples for the last "
            f"two kept trajectories (#{j} length {L_j}, #{next_idx} length {L_curr}) "
            f"is < 2 * min_kept_length = {2 * min_kept_length}; cannot avoid a stub."
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

    # Defensive: feasibility should be guaranteed by the
    # budget >= 2*min_kept_length check above plus L_j + L_curr >= budget
    # (since budget = L_j + delta and delta < L_curr). Validate anyway in case
    # of arithmetic surprises.
    if L_j_kept < min_kept_length or L_curr_kept < min_kept_length:
        raise ValueError(
            f"truncate_chronological_balanced: post-clamp split "
            f"({L_j_kept}, {L_curr_kept}) violates min_kept_length={min_kept_length}; "
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


def delay_embed_ragged(
    padded: torch.Tensor,
    lengths: torch.Tensor,
    n_delays: int,
    delay_spacing: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-trajectory delay-embed a NaN-padded ragged tensor.

    Each trajectory is delay-embedded *independently* on its valid
    ``[0:lengths[i]]`` slice via
    :func:`JacobianODE.jacobians.data.splitting.embed_signal_torch`. This
    avoids mixing values across the NaN-padded boundary (which would
    happen if you delay-embedded the whole padded tensor in one shot).
    The resulting embedding has length
    ``lengths[i] - (n_delays - 1) * delay_spacing`` per trajectory; the
    output is re-padded to a common ``max_T_de`` with NaN.

    Parameters
    ----------
    padded : torch.Tensor of shape ``(N, max_T, D)``
        NaN-padded raw trajectories (typically from
        :func:`pad_trajs_to_max`).
    lengths : torch.Tensor of shape ``(N,)``
        Valid sample count per trajectory.
    n_delays : int
        Number of delay copies. ``n_delays=1`` is a no-op (returns
        copies of the inputs).
    delay_spacing : int, default 1
        Time-step gap between consecutive delay copies.

    Returns
    -------
    de_padded : torch.Tensor of shape ``(N, max_T_de, D * n_delays)``
        NaN-padded delay-embedded trajectories. Layout matches
        :func:`embed_signal_torch`: most recent state at columns
        ``0:D``, then ``D:2D`` for one step back, etc.
    de_lengths : torch.Tensor of shape ``(N,)``, dtype int64
        New valid length per trajectory:
        ``max(0, lengths[i] - (n_delays - 1) * delay_spacing)``.
        Trajectories shorter than the embedding window contribute zero
        valid samples.

    Raises
    ------
    ValueError
        If ``padded`` is not 3-D, ``lengths`` shape disagrees with
        ``padded.shape[0]``, or ``n_delays`` / ``delay_spacing`` are
        not positive.
    """
    if padded.ndim != 3:
        raise ValueError(
            f"delay_embed_ragged: padded must be 3-D (N, max_T, D); "
            f"got shape {tuple(padded.shape)}"
        )
    if lengths.ndim != 1 or lengths.shape[0] != padded.shape[0]:
        raise ValueError(
            f"delay_embed_ragged: lengths must be 1-D matching padded.shape[0]; "
            f"got lengths shape {tuple(lengths.shape)}, padded shape {tuple(padded.shape)}"
        )
    if n_delays < 1:
        raise ValueError(f"delay_embed_ragged: n_delays must be >= 1; got {n_delays}")
    if delay_spacing < 1:
        raise ValueError(
            f"delay_embed_ragged: delay_spacing must be >= 1; got {delay_spacing}"
        )

    if n_delays == 1:
        return padded.clone(), lengths.clone().to(torch.long)

    # Lazy import to avoid a circular import (splitting imports ragged
    # indirectly through data/__init__.py).
    from .splitting import embed_signal_torch

    N, _, D = padded.shape
    de_lengths = torch.clamp(lengths - (n_delays - 1) * delay_spacing, min=0).to(torch.long)
    max_T_de = int(de_lengths.max().item()) if N > 0 else 0

    out = torch.full(
        (N, max_T_de, D * n_delays),
        fill_value=float("nan"),
        dtype=padded.dtype,
    )
    for i in range(N):
        L_i = int(lengths[i].item())
        L_i_de = int(de_lengths[i].item())
        if L_i_de <= 0:
            continue
        # embed_signal_torch is no-grad and shape-stable; pass the valid
        # slice to keep NaN out of the embedding.
        de = embed_signal_torch(padded[i, :L_i], n_delays, delay_spacing)
        out[i, :L_i_de] = de.to(padded.dtype)

    return out, de_lengths


def split_balanced_by_timepoints(
    lengths: torch.Tensor | np.ndarray,
    source_id: torch.Tensor | np.ndarray,
    train_percent: float,
    test_percent: float,
    *,
    seed: int = 42,
    log: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trajectory-level train/val/test split that balances *timepoints*.

    For each unique ``source_id`` value, greedily assigns parent
    trajectories to train / val / test buckets so the **sum of valid
    timepoints** in each bucket lands close to its target fraction. The
    split is at the trajectory level — every parent trajectory's
    timepoints stay in a single bucket, eliminating the leakage path
    where sub-windows from the same parent end up in different splits.

    Algorithm:

    1. For each source ``s``: enumerate trajectory indices in the
       source, compute total timepoints, and set targets
       ``train_pct * total``, ``val_pct * total``, ``test_pct * total``.
    2. Shuffle the source's trajectory indices with the seed.
    3. For each trajectory in shuffled order, assign it to whichever
       bucket has the largest remaining deficit (in fraction-of-target
       units), then advance that bucket's running sum.

    Parameters
    ----------
    lengths : array-like of shape ``(N_traj,)``
        Per-trajectory valid sample count (typically the
        delay-embedded lengths from :func:`delay_embed_ragged`, so the
        split is balanced over the *delay-embedded* sample budget,
        which is what the dataloader actually sees).
    source_id : array-like of shape ``(N_traj,)``
        Per-trajectory integer source/condition ID. Each unique value
        is split independently so condition counts stay balanced
        across splits.
    train_percent, test_percent : float
        Fraction of timepoints to put in train / test (val gets the
        remainder).
    seed : int, default 42
        Shuffle seed (same seed → same assignment).
    log : logging.Logger, optional
        If provided, emits one info line per source with the achieved
        vs target fractions — useful for sanity-checking that the split
        actually approximates the requested ratios on small data.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray of int
        Trajectory indices (into the original ``lengths`` /
        ``source_id`` arrays) for each split. Each parent trajectory
        appears in exactly one of the three.

    Raises
    ------
    ValueError
        If ``train_percent + test_percent > 1``; or if any source has
        fewer than 3 trajectories (need at least one per bucket).
    """
    val_percent = 1.0 - train_percent - test_percent
    if train_percent < 0 or test_percent < 0 or val_percent < 0:
        raise ValueError(
            f"train_percent={train_percent}, test_percent={test_percent}: "
            f"all of (train, val=1-train-test, test) must be in [0, 1]"
        )

    rng = np.random.RandomState(seed)
    src_arr = np.asarray(source_id)
    if isinstance(lengths, torch.Tensor):
        lengths_arr = lengths.detach().cpu().numpy().astype(np.int64)
    else:
        lengths_arr = np.asarray(lengths).astype(np.int64)
    if lengths_arr.shape != src_arr.shape:
        raise ValueError(
            f"split_balanced_by_timepoints: lengths shape {lengths_arr.shape} != "
            f"source_id shape {src_arr.shape}"
        )

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for s in np.unique(src_arr):
        cond_mask = src_arr == s
        cond_traj_idx = np.where(cond_mask)[0]
        if cond_traj_idx.size < 3:
            raise ValueError(
                f"split_balanced_by_timepoints: source {s!r} has only "
                f"{cond_traj_idx.size} trajectories; need at least 3 to fill "
                f"train+val+test."
            )
        cond_lengths = lengths_arr[cond_traj_idx]
        total_tp = int(cond_lengths.sum())
        targets = {
            "train": train_percent * total_tp,
            "val": val_percent * total_tp,
            "test": test_percent * total_tp,
        }
        cums = {"train": 0, "val": 0, "test": 0}
        buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        order = rng.permutation(cond_traj_idx.size)
        for j in order:
            traj_i = int(cond_traj_idx[j])
            L = int(cond_lengths[j])
            # Pick bucket with the largest remaining deficit, normalized by
            # its target (so a bucket whose target is 0 — e.g. test_pct=0 —
            # never wins; a bucket nearly full doesn't keep grabbing).
            deficits = {
                b: (targets[b] - cums[b]) / targets[b] if targets[b] > 0 else -np.inf
                for b in cums
            }
            best = max(deficits, key=deficits.get)
            buckets[best].append(traj_i)
            cums[best] += L
        train_idx.extend(buckets["train"])
        val_idx.extend(buckets["val"])
        test_idx.extend(buckets["test"])

        # Sanity: each bucket must have at least one trajectory if its target
        # fraction is non-zero (otherwise downstream dataloader build will
        # produce an empty split). Greedy rarely fails this on real data, but
        # surface it loudly when it does.
        for b, target in targets.items():
            if target > 0 and not buckets[b]:
                raise ValueError(
                    f"split_balanced_by_timepoints: source {s!r}: greedy "
                    f"assignment produced an empty {b!r} bucket despite "
                    f"target_fraction>0; got {len(cond_traj_idx)} trajectories "
                    f"with lengths={cond_lengths.tolist()}; consider a different "
                    f"seed or larger N."
                )

        if log is not None:
            log.info(
                f"  source {int(s)}: {len(cond_traj_idx)} trajectories, "
                f"{total_tp} timepoints | "
                f"train: {len(buckets['train'])} trajs / {cums['train']} tp "
                f"({cums['train']/max(total_tp,1):.3f} vs target {train_percent:.3f}) | "
                f"val: {len(buckets['val'])} trajs / {cums['val']} tp "
                f"({cums['val']/max(total_tp,1):.3f} vs target {val_percent:.3f}) | "
                f"test: {len(buckets['test'])} trajs / {cums['test']} tp "
                f"({cums['test']/max(total_tp,1):.3f} vs target {test_percent:.3f})"
            )

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )
