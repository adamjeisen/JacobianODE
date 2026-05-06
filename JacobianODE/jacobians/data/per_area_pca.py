"""Per-area PCA reduction as a pre-delay-embed step.

For multi-area observations (DirectSum encoder use-case), the raw
channel count per area is often much higher than the area's intrinsic
dimensionality. Running per-area PCA at e.g. 99% variance BEFORE delay
embedding shrinks the input space the encoder + dynamics MLP have to
swallow without throwing away any useful variance.

The motivating measurement: on Mary propofol resting LFP, per-area
99%-variance PCA reduces 249 raw channels → 60 (24% of raw). After
delay-embedding the reduced data and running the JacobianODE pipeline's
own autodim PCA, the dynamic subspace shrinks ~44% (e.g. 89 → 50 dims
at n_delays=10). See
``mindcontrol/diagnostics/per_area_pca_pipeline.py`` for the full
diagnostic.

This module provides:

* :func:`fit_per_area_pca` — fit per-area PCA components on a
  NaN-padded ragged tensor, choosing the number of components per area
  to capture ``threshold`` cumulative variance.
* :func:`apply_per_area_pca` — project raw obs through the fitted
  per-area PCA, returning a new ragged tensor + new ``area_indices``
  layout for downstream encoders.

Pure: no logging, no cfg dependencies. Caller (e.g. ``train_from_arrays``)
is responsible for stashing the fitted components in cfg if it wants to
roundtrip predictions back to obs space later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class PerAreaPCA:
    """One area's fitted PCA: top-K components + center + dim metadata.

    Storing as plain torch tensors so the bundle pickles into Lightning
    checkpoints / cfgs without surprises.
    """
    components: torch.Tensor       # (k, n_in), top-k right singular vectors
    mean: torch.Tensor              # (n_in,) per-channel mean (subtracted before projection)
    n_in: int
    k: int                          # number of components kept
    cum_var: float                  # cumulative variance fraction at k


@dataclass
class PerAreaPCAFit:
    """Result of fit_per_area_pca over a list of per-area channel partitions.

    ``area_indices_in`` is the original raw partition (1-D channel
    indices into the input ``values``'s last dim).
    ``area_indices_out`` is the corresponding partition in the projected
    space (each area's k contiguous output dims).
    """
    per_area: list[PerAreaPCA]
    area_indices_in: list[list[int]]
    area_indices_out: list[list[int]]
    n_total_out: int                # sum_areas(k)
    threshold: float                # threshold used for selection
    area_names: tuple[str, ...] | None = None


def _collect_valid_samples(
    values: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Return ``(sum_lengths, D)`` of just the valid (pre-padding) samples
    concatenated across all trajectories. Stays on the same device as
    ``values`` so the SVD can run on GPU when appropriate."""
    chunks = []
    for i in range(values.shape[0]):
        L = int(lengths[i].item())
        if L > 0:
            chunks.append(values[i, :L])
    if not chunks:
        return torch.zeros(0, values.shape[-1], dtype=values.dtype, device=values.device)
    return torch.cat(chunks, dim=0)


def fit_per_area_pca(
    values: torch.Tensor,                      # (K, T_max, D), NaN-padded
    lengths: torch.Tensor,                     # (K,)
    area_indices: Sequence[Sequence[int]],     # per-area channel index lists
    *,
    threshold: float = 0.99,
    area_names: Sequence[str] | None = None,
) -> PerAreaPCAFit:
    """Fit per-area PCA at the given variance threshold.

    For each area in ``area_indices``, we slice the raw obs tensor to
    that area's channels, concatenate the valid samples across all
    trajectories, run an SVD on the centered matrix, and keep the
    smallest k such that cumulative variance ≥ ``threshold``.

    Components are stored as ``(k, n_in)`` tensors (one row per
    component). To project: ``y = (x - mean) @ components.T``.

    Parameters
    ----------
    values : torch.Tensor of shape ``(K, T_max, D)``
        NaN-padded raw observations.
    lengths : torch.Tensor of shape ``(K,)``
        Valid sample count per trajectory.
    area_indices : list of list of int
        Per-area channel partitions (1-D indices into ``D``). Each
        sub-list slices ``values[..., idx]`` to that area's data.
    threshold : float
        Cumulative variance threshold. The smallest k giving
        ``cum_var[k-1] >= threshold`` is kept.
    area_names : tuple of str, optional
        Echoed back in the result for downstream report metadata.

    Returns
    -------
    :class:`PerAreaPCAFit`
        Fitted PCA bundle + projected-space ``area_indices_out`` so the
        caller can update the encoder's ``area_indices`` field.
    """
    if values.ndim != 3:
        raise ValueError(
            f"fit_per_area_pca: values must be 3-D (K, T_max, D); got {tuple(values.shape)}"
        )
    if lengths.ndim != 1 or lengths.shape[0] != values.shape[0]:
        raise ValueError(
            f"fit_per_area_pca: lengths shape {tuple(lengths.shape)} != "
            f"(K={values.shape[0]},)"
        )
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1); got {threshold}")

    per_area: list[PerAreaPCA] = []
    area_indices_out: list[list[int]] = []
    cursor = 0
    for ai, idxs in enumerate(area_indices):
        idxs_list = list(idxs)
        if not idxs_list:
            raise ValueError(f"fit_per_area_pca: area {ai} has empty channel list")
        area_view = values[..., idxs_list]                   # (K, T_max, n_in)
        valid = _collect_valid_samples(area_view, lengths)   # (sum_lengths, n_in)
        if valid.shape[0] == 0:
            raise ValueError(
                f"fit_per_area_pca: area {ai} has no valid samples (all lengths == 0?)"
            )
        # Center + SVD. Use float64 internally for numerical stability;
        # the cost is small (n_in is at most a few hundred).
        v64 = valid.to(torch.float64)
        mean = v64.mean(dim=0, keepdim=True)
        centered = v64 - mean
        # Use full_matrices=False; only need the right singular vectors V.
        # torch.linalg.svd returns (U, S, Vh); Vh shape (min(M,N), N) for thin SVD.
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        var = s ** 2
        if var.sum() == 0:
            raise ValueError(f"fit_per_area_pca: area {ai} has zero variance")
        cum = torch.cumsum(var, dim=0) / var.sum()
        # Smallest k such that cum[k-1] >= threshold
        k = int((cum >= threshold).nonzero(as_tuple=True)[0][0].item()) + 1
        components = vh[:k].to(values.dtype)   # (k, n_in)
        mean_kept = mean.squeeze(0).to(values.dtype)   # (n_in,)

        per_area.append(PerAreaPCA(
            components=components,
            mean=mean_kept,
            n_in=len(idxs_list),
            k=k,
            cum_var=float(cum[k - 1].item()),
        ))
        area_indices_out.append(list(range(cursor, cursor + k)))
        cursor += k

    return PerAreaPCAFit(
        per_area=per_area,
        area_indices_in=[list(idx) for idx in area_indices],
        area_indices_out=area_indices_out,
        n_total_out=cursor,
        threshold=threshold,
        area_names=tuple(area_names) if area_names is not None else None,
    )


def apply_per_area_pca(
    values: torch.Tensor,                # (K, T_max, D), NaN-padded
    lengths: torch.Tensor,
    fit: PerAreaPCAFit,
) -> torch.Tensor:
    """Project a NaN-padded ragged tensor through per-area PCA.

    Output ``(K, T_max, n_total_out)``: the k-component projection per
    area concatenated along the feature axis (in the same area order
    as ``fit.per_area``). NaN-padded entries stay NaN.
    """
    K, T_max, _ = values.shape
    out = torch.full(
        (K, T_max, fit.n_total_out),
        fill_value=float("nan"),
        dtype=values.dtype,
        device=values.device,
    )
    for ai, pca in enumerate(fit.per_area):
        idxs_in = fit.area_indices_in[ai]
        idxs_out = fit.area_indices_out[ai]
        # Select the area's input channels
        area_view = values[..., idxs_in]   # (K, T_max, n_in)
        # Project per trajectory's valid region (NaN handling)
        for i in range(K):
            L_i = int(lengths[i].item())
            if L_i <= 0:
                continue
            x = area_view[i, :L_i]                            # (L_i, n_in)
            y = (x - pca.mean.to(x.device)) @ pca.components.to(x.device).T   # (L_i, k)
            out[i, :L_i, idxs_out[0]:idxs_out[-1] + 1] = y
    return out


def inverse_per_area_pca(
    reduced: torch.Tensor,                # (..., n_total_out)
    fit: PerAreaPCAFit,
) -> torch.Tensor:
    """Project a reduced-space tensor BACK to the raw obs space (in
    the original raw channel ordering).

    Output ``(..., D_raw)`` where ``D_raw = sum_areas(n_in)`` and the
    layout matches the original ``values[..., idxs_in_area_i]`` partitioning.
    Useful for reporting predictions in the original coordinate system.
    """
    *prefix, n_out = reduced.shape
    if n_out != fit.n_total_out:
        raise ValueError(
            f"inverse_per_area_pca: reduced.shape[-1]={n_out} != "
            f"fit.n_total_out={fit.n_total_out}"
        )
    D_raw = sum(pca.n_in for pca in fit.per_area)
    out = torch.zeros(*prefix, D_raw, dtype=reduced.dtype, device=reduced.device)
    for ai, pca in enumerate(fit.per_area):
        idxs_out = fit.area_indices_out[ai]
        idxs_in = fit.area_indices_in[ai]
        # ``reduced[..., idxs_out[0]:idxs_out[-1]+1]`` slices out this area's projection
        z = reduced[..., idxs_out[0]:idxs_out[-1] + 1]              # (..., k)
        x_centered = z @ pca.components.to(z.device)                # (..., n_in)
        x = x_centered + pca.mean.to(x_centered.device)
        # Scatter back to original raw indices
        for col_local, col_raw in enumerate(idxs_in):
            out[..., col_raw] = x[..., col_local]
    return out
