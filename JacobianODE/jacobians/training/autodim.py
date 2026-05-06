"""Automatic dimension selection for the latent dynamic subspace.

Provides :func:`infer_n_target_dims` which runs PCA or FNN on the
training delay-embedded sequences and returns the recommended
n_target_dims (single int) or n_target_dims_per_block (list of ints,
for DirectSum encoders).

Pure function: does NOT mutate cfg. Caller decides which fields to
write back. Extracted from inline logic in
:func:`JacobianODE.jacobians.run_jacobians.train` so both ``train(cfg)``
and the new ``train_from_arrays(...)`` API call the same code path.
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class InferNTargetDimsResult(NamedTuple):
    """Result of :func:`infer_n_target_dims`.

    Attributes
    ----------
    n_target_dims_total : int
        Total dynamic-subspace dim. For DirectSum, this is
        ``sum(n_target_dims_per_block)``.
    n_target_dims_per_block : list[int] or None
        For DirectSum encoders: list of ``n_target_dims`` per area.
        For single-encoder: None.
    pca_cum_var : float or list[float] or None
        Diagnostic. For PCA single-encoder: scalar cumulative variance
        captured at the picked k. For PCA DirectSum: list (per area).
        For FNN: None (FNN doesn't expose this).
    method : str
        Echoes the chosen method, ``'pca'`` or ``'fnn'``.
    is_direct_sum : bool
        True if the encoder is a DirectSum encoder (per-area
        partition was used).
    """
    n_target_dims_total: int
    n_target_dims_per_block: Optional[list[int]]
    pca_cum_var: Union[float, list[float], None]
    method: str
    is_direct_sum: bool


def infer_n_target_dims(
    cfg: DictConfig,
    train_seq: torch.Tensor,
) -> Optional[InferNTargetDimsResult]:
    """Infer the dynamic-subspace dimensionality from training data.

    Reads the following cfg fields:

    - ``model.n_target_dim_method`` (default ``'pca'``): ``'pca'`` or
      ``'fnn'``.
    - ``model.n_target_var_threshold``: PCA cumulative-variance threshold
      (e.g. ``0.99``). Required for ``method='pca'`` to actually run.
    - ``model.n_target_fnn_threshold`` (default ``0.01``): FNN drop
      threshold. Used only when ``method='fnn'``.
    - ``model.encoder._target_``: detected as DirectSum if it ends in
      ``DirectSumCouplingEncoder``.
    - ``model.encoder.area_indices``: required (and read) when
      DirectSum. Each area's input gets its own per-area PCA / FNN.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object. NOT MUTATED.
    train_seq : torch.Tensor
        Training delay-embedded sequences of shape
        ``(N_traj, T, D_embed)``. Flattened across (N_traj, T) before
        the eigendecomposition / FNN call.

    Returns
    -------
    InferNTargetDimsResult or None
        ``None`` when no autodim is needed (i.e., ``method='pca'`` AND
        ``n_target_var_threshold is None``). The caller then uses the
        existing cfg ``n_target_dims`` value.

    Raises
    ------
    ValueError
        If ``method`` is not ``'pca'`` or ``'fnn'``.
    """
    method = OmegaConf.select(cfg, "model.n_target_dim_method", default="pca")
    var_threshold = OmegaConf.select(cfg, "model.n_target_var_threshold", default=None)
    fnn_threshold = OmegaConf.select(cfg, "model.n_target_fnn_threshold", default=0.01)
    if method not in ("pca", "fnn"):
        raise ValueError(
            f"model.n_target_dim_method must be 'pca' or 'fnn', got {method!r}"
        )
    is_direct_sum = (
        str(OmegaConf.select(cfg, "model.encoder._target_", default=""))
        .endswith("DirectSumCouplingEncoder")
    )

    if method == "fnn":
        return _infer_fnn(cfg, train_seq, fnn_threshold, is_direct_sum)
    elif var_threshold is not None:
        return _infer_pca(cfg, train_seq, var_threshold, is_direct_sum)
    else:
        # No autodim configured. Caller should leave n_target_dims as-is.
        return None


def _infer_fnn(
    cfg: DictConfig,
    train_seq: torch.Tensor,
    fnn_threshold: float,
    is_direct_sum: bool,
) -> InferNTargetDimsResult:
    """FNN-based dimension estimation. Per-area for DirectSum, global
    otherwise."""
    from JacobianODE.fnn.dim_estimator import fnn_dim_estimate

    flat = train_seq.reshape(-1, train_seq.shape[-1]).to(torch.float64)
    flat_np = flat.cpu().numpy()

    if is_direct_sum:
        area_indices = OmegaConf.to_container(
            cfg.model.encoder.area_indices, resolve=True
        )
        n_target_per_block = []
        for i, idxs in enumerate(area_indices):
            n_k = int(fnn_dim_estimate(
                flat_np[:, idxs], threshold=fnn_threshold,
            ))
            n_target_per_block.append(n_k)
            logger.info(
                f"  area {i}: input_dims={len(idxs)}, "
                f"n_target_dims (FNN)={n_k}"
            )
        total = int(sum(n_target_per_block))
        logger.info(
            f"DirectSum FNN-auto: n_target_dims_per_block={n_target_per_block} "
            f"(total={total}, fnn_threshold={fnn_threshold})"
        )
        return InferNTargetDimsResult(
            n_target_dims_total=total,
            n_target_dims_per_block=list(n_target_per_block),
            pca_cum_var=None,
            method="fnn",
            is_direct_sum=True,
        )

    # Single-encoder global FNN.
    n_target = int(fnn_dim_estimate(flat_np, threshold=fnn_threshold))
    logger.info(
        f"FNN-auto n_target_dims: D_embed={flat_np.shape[-1]}, "
        f"N_samples={flat_np.shape[0]}, fnn_threshold={fnn_threshold}, "
        f"chose n_target_dims={n_target}"
    )
    return InferNTargetDimsResult(
        n_target_dims_total=n_target,
        n_target_dims_per_block=None,
        pca_cum_var=None,
        method="fnn",
        is_direct_sum=False,
    )


def _infer_pca(
    cfg: DictConfig,
    train_seq: torch.Tensor,
    var_threshold: float,
    is_direct_sum: bool,
) -> InferNTargetDimsResult:
    """PCA-based dimension estimation. Per-area for DirectSum, global
    otherwise. Picks the smallest k whose cumulative explained variance
    >= ``var_threshold``."""
    flat = train_seq.reshape(-1, train_seq.shape[-1]).to(torch.float64)

    if is_direct_sum:
        area_indices = OmegaConf.to_container(
            cfg.model.encoder.area_indices, resolve=True
        )
        n_target_per_block = []
        cum_at_pick = []
        for i, idxs in enumerate(area_indices):
            x_area = flat[:, idxs]
            x_area = x_area - x_area.mean(dim=0, keepdim=True)
            cov_a = (x_area.T @ x_area) / (x_area.shape[0] - 1)
            eigvals_a = torch.linalg.eigvalsh(cov_a).flip(0).clamp_min(0.0)
            explained_a = eigvals_a / eigvals_a.sum()
            cum_var_a = explained_a.cumsum(0)
            n_k = int((cum_var_a >= var_threshold).float().argmax().item()) + 1
            n_target_per_block.append(n_k)
            cum_at_pick.append(float(cum_var_a[n_k - 1].item()))
            logger.info(
                f"  area {i}: input_dims={len(idxs)}, n_target_dims={n_k}, "
                f"cum_var_at_pick={cum_at_pick[-1]:.4f}"
            )
        total = int(sum(n_target_per_block))
        logger.info(
            f"DirectSum PCA-auto: n_target_dims_per_block={n_target_per_block} "
            f"(total={total}, threshold={var_threshold})"
        )
        return InferNTargetDimsResult(
            n_target_dims_total=total,
            n_target_dims_per_block=list(n_target_per_block),
            pca_cum_var=cum_at_pick,
            method="pca",
            is_direct_sum=True,
        )

    # Single-encoder global PCA.
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (flat_centered.T @ flat_centered) / (flat_centered.shape[0] - 1)
    # eigh returns ascending eigvals; flip to descending.
    eigvals_asc, _ = torch.linalg.eigh(cov)
    eigvals = eigvals_asc.flip(0).clamp_min(0.0)
    explained = eigvals / eigvals.sum()
    cum_var = explained.cumsum(0)
    _cum = [f"{v:.4f}" for v in cum_var[: min(10, len(cum_var))].tolist()]
    _exp = [f"{v:.4f}" for v in explained[: min(10, len(explained))].tolist()]
    logger.info(
        f"PCA on training delay embeddings: D_embed={flat.shape[-1]}, "
        f"N_samples={flat.shape[0]}"
    )
    logger.info(f"  explained variance (first 10): {_exp}")
    logger.info(f"  cumulative variance (first 10): {_cum}")

    n_target = int((cum_var >= var_threshold).float().argmax().item()) + 1
    logger.info(
        f"PCA-auto n_target_dims: threshold={var_threshold}, "
        f"chose n_target_dims={n_target}"
    )
    cum_at_pick = float(cum_var[n_target - 1].item())
    return InferNTargetDimsResult(
        n_target_dims_total=n_target,
        n_target_dims_per_block=None,
        pca_cum_var=cum_at_pick,
        method="pca",
        is_direct_sum=False,
    )
