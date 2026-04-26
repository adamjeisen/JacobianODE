"""FNN-based intrinsic-dimensionality estimator.

Single source of truth for the whitened PCA-FNN diagnostic + autodim path.
Wraps :func:`JacobianODE.fnn.regularizers.loss_false` (with ``use_pca=True``,
which post-PCA-rotation rescales each PC to unit variance) and applies a
"stop at minimum" rule to interpret the false-neighbor curve, robust to the
whitening tail that can re-inflate false-neighbor rates past the true
embedding dim.

Convention (matches Kennel et al. 1992 / Gilpin NeurIPS 2020 implementation):
``frac_false[d]`` is the fraction of dim-``d`` nearest neighbors that turn
out NOT to be neighbors at dim ``d+1``. Small ``frac_false[d]`` ⇒ ``d``
coords are sufficient.
"""
from __future__ import annotations

import numpy as np
import torch

from .regularizers import loss_false


def fnn_dim_estimate(
    x: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.01,
    n_samples: int = 2500,
    k: int = 1,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> int:
    """Return the FNN-estimated embedding dimension of ``x``.

    Parameters
    ----------
    x : (N, D) array
        Sample points. Will be PCA-whitened internally before the Kennel
        false-neighbor test (see module docstring for why whitening matters).
    threshold : float, default 0.01
        Classic Kennel cutoff. Smallest ``d`` with ``frac_false[d] <=
        threshold`` is returned.
    n_samples : int, default 2500
        Subsample cap inside ``loss_false`` (FNN is O(N^2 D) memory).
    k : int, default 1
        Nearest-neighbor count for the Kennel test. ``k=1`` is canonical;
        higher k smooths but biases the dim transition outward.
    device : str | torch.device | None
        Where to run the FNN computation. ``None`` keeps the input's device
        (CPU for numpy inputs).
    dtype : torch.dtype, default torch.float64
        Compute dtype. ``float32`` halves the O(N^2 D) memory — needed for
        n_samples >> 2500 on GPU.

    Returns
    -------
    dim : int
        Smallest ``d`` satisfying either:
          (a) ``frac_false[d] <= threshold`` (classic Kennel), or
          (b) ``frac_false[d+1] > frac_false[d]`` (curve stops decreasing —
              handles whitening tails where trailing PCs re-inflate the
              false-neighbor rate past the true embedding dim).
        Falls back to ``D`` if neither condition fires within the available
        embedding dimensions.
    """
    xt = torch.as_tensor(np.asarray(x), dtype=dtype)
    if device is not None:
        xt = xt.to(device)
    _, weights = loss_false(
        xt, k=k, use_pca=True, n_samples=n_samples, return_fnn_weights=True,
    )
    frac = 1.0 - weights.detach().cpu().numpy()
    # loss_false pads weights[0] = 1.0 (no transition before dim 1); blank out.
    frac[0] = np.nan
    D = len(frac)
    for d in range(1, D):
        f = frac[d]
        if f <= threshold:
            return d
        if d + 1 < D and frac[d + 1] > f:
            return d
    return D
