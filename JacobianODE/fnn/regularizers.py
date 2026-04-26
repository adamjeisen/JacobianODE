"""
PyTorch implementation of the false nearest neighbor (FNN) regularizer
and covariance (DeCov) regularizer.

Original TensorFlow implementation: https://github.com/williamgilpin/fnn
Reference: Gilpin, "Deep reconstruction of strange attractors from time series"
           NeurIPS 2020. https://arxiv.org/abs/2002.05909

FNN regularizer based on: Kennel, Brown, and Abarbanel.
           "Determining embedding dimension for phase-space reconstruction
           using a geometrical construction." Phys Rev A, 1992.

DeCov regularizer based on: Cogswell et al. ICLR 2016.
"""
import math

import torch
import torch.nn as nn

def loss_false(
        code_batch: torch.Tensor,
        k: int = 1,
        normalize: bool = False,
        return_fnn_weights: bool = False,
        elementwise_regularization: bool = False,
        use_pca: bool = False,
        n_samples: int | None = None,
        sparsify: bool = False,
    ) -> torch.Tensor:
    """Activity regularizer based on the False-Nearest-Neighbor algorithm.

    For each embedding dimension d (from 1 to n_latent), computes pairwise
    distances using only the first d coordinates. Identifies neighbors that
    become "false" when adding the next coordinate, and uses this to weight
    an L2 activity penalty on each latent dimension.

    Parameters
    ----------
    code_batch : torch.Tensor
        (batch_size, n_latent) tensor of encoded latent representations.
    k : int
        Number of nearest neighbors to consider. Default 1.
    normalize : bool
        Whether to normalize the loss by the variance of the activations.
    return_fnn_weights : bool
        Whether to return the FNN weights.
    elementwise_regularization : bool
        Whether to use elementwise regularization.
        If True, the loss is computed as the sum of the elementwise regularization terms. (E[W A^2])
        If False, the loss is computed as the sum of the FNN weights. (E[W] * E[A^2])
    use_pca : bool
        If True, project onto principal components (right singular vectors) before
        computing the loss, making it rotationally invariant. Default False.
    n_samples : int or None
        If not None, randomly subsample this many points to compute the loss
        (useful to cap O(N^2) pairwise cost). If None, use all points. Default None.
    sparsify : bool
        Whether to sparsify the embedding by penalizing the L1 norm of the embedding.
    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.
    """
    # Optional subsampling
    if n_samples is not None and len(code_batch) > n_samples:
        idx = torch.randperm(len(code_batch), device=code_batch.device)[:n_samples]
        code_batch = code_batch[idx]

    # Optional PCA-whitening: rotate to PC basis, then rescale each PC to unit
    # variance. Whitening matters because Kennel's FNN test (rtol on distance
    # ratios, atol vs characteristic scale) assumes equal-scale coords — a raw
    # PCA rotation keeps large variance differences across PCs, which biases
    # the test toward under-reporting dimension on low-D data.
    if use_pca:
        z_centered = code_batch - code_batch.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(z_centered, full_matrices=False)
        code_batch = z_centered @ Vh.T
        code_batch = code_batch / code_batch.std(dim=0, keepdim=True).clamp(min=1e-12)

    n_batch, n_latent = code_batch.shape
    device = code_batch.device

    # Thresholds from Kennel et al. 1992 / Gilpin (NeurIPS 2020) TF implementation
    rtol = 20.0
    atol = 2.0

    # Lower triangular mask: for dimension d, keep only first d coordinates
    tri_mask = torch.tril(
        torch.ones(n_latent, n_latent, dtype=torch.float32, device=device)
    )
    # batch_masked: (n_latent, batch_size, n_latent)
    # Dim 0: how many dimensions to keep
    # Dim 1: batch index
    # Dim 2: the actual coordinate embedding
    batch_masked = tri_mask[:, None, :] * code_batch[None, :, :]

    # Pairwise squared distances for each incremental dimension set
    # Use the fact that ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    # X_sq computes the squared distance of each point to itself using the coordinate embeddings
    X_sq = (batch_masked * batch_masked).sum(dim=2, keepdim=True) # Dimensions: (n_latent, batch_size, 1)
    pdist_vector = (
        X_sq
        + X_sq.transpose(1, 2)
        - 2 * torch.bmm(batch_masked, batch_masked.transpose(1, 2))
    )
    all_dists = pdist_vector  # (n_latent, batch_size, batch_size)
    # in all_dists, the (i, j, k) element is the distance between the j-th point and the k-th point using i dimensions
    # this is equivalent to $D^2_{abm}$ in the paper where i = a, j = b, k = m

    # Characteristic scale per dimension: sqrt(mean variance / d)
    # stds: (n_latent, 1, n_latent)
    # the first dimension is the number of embedding coordinates to use
    # the last dimension is the actual embedding coordinates (the standard deviation of each coordinate over the batch)
    stds = torch.std(batch_masked, dim=1, keepdim=True)  # (n_latent, 1, n_latent)
    # all_ra = torch.sqrt(
    #     (1.0 / torch.arange(1, 1 + n_latent, dtype=torch.float32, device=device))
    #     * (stds ** 2).sum(dim=2).squeeze(1)
    # )  # (n_latent,)
    all_ra_squared = (1.0 / torch.arange(1, 1 + n_latent, dtype=torch.float32, device=device)) * (stds ** 2).sum(dim=2).squeeze(1)
    # all_ra[i] is the characteristic size of the attractor when using the first i dimensions
    # it is equivalent to $\sqrt{\mathcal{R}^2_i}$ in the paper

    # Avoid zero distances
    all_dists = torch.clamp(all_dists, min=1e-14)

    # Find k+1 nearest neighbors (smallest distances)
    _, inds = torch.topk(-all_dists, k + 1, dim=-1)
    # inds: (n_latent, batch_size, k+1), where the last dimensions contains the indices of the k+1 nearest neighbors

    # Gather neighbor distances at dimension d
    # neighbor_dists_d: (n_latent, batch_size, k+1), where the last dimensions contains the (squared) distances to the k+1 nearest neighbors
    # this is equivalent to $\tilde{D}^2_{abm}$ in the paper where i = a, j = b, k = m
    neighbor_dists_d = torch.gather(all_dists, 2, inds)

    # Gather distances at dimension d+1 using neighbors found at dimension d
    # neighbor_new_dists: (n_latent - 1, batch_size, k+1)
    # where the last dimensions contains the (squared) distances (in the current dimension) to the k+1 nearest neighbors of the PREVIOUS dimension
    # this is equivalent to $\tilde{D}'^2_{abm}$ in the paper where i = a, j = b, k = m
    neighbor_new_dists = torch.gather(all_dists[1:], 2, inds[:-1])

    # Eq. 4 of Kennel et al.: ratio of distance change, matching the Gilpin TF implementation
    # scaled_dist: (n_latent - 1, batch_size, k+1), sqrt of the normalized distance ratio
    # scaled_dist = torch.sqrt(
    #     torch.clamp(
    #         (neighbor_new_dists - neighbor_dists_d[:-1]) / neighbor_dists_d[:-1],
    #         min=0.0,
    #     )
    # )
    scaled_dist_squared = torch.clamp(
        (neighbor_new_dists - neighbor_dists_d[:-1]) / neighbor_dists_d[:-1],
        min=0.0,
    )

    # Kennel condition #1: distance ratio exceeds threshold
    # is_false_change = scaled_dist > rtol
    is_false_change = scaled_dist_squared > rtol**2
    # Kennel condition #2: absolute distance exceeds attractor scale threshold
    # is_large_jump = neighbor_new_dists > atol * all_ra[:-1, None, None]
    is_large_jump = neighbor_new_dists > atol**2 * all_ra_squared[:-1, None, None]

    is_false_neighbor = torch.logical_or(is_false_change, is_large_jump)
    total_false_neighbors = is_false_neighbor.to(torch.int32)[..., 1:(k + 1)]
    # total false neighbors has shape (n_latent - 1, batch_size, k)

    # Weight: fraction of true (non-false) neighbors per dimension
    if elementwise_regularization:
        # reg_weights has shape (batch_size, n_latent), where the (i, j)-th element is the fraction of true (non-false) neighbors in dimension j for the i-th batch
        reg_weights = (1 - total_false_neighbors).to(torch.float64).mean(dim=-1).transpose(0, 1)
        reg_weights = torch.nn.functional.pad(reg_weights, (1, 0))  # pad zero for dim 0
    else:
        # reg_weights has shape (n_latent,), where the i-th element is the fraction of true (non-false) neighbors in dimension i
        reg_weights = 1 - total_false_neighbors.to(torch.float64).mean(dim=(1, 2))
        reg_weights = torch.nn.functional.pad(reg_weights, (1, 0))  # pad zero for dim 0
        # now reg_weights has shape (n_latent,)

    # elementwise_reg = (1 - total_false_neighbors).to(torch.float64).mean(dim=-1).transpose(0, 1)
    # torch.sum(elementwise_reg*(code_batch[...,1:]**2))/(code_batch[..., 1:]**2).sum()

    # Weighted L1 activity regularization
    if elementwise_regularization:
        # take the mean over batches and sum over latents
        # loss_j = (1/B) \sum_i ^ B W_ij * A_ij^2
        if sparsify:
            loss = (reg_weights * (code_batch).abs()).mean(dim=0).sum()
        else:
            loss = (reg_weights * (code_batch**2)).mean(dim=0).sum()
    else:
        # take the mean activity over batches and sum over latents
        if sparsify:
            activations_batch_averaged = torch.mean(code_batch.abs(), dim=0)
        else:
            activations_batch_averaged = torch.mean(code_batch ** 2, dim=0)
        loss = torch.sum(reg_weights * activations_batch_averaged)

    if normalize:
        # return loss.float() / (n_latent * code_batch.var())
        epsilon = 1e-8
        if elementwise_regularization:
            if sparsify:
                denom = code_batch.abs().mean(dim=0).sum() + epsilon
            else:
                denom = (code_batch**2).mean(dim=0).sum() + epsilon
        else:
            denom = activations_batch_averaged.sum() + epsilon
        if return_fnn_weights:
            return loss.float() / denom, reg_weights
        else:
            return loss.float() / denom
    else:
        if return_fnn_weights:
            return loss.float(), reg_weights
        else:
            return loss.float()


def loss_cov(a: torch.Tensor, whiten: bool = False) -> torch.Tensor:
    """Covariance loss to orthogonalize activations.

    Penalizes off-diagonal elements of the batch covariance matrix,
    encouraging decorrelated latent features.

    Parameters
    ----------
    a : torch.Tensor
        (batch_size, n_features) layer activations.
    whiten : bool
        Whether to standardize features before computing covariance.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.

    Reference
    ---------
    Cogswell et al. "Reducing Overfitting in Deep Networks by Decorrelating
    Representations." ICLR 2016.
    """
    a_mean = a.mean(dim=0)
    n_batch = a.shape[0]

    aw = a - a_mean
    if whiten:
        a_std = a.std(dim=0)
        aw = aw / a_std

    cov = (1.0 / n_batch) * (aw.T @ aw)

    loss = 0.5 * (
        torch.norm(cov) ** 2 - torch.norm(torch.diagonal(cov)) ** 2
    )

    return loss


class FNN(nn.Module):
    """Activity regularizer that penalizes false nearest neighbors.

    Parameters
    ----------
    strength : float
        Relative strength of the regularizer.
    k : int
        Number of neighbors for distance calculation (traditionally 1).
    """

    def __init__(self, strength: float, k: int = 1):
        super().__init__()
        self.strength = strength
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strength * loss_false(x, k=self.k)


class DeCov(nn.Module):
    """Activity regularizer that enforces orthogonality via covariance loss.

    Parameters
    ----------
    strength : float
        Relative strength of the regularizer.
    """

    def __init__(self, strength: float):
        super().__init__()
        self.strength = strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strength * loss_cov(x)

class Amplification(nn.Module):
    """Activity regularizer that penalizes noise amplification in delay embeddings.

    Measures how much noise gets amplified in the embedding space by comparing
    the variance of neighboring trajectories over time with the distance
    between neighbors in the embedding space. Encourages embeddings where
    nearby points remain close over time.

    Parameters
    ----------
    strength : float
        Relative strength of the regularizer.
    n_neighbors : int
        Number of nearest neighbors to consider.
    max_T : int
        Maximum number of time steps to look ahead.
    normalize : bool
        Whether to normalize the amplification by the sum of 1/eps_k.
    epsilon : float
        Small constant to avoid division by zero.
    """

    def __init__(self, strength: float, n_neighbors: int = 10, max_T: int = 5,
                 normalize: bool = False, epsilon: float = 1e-8):
        super().__init__()
        self.strength = strength
        self.n_neighbors = n_neighbors
        self.max_T = max_T
        self.normalize = normalize
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the amplification regularization loss.

        Parameters
        ----------
        x : torch.Tensor
            (batch_size, seq_len, embedding_dim) embedding sequences.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value.
        """
        return self.strength * loss_amplification(
            x,
            n_neighbors=self.n_neighbors,
            max_T=self.max_T,
            normalize=self.normalize,
            epsilon=self.epsilon,
        )


def _knn_indices_sq_euclidean_batched(
    emb: torch.Tensor,
    k: int,
    batch_size: int,
) -> torch.Tensor:
    """k nearest neighbor indices (rows of ``emb``) without an :math:`N \\times N` distance matrix.

    Uses squared Euclidean distance; ordering matches Euclidean ``cdist`` for neighbor sets
    (ties may order differently).

    Parameters
    ----------
    emb : torch.Tensor
        ``(n_pts, d)`` embedding vectors.
    k : int
        Number of neighbors per row (including the point itself if it is among the smallest).
    batch_size : int
        Number of query rows per block. Peak extra memory is ``O(batch_size * n_pts)`` floats.

    Returns
    -------
    torch.Tensor
        ``(n_pts, k)`` ``long`` indices into rows of ``emb``.
    """
    n_pts, _d = emb.shape
    device = emb.device
    out = torch.empty((n_pts, k), dtype=torch.long, device=device)
    emb_t = emb.T.contiguous()
    all_sq = (emb * emb).sum(dim=1)
    for start in range(0, n_pts, batch_size):
        end = min(start + batch_size, n_pts)
        q = emb[start:end]
        q_sq = (q * q).sum(dim=1, keepdim=True)
        dist2 = q_sq + all_sq.unsqueeze(0) - 2.0 * torch.mm(q, emb_t)
        dist2 = torch.clamp(dist2, min=0.0)
        _, idx = torch.topk(dist2, k, largest=False, dim=1)
        out[start:end] = idx
    return out


def loss_amplification(
    embedding: torch.Tensor,
    data: torch.Tensor | None = None,
    n_neighbors: int = 10,
    max_T: int = 5,
    normalize: bool = False,
    epsilon: float = 1e-8,
    knn_batch_size: int | None = None,
) -> torch.Tensor:
    """Compute noise amplification (sigma) for a time-delay embedding.

    Fully differentiable w.r.t. ``embedding`` (and ``data`` if provided).
    Neighbor *selection* is performed under ``torch.no_grad`` (discrete
    operation), but all subsequent distance and variance computations
    maintain the computation graph.

    Parameters
    ----------
    embedding : torch.Tensor
        (..., T, latent_dim) embedding time series.  Leading batch/trial
        dimensions are flattened automatically.
    data : torch.Tensor, optional
        (..., T, D) raw time series aligned with *embedding*.  If ``None``,
        *embedding* is used as both the neighbor space and the data space.
    n_neighbors : int
        Number of nearest neighbors (including self).
    max_T : int
        Number of time steps to look ahead.
    normalize : bool
        Divide sigma by sum(1 / eps_k).
    epsilon : float
        Small constant to avoid division by zero.
    knn_batch_size : int, optional
        Rows per block when finding k-NN in embedding space.  Smaller values use less
        peak GPU memory (distance blocks are ``(batch, n_pts)`` instead of ``(n_pts, n_pts)``).
        Default picks a batch size targeting ~128 MiB per block from ``n_pts``.

    Notes
    -----
    Pairwise neighbor spread ``eps_k`` uses the identity
    :math:`\\sum_{i,j}\\|n_i-n_j\\|^2 = 2K\\sum_k\\|n_k\\|^2 - 2\\|\\sum_k n_k\\|^2`
    to avoid materializing an ``(n_\\text{pts}, K, K, D)`` difference tensor.

    Returns
    -------
    sigma : torch.Tensor
        Scalar noise-amplification metric.
    """
    if data is None:
        data = embedding

    # Handle complex embeddings
    if torch.is_complex(embedding):
        embedding = torch.cat([embedding.real, embedding.imag], dim=-1)

    # Build T-shifted data slices and truncated embedding  ──────────────
    # data: (..., T_total, D),  embedding: (..., T_total, latent_dim)
    data_ahead = torch.stack(
        [data[..., t:-(max_T - t), :].reshape(-1, data.shape[-1])
         for t in range(max_T)],
        dim=0,
    )  # (max_T, n_pts, D)

    emb_flat = embedding[..., :-max_T, :].reshape(
        -1, embedding.shape[-1]
    )  # (n_pts, latent_dim)

    n_pts = emb_flat.shape[0]
    if knn_batch_size is None:
        # ~128 MiB float32 budget for one (batch, n_pts) distance block
        _target_block_bytes = 128 * 1024 * 1024
        knn_batch_size = max(
            1,
            min(8192, _target_block_bytes // max(1, n_pts * 4)),
        )

    # k-NN in embedding space (discrete selection, no grad) ─────────────
    with torch.no_grad():
        indices = _knn_indices_sq_euclidean_batched(
            emb_flat, n_neighbors, knn_batch_size,
        )                                                  # (n_pts, K)

    # eps_k: mean pairwise squared distance among neighbors ─────────────
    neighbors = emb_flat[indices]                          # (n_pts, K, latent_dim)
    K = n_neighbors
    # sum_{i,j} ||n_i - n_j||^2 = 2K * sum_k ||n_k||^2 - 2 ||sum_k n_k||^2  (no K×K×D tensor)
    sum_sq_norms = (neighbors * neighbors).sum(dim=(1, 2))
    sum_vec = neighbors.sum(dim=1)
    sq_norm_sum_vec = (sum_vec * sum_vec).sum(dim=-1)
    pairwise_sq_sum = 2.0 * K * sum_sq_norms - 2.0 * sq_norm_sum_vec
    eps_k = pairwise_sq_sum / (K * (K - 1) * emb_flat.shape[-1])

    # E_k(T): neighbor-variance of data T steps ahead ──────────────────
    E_k_list: list[torch.Tensor] = []
    for t in range(max_T):
        data_t_nbrs = data_ahead[t][indices]               # (n_pts, K, D)
        mu = data_t_nbrs.mean(dim=1, keepdim=True)         # (n_pts, 1, D)
        E_kT = (data_t_nbrs - mu).pow(2).mean(dim=1).sum(dim=-1)  # (n_pts,)
        E_k_list.append(E_kT)
    E_k = torch.stack(E_k_list, dim=0)                     # (max_T, n_pts)

    # sigma ─────────────────────────────────────────────────────────────
    sig = (E_k / (eps_k + epsilon)).mean()

    if normalize:
        sig = sig / (1.0 / (eps_k + epsilon)).mean()

    return sig


def tangent_space_entropy(
    dz: torch.Tensor,
    jacobians: torch.Tensor,
    mode: str = "quadratic",
) -> torch.Tensor:
    """Tangent space entropy loss from pre-computed Jacobians and velocities.

    Projects latent velocity ``dz`` onto the left singular vectors of the
    (detached) Jacobians, then minimises the entropy of the per-direction
    energy distribution so that dynamics concentrate along fewer intrinsic
    directions.

    Gradients flow through ``dz`` only — Jacobians are detached before SVD.

    Parameters
    ----------
    dz : torch.Tensor
        Latent velocities of shape ``(M, d)``.
    jacobians : torch.Tensor
        Jacobian matrices of shape ``(M, d_out, d_in)``.  Detached
        internally before SVD.
    mode : str
        Entropy formula: ``'shannon'``, ``'quadratic'``, or ``'renyi_half'``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    K = min(jacobians.shape[-2], jacobians.shape[-1])

    with torch.no_grad():
        U, _, _ = torch.linalg.svd(jacobians.detach(), full_matrices=False)
        # U: (M, d_out, K)

    # Project dz onto U columns: (M, K) = bmm(U^T, dz)
    projections = torch.bmm(
        U.transpose(-2, -1), dz.unsqueeze(-1)
    ).squeeze(-1)
    squared_projections = projections ** 2

    # Energy per dimension, averaged over batch
    E = squared_projections.mean(dim=0)  # (K,)
    p = E / (E.sum() + 1e-10)

    # Entropy
    if mode == "shannon":
        eps = 1e-10
        entropy = -(p * torch.log(p + eps)).sum()
        max_ent = math.log(K) if K > 1 else 1.0
        loss = entropy / max_ent
    elif mode == "quadratic":
        loss = 1.0 - (p ** 2).sum()
    elif mode == "renyi_half":
        eps = 1e-10
        loss = 2.0 * torch.log(torch.sqrt(p + eps).sum() + eps)
    else:
        raise ValueError(
            f"Unknown tangent space entropy mode: {mode!r}. "
            f"Choose from 'shannon', 'quadratic', 'renyi_half'."
        )
    return loss