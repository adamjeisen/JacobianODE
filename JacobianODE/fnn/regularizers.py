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

import torch
import torch.nn as nn


def loss_false(code_batch: torch.Tensor, k: int = 1) -> torch.Tensor:
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

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.
    """
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
    all_ra = torch.sqrt(
        (1.0 / torch.arange(1, 1 + n_latent, dtype=torch.float32, device=device))
        * (stds ** 2).sum(dim=2).squeeze(1)
    )  # (n_latent,)
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
    scaled_dist = torch.sqrt(
        torch.clamp(
            (neighbor_new_dists - neighbor_dists_d[:-1]) / neighbor_dists_d[:-1],
            min=0.0,
        )
    )

    # Kennel condition #1: distance ratio exceeds threshold
    is_false_change = scaled_dist > rtol
    # Kennel condition #2: absolute distance exceeds attractor scale threshold
    is_large_jump = neighbor_new_dists > atol * all_ra[:-1, None, None]

    is_false_neighbor = torch.logical_or(is_false_change, is_large_jump)
    total_false_neighbors = is_false_neighbor.to(torch.int32)[..., 1:(k + 1)]
    # total false neighbors has shape (n_latent - 1, batch_size, k)

    # Weight: fraction of true (non-false) neighbors per dimension
    # reg_weights has shape (n_latent,), where the i-th element is the fraction of true (non-false) neighbors in dimension i
    reg_weights = 1 - total_false_neighbors.to(torch.float64).mean(dim=(1, 2))
    reg_weights = torch.nn.functional.pad(reg_weights, (1, 0))  # pad zero for dim 0
    # now reg_weights has shape (n_latent,)

    # RMS activity per latent dimension, matching Gilpin TF implementation
    activations_batch_averaged = torch.sqrt(
        torch.mean(code_batch ** 2, dim=0)
    ).to(torch.float64)

    # Weighted L1 activity regularization
    loss = torch.sum(reg_weights * activations_batch_averaged)

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
