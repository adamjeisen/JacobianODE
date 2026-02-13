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

    # Thresholds from Kennel et al. 1992
    rtol = 20.0
    atol = 2.0

    # Lower triangular mask: for dimension d, keep only first d coordinates
    tri_mask = torch.tril(
        torch.ones(n_latent, n_latent, dtype=torch.float32, device=device)
    )
    # batch_masked: (n_latent, batch_size, n_latent)
    batch_masked = tri_mask[:, None, :] * code_batch[None, :, :]

    # Pairwise squared distances for each incremental dimension set
    X_sq = (batch_masked * batch_masked).sum(dim=2, keepdim=True)
    pdist_vector = (
        X_sq
        + X_sq.transpose(1, 2)
        - 2 * torch.bmm(batch_masked, batch_masked.transpose(1, 2))
    )
    all_dists = pdist_vector  # (n_latent, batch_size, batch_size)

    # Characteristic scale per dimension: sqrt(mean variance / d)
    stds = torch.std(batch_masked, dim=1, keepdim=True)  # (n_latent, 1, n_latent)
    all_ra = torch.sqrt(
        (1.0 / torch.arange(1, 1 + n_latent, dtype=torch.float32, device=device))
        * (stds ** 2).sum(dim=2).squeeze(1)
    )  # (n_latent,)

    # Avoid zero distances
    all_dists = torch.clamp(all_dists, min=1e-14)

    # Find k+1 nearest neighbors (smallest distances)
    _, inds = torch.topk(-all_dists, k + 1, dim=-1)
    # inds: (n_latent, batch_size, k+1)

    # Gather neighbor distances at dimension d
    neighbor_dists_d = torch.gather(all_dists, 2, inds)

    # Gather distances at dimension d+1 using neighbors found at dimension d
    neighbor_new_dists = torch.gather(all_dists[1:], 2, inds[:-1])

    # Eq. 4 of Kennel et al.: ratio of distance change
    scaled_dist = torch.sqrt(
        (neighbor_new_dists - neighbor_dists_d[:-1])
        / neighbor_dists_d[:-1]
    )

    # Kennel condition #1: distance ratio exceeds threshold
    is_false_change = scaled_dist > rtol
    # Kennel condition #2: absolute distance exceeds threshold
    is_large_jump = neighbor_new_dists > atol * all_ra[:-1, None, None]

    is_false_neighbor = torch.logical_or(is_false_change, is_large_jump)
    total_false_neighbors = is_false_neighbor.to(torch.int32)[..., 1:(k + 1)]

    # Weight: fraction of true (non-false) neighbors per dimension
    reg_weights = 1 - total_false_neighbors.to(torch.float64).mean(dim=(1, 2))
    reg_weights = torch.nn.functional.pad(reg_weights, (1, 0))  # pad zero for dim 0

    # Average batch activity per latent dimension
    activations_batch_averaged = torch.sqrt(
        torch.mean(code_batch ** 2, dim=0)
    ).to(torch.float64)

    # Weighted L2 activity regularization
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
