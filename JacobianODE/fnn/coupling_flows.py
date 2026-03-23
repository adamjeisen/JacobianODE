"""Affine coupling flow encoder — an invertible (diffeomorphic) neural network.

Implements a RealNVP-style stack of affine coupling layers with fixed random
permutations between layers.  The encoder is dimension-preserving (n_latent ==
n_input) and its exact analytical inverse serves as the decoder.

Reference: Dinh et al., "Density estimation using Real-NVP", ICLR 2017.

Example
-------
>>> encoder = AffineCouplingEncoder(n_input=20, n_coupling_layers=8)
>>> x = torch.randn(4, 50, 20)          # (batch, time, dim)
>>> z = encoder(x)                        # forward / encode
>>> x_rec = encoder.inverse(z)            # exact inverse / decode
>>> assert torch.allclose(x, x_rec, atol=1e-5)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Conditioner MLP (internal helper)
# ---------------------------------------------------------------------------

def _build_conditioner(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    n_hidden_layers: int = 2,
    zero_init: bool = True,
) -> nn.Sequential:
    """Build a small MLP used as the conditioner inside a coupling layer.

    Parameters
    ----------
    input_dim : int
        Number of input features (the "fixed" partition).
    output_dim : int
        Number of output features (2 * transformed_dim for log_s and t).
    hidden_dim : int
        Width of hidden layers.
    n_hidden_layers : int
        Number of hidden layers (minimum 1).
    zero_init : bool
        If True, zero-initialise the last linear layer so the coupling layer
        starts as the identity map.
    """
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(n_hidden_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    last = nn.Linear(hidden_dim, output_dim)
    if zero_init:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Affine Coupling Layer
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer (RealNVP style).

    Splits the last dimension at *split_dim*.  The first ``split_dim`` features
    are kept fixed and fed into a conditioner MLP that predicts ``(log_s, t)``
    for the remaining features.

    Parameters
    ----------
    dim : int
        Total feature dimension.
    split_dim : int
        Number of dimensions in the fixed partition.
    hidden_dim : int
        Conditioner MLP hidden width.
    n_hidden_layers : int
        Conditioner MLP depth.
    scale_activation : str
        ``'tanh'`` bounds log_s via ``scale_clamp * tanh(log_s)``; ``'none'``
        uses raw clamping.
    scale_clamp : float
        Maximum absolute value of log_s.
    zero_init : bool
        Zero-initialise the last conditioner layer (identity-init trick).
    """

    def __init__(
        self,
        dim: int,
        split_dim: int | None = None,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        scale_activation: str = "tanh",
        scale_clamp: float = 3.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if split_dim is None:
            split_dim = dim // 2
        self.dim = dim
        self.split_dim = split_dim
        self.transform_dim = dim - split_dim
        self.scale_activation = scale_activation
        self.scale_clamp = scale_clamp

        self.conditioner = _build_conditioner(
            input_dim=split_dim,
            output_dim=2 * self.transform_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            zero_init=zero_init,
        )

    # ----- helpers -----

    def _compute_st(self, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run conditioner and return (log_s, t)."""
        st = self.conditioner(x_a)
        log_s, t = st.chunk(2, dim=-1)
        if self.scale_activation == "tanh":
            log_s = self.scale_clamp * torch.tanh(log_s)
        else:
            log_s = log_s.clamp(-self.scale_clamp, self.scale_clamp)
        return log_s, t

    # ----- forward / inverse -----

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (..., D)

        Returns
        -------
        y : Tensor (..., D)
        log_det : Tensor (...)   sum of log |ds/dx| over transformed dims
        """
        x_a = x[..., : self.split_dim]
        x_b = x[..., self.split_dim :]
        log_s, t = self._compute_st(x_a)
        y_b = x_b * torch.exp(log_s) + t
        y = torch.cat([x_a, y_b], dim=-1)
        log_det = log_s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Exact analytical inverse.

        Parameters
        ----------
        y : Tensor (..., D)

        Returns
        -------
        x : Tensor (..., D)
        """
        y_a = y[..., : self.split_dim]
        y_b = y[..., self.split_dim :]
        log_s, t = self._compute_st(y_a)  # y_a == x_a
        x_b = (y_b - t) * torch.exp(-log_s)
        return torch.cat([y_a, x_b], dim=-1)


# ---------------------------------------------------------------------------
# Fixed Permutation Layer
# ---------------------------------------------------------------------------

class FixedPermutation(nn.Module):
    """Deterministic random permutation of the feature dimension.

    Uses a local ``torch.Generator`` seeded with *seed* so the global RNG
    state is never touched.

    Parameters
    ----------
    dim : int
        Feature dimension to permute.
    seed : int
        Seed for reproducibility.
    """

    def __init__(self, dim: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(dim, generator=g)
        self.register_buffer("perm", perm)
        self.register_buffer("perm_inv", torch.argsort(perm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm]

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y[..., self.perm_inv]


# ---------------------------------------------------------------------------
# Affine Coupling Encoder (full stack)
# ---------------------------------------------------------------------------

class AffineCouplingEncoder(nn.Module):
    """Invertible encoder built from a stack of affine coupling layers.

    A fixed random permutation is inserted between every pair of consecutive
    coupling layers so that all dimensions eventually act as both conditioner
    and transformed variables.

    The encoder is **dimension-preserving**: ``n_latent == n_input``.

    Parameters
    ----------
    n_input : int
        Feature dimension (injected at runtime from the data pipeline).
    n_coupling_layers : int
        Number of coupling layers.
    hidden_dim : int
        Conditioner MLP hidden width.
    n_hidden_layers : int
        Conditioner MLP depth.
    scale_activation : str
        ``'tanh'`` or ``'none'`` — how to bound log-scale.
    scale_clamp : float
        Maximum absolute value of log-scale.
    zero_init : bool
        Zero-initialise last conditioner layer for identity-init.
    permutation_seed : int
        Base seed for generating the fixed permutations.  Layer *i* uses
        ``permutation_seed + i``.
    """

    def __init__(
        self,
        n_input: int,
        n_coupling_layers: int = 6,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        scale_activation: str = "tanh",
        scale_clamp: float = 3.0,
        zero_init: bool = True,
        permutation_seed: int = 0,
    ) -> None:
        super().__init__()
        self._n_input = n_input
        split_dim = n_input // 2

        self.coupling_layers = nn.ModuleList()
        self.permutations = nn.ModuleList()

        for i in range(n_coupling_layers):
            self.coupling_layers.append(
                AffineCouplingLayer(
                    dim=n_input,
                    split_dim=split_dim,
                    hidden_dim=hidden_dim,
                    n_hidden_layers=n_hidden_layers,
                    scale_activation=scale_activation,
                    scale_clamp=scale_clamp,
                    zero_init=zero_init,
                )
            )
            # Permutation after every layer except the last
            if i < n_coupling_layers - 1:
                self.permutations.append(
                    FixedPermutation(dim=n_input, seed=permutation_seed + i)
                )

    # ----- properties -----

    @property
    def n_latent(self) -> int:
        """Latent dimension (== input dimension for coupling flows)."""
        return self._n_input

    # ----- forward / inverse -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: apply coupling layers with interleaved permutations.

        Parameters
        ----------
        x : Tensor (B, T, D) or (B, D)

        Returns
        -------
        z : Tensor, same shape as x
        """
        z = x
        for i, layer in enumerate(self.coupling_layers):
            z, _ = layer(z)
            if i < len(self.permutations):
                z = self.permutations[i](z)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Decode: apply layers in reverse order (undo permutations first).

        Parameters
        ----------
        z : Tensor (B, T, D) or (B, D)

        Returns
        -------
        x : Tensor, same shape as z
        """
        y = z
        for i in reversed(range(len(self.coupling_layers))):
            if i < len(self.permutations):
                y = self.permutations[i].inverse(y)
            y = self.coupling_layers[i].inverse(y)
        return y

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`inverse`."""
        return self.inverse(z)

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total log |det J| of the forward map (for diagnostics).

        Parameters
        ----------
        x : Tensor (B, T, D) or (B, D)

        Returns
        -------
        log_det : Tensor (B, T) or (B,)
        """
        z = x
        total_log_det = torch.zeros(
            x.shape[:-1], device=x.device, dtype=x.dtype
        )
        for i, layer in enumerate(self.coupling_layers):
            z, ld = layer(z)
            total_log_det = total_log_det + ld
            if i < len(self.permutations):
                z = self.permutations[i](z)
        return total_log_det
