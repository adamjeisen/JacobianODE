"""Additive coupling flow encoder — an invertible (diffeomorphic) neural network.

Implements a NICE-style stack of additive coupling layers (volume-preserving,
det J = 1) with either fixed random permutations or learnable Cayley
orthogonal mixers between layers. The encoder is dimension-preserving
(n_latent == n_input) and its exact analytical inverse serves as the decoder.

Reference: Dinh et al., "NICE: Non-linear Independent Components Estimation",
ICLR Workshop 2015.

Example
-------
>>> encoder = CouplingEncoder(n_input=20, n_coupling_layers=8)
>>> x = torch.randn(4, 50, 20)          # (batch, time, dim)
>>> z = encoder(x)                        # forward / encode
>>> x_rec = encoder.inverse(z)            # exact inverse / decode
>>> assert torch.allclose(x, x_rec, atol=1e-5)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_conditioner(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    n_hidden_layers: int = 2,
    zero_init: bool = True,
    near_identity_std: float = 0.0,
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
        starts as the identity map. Ignored when ``near_identity_std > 0``.
    near_identity_std : float
        If > 0, initialise the last layer's weight from ``N(0, near_identity_std)``
        and zero the bias instead of a strict zero-init. The coupling layer is
        then *approximately* the identity at init (output magnitude scales with
        near_identity_std × ‖hidden activations‖), which gives the conditioner's
        hidden weights a non-zero gradient on step 1 — they are stuck for one
        step under strict zero_init. Default 0.0 preserves the strict zero-init
        behaviour. Only takes effect when ``zero_init=True``.
    """
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(n_hidden_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    last = nn.Linear(hidden_dim, output_dim)
    if zero_init:
        if near_identity_std > 0:
            nn.init.normal_(last.weight, mean=0.0, std=near_identity_std)
        else:
            nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class AdditiveCouplingLayer(nn.Module):
    """Additive coupling layer (NICE-style) — volume-preserving by construction.

    Splits the last dimension at *split_dim*.  The first ``split_dim`` features
    are kept fixed and fed into a conditioner MLP that predicts a translation
    ``t`` for the remaining features.

    Forward:  ``y_a = x_a``,  ``y_b = x_b + t(x_a)``
    Inverse:  ``x_a = y_a``,  ``x_b = y_b - t(y_a)``

    The Jacobian is lower-triangular with ones on the diagonal, so
    ``det(J) = 1`` always.  This makes the layer volume-preserving regardless
    of the conditioner complexity.

    Parameters
    ----------
    dim : int
        Total feature dimension.
    split_dim : int or None
        Number of dimensions in the fixed partition (default: ``dim // 2``).
    hidden_dim : int
        Conditioner MLP hidden width.
    n_hidden_layers : int
        Conditioner MLP depth.
    zero_init : bool
        Zero-initialise the last conditioner layer so the coupling starts as
        the identity map.
    """

    def __init__(
        self,
        dim: int,
        split_dim: int | None = None,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
    ) -> None:
        super().__init__()
        if split_dim is None:
            split_dim = dim // 2
        self.dim = dim
        self.split_dim = split_dim
        self.transform_dim = dim - split_dim

        # Translation-only conditioner: output_dim = transform_dim (not 2x)
        self.conditioner = _build_conditioner(
            input_dim=split_dim,
            output_dim=self.transform_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            zero_init=zero_init,
            near_identity_std=near_identity_std,
        )

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
        log_det : Tensor (...)   always zero (volume-preserving)
        """
        x_a = x[..., : self.split_dim]
        x_b = x[..., self.split_dim :]
        t = self.conditioner(x_a)
        y_b = x_b + t
        y = torch.cat([x_a, y_b], dim=-1)
        log_det = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
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
        t = self.conditioner(y_a)  # y_a == x_a
        x_b = y_b - t
        return torch.cat([y_a, x_b], dim=-1)


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


class FixedOrthogonal(nn.Module):
    """Deterministic orthogonal linear transform on the feature dim.

    Stores a fixed orthogonal matrix ``Q`` of shape ``(D, D)`` in a buffer.
    Forward applies ``Q`` along the last dim of the input; inverse uses
    ``Q^T``. Volume-preserving (``|det Q| = 1``), so composes cleanly with
    additive coupling layers.

    The matrix is *re-orthogonalised* once at construction via QR (with
    sign-corrected diagonal) so that ``Q Q^T = I`` to float-precision
    regardless of small numerical drift in the input.

    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix of shape ``(D, D)``. Must be approximately orthogonal
        (``||M M^T - I||_F < 1e-2``); reorthogonalised internally.
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"FixedOrthogonal expects a square matrix, got shape {tuple(matrix.shape)}"
            )
        D = matrix.shape[0]
        m64 = matrix.detach().double()
        err = torch.linalg.norm(m64 @ m64.T - torch.eye(D, dtype=torch.float64)).item()
        if err > 1e-2:
            raise ValueError(
                f"FixedOrthogonal input matrix not orthogonal: ||M M^T - I||_F = {err:.3g}"
            )
        # Re-orthogonalise via QR to remove numerical drift; preserve
        # column-sign convention so det(Q) is determined by the input,
        # not by QR's arbitrary sign choice.
        Q, R = torch.linalg.qr(m64)
        sign = torch.diag(R).sign()
        sign[sign == 0] = 1.0
        Q = Q * sign[None, :]
        Q = Q.float()
        self.register_buffer("matrix", Q)
        self.register_buffer("matrix_T", Q.T.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Q along last dim: (Q @ x_last) for each row in leading dims.
        # Equivalent to x @ Q^T.
        return x @ self.matrix_T

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.matrix


class CayleyOrthogonal(nn.Module):
    """Learnable orthogonal mixing layer parameterised by the Cayley transform.

    Drop-in replacement for :class:`FixedPermutation` — same forward/inverse
    interface, same volume-preservation (``|det Q| = 1``), but the rotation
    is **learnable** rather than a fixed permutation. This gives the
    encoder a smooth, gradient-trainable way to discover non-axis-aligned
    rotations between coupling layers.

    Parameterisation:
        ``A = U - U^T``  (skew-symmetric, built from the strict
        upper triangle of the parameter ``U``)
        ``Q = (I - A) (I + A)^{-1}``  (Cayley transform → orthogonal SO(n))

    Properties:
        * ``Q^{-1} = Q^T`` exactly (orthogonal),
        * ``det Q = +1`` (rotation, not reflection),
        * ``Q = I`` when ``U = 0``: layer is *exactly* the identity at
          init, so a model trained without Cayley layers is a valid
          starting point for the Cayley variant (drop-in upgrade).

    Numerical considerations:
        * The transform has a singularity at ``A = -I`` (rotations of 180°
          along some axis). For trained encoders that drift smoothly from
          ``U = 0`` this is not reached in practice.
        * Implemented via ``torch.linalg.solve`` for stability of the
          ``(I + A)^{-1}`` term.

    Parameters
    ----------
    dim : int
        Feature dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        # Free upper-triangle parameter; we project to skew-symmetric in
        # forward by computing (U - U^T) which zeroes the diagonal and
        # the strict-lower triangle automatically.
        self.U = nn.Parameter(torch.zeros(dim, dim))

    def _orthogonal(self) -> torch.Tensor:
        # Build skew-symmetric A from the strict upper triangle of U.
        U_strict = torch.triu(self.U, diagonal=1)
        A = U_strict - U_strict.transpose(-1, -2)
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        # Cayley: Q = (I - A) @ (I + A)^{-1}.
        # Use solve to avoid an explicit matrix inverse:
        #     Q = (I - A) @ X   with   (I + A) X = I   ⇒   X = solve(I+A, I)
        # but the equivalent one-line form via the system
        #     (I + A) Y = (I - A)   ⇒   Y = solve(I+A, I-A) = (I+A)^{-1} (I-A)
        # gives Y, not Q. We want Q = (I-A)(I+A)^{-1} = Y^T (since for
        # skew A: (I+A)^T = I-A and (I-A)^T = I+A, so Y^T applied to a
        # symmetric quantity gives... actually it's cleaner to compute
        # solve and then transpose using A's skew-symmetry:
        #     Q^T = (I+A)^{-T} (I-A)^T = (I-A)^{-1} (I+A) = solve(I-A, I+A)
        # so Q = solve(I-A, I+A)^T.
        Q_T = torch.linalg.solve(I - A, I + A)
        return Q_T.transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self._orthogonal()
        # Apply rotation to the last dim of x: y = Q x_last → x @ Q^T.
        return x @ Q.transpose(-1, -2)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        Q = self._orthogonal()
        # Q^{-1} = Q^T, so x = Q^T y_last → y @ Q.
        return y @ Q
class CouplingEncoder(nn.Module):
    """Invertible encoder built from additive (NICE-style) coupling layers.

    Each layer is an :class:`AdditiveCouplingLayer` (volume-preserving by
    construction, det J = 1, identity at init when ``zero_init=True``).
    Between consecutive coupling layers a mixing operator is applied: either
    a :class:`FixedPermutation` (default) or, when ``use_cayley_perms=True``,
    a learnable :class:`CayleyOrthogonal` rotation that initialises to the
    identity.

    Parameters
    ----------
    n_input : int
        Feature dimension (injected at runtime from the data pipeline).
    n_coupling_layers : int
        Number of additive coupling layers.
    hidden_dim : int
        Conditioner MLP hidden width.
    n_hidden_layers : int
        Conditioner MLP depth.
    zero_init : bool
        Zero-initialise the last conditioner layer for identity-at-init.
    near_identity_std : float
        If > 0, initialise the last conditioner layer with normal noise of
        this std instead of zeros (small departure from identity at init).
    permutation_seed : int
        Base seed for fixed permutations. Layer *i* uses
        ``permutation_seed + i``. Ignored when ``use_cayley_perms=True``.
    final_perm_identity : bool
        If True, append a final FixedPermutation chosen so the composition
        of all permutations is the identity. Combined with ``zero_init=True``
        this makes the whole encoder the identity at init, so
        ``z[..., :n_target_dims] == x[..., :n_target_dims]`` — the
        n_target_dims most recent observations land in the dynamic subspace
        deterministically, independent of the permutation seed.
    use_cayley_perms : bool
        If True, use learnable :class:`CayleyOrthogonal` mixers between
        coupling layers instead of fixed random permutations. Cayley layers
        initialise to the identity, so an encoder built with this flag is
        exactly equivalent to one without it at init — gradient then has
        freedom to learn arbitrary inter-layer rotations.
    """

    def __init__(
        self,
        n_input: int,
        n_coupling_layers: int = 8,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
        permutation_seed: int = 0,
        final_perm_identity: bool = False,
        use_cayley_perms: bool = False,
        **unused_kwargs,
    ) -> None:
        super().__init__()
        self._n_input = n_input
        self._use_cayley_perms = use_cayley_perms
        split_dim = n_input // 2

        self.coupling_layers = nn.ModuleList()
        self.permutations = nn.ModuleList()

        for i in range(n_coupling_layers):
            self.coupling_layers.append(
                AdditiveCouplingLayer(
                    dim=n_input,
                    split_dim=split_dim,
                    hidden_dim=hidden_dim,
                    n_hidden_layers=n_hidden_layers,
                    zero_init=zero_init,
                    near_identity_std=near_identity_std,
                )
            )
            if i < n_coupling_layers - 1:
                if use_cayley_perms:
                    self.permutations.append(CayleyOrthogonal(dim=n_input))
                else:
                    self.permutations.append(
                        FixedPermutation(dim=n_input, seed=permutation_seed + i)
                    )

        self.final_permutation: FixedPermutation | None = None
        if final_perm_identity:
            if use_cayley_perms:
                # Cayley layers init to identity, so the composition is
                # identity at init; final perm is itself the identity.
                fp = FixedPermutation(dim=n_input, seed=0)
                fp.perm.copy_(torch.arange(n_input))
                fp.perm_inv.copy_(torch.arange(n_input))
                self.final_permutation = fp
            else:
                # R = composition of inter-layer random perms.
                idx = torch.arange(n_input)
                for p in self.permutations:
                    idx = idx[p.perm]
                # final_perm = R^{-1}, so at init the full composition is identity.
                inv = torch.argsort(idx)
                fp = FixedPermutation(dim=n_input, seed=0)
                fp.perm.copy_(inv)
                fp.perm_inv.copy_(torch.argsort(inv))
                self.final_permutation = fp

    # ----- properties -----

    @property
    def n_latent(self) -> int:
        """Latent dimension (== input dimension for coupling flows)."""
        return self._n_input

    # ----- forward / inverse -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: coupling → permutation, repeated.

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
        if self.final_permutation is not None:
            z = self.final_permutation(z)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Decode: undo all layers in reverse order.

        Parameters
        ----------
        z : Tensor (B, T, D) or (B, D)

        Returns
        -------
        x : Tensor, same shape as z
        """
        y = z
        if self.final_permutation is not None:
            y = self.final_permutation.inverse(y)
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
        if self.final_permutation is not None:
            z = self.final_permutation(z)
        return total_log_det


class DirectSumCouplingEncoder(nn.Module):
    """Direct-sum composition of :class:`CouplingEncoder`s — one per subsystem.

    Partitions the input axis into N subsystems via ``area_indices`` (a list
    of index lists, one per subsystem). Each subsystem is encoded by its
    own :class:`CouplingEncoder` independently; the per-area latent outputs
    are then reordered into a dyn-first-then-null layout so the combined
    dynamic subspace is contiguous at positions ``[0, sum(n_target_dims_per_block))``.

    The encoder Jacobian ``dz_area_i/dx_area_j`` is zero for ``i != j`` by
    construction — the only cross-area information flow possible downstream
    must come from the dynamics MLP, not from encoder mixing. This is the
    structural property that enables clean controllability analyses in the
    latent space.

    Forward path::

        x (..., n_input)
          → gather per-area via area_indices → (x_0, x_1, ..., x_{N-1})
          → each x_i through its CouplingEncoder → (z_0_block, ..., z_{N-1}_block)
          → each z_i_block split into [dyn_i (size k_i) ‖ null_i (size n_i - k_i)]
          → concatenate [dyn_0, ..., dyn_{N-1}, null_0, ..., null_{N-1}]
          → z (..., n_input)  with z[..., :sum(k_i)] = all dyn parts.

    Inverse (decode) path reverses this exactly: un-group z back to per-area
    ``[dyn_i ‖ null_i]`` layout, apply each sub-encoder's inverse, and
    scatter each area's output back to its original input indices. The
    decoder sees the *same units* the sub-encoder produced at forward time
    (same unit ordering), and the reconstructed x lands at the same input
    positions it came from.

    Parameters
    ----------
    area_indices : list[list[int]]
        Partition of the input axis into subsystems. Each sublist is the
        set of input indices for one area. Every index in
        ``[0, sum(len(a) for a in area_indices))`` must appear exactly once;
        indices within a sublist need not be contiguous. Example for the
        WMTask 128-D biological RNN (N1=N2=64): ``[[0..63], [64..127]]``.
    n_target_dims_per_block : list[int]
        Per-area size of the dynamic subspace. ``sum(n_target_dims_per_block)``
        is the total dynamic-subspace dim that downstream code (e.g.
        ``LitLatentJacobianODE._split_latent``) will slice.
    n_coupling_layers, hidden_dim, n_hidden_layers, zero_init,
    near_identity_std, final_perm_identity, use_cayley_perms
        Shared across all sub-encoders — see :class:`CouplingEncoder` for
        semantics. The "same MLP" invariant (every sub-encoder uses identical
        conditioner MLP hyperparameters) is enforced by passing the same
        kwargs to each sub-encoder.
    permutation_seed_base : int
        Base seed; sub-encoder *i* uses ``permutation_seed_base + i`` so each
        area gets an independent, deterministic random inter-layer
        permutation sequence (ignored when ``use_cayley_perms=True``).
    """

    def __init__(
        self,
        area_indices: list[list[int]],
        n_target_dims_per_block: list[int],
        # Shared MLP / coupling settings — mirror CouplingEncoder
        n_coupling_layers: int = 8,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
        permutation_seed_base: int = 0,
        # routing at init
        final_perm_identity: bool = False,
        # learnable orthogonal mixing
        use_cayley_perms: bool = False,
        **unused_kwargs,
    ) -> None:
        super().__init__()
        if len(area_indices) == 0:
            raise ValueError("area_indices must contain at least one area.")
        if len(area_indices) != len(n_target_dims_per_block):
            raise ValueError(
                f"area_indices has {len(area_indices)} areas but "
                f"n_target_dims_per_block has {len(n_target_dims_per_block)}."
            )
        block_sizes = [len(a) for a in area_indices]
        for i, (k, n) in enumerate(zip(n_target_dims_per_block, block_sizes)):
            if not (0 <= k <= n):
                raise ValueError(
                    f"n_target_dims_per_block[{i}]={k} must be in [0, {n}] "
                    f"(area size)."
                )
        # Partition invariant: every input index appears exactly once.
        flat = [int(i) for a in area_indices for i in a]
        if sorted(flat) != list(range(len(flat))):
            raise ValueError(
                "area_indices must partition [0, n_input): every index in "
                "[0, sum(block_sizes)) must appear exactly once across the "
                "sublists (no gaps, no duplicates)."
            )

        self._n_input = len(flat)
        self._k_per_block = tuple(int(k) for k in n_target_dims_per_block)
        self._block_sizes = tuple(block_sizes)
        self._n_areas = len(area_indices)

        # Store per-area index lists as LongTensor buffers so gather/scatter
        # runs on the model's device and survives state_dict save/load.
        for i, a in enumerate(area_indices):
            self.register_buffer(
                f"_area_idx_{i}", torch.as_tensor(a, dtype=torch.long)
            )

        # Shared hyperparameters — identical across sub-encoders except for
        # (n_input, permutation_seed). The "same MLP" invariant lives here.
        shared = dict(
            n_coupling_layers=n_coupling_layers,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            zero_init=zero_init,
            near_identity_std=near_identity_std,
            final_perm_identity=final_perm_identity,
            use_cayley_perms=use_cayley_perms,
        )
        self.blocks = nn.ModuleList([
            CouplingEncoder(
                n_input=n, permutation_seed=permutation_seed_base + i, **shared
            )
            for i, n in enumerate(block_sizes)
        ])

    # ----- helpers -----

    def _area_idx(self, i: int) -> torch.Tensor:
        return getattr(self, f"_area_idx_{i}")

    # ----- properties -----

    @property
    def n_latent(self) -> int:
        """Total latent dim — equals total input dim (each sub-encoder is D_i → D_i)."""
        return self._n_input

    @property
    def n_target_dims(self) -> int:
        """Total dynamic-subspace dim across all areas."""
        return sum(self._k_per_block)

    # ----- forward / inverse -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: gather per-area → per-area encode → reorder dyn-first.

        Parameters
        ----------
        x : Tensor (..., n_input)

        Returns
        -------
        z : Tensor, same shape as x, with layout
            ``[dyn_0, dyn_1, ..., dyn_{N-1}, null_0, null_1, ..., null_{N-1}]``.
        """
        # Per-area gather along last dim.
        xs = [x.index_select(-1, self._area_idx(i)) for i in range(self._n_areas)]
        # Per-area encode.
        zs = [blk(xi) for blk, xi in zip(self.blocks, xs)]
        # Group dyn parts, then null parts.
        dyn_parts = [z[..., :k] for z, k in zip(zs, self._k_per_block)]
        null_parts = [z[..., k:] for z, k in zip(zs, self._k_per_block)]
        return torch.cat(dyn_parts + null_parts, dim=-1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Decode: un-group z → per-area decode → scatter to original indices.

        Parameters
        ----------
        z : Tensor (..., n_input) in the dyn-first layout produced by :meth:`forward`.

        Returns
        -------
        x : Tensor, same shape as z, with each area's reconstructed values
            placed at the input indices they came from.
        """
        # 1) Un-group back to per-area [dyn || null] layout (exactly what each
        #    sub-encoder produced at forward time — the decoder must see the
        #    same unit ordering).
        offset = 0
        dyn_parts = []
        for k in self._k_per_block:
            dyn_parts.append(z[..., offset:offset + k])
            offset += k
        null_parts = []
        for i, k in enumerate(self._k_per_block):
            n_null = int(self._area_idx(i).numel()) - k
            null_parts.append(z[..., offset:offset + n_null])
            offset += n_null
        blocks_z = [
            torch.cat([dp, np_], dim=-1) for dp, np_ in zip(dyn_parts, null_parts)
        ]
        # 2) Per-area inverse.
        xs = [blk.inverse(bz) for blk, bz in zip(self.blocks, blocks_z)]
        # 3) Scatter back to the original input layout — each area's reconstructed
        #    values land at exactly the indices they were gathered from.
        shape = list(z.shape)
        shape[-1] = self._n_input
        x_out = z.new_zeros(shape)
        for i, xi in enumerate(xs):
            x_out.index_copy_(-1, self._area_idx(i), xi)
        return x_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`inverse`."""
        return self.inverse(z)


# ---------------------------------------------------------------------------
# Masked Linear Layer  (for MADE)
# ---------------------------------------------------------------------------

class MaskedLinear(nn.Module):
    """Linear layer with a fixed binary mask on the weight matrix.

    The mask zeros out specific connections, preventing information flow
    from certain input dimensions to certain output dimensions.  Used to
    enforce the autoregressive property in :class:`MADE`.

    Parameters
    ----------
    in_features : int
    out_features : int
    mask : Tensor (out_features, in_features)
        Binary mask.  Connections where ``mask == 0`` are blocked.
    """

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("mask", mask.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


# ---------------------------------------------------------------------------
# MADE  (Germain et al., "MADE: Masked Autoencoder for Distribution
#         Estimation", ICML 2015)
# ---------------------------------------------------------------------------

class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.

    Produces ``n_features * output_dim_per_input`` outputs where output
    group *i* (of size ``output_dim_per_input``) depends **only** on inputs
    ``0 .. i-1``.  Output group 0 depends on no inputs (learned bias only).

    This is the standard building block for Masked Autoregressive Flows
    (MAF, Papamakarios et al. 2017).

    Parameters
    ----------
    n_features : int
        Input/output dimensionality (D).
    hidden_dim : int
        Width of each hidden layer.
    n_hidden_layers : int
        Number of hidden layers (minimum 1).
    output_dim_per_input : int
        Number of outputs per input dimension (e.g. ``3*K + 1`` for RQS
        with *K* bins).
    zero_init : bool
        Zero-initialise the last layer so the network starts as identity.
    seed : int
        Seed for the hidden-unit ordering assignment.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        output_dim_per_input: int = 1,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.output_dim_per_input = output_dim_per_input

        # --- Assign orderings ------------------------------------------------
        # Input ordering: 0, 1, ..., D-1
        # Hidden ordering: each unit gets a random assignment in [0, D-1]
        #   meaning it is allowed to see inputs 0..m[k]
        # Output ordering: output group i must depend on inputs 0..i-1,
        #   so its ordering value is i  (strictly less-than masking)
        rng = torch.Generator().manual_seed(seed)

        degrees: list[torch.Tensor] = []
        # Input degrees: 0, 1, ..., D-1
        degrees.append(torch.arange(n_features))

        # Hidden degrees: uniformly in [0, D-1]
        for _ in range(n_hidden_layers):
            h_degrees = torch.randint(0, n_features, (hidden_dim,), generator=rng)
            degrees.append(h_degrees)

        # Output degrees: 0, 1, ..., D-1  (each repeated output_dim_per_input times)
        degrees.append(torch.arange(n_features).repeat_interleave(output_dim_per_input))

        # --- Build masked layers ---------------------------------------------
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for l in range(n_hidden_layers + 1):
            d_in = degrees[l]
            d_out = degrees[l + 1]

            if l < n_hidden_layers:
                # Hidden layer: output unit j can receive from input unit i
                # if d_in[i] <= d_out[j]  (≤ means "can see up to that index")
                mask = (d_in.unsqueeze(0) <= d_out.unsqueeze(1)).float()
                self.layers.append(MaskedLinear(len(d_in), len(d_out), mask))
                self.activations.append(nn.GELU())
            else:
                # Output layer: strictly less-than  (d_in[i] < d_out[j])
                # so output group i depends on inputs 0..i-1 only
                mask = (d_in.unsqueeze(0) < d_out.unsqueeze(1)).float()
                last = MaskedLinear(len(d_in), len(d_out), mask)
                if zero_init:
                    if near_identity_std > 0:
                        nn.init.normal_(
                            last.linear.weight, mean=0.0, std=near_identity_std
                        )
                    else:
                        nn.init.zeros_(last.linear.weight)
                    nn.init.zeros_(last.linear.bias)
                self.layers.append(last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (..., D)

        Returns
        -------
        out : Tensor (..., D * output_dim_per_input)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.activations):
                h = self.activations[i](h)
        return h


# ---------------------------------------------------------------------------
# Spline Autoregressive Layer  (Coccaro et al., arXiv:2302.12024)
# ---------------------------------------------------------------------------

class SplineAutoregressiveLayer(nn.Module):
    """Single autoregressive layer using rational-quadratic splines.

    Forward (encode): one MADE pass produces spline parameters for all
    dimensions simultaneously, then all RQS transforms are applied in
    parallel.  This is O(1) in the number of dimensions.

    Inverse (decode): inherently sequential — dimension *i* requires the
    already-inverted dimensions ``0..i-1`` to compute its conditioner
    output.  This is O(D) sequential MADE passes.

    Parameters
    ----------
    dim : int
        Feature dimension.
    hidden_dim : int
        MADE hidden width.
    n_hidden_layers : int
        MADE depth.
    num_bins : int
        Number of spline segments (K).
    tail_bound : float
        Linear tails outside ``[-tail_bound, tail_bound]``.
    zero_init : bool
        Zero-initialise MADE output layer (identity-init trick).
    seed : int
        Seed for the MADE hidden-unit ordering.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        # K widths + K heights + (K+1) derivatives per dimension
        self._params_per_dim = 3 * num_bins + 1

        self.made = MADE(
            n_features=dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim_per_input=self._params_per_dim,
            zero_init=zero_init,
            near_identity_std=near_identity_std,
            seed=seed,
        )

    def _split_params(
        self, raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape MADE output into (widths, heights, derivatives).

        Parameters
        ----------
        raw : Tensor (..., D * params_per_dim)

        Returns
        -------
        w : Tensor (..., D, K)
        h : Tensor (..., D, K)
        d : Tensor (..., D, K+1)
        """
        K = self.num_bins
        # (..., D, params_per_dim)
        raw = raw.reshape(raw.shape[:-1] + (self.dim, self._params_per_dim))
        w = raw[..., :K]
        h = raw[..., K : 2 * K]
        d = raw[..., 2 * K :]
        return w, h, d

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (parallel).

        Parameters
        ----------
        x : Tensor (..., D)

        Returns
        -------
        y : Tensor (..., D)
        log_det : Tensor (...)   summed over feature dim
        """
        raw = self.made(x)
        w, h, d = self._split_params(raw)
        y, logabsdet = rational_quadratic_spline(
            x, w, h, d, inverse=False, tail_bound=self.tail_bound
        )
        return y, logabsdet.sum(dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass (sequential over dimensions).

        Parameters
        ----------
        y : Tensor (..., D)

        Returns
        -------
        x : Tensor (..., D)
        """
        x = torch.zeros_like(y)
        for i in range(self.dim):
            # Run full MADE on the partially-filled x
            raw = self.made(x)
            w, h, d = self._split_params(raw)
            # Invert only dimension i
            x_i, _ = rational_quadratic_spline(
                y[..., i : i + 1],
                w[..., i : i + 1, :],
                h[..., i : i + 1, :],
                d[..., i : i + 1, :],
                inverse=True,
                tail_bound=self.tail_bound,
            )
            x = x.clone()
            x[..., i] = x_i.squeeze(-1)
        return x


# ---------------------------------------------------------------------------
# Spline Autoregressive Encoder  (full stack)
# ---------------------------------------------------------------------------

class SplineAutoregressiveEncoder(nn.Module):
    """Invertible encoder built from stacked autoregressive spline layers.

    Implements the A-RQS architecture from Coccaro et al. (arXiv:2302.12024),
    which demonstrates superior accuracy and stability compared to coupling-
    based spline flows (C-RQS), especially at higher dimensionalities.

    Like coupling encoders, this is **dimension-preserving**: ``n_latent ==
    n_input``.  Fixed permutations between layers ensure all dimensions
    participate in conditioning.

    **Performance note**: the forward pass (encode) is parallel via MADE,
    but the inverse pass (decode) is inherently sequential — O(D) MADE
    evaluations per layer.  For typical delay-embedded dimensions (D ~
    10-50) this is acceptable; the ODE integration is the dominant cost.

    Parameters
    ----------
    n_input : int
        Feature dimension (injected at runtime).
    n_layers : int
        Number of autoregressive layers.
    hidden_dim : int
        MADE hidden width.
    n_hidden_layers : int
        MADE depth.
    num_bins : int
        Number of RQS segments.
    tail_bound : float
        Linear tails outside ``[-tail_bound, tail_bound]``.
    use_actnorm : bool
        Insert :class:`ActNorm` after each autoregressive layer.
    zero_init : bool
        Zero-initialise MADE output layers (identity-init trick).
    permutation_seed : int
        Base seed for inter-layer permutations.  Layer *i* uses
        ``permutation_seed + i``.
    use_loft : bool
        Append :class:`LOFTLayer` after all autoregressive layers.
    loft_tau : float
        LOFT threshold.
    """

    def __init__(
        self,
        n_input: int,
        n_layers: int = 8,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        use_actnorm: bool = True,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
        permutation_seed: int = 0,
        use_loft: bool = False,
        loft_tau: float = 100.0,
    ) -> None:
        super().__init__()
        self._n_input = n_input

        self.ar_layers = nn.ModuleList()
        self.actnorms = nn.ModuleList()
        self.permutations = nn.ModuleList()
        self._use_actnorm = use_actnorm

        for i in range(n_layers):
            self.ar_layers.append(
                SplineAutoregressiveLayer(
                    dim=n_input,
                    hidden_dim=hidden_dim,
                    n_hidden_layers=n_hidden_layers,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    zero_init=zero_init,
                    near_identity_std=near_identity_std,
                    seed=permutation_seed + i,
                )
            )
            if use_actnorm:
                self.actnorms.append(ActNorm(dim=n_input))
            # Permutation between consecutive layers (not after last)
            if i < n_layers - 1:
                self.permutations.append(
                    FixedPermutation(dim=n_input, seed=permutation_seed + i)
                )

        self.loft: LOFTLayer | None = LOFTLayer(tau=loft_tau) if use_loft else None

    @property
    def n_latent(self) -> int:
        """Latent dimension (== input dimension, dimension-preserving)."""
        return self._n_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: autoregressive layers with interleaved permutations.

        Parameters
        ----------
        x : Tensor (B, T, D) or (B, D)

        Returns
        -------
        z : Tensor, same shape as x
        """
        z = x
        for i, layer in enumerate(self.ar_layers):
            z, _ = layer(z)
            if self._use_actnorm:
                z, _ = self.actnorms[i](z)
            if i < len(self.permutations):
                z = self.permutations[i](z)
        if self.loft is not None:
            z = self.loft(z)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Decode: undo all layers in reverse order.

        Parameters
        ----------
        z : Tensor (B, T, D) or (B, D)

        Returns
        -------
        x : Tensor, same shape as z
        """
        y = z
        if self.loft is not None:
            y = self.loft.inverse(y)
        for i in reversed(range(len(self.ar_layers))):
            if i < len(self.permutations):
                y = self.permutations[i].inverse(y)
            if self._use_actnorm:
                y = self.actnorms[i].inverse(y)
            y = self.ar_layers[i].inverse(y)
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
        for i, layer in enumerate(self.ar_layers):
            z, ld = layer(z)
            total_log_det = total_log_det + ld
            if self._use_actnorm:
                z, ld = self.actnorms[i](z)
                total_log_det = total_log_det + ld
            if i < len(self.permutations):
                z = self.permutations[i](z)
        if self.loft is not None:
            total_log_det = total_log_det + self.loft.log_det(z)
        return total_log_det


# ---------------------------------------------------------------------------
# Analytic Autoregressive Layer  (generic for all analytic bijections)
# ---------------------------------------------------------------------------

class AnalyticAutoregressiveLayer(nn.Module):
    """Single autoregressive layer using an analytic bijection.

    Drop-in replacement for :class:`SplineAutoregressiveLayer` — same
    ``(y, log_det)`` return signature — but uses a globally-smooth,
    analytically-invertible bijection instead of rational-quadratic splines.

    Parameters
    ----------
    dim : int
        Feature dimension.
    bijection_type : str
        One of ``'cubic_rational'``, ``'sinh'``, ``'cubic_conjugation'``.
    hidden_dim : int
        MADE hidden width.
    n_hidden_layers : int
        MADE depth.
    zero_init : bool
        Zero-initialise MADE output layer (identity-init trick).
    seed : int
        Seed for the MADE hidden-unit ordering.
    """

    def __init__(
        self,
        dim: int,
        bijection_type: str = "cubic_rational",
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if bijection_type not in _ANALYTIC_BIJECTIONS:
            raise ValueError(
                f"Unknown bijection_type {bijection_type!r}. "
                f"Choose from {list(_ANALYTIC_BIJECTIONS.keys())}"
            )
        self.dim = dim
        self._bijection_fn, self._params_per_dim = _ANALYTIC_BIJECTIONS[bijection_type]

        self.made = MADE(
            n_features=dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim_per_input=self._params_per_dim,
            zero_init=zero_init,
            near_identity_std=near_identity_std,
            seed=seed,
        )

    def _reshape_params(self, raw: torch.Tensor) -> torch.Tensor:
        """Reshape MADE output to (..., D, params_per_dim)."""
        return raw.reshape(raw.shape[:-1] + (self.dim, self._params_per_dim))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (parallel).

        Returns
        -------
        y : Tensor (..., D)
        log_det : Tensor (...)
        """
        raw = self.made(x)
        params = self._reshape_params(raw)
        y, logabsdet = self._bijection_fn(x, params, inverse=False)
        return y, logabsdet.sum(dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass (sequential over dimensions).

        Returns
        -------
        x : Tensor (..., D)
        """
        x = torch.zeros_like(y)
        for i in range(self.dim):
            raw = self.made(x)
            params = self._reshape_params(raw)
            x_i, _ = self._bijection_fn(
                y[..., i : i + 1],
                params[..., i : i + 1, :],
                inverse=True,
            )
            x = x.clone()
            x[..., i] = x_i.squeeze(-1)
        return x


# ---------------------------------------------------------------------------
# Analytic Autoregressive Encoder  (full stack)
# ---------------------------------------------------------------------------

class AnalyticAutoregressiveEncoder(nn.Module):
    """Invertible encoder built from stacked analytic autoregressive layers.

    Same structure as :class:`SplineAutoregressiveEncoder` but uses analytic
    bijections (Gerdes & Cheng, arXiv:2601.10774) instead of rational-quadratic
    splines.

    Parameters
    ----------
    n_input : int
        Feature dimension (injected at runtime).
    bijection_type : str
        One of ``'cubic_rational'``, ``'sinh'``, ``'cubic_conjugation'``.
    n_layers : int
        Number of autoregressive layers.
    hidden_dim : int
        MADE hidden width.
    n_hidden_layers : int
        MADE depth.
    use_actnorm : bool
        Insert :class:`ActNorm` after each autoregressive layer.
    zero_init : bool
        Zero-initialise MADE output layers (identity-init trick).
    permutation_seed : int
        Base seed for inter-layer permutations.
    use_loft : bool
        Append :class:`LOFTLayer` after all autoregressive layers.
    loft_tau : float
        LOFT threshold.
    """

    def __init__(
        self,
        n_input: int,
        bijection_type: str = "cubic_rational",
        n_layers: int = 8,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        use_actnorm: bool = True,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
        permutation_seed: int = 0,
        use_loft: bool = False,
        loft_tau: float = 100.0,
    ) -> None:
        super().__init__()
        self._n_input = n_input

        self.ar_layers = nn.ModuleList()
        self.actnorms = nn.ModuleList()
        self.permutations = nn.ModuleList()
        self._use_actnorm = use_actnorm

        for i in range(n_layers):
            self.ar_layers.append(
                AnalyticAutoregressiveLayer(
                    dim=n_input,
                    bijection_type=bijection_type,
                    hidden_dim=hidden_dim,
                    n_hidden_layers=n_hidden_layers,
                    zero_init=zero_init,
                    near_identity_std=near_identity_std,
                    seed=permutation_seed + i,
                )
            )
            if use_actnorm:
                self.actnorms.append(ActNorm(dim=n_input))
            if i < n_layers - 1:
                self.permutations.append(
                    FixedPermutation(dim=n_input, seed=permutation_seed + i)
                )

        self.loft: LOFTLayer | None = LOFTLayer(tau=loft_tau) if use_loft else None

    @property
    def n_latent(self) -> int:
        return self._n_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for i, layer in enumerate(self.ar_layers):
            z, _ = layer(z)
            if self._use_actnorm:
                z, _ = self.actnorms[i](z)
            if i < len(self.permutations):
                z = self.permutations[i](z)
        if self.loft is not None:
            z = self.loft(z)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        y = z
        if self.loft is not None:
            y = self.loft.inverse(y)
        for i in reversed(range(len(self.ar_layers))):
            if i < len(self.permutations):
                y = self.permutations[i].inverse(y)
            if self._use_actnorm:
                y = self.actnorms[i].inverse(y)
            y = self.ar_layers[i].inverse(y)
        return y

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inverse(z)

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        total_log_det = torch.zeros(
            x.shape[:-1], device=x.device, dtype=x.dtype
        )
        for i, layer in enumerate(self.ar_layers):
            z, ld = layer(z)
            total_log_det = total_log_det + ld
            if self._use_actnorm:
                z, ld = self.actnorms[i](z)
                total_log_det = total_log_det + ld
            if i < len(self.permutations):
                z = self.permutations[i](z)
        if self.loft is not None:
            total_log_det = total_log_det + self.loft.log_det(z)
        return total_log_det
