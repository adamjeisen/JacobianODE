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
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Asymmetric Soft Clamp  (Andrade 2024, Eq. 7, arXiv:2402.16408)
# ---------------------------------------------------------------------------

def asymmetric_soft_clamp(
    s: torch.Tensor,
    alpha_pos: float = 0.1,
    alpha_neg: float = 2.0,
) -> torch.Tensor:
    """Asymmetric arctan-based soft clamp for the log-scale parameter.

    Tightly bounds expansion (positive *s*) via ``alpha_pos`` while allowing
    more compression (negative *s*) via ``alpha_neg``.  Fully differentiable
    and vectorised — no Python ``if``/``else`` branching.

    Parameters
    ----------
    s : Tensor
        Raw (unclamped) log-scale values.
    alpha_pos : float
        Controls upper saturation.  Smaller ⇒ tighter expansion bound.
    alpha_neg : float
        Controls lower saturation.  Larger ⇒ looser compression bound.
    """
    pos = (2.0 / torch.pi) * alpha_pos * torch.arctan(s / alpha_pos)
    neg = (2.0 / torch.pi) * alpha_neg * torch.arctan(s / alpha_neg)
    return torch.where(s >= 0, pos, neg)


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
# Rational Quadratic Spline  (Durkan et al., "Neural Spline Flows", NeurIPS 2019)
# ---------------------------------------------------------------------------

def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rational-quadratic spline transform with linear tails.

    Inputs outside ``[-tail_bound, tail_bound]`` pass through unchanged
    (linear tails, zero log-det contribution).

    Parameters
    ----------
    inputs : Tensor (..., D)
    unnormalized_widths : Tensor (..., D, K)
    unnormalized_heights : Tensor (..., D, K)
    unnormalized_derivatives : Tensor (..., D, K+1)
    inverse : bool
    tail_bound : float
    min_bin_width, min_bin_height, min_derivative : float
        Numerical safety floors.

    Returns
    -------
    outputs : Tensor (..., D)
    logabsdet : Tensor (..., D)   — per-dimension (caller sums if needed)
    """
    num_bins = unnormalized_widths.shape[-1]
    left, right = -tail_bound, tail_bound
    bottom, top = -tail_bound, tail_bound

    if min_bin_width * num_bins > 1.0:
        raise ValueError(f"min_bin_width {min_bin_width} too large for {num_bins} bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError(f"min_bin_height {min_bin_height} too large for {num_bins} bins")

    # --- Widths: softmax → clipped → cumulative --------------------------
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # --- Heights: same treatment ------------------------------------------
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # --- Derivatives: strictly positive -----------------------------------
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    # --- Clamp inputs into [-B, B] for bin lookup; we blend tails later ---
    if inverse:
        inside_mask = (inputs >= bottom) & (inputs <= top)
        clamped = inputs.clamp(bottom, top)
    else:
        inside_mask = (inputs >= left) & (inputs <= right)
        clamped = inputs.clamp(left, right)

    # --- Bin selection via searchsorted ------------------------------------
    # cumwidths/cumheights: (..., D, K+1),  clamped: (..., D)
    if inverse:
        bin_edges = cumheights.clone()
    else:
        bin_edges = cumwidths.clone()
    bin_edges[..., -1] += 1e-6  # nudge last edge so boundary is included
    bin_idx = torch.sum(clamped.unsqueeze(-1) >= bin_edges, dim=-1) - 1
    bin_idx = bin_idx.clamp(0, num_bins - 1)

    # --- Gather per-element bin parameters --------------------------------
    idx = bin_idx.unsqueeze(-1)  # (..., D, 1)
    input_cumwidths = cumwidths.gather(-1, idx).squeeze(-1)
    input_bin_widths = widths.gather(-1, idx).squeeze(-1)
    input_cumheights = cumheights.gather(-1, idx).squeeze(-1)
    input_heights = heights.gather(-1, idx).squeeze(-1)
    input_delta = input_heights / input_bin_widths
    input_derivatives = derivatives.gather(-1, idx).squeeze(-1)
    input_derivatives_p1 = derivatives[..., 1:].gather(-1, idx).squeeze(-1)

    # --- Forward transform ------------------------------------------------
    if not inverse:
        theta = (clamped - input_cumwidths) / input_bin_widths
        theta = theta.clamp(0.0, 1.0)

        one_m_theta = 1.0 - theta
        dkp1 = input_derivatives_p1
        dk = input_derivatives

        numerator = input_heights * (input_delta * theta.pow(2) + dk * theta * one_m_theta)
        denominator = input_delta + (dkp1 + dk - 2.0 * input_delta) * theta * one_m_theta
        spline_out = input_cumheights + numerator / denominator

        deriv_numerator = input_delta.pow(2) * (
            dkp1 * theta.pow(2)
            + 2.0 * input_delta * theta * one_m_theta
            + dk * one_m_theta.pow(2)
        )
        log_deriv = torch.log(deriv_numerator + 1e-12) - 2.0 * torch.log(denominator.abs() + 1e-12)

    # --- Inverse transform ------------------------------------------------
    else:
        a = (clamped - input_cumheights) * (input_derivatives_p1 + input_derivatives - 2.0 * input_delta)
        a = a + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (clamped - input_cumheights) * (
            input_derivatives_p1 + input_derivatives - 2.0 * input_delta
        )
        c = -input_delta * (clamped - input_cumheights)

        discriminant = b.pow(2) - 4.0 * a * c
        discriminant = discriminant.clamp(min=0.0)

        root = (2.0 * c) / (-b - torch.sqrt(discriminant))
        root = root.clamp(0.0, 1.0)

        spline_out = root * input_bin_widths + input_cumwidths

        one_m_root = 1.0 - root
        dkp1 = input_derivatives_p1
        dk = input_derivatives
        denominator = input_delta + (dkp1 + dk - 2.0 * input_delta) * root * one_m_root
        deriv_numerator = input_delta.pow(2) * (
            dkp1 * root.pow(2)
            + 2.0 * input_delta * root * one_m_root
            + dk * one_m_root.pow(2)
        )
        log_deriv = torch.log(deriv_numerator + 1e-12) - 2.0 * torch.log(denominator.abs() + 1e-12)
        log_deriv = -log_deriv  # negative for inverse

    # --- Blend: spline inside, identity outside ---------------------------
    outputs = torch.where(inside_mask, spline_out, inputs)
    logabsdet = torch.where(inside_mask, log_deriv, torch.zeros_like(log_deriv))

    return outputs, logabsdet


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
        uses raw clamping.  Ignored when ``clamp_type='asymmetric'``.
    scale_clamp : float
        Maximum absolute value of log_s.  Ignored when
        ``clamp_type='asymmetric'``.
    zero_init : bool
        Zero-initialise the last conditioner layer (identity-init trick).
    clamp_type : str
        ``'symmetric'`` (default, existing behaviour) or ``'asymmetric'``
        (arctan-based, Andrade 2024).
    alpha_pos : float
        Asymmetric clamp upper-bound rate (expansion).  Only used when
        ``clamp_type='asymmetric'``.
    alpha_neg : float
        Asymmetric clamp lower-bound rate (compression).  Only used when
        ``clamp_type='asymmetric'``.
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
        clamp_type: str = "symmetric",
        alpha_pos: float = 0.1,
        alpha_neg: float = 2.0,
    ) -> None:
        super().__init__()
        if split_dim is None:
            split_dim = dim // 2
        self.dim = dim
        self.split_dim = split_dim
        self.transform_dim = dim - split_dim
        self.scale_activation = scale_activation
        self.scale_clamp = scale_clamp
        self.clamp_type = clamp_type
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

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
        if self.clamp_type == "asymmetric":
            log_s = asymmetric_soft_clamp(log_s, self.alpha_pos, self.alpha_neg)
        elif self.scale_activation == "tanh":
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
# Spline Coupling Layer  (Durkan et al., "Neural Spline Flows", NeurIPS 2019)
# ---------------------------------------------------------------------------

class SplineCouplingLayer(nn.Module):
    """Single coupling layer using rational-quadratic splines.

    Drop-in replacement for :class:`AffineCouplingLayer` — same split logic,
    same ``(y, log_det)`` return signature — but uses a piecewise monotonic
    spline instead of an affine transform.

    Parameters
    ----------
    dim : int
        Total feature dimension.
    split_dim : int | None
        Fixed-partition size (default ``dim // 2``).
    hidden_dim : int
        Conditioner MLP hidden width.
    n_hidden_layers : int
        Conditioner MLP depth.
    num_bins : int
        Number of spline segments.
    tail_bound : float
        Inputs outside ``[-tail_bound, tail_bound]`` use linear tails.
    zero_init : bool
        Zero-initialise the last conditioner layer (identity-init trick).
    """

    def __init__(
        self,
        dim: int,
        split_dim: int | None = None,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if split_dim is None:
            split_dim = dim // 2
        self.dim = dim
        self.split_dim = split_dim
        self.transform_dim = dim - split_dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        # K widths + K heights + (K+1) derivatives per transformed dimension
        conditioner_out = self.transform_dim * (3 * num_bins + 1)
        self.conditioner = _build_conditioner(
            input_dim=split_dim,
            output_dim=conditioner_out,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            zero_init=zero_init,
        )

    def _get_spline_params(
        self, x_a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run conditioner and reshape into (widths, heights, derivatives)."""
        raw = self.conditioner(x_a)  # (..., transform_dim * (3K+1))
        K = self.num_bins
        # Reshape to (..., transform_dim, 3K+1) then split
        shape = raw.shape[:-1] + (self.transform_dim, 3 * K + 1)
        raw = raw.reshape(shape)
        w = raw[..., :K]
        h = raw[..., K : 2 * K]
        d = raw[..., 2 * K :]
        return w, h, d

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        y : Tensor (..., D)
        log_det : Tensor (...)
        """
        x_a = x[..., : self.split_dim]
        x_b = x[..., self.split_dim :]
        w, h, d = self._get_spline_params(x_a)
        y_b, logabsdet = rational_quadratic_spline(
            x_b, w, h, d, inverse=False, tail_bound=self.tail_bound
        )
        y = torch.cat([x_a, y_b], dim=-1)
        return y, logabsdet.sum(dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Exact analytical inverse.

        Returns
        -------
        x : Tensor (..., D)
        """
        y_a = y[..., : self.split_dim]
        y_b = y[..., self.split_dim :]
        w, h, d = self._get_spline_params(y_a)  # y_a == x_a
        x_b, _ = rational_quadratic_spline(
            y_b, w, h, d, inverse=True, tail_bound=self.tail_bound
        )
        return torch.cat([y_a, x_b], dim=-1)


# ---------------------------------------------------------------------------
# ActNorm  (Kingma & Dhariwal, "Glow", NeurIPS 2018)
# ---------------------------------------------------------------------------

class ActNorm(nn.Module):
    """Activation normalization with data-dependent initialization.

    Learnable per-feature affine transform.  On the first forward pass the
    parameters are set so that the output has zero mean and unit variance,
    after which they are trained normally.

    Parameters
    ----------
    dim : int
        Feature dimension (last axis).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.register_buffer("_initialized", torch.tensor(False))

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        """Set parameters so that output ≈ zero-mean, unit-variance."""
        # Flatten all dims except last for statistics
        flat = x.reshape(-1, x.shape[-1])
        mu = flat.mean(dim=0)
        std = flat.std(dim=0).clamp(min=1e-6)
        self.log_scale.data.copy_(-torch.log(std))
        self.shift.data.copy_(-mu / std)
        self._initialized.fill_(True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        y : Tensor, same shape as x
        log_det : Tensor (...) — summed over feature dim
        """
        if not self._initialized:
            self._initialize(x)
        scale = torch.exp(self.log_scale)
        y = scale * x + self.shift
        log_det = self.log_scale.sum().expand(x.shape[:-1])
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Exact inverse."""
        scale = torch.exp(self.log_scale)
        return (y - self.shift) / scale


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
# LOFT — Log Soft Extension  (Andrade 2024, Eq. 10, arXiv:2402.16408)
# ---------------------------------------------------------------------------

class LOFTLayer(nn.Module):
    """Bijective layer that logarithmically squashes outlier values.

    Acts as the identity inside ``[-tau, tau]`` and grows only
    logarithmically beyond that range.  Placed after all coupling blocks
    to compress any residual extreme values.

    Parameters
    ----------
    tau : float
        Threshold beyond which logarithmic compression activates.
    """

    def __init__(self, tau: float = 100.0) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x → z (compress outliers)."""
        abs_x = x.abs()
        return x.sign() * (
            torch.log(torch.clamp(abs_x - self.tau, min=0.0) + 1.0)
            + torch.clamp(abs_x, max=self.tau)
        )

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: z → x (expand back)."""
        abs_z = z.abs()
        return z.sign() * (
            torch.exp(torch.clamp(abs_z - self.tau, min=0.0)) - 1.0
            + torch.clamp(abs_z, max=self.tau)
        )

    def log_det(self, x: torch.Tensor) -> torch.Tensor:
        """Log |det J| of the forward map, summed over the feature dim.

        Must be evaluated at *x* (the **input** to the layer), not the output.
        """
        abs_x = x.abs()
        return -torch.log(
            torch.clamp(abs_x - self.tau, min=0.0) + 1.0
        ).sum(dim=-1)


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
        ``'tanh'`` or ``'none'`` — how to bound log-scale.  Ignored when
        ``clamp_type='asymmetric'``.
    scale_clamp : float
        Maximum absolute value of log-scale.  Ignored when
        ``clamp_type='asymmetric'``.
    zero_init : bool
        Zero-initialise last conditioner layer for identity-init.
    permutation_seed : int
        Base seed for generating the fixed permutations.  Layer *i* uses
        ``permutation_seed + i``.
    clamp_type : str
        ``'symmetric'`` (default) or ``'asymmetric'`` (Andrade 2024).
    alpha_pos : float
        Asymmetric clamp expansion bound.
    alpha_neg : float
        Asymmetric clamp compression bound.
    use_loft : bool
        If True, append a :class:`LOFTLayer` after all coupling blocks.
    loft_tau : float
        LOFT threshold (identity inside ``[-tau, tau]``).
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
        clamp_type: str = "symmetric",
        alpha_pos: float = 0.1,
        alpha_neg: float = 2.0,
        use_loft: bool = False,
        loft_tau: float = 100.0,
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
                    clamp_type=clamp_type,
                    alpha_pos=alpha_pos,
                    alpha_neg=alpha_neg,
                )
            )
            # Permutation after every layer except the last
            if i < n_coupling_layers - 1:
                self.permutations.append(
                    FixedPermutation(dim=n_input, seed=permutation_seed + i)
                )

        # Optional LOFT layer (appended after all coupling blocks)
        self.loft: LOFTLayer | None = LOFTLayer(tau=loft_tau) if use_loft else None

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
        if self.loft is not None:
            z = self.loft(z)
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
        if self.loft is not None:
            y = self.loft.inverse(y)
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
        # LOFT log-det evaluated at z (the pre-LOFT tensor)
        if self.loft is not None:
            total_log_det = total_log_det + self.loft.log_det(z)
        return total_log_det


# ---------------------------------------------------------------------------
# CouplingEncoder — generalised stack (affine or spline, optional ActNorm)
# ---------------------------------------------------------------------------

class CouplingEncoder(nn.Module):
    """Invertible encoder supporting affine or spline coupling with optional ActNorm.

    Generalises :class:`AffineCouplingEncoder` to also support rational-quadratic
    spline coupling layers and per-layer activation normalisation.

    Parameters
    ----------
    n_input : int
        Feature dimension (injected at runtime from the data pipeline).
    n_coupling_layers : int
        Number of coupling layers.
    coupling_type : str
        ``'affine'`` or ``'spline'``.
    use_actnorm : bool
        Insert an :class:`ActNorm` layer after each coupling layer.
    hidden_dim : int
        Conditioner MLP hidden width.
    n_hidden_layers : int
        Conditioner MLP depth.
    zero_init : bool
        Zero-initialise last conditioner layer for identity-init.
    permutation_seed : int
        Base seed for fixed permutations.  Layer *i* uses
        ``permutation_seed + i``.
    use_loft : bool
        If True, append a :class:`LOFTLayer` after all coupling blocks.
    loft_tau : float
        LOFT threshold.
    scale_activation : str
        (affine only) ``'tanh'`` or ``'none'``.
    scale_clamp : float
        (affine only) Maximum absolute value of log-scale.
    clamp_type : str
        (affine only) ``'symmetric'`` or ``'asymmetric'``.
    alpha_pos, alpha_neg : float
        (affine only) Asymmetric clamp parameters.
    num_bins : int
        (spline only) Number of spline segments.
    tail_bound : float
        (spline only) Linear tails outside ``[-tail_bound, tail_bound]``.
    """

    def __init__(
        self,
        n_input: int,
        n_coupling_layers: int = 8,
        coupling_type: str = "affine",
        use_actnorm: bool = False,
        # shared
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        zero_init: bool = True,
        permutation_seed: int = 0,
        use_loft: bool = False,
        loft_tau: float = 100.0,
        # affine-specific
        scale_activation: str = "tanh",
        scale_clamp: float = 3.0,
        clamp_type: str = "symmetric",
        alpha_pos: float = 0.1,
        alpha_neg: float = 2.0,
        # spline-specific
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ) -> None:
        super().__init__()
        self._n_input = n_input
        self._coupling_type = coupling_type
        self._use_actnorm = use_actnorm
        split_dim = n_input // 2

        self.coupling_layers = nn.ModuleList()
        self.actnorms = nn.ModuleList()
        self.permutations = nn.ModuleList()

        for i in range(n_coupling_layers):
            # --- coupling layer ---
            if coupling_type == "affine":
                self.coupling_layers.append(
                    AffineCouplingLayer(
                        dim=n_input,
                        split_dim=split_dim,
                        hidden_dim=hidden_dim,
                        n_hidden_layers=n_hidden_layers,
                        scale_activation=scale_activation,
                        scale_clamp=scale_clamp,
                        zero_init=zero_init,
                        clamp_type=clamp_type,
                        alpha_pos=alpha_pos,
                        alpha_neg=alpha_neg,
                    )
                )
            elif coupling_type == "spline":
                self.coupling_layers.append(
                    SplineCouplingLayer(
                        dim=n_input,
                        split_dim=split_dim,
                        hidden_dim=hidden_dim,
                        n_hidden_layers=n_hidden_layers,
                        num_bins=num_bins,
                        tail_bound=tail_bound,
                        zero_init=zero_init,
                    )
                )
            else:
                raise ValueError(f"Unknown coupling_type: {coupling_type!r}")

            # --- actnorm (after coupling, before permutation) ---
            if use_actnorm:
                self.actnorms.append(ActNorm(dim=n_input))

            # --- permutation (between consecutive layers, not after last) ---
            if i < n_coupling_layers - 1:
                self.permutations.append(
                    FixedPermutation(dim=n_input, seed=permutation_seed + i)
                )

        self.loft: LOFTLayer | None = LOFTLayer(tau=loft_tau) if use_loft else None

    # ----- properties -----

    @property
    def n_latent(self) -> int:
        """Latent dimension (== input dimension for coupling flows)."""
        return self._n_input

    # ----- forward / inverse -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: coupling → [actnorm →] permutation, repeated.

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
        for i in reversed(range(len(self.coupling_layers))):
            if i < len(self.permutations):
                y = self.permutations[i].inverse(y)
            if self._use_actnorm:
                y = self.actnorms[i].inverse(y)
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
            if self._use_actnorm:
                z, ld = self.actnorms[i](z)
                total_log_det = total_log_det + ld
            if i < len(self.permutations):
                z = self.permutations[i](z)
        if self.loft is not None:
            total_log_det = total_log_det + self.loft.log_det(z)
        return total_log_det
