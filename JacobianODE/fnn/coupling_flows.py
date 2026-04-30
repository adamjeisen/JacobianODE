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
# Analytic Bijections  (Gerdes & Cheng, arXiv:2601.10774)
# ---------------------------------------------------------------------------
# Three families of C-infinity, globally-defined, analytically-invertible
# scalar bijections.  Each function takes raw unconstrained network outputs
# and applies internal parameter constraints.  All achieve identity-init
# when the network output is zero (enabled by zero_init on the conditioner).

def _solve_cubic(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Solve a·x³ + b·x² + c·x + d = 0 via Cardano's formula.

    Uses the numerically stable form that avoids catastrophic cancellation
    by choosing the larger-magnitude branch of d₁ ± √(d₁²−4d₀³).
    Includes one Newton refinement step for float32 precision.

    Reference: Gerdes & Cheng (2026), ``bijx`` library.
    """
    d0 = b.pow(2) - 3.0 * a * c
    d1 = 2.0 * b.pow(3) - 9.0 * a * b * c + 27.0 * a.pow(2) * d

    disc_inner = (d1.pow(2) - 4.0 * d0.pow(3)).clamp(min=0.0)
    sqrt_disc = torch.sqrt(disc_inner)

    minus = d1 - sqrt_disc
    plus = d1 + sqrt_disc
    # Choose the branch with larger magnitude to avoid cancellation
    c_arg = torch.where(minus.abs() < plus.abs(), plus, minus)

    C = torch.sign(c_arg) * (c_arg.abs() / 2.0).pow(1.0 / 3.0)

    # d0/C avoids division by zero when C → 0 (means d0 → 0 too)
    d0_over_C = torch.where(
        C.abs() > 1e-12,
        d0 / C,
        torch.zeros_like(C),
    )
    x = -(b + C + d0_over_C) / (3.0 * a)

    # One Newton refinement step:  x ← x − f(x)/f'(x)
    fx = a * x.pow(3) + b * x.pow(2) + c * x + d
    fpx = 3.0 * a * x.pow(2) + 2.0 * b * x + c
    x = x - fx / fpx.clamp(min=1e-12)

    return x


def cubic_rational_bijection(
    inputs: torch.Tensor,
    params: torch.Tensor,
    inverse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Cubic-rational bijection:  y = x + α·x / (1 + β·x²), centered at γ.

    Parameters
    ----------
    inputs : Tensor (..., D)
    params : Tensor (..., D, 3)
        Raw unconstrained outputs ``[raw_alpha, raw_beta, raw_gamma]``.
    inverse : bool

    Returns
    -------
    outputs : Tensor (..., D)
    logabsdet : Tensor (..., D)   per-dimension
    """
    raw_alpha = params[..., 0]
    raw_beta = params[..., 1]
    raw_gamma = params[..., 2]

    # Constraints: α ∈ (-1, 8), β > 0.
    # Identity-init at raw=0: sigmoid(log(1/8)) = 1/9 → α = -1 + 9·(1/9) = 0.
    _LOG_ONE_EIGHTH = -2.0794415416798357  # log(1/8)
    alpha = -1.0 + 9.0 * torch.sigmoid(raw_alpha + _LOG_ONE_EIGHTH)
    beta = F.softplus(raw_beta) + 1e-6
    gamma = raw_gamma

    if not inverse:
        u = inputs - gamma
        bu2 = beta * u.pow(2)
        denom = 1.0 + bu2                      # always > 0
        outputs = inputs + alpha * u / denom

        # dy/dx = 1 + α·(1 − β·u²) / (1 + β·u²)²
        #       = ((1+β·u²)² + α·(1 − β·u²)) / (1+β·u²)²
        numer = denom.pow(2) + alpha * (1.0 - bu2)
        logabsdet = torch.log(numer.abs() + 1e-12) - 2.0 * torch.log(denom)
    else:
        # Solve for v = x − γ given w = y − γ:
        #   w = v + α·v/(1+β·v²)
        #   ⟹ β·v³ − βw·v² + (1+α)·v − w = 0
        w = inputs - gamma
        v = _solve_cubic(beta, -beta * w, 1.0 + alpha, -w)
        outputs = v + gamma

        # Log-det of the *forward* map evaluated at x=output
        u_fwd = outputs - gamma
        bu2 = beta * u_fwd.pow(2)
        denom = 1.0 + bu2
        numer = denom.pow(2) + alpha * (1.0 - bu2)
        logabsdet = -(torch.log(numer.abs() + 1e-12) - 2.0 * torch.log(denom))

    return outputs, logabsdet


def sinh_conjugation_bijection(
    inputs: torch.Tensor,
    params: torch.Tensor,
    inverse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Sinh-conjugation bijection.

    Forward: ``y = α·arcsinh(eᵘ·(eᵛ·sinh((x−γ)/α) + β)) + γ``
    Inverse: swap μ↔−ν, negate β.

    Parameters
    ----------
    inputs : Tensor (..., D)
    params : Tensor (..., D, 5)
        Raw ``[raw_alpha, raw_mu, raw_nu, raw_beta, raw_gamma]``.
    inverse : bool

    Returns
    -------
    outputs, logabsdet : Tensor (..., D)
    """
    raw_alpha = params[..., 0]
    raw_mu = params[..., 1]
    raw_nu = params[..., 2]
    raw_beta = params[..., 3]
    raw_gamma = params[..., 4]

    alpha = F.softplus(raw_alpha) + 1e-6
    gamma = raw_gamma

    if not inverse:
        mu, nu, beta = raw_mu, raw_nu, raw_beta
    else:
        mu, nu, beta = -raw_nu, -raw_mu, -raw_beta

    THRESH = 15.0
    a = (inputs - gamma) / alpha

    # sinh with overflow protection
    a_clamped = a.clamp(-THRESH, THRESH)
    sinh_a = torch.sinh(a_clamped)
    # For |a| > THRESH: sinh(a) ≈ sign(a)·exp(|a|)/2
    sinh_a_large = torch.sign(a) * torch.exp(a.abs().clamp(max=80.0)) / 2.0
    sinh_val = torch.where(a.abs() <= THRESH, sinh_a, sinh_a_large)

    arg = torch.exp(mu) * (torch.exp(nu) * sinh_val + beta)
    outputs = alpha * torch.arcsinh(arg) + gamma

    # --- log |dy/dx| ---
    # dy/dx = (exp(mu+nu) · cosh(a)) / sqrt(1 + arg²)
    # log|dy/dx| = mu + nu + log(cosh(a)) − 0.5·log(1 + arg²)
    log_cosh_a = torch.where(
        a_clamped.abs() < THRESH,
        torch.log(torch.cosh(a_clamped).clamp(min=1e-30)),
        a.abs() - 0.6931471805599453,  # |a| − ln(2)
    )
    log_one_plus_arg2 = torch.where(
        arg.pow(2) < 1e8,
        torch.log1p(arg.pow(2)),
        2.0 * torch.log(arg.abs().clamp(min=1e-30)),
    )
    logabsdet_fwd = mu + nu + log_cosh_a - 0.5 * log_one_plus_arg2

    # When inverse=True we applied the forward formula with swapped params,
    # which directly computes the inverse map.  logabsdet_fwd is therefore
    # log|d(inverse)/d(y)|, which is what callers expect.
    logabsdet = logabsdet_fwd

    return outputs, logabsdet


def cubic_conjugation_bijection(
    inputs: torch.Tensor,
    params: torch.Tensor,
    inverse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Cubic-conjugation bijection: ``g⁻¹(g(x−γ) + β) + γ`` with ``g(t) = a·t + b·t³``.

    Inverse: same formula with β → −β.

    Parameters
    ----------
    inputs : Tensor (..., D)
    params : Tensor (..., D, 4)
        Raw ``[raw_a, raw_b, raw_beta, raw_gamma]``.
    inverse : bool

    Returns
    -------
    outputs, logabsdet : Tensor (..., D)
    """
    raw_a = params[..., 0]
    raw_b = params[..., 1]
    raw_beta = params[..., 2]
    raw_gamma = params[..., 3]

    a = F.softplus(raw_a) + 1e-6
    b = F.softplus(raw_b) + 1e-6
    gamma = raw_gamma
    beta = -raw_beta if inverse else raw_beta

    u = inputs - gamma
    # g(u) = a·u + b·u³
    g_u = a * u + b * u.pow(3)
    # g(u) + β
    g_u_shifted = g_u + beta

    # g⁻¹(g_u_shifted): solve b·v³ + a·v − g_u_shifted = 0
    v = _solve_cubic(b, torch.zeros_like(b), a, -g_u_shifted)
    outputs = v + gamma

    # log|dy/dx| = log|g'(u)| − log|g'(v)|  where g'(t) = a + 3b·t²
    # When inverse=True, beta was negated, so u and v are swapped relative
    # to the forward direction.  The formula naturally produces log|dx/dy|
    # (the inverse log-det) without extra negation.
    gp_u = a + 3.0 * b * u.pow(2)
    gp_v = a + 3.0 * b * v.pow(2)
    logabsdet = torch.log(gp_u.clamp(min=1e-12)) - torch.log(gp_v.clamp(min=1e-12))

    return outputs, logabsdet


# Registry: name → (function, params_per_dim)
_ANALYTIC_BIJECTIONS = {
    "cubic_rational": (cubic_rational_bijection, 3),
    "sinh": (sinh_conjugation_bijection, 5),
    "cubic_conjugation": (cubic_conjugation_bijection, 4),
}


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
        near_identity_std: float = 0.0,
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
            near_identity_std=near_identity_std,
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
# Additive (NICE-style) Coupling Layer — volume-preserving by construction
# ---------------------------------------------------------------------------


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


class AdditiveFlow(nn.Module):
    """Volume-preserving flow from stacked additive coupling layers.

    Composes *n_coupling_layers* :class:`AdditiveCouplingLayer` modules with
    :class:`FixedPermutation` layers in between so that all dimensions interact.
    Since each additive coupling has ``det(J) = 1`` and each permutation has
    ``|det| = 1``, the full composition is volume-preserving by construction.

    Parameters
    ----------
    n_dims : int
        Feature dimension.
    n_coupling_layers : int
        Number of additive coupling layers.
    hidden_dim : int
        Hidden width of the conditioner MLPs inside each coupling layer.
    n_hidden_layers : int
        Depth of the conditioner MLPs.
    permutation_seed : int
        Base seed for reproducible permutations between layers.
    zero_init : bool
        Zero-initialise the last layer of each conditioner so the flow starts
        as the identity map.
    """

    def __init__(
        self,
        n_dims: int,
        n_coupling_layers: int = 6,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        permutation_seed: int = 0,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_dims = n_dims

        self.layers = nn.ModuleList(
            [
                AdditiveCouplingLayer(
                    dim=n_dims,
                    hidden_dim=hidden_dim,
                    n_hidden_layers=n_hidden_layers,
                    zero_init=zero_init,
                    near_identity_std=near_identity_std,
                )
                for _ in range(n_coupling_layers)
            ]
        )
        # Permutations between consecutive coupling layers (not after last)
        self.permutations = nn.ModuleList(
            [
                FixedPermutation(n_dims, seed=permutation_seed + i)
                for i in range(n_coupling_layers - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map from z-space to noise space u.

        Parameters
        ----------
        x : Tensor (..., n_dims)

        Returns
        -------
        y : Tensor (..., n_dims)
        """
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            if i < len(self.permutations):
                x = self.permutations[i](x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Map from noise space u back to z-space.

        Parameters
        ----------
        y : Tensor (..., n_dims)

        Returns
        -------
        x : Tensor (..., n_dims)
        """
        for i in reversed(range(len(self.layers))):
            if i < len(self.permutations):
                y = self.permutations[i].inverse(y)
            y = self.layers[i].inverse(y)
        return y


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
        near_identity_std: float = 0.0,
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
            near_identity_std=near_identity_std,
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
# Analytic Coupling Layer  (generic for all analytic bijections)
# ---------------------------------------------------------------------------

class AnalyticCouplingLayer(nn.Module):
    """Single coupling layer using an analytic bijection.

    Drop-in replacement for :class:`SplineCouplingLayer` — same split logic,
    same ``(y, log_det)`` return signature — but uses a globally-smooth,
    analytically-invertible bijection instead of rational-quadratic splines.

    Parameters
    ----------
    dim : int
        Total feature dimension.
    bijection_type : str
        One of ``'cubic_rational'``, ``'sinh'``, ``'cubic_conjugation'``.
    split_dim : int | None
        Fixed-partition size (default ``dim // 2``).
    hidden_dim : int
        Conditioner MLP hidden width.
    n_hidden_layers : int
        Conditioner MLP depth.
    zero_init : bool
        Zero-initialise the last conditioner layer (identity-init trick).
    """

    def __init__(
        self,
        dim: int,
        bijection_type: str = "cubic_rational",
        split_dim: int | None = None,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
    ) -> None:
        super().__init__()
        if split_dim is None:
            split_dim = dim // 2
        if bijection_type not in _ANALYTIC_BIJECTIONS:
            raise ValueError(
                f"Unknown bijection_type {bijection_type!r}. "
                f"Choose from {list(_ANALYTIC_BIJECTIONS.keys())}"
            )
        self.dim = dim
        self.split_dim = split_dim
        self.transform_dim = dim - split_dim
        self._bijection_fn, self._params_per_dim = _ANALYTIC_BIJECTIONS[bijection_type]

        conditioner_out = self.transform_dim * self._params_per_dim
        self.conditioner = _build_conditioner(
            input_dim=split_dim,
            output_dim=conditioner_out,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            zero_init=zero_init,
            near_identity_std=near_identity_std,
        )

    def _get_params(self, x_a: torch.Tensor) -> torch.Tensor:
        """Run conditioner and reshape to (..., transform_dim, params_per_dim)."""
        raw = self.conditioner(x_a)
        return raw.reshape(raw.shape[:-1] + (self.transform_dim, self._params_per_dim))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_a = x[..., : self.split_dim]
        x_b = x[..., self.split_dim :]
        params = self._get_params(x_a)
        y_b, logabsdet = self._bijection_fn(x_b, params, inverse=False)
        y = torch.cat([x_a, y_b], dim=-1)
        return y, logabsdet.sum(dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_a = y[..., : self.split_dim]
        y_b = y[..., self.split_dim :]
        params = self._get_params(y_a)  # y_a == x_a
        x_b, _ = self._bijection_fn(y_b, params, inverse=True)
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
        near_identity_std: float = 0.0,
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
                    near_identity_std=near_identity_std,
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
        ``'affine'``, ``'spline'``, ``'additive'``, ``'cubic_rational'``,
        ``'sinh'``, or ``'cubic_conjugation'``.
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
    final_perm_identity : bool
        If True, append one final FixedPermutation whose buffer is chosen so
        that the composition of all permutations is the identity. Combined
        with ``zero_init=True`` this makes the entire encoder the identity
        function at initialization, so ``z[..., :n_target_dims] ==
        x[..., :n_target_dims]`` — the n_target_dims most recent observations
        land in the dynamic subspace deterministically, independent of the
        permutation seed. Expressivity is preserved because the inter-layer
        permutations are still fully random (they drive mixing across coupling
        layers); the final permutation only relabels output dimensions by a
        fixed bijection.
    use_cayley_perms : bool
        If True, replace the inter-layer ``FixedPermutation`` mixers with
        learnable :class:`CayleyOrthogonal` layers. These initialise to the
        identity, so an encoder built with this flag is *exactly* equivalent
        to one without it at initialisation; gradient is then free to learn
        non-axis-aligned rotations between coupling layers, helping the
        encoder discover non-trivial alignments (e.g., aligning data
        principal axes with z_dyn axes for partial-obs / autodim setups).
        Mutually exclusive with ``init_pca_basis=True``. When combined with
        ``final_perm_identity=True`` the final fixed permutation is the
        identity (since Cayley layers compose to identity at init).
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
        near_identity_std: float = 0.0,
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
        # routing at init
        final_perm_identity: bool = False,
        init_pca_basis: bool = False,
        pca_basis: torch.Tensor | None = None,
        # learnable orthogonal mixing
        use_cayley_perms: bool = False,
    ) -> None:
        super().__init__()
        self._n_input = n_input
        self._coupling_type = coupling_type
        self._use_actnorm = use_actnorm
        self._use_cayley_perms = use_cayley_perms
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
                        near_identity_std=near_identity_std,
                        clamp_type=clamp_type,
                        alpha_pos=alpha_pos,
                        alpha_neg=alpha_neg,
                    )
                )
            elif coupling_type == "additive":
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
                        near_identity_std=near_identity_std,
                    )
                )
            elif coupling_type in _ANALYTIC_BIJECTIONS:
                self.coupling_layers.append(
                    AnalyticCouplingLayer(
                        dim=n_input,
                        bijection_type=coupling_type,
                        split_dim=split_dim,
                        hidden_dim=hidden_dim,
                        n_hidden_layers=n_hidden_layers,
                        zero_init=zero_init,
                        near_identity_std=near_identity_std,
                    )
                )
            else:
                raise ValueError(f"Unknown coupling_type: {coupling_type!r}")

            # --- actnorm (after coupling, before permutation) ---
            if use_actnorm:
                self.actnorms.append(ActNorm(dim=n_input))

            # --- mixing (between consecutive layers, not after last) ---
            # When ``use_cayley_perms=True``, use a learnable orthogonal
            # layer parameterised by the Cayley transform instead of a
            # fixed random permutation. CayleyOrthogonal initialises to
            # the identity (U=0), so an encoder built with this flag is
            # exactly equivalent to one without it at init — gradient
            # then has the freedom to learn arbitrary inter-layer
            # rotations during training.
            if i < n_coupling_layers - 1:
                if use_cayley_perms:
                    self.permutations.append(CayleyOrthogonal(dim=n_input))
                else:
                    self.permutations.append(
                        FixedPermutation(dim=n_input, seed=permutation_seed + i)
                    )

        self.loft: LOFTLayer | None = LOFTLayer(tau=loft_tau) if use_loft else None

        # Optional final permutation chosen so the whole encoder is the
        # identity at init (given zero_init couplings): routes input dim k
        # to output dim k for every k, so z_dyn = z[..., :n_target_dims] =
        # the n_target_dims most recent observations. Preserves full
        # expressivity — the inter-layer FixedPermutations are still random
        # for mixing; this layer only relabels output dims deterministically.
        if final_perm_identity and init_pca_basis:
            raise ValueError(
                "final_perm_identity and init_pca_basis are mutually exclusive — "
                "they configure two different choices for the encoder's final layer "
                "(identity init vs PCA-basis init). Pick one."
            )
        if init_pca_basis and use_cayley_perms:
            # init_pca_basis composes per-permutation index arrays (`.perm`)
            # to derive the final fixed orthogonal Q; CayleyOrthogonal layers
            # are not permutations and don't expose `.perm`.
            raise ValueError(
                "init_pca_basis is not compatible with use_cayley_perms=True. "
                "Use final_perm_identity=True with use_cayley_perms instead."
            )

        self.final_permutation: FixedPermutation | None = None
        self.final_orthogonal: FixedOrthogonal | None = None

        if final_perm_identity:
            if use_cayley_perms:
                # Cayley layers initialise to identity (U=0), so the
                # composition of inter-layer mixers is identity at init.
                # The "final perm to make the whole encoder identity"
                # is therefore the identity permutation itself. Use a
                # FixedPermutation buffer with arange so the encode/
                # decode forward path is unchanged structurally.
                fp = FixedPermutation(dim=n_input, seed=0)
                fp.perm.copy_(torch.arange(n_input))
                fp.perm_inv.copy_(torch.arange(n_input))
                self.final_permutation = fp
            else:
                # R = composition of inter-layer random perms (applied in forward order).
                idx = torch.arange(n_input)
                for p in self.permutations:
                    idx = idx[p.perm]
                # final_perm = R^{-1}, so at init the full composition is identity.
                inv = torch.argsort(idx)
                fp = FixedPermutation(dim=n_input, seed=0)
                fp.perm.copy_(inv)
                fp.perm_inv.copy_(torch.argsort(inv))
                self.final_permutation = fp
        elif init_pca_basis:
            # Append a fixed orthogonal Q chosen so that, at init (couplings
            # are identity), the whole encoder applies V to the input — i.e.
            # z = V @ x where V = pca_basis is the PCA rotation computed
            # offline from training data. With z_dyn = z[..., :n_target_dims]
            # this means z_dyn at init is the top-n_target PCs of the input.
            #
            # Derivation: at init the coupling stack is identity, so
            #   forward(x) = Q @ P(x), where P(x)[i] = x[P_indices[i]]
            #              = (Q @ P_mat) @ x  with P_mat[i,j] = 1 iff j == P_indices[i].
            # We want this = V @ x, hence Q = V @ P_mat.T (since P_mat is
            # orthogonal). P_mat.T is itself a permutation by P_inv_indices.
            if pca_basis is None:
                raise ValueError(
                    "init_pca_basis=True requires the pca_basis kwarg "
                    "(D x D orthogonal matrix). None was passed."
                )
            if not isinstance(pca_basis, torch.Tensor):
                pca_basis = torch.as_tensor(pca_basis)
            if pca_basis.shape != (n_input, n_input):
                raise ValueError(
                    f"pca_basis must have shape ({n_input}, {n_input}), "
                    f"got {tuple(pca_basis.shape)}"
                )
            # Composed inter-layer permutation as index array.
            P_indices = torch.arange(n_input)
            for p in self.permutations:
                P_indices = P_indices[p.perm]
            # P_mat[i, j] = 1 iff j == P_indices[i].
            P_mat = torch.zeros(n_input, n_input, dtype=torch.float64)
            P_mat[torch.arange(n_input), P_indices] = 1.0
            V = pca_basis.detach().double()
            Q = V @ P_mat.T  # so that Q @ P_mat = V → forward at init is V @ x.
            self.final_orthogonal = FixedOrthogonal(Q)

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
        if self.final_permutation is not None:
            z = self.final_permutation(z)
        if self.final_orthogonal is not None:
            z = self.final_orthogonal(z)
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
        if self.final_orthogonal is not None:
            y = self.final_orthogonal.inverse(y)
        if self.final_permutation is not None:
            y = self.final_permutation.inverse(y)
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
        if self.final_permutation is not None:
            z = self.final_permutation(z)
        if self.final_orthogonal is not None:
            z = self.final_orthogonal(z)
        if self.loft is not None:
            total_log_det = total_log_det + self.loft.log_det(z)
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
    n_coupling_layers, coupling_type, hidden_dim, n_hidden_layers, zero_init,
    use_actnorm, use_loft, loft_tau, scale_activation, scale_clamp,
    clamp_type, alpha_pos, alpha_neg, num_bins, tail_bound, final_perm_identity
        Shared across all sub-encoders — see :class:`CouplingEncoder` for
        semantics. The "same MLP" invariant (every sub-encoder uses identical
        conditioner MLP hyperparameters) is enforced by passing the same
        kwargs to each sub-encoder.
    permutation_seed_base : int
        Base seed; sub-encoder *i* uses ``permutation_seed_base + i`` so each
        area gets an independent, deterministic random inter-layer
        permutation sequence.

    Notes
    -----
    PCA-basis init (``init_pca_basis`` in :class:`CouplingEncoder`) is not
    exposed here — it would require threading per-area PCA bases through
    the data pipeline. ``final_perm_identity=True`` + ``zero_init=True``
    already gives identity-at-init per area, which is the sensible default
    for the first iteration.
    """

    def __init__(
        self,
        area_indices: list[list[int]],
        n_target_dims_per_block: list[int],
        # Shared MLP / coupling settings — mirror CouplingEncoder
        n_coupling_layers: int = 8,
        coupling_type: str = "additive",
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        zero_init: bool = True,
        near_identity_std: float = 0.0,
        use_actnorm: bool = False,
        permutation_seed_base: int = 0,
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
        # routing at init
        final_perm_identity: bool = False,
        # learnable orthogonal mixing
        use_cayley_perms: bool = False,
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
            coupling_type=coupling_type,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            zero_init=zero_init,
            near_identity_std=near_identity_std,
            use_actnorm=use_actnorm,
            use_loft=use_loft,
            loft_tau=loft_tau,
            scale_activation=scale_activation,
            scale_clamp=scale_clamp,
            clamp_type=clamp_type,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            num_bins=num_bins,
            tail_bound=tail_bound,
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
