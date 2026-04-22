"""Sanity checks for compute_all_gramians.

Focus: the new ``discrete=True`` path. Two cross-checks:

1. ZOH equivalence. If we discretise a continuous system (A_c, B_c, C_c)
   with step dt into (A_d = exp(A_c dt), B_d = B_c sqrt(dt), C_d = C_c sqrt(dt))
   and feed the discrete matrices with ``discrete=True``, we should recover
   exactly the same Gramians as feeding (A_c, B_c, C_c, dt) with
   ``discrete=False``. The two code paths run different ops but must land
   on the same recursion.

2. Closed-form discrete Lyapunov. For a stable time-invariant discrete
   system, W_r = sum_{k=0}^{T-1} A_d^k B_d B_d^T (A_d^T)^k. Compare the
   function's terminal output against the explicit sum at small T.
"""
from __future__ import annotations

import torch

from JacobianODE.control.gramians import compute_all_gramians


def _random_stable_system(N: int, M_in: int, M_out: int, T: int,
                          batch: int = 2, seed: int = 0) -> tuple:
    """Random A with eigenvalues in the left half-plane (stable continuous).
    Returns (A_c, B, C) all constant-in-time, shape (batch, T, ...)."""
    g = torch.Generator().manual_seed(seed)
    A0 = torch.randn(batch, N, N, generator=g, dtype=torch.float64)
    # Make A stable: A = -A^T A / ||...|| - s*I, negative real parts
    A0 = -(A0 @ A0.transpose(1, 2)) / N - 0.5 * torch.eye(N, dtype=torch.float64)[None]
    B0 = torch.randn(batch, N, M_in, generator=g, dtype=torch.float64)
    C0 = torch.randn(batch, M_out, N, generator=g, dtype=torch.float64)
    # Broadcast to (batch, T, ...)
    A = A0[:, None].expand(batch, T, N, N).contiguous()
    B = B0[:, None].expand(batch, T, N, M_in).contiguous()
    C = C0[:, None].expand(batch, T, M_out, N).contiguous()
    return A, B, C


def test_discrete_matches_continuous_under_zoh():
    """ZOH-discretise a continuous LTI system and verify discrete=True
    agrees with discrete=False to machine precision."""
    N, M_in, M_out, T = 4, 2, 3, 20
    dt = 0.05

    A_c, B_c, C_c = _random_stable_system(N, M_in, M_out, T)

    import math

    # Continuous-time call
    (W_r_c, W_c_c, W_o_c), _ = compute_all_gramians(
        A_c, B_c, C_c, dt, return_spectrums=True, discrete=False,
    )

    # Build discrete matrices that encode the same recursion
    A_d = torch.linalg.matrix_exp(A_c * dt)
    B_d = B_c * math.sqrt(dt)
    C_d = C_c * math.sqrt(dt)

    (W_r_d, W_c_d, W_o_d), _ = compute_all_gramians(
        A_d, B_d, C_d, dt=1.0, return_spectrums=True, discrete=True,
    )

    # Both paths build the identical Z = [M S; driving^T] at every step,
    # so outputs must match to float64 noise.
    atol = 1e-10
    assert torch.allclose(W_r_c, W_r_d, atol=atol), (W_r_c - W_r_d).abs().max()
    assert torch.allclose(W_c_c, W_c_d, atol=atol), (W_c_c - W_c_d).abs().max()
    assert torch.allclose(W_o_c, W_o_d, atol=atol), (W_o_c - W_o_d).abs().max()


def test_discrete_matches_closed_form_sum():
    """W_r (terminal, discrete) = sum_{k=0}^{T-1} A^k B B^T (A^T)^k."""
    N, M_in, T = 3, 2, 8
    torch.manual_seed(1)
    # A with all eigenvalues inside the unit disk (stable discrete)
    A0 = 0.5 * torch.randn(N, N, dtype=torch.float64)
    # Contract spectral radius to be safe
    sr = torch.linalg.eigvals(A0).abs().max()
    A0 = 0.7 / sr * A0
    B0 = torch.randn(N, M_in, dtype=torch.float64)
    C0 = torch.randn(1, N, dtype=torch.float64)  # M_out=1 (unused for W_r check)

    # Shape (batch=1, T, ...)
    A = A0[None, None].expand(1, T, N, N).contiguous()
    B = B0[None, None].expand(1, T, N, M_in).contiguous()
    C = C0[None, None].expand(1, T, 1, N).contiguous()

    (W_r, _, _), _ = compute_all_gramians(
        A, B, C, dt=1.0, return_spectrums=True, discrete=True,
    )

    # Closed form: W_r(T) = sum_{k=0}^{T-1} A^{T-1-k} B B^T (A^T)^{T-1-k}
    # (equivalently sum over k=0..T-1 of A^k B B^T (A^T)^k by re-indexing)
    W_ref = torch.zeros(N, N, dtype=torch.float64)
    Apk = torch.eye(N, dtype=torch.float64)
    for _ in range(T):
        W_ref = W_ref + Apk @ B0 @ B0.T @ Apk.T
        Apk = A0 @ Apk

    assert torch.allclose(W_r[0], W_ref, atol=1e-10), (W_r[0] - W_ref).abs().max()


def test_discrete_symmetric_psd():
    """Basic invariant: Gramians are symmetric PSD."""
    N, M_in, M_out, T = 4, 2, 2, 10
    A_c, B_c, C_c = _random_stable_system(N, M_in, M_out, T, seed=7)
    A_d = torch.linalg.matrix_exp(A_c * 0.1)

    (W_r, W_c, W_o), _ = compute_all_gramians(
        A_d, B_c, C_c, dt=1.0, return_spectrums=True, discrete=True,
    )
    for W in (W_r, W_c, W_o):
        asym = (W - W.transpose(-1, -2)).abs().max()
        min_eig = torch.linalg.eigvalsh(W).min()
        assert asym < 1e-10, f"asymmetry {asym}"
        assert min_eig > -1e-10, f"min eig {min_eig}"
