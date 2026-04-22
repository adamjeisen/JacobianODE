"""Batched square-root QR Gramian computation for LTV systems.

Computes reachability, controllability, and observability Gramians along
a trajectory-linearised (LTV) system
    dx/dt = A(t) x + B(t) u
    y     = C(t) x
ZOH-discretised with step ``dt`` so that ``M_k = exp(A_k dt)``.

Set ``discrete=True`` if ``A, B, C`` are already the one-step discrete-time
matrices (i.e. ``x_{k+1} = A x_k + B u_k``). In that case ``dt`` is ignored,
``M_k = A_k`` directly (no matrix_exp), ``P_k = A_k^{-1}``, and ``B, C``
enter unscaled (no sqrt(dt)).

The algorithm is the standard square-root low-rank Lyapunov iteration: each
step concatenates the propagated square-root factor with the scaled driving
term, then takes a reduced QR of the transpose. This preserves rank across
iterations, which is what makes the Gramian spectrum reliably computable
even when the dynamic range spans many orders of magnitude (e.g. chaotic A).

The optional ``rescale`` mode adds a scalar-per-batch log-prefactor that
soaks up exponential magnitude, keeping the square-root factor Frobenius-
bounded. Use this when ``lambda_max * T`` pushes float64 limits.
"""
from __future__ import annotations

import math

import torch


def compute_all_gramians(
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    dt: float,
    return_sequences: bool = False,
    return_spectrums: bool = True,
    rescale: bool = False,
    discrete: bool = False,
) -> tuple:
    """Compute reachability, controllability, and observability Gramians.

    Parameters
    ----------
    A : torch.Tensor
        State matrices, shape ``(batch, T, N, N)``.
    B_mat : torch.Tensor
        Input matrices, shape ``(batch, T, N, M_in)``.
    C_mat : torch.Tensor
        Output matrices, shape ``(batch, T, M_out, N)``.
    dt : float
        Discretisation step. Ignored when ``discrete=True``.
    return_sequences : bool, default False
        If True, return per-timestep time series (leading dim ``(batch, T, ...)``).
        Otherwise return only the terminal Gramian.
    return_spectrums : bool, default True
        If True, also return the eigenvalues of each Gramian. Computed from
        the singular values of the square-root factors:
        - ``rescale=False``: ``lambda_i = sigma_i(S)**2``.
        - ``rescale=True``:  ``log lambda_i = 2 l + 2 log sigma_i(S)``.
    rescale : bool, default False
        Scalar-per-batch log-prefactor rescaling. Use when the horizon times
        max Lyapunov exponent exceeds ~300 (float64 overflow). With
        ``rescale=True``:
        * The "Gramian" output slots are replaced by the factored pair
          ``(S, l)`` such that ``W = exp(2 l) * S @ S.T``. ``S`` has the
          same shape as before; ``l`` has shape ``(batch,)`` (or
          ``(batch, T)`` when ``return_sequences=True``).
        * ``spec_*`` entries contain **log** eigenvalues of the true Gramian.
    discrete : bool, default False
        If True, treat ``A, B, C`` as one-step discrete-time matrices
        (``x_{k+1} = A x_k + B u_k``). Skips the matrix_exp, uses
        ``A_k^{-1}`` for the backward pass, and drops the sqrt(dt) factors
        on ``B, C``. ``dt`` is ignored. Requires ``A_k`` nonsingular.

    Returns
    -------
    Without ``return_spectrums``: the Gramian output (materialised ``W``
    when ``rescale=False``, or the factored ``(S, l)`` pair when
    ``rescale=True``) as a 3-tuple ``(reach, ctrl, obs)``.

    With ``return_spectrums``: ``(gramian_out, (spec_r, spec_c, spec_o))``
    where each ``spec_*`` holds eigenvalues (or log eigenvalues) in
    descending order along the last axis.
    """
    batch_size, T, N, _ = A.shape
    device, dtype = A.device, A.dtype

    # 1. Precompute transitions and scaled I/O matrices
    if discrete:
        M_k = A                              # A is already the one-step map
        P_k = torch.linalg.inv(A)            # backward = inverse one-step map
        B_tilde = B_mat                      # no Riemann-weight sqrt(dt)
        C_tilde = C_mat
    else:
        M_k = torch.linalg.matrix_exp(A * dt)   # forward state transition
        P_k = torch.linalg.matrix_exp(-A * dt)  # backward state transition (for W_c)
        B_tilde = B_mat * math.sqrt(dt)
        C_tilde = C_mat * math.sqrt(dt)

    # 2. Square-root factors (initialised to zero => W_*(0) = 0)
    S_r = torch.zeros(batch_size, N, N, dtype=dtype, device=device)
    S_c = torch.zeros(batch_size, N, N, dtype=dtype, device=device)
    S_o = torch.zeros(batch_size, N, N, dtype=dtype, device=device)

    # 3. Per-batch scalar log-prefactors (no-op when rescale=False)
    l_r = torch.zeros(batch_size, dtype=dtype, device=device)
    l_c = torch.zeros(batch_size, dtype=dtype, device=device)
    l_o = torch.zeros(batch_size, dtype=dtype, device=device)

    _eps = torch.finfo(dtype).tiny

    # Sequence buffers (populated only if return_sequences=True)
    S_r_seq, S_c_seq, S_o_seq = [], [], []
    W_r_seq, W_c_seq, W_o_seq = [], [], []
    l_r_seq, l_c_seq, l_o_seq = [], [], []
    spec_r_seq, spec_c_seq, spec_o_seq = [], [], []

    def _qr_step(transformed_S: torch.Tensor, driving: torch.Tensor,
                 log_accum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One square-root update: Z = [transformed_S; driving] (row-stacked),
        optional Frobenius rescaling, reduced QR, S_next = R^T."""
        Z = torch.cat([transformed_S, driving], dim=1)  # (batch, N+m, N)
        log_next = log_accum
        if rescale:
            rho = torch.clamp(
                Z.reshape(Z.shape[0], -1).norm(dim=1),
                min=_eps,
            )
            log_next = log_accum + torch.log(rho)
            Z = Z / rho[:, None, None]
        _, R = torch.linalg.qr(Z)
        return R.transpose(1, 2), log_next

    def _drive(raw: torch.Tensor, log_accum: torch.Tensor) -> torch.Tensor:
        """Pre-scale the driving term so its magnitude is in the current
        'S-frame' (i.e. divided by e^l). No-op when rescale is off."""
        if not rescale:
            return raw
        return raw * torch.exp(-log_accum)[:, None, None]

    def _record(S: torch.Tensor, log_accum: torch.Tensor,
                S_seq: list, W_seq: list, l_seq: list, spec_seq: list) -> None:
        S_seq.append(S)
        l_seq.append(log_accum)
        if not rescale:
            W_seq.append(torch.bmm(S, S.transpose(1, 2)))
        if return_spectrums:
            sigma = torch.linalg.svdvals(S)
            if rescale:
                log_spec = 2 * log_accum[:, None] + 2 * torch.log(
                    torch.clamp(sigma, min=_eps)
                )
                spec_seq.append(log_spec)
            else:
                spec_seq.append(torch.square(sigma))

    # 4. Forward pass: Reachability
    for k in range(T):
        transformed_S_r = torch.bmm(M_k[:, k], S_r).transpose(1, 2)
        B_T = _drive(B_tilde[:, k], l_r).transpose(1, 2)
        S_r, l_r = _qr_step(transformed_S_r, B_T, l_r)
        if return_sequences:
            _record(S_r, l_r, S_r_seq, W_r_seq, l_r_seq, spec_r_seq)

    # 5. Backward pass: Controllability and Observability
    for k in range(T - 1, -1, -1):
        # Controllability (backward integration via P_k)
        transformed_S_c = torch.bmm(P_k[:, k], S_c).transpose(1, 2)
        B_T = _drive(B_tilde[:, k], l_c).transpose(1, 2)
        S_c, l_c = _qr_step(transformed_S_c, B_T, l_c)

        # Observability (pull back through M_k^T)
        transformed_S_o = torch.bmm(S_o.transpose(1, 2), M_k[:, k])
        C_k = _drive(C_tilde[:, k], l_o)
        S_o, l_o = _qr_step(transformed_S_o, C_k, l_o)

        if return_sequences:
            _record(S_c, l_c, S_c_seq, W_c_seq, l_c_seq, spec_c_seq)
            _record(S_o, l_o, S_o_seq, W_o_seq, l_o_seq, spec_o_seq)

    # 6. Assemble output
    if return_sequences:
        S_c_seq.reverse(); S_o_seq.reverse()
        l_c_seq.reverse(); l_o_seq.reverse()
        if not rescale:
            W_c_seq.reverse(); W_o_seq.reverse()
        if return_spectrums:
            spec_c_seq.reverse(); spec_o_seq.reverse()

        if rescale:
            S_r_t = torch.stack(S_r_seq, dim=1)
            S_c_t = torch.stack(S_c_seq, dim=1)
            S_o_t = torch.stack(S_o_seq, dim=1)
            l_r_t = torch.stack(l_r_seq, dim=1)
            l_c_t = torch.stack(l_c_seq, dim=1)
            l_o_t = torch.stack(l_o_seq, dim=1)
            gram_out = ((S_r_t, l_r_t), (S_c_t, l_c_t), (S_o_t, l_o_t))
        else:
            gram_out = (
                torch.stack(W_r_seq, dim=1),
                torch.stack(W_c_seq, dim=1),
                torch.stack(W_o_seq, dim=1),
            )
        if return_spectrums:
            spec_out = (
                torch.stack(spec_r_seq, dim=1),
                torch.stack(spec_c_seq, dim=1),
                torch.stack(spec_o_seq, dim=1),
            )
            return gram_out, spec_out
        return gram_out

    # Terminal-only outputs
    if rescale:
        gram_out = ((S_r, l_r), (S_c, l_c), (S_o, l_o))
    else:
        gram_out = (
            torch.bmm(S_r, S_r.transpose(1, 2)),
            torch.bmm(S_c, S_c.transpose(1, 2)),
            torch.bmm(S_o, S_o.transpose(1, 2)),
        )

    if return_spectrums:
        def _spec(S: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
            sigma = torch.linalg.svdvals(S)
            if rescale:
                return 2 * l[:, None] + 2 * torch.log(torch.clamp(sigma, min=_eps))
            return torch.square(sigma)
        spec_out = (_spec(S_r, l_r), _spec(S_c, l_c), _spec(S_o, l_o))
        return gram_out, spec_out
    return gram_out
