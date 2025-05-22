import math
import torch

# ============================================================================
# Helper: Validate inputs.
# ============================================================================
def _validate_input(t, X):
    if not t.is_floating_point():
        raise ValueError("t must be a floating point tensor.")
    if not X.is_floating_point():
        raise ValueError("X must be a floating point tensor.")
    if t.dim() != 1:
        raise ValueError(f"t must be one-dimensional; got shape {tuple(t.shape)}")
    if X.size(-2) != t.size(0):
        raise ValueError("The length of t must match the time dimension (second-to-last dimension) of X.")
    if t.size(0) < 2:
        raise ValueError("t must have at least two entries.")
    if not torch.all(t[1:] > t[:-1]):
        raise ValueError("t must be strictly increasing.")

# ============================================================================
# New helper: Batched tridiagonal solver using torch.linalg.solve.
#
# Instead of a Python loop over the rows, we build the full (small) tridiagonal matrix
# and solve in one call. This is much more GPU-friendly.
# lower: shape (n-1,), diag: shape (n,), upper: shape (n-1,)
# rhs: shape (M, n)
# ============================================================================
def _batched_tridiagonal_solve(lower, diag, upper, rhs):
    # rhs is of shape (M, n) where M is the batch size.
    M, n = rhs.shape
    # Build a full (M, n, n) matrix.
    A = torch.zeros((M, n, n), dtype=rhs.dtype, device=rhs.device)
    idx = torch.arange(n, device=rhs.device)
    A[:, idx, idx] = diag  # main diagonal
    A[:, idx[1:], idx[:-1]] = lower  # lower diagonal
    A[:, idx[:-1], idx[1:]] = upper   # upper diagonal
    # Solve the batched system.
    return torch.linalg.solve(A, rhs)

# ============================================================================
# Function: Compute piecewise cubic spline coefficients (natural case shown).
#
# For a given 1D tensor t of knot times and a tensor x of shape (..., N, D)
# (with N knots), compute the coefficients of the cubic spline that satisfies
# S''(t[0]) = S''(t[-1]) = 0.
#
# In the natural spline case, we solve for m[1],...,m[N-2] (the second derivatives)
# with m[0] = m[N-1] = 0.
# ============================================================================
def piecewise_cubic_spline_coeffs(t, x, bc_type='natural', bc_start=None, bc_end=None):
    _validate_input(t, x)
    N = t.size(0)  # number of knots
    h = t[1:] - t[:-1]  # shape (N-1)

    # Assume x has shape (..., N, D); flatten all but the time dimension.
    orig_shape = x.shape
    x_perm = x.transpose(-2, -1)  # now shape (..., D, N)
    M = x_perm.numel() // N
    f_flat = x_perm.reshape(M, N)

    device = x.device
    dtype = x.dtype

    if bc_type == 'natural':
        if N == 2:
            m_flat = torch.zeros((M, 2), dtype=dtype, device=device)
        else:
            # Number of unknowns is N-2 (for m[1], ..., m[N-2])
            # Compute r: shape (M, N-2)
            h_left = h[:-1].unsqueeze(0)    # shape (1, N-2)
            h_right = h[1:].unsqueeze(0)     # shape (1, N-2)
            r = 6 * ( (f_flat[:, 2:] - f_flat[:, 1:-1]) / h_right -
                      (f_flat[:, 1:-1] - f_flat[:, :-2]) / h_left )
            # Build the three diagonals for the system of size N-2:
            lower_diag = h[1:-1].clone()        # shape (N-3,)
            diag = 2 * (h[:-1] + h[1:])           # shape (N-2,)
            upper_diag = h[1:-1].clone()          # shape (N-3,)
            # Solve the system (r has shape (M, N-2), but note that the lower and upper diagonals must be of length N-3)
            m_interior = _batched_tridiagonal_solve(lower_diag, diag, upper_diag, r)  # shape (M, N-2)
            # Assemble full m with boundary values zero.
            m_flat = torch.zeros((M, N), dtype=dtype, device=device)
            m_flat[:, 1:-1] = m_interior
    elif bc_type == 'clamped':
        # ... (similarly update clamped case if needed)
        raise NotImplementedError("Clamped boundary conditions not shown in this example.")
    else:
        raise ValueError("Unsupported bc_type. Use 'natural' or 'clamped'.")

    # (Continue with the rest of the function to compute a, b, c, d.)
    # Reshape m_flat back to shape (..., N, D)
    m_perm = m_flat.reshape(*x_perm.shape[:-1], N)
    m = m_perm.transpose(-2, -1)

    h_shape = [1] * (x.dim() - 2) + [N - 1, 1]
    h_reshaped = h.view(h_shape)
    a = x[..., :-1, :].clone()
    b = (x[..., 1:, :] - x[..., :-1, :]) / h_reshaped - h_reshaped * (2 * m[..., :-1, :] + m[..., 1:, :]) / 6
    c = m[..., :-1, :] / 2
    d = (m[..., 1:, :] - m[..., :-1, :]) / (6 * h_reshaped)
    return t, a, b, c, d

# ============================================================================
# Class: PiecewiseCubicSpline (same API as before)
# ============================================================================
class PiecewiseCubicSpline:
    def __init__(self, t, x, bc_type='natural', bc_start=None, bc_end=None):
        _validate_input(t, x)
        self.t = t.clone()
        self.x = x.clone()
        self.bc_type = bc_type
        self.bc_start = bc_start
        self.bc_end = bc_end
        self._compute_coeffs()

    def _compute_coeffs(self):
        self.t, self.a, self.b, self.c, self.d = piecewise_cubic_spline_coeffs(
            self.t, self.x, bc_type=self.bc_type, bc_start=self.bc_start, bc_end=self.bc_end)

    def _find_interval(self, t_query):
        indices = torch.bucketize(t_query, self.t) - 1
        indices = indices.clamp(0, self.t.numel() - 2)
        t_lower = self.t[indices]
        dt = t_query - t_lower
        return dt, indices

    def evaluate(self, t_query):
        dt, indices = self._find_interval(t_query)
        dt_exp = dt.unsqueeze(-1)
        a_val = self.a[..., indices, :]
        b_val = self.b[..., indices, :]
        c_val = self.c[..., indices, :]
        d_val = self.d[..., indices, :]
        return a_val + b_val * dt_exp + c_val * dt_exp**2 + d_val * dt_exp**3

    def derivative(self, t_query, order=1):
        dt, indices = self._find_interval(t_query)
        dt_exp = dt.unsqueeze(-1)
        b_val = self.b[..., indices, :]
        c_val = self.c[..., indices, :]
        d_val = self.d[..., indices, :]
        if order == 1:
            return b_val + 2 * c_val * dt_exp + 3 * d_val * dt_exp**2
        elif order == 2:
            return 2 * c_val + 6 * d_val * dt_exp
        else:
            raise ValueError("Only derivatives of order 1 and 2 are implemented.")

    def update_left(self, new_t, new_x):
        if new_t >= self.t[0]:
            raise ValueError("new_t must be less than the current first knot (t[0]).")
        new_t_tensor = torch.tensor([new_t], dtype=self.t.dtype, device=self.t.device)
        self.t = torch.cat([new_t_tensor, self.t])
        new_x_tensor = new_x.unsqueeze(-2)
        self.x = torch.cat([new_x_tensor, self.x], dim=-2)
        self._compute_coeffs()

    def update_right(self, new_t, new_x):
        if new_t <= self.t[-1]:
            raise ValueError("new_t must be greater than the current last knot (t[-1]).")
        new_t_tensor = torch.tensor([new_t], dtype=self.t.dtype, device=self.t.device)
        self.t = torch.cat([self.t, new_t_tensor])
        new_x_tensor = new_x.unsqueeze(-2)
        self.x = torch.cat([self.x, new_x_tensor], dim=-2)
        self._compute_coeffs()

# ============================================================================
# Example usage:
# ============================================================================
if __name__ == "__main__":
    t = torch.linspace(0, 1, steps=7)
    x = torch.sin(torch.linspace(0, math.pi, steps=7)).unsqueeze(0).unsqueeze(-1)
    x = x.repeat(2, 1, 3)

    # Try on GPU:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = t.to(device)
    x = x.to(device)

    spline = PiecewiseCubicSpline(t, x, bc_type='natural')
    query_time = torch.tensor([0.4, 0.5], device=device)
    y = spline.evaluate(query_time)
    dy = spline.derivative(query_time, order=1)
    print("Spline evaluated at {}:\n{}".format(query_time, y))
    print("First derivative at {}:\n{}".format(query_time, dy))
