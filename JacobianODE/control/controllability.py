import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import torch
from torch.linalg import LinAlgError
from torchdiffeq import odeint
from tqdm.auto import tqdm

def controllability_and_reachability(A_func, B_func, x, dt, n=None, n_ctrl_steps=10, integration_step_scale=1, return_all=False, verbose=False):
    """Compute controllability and reachability Gramians for a time-varying linear system.

    This function computes the controllability and reachability Gramians by solving the differential Lyapunov equations
    using numerical integration. The Gramians characterize how controllable/reachable the system is from different states.

    WARNING: CONTROLLABILITY GRAMIAN IS NOT BEING COMPUTED

    Args:
        A_func (Callable): Function that returns the system matrix A(t) at time t
        B_func (Callable): Function that returns the input matrix B(t) at time t
        x (torch.Tensor): State trajectory tensor
        dt (float): Time step size
        n (Optional[int]): State dimension. If None, inferred from x.shape[-1]
        n_ctrl_steps (int): Number of control steps to consider for each interval
        integration_step_scale (int): Scale factor for number of integration steps
        return_all (bool): If True, return Gramians for all time steps. If False, only return final values
        verbose (bool): Whether to show progress bar

    Returns:
        tuple: (W_c, W_r) where:
            - W_c: Controllability Gramian
            - W_r: Reachability Gramian
            Each Gramian has shape (*x.shape[:-2], n_intervals, n, n) if return_all=True,
            or (*x.shape[:-2], n_intervals, n_ctrl_steps, n, n) if return_all=False
    """
    if n is None:
        n = x.shape[-1]
    
    dtype = x.dtype
    device = x.device

    n_integration_steps = n_ctrl_steps*integration_step_scale

    def W_r_dot(t, W_r):
        W_r = W_r.reshape(*W_r.shape[:-1], int(np.sqrt(W_r.shape[-1])), int(np.sqrt(W_r.shape[-1])))
        A = A_func(t)
        B = B_func(t)
        W_r_dot = B @ B.transpose(-2, -1) + A @ W_r + W_r @ A.transpose(-2, -1)
        return W_r_dot.reshape(*W_r_dot.shape[:-2], -1)

    # def W_c_dot(t, W_c):
    #     W_c = W_c.reshape(*W_c.shape[:-1], int(np.sqrt(W_c.shape[-1])), int(np.sqrt(W_c.shape[-1])))
    #     A = A_func(t)
    #     B = B_func(t)
    #     W_c_dot = -B @ B.transpose(-2, -1) + A @ W_c + W_c @ A.transpose(-2, -1)
    #     return W_c_dot.reshape(*W_c_dot.shape[:-2], -1)

    interval_ends = np.arange(n_ctrl_steps, x.shape[-2] + 1)

    if return_all:
        W_r = torch.zeros(*x.shape[:-2], len(interval_ends), n_ctrl_steps, n, n).type(dtype).to(device)
        W_c = torch.zeros(*x.shape[:-2], len(interval_ends), n_ctrl_steps, n, n).type(dtype).to(device)
    else:
        W_r = torch.zeros(*x.shape[:-2], len(interval_ends),  n, n).type(dtype).to(device)
        W_c = torch.zeros(*x.shape[:-2], len(interval_ends), n, n).type(dtype).to(device)
    for i in tqdm(interval_ends, disable=not verbose):
        t_0 = (i - n_ctrl_steps)*dt
        t_1 = (i)*dt
        odeint_kwargs = {"method": "rk4", "rtol": 1e-9, "atol": 1e-11}
        # odeint_kwargs = {"method": "euler", "rtol": 1e-9, "atol": 1e-11}
        # t_eval = torch.arange(t_0, t_1 + dt/2, dt/dt_scale)
        # t_eval = torch.tensor([t_0, t_1])
        t_eval = torch.linspace(t_0, t_1, n_integration_steps).type(dtype).to(device)

        W_r_sol =odeint(
            W_r_dot, 
            torch.zeros(*x.shape[:-2], n*n).type(dtype).to(device),
            t_eval,
            **odeint_kwargs
        )
        W_r_sol = W_r_sol.reshape(*W_r_sol.shape[:-1], int(np.sqrt(W_r_sol.shape[-1])), int(np.sqrt(W_r_sol.shape[-1])))
        if return_all:
            W_r_sol = W_r_sol[::integration_step_scale]
            if len(W_r_sol.shape) == 4:
                W_r_sol = W_r_sol.transpose(0, 1)
            elif len(W_r_sol.shape) > 5:
                raise NotImplementedError("Only one batch dimension supported for now")
            
            W_r[..., i - n_ctrl_steps, :, :, :] = W_r_sol
        else:
            W_r_sol = W_r_sol[-1]
            W_r[..., i - n_ctrl_steps, :, :] = W_r_sol
        
        # # odeint_kwargs = {"method": "implicit_midpoint", "rtol": 1e-1, "atol": 1e-1}
        # W_c_sol =odeint(
        #     W_c_dot,
        #     torch.zeros(*x.shape[:-2], n*n).type(dtype).to(device),
        #     torch.flip(t_eval, [-1]),
        #     **odeint_kwargs,
        # )
        # W_c_sol = W_c_sol.reshape(*W_c_sol.shape[:-1], int(np.sqrt(W_c_sol.shape[-1])), int(np.sqrt(W_c_sol.shape[-1])))
        # if return_all:
        #     W_c_sol = W_c_sol[::integration_step_scale]
        #     if len(W_c_sol.shape) == 4:
        #         W_c_sol = W_c_sol.transpose(0, 1)
        #     elif len(W_c_sol.shape) > 5:
        #         raise NotImplementedError("Only one batch dimension supported for now")
            
        #     W_c[..., i - n_ctrl_steps, :, :, :] = W_c_sol
        # else:
        #     W_c_sol = W_c_sol[-1]
        #     W_c[..., i - n_ctrl_steps, :, :] = W_c_sol
    
    return W_c, W_r

def get_gramian_eigs(W):
    """Compute eigenvalues of Gramian matrices.

    This function computes the eigenvalues of controllability/reachability Gramians,
    handling edge cases like infinite values and NaNs. Negative eigenvalues are clamped to 0.

    Args:
        W (torch.Tensor): Gramian matrix or batch of Gramians

    Returns:
        torch.Tensor: Eigenvalues of the Gramian(s), sorted in descending order.
                     Shape matches W.shape[:-2] (removing the last two dimensions)
    """
    gram_eigs = torch.zeros(W.shape[:-1]).type(W.dtype).to(W.device)

    finite = (torch.abs(W) == np.inf).sum(axis=(-1, -2)) == 0
    not_nan = (torch.isnan(W) == 1).sum(axis=(-2, -1)) == 0
    if finite.sum() != torch.prod(torch.tensor(W.shape[:-2])) or not_nan.sum() != torch.prod(torch.tensor(W.shape[:-2])):
        print(f"WARNING: W has {finite.sum()} infs and {not_nan.sum()} NaNs")
    usable = finite & not_nan
    gram_eigs[usable] = torch.flip(torch.sort(torch.real(torch.linalg.eigvals(W[usable]))).values, dims=(-1,))

    negative_indices = torch.where(gram_eigs < 0)
    if len(negative_indices[0]) > 0:
        print(f"WARNING: W has {len(negative_indices[0])} negative eigenvalues. Clamping to 0.")
        gram_eigs[negative_indices] = 0
    return gram_eigs