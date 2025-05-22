import gc
import numpy as np
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm
from typing import Any, Callable, Optional, Union, Tuple

from .jacobianODE import JacobianODE, JacobianODEint
from .metrics import mase, mse, r2_score, smape
from .teacher_forcing import get_alpha_exact, get_alpha_explogapprox, get_alpha_lyap


METRIC_DICT = {
    'mse': mse,
    'mase': mase,
    'r2_score': r2_score,
    'smape': smape,
}

def horizon_aware_loss(predictions, targets, dt=1, alpha=0.1):
    """Compute an exponentially weighted horizon-aware loss.

    This loss function weights prediction errors based on their temporal distance,
    with earlier time steps having higher weights. The weighting follows an
    exponential decay with rate alpha.

    Args:
        predictions (torch.Tensor): Predicted trajectory of shape (batch, time, dim)
        targets (torch.Tensor): Ground truth trajectory of shape (batch, time, dim)
        dt (float): Time step size (default: 1)
        alpha (float): Decay rate for weighting (default: 0.1)
                     Higher values mean more weight on earlier time steps.
                     For example, alpha=0.1 means by the 10th time step,
                     the weight is e^-1 ≈ 0.368

    Returns:
        torch.Tensor: Scalar loss value, mean of weighted MSE across all batches
    """
    # Compute element-wise loss (e.g., MSE)
    loss_per_timestep = F.mse_loss(predictions, targets, reduction='none')  # (batch, time, dim)

    # Reduce over dimensions
    loss_per_timestep = loss_per_timestep.mean(dim=-1)  # (batch, time)

    # Compute exponential weights
    timesteps = torch.arange(targets.shape[-2], dtype=torch.float32, device=targets.device)
    # alpha represents the decay rate in units of dt^-1
    # thus alpha = 0.1 means that by the 10th time step, the weight is e^-1 = 0.36787944117144233
    weights = torch.exp(-alpha * timesteps)  # (time,)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Apply weights
    weighted_loss = (loss_per_timestep * weights).sum(dim=-1)  # (batches,)

    return weighted_loss.mean()  # Scalar loss

def make_loops(pts, n_loops, n_loop_pts=0):
    """Generate loop trajectories by randomly sampling and concatenating points.

    This function creates closed loops by randomly sampling points from the input
    trajectories and concatenating them with their starting points.

    Args:
        pts (torch.Tensor): Input points/trajectories
        n_loops (int): Number of loops to generate
        n_loop_pts (int): Number of points per loop (default: 0)
                         If 0, uses the length of input trajectories

    Returns:
        torch.Tensor: Generated loop trajectories of shape (n_loops, n_loop_pts+1, dim)
    """
    pts = pts.reshape(-1, pts.shape[-1])
    n_choices = torch.prod(torch.tensor(pts.shape[:-1]))
    loop_pts = pts[np.random.choice(n_choices, size=(n_loops, n_loop_pts), replace=True)]
    loop_pts = torch.cat((loop_pts, loop_pts[..., [0], :]), dim=-2)
    return loop_pts

class TeacherForcingLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler that adapts based on teacher forcing.

    This scheduler adjusts the learning rate based on the current teacher forcing
    coefficient, allowing for smoother training as teacher forcing is annealed.

    Args:
        optimizer (Optimizer): PyTorch optimizer
        lit_model (LitBase): Lightning model containing teacher forcing parameters
        min_lr (float): Minimum learning rate
        k (float): Scaling factor for learning rate adjustment (default: 0)
    """
    def __init__(self, optimizer, lit_model, min_lr, k=0):
        self.lit_model = lit_model
        self.min_lr = min_lr
        self.start_lr = optimizer.param_groups[0]['lr']
        self.k = k
        super().__init__(optimizer)
    
    def scale_factor(self, alpha):
        return alpha/(alpha + (1 - alpha)*np.exp(-self.k*alpha))

    def get_lr(self):
        # lr = min_lr + alpha_teacher_forcing * (start_lr - min_lr)
        alpha = self.lit_model.alpha_teacher_forcing
        alpha = (alpha - self.lit_model.min_alpha_teacher_forcing)/(1 - self.lit_model.min_alpha_teacher_forcing)
        new_lr = self.min_lr + self.scale_factor(alpha) * (self.start_lr - self.min_lr)
        return [new_lr for _ in self.base_lrs]

def loop_closure(batch, jac_func, dt=1, n_loops=None, n_loop_pts=None, loop_path='line',
                int_method='Trapezoid', loop_closure_interp_pts=2, mix_trajectories=True,
                alpha=1, return_loop_pts=False):
    """Compute loop closure integrals for validation of path independence.

    This function generates loop trajectories and computes their integrals using
    either line segments or splines. It's used to validate that the system's
    dynamics are path-independent (i.e., integrals around closed loops should be zero).

    Args:
        batch (torch.Tensor): Input batch of trajectories
        jac_func (Callable): Function to compute Jacobian matrices
        dt (float): Time step size (default: 1)
        n_loops (Optional[int]): Number of loops to generate (default: None)
                                If None, uses batch size
        n_loop_pts (Optional[int]): Points per loop (default: None)
                                   If None, uses trajectory length
        loop_path (str): Integration path type ('line' or 'spline') (default: 'line')
        int_method (str): Integration method (default: 'Trapezoid')
        loop_closure_interp_pts (int): Number of interpolation points (default: 2)
        mix_trajectories (bool): Whether to mix points from different trajectories (default: True)
        alpha (float): Teacher forcing coefficient (default: 1)
        return_loop_pts (bool): Whether to return loop points (default: False)

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
            If return_loop_pts is False: Loop closure integrals
            If return_loop_pts is True: Tuple of (integrals, loop points)
    """
    if n_loops is None:
        n_loops = torch.prod(torch.tensor(batch.shape[:-2]))
    if n_loop_pts is None:
        n_loop_pts = batch.shape[-2]

    if mix_trajectories:    
        loop_pts = make_loops(batch, n_loops, n_loop_pts).type(batch.dtype).to(batch.device)
    else:
        loop_pts = batch[..., torch.randperm(batch.shape[-2]), :]
        loop_pts = torch.cat((loop_pts[..., :n_loop_pts, :], loop_pts[..., [0], :]), dim=-2)
    
    loop_pts_tf = torch.zeros_like(loop_pts)
    loop_pts_tf[..., 0, :] = loop_pts[..., 0, :]
    loop_pts_tf[..., 1, :] = loop_pts[..., 1, :]

    if loop_path == 'spline':
        jacobian_ode = JacobianODE(loop_pts, jac_func, dt=dt)
        s = torch.tensor(0, dtype=batch.dtype, device=batch.device)
        t = torch.tensor((loop_pts.shape[-2] - 1)*dt, dtype=batch.dtype, device=batch.device)
        loop_int = jacobian_ode.H(s, t, N=(loop_pts.shape[-2] - 1)*loop_closure_interp_pts + 2)
    elif loop_path == 'line':
        N = loop_closure_interp_pts + 2
        loop_int = torch.zeros(*loop_pts.shape[:-2], loop_pts.shape[-1]).type(batch.dtype).to(batch.device)
        jacobian_ode = JacobianODE(loop_pts, jac_func, dt=dt, fit_spline=False, int_method=int_method)
        for _t in range(loop_pts.shape[-2] - 1):
            s = torch.tensor(_t*dt, dtype=batch.dtype, device=batch.device)
            t = torch.tensor((_t + 1)*dt, dtype=batch.dtype, device=batch.device)
            x_s = alpha*loop_pts[..., _t, :] + (1 - alpha)*loop_pts_tf[..., _t, :]
            x_t = alpha*loop_pts[..., _t + 1, :] + (1 - alpha)*loop_pts_tf[..., _t + 1, :]
            loop_ret = jacobian_ode.H(s, t, x_s, x_t, inner_path="line", N=N)
            loop_int += loop_ret
            loop_pts_tf[..., _t + 1, :] = loop_ret
    else:
        raise ValueError(f"Loop path {loop_path} not recognized")
    if return_loop_pts:
        return loop_int, loop_pts
    else:
        return loop_int

def get_jac_reg_terms(batch, jac_func, dt, deriv_func=None, int_method='Trapezoid',
                     N=10, randomize=False):
    """Compute various regularization terms for Jacobian-based training.

    This function computes several regularization terms used in training:
    - Base integration: Integration from initial point to each point
    - Sequential integration: Integration between consecutive points
    - Path consistency: Difference between direct and sequential integration
    - Loop closure: Sum of sequential integrations around loops

    Args:
        batch (torch.Tensor): Input batch of trajectories
        jac_func (Callable): Function to compute Jacobian matrices
        dt (float): Time step size
        deriv_func (Optional[Callable]): Function to compute true derivatives (default: None)
        int_method (str): Integration method (default: 'Trapezoid')
        N (int): Number of integration points (default: 10)
        randomize (bool): Whether to randomize batch and time dimensions (default: False)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - Base integration values
            - Sequential integration values
            - Path consistency values
            - Loop closure values
    """
    f_diff_jac_base = torch.zeros_like(batch)
    f_diff_jac_seq = torch.zeros_like(batch)
    if deriv_func is not None:
        f_diff_deriv_base = torch.zeros_like(batch)
        f_diff_deriv_seq = torch.zeros_like(batch)

    if randomize:
        batch_size, time_steps, dims = batch.shape

        # Randomize the batch dimension
        batch_indices = torch.randperm(batch_size)
        batch = batch[batch_indices]

        # Randomize the time dimension
        time_indices = torch.randperm(time_steps)
        batch = batch[:, time_indices, :]

    s_0 = torch.tensor(0*dt).type(batch.dtype).to(batch.device)
    x_s_0 = batch[..., 0, :]

    loop = torch.cat((batch, x_s_0.unsqueeze(-2)), dim=-2)

    deriv_vals = deriv_func(torch.arange(loop.shape[-2]).type(batch.dtype).to(batch.device)*dt, loop)

    jacobian_ode = JacobianODE(batch, jac_func, dt=dt, int_method=int_method, fit_spline=False)
    for t_idx in range(1, loop.shape[-2]):
        s = (torch.tensor(t_idx-1)*dt).type(batch.dtype).to(batch.device)
        t = (torch.tensor(t_idx)*dt).type(batch.dtype).to(batch.device)
        x_s = loop[..., t_idx - 1, :]
        x_t = loop[..., t_idx, :]

        f_diff_jac_base[..., t_idx - 1, :] = jacobian_ode.H(s_0, t, x_s_0, x_t, inner_path="line", N=N)
        f_diff_jac_seq[..., t_idx - 1, :] = jacobian_ode.H(s, t, x_s, x_t, inner_path="line", N=N)

        if deriv_func is not None:
            f_diff_deriv_base[..., t_idx - 1, :] = deriv_vals[..., t_idx, :] - deriv_vals[..., 0, :]
            f_diff_deriv_seq[..., t_idx -1, :] = deriv_vals[..., t_idx, :] - deriv_vals[..., t_idx - 1, :]

    base_integration_vals = f_diff_jac_base - f_diff_deriv_base
    seq_integration_vals = f_diff_jac_seq - f_diff_deriv_seq
    path_consistency_vals = f_diff_jac_base - torch.cumsum(f_diff_jac_seq, dim=-2)
    loop_closure_vals = f_diff_jac_seq.sum(axis=-2)

    return base_integration_vals, seq_integration_vals, path_consistency_vals, loop_closure_vals

def compute_neuralode_jacobians(neuralode, batch, t=0, batch_idx=0, dataloader_idx=0):
    """Compute Jacobian matrices for a NeuralODE model.

    This function computes the Jacobian matrices of the neural network's output
    with respect to its input using forward-mode automatic differentiation.

    Args:
        neuralode (NeuralODE): The NeuralODE model
        batch (torch.Tensor): Input batch of shape (..., T, D) or (..., D)
        t (int): Time step (unused, kept for interface consistency)
        batch_idx (int): Batch index (unused, kept for interface consistency)
        dataloader_idx (int): Dataloader index (unused, kept for interface consistency)

    Returns:
        torch.Tensor: Jacobian matrices of shape (..., T, D, D) or (..., D, D)
    """
    reshape = False
    if len(batch.shape) > 3:
        reshape = True
        batches = batch.shape[:-2]
        batch = batch.reshape(-1, batch.shape[-2], batch.shape[-1])
    # jacs = torch.func.vmap(torch.func.jacrev(lambda x: self.model.net(x)))(batch)
    jacs = torch.func.vmap(torch.func.jacfwd(lambda x: neuralode.net(x)))(batch)
    if len(jacs.shape) >= 4:
        jacs = jacs.transpose(-3, -2)
        modified_jacs = jacs.clone()
        modified_jacs = modified_jacs[..., torch.arange(jacs.shape[-4]), torch.arange(jacs.shape[-3]), :, :]
    else:
        modified_jacs = jacs
    if reshape:
        modified_jacs = modified_jacs.reshape(*batches, -1, modified_jacs.shape[-2], modified_jacs.shape[-1])
    return modified_jacs

def hessian_loss(traj, model):
    """Compute a loss based on the symmetry of Hessian matrices.

    This function computes a loss that penalizes asymmetry in the Hessian matrices
    of the model's output with respect to its input. This helps ensure the model
    learns a conservative vector field.

    Args:
        traj (torch.Tensor): Input trajectories
        model (nn.Module): Neural network model

    Returns:
        torch.Tensor: Mean norm of asymmetric part of Hessians
    """
    hessians = torch.func.vmap(torch.func.jacfwd(lambda x: model(x)))(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape[0], traj.shape[1], traj.shape[-1], traj.shape[-1], traj.shape[-1])
    hess_loss = torch.mean(torch.linalg.norm(hessians - hessians.transpose(-2, -1), dim=(-2, -1)))
    return hess_loss

def path_consistency_loss(batch, jac_func, dt, int_method='Trapezoid', N=20, pivot_ind=None):
    """Compute path consistency loss for validating integration.

    This function computes the difference between direct integration and
    sequential integration along paths, which should be zero for path-independent
    systems.

    Args:
        batch (torch.Tensor): Input batch of trajectories
        jac_func (Callable): Function to compute Jacobian matrices
        dt (float): Time step size
        int_method (str): Integration method (default: 'Trapezoid')
        N (int): Number of integration points (default: 20)
        pivot_ind (Optional[int]): Index to use as pivot point (default: None)
                                 If None, uses consecutive triplets

    Returns:
        torch.Tensor: Path consistency errors
    """
    jacobian_ode = JacobianODE(batch, jac_func, dt=dt, int_method=int_method, fit_spline=False)
    if pivot_ind is None:
        path_consistency_ret = torch.zeros_like(batch[..., 2:, :])
        range_vals = torch.arange(2, batch.shape[-2])
    else:
        path_consistency_ret = torch.zeros_like(batch[..., pivot_ind + 1:, :])
        range_vals = torch.arange(pivot_ind + 1, batch.shape[-2])
    for t_idx in range_vals:
        if pivot_ind is not None:
            r = torch.tensor(0*dt).type(batch.dtype).to(batch.device)
            x_r = batch[..., 0, :]
            s = torch.tensor(pivot_ind*dt).type(batch.dtype).to(batch.device)
            x_s = batch[..., pivot_ind, :]
        else:
            r = (torch.tensor(t_idx-2)*dt).type(batch.dtype).to(batch.device)
            x_r = batch[..., t_idx - 2, :]
            s = (torch.tensor(t_idx-1)*dt).type(batch.dtype).to(batch.device)
            x_s = batch[..., t_idx - 1, :]
        t = (torch.tensor(t_idx)*dt).type(batch.dtype).to(batch.device)
        x_t = batch[..., t_idx, :]
        jac_rs = jacobian_ode.H(r, s, x_r, x_s, inner_path="line", N=N)
        jac_st = jacobian_ode.H(s, t, x_s, x_t, inner_path="line", N=N)
        jac_rt = jacobian_ode.H(r, t, x_r, x_t, inner_path="line", N=N)
        
        if pivot_ind is None:
            path_consistency_ret[..., t_idx - 2, :] = jac_rs + jac_st - jac_rt
        else:
            path_consistency_ret[..., t_idx - pivot_ind - 1, :] = jac_rs + jac_st - jac_rt
    return path_consistency_ret


class LitBase(L.LightningModule):
    """Base Lightning module for training neural networks on dynamical systems.

    This class implements a PyTorch Lightning module for training neural networks
    to learn dynamical systems, with support for various training strategies including
    teacher forcing, noise injection, and multiple loss terms.

    Args:
        model (nn.Module): The neural network model to train
        deriv_model (Optional[nn.Module]): Optional model for computing derivatives
        dt (float): Time step size for integration (default: 1)
        eq (Optional[Any]): Optional equation object containing true dynamics
        direct (bool): Whether to use direct prediction or integration (default: True)
        save_dir (Optional[str]): Directory to save model checkpoints
        loss_func (str): Loss function type ('mse' or 'hal' for horizon-aware loss) (default: 'mse')
        alpha_hal (float): Alpha parameter for horizon-aware loss (default: 0.1)
        l2_penalty (float): L2 regularization weight (default: 0.0)
        l1_penalty (float): L1 regularization weight (default: 0.0)
        obs_noise_scale (float): Scale of observation noise to add during training (default: 0.0)
        final_obs_noise_scale (float): Final scale of observation noise after annealing (default: 0.0)
        y0_noise_scale (float): Scale of noise to add to initial conditions (default: 0.0)
        noise_annealing (bool): Whether to anneal observation noise (default: True)
        log_interval (int): Interval for logging training metrics (default: 1)
        jac_loss_interval (int): Interval for computing Jacobian loss (default: 1)
        alpha_teacher_forcing (float): Initial teacher forcing coefficient (default: 1)
        teacher_forcing_annealing (bool): Whether to anneal teacher forcing (default: True)
        gamma_teacher_forcing (float): Decay rate for teacher forcing (default: 0.999)
        teacher_forcing_update_interval (int): Interval for updating teacher forcing (default: 1)
        teacher_forcing_steps (int): Number of steps to use teacher forcing (default: 1)
        min_alpha_teacher_forcing (float): Minimum teacher forcing coefficient (default: 0)
        alpha_validation (float): Teacher forcing coefficient for validation (default: 0)
        obs_noise_scale_validation (float): Observation noise scale for validation (default: 1e-2)
        loss_func_validation (Optional[str]): Loss function for validation (default: None)
        traj_init_steps_validation (Optional[int]): Initial steps for validation trajectories (default: None)
        inner_N_validation (Optional[int]): Number of integration steps for validation (default: None)
        data_type (Optional[str]): Type of data being used (default: None)
        lles_max_T (int): Maximum time for computing Lyapunov exponents (default: 10)
        lles_n_neighbors (int): Number of neighbors for Lyapunov exponent computation (default: 3)
        optimizer (str): Optimizer to use ('AdamW' or 'Adam') (default: 'AdamW')
        optimizer_kwargs (dict): Keyword arguments for optimizer (default: {'lr': 1e-4})
        gradient_clip_val (float): Value for gradient clipping (default: 1.0)
        gradient_clip_algorithm (str): Algorithm for gradient clipping (default: 'norm')
        jacobianODEint_kwargs (dict): Keyword arguments for JacobianODE integration (default: {})
        min_traj_init_steps (int): Minimum initial steps for trajectories (default: 2)
        max_traj_init_steps (Optional[int]): Maximum initial steps for trajectories (default: None)
        use_scheduler (bool): Whether to use learning rate scheduler (default: False)
        min_lr (Optional[float]): Minimum learning rate for scheduler (default: None)
        k_scale (Optional[float]): Scaling factor for scheduler (default: None)
        jac_penalty (float): Weight for Jacobian regularization (default: 0.0)
        jac_norm_ord (str): Order of norm for Jacobian regularization (default: 'fro')
        loop_closure_training (bool): Whether to use loop closure training (default: False)
        mix_trajectories (bool): Whether to mix trajectories during training (default: True)
        loop_closure_interp_pts (int): Number of interpolation points for loop closure (default: 10)
        max_loop_closure_interp_pts (int): Maximum interpolation points for loop closure (default: 10)
        loop_closure_int_method (str): Integration method for loop closure (default: 'Trapezoid')
        n_loops (Optional[int]): Number of loops for loop closure (default: None)
        n_loop_pts (Optional[int]): Number of points per loop (default: None)
        loop_path (str): Path type for loop closure ('line' or 'spline') (default: 'line')
        loop_closure_weight (float): Weight for loop closure loss (default: 1.0)
        final_loop_closure_weight (Optional[float]): Final weight for loop closure loss (default: None)
        min_loop_closure_loss (float): Minimum loop closure loss (default: 0.0)
        obs_noise_scale_loop (Optional[float]): Observation noise scale for loop closure (default: None)
        trajectory_training (bool): Whether to use trajectory training (default: True)
        random_walk_training (bool): Whether to use random walk training (default: False)
        random_walk_weight (float): Weight for random walk loss (default: 0.0)
        min_random_walk_steps (int): Minimum steps for random walk (default: 1)
        max_random_walk_steps (Optional[int]): Maximum steps for random walk (default: None)
        rescaling_sigma (float): Sigma for rescaling (default: 1)
        deriv_trajectory_weight (float): Weight for derivative trajectory loss (default: 1)
        final_deriv_trajectory_weight (Optional[float]): Final weight for derivative trajectory loss (default: None)
        base_integration_weight (float): Weight for base integration loss (default: 1)
        final_base_integration_weight (Optional[float]): Final weight for base integration loss (default: None)
        seq_integration_weight (float): Weight for sequential integration loss (default: 1)
        final_seq_integration_weight (Optional[float]): Final weight for sequential integration loss (default: None)
        path_consistency_weight (float): Weight for path consistency loss (default: 1)
        final_path_consistency_weight (Optional[float]): Final weight for path consistency loss (default: None)
        node_jacobians_weight (float): Weight for NeuralODE Jacobian loss (default: 0)
        use_base_deriv_pt (bool): Whether to use base derivative point (default: False)
        base_pt_init (Optional[torch.Tensor]): Initial base point (default: None)
        base_deriv_pt_init (Optional[torch.Tensor]): Initial base derivative point (default: None)
        hessian_weight (float): Weight for Hessian loss (default: 0)
        n_delays (Optional[int]): Number of delays for embedding (default: None)
        obs_dim (Optional[int]): Dimension of observations (default: None)
        obs_only_loss (bool): Whether to use observation-only loss (default: False)
        early_stopping_patience (int): Patience for early stopping (default: 5)
        early_stopping_mode (str): Mode for early stopping ('min' or 'max') (default: 'min')
        percent_thresh (float): Threshold for percent improvement (default: 0.01)
        mu (float): Mean for noise (default: 0)
        sigma (float): Standard deviation for noise (default: 1)
    """
    def __init__(
                    self, 
                    model,
                    deriv_model=None,
                    dt=1,
                    eq=None,
                    direct=True,
                    save_dir=None, 
                    loss_func='mse',
                    alpha_hal=0.1,
                    l2_penalty=0.0,
                    l1_penalty=0.0,
                    obs_noise_scale=0.0,
                    final_obs_noise_scale=0.0,
                    y0_noise_scale=0.0,
                    noise_annealing=True,
                    log_interval=1,
                    jac_loss_interval=1,
                    alpha_teacher_forcing=1,
                    teacher_forcing_annealing=True,
                    gamma_teacher_forcing=0.999,
                    teacher_forcing_update_interval=1,
                    teacher_forcing_steps=1,
                    min_alpha_teacher_forcing=0,
                    alpha_validation=0,
                    obs_noise_scale_validation=1e-2,
                    loss_func_validation=None,
                    traj_init_steps_validation=None,
                    inner_N_validation=None,
                    data_type=None,
                    lles_max_T=10,
                    lles_n_neighbors=3,
                    optimizer='AdamW',
                    optimizer_kwargs={'lr': 1e-4},
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm='norm',
                    jacobianODEint_kwargs={},
                    min_traj_init_steps=2,
                    max_traj_init_steps=None,
                    use_scheduler=False,
                    min_lr=None,
                    k_scale=None,
                    jac_penalty=0.0,
                    jac_norm_ord='fro',
                    loop_closure_training=False,
                    mix_trajectories=True,  
                    loop_closure_interp_pts=10,
                    max_loop_closure_interp_pts=10,
                    loop_closure_int_method='Trapezoid',
                    n_loops=None,
                    n_loop_pts=None,
                    loop_path='line',
                    loop_closure_weight=1.0,
                    final_loop_closure_weight=None,
                    min_loop_closure_loss=0.0,
                    obs_noise_scale_loop=None,
                    trajectory_training=True,
                    random_walk_training=False,
                    random_walk_weight=0.0,
                    min_random_walk_steps=1,
                    max_random_walk_steps=None,
                    rescaling_sigma=1,
                    deriv_trajectory_weight=1,
                    final_deriv_trajectory_weight=None,
                    base_integration_weight=1,
                    final_base_integration_weight=None,
                    seq_integration_weight=1,
                    final_seq_integration_weight=None,
                    path_consistency_weight=1,
                    final_path_consistency_weight=None,
                    node_jacobians_weight=0,
                    use_base_deriv_pt=False,
                    base_pt_init=None,
                    base_deriv_pt_init=None,
                    hessian_weight=0,
                    n_delays=None,
                    obs_dim=None,
                    obs_only_loss=False,
                    early_stopping_patience=5,
                    early_stopping_mode='min',
                    percent_thresh=0.01,
                    mu=0,
                    sigma=1
                ):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.deriv_model = deriv_model
        self.dt = dt
        self.eq = eq
        self.direct = direct

        self.save_dir = save_dir
        self.alpha_hal = alpha_hal
        if loss_func == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_func == 'horizon_aware':
            self.criterion = self.horizon_aware_criterion
        else: # not implemented
            raise ValueError(f"Loss function {loss_func} not implemented")
 
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.obs_noise_scale = obs_noise_scale
        self.final_obs_noise_scale = final_obs_noise_scale
        self.y0_noise_scale = y0_noise_scale
        self.noise_annealing = noise_annealing
        self.log_interval = log_interval
        self.jac_loss_interval = jac_loss_interval

        self.alpha_teacher_forcing = alpha_teacher_forcing
        self.teacher_forcing_annealing = teacher_forcing_annealing
        self.gamma_teacher_forcing = gamma_teacher_forcing
        self.teacher_forcing_steps = teacher_forcing_steps
        self.teacher_forcing_update_interval = teacher_forcing_update_interval

        self.min_alpha_teacher_forcing = min_alpha_teacher_forcing
        self.alpha_validation = alpha_validation
        self.obs_noise_scale_validation = obs_noise_scale_validation

        if loss_func_validation is None:
            self.criterion_validation = self.criterion
        else:
            if loss_func_validation == 'mse':
                self.criterion_validation = nn.MSELoss()
            elif loss_func_validation == 'horizon_aware':
                self.criterion_validation = self.horizon_aware_criterion
            else:
                raise ValueError(f"Loss function {loss_func_validation} not implemented")
        
        self.traj_init_steps_validation = traj_init_steps_validation
        self.inner_N_validation = inner_N_validation
        

        self.data_type = data_type

        
        self.lles_max_T = lles_max_T
        self.lles_n_neighbors = lles_n_neighbors

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        self.jacobianODEint_kwargs = jacobianODEint_kwargs
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.k_scale = k_scale
        self.jac_penalty = jac_penalty
        self.jac_norm_ord = jac_norm_ord
        self.loop_closure_training = loop_closure_training
        self.loop_closure_interp_pts = loop_closure_interp_pts
        self.max_loop_closure_interp_pts = max_loop_closure_interp_pts
        if self.max_loop_closure_interp_pts is None:
            self.max_loop_closure_interp_pts = self.loop_closure_interp_pts
        self.loop_closure_int_method = loop_closure_int_method
        self.n_loops = n_loops
        self.n_loop_pts = n_loop_pts
        self.loop_path = loop_path
        self.loop_closure_weight = loop_closure_weight
        self.final_loop_closure_weight = final_loop_closure_weight
        self.min_loop_closure_loss = min_loop_closure_loss
        if final_loop_closure_weight is None:
            self.final_loop_closure_weight = loop_closure_weight
        else:
            self.final_loop_closure_weight = final_loop_closure_weight
        self.obs_noise_scale_loop = obs_noise_scale_loop
        self.mix_trajectories = mix_trajectories
        self.min_traj_init_steps = min_traj_init_steps
        if max_traj_init_steps is None:
            self.max_traj_init_steps = self.min_traj_init_steps
        else:
            self.max_traj_init_steps = max_traj_init_steps
        self.trajectory_training = trajectory_training
        self.random_walk_training = random_walk_training
        self.random_walk_weight = random_walk_weight
        self.min_random_walk_steps = min_random_walk_steps
        if max_random_walk_steps is None:
            self.max_random_walk_steps = self.min_random_walk_steps
        else:
            self.max_random_walk_steps = max_random_walk_steps
        self.deriv_trajectory_weight = deriv_trajectory_weight
        self.rescaling_sigma = rescaling_sigma
        self.base_integration_weight = base_integration_weight
        if final_base_integration_weight is None:
            self.final_base_integration_weight = base_integration_weight
        else:
            self.final_base_integration_weight = final_base_integration_weight
        self.seq_integration_weight = seq_integration_weight
        if final_seq_integration_weight is None:
            self.final_seq_integration_weight = seq_integration_weight
        else:
            self.final_seq_integration_weight = final_seq_integration_weight
        self.path_consistency_weight = path_consistency_weight
        if final_path_consistency_weight is None:
            self.final_path_consistency_weight = path_consistency_weight
        else:
            self.final_path_consistency_weight = final_path_consistency_weight
        self.node_jacobians_weight = node_jacobians_weight
        # self.train_dataloader_names = []
        # self.val_dataloader_names = []
        
        self.use_base_deriv_pt = use_base_deriv_pt

        if self.use_base_deriv_pt:
            if base_pt_init is None:
                # self.base_pt = nn.Parameter(torch.zeros(1, 1, self.model.input_dim))
                self.base_pt = nn.Parameter(torch.randn(self.model.input_dim))
            else:
                self.base_pt = nn.Parameter(base_pt_init)
            if base_deriv_pt_init is None:
                # self.base_deriv_pt = nn.Parameter(torch.zeros(self.model.input_dim))
                self.base_deriv_pt = nn.Parameter(torch.randn(self.model.input_dim))
            else:
                self.base_deriv_pt = nn.Parameter(base_deriv_pt_init)
            # repeat the base_deriv_pt for each batch
            self.base_deriv_func = self.base_deriv_func
        else:
            self.base_pt = None
            self.base_deriv_pt = None
            self.base_deriv_func = None
        
        self.hessian_weight = hessian_weight
        self.l1_penalty = l1_penalty

        self.n_delays = n_delays
        self.obs_dim = obs_dim
        self.obs_only_loss = obs_only_loss
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_mode = early_stopping_mode
        self.percent_thresh = percent_thresh

        self.mu = mu
        self.sigma = sigma

        # Initialize validation loss tracking
        self.validation_losses = []
        self.percent_improvements = []
    def base_deriv_func(self, _t, _x):
        """Compute the base derivative function.

        Args:
            _t (float): Time (unused)
            _x (torch.Tensor): State

        Returns:
            torch.Tensor: Derivative of state
        """
        return self.base_deriv_pt

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Model output
        """
        return self.model(x)

    def horizon_aware_criterion(self, x, y):
        """Compute horizon-aware loss between predictions and targets.

        Args:
            x (torch.Tensor): Predictions
            y (torch.Tensor): Targets

        Returns:
            torch.Tensor: Horizon-aware loss value
        """
        return horizon_aware_loss(x, y, alpha=self.alpha_hal)
    
    def trajectory_model_step(
                    self, 
                    batch, 
                    batch_idx=0, 
                    dataloader_idx=0, 
                    all_metrics=False, 
                    direct=None, 
                    obs_noise_scale=None, 
                    noise_annealing=None,
                    alpha_teacher_forcing=None,
                    teacher_forcing_steps=None,
                    jacobianODEint_kwargs=None,
                    rand_init_traj_steps=True,
                    criterion=None,
                    verbose=False,
                ):
        """Perform a single training step for trajectory prediction.

        Args:
            batch (torch.Tensor): Input batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of current dataloader
            all_metrics (bool): Whether to compute all metrics
            direct (Optional[bool]): Whether to use direct prediction
            obs_noise_scale (Optional[float]): Scale of observation noise
            noise_annealing (Optional[bool]): Whether to anneal noise
            alpha_teacher_forcing (Optional[float]): Teacher forcing coefficient
            teacher_forcing_steps (Optional[int]): Number of teacher forcing steps
            jacobianODEint_kwargs (Optional[dict]): Integration parameters
            rand_init_traj_steps (bool): Whether to randomize initial steps
            criterion (Optional[Callable]): Custom loss function
            verbose (bool): Whether to print progress

        Returns:
            dict: Dictionary containing loss values and metrics
        """
        if direct is None:
            direct = self.direct
        if obs_noise_scale is None:
            obs_noise_scale = self.obs_noise_scale
        if noise_annealing is None:
            noise_annealing = self.noise_annealing
        if alpha_teacher_forcing is None:
            alpha_teacher_forcing = self.alpha_teacher_forcing
        if teacher_forcing_steps is None:
            teacher_forcing_steps = self.teacher_forcing_steps
        if criterion is None:
            criterion = self.criterion
        if jacobianODEint_kwargs is None:
            jacobianODEint_kwargs = self.jacobianODEint_kwargs

        
        alpha = alpha_teacher_forcing

        if noise_annealing:
            pct_anneal = (alpha - self.min_alpha_teacher_forcing)/(1 - self.min_alpha_teacher_forcing)
            obs_noise_scale = pct_anneal*(self.obs_noise_scale - self.final_obs_noise_scale) + self.final_obs_noise_scale

        batch = batch.type(self.dtype)
        label = batch.detach().clone() # Detach to prevent gradient flow through label
        batch = batch + (torch.randn(*batch.shape)*obs_noise_scale).type(batch.dtype).to(batch.device)

        # sample traj_init_steps as an integer between min_traj_init_steps and max_traj_init_steps inclusive
        if rand_init_traj_steps:
            traj_init_steps = np.random.randint(self.min_traj_init_steps, self.max_traj_init_steps + 1)
            jacobianODEint_kwargs['traj_init_steps'] = traj_init_steps
        else:
            if 'traj_init_steps' not in jacobianODEint_kwargs:
                jacobianODEint_kwargs['traj_init_steps'] = 2
        
        if self.y0_noise_scale > 0:
            batch[..., :jacobianODEint_kwargs['traj_init_steps'], :] = batch[..., :jacobianODEint_kwargs['traj_init_steps'], :] + (torch.randn(*batch[..., :jacobianODEint_kwargs['traj_init_steps'], :].shape)*self.y0_noise_scale).type(batch.dtype).to(batch.device)

        if 'inner_path' in jacobianODEint_kwargs:
            if jacobianODEint_kwargs['inner_path'] == 'random_walk':
                n_random_walk_steps = np.random.randint(self.min_random_walk_steps, self.max_random_walk_steps + 1)
                jacobianODEint_kwargs['n_random_walk_steps'] = n_random_walk_steps

        if direct:
            if self.deriv_model is not None:
                deriv_func = self.deriv_model
            elif self.use_base_deriv_pt:
                batch = torch.cat([self.base_pt.repeat(batch.shape[0], 1, 1), batch], dim=1)
                deriv_func = self.base_deriv_func
                jacobianODEint_kwargs['traj_init_steps'] = 2
                jacobianODEint_kwargs['fast_mode_base_ind'] = 0
            else:
                deriv_func = None
            jacobian_odeint = JacobianODEint(self.compute_jacobians, self.dt)
            outputs = jacobian_odeint.generate_dynamics(
                                        batch, 
                                        verbose=verbose,
                                        alpha_teacher_forcing=alpha,
                                        teacher_forcing_steps=teacher_forcing_steps,
                                        # deriv_func=self.model if 'NeuralODE' in self.model.__class__.__name__ else None,
                                        # deriv_func=lambda _t, _x: self.eq.rhs(_x, _t),
                                        deriv_func=deriv_func,
                                        scale_interp_pts=True,
                                        fast_mode=True,
                                        # fast_mode_base_ind=1, # TODO: CHECK THIS!!!!!!
                                        **jacobianODEint_kwargs
                                    )
            if self.use_base_deriv_pt:
                outputs = outputs[:, 1:, :]
        else:
            outputs = torch.zeros(batch.shape).to(batch.device)
            if self.deriv_model is not None:
                model = self.deriv_model
            else:
                model = self.model
            if any(model_type in model.__class__.__name__ for model_type in ['NeuralODE', 'Transformer', 'MLP']):
                outputs = model.generate(batch, alpha=alpha_teacher_forcing)
            else:
                outputs = model(batch, alpha=alpha_teacher_forcing)
            if 'shPLRNN' in model.__class__.__name__:
                outputs = outputs[0]
        
        if self.obs_only_loss and self.n_delays is not None and self.n_delays > 1:
            
            if not direct:
                outputs_cropped = outputs[..., 1:, :][..., :self.obs_dim]
                label_cropped = label[..., 1:, :][..., :self.obs_dim]
            else:
                outputs_cropped = outputs[..., jacobianODEint_kwargs['traj_init_steps']:, :][..., :self.obs_dim]
                label_cropped = label[..., jacobianODEint_kwargs['traj_init_steps']:, :][..., :self.obs_dim]
        else:
            if not direct:
                outputs_cropped = outputs[..., 1:, :]
                label_cropped = label[..., 1:, :]
            else:
                outputs_cropped = outputs[..., jacobianODEint_kwargs['traj_init_steps']:, :]
                label_cropped = label[..., jacobianODEint_kwargs['traj_init_steps']:, :]
        loss = criterion(outputs_cropped, label_cropped)

        if all_metrics:
            metric_vals = self.calc_metrics(label_cropped, outputs_cropped)
        else:
            metric_vals = {}
            metric_vals['mase'] = mase(label_cropped, outputs_cropped)
            metric_vals['r2_score'] = r2_score(label_cropped, outputs_cropped)

        return {'loss': loss, 'metric_vals': metric_vals, 'outputs': outputs}
    
    def loop_closure_model_step(
            self, 
            batch, 
            batch_idx=0, 
            dataloader_idx=0,
            obs_noise_scale_loop=None, 
            noise_annealing=None,
            mix_trajectories=True,
            n_loops=None,
            n_loop_pts=None,
            loop_path='line',
            loop_closure_interp_pts=None,
            loop_closure_int_method=None,
        ):
        """Perform a single training step for loop closure.

        Args:
            batch (torch.Tensor): Input batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of current dataloader
            obs_noise_scale_loop (Optional[float]): Scale of observation noise
            noise_annealing (Optional[bool]): Whether to anneal noise
            mix_trajectories (bool): Whether to mix trajectories
            n_loops (Optional[int]): Number of loops
            n_loop_pts (Optional[int]): Points per loop
            loop_path (str): Path type for loop closure
            loop_closure_interp_pts (Optional[int]): Interpolation points
            loop_closure_int_method (Optional[str]): Integration method

        Returns:
            dict: Dictionary containing loss values and metrics
        """
        if obs_noise_scale_loop is None:
            obs_noise_scale_loop = self.obs_noise_scale_loop
        if noise_annealing is None:
            noise_annealing = self.noise_annealing
        if mix_trajectories is None:
            mix_trajectories = self.mix_trajectories
        if n_loops is None:
            n_loops = self.n_loops
        if n_loop_pts is None:
            n_loop_pts = self.n_loop_pts
        if loop_path is None:
            loop_path = self.loop_path
        if loop_closure_int_method is None:
            loop_closure_int_method = self.loop_closure_int_method
        if noise_annealing:
            alpha = (self.alpha_teacher_forcing - self.min_alpha_teacher_forcing)/(1 - self.min_alpha_teacher_forcing)
            obs_noise_scale_loop = (alpha - self.min_alpha_teacher_forcing)/(1 - self.min_alpha_teacher_forcing)*obs_noise_scale_loop

        if loop_closure_interp_pts is None:
            loop_closure_interp_pts = np.random.randint(self.loop_closure_interp_pts, self.max_loop_closure_interp_pts + 1)

        if obs_noise_scale_loop > 0:
            batch = batch +(torch.randn(*batch.shape)*obs_noise_scale_loop).type(batch.dtype).to(batch.device)

        loop_int = loop_closure(batch, self.compute_jacobians, dt=self.dt, n_loops=n_loops, n_loop_pts=n_loop_pts, loop_path=loop_path, loop_closure_interp_pts=loop_closure_interp_pts, mix_trajectories=mix_trajectories, int_method=loop_closure_int_method)
        # loop_int, err_bound = loop_closure_with_est_error(batch, self.compute_jacobians, dt=self.dt, n_loops=n_loops, n_loop_pts=n_loop_pts, loop_path=loop_path, loop_closure_interp_pts=loop_closure_interp_pts, mix_trajectories=mix_trajectories, int_method=loop_closure_int_method, return_err_bound=True)


        loop_zeros = torch.zeros_like(loop_int)

        # loop_loss = self.criterion(loop_zeros, loop_int)
        loop_loss = (loop_int**2).mean()
        # loop_loss = torch.clamp(torch.linalg.norm(loop_int, dim=-1) - err_bound, 0, None).mean()

        metric_vals = dict(
            mse=mse(loop_zeros, loop_int),
            # r2_score=r2_score(loop_zeros.flatten(), loop_int.flatten())
        )

        return {'loss': loop_loss, 'metric_vals': metric_vals, 'outputs': loop_int}
    
    # def shortcuts_model_step(
    #         self,
    #         batch,
    #         batch_idx=0,
    #         dataloader_idx=0,
    #         obs_noise_scale=None,
    #         noise_annealing=None,
            
    #     )
        
    def update_alpha_teacher_forcing(self, jacs_pred, batch_idx):
        """Update the teacher forcing coefficient based on training progress.

        Args:
            jacs_pred (torch.Tensor): Predicted Jacobians
            batch_idx (int): Current batch index
        """
        if self.teacher_forcing_annealing and (batch_idx + 1) % self.teacher_forcing_update_interval == 0:
            # alpha = get_alpha_explogapprox(torch.linalg.matrix_exp(jacs_pred*self.dt))
            alpha = get_alpha_lyap(torch.linalg.matrix_exp(jacs_pred*self.dt))
            if isinstance(alpha, torch.Tensor):
                alpha = float(alpha.cpu())
            alpha_teacher_forcing = self.alpha_teacher_forcing*self.gamma_teacher_forcing + (1 - self.gamma_teacher_forcing)*alpha
            alpha_teacher_forcing = alpha_teacher_forcing if alpha_teacher_forcing > self.min_alpha_teacher_forcing else self.min_alpha_teacher_forcing
            self.alpha_teacher_forcing = alpha_teacher_forcing

    def get_pred_jacs(self, batch):
        """Get predicted Jacobians for a batch.

        Args:
            batch (torch.Tensor): Input batch

        Returns:
            torch.Tensor: Predicted Jacobians
        """
        if any(model_type in self.model.__class__.__name__ for model_type in ['NeuralODE']) and self.jac_penalty == 0:
            with torch.no_grad():
                jacs_pred = self.compute_jacobians(batch)
        else:
            jacs_pred = self.compute_jacobians(batch)
        return jacs_pred


    def get_true_jacs(self, batch):
        """Get true Jacobians for a batch.

        Args:
            batch (torch.Tensor): Input batch

        Returns:
            torch.Tensor: True Jacobians
        """
        if self.data_type == 'dysts':
            batch = batch.cpu().numpy()
        # return self.eq.jac(batch, np.arange(batch.shape[1]), discrete=discrete)
        return self.eq.jac(batch*self.sigma + self.mu, np.arange(batch.shape[1]))

    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Perform a single training step.

        Args:
            batch (torch.Tensor): Input batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of current dataloader

        Returns:
            torch.Tensor: Training loss
        """
        if 'NeuralODE' in self.model.__class__.__name__ and not self.teacher_forcing_annealing:
            jacs_pred = None
        else:
            jacs_pred = self.get_pred_jacs(batch)
        if jacs_pred is not None:
            jac_norm = torch.linalg.norm(jacs_pred, dim=(-2, -1), ord=self.jac_norm_ord).mean()
            self.update_alpha_teacher_forcing(jacs_pred.detach(), batch_idx)
        else:
            jac_norm = None
        
        train_rets = {}
        if self.trajectory_training:
            train_rets['trajectory'] = self.trajectory_model_step(batch, batch_idx, dataloader_idx)
        if self.deriv_model is not None and not (not self.direct and 'NeuralODE' in self.deriv_model.__class__.__name__):
            train_rets['deriv_trajectory'] = self.trajectory_model_step(batch, batch_idx, dataloader_idx, direct=False)
        if self.loop_closure_training:
            if ('NeuralODE' in self.model.__class__.__name__ and self.loop_closure_weight > 0) or ('NeuralODE' not in self.model.__class__.__name__):
                train_rets['loop_closure'] = self.loop_closure_model_step(batch, batch_idx, dataloader_idx)
                if self.min_loop_closure_loss > 0:
                    train_rets['loop_closure']['loss'] = torch.maximum(train_rets['loop_closure']['loss'], torch.tensor(self.min_loop_closure_loss)) - self.min_loop_closure_loss
        if self.path_consistency_weight > 0:
            path_consistency_ret = path_consistency_loss(batch, self.compute_jacobians, dt=self.dt, int_method=self.loop_closure_int_method, N=self.jacobianODEint_kwargs['inner_N'], pivot_ind=self.min_traj_init_steps)
            train_rets['path_consistency'] = {'loss': torch.mean(path_consistency_ret**2), 'metric_vals': {}}

        if self.hessian_weight > 0:
            train_rets['hessian'] = {'loss': hessian_loss(batch, self.model), 'metric_vals': {}}
        if self.random_walk_training:
            jacobianODEint_kwargs = self.jacobianODEint_kwargs.copy()
            jacobianODEint_kwargs['inner_path'] = 'random_walk'
            train_rets['random_walk'] = self.trajectory_model_step(batch, batch_idx, dataloader_idx, jacobianODEint_kwargs=jacobianODEint_kwargs)

        
        if self.deriv_model is not None and 'NeuralODE' not in self.model.__class__.__name__:
            base_integration_vals, seq_integration_vals, path_consistency_vals, loop_closure_vals = get_jac_reg_terms(batch, self.compute_jacobians, dt=self.dt, deriv_func=self.deriv_model, int_method=self.loop_closure_int_method, N=self.jacobianODEint_kwargs['inner_N'], randomize=True)
            # base_integration_vals, seq_integration_vals, path_consistency_vals, loop_closure_vals = get_jac_reg_terms(batch, self.compute_jacobians, dt=self.dt, deriv_func=lambda _t, _x: self.eq.rhs(_x, _t), int_method=self.loop_closure_int_method, N=self.jacobianODEint_kwargs['inner_N'], randomize=True)
            train_rets['base_integration'] = {'loss': torch.mean(base_integration_vals**2), 'metric_vals': {}}
            train_rets['seq_integration'] = {'loss': torch.mean(seq_integration_vals**2), 'metric_vals': {}}
            train_rets['path_consistency'] = {'loss': torch.mean(path_consistency_vals**2), 'metric_vals': {}}
            train_rets['loop_closure'] = {'loss': torch.mean(loop_closure_vals**2), 'metric_vals': {}}
            if self.min_loop_closure_loss > 0:
                train_rets['loop_closure']['loss'] = torch.maximum(train_rets['loop_closure']['loss'], torch.tensor(self.min_loop_closure_loss)) - self.min_loop_closure_loss

        if self.deriv_model is not None and 'NeuralODE' in self.deriv_model.__class__.__name__ and self.node_jacobians_weight > 0:
            node_jacs_pred = compute_neuralode_jacobians(self.deriv_model, batch)
            train_rets['node_jacobians'] = {'loss': torch.mean((node_jacs_pred - jacs_pred)**2), 'metric_vals': {}}

        if 'NeuralODE' in self.model.__class__.__name__:
            train_rets['trajectory_indirect'] = self.trajectory_model_step(batch, batch_idx, dataloader_idx, direct=False)

        total_loss = 0
        alpha = (self.alpha_teacher_forcing - self.min_alpha_teacher_forcing)/(1 - self.min_alpha_teacher_forcing)
        trajectory_weight = 1
        loop_closure_weight = (1 - alpha)*self.final_loop_closure_weight + alpha*self.loop_closure_weight
        random_walk_weight = self.random_walk_weight
        deriv_trajectory_weight = self.deriv_trajectory_weight
        hessian_weight = self.hessian_weight
       
        for pred_type, ret_dict in train_rets.items():
            if torch.isnan(ret_dict['loss']):
                # Warn if loss is nan
                print(f"Warning: Loss is nan for pred type {pred_type} on epoch {self.current_epoch} batch {batch_idx}")
            loss_val = ret_dict['loss'] if not torch.isnan(ret_dict['loss']) else 0
            loss_weight = 1
            if 'trajectory' in pred_type:
                loss_weight *= trajectory_weight
            if 'deriv_trajectory' in pred_type:
                loss_weight *= deriv_trajectory_weight
            if 'loop_closure' in pred_type:
                # loss_weight *= loop_closure_weight * (1/self.n_loop_pts)
                loss_weight *= loop_closure_weight
            if 'random_walk' in pred_type:
                loss_weight *= random_walk_weight
            if 'base_integration' in pred_type:
                loss_weight *= self.base_integration_weight
            if 'seq_integration' in pred_type:
                loss_weight *= self.seq_integration_weight
            if 'path_consistency' in pred_type:
                loss_weight *= self.path_consistency_weight
            if 'node_jacobians' in pred_type:
                loss_weight *= self.node_jacobians_weight
            if 'hessian' in pred_type:
                loss_weight *= hessian_weight
            total_loss += loss_weight*loss_val
        if jac_norm is not None:
            total_loss += self.jac_penalty*jac_norm
        l1_loss = torch.sum(torch.abs(torch.cat([p.view(-1) for p in self.get_main_params()], dim=0)))
        if self.l1_penalty > 0:
            total_loss += self.l1_penalty*l1_loss
        
        if ((batch_idx + 1) % self.log_interval) == 0:
            self.log_training_metrics(
                train_rets=train_rets,
                total_loss=total_loss,
                jac_norm=jac_norm,
                l1_loss=l1_loss,
                batch=batch,
                jacs_pred=jacs_pred,
                batch_idx=batch_idx,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True
            ) 

        return total_loss

    def jac_combo_validation_step(self, batch, batch_idx, dataloader_idx, log_metrics=True):
        """Perform validation step with Jacobian combination.

        Args:
            batch (torch.Tensor): Input batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of current dataloader
            log_metrics (bool): Whether to log metrics

        Returns:
            dict: Dictionary containing validation metrics
        """
        alpha = self.alpha_validation

        val_rets = {}
        jac_outputs = self.model.generate_jac(batch, alpha=alpha, jacobianODEint_kwargs=self.jacobianODEint_kwargs)
        metric_vals = self.calc_metrics(batch[..., 1:, :], jac_outputs[..., 1:, :])
        val_rets['trajectory'] = {'loss': torch.mean((jac_outputs[..., 1:, :] - batch[..., 1:, :])**2), 'metric_vals': metric_vals, 'outputs': jac_outputs}
        
        loop_closure_ret = loop_closure(batch, self.compute_jacobians, dt=self.dt, n_loops=self.n_loops, n_loop_pts=self.n_loop_pts, loop_path=self.loop_path, loop_closure_interp_pts=self.loop_closure_interp_pts, mix_trajectories=self.mix_trajectories, int_method=self.loop_closure_int_method)
        val_loop_closure = {'loss': torch.mean(loop_closure_ret**2), 'metric_vals': {}}

        if log_metrics:
            self.log_validation_metrics(
                val_rets=val_rets,
                batch=batch,
                sync_dist=True,
                val_loop_closure=val_loop_closure,
            )
        
        return sum(val_rets[pred_type]['loss'] for pred_type in val_rets.keys())
    def validation_step(self, batch, batch_idx=0, dataloader_idx=0, log_metrics=True):
        """Perform a single validation step.

        Args:
            batch (torch.Tensor): Input batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of current dataloader
            log_metrics (bool): Whether to log metrics

        Returns:
            dict: Dictionary containing validation metrics
        """
        if 'JacComboODE' in self.model.__class__.__name__:
            self.jac_combo_validation_step(batch, batch_idx, dataloader_idx)
            return

        # dataloader_name = self.val_dataloader_names[dataloader_idx]
        jacobianODEint_kwargs = self.jacobianODEint_kwargs.copy()
        jacobianODEint_kwargs['traj_init_steps'] = self.traj_init_steps_validation
        jacobianODEint_kwargs['inner_N'] = self.inner_N_validation
        model_step_kwargs = {
            'alpha_teacher_forcing': self.alpha_validation, 
            'obs_noise_scale': self.obs_noise_scale_validation,
            'criterion': self.criterion_validation,
            'noise_annealing': False,
            'jacobianODEint_kwargs': jacobianODEint_kwargs,
            'rand_init_traj_steps': False,
        }

        val_rets = {}
        val_rets['trajectory'] = self.trajectory_model_step(batch, batch_idx, dataloader_idx, **model_step_kwargs)
        if self.path_consistency_weight > 0:
            if 'NeuralODE' not in self.model.__class__.__name__:
                path_consistency_ret = path_consistency_loss(batch, self.compute_jacobians, dt=self.dt, int_method=self.loop_closure_int_method, N=self.jacobianODEint_kwargs['inner_N'], pivot_ind=self.min_traj_init_steps)
                val_loop_closure = {'loss': torch.mean(path_consistency_ret**2), 'metric_vals': {}}
                n_loop_pts = 3
                loop_error_est = None
            else:
                val_loop_closure = None
                n_loop_pts = None
                loop_error_est = None
        else:
            if 'NeuralODE' not in self.model.__class__.__name__:
                val_loop_closure = self.loop_closure_model_step(batch, batch_idx, dataloader_idx)
            else:
                val_loop_closure = None
                # n_loop_pts = None
                # loop_error_est  = None

            # val_loop_closure = self.loop_closure_model_step(batch, batch_idx, dataloader_idx)
            loop_error_est = None
            n_loop_pts = None

        if log_metrics:
            self.log_validation_metrics(
                val_rets=val_rets,
                batch=batch,
                sync_dist=True,
                val_loop_closure=val_loop_closure,
                loop_error_est=loop_error_est,
            )

        total_loss = sum(val_rets[pred_type]['loss'] for pred_type in val_rets.keys())
        
        # Store the batch loss for later use in on_validation_epoch_end
        if not hasattr(self, 'current_epoch_val_losses'):
            self.current_epoch_val_losses = []
        self.current_epoch_val_losses.append(total_loss.item())

        return total_loss

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch.

        Tracks validation losses for percent improvement calculation and updates
        teacher forcing coefficient if enabled.
        """
        # Track validation losses for percent improvement calculation
        if hasattr(self, 'current_epoch_val_losses'):
            mean_val_loss = sum(self.current_epoch_val_losses) / len(self.current_epoch_val_losses)
            self.validation_losses.append(mean_val_loss)
            if len(self.validation_losses) > 1:
                prev_loss = self.validation_losses[-2]
                curr_loss = self.validation_losses[-1]
                if prev_loss > curr_loss:
                    percent_improvement = (prev_loss - curr_loss) / prev_loss
                    self.percent_improvements.append(percent_improvement)
                    # print(f"  Current percent improvement: {percent_improvement:.4f}")
                else:
                    self.percent_improvements.append(0.0)
                    # print(f"  No improvement in validation loss")
                if self.current_epoch > 0:
                    self.log("percent_improvement", self.percent_improvements[-1], on_epoch=True, sync_dist=True)
            
            # Clear the current epoch losses
            self.current_epoch_val_losses = []

    def log_training_metrics(self, train_rets, total_loss, jac_norm, l1_loss, batch,
                           jacs_pred=None, batch_idx=0, on_step=True, on_epoch=True,
                           sync_dist=True, prog_bar=True):
        """Log training metrics.

        Args:
            train_rets (dict): Training results
            total_loss (torch.Tensor): Total loss value
            jac_norm (torch.Tensor): Jacobian norm
            l1_loss (torch.Tensor): L1 loss value
            batch (torch.Tensor): Input batch
            jacs_pred (Optional[torch.Tensor]): Predicted Jacobians
            batch_idx (int): Current batch index
            on_step (bool): Whether to log on step
            on_epoch (bool): Whether to log on epoch
            sync_dist (bool): Whether to sync across distributed training
            prog_bar (bool): Whether to show in progress bar
        """
        # Log losses and basic metrics
        alpha = (self.alpha_teacher_forcing - self.min_alpha_teacher_forcing)/(1 - self.min_alpha_teacher_forcing)
        trajectory_weight = 1
        loop_closure_weight = (1 - alpha)*self.final_loop_closure_weight + alpha*self.loop_closure_weight
        random_walk_weight = self.random_walk_weight
        
        for pred_type, ret_dict in train_rets.items():
            loss, metric_vals = ret_dict['loss'], ret_dict['metric_vals']
            self.log(f"{pred_type} train_loss", loss, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
            for metric, val in metric_vals.items():
                # self.log(f"{pred_type} train {metric}", val, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
                loss_weight = 1
                if 'trajectory' in pred_type:
                    loss_weight *= trajectory_weight
                if 'loop_closure' in pred_type:
                    loss_weight *= loop_closure_weight
                if 'random_walk' in pred_type:
                    loss_weight *= random_walk_weight
                if 'base_integration' in pred_type:
                    loss_weight *= self.base_integration_weight
                self.log(f"{pred_type} train {metric}", val*loss_weight, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
                
        self.log(f"total train loss", total_loss, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
        if jac_norm is not None:
            self.log(f"train jac norm", jac_norm, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
        self.log(f"train l1 norm", l1_loss, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)

        if self.teacher_forcing_annealing:
            self.log(f"alpha teacher forcing", self.alpha_teacher_forcing, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)

        # Log gradient statistics
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                self.log(f"gradients/{name}_norm", grad_norm, on_step=True, on_epoch=True, sync_dist=sync_dist)
                self.log(f"gradients/{name}_mean", param.grad.mean(), on_step=True, on_epoch=True, sync_dist=sync_dist)
                self.log(f"gradients/{name}_std", param.grad.std(), on_step=True, on_epoch=True, sync_dist=sync_dist)
                self.log(f"gradients/{name}_max", param.grad.max(), on_step=True, on_epoch=True, sync_dist=sync_dist)
                self.log(f"gradients/{name}_min", param.grad.min(), on_step=True, on_epoch=True, sync_dist=sync_dist)

        # Log Jacobian metrics
        if self.eq is not None and jacs_pred is not None:
            jacs_true = self.get_true_jacs(batch)
            if jacs_pred is None:
                jacs_pred = self.compute_jacobians(batch)
            jacs_pred_cpu = jacs_pred.detach().cpu().numpy()
            if isinstance(jacs_true, torch.Tensor):
                jacs_true_cpu = jacs_true.detach().cpu().numpy()
            else:
                jacs_true_cpu = jacs_true.copy()
                jacs_true = torch.from_numpy(jacs_true)
            jac_loss = mse(jacs_true_cpu, jacs_pred_cpu)
            self.log(f"train jac loss", jac_loss, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
            self.log(f"train jac r2_score", r2_score(jacs_true_cpu.flatten(), jacs_pred_cpu.flatten()), 
                    on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist, prog_bar=prog_bar)
            self.log(f"train jac nuclear norm loss", torch.norm(jacs_pred.to(batch.device) - jacs_true.to(batch.device), p='nuc', dim=(-2, -1)).mean(), on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
        
            if self.deriv_model is not None:
                if self.data_type == 'dysts':
                    deriv_true_cpu = self.eq.rhs(batch.cpu().numpy(), np.arange(batch.shape[-2])*self.dt)
                else:
                    deriv_true = self.eq.rhs(batch, batch.shape[-2]*self.dt)
                    deriv_true_cpu = deriv_true.cpu().numpy()
                deriv_pred = self.deriv_model(torch.arange(batch.shape[-2]).to(batch.device)*self.dt, batch)
                deriv_pred_cpu = deriv_pred.detach().cpu().numpy()
                deriv_loss = mse(deriv_true_cpu, deriv_pred_cpu)
                self.log(f"train deriv loss", deriv_loss, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
                self.log(f"train deriv r2_score", r2_score(deriv_true_cpu.flatten(), deriv_pred_cpu.flatten()), 
                        on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist, prog_bar=prog_bar)

    def log_validation_metrics(self, val_rets, batch, sync_dist=True, val_loop_closure=None):
        """Log validation metrics.

        Args:
            val_rets (dict): Validation results
            batch (torch.Tensor): Input batch
            sync_dist (bool): Whether to sync across distributed training
            val_loop_closure (Optional[dict]): Loop closure validation results
        """
        # Log basic metrics
        for pred_type, ret_dict in val_rets.items():
            loss, metric_vals = ret_dict['loss'], ret_dict['metric_vals']
            self.log(f"{pred_type} val_loss", loss, sync_dist=sync_dist, add_dataloader_idx=False)
            for metric, val in metric_vals.items():
                self.log(f"{pred_type} val {metric}", val, sync_dist=sync_dist, add_dataloader_idx=False)
        
        # log mean loss across all predictions
        mean_val_loss = torch.stack([val_rets[pred_type]['loss'] for pred_type in val_rets.keys()]).mean()
        self.log(f"mean val loss", mean_val_loss, sync_dist=sync_dist)

        if val_loop_closure is not None:
            self.log(f"val loop closure loss", val_loop_closure['loss'], sync_dist=sync_dist, add_dataloader_idx=False)
        # Log Jacobian metrics if equation is available
        if self.eq is not None:
            jacs_true = self.get_true_jacs(batch)
            if 'NeuralODE' in self.model.__class__.__name__:
                jacs_pred = torch.stack([self.get_pred_jacs(batch[[i]]) for i in range(batch.shape[0])])
            else:
                jacs_pred = self.get_pred_jacs(batch)
            
            if isinstance(jacs_true, torch.Tensor):
                jacs_true_cpu = jacs_true.detach().cpu().numpy()
            else:
                jacs_true_cpu = jacs_true
            
            jacs_pred_cpu = jacs_pred.detach().cpu().numpy()
            
            self.log(f"val jac loss", 
                    mse(jacs_true_cpu, jacs_pred_cpu), 
                    sync_dist=sync_dist, 
                    add_dataloader_idx=False)
            
            self.log(f"val jac r2_score", 
                    r2_score(jacs_true_cpu.flatten(), jacs_pred_cpu.flatten()), 
                    sync_dist=sync_dist, 
                    add_dataloader_idx=False)
            
            if self.deriv_model is not None:
                if self.data_type == 'dysts':
                    deriv_true_cpu = self.eq.rhs(batch.cpu().numpy(), np.arange(batch.shape[-2])*self.dt)
                else:
                    deriv_true = self.eq.rhs(batch, batch.shape[-2]*self.dt)
                    deriv_true_cpu = deriv_true.cpu().numpy()
                deriv_pred = self.deriv_model(torch.arange(batch.shape[-2]).to(batch.device)*self.dt, batch)

                
                deriv_pred_cpu = deriv_pred.detach().cpu().numpy()

                self.log(f"val deriv loss", 
                        mse(deriv_true_cpu, deriv_pred_cpu), 
                        sync_dist=sync_dist, 
                        add_dataloader_idx=False)
                self.log(f"val deriv r2_score", 
                        r2_score(deriv_true_cpu.flatten(), deriv_pred_cpu.flatten()), 
                        sync_dist=sync_dist, 
                        add_dataloader_idx=False)

    def calc_metrics(self, y_true, y_pred):
        """Calculate various metrics between true and predicted values.

        Args:
            y_true (torch.Tensor): True values
            y_pred (torch.Tensor): Predicted values

        Returns:
            dict: Dictionary containing computed metrics
        """
        metric_vals = dict() 
        for metric, metric_func in METRIC_DICT.items():
            metric_vals[metric] = metric_func(y_true, y_pred)
        
        return metric_vals

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            Union[torch.optim.Optimizer, dict]: Optimizer or dictionary with optimizer and scheduler
        """
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_kwargs)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
        elif self.optimizer == 'RAdam':
            optimizer = torch.optim.RAdam(self.parameters(), **self.optimizer_kwargs)
        elif self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)
        elif self.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), **self.optimizer_kwargs)
        elif self.optimizer == 'SSBroyden':
            optimizer = SelfScaledBroyden(self.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f'Optimizer {self.optimizer} not recognized')

        if self.use_scheduler:
            scheduler = TeacherForcingLRScheduler(
                optimizer,
                lit_model=self,
                min_lr=self.min_lr,
                k=self.k_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
                "gradient_clip_val": self.gradient_clip_val,
                "gradient_clip_algorithm": self.gradient_clip_algorithm,
            }
        else:
            return {
                "optimizer": optimizer,
                "gradient_clip_val": self.gradient_clip_val,
                "gradient_clip_algorithm": self.gradient_clip_algorithm,
            }

    def optimizer_step(self, *args, **kwargs):
        """Custom optimizer step with gradient clipping.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().optimizer_step(*args, **kwargs)
        # do something on_after_optimizer_step
    
    def generate(self, x, alpha=1.0):
        """Generate predictions using the model.

        Args:
            x (torch.Tensor): Input tensor
            alpha (float): Teacher forcing coefficient

        Returns:
            torch.Tensor: Generated predictions
        """
        return self.model.generate(x, alpha=alpha)

    # In your MLP or LightningModule:
    def get_main_params(self):
        """Get main model parameters, excluding Lipschitz parameters.

        Returns:
            List[torch.Tensor]: List of parameter tensors
        """
        # Exclude any parameter that is part of the Lipschitz update
        return [p for n, p in self.named_parameters() if "lipschitz_constant" not in n]

class Sin(nn.Module):
    """Sine activation function module."""

    def forward(self, x):
        """Apply sine activation function.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Sine of input
        """
        return torch.sin(x)

def get_activation_func(activation):
    """Get activation function by name.

    Args:
        activation (str): Name of activation function ('relu', 'tanh', 'sin', etc.)

    Returns:
        nn.Module: Activation function module

    Raises:
        ValueError: If activation function name is not recognized
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'sin':
        return Sin()
    else:
        raise ValueError(f'Activation function {activation} not recognized')

class PercentEarlyStopping(EarlyStopping):
    """Early stopping based on percentage improvement threshold.

    This early stopping callback monitors validation loss and stops training
    when the percentage improvement falls below a threshold for a specified
    number of epochs.

    Args:
        percent_thresh (float): Minimum percentage improvement required (default: 0.01)
        *args: Additional arguments for EarlyStopping
        **kwargs: Additional keyword arguments for EarlyStopping
    """
    def __init__(self, *args, **kwargs):
        # Extract percent_thresh before calling parent init
        self.percent_thresh = kwargs.pop('percent_thresh', 0.01)
        super().__init__(*args, **kwargs)
        self.prev_loss = None
        self.wait_count = 0

    def _evaluate_stopping_criteria(self, current):
        if self.prev_loss is None:
            self.prev_loss = current
            return False, None

        # Calculate percent improvement
        if self.prev_loss > current:
            percent_improvement = (self.prev_loss - current) / self.prev_loss
            if percent_improvement < self.percent_thresh:
                self.wait_count += 1
            else:
                self.wait_count = 0
        else:
            self.wait_count += 1

        self.prev_loss = current
        
        # Check if we've waited long enough
        if self.wait_count >= self.patience:
            self.wait_count = 0  # Reset for potential future use
            return True, None
            
        return False, None