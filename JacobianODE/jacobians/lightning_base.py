import gc
import math
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pathlib import Path
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm
from typing import Any, Callable, Optional, Union, Tuple

from .jacobianODE import JacobianODE, JacobianODEint
from .metrics import mase, mse, r2_score, smape, normalized_mse, GeneralizedNormalizedMSE
from .teacher_forcing import get_alpha_exact, get_alpha_explogapprox, get_alpha_lyap


# PyTorch 2.6+ changed the default of torch.load to weights_only=True, which
# rejects pickles containing numpy scalar types (e.g. np.float64) unless they
# are explicitly allowlisted. Stage A checkpoints saved by Lightning include
# the LightningModule state and (historically) the TeacherForcingLRScheduler
# state, both of which can contain numpy scalars. Allowlist the ones that
# arise in practice so Stage B can resume from those checkpoints. We control
# the checkpoints, so the security model that motivates weights_only=True
# does not apply here.
try:
    import numpy._core.multiarray  # type: ignore[import-not-found]
    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar,
        np.dtype,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
        np.dtypes.Int64DType,
        np.dtypes.Int32DType,
        np.ndarray,
    ])
except (ImportError, AttributeError):
    # Older numpy / older PyTorch — silently skip; weights_only loading
    # may still fail and require explicit weights_only=False at the call
    # site.
    pass


METRIC_DICT = {
    'mse': mse,
    'mase': mase,
    'r2_score': r2_score,
    'smape': smape,
}

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

    def state_dict(self):
        # Mirror the base class behaviour (which excludes ``optimizer``) and
        # additionally exclude ``lit_model``: the LightningModule reference
        # would otherwise be pickled into the scheduler state via the parent
        # class's ``{k: v for k, v in self.__dict__.items() if k != 'optimizer'}``
        # default, ballooning the checkpoint and embedding numpy scalars that
        # PyTorch 2.6+ ``weights_only=True`` loaders refuse. The scheduler
        # always re-acquires the LightningModule reference via the constructor
        # in ``configure_optimizers`` on resume, so it does not need to be
        # serialized.
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ("optimizer", "lit_model")
        }

    def load_state_dict(self, state_dict):
        # Defensive: if a legacy checkpoint contains ``lit_model``, drop it on
        # load so the in-memory reference set by ``__init__`` is not clobbered
        # by the (potentially stale and wrong-typed) saved value.
        clean = {k: v for k, v in state_dict.items() if k != "lit_model"}
        self.__dict__.update(clean)

def loop_closure(
        batch, 
        jac_func, 
        dt=1, 
        n_loops=None, 
        n_loop_pts=None, 
        loop_path='line',
        int_method='Trapezoid', 
        loop_closure_interp_pts=2, 
        mix_trajectories=True,
        alpha=1, 
        return_loop_pts=False
    ):
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

class LitBase(L.LightningModule):
    """Base Lightning module for training neural networks on dynamical systems.

    This class implements a PyTorch Lightning module for training neural networks
    to learn dynamical systems, with support for various training strategies including
    teacher forcing, noise injection, and multiple loss terms.

    Args:
        model (nn.Module): The neural network model to train
        dt (float): Time step size for integration (default: 1)
        eq (Optional[Any]): Optional equation object containing true dynamics
        direct (bool): Whether to use direct prediction or integration (default: True)
        save_dir (Optional[str]): Directory to save model checkpoints
        loss_func (str): Loss function type ('mse' or 'hal' for horizon-aware loss) (default: 'mse')
        alpha_hal (float): Alpha parameter for horizon-aware loss (default: 0.1)
        obs_noise_scale (float): Scale of observation noise to add during training (default: 0.0)
        log_interval (int): Interval for logging training metrics (default: 1)
        jac_loss_interval (int): Interval for computing Jacobian loss (default: 1)
        alpha_teacher_forcing (float): Initial teacher forcing coefficient (default: 1)
        teacher_forcing_annealing (bool): Whether to anneal teacher forcing (default: True)
        gamma_teacher_forcing (float): Decay rate for teacher forcing (default: 0.999)
        teacher_forcing_update_interval (int): Interval for updating teacher forcing (default: 1)
        teacher_forcing_steps (int): Number of steps to use teacher forcing (default: 1)
        min_alpha_teacher_forcing (float): Minimum teacher forcing coefficient (default: 0)
        alpha_validation (float): Teacher forcing coefficient for validation (default: 0)
        data_type (Optional[str]): Type of data being used (default: None)
        optimizer (str): Optimizer to use ('AdamW' or 'Adam') (default: 'AdamW')
        optimizer_kwargs (dict): Keyword arguments for optimizer (default: {'lr': 1e-4})
        gradient_clip_val (float): Value for gradient clipping (default: 1.0)
        gradient_clip_algorithm (str): Algorithm for gradient clipping (default: 'norm')
        jacobianODEint_kwargs (dict): Keyword arguments for JacobianODE integration (default: {})
        min_traj_init_steps (int): Minimum initial steps for trajectories (default: 2)
        max_traj_init_steps (Optional[int]): Maximum initial steps for trajectories (default: None)
        use_scheduler (bool): Whether to use learning rate scheduler (default: False).
            When True, always uses TeacherForcingLRScheduler (LR coupled to teacher-forcing α).
        min_lr (Optional[float]): Minimum learning rate for scheduler (default: None)
        k_scale (Optional[float]): Scaling factor for teacher_forcing scheduler (default: None)
        jac_penalty (float): Weight for Jacobian regularization (default: 0.0)
        jac_norm_ord (str): Order of norm for Jacobian regularization (default: 'fro')
        loop_closure_training (bool): Whether to use loop closure training (default: True)
        mix_trajectories (bool): Whether to mix trajectories during training (default: True)
        loop_closure_interp_pts (int): Number of interpolation points for loop closure (default: 20)
        n_loops (Optional[int]): Number of loops for loop closure (default: None)
        n_loop_pts (Optional[int]): Number of points per loop (default: None)
        loop_path (str): Path type for loop closure ('line' or 'spline') (default: 'line')
        loop_closure_weight (float): Weight for loop closure loss (default: 1.0)
        trajectory_training (bool): Whether to use trajectory training (default: True)
        use_base_deriv_pt (bool): Whether to use base derivative point (default: False)
        base_pt_init (Optional[torch.Tensor]): Initial base point (default: None)
        base_deriv_pt_init (Optional[torch.Tensor]): Initial base derivative point (default: None)
        n_delays (Optional[int]): Number of delays for embedding (default: None)
        obs_dim (Optional[int]): Dimension of observations (default: None)
        obs_only_loss (bool): Whether to use observation-only loss (default: False)
        early_stopping_patience (int): Patience for early stopping (default: 5)
        early_stopping_mode (str): Mode for early stopping ('min' or 'max') (default: 'min')
        percent_thresh (float): Threshold for percent improvement (default: 0.01)
    """
    def __init__(
                    self, 
                    model,
                    dt=1,
                    eq=None,
                    direct=True,
                    save_dir=None, 
                    loss_func='mse',
                    alpha_hal=0.1,
                    obs_noise_scale=0.0,
                    log_interval=1,
                    jac_loss_interval=1,
                    alpha_teacher_forcing=1,
                    teacher_forcing_annealing=True,
                    gamma_teacher_forcing=0.999,
                    teacher_forcing_update_interval=1,
                    teacher_forcing_steps=1,
                    min_alpha_teacher_forcing=0,
                    alpha_validation=0,
                    data_type=None,
                    optimizer='AdamW',
                    optimizer_kwargs={'lr': 1e-4},
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm='norm',
                    jacobianODEint_kwargs={},
                    use_scheduler=False,
                    min_lr=None,
                    k_scale=None,
                    jac_penalty=0.0,
                    jac_norm_ord='fro',
                    loop_closure_training=True,
                    mix_trajectories=True,
                    loop_closure_interp_pts=20,
                    n_loops=None,
                    n_loop_pts=None,
                    loop_path='line',
                    loop_closure_weight=1.0,
                    trajectory_training=True,
                    use_base_deriv_pt=False,
                    base_pt_init=None,
                    base_deriv_pt_init=None,
                    n_delays=None,
                    obs_dim=None,
                    obs_only_loss=False,
                    early_stopping_patience=5,
                    early_stopping_mode='min',
                    percent_thresh=0.01,
                    mu=0,
                    sigma=1,
                    noise_scale_factor=1.0,
                    generalized_variance=None,
                    gen_variance_mode='fixed',
                    **kwargs
                ):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.dt = dt
        self.eq = eq
        self.direct = direct

        self.save_dir = save_dir
        self.alpha_hal = alpha_hal
        if loss_func == 'mse':
            self.criterion = nn.MSELoss()
            self.latent_criterion = self.criterion
        elif loss_func == 'normalized_mse':
            self.criterion = normalized_mse
            self.latent_criterion = self.criterion
        elif loss_func == 'generalized_normalized_mse':
            if generalized_variance is None:
                raise ValueError(
                    "loss_func='generalized_normalized_mse' requires "
                    "generalized_variance to be precomputed and passed in."
                )
            if gen_variance_mode not in ('fixed', 'adaptive_latent'):
                raise ValueError(
                    f"gen_variance_mode must be 'fixed' or 'adaptive_latent', "
                    f"got '{gen_variance_mode}'"
                )
            # Obs-space criterion (used for decoded prediction + reconstruction):
            # denominator = det(Cov(x_recent))^(1/n_recent_dims). Fixed across training.
            self.criterion = GeneralizedNormalizedMSE(generalized_variance)
            if gen_variance_mode == 'fixed':
                # Single-scalar mode: same denom for latent-space loss too.
                self.latent_criterion = self.criterion
            else:
                # Adaptive mode: separate criterion for z_dyn loss, with a
                # denominator that gets updated at the start of each training
                # epoch (see LitLatentJacobianODE.on_train_epoch_start).
                # Initialized to the obs-space denom as a safe default.
                self.latent_criterion = GeneralizedNormalizedMSE(generalized_variance)
        else:
            raise ValueError(f"Loss function {loss_func} not implemented")
        self._gen_variance_mode = gen_variance_mode

        self.obs_noise_scale = obs_noise_scale
        self.noise_scale_factor = noise_scale_factor
        self.log_interval = log_interval
        self.jac_loss_interval = jac_loss_interval

        self.alpha_teacher_forcing = alpha_teacher_forcing
        self.teacher_forcing_annealing = teacher_forcing_annealing
        self.gamma_teacher_forcing = gamma_teacher_forcing
        self.teacher_forcing_steps = teacher_forcing_steps
        self.teacher_forcing_update_interval = teacher_forcing_update_interval

        self.min_alpha_teacher_forcing = min_alpha_teacher_forcing
        self.alpha_validation = alpha_validation

        self.data_type = data_type

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
        self.n_loops = n_loops
        self.n_loop_pts = n_loop_pts
        self.loop_path = loop_path
        self.loop_closure_weight = loop_closure_weight
        self.mix_trajectories = mix_trajectories
        self.trajectory_training = trajectory_training

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

    # ------------------------------------------------------------------
    # Checkpoint state preservation
    # ------------------------------------------------------------------
    # Lightning's standard ckpt saves model state_dict, optimizer state,
    # and LR scheduler state — but NOT plain Python attributes on the
    # LightningModule. ``alpha_teacher_forcing`` is updated each batch
    # via ``update_alpha_teacher_forcing`` (a teacher-forcing anneal
    # schedule) and lives as a plain float on ``self``. Without these
    # hooks, a ckpt-resume restarts alpha at the YAML init value
    # (typically 1.0) regardless of how far it had annealed, which
    # introduces a discontinuity in training dynamics.
    #
    # Stored under a dedicated ``lit_runtime_state`` key so the diff
    # to existing ckpts is additive: older ckpts (no key) load with
    # whatever alpha __init__ set; newer ckpts restore the saved value.
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["lit_runtime_state"] = {
            "alpha_teacher_forcing": float(self.alpha_teacher_forcing),
            "teacher_forcing_steps": int(self.teacher_forcing_steps or 0),
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        super().on_load_checkpoint(checkpoint)
        state = checkpoint.get("lit_runtime_state") or {}
        if "alpha_teacher_forcing" in state:
            self.alpha_teacher_forcing = float(state["alpha_teacher_forcing"])
        if "teacher_forcing_steps" in state:
            self.teacher_forcing_steps = int(state["teacher_forcing_steps"])

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

    def trajectory_model_step(
                    self, 
                    batch, 
                    batch_idx=0, 
                    dataloader_idx=0, 
                    all_metrics=False, 
                    direct=None, 
                    obs_noise_scale=None, 
                    alpha_teacher_forcing=None,
                    teacher_forcing_steps=None,
                    jacobianODEint_kwargs=None,
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
            alpha_teacher_forcing (Optional[float]): Teacher forcing coefficient
            teacher_forcing_steps (Optional[int]): Number of teacher forcing steps
            jacobianODEint_kwargs (Optional[dict]): Integration parameters
            criterion (Optional[Callable]): Custom loss function
            verbose (bool): Whether to print progress

        Returns:
            dict: Dictionary containing loss values and metrics
        """
        if direct is None:
            direct = self.direct
        if obs_noise_scale is None:
            obs_noise_scale = self.obs_noise_scale
        if alpha_teacher_forcing is None:
            alpha_teacher_forcing = self.alpha_teacher_forcing
        if teacher_forcing_steps is None:
            teacher_forcing_steps = self.teacher_forcing_steps
        if criterion is None:
            criterion = self.criterion
        if jacobianODEint_kwargs is None:
            jacobianODEint_kwargs = self.jacobianODEint_kwargs

        
        alpha = alpha_teacher_forcing

        batch = batch.type(self.dtype)
        label = batch.detach().clone() # Detach to prevent gradient flow through label
        scaled_noise = obs_noise_scale * self.noise_scale_factor
        batch = batch + (torch.randn(*batch.shape)*scaled_noise).type(batch.dtype).to(batch.device)

        # sample traj_init_steps as an integer between min_traj_init_steps and max_traj_init_steps inclusive
        if 'traj_init_steps' not in jacobianODEint_kwargs:
            jacobianODEint_kwargs['traj_init_steps'] = 2
        
        if direct:
            if self.use_base_deriv_pt:
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
            # Store raw MAE components so callers can aggregate correctly
            # (ratio-of-means instead of mean-of-ratios).
            metric_vals['model_mae'] = torch.mean(torch.abs(label_cropped - outputs_cropped))
            if label_cropped.dim() == 3:
                metric_vals['persistence_mae'] = torch.mean(
                    torch.abs(label_cropped[:, 1:] - label_cropped[:, :-1]))
            else:
                metric_vals['persistence_mae'] = torch.mean(
                    torch.abs(label_cropped[1:] - label_cropped[:-1]))
            metric_vals['r2_score'] = r2_score(label_cropped, outputs_cropped)

        return {'loss': loss, 'metric_vals': metric_vals, 'outputs': outputs}
    
    def loop_closure_model_step(
            self,
            batch,
            batch_idx=0,
            dataloader_idx=0,
            mix_trajectories=None,
            n_loops=None,
            n_loop_pts=None,
            loop_path=None,
            loop_closure_interp_pts=None,
        ):
        """Perform a single training step for loop closure.

        Args:
            batch (torch.Tensor): Input batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of current dataloader
            mix_trajectories (bool): Whether to mix trajectories
            n_loops (Optional[int]): Number of loops
            n_loop_pts (Optional[int]): Points per loop
            loop_path (str): Path type for loop closure
            loop_closure_interp_pts (Optional[int]): Interpolation points

        Returns:
            dict: Dictionary containing loss values and metrics
        """
        if mix_trajectories is None:
            mix_trajectories = self.mix_trajectories
        if n_loops is None:
            n_loops = self.n_loops
        if n_loop_pts is None:
            n_loop_pts = self.n_loop_pts
        if loop_path is None:
            loop_path = self.loop_path
        if loop_closure_interp_pts is None:
            loop_closure_interp_pts = self.loop_closure_interp_pts

        loop_int = loop_closure(batch, self.compute_jacobians, dt=self.dt, n_loops=n_loops, n_loop_pts=n_loop_pts, loop_path=loop_path, loop_closure_interp_pts=loop_closure_interp_pts, mix_trajectories=mix_trajectories, int_method='Trapezoid')


        loop_zeros = torch.zeros_like(loop_int)

        # loop_loss = self.criterion(loop_zeros, loop_int)
        loop_loss = (loop_int**2).mean()
        # loop_loss = torch.clamp(torch.linalg.norm(loop_int, dim=-1) - err_bound, 0, None).mean()

        metric_vals = dict(
            mse=mse(loop_zeros, loop_int),
            # r2_score=r2_score(loop_zeros.flatten(), loop_int.flatten())
        )

        return {'loss': loop_loss, 'metric_vals': metric_vals, 'outputs': loop_int}

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
        if 'NeuralODE' in self.model.__class__.__name__ and not self.teacher_forcing_annealing and self.jac_penalty == 0:
            jacs_pred = None
        else:
            jacs_pred = self.get_pred_jacs(batch)
        if jacs_pred is not None:
            jac_norm = torch.linalg.norm(jacs_pred, dim=(-2, -1), ord=self.jac_norm_ord).mean()
            encoder_warmup_epochs = getattr(self, 'encoder_warmup_epochs', 0)
            if self.current_epoch >= encoder_warmup_epochs:
                self.update_alpha_teacher_forcing(jacs_pred.detach(), batch_idx)
        else:
            jac_norm = None
        
        train_rets = {}
        if self.trajectory_training:
            train_rets['trajectory'] = self.trajectory_model_step(batch, batch_idx, dataloader_idx)
        if self.loop_closure_training:
            if ('NeuralODE' in self.model.__class__.__name__ and self.loop_closure_weight > 0) or ('NeuralODE' not in self.model.__class__.__name__):
                train_rets['loop_closure'] = self.loop_closure_model_step(batch, batch_idx, dataloader_idx)

        total_loss = 0
        trajectory_weight = 1
        loop_closure_weight = self.loop_closure_weight
       
        for pred_type, ret_dict in train_rets.items():
            if torch.isnan(ret_dict['loss']):
                # Warn if loss is nan
                print(f"Warning: Loss is nan for pred type {pred_type} on epoch {self.current_epoch} batch {batch_idx}")
            loss_val = ret_dict['loss'] if not torch.isnan(ret_dict['loss']) else 0
            loss_weight = 1
            if 'trajectory' in pred_type:
                loss_weight *= trajectory_weight
            if 'loop_closure' in pred_type:
                # loss_weight *= loop_closure_weight * (1/self.n_loop_pts)
                loss_weight *= loop_closure_weight
            
            total_loss += loss_weight*loss_val
        if jac_norm is not None:
            total_loss += self.jac_penalty*jac_norm
        l1_loss = torch.sum(torch.abs(torch.cat([p.view(-1) for p in self.get_main_params()], dim=0)))
        
        if ((batch_idx + 1) % self.log_interval) == 0:
            self.log_training_metrics(
                train_rets=train_rets,
                total_loss=total_loss,
                jac_norm=jac_norm,
                l1_loss=l1_loss,
                batch=batch,
                jacs_pred=jacs_pred,
                batch_idx=batch_idx,
                on_step=False,
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
        
        val_loop_closure = self.loop_closure_model_step(batch, batch_idx, dataloader_idx)

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
        model_step_kwargs = {
            'alpha_teacher_forcing': self.alpha_validation, 
            'obs_noise_scale': 0,
        }

        val_rets = {}
        val_rets['trajectory'] = self.trajectory_model_step(batch, batch_idx, dataloader_idx, **model_step_kwargs)

        if 'NeuralODE' not in self.model.__class__.__name__:
            val_loop_closure = self.loop_closure_model_step(batch, batch_idx, dataloader_idx)
        else:
            val_loop_closure = None

        if log_metrics:
            self.log_validation_metrics(
                val_rets=val_rets,
                batch=batch,
                sync_dist=True,
                val_loop_closure=val_loop_closure,
            )

        total_loss = sum(val_rets[pred_type]['loss'] for pred_type in val_rets.keys())
        
        # Track the same metric that PercentEarlyStopping monitors ("mean val loss")
        mean_val_loss = torch.stack(
            [val_rets[pt]['loss'] for pt in val_rets]
        ).mean()
        if not hasattr(self, 'current_epoch_val_losses'):
            self.current_epoch_val_losses = []
        self.current_epoch_val_losses.append(mean_val_loss.item())

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
                # Use last non-NaN losses for comparison (NaN comparisons always return False)
                prev_loss = next(
                    (x for x in reversed(self.validation_losses[:-1]) if not math.isnan(x)),
                    None,
                )
                curr_loss = self.validation_losses[-1]
                if (
                    prev_loss is not None
                    and not math.isnan(curr_loss)
                    and prev_loss > curr_loss
                ):
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

        # One-step MASE (ratio-of-means across batches for C1 diagnostic)
        if hasattr(self, '_val_one_step_model_maes') and self._val_one_step_model_maes:
            avg_model_mae = sum(self._val_one_step_model_maes) / len(self._val_one_step_model_maes)
            avg_persist_mae = sum(self._val_one_step_persistence_maes) / len(self._val_one_step_persistence_maes)
            one_step_mase = avg_model_mae / max(avg_persist_mae, 1e-8)
            self.log("val/one_step_mase", one_step_mase, sync_dist=True)
            self._val_one_step_model_maes = []
            self._val_one_step_persistence_maes = []

        # Fast eigenvalue fraction (C3 diagnostic)
        if hasattr(self, '_val_eig_too_fast') and self._val_eig_total:
            total_too_fast = sum(self._val_eig_too_fast)
            total_eigs = sum(self._val_eig_total)
            frac = total_too_fast / max(total_eigs, 1)
            self.log("val/fast_eigenvalue_fraction", frac, sync_dist=True)
            self._val_eig_too_fast = []
            self._val_eig_total = []

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
        # trajectory_weight = 1
        # loop_closure_weight = self.loop_closure_weight
        
        for pred_type, ret_dict in train_rets.items():
            loss, metric_vals = ret_dict['loss'], ret_dict['metric_vals']
            self.log(f"{pred_type} train_loss", loss, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
            for metric, val in metric_vals.items():
                if pred_type != 'trajectory':
                    continue
                # self.log(f"{pred_type} train {metric}", val, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
                loss_weight = 1
                # if 'trajectory' in pred_type:
                #     loss_weight *= trajectory_weight
                # if 'loop_closure' in pred_type:
                #     loss_weight *= loop_closure_weight
                self.log(f"{pred_type} train {metric}", val*loss_weight, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
                
        self.log(f"total train loss", total_loss, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
        if jac_norm is not None:
            self.log(f"train jac norm", jac_norm, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)
        self.log(f"train l1 norm", l1_loss, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)

        if self.teacher_forcing_annealing:
            self.log(f"alpha teacher forcing", self.alpha_teacher_forcing, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)

        # Log Jacobian metrics
        if self.eq is not None and jacs_pred is not None:
            jacs_true = self.get_true_jacs(batch)
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
            # self.log(f"train jac nuclear norm loss", torch.norm(jacs_pred.to(batch.device) - jacs_true.to(batch.device), p='nuc', dim=(-2, -1)).mean(), on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)

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
                if pred_type != 'trajectory':
                    continue
                self.log(f"{pred_type} val {metric}", val, sync_dist=sync_dist, add_dataloader_idx=False)
        
        # log mean loss across all predictions
        mean_val_loss = torch.stack([val_rets[pred_type]['loss'] for pred_type in val_rets.keys()]).mean()
        self.log(f"mean val loss", mean_val_loss, sync_dist=sync_dist)

        if val_loop_closure is not None:
            self.log(f"val loop closure loss", val_loop_closure['metric_vals']['mse'], sync_dist=sync_dist, add_dataloader_idx=False)
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
        else:
            raise ValueError(f'Optimizer {self.optimizer} not recognized')

        if self.use_scheduler:
            scheduler = TeacherForcingLRScheduler(
                optimizer,
                lit_model=self,
                min_lr=self.min_lr,
                k=self.k_scale,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return {
                "optimizer": optimizer,
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
        min_epochs (int): Minimum number of epochs to complete before early stopping
            can trigger. Training always runs at least this many epochs. (default: 0)
        *args: Additional arguments for EarlyStopping
        **kwargs: Additional keyword arguments for EarlyStopping
    """
    def __init__(self, *args, **kwargs):
        # Extract percent_thresh and min_epochs before calling parent init
        self.percent_thresh = kwargs.pop('percent_thresh', 0.01)
        self.min_epochs = kwargs.pop('min_epochs', 0)
        super().__init__(*args, **kwargs)
        self.prev_loss = None
        self.wait_count = 0

    def _run_early_stopping_check(self, trainer):
        """Skip early stopping check until min_epochs have completed."""
        if trainer.current_epoch < self.min_epochs:
            return
        super()._run_early_stopping_check(trainer)

    def _evaluate_stopping_criteria(self, current):
        # Convert to Python float for NaN/inf checks (current may be a 0-dim tensor)
        current_val = current.item() if hasattr(current, 'item') else current
        try:
            current_is_nan = not math.isfinite(float(current_val))
        except (TypeError, ValueError):
            current_is_nan = True

        # Ignore NaN/Inf: don't update baseline, count as no improvement
        if current_is_nan:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.wait_count = 0
                return True, None
            return False, None

        if self.prev_loss is None:
            self.prev_loss = current_val
            return False, None

        # prev_loss may be stale NaN/Inf from a previous epoch; reset to current valid value
        try:
            prev_is_nan = not math.isfinite(float(self.prev_loss))
        except (TypeError, ValueError):
            prev_is_nan = True
        if prev_is_nan:
            self.prev_loss = current_val
            self.wait_count = 0  # New baseline, don't count against patience
            return False, None

        # Calculate percent improvement (both values are valid)
        if self.prev_loss > current_val:
            percent_improvement = (self.prev_loss - current_val) / self.prev_loss
            if percent_improvement < self.percent_thresh:
                self.wait_count += 1
            else:
                self.wait_count = 0
        else:
            self.wait_count += 1

        self.prev_loss = current_val

        # Check if we've waited long enough
        if self.wait_count >= self.patience:
            self.wait_count = 0  # Reset for potential future use
            return True, None

        return False, None


class ShadowPercentEarlyStoppingCheckpoint(L.Callback):
    """Shadow checkpoint that freezes at the state a *hypothetical*
    earlier-triggering :class:`PercentEarlyStopping` would have stopped on.

    Runs in parallel with the real early-stopping + checkpointing. Does NOT
    stop training itself — the real ES/patience config still controls when
    training ends. This callback simply tracks a second, parallel
    PercentEarlyStopping counter (with a typically smaller ``shadow_patience``)
    and writes a checkpoint whenever the monitored metric hits a new best,
    freezing the file once the shadow counter trips.

    At the end of training you have two checkpoints per run: the "real" one
    saved by the trainer's ModelCheckpoint, and the "shadow" one saved here
    — exactly the state the run would have been in if it had early-stopped
    under the shadow patience.

    Semantics mirror :class:`PercentEarlyStopping` exactly (wait_count
    compares each epoch's val to the *previous* epoch's val, not best-so-far).
    Independently, a running ``best_loss`` is tracked so the saved checkpoint
    is always the best observed val-loss up to (and including) the trigger
    epoch — not just the last-improving-delta epoch.

    Parameters
    ----------
    monitor : str
        Metric to track (same as ES monitor).
    shadow_patience : int
        How many consecutive sub-threshold updates to tolerate before freezing.
    percent_thresh : float
        Same semantics as :class:`PercentEarlyStopping.percent_thresh`.
    min_epochs : int
        Don't trigger before this many epochs.
    filename : str
        Base filename (no extension) for the frozen checkpoint.
    """

    def __init__(
        self,
        monitor: str = "trajectory val_loss",
        shadow_patience: int = 2,
        percent_thresh: float = 0.01,
        min_epochs: int = 0,
        filename: str = "es_shadow-best",
    ):
        super().__init__()
        self._monitor = monitor
        self._shadow_patience = int(shadow_patience)
        self._percent_thresh = float(percent_thresh)
        self._min_epochs = int(min_epochs)
        self._filename = filename if filename.endswith(".ckpt") else filename + ".ckpt"

        # PercentEarlyStopping-equivalent state
        self._prev_loss: float | None = None
        self._wait_count = 0
        # Running best + triggered flag
        self._best_loss: float | None = None
        self._best_epoch: int | None = None
        self._triggered = False
        self._dirpath: Path | None = None

    # ---- Lightning lifecycle hooks ---------------------------------------

    def _resolve_dirpath(self, trainer) -> Path | None:
        """Inherit the dirpath from the trainer's primary ModelCheckpoint so
        shadow checkpoints land in the same per-run checkpoint directory."""
        if self._dirpath is not None:
            return self._dirpath
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and getattr(cb, "dirpath", None):
                self._dirpath = Path(cb.dirpath)
                return self._dirpath
        return None

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._triggered:
            return
        dirpath = self._resolve_dirpath(trainer)
        if dirpath is None:
            return

        val = trainer.callback_metrics.get(self._monitor)
        if val is None:
            return
        try:
            current = float(val.item() if hasattr(val, "item") else val)
        except (TypeError, ValueError):
            return

        # NaN/Inf: treat as no improvement (parallels PercentEarlyStopping).
        if not math.isfinite(current):
            self._wait_count += 1
            self._maybe_trigger(trainer.current_epoch)
            return

        # Running-best tracking: always save on strict improvement, so the
        # shadow file is the best val-loss checkpoint up to the current epoch.
        is_new_best = self._best_loss is None or current < self._best_loss
        if is_new_best:
            self._best_loss = current
            self._best_epoch = trainer.current_epoch
            filepath = dirpath / self._filename
            dirpath.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(filepath))

        # PercentEarlyStopping-equivalent wait_count update.
        if self._prev_loss is None:
            self._prev_loss = current
        else:
            prev = self._prev_loss
            if not math.isfinite(prev):
                self._prev_loss = current
                self._wait_count = 0
            elif prev > current:
                pct = (prev - current) / prev
                if pct < self._percent_thresh:
                    self._wait_count += 1
                else:
                    self._wait_count = 0
            else:
                self._wait_count += 1
            self._prev_loss = current

        self._maybe_trigger(trainer.current_epoch)

    def _maybe_trigger(self, epoch: int):
        if epoch < self._min_epochs:
            return
        if self._wait_count >= self._shadow_patience and not self._triggered:
            self._triggered = True

    # ---- State dict for resume safety (preempt-safe recovery) ------------

    def state_dict(self):
        return {
            "prev_loss": self._prev_loss,
            "wait_count": self._wait_count,
            "best_loss": self._best_loss,
            "best_epoch": self._best_epoch,
            "triggered": self._triggered,
            "dirpath": str(self._dirpath) if self._dirpath is not None else None,
        }

    def load_state_dict(self, state_dict):
        self._prev_loss = state_dict.get("prev_loss")
        self._wait_count = int(state_dict.get("wait_count", 0))
        self._best_loss = state_dict.get("best_loss")
        self._best_epoch = state_dict.get("best_epoch")
        self._triggered = bool(state_dict.get("triggered", False))
        dp = state_dict.get("dirpath")
        self._dirpath = Path(dp) if dp else None


class _OptunaTrialMixin:
    """Shared logic for Optuna callbacks that need to find their own trial in the DB.

    Since submitit_slurm workers don't have access to the Optuna Trial object,
    these callbacks identify their trial via a unique worker ID stored as a
    user attribute on first contact.

    **Important**: ``study.trials`` returns ``FrozenTrial`` snapshots whose
    ``set_user_attr`` only modifies an in-memory dict — it does NOT persist to
    the database.  All writes must go through ``study._storage`` instead.
    """

    WORKER_ID_KEY = "worker_id"

    def _get_study(self):
        """Load the Optuna study from storage (read/write)."""
        import optuna
        return optuna.load_study(
            study_name=self.study_name, storage=self.storage
        )

    @staticmethod
    def _set_trial_attr(study, trial, key, value):
        """Persist a user attribute to the database (not just in-memory)."""
        study._storage.set_trial_user_attr(trial._trial_id, key, value)

    def _get_worker_id(self):
        """Return a unique ID for this worker process."""
        import os
        return f"{os.getpid()}_{id(self)}"

    def _find_own_trial(self, study):
        """Find this worker's trial by worker_id, or claim an unclaimed RUNNING trial."""
        import optuna
        worker_id = self._get_worker_id()

        # First, look for a trial we've already claimed
        for trial in reversed(study.trials):
            if trial.user_attrs.get(self.WORKER_ID_KEY) == worker_id:
                return trial

        # Claim the most recent unclaimed RUNNING trial
        for trial in reversed(study.trials):
            if (
                trial.state == optuna.trial.TrialState.RUNNING
                and self.WORKER_ID_KEY not in trial.user_attrs
            ):
                self._set_trial_attr(study, trial, self.WORKER_ID_KEY, worker_id)
                return trial
        return None


class OptunaProgressCallback(L.Callback, _OptunaTrialMixin):
    """Write the best-so-far validation loss to the Optuna DB every validation epoch.

    If the SLURM job times out before ``run_jacobians`` returns, the sweeper
    coordinator marks the trial as FAIL and the return value is lost.  This
    callback ensures the DB always contains the most recent best loss as a
    user attribute (``best_so_far``), so timed-out trials can still be analyzed
    and their results recovered.

    Args:
        monitor: Metric name to track from ``trainer.callback_metrics``.
        study_name: Optuna study name (must match the coordinator).
        storage: Optuna storage URL (e.g. ``sqlite:///path.db``).
    """

    BEST_ATTR = "best_so_far"
    EPOCH_ATTR = "best_so_far_epoch"

    def __init__(self, monitor: str, study_name: str, storage: str):
        super().__init__()
        self.monitor = monitor
        self.study_name = study_name
        self.storage = storage
        self._best = float("inf")

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current_val = float(current)

        if current_val < self._best:
            self._best = current_val

        try:
            study = self._get_study()
            trial = self._find_own_trial(study)
            if trial is not None:
                self._set_trial_attr(study, trial, self.BEST_ATTR, self._best)
                self._set_trial_attr(study, trial, self.EPOCH_ATTR, trainer.current_epoch)
        except Exception as e:
            pl_module.print(f"[OptunaProgress] Could not write to study DB: {e}")


class OptunaConstrainedProgressCallback(L.Callback, _OptunaTrialMixin):
    """Write feasibility-aware best-so-far loss to the Optuna DB every validation epoch.

    Extends the logic of :class:`OptunaProgressCallback` with a physics constraint:
    a feasible result (constraint metric < threshold) always beats an infeasible
    one regardless of loss.  Among results with the same feasibility, lower loss wins.

    This enables Optuna's ``TPESampler(constraints_func=...)`` to steer away from
    hyperparameter regions that produce low loss but violate physics criteria
    (e.g. loop closure loss too high).

    Args:
        monitor: Primary objective metric (e.g. ``"trajectory val_loss"``).
        constraint_metric: Metric to check feasibility against
            (e.g. ``"val/loop_closure_loss"``).
        constraint_threshold: Maximum allowed value for the constraint metric.
            Values **at or below** this threshold are considered feasible.
        study_name: Optuna study name (must match the coordinator).
        storage: Optuna storage URL (e.g. ``sqlite:///path.db``).
    """

    BEST_ATTR = "best_so_far"
    EPOCH_ATTR = "best_so_far_epoch"
    FEASIBLE_ATTR = "best_so_far_feasible"
    CONSTRAINT_ATTR = "best_so_far_constraint"

    def __init__(
        self,
        monitor: str,
        constraint_metric: str,
        constraint_threshold: float,
        study_name: str,
        storage: str,
    ):
        super().__init__()
        self.monitor = monitor
        self.constraint_metric = constraint_metric
        self.constraint_threshold = constraint_threshold
        self.study_name = study_name
        self.storage = storage
        self._best = float("inf")
        self._best_feasible = False

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current_val = float(current)

        constraint_val = trainer.callback_metrics.get(self.constraint_metric)
        current_feasible = (
            constraint_val is not None
            and float(constraint_val) <= self.constraint_threshold
        )

        # Update best: feasible always beats infeasible; within same
        # feasibility class, lower loss wins.
        update = False
        if current_feasible and not self._best_feasible:
            # First feasible result replaces any infeasible best
            update = True
        elif current_feasible == self._best_feasible and current_val < self._best:
            # Same feasibility class — lower loss wins
            update = True

        if update:
            self._best = current_val
            self._best_feasible = current_feasible

        # Always write the real loss — the coordinator uses constraints_func
        # for feasibility separation, so it needs the actual value to rank
        # infeasible trials by quality too.

        try:
            study = self._get_study()
            trial = self._find_own_trial(study)
            if trial is not None:
                self._set_trial_attr(study, trial, self.BEST_ATTR, self._best)
                self._set_trial_attr(study, trial, self.EPOCH_ATTR, trainer.current_epoch)
                self._set_trial_attr(study, trial, self.FEASIBLE_ATTR, self._best_feasible)
                # Store the raw constraint value for analysis
                if constraint_val is not None:
                    self._set_trial_attr(study, trial, self.CONSTRAINT_ATTR, float(constraint_val))
        except Exception as e:
            pl_module.print(f"[OptunaConstrainedProgress] Could not write to study DB: {e}")


class OptunaPruneCallback(L.Callback, _OptunaTrialMixin):
    """Prune Optuna trials early by comparing against other trials at the same epoch.

    At ``prune_epoch``, this callback:
    1. Records the current trial's monitored metric at this epoch into the Optuna
       study as a user attribute (``prune_epoch_loss``).
    2. Reads the ``prune_epoch_loss`` values from all other trials that have
       already passed this epoch.
    3. If the current value is worse than the configured quantile of those
       values, sets ``trainer.should_stop = True``.

    Optionally, a physics constraint can be checked at prune epoch: if
    ``constraint_metric`` exceeds ``constraint_threshold``, the trial is
    stopped immediately (no need to wait for enough reference trials).

    This works with the ``submitit_slurm`` launcher where the Optuna ``trial``
    object is not available in the worker process — it reads and writes the
    shared SQLite study DB directly.

    Args:
        prune_epoch: Epoch at which to evaluate pruning (0-indexed).
        monitor: Metric name to read from ``trainer.callback_metrics``.
        study_name: Optuna study name (must match the coordinator).
        storage: Optuna storage URL (e.g. ``sqlite:///path.db``).
        min_completed: Minimum number of trials with ``prune_epoch_loss``
            recorded before pruning is considered.  Defaults to 5.
        quantile: Prune if the metric is above this quantile of other trials'
            ``prune_epoch_loss`` values.  Defaults to 0.5 (median).
        constraint_metric: Optional metric name for physics constraint check.
            If provided, trials that exceed ``constraint_threshold`` at
            ``prune_epoch`` are stopped immediately.
        constraint_threshold: Maximum allowed value for ``constraint_metric``.
    """

    # Key used to store the intermediate loss in each trial's user_attrs
    ATTR_KEY = "prune_epoch_loss"

    def __init__(
        self,
        prune_epoch: int,
        monitor: str,
        study_name: str,
        storage: str,
        min_completed: int = 5,
        quantile: float = 0.5,
        constraint_metric: str = None,
        constraint_threshold: float = None,
    ):
        super().__init__()
        self.prune_epoch = prune_epoch
        self.monitor = monitor
        self.study_name = study_name
        self.storage = storage
        self.min_completed = min_completed
        self.quantile = quantile
        self.constraint_metric = constraint_metric
        self.constraint_threshold = constraint_threshold
        self._recorded = False

    def _get_reference_losses(self, study):
        """Collect prune_epoch_loss values from all trials that have recorded one."""
        return [
            t.user_attrs[self.ATTR_KEY]
            for t in study.trials
            if self.ATTR_KEY in t.user_attrs
        ]

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch != self.prune_epoch:
            return
        if self._recorded:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current_val = float(current)

        # --- Constraint check (independent of reference trials) ---
        if self.constraint_metric and self.constraint_threshold is not None:
            constraint_val = trainer.callback_metrics.get(self.constraint_metric)
            if constraint_val is not None and float(constraint_val) > self.constraint_threshold:
                trainer.should_stop = True
                pl_module.print(
                    f"[OptunaPrune] epoch {self.prune_epoch}: "
                    f"constraint {self.constraint_metric}={float(constraint_val):.6f} > "
                    f"threshold {self.constraint_threshold:.6f} — stopping (infeasible)."
                )
                # Still record for other trials' reference, then return
                self._record_to_db(current_val)
                return

        # --- Standard quantile-based pruning ---
        try:
            study = self._get_study()

            # Always record this trial's prune-epoch loss
            trial = self._find_own_trial(study)
            if trial is not None:
                self._set_trial_attr(study, trial, self.ATTR_KEY, current_val)
                self._set_trial_attr(study, trial, "prune_epoch", self.prune_epoch)
            self._recorded = True

            # Reload to see freshly written data + other trials
            study = self._get_study()
            reference = self._get_reference_losses(study)
        except Exception as e:
            pl_module.print(f"[OptunaPrune] Could not access study DB: {e}")
            return

        # Exclude our own value from the reference distribution
        # (it was just added; we want to compare against *other* trials)
        other_losses = [v for v in reference if v != current_val]
        # If all values are the same as ours, fall back to full list
        if not other_losses:
            other_losses = reference

        if len(other_losses) < self.min_completed:
            pl_module.print(
                f"[OptunaPrune] epoch {self.prune_epoch}: "
                f"{self.monitor}={current_val:.6f}, "
                f"only {len(other_losses)} reference trials "
                f"(need {self.min_completed}) — not pruning."
            )
            return

        threshold = float(np.quantile(other_losses, self.quantile))
        if current_val > threshold:
            trainer.should_stop = True
            pl_module.print(
                f"[OptunaPrune] epoch {self.prune_epoch}: "
                f"{self.monitor}={current_val:.6f} > "
                f"quantile({self.quantile})={threshold:.6f} of "
                f"{len(other_losses)} trials at same epoch — stopping early."
            )
        else:
            pl_module.print(
                f"[OptunaPrune] epoch {self.prune_epoch}: "
                f"{self.monitor}={current_val:.6f} <= "
                f"quantile({self.quantile})={threshold:.6f} of "
                f"{len(other_losses)} trials — continuing."
            )

    def _record_to_db(self, current_val):
        """Record prune-epoch loss to the DB (used when stopping early for constraint)."""
        try:
            study = self._get_study()
            trial = self._find_own_trial(study)
            if trial is not None:
                self._set_trial_attr(study, trial, self.ATTR_KEY, current_val)
                self._set_trial_attr(study, trial, "prune_epoch", self.prune_epoch)
            self._recorded = True
        except Exception:
            pass