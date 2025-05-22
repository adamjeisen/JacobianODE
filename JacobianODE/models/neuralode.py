import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..jacobians.lightning_base import LitBase
from .mlp import MLP, LitMLP



class NeuralODE(nn.Module):
    """A neural ordinary differential equation model.

    This class implements a neural ODE, which uses a neural network to parameterize
    the dynamics of an ordinary differential equation. The model can be used for
    both prediction and generation of time series data.

    The dynamics are defined as:
        dy/dt = f(y) = net(y)
    where net is a neural network (MLP) that predicts the time derivative.

    Args:
        input_dim (int): Dimension of the state space
        dt (float): Time step size for numerical integration
        mlp_kwargs (dict): Keyword arguments for the underlying MLP network.
                          If not specified, uses default values:
                          - hidden_dim=100
                          - num_layers=2
    """
    def __init__(self, input_dim, dt=1, mlp_kwargs={}):
        super(NeuralODE, self).__init__()

        if 'hidden_dim' not in mlp_kwargs:
            mlp_kwargs['hidden_dim'] = 100
        if 'num_layers' not in mlp_kwargs:
            mlp_kwargs['num_layers'] = 2

        mlp_kwargs['input_dim'] = input_dim
        mlp_kwargs['output_dim'] = input_dim
        self.net = MLP(**mlp_kwargs)
        self.dt = dt

    def forward(self, t, y):
        """Compute the time derivative dy/dt at time t and state y.

        This method defines the dynamics of the ODE system. The time t is included
        for compatibility with torchdiffeq but is not used in the computation.

        Args:
            t (torch.Tensor): Current time (unused)
            y (torch.Tensor): Current state

        Returns:
            torch.Tensor: Time derivative dy/dt
        """
        return self.net(y)
    
    def generate(self, x, nt=None, alpha=0, teacher_forcing_steps=1):
        """Generate a trajectory by solving the neural ODE.

        This method can operate in two modes:
        1. Direct integration: If nt is specified, integrates the ODE for nt steps
        2. Teacher forcing: If nt is None, uses teacher forcing with alpha parameter
           to generate a trajectory of the same length as the input

        Args:
            x (torch.Tensor): Initial condition or input sequence
                            If nt is specified: shape (..., D) for initial condition
                            If nt is None: shape (..., T, D) for input sequence
            nt (Optional[int]): Number of time steps to predict. If None, uses length of x
            alpha (float): Teacher forcing coefficient (0 to 1)
                          - alpha=0: No teacher forcing (pure prediction)
                          - alpha=1: Full teacher forcing
            teacher_forcing_steps (int): Number of steps to use teacher forcing at once

        Returns:
            torch.Tensor: Generated trajectory
                         If nt is specified: shape (..., nt, D)
                         If nt is None: shape (..., T, D)
        """
        if nt is not None:
            # Direct integration mode
            t_test = torch.arange(nt).type(x.dtype).to(x.device)*self.dt
            sol_pred = odeint(
                self, 
                x,
                t_test,
                method='rk4'
            )

            # Rearrange dimensions to put time dimension after batch dimensions
            dims = list(range(1, len(sol_pred.shape) - 1))  # This will be [1, 2, ..., n]
            dims = dims + [0, len(sol_pred.shape) - 1]     # Add T (dim 0) and D (last dim)
            sol_pred = sol_pred.permute(*dims)
        else:
            # Teacher forcing mode
            nt = x.shape[-2]
            if alpha == 0:
                return self.generate(x[..., 0, :], nt=nt, alpha=0)
            sol_pred = x[..., [0], :]
            t = 0
            while t < nt - 1:
                n_steps = min(teacher_forcing_steps, nt - t - 1)
                ic_tf = (1-alpha)*sol_pred[..., -1, :] + alpha*x[..., t, :]
                pred_pts = self.generate(ic_tf, nt=n_steps + 1, alpha=0)[..., 1:, :]
                sol_pred = torch.cat([sol_pred, pred_pts], dim=-2)
                t += n_steps
        
        return sol_pred

class LitNeuralODE(LitBase):
    """PyTorch Lightning module for the NeuralODE model.

    This class extends LitBase to implement a PyTorch Lightning module for the NeuralODE model.
    It adds methods for computing Jacobians of the dynamics.

    Args:
        direct (bool): Whether to compute Jacobians directly or using autograd
        rescaling_sigma (float): Scaling factor for Jacobians
        dt (float): Time step size for Jacobian computation
    """

    def compute_jacobians(self, batch, t=0, batch_idx=0, dataloader_idx=0):
        """Compute Jacobian matrices of the neural ODE dynamics.

        This method computes the Jacobian matrices of the neural network's output
        with respect to its input, which represents the local linearization of the
        ODE dynamics.

        Args:
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
        if len(batch.shape) == 1:
            batch = batch.unsqueeze(0)
        
        # Compute Jacobians using forward-mode automatic differentiation
        jacs = torch.func.vmap(torch.func.jacfwd(lambda x: self.model.net(x)))(batch)
        
        # Handle different input shapes and rearrange dimensions
        if len(jacs.shape) >= 4:
            jacs = jacs.transpose(-3, -2)
            modified_jacs = jacs.clone()
            modified_jacs = modified_jacs[..., torch.arange(jacs.shape[-4]), torch.arange(jacs.shape[-3]), :, :]
        else:
            modified_jacs = jacs
            
        if reshape:
            modified_jacs = modified_jacs.reshape(*batches, -1, modified_jacs.shape[-2], modified_jacs.shape[-1])
        return modified_jacs