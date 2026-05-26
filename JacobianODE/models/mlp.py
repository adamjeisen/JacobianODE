from collections.abc import Iterable
import torch
from torch import nn
import torch.nn.functional as F

from ..jacobians.lightning_base import get_activation_func, LitBase

# ----------------------------------------
# MLP
# ----------------------------------------

class ResidualBlock(nn.Module):
    """A residual block that adds the input to the output of a linear layer.

    This block implements a residual connection where the input is added to the output
    of a linear transformation, followed by activation and dropout. This helps with
    gradient flow in deep networks.

    Optionally, a per-call condition tensor ``c`` (shape (..., c_dim)) can be
    concatenated to the block's input before the linear projection. This is
    only active when ``c_dim > 0``; the linear is then sized
    ``(in_dim + c_dim) -> out_dim``. The residual skip still adds the
    unchanged ``x`` of shape (..., in_dim) — c is fused only inside the
    nonlinear branch, keeping the skip path identity.

    Args:
        in_dim (int): Input dimension (= skip-path width)
        out_dim (int): Output dimension (must match in_dim for residual connection)
        c_dim (int): Condition-vector width to concat at this block's input.
            ``0`` (default) = legacy behaviour (no per-block injection).
        activation (Optional[nn.Module]): Activation function to use. If None, uses identity
        dropout (float): Dropout probability. If 0, no dropout is applied
    """
    def __init__(self, in_dim, out_dim, c_dim=0, activation=None, dropout=0.0):
        super().__init__()
        if in_dim != out_dim:
            raise ValueError("Input and output dimensions must match for residual connection")
        self.c_dim = int(c_dim)
        self.linear = nn.Linear(in_dim + self.c_dim, out_dim)
        self.activation = activation if activation is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, c=None):
        """Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_dim)
            c (Optional[torch.Tensor]): Per-sample condition of shape
                (B, c_dim) or matching x's leading dims. Required when
                ``c_dim > 0``; ignored otherwise.

        Returns:
            torch.Tensor: Output tensor of shape (..., out_dim)
        """
        if self.c_dim > 0:
            if c is None:
                raise ValueError(
                    f"ResidualBlock was built with c_dim={self.c_dim} "
                    f"but forward was called with c=None.")
            if c.ndim == x.ndim:
                c_bcast = c
            else:
                view_shape = (c.shape[0],) + (1,) * (x.ndim - 2) + (c.shape[-1],)
                c_bcast = c.view(view_shape).expand(*x.shape[:-1], c.shape[-1])
            x_in = torch.cat([x, c_bcast], dim=-1)
        else:
            x_in = x
        return self.dropout(self.activation(self.linear(x_in))) + x

class MLP(nn.Module):
    """A multi-layer perceptron with various architectural options.

    This MLP implementation supports several advanced features:
    - Residual connections
    - Layer normalization
    - Batch normalization
    - Dropout
    - Custom activation functions
    - Variable hidden layer dimensions
    - Mean and scale parameters for output

    Args:
        input_dim (int): Dimension of input features
        hidden_dim (Union[int, List[int]]): Hidden layer dimension(s). If int, all layers use same dimension.
                                           If list, must match num_layers length.
        num_layers (int): Number of hidden layers
        output_dim (int): Dimension of output features
        residuals (bool): Whether to use residual connections
        dropout (float): Dropout probability
        activation (str): Name of activation function to use
    """
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            residuals=False,
            dropout=0.0,
            activation='relu',
            condition_dim=0,
            condition_inject_per_layer=False,
        ):
        """``condition_inject_per_layer`` (default False = legacy):
        when True AND ``condition_dim > 0`` AND ``residuals=True``, the
        condition vector ``c`` is re-concatenated at the input of every
        residual block (not just the first input layer). Each
        ResidualBlock's linear is sized ``(hidden + c_dim) -> hidden``;
        the skip path stays identity. The output projection still does
        not see ``c`` directly (it's a pure projection). False keeps the
        legacy behaviour (c only enters at the input layer).
        """
        super(MLP, self).__init__()

        self.residuals = residuals
        self.dropout = dropout
        self.activation = activation
        self.condition_dim = condition_dim
        self.condition_inject_per_layer = bool(condition_inject_per_layer)

        self.layers = nn.ModuleList()

        # Check if hidden_dim is a list; if not, create a list with repeated hidden_dim
        if isinstance(hidden_dim, Iterable) and not isinstance(hidden_dim, (str, bytes)):
            self.hidden_dims = hidden_dim
        else:
            self.hidden_dims = [hidden_dim] * num_layers

        # Ensure the number of hidden dimensions matches the number of layers
        if len(self.hidden_dims) != num_layers:
            raise ValueError("Length of hidden_dim list must match num_layers.")

        # When conditioned, the first layer takes [x; c]; size it accordingly.
        first_in = input_dim + condition_dim

        if residuals:
            # Add input layer
            self.layers.extend(self._create_layer(first_in, self.hidden_dims[0], layer_idx=0, dropout=0.0))

            # Add hidden layers
            block_c_dim = (self.condition_dim
                           if self.condition_inject_per_layer else 0)
            for i in range(1, num_layers):
                self.layers.extend(self._create_layer_with_residuals(
                    self.hidden_dims[i-1], self.hidden_dims[i],
                    c_dim=block_c_dim, layer_idx=i))

            # Add output layer
            self.layers.extend(self._create_layer(self.hidden_dims[-1], output_dim, activation=None, dropout=0.0, no_activation=True, layer_idx=num_layers))
        else:
            # Add input layer
            self.layers.extend(self._create_layer(first_in, self.hidden_dims[0], layer_idx=0, dropout=0.0))

            # Add hidden layers
            for i in range(1, num_layers):
                self.layers.extend(self._create_layer(self.hidden_dims[i-1], self.hidden_dims[i], layer_idx=i))

            # Add output layer
            self.layers.extend(self._create_layer(self.hidden_dims[-1], output_dim, activation=None, dropout=0.0, no_activation=True, layer_idx=num_layers))

        self.MODEL_TYPE = 'MLP'

        self.input_dim = input_dim

    def _create_layer_with_residuals(self, in_dim, out_dim, activation=None,
                                      dropout=None, no_activation=False,
                                      layer_idx=None, c_dim=0):
        """Create a layer with residual connections.

        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            activation (Optional[str]): Activation function name
            dropout (Optional[float]): Dropout probability
            no_activation (bool): Whether to skip activation
            layer_idx (Optional[int]): Layer index for debugging
            c_dim (int): Per-block condition-vector width (0 = no per-block
                c injection; matches legacy behaviour).

        Returns:
            List[nn.Module]: List containing a ResidualBlock
        """
        if activation is None:
            activation = self.activation
        if dropout is None:
            dropout = self.dropout

        if no_activation:
            return [nn.Linear(in_dim, out_dim)]

        return [
            ResidualBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                c_dim=c_dim,
                activation=get_activation_func(activation),
                dropout=dropout
            )
        ]
    
    def _create_layer(self, in_dim, out_dim, activation=None, dropout=None, no_activation=False, layer_idx=None):
        """Create a standard neural network layer.

        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            activation (Optional[str]): Activation function name
            dropout (Optional[float]): Dropout probability
            no_activation (bool): Whether to skip activation
            layer_idx (Optional[int]): Layer index for debugging

        Returns:
            List[nn.Module]: List of layer components (normalization, linear, activation, dropout)
        """
        if activation is None:
            activation = self.activation
        if dropout is None:
            dropout = self.dropout
        layers = []

        layers.append(nn.Linear(in_dim, out_dim))
    
        if not no_activation:
            layers.append(get_activation_func(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        # return nn.Sequential(*layers)
        return layers

    def forward(self, x, c=None):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim).
            c (torch.Tensor or None): Optional per-sample condition of
                shape (B, condition_dim). When ``condition_dim > 0``, ``c``
                is concatenated to ``x`` along the last dim (broadcast over
                intermediate dims). Required if ``condition_dim > 0``;
                ignored otherwise.

        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim)
        """
        c_orig = c
        if self.condition_dim > 0:
            if c is None:
                raise ValueError(
                    f"MLP was built with condition_dim={self.condition_dim} "
                    f"but forward was called with c=None. Pass the per-sample "
                    f"condition tensor of shape (B, {self.condition_dim})."
                )
            if c.shape[-1] != self.condition_dim:
                raise ValueError(
                    f"MLP condition_dim={self.condition_dim} but got c with "
                    f"last-dim={c.shape[-1]}."
                )
            # Broadcast c over the intermediate dims of x (typically time).
            if c.ndim == x.ndim:
                c_bcast = c
            else:
                view_shape = (c.shape[0],) + (1,) * (x.ndim - 2) + (c.shape[-1],)
                c_bcast = c.view(view_shape).expand(*x.shape[:-1], c.shape[-1])
            x = torch.cat([x, c_bcast], dim=-1)
        # When per-layer condition injection is on, ResidualBlocks were
        # built with c_dim>0 and need c at their .forward(). Other layer
        # types (Linear / Activation / final projection) take only x.
        for layer in self.layers:
            if (self.condition_inject_per_layer
                    and isinstance(layer, ResidualBlock)
                    and layer.c_dim > 0):
                x = layer(x, c=c_orig)
            else:
                x = layer(x)
        return x
    
    # implement forward but with teacher forcing
    # the teacher forcing parameter is alpha
    # if alpha is 0, then we use the previous step as the input
    # if alpha is 1, then we use the previous step as the input
    # if alpha is between 0 and 1, then we use a combination of the previous step and the input
    def generate(self, x, alpha=1.0):
        """Generate a sequence using teacher forcing.

        This method generates a sequence by using a combination of the model's predictions
        and the true inputs (teacher forcing). The alpha parameter controls the mixing:
        - alpha=0: Use only model predictions
        - alpha=1: Use only true inputs
        - 0<alpha<1: Mix predictions and true inputs

        Args:
            x (torch.Tensor): Input sequence of shape (..., T, input_dim)
            alpha (float): Teacher forcing parameter between 0 and 1

        Returns:
            torch.Tensor: Generated sequence of shape (..., T, output_dim)
        """
        x_out = [x[..., [0], :]]
        for t in range(1, x.shape[-2]):
            x_tf = (1 - alpha) * x_out[-1] + alpha * x[..., [t-1], :]
            x_t = self(x_tf)
            x_out.append(x_t)
        return torch.cat(x_out, dim=-2)

# define the LightningModule
class LitMLP(LitBase):
    """PyTorch Lightning module for the MLP model.

    This class extends LitBase to implement a PyTorch Lightning module for the MLP model.
    It adds methods for computing Jacobians and logging model devices.

    Args:
        direct (bool): Whether to compute Jacobians directly or using autograd
        dt (float): Time step size for Jacobian computation
    """

    def compute_jacobians(self, batch, t=0, batch_idx=0, dataloader_idx=0):
        """Compute Jacobian matrices for a batch of inputs.

        This method computes the Jacobian matrices of the network's output with respect
        to its input. It can compute Jacobians either directly (if self.direct is True)
        or using automatic differentiation.

        Args:
            batch (torch.Tensor): Input batch of shape (..., T, input_dim)
            t (int): Time step (unused, kept for interface consistency)
            batch_idx (int): Batch index (unused, kept for interface consistency)
            dataloader_idx (int): Dataloader index (unused, kept for interface consistency)

        Returns:
            torch.Tensor: Jacobian matrices of shape (..., T, output_dim, input_dim)
        """
        if self.direct:
            return self.model(batch).reshape(*batch.shape[:-1], batch.shape[-1], batch.shape[-1])
        else: # not direct jacobian estimation
            reshape = False
            if len(batch.shape) > 3:
                reshape = True
                batches = batch.shape[:-2]
                batch = batch.reshape(-1, batch.shape[-2], batch.shape[-1])
            # reverse mode
            # jacs = torch.func.vmap(torch.func.jacrev(lambda x: self(x)))(batch)
            # forward mode
            jacs = torch.func.vmap(torch.func.jacfwd(lambda x: self.model(x)))(batch)
            jacs = jacs.transpose(-3, -2)
            modified_jacs = jacs.clone()
            modified_jacs = modified_jacs[..., torch.arange(jacs.shape[-4]), torch.arange(jacs.shape[-3]), :, :]
            # modified_jacs = (modified_jacs - torch.eye(jacs.shape[-1]).to(jacs.device))/self.dt
            if reshape:
                modified_jacs = modified_jacs.reshape(*batches, -1, modified_jacs.shape[-2], modified_jacs.shape[-1])
            return modified_jacs
    
    @staticmethod
    def log_model_devices(model):
        """Log the devices of all model parameters and buffers.

        This utility function prints the device location of all parameters and buffers
        in the model, which is useful for debugging device placement issues.

        Args:
            model (nn.Module): The model to inspect
        """
        for name, param in model.named_parameters():
            print(f"Parameter: {name}, device: {param.device}")
        for name, buf in model.named_buffers():
            print(f"Buffer: {name}, device: {buf.device}")
    