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

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension (must match in_dim for residual connection)
        activation (Optional[nn.Module]): Activation function to use. If None, uses identity
        dropout (float): Dropout probability. If 0, no dropout is applied
    """
    def __init__(self, in_dim, out_dim, activation=None, dropout=0.0):
        super().__init__()
        if in_dim != out_dim:
            raise ValueError("Input and output dimensions must match for residual connection")
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation if activation is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        """Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_dim)

        Returns:
            torch.Tensor: Output tensor of shape (..., out_dim)
        """
        return self.dropout(self.activation(self.linear(x))) + x

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
        use_pre_layer_norm (bool): Whether to use layer normalization before each linear layer
        use_pre_batch_norm (bool): Whether to use batch normalization before each linear layer
        use_mean_and_scale (bool): Whether to learn mean and scale parameters for output
        final_layer_scale_factor (float): Scale factor for the final layer
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
            use_pre_layer_norm=False, 
            use_pre_batch_norm=False, 
            use_mean_and_scale=False, 
            final_layer_scale_factor=1.0,
        ):
        super(MLP, self).__init__()

        self.residuals = residuals
        self.dropout = dropout
        self.activation = activation
        self.use_pre_layer_norm = use_pre_layer_norm
        self.use_pre_batch_norm = use_pre_batch_norm
        self.use_mean_and_scale = use_mean_and_scale

        if final_layer_scale_factor is None:
            final_layer_scale_factor = 1.0
        self.final_layer_scale_factor = final_layer_scale_factor

        self.layers = nn.ModuleList()
        
        # Check if hidden_dim is a list; if not, create a list with repeated hidden_dim
        if isinstance(hidden_dim, Iterable) and not isinstance(hidden_dim, (str, bytes)):
            self.hidden_dims = hidden_dim
        else:
            self.hidden_dims = [hidden_dim] * num_layers

        # Ensure the number of hidden dimensions matches the number of layers
        if len(self.hidden_dims) != num_layers:
            raise ValueError("Length of hidden_dim list must match num_layers.")

        if residuals:
            # Add input layer
            self.layers.extend(self._create_layer(input_dim, self.hidden_dims[0], layer_idx=0, dropout=0.0))
            
            # Add hidden layers
            for i in range(1, num_layers):
                self.layers.extend(self._create_layer_with_residuals(self.hidden_dims[i-1], self.hidden_dims[i], layer_idx=i))

            # Add output layer
            self.layers.extend(self._create_layer(self.hidden_dims[-1], output_dim, activation=None, dropout=0.0, no_activation=True, layer_idx=num_layers))
        else:
            # Add input layer

            self.layers.extend(self._create_layer(input_dim, self.hidden_dims[0], layer_idx=0, dropout=0.0))
            
            # Add hidden layers
            for i in range(1, num_layers):
                self.layers.extend(self._create_layer(self.hidden_dims[i-1], self.hidden_dims[i], layer_idx=i))

            # Add output layer
            self.layers.extend(self._create_layer(self.hidden_dims[-1], output_dim, activation=None, dropout=0.0, no_activation=True, layer_idx=num_layers))

        if use_mean_and_scale:
            self.mu = nn.Parameter(torch.zeros(1))
            self.log_sigma = nn.Parameter(torch.zeros(1))
        else:
            self.mu = torch.tensor(0)
            self.log_sigma = torch.tensor(0)

        # self.scale_final_layer(final_layer_scale_factor)

        self.MODEL_TYPE = 'MLP'
        
        self.input_dim = input_dim

    def _create_layer_with_residuals(self, in_dim, out_dim, activation=None, dropout=None, no_activation=False, layer_idx=None):
        """Create a layer with residual connections.

        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            activation (Optional[str]): Activation function name
            dropout (Optional[float]): Dropout probability
            no_activation (bool): Whether to skip activation
            layer_idx (Optional[int]): Layer index for debugging

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
        if self.use_pre_layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        if self.use_pre_batch_norm:
            layers.append(nn.BatchNorm1d(in_dim))

        layers.append(nn.Linear(in_dim, out_dim))
    
        if not no_activation:
            layers.append(get_activation_func(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x*torch.exp(self.log_sigma) + self.mu
    
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
        rescaling_sigma (float): Scaling factor for Jacobians
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
            return self.model(batch).reshape(*batch.shape[:-1], batch.shape[-1], batch.shape[-1])*self.rescaling_sigma
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
    