"""
Sequence autoencoder architectures for time series embedding.

Instead of requiring a pre-computed delay embedding, these architectures
take a full (T, D) sequence of partial observations and produce
per-timestep latent vectors: (B, T, D) -> (B, T, D').

This is a learned, flexible alternative to fixed delay embeddings — the
sequence encoder uses causal context to construct a richer per-timestep
representation.  The decoder is a simple pointwise MLP (D' -> D).

Architectures
-------------
- TransformerEncoder (with / without temporal positional embeddings)
- SSMEncoder         (diagonal S4-style, with / without temporal pos. emb.)
- TCNEncoder         (causal dilated temporal convolutions)
- TCNSpatialEncoder  (TCN mixed with spatial convolutions over D)

All encoders expose an ``encode(x) -> (B, T, D')`` interface.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.  x: (B, T, d_model)."""
        return x + self.pe[:, : x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Pointwise MLP decoder  (D' -> D per timestep)
# ---------------------------------------------------------------------------

class StepDecoder(nn.Module):
    """Decode per-timestep latent vectors back to observation space.

    A simple MLP applied independently to each timestep.

    Parameters
    ----------
    n_latent : int
        Latent dimensionality (D').
    n_output : int
        Output feature dimension (D).
    hidden_dim : int
        Hidden layer width.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        n_latent: int,
        n_output: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_output = n_output
        layers: list[nn.Module] = []
        in_dim = n_latent
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D') -> (B, T, D)  or  (B, D') -> (B, D)."""
        return self.net(z)


# ===================================================================
# 1. Transformer Encoder
# ===================================================================

class TransformerSequenceEncoder(nn.Module):
    """Transformer encoder that maps (B, T, D) -> (B, T, D').

    Parameters
    ----------
    n_input : int
        Input feature dimension D.
    n_latent : int
        Latent dimension N.
    d_model : int
        Internal Transformer dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        FFN hidden dim inside each Transformer layer.
    dropout : float
        Dropout rate.
    use_positional_encoding : bool
        Whether to add temporal positional embeddings.
    positional_encoding_type : str
        ``"sinusoidal"`` or ``"learned"``.
    max_len : int
        Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        positional_encoding_type: str = "sinusoidal",
        max_len: int = 5000,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.input_proj = nn.Linear(n_input, d_model)

        self.pos_enc: Optional[nn.Module] = None
        if use_positional_encoding:
            if positional_encoding_type == "learned":
                self.pos_enc = LearnedPositionalEncoding(d_model, max_len)
            else:
                self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.to_latent = nn.Linear(d_model, n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D')."""
        h = self.input_proj(x)  # (B, T, d_model)
        if self.pos_enc is not None:
            h = self.pos_enc(h)
        # Causal mask: position i can only attend to positions <= i
        T = h.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=h.device, dtype=h.dtype
        )
        h = self.transformer(h, mask=causal_mask)  # (B, T, d_model)
        return self.to_latent(h)  # (B, T, D')


# ===================================================================
# 2. State Space Model (diagonal S4-style)
# ===================================================================

class LRULayer(nn.Module):
    """Linear Recurrent Unit (Orvieto et al., 2023).

    Diagonal complex-valued linear recurrence followed by a gated
    feed-forward sublayer (GEGLU), with pre-LayerNorm + residual for both.

    Recurrence (per channel d, state dimension N):
        lambda  = exp(-exp(nu) + i * exp(theta))
        gamma   = sqrt(1 - |lambda|^2)                  (input normalisation)
        x[t]    = lambda * x[t-1] + gamma * B * u[t]   (complex state)
        y[t]    = Re(C * x[t]) + D * u[t]              (real output)

    Feed-forward (GEGLU):
        h   = GELU(W_gate * y) * (W_up * y)
        out = W_down * h

    Parameters
    ----------
    d_model   : int   – feature dimension
    d_state   : int   – SSM state dimension per channel
    r_min     : float – minimum initial |lambda| (0 = may decay to zero)
    r_max     : float – maximum initial |lambda| (1 = unit-circle, long memory)
    ffn_expand: int   – FFN hidden dim = d_model * ffn_expand
    dropout   : float – dropout after each sublayer
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        r_min: float = 0.0,
        r_max: float = 1.0,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Stability parameterisation: |lambda| = exp(-exp(nu)) ∈ [r_min, r_max]
        r = r_min + (r_max - r_min) * torch.rand(d_model, d_state)
        self.nu_log = nn.Parameter(torch.log(-torch.log(r.clamp(min=1e-6))))

        # Phase: theta = exp(theta_log) ∈ (0, pi)
        theta_init = math.pi * torch.rand(d_model, d_state).clamp(min=1e-6)
        self.theta_log = nn.Parameter(torch.log(theta_init))

        # Complex B (input→state) and C (state→output)
        self.B_re = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.B_im = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model))

        # Pre-norms
        self.norm_ssm = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # GEGLU feed-forward
        ffn_dim = d_model * ffn_expand
        self.ffn_gate = nn.Linear(d_model, ffn_dim)
        self.ffn_up   = nn.Linear(d_model, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, T, d_model) -> (B, T, d_model)."""
        B_size, T, _ = u.shape

        # --- SSM sublayer (pre-norm + residual) ---
        u_n = self.norm_ssm(u)

        mag   = torch.exp(-torch.exp(self.nu_log))               # (d_model, d_state)
        phase = torch.exp(self.theta_log)
        lre   = mag * torch.cos(phase)                           # Re(lambda)
        lim   = mag * torch.sin(phase)                           # Im(lambda)
        gamma = torch.sqrt((1.0 - mag ** 2).clamp(min=1e-6))    # input normalisation

        # b[t] = gamma * B * u_n[t]  —  (B, T, d_model, d_state)
        bre = (gamma * self.B_re)[None, None] * u_n[..., None]
        bim = (gamma * self.B_im)[None, None] * u_n[..., None]

        # Parallel inclusive scan: x[t] = lambda * x[t-1] + b[t],  x[-1] = 0
        D_flat = self.d_model * self.d_state
        xr = bre.reshape(B_size, T, D_flat)
        xi = bim.reshape(B_size, T, D_flat)
        ar = lre.reshape(1, 1, D_flat).expand(B_size, T, -1).clone()
        ai = lim.reshape(1, 1, D_flat).expand(B_size, T, -1).clone()

        k = 1
        while k < T:
            xr_s = F.pad(xr[:, :-k], (0, 0, k, 0))
            xi_s = F.pad(xi[:, :-k], (0, 0, k, 0))
            ar_s = F.pad(ar[:, :-k], (0, 0, k, 0), value=1.0)
            ai_s = F.pad(ai[:, :-k], (0, 0, k, 0), value=0.0)
            new_xr = ar * xr_s - ai * xi_s + xr
            new_xi = ar * xi_s + ai * xr_s + xi
            new_ar = ar * ar_s - ai * ai_s
            new_ai = ar * ai_s + ai * ar_s
            xr, xi, ar, ai = new_xr, new_xi, new_ar, new_ai
            k *= 2

        xr = xr.reshape(B_size, T, self.d_model, self.d_state)
        xi = xi.reshape(B_size, T, self.d_model, self.d_state)

        # y[t] = Re(C * x[t]) + D * u_n[t]
        y  = (self.C_re[None, None] * xr - self.C_im[None, None] * xi).sum(-1)
        y  = y + self.D[None, None] * u_n
        y  = self.dropout(y)
        h  = u + y   # residual

        # --- FFN sublayer (pre-norm + residual, GEGLU) ---
        h_n = self.norm_ffn(h)
        out = self.ffn_down(F.gelu(self.ffn_gate(h_n)) * self.ffn_up(h_n))
        out = self.dropout(out)
        return h + out


class SSMSequenceEncoder(nn.Module):
    """Stack of LRU layers: (B, T, D) -> (B, T, D').

    Parameters
    ----------
    n_input   : int   – Input feature dimension D.
    n_latent  : int   – Latent dimension N.
    d_model   : int   – Internal model dimension.
    d_state   : int   – SSM state dimension per channel.
    n_layers  : int   – Number of LRU layers.
    r_min     : float – Minimum initial |lambda| (see LRULayer).
    r_max     : float – Maximum initial |lambda| (see LRULayer).
    ffn_expand: int   – FFN expansion factor inside each LRU layer.
    dropout   : float – Dropout rate.
    use_positional_encoding : bool – Whether to add temporal positional embeddings.
    positional_encoding_type: str  – ``"sinusoidal"`` or ``"learned"``.
    max_len   : int   – Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 3,
        r_min: float = 0.0,
        r_max: float = 1.0,
        ffn_expand: int = 2,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        positional_encoding_type: str = "sinusoidal",
        max_len: int = 5000,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.input_proj = nn.Linear(n_input, d_model)

        self.pos_enc: Optional[nn.Module] = None
        if use_positional_encoding:
            if positional_encoding_type == "learned":
                self.pos_enc = LearnedPositionalEncoding(d_model, max_len)
            else:
                self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            LRULayer(d_model, d_state, r_min, r_max, ffn_expand, dropout)
            for _ in range(n_layers)
        ])
        self.to_latent = nn.Linear(d_model, n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D')."""
        h = self.input_proj(x)
        if self.pos_enc is not None:
            h = self.pos_enc(h)
        for layer in self.layers:
            h = layer(h)
        return self.to_latent(h)  # (B, T, D')


# ===================================================================
# 3. Temporal Convolutional Network (TCN)
# ===================================================================

class CausalConv1d(nn.Module):
    """Causal (left-padded) 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C_out, T)."""
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual block for the Temporal Convolutional Network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C_out, T)."""
        res = self.residual(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.activation(self.norm2(self.conv2(out)))
        out = self.dropout(out)
        return out + res


class TCNSequenceEncoder(nn.Module):
    """Temporal Convolutional Network: (B, T, D) -> (B, T, D').

    Uses exponentially increasing dilation factors to cover the full
    receptive field.

    Parameters
    ----------
    n_input : int
        Input feature dimension D.
    n_latent : int
        Latent dimension N.
    n_channels : int
        Number of channels in each TCN block.
    kernel_size : int
        Convolution kernel size.
    n_layers : int
        Number of TCN blocks (dilation doubles each layer).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_channels: int = 64,
        kernel_size: int = 7,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_latent = n_latent
        blocks: list[nn.Module] = []
        in_ch = n_input
        for i in range(n_layers):
            dilation = 2 ** i
            blocks.append(
                TCNBlock(in_ch, n_channels, kernel_size, dilation, dropout)
            )
            in_ch = n_channels
        self.tcn = nn.Sequential(*blocks)
        self.to_latent = nn.Linear(n_channels, n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D')."""
        h = x.transpose(1, 2)  # (B, D, T)
        h = self.tcn(h)  # (B, n_channels, T)
        h = h.transpose(1, 2)  # (B, T, n_channels)
        return self.to_latent(h)  # (B, T, D')


# ===================================================================
# 4. TCN + Spatial Convolution mixed encoder
# ===================================================================

class SpatialConv1d(nn.Module):
    """1-D convolution over the D spatial (feature) dimensions.

    Treats each time step independently and convolves over the D-axis.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, out_channels).

        We treat T as batch, D as the spatial length, and in_channels=1
        unless the caller has already expanded channels.
        """
        B, T, D = x.shape
        # Reshape: (B*T, 1, D) if we treat each timestep independently
        # But we want richer spatial mixing, so use in_channels as feature dim
        # Convention: channels-first for conv1d
        h = x.reshape(B * T, 1, D)
        h = self.conv(h)  # (B*T, out_channels, D')
        # Pool over spatial dim to get (B*T, out_channels)
        h = h.mean(dim=-1)
        return h.reshape(B, T, -1)


class SpatialBlock(nn.Module):
    """Spatial convolution block with residual connection."""

    def __init__(self, d_model: int, spatial_dim: int, kernel_size: int = 3):
        super().__init__()
        # Project from d_model back to spatial_dim, convolve, project back
        self.to_spatial = nn.Linear(d_model, spatial_dim)
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, d_model, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, spatial_dim: int) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        B, T, C = x.shape
        # Project to spatial domain
        s = self.to_spatial(x)  # (B, T, spatial_dim)
        # Convolve over spatial dim per timestep
        s = s.reshape(B * T, 1, spatial_dim)
        s = self.conv(s)  # (B*T, d_model, spatial_dim)
        s = s.mean(dim=-1).reshape(B, T, C)  # (B, T, d_model)
        return self.norm(self.activation(s) + x)


class TCNSpatialSequenceEncoder(nn.Module):
    """Interleaved TCN (temporal) + spatial convolution encoder.

    Alternates between temporal convolution layers (along T) and spatial
    convolution layers (along D) to jointly capture temporal dynamics
    and cross-channel interactions.

    Parameters
    ----------
    n_input : int
        Input feature dimension D.
    n_latent : int
        Latent dimension N.
    n_channels : int
        Number of channels in each TCN block.
    kernel_size_temporal : int
        Kernel size for temporal convolutions.
    kernel_size_spatial : int
        Kernel size for spatial convolutions.
    n_layers : int
        Number of interleaved (temporal + spatial) blocks.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_channels: int = 64,
        kernel_size_temporal: int = 7,
        kernel_size_spatial: int = 3,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_input = n_input

        self.input_proj = nn.Linear(n_input, n_channels)

        self.temporal_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.temporal_blocks.append(
                TCNBlock(n_channels, n_channels, kernel_size_temporal, dilation, dropout)
            )
            self.spatial_blocks.append(
                SpatialBlock(n_channels, n_input, kernel_size_spatial)
            )

        self.to_latent = nn.Linear(n_channels, n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D')."""
        h = self.input_proj(x)  # (B, T, n_channels)

        for t_block, s_block in zip(self.temporal_blocks, self.spatial_blocks):
            # Temporal conv (channels-first)
            h = t_block(h.transpose(1, 2)).transpose(1, 2)  # (B, T, n_channels)
            # Spatial conv
            h = s_block(h, self.n_input)  # (B, T, n_channels)

        return self.to_latent(h)  # (B, T, D')


# ===================================================================
# 5. Pointwise MLP (no temporal context)
# ===================================================================

class MLPSequenceEncoder(nn.Module):
    """Pointwise MLP encoder: (B, T, D) -> (B, T, D').

    Applies the same MLP independently at every timestep.  Unlike the
    SSM / Transformer / TCN encoders, this adds **no temporal context**
    — each output z_t depends only on x_t.  When paired with a delay
    embedding as input, the delay embedding alone provides temporal
    context (Takens' theorem), avoiding the double-smoothing that
    recurrent encoders introduce.

    Parameters
    ----------
    n_input  : int   – Input feature dimension D (e.g. n_delays * n_obs).
    n_latent : int   – Latent dimension D'.
    hidden_dim : int – Hidden layer width.
    n_layers : int   – Number of hidden layers.
    dropout  : float – Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_latent = n_latent

        layers: list[nn.Module] = []
        in_dim = n_input
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_latent))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D')."""
        return self.net(x)


# ===================================================================
# Unified wrapper:  Encoder + Decoder = SequenceAutoencoder
# ===================================================================

class SequenceAutoencoder(nn.Module):
    """Wraps any sequence encoder + pointwise step decoder.

    Provides the ``encode`` / ``decode`` / ``forward`` interface expected
    by the FNN regulariser and training loop.

    Parameters
    ----------
    encoder : nn.Module
        Must map (B, T, D) -> (B, T, D').
    decoder : StepDecoder
        Must map (B, T, D') -> (B, T, D).
    """

    def __init__(self, encoder: nn.Module, decoder: StepDecoder, context_margin: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.context_margin = context_margin

    @property
    def n_latent(self) -> int:
        return self.encoder.n_latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D')."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, D') -> (B, T, D)  or  (B, D') -> (B, D)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)  per-timestep reconstruction."""
        return self.decode(self.encode(x))


# ===================================================================
# Factory helpers
# ===================================================================

def build_transformer(
    n_input: int,
    n_latent: int,
    use_positional_encoding: bool = True,
    decoder_hidden: int = 128,
    decoder_layers: int = 2,
    decoder_n_output: int = None,
    context_margin: int = 0,
    **encoder_kwargs,
) -> SequenceAutoencoder:
    """Build a Transformer-based SequenceAutoencoder."""
    encoder = TransformerSequenceEncoder(
        n_input=n_input,
        n_latent=n_latent,
        use_positional_encoding=use_positional_encoding,
        **encoder_kwargs,
    )
    decoder = StepDecoder(n_latent, decoder_n_output or n_input, decoder_hidden, decoder_layers)
    return SequenceAutoencoder(encoder, decoder, context_margin=context_margin)


def build_ssm(
    n_input: int,
    n_latent: int,
    use_positional_encoding: bool = True,
    decoder_hidden: int = 128,
    decoder_layers: int = 2,
    decoder_n_output: int = None,
    context_margin: int = 0,
    **encoder_kwargs,
) -> SequenceAutoencoder:
    """Build an SSM-based SequenceAutoencoder."""
    encoder = SSMSequenceEncoder(
        n_input=n_input,
        n_latent=n_latent,
        use_positional_encoding=use_positional_encoding,
        **encoder_kwargs,
    )
    decoder = StepDecoder(n_latent, decoder_n_output or n_input, decoder_hidden, decoder_layers)
    return SequenceAutoencoder(encoder, decoder, context_margin=context_margin)


def build_tcn(
    n_input: int,
    n_latent: int,
    decoder_hidden: int = 128,
    decoder_layers: int = 2,
    decoder_n_output: int = None,
    context_margin: int = 0,
    **encoder_kwargs,
) -> SequenceAutoencoder:
    """Build a TCN-based SequenceAutoencoder."""
    encoder = TCNSequenceEncoder(
        n_input=n_input,
        n_latent=n_latent,
        **encoder_kwargs,
    )
    decoder = StepDecoder(n_latent, decoder_n_output or n_input, decoder_hidden, decoder_layers)
    return SequenceAutoencoder(encoder, decoder, context_margin=context_margin)


def build_tcn_spatial(
    n_input: int,
    n_latent: int,
    decoder_hidden: int = 128,
    decoder_layers: int = 2,
    decoder_n_output: int = None,
    context_margin: int = 0,
    **encoder_kwargs,
) -> SequenceAutoencoder:
    """Build a TCN+Spatial SequenceAutoencoder."""
    encoder = TCNSpatialSequenceEncoder(
        n_input=n_input,
        n_latent=n_latent,
        **encoder_kwargs,
    )
    decoder = StepDecoder(n_latent, decoder_n_output or n_input, decoder_hidden, decoder_layers)
    return SequenceAutoencoder(encoder, decoder, context_margin=context_margin)


def build_mlp(
    n_input: int,
    n_latent: int,
    decoder_hidden: int = 128,
    decoder_layers: int = 2,
    decoder_n_output: int = None,
    context_margin: int = 0,
    **encoder_kwargs,
) -> SequenceAutoencoder:
    """Build a pointwise-MLP SequenceAutoencoder (no temporal context)."""
    encoder = MLPSequenceEncoder(
        n_input=n_input,
        n_latent=n_latent,
        **encoder_kwargs,
    )
    decoder = StepDecoder(n_latent, decoder_n_output or n_input, decoder_hidden, decoder_layers)
    return SequenceAutoencoder(encoder, decoder, context_margin=context_margin)
