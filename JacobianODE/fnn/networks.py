"""
PyTorch autoencoder architectures for time series embedding with FNN regularization.

Original TensorFlow implementation: https://github.com/williamgilpin/fnn
Reference: Gilpin, "Deep reconstruction of strange attractors from time series"
           NeurIPS 2020. https://arxiv.org/abs/2002.05909
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNoise(nn.Module):
    """Adds Gaussian noise during training only (like tf.keras.layers.GaussianNoise)."""

    def __init__(self, stddev: float = 0.5):
        super().__init__()
        self.stddev = stddev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.stddev > 0:
            return x + torch.randn_like(x) * self.stddev
        return x


class MLPAutoencoder(nn.Module):
    """Fully-connected autoencoder for time series.

    Architecture:
        Encoder: Flatten -> GaussianNoise -> [Linear -> BN -> ELU] * depth -> Linear -> BN
        Decoder: GaussianNoise -> [Linear -> BN -> ELU] * depth -> Linear -> BN -> Reshape

    Parameters
    ----------
    n_latent : int
        Dimensionality of the latent space (embedding dimension).
    time_window : int
        Number of time steps in each input window.
    n_features : int
        Number of channels/features in the time series.
    network_shape : list of int
        Hidden layer sizes for encoder (reversed for decoder).
    """

    def __init__(
        self,
        n_latent: int,
        time_window: int,
        n_features: int = 1,
        network_shape: list = None,
    ):
        super().__init__()
        if network_shape is None:
            network_shape = [10, 10]

        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features

        # --- Encoder ---
        encoder_layers = []
        in_features = time_window * n_features
        encoder_layers.append(nn.Flatten())
        encoder_layers.append(GaussianNoise(0.5))
        for hidden_size in network_shape:
            encoder_layers.append(nn.Linear(in_features, hidden_size))
            encoder_layers.append(nn.BatchNorm1d(hidden_size))
            encoder_layers.append(nn.ELU())
            in_features = hidden_size
        encoder_layers.append(nn.Linear(in_features, n_latent))
        encoder_layers.append(nn.BatchNorm1d(n_latent))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder ---
        decoder_layers = []
        in_features = n_latent
        decoder_layers.append(GaussianNoise(0.5))
        for hidden_size in network_shape[::-1]:
            decoder_layers.append(nn.Linear(in_features, hidden_size))
            decoder_layers.append(nn.BatchNorm1d(hidden_size))
            decoder_layers.append(nn.ELU())
            in_features = hidden_size
        decoder_layers.append(nn.Linear(in_features, time_window * n_features))
        decoder_layers.append(nn.BatchNorm1d(time_window * n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input windows to latent space.

        Parameters
        ----------
        x : torch.Tensor
            (batch_size, time_window, n_features) input.

        Returns
        -------
        torch.Tensor
            (batch_size, n_latent) latent representation.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to reconstructed windows.

        Parameters
        ----------
        z : torch.Tensor
            (batch_size, n_latent) latent representation.

        Returns
        -------
        torch.Tensor
            (batch_size, time_window, n_features) reconstruction.
        """
        out = self.decoder(z)
        return out.view(-1, self.time_window, self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder for time series.

    Architecture:
        Encoder: GaussianNoise -> [LSTM -> BN -> ELU] * depth -> LSTM(final, no return_seq) -> BN
        Decoder: GaussianNoise -> RepeatVector -> LSTM -> [ReverseLSTM] * depth -> ReverseLSTM(final) -> BN

    Parameters
    ----------
    n_latent : int
        Dimensionality of the latent space (embedding dimension).
    time_window : int
        Number of time steps in each input window.
    n_features : int
        Number of channels/features in the time series.
    network_shape : list of int
        Hidden layer sizes for intermediate LSTM layers.
    """

    def __init__(
        self,
        n_latent: int,
        time_window: int,
        n_features: int = 1,
        network_shape: list = None,
    ):
        super().__init__()
        if network_shape is None:
            network_shape = []

        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.network_shape = network_shape

        # --- Encoder ---
        self.encoder_noise = GaussianNoise(0.5)

        self.encoder_lstms = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        in_size = n_features
        for hidden_size in network_shape:
            self.encoder_lstms.append(
                nn.LSTM(in_size, hidden_size, batch_first=True)
            )
            self.encoder_bns.append(nn.BatchNorm1d(hidden_size))
            in_size = hidden_size

        self.encoder_final = nn.LSTM(in_size, n_latent, batch_first=True)
        self.encoder_final_bn = nn.BatchNorm1d(n_latent)

        # --- Decoder ---
        self.decoder_noise = GaussianNoise(0.5)
        self.decoder_initial = nn.LSTM(n_latent, n_latent, batch_first=True)

        self.decoder_lstms = nn.ModuleList()
        in_size = n_latent
        for hidden_size in network_shape[::-1]:
            self.decoder_lstms.append(
                nn.LSTM(in_size, hidden_size, batch_first=True)
            )
            in_size = hidden_size

        self.decoder_final = nn.LSTM(in_size, n_features, batch_first=True)
        self.decoder_final_bn = nn.BatchNorm1d(n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input windows to latent space.

        Parameters
        ----------
        x : torch.Tensor
            (batch_size, time_window, n_features) input.

        Returns
        -------
        torch.Tensor
            (batch_size, n_latent) latent representation.
        """
        h = self.encoder_noise(x)

        for lstm, bn in zip(self.encoder_lstms, self.encoder_bns):
            h, _ = lstm(h)
            # BN over features: (B, T, C) -> (B, C, T) -> BN -> (B, T, C)
            h = bn(h.transpose(1, 2)).transpose(1, 2)
            h = F.elu(h)

        output, (h_n, _) = self.encoder_final(h)
        # h_n: (1, batch_size, n_latent) -> (batch_size, n_latent)
        latent = h_n.squeeze(0)
        latent = self.encoder_final_bn(latent)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to reconstructed windows.

        Parameters
        ----------
        z : torch.Tensor
            (batch_size, n_latent) latent representation.

        Returns
        -------
        torch.Tensor
            (batch_size, time_window, n_features) reconstruction.
        """
        z = self.decoder_noise(z)
        # RepeatVector: (B, n_latent) -> (B, time_window, n_latent)
        h = z.unsqueeze(1).repeat(1, self.time_window, 1)

        h, _ = self.decoder_initial(h)

        # Decoder intermediate LSTMs (go_backwards=True: flip input, run, flip output)
        for lstm in self.decoder_lstms:
            h = torch.flip(h, [1])
            h, _ = lstm(h)
            h = torch.flip(h, [1])

        # Final decoder LSTM (also go_backwards)
        h = torch.flip(h, [1])
        h, _ = self.decoder_final(h)
        h = torch.flip(h, [1])

        # BN over features
        h = self.decoder_final_bn(h.transpose(1, 2)).transpose(1, 2)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
