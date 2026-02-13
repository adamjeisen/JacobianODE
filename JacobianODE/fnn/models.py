"""
Sklearn-style API for time series embedding using FNN-regularized autoencoders.

Provides fit()/transform() wrappers around the PyTorch autoencoder networks.
Includes both neural network embeddings (MLP, LSTM) and classical baselines
(ETD/PCA, constant-lag, AMI, tICA).

Original TensorFlow implementation: https://github.com/williamgilpin/fnn
Reference: Gilpin, "Deep reconstruction of strange attractors from time series"
           NeurIPS 2020. https://arxiv.org/abs/2002.05909
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA, KernelPCA, SparsePCA
from sklearn.metrics import mutual_info_score
from scipy.signal import savgol_filter, argrelextrema

from .networks import MLPAutoencoder, LSTMAutoencoder
from .regularizers import FNN, DeCov
from .utils import hankel_matrix, resample_dataset, standardize_ts
from .tica import tICA


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TimeSeriesEmbedding:
    """Base class for time series embedding.

    Parameters
    ----------
    n_latent : int
        Embedding dimension.
    time_window : int
        Number of time steps per input window.
    n_features : int
        Number of channels in the time series.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        n_latent: int,
        time_window: int = 10,
        n_features: int = 1,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None, **kwargs):
        """Fit the model and embed X."""
        self.fit(X, **kwargs)
        return self.transform(X)


# ---------------------------------------------------------------------------
# Classical embedding baselines
# ---------------------------------------------------------------------------

class ETDEmbedding(TimeSeriesEmbedding):
    """Embed time series using PCA / Eigen-Time-Delay (Broomhead-King) coordinates.

    Parameters
    ----------
    sparse : bool
        Use SparsePCA.
    kernel : str or callable, optional
        Kernel for KernelPCA.
    """

    def __init__(self, *args, sparse: bool = False, kernel=None, **kwargs):
        super().__init__(*args, **kwargs)
        if kernel:
            self.model = KernelPCA(
                n_components=self.n_latent, kernel=kernel,
                random_state=self.random_state, copy_X=False,
            )
        elif sparse:
            self.model = SparsePCA(
                n_components=self.n_latent, random_state=self.random_state,
            )
        else:
            self.model = PCA(
                n_components=self.n_latent, random_state=self.random_state,
            )

    def fit(self, X, y=None, subsample=None):
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, self.time_window)
        if subsample:
            _, X_train = resample_dataset(
                X_train, subsample, random_state=self.random_state,
            )
        self.model.fit(np.reshape(X_train, (X_train.shape[0], -1)))

    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        return self.model.transform(X_test)


class ConstantLagEmbedding(TimeSeriesEmbedding):
    """Embed using constant (fixed) lag between values (Takens 1981).

    Parameters
    ----------
    lag_time : int
        Constant lag time.
    """

    def __init__(self, *args, lag_time: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lag_time = lag_time

    def fit(self, X, y=None, lag_cutoff=None):
        pass

    def transform(self, X, y=None):
        tau = self.time_window * self.lag_time
        X_test = hankel_matrix(
            standardize_ts(X), q=tau, p=len(X) - tau,
        )
        X_test = X_test[:, ::self.lag_time, :]
        return np.squeeze(X_test)


class AMIEmbedding(ConstantLagEmbedding):
    """Embed using averaged mutual information (Fraser & Swinney 1986).

    Automatically selects the lag time as the first minimum of the
    mutual information function.

    Parameters
    ----------
    lag_cutoff : int, optional
        Maximum time lag to consider.
    """

    def __init__(self, *args, lag_cutoff: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lag_cutoff = lag_cutoff

    @staticmethod
    def _compute_mutual_information(x, y, bins=None):
        if bins:
            b = np.histogram2d(x, y, bins)[0]
        else:
            b = np.histogram2d(x, y, len(x))[0]
        return mutual_info_score(None, None, contingency=b)

    @classmethod
    def _mutual_information_lagged(cls, data, max_time, bins=None):
        all_mi = []
        for tau in range(1, max_time):
            unlagged = data[:-tau]
            lagged = np.roll(data, -tau)[:-tau]
            joint = np.vstack((unlagged, lagged))
            all_mi.append(cls._compute_mutual_information(joint[0, :], joint[1, :], bins))
        return np.array(all_mi)

    @staticmethod
    def _find_minima(x, smoothing_radius=None):
        if smoothing_radius:
            if smoothing_radius % 2 == 0:
                smoothing_radius += 1
            x = savgol_filter(x, smoothing_radius, 3)
        return argrelextrema(x, np.less)

    def fit(self, X, y=None, verbose=False, bins=None, timescale=None):
        Xs = standardize_ts(X)
        if not self.lag_cutoff:
            self.lag_cutoff = int(np.floor(len(Xs) / 2))
        lagged_mi_vals = self._mutual_information_lagged(Xs, self.lag_cutoff, bins)
        lag_times = self._find_minima(lagged_mi_vals, timescale)[0]
        self.lag_time = lag_times[0]


class TICAEmbedding(TimeSeriesEmbedding):
    """Embed time series using time-lagged independent component analysis (tICA).

    Parameters
    ----------
    time_lag : int
        Number of time steps to lag before embedding.
    """

    def __init__(self, *args, time_lag: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_lag = time_lag
        if time_lag > 0:
            self.model = tICA(n_components=self.n_latent, lag_time=time_lag)
        elif time_lag == 0:
            self.model = FastICA(
                n_components=self.n_latent, random_state=self.random_state,
            )
        else:
            raise ValueError("Time delay parameter must be >= 0.")

    def fit(self, X, y=None, subsample=None):
        Xs = standardize_ts(X)
        X_train = hankel_matrix(Xs, self.time_window)
        if subsample:
            _, X_train = resample_dataset(
                X_train, subsample, random_state=self.random_state,
            )
        flat = np.reshape(X_train, (X_train.shape[0], -1))
        if self.time_lag > 0:
            self.model.fit([flat])
        else:
            self.model.fit(flat)

    def transform(self, X, y=None):
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        flat = np.reshape(X_test, (X_test.shape[0], -1))
        if self.time_lag > 0:
            return self.model.transform([flat])[0]
        else:
            return self.model.transform(flat)


# ---------------------------------------------------------------------------
# Neural network embedding (PyTorch training loop)
# ---------------------------------------------------------------------------

class NeuralNetworkEmbedding(TimeSeriesEmbedding):
    """Base class for autoencoder-based time series embeddings.

    Uses a simple PyTorch training loop for the sklearn-style fit() API.
    For full-featured training with Lightning/W&B/Hydra, use the
    LitFNNAutoencoder from fnn.lightning instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None  # Set by subclasses
        self.train_history = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X,
        y=None,
        subsample=None,
        tau: int = 0,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        train_steps: int = 200,
        loss: str = "mse",
        verbose: int = 0,
        optimizer: str = "adam",
        early_stopping: bool = False,
        latent_regularizer=None,
    ):
        """Fit the autoencoder on a time series.

        Parameters
        ----------
        X : np.ndarray
            (T,) or (T, D) time series.
        tau : int
            Prediction horizon offset. 0 = pure autoencoder.
        learning_rate : float
            Learning rate.
        batch_size : int
            Batch size.
        train_steps : int
            Number of training epochs.
        verbose : int
            Verbosity (0=silent, 1=progress, 2=detailed).
        optimizer : str
            'adam' or 'nadam'.
        early_stopping : bool
            Stop if loss plateaus.
        latent_regularizer : nn.Module, optional
            FNN or DeCov regularizer to apply to latent codes.
        """
        # Prepare data
        Xs = standardize_ts(X)
        X0 = hankel_matrix(Xs, self.time_window + tau)
        X_train = X0[:, :self.time_window]
        Y_train = X0[:, -self.time_window:]

        if subsample:
            indices, _ = resample_dataset(
                X_train, subsample, random_state=self.random_state,
            )
            X_train = X_train[indices]
            Y_train = Y_train[indices]

        X_t = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        Y_t = torch.as_tensor(Y_train, dtype=torch.float32, device=self.device)

        self.model = self.model.to(self.device)

        # Set up optimizer
        if optimizer == "nadam":
            opt = torch.optim.NAdam(self.model.parameters(), lr=learning_rate)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        history = {"loss": []}
        best_loss = float("inf")
        patience_counter = 0

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        n_samples = len(X_t)
        for epoch in range(train_steps):
            # Shuffle
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                x_batch = X_t[idx]
                y_batch = Y_t[idx]

                opt.zero_grad()

                recon = self.model(x_batch)
                recon_loss = torch.nn.functional.mse_loss(recon, y_batch)

                reg_loss = torch.tensor(0.0, device=self.device)
                if latent_regularizer is not None:
                    latent = self.model.encode(x_batch)
                    reg_loss = latent_regularizer(latent)

                total_loss = recon_loss + reg_loss
                total_loss.backward()
                opt.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

            if verbose >= 1 and (epoch % max(1, train_steps // 20) == 0 or epoch == train_steps - 1):
                print(f"Epoch {epoch+1}/{train_steps} - loss: {avg_loss:.6f}")

            # Early stopping
            if early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 3:
                        if verbose >= 1:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

        self.train_history = history
        self.model.eval()

    def transform(self, X, y=None):
        """Embed a time series into the latent space.

        Parameters
        ----------
        X : np.ndarray
            (T,) or (T, D) time series.

        Returns
        -------
        np.ndarray
            (n_windows, n_latent) embedding.
        """
        X_test = hankel_matrix(standardize_ts(X), self.time_window)
        X_t = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(X_t)

        return latent.cpu().numpy()


class MLPEmbedding(NeuralNetworkEmbedding):
    """MLP autoencoder embedding with optional FNN regularization.

    Parameters
    ----------
    n_latent : int
        Embedding dimension.
    time_window : int
        Input window length.
    n_features : int
        Number of input channels.
    network_shape : list of int
        Hidden layer sizes.
    latent_regularizer : nn.Module, optional
        Regularizer (e.g., FNN(strength=1.0)).
    """

    def __init__(self, *args, network_shape=None, latent_regularizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if network_shape is None:
            network_shape = [10, 10]
        self._network_shape = network_shape
        self._latent_regularizer = latent_regularizer
        self.model = MLPAutoencoder(
            self.n_latent,
            self.time_window,
            n_features=self.n_features,
            network_shape=network_shape,
        )

    def fit(self, X, y=None, **kwargs):
        if "latent_regularizer" not in kwargs and self._latent_regularizer is not None:
            kwargs["latent_regularizer"] = self._latent_regularizer
        super().fit(X, y=y, **kwargs)


class LSTMEmbedding(NeuralNetworkEmbedding):
    """LSTM autoencoder embedding with optional FNN regularization.

    Parameters
    ----------
    n_latent : int
        Embedding dimension.
    time_window : int
        Input window length.
    n_features : int
        Number of input channels.
    network_shape : list of int
        Hidden sizes for intermediate LSTM layers.
    latent_regularizer : nn.Module, optional
        Regularizer (e.g., FNN(strength=1.0)).
    """

    def __init__(self, *args, network_shape=None, latent_regularizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if network_shape is None:
            network_shape = []
        self._network_shape = network_shape
        self._latent_regularizer = latent_regularizer
        self.model = LSTMAutoencoder(
            self.n_latent,
            self.time_window,
            n_features=self.n_features,
            network_shape=network_shape,
        )

    def fit(self, X, y=None, **kwargs):
        if "latent_regularizer" not in kwargs and self._latent_regularizer is not None:
            kwargs["latent_regularizer"] = self._latent_regularizer
        super().fit(X, y=y, **kwargs)
