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
    cache_normalization : bool
        If True, cache the training-set mean and std during fit() and
        reuse them in transform(). If False (default), each call to
        transform() standardizes using the new data's own statistics,
        matching the original FNN library behavior.
    """

    def __init__(
        self,
        n_latent: int,
        time_window: int = 10,
        n_features: int = 1,
        random_state: Optional[int] = None,
        cache_normalization: bool = False,
        **kwargs,
    ):
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.random_state = random_state
        self.cache_normalization = cache_normalization
        self._train_mean = None
        self._train_std = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standardize a time series, optionally caching statistics.

        Parameters
        ----------
        X : np.ndarray
            (T,), (T, D), or (N, T, D) time series.
        fit : bool
            If True (during fit), compute and cache mean/std.
            If False (during transform), reuse cached values when
            cache_normalization is enabled.

        Returns
        -------
        np.ndarray
            Standardized time series.
        """
        if fit and self.cache_normalization:
            if X.ndim == 3:
                # 3D: compute global stats across all trials and time
                self._train_mean = np.mean(X, axis=(0, 1), keepdims=True)  # (1, 1, D)
                self._train_std = np.std(X, axis=(0, 1), keepdims=True)    # (1, 1, D)
            else:
                self._train_mean = np.mean(X, axis=0, keepdims=True)
                self._train_std = np.std(X, axis=0, keepdims=True)
            self._train_std[self._train_std == 0] = 1
            return (X - self._train_mean) / self._train_std

        if not fit and self.cache_normalization and self._train_mean is not None:
            return (X - self._train_mean) / self._train_std

        return standardize_ts(X)

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
    tau : int
        Time-point separation (delay) between successive columns in the
        Hankel window.  Defaults to 1 (consecutive time points).
    """

    def __init__(self, *args, sparse: bool = False, kernel=None, tau: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
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
        Xs = self._standardize(X, fit=True)
        X_train = hankel_matrix(Xs, self.time_window, tau=self.tau)
        # 4D from 3D input: collapse (N, n_windows) → (N*n_windows)
        if X_train.ndim == 4:
            N, n_win = X_train.shape[:2]
            X_train = X_train.reshape(N * n_win, *X_train.shape[2:])
        if subsample:
            _, X_train = resample_dataset(
                X_train, subsample, random_state=self.random_state,
            )
        self.model.fit(np.reshape(X_train, (X_train.shape[0], -1)))

    def transform(self, X, y=None):
        X_test = hankel_matrix(self._standardize(X), self.time_window, tau=self.tau)
        # 4D from 3D input: collapse and restore trial dimension
        trial_shape = None
        if X_test.ndim == 4:
            N, n_win = X_test.shape[:2]
            trial_shape = (N, n_win)
            X_test = X_test.reshape(N * n_win, *X_test.shape[2:])
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        result = self.model.transform(X_test)
        if trial_shape is not None:
            result = result.reshape(*trial_shape, result.shape[-1])
        return result


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
            self._standardize(X), q=tau, p=len(X) - tau,
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
        Xs = self._standardize(X, fit=True)
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
        Xs = self._standardize(X, fit=True)
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
        X_test = hankel_matrix(self._standardize(X), self.time_window)
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

    def _build_model(self):
        """Rebuild the autoencoder network. Subclasses must override."""
        raise NotImplementedError

    def count_parameters(self):
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(
        self,
        X,
        y=None,
        subsample=None,
        tau: int = 0,
        # learning_rate: float = 1e-3,
        learning_rate: float = 1e-4,
        # batch_size: int = 100,
        batch_size: int = 64,
        train_steps: int = 200,
        loss: str = "mse",
        verbose: int = 0,
        optimizer: str = "adam",
        early_stopping: bool = False,
        early_stopping_mode: str = "percent_thresh",
        early_stopping_patience: int = 2,
        percent_thresh: float = 0.01,
        latent_regularizer=None,
    ):
        """Fit the autoencoder on a time series.

        Parameters
        ----------
        X : np.ndarray
            (T,), (T, D), or (N, T, D) time series.
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
        early_stopping_mode : str
            'percent_thresh' (default) uses percent improvement threshold;
            'absolute' uses the original absolute-improvement check.
        early_stopping_patience : int
            Number of consecutive epochs below threshold before stopping.
        percent_thresh : float
            Minimum fractional improvement required per epoch (default 0.01 = 1%).
        latent_regularizer : nn.Module, optional
            FNN or DeCov regularizer to apply to latent codes.
        """
        # Auto-detect n_features from input data
        n_features = 1 if X.ndim == 1 else X.shape[-1]
        if n_features != self.n_features:
            self.n_features = n_features
            self._build_model()

        # Prepare data
        Xs = self._standardize(X, fit=True)
        X0 = hankel_matrix(Xs, self.time_window + tau)

        # 4D from 3D input: (N, n_windows, tw+tau, D)
        if X0.ndim == 4:
            X_train = X0[:, :, :self.time_window]
            Y_train = X0[:, :, -self.time_window:]
            N, n_win = X_train.shape[:2]
            X_train = X_train.reshape(N * n_win, *X_train.shape[2:])
            Y_train = Y_train.reshape(N * n_win, *Y_train.shape[2:])
        else:
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
            epoch_recon_loss = 0.0
            epoch_reg_loss = 0.0
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
                epoch_recon_loss += recon_loss.item()
                epoch_reg_loss += reg_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_recon_loss = epoch_recon_loss / max(n_batches, 1)
            avg_reg_loss = epoch_reg_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

            if verbose >= 1 and (epoch % max(1, train_steps // 20) == 0 or epoch == train_steps - 1):
                msg = f"Epoch {epoch+1}/{train_steps} - loss: {avg_loss:.6f}"
                if latent_regularizer is not None:
                    msg += f" (recon: {avg_recon_loss:.6f}, reg: {avg_reg_loss:.6f})"
                print(msg)

            # Early stopping
            if early_stopping:
                if early_stopping_mode == "percent_thresh":
                    if best_loss < float("inf") and best_loss > 0:
                        improvement = (best_loss - avg_loss) / best_loss
                        if improvement < percent_thresh:
                            patience_counter += 1
                        else:
                            patience_counter = 0
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                else:
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                if patience_counter >= early_stopping_patience:
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
            (T,), (T, D), or (N, T, D) time series.

        Returns
        -------
        np.ndarray
            (n_windows, n_latent) or (N, n_windows, n_latent) embedding.
        """
        X_test = hankel_matrix(self._standardize(X), self.time_window)

        # 4D from 3D input: (N, n_windows, tw, D)
        trial_shape = None
        if X_test.ndim == 4:
            N, n_win = X_test.shape[:2]
            trial_shape = (N, n_win)
            X_test = X_test.reshape(N * n_win, *X_test.shape[2:])

        X_t = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(X_t)

        result = latent.cpu().numpy()
        if trial_shape is not None:
            result = result.reshape(*trial_shape, result.shape[-1])
        return result

    def reconstruct(self, X, y=None):
        """Reconstruct a time series through the autoencoder.

        Parameters
        ----------
        X : np.ndarray
            (T,), (T, D), or (N, T, D) time series.

        Returns
        -------
        tuple of np.ndarray
            (X_windows, recon) where both have shape
            (n_windows, time_window, n_features) or
            (N, n_windows, time_window, n_features) for 3D input.
        """
        X_test = hankel_matrix(self._standardize(X), self.time_window)

        # 4D from 3D input: (N, n_windows, tw, D)
        trial_shape = None
        if X_test.ndim == 4:
            N, n_win = X_test.shape[:2]
            trial_shape = (N, n_win)
            X_flat = X_test.reshape(N * n_win, *X_test.shape[2:])
        else:
            X_flat = X_test

        X_t = torch.as_tensor(X_flat, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_t)

        recon_np = recon.cpu().numpy()
        if trial_shape is not None:
            X_test = X_test  # already (N, n_windows, tw, D)
            recon_np = recon_np.reshape(*trial_shape, *recon_np.shape[1:])
        return X_test, recon_np


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
        self._build_model()

    def _build_model(self):
        self.model = MLPAutoencoder(
            self.n_latent,
            self.time_window,
            n_features=self.n_features,
            network_shape=self._network_shape,
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
        self._build_model()

    def _build_model(self):
        self.model = LSTMAutoencoder(
            self.n_latent,
            self.time_window,
            n_features=self.n_features,
            network_shape=self._network_shape,
        )

    def fit(self, X, y=None, **kwargs):
        if "latent_regularizer" not in kwargs and self._latent_regularizer is not None:
            kwargs["latent_regularizer"] = self._latent_regularizer
        super().fit(X, y=y, **kwargs)
