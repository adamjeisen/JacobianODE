"""
Sklearn-style API for sequence-to-next-step prediction models.

These models bypass the delay-embedding step entirely: they accept a raw
(T, D) or (N, T, D) time series, window it into overlapping subsequences,
and train a sequence encoder to predict the *next* time step from each
window.  FNN / DeCov regularisation is applied to the N-dimensional latent.

Example
-------
>>> from JacobianODE.fnn.sequence_models import TransformerEmbedding
>>> from JacobianODE.fnn import FNN
>>> model = TransformerEmbedding(
...     n_latent=6, time_window=64, n_features=1,
...     use_positional_encoding=True,
...     latent_regularizer=FNN(0.01),
... )
>>> model.fit(x_train, train_steps=300, verbose=1)
>>> latent = model.transform(x_test)         # (n_windows, 6)
>>> pred   = model.predict_next(x_test)       # (n_windows, 1)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .sequence_networks import (
    build_transformer,
    build_ssm,
    build_tcn,
    build_tcn_spatial,
    SequenceAutoencoder,
)
from .utils import standardize_ts


# ---------------------------------------------------------------------------
# Windowing utility
# ---------------------------------------------------------------------------

def sliding_windows(data: np.ndarray, window: int):
    """Create sliding windows with next-step targets.

    Parameters
    ----------
    data : np.ndarray
        (T, D) or (N, T, D) time series.

    Returns
    -------
    X : np.ndarray
        (n_windows, window, D) input windows.
    Y : np.ndarray
        (n_windows, D) next-step targets.
    """
    if data.ndim == 1:
        data = data[:, None]

    if data.ndim == 3:
        # (N, T, D) -> stack all trials
        all_X, all_Y = [], []
        for trial in data:
            x, y = sliding_windows(trial, window)
            all_X.append(x)
            all_Y.append(y)
        return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0)

    T, D = data.shape
    n_windows = T - window
    X = np.stack([data[i : i + window] for i in range(n_windows)], axis=0)
    Y = data[window:]  # (n_windows, D)
    return X, Y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SequenceEmbedding:
    """Base class for sequence-to-next-step embeddings.

    Parameters
    ----------
    n_latent : int
        Latent / embedding dimension (N).
    time_window : int
        Number of time steps per input window.
    n_features : int
        Number of channels (D) in the time series.
    random_state : int, optional
        Random seed.
    latent_regularizer : nn.Module, optional
        FNN or DeCov regularizer.
    """

    def __init__(
        self,
        n_latent: int,
        time_window: int = 64,
        n_features: int = 1,
        random_state: Optional[int] = None,
        latent_regularizer: Optional[nn.Module] = None,
        **kwargs,
    ):
        self.n_latent = n_latent
        self.time_window = time_window
        self.n_features = n_features
        self.random_state = random_state
        self.latent_regularizer = latent_regularizer
        self.model: Optional[SequenceAutoencoder] = None
        self.train_history: Optional[dict] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None

    def _build_model(self) -> SequenceAutoencoder:
        raise NotImplementedError

    def count_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            if X.ndim == 3:
                self._train_mean = np.mean(X, axis=(0, 1), keepdims=False)  # (D,)
                self._train_std = np.std(X, axis=(0, 1), keepdims=False)    # (D,)
            else:
                self._train_mean = np.mean(X, axis=0, keepdims=False)  # (D,)
                self._train_std = np.std(X, axis=0, keepdims=False)    # (D,)
            self._train_std[self._train_std == 0] = 1.0
        if self._train_mean is not None:
            # Broadcasting works for any input shape: (T, D), (N, T, D), etc.
            return (X - self._train_mean) / self._train_std
        return standardize_ts(X)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y=None,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        train_steps: int = 200,
        verbose: int = 0,
        optimizer: str = "adam",
        weight_decay: float = 1e-5,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        percent_thresh: float = 0.005,
    ):
        """Train the model on a time series.

        Parameters
        ----------
        X : np.ndarray
            (T,), (T, D), or (N, T, D).
        learning_rate : float
            Learning rate.
        batch_size : int
            Batch size.
        train_steps : int
            Max number of training epochs.
        verbose : int
            0=silent, 1=progress, 2=detailed.
        optimizer : str
            ``"adam"`` or ``"adamw"``.
        weight_decay : float
            L2 regularization.
        early_stopping : bool
            Enable early stopping.
        early_stopping_patience : int
            Epochs without improvement before stopping.
        percent_thresh : float
            Minimum fractional improvement per epoch.
        """
        # Auto-detect features
        n_features = 1 if X.ndim == 1 else X.shape[-1]
        if n_features != self.n_features:
            self.n_features = n_features

        self.model = self._build_model()

        Xs = self._standardize(X, fit=True)
        X_win, Y_next = sliding_windows(Xs, self.time_window)

        X_t = torch.as_tensor(X_win, dtype=torch.float32, device=self.device)
        Y_t = torch.as_tensor(Y_next, dtype=torch.float32, device=self.device)

        self.model = self.model.to(self.device)

        if optimizer == "adamw":
            opt = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            opt = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_steps)

        self.model.train()
        history = {"loss": [], "pred_loss": [], "reg_loss": []}
        best_loss = float("inf")
        patience_counter = 0

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        n_samples = len(X_t)
        for epoch in range(train_steps):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            epoch_pred = 0.0
            epoch_reg = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                x_batch = X_t[idx]
                y_batch = Y_t[idx]

                opt.zero_grad()

                pred = self.model(x_batch)  # (B, D)
                pred_loss = torch.nn.functional.mse_loss(pred, y_batch)

                reg_loss = torch.tensor(0.0, device=self.device)
                if self.latent_regularizer is not None:
                    latent = self.model.encode(x_batch)
                    reg_loss = self.latent_regularizer(latent)

                total_loss = pred_loss + reg_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                epoch_loss += total_loss.item()
                epoch_pred += pred_loss.item()
                epoch_reg += reg_loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_pred = epoch_pred / max(n_batches, 1)
            avg_reg = epoch_reg / max(n_batches, 1)
            history["loss"].append(avg_loss)
            history["pred_loss"].append(avg_pred)
            history["reg_loss"].append(avg_reg)

            if verbose >= 1 and (
                epoch % max(1, train_steps // 20) == 0 or epoch == train_steps - 1
            ):
                msg = f"Epoch {epoch+1}/{train_steps} - loss: {avg_loss:.6f}  pred: {avg_pred:.6f}"
                if self.latent_regularizer is not None:
                    msg += f"  reg: {avg_reg:.6f}"
                print(msg)

            if early_stopping:
                if best_loss < float("inf") and best_loss > 0:
                    improvement = (best_loss - avg_loss) / best_loss
                    if improvement < percent_thresh:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                if avg_loss < best_loss:
                    best_loss = avg_loss
                if patience_counter >= early_stopping_patience:
                    if verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        self.train_history = history
        self.model.eval()

    # ------------------------------------------------------------------
    # transform / predict
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed a time series into the latent space.

        Returns
        -------
        np.ndarray
            (n_windows, N) latent embedding.
        """
        Xs = self._standardize(X)
        X_win, _ = sliding_windows(Xs, self.time_window)
        X_t = torch.as_tensor(X_win, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(X_t)
        return latent.cpu().numpy()

    def predict_next(self, X: np.ndarray) -> np.ndarray:
        """Predict the next step for each window.

        Returns
        -------
        np.ndarray
            (n_windows, D) next-step predictions.
        """
        Xs = self._standardize(X)
        X_win, _ = sliding_windows(Xs, self.time_window)
        X_t = torch.as_tensor(X_win, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t)
        return preds.cpu().numpy()

    def predict_trajectory(
        self, X: np.ndarray, n_steps: int = 100
    ) -> np.ndarray:
        """Autoregressively predict a trajectory.

        Uses the last ``time_window`` steps of X as the initial seed,
        then rolls forward by appending predictions.

        Parameters
        ----------
        X : np.ndarray
            (T, D) time series seed.
        n_steps : int
            Number of steps to predict forward.

        Returns
        -------
        np.ndarray
            (n_steps, D) predicted trajectory.
        """
        Xs = self._standardize(X)
        if Xs.ndim == 1:
            Xs = Xs[:, None]
        # Take the last time_window points as seed
        window = Xs[-self.time_window :].copy()  # (tw, D)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for _ in range(n_steps):
                x_t = torch.as_tensor(
                    window[None], dtype=torch.float32, device=self.device
                )
                next_step = self.model(x_t).cpu().numpy()[0]  # (D,)
                preds.append(next_step)
                window = np.concatenate(
                    [window[1:], next_step[None]], axis=0
                )

        preds = np.array(preds)
        # Un-standardize
        if self._train_mean is not None:
            mean = self._train_mean.squeeze()
            std = self._train_std.squeeze()
            preds = preds * std + mean
        return preds


# ---------------------------------------------------------------------------
# Concrete embedding classes
# ---------------------------------------------------------------------------

class TransformerEmbedding(SequenceEmbedding):
    """Transformer-based sequence embedding.

    Parameters
    ----------
    use_positional_encoding : bool
        Whether to add temporal positional embeddings.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        FFN hidden dim inside Transformer layers.
    dropout : float
        Dropout rate.
    decoder_hidden : int
        Decoder MLP hidden dim.
    decoder_layers : int
        Number of decoder MLP layers.
    """

    def __init__(
        self,
        *args,
        use_positional_encoding: bool = True,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        decoder_layers: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._encoder_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self._use_pe = use_positional_encoding
        self._decoder_hidden = decoder_hidden
        self._decoder_layers = decoder_layers

    def _build_model(self) -> SequenceAutoencoder:
        return build_transformer(
            n_input=self.n_features,
            n_latent=self.n_latent,
            use_positional_encoding=self._use_pe,
            decoder_hidden=self._decoder_hidden,
            decoder_layers=self._decoder_layers,
            **self._encoder_kwargs,
        )


class SSMEmbedding(SequenceEmbedding):
    """SSM-based (diagonal S4) sequence embedding.

    Parameters
    ----------
    use_positional_encoding : bool
        Whether to add temporal positional embeddings.
    d_model : int
        SSM hidden dimension.
    d_state : int
        SSM state dimension per channel.
    n_layers : int
        Number of SSM layers.
    dropout : float
        Dropout rate.
    decoder_hidden : int
        Decoder MLP hidden dim.
    decoder_layers : int
        Number of decoder MLP layers.
    """

    def __init__(
        self,
        *args,
        use_positional_encoding: bool = True,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        decoder_layers: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._encoder_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
        )
        self._use_pe = use_positional_encoding
        self._decoder_hidden = decoder_hidden
        self._decoder_layers = decoder_layers

    def _build_model(self) -> SequenceAutoencoder:
        return build_ssm(
            n_input=self.n_features,
            n_latent=self.n_latent,
            use_positional_encoding=self._use_pe,
            decoder_hidden=self._decoder_hidden,
            decoder_layers=self._decoder_layers,
            **self._encoder_kwargs,
        )


class TCNEmbedding(SequenceEmbedding):
    """TCN-based sequence embedding.

    Parameters
    ----------
    n_channels : int
        Number of channels in TCN blocks.
    kernel_size : int
        Convolution kernel size.
    n_layers : int
        Number of TCN blocks.
    dropout : float
        Dropout rate.
    decoder_hidden : int
        Decoder MLP hidden dim.
    decoder_layers : int
        Number of decoder MLP layers.
    """

    def __init__(
        self,
        *args,
        n_channels: int = 64,
        kernel_size: int = 7,
        n_layers: int = 4,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        decoder_layers: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._encoder_kwargs = dict(
            n_channels=n_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            dropout=dropout,
        )
        self._decoder_hidden = decoder_hidden
        self._decoder_layers = decoder_layers

    def _build_model(self) -> SequenceAutoencoder:
        return build_tcn(
            n_input=self.n_features,
            n_latent=self.n_latent,
            decoder_hidden=self._decoder_hidden,
            decoder_layers=self._decoder_layers,
            **self._encoder_kwargs,
        )


class TCNSpatialEmbedding(SequenceEmbedding):
    """TCN + Spatial convolution sequence embedding.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    kernel_size_temporal : int
        Kernel size for temporal convolutions.
    kernel_size_spatial : int
        Kernel size for spatial convolutions.
    n_layers : int
        Number of interleaved blocks.
    dropout : float
        Dropout rate.
    decoder_hidden : int
        Decoder MLP hidden dim.
    decoder_layers : int
        Number of decoder MLP layers.
    """

    def __init__(
        self,
        *args,
        n_channels: int = 64,
        kernel_size_temporal: int = 7,
        kernel_size_spatial: int = 3,
        n_layers: int = 4,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        decoder_layers: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._encoder_kwargs = dict(
            n_channels=n_channels,
            kernel_size_temporal=kernel_size_temporal,
            kernel_size_spatial=kernel_size_spatial,
            n_layers=n_layers,
            dropout=dropout,
        )
        self._decoder_hidden = decoder_hidden
        self._decoder_layers = decoder_layers

    def _build_model(self) -> SequenceAutoencoder:
        return build_tcn_spatial(
            n_input=self.n_features,
            n_latent=self.n_latent,
            decoder_hidden=self._decoder_hidden,
            decoder_layers=self._decoder_layers,
            **self._encoder_kwargs,
        )
