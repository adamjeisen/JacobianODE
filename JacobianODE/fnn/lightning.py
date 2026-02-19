"""
PyTorch Lightning module for FNN autoencoder training.

Follows the JacobianODE training patterns: Lightning module + W&B + Hydra.

Original TensorFlow implementation: https://github.com/williamgilpin/fnn
Reference: Gilpin, "Deep reconstruction of strange attractors from time series"
           NeurIPS 2020. https://arxiv.org/abs/2002.05909
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from JacobianODE.jacobians.lightning_base import PercentEarlyStopping
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from .utils import hankel_matrix, standardize_ts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HankelDataset(Dataset):
    """Dataset of Hankel-windowed time series for autoencoder training.

    Parameters
    ----------
    X : np.ndarray
        Input windows, shape (n_samples, time_window, n_features).
    Y : np.ndarray
        Target windows, shape (n_samples, time_window, n_features).
        When tau=0, this is the same as X (pure autoencoder).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def prepare_data(
    time_series: np.ndarray,
    time_window: int,
    tau: int = 0,
    subsample: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare Hankel-windowed data from a raw time series.

    Parameters
    ----------
    time_series : np.ndarray
        (T,), (T, D), or (N, T, D) time series.
    time_window : int
        Window length for the Hankel matrix.
    tau : int
        Prediction horizon. 0 = pure autoencoder.
    subsample : int, optional
        If set, randomly sample this many windows.
    random_state : int, optional
        Random seed for subsampling.

    Returns
    -------
    X, Y : np.ndarray
        Input and target windows, each (n_samples, time_window, n_features).
        For 3D input, trials are flattened into the batch dimension.
    """
    Xs = standardize_ts(time_series)
    X0 = hankel_matrix(Xs, time_window + tau)

    # 4D from 3D input: (N, n_windows, tw+tau, D) -> flatten trials into batch
    if X0.ndim == 4:
        X = X0[:, :, :time_window]
        Y = X0[:, :, -time_window:]
        N, n_win = X.shape[:2]
        X = X.reshape(N * n_win, *X.shape[2:])
        Y = Y.reshape(N * n_win, *Y.shape[2:])
    else:
        X = X0[:, :time_window]
        Y = X0[:, -time_window:]

    if subsample is not None and subsample < len(X):
        np.random.seed(random_state)
        indices = np.random.choice(len(X), subsample, replace=False)
        X = X[indices]
        Y = Y[indices]

    return X, Y


def create_dataloaders(
    time_series: np.ndarray,
    time_window: int,
    tau: int = 0,
    batch_size: int = 100,
    train_split: float = 0.8,
    random_state: Optional[int] = None,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders from a raw time series.

    Parameters
    ----------
    time_series : np.ndarray
        (T,) or (T, D) raw time series.
    time_window : int
        Window length for the Hankel matrix.
    tau : int
        Prediction horizon.
    batch_size : int
        Batch size.
    train_split : float
        Fraction of data for training.
    random_state : int, optional
        Random seed.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    X, Y = prepare_data(time_series, time_window, tau=tau, random_state=random_state)

    n_train = int(len(X) * train_split)
    train_dataset = HankelDataset(X[:n_train], Y[:n_train])
    val_dataset = HankelDataset(X[n_train:], Y[n_train:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class LitFNNAutoencoder(L.LightningModule):
    """PyTorch Lightning module for FNN autoencoder training.

    Wraps an autoencoder (MLPAutoencoder or LSTMAutoencoder) and handles:
    - Reconstruction loss (MSE)
    - Latent-space regularization (FNN or DeCov)
    - Optimizer configuration
    - Metric logging

    Parameters
    ----------
    model : nn.Module
        Autoencoder with `encode()`, `decode()`, and `forward()` methods.
    latent_regularizer : nn.Module, optional
        Regularizer applied to latent activations (e.g., FNN or DeCov).
    optimizer : str
        Optimizer name: 'Adam', 'AdamW', or 'NAdam'.
    optimizer_kwargs : dict
        Keyword arguments for the optimizer (e.g., {'lr': 1e-3}).
    tau : int
        Prediction horizon offset. 0 = pure autoencoder.
    gradient_clip_val : float
        Gradient clipping value.
    gradient_clip_algorithm : str
        Gradient clipping algorithm ('norm' or 'value').
    log_interval : int
        Log metrics every N batches.
    """

    def __init__(
        self,
        model,
        latent_regularizer=None,
        optimizer: str = "Adam",
        optimizer_kwargs: Optional[Dict] = None,
        tau: int = 0,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        log_interval: int = 1,
    ):
        super().__init__()
        self.model = model
        self.latent_regularizer = latent_regularizer
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.tau = tau
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.log_interval = log_interval

    def forward(self, x):
        return self.model(x)

    def encode(self, x):
        return self.model.encode(x)

    def count_parameters(self):
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _compute_loss(self, batch):
        x, y = batch
        recon = self.model(x)
        recon_loss = F.mse_loss(recon, y)

        reg_loss = torch.tensor(0.0, device=recon.device)
        if self.latent_regularizer is not None:
            latent = self.model.encode(x)
            reg_loss = self.latent_regularizer(latent)

        total_loss = recon_loss + reg_loss
        return total_loss, recon_loss, reg_loss

    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, reg_loss = self._compute_loss(batch)

        if batch_idx % self.log_interval == 0:
            self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True)
            if self.latent_regularizer is not None:
                self.log("train_reg_loss", reg_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, reg_loss = self._compute_loss(batch)

        self.log("val_loss", total_loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        if self.latent_regularizer is not None:
            self.log("val_reg_loss", reg_loss, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "NAdam": torch.optim.NAdam,
            "SGD": torch.optim.SGD,
        }
        opt_cls = optimizers.get(self.optimizer_name, torch.optim.Adam)
        optimizer = opt_cls(self.parameters(), **self.optimizer_kwargs)
        return {"optimizer": optimizer}


# ---------------------------------------------------------------------------
# Training utilities (following JacobianODE patterns)
# ---------------------------------------------------------------------------

def make_fnn_model(cfg: DictConfig) -> LitFNNAutoencoder:
    """Create an FNN autoencoder model from Hydra config.

    Instantiates the autoencoder network and wraps it in a Lightning module,
    following the JacobianODE pattern of Hydra instantiate().

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with `model.params` and `training.lightning` sections.

    Returns
    -------
    LitFNNAutoencoder
    """
    # Instantiate the autoencoder network
    ae_model = instantiate(cfg.model.params)

    # Instantiate the regularizer if configured
    latent_regularizer = None
    if "regularizer" in cfg.training and cfg.training.regularizer is not None:
        latent_regularizer = instantiate(cfg.training.regularizer)

    # Instantiate the Lightning module
    lit_model = instantiate(
        cfg.training.lightning,
        model=ae_model,
        latent_regularizer=latent_regularizer,
    )

    return lit_model


def setup_fnn_wandb(
    cfg: DictConfig,
) -> Tuple[str, str, Optional[str]]:
    """Set up W&B logging for FNN training.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config.

    Returns
    -------
    name, project, entity : str, str, Optional[str]
    """
    model_cls = cfg.model.params._target_.split(".")[-1]
    name_parts = [model_cls]
    name_parts.append(f"n_latent_{cfg.model.params.n_latent}")
    name_parts.append(f"tw_{cfg.model.params.time_window}")

    if "network_shape" in cfg.model.params and cfg.model.params.network_shape:
        name_parts.append(f"shape_{cfg.model.params.network_shape}")

    if "regularizer" in cfg.training and cfg.training.regularizer is not None:
        reg_cls = cfg.training.regularizer._target_.split(".")[-1]
        name_parts.append(f"reg_{reg_cls}")
        if "strength" in cfg.training.regularizer:
            name_parts.append(f"str_{cfg.training.regularizer.strength}")

    lr = cfg.training.lightning.optimizer_kwargs.get("lr", "?")
    name_parts.append(f"lr_{lr}")

    name = "__".join(str(p) for p in name_parts)
    project = "FNN_Embedding"

    entity = os.environ.get("WANDB_ENTITY", None)

    return name, project, entity


def train_fnn_model(
    cfg: DictConfig,
    lit_model: LitFNNAutoencoder,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    name: str,
    project: str,
    entity: Optional[str] = None,
) -> None:
    """Train an FNN autoencoder using PyTorch Lightning.

    Follows the JacobianODE training pattern: callbacks, W&B logger,
    gradient clipping, DDP strategy.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with `training` section.
    lit_model : LitFNNAutoencoder
        Lightning model to train.
    train_dataloader : DataLoader
        Training data.
    val_dataloader : DataLoader
        Validation data.
    name : str
        W&B run name.
    project : str
        W&B project name.
    entity : str, optional
        W&B entity/team.
    """
    # Set up W&B logger
    logger_kwargs = {"name": name, "project": project}
    if entity is not None:
        logger_kwargs["entity"] = entity

    experiment_logger = instantiate(cfg.training.logger, **logger_kwargs)

    if os.getenv("LOCAL_RANK") == "0" or os.getenv("LOCAL_RANK") is None:
        experiment_logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True)
        )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.early_stopping.monitor,
        save_top_k=1,
        mode=cfg.training.early_stopping.mode,
    )

    es_mode = cfg.training.early_stopping.get("early_stopping_mode", "percent_thresh")
    if es_mode == "percent_thresh":
        early_stopping_callback = PercentEarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            percent_thresh=cfg.training.early_stopping.get("percent_thresh", 0.01),
        )
    else:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
        )

    # DDP strategy
    try:
        from JacobianODE.jacobians.core.types import in_ipython
        strategy = "ddp_notebook" if in_ipython() else "ddp"
    except ImportError:
        strategy = "auto"

    # Trainer
    trainer = L.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=experiment_logger,
        log_every_n_steps=10,
        gradient_clip_val=lit_model.gradient_clip_val,
        gradient_clip_algorithm=lit_model.gradient_clip_algorithm,
        **cfg.training.trainer_params,
        devices="auto",
        strategy=strategy,
    )

    logger.info(f"Starting FNN training run: {name}")
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    wandb.finish()
    logger.info(f"FNN training complete for run: {name}")
