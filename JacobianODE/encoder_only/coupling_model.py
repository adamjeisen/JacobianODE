"""Lightning module for supervised coupling-flow training.

Trains an :class:`~JacobianODE.fnn.coupling_flows.AffineCouplingEncoder` to map
delay-embedded observations to a supervised target (e.g. [x, y, z, 0, …, 0]).
The exact analytical inverse of the coupling stack serves as the decoder — no
separate decoder network is needed.

Designed to work with the shared :func:`~JacobianODE.jacobians.training.train_model`
infrastructure (W&B, early stopping, checkpointing).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from ..jacobians.metrics import normalized_mse, r2_score


class LitCouplingFlow(L.LightningModule):
    """Lightning module for supervised coupling-flow training.

    Expects paired ``(input, target)`` batches from the DataLoader.

    Parameters
    ----------
    encoder : nn.Module
        An :class:`AffineCouplingEncoder` (or any invertible module with
        ``forward``, ``inverse``, and ``n_latent`` attribute).
    n_obs : int
        Input / output dimension D.
    n_target_dims : int
        Number of meaningful dimensions in the target (e.g. 3 for Lorenz).
        The remaining ``D - n_target_dims`` dimensions are driven toward zero.
    zero_penalty_weight : float
        Relative weight for the zero-padding MSE loss.
    decoder_recon_weight : float
        Weight for the decoder reconstruction loss.  Maps the supervised
        target back through the inverse and computes MSE against the original
        input.  How dimensions are weighted depends on ``reconstruction_mode``.
    reconstruction_mode : str
        Controls which delay dimensions participate in the decoder
        reconstruction loss:

        - ``'uniform'``: standard unweighted MSE across all D dimensions.
        - ``'harmonic'``: weighted MSE with normalised harmonic mask
          ``1/(k+1)`` so recent delays contribute more.  Mean of the mask
          is 1.0, preserving gradient scale.
        - ``'most_recent'``: MSE only on the single most-recent delay
          dimension (index 0, per ``embed_signal_torch`` ordering).
    optimizer : str
        ``'AdamW'`` or ``'Adam'``.
    optimizer_kwargs : dict, optional
        Keyword arguments for the optimizer (e.g. ``{'lr': 1e-4}``).
    use_scheduler : bool
        Whether to use ReduceLROnPlateau.
    min_lr : float
        Minimum learning rate for the scheduler.
    lr_scheduler_patience : int
        Patience (epochs) before reducing LR.
    lr_scheduler_factor : float
        Factor by which to reduce the LR.
    gradient_clip_val : float
        Gradient clipping value (used by ``train_model``).
    gradient_clip_algorithm : str
        Gradient clipping algorithm (``'norm'`` or ``'value'``).
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_obs: int,
        n_target_dims: int = 3,
        zero_penalty_weight: float = 1.0,
        decoder_recon_weight: float = 0.0,
        reconstruction_mode: str = "uniform",
        # Optimizer
        optimizer: str = "AdamW",
        optimizer_kwargs: Optional[dict] = None,
        use_scheduler: bool = True,
        min_lr: float = 1e-6,
        lr_scheduler_patience: int = 10,
        lr_scheduler_factor: float = 0.5,
        # Gradient clipping (read by train_model)
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        **kwargs,  # absorb extra Hydra keys gracefully
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_obs = n_obs
        self.n_target_dims = n_target_dims
        self.zero_penalty_weight = zero_penalty_weight
        self.decoder_recon_weight = decoder_recon_weight

        if reconstruction_mode not in ("uniform", "harmonic", "most_recent"):
            raise ValueError(
                f"reconstruction_mode must be 'uniform', 'harmonic', or "
                f"'most_recent', got '{reconstruction_mode}'"
            )
        self.reconstruction_mode = reconstruction_mode

        # Harmonic temporal prior: w_k = 1/(k+1), normalised so mean(w) = 1.0.
        if reconstruction_mode == "harmonic":
            w_raw = 1.0 / (torch.arange(n_obs, dtype=torch.float32) + 1.0)
            w_norm = w_raw * (n_obs / w_raw.sum())
            self.register_buffer("recon_weights", w_norm)

        # Optimizer config
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4, "weight_decay": 1e-4}
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

        # Gradient clipping (exposed for train_model compatibility)
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Validation tracking (percent_improvement, matching LitEncoderDecoder)
        self.validation_losses: list[float] = []
        self.percent_improvements: list[float] = []
        self.current_epoch_val_losses: list[float] = []

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        x_input: torch.Tensor,
        target: torch.Tensor,
        prefix: str,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute target reconstruction + zero-padding losses.

        Parameters
        ----------
        x_input : Tensor (B, T, D)
            Delay-embedded input sequences.
        target : Tensor (B, T, D)
            Supervised target: [x, y, z, 0, …, 0].
        prefix : str
            ``'train'`` or ``'val'`` for metric key namespacing.

        Returns
        -------
        total_loss : Tensor (scalar)
        log_dict : dict[str, Tensor]
        """
        z = self.encoder(x_input)  # (B, T, D)

        # Split into target dims and zero-padding dims
        z_target = z[..., : self.n_target_dims]
        z_zero = z[..., self.n_target_dims :]
        y_target = target[..., : self.n_target_dims]

        # Target reconstruction loss (scale-invariant MSE)
        target_loss = normalized_mse(y_target, z_target)

        # Zero-padding loss
        zero_loss = F.mse_loss(z_zero, torch.zeros_like(z_zero))

        total_loss = target_loss + self.zero_penalty_weight * zero_loss

        # Decoder reconstruction loss: map the supervised target back through
        # the inverse and compare against the original input in full (B, T, D)
        # space.  The target is [x, y, z, 0, …, 0] — running the decoder on
        # it should recover the delay embedding.
        if self.decoder_recon_weight > 0:
            x_recon = self.encoder.inverse(target)
            if self.reconstruction_mode == "harmonic":
                # Weighted MSE: per-element squared error weighted by harmonic
                # prior, then averaged.  recon_weights broadcasts over (B, T).
                sq_err = F.mse_loss(x_recon, x_input, reduction="none")  # (B, T, D)
                decoder_recon_loss = (sq_err * self.recon_weights).mean()
            elif self.reconstruction_mode == "most_recent":
                # MSE only on the most-recent delay (index 0).
                decoder_recon_loss = F.mse_loss(
                    x_recon[..., 0], x_input[..., 0]
                )
            else:  # uniform
                decoder_recon_loss = normalized_mse(x_input, x_recon)
            total_loss = total_loss + self.decoder_recon_weight * decoder_recon_loss
        else:
            decoder_recon_loss = torch.tensor(0.0, device=z.device)

        # Metrics
        log_dict: Dict[str, torch.Tensor] = {}
        log_dict[f"{prefix}/target_loss"] = target_loss.detach()
        log_dict[f"{prefix}/zero_loss"] = zero_loss.detach()
        log_dict[f"{prefix}/decoder_recon_loss"] = decoder_recon_loss.detach()
        log_dict[f"{prefix}/total_loss"] = total_loss.detach()

        # Per-dimension R2
        with torch.no_grad():
            r2 = r2_score(y_target, z_target)
            log_dict[f"{prefix}/target_r2"] = r2

            # Inverse consistency diagnostic (sampled for speed)
            x_rec = self.encoder.inverse(z)
            inv_err = F.mse_loss(x_rec, x_input)
            log_dict[f"{prefix}/inverse_consistency_mse"] = inv_err

        return total_loss, log_dict

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x_input, target = batch
        loss, log_dict = self._compute_losses(x_input, target, "train")
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_input, target = batch
        loss, log_dict = self._compute_losses(x_input, target, "val")
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Aliases expected by the shared train_model / trainer utilities
        self.log("mean val loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("trajectory val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.current_epoch_val_losses.append(loss.detach().item())
        return loss

    def on_validation_epoch_end(self):
        """Track validation losses and log percent_improvement."""
        if self.current_epoch_val_losses:
            mean_val_loss = sum(self.current_epoch_val_losses) / len(
                self.current_epoch_val_losses
            )
            self.validation_losses.append(mean_val_loss)
            if len(self.validation_losses) > 1:
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
                else:
                    self.percent_improvements.append(0.0)
                if self.current_epoch > 0:
                    self.log(
                        "percent_improvement",
                        self.percent_improvements[-1],
                        on_epoch=True,
                        sync_dist=True,
                    )
            self.current_epoch_val_losses.clear()

    def configure_optimizers(self):
        if self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)
        else:
            optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)

        if not self.use_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
