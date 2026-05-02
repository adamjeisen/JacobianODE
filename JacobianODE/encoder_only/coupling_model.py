"""Lightning modules for coupling-flow encoder training.

Contains two Lightning modules:

* :class:`LitCouplingFlow` — *supervised* training against a known target
  (e.g. [x, y, z, 0, …, 0]).  Used in the Coupling Flow test-frame notebook.
* :class:`LitUnsupervisedCouplingFlow` — *self-supervised* training.  The
  encoder maps x → z, the dynamic subspace ``z_dyn`` is zero-padded, and the
  inverse flow reconstructs ``x̂``.  No ground-truth state is required.

Both are designed to work with the shared
:func:`~JacobianODE.jacobians.training.train_model` infrastructure.
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
        A :class:`CouplingEncoder` (or any invertible module with
        ``forward``, ``inverse``, and ``n_latent`` attribute).
    n_obs : int
        Input / output dimension D.
    n_target_dims : int
        Number of meaningful dimensions in the target (e.g. 3 for Lorenz).
        The remaining ``D - n_target_dims`` dimensions are driven toward zero.
    kl_null_weight : float or None
        Weight for null-space MSE penalty (structural constraint).
        If ``None``, defaults to ``kl_dyn_weight`` (coupled single-knob control).
    kl_dyn_weight : float
        Weight for dynamic subspace KL divergence (smoothness regularizer).
    kl_divergence_weight : float or None
        DEPRECATED — if set, maps to both kl_null_weight and kl_dyn_weight.
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
        kl_null_weight: float = None,
        kl_dyn_weight: float = 0.0,
        kl_divergence_weight: float = None,  # DEPRECATED
        decoder_recon_weight: float = 0.0,
        reconstruction_mode: str = "uniform",
        # VAE reparameterization on dynamic subspace
        use_vae: bool = False,
        vae_sample_all_losses: bool = False,
        kl_warmup_epochs: int = 0,
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
        if kl_divergence_weight is not None:
            self.kl_null_weight = kl_divergence_weight
            self.kl_dyn_weight = kl_divergence_weight
        else:
            self.kl_dyn_weight = kl_dyn_weight
            self.kl_null_weight = kl_dyn_weight if kl_null_weight is None else kl_null_weight
        self.decoder_recon_weight = decoder_recon_weight

        # VAE reparameterization on dynamic subspace
        self.use_vae = use_vae
        self.vae_sample_all_losses = vae_sample_all_losses
        self.kl_warmup_epochs = kl_warmup_epochs
        if use_vae:
            self.log_var_proj = nn.Linear(n_target_dims, n_target_dims)
            nn.init.zeros_(self.log_var_proj.weight)
            nn.init.constant_(self.log_var_proj.bias, -6.0)  # sigma ≈ 0.05

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

    @staticmethod
    def _kl_divergence(mu, log_var):
        """Standard Gaussian KL divergence: KL(q(z|x) || N(0, I))."""
        return -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

    def _effective_kl_weights(self):
        """Return (effective_null_weight, effective_dyn_weight) after warmup."""
        if self.kl_warmup_epochs <= 0:
            return self.kl_null_weight, self.kl_dyn_weight
        ramp = min(self.current_epoch / self.kl_warmup_epochs, 1.0)
        return self.kl_null_weight * ramp, self.kl_dyn_weight * ramp

    def _compute_losses(
        self,
        x_input: torch.Tensor,
        target: torch.Tensor,
        prefix: str,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute target reconstruction + KL losses.

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
        mu_target = z[..., : self.n_target_dims]
        z_zero = z[..., self.n_target_dims :]
        y_target = target[..., : self.n_target_dims]

        # VAE reparameterization on dynamic portion
        kl_dyn_loss = torch.tensor(0.0, device=z.device)
        if self.use_vae:
            log_var = self.log_var_proj(mu_target)
            if self.training:
                std = torch.exp(0.5 * log_var)
                z_target = mu_target + std * torch.randn_like(std)
            else:
                z_target = mu_target
            kl_dyn_loss = self._kl_divergence(mu_target, log_var)
        else:
            z_target = mu_target

        # Target reconstruction loss (scale-invariant MSE)
        target_loss = normalized_mse(y_target, z_target)

        # Null-space penalty (MSE to zero)
        kl_null_loss = F.mse_loss(z_zero, torch.zeros_like(z_zero))

        # Separate KL weights for null and dyn
        eff_null_w, eff_dyn_w = self._effective_kl_weights()
        total_loss = target_loss
        if eff_null_w > 0:
            total_loss = total_loss + eff_null_w * kl_null_loss
        if eff_dyn_w > 0:
            total_loss = total_loss + eff_dyn_w * kl_dyn_loss

        # Decoder reconstruction loss: map the supervised target back through
        # the inverse and compare against the original input in full (B, T, D)
        # space.  The target is [x, y, z, 0, …, 0] — running the decoder on
        # it should recover the delay embedding.
        if self.decoder_recon_weight > 0:
            x_recon = self.encoder.inverse(target)
            if self.reconstruction_mode == "harmonic":
                sq_err = F.mse_loss(x_recon, x_input, reduction="none")  # (B, T, D)
                decoder_recon_loss = (sq_err * self.recon_weights).mean()
            elif self.reconstruction_mode == "most_recent":
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
        log_dict[f"{prefix}/kl_null_loss"] = kl_null_loss.detach()
        log_dict[f"{prefix}/kl_dyn_loss"] = kl_dyn_loss.detach()
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


class LitUnsupervisedCouplingFlow(L.LightningModule):
    """Self-supervised coupling-flow encoder.

    No ground-truth state is required.  The training signal comes from two
    losses:

    1. **Reconstruction loss** — encode ``x → z``, split into dynamic
       (``z_dyn``) and null (``z_null``) subspaces, zero-pad the null dims
       to form ``[z_dyn, 0…0]``, run the inverse flow → ``x̂``, and compute
       MSE(x, x̂).
    2. **KL divergence loss** — ``z_null`` is penalised toward zero (MSE).
       When ``use_vae=True``, ``z_dyn`` additionally incurs a standard
       Gaussian KL via reparameterization.

    Expects ``(tensor,)`` batches (same format as the sequence-encoder
    pipeline).

    Parameters
    ----------
    encoder : nn.Module
        Invertible encoder with ``forward`` / ``inverse`` (coupling flow).
    n_obs : int
        Input dimension D (delay-embedded observation size).
    n_target_dims : int
        Number of dynamic subspace dimensions.
    kl_null_weight : float or None
        Weight for null-space MSE penalty (structural constraint).
        If ``None``, defaults to ``kl_dyn_weight`` (coupled single-knob control).
    kl_dyn_weight : float
        Weight for dynamic subspace KL divergence (smoothness regularizer).
    kl_divergence_weight : float or None
        DEPRECATED — if set, maps to both kl_null_weight and kl_dyn_weight.
    reconstruction_mode : str
        ``'uniform'``, ``'harmonic'``, or ``'most_recent'``.
    use_vae : bool
        Enable VAE reparameterization on the dynamic subspace.
    vae_sample_all_losses : bool
        Unused (kept for config compatibility with LitCouplingFlow).
    kl_warmup_epochs : int
        Linear KL weight ramp length; 0 = fixed weight.
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_obs: int,
        n_target_dims: int = 3,
        kl_null_weight: float = None,
        kl_dyn_weight: float = 0.0,
        kl_divergence_weight: float = None,  # DEPRECATED
        reconstruction_mode: str = "uniform",
        # VAE
        use_vae: bool = False,
        vae_sample_all_losses: bool = False,
        kl_warmup_epochs: int = 0,
        # Optimizer
        optimizer: str = "AdamW",
        optimizer_kwargs: Optional[dict] = None,
        use_scheduler: bool = True,
        min_lr: float = 1e-6,
        lr_scheduler_patience: int = 10,
        lr_scheduler_factor: float = 0.5,
        # Gradient clipping
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_obs = n_obs
        self.n_target_dims = n_target_dims
        if kl_divergence_weight is not None:
            self.kl_null_weight = kl_divergence_weight
            self.kl_dyn_weight = kl_divergence_weight
        else:
            self.kl_dyn_weight = kl_dyn_weight
            self.kl_null_weight = kl_dyn_weight if kl_null_weight is None else kl_null_weight

        # VAE
        self.use_vae = use_vae
        self.kl_warmup_epochs = kl_warmup_epochs
        if use_vae:
            self.log_var_proj = nn.Linear(n_target_dims, n_target_dims)
            nn.init.zeros_(self.log_var_proj.weight)
            nn.init.constant_(self.log_var_proj.bias, -6.0)  # sigma ≈ 0.05

        # Reconstruction mode
        if reconstruction_mode not in ("uniform", "harmonic", "most_recent"):
            raise ValueError(
                f"reconstruction_mode must be 'uniform', 'harmonic', or "
                f"'most_recent', got '{reconstruction_mode}'"
            )
        self.reconstruction_mode = reconstruction_mode

        if reconstruction_mode == "harmonic":
            w_raw = 1.0 / (torch.arange(n_obs, dtype=torch.float32) + 1.0)
            w_norm = w_raw * (n_obs / w_raw.sum())
            self.register_buffer("recon_weights", w_norm)

        # Optimizer
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4, "weight_decay": 1e-4}
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

        # Gradient clipping
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Validation tracking
        self.validation_losses: list[float] = []
        self.percent_improvements: list[float] = []
        self.current_epoch_val_losses: list[float] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_divergence(mu, log_var):
        """Standard Gaussian KL divergence: KL(q(z|x) || N(0, I))."""
        return -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

    def _effective_kl_weights(self):
        """Return (effective_null_weight, effective_dyn_weight) after warmup."""
        if self.kl_warmup_epochs <= 0:
            return self.kl_null_weight, self.kl_dyn_weight
        ramp = min(self.current_epoch / self.kl_warmup_epochs, 1.0)
        return self.kl_null_weight * ramp, self.kl_dyn_weight * ramp

    def _reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss respecting ``reconstruction_mode``."""
        if self.reconstruction_mode == "harmonic":
            sq_err = F.mse_loss(x_hat, x, reduction="none")  # (B, T, D)
            return (sq_err * self.recon_weights).mean()
        elif self.reconstruction_mode == "most_recent":
            return F.mse_loss(x_hat[..., 0], x[..., 0])
        else:  # uniform
            return F.mse_loss(x_hat, x)

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        x: torch.Tensor,
        prefix: str,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Self-supervised losses: reconstruction + KL.

        Parameters
        ----------
        x : Tensor (B, T, D)
            Delay-embedded input.
        prefix : str
            ``'train'`` or ``'val'``.
        """
        z = self.encoder(x)  # (B, T, D)

        z_dyn = z[..., : self.n_target_dims]
        z_null = z[..., self.n_target_dims :]

        # VAE reparameterization on dynamic portion
        kl_dyn_loss = torch.tensor(0.0, device=z.device)
        if self.use_vae:
            log_var = self.log_var_proj(z_dyn)
            if self.training:
                std = torch.exp(0.5 * log_var)
                z_dyn_recon = z_dyn + std * torch.randn_like(std)
            else:
                z_dyn_recon = z_dyn
            kl_dyn_loss = self._kl_divergence(z_dyn, log_var)
        else:
            z_dyn_recon = z_dyn

        # Zero-pad and decode
        z_padded = torch.cat(
            [z_dyn_recon, torch.zeros_like(z_null)], dim=-1
        )
        x_hat = self.encoder.inverse(z_padded)

        # Losses
        recon_loss = self._reconstruction_loss(x, x_hat)

        kl_null_loss = F.mse_loss(z_null, torch.zeros_like(z_null))

        eff_null_w, eff_dyn_w = self._effective_kl_weights()
        total_loss = recon_loss
        if eff_null_w > 0:
            total_loss = total_loss + eff_null_w * kl_null_loss
        if eff_dyn_w > 0:
            total_loss = total_loss + eff_dyn_w * kl_dyn_loss

        # Metrics
        log_dict: Dict[str, torch.Tensor] = {
            f"{prefix}/recon_loss": recon_loss.detach(),
            f"{prefix}/kl_null_loss": kl_null_loss.detach(),
            f"{prefix}/kl_dyn_loss": kl_dyn_loss.detach() if isinstance(kl_dyn_loss, torch.Tensor) else kl_dyn_loss,
            f"{prefix}/total_loss": total_loss.detach(),
        }

        with torch.no_grad():
            # Inverse consistency: full z → inverse → compare to x
            x_roundtrip = self.encoder.inverse(z)
            log_dict[f"{prefix}/inverse_consistency_mse"] = F.mse_loss(
                x_roundtrip, x
            )

        return total_loss, log_dict

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss, log_dict = self._compute_losses(x, "train")
        if torch.isnan(loss) or torch.isinf(loss):
            # Spline inverse can hit numerical singularities with zero-padded
            # inputs; skip the batch rather than corrupting the optimiser state.
            return None
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss, log_dict = self._compute_losses(x, "val")
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Early stopping / checkpointing / sweep selection use recon loss only
        recon_loss = log_dict["val/recon_loss"]
        self.log("mean val loss", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("trajectory val_loss", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.current_epoch_val_losses.append(recon_loss.item())
        return loss

    def on_validation_epoch_end(self):
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
