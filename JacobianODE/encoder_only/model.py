"""Encoder-only representation learning with sequence models.

Trains a causal sequence encoder (Transformer / SSM / TCN) using
same-state reconstruction and/or next-state prediction as self-supervised
objectives, plus optional geometric regularisation losses.

Architecture
------------
x : (B, T, D_obs)
    ↓  encoder  [TransformerSequenceEncoder | SSMSequenceEncoder | TCNSequenceEncoder]
z : (B, T, N_LATENT)
    ↓  same_state_decoder  (optional MLP)           ↓  next_state_decoder  (optional MLP)
x̂ : (B, T, D_obs)                  x̂_{t+1..t+k} : (B, T-k, k * D_obs_pred)

When k_steps_ahead > 1, the next-state decoder predicts all k future steps jointly
from z_t: [x_{t+1}, ..., x_{t+k}].  D_obs_pred (n_obs_pred) is the raw observation
dimension for the targets, which may differ from D_obs when delay embedding is used.

Regularisation losses applied to the latent sequence z:
  - FNN   : false-nearest-neighbour (Gilpin NeurIPS 2020)
  - Ampli.: noise amplification in embedding space
  - DeCov : off-diagonal covariance penalty (Cogswell ICLR 2016)
  - TSE   : tangent space entropy — concentrates latent dynamics along
            fewer intrinsic dimensions of the encoder Jacobian's SVD
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import lightning as L

from ..fnn.regularizers import loss_amplification, loss_cov, loss_false
from ..jacobians.metrics import normalized_mse, r2_score


class LitEncoderDecoder(L.LightningModule):
    """Lightning module for encoder-only representation learning.

    Parameters
    ----------
    encoder : nn.Module
        Causal sequence encoder: (B, T, D_obs) -> (B, T, N_LATENT).
        Must expose an ``n_latent`` attribute.
    n_obs : int
        Observation dimension D_obs.
    use_same_state_decoder : bool
        Enable same-state reconstruction head: decode(z_t) ≈ x_t.
    use_next_state_decoder : bool
        Enable next-state prediction head: decode(z_t) ≈ x_{t+1}.
    decoder_hidden_dim : int
        Hidden width for both MLP decoder heads.
    decoder_n_layers : int
        Number of hidden layers in each MLP decoder head.
    same_state_weight : float
        Loss coefficient for the same-state reconstruction objective.
    next_state_weight : float
        Loss coefficient for the next-state prediction objective.
    fnn_weight : float
        Loss coefficient for the FNN regulariser.  0 disables it.
    fnn_normalize : bool
        Whether to normalize the FNN loss by the variance of the activations.
    fnn_elementwise_regularization : bool
        Whether to use elementwise regularization in FNN.
        If True, the loss is computed as E[W * A^2].
        If False, the loss is computed as E[W] * E[A^2].
    fnn_sparsify : bool
        Whether to sparsify the embedding by penalizing the L1 norm of the embedding.
    amplification_weight : float
        Loss coefficient for the noise-amplification regulariser.  0 disables it.
    decov_weight : float
        Loss coefficient for the DeCov regulariser.  0 disables it.
    jacobian_nuclear_weight : float
        Loss coefficient for the encoder Jacobian nuclear-norm regulariser.
        At each point, computes the mean of singular values of dz/dx; regularises
        the batch mean.  0 disables it.
    jacobian_nuclear_n_samples : int, optional
        Max number of points to subsample when computing the Jacobian nuclear loss
        (O(N) SVDs).  If None, use all points.
    tangent_entropy_weight : float
        Loss coefficient for the tangent space entropy regulariser.  Projects
        latent velocity dz = z_{t+1} - z_t onto the encoder Jacobian's left
        singular vectors (computed at x_t, detached) and minimises the entropy
        of the resulting per-dimension energy distribution.  0 disables it.
    tangent_entropy_n_samples : int, optional
        Max number of points to subsample for the tangent entropy Jacobian
        computation.  If None, use all points.
    tangent_entropy_mode : str
        Entropy formula: ``'shannon'`` (-sum p log p, normalised),
        ``'quadratic'`` (1 - sum p^2, smooth polynomial gradients),
        or ``'renyi_half'`` (Renyi alpha=0.5, aggressive spread penalty).
    amplification_n_neighbors : int
        k-NN neighbourhood size for the amplification loss.
    amplification_max_T : int
        Look-ahead horizon (time steps) for the amplification loss.
    optimizer : str
        ``'AdamW'`` or ``'Adam'``.
    optimizer_kwargs : dict, optional
        Keyword arguments forwarded to the optimiser constructor
        (e.g. ``{'lr': 1e-4, 'weight_decay': 1e-4}``).
    use_scheduler : bool
        Attach a ``ReduceLROnPlateau`` learning-rate scheduler.
    min_lr : float
        Lower bound on the learning rate used by the scheduler.
    lr_scheduler_patience : int
        Patience (epochs) for ``ReduceLROnPlateau``.
    lr_scheduler_factor : float
        Reduction factor for ``ReduceLROnPlateau``.
    gradient_clip_val : float
        Gradient-clipping threshold (passed through to ``L.Trainer``).
    gradient_clip_algorithm : str
        ``'norm'`` or ``'value'`` (passed through to ``L.Trainer``).
    context_margin : int
        Number of leading time-steps to discard from the encoder output
        (useful when causal encoders need a warm-up window for reliable
        representations).
    next_state_burn_in : int
        Number of leading time-steps to skip when computing the next-state
        prediction loss. With partial observations, the encoder may need
        several steps to form a reliable latent before predicting the
        next observation. 0 means no burn-in.
    k_steps_ahead : int
        Number of future steps predicted by the next-state decoder.
        When k=1 (default), the decoder predicts x_{t+1} only.
        When k>1, the decoder jointly predicts [x_{t+1}, ..., x_{t+k}]
        as a flat vector of size k * n_obs_pred, enforcing multi-step
        consistency in the latent representation.
    n_obs_pred : int, optional
        Raw observation dimension used as next-state prediction targets.
        Defaults to n_obs (= input dim).  Set this to the number of
        observed variables (before delay embedding) when delay embedding
        is active, so that targets are un-embedded: the decoder predicts
        [x_{t+1}, ..., x_{t+k}] as raw observations rather than
        delay-embedded states.
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_obs: int,
        use_same_state_decoder: bool = True,
        use_next_state_decoder: bool = False,
        decoder_hidden_dim: int = 128,
        decoder_n_layers: int = 2,
        same_state_weight: float = 1.0,
        next_state_weight: float = 1.0,
        fnn_weight: float = 0.0,
        fnn_normalize: bool = False,
        fnn_elementwise_regularization: bool = False,
        fnn_use_pca: bool = False,
        fnn_n_samples: Optional[int] = None,
        fnn_sparsify: bool = False,
        amplification_weight: float = 0.0,
        decov_weight: float = 0.0,
        jacobian_nuclear_weight: float = 0.0,
        jacobian_nuclear_n_samples: Optional[int] = 256,
        tangent_entropy_weight: float = 0.0,
        tangent_entropy_n_samples: Optional[int] = 256,
        tangent_entropy_mode: str = "quadratic",
        amplification_n_neighbors: int = 10,
        amplification_max_T: int = 5,
        optimizer: str = "AdamW",
        optimizer_kwargs: Optional[dict] = None,
        use_scheduler: bool = True,
        min_lr: float = 1e-6,
        lr_scheduler_patience: int = 10,
        lr_scheduler_factor: float = 0.5,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        context_margin: int = 0,
        next_state_burn_in: int = 0,
        k_steps_ahead: int = 1,
        n_obs_pred: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.n_latent: int = encoder.n_latent
        self.n_obs = n_obs
        self.context_margin = context_margin
        self.next_state_burn_in = next_state_burn_in
        self.k_steps_ahead = k_steps_ahead
        self.n_obs_pred = n_obs_pred if n_obs_pred is not None else n_obs

        self.use_same_state_decoder = use_same_state_decoder
        self.use_next_state_decoder = use_next_state_decoder
        self.same_state_weight = same_state_weight
        self.next_state_weight = next_state_weight
        self.fnn_weight = fnn_weight
        self.fnn_normalize = fnn_normalize
        self.fnn_elementwise_regularization = fnn_elementwise_regularization
        self.fnn_use_pca = fnn_use_pca
        self.fnn_n_samples = fnn_n_samples
        self.fnn_sparsify = fnn_sparsify
        self.amplification_weight = amplification_weight
        self.decov_weight = decov_weight
        self.jacobian_nuclear_weight = jacobian_nuclear_weight
        self.jacobian_nuclear_n_samples = int(jacobian_nuclear_n_samples) if jacobian_nuclear_n_samples is not None and str(jacobian_nuclear_n_samples).lower() != "none" else None
        self.tangent_entropy_weight = tangent_entropy_weight
        self.tangent_entropy_n_samples = int(tangent_entropy_n_samples) if tangent_entropy_n_samples is not None and str(tangent_entropy_n_samples).lower() != "none" else None
        self.tangent_entropy_mode = tangent_entropy_mode
        self.amplification_n_neighbors = amplification_n_neighbors
        self.amplification_max_T = amplification_max_T

        # Build decoder heads
        # same-state: z_t → x_t  (full obs dim, including any delay embedding)
        # next-state: z_t → [x_{t+1}, ..., x_{t+k}]  (k * raw obs dim)
        if use_same_state_decoder:
            self.same_state_decoder = self._build_decoder(
                decoder_hidden_dim, decoder_n_layers, out_dim=self.n_obs
            )
        if use_next_state_decoder:
            self.next_state_decoder = self._build_decoder(
                decoder_hidden_dim, decoder_n_layers,
                out_dim=self.k_steps_ahead * self.n_obs_pred,
            )

        # Require at least one decoder OR at least one regularization loss
        no_decoders = not (use_same_state_decoder or use_next_state_decoder)
        no_reg = (
            fnn_weight == 0
            and amplification_weight == 0
            and decov_weight == 0
            and jacobian_nuclear_weight == 0
            and tangent_entropy_weight == 0
        )
        if no_decoders and no_reg:
            raise ValueError(
                "At least one of use_same_state_decoder, use_next_state_decoder, "
                "or a regularisation weight (fnn, amplification, decov, jacobian_nuclear) must be enabled."
            )

        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4, "weight_decay": 1e-4}
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

        # Exposed for trainer.py compatibility
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Validation loss tracking for percent_improvement (like JacobianODE)
        self.validation_losses: list[float] = []
        self.percent_improvements: list[float] = []
        self.current_epoch_val_losses: list[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_decoder(self, hidden_dim: int, n_layers: int, out_dim: int) -> nn.Sequential:
        """Build a pointwise MLP: N_LATENT -> (hidden)^n_layers -> out_dim."""
        layers: list[nn.Module] = []
        in_dim = self.n_latent
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def _latent_utilization(self, z: torch.Tensor) -> torch.Tensor:
        """Entropy-based utilization ∈ [0, 1].

        1.0 = all latent dimensions carry equal variance.
        0.0 = a single dimension carries all variance.

        Parameters
        ----------
        z : torch.Tensor
            (B, T, N_LATENT) latent sequence.
        """
        var = z.float().var(dim=(0, 1)).clamp(min=1e-10)  # (N_LATENT,)
        p = var / var.sum()
        entropy = -(p * torch.log(p)).sum()
        max_entropy = math.log(self.n_latent) if self.n_latent > 1 else 1.0
        return entropy / max_entropy

    def _encoder_jacobian_nuclear_loss(
        self, x_valid: torch.Tensor
    ) -> torch.Tensor:
        """Mean of singular values of encoder Jacobian, averaged over points.

        For each point x_i, computes the encoder Jacobian J = dz/dx, its singular
        values, and their mean. The loss is the batch mean of these per-point
        means (equivalent to nuclear norm / min(n_latent, n_obs) per point).

        Parameters
        ----------
        x_valid : torch.Tensor
            (B, T', D_obs) observation batch (after context margin).

        Returns
        -------
        torch.Tensor
            Scalar loss: mean over (subsampled) points of mean(singular values).
        """
        x_flat = x_valid.reshape(-1, self.n_obs)
        N = x_flat.shape[0]
        if N == 0:
            return x_valid.new_zeros(1).squeeze()

        if (
            self.jacobian_nuclear_n_samples is not None
            and N > self.jacobian_nuclear_n_samples
        ):
            idx = torch.randperm(N, device=x_valid.device)[
                : self.jacobian_nuclear_n_samples
            ]
            x_sample = x_flat[idx]
        else:
            x_sample = x_flat

        # Use autograd.grad to compute Jacobian (no jacfwd/vmap — avoids dropout randomness error).
        # J[i,j] = dz[i]/dx[j]; compute row i via grad(z[i], x) with grad_outputs=e_i.
        # enable_grad() is needed so this works inside validation_step (which runs under no_grad).
        mean_svals_list: list[torch.Tensor] = []
        with torch.enable_grad():
            for i in range(x_sample.shape[0]):
                x_i = x_sample[i].detach().requires_grad_(True)
                z_i = self.encoder(x_i.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                # J: (n_latent, n_obs); build row by row
                rows: list[torch.Tensor] = []
                for j in range(self.n_latent):
                    g_out = torch.zeros_like(z_i)
                    g_out[j] = 1.0
                    (g,) = torch.autograd.grad(
                        z_i,
                        x_i,
                        grad_outputs=g_out,
                        retain_graph=(j < self.n_latent - 1),
                        create_graph=True,
                    )
                    rows.append(g)
                J_i = torch.stack(rows, dim=0)  # (n_latent, n_obs)
                svals_i = torch.linalg.svdvals(J_i)
                mean_svals_list.append(svals_i.mean())
        return torch.stack(mean_svals_list).mean()

    def _tangent_space_entropy_loss(
        self, x_valid: torch.Tensor, z_valid: torch.Tensor
    ) -> torch.Tensor:
        """Tangent space entropy loss.

        Projects latent velocity ``dz = z_{t+1} - z_t`` onto the encoder
        Jacobian's left singular vectors U (computed at x_t, **detached**),
        then minimises the entropy of the per-dimension energy distribution
        to concentrate dynamics along fewer intrinsic directions.

        Gradients flow through ``dz`` (and thus back into the encoder) but
        **not** through U, so the loss steers the representation without
        reshaping the tangent basis simultaneously.

        Parameters
        ----------
        x_valid : torch.Tensor
            (B, T', D_obs) observation batch (after context margin).
        z_valid : torch.Tensor
            (B, T', N_LATENT) encoded latent batch (after context margin).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        B, T, _ = z_valid.shape
        if T < 2:
            return z_valid.new_zeros(1).squeeze()

        # dz = z_{t+1} - z_t; Jacobian computed at x_t (first-order Taylor)
        dz = z_valid[:, 1:, :] - z_valid[:, :-1, :]   # (B, T-1, n_latent)
        x_for_jac = x_valid[:, :-1, :]                 # (B, T-1, n_obs)

        dz_flat = dz.reshape(-1, self.n_latent)         # (N, n_latent)
        x_flat = x_for_jac.reshape(-1, self.n_obs)      # (N, n_obs)
        N = dz_flat.shape[0]
        if N == 0:
            return z_valid.new_zeros(1).squeeze()

        # Subsample
        if (
            self.tangent_entropy_n_samples is not None
            and N > self.tangent_entropy_n_samples
        ):
            idx = torch.randperm(N, device=z_valid.device)[
                : self.tangent_entropy_n_samples
            ]
            x_sample = x_flat[idx]
            dz_sample = dz_flat[idx]
        else:
            x_sample = x_flat
            dz_sample = dz_flat

        K = min(self.n_latent, self.n_obs)

        # Vectorised Jacobian computation via vmap — no gradient needed
        # through U, so we can use torch.no_grad() + functional transforms.
        # Temporarily switch encoder to eval mode to disable dropout (which
        # is incompatible with vmap's deterministic requirement) and to get
        # a clean deterministic Jacobian.
        def _encode_point(x: torch.Tensor) -> torch.Tensor:
            """Single-point encoder: (n_obs,) -> (n_latent,)."""
            return self.encoder(x.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # Pick jacrev vs jacfwd based on which dimension is smaller:
        # jacrev is O(n_out) backward passes, jacfwd is O(n_in) forward passes.
        if self.n_latent <= self.n_obs:
            jac_fn = torch.func.jacrev(_encode_point)
        else:
            jac_fn = torch.func.jacfwd(_encode_point)

        was_training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                # vmap over the sample dimension: (M, n_obs) -> (M, n_latent, n_obs)
                J_all = torch.func.vmap(jac_fn)(x_sample)
                # SVD: U is (M, n_latent, K)
                U_all, _, _ = torch.linalg.svd(J_all, full_matrices=False)
        finally:
            if was_training:
                self.encoder.train()

        # Project dz onto detached U: grad flows through dz_sample only.
        # U_all: (M, n_latent, K), dz_sample: (M, n_latent)
        # projections: (M, K) = bmm(U^T, dz)
        projections = torch.bmm(
            U_all.transpose(-2, -1), dz_sample.unsqueeze(-1)
        ).squeeze(-1)
        squared_projections = projections ** 2

        # Energy per dimension, averaged over batch
        E = squared_projections.mean(dim=0)  # (K,)
        p = E / (E.sum() + 1e-10)

        # Entropy
        if self.tangent_entropy_mode == "shannon":
            eps = 1e-10
            entropy = -(p * torch.log(p + eps)).sum()
            max_ent = math.log(K) if K > 1 else 1.0
            loss = entropy / max_ent
        elif self.tangent_entropy_mode == "quadratic":
            loss = 1.0 - (p ** 2).sum()
        elif self.tangent_entropy_mode == "renyi_half":
            eps = 1e-10
            loss = 2.0 * torch.log(torch.sqrt(p + eps).sum() + eps)
        else:
            raise ValueError(
                f"Unknown tangent_entropy_mode: {self.tangent_entropy_mode!r}. "
                f"Choose from 'shannon', 'quadratic', 'renyi_half'."
            )
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations.

        Parameters
        ----------
        x : torch.Tensor
            (B, T, D_obs)

        Returns
        -------
        torch.Tensor
            (B, T, N_LATENT)
        """
        return self.encoder(x)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_losses(
        self, x: torch.Tensor, prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute all losses and diagnostic metrics.

        Parameters
        ----------
        x : torch.Tensor
            Observation batch (B, T, D_obs).
        prefix : str
            Logging prefix, e.g. ``'train'`` or ``'val'``.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar total loss.
        log_dict : dict
            Metrics to log to W&B (all scalars, detached where appropriate).
        """
        z = self.encoder(x)  # (B, T, N_LATENT)

        # Apply context margin: skip first context_margin timesteps
        if self.context_margin > 0:
            z_valid = z[:, self.context_margin:]
            x_valid = x[:, self.context_margin:]
        else:
            z_valid = z
            x_valid = x

        total_loss = x.new_zeros(1).squeeze()
        log_dict: Dict[str, torch.Tensor] = {}

        # ----------------------------------------------------------
        # Same-state reconstruction:  decode(z_t) ≈ x_t
        # ----------------------------------------------------------
        if self.use_same_state_decoder:
            x_hat = self.same_state_decoder(z_valid)  # (B, T', D_obs)
            same_loss = normalized_mse(x_valid, x_hat)
            total_loss = total_loss + self.same_state_weight * same_loss
            log_dict[f"{prefix}/same_state_loss"] = same_loss.detach()
            with torch.no_grad():
                log_dict[f"{prefix}/same_state_r2"] = r2_score(x_valid, x_hat)

        # ----------------------------------------------------------
        # Next-state prediction:  decode(z_t) ≈ [x_{t+1}, ..., x_{t+k}]
        # With burn-in: skip the first next_state_burn_in predictions.
        # Targets use the first n_obs_pred dims of x (raw obs before any
        # delay embedding), so k_steps_ahead > 1 forces multi-step consistency
        # without the redundancy of predicting shifted delay-embedded states.
        # ----------------------------------------------------------
        if self.use_next_state_decoder:
            burn_in = self.next_state_burn_in
            k = self.k_steps_ahead
            T_valid = z_valid.shape[1]
            T_pred = T_valid - burn_in - k  # number of valid prediction positions
            if T_pred > 0:
                z_for_next = z_valid[:, burn_in:burn_in + T_pred]  # (B, T_pred, N_LATENT)

                # Stack k future raw-obs targets: (B, T_pred, k, n_obs_pred)
                target_steps = torch.stack(
                    [x_valid[:, burn_in + j:burn_in + j + T_pred, :self.n_obs_pred]
                     for j in range(1, k + 1)],
                    dim=2,
                )
                target_flat = target_steps.reshape(
                    target_steps.shape[0], target_steps.shape[1],
                    k * self.n_obs_pred,
                )  # (B, T_pred, k * n_obs_pred)

                x_next_hat = self.next_state_decoder(z_for_next)  # (B, T_pred, k * n_obs_pred)
                next_loss = normalized_mse(target_flat, x_next_hat)
                total_loss = total_loss + self.next_state_weight * next_loss
                log_dict[f"{prefix}/next_state_loss"] = next_loss.detach()
                with torch.no_grad():
                    log_dict[f"{prefix}/next_state_r2"] = r2_score(target_flat, x_next_hat)
            else:
                # No valid predictions after burn-in; skip this loss
                log_dict[f"{prefix}/next_state_loss"] = x.new_tensor(float("nan"))
                log_dict[f"{prefix}/next_state_r2"] = x.new_tensor(float("nan"))

        # ----------------------------------------------------------
        # FNN regularisation (O(N²) — subsample to ≤1024 points)
        # ----------------------------------------------------------
        if self.fnn_weight > 0:
            z_flat = z_valid.reshape(-1, self.n_latent).float()
            fnn_loss = loss_false(
                z_flat,
                normalize=self.fnn_normalize,
                elementwise_regularization=self.fnn_elementwise_regularization,
                use_pca=self.fnn_use_pca,
                n_samples=self.fnn_n_samples,
                sparsify=self.fnn_sparsify,
            )
            total_loss = total_loss + self.fnn_weight * fnn_loss
            log_dict[f"{prefix}/fnn_loss"] = fnn_loss.detach()

        # ----------------------------------------------------------
        # Noise amplification regularisation
        # ----------------------------------------------------------
        if self.amplification_weight > 0:
            amp_loss = loss_amplification(
                embedding=z_valid.float(),
                data=x_valid.float(),
                n_neighbors=self.amplification_n_neighbors,
                max_T=self.amplification_max_T,
                normalize=True,
            )
            total_loss = total_loss + self.amplification_weight * amp_loss
            log_dict[f"{prefix}/amplification_loss"] = amp_loss.detach()

        # ----------------------------------------------------------
        # DeCov regularisation
        # ----------------------------------------------------------
        if self.decov_weight > 0:
            z_flat = z_valid.reshape(-1, self.n_latent)
            decov_loss = loss_cov(z_flat)
            total_loss = total_loss + self.decov_weight * decov_loss
            log_dict[f"{prefix}/decov_loss"] = decov_loss.detach()

        # ----------------------------------------------------------
        # Encoder Jacobian nuclear-norm regularisation
        # At each point: mean(singular values of dz/dx).  Penalises large
        # singular values to encourage contractive / low-rank encoder maps.
        # ----------------------------------------------------------
        if self.jacobian_nuclear_weight > 0:
            jac_nuc_loss = self._encoder_jacobian_nuclear_loss(x_valid)
            total_loss = total_loss + self.jacobian_nuclear_weight * jac_nuc_loss
            log_dict[f"{prefix}/jacobian_nuclear_loss"] = jac_nuc_loss.detach()

        # ----------------------------------------------------------
        # Tangent space entropy regularisation
        # Project latent velocity dz onto local encoder tangent basis U,
        # then minimise entropy of the per-dimension energy distribution
        # to concentrate dynamics along fewer intrinsic directions.
        # ----------------------------------------------------------
        if self.tangent_entropy_weight > 0:
            tse_loss = self._tangent_space_entropy_loss(x_valid, z_valid)
            total_loss = total_loss + self.tangent_entropy_weight * tse_loss
            log_dict[f"{prefix}/tangent_entropy_loss"] = tse_loss.detach()

        # ----------------------------------------------------------
        # Latent utilization (diagnostic, no gradient)
        # ----------------------------------------------------------
        with torch.no_grad():
            log_dict[f"{prefix}/latent_utilization"] = self._latent_utilization(
                z_valid
            )

        log_dict[f"{prefix}/total_loss"] = total_loss.detach()
        return total_loss, log_dict

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss, log_dict = self._compute_losses(x, "train")
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss, log_dict = self._compute_losses(x, "val")
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Aliases expected by the shared train_model / trainer utilities
        self.log("mean val loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("trajectory val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.current_epoch_val_losses.append(loss.detach().item())
        return loss

    def on_validation_epoch_end(self):
        """Track validation losses and log percent_improvement (like JacobianODE)."""
        if self.current_epoch_val_losses:
            mean_val_loss = sum(self.current_epoch_val_losses) / len(
                self.current_epoch_val_losses
            )
            self.validation_losses.append(mean_val_loss)
            if len(self.validation_losses) > 1:
                # Use last non-NaN losses for comparison (NaN comparisons always return False)
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
