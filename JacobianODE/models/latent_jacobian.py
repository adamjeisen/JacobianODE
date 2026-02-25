"""Latent-space Jacobian ODE model with learned encoder/decoder.

This module implements end-to-end training of an autoencoder paired with
JacobianODE dynamics prediction in the latent space.  The encoder maps
partial observations into a latent embedding where Jacobian-based forward
integration is performed, and the decoder maps predictions back to
observation space.  The prediction loss backpropagates through the full
pipeline, forcing the encoder to discover dynamically coherent latent
representations.

Architecture:
    Observations (B, T, D_obs)
        -> Encoder -> Latent trajectory (B, T', D_latent)
        -> JacobianODEint on sub-windows -> Predicted latent trajectory
        -> Decoder -> Predicted observations
"""

import numpy as np
import torch
import torch.nn as nn

from ..jacobians.lightning_base import LitBase, loop_closure
from ..jacobians.jacobianODE import JacobianODEint
from ..jacobians.metrics import mase, mse, r2_score, normalized_mse


class LitLatentJacobianODE(LitBase):
    """Lightning module for end-to-end latent-space Jacobian ODE training.

    Trains an encoder, Jacobian model, and decoder jointly.  The Jacobian
    model operates in the learned latent space; the prediction loss in
    observation space forces the encoder to preserve attractor geometry.

    Parameters
    ----------
    model : nn.Module
        The Jacobian MLP (maps latent vectors to flattened Jacobian matrices).
    encoder : nn.Module
        Autoencoder with ``encode`` and ``decode`` methods.
        Window-based encoders (MLPAutoencoder, LSTMAutoencoder) must have a
        ``time_window`` attribute.
    prediction_steps : int
        Number of latent time steps to predict forward.  The JacobianODEint
        sub-window length is ``traj_init_steps + prediction_steps``.
    **kwargs
        All remaining keyword arguments are forwarded to ``LitBase``.
    """

    def __init__(
        self,
        model,
        encoder,
        prediction_steps=10,
        encoder_warmup_epochs=0,
        jac_window_stride=None,
        true_lyapunov_exponents=None,
        reconstruction_loss_weight=1.0,
        latent_prediction_loss_weight=0.0,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.encoder = encoder
        self.prediction_steps = prediction_steps
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.latent_prediction_loss_weight = latent_prediction_loss_weight
        # Stride for JacobianODE sub-windows within each encoded batch.
        # Defaults to prediction_steps (non-overlapping).
        self.jac_window_stride = jac_window_stride if jac_window_stride is not None else prediction_steps

        # Store precomputed true Lyapunov exponents for logging.
        # In partially observed settings, true Jacobians can't be computed
        # from the batch, but Lyapunov exponents are coordinate-invariant
        # and can be precomputed from the full system.
        if true_lyapunov_exponents is not None:
            self.register_buffer(
                'true_lyapunov_exponents',
                torch.as_tensor(true_lyapunov_exponents, dtype=torch.float32),
            )
        else:
            self.true_lyapunov_exponents = None

    # ------------------------------------------------------------------
    # Encoder / decoder abstractions
    # ------------------------------------------------------------------

    def encode_trajectory(self, batch):
        """Encode an observation trajectory into latent space.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations of shape ``(B, T, D_obs)``.

        Returns
        -------
        torch.Tensor
            Latent trajectory of shape ``(B, T', D_latent)``.
            For window-based encoders ``T' = T - w + 1``.
            For sequence-based encoders with ``context_margin > 0``,
            the first ``context_margin`` timesteps are dropped from the
            latent output (they lack sufficient causal context).
        """
        if hasattr(self.encoder, 'time_window'):
            # Window-based encoder: unfold into sliding windows
            B, T, D = batch.shape
            w = self.encoder.time_window
            # (B, T', w, D)
            windows = batch.unfold(1, w, 1).permute(0, 1, 3, 2)
            T_prime = windows.shape[1]
            # Flatten batch and time for encoding
            windows_flat = windows.reshape(B * T_prime, w, D)
            z_flat = self.encoder.encode(windows_flat)  # (B*T', D_latent)
            z = z_flat.reshape(B, T_prime, -1)
        else:
            # Sequence-based encoder: process full sequence
            z = self.encoder.encode(batch)
            # Drop initial embeddings that lack sufficient causal context
            margin = getattr(self.encoder, 'context_margin', 0)
            if margin > 0:
                z = z[:, margin:, :]
        return z

    def decode_trajectory(self, z):
        """Decode latent vectors back to observation space.

        Parameters
        ----------
        z : torch.Tensor
            Latent trajectory of shape ``(B, T', D_latent)`` or
            ``(B*T', D_latent)``.

        Returns
        -------
        torch.Tensor
            Decoded observations.  For window-based encoders the shape is
            ``(B, T', time_window, n_features)``.  For sequence-based
            encoders the shape depends on the decoder.
        """
        if hasattr(self.encoder, 'time_window'):
            leading_shape = z.shape[:-1]
            z_flat = z.reshape(-1, z.shape[-1])
            decoded = self.encoder.decode(z_flat)  # (N, w, D_obs)
            decoded = decoded.reshape(*leading_shape, *decoded.shape[1:])
        else:
            decoded = self.encoder.decode(z)
        return decoded

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def compute_jacobians(self, batch, t=0, batch_idx=0, dataloader_idx=0):
        """Compute Jacobians in latent space using the direct MLP.

        Parameters
        ----------
        batch : torch.Tensor
            Latent trajectory of shape ``(..., T, D_latent)`` or
            ``(..., D_latent)``.

        Returns
        -------
        torch.Tensor
            Jacobian matrices of shape ``(..., T, D_lat, D_lat)`` or
            ``(..., D_lat, D_lat)``.
        """
        d = batch.shape[-1]
        return self.model(batch).reshape(*batch.shape[:-1], d, d)

    # ------------------------------------------------------------------
    # Lyapunov exponents
    # ------------------------------------------------------------------

    @staticmethod
    def compute_lyapunov_exponents(jacs, dt):
        """Compute Lyapunov exponents via QR decomposition.

        Parameters
        ----------
        jacs : torch.Tensor
            Sequence of Jacobian matrices along a trajectory,
            shape ``(T, D, D)`` or ``(B, T, D, D)``.
        dt : float
            Time step between successive Jacobians.

        Returns
        -------
        torch.Tensor
            Lyapunov exponents of shape ``(D,)`` or ``(B, D)``, sorted
            descending.
        """
        unbatched = jacs.ndim == 3
        if unbatched:
            jacs = jacs.unsqueeze(0)

        B, T, D, _ = jacs.shape
        Q = torch.eye(D, dtype=jacs.dtype, device=jacs.device).expand(B, -1, -1).clone()
        log_diag_sum = torch.zeros(B, D, dtype=jacs.dtype, device=jacs.device)

        for t in range(T):
            M = torch.linalg.matrix_exp(jacs[:, t] * dt)
            Z = M @ Q
            Q, R = torch.linalg.qr(Z)
            diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
            signs = torch.sign(diag_R)
            signs[signs == 0] = 1.0
            Q = Q * signs.unsqueeze(-2)
            R = R * signs.unsqueeze(-1)
            log_diag_sum += torch.log(torch.abs(torch.diagonal(R, dim1=-2, dim2=-1)))

        exponents = log_diag_sum / (T * dt)
        exponents = exponents.sort(descending=True, dim=-1).values

        if unbatched:
            exponents = exponents.squeeze(0)
        return exponents

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _extract_obs_targets(self, batch, start_indices, n_windows_actual,
                              traj_init_steps):
        """Extract observation-space targets for predicted latent steps.

        For window-based encoders, latent index ``t`` was encoded from
        ``batch[:, t:t+w, :]``.  The target for each predicted latent step
        is the full observation window, giving shape
        ``(N, prediction_steps, w, D_obs)``.

        For sequence-based encoders, latent index ``t`` maps directly to
        ``batch[:, t, :]``, giving shape ``(N, prediction_steps, D_obs)``.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.
        start_indices : torch.Tensor
            Per-sub-window start indices into the latent trajectory, shape
            ``(N,)`` where ``N = B * n_windows_actual``.
        n_windows_actual : int
            Number of sub-windows per batch element.
        traj_init_steps : int

        return_latent : bool
            Whether to return the latent targets instead of the observation targets.

        Returns
        -------
        torch.Tensor
            Observation targets.
        """
        prediction_steps = self.prediction_steps
        B = batch.shape[0]

        if hasattr(self.encoder, 'time_window'):
            w = self.encoder.time_window
            targets = []
            for idx in range(start_indices.shape[0]):
                b = idx // n_windows_actual
                s = start_indices[idx].item()
                # Each predicted latent step t -> obs window batch[b, t:t+w, :]
                windows = []
                for t in range(prediction_steps):
                    latent_idx = s + traj_init_steps + t
                    windows.append(batch[b, latent_idx:latent_idx + w, :])
                targets.append(torch.stack(windows))  # (prediction_steps, w, D_obs)
            return torch.stack(targets)  # (N, prediction_steps, w, D_obs)
        else:
            # For sequence-based encoders, latent index t corresponds to
            # observation index t + context_margin (since the first
            # context_margin latents were dropped in encode_trajectory).
            margin = getattr(self.encoder, 'context_margin', 0)
            targets = []
            for idx in range(start_indices.shape[0]):
                b = idx // n_windows_actual
                s = start_indices[idx].item()
                obs_start = s + traj_init_steps + margin
                obs_end = obs_start + prediction_steps
                targets.append(batch[b, obs_start:obs_end, :])
            return torch.stack(targets)  # (N, prediction_steps, D_obs)

    def trajectory_model_step(
        self,
        batch,
        batch_idx=0,
        dataloader_idx=0,
        all_metrics=False,
        direct=None,
        obs_noise_scale=None,
        alpha_teacher_forcing=None,
        teacher_forcing_steps=None,
        jacobianODEint_kwargs=None,
        criterion=None,
        verbose=False,
        return_decoded=False,
    ):
        """Trajectory prediction step in latent space.

        1. Encode full observation sequence to latent trajectory.
        2. Extract *strided* sub-windows (not just one per batch element).
        3. Integrate forward in latent space via JacobianODEint.
        4. Decode predicted latent vectors to observation space.
        5. Compute loss against raw observation targets (not decoded true z).
        """
        if obs_noise_scale is None:
            obs_noise_scale = self.obs_noise_scale
        if alpha_teacher_forcing is None:
            alpha_teacher_forcing = self.alpha_teacher_forcing
        if teacher_forcing_steps is None:
            teacher_forcing_steps = self.teacher_forcing_steps
        if criterion is None:
            criterion = self.criterion
        if jacobianODEint_kwargs is None:
            jacobianODEint_kwargs = self.jacobianODEint_kwargs

        batch = batch.type(self.dtype)
        label = batch.detach().clone()

        # Add observation noise during training
        batch_noisy = batch + (torch.randn_like(batch) * obs_noise_scale)

        # 1. Encode full observation sequence
        z_full = self.encode_trajectory(batch_noisy)  # (B, T', D_latent)

        # 2. Determine sub-window parameters
        traj_init_steps = jacobianODEint_kwargs.get('traj_init_steps', 15)
        jac_window_len = traj_init_steps + self.prediction_steps

        B, T_prime, D_latent = z_full.shape
        if T_prime < jac_window_len:
            raise ValueError(
                f"Latent trajectory length ({T_prime}) is shorter than "
                f"required JacobianODE window ({jac_window_len} = "
                f"traj_init_steps={traj_init_steps} + "
                f"prediction_steps={self.prediction_steps}). "
                f"Increase observation sequence length or reduce prediction_steps."
            )

        # Extract strided sub-windows from each trajectory
        stride = self.jac_window_stride
        n_windows = max(1, (T_prime - jac_window_len) // stride + 1)
        all_starts = []
        for b in range(B):
            for w_idx in range(n_windows):
                start = w_idx * stride
                if start + jac_window_len <= T_prime:
                    all_starts.append(start)
                    # Repeat z_full[b] — we'll gather below
        # Build start_indices tensor: (N,) where N = B * n_windows
        n_windows_actual = len(all_starts) // B
        start_indices = torch.tensor(
            all_starts, device=z_full.device, dtype=torch.long
        )

        # Gather sub-windows: (N, jac_window_len, D_latent)
        z_windows_list = []
        for idx in range(start_indices.shape[0]):
            b = idx // n_windows_actual
            s = start_indices[idx]
            z_windows_list.append(z_full[b, s:s + jac_window_len])
        z_windows = torch.stack(z_windows_list)  # (N, jac_window_len, D_latent)

        # 3. Run JacobianODEint on all sub-windows
        jacobian_odeint = JacobianODEint(self.compute_jacobians, self.dt)
        z_pred = jacobian_odeint.generate_dynamics(
            z_windows,
            alpha_teacher_forcing=alpha_teacher_forcing,
            teacher_forcing_steps=teacher_forcing_steps,
            fast_mode=True,
            scale_interp_pts=True,
            **{k: v for k, v in jacobianODEint_kwargs.items()
               if k != 'traj_init_steps'},
            traj_init_steps=traj_init_steps,
        )  # (N, jac_window_len, D_latent)

        # 4. Crop to prediction portion
        z_pred_crop = z_pred[..., traj_init_steps:, :]  # (N, prediction_steps, D_lat)
        z_true_crop = z_windows[:, traj_init_steps:, :]  # (N, prediction_steps, D_lat)

        # 5. Decode predicted latents and compare to raw observation targets
        decoded_pred = self.decode_trajectory(z_pred_crop)
        # Window-based: (N, prediction_steps, w, D_obs)
        # Sequence-based: (N, prediction_steps, D_obs)

        # Build observation-space targets from the original (clean) batch
        obs_targets = self._extract_obs_targets(
            label, start_indices, n_windows_actual, traj_init_steps
        )  # same shape as decoded_pred

        # Variance-normalized losses: each ≈ mean_d(1 - R²_d), so all on the
        # same scale regardless of whether we are in observation or latent space.
        loss = normalized_mse(decoded_pred, obs_targets)

        # Metrics — flatten window dims for scalar metrics
        metric_vals = {}
        with torch.no_grad():
            metric_vals['mase'] = mase(obs_targets, decoded_pred)
            pred_flat = decoded_pred.reshape(decoded_pred.shape[0], -1)
            tgt_flat = obs_targets.reshape(obs_targets.shape[0], -1)
            metric_vals['r2_score'] = r2_score(tgt_flat, pred_flat)

        # Latent prediction loss (computed outside no_grad so gradients flow
        # back through the encoder, penalising unpredictable latent dims).
        latent_pred_loss = normalized_mse(z_pred_crop, z_true_crop)
        metric_vals['latent_pred_loss'] = latent_pred_loss

        if return_decoded:
            return {'loss': loss, 'metric_vals': metric_vals, 'outputs': z_pred, 'decoded': decoded_pred, 'targets': obs_targets}
        else:
            return {'loss': loss, 'metric_vals': metric_vals, 'outputs': z_pred}

    # ------------------------------------------------------------------
    # Reconstruction loss
    # ------------------------------------------------------------------

    def _reconstruction_loss(self, batch, z_full=None):
        """Compute observation-space reconstruction loss: decode(encode(x)) ≈ x.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.
        z_full : torch.Tensor, optional
            Pre-computed latent trajectory ``(B, T', D_latent)``.  If None,
            the batch is encoded fresh.

        Returns
        -------
        torch.Tensor
            Scalar MSE reconstruction loss.
        """
        if z_full is None:
            z_full = self.encode_trajectory(batch)

        recon_decoded = self.decode_trajectory(z_full)

        if hasattr(self.encoder, 'time_window'):
            w = self.encoder.time_window
            B, T, D = batch.shape
            recon_targets = batch.unfold(1, w, 1).permute(0, 1, 3, 2)  # (B, T', w, D)
        else:
            margin = getattr(self.encoder, 'context_margin', 0)
            recon_targets = batch[:, margin:, :] if margin > 0 else batch

        return normalized_mse(recon_decoded, recon_targets)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _warmup_step(self, batch):
        """Encoder warmup: reconstruction loss only, no Jacobian prediction.

        Encodes observation windows to latent, decodes back, and computes
        MSE against the original windows.  Freezes the Jacobian model.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.

        Returns
        -------
        torch.Tensor
            Scalar reconstruction loss.
        """
        batch = batch.type(self.dtype)

        z_full = self.encode_trajectory(batch)        # (B, T', D_latent)
        decoded = self.decode_trajectory(z_full)       # (B, T', w, D_obs)

        # Build targets: for each latent index t, target is batch[:, t:t+w, :]
        if hasattr(self.encoder, 'time_window'):
            w = self.encoder.time_window
            B, T, D = batch.shape
            targets = batch.unfold(1, w, 1).permute(0, 1, 3, 2)  # (B, T', w, D)
        else:
            targets = batch

        loss = nn.functional.mse_loss(decoded, targets)

        self.log("warmup recon_loss", loss, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Full training step: encode, predict, decode, loop closure.

        During the first ``encoder_warmup_epochs``, only the autoencoder
        reconstruction loss is used (no Jacobian prediction or loop closure).

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations of shape ``(B, T, D_obs)``.
        """
        # Encoder warmup phase: reconstruction only
        if self.current_epoch < self.encoder_warmup_epochs:
            return self._warmup_step(batch)

        batch = batch.type(self.dtype)

        # Encode for teacher forcing update
        with torch.no_grad():
            z_for_tf = self.encode_trajectory(batch)
            jacs_pred = self.compute_jacobians(z_for_tf)

        jac_norm = torch.linalg.norm(
            jacs_pred, dim=(-2, -1), ord=self.jac_norm_ord
        ).mean()
        self.update_alpha_teacher_forcing(jacs_pred.detach(), batch_idx)

        train_rets = {}

        # Trajectory prediction loss
        if self.trajectory_training:
            train_rets['trajectory'] = self.trajectory_model_step(
                batch, batch_idx, dataloader_idx
            )

        # Encode once for loop closure and/or reconstruction
        z_full = self.encode_trajectory(batch)

        # Loop closure in latent space
        if self.loop_closure_training:
            train_rets['loop_closure'] = self.loop_closure_model_step(
                z_full, batch_idx, dataloader_idx
            )

        # Combine losses
        total_loss = 0
        for pred_type, ret_dict in train_rets.items():
            if torch.isnan(ret_dict['loss']):
                print(
                    f"Warning: Loss is nan for pred type {pred_type} "
                    f"on epoch {self.current_epoch} batch {batch_idx}"
                )
            loss_val = ret_dict['loss'] if not torch.isnan(ret_dict['loss']) else 0
            loss_weight = 1.0
            if 'loop_closure' in pred_type:
                loss_weight *= self.loop_closure_weight
            total_loss += loss_weight * loss_val

        if self.jac_penalty > 0:
            total_loss += self.jac_penalty * jac_norm

        # Reconstruction loss: decode(encode(x)) ≈ x in observation space
        recon_loss = None
        if self.reconstruction_loss_weight > 0:
            recon_loss = self._reconstruction_loss(batch, z_full=z_full)
            total_loss += self.reconstruction_loss_weight * recon_loss

        # Latent prediction loss: JacobianODE(z_t) ≈ z_{t+k} in latent space
        latent_pred_loss = None
        if self.latent_prediction_loss_weight > 0 and self.trajectory_training:
            latent_pred_loss = train_rets['trajectory']['metric_vals'].get('latent_pred_loss')
            if latent_pred_loss is not None:
                total_loss += self.latent_prediction_loss_weight * latent_pred_loss

        l1_loss = torch.sum(torch.abs(
            torch.cat([p.view(-1) for p in self.get_main_params()], dim=0)
        ))

        if ((batch_idx + 1) % self.log_interval) == 0:
            self.log_training_metrics(
                train_rets=train_rets,
                total_loss=total_loss,
                jac_norm=jac_norm,
                l1_loss=l1_loss,
                batch=batch,
                jacs_pred=jacs_pred,
                batch_idx=batch_idx,
                recon_loss=recon_loss,
                latent_pred_loss=latent_pred_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

        return total_loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0, log_metrics=True):
        """Validation step with Lyapunov exponent diagnostics.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations of shape ``(B, T, D_obs)``.
        """
        batch = batch.type(self.dtype)

        model_step_kwargs = {
            'alpha_teacher_forcing': self.alpha_validation,
            'obs_noise_scale': 0,
        }

        val_rets = {}
        val_rets['trajectory'] = self.trajectory_model_step(
            batch, batch_idx, dataloader_idx, **model_step_kwargs
        )

        # Encode once for loop closure and reconstruction
        z_full = self.encode_trajectory(batch)
        val_loop_closure = self.loop_closure_model_step(
            z_full, batch_idx, dataloader_idx
        )

        # Reconstruction loss
        val_recon_loss = None
        if self.reconstruction_loss_weight > 0:
            with torch.no_grad():
                val_recon_loss = self._reconstruction_loss(batch, z_full=z_full)

        # Latent prediction loss (already computed inside trajectory_model_step)
        val_latent_pred_loss = val_rets['trajectory']['metric_vals'].get('latent_pred_loss')

        if log_metrics:
            self.log_validation_metrics(
                val_rets=val_rets,
                batch=batch,
                sync_dist=True,
                val_loop_closure=val_loop_closure,
                val_recon_loss=val_recon_loss,
                val_latent_pred_loss=val_latent_pred_loss,
            )

        total_loss = sum(
            val_rets[pred_type]['loss'] for pred_type in val_rets
        )
        if val_recon_loss is not None:
            total_loss = total_loss + self.reconstruction_loss_weight * val_recon_loss
        if val_latent_pred_loss is not None:
            total_loss = total_loss + self.latent_prediction_loss_weight * val_latent_pred_loss

        if not hasattr(self, 'current_epoch_val_losses'):
            self.current_epoch_val_losses = []
        self.current_epoch_val_losses.append(total_loss.item())

        return total_loss

    # ------------------------------------------------------------------
    # Jacobian logging overrides
    # ------------------------------------------------------------------

    def _log_lyapunov_comparison(self, batch, prefix, sync_dist=True, **log_kwargs):
        """Compute predicted Lyapunov exponents and log comparison with true values.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.
        prefix : str
            Logging prefix, e.g. ``"train"`` or ``"val"``.
        sync_dist : bool
        """
        if self.true_lyapunov_exponents is None:
            return

        try:
            with torch.no_grad():
                z = self.encode_trajectory(batch)
                # Use first batch element for a long trajectory
                jacs = self.compute_jacobians(z[0:1])[0]  # (T', D, D)
                pred_le = self.compute_lyapunov_exponents(jacs, self.dt)

                # Compare the top-k exponents (k = number of true exponents)
                k = len(self.true_lyapunov_exponents)
                pred_top_k = pred_le[:k]
                true_le = self.true_lyapunov_exponents.to(pred_top_k.device)

                le_mse = torch.mean((pred_top_k - true_le) ** 2)
                self.log(
                    f"{prefix} lyapunov_mse", le_mse,
                    sync_dist=sync_dist, add_dataloader_idx=False,
                    **log_kwargs,
                )

                # Log individual exponents
                for i, (pred, true) in enumerate(zip(pred_top_k, true_le)):
                    self.log(
                        f"{prefix} lyapunov_{i} (pred)", pred.item(),
                        sync_dist=sync_dist, add_dataloader_idx=False,
                        **log_kwargs,
                    )
        except Exception:
            pass  # Don't fail training/validation on diagnostic errors

    def _log_latent_utilization(self, batch, prefix, **log_kwargs):
        """Log sum of normalized latent variances (utilization score).

        The per-dimension variances are divided by the largest variance
        and summed.  A value of ~1 means only one dimension is active;
        a value approaching ``D_latent`` means all dimensions are
        equally utilized.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.
        prefix : str
            Logging prefix, e.g. ``"train"`` or ``"val"``.
        """
        try:
            with torch.no_grad():
                z = self.encode_trajectory(batch)
                z_var = z.var(dim=(0, 1))  # (D_latent,)
                max_var = z_var.max()
                if max_var > 0:
                    utilization = (z_var / max_var).sum().item()
                else:
                    utilization = 0.0
                self.log(f"{prefix} latent_utilization", utilization, **log_kwargs)
        except Exception:
            pass

    def log_training_metrics(self, train_rets, total_loss, jac_norm, l1_loss,
                              batch, jacs_pred, batch_idx, recon_loss=None,
                              latent_pred_loss=None,
                              on_step=False, on_epoch=True, sync_dist=True,
                              prog_bar=True):
        """Log training metrics.

        Overrides the base class to skip true-Jacobian comparison (not
        possible for partially observed systems) and instead log Lyapunov
        exponent diagnostics.
        """
        log_kwargs = dict(on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)

        for pred_type, ret_dict in train_rets.items():
            loss, metric_vals = ret_dict['loss'], ret_dict['metric_vals']
            self.log(f"{pred_type} train_loss", loss, **log_kwargs)
            for metric, val in metric_vals.items():
                if pred_type != 'trajectory':
                    continue
                self.log(f"{pred_type} train {metric}", val, **log_kwargs)

        self.log("total train loss", total_loss, **log_kwargs)
        if jac_norm is not None:
            self.log("train jac norm", jac_norm, **log_kwargs)
        self.log("train l1 norm", l1_loss, **log_kwargs)
        if recon_loss is not None:
            self.log("train recon_loss", recon_loss, **log_kwargs)
        if latent_pred_loss is not None:
            self.log("train latent_pred_loss", latent_pred_loss, **log_kwargs)

        if self.teacher_forcing_annealing:
            self.log("alpha teacher forcing", self.alpha_teacher_forcing, **log_kwargs)

        self._log_lyapunov_comparison(batch, "train", **log_kwargs)
        self._log_latent_utilization(batch, "train", **log_kwargs)

    def log_validation_metrics(self, val_rets, batch, sync_dist=True,
                               val_loop_closure=None, val_recon_loss=None,
                               val_latent_pred_loss=None):
        """Log validation metrics.

        Overrides the base class to skip true-Jacobian comparison and
        instead log Lyapunov exponent diagnostics.
        """
        for pred_type, ret_dict in val_rets.items():
            loss, metric_vals = ret_dict['loss'], ret_dict['metric_vals']
            self.log(f"{pred_type} val_loss", loss, sync_dist=sync_dist, add_dataloader_idx=False)
            for metric, val in metric_vals.items():
                if pred_type != 'trajectory':
                    continue
                self.log(f"{pred_type} val {metric}", val, sync_dist=sync_dist, add_dataloader_idx=False)

        mean_val_loss = torch.stack(
            [val_rets[pt]['loss'] for pt in val_rets]
        ).mean()
        self.log("mean val loss", mean_val_loss, sync_dist=sync_dist)

        if val_loop_closure is not None:
            self.log(
                "val loop closure loss",
                val_loop_closure['loss'],
                sync_dist=sync_dist,
                add_dataloader_idx=False,
            )

        if val_recon_loss is not None:
            self.log("val recon_loss", val_recon_loss, sync_dist=sync_dist,
                     add_dataloader_idx=False)
        if val_latent_pred_loss is not None:
            self.log("val latent_pred_loss", val_latent_pred_loss, sync_dist=sync_dist,
                     add_dataloader_idx=False)

        self._log_lyapunov_comparison(batch, "val", sync_dist=sync_dist)
        self._log_latent_utilization(batch, "val", sync_dist=sync_dist)

    def get_pred_jacs(self, batch):
        """Override to compute Jacobians in latent space."""
        z = self.encode_trajectory(batch)
        return self.compute_jacobians(z)
