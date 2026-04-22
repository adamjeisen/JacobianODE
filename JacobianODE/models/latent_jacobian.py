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

import math
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..jacobians.lightning_base import LitBase, loop_closure
from ..jacobians.jacobianODE import JacobianODEint
from ..jacobians.metrics import mase, mse, r2_score, normalized_mse
from ..fnn.regularizers import loss_false, tangent_space_entropy


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
        dynamics_warmup_epochs=0,
        jac_window_stride=None,
        true_lyapunov_exponents=None,
        reconstruction_loss_weight=1.0,
        latent_prediction_loss_weight=0.0,
        jac_consistency_weight=0.0,
        fnn_weight=0.0,
        fnn_normalize=False,
        fnn_elementwise_regularization=False,
        fnn_use_pca=False,
        fnn_n_samples=None,
        latent_noise_scale=0.0,
        latent_noise_per_step=False,
        precompute_latent_noise_factor=True,
        learn_r2_weight=False,
        learn_loop_closure_weight=False,
        learn_fnn_weight=False,
        learn_jac_cons_weight=False,
        learn_jac_norm_weight=False,
        log_var_init='naive',
        decode_only_recent=False,
        # Subspace splitting (for dimension-preserving encoders)
        n_target_dims=None,
        n_recent_dims=None,
        kl_null_weight=None,
        kl_dyn_weight=0.0,
        kl_divergence_weight=None,  # DEPRECATED — mapped to both kl weights for back-compat
        # VAE reparameterization on dynamic subspace
        use_vae=False,
        vae_sample_all_losses=False,
        kl_warmup_epochs=0,
        geometric_noise=None,
        jacobian_noise=None,
        # Reconstruction mode
        reconstruction_mode='uniform',
        # When True, the trajectory (rollout prediction) loss at TRAINING time
        # uses reconstruction_mode='most_recent' regardless of the
        # reconstruction_mode above, which is otherwise only honored for the
        # encoder-decoder reconstruction loss. This mirrors what validation
        # already hardcodes: trajectory is always evaluated most-recent-only
        # because that is the forecastable quantity (and because uniform
        # averaging dilutes the "genuinely hard" prediction signal by 1/n_delays).
        # Default True so new sweeps get the matched behaviour without opt-in.
        trajectory_loss_most_recent=True,
        # Tangent space entropy
        tangent_entropy_weight=0.0,
        tangent_entropy_mode='quadratic',
        tangent_entropy_n_samples=256,
        # Soft diffeomorphism cycle-consistency
        diffeomorphism_weight=0.0,
        # Eigenvalue diagnostics
        n_eigval_jacobians=None,
        # Step profiling.  profile_output=None disables entirely.
        # profile_output="stdout"        → print table to terminal / SLURM log
        # profile_output="wandb"         → log each component via self.log_dict()
        # profile_output="<path>.csv"    → append CSV rows to that file
        profile_output=None,
        profile_steps=1,   # print/log every N steps
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.encoder = encoder
        self.decode_only_recent = decode_only_recent
        self.prediction_steps = prediction_steps
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.dynamics_warmup_epochs = dynamics_warmup_epochs
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.latent_prediction_loss_weight = latent_prediction_loss_weight
        self.jac_consistency_weight = jac_consistency_weight
        self.fnn_weight = fnn_weight
        self.fnn_normalize = fnn_normalize
        self.fnn_elementwise_regularization = fnn_elementwise_regularization
        self.fnn_use_pca = fnn_use_pca
        self.fnn_n_samples = fnn_n_samples
        self.latent_noise_scale = latent_noise_scale
        self.latent_noise_per_step = latent_noise_per_step
        self.precompute_latent_noise_factor = precompute_latent_noise_factor
        self._latent_noise_scale_factor = None
        # Stride for JacobianODE sub-windows within each encoded batch.
        # Defaults to prediction_steps (non-overlapping).
        self.jac_window_stride = jac_window_stride if jac_window_stride is not None else prediction_steps

        # Subspace splitting for dimension-preserving encoders
        self.n_target_dims = n_target_dims
        self.n_recent_dims = n_recent_dims

        # KL weights — null (structural) and dyn (smoothness)
        if kl_divergence_weight is not None:
            # DEPRECATED back-compat: map old unified weight to both
            self.kl_null_weight = kl_divergence_weight
            self.kl_dyn_weight = kl_divergence_weight
        else:
            self.kl_dyn_weight = kl_dyn_weight
            # None → couple to kl_dyn_weight (single-knob control)
            self.kl_null_weight = kl_dyn_weight if kl_null_weight is None else kl_null_weight

        # VAE reparameterization on dynamic subspace
        self.use_vae = use_vae
        self.vae_sample_all_losses = vae_sample_all_losses
        self.kl_warmup_epochs = kl_warmup_epochs
        if use_vae:
            if n_target_dims is None:
                raise ValueError("use_vae=True requires n_target_dims to be set")
            self.log_var_proj = nn.Linear(n_target_dims, n_target_dims)
            nn.init.zeros_(self.log_var_proj.weight)
            nn.init.constant_(self.log_var_proj.bias, -6.0)  # sigma ≈ 0.05

        # Geometric noise: learned volume-preserving noise warping (z → u)
        self.geometric_noise_enabled = False
        if geometric_noise is not None and geometric_noise.get("enabled", False):
            if not use_vae:
                raise ValueError("geometric_noise requires use_vae=True")
            if n_target_dims is None:
                raise ValueError("geometric_noise requires n_target_dims")
            from JacobianODE.fnn.coupling_flows import AdditiveFlow
            self.noise_warp = AdditiveFlow(
                n_dims=n_target_dims,
                n_coupling_layers=geometric_noise.get("n_coupling_layers", 6),
                hidden_dim=geometric_noise.get("hidden_dim", 128),
                n_hidden_layers=geometric_noise.get("n_hidden_layers", 2),
                permutation_seed=geometric_noise.get("permutation_seed", 0),
            )
            self.geometric_noise_enabled = True

        # Jacobian noise: parameter-free volume-preserving noise routing
        self.jacobian_noise_enabled = False
        self.jacobian_noise_lookahead_steps = None
        if jacobian_noise is not None and jacobian_noise.get("enabled", False):
            if not use_vae:
                raise ValueError("jacobian_noise requires use_vae=True")
            if n_target_dims is None:
                raise ValueError("jacobian_noise requires n_target_dims")
            if self.geometric_noise_enabled:
                raise ValueError(
                    "jacobian_noise and geometric_noise are mutually exclusive"
                )
            self.jacobian_noise_enabled = True
            self.jacobian_noise_lookahead_steps = jacobian_noise.get(
                "lookahead_steps", None
            )
            self.jacobian_noise_orthogonalize = jacobian_noise.get(
                "orthogonalize", True
            )

        # Reconstruction mode
        if reconstruction_mode not in ('uniform', 'harmonic', 'most_recent'):
            raise ValueError(
                f"reconstruction_mode must be 'uniform', 'harmonic', or "
                f"'most_recent', got '{reconstruction_mode}'"
            )
        self.reconstruction_mode = reconstruction_mode
        self.trajectory_loss_most_recent = trajectory_loss_most_recent
        if reconstruction_mode == 'harmonic':
            obs_dim = encoder.n_latent
            w_raw = 1.0 / (torch.arange(obs_dim, dtype=torch.float32) + 1.0)
            w_norm = w_raw * (obs_dim / w_raw.sum())
            self.register_buffer('recon_weights', w_norm)

        # Tangent space entropy
        self.tangent_entropy_weight = tangent_entropy_weight
        self.tangent_entropy_mode = tangent_entropy_mode
        self.tangent_entropy_n_samples = tangent_entropy_n_samples

        # Soft diffeomorphism cycle-consistency
        self.diffeomorphism_weight = diffeomorphism_weight

        # Number of B×T Jacobian matrices to randomly sample for eigenvalue
        # computation during validation. None means use all.
        self.n_eigval_jacobians = n_eigval_jacobians

        # Step profiling
        self.profile_output = profile_output
        self.profile_steps = max(1, profile_steps)
        self._profile_log: list = []
        self._profile_step_count = 0

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

        # Homoscedastic uncertainty weighting (Kendall et al. 2018).
        # For each enabled group a learnable log-variance s_i is registered.
        # The weighted contribution becomes: 0.5 * exp(-s_i) * L_i + 0.5 * s_i
        self.learn_r2_weight = learn_r2_weight
        self.learn_loop_closure_weight = learn_loop_closure_weight
        self.learn_fnn_weight = learn_fnn_weight
        self.learn_jac_cons_weight = learn_jac_cons_weight
        self.learn_jac_norm_weight = learn_jac_norm_weight
        self.log_var_init = log_var_init
        self._log_var_auto_initialized = False

        if learn_r2_weight:
            self.log_var_r2 = nn.Parameter(torch.zeros(1))
        if learn_loop_closure_weight:
            self.log_var_loop_closure = nn.Parameter(torch.zeros(1))
        if learn_fnn_weight:
            self.log_var_fnn = nn.Parameter(torch.zeros(1))
        if learn_jac_cons_weight:
            self.log_var_jac_cons = nn.Parameter(torch.zeros(1))
        if learn_jac_norm_weight:
            self.log_var_jac_norm = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Step profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _timed(self, name: str):
        """Time a code block with GPU synchronization.

        A no-op when ``self.profile_output is None``.  Results accumulate in
        ``self._profile_log`` as ``(name, ms)`` pairs and are emitted by
        :meth:`_maybe_emit_profile` at the end of each step.
        """
        if self.profile_output is None:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._profile_log.append((name, (time.perf_counter() - t0) * 1000.0))

    def _maybe_emit_profile(self, label: str):
        """Emit a timing summary to the configured destination.

        Called at the end of each training/validation step.  Routes to:
        - ``"stdout"``      — formatted table printed to terminal / SLURM log
        - ``"wandb"``       — per-component ms logged via ``self.log_dict()``
        - any other string  — CSV rows appended to that file path
        """
        if self.profile_output is None or not self._profile_log:
            self._profile_log.clear()
            return
        self._profile_step_count += 1
        if self._profile_step_count % self.profile_steps != 0:
            self._profile_log.clear()
            return

        log = self._profile_log
        total = sum(dt for _, dt in log)

        if self.profile_output == "stdout":
            w = max(len(n) for n, _ in log)
            print(f"\n[PROFILE {label} | step {self._profile_step_count}]")
            for name, dt in log:
                pct = 100.0 * dt / total if total > 0 else 0.0
                print(f"  {name:<{w}s}  {dt:8.1f} ms  ({pct:4.1f}%)")
            print(f"  {'TOTAL':<{w}s}  {total:8.1f} ms")

        elif self.profile_output == "wandb":
            metrics = {f"profile/{label}/{name}": dt for name, dt in log}
            metrics[f"profile/{label}/TOTAL"] = total
            self.log_dict(metrics, on_step=True, on_epoch=False)

        else:  # treat as a file path
            import csv
            import os
            write_header = not os.path.exists(self.profile_output)
            with open(self.profile_output, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["step", "label", "component", "ms", "pct"])
                for name, dt in log:
                    pct = 100.0 * dt / total if total > 0 else 0.0
                    writer.writerow([self._profile_step_count, label, name,
                                     f"{dt:.3f}", f"{pct:.2f}"])
                writer.writerow([self._profile_step_count, label, "TOTAL",
                                 f"{total:.3f}", "100.00"])

        self._profile_log.clear()

    # ------------------------------------------------------------------
    # Encoder / decoder abstractions
    # ------------------------------------------------------------------

    @property
    def _n_recent_dims(self):
        """Number of raw observation dims (first delay coordinate).

        Derived at access time so it works with checkpoints saved before
        this property existed.  Explicit ``n_recent_dims`` (set by
        ``config.py``) takes priority over ``decoder.n_output``, which
        may equal the full delay-embedding dimension for
        dimension-preserving encoders (e.g. ``latent_mlp_diffeo``).
        """
        if getattr(self, 'n_recent_dims', None) is not None:
            return self.n_recent_dims
        if hasattr(self.encoder, 'decoder') and hasattr(self.encoder.decoder, 'n_output'):
            return self.encoder.decoder.n_output
        return None

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
    # Subspace splitting helpers
    # ------------------------------------------------------------------

    def _split_latent(self, z_full):
        """Split encoded z into dynamic and null subspaces.

        Returns ``(z_full, None)`` when ``n_target_dims is None``
        (backward-compatible no-op).
        """
        if self.n_target_dims is None:
            return z_full, None
        return z_full[..., :self.n_target_dims], z_full[..., self.n_target_dims:]

    def _pad_to_full_dim(self, z_dyn):
        """Zero-pad ``z_dyn`` back to full encoder dimension for decoding.

        No-op when ``n_target_dims is None``.
        """
        if self.n_target_dims is None:
            return z_dyn
        pad = z_dyn.new_zeros(
            *z_dyn.shape[:-1], self.encoder.n_latent - self.n_target_dims
        )
        return torch.cat([z_dyn, pad], dim=-1)

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    def _vae_reparameterize(self, mu_dyn):
        """Apply VAE reparameterization trick to dynamic subspace.

        Parameters
        ----------
        mu_dyn : torch.Tensor
            Mean of the dynamic latent, shape ``(..., n_target_dims)``.

        Returns
        -------
        z_dyn : torch.Tensor
            Sampled (training) or deterministic (eval) latent.
        log_var : torch.Tensor or None
            Log-variance, same shape as ``mu_dyn``. ``None`` when VAE is off.
        kl_mu : torch.Tensor or None
            Mean to use in KL computation.  When geometric noise is enabled
            this is ``u = f(mu_dyn)`` (noise-space mean); otherwise it equals
            ``mu_dyn``.  ``None`` when VAE is off.
        """
        if not self.use_vae:
            return mu_dyn, None, None

        # Geometric noise path: map to u-space, sample there, map back
        if self.geometric_noise_enabled and self.training:
            u = self.noise_warp(mu_dyn)
            log_var = self.log_var_proj(u)
            std = torch.exp(0.5 * log_var)
            u_noisy = u + std * torch.randn_like(std)
            z_dyn = self.noise_warp.inverse(u_noisy)
            return z_dyn, log_var, u

        # Jacobian noise path: route isotropic noise through volume-preserving
        # matrix derived from the K-step lookahead traceless Jacobians.
        # KL is computed in U-space (on mu_dyn) — deliberate design choice.
        if self.jacobian_noise_enabled and self.training:
            A = self._compute_routing_matrices(mu_dyn)     # (..., T, D, D)
            log_var = self.log_var_proj(mu_dyn)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            noise = std * eps                               # (..., T, D)
            routed_noise = torch.einsum('...ij,...j->...i', A, noise)
            z_dyn = mu_dyn + routed_noise
            return z_dyn, log_var, mu_dyn

        # Standard VAE path (no geometric/jacobian noise, or eval mode)
        log_var = self.log_var_proj(mu_dyn)
        if self.training:
            std = torch.exp(0.5 * log_var)
            z_dyn = mu_dyn + std * torch.randn_like(std)
        else:
            z_dyn = mu_dyn
        return z_dyn, log_var, mu_dyn

    @staticmethod
    def _kl_divergence(mu, log_var):
        """Standard Gaussian KL divergence: KL(q(z|x) || N(0, I)).

        Returns a scalar (mean over all elements).
        """
        return -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

    def _effective_kl_weights(self):
        """Return (effective_null_weight, effective_dyn_weight) after warmup."""
        if self.kl_warmup_epochs <= 0:
            return self.kl_null_weight, self.kl_dyn_weight
        ramp = min(self.current_epoch / self.kl_warmup_epochs, 1.0)
        return self.kl_null_weight * ramp, self.kl_dyn_weight * ramp

    def _compute_routing_matrices(self, mu_dyn):
        """Compute K-step lookahead volume-preserving routing matrices.

        Uses the Jacobian MLP (detached) to build a matrix that routes noise
        into the stable manifold of the dynamics.  Each single-step matrix
        ``E_t = exp(-tilde{J}_t dt)`` is volume-preserving because
        ``tilde{J}_t`` is traceless, and their product preserves this property.

        Parameters
        ----------
        mu_dyn : torch.Tensor
            Deterministic encoder means, shape ``(B, T, D)`` or ``(T, D)``.

        Returns
        -------
        A : torch.Tensor
            Routing matrices, same leading shape as *mu_dyn* plus ``(D, D)``.
        """
        D = mu_dyn.shape[-1]
        T = mu_dyn.shape[-2]
        K = self.jacobian_noise_lookahead_steps or self.prediction_steps

        # 1. Jacobians at all points — detached so prediction loss doesn't
        #    alter the vector field just to satisfy noise routing.
        with torch.no_grad():
            J = self.compute_jacobians(mu_dyn)          # (..., T, D, D)

        # 2. Make traceless: J_tilde = J - tr(J)/D * I
        trace_J = torch.diagonal(J, dim1=-2, dim2=-1).sum(-1)  # (..., T)
        eye = torch.eye(D, device=mu_dyn.device, dtype=mu_dyn.dtype)
        J_tilde = J - (trace_J / D)[..., None, None] * eye

        # 3. Single-step backward matrices: E_t = exp(-J_tilde_t * dt)
        E = torch.linalg.matrix_exp(-J_tilde * self.dt)  # (..., T, D, D)

        # 4. Rolling window product: A_t = E_t @ E_{t+1} @ ... @ E_{t+K_eff-1}
        #    Start with A = E (K=1), then progressively multiply in future steps.
        #    For timesteps near the end where fewer than K future steps exist,
        #    A retains the partial product (K_eff < K).
        A = E.clone()
        for k in range(1, K):
            t_end = T - k
            if t_end <= 0:
                break
            A[..., :t_end, :, :] = A[..., :t_end, :, :] @ E[..., k:k + t_end, :, :]

        # 5. Optionally orthogonalize via polar decomposition: A = U @ P.
        #    Keep only the orthogonal factor U = U_svd @ Vh so that
        #    ||A @ noise|| = ||noise|| (pure rotation, no scaling).
        if self.jacobian_noise_orthogonalize:
            U_svd, _, Vh_svd = torch.linalg.svd(A)
            A = U_svd @ Vh_svd

        return A

    def _tangent_space_entropy_loss(self, batch, z_full):
        """Tangent space entropy using **encoder** Jacobians (dz/dx).

        Projects latent velocity ``dz = z_{t+1} - z_t`` onto the encoder
        Jacobian's left singular vectors U (computed at x_t, **detached**),
        then minimises the entropy of the per-dimension energy distribution.

        For coupling encoders (``n_target_dims is not None``), only the
        dynamic subspace ``z_dyn`` is used for ``dz`` and only the
        corresponding Jacobian rows are computed.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.
        z_full : torch.Tensor
            Full encoded latent ``(B, T', D_latent)``.

        Returns
        -------
        torch.Tensor
            Scalar loss (zero if T' < 2).
        """
        z_dyn, _ = self._split_latent(z_full)
        n_dyn = z_dyn.shape[-1]
        B, T, _ = z_dyn.shape
        if T < 2:
            return z_dyn.new_zeros(1).squeeze()

        dz = z_dyn[:, 1:, :] - z_dyn[:, :-1, :]  # (B, T-1, n_dyn)

        # Build the observation tensor that aligns with each dz timestep.
        if hasattr(self.encoder, 'time_window'):
            w = self.encoder.time_window
            n_obs = w * batch.shape[-1]
            windows = batch.unfold(1, w, 1).permute(0, 1, 3, 2)  # (B, T', w, D)
            x_for_jac = windows[:, :-1].reshape(B, T - 1, -1)    # (B, T-1, w*D)
        else:
            margin = getattr(self.encoder, 'context_margin', 0)
            n_obs = batch.shape[-1]
            x_for_jac = batch[:, margin:margin + T - 1, :]       # (B, T-1, D_obs)

        dz_flat = dz.reshape(-1, n_dyn)
        x_flat = x_for_jac.reshape(-1, n_obs)
        N = dz_flat.shape[0]
        if N == 0:
            return z_dyn.new_zeros(1).squeeze()

        # Subsample
        if self.tangent_entropy_n_samples is not None and N > self.tangent_entropy_n_samples:
            idx = torch.randperm(N, device=z_dyn.device)[:self.tangent_entropy_n_samples]
            x_sample = x_flat[idx]
            dz_sample = dz_flat[idx]
        else:
            x_sample = x_flat
            dz_sample = dz_flat

        # Single-point encoder function for autograd Jacobian computation.
        # For coupling encoders, return only z_dyn dims so the Jacobian is
        # (n_dyn, n_obs) rather than (n_latent, n_obs).
        n_target = self.n_target_dims  # None for non-coupling
        if hasattr(self.encoder, 'time_window'):
            w = self.encoder.time_window
            D = batch.shape[-1]

            def _encode_point(x_flat_pt):
                z = self.encoder.encode(x_flat_pt.reshape(1, w, D)).squeeze(0)
                return z[:n_target] if n_target is not None else z
        else:
            def _encode_point(x_pt):
                z = self.encoder.encode(
                    x_pt.unsqueeze(0).unsqueeze(0)
                ).squeeze(0).squeeze(0)
                return z[:n_target] if n_target is not None else z

        # Pick jacrev vs jacfwd based on output vs input dimension.
        if n_dyn <= n_obs:
            jac_fn = torch.func.jacrev(_encode_point)
        else:
            jac_fn = torch.func.jacfwd(_encode_point)

        was_training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                J_all = torch.func.vmap(jac_fn)(x_sample)  # (M, n_dyn, n_obs)
        finally:
            if was_training:
                self.encoder.train()

        return tangent_space_entropy(dz_sample, J_all, mode=self.tangent_entropy_mode)

    def _weighted_obs_loss(self, targets, predictions, mode=None):
        """Observation-space loss with ``reconstruction_mode`` weighting.

        Applied to both reconstruction and trajectory prediction losses.
        Uses ``self.criterion`` (configured via ``loss_func``) as the
        underlying loss function, so the scale is consistent with other
        loss terms (e.g. loop closure).

        Parameters
        ----------
        mode : Optional[str]
            Override for ``self.reconstruction_mode``. Primary use case:
            validation passes ``mode='most_recent'`` so the monitored val
            loss reflects forward-rollout fidelity on the live frame,
            independent of what training uses for its gradient.
        """
        effective = mode if mode is not None else self.reconstruction_mode
        if effective == 'harmonic':
            # sq_err = F.mse_loss(predictions, targets, reduction='none')
            # TODO: fix this because normalized_mse doesn't support reduction='none'
            sq_err = self.criterion(targets, predictions)
            return (sq_err * self.recon_weights).mean()
        elif effective == 'most_recent':
            d = self._n_recent_dims
            if d is not None:
                return self.criterion(targets[..., :d], predictions[..., :d])
            return self.criterion(targets, predictions)
        else:  # uniform
            return self.criterion(targets, predictions)

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
                              traj_init_steps, prediction_steps=None):
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
        prediction_steps : int, optional
            Number of steps to predict. Defaults to ``self.prediction_steps``.
            Used when ``strided=False`` to predict the full trajectory remainder.

        return_latent : bool
            Whether to return the latent targets instead of the observation targets.

        Returns
        -------
        torch.Tensor
            Observation targets.
        """
        prediction_steps = prediction_steps if prediction_steps is not None else self.prediction_steps
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
            targets = torch.stack(targets)  # (N, prediction_steps, w, D_obs)
            if self.decode_only_recent:
                targets = targets[..., :self.encoder.decoder.n_output]
            return targets
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
            targets = torch.stack(targets)  # (N, prediction_steps, D_obs)
            if self.decode_only_recent:
                targets = targets[..., :self.encoder.decoder.n_output]
            return targets

    def trajectory_model_step(
        self,
        batch,
        batch_idx=0,
        dataloader_idx=0,
        all_metrics=False,
        direct=None,
        obs_noise_scale=None,
        latent_noise_scale=None,
        latent_noise_per_step=None,
        alpha_teacher_forcing=None,
        teacher_forcing_steps=None,
        jacobianODEint_kwargs=None,
        criterion=None,
        verbose=False,
        return_decoded=False,
        strided=True,
        reconstruction_mode=None,
    ):
        """Trajectory prediction step in latent space.

        1. Encode full observation sequence to latent trajectory.
        2. Extract sub-windows: if strided=True, use strided sub-windows (multiple
           per batch element); if strided=False, use one full-length window per
           batch element.
        3. Integrate forward in latent space via JacobianODEint.
        4. Decode predicted latent vectors to observation space.
        5. Compute loss against raw observation targets (not decoded true z).
        """
        if obs_noise_scale is None:
            obs_noise_scale = self.obs_noise_scale
        if latent_noise_scale is None:
            latent_noise_scale = self.latent_noise_scale
        if latent_noise_per_step is None:
            latent_noise_per_step = self.latent_noise_per_step
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

        # 1. Encode full observation sequence
        with self._timed("traj/1.encode"):
            # Add observation noise during training
            scaled_noise = obs_noise_scale * self.noise_scale_factor
            batch_noisy = batch + (torch.randn_like(batch) * scaled_noise)

            z_full = self.encode_trajectory(batch_noisy)  # (B, T', D_latent)

            # Split into dynamic subspace and null subspace
            mu_dyn, _z_null = self._split_latent(z_full)
            z_dyn_sampled, _, _ = self._vae_reparameterize(mu_dyn)
            z_dyn = z_dyn_sampled if self.vae_sample_all_losses else mu_dyn
            # Latent prediction target. When obs noise is injected, encoding
            # the noisy batch produces a noisy version of the true latent
            # trajectory — a poor regression target for latent_pred_loss.
            # Re-encode the *clean* observations and use that as the target,
            # detached so gradients from latent_pred_loss flow only through
            # the noisy-encoder + dynamics path (the denoising-via-rollout
            # signal), not through the target itself.
            if obs_noise_scale > 0:
                with torch.no_grad():
                    z_full_clean = self.encode_trajectory(batch)
                    mu_dyn_clean, _ = self._split_latent(z_full_clean)
                z_dyn_clean = mu_dyn_clean
            else:
                z_dyn_clean = mu_dyn  # target is always the mean, not the sample

            # Add isotropic noise in latent space before propagation.
            if latent_noise_scale > 0:
                if self._latent_noise_scale_factor is not None:
                    factor = self._latent_noise_scale_factor
                else:
                    factor = z_dyn.detach().norm(dim=-1).mean() / math.sqrt(z_dyn.shape[-1])
                if latent_noise_per_step:
                    noise = torch.randn_like(z_dyn) * latent_noise_scale * factor
                else:
                    noise = torch.randn(
                        z_dyn.shape[0], 1, z_dyn.shape[-1],
                        device=z_dyn.device, dtype=z_dyn.dtype,
                    ) * latent_noise_scale * factor
                z_dyn = z_dyn + noise

        # 2. Determine sub-window parameters and gather windows
        with self._timed("traj/2.window_gather"):
            traj_init_steps = jacobianODEint_kwargs.get('traj_init_steps', 15)
            B, T_prime, D_dyn = z_dyn.shape

            if strided:
                jac_window_len = traj_init_steps + self.prediction_steps
                prediction_steps_actual = self.prediction_steps
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
                n_windows_actual = len(all_starts) // B
                start_indices = torch.tensor(
                    all_starts, device=z_dyn.device, dtype=torch.long
                )
            else:
                # Non-strided: one full-length window per batch element
                jac_window_len = T_prime
                prediction_steps_actual = T_prime - traj_init_steps
                if prediction_steps_actual <= 0:
                    raise ValueError(
                        f"Latent trajectory length ({T_prime}) too short for "
                        f"traj_init_steps={traj_init_steps}. Need T' > traj_init_steps."
                    )
                n_windows_actual = 1
                start_indices = torch.zeros(B, device=z_dyn.device, dtype=torch.long)

        # traj/2.window_gather continues — stack the sub-windows
        with self._timed("traj/2b.window_stack"):
            z_windows_list = []
            z_clean_windows_list = []
            for idx in range(start_indices.shape[0]):
                b = idx // n_windows_actual
                s = start_indices[idx]
                z_windows_list.append(z_dyn[b, s:s + jac_window_len])
                z_clean_windows_list.append(z_dyn_clean[b, s:s + jac_window_len])
            z_windows = torch.stack(z_windows_list)  # (N, jac_window_len, D_dyn)
            z_clean_windows = torch.stack(z_clean_windows_list)

        # 3. Run JacobianODEint on z_dyn sub-windows
        with self._timed("traj/3.jacobianODEint"):
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
            )  # (N, jac_window_len, D_dyn)

        # 4 & 5. Crop, decode, extract obs targets, compute losses
        with self._timed("traj/4.decode"):
            z_pred_crop = z_pred[..., traj_init_steps:, :]  # (N, prediction_steps, D_dyn)
            z_true_crop = z_clean_windows[:, traj_init_steps:, :]
            z_pred_padded = self._pad_to_full_dim(z_pred_crop)
            decoded_pred = self.decode_trajectory(z_pred_padded)

        with self._timed("traj/5.obs_targets"):
            obs_targets = self._extract_obs_targets(
                label, start_indices, n_windows_actual, traj_init_steps,
                prediction_steps=prediction_steps_actual,
            )  # same shape as decoded_pred

        with self._timed("traj/6.loss_and_metrics"):
            # Observation-space loss with reconstruction_mode weighting.
            # When ``reconstruction_mode`` kwarg is given (e.g. by
            # validation_step as 'most_recent'), it overrides
            # self.reconstruction_mode for THIS call only — so training can
            # use a wider-window mode (e.g. 'uniform') while validation's
            # monitored loss reflects forward-rollout fidelity on the live
            # frame.
            effective_mode = (
                reconstruction_mode if reconstruction_mode is not None
                else self.reconstruction_mode
            )
            loss = self._weighted_obs_loss(
                obs_targets, decoded_pred, mode=effective_mode,
            )

            # Metrics — when most_recent mode, evaluate on index 0 only so
            # that the unsupervised chaotic tail doesn't corrupt diagnostics.
            metric_vals = {}
            if effective_mode == 'most_recent':
                d = self._n_recent_dims
                if d is not None:
                    obs_for_metrics = obs_targets[..., :d]
                    dec_for_metrics = decoded_pred[..., :d]
                else:
                    obs_for_metrics = obs_targets
                    dec_for_metrics = decoded_pred
            else:
                obs_for_metrics = obs_targets
                dec_for_metrics = decoded_pred

            with torch.no_grad():
                # Flatten window dims for scalar metrics: (N, T, w, D) → (N, T, w*D)
                obs_m = obs_for_metrics.reshape(obs_for_metrics.shape[0], obs_for_metrics.shape[1], -1) if obs_for_metrics.dim() > 3 else obs_for_metrics
                dec_m = dec_for_metrics.reshape(dec_for_metrics.shape[0], dec_for_metrics.shape[1], -1) if dec_for_metrics.dim() > 3 else dec_for_metrics
                metric_vals['mase'] = mase(obs_m, dec_m)
                # Store raw MAE components so callers can aggregate correctly
                # (ratio-of-means instead of mean-of-ratios).
                metric_vals['model_mae'] = torch.mean(torch.abs(obs_m - dec_m))
                if obs_m.dim() == 3:
                    metric_vals['persistence_mae'] = torch.mean(
                        torch.abs(obs_m[:, 1:] - obs_m[:, :-1]))
                else:
                    metric_vals['persistence_mae'] = torch.mean(
                        torch.abs(obs_m[1:] - obs_m[:-1]))
                pred_flat = dec_m.reshape(dec_m.shape[0], -1)
                tgt_flat = obs_m.reshape(obs_m.shape[0], -1)
                metric_vals['r2_score'] = r2_score(tgt_flat, pred_flat)

            # Latent prediction loss in z_dyn space (computed outside no_grad so
            # gradients flow back through the encoder). Uses latent_criterion,
            # which may differ from criterion when gen_variance_mode='adaptive_latent'
            # (denominator recomputed from Cov(z_dyn) each epoch).
            latent_pred_loss = self.latent_criterion(z_true_crop, z_pred_crop)
            metric_vals['latent_pred_loss'] = latent_pred_loss
            with torch.no_grad():
                z_pred_flat = z_pred_crop.reshape(z_pred_crop.shape[0], -1)
                z_true_flat = z_true_crop.reshape(z_true_crop.shape[0], -1)
                metric_vals['latent_pred_r2'] = r2_score(z_true_flat, z_pred_flat)

        if return_decoded:
            return {'loss': loss, 'metric_vals': metric_vals, 'outputs': z_pred, 'decoded': decoded_pred, 'targets': obs_targets}
        else:
            return {'loss': loss, 'metric_vals': metric_vals, 'outputs': z_pred}

    # ------------------------------------------------------------------
    # Reconstruction loss
    # ------------------------------------------------------------------

    def _reconstruction_loss(self, batch, z_full=None):
        """Compute observation-space reconstruction loss: decode(encode(x)) ≈ x.

        When ``n_target_dims`` is set, only the dynamic subspace is used for
        decoding (padded with zeros), and the loss weighting follows
        ``reconstruction_mode``.

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
            Scalar reconstruction loss.
        """
        if z_full is None:
            z_full = self.encode_trajectory(batch)

        z_dyn, _ = self._split_latent(z_full)
        z_padded = self._pad_to_full_dim(z_dyn)
        recon_decoded = self.decode_trajectory(z_padded)

        if hasattr(self.encoder, 'time_window'):
            w = self.encoder.time_window
            B, T, D = batch.shape
            recon_targets = batch.unfold(1, w, 1).permute(0, 1, 3, 2)  # (B, T', w, D)
        else:
            margin = getattr(self.encoder, 'context_margin', 0)
            recon_targets = batch[:, margin:, :] if margin > 0 else batch

        if self.decode_only_recent:
            recon_targets = recon_targets[..., :self.encoder.decoder.n_output]

        return self._weighted_obs_loss(recon_targets, recon_decoded)

    def _diffeomorphism_loss(self, batch, z_full):
        """Soft diffeomorphism penalty: encourage E and D to be approximate inverses.

        Forward cycle: normalized_mse(x, D(E(x))) — full reconstruction through
        the ENTIRE latent space (no subspace splitting or zero-padding).
        Backward cycle: normalized_mse(z, E(D(z))) — re-encoding consistency.

        Unlike ``_reconstruction_loss``, this operates on the full z (not
        split/padded) so that it penalizes information loss in ALL latent
        dimensions.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.
        z_full : torch.Tensor
            Full latent trajectory ``(B, T', D_latent)`` from
            ``encode_trajectory``.

        Returns
        -------
        torch.Tensor
            Scalar loss (average of forward and backward cycle losses).
        """
        if hasattr(self.encoder, 'time_window'):
            raise NotImplementedError(
                "Diffeomorphism loss is not supported for window-based encoders"
            )

        # Forward cycle: D(E(x)) should reconstruct x
        x_recon = self.decode_trajectory(z_full)  # (B, T', D_obs)

        margin = getattr(self.encoder, 'context_margin', 0)
        x_target = batch[:, margin:, :] if margin > 0 else batch

        forward_loss = normalized_mse(x_target, x_recon)

        # Backward cycle: E(D(z)) should recover z
        z_reencoded = self.encoder.encode(x_recon)  # (B, T', D_latent)
        backward_loss = normalized_mse(z_full, z_reencoded)

        return 0.5 * (forward_loss + backward_loss)

    def _jac_consistency_loss(self, z, jacs):
        """Jacobian-consistency loss in latent space.

        Measures how well the learned Jacobians propagate velocity vectors
        forward via the variational equation: if dz/dt = f(z), then a
        displacement dz evolves as d(dz)/dt = J dz, so
        dz(t+dt) ≈ e^{J dt} dz(t).

        Loss: ||e^{J dt}(z_{t+1}-z_t) - (z_{t+2}-z_{t+1})||^2 / var(z)

        Parameters
        ----------
        z : torch.Tensor
            Latent trajectory ``(B, T, D_latent)``.
        jacs : torch.Tensor
            Predicted Jacobians ``(B, T, D_latent, D_latent)``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        J_exp = torch.matrix_exp(jacs[:, :-2] * self.dt)          # (B, T-2, D, D)
        vel = z[:, 1:] - z[:, :-1]                                 # (B, T-1, D)
        vel_t1_pred = (J_exp @ vel[:, :-1].unsqueeze(-1)).squeeze(-1)  # (B, T-2, D)
        return (vel[:, 1:] - vel_t1_pred).pow(2).mean() / z.var()

    # ------------------------------------------------------------------
    # Homoscedastic uncertainty weighting helpers
    # ------------------------------------------------------------------

    def _homoscedastic_loss(self, loss, log_var):
        """Apply homoscedastic uncertainty weighting to a loss term.

        Computes ``0.5 * exp(-log_var) * loss + 0.5 * log_var``.
        The first term down-weights large losses; the second acts as a
        learned-regularisation penalty that prevents log_var from growing
        without bound.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar (or broadcastable) raw loss value.
        log_var : nn.Parameter
            Learnable log-variance scalar ``s_i``.
        """
        return 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var

    def _auto_init_log_vars(self, r2_loss=None, loop_loss=None, fnn_loss=None,
                             jac_cons_loss=None, jac_norm=None):
        """Initialise log_var parameters to log of initial loss magnitudes.

        Sets each active ``log_var`` parameter to ``log(raw_loss)`` so that
        the first-step gradient contribution is normalised and no extreme
        spikes occur.  Called once, on the first training batch.

        Parameters
        ----------
        r2_loss, loop_loss, fnn_loss, jac_cons_loss, jac_norm : Tensor or None
            Raw (unweighted) loss values for each group.  Groups whose
            corresponding flag is False are ignored.
        """
        eps = 1e-8
        pairs = [
            (self.learn_r2_weight,           getattr(self, 'log_var_r2', None),           r2_loss),
            (self.learn_loop_closure_weight, getattr(self, 'log_var_loop_closure', None), loop_loss),
            (self.learn_fnn_weight,          getattr(self, 'log_var_fnn', None),          fnn_loss),
            (self.learn_jac_cons_weight,     getattr(self, 'log_var_jac_cons', None),     jac_cons_loss),
            (self.learn_jac_norm_weight,     getattr(self, 'log_var_jac_norm', None),     jac_norm),
        ]
        with torch.no_grad():
            for enabled, param, raw in pairs:
                if enabled and param is not None and raw is not None:
                    val = float(raw.detach()) if isinstance(raw, torch.Tensor) else float(raw)
                    if val > eps:
                        param.data.fill_(torch.tensor(val).log().item())
        self._log_var_auto_initialized = True

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _warmup_step(self, batch):
        """Encoder warmup: reconstruction + encoder-side regularisers.

        Encodes observation windows to latent, decodes back, and computes
        MSE against the original windows.  No Jacobian prediction or loop
        closure.  Tangent-space entropy (using encoder Jacobians) and
        zero-padding penalty (for coupling encoders) are included so
        the latent geometry is shaped from the start.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.

        Returns
        -------
        torch.Tensor
            Scalar total warmup loss.
        """
        batch = batch.type(self.dtype)

        z_full = self.encode_trajectory(batch)
        mu_dyn, z_null = self._split_latent(z_full)
        z_dyn_sampled, log_var, kl_mu = self._vae_reparameterize(mu_dyn)

        # Reconstruction uses sampled z_dyn when VAE is active
        if self.use_vae and z_null is not None:
            z_full_for_recon = torch.cat([z_dyn_sampled, z_null], dim=-1)
        else:
            z_full_for_recon = z_full
        recon_loss = self._reconstruction_loss(batch, z_full_for_recon)
        loss = recon_loss

        # KL divergence: null penalty (structural) + optional VAE KL (smoothness)
        kl_null_loss = None
        kl_dyn_loss = None
        eff_null_w, eff_dyn_w = self._effective_kl_weights()
        if self.n_target_dims is not None:
            if z_null is not None and z_null.numel() > 0:
                kl_null_loss = F.mse_loss(z_null, torch.zeros_like(z_null))
            else:
                kl_null_loss = torch.tensor(0.0, device=batch.device)
            if self.use_vae and log_var is not None:
                kl_dyn_loss = self._kl_divergence(kl_mu, log_var)
            else:
                kl_dyn_loss = torch.tensor(0.0, device=batch.device)
            if eff_null_w > 0 and kl_null_loss is not None:
                loss = loss + eff_null_w * kl_null_loss
            if eff_dyn_w > 0 and kl_dyn_loss is not None:
                loss = loss + eff_dyn_w * kl_dyn_loss

        # Tangent space entropy (encoder Jacobians via autograd)
        tangent_entropy_loss = None
        if self.tangent_entropy_weight > 0:
            tangent_entropy_loss = self._tangent_space_entropy_loss(batch, z_full)
            loss = loss + self.tangent_entropy_weight * tangent_entropy_loss

        # Diffeomorphism cycle-consistency (uses sampled z when VAE is active)
        diffeo_loss = None
        if self.diffeomorphism_weight > 0:
            diffeo_loss = self._diffeomorphism_loss(batch, z_full_for_recon)
            loss = loss + self.diffeomorphism_weight * diffeo_loss

        log_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("warmup recon_loss", recon_loss, **log_kwargs)
        if kl_null_loss is not None:
            self.log("warmup kl_null_loss", kl_null_loss, **log_kwargs)
        if kl_dyn_loss is not None:
            self.log("warmup kl_dyn_loss", kl_dyn_loss, **log_kwargs)
        if tangent_entropy_loss is not None:
            self.log("warmup tangent_entropy_loss", tangent_entropy_loss, **log_kwargs)
        if diffeo_loss is not None:
            self.log("warmup diffeo_loss", diffeo_loss, **log_kwargs)
        return loss

    def _dynamics_warmup_step(self, batch):
        """Dynamics warmup: full training step with encoder frozen.

        Runs the standard training step but with encoder parameters frozen
        so that only the Jacobian model (dynamics) adapts to the latent
        space discovered during encoder warmup.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations ``(B, T, D_obs)``.

        Returns
        -------
        torch.Tensor
            Scalar total loss (same as full training step).
        """
        # Freeze encoder
        encoder_grad_state = {}
        for name, param in self.encoder.named_parameters():
            encoder_grad_state[name] = param.requires_grad
            param.requires_grad = False

        try:
            loss = self._full_training_step(batch)
        finally:
            # Restore encoder grad state
            for name, param in self.encoder.named_parameters():
                param.requires_grad = encoder_grad_state[name]

        return loss

    # Number of batches sampled to estimate Cov(z_dyn) for adaptive-latent mode.
    ADAPTIVE_LATENT_N_BATCHES = 50

    def on_train_epoch_start(self):
        """Recompute the latent-space generalized variance if in adaptive mode.

        Processes a small number of training batches through the (possibly
        partially-trained) encoder to estimate ``det(Cov(z_dyn))^(1/D_dyn)``.
        That scalar becomes the denominator for the latent prediction loss
        over the upcoming epoch.
        """
        super().on_train_epoch_start()
        mode = getattr(self, '_gen_variance_mode', 'fixed')
        if mode != 'adaptive_latent':
            return

        train_dl = self.trainer.train_dataloader
        if train_dl is None:
            return

        was_training = self.training
        self.eval()
        z_samples = []
        with torch.no_grad():
            for i, batch in enumerate(train_dl):
                if i >= self.ADAPTIVE_LATENT_N_BATCHES:
                    break
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.type(self.dtype).to(self.device)
                z_full = self.encode_trajectory(batch)
                z_dyn, _ = self._split_latent(z_full)
                z_samples.append(z_dyn.reshape(-1, z_dyn.shape[-1]))

        if was_training:
            self.train()

        if not z_samples:
            return

        from ..jacobians.metrics import compute_generalized_variance
        all_z = torch.cat(z_samples, dim=0)
        new_denom = compute_generalized_variance(all_z)
        self.latent_criterion.set_denom(new_denom)
        self.log(
            "latent_gen_var", new_denom,
            on_step=False, on_epoch=True, sync_dist=True,
        )

    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Full training step: encode, predict, decode, loop closure.

        During the first ``encoder_warmup_epochs``, only the autoencoder
        reconstruction loss is used (no Jacobian prediction or loop closure).
        During the next ``dynamics_warmup_epochs``, the encoder is frozen
        and only the Jacobian model is trained.  Joint training begins at
        epoch ``encoder_warmup_epochs + dynamics_warmup_epochs``.

        Parameters
        ----------
        batch : torch.Tensor
            Raw observations of shape ``(B, T, D_obs)``.
        """
        # Encoder warmup phase: reconstruction only
        if self.current_epoch < self.encoder_warmup_epochs:
            return self._warmup_step(batch)

        # Dynamics warmup phase: encoder frozen, full training step
        if self.current_epoch < self.encoder_warmup_epochs + self.dynamics_warmup_epochs:
            return self._dynamics_warmup_step(batch)

        return self._full_training_step(batch, batch_idx, dataloader_idx)

    def _full_training_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Core training logic shared by ``training_step`` and ``_dynamics_warmup_step``."""
        batch = batch.type(self.dtype)

        # Encode for teacher forcing update (use z_dyn for Jacobians)
        with self._timed("train/1.encode_tf+compute_jacs"):
            with torch.no_grad():
                z_for_tf = self.encode_trajectory(batch)
                z_for_tf_dyn, _ = self._split_latent(z_for_tf)
                jacs_pred = self.compute_jacobians(z_for_tf_dyn)
            jac_norm = torch.linalg.norm(
                jacs_pred, dim=(-2, -1), ord=self.jac_norm_ord
            ).mean()
            self.update_alpha_teacher_forcing(jacs_pred.detach(), batch_idx)

        train_rets = {}

        # Trajectory prediction loss. When trajectory_loss_most_recent is True,
        # override the reconstruction_mode just for this term — matches val's
        # hardcoded 'most_recent' so train/val diagnose the same quantity, and
        # avoids the 1/n_delays dilution of genuine forecast error under uniform.
        with self._timed("train/2.trajectory_step"):
            if self.trajectory_training:
                traj_kwargs = {}
                if self.trajectory_loss_most_recent:
                    traj_kwargs['reconstruction_mode'] = 'most_recent'
                train_rets['trajectory'] = self.trajectory_model_step(
                    batch, batch_idx, dataloader_idx, **traj_kwargs
                )

        # Encode once for loop closure, reconstruction, and regularizers
        with self._timed("train/3.encode_recon"):
            z_full = self.encode_trajectory(batch)
            mu_dyn, z_null = self._split_latent(z_full)
            z_dyn_sampled, log_var, kl_mu = self._vae_reparameterize(mu_dyn)
            # Choose which z_dyn downstream losses see
            z_dyn = z_dyn_sampled if self.vae_sample_all_losses else mu_dyn

        # Loop closure in latent space (operates on z_dyn)
        with self._timed("train/4.loop_closure"):
            if self.loop_closure_training:
                train_rets['loop_closure'] = self.loop_closure_model_step(
                    z_dyn, batch_idx, dataloader_idx
                )

        # ----------------------------------------------------------------
        # Compute per-group losses
        # ----------------------------------------------------------------

        # --- Group 1: R²-like (trajectory + recon + latent_pred) ---
        r2_loss = torch.zeros(1, device=batch.device, dtype=batch.dtype).squeeze()
        if self.trajectory_training and 'trajectory' in train_rets:
            traj_loss = train_rets['trajectory']['loss']
            if torch.isnan(traj_loss):
                print(
                    f"Warning: Loss is nan for pred type trajectory "
                    f"on epoch {self.current_epoch} batch {batch_idx}"
                )
            else:
                r2_loss = r2_loss + traj_loss

        # Build z_full with sampled z_dyn for reconstruction & diffeomorphism
        if self.use_vae and z_null is not None:
            z_full_for_recon = torch.cat([z_dyn_sampled, z_null], dim=-1)
        else:
            z_full_for_recon = z_full

        # Reconstruction loss: decode(encode(x)) ≈ x in observation space
        recon_loss = None
        with self._timed("train/5.reconstruction_loss"):
            if self.reconstruction_loss_weight > 0:
                recon_loss = self._reconstruction_loss(batch, z_full=z_full_for_recon)
                r2_loss = r2_loss + self.reconstruction_loss_weight * recon_loss

        # Latent prediction loss: JacobianODE(z_t) ≈ z_{t+k} in latent space
        latent_pred_loss = None
        if self.latent_prediction_loss_weight > 0 and self.trajectory_training:
            latent_pred_loss = train_rets['trajectory']['metric_vals'].get('latent_pred_loss')
            if latent_pred_loss is not None:
                r2_loss = r2_loss + self.latent_prediction_loss_weight * latent_pred_loss

        # --- Group 2: Loop closure ---
        loop_loss = None
        if self.loop_closure_training and 'loop_closure' in train_rets:
            lc = train_rets['loop_closure']['loss']
            if torch.isnan(lc):
                print(
                    f"Warning: Loss is nan for pred type loop_closure "
                    f"on epoch {self.current_epoch} batch {batch_idx}"
                )
            else:
                loop_loss = lc

        # --- Group 3: FNN regularization (on z_dyn) ---
        fnn_loss = None
        with self._timed("train/6.fnn_loss"):
            if self.fnn_weight > 0 or self.learn_fnn_weight:
                z_flat = z_dyn.reshape(-1, z_dyn.shape[-1])
                fnn_loss = loss_false(
                    z_flat,
                    normalize=self.fnn_normalize,
                    elementwise_regularization=self.fnn_elementwise_regularization,
                    use_pca=self.fnn_use_pca,
                    n_samples=self.fnn_n_samples,
                )

        # --- Group 4: Jac-consistency (on z_dyn) ---
        jac_cons_loss = None
        jacs_for_cons = None
        with self._timed("train/7.jac_consistency"):
            if self.jac_consistency_weight > 0 or self.learn_jac_cons_weight:
                jacs_for_cons = self.compute_jacobians(z_dyn)
                jac_cons_loss = self._jac_consistency_loss(z_dyn, jacs_for_cons)

        # --- Group 5: Jac norm/penalty ---
        # (jac_norm is already computed above for teacher-forcing; reused here)

        # --- Group 6: KL divergence (null + optional VAE dyn, separate weights) ---
        kl_null_loss = None
        kl_dyn_loss = None
        eff_null_w, eff_dyn_w = self._effective_kl_weights()
        if self.n_target_dims is not None:
            if z_null is not None and z_null.numel() > 0:
                kl_null_loss = F.mse_loss(z_null, torch.zeros_like(z_null))
            else:
                kl_null_loss = torch.tensor(0.0, device=batch.device)
            if self.use_vae and log_var is not None:
                kl_dyn_loss = self._kl_divergence(kl_mu, log_var)
            else:
                kl_dyn_loss = torch.tensor(0.0, device=batch.device)

        # --- Group 7: Tangent space entropy (encoder Jacobians via autograd) ---
        tangent_entropy_loss = None
        with self._timed("train/8.tangent_entropy"):
            if self.tangent_entropy_weight > 0:
                tangent_entropy_loss = self._tangent_space_entropy_loss(batch, z_full)

        # --- Group 8: Soft diffeomorphism cycle-consistency ---
        diffeo_loss = None
        with self._timed("train/9.diffeomorphism"):
            if self.diffeomorphism_weight > 0:
                diffeo_loss = self._diffeomorphism_loss(batch, z_full_for_recon)

        # ----------------------------------------------------------------
        # Auto-initialise log_var parameters on the very first batch
        # ----------------------------------------------------------------
        if self.log_var_init == 'auto' and not self._log_var_auto_initialized:
            self._auto_init_log_vars(
                r2_loss=r2_loss,
                loop_loss=loop_loss,
                fnn_loss=fnn_loss,
                jac_cons_loss=jac_cons_loss,
                jac_norm=jac_norm,
            )

        # ----------------------------------------------------------------
        # Combine into total_loss, applying homoscedastic or fixed weights
        # ----------------------------------------------------------------
        total_loss = torch.zeros(1, device=batch.device, dtype=batch.dtype).squeeze()

        # R²-like group
        if self.learn_r2_weight:
            total_loss = total_loss + self._homoscedastic_loss(r2_loss, self.log_var_r2)
        else:
            total_loss = total_loss + r2_loss

        # Loop closure group
        if loop_loss is not None:
            if self.learn_loop_closure_weight:
                total_loss = total_loss + self._homoscedastic_loss(loop_loss, self.log_var_loop_closure)
            else:
                total_loss = total_loss + self.loop_closure_weight * loop_loss

        # Jac norm group
        if self.learn_jac_norm_weight:
            total_loss = total_loss + self._homoscedastic_loss(jac_norm, self.log_var_jac_norm)
        elif self.jac_penalty > 0:
            total_loss = total_loss + self.jac_penalty * jac_norm

        # Jac consistency group
        if jac_cons_loss is not None:
            if self.learn_jac_cons_weight:
                total_loss = total_loss + self._homoscedastic_loss(jac_cons_loss, self.log_var_jac_cons)
            else:
                total_loss = total_loss + self.jac_consistency_weight * jac_cons_loss

        # FNN group
        if fnn_loss is not None:
            if self.learn_fnn_weight:
                total_loss = total_loss + self._homoscedastic_loss(fnn_loss, self.log_var_fnn)
            elif self.fnn_weight > 0:
                total_loss = total_loss + self.fnn_weight * fnn_loss

        # KL divergence (null + dyn, separate weights)
        if kl_null_loss is not None and eff_null_w > 0:
            total_loss = total_loss + eff_null_w * kl_null_loss
        if kl_dyn_loss is not None and eff_dyn_w > 0:
            total_loss = total_loss + eff_dyn_w * kl_dyn_loss

        # Tangent space entropy
        if tangent_entropy_loss is not None:
            total_loss = total_loss + self.tangent_entropy_weight * tangent_entropy_loss

        # Diffeomorphism cycle-consistency
        if diffeo_loss is not None:
            total_loss = total_loss + self.diffeomorphism_weight * diffeo_loss

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
                jac_cons_loss=jac_cons_loss,
                fnn_loss=fnn_loss,
                kl_null_loss=kl_null_loss,
                kl_dyn_loss=kl_dyn_loss,
                tangent_entropy_loss=tangent_entropy_loss,
                diffeo_loss=diffeo_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

        self._maybe_emit_profile("train")
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

        # Validation always monitors the 'most_recent' (live-frame) variant
        # of the trajectory loss — this is the quantity that actually
        # tracks forward-rollout fidelity and is what's used by
        # EarlyStopping / ModelCheckpoint. Training can still use a wider
        # reconstruction_mode (e.g. 'uniform') to get a richer gradient
        # from all delay-embedded components; that choice doesn't affect
        # which checkpoint gets selected.
        model_step_kwargs = {
            'alpha_teacher_forcing': self.alpha_validation,
            'obs_noise_scale': 0,
            'latent_noise_scale': 0,
            'reconstruction_mode': 'most_recent',
        }

        val_rets = {}
        with self._timed("val/1.trajectory_step"):
            val_rets['trajectory'] = self.trajectory_model_step(
                batch, batch_idx, dataloader_idx, **model_step_kwargs
            )

        # Encode once for loop closure, reconstruction, and diagnostics
        with self._timed("val/2.encode"):
            z_full = self.encode_trajectory(batch)
            mu_dyn, z_null = self._split_latent(z_full)
            z_dyn, log_var, kl_mu = self._vae_reparameterize(mu_dyn)  # deterministic in eval

        with self._timed("val/3.loop_closure"):
            val_loop_closure = self.loop_closure_model_step(
                z_dyn, batch_idx, dataloader_idx
            )

        # Reconstruction loss
        val_recon_loss = None
        with self._timed("val/4.reconstruction_loss"):
            if self.reconstruction_loss_weight > 0:
                with torch.no_grad():
                    val_recon_loss = self._reconstruction_loss(batch, z_full=z_full)

        # KL divergence / null penalty
        val_kl_null_loss = None
        val_kl_dyn_loss = None
        if self.n_target_dims is not None:
            with torch.no_grad():
                if z_null is not None and z_null.numel() > 0:
                    val_kl_null_loss = F.mse_loss(z_null, torch.zeros_like(z_null))
                else:
                    val_kl_null_loss = torch.tensor(0.0, device=batch.device)
                if self.use_vae and log_var is not None:
                    val_kl_dyn_loss = self._kl_divergence(kl_mu, log_var)
                else:
                    val_kl_dyn_loss = torch.tensor(0.0, device=batch.device)

        # Diffeomorphism cycle-consistency
        val_diffeo_loss = None
        if self.diffeomorphism_weight > 0:
            with torch.no_grad():
                val_diffeo_loss = self._diffeomorphism_loss(batch, z_full)

        # Latent prediction loss (already computed inside trajectory_model_step)
        val_latent_pred_loss = val_rets['trajectory']['metric_vals'].get('latent_pred_loss')

        # One-step teacher-forced prediction (for MASE diagnostic, C1)
        with self._timed("val/5.one_step_traj"):
            with torch.no_grad():
                one_step_ret = self.trajectory_model_step(
                    batch, batch_idx, dataloader_idx,
                    alpha_teacher_forcing=1,
                    obs_noise_scale=0,
                    latent_noise_scale=0,
                )
        # Accumulate raw MAE components for ratio-of-means aggregation
        if not hasattr(self, '_val_one_step_model_maes'):
            self._val_one_step_model_maes = []
            self._val_one_step_persistence_maes = []
        self._val_one_step_model_maes.append(
            one_step_ret['metric_vals']['model_mae'].float().item()
        )
        self._val_one_step_persistence_maes.append(
            one_step_ret['metric_vals']['persistence_mae'].float().item()
        )

        # Fast eigenvalue fraction (C3 diagnostic)
        with self._timed("val/6.eigvals"):
            with torch.no_grad():
                pred_jacs = self.compute_jacobians(z_dyn)  # (B, T, D, D)
                B, T, D, _ = pred_jacs.shape
                jacs_flat = pred_jacs.reshape(B * T, D, D)
                if self.n_eigval_jacobians is not None and self.n_eigval_jacobians < B * T:
                    idx = torch.randperm(B * T, device=jacs_flat.device)[:self.n_eigval_jacobians]
                    jacs_flat = jacs_flat[idx]
                finite_mask = torch.isfinite(jacs_flat).all(dim=-1).all(dim=-1)
                jacs_finite = jacs_flat[finite_mask]
                if jacs_finite.numel() > 0:
                    eigs_real = torch.linalg.eigvals(jacs_finite).real.flatten()
                    threshold = -1.0 / self.dt
                    n_too_fast = torch.sum(eigs_real <= threshold).float().item()
                    n_total = len(eigs_real)
                else:
                    n_too_fast = 0.0
                    n_total = 0
        if not hasattr(self, '_val_eig_too_fast'):
            self._val_eig_too_fast = []
            self._val_eig_total = []
        self._val_eig_too_fast.append(n_too_fast)
        self._val_eig_total.append(n_total)

        with self._timed("val/7.log_metrics"):
            if log_metrics:
                self.log_validation_metrics(
                    val_rets=val_rets,
                    batch=batch,
                    sync_dist=True,
                    val_loop_closure=val_loop_closure,
                    val_recon_loss=val_recon_loss,
                    val_latent_pred_loss=val_latent_pred_loss,
                    val_kl_null_loss=val_kl_null_loss,
                    val_kl_dyn_loss=val_kl_dyn_loss,
                    val_one_step_loss=one_step_ret['loss'],
                    val_diffeo_loss=val_diffeo_loss,
                )

        total_loss = sum(
            val_rets[pred_type]['loss'] for pred_type in val_rets
        )
        if val_recon_loss is not None:
            total_loss = total_loss + self.reconstruction_loss_weight * val_recon_loss
        if val_latent_pred_loss is not None:
            total_loss = total_loss + self.latent_prediction_loss_weight * val_latent_pred_loss

        # Track the same metric that PercentEarlyStopping monitors
        mean_val_loss = torch.stack(
            [val_rets[pt]['loss'] for pt in val_rets]
        ).mean()
        if not hasattr(self, 'current_epoch_val_losses'):
            self.current_epoch_val_losses = []
        self.current_epoch_val_losses.append(mean_val_loss.item())

        self._maybe_emit_profile("val")
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
                z_dyn, _ = self._split_latent(z)
                # Use first batch element for a long trajectory
                jacs = self.compute_jacobians(z_dyn[0:1])[0]  # (T', D, D)
                pred_le = self.compute_lyapunov_exponents(jacs, self.dt)

                # Compare the top-k exponents (k = number of true exponents)
                k = len(self.true_lyapunov_exponents)
                pred_top_k = pred_le[:k]
                true_le = self.true_lyapunov_exponents.to(pred_top_k.device)

                le_mse = torch.mean((pred_top_k - true_le) ** 2)
                self.log(
                    f"{prefix}/lyapunov_mse", le_mse,
                    sync_dist=sync_dist, add_dataloader_idx=False,
                    **log_kwargs,
                )

                # Log individual exponents
                for i, (pred, true) in enumerate(zip(pred_top_k, true_le)):
                    self.log(
                        f"{prefix}/lyapunov_{i}_pred", pred.item(),
                        sync_dist=sync_dist, add_dataloader_idx=False,
                        **log_kwargs,
                    )
        except Exception:
            pass  # Don't fail training/validation on diagnostic errors

    def _log_latent_utilization(self, batch, prefix, **log_kwargs):
        """Log entropy-based latent utilization score in [0, 1].

        1.0 = all latent dimensions carry equal variance.
        0.0 = a single dimension carries all variance.

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
                z_dyn, _ = self._split_latent(z)
                var = z_dyn.float().var(dim=(0, 1)).clamp(min=1e-10)  # (D_dyn,)
                p = var / var.sum()
                entropy = -(p * torch.log(p)).sum()
                n_dyn = z_dyn.shape[-1]
                max_entropy = math.log(n_dyn) if n_dyn > 1 else 1.0
                utilization = (entropy / max_entropy).item()
                self.log(f"{prefix}/latent_utilization", utilization, **log_kwargs)
        except Exception:
            pass

    def log_training_metrics(self, train_rets, total_loss, jac_norm, l1_loss,
                              batch, jacs_pred, batch_idx, recon_loss=None,
                              latent_pred_loss=None, jac_cons_loss=None,
                              fnn_loss=None,
                              kl_null_loss=None, kl_dyn_loss=None,
                              tangent_entropy_loss=None,
                              diffeo_loss=None,
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
            self.log(f"train/{pred_type}_loss", loss, **log_kwargs)
            for metric, val in metric_vals.items():
                if pred_type != 'trajectory':
                    continue
                self.log(f"train/{pred_type}_{metric}", val, **log_kwargs)

        self.log("train/total_loss", total_loss, **log_kwargs)
        if jac_norm is not None:
            self.log("train/jac_norm", jac_norm, **log_kwargs)
        self.log("train/l1_norm", l1_loss, **log_kwargs)
        if recon_loss is not None:
            self.log("train/recon_loss", recon_loss, **log_kwargs)
        if latent_pred_loss is not None:
            self.log("train/latent_pred_loss", latent_pred_loss, **log_kwargs)
        latent_pred_r2 = train_rets.get('trajectory', {}).get('metric_vals', {}).get('latent_pred_r2')
        if latent_pred_r2 is not None:
            self.log("train/latent_pred_r2", latent_pred_r2, **log_kwargs)
        if jac_cons_loss is not None:
            self.log("train/jac_cons_loss", jac_cons_loss, **log_kwargs)
        if fnn_loss is not None:
            self.log("train/fnn_loss", fnn_loss, **log_kwargs)
        if kl_null_loss is not None:
            self.log("train/kl_null_loss", kl_null_loss, **log_kwargs)
        if kl_dyn_loss is not None:
            self.log("train/kl_dyn_loss", kl_dyn_loss, **log_kwargs)
        if tangent_entropy_loss is not None:
            self.log("train/tangent_entropy_loss", tangent_entropy_loss, **log_kwargs)
        if diffeo_loss is not None:
            self.log("train/diffeo_loss", diffeo_loss, **log_kwargs)

        if self.learn_r2_weight:
            self.log("train/log_var_r2", self.log_var_r2.squeeze(), **log_kwargs)
        if self.learn_loop_closure_weight:
            self.log("train/log_var_loop_closure", self.log_var_loop_closure.squeeze(), **log_kwargs)
        if self.learn_fnn_weight:
            self.log("train/log_var_fnn", self.log_var_fnn.squeeze(), **log_kwargs)
        if self.learn_jac_cons_weight:
            self.log("train/log_var_jac_cons", self.log_var_jac_cons.squeeze(), **log_kwargs)
        if self.learn_jac_norm_weight:
            self.log("train/log_var_jac_norm", self.log_var_jac_norm.squeeze(), **log_kwargs)

        if self.teacher_forcing_annealing:
            self.log("train/alpha_teacher_forcing", self.alpha_teacher_forcing, **log_kwargs)

        self._log_lyapunov_comparison(batch, "train", **log_kwargs)
        self._log_latent_utilization(batch, "train", **log_kwargs)

    def log_validation_metrics(self, val_rets, batch, sync_dist=True,
                               val_loop_closure=None, val_recon_loss=None,
                               val_latent_pred_loss=None, val_kl_null_loss=None,
                               val_kl_dyn_loss=None, val_one_step_loss=None,
                               val_diffeo_loss=None):
        """Log validation metrics.

        Overrides the base class to skip true-Jacobian comparison and
        instead log Lyapunov exponent diagnostics.
        """
        for pred_type, ret_dict in val_rets.items():
            loss, metric_vals = ret_dict['loss'], ret_dict['metric_vals']
            self.log(f"val/{pred_type}_loss", loss, sync_dist=sync_dist, add_dataloader_idx=False)
            for metric, val in metric_vals.items():
                if pred_type != 'trajectory':
                    continue
                self.log(f"val/{pred_type}_{metric}", val, sync_dist=sync_dist, add_dataloader_idx=False)

        mean_val_loss = torch.stack(
            [val_rets[pt]['loss'] for pt in val_rets]
        ).mean()
        # Canonical name used by early stopping, checkpoint, and loader
        self.log("mean val loss", mean_val_loss, sync_dist=sync_dist)
        # Alias used by traj checkpoint in trainer.py
        self.log("trajectory val_loss", val_rets['trajectory']['loss'], sync_dist=sync_dist)

        if val_loop_closure is not None:
            self.log(
                "val/loop_closure_loss",
                val_loop_closure['metric_vals']['mse'],
                sync_dist=sync_dist,
                add_dataloader_idx=False,
            )

        if val_recon_loss is not None:
            self.log("val/recon_loss", val_recon_loss, sync_dist=sync_dist,
                     add_dataloader_idx=False)
        if val_latent_pred_loss is not None:
            self.log("val/latent_pred_loss", val_latent_pred_loss, sync_dist=sync_dist,
                     add_dataloader_idx=False)
        val_latent_pred_r2 = val_rets.get('trajectory', {}).get('metric_vals', {}).get('latent_pred_r2')
        if val_latent_pred_r2 is not None:
            self.log("val/latent_pred_r2", val_latent_pred_r2, sync_dist=sync_dist,
                     add_dataloader_idx=False)
        if val_kl_null_loss is not None:
            self.log("val/kl_null_loss", val_kl_null_loss, sync_dist=sync_dist,
                     add_dataloader_idx=False)
        if val_kl_dyn_loss is not None:
            self.log("val/kl_dyn_loss", val_kl_dyn_loss, sync_dist=sync_dist,
                     add_dataloader_idx=False)
        if val_one_step_loss is not None:
            self.log("val/one_step_loss", val_one_step_loss, sync_dist=sync_dist,
                     add_dataloader_idx=False)
        if val_diffeo_loss is not None:
            self.log("val/diffeo_loss", val_diffeo_loss, sync_dist=sync_dist,
                     add_dataloader_idx=False)

        self._log_lyapunov_comparison(batch, "val", sync_dist=sync_dist)
        self._log_latent_utilization(batch, "val", sync_dist=sync_dist)

    def get_pred_jacs(self, batch):
        """Override to compute Jacobians in latent space (z_dyn)."""
        z = self.encode_trajectory(batch)
        z_dyn, _ = self._split_latent(z)
        return self.compute_jacobians(z_dyn)
