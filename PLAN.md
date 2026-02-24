# Causal Sequence Encoder + Latent JacobianODE Training Pipeline

## Answers to Your Questions

### Q1: FNN + Jacobian dimensionality / spurious Lyapunov exponents

Even with FNN driving certain latent dimensions to zero, the Jacobian is still 10x10. What happens to Lyapunov exponents for effectively-zero dimensions?

**Answer:** If the encoder pushes k dimensions to near-zero, the Jacobian rows/columns corresponding to those dimensions will have near-zero gradients and will learn arbitrary (likely small) values. The QR-based Lyapunov computation (`compute_lyapunov_exponents`) will still produce 10 exponents. The k "dead" dimensions will produce exponents that are **large and negative** (strongly contracting) because the dynamics along those dimensions are trivially stable — the latent values are pinned near zero, so the effective dynamics are `dz_i/dt ≈ 0`, giving Lyapunov exponents ≈ 0 or slightly negative. This means:

- The **top few Lyapunov exponents will be correct** (corresponding to the active latent dimensions that capture the true attractor geometry).
- The remaining exponents will be spurious negatives — harmless for analysis since you'd only compare the top-k (where k = true system dimension) with ground truth.
- It does NOT matter that the intrinsic dimension is lower than 10, as long as you only interpret the top-k exponents.

**However**, there's a subtler risk: if the Jacobian MLP learns cross-coupling between active and dead dimensions (e.g., nonzero off-diagonal blocks), this can introduce noise into the active exponents. The FNN regularizer helps by explicitly penalizing unused dimensions, making the dead dimensions truly inert.

### Q2: Alternatives to FNN for dimensionality

Options besides FNN:

1. **Just let it be.** The prediction task itself provides strong pressure to use the right number of dimensions. Monitor per-dimension variance during training — if it naturally separates into active/inactive dimensions, you don't need explicit regularization.

2. **Activity regularization (L1 on latent activations).** Simpler than FNN — just add `lambda * |z|_1` to the loss. This pushes unused dimensions toward zero. Less principled than FNN but much simpler.

3. **Bottleneck scheduling.** Start with `n_latent=3` (matching the known Lorenz dimension) and increase if the prediction loss plateaus. This avoids the problem entirely by not over-parameterizing.

4. **DeCov regularizer** (already in your codebase, `regularizers.py:142-179`). Penalizes correlation between latent dimensions, encouraging each active dimension to capture independent information.

### Q3: Causality, windowing, and gradient flow

**Now addressed in the implementation plan below.** Key decisions:

- **All encoders must be causal** — you're correct, this is non-negotiable.
- **`context_margin` parameter** — added, default 0, controls how many initial latent embeddings to skip.
- **Gradient flow:** Yes, gradients from the predicted observations backpropagate through ALL latents in the encoder window, including the `t_i` initialization steps that aren't directly predicted on. This is actually desirable — it tells the encoder "produce embeddings at these early positions that are useful for the JacobianODE to initialize its spline fit." The JacobianODE uses those `t_i` steps to fit its interpolation, so the encoder gets gradient signal saying "make these embeddings dynamically coherent."

---

## Implementation Plan

### Step 1: Make Transformer Encoder Causal

**File:** `JacobianODE/fnn/sequence_networks.py`

Modify `TransformerSequenceEncoder.forward()` to generate and pass a causal attention mask:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: (B, T, D) -> (B, T, D')."""
    h = self.input_proj(x)  # (B, T, d_model)
    if self.pos_enc is not None:
        h = self.pos_enc(h)
    # Generate causal mask: upper-triangular = -inf
    T = h.size(1)
    causal_mask = nn.Transformer.generate_square_subsequent_mask(
        T, device=h.device, dtype=h.dtype
    )
    h = self.transformer(h, mask=causal_mask)  # (B, T, d_model)
    return self.to_latent(h)  # (B, T, D')
```

This is a one-line change (plus mask generation). `generate_square_subsequent_mask` produces the standard causal mask where position `i` can only attend to positions `<= i`.

### Step 2: Add `context_margin` Parameter

**File:** `JacobianODE/fnn/sequence_networks.py`

Add `context_margin` as a parameter to `SequenceAutoencoder` (and propagate through factory functions):

```python
class SequenceAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, context_margin=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.context_margin = context_margin
```

**File:** `JacobianODE/models/latent_jacobian.py`

In `encode_trajectory`, after encoding, skip the first `context_margin` timesteps:

```python
def encode_trajectory(self, batch):
    if hasattr(self.encoder, 'time_window'):
        # ... window-based encoding (unchanged) ...
    else:
        z = self.encoder.encode(batch)
    # Skip initial embeddings that lack sufficient causal context
    margin = getattr(self.encoder, 'context_margin', 0)
    if margin > 0:
        z = z[:, margin:, :]
    return z
```

This means observation-space target extraction must also account for the margin offset. In `_extract_obs_targets`, the obs index for latent index `t` becomes `t + margin` (for sequence-based encoders).

### Step 3: Set `encoder_warmup_epochs` Default to 0

**File:** `JacobianODE/models/latent_jacobian.py`

The default is already `encoder_warmup_epochs=0` in the constructor (line 54). No code change needed — but the config file should explicitly set it to 0.

**File:** `JacobianODE/jacobians/conf/model/latent_mlp.yaml`

Add:
```yaml
encoder_warmup_epochs: 0
```

### Step 4: Add Sequence Encoder Config Files

**New file:** `JacobianODE/jacobians/conf/model/latent_transformer.yaml`

```yaml
params:
  _target_: JacobianODE.models.mlp.MLP
  input_dim: null    # = n_latent, set at runtime
  hidden_dim: [256, 512, 512]
  num_layers: 3
  output_dim: null   # = n_latent * n_latent, set at runtime
  residuals: False
  dropout: 0.0
  activation: 'silu'

encoder:
  _target_: JacobianODE.fnn.sequence_networks.build_transformer
  n_input: 1
  n_latent: 10
  d_model: 64
  n_heads: 4
  n_layers: 3
  dim_feedforward: 128
  dropout: 0.1
  use_positional_encoding: True
  positional_encoding_type: sinusoidal
  decoder_hidden: 128
  decoder_layers: 2
  context_margin: 0

prediction_steps: 10
encoder_warmup_epochs: 0
```

Similar configs for SSM and TCN variants.

### Step 5: Implement the Batching Pipeline

This is the core of the data flow. The pipeline:

1. **Raw data:** `K x T_tot x D` (K trajectories, T_tot total timesteps, D observed dims)
2. **Sliding window batching:** Create batches of shape `B x T x D` where `T` is the encoder's time window. Windows are spaced by a configurable `batch_stride` parameter.
3. **Encoder:** `B x T x D` -> `B x T x D'` (causal latent embeddings)
4. **JacobianODE sub-windowing:** From the `B x T x D'` latent, extract `N x t_J x D'` windows where `t_J = t_i + t_p`. These are spaced by a configurable `jac_window_stride`.
5. **Prediction loss:** Only on the `t_p` predicted steps (after decoding back to obs space).

**Changes to `latent_jacobian.py`:**

The current code already does steps 3-5 correctly (lines 310-369). The main change is:

- Add `jac_window_stride` parameter (currently hardcoded as `self.prediction_steps` for non-overlapping windows). Make it configurable so windows can overlap.
- Make sure `_extract_obs_targets` accounts for `context_margin` offset.

**Changes to the data pipeline / config:**

- Add `batch_window_length` to the data/training config — the `T` for encoder batching.
- Add `batch_stride` — spacing between sliding windows when creating batches from raw trajectories.
- Add `jac_window_stride` — spacing between JacobianODE sub-windows within each encoded batch.

These should go in the training config:

```yaml
# In training.yaml (additions)
batch_window_length: 200   # T: encoder processes this many timesteps
batch_stride: 50           # spacing between sliding windows for batching
jac_window_stride: null    # defaults to prediction_steps (non-overlapping)
```

### Step 6: Update Factory Functions for `context_margin`

**File:** `JacobianODE/fnn/sequence_networks.py`

All `build_*` functions need to accept and pass through `context_margin`:

```python
def build_transformer(n_input, n_latent, context_margin=0, ...):
    encoder = TransformerSequenceEncoder(...)
    decoder = StepDecoder(...)
    return SequenceAutoencoder(encoder, decoder, context_margin=context_margin)
```

Same for `build_ssm`, `build_tcn`, `build_tcn_spatial`.

### Step 7: Latent Utilization Monitoring

**File:** `JacobianODE/models/latent_jacobian.py`

In `log_training_metrics` and `log_validation_metrics`, add per-dimension variance tracking:

```python
with torch.no_grad():
    z = self.encode_trajectory(batch)
    # Per-dimension variance across batch and time
    z_var = z.var(dim=(0, 1))  # (D_latent,)
    for i, v in enumerate(z_var):
        self.log(f"{prefix} latent_var_dim_{i}", v.item(), ...)
    # Number of "active" dimensions (variance > threshold)
    n_active = (z_var > 1e-4).sum().item()
    self.log(f"{prefix} n_active_latent_dims", n_active, ...)
```

---

## Summary of All Changes

| File | Change |
|------|--------|
| `sequence_networks.py` | Add causal mask to Transformer forward |
| `sequence_networks.py` | Add `context_margin` to `SequenceAutoencoder` and all `build_*` factories |
| `latent_jacobian.py` | Apply `context_margin` offset in `encode_trajectory` |
| `latent_jacobian.py` | Account for margin in `_extract_obs_targets` |
| `latent_jacobian.py` | Make `jac_window_stride` configurable |
| `latent_jacobian.py` | Add latent utilization monitoring to logging |
| `conf/model/latent_transformer.yaml` | New config for Transformer encoder |
| `conf/model/latent_ssm.yaml` | New config for SSM encoder |
| `conf/model/latent_tcn.yaml` | New config for TCN encoder |
| `conf/training/training.yaml` | Add `batch_window_length`, `batch_stride`, `jac_window_stride` |
| `conf/model/latent_mlp.yaml` | Add `encoder_warmup_epochs: 0` |
