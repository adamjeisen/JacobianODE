# %% [markdown]
# # Observation-space Lyapunov spectrum: run y7938e7x
#
# Run: `y7938e7x` in
# `lorenz_partial_25d_additive_mse_uniform_p30_obsnoise005__lc_sweep`.
#
# Compose the model's observation-space Jacobian. The cleanest derivation:
# the model's per-step rollout is y → E(y)=(z_dyn, z_null) → (z_dyn + dt·f,
# z_null) → D(z'). Linearising:
#
#     δy' = δy + dt · J_{D,dyn}(z) · J_f(z_dyn) · J_{E,dyn}(y) · δy
#
# so the obs-space *vector-field* Jacobian is the rank-3 product
#
#     J_obs(y) = J_{D,dyn} · J_f · J_{E,dyn}                 (25×25)
#
# where
#  • `J_{E,dyn} = ∂z_dyn/∂y`  (3×25, jacrev of the first-3 encoder outputs),
#  • `J_f       = ∂(latent vector field)/∂z_dyn`  (3×3, the model's
#    MLP-Jacobian network),
#  • `J_{D,dyn} = ∂y/∂z_dyn` at fixed `z_null`  (25×3, jacrev of the
#    decoder restricted to varying z_dyn).
#
# Equivalent block-diagonal form (avoid the full 25×25 inverse — cheaper
# and avoids accidentally substituting `J_{E,dyn}^+` for `J_{D,dyn}`, which
# is *not* the same operator in general):
#
#     J_obs = J_E^{-1} · block_diag(J_f, 0_22) · J_E.
#
# We compute via the direct route and assert agreement with the inverse
# route at a sample point.
#
# Then run the standard QR-based Lyapunov accumulation
# (`LitLatentJacobianODE.compute_lyapunov_exponents`) on the (T, 25, 25)
# sequence and compare against the empirical 3-d Lorenz spectrum the
# sweep's analyze pipeline already saved.

# %%
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
from JacobianODE.jacobians import create_dataloaders
from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

PROJECT = "JacobianODE/Lorenz_INDpartial_N25_D1_NormTrue_T3__JacobianODE"
RUN_ID = "y7938e7x"
GROUP = "lorenz_partial_25d_additive_mse_uniform_p30_obsnoise005__lc_sweep"
METRICS_JSON = (
    Path("/home/adameisen/Documents/jacobian-analyses")
    / "Lorenz_INDpartial_N25_D1_NormTrue_T3__JacobianODE"
    / GROUP / "metrics.json"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
run, cfg, eq, dt, values, _, _, _, _, lit_model = load_run(
    PROJECT, run_id=RUN_ID, generate_data=True, verbose=True,
)
train_dl, val_dl, test_dl, trajs = create_dataloaders(
    cfg, values, verbose=True, return_full_obs=True,
)
load_checkpoint(run, cfg, lit_model, epoch=None, verbose=True)
lit_model = lit_model.to(DEVICE).eval()

n_obs = trajs["test_trajs"].sequence.shape[-1]
n_target = lit_model.n_target_dims  # 3 = z_dyn dim
encoder = lit_model.encoder
print(f"\nn_obs={n_obs}, n_target_dims={n_target}, dt={float(dt):.4g}")

# %%
# Take the first test trajectory (full length; cap to N_PTS for tractability).
N_PTS = 800
y_full = torch.as_tensor(trajs["test_trajs"].sequence[0]).float().to(DEVICE)
T = y_full.shape[0]
idxs = torch.linspace(0, T - 1, N_PTS).long()
y_sample = y_full[idxs]
print(f"Using {N_PTS} of {T} trajectory points.")

# %%
# Per-point encoder/decoder helpers (vmap-able 1-d tensor in/out).
def encode_pt(y):
    return encoder.encode(y.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

def encode_dyn(y):
    # Just the dyn-subspace output (3,) so jacrev gives a 3×25 matrix.
    return encode_pt(y)[:n_target]

def decode_with_split(z_dyn, z_null):
    z = torch.cat([z_dyn, z_null], dim=-1)
    return encoder.decode(z.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

# Encode, then split into dyn / null.
with torch.no_grad():
    z_all = torch.func.vmap(encode_pt)(y_sample)  # (N, 25)
z_dyn_all = z_all[:, :n_target].contiguous()           # (N, 3)
z_null_all = z_all[:, n_target:].contiguous()          # (N, 22)

# J_{E,dyn} (3×25) — jacrev of the first-3 encoder outputs at each y.
with torch.no_grad():
    J_E_dyn_all = torch.func.vmap(torch.func.jacrev(encode_dyn))(y_sample)  # (N, 3, 25)
print(f"J_E_dyn_all: {tuple(J_E_dyn_all.shape)}")

# J_{D,dyn} (25×3) — jacrev of the decoder wrt z_dyn at fixed z_null.
with torch.no_grad():
    J_D_dyn_all = torch.func.vmap(
        torch.func.jacrev(decode_with_split, argnums=0)
    )(z_dyn_all, z_null_all)  # (N, 25, 3)
print(f"J_D_dyn_all: {tuple(J_D_dyn_all.shape)}")

# Latent dyn Jacobians J_f (3×3) at each z_dyn.
with torch.no_grad():
    J_f_all = lit_model.compute_jacobians(z_dyn_all.unsqueeze(0)).squeeze(0)  # (N, 3, 3)
print(f"J_f_all: {tuple(J_f_all.shape)}")

# Compose: J_obs = J_{D,dyn} · J_f · J_{E,dyn}  (25×25, rank ≤ 3).
J_obs_all = J_D_dyn_all @ J_f_all @ J_E_dyn_all  # (N, 25, 25)
print(f"J_obs_all: {tuple(J_obs_all.shape)}")

# --- Sanity: regression-check the direct route against the
# block-diagonal-via-inverse route at the first sample point. They should
# agree to ~machine precision; if they don't, something's mislabelled.
with torch.no_grad():
    J_E0 = torch.func.jacrev(encode_pt)(y_sample[0])  # (25, 25)
    J_D0 = torch.linalg.inv(J_E0)                     # (25, 25)
    block = torch.zeros(n_obs, n_obs, device=DEVICE, dtype=J_E0.dtype)
    block[:n_target, :n_target] = J_f_all[0]
    J_obs0_via_inv = J_D0 @ block @ J_E0
    err = (J_obs_all[0] - J_obs0_via_inv).abs().max().item()
    print(f"\nsanity: max |direct − block-diag-via-inverse| at point 0 = {err:.2e}")
    assert err < 1e-4, "obs-Jacobian compositions disagree — investigate"

# %%
# QR-based Lyapunov accumulation. compute_lyapunov_exponents wants
# (T, D, D) for a single trajectory.
with torch.no_grad():
    pred_lyap = LitLatentJacobianODE.compute_lyapunov_exponents(
        J_obs_all, dt=float(dt),
    )
pred_sorted = pred_lyap.sort(descending=True).values.cpu().numpy()
print("Model obs-space Lyapunov spectrum (sorted, descending):")
print(np.array2string(pred_sorted, precision=3, suppress_small=False))

# %%
# Empirical Lorenz spectrum: prefer the value the sweep's analyze pipeline
# already computed (eq.jac on full-state trajectories), fall back to
# canonical Lorenz exponents.
empirical = None
try:
    md = json.loads(METRICS_JSON.read_text())
    empirical = md.get("empirical_lyapunov_spectrum")
    if empirical is not None:
        empirical = np.array(empirical, dtype=float)
        print(f"Empirical spectrum from sweep metrics.json: {empirical}")
except Exception as e:
    print(f"Couldn't load metrics.json: {e}")
if empirical is None:
    empirical = np.array([0.906, 0.0, -14.572])  # canonical Lorenz
    print(f"Falling back to canonical Lorenz exponents: {empirical}")

# %%
# Plot.
fig, ax = plt.subplots(figsize=(10, 6))
xs = np.arange(1, n_obs + 1)
ax.plot(xs, pred_sorted, "o-", color="C0", markersize=7, lw=1.5,
        label=f"model obs-space (25 dims, run {RUN_ID})")
# Empirical spectrum: just markers at indices 1,2,3 since it's only 3 values
xs_emp = np.arange(1, len(empirical) + 1)
ax.plot(xs_emp, empirical, "s", color="C3", markersize=10,
        label=f"empirical (Lorenz, {len(empirical)} dims)")
# Annotate first 3 model exponents with their numeric values. Stagger
# the offsets vertically so the three labels don't pile on each other
# when the top of the spectrum is flat.
_offsets = [(12, 18), (12, 0), (12, -18)]
for i in range(3):
    ax.annotate(
        f"λ{i+1}={pred_sorted[i]:+.3f}",
        (xs[i], pred_sorted[i]),
        textcoords="offset points", xytext=_offsets[i],
        fontsize=10, color="C0",
        arrowprops=dict(arrowstyle="-", color="C0", lw=0.7, alpha=0.5),
    )
# Reference horizontal lines at the empirical exponents.
for ev in empirical:
    ax.axhline(ev, color="C3", ls="--", lw=0.8, alpha=0.4)
ax.axhline(0, color="k", lw=0.5, alpha=0.5)
ax.set_xlabel("Lyapunov index")
ax.set_ylabel("Lyapunov exponent")
ax.set_title(
    f"Obs-space Lyapunov spectrum — run {RUN_ID}\n"
    f"({GROUP})"
)
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_dir = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"obs_lyapunov_{RUN_ID}.png"
fig.savefig(out_path, dpi=130)
print(f"\nSaved plot to {out_path}")
plt.show()
