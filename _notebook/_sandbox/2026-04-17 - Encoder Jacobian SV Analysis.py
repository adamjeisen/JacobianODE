# %% [markdown]
# # Encoder Jacobian singular-value analysis
#
# Run `39prin7v` (best traj_loss in
# `lorenz_partial_25d_additive_mse_uniform_p30_obsnoise001__lc_sweep`).
#
# Question: how does the encoder distribute stretch across obs-space
# directions, and in particular: which obs directions get projected into
# z_dyn (first 3 latent dims, used by the ODE) vs z_null (remaining 22)?
#
# Volume preservation (additive coupling) pins `prod(sigma_i(J_E)) = 1`,
# so we expect a mix of >1 (expanding) and <1 (contracting) singular
# values, with the product 1. The question is where they live and how
# they relate to obs-space variance.

# %%
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
from JacobianODE.jacobians import create_dataloaders

PROJECT = "JacobianODE/Lorenz_INDpartial_N25_D1_NormTrue_T3__JacobianODE"
RUN_ID = "39prin7v"
N_POINTS = 1024
DEVICE = "cpu"

# %%
run, cfg, eq, dt, values, _, _, _, _, lit_model = load_run(
    PROJECT, run_id=RUN_ID, generate_data=True, verbose=True,
)
train_dl, val_dl, test_dl, trajs = create_dataloaders(
    cfg, values, verbose=True, return_full_obs=True,
)
load_checkpoint(run, cfg, lit_model, epoch=None, verbose=True)
lit_model = lit_model.to(DEVICE).eval()

n_target = lit_model.n_target_dims
encoder = lit_model.encoder
n_latent = encoder.n_latent
print(f"\nn_latent={n_latent}, n_target_dims={n_target}")

# %%
# Obs-space points from validation/test trajectories (noise-baked-in).
val_seq = trajs["val_trajs"].sequence  # (n_traj, T, n_obs)
obs = val_seq.reshape(-1, val_seq.shape[-1]).to(DEVICE)
print(f"Total obs points: {obs.shape[0]}, n_obs={obs.shape[1]}")

torch.manual_seed(0)
idx = torch.randperm(obs.shape[0])[:N_POINTS]
x_sample = obs[idx]

# Obs-space covariance & eigendecomposition.
obs_mean = obs.mean(0)
X = obs - obs_mean
obs_cov = X.T @ X / (obs.shape[0] - 1)
obs_evals, obs_evecs = torch.linalg.eigh(obs_cov)  # ascending
order = obs_evals.argsort(descending=True)
obs_evals = obs_evals[order]
obs_evecs = obs_evecs[:, order]  # columns are PCs, descending
obs_total_var = obs_evals.sum().item()
print(f"Obs total var: {obs_total_var:.4f}")
print(f"Obs var top-5 eigvals: {obs_evals[:5].numpy()}")
print(f"Obs var bottom-5 eigvals: {obs_evals[-5:].numpy()}")

# %%
# Per-point J_E via torch.func (N_POINTS × n_latent × n_obs)
def _encode_point(x_pt):
    z = encoder.encode(x_pt.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return z

jac_fn = torch.func.vmap(torch.func.jacrev(_encode_point))
with torch.no_grad():
    J_all = jac_fn(x_sample)  # (N, n_latent, n_obs)
print(f"J_all shape: {tuple(J_all.shape)}")

# Check volume preservation.
log_absdets = torch.linalg.slogdet(J_all).logabsdet
print(f"log|det J_E|: mean={log_absdets.mean().item():+.3e}, "
      f"std={log_absdets.std().item():.3e}  (should be ~0)")

# %%
# SVDs of full J, dyn-row block, null-row block.
# For the row blocks we use full_matrices=False so V is (n_obs, k).
J_dyn = J_all[:, :n_target, :]           # (N, 3, 25)
J_null = J_all[:, n_target:, :]          # (N, 22, 25)

U_full, S_full, Vh_full = torch.linalg.svd(J_all, full_matrices=False)
U_dyn, S_dyn, Vh_dyn = torch.linalg.svd(J_dyn, full_matrices=False)
U_null, S_null, Vh_null = torch.linalg.svd(J_null, full_matrices=False)

V_dyn = Vh_dyn.transpose(-1, -2)   # (N, n_obs, n_target)   obs-space dirs the dyn block sees
V_null = Vh_null.transpose(-1, -2)  # (N, n_obs, n_latent - n_target)

print(f"\nFull J_E sv (median, descending):\n  "
      f"{np.array2string(S_full.median(0).values.numpy(), precision=3, suppress_small=True)}")
print(f"\nDyn block sv (median): "
      f"{S_dyn.median(0).values.numpy().round(3)}")
print(f"Null block sv (median): "
      f"{S_null.median(0).values.numpy().round(3)}")

# %%
# How much obs-space variance lives in the row span of each block?
#   frac_in_dyn = tr(V_dyn V_dyn^T  obs_cov) / tr(obs_cov)
P_dyn = V_dyn @ V_dyn.transpose(-1, -2)      # (N, n_obs, n_obs)
P_null = V_null @ V_null.transpose(-1, -2)
var_in_dyn = torch.einsum('nij,ji->n', P_dyn, obs_cov) / obs_total_var
var_in_null = torch.einsum('nij,ji->n', P_null, obs_cov) / obs_total_var
print(f"\nFraction of obs variance in dyn V-span:  "
      f"mean={var_in_dyn.mean().item():.3f}, median={var_in_dyn.median().item():.3f}")
print(f"Fraction of obs variance in null V-span: "
      f"mean={var_in_null.mean().item():.3f}, median={var_in_null.median().item():.3f}")
print(f"(Sum should be 1:  {(var_in_dyn + var_in_null).mean().item():.3f})")

# %%
# Per obs-PC: how much does each block pick it up, on average?
# For PC k, compute <v_k, P_dyn v_k> averaged over sample points.
pc_weight_dyn = torch.einsum('nij,jk,ik->kn', P_dyn, obs_evecs, obs_evecs).mean(1)
pc_weight_null = torch.einsum('nij,jk,ik->kn', P_null, obs_evecs, obs_evecs).mean(1)
print("\nPer-obs-PC coverage by dyn/null row span (mean over points):")
print(f"{'PC':>4} {'eigval':>10} {'dyn weight':>12} {'null weight':>12}")
for k in range(n_latent):
    print(f"{k:>4} {obs_evals[k].item():>10.4g} "
          f"{pc_weight_dyn[k].item():>12.3f} {pc_weight_null[k].item():>12.3f}")

# %%
# Also: "expansion" in each obs-PC direction.
# ||J_E v_k||^2 = singular-value-squared energy mapped by J_E from direction v_k.
#   full block gives stretch-squared = sum over all latent rows of (J_E v_k)_i^2
#   dyn block gives contribution to z_dyn
#   null block gives contribution to z_null
# Averaged over points.
# Shape gymnastics: J_all is (N, 25, 25). obs_evecs is (25, 25).
#   J v_k is (N, 25) for each k.
Jv = torch.einsum('nij,jk->nik', J_all, obs_evecs)  # (N, n_latent, n_pc)
stretch_full = (Jv ** 2).sum(1).mean(0)                      # (n_pc,)
stretch_dyn = (Jv[:, :n_target, :] ** 2).sum(1).mean(0)       # (n_pc,)
stretch_null = (Jv[:, n_target:, :] ** 2).sum(1).mean(0)      # (n_pc,)
print("\nPer obs-PC: ||J v_k||^2 (mean over points), split dyn vs null:")
print(f"{'PC':>4} {'eigval':>10} {'full':>8} {'dyn':>8} {'null':>8} "
      f"{'dyn/full':>10} {'dyn*eigval':>12}")
for k in range(n_latent):
    ev = obs_evals[k].item()
    f_ = stretch_full[k].item()
    d_ = stretch_dyn[k].item()
    n_ = stretch_null[k].item()
    print(f"{k:>4} {ev:>10.4g} {f_:>8.3f} {d_:>8.3f} {n_:>8.3f} "
          f"{(d_/f_):>10.3f} {(d_ * ev):>12.4g}")

# %%
# --- Plot ---
out_dir = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox")
out_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. Full J_E singular spectrum
ax = axes[0, 0]
q = S_full.quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0).numpy()
k = np.arange(1, n_latent + 1)
ax.semilogy(k, q[1], 'o-', label='median')
ax.fill_between(k, q[0], q[2], alpha=0.3)
ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='1 (volume ref)')
ax.set_xlabel('index')
ax.set_ylabel('singular value')
ax.set_title(f'Full $J_E$ singular spectrum (N={N_POINTS})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Row-block singular spectra
ax = axes[0, 1]
ax.semilogy(np.arange(1, n_target + 1),
            S_dyn.median(0).values.numpy(), 'o-',
            label=f'dyn rows (k={n_target})', color='C1')
ax.semilogy(np.arange(1, n_latent - n_target + 1),
            S_null.median(0).values.numpy(), 's-',
            label=f'null rows (k={n_latent - n_target})', color='C2')
ax.axhline(1.0, color='k', ls='--', alpha=0.5)
ax.set_xlabel('index')
ax.set_ylabel('singular value (median)')
ax.set_title('Dyn vs null row-block SVD')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Obs-space variance spectrum
ax = axes[0, 2]
ax.semilogy(np.arange(1, n_latent + 1), obs_evals.numpy(), 'o-', color='C3')
ax.set_xlabel('obs PC index')
ax.set_ylabel('eigenvalue')
ax.set_title('Obs-space covariance eigenvalues')
ax.grid(True, alpha=0.3)

# 4. Fraction of obs variance in dyn span
ax = axes[1, 0]
ax.hist(var_in_dyn.numpy(), bins=40, color='C1', alpha=0.7)
ax.axvline(var_in_dyn.mean().item(), color='r', ls='--',
           label=f'mean={var_in_dyn.mean().item():.3f}')
ax.set_xlabel('frac of obs var in z_dyn row span')
ax.set_ylabel('count (over points)')
ax.set_title(f'z_dyn span captures obs variance')
ax.legend()

# 5. Per obs-PC weight on dyn vs null rows
ax = axes[1, 1]
xk = np.arange(n_latent)
ax.bar(xk - 0.2, pc_weight_dyn.numpy(), width=0.4, label='dyn', color='C1')
ax.bar(xk + 0.2, pc_weight_null.numpy(), width=0.4, label='null', color='C2')
ax.set_xlabel('obs PC (descending variance)')
ax.set_ylabel('mean block coverage')
ax.set_title('Which obs-PC does each row block see?')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 6. Expansion (||J v_k||^2) per obs-PC, stacked by block
ax = axes[1, 2]
xk = np.arange(n_latent)
ax.bar(xk, stretch_dyn.numpy(), label='dyn contribution', color='C1')
ax.bar(xk, stretch_null.numpy(), bottom=stretch_dyn.numpy(),
       label='null contribution', color='C2', alpha=0.8)
ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='isometry')
ax.set_xlabel('obs PC (descending variance)')
ax.set_ylabel(r'$\mathbb{E}\,\|J_E v_k\|^2$')
ax.set_title('Per-obs-PC stretch (row-block split)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle(f'Encoder Jacobian structure — run {RUN_ID} (lc=1e-5, obs_noise=0.01)')
fig.tight_layout()
outfile = out_dir / f'encoder_jacobian_sv_{RUN_ID}.png'
fig.savefig(outfile, dpi=120)
print(f"\nSaved plot to {outfile}")
plt.show()
