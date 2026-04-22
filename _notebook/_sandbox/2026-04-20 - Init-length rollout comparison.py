# %% [markdown]
# # Diagnostic: rollout + Lyapunov spectrum vs. `traj_init_steps`
#
# For run 16s8uli1 (p30 obsnoise005 ndelays sweep, n_delays=50), compare
# four rollout configurations:
#
# | name                          | init | rollout | init/roll | training ratio? |
# |---                            |---   |---      |---        |---              |
# | train-native                  | 15   | 30      | 0.50      | ✓ matches       |
# | train-init extrapolated       | 15   | 500     | 0.03      | init matches training, but rollout is 17× longer |
# | ratio-matched (50% init)      | 250  | 500     | 0.50      | ratio matches training, scaled up |
# | current Lyap analysis path    | 45   | 400     | 0.11      | OOD on both axes |
#
# For each:
#   - Free-run rollout from encoded initial condition, integration kwargs
#     pinned to the training config (interp_pts=4, inner_N=20, fast_mode=True).
#   - Decode z_dyn → obs dim 0 for visual comparison with truth.
#   - Compute Jacobians along z_pred and QR-based Lyapunov exponents for
#     configs long enough to converge (rollout ≥ 500).
#
# Output two diagnostic PNGs:
#   - obs-space overlay per config
#   - Lyapunov spectrum per config, with the canonical Lorenz spectrum reference

# %%
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
from JacobianODE.jacobians import create_dataloaders
from JacobianODE.jacobians.jacobianODE import JacobianODEint
from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

PROJECT = "JacobianODE/Lorenz_INDpartial_NDsweep_D1_NormTrue__JacobianODE"
RUN_ID = "16s8uli1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

# %%
run, cfg, eq, dt, values, _, _, _, _, lit_model = load_run(
    PROJECT, run_id=RUN_ID, generate_data=True, verbose=False,
)
train_dl, val_dl, test_dl, trajs = create_dataloaders(
    cfg, values, verbose=False, return_full_obs=True,
)
load_checkpoint(run, cfg, lit_model, epoch=None, verbose=False)
lit_model = lit_model.to(DEVICE).eval()

n_target = lit_model.n_target_dims
ikw = dict(cfg.training.lightning.jacobianODEint_kwargs)
print(f"training kwargs: {ikw}")
print(f"n_target_dims={n_target}, dt={float(dt):.5g}")

# %%
# Encode the first test trajectory once; each config slices a prefix of it.
traj_full = torch.as_tensor(trajs["test_trajs"].sequence[[0]]).float().to(DEVICE)  # (1, T_full, D)
with torch.no_grad():
    z_full_encoded = lit_model.encode_trajectory(traj_full)  # (1, T_full, n_latent)
    z_dyn_full, _ = lit_model._split_latent(z_full_encoded)  # (1, T_full, n_target)
T_full = z_dyn_full.shape[1]
print(f"T_full (test traj length): {T_full}")

configs = [
    dict(name="train-native",           init=15,  roll=30,   lyap=False),
    dict(name="train-init extrapolated",init=15,  roll=500,  lyap=True),
    dict(name="ratio-matched (init=½ roll)", init=250, roll=500, lyap=True),
    dict(name="current Lyap path",      init=45,  roll=400,  lyap=True),
]

jac_ode = JacobianODEint(lit_model.compute_jacobians, float(dt))

results = []
for cfg_ in configs:
    init_n, roll_n = cfg_["init"], cfg_["roll"]
    total = init_n + roll_n
    # Build input: first `init_n` are true encoded, the next `roll_n` are zeros.
    if init_n > T_full:
        raise ValueError(f"need >={init_n} init points, only have {T_full}")
    z_init = z_dyn_full[:, :init_n, :]  # (1, init_n, n_target)
    z_padded = torch.cat(
        [z_init, torch.zeros(1, roll_n, n_target, device=DEVICE)], dim=1
    )
    with torch.no_grad():
        z_pred = jac_ode.generate_dynamics(
            z_padded,
            traj_init_steps=init_n,
            alpha_teacher_forcing=0.0,
            fast_mode=True,
            verbose=False,
            interp_pts=ikw.get("interp_pts", 4),
            inner_N=ikw.get("inner_N", 20),
        )[0]  # (total, n_target)
    # Decode obs dim 0.
    with torch.no_grad():
        z_padded_out = lit_model._pad_to_full_dim(z_pred.unsqueeze(0))
        obs = lit_model.decode_trajectory(z_padded_out)[0][..., 0].cpu().numpy()
    # Lyapunov: compute only for configs with enough post-init length.
    lam = None
    if cfg_["lyap"]:
        with torch.no_grad():
            jacs = lit_model.compute_jacobians(z_pred.unsqueeze(0))   # (1, total, D, D)
            # Drop first init_n + 50 steps so we're past any init influence.
            drop = init_n + 50
            lam = LitLatentJacobianODE.compute_lyapunov_exponents(
                jacs[:, drop:, :, :], float(dt),
            )[0].cpu().numpy()
    # Amplitude stat for "collapsed vs. chaotic" at a glance.
    post_init = obs[init_n:]
    std_post = float(post_init.std())
    results.append(dict(
        **cfg_,
        obs=obs,
        lam=lam,
        std_post=std_post,
    ))
    print(f"  {cfg_['name']:30s} init={init_n:>3} roll={roll_n:>4} "
          f"std(post-init)={std_post:.4g}"
          + (f"  λ={lam.tolist()}" if lam is not None else ""))

# Empirical Lorenz reference.
empirical = np.array([0.906, 0.0, -14.572])

# %%
# --- Plot 1: trajectories overlaid ---
true_obs = traj_full[0, :, 0].cpu().numpy()
fig, axes = plt.subplots(len(configs), 1, figsize=(14, 2.6 * len(configs)),
                        sharex=False)
for ax, r in zip(axes, results):
    init_n = r["init"]
    xs = np.arange(len(r["obs"]))
    ax.plot(np.arange(len(true_obs)), true_obs, color="C0", lw=0.8, alpha=0.7,
            label="true")
    ax.plot(xs, r["obs"], color="C3", lw=1.2, label="model rollout")
    ax.axvline(init_n, color="k", ls=":", lw=0.6,
               label=f"end of init ({init_n})")
    ax.set_title(
        f"{r['name']}: init={init_n}, rollout={r['roll']}, "
        f"std(post-init)={r['std_post']:.3g}"
    )
    ax.set_ylabel("obs dim 0")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time step")
fig.suptitle(
    f"Run {RUN_ID}: rollout trajectory vs. init length"
    f"  (integration kwargs pinned to training config)", y=1.002
)
fig.tight_layout()
out1 = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox") / \
       f"init_length_rollout_{RUN_ID}.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"Saved {out1}")

# --- Plot 2: Lyapunov spectra ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
x = np.arange(n_target)
w = 0.20
offsets = np.linspace(-1.5, 1.5, max(1, sum(1 for r in results if r["lam"] is not None) + 1)) * w
idx = 0
for r in results:
    if r["lam"] is None:
        continue
    lam_sorted = np.sort(r["lam"])[::-1]
    ax2.bar(x + offsets[idx], lam_sorted, w,
            label=f"init={r['init']}, roll={r['roll']}  (std={r['std_post']:.2g})")
    idx += 1
ax2.bar(x + offsets[idx], empirical, w, color="k", alpha=0.6,
        label="empirical (Lorenz 0.906, 0, −14.572)")
ax2.axhline(0, color="k", lw=0.5, ls="--")
ax2.set_xlabel("Lyapunov index")
ax2.set_ylabel(r"$\lambda_i$")
ax2.set_title(f"Run {RUN_ID}: Lyapunov spectrum vs. init length")
ax2.legend(loc="lower left", fontsize=8)
ax2.set_xticks(x)
fig2.tight_layout()
out2 = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox") / \
       f"init_length_lyapunov_{RUN_ID}.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved {out2}")
