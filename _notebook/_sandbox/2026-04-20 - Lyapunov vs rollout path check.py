# %% [markdown]
# # Discrepancy investigation: Lyapunov says chaotic, long-rollout says fixed point
#
# Hypothesis: the batch+burn-in Lyapunov computation (run_analytics.py
# line ~1476-1488) calls ``jacobian_odeint.generate_dynamics`` directly
# with custom ``interp_pts=4, inner_N=20``, whereas the long_trajectory
# plot uses ``trajectory_model_step`` which forwards the config's
# ``jacobianODEint_kwargs`` (``interp_pts=15, inner_N=4``). If those two
# integration schedules produce materially different dynamics, the
# Lyapunov-from-rollout analysis and the long-trajectory plot are
# describing *different systems*.
#
# Load run 16s8uli1 (p30 obsnoise005 ndelays sweep, n_delays=50, the
# chosen run whose long-trajectory showed fixed-point collapse). Roll
# out the same initial condition two ways and compare.

# %%
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
from JacobianODE.jacobians import create_dataloaders
from JacobianODE.jacobians.jacobianODE import JacobianODEint

PROJECT = "JacobianODE/Lorenz_INDpartial_NDsweep_D1_NormTrue__JacobianODE"
RUN_ID = "16s8uli1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
run, cfg, eq, dt, values, _, _, _, _, lit_model = load_run(
    PROJECT, run_id=RUN_ID, generate_data=True, verbose=False,
)
train_dl, val_dl, test_dl, trajs = create_dataloaders(
    cfg, values, verbose=False, return_full_obs=True,
)
load_checkpoint(run, cfg, lit_model, epoch=None, verbose=False)
lit_model = lit_model.to(DEVICE).eval()

print(f"n_delays={cfg.data.train_test_params.delay_embedding_params.n_delays}")
print(f"prediction_steps={cfg.model.prediction_steps}")
print(f"dt={float(dt):.5g}")
print(f"trajectory_training jacobianODEint_kwargs: "
      f"{dict(cfg.training.lightning.jacobianODEint_kwargs)}")

# %%
# Single initial condition — first test trajectory (same as long_trajectory plot).
traj_long = torch.as_tensor(trajs["test_trajs"].sequence[[0]]).float().to(DEVICE)
print(f"traj_long shape: {tuple(traj_long.shape)}")

# Path A: long_trajectory plot path — trajectory_model_step with config kwargs.
with torch.no_grad():
    rd = lit_model.trajectory_model_step(
        traj_long, alpha_teacher_forcing=0.0, obs_noise_scale=0,
        strided=False, return_decoded=True,
    )
    z_pred_A = rd["outputs"][0]            # (T, D_dyn)  rolled-out latent
print(f"Path A (trajectory_model_step, full-length free rollout) z_pred shape: "
      f"{tuple(z_pred_A.shape)}")

# Path B mirrors EXACTLY what run_analytics.py does for the
# batch+burn-in Lyapunov: take a seq_length-long window (T_true = seq_length
# from training = 45 for p30), encode it, then roll free for 400 burn-in
# steps. This is the analysis the positive λ_max came from.
n_target = lit_model.n_target_dims
seq_length = int(cfg.data.train_test_params.seq_length)  # 45
window = traj_long[:, :seq_length, :]  # (1, seq_length, D_obs)
z_full_win = lit_model.encode_trajectory(window)
z_dyn_win, _ = lit_model._split_latent(z_full_win)
T_true = z_dyn_win.shape[1]
jac_ode = JacobianODEint(lit_model.compute_jacobians, float(dt))

def rollout_B_with_burn(burn_in_steps):
    padded = torch.cat(
        [z_dyn_win, torch.zeros(1, burn_in_steps, n_target, device=DEVICE)],
        dim=1,
    )
    with torch.no_grad():
        return jac_ode.generate_dynamics(
            padded,
            traj_init_steps=T_true,
            alpha_teacher_forcing=0.0,
            fast_mode=True,
            verbose=False,
            interp_pts=4,
            inner_N=20,
        )[0]

# Default Lyap-analysis burn-in.
z_pred_B = rollout_B_with_burn(400)
print(f"Path B (seq_length={seq_length} init + burn_in=400) z_pred shape: "
      f"{tuple(z_pred_B.shape)}")

# Extended burn-in to match Path A's free-rollout horizon.
z_pred_B_long = rollout_B_with_burn(1500)
print(f"Path B-long (seq_length={seq_length} init + burn_in=1500) z_pred shape: "
      f"{tuple(z_pred_B_long.shape)}")

# %%
# Decode each to obs-space (first feature = 'most recent' obs dim).
def decode_first_feature(z_dyn):
    with torch.no_grad():
        z_full = lit_model._pad_to_full_dim(z_dyn.detach().unsqueeze(0))
        obs = lit_model.decode_trajectory(z_full)[0]  # (T, n_delays)
    return obs[..., 0].detach().cpu().numpy()

obs_A = decode_first_feature(z_pred_A)
obs_B = decode_first_feature(z_pred_B)
obs_B_long = decode_first_feature(z_pred_B_long)
true_obs = traj_long[0, :, 0].cpu().numpy()
print(f"len A={len(obs_A)}  B={len(obs_B)}  B_long={len(obs_B_long)}  true={len(true_obs)}")

# %%
# Plot.
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(true_obs, lw=0.8, alpha=0.7, label="true", color="C0")
ax.plot(obs_A, lw=1.3, label=f"A: long_trajectory path (free rollout, ~{len(obs_A)} steps)",
        color="C1")
ax.plot(obs_B, lw=1.0,
        label=f"B: Lyap-analysis rollout (T_true={T_true} init + 400 burn_in)",
        color="C3", alpha=0.9)
ax.plot(obs_B_long, lw=1.0,
        label=f"B-long: same as B but 1500 burn_in (~Path A horizon)",
        color="C2", alpha=0.7, linestyle="--")
ax.axvline(T_true, color="k", ls=":", lw=0.5, label=f"B: end of init (T_true={T_true})")
ax.set_xlabel("Time step")
ax.set_ylabel("obs dim 0")
ax.set_title(
    f"Run {RUN_ID}: three rollout paths — "
    f"'long_trajectory' (A) vs Lyap-analysis rollout (B) vs training-kwarg rollout (C)"
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

out = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox") / \
      f"lyap_vs_rollout_path_{RUN_ID}.png"
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"Saved {out}")

# %%
# Also: compute per-path variance / range over the second half to
# quantify "collapse vs chaotic".
for name, obs in [("A", obs_A), ("B", obs_B), ("B_long", obs_B_long), ("true", true_obs)]:
    half = obs[len(obs) // 2:]
    print(f"{name}: std(2nd half) = {half.std():.4g}, range = {half.max() - half.min():.4g}")
