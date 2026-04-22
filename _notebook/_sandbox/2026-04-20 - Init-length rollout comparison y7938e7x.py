# %% [markdown]
# # Diagnostic rerun: 16s8uli1 with the same format as 39prin7v
#
# Five Lyapunov estimates + three rollout panels for the run whose
# long_trajectory originally showed fixed-point collapse.
#
#   - empirical (eq.jac on true Lorenz test data) — pulled from sweep's
#     metrics.json (NOT canonical textbook Lorenz numbers)
#   - sweep per_run_lyapunov (static Jac along encoded true trajectory)
#   - my static recompute (cross-check)
#   - rollout: init=15 (training default), roll=500
#   - rollout: init=250 (ratio-matched, 1:2), roll=500
#   - rollout: init=45 (current Lyap-path default), roll=400
#
# All rollouts use the training config's integration kwargs.

# %%
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
from JacobianODE.jacobians import create_dataloaders
from JacobianODE.jacobians.jacobianODE import JacobianODEint
from JacobianODE.models.latent_jacobian import LitLatentJacobianODE

PROJECT = "JacobianODE/Lorenz_INDpartial_N25_D1_NormTrue_T3__JacobianODE"
SWEEP_GROUP = "lorenz_partial_25d_additive_mse_uniform_p30_obsnoise005__lc_sweep"
RUN_ID = "y7938e7x"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METRICS_PATH = (
    Path("/home/adameisen/Documents/jacobian-analyses")
    / "Lorenz_INDpartial_N25_D1_NormTrue_T3__JacobianODE"
    / SWEEP_GROUP
    / "metrics.json"
)
_m = json.loads(METRICS_PATH.read_text())
empirical = np.asarray(_m["empirical_lyapunov_spectrum"], dtype=float)
sweep_pr = np.asarray(
    _m["per_run_lyapunov"][RUN_ID]["lambda_spectrum"], dtype=float
)
print(f"empirical (eq.jac): {empirical}")
print(f"sweep per_run_lyapunov (static Jac on true traj): {sweep_pr}")
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
print(f"n_target_dims={n_target}, n_delays="
      f"{cfg.data.train_test_params.delay_embedding_params.n_delays}, "
      f"obs_noise={cfg.data.postprocessing.obs_noise}, dt={float(dt):.5g}")
print(f"training kwargs: {ikw}")

# %%
traj_full = torch.as_tensor(trajs["test_trajs"].sequence[[0]]).float().to(DEVICE)
with torch.no_grad():
    z_full_encoded = lit_model.encode_trajectory(traj_full)
    z_dyn_true, _ = lit_model._split_latent(z_full_encoded)
T_full = z_dyn_true.shape[1]
print(f"T_full: {T_full}")

with torch.no_grad():
    jacs_static = lit_model.compute_jacobians(z_dyn_true)
    lam_static = LitLatentJacobianODE.compute_lyapunov_exponents(
        jacs_static, float(dt),
    )[0].cpu().numpy()
print(f"(1) static (my recompute): {np.sort(lam_static)[::-1]}")

jac_ode = JacobianODEint(lit_model.compute_jacobians, float(dt))

def rollout_and_lyap(init_n, roll_n):
    if init_n > T_full:
        raise ValueError(f"need {init_n} init points, have {T_full}")
    z_init = z_dyn_true[:, :init_n, :]
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
        )[0]
        z_pad_out = lit_model._pad_to_full_dim(z_pred.unsqueeze(0))
        obs = lit_model.decode_trajectory(z_pad_out)[0][..., 0].cpu().numpy()
    drop = init_n + 50
    with torch.no_grad():
        jacs = lit_model.compute_jacobians(z_pred.unsqueeze(0))
        lam = LitLatentJacobianODE.compute_lyapunov_exponents(
            jacs[:, drop:, :, :], float(dt),
        )[0].cpu().numpy()
    return obs, lam, z_pred.cpu().numpy()

configs = [
    dict(name="rollout: init=15 train default, roll=500", init=15, roll=500),
    dict(name="rollout: init=250 ratio-matched (½ of roll)", init=250, roll=500),
    dict(name="rollout: init=45 Lyap-path default, roll=400", init=45, roll=400),
]
results = []
for c in configs:
    obs, lam, z_pred = rollout_and_lyap(c["init"], c["roll"])
    post = obs[c["init"]:]
    lam_sorted = np.sort(lam)[::-1]
    c.update(obs=obs, lam=lam_sorted, std_post=float(post.std()))
    results.append(c)
    print(f"{c['name']}: std(post-init)={c['std_post']:.4g}  λ={lam_sorted.tolist()}")

# %% trajectory plot
true_obs = traj_full[0, :, 0].cpu().numpy()
fig, axes = plt.subplots(len(results), 1, figsize=(14, 2.6 * len(results)))
if len(results) == 1: axes = [axes]
for ax, r in zip(axes, results):
    init_n = r["init"]
    xs = np.arange(len(r["obs"]))
    ax.plot(np.arange(len(true_obs)), true_obs, color="C0", lw=0.8, alpha=0.7, label="true")
    ax.plot(xs, r["obs"], color="C3", lw=1.2, label="model rollout")
    ax.axvline(init_n, color="k", ls=":", lw=0.6, label=f"end of init ({init_n})")
    ax.set_title(
        f"{r['name']}: std(post-init)={r['std_post']:.3g}  "
        f"λ=[{r['lam'][0]:+.3f}, {r['lam'][1]:+.3f}, {r['lam'][2]:+.3f}]"
    )
    ax.set_ylabel("obs dim 0")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time step")
fig.suptitle(
    f"Run {RUN_ID} (p30 obsnoise005 LC sweep chosen, n_delays=25, LC=1e-5):\n"
    f"rollout behavior vs. init length",
    y=1.002
)
fig.tight_layout()
out1 = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox") / \
       f"init_length_rollout_{RUN_ID}.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"Saved {out1}")

# %% Lyapunov bar plot
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
x = np.arange(n_target)
w = 0.15
bars_data = [
    ("empirical (eq.jac on true Lorenz)", empirical, "k"),
    ("sweep per_run_lyapunov (static Jac on true traj)", np.sort(sweep_pr)[::-1], "#606060"),
    ("(1) static — my recompute", np.sort(lam_static)[::-1], "C7"),
]
for c in results:
    label = c["name"] + f"  std={c['std_post']:.2g}"
    bars_data.append((label, c["lam"], None))
offsets = np.linspace(-((len(bars_data)-1)*w)/2, ((len(bars_data)-1)*w)/2, len(bars_data))
colors = ["k", "#606060", "C7", "C0", "C1", "C3"]
for (label, vals, col), off in zip(bars_data, offsets):
    ax2.bar(x + off, vals, w, label=label, color=col)
ax2.axhline(0, color="k", lw=0.5, ls="--")
ax2.set_xticks(x)
ax2.set_xlabel("Lyapunov index")
ax2.set_ylabel(r"$\lambda_i$")
ax2.set_title(
    f"Run {RUN_ID}: Lyapunov spectrum, five ways.\n"
    f"(empirical is from eq.jac on TRUE Lorenz data for THIS sweep, not canonical numbers.)"
)
ax2.legend(loc="lower left", fontsize=8)
fig2.tight_layout()
out2 = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox") / \
       f"init_length_lyapunov_{RUN_ID}.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved {out2}")
