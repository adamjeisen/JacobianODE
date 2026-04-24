# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # WMTask — Cross-Area Gramians: vanilla (no-encoder) JacobianODE vs GT
#
# Sibling of `control_analysis_wmtask_direct_sum.py`. The "vanilla" here
# means a JacobianODE model with NO encoder — the MLP learns to predict
# the Jacobian directly in observation space. Uses the best run from
# `wmtask_vanilla_mse_p30__lc_sweep`.
#
# Because there's no encoder, `compute_jacobians(x)` returns
# `dx_dot/dx` in observation space directly — no basis change to apply
# before comparing with ground truth.

# %%
import datetime as _dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from JacobianODE.control import compute_all_gramians
from JacobianODE.jacobians.checkpoints.loader import load_run
from wmtask.trajectories import load_wmtask_for_jacobianode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ## Load the vanilla WMTask JacobianODE model

# %%
PROJECT = "JacobianODE/WMTask_identity_encoder_verification"
RUN_ID = "pip3w5po"  # best traj val in wmtask_vanilla_mse_p30__lc_sweep (LC=1e-6)

run, cfg, _eq_unused, dt_from_load, _values_unused, _train_dl, _val_dl, _test_dl, _trajs, lit_model = load_run(
    project=PROJECT, run_id=RUN_ID, verbose=True, generate_data=False,
)
lit_model.to(device).eval()

traj_val_wandb = float(run.summary.get("trajectory val_loss"))
print(f"wandb traj_val_loss: {traj_val_wandb:.4e}")
print(f"model children: {[n for n, _ in lit_model.named_children()]}")
has_encoder = hasattr(lit_model, "encoder") and lit_model.encoder is not None
print(f"has encoder: {has_encoder}")

# %% [markdown]
# ## Load reference WMTask trajectories + ground-truth bio RNN

# %%
eq_ref, sol, dt_ref = load_wmtask_for_jacobianode(
    project="WMSelectionTask__cue_time_0.1__response_time_0.25__enforce_fixation_False",
    name=(
        "BiologicalRNN__cue_time_0.1__learning_rate_0.0005__max_epochs_42"
        "__N1_64__N2_64__tau_0.05__dt_0.02__eig_lower_bound_0.1__init_mode_random"
    ),
    model_to_load="final",
    dataloader_to_use="all",
    traj_window="delay2",
    verbose=True,
    use_cache=True,
    device=device,
)
x_traj = sol["values"]
if not torch.is_tensor(x_traj):
    x_traj = torch.as_tensor(x_traj)
x_traj = x_traj.to(device).float()
print(f"trajectories: {x_traj.shape}   dt={dt_ref}")
dt = dt_ref if dt_from_load is None else dt_from_load

N_BATCH = 128
if N_BATCH > x_traj.shape[0]:
    N_BATCH = x_traj.shape[0]
x_batch = x_traj[:N_BATCH]
T_total = x_batch.shape[1]

# %% [markdown]
# ## Compute the LEARNED obs-space Jacobian
#
# For a no-encoder model, `compute_jacobians(x)` operates directly on
# observations and returns ``dx_dot/dx`` — the obs-space local
# linearization. No encoder/decoder composition needed.

# %%
with torch.no_grad():
    J_learned = lit_model.compute_jacobians(x_batch).cpu()   # (B, T, D, D)
print(f"J_learned: {tuple(J_learned.shape)}  (obs-space, on CPU)")

# Sanity: quick check of the learned Jacobian block magnitudes by area.
J_mean = J_learned.abs().mean(dim=(0, 1))
print(f"[info] learned |J| block means:")
print(f"  within-visual:        {J_mean[:64, :64].mean():.3e}")
print(f"  within-cognitive:     {J_mean[64:, 64:].mean():.3e}")
print(f"  vis→cog (J[cog,vis]): {J_mean[64:, :64].mean():.3e}")
print(f"  cog→vis (J[vis,cog]): {J_mean[:64, 64:].mean():.3e}")

# %% [markdown]
# ## Compute GROUND TRUTH Jacobians via eq_ref.jac

# %%
with torch.no_grad():
    J_gt = eq_ref.jac(x_batch, t=0).cpu()
print(f"J_gt: {tuple(J_gt.shape)}  (on CPU)")

# %% [markdown]
# ## Cross-area Gramians: learned vs ground truth

# %%
VISUAL, COGNITIVE = slice(0, 64), slice(64, 128)


def extract_blocks(J, target, source):
    return J[..., target, target], J[..., target, source], J[..., source, target]


def run_gramians(J, target, source, dt):
    A, B, C = extract_blocks(J, target, source)
    (_, _, _), (lsr, lsc, lso) = compute_all_gramians(
        A, B, C, dt=dt,
        return_sequences=True, return_spectrums=True, rescale=True,
    )
    return lsr, lsc, lso


def summarize(lsr, lsc, lso):
    def _tr(spec): return torch.logsumexp(spec, dim=-1).cpu()
    def _mn(spec): return spec[..., -1].cpu()
    return {
        "reach": {"trace": _tr(lsr), "min": _mn(lsr)},
        "ctrl":  {"trace": _tr(lsc), "min": _mn(lsc)},
        "obs":   {"trace": _tr(lso), "min": _mn(lso)},
    }


print("LEARNED vis→cog:")
vc_lrn = summarize(*run_gramians(J_learned, COGNITIVE, VISUAL, dt))
print("LEARNED cog→vis:")
cv_lrn = summarize(*run_gramians(J_learned, VISUAL, COGNITIVE, dt))
print("GT vis→cog:")
vc_gt = summarize(*run_gramians(J_gt, COGNITIVE, VISUAL, dt))
print("GT cog→vis:")
cv_gt = summarize(*run_gramians(J_gt, VISUAL, COGNITIVE, dt))

# %% [markdown]
# ## Plot — twin axes, GT solid (left), learned dotted (right)

# %%
def _stats(x):
    return x.mean(dim=0).numpy(), x.std(dim=0).numpy()


colors = {"vis→cog": "C0", "cog→vis": "C3"}

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
t_axis = np.arange(T_total) * dt
for col, gk in enumerate(["reach", "ctrl", "obs"]):
    for row, stat in enumerate(["trace", "min"]):
        ax_gt = axes[row][col]
        ax_lrn = ax_gt.twinx()
        handles_for_legend, labels_for_legend = [], []
        for dir_label, lrn, gt in [
            ("vis→cog", vc_lrn, vc_gt),
            ("cog→vis", cv_lrn, cv_gt),
        ]:
            c = colors[dir_label]
            m_gt, s_gt = _stats(gt[gk][stat])
            (h_gt,) = ax_gt.plot(t_axis, m_gt, c + "-", lw=2,
                                 label=f"{dir_label} (GT)")
            ax_gt.fill_between(t_axis, m_gt - s_gt, m_gt + s_gt,
                               color=c, alpha=0.15)
            m, s = _stats(lrn[gk][stat])
            (h_lrn,) = ax_lrn.plot(t_axis, m, c + ":", lw=2,
                                   label=f"{dir_label} (learned)")
            ax_lrn.fill_between(t_axis, m - s, m + s, color=c, alpha=0.10)
            handles_for_legend.extend([h_gt, h_lrn])
            labels_for_legend.extend([f"{dir_label} (GT, solid)",
                                      f"{dir_label} (learned, dotted)"])
        stat_label = "log trace" if stat == "trace" else "log min eigenvalue"
        ax_gt.set_title(f"{gk}  |  {stat_label}")
        ax_gt.set_xlabel("window time (s)")
        ax_gt.set_ylabel("GT  log λ")
        ax_lrn.set_ylabel("learned  log λ")
        ax_gt.grid(True, alpha=0.3)
        if col == 0 and row == 0:
            ax_gt.legend(handles_for_legend, labels_for_legend, loc="best", fontsize=7)
fig.suptitle(
    f"Cross-area Gramians — GROUND TRUTH (solid, left axis) vs VANILLA no-encoder "
    f"JacobianODE (dotted, right axis)\n"
    f"Run {RUN_ID} (group wmtask_vanilla_mse_p30__lc_sweep, LC=1e-6) · "
    f"obs-space Jacobians",
    y=1.01,
)
fig.tight_layout()

FIG_DIR = Path("/home/adameisen/Documents/jacobian-analyses/diagnostics")
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig_path = FIG_DIR / f"wmtask_vanilla_cross_area_gramians_vs_gt_{RUN_ID}.png"
fig.savefig(fig_path, dpi=130, bbox_inches="tight")
print(f"\nSaved → {fig_path}")

# %%
print(f"\n=== Terminal-step: learned vs ground truth ===")
rows = []
for dir_label, lrn, gt in [("vis→cog", vc_lrn, vc_gt), ("cog→vis", cv_lrn, cv_gt)]:
    for gk in ["reach", "ctrl", "obs"]:
        lt_l = float(lrn[gk]["trace"][:, -1].mean())
        lm_l = float(lrn[gk]["min"][:, -1].mean())
        lt_g = float(gt[gk]["trace"][:, -1].mean())
        lm_g = float(gt[gk]["min"][:, -1].mean())
        print(f"  {dir_label:>10}  {gk:>5}  learned log-tr {lt_l:+6.3f}  "
              f"GT {lt_g:+6.3f}  (Δ {lt_l - lt_g:+6.3f})")
        rows.append(dict(direction=dir_label, gramian=gk,
                         learned_log_trace_final=lt_l, gt_log_trace_final=lt_g,
                         learned_log_min_final=lm_l, gt_log_min_final=lm_g))

summary_path = FIG_DIR / f"wmtask_vanilla_cross_area_gramians_vs_gt_{RUN_ID}.json"
summary_path.write_text(json.dumps({
    "run_id": RUN_ID,
    "group": "wmtask_vanilla_mse_p30__lc_sweep",
    "model_kind": "no_encoder_vanilla",
    "n_trajectories": N_BATCH,
    "T": T_total,
    "dt": dt,
    "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    "rows": rows,
    "wandb_traj_val_loss": traj_val_wandb,
}, indent=2))
print(f"Saved summary → {summary_path}")
