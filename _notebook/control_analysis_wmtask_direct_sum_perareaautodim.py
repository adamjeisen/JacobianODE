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
# # WMTask — Cross-Area Gramians: DirectSum **per-area-autodim** best run
#
# Sibling of `control_analysis_wmtask_direct_sum.py` (which used the full-
# 128 DirectSum encoder). Here we load the best run from the per-area
# PCA-auto DirectSum sweep — `wmtask_direct_sum_additive_splitmode_p30_perareaautodim__lc_x_obsnoisescale_sweep`
# — which has `n_target_dims_per_block=[17, 14]` (visual → 17, cognitive →
# 14), so z_dyn is 31-D.
#
# **Caveat on the GT overlay**: GT Gramians come from `eq.jac` in 128-D
# observation space. Our learned Gramians live in the 31-D z_dyn subspace.
# Under the learned encoder these coordinate systems are linked by a
# non-bijective projection (we keep only 17/64 and 14/64 dims per area).
# So absolute magnitudes won't match GT; the vis↔cog asymmetry is still
# the meaningful comparison signal.

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

# %%
PROJECT = "JacobianODE/WMTask_identity_encoder_verification"
RUN_ID = "r9mia371"  # best traj in perareaautodim sweep (LC=0, obs_noise_scale=0.05, n_target_dims_per_block=[17,14])

run, cfg, _eq_unused, dt_from_load, _values_unused, *_unused, lit_model = load_run(
    project=PROJECT, run_id=RUN_ID, verbose=True, generate_data=False,
)
lit_model.to(device).eval()
print(f"encoder: {type(lit_model.encoder).__name__}")
print(f"n_target_dims: {lit_model.n_target_dims}")
print(f"n_target_dims_per_block: {lit_model.encoder._k_per_block}")
print(f"block_sizes (input per area): {lit_model.encoder._block_sizes}")

N_VIS_DYN, N_COG_DYN = lit_model.encoder._k_per_block
LATENT_VIS = slice(0, N_VIS_DYN)
LATENT_COG = slice(N_VIS_DYN, N_VIS_DYN + N_COG_DYN)

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
dt = dt_ref if dt_from_load is None else dt_from_load

N_BATCH = 128
if N_BATCH > x_traj.shape[0]:
    N_BATCH = x_traj.shape[0]
x_batch = x_traj[:N_BATCH]
T_total = x_batch.shape[1]
print(f"trajectories: {x_batch.shape}  dt={dt}")

# %%
with torch.no_grad():
    z_full = lit_model.encode_trajectory(x_batch)
    z_dyn, _z_null = lit_model._split_latent(z_full)
    J_learned = lit_model.compute_jacobians(z_dyn).cpu()  # (B, T, 31, 31)
print(f"z_dyn: {tuple(z_dyn.shape)}")
print(f"J_learned (latent, {N_VIS_DYN+N_COG_DYN}x{N_VIS_DYN+N_COG_DYN}): {tuple(J_learned.shape)}")
print(f"  within-visual (mean |J|):  {J_learned[..., LATENT_VIS, LATENT_VIS].abs().mean():.3e}")
print(f"  within-cognitive:          {J_learned[..., LATENT_COG, LATENT_COG].abs().mean():.3e}")
print(f"  vis→cog cross J_cv:        {J_learned[..., LATENT_COG, LATENT_VIS].abs().mean():.3e}")
print(f"  cog→vis cross J_vc:        {J_learned[..., LATENT_VIS, LATENT_COG].abs().mean():.3e}")

# %%
with torch.no_grad():
    J_gt = eq_ref.jac(x_batch, t=0).cpu()
print(f"J_gt (obs-space, 128x128): {tuple(J_gt.shape)}")

VISUAL_OBS, COGNITIVE_OBS = slice(0, 64), slice(64, 128)

# %%
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

print("LEARNED vis→cog  (latent space, 17→14):")
vc_lrn = summarize(*run_gramians(J_learned, LATENT_COG, LATENT_VIS, dt))
print("LEARNED cog→vis  (latent space, 14→17):")
cv_lrn = summarize(*run_gramians(J_learned, LATENT_VIS, LATENT_COG, dt))
print("GT vis→cog  (obs space, 64→64):")
vc_gt = summarize(*run_gramians(J_gt, COGNITIVE_OBS, VISUAL_OBS, dt))
print("GT cog→vis  (obs space, 64→64):")
cv_gt = summarize(*run_gramians(J_gt, VISUAL_OBS, COGNITIVE_OBS, dt))

# %%
def _stats(x):
    return x.mean(dim=0).numpy(), x.std(dim=0).numpy()

colors = {"vis→cog": "C0", "cog→vis": "C3"}

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
t_axis = np.arange(T_total) * dt
for col, gk in enumerate(["reach", "ctrl", "obs"]):
    for row, stat in enumerate(["trace", "min"]):
        ax_gt = axes[row][col]        # left axis: ground truth (128-D obs space)
        ax_lrn = ax_gt.twinx()        # right axis: learned (31-D latent)
        handles_for_legend, labels_for_legend = [], []
        for dir_label, lrn, gt in [
            ("vis→cog", vc_lrn, vc_gt),
            ("cog→vis", cv_lrn, cv_gt),
        ]:
            c = colors[dir_label]
            m_gt, s_gt = _stats(gt[gk][stat])
            (h_gt,) = ax_gt.plot(t_axis, m_gt, c + "-", lw=2,
                                 label=f"{dir_label} (GT)")
            ax_gt.fill_between(t_axis, m_gt - s_gt, m_gt + s_gt, color=c, alpha=0.15)
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
    f"Cross-area Gramians — GROUND TRUTH (solid, 128-D obs) vs DirectSum per-area-autodim "
    f"LEARNED (dotted, 31-D latent)\n"
    f"Run {RUN_ID}  ·  n_target_dims_per_block={list(lit_model.encoder._k_per_block)}  ·  "
    f"LC=0, obs_noise_scale=0.05  ·  absolute magnitudes not directly comparable (projection)",
    y=1.01,
)
fig.tight_layout()

FIG_DIR = Path("/home/adameisen/Documents/jacobian-analyses/diagnostics")
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig_path = FIG_DIR / f"wmtask_direct_sum_perareaautodim_cross_area_gramians_vs_gt_{RUN_ID}.png"
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
              f"GT {lt_g:+6.3f}")
        rows.append(dict(direction=dir_label, gramian=gk,
                         learned_log_trace_final=lt_l, gt_log_trace_final=lt_g,
                         learned_log_min_final=lm_l, gt_log_min_final=lm_g))

summary_path = FIG_DIR / f"wmtask_direct_sum_perareaautodim_cross_area_gramians_vs_gt_{RUN_ID}.json"
summary_path.write_text(json.dumps({
    "run_id": RUN_ID,
    "group": "wmtask_direct_sum_additive_splitmode_p30_perareaautodim__lc_x_obsnoisescale_sweep",
    "lc_weight": 0.0,
    "obs_noise_scale": 0.05,
    "n_target_dims_per_block": list(lit_model.encoder._k_per_block),
    "rows": rows,
    "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
}, indent=2))
print(f"Saved summary → {summary_path}")
