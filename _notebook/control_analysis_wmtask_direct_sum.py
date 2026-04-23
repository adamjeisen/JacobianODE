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
# # WMTask — Cross-Area Control Analysis via LEARNED Direct-Sum Jacobians
#
# Adapts `control_analysis_wmtask_cross_area.py` (which used ground-truth
# `eq.jac` from the trained bio RNN) to instead use the **learned**
# Jacobians from the DirectSumCouplingEncoder JacobianODE run. The test
# is whether our trained model's local linearization in latent space
# reproduces the cross-area controllability structure of the reference
# RNN — that's the whole point of the block-diagonal encoder.
#
# Run used: `hzznon4n` (obs_noise_scale=0.01, best traj val loss 5.88e-3)
# from the 3-run DirectSum sweep.

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
# ## Load the trained DirectSumCouplingEncoder JacobianODE model

# %%
PROJECT = "JacobianODE/WMTask_identity_encoder_verification"
RUN_ID = "hzznon4n"  # obs_noise_scale=0.01, best traj val of the 3-run sweep

run, cfg, eq_unused, dt, values, _train_dl, _val_dl, _test_dl, _trajs, lit_model = load_run(
    project=PROJECT,
    run_id=RUN_ID,
    verbose=True,
    generate_data=False,
)
lit_model.to(device).eval()
print(f"loaded: {run.name}  (state={run.state})")
print(f"encoder type: {type(lit_model.encoder).__name__}")
print(f"n_target_dims: {lit_model.n_target_dims}")
if hasattr(lit_model.encoder, "_block_sizes"):
    print(f"block sizes: {lit_model.encoder._block_sizes}")
    print(f"n_target_dims_per_block: {lit_model.encoder._k_per_block}")

# %% [markdown]
# ## Load reference trajectories (same as the ground-truth analysis)

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
print(f"trajectories: {x_traj.shape} (n_trajs, T, D)   dt={dt_ref}")
assert x_traj.shape[-1] == 128, f"expected 128-D obs, got {x_traj.shape[-1]}"
if dt is not None:
    assert abs(dt - dt_ref) < 1e-10, f"dt mismatch: model dt={dt} vs ref={dt_ref}"
else:
    # load_run didn't surface dt — fall back to the reference dt.
    dt = dt_ref

# %% [markdown]
# ## Batch + encode to latent z_dyn

# %%
VISUAL = slice(0, 64)      # z_dyn_vis (first n_target_dims_per_block[0])
COGNITIVE = slice(64, 128) # z_dyn_cog
N_BATCH = 128
n_available = x_traj.shape[0]
if N_BATCH > n_available:
    N_BATCH = n_available
x_batch = x_traj[:N_BATCH]            # (N_BATCH, T, 128)
T_total = x_batch.shape[1]

with torch.no_grad():
    # encode_trajectory handles (B, T, D) → (B, T, n_latent).
    z_full = lit_model.encode_trajectory(x_batch)
    z_dyn, _z_null = lit_model._split_latent(z_full)
print(f"z_dyn: {tuple(z_dyn.shape)}  (should be ({N_BATCH}, {T_total}, 128))")

# %% [markdown]
# ## Compute the dynamics-MLP Jacobian along the z_dyn trajectory
#
# `compute_jacobians` returns `dz_dot/dz_dyn` (= local linearization of
# the learned latent dynamics) with shape (B, T, D_dyn, D_dyn). For
# DirectSumCouplingEncoder with n_target_dims_per_block=[64, 64], the
# first 64 rows/cols are the visual-area latent dims, the next 64 are
# cognitive — so the area-block partition on this Jacobian corresponds
# directly to area-block partition on the observation.

# %%
with torch.no_grad():
    J_latent = lit_model.compute_jacobians(z_dyn)  # (B, T, 128, 128)
print(f"J_latent: {tuple(J_latent.shape)}")

# %% [markdown]
# ## Block extraction and Gramian analysis
#
# Matching the reference analysis (`control_analysis_wmtask_cross_area.py`):
# for each ordered pair (source → target), state = target, input = source,
# output = source.

# %%
def extract_blocks(J, target, source):
    A = J[..., target, target]
    B = J[..., target, source]
    C = J[..., source, target]
    return A, B, C


def run_area_gramians(J, target, source, dt):
    A, B, C = extract_blocks(J, target, source)
    print(f"  A:{tuple(A.shape)}, B:{tuple(B.shape)}, C:{tuple(C.shape)}")
    (_, _, _), (log_spec_r, log_spec_c, log_spec_o) = compute_all_gramians(
        A, B, C, dt=dt,
        return_sequences=True,
        return_spectrums=True,
        rescale=True,
    )
    return log_spec_r, log_spec_c, log_spec_o


print("VISUAL → COGNITIVE")
vc_spec_r, vc_spec_c, vc_spec_o = run_area_gramians(J_latent, COGNITIVE, VISUAL, dt_ref)
print("COGNITIVE → VISUAL")
cv_spec_r, cv_spec_c, cv_spec_o = run_area_gramians(J_latent, VISUAL, COGNITIVE, dt_ref)

# %% [markdown]
# ## Summaries + plot

# %%
def _log_trace(log_spec): return torch.logsumexp(log_spec, dim=-1)
def _log_min_eig(log_spec): return log_spec[..., -1]

def _summary(log_spec):
    lt = _log_trace(log_spec).cpu()
    lm = _log_min_eig(log_spec).cpu()
    return dict(
        log_trace_mean=lt.mean(dim=0).numpy(),
        log_trace_std=lt.std(dim=0).numpy(),
        log_min_mean=lm.mean(dim=0).numpy(),
        log_min_std=lm.std(dim=0).numpy(),
    )


summaries = {
    "visual→cognitive": {
        "reach": _summary(vc_spec_r),
        "ctrl":  _summary(vc_spec_c),
        "obs":   _summary(vc_spec_o),
    },
    "cognitive→visual": {
        "reach": _summary(cv_spec_r),
        "ctrl":  _summary(cv_spec_c),
        "obs":   _summary(cv_spec_o),
    },
}

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
t_axis = np.arange(T_total) * dt_ref
for row, pair in enumerate(summaries):
    for col, gk in enumerate(["reach", "ctrl", "obs"]):
        ax = axes[row][col]
        s = summaries[pair][gk]
        ax.plot(t_axis, s["log_trace_mean"], "C0-", lw=2, label="log trace")
        ax.fill_between(t_axis,
                        s["log_trace_mean"] - s["log_trace_std"],
                        s["log_trace_mean"] + s["log_trace_std"],
                        color="C0", alpha=0.2)
        ax.plot(t_axis, s["log_min_mean"], "C3-", lw=2, label="log min-eig")
        ax.fill_between(t_axis,
                        s["log_min_mean"] - s["log_min_std"],
                        s["log_min_mean"] + s["log_min_std"],
                        color="C3", alpha=0.2)
        ax.set_title(f"{pair}  |  {gk}")
        ax.set_xlabel("window time (s)")
        ax.set_ylabel("log λ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
fig.suptitle(
    f"Cross-area Gramians — DirectSumCouplingEncoder model ({RUN_ID}, "
    f"obs_noise_scale=0.01)\n"
    "Latent-space Jacobians; block partition at idx 64 (visual vs cognitive)",
    y=1.02,
)
fig.tight_layout()

FIG_DIR = Path("/home/adameisen/Documents/jacobian-analyses/diagnostics")
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig_path = FIG_DIR / f"wmtask_direct_sum_cross_area_gramians_{RUN_ID}.png"
fig.savefig(fig_path, dpi=130, bbox_inches="tight")
print(f"Saved → {fig_path}")

# %% [markdown]
# ## Terminal-step summary table (comparable to ground-truth report)

# %%
def _final(arr): return float(arr[-1])

rows = []
for pair, by_g in summaries.items():
    for gk, s in by_g.items():
        rows.append(dict(
            pair=pair, gramian=gk,
            log_trace_final=_final(s["log_trace_mean"]),
            log_trace_std_final=_final(s["log_trace_std"]),
            log_min_final=_final(s["log_min_mean"]),
            log_min_std_final=_final(s["log_min_std"]),
        ))

print(f"\n=== Terminal-step summary (run {RUN_ID}, latent-space J) ===")
print(f"{'pair':>20}  {'gram':>5}  {'log_trace':>16}  {'log_min':>16}")
for r in rows:
    print(f"{r['pair']:>20}  {r['gramian']:>5}  "
          f"{r['log_trace_final']:+6.3f} ± {r['log_trace_std_final']:.3f}  "
          f"{r['log_min_final']:+7.3f} ± {r['log_min_std_final']:.3f}")

# %% [markdown]
# ## Side-by-side with ground-truth report (2026-04-15)

# %%
# Ground truth from `_notebook/reports/2026-04-15 - WMTask Cross-Area Gramians.md`.
GT = {
    ("visual→cognitive", "reach"): (+4.893, 0.385, -6.624, 0.172),
    ("visual→cognitive", "ctrl"):  (+2.787, 0.051, -13.473, 0.200),
    ("visual→cognitive", "obs"):   (+0.321, 0.138, -18.894, 1.338),
    ("cognitive→visual", "reach"): (+2.678, 0.178, -7.771, 0.384),
    ("cognitive→visual", "ctrl"):  (+0.321, 0.138, -18.894, 1.338),
    ("cognitive→visual", "obs"):   (+2.787, 0.051, -13.473, 0.200),
}

print(f"\n=== Side-by-side: learned vs ground-truth ===")
print(f"{'pair':>20}  {'gram':>5}  {'learned logtr':>16}  {'gt logtr':>12}  Δlogtr  "
      f"{'learned logmin':>16}  {'gt logmin':>12}  Δlogmin")
for r in rows:
    lt_l = r["log_trace_final"]; lm_l = r["log_min_final"]
    lt_g, lt_g_std, lm_g, lm_g_std = GT[(r["pair"], r["gramian"])]
    print(
        f"{r['pair']:>20}  {r['gramian']:>5}  "
        f"{lt_l:+6.3f} ± {r['log_trace_std_final']:.2f}  "
        f"{lt_g:+6.3f} ± {lt_g_std:.2f}  {lt_l - lt_g:+6.3f}  "
        f"{lm_l:+6.3f} ± {r['log_min_std_final']:.2f}  "
        f"{lm_g:+6.3f} ± {lm_g_std:.2f}  {lm_l - lm_g:+6.3f}"
    )

# Save JSON alongside
summary_path = FIG_DIR / f"wmtask_direct_sum_cross_area_gramians_{RUN_ID}.json"
summary_path.write_text(json.dumps({
    "run_id": RUN_ID,
    "obs_noise_scale": 0.01,
    "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    "rows": rows,
    "ground_truth": {
        f"{p}|{g}": dict(log_trace=v[0], log_trace_std=v[1],
                         log_min=v[2], log_min_std=v[3])
        for (p, g), v in GT.items()
    },
}, indent=2))
print(f"Saved summary → {summary_path}")
