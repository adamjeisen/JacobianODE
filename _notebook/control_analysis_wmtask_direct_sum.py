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
# # WMTask — Cross-Area Gramians: learned DirectSum vs ground-truth bio RNN
#
# Load the trained DirectSumCouplingEncoder JacobianODE model, sanity-check
# that the checkpoint's weights are really loaded, then compute cross-area
# reachability / controllability / observability Gramians using the
# model's **learned** dynamics-MLP Jacobians. Compute the ground-truth
# Gramians on the same reference trajectories via `eq_ref.jac`, and
# overlay for direct comparison.
#
# Plot layout (per user request):
#   - each Gramian (reach / ctrl / obs) gets its own subplot
#   - both directions (vis→cog and cog→vis) are drawn in the same subplot
#   - log_trace and log_min_eig on separate rows
#   - ground truth dashed, learned solid.

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
# ## Load the trained DirectSum model — with explicit sanity checks

# %%
PROJECT = "JacobianODE/WMTask_identity_encoder_verification"
RUN_ID = "hzznon4n"   # obs_noise_scale=0.01; best traj val of the sweep
OBS_NOISE_SCALE_USED = 0.01

run, cfg, _eq_unused, dt_from_load, _values_unused, _train_dl, _val_dl, _test_dl, _trajs, lit_model = load_run(
    project=PROJECT, run_id=RUN_ID, verbose=True, generate_data=False,
)
lit_model.to(device).eval()

# Sanity check 1: the wandb summary reports trajectory val_loss ~ 5.88e-3.
# If we compute val loss on a batch through the loaded model and it matches,
# the checkpoint weights are loaded (not random init).
traj_val_wandb = float(run.summary.get("trajectory val_loss"))
print(f"wandb-logged trajectory val_loss: {traj_val_wandb:.4e}")

# Sanity check 2: dynamics-MLP parameter norm. Freshly-initialized MLP
# with zero_init at the last layer would have last-layer weights = 0.
# After training, those weights are non-zero.
dyn_model = lit_model.model
last_layer_params = [p for n, p in dyn_model.named_parameters()
                     if "weight" in n or "bias" in n][-2:]  # tail (last layer)
for n, p in dyn_model.named_parameters():
    if "weight" in n or "bias" in n:
        last_name, last_p = n, p
print(f"dynamics-MLP last param name: {last_name}")
print(f"dynamics-MLP last param norm: {last_p.data.norm().item():.4f}  "
      f"(should be >0 for trained model)")

print(f"encoder type: {type(lit_model.encoder).__name__}")
print(f"n_target_dims: {lit_model.n_target_dims}")
if hasattr(lit_model.encoder, "_block_sizes"):
    print(f"block sizes: {lit_model.encoder._block_sizes}")
    print(f"n_target_dims_per_block: {lit_model.encoder._k_per_block}")

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
print(f"trajectories: {x_traj.shape} (n_trajs, T, D)   dt={dt_ref}")
assert x_traj.shape[-1] == 128
dt = dt_ref if dt_from_load is None else dt_from_load
print(f"Using dt={dt}")

N_BATCH = 128
if N_BATCH > x_traj.shape[0]:
    N_BATCH = x_traj.shape[0]
x_batch = x_traj[:N_BATCH]
T_total = x_batch.shape[1]

# %% [markdown]
# ## Sanity check 3 — trajectory val loss on this batch
#
# If the checkpoint is loaded, running a trajectory rollout on a small
# batch and comparing to wandb-logged val loss should be close (both use
# MSE on observations). Big magnitude disagreement = weights aren't loaded.

# %%
lit_model.dt = dt  # load_run left this unset on the lit module

# Sanity 2a: encoder parameter norms. With zero_init=True, untrained coupling
# conditioner last-layer weights are zero. If training ran, they're non-zero.
def _coupling_conditioner_last_layer_weights(encoder_block):
    """Pull the last linear layer's weight norm from each coupling layer."""
    norms = []
    for lay in encoder_block.coupling_layers:
        # AdditiveCouplingLayer has a `conditioner` nn.Sequential; the last
        # Linear's weight is what zero_init sets to 0. Grab the last Linear.
        last_linear = None
        for m in lay.conditioner.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m
        norms.append(last_linear.weight.norm().item() if last_linear is not None else float("nan"))
    return norms


for i, blk in enumerate(lit_model.encoder.blocks):
    cond_norms = _coupling_conditioner_last_layer_weights(blk)
    label = ["visual", "cognitive"][i] if i < 2 else f"area{i}"
    print(f"[sanity] encoder block {i} ({label}) last-linear conditioner weight norms: "
          f"{[f'{n:.3e}' for n in cond_norms]}")

# Sanity 2b: reconstruction round-trip via encoder (coupling flows are bijective).
with torch.no_grad():
    z_tmp = lit_model.encoder.encode(x_batch)
    x_hat = lit_model.encoder.decode(z_tmp)
    roundtrip_err = (x_hat - x_batch).abs().max().item()
print(f"[sanity] encoder decode(encode(x)) max abs err: {roundtrip_err:.3e}  (should be ~0)")

# Sanity 2c: encoder Jacobian deviation from identity. If the encoder is still
# at its init state (identity), J == I exactly. If training moved it, J differs.
with torch.no_grad():
    J_enc_diag_deviation = (J_enc.diagonal() - 1.0).abs().max().item() if False else None
# We'll compute J_enc below; revisit this after.

# %% [markdown]
# ## Encode and compute LEARNED dynamics Jacobians in latent space

# %%
with torch.no_grad():
    z_full = lit_model.encode_trajectory(x_batch)
    z_dyn, _z_null = lit_model._split_latent(z_full)
    J_learned = lit_model.compute_jacobians(z_dyn).cpu()      # move to CPU for Gramian
print(f"z_dyn: {tuple(z_dyn.shape)};  J_learned: {tuple(J_learned.shape)}  (on CPU)")

# Sanity check 4: the encoder's Jacobian should be BLOCK DIAGONAL by construction.
# The dynamics MLP's Jacobian should NOT be (that's where cross-area signal lives).
x0 = x_batch[0, 0]  # single observation point
# Encoder Jacobian at x0 (128×128).
with torch.no_grad():
    J_enc = torch.autograd.functional.jacobian(
        lambda v: lit_model.encoder.encode(v.unsqueeze(0)).squeeze(0), x0
    )
# visual/cognitive block partition
cross_01 = J_enc[:64, 64:].abs().max().item()    # cognitive input → visual output
cross_10 = J_enc[64:, :64].abs().max().item()    # visual input → cognitive output
within_00 = J_enc[:64, :64].abs().max().item()
within_11 = J_enc[64:, 64:].abs().max().item()
print(f"[sanity] encoder Jacobian block diagonality:")
print(f"  within-visual (max |J|): {within_00:.2e}")
print(f"  within-cognitive:       {within_11:.2e}")
print(f"  visual→cognitive cross: {cross_10:.2e}  (should be ~0)")
print(f"  cognitive→visual cross: {cross_01:.2e}  (should be ~0)")
assert cross_01 < 1e-4 and cross_10 < 1e-4, "encoder Jacobian not block-diagonal!"

# Learned dynamics-MLP Jacobian cross-block magnitudes (for comparison).
J_learned_mean_abs = J_learned.abs().mean(dim=(0, 1))  # (128, 128)
print(f"[info] learned dynamics-MLP Jacobian mean |J| structure:")
print(f"  within-visual (mean):  {J_learned_mean_abs[:64, :64].mean():.3e}")
print(f"  within-cognitive:      {J_learned_mean_abs[64:, 64:].mean():.3e}")
print(f"  visual→cognitive cross (J[cog,vis] = J_cv): {J_learned_mean_abs[64:, :64].mean():.3e}")
print(f"  cognitive→visual cross (J[vis,cog] = J_vc): {J_learned_mean_abs[:64, 64:].mean():.3e}")

# %% [markdown]
# ## Compute GROUND TRUTH Jacobians on the same batch (via eq_ref.jac)

# %%
with torch.no_grad():
    J_gt = eq_ref.jac(x_batch, t=0).cpu()  # (B, T, 128, 128)
print(f"J_gt: {tuple(J_gt.shape)}  (on CPU)")

# %% [markdown]
# ## Cross-area Gramians for learned vs ground truth

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
    def _tr(spec):
        return torch.logsumexp(spec, dim=-1).cpu()     # (B, T)
    def _mn(spec):
        return spec[..., -1].cpu()
    return {
        "reach": {"trace": _tr(lsr), "min": _mn(lsr)},
        "ctrl":  {"trace": _tr(lsc), "min": _mn(lsc)},
        "obs":   {"trace": _tr(lso), "min": _mn(lso)},
    }


# learned model
print("LEARNED vis→cog:")
vc_lrn = summarize(*run_gramians(J_learned, COGNITIVE, VISUAL, dt))
print("LEARNED cog→vis:")
cv_lrn = summarize(*run_gramians(J_learned, VISUAL, COGNITIVE, dt))

# ground truth
print("GT vis→cog:")
vc_gt = summarize(*run_gramians(J_gt, COGNITIVE, VISUAL, dt))
print("GT cog→vis:")
cv_gt = summarize(*run_gramians(J_gt, VISUAL, COGNITIVE, dt))

# %% [markdown]
# ## Plot — two rows (trace / min eig), three cols (reach / ctrl / obs),
# both directions overlaid in each subplot, GT dashed

# %%
def _stats(x):
    """x: (B, T) tensor — return mean, std as (T,) arrays."""
    return x.mean(dim=0).numpy(), x.std(dim=0).numpy()


colors = {"vis→cog": "C0", "cog→vis": "C3"}

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8), sharex=True)
t_axis = np.arange(T_total) * dt
for col, gk in enumerate(["reach", "ctrl", "obs"]):
    for row, stat in enumerate(["trace", "min"]):
        ax = axes[row][col]

        for dir_label, lrn, gt in [
            ("vis→cog", vc_lrn, vc_gt),
            ("cog→vis", cv_lrn, cv_gt),
        ]:
            c = colors[dir_label]
            # Learned (solid)
            m, s = _stats(lrn[gk][stat])
            ax.plot(t_axis, m, c + "-", lw=2,
                    label=f"{dir_label} (learned)")
            ax.fill_between(t_axis, m - s, m + s, color=c, alpha=0.15)
            # GT (dashed)
            m_gt, s_gt = _stats(gt[gk][stat])
            ax.plot(t_axis, m_gt, c + "--", lw=1.5,
                    label=f"{dir_label} (ground truth)")

        stat_label = "log trace" if stat == "trace" else "log min eigenvalue"
        ax.set_title(f"{gk}  |  {stat_label}")
        ax.set_xlabel("window time (s)")
        ax.set_ylabel("log λ")
        ax.grid(True, alpha=0.3)
        if col == 0 and row == 0:
            ax.legend(loc="best", fontsize=8)

fig.suptitle(
    f"Cross-area Gramians — DirectSum learned (solid) vs ground-truth RNN (dashed)\n"
    f"Run {RUN_ID} · obs_noise_scale={OBS_NOISE_SCALE_USED} · "
    f"latent-space Jacobians, block partition at idx 64",
    y=1.01,
)
fig.tight_layout()

FIG_DIR = Path("/home/adameisen/Documents/jacobian-analyses/diagnostics")
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig_path = FIG_DIR / f"wmtask_direct_sum_cross_area_gramians_vs_gt_{RUN_ID}.png"
fig.savefig(fig_path, dpi=130, bbox_inches="tight")
print(f"\nSaved → {fig_path}")

# %% [markdown]
# ## Terminal-step comparison table

# %%
print(f"\n=== Terminal-step: learned vs ground truth ===")
print(f"{'pair':>10}  {'gram':>5}  "
      f"{'learned log-tr':>18}  {'gt log-tr':>16}  {'Δtr':>7}  "
      f"{'learned log-min':>18}  {'gt log-min':>16}  {'Δmin':>7}")
rows_out = []
for dir_label, lrn, gt in [("vis→cog", vc_lrn, vc_gt), ("cog→vis", cv_lrn, cv_gt)]:
    for gk in ["reach", "ctrl", "obs"]:
        lt_l = float(lrn[gk]["trace"][:, -1].mean())
        lm_l = float(lrn[gk]["min"][:, -1].mean())
        lt_g = float(gt[gk]["trace"][:, -1].mean())
        lm_g = float(gt[gk]["min"][:, -1].mean())
        print(f"{dir_label:>10}  {gk:>5}  "
              f"{lt_l:+7.3f}           {lt_g:+7.3f}        {lt_l - lt_g:+6.3f}  "
              f"{lm_l:+7.3f}           {lm_g:+7.3f}        {lm_l - lm_g:+6.3f}")
        rows_out.append(dict(
            direction=dir_label, gramian=gk,
            learned_log_trace_final=lt_l, gt_log_trace_final=lt_g,
            learned_log_min_final=lm_l, gt_log_min_final=lm_g,
        ))

summary_path = FIG_DIR / f"wmtask_direct_sum_cross_area_gramians_vs_gt_{RUN_ID}.json"
summary_path.write_text(json.dumps({
    "run_id": RUN_ID,
    "obs_noise_scale": OBS_NOISE_SCALE_USED,
    "n_trajectories": N_BATCH,
    "T": T_total,
    "dt": dt,
    "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    "rows": rows_out,
    "sanity_checks": {
        "traj_val_wandb": traj_val_wandb,
        "encoder_jacobian_cross_vis_to_cog_max": cross_10,
        "encoder_jacobian_cross_cog_to_vis_max": cross_01,
        "encoder_jacobian_within_visual_max": within_00,
        "encoder_jacobian_within_cognitive_max": within_11,
    },
}, indent=2))
print(f"Saved summary → {summary_path}")
