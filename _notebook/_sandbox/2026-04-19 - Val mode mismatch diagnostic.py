# %% [markdown]
# # Validation reconstruction-mode mismatch diagnostic
#
# Validation logs `trajectory val_loss` with `reconstruction_mode='most_recent'`
# (loss only on feature-0 of the obs, which is the live-frame forecast — the
# only obs the model doesn't already have in its input window). Training,
# however, optimises `reconstruction_mode='uniform'` (loss across *all*
# `n_delays × prediction_steps` features). The qxpj8xpn pathology — frozen
# val loss but moving training metrics at n_delays=30 — surfaces this gap.
#
# Before changing validation to match training, we need to know:
# **does ranking by `most_recent` agree with ranking by `uniform`?** If yes,
# switching the val metric is safe. If no, we have a real conflict between
# what the model is being optimised for and what we care about, and the fix
# is more involved than just changing the validation kwarg.
#
# This script: for every run in the past `*_uniform_*__lc_sweep` groups,
# load the best checkpoint, recompute trajectory val loss two ways at
# alpha_teacher_forcing=0, and scatter the two against each other.

# %%
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint
from JacobianODE.jacobians import create_dataloaders

PROJECT = "JacobianODE/Lorenz_INDpartial_N25_D1_NormTrue_T3__JacobianODE"
GROUPS = [
    "lorenz_partial_25d_additive_mse_uniform_p30__lc_sweep",          # clean (no obs noise)
    "lorenz_partial_25d_additive_mse_uniform_p30_obsnoise001__lc_sweep",
    "lorenz_partial_25d_additive_mse_uniform_p30_obsnoise005__lc_sweep",
    "lorenz_partial_25d_additive_mse_uniform_p30_obsnoise001__nolpl_lc_sweep",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_VAL_BATCHES = 8  # cap per-run cost; ~256 examples at batch_size=32

# %%
import wandb
api = wandb.Api(timeout=60)

def collect_runs():
    rows = []
    for grp in GROUPS:
        runs = list(api.runs(PROJECT, filters={"group": grp}))
        print(f"{grp}: {len(runs)} runs")
        for r in runs:
            cfg = dict(r.config)
            lc = cfg.get("training", {}).get("lightning", {}).get("loop_closure_weight")
            rows.append({"group": grp, "run_id": r.id, "lc_weight": lc, "state": r.state})
    return rows

run_rows = collect_runs()
print(f"\ntotal: {len(run_rows)} runs\n")

# %%
@torch.no_grad()
def eval_traj_modes(lit_model, val_dl, n_batches):
    """Average free-running trajectory loss in both reconstruction modes."""
    losses_mr, losses_unif = [], []
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        batch = batch.to(DEVICE).float()
        ret_mr = lit_model.trajectory_model_step(
            batch, alpha_teacher_forcing=0,
            obs_noise_scale=0, latent_noise_scale=0,
            reconstruction_mode="most_recent",
        )
        ret_unif = lit_model.trajectory_model_step(
            batch, alpha_teacher_forcing=0,
            obs_noise_scale=0, latent_noise_scale=0,
            reconstruction_mode="uniform",
        )
        losses_mr.append(float(ret_mr["loss"].item()))
        losses_unif.append(float(ret_unif["loss"].item()))
    return float(np.mean(losses_mr)), float(np.mean(losses_unif))

# %%
results = []
for i, row in enumerate(run_rows):
    print(f"[{i+1}/{len(run_rows)}] {row['group'][:50]}... / {row['run_id']}")
    try:
        run, cfg, eq, dt, values, _, _, _, _, lit_model = load_run(
            PROJECT, run_id=row["run_id"], generate_data=True, verbose=False,
        )
        train_dl, val_dl, test_dl, trajs = create_dataloaders(
            cfg, values, verbose=False, return_full_obs=True,
        )
        load_checkpoint(run, cfg, lit_model, epoch=None, verbose=False)
        lit_model = lit_model.to(DEVICE).eval()
        l_mr, l_unif = eval_traj_modes(lit_model, val_dl, N_VAL_BATCHES)
        results.append({**row, "loss_mr": l_mr, "loss_unif": l_unif})
        print(f"    mr={l_mr:.4g}  unif={l_unif:.4g}")
    except Exception as e:
        print(f"    SKIPPED: {type(e).__name__}: {e}")
        results.append({**row, "loss_mr": None, "loss_unif": None,
                        "error": f"{type(e).__name__}: {e}"})
    # Free GPU memory before next run.
    try:
        del lit_model, run, cfg, train_dl, val_dl, test_dl, trajs, values
        torch.cuda.empty_cache()
    except Exception:
        pass

# %%
# Save raw to JSON for re-plotting later without re-running.
out_dir = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox")
json_path = out_dir / "val_mode_mismatch_results.json"
json_path.write_text(json.dumps(results, indent=2))
print(f"\nWrote {json_path}")

# %%
# Plot.
ok = [r for r in results if r.get("loss_mr") is not None]
fig, ax = plt.subplots(figsize=(9, 8))
group_colors = {g: f"C{i}" for i, g in enumerate(GROUPS)}

for grp in GROUPS:
    grp_rows = [r for r in ok if r["group"] == grp]
    if not grp_rows:
        continue
    xs = [r["loss_unif"] for r in grp_rows]
    ys = [r["loss_mr"] for r in grp_rows]
    label = grp.replace("lorenz_partial_25d_additive_mse_uniform_p30", "p30") \
               .replace("__lc_sweep", "")
    ax.scatter(xs, ys, color=group_colors[grp], s=60, alpha=0.8, label=label)
    # Mark the best-by-mr run within this group.
    best_mr_idx = int(np.argmin(ys))
    best_unif_idx = int(np.argmin(xs))
    ax.annotate(
        f"best-mr\nlc={grp_rows[best_mr_idx]['lc_weight']:.0e}",
        (xs[best_mr_idx], ys[best_mr_idx]),
        textcoords="offset points", xytext=(8, -8), fontsize=8,
        color=group_colors[grp],
    )
    if best_unif_idx != best_mr_idx:
        # Different "best" — flag it explicitly.
        ax.annotate(
            f"best-unif\nlc={grp_rows[best_unif_idx]['lc_weight']:.0e}",
            (xs[best_unif_idx], ys[best_unif_idx]),
            textcoords="offset points", xytext=(8, 8), fontsize=8,
            color=group_colors[grp], style="italic",
        )

# Identity line (would mean perfect agreement of the two losses' magnitudes).
all_x = np.array([r["loss_unif"] for r in ok])
all_y = np.array([r["loss_mr"] for r in ok])
xy_min = max(min(all_x.min(), all_y.min()), 1e-5)
xy_max = max(all_x.max(), all_y.max()) * 1.1
ax.plot([xy_min, xy_max], [xy_min, xy_max], "k--", lw=0.5, alpha=0.5,
        label="identity (loss_mr == loss_unif)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("trajectory val loss — `uniform` mode (all features)")
ax.set_ylabel("trajectory val loss — `most_recent` mode (feature 0 only)")
ax.set_title(
    "Val-mode mismatch across past uniform sweeps\n"
    "α_TF=0 free-running rollout, both losses on the same batch sequence"
)
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3, which="both")

# Per-group rank correlation in the title insert.
from scipy.stats import spearmanr
corr_text = []
for grp in GROUPS:
    grp_rows = [r for r in ok if r["group"] == grp]
    if len(grp_rows) < 3:
        continue
    xs = [r["loss_unif"] for r in grp_rows]
    ys = [r["loss_mr"] for r in grp_rows]
    rho, _ = spearmanr(xs, ys)
    short = grp.replace("lorenz_partial_25d_additive_mse_uniform_p30", "p30") \
                .replace("__lc_sweep", "")
    corr_text.append(f"{short}: ρ={rho:+.2f}")
ax.text(
    0.98, 0.02,
    "Spearman ρ (uniform vs most_recent ranking):\n" + "\n".join(corr_text),
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=8, family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

fig.tight_layout()
out_path = out_dir / "val_mode_mismatch_diagnostic.png"
fig.savefig(out_path, dpi=130)
print(f"Saved {out_path}")
plt.show()
