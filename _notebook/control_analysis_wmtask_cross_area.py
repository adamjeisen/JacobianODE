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
# # WMTask — Cross-Area Control Analysis via Ground-Truth Gramians
#
# Load the same trained WMTask `init_mode_random` biological RNN used across
# the WMTask experiments, compute the **ground-truth** Jacobians along each
# trajectory (via `eq.jac`), then run the batched square-root QR Gramian
# code on windows to quantify cross-area control strength between the
# **visual** area (indices 0–63) and the **cognitive** area (indices 64–127).
#
# For each ordered pair `(source → target)` of areas, define the LTV
# control system:
#
# $$ \dot x_\text{target} = A\, x_\text{target} + B\, x_\text{source},
#    \qquad y = C\, x_\text{target} $$
#
# where:
# - `A` = target-intrinsic Jacobian block (`target × target`)
# - `B` = source-to-target block (how source drives target)
# - `C` = target-to-source block (how target "projects" back into source)
#
# Gramians are computed with `rescale=True` so the magnitude lives in a
# scalar log register (safe for long horizons).

# %%
import datetime as _dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from JacobianODE.control import compute_all_gramians
from wmtask.trajectories import load_wmtask_for_jacobianode

# Force CPU: 128-D Gramians on a short window are fast on CPU, and this
# machine's GPU is often saturated by Jupyter kernels. Change as needed.
device = torch.device("cpu")
print(f"Device: {device}")

# %% [markdown]
# ## Load the WMTask `init_mode_random` model + trajectories
#
# Matches the WMTask experiment YAMLs exactly (`conf/data/wmtask.yaml`).

# %%
eq, sol, dt = load_wmtask_for_jacobianode(
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
values = sol["values"]
if not torch.is_tensor(values):
    values = torch.as_tensor(values)
values = values.to(device).float()
print(f"trajectories: {values.shape} (n_trajs, T, D)   dt = {dt}")

# %% [markdown]
# ## Area indices and configuration

# %%
VISUAL = slice(0, 64)
COGNITIVE = slice(64, 128)

N_BATCH = 128        # number of trajectories to analyse
WINDOW_SIZE = None   # None -> use the full trajectory length as one window
STRIDE = None        # ignored when WINDOW_SIZE is full length

n_available = values.shape[0]
if N_BATCH > n_available:
    print(f"Requested N_BATCH={N_BATCH} > available {n_available}, capping.")
    N_BATCH = n_available

traj_batch = values[:N_BATCH]
T_total = traj_batch.shape[1]
if WINDOW_SIZE is None:
    WINDOW_SIZE = T_total
    STRIDE = T_total  # only one window per trajectory
n_windows = (T_total - WINDOW_SIZE) // STRIDE + 1
print(
    f"Using {N_BATCH} trajs × {n_windows} window(s) "
    f"(size {WINDOW_SIZE}, stride {STRIDE}) over T={T_total}. "
    f"Total batched windows: {N_BATCH * n_windows}"
)

# %% [markdown]
# ## Compute ground-truth Jacobians along each trajectory

# %%
with torch.no_grad():
    J_full = eq.jac(traj_batch, t=0)   # (N_BATCH, T_total, D, D)
print(f"J_full shape: {tuple(J_full.shape)}")

# %% [markdown]
# ## Build rolling windows

# %%
def _roll_windows(x: torch.Tensor, window: int, stride: int) -> torch.Tensor:
    """(B, T, *rest) -> (B * n_windows, window, *rest), traj-major order."""
    B, T = x.shape[:2]
    rest = x.shape[2:]
    starts = torch.arange(0, T - window + 1, stride, device=x.device)
    # gather windows: (B, n_windows, window, *rest)
    out = torch.stack([x[:, s:s + window] for s in starts.tolist()], dim=1)
    return out.reshape(B * len(starts), window, *rest)


J_windows = _roll_windows(J_full, WINDOW_SIZE, STRIDE)
print(f"J_windows shape: {tuple(J_windows.shape)}")
total_batch = J_windows.shape[0]

# %% [markdown]
# ## Block extraction per area ordering
#
# Convention: for `(source → target)` control analysis, state = target,
# input = source, output = source.

# %%
def extract_blocks(J: torch.Tensor, target: slice, source: slice):
    """Return (A, B, C) for the LTV system with state=target, input=source,
    output=source. J shape (..., D, D)."""
    A = J[..., target, target]     # target × target
    B = J[..., target, source]     # target × source (source drives target)
    C = J[..., source, target]     # source × target (target projects into source)
    return A, B, C


# %% [markdown]
# ## Run Gramian analysis for both orderings

# %%
def run_area_gramians(J_win: torch.Tensor, target: slice, source: slice, dt: float):
    """Compute per-window trace/min-eig time series for one area ordering."""
    A, B, C = extract_blocks(J_win, target, source)
    print(
        f"  A:{tuple(A.shape)}, B:{tuple(B.shape)}, C:{tuple(C.shape)}"
    )
    (_, _, _), (log_spec_r, log_spec_c, log_spec_o) = compute_all_gramians(
        A, B, C, dt=dt,
        return_sequences=True,
        return_spectrums=True,
        rescale=True,
    )
    return log_spec_r, log_spec_c, log_spec_o


print("VISUAL → COGNITIVE")
vc_spec_r, vc_spec_c, vc_spec_o = run_area_gramians(
    J_windows, target=COGNITIVE, source=VISUAL, dt=dt,
)
print("COGNITIVE → VISUAL")
cv_spec_r, cv_spec_c, cv_spec_o = run_area_gramians(
    J_windows, target=VISUAL, source=COGNITIVE, dt=dt,
)

# %% [markdown]
# ## Extract log-trace and log-min-eig time series, average across batched windows
#
# For each Gramian, the spectrum is stored in log form (descending). So:
# - **log(trace(W))** = `logsumexp(log_spec, dim=-1)`
# - **log(min eig(W))** = `log_spec[..., -1]` (last entry = smallest).
#
# Averaging across batched windows uses the geometric mean (mean of logs)
# because the raw traces span many orders of magnitude.

# %%
def _log_trace(log_spec):
    return torch.logsumexp(log_spec, dim=-1)          # (B, T)


def _log_min_eig(log_spec):
    return log_spec[..., -1]                          # (B, T)


def _summary(log_spec):
    lt = _log_trace(log_spec).cpu()                   # (B, T)
    lm = _log_min_eig(log_spec).cpu()                 # (B, T)
    return {
        "log_trace_mean": lt.mean(dim=0).numpy(),
        "log_trace_std":  lt.std(dim=0).numpy(),
        "log_min_mean":   lm.mean(dim=0).numpy(),
        "log_min_std":    lm.std(dim=0).numpy(),
    }


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

# %% [markdown]
# ## Plot log-trace and log-min-eig for each ordering × Gramian

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
t_axis = np.arange(WINDOW_SIZE) * dt
for row, pair in enumerate(summaries):
    for col, gk in enumerate(["reach", "ctrl", "obs"]):
        ax = axes[row][col]
        s = summaries[pair][gk]
        ax.plot(t_axis, s["log_trace_mean"], "C0-", lw=2, label="log trace")
        ax.fill_between(
            t_axis,
            s["log_trace_mean"] - s["log_trace_std"],
            s["log_trace_mean"] + s["log_trace_std"],
            color="C0", alpha=0.2,
        )
        ax.plot(t_axis, s["log_min_mean"], "C3-", lw=2, label="log min-eig")
        ax.fill_between(
            t_axis,
            s["log_min_mean"] - s["log_min_std"],
            s["log_min_mean"] + s["log_min_std"],
            color="C3", alpha=0.2,
        )
        ax.set_title(f"{pair}  |  {gk}")
        ax.set_xlabel("window time (s)")
        ax.set_ylabel("log λ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
fig.tight_layout()

REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

fig_path = FIG_DIR / "wmtask_cross_area_gramians.png"
fig.savefig(fig_path, dpi=130, bbox_inches="tight")
print(f"Saved figure → {fig_path}")

# %% [markdown]
# ## Terminal-step summary (end of each window, averaged across batches)

# %%
def _final(arr):
    return float(arr[-1])


table_rows = []
for pair, by_g in summaries.items():
    for gk, s in by_g.items():
        table_rows.append({
            "pair": pair,
            "gramian": gk,
            "log_trace_final": _final(s["log_trace_mean"]),
            "log_trace_std_final": _final(s["log_trace_std"]),
            "log_min_final": _final(s["log_min_mean"]),
            "log_min_std_final": _final(s["log_min_std"]),
        })

import pandas as pd
table = pd.DataFrame(table_rows).set_index(["pair", "gramian"])
print(table.round(3))

# %% [markdown]
# ## Write markdown report

# %%
today = _dt.date.today().isoformat()
report_path = REPORT_DIR / f"{today} - WMTask Cross-Area Gramians.md"


def _fmt(x, p=2):
    return f"{x:+.{p}f}"


lines = []
lines.append(f"# WMTask Cross-Area Gramian Analysis — {today}")
lines.append("")
lines.append(
    "Ground-truth Jacobians (via `eq.jac`) of the trained `init_mode_random` "
    "biological RNN used across the WMTask experiments, analysed as an LTV "
    "control system. For each ordered pair `(source → target)` of areas "
    "(visual = idx 0–63, cognitive = idx 64–127), the state is the target, "
    "the input and the output are the source. Reachability / controllability "
    "/ observability Gramians are computed on rolling windows with the "
    "rescaled square-root QR algorithm in `JacobianODE.control`."
)
lines.append("")
lines.append("## Configuration")
lines.append("")
lines.append(f"- Trajectories analysed: **{N_BATCH}**")
lines.append(f"- Trajectory length T: **{T_total}**")
lines.append(f"- Window size / stride: **{WINDOW_SIZE} / {STRIDE}**")
lines.append(f"- Windows per trajectory: **{n_windows}** (total batch = {total_batch})")
lines.append(f"- dt: **{dt}**")
lines.append(f"- Rescaled square-root QR: **True**")
lines.append("")
lines.append("## Time series (log trace / log min eig, mean ± std over windows)")
lines.append("")
lines.append(f"![cross-area-gramians](figures/wmtask_cross_area_gramians.png)")
lines.append("")
lines.append("## Terminal-step summary")
lines.append("")
_md_header = "| pair | gramian | log_trace_final ± std | log_min_final ± std |"
_md_sep = "|------|---------|-----------------------|---------------------|"
lines.append(_md_header)
lines.append(_md_sep)
for (pair, gk), row in table.round(3).iterrows():
    lines.append(
        f"| {pair} | {gk} | "
        f"{row['log_trace_final']:+.3f} ± {row['log_trace_std_final']:.3f} | "
        f"{row['log_min_final']:+.3f} ± {row['log_min_std_final']:.3f} |"
    )
lines.append("")
lines.append("## Reading the figure")
lines.append("")
lines.append(
    "- **Blue (log trace)** grows roughly linearly when the target block has "
    "net-positive Lyapunov content driven by the source; slope proportional "
    "to the dominant local growth rate."
)
lines.append(
    "- **Red (log min-eig)** is the smallest eigenvalue of the Gramian. "
    "Large spread between trace and min-eig indicates the system is "
    "*directionally* controllable / observable — only a few modes in the "
    "target block are reached / seen through the source coupling."
)
lines.append(
    "- Compare **reach** vs **ctrl** panels: reach grows forward (what can "
    "the source reach in the target starting from zero), ctrl is the "
    "dual backward integral (what source history is needed to steer the "
    "target to origin). For stable forward blocks these should be similar; "
    "for unstable or chaotic target dynamics they can differ sharply."
)
lines.append(
    "- **Obs** panel uses `C = J_{source,target}` — how target state projects "
    "back into the source area. A large obs Gramian means the target's "
    "dynamics are visible through the source's observations over the window."
)
lines.append("")
lines.append("## Notes / caveats")
lines.append("")
lines.append(
    "- Ground-truth Jacobians come directly from the trained RNN's autograd "
    "`eq.jac`, so these numbers describe the **actual** local dynamics of "
    "the reference model — no learned-Jacobian-MLP approximation involved."
)
lines.append(
    "- A small batch (N_BATCH runs) is used for speed. Trends should be "
    "stable under increasing N_BATCH; std bands indicate window-to-window "
    "variability."
)
lines.append(
    "- Rescaled square-root form: log-spectra are exact log-eigenvalues of "
    "the true Gramian (not approximate), so cross-time comparisons are valid."
)

report_path.write_text("\n".join(lines) + "\n")
print(f"Wrote report → {report_path}")

# %% [markdown]
# ---
#
# # Predicted Gramians from a trained vanilla JacobianODE
#
# Loads a specific trained vanilla JacobianODE run (no latent encoder —
# predicts J directly from observation space), evaluates its predicted
# Jacobians on the **same** trajectories used above, and overlays the
# resulting Gramian spectra on the empirical curves via `twinx` so we can
# judge *relative* shape and slope without worrying about different
# absolute scales.
#
# Run identifier:
# - project: `WMTask_identity_encoder_verification`
# - group:   `vanilla__lc1e-4_obs0.0`
# - run_id:  `632q2xus`

# %%
from JacobianODE.jacobians.checkpoints.loader import load_run, load_checkpoint

PRED_PROJECT = "WMTask_identity_encoder_verification"
PRED_RUN_ID = "632q2xus"
PRED_SAVE_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs"

(pred_run_obj, pred_cfg, _pred_eq, _pred_dt, _pred_values, _, _, _,
 _pred_trajs, pred_lit_model) = load_run(
    f"JacobianODE/{PRED_PROJECT}",
    run_id=PRED_RUN_ID,
    save_dir=PRED_SAVE_DIR,
    generate_data=False,  # reuse the values/trajs already loaded above
    verbose=False,
)
load_checkpoint(
    pred_run_obj, pred_cfg, pred_lit_model,
    save_dir=PRED_SAVE_DIR, verbose=False,
)
pred_lit_model = pred_lit_model.to(device).eval()
print(f"Loaded {type(pred_lit_model).__name__} from run {PRED_RUN_ID}")

# %%
with torch.no_grad():
    pred_J_full = pred_lit_model.compute_jacobians(traj_batch)  # (N, T, 128, 128)
print(f"pred_J_full shape: {tuple(pred_J_full.shape)}")
assert pred_J_full.shape == J_full.shape, (
    f"predicted J shape {tuple(pred_J_full.shape)} != empirical {tuple(J_full.shape)}"
)

# %%
pred_J_windows = _roll_windows(pred_J_full, WINDOW_SIZE, STRIDE)

print("PRED  VISUAL → COGNITIVE")
p_vc_spec_r, p_vc_spec_c, p_vc_spec_o = run_area_gramians(
    pred_J_windows, target=COGNITIVE, source=VISUAL, dt=dt,
)
print("PRED  COGNITIVE → VISUAL")
p_cv_spec_r, p_cv_spec_c, p_cv_spec_o = run_area_gramians(
    pred_J_windows, target=VISUAL, source=COGNITIVE, dt=dt,
)

pred_summaries = {
    "visual→cognitive": {
        "reach": _summary(p_vc_spec_r),
        "ctrl":  _summary(p_vc_spec_c),
        "obs":   _summary(p_vc_spec_o),
    },
    "cognitive→visual": {
        "reach": _summary(p_cv_spec_r),
        "ctrl":  _summary(p_cv_spec_c),
        "obs":   _summary(p_cv_spec_o),
    },
}

# %% [markdown]
# ## Overlay plot: predicted vs empirical (twinx)
#
# Left axis (solid, blue/red): empirical log trace / log min-eig.
# Right axis (dashed, cyan/magenta): predicted.

# %%
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for row, pair in enumerate(summaries):
    for col, gk in enumerate(["reach", "ctrl", "obs"]):
        ax = axes2[row][col]
        e = summaries[pair][gk]
        p = pred_summaries[pair][gk]

        # Left axis: empirical
        ax.plot(t_axis, e["log_trace_mean"], "C0-", lw=2, label="empirical trace")
        ax.fill_between(t_axis, e["log_trace_mean"] - e["log_trace_std"],
                        e["log_trace_mean"] + e["log_trace_std"],
                        color="C0", alpha=0.18)
        ax.plot(t_axis, e["log_min_mean"], "C3-", lw=2, label="empirical min-eig")
        ax.fill_between(t_axis, e["log_min_mean"] - e["log_min_std"],
                        e["log_min_mean"] + e["log_min_std"],
                        color="C3", alpha=0.18)
        ax.set_ylabel("log λ  (empirical)", color="k")
        ax.grid(True, alpha=0.3)

        # Right axis (twin): predicted
        axp = ax.twinx()
        axp.plot(t_axis, p["log_trace_mean"], "c--", lw=2, label="pred trace")
        axp.fill_between(t_axis, p["log_trace_mean"] - p["log_trace_std"],
                         p["log_trace_mean"] + p["log_trace_std"],
                         color="c", alpha=0.18)
        axp.plot(t_axis, p["log_min_mean"], "m--", lw=2, label="pred min-eig")
        axp.fill_between(t_axis, p["log_min_mean"] - p["log_min_std"],
                         p["log_min_mean"] + p["log_min_std"],
                         color="m", alpha=0.18)
        axp.set_ylabel("log λ  (predicted)", color="gray")
        axp.tick_params(axis="y", labelcolor="gray")

        # Combined legend (first subplot only, to keep the others clean)
        if row == 0 and col == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = axp.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

        ax.set_title(f"{pair}  |  {gk}")
        ax.set_xlabel("window time (s)")

fig2.suptitle(
    f"Empirical (solid) vs trained vanilla JacobianODE run {PRED_RUN_ID} (dashed) — twin axes",
    y=1.02,
)
fig2.tight_layout()

fig2_path = FIG_DIR / "wmtask_cross_area_gramians_pred_vs_empirical.png"
fig2.savefig(fig2_path, dpi=130, bbox_inches="tight")
print(f"Saved overlay figure → {fig2_path}")

# %%
pred_table_rows = []
for pair, by_g in pred_summaries.items():
    for gk, s in by_g.items():
        pred_table_rows.append({
            "pair": pair,
            "gramian": gk,
            "log_trace_final": float(s["log_trace_mean"][-1]),
            "log_trace_std_final": float(s["log_trace_std"][-1]),
            "log_min_final": float(s["log_min_mean"][-1]),
            "log_min_std_final": float(s["log_min_std"][-1]),
        })
pred_table = pd.DataFrame(pred_table_rows).set_index(["pair", "gramian"])
print(pred_table.round(3))

# %% [markdown]
# ## Append predicted-vs-empirical section to report

# %%
pred_section = []
pred_section.append("")
pred_section.append("---")
pred_section.append("")
pred_section.append(
    f"## Predicted vs empirical Gramians — vanilla JacobianODE run `{PRED_RUN_ID}`"
)
pred_section.append("")
pred_section.append(
    f"Same trajectories, same area block decomposition, but the Jacobians "
    f"come from a trained **vanilla JacobianODE** (no latent encoder — the "
    f"MLP predicts `J(x)` directly from the 128-D state). Loaded from "
    f"W&B project `{PRED_PROJECT}`, group `vanilla__lc1e-4_obs0.0`, "
    f"run `{PRED_RUN_ID}`. The overlay uses `twinx` so empirical and "
    f"predicted log-λ curves can share an x-axis without their absolute "
    f"scales conflating."
)
pred_section.append("")
pred_section.append(
    f"![pred-vs-empirical]"
    f"(figures/wmtask_cross_area_gramians_pred_vs_empirical.png)"
)
pred_section.append("")
pred_section.append("### Predicted-Jacobian terminal-step summary")
pred_section.append("")
pred_section.append(_md_header)
pred_section.append(_md_sep)
for (pair, gk), row in pred_table.round(3).iterrows():
    pred_section.append(
        f"| {pair} | {gk} | "
        f"{row['log_trace_final']:+.3f} ± {row['log_trace_std_final']:.3f} | "
        f"{row['log_min_final']:+.3f} ± {row['log_min_std_final']:.3f} |"
    )
pred_section.append("")
pred_section.append("### Reading the overlay")
pred_section.append("")
pred_section.append(
    "- **Relative shape** is what matters, not absolute scale — the "
    "predicted and empirical Jacobians have different per-entry "
    "magnitudes, so log-λ offsets don't translate to 'predicted is "
    "wrong by X.' What's diagnostic is whether the slopes, curvature, "
    "and trace-vs-min-eig spread track each other."
)
pred_section.append(
    "- If predicted curves rise in parallel with empirical, the model "
    "has captured the dominant growth rate. If predicted lines are "
    "flat while empirical grows, the model is under-representing "
    "unstable directions. If predicted grows much faster, the model "
    "has over-amplified directions that empirically are near-neutral."
)
pred_section.append(
    "- Compare the two pairs (`visual→cognitive` vs `cognitive→visual`): "
    "the asymmetry of cross-area coupling (visual drives cognitive more "
    "than vice-versa, in empirical) should show up in the predicted "
    "trace slopes too if the model has learned the cross-block structure."
)

with report_path.open("a") as _f:
    _f.write("\n".join(pred_section) + "\n")
print(f"Appended predicted-vs-empirical section to {report_path}")
