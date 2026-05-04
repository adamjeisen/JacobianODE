"""Post-hoc analytics for a trained LitLatentJacobianODE model.

Provides a ``run_analytics`` entry point that loads a model from W&B, runs a
battery of diagnostics, and saves or displays the resulting figures.

Example usage::

    from JacobianODE.jacobians.run_analytics import run_analytics

    # Analyse a specific run:
    run_analytics(
        wandb_entity="JacobianODE",
        wandb_project="Lorenz_IND0_N25_D1_NormTrue_T3__spline_coupling__JacobianODE",
        save_dir="/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs",
        run_id="abc123",
        true_lyapunov=[0.91, 0.0, -14.57],
        output=["show"],
    )

    # Auto-select best run from a sweep group:
    run_analytics(
        wandb_entity="JacobianODE",
        wandb_project="Lorenz_IND0_N25_D1_NormTrue_T3__spline_coupling__JacobianODE",
        save_dir="/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs",
        wandb_group="my_sweep_group",
        true_lyapunov=[0.91, 0.0, -14.57],
        output=["show", "html"],
        output_dir="reports",
    )

Output options
--------------
``output`` accepts a string or list of strings:

- ``"show"``  – call ``plt.show()`` for each figure (suitable for notebooks and
  interactive terminals).
- ``"save"``  – write each figure as a PNG to ``output_dir``.
- ``"html"``  – write a single self-contained HTML file (figures embedded as
  base64 PNGs, metrics as readable text) to ``output_dir/<stem>.html``.
- ``"pdf"``   – collect all figures into a single multi-page PDF at
  ``output_dir/<stem>.pdf``.
- ``"return"``– skip display/saving and return a ``dict[str, Figure]`` instead.

Sections
--------
``sections`` is ``None`` (run all) or a list drawn from:

- ``"sweep_overview"``           – 4-panel sweep criteria chart (needs *wandb_group*)
- ``"reconstruction"``           – true vs decoded trajectory
- ``"mase"``                     – teacher-forced vs free-running MASE
- ``"latent_utilization"``       – per-dim variance & entropy-based utilisation
- ``"lyapunov"``                 – Lyapunov spectrum (predicted, empirical, literature)
- ``"kaplan_yorke"``             – Kaplan-Yorke dimension comparison
- ``"prediction_windows"``       – per-window nMSE distribution
- ``"prediction_detail"``        – latent + obs plots for the median-loss window
- ``"long_trajectory"``          – full-length free-running prediction
- ``"encoder_decoder_jacobians"``– encoder/decoder Jacobian column-norm plots
- ``"amplification"``            – noise amplification loss (true state vs latent)
"""

from __future__ import annotations

import base64
import math
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

from .metrics import r2_score, normalized_mse as nmse_fn, mase as mase_fn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ALL_SECTIONS = [
    "sweep_overview",
    "reconstruction",
    "mase",
    "latent_utilization",
    "lyapunov",
    "kaplan_yorke",
    "prediction_windows",
    "prediction_detail",
    "long_trajectory",
    "encoder_decoder_jacobians",
    "amplification",
    "tangent_spectrum",
]


def _kaplan_yorke_dim(lyap_exps: torch.Tensor) -> torch.Tensor:
    """Kaplan-Yorke dimension D_KY from Lyapunov exponents.

    Accepts shape ``(N,)`` (single trajectory) or ``(B, N)`` (batch).
    Returns scalar or ``(B,)`` tensor respectively.
    """
    lyap_exps = torch.as_tensor(lyap_exps)
    if lyap_exps.ndim == 1:
        lyap_exps = lyap_exps.unsqueeze(0)
    B, N = lyap_exps.shape
    lyap_sorted, _ = torch.sort(lyap_exps, descending=True, dim=1)
    cumsum = torch.cumsum(lyap_sorted, dim=1)
    mask_pos = cumsum > 0
    k_idx = mask_pos.sum(dim=1) - 1
    D_KY = torch.zeros(B, dtype=lyap_exps.dtype, device=lyap_exps.device)
    for b in range(B):
        k = k_idx[b].item()
        if k < 0:
            D_KY[b] = 0.0
            continue
        sum_k = cumsum[b, k]
        if (k + 1) < N and lyap_sorted[b, k + 1] != 0:
            D_KY[b] = (k + 1) + sum_k / torch.abs(lyap_sorted[b, k + 1])
        else:
            D_KY[b] = float(N)
    return D_KY if D_KY.shape[0] > 1 else D_KY[0]


def _participation_ratio(X: np.ndarray) -> float:
    X_flat = X.reshape(-1, X.shape[-1])
    cov = np.cov(X_flat, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    return float((eigvals.sum()) ** 2 / ((eigvals ** 2).sum() + 1e-12))


def _get_coupling_info(cfg) -> tuple[bool, int | None]:
    """Return (is_coupling, n_target_dims)."""
    n_target_dims = OmegaConf.select(cfg, "model.n_target_dims", default=None)
    return (n_target_dims is not None), n_target_dims


def _z_dyn(z: torch.Tensor, n_target_dims: int | None) -> torch.Tensor:
    if n_target_dims is not None:
        return z[..., :n_target_dims]
    return z


def _z_null(z: torch.Tensor, n_target_dims: int | None) -> torch.Tensor | None:
    """Return the null (non-dynamics) latent slice, or ``None`` if there is no null subspace."""
    if n_target_dims is None:
        return None
    if n_target_dims >= z.shape[-1]:
        return None
    return z[..., n_target_dims:]


def _save_or_show(fig: plt.Figure, name: str, output: list[str],
                  output_dir: Path | None, pdf: PdfPages | None) -> None:
    if "save" in output and output_dir is not None:
        fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=150)
    if "pdf" in output and pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    if "show" in output:
        plt.show()
    if "return" not in output:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot functions — each accepts precomputed data and returns a Figure
# ---------------------------------------------------------------------------

def plot_sweep_pareto(
    all_diagnostics: list,
    best_index: int | None,
    ranking_method: str,
) -> plt.Figure:
    """Log-log scatter of loop closure loss vs trajectory loss with Pareto front.

    Highlights the chosen run and labels the ranking method used.
    """
    traj_losses = np.array([d.trajectory_val_loss for d in all_diagnostics])
    lc_losses = np.array([
        d.loop_closure_loss if d.loop_closure_loss is not None else np.nan
        for d in all_diagnostics
    ])
    valid = np.isfinite(lc_losses) & np.isfinite(traj_losses)

    # --- Pareto front (lower is better on both) ---
    pareto_mask = np.zeros(len(traj_losses), dtype=bool)
    for i in range(len(traj_losses)):
        if not valid[i]:
            continue
        dominated = False
        for j in range(len(traj_losses)):
            if i == j or not valid[j]:
                continue
            if (traj_losses[j] <= traj_losses[i] and lc_losses[j] <= lc_losses[i]
                    and (traj_losses[j] < traj_losses[i] or lc_losses[j] < lc_losses[i])):
                dominated = True
                break
        if not dominated:
            pareto_mask[i] = True

    pareto_idx = np.where(pareto_mask)[0]
    pareto_order = pareto_idx[np.argsort(lc_losses[pareto_idx])]

    fig, ax = plt.subplots(figsize=(8, 6))

    # All runs
    ax.scatter(
        lc_losses[valid], traj_losses[valid],
        s=40, alpha=0.6, edgecolors="black", linewidths=0.5,
        c="tab:blue", label="All runs",
    )

    # Pareto front line
    if len(pareto_order) >= 2:
        ax.plot(
            lc_losses[pareto_order], traj_losses[pareto_order],
            "r-o", markersize=5, linewidth=1.5, label="Pareto front", zorder=5,
        )

    # Chosen run
    if best_index is not None and valid[best_index]:
        ax.scatter(
            [lc_losses[best_index]], [traj_losses[best_index]],
            marker="*", s=400, c="gold", edgecolors="black",
            linewidths=1.5, zorder=10,
            label=f"Selected ({ranking_method})",
        )

    ax.set_xlabel("Loop closure loss")
    ax.set_ylabel("Trajectory val loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Sweep Pareto Front — selection: {ranking_method}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_sweep_overview(
    all_diagnostics: list,
    result,
    sweep_lambdas: list[float],
    n_dyn: int,
) -> plt.Figure:
    """4-panel bar chart of sweep selection criteria.

    ``n_dyn`` is the dimensionality of the loop-closure space (the
    dynamic subspace, ``n_target_dims``, for latent models with a
    subspace split; equal to ``n_latent`` otherwise). The C2 threshold
    drawn on panel (0, 1) is ``sqrt(n_dyn)``.
    """
    one_step_mases = [m.one_step_mase for m in all_diagnostics]
    loop_closure_losses = [m.loop_closure_loss for m in all_diagnostics]
    eig_fracs = [m.fast_eigenvalue_fraction for m in all_diagnostics]
    traj_losses = [m.trajectory_val_loss for m in all_diagnostics]

    colors = [
        "tab:green" if i in result.surviving_indices else "tab:red"
        for i in range(len(all_diagnostics))
    ]
    x_labels = [str(v) for v in sweep_lambdas]
    x_pos = list(range(len(sweep_lambdas)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # C1: one-step MASE
    axes[0, 0].bar(x_pos, one_step_mases, color=colors)
    if result.best_index is not None:
        axes[0, 0].bar(result.best_index, one_step_mases[result.best_index],
                       color="gold", edgecolor="black", linewidth=2, label="Selected")
    axes[0, 0].axhline(y=1.0, color="k", linestyle="--", lw=1, label="persistence")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[0, 0].set_ylabel("MASE")
    axes[0, 0].set_title("C1: One-step MASE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)

    # C2: loop closure loss
    axes[0, 1].bar(x_pos, loop_closure_losses, color=colors)
    if result.best_index is not None:
        axes[0, 1].bar(result.best_index, loop_closure_losses[result.best_index],
                       color="gold", edgecolor="black", linewidth=2, label="Selected")
    axes[0, 1].axhline(y=np.sqrt(n_dyn), color="k", linestyle="--", lw=1,
                       label=f"sqrt(n_dyn)={np.sqrt(n_dyn):.2f}")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[0, 1].set_ylabel("Loop closure loss")
    axes[0, 1].set_title("C2: Loop Closure")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_yscale("log")

    # C3: fast eigenvalue fraction
    axes[1, 0].bar(x_pos, eig_fracs, color=colors)
    if result.best_index is not None:
        axes[1, 0].bar(result.best_index, eig_fracs[result.best_index],
                       color="gold", edgecolor="black", linewidth=2, label="Selected")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[1, 0].set_ylabel("Fraction fast eigenvalues")
    axes[1, 0].set_title("C3: Eigenvalue Fraction")
    axes[1, 0].legend(fontsize=8)

    # Traj val loss
    axes[1, 1].bar(x_pos, traj_losses, color=colors)
    if result.best_index is not None:
        axes[1, 1].bar(result.best_index, traj_losses[result.best_index],
                       color="gold", edgecolor="black", linewidth=2, label="Selected")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[1, 1].set_ylabel("Trajectory val loss")
    axes[1, 1].set_title("Trajectory Loss (selection target)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_yscale("log")

    for ax in axes.flat:
        ax.set_xlabel("loop_closure_weight")

    fig.suptitle(
        "Sweep Model Selection\n(green = passes all criteria; gold = selected best)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_reconstruction(
    plot_targets: np.ndarray,
    traj_decoded: np.ndarray,
    r2_val: float,
    recon_nmse: float,
    is_coupling: bool = False,
    inv_err: float | None = None,
    null_rms: float | None = None,
) -> plt.Figure:
    """True vs decoded trajectory with R² annotation."""
    T_plot = min(plot_targets.shape[0], traj_decoded.shape[0])
    n_dims = min(10, plot_targets.shape[-1], traj_decoded.shape[-1])

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(n_dims):
        ax.plot(plot_targets[:T_plot, i], c=f"C{i}", label=f"true dim {i}")
    for i in range(n_dims):
        ax.plot(traj_decoded[:T_plot, i], c=f"C{i}", linestyle="--", label=f"decoded dim {i}")

    title = f"True vs. Decoded (no prediction)\n$R^2 = {r2_val:.4f}$,  recon nMSE = {recon_nmse:.6f}"
    if is_coupling and inv_err is not None:
        title += f"\nInverse consistency MSE = {inv_err:.2e}"
        if null_rms is not None:
            title += f",  null RMS = {null_rms:.2e}"
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    return fig


def plot_mase(forced_mase: float, free_mase: float) -> plt.Figure:
    """Bar chart comparing teacher-forced and free-running MASE."""
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Teacher-forced\n(α=1)", "Free-running\n(α=0)"],
        [forced_mase, free_mase],
        color=["C0", "C1"],
        alpha=0.8,
    )
    ax.axhline(y=1.0, color="k", linestyle="--", lw=1, label="persistence (MASE=1)")
    for bar, val in zip(bars, [forced_mase, free_mase]):
        ax.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("MASE")
    ax.set_title("Teacher-forced vs Free-running MASE")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_latent_utilization(
    dim_var: np.ndarray,
    utilization: float,
    n_dyn: int,
    null_rms_per_t: np.ndarray | None = None,
) -> plt.Figure:
    """Per-dimension fractional variance and (optionally) null-subspace RMS."""
    n_plots = 2 if null_rms_per_t is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    ax.bar(range(len(dim_var)), dim_var / dim_var.sum(), color="steelblue")
    ax.set_xlabel("Latent dimension (dynamic)")
    ax.set_ylabel("Fractional variance")
    ax.set_title(
        f"Latent dimension utilisation (D_dyn={n_dyn})\nEntropy-based: {utilization:.3f}  (1.0 = uniform)"
    )

    if null_rms_per_t is not None:
        axes[1].plot(null_rms_per_t)
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Null subspace RMS")
        axes[1].set_title("Null Subspace Magnitude Over Time")

    plt.tight_layout()
    return fig


def _plot_lyapunov_bars(
    ax: plt.Axes,
    n_lyaps: int,
    pred_np: np.ndarray,
    pred_std_np: np.ndarray,
    emp_np: np.ndarray | None,
    emp_std_np: np.ndarray | None,
    true_lyapunov: list[float] | None,
    full_lyap_np: np.ndarray | None,
    full_lyap_std_np: np.ndarray | None,
    loop_closure_weight: float | None,
    title_suffix: str = "",
) -> None:
    """Draw grouped Lyapunov bar chart on *ax* for exponent indices 0..n_lyaps-1."""
    x_idx = np.arange(n_lyaps)

    n_groups = sum([
        1,  # batch+burn-in always present
        full_lyap_np is not None and len(full_lyap_np) > 0,
        emp_np is not None and len(emp_np) > 0,
        true_lyapunov is not None,
    ])
    bar_w = min(0.8 / n_groups, 0.25)
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * bar_w

    bar_idx = 0
    ax.bar(x_idx + offsets[bar_idx], pred_np[:n_lyaps], width=bar_w,
           yerr=pred_std_np[:n_lyaps], capsize=3, label="Predicted (batch + burn-in)", alpha=0.8)
    bar_idx += 1

    if full_lyap_np is not None and len(full_lyap_np) > 0:
        ax.bar(x_idx + offsets[bar_idx], full_lyap_np[:n_lyaps], width=bar_w,
               yerr=full_lyap_std_np[:n_lyaps] if full_lyap_std_np is not None else None,
               capsize=3, label="Predicted (full trajectory)", alpha=0.8)
        bar_idx += 1

    if emp_np is not None and len(emp_np) > 0:
        ax.bar(x_idx + offsets[bar_idx], emp_np[:n_lyaps], width=bar_w,
               yerr=emp_std_np[:n_lyaps] if emp_std_np is not None else None,
               capsize=3, label="Empirical (true Jacobian)", alpha=0.8)
        bar_idx += 1

    if true_lyapunov is not None:
        ax.bar(x_idx + offsets[bar_idx], true_lyapunov[:n_lyaps], width=bar_w,
               label="Literature", alpha=0.8)

    ax.axhline(y=0, color="k", linestyle="--", lw=0.5)
    ax.set_xticks(x_idx)
    ax.set_xlabel("Exponent index")
    ax.set_ylabel("Lyapunov exponent")
    title = "Lyapunov Spectrum"
    if title_suffix:
        title += f" ({title_suffix})"
    if loop_closure_weight is not None:
        title += f" (loop_closure_weight={loop_closure_weight})"
    ax.set_title(title)
    ax.legend()


def plot_lyapunov_spectrum(
    pred_np: np.ndarray,
    pred_std_np: np.ndarray,
    emp_np: np.ndarray | None = None,
    emp_std_np: np.ndarray | None = None,
    true_lyapunov: list[float] | None = None,
    loop_closure_weight: float | None = None,
    full_lyap_np: np.ndarray | None = None,
    full_lyap_std_np: np.ndarray | None = None,
) -> plt.Figure | list[plt.Figure]:
    """Bar chart of predicted vs empirical vs literature Lyapunov spectrum.

    ``pred_np`` / ``pred_std_np`` are the batch+burn-in estimates.
    ``full_lyap_np`` / ``full_lyap_std_np`` are the full-trajectory estimates.

    If there are more than 20 exponents, returns a list of two figures:
    one with all exponents and one zoomed into the first 10.
    """
    n_lyaps = len(pred_np)
    if full_lyap_np is not None and len(full_lyap_np) > 0:
        n_lyaps = min(n_lyaps, len(full_lyap_np))
    if emp_np is not None and len(emp_np) > 0:
        n_lyaps = min(n_lyaps, len(emp_np))
    if true_lyapunov is not None:
        n_lyaps = min(n_lyaps, len(true_lyapunov))

    # --- main figure (all exponents) ---
    fig_all, ax_all = plt.subplots(figsize=(14, 5))
    _plot_lyapunov_bars(ax_all, n_lyaps, pred_np, pred_std_np, emp_np, emp_std_np,
                        true_lyapunov, full_lyap_np, full_lyap_std_np,
                        loop_closure_weight, title_suffix="all exponents" if n_lyaps > 20 else "")
    plt.tight_layout()

    if n_lyaps <= 20:
        return fig_all

    # --- zoomed figure (first 10) ---
    n_zoom = 10
    fig_zoom, ax_zoom = plt.subplots(figsize=(14, 5))
    _plot_lyapunov_bars(ax_zoom, n_zoom, pred_np, pred_std_np, emp_np, emp_std_np,
                        true_lyapunov, full_lyap_np, full_lyap_std_np,
                        loop_closure_weight, title_suffix="first 10")
    plt.tight_layout()

    return [fig_all, fig_zoom]


def plot_lyapunov_spectrum_by_condition(
    pred_per_group: dict[str, np.ndarray],   # label -> (n_in_group, n_lyaps)
    emp_per_group: dict[str, np.ndarray],    # label -> (n_in_group, n_lyaps)
    true_lyapunov: list[float] | None = None,
    loop_closure_weight: float | None = None,
) -> plt.Figure | list[plt.Figure]:
    """Per-condition Lyapunov spectrum overlay.

    Used when the model was trained with per-trajectory conditions
    (combined-source dataloader). One curve per (condition × {pred, emp})
    on a single axis: e.g. with two conditions you get four curves.

    Each curve uses mean ± std across the trajectories in that condition
    group, drawn as bars with error caps (matches the unconditioned
    plot's visual style).
    """
    # Determine n_lyaps as the min across all groups.
    n_lyaps = min(
        min((arr.shape[1] for arr in pred_per_group.values()), default=10**9),
        min((arr.shape[1] for arr in emp_per_group.values()), default=10**9),
    )
    if true_lyapunov is not None:
        n_lyaps = min(n_lyaps, len(true_lyapunov))

    def _draw(ax: plt.Axes, k: int) -> None:
        x_idx = np.arange(k)
        # One bar per (group × {pred, emp}) plus optional literature.
        n_bars_per_group = 2  # pred + emp
        groups_pred = list(pred_per_group.keys())
        groups_emp = list(emp_per_group.keys())
        # Use union of group labels in stable order: pred groups first.
        labels = list(dict.fromkeys(groups_pred + groups_emp))
        n_bars = len(labels) * n_bars_per_group + (1 if true_lyapunov is not None else 0)
        bar_w = min(0.8 / n_bars, 0.18)
        offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w

        bar_idx = 0
        for label in labels:
            if label in pred_per_group:
                arr = pred_per_group[label]
                m = arr.mean(axis=0)[:k]
                s = arr.std(axis=0)[:k]
                ax.bar(x_idx + offsets[bar_idx], m, width=bar_w, yerr=s,
                       capsize=3, label=f"Predicted [{label}]", alpha=0.8)
                bar_idx += 1
            if label in emp_per_group:
                arr = emp_per_group[label]
                m = arr.mean(axis=0)[:k]
                s = arr.std(axis=0)[:k]
                ax.bar(x_idx + offsets[bar_idx], m, width=bar_w, yerr=s,
                       capsize=3, label=f"Empirical [{label}]", alpha=0.8)
                bar_idx += 1
        if true_lyapunov is not None:
            ax.bar(x_idx + offsets[bar_idx], true_lyapunov[:k], width=bar_w,
                   label="Literature", alpha=0.8)

        ax.axhline(y=0, color="k", linestyle="--", lw=0.5)
        ax.set_xticks(x_idx)
        ax.set_xlabel("Exponent index")
        ax.set_ylabel("Lyapunov exponent")
        title = "Lyapunov Spectrum (by condition)"
        if loop_closure_weight is not None:
            title += f" (loop_closure_weight={loop_closure_weight})"
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    fig_all, ax_all = plt.subplots(figsize=(14, 5))
    _draw(ax_all, n_lyaps)
    plt.tight_layout()
    if n_lyaps <= 20:
        return fig_all
    fig_zoom, ax_zoom = plt.subplots(figsize=(14, 5))
    _draw(ax_zoom, 10)
    plt.tight_layout()
    return [fig_all, fig_zoom]


def plot_kaplan_yorke(
    ky_pred_np: np.ndarray,
    ky_emp_np: np.ndarray | None = None,
    ky_theory: float | None = None,
) -> plt.Figure:
    """1:1 scatter and per-trajectory bar chart of Kaplan-Yorke dimension."""
    has_emp = ky_emp_np is not None and len(ky_emp_np) > 0

    # Align arrays so the per-trajectory scatter/bar charts are consistent
    if has_emp and len(ky_pred_np) != len(ky_emp_np):
        n_common = min(len(ky_pred_np), len(ky_emp_np))
        ky_pred_np = ky_pred_np[:n_common]
        ky_emp_np = ky_emp_np[:n_common]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: 1:1 scatter
    ax = axes[0]
    if has_emp:
        lim_lo = min(ky_pred_np.min(), ky_emp_np.min()) - 0.05
        lim_hi = max(ky_pred_np.max(), ky_emp_np.max()) + 0.05
        ax.scatter(ky_pred_np, ky_emp_np, s=120,
                   c=np.arange(len(ky_pred_np)), cmap="viridis",
                   edgecolors="black", linewidths=1.5, zorder=3)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.5,
                label="y = x (perfect agreement)", alpha=0.7)
    else:
        lim_lo, lim_hi = ky_pred_np.min() - 0.05, ky_pred_np.max() + 0.05
        ax.scatter(np.arange(len(ky_pred_np)), ky_pred_np, s=80,
                   c=np.arange(len(ky_pred_np)), cmap="viridis",
                   edgecolors="black", linewidths=1.5, zorder=3)

    if ky_theory is not None:
        ax.axhline(ky_theory, color="#e74c3c", ls=":", lw=1.5,
                   label=f"Theoretical ≈ {ky_theory:.3f}")
        if has_emp:
            ax.axvline(ky_theory, color="#e74c3c", ls=":", lw=1.5)
    ax.set_xlabel("Predicted D_KY (model)", fontsize=11)
    ax.set_ylabel("Empirical D_KY (true Jacobian)" if has_emp else "D_KY", fontsize=11)
    ax.set_title("Kaplan–Yorke dimension: Model vs Ground Truth", fontsize=12)
    if has_emp:
        ax.set_aspect("equal")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: per-trajectory bar chart
    ax = axes[1]
    x = np.arange(len(ky_pred_np))
    width = 0.35
    ax.bar(x - width / 2 if has_emp else x, ky_pred_np, width,
           label="Predicted", color="#3498db", alpha=0.9)
    if has_emp:
        ax.bar(x + width / 2, ky_emp_np, width,
               label="Empirical", color="#2ecc71", alpha=0.9)
    if ky_theory is not None:
        ax.axhline(ky_theory, color="#e74c3c", ls="--", lw=1.5,
                   label=f"Theoretical ≈ {ky_theory:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Traj {i}" for i in range(len(ky_pred_np))])
    ax.set_ylabel("Kaplan–Yorke dimension")
    ax.set_title("Per-trajectory D_KY comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    mean_pred, std_pred = ky_pred_np.mean(), ky_pred_np.std()
    suptitle = f"Predicted: {mean_pred:.4f} ± {std_pred:.4f}"
    if has_emp:
        mean_emp, std_emp = ky_emp_np.mean(), ky_emp_np.std()
        suptitle += f"   |   Empirical: {mean_emp:.4f} ± {std_emp:.4f}"
    if ky_theory:
        suptitle += f"   |   Theoretical: {ky_theory:.3f}"
    fig.suptitle(suptitle, fontsize=10, y=1.02)
    plt.tight_layout()
    return fig


def plot_pca_kaplan_yorke(
    Z_flat: np.ndarray,
    X_true_flat: np.ndarray,
    mean_ky_latent: float,
    mean_ky_true: float | None = None,
    mean_ky_latent_burnin: float | None = None,
    label_true: str = "True state",
) -> plt.Figure:
    """Side-by-side PC1×PC2 scatter for latent space vs. true state."""
    pca_latent = PCA(n_components=2).fit(Z_flat)
    Z_pc = pca_latent.transform(Z_flat)

    n_cols = 1 if X_true_flat is None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    axes[0].scatter(Z_pc[:, 0], Z_pc[:, 1], s=1, alpha=0.4,
                    c=Z_pc[:, 0], cmap="viridis", rasterized=True)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    _latent_title = f"Latents\nD_KY (full) = {mean_ky_latent:.3f}"
    if mean_ky_latent_burnin is not None:
        _latent_title += f",  D_KY (burn-in) = {mean_ky_latent_burnin:.3f}"
    axes[0].set_title(_latent_title)

    if X_true_flat is not None:
        pca_true = PCA(n_components=2).fit(X_true_flat)
        X_pc = pca_true.transform(X_true_flat)
        axes[1].scatter(X_pc[:, 0], X_pc[:, 1], s=1, alpha=0.4,
                        c=X_pc[:, 0], cmap="viridis", rasterized=True)
        axes[1].set_xlabel("PC 1")
        axes[1].set_ylabel("PC 2")
        ky_true_str = f"{mean_ky_true:.3f}" if mean_ky_true is not None else "N/A"
        axes[1].set_title(f"{label_true}\nmean D_KY = {ky_true_str}")

    fig.suptitle("PC 1 × PC 2", fontsize=12)
    plt.tight_layout()
    return fig


def plot_prediction_windows(all_per_window_nmse: np.ndarray) -> plt.Figure:
    """Line plot of per-window nMSE across all test windows."""
    median_idx = int(np.argsort(all_per_window_nmse)[len(all_per_window_nmse) // 2])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(all_per_window_nmse, "k-", lw=0.8, alpha=0.7)
    ax.axhline(np.median(all_per_window_nmse), color="C0", ls="--", lw=1.5,
               label=f"median = {np.median(all_per_window_nmse):.4f}")
    ax.axhline(all_per_window_nmse.mean(), color="C1", ls="--", lw=1.5,
               label=f"mean = {all_per_window_nmse.mean():.4f}")
    ax.scatter([0], [all_per_window_nmse[0]], c="red", s=50, zorder=5, label="window 0")
    ax.scatter([median_idx], [all_per_window_nmse[median_idx]], c="C0", s=50,
               zorder=5, marker="D", label=f"median window ({median_idx})")
    ax.set_xlabel("Window index")
    ax.set_ylabel("nMSE")
    ax.set_title(f"Per-window nMSE ({len(all_per_window_nmse)} windows)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    return fig


def plot_prediction_detail(
    z_pred: np.ndarray,
    z_true: np.ndarray,
    obs_pred: np.ndarray,
    obs_true: np.ndarray,
    sigma: float,
    mu: float | np.ndarray,
    traj_init_steps: int,
    nmse_val: float,
    title_suffix: str = "",
    latent_nmse_val: float | None = None,
    n_metric_dims: int | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Latent and observation space plots for a single prediction window.

    Returns ``(fig_latent, fig_obs)``.
    """
    n_latent = z_pred.shape[-1]
    n_cols = min(5, n_latent)
    n_rows = (n_latent + n_cols - 1) // n_cols
    t_steps = np.arange(z_pred.shape[0])

    # Latent figure
    fig_z, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for i in range(n_latent):
        ax = axes[i // n_cols, i % n_cols]
        ax.plot(t_steps[:len(z_true)], z_true[:, i], "k-", lw=1.5, label="True (encoded)")
        ax.plot(t_steps[:len(z_pred)], z_pred[:, i], "r--", lw=1.5, label="Predicted")
        ax.axvline(x=traj_init_steps, color="gray", ls=":", lw=1)
        ax.set_title(f"$z_{{{i}}}$")
        if i == 0:
            ax.legend(fontsize=8)
    for i in range(n_latent, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)
    # Show latent-space nMSE if available, otherwise fall back to obs nMSE
    _z_nmse = latent_nmse_val if latent_nmse_val is not None else nmse_val
    fig_z.suptitle(
        f"Latent Space: {title_suffix} (nMSE={_z_nmse:.4f})", fontsize=14, y=1.02
    )
    plt.tight_layout()

    # Observation space figure — slice to metric dims to match model training
    _obs_pred = obs_pred[..., :n_metric_dims] if n_metric_dims else obs_pred
    _obs_true = obs_true[..., :n_metric_dims] if n_metric_dims else obs_true
    obs_pred_phys = _obs_pred * sigma + mu
    obs_true_phys = _obs_true * sigma + mu
    D_obs = _obs_pred.shape[-1]
    dim_labels = [chr(ord("x") + i) for i in range(26)]
    t_pred = np.arange(_obs_pred.shape[0])

    fig_obs, axes_p = plt.subplots(1, min(D_obs, 3), figsize=(5 * min(D_obs, 3), 4), squeeze=False)
    for d in range(min(D_obs, 3)):
        ax = axes_p[0, d]
        label = dim_labels[d] if d < len(dim_labels) else f"dim {d}"
        ax.plot(t_pred[:len(obs_true_phys)], obs_true_phys[:, d], "k-", lw=1.5, label="True")
        ax.plot(t_pred[:len(obs_pred_phys)], obs_pred_phys[:, d], "r--", lw=1.5, label="Predicted")
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
    window_mase = float(mase_fn(obs_true_phys, obs_pred_phys))
    fig_obs.suptitle(
        f"Observation Space: {title_suffix} (nMSE={nmse_val:.4f}, MASE={window_mase:.4f})",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    return fig_z, fig_obs


def plot_long_trajectory(
    traj_true: np.ndarray,
    decoded_pred: np.ndarray,
    latent_true: np.ndarray,
    latent_pred: np.ndarray,
    traj_init_steps: int = 15,
) -> plt.Figure:
    """Two-panel plot: obs dim 0 and latent dim 0 over a long free-running run.

    Legacy single-seed plot kept for vanilla (non-latent) models and
    backward-compat. Latent models use ``plot_seeded_rollouts`` below.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    T = min(traj_true.shape[0], decoded_pred.shape[0])
    axs[0].plot(traj_true[:T, 0], label="True Observation")
    axs[0].plot(decoded_pred[:T, 0], label="Predicted Observation")
    axs[0].axvline(traj_init_steps, color="k", linestyle="--", alpha=0.7,
                   label="Prediction Start")
    axs[0].set_title("Observation Space")
    axs[0].set_ylabel("Obs dim 0")
    axs[0].legend(fontsize=9)

    T_lat = min(latent_true.shape[0], latent_pred.shape[0])
    axs[1].plot(latent_true[:T_lat, 0], linestyle="--", label="Encoded Latent")
    axs[1].plot(latent_pred[:T_lat, 0], label="Predicted Latent")
    axs[1].axvline(traj_init_steps, color="k", linestyle="--", alpha=0.7,
                   label="Prediction Start")
    axs[1].set_title("Latent Space")
    axs[1].set_ylabel("Latent dim 0")
    axs[1].set_xlabel("Time")
    axs[1].legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_seeded_rollouts(
    traj_true: np.ndarray,
    rollouts: list[dict],
) -> plt.Figure:
    """Multi-seed rollout plot for latent models.

    ``rollouts`` is a list of dicts each with keys ``name``, ``init_n``,
    ``decoded`` (obs dim 0 time series), ``latent_true`` (encoded z_dyn
    dim 0) and ``latent_pred`` (rolled-out z_dyn dim 0). Each config
    gets a row with obs (left) and latent (right) panels, annotated
    with std(post-init) as a quick "collapsed or still moving?" cue.
    """
    n = len(rollouts)
    fig, axs = plt.subplots(n, 2, figsize=(14, 2.8 * n), squeeze=False)
    for i, r in enumerate(rollouts):
        init_n = r["init_n"]
        decoded = r["decoded"]
        l_true = r["latent_true"]
        l_pred = r["latent_pred"]
        post_init = decoded[init_n:]
        std_post = float(np.std(post_init)) if len(post_init) > 0 else float("nan")

        ax_o = axs[i, 0]
        T = min(traj_true.shape[0], decoded.shape[0])
        ax_o.plot(traj_true[:T, 0], label="true obs", lw=0.8, alpha=0.7, color="C0")
        ax_o.plot(decoded[:T, 0], label="predicted obs", color="C3", lw=1.1)
        ax_o.axvline(init_n, color="k", ls=":", lw=0.6,
                     label=f"init ends (t={init_n})")
        ax_o.set_title(
            f"{r['name']}: init={init_n}, rollout={T - init_n},  "
            f"std(post-init)={std_post:.3g}"
        )
        ax_o.set_ylabel("Obs dim 0")
        ax_o.legend(fontsize=8, loc="upper right")
        ax_o.grid(True, alpha=0.3)

        ax_l = axs[i, 1]
        T_lat = min(l_true.shape[0], l_pred.shape[0])
        ax_l.plot(l_true[:T_lat, 0], ls="--", label="encoded z_dyn[0]",
                  color="C0", alpha=0.7, lw=0.9)
        ax_l.plot(l_pred[:T_lat, 0], label="rolled z_dyn[0]",
                  color="C3", lw=1.1)
        ax_l.axvline(init_n, color="k", ls=":", lw=0.6)
        ax_l.set_title("Latent dim 0")
        ax_l.set_ylabel("z_dyn dim 0")
        ax_l.legend(fontsize=8, loc="upper right")
        ax_l.grid(True, alpha=0.3)
    axs[-1, 0].set_xlabel("time step")
    axs[-1, 1].set_xlabel("time step")
    fig.suptitle(
        "Free-running rollouts under three init regimes\n"
        "(same integration kwargs as training; only traj_init_steps varies)",
        y=1.00,
    )
    plt.tight_layout()
    return fig


def plot_encoder_decoder_jacobians(
    encoder_jacobian: torch.Tensor,
    decoder_jacobian: torch.Tensor,
) -> plt.Figure:
    """Column/row norm plots for encoder and decoder Jacobians."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    dec_norms = (decoder_jacobian.norm(dim=1).mean(dim=0).detach().cpu().numpy()
                 / np.sqrt(decoder_jacobian.shape[1]))
    enc_norms = (encoder_jacobian.norm(dim=-1).mean(dim=0).detach().cpu().numpy()
                 / np.sqrt(encoder_jacobian.shape[-1]))
    axs[0].plot(dec_norms)
    axs[0].set_title(
        "Decoder Jacobian Column-wise Norm Mean Over Samples\n"
        "(Normalized by sqrt(delay dimension))"
    )
    axs[0].set_ylabel("Mean Decoder Jacobian Norm")
    axs[1].plot(enc_norms)
    axs[1].set_title(
        "Encoder Jacobian Row-wise Norm Mean Over Samples\n"
        "(Normalized by sqrt(delay dimension))"
    )
    axs[1].set_ylabel("Mean Encoder Jacobian Norm")
    axs[1].set_xlabel("Latent/Output Dimension (index)")
    plt.tight_layout()
    return fig


def plot_tangent_spectrum(
    energy: np.ndarray,
    spectrum: np.ndarray,
    n_pairs: int,
    n_dyn: int,
    n_obs: int,
    expected_intrinsic_dim: int | None = 3,
) -> plt.Figure:
    """Two-panel ranked spectrum of latent tangents projected onto encoder Jacobian.

    Left:  raw per-direction energy (log-y).
    Right: cumulative fraction of energy (linear-y), with the
           ``expected_intrinsic_dim`` reference line. For Lorenz this is 3.
    Both panels x = ranked tangent direction (1-indexed for readability).
    """
    K = len(energy)
    x = np.arange(1, K + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: per-direction energy, log-y.
    axes[0].plot(x, energy, marker="o", lw=1.4, color="C0")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ranked tangent direction (1-indexed)")
    axes[0].set_ylabel("Mean squared projection (energy)")
    axes[0].set_title(f"Per-direction energy (K={K} = min(n_dyn={n_dyn}, n_obs={n_obs}))")
    axes[0].grid(True, which="both", alpha=0.3)

    # Panel 2: cumulative fraction of energy.
    cum = np.cumsum(spectrum)
    axes[1].plot(x, cum, marker="o", lw=1.4, color="C0",
                 label="cumulative fraction")
    if expected_intrinsic_dim is not None and expected_intrinsic_dim <= K:
        axes[1].axvline(
            expected_intrinsic_dim, color="r", ls="--", lw=1,
            label=f"expected intrinsic dim = {expected_intrinsic_dim}",
        )
        cum_at_expected = float(cum[expected_intrinsic_dim - 1])
        axes[1].axhline(
            cum_at_expected, color="r", ls=":", lw=0.8, alpha=0.6,
            label=f"cum @ dim {expected_intrinsic_dim} = {cum_at_expected:.4f}",
        )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Ranked tangent direction (1-indexed)")
    axes[1].set_ylabel("Cumulative fraction of energy")
    axes[1].set_title(f"Cumulative spectrum (n_pairs = {n_pairs})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_amplification(
    amp_loss_true: float,
    amp_loss_latent: float,
    n_obs_dims: int,
    n_latent_dims: int,
) -> plt.Figure:
    """Bar chart of noise amplification loss: true state vs latent space."""
    labels = [f"True state\n(D={n_obs_dims})", f"Latent\n(D={n_latent_dims})"]
    values = [amp_loss_true, amp_loss_latent]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["C0", "C1"], alpha=0.7)
    ax.set_ylabel("Amplification loss")
    ax.set_title("Noise Amplification Loss")
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom",
        )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_analytics(
    wandb_entity: str,
    wandb_project: str,
    save_dir: str,
    *,
    run_id: str | None = None,
    epoch: int | None = None,
    wandb_group: str | None = None,
    true_lyapunov: list[float] | None = None,
    output: str | Sequence[str] = "show",
    output_dir: str | Path | None = None,
    sections: list[str] | None = None,
    device: str | None = None,
    # Inference parameters
    n_sample: int = 128,
    n_mase_batches: int = 10,
    n_amp_trajs: int = 64,
    n_amp_neighbors: int = 10,
    n_amp_max_t: int = 10,
    n_pred_trajs: int = 3,
    lyapunov_burn_in_steps: int = 400,
    lyapunov_burn_in_drop: int = 100,
    # Precomputed sweep results (skip re-running selection if provided)
    sweep_diagnostics: list | None = None,
    sweep_result: Any | None = None,
    sweep_lambdas: list[float] | None = None,
    ranking_method: str = "pareto_knee", # "best_traj_loss" | "pareto_knee" | "geo_rank" | "minimax_rank" | "geo_log_score" | "minimax_log_score"
    eigenvalue_threshold: float = 0.001,
    return_model: bool = False,
    use_all_runs: bool = False,
) -> Any:
    """Run the full analytics suite on a trained LitLatentJacobianODE model.

    Parameters
    ----------
    wandb_entity : str
        W&B entity (team/user) name.
    wandb_project : str
        W&B project name (without entity prefix).
    save_dir : str
        Directory containing Lightning checkpoints (passed to ``load_run``).
    run_id : str, optional
        W&B run ID for the model to analyse.  If ``None``, the best run is
        auto-selected from the project (optionally filtered by *wandb_group*).
    epoch : int, optional
        Epoch checkpoint to load.  If ``None`` (default), the checkpoint with
        the lowest validation loss is used.  Raises ``FileNotFoundError`` if
        the requested epoch has no saved checkpoint.
    wandb_group : str, optional
        W&B group name.  When *run_id* is ``None``, filters the auto-selection
        to runs in this group.  If also ``None``, all runs in the project are
        considered.  Also used by the ``"sweep_overview"`` section.
    true_lyapunov : list of float, optional
        Known ground-truth Lyapunov exponents for comparison plots (e.g.
        ``[0.91, 0.0, -14.57]`` for Lorenz).  Pass ``None`` for systems
        without known exponents.
    output : str or sequence of str
        One or more of ``"show"``, ``"save"``, ``"pdf"``, ``"return"``.
    output_dir : str or Path, optional
        Required when ``output`` contains ``"save"`` or ``"pdf"``.
    sections : list of str, optional
        Subset of sections to run.  ``None`` runs all.  Available sections:
        ``"sweep_overview"``, ``"reconstruction"``, ``"mase"``,
        ``"latent_utilization"``, ``"lyapunov"``, ``"kaplan_yorke"``,
        ``"prediction_windows"``, ``"prediction_detail"``,
        ``"long_trajectory"``, ``"encoder_decoder_jacobians"``,
        ``"amplification"``.
    device : str, optional
        PyTorch device string (e.g. ``"cuda"``).  Auto-detected if ``None``.
    n_sample : int
        Number of trajectories sampled for Lyapunov / Jacobian computations.
    n_mase_batches : int
        Number of validation batches for MASE computation.
    n_amp_trajs : int
        Number of trajectories sampled for amplification loss.
    n_amp_neighbors : int
        Number of neighbours for amplification loss.
    n_amp_max_t : int
        Max time horizon for amplification loss.
    n_pred_trajs : int
        Number of test trajectories used for prediction-window analysis.
    lyapunov_burn_in_steps : int
        Artificial burn-in steps prepended for Lyapunov estimation.
    lyapunov_burn_in_drop : int
        Steps dropped from the start of the burn-in when computing exponents.
    sweep_diagnostics : list, optional
        Pre-computed ``DiagnosticMetrics`` list from ``select_from_wandb_runs``.
        Avoids re-running the expensive sweep selection.
    sweep_result : optional
        Pre-computed ``SelectionResult`` from ``select_from_wandb_runs``.
    sweep_lambdas : list of float, optional
        Loop-closure weight values corresponding to ``sweep_diagnostics``.
    return_model : bool
        If ``True``, append the loaded ``LitLatentJacobianODE`` model and its
        W&B run ID to the return value.

    Returns
    -------
    dict[str, Figure] or tuple or None
        * Default (``return_model=False``): returns ``{section: figure}`` when
          ``"return"`` is in *output*, else ``None``.
        * ``return_model=True``: returns ``(figures_or_none, lit_model, run_id)``.
    """
    # ------------------------------------------------------------------ setup
    if isinstance(output, str):
        output = [output]
    else:
        output = list(output)

    if ("save" in output or "pdf" in output) and output_dir is None:
        raise ValueError("output_dir must be provided when output contains 'save' or 'pdf'.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Auto-select best run from sweep if run_id is not provided ----------
    if run_id is None:
        from .tuning import select_best_from_sweep

        if wandb_group is not None:
            print(f"No run_id provided — selecting best run from group '{wandb_group}' ...")
        else:
            print(f"No run_id provided — selecting best run from all runs in '{wandb_project}' ...")
        run_id, _auto_sweep_result, _auto_discovered = select_best_from_sweep(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            save_dir=save_dir,
            wandb_group=wandb_group,
            ranking_method=ranking_method,
            eigenvalue_threshold=eigenvalue_threshold,
            verbose=True,
            use_all_runs=use_all_runs,
        )
        print(f"Auto-selected run_id: {run_id}")

        # Populate sweep precomputed args so sweep_overview doesn't re-run
        if sweep_diagnostics is None:
            sweep_diagnostics = _auto_sweep_result.all_diagnostics
        if sweep_result is None:
            sweep_result = _auto_sweep_result.selection
        if sweep_lambdas is None:
            sweep_lambdas = _auto_discovered.lambdas

        # --- Print Pareto frontier runs and all ranking method picks ----------
        _diags = _auto_sweep_result.all_diagnostics
        _rids = _auto_discovered.run_ids
        _traj = np.array([d.trajectory_val_loss for d in _diags])
        _lc = np.array([
            d.loop_closure_loss if d.loop_closure_loss is not None else np.nan
            for d in _diags
        ])
        _valid = np.isfinite(_lc) & np.isfinite(_traj)

        # Compute Pareto frontier
        _pareto_mask = np.zeros(len(_traj), dtype=bool)
        for _i in range(len(_traj)):
            if not _valid[_i]:
                continue
            _dominated = False
            for _j in range(len(_traj)):
                if _i == _j or not _valid[_j]:
                    continue
                if (_traj[_j] <= _traj[_i] and _lc[_j] <= _lc[_i]
                        and (_traj[_j] < _traj[_i] or _lc[_j] < _lc[_i])):
                    _dominated = True
                    break
            if not _dominated:
                _pareto_mask[_i] = True

        _pareto_idx = np.where(_pareto_mask)[0]
        _pareto_order = _pareto_idx[np.argsort(_lc[_pareto_idx])]

        print(f"\n{'='*70}")
        print(f"PARETO FRONTIER RUNS ({len(_pareto_order)} runs)")
        print(f"{'='*70}")
        print(f"  {'Run ID':<12s}  {'LC Loss':>14s}  {'Traj Val Loss':>14s}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*14}")
        for _pi in _pareto_order:
            _marker = " <-- selected" if _rids[_pi] == run_id else ""
            print(f"  {_rids[_pi]:<12s}  {_lc[_pi]:>14.6f}  {_traj[_pi]:>14.6f}{_marker}")

        # Run all ranking methods and print which run each picks
        from .tuning.ranking import ALL_RANKING_METHODS, rank_survivors
        _survivors = _auto_sweep_result.selection.surviving_indices
        print(f"\n{'='*70}")
        print(f"RANKING METHOD COMPARISON (over {len(_survivors)} survivors)")
        print(f"{'='*70}")
        print(f"  {'Method':<22s}  {'Run ID':<12s}  {'LC Loss':>14s}  {'Traj Val Loss':>14s}")
        print(f"  {'-'*22}  {'-'*12}  {'-'*14}  {'-'*14}")
        for _method in ALL_RANKING_METHODS:
            try:
                _best_idx = rank_survivors(_survivors, _diags, method=_method)
                _marker = " <-- active" if _method == ranking_method else ""
                print(
                    f"  {_method:<22s}  {_rids[_best_idx]:<12s}  "
                    f"{_lc[_best_idx]:>14.6f}  {_traj[_best_idx]:>14.6f}{_marker}"
                )
            except Exception as _e:
                print(f"  {_method:<22s}  ERROR: {_e}")
        print(f"{'='*70}\n")

    active_sections = set(sections if sections is not None else _ALL_SECTIONS)

    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wandb_project_path = f"{wandb_entity}/{wandb_project}"

    # ----------------------------------------------------------- load model
    from . import (
        create_dataloaders,
        load_run,
        load_checkpoint,
    )
    from ..models.latent_jacobian import LitLatentJacobianODE
    from ..fnn import loss_amplification
    from .jacobianODE import JacobianODEint

    print(f"Loading run {run_id} from {wandb_project_path} ...")
    run, cfg, eq, dt, values, _, _, _, _, lit_model = load_run(
        wandb_project_path,
        run_id=run_id,
        save_dir=save_dir,
        generate_data=True,
        verbose=True,
    )

    train_dl, val_dl, test_dl, trajs = create_dataloaders(
        cfg, values, verbose=True, return_full_obs=True
    )

    load_checkpoint(run, cfg, lit_model, save_dir=save_dir, epoch=epoch, verbose=True)

    if true_lyapunov is not None:
        lit_model.true_lyapunov_exponents = torch.tensor(true_lyapunov, dtype=torch.float32)

    lit_model = lit_model.to(device_obj)
    lit_model.eval()

    is_latent = isinstance(lit_model, LitLatentJacobianODE)

    # Encoder-only runs have no trained dynamics, so dynamics-only analytics
    # sections (Lyapunov, Kaplan-Yorke, MASE, prediction windows, long
    # trajectory rollouts) are meaningless. Strip them from the active set
    # before the orchestration loop. Tangent-spectrum, reconstruction,
    # latent_utilization, and encoder/decoder Jacobian sections still apply.
    encoder_only_mode = bool(getattr(lit_model, "encoder_only_mode", False))
    if encoder_only_mode:
        _dropped = {
            "sweep_overview",  # uses C1/C2/C3 selection criteria → all dynamics
            "mase",            # one-step teacher-forced trajectory prediction
            "lyapunov",        # dynamics-MLP Jacobians
            "kaplan_yorke",    # derived from Lyapunov spectrum
            "prediction_windows",
            "prediction_detail",
            "long_trajectory",
        }
        _skipped = active_sections & _dropped
        if _skipped:
            print(
                f"encoder_only_mode=True → skipping dynamics-only sections: "
                f"{sorted(_skipped)}"
            )
        active_sections = active_sections - _dropped

    # Conditioned models (encoder.condition_dim > 0) require per-sample c
    # threaded through every encode_trajectory / compute_jacobians call.
    # Each section pulls c from one of two sources:
    #   _test_seq_cond_full : (n_test_sequences, condition_dim) — aligns
    #     with test_dl.dataset.sequence (per-window, tiled across sliding
    #     windows of each trajectory)
    #   _test_traj_cond_full: (n_test_trajectories, condition_dim) —
    #     aligns with trajs['test_trajs'].sequence (per-trajectory)
    # Both are None when not conditioned; model.* methods accept c=None.
    _is_conditioned_model = bool(getattr(getattr(lit_model, "encoder", None), "condition_dim", 0))
    _test_seq_cond_full = None
    _test_traj_cond_full = None
    if _is_conditioned_model:
        _ds_cond = getattr(test_dl.dataset, "condition", None)
        if _ds_cond is not None:
            _test_seq_cond_full = _ds_cond.to(device_obj)
        _trajs_cond = trajs.get("test_condition") if isinstance(trajs, dict) else None
        if _trajs_cond is not None:
            _test_traj_cond_full = torch.as_tensor(_trajs_cond).float().to(device_obj)

    def _seq_c(idx):
        """Per-sequence-window condition slice; None when not conditioned."""
        return _test_seq_cond_full[idx] if _test_seq_cond_full is not None else None

    def _traj_c(idx):
        """Per-trajectory condition slice; None when not conditioned."""
        return _test_traj_cond_full[idx] if _test_traj_cond_full is not None else None

    # -------------------------------------------------------- config extraction
    mu = float(cfg.data.postprocessing.mu) if np.isscalar(cfg.data.postprocessing.mu) else np.array(cfg.data.postprocessing.mu)
    sigma = float(cfg.data.postprocessing.sigma)
    n_latent = OmegaConf.select(cfg, "model.encoder.n_latent", default=None)
    # Use the actual delay-embedded dimensionality (from the dataloaders),
    # not the raw pre-embedding values which may have more dimensions.
    n_dims = trajs["test_trajs"].sequence.shape[-1]
    if n_latent is None:
        n_latent = n_dims
    is_coupling, n_target_dims = _get_coupling_info(cfg)
    n_dyn = n_target_dims if is_coupling else n_latent

    delay_params = cfg.data.train_test_params.delay_embedding_params
    n_delays = delay_params.n_delays
    delay_spacing = delay_params.delay_spacing
    observed_indices = delay_params.observed_indices
    time_offset = (n_delays - 1) * delay_spacing

    traj_init_steps = lit_model.jacobianODEint_kwargs.get("traj_init_steps", 15)
    if is_latent:
        jac_window_len = traj_init_steps + lit_model.prediction_steps
        jac_window_stride = lit_model.jac_window_stride
    else:
        # Non-latent: no windowing — use the full trajectory length
        T_test = trajs["test_trajs"].sequence.shape[1]
        jac_window_len = T_test
        jac_window_stride = T_test

    test_trajs_full = trajs.get("test_trajs_full", trajs["test_trajs"]).sequence

    # Decoder output dims (may be < n_input for decode_only_recent models)
    if is_latent:
        if hasattr(lit_model.encoder, "decoder"):
            decoder_n_out = lit_model.encoder.decoder.n_output
        else:
            decoder_n_out = lit_model.encoder.n_latent
    else:
        decoder_n_out = n_dims

    # Number of obs dims to use for metrics, matching the model's behaviour:
    # when reconstruction_mode='most_recent', metrics use only the first d dims.
    _n_metric_dims = getattr(lit_model, '_n_recent_dims', None)
    recon_mode = getattr(lit_model, 'reconstruction_mode', 'uniform')
    if recon_mode != 'most_recent' or _n_metric_dims is None:
        _n_metric_dims = None  # use all dims

    # ---------------------------------------------------------------- output helpers
    # Build a shared filename stem used by both PDF and HTML outputs.
    _report_stem = ""
    if any(m in output for m in ("save", "pdf", "html")) and output_dir is not None:
        _date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _name_parts = [_date_str, wandb_project, run_id]
        _report_stem = "__".join(_name_parts)

    pdf_handle: PdfPages | None = None
    if "pdf" in output and output_dir is not None:
        pdf_path = output_dir / (_report_stem + ".pdf")
        pdf_handle = PdfPages(pdf_path)
        print(f"PDF report will be saved to: {pdf_path}")

    html_path: Path | None = None
    _html_parts: list[str] = []
    if "html" in output and output_dir is not None:
        html_path = output_dir / (_report_stem + ".html")
        print(f"HTML report will be saved to: {html_path}")

    figures: dict[str, plt.Figure] = {}
    # Shared state for cross-section data reuse
    _state: dict[str, Any] = {}
    # Accumulated text lines for the final summary section
    _summary_lines: list[str] = []

    def _emit(name: str, fig: plt.Figure) -> None:
        figures[name] = fig
        _save_or_show(fig, name, output, output_dir, pdf_handle)
        if "html" in output:
            _buf = BytesIO()
            fig.savefig(_buf, format="png", bbox_inches="tight", dpi=150)
            _b64 = base64.b64encode(_buf.getvalue()).decode()
            _html_parts.append(
                f'<div class="figure">'
                f'<img src="data:image/png;base64,{_b64}" alt="{name}"/>'
                f'</div>\n'
            )

    def _html_section(title: str, lines: list[str]) -> None:
        """Append a titled <pre> block to the HTML report."""
        if "html" not in output:
            return
        _escaped = "\n".join(lines).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        _html_parts.append(f'<h2>{title}</h2>\n<pre>{_escaped}</pre>\n')

    try:
        # ============================================================
        # 0. Header / run-info text page
        # ============================================================
        _header_lines = [
            f"Run ID         : {run_id}",
            f"Project        : {wandb_project}",
            f"Entity         : {wandb_entity}",
            f"Group          : {wandb_group or '(none)'}",
            f"Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Model configuration",
            f"  n_latent     : {n_latent}",
            f"  n_dyn        : {n_dyn}",
            f"  is_coupling  : {is_coupling}",
            f"  n_delays     : {n_delays}",
            f"  delay_spacing: {delay_spacing}",
            f"  traj_init_steps : {traj_init_steps}",
            f"  prediction_steps: {lit_model.prediction_steps if is_latent else 'N/A (non-latent)'}",
            "",
            "Data",
            f"  n_dims (obs) : {n_dims}",
            f"  sigma        : {sigma:.6g}",
            f"  mu           : {mu if np.isscalar(mu) else list(np.round(mu, 4))}",
            "",
            "Inference settings",
            f"  n_sample (Lyapunov/Jacobian): {n_sample}",
            f"  lyapunov_burn_in_steps      : {lyapunov_burn_in_steps}",
            f"  lyapunov_burn_in_drop       : {lyapunov_burn_in_drop}",
            f"  n_mase_batches              : {n_mase_batches}",
            f"  n_pred_trajs                : {n_pred_trajs}",
        ]
        if true_lyapunov is not None:
            _header_lines += ["", f"True Lyapunov  : {true_lyapunov}"]

        # -- Chosen run diagnostics (from sweep selection) --
        if sweep_result is not None and sweep_result.best_metrics is not None:
            _bm = sweep_result.best_metrics
            _header_lines += [
                "",
                f"Selection method: {ranking_method}",
                "",
                "Selected run diagnostics",
                f"  trajectory val loss       : {_bm.trajectory_val_loss:.6f}",
                f"  loop closure loss         : {_bm.loop_closure_loss:.6f}" if _bm.loop_closure_loss is not None else "  loop closure loss         : N/A",
                f"  one-step MASE             : {_bm.one_step_mase:.6f}",
                f"  fast eigenvalue fraction  : {_bm.fast_eigenvalue_fraction:.6f}",
            ]

        _summary_lines += _header_lines
        _html_section("Run Summary", _header_lines)

        # ============================================================
        # 1. Sweep overview
        # ============================================================
        if "sweep_overview" in active_sections:
            if sweep_diagnostics is not None and sweep_result is not None and sweep_lambdas is not None:
                fig = plot_sweep_overview(sweep_diagnostics, sweep_result, sweep_lambdas, n_dyn)
                _emit("sweep_overview", fig)

                # Pareto front plot (log-log lc_loss vs traj_loss)
                fig_pareto = plot_sweep_pareto(
                    sweep_diagnostics, sweep_result.best_index, ranking_method,
                )
                _emit("sweep_pareto", fig_pareto)
            else:
                print("Skipping 'sweep_overview': provide wandb_group or precomputed sweep_diagnostics.")

        # ============================================================
        # 2. Reconstruction
        # ============================================================
        if "reconstruction" in active_sections and not is_latent:
            print("Skipping 'reconstruction': requires a latent encoder/decoder model.")
        if "reconstruction" in active_sections and is_latent:
            print("Computing reconstruction ...")
            test_trajs_obs = trajs["test_trajs"].sequence
            test_trajs_full_aligned = test_trajs_full[:, time_offset:]

            with torch.no_grad():
                # compute recon nMSE on first 16 test samples
                batch = test_dl.dataset.sequence[:16].to(device_obj)
                batch_c = _seq_c(slice(0, 16))
                z_enc = lit_model.encode_trajectory(batch, batch_c)
                recon = lit_model.decode_trajectory(z_enc, batch_c)
                margin = getattr(lit_model.encoder, "context_margin", 0)
                targets = batch[:, margin:] if margin > 0 else batch
                targets = targets[..., :decoder_n_out]
                recon_mse_val = ((recon - targets) ** 2).mean().item()
                recon_nmse_val = recon_mse_val / targets.var().item()

                # Training-equivalent reconstruction loss: routes through the
                # dyn subspace and applies the model's reconstruction_mode
                # weighting ('most_recent' or 'uniform'). Matches exactly what
                # the lightning loop minimises, unlike recon_mse/nMSE above
                # which are raw sequence-wide metrics.
                try:
                    train_recon_loss_val = float(
                        lit_model._reconstruction_loss(batch, c=batch_c).item()
                    )
                except Exception as e:
                    print(f"  (skipped training-eqv reconstruction loss: {e})")
                    train_recon_loss_val = None

                inv_err_val: float | None = None
                null_rms_val: float | None = None
                if is_coupling:
                    x_rt = lit_model.decode_trajectory(z_enc, batch_c)
                    inv_err_val = float(F.mse_loss(x_rt, batch).item())
                    z_n = _z_null(z_enc, n_target_dims)
                    null_rms_val = (
                        float(z_n.pow(2).mean().sqrt().item())
                        if z_n is not None and z_n.numel() > 0
                        else None
                    )

                # Build decoded trajectory for first test trajectory
                test_trajs_obs_t = test_trajs_obs.to(device_obj)
                test_trajs_c = _traj_c(slice(None))
                latent_full = lit_model.encode_trajectory(test_trajs_obs_t, test_trajs_c)
                if is_coupling:
                    latent_dyn = _z_dyn(latent_full, n_target_dims)
                    latent_padded = lit_model._pad_to_full_dim(latent_dyn)
                else:
                    latent_padded = latent_full
                decoded_full = lit_model.decode_trajectory(latent_padded, test_trajs_c)

            latent_full_cpu = latent_full.cpu()
            decoded_full_cpu = decoded_full.cpu()

            # Determine plot targets and dims
            if getattr(lit_model, "decode_only_recent", False):
                if observed_indices != "all":
                    plot_targets_np = test_trajs_full_aligned[..., observed_indices][0].numpy()
                else:
                    plot_targets_np = test_trajs_full_aligned[0].numpy()
            else:
                plot_targets_np = test_trajs_obs[0].numpy()

            traj_decoded_np = decoded_full_cpu[0].numpy()

            if lit_model.reconstruction_mode == "most_recent":
                plot_dims = np.arange(lit_model.n_recent_dims)
            else:
                n_d = min(10, plot_targets_np.shape[-1], traj_decoded_np.shape[-1])
                plot_dims = np.arange(n_d)

            T_plot = min(plot_targets_np.shape[0], traj_decoded_np.shape[0])
            r2_val = float(r2_score(
                torch.tensor(plot_targets_np[:T_plot][..., plot_dims]).reshape(-1, len(plot_dims)),
                torch.tensor(traj_decoded_np[:T_plot][..., plot_dims]).reshape(-1, len(plot_dims)),
            ))

            _recon_lines = [
                f"R²        : {r2_val:.6f}",
                f"recon nMSE: {recon_nmse_val:.6f}",
            ]
            if train_recon_loss_val is not None:
                mode = getattr(lit_model, "reconstruction_mode", "?")
                _recon_lines.append(
                    f"Training recon loss ({mode}): {train_recon_loss_val:.6f}"
                )
            if is_coupling and inv_err_val is not None:
                _recon_lines.append(f"Inverse consistency MSE: {inv_err_val:.2e}")
                if null_rms_val is not None:
                    _recon_lines.append(f"Null subspace RMS:       {null_rms_val:.2e}")
            _html_section("Reconstruction", _recon_lines)
            fig = plot_reconstruction(
                plot_targets_np[..., plot_dims], traj_decoded_np[..., plot_dims],
                r2_val, recon_nmse_val,
                is_coupling=is_coupling, inv_err=inv_err_val, null_rms=null_rms_val,
            )
            _emit("reconstruction", fig)
            _summary_lines += ["", "=== Reconstruction ==="] + _recon_lines

        # ============================================================
        # 3. MASE
        # ============================================================
        if "mase" in active_sections:
            print("Computing MASE ...")
            model_maes_forced, model_maes_free, persistence_maes = [], [], []
            generator = torch.Generator().manual_seed(42)
            num_samples = n_mase_batches * val_dl.batch_size
            rand_sampler = RandomSampler(
                val_dl.dataset, num_samples=num_samples, replacement=False, generator=generator
            )
            rand_dl = DataLoader(
                val_dl.dataset, batch_size=val_dl.batch_size, sampler=rand_sampler,
                num_workers=val_dl.num_workers,
                pin_memory=getattr(val_dl, "pin_memory", False),
            )
            # When val_dl uses collate_with_optional_condition, items are
            # (batch, c) tuples; otherwise plain tensors. _unpack_batch
            # accepts both forms uniformly.
            for i, item in enumerate(tqdm(rand_dl, desc="MASE batches", total=n_mase_batches)):
                if i >= n_mase_batches:
                    break
                batch, batch_c = lit_model._unpack_batch(item)
                batch = batch.to(device_obj)
                if batch_c is not None:
                    batch_c = batch_c.to(device_obj)
                with torch.no_grad():
                    _tms_kw = dict(alpha_teacher_forcing=1, c=batch_c)
                    if is_latent:
                        _tms_kw["return_decoded"] = True
                    ret_f = lit_model.trajectory_model_step(batch, **_tms_kw)
                    _tms_kw["alpha_teacher_forcing"] = 0
                    ret_r = lit_model.trajectory_model_step(batch, **_tms_kw)
                model_maes_forced.append(ret_f["metric_vals"]["model_mae"].item())
                model_maes_free.append(ret_r["metric_vals"]["model_mae"].item())
                persistence_maes.append(ret_r["metric_vals"]["persistence_mae"].item())

            mean_persist = np.mean(persistence_maes)
            forced_mase = np.mean(model_maes_forced) / mean_persist
            free_mase = np.mean(model_maes_free) / mean_persist
            print(f"Teacher-forced MASE: {forced_mase:.4f}")
            print(f"Free-running MASE:   {free_mase:.4f}")

            _mase_lines = [
                f"Teacher-forced MASE: {forced_mase:.4f}",
                f"Free-running MASE:   {free_mase:.4f}",
                f"Persistence MAE:     {mean_persist:.6f}",
            ]
            _html_section("MASE", _mase_lines)
            fig = plot_mase(forced_mase, free_mase)
            _emit("mase", fig)
            _summary_lines += ["", "=== MASE ==="] + _mase_lines

        # ============================================================
        # 4. Latent utilization
        # ============================================================
        if "latent_utilization" in active_sections and not is_latent:
            print("Skipping 'latent_utilization': requires a latent encoder model.")
        if "latent_utilization" in active_sections and is_latent:
            print("Computing latent utilization ...")
            traj_key = "train_trajs"
            trajs_obs_lat = trajs[traj_key].sequence
            # Per-trajectory condition for the train split (matches
            # trajs['train_trajs'].sequence ordering). None when not conditioned.
            _train_cond_arr = trajs.get("train_condition") if isinstance(trajs, dict) else None
            _train_cond_t = (
                torch.as_tensor(_train_cond_arr).float().to(device_obj)
                if (_is_conditioned_model and _train_cond_arr is not None) else None
            )
            all_latents = []
            with torch.no_grad():
                for i in range(0, trajs_obs_lat.shape[0], 8):
                    x = torch.as_tensor(trajs_obs_lat[i:i + 8]).float().to(device_obj)
                    c_chunk = _train_cond_t[i:i + 8] if _train_cond_t is not None else None
                    z = lit_model.encode_trajectory(x, c_chunk)
                    all_latents.append(z.cpu())
            Z = torch.cat(all_latents, dim=0).numpy()
            Z_dyn = _z_dyn(torch.from_numpy(Z), n_target_dims).numpy()
            Z_flat = Z_dyn.reshape(-1, Z_dyn.shape[-1])
            _state["Z_flat"] = Z_flat

            dim_var = Z_flat.var(axis=0)
            p = dim_var / dim_var.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            utilization = entropy / math.log(n_dyn) if n_dyn > 1 else 1.0
            print(f"Entropy-based utilization: {utilization:.3f}")

            null_rms_per_t: np.ndarray | None = None
            null_mean_rms: float | None = None
            if is_coupling:
                Z_null_arr = _z_null(torch.from_numpy(Z), n_target_dims)
                if Z_null_arr is not None and Z_null_arr.numel() > 0:
                    Zn_np = Z_null_arr.numpy()
                    null_rms_per_t = np.sqrt((Zn_np ** 2).mean(axis=(0, 2)))
                    null_mean_rms = float(np.sqrt((Zn_np ** 2).mean()))
                    print(f"Null subspace mean RMS: {null_mean_rms:.6e}")

            _util_lines = [
                f"Entropy-based utilization: {utilization:.4f}  (1.0 = uniform)",
                f"n_dyn: {n_dyn}",
                "Per-dim fractional variance:",
            ] + [f"  dim {i:2d}: {v / dim_var.sum():.4f}" for i, v in enumerate(dim_var)]
            if is_coupling and null_mean_rms is not None:
                _util_lines.append(f"Null subspace mean RMS: {null_mean_rms:.6e}")
            _html_section("Latent Utilization", _util_lines)
            fig = plot_latent_utilization(dim_var, utilization, n_dyn, null_rms_per_t)
            _emit("latent_utilization", fig)
            _summary_lines += ["", "=== Latent Utilization ==="] + _util_lines

        # ============================================================
        # 5. Lyapunov spectrum
        # ============================================================
        if "lyapunov" in active_sections:
            print("Computing Lyapunov exponents ...")
            # Detect per-trajectory conditions (combined-loader path). When
            # present, the dynamics MLP and encoder were trained with c, so
            # compute_jacobians MUST be called with the right per-traj c —
            # passing c=None into a conditioned MLP raises. Below we thread
            # c into the predicted-Jac computation and use the per-source eq
            # from trajs["source_eqs_by_condition"] for empirical Jacobians.
            _test_condition_arr = trajs.get("test_condition") if isinstance(trajs, dict) else None
            _src_eqs_by_cond = trajs.get("source_eqs_by_condition") if isinstance(trajs, dict) else None
            _is_conditioned = _test_condition_arr is not None and getattr(
                getattr(lit_model, "encoder", None), "condition_dim", 0
            ) > 0
            with torch.no_grad():
                # --- Full-length trajectory Lyapunov (PLOTTED) ---
                # Uses trajs['test_trajs'].sequence — the actual full-length test
                # trajectories, not the windowed batches from the dataloader.
                # Process one trajectory at a time: encode → Jacobians (T_full, D, D)
                # → Lyapunov (D,) for that single trajectory.
                traj_seq_full = torch.as_tensor(
                    trajs["test_trajs"].sequence
                ).float().to(device_obj)
                _c_full_t = (
                    torch.as_tensor(_test_condition_arr).float().to(device_obj)
                    if _is_conditioned else None
                )
                z_seq_full = (
                    lit_model.encode_trajectory(traj_seq_full, _c_full_t)
                    if is_latent and _is_conditioned
                    else (lit_model.encode_trajectory(traj_seq_full) if is_latent else traj_seq_full)
                )
                z_for_jac_full = _z_dyn(z_seq_full, n_target_dims)
                n_full_trajs = z_for_jac_full.shape[0]
                print(f"  Computing full-trajectory Lyapunov ({n_full_trajs} test trajs, "
                      f"T={traj_seq_full.shape[1]}) ...")
                # Chunked-batched: compute_jacobians and compute_lyapunov_exponents
                # both broadcast over a leading batch dim. Process the trajectories
                # in chunks to bound peak memory (jacs are (B_chunk, T, D, D)) while
                # keeping everything on device until the final stack.
                _lyap_chunk_size = 64
                _lyap_full_list = []
                _n_chunks = (n_full_trajs + _lyap_chunk_size - 1) // _lyap_chunk_size
                for _ci in tqdm(
                    range(0, n_full_trajs, _lyap_chunk_size),
                    total=_n_chunks,
                    desc="    full-traj Lyap chunks",
                ):
                    _z_chunk = z_for_jac_full[_ci:_ci + _lyap_chunk_size]
                    _c_chunk = _c_full_t[_ci:_ci + _lyap_chunk_size] if _is_conditioned else None
                    _jacs_chunk = lit_model.compute_jacobians(_z_chunk, c=_c_chunk)
                    _le_chunk = LitLatentJacobianODE.compute_lyapunov_exponents(
                        _jacs_chunk, dt
                    )
                    _lyap_full_list.append(_le_chunk.cpu())
                all_pred_lyap_full = torch.cat(_lyap_full_list, dim=0)
                _state["all_pred_lyap_full"] = all_pred_lyap_full

                # --- Batch + burn-in Lyapunov (128 sampled windowed trajs) ---
                # Skip for conditioned models: JacobianODEint.generate_dynamics
                # calls compute_jacobians without c, which would raise on a
                # conditioned MLP. Full-trajectory Lyap (above) is the
                # primary signal; burn-in is supplementary and would need a
                # per-condition refactor to support c-aware rollout.
                if _is_conditioned:
                    all_pred_lyap = all_pred_lyap_full  # alias for downstream code
                else:
                    traj_batched_t = torch.as_tensor(
                        test_dl.dataset.sequence
                    ).float().to(device_obj)
                    # Encode in chunks to avoid OOM on large test sets
                    if is_latent:
                        _enc_chunks = []
                        _chunk_size = 64
                        for _ci in range(0, traj_batched_t.shape[0], _chunk_size):
                            _enc_chunks.append(
                                lit_model.encode_trajectory(traj_batched_t[_ci:_ci + _chunk_size])
                            )
                        z_batched_t = torch.cat(_enc_chunks, dim=0)
                    else:
                        z_batched_t = traj_batched_t
                    z_for_jac = _z_dyn(z_batched_t, n_target_dims)
                    gen = torch.Generator().manual_seed(0)
                    perm = torch.randperm(z_for_jac.shape[0], generator=gen)[:n_sample]
                    z_sampled = z_for_jac[perm]

                    B, T_true, D_dyn = z_sampled.shape
                    z_padded = torch.cat(
                        [z_sampled, torch.zeros(B, lyapunov_burn_in_steps, D_dyn, device=device_obj)],
                        dim=1,
                    )
                    jacobian_odeint = JacobianODEint(lit_model.compute_jacobians, dt)
                    z_combined = jacobian_odeint.generate_dynamics(
                        z_padded,
                        traj_init_steps=T_true,
                        alpha_teacher_forcing=0.0,
                        fast_mode=True,
                        verbose=False,
                        interp_pts=4,
                        inner_N=20,
                    )
                    jacs_burn = lit_model.compute_jacobians(z_combined)
                    all_pred_lyap = LitLatentJacobianODE.compute_lyapunov_exponents(
                        jacs_burn[:, lyapunov_burn_in_drop:], dt
                    )

            pred_lyap = all_pred_lyap.mean(dim=0).cpu()
            pred_lyap_std = all_pred_lyap.std(dim=0).cpu()
            pred_np = pred_lyap.numpy()
            pred_std_np = pred_lyap_std.numpy()
            print(f"Predicted Lyapunov exponents (batch+burn-in, {n_sample} windowed trajs):")
            for i, (le, std) in enumerate(zip(pred_lyap, pred_lyap_std)):
                print(f"  λ_{i+1} = {le.item():+.4f} ± {std.item():.4f}")

            pred_lyap_full_mean = all_pred_lyap_full.mean(dim=0)
            pred_lyap_full_std_t = all_pred_lyap_full.std(dim=0)
            pred_full_np = pred_lyap_full_mean.numpy()
            pred_full_std_np = pred_lyap_full_std_t.numpy()
            print(f"Predicted Lyapunov exponents (full-length, {n_full_trajs} test trajs):")
            for i, (le, std) in enumerate(zip(pred_lyap_full_mean, pred_lyap_full_std_t)):
                print(f"  λ_{i+1} = {le.item():+.4f} ± {std.item():.4f}")

            if true_lyapunov:
                print(f"True:      {true_lyapunov}")

            # Empirical from analytical Jacobian (if eq available).
            # For conditioned runs, use trajs["source_eqs_by_condition"] —
            # a list of (cond_row, eq) — to apply the matching per-source eq
            # to each subset of trajectories. Falls back to the single `eq`
            # when conditioning isn't in play (back-compat).
            emp_np: np.ndarray | None = None
            emp_std_np: np.ndarray | None = None
            _has_eq_source = (eq is not None) or (_src_eqs_by_cond is not None and len(_src_eqs_by_cond) > 0)
            if _has_eq_source:
                mu_val = cfg.data.postprocessing.mu
                sigma_norm = cfg.data.postprocessing.sigma
                if "test_trajs_full" in trajs:
                    traj_full_np = trajs["test_trajs_full"].sequence
                else:
                    traj_full_np = trajs["test_trajs"].sequence
                traj_raw = np.asarray(traj_full_np) * sigma_norm + mu_val

                # Build a list of (label, mask, src_eq) groups to iterate over.
                # Single-source: one group spanning all trajectories with the
                # provided `eq`. Multi-source: one group per (cond_row, eq) in
                # the lookup, masked by per-traj condition match.
                n_test_t = traj_raw.shape[0]
                _emp_groups: list[tuple[str, np.ndarray, Any]] = []
                if _is_conditioned and _src_eqs_by_cond:
                    test_cond_arr = np.asarray(_test_condition_arr)
                    for cond_row, src_eq in _src_eqs_by_cond:
                        mask = np.all(test_cond_arr == cond_row, axis=1)
                        if not mask.any():
                            continue
                        _emp_groups.append(
                            (f"c={cond_row.tolist()}", mask, src_eq)
                        )
                else:
                    _emp_groups.append(("all", np.ones(n_test_t, dtype=bool), eq))

                # Compute per-group empirical Lyap, store on `_state` as a
                # dict {label: tensor (n_in_group, n_lyaps)} for the plot/report
                # path. Also build the legacy `all_emp_lyap_t` (concatenated)
                # so KY dimension and other downstream code keeps working.
                emp_per_group: dict[str, torch.Tensor] = {}
                concat_lyaps: list[torch.Tensor] = []
                for label, mask, src_eq in _emp_groups:
                    sub_raw = traj_raw[mask]
                    if hasattr(src_eq, "model"):
                        sub_raw_t = torch.as_tensor(sub_raw).float().to(device_obj)
                        _emp_chunk_size = 64
                        n_sub = sub_raw_t.shape[0]
                        _emp_n_chunks = (n_sub + _emp_chunk_size - 1) // _emp_chunk_size
                        _emp_chunks: list[torch.Tensor] = []
                        for _ci in tqdm(
                            range(0, n_sub, _emp_chunk_size),
                            total=_emp_n_chunks,
                            desc=f"    empirical Lyap chunks ({label})",
                        ):
                            _traj_chunk = sub_raw_t[_ci:_ci + _emp_chunk_size]
                            _jacs_chunk = src_eq.jac(_traj_chunk, t=0)
                            _le_chunk = LitLatentJacobianODE.compute_lyapunov_exponents(
                                _jacs_chunk, dt
                            )
                            _emp_chunks.append(_le_chunk.cpu())
                        group_lyap = torch.cat(_emp_chunks, dim=0)
                    else:
                        # Numpy-based dysts eq.jac: D is small (3-5), per-traj loop is
                        # cheap and the internal jac dispatcher already handles only
                        # specific input shapes. Leave it untouched.
                        group_lyaps = []
                        for i in range(sub_raw.shape[0]):
                            traj_i = sub_raw[i]
                            jacs_np = src_eq.jac(traj_i, t=0)
                            jacs_t = torch.as_tensor(jacs_np).float()
                            le_i = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_t, dt)
                            group_lyaps.append(le_i)
                        group_lyap = torch.stack(group_lyaps).cpu()
                    emp_per_group[label] = group_lyap
                    concat_lyaps.append(group_lyap)
                all_emp_lyap_t = torch.cat(concat_lyaps, dim=0)
                _state["all_emp_lyap_t"] = all_emp_lyap_t
                _state["emp_lyap_per_group"] = emp_per_group
                emp_np = all_emp_lyap_t.mean(dim=0).numpy()
                emp_std_np = all_emp_lyap_t.std(dim=0).numpy()
                print("Empirical Lyapunov exponents (mean ± std, all trajectories):")
                for i, (le, std) in enumerate(zip(emp_np, emp_std_np)):
                    print(f"  λ_{i+1} = {le:+.4f} ± {std:.4f}")
                if _is_conditioned and len(emp_per_group) > 1:
                    print("Empirical Lyapunov per condition:")
                    for label, lyap in emp_per_group.items():
                        print(f"  {label}:")
                        m = lyap.mean(dim=0).numpy()
                        s = lyap.std(dim=0).numpy()
                        for i, (le, std) in enumerate(zip(m, s)):
                            print(f"    λ_{i+1} = {le:+.4f} ± {std:.4f}")

            _state["all_pred_lyap"] = all_pred_lyap.cpu()

            # Per-condition predicted Lyap groups: same condition partition
            # used for empirical above, but applied to the full-trajectory
            # predicted Lyap. Stored on _state so the plot function can
            # overlay one curve per condition for both pred and emp.
            pred_lyap_per_group: dict[str, torch.Tensor] = {}
            if _is_conditioned and _src_eqs_by_cond:
                test_cond_arr = np.asarray(_test_condition_arr)
                for cond_row, _src_eq in _src_eqs_by_cond:
                    mask = np.all(test_cond_arr == cond_row, axis=1)
                    if not mask.any():
                        continue
                    label = f"c={cond_row.tolist()}"
                    pred_lyap_per_group[label] = all_pred_lyap_full[mask]
            _state["pred_lyap_per_group"] = pred_lyap_per_group

            best_lambda = None
            try:
                best_lambda = float(
                    cfg.get("training", {}).get("lightning", {}).get("loop_closure_weight")
                )
            except Exception:
                pass

            _lyap_lines = [f"Predicted (batch+burn-in, {n_sample} trajs):"]
            for i, (le, std) in enumerate(zip(pred_np, pred_std_np)):
                _lyap_lines.append(f"  λ_{i+1:2d} = {le:+.4f} ± {std:.4f}")
            _lyap_lines.append(f"Predicted (full-length, {n_full_trajs} test trajs):")
            for i, (le, std) in enumerate(zip(pred_full_np, pred_full_std_np)):
                _lyap_lines.append(f"  λ_{i+1:2d} = {le:+.4f} ± {std:.4f}")
            if emp_np is not None:
                _lyap_lines.append("Empirical (ground-truth Jacobian):")
                for i, (le, std) in enumerate(zip(emp_np, emp_std_np)):
                    _lyap_lines.append(f"  λ_{i+1:2d} = {le:+.4f} ± {std:.4f}")
            if true_lyapunov is not None:
                _lyap_lines.append(f"Literature: {true_lyapunov}")
            _html_section("Lyapunov Spectrum", _lyap_lines)
            _summary_lines += ["", "=== Lyapunov Spectrum ==="] + _lyap_lines

            if _is_conditioned and pred_lyap_per_group and _state.get("emp_lyap_per_group"):
                # Per-condition overlay: one curve per (condition × {pred, emp}).
                result = plot_lyapunov_spectrum_by_condition(
                    pred_per_group={k: v.numpy() for k, v in pred_lyap_per_group.items()},
                    emp_per_group={k: v.numpy() for k, v in _state["emp_lyap_per_group"].items()},
                    true_lyapunov=true_lyapunov,
                    loop_closure_weight=best_lambda,
                )
            else:
                result = plot_lyapunov_spectrum(
                    pred_np, pred_std_np, emp_np, emp_std_np,
                    true_lyapunov=true_lyapunov, loop_closure_weight=best_lambda,
                    full_lyap_np=pred_full_np, full_lyap_std_np=pred_full_std_np,
                )
            if isinstance(result, list):
                _emit("lyapunov", result[0])
                _emit("lyapunov_top10", result[1])
            else:
                _emit("lyapunov", result)

        # ============================================================
        # 6. Kaplan-Yorke dimension
        # ============================================================
        if "kaplan_yorke" in active_sections:
            # Prefer full-length Lyapunov (per-trajectory, aligns with empirical count).
            # Fall back to burn-in version, then recompute if neither is available.
            if "all_pred_lyap_full" in _state:
                all_pred_lyap_ky = _state["all_pred_lyap_full"]
            elif "all_pred_lyap" in _state:
                all_pred_lyap_ky = _state["all_pred_lyap"]
            else:
                print("Computing Lyapunov exponents for KY dimension (full-length, chunk-batched) ...")
                with torch.no_grad():
                    traj_full_t = torch.as_tensor(test_dl.dataset.sequence).float().to(device_obj)
                    z_full_t = lit_model.encode_trajectory(traj_full_t) if is_latent else traj_full_t
                    z_for_jac_ky = _z_dyn(z_full_t, n_target_dims)
                    _ky_n = z_for_jac_ky.shape[0]
                    _ky_chunk_size = 64
                    _ky_n_chunks = (_ky_n + _ky_chunk_size - 1) // _ky_chunk_size
                    _lyap_ky_list = []
                    for _ci in tqdm(
                        range(0, _ky_n, _ky_chunk_size),
                        total=_ky_n_chunks,
                        desc="    KY Lyap chunks",
                    ):
                        _z_chunk = z_for_jac_ky[_ci:_ci + _ky_chunk_size]
                        _jacs_chunk = lit_model.compute_jacobians(_z_chunk)
                        _le_chunk = LitLatentJacobianODE.compute_lyapunov_exponents(
                            _jacs_chunk, dt
                        )
                        _lyap_ky_list.append(_le_chunk.cpu())
                    all_pred_lyap_ky = torch.cat(_lyap_ky_list, dim=0)

            ky_val = _kaplan_yorke_dim(all_pred_lyap_ky)
            ky_pred_np = np.atleast_1d(ky_val.cpu().numpy() if torch.is_tensor(ky_val) else np.array(ky_val))

            ky_emp_np_ky: np.ndarray | None = None
            if "all_emp_lyap_t" in _state:
                ky_emp_t = _kaplan_yorke_dim(_state["all_emp_lyap_t"])
                ky_emp_np_ky = np.atleast_1d(
                    ky_emp_t.cpu().numpy() if torch.is_tensor(ky_emp_t) else np.array(ky_emp_t)
                )

            ky_theory: float | None = None
            if true_lyapunov is not None:
                lyap_t = np.sort(true_lyapunov)[::-1]
                cum = np.cumsum(lyap_t)
                k = np.where(cum > 0)[0][-1] if np.any(cum > 0) else -1
                if k >= 0 and (k + 1) < len(lyap_t) and lyap_t[k + 1] != 0:
                    ky_theory = float((k + 1) + cum[k] / np.abs(lyap_t[k + 1]))

            print(f"Mean KY dim (predicted): {ky_pred_np.mean():.3f} ± {ky_pred_np.std():.3f}")
            if ky_emp_np_ky is not None:
                print(f"Mean KY dim (empirical): {ky_emp_np_ky.mean():.3f} ± {ky_emp_np_ky.std():.3f}")
            if ky_theory is not None:
                print(f"KY dim (theory):         {ky_theory:.3f}")

            _ky_lines = [
                f"Predicted (full-length): {ky_pred_np.mean():.4f} ± {ky_pred_np.std():.4f}",
            ]
            if ky_emp_np_ky is not None:
                _ky_lines.append(
                    f"Empirical:               {ky_emp_np_ky.mean():.4f} ± {ky_emp_np_ky.std():.4f}"
                )
            if ky_theory is not None:
                _ky_lines.append(f"Theory:                  {ky_theory:.4f}")
            _html_section("Kaplan-Yorke Dimension", _ky_lines)
            _summary_lines += ["", "=== Kaplan-Yorke Dimension ==="] + _ky_lines

            fig = plot_kaplan_yorke(ky_pred_np, ky_emp_np_ky, ky_theory)
            _emit("kaplan_yorke", fig)

            # PCA scatter: latent space vs. true state
            # Z_flat from latent_utilization section (or recompute here)
            if "Z_flat" not in _state:
                print("  Computing Z_flat for PCA (latent_utilization was skipped) ...")
                _trajs_lat = trajs["train_trajs"].sequence
                _all_lat = []
                with torch.no_grad():
                    for _i in range(0, _trajs_lat.shape[0], 8):
                        _x = torch.as_tensor(_trajs_lat[_i:_i + 8]).float().to(device_obj)
                        _z = lit_model.encode_trajectory(_x) if is_latent else _x
                        _all_lat.append(_z_dyn(_z, n_target_dims).cpu())
                _Z = torch.cat(_all_lat, dim=0).numpy()
                _state["Z_flat"] = _Z.reshape(-1, _Z.shape[-1])

            _Z_flat_pca = _state["Z_flat"]
            _X_true_src = trajs.get("train_trajs_full", trajs["train_trajs"])
            _X_true_flat = np.asarray(
                _X_true_src.sequence
            ).reshape(-1, _X_true_src.sequence.shape[-1])
            _mean_ky_true_pca = float(ky_emp_np_ky.mean()) if ky_emp_np_ky is not None else None

            # Compute burn-in D_KY if burn-in Lyapunov exponents are available
            _mean_ky_burnin: float | None = None
            if "all_pred_lyap" in _state:
                _ky_burnin = _kaplan_yorke_dim(_state["all_pred_lyap"])
                _ky_burnin_np = np.atleast_1d(
                    _ky_burnin.cpu().numpy() if torch.is_tensor(_ky_burnin) else np.array(_ky_burnin)
                )
                _mean_ky_burnin = float(_ky_burnin_np.mean())
                print(f"Mean KY dim (burn-in):   {_mean_ky_burnin:.3f} ± {_ky_burnin_np.std():.3f}")

            fig_pca = plot_pca_kaplan_yorke(
                _Z_flat_pca, _X_true_flat,
                mean_ky_latent=float(ky_pred_np.mean()),
                mean_ky_true=_mean_ky_true_pca,
                mean_ky_latent_burnin=_mean_ky_burnin,
            )
            _emit("kaplan_yorke_pca", fig_pca)

        # ============================================================
        # 7 & 8. Prediction windows + detail
        # ============================================================
        if "prediction_windows" in active_sections or "prediction_detail" in active_sections:
            print("Computing prediction windows ...")
            test_trajs_obs_pw = trajs["test_trajs"].sequence
            N_TEST = min(n_pred_trajs, test_trajs_obs_pw.shape[0])

            all_decoded_pred_pw, all_obs_targets_pw = [], []
            all_z_pred_pw, all_z_true_windows_pw = [], []
            all_per_window_nmse_pw = []

            for t_idx in range(N_TEST):
                traj_i = torch.as_tensor(
                    test_trajs_obs_pw[t_idx:t_idx + 1]
                ).float().to(device_obj)
                c_i = _traj_c(slice(t_idx, t_idx + 1))
                with torch.no_grad():
                    _pw_kw: dict = dict(alpha_teacher_forcing=0.0, obs_noise_scale=0, c=c_i)
                    if is_latent:
                        _pw_kw["return_decoded"] = True
                    rd = lit_model.trajectory_model_step(traj_i, **_pw_kw)
                    z_true_i = lit_model.encode_trajectory(traj_i, c_i) if is_latent else traj_i

                z_pred_i = rd["outputs"].cpu()
                if is_latent:
                    dec_pred = rd["decoded"].cpu()
                    obs_tgt = rd["targets"].cpu()
                else:
                    # Non-latent: outputs ARE obs-space predictions; targets
                    # are the label shifted by traj_init_steps.
                    _tis = traj_init_steps
                    dec_pred = z_pred_i[..., _tis:, :]
                    obs_tgt = traj_i[..., _tis:, :].cpu()

                # Slice to metric dims (matches model's reconstruction_mode)
                _dp = dec_pred[..., :_n_metric_dims] if _n_metric_dims else dec_pred
                _ot = obs_tgt[..., :_n_metric_dims] if _n_metric_dims else obs_tgt

                mean_var = _ot.reshape(-1, _ot.shape[-1]).var(dim=0).mean().clamp(min=1e-8)
                for w in range(_dp.shape[0]):
                    w_mse = (_dp[w] - _ot[w]).pow(2).mean()
                    all_per_window_nmse_pw.append((w_mse / mean_var).item())

                z_true_np = z_true_i[0].cpu()
                T_prime = z_true_np.shape[0]
                n_windows = max(1, (T_prime - jac_window_len) // jac_window_stride + 1)
                for w in range(n_windows):
                    start = w * jac_window_stride
                    if start + jac_window_len <= T_prime:
                        all_z_true_windows_pw.append(z_true_np[start:start + jac_window_len].numpy())

                all_decoded_pred_pw.append(dec_pred.numpy())
                all_obs_targets_pw.append(obs_tgt.numpy())
                all_z_pred_pw.append(z_pred_i.numpy())

            all_decoded_pred_pw_arr = np.concatenate(all_decoded_pred_pw, axis=0)
            all_obs_targets_pw_arr = np.concatenate(all_obs_targets_pw, axis=0)
            all_z_pred_pw_arr = np.concatenate(all_z_pred_pw, axis=0)
            all_z_true_windows_arr = np.array(all_z_true_windows_pw)
            all_per_window_nmse_arr = np.array(all_per_window_nmse_pw)

            print(
                f"Windows: {len(all_per_window_nmse_arr)} — "
                f"nMSE min={all_per_window_nmse_arr.min():.4f}, "
                f"median={np.median(all_per_window_nmse_arr):.4f}, "
                f"mean={all_per_window_nmse_arr.mean():.4f}, "
                f"max={all_per_window_nmse_arr.max():.4f}"
            )

            if "prediction_windows" in active_sections:
                fig = plot_prediction_windows(all_per_window_nmse_arr)
                _emit("prediction_windows", fig)

            if "prediction_detail" in active_sections:
                median_idx = int(np.argsort(all_per_window_nmse_arr)[len(all_per_window_nmse_arr) // 2])
                z_pred_med = all_z_pred_pw_arr[median_idx]
                z_true_med = all_z_true_windows_arr[median_idx] if len(all_z_true_windows_arr) > median_idx else z_pred_med * 0
                obs_pred_med = all_decoded_pred_pw_arr[median_idx]
                obs_true_med = all_obs_targets_pw_arr[median_idx]

                # Compute latent-space nMSE on the prediction portion only,
                # matching dynamic-subspace dims between z_pred and z_true.
                D_dyn = z_pred_med.shape[-1]
                z_pred_pred = z_pred_med[traj_init_steps:]
                z_true_pred = z_true_med[traj_init_steps:, :D_dyn]
                _z_var = z_true_pred.var(axis=0).mean()
                _z_mse = ((z_pred_pred - z_true_pred) ** 2).mean()
                latent_nmse = float(_z_mse / max(_z_var, 1e-8))

                fig_z, fig_obs = plot_prediction_detail(
                    z_pred_med, z_true_med, obs_pred_med, obs_true_med,
                    sigma=sigma, mu=mu,
                    traj_init_steps=traj_init_steps,
                    nmse_val=all_per_window_nmse_arr[median_idx],
                    title_suffix="Median-loss window",
                    latent_nmse_val=latent_nmse,
                    n_metric_dims=_n_metric_dims,
                )
                _emit("prediction_detail_latent", fig_z)
                _emit("prediction_detail_obs", fig_obs)

        # ============================================================
        # 9. Long trajectory (free-running rollouts at multiple init regimes)
        # ============================================================
        if "long_trajectory" in active_sections:
            print("Computing long-trajectory free-running rollouts ...")
            test_trajs_obs_lt = trajs["test_trajs"].sequence
            traj_long = torch.as_tensor(test_trajs_obs_lt[[0]]).float().to(device_obj)
            traj_long_c = _traj_c(slice(0, 1))
            T_full = traj_long.shape[1]

            if not is_latent:
                # Vanilla models: no encoder — fall back to the legacy
                # single-seed plot using trajectory_model_step.
                with torch.no_grad():
                    rd_long = lit_model.trajectory_model_step(
                        traj_long, alpha_teacher_forcing=0.0, obs_noise_scale=0,
                        c=traj_long_c,
                    )
                traj_true_lt = traj_long[0].cpu().numpy()
                decoded_pred_lt = rd_long["outputs"][0].cpu().numpy()
                fig = plot_long_trajectory(
                    traj_true_lt, decoded_pred_lt,
                    traj_long[0].cpu().numpy(),
                    rd_long["outputs"][0].cpu().numpy(),
                    traj_init_steps=traj_init_steps,
                )
                _emit("long_trajectory", fig)
            else:
                # Latent model: roll out three init regimes and show each
                # in a row.  All three use the training config's
                # integration kwargs (only traj_init_steps varies).
                from .jacobianODE import JacobianODEint
                ikw = dict(cfg.training.lightning.jacobianODEint_kwargs)

                # Encode once; each seed slices from the front and pads
                # the rest with zeros so generate_dynamics rolls forward.
                with torch.no_grad():
                    z_full_enc = lit_model.encode_trajectory(traj_long, traj_long_c)
                z_dyn_enc, _ = lit_model._split_latent(z_full_enc)  # (1, T_full, D_dyn)
                D_dyn = z_dyn_enc.shape[-1]

                # Three init regimes.
                # 1) train seeding: the traj_init_steps used during
                #    training (typically 15 for seq_length=45).
                # 2) proportional seeding: init = T_full / 3 so the
                #    init:rollout ratio matches training's 1:2.
                # 3) triple seeding: 3× training's init, i.e. 45 for the
                #    standard config — what the current batch+burn-in
                #    Lyapunov analysis effectively uses.
                train_init = int(ikw.get("traj_init_steps", 15))
                seeds = [
                    ("train seeding",         train_init),
                    ("proportional seeding (init=T/3)", T_full // 3),
                    (f"triple seeding (3× train = {3 * train_init})",
                     3 * train_init),
                ]
                # Drop seeds whose init >= T_full; keep order.
                seeds = [(n, k) for (n, k) in seeds if 0 < k < T_full]

                # Bind the per-trajectory c into compute_jacobians so the
                # integrator's variadic call (z, t) doesn't need to know about c.
                _jac_fn = (
                    (lambda z, *_a, _c=traj_long_c, **_k: lit_model.compute_jacobians(z, c=_c))
                    if traj_long_c is not None else lit_model.compute_jacobians
                )
                jac_ode = JacobianODEint(_jac_fn, dt)
                rollouts = []
                for name, init_n in seeds:
                    roll_n = T_full - init_n
                    z_init = z_dyn_enc[:, :init_n, :]
                    z_padded = torch.cat(
                        [z_init,
                         torch.zeros(1, roll_n, D_dyn, device=device_obj)],
                        dim=1,
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
                        )  # (1, T_full, D_dyn)
                        z_full_pred = lit_model._pad_to_full_dim(z_pred)
                        decoded = lit_model.decode_trajectory(z_full_pred, traj_long_c)[0]
                    rollouts.append(dict(
                        name=name,
                        init_n=init_n,
                        decoded=decoded.cpu().numpy(),
                        latent_true=_z_dyn(z_full_enc[0], n_target_dims).cpu().numpy(),
                        latent_pred=z_pred[0].cpu().numpy(),
                    ))

                fig = plot_seeded_rollouts(traj_long[0].cpu().numpy(), rollouts)
                _emit("long_trajectory", fig)

        # ============================================================
        # 10. Encoder/decoder Jacobians
        # ============================================================
        if "encoder_decoder_jacobians" in active_sections and not is_latent:
            print("Skipping 'encoder_decoder_jacobians': requires a latent encoder/decoder model.")
        if "encoder_decoder_jacobians" in active_sections and is_latent:
            print("Computing encoder/decoder Jacobians ...")
            from torch.func import vmap, jacrev, jacfwd

            N_JAC = min(n_sample, 512)
            _enc = lit_model.encoder
            _enc.eval()
            _traj_jac = torch.as_tensor(
                trajs["test_trajs"].sequence, dtype=torch.float32, device=device_obj
            )
            D_obs_jac = _traj_jac.shape[-1]
            rng_jac = np.random.default_rng(123)

            # Probe encoder interface
            _probe = torch.zeros(1, D_obs_jac, device=device_obj, dtype=_traj_jac.dtype)
            with torch.no_grad():
                try:
                    _enc.encode(_probe)
                    enc_accepts_flat = True
                except Exception:
                    enc_accepts_flat = False

            # Per-trajectory condition tensor for the test split. We later
            # slice it to whichever subset of trajectories the encoder
            # branch picked. None when not conditioned.
            _ed_traj_cond = _test_traj_cond_full

            with torch.no_grad():
                if hasattr(_enc, "time_window"):
                    _w = _enc.time_window
                    _windows = _traj_jac.unfold(1, _w, 1).permute(0, 1, 3, 2).reshape(-1, _w, D_obs_jac)
                    n_win = _windows.shape[0]
                    idx_w = rng_jac.choice(n_win, min(N_JAC, n_win), replace=False)
                    _windows_s = _windows[torch.from_numpy(idx_w).to(device_obj)]
                    # Each window came from a specific trajectory; map window
                    # idx → traj idx so we can pull the matching condition.
                    if _ed_traj_cond is not None:
                        n_windows_per_traj = n_win // _traj_jac.shape[0]
                        traj_idx_for_w = idx_w // n_windows_per_traj
                        _windows_c = _ed_traj_cond[torch.as_tensor(traj_idx_for_w, device=device_obj)]
                    else:
                        _windows_c = None

                    def _enc_one(x, c=None): return _enc.encode(x.unsqueeze(0), c.unsqueeze(0) if c is not None else None).squeeze(0)
                    def _dec_one(z, c=None): return _enc.decode(z.unsqueeze(0), c.unsqueeze(0) if c is not None else None).squeeze(0)

                    if _windows_c is not None:
                        _z_jac = _enc.encode(_windows_s, _windows_c)
                        encoder_jacobian = vmap(jacrev(_enc_one, argnums=0))(_windows_s, _windows_c)
                        decoder_jacobian = vmap(jacfwd(_dec_one, argnums=0))(_z_jac, _windows_c)
                    else:
                        _z_jac = _enc.encode(_windows_s)
                        encoder_jacobian = vmap(jacrev(lambda x: _enc_one(x)))(_windows_s)
                        decoder_jacobian = vmap(jacfwd(lambda z: _dec_one(z)))(_z_jac)

                elif enc_accepts_flat:
                    test_pts = _traj_jac.reshape(-1, D_obs_jac)
                    n_pts = test_pts.shape[0]
                    idx_p = rng_jac.choice(n_pts, min(N_JAC, n_pts), replace=False)
                    x_flat = test_pts[torch.from_numpy(idx_p).to(device_obj)]
                    if _ed_traj_cond is not None:
                        T = _traj_jac.shape[1]
                        traj_idx_for_p = idx_p // T  # each test_pts row came from this trajectory
                        x_c = _ed_traj_cond[torch.as_tensor(traj_idx_for_p, device=device_obj)]
                    else:
                        x_c = None

                    def _enc_one(x, c=None): return _enc.encode(x.unsqueeze(0), c.unsqueeze(0) if c is not None else None).squeeze(0)
                    def _dec_one(z, c=None): return _enc.decode(z.unsqueeze(0), c.unsqueeze(0) if c is not None else None).squeeze(0)

                    if x_c is not None:
                        _z_jac = _enc.encode(x_flat, x_c)
                        encoder_jacobian = vmap(jacrev(_enc_one, argnums=0))(x_flat, x_c)
                        decoder_jacobian = vmap(jacfwd(_dec_one, argnums=0))(_z_jac, x_c)
                    else:
                        _z_jac = _enc.encode(x_flat)
                        encoder_jacobian = vmap(jacrev(lambda x: _enc_one(x)))(x_flat)
                        decoder_jacobian = vmap(jacfwd(lambda z: _dec_one(z)))(_z_jac)

                else:
                    B_jac = _traj_jac.shape[0]
                    idx_t = rng_jac.choice(B_jac, min(N_JAC, B_jac), replace=False)
                    idx_t_t = torch.from_numpy(idx_t).to(device_obj)
                    _traj_s = _traj_jac[idx_t_t]
                    _traj_s_c = _ed_traj_cond[idx_t_t] if _ed_traj_cond is not None else None

                    def _enc_traj(x, c=None):
                        return lit_model.encode_trajectory(x.unsqueeze(0), c.unsqueeze(0) if c is not None else None).squeeze(0)
                    def _dec_traj(z, c=None):
                        return lit_model.decode_trajectory(z.unsqueeze(0), c.unsqueeze(0) if c is not None else None).squeeze(0)

                    if _traj_s_c is not None:
                        _z_jac = lit_model.encode_trajectory(_traj_s, _traj_s_c)
                        encoder_jacobian = vmap(jacrev(_enc_traj, argnums=0))(_traj_s, _traj_s_c)
                        decoder_jacobian = vmap(jacfwd(_dec_traj, argnums=0))(_z_jac, _traj_s_c)
                    else:
                        _z_jac = lit_model.encode_trajectory(_traj_s)
                        encoder_jacobian = vmap(jacrev(lambda x: _enc_traj(x)))(_traj_s)
                        decoder_jacobian = vmap(jacfwd(lambda z: _dec_traj(z)))(_z_jac)

            print(f"encoder_jacobian: {tuple(encoder_jacobian.shape)}")
            print(f"decoder_jacobian: {tuple(decoder_jacobian.shape)}")
            fig = plot_encoder_decoder_jacobians(encoder_jacobian, decoder_jacobian)
            _emit("encoder_decoder_jacobians", fig)

        # ============================================================
        # 11. Amplification loss
        # ============================================================
        if "amplification" in active_sections:
            print("Computing amplification loss ...")

            def _extract_seqs(x: torch.Tensor, sl: int) -> torch.Tensor:
                B, T, D = x.shape
                seqs = []
                for b in range(B):
                    for t0 in range(T - sl + 1):
                        seqs.append(x[b, t0:t0 + sl])
                return torch.stack(seqs) if seqs else torch.empty(0, sl, D)

            test_trajs_obs_amp = trajs["test_trajs"].sequence
            test_trajs_full_amp = trajs.get("test_trajs_full", trajs["test_trajs"]).sequence
            T_obs = test_trajs_obs_amp.shape[1]

            # Adapt seq_length to available data: need at least max_T+1 steps
            # after slicing, and enough points for n_neighbors.
            seq_length = min(45, T_obs)
            # n_pts = n_seqs * (seq_length - max_T); need n_pts >= n_neighbors
            min_seq_length = n_amp_max_t + 1
            if seq_length < min_seq_length:
                print(
                    f"  Skipping amplification: trajectory length ({T_obs}) too short "
                    f"for max_T={n_amp_max_t}."
                )
            else:
                x_de = _extract_seqs(test_trajs_obs_amp, seq_length)
                x_orig = _extract_seqs(test_trajs_full_amp[:, -T_obs:], seq_length)

                B_amp = x_de.shape[0]
                if B_amp == 0:
                    print("  Skipping amplification: no sequences could be extracted.")
                else:
                    rng_amp = np.random.default_rng(42)
                    n_selected = min(n_amp_trajs, B_amp)
                    idx_amp = rng_amp.choice(B_amp, n_selected, replace=False)

                    # Verify enough data points for k-NN
                    n_pts = n_selected * (seq_length - n_amp_max_t)
                    if n_pts < n_amp_neighbors:
                        print(
                            f"  Skipping amplification: only {n_pts} data points "
                            f"but n_neighbors={n_amp_neighbors}."
                        )
                    else:
                        X_de_s = x_de[torch.from_numpy(idx_amp)].to(device_obj)
                        X_orig_s = x_orig[torch.from_numpy(idx_amp)].to(device_obj)
                        # Each extracted sequence came from a specific
                        # trajectory: window order is (b, t0) so
                        # source_traj_idx = idx // (T_obs - seq_length + 1).
                        if _is_conditioned_model and _test_traj_cond_full is not None:
                            n_per_traj = T_obs - seq_length + 1
                            traj_idx_for_amp = idx_amp // n_per_traj
                            X_amp_c = _test_traj_cond_full[
                                torch.as_tensor(traj_idx_for_amp, device=device_obj)
                            ]
                        else:
                            X_amp_c = None

                        with torch.no_grad():
                            X_latent_amp = (
                                lit_model.encode_trajectory(X_de_s, X_amp_c)
                                if is_latent else X_de_s
                            )
                            amp_true = loss_amplification(
                                X_de_s, X_orig_s[..., [0]],
                                n_neighbors=n_amp_neighbors, max_T=n_amp_max_t, normalize=True,
                            ).item()
                            amp_latent = loss_amplification(
                                X_latent_amp, X_orig_s[..., [0]],
                                n_neighbors=n_amp_neighbors, max_T=n_amp_max_t, normalize=True,
                            ).item()

                        print(f"Amplification loss — True state: {amp_true:.6f}")
                        print(f"Amplification loss — Latent:     {amp_latent:.6f}")

                        fig = plot_amplification(
                            amp_true, amp_latent,
                            n_obs_dims=X_orig_s.shape[-1],
                            n_latent_dims=X_de_s.shape[-1],
                        )
                        _amp_lines = [
                            f"True state (D={X_orig_s.shape[-1]}): {amp_true:.6f}",
                            f"Latent     (D={X_de_s.shape[-1]}):   {amp_latent:.6f}",
                        ]
                        _html_section("Amplification Loss", _amp_lines)
                        _emit("amplification", fig)
                        _summary_lines += ["", "=== Amplification Loss ==="] + _amp_lines

        # ============================================================
        # 12. Tangent-space spectrum (encoder Jacobian × latent velocity)
        # ============================================================
        if "tangent_spectrum" in active_sections and is_latent and hasattr(
            lit_model, "compute_tangent_spectrum"
        ) and _is_conditioned_model:
            # compute_tangent_spectrum's internal vmap+jacrev path doesn't yet
            # accept a per-sample c. Fixing it requires per-sample-c-aware
            # vmap'd Jacobians inside _encoder_jacobian_at — substantial; see
            # the analogous compute_jacobians fix in the lyapunov section.
            # Skip explicitly here rather than crash the section.
            print(
                "Skipping 'tangent_spectrum' for conditioned model: "
                "compute_tangent_spectrum needs c plumbing through "
                "_encoder_jacobian_at's vmap+jacrev (TODO)."
            )
        elif "tangent_spectrum" in active_sections and is_latent and hasattr(
            lit_model, "compute_tangent_spectrum"
        ):
            print("Computing tangent space spectrum ...")
            ts_batch = trajs["test_trajs"].sequence
            # Cap input to 16 trajectories so the per-pair Jacobian computation
            # (vmap'd jacrev/jacfwd over up to ~3000 points) stays bounded.
            if ts_batch.shape[0] > 16:
                ts_batch = ts_batch[:16]
            ts_batch = ts_batch.to(device_obj)
            try:
                ts_result = lit_model.compute_tangent_spectrum(
                    ts_batch, n_samples=512,
                )
                E_np = ts_result["energy"].cpu().numpy()
                p_np = ts_result["spectrum"].cpu().numpy()
                K = len(E_np)
                cum = np.cumsum(p_np)
                # For partial-obs Lorenz the underlying attractor is 3-D; for
                # other systems the user can still read the curve and judge
                # where energy plateaus. Pass 3 as a default reference line.
                fig = plot_tangent_spectrum(
                    E_np, p_np, ts_result["n_pairs"],
                    n_dyn=ts_result["n_dyn"], n_obs=ts_result["n_obs"],
                    expected_intrinsic_dim=3,
                )
                _emit("tangent_spectrum", fig)
                _ts_lines = [
                    f"K (= min(n_dyn, n_obs)): {K}",
                    f"n_pairs: {ts_result['n_pairs']}",
                    "Top-5 spectrum: " + ", ".join(
                        f"{v:.4f}" for v in p_np[:min(5, K)]
                    ),
                    f"Cumulative @ dim 3: {float(cum[2]):.4f}" if K >= 3 else "",
                    f"Cumulative @ dim 5: {float(cum[4]):.4f}" if K >= 5 else "",
                ]
                _ts_lines = [ln for ln in _ts_lines if ln]
                _html_section("Tangent Space Spectrum", _ts_lines)
                _summary_lines += ["", "=== Tangent Space Spectrum ==="] + _ts_lines
            except Exception as exc:
                print(f"  tangent_spectrum failed: {exc}")

        # ============================================================
        # Final: consolidated summary section (HTML only — always last)
        # ============================================================
        _html_section("Full Metrics Summary", _summary_lines)

    finally:
        if pdf_handle is not None:
            pdf_handle.close()
            print("PDF saved.")
        if "html" in output and html_path is not None and _html_parts:
            _run_title = f"Analytics Report — {run_id}"
            _body = "\n".join(_html_parts)
            _html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{_run_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 24px 32px; background: #fff; color: #222; }}
    h1   {{ font-size: 1.6em; margin-bottom: 0.15em; }}
    h2   {{ font-size: 1.15em; color: #444; border-bottom: 2px solid #e0e0e0;
            padding-bottom: 4px; margin-top: 2.2em; }}
    pre  {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
            padding: 12px 16px; font-size: 12.5px; line-height: 1.6;
            overflow-x: auto; white-space: pre-wrap; word-break: break-word; }}
    .figure {{ margin: 18px 0; text-align: center; }}
    img  {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }}
    .subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 2em; }}
  </style>
</head>
<body>
<h1>{_run_title}</h1>
<p class="subtitle">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp;
  Project: {wandb_project} &nbsp;|&nbsp;
  Group: {wandb_group or "(none)"}
</p>
{_body}
</body>
</html>"""
            html_path.write_text(_html_doc, encoding="utf-8")
            print(f"HTML report saved to: {html_path}")

    if return_model:
        result = figures if "return" in output else None
        return result, lit_model, run_id
    if "return" in output:
        return figures
    return None
