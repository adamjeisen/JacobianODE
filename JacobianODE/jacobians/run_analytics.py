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
from typing import Any

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
    if n_target_dims is not None:
        return z[..., n_target_dims:]
    return None


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

def plot_sweep_overview(
    all_diagnostics: list,
    result,
    sweep_lambdas: list[float],
    n_latent: int,
) -> plt.Figure:
    """4-panel bar chart of sweep selection criteria."""
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
    axes[0, 1].axhline(y=np.sqrt(n_latent), color="k", linestyle="--", lw=1,
                       label=f"sqrt(n_latent)={np.sqrt(n_latent):.2f}")
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
        title += f"\nInverse consistency MSE = {inv_err:.2e},  null RMS = {null_rms:.2e}"
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


def plot_lyapunov_spectrum(
    pred_np: np.ndarray,
    pred_std_np: np.ndarray,
    emp_np: np.ndarray | None = None,
    emp_std_np: np.ndarray | None = None,
    true_lyapunov: list[float] | None = None,
    loop_closure_weight: float | None = None,
    full_lyap_np: np.ndarray | None = None,
    full_lyap_std_np: np.ndarray | None = None,
) -> plt.Figure:
    """Bar chart of predicted vs empirical vs literature Lyapunov spectrum.

    ``pred_np`` / ``pred_std_np`` are the batch+burn-in estimates.
    ``full_lyap_np`` / ``full_lyap_std_np`` are the full-trajectory estimates.
    """
    n_lyaps = len(pred_np)
    if full_lyap_np is not None and len(full_lyap_np) > 0:
        n_lyaps = min(n_lyaps, len(full_lyap_np))
    if emp_np is not None and len(emp_np) > 0:
        n_lyaps = min(n_lyaps, len(emp_np))
    if true_lyapunov is not None:
        n_lyaps = min(n_lyaps, 10)  # cap for readability

    x_idx = np.arange(n_lyaps)

    # Dynamic bar width so bars don't overlap regardless of how many groups we have
    n_groups = sum([
        1,  # batch+burn-in always present
        full_lyap_np is not None and len(full_lyap_np) > 0,
        emp_np is not None and len(emp_np) > 0,
        true_lyapunov is not None,
    ])
    bar_w = min(0.8 / n_groups, 0.25)
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * bar_w

    bar_idx = 0
    fig, ax = plt.subplots(figsize=(10, 4))
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
    if loop_closure_weight is not None:
        title += f" (loop_closure_weight={loop_closure_weight})"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


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
    axes[0].set_title(f"Latents\nmean D_KY = {mean_ky_latent:.3f}")

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
    fig_z.suptitle(
        f"Latent Space: {title_suffix} (nMSE={nmse_val:.4f})", fontsize=14, y=1.02
    )
    plt.tight_layout()

    # Observation space figure
    obs_pred_phys = obs_pred * sigma + mu
    obs_true_phys = obs_true * sigma + mu
    D_obs = obs_pred.shape[-1]
    dim_labels = [chr(ord("x") + i) for i in range(26)]
    t_pred = np.arange(obs_pred.shape[0])

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
        f"Observation Space: {title_suffix} (MASE={window_mase:.4f})", fontsize=14, y=1.02
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
    """Two-panel plot: obs dim 0 and latent dim 0 over a long free-running run."""
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
    output: str | list[str] = "show",
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
) -> dict[str, plt.Figure] | None:
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
    output : str or list of str
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

    Returns
    -------
    dict[str, Figure] or None
        If ``"return"`` is in *output*, returns ``{section_name: figure}``.
        Otherwise returns ``None``.
    """
    # ------------------------------------------------------------------ setup
    if isinstance(output, str):
        output = [output]

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
            verbose=True,
        )
        print(f"Auto-selected run_id: {run_id}")

        # Populate sweep precomputed args so sweep_overview doesn't re-run
        if sweep_diagnostics is None:
            sweep_diagnostics = _auto_sweep_result.all_diagnostics
        if sweep_result is None:
            sweep_result = _auto_sweep_result.selection
        if sweep_lambdas is None:
            sweep_lambdas = _auto_discovered.lambdas

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

    # -------------------------------------------------------- config extraction
    mu = float(cfg.data.postprocessing.mu) if np.isscalar(cfg.data.postprocessing.mu) else np.array(cfg.data.postprocessing.mu)
    sigma = float(cfg.data.postprocessing.sigma)
    n_latent = OmegaConf.select(cfg, "model.encoder.n_latent", default=None)
    n_dims = values.shape[-1]
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
    jac_window_len = traj_init_steps + lit_model.prediction_steps
    jac_window_stride = lit_model.jac_window_stride

    test_trajs_full = trajs.get("test_trajs_full", trajs["test_trajs"]).sequence

    # Decoder output dims (may be < n_input for decode_only_recent models)
    if hasattr(lit_model.encoder, "decoder"):
        decoder_n_out = lit_model.encoder.decoder.n_output
    else:
        decoder_n_out = lit_model.encoder.n_latent

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
            f"  prediction_steps: {lit_model.prediction_steps}",
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
        _summary_lines += _header_lines
        _html_section("Run Summary", _header_lines)

        # ============================================================
        # 1. Sweep overview
        # ============================================================
        if "sweep_overview" in active_sections:
            if sweep_diagnostics is not None and sweep_result is not None and sweep_lambdas is not None:
                fig = plot_sweep_overview(sweep_diagnostics, sweep_result, sweep_lambdas, n_latent)
                _emit("sweep_overview", fig)
            else:
                print("Skipping 'sweep_overview': provide wandb_group or precomputed sweep_diagnostics.")

        # ============================================================
        # 2. Reconstruction
        # ============================================================
        if "reconstruction" in active_sections:
            print("Computing reconstruction ...")
            test_trajs_obs = trajs["test_trajs"].sequence
            test_trajs_full_aligned = test_trajs_full[:, time_offset:]

            with torch.no_grad():
                # compute recon nMSE on first 16 test samples
                batch = test_dl.dataset.sequence[:16].to(device_obj)
                z_enc = lit_model.encode_trajectory(batch)
                recon = lit_model.decode_trajectory(z_enc)
                margin = getattr(lit_model.encoder, "context_margin", 0)
                targets = batch[:, margin:] if margin > 0 else batch
                targets = targets[..., :decoder_n_out]
                recon_mse_val = ((recon - targets) ** 2).mean().item()
                recon_nmse_val = recon_mse_val / targets.var().item()

                inv_err_val: float | None = None
                null_rms_val: float | None = None
                if is_coupling:
                    x_rt = lit_model.decode_trajectory(z_enc)
                    inv_err_val = float(F.mse_loss(x_rt, batch).item())
                    z_n = _z_null(z_enc, n_target_dims)
                    null_rms_val = float(z_n.pow(2).mean().sqrt().item()) if z_n is not None else 0.0

                # Build decoded trajectory for first test trajectory
                latent_full = lit_model.encode_trajectory(test_trajs_obs.to(device_obj))
                if is_coupling:
                    latent_dyn = _z_dyn(latent_full, n_target_dims)
                    latent_padded = lit_model._pad_to_full_dim(latent_dyn)
                else:
                    latent_padded = latent_full
                decoded_full = lit_model.decode_trajectory(latent_padded)

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
            if is_coupling and inv_err_val is not None:
                _recon_lines.append(f"Inverse consistency MSE: {inv_err_val:.2e}")
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
            for i, batch in enumerate(tqdm(rand_dl, desc="MASE batches", total=n_mase_batches)):
                if i >= n_mase_batches:
                    break
                batch = batch.to(device_obj)
                with torch.no_grad():
                    ret_f = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=1, return_decoded=True)
                    ret_r = lit_model.trajectory_model_step(batch, alpha_teacher_forcing=0, return_decoded=True)
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
        if "latent_utilization" in active_sections:
            print("Computing latent utilization ...")
            traj_key = "train_trajs"
            trajs_obs_lat = trajs[traj_key].sequence
            all_latents = []
            with torch.no_grad():
                for i in range(0, trajs_obs_lat.shape[0], 8):
                    x = torch.as_tensor(trajs_obs_lat[i:i + 8]).float().to(device_obj)
                    z = lit_model.encode_trajectory(x)
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
            if is_coupling:
                Z_null_arr = _z_null(torch.from_numpy(Z), n_target_dims)
                if Z_null_arr is not None:
                    null_rms_per_t = np.sqrt((Z_null_arr.numpy() ** 2).mean(axis=(0, 2)))
                    print(f"Null subspace mean RMS: {np.sqrt((Z_null_arr.numpy() ** 2).mean()):.6e}")

            _util_lines = [
                f"Entropy-based utilization: {utilization:.4f}  (1.0 = uniform)",
                f"n_dyn: {n_dyn}",
                "Per-dim fractional variance:",
            ] + [f"  dim {i:2d}: {v / dim_var.sum():.4f}" for i, v in enumerate(dim_var)]
            if is_coupling and null_rms_per_t is not None:
                _util_lines.append(
                    f"Null subspace mean RMS: {np.sqrt((Z_null_arr.numpy() ** 2).mean()):.6e}"
                )
            _html_section("Latent Utilization", _util_lines)
            fig = plot_latent_utilization(dim_var, utilization, n_dyn, null_rms_per_t)
            _emit("latent_utilization", fig)
            _summary_lines += ["", "=== Latent Utilization ==="] + _util_lines

        # ============================================================
        # 5. Lyapunov spectrum
        # ============================================================
        if "lyapunov" in active_sections:
            print("Computing Lyapunov exponents ...")
            with torch.no_grad():
                # --- Full-length trajectory Lyapunov (PLOTTED) ---
                # Uses trajs['test_trajs'].sequence — the actual full-length test
                # trajectories, not the windowed batches from the dataloader.
                # Process one trajectory at a time: encode → Jacobians (T_full, D, D)
                # → Lyapunov (D,) for that single trajectory.
                traj_seq_full = torch.as_tensor(
                    trajs["test_trajs"].sequence
                ).float().to(device_obj)
                z_seq_full = lit_model.encode_trajectory(traj_seq_full)
                z_for_jac_full = _z_dyn(z_seq_full, n_target_dims)
                n_full_trajs = z_for_jac_full.shape[0]
                print(f"  Computing full-trajectory Lyapunov ({n_full_trajs} test trajs, "
                      f"T={traj_seq_full.shape[1]}) ...")
                _lyap_full_list = []
                for _i in range(n_full_trajs):
                    _jacs_i = lit_model.compute_jacobians(z_for_jac_full[_i:_i + 1])[0]
                    _le_i = LitLatentJacobianODE.compute_lyapunov_exponents(_jacs_i.cpu(), dt)
                    _lyap_full_list.append(_le_i)
                    if _i < 3:
                        print(f"    Traj {_i}: {_le_i.numpy()}")
                all_pred_lyap_full = torch.stack(_lyap_full_list)
                _state["all_pred_lyap_full"] = all_pred_lyap_full

                # --- Batch + burn-in Lyapunov (128 sampled windowed trajs) ---
                # Uses test_dl.dataset.sequence — the windowed/batched sequences.
                traj_batched_t = torch.as_tensor(
                    test_dl.dataset.sequence
                ).float().to(device_obj)
                z_batched_t = lit_model.encode_trajectory(traj_batched_t)
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

            # Empirical from analytical Jacobian (if eq available)
            emp_np: np.ndarray | None = None
            emp_std_np: np.ndarray | None = None
            if eq is not None:
                mu_val = cfg.data.postprocessing.mu
                sigma_norm = cfg.data.postprocessing.sigma
                if "test_trajs_full" in trajs:
                    traj_full_np = trajs["test_trajs_full"].sequence
                else:
                    traj_full_np = trajs["test_trajs"].sequence
                traj_raw = np.asarray(traj_full_np) * sigma_norm + mu_val

                n_test_t = traj_raw.shape[0]
                all_emp_lyap = []
                for i in range(n_test_t):
                    traj_i = traj_raw[i]
                    if hasattr(eq, "model"):
                        traj_i = torch.as_tensor(traj_i).float()
                    jacs_np = eq.jac(traj_i, t=0)
                    jacs_t = torch.as_tensor(jacs_np).float()
                    le_i = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_t, dt)
                    all_emp_lyap.append(le_i)
                all_emp_lyap_t = torch.stack(all_emp_lyap).cpu()
                _state["all_emp_lyap_t"] = all_emp_lyap_t
                emp_np = all_emp_lyap_t.mean(dim=0).numpy()
                emp_std_np = all_emp_lyap_t.std(dim=0).numpy()
                print("Empirical Lyapunov exponents (mean ± std):")
                for i, (le, std) in enumerate(zip(emp_np, emp_std_np)):
                    print(f"  λ_{i+1} = {le:+.4f} ± {std:.4f}")

            _state["all_pred_lyap"] = all_pred_lyap.cpu()

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

            fig = plot_lyapunov_spectrum(
                pred_np, pred_std_np, emp_np, emp_std_np,
                true_lyapunov=true_lyapunov, loop_closure_weight=best_lambda,
                full_lyap_np=pred_full_np, full_lyap_std_np=pred_full_std_np,
            )
            _emit("lyapunov", fig)

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
                print("Computing Lyapunov exponents for KY dimension (full-length, per-traj) ...")
                with torch.no_grad():
                    traj_full_t = torch.as_tensor(test_dl.dataset.sequence).float().to(device_obj)
                    z_full_t = lit_model.encode_trajectory(traj_full_t)
                    z_for_jac_ky = _z_dyn(z_full_t, n_target_dims)
                    _lyap_ky_list = []
                    for _i in range(z_for_jac_ky.shape[0]):
                        _jacs_i = lit_model.compute_jacobians(z_for_jac_ky[_i:_i + 1])[0]
                        _lyap_ky_list.append(
                            LitLatentJacobianODE.compute_lyapunov_exponents(_jacs_i.cpu(), dt)
                        )
                    all_pred_lyap_ky = torch.stack(_lyap_ky_list)

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
                        _all_lat.append(_z_dyn(lit_model.encode_trajectory(_x), n_target_dims).cpu())
                _Z = torch.cat(_all_lat, dim=0).numpy()
                _state["Z_flat"] = _Z.reshape(-1, _Z.shape[-1])

            _Z_flat_pca = _state["Z_flat"]
            _X_true_flat = np.asarray(
                trajs.get("test_trajs_full", trajs["test_trajs"]).sequence
            ).reshape(-1, test_trajs_full.shape[-1])
            _mean_ky_true_pca = float(ky_emp_np_ky.mean()) if ky_emp_np_ky is not None else None

            fig_pca = plot_pca_kaplan_yorke(
                _Z_flat_pca, _X_true_flat,
                mean_ky_latent=float(ky_pred_np.mean()),
                mean_ky_true=_mean_ky_true_pca,
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
                with torch.no_grad():
                    rd = lit_model.trajectory_model_step(
                        traj_i, alpha_teacher_forcing=0.0, obs_noise_scale=0, return_decoded=True
                    )
                    z_true_i = lit_model.encode_trajectory(traj_i)

                dec_pred = rd["decoded"].cpu()
                obs_tgt = rd["targets"].cpu()
                z_pred_i = rd["outputs"].cpu()

                mean_var = obs_tgt.reshape(-1, obs_tgt.shape[-1]).var(dim=0).mean().clamp(min=1e-8)
                for w in range(dec_pred.shape[0]):
                    w_mse = (dec_pred[w] - obs_tgt[w]).pow(2).mean()
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

                fig_z, fig_obs = plot_prediction_detail(
                    z_pred_med, z_true_med, obs_pred_med, obs_true_med,
                    sigma=sigma, mu=mu,
                    traj_init_steps=traj_init_steps,
                    nmse_val=all_per_window_nmse_arr[median_idx],
                    title_suffix="Median-loss window",
                )
                _emit("prediction_detail_latent", fig_z)
                _emit("prediction_detail_obs", fig_obs)

        # ============================================================
        # 9. Long trajectory
        # ============================================================
        if "long_trajectory" in active_sections:
            print("Computing long trajectory prediction ...")
            test_trajs_obs_lt = trajs["test_trajs"].sequence
            traj_long = torch.as_tensor(test_trajs_obs_lt[[0]]).float().to(device_obj)

            with torch.no_grad():
                rd_long = lit_model.trajectory_model_step(
                    traj_long, alpha_teacher_forcing=0.0, obs_noise_scale=0,
                    return_decoded=True, strided=False,
                )
                encoded_lt = lit_model.encode_trajectory(traj_long)
                z_pred_full_lt = lit_model._pad_to_full_dim(rd_long["outputs"])
                decoded_lt = lit_model.decode_trajectory(z_pred_full_lt)

            traj_true_lt = traj_long[0].cpu().numpy()
            decoded_pred_lt = decoded_lt[0].cpu().numpy()
            latent_true_lt = _z_dyn(encoded_lt[0], n_target_dims).cpu().numpy()
            latent_pred_lt = rd_long["outputs"][0].cpu().numpy()

            fig = plot_long_trajectory(
                traj_true_lt, decoded_pred_lt,
                latent_true_lt, latent_pred_lt,
                traj_init_steps=traj_init_steps,
            )
            _emit("long_trajectory", fig)

        # ============================================================
        # 10. Encoder/decoder Jacobians
        # ============================================================
        if "encoder_decoder_jacobians" in active_sections:
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

            with torch.no_grad():
                if hasattr(_enc, "time_window"):
                    _w = _enc.time_window
                    _windows = _traj_jac.unfold(1, _w, 1).permute(0, 1, 3, 2).reshape(-1, _w, D_obs_jac)
                    n_win = _windows.shape[0]
                    idx_w = rng_jac.choice(n_win, min(N_JAC, n_win), replace=False)
                    _windows_s = _windows[torch.from_numpy(idx_w).to(device_obj)]

                    def _enc_one(x): return _enc.encode(x.unsqueeze(0)).squeeze(0)
                    def _dec_one(z): return _enc.decode(z.unsqueeze(0)).squeeze(0)

                    _z_jac = _enc.encode(_windows_s)
                    encoder_jacobian = vmap(jacrev(_enc_one))(_windows_s)
                    decoder_jacobian = vmap(jacfwd(_dec_one))(_z_jac)

                elif enc_accepts_flat:
                    test_pts = _traj_jac.reshape(-1, D_obs_jac)
                    n_pts = test_pts.shape[0]
                    idx_p = rng_jac.choice(n_pts, min(N_JAC, n_pts), replace=False)
                    x_flat = test_pts[torch.from_numpy(idx_p).to(device_obj)]

                    def _enc_one(x): return _enc.encode(x.unsqueeze(0)).squeeze(0)
                    def _dec_one(z): return _enc.decode(z.unsqueeze(0)).squeeze(0)

                    _z_jac = _enc.encode(x_flat)
                    encoder_jacobian = vmap(jacrev(_enc_one))(x_flat)
                    decoder_jacobian = vmap(jacfwd(_dec_one))(_z_jac)

                else:
                    B_jac = _traj_jac.shape[0]
                    idx_t = rng_jac.choice(B_jac, min(N_JAC, B_jac), replace=False)
                    _traj_s = _traj_jac[torch.from_numpy(idx_t).to(device_obj)]

                    def _enc_traj(x): return lit_model.encode_trajectory(x.unsqueeze(0)).squeeze(0)
                    def _dec_traj(z): return lit_model.decode_trajectory(z.unsqueeze(0)).squeeze(0)

                    _z_jac = lit_model.encode_trajectory(_traj_s)
                    encoder_jacobian = vmap(jacrev(_enc_traj))(_traj_s)
                    decoder_jacobian = vmap(jacfwd(_dec_traj))(_z_jac)

            print(f"encoder_jacobian: {tuple(encoder_jacobian.shape)}")
            print(f"decoder_jacobian: {tuple(decoder_jacobian.shape)}")
            fig = plot_encoder_decoder_jacobians(encoder_jacobian, decoder_jacobian)
            _emit("encoder_decoder_jacobians", fig)

        # ============================================================
        # 11. Amplification loss
        # ============================================================
        if "amplification" in active_sections:
            print("Computing amplification loss ...")
            seq_length = 45

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
            x_de = _extract_seqs(test_trajs_obs_amp, seq_length)
            x_orig = _extract_seqs(test_trajs_full_amp[:, -T_obs:], seq_length)

            rng_amp = np.random.default_rng(42)
            B_amp = x_de.shape[0]
            idx_amp = rng_amp.choice(B_amp, min(n_amp_trajs, B_amp), replace=False)

            X_de_s = x_de[torch.from_numpy(idx_amp)].to(device_obj)
            X_orig_s = x_orig[torch.from_numpy(idx_amp)].to(device_obj)

            with torch.no_grad():
                X_latent_amp = lit_model.encode_trajectory(X_de_s)
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

    if "return" in output:
        return figures
    return None
