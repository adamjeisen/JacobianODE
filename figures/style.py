"""Paper-wide plot style + helpers.

Use ``apply_style()`` at the top of any plot script. Save figures with
``save_fig(fig, name, out_dir)`` to get consistent PNG + PDF output.

Two presets are exposed:

* ``apply_style("paper")``  – the default. ~7.5pt-ish text, 1-column figure
  widths, sans-serif. Suitable for NeurIPS / Nature SI.
* ``apply_style("slides")`` – larger fonts for talks. Same colors.

Color palette lives in :data:`COLORS` and is keyed by a *role* string
(e.g. "pred_batch", "empirical"), not by a series index — so plots stay
visually consistent across panels even when the data order changes.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Paper widths (inches), single- and two-column.
COL_WIDTH_PAPER = 3.5
TWO_COL_WIDTH_PAPER = 7.0


# Role → hex. Borrowed from ChaoticConsciousness Fig 4 + S2: pink for the
# model output (JacobianODE) and charcoal for ground-truth references.
# All three Lyapunov series use these two roles — the two JacobianODE
# series (batch+burnin, full trajectory) share the same pink and are
# differentiated by linestyle in plot.py.
JACODE_PINK = "#e12d8a"
GROUND_CHARCOAL = "#3a3a3a"

COLORS = {
    # Lyapunov-spectrum series
    "pred_batch": JACODE_PINK,
    "pred_full":  JACODE_PINK,
    "empirical":  GROUND_CHARCOAL,
    # Generic semantic
    "model":      JACODE_PINK,        # JacobianODE / model outputs
    "true":       GROUND_CHARCOAL,    # ground-truth references
    "baseline":   "#bbbbbb",          # light gray (persistence-style)
    # Sweep-axis encoding (chill pastel pair, also from ChaoticConsciousness)
    "nt99":       "#809BCE",          # soft blue
    "nt95":       "#e12d8a",          # JacobianODE pink
}


def apply_style(preset: str = "paper") -> None:
    """Set Matplotlib rcParams for the given preset.

    Idempotent; safe to call multiple times in a notebook.
    """
    if preset == "paper":
        font_sz = 8
        title_sz = 9
        line_w = 1.0
        marker_sz = 4.0
    elif preset == "slides":
        font_sz = 14
        title_sz = 16
        line_w = 1.5
        marker_sz = 7.0
    else:
        raise ValueError(f"unknown preset {preset!r}; use 'paper' or 'slides'")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": font_sz,
        "axes.titlesize": title_sz,
        "axes.labelsize": font_sz,
        "xtick.labelsize": font_sz - 1,
        "ytick.labelsize": font_sz - 1,
        "legend.fontsize": font_sz - 1,
        "axes.linewidth": 0.8,
        "lines.linewidth": line_w,
        "lines.markersize": marker_sz,
        "lines.markeredgewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "pdf.fonttype": 42,        # embed TrueType (vector-editable in Illustrator)
        "ps.fonttype": 42,
    })


def save_fig(
    fig: Figure,
    name: str,
    out_dir: Path | str,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 200,
) -> list[Path]:
    """Save ``fig`` as ``name.<fmt>`` in ``out_dir`` for each requested format.

    Creates the directory if it doesn't exist. Returns the list of paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, format=fmt)
        saved.append(path)
    return saved


def panel_label(
    ax: plt.Axes, label: str, *,
    x: float = -0.12, y: float = 1.05, fontsize: int = 10,
) -> None:
    """Place a panel label (e.g. ``"A"``) in the top-left of ``ax``."""
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        ha="right", va="bottom",
    )


def sem(arr, axis: int = 0) -> "np.ndarray":
    """Standard error of the mean along ``axis``. Handles NaN-safe by default."""
    import numpy as np
    arr = np.asarray(arr)
    n = np.sum(~np.isnan(arr), axis=axis)
    n = np.where(n <= 1, np.nan, n)  # SEM undefined for n<=1
    std = np.nanstd(arr, axis=axis, ddof=1)
    return std / np.sqrt(n)
