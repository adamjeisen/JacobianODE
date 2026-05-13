"""Plot the Lyapunov spectrum from a pre-computed npz.

Three series, all rendered as markers (with SEM error bars) connected by
plot lines:

* ``pred_batch_burnin`` — predicted Lyapunov from model-integrated batch
  rollouts (n_batch sampled windows, prepended burn-in, drop initial steps).
* ``pred_full``        — predicted Lyapunov from model Jacobian evaluated
  at TRUE test-trajectory points (per-trajectory).
* ``empirical_full``   — Lyapunov from the analytical / ground-truth
  Jacobian at the same true test-trajectory points (per-trajectory).

SEM is computed across the leading axis (per-batch or per-trajectory). x
axis is the exponent index. Saved as PNG + PDF in ``--out``.

Usage::

    python -m JacobianODE.figures.lyapunov.plot \\
        --npz ~/Documents/paper-figures/<group>/lyapunov/lyapunov.npz \\
        --out ~/Documents/paper-figures/<group>/lyapunov/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# allow `python figures/lyapunov/plot.py ...` invocation as well
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figures.style import COLORS, apply_style, save_fig, sem  # noqa: E402


SERIES_SPEC = [
    # (npz_key, display label, color, marker, linestyle)
    # Both JacobianODE series share the pink (model role); differentiated
    # by linestyle. Empirical (true Jacobian) is the charcoal ground-truth.
    ("pred_batch_burnin", "Predicted (batch + burn-in)", COLORS["pred_batch"], "o", ":"),
    ("pred_full",         "Predicted (full trajectory)", COLORS["pred_full"],  "s", "-"),
    ("empirical_full",    "Empirical (true Jacobian)",   COLORS["empirical"],  "D", "-"),
]


def plot_lyapunov(
    npz_path: Path,
    out_dir: Path,
    *,
    preset: str = "paper",
    fig_name: str = "lyapunov_spectrum",
    width_in: float = 5.0,
    height_in: float = 3.2,
) -> Path:
    """Render the Lyapunov spectrum plot from ``npz_path``.

    Returns the path to the saved PNG.
    """
    apply_style(preset)
    data = np.load(npz_path, allow_pickle=False)

    # n_lyaps from the first series that's present + non-empty.
    n_lyaps = None
    for key, *_ in SERIES_SPEC:
        if key in data and data[key].size:
            n_lyaps = data[key].shape[-1]
            break
    if n_lyaps is None:
        raise ValueError(f"no recognized series in {npz_path}")
    x_idx = np.arange(n_lyaps)

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    for key, label, color, marker, linestyle in SERIES_SPEC:
        if key not in data or not data[key].size:
            continue
        arr = data[key]  # (n_samples, n_lyaps)
        mean = np.nanmean(arr, axis=0)
        err = sem(arr, axis=0)
        ax.errorbar(
            x_idx, mean[:n_lyaps], yerr=err[:n_lyaps],
            marker=marker, linestyle=linestyle, color=color, label=label,
            capsize=2.5, capthick=0.8, elinewidth=0.8,
            zorder=3,
        )

    ax.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.6, zorder=1)
    ax.set_xlabel("Exponent index")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_xticks(x_idx)
    ax.legend(frameon=False, loc="best")
    # No title: filename / output dir encode the (group, run) metadata.

    fig.tight_layout()
    saved = save_fig(fig, fig_name, out_dir)
    plt.close(fig)
    return saved[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--npz", type=Path, required=True, help="Path to the lyapunov.npz from eval.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir; defaults to the npz's parent")
    ap.add_argument("--preset", choices=("paper", "slides"), default="paper")
    ap.add_argument("--name", default="lyapunov_spectrum", help="Output filename stem")
    ap.add_argument("--width", type=float, default=5.0, help="Figure width (in)")
    ap.add_argument("--height", type=float, default=3.2, help="Figure height (in)")
    args = ap.parse_args()

    out_dir = args.out or args.npz.parent
    png = plot_lyapunov(
        args.npz, out_dir,
        preset=args.preset, fig_name=args.name,
        width_in=args.width, height_in=args.height,
    )
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
