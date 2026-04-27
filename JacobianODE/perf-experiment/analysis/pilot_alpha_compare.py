"""Compare a pilot's alpha trajectory to the cached W&B baseline.

Usage:
  python pilot_alpha_compare.py <pilot_perf.json> <label>

Plots: pilot's alpha vs batch_idx alongside W&B median alpha vs batch_idx.
The W&B alpha is one value per epoch (Lightning aggregates), so we render it
at the end of each epoch (batch_idx = epoch * 200 since limit_train_batches=200).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "cache"
FIGS = Path(__file__).parent / "figures"


def load_pilot(perf_json):
    recs = json.load(open(perf_json))
    df = pd.DataFrame(recs)
    df["batch_idx_global"] = np.arange(len(df))  # cumulative batch counter
    return df


def load_wandb_alpha(slug="lorenz_current"):
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    series = []
    for rid in cfg["run_id"]:
        g = hist[hist["run_id"] == rid].sort_values("_step")
        if g.empty:
            continue
        a = pd.to_numeric(g["train/alpha_teacher_forcing"], errors="coerce")
        e = pd.to_numeric(g["epoch"], errors="coerce")
        ok = a.notna() & e.notna()
        if not ok.any():
            continue
        series.append(pd.Series(a[ok].values, index=e[ok].astype(int).values))
    common = sorted(set().union(*(s.index for s in series)))
    med = []
    for x in common:
        vals = [s.get(x) for s in series if x in s.index]
        vals = [v for v in vals if pd.notna(v)]
        med.append(np.median(vals) if vals else np.nan)
    return pd.Series(med, index=common)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    pilot_paths = sys.argv[1:-1] if len(sys.argv) >= 4 else [sys.argv[1]]
    out_label = sys.argv[-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    BATCHES_PER_EPOCH = 200

    # Baseline (W&B median)
    wb = load_wandb_alpha()
    # Convert epoch axis to batch-idx-equivalent
    wb_batch = wb.index * BATCHES_PER_EPOCH
    axes[0].plot(wb.index, wb.values, "k", linewidth=2, label="W&B baseline (interval=5, γ=0.999)")
    axes[1].plot(wb_batch, wb.values, "k", linewidth=2, label="W&B baseline")

    colors = ["C0", "C1", "C2", "C3"]
    for i, p in enumerate(pilot_paths):
        df = load_pilot(p)
        if "alpha_teacher_forcing" not in df.columns:
            print(f"[skip] {p}: no alpha column")
            continue
        a = pd.to_numeric(df["alpha_teacher_forcing"], errors="coerce")
        ok = a.notna()
        # Convert pilot batch_idx_global to epoch
        ep = df.loc[ok, "batch_idx_global"] / BATCHES_PER_EPOCH
        label = Path(p).stem
        axes[0].plot(ep, a[ok], colors[i], alpha=0.9, label=label)
        axes[1].plot(df.loc[ok, "batch_idx_global"], a[ok], colors[i], alpha=0.9, label=label)

    for ax, xlabel in zip(axes, ["epoch", "batch index (cumulative)"]):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("alpha_teacher_forcing")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_xlim(0, 12)  # zoom into pilot range
    axes[1].set_xlim(0, 2400)
    fig.suptitle("Pilot α trajectory vs W&B baseline (lorenz_current)")
    fig.tight_layout()
    out = FIGS / f"pilot_alpha_{out_label}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out.name}")

    # Print key numbers
    print("\n=== Pilot α decay numbers ===")
    print(f"W&B baseline median α at epoch 1: {wb.get(1, np.nan):.3f}")
    print(f"W&B baseline median α at epoch 4: {wb.get(4, np.nan):.3f}")
    print(f"W&B baseline median α at epoch 8: {wb.get(8, np.nan):.3f}")
    for p in pilot_paths:
        df = load_pilot(p)
        a = pd.to_numeric(df.get("alpha_teacher_forcing"), errors="coerce")
        if a is None or not a.notna().any():
            continue
        # Sample at batch 200 (=epoch 1), 800 (=4), 1600 (=8)
        for label, b in [("epoch 1 (~b=200)", 200), ("epoch 4 (~b=800)", 800), ("epoch 8 (~b=1600)", 1600)]:
            if b - 1 < len(df):
                print(f"  pilot {Path(p).name} α at {label}: {a.iloc[b-1]:.3f}")
            else:
                last = a.dropna().iloc[-1]
                print(f"  pilot {Path(p).name} α at end (n={len(df)}): {last:.3f}")


if __name__ == "__main__":
    main()
