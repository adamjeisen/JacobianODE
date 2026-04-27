"""Plot alpha_teacher_forcing trajectory across all runs in lorenz_current."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "cache"
FIGS = Path(__file__).parent / "figures"


def per_epoch(hist, run_id):
    g = hist[hist["run_id"] == run_id].sort_values("_step").copy()
    if g.empty:
        return g
    g["epoch"] = g["epoch"].ffill()
    keep = [c for c in g.columns if c not in ("run_id", "era", "epoch")]
    out = g.groupby("epoch")[keep].agg(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)
    return out.reset_index().sort_values("epoch")


def plot(slug):
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    all_alphas = []
    for rid in cfg["run_id"]:
        ep = per_epoch(hist, rid)
        if ep.empty or "train/alpha_teacher_forcing" not in ep.columns:
            continue
        a = pd.to_numeric(ep["train/alpha_teacher_forcing"], errors="coerce")
        e = ep["epoch"]
        ok = a.notna() & e.notna()
        if not ok.any():
            continue
        ax.plot(e[ok].values, a[ok].values, color="C0", alpha=0.25)
        all_alphas.append(pd.Series(a[ok].values, index=e[ok].astype(int).values))

    common = sorted(set().union(*(s.index for s in all_alphas)))
    med = []
    for x in common:
        vals = [s.get(x) for s in all_alphas if x in s.index]
        vals = [v for v in vals if pd.notna(v)]
        med.append(np.median(vals) if vals else np.nan)
    ax.plot(common, med, "k", linewidth=2.5, label="median")

    # First-derivative analysis
    med_arr = np.array(med)
    nz = ~np.isnan(med_arr)
    e_arr = np.array(common)[nz]
    m_arr = med_arr[nz]
    d_alpha = np.diff(m_arr)
    print(f"  median alpha at epoch 0: {m_arr[0]:.3f}")
    print(f"  median alpha at epoch 50: {m_arr[min(50, len(m_arr)-1)]:.3f}")
    print(f"  median alpha at epoch 100: {m_arr[min(100, len(m_arr)-1)]:.3f}")
    # rate of decay during steepest period
    smoothed = np.convolve(d_alpha, np.ones(5)/5, mode="valid")
    if smoothed.size:
        steep_idx = np.argmin(smoothed)
        steep_rate = float(smoothed[steep_idx])
        steep_ep = int(e_arr[steep_idx])
        print(f"  steepest descent: {steep_rate:.4f}/epoch at epoch {steep_ep}")
    if (m_arr <= 0.06).any():
        first = int(e_arr[(m_arr <= 0.06).argmax()])
        print(f"  first epoch with median α ≤ 0.06: {first}")

    ax.set_xlabel("epoch")
    ax.set_ylabel("train/alpha_teacher_forcing")
    ax.set_title(f"{slug}: alpha_teacher_forcing trajectory across {len(all_alphas)} runs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = FIGS / f"alpha_{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out.name}")


for slug in ["lorenz_current", "lorenz_current_obsnoise001"]:
    print(f"=== {slug} ===")
    plot(slug)
