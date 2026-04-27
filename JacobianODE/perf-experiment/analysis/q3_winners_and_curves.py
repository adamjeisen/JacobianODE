"""Q3: Are sweep winners separable early?
Q4: Plot loss curves across eras
Q5: Loss-term contribution audit

For each sweep group, find the eventual winner (by mean val loss / traj val loss)
and check at which epoch we could already pick it from training-loss curves.

Also plot val loss curves on log y for both eras + audit which losses are
"earning their cost" (non-flat).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "cache"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def per_epoch(hist: pd.DataFrame, run_id: str) -> pd.DataFrame:
    g = hist[hist["run_id"] == run_id].sort_values("_step").copy()
    if g.empty:
        return g
    g["epoch"] = g["epoch"].ffill()
    keep = [c for c in g.columns if c not in ("run_id", "era", "epoch")]
    out = g.groupby("epoch")[keep].agg(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)
    return out.reset_index().sort_values("epoch")


def collect_curves(slug: str, metric: str) -> dict[str, pd.Series]:
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    out = {}
    for run_id in cfg["run_id"]:
        ep = per_epoch(hist, run_id)
        if ep.empty or metric not in ep.columns:
            continue
        s = pd.to_numeric(ep[metric], errors="coerce")
        idx = ep["epoch"].astype(int)
        s.index = idx
        out[run_id] = s.dropna()
    return out


def plot_q4_curves():
    """Plot trajectory val loss for lorenz_current vs wmtask_vanilla_old."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    metric = "trajectory val_loss"
    for ax, slug in zip(axes, ["lorenz_current", "lorenz_current_obsnoise001", "wmtask_vanilla_old"]):
        curves = collect_curves(slug, metric)
        if not curves:
            continue
        for rid, s in curves.items():
            ax.plot(s.index, s.values, color="C0", alpha=0.25)
        # Median across runs
        common = sorted(set().union(*(s.index for s in curves.values())))
        med = []
        for e in common:
            vals = [s.get(e) for s in curves.values() if e in s.index]
            vals = [v for v in vals if pd.notna(v)]
            med.append(np.median(vals) if vals else np.nan)
        ax.plot(common, med, "k", linewidth=2, label="median")
        ax.set_yscale("log")
        ax.set_title(f"{slug}\nn={len(curves)}, metric={metric}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric + " (log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = FIGS / "q4_val_curves_by_era.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out.name}")


def plot_q5_loss_terms(slug: str = "lorenz_current"):
    """Plot all train loss terms on the same axis (log y) for a single group."""
    metrics = [
        "train/recon_loss",
        "train/latent_pred_loss",
        "train/loop_closure_loss",
        "train/trajectory_loss",
        "train/jac_norm",
        "train/l1_norm",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, m in zip(axes.flat, metrics):
        curves = collect_curves(slug, m)
        if not curves:
            ax.set_title(f"{m} (no data)")
            continue
        for rid, s in curves.items():
            # plot positive values only (log scale)
            v = s[s > 0]
            ax.plot(v.index, v.values, color="C0", alpha=0.25)
        # median (positive only)
        common = sorted(set().union(*(s.index for s in curves.values())))
        med = []
        for e in common:
            vals = [s.get(e) for s in curves.values() if e in s.index]
            vals = [v for v in vals if pd.notna(v) and v > 0]
            med.append(np.median(vals) if vals else np.nan)
        ax.plot(common, med, "k", linewidth=2, label="median")
        ax.set_yscale("log")
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"{slug}: train loss terms (log y, n_runs varies)")
    fig.tight_layout()
    out = FIGS / f"q5_loss_terms_{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out.name}")


def winner_separability(slug: str):
    """For the sweep group, find each run's final best val loss and compute
    Spearman correlation between val loss at epoch K and final-best val loss,
    across runs, sweeping K. The earliest K where corr is high tells us when
    we can already predict the winner."""
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")

    # Build (run_id → (best_so_far series, final_best))
    series = {}
    for run_id in cfg["run_id"]:
        ep = per_epoch(hist, run_id)
        if ep.empty or "trajectory val_loss" not in ep.columns:
            continue
        tv = pd.to_numeric(ep["trajectory val_loss"], errors="coerce")
        idx = ep["epoch"].astype(int)
        tv.index = idx
        tv = tv.dropna()
        if tv.empty:
            continue
        bsf = tv.cummin()
        series[run_id] = (bsf, float(bsf.iloc[-1]))

    if not series:
        return pd.DataFrame()

    # Final-best per run
    final_best = pd.Series({rid: f for rid, (_, f) in series.items()})
    # For each epoch K, gather val_at_K per run (where defined)
    all_epochs = sorted(set().union(*(s.index for s, _ in series.values())))
    out = []
    for K in all_epochs:
        vals = {}
        for rid, (bsf, _) in series.items():
            if K in bsf.index:
                vals[rid] = float(bsf.loc[K])
        if len(vals) < 5:
            continue
        v = pd.Series(vals)
        common = v.index.intersection(final_best.index)
        if len(common) < 5:
            continue
        rho = v.loc[common].rank().corr(final_best.loc[common].rank(), method="pearson")
        # Top-3 hit-rate: of the eventual top-3 runs, how many are in top-3 by val@K?
        top3_final = set(final_best.nsmallest(3).index)
        top3_at_K = set(v.loc[common].nsmallest(3).index)
        out.append({
            "epoch": K,
            "n_runs_with_data": len(common),
            "spearman_to_final": rho,
            "top3_overlap": len(top3_final & top3_at_K),
        })
    return pd.DataFrame(out)


def main():
    out_lines = ["# Q3 / Q4 / Q5 findings\n"]

    # Q4: plot
    plot_q4_curves()

    # Q5: plot loss terms for current-era group
    plot_q5_loss_terms("lorenz_current")
    plot_q5_loss_terms("lorenz_current_obsnoise001")

    # Q3: winner separability
    out_lines.append("\n## Q3: When can we identify the eventual winner?\n")
    for slug in ["lorenz_current", "lorenz_current_obsnoise001", "wmtask_vanilla_old"]:
        df = winner_separability(slug)
        if df.empty:
            continue
        df.to_csv(Path(__file__).parent / f"q3_winner_sep_{slug}.csv", index=False)
        out_lines.append(f"\n### {slug}\n")
        # Show how rank-correlation evolves
        # First epoch where spearman_to_final >= 0.7 and >= 0.9
        first_07 = df[df["spearman_to_final"] >= 0.7].head(1)
        first_09 = df[df["spearman_to_final"] >= 0.9].head(1)
        last_e = int(df["epoch"].max())
        out_lines.append(f"- max epoch in series: {last_e}")
        if not first_07.empty:
            e = int(first_07["epoch"].iloc[0])
            out_lines.append(f"- spearman ≥ 0.7 first reached at epoch {e} ({e/last_e*100:.0f}% of runtime)")
        if not first_09.empty:
            e = int(first_09["epoch"].iloc[0])
            out_lines.append(f"- spearman ≥ 0.9 first reached at epoch {e} ({e/last_e*100:.0f}% of runtime)")
        # Summary table at quartiles
        out_lines.append("```")
        out_lines.append(f"{'epoch':>6s} {'n':>4s} {'spearman':>10s} {'top3_overlap':>13s}")
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            i = int((len(df) - 1) * q)
            r = df.iloc[i]
            out_lines.append(
                f"{int(r['epoch']):>6d} {int(r['n_runs_with_data']):>4d} "
                f"{r['spearman_to_final']:>10.3f} {int(r['top3_overlap']):>13d}/3"
            )
        out_lines.append("```\n")

    summary = Path(__file__).parent / "q3q4q5_summary.md"
    summary.write_text("\n".join(out_lines))
    print(f"wrote {summary.name}")


if __name__ == "__main__":
    main()
