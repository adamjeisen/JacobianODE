"""Q1 + Q2: For each run in the lorenz_current group, identify:
  - epoch at which each train-loss term reaches plateau (within X% of run-final)
  - epoch with best trajectory val_loss (and best mean val loss)
  - walltime fraction spent past best
  - epoch where val loss is within X% of best

Outputs a CSV summary per run + a markdown table snippet.
"""

from __future__ import annotations

import argparse
import json
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
    """Collapse multi-row-per-epoch into one row per epoch with the union
    of train/* and val/* metrics filled in."""
    g = hist[hist["run_id"] == run_id].sort_values("_step").copy()
    if g.empty:
        return g
    # Forward-fill epoch then group: each epoch may have multiple log rows
    # (one for train metrics, one for val). Take last non-null per metric.
    g["epoch"] = g["epoch"].ffill()
    keep = [c for c in g.columns if c not in ("run_id", "era", "epoch")]
    out = g.groupby("epoch")[keep].agg(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)
    out = out.reset_index().sort_values("epoch")
    return out


def saturation_epoch(values: np.ndarray, tol_rel: float = 0.05, decreasing: bool = True) -> float:
    """First epoch where the curve is within tol_rel of its eventual best.

    For a decreasing series (loss): "within tol_rel" means
    `value <= best * (1 + tol_rel)` where `best` is min over the run.
    Returns the epoch index (1-based count) or NaN if never.
    """
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).any():
        return np.nan
    if decreasing:
        best = np.nanmin(v)
        gate = best * (1.0 + tol_rel) if best > 0 else best + abs(best) * tol_rel + 1e-12
        hit = np.where(v <= gate)[0]
    else:
        best = np.nanmax(v)
        gate = best * (1.0 - tol_rel) if best > 0 else best - abs(best) * tol_rel - 1e-12
        hit = np.where(v >= gate)[0]
    if hit.size == 0:
        return np.nan
    return int(hit[0])


def analyse(slug: str, tol_rel: float = 0.05) -> pd.DataFrame:
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    rows = []

    LOSS_KEYS = [
        "train/recon_loss",
        "train/latent_pred_loss",
        "train/loop_closure_loss",
        "train/trajectory_loss",
        "train/total_loss",
        "trajectory val_loss",
        "mean val loss",
    ]

    for run_id in cfg["run_id"]:
        ep = per_epoch(hist, run_id)
        if ep.empty:
            continue
        n_epochs = int(ep["epoch"].max()) + 1
        runtime_total = float(ep["_runtime"].iloc[-1])
        # First-row baseline runtime — early _runtime row may include data prep.
        # Use the diff between epoch 1 and epoch 0 as per-epoch
        per_ep_dt = (ep["_runtime"].diff().dropna().median()
                     if ep["_runtime"].notna().sum() >= 2 else np.nan)

        row = {
            "slug": slug,
            "run_id": run_id,
            "n_epochs": n_epochs,
            "runtime_total_s": runtime_total,
            "per_epoch_s_median": per_ep_dt,
        }
        # Saturation epochs by tol_rel
        for k in LOSS_KEYS:
            if k not in ep.columns:
                row[f"sat@{tol_rel:.2f}_{k}"] = np.nan
                continue
            v = pd.to_numeric(ep[k], errors="coerce").values
            row[f"sat@{tol_rel:.2f}_{k}"] = saturation_epoch(v, tol_rel=tol_rel)

        # Best trajectory val loss epoch + walltime there
        if "trajectory val_loss" in ep.columns and ep["trajectory val_loss"].notna().any():
            tv = pd.to_numeric(ep["trajectory val_loss"], errors="coerce")
            ep["trajectory val_loss"] = tv  # keep coerced for downstream rows
            best_idx = int(tv.idxmin())
            best_epoch = int(ep.loc[best_idx, "epoch"])
            best_runtime = float(ep.loc[best_idx, "_runtime"])
            row["best_traj_val"] = float(tv.min())
            row["best_traj_val_epoch"] = best_epoch
            row["best_traj_val_runtime_s"] = best_runtime
            row["walltime_past_best_s"] = max(0.0, runtime_total - best_runtime)
            row["walltime_past_best_frac"] = (
                row["walltime_past_best_s"] / runtime_total if runtime_total > 0 else np.nan
            )
            # Epoch at which val is within tol_rel of best
            row[f"first_traj_val_within_{tol_rel:.2f}_epoch"] = saturation_epoch(
                tv.values, tol_rel=tol_rel
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=0.05,
                        help="relative tolerance for plateau (default 0.05 = within 5%% of best)")
    args = parser.parse_args()

    out_lines = ["# Q1 / Q2: per-run plateau and post-best walltime\n"]
    for slug in ["lorenz_current", "lorenz_current_obsnoise001", "wmtask_vanilla_old"]:
        df = analyse(slug, tol_rel=args.tol)
        if df.empty:
            print(f"[skip] {slug}: empty")
            continue
        out_path = Path(__file__).parent / f"q1_summary_{slug}.csv"
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path.name}")

        # Group-level summary
        out_lines.append(f"\n## {slug} (n={len(df)})\n")
        cols_to_show = [
            "n_epochs", "runtime_total_s", "per_epoch_s_median",
            "best_traj_val", "best_traj_val_epoch", "best_traj_val_runtime_s",
            "walltime_past_best_s", "walltime_past_best_frac",
            f"first_traj_val_within_{args.tol:.2f}_epoch",
            f"sat@{args.tol:.2f}_train/recon_loss",
            f"sat@{args.tol:.2f}_train/latent_pred_loss",
            f"sat@{args.tol:.2f}_train/loop_closure_loss",
            f"sat@{args.tol:.2f}_train/trajectory_loss",
        ]
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        sub = df[cols_to_show].agg(["median", "mean", "min", "max"]).round(3)
        out_lines.append("```\n" + sub.to_string() + "\n```\n")

        # Show how much walltime is "past best" (median across runs)
        if "walltime_past_best_frac" in df.columns:
            med = df["walltime_past_best_frac"].median()
            out_lines.append(
                f"\n**Median walltime past best traj val:** {med*100:.1f}% "
                f"(n={df['walltime_past_best_frac'].notna().sum()})\n"
            )
        if f"first_traj_val_within_{args.tol:.2f}_epoch" in df.columns:
            col = f"first_traj_val_within_{args.tol:.2f}_epoch"
            med_e = df[col].median()
            med_n = df["n_epochs"].median()
            out_lines.append(
                f"**Median epoch at which traj val is within {args.tol*100:.0f}% of best:** "
                f"{med_e:.0f} of {med_n:.0f} epochs total "
                f"({(med_e / med_n * 100):.1f}% of run length)\n"
            )

    summary_md = Path(__file__).parent / "q1q2_summary.md"
    summary_md.write_text("\n".join(out_lines))
    print(f"wrote {summary_md.name}")


if __name__ == "__main__":
    main()
