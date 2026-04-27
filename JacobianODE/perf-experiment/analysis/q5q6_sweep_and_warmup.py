"""Q5 + Q6: sweep-axis dominance, warmup behaviour, alpha trajectory.

For each sweep group:
  - bin runs by loop_closure_weight (and other swept axes)
  - within each bin: median trajectory val loss curves and per-run
    final val loss
  - check whether the swept axis actually changes ranking dramatically
  - inspect alpha_teacher_forcing trajectory
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "cache"
FIGS = Path(__file__).parent / "figures"


def per_epoch(hist: pd.DataFrame, run_id: str) -> pd.DataFrame:
    g = hist[hist["run_id"] == run_id].sort_values("_step").copy()
    if g.empty:
        return g
    g["epoch"] = g["epoch"].ffill()
    keep = [c for c in g.columns if c not in ("run_id", "era", "epoch")]
    out = g.groupby("epoch")[keep].agg(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)
    return out.reset_index().sort_values("epoch")


def sweep_axis_summary(slug: str, axis_col: str = "training.lightning.loop_closure_weight") -> str:
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    if axis_col not in cfg.columns or cfg[axis_col].isna().all():
        return f"  no values for {axis_col} in {slug}\n"

    # final best val loss per run (recompute from history)
    finals = []
    for run_id in cfg["run_id"]:
        ep = per_epoch(hist, run_id)
        if ep.empty or "trajectory val_loss" not in ep.columns:
            continue
        tv = pd.to_numeric(ep["trajectory val_loss"], errors="coerce")
        if not tv.notna().any():
            continue
        finals.append({"run_id": run_id, "final_best": float(tv.min()),
                       "final_runtime_s": float(pd.to_numeric(ep["_runtime"], errors="coerce").iloc[-1])})
    fin = pd.DataFrame(finals)
    merged = fin.merge(cfg[["run_id", axis_col, "data.flow.random_state"]], on="run_id")
    merged["axis_val"] = pd.to_numeric(merged[axis_col], errors="coerce")

    out = []
    out.append(f"\n#### {slug} — sweep axis {axis_col}\n")
    out.append("```")
    out.append(merged[["axis_val", "final_best", "final_runtime_s",
                       "data.flow.random_state"]]
               .sort_values(["axis_val", "data.flow.random_state"]).to_string(index=False))
    out.append("```\n")

    # Group by axis: median + std of final best
    grouped = merged.groupby("axis_val")["final_best"].agg(["count", "median", "min", "max"])
    out.append(f"\n**Final best val by {axis_col}:**\n")
    out.append("```")
    out.append(grouped.round(5).to_string())
    out.append("```\n")

    # Spread within axis group vs spread across axis groups: does the sweep matter?
    if grouped["median"].notna().any():
        within_spread = merged.groupby("axis_val")["final_best"].std().median()
        between_spread = grouped["median"].std()
        out.append(
            f"\n**Within-axis-bin std of final_best (median across bins):** "
            f"{within_spread:.5f}\n"
        )
        out.append(
            f"**Between-axis-bin std of bin-medians:** {between_spread:.5f}\n"
        )
        if within_spread > 0:
            out.append(
                f"**Ratio between/within:** {between_spread/within_spread:.2f}× "
                f"({'sweep DOES matter' if between_spread > 2*within_spread else 'sweep marginal'})\n"
            )
    return "\n".join(out)


def alpha_traj_summary(slug: str) -> str:
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    metric = "train/alpha_teacher_forcing"
    if metric not in hist.columns:
        return f"\n#### {slug} — alpha not logged\n"
    out = [f"\n#### {slug} — alpha_teacher_forcing trajectory\n"]
    finals = []
    for run_id in cfg["run_id"]:
        ep = per_epoch(hist, run_id)
        if ep.empty or metric not in ep.columns:
            continue
        a = pd.to_numeric(ep[metric], errors="coerce").dropna()
        if a.empty:
            continue
        finals.append({"run_id": run_id, "alpha_start": float(a.iloc[0]),
                       "alpha_final": float(a.iloc[-1]),
                       "alpha_min": float(a.min()),
                       "epoch_at_min": int(a.idxmin())})
    df = pd.DataFrame(finals)
    out.append(f"n_runs={len(df)}, alpha_start median={df['alpha_start'].median():.3f}, "
               f"alpha_final median={df['alpha_final'].median():.4f}, "
               f"alpha_min median={df['alpha_min'].median():.4f}, "
               f"epoch_at_min median={df['epoch_at_min'].median():.0f}\n")
    return "\n".join(out)


def warmup_phase_summary(slug: str) -> str:
    """Inspect what changes during encoder_warmup_epochs vs after."""
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    out = [f"\n#### {slug} — encoder/dynamics warmup config\n"]
    enc_w = cfg.get("training.lightning.encoder_warmup_epochs", pd.Series(dtype=float)).dropna().unique()
    dyn_w = cfg.get("training.lightning.dynamics_warmup_epochs", pd.Series(dtype=float)).dropna().unique()
    out.append(f"encoder_warmup_epochs unique values in cfg: {enc_w}\n")
    out.append(f"dynamics_warmup_epochs unique values in cfg: {dyn_w}\n")
    return "\n".join(out)


def es_config_summary(slug: str) -> str:
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    out = [f"\n#### {slug} — early-stopping config\n"]
    cols = [
        "training.early_stopping.early_stopping_patience",
        "training.early_stopping.early_stopping_mode",
        "training.early_stopping.percent_thresh",
        "training.trainer_params.max_epochs",
        "training.trainer_params.max_steps",
        "training.trainer_params.limit_train_batches",
        "training.trainer_params.accumulate_grad_batches",
    ]
    out.append("```")
    for c in cols:
        if c not in cfg.columns:
            continue
        vals = cfg[c].dropna().unique()
        out.append(f"  {c}: {list(vals)}")
    out.append("```\n")
    return "\n".join(out)


def main():
    out_lines = ["# Q5 / Q6: sweep dominance, warmup, ES, alpha traj\n"]
    for slug in ["lorenz_current", "lorenz_current_obsnoise001", "wmtask_vanilla_old"]:
        out_lines.append(es_config_summary(slug))
        out_lines.append(warmup_phase_summary(slug))
        out_lines.append(sweep_axis_summary(slug))
        out_lines.append(alpha_traj_summary(slug))
    summary = Path(__file__).parent / "q5q6_summary.md"
    summary.write_text("\n".join(out_lines))
    print(f"wrote {summary.name}")


if __name__ == "__main__":
    main()
