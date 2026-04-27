"""Compare Step 1 validation sweep against cached baseline on the 3 quality axes:

(1) trajectory val_loss — per-config, per-bin, top-1
(2) Jacobian quality — Lyapunov spectrum from each report's metrics.json
(3) Sweep winner stability — does the same lc_w win in each n_delays bin?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS = Path("/workspaces/jacobian-analyses/Lorenz_INDpartial_NDInitSweep_autodim_D1_NormTrue__JacobianODE")
BASELINE_GROUP = "lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep"
NEW_GROUP = "lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep__step1_int1_maxep80"


def load_metrics(group):
    p = REPORTS / group / "metrics.json"
    return json.load(open(p))


def per_run_table(metrics):
    rows = []
    for r in metrics["metrics_summary"]["per_run"]:
        sc = r.get("swept_config", {})
        rows.append({
            "run_id": r["run_id"],
            "lc_w": r.get("lc_weight"),
            "n_delays": sc.get("data.train_test_params.delay_embedding_params.n_delays"),
            "best_traj_loss": r.get("best_traj_loss"),
            "best_traj_loss_epoch": r.get("best_traj_loss_epoch"),
            "r2_at_best_tl": r.get("r2_at_best_tl"),
            "best_mase": r.get("best_mase"),
            "n_epochs": r.get("n_epochs"),
            "lc_loss_at_best_tl": r.get("lc_loss_at_best_tl"),
        })
    return pd.DataFrame(rows)


def axis1_trajectory(base_df, new_df):
    out = []
    out.append("\n## Axis 1: trajectory val_loss\n")
    # Per (n_delays, lc_w) bin: median final-best across replicate seeds (3/bin)
    for label, df in [("BASELINE (interval=5, max_ep=200, ~109 ran)", base_df),
                      ("NEW (interval=1, max_ep=80, ~79 ran)", new_df)]:
        g = (df.groupby(["n_delays", "lc_w"])["best_traj_loss"]
             .agg(["count", "median", "min", "max"]).round(5))
        out.append(f"\n### {label}\n```")
        out.append(g.to_string())
        out.append("```")

    # Side-by-side per (n_delays, lc_w)
    base_med = base_df.groupby(["n_delays", "lc_w"])["best_traj_loss"].median().rename("baseline_median")
    new_med = new_df.groupby(["n_delays", "lc_w"])["best_traj_loss"].median().rename("new_median")
    cmp = pd.concat([base_med, new_med], axis=1)
    cmp["delta"] = cmp["new_median"] - cmp["baseline_median"]
    cmp["rel_delta"] = cmp["delta"] / cmp["baseline_median"]
    out.append("\n### Per-bin comparison (median across 3 replicate seeds)\n```")
    out.append(cmp.round(5).to_string())
    out.append("```")
    # Headline: across all 21 bins, what's the median rel_delta?
    out.append(f"\n**Median rel_delta across all 21 bins**: {cmp['rel_delta'].median()*100:+.1f}%")
    out.append(f"**Worst rel_delta**: {cmp['rel_delta'].max()*100:+.1f}% (at {cmp['rel_delta'].idxmax()})")
    out.append(f"**Best rel_delta**: {cmp['rel_delta'].min()*100:+.1f}% (at {cmp['rel_delta'].idxmin()})")
    return "\n".join(out)


def axis2_jacobian(base_metrics, new_metrics):
    """Compare Lyapunov spectra and R² at best traj loss."""
    out = ["\n## Axis 2: Jacobian / Lyapunov quality\n"]

    base_pl = base_metrics["metrics_summary"].get("per_run_lyapunov", [])
    new_pl = new_metrics["metrics_summary"].get("per_run_lyapunov", [])

    out.append(f"\n### Per-run Lyapunov spectrum (chosen run + others)\n")
    out.append("\n**Empirical (true) Lyapunov spectrum:**")
    base_emp = base_metrics["metrics_summary"].get("empirical_lyapunov_spectrum")
    new_emp = new_metrics["metrics_summary"].get("empirical_lyapunov_spectrum")
    out.append(f"  baseline: {base_emp}")
    out.append(f"  new:      {new_emp}")

    out.append("\n**Predicted Lyapunov (chosen run, by best_traj_loss):**")
    chosen_base = base_metrics.get("chosen_run") or {}
    chosen_new = new_metrics.get("chosen_run") or {}
    # Look up Lyapunov for chosen runs
    def find_pl(pl_list, run_id):
        for r in pl_list:
            if r.get("run_id") == run_id:
                return r
        return None

    cb = find_pl(base_pl, chosen_base.get("run_id"))
    cn = find_pl(new_pl, chosen_new.get("run_id"))
    if cb:
        out.append(f"  baseline chosen={chosen_base.get('run_id')}: pred LE = {cb.get('predicted_lyapunov_full', cb.get('predicted_lyapunov'))}")
    if cn:
        out.append(f"  new chosen={chosen_new.get('run_id')}: pred LE = {cn.get('predicted_lyapunov_full', cn.get('predicted_lyapunov'))}")

    # Per-run Lyapunov error (if computed)
    base_err = base_metrics["metrics_summary"].get("per_run_lyapunov_error")
    new_err = new_metrics["metrics_summary"].get("per_run_lyapunov_error")
    out.append(f"\n**per_run_lyapunov_error (RMSE vs true) — baseline:** {base_err}")
    out.append(f"**per_run_lyapunov_error (RMSE vs true) — new:**      {new_err}")

    # Compare R² at best traj loss across runs
    base_df = per_run_table(base_metrics)
    new_df = per_run_table(new_metrics)
    out.append("\n### R² at best trajectory loss (proxy for Jacobian quality)\n```")
    out.append(f"  baseline: median={base_df['r2_at_best_tl'].median():.4f}, min={base_df['r2_at_best_tl'].min():.4f}, max={base_df['r2_at_best_tl'].max():.4f}")
    out.append(f"  new:      median={new_df['r2_at_best_tl'].median():.4f}, min={new_df['r2_at_best_tl'].min():.4f}, max={new_df['r2_at_best_tl'].max():.4f}")
    out.append("```")
    return "\n".join(out)


def axis3_winners(base_df, new_df):
    out = ["\n## Axis 3: Sweep winner stability\n"]
    # Overall top-1 by best_traj_loss
    b1 = base_df.nsmallest(1, "best_traj_loss")
    n1 = new_df.nsmallest(1, "best_traj_loss")
    out.append(f"\n**Baseline top-1**: lc_w={b1['lc_w'].iloc[0]:g}, n_delays={int(b1['n_delays'].iloc[0])}, "
               f"traj_loss={b1['best_traj_loss'].iloc[0]:.5f}")
    out.append(f"**New top-1**:      lc_w={n1['lc_w'].iloc[0]:g}, n_delays={int(n1['n_delays'].iloc[0])}, "
               f"traj_loss={n1['best_traj_loss'].iloc[0]:.5f}")

    # Per n_delays bin: which lc_w wins?
    out.append("\n### Per-n_delays bin winner (lc_w that minimizes median traj_loss)\n```")
    for nd in sorted(base_df["n_delays"].dropna().unique()):
        bm = base_df[base_df["n_delays"] == nd].groupby("lc_w")["best_traj_loss"].median()
        nm = new_df[new_df["n_delays"] == nd].groupby("lc_w")["best_traj_loss"].median()
        b_winner = bm.idxmin()
        n_winner = nm.idxmin()
        match = "✓" if b_winner == n_winner else "✗"
        out.append(f"  n_delays={int(nd)}: baseline winner lc_w={b_winner:g}  vs  new winner lc_w={n_winner:g}  {match}")
    out.append("```")

    # Top-3 overlap
    b3 = set(base_df.nsmallest(3, "best_traj_loss")["run_id"])
    n3 = set(new_df.nsmallest(3, "best_traj_loss")["run_id"])
    # Match by (n_delays, lc_w) since run_ids differ across sweeps
    bk = base_df.nsmallest(3, "best_traj_loss")[["n_delays", "lc_w"]].apply(tuple, axis=1).tolist()
    nk = new_df.nsmallest(3, "best_traj_loss")[["n_delays", "lc_w"]].apply(tuple, axis=1).tolist()
    out.append(f"\n**Top-3 (n_delays, lc_w) matches**:")
    out.append(f"  baseline: {bk}")
    out.append(f"  new:      {nk}")
    overlap = set(bk) & set(nk)
    out.append(f"  overlap: {len(overlap)}/3 — {overlap}")

    # Ranking correlation
    bm_full = base_df.groupby(["n_delays", "lc_w"])["best_traj_loss"].median()
    nm_full = new_df.groupby(["n_delays", "lc_w"])["best_traj_loss"].median()
    common = bm_full.index.intersection(nm_full.index)
    rho = bm_full.loc[common].rank().corr(nm_full.loc[common].rank(), method="pearson")
    out.append(f"\n**Spearman rank correlation across all 21 (n_delays, lc_w) bins**: {rho:.3f}")
    return "\n".join(out)


def axis_walltime(base_df, new_df):
    """Bonus: how much faster did the new sweep run end-to-end?"""
    return ""  # already known from cfg metadata


def main():
    print("Loading metrics.json for both groups...")
    base_metrics = load_metrics(BASELINE_GROUP)
    new_metrics = load_metrics(NEW_GROUP)

    base_df = per_run_table(base_metrics)
    new_df = per_run_table(new_metrics)
    print(f"baseline: {len(base_df)} runs, new: {len(new_df)} runs")

    out = ["# Step 1 validation: NEW (interval=1, max_ep=80) vs BASELINE (interval=5, max_ep=200)\n"]
    out.append(f"baseline group: {BASELINE_GROUP}")
    out.append(f"new group: {NEW_GROUP}\n")

    # Walltime context
    base_ep = base_df["n_epochs"].median()
    new_ep = new_df["n_epochs"].median()
    out.append(f"**Median epochs to ES/end**: baseline={base_ep:.0f}, new={new_ep:.0f}")
    out.append(f"  → with the same per-epoch cost, **new is ≈ {(base_ep/new_ep):.2f}× faster** wall-clock")

    out.append(axis1_trajectory(base_df, new_df))
    out.append(axis2_jacobian(base_metrics, new_metrics))
    out.append(axis3_winners(base_df, new_df))

    summary = Path(__file__).parent / "validation_compare.md"
    summary.write_text("\n".join(out))
    print(f"wrote {summary.name}")
    # Also dump tables for reference
    base_df.to_csv(Path(__file__).parent / "validation_base.csv", index=False)
    new_df.to_csv(Path(__file__).parent / "validation_new.csv", index=False)


if __name__ == "__main__":
    main()
