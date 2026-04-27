"""Two-stage sweep simulation across ALL active groups in the last 30 days.

For each (project, group) with >=5 finished runs:
  - pull only val-loss + epoch + runtime via scan_history(keys=...)
  - for each Stage-A epoch K in {10, 15, 20, 30, 50}:
      - rank by best-so-far val at epoch K
      - record overlap of top-frac vs eventual top-frac (frac=1/3, 1/2)
  - aggregate across groups: how often does the eventual top-1 / top-3 survive?
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

CACHE = Path(__file__).parent / "cache_two_stage"
CACHE.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path(__file__).parent / "two_stage_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Metric name candidates — varies across eras.
VAL_KEYS = ["trajectory val_loss", "mean val loss"]
EPOCH_KEY = "epoch"
RUNTIME_KEY = "_runtime"


def discover_groups(min_runs: int = 5, days: int = 30):
    api = wandb.Api()
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    out = []
    projs = list(api.projects(entity="JacobianODE"))
    for p in projs:
        try:
            runs = list(api.runs(f"JacobianODE/{p.name}", per_page=500,
                                 filters={"state": "finished"}))
        except Exception:
            continue
        by_g = {}
        for r in runs:
            try:
                ca = dt.datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
            except Exception:
                continue
            if ca < cutoff:
                continue
            by_g.setdefault(r.group, []).append(r)
        for g, rs in by_g.items():
            if len(rs) >= min_runs:
                out.append((p.name, g, rs))
    return out


def pull_curves(project: str, group: str, runs: list, force: bool = False):
    """For each run, return DataFrame with columns: epoch, best_so_far_val.
    Returns dict run_id -> Series indexed by epoch with best-so-far val loss."""
    safe_g = group.replace("/", "_")
    cache_path = CACHE / f"{project}__{safe_g}.pkl.gz"
    if not force and cache_path.exists():
        return pd.read_pickle(cache_path)

    series_by_run = {}
    for r in runs:
        # Try each candidate val key
        rows = []
        try:
            for row in r.scan_history(keys=[EPOCH_KEY, RUNTIME_KEY] + VAL_KEYS):
                rows.append(row)
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # Pick first available val key with data
        for k in VAL_KEYS:
            if k in df.columns and pd.to_numeric(df[k], errors="coerce").notna().any():
                df["_val"] = pd.to_numeric(df[k], errors="coerce")
                break
        else:
            continue
        if "epoch" not in df.columns:
            continue
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.dropna(subset=["epoch", "_val"])
        if df.empty:
            continue
        # Per-epoch min (best in epoch)
        per_ep = df.groupby(df["epoch"].astype(int))["_val"].min()
        bsf = per_ep.cummin()
        series_by_run[r.id] = bsf
    pd.to_pickle(series_by_run, cache_path)
    return series_by_run


def simulate(series_by_run: dict, K_list=(5, 10, 15, 20, 30, 50)):
    """For each K, compute:
      - n_runs_with_data_at_K
      - whether eventual best-1 was in top-fraction at K (frac=1/3, 1/2, 2/3)
      - whether eventual best-3 had >= 1, 2, 3 in top-fraction at K
    """
    n = len(series_by_run)
    if n < 4:
        return None
    # Final-best: last value of each best_so_far series
    final = pd.Series({rid: float(s.iloc[-1]) for rid, s in series_by_run.items()
                       if not s.empty})
    if len(final) < 4:
        return None
    final = final.sort_values()
    eventual_top1 = final.index[0]
    eventual_top3 = set(final.index[:3])
    rows = []
    for K in K_list:
        valK = {}
        for rid, s in series_by_run.items():
            if K in s.index:
                valK[rid] = float(s.loc[K])
            else:
                # If the run ended before epoch K, use its run-final
                if not s.empty:
                    valK[rid] = float(s.iloc[-1])
        if len(valK) < 4:
            continue
        v = pd.Series(valK).sort_values()
        n_total = len(v)
        # For each fraction, compute the cull threshold
        rec = {"K": K, "n_runs": n_total, "n_finished_at_K": int(sum(K in s.index for s in series_by_run.values()))}
        for frac in [0.33, 0.5, 0.67]:
            keep = max(1, int(np.ceil(n_total * frac)))
            survivors = set(v.index[:keep])
            rec[f"top1_in_top{int(frac*100)}pct@K"] = int(eventual_top1 in survivors)
            rec[f"top3_in_top{int(frac*100)}pct@K"] = len(eventual_top3 & survivors)
        rec["spearman"] = v.rank().corr(final.rank(), method="pearson")
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    print("=== discovering groups ===")
    pairs = discover_groups(min_runs=5, days=30)
    print(f"  {len(pairs)} (project, group) pairs found")

    all_results = []
    for i, (p, g, rs) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {p} / {g}  (n_runs={len(rs)})")
        try:
            series = pull_curves(p, g, rs)
        except Exception as e:
            print(f"  ERROR pulling: {e}")
            continue
        if not series:
            print(f"  no usable data (no val key found)")
            continue
        sim = simulate(series)
        if sim is None or sim.empty:
            print(f"  not enough data for simulation")
            continue
        sim["project"] = p
        sim["group"] = g
        sim["n_runs_total"] = len(rs)
        all_results.append(sim)
        # Brief per-group summary
        for K in [10, 30]:
            row = sim[sim["K"] == K]
            if not row.empty:
                r = row.iloc[0]
                print(f"  K={K}: n={int(r['n_runs'])} top1∈top50%@K={r['top1_in_top50pct@K']}, "
                      f"top3∈top50%@K={r['top3_in_top50pct@K']}/3 spearman={r['spearman']:.3f}")

    if not all_results:
        print("no usable groups")
        return
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_results.csv", index=False)
    print(f"\nwrote all_results.csv with {len(combined)} rows ({combined['group'].nunique()} groups)")

    # Aggregate: at each K and fraction, % of groups where top-1 survived
    agg_lines = ["# Two-stage sweep simulation — aggregate across all groups\n"]
    for K in sorted(combined["K"].unique()):
        agg_lines.append(f"\n## Stage A epoch K={K}\n")
        sub = combined[combined["K"] == K]
        n_groups = len(sub)
        agg_lines.append(f"n_groups: {n_groups}\n")
        for frac in [33, 50, 67]:
            top1_kept = sub[f"top1_in_top{frac}pct@K"].sum()
            top3_kept = sub[f"top3_in_top{frac}pct@K"]
            agg_lines.append(
                f"- top-{frac}% cull: top-1 survives in **{top1_kept}/{n_groups} = "
                f"{top1_kept/n_groups*100:.0f}%** of groups; "
                f"of eventual top-3, median {top3_kept.median():.1f} of 3 survive "
                f"(min={top3_kept.min()}, p25={top3_kept.quantile(0.25):.1f})"
            )
        med_rho = sub["spearman"].median()
        agg_lines.append(f"- median rank correlation @ K: {med_rho:.3f}\n")

    md_path = OUT_DIR / "summary.md"
    md_path.write_text("\n".join(agg_lines))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
