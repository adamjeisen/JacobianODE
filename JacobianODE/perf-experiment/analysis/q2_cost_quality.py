"""How much val-loss quality do we lose if we stop training early?

For each run, compute val loss at fractions f ∈ {0.25, 0.5, 0.75, 0.9} of
total runtime. Compare to run-final best.

Also: the alternative framing — at what walltime would each run already be
within tol_rel of its eventual best?
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


def per_epoch(hist: pd.DataFrame, run_id: str) -> pd.DataFrame:
    g = hist[hist["run_id"] == run_id].sort_values("_step").copy()
    if g.empty:
        return g
    g["epoch"] = g["epoch"].ffill()
    keep = [c for c in g.columns if c not in ("run_id", "era", "epoch")]
    out = g.groupby("epoch")[keep].agg(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)
    return out.reset_index().sort_values("epoch")


def analyse(slug: str):
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    rows = []
    for run_id in cfg["run_id"]:
        ep = per_epoch(hist, run_id)
        if ep.empty or "trajectory val_loss" not in ep.columns:
            continue
        tv = pd.to_numeric(ep["trajectory val_loss"], errors="coerce")
        rt = pd.to_numeric(ep["_runtime"], errors="coerce")
        ok = tv.notna() & rt.notna()
        if not ok.any():
            continue
        tv = tv[ok].values
        rt = rt[ok].values
        ep_n = ep["epoch"][ok].values
        # Best-so-far at each epoch (running min)
        best_so_far = np.minimum.accumulate(tv)
        run_final_best = best_so_far[-1]
        total_rt = rt[-1]
        out = {"slug": slug, "run_id": run_id, "n_val_epochs": len(tv), "total_rt_s": total_rt,
               "run_final_best": run_final_best}
        # Quality at fractions of total runtime
        for f in [0.25, 0.5, 0.75, 0.9, 1.0]:
            target = f * total_rt
            i = np.searchsorted(rt, target, side="right") - 1
            i = max(0, i)
            out[f"best_at_{int(f*100)}pct_rt"] = float(best_so_far[i])
            out[f"epoch_at_{int(f*100)}pct_rt"] = float(ep_n[i])
            out[f"rel_excess_at_{int(f*100)}pct_rt"] = (
                float(best_so_far[i]) / run_final_best - 1.0 if run_final_best > 0 else np.nan
            )
        # First runtime within tolerance of final best
        for tol in [0.05, 0.1, 0.2]:
            gate = run_final_best * (1.0 + tol)
            hit = np.where(best_so_far <= gate)[0]
            if hit.size:
                first_i = int(hit[0])
                out[f"rt_within_{int(tol*100)}pct_s"] = float(rt[first_i])
                out[f"rt_within_{int(tol*100)}pct_frac"] = float(rt[first_i]) / total_rt
                out[f"epoch_within_{int(tol*100)}pct"] = float(ep_n[first_i])
            else:
                out[f"rt_within_{int(tol*100)}pct_s"] = np.nan
                out[f"rt_within_{int(tol*100)}pct_frac"] = np.nan
                out[f"epoch_within_{int(tol*100)}pct"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def fmt_pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "—"


def main():
    out_lines = ["# Q2: Cost / quality curve\n"]
    for slug in ["lorenz_current", "lorenz_current_obsnoise001", "wmtask_vanilla_old"]:
        df = analyse(slug)
        if df.empty:
            continue
        df.to_csv(Path(__file__).parent / f"q2_costquality_{slug}.csv", index=False)
        out_lines.append(f"\n## {slug} (n={len(df)})\n")

        # If we stopped at fraction f of runtime, what's the median quality vs final best?
        out_lines.append("\n**Quality if we stopped at runtime fraction f:**\n")
        out_lines.append("```")
        out_lines.append(f"{'f':>4s}  {'median_rel_excess':>18s}  {'p90_rel_excess':>15s}  {'median_epoch':>14s}")
        for f in [25, 50, 75, 90]:
            ex = df[f"rel_excess_at_{f}pct_rt"]
            ep = df[f"epoch_at_{f}pct_rt"]
            out_lines.append(
                f"{f/100:.2f}  {fmt_pct(ex.median()):>18s}  "
                f"{fmt_pct(ex.quantile(0.9)):>15s}  {ep.median():>14.0f}"
            )
        out_lines.append("```\n")

        # Time to within-tol-of-final-best
        out_lines.append("\n**Walltime to reach within tol of run-final best (median across runs):**\n")
        out_lines.append("```")
        out_lines.append(f"{'tol':>5s}  {'median_rt_s':>12s}  {'frac_total_rt':>14s}  {'median_epoch':>14s}")
        for tol in [5, 10, 20]:
            rt = df[f"rt_within_{tol}pct_s"]
            fr = df[f"rt_within_{tol}pct_frac"]
            ep = df[f"epoch_within_{tol}pct"]
            out_lines.append(
                f"{tol:>4d}%  {rt.median():>12.0f}  {fmt_pct(fr.median()):>14s}  {ep.median():>14.0f}"
            )
        out_lines.append("```\n")

    summary = Path(__file__).parent / "q2_costquality_summary.md"
    summary.write_text("\n".join(out_lines))
    print(f"wrote {summary.name}")


if __name__ == "__main__":
    main()
