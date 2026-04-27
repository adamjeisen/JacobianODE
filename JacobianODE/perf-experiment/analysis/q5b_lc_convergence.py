"""Does loop_closure_weight=0 converge faster than higher weights?
Plot val curves grouped by loop_closure_weight and report time-to-target."""

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


def collect(slug: str):
    cfg = pd.read_pickle(CACHE / f"{slug}__configs.pkl.gz")
    hist = pd.read_pickle(CACHE / f"{slug}__history.pkl.gz")
    rows = []
    for _, c in cfg.iterrows():
        run_id = c["run_id"]
        ep = per_epoch(hist, run_id)
        if ep.empty or "trajectory val_loss" not in ep.columns:
            continue
        tv = pd.to_numeric(ep["trajectory val_loss"], errors="coerce")
        rt = pd.to_numeric(ep["_runtime"], errors="coerce")
        ok = tv.notna() & rt.notna()
        if not ok.any():
            continue
        bsf = tv[ok].cummin().values
        runtime = rt[ok].values
        ep_n = ep["epoch"][ok].astype(int).values
        rows.append({
            "run_id": run_id,
            "lc_w": float(c.get("training.lightning.loop_closure_weight", np.nan) or np.nan),
            "epochs": ep_n,
            "best_so_far": bsf,
            "runtime": runtime,
            "final_best": bsf[-1],
            "total_rt": runtime[-1],
        })
    return rows


def fig_per_lc_curves(slug: str):
    rows = collect(slug)
    by_lc = {}
    for r in rows:
        by_lc.setdefault(r["lc_w"], []).append(r)
    keys = sorted(k for k in by_lc.keys() if not pd.isna(k))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for k in keys:
        runs = by_lc[k]
        # Median of best-so-far across runs at common epochs
        common_ep = sorted(set().union(*(set(r["epochs"]) for r in runs)))
        med = []
        for e in common_ep:
            vals = [r["best_so_far"][np.where(r["epochs"] == e)[0][0]]
                    for r in runs if e in r["epochs"]]
            med.append(np.median(vals))
        label = f"lc_w={k:g} (n={len(runs)})"
        axes[0].plot(common_ep, med, label=label, linewidth=1.5)

        med_rt = []
        for e in common_ep:
            vals = [r["runtime"][np.where(r["epochs"] == e)[0][0]]
                    for r in runs if e in r["epochs"]]
            med_rt.append(np.median(vals))
        axes[1].plot(med_rt, med, label=label, linewidth=1.5)

    for ax in axes:
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("best-so-far trajectory val loss")
    axes[0].set_title(f"{slug}: median best-so-far by lc_w (vs epoch)")
    axes[1].set_xlabel("runtime (s)")
    axes[1].set_title(f"{slug}: median best-so-far by lc_w (vs runtime)")
    fig.tight_layout()
    out = FIGS / f"q5b_lc_curves_{slug}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out.name}")


def time_to_target(slug: str, target_rel: float = 0.10):
    """How many epochs to reach within target_rel of the GROUP-WIDE final best?
    Stratified by lc_w."""
    rows = collect(slug)
    if not rows:
        return ""
    group_min = min(r["final_best"] for r in rows)
    gate = group_min * (1.0 + target_rel)

    out = [f"\n#### {slug} — time to within {target_rel*100:.0f}% of group-wide final-best ({group_min:.5f})\n"]
    out.append("```")
    out.append(f"{'lc_w':>10s} {'n':>3s} {'med_ep':>7s} {'med_rt_s':>10s} {'pct_reach':>10s}")
    by_lc = {}
    for r in rows:
        by_lc.setdefault(r["lc_w"], []).append(r)
    for k in sorted(by_lc):
        if pd.isna(k):
            continue
        runs = by_lc[k]
        eps_to_gate = []
        rts_to_gate = []
        for r in runs:
            hit = np.where(r["best_so_far"] <= gate)[0]
            if hit.size:
                eps_to_gate.append(int(r["epochs"][hit[0]]))
                rts_to_gate.append(float(r["runtime"][hit[0]]))
        n_reach = len(eps_to_gate)
        n = len(runs)
        med_ep = np.median(eps_to_gate) if eps_to_gate else float("nan")
        med_rt = np.median(rts_to_gate) if rts_to_gate else float("nan")
        out.append(f"{k:>10g} {n:>3d} {med_ep:>7.0f} {med_rt:>10.0f} {n_reach}/{n}")
    out.append("```\n")
    return "\n".join(out)


def main():
    out_lines = ["# Q5b: lc-weight effect on convergence speed\n"]
    for slug in ["lorenz_current", "lorenz_current_obsnoise001"]:
        fig_per_lc_curves(slug)
        out_lines.append(time_to_target(slug, 0.05))
        out_lines.append(time_to_target(slug, 0.10))
        out_lines.append(time_to_target(slug, 0.20))
    summary = Path(__file__).parent / "q5b_summary.md"
    summary.write_text("\n".join(out_lines))
    print(f"wrote {summary.name}")


if __name__ == "__main__":
    main()
