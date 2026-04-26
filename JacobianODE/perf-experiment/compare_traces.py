"""Compare per-step traces from PerStepTimingCallback.

Usage:
    python compare_traces.py <baseline.json> <candidate.json> [warmup=5] [atol=1e-4]

Reports:
- per-step time mean ± std (after dropping warmup)
- delta (candidate vs baseline)
- max |loss diff| and whether numpy.allclose passes at atol
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(p: str):
    with open(p) as f:
        rec = json.load(f)
    steps = np.array([r["step"] for r in rec])
    losses = np.array([r["loss"] for r in rec], dtype=float)
    times = np.array([r["walltime_seconds"] for r in rec], dtype=float)
    return steps, losses, times


def summarize(name: str, times: np.ndarray, warmup: int = 5):
    body = times[warmup:]
    print(f"  {name}: n_steps={len(times)} | warmup={warmup} dropped | "
          f"mean_per_step={body.mean():.4f}s ± {body.std():.4f}")
    return body.mean()


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    base_p, cand_p = sys.argv[1], sys.argv[2]
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4

    bs, bl, bt = load(base_p)
    cs, cl, ct = load(cand_p)

    print(f"Baseline:  {Path(base_p).name}")
    print(f"Candidate: {Path(cand_p).name}")

    n = min(len(bl), len(cl))
    print(f"Comparing {n} steps")
    bl_n = bl[:n]
    cl_n = cl[:n]
    diff = np.abs(bl_n - cl_n)
    nan_mask = np.isnan(bl_n) | np.isnan(cl_n)
    finite_diff = diff[~nan_mask]
    max_abs = float(finite_diff.max()) if len(finite_diff) else float("nan")
    rel = finite_diff / (np.abs(bl_n[~nan_mask]) + 1e-12)
    max_rel = float(rel.max()) if len(rel) else float("nan")

    print(f"\nLoss regression:")
    print(f"  max |Δloss|     = {max_abs:.3e}")
    print(f"  max relΔ        = {max_rel:.3e}")
    print(f"  allclose atol={atol}: {np.allclose(bl_n[~nan_mask], cl_n[~nan_mask], atol=atol, rtol=0)}")

    print(f"\nPer-step time:")
    bm = summarize("baseline ", bt, warmup)
    cm = summarize("candidate", ct, warmup)
    delta_pct = (cm - bm) / bm * 100.0 if bm > 0 else float("nan")
    print(f"  candidate vs baseline: {delta_pct:+.2f}% per-step ({cm-bm:+.4f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
