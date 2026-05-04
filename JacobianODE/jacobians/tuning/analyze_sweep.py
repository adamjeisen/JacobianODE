"""Run the full analysis pipeline for a completed sweep.

Given the name of a sweep whose ``.done.json`` sentinel exists in
``$SWEEPS_DIR/done/``, this script:

1. Loads the sentinel (for the embedded expected_snapshot / experiment_metadata).
2. Runs ``run_analytics`` with the standard section set (``sweep_overview``,
   ``lyapunov``, ``prediction_windows``, ``mase``), saving figures as PNGs.
3. Extracts a compact ``metrics.json`` summarising:
   - Sweep-wide best-run metrics per obs_noise_scale.
   - Best run's Lyapunov spectrum (predicted + ground truth if available).
   - A Pass/Fail/Partial verdict against each success criterion from the
     experiment metadata (best-effort string matching — Claude can refine).
4. Copies the sentinel's ``experiment_metadata`` and ``expected_snapshot``
   into the output dir as ``context.json`` for the report-writing agent.

Output layout:

    $SWEEPS_DIR/analysis/<group>/
        figures/            PNGs keyed by section name
        metrics.json        compact structured summary
        context.json        sentinel fields needed for the report
        run_analytics.log   full stdout/stderr for debugging

Intended to be invoked on engaging via sbatch (GPU required for Lyapunov).
Usage:

    python -m JacobianODE.jacobians.tuning.analyze_sweep <group> \\
        [--sweeps-dir /orcd/.../sweeps] \\
        [--save-dir /orcd/.../lightning/latent_jac_runs]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("JacobianODE.analyze_sweep")

DEFAULT_SWEEPS_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps"
DEFAULT_SAVE_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs"

ANALYTICS_SECTIONS = [
    "sweep_overview",
    "reconstruction",
    "mase",
    "latent_utilization",
    "lyapunov",
    "kaplan_yorke",
    "prediction_windows",
    "prediction_detail",
    "long_trajectory",
    "encoder_decoder_jacobians",
    "amplification",
    "tangent_spectrum",
]


def iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


_SWEPT_PATH_EXCLUDES_PREFIX = ("hydra.", "_wandb")
_SWEPT_PATH_EXCLUDES_EXACT = {
    "wandb_entity",
    "wandb_project",
    "wandb_group",
    "wandb_id",
    "wandb_name",
    # Paths that trivially vary across runs but aren't scientifically swept.
    "training.logger.save_dir",
    "training.logger_save_dirs",
    # Auto-resolved per-area encoder layouts: deterministic functions of the
    # actually-swept axes (n_observed_per_area, n_delays) via the
    # auto_partial_obs sentinel. Including them blows up the per-run table
    # — each cell renders as a long index list, and the sweep axis
    # (n_delays) already encodes the variation.
    "model.encoder.area_indices",
    "model.encoder.n_target_dims_per_block",
}


def _flatten_config(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted-path keys.

    Non-dict values are recorded as-is. Lists are treated as leaves
    (we don't index into them, since list-position doesn't correspond to
    a Hydra override path)."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_config(v, path))
            else:
                out[path] = v
    return out


def _discover_swept_paths(configs: list[dict[str, Any]]) -> list[str]:
    """Return the sorted list of flattened config paths whose value varies
    across ``configs`` (more than one distinct stringified value).

    Robustness note: an orphan wandb run whose process died mid-init can
    log an almost-empty config. Naïvely diffing against that run would
    mark *every* config path as "varying" (present in the healthy runs,
    absent in the broken one), producing a per-run table with hundreds of
    spurious columns. We defend against that by dropping configs whose
    flattened path count is dramatically below the median (<20%) before
    doing the diff. The broken run still flows through per-run matching
    and surfaces as ``unmatched_runs``.
    """
    if len(configs) < 2:
        return []
    flat = [_flatten_config(c) for c in configs]
    # Drop degenerate configs before discovering swept axes.
    sizes = sorted(len(f) for f in flat)
    median_size = sizes[len(sizes) // 2]
    threshold = max(1, int(0.2 * median_size))
    filtered = [f for f in flat if len(f) >= threshold]
    if len(filtered) < len(flat):
        logger.warning(
            f"_discover_swept_paths: dropped "
            f"{len(flat) - len(filtered)} degenerate config(s) (size < "
            f"{threshold}; median {median_size}) before diffing."
        )
    flat = filtered
    if len(flat) < 2:
        return []
    all_paths = set().union(*flat)
    varying: list[str] = []
    for p in all_paths:
        if any(p.startswith(pre) for pre in _SWEPT_PATH_EXCLUDES_PREFIX):
            continue
        if p in _SWEPT_PATH_EXCLUDES_EXACT:
            continue
        vals = {repr(f.get(p)) for f in flat}
        if len(vals) > 1:
            varying.append(p)
    return sorted(varying)


def summarize_sweep_from_wandb(
    wandb_entity: str,
    wandb_project: str,
    group: str,
    resolved_runs: list | None = None,
) -> dict:
    """Pull per-run best metrics from wandb and group them by obs_noise_scale.

    When ``resolved_runs`` is supplied (from the sentinel's
    ``expected.hydra.resolved_runs``), every wandb run in the group is
    matched to a ``run_idx`` via ``match_run_to_idx`` and the best wandb
    run per slot is kept in ``per_run``. Wandb runs that don't match any
    slot are surfaced in ``unmatched_runs`` (never silently dropped), and
    slots that had ``>1`` matching wandb run are surfaced in
    ``duplicate_matches`` with the chosen run_id and the other candidate
    IDs. An ``expected_run_count`` sanity check is also recorded so a
    deviation from the sentinel's count shows up loudly in the report.

    When ``resolved_runs`` is None, the legacy behavior applies: every
    wandb run becomes one ``per_run`` row (good for back-compat with
    notebooks / direct calls).

    Returns a dict of:
        {
            "per_run": [{run_id, run_idx, lc_weight, obs_noise_scale,
                         best_traj_loss, best_mase,
                         swept_config: {path: value}, ...}],
            "unmatched_runs": [{run_id, ...}, ...],       # new
            "duplicate_matches": [{run_idx, chosen, others: [...]}],  # new
            "expected_run_count": int | None,             # new
            "matched_run_count": int,                     # new
            "best_by_obs_noise_scale": {ons: {...}},
            "overall_chosen_run": {run_id, ..., swept_config},
            "overall_best_mase": {...},
            "swept_paths": [path, ...],
        }
    """
    import wandb
    from .monitor import match_run_to_idx

    api = wandb.Api()
    runs = list(api.runs(f"{wandb_entity}/{wandb_project}", filters={"group": group}))
    # Capture every run's config (keyed by run.id) up front so we can discover
    # which paths actually vary across the sweep. The discovered list drives
    # the per-run `swept_config` column set below.
    configs_by_rid: dict[str, dict[str, Any]] = {r.id: dict(r.config) for r in runs}
    swept_paths = _discover_swept_paths(list(configs_by_rid.values()))
    # Assemble one row per wandb run first, then fold down per run_idx.
    all_rows: list[dict[str, Any]] = []
    # Columns we want; handle both latent-model and vanilla-model key naming.
    cols = [
        "epoch",
        "val/trajectory_loss", "trajectory val_loss",
        "val/loop_closure_loss", "val loop closure loss",
        "val/trajectory_mase", "trajectory val mase",
        "val/trajectory_r2_score", "trajectory val r2_score",
    ]
    for r in runs:
        cfg = dict(r.config)
        lc = _nested_get(cfg, "training.lightning.loop_closure_weight")
        ons = _nested_get(cfg, "training.lightning.obs_noise_scale")

        # `run.history()` returns a pandas DataFrame with ALL logged keys (no
        # AND-filter trap like scan_history(keys=[...])). Request a large
        # enough `samples` to cover all epochs.
        try:
            df = r.history(samples=10_000)
        except Exception as e:
            logger.warning(f"history() failed for run {r.id}: {e}")
            continue

        if df.empty:
            continue

        rows = []
        for _, row in df.iterrows():
            # val/trajectory_loss is latent-model naming; `trajectory val_loss`
            # (space, no slash) is vanilla-model naming.
            tl = row.get("val/trajectory_loss")
            if tl is None or (isinstance(tl, float) and tl != tl):  # NaN check
                tl = row.get("trajectory val_loss")
            if tl is None or (isinstance(tl, float) and tl != tl):
                continue
            lc_loss = row.get("val/loop_closure_loss")
            if lc_loss is None or (isinstance(lc_loss, float) and lc_loss != lc_loss):
                lc_loss = row.get("val loop closure loss")
            mase = row.get("val/trajectory_mase")
            if mase is None or (isinstance(mase, float) and mase != mase):
                mase = row.get("trajectory val mase")
            r2 = row.get("val/trajectory_r2_score")
            if r2 is None or (isinstance(r2, float) and r2 != r2):
                r2 = row.get("trajectory val r2_score")
            rows.append({
                "epoch": row.get("epoch"),
                "traj_loss": float(tl),
                "lc_loss": float(lc_loss) if lc_loss is not None and lc_loss == lc_loss else None,
                "mase": float(mase) if mase is not None and mase == mase else None,
                "r2": float(r2) if r2 is not None and r2 == r2 else None,
            })
        if not rows:
            continue
        best_by_tl = min(rows, key=lambda d: d["traj_loss"])
        rows_with_mase = [d for d in rows if d.get("mase") is not None]
        best_by_mase = (
            min(rows_with_mase, key=lambda d: d["mase"])
            if rows_with_mase else best_by_tl
        )
        flat_cfg = _flatten_config(cfg)
        swept_config = {p: flat_cfg.get(p) for p in swept_paths}
        # Match to a run_idx using the sweep's declared overrides. Only
        # meaningful when resolved_runs is passed in; None otherwise.
        run_idx = (
            match_run_to_idx(cfg, resolved_runs) if resolved_runs else None
        )
        all_rows.append({
            "run_id": r.id,
            "run_idx": run_idx,
            "state": r.state,
            "lc_weight": _to_float_or_none(lc),
            "obs_noise_scale": _to_float_or_none(ons),
            "best_traj_loss": best_by_tl["traj_loss"],
            "best_traj_loss_epoch": best_by_tl["epoch"],
            "best_mase": best_by_mase["mase"],
            "best_mase_epoch": best_by_mase["epoch"],
            "lc_loss_at_best_tl": best_by_tl["lc_loss"],
            "r2_at_best_tl": best_by_tl["r2"],
            "n_epochs": len(rows),
            # All swept axes, so the report can show *which* config a row is.
            "swept_config": swept_config,
        })

    # -------------------------------------------------------------
    # Fold wandb runs down to one entry per run_idx.
    # Safeguard 1: unmatched wandb runs go to ``unmatched_runs`` —
    #              never silently dropped.
    # Safeguard 2: when >1 wandb run matches a slot, log both IDs and
    #              which was chosen (by best_traj_loss) in
    #              ``duplicate_matches``.
    # Safeguard 3: record ``expected_run_count`` and
    #              ``matched_run_count`` for a loud render-time warning
    #              when they don't agree.
    # When resolved_runs is None (legacy callers), skip the fold-down:
    # per_run stays as "one row per wandb run" for back-compat.
    # -------------------------------------------------------------
    unmatched_runs: list[dict[str, Any]] = []
    duplicate_matches: list[dict[str, Any]] = []
    expected_count: int | None = (
        len(resolved_runs) if resolved_runs is not None else None
    )

    if resolved_runs is None:
        per_run = all_rows
    else:
        by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in all_rows:
            if row["run_idx"] is None:
                unmatched_runs.append(row)
            else:
                by_idx[row["run_idx"]].append(row)

        per_run = []
        for idx, rows in sorted(by_idx.items()):
            rows_with_tl = [r for r in rows if r["best_traj_loss"] is not None]
            if not rows_with_tl:
                # Treat fully-empty slots (e.g. wandb runs that matched but
                # have no val data) the same way as unmatched — keep them
                # visible rather than silently dropping.
                unmatched_runs.extend(rows)
                continue
            chosen = min(rows_with_tl, key=lambda r: r["best_traj_loss"])
            per_run.append(chosen)
            if len(rows) > 1:
                others = [r for r in rows if r["run_id"] != chosen["run_id"]]
                duplicate_matches.append({
                    "run_idx": idx,
                    "chosen_run_id": chosen["run_id"],
                    "other_run_ids": [r["run_id"] for r in others],
                    "n_candidates": len(rows),
                })
                logger.warning(
                    f"run_idx={idx}: {len(rows)} wandb runs matched "
                    f"(chose {chosen['run_id']} by best_traj_loss; "
                    f"dropped {[r['run_id'] for r in others]})"
                )
        if unmatched_runs:
            logger.warning(
                f"{len(unmatched_runs)} wandb runs did not match any run_idx "
                f"and are surfaced in metrics_summary.unmatched_runs "
                f"(IDs: {[r['run_id'] for r in unmatched_runs]})"
            )
        if expected_count is not None and len(per_run) != expected_count:
            logger.warning(
                f"matched_run_count ({len(per_run)}) != expected_run_count "
                f"({expected_count}) for group {group}; some slots may be "
                f"missing or the sweep is still in progress."
            )

    # Group best per obs_noise_scale
    by_ons: dict[Any, list[dict]] = defaultdict(list)
    for row in per_run:
        by_ons[row["obs_noise_scale"]].append(row)
    best_by_ons = {}
    for ons, rows in by_ons.items():
        rows_with_tl = [r for r in rows if r["best_traj_loss"] is not None]
        if not rows_with_tl:
            continue
        best = min(rows_with_tl, key=lambda r: r["best_traj_loss"])
        best_by_ons[str(ons)] = best

    # `overall_chosen_run` matches the ranking criterion used by
    # `run_analytics` (best_traj_loss among survivors). This is the run whose
    # checkpoint gets loaded for the deep analysis and whose figures end up in
    # the report — so it's what the Results section should feature.
    # `overall_best_mase` is retained for back-compat (older consumers).
    overall_chosen = min(
        (r for r in per_run if r["best_traj_loss"] is not None),
        key=lambda r: r["best_traj_loss"],
        default=None,
    )
    overall_best_mase = min(
        (r for r in per_run if r["best_mase"] is not None),
        key=lambda r: r["best_mase"],
        default=None,
    )
    return {
        "per_run": per_run,
        "unmatched_runs": unmatched_runs,
        "duplicate_matches": duplicate_matches,
        "expected_run_count": expected_count,
        "matched_run_count": len(per_run),
        "best_by_obs_noise_scale": best_by_ons,
        "overall_chosen_run": overall_chosen,
        "overall_best_mase": overall_best_mase,
        "swept_paths": swept_paths,
        "n_runs": len(per_run),
    }


def _nested_get(d: Any, key: str, default=None):
    cur = d
    for p in key.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _to_float_or_none(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_full_analytics(
    wandb_entity: str,
    wandb_project: str,
    group: str,
    save_dir: Path,
    output_dir: Path,
    true_lyapunov: list | None = None,
    lyapunov_burn_in_steps: int = 400,
    lyapunov_burn_in_drop: int = 100,
    sections_override: list[str] | None = None,
    eigenvalue_threshold: float = float("inf"),
) -> tuple[dict[str, str], str]:
    """Invoke run_analytics with the standard (or overridden) section set.

    Captures run_analytics's stdout into ``output_dir/run_analytics.log`` so
    the printed per-run diagnostics / Lyapunov tables end up in the report.

    Returns ``(figure_map, captured_stdout)`` — the figure map is
    ``{section: figure_path}`` for the PNGs saved to ``output_dir/figures/``.
    """
    import contextlib
    import io

    from ..run_analytics import run_analytics

    sections = sections_override if sections_override is not None else ANALYTICS_SECTIONS
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running run_analytics for group={group}, sections={sections}")

    # Capture stdout (the bulk of run_analytics's diagnostic output prints
    # to stdout) into a string buffer and mirror to a log file.
    buf = io.StringIO()

    class Tee:
        def __init__(self, a, b):
            self.a, self.b = a, b
        def write(self, s):
            self.a.write(s); self.b.write(s)
        def flush(self):
            self.a.flush(); self.b.flush()

    with contextlib.redirect_stdout(Tee(sys.stdout, buf)):
        figures = run_analytics(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            save_dir=str(save_dir),
            wandb_group=group,
            true_lyapunov=true_lyapunov,
            output=["save", "return"],
            output_dir=str(figures_dir),
            sections=sections,
            ranking_method="best_traj_loss",
            eigenvalue_threshold=eigenvalue_threshold,
            lyapunov_burn_in_steps=lyapunov_burn_in_steps,
            lyapunov_burn_in_drop=lyapunov_burn_in_drop,
            use_all_runs=True,
            return_model=False,
        )
    captured = buf.getvalue()
    (output_dir / "run_analytics.log").write_text(captured)

    # Record which sections actually produced figures
    saved = {}
    if isinstance(figures, dict):
        for section, fig in figures.items():
            if fig is None:
                continue
            # run_analytics saves as <section>.png by default
            candidate = figures_dir / f"{section}.png"
            if candidate.is_file():
                saved[section] = str(candidate)
            else:
                # Fallback: any PNG matching the section substring
                for p in figures_dir.glob(f"*{section}*.png"):
                    saved[section] = str(p)
                    break
    return saved, captured


def evaluate_success_criteria(criteria: list[str], metrics_summary: dict) -> list[dict]:
    """Best-effort automated verdict per success criterion.

    Criteria are free-text strings. We do substring matching against simple
    numeric targets (e.g. "MASE < 0.86"). Anything we can't parse gets a
    Pass/Fail/Unknown verdict of Unknown, leaving Claude to judge.
    """
    import re

    # Prefer the run chosen by the ranking criterion (best_traj_loss); fall
    # back to the legacy best-by-MASE entry for pre-schema-2 metrics files.
    featured = (
        metrics_summary.get("overall_chosen_run")
        or metrics_summary.get("overall_best_mase")
        or {}
    )
    best_mase = featured.get("best_mase")
    best_lc_loss = featured.get("lc_loss_at_best_tl")
    best_r2 = featured.get("r2_at_best_tl")

    verdicts = []
    for c in criteria:
        verdict = {"criterion": c, "verdict": "Unknown", "note": ""}
        c_lower = c.lower()
        m_num = re.search(r"([<>]=?)\s*([0-9.]+)", c)
        try:
            if "mase" in c_lower and m_num and best_mase is not None:
                op, thr = m_num.group(1), float(m_num.group(2))
                ok = eval(f"{best_mase} {op} {thr}")
                verdict["verdict"] = "Pass" if ok else "Fail"
                verdict["note"] = f"Best MASE = {best_mase:.4f}; threshold {op} {thr}"
            elif ("r2" in c_lower or "r²" in c_lower) and m_num and best_r2 is not None:
                op, thr = m_num.group(1), float(m_num.group(2))
                ok = eval(f"{best_r2} {op} {thr}")
                verdict["verdict"] = "Pass" if ok else "Fail"
                verdict["note"] = f"Best R² = {best_r2:.4f}; threshold {op} {thr}"
            elif "loop closure" in c_lower and "explosion" in c_lower:
                # Heuristic: no explosion if max seen loop closure < 10
                per_run = metrics_summary.get("per_run", [])
                worst_lc = max(
                    (r["lc_loss_at_best_tl"] for r in per_run if r.get("lc_loss_at_best_tl") is not None),
                    default=None,
                )
                if worst_lc is not None:
                    verdict["verdict"] = "Pass" if worst_lc < 10 else "Fail"
                    verdict["note"] = f"Worst LC loss at best_tl = {worst_lc:.3f}"
        except Exception as e:
            verdict["note"] = f"Automated check failed: {e}"
        verdicts.append(verdict)
    return verdicts


def _compute_empirical_lyapunov(
    eq,
    trajs_raw: Any,
    dt: float,
    mu: float,
    sigma: float,
    device,
    chunk_size: int = 64,
) -> tuple[Any, Any] | tuple[None, None]:
    """Compute the empirical Lyapunov spectrum from ground-truth Jacobians.

    Mirrors the empirical-Lyapunov path in ``run_analytics.py``:
    - Denormalize trajectories (``traj * sigma + mu``).
    - Query ``eq.jac(traj, t=0)`` to get the true Jacobian at each point.
    - Run the QR-based ``compute_lyapunov_exponents`` on those Jacobians.

    Returns (mean_spectrum, per_traj_spectra) or (None, None) on failure.
    """
    import numpy as np
    import torch
    from ...models.latent_jacobian import LitLatentJacobianODE

    if eq is None:
        return None, None
    try:
        traj_raw_np = np.asarray(trajs_raw) * sigma + mu
        if hasattr(eq, "model"):
            # Torch-native eq.jac (wmtask): batched over leading dim.
            traj_t = torch.as_tensor(traj_raw_np).float().to(device)
            n = traj_t.shape[0]
            chunks: list = []
            for start in range(0, n, chunk_size):
                chunk = traj_t[start:start + chunk_size]
                jacs_chunk = eq.jac(chunk, t=0)
                le_chunk = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_chunk, dt)
                chunks.append(le_chunk.detach().cpu())
                del jacs_chunk, le_chunk
            all_emp = torch.cat(chunks, dim=0).numpy()  # (n, D)
        else:
            # Numpy-based dysts eq.jac: per-traj loop (D is small).
            all_emp_list = []
            for i in range(traj_raw_np.shape[0]):
                jacs_np = eq.jac(traj_raw_np[i], t=0)
                jacs_t = torch.as_tensor(jacs_np).float()
                le_i = LitLatentJacobianODE.compute_lyapunov_exponents(jacs_t, dt)
                all_emp_list.append(le_i.detach().cpu().numpy())
            all_emp = np.stack(all_emp_list, axis=0)
        return all_emp.mean(axis=0), all_emp
    except Exception as e:
        logger.warning(f"empirical Lyapunov computation failed: {e}")
        return None, None


def compute_per_run_lyapunov(
    wandb_entity: str,
    wandb_project: str,
    group: str,
    save_dir: Path,
    output_dir: Path,
    true_lyapunov: list | None = None,
    n_sample_trajectories: int = 24,
    chunk_size: int = 8,
    expected_snapshot: dict | None = None,
) -> dict[str, Any]:
    """Compute the Lyapunov spectrum for every run in the sweep.

    Reuses the WMTask trajectories / dataloaders across runs to avoid
    reloading data 27 times. Each run's model is loaded, its latent
    trajectory is computed, then Jacobians along the trajectory, then
    Lyapunov exponents via QR.

    Produces:
    - ``metrics_summary['per_run_lyapunov']`` — dict of run_id to
      ``{lambda_spectrum: [...], lambda_max, lambda_sum, ky_dim, error}``
    - ``figures/lyapunov_vs_val_loss.png`` — scatter of lambda_max vs traj_loss
    - ``figures/lyapunov_spectra_overlay.png`` — all per-run spectra on one plot

    Returns the per-run dict.
    """
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb as _wandb

    from ..checkpoints.loader import load_run, load_checkpoint
    from ...models.latent_jacobian import LitLatentJacobianODE

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List the group's runs (in wandb order). We used to filter on
    # state="finished" but that misses runs still actively training (state
    # "running") or preempt-crashed but with saved checkpoints (state
    # "crashed"). Drop the state filter and rely on a checkpoint-dir
    # existence check: if {save_dir}/{project}/{run_id}/checkpoints is
    # present, we have enough to compute a Lyapunov spectrum from the
    # best-by-metric checkpoint, regardless of whether the run has
    # formally finished yet.
    api = _wandb.Api()
    raw_runs = list(api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters={"group": group},
    ))
    ckpt_base = Path(save_dir) / wandb_project
    all_runs = [
        r for r in raw_runs
        if (ckpt_base / r.id / "checkpoints").is_dir()
    ]
    dropped = len(raw_runs) - len(all_runs)
    logger.info(
        f"Computing per-run Lyapunov for {len(all_runs)} runs with checkpoints "
        f"({dropped} run(s) skipped for missing checkpoint dir) — states: "
        + ", ".join(sorted({r.state for r in all_runs}))
    )

    # Trajectory cache. Each run gets its own delay-embedded trajectory
    # because n_delays can vary across a sweep (e.g. the ndelays sweeps), so
    # we can't reuse the first run's shape. The empirical ground-truth
    # Lyapunov spectrum is n_delays-independent (it's computed from eq.jac
    # on the raw full-state trajectory) so we cache only that across runs.
    dt_cached = None
    per_run: dict[str, Any] = {}

    # Empirical ground-truth Lyapunov spectrum, computed once from eq.jac on
    # the full-state trajectories (see run_analytics.py for the canonical path).
    empirical_mean: np.ndarray | None = None
    empirical_per_traj: np.ndarray | None = None
    empirical_per_condition: dict[str, np.ndarray] = {}

    for i, run in enumerate(all_runs):
        run_id = run.id
        try:
            # Always generate_data=True: different runs in the same sweep can
            # have different n_delays / observed_indices, so the delay-embedded
            # trajectory shape varies per run. Reusing the first run's data
            # caused every subsequent run with a different n_delays to crash
            # with shape-mismatch errors from the encoder.
            # return_full_obs=True on the FIRST run only — that gives us the
            # raw full-state trajectory for the empirical Lyapunov spectrum.
            loaded = load_run(
                f"{wandb_entity}/{wandb_project}",
                run_id=run_id,
                save_dir=str(save_dir),
                generate_data=True,
                verbose=False,
                return_full_obs=(i == 0),
            )
            run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded
            if i == 0:
                dt_cached = dt
                if trajs is not None and "test_trajs" in trajs:
                    # Compute empirical spectrum (same data/eq across runs).
                    # For combined-loader (conditioned) runs, compute it
                    # PER SOURCE — each condition has its own ground-truth
                    # eq object and its own subset of trajectories.
                    mu_val = cfg.data.postprocessing.get("mu", 0.0)
                    sigma_val = cfg.data.postprocessing.get("sigma", 1.0)
                    # Empirical needs the raw full-dim state (for eq.jac).
                    # train_trajs_full matches train_trajs's time range.
                    if "train_trajs_full" in trajs:
                        traj_for_emp = trajs["train_trajs_full"].sequence
                    elif values is not None:
                        traj_for_emp = values
                    else:
                        traj_for_emp = (
                            trajs["test_trajs_full"].sequence
                            if "test_trajs_full" in trajs
                            else trajs["test_trajs"].sequence
                        )
                    logger.info("Computing empirical ground-truth Lyapunov spectrum...")
                    src_by_c = trajs.get("source_eqs_by_condition")
                    train_cond = trajs.get("train_condition")
                    if src_by_c and train_cond is not None and len(src_by_c) > 1:
                        # Per-condition empirical: for each (cond_row, src_eq),
                        # restrict to that source's trajectory subset and
                        # compute lyap with the matching ground-truth eq.
                        train_cond_arr = np.asarray(train_cond)
                        traj_for_emp_arr = traj_for_emp.detach().cpu().numpy() if hasattr(traj_for_emp, 'cpu') else np.asarray(traj_for_emp)
                        empirical_per_condition: dict[str, np.ndarray] = {}
                        chunks_for_overall: list[np.ndarray] = []
                        for cond_row, src_eq in src_by_c:
                            mask = np.all(train_cond_arr == cond_row, axis=1)
                            if not mask.any():
                                continue
                            sub_traj = traj_for_emp_arr[mask]
                            label = f"c={cond_row.tolist()}"
                            sub_mean, sub_per_traj = _compute_empirical_lyapunov(
                                src_eq, sub_traj, dt, mu_val, sigma_val, device,
                            )
                            if sub_mean is not None:
                                empirical_per_condition[label] = sub_mean
                                chunks_for_overall.append(sub_per_traj)
                                logger.info(
                                    f"  empirical {label}: λ_max={sub_mean[0]:+.4f}, "
                                    f"λ_min={sub_mean[-1]:+.4f}, Σλ={sub_mean.sum():.3f}"
                                )
                        # Overall mean = simple mean across conditions for back-compat.
                        if empirical_per_condition:
                            empirical_mean = np.mean(
                                np.stack(list(empirical_per_condition.values())), axis=0
                            )
                            empirical_per_traj = np.concatenate(chunks_for_overall, axis=0)
                    else:
                        empirical_per_condition = {}
                        empirical_mean, empirical_per_traj = _compute_empirical_lyapunov(
                            eq, traj_for_emp, dt, mu_val, sigma_val, device,
                        )
                    if empirical_mean is not None:
                        logger.info(
                            f"Empirical (overall) λ₁={empirical_mean[0]:.4f}, "
                            f"λ_min={empirical_mean[-1]:.4f}, "
                            f"Σλ={empirical_mean.sum():.3f}"
                        )

            load_checkpoint(run_obj, cfg, lit_model, save_dir=str(save_dir), verbose=False)
            lit_model = lit_model.to(device).eval()

            # Per-run model trajectories (delay-embedded with THIS run's
            # n_delays / observed_indices). Prefer train_trajs for its
            # longer contiguous slice — Lyapunov needs many Lyapunov times
            # to converge.
            if trajs is None or "test_trajs" not in trajs:
                raise RuntimeError(f"No trajectories returned by load_run for {run_id}")
            if "train_trajs" in trajs:
                model_seq = trajs["train_trajs"].sequence
                _cond_seq = trajs.get("train_condition")
            else:
                model_seq = trajs["test_trajs"].sequence
                _cond_seq = trajs.get("test_condition")
            # For conditioned models, the balanced split concatenates
            # source-0 trajectories before source-1 trajectories, so a
            # naive `model_seq[:n_sample_trajectories]` slice grabs only
            # source 0 — every per-run spectrum then comes back as a
            # single-condition spectrum, which is exactly the "I see only
            # one condition" bug. Pick a balanced subsample instead:
            # take roughly n_sample_trajectories / n_unique_c trajs from
            # each condition group.
            _has_cdim = bool(getattr(getattr(lit_model, "encoder", None), "condition_dim", 0))
            _cond_seq_arr = (
                np.asarray(_cond_seq) if (_has_cdim and _cond_seq is not None) else None
            )
            if _cond_seq_arr is not None:
                unique_c = np.unique(_cond_seq_arr, axis=0)
                per_group = max(1, n_sample_trajectories // len(unique_c))
                picked_idx: list[int] = []
                for cond_row in unique_c:
                    mask = np.all(_cond_seq_arr == cond_row, axis=1)
                    g_idx = np.where(mask)[0][:per_group].tolist()
                    picked_idx.extend(g_idx)
                # Index on CPU (model_seq is a CPU tensor) THEN move the
                # selected trajectories to device. Indexing CPU tensors
                # with a CUDA index raises "indices should be either on
                # cpu or on the same device as the indexed tensor".
                picked_idx_cpu = torch.tensor(picked_idx, dtype=torch.long)
                test_trajs_this_run = model_seq[picked_idx_cpu].to(device)
                _cond_for_run = torch.as_tensor(
                    _cond_seq_arr[picked_idx]
                ).float().to(device)
            else:
                test_trajs_this_run = model_seq[:n_sample_trajectories].to(device)
                _cond_for_run = None

            # Compute Jacobians along the test trajectories (chunked). Vanilla
            # JacobianODE (LitMLP) has no encoder — operate directly on the
            # observation-space trajectory. Latent models route through the
            # encoder + dyn-subspace split first.
            is_vanilla = not hasattr(lit_model, "encode_trajectory")
            lambdas = []
            with torch.no_grad():
                for start in range(0, test_trajs_this_run.shape[0], chunk_size):
                    chunk = test_trajs_this_run[start : start + chunk_size]
                    c_chunk = (
                        _cond_for_run[start : start + chunk_size]
                        if _cond_for_run is not None else None
                    )
                    if is_vanilla:
                        jacs = lit_model.compute_jacobians(chunk, c=c_chunk)  # (B, T, D, D)
                    else:
                        z_full = (
                            lit_model.encode_trajectory(chunk, c_chunk)
                            if c_chunk is not None
                            else lit_model.encode_trajectory(chunk)
                        )
                        mu_dyn, _ = lit_model._split_latent(z_full)
                        jacs = lit_model.compute_jacobians(mu_dyn, c=c_chunk)  # (B, T, D, D)
                        del z_full, mu_dyn
                    lams = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt)
                    lambdas.append(lams.detach().cpu())
                    del jacs
            lambda_per_traj = torch.cat(lambdas, dim=0).numpy()  # (B, D)
            lambda_mean = lambda_per_traj.mean(axis=0)            # (D,)
            lambda_max = float(lambda_mean.max())
            lambda_sum = float(lambda_mean.sum())
            # Kaplan-Yorke (rough): largest k such that sum of top-k >= 0
            cum = np.cumsum(lambda_mean)
            k_star = int(np.searchsorted(-cum, 0))  # first index where cum crosses 0
            ky_dim = None
            if 0 < k_star < len(lambda_mean):
                # fractional correction
                ky_dim = float(k_star + cum[k_star - 1] / max(abs(lambda_mean[k_star]), 1e-12))

            # Per-condition spectra: for conditioned models, group
            # lambda_per_traj rows by the per-trajectory condition value
            # so the per-run plot can show one spectrum per (run, condition).
            # Otherwise empty.
            per_condition: dict[str, dict] = {}
            if _cond_for_run is not None:
                cond_np = _cond_for_run.cpu().numpy()
                # Take only the rows that actually contributed to lambda_per_traj
                cond_used = cond_np[: lambda_per_traj.shape[0]]
                unique_c = np.unique(cond_used, axis=0)
                for cond_row in unique_c:
                    mask = np.all(cond_used == cond_row, axis=1)
                    if not mask.any():
                        continue
                    sub = lambda_per_traj[mask]
                    sub_mean = sub.mean(axis=0)
                    label = f"c={cond_row.tolist()}"
                    per_condition[label] = {
                        "lambda_spectrum": sub_mean.tolist(),
                        "lambda_max": float(sub_mean.max()),
                        "lambda_sum": float(sub_mean.sum()),
                        "n_trajs": int(mask.sum()),
                    }
            per_run[run_id] = {
                "lambda_spectrum": lambda_mean.tolist(),
                "lambda_max": lambda_max,
                "lambda_sum": lambda_sum,
                "kaplan_yorke_dim": ky_dim,
                "per_condition": per_condition,
                "error": None,
            }
            if per_condition:
                cond_summary = "  ".join(
                    f"{lbl} λ_max={d['lambda_max']:.4f}" for lbl, d in per_condition.items()
                )
                logger.info(f"  [{i + 1}/{len(all_runs)}] {run_id}: combined λ_max={lambda_max:.4f}  | per-cond: {cond_summary}")
            else:
                logger.info(f"  [{i + 1}/{len(all_runs)}] {run_id}: λ_max={lambda_max:.4f}, sum={lambda_sum:.2f}")
        except Exception as e:
            per_run[run_id] = {"error": f"{type(e).__name__}: {e}"}
            logger.warning(f"  [{i + 1}/{len(all_runs)}] {run_id}: FAILED — {e}")
        finally:
            # Free GPU memory between runs
            try:
                del lit_model
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Cross-run plots ---
    # Pair per-run Lyapunov with trajectory val loss (from wandb summary).
    success = [(rid, d) for rid, d in per_run.items() if d.get("error") is None]
    if not success:
        return per_run

    cfgs = {r.id: dict(r.config) for r in all_runs}
    summaries = {r.id: dict(r.summary) for r in all_runs}

    # ---- Detect swept keys + map wandb run -> run_idx -------------------
    # Swept keys: config paths with >1 unique value across successful runs.
    # Pulled from the sentinel's overrides_template when available, else
    # discovered empirically by diffing cfgs.
    swept_keys: list[str] = []
    if expected_snapshot:
        tpl = expected_snapshot.get("hydra", {}).get("overrides_template", []) or []
        for ov in tpl:
            if "=" not in ov:
                continue
            k, v = ov.split("=", 1)
            if "," in v and k != "experiment":
                swept_keys.append(k)
    if not swept_keys:
        # Fallback: discover from cfgs (any key whose value varies across runs)
        from collections import defaultdict
        seen: dict[str, set] = defaultdict(set)
        # Walk only the leaves we already know about — the sweep grids in
        # this project all live under training.lightning.* and data.*.
        candidate_paths = [
            "training.lightning.loop_closure_weight",
            "training.lightning.obs_noise_scale",
            "training.lightning.latent_prediction_loss_weight",
            "training.lightning.reconstruction_loss_weight",
            "training.lightning.loss_func",
            "model.kl_null_weight",
            "model.kl_dyn_weight",
        ]
        for rid, _ in success:
            for p in candidate_paths:
                seen[p].add(repr(_nested_get(cfgs.get(rid, {}), p)))
        swept_keys = [p for p, vs in seen.items() if len(vs) > 1]

    # Map wandb run id -> run_idx via the sentinel's resolved_runs (when
    # available). Falls back to enumeration order.
    rid_to_idx: dict[str, int | None] = {}
    if expected_snapshot:
        try:
            from .monitor import match_run_to_idx
            resolved_runs = expected_snapshot.get("hydra", {}).get("resolved_runs", [])
            for rid, _ in success:
                rid_to_idx[rid] = match_run_to_idx(cfgs.get(rid, {}), resolved_runs)
        except Exception as e:
            logger.warning(f"run_idx matching failed: {e}")
    for i, (rid, _) in enumerate(success):
        rid_to_idx.setdefault(rid, None)

    def _short_param_label(key: str) -> str:
        # Last segment of the dotted path, sans common prefixes.
        return key.rsplit(".", 1)[-1]

    def _fmt_param_value(v):
        f = _to_float_or_none(v)
        if f is None:
            return str(v)
        if f == 0:
            return "0"
        if abs(f) < 1e-2 or abs(f) >= 1e3:
            return f"{f:.0e}"
        return f"{f:g}"

    def _per_run_title(rid: str) -> str:
        # Always include the short wandb run id so the subplot is
        # unambiguously traceable back to wandb; add Hydra run_idx when
        # available (matches sentinel's resolved_runs).
        idx = rid_to_idx.get(rid)
        head = f"idx={idx} | {rid[:8]}" if idx is not None else rid[:8]
        cfg = cfgs.get(rid, {})
        parts = [head]
        for k in swept_keys:
            v = _nested_get(cfg, k)
            parts.append(f"{_short_param_label(k)}={_fmt_param_value(v)}")
        return "  ".join(parts)
    # Preference: user-provided literature values > computed empirical spectrum.
    if true_lyapunov:
        true_arr = np.array(true_lyapunov, dtype=float)
        true_label = "True spectrum (literature)"
    elif empirical_mean is not None:
        true_arr = np.asarray(empirical_mean)
        true_label = "Empirical (ground-truth Jacobian)"
    else:
        true_arr = None
        true_label = None

    # ------- Figure 1: overlay + lambda_max vs val loss scatter -------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))

        # Color each spectrum by its run's LC weight (log scale) so the plot
        # stays legible without a 27-entry legend.
        lc_vals = []
        for rid, _ in success:
            lc = _nested_get(cfgs.get(rid, {}), "training.lightning.loop_closure_weight")
            lc_vals.append(_to_float_or_none(lc) or 0.0)
        lc_for_color = np.array([max(v, 1e-12) for v in lc_vals])
        norm = matplotlib.colors.LogNorm(vmin=lc_for_color.min(), vmax=lc_for_color.max())
        cmap = plt.cm.viridis
        # If runs carry per-condition spectra (combined-loader runs), draw
        # one curve per (run × condition) using a different linestyle per
        # condition. Otherwise one curve per run as before.
        any_per_cond = any(d.get("per_condition") for _, d in success)
        cond_linestyles = ["-", "--", ":", "-."]
        cond_label_to_style: dict[str, str] = {}
        for (rid, d), lc in zip(success, lc_vals):
            color = cmap(norm(max(lc, 1e-12)))
            per_cond = d.get("per_condition") or {}
            if per_cond:
                for cond_label, cond_d in per_cond.items():
                    if cond_label not in cond_label_to_style:
                        cond_label_to_style[cond_label] = cond_linestyles[
                            len(cond_label_to_style) % len(cond_linestyles)
                        ]
                    ax[0].plot(
                        cond_d["lambda_spectrum"],
                        color=color,
                        linestyle=cond_label_to_style[cond_label],
                        alpha=0.6, lw=1.0,
                    )
            else:
                ax[0].plot(
                    d["lambda_spectrum"],
                    color=color,
                    alpha=0.6, lw=1.0,
                )
        # Per-condition empirical: when present, draw one black curve per
        # condition using the SAME linestyle convention as the predicted
        # spectra above. The two pairs visually align (pred c=-1 with emp
        # c=-1, both solid; pred c=+1 with emp c=+1, both dashed) so the
        # reader can see "for this condition, how close is pred to emp".
        if empirical_per_condition:
            for cond_label, emp_spec in empirical_per_condition.items():
                ls = cond_label_to_style.get(cond_label, "-")
                ax[0].plot(emp_spec, color="black", linestyle=ls, lw=2.5,
                           label=f"empirical {cond_label}", zorder=10)
        elif true_arr is not None:
            ax[0].plot(true_arr, color="black", lw=2.5, label=true_label, zorder=10)
        # Add a linestyle legend for conditions when present (separate from
        # the LC colorbar — colors carry LC weight, linestyles carry condition).
        if cond_label_to_style:
            from matplotlib.lines import Line2D
            cond_handles = [
                Line2D([0], [0], color="k", linestyle=ls, lw=1.0, label=f"pred {lbl}")
                for lbl, ls in cond_label_to_style.items()
            ]
            if empirical_per_condition:
                for cond_label in cond_label_to_style:
                    if cond_label in empirical_per_condition:
                        cond_handles.append(Line2D(
                            [0], [0], color="black", lw=2.5,
                            linestyle=cond_label_to_style[cond_label],
                            label=f"empirical {cond_label}",
                        ))
            elif true_arr is not None:
                cond_handles.append(Line2D([0], [0], color="black", lw=2.5, label=true_label))
            ax[0].legend(handles=cond_handles, loc="upper right", fontsize=8)
        elif true_arr is not None:
            ax[0].legend(loc="upper right")
        ax[0].axhline(0, color="k", lw=0.5, ls="--")
        ax[0].set_xlabel("Index (sorted desc)")
        ax[0].set_ylabel(r"$\lambda_i$")
        n_curves = sum(len(d.get("per_condition") or {}) or 1 for _, d in success)
        title_extra = f"  ({n_curves} curves over {len(success)} runs × conditions)" if any_per_cond else ""
        ax[0].set_title(f"Lyapunov spectra, {len(success)} runs  (color = LC weight){title_extra}")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        plt.colorbar(sm, ax=ax[0], label="LC weight", fraction=0.04, pad=0.04)

        # Right panel: λ_max vs trajectory val loss
        xs, ys, cs = [], [], []
        for rid, d in success:
            s = summaries.get(rid, {})
            tl = s.get("val/trajectory_loss") or s.get("trajectory val_loss")
            if tl is None:
                continue
            xs.append(d["lambda_max"])
            ys.append(float(tl))
            cs.append(_to_float_or_none(_nested_get(cfgs.get(rid, {}), "training.lightning.loop_closure_weight")) or 0.0)
        if xs:
            sc = ax[1].scatter(xs, ys, c=np.array(cs) + 1e-12, norm=norm, cmap=cmap, s=30)
            ax[1].set_xlabel(r"$\lambda_{\max}$ (predicted)")
            ax[1].set_ylabel("trajectory val loss")
            ax[1].set_yscale("log")
            ax[1].set_title("Leading Lyap. vs val loss")
            plt.colorbar(sc, ax=ax[1], label="LC weight", fraction=0.04, pad=0.04)
        fig.tight_layout()
        path = figures_dir / "per_run_lyapunov.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {path}")
    except Exception as e:
        logger.exception(f"per_run_lyapunov summary plot failed: {e}")

    # ------- Figures 2 & 3: only if we have ground-truth Lyapunov -------
    if true_arr is not None:
        # Figure 2: small-multiples grid, one subplot per run, pred vs true
        try:
            n = len(success)
            ncol = 4
            nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow), squeeze=False
            )
            # Linestyle / color convention shared across all subplots:
            # condition selects linestyle (same as the summary plot),
            # pred = blue, empirical = black.
            cond_ls_map = {
                lbl: ls
                for lbl, ls in zip(
                    sorted(empirical_per_condition.keys()) if empirical_per_condition
                    else (sorted({k for _, d in success for k in (d.get("per_condition") or {})})),
                    ["-", "--", ":", "-."],
                )
            }
            for i, (rid, d) in enumerate(success):
                row, col = i // ncol, i % ncol
                a = axes[row][col]
                lit_label = "literature" if true_lyapunov else "empirical"
                # Empirical: per-condition when available, else single curve.
                if empirical_per_condition:
                    for cond_label, emp_spec in empirical_per_condition.items():
                        ls = cond_ls_map.get(cond_label, "-")
                        a.plot(emp_spec, color="black", lw=1.5, linestyle=ls,
                               label=f"emp {cond_label}")
                else:
                    a.plot(true_arr, "k-", lw=1.5, label=lit_label)
                # Predicted: per-condition when this run was trained with c.
                per_cond_pred = d.get("per_condition") or {}
                if per_cond_pred:
                    for cond_label, cond_d in per_cond_pred.items():
                        ls = cond_ls_map.get(cond_label, "-")
                        a.plot(np.array(cond_d["lambda_spectrum"]),
                               color="C0", lw=1.0, ms=2, marker="o",
                               linestyle=ls, label=f"pred {cond_label}")
                else:
                    pred = np.array(d["lambda_spectrum"])
                    a.plot(pred, "C0o-", lw=1, ms=3, label="pred")
                a.axhline(0, color="gray", lw=0.5, ls="--")
                s = summaries.get(rid, {})
                tl = s.get("val/trajectory_loss") or s.get("trajectory val_loss")
                title = _per_run_title(rid)
                if tl is not None:
                    title += f"\ntraj_loss={float(tl):.4f}"
                a.set_title(title, fontsize=11)
                a.tick_params(labelsize=10)
                if i == 0:
                    a.legend(fontsize=8, loc="upper right")
            # Hide unused axes
            for j in range(n, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle(
                "Per-run Lyapunov spectrum: predicted vs true",
                y=1.01, fontsize=14,
            )
            fig.tight_layout()
            path = figures_dir / "per_run_lyapunov_vs_true.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Wrote {path}")
        except Exception as e:
            logger.exception(f"per_run_lyapunov_vs_true grid failed: {e}")

        # Figure 2b: per-run relative error per Lyapunov exponent
        # For each run, show signed relative error of predicted vs true spectrum,
        # as a bar chart indexed by exponent. Raw |λ_true| denominator — no
        # epsilon floor. For empirically-near-zero exponents (e.g. Lorenz's
        # λ₂ ≈ 0) a small absolute error becomes a huge relative error, which
        # is the faithful representation: the prediction is genuinely off by
        # a large fraction of the true (tiny) value.
        try:
            n = len(success)
            ncol = 4
            nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow), squeeze=False
            )
            L_true = len(true_arr)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.abs(true_arr)
            # Figure out full spectrum length (max over all runs). Positions
            # beyond L_true are shown with no comparable true value, so the
            # rel-err entries for those indices are N/A — matches the existing
            # handling for true≈0 where denom is undefined.
            L_pred = max((len(d["lambda_spectrum"]) for _, d in success), default=L_true)
            x_idx = np.arange(L_pred)
            y_clip = 2000.0
            for i, (rid, d) in enumerate(success):
                row, col = i // ncol, i % ncol
                a = axes[row][col]
                pred_full = np.array(d["lambda_spectrum"])
                rel_err = np.full(L_pred, np.nan)
                n_compare = min(L_true, len(pred_full))
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err[:n_compare] = (
                        100.0 * (pred_full[:n_compare] - true_arr[:n_compare])
                        / denom[:n_compare]
                    )
                rel_err_finite = np.where(np.isfinite(rel_err), rel_err, np.nan)
                colors = ["C0" if (np.isnan(e) or e <= 0) else "C3" for e in rel_err_finite]
                clipped = np.clip(rel_err_finite, -y_clip, y_clip)
                a.bar(x_idx, np.nan_to_num(clipped, nan=0.0), color=colors, alpha=0.85)
                a.axhline(0, color="gray", lw=0.7)
                for xi, v in zip(x_idx, rel_err):
                    if not np.isfinite(v):
                        a.text(xi, 0, "N/A", ha="center", va="center", fontsize=9, color="gray")
                        continue
                    a.text(
                        xi,
                        np.clip(v, -y_clip * 0.95, y_clip * 0.95),
                        f"{v:+.0f}%",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=9,
                    )
                a.set_xticks(x_idx)
                a.set_xlabel(r"$\lambda$ index")
                a.set_ylabel("rel. err (%)")
                a.set_ylim(-y_clip, y_clip)
                s = summaries.get(rid, {})
                tl = s.get("val/trajectory_loss") or s.get("trajectory val_loss")
                title = _per_run_title(rid)
                if tl is not None:
                    title += f"\ntraj_loss={float(tl):.4f}"
                a.set_title(title, fontsize=11)
                a.tick_params(labelsize=10)
            for j in range(n, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle(
                "Per-run Lyapunov relative error (pred vs "
                f"{'literature' if true_lyapunov else 'empirical'})",
                y=1.01, fontsize=14,
            )
            fig.tight_layout()
            path = figures_dir / "per_run_lyapunov_relerr.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Wrote {path}")
        except Exception as e:
            logger.exception(f"per_run_lyapunov_relerr grid failed: {e}")

        # Figure 3: scatter of spectrum MSE vs trajectory val loss
        try:
            xs, ys, cs, rids = [], [], [], []
            L = len(true_arr)
            for rid, d in success:
                s = summaries.get(rid, {})
                tl = s.get("val/trajectory_loss") or s.get("trajectory val_loss")
                if tl is None:
                    continue
                pred = np.array(d["lambda_spectrum"])[:L]
                spec_mse = float(np.mean((pred - true_arr[:len(pred)]) ** 2))
                xs.append(float(tl))
                ys.append(spec_mse)
                cs.append(_to_float_or_none(_nested_get(cfgs.get(rid, {}), "training.lightning.loop_closure_weight")) or 0.0)
                rids.append(rid)
                # Stash spec_mse back into per_run for the report
                per_run[rid]["spectrum_mse_vs_true"] = spec_mse
            if xs:
                fig, a = plt.subplots(figsize=(7, 5))
                sc = a.scatter(xs, ys, c=np.array(cs) + 1e-12,
                               norm=matplotlib.colors.LogNorm(),
                               cmap="viridis", s=36)
                a.set_xlabel("trajectory val loss")
                a.set_ylabel("Lyapunov spectrum MSE (pred vs true)")
                a.set_xscale("log"); a.set_yscale("log")
                a.set_title("Spectrum recovery quality vs prediction loss")
                plt.colorbar(sc, ax=a, label="LC weight")
                fig.tight_layout()
                path = figures_dir / "lyapunov_spectrum_mse_vs_val_loss.png"
                fig.savefig(path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Wrote {path}")
        except Exception as e:
            logger.exception(f"spectrum_mse vs val_loss failed: {e}")

    return {
        "per_run": per_run,
        "empirical_mean": empirical_mean.tolist() if empirical_mean is not None else None,
        "empirical_per_traj_shape": list(empirical_per_traj.shape) if empirical_per_traj is not None else None,
        "empirical_per_condition": {
            k: v.tolist() for k, v in empirical_per_condition.items()
        } if empirical_per_condition else None,
    }


def compute_per_run_tangent_spectrum(
    wandb_entity: str,
    wandb_project: str,
    group: str,
    save_dir: Path,
    output_dir: Path,
    *,
    metrics_summary: dict | None = None,
    n_sample_trajectories: int = 8,
    n_pair_samples: int = 512,
    expected_intrinsic_dim: int = 3,
) -> dict[str, Any]:
    """Compute the ranked tangent-direction spectrum for every run in the sweep.

    For each run with a checkpoint: load the model, encode a sample of test
    trajectories, and call ``lit_model.compute_tangent_spectrum`` to get the
    ranked per-direction energy of latent tangents (z_{t+1} - z_t) projected
    onto the encoder Jacobian. Hopefully concentrates on the top
    ``expected_intrinsic_dim`` components for a system whose underlying
    attractor is that-dim (e.g. 3 for Lorenz).

    Produces a two-panel figure ``per_run_tangent_spectrum.png``: per-direction
    energy (log-y) and cumulative fraction (linear-y), one curve per run,
    with legend labels derived from each run's swept_config when available.

    Returns a dict mapping run_id -> {energy, spectrum, n_pairs, error}.
    """
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb as _wandb

    from ..checkpoints.loader import load_run, load_checkpoint

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    api = _wandb.Api()
    raw_runs = list(api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters={"group": group},
    ))
    ckpt_base = Path(save_dir) / wandb_project
    all_runs = [r for r in raw_runs if (ckpt_base / r.id / "checkpoints").is_dir()]
    dropped = len(raw_runs) - len(all_runs)
    logger.info(
        f"Computing per-run tangent spectrum for {len(all_runs)} runs with checkpoints "
        f"({dropped} skipped for missing checkpoint dir)"
    )

    # Map run_id -> swept_config for legend labels (if metrics_summary
    # was provided). Falls back to run_id alone when unavailable.
    swept_by_rid: dict[str, dict] = {}
    if metrics_summary is not None:
        for entry in metrics_summary.get("per_run", []) or []:
            rid = entry.get("run_id")
            sc = entry.get("swept_config") or {}
            if rid:
                swept_by_rid[rid] = sc

    per_run: dict[str, Any] = {}
    for i, run in enumerate(all_runs):
        run_id = run.id
        try:
            loaded = load_run(
                f"{wandb_entity}/{wandb_project}",
                run_id=run_id,
                save_dir=str(save_dir),
                generate_data=True,
                verbose=False,
                return_full_obs=False,
            )
            run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded
            if not hasattr(lit_model, "compute_tangent_spectrum"):
                per_run[run_id] = {"error": "model has no compute_tangent_spectrum"}
                continue

            # compute_tangent_spectrum's internal vmap+jacrev path doesn't yet
            # accept a per-sample c. Fixing it requires per-sample-c-aware
            # vmap'd Jacobians inside _encoder_jacobian_at — same TODO as the
            # chosen-run section. Skip with a clear marker (NOT a generic
            # exception swallow) so the figure cleanly shows "skipped" runs.
            if bool(getattr(getattr(lit_model, "encoder", None), "condition_dim", 0)):
                per_run[run_id] = {
                    "error": "skipped: tangent_spectrum doesn't yet thread c"
                }
                continue

            load_checkpoint(run_obj, cfg, lit_model, save_dir=str(save_dir), verbose=False)
            lit_model = lit_model.to(device).eval()

            if trajs is None or "test_trajs" not in trajs:
                raise RuntimeError(f"No test_trajs from load_run for {run_id}")
            ts_batch = trajs["test_trajs"].sequence
            if ts_batch.shape[0] > n_sample_trajectories:
                ts_batch = ts_batch[:n_sample_trajectories]
            ts_batch = ts_batch.to(device)

            result = lit_model.compute_tangent_spectrum(ts_batch, n_samples=n_pair_samples)
            per_run[run_id] = {
                "energy": result["energy"].cpu().numpy().tolist(),
                "spectrum": result["spectrum"].cpu().numpy().tolist(),
                "n_pairs": int(result["n_pairs"]),
                "n_dyn": int(result["n_dyn"]),
                "n_obs": int(result["n_obs"]),
                "error": None,
            }
            logger.info(
                f"  [{i + 1}/{len(all_runs)}] {run_id}: "
                f"K={result['n_dyn']}, top-3 cum frac="
                f"{float(np.cumsum(result['spectrum'].cpu().numpy())[2]):.4f}"
                if result["n_dyn"] >= 3 else
                f"  [{i + 1}/{len(all_runs)}] {run_id}: K={result['n_dyn']}"
            )
        except Exception as e:
            per_run[run_id] = {"error": f"{type(e).__name__}: {e}"}
            logger.warning(f"  [{i + 1}/{len(all_runs)}] {run_id}: FAILED — {e}")
        finally:
            try:
                del lit_model
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Overlay plot ----
    success = [(rid, d) for rid, d in per_run.items() if d.get("error") is None]
    if not success:
        logger.warning("per_run_tangent_spectrum: no successful runs to plot")
        return {"per_run": per_run}

    def _short_label(rid: str) -> str:
        sc = swept_by_rid.get(rid, {})
        if not sc:
            return rid
        bits = []
        for k, v in sc.items():
            short = k.rsplit(".", 1)[-1]
            if isinstance(v, float):
                bits.append(f"{short}={v:.0e}" if v != 0 else f"{short}=0")
            else:
                bits.append(f"{short}={v}")
        return " ".join(bits)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        n = len(success)
        cmap = plt.cm.tab20 if n > 10 else plt.cm.tab10
        for i, (rid, d) in enumerate(success):
            E = np.asarray(d["energy"])
            p = np.asarray(d["spectrum"])
            x = np.arange(1, len(E) + 1)
            color = cmap(i % cmap.N)
            label = _short_label(rid)
            axes[0].plot(x, np.maximum(E, 1e-20), marker="o", ms=3,
                         lw=1.2, color=color, label=label, alpha=0.85)
            axes[1].plot(x, np.cumsum(p), marker="o", ms=3,
                         lw=1.2, color=color, label=label, alpha=0.85)

        axes[0].set_yscale("log")
        axes[0].set_xlabel("Ranked tangent direction (1-indexed)")
        axes[0].set_ylabel("Mean squared projection (energy)")
        axes[0].set_title(f"Per-direction energy ({n} runs)")
        axes[0].grid(True, which="both", alpha=0.3)

        axes[1].axvline(
            expected_intrinsic_dim, color="r", ls="--", lw=1,
            label=f"expected intrinsic dim = {expected_intrinsic_dim}",
        )
        axes[1].set_ylim(0, 1.02)
        axes[1].set_xlabel("Ranked tangent direction (1-indexed)")
        axes[1].set_ylabel("Cumulative fraction of energy")
        axes[1].set_title("Cumulative spectrum")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="lower right", fontsize=7, ncol=2 if n > 10 else 1)

        fig.tight_layout()
        path = figures_dir / "per_run_tangent_spectrum.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {path}")
    except Exception as e:
        logger.exception(f"per_run_tangent_spectrum overlay plot failed: {e}")

    return {"per_run": per_run}


def write_context(sentinel: dict, output_dir: Path) -> None:
    """Write context.json — everything the report writer needs from the sentinel."""
    expected = sentinel.get("expected_snapshot", {})
    context = {
        "group": sentinel.get("group"),
        "completed_at": sentinel.get("completed_at"),
        "outcome": sentinel.get("outcome"),
        "experiment_metadata": expected.get("experiment_metadata", {}),
        "expected_run_count": expected.get("expected_run_count"),
        "hydra": expected.get("hydra"),
        "wandb": expected.get("wandb"),
        "git": expected.get("git"),
        "launched_at": expected.get("launched_at"),
    }
    (output_dir / "context.json").write_text(json.dumps(context, indent=2) + "\n")


def analyze(
    group: str, sweeps_dir: Path, save_dir: Path,
    true_lyapunov: list | None = None,
    lyapunov_burn_in_steps: int = 400,
    lyapunov_burn_in_drop: int = 100,
    eigenvalue_threshold: float = float("inf"),
) -> Path:
    """Top-level: analyse one sweep's sentinel → produce analysis/<group>/."""
    # Accept the sentinel from any of done/, processed/, or failed/. Auto-analyze
    # or a prior run may have already moved it; the sentinel content is the same
    # regardless and re-running the analysis shouldn't require manually relocating it.
    done_path = None
    for sub in ("done", "processed", "failed"):
        cand = sweeps_dir / sub / f"{group}.done.json"
        if cand.is_file():
            done_path = cand
            break
    if done_path is None:
        raise FileNotFoundError(
            f"No sentinel found for group '{group}' in done/, processed/, or failed/"
        )

    sentinel = json.loads(done_path.read_text())
    wandb_info = sentinel.get("expected_snapshot", {}).get("wandb", {})
    wandb_entity = wandb_info.get("entity")
    wandb_project = wandb_info.get("project")
    if not (wandb_entity and wandb_project):
        raise ValueError(f"Sentinel {done_path} missing wandb.entity / wandb.project")

    output_dir = sweeps_dir / "analysis" / group
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing sweep '{group}' -> {output_dir}")

    # Copy context from sentinel FIRST so Claude has something to work with even
    # if run_analytics later crashes.
    write_context(sentinel, output_dir)

    # Pull per-run metrics straight from wandb (cheap, no model load).
    # Pass the sentinel's resolved_runs so we can match wandb runs back to
    # expected run_idx slots and (a) dedupe retries + orphans per slot,
    # (b) surface anything that fails to match in ``unmatched_runs``.
    resolved_runs = (
        sentinel.get("expected_snapshot", {}).get("hydra", {}).get("resolved_runs", [])
    )
    metrics_summary = summarize_sweep_from_wandb(
        wandb_entity, wandb_project, group, resolved_runs=resolved_runs,
    )

    # Evaluate success criteria using the compact metrics summary.
    criteria = (
        sentinel.get("expected_snapshot", {})
        .get("experiment_metadata", {})
        .values()
    )
    all_criteria: list[str] = []
    for m in criteria:
        sc = m.get("success_criteria") or []
        if isinstance(sc, list):
            all_criteria.extend(sc)
        elif isinstance(sc, str):
            all_criteria.append(sc)
    verdicts = evaluate_success_criteria(all_criteria, metrics_summary)

    # Run the heavy analytics (may load a model and compute Lyapunov exponents).
    figure_map = {}
    analytics_log = ""
    analytics_error = None
    try:
        figure_map, analytics_log = run_full_analytics(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group=group,
            save_dir=save_dir,
            output_dir=output_dir,
            true_lyapunov=true_lyapunov,
            lyapunov_burn_in_steps=lyapunov_burn_in_steps,
            lyapunov_burn_in_drop=lyapunov_burn_in_drop,
            eigenvalue_threshold=eigenvalue_threshold,
        )
    except Exception as e:
        analytics_error = f"{type(e).__name__}: {e}"
        logger.exception("run_analytics raised")

    # Recover any figures that were written to disk before the exception.
    # run_analytics emits each section's PNG immediately via _emit, so a
    # mid-loop OOM (e.g. encoder_decoder_jacobians' vmap'd jacrev blowing
    # up) would lose only the in-memory figure_map, not the on-disk PNGs.
    # Scan the figures dir and merge anything not already in figure_map so
    # report.md / dashboard get the full set of what actually rendered.
    figures_dir_local = output_dir / "figures"
    if figures_dir_local.is_dir():
        for png in figures_dir_local.glob("*.png"):
            section_name = png.stem
            if section_name not in figure_map:
                figure_map[section_name] = str(png)

    # Per-run Lyapunov spectra (across the whole sweep). Expensive (~10-20s per
    # run) but valuable — gives both the spectrum overlay and λ_max vs val_loss
    # scatter for identifying runs with unphysical dynamics.
    # Skip for encoder-only runs: no trained dynamics MLP, nothing to
    # compute eigenvalues of. Detect via any run's wandb config.
    per_run_lyapunov = {}
    empirical_lyapunov_mean = None
    per_run_lyap_error = None
    _skip_per_run_lyap = False
    try:
        import wandb as _wandb
        _api = _wandb.Api()
        _probe = list(_api.runs(
            f"{wandb_entity}/{wandb_project}",
            filters={"group": group}, per_page=1,
        ))
        if _probe:
            _cfg0 = dict(_probe[0].config)
            if bool(_cfg0.get("model", {}).get("encoder_only_mode", False)):
                _skip_per_run_lyap = True
                logger.info(
                    "encoder_only_mode=True on this sweep — skipping per-run Lyapunov"
                )
    except Exception as _e:
        logger.debug(f"encoder_only_mode probe failed (proceeding): {_e}")

    if _skip_per_run_lyap:
        per_run_lyap_error = "skipped: encoder_only_mode"
    else:
        try:
            result = compute_per_run_lyapunov(
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                group=group,
                save_dir=save_dir,
                output_dir=output_dir,
                true_lyapunov=true_lyapunov,
                expected_snapshot=sentinel.get("expected_snapshot", {}),
            )
            per_run_lyapunov = result.get("per_run", {})
            empirical_lyapunov_mean = result.get("empirical_mean")
            # Register any produced per-run plots
            for name in (
                "per_run_lyapunov",
                "per_run_lyapunov_vs_true",
                "per_run_lyapunov_relerr",
                "lyapunov_spectrum_mse_vs_val_loss",
            ):
                p = output_dir / "figures" / f"{name}.png"
                if p.is_file():
                    figure_map[name] = str(p)
        except Exception as e:
            per_run_lyap_error = f"{type(e).__name__}: {e}"
            logger.exception("per-run Lyapunov computation raised")

    # Per-run tangent spectrum (across the whole sweep). Cheap relative to
    # Lyapunov (~1-2 s per run, just one batch through the encoder + per-pair
    # Jacobian via vmap). Runs for ANY model with an encoder, including
    # encoder-only sweeps where this is the headline diagnostic.
    per_run_tangent_spectrum = {}
    per_run_tangent_error = None
    try:
        ts_result = compute_per_run_tangent_spectrum(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group=group,
            save_dir=save_dir,
            output_dir=output_dir,
            metrics_summary=metrics_summary,
        )
        per_run_tangent_spectrum = ts_result.get("per_run", {})
        p = output_dir / "figures" / "per_run_tangent_spectrum.png"
        if p.is_file():
            figure_map["per_run_tangent_spectrum"] = str(p)
    except Exception as e:
        per_run_tangent_error = f"{type(e).__name__}: {e}"
        logger.exception("per-run tangent spectrum computation raised")

    metrics_doc = {
        # v2 adds: metrics_summary.swept_paths, metrics_summary.overall_chosen_run,
        # and per_run[*].swept_config (see summarize_sweep_from_wandb).
        # v3 adds: per_run is now one entry per run_idx (best by best_traj_loss);
        # metrics_summary.unmatched_runs, duplicate_matches, expected_run_count,
        # matched_run_count. per_run[*].run_idx.
        "schema_version": 3,
        "group": group,
        "analyzed_at": iso_now(),
        "metrics_summary": metrics_summary,
        "per_run_lyapunov": per_run_lyapunov,
        "empirical_lyapunov_spectrum": empirical_lyapunov_mean,
        "per_run_lyapunov_error": per_run_lyap_error,
        "per_run_tangent_spectrum": per_run_tangent_spectrum,
        "per_run_tangent_spectrum_error": per_run_tangent_error,
        "success_criteria_verdicts": verdicts,
        "figures": figure_map,
        "analytics_error": analytics_error,
        "true_lyapunov": true_lyapunov,
        "analytics_log_file": "run_analytics.log",  # lives next to metrics.json
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2) + "\n")
    logger.info(f"Wrote analysis -> {output_dir}")
    return output_dir


def backfill(
    group: str, sweeps_dir: Path, save_dir: Path,
    true_lyapunov: list | None = None,
    lyapunov_burn_in_steps: int = 400,
    lyapunov_burn_in_drop: int = 100,
    eigenvalue_threshold: float = float("inf"),
) -> Path:
    """Re-run only missing analytics sections for an existing report.

    Reads ``analysis/<group>/metrics.json`` to find which figure files
    already exist, computes the missing sections (by comparing against
    ``ANALYTICS_SECTIONS``), then merges the new figures into the
    existing report without re-running per-run diagnostics or
    per-run Lyapunov computation.
    """
    output_dir = sweeps_dir / "analysis" / group
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No existing metrics.json for '{group}' — run full analyze first"
        )

    metrics_doc = json.loads(metrics_path.read_text())
    existing_figures = set(metrics_doc.get("figures", {}).keys())

    # Figure out which ANALYTICS_SECTIONS would produce new figures.
    # Each section's figure key is typically the section name itself.
    missing_sections = [
        s for s in ANALYTICS_SECTIONS
        if s not in existing_figures
    ]
    if not missing_sections:
        logger.info(f"[backfill {group}] all sections present, nothing to do")
        return output_dir

    logger.info(
        f"[backfill {group}] missing sections: {missing_sections}  "
        f"(existing: {sorted(existing_figures)})"
    )

    # Read sentinel for wandb info
    done_path = None
    for sub in ("done", "processed", "failed"):
        cand = sweeps_dir / sub / f"{group}.done.json"
        if cand.is_file():
            done_path = cand
            break
    # Fall back to context.json if sentinel is gone
    if done_path is not None:
        sentinel = json.loads(done_path.read_text())
        wandb_info = sentinel.get("expected_snapshot", {}).get("wandb", {})
    else:
        ctx = json.loads((output_dir / "context.json").read_text())
        wandb_info = ctx.get("wandb", {})

    wandb_entity = wandb_info.get("entity")
    wandb_project = wandb_info.get("project")
    if not (wandb_entity and wandb_project):
        raise ValueError(f"Cannot determine wandb entity/project for backfill of {group}")

    # Run only the missing sections via run_analytics. Per-run sections
    # (Lyapunov, tangent-spectrum) are intentionally NOT re-run here — they
    # loop over every run in the sweep, are too heavy for the controller's
    # login-node inline backfill, and belong in the full sbatch path. To
    # rebuild per-run figures, move the sentinel from processed/ → done/ so
    # the controller dispatches a fresh analysis sbatch to ou_bcs_high.
    try:
        new_figures, analytics_log = run_full_analytics(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            group=group,
            save_dir=save_dir,
            output_dir=output_dir,
            true_lyapunov=true_lyapunov,
            lyapunov_burn_in_steps=lyapunov_burn_in_steps,
            lyapunov_burn_in_drop=lyapunov_burn_in_drop,
            eigenvalue_threshold=eigenvalue_threshold,
            sections_override=missing_sections,
        )
    except Exception as e:
        logger.exception(f"[backfill {group}] run_analytics raised: {e}")
        new_figures = {}
        analytics_log = ""

    # Merge new figures into existing metrics_doc
    figure_map = metrics_doc.get("figures", {})
    figure_map.update(new_figures)
    metrics_doc["figures"] = figure_map
    metrics_doc["analyzed_at"] = iso_now()
    metrics_path.write_text(json.dumps(metrics_doc, indent=2) + "\n")

    # Append analytics log
    log_path = output_dir / "run_analytics.log"
    with log_path.open("a") as f:
        f.write(f"\n\n--- backfill {iso_now()} sections={missing_sections} ---\n")
        f.write(analytics_log)

    logger.info(
        f"[backfill {group}] added {len(new_figures)} figure(s): "
        f"{sorted(new_figures.keys())}"
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("group", help="wandb group name of the completed sweep")
    parser.add_argument("--backfill", action="store_true",
                        help="Only run missing analytics sections (skip per-run "
                             "diagnostics and Lyapunov). Reads existing metrics.json "
                             "to determine what's already present.")
    parser.add_argument("--sweeps-dir", default=None)
    parser.add_argument("--save-dir", default=None,
                        help="Directory containing Lightning checkpoints")
    parser.add_argument("--true-lyapunov", default=None,
                        help="Comma-separated literature Lyapunov exponents, e.g. '0.91,0,-14.57'. "
                             "If not given, the empirical spectrum (computed from eq.jac on the "
                             "test trajectories) is used as the 'true' reference.")
    parser.add_argument("--lyapunov-burn-in-steps", type=int, default=400,
                        help="Extra integration steps appended after the real trajectory for "
                             "the batch+burn-in predicted Lyapunov variant (default: 400)")
    parser.add_argument("--lyapunov-burn-in-drop", type=int, default=100,
                        help="Initial Jacobians to drop from the Lyapunov QR so Q can converge "
                             "(default: 100)")
    parser.add_argument("--eigenvalue-threshold", type=float, default=float("inf"),
                        help="C3 threshold for fast_eigenvalue_fraction in best-run "
                             "selection. Default inf disables C3 entirely — the flat "
                             "0.001 default is nonsensical at high n_target_dims where "
                             "off-manifold contracting modes legitimately register as "
                             "'fast'. Pass 0.001 to restore the original behavior.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sweeps_dir = Path(
        args.sweeps_dir or os.environ.get("SWEEPS_DIR") or DEFAULT_SWEEPS_DIR
    )
    save_dir = Path(args.save_dir or DEFAULT_SAVE_DIR)
    true_lyapunov = None
    if args.true_lyapunov:
        true_lyapunov = [float(x) for x in args.true_lyapunov.split(",")]

    try:
        fn = backfill if args.backfill else analyze
        fn(
            args.group, sweeps_dir, save_dir,
            true_lyapunov=true_lyapunov,
            lyapunov_burn_in_steps=args.lyapunov_burn_in_steps,
            lyapunov_burn_in_drop=args.lyapunov_burn_in_drop,
            eigenvalue_threshold=args.eigenvalue_threshold,
        )
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
