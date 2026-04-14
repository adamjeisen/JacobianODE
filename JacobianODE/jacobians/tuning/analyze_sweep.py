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
    "lyapunov",
    "prediction_windows",
    "mase",
]


def iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def summarize_sweep_from_wandb(wandb_entity: str, wandb_project: str, group: str) -> dict:
    """Pull per-run best metrics from wandb and group them by obs_noise_scale.

    Returns a dict of:
        {
            "per_run": [{lc_weight, obs_noise_scale, best_traj_loss, best_mase, ...}],
            "best_by_obs_noise_scale": {ons: {lc_weight, best_traj_loss, best_mase, ...}},
            "overall_best_mase": {run_id, lc_weight, obs_noise_scale, best_mase, ...},
        }
    """
    import wandb
    api = wandb.Api()
    runs = list(api.runs(f"{wandb_entity}/{wandb_project}", filters={"group": group}))
    per_run: list[dict[str, Any]] = []
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
        per_run.append({
            "run_id": r.id,
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
        })

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

    overall_best = min(
        (r for r in per_run if r["best_mase"] is not None),
        key=lambda r: r["best_mase"],
        default=None,
    )
    return {
        "per_run": per_run,
        "best_by_obs_noise_scale": best_by_ons,
        "overall_best_mase": overall_best,
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
) -> tuple[dict[str, str], str]:
    """Invoke run_analytics with the standard section set.

    Captures run_analytics's stdout into ``output_dir/run_analytics.log`` so
    the printed per-run diagnostics / Lyapunov tables end up in the report.

    Returns ``(figure_map, captured_stdout)`` — the figure map is
    ``{section: figure_path}`` for the PNGs saved to ``output_dir/figures/``.
    """
    import contextlib
    import io

    from ..run_analytics import run_analytics

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running run_analytics for group={group}, sections={ANALYTICS_SECTIONS}")

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
            sections=ANALYTICS_SECTIONS,
            ranking_method="best_traj_loss",
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

    best_mase = (metrics_summary.get("overall_best_mase") or {}).get("best_mase")
    best_lc_loss = (metrics_summary.get("overall_best_mase") or {}).get("lc_loss_at_best_tl")
    best_r2 = (metrics_summary.get("overall_best_mase") or {}).get("r2_at_best_tl")

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

    # List the group's runs (in wandb order)
    api = _wandb.Api()
    all_runs = list(api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters={"group": group, "state": "finished"},
    ))
    logger.info(f"Computing per-run Lyapunov for {len(all_runs)} finished runs")

    # Cache the test trajectories across runs — only the FIRST load_run call
    # actually generates them; subsequent calls pass generate_data=False.
    test_trajs_cached = None
    dt_cached = None
    per_run: dict[str, Any] = {}

    # Empirical ground-truth Lyapunov spectrum, computed once from eq.jac on
    # the test trajectories (see run_analytics.py for the canonical path).
    empirical_mean: np.ndarray | None = None
    empirical_per_traj: np.ndarray | None = None

    for i, run in enumerate(all_runs):
        run_id = run.id
        try:
            if i == 0:
                # return_full_obs=True so that partial-obs / delay-embedded
                # runs still have the underlying full state available for
                # computing the empirical Lyapunov spectrum via eq.jac.
                loaded = load_run(
                    f"{wandb_entity}/{wandb_project}",
                    run_id=run_id,
                    save_dir=str(save_dir),
                    generate_data=True,
                    verbose=False,
                    return_full_obs=True,
                )
            else:
                loaded = load_run(
                    f"{wandb_entity}/{wandb_project}",
                    run_id=run_id,
                    save_dir=str(save_dir),
                    generate_data=False,
                    dt=dt_cached,
                    verbose=False,
                )
            run_obj, cfg, eq, dt, values, _, _, _, trajs, lit_model = loaded
            if i == 0:
                dt_cached = dt
                if trajs is not None and "test_trajs" in trajs:
                    test_seq = trajs["test_trajs"].sequence
                    test_trajs_cached = test_seq[:n_sample_trajectories].to(device)

                    # Compute empirical spectrum ONCE (same data/eq across runs).
                    mu_val = cfg.data.postprocessing.get("mu", 0.0)
                    sigma_val = cfg.data.postprocessing.get("sigma", 1.0)
                    # Use FULL test set (not subsampled) for empirical since
                    # ground-truth Jacobians are cheap.
                    traj_for_emp = (
                        trajs["test_trajs_full"].sequence
                        if "test_trajs_full" in trajs
                        else trajs["test_trajs"].sequence
                    )
                    logger.info("Computing empirical ground-truth Lyapunov spectrum...")
                    empirical_mean, empirical_per_traj = _compute_empirical_lyapunov(
                        eq, traj_for_emp, dt, mu_val, sigma_val, device,
                    )
                    if empirical_mean is not None:
                        logger.info(
                            f"Empirical λ₁={empirical_mean[0]:.4f}, "
                            f"λ_min={empirical_mean[-1]:.4f}, "
                            f"Σλ={empirical_mean.sum():.3f}"
                        )

            load_checkpoint(run_obj, cfg, lit_model, save_dir=str(save_dir), verbose=False)
            lit_model = lit_model.to(device).eval()

            if test_trajs_cached is None:
                raise RuntimeError(
                    "No cached test trajectories — first run must have generate_data=True"
                )

            # Compute Jacobians along the test trajectories (chunked).
            lambdas = []
            with torch.no_grad():
                for start in range(0, test_trajs_cached.shape[0], chunk_size):
                    chunk = test_trajs_cached[start : start + chunk_size]
                    z_full = lit_model.encode_trajectory(chunk)
                    mu_dyn, _ = lit_model._split_latent(z_full)
                    jacs = lit_model.compute_jacobians(mu_dyn)  # (B, T, D, D)
                    lams = LitLatentJacobianODE.compute_lyapunov_exponents(jacs, dt)
                    lambdas.append(lams.detach().cpu())
                    del z_full, mu_dyn, jacs
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
            per_run[run_id] = {
                "lambda_spectrum": lambda_mean.tolist(),
                "lambda_max": lambda_max,
                "lambda_sum": lambda_sum,
                "kaplan_yorke_dim": ky_dim,
                "error": None,
            }
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
        idx = rid_to_idx.get(rid)
        idx_str = f"idx={idx}" if idx is not None else f"id={rid[:8]}"
        cfg = cfgs.get(rid, {})
        parts = [idx_str]
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
        for (rid, d), lc in zip(success, lc_vals):
            ax[0].plot(
                d["lambda_spectrum"],
                color=cmap(norm(max(lc, 1e-12))),
                alpha=0.6, lw=1.0,
            )
        if true_arr is not None:
            ax[0].plot(true_arr, color="black", lw=2.5, label=true_label, zorder=10)
            ax[0].legend(loc="upper right")
        ax[0].axhline(0, color="k", lw=0.5, ls="--")
        ax[0].set_xlabel("Index (sorted desc)")
        ax[0].set_ylabel(r"$\lambda_i$")
        ax[0].set_title(f"Lyapunov spectra, {len(success)} runs  (color = LC weight)")
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
            L = len(true_arr)
            for i, (rid, d) in enumerate(success):
                row, col = i // ncol, i % ncol
                a = axes[row][col]
                pred = np.array(d["lambda_spectrum"])[:L]
                lit_label = "literature" if true_lyapunov else "empirical"
                a.plot(true_arr, "k-", lw=1.5, label=lit_label)
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
                    a.legend(fontsize=10, loc="upper right")
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

        # Figure 3: scatter of spectrum MSE vs trajectory val loss
        try:
            xs, ys, cs, rids = [], [], [], []
            for rid, d in success:
                s = summaries.get(rid, {})
                tl = s.get("val/trajectory_loss") or s.get("trajectory val_loss")
                if tl is None:
                    continue
                pred = np.array(d["lambda_spectrum"])[:L]
                spec_mse = float(np.mean((pred - true_arr) ** 2))
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
    }


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
) -> Path:
    """Top-level: analyse one sweep's sentinel → produce analysis/<group>/."""
    done_path = sweeps_dir / "done" / f"{group}.done.json"
    if not done_path.is_file():
        # Also check processed/ in case we're reanalysing an already-consumed sweep
        alt = sweeps_dir / "processed" / f"{group}.done.json"
        if alt.is_file():
            done_path = alt
        else:
            raise FileNotFoundError(f"No sentinel found for group '{group}' in done/ or processed/")

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
    metrics_summary = summarize_sweep_from_wandb(wandb_entity, wandb_project, group)

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
        )
    except Exception as e:
        analytics_error = f"{type(e).__name__}: {e}"
        logger.exception("run_analytics raised")

    # Per-run Lyapunov spectra (across the whole sweep). Expensive (~10-20s per
    # run) but valuable — gives both the spectrum overlay and λ_max vs val_loss
    # scatter for identifying runs with unphysical dynamics.
    per_run_lyapunov = {}
    empirical_lyapunov_mean = None
    per_run_lyap_error = None
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
            "lyapunov_spectrum_mse_vs_val_loss",
        ):
            p = output_dir / "figures" / f"{name}.png"
            if p.is_file():
                figure_map[name] = str(p)
    except Exception as e:
        per_run_lyap_error = f"{type(e).__name__}: {e}"
        logger.exception("per-run Lyapunov computation raised")

    metrics_doc = {
        "schema_version": 1,
        "group": group,
        "analyzed_at": iso_now(),
        "metrics_summary": metrics_summary,
        "per_run_lyapunov": per_run_lyapunov,
        "empirical_lyapunov_spectrum": empirical_lyapunov_mean,
        "per_run_lyapunov_error": per_run_lyap_error,
        "success_criteria_verdicts": verdicts,
        "figures": figure_map,
        "analytics_error": analytics_error,
        "true_lyapunov": true_lyapunov,
        "analytics_log_file": "run_analytics.log",  # lives next to metrics.json
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2) + "\n")
    logger.info(f"Wrote analysis -> {output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("group", help="wandb group name of the completed sweep")
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
        analyze(
            args.group, sweeps_dir, save_dir,
            true_lyapunov=true_lyapunov,
            lyapunov_burn_in_steps=args.lyapunov_burn_in_steps,
            lyapunov_burn_in_drop=args.lyapunov_burn_in_drop,
        )
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
