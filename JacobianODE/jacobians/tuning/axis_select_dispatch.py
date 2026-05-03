"""Scout → grid chain dispatch: pick top-K values along one axis from a
scout sweep's wandb runs, then submit a follow-up grid sweep with that
axis pinned to the chosen values.

Usage:
    uv run --no-sync python -m JacobianODE.jacobians.tuning.axis_select_dispatch \\
        --group <scout_wandb_group> \\
        --axis  data.train_test_params.delay_embedding_params.n_delays \\
        --top-k 3 \\
        --metric "trajectory val_loss" \\
        --next-experiment <grid_yaml_name> \\
        [--next-two-stage-json '{"stage_a_epochs":20,"full_max_epochs":100,"cull_fraction":0.5}'] \\
        [--next-migrate-to-json '{"partition":"mit_normal_gpu"}'] \\
        [--dry-run]

Writes ONE follow-up instruction YAML to jacobian-reports/instructions/pending/
with overrides=["<axis>=<v1,v2,...,vk>"], optionally carrying a
``two_stage:`` and/or ``migrate_to:`` block. The existing controller
flow (prepare_sweep + jsweep + reap + two_stage_cull) handles the rest.

Selection is metric-only (lower is better) — matches the convention
already used by two_stage_cull.pick_survivors and the deprecated
launch-sweep2-from-sweep1 script.

Idempotency sentinel at ``sweeps/active/<group>.chain_dispatched``;
second invocation with sentinel present is a no-op.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SWEEPS_DIR = Path(
    "/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps"
)
DEFAULT_METRIC = "trajectory val_loss"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def chain_dispatched_marker(group: str, sweeps_dir: Path | None = None) -> Path:
    """Sentinel path used by auto-dispatch to avoid double-firing axis_select."""
    base = sweeps_dir or DEFAULT_SWEEPS_DIR
    return base / "active" / f"{group}.chain_dispatched"


# ---------------------------------------------------------------------------
# Bucket-and-rank
# ---------------------------------------------------------------------------

def _dotted_lookup(d: Any, dotted: str):
    """Walk a dotted path through a nested dict/OmegaConf-like config."""
    cur = d
    for part in dotted.split("."):
        if cur is None:
            return None
        if hasattr(cur, "get"):
            cur = cur.get(part)
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _best_metric_for_run(run, metric: str) -> tuple[float | None, str | None]:
    """Return (best_value, skip_reason).

    Skip only ``state == "running"`` (training not done — best value isn't
    final). All other states (``finished``, ``crashed``, ``failed``,
    ``killed``, …) are scanned for the metric. SLURM-timeout-killed cells
    typically have hours of training behind them and a meaningful best
    loss; the metric-validity check below excludes ones with no recorded
    data. Excluding crashed runs blanket-style biases the selection
    against larger-model cells, which are the most likely to time out
    AND the most likely to be best.
    """
    state = getattr(run, "state", None) or "?"
    if state == "running":
        return None, f"state={state}"
    best = math.inf
    try:
        for row in run.scan_history(keys=[metric]):
            v = row.get(metric)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v < best:
                best = v
    except Exception as e:
        return None, f"history_query_failed: {e}"
    if math.isinf(best):
        return None, f"no_value_for_{metric}"
    return float(best), None


def bucket_and_rank(
    runs: list, axis: str, metric: str,
) -> tuple[dict, list[dict]]:
    """Bucket finished runs by ``cfg[<axis>]`` and find each bucket's
    best ``metric`` value.

    Returns (buckets, audit) where:
      - buckets: dict[axis_value (str) → {"best_metric": float,
                                           "n_runs": int,
                                           "best_run_id": str}]
      - audit: list[dict] with per-run trace including skip reasons.
    """
    audit: list[dict] = []
    bucket_best: dict[str, dict] = {}
    for r in runs:
        rid = getattr(r, "id", None) or "?"
        cfg = dict(getattr(r, "config", {}) or {})
        axis_val = _dotted_lookup(cfg, axis)
        if axis_val is None:
            audit.append({"run_id": rid, "axis_value": None,
                          "best_metric": None, "kept": False,
                          "skip_reason": f"no_value_at_axis={axis}"})
            continue
        # Bucket key as string for stable grouping (works for ints, floats,
        # strings alike); we cast back when constructing the override CSV.
        bk = str(axis_val)
        best, skip = _best_metric_for_run(r, metric)
        if best is None:
            audit.append({"run_id": rid, "axis_value": bk,
                          "best_metric": None, "kept": False,
                          "skip_reason": skip})
            continue
        cur = bucket_best.get(bk)
        if cur is None or best < cur["best_metric"]:
            bucket_best[bk] = {
                "best_metric": best,
                "best_run_id": rid,
                "n_runs": (cur["n_runs"] + 1) if cur else 1,
            }
        else:
            cur["n_runs"] += 1
        audit.append({"run_id": rid, "axis_value": bk,
                      "best_metric": best, "kept": None,  # filled later
                      "skip_reason": None})
    return bucket_best, audit


def select_top_k(
    bucket_best: dict, top_k: int,
) -> list[str]:
    """Return the top-K axis values (as stringified keys) by best metric.
    Lower metric = better. Stable: ties broken by string-sort of the
    axis-value key so output is deterministic."""
    ordered = sorted(
        bucket_best.items(),
        key=lambda kv: (kv[1]["best_metric"], kv[0]),
    )
    return [k for k, _ in ordered[:top_k]]


def _stringify_axis_value(s: str) -> str:
    """Best-effort conversion back to a clean override-friendly value.
    Strips trailing '.0' from integer-valued floats so '6.0' → '6' (matches
    how the user typed the original sweep_grid)."""
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return s


# ---------------------------------------------------------------------------
# Wandb helpers (reuse the project-discovery from two_stage_cull)
# ---------------------------------------------------------------------------

def _resolve_project(api, group: str, project: str | None) -> str | None:
    """Walk JacobianODE entity's projects to find the one containing this
    group, if --project not provided. Mirrors two_stage_cull's helper."""
    if project:
        return project
    try:
        for p in api.projects(entity="JacobianODE"):
            try:
                rr = list(api.runs(
                    f"JacobianODE/{p.name}",
                    filters={"group": group}, per_page=1,
                ))
                if rr:
                    return p.name
            except Exception:
                continue
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True,
                        help="Scout sweep's wandb_group")
    parser.add_argument("--axis", required=True,
                        help="Dotted config path to the axis being selected, "
                             "e.g. data.train_test_params.delay_embedding_params.n_delays")
    parser.add_argument("--top-k", type=int, required=True,
                        help="How many distinct axis values to pick (top-K by metric)")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help="Wandb metric to rank by (lower = better)")
    parser.add_argument("--next-experiment", required=True,
                        help="Experiment YAML name to invoke for the follow-up sweep")
    parser.add_argument("--next-two-stage-json",
                        help="JSON-encoded two_stage block for the follow-up "
                             "instruction (optional)")
    parser.add_argument("--next-migrate-to-json",
                        help="JSON-encoded migrate_to block for the follow-up "
                             "instruction (optional)")
    parser.add_argument("--project",
                        help="Wandb project (auto-detected from group if omitted)")
    parser.add_argument("--sweeps-dir", default=None)
    parser.add_argument("--audit-out", default=None,
                        help="Where to write the audit JSON; default "
                             "/tmp/axis_select_<group>.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't push the instruction, don't write the sentinel; "
                             "only print what would be done.")
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

    # 1. Idempotency sentinel
    sentinel = chain_dispatched_marker(args.group, sweeps_dir)
    if sentinel.exists() and not args.dry_run:
        logger.info(f"sentinel exists at {sentinel}; chain already dispatched. no-op.")
        return 0

    # 2. Pull scout runs from wandb
    import wandb
    api = wandb.Api()
    project = _resolve_project(api, args.group, args.project)
    if not project:
        logger.error(f"could not resolve wandb project for group {args.group!r}")
        return 1
    logger.info(f"querying JacobianODE/{project} for group={args.group!r}")
    runs = list(api.runs(f"JacobianODE/{project}",
                         filters={"group": args.group}))
    logger.info(f"  found {len(runs)} runs")

    # 3. Bucket by axis + rank by metric
    buckets, audit = bucket_and_rank(runs, args.axis, args.metric)
    if not buckets:
        logger.error(f"no rankable runs in group (all runs failed/skipped). "
                     f"audit:\n{json.dumps(audit, indent=2)}")
        return 2
    logger.info(f"  bucketed {len(buckets)} distinct axis values "
                f"(axis={args.axis})")
    for k, v in sorted(buckets.items(), key=lambda kv: kv[1]["best_metric"]):
        logger.info(f"    {args.axis}={k}  best_{args.metric}={v['best_metric']:.4e}  "
                    f"(n_runs={v['n_runs']}, best_run={v['best_run_id']})")

    # 4. Select top K
    if args.top_k < 1:
        logger.error(f"--top-k must be >=1, got {args.top_k}")
        return 1
    chosen = select_top_k(buckets, args.top_k)
    logger.info(f"  top-{args.top_k} chosen: {chosen}")

    # Mark audit with kept flag now that we know survivors
    chosen_set = set(chosen)
    for row in audit:
        if row.get("axis_value") is not None and row.get("best_metric") is not None:
            row["kept"] = (row["axis_value"] in chosen_set)

    # 5. Compose the next instruction
    csv = ",".join(_stringify_axis_value(v) for v in chosen)
    overrides = [f"{args.axis}={csv}"]
    two_stage_block = (
        json.loads(args.next_two_stage_json)
        if args.next_two_stage_json else None
    )
    migrate_to_block = (
        json.loads(args.next_migrate_to_json)
        if args.next_migrate_to_json else None
    )

    logger.info(f"next instruction:")
    logger.info(f"  experiment: {args.next_experiment}")
    logger.info(f"  overrides:  {overrides}")
    if two_stage_block:
        logger.info(f"  two_stage:  {two_stage_block}")
    if migrate_to_block:
        logger.info(f"  migrate_to: {migrate_to_block}")

    # 6. Dump audit (always, for debugging)
    audit_doc = {
        "scout_group": args.group,
        "axis": args.axis,
        "metric": args.metric,
        "top_k": args.top_k,
        "chosen_axis_values": chosen,
        "next_experiment": args.next_experiment,
        "next_two_stage": two_stage_block,
        "next_migrate_to": migrate_to_block,
        "buckets": buckets,
        "per_run_audit": audit,
    }
    audit_out = Path(
        args.audit_out
        or f"/tmp/axis_select_{args.group}.json"
    )
    audit_out.write_text(json.dumps(audit_doc, indent=2, default=str))
    logger.info(f"audit written to {audit_out}")

    if args.dry_run:
        logger.info("--dry-run: not writing instruction or sentinel.")
        return 0

    # 7. Push follow-up instruction (reuses two_stage_cull's helper)
    from .two_stage_cull import _write_and_push_instruction
    # Use the chosen-csv as a stable run_id-like suffix in the filename.
    # Truncate for filesystem safety.
    csv_tag = csv.replace(",", "-")[:32]
    _write_and_push_instruction(
        experiment=args.next_experiment,
        overrides=overrides,
        run_id=f"axissel_{csv_tag}",
        two_stage=two_stage_block,
        migrate_to=migrate_to_block,
        kind="chain",
    )

    # 8. Write sentinel (idempotency for restart-safety)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({
        "scout_group": args.group,
        "next_experiment": args.next_experiment,
        "chosen_axis_values": chosen,
    }, indent=2))
    logger.info(f"wrote sentinel: {sentinel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
