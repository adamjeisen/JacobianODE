"""Two-stage sweep protocol: pick survivors from a Stage-A wandb group +
dispatch Stage B with each survivor resuming from its Stage-A last.ckpt.

Usage:
    uv run --no-sync python -m JacobianODE.jacobians.tuning.two_stage_cull \\
        --group <stage_a_wandb_group> \\
        --full-max-epochs 200 \\
        --cull-fraction 0.5

Validated empirically on 71 historical sweeps: at K=20 epochs Stage A
budget + 50% cull, the eventual top-1 survives in 65/71 (92%) cases;
top-3 median survival is 3/3. See JacobianODE/perf-experiment/analysis/
two_stage_out/summary.md (separate worktree) for the replay methodology.

Two callers share this module:
  - ``trainer.py`` imports ``compute_cell_key`` and ``two_stage_ckpt_path``
    so the runtime ckpt-save side and the cull-tool reconstruction side
    produce the same hash for the same swept-cell config.
  - The CLI (``main()``) is invoked manually OR by the engaging-controller
    auto-dispatch hook after a Stage-A analysis report publishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Where Stage-A last.ckpt files live. MUST match _two_stage_ckpt_path's
# construction so the CLI can reconstruct survivor paths from wandb cfg.
TWO_STAGE_ROOT = Path(
    "/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps/two_stage_ckpts"
)

# Suffix conventions
STAGE_A_SUFFIX = "__stage_a"
DEFAULT_STAGE_B_SUFFIX = "__stage_b"

# Default cull metric (matches the 71-sweep validation).
DEFAULT_METRIC = "trajectory val_loss"


# ---------------------------------------------------------------------------
# Shared cell-key + path machinery (called from both trainer.py and CLI)
# ---------------------------------------------------------------------------

def _dotted_lookup(cfg: Any, dotted: str) -> Any:
    """Resolve a dotted-path key (e.g. 'training.lightning.loop_closure_weight')
    against a dict-like config. Returns None on any missing segment.

    Works on both DictConfig and plain dict.
    """
    cur = cfg
    for part in dotted.split("."):
        if cur is None:
            return None
        if hasattr(cur, "get"):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def _normalize_value(v: Any) -> Any:
    """Coerce numeric-looking strings to floats so that '1e-5' (Hydra
    string override) and 1e-5 (resolved float) hash to the same value.

    Bare ints stay ints; strings that don't parse as numbers stay strings.
    """
    if isinstance(v, str):
        try:
            f = float(v)
            # Preserve int-ness when the value is an integer
            if f.is_integer() and "." not in v and "e" not in v.lower():
                return int(f)
            return f
        except ValueError:
            return v
    return v


def compute_cell_key(cfg: Any) -> str:
    """Derive a stable hash that uniquely identifies a swept-grid cell.

    Reads ``cfg.sweep_grid``'s keys (the sweep declaration), then dotted-
    looks-up each key's resolved value in the full cfg. The resulting
    sorted (key, value) pairs are JSON-serialized and hashed. Determinism
    requirements:
      - dict iteration is sorted by key
      - JSON dump uses sort_keys=True
      - numeric strings normalized to floats so '1e-5' matches 1e-5
      - default=str fallback for any non-JSON-native value (eg numpy scalars)

    Returns first 16 hex chars of SHA-256 (collision-resistant for the
    grid sizes we use, 64 chars feels excessive for filesystem paths).
    """
    sg = cfg.get("sweep_grid") if hasattr(cfg, "get") else None
    if not sg:
        # No sweep_grid declared → all cells are the same. Use a fixed key.
        return "no_sweep_grid"
    pairs = sorted(
        (k, _normalize_value(_dotted_lookup(cfg, k)))
        for k in sg.keys()
    )
    blob = json.dumps(pairs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def two_stage_ckpt_path(cfg: Any) -> Path | None:
    """Stable last-ckpt path for the two-stage protocol.

    Keyed on (wandb_group, cell-key). Identical config in a different
    SLURM array hits the same path → Stage B's cull tool finds Stage A's
    last.ckpt regardless of SLURM array IDs.

    Returns None when essential cfg fields are missing — caller treats
    that as "two-stage disabled for this run".
    """
    base = _dotted_lookup(cfg, "training.logger_save_dirs")
    group = cfg.get("wandb_group") if hasattr(cfg, "get") else None
    if not base or not group:
        return None
    cell_key = compute_cell_key(cfg)
    return TWO_STAGE_ROOT / group / cell_key / "last.ckpt"


def stage_b_dispatched_marker(group: str, sweeps_dir: Path | None = None) -> Path:
    """Sentinel path used by auto-dispatch to avoid double-firing Stage B."""
    base = sweeps_dir or Path("/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps")
    return base / "active" / f"{group}.stage_b_dispatched"


# ---------------------------------------------------------------------------
# Cull algorithm
# ---------------------------------------------------------------------------

def pick_survivors(
    runs: list, metric: str, cull_fraction: float
) -> tuple[list, list[dict]]:
    """Rank runs by best-so-far ``metric`` (lower is better) and return
    the top ``(1 - cull_fraction)`` of them. Skips runs whose state is
    not 'finished' and runs without any value for the metric.

    Returns (survivors, audit) where audit is a list of dicts
    ``{run_id, state, best_metric, kept}`` for ALL ranked runs (including
    the culled and skipped ones).
    """
    audit: list[dict] = []
    scored: list[tuple[float, Any]] = []
    for r in runs:
        rid = getattr(r, "id", None) or "?"
        state = getattr(r, "state", None) or "?"
        if state != "finished":
            audit.append({"run_id": rid, "state": state,
                          "best_metric": None, "kept": False,
                          "skip_reason": f"state={state}"})
            continue
        best = math.inf
        try:
            for row in r.scan_history(keys=[metric]):
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
            audit.append({"run_id": rid, "state": state,
                          "best_metric": None, "kept": False,
                          "skip_reason": f"history_query_failed: {e}"})
            continue
        if math.isinf(best):
            audit.append({"run_id": rid, "state": state,
                          "best_metric": None, "kept": False,
                          "skip_reason": f"no_value_for_{metric}"})
            continue
        scored.append((best, r))

    scored.sort(key=lambda kv: kv[0])
    keep_n = max(1, int(round(len(scored) * (1 - cull_fraction))))
    survivors = [r for _, r in scored[:keep_n]]
    survivor_ids = {getattr(r, "id", None) for r in survivors}

    for best, r in scored:
        rid = getattr(r, "id", None) or "?"
        audit.append({"run_id": rid, "state": "finished",
                      "best_metric": float(best),
                      "kept": rid in survivor_ids,
                      "skip_reason": None})
    return survivors, audit


def swept_overrides_from_config(cfg: Any) -> list[str]:
    """Reconstruct the Hydra override CLI strings that produced this run's
    swept-grid cell. Used by Stage B to reproduce the same cell."""
    sg = cfg.get("sweep_grid") if hasattr(cfg, "get") else None
    if not sg:
        return []
    out: list[str] = []
    for key in sorted(sg.keys()):
        v = _dotted_lookup(cfg, key)
        if v is None:
            continue
        out.append(f"{key}={v}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _experiment_from_run(run) -> str | None:
    """Best-effort experiment-name lookup from a wandb run's metadata.

    The instruction-file workflow records the experiment name into
    ``cfg.metadata.experiment`` (or just ``cfg.experiment``); fall back
    to the wandb group name if neither is present (some Hydra configs
    don't surface the experiment name into the resolved cfg).

    NOTE: The wandb-group fallback is only safe when group name == YAML
    name. For two-stage groups (e.g. ``..._sweep__stage_a``), the group
    has a ``__stage_a`` suffix that doesn't exist as a YAML — using the
    group as experiment name then makes prepare_sweep crash with
    "No experiment YAML at ...". Prefer ``_experiment_from_local_expected``
    when running on a host that has access to the SWEEPS_DIR.
    """
    cfg = dict(getattr(run, "config", {}) or {})
    md = cfg.get("metadata") or {}
    if isinstance(md, dict) and md.get("experiment"):
        return md["experiment"]
    if cfg.get("experiment"):
        return cfg["experiment"]
    return getattr(run, "group", None)


def _experiment_from_local_expected(
    group: str, sweeps_dir: Path | None = None,
) -> str | None:
    """Authoritative experiment-name lookup from Stage A's expected.json.

    expected.json's ``hydra.resolved_runs[i].experiment`` is the actual
    YAML name that ran (set by prepare_sweep at submission time). This
    is more reliable than wandb-config heuristics, which can fall back
    to the wandb group name (which has the ``__stage_a`` suffix and
    doesn't exist as a YAML).
    """
    import os
    if sweeps_dir is None:
        sweeps_dir = Path(os.environ.get(
            "SWEEPS_DIR",
            "/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps",
        ))
    for sub in ("active", "done", "processed"):
        ep = sweeps_dir / sub / f"{group}.expected.json"
        if ep.is_file():
            try:
                doc = json.loads(ep.read_text())
                runs = doc.get("hydra", {}).get("resolved_runs", [])
                if runs:
                    exp = runs[0].get("experiment")
                    if exp:
                        return exp
            except Exception:
                pass
    return None


def _resolve_project(api, group: str, project: str | None) -> str | None:
    """If --project not given, walk the entity's projects and find the one
    that contains a run in this group. Cheap when there are few projects."""
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


def _write_and_push_instruction(
    experiment: str,
    overrides: list[str],
    run_id: str,
    migrate_to: dict | None = None,
    repo_dir: Path | None = None,
) -> None:
    """Write a sweep instruction file directly to jacobian-reports/instructions/pending
    and push. Mirrors what ~/bin/j-submit does, but lets us include
    arbitrary block-level fields like ``migrate_to:`` that the bash
    script doesn't support.

    Used by the two_stage cull tool when Stage A had ``migrate_to:`` so
    Stage B inherits it.
    """
    import datetime
    repo = repo_dir or (Path.home() / "Documents" / "jacobian-analyses"
                         if (Path.home() / "Documents" / "jacobian-analyses").is_dir()
                         else Path.home() / "code" / "jacobian-reports")
    pending = repo / "instructions" / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{ts}-{experiment}-stage-b-{run_id}.yaml"
    target = pending / fname

    lines = [f"experiment: {experiment}", "overrides:"]
    for ov in overrides:
        # Quote each override to preserve = and special chars.
        lines.append(f'  - "{ov}"')
    if migrate_to is not None:
        lines.append("migrate_to:")
        for k, v in migrate_to.items():
            lines.append(f"  {k}: {json.dumps(v)}")
    import socket
    lines.append(f"submitted_at: {ts}")
    lines.append(f"submitted_from: {socket.gethostname()}")
    target.write_text("\n".join(lines) + "\n")

    # git pull --rebase, add, commit, push
    subprocess.run(["git", "pull", "--rebase", "--autostash"],
                   cwd=repo, check=False, capture_output=True)
    subprocess.run(["git", "add", str(target)], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"submit (stage-b): {experiment} run={run_id}"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(["git", "push"], cwd=repo, check=True, capture_output=True)
    logger.info(f"  wrote + pushed {fname}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True,
                        help="Stage-A wandb_group (typically ends in __stage_a)")
    parser.add_argument("--project", default=None,
                        help="wandb project; auto-detected from one run "
                             "in the group if omitted")
    parser.add_argument("--entity", default="JacobianODE")
    parser.add_argument("--full-max-epochs", type=int, required=True,
                        help="max_epochs override for Stage B (typically the "
                             "original sweep's intended epoch budget)")
    parser.add_argument("--cull-fraction", type=float, default=0.5,
                        help="fraction of runs to cull. 0.5 = keep top half. "
                             "Empirically: 50%% cull -> 92%% top-1 survival; "
                             "33%% cull -> 100%% top-1 survival.")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help="metric to rank runs by (lower is better)")
    parser.add_argument("--stage-b-suffix", default=DEFAULT_STAGE_B_SUFFIX,
                        help="suffix appended to the original group name "
                             "for Stage B runs (default __stage_b)")
    parser.add_argument("--dry-run", action="store_true",
                        help="don't j-submit; just write survivors.json + log")
    parser.add_argument("--two-stage-root", default=str(TWO_STAGE_ROOT),
                        help="override TWO_STAGE_ROOT (testing only)")
    parser.add_argument("--audit-out", default=None,
                        help="path to write survivors.json (default: "
                             "/tmp/two_stage_survivors_<group>.json)")
    parser.add_argument("--migrate-to-json", default=None,
                        help="JSON-encoded migrate_to block to inherit into "
                             "Stage B's instruction YAML (e.g. "
                             "'{\"partition\": \"mit_normal_gpu\"}'). When "
                             "set, bypasses j-submit and writes the "
                             "instruction file directly so the block is "
                             "included.")
    parser.add_argument("--experiment", default=None,
                        help="Override experiment name. If unset, looks up "
                             "from Stage A's expected.json "
                             "(hydra.resolved_runs[0].experiment), then "
                             "falls back to wandb-config heuristics. The "
                             "explicit override is required when running "
                             "on a host without SWEEPS_DIR access.")
    args = parser.parse_args(argv)

    migrate_to_block = None
    if args.migrate_to_json:
        try:
            migrate_to_block = json.loads(args.migrate_to_json)
        except json.JSONDecodeError as e:
            logger.error(f"invalid --migrate-to-json: {e}")
            return 1
        if not isinstance(migrate_to_block, dict):
            logger.error(f"--migrate-to-json must decode to a dict, got "
                         f"{type(migrate_to_block).__name__}")
            return 1

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import wandb
    api = wandb.Api()
    project = _resolve_project(api, args.group, args.project)
    if project is None:
        logger.error(f"could not find a project containing group={args.group!r}")
        return 1
    logger.info(f"group={args.group}  project={project}  metric={args.metric}  "
                f"cull_fraction={args.cull_fraction}")

    runs = list(api.runs(f"{args.entity}/{project}",
                         filters={"group": args.group}, per_page=200))
    logger.info(f"  found {len(runs)} run(s) in group")
    if not runs:
        logger.error("no runs to cull; exiting")
        return 1

    survivors, audit = pick_survivors(runs, args.metric, args.cull_fraction)
    n_finished = sum(1 for a in audit if a["state"] == "finished")
    n_kept = sum(1 for a in audit if a.get("kept"))
    logger.info(f"  finished: {n_finished}/{len(audit)}  kept: {n_kept}")

    # Build Stage-B per-survivor instructions
    two_stage_root = Path(args.two_stage_root)
    stage_b_group = args.group
    if stage_b_group.endswith(STAGE_A_SUFFIX):
        stage_b_group = stage_b_group[:-len(STAGE_A_SUFFIX)] + args.stage_b_suffix
    else:
        stage_b_group = stage_b_group + args.stage_b_suffix

    survivors_meta: list[dict] = []
    skipped_no_ckpt: list[str] = []
    for r in survivors:
        cfg = dict(r.config or {})
        # Reconstruct cell-key from the run's resolved cfg
        cell_key = compute_cell_key(cfg)
        ckpt = two_stage_root / args.group / cell_key / "last.ckpt"
        if not ckpt.is_file():
            logger.warning(f"  no ckpt at {ckpt}; skipping {r.id}")
            skipped_no_ckpt.append(r.id)
            continue
        overrides = swept_overrides_from_config(cfg) + [
            f"training.trainer_params.max_epochs={args.full_max_epochs}",
            f"wandb_group={stage_b_group}",
            f"+training.ckpt_path={ckpt}",
        ]
        survivors_meta.append({
            "run_id": r.id, "cell_key": cell_key,
            "ckpt": str(ckpt), "overrides": overrides,
        })

    # Audit trail (kept runs + culled + skipped, with reasons)
    audit_doc = {
        "stage_a_group": args.group,
        "stage_b_group": stage_b_group,
        "metric": args.metric,
        "cull_fraction": args.cull_fraction,
        "full_max_epochs": args.full_max_epochs,
        "audit": audit,
        "survivors": survivors_meta,
        "skipped_no_ckpt": skipped_no_ckpt,
    }
    audit_path = Path(args.audit_out or f"/tmp/two_stage_survivors_{args.group}.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_doc, indent=2))
    logger.info(f"  wrote {audit_path}")

    if args.dry_run:
        logger.info("(dry-run: no j-submit fired)")
        return 0
    if not survivors_meta:
        logger.error("no survivors with ckpts to dispatch; exiting")
        return 1

    # Resolve experiment name. Priority:
    #   1. --experiment CLI override (explicit, controller passes this)
    #   2. Stage A's expected.json hydra.resolved_runs[0].experiment
    #   3. wandb-config heuristic (fallback only — can crash with
    #      "No experiment YAML at <group>.yaml" for two-stage groups)
    experiment = (
        args.experiment
        or _experiment_from_local_expected(args.group)
        or _experiment_from_run(survivors[0])
    )
    if not experiment:
        logger.error("could not infer experiment name; pass --experiment "
                     "explicitly. Aborting.")
        return 1
    logger.info(f"  dispatching {len(survivors_meta)} stage-B run(s) "
                f"under experiment={experiment}"
                + (f" with migrate_to={migrate_to_block}" if migrate_to_block else ""))
    if migrate_to_block:
        # Bypass j-submit so we can include the migrate_to: block in the
        # instruction YAML. j-submit's bash script doesn't support
        # arbitrary block-level fields.
        for s in survivors_meta:
            try:
                _write_and_push_instruction(
                    experiment, s["overrides"], s["run_id"],
                    migrate_to=migrate_to_block,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"instruction write/push failed for {s['run_id']}: {e}")
                return 1
    else:
        j_submit = Path.home() / "bin" / "j-submit"
        for s in survivors_meta:
            cmd = [str(j_submit), experiment] + s["overrides"]
            logger.info(f"  $ {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"j-submit failed for {s['run_id']}: {e}")
                return 1

    # Idempotency sentinel for engaging-controller auto-dispatch
    marker = stage_b_dispatched_marker(args.group)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "dispatched_at_iso": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "n_survivors": len(survivors_meta),
        "stage_b_group": stage_b_group,
    }, indent=2))
    logger.info(f"  wrote idempotency marker {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
