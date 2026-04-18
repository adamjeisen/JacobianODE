"""Prepare a JacobianODE sweep submission.

Given a set of Hydra overrides (e.g. ``experiment=foo training.lightning.lc=0,1``),
this module:

1. Locates and loads each referenced experiment YAML.
2. Validates that each has the required ``metadata`` block (description,
   hypothesis, success_criteria). Errors early if missing.
3. Resolves the Cartesian product of grid overrides into a concrete list of
   per-run override sets (matching Hydra's multirun enumeration order).
4. Emits a JSON document suitable for writing to
   ``$SWEEPS_DIR/active/<group>.expected.json``.

This is intended to be invoked by the ``engaging-submit`` bash wrapper
BEFORE the sweep is submitted, so that missing metadata blocks or malformed
experiments abort the submission cleanly.

Usage:
    python -m JacobianODE.jacobians.tuning.prepare_sweep \\
        --git-branch latent-JacobianODE \\
        --git-commit a9074ca \\
        --git-repo-dir /home/eisenaj/code/JacobianODE \\
        --launched-from endeavour \\
        [--output expected.json] \\
        experiment=... training.lightning.loop_closure_weight=0,1e-4,1e-2 ...

The output JSON carries everything needed to (a) resubmit any failed run
and (b) provide the analysis agent with the experiment context.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
REQUIRED_META_FIELDS = ("description", "hypothesis", "success_criteria")

# Hard-coded SLURM defaults matching conf/slurm/default.yaml on engaging.
# The monitor uses these to construct resubmission sbatch commands.
DEFAULT_SLURM = {
    "timeout_min": 180,
    "partition": "ou_bcs_normal",
    "gres": "gpu:1",
    "cpus_per_task": 4,
    "mem": "16GB",
}

DEFAULT_RETRY = {
    "cap_per_run": 2,
    "min_elapsed_before_done_sec": 600,
}


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` to find the JacobianODE repo root."""
    p = Path(start or Path.cwd()).resolve()
    while True:
        if (p / "JacobianODE" / "jacobians" / "conf").is_dir():
            return p
        if p == p.parent:
            raise RuntimeError(
                "Not inside a JacobianODE repo (no JacobianODE/jacobians/conf found)"
            )
        p = p.parent


def parse_overrides(args: list[str]) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Separate Hydra overrides into (experiments, grid overrides, raw).

    Returns
    -------
    experiments
        List of experiment names (e.g. from ``experiment=foo,bar`` → ['foo', 'bar']).
    grid
        Dict preserving insertion order: ``override_key`` → list of values
        (singleton list if no comma in the value).
    raw
        The original argv (minus anything that doesn't look like an override)
        for round-trip reconstruction.
    """
    experiments: list[str] = []
    grid: dict[str, list[str]] = {}
    raw: list[str] = []
    for a in args:
        if "=" not in a:
            continue  # skip flags like --multirun
        raw.append(a)
        k, v = a.split("=", 1)
        v = v.strip()
        if k == "experiment":
            experiments.extend(x.strip() for x in v.split(","))
        else:
            # Strip outer quotes if present (bash may preserve them)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            values = [x.strip() for x in v.split(",")]
            grid[k] = values
    return experiments, grid, raw


def resolve_sweep_grid(
    experiments: list[str], grid: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Enumerate the Cartesian product.

    Hydra's multirun enumerates grid overrides in declaration order, with
    earlier keys cycling slower (outer loop). itertools.product matches this
    when grid keys are iterated in declaration order.

    The outer iteration is over experiments × grid, same as Hydra's
    ``experiment=a,b training.foo=1,2`` expansion.
    """
    keys = list(grid.keys())
    values_lists = [grid[k] for k in keys]
    combos = list(itertools.product(*values_lists)) if keys else [()]
    resolved: list[dict[str, Any]] = []
    idx = 0
    for exp in experiments:
        for combo in combos:
            overrides = [f"{k}={v}" for k, v in zip(keys, combo)]
            resolved.append({
                "run_idx": idx,
                "experiment": exp,
                "overrides": overrides,
            })
            idx += 1
    return resolved


def load_experiment_yaml(repo: Path, name: str) -> dict:
    path = repo / "JacobianODE" / "jacobians" / "conf" / "experiment" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No experiment YAML at {path}")
    with open(path) as f:
        doc = yaml.safe_load(f)
    if doc is None:
        raise ValueError(f"Experiment YAML at {path} is empty")
    return doc


def validate_metadata(doc: dict, exp_name: str) -> dict:
    """Require that ``metadata`` has the three structured fields.

    Raises ValueError with an actionable message if missing.
    """
    meta = doc.get("metadata")
    if meta is None:
        raise ValueError(
            f"Experiment '{exp_name}' is missing the required 'metadata:' block.\n"
            f"Add it to the YAML with fields: {', '.join(REQUIRED_META_FIELDS)}."
        )
    if not isinstance(meta, dict):
        raise ValueError(
            f"Experiment '{exp_name}' has 'metadata' but it is not a mapping."
        )
    missing = [f for f in REQUIRED_META_FIELDS if f not in meta or meta[f] in (None, "")]
    if missing:
        raise ValueError(
            f"Experiment '{exp_name}' metadata is missing/empty fields: {missing}.\n"
            f"Required: {', '.join(REQUIRED_META_FIELDS)}."
        )
    return meta


def extract_wandb(doc: dict, overrides: list[str]) -> dict[str, str | None]:
    """Determine wandb entity/project/group for this sweep.

    Overrides take precedence over the YAML default. If ``wandb_group``
    contains Hydra interpolation (``${...}``) that we can't resolve without
    composing the full config, the unresolved string is returned as-is —
    the user can override explicitly.
    """
    entity = doc.get("wandb_entity")
    project = doc.get("wandb_project")
    group = doc.get("wandb_group")
    for o in overrides:
        if "=" not in o:
            continue
        k, v = o.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if k == "wandb_group":
            group = v
        elif k == "wandb_project":
            project = v
        elif k == "wandb_entity":
            entity = v
    return {"entity": entity, "project": project, "group": group}


def build_expected(
    args: list[str],
    *,
    git_branch: str = "",
    git_commit: str = "",
    git_repo_dir: str = "",
    launched_from: str = "",
    repo: Path | None = None,
) -> dict[str, Any]:
    """Construct the full ``expected.json`` document.

    Raises ValueError on validation failures (missing YAML, missing metadata).
    """
    if repo is None:
        repo = find_repo_root()
    experiments, grid, raw = parse_overrides(args)
    if not experiments:
        raise ValueError(
            "No 'experiment=<name>' argument found. Every sweep must reference "
            "at least one experiment YAML."
        )

    meta_by_exp: dict[str, Any] = {}
    docs_by_exp: dict[str, Any] = {}
    sweep_grid_yaml: dict[str, list[str]] = {}
    for exp in experiments:
        doc = load_experiment_yaml(repo, exp)
        meta = validate_metadata(doc, exp)
        meta_by_exp[exp] = meta
        docs_by_exp[exp] = doc
        # Merge any sweep_grid block declared in the YAML.
        y_grid = doc.get("sweep_grid")
        if y_grid:
            for k, v in y_grid.items():
                values = (
                    [s.strip() for s in str(v).split(",")]
                    if not isinstance(v, list)
                    else [str(x) for x in v]
                )
                # CLI overrides (if any) take precedence over YAML values.
                if k not in grid:
                    sweep_grid_yaml[k] = values

    # CLI grid wins when both are present.
    effective_grid = {**sweep_grid_yaml, **grid}

    resolved = resolve_sweep_grid(experiments, effective_grid)

    # Use first experiment's YAML + the full override list (not just run[0]'s
    # subset) so that sweep-wide overrides like wandb_group=... are picked up.
    first_doc = docs_by_exp[experiments[0]]
    wandb_info = extract_wandb(first_doc, raw)

    # Reject gridded wandb_group — a sweep with multiple groups would need
    # multiple sentinels, which breaks the one-sentinel-per-sweep model.
    if "wandb_group" in grid and len(grid["wandb_group"]) > 1:
        raise ValueError(
            "wandb_group cannot be swept — each sweep has exactly one sentinel. "
            "Split into separate submissions if you need multiple groups."
        )

    launched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # If the caller passed a Hydra launcher partition override in ``raw``
    # (e.g. ``hydra.launcher.partition=ou_bcs_low``), carry it into
    # ``slurm.partition`` so monitor-initiated retries go to the same
    # partition as the original sweep. Without this, retries always land
    # on the DEFAULT_SLURM partition regardless of where the sweep ran.
    slurm_config = {
        "initial_array_job_id": None,
        **DEFAULT_SLURM,
    }
    for ov in raw:
        if ov.startswith("hydra.launcher.partition="):
            slurm_config["partition"] = ov.split("=", 1)[1]
            break

    return {
        "schema_version": SCHEMA_VERSION,
        "wandb": wandb_info,
        "launched_at": launched_at,
        "launched_from": launched_from or socket.gethostname(),
        "git": {
            "branch": git_branch,
            "commit": git_commit,
            "repo_dir": git_repo_dir,
        },
        "hydra": {
            "experiments": experiments,
            "multirun_dir": None,  # set by the launcher after Hydra picks it
            "overrides_template": raw,
            "resolved_runs": resolved,
        },
        "expected_run_count": len(resolved),
        "slurm": slurm_config,
        "retry": DEFAULT_RETRY,
        "experiment_metadata": meta_by_exp,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate + resolve a JacobianODE sweep for submission.",
        usage="%(prog)s [options] <hydra overrides...>",
    )
    parser.add_argument("--git-branch", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--git-repo-dir", default="")
    parser.add_argument("--launched-from", default="")
    parser.add_argument(
        "--output", "-o", default="-",
        help="Output path for JSON (default: stdout)",
    )
    parser.add_argument(
        "--print-group", action="store_true",
        help="Instead of emitting JSON, print just the resolved wandb_group name.",
    )
    parser.add_argument(
        "--print-sweep-args", action="store_true",
        help=(
            "Print the effective sweep-grid overrides (from YAML + CLI) as "
            "space-separated Hydra override strings, e.g.:\n"
            "  training.lightning.loop_closure_weight=0,1e-6,...  "
            "training.lightning.obs_noise_scale=0,0.01,0.05\n"
            "Useful for engaging-submit to append these to the jsweep command."
        ),
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    try:
        doc = build_expected(
            args.overrides,
            git_branch=args.git_branch,
            git_commit=args.git_commit,
            git_repo_dir=args.git_repo_dir,
            launched_from=args.launched_from,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.print_group:
        group = doc["wandb"].get("group")
        if not group:
            print("ERROR: No wandb_group resolved for this sweep.", file=sys.stderr)
            return 2
        print(group)
        return 0

    if args.print_sweep_args:
        # Reconstruct the effective grid from the resolved_runs by collecting
        # the distinct values seen per override key.
        seen: dict[str, list[str]] = {}
        for r in doc["hydra"]["resolved_runs"]:
            for ov in r["overrides"]:
                if "=" not in ov:
                    continue
                k, v = ov.split("=", 1)
                seen.setdefault(k, [])
                if v not in seen[k]:
                    seen[k].append(v)
        # Only emit keys with >=2 distinct values (i.e. the actual sweep axes).
        overrides = [
            f"{k}={','.join(vs)}"
            for k, vs in seen.items()
            if len(vs) >= 2
        ]
        print(" ".join(overrides))
        return 0

    text = json.dumps(doc, indent=2, sort_keys=False)
    if args.output == "-":
        print(text)
    else:
        with open(args.output, "w") as f:
            f.write(text)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
