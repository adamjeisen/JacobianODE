"""Single-cell sbatch submission helper.

Used by three call sites:
  1. ``monitor.resubmit_run`` — retries a crashed cell on the partition
     recorded in ``expected.slurm``.
  2. ``engaging-controller._submit_one_split`` — initial submission of a
     migrate-eligible sweep, splitting cells across mit/ou_bcs.
  3. ``migration.migrate_one`` — moving a pending cell from ou_bcs_normal
     to mit_normal_gpu.

Pure: constructs and runs one sbatch invocation, returns the bare job id.
No journaling, no state writes — caller owns durability.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Path to uv on the engaging cluster (matches monitor.resubmit_run's hardcode).
UV_PATH = "/home/eisenaj/.local/bin/uv"


@dataclass(frozen=True)
class PartitionSpec:
    """Concrete SLURM submission target.

    Fields with ``None`` are simply omitted from the sbatch command. This
    lets ou_bcs_normal (no account/qos) and mit_normal_gpu (with both)
    use the same submit_cell() path.
    """
    partition: str
    account: Optional[str] = None
    qos: Optional[str] = None
    gres: str = "gpu:1"            # SLURM --gres value, e.g. "gpu:h200:1"
    cpus_per_task: int = 4
    mem: str = "16GB"
    timeout_min: int = 180


# The only two partition specs we use. Defined here so all call sites
# share one source of truth.
OU_BCS_NORMAL = PartitionSpec(
    partition="ou_bcs_normal",
    gres="gpu:1",
)

MIT_NORMAL_GPU = PartitionSpec(
    partition="mit_normal_gpu",
    account="mit_amf_advanced_gpu",
    qos="mit_amf_advanced_gpu",
    gres="gpu:h200:1",
)


def spec_from_expected_slurm(slurm: dict) -> PartitionSpec:
    """Reconstruct a PartitionSpec from an ``expected.json["slurm"]`` dict.

    Used by monitor retries: read whatever partition the sweep was last
    submitted to, replay it. Maintains backwards compatibility with the
    pre-migration field set (no account/qos in legacy expected.json files).
    """
    return PartitionSpec(
        partition=slurm.get("partition", "ou_bcs_normal"),
        account=slurm.get("account"),
        qos=slurm.get("qos"),
        gres=slurm.get("gres", "gpu:1"),
        cpus_per_task=slurm.get("cpus_per_task", 4),
        mem=slurm.get("mem", "16GB"),
        timeout_min=slurm.get("timeout_min", 180),
    )


def build_sbatch_args(
    partition: PartitionSpec,
    job_name: str,
    log_dir: Path,
    wrap_cmd: str,
) -> list[str]:
    """Build the sbatch argv for a single-cell submission. Pure function —
    safe to test by string-comparison without invoking sbatch.
    """
    args = [
        "sbatch",
        "--parsable",
        f"--partition={partition.partition}",
        f"--gres={partition.gres}",
        f"--cpus-per-task={partition.cpus_per_task}",
        f"--mem={partition.mem}",
        f"--time={partition.timeout_min}",
        f"--job-name={job_name}",
        f"--output={log_dir}/{job_name}_%j.out",
        f"--error={log_dir}/{job_name}_%j.err",
    ]
    # Account/qos are partition-specific; omit when unset so the user's
    # default association applies (which is what ou_bcs_normal needs).
    if partition.account:
        args.append(f"--account={partition.account}")
    if partition.qos:
        args.append(f"--qos={partition.qos}")
    args.extend(["--wrap", wrap_cmd])
    return args


def build_run_cmd(
    repo_dir: str,
    experiment: str,
    overrides: list[str],
) -> str:
    """Build the wrapped run command string. Same pattern as
    ``monitor.resubmit_run`` — single-run (not --multirun) invocation."""
    parts = [
        "cd", repo_dir, "&&", "OPENBLAS_NUM_THREADS=4",
        UV_PATH, "run", "--no-sync",
        "python", "-m", "JacobianODE.jacobians.run_jacobians",
        f"experiment={experiment}",
    ] + list(overrides)
    return " ".join(parts)


def submit_cell(
    expected: dict,
    run_idx: int,
    partition: PartitionSpec,
    sweeps_dir: Path,
    job_name_prefix: str = "jacobian",
) -> str:
    """Submit a single sweep cell as a one-off sbatch job. Returns the
    SLURM job id (bare, no array suffix). Raises on sbatch failure.

    Reads cell-specific Hydra config (experiment + overrides) from
    ``expected.hydra.resolved_runs[run_idx]``. Repo dir from
    ``expected.git.repo_dir``. Group name (for log file naming) from
    ``expected.wandb.group``.
    """
    resolved = next(
        r for r in expected["hydra"]["resolved_runs"]
        if r["run_idx"] == run_idx
    )
    experiment = resolved["experiment"]
    overrides = resolved["overrides"]
    repo_dir = expected["git"]["repo_dir"]
    group = expected["wandb"]["group"]

    wrap_cmd = build_run_cmd(repo_dir, experiment, overrides)

    log_dir = sweeps_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"{job_name_prefix}_{group}_r{run_idx}"
    sbatch_args = build_sbatch_args(partition, job_name, log_dir, wrap_cmd)

    result = subprocess.run(
        sbatch_args, check=True, capture_output=True, text=True,
    )
    jid = result.stdout.strip()
    logger.info(
        f"submit_cell {group}/run_idx={run_idx} -> SLURM job {jid} "
        f"on partition={partition.partition}"
    )
    return jid


def build_mc_run_cmd(
    mc_repo: str,
    instruction_path: str,
    cell_index: int,
    uv: str = UV_PATH,
) -> str:
    """Build the wrapped run command for a MindControl cell.

    Mirrors the wrap built in ``mindcontrol/cli/mc_sbatch.py``, but with a
    fixed ``--cell-index`` instead of ``$SLURM_ARRAY_TASK_ID`` since
    migrated cells run as standalone (non-array) sbatch jobs.
    """
    parts = [
        "cd", mc_repo, "&&",
        "OPENBLAS_NUM_THREADS=4", "HDF5_USE_FILE_LOCKING=FALSE",
        uv, "run", "--no-sync",
        "python", "-m", "mindcontrol.sweep_cell",
        "--instruction-path", instruction_path,
        "--cell-index", str(cell_index),
    ]
    return " ".join(parts)


def submit_mc_cell(
    expected: dict,
    run_idx: int,
    partition: PartitionSpec,
    sweeps_dir: Path,
    job_name_prefix: str = "mc_migrated",
) -> str:
    """Submit a single MindControl sweep cell as a one-off sbatch job.

    Parallel to :func:`submit_cell` but for MC sweeps. Reads MC-specific
    fields from ``expected["mc"]`` (instruction_path, mc_repo, uv).
    Returns the SLURM job id (bare, no array suffix). Raises on sbatch
    failure.

    See JacobianODE/docs/mc_migration_plan.md for the migration design.
    """
    mc = expected["mc"]
    instruction_path = mc["instruction_path"]
    mc_repo = mc["mc_repo"]
    uv = mc.get("uv", UV_PATH)
    group = expected["wandb"]["group"]

    wrap_cmd = build_mc_run_cmd(mc_repo, instruction_path, run_idx, uv=uv)

    log_dir = sweeps_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"{job_name_prefix}_{group[:40]}_r{run_idx}"
    sbatch_args = build_sbatch_args(partition, job_name, log_dir, wrap_cmd)

    result = subprocess.run(
        sbatch_args, check=True, capture_output=True, text=True,
    )
    jid = result.stdout.strip()
    logger.info(
        f"submit_mc_cell {group}/cell_index={run_idx} -> SLURM job {jid} "
        f"on partition={partition.partition}"
    )
    return jid
