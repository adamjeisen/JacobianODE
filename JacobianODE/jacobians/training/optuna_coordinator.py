"""Async Optuna coordinator for JacobianODE hyperparameter sweeps.

Replaces the Hydra-based Optuna sweeper with a direct Optuna + submitit
coordinator that achieves **true backfilling**: as soon as one SLURM job
finishes, a new trial is queued immediately (no batch-level blocking).

Usage (from notebook or coordinator script)::

    from JacobianODE.jacobians.training.optuna_coordinator import OptunaCoordinator

    coordinator = OptunaCoordinator(
        fixed_overrides=overrides,
        search_space=SEARCH_SPACE,
        study_name=STUDY_NAME,
        storage=OPTUNA_STORAGE,
        entry_point="python -m JacobianODE.jacobians.run_jacobians",
        repo_root="/path/to/JacobianODE",
    )
    coordinator.run(n_trials=40, n_parallel=10)

Grid sweeps are unaffected — they continue to use ``--multirun`` + ``slurm=default``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna

log = logging.getLogger(__name__)


def _to_optuna_distribution(spec: Dict[str, Any]) -> optuna.distributions.BaseDistribution:
    """Convert a search-space spec dict to an Optuna distribution.

    Supports the same format used in the notebook::

        {"type": "float", "low": 1e-7, "high": 10.0, "log": True}
        {"type": "int",   "low": 1,    "high": 100,   "log": False}
        {"type": "categorical", "choices": ["a", "b", "c"]}
    """
    ptype = spec["type"]
    if ptype == "float":
        return optuna.distributions.FloatDistribution(
            low=spec["low"], high=spec["high"], log=spec.get("log", False),
        )
    if ptype == "int":
        return optuna.distributions.IntDistribution(
            low=int(spec["low"]), high=int(spec["high"]),
            log=spec.get("log", False),
        )
    if ptype == "categorical":
        return optuna.distributions.CategoricalDistribution(spec["choices"])
    raise ValueError(f"Unknown distribution type: {ptype}")


def _worker(cmd_args: List[str], cwd: str) -> int:
    """Thin wrapper submitted to SLURM via submitit.

    Runs the Hydra entry point as a subprocess and returns the exit code.
    The actual objective value is communicated via the Optuna SQLite DB
    (written by OptunaProgressCallback during training).
    """
    import subprocess
    os.chdir(cwd)
    result = subprocess.run(cmd_args, capture_output=False)
    return result.returncode


class OptunaCoordinator:
    """Async Optuna sweep coordinator with true SLURM backfilling.

    Args:
        fixed_overrides: List of Hydra override strings that are constant
            across all trials (data config, model architecture, etc.).
        search_space: Dict mapping Hydra config paths to distribution specs.
        study_name: Optuna study name.
        storage: Optuna storage URL (e.g. ``"sqlite:///path.db"``).
        entry_point: Python command to run the training script.
        repo_root: Absolute path to the repository root.
        sampler_seed: Random seed for the TPE sampler.
        n_startup_trials: Number of random trials before TPE kicks in.
        slurm_partition: SLURM partition for worker jobs.
        slurm_gpus_per_node: GPUs per worker job.
        slurm_cpus_per_task: CPUs per worker job.
        slurm_mem_gb: Memory (GB) per worker job.
        slurm_timeout_min: SLURM timeout (minutes) per worker job.
        slurm_exclude: Nodes to exclude.
        slurm_additional_params: Extra params passed to ``executor.update_parameters()``.
        poll_interval: Seconds between polling for completed jobs.
        constraint_metric: Optional metric name for feasibility check
            (e.g. ``"val/loop_closure_loss"``).
        constraint_threshold: Maximum allowed value for constraint_metric.
    """

    # Attribute keys used by OptunaProgressCallback / OptunaConstrainedProgressCallback
    BEST_ATTR = "best_so_far"
    FEASIBLE_ATTR = "best_so_far_feasible"
    CONSTRAINT_ATTR = "best_so_far_constraint"

    def __init__(
        self,
        fixed_overrides: List[str],
        search_space: Dict[str, Dict[str, Any]],
        study_name: str,
        storage: str,
        entry_point: str = "python -m JacobianODE.jacobians.run_jacobians",
        repo_root: str = ".",
        sampler_seed: int = 42,
        n_startup_trials: int = 10,
        # SLURM worker params
        slurm_partition: str = "ou_bcs_normal",
        slurm_gpus_per_node: int = 1,
        slurm_cpus_per_task: int = 4,
        slurm_mem_gb: int = 16,
        slurm_timeout_min: int = 180,
        slurm_exclude: str = "node4000",
        slurm_additional_params: Optional[Dict[str, Any]] = None,
        poll_interval: float = 30.0,
        # Constraint
        constraint_metric: Optional[str] = None,
        constraint_threshold: Optional[float] = None,
    ):
        # Strip slurm=default and any hydra sweeper overrides — the coordinator
        # handles SLURM submission directly, not Hydra's launcher.
        _skip_prefixes = ("slurm=", "hydra/sweeper=", "hydra.sweeper.", "+sweeper=")
        self.fixed_overrides = [
            o for o in fixed_overrides
            if not any(o.startswith(p) for p in _skip_prefixes)
        ]
        self.search_space = {
            k: _to_optuna_distribution(v) for k, v in search_space.items()
        }
        self.study_name = study_name
        self.storage = storage
        self.entry_point = entry_point.split()
        self.repo_root = str(Path(repo_root).resolve())
        self.sampler_seed = sampler_seed
        self.n_startup_trials = n_startup_trials
        self.poll_interval = poll_interval
        self.constraint_metric = constraint_metric
        self.constraint_threshold = constraint_threshold

        # SLURM
        self._slurm_params = {
            "slurm_partition": slurm_partition,
            "gpus_per_node": slurm_gpus_per_node,
            "cpus_per_task": slurm_cpus_per_task,
            "mem_gb": slurm_mem_gb,
            "timeout_min": slurm_timeout_min,
            "slurm_exclude": slurm_exclude,
        }
        if slurm_additional_params:
            self._slurm_params.update(slurm_additional_params)

    def _constraints_func(self, frozen_trial: optuna.trial.FrozenTrial) -> List[float]:
        """Constraint function for TPESampler.

        Returns a list with one element: the constraint violation value.
        <=0 means feasible, >0 means violated.  TPE uses this to
        preferentially sample from feasible regions of hyperparameter space.
        """
        constraint_val = frozen_trial.user_attrs.get(self.CONSTRAINT_ATTR)
        if constraint_val is None:
            # No constraint data yet (trial still running or crashed early).
            # Treat as infeasible so TPE doesn't favor unknown regions.
            return [1.0]
        return [constraint_val - self.constraint_threshold]

    def _create_study(self) -> optuna.Study:
        """Create or load the Optuna study."""
        sampler_kwargs = {
            "seed": self.sampler_seed,
            "n_startup_trials": self.n_startup_trials,
        }
        if self.constraint_metric and self.constraint_threshold is not None:
            sampler_kwargs["constraints_func"] = self._constraints_func
            log.info(
                f"Constrained TPE enabled: {self.constraint_metric} "
                f"<= {self.constraint_threshold}"
            )

        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            direction="minimize",
            load_if_exists=True,
        )
        log.info(f"Study: {study.study_name} ({self.storage})")
        log.info(f"Sampler: TPESampler(seed={self.sampler_seed}, n_startup={self.n_startup_trials})")
        return study

    def _create_executor(self, log_folder: Path):
        """Create a submitit SLURM executor."""
        import submitit
        executor = submitit.AutoExecutor(cluster="slurm", folder=str(log_folder))
        executor.update_parameters(**self._slurm_params)
        return executor

    def _create_trial_executor(self, trial: optuna.Trial, base_log_folder: Path):
        """Create a submitit executor with a per-trial log subfolder."""
        trial_folder = base_log_folder / f"trial_{trial.number:04d}"
        trial_folder.mkdir(parents=True, exist_ok=True)
        return self._create_executor(trial_folder)

    def _build_overrides(self, trial: optuna.Trial) -> List[str]:
        """Build the full Hydra override list for a trial.

        ``trial.params`` is already populated by ``study.ask(distributions)``.
        """
        sampled_overrides = [f"{k}={v}" for k, v in trial.params.items()]

        # Add study/storage so callbacks can write to the DB
        meta_overrides = [
            f"optuna_study_name={self.study_name}",
            f"optuna_storage={self.storage}",
        ]

        # Add constraint overrides if configured
        if self.constraint_metric and self.constraint_threshold is not None:
            meta_overrides += [
                f"training.optuna_constraint_metric={self.constraint_metric}",
                f"training.optuna_constraint_threshold={self.constraint_threshold}",
            ]

        return self.fixed_overrides + sampled_overrides + meta_overrides

    def _submit_trial(self, trial: optuna.Trial, base_log_folder: Path) -> Any:
        """Submit a single trial as a SLURM job."""
        executor = self._create_trial_executor(trial, base_log_folder)
        overrides = self._build_overrides(trial)
        cmd = self.entry_point + overrides
        job = executor.submit(_worker, cmd, self.repo_root)
        log.info(
            f"  Trial {trial.number} → SLURM job {job.job_id}  "
            f"params={trial.params}"
        )
        return job

    def _collect_result(self, study: optuna.Study, trial: optuna.Trial, job) -> None:
        """Collect the result of a finished job and tell the study."""
        import submitit

        trial_in_study = study.trials[trial.number]

        # Check if the job succeeded, failed, or timed out
        try:
            exit_code = job.result()
            job_ok = (exit_code == 0)
        except submitit.core.utils.UncompletedJobError:
            job_ok = False
        except Exception:
            job_ok = False

        # Read best_so_far from DB (written by OptunaProgressCallback).
        # Report the actual loss value — TPE's constraints_func handles
        # feasibility separation, so it needs the real value to also rank
        # infeasible trials by quality.
        best = trial_in_study.user_attrs.get(self.BEST_ATTR)
        feasible = trial_in_study.user_attrs.get(self.FEASIBLE_ATTR, True)

        if best is not None and math.isfinite(best):
            study.tell(trial, values=[best], state=optuna.trial.TrialState.COMPLETE)
            tag = "" if feasible else " [infeasible]"
            log.info(
                f"  Trial {trial.number}: COMPLETE{tag}, "
                f"best_so_far={best:.6f}, params={trial.params}"
            )
        else:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            log.warning(
                f"  Trial {trial.number}: FAIL (no best_so_far in DB, "
                f"job_ok={job_ok}), params={trial.params}"
            )

    def run(
        self,
        n_trials: int = 40,
        n_parallel: int = 10,
        log_folder: Optional[str] = None,
    ) -> optuna.Study:
        """Run the Optuna sweep with true backfilling.

        Maintains ``n_parallel`` concurrent SLURM jobs at all times.
        As soon as one finishes, a new trial is sampled and submitted.

        Args:
            n_trials: Total number of trials.
            n_parallel: Maximum concurrent SLURM jobs.
            log_folder: Directory for submitit logs. Defaults to
                ``{repo_root}/.submitit_logs/{study_name}``.

        Returns:
            The Optuna study object.
        """
        study = self._create_study()

        if log_folder is None:
            log_folder = os.path.join(
                self.repo_root, ".submitit_logs", self.study_name
            )
        log_path = Path(log_folder)
        log_path.mkdir(parents=True, exist_ok=True)

        # Track active jobs: trial.number -> (trial, submitit_job)
        active: Dict[int, tuple] = {}
        n_queued = 0  # Total trials queued so far (including active)

        log.info(
            f"Starting sweep: {n_trials} trials, {n_parallel} parallel, "
            f"poll every {self.poll_interval}s"
        )
        log.info(f"Search space: {list(self.search_space.keys())}")

        # Skip trials that were already completed in a previous run
        # (supports resuming)
        n_already_done = len([
            t for t in study.trials
            if t.state in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED,
            )
        ])
        if n_already_done > 0:
            log.info(f"Resuming: {n_already_done} trials already done")
            n_queued = n_already_done

        try:
            while True:
                # Fill up to n_parallel active jobs
                while len(active) < n_parallel and n_queued < n_trials:
                    trial = study.ask(self.search_space)
                    job = self._submit_trial(trial, log_path)
                    active[trial.number] = (trial, job)
                    n_queued += 1

                if not active:
                    break  # All done

                # Poll for completed jobs
                time.sleep(self.poll_interval)

                # Reload study to see callback updates
                study = optuna.load_study(
                    study_name=self.study_name, storage=self.storage
                )

                for trial_num, (trial, job) in list(active.items()):
                    if job.done():
                        self._collect_result(study, trial, job)
                        del active[trial_num]

                # Progress report
                n_complete = len([
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ])
                n_fail = len([
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.FAIL
                ])
                n_running = len(active)
                log.info(
                    f"[Progress] {n_complete} complete, {n_fail} failed, "
                    f"{n_running} running, {n_trials - n_queued} remaining"
                )

        except KeyboardInterrupt:
            log.warning("Interrupted — active jobs will continue on SLURM.")
            log.warning(f"Resume with the same study_name={self.study_name}")

        # Final summary
        study = optuna.load_study(
            study_name=self.study_name, storage=self.storage
        )
        n_complete = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        n_fail = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.FAIL
        ])
        log.info(
            f"Sweep finished: {n_complete} completed, {n_fail} failed "
            f"out of {n_trials} total."
        )

        try:
            best = study.best_trial
            log.info(f"Best trial: {best.number}")
            log.info(f"  Value: {best.value}")
            log.info(f"  Params: {best.params}")
        except ValueError:
            log.warning("No completed trials.")

        return study
