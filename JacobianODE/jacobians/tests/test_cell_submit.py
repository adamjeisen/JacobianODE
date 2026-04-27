"""Unit tests for cell_submit.py — sbatch command construction.

The actual sbatch invocation is mocked; we only verify the command
string is correct for each PartitionSpec. The downstream behavior of
SLURM is out of scope for unit tests.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from JacobianODE.jacobians.tuning.cell_submit import (
    MIT_NORMAL_GPU,
    OU_BCS_NORMAL,
    PartitionSpec,
    build_run_cmd,
    build_sbatch_args,
    spec_from_expected_slurm,
    submit_cell,
)


class TestPartitionSpecConstants:
    def test_ou_bcs_normal_omits_account_qos(self):
        """OU_BCS_NORMAL should leave account/qos unset — that partition
        rejects mit_amf_advanced_gpu QOS."""
        assert OU_BCS_NORMAL.account is None
        assert OU_BCS_NORMAL.qos is None
        assert OU_BCS_NORMAL.partition == "ou_bcs_normal"

    def test_mit_normal_gpu_sets_account_qos_and_h200(self):
        """MIT_NORMAL_GPU must carry the QOS triple and H200 gres."""
        assert MIT_NORMAL_GPU.partition == "mit_normal_gpu"
        assert MIT_NORMAL_GPU.account == "mit_amf_advanced_gpu"
        assert MIT_NORMAL_GPU.qos == "mit_amf_advanced_gpu"
        assert MIT_NORMAL_GPU.gres == "gpu:h200:1"


class TestBuildSbatchArgs:
    def test_ou_bcs_no_account_qos_flags(self):
        """When account/qos are None, the flags must be absent. Otherwise
        ou_bcs_normal will reject the submission."""
        args = build_sbatch_args(
            OU_BCS_NORMAL, "myjob", Path("/tmp/logs"), "echo hi",
        )
        joined = " ".join(args)
        assert "--account" not in joined
        assert "--qos" not in joined
        assert "--partition=ou_bcs_normal" in joined
        assert "--gres=gpu:1" in joined
        assert "--wrap" in args
        assert args[args.index("--wrap") + 1] == "echo hi"

    def test_mit_includes_account_qos_h200(self):
        args = build_sbatch_args(
            MIT_NORMAL_GPU, "myjob", Path("/tmp/logs"), "echo hi",
        )
        joined = " ".join(args)
        assert "--account=mit_amf_advanced_gpu" in joined
        assert "--qos=mit_amf_advanced_gpu" in joined
        assert "--gres=gpu:h200:1" in joined
        assert "--partition=mit_normal_gpu" in joined

    def test_log_file_paths_use_job_name(self):
        """Log paths embed the job name so logs are searchable per cell."""
        args = build_sbatch_args(
            OU_BCS_NORMAL, "jacobian_mygroup_r5",
            Path("/sweeps/logs"), "echo hi",
        )
        joined = " ".join(args)
        assert "/sweeps/logs/jacobian_mygroup_r5_%j.out" in joined
        assert "/sweeps/logs/jacobian_mygroup_r5_%j.err" in joined

    def test_cpus_mem_time_pass_through(self):
        spec = PartitionSpec(
            partition="testpart", cpus_per_task=8, mem="32GB", timeout_min=600,
        )
        args = build_sbatch_args(spec, "j", Path("/tmp"), "echo hi")
        joined = " ".join(args)
        assert "--cpus-per-task=8" in joined
        assert "--mem=32GB" in joined
        assert "--time=600" in joined


class TestBuildRunCmd:
    def test_includes_uv_run_no_sync(self):
        """Must use uv run --no-sync to avoid clobbering pinned deps."""
        cmd = build_run_cmd("/repo", "myexp", ["a=1", "b=2"])
        assert "uv" in cmd
        assert "run" in cmd
        assert "--no-sync" in cmd

    def test_overrides_appended_after_experiment(self):
        cmd = build_run_cmd("/repo", "myexp", ["foo.bar=baz", "qux=1e-5"])
        # experiment must come before overrides; overrides preserve order
        i_exp = cmd.index("experiment=myexp")
        i_foo = cmd.index("foo.bar=baz")
        i_qux = cmd.index("qux=1e-5")
        assert i_exp < i_foo < i_qux

    def test_cd_to_repo_dir(self):
        cmd = build_run_cmd("/path/to/repo", "myexp", [])
        assert cmd.startswith("cd /path/to/repo &&")


class TestSpecFromExpectedSlurm:
    def test_legacy_no_account_qos(self):
        """Pre-migration expected.json files have only partition/gres/etc.
        — must still produce a usable PartitionSpec with None for account/qos."""
        spec = spec_from_expected_slurm({
            "partition": "ou_bcs_normal", "gres": "gpu:1",
            "cpus_per_task": 4, "mem": "16GB", "timeout_min": 180,
        })
        assert spec.partition == "ou_bcs_normal"
        assert spec.account is None
        assert spec.qos is None

    def test_with_mit_fields(self):
        """Migrated sweeps have account+qos in expected.slurm — round-trips."""
        spec = spec_from_expected_slurm({
            "partition": "mit_normal_gpu",
            "account": "mit_amf_advanced_gpu",
            "qos": "mit_amf_advanced_gpu",
            "gres": "gpu:h200:1",
        })
        assert spec.account == "mit_amf_advanced_gpu"
        assert spec.qos == "mit_amf_advanced_gpu"

    def test_empty_dict_returns_defaults(self):
        spec = spec_from_expected_slurm({})
        assert spec.partition == "ou_bcs_normal"  # default fallback


class TestSubmitCell:
    """submit_cell ends in subprocess.run(['sbatch', ...]). Mock that."""

    def _make_expected(self):
        return {
            "wandb": {"group": "mygroup"},
            "git": {"repo_dir": "/repo"},
            "hydra": {"resolved_runs": [
                {"run_idx": 0, "experiment": "exp_a",
                 "overrides": ["lc=1e-5"]},
                {"run_idx": 1, "experiment": "exp_a",
                 "overrides": ["lc=1e-4"]},
            ]},
        }

    def test_returns_jobid_from_sbatch_stdout(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345\n", returncode=0,
            )
            jid = submit_cell(
                self._make_expected(), 0, MIT_NORMAL_GPU, tmp_path,
            )
            assert jid == "12345"

    def test_picks_correct_run_idx_overrides(self, tmp_path):
        """run_idx=1 should pick lc=1e-4, not lc=1e-5."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="999", returncode=0)
            submit_cell(
                self._make_expected(), 1, OU_BCS_NORMAL, tmp_path,
            )
            sbatch_argv = mock_run.call_args[0][0]
            wrap_idx = sbatch_argv.index("--wrap")
            wrap_cmd = sbatch_argv[wrap_idx + 1]
            assert "lc=1e-4" in wrap_cmd
            assert "lc=1e-5" not in wrap_cmd

    def test_passes_partition_spec_through(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="999", returncode=0)
            submit_cell(
                self._make_expected(), 0, MIT_NORMAL_GPU, tmp_path,
            )
            argv = mock_run.call_args[0][0]
            joined = " ".join(argv)
            assert "--partition=mit_normal_gpu" in joined
            assert "--account=mit_amf_advanced_gpu" in joined
            assert "--qos=mit_amf_advanced_gpu" in joined
            assert "--gres=gpu:h200:1" in joined

    def test_raises_on_sbatch_failure(self, tmp_path):
        """Must propagate CalledProcessError so caller can handle recovery."""
        import subprocess
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["sbatch"], stderr="bad",
            )
            with pytest.raises(subprocess.CalledProcessError):
                submit_cell(
                    self._make_expected(), 0, MIT_NORMAL_GPU, tmp_path,
                )

    def test_log_dir_created(self, tmp_path):
        """submit_cell creates sweeps_dir/logs if absent."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="1", returncode=0)
            submit_cell(
                self._make_expected(), 0, OU_BCS_NORMAL, tmp_path,
            )
            assert (tmp_path / "logs").is_dir()
