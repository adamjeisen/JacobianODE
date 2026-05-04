"""Tests for the multi-source combined dataloader.

Covers:
  - load_combined_for_jacobianode concatenates trajectories + builds the
    per-trajectory condition / source_id tensors.
  - Validation: dt mismatch, shape mismatch, condition_dim mismatch.
  - generate_train_and_test_sets with split_groups produces a balanced
    per-condition split (each split has the same condition mix).
  - generate_train_and_test_sets with `condition` produces datasets that
    yield (traj, c) tuples and the per-sequence c matches the trajectory
    layout expected by the existing _examples loops.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from JacobianODE.jacobians.data.combined_loader import load_combined_for_jacobianode
from JacobianODE.jacobians.data.splitting import (
    TimeSeriesDataset,
    collate_with_optional_condition,
    generate_train_and_test_sets,
)


def _mock_loader(n_traj, T, D, dt, value):
    """Return a zero-arg loader that produces (eq, sol, dt) like wmtask does.
    Trajectory values are constant per-source so we can verify routing."""
    def _load():
        sol = {"values": np.full((n_traj, T, D), value, dtype=np.float32)}
        return None, sol, dt
    return _load


class TestCombinedLoader:
    def test_concatenates_and_tags(self):
        sources = [
            {"loader": _mock_loader(5, 20, 4, 0.01, value=1.0), "condition": [-1.0]},
            {"loader": _mock_loader(7, 20, 4, 0.01, value=2.0), "condition": [1.0]},
        ]
        eq, sol, dt = load_combined_for_jacobianode(sources)
        assert sol["values"].shape == (12, 20, 4)
        # Source order preserved: first 5 trajs are source 0 (value 1.0),
        # next 7 are source 1 (value 2.0).
        assert np.all(sol["values"][:5] == 1.0)
        assert np.all(sol["values"][5:] == 2.0)
        # Per-trajectory condition broadcast correctly.
        assert sol["condition"].shape == (12, 1)
        assert np.all(sol["condition"][:5] == -1.0)
        assert np.all(sol["condition"][5:] == 1.0)
        # Source id integer.
        assert sol["source_id"].shape == (12,)
        assert np.all(sol["source_id"][:5] == 0)
        assert np.all(sol["source_id"][5:] == 1)
        assert dt == 0.01

    def test_raises_on_empty_sources(self):
        with pytest.raises(ValueError, match="empty"):
            load_combined_for_jacobianode([])

    def test_raises_on_dt_mismatch(self):
        sources = [
            {"loader": _mock_loader(3, 10, 2, 0.01, value=0.0), "condition": [0.0]},
            {"loader": _mock_loader(3, 10, 2, 0.02, value=0.0), "condition": [1.0]},
        ]
        with pytest.raises(ValueError, match="dt="):
            load_combined_for_jacobianode(sources)

    def test_raises_on_shape_mismatch(self):
        sources = [
            {"loader": _mock_loader(3, 10, 2, 0.01, value=0.0), "condition": [0.0]},
            {"loader": _mock_loader(3, 11, 2, 0.01, value=0.0), "condition": [1.0]},
        ]
        with pytest.raises(ValueError, match="per-trajectory shape"):
            load_combined_for_jacobianode(sources)

    def test_raises_on_condition_dim_mismatch(self):
        sources = [
            {"loader": _mock_loader(3, 10, 2, 0.01, value=0.0), "condition": [0.0]},
            {"loader": _mock_loader(3, 10, 2, 0.01, value=0.0), "condition": [0.0, 1.0]},
        ]
        with pytest.raises(ValueError, match="condition_dim"):
            load_combined_for_jacobianode(sources)


class TestBalancedSplit:
    def test_each_group_split_independently(self):
        np.random.seed(42)
        # 100 trajectories, 50 from each source.
        pts = np.random.randn(100, 30, 4).astype(np.float32)
        source_id = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        condition = np.concatenate([
            np.full((50, 1), -1.0), np.full((50, 1), 1.0),
        ]).astype(np.float32)

        train_ds, val_ds, test_ds, _ = generate_train_and_test_sets(
            pts, seq_length=10, seq_spacing=5, train_percent=0.8,
            test_percent=0.05, split_by='trajectory',
            condition=condition, split_groups=source_id,
        )

        # Each split should contain the same fraction of c=-1 vs c=+1.
        for ds in (train_ds, val_ds, test_ds):
            cond = ds.condition
            n_neg = int((cond[:, 0] == -1.0).sum().item())
            n_pos = int((cond[:, 0] == 1.0).sum().item())
            assert n_neg > 0 and n_pos > 0, (
                f"split missing one of the conditions: n_neg={n_neg} n_pos={n_pos}"
            )
            # 50/50 source mix → conditions should be exactly balanced
            # (each source's split fraction is identical).
            assert n_neg == n_pos, (
                f"split not balanced: n_neg={n_neg} n_pos={n_pos}"
            )

    def test_no_split_groups_back_compat(self):
        """Without split_groups the function must behave identically to
        the prior single-group split."""
        pts = np.random.randn(20, 30, 3).astype(np.float32)
        np.random.seed(123)
        a_train, a_val, a_test, _ = generate_train_and_test_sets(
            pts, seq_length=10, seq_spacing=5, train_percent=0.8,
            test_percent=0.05, split_by='trajectory',
        )
        np.random.seed(123)
        b_train, b_val, b_test, _ = generate_train_and_test_sets(
            pts, seq_length=10, seq_spacing=5, train_percent=0.8,
            test_percent=0.05, split_by='trajectory',
            condition=None, split_groups=None,
        )
        torch.testing.assert_close(a_train.sequence, b_train.sequence)
        torch.testing.assert_close(a_val.sequence, b_val.sequence)
        torch.testing.assert_close(a_test.sequence, b_test.sequence)


class TestPerSequenceCondition:
    def test_dataset_yields_traj_and_c_tuples(self):
        """A condition tensor should propagate to the dataset and yield
        (traj, c) items at the per-sequence level."""
        np.random.seed(7)
        pts = np.random.randn(30, 24, 4).astype(np.float32)
        source_id = np.concatenate([np.zeros(15), np.ones(15)]).astype(np.int64)
        condition = np.concatenate([
            np.full((15, 1), -1.0), np.full((15, 1), 1.0),
        ]).astype(np.float32)

        train_ds, _, _, _ = generate_train_and_test_sets(
            pts, seq_length=8, seq_spacing=4, train_percent=0.8,
            test_percent=0.05, split_by='trajectory',
            condition=condition, split_groups=source_id,
        )

        # __getitem__ returns (traj, c).
        traj, c = train_ds[0]
        assert traj.shape == (8, 4)
        assert c.shape == (1,)
        assert c.item() in (-1.0, 1.0)

        # Collate produces (batch, c) tuple stacked correctly.
        items = [train_ds[i] for i in range(min(4, len(train_ds)))]
        batch, c_batch = collate_with_optional_condition(items)
        assert batch.shape == (len(items), 8, 4)
        assert c_batch.shape == (len(items), 1)
