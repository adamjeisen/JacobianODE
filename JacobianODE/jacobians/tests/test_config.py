"""Tests for configuration utilities."""

from __future__ import annotations

import copy

import pytest
from omegaconf import OmegaConf

from ..core.config import initialize_config, resolve_observed_indices
from ..core.reproducibility import seed_everything


class TestSeedEverything:
    """Tests for seed_everything function."""

    def test_seed_sets_numpy_seed(self):
        """Test that seed_everything sets numpy random seed."""
        import numpy as np

        seed_everything(42)
        val1 = np.random.random()

        seed_everything(42)
        val2 = np.random.random()

        assert val1 == val2

    def test_seed_sets_torch_seed(self):
        """Test that seed_everything sets torch random seed."""
        import torch

        seed_everything(42)
        val1 = torch.rand(1).item()

        seed_everything(42)
        val2 = torch.rand(1).item()

        assert val1 == val2

    def test_seed_sets_python_random(self):
        """Test that seed_everything sets Python random seed."""
        import random

        seed_everything(42)
        val1 = random.random()

        seed_everything(42)
        val2 = random.random()

        assert val1 == val2

    def test_seed_returns_seed(self):
        """Test that seed_everything returns the seed."""
        result = seed_everything(123)
        assert result == 123

    def test_seed_negative_raises(self):
        """Test that negative seed raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            seed_everything(-1)


class TestInitializeConfig:
    """Tests for initialize_config function."""

    def test_initialize_config_returns_new_object(self):
        """Test that initialize_config returns a new config object."""
        from omegaconf import OmegaConf

        # Create a minimal config
        cfg = OmegaConf.create({
            "data": {
                "data_type": "custom",
                "flow": {"dim": 3},
                "train_test_params": {
                    "delay_embedding_params": {
                        "observed_indices": "all",
                        "n_delays": 1,
                    }
                },
            },
            "model": {
                "params": {
                    "_target_": "JacobianODE.jacobians.models.MLP",
                    "input_dim": None,
                }
            },
            "training": {
                "lightning": {
                    "_target_": "JacobianODE.jacobians.lightning_base.LitBase",
                    "direct": True,
                }
            },
        })

        original_input_dim = cfg.model.params.input_dim

        initialized = initialize_config(cfg)

        # Original should be unchanged
        assert cfg.model.params.input_dim == original_input_dim

        # Initialized should be different object
        assert cfg is not initialized

    def test_initialize_config_custom_data_requires_dim(self):
        """Test that custom data requires data.flow.dim."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "data": {
                "data_type": "custom",
                "flow": {"dim": None},  # Missing dim
                "train_test_params": {
                    "delay_embedding_params": {
                        "observed_indices": "all",
                        "n_delays": 1,
                    }
                },
            },
            "model": {
                "params": {
                    "_target_": "JacobianODE.jacobians.models.MLP",
                }
            },
            "training": {
                "lightning": {
                    "_target_": "JacobianODE.jacobians.lightning_base.LitBase",
                    "direct": True,
                }
            },
        })

        with pytest.raises(ValueError, match="data.flow.dim"):
            initialize_config(cfg)

    def test_initialize_config_sets_input_dim(self):
        """Test that initialize_config sets input_dim from data."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "data": {
                "data_type": "custom",
                "flow": {"dim": 10},
                "train_test_params": {
                    "delay_embedding_params": {
                        "observed_indices": "all",
                        "n_delays": 1,
                    }
                },
            },
            "model": {
                "params": {
                    "_target_": "JacobianODE.jacobians.models.MLP",
                    "input_dim": None,
                }
            },
            "training": {
                "lightning": {
                    "_target_": "JacobianODE.jacobians.lightning_base.LitBase",
                    "direct": True,
                }
            },
        })

        initialized = initialize_config(cfg)

        assert initialized.model.params.input_dim == 10

    def test_initialize_config_sets_output_dim_direct(self):
        """Test that initialize_config sets output_dim = dim^2 for direct mode."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "data": {
                "data_type": "custom",
                "flow": {"dim": 5},
                "train_test_params": {
                    "delay_embedding_params": {
                        "observed_indices": "all",
                        "n_delays": 1,
                    }
                },
            },
            "model": {
                "params": {
                    "_target_": "JacobianODE.jacobians.models.MLP",
                    "input_dim": None,
                    "output_dim": None,
                }
            },
            "training": {
                "lightning": {
                    "_target_": "JacobianODE.jacobians.lightning_base.LitBase",
                    "direct": True,
                }
            },
        })

        initialized = initialize_config(cfg)

        assert initialized.model.params.output_dim == 25  # 5^2


class TestResolveObservedIndices:
    """Tests for resolve_observed_indices."""

    @staticmethod
    def _wmtask_cfg(delay_params):
        return OmegaConf.create({
            "data": {
                "data_type": "wmtask",
                "flow": {"dim": 128, "random_state": 42},
                "train_test_params": {"delay_embedding_params": delay_params},
            },
        })

    def test_all_is_noop(self):
        cfg = self._wmtask_cfg({"observed_indices": "all"})
        resolve_observed_indices(cfg)
        assert cfg.data.train_test_params.delay_embedding_params.observed_indices == "all"

    def test_explicit_list_is_noop(self):
        cfg = self._wmtask_cfg({"observed_indices": [3, 7, 19]})
        resolve_observed_indices(cfg)
        assert list(
            cfg.data.train_test_params.delay_embedding_params.observed_indices
        ) == [3, 7, 19]

    def test_random_flat_picks_n_observed_sorted(self):
        cfg = self._wmtask_cfg({
            "observed_indices": "random",
            "n_observed": 5,
            "partial_obs_seed": 7,
        })
        resolve_observed_indices(cfg)
        idx = list(cfg.data.train_test_params.delay_embedding_params.observed_indices)
        assert len(idx) == 5
        assert all(0 <= i < 128 for i in idx)
        assert idx == sorted(idx)
        assert len(set(idx)) == 5  # no dupes

    def test_random_flat_seed_reproducible(self):
        a = self._wmtask_cfg({"observed_indices": "random", "n_observed": 8,
                              "partial_obs_seed": 11})
        b = self._wmtask_cfg({"observed_indices": "random", "n_observed": 8,
                              "partial_obs_seed": 11})
        resolve_observed_indices(a)
        resolve_observed_indices(b)
        assert list(a.data.train_test_params.delay_embedding_params.observed_indices) \
            == list(b.data.train_test_params.delay_embedding_params.observed_indices)

    def test_random_flat_requires_n_observed(self):
        cfg = self._wmtask_cfg({"observed_indices": "random"})
        with pytest.raises(ValueError, match="n_observed"):
            resolve_observed_indices(cfg)

    def test_per_area_picks_correct_counts_and_preserves_order(self):
        cfg = self._wmtask_cfg({
            "observed_indices": "random",
            "n_observed_per_area": [3, 4],
            "obs_area_indices": [list(range(64)), list(range(64, 128))],
            "partial_obs_seed": 13,
        })
        resolve_observed_indices(cfg)
        idx = list(cfg.data.train_test_params.delay_embedding_params.observed_indices)
        assert len(idx) == 7
        # First 3 are from visual block (0..63), next 4 are from cognitive (64..127)
        assert all(0 <= i < 64 for i in idx[:3])
        assert all(64 <= i < 128 for i in idx[3:])
        # Within each area, sorted
        assert idx[:3] == sorted(idx[:3])
        assert idx[3:] == sorted(idx[3:])
        # Globally NOT sorted (visual indices < cognitive indices is enforced by partition,
        # but the test is that no global sort step happens)
        assert len(set(idx)) == 7

    def test_per_area_rejects_with_n_observed(self):
        cfg = self._wmtask_cfg({
            "observed_indices": "random",
            "n_observed": 5,
            "n_observed_per_area": [3, 2],
            "obs_area_indices": [[0, 1, 2], [10, 11, 12, 13]],
        })
        with pytest.raises(ValueError, match="Pick one mode"):
            resolve_observed_indices(cfg)

    def test_per_area_count_too_large(self):
        cfg = self._wmtask_cfg({
            "observed_indices": "random",
            "n_observed_per_area": [10, 4],          # 10 > area size 5
            "obs_area_indices": [list(range(5)), list(range(5, 10))],
        })
        with pytest.raises(ValueError, match=r"n_observed_per_area\[0\]"):
            resolve_observed_indices(cfg)

    def test_per_area_index_out_of_range(self):
        cfg = self._wmtask_cfg({
            "observed_indices": "random",
            "n_observed_per_area": [2, 2],
            "obs_area_indices": [[0, 1], [200, 201]],   # 200 >= total_dim=128
        })
        with pytest.raises(ValueError, match="outside"):
            resolve_observed_indices(cfg)

    def test_per_area_duplicate_index_across_areas(self):
        cfg = self._wmtask_cfg({
            "observed_indices": "random",
            "n_observed_per_area": [1, 1],
            "obs_area_indices": [[5, 6], [6, 7]],   # 6 in both
        })
        with pytest.raises(ValueError, match="duplicate"):
            resolve_observed_indices(cfg)

    def test_per_area_seed_reproducible(self):
        a = self._wmtask_cfg({
            "observed_indices": "random",
            "n_observed_per_area": [4, 5],
            "obs_area_indices": [list(range(64)), list(range(64, 128))],
            "partial_obs_seed": 99,
        })
        b = copy.deepcopy(a)
        resolve_observed_indices(a)
        resolve_observed_indices(b)
        assert list(a.data.train_test_params.delay_embedding_params.observed_indices) \
            == list(b.data.train_test_params.delay_embedding_params.observed_indices)
