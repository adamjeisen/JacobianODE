"""Tests for configuration utilities."""

from __future__ import annotations

import copy

import pytest

from ..core.config import initialize_config
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
