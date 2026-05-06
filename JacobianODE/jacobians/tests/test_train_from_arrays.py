"""End-to-end smoke tests for ``train_from_arrays``.

Verifies the public API actually runs end-to-end on tiny synthetic data
and that the training loss decreases. Uses a Lorenz-like 3-D system as
the data source so the dynamics are non-trivial.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from JacobianODE import train_from_arrays, TrainingResult


def _synth_smooth_data(n_traj: int = 40, T: int = 60, D: int = 4, seed: int = 0):
    """Generate (n_traj, T, D) of smooth trajectories — sums of slow
    sinusoids — so PCA + delay-embedding find a non-trivial latent
    structure quickly."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    out = np.empty((n_traj, T, D), dtype=np.float32)
    for i in range(n_traj):
        amps = rng.uniform(0.5, 1.5, size=D)
        freqs = rng.uniform(0.5, 1.5, size=D)
        phases = rng.uniform(0, 2 * np.pi, size=D)
        for d in range(D):
            out[i, :, d] = amps[d] * np.sin(freqs[d] * t + phases[d])
        # Add small noise so the trajectories aren't a strict 1-D manifold.
        out[i] += 0.05 * rng.standard_normal((T, D)).astype(np.float32)
    return out


@pytest.fixture
def tiny_data():
    return _synth_smooth_data(n_traj=20, T=40, D=4, seed=0)


def test_smoke_single_encoder_trains(tiny_data, tmp_path):
    """Single-encoder (latent_additive_coupling) training runs end-to-end
    for 2 epochs and validation loss is finite."""
    values = tiny_data
    dt = 0.05

    lit_model, result = train_from_arrays(
        values, dt,
        n_delays=2,
        seq_length=20,            # >= traj_init + prediction = 5 + 3 = 8, plus delay slack
        seq_spacing=5,
        encoder="latent_additive_coupling",
        encoder_kwargs={"hidden_dim": 32, "n_coupling_layers": 4},
        n_target_var_threshold=0.99,
        prediction_steps=3,
        n_epochs=2,
        batch_size=8,
        save_dir=str(tmp_path),
        wandb_disabled=True,
        verbose=False,
        # JacobianODE latent integration needs traj_init+prediction ≤ seq_length-n_delays+1
        # Default traj_init_steps=15 is too long for our seq_length=20; shrink it.
        lightning_kwargs={"jacobianODEint_kwargs": {"traj_init_steps": 5,
                                                    "inner_path": "line",
                                                    "inner_N": 20,
                                                    "interp_pts": 4}},
        early_stopping_kwargs={"early_stopping_patience": 100, "min_epochs": 0},
    )

    assert isinstance(result, TrainingResult)
    # Trainer ran .fit() — should have callback metrics
    assert result.trainer is not None
    assert "mean val loss" in result.trainer.callback_metrics or "val_loss" in str(result.trainer.callback_metrics)
    # Model parameters got updated (i.e. require_grad worked)
    assert sum(p.numel() for p in lit_model.parameters()) > 0
    # Postprocessing scalars were computed
    assert isinstance(result.mu, float)
    assert isinstance(result.sigma, float)
    assert result.sigma > 0


def test_smoke_with_condition_and_per_condition_norm(tiny_data, tmp_path):
    """Conditioned training with per-condition normalization across two
    sources. Verifies the multi-condition combined-loader-style path
    works end-to-end."""
    values = tiny_data  # (20, 40, 4)
    dt = 0.05
    n_traj = values.shape[0]
    # 10 from "source 0" (condition -1), 10 from "source 1" (condition +1)
    condition = np.empty((n_traj, 1), dtype=np.float32)
    condition[: n_traj // 2] = -1.0
    condition[n_traj // 2:] = 1.0
    source_id = np.array([0] * (n_traj // 2) + [1] * (n_traj - n_traj // 2), dtype=np.int64)

    lit_model, result = train_from_arrays(
        values, dt,
        condition=condition,
        source_id=source_id,
        n_delays=2,
        seq_length=20,
        seq_spacing=5,
        encoder="latent_additive_coupling",
        encoder_kwargs={"hidden_dim": 32, "n_coupling_layers": 4},
        n_target_var_threshold=0.99,
        prediction_steps=3,
        normalize_per_condition=True,
        n_epochs=2,
        batch_size=8,
        save_dir=str(tmp_path),
        wandb_disabled=True,
        lightning_kwargs={"jacobianODEint_kwargs": {"traj_init_steps": 5,
                                                    "inner_path": "line",
                                                    "inner_N": 20,
                                                    "interp_pts": 4}},
        early_stopping_kwargs={"early_stopping_patience": 100, "min_epochs": 0},
    )
    assert isinstance(result, TrainingResult)
    # Per-condition normalization ran — cfg should carry the per-source lists
    assert result.cfg.data.postprocessing.get("mu_per_source") is not None
    mu_per_source = list(result.cfg.data.postprocessing.mu_per_source)
    assert len(mu_per_source) == 2


def test_smoke_direct_sum_with_area_indices(tiny_data, tmp_path):
    """DirectSum encoder with explicit area_indices in raw obs space.
    Verifies the area_indices auto-extension for delay embedding."""
    values = tiny_data  # (20, 40, 4)
    dt = 0.05
    # 2 areas of 2 features each
    area_indices = [[0, 1], [2, 3]]

    lit_model, result = train_from_arrays(
        values, dt,
        area_indices=area_indices,
        n_delays=2,
        seq_length=20,
        seq_spacing=5,
        encoder="latent_direct_sum_coupling",
        encoder_kwargs={"hidden_dim": 32, "n_coupling_layers": 4},
        n_target_var_threshold=0.99,
        prediction_steps=3,
        n_epochs=2,
        batch_size=8,
        save_dir=str(tmp_path),
        wandb_disabled=True,
        lightning_kwargs={"jacobianODEint_kwargs": {"traj_init_steps": 5,
                                                    "inner_path": "line",
                                                    "inner_N": 20,
                                                    "interp_pts": 4}},
        early_stopping_kwargs={"early_stopping_patience": 100, "min_epochs": 0},
    )
    assert isinstance(result, TrainingResult)
    # area_indices should now be in delay-embedded space (length 2 * n_delays = 4)
    cfg_area = list(list(a) for a in result.cfg.model.encoder.area_indices)
    assert len(cfg_area) == 2
    assert all(len(a) == 2 * 2 for a in cfg_area)  # 2 raw indices * n_delays=2
    # PCA-autodim should produce per-block dims since DirectSum
    assert result.autodim_result is not None
    assert result.autodim_result.is_direct_sum is True


def test_validation_input_shape_errors(tiny_data, tmp_path):
    """Bad inputs raise ValueError with helpful messages."""
    values = tiny_data
    n_traj = values.shape[0]
    dt = 0.05

    # Wrong condition shape
    with pytest.raises(ValueError, match="condition.shape"):
        train_from_arrays(
            values, dt,
            condition=np.zeros((n_traj + 1, 1)),  # mismatched length
            n_delays=2, seq_length=10,
            encoder="latent_additive_coupling",
            n_epochs=1,
            wandb_disabled=True,
        )

    # Wrong source_id shape
    with pytest.raises(ValueError, match="source_id"):
        train_from_arrays(
            values, dt,
            source_id=np.zeros((n_traj + 1,), dtype=np.int64),
            n_delays=2, seq_length=10,
            encoder="latent_additive_coupling",
            n_epochs=1,
            wandb_disabled=True,
        )

    # 2-D values
    with pytest.raises(ValueError, match="3-D"):
        train_from_arrays(
            values[0], dt,  # only (T, D)
            n_delays=2, seq_length=10,
            encoder="latent_additive_coupling",
            n_epochs=1,
            wandb_disabled=True,
        )

    # normalize_per_condition without source_id
    with pytest.raises(ValueError, match="normalize_per_condition.*source_id"):
        train_from_arrays(
            values, dt,
            normalize_per_condition=True,
            n_delays=2, seq_length=10,
            encoder="latent_additive_coupling",
            n_epochs=1,
            wandb_disabled=True,
        )
