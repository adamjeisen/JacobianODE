"""Tests for ``extend_area_indices_for_delay_embedding`` — the helper
that lifts raw-obs per-area indices into the delay-embedded
encoder-input space."""
from __future__ import annotations

import pytest
import torch

from JacobianODE.jacobians.core.config import (
    extend_area_indices_for_delay_embedding,
)
from JacobianODE.jacobians.data.splitting import embed_signal_torch


class TestBehavior:
    def test_basic_two_areas(self):
        """Worked example from docstring."""
        out = extend_area_indices_for_delay_embedding(
            [[0, 1], [2, 3]], n_delays=3, n_features=4,
        )
        assert out == [[0, 1, 4, 5, 8, 9], [2, 3, 6, 7, 10, 11]]

    def test_n_delays_1_is_passthrough(self):
        """With n_delays=1, no extension — output equals input."""
        in_idx = [[0, 2, 5], [1, 3, 4]]
        out = extend_area_indices_for_delay_embedding(in_idx, n_delays=1, n_features=6)
        assert out == [[0, 2, 5], [1, 3, 4]]

    def test_non_contiguous_indices(self):
        """Indices need not be contiguous within an area."""
        out = extend_area_indices_for_delay_embedding(
            [[0, 3], [1, 2]], n_delays=2, n_features=4,
        )
        # area 0 (raw [0,3]) → delay 0: [0,3], delay 1: [4,7] → [0,3,4,7]
        # area 1 (raw [1,2]) → delay 0: [1,2], delay 1: [5,6] → [1,2,5,6]
        assert out == [[0, 3, 4, 7], [1, 2, 5, 6]]

    def test_three_areas(self):
        """Scales to N>2 areas."""
        out = extend_area_indices_for_delay_embedding(
            [[0], [1], [2]], n_delays=2, n_features=3,
        )
        # Each area picks 1 raw idx; with n_delays=2 each becomes 2 delay-embedded.
        # area 0: delay 0 [0], delay 1 [3]  → [0, 3]
        # area 1: delay 0 [1], delay 1 [4]  → [1, 4]
        # area 2: delay 0 [2], delay 1 [5]  → [2, 5]
        assert out == [[0, 3], [1, 4], [2, 5]]


class TestLayoutMatchesEmbedSignalTorch:
    """The whole point of this helper is that the indices it produces
    correctly select per-area data from a delay-embedded tensor. Verify
    end-to-end against the actual delay-embedding routine."""

    def test_indices_select_correct_area_data(self):
        """Build raw data with distinguishable per-area values, delay-embed it,
        then verify our extended indices select exactly each area's chunks."""
        # Raw shape (1, 5, 4): T=5 timesteps, D=4 features.
        # Area 0 = features [0, 1] with values 100+ (visual)
        # Area 1 = features [2, 3] with values 200+ (cognitive)
        raw = torch.zeros(1, 5, 4)
        for t in range(5):
            raw[0, t, 0] = 100 + t       # vis ch 0 over time
            raw[0, t, 1] = 110 + t       # vis ch 1
            raw[0, t, 2] = 200 + t       # cog ch 0
            raw[0, t, 3] = 210 + t       # cog ch 1

        n_delays = 3
        embedded = embed_signal_torch(raw, n_delays=n_delays)
        # Embedded shape: (1, 5 - 3 + 1, 4*3) = (1, 3, 12)
        assert embedded.shape == (1, 3, 12)

        # Helper-extended area_indices for delay-embedded space
        area_indices = extend_area_indices_for_delay_embedding(
            [[0, 1], [2, 3]], n_delays=n_delays, n_features=4,
        )
        vis_de = embedded[..., area_indices[0]]
        cog_de = embedded[..., area_indices[1]]

        # Visual area (raw 0,1) is in the 100s; cognitive (raw 2,3) in the 200s.
        # After our extension, ALL values in vis_de should be 100s, ALL in cog_de should be 200s.
        assert vis_de.min() >= 100 and vis_de.max() < 200
        assert cog_de.min() >= 200 and cog_de.max() < 300


class TestValidation:
    def test_n_delays_zero_raises(self):
        with pytest.raises(ValueError, match="n_delays must be >= 1"):
            extend_area_indices_for_delay_embedding([[0, 1]], n_delays=0, n_features=4)

    def test_n_delays_negative_raises(self):
        with pytest.raises(ValueError, match="n_delays must be >= 1"):
            extend_area_indices_for_delay_embedding([[0, 1]], n_delays=-1, n_features=4)

    def test_empty_area_indices_raises(self):
        with pytest.raises(ValueError, match="area_indices is empty"):
            extend_area_indices_for_delay_embedding([], n_delays=2, n_features=4)

    def test_empty_inner_list_raises(self):
        with pytest.raises(ValueError, match="empty index list"):
            extend_area_indices_for_delay_embedding(
                [[0, 1], []], n_delays=2, n_features=4,
            )

    def test_index_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            extend_area_indices_for_delay_embedding(
                [[0, 5]], n_delays=2, n_features=4,
            )

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            extend_area_indices_for_delay_embedding(
                [[-1, 0]], n_delays=2, n_features=4,
            )
