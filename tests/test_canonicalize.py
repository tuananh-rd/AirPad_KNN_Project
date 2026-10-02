"""Tests for src.features.canonicalize – normalization pipeline.

Deterministic, no randomness. Covers:
- Wrist origin translation
- Scale normalization (including epsilon guard)
- Left-hand mirroring
- Rotation normalization (middle-finger-MCP → +y)
- Z-axis weighting
- Round-trip shape invariant
- Feature version constant
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.features.canonicalize import canonicalize, FEATURE_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_hand(wrist=(0.0, 0.0, 0.0), mcp9=(0.0, 1.0, 0.0), fill_value=0.5):
    """Create a (21,3) landmark array with explicit wrist and MCP-9."""
    arr = np.full((21, 3), fill_value, dtype=float)
    arr[0] = wrist
    arr[9] = mcp9
    return arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFeatureVersion:
    def test_version_is_int(self):
        assert isinstance(FEATURE_VERSION, int)
        assert FEATURE_VERSION >= 1


class TestOutputShape:
    def test_output_shape_63(self):
        hand = _make_hand()
        out = canonicalize(hand)
        assert out.shape == (63,), f"Expected (63,), got {out.shape}"

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            canonicalize(np.zeros((20, 3)))
        with pytest.raises(ValueError):
            canonicalize(np.zeros((21, 2)))


class TestWristOrigin:
    def test_wrist_becomes_origin(self):
        hand = _make_hand(wrist=(3.0, 5.0, 1.0))
        out = canonicalize(hand, normalize_rotation=False).reshape(21, 3)
        np.testing.assert_allclose(out[0, :2], [0.0, 0.0], atol=1e-10)

    def test_translation_preserves_relative_distances(self):
        hand = _make_hand(wrist=(10.0, 20.0, 0.0), mcp9=(10.0, 21.0, 0.0))
        out = canonicalize(hand, normalize_rotation=False, z_weight=1.0).reshape(21, 3)
        # After wrist-origin + scale, mcp9 direction should be preserved
        # mcp9-wrist = (0, 1, 0), which is already along +y
        assert out[9, 1] > 0  # y > 0 for mcp9


class TestScaleNormalization:
    def test_max_distance_is_one(self):
        hand = np.zeros((21, 3))
        hand[0] = [0, 0, 0]
        hand[8] = [0, 5, 0]  # index tip far away
        out = canonicalize(hand, normalize_rotation=False, z_weight=1.0).reshape(21, 3)
        distances = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(distances.max(), 1.0, atol=1e-6)

    def test_epsilon_guard_zero_hand(self):
        """All landmarks at same point → epsilon prevents division by zero."""
        hand = np.zeros((21, 3))
        out = canonicalize(hand, normalize_rotation=False)
        assert np.all(np.isfinite(out))


class TestMirrorHandedness:
    def test_left_hand_mirror_flips_x(self):
        hand = _make_hand(wrist=(0, 0, 0), mcp9=(0, 1, 0))
        hand[5] = [1.0, 0.5, 0.0]  # index MCP on +x side

        out_right = canonicalize(hand, handedness='Right', mirror_to_right=True,
                                  normalize_rotation=False, z_weight=1.0).reshape(21, 3)
        out_left = canonicalize(hand, handedness='Left', mirror_to_right=True,
                                 normalize_rotation=False, z_weight=1.0).reshape(21, 3)

        # x-coords should be negated (after translation+scaling)
        np.testing.assert_allclose(out_left[:, 0], -out_right[:, 0], atol=1e-10)
        np.testing.assert_allclose(out_left[:, 1], out_right[:, 1], atol=1e-10)

    def test_no_mirror_when_disabled(self):
        hand = _make_hand()
        out_a = canonicalize(hand, handedness='Left', mirror_to_right=False,
                              normalize_rotation=False, z_weight=1.0)
        out_b = canonicalize(hand, handedness='Right', mirror_to_right=False,
                              normalize_rotation=False, z_weight=1.0)
        np.testing.assert_allclose(out_a, out_b, atol=1e-10)


class TestRotationNormalization:
    def test_mcp9_aligned_to_positive_y(self):
        """After rotation, MCP-9 should lie on the +y axis (x≈0)."""
        hand = np.zeros((21, 3))
        hand[9] = [1.0, 1.0, 0.0]  # 45° angle
        out = canonicalize(hand, normalize_rotation=True, z_weight=1.0).reshape(21, 3)
        # MCP-9's x should be ≈0 (aligned to y-axis)
        assert abs(out[9, 0]) < 1e-6, f"MCP-9 x={out[9, 0]}, expected ~0"
        assert out[9, 1] > 0, "MCP-9 y should be positive"

    def test_rotation_preserves_distances(self):
        """2D rotation should preserve Euclidean distances."""
        np.random.seed(42)
        hand = np.random.randn(21, 3) * 0.1
        hand[0] = [0, 0, 0]  # wrist at origin
        hand[9] = [0.3, 0.4, 0.0]

        out_rot = canonicalize(hand, normalize_rotation=True, z_weight=1.0).reshape(21, 3)
        out_norot = canonicalize(hand, normalize_rotation=False, z_weight=1.0).reshape(21, 3)

        # Distances between any two points should be preserved (in xy-plane)
        for i in range(5):
            for j in range(i + 1, 5):
                d_rot = np.linalg.norm(out_rot[i, :2] - out_rot[j, :2])
                d_norot = np.linalg.norm(out_norot[i, :2] - out_norot[j, :2])
                np.testing.assert_allclose(d_rot, d_norot, atol=1e-6)


class TestZWeight:
    def test_z_weighted(self):
        hand = np.zeros((21, 3))
        hand[0] = [0, 0, 0]
        hand[9] = [0, 1, 0]
        hand[5] = [0, 0.5, 1.0]  # z=1.0

        out = canonicalize(hand, z_weight=0.3, normalize_rotation=False).reshape(21, 3)
        # After scale normalization, z of landmark 5 should be 0.3 * (original_z / scale)
        # The key test: z is reduced by z_weight factor
        out_full = canonicalize(hand, z_weight=1.0, normalize_rotation=False).reshape(21, 3)
        np.testing.assert_allclose(out[:, 2], out_full[:, 2] * 0.3, atol=1e-10)


class TestDeterminism:
    def test_same_input_same_output(self):
        np.random.seed(123)
        hand = np.random.randn(21, 3)
        out1 = canonicalize(hand.copy())
        out2 = canonicalize(hand.copy())
        np.testing.assert_array_equal(out1, out2)
