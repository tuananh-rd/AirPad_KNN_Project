"""Deterministic unit and integration tests for src.motion modules.

Covers:
- EMAFilter and OneEuroFilter (smoothing, edge cases, speed adaptivity, resets, seconds contract, non-monotonic guards)
- CursorMapper (ROI normalization, camera mirroring, clamping, dead-zone, gain, reset_baseline, virtual clock when timestamp=None)
- ScrollEngine (dominant-axis lock, dead-zone, gain, rate limit, first-event dispatch, reset)
- ZoomEngine (aperture tracking, semantic zoom steps, dead-zone, cooldown, reset)
- SwipeEngine (displacement, velocity, one-shot emission, cooldown, re-arm)
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.motion.filters import EMAFilter, OneEuroFilter
from src.motion.cursor import CursorMapper
from src.motion.scroll import ScrollEngine
from src.motion.zoom import ZoomEngine
from src.motion.swipe import SwipeEngine


# ===================================================================
# 1. EMAFilter Tests
# ===================================================================
class TestEMAFilter:
    def test_initial_value(self):
        f = EMAFilter(alpha=0.5)
        res = f.filter(10.0, 20.0)
        assert res == (10.0, 20.0)
        assert f.value == (10.0, 20.0)

    def test_smoothing_step_response(self):
        f = EMAFilter(alpha=0.5)
        f.filter(0.0, 0.0)
        # Step input of (10.0, 10.0):
        # Frame 1: 0.5 * 10 + 0.5 * 0 = 5.0
        v1 = f.filter(10.0, 10.0)
        assert math.isclose(v1[0], 5.0)
        assert math.isclose(v1[1], 5.0)

        # Frame 2: 0.5 * 10 + 0.5 * 5 = 7.5
        v2 = f.filter(10.0, 10.0)
        assert math.isclose(v2[0], 7.5)

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            EMAFilter(alpha=0.0)
        with pytest.raises(ValueError):
            EMAFilter(alpha=1.5)

    def test_reset(self):
        f = EMAFilter(alpha=0.5)
        f.filter(5.0, 5.0)
        f.reset()
        assert f.value is None
        # Next value should be treated as initial
        assert f.filter(8.0, 9.0) == (8.0, 9.0)


# ===================================================================
# 2. OneEuroFilter Tests
# ===================================================================
class TestOneEuroFilter:
    def test_initial_value(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        res = f.filter(100.0, 200.0, timestamp=0.0)
        assert res == (100.0, 200.0)

    def test_jitter_reduction_at_low_speed(self):
        """Under micro-tremors (low speed), filter should heavily smooth the signal."""
        f = OneEuroFilter(min_cutoff=0.5, beta=0.001)
        f.filter(0.0, 0.0, timestamp=0.0)

        # Oscillate with micro tremors: 0.1, -0.1, 0.1, -0.1
        t = 0.0
        outputs = []
        for _ in range(10):
            t += 0.016  # ~60 fps in seconds
            val = f.filter(0.1, 0.1, timestamp=t)
            outputs.append(val[0])

        assert outputs[-1] < 0.1

    def test_responsiveness_at_high_speed(self):
        """At high speed, beta increases cutoff frequency to track rapidly with low lag."""
        f_low_beta = OneEuroFilter(min_cutoff=1.0, beta=0.0001)
        f_high_beta = OneEuroFilter(min_cutoff=1.0, beta=1.0)

        f_low_beta.filter(0.0, 0.0, timestamp=0.0)
        f_high_beta.filter(0.0, 0.0, timestamp=0.0)

        # Large jump at t=0.016 (16ms converted to seconds)
        out_low = f_low_beta.filter(100.0, 0.0, timestamp=0.016)[0]
        out_high = f_high_beta.filter(100.0, 0.0, timestamp=0.016)[0]

        assert out_high > out_low

    def test_seconds_contract_and_backward_timestamp_guard(self):
        """Caller passes timestamps in seconds (16ms = 0.016s).

        Backward timestamp (dt <= 0) must return previous value and NOT update internal baseline.
        """
        f = OneEuroFilter()
        # Initial at t=1.0s
        f.filter(10.0, 10.0, timestamp=1.0)

        # Step forward at t=1.016s (16ms delta)
        step1 = f.filter(20.0, 20.0, timestamp=1.016)
        assert step1[0] > 10.0

        # Backward timestamp at t=0.5s: must NOT update internal baseline and return step1
        backward = f.filter(50.0, 50.0, timestamp=0.5)
        assert backward == step1

        # Forward from 1.016s to 1.032s (another 16ms step): must compute forward from 1.016s!
        step2 = f.filter(25.0, 25.0, timestamp=1.032)
        assert step2[0] > step1[0]

    def test_non_monotonic_or_zero_dt(self):
        f = OneEuroFilter()
        f.filter(10.0, 10.0, timestamp=1.0)
        # Duplicate timestamp: should safely return previous without crashing
        res = f.filter(20.0, 20.0, timestamp=1.0)
        assert res == (10.0, 10.0)

    def test_reset(self):
        f = OneEuroFilter()
        f.filter(10.0, 10.0, timestamp=1.0)
        f.reset()
        res = f.filter(50.0, 60.0, timestamp=0.0)
        assert res == (50.0, 60.0)


# ===================================================================
# 3. CursorMapper Tests
# ===================================================================
class TestCursorMapper:
    def test_default_map_point_advances_without_timestamp(self):
        """map_point(.2,.5) then map_point(.8,.5) with timestamp=None must NOT return same coordinates."""
        mapper = CursorMapper()  # default filter is one_euro
        p1 = mapper.map_point(0.2, 0.5)
        p2 = mapper.map_point(0.8, 0.5)
        # Coordinates must change when input moves from 0.2 to 0.8
        assert p1 != p2
        # With default mirror_x=True, moving raw x from 0.2 to 0.8 moves cursor from right to left
        assert p1[0] != p2[0]

    def test_roi_and_screen_mapping(self):
        # ROI: x in [0.2, 0.8], y in [0.2, 0.8], Screen: 1000 x 1000, mirror_x=False, no filter
        mapper = CursorMapper(
            screen_size=(1000, 1000),
            roi_x_min=0.2,
            roi_x_max=0.8,
            roi_y_min=0.2,
            roi_y_max=0.8,
            mirror_x=False,
            dead_zone=0.0,
            gain_x=1.0,
            gain_y=1.0,
            filter_type="none",
        )
        # Center of ROI: (0.5, 0.5) -> normalized (0.5, 0.5) -> screen (500, 500)
        pos = mapper.map_point(0.5, 0.5)
        assert abs(pos[0] - 500) <= 1
        assert abs(pos[1] - 500) <= 1

        # Min of ROI: (0.2, 0.2) -> screen (0, 0)
        pos_min = mapper.map_point(0.2, 0.2)
        assert pos_min == (0, 0)

        # Max of ROI: (0.8, 0.8) -> screen (999, 999)
        pos_max = mapper.map_point(0.8, 0.8)
        assert pos_max == (999, 999)

    def test_camera_mirroring(self):
        # mirror_x=True: x=0.2 becomes 1.0 - 0.2 = 0.8
        mapper = CursorMapper(
            screen_size=(1000, 1000),
            roi_x_min=0.0,
            roi_x_max=1.0,
            roi_y_min=0.0,
            roi_y_max=1.0,
            mirror_x=True,
            dead_zone=0.0,
            filter_type="none",
        )
        pos = mapper.map_point(0.2, 0.5)
        # Mirrored x is 0.8 -> screen_x ~ 800
        assert abs(pos[0] - 800) <= 1

    def test_clamping_beyond_roi(self):
        mapper = CursorMapper(
            screen_size=(1920, 1080),
            roi_x_min=0.2,
            roi_x_max=0.8,
            roi_y_min=0.2,
            roi_y_max=0.8,
            mirror_x=False,
            filter_type="none",
        )
        pos = mapper.map_point(0.05, 0.95)
        assert pos[0] == 0
        assert pos[1] == 1079

    def test_dead_zone(self):
        mapper = CursorMapper(
            screen_size=(1000, 1000),
            roi_x_min=0.0,
            roi_x_max=1.0,
            roi_y_min=0.0,
            roi_y_max=1.0,
            mirror_x=False,
            dead_zone=0.01,
            filter_type="none",
        )
        p1 = mapper.map_point(0.5, 0.5)
        p2 = mapper.map_point(0.503, 0.504)
        assert p1 == p2

    def test_reset_baseline_prevents_jump(self):
        mapper = CursorMapper(
            screen_size=(1000, 1000),
            roi_x_min=0.0,
            roi_x_max=1.0,
            roi_y_min=0.0,
            roi_y_max=1.0,
            mirror_x=False,
            dead_zone=0.05,
            filter_type="none",
        )
        mapper.map_point(0.2, 0.2)
        mapper.reset_baseline()
        assert mapper._prev_roi_pos is None
        pos = mapper.map_point(0.8, 0.8)
        assert abs(pos[0] - 800) <= 1


# ===================================================================
# 4. ScrollEngine Tests
# ===================================================================
class TestScrollEngine:
    def test_dead_zone_suppression(self):
        engine = ScrollEngine(dead_zone=0.01, gain=100.0)
        steps = engine.update(dx=0.003, dy=0.004, timestamp_ms=100.0)
        assert steps == (0, 0)

    def test_dominant_axis_lock_y(self):
        engine = ScrollEngine(dead_zone=0.001, gain=100.0, axis_lock_ratio=1.2, rate_limit_interval_ms=0.0)
        steps = engine.update(dx=0.01, dy=0.05, timestamp_ms=50.0)
        assert engine.locked_axis == "Y"
        assert steps[0] == 0
        assert steps[1] > 0

        steps2 = engine.update(dx=0.03, dy=0.04, timestamp_ms=100.0)
        assert steps2[0] == 0
        assert steps2[1] > 0

    def test_dominant_axis_lock_x(self):
        engine = ScrollEngine(dead_zone=0.001, gain=100.0, axis_lock_ratio=1.2, rate_limit_interval_ms=0.0)
        steps = engine.update(dx=0.08, dy=0.01, timestamp_ms=50.0)
        assert engine.locked_axis == "X"
        assert steps[0] > 0
        assert steps[1] == 0

    def test_rate_limiting(self):
        """First event at t=10ms must NOT be suppressed by rate-limit."""
        engine = ScrollEngine(dead_zone=0.001, gain=100.0, rate_limit_interval_ms=50.0)
        s1 = engine.update(dx=0.0, dy=0.1, timestamp_ms=10.0)
        assert s1[1] > 0, "First event must emit without false rate-limit suppression"

        # Too soon (10ms < 50ms interval) -> must return (0, 0)
        s2 = engine.update(dx=0.0, dy=0.1, timestamp_ms=20.0)
        assert s2 == (0, 0)

        # After interval (65ms > 50ms) -> emits
        s3 = engine.update(dx=0.0, dy=0.1, timestamp_ms=65.0)
        assert s3[1] > 0

    def test_reset(self):
        engine = ScrollEngine()
        engine.update(dx=0.0, dy=0.08, timestamp_ms=10.0)
        assert engine.locked_axis == "Y"
        engine.reset()
        assert engine.locked_axis is None
        assert engine._last_scroll_time_ms is None


# ===================================================================
# 5. ZoomEngine Tests
# ===================================================================
class TestZoomEngine:
    def test_first_frame_captures_baseline(self):
        z = ZoomEngine(step_threshold=0.04)
        steps = z.update(current_distance=0.15, timestamp_ms=100.0)
        assert steps == 0
        assert z.baseline == 0.15

    def test_zoom_in_and_out(self):
        z = ZoomEngine(step_threshold=0.03, dead_zone=0.005, cooldown_ms=50.0)
        z.update(current_distance=0.10, timestamp_ms=100.0)

        step_in = z.update(current_distance=0.165, timestamp_ms=160.0)
        assert step_in == 2

        step_out = z.update(current_distance=0.10, timestamp_ms=220.0)
        assert step_out < 0

    def test_dead_zone_and_cooldown(self):
        z = ZoomEngine(step_threshold=0.03, dead_zone=0.01, cooldown_ms=100.0)
        z.update(current_distance=0.20, timestamp_ms=0.0)

        assert z.update(current_distance=0.205, timestamp_ms=50.0) == 0

        z.update(current_distance=0.25, timestamp_ms=150.0)
        assert z.update(current_distance=0.30, timestamp_ms=180.0) == 0

    def test_compute_aperture_helper(self):
        landmarks = np.zeros((21, 3))
        landmarks[4] = [0.1, 0.2, 0.0]  # thumb tip
        landmarks[8] = [0.4, 0.6, 0.0]  # index tip (dx=0.3, dy=0.4 -> dist=0.5)
        aperture = ZoomEngine.compute_aperture(landmarks)
        assert math.isclose(aperture, 0.5, rel_tol=1e-5)


# ===================================================================
# 6. SwipeEngine Tests
# ===================================================================
class TestSwipeEngine:
    def test_one_shot_forward_swipe(self):
        engine = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=300.0)
        assert engine.update(x=0.2, timestamp_ms=0.0) is None

        event = engine.update(x=0.32, timestamp_ms=100.0)
        assert event == "FORWARD"

        assert engine.update(x=0.35, timestamp_ms=120.0) is None
        assert engine.update(x=0.36, timestamp_ms=200.0) is None

    def test_one_shot_back_swipe(self):
        engine = SwipeEngine(min_displacement=0.08, min_velocity=0.0003)
        engine.update(x=0.8, timestamp_ms=0.0)
        event = engine.update(x=0.65, timestamp_ms=100.0)
        assert event == "BACK"
        assert engine.update(x=0.60, timestamp_ms=150.0) is None

    def test_slow_movement_rejected(self):
        engine = SwipeEngine(min_displacement=0.08, min_velocity=0.001)
        engine.update(x=0.2, timestamp_ms=0.0)
        event = engine.update(x=0.30, timestamp_ms=1000.0)
        assert event is None

    def test_rearm_after_pose_exit(self):
        engine = SwipeEngine(min_displacement=0.08, min_velocity=0.0003)
        engine.update(x=0.2, timestamp_ms=0.0)
        assert engine.update(x=0.32, timestamp_ms=100.0) == "FORWARD"

        engine.update(x=0.32, timestamp_ms=150.0, is_three_fingers=False)
        assert engine.is_armed

        engine.update(x=0.5, timestamp_ms=200.0, is_three_fingers=True)
        assert engine.update(x=0.62, timestamp_ms=300.0, is_three_fingers=True) == "FORWARD"
