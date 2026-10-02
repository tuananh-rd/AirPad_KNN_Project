"""Comprehensive tests for Hybrid Assist Profile (HYB-01).

Invariants verified:
1. Assist starts in OFF state by default in hybrid profile.
2. SPACE key / toggle_assist toggles between OFF and READY; cannot clear latched emergency stop.
3. In OFF state, zero platform actions are emitted for any pose.
4. Physical mouse ownership: For ALL 9 poses, zero MOVE, CLICK, DOWN, UP, ZOOM, or SWIPE (Alt+arrows) are emitted.
5. TWO_FINGERS_PARALLEL directly scrolls vertically without POINT_ONE arming:
   - Direction: Hand UP in mirrored preview -> positive dy (scroll UP).
   - Direction: Hand DOWN in mirrored preview -> negative dy (scroll DOWN).
   - Slow cumulative motion (100 frames of dy=0.002) accumulates without dead-zone drop (emits ~23 units).
   - Stationary jitter (+-0.0015 over 100 frames) emits 0 scroll units.
   - Forward-return subthreshold recoil emits 0 reverse scroll units.
   - Gain parity across 15, 30, and 60 FPS within bounded tolerance (+-1 step).
   - Horizontal scroll is suppressed.
6. THREE_FINGERS left/right swipe switches tabs:
   - Preview LEFT -> PREV tab (Ctrl+Shift+Tab).
   - Preview RIGHT -> NEXT tab (Ctrl+Tab).
   - One-shot emission (does not re-fire while held).
   - Hybrid mode rearms ONLY on observed pose exit, not neutral return while still THREE.
   - Legacy mode preserves neutral return rearm without pose exit.
   - Cooldown and rearm verification across exit, re-entry, hand loss, and assist toggle.
7. WindowsController & DryRunController switch_tab:
   - Modifier tracking: ctrl, shift, tab in _held_keys before OS press.
   - Modifiers released in finally block.
   - Fault injection: partial keyDown failure attempts immediate modifier release in finally.
   - Fault injection: unconfirmed press retains key in _held_keys for cleanup retry.
   - Fault injection: modifier keyUp failure retains key for cleanup retry.
   - Stop check: pre-stopped and reentrant gate stop prevents dispatch.
8. Classifier absence fail-closed contract:
   - When classifier is None, zero actions are emitted across all poses.
9. Mocked Camera SPACE/HUD and CLI profile:
   - Camera loop SPACE key toggles assist state.
   - Camera loop renders HUD overlay with profile and assist state.
   - CLI --profile {legacy,hybrid} parsed correctly.
   - Real control banner warns that gestures trigger scroll/tabs (not move/click) and corner stop is unavailable.
10. Absolute test isolation: globally mocked PyAutoGUI and keyboard, 0 real OS events.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# Ensure mocked PyAutoGUI and keyboard before any imports
mock_pyautogui = MagicMock()
mock_pyautogui.size.return_value = (1920, 1080)
mock_keyboard = MagicMock()
sys.modules['pyautogui'] = mock_pyautogui
sys.modules['keyboard'] = mock_keyboard

from app import create_pipeline, main, run_camera
from src.gesture.pose import HandPose
from src.gesture.fsm import FSMState
from src.gesture.temporal import MajorityFilter, StabilityGate
from src.motion.scroll import ScrollEngine
from src.motion.swipe import SwipeEngine
from src.platform.dry_run import DryRunController
from src.platform.interface import KeyEvent, MouseEvent
from src.platform.windows import WindowsController
from src.runtime.pipeline import AirPadPipeline, PipelineResult
from src.safety.failsafe import SafetyGate


def _reset_mock_state():
    mock_pyautogui.reset_mock()
    mock_keyboard.reset_mock()
    for attr in ("keyDown", "keyUp", "press", "mouseUp", "mouseDown", "moveTo", "scroll", "hscroll", "click"):
        target = getattr(mock_pyautogui, attr, None)
        if target is not None:
            target.side_effect = None
            target.reset_mock()
    mock_pyautogui.size.side_effect = None
    mock_pyautogui.size.return_value = (1920, 1080)


@pytest.fixture(autouse=True)
def reset_os_mocks():
    _reset_mock_state()
    try:
        yield
    finally:
        _reset_mock_state()


class FakeVideoCapture:
    def __init__(self, frames=None, is_opened=True):
        self._frames = list(frames) if frames is not None else []
        self._is_opened = is_opened
        self.released = False
        self.idx = 0

    def isOpened(self) -> bool:
        return self._is_opened

    def read(self):
        if not self._is_opened or self.idx >= len(self._frames):
            return False, None
        frame = self._frames[self.idx]
        self.idx += 1
        return True, frame

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, detected_hands=None):
        self.detected_hands = list(detected_hands) if detected_hands is not None else []
        self.frames_received = []
        self.closed = False
        self.idx = 0

    def detect(self, frame: np.ndarray):
        self.frames_received.append(frame.copy())
        if self.idx < len(self.detected_hands):
            res = self.detected_hands[self.idx]
            self.idx += 1
            return res
        return []

    def close(self):
        self.closed = True


class MockLabelClassifier:
    """Label-controlled classifier stub for deterministic testing."""
    def __init__(self, pose: str = "TWO_FINGERS_PARALLEL"):
        self.pose = pose

    def predict_with_rejection(self, *args, **kwargs):
        return [self.pose]


_DEFAULT_CLASSIFIER = object()


def _make_dummy_landmarks(index_y: float = 0.5, middle_y: float = 0.5, wrist_x: float = 0.5) -> np.ndarray:
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = [wrist_x, 0.8, 0.0]    # wrist
    lms[8] = [0.5, index_y, 0.0]   # index tip
    lms[12] = [0.5, middle_y, 0.0] # middle tip
    return lms


def _make_test_pipeline(
    controller=None,
    scroll_engine=None,
    swipe_engine=None,
    profile="hybrid",
    gate=None,
    classifier=_DEFAULT_CLASSIFIER,
) -> AirPadPipeline:
    clf = MockLabelClassifier("TWO_FINGERS_PARALLEL") if classifier is _DEFAULT_CLASSIFIER else classifier
    return AirPadPipeline(
        config={},
        controller=controller or DryRunController(),
        classifier=clf,
        scroll_engine=scroll_engine,
        swipe_engine=swipe_engine,
        majority_filter=MajorityFilter(window_size=5, majority_ratio=0.6),
        stability_gate=StabilityGate(required_count=2),
        gate=gate,
        profile=profile,
    )


# ===================================================================
# 1. Profile Initialization & Assist Toggle
# ===================================================================
class TestHybridProfileState:
    def test_default_hybrid_starts_in_off_state(self):
        pipe = _make_test_pipeline(profile="hybrid")
        assert pipe.profile == "hybrid"
        assert pipe.assist_state == "OFF"

    def test_legacy_profile_starts_in_legacy_state(self):
        pipe = _make_test_pipeline(profile="legacy")
        assert pipe.profile == "legacy"
        assert pipe.assist_state == "LEGACY"

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            _make_test_pipeline(profile="invalid_profile")

    def test_toggle_assist_cycles_off_and_ready(self):
        pipe = _make_test_pipeline(profile="hybrid")
        assert pipe.assist_state == "OFF"

        state1 = pipe.toggle_assist()
        assert state1 == "READY"
        assert pipe.assist_state == "READY"

        state2 = pipe.toggle_assist()
        assert state2 == "OFF"
        assert pipe.assist_state == "OFF"

    def test_toggle_assist_cannot_clear_emergency_stop(self):
        pipe = _make_test_pipeline(profile="hybrid")
        pipe.trigger_emergency_stop()
        assert pipe.is_emergency_stopped is True

        res = pipe.toggle_assist()
        assert res == "EMERGENCY_STOPPED"
        assert pipe.is_emergency_stopped is True
        assert pipe.assist_state == "EMERGENCY_STOPPED"

    def test_off_state_suppresses_all_gestures(self):
        ctrl = DryRunController()
        pipe = _make_test_pipeline(controller=ctrl, profile="hybrid")
        assert pipe.assist_state == "OFF"

        # Even with hand present and valid coordinates, zero events emitted while OFF
        lms = _make_dummy_landmarks(index_y=0.4, middle_y=0.4)
        for t in range(10):
            res = pipe.process_landmarks(lms, timestamp_ms=t * 33.3)
            assert res.action_fired is None
        assert len(ctrl.event_log) == 0

    def test_classifier_absence_fails_closed(self):
        """When classifier is None, hybrid mode must fail closed (zero actions dispatched)."""
        ctrl = DryRunController()
        pipe = _make_test_pipeline(controller=ctrl, profile="hybrid", classifier=None)
        pipe.toggle_assist()  # READY
        assert pipe.classifier is None

        lms = _make_dummy_landmarks(index_y=0.40, middle_y=0.40)
        # Even if temporal filters output TWO_FINGERS_PARALLEL or THREE_FINGERS:
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)):
            res = pipe.process_landmarks(lms, timestamp_ms=100.0)
            assert res.action_fired is None
            assert len(ctrl.event_log) == 0

        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
            res = pipe.process_landmarks(lms, timestamp_ms=200.0)
            assert res.action_fired is None
            assert len(ctrl.event_log) == 0


# ===================================================================
# 2. Physical Mouse Ownership: Zero MOVE, CLICK, DOWN, UP, ZOOM
# ===================================================================
class TestPhysicalMouseOwnership:
    def test_all_poses_emit_zero_mouse_actions_in_hybrid(self):
        ctrl = DryRunController()
        gate = SafetyGate()
        clf = MockLabelClassifier("UNKNOWN")
        pipe = _make_test_pipeline(controller=ctrl, gate=gate, profile="hybrid", classifier=clf)
        pipe.toggle_assist()  # assist READY

        all_test_poses = [
            HandPose.POINT_ONE,
            HandPose.PINCH_CLOSED,
            HandPose.PINCH_MIDDLE,
            HandPose.OPEN_PALM,
            HandPose.FIST,
            HandPose.L_SHAPE,
            HandPose.OTHER,
            HandPose.UNKNOWN,
        ]

        lms = _make_dummy_landmarks()

        for pose in all_test_poses:
            clf.pose = pose.name
            with patch.object(pipe.majority_filter, 'update', return_value=pose), \
                 patch.object(pipe.stability_gate, 'update', return_value=(pose, True)):
                res = pipe.process_landmarks(lms, timestamp_ms=100.0)

                assert res.cursor_pos is None, f"Pose {pose.name} must NOT set cursor_pos"
                assert res.action_fired is None, f"Pose {pose.name} must NOT fire actions"
                assert len(ctrl.event_log) == 0, f"Pose {pose.name} must NOT dispatch platform events"
                assert len(ctrl.held_buttons) == 0, f"Pose {pose.name} must NOT hold buttons"


# ===================================================================
# 3. Two-Finger Vertical Scroll Engine & Direction Contract
# ===================================================================
class TestHybridScroll:
    def test_direct_entry_without_point_one_arming(self):
        """In hybrid mode, TWO_FINGERS_PARALLEL scrolls directly without POINT_ONE dwell."""
        ctrl = DryRunController()
        pipe = _make_test_pipeline(
            controller=ctrl,
            scroll_engine=ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid"),
            profile="hybrid",
        )
        pipe.toggle_assist()

        # Frame 1: baseline capture
        lms1 = _make_dummy_landmarks(index_y=0.50, middle_y=0.50)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)):
            r1 = pipe.process_landmarks(lms1, timestamp_ms=0.0)
            assert r1.action_fired is None  # first frame baseline capture

        # Frame 2: hand moves UP in preview (y decreases from 0.50 to 0.40)
        lms2 = _make_dummy_landmarks(index_y=0.40, middle_y=0.40)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)):
            r2 = pipe.process_landmarks(lms2, timestamp_ms=33.3)
            assert r2.action_fired == "SCROLL"
            assert len(ctrl.event_log) == 1
            evt = ctrl.event_log[0]
            assert evt["type"] == MouseEvent.SCROLL
            assert evt["dy"] > 0, "Hand UP must produce positive scroll (scroll UP)"

    def test_hand_down_scrolls_down(self):
        ctrl = DryRunController()
        pipe = _make_test_pipeline(
            controller=ctrl,
            scroll_engine=ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid"),
            profile="hybrid",
        )
        pipe.toggle_assist()

        # Frame 1: baseline
        lms1 = _make_dummy_landmarks(index_y=0.50, middle_y=0.50)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)):
            pipe.process_landmarks(lms1, timestamp_ms=0.0)

        # Frame 2: hand moves DOWN in preview (y increases from 0.50 to 0.60)
        lms2 = _make_dummy_landmarks(index_y=0.60, middle_y=0.60)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)):
            r2 = pipe.process_landmarks(lms2, timestamp_ms=33.3)
            assert r2.action_fired == "SCROLL"
            assert len(ctrl.event_log) == 1
            evt = ctrl.event_log[0]
            assert evt["type"] == MouseEvent.SCROLL
            assert evt["dy"] < 0, "Hand DOWN must produce negative scroll (scroll DOWN)"

    def test_slow_cumulative_motion_preserved_without_deadzone_drop(self):
        """100 frames of slow dy=0.002 (total 0.200 displacement) must emit ~23 units."""
        engine = ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid", rate_limit_interval_ms=0.0)
        total_steps = 0
        for i in range(100):
            steps = engine.update(dx=0.0, dy=0.002, timestamp_ms=i * 33.3)
            total_steps += steps[1]

        # Effective displacement = 0.200 - 0.005 = 0.195
        # Expected steps = int(0.195 * 120) = 23
        assert total_steps == 23, f"Expected 23 scroll steps from cumulative motion, got {total_steps}"

    def test_stationary_jitter_suppression(self):
        """100 frames of stationary jitter (+-0.0015) around 0 must emit exactly 0 scroll units."""
        engine = ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid", rate_limit_interval_ms=0.0)
        total_steps = 0
        jitter = [0.0015, -0.0012, 0.0010, -0.0014, 0.0008, -0.0009] * 17
        for i, dy in enumerate(jitter[:100]):
            steps = engine.update(dx=0.0, dy=dy, timestamp_ms=i * 33.3)
            total_steps += abs(steps[1])

        assert total_steps == 0, f"Expected 0 steps from stationary jitter, got {total_steps}"

    def test_forward_return_subthreshold_recoil_suppressed(self):
        """After scrolling UP, a slight downward finger relaxation must NOT emit reverse scroll."""
        engine = ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid", rate_limit_interval_ms=0.0)
        # Scroll up 10 frames of 0.01 -> emits steps
        up_steps = sum(engine.update(0.0, 0.01, i * 33.3)[1] for i in range(10))
        assert up_steps > 0

        # Subthreshold recoil downward: 2 frames of -0.002
        recoil1 = engine.update(0.0, -0.002, 350.0)[1]
        recoil2 = engine.update(0.0, -0.002, 383.3)[1]
        assert recoil1 == 0
        assert recoil2 == 0

    def test_fps_gain_parity_15_30_60(self):
        """Identical distance traveled (0.200) in 1 second yields identical units at 15, 30, and 60 FPS."""
        total_dist = 0.200
        rates = [15, 30, 60]
        results = {}

        for fps in rates:
            engine = ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid", rate_limit_interval_ms=0.0)
            n_frames = fps
            dy_per_frame = total_dist / n_frames
            dt_ms = 1000.0 / fps

            steps_emitted = 0
            for i in range(n_frames):
                s = engine.update(0.0, dy_per_frame, i * dt_ms)
                steps_emitted += s[1]
            results[fps] = steps_emitted

        assert results[15] == 23
        assert results[30] == 23
        assert results[60] == 23

    def test_stationary_jitter_and_single_frame_flicker_suppressed(self):
        """End-to-end pipeline test: stationary jitter and single-frame pose flicker produce 0 scroll events."""
        ctrl = DryRunController()
        clf = MockLabelClassifier("TWO_FINGERS_PARALLEL")
        pipe = _make_test_pipeline(
            controller=ctrl,
            scroll_engine=ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid"),
            profile="hybrid",
            classifier=clf,
        )
        pipe.toggle_assist()

        # Frame 1: baseline
        lms1 = _make_dummy_landmarks(index_y=0.5000, middle_y=0.5000)
        pipe.process_landmarks(lms1, timestamp_ms=0.0)

        # Micro-jitter within +-0.0015 over 20 frames
        jitter_vals = [0.0012, -0.0011, 0.0009, -0.0014, 0.0010] * 4
        for i, dj in enumerate(jitter_vals):
            lms_j = _make_dummy_landmarks(index_y=0.5000 + dj, middle_y=0.5000 + dj)
            res = pipe.process_landmarks(lms_j, timestamp_ms=(i + 1) * 33.3)
            assert res.action_fired is None
        assert len(ctrl.event_log) == 0

        # Single frame flicker to OTHER
        clf.pose = "OTHER"
        lms_flicker = _make_dummy_landmarks(index_y=0.40, middle_y=0.40)
        res_fl = pipe.process_landmarks(lms_flicker, timestamp_ms=750.0)
        assert res_fl.action_fired is None
        assert len(ctrl.event_log) == 0


# ===================================================================
# 4. Three-Finger Tab Switching Contract
# ===================================================================
class TestHybridTabSwitch:
    def test_swipe_preview_left_triggers_prev_tab(self):
        ctrl = DryRunController()
        swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, mode="hybrid")
        pipe = _make_test_pipeline(
            controller=ctrl,
            swipe_engine=swipe_eng,
            profile="hybrid",
            classifier=MockLabelClassifier("THREE_FINGERS"),
        )
        pipe.toggle_assist()

        # Start at wrist_x = 0.50
        lms1 = _make_dummy_landmarks(wrist_x=0.50)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
            r1 = pipe.process_landmarks(lms1, timestamp_ms=100.0)
            assert r1.action_fired is None

        # Swipe LEFT in preview (dx = -0.15 < -0.08 in 50ms)
        lms2 = _make_dummy_landmarks(wrist_x=0.35)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
            r2 = pipe.process_landmarks(lms2, timestamp_ms=150.0)
            assert r2.action_fired == "TAB_PREV"

        # Verify DryRunController logged ctrl + shift + tab
        keys_down = [e["key"] for e in ctrl.event_log if e["type"] == KeyEvent.KEY_DOWN]
        keys_up = [e["key"] for e in ctrl.event_log if e["type"] == KeyEvent.KEY_UP]
        assert "ctrl" in keys_down
        assert "shift" in keys_down
        assert "tab" in keys_down
        assert "tab" in keys_up
        assert "shift" in keys_up
        assert "ctrl" in keys_up
        assert len(ctrl.held_keys) == 0

    def test_swipe_preview_right_triggers_next_tab(self):
        ctrl = DryRunController()
        swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, mode="hybrid")
        pipe = _make_test_pipeline(
            controller=ctrl,
            swipe_engine=swipe_eng,
            profile="hybrid",
            classifier=MockLabelClassifier("THREE_FINGERS"),
        )
        pipe.toggle_assist()

        # Start at wrist_x = 0.50
        lms1 = _make_dummy_landmarks(wrist_x=0.50)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
            pipe.process_landmarks(lms1, timestamp_ms=100.0)

        # Swipe RIGHT in preview (dx = +0.15 > 0.08 in 50ms)
        lms2 = _make_dummy_landmarks(wrist_x=0.65)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
            r2 = pipe.process_landmarks(lms2, timestamp_ms=150.0)
            assert r2.action_fired == "TAB_NEXT"

        # Verify DryRunController logged ctrl + tab
        keys_down = [e["key"] for e in ctrl.event_log if e["type"] == KeyEvent.KEY_DOWN]
        keys_up = [e["key"] for e in ctrl.event_log if e["type"] == KeyEvent.KEY_UP]
        assert "ctrl" in keys_down
        assert "tab" in keys_down
        assert "shift" not in keys_down
        assert "tab" in keys_up
        assert "ctrl" in keys_up
        assert len(ctrl.held_keys) == 0

    def test_tab_swipe_is_one_shot_while_held(self):
        """Holding the swiped pose must not repeatedly emit tab events."""
        ctrl = DryRunController()
        swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, mode="hybrid")
        pipe = _make_test_pipeline(
            controller=ctrl,
            swipe_engine=swipe_eng,
            profile="hybrid",
            classifier=MockLabelClassifier("THREE_FINGERS"),
        )
        pipe.toggle_assist()

        # Frame 1: start
        lms1 = _make_dummy_landmarks(wrist_x=0.50)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
            pipe.process_landmarks(lms1, timestamp_ms=100.0)

        # Frame 2: trigger
        lms2 = _make_dummy_landmarks(wrist_x=0.65)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
            r2 = pipe.process_landmarks(lms2, timestamp_ms=150.0)
            assert r2.action_fired == "TAB_NEXT"

        # Frames 3..10: held at 0.65
        for t in range(160, 300, 20):
            with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
                 patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
                r = pipe.process_landmarks(lms2, timestamp_ms=float(t))
                assert r.action_fired is None, "Tab swipe must be one-shot while held"

    def test_hybrid_swipe_rearms_only_on_pose_exit_not_neutral_return(self):
        """In hybrid mode, swipe MUST rearm only after observed pose exit,
        NOT upon neutral return while still holding THREE_FINGERS."""
        engine = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")

        # 1. Swipe right: .50 @ 1000ms -> .61 @ 1200ms -> fires NEXT
        assert engine.update(0.50, 1000.0, is_three_fingers=True) is None
        evt1 = engine.update(0.61, 1200.0, is_three_fingers=True)
        assert evt1 == "NEXT"

        # 2. Return toward neutral (.50 @ 1700ms, .49 @ 1730ms) and overshoot left (.39 @ 1830ms)
        # while CONTINUOUSLY holding THREE_FINGERS (is_three_fingers=True).
        # Invariant: Must NOT rearm, must NOT fire PREV!
        assert engine.update(0.50, 1700.0, is_three_fingers=True) is None
        assert engine.update(0.49, 1730.0, is_three_fingers=True) is None
        assert engine.update(0.39, 1830.0, is_three_fingers=True) is None
        assert not engine.is_armed

        # 3. Observed pose exit: user releases three fingers at 1850ms
        assert engine.update(0.39, 1850.0, is_three_fingers=False) is None

        # 4. Wait cooldown (400ms): re-enter THREE at 2300ms (.50), swipe left (.39 @ 2500ms)
        assert engine.update(0.50, 2300.0, is_three_fingers=True) is None
        evt2 = engine.update(0.39, 2500.0, is_three_fingers=True)
        assert evt2 == "PREV"

    def test_legacy_swipe_rearms_on_neutral_return_without_pose_exit(self):
        """In legacy mode, neutral return while still holding pose DOES rearm."""
        engine = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="legacy")

        # 1. Swipe right: .50 @ 1000ms -> .61 @ 1200ms -> fires FORWARD
        assert engine.update(0.50, 1000.0, is_three_fingers=True) is None
        evt1 = engine.update(0.61, 1200.0, is_three_fingers=True)
        assert evt1 == "FORWARD"

        # 2. Return to neutral (.50 @ 1700ms, within rearm_threshold 0.04)
        assert engine.update(0.50, 1700.0, is_three_fingers=True) is None
        assert engine.is_armed, "Legacy mode must rearm on neutral return"

        # 3. Overshoot left: frame 1 establishes baseline at 0.50, frame 2 moves left to 0.39 -> fires BACK
        assert engine.update(0.50, 1750.0, is_three_fingers=True) is None
        evt2 = engine.update(0.39, 1850.0, is_three_fingers=True)
        assert evt2 == "BACK"

    def test_assist_off_preserves_cooldown(self):
        """Cooldown timer is preserved when assist toggles OFF, preventing rapid re-fire."""
        ctrl = DryRunController()
        swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        clf = MockLabelClassifier("THREE_FINGERS")
        pipe = _make_test_pipeline(controller=ctrl, swipe_engine=swipe_eng, profile="hybrid", classifier=clf)
        pipe.toggle_assist()

        def _step(x, t_ms):
            lms = _make_dummy_landmarks(wrist_x=x)
            with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
                 patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
                return pipe.process_landmarks(lms, timestamp_ms=t_ms)

        # Swipe 1 at t=1000..1200ms -> fires NEXT
        _step(0.50, 1000.0)
        r1 = _step(0.65, 1200.0)
        assert r1.action_fired == "TAB_NEXT"

        # Toggle OFF at 1220ms, then ON at 1300ms (within 400ms cooldown)
        pipe.toggle_assist()  # OFF
        pipe.toggle_assist()  # READY

        # Attempt swipe at 1350..1450ms (only 250ms after Swipe 1)
        _step(0.50, 1350.0)
        r2 = _step(0.65, 1450.0)
        assert r2.action_fired is None
        assert len([e for e in ctrl.event_log if e["type"] == KeyEvent.KEY_DOWN and e["key"] == "tab"]) == 1

    def test_hand_loss_preserves_cooldown(self):
        """Cooldown timer is preserved across hand loss, preventing rapid re-fire."""
        ctrl = DryRunController()
        swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        clf = MockLabelClassifier("THREE_FINGERS")
        pipe = _make_test_pipeline(controller=ctrl, swipe_engine=swipe_eng, profile="hybrid", classifier=clf)
        pipe.toggle_assist()

        def _step(x, t_ms):
            lms = _make_dummy_landmarks(wrist_x=x)
            with patch.object(pipe.majority_filter, 'update', return_value=HandPose.THREE_FINGERS), \
                 patch.object(pipe.stability_gate, 'update', return_value=(HandPose.THREE_FINGERS, True)):
                return pipe.process_landmarks(lms, timestamp_ms=t_ms)

        # Swipe 1 at t=1000..1200ms -> fires NEXT
        _step(0.50, 1000.0)
        r1 = _step(0.65, 1200.0)
        assert r1.action_fired == "TAB_NEXT"

        # Hand lost at 1230ms
        pipe.process_landmarks(None, timestamp_ms=1230.0)

        # Hand re-acquired at 1350ms, attempt swipe at 1450ms (only 250ms after Swipe 1)
        _step(0.50, 1350.0)
        r2 = _step(0.65, 1450.0)
        assert r2.action_fired is None
        assert len([e for e in ctrl.event_log if e["type"] == KeyEvent.KEY_DOWN and e["key"] == "tab"]) == 1


# ===================================================================
# 5. Controller Safety & Modifiers Fault Injection
# ===================================================================
class TestControllerTabSwitchingSafety:
    def test_windows_controller_prev_tab_invokes_ctrl_shift_tab(self):
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        ctrl.switch_tab("PREV")

        mock_pyautogui.keyDown.assert_has_calls([call('ctrl'), call('shift')], any_order=False)
        mock_pyautogui.press.assert_called_once_with('tab')
        mock_pyautogui.keyUp.assert_has_calls([call('shift'), call('ctrl')], any_order=False)
        assert len(ctrl.held_keys) == 0

    def test_windows_controller_next_tab_invokes_ctrl_tab(self):
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        ctrl.switch_tab("NEXT")

        mock_pyautogui.keyDown.assert_has_calls([call('ctrl')], any_order=False)
        mock_pyautogui.press.assert_called_once_with('tab')
        mock_pyautogui.keyUp.assert_has_calls([call('ctrl')], any_order=False)
        assert len(ctrl.held_keys) == 0

    def test_windows_controller_press_failure_retains_tab_for_cleanup(self):
        """If pyautogui.press raises, tab is retained in _held_keys for cleanup retry."""
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        mock_pyautogui.press.side_effect = RuntimeError("OS press failed")
        try:
            with pytest.raises(RuntimeError, match="OS press failed"):
                ctrl.switch_tab("NEXT")

            mock_pyautogui.keyUp.assert_called_with('ctrl')
            assert 'tab' in ctrl.held_keys
        finally:
            mock_pyautogui.press.side_effect = None

        ctrl.cleanup()
        assert len(ctrl.held_keys) == 0

    def test_pre_stopped_gate_prevents_tab_dispatch(self):
        gate = SafetyGate()
        gate.trigger_stop(lambda: None)
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        ctrl.switch_tab("NEXT")

        mock_pyautogui.keyDown.assert_not_called()
        mock_pyautogui.keyUp.assert_not_called()
        mock_pyautogui.press.assert_not_called()
        assert len(ctrl.held_keys) == 0

    def test_gated_keydown_refused_after_stop_does_not_keyup_unowned_modifier(self):
        """If gate is stopped, key_down is refused; finally block must NOT keyUp unowned physical modifier."""
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        # Stop the gate before dispatch
        gate.trigger_stop(lambda: None)

        ctrl.switch_tab("PREV")

        # Refused by gate: neither keyDown nor keyUp should have been called!
        mock_pyautogui.keyDown.assert_not_called()
        mock_pyautogui.keyUp.assert_not_called()
        mock_pyautogui.press.assert_not_called()
        assert len(ctrl.held_keys) == 0

    def test_stop_between_modifiers_only_releases_acquired_modifier(self):
        """If stop triggers after ctrl keyDown but before shift keyDown, only ctrl was acquired;
        finally attempts keyUp('ctrl') and does NOT keyUp('shift')."""
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        def mock_key_down(key):
            if key == "ctrl":
                gate.trigger_stop(lambda: None)

        mock_pyautogui.keyDown.side_effect = mock_key_down
        try:
            ctrl.switch_tab("PREV")

            # shift key_down was refused by stopped gate
            # finally must only release acquired 'ctrl', NOT unowned 'shift'!
            mock_pyautogui.keyUp.assert_called_once_with("ctrl")
            assert len(ctrl.held_keys) == 0
        finally:
            mock_pyautogui.keyDown.side_effect = None

    def test_windows_controller_prev_tab_partial_keydown_shift_failure(self):
        """PREV tab partial keyDown shift failure trace downctrl/downshift/upctrl leavesShift:
        attempted_mods includes shift before keyDown so finally attempts shiftUp immediately,
        retaining only failed releases."""
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        def mock_key_down(key):
            if key == "shift":
                raise RuntimeError("Shift keyDown failed")

        mock_pyautogui.keyDown.side_effect = mock_key_down
        try:
            with pytest.raises(RuntimeError, match="Shift keyDown failed"):
                ctrl.switch_tab("PREV")

            # Invariant: both shift and ctrl were attempted released in finally block
            mock_pyautogui.keyUp.assert_has_calls([call("shift"), call("ctrl")], any_order=False)
            assert len(ctrl.held_keys) == 0
        finally:
            mock_pyautogui.keyDown.side_effect = None

    def test_windows_controller_prev_tab_keydown_ctrl_failure(self):
        """If ctrl keyDown fails, ctrl is attempted released in finally, no keys retained."""
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        mock_pyautogui.keyDown.side_effect = RuntimeError("Ctrl keyDown failed")
        try:
            with pytest.raises(RuntimeError, match="Ctrl keyDown failed"):
                ctrl.switch_tab("PREV")

            mock_pyautogui.keyUp.assert_called_once_with("ctrl")
            assert len(ctrl.held_keys) == 0
        finally:
            mock_pyautogui.keyDown.side_effect = None

    def test_windows_controller_prev_tab_keyup_shift_failure_retains_shift(self):
        """If keyUp('shift') in finally fails, 'shift' is retained in held_keys for cleanup retry,
        while 'ctrl' is still released."""
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        def mock_key_up(key):
            if key == "shift":
                raise RuntimeError("Shift keyUp failed")

        mock_pyautogui.keyUp.side_effect = mock_key_up
        try:
            with pytest.raises(RuntimeError, match="Shift keyUp failed"):
                ctrl.switch_tab("PREV")

            mock_pyautogui.keyUp.assert_has_calls([call("shift"), call("ctrl")], any_order=False)
            assert "shift" in ctrl.held_keys
            assert "ctrl" not in ctrl.held_keys
        finally:
            mock_pyautogui.keyUp.side_effect = None

        # Subsequent cleanup retries shift
        ctrl.cleanup()
        assert len(ctrl.held_keys) == 0

    def test_windows_controller_next_tab_keyup_ctrl_failure_retains_ctrl(self):
        """If keyUp('ctrl') fails in finally, 'ctrl' is retained in held_keys."""
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        mock_pyautogui.keyUp.side_effect = RuntimeError("Ctrl keyUp failed")
        try:
            with pytest.raises(RuntimeError, match="Ctrl keyUp failed"):
                ctrl.switch_tab("NEXT")

            assert "ctrl" in ctrl.held_keys
        finally:
            mock_pyautogui.keyUp.side_effect = None

        ctrl.cleanup()
        assert len(ctrl.held_keys) == 0

    def test_reentrant_stop_during_tab_switch(self):
        """If stop is triggered during tab switch execution, subsequent key actions are suppressed."""
        gate = SafetyGate()
        ctrl = WindowsController(allow_real_control=True, gate=gate)

        def mock_press(key):
            gate.trigger_stop(lambda: None)

        mock_pyautogui.press.side_effect = mock_press
        try:
            ctrl.switch_tab("NEXT")

            # Tab was pressed, but cleanup released ctrl
            assert len(ctrl.held_keys) == 0
        finally:
            mock_pyautogui.press.side_effect = None


# ===================================================================
# 6. Hybrid Camera SPACE/HUD and CLI Profile
# ===================================================================
class TestHybridCameraAndCLI:
    @pytest.fixture(autouse=True)
    def mock_model_bundle_loader(self):
        from src.ml.bundle import ModelBundle
        dummy_bundle = ModelBundle(
            feature_version=1,
            labels=['NO_HAND'],
            algorithm='rbf_svm',
            hyperparameters={'C': 1.0, 'gamma': 'scale'},
            train_samples=0,
            created_at='2026-10-02T00:00:00',
            metadata={},
            model_data={'X_train': [], 'y_train': []},
        )
        with patch("tools.train.load_bundle", return_value=dummy_bundle), \
             patch("tools.train.load_model_from_bundle", return_value=(MagicMock(), MagicMock(), {})):
            yield

    def test_cli_profile_argument(self, tmp_path):
        """CLI argument --profile {legacy,hybrid} is parsed properly."""
        tiny_fixture = tmp_path / "tiny.json"
        tiny_fixture.write_text(json.dumps([{"frame_index": 0, "timestamp_ms": 0.0, "landmarks": []}]))
        # Legacy profile runs code 0
        code_legacy = main(["--replay", str(tiny_fixture), "--profile", "legacy"])
        assert code_legacy == 0
        # Hybrid profile runs code 0
        code_hybrid = main(["--replay", str(tiny_fixture), "--profile", "hybrid"])
        assert code_hybrid == 0

    def test_hybrid_warning_banner_contents(self, capsys, tmp_path):
        """--real-control with --profile hybrid must warn about scroll/tabs not move/click,
        and corner stop unavailable without mapped pointer."""
        tiny_fixture = tmp_path / "tiny.json"
        tiny_fixture.write_text(json.dumps([{"frame_index": 0, "timestamp_ms": 0.0, "landmarks": []}]))
        mock_keyboard = MagicMock()
        with patch("src.platform.windows.WindowsController") as mock_ctrl_class, \
             patch.dict(sys.modules, {"keyboard": mock_keyboard}):
            mock_inst = mock_ctrl_class.return_value
            mock_inst.gate = None
            mock_inst.held_buttons = set()
            mock_inst.held_keys = set()
            exit_code = main(["--replay", str(tiny_fixture), "--real-control", "--profile", "hybrid"])
            assert exit_code == 0

        captured = capsys.readouterr()
        assert "scroll and tab switching (not move/click)" in captured.out
        assert "Corner stop is unavailable" in captured.out
        assert "Ctrl+Alt+Q" in captured.out

    def test_legacy_warning_banner_contents(self, capsys, tmp_path):
        """--real-control with --profile legacy prints pointer move/click warning."""
        tiny_fixture = tmp_path / "tiny.json"
        tiny_fixture.write_text(json.dumps([{"frame_index": 0, "timestamp_ms": 0.0, "landmarks": []}]))
        mock_keyboard = MagicMock()
        with patch("src.platform.windows.WindowsController") as mock_ctrl_class, \
             patch.dict(sys.modules, {"keyboard": mock_keyboard}):
            mock_inst = mock_ctrl_class.return_value
            mock_inst.gate = None
            mock_inst.held_buttons = set()
            mock_inst.held_keys = set()
            exit_code = main(["--replay", str(tiny_fixture), "--real-control", "--profile", "legacy"])
            assert exit_code == 0

        captured = capsys.readouterr()
        assert "move your actual mouse cursor and click" in captured.out
        assert "push cursor into screen corner" in captured.out

    def test_camera_space_key_toggles_assist_state(self):
        """Mocked camera loop SPACE key (ord(' ')) toggles assist between OFF and READY."""
        import cv2
        fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8), np.zeros((40, 40, 3), dtype=np.uint8)])
        fake_detector = FakeDetector()

        key_presses = [ord(' '), ord('q')]
        key_idx = 0
        def mock_wait_key(delay):
            nonlocal key_idx
            k = key_presses[key_idx]
            key_idx += 1
            return k

        captured_pipe = []
        real_cp = create_pipeline
        def spy_cp(config, allow_real_control=False, model_bundle=None, profile="legacy"):
            p = real_cp(config, allow_real_control=allow_real_control, model_bundle=model_bundle, profile=profile)
            captured_pipe.append(p)
            return p

        with patch("cv2.VideoCapture", return_value=fake_cap), \
             patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
             patch("app.create_pipeline", side_effect=spy_cp), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", side_effect=mock_wait_key):

            exit_code = run_camera(allow_real_control=False, profile="hybrid")
            assert exit_code == 0

        assert len(captured_pipe) == 1
        pipe = captured_pipe[0]
        # Started at OFF, SPACE was pressed -> toggled to READY
        assert pipe.assist_state == "READY"

    def test_camera_hud_hybrid_profile_overlay(self):
        """Mocked camera loop renders HYBRID profile, assist state, and DRY RUN text."""
        import cv2
        fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
        fake_detector = FakeDetector()

        hud_texts = []
        def spy_put_text(img, text, *args, **kwargs):
            hud_texts.append(text)

        with patch("cv2.VideoCapture", return_value=fake_cap), \
             patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
             patch("cv2.imshow"), \
             patch("cv2.putText", side_effect=spy_put_text), \
             patch("cv2.waitKey", return_value=ord('q')):

            exit_code = run_camera(allow_real_control=False, profile="hybrid")
            assert exit_code == 0

        # Invariant: HUD must display HYBRID profile, assist state, and DRY RUN
        assert any("HYBRID" in t for t in hud_texts)
        assert any("Assist: OFF" in t for t in hud_texts)
        assert any("DRY RUN" in t for t in hud_texts)


# ===================================================================
# 7. Hybrid Defect Regressions & Concurrency
# ===================================================================
class TestHybridDefectRegressions:
    def test_raw_filtered_mismatch_prevents_scroll_and_resets_baseline(self):
        """If raw pose changes away from TWO_FINGERS_PARALLEL, residual filtered
        TWO_FINGERS_PARALLEL must NOT emit scroll, and must reset baseline."""
        ctrl = DryRunController()
        pipe = _make_test_pipeline(
            controller=ctrl,
            scroll_engine=ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid"),
            profile="hybrid",
        )
        pipe.toggle_assist()

        # Warm up 10 frames of TWO_FINGERS_PARALLEL
        lms_two = _make_dummy_landmarks(index_y=0.50, middle_y=0.50)
        for t in range(10):
            with patch.object(pipe, 'classifier', create=True) as mock_cls:
                mock_cls.predict_with_rejection.return_value = ["TWO_FINGERS_PARALLEL"]
                pipe.process_landmarks(lms_two, timestamp_ms=t * 33.3)

        assert pipe._prev_scroll_pos == (0.5, 0.5)

        # Frame 11: raw pose changes to PINCH_CLOSED, index_y moves from 0.50 to 0.30
        lms_pinch = _make_dummy_landmarks(index_y=0.30, middle_y=0.30)
        with patch.object(pipe, 'classifier', create=True) as mock_cls:
            mock_cls.predict_with_rejection.return_value = ["PINCH_CLOSED"]
            # Filter still yields TWO_FINGERS_PARALLEL due to majority window
            res = pipe.process_landmarks(lms_pinch, timestamp_ms=333.3)

        # Invariant: raw/filtered mismatch prevents scroll and clears baseline
        assert res.action_fired is None
        assert len(ctrl.event_log) == 0
        assert pipe._prev_scroll_pos is None

    def test_global_cooldown_preserved_across_pose_exit_and_reentry(self):
        """Global cooldown (400ms) is preserved across exit, other poses, and re-entry."""
        ctrl = DryRunController()
        swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        pipe = _make_test_pipeline(controller=ctrl, swipe_engine=swipe_eng, profile="hybrid")
        pipe.toggle_assist()

        def _step(pose_name, x, t_ms):
            lms = _make_dummy_landmarks(wrist_x=x)
            with patch.object(pipe.majority_filter, 'update', return_value=HandPose[pose_name]), \
                 patch.object(pipe.stability_gate, 'update', return_value=(HandPose[pose_name], True)), \
                 patch.object(pipe, 'classifier', create=True) as mock_cls:
                mock_cls.predict_with_rejection.return_value = [pose_name]
                return pipe.process_landmarks(lms, timestamp_ms=t_ms)

        # 1. THREE at 1000ms (x=0.50), then swipe to 0.61 at 1200ms -> fires NEXT
        _step("THREE_FINGERS", 0.50, 1000.0)
        r1 = _step("THREE_FINGERS", 0.61, 1200.0)
        assert r1.action_fired == "TAB_NEXT"

        # 2. OTHER at 1230ms (exit pose)
        _step("OTHER", 0.50, 1230.0)

        # 3. THREE at 1250ms (x=0.50), then 0.61 at 1450ms (only 250ms after first swipe)
        _step("THREE_FINGERS", 0.50, 1250.0)
        r2 = _step("THREE_FINGERS", 0.61, 1450.0)
        # Invariant: within 400ms cooldown, must NOT fire
        assert r2.action_fired is None
        tab_events = [e for e in ctrl.event_log if e["type"] == KeyEvent.KEY_DOWN and e["key"] == "tab"]
        assert len(tab_events) == 1

        # 4. Now advance past cooldown: THREE at 1650ms, then swipe to 0.62 at 1850ms
        _step("THREE_FINGERS", 0.50, 1650.0)
        r3 = _step("THREE_FINGERS", 0.62, 1850.0)
        assert r3.action_fired == "TAB_NEXT"
        tab_events = [e for e in ctrl.event_log if e["type"] == KeyEvent.KEY_DOWN and e["key"] == "tab"]
        assert len(tab_events) == 2

    def test_reentrant_toggle_off_during_rate_limiter_allow(self):
        """If toggle_assist turns assist OFF during rate_limiter.allow callback,
        the in-flight frame must NOT emit scroll."""
        ctrl = DryRunController()
        scroll_eng = ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid")
        pipe = _make_test_pipeline(controller=ctrl, scroll_engine=scroll_eng, profile="hybrid")
        pipe.toggle_assist()

        # Frame 1: baseline at cy=0.50
        lms1 = _make_dummy_landmarks(index_y=0.50, middle_y=0.50)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)), \
             patch.object(pipe, 'classifier', create=True) as mock_cls:
            mock_cls.predict_with_rejection.return_value = ["TWO_FINGERS_PARALLEL"]
            pipe.process_landmarks(lms1, timestamp_ms=0.0)

        # Mock RateLimiter whose allow() toggles assist to OFF then returns True
        mock_limiter = MagicMock()
        def _allow_with_reentrant_off():
            pipe.toggle_assist()  # Toggles READY -> OFF
            return True
        mock_limiter.allow.side_effect = _allow_with_reentrant_off
        pipe.rate_limiter = mock_limiter

        # Frame 2: hand moves UP (index_y=0.40)
        lms2 = _make_dummy_landmarks(index_y=0.40, middle_y=0.40)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)), \
             patch.object(pipe, 'classifier', create=True) as mock_cls:
            mock_cls.predict_with_rejection.return_value = ["TWO_FINGERS_PARALLEL"]
            res = pipe.process_landmarks(lms2, timestamp_ms=33.3)

        assert pipe.assist_state == "OFF"
        assert res.action_fired is None
        assert len(ctrl.event_log) == 0

    def test_thread_handshake_toggle_assist_serialized_with_gate(self):
        """Cross-thread toggle_assist serializes cleanly with active frame transaction."""
        import threading
        gate = SafetyGate()
        ctrl = DryRunController()
        scroll_eng = ScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid")
        pipe = _make_test_pipeline(controller=ctrl, scroll_engine=scroll_eng, gate=gate, profile="hybrid")
        pipe.toggle_assist()

        in_allow = threading.Event()
        can_proceed = threading.Event()
        toggle_started = threading.Event()
        toggle_done = threading.Event()

        mock_limiter = MagicMock()
        def _blocking_allow():
            in_allow.set()
            can_proceed.wait(timeout=2.0)
            return True
        mock_limiter.allow.side_effect = _blocking_allow
        pipe.rate_limiter = mock_limiter

        # Warm baseline
        lms1 = _make_dummy_landmarks(index_y=0.50, middle_y=0.50)
        with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
             patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)), \
             patch.object(pipe, 'classifier', create=True) as mock_cls:
            mock_cls.predict_with_rejection.return_value = ["TWO_FINGERS_PARALLEL"]
            pipe.process_landmarks(lms1, timestamp_ms=0.0)

        # Thread 1: processes frame that enters gate transaction
        lms2 = _make_dummy_landmarks(index_y=0.40, middle_y=0.40)
        def _frame_worker():
            with patch.object(pipe.majority_filter, 'update', return_value=HandPose.TWO_FINGERS_PARALLEL), \
                 patch.object(pipe.stability_gate, 'update', return_value=(HandPose.TWO_FINGERS_PARALLEL, True)), \
                 patch.object(pipe, 'classifier', create=True) as mock_cls:
                mock_cls.predict_with_rejection.return_value = ["TWO_FINGERS_PARALLEL"]
                pipe.process_landmarks(lms2, timestamp_ms=33.3)

        t1 = threading.Thread(target=_frame_worker)
        t1.start()
        assert in_allow.wait(timeout=2.0)

        # Thread 2: attempts toggle_assist while Thread 1 holds gate
        def _toggle_worker():
            toggle_started.set()
            pipe.toggle_assist()
            toggle_done.set()

        t2 = threading.Thread(target=_toggle_worker)
        t2.start()
        assert toggle_started.wait(timeout=2.0)

        # Verify t2 is waiting for gate lock
        assert not toggle_done.is_set()

        # Let frame thread finish
        can_proceed.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        assert not t1.is_alive()
        assert not t2.is_alive()
        assert toggle_done.is_set()

    def test_nohand_vs_toggle_off_handshake_no_auto_reactivation(self):
        """P1 repro 2b10b5 regression:
        Frame thread receives nohand frame while assist is READY.
        Main thread calls toggle_assist() to transition to OFF.
        Regardless of thread interleaving, assist state must remain OFF,
        and never silently re-enable to READY.
        """
        import threading
        config = {"safety": {"allow_real_control": False}}
        gate = SafetyGate()
        pipe = AirPadPipeline(
            config=config,
            controller=DryRunController(),
            profile="hybrid",
            gate=gate,
        )
        pipe.toggle_assist()
        assert pipe.assist_state == "READY"

        thread_finished = threading.Event()

        def run_nohand_frame():
            pipe.process_landmarks(None, timestamp_ms=100.0)
            thread_finished.set()

        # Thread 1: toggle assist OFF
        pipe.toggle_assist()
        assert pipe.assist_state == "OFF"

        # Now run nohand frame
        t = threading.Thread(target=run_nohand_frame)
        t.start()
        t.join(timeout=2.0)
        assert thread_finished.is_set()

        # Invariant: assist_state MUST remain OFF, no auto-reactivation!
        assert pipe.assist_state == "OFF"

    def test_nohand_thread_interleaved_with_toggle_off_serialized(self):
        """Test concurrent nohand processing and toggle_assist(OFF).
        Gate lock guarantees serial execution: either nohand completes first (READY)
        then toggle sets OFF, OR toggle sets OFF first and nohand preserves OFF.
        In no scenario can the final state be READY.
        """
        import threading
        config = {"safety": {"allow_real_control": False}}
        gate = SafetyGate()
        pipe = AirPadPipeline(
            config=config,
            controller=DryRunController(),
            profile="hybrid",
            gate=gate,
        )
        pipe.toggle_assist()
        assert pipe.assist_state == "READY"

        barrier = threading.Barrier(2)
        errors = []

        def worker_nohand():
            try:
                barrier.wait(timeout=2.0)
                pipe.process_landmarks(None, timestamp_ms=50.0)
            except Exception as e:
                errors.append(e)

        def worker_toggle():
            try:
                barrier.wait(timeout=2.0)
                pipe.toggle_assist()  # READY -> OFF
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker_nohand)
        t2 = threading.Thread(target=worker_toggle)
        t1.start()
        t2.start()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        assert len(errors) == 0
        # If toggle ran, it was either before or after nohand.
        # If toggle ran first: assist is OFF, nohand sees OFF and keeps it OFF.
        # If nohand ran first: assist was set to READY by nohand, then toggle set it to OFF.
        # Either way, final state MUST be OFF!
        assert pipe.assist_state == "OFF"

    def test_no_auto_reactivation_from_off_across_all_frame_types(self):
        """When assist is OFF, no hand, mismatch, or pose frames can ever set assist_state to READY."""
        config = {"safety": {"allow_real_control": False}}
        gate = SafetyGate()
        pipe = AirPadPipeline(
            config=config,
            controller=DryRunController(),
            profile="hybrid",
            gate=gate,
        )
        assert pipe.assist_state == "OFF"

        # 1. No hand frame
        res = pipe.process_landmarks(None, timestamp_ms=10.0)
        assert pipe.assist_state == "OFF"
        assert res.telemetry.get("assist_state") == "OFF"

        # 2. Hand frame (all other poses, e.g. POINT_ONE)
        lms = np.zeros((21, 3), dtype=np.float32)
        res = pipe.process_landmarks(lms, timestamp_ms=20.0)
        assert pipe.assist_state == "OFF"
        assert res.telemetry.get("assist_state") == "OFF"
        assert res.action_fired is None

    def test_swipe_engine_stationary_hold_then_deliberate_swipe_both_directions(self):
        """Hold 2s / 10s then deliberate .12/150-190ms swipe in both directions must fire exactly once."""
        # 1. Hold 2s (every 40ms) then deliberate right swipe (.12 in 190ms) -> NEXT
        eng_right = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        events_right = []
        for t in range(1000, 3040, 40):
            ev = eng_right.update(0.50, float(t))
            if ev:
                events_right.append((t, ev))
        ev = eng_right.update(0.62, 3190.0)
        if ev:
            events_right.append((3190, ev))
        assert events_right == [(3190, "NEXT")], f"Expected exactly one NEXT, got {events_right}"

        # 2. Hold 10s (every 40ms) then deliberate left swipe (.12 in 150ms) -> PREV
        eng_left = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        events_left = []
        for t in range(0, 10040, 40):
            ev = eng_left.update(0.50, float(t))
            if ev:
                events_left.append((t, ev))
        ev = eng_left.update(0.38, 10190.0)
        if ev:
            events_left.append((10190, ev))
        assert events_left == [(10190, "PREV")], f"Expected exactly one PREV, got {events_left}"

    def test_swipe_engine_stationary_jitter_and_slow_drift_suppression(self):
        """Stationary jitter and slow drift must NOT fire swipe events."""
        # 1. 10s of stationary jitter (+-0.0015 every 33ms)
        eng_jitter = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        jitter_events = []
        for t in range(0, 10000, 33):
            jitter = 0.0015 if (t // 33) % 2 == 0 else -0.0015
            ev = eng_jitter.update(0.50 + jitter, float(t))
            if ev:
                jitter_events.append(ev)
        assert len(jitter_events) == 0, f"Stationary jitter emitted events: {jitter_events}"

        # 2. 10s of slow drift (0.50 -> 0.70 at velocity 0.00002 << 0.0003)
        eng_drift = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        drift_events = []
        for t in range(0, 10000, 40):
            val = 0.50 + 0.20 * (t / 10000.0)
            ev = eng_drift.update(val, float(t))
            if ev:
                drift_events.append(ev)
        assert len(drift_events) == 0, f"Slow drift emitted events: {drift_events}"

    def test_swipe_engine_neutral_return_after_fire_never_rearms(self):
        """Neutral return and overshoot while still holding THREE_FINGERS must never rearm."""
        eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        eng.update(0.50, 0.0)
        ev1 = eng.update(0.62, 150.0)
        assert ev1 == "NEXT"

        # Neutral return at 600ms (crossing 400ms cooldown) while still THREE
        assert eng.update(0.50, 600.0, is_three_fingers=True) is None
        assert not eng.is_armed

        # Overshoot to left at 750ms while still THREE
        assert eng.update(0.38, 750.0, is_three_fingers=True) is None
        assert not eng.is_armed

    def test_swipe_engine_persistent_cooldown_and_pose_exit_contract(self):
        """Rearm requires both observed pose exit AND cooldown passage."""
        eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        eng.update(0.50, 0.0)
        ev1 = eng.update(0.38, 150.0)
        assert ev1 == "PREV"

        # Exit pose at 200ms
        eng.update(0.38, 200.0, is_three_fingers=False)

        # Re-enter at 300ms (dt = 300 - 150 = 150ms < 400ms cooldown) -> not armed yet
        assert eng.update(0.50, 300.0, is_three_fingers=True) is None
        assert not eng.is_armed

        # Attempt swipe at 350ms -> nothing
        assert eng.update(0.62, 350.0, is_three_fingers=True) is None

        # Exit again at 400ms
        eng.update(0.62, 400.0, is_three_fingers=False)

        # Re-enter after cooldown passes (t = 600ms, dt = 600 - 150 = 450ms >= 400ms)
        assert eng.update(0.50, 600.0, is_three_fingers=True) is None
        assert eng.is_armed

        # Swipe right at 750ms -> NEXT
        ev2 = eng.update(0.62, 750.0, is_three_fingers=True)
        assert ev2 == "NEXT"

    def test_hybrid_pipeline_default_filters_long_warm_three_then_swipe(self):
        """Pipeline with default temporal filters: warm THREE_FINGERS for 2s, then deliberate swipe."""
        clf = MockLabelClassifier("THREE_FINGERS")
        ctrl = DryRunController()
        swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
        pipe = _make_test_pipeline(
            controller=ctrl,
            swipe_engine=swipe_eng,
            profile="hybrid",
            classifier=clf,
        )
        pipe.toggle_assist()
        assert pipe.assist_state == "READY"

        actions = []
        # Hold warm THREE_FINGERS for 2000ms (50 frames at 40ms interval)
        for i in range(50):
            t_ms = i * 40.0
            lms = _make_dummy_landmarks(wrist_x=0.50)
            res = pipe.process_landmarks(lms, timestamp_ms=t_ms)
            if res.action_fired:
                actions.append((t_ms, res.action_fired))

        # Deliberate swipe to 0.62 in 160ms (4 frames at 40ms)
        for i in range(1, 5):
            t_ms = 2000.0 + i * 40.0
            wx = 0.50 + 0.12 * (i / 4.0)
            lms = _make_dummy_landmarks(wrist_x=wx)
            res = pipe.process_landmarks(lms, timestamp_ms=t_ms)
            if res.action_fired:
                actions.append((t_ms, res.action_fired))

        assert len(actions) == 1, f"Expected exactly one action, got {actions}"
        assert actions[0][1] == "TAB_NEXT"


    def test_legacy_swipe_engine_exact_original_comparison_regression(self):
        """Legacy SwipeEngine must produce identical results to saved-original engine.

        Regression test for the conditional-return bug where the hybrid
        fall-through after rearm was accidentally applied to legacy mode,
        allowing an immediate BACK fire on the rearm frame.

        Saved-original sequence: x=.5@1000, .61@1200, .5@1700, .39@1730
        Expected: None, FORWARD, None, None (rearm frame returns None, and
        the next frame starts fresh trajectory from x=.5, not enough dt for
        the x=.39 displacement to trigger).

        This test loads the saved-original engine from before/ and asserts
        byte-for-byte behavioral equivalence with current legacy mode.
        """
        import importlib.util

        # Load saved-original engine
        spec = importlib.util.spec_from_file_location(
            "orig_swipe",
            os.path.join("reports", "hyb01", "before", "src", "motion", "swipe.py"),
        )
        orig_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orig_mod)

        # Identical test sequences
        sequences = [
            # Original regression sequence from user repro
            [
                (0.50, 1000.0, True),
                (0.61, 1200.0, True),
                (0.50, 1700.0, True),
                (0.39, 1730.0, True),
            ],
            # Extended: after rearm, next trajectory should start fresh
            [
                (0.50, 1000.0, True),
                (0.61, 1200.0, True),   # FORWARD
                (0.50, 1700.0, True),   # rearm (within threshold, cooldown passed)
                (0.50, 1800.0, True),   # fresh start_x
                (0.39, 1830.0, True),   # too fast? check displacement .11, dt=30, vel=.00367 -> BACK? No, start_x is .50 from 1800
            ],
            # Pose exit and re-enter
            [
                (0.50, 1000.0, True),
                (0.61, 1200.0, True),   # FORWARD
                (0.50, 1400.0, False),  # exit -> rearm
                (0.50, 1500.0, True),   # re-enter, fresh start
                (0.39, 1700.0, True),   # BACK (new trajectory, .11 in 200ms)
            ],
        ]

        for seq_idx, sequence in enumerate(sequences):
            orig_eng = orig_mod.SwipeEngine()
            curr_eng = SwipeEngine()  # default = legacy

            for step_idx, (x, t, is_three) in enumerate(sequence):
                orig_result = orig_eng.update(x, t, is_three_fingers=is_three)
                curr_result = curr_eng.update(x, t, is_three_fingers=is_three)
                assert orig_result == curr_result, (
                    f"Seq {seq_idx} step {step_idx}: x={x} t={t} "
                    f"three={is_three}: original={orig_result}, "
                    f"current_legacy={curr_result}"
                )

    def test_legacy_rearm_frame_unconditional_return_none(self):
        """Legacy mode: the frame that triggers rearm must unconditionally return None.

        This matches saved-original behavior where the entire fired/rearm
        block returns None regardless of rearm success.
        """
        eng = SwipeEngine()  # legacy

        # Fire FORWARD
        eng.update(0.50, 1000.0)
        r_fire = eng.update(0.61, 1200.0)
        assert r_fire == "FORWARD"
        assert eng._fired is True

        # Frame at 1700ms: cooldown passed (500ms >= 400ms), x=0.50 within
        # rearm_threshold of start_x=0.50. Rearm happens, but frame returns None.
        r_rearm = eng.update(0.50, 1700.0)
        assert r_rearm is None, f"Rearm frame must return None, got {r_rearm}"
        # After rearm, engine is armed and start_x is cleared
        assert eng.is_armed
        assert eng._start_x is None
        assert eng._fired is False

        # Next frame: rapid reverse displacement should NOT fire on the rearm frame
        # but CAN set up new trajectory starting from this frame
        r_next = eng.update(0.39, 1730.0)
        # This frame establishes _start_x = 0.39; no event yet (just started)
        assert r_next is None, f"Frame after rearm sets start_x, got {r_next}"
