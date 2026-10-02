"""Tests for src.gesture.fsm – GestureFSM state machine.

Covers the full gesture contract:
- Startup PAUSED
- POINT_ONE stable → MOVING
- Quick pinch → LEFT_CLICK exactly once
- Pinch hold → DRAG (mouseDown once, mouseUp on release)
- Hand loss cleanup
- Emergency stop
- OPEN_PALM/NO_HAND → PAUSED + cleanup
- Debounce prevents double-click
- Scroll entry/exit with hysteresis
- Zoom dwell → latch
"""
from __future__ import annotations

import numpy as np
import pytest

from src.gesture.fsm import GestureFSM, FSMState
from src.gesture.pose import HandPose
from src.platform.dry_run import DryRunController
from src.platform.interface import MouseEvent, KeyEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_fsm(config_overrides: dict | None = None) -> tuple[GestureFSM, DryRunController]:
    """Create an FSM with a DryRunController and sensible test config."""
    config = {
        'drag_threshold_ms': 400.0,
        'click_min_ms': 80.0,
        'click_max_ms': 300.0,
        'dwell_zoom_ms': 250.0,
        'hysteresis_exit_count': 3,
        'debounce_ms': 50.0,
        'point_one_stable_ms': 100.0,
    }
    if config_overrides:
        config.update(config_overrides)
    ctrl = DryRunController()
    fsm = GestureFSM(ctrl, config)
    return fsm, ctrl


def transition_to_moving(fsm: GestureFSM, start_ms: float = 0.0) -> float:
    """Helper: transition FSM from PAUSED to MOVING."""
    t = start_ms
    fsm.update(HandPose.POINT_ONE, t)
    t += 50
    fsm.update(HandPose.POINT_ONE, t)
    t += 60  # total 110ms >= point_one_stable_ms
    fsm.update(HandPose.POINT_ONE, t)
    assert fsm.state == FSMState.MOVING, f"Expected MOVING, got {fsm.state}"
    return t


def get_event_types(ctrl: DryRunController) -> list:
    """Extract just the event type from logged events."""
    return [e['type'] for e in ctrl.event_log]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestStartupPaused:
    def test_initial_state_is_paused(self):
        fsm, _ = make_fsm()
        assert fsm.state == FSMState.PAUSED

    def test_stays_paused_on_open_palm(self):
        fsm, _ = make_fsm()
        fsm.update(HandPose.OPEN_PALM, 0.0)
        assert fsm.state == FSMState.PAUSED

    def test_stays_paused_on_no_hand(self):
        fsm, _ = make_fsm()
        fsm.update(HandPose.NO_HAND, 0.0)
        assert fsm.state == FSMState.PAUSED


class TestPointOneToMoving:
    def test_transitions_after_stable_duration(self):
        fsm, _ = make_fsm()
        fsm.update(HandPose.POINT_ONE, 0.0)
        assert fsm.state == FSMState.PAUSED
        fsm.update(HandPose.POINT_ONE, 50.0)
        assert fsm.state == FSMState.PAUSED
        fsm.update(HandPose.POINT_ONE, 110.0)
        assert fsm.state == FSMState.MOVING

    def test_resets_timer_on_interruption(self):
        fsm, _ = make_fsm()
        fsm.update(HandPose.POINT_ONE, 0.0)
        fsm.update(HandPose.FIST, 50.0)  # interrupts
        fsm.update(HandPose.POINT_ONE, 100.0)  # restart
        fsm.update(HandPose.POINT_ONE, 150.0)
        assert fsm.state == FSMState.PAUSED  # only 50ms since restart
        fsm.update(HandPose.POINT_ONE, 210.0)  # now 110ms
        assert fsm.state == FSMState.MOVING


class TestLeftClick:
    def test_click_fires_exactly_once(self):
        """Quick pinch (80-300ms) → LEFT_CLICK exactly once."""
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        # Enter pinch
        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        assert fsm.state == FSMState.CLICK_PENDING

        # Release after 150ms (within click range)
        t += 150
        fsm.update(HandPose.POINT_ONE, t)
        assert fsm.state == FSMState.MOVING

        clicks = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_CLICK]
        assert len(clicks) == 1, f"Expected 1 click, got {len(clicks)}"

    def test_no_click_if_too_fast(self):
        """Pinch < 80ms → accidental, no click."""
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 30  # only 30ms < click_min_ms
        fsm.update(HandPose.POINT_ONE, t)
        assert fsm.state == FSMState.MOVING

        clicks = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_CLICK]
        assert len(clicks) == 0

    def test_no_click_if_too_slow(self):
        """Pinch > click_max_ms but < drag_threshold → rejected."""
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 350  # 350ms > click_max_ms(300) but < drag_threshold(400)
        fsm.update(HandPose.POINT_ONE, t)

        clicks = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_CLICK]
        assert len(clicks) == 0


class TestDrag:
    def test_drag_mousedown_once_mouseup_on_release(self):
        """Pinch held > drag_threshold → mouseDown once, mouseUp on release."""
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        # Enter pinch
        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        assert fsm.state == FSMState.CLICK_PENDING

        # Hold past drag threshold
        t += 200
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 200  # total 400ms = drag_threshold
        fsm.update(HandPose.PINCH_CLOSED, t)
        assert fsm.state == FSMState.DRAGGING

        downs = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_DOWN]
        assert len(downs) == 1, f"Expected 1 mouseDown, got {len(downs)}"

        # Continue holding → no additional mouseDown
        t += 100
        fsm.update(HandPose.PINCH_CLOSED, t)
        downs = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_DOWN]
        assert len(downs) == 1

        # Release
        t += 10
        fsm.update(HandPose.POINT_ONE, t)
        assert fsm.state == FSMState.MOVING

        ups = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_UP]
        assert len(ups) == 1

    def test_drag_mouseup_on_hand_loss(self):
        """If hand is lost during drag, mouseUp must fire."""
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        # Enter drag
        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 500
        fsm.update(HandPose.PINCH_CLOSED, t)
        assert fsm.state == FSMState.DRAGGING

        # Simulate hand loss → cleanup
        fsm.on_hand_loss()
        assert fsm.state == FSMState.PAUSED

        ups = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_UP]
        assert len(ups) >= 1, "mouseUp must fire on hand loss during drag"


class TestHandLossCleanup:
    def test_cleanup_on_hand_loss(self):
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        # Manually hold a button
        ctrl.mouse_down('left')
        ctrl.key_down('shift')

        fsm.on_hand_loss()
        assert fsm.state == FSMState.PAUSED
        assert len(ctrl.held_buttons) == 0
        assert len(ctrl.held_keys) == 0

    def test_cleanup_on_error(self):
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        ctrl.mouse_down('left')
        fsm.on_error()
        assert fsm.state == FSMState.PAUSED
        assert len(ctrl.held_buttons) == 0

    def test_cleanup_on_emergency_stop(self):
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        ctrl.mouse_down('left')
        fsm.on_emergency_stop()
        assert fsm.state == FSMState.PAUSED
        assert len(ctrl.held_buttons) == 0

    def test_open_palm_resets_to_paused(self):
        """OPEN_PALM from any state → PAUSED + cleanup."""
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        fsm.update(HandPose.OPEN_PALM, t + 10)
        assert fsm.state == FSMState.PAUSED

    def test_no_hand_resets_to_paused(self):
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        fsm.update(HandPose.NO_HAND, t + 10)
        assert fsm.state == FSMState.PAUSED


class TestDebounce:
    def test_debounce_prevents_double_click(self):
        """Two rapid clicks within debounce_ms → only one fires."""
        fsm, ctrl = make_fsm({'debounce_ms': 200.0})
        t = transition_to_moving(fsm)

        # First click
        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 150  # 150ms — valid click
        fsm.update(HandPose.POINT_ONE, t)

        # Second click immediately
        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 150
        fsm.update(HandPose.POINT_ONE, t)

        clicks = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_CLICK]
        assert len(clicks) == 1, f"Debounce should prevent double click, got {len(clicks)}"


class TestScrolling:
    def test_enter_scroll_from_moving(self):
        fsm, _ = make_fsm()
        t = transition_to_moving(fsm)

        fsm.update(HandPose.TWO_FINGERS_PARALLEL, t + 10)
        assert fsm.state == FSMState.SCROLLING

    def test_exit_scroll_with_hysteresis(self):
        """Need hysteresis_exit_count consecutive non-scroll frames to exit."""
        fsm, _ = make_fsm({'hysteresis_exit_count': 3})
        t = transition_to_moving(fsm)

        fsm.update(HandPose.TWO_FINGERS_PARALLEL, t + 10)
        assert fsm.state == FSMState.SCROLLING

        t += 20
        # 1 frame of POINT_ONE - not enough
        fsm.update(HandPose.POINT_ONE, t)
        assert fsm.state == FSMState.SCROLLING

        t += 10
        fsm.update(HandPose.POINT_ONE, t)
        assert fsm.state == FSMState.SCROLLING

        t += 10
        fsm.update(HandPose.POINT_ONE, t)  # 3rd consecutive
        assert fsm.state == FSMState.MOVING


class TestZoom:
    def test_zoom_dwell_to_latch(self):
        fsm, _ = make_fsm({'dwell_zoom_ms': 250.0})
        t = transition_to_moving(fsm)

        t += 10
        fsm.update(HandPose.L_SHAPE, t)
        assert fsm.state == FSMState.ZOOM_DWELL

        t += 300  # > 250ms dwell
        fsm.update(HandPose.L_SHAPE, t)
        assert fsm.state == FSMState.ZOOM_LATCHED

    def test_zoom_abort_before_dwell(self):
        fsm, _ = make_fsm({'dwell_zoom_ms': 250.0})
        t = transition_to_moving(fsm)

        t += 10
        fsm.update(HandPose.L_SHAPE, t)
        assert fsm.state == FSMState.ZOOM_DWELL

        t += 100  # only 100ms, < 250ms dwell
        fsm.update(HandPose.POINT_ONE, t)
        assert fsm.state == FSMState.MOVING


class TestEmergencyHotkeyViaFSM:
    def test_emergency_stop_event(self):
        """Emergency stop can be triggered and tested via simulated event."""
        from src.safety.failsafe import EmergencyHotkey

        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        # Simulate held button
        ctrl.mouse_down('left')

        hotkey = EmergencyHotkey("ctrl+alt+q", fsm.on_emergency_stop, dry_run=True)
        hotkey.trigger_emergency()

        assert fsm.state == FSMState.PAUSED
        assert len(ctrl.held_buttons) == 0


class TestReplayScenario:
    def test_full_click_drag_handloss_replay(self):
        """Full scenario replay: move → click → move → drag → hand loss."""
        fsm, ctrl = make_fsm()

        # 1. Transition to MOVING
        t = transition_to_moving(fsm)

        # 2. Quick click
        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 150
        fsm.update(HandPose.POINT_ONE, t)
        assert fsm.state == FSMState.MOVING

        # 3. Start drag
        t += 100
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 500
        fsm.update(HandPose.PINCH_CLOSED, t)
        assert fsm.state == FSMState.DRAGGING

        # 4. Hand loss during drag
        fsm.on_hand_loss()
        assert fsm.state == FSMState.PAUSED

        # Verify: 1 click, 1 mouseDown, 1+ mouseUp
        clicks = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_CLICK]
        downs = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_DOWN]
        ups = [e for e in ctrl.event_log if e['type'] == MouseEvent.LEFT_UP]

        assert len(clicks) == 1
        assert len(downs) == 1
        assert len(ups) >= 1  # cleanup mouseUp
