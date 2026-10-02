"""Deterministic tests for 2A-SAFETY requirements:

1) PAUSED reset: POINT_ONE@0 -> OPEN_PALM@50 -> POINT_ONE@1000 must remain PAUSED until 100ms
   stability is reached. OPEN_PALM/NO_HAND always resets pending/cleanup idempotently, even in PAUSED.
2) Gated gestures: After UNKNOWN/OTHER, all click/right-click/drag/scroll/zoom are locked until
   reacquire_stable_frames consecutive POINT_ONE frames. FIST resets the sequence.
   UNKNOWN in SCROLLING/ZOOM_LATCHED stops output immediately.
3) Clean release: CLICK_PENDING -> PINCH_MIDDLE and RIGHT_CLICK_PENDING -> PINCH_CLOSED do NOT fire click.
   Only clean release to POINT_ONE fires click. OPEN_PALM/NO_HAND cancels and pauses.
4) mouse_up failure: Fake controller raising on mouse_up is NOT swallowed; button remains marked down;
   FSM does not transition to MOVING/released; cursor control is blocked until release confirmed.
5) No pose=None bypass: can_move_cursor(pose) strictly requires explicit HandPose.
6) Config validation: reacquire_stable_frames >= 1 and click_min_ms <= click_max_ms < drag_threshold_ms.
"""
from __future__ import annotations

import pytest

from src.gesture.fsm import GestureFSM, FSMState
from src.gesture.pose import HandPose
from src.platform.dry_run import DryRunController
from src.platform.interface import MouseEvent


class FailingMouseUpController(DryRunController):
    """Fake controller that fails on mouse_up to test safety propagation."""
    def mouse_up(self, button: str = 'left') -> None:
        raise RuntimeError("OS mouse_up failed")


def make_fsm(config_overrides: dict | None = None, controller: DryRunController | None = None) -> tuple[GestureFSM, DryRunController]:
    config = {
        'drag_threshold_ms': 400.0,
        'click_min_ms': 80.0,
        'click_max_ms': 300.0,
        'dwell_zoom_ms': 250.0,
        'hysteresis_exit_count': 3,
        'debounce_ms': 50.0,
        'point_one_stable_ms': 100.0,
        'enable_right_click': True,
        'reacquire_stable_frames': 2,
        'unknown_timeout_frames': 4,
    }
    if config_overrides:
        config.update(config_overrides)
    ctrl = controller if controller is not None else DryRunController()
    fsm = GestureFSM(ctrl, config)
    return fsm, ctrl


def transition_to_moving(fsm: GestureFSM, start_ms: float = 0.0) -> float:
    t = start_ms
    fsm.update(HandPose.POINT_ONE, t)
    t += 50
    fsm.update(HandPose.POINT_ONE, t)
    t += 60  # total 110ms >= point_one_stable_ms
    fsm.update(HandPose.POINT_ONE, t)
    assert fsm.state == FSMState.MOVING
    return t


# ===================================================================
# 1. PAUSED Reset & Idempotent Cleanup Tests
# ===================================================================
class TestPausedResetSafety:
    def test_open_palm_in_paused_resets_pending_timer(self):
        """POINT_ONE@0 -> OPEN_PALM@50 -> POINT_ONE@1000 must remain PAUSED until +100ms stable."""
        fsm, _ = make_fsm({'point_one_stable_ms': 100.0})

        fsm.update(HandPose.POINT_ONE, 0.0)
        assert fsm.state == FSMState.PAUSED

        # OPEN_PALM at 50ms while already in PAUSED: MUST reset pending point_one timer!
        fsm.update(HandPose.OPEN_PALM, 50.0)
        assert fsm.state == FSMState.PAUSED
        assert fsm._point_one_start_ms is None

        # POINT_ONE resumes at t=1000ms: must NOT immediately jump to MOVING!
        fsm.update(HandPose.POINT_ONE, 1000.0)
        assert fsm.state == FSMState.PAUSED

        # 50ms later (1050ms): still PAUSED (only 50ms stable)
        fsm.update(HandPose.POINT_ONE, 1050.0)
        assert fsm.state == FSMState.PAUSED

        # Full 100ms reached at 1100ms: now enters MOVING
        fsm.update(HandPose.POINT_ONE, 1105.0)
        assert fsm.state == FSMState.MOVING

    def test_no_hand_in_paused_idempotent_cleanup(self):
        fsm, ctrl = make_fsm()
        fsm.update(HandPose.NO_HAND, 10.0)
        assert fsm.state == FSMState.PAUSED
        assert len(ctrl.held_buttons) == 0


# ===================================================================
# 2. Gesture Locking & Reacquisition Gating Tests
# ===================================================================
class TestReacquisitionGating:
    def test_unknown_locks_click_until_reacquired(self):
        """MOVING -> UNKNOWN -> PINCH_CLOSED -> POINT_ONE(150ms) must NOT fire LEFT_CLICK."""
        fsm, ctrl = make_fsm({'reacquire_stable_frames': 2})
        t = transition_to_moving(fsm)

        # UNKNOWN degrades tracking
        t += 20
        fsm.update(HandPose.UNKNOWN, t)
        assert fsm.state == FSMState.MOVING
        assert not fsm.is_reacquired

        # User attempts PINCH_CLOSED immediately while unacquired
        t += 20
        fsm.update(HandPose.PINCH_CLOSED, t)
        # Must NOT enter CLICK_PENDING!
        assert fsm.state != FSMState.CLICK_PENDING

        # Release to POINT_ONE after 150ms
        t += 150
        fsm.update(HandPose.POINT_ONE, t)
        assert len(ctrl.event_log) == 0, "Accidental click must not fire after UNKNOWN without reacquisition"

    def test_fist_resets_reacquisition_counter(self):
        fsm, _ = make_fsm({'reacquire_stable_frames': 2})
        t = transition_to_moving(fsm)

        # Drop tracking
        fsm.update(HandPose.UNKNOWN, t + 10)
        assert not fsm.is_reacquired

        # 1st POINT_ONE frame
        fsm.update(HandPose.POINT_ONE, t + 20)
        assert fsm._reacquire_count == 1

        # FIST frame interrupts and resets the sequence
        fsm.update(HandPose.FIST, t + 30)
        assert fsm._reacquire_count == 0
        assert not fsm.is_reacquired

        # Next POINT_ONE is only frame 1 again
        fsm.update(HandPose.POINT_ONE, t + 40)
        assert fsm._reacquire_count == 1
        assert not fsm.is_reacquired

        # 2nd consecutive POINT_ONE -> reacquired
        fsm.update(HandPose.POINT_ONE, t + 50)
        assert fsm.is_reacquired

    def test_unknown_in_scrolling_and_zoom_halts_output(self):
        fsm, _ = make_fsm()
        t = transition_to_moving(fsm)

        # Enter scrolling
        fsm.update(HandPose.TWO_FINGERS_PARALLEL, t + 10)
        assert fsm.state == FSMState.SCROLLING
        assert fsm.is_scrolling_active(HandPose.TWO_FINGERS_PARALLEL) is True

        # UNKNOWN in SCROLLING must immediately halt output
        fsm.update(HandPose.UNKNOWN, t + 20)
        assert fsm.is_scrolling_active(HandPose.UNKNOWN) is False

        # Transition to ZOOM_LATCHED
        fsm._force_cleanup()
        t = transition_to_moving(fsm, start_ms=200.0)
        fsm.update(HandPose.L_SHAPE, t + 10)
        fsm.update(HandPose.L_SHAPE, t + 300)
        assert fsm.state == FSMState.ZOOM_LATCHED
        assert fsm.is_zoom_active(HandPose.L_SHAPE) is True

        # UNKNOWN in ZOOM_LATCHED must immediately halt output
        fsm.update(HandPose.UNKNOWN, t + 320)
        assert fsm.is_zoom_active(HandPose.UNKNOWN) is False


# ===================================================================
# 3. Clean Release Requirements Tests
# ===================================================================
class TestCleanReleaseRequirements:
    def test_click_pending_to_pinch_middle_cancels_click(self):
        """CLICK_PENDING -> PINCH_MIDDLE must NOT fire click."""
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        assert fsm.state == FSMState.CLICK_PENDING

        # Finger slips into PINCH_MIDDLE at 150ms
        t += 150
        fsm.update(HandPose.PINCH_MIDDLE, t)
        assert fsm.state == FSMState.MOVING

        clicks = [e for e in ctrl.event_log if e['type'] in (MouseEvent.LEFT_CLICK, MouseEvent.RIGHT_CLICK)]
        assert len(clicks) == 0, "Transition from CLICK_PENDING to PINCH_MIDDLE must not click"

    def test_right_click_pending_to_pinch_closed_cancels_click(self):
        """RIGHT_CLICK_PENDING -> PINCH_CLOSED must NOT fire click."""
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        t += 10
        fsm.update(HandPose.PINCH_MIDDLE, t)
        assert fsm.state == FSMState.RIGHT_CLICK_PENDING

        # Finger slips into PINCH_CLOSED at 150ms
        t += 150
        fsm.update(HandPose.PINCH_CLOSED, t)
        assert fsm.state == FSMState.MOVING

        clicks = [e for e in ctrl.event_log if e['type'] in (MouseEvent.LEFT_CLICK, MouseEvent.RIGHT_CLICK)]
        assert len(clicks) == 0, "Transition from RIGHT_CLICK_PENDING to PINCH_CLOSED must not click"

    def test_click_pending_open_palm_cancels_and_pauses(self):
        fsm, ctrl = make_fsm()
        t = transition_to_moving(fsm)

        fsm.update(HandPose.PINCH_CLOSED, t + 10)
        assert fsm.state == FSMState.CLICK_PENDING

        fsm.update(HandPose.OPEN_PALM, t + 150)
        assert fsm.state == FSMState.PAUSED
        assert len(ctrl.event_log) == 0


# ===================================================================
# 4. Error Handling During mouse_up
# ===================================================================
class TestMouseUpFailureSafety:
    def test_mouseup_exception_not_swallowed_and_blocks_control(self):
        """If mouse_up fails, FSM propagates error, retains button state, and blocks cursor control."""
        failing_ctrl = FailingMouseUpController()
        fsm, _ = make_fsm(controller=failing_ctrl)

        t = transition_to_moving(fsm)
        t += 10
        fsm.update(HandPose.PINCH_CLOSED, t)
        t += 450
        fsm.update(HandPose.PINCH_CLOSED, t)
        assert fsm.state == FSMState.DRAGGING
        assert fsm.is_mouse_down is True

        # Now release pinch -> calls mouse_up which raises RuntimeError
        with pytest.raises(RuntimeError, match="OS mouse_up failed"):
            fsm.update(HandPose.POINT_ONE, t + 20)

        # Invariant checks:
        # 1. Button is NOT marked released
        assert fsm.is_mouse_down is True
        # 2. State does NOT falsely show MOVING
        assert fsm.state != FSMState.MOVING
        # 3. Cursor control is blocked
        assert fsm.can_move_cursor(HandPose.POINT_ONE) is False


# ===================================================================
# 5. can_move_cursor Signature & Gating
# ===================================================================
class TestCanMoveCursorBypassRemoval:
    def test_can_move_cursor_rejects_none_and_invalid(self):
        fsm, _ = make_fsm()
        transition_to_moving(fsm)

        # None is no longer bypassed
        assert fsm.can_move_cursor(None) is False
        # Only explicit HandPose.POINT_ONE is accepted
        assert fsm.can_move_cursor(HandPose.POINT_ONE) is True
        assert fsm.can_move_cursor(HandPose.PINCH_CLOSED) is False
        assert fsm.can_move_cursor(HandPose.UNKNOWN) is False


# ===================================================================
# 6. Minimum Config Validation
# ===================================================================
class TestConfigValidation:
    def test_reacquire_stable_frames_minimum(self):
        with pytest.raises(ValueError, match="reacquire_stable_frames must be >= 1"):
            make_fsm({'reacquire_stable_frames': 0})

    def test_timing_hierarchy_validation(self):
        # click_max >= drag_threshold is invalid
        with pytest.raises(ValueError, match="Invalid gesture timing hierarchy"):
            make_fsm({'click_min_ms': 80, 'click_max_ms': 450, 'drag_threshold_ms': 400})

        # click_min > click_max is invalid
        with pytest.raises(ValueError, match="Invalid gesture timing hierarchy"):
            make_fsm({'click_min_ms': 350, 'click_max_ms': 300, 'drag_threshold_ms': 400})
