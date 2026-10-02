"""
Unit tests for the ACC-09 Synchronized Safety Contract.

Verifies:
1. OS subsystem isolation fixture: pyautogui and keyboard globally mocked to guarantee
   zero physical OS mouse/keyboard/hook side-effects during tests.
2. Reentrant same-thread stop linearization: deferred cleanup executes after outermost
   dispatch and down bookkeeping complete (['operation_started', 'operation_finishes_down', 'release']).
3. Deterministic cross-thread stop serialization handshake: proves stop is latched while an actual
   WindowsController+FSM operation is in flight, followed by serialized cleanup and bounded thread joins.
4. Defect A: Stop callback inside rate_limiter.allow halts dispatch and emits EMERGENCY_STOP.
5. Defect B: Stop during drag start mouseDown leaves PAUSED, _is_mouse_down=False, held_buttons empty.
6. Defect C1: Failed mouseUp in cleanup retains held state, propagates exception, succeeds on retry.
7. Defect C1 (Keys): Failed keyUp in cleanup retains held state, propagates exception, succeeds on retry.
8. Defect C2: Down exceptions (mouseDown, keyDown) conservatively retain pending release for cleanup retry.
9. Defect C2 (Click): mouse_click partial down failure conservatively retains button for cleanup retry.
10. Defect C3: Modifier keys (Ctrl in zoom, Alt in swipe) are tracked in held_keys and released on cleanup.
11. Defect D: Fail-closed real OS control admission refuses startup and rolls back on:
    - Missing hotkey or dry_run hotkey
    - Keyboard hook registration exception
    - Missing keyboard library (ImportError)
    - Registration returning is_registered == False
12. Rollback on constructor feature_version mismatch after successful hook registration.
13. SafetyGate mismatch rejection across controller, FSM, and pipeline.
14. FSM serialization: Stop during left or right click callback leaves state PAUSED (no stale MOVING).
15. DryRun gate wiring: Controller and FSM gate references are bound to pipeline SafetyGate.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.safety.failsafe import SafetyGate, RateLimiter, FailsafeMonitor, EmergencyHotkey
from src.platform.windows import WindowsController
from src.platform.dry_run import DryRunController
from src.gesture.fsm import GestureFSM, FSMState
from src.gesture.pose import HandPose
from src.runtime.pipeline import AirPadPipeline, PipelineResult
from tools.train import ModelBundle
import app


@pytest.fixture(autouse=True)
def isolate_os_subsystems(monkeypatch):
    """Isolate all OS subsystems: mock pyautogui and keyboard globally in this test module."""
    mock_gui = MagicMock()
    mock_gui.size.return_value = (1920, 1080)
    mock_gui.hscroll = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", mock_gui)

    mock_kb = MagicMock()
    monkeypatch.setitem(sys.modules, "keyboard", mock_kb)
    return mock_gui, mock_kb


@pytest.fixture
def mock_gui(isolate_os_subsystems):
    """Provide access to the mocked pyautogui fixture."""
    gui, _ = isolate_os_subsystems
    return gui


@pytest.fixture
def mock_keyboard(isolate_os_subsystems):
    """Provide access to the mocked keyboard fixture."""
    _, kb = isolate_os_subsystems
    return kb


# ---------------------------------------------------------------------------
# 1. Reentrant same-thread stop linearization
# ---------------------------------------------------------------------------
def test_reentrant_stop_linearization_ordering():
    """Verify same-thread stop defers cleanup until active dispatch finishes down."""
    gate = SafetyGate()
    trace: List[str] = []

    def cleanup_fn():
        trace.append("release")

    def action_that_calls_trigger_stop_then_finishes_down():
        trace.append("operation_started")
        gate.trigger_stop(cleanup_fn)
        # Action completes its down operation and bookkeeping AFTER trigger_stop
        trace.append("operation_finishes_down")
        return "done"

    res = gate.dispatch(action_that_calls_trigger_stop_then_finishes_down)

    assert res == "done"
    assert trace == ["operation_started", "operation_finishes_down", "release"], (
        f"Expected down to finish before cleanup, got {trace}"
    )
    assert gate.is_stopped is True


# ---------------------------------------------------------------------------
# 2. Deterministic cross-thread stop serialization handshake
# ---------------------------------------------------------------------------
def test_cross_thread_stop_serialization_handshake(mock_gui, mock_keyboard):
    """Deterministic handshake proving stop latched while FSM+WindowsController operation in flight.

    Verifies:
    - Frame thread executes GestureFSM.update(PINCH_CLOSED, 500.0) triggering drag start (mouseDown).
    - sys.settrace intercepts right before self._is_mouse_down = True bookkeeping assignment.
    - Stop thread observes bookkeeping reached, calls pipeline.trigger_emergency_stop().
    - Frame thread observes stop is latched (gate.stop_requested_event is set) and resumes without artificial deadlock.
    - Stop thread cleanup is serialized and completes only AFTER caller bookkeeping finishes.
    - Final state is PAUSED, _is_mouse_down is False, held_buttons is empty.
    - Subsequent forward actions are refused.
    """
    import inspect
    gate = SafetyGate()
    ctrl = WindowsController(allow_real_control=True, gate=gate)
    fsm = GestureFSM(ctrl, {})
    fsm.gate = gate
    fsm.state = FSMState.CLICK_PENDING
    fsm._pinch_start_ms = 0.0

    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey.keyboard = mock_keyboard
    pipe = AirPadPipeline(config={}, controller=ctrl, fsm=fsm, emergency_hotkey=hotkey, gate=gate)

    target_func = getattr(GestureFSM, "_update_impl", GestureFSM.update)
    target_code = target_func.__code__
    lines, start_line = inspect.getsourcelines(target_func)
    target_lineno = None
    for idx, line in enumerate(lines):
        if "self._is_mouse_down = True" in line:
            target_lineno = start_line + idx
            break
    assert target_lineno is not None, "Target bookkeeping line not found in GestureFSM"

    order: List[str] = []
    bookkeeping_reached = threading.Event()
    stop_done = threading.Event()
    thread_errors: List[tuple] = []

    orig_cleanup = ctrl.cleanup
    def cleanup_with_order():
        order.append("cleanup_executed")
        return orig_cleanup()
    ctrl.cleanup = cleanup_with_order

    def trace_func(frame, event, arg):
        if event == "line" and frame.f_code is target_code and frame.f_lineno == target_lineno:
            bookkeeping_reached.set()
            # Observable latch handshake: wait until stop is latched (without artificial deadlock)
            latched = gate.stop_requested_event.wait(timeout=3.0)
            if not latched:
                raise TimeoutError("Stop latch was not observed within timeout")
            order.append("bookkeeping_assigned")
            # Clear trace after firing once
            sys.settrace(None)
            return None
        return trace_func

    def frame_worker():
        try:
            sys.settrace(trace_func)
            fsm.update(HandPose.PINCH_CLOSED, 500.0)
        except Exception as e:
            thread_errors.append(("frame", e))

    def stop_worker():
        try:
            assert bookkeeping_reached.wait(timeout=3.0), "Frame thread did not reach bookkeeping line"
            pipe.trigger_emergency_stop()
            stop_done.set()
        except Exception as e:
            thread_errors.append(("stop", e))

    t_frame = threading.Thread(target=frame_worker)
    t_stop = threading.Thread(target=stop_worker)

    t_frame.start()
    t_stop.start()

    t_frame.join(timeout=4.0)
    assert not t_frame.is_alive(), "Frame thread timed out or deadlocked"

    t_stop.join(timeout=4.0)
    assert not t_stop.is_alive(), "Stop thread timed out or deadlocked"

    assert len(thread_errors) == 0, f"Thread errors: {thread_errors}"
    assert stop_done.is_set(), "Stop thread did not finish successfully"
    assert order == ["bookkeeping_assigned", "cleanup_executed", "cleanup_executed"], (
        f"Protected bookkeeping must complete before cleanup executes; got {order}"
    )
    assert gate.is_stopped is True
    assert fsm.state == FSMState.PAUSED
    assert fsm.is_mouse_down is False
    assert len(ctrl.held_buttons) == 0
    assert ctrl.held_buttons == set()

    # Subsequent forward operations must be refused
    dummy_lms = np.full((21, 3), 0.5, dtype=np.float32)
    res_after = pipe.process_landmarks(dummy_lms, "Right", 600.0)
    assert res_after.action_fired == "EMERGENCY_STOP"
    assert res_after.fsm_state == FSMState.PAUSED


def test_cross_thread_pipeline_forward_output_suppression():
    """Deterministic handshake verifying forward output is suppressed when emergency stop is triggered.

    Verifies:
    - Pipeline is running with DryRunController.
    - Frame thread calls process_landmarks while in MOVING state (ready to emit MOVE).
    - Stop thread triggers emergency stop during frame processing.
    - Forward mouse_move is rejected and never logged in event_log.
    - Frame returns action_fired == "EMERGENCY_STOP".
    """
    import inspect
    gate = SafetyGate()
    ctrl = DryRunController()
    ctrl.gate = gate

    fsm = GestureFSM(ctrl, {})
    fsm.gate = gate
    fsm.state = FSMState.MOVING
    fsm._reacquire_count = 2

    mock_mapper = MagicMock()
    mock_mapper.map_point.return_value = (500, 500)
    mock_mapper.get_last_screen_pos.return_value = (500, 500)

    rate_limiter = RateLimiter(max_events_per_second=60)

    mock_classifier = MagicMock()
    mock_classifier.predict_with_rejection.return_value = np.array(["POINT_ONE"])

    pipeline = AirPadPipeline(
        config={},
        controller=ctrl,
        classifier=mock_classifier,
        fsm=fsm,
        cursor_mapper=mock_mapper,
        rate_limiter=rate_limiter,
        gate=gate,
    )

    lines, start_line = inspect.getsourcelines(AirPadPipeline.process_landmarks)
    target_lineno = None
    for idx, line in enumerate(lines):
        if "self.controller.mouse_move" in line:
            target_lineno = start_line + idx
            break
    assert target_lineno is not None, "Target mouse_move line not found in process_landmarks"

    motion_reached = threading.Event()
    stop_done = threading.Event()
    thread_errors: List[tuple] = []
    results: List[Any] = []

    def trace_func(frame, event, arg):
        if event == "line" and frame.f_code.co_name == "_execute_motion" and frame.f_lineno == target_lineno:
            motion_reached.set()
            latched = gate.stop_requested_event.wait(timeout=3.0)
            if not latched:
                raise TimeoutError("Stop latch was not observed within timeout")
            sys.settrace(None)
            return None
        return trace_func

    landmarks = np.zeros((21, 3), dtype=np.float32)

    def frame_worker():
        try:
            sys.settrace(trace_func)
            res = pipeline.process_landmarks(landmarks, timestamp_ms=100.0)
            results.append(res)
        except Exception as e:
            thread_errors.append(("frame", e))

    def stop_worker():
        try:
            assert motion_reached.wait(timeout=3.0), "Frame thread did not reach motion line"
            pipeline.trigger_emergency_stop()
            stop_done.set()
        except Exception as e:
            thread_errors.append(("stop", e))

    t_frame = threading.Thread(target=frame_worker)
    t_stop = threading.Thread(target=stop_worker)

    t_frame.start()
    t_stop.start()

    t_frame.join(timeout=4.0)
    assert not t_frame.is_alive(), "Frame thread timed out or deadlocked"

    t_stop.join(timeout=4.0)
    assert not t_stop.is_alive(), "Stop thread timed out or deadlocked"

    assert len(thread_errors) == 0, f"Thread errors: {thread_errors}"
    assert stop_done.is_set(), "Stop thread did not finish successfully"
    assert len(results) == 1, "Frame worker did not produce result"
    assert results[0].action_fired == "EMERGENCY_STOP"
    assert len(ctrl.event_log) == 0, f"No forward events should be logged once stopped, got {ctrl.event_log}"
    assert pipeline.is_emergency_stopped is True
    assert fsm.state == FSMState.PAUSED


# ---------------------------------------------------------------------------
# 3. Defect A: Stop callback inside rate_limiter.allow
# ---------------------------------------------------------------------------
def test_defect_a_stop_in_rate_limiter_allow_halts_move():
    """Verify stop triggered during rate_limiter.allow prevents MOVE dispatch and emits EMERGENCY_STOP."""
    gate = SafetyGate()
    ctrl = DryRunController()
    ctrl.gate = gate
    ctrl.mouse_move = MagicMock(wraps=ctrl.mouse_move)

    fsm = GestureFSM(ctrl, {})
    fsm.gate = gate
    fsm.state = FSMState.MOVING
    fsm._reacquire_count = 2

    mock_mapper = MagicMock()
    mock_mapper.map_point.return_value = (500, 500)

    rate_limiter = RateLimiter(max_events_per_second=60)

    mock_classifier = MagicMock()
    mock_classifier.predict_with_rejection.return_value = np.array(["POINT_ONE"])

    pipeline = AirPadPipeline(
        config={},
        controller=ctrl,
        classifier=mock_classifier,
        fsm=fsm,
        cursor_mapper=mock_mapper,
        rate_limiter=rate_limiter,
        gate=gate,
    )

    # Intercept rate_limiter.allow to trigger emergency stop mid-check
    def stop_inside_rate_limiter():
        pipeline.trigger_emergency_stop()
        return True

    pipeline.rate_limiter.allow = stop_inside_rate_limiter

    landmarks = np.zeros((21, 3))
    res = pipeline.process_landmarks(landmarks, timestamp_ms=100.0)

    assert pipeline.is_emergency_stopped is True
    assert fsm.state == FSMState.PAUSED
    assert res.action_fired == "EMERGENCY_STOP"
    assert ctrl.mouse_move.call_count == 0


# ---------------------------------------------------------------------------
# 4. Defect B: Reentrant stop during drag start mouseDown
# ---------------------------------------------------------------------------
def test_defect_b_drag_start_reentrant_stop_cleans_bookkeeping(mock_gui, mock_keyboard):
    """Verify stop during mouseDown in drag start leaves FSM PAUSED, _is_mouse_down False, and held_buttons empty."""
    gate = SafetyGate()
    ctrl = WindowsController(allow_real_control=True, gate=gate)
    fsm_cfg = {
        "click_min_ms": 20.0,
        "click_max_ms": 50.0,
        "drag_threshold_ms": 100.0,
    }
    fsm = GestureFSM(ctrl, fsm_cfg)
    fsm.gate = gate
    fsm.state = FSMState.CLICK_PENDING
    fsm._pinch_start_ms = 0.0

    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey.keyboard = mock_keyboard

    pipeline = AirPadPipeline(
        config={},
        controller=ctrl,
        fsm=fsm,
        emergency_hotkey=hotkey,
        gate=gate,
    )

    # During mock pyautogui.mouseDown, trigger emergency stop reentrantly
    def fake_mouse_down(**kw):
        pipeline.trigger_emergency_stop()

    mock_gui.mouseDown.side_effect = fake_mouse_down

    fsm.update(HandPose.PINCH_CLOSED, timestamp_ms=150.0)

    assert pipeline.is_emergency_stopped is True
    assert fsm.state == FSMState.PAUSED
    assert fsm.is_mouse_down is False
    assert len(ctrl.held_buttons) == 0
    assert len(ctrl._held_buttons) == 0


# ---------------------------------------------------------------------------
# 5. Defect C1: Cleanup retention and retry on mouseUp/keyUp failure
# ---------------------------------------------------------------------------
def test_defect_c1_cleanup_failure_retains_for_retry(mock_gui):
    """Verify failed mouseUp release in cleanup keeps button in held tracking and retry releases it."""
    ctrl = WindowsController(allow_real_control=True)
    ctrl.mouse_down("left")
    ctrl.mouse_down("right")
    ctrl.key_down("shift")

    # Fail first button release, succeed on all subsequent calls
    mock_gui.mouseUp.side_effect = [Exception("mouseUp failed"), None, None]

    with pytest.raises(Exception, match="mouseUp failed"):
        ctrl.cleanup()

    # The failed button remains in held_buttons
    assert len(ctrl.held_buttons) == 1
    assert len(ctrl.held_keys) == 0

    # Second cleanup retries and succeeds
    ctrl.cleanup()
    assert len(ctrl.held_buttons) == 0
    assert len(ctrl.held_keys) == 0


def test_cleanup_key_up_failure_retry(mock_gui):
    """Verify failed keyUp release in cleanup keeps key in held tracking and retry releases it."""
    ctrl = WindowsController(allow_real_control=True)
    ctrl.key_down("ctrl")
    ctrl.key_down("alt")

    # Fail first keyUp release, succeed on subsequent
    mock_gui.keyUp.side_effect = [RuntimeError("keyUp driver error"), None, None]

    with pytest.raises(RuntimeError, match="keyUp driver error"):
        ctrl.cleanup()

    assert len(ctrl.held_keys) == 1
    # Second cleanup retries and succeeds
    ctrl.cleanup()
    assert len(ctrl.held_keys) == 0


# ---------------------------------------------------------------------------
# 6. Defect C2: Down exceptions retain tracked state for cleanup retry
# ---------------------------------------------------------------------------
def test_defect_c2_down_exception_retains_tracked_state(mock_gui):
    """Verify when mouseDown raises after partial effect, button is retained and released on cleanup."""
    ctrl = WindowsController(allow_real_control=True)

    # Simulate mouseDown raising after partial OS/driver interaction
    mock_gui.mouseDown.side_effect = RuntimeError("OS driver failure on down")

    with pytest.raises(RuntimeError, match="OS driver failure"):
        ctrl.mouse_down("left")

    # Conservative safety: 'left' is retained so cleanup can release it
    assert "left" in ctrl.held_buttons

    # Reset mock and run cleanup: mouseUp must be attempted
    mock_gui.mouseUp.side_effect = None
    ctrl.cleanup()

    mock_gui.mouseUp.assert_called_with(button="left")
    assert len(ctrl.held_buttons) == 0


def test_key_down_failure_retains_tracking(mock_gui):
    """Verify when keyDown raises after partial effect, key is retained and released on cleanup."""
    ctrl = WindowsController(allow_real_control=True)
    mock_gui.keyDown.side_effect = RuntimeError("Key down hardware error")

    with pytest.raises(RuntimeError, match="Key down hardware error"):
        ctrl.key_down("shift")

    assert "shift" in ctrl.held_keys
    mock_gui.keyUp.side_effect = None
    ctrl.cleanup()
    mock_gui.keyUp.assert_called_with("shift")
    assert len(ctrl.held_keys) == 0


def test_mouse_click_partial_down_failure_retains_tracking(mock_gui):
    """Verify when mouse_click fails partially, button is conservatively retained and released on cleanup."""
    ctrl = WindowsController(allow_real_control=True)
    mock_gui.click.side_effect = RuntimeError("Click stub failure mid-event")

    with pytest.raises(RuntimeError, match="Click stub failure"):
        ctrl.mouse_click("left")

    # Conservative safety: 'left' is retained because mouse could be physically down
    assert "left" in ctrl.held_buttons
    ctrl.cleanup()
    mock_gui.mouseUp.assert_called_with(button="left")
    assert len(ctrl.held_buttons) == 0


# ---------------------------------------------------------------------------
# 7. Defect C3: Modifier keys (Ctrl/Alt) in zoom/swipe tracking and cleanup
# ---------------------------------------------------------------------------
def test_defect_c3_modifier_keys_tracked_and_cleaned(mock_gui):
    """Verify Ctrl in zoom is tracked and released on emergency cleanup."""
    ctrl = WindowsController(allow_real_control=True)

    def stop_during_zoom(steps):
        assert "ctrl" in ctrl.held_keys
        ctrl.gate.trigger_stop(ctrl.cleanup)

    mock_gui.scroll.side_effect = stop_during_zoom

    ctrl.zoom(3)

    assert ctrl.gate.is_stopped is True
    assert len(ctrl.held_keys) == 0
    mock_gui.keyUp.assert_called_with("ctrl")


def test_alt_swipe_modifier_tracked_and_cleaned(mock_gui):
    """Verify Alt in swipe is tracked and released on emergency cleanup."""
    ctrl = WindowsController(allow_real_control=True)

    def stop_during_swipe(key):
        assert "alt" in ctrl.held_keys
        ctrl.gate.trigger_stop(ctrl.cleanup)

    mock_gui.press.side_effect = stop_during_swipe

    ctrl.swipe("BACK")

    assert ctrl.gate.is_stopped is True
    assert len(ctrl.held_keys) == 0
    mock_gui.keyUp.assert_called_with("alt")


# ---------------------------------------------------------------------------
# 8. Defect D: Fail-closed real OS control admission
# ---------------------------------------------------------------------------
def test_defect_d_real_control_admission_refused_without_hotkey():
    """Verify real control pipeline refuses admission if hotkey is missing or dry_run."""
    ctrl = WindowsController(allow_real_control=True)

    with pytest.raises(RuntimeError, match="emergency hotkey is required"):
        AirPadPipeline(config={}, controller=ctrl, emergency_hotkey=None)

    dry_hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=True)
    with pytest.raises(RuntimeError, match="cannot be dry_run"):
        AirPadPipeline(config={}, controller=ctrl, emergency_hotkey=dry_hotkey)


def test_defect_d_real_control_admission_rolls_back_on_hook_failure():
    """Verify real control rolls back and unregisters if keyboard hook registration fails."""
    ctrl = WindowsController(allow_real_control=True)
    ctrl.cleanup = MagicMock(wraps=ctrl.cleanup)

    fake_keyboard = MagicMock()
    fake_keyboard.add_hotkey.side_effect = RuntimeError("Access denied for low-level hook")

    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey.keyboard = fake_keyboard

    with pytest.raises(RuntimeError, match="emergency hotkey registration failed"):
        AirPadPipeline(config={}, controller=ctrl, emergency_hotkey=hotkey)

    assert ctrl.cleanup.call_count >= 1
    assert hotkey.is_registered is False


def test_admission_refused_when_keyboard_missing():
    """Verify real control refuses admission if keyboard module is missing."""
    ctrl = WindowsController(allow_real_control=True)
    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey.keyboard = None
    hotkey._import_error = ImportError("No module named 'keyboard'")

    with pytest.raises(RuntimeError, match="'keyboard' library unavailable"):
        AirPadPipeline(config={}, controller=ctrl, emergency_hotkey=hotkey)


def test_admission_refused_when_register_returns_unregistered():
    """Verify real control refuses admission if hotkey.register fails to latch is_registered."""
    ctrl = WindowsController(allow_real_control=True)
    fake_keyboard = MagicMock()
    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey.keyboard = fake_keyboard
    # Simulate register returning without setting _registered = True
    hotkey.register = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="emergency hotkey not registered"):
        AirPadPipeline(config={}, controller=ctrl, emergency_hotkey=hotkey)


# ---------------------------------------------------------------------------
# 9. Constructor rollback on feature_version mismatch
# ---------------------------------------------------------------------------
def test_constructor_feature_version_mismatch_rolls_back_hotkey(mock_keyboard):
    """Verify feature_version mismatch in constructor unregisters hotkey and cleans up controller."""
    ctrl = WindowsController(allow_real_control=True)
    ctrl.cleanup = MagicMock(wraps=ctrl.cleanup)

    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey.keyboard = mock_keyboard

    bad_bundle = MagicMock(spec=ModelBundle)
    bad_bundle.feature_version = 2

    with pytest.raises(ValueError, match="feature version mismatch"):
        AirPadPipeline(config={}, controller=ctrl, emergency_hotkey=hotkey, model_bundle=bad_bundle)

    assert ctrl.cleanup.call_count >= 1
    assert mock_keyboard.remove_hotkey.called


# ---------------------------------------------------------------------------
# 10. SafetyGate mismatch rejection across controller, FSM, and pipeline
# ---------------------------------------------------------------------------
def test_gate_mismatch_rejected():
    """Verify explicit gate mismatch with controller.gate or fsm.gate is rejected."""
    gate_a = SafetyGate()
    gate_b = SafetyGate()

    ctrl = DryRunController()
    ctrl.gate = gate_a

    with pytest.raises(ValueError, match="SafetyGate mismatch between explicit gate and controller.gate"):
        AirPadPipeline(config={}, controller=ctrl, gate=gate_b)

    fsm = GestureFSM(ctrl, {})
    fsm.gate = gate_b
    ctrl.gate = gate_a

    with pytest.raises(ValueError, match="SafetyGate mismatch"):
        AirPadPipeline(config={}, controller=ctrl, fsm=fsm)


# ---------------------------------------------------------------------------
# 11. FSM serialization: Stop during click callback leaves state PAUSED
# ---------------------------------------------------------------------------
def test_fsm_stop_during_click_callback_leaves_paused():
    """Verify stop triggered during left click does not leave FSM in MOVING."""
    gate = SafetyGate()
    ctrl = DryRunController()
    ctrl.gate = gate

    fsm = GestureFSM(ctrl, {"click_min_ms": 50.0, "click_max_ms": 200.0})
    fsm.gate = gate
    fsm.state = FSMState.CLICK_PENDING
    fsm._pinch_start_ms = 0.0

    def click_and_stop(button="left"):
        gate.trigger_stop(fsm.on_emergency_stop)

    ctrl.mouse_click = click_and_stop

    fsm.update(HandPose.POINT_ONE, timestamp_ms=100.0)

    assert gate.is_stopped is True
    assert fsm.state == FSMState.PAUSED
    assert fsm.is_stopped is True

    # Subsequent frames remain latched in PAUSED
    fsm.update(HandPose.POINT_ONE, timestamp_ms=200.0)
    assert fsm.state == FSMState.PAUSED


def test_fsm_stop_during_right_click_callback_leaves_paused():
    """Verify stop triggered during right click does not leave FSM in MOVING."""
    gate = SafetyGate()
    ctrl = DryRunController()
    ctrl.gate = gate

    fsm = GestureFSM(ctrl, {"click_min_ms": 50.0, "click_max_ms": 200.0})
    fsm.gate = gate
    fsm.state = FSMState.RIGHT_CLICK_PENDING
    fsm._right_pinch_start_ms = 0.0

    def click_and_stop(button="right"):
        gate.trigger_stop(fsm.on_emergency_stop)

    ctrl.mouse_click = click_and_stop

    fsm.update(HandPose.POINT_ONE, timestamp_ms=100.0)

    assert gate.is_stopped is True
    assert fsm.state == FSMState.PAUSED
    assert fsm.is_stopped is True


# ---------------------------------------------------------------------------
# 12. DryRun gate wiring
# ---------------------------------------------------------------------------
def test_dry_run_gate_wiring_and_rejection():
    """Verify DryRunController is wired to SafetyGate and respects stop."""
    pipeline = app.create_pipeline(app.load_config(), allow_real_control=False)

    assert pipeline.gate is not None
    assert pipeline.controller.gate is pipeline.gate
    assert pipeline.fsm.gate is pipeline.gate

    pipeline.trigger_emergency_stop()
    assert pipeline.is_emergency_stopped is True
    assert pipeline.controller.gate.is_stopped is True
    assert pipeline.fsm.is_stopped is True


# ---------------------------------------------------------------------------
# 13. EmergencyHotkey retryable unregister
# ---------------------------------------------------------------------------
def test_emergency_hotkey_unregister_retry(mock_keyboard):
    """Verify failed remove_hotkey in unregister retains _registered=True and succeeds on retry."""
    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey.keyboard = mock_keyboard
    hotkey.register()
    assert hotkey.is_registered is True

    # First unregister fails (e.g. OS hook removal error)
    mock_keyboard.remove_hotkey.side_effect = RuntimeError("Hook remove failure")
    with pytest.raises(RuntimeError, match="Failed to unregister emergency hotkey"):
        hotkey.unregister()

    # Retained on failure: avoids false rollback success
    assert hotkey.is_registered is True

    # Second unregister succeeds
    mock_keyboard.remove_hotkey.side_effect = None
    hotkey.unregister()
    assert hotkey.is_registered is False
    assert mock_keyboard.remove_hotkey.call_count == 2


# ---------------------------------------------------------------------------
# 14. Partial press failure and composite finally keyUp failures
# ---------------------------------------------------------------------------
def test_swipe_partial_press_failure_retains_and_cleans_up(mock_gui):
    """Verify swipe partial press failure retains key and releases both physical and tracked on cleanup."""
    ctrl = WindowsController(allow_real_control=True)
    physical_held = set()

    def fake_press(key):
        physical_held.add(key)
        raise RuntimeError("press failed after keyDown")

    def fake_key_up(key):
        physical_held.discard(key)

    mock_gui.press.side_effect = fake_press
    mock_gui.keyUp.side_effect = fake_key_up

    with pytest.raises(RuntimeError, match="press failed after keyDown"):
        ctrl.swipe("BACK")

    assert "left" in ctrl.held_keys, "Navigation key must be retained in held_keys on partial press failure"
    assert "left" in physical_held, "Physical key was left held by press failure"
    assert "alt" not in ctrl.held_keys, "Alt modifier was cleaned up in finally block"

    # Cleanup releases retained navigation key and clears physical mock set
    ctrl.cleanup()
    assert len(ctrl.held_keys) == 0
    assert len(physical_held) == 0


def test_composite_finally_keyup_failures_retained_for_cleanup(mock_gui):
    """Verify keyUp failure during composite finally in zoom and swipe retains modifier keys for cleanup retry."""
    ctrl = WindowsController(allow_real_control=True)

    # In zoom: scroll succeeds, but keyUp('ctrl') fails in finally
    mock_gui.keyUp.side_effect = RuntimeError("Driver failure on keyUp")
    with pytest.raises(RuntimeError, match="Driver failure on keyUp"):
        ctrl.zoom(2)

    assert "ctrl" in ctrl.held_keys, "Ctrl must be retained in held_keys when finally keyUp fails"

    # In swipe: press succeeds, but keyUp('alt') fails in finally
    with pytest.raises(RuntimeError, match="Driver failure on keyUp"):
        ctrl.swipe("FORWARD")

    assert "alt" in ctrl.held_keys, "Alt must be retained in held_keys when finally keyUp fails"

    # Now cleanup is called: first attempt propagates error
    with pytest.raises(RuntimeError, match="Driver failure on keyUp"):
        ctrl.cleanup()

    assert "ctrl" in ctrl.held_keys and "alt" in ctrl.held_keys

    # Next attempt succeeds
    mock_gui.keyUp.side_effect = None
    ctrl.cleanup()
    assert len(ctrl.held_keys) == 0


# ---------------------------------------------------------------------------
# 15. Failed emergency cleanup preserves FSM down across stopped frames
# ---------------------------------------------------------------------------
def test_failed_cleanup_preserves_fsm_down_across_stopped_frames_until_successful_retry(mock_gui):
    """Verify failed cleanup leaves _is_mouse_down=True, frames while stopped DO NOT clear it, and retry clears."""
    gate = SafetyGate()
    ctrl = WindowsController(allow_real_control=True, gate=gate)
    fsm = GestureFSM(ctrl, {})
    fsm.gate = gate

    # Start dragging
    ctrl.mouse_down("left")
    fsm.state = FSMState.DRAGGING
    fsm._is_mouse_down = True

    # Emergency stop triggers, but mouseUp fails in driver
    mock_gui.mouseUp.side_effect = RuntimeError("mouseUp driver failure during emergency cleanup")
    with pytest.raises(RuntimeError, match="mouseUp driver failure"):
        fsm.on_emergency_stop()

    assert fsm.is_stopped is True
    assert fsm.is_mouse_down is True, "Failed release must leave _is_mouse_down True"
    assert "left" in ctrl.held_buttons, "Failed release must retain button in held_buttons"

    # Next frame arrives while stopped: must NOT clear _is_mouse_down!
    state = fsm.update(HandPose.POINT_ONE, timestamp_ms=100.0)
    assert state == FSMState.PAUSED
    assert fsm.is_mouse_down is True, "Stopped frame must NOT clear _is_mouse_down while release is still pending!"
    assert "left" in ctrl.held_buttons

    # Successful explicit cleanup retry clears held state confirmed
    mock_gui.mouseUp.side_effect = None
    fsm.on_emergency_stop()
    assert fsm.is_mouse_down is False
    assert len(ctrl.held_buttons) == 0


# ---------------------------------------------------------------------------
# 16. Registered hotkey with missing backend fails and retains registration
# ---------------------------------------------------------------------------
def test_registered_hotkey_with_missing_backend_fails_and_retains_registration(mock_keyboard):
    """Verify registered hook with missing backend raises on unregister and retains _registered=True."""
    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey.keyboard = mock_keyboard
    hotkey.register()
    assert hotkey.is_registered is True

    # Backend becomes None (e.g. unhook error or missing backend)
    hotkey.keyboard = None
    with pytest.raises(RuntimeError, match="library unavailable"):
        hotkey.unregister()

    assert hotkey.is_registered is True, "Must retain registration when backend is missing and cannot unhook"

    # Restoring backend allows successful unregister
    hotkey.keyboard = mock_keyboard
    hotkey.unregister()
    assert hotkey.is_registered is False
    mock_keyboard.remove_hotkey.assert_called_with("ctrl+alt+q")

