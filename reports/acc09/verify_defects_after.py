"""
Verify resolution of safety defects (P1 A, B, C, D and same-thread gate reentrancy)
against live updated modules on disk.
This script proves all defects are deterministically resolved on the post-fix code.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
from unittest.mock import MagicMock, patch
import numpy as np

from src.safety.failsafe import SafetyGate, RateLimiter, FailsafeMonitor, EmergencyHotkey
from src.platform.windows import WindowsController
from src.platform.dry_run import DryRunController
from src.platform.interface import MouseEvent
from src.gesture.fsm import GestureFSM, FSMState
from src.gesture.pose import HandPose
from src.runtime.pipeline import AirPadPipeline, PipelineResult
from src.motion.cursor import CursorMapper
from src.config.loader import load_config
import app

print("=" * 70)
print("ACC-09 POST-FIX SAFETY DEFECT VERIFICATION")
print("=" * 70)

# ==============================================================================
# TEST 1: Reentrant stop linearization in SafetyGate
# ==============================================================================
print("\n[VERIFICATION 1] Reentrant stop linearization ordering...")
gate = SafetyGate()
trace = []

def cleanup_fn():
    trace.append("release")

def action_that_calls_trigger_stop_then_finishes_down():
    trace.append("operation_started")
    gate.trigger_stop(cleanup_fn)
    trace.append("operation_finishes_down")
    return "ok"

res = gate.dispatch(action_that_calls_trigger_stop_then_finishes_down)
print(f"  -> Execution trace: {trace}")
assert trace == ["operation_started", "operation_finishes_down", "release"], (
    f"Failed linearization: expected down before release, got {trace}"
)
assert gate.is_stopped is True
print("  [PASSED] Reentrant stop correctly linearizes: down finishes before release.")

# ==============================================================================
# TEST 2: Defect A - Stop in rate_limiter.allow halts MOVE
# ==============================================================================
print("\n[VERIFICATION 2] Defect A fix: Stop in rate_limiter.allow...")
config = load_config()
ctrl_a = DryRunController()
fsm_a = GestureFSM(ctrl_a, config.get("gesture", {}))
fsm_a.state = FSMState.MOVING
fsm_a._reacquire_count = 5
fsm_a._unknown_count = 0

cursor_mapper_a = CursorMapper()
failsafe_mon_a = FailsafeMonitor(1920, 1080, 10)
rate_limiter_a = RateLimiter(60)

class AlwaysPointOneClassifier:
    def predict_with_rejection(self, canon_flat, decision_threshold=0.0, margin_threshold=0.15, unknown_label="UNKNOWN"):
        return ["POINT_ONE"]

pipe_a = AirPadPipeline(
    config=config,
    controller=ctrl_a,
    classifier=AlwaysPointOneClassifier(),
    fsm=fsm_a,
    cursor_mapper=cursor_mapper_a,
    failsafe_monitor=failsafe_mon_a,
    rate_limiter=rate_limiter_a,
)

def stop_inside_rate_limiter():
    pipe_a.trigger_emergency_stop()
    return True

rate_limiter_a.allow = stop_inside_rate_limiter

dummy_lms = np.full((21, 3), 0.5, dtype=np.float32)
res_a = pipe_a.process_landmarks(dummy_lms, "Right", 1000.0)

move_events_a = [e for e in ctrl_a.event_log if e.get("type") == MouseEvent.MOVE]
print(f"  -> pipe_a.is_emergency_stopped = {pipe_a.is_emergency_stopped}")
print(f"  -> fsm_a.state = {fsm_a.state.name}")
print(f"  -> action_fired = {res_a.action_fired}")
print(f"  -> MOVE events emitted after stop = {len(move_events_a)}")
assert pipe_a.is_emergency_stopped is True
assert fsm_a.state == FSMState.PAUSED
assert len(move_events_a) == 0, "No MOVE event must be emitted after stop!"
assert res_a.action_fired == "EMERGENCY_STOP"
print("  [PASSED] Defect A resolved: MOVE halted, EMERGENCY_STOP emitted.")

# ==============================================================================
# TEST 3: Defect B - Reentrant stop during drag start mouseDown
# ==============================================================================
print("\n[VERIFICATION 3] Defect B fix: Reentrant stop during drag start mouseDown...")
mock_gui_b = MagicMock()
mock_gui_b.size.return_value = (1920, 1080)
with patch.dict(sys.modules, {"pyautogui": mock_gui_b}):
    ctrl_b = WindowsController(allow_real_control=True)
    fsm_cfg_b = config.get("gesture", {}).copy()
    fsm_b = GestureFSM(ctrl_b, fsm_cfg_b)
    fsm_b.state = FSMState.CLICK_PENDING
    fsm_b._pinch_start_ms = 0.0

    fake_kb_b = MagicMock()
    hotkey_b = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey_b.keyboard = fake_kb_b

    pipe_b = AirPadPipeline(
        config=config,
        controller=ctrl_b,
        fsm=fsm_b,
        emergency_hotkey=hotkey_b,
    )

    def stop_during_mouse_down(**kw):
        pipe_b.trigger_emergency_stop()

    mock_gui_b.mouseDown.side_effect = stop_during_mouse_down

    fsm_b.update(HandPose.PINCH_CLOSED, 500.0)

    print(f"  -> pipe_b.is_emergency_stopped = {pipe_b.is_emergency_stopped}")
    print(f"  -> fsm_b.state = {fsm_b.state.name}")
    print(f"  -> fsm_b.is_mouse_down = {fsm_b.is_mouse_down}")
    print(f"  -> ctrl_b.held_buttons = {ctrl_b.held_buttons}")
    assert pipe_b.is_emergency_stopped is True
    assert fsm_b.state == FSMState.PAUSED
    assert fsm_b.is_mouse_down is False
    assert len(ctrl_b.held_buttons) == 0
    print("  [PASSED] Defect B resolved: state is PAUSED, mouse_down is False, held_buttons empty.")

# ==============================================================================
# TEST 4: Defect C1 - Cleanup retry retention and exception propagation
# ==============================================================================
print("\n[VERIFICATION 4] Defect C1 fix: Cleanup retry on failure...")
mock_gui_c = MagicMock()
mock_gui_c.size.return_value = (1920, 1080)
with patch.dict(sys.modules, {"pyautogui": mock_gui_c}):
    ctrl_c = WindowsController(allow_real_control=True)
    ctrl_c.mouse_down("left")
    ctrl_c.mouse_down("right")

    # Fail first button release, succeed on next
    mock_gui_c.mouseUp.side_effect = [RuntimeError("release failed"), None, None]

    try:
        ctrl_c.cleanup()
        assert False, "Expected cleanup to propagate exception"
    except RuntimeError as e:
        print(f"  -> Caught expected cleanup exception: {e}")

    print(f"  -> ctrl_c.held_buttons after pass 1 = {ctrl_c.held_buttons}")
    assert len(ctrl_c.held_buttons) == 1, "Failed button must be retained for retry"

    # Second cleanup retries and succeeds
    ctrl_c.cleanup()
    print(f"  -> ctrl_c.held_buttons after pass 2 = {ctrl_c.held_buttons}")
    assert len(ctrl_c.held_buttons) == 0, "Second cleanup must clear remaining button"
    print("  [PASSED] Defect C1 resolved: failed button retained and cleared on retry.")

# ==============================================================================
# TEST 5: Defect C2 - Down exception retains tracked state for cleanup
# ==============================================================================
print("\n[VERIFICATION 5] Defect C2 fix: Down exception retains tracked state...")
mock_gui_c2 = MagicMock()
mock_gui_c2.size.return_value = (1920, 1080)
with patch.dict(sys.modules, {"pyautogui": mock_gui_c2}):
    ctrl_c2 = WindowsController(allow_real_control=True)
    mock_gui_c2.mouseDown.side_effect = RuntimeError("Driver failure on press")

    try:
        ctrl_c2.mouse_down("left")
        assert False, "Expected down to raise"
    except RuntimeError as e:
        pass

    assert "left" in ctrl_c2.held_buttons, "Button must be retained in tracking on down exception"
    mock_gui_c2.mouseUp.side_effect = None
    ctrl_c2.cleanup()
    assert len(ctrl_c2.held_buttons) == 0
    mock_gui_c2.mouseUp.assert_called_with(button="left")
    print("  [PASSED] Defect C2 resolved: button retained on down error and released by cleanup.")

# ==============================================================================
# TEST 6: Defect C3 - Modifier keys in zoom/swipe tracking and cleanup
# ==============================================================================
print("\n[VERIFICATION 6] Defect C3 fix: Modifier keys in zoom/swipe...")
mock_gui_c3 = MagicMock()
mock_gui_c3.size.return_value = (1920, 1080)
with patch.dict(sys.modules, {"pyautogui": mock_gui_c3}):
    ctrl_c3 = WindowsController(allow_real_control=True)

    def stop_during_zoom(steps):
        assert "ctrl" in ctrl_c3.held_keys
        ctrl_c3.gate.trigger_stop(ctrl_c3.cleanup)

    mock_gui_c3.scroll.side_effect = stop_during_zoom
    ctrl_c3.zoom(2)
    assert ctrl_c3.gate.is_stopped is True
    assert len(ctrl_c3.held_keys) == 0
    mock_gui_c3.keyUp.assert_called_with("ctrl")
    print("  [PASSED] Defect C3 resolved: ctrl modifier tracked and released on emergency stop.")

# ==============================================================================
# TEST 7: Defect D - Fail-closed real OS control admission
# ==============================================================================
print("\n[VERIFICATION 7] Defect D fix: Fail-closed real OS control admission...")
mock_gui_d = MagicMock()
mock_gui_d.size.return_value = (1920, 1080)
with patch.dict(sys.modules, {"pyautogui": mock_gui_d}):
    ctrl_d = WindowsController(allow_real_control=True)

    # Missing hotkey
    try:
        AirPadPipeline(config={}, controller=ctrl_d, emergency_hotkey=None)
        assert False, "Admission should fail without hotkey"
    except RuntimeError as e:
        print(f"  -> Caught expected missing hotkey error: {e}")

    # Hook registration failure
    fake_kb_d = MagicMock()
    fake_kb_d.add_hotkey.side_effect = RuntimeError("Hook denied")
    hotkey_d = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    hotkey_d.keyboard = fake_kb_d

    try:
        AirPadPipeline(config={}, controller=ctrl_d, emergency_hotkey=hotkey_d)
        assert False, "Admission should fail when hook fails"
    except RuntimeError as e:
        print(f"  -> Caught expected hook failure: {e}")

    assert hotkey_d.is_registered is False
    print("  [PASSED] Defect D resolved: real control admission refused and rolled back.")

# ==============================================================================
# TEST 8: FSM serialization during click callback
# ==============================================================================
print("\n[VERIFICATION 8] FSM serialization during click callback...")
gate_fsm = SafetyGate()
ctrl_fsm = DryRunController()
ctrl_fsm.gate = gate_fsm
fsm_e = GestureFSM(ctrl_fsm, {"click_min_ms": 50.0, "click_max_ms": 200.0})
fsm_e.gate = gate_fsm
fsm_e.state = FSMState.CLICK_PENDING
fsm_e._pinch_start_ms = 0.0

def stop_in_click(button="left"):
    gate_fsm.trigger_stop(fsm_e.on_emergency_stop)

ctrl_fsm.mouse_click = stop_in_click
fsm_e.update(HandPose.POINT_ONE, 100.0)

assert gate_fsm.is_stopped is True
assert fsm_e.state == FSMState.PAUSED
print("  [PASSED] FSM click callback stop correctly leaves state in PAUSED (no stale MOVING).")

# ==============================================================================
# TEST 9: Swipe partial press failure & composite finally keyUp
# ==============================================================================
print("\n[VERIFICATION 9] Swipe partial press failure & composite finally keyUp...")
mock_gui_9 = MagicMock()
mock_gui_9.size.return_value = (1920, 1080)
with patch.dict(sys.modules, {"pyautogui": mock_gui_9}):
    ctrl_9 = WindowsController(allow_real_control=True)
    phys_held_9 = set()

    def fake_press_9(key):
        phys_held_9.add(key)
        raise RuntimeError("press failed after keyDown")

    def fake_keyup_9(key):
        phys_held_9.discard(key)

    mock_gui_9.press.side_effect = fake_press_9
    mock_gui_9.keyUp.side_effect = fake_keyup_9

    try:
        ctrl_9.swipe("BACK")
        assert False, "Expected swipe to raise on press failure"
    except RuntimeError as e:
        pass

    assert "left" in ctrl_9.held_keys, "Navigation key must be retained on partial press failure"
    assert "left" in phys_held_9, "Physical mock must contain held key"

    ctrl_9.cleanup()
    assert len(ctrl_9.held_keys) == 0
    assert len(phys_held_9) == 0
    print("  [PASSED] Swipe partial press retains key and cleanup releases physical & tracked state.")

# ==============================================================================
# TEST 10: EmergencyHotkey unregister failure retry
# ==============================================================================
print("\n[VERIFICATION 10] EmergencyHotkey unregister failure retention & retry...")
fake_kb_10 = MagicMock()
hotkey_10 = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
hotkey_10.keyboard = fake_kb_10
hotkey_10.register()
assert hotkey_10.is_registered is True

fake_kb_10.remove_hotkey.side_effect = RuntimeError("Hook remove failure")
try:
    hotkey_10.unregister()
    assert False, "Expected unregister to raise on failure"
except RuntimeError as e:
    pass

assert hotkey_10.is_registered is True, "Failed unregister must retain _registered=True"

fake_kb_10.remove_hotkey.side_effect = None
hotkey_10.unregister()
assert hotkey_10.is_registered is False, "Successful retry must clear _registered flag"
print("  [PASSED] EmergencyHotkey unregister failure retains registration and retries cleanly.")

# ==============================================================================
# TEST 11: Failed cleanup preserves FSM down across stopped frames until retry
# ==============================================================================
print("\n[VERIFICATION 11] Failed cleanup preserves _is_mouse_down across stopped frames...")
mock_gui_11 = MagicMock()
mock_gui_11.size.return_value = (1920, 1080)
with patch.dict(sys.modules, {"pyautogui": mock_gui_11}):
    gate_11 = SafetyGate()
    ctrl_11 = WindowsController(allow_real_control=True, gate=gate_11)
    fsm_11 = GestureFSM(ctrl_11, {})
    fsm_11.gate = gate_11

    ctrl_11.mouse_down("left")
    fsm_11.state = FSMState.DRAGGING
    fsm_11._is_mouse_down = True

    mock_gui_11.mouseUp.side_effect = RuntimeError("mouseUp driver failure during emergency cleanup")
    try:
        fsm_11.on_emergency_stop()
        assert False, "Expected on_emergency_stop to raise on cleanup failure"
    except RuntimeError:
        pass

    assert fsm_11.is_stopped is True
    assert fsm_11.is_mouse_down is True, "Failed release must leave _is_mouse_down True"
    assert "left" in ctrl_11.held_buttons, "Failed release must retain button in held_buttons"

    # Frame arrives while stopped: must NOT clear _is_mouse_down
    st = fsm_11.update(HandPose.POINT_ONE, timestamp_ms=100.0)
    assert st == FSMState.PAUSED
    assert fsm_11.is_mouse_down is True, "Stopped frame must NOT clear _is_mouse_down while release pending!"

    # Explicit cleanup retry succeeds
    mock_gui_11.mouseUp.side_effect = None
    fsm_11.on_emergency_stop()
    assert fsm_11.is_mouse_down is False
    assert len(ctrl_11.held_buttons) == 0
    print("  [PASSED] FSM down preserved across stopped frames until confirmed cleanup retry.")

# ==============================================================================
# TEST 12: Registered hotkey with missing backend fails and retains registration
# ==============================================================================
print("\n[VERIFICATION 12] Registered hotkey with missing backend fails & retains registration...")
fake_kb_12 = MagicMock()
hotkey_12 = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
hotkey_12.keyboard = fake_kb_12
hotkey_12.register()
assert hotkey_12.is_registered is True

hotkey_12.keyboard = None
try:
    hotkey_12.unregister()
    assert False, "Expected unregister to raise when backend is missing"
except RuntimeError as e:
    pass

assert hotkey_12.is_registered is True, "Must retain registration when backend missing"

hotkey_12.keyboard = fake_kb_12
hotkey_12.unregister()
assert hotkey_12.is_registered is False
print("  [PASSED] Hotkey unregister fails closed on missing backend and retries cleanly.")

print("\n" + "=" * 70)
print("ALL POST-FIX SAFETY CONTRACT VERIFICATIONS PASSED (12/12)!")
print("=" * 70)
