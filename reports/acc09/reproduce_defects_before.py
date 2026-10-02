"""
Reproduce pre-fix safety defects (P1 A, B, C, D) against saved original modules in reports/acc09/before/.
This script proves the defects exist on the pre-fix code without modifying or replacing any live modules.
"""
import importlib.util
from importlib.machinery import SourceFileLoader
import os
import sys
sys.path.insert(0, os.path.abspath("."))
from unittest.mock import MagicMock, patch

print("=" * 70)
print("ACC-09 DEFECT REPRODUCTION ON SAVED ORIGINAL CODE (BEFORE EDITS)")
print("=" * 70)

# Helper to load modules from reports/acc09/before/
before_dir = os.path.abspath("reports/acc09/before")

def load_before_module(name, filename):
    path = os.path.join(before_dir, filename)
    assert os.path.isfile(path), f"Missing {path}"
    loader = SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

failsafe_before = load_before_module("failsafe_before", "failsafe.py")
windows_before = load_before_module("windows_before", "windows.py")
fsm_before = load_before_module("fsm_before", "fsm.py")
pipeline_before = load_before_module("pipeline_before", "pipeline.py")

# ==============================================================================
# DEFECT A: Stop callback inside rate_limiter.allow after corner check yields
# stopped=True, FSM=PAUSED, but MOVE emitted afterward.
# ==============================================================================
print("\n[TEST A] Reproducing Defect A: Stop in rate_limiter.allow emits MOVE afterward...")
from src.gesture.pose import HandPose
from src.platform.dry_run import DryRunController
from src.platform.interface import MouseEvent
from src.config.loader import load_config
import numpy as np

config = load_config()
ctrl_a = DryRunController()
fsm_a = fsm_before.GestureFSM(ctrl_a, config.get("gesture", {}))

# Enter MOVING state
fsm_a.state = fsm_before.FSMState.MOVING
fsm_a._reacquire_count = 5
fsm_a._unknown_count = 0

from src.motion.cursor import CursorMapper
from src.safety.failsafe import FailsafeMonitor, RateLimiter

cursor_mapper_a = CursorMapper()
failsafe_mon_a = FailsafeMonitor(1920, 1080, 10)
rate_limiter_a = RateLimiter(60)

class AlwaysPointOneClassifier:
    def predict_with_rejection(self, canon_flat, decision_threshold=0.0, margin_threshold=0.15, unknown_label="UNKNOWN"):
        return ["POINT_ONE"]

pipe_a = pipeline_before.AirPadPipeline(
    config=config,
    controller=ctrl_a,
    classifier=AlwaysPointOneClassifier(),
    fsm=fsm_a,
    cursor_mapper=cursor_mapper_a,
    failsafe_monitor=failsafe_mon_a,
    rate_limiter=rate_limiter_a,
)

# Intercept rate_limiter.allow to trigger emergency stop mid-flight
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
assert fsm_a.state == fsm_before.FSMState.PAUSED
# On original code, MOVE WAS EMITTED!
assert len(move_events_a) == 1, "DEFECT A REPRODUCED: MOVE was emitted despite emergency stop!"
assert res_a.action_fired == "MOVE"
print("  [CONFIRMED] Defect A reproduced on before-code: MOVE event leaked after emergency stop.")

# ==============================================================================
# DEFECT B: Mock mouseDown calls trigger_emergency_stop during drag start:
# cleanup runs before bookkeeping, leaving stopped=True, PAUSED, _is_mouse_down=True,
# and held_buttons={'left'}.
# ==============================================================================
print("\n[TEST B] Reproducing Defect B: Stop during drag start leaves held left button...")
fake_pyautogui = MagicMock()

with patch("src.platform.windows.WindowsController.__init__", return_value=None):
    ctrl_b = windows_before.WindowsController(allow_real_control=True)
    ctrl_b._held_buttons = set()
    ctrl_b._held_keys = set()
    ctrl_b.pyautogui = fake_pyautogui

fsm_b = fsm_before.GestureFSM(ctrl_b, config.get("gesture", {}))

# Enter MOVING state
fsm_b.state = fsm_before.FSMState.MOVING
fsm_b._reacquire_count = 5
fsm_b._unknown_count = 0

# Dummy pipeline to connect stop
class DummyPipelineB:
    def __init__(self, fsm, ctrl):
        self.fsm = fsm
        self.controller = ctrl
        self.is_emergency_stopped = False
    def trigger_emergency_stop(self):
        self.is_emergency_stopped = True
        self.fsm.on_emergency_stop()
        self.controller.cleanup()

pipe_b = DummyPipelineB(fsm_b, ctrl_b)

def stop_inside_mouse_down(button='left'):
    pipe_b.trigger_emergency_stop()

fake_pyautogui.mouseDown.side_effect = stop_inside_mouse_down

# Frame 1: PINCH_CLOSED initiates click pending
fsm_b.update(HandPose.PINCH_CLOSED, 1000.0)
# Frame 2: PINCH_CLOSED held past drag_threshold_ms (400ms) triggers drag start
fsm_b.update(HandPose.PINCH_CLOSED, 1450.0)

print(f"  -> pipe_b.is_emergency_stopped = {pipe_b.is_emergency_stopped}")
print(f"  -> fsm_b.state = {fsm_b.state.name}")
print(f"  -> fsm_b.is_mouse_down = {fsm_b.is_mouse_down}")
print(f"  -> ctrl_b.held_buttons = {ctrl_b._held_buttons}")
assert pipe_b.is_emergency_stopped is True
assert fsm_b.state == fsm_before.FSMState.PAUSED
# On original code, button was added AFTER cleanup, leaving left held and _is_mouse_down=True!
assert fsm_b.is_mouse_down is True, "DEFECT B REPRODUCED: fsm._is_mouse_down is True after stop!"
assert ctrl_b._held_buttons == {"left"}, "DEFECT B REPRODUCED: controller._held_buttons contains 'left' after stop!"
print("  [CONFIRMED] Defect B reproduced on before-code: left button remains held after emergency stop.")

# ==============================================================================
# DEFECT C: failing mouseUp during cleanup discards left anyway; second cleanup
# never retries. Ctrl/Alt in zoom/swipe bypass key tracking.
# ==============================================================================
print("\n[TEST C] Reproducing Defect C: Failing mouseUp discarded anyway; modifier keys untracked...")
ctrl_c = windows_before.WindowsController(allow_real_control=True)
fake_pyautogui_c = MagicMock()
ctrl_c.pyautogui = fake_pyautogui_c

ctrl_c._held_buttons.add("left")
fake_pyautogui_c.mouseUp.side_effect = RuntimeError("Windows OS mouseUp fault")

# Call cleanup once: fails
ctrl_c.cleanup()
print(f"  -> ctrl_c._held_buttons after failed cleanup = {ctrl_c._held_buttons}")
# On original code, finally discarded 'left' despite failure!
assert len(ctrl_c._held_buttons) == 0, "DEFECT C REPRODUCED: _held_buttons discarded failed release!"

# Call cleanup second time: no attempt to retry because _held_buttons was discarded
fake_pyautogui_c.mouseUp.reset_mock()
ctrl_c.cleanup()
assert fake_pyautogui_c.mouseUp.call_count == 0, "DEFECT C REPRODUCED: Second cleanup did not retry!"
print("  [CONFIRMED] Defect C1 reproduced on before-code: Failed mouseUp discarded; retry missed.")

# Untracked zoom Ctrl key:
ctrl_c_zoom = windows_before.WindowsController(allow_real_control=True)
fake_pyautogui_zoom = MagicMock()
ctrl_c_zoom.pyautogui = fake_pyautogui_zoom

def inspect_scroll_keys(steps):
    # During scroll, check if ctrl is in _held_keys
    print(f"  -> During zoom scroll, ctrl in _held_keys = {'ctrl' in ctrl_c_zoom._held_keys}")
    assert 'ctrl' not in ctrl_c_zoom._held_keys, "DEFECT C2 REPRODUCED: 'ctrl' bypassed _held_keys tracking!"

fake_pyautogui_zoom.scroll.side_effect = inspect_scroll_keys
ctrl_c_zoom.zoom(5)
print("  [CONFIRMED] Defect C2 reproduced on before-code: Zoom Ctrl key bypassed _held_keys tracking.")

# ==============================================================================
# DEFECT D: failed emergency hotkey registration still admits real-control pipeline.
# ==============================================================================
print("\n[TEST D] Reproducing Defect D: Failed hotkey registration admits real-control pipeline...")
with patch.dict("sys.modules", {"keyboard": None}):
    hk_d = failsafe_before.EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
    # On original code, missing keyboard set dry_run=True and did not raise!
    assert hk_d.dry_run is True
    hk_d.register()
    assert hk_d._registered is False

# Failing add_hotkey
fake_keyboard = MagicMock()
fake_keyboard.add_hotkey.side_effect = RuntimeError("Hook conflict / privilege error")

hk_d2 = failsafe_before.EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=False)
hk_d2.keyboard = fake_keyboard
hk_d2.register()  # Swallows error and leaves _registered = False
assert hk_d2._registered is False

# In pipeline_before with real controller and failing hotkey:
ctrl_d = windows_before.WindowsController(allow_real_control=True)
pipe_d = pipeline_before.AirPadPipeline(
    config=config,
    controller=ctrl_d,
    emergency_hotkey=hk_d2,
)
print(f"  -> pipe_d created with WindowsController: {type(pipe_d.controller).__name__}")
print(f"  -> hotkey registered: {pipe_d.emergency_hotkey._registered}")
print(f"  -> pipeline emergency_stopped: {pipe_d.is_emergency_stopped}")
assert pipe_d.emergency_hotkey._registered is False
assert pipe_d.is_emergency_stopped is False
assert isinstance(pipe_d.controller, windows_before.WindowsController)
print("  [CONFIRMED] Defect D reproduced on before-code: Real-control pipeline admitted without working emergency hotkey!")

print("\n" + "=" * 70)
print("ALL 4 PRE-FIX DEFECTS POSITIVELY REPRODUCED ON SAVED ORIGINAL CODE")
print("=" * 70)
