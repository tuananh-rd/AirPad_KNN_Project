"""Label-Controlled Dynamic Synthetic Intent Benchmark (HYB-01 - v4).

Benchmarks dynamic user intent scenarios comparing saved-original Legacy profile
against Hybrid Assist profile:
1. Slow continuous 2-finger scroll: 100 frames with dy = 0.002 (total 0.200).
   - Hash-pinned original engine from reports/hyb01/before/src/motion/scroll.py:
     discards sub-deadzone increments (0.002 < 0.005), emitting 0 scroll units.
   - Hybrid engine: accumulates sub-deadband increments, emitting 23 units.
   - Discloses idealized Majority(1)/Stability(1) unit test configuration.
2. Stationary 2-finger jitter: 100 frames with dy in [-0.0015, 0.0015].
   - Asserts total scroll event count == 0 and absolute scroll units == 0.
3. 3-finger tab switching with neutral return & cooldown:
   - Left swipe at 150ms -> TAB_PREV (Ctrl+Shift+Tab).
   - Neutral return at 600ms (crossing 400ms cooldown, 600 - 150 = 450ms > 400ms)
     and neutral hold at 630ms, followed by right overshoot at 750ms WHILE STILL
     holding 3 fingers: Hybrid suppresses rearm (asserts 0 pre-exit actions).
   - Observed pose exit to OTHER at 800ms -> Re-entry as 3 fingers at 850ms.
   - Right swipe at 1000ms -> TAB_NEXT (Ctrl+Tab).
4. Assist OFF transition:
   - Starts ON (READY), transitions to OFF under default temporal filters
     (Majority 5, Stability 2), processes raw/filtered transitions, emitting 0 events.
5. Dynamic forbidden action count computation with ordered key tracking:
   - Tracks held keys across ordered events to accurately detect Ctrl+Wheel (ZOOM)
     and Alt+Left/Right (HISTORY navigation).
   - Flags all non-whitelisted types/keys.
   - Verified via negative controls (injected Ctrl+SCROLL, Alt+Left, MOVE, Alt+Tab, hscroll, Alt-alone).
6. Exclusive artifact creation:
   - Default output: reports/hyb01/hybrid_synthetic_benchmark_results_v4.json
   - Written exclusively using open(path, 'x').
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Ensure mocked PyAutoGUI and keyboard before imports
mock_pyautogui = MagicMock()
mock_pyautogui.size.return_value = (1920, 1080)
mock_keyboard = MagicMock()
sys.modules['pyautogui'] = mock_pyautogui
sys.modules['keyboard'] = mock_keyboard

from src.gesture.pose import HandPose
from src.gesture.temporal import MajorityFilter, StabilityGate
from src.motion.scroll import ScrollEngine as HybridScrollEngine
from src.motion.swipe import SwipeEngine
from src.platform.dry_run import DryRunController
from src.platform.interface import MouseEvent, KeyEvent
from src.runtime.pipeline import AirPadPipeline
from src.safety.failsafe import SafetyGate

# Pin original scroll engine path and hash
ORIGINAL_SCROLL_PATH = "reports/hyb01/before/src/motion/scroll.py"
EXPECTED_ORIGINAL_SCROLL_SHA256 = "EEE878AD3DD049E975EC2C1475854C1BFDE31873E14F7C175A22B6351DD9C5EC"


def verify_sha256(path: str, expected_hash: str) -> None:
    assert os.path.exists(path), f"File not found: {path}"
    with open(path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest().upper()
    assert actual_hash == expected_hash, (
        f"Cryptographic hash mismatch for {path}!\n"
        f"  Expected: {expected_hash}\n"
        f"  Got:      {actual_hash}"
    )


def load_original_scroll_engine():
    verify_sha256(ORIGINAL_SCROLL_PATH, EXPECTED_ORIGINAL_SCROLL_SHA256)
    spec = importlib.util.spec_from_file_location("original_scroll_module", ORIGINAL_SCROLL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ScrollEngine


class MockClassifier:
    """Deterministic label classifier."""
    def __init__(self, label: str):
        self.label = label

    def predict_with_rejection(self, *args, **kwargs):
        return [self.label]


def make_landmarks(index_y: float = 0.5, middle_y: float = 0.5, wrist_x: float = 0.5) -> np.ndarray:
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = [wrist_x, 0.8, 0.0]    # wrist
    lms[8] = [0.5, index_y, 0.0]   # index tip
    lms[12] = [0.5, middle_y, 0.0] # middle tip
    return lms


def compute_forbidden_counts(events: List[Dict[str, Any]], actions: List[str]) -> Dict[str, int]:
    """Dynamically compute forbidden counts by tracking ordered event stream and key state.

    Ordered-state auditor (shared logic with run_hybrid_benchmark.py):
    - Tracks held modifier keys across ordered KEY_DOWN / KEY_UP events.
    - MOVE, CLICK, DRAG always forbidden in Hybrid.
    - SCROLL: Ctrl+wheel -> forbidden_zooms; Alt+SCROLL -> forbidden_history;
      horizontal scroll (dx != 0) -> forbidden_other; plain vertical scroll whitelisted.
    - KEY_DOWN/KEY_UP: Alt+Left/Right -> forbidden_history; Alt+Tab -> forbidden_other
      (window switch); ctrl/shift/tab alone whitelisted; Alt key alone flagged as
      forbidden_other (menu focus) while tracked for combinations;
      any other key -> forbidden_other.
    - LEFT_UP / mouseUp: always forbidden in Hybrid (including cleanup-marked, since
      Hybrid never generates mouse button presses so any mouseUp is spurious).
    - Unknown event types -> forbidden_other.
    """
    forbidden_moves = 0
    forbidden_clicks = 0
    forbidden_drags = 0
    forbidden_zooms = 0
    forbidden_history = 0
    forbidden_other = 0

    tracked_held_keys = set()

    for e in events:
        etype = e.get("type")
        etype_str = str(etype.name if hasattr(etype, "name") else etype)
        key = str(e.get("key", "")).lower()

        # Track held modifier keys across ordered events
        if etype in (KeyEvent.KEY_DOWN, "KEY_DOWN"):
            tracked_held_keys.add(key)
        elif etype in (KeyEvent.KEY_UP, "KEY_UP"):
            tracked_held_keys.discard(key)

        # 1. MOVE
        if etype in (MouseEvent.MOVE, "MOVE") or "MOVE" in etype_str:
            forbidden_moves += 1

        # 2. CLICK
        elif etype in (MouseEvent.LEFT_CLICK, MouseEvent.RIGHT_CLICK, "LEFT_CLICK", "RIGHT_CLICK"):
            forbidden_clicks += 1

        # 3. DRAG (LEFT_DOWN always forbidden; LEFT_UP also always forbidden in
        #    Hybrid since Hybrid never generates mouse button presses, so any
        #    mouseUp/LEFT_UP—including cleanup-marked—is spurious)
        elif etype in (MouseEvent.LEFT_DOWN, "LEFT_DOWN"):
            forbidden_drags += 1
        elif etype in (MouseEvent.LEFT_UP, "LEFT_UP"):
            forbidden_drags += 1

        # 4. SCROLL: check for zoom (Ctrl+wheel), history (Alt+scroll),
        #    horizontal scroll (dx != 0), or whitelisted vertical scroll
        elif etype in (MouseEvent.SCROLL, "SCROLL"):
            dx = e.get("dx", 0)
            if "ctrl" in tracked_held_keys:
                # Ctrl + Wheel is zoom
                forbidden_zooms += 1
            elif "alt" in tracked_held_keys:
                forbidden_history += 1
            elif dx != 0:
                # Horizontal scroll is not whitelisted in Hybrid
                forbidden_other += 1
            # else: normal whitelisted vertical scroll (without ctrl/alt, dx==0)

        # 5. KEY_DOWN / KEY_UP / KEY_PRESS: detect forbidden key combinations
        elif etype in (KeyEvent.KEY_DOWN, KeyEvent.KEY_UP, "KEY_DOWN", "KEY_UP", "KEY_PRESS"):
            # Alt+Left/Right = history navigation
            if "alt" in tracked_held_keys and key in ("left", "right"):
                if etype in (KeyEvent.KEY_DOWN, "KEY_DOWN", "KEY_PRESS"):
                    forbidden_history += 1
            # Alt+Tab = window switch (forbidden)
            elif "alt" in tracked_held_keys and key == "tab":
                if etype in (KeyEvent.KEY_DOWN, "KEY_DOWN", "KEY_PRESS"):
                    forbidden_other += 1
            # Whitelisted keys for Hybrid assist: ctrl, shift, tab (for Ctrl+Tab / Ctrl+Shift+Tab)
            elif key in ("ctrl", "shift", "tab"):
                pass
            # Alt key: still tracked in tracked_held_keys above for Alt+Tab /
            # Alt+Left/Right combination detection, but the key event itself
            # is forbidden (Alt alone can focus OS menu bars)
            elif key == "alt":
                forbidden_other += 1
            # Any other key is forbidden
            else:
                forbidden_other += 1

        else:
            # Unknown event type
            forbidden_other += 1

    # Check actions trace for forbidden action names
    for act in actions:
        if act == "MOVE":
            forbidden_moves += 1
        elif act in ("LEFT_CLICK", "RIGHT_CLICK"):
            forbidden_clicks += 1
        elif act in ("DRAG", "LEFT_DOWN", "LEFT_UP"):
            forbidden_drags += 1
        elif act == "ZOOM":
            forbidden_zooms += 1
        elif act in ("SWIPE_BACK", "SWIPE_FORWARD"):
            forbidden_history += 1
        elif act in ("SCROLL", "TAB_NEXT", "TAB_PREV"):
            pass  # whitelisted
        else:
            # Unknown action
            forbidden_other += 1

    return {
        "forbidden_moves": forbidden_moves,
        "forbidden_clicks": forbidden_clicks,
        "forbidden_drags": forbidden_drags,
        "forbidden_zooms": forbidden_zooms,
        "forbidden_history": forbidden_history,
        "forbidden_other": forbidden_other,
    }


def verify_negative_controls() -> Dict[str, Any]:
    """Verify that compute_forbidden_counts positively catches injected violations."""
    # Negative control 1: Injected Ctrl + Scroll (Zoom)
    test_zoom_events = [
        {"type": KeyEvent.KEY_DOWN, "key": "ctrl"},
        {"type": MouseEvent.SCROLL, "dx": 0, "dy": 5},
        {"type": KeyEvent.KEY_UP, "key": "ctrl"},
    ]
    zoom_res = compute_forbidden_counts(test_zoom_events, [])
    assert zoom_res["forbidden_zooms"] == 1, f"Failed negative control: Ctrl+Scroll not detected! Got: {zoom_res}"

    # Negative control 2: Injected Alt + Left (History navigation)
    test_hist_events = [
        {"type": KeyEvent.KEY_DOWN, "key": "alt"},
        {"type": KeyEvent.KEY_DOWN, "key": "left"},
        {"type": KeyEvent.KEY_UP, "key": "left"},
        {"type": KeyEvent.KEY_UP, "key": "alt"},
    ]
    hist_res = compute_forbidden_counts(test_hist_events, [])
    assert hist_res["forbidden_history"] == 1, f"Failed negative control: Alt+Left not detected! Got: {hist_res}"

    # Negative control 3: Injected non-whitelisted MOVE
    test_move_events = [
        {"type": MouseEvent.MOVE, "x": 500, "y": 500},
    ]
    move_res = compute_forbidden_counts(test_move_events, [])
    assert move_res["forbidden_moves"] == 1, f"Failed negative control: MOVE not detected! Got: {move_res}"
    # Negative control 4: Injected Alt+Tab (window switch -> forbidden_other)
    #    Alt down (1) + Alt+Tab down (2) + Alt up (3) = 3 forbidden_other
    test_alt_tab_events = [
        {"type": KeyEvent.KEY_DOWN, "key": "alt"},
        {"type": KeyEvent.KEY_DOWN, "key": "tab"},
        {"type": KeyEvent.KEY_UP, "key": "tab"},
        {"type": KeyEvent.KEY_UP, "key": "alt"},
    ]
    alt_tab_res = compute_forbidden_counts(test_alt_tab_events, [])
    assert alt_tab_res["forbidden_other"] == 3, f"Failed negative control: Alt+Tab expected forbidden_other=3! Got: {alt_tab_res}"

    # Negative control 5: Injected horizontal scroll (dx != 0 -> forbidden_other)
    test_hscroll_events = [
        {"type": MouseEvent.SCROLL, "dx": 3, "dy": 0},
    ]
    hscroll_res = compute_forbidden_counts(test_hscroll_events, [])
    assert hscroll_res["forbidden_other"] == 1, f"Failed negative control: Horizontal scroll not detected! Got: {hscroll_res}"

    # Negative control 6: Injected Alt-alone (menu focus -> forbidden_other)
    test_alt_alone_events = [
        {"type": KeyEvent.KEY_DOWN, "key": "alt"},
        {"type": KeyEvent.KEY_UP, "key": "alt"},
    ]
    alt_alone_res = compute_forbidden_counts(test_alt_alone_events, [])
    assert alt_alone_res["forbidden_other"] == 2, f"Failed negative control: Alt-alone expected forbidden_other=2! Got: {alt_alone_res}"

    return {
        "injected_zoom_detected": zoom_res["forbidden_zooms"] == 1,
        "injected_history_detected": hist_res["forbidden_history"] == 1,
        "injected_move_detected": move_res["forbidden_moves"] == 1,
        "injected_alt_tab_detected": alt_tab_res["forbidden_other"] == 3,
        "injected_horizontal_scroll_detected": hscroll_res["forbidden_other"] == 1,
        "injected_alt_alone_detected": alt_alone_res["forbidden_other"] == 2,
    }


def benchmark_slow_scroll() -> Dict[str, Any]:
    """100 frames with dy = 0.002 per frame (total dy = 0.200)."""
    # 1. Saved-Original Legacy ScrollEngine
    OriginalScrollEngine = load_original_scroll_engine()
    orig_engine = OriginalScrollEngine(dead_zone=0.005, gain=120.0)
    orig_steps = []
    for i in range(1, 101):
        t_ms = i * 33.3
        steps = orig_engine.update(0.0, 0.002, t_ms)
        orig_steps.append(steps)
    orig_total_scroll = sum(s[1] for s in orig_steps)

    # 2. New Hybrid ScrollEngine standalone
    hybrid_engine = HybridScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid")
    hybrid_steps = []
    for i in range(1, 101):
        t_ms = i * 33.3
        steps = hybrid_engine.update(0.0, 0.002, t_ms)
        hybrid_steps.append(steps)
    hybrid_total_scroll = sum(s[1] for s in hybrid_steps)

    # 3. New Hybrid AirPadPipeline
    gate = SafetyGate()
    ctrl = DryRunController()
    clf = MockClassifier("TWO_FINGERS_PARALLEL")
    pipe = AirPadPipeline(
        config={},
        controller=ctrl,
        classifier=clf,
        scroll_engine=HybridScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid"),
        majority_filter=MajorityFilter(window_size=1, majority_ratio=1.0),
        stability_gate=StabilityGate(required_count=1),
        gate=gate,
        profile="hybrid",
    )
    pipe.toggle_assist()

    pipeline_actions = []
    try:
        # Initial baseline frame at cy = 0.600
        lms0 = make_landmarks(index_y=0.600, middle_y=0.600)
        pipe.process_landmarks(lms0, timestamp_ms=0.0)

        # 100 frames moving UP slowly (cy decreases by 0.002 each frame)
        for i in range(1, 101):
            cy = 0.600 - (i * 0.002)
            lms = make_landmarks(index_y=cy, middle_y=cy)
            t_ms = i * 33.3
            res = pipe.process_landmarks(lms, timestamp_ms=t_ms)
            if res.action_fired:
                pipeline_actions.append(res.action_fired)
    finally:
        pipe.cleanup()

    events = list(ctrl.event_log)
    pipe_scroll_units = sum(int(e.get("dy", 0)) for e in events if e["type"] == MouseEvent.SCROLL)
    forbidden = compute_forbidden_counts(events, pipeline_actions)

    return {
        "frames": 100,
        "per_frame_dy": 0.002,
        "total_motion_delta": 0.200,
        "saved_original_engine": {
            "path": ORIGINAL_SCROLL_PATH,
            "sha256": EXPECTED_ORIGINAL_SCROLL_SHA256,
            "scroll_units_emitted": orig_total_scroll,
            "deadzone_dropping_confirmed": orig_total_scroll == 0,
        },
        "hybrid_standalone_engine": {
            "scroll_units_emitted": hybrid_total_scroll,
            "accumulation_success": hybrid_total_scroll == 23,
        },
        "hybrid_pipeline": {
            "scroll_units_emitted": pipe_scroll_units,
            "actions_fired": dict(Counter(pipeline_actions)),
            "forbidden_counts": forbidden,
            "held_buttons_clean": len(ctrl.held_buttons) == 0,
            "held_keys_clean": len(ctrl.held_keys) == 0,
        },
        "disclosure": "Idealized Majority(1)/Stability(1) filter configuration used to isolate mathematical accumulator behavior on exact motion vectors. Does not represent production filters (Majority 5, Stability 2) or human hand tremor.",
    }


def benchmark_stationary_jitter() -> Dict[str, Any]:
    """100 frames with jitter alternating +-0.0015 around cy = 0.500."""
    gate = SafetyGate()
    ctrl = DryRunController()
    scroll_eng = HybridScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid")
    clf = MockClassifier("TWO_FINGERS_PARALLEL")
    pipe = AirPadPipeline(
        config={},
        controller=ctrl,
        classifier=clf,
        scroll_engine=scroll_eng,
        majority_filter=MajorityFilter(window_size=1, majority_ratio=1.0),
        stability_gate=StabilityGate(required_count=1),
        gate=gate,
        profile="hybrid",
    )
    pipe.toggle_assist()

    actions = []
    try:
        # Baseline
        lms0 = make_landmarks(index_y=0.500, middle_y=0.500)
        pipe.process_landmarks(lms0, timestamp_ms=0.0)

        # 100 jitter frames
        for i in range(1, 101):
            jitter = 0.0015 if (i % 2 == 0) else -0.0015
            cy = 0.500 + jitter
            lms = make_landmarks(index_y=cy, middle_y=cy)
            t_ms = i * 33.3
            res = pipe.process_landmarks(lms, timestamp_ms=t_ms)
            if res.action_fired:
                actions.append(res.action_fired)
    finally:
        pipe.cleanup()

    events = list(ctrl.event_log)
    total_events = len(events)
    abs_scroll_units = sum(abs(int(e.get("dy", 0))) for e in events if e["type"] == MouseEvent.SCROLL)
    net_scroll_units = sum(int(e.get("dy", 0)) for e in events if e["type"] == MouseEvent.SCROLL)
    forbidden = compute_forbidden_counts(events, actions)

    return {
        "frames": 100,
        "jitter_amplitude": 0.0015,
        "total_events": total_events,
        "absolute_scroll_units": abs_scroll_units,
        "net_scroll_units": net_scroll_units,
        "jitter_suppressed": total_events == 0 and abs_scroll_units == 0,
        "actions_fired": dict(Counter(actions)),
        "forbidden_counts": forbidden,
        "held_buttons_clean": len(ctrl.held_buttons) == 0,
        "held_keys_clean": len(ctrl.held_keys) == 0,
    }


def benchmark_tab_switch_neutral_rearm() -> Dict[str, Any]:
    """Test tab switching rearm contract:

    PREV150 -> neutral600 -> neutral630 -> overshoot750 -> exit800 -> reentry850 -> NEXT1000.
    Neutral return occurs AFTER 400ms cooldown (600 - 150 = 450ms > 400ms) and overshoots at 750ms
    WHILE STILL holding THREE_FINGERS: asserts 0 actions fired pre-exit.
    Rearms only after observed pose exit at 800ms and re-entry at 850ms, allowing NEXT at 1000ms.
    """
    gate = SafetyGate()
    ctrl = DryRunController()
    swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
    clf = MockClassifier("THREE_FINGERS")
    pipe = AirPadPipeline(
        config={},
        controller=ctrl,
        classifier=clf,
        swipe_engine=swipe_eng,
        majority_filter=MajorityFilter(window_size=1, majority_ratio=1.0),
        stability_gate=StabilityGate(required_count=1),
        gate=gate,
        profile="hybrid",
    )
    pipe.toggle_assist()

    all_actions: List[str] = []
    pre_exit_actions: List[Dict[str, Any]] = []

    try:
        # Phase 0: Init at neutral wrist_x = 0.50 at t = 0.0ms
        pipe.process_landmarks(make_landmarks(wrist_x=0.50), timestamp_ms=0.0)

        # Phase 1: Left swipe (wrist moves 0.50 -> 0.38 at 150ms) -> fires TAB_PREV
        r150 = pipe.process_landmarks(make_landmarks(wrist_x=0.38), timestamp_ms=150.0)
        if r150.action_fired:
            all_actions.append(r150.action_fired)

        # Phase 2: Neutral return to 0.50 at 600ms (crossing 400ms cooldown: 600 - 150 = 450ms > 400ms)
        # WHILE STILL holding THREE_FINGERS! Must NOT rearm!
        r600 = pipe.process_landmarks(make_landmarks(wrist_x=0.50), timestamp_ms=600.0)
        if r600.action_fired:
            all_actions.append(r600.action_fired)
            pre_exit_actions.append({"t_ms": 600.0, "action": r600.action_fired})

        # Phase 3: Neutral hold at 630ms
        r630 = pipe.process_landmarks(make_landmarks(wrist_x=0.50), timestamp_ms=630.0)
        if r630.action_fired:
            all_actions.append(r630.action_fired)
            pre_exit_actions.append({"t_ms": 630.0, "action": r630.action_fired})

        # Phase 4: Overshoot to 0.62 at 750ms WHILE STILL holding THREE_FINGERS!
        r750 = pipe.process_landmarks(make_landmarks(wrist_x=0.62), timestamp_ms=750.0)
        if r750.action_fired:
            all_actions.append(r750.action_fired)
            pre_exit_actions.append({"t_ms": 750.0, "action": r750.action_fired})

        # Phase 5: Observed pose exit to OTHER at 800ms
        clf.label = "OTHER"
        r800 = pipe.process_landmarks(make_landmarks(wrist_x=0.50), timestamp_ms=800.0)
        if r800.action_fired:
            all_actions.append(r800.action_fired)

        # Phase 6: Re-entry as THREE_FINGERS at neutral 0.50 at 850ms
        clf.label = "THREE_FINGERS"
        r850 = pipe.process_landmarks(make_landmarks(wrist_x=0.50), timestamp_ms=850.0)
        if r850.action_fired:
            all_actions.append(r850.action_fired)

        # Phase 7: Right swipe (wrist moves 0.50 -> 0.62 at 1000ms) -> fires TAB_NEXT
        r1000 = pipe.process_landmarks(make_landmarks(wrist_x=0.62), timestamp_ms=1000.0)
        if r1000.action_fired:
            all_actions.append(r1000.action_fired)

    finally:
        pipe.cleanup()

    events = list(ctrl.event_log)
    tab_events = [e for e in events if e.get("type") in (KeyEvent.KEY_DOWN, "KEY_DOWN") and e.get("key") == "tab"]
    forbidden = compute_forbidden_counts(events, all_actions)

    rearm_suppressed = len(pre_exit_actions) == 0 and all_actions == ["TAB_PREV", "TAB_NEXT"]

    return {
        "timeline_ms": [0.0, 150.0, 600.0, 630.0, 750.0, 800.0, 850.0, 1000.0],
        "all_actions": all_actions,
        "pre_exit_actions_during_neutral_and_overshoot": pre_exit_actions,
        "actions_fired": dict(Counter(all_actions)),
        "tab_key_events": len(tab_events),
        "cooldown_crossed_while_still_three_rearm_suppressed": rearm_suppressed,
        "forbidden_counts": forbidden,
        "held_buttons_clean": len(ctrl.held_buttons) == 0,
        "held_keys_clean": len(ctrl.held_keys) == 0,
    }


def benchmark_assist_off_transition() -> Dict[str, Any]:
    """Assist transitions ON -> OFF under default temporal filters; verifies 0 actions during raw/filtered transition."""
    gate = SafetyGate()
    ctrl = DryRunController()
    scroll_eng = HybridScrollEngine(dead_zone=0.005, gain=120.0, mode="hybrid")
    swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
    clf = MockClassifier("TWO_FINGERS_PARALLEL")
    pipe = AirPadPipeline(
        config={},
        controller=ctrl,
        classifier=clf,
        scroll_engine=scroll_eng,
        swipe_engine=swipe_eng,
        majority_filter=MajorityFilter(window_size=5, majority_ratio=0.6),
        stability_gate=StabilityGate(required_count=2),
        gate=gate,
        profile="hybrid",
    )
    # 1. Start ON (toggle OFF -> READY)
    pipe.toggle_assist()
    assert pipe.assist_state == "READY"

    # Warm up 5 frames in ON state
    for i in range(5):
        lms_warm = make_landmarks(index_y=0.50, middle_y=0.50)
        pipe.process_landmarks(lms_warm, timestamp_ms=i * 33.3)

    # 2. Toggle READY -> OFF (Actual ON -> OFF transition)
    pipe.toggle_assist()
    assert pipe.assist_state == "OFF"

    actions = []
    try:
        # 20 frames of raw TWO_FINGERS moving up
        for i in range(5, 25):
            lms = make_landmarks(index_y=0.50 - (i - 5) * 0.01, middle_y=0.50 - (i - 5) * 0.01)
            res = pipe.process_landmarks(lms, timestamp_ms=i * 33.3)
            if res.action_fired:
                actions.append(res.action_fired)

        # 10 transition frames switching raw pose from TWO to THREE via OTHER
        for i in range(25, 35):
            clf.label = "OTHER" if i < 30 else "THREE_FINGERS"
            lms = make_landmarks(wrist_x=0.50)
            res = pipe.process_landmarks(lms, timestamp_ms=i * 33.3)
            if res.action_fired:
                actions.append(res.action_fired)

        # 15 frames of THREE_FINGERS swipe attempt
        clf.label = "THREE_FINGERS"
        for i in range(35, 50):
            lms = make_landmarks(wrist_x=0.65)
            res = pipe.process_landmarks(lms, timestamp_ms=i * 33.3)
            if res.action_fired:
                actions.append(res.action_fired)
    finally:
        pipe.cleanup()

    events = list(ctrl.event_log)
    forbidden = compute_forbidden_counts(events, actions)

    return {
        "initial_state": "READY",
        "transition_to": "OFF",
        "temporal_filters": "Majority(5, 0.6) + Stability(2)",
        "total_frames_in_off": 45,
        "total_events": len(events),
        "actions_fired": dict(Counter(actions)),
        "forbidden_counts": forbidden,
        "held_buttons_clean": len(ctrl.held_buttons) == 0,
        "held_keys_clean": len(ctrl.held_keys) == 0,
    }


def benchmark_warm_three_then_swipe() -> Dict[str, Any]:
    """Tests that holding THREE_FINGERS stationary for 2000ms does NOT dilute swipe velocity:

    50 frames of stationary hold (t = 0..1960ms, x = 0.50), then deliberate swipe (t = 2000..2160ms, x = 0.50 -> 0.62).
    Under default temporal filters (Majority 5, Stability 2), must emit exactly 1 TAB_NEXT event.
    """
    gate = SafetyGate()
    ctrl = DryRunController()
    swipe_eng = SwipeEngine(min_displacement=0.08, min_velocity=0.0003, cooldown_ms=400.0, mode="hybrid")
    clf = MockClassifier("THREE_FINGERS")
    pipe = AirPadPipeline(
        config={},
        controller=ctrl,
        classifier=clf,
        swipe_engine=swipe_eng,
        majority_filter=MajorityFilter(window_size=5, majority_ratio=0.6),
        stability_gate=StabilityGate(required_count=2),
        gate=gate,
        profile="hybrid",
    )
    pipe.toggle_assist()
    assert pipe.assist_state == "READY"

    actions = []
    # 50 frames stationary hold at x=0.50 (2000ms)
    for i in range(50):
        t_ms = i * 40.0
        lms = make_landmarks(wrist_x=0.50)
        res = pipe.process_landmarks(lms, timestamp_ms=t_ms)
        if res.action_fired:
            actions.append(res.action_fired)

    # Deliberate swipe to 0.62 in 160ms
    for i in range(1, 5):
        t_ms = 2000.0 + i * 40.0
        wx = 0.50 + 0.12 * (i / 4.0)
        lms = make_landmarks(wrist_x=wx)
        res = pipe.process_landmarks(lms, timestamp_ms=t_ms)
        if res.action_fired:
            actions.append(res.action_fired)

    pipe.cleanup()

    events = list(ctrl.event_log)
    forbidden = compute_forbidden_counts(events, actions)

    return {
        "warm_hold_duration_ms": 2000.0,
        "warm_hold_frames": 50,
        "swipe_duration_ms": 160.0,
        "swipe_displacement": 0.12,
        "total_actions": len(actions),
        "actions_fired": dict(Counter(actions)),
        "warm_hold_swipe_success": actions == ["TAB_NEXT"],
        "forbidden_counts": forbidden,
        "held_buttons_clean": len(ctrl.held_buttons) == 0,
        "held_keys_clean": len(ctrl.held_keys) == 0,
    }


def run_benchmark(output_path: str = "reports/hyb01/hybrid_synthetic_benchmark_results_v4.json") -> int:
    print(f"Running Label-Controlled Dynamic Synthetic Intent Benchmark (v4)...")
    print(f"Output will be saved exclusively to: {output_path}")

    # 0. Negative controls verification
    neg_control_results = verify_negative_controls()
    print("\n[0] Negative Controls Verification:")
    print(f"  Injected Zoom (Ctrl+Wheel) detected: {neg_control_results['injected_zoom_detected']}")
    print(f"  Injected History (Alt+Left) detected: {neg_control_results['injected_history_detected']}")
    print(f"  Injected MOVE detected: {neg_control_results['injected_move_detected']}")
    print(f"  Injected Alt+Tab (window switch) detected: {neg_control_results['injected_alt_tab_detected']}")
    print(f"  Injected Horizontal SCROLL (dx!=0) detected: {neg_control_results['injected_horizontal_scroll_detected']}")
    print(f"  Injected Alt-alone (menu focus) detected: {neg_control_results['injected_alt_alone_detected']}")

    # 1. Slow scroll comparison against hash-pinned saved-original
    slow_results = benchmark_slow_scroll()
    print(f"\n[1] Slow Continuous Scroll (100 frames, dy = 0.002, total motion = 0.200):")
    print(f"  Saved-Original Engine ({ORIGINAL_SCROLL_PATH}): {slow_results['saved_original_engine']['scroll_units_emitted']} units (CONFIRMED DEADZONE-DROPPING DEFECT)")
    print(f"  Hybrid Standalone Engine: {slow_results['hybrid_standalone_engine']['scroll_units_emitted']} units (ACCUMULATED)")
    print(f"  Hybrid Pipeline Output: {slow_results['hybrid_pipeline']['scroll_units_emitted']} units, forbidden moves: {slow_results['hybrid_pipeline']['forbidden_counts']['forbidden_moves']}")
    assert slow_results["saved_original_engine"]["scroll_units_emitted"] == 0
    assert slow_results["hybrid_standalone_engine"]["scroll_units_emitted"] == 23
    assert slow_results["hybrid_pipeline"]["scroll_units_emitted"] == 23
    assert sum(slow_results["hybrid_pipeline"]["forbidden_counts"].values()) == 0
    assert slow_results["hybrid_pipeline"]["held_buttons_clean"]
    assert slow_results["hybrid_pipeline"]["held_keys_clean"]

    # 2. Stationary jitter
    jitter_results = benchmark_stationary_jitter()
    print(f"\n[2] Stationary Jitter Suppression (100 frames, +-0.0015):")
    print(f"  Total events: {jitter_results['total_events']}, Absolute scroll units: {jitter_results['absolute_scroll_units']}")
    assert jitter_results["total_events"] == 0
    assert jitter_results["absolute_scroll_units"] == 0
    assert sum(jitter_results["forbidden_counts"].values()) == 0
    assert jitter_results["held_buttons_clean"]
    assert jitter_results["held_keys_clean"]

    # 3. Tab switch with neutral return across 400ms cooldown
    tab_results = benchmark_tab_switch_neutral_rearm()
    print(f"\n[3] 3-Finger Tab Switching & Cooldown-Crossed Neutral Return:")
    print(f"  Actions fired: {tab_results['actions_fired']}")
    print(f"  Pre-exit actions during neutral/overshoot: {tab_results['pre_exit_actions_during_neutral_and_overshoot']}")
    print(f"  Rearm suppressed while still 3 fingers: {tab_results['cooldown_crossed_while_still_three_rearm_suppressed']}")
    assert tab_results["actions_fired"].get("TAB_PREV") == 1
    assert tab_results["actions_fired"].get("TAB_NEXT") == 1
    assert sum(tab_results["actions_fired"].values()) == 2
    assert tab_results["cooldown_crossed_while_still_three_rearm_suppressed"] is True
    assert len(tab_results["pre_exit_actions_during_neutral_and_overshoot"]) == 0
    assert sum(tab_results["forbidden_counts"].values()) == 0
    assert tab_results["held_buttons_clean"]
    assert tab_results["held_keys_clean"]

    # 4. Assist OFF transition
    off_results = benchmark_assist_off_transition()
    print(f"\n[4] Assist ON -> OFF Transition with Default Filters:")
    print(f"  Total events in OFF: {off_results['total_events']}")
    assert off_results["total_events"] == 0
    assert sum(off_results["forbidden_counts"].values()) == 0
    assert off_results["held_buttons_clean"]
    assert off_results["held_keys_clean"]

    # 5. Warm THREE_FINGERS hold (2s) then deliberate swipe under default filters
    warm_results = benchmark_warm_three_then_swipe()
    print(f"\n[5] Warm THREE_FINGERS Hold (2s) Then Deliberate Swipe:")
    print(f"  Actions fired: {warm_results['actions_fired']}, Success: {warm_results['warm_hold_swipe_success']}")
    assert warm_results["warm_hold_swipe_success"] is True
    assert sum(warm_results["forbidden_counts"].values()) == 0
    assert warm_results["held_buttons_clean"]
    assert warm_results["held_keys_clean"]

    benchmark_summary = {
        "benchmark_type": "dynamic_synthetic_intent_benchmark_v4",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pinned_baselines": {
            "original_scroll_engine": {
                "path": ORIGINAL_SCROLL_PATH,
                "sha256": EXPECTED_ORIGINAL_SCROLL_SHA256,
            },
        },
        "negative_controls": neg_control_results,
        "slow_continuous_scroll": slow_results,
        "stationary_jitter_suppression": jitter_results,
        "tab_switching_neutral_rearm_contract": tab_results,
        "assist_off_transition": off_results,
        "warm_hold_deliberate_swipe": warm_results,
        "safety_audit": {
            "all_held_buttons_clean": all([
                slow_results["hybrid_pipeline"]["held_buttons_clean"],
                jitter_results["held_buttons_clean"],
                tab_results["held_buttons_clean"],
                off_results["held_buttons_clean"],
                warm_results["held_buttons_clean"],
            ]),
            "all_held_keys_clean": all([
                slow_results["hybrid_pipeline"]["held_keys_clean"],
                jitter_results["held_keys_clean"],
                tab_results["held_keys_clean"],
                off_results["held_keys_clean"],
                warm_results["held_keys_clean"],
            ]),
            "all_forbidden_counts": {
                "slow_scroll": slow_results["hybrid_pipeline"]["forbidden_counts"],
                "jitter": jitter_results["forbidden_counts"],
                "tab": tab_results["forbidden_counts"],
                "off_transition": off_results["forbidden_counts"],
                "warm_hold_swipe": warm_results["forbidden_counts"],
            },
            "all_forbidden_counts_zero": all(
                sum(counts.values()) == 0 for counts in [
                    slow_results["hybrid_pipeline"]["forbidden_counts"],
                    jitter_results["forbidden_counts"],
                    tab_results["forbidden_counts"],
                    off_results["forbidden_counts"],
                    warm_results["forbidden_counts"],
                ]
            ),
        },
    }

    # Exclusive artifact creation fail-closed
    with open(output_path, "x", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\nSynthetic Intent Benchmark successfully created at: {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Synthetic Intent Benchmark (v4)")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/hyb01/hybrid_synthetic_benchmark_results_v4.json",
        help="Path for exclusive creation of output benchmark JSON (default: v4)",
    )
    args = parser.parse_args()
    sys.exit(run_benchmark(output_path=args.output))
