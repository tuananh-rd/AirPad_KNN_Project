"""Headless Action Benchmark for Hybrid Assist Profile (HYB-01).

Compares legacy profile vs hybrid profile over frozen ACC-06/07 shadow dataset
(3,320 frames, trial_shadow_obs_20261001_162213_fb6d49.jsonl).

Verifies invariants:
1. Filter frame records: Exactly 3,320 frame records (excludes header),
   asserts detected == 1875 and absent == 1445.
2. Cryptographic input pin: Verifies SHA-256 of shadow dataset, candidate bundle,
   and frozen ACC-07 dry-run replay JSON before execution.
3. Legacy profile: 100% exact match with frozen ACC-07 dry-run baseline
   (156 events: 151 MOVE, 5 SCROLL), verifying both total counts (counts_match)
   and per-event payload equality (per_event_match).
4. Hybrid profile (OFF state): Exactly 0 controller events across all 3,320 frames.
5. Hybrid profile (ACTIVE state):
   - MOVE events: Dynamically computed == 0 (physical mouse exclusive ownership).
   - CLICK / DOWN / UP events: Dynamically computed == 0 (physical mouse exclusive ownership).
   - ZOOM events: Dynamically computed == 0 (including Ctrl+wheel).
   - HISTORY events: Dynamically computed == 0 (including Alt+Left/Right).
   - SCROLL events: Whitelisted two-finger vertical assist only.
   - TAB events: Whitelisted three-finger tab switching only.
   - Safety invariant: 0 held buttons and 0 held keys after cleanup in finally.
6. Offline FPS disclosure: Headless CPU throughput, not live webcam camera latency.
7. Exclusive artifact output: Writes new artifact exclusively using open(path, 'x').
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List
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

from app import create_pipeline
from src.config.loader import load_config
from tools.train import load_bundle
from src.platform.interface import MouseEvent, KeyEvent

SHADOW_LOG_PATH = "reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl"
EXPECTED_SHADOW_LOG_SHA256 = "2E2753DEF2D95BF9A33CC88C55EABA6F19560B7CD5DF2815CC037BAA3ABC8CD2"

MODEL_BUNDLE_PATH = "reports/acc05/candidate_personal_svm.json"
EXPECTED_MODEL_BUNDLE_SHA256 = "66091D52D9DD6848FD8E35FC2641A31C4C2FFC9008329A4D2E9F85F8DD1246BE"

ACC07_REPLAY_PATH = "reports/acc07/dryrun_replay_events_v2.json"
EXPECTED_ACC07_REPLAY_SHA256 = "6BF9ABD3E8CA96B8BDCBD928E97A11A5B997E9C99072158A95A1AA97B03C3646"


def verify_sha256(path: str, expected_hash: str) -> None:
    assert os.path.exists(path), f"File not found: {path}"
    with open(path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest().upper()
    assert actual_hash == expected_hash, (
        f"Cryptographic hash mismatch for {path}!\n"
        f"  Expected: {expected_hash}\n"
        f"  Got:      {actual_hash}"
    )


def compute_forbidden_counts(events: List[Dict[str, Any]], actions: List[str]) -> Dict[str, int]:
    """Dynamically compute forbidden counts by tracking ordered event stream and key state.

    Ordered-state auditor (shared logic with run_synthetic_intent_benchmark.py):
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

    tracked_held_keys: set = set()

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


def run_benchmark(output_path: str = "reports/hyb01/hybrid_shadow_benchmark_results_v3.json") -> int:
    print(f"Verifying input hashes before benchmark...")
    verify_sha256(SHADOW_LOG_PATH, EXPECTED_SHADOW_LOG_SHA256)
    verify_sha256(MODEL_BUNDLE_PATH, EXPECTED_MODEL_BUNDLE_SHA256)
    verify_sha256(ACC07_REPLAY_PATH, EXPECTED_ACC07_REPLAY_SHA256)
    print("All input SHA-256 hashes successfully verified.")

    with open(SHADOW_LOG_PATH, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    # Filter frame records only (excluding header)
    frames = [rec for rec in all_records if rec.get("record_type") == "frame"]
    total_frames = len(frames)
    detected_count = sum(1 for f in frames if bool(f.get("hand_detected", False)))
    absent_count = sum(1 for f in frames if not bool(f.get("hand_detected", False)))

    print(f"\nLoaded {total_frames} frame records from shadow dataset (header excluded, total lines: {len(all_records)}).")
    print(f"Hand detected frames: {detected_count}, Absent frames: {absent_count}")

    assert len(all_records) == 3321, f"Expected 3321 total lines, got {len(all_records)}"
    assert total_frames == 3320, f"Expected 3320 frames, got {total_frames}"
    assert detected_count == 1875, f"Expected 1875 detected frames, got {detected_count}"
    assert absent_count == 1445, f"Expected 1445 absent frames, got {absent_count}"

    bundle = load_bundle(MODEL_BUNDLE_PATH)
    config = load_config()

    # Load frozen ACC-07 dry-run replay reference for exact per-event comparison
    with open(ACC07_REPLAY_PATH, "r", encoding="utf-8") as f:
        acc07_data = json.load(f)
    acc07_action_events = acc07_data.get("action_events", [])
    acc07_flat_events = [ev for item in acc07_action_events for ev in item.get("events", [])]
    assert len(acc07_flat_events) == 156, f"Expected 156 frozen ACC-07 events, got {len(acc07_flat_events)}"

    # -------------------------------------------------------------
    # Run 1: Legacy Profile (Exact ACC-07 Replay Verification)
    # -------------------------------------------------------------
    pipe_legacy = create_pipeline(config, allow_real_control=False, model_bundle=bundle, profile="legacy")
    ctrl_legacy = pipe_legacy.controller
    curr_time_leg = [0.0]
    if pipe_legacy.rate_limiter:
        rl = pipe_legacy.rate_limiter
        def sim_allow():
            with rl._lock:
                now = curr_time_leg[0]
                if now - rl.last_event_time >= rl.interval:
                    rl.last_event_time = now
                    return True
                return False
        rl.allow = sim_allow

    t_legacy = 0.0
    legacy_actions = Counter()
    try:
        t0 = time.perf_counter()
        for rec in frames:
            ts_sec = float(rec["timestamp"])
            curr_time_leg[0] = ts_sec
            raw_lms = rec.get("raw_landmarks")
            lms = np.asarray(raw_lms, dtype=np.float32) if raw_lms and len(raw_lms) == 21 else None
            res = pipe_legacy.process_landmarks(lms, handedness=rec.get("handedness") or "Right", timestamp_ms=ts_sec * 1000.0)
            if res.action_fired:
                legacy_actions[res.action_fired] += 1
        t_legacy = time.perf_counter() - t0
    finally:
        pipe_legacy.cleanup()

    legacy_events = list(ctrl_legacy.event_log)
    legacy_event_types = Counter(e["type"] for e in legacy_events)

    print(f"\n--- Legacy Profile Results ---")
    print(f"Elapsed: {t_legacy:.3f}s ({total_frames / t_legacy:.1f} headless fps)")
    print(f"Controller events: {len(legacy_events)}: {dict(legacy_event_types)}")
    print(f"Actions fired: {dict(legacy_actions)}")
    print(f"Held buttons after cleanup: {ctrl_legacy.held_buttons}")
    print(f"Held keys after cleanup: {ctrl_legacy.held_keys}")

    # Exact frozen ACC-07 baseline agreement: 156 events (151 MOVE, 5 SCROLL)
    assert len(legacy_events) == 156, f"Expected 156 events in legacy, got {len(legacy_events)}"
    assert legacy_event_types[MouseEvent.MOVE] == 151, f"Expected 151 MOVE events, got {legacy_event_types[MouseEvent.MOVE]}"
    assert legacy_event_types[MouseEvent.SCROLL] == 5, f"Expected 5 SCROLL events, got {legacy_event_types[MouseEvent.SCROLL]}"
    assert len(ctrl_legacy.held_buttons) == 0
    assert len(ctrl_legacy.held_keys) == 0

    # Exact per-event equality verification against frozen ACC-07 replay
    per_event_matches = True
    mismatched_idx = None
    for idx, (leg, acc) in enumerate(zip(legacy_events, acc07_flat_events)):
        leg_norm = dict(leg)
        leg_norm["type"] = leg["type"].name if hasattr(leg["type"], "name") else str(leg["type"])
        if leg_norm != acc:
            per_event_matches = False
            mismatched_idx = idx
            break
    assert per_event_matches, f"Legacy replay event mismatch at index {mismatched_idx}!"
    print(f"ACC-07 Replay Exact Per-Event Match: {per_event_matches} (all 156 events match)")

    # -------------------------------------------------------------
    # Run 2: Hybrid Profile (Default OFF)
    # -------------------------------------------------------------
    pipe_hybrid_off = create_pipeline(config, allow_real_control=False, model_bundle=bundle, profile="hybrid")
    ctrl_hybrid_off = pipe_hybrid_off.controller
    assert pipe_hybrid_off.assist_state == "OFF"

    t_hybrid_off = 0.0
    hybrid_off_actions: List[str] = []
    try:
        t0 = time.perf_counter()
        for rec in frames:
            ts_sec = float(rec["timestamp"])
            raw_lms = rec.get("raw_landmarks")
            lms = np.asarray(raw_lms, dtype=np.float32) if raw_lms and len(raw_lms) == 21 else None
            res = pipe_hybrid_off.process_landmarks(lms, handedness=rec.get("handedness") or "Right", timestamp_ms=ts_sec * 1000.0)
            if res.action_fired:
                hybrid_off_actions.append(res.action_fired)
        t_hybrid_off = time.perf_counter() - t0
    finally:
        pipe_hybrid_off.cleanup()

    hybrid_off_events = list(ctrl_hybrid_off.event_log)
    hybrid_off_forbidden = compute_forbidden_counts(hybrid_off_events, hybrid_off_actions)

    print(f"\n--- Hybrid Profile (OFF) Results ---")
    print(f"Elapsed: {t_hybrid_off:.3f}s ({total_frames / t_hybrid_off:.1f} headless fps)")
    print(f"Controller events: {len(hybrid_off_events)} (expected: 0)")
    print(f"Actions fired: {dict(Counter(hybrid_off_actions))} (expected: 0)")
    assert len(hybrid_off_events) == 0, "OFF state must emit 0 controller events"
    assert sum(hybrid_off_forbidden.values()) == 0, f"Forbidden events in OFF state: {hybrid_off_forbidden}"
    assert len(ctrl_hybrid_off.held_buttons) == 0
    assert len(ctrl_hybrid_off.held_keys) == 0

    # -------------------------------------------------------------
    # Run 3: Hybrid Profile (Active READY)
    # -------------------------------------------------------------
    pipe_hybrid_on = create_pipeline(config, allow_real_control=False, model_bundle=bundle, profile="hybrid")
    ctrl_hybrid_on = pipe_hybrid_on.controller
    pipe_hybrid_on.toggle_assist()
    assert pipe_hybrid_on.assist_state == "READY"
    curr_time_on = [0.0]
    if pipe_hybrid_on.rate_limiter:
        rl = pipe_hybrid_on.rate_limiter
        def sim_allow_on():
            with rl._lock:
                now = curr_time_on[0]
                if now - rl.last_event_time >= rl.interval:
                    rl.last_event_time = now
                    return True
                return False
        rl.allow = sim_allow_on

    t_hybrid_on = 0.0
    hybrid_on_actions: List[str] = []
    try:
        t0 = time.perf_counter()
        for rec in frames:
            ts_sec = float(rec["timestamp"])
            curr_time_on[0] = ts_sec
            raw_lms = rec.get("raw_landmarks")
            lms = np.asarray(raw_lms, dtype=np.float32) if raw_lms and len(raw_lms) == 21 else None
            res = pipe_hybrid_on.process_landmarks(lms, handedness=rec.get("handedness") or "Right", timestamp_ms=ts_sec * 1000.0)
            if res.action_fired:
                hybrid_on_actions.append(res.action_fired)
        t_hybrid_on = time.perf_counter() - t0
    finally:
        pipe_hybrid_on.cleanup()

    hybrid_on_events = list(ctrl_hybrid_on.event_log)
    hybrid_on_event_types = Counter(e["type"] for e in hybrid_on_events)
    hybrid_on_forbidden = compute_forbidden_counts(hybrid_on_events, hybrid_on_actions)

    print(f"\n--- Hybrid Profile (ACTIVE) Results ---")
    print(f"Elapsed: {t_hybrid_on:.3f}s ({total_frames / t_hybrid_on:.1f} headless fps)")
    print(f"Controller events: {len(hybrid_on_events)}: {dict(hybrid_on_event_types)}")
    print(f"Actions fired: {dict(Counter(hybrid_on_actions))}")
    print(f"Forbidden counts: {hybrid_on_forbidden}")
    print(f"Held buttons after cleanup: {ctrl_hybrid_on.held_buttons}")
    print(f"Held keys after cleanup: {ctrl_hybrid_on.held_keys}")

    # Physical mouse ownership assertions:
    assert hybrid_on_forbidden["forbidden_moves"] == 0, f"Forbidden MOVE events detected: {hybrid_on_forbidden['forbidden_moves']}"
    assert hybrid_on_forbidden["forbidden_clicks"] == 0, f"Forbidden CLICK events detected: {hybrid_on_forbidden['forbidden_clicks']}"
    assert hybrid_on_forbidden["forbidden_drags"] == 0, f"Forbidden DRAG events detected: {hybrid_on_forbidden['forbidden_drags']}"
    assert hybrid_on_forbidden["forbidden_zooms"] == 0, f"Forbidden ZOOM events detected: {hybrid_on_forbidden['forbidden_zooms']}"
    assert hybrid_on_forbidden["forbidden_history"] == 0, f"Forbidden HISTORY events detected: {hybrid_on_forbidden['forbidden_history']}"
    assert hybrid_on_forbidden["forbidden_other"] == 0, f"Forbidden OTHER events detected: {hybrid_on_forbidden['forbidden_other']}"
    assert len(ctrl_hybrid_on.held_buttons) == 0, "No held buttons after cleanup"
    assert len(ctrl_hybrid_on.held_keys) == 0, "No held keys after cleanup"

    # -------------------------------------------------------------
    # Negative Controls: Verify auditor detects injected forbidden events
    # -------------------------------------------------------------
    print(f"\n--- Negative Controls Verification ---")

    # 1. Ctrl+wheel -> zoom
    neg_zoom_events: List[Dict[str, Any]] = [
        {"type": KeyEvent.KEY_DOWN, "key": "ctrl"},
        {"type": MouseEvent.SCROLL, "dx": 0, "dy": 3},
        {"type": KeyEvent.KEY_UP, "key": "ctrl"},
    ]
    neg_zoom_result = compute_forbidden_counts(neg_zoom_events, [])
    injected_zoom_ok = neg_zoom_result["forbidden_zooms"] == 1
    print(f"  Injected Zoom (Ctrl+Wheel) detected: {injected_zoom_ok}")
    assert injected_zoom_ok, f"Negative control: Ctrl+Wheel not detected as zoom: {neg_zoom_result}"

    # 2. Alt+Left -> history
    neg_history_events: List[Dict[str, Any]] = [
        {"type": KeyEvent.KEY_DOWN, "key": "alt"},
        {"type": KeyEvent.KEY_DOWN, "key": "left"},
        {"type": KeyEvent.KEY_UP, "key": "left"},
        {"type": KeyEvent.KEY_UP, "key": "alt"},
    ]
    neg_history_result = compute_forbidden_counts(neg_history_events, [])
    injected_history_ok = neg_history_result["forbidden_history"] == 1
    print(f"  Injected History (Alt+Left) detected: {injected_history_ok}")
    assert injected_history_ok, f"Negative control: Alt+Left not detected as history: {neg_history_result}"

    # 3. MOVE injection
    neg_move_events: List[Dict[str, Any]] = [
        {"type": MouseEvent.MOVE, "x": 100, "y": 200},
    ]
    neg_move_result = compute_forbidden_counts(neg_move_events, [])
    injected_move_ok = neg_move_result["forbidden_moves"] == 1
    print(f"  Injected MOVE detected: {injected_move_ok}")
    assert injected_move_ok, f"Negative control: MOVE not detected: {neg_move_result}"

    # 4. Alt+Tab -> window switch (forbidden_other)
    #    Alt down (1) + Alt+Tab down (2) + Alt up (3) = 3 forbidden_other
    neg_alt_tab_events: List[Dict[str, Any]] = [
        {"type": KeyEvent.KEY_DOWN, "key": "alt"},
        {"type": KeyEvent.KEY_DOWN, "key": "tab"},
        {"type": KeyEvent.KEY_UP, "key": "tab"},
        {"type": KeyEvent.KEY_UP, "key": "alt"},
    ]
    neg_alt_tab_result = compute_forbidden_counts(neg_alt_tab_events, [])
    injected_alt_tab_ok = neg_alt_tab_result["forbidden_other"] == 3
    print(f"  Injected Alt+Tab (window switch) detected: {injected_alt_tab_ok} (forbidden_other={neg_alt_tab_result['forbidden_other']})")
    assert injected_alt_tab_ok, f"Negative control: Alt+Tab expected forbidden_other=3, got: {neg_alt_tab_result}"

    # 5. Horizontal scroll (dx != 0) -> forbidden_other
    neg_hscroll_events: List[Dict[str, Any]] = [
        {"type": MouseEvent.SCROLL, "dx": 3, "dy": 0},
    ]
    neg_hscroll_result = compute_forbidden_counts(neg_hscroll_events, [])
    injected_hscroll_ok = neg_hscroll_result["forbidden_other"] == 1
    print(f"  Injected Horizontal SCROLL (dx!=0) detected: {injected_hscroll_ok}")
    assert injected_hscroll_ok, f"Negative control: Horizontal scroll not detected: {neg_hscroll_result}"

    # 6. Alt-alone -> forbidden_other (Alt alone can focus OS menu bars)
    neg_alt_alone_events: List[Dict[str, Any]] = [
        {"type": KeyEvent.KEY_DOWN, "key": "alt"},
        {"type": KeyEvent.KEY_UP, "key": "alt"},
    ]
    neg_alt_alone_result = compute_forbidden_counts(neg_alt_alone_events, [])
    injected_alt_alone_ok = neg_alt_alone_result["forbidden_other"] == 2
    print(f"  Injected Alt-alone (menu focus) detected: {injected_alt_alone_ok} (forbidden_other={neg_alt_alone_result['forbidden_other']})")
    assert injected_alt_alone_ok, f"Negative control: Alt-alone expected forbidden_other=2, got: {neg_alt_alone_result}"

    benchmark_summary = {
        "benchmark_type": "hybrid_shadow_benchmark_v3",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline_fps_disclosure": "Offline FPS measures headless CPU execution throughput and is not live webcam camera latency.",
        "negative_controls": {
            "injected_zoom_detected": injected_zoom_ok,
            "injected_history_detected": injected_history_ok,
            "injected_move_detected": injected_move_ok,
            "injected_alt_tab_detected": injected_alt_tab_ok,
            "injected_horizontal_scroll_detected": injected_hscroll_ok,
            "injected_alt_alone_detected": injected_alt_alone_ok,
        },
        "inputs": {
            "shadow_log": {
                "path": SHADOW_LOG_PATH,
                "sha256": EXPECTED_SHADOW_LOG_SHA256,
                "total_lines": len(all_records),
                "header_records": 1,
                "frame_records": total_frames,
                "detected_frames": detected_count,
                "absent_frames": absent_count,
            },
            "model_bundle": {
                "path": MODEL_BUNDLE_PATH,
                "sha256": EXPECTED_MODEL_BUNDLE_SHA256,
            },
            "acc07_reference": {
                "path": ACC07_REPLAY_PATH,
                "sha256": EXPECTED_ACC07_REPLAY_SHA256,
                "total_events": len(acc07_flat_events),
            },
        },
        "legacy": {
            "elapsed_seconds": round(t_legacy, 3),
            "headless_fps": round(total_frames / t_legacy, 1),
            "total_events": len(legacy_events),
            "events_by_type": {str(k): v for k, v in legacy_event_types.items()},
            "actions_fired": dict(legacy_actions),
            "counts_match": len(legacy_events) == 156 and legacy_event_types[MouseEvent.MOVE] == 151 and legacy_event_types[MouseEvent.SCROLL] == 5,
            "per_event_match": per_event_matches,
            "held_buttons_cleanup": list(ctrl_legacy.held_buttons),
            "held_keys_cleanup": list(ctrl_legacy.held_keys),
        },
        "hybrid_off": {
            "elapsed_seconds": round(t_hybrid_off, 3),
            "headless_fps": round(total_frames / t_hybrid_off, 1),
            "total_events": len(hybrid_off_events),
            "actions_fired": dict(Counter(hybrid_off_actions)),
            "forbidden_counts": hybrid_off_forbidden,
            "held_buttons_cleanup": list(ctrl_hybrid_off.held_buttons),
            "held_keys_cleanup": list(ctrl_hybrid_off.held_keys),
        },
        "hybrid_active": {
            "elapsed_seconds": round(t_hybrid_on, 3),
            "headless_fps": round(total_frames / t_hybrid_on, 1),
            "total_events": len(hybrid_on_events),
            "events_by_type": {str(k): v for k, v in hybrid_on_event_types.items()},
            "actions_fired": dict(Counter(hybrid_on_actions)),
            "forbidden_counts": hybrid_on_forbidden,
            "held_buttons_cleanup": list(ctrl_hybrid_on.held_buttons),
            "held_keys_cleanup": list(ctrl_hybrid_on.held_keys),
        },
    }

    # Write exclusively to new output path (must not overwrite existing files)
    print(f"\nWriting benchmark results exclusively to: {output_path}")
    with open(output_path, "x", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"Benchmark results successfully written to {output_path}.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Headless Action Benchmark for Hybrid Assist Profile (HYB-01)")
    parser.add_argument(
        "--output",
        default="reports/hyb01/hybrid_shadow_benchmark_results_v3.json",
        help="Output path for benchmark results JSON (must not already exist, written with exclusive creation)",
    )
    args = parser.parse_args()
    return run_benchmark(output_path=args.output)


if __name__ == "__main__":
    sys.exit(main())
