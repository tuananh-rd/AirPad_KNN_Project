"""AirPad ACC-06 Bounded Shadow-Only Fresh Observation Trial Tool.

Standalone, fail-closed experimental tool for observing and evaluating a frozen
candidate gesture classifier on live (or replayed) camera frames without action dispatch.

ZERO ACTION DISPATCH BY CONSTRUCTION:
- Never imports or instantiates platform controllers, window controllers,
  event synthesizers, pyautogui, pynput, or runtime pipeline FSMs.
- CLI rejects --real-control and unsafe execution flags before camera initialization.
- Actual system commands dispatched is strictly 0.

GUIDED PROTOCOL:
- Preview starts in WAITING phase.
- Recording starts ONLY when user presses SPACE once.
- Predeclared stage sequence:
  1. NO_HAND (Initial Baseline)
  2..10. Numbered gestures 1..9 in collection order:
      [1] PINCH_CLOSED, [2] PINCH_MIDDLE, [3] L_SHAPE, [4] OTHER,
      [5] POINT_ONE, [6] OPEN_PALM, [7] FIST, [8] TWO_FINGERS_PARALLEL, [9] THREE_FINGERS
  11. OTHER (Post-Gesture Catch-All)
  12. NO_HAND (Final Post-Trial Negative)
- Each stage runs a Preparation phase (is_transition=True, >=3.0s default, excluded from scored metrics)
  followed by an Observation phase (is_transition=False, >=5.0s default, included in scored metrics).
- Auto-advances across stages; no per-gesture key or save key needed.
- Continuous incremental flushing to JSONL; summary JSON written on completion or early exit.
- Q / ESC / window close causes immediate safe exit marked INCOMPLETE.

IMPORTANT CAVEAT:
Labels recorded are declared INTENDED poses, not independently verified physical ground truth.
All computed metrics are observational intended-pose metrics on same-user one-take captures.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.canonicalize import canonicalize
from src.vision.adapter import MediaPipeAdapter
from src.vision.landmarks import HandLandmarks
from tools.train import ModelBundle, load_model_from_bundle

# Pinned immutable artifacts
DEFAULT_CANDIDATE_PATH = os.path.join(PROJECT_ROOT, "reports", "acc05", "candidate_personal_svm.json")
PINNED_CANDIDATE_SHA256 = "66091d52d9dd6848fd8e35fc2641a31c4c2ffc9008329a4d2e9f85f8dd1246be"

DEFAULT_BASELINE_PATH = os.path.join(PROJECT_ROOT, "models", "airpad_svm.json")
PINNED_BASELINE_SHA256 = "4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977"

DEFAULT_DETECTOR_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
PINNED_DETECTOR_TASK_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

# Frozen decision thresholds
DECISION_THRESHOLD = 0.0
MARGIN_THRESHOLD = 0.15
FEATURE_VERSION = 1

# Canonical 9 static target classes in collection order
COLLECTION_GESTURES: List[Tuple[int, str]] = [
    (1, "PINCH_CLOSED"),
    (2, "PINCH_MIDDLE"),
    (3, "L_SHAPE"),
    (4, "OTHER"),
    (5, "POINT_ONE"),
    (6, "OPEN_PALM"),
    (7, "FIST"),
    (8, "TWO_FINGERS_PARALLEL"),
    (9, "THREE_FINGERS"),
]

TARGET_ACTION_CLASSES = {
    "PINCH_CLOSED",
    "PINCH_MIDDLE",
    "L_SHAPE",
    "POINT_ONE",
    "OPEN_PALM",
    "FIST",
    "TWO_FINGERS_PARALLEL",
    "THREE_FINGERS",
}

ALL_9_CLASSES = [name for _, name in COLLECTION_GESTURES]

# Predeclared fixed guided sequence: NO_HAND, 1..9, OTHER again, NO_HAND again (12 blocks)
GUIDED_STAGES: List[Dict[str, Any]] = [
    {"stage_index": 0, "stage_id": 1, "intended_label": "NO_HAND", "display_name": "NO_HAND (Initial Baseline)"},
    {"stage_index": 1, "stage_id": 2, "intended_label": "PINCH_CLOSED", "display_name": "[1] PINCH_CLOSED"},
    {"stage_index": 2, "stage_id": 3, "intended_label": "PINCH_MIDDLE", "display_name": "[2] PINCH_MIDDLE"},
    {"stage_index": 3, "stage_id": 4, "intended_label": "L_SHAPE", "display_name": "[3] L_SHAPE"},
    {"stage_index": 4, "stage_id": 5, "intended_label": "OTHER", "display_name": "[4] OTHER"},
    {"stage_index": 5, "stage_id": 6, "intended_label": "POINT_ONE", "display_name": "[5] POINT_ONE"},
    {"stage_index": 6, "stage_id": 7, "intended_label": "OPEN_PALM", "display_name": "[6] OPEN_PALM"},
    {"stage_index": 7, "stage_id": 8, "intended_label": "FIST", "display_name": "[7] FIST"},
    {"stage_index": 8, "stage_id": 9, "intended_label": "TWO_FINGERS_PARALLEL", "display_name": "[8] TWO_FINGERS_PARALLEL"},
    {"stage_index": 9, "stage_id": 10, "intended_label": "THREE_FINGERS", "display_name": "[9] THREE_FINGERS"},
    {"stage_index": 10, "stage_id": 11, "intended_label": "OTHER", "display_name": "OTHER (Post-Gesture Catch-All)"},
    {"stage_index": 11, "stage_id": 12, "intended_label": "NO_HAND", "display_name": "NO_HAND (Final Post-Trial Negative)"},
]

FORBIDDEN_FLAGS = {"--real-control", "--dispatch", "--execute", "--enable-control", "--live-control"}


def compute_file_sha256(filepath: str) -> str:
    """Compute standard SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_forbidden_cli_args(args_list: Optional[List[str]] = None) -> None:
    """Reject unsafe flags before any camera or model initialization."""
    check_args = args_list if args_list is not None else sys.argv[1:]
    for arg in check_args:
        lower = arg.lower()
        if any(lower == flag or lower.startswith(flag + "=") for flag in FORBIDDEN_FLAGS):
            raise ValueError(
                f"[SAFETY FAIL-CLOSED] Forbidden action-dispatch flag '{arg}' detected. "
                f"Shadow trial is classification and observability ONLY. Zero commands dispatched by construction."
            )


def validate_trial_parameters(user_id: str, prep_duration: float, obs_duration: float) -> None:
    """Validate user identifier and phase durations fail-closed before any file/device actions."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError(f"[FAIL-CLOSED] user_id must be a non-empty string, got: {user_id!r}")

    for name, val in [("prep_duration", prep_duration), ("obs_duration", obs_duration)]:
        if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"[FAIL-CLOSED] {name} must be a positive finite number (>0), got: {val!r}")


def validate_shadow_trial_paths(
    output_jsonl: str,
    output_summary: str,
    candidate_bundle_path: str,
    task_model_path: str,
    baseline_path: Optional[str] = None,
) -> None:
    """Validate all input and output paths for existence, extension, and anti-aliasing guards."""
    # 1. Output extension checks
    if not output_jsonl.lower().endswith(".jsonl"):
        raise ValueError(f"[FAIL-CLOSED] output_jsonl must end with literal .jsonl, got: '{output_jsonl}'")
    if not output_summary.lower().endswith(".json"):
        raise ValueError(f"[FAIL-CLOSED] output_summary must end with literal .json, got: '{output_summary}'")

    # 2. Mandatory candidate bundle existence
    if not os.path.isfile(candidate_bundle_path):
        raise FileNotFoundError(f"[FAIL-CLOSED] Candidate bundle not found at: '{candidate_bundle_path}'")

    # 3. Mandatory local detector task model existence (never allow implicit network download)
    if not os.path.isfile(task_model_path):
        raise FileNotFoundError(
            f"[FAIL-CLOSED] Local detector task model missing at: '{task_model_path}'. "
            f"Implicit network downloads are strictly prohibited in ACC-06."
        )

    # 4. Anti-aliasing guards: outputs must not alias candidate bundle, detector model, or baseline
    norm_jsonl = os.path.normcase(os.path.abspath(output_jsonl))
    norm_summary = os.path.normcase(os.path.abspath(output_summary))
    norm_cand = os.path.normcase(os.path.abspath(candidate_bundle_path))
    norm_task = os.path.normcase(os.path.abspath(task_model_path))

    if norm_jsonl == norm_summary:
        raise ValueError("[FAIL-CLOSED] output_jsonl and output_summary alias the same file path.")
    if norm_jsonl in (norm_cand, norm_task):
        raise ValueError(f"[FAIL-CLOSED] output_jsonl aliases an immutable input file: '{output_jsonl}'")
    if norm_summary in (norm_cand, norm_task):
        raise ValueError(f"[FAIL-CLOSED] output_summary aliases an immutable input file: '{output_summary}'")

    if baseline_path is not None and os.path.exists(baseline_path):
        norm_base = os.path.normcase(os.path.abspath(baseline_path))
        if norm_jsonl == norm_base or norm_summary == norm_base:
            raise ValueError("[FAIL-CLOSED] Output aliases baseline model path.")

    # 5. Output collision checks (fail-closed if output already exists)
    if os.path.exists(output_jsonl):
        raise FileExistsError(f"[FAIL-CLOSED] Output file already exists: '{output_jsonl}'. Overwriting is prohibited.")
    if os.path.exists(output_summary):
        raise FileExistsError(f"[FAIL-CLOSED] Output file already exists: '{output_summary}'. Overwriting is prohibited.")


def _safe_close(handle: Any) -> None:
    """Close a file handle safely without raising exceptions."""
    if handle is not None and not getattr(handle, "closed", True):
        try:
            handle.close()
        except Exception:
            pass


def reserve_output_handles(output_jsonl: str, output_summary: str) -> Tuple[Any, Any]:
    """Create and reserve exclusive handles for both outputs sequentially using mode 'x'.

    Note: Reservation of the pair is sequential rather than truly atomic across the filesystem.
    Each individual open(..., 'x') is atomic. If opening the summary handle fails (e.g. output_summary
    already exists), the newly created empty JSONL handle is closed and its empty file safely
    removed, while the pre-existing collision summary file is strictly preserved.
    """
    p_dir_jsonl = os.path.dirname(output_jsonl)
    if p_dir_jsonl:
        os.makedirs(p_dir_jsonl, exist_ok=True)
    p_dir_summary = os.path.dirname(output_summary)
    if p_dir_summary:
        os.makedirs(p_dir_summary, exist_ok=True)

    f_jsonl = open(output_jsonl, "x", encoding="utf-8")
    try:
        f_summary = open(output_summary, "x", encoding="utf-8")
    except Exception:
        _safe_close(f_jsonl)
        try:
            if os.path.exists(output_jsonl) and os.path.getsize(output_jsonl) == 0:
                os.remove(output_jsonl)
        except Exception:
            pass
        raise
    return f_jsonl, f_summary


def validate_and_load_candidate_bundle(
    bundle_path: str,
    expected_sha256: Optional[str] = PINNED_CANDIDATE_SHA256,
) -> Tuple[Any, Any, ModelBundle]:
    """Verify hash, validate metadata schema, and deterministically restore classifier and scaler."""
    actual_hash = compute_file_sha256(bundle_path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError(
            f"[FAIL-CLOSED] Candidate bundle SHA-256 mismatch!\n"
            f"  Expected: {expected_sha256}\n"
            f"  Actual:   {actual_hash}\n"
            f"  Path:     {bundle_path}"
        )

    bundle = ModelBundle.load(bundle_path)
    if bundle.feature_version != FEATURE_VERSION:
        raise ValueError(
            f"[FAIL-CLOSED] Candidate bundle feature_version {bundle.feature_version} != expected {FEATURE_VERSION}"
        )

    # Validate exact 9 classes
    if set(bundle.labels) != set(ALL_9_CLASSES) or len(bundle.labels) != 9:
        raise ValueError(
            f"[FAIL-CLOSED] Candidate bundle must contain exactly the 9 canonical target classes.\n"
            f"  Expected: {set(ALL_9_CLASSES)}\n"
            f"  Got:      {set(bundle.labels)}"
        )

    # Validate frozen hyperparameters
    hp = bundle.hyperparameters
    if hp.get("C") != 10.0 or hp.get("gamma") != 0.01 or hp.get("class_weight") != "balanced":
        raise ValueError(
            f"[FAIL-CLOSED] Candidate bundle hyperparameters mismatch!\n"
            f"  Expected: {{'C': 10.0, 'gamma': 0.01, 'class_weight': 'balanced'}}\n"
            f"  Got:      {hp}"
        )

    # Validate scaler metadata
    scaler_info = bundle.metadata.get("scaler")
    if not scaler_info or "mean" not in scaler_info or "scale" not in scaler_info:
        raise ValueError("[FAIL-CLOSED] Candidate bundle missing valid StandardScaler metadata")
    if len(scaler_info["mean"]) != 63 or len(scaler_info["scale"]) != 63:
        raise ValueError("[FAIL-CLOSED] Candidate scaler mean/scale dimensions must be 63")

    classifier, scaler, loaded_bundle = load_model_from_bundle(bundle_path)
    return classifier, scaler, loaded_bundle


def pre_process_frame_bgr_to_rgb(bgr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Execute required video pre-processing: horizontal flip then explicit BGR->RGB conversion.

    Returns:
        (flipped_bgr, rgb_frame)
        - flipped_bgr: for preview annotation and display
        - rgb_frame: passed exclusively to MediaPipeAdapter.detect()
    """
    flipped_bgr = cv2.flip(bgr_frame, 1)
    rgb_frame = cv2.cvtColor(flipped_bgr, cv2.COLOR_BGR2RGB)
    return flipped_bgr, rgb_frame


def classify_hand(
    hand: HandLandmarks,
    classifier: Any,
    scaler: Any,
    decision_threshold: float = DECISION_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> Dict[str, Any]:
    """Validate finite landmarks, canonicalize, scale, and compute raw and rejection predictions."""
    raw_21x3 = hand.to_numpy()

    # Fail closed on non-finite landmarks or incorrect shape
    if not isinstance(raw_21x3, np.ndarray) or raw_21x3.shape != (21, 3):
        raise ValueError(f"[FAIL-CLOSED] Hand landmarks shape must be (21, 3), got {getattr(raw_21x3, 'shape', None)}")
    if not np.all(np.isfinite(raw_21x3)):
        raise ValueError("[FAIL-CLOSED] Hand landmarks contain NaN or Inf values")

    # Validate handedness
    handedness = hand.handedness
    if not handedness or not isinstance(handedness, str) or handedness.strip().lower() not in ("right", "left"):
        raise ValueError(f"[FAIL-CLOSED] Invalid handedness: {handedness!r} (must be 'Right' or 'Left')")
    handedness = "Right" if handedness.strip().lower() == "right" else "Left"

    confidence = float(hand.confidence) if hand.confidence is not None else 0.0
    if not np.isfinite(confidence):
        raise ValueError(f"[FAIL-CLOSED] Invalid hand confidence: {confidence!r}")

    canonical_63 = canonicalize(raw_21x3, handedness=handedness)
    canonical_row = canonical_63.reshape(1, -1)

    scores = classifier.decision_function(canonical_row)[0]
    best_idx = int(np.argmax(scores))
    raw_pred = str(classifier.model.classes_[best_idx])
    best_score = float(scores[best_idx])

    if len(scores) > 1:
        top_two = np.partition(scores, -2)[-2:]
        margin = float(top_two[1] - top_two[0])
    else:
        margin = float("inf")

    # Rejection policy
    if best_score < decision_threshold or margin < margin_threshold:
        rej_pred = "UNKNOWN"
    else:
        rej_pred = raw_pred

    all_scores = {str(cls_name): float(scores[i]) for i, cls_name in enumerate(classifier.model.classes_)}

    return {
        "raw_landmarks": raw_21x3.tolist(),
        "canonical_landmarks": canonical_63.tolist(),
        "handedness": handedness,
        "hand_confidence": confidence,
        "raw_prediction": raw_pred,
        "rejected_prediction": rej_pred,
        "best_score": best_score,
        "margin": margin,
        "all_scores": all_scores,
    }


def draw_hud_overlay(
    frame_bgr: np.ndarray,
    stage_name: str,
    phase: str,
    time_remaining: float,
    current_prediction: str,
    current_margin: Optional[float],
    hand_detected: bool,
    commands_count: int = 0,
) -> np.ndarray:
    """Render high-visibility visual guide with countdown and strict 0-command status.

    Multi-line banner layout fits both 640x480 and 320x240 resolutions:
    - Line 1 (y22): System title and zero action constraint
    - Line 2 (y56): Intended stage name, dynamically scaled via getTextSize to fit width
    - Line 3 (y84): Current phase and countdown timer on separate line (no horizontal overlap)
    - Line 4 (y108): Phase instructional status
    - Bottom bar (h-40 to h): Hand detection status, model prediction, locked 0 commands
    """
    canvas = frame_bgr.copy()
    h, w, _ = canvas.shape

    # Top banner background (120px tall)
    banner_height = 120
    banner_color = (30, 30, 30)
    if phase == "WAITING":
        banner_color = (70, 40, 20)  # Muted brown/amber
    elif phase == "PREPARATION":
        banner_color = (20, 50, 80)  # Deep orange/amber tone in BGR
    elif phase == "OBSERVATION":
        banner_color = (20, 70, 20)  # Green tone

    cv2.rectangle(canvas, (0, 0), (w, banner_height), banner_color, -1)
    cv2.line(canvas, (0, banner_height), (w, banner_height), (200, 200, 200), 2)

    # Line 1: Header title
    title_text = "AIRPAD SHADOW TRIAL [OBSERVATION ONLY | 0 ACTIONS DISPATCHED]"
    if w < 500:
        title_text = "AIRPAD SHADOW TRIAL [0 ACTIONS]"
    title_scale = 0.5 if w >= 500 else 0.4
    cv2.putText(
        canvas,
        title_text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (0, 255, 255),
        1 if w < 500 else 2,
        cv2.LINE_AA,
    )

    if phase == "WAITING":
        wait_text = "STATUS: WAITING -- PRESS [SPACE] TO START"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.7 if w >= 600 else 0.5
        cv2.putText(canvas, wait_text, (10, 60), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Press [Q] or [ESC] to quit",
            (10, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        # Line 2: Stage Name - dynamically fit to width so long names like [8] TWO_FINGERS_PARALLEL never clip
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.75 if w >= 600 else 0.55
        max_stage_w = w - 20
        text_size, _ = cv2.getTextSize(stage_name, font, font_scale, 2)
        while text_size[0] > max_stage_w and font_scale > 0.35:
            font_scale -= 0.05
            text_size, _ = cv2.getTextSize(stage_name, font, font_scale, 2)

        cv2.putText(canvas, stage_name, (10, 56), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # Line 3: Separate line for Countdown and Phase
        phase_label = f"Phase: {phase}"
        timer_text = f"Time Left: {max(0.0, time_remaining):.1f}s"
        cv2.putText(canvas, phase_label, (10, 84), cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 240, 255), 1, cv2.LINE_AA)

        t_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
        timer_x = max(w - t_size[0] - 10, 160)
        cv2.putText(canvas, timer_text, (timer_x, 84), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        # Line 4: Instructional guidance
        status_sub = (
            "TRANSITION -- HOLD INTENDED POSE"
            if phase == "PREPARATION"
            else "STABLE OBSERVATION -- RECORDING ACTIVE"
        )
        if w < 400:
            status_sub = "TRANSITION (NOT SCORED)" if phase == "PREPARATION" else "SCORED OBSERVATION"
        cv2.putText(canvas, status_sub, (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)

    # Bottom status bar (40px tall)
    bar_top = h - 40
    cv2.rectangle(canvas, (0, bar_top), (w, h), (20, 20, 20), -1)
    cv2.line(canvas, (0, bar_top), (w, bar_top), (100, 100, 100), 1)

    det_text = "Hand: YES" if hand_detected else "Hand: NO"
    det_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(canvas, det_text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 2, cv2.LINE_AA)

    pred_str = f"Pred: {current_prediction}"
    if current_margin is not None and w >= 500:
        pred_str += f" (m={current_margin:.2f})"
    pred_x = 120 if w >= 500 else 100
    cv2.putText(canvas, pred_str, (pred_x, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cmd_text = "Cmds: 0 (LOCKED)"
    c_size, _ = cv2.getTextSize(cmd_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(canvas, cmd_text, (w - c_size[0] - 10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

    return canvas


def compute_shadow_trial_summary(
    records: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute observational summary metrics across NO_HAND and target gesture stages."""
    # Filter records by phase (ignore header records)
    frame_records = [r for r in records if r.get("record_type") == "frame"]
    waiting_records = [r for r in frame_records if r["phase"] == "WAITING"]
    prep_records = [r for r in frame_records if r["phase"] == "PREPARATION"]
    stable_records = [r for r in frame_records if r["phase"] == "OBSERVATION"]

    # 1. NO_HAND stages metrics (Initial stage 0 and Final stage 11)
    stable_no_hand = [r for r in stable_records if r["intended_label"] == "NO_HAND"]
    no_hand_total = len(stable_no_hand)
    no_hand_detected = sum(1 for r in stable_no_hand if r["hand_detected"])
    no_hand_false_target_predictions = sum(
        1 for r in stable_no_hand if r.get("rejected_prediction") in TARGET_ACTION_CLASSES
    )
    no_hand_other_predictions = sum(
        1 for r in stable_no_hand if r.get("rejected_prediction") == "OTHER"
    )

    # 2. Target gesture stages metrics (Stages with intended_label in ALL_9_CLASSES)
    stable_target_records = [r for r in stable_records if r["intended_label"] != "NO_HAND"]
    target_total = len(stable_target_records)
    target_detected = sum(1 for r in stable_target_records if r["hand_detected"])
    target_detection_rate = (target_detected / target_total) if target_total > 0 else 0.0
    detector_loss_count = target_total - target_detected

    # Target predictions evaluation (FULL DENOMINATOR: detector losses count as errors)
    correct_rejection = 0
    correct_raw = 0
    rejections_count = 0
    other_to_target_count = 0
    other_support = 0

    per_class_stats: Dict[str, Dict[str, int]] = {
        cls_name: {"support": 0, "correct_raw": 0, "correct_rejection": 0, "detected": 0}
        for cls_name in ALL_9_CLASSES
    }

    for r in stable_target_records:
        intended = r["intended_label"]
        raw_pred = r.get("raw_prediction")
        rej_pred = r.get("rejected_prediction")

        if intended in per_class_stats:
            per_class_stats[intended]["support"] += 1
            if r["hand_detected"]:
                per_class_stats[intended]["detected"] += 1

        if rej_pred == "UNKNOWN":
            rejections_count += 1

        if raw_pred == intended:
            correct_raw += 1
            if intended in per_class_stats:
                per_class_stats[intended]["correct_raw"] += 1

        if rej_pred == intended:
            correct_rejection += 1
            if intended in per_class_stats:
                per_class_stats[intended]["correct_rejection"] += 1

        if intended == "OTHER":
            other_support += 1
            if rej_pred in TARGET_ACTION_CLASSES:
                other_to_target_count += 1

    raw_accuracy = (correct_raw / target_total) if target_total > 0 else 0.0
    rejection_accuracy = (correct_rejection / target_total) if target_total > 0 else 0.0
    rejection_rate = (rejections_count / target_total) if target_total > 0 else 0.0

    # Per-class recall and precision table
    # FIXED 9-CLASS DENOMINATOR: Never drop support-0 classes from Macro-F1 denominator
    per_class_recall: Dict[str, Dict[str, Any]] = {}
    all_9_f1: List[float] = []

    for cls_name in ALL_9_CLASSES:
        stat = per_class_stats[cls_name]
        supp = stat["support"]
        rec = (stat["correct_rejection"] / supp) if supp > 0 else 0.0
        pred_count = sum(1 for r in stable_target_records if r.get("rejected_prediction") == cls_name)
        prec = (stat["correct_rejection"] / pred_count) if pred_count > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        all_9_f1.append(f1)
        per_class_recall[cls_name] = {
            "support": supp,
            "detected_frames": stat["detected"],
            "detection_rate": (stat["detected"] / supp) if supp > 0 else 0.0,
            "correct_rejection": stat["correct_rejection"],
            "recall": rec,
            "precision": prec,
            "f1": f1,
        }

    macro_f1 = float(sum(all_9_f1) / 9.0)

    # Per-stage counts
    stages_breakdown = []
    for stage_info in GUIDED_STAGES:
        s_idx = stage_info["stage_index"]
        s_records = [r for r in frame_records if r.get("stage_index") == s_idx]
        s_prep = [r for r in s_records if r["phase"] == "PREPARATION"]
        s_obs = [r for r in s_records if r["phase"] == "OBSERVATION"]
        s_detected = sum(1 for r in s_obs if r["hand_detected"])
        stages_breakdown.append({
            "stage_index": s_idx,
            "stage_id": stage_info["stage_id"],
            "intended_label": stage_info["intended_label"],
            "display_name": stage_info["display_name"],
            "total_frames": len(s_records),
            "prep_frames": len(s_prep),
            "stable_frames": len(s_obs),
            "stable_hand_detected_frames": s_detected,
        })

    summary = {
        "task_id": "ACC-06",
        "description": "ACC-06 Bounded Shadow-Only Fresh Observation Trial Summary",
        "metadata": metadata,
        "frame_counts": {
            "total_frames_captured": len(frame_records),
            "waiting_frames": len(waiting_records),
            "preparation_frames": len(prep_records),
            "stable_observation_frames": len(stable_records),
            "stable_no_hand_frames": no_hand_total,
            "stable_target_gesture_frames": target_total,
        },
        "no_hand_baseline_metrics": {
            "total_stable_no_hand_frames": no_hand_total,
            "hand_detected_count": no_hand_detected,
            "hand_detection_rate": (no_hand_detected / no_hand_total) if no_hand_total > 0 else 0.0,
            "false_target_predictions_count": no_hand_false_target_predictions,
            "other_predictions_count": no_hand_other_predictions,
            "clean_no_hand_rate": ((no_hand_total - no_hand_detected) / no_hand_total) if no_hand_total > 0 else 0.0,
        },
        "target_gesture_metrics": {
            "total_target_support": target_total,
            "hand_detected_count": target_detected,
            "hand_detection_rate": target_detection_rate,
            "detector_loss_count": detector_loss_count,
            "rejections_count": rejections_count,
            "rejection_rate": rejection_rate,
            "raw_accuracy": raw_accuracy,
            "rejection_accuracy": rejection_accuracy,
            "macro_f1": macro_f1,
            "pinch_closed_recall": per_class_recall.get("PINCH_CLOSED", {}).get("recall", 0.0),
            "pinch_middle_recall": per_class_recall.get("PINCH_MIDDLE", {}).get("recall", 0.0),
            "other_to_target_count": other_to_target_count,
            "other_support": other_support,
            "other_to_target_rate": (other_to_target_count / other_support) if other_support > 0 else 0.0,
            "per_class": per_class_recall,
        },
        "stages_breakdown": stages_breakdown,
        "protocol_notes": {
            "disclaimer": "Start labels are declared intended poses, not independently verified physical ground truth.",
            "denominator_rule": "Detector losses (no-hand frames) during target stages are strictly INCLUDED in denominator as errors.",
            "macro_f1_rule": "Macro-F1 is strictly averaged over a fixed 9-class denominator (support-0 classes contribute 0.0).",
            "action_commands_dispatched": 0,
            "statistical_claims": "Zero frame-i.i.d. significance claims. Evaluated on single-user same-day capture take.",
        },
    }
    return summary


def run_shadow_trial(
    candidate_bundle_path: str = DEFAULT_CANDIDATE_PATH,
    task_model_path: str = DEFAULT_DETECTOR_TASK_PATH,
    output_jsonl: Optional[str] = None,
    output_summary: Optional[str] = None,
    user_id: str = "tuana",
    camera_id: int = 0,
    prep_duration: float = 3.0,
    obs_duration: float = 5.0,
    headless: bool = False,
    camera_stream: Optional[Iterator[Tuple[bool, np.ndarray]]] = None,
    key_provider: Optional[Callable[[], int]] = None,
    clock_provider: Optional[Callable[[], Tuple[float, float]]] = None,
    detector_override: Optional[Any] = None,
    window_title: str = "AirPad Shadow Trial",
) -> Dict[str, Any]:
    """Execute the bounded shadow-only guided observation trial."""
    # 0. Safety pre-flight: check forbidden CLI flags
    check_forbidden_cli_args()

    # 1. Validate parameters (finite positive durations, non-blank user) BEFORE outputs or camera
    validate_trial_parameters(user_id=user_id, prep_duration=prep_duration, obs_duration=obs_duration)

    # 2. Check test harness fixtures and provenance
    has_test_fixtures = (
        camera_stream is not None
        or detector_override is not None
        or clock_provider is not None
    )

    if has_test_fixtures:
        provenance = "synthetic_test_harness"
        if camera_stream is None and (detector_override is not None or clock_provider is not None):
            raise ValueError(
                "[SAFETY FAIL-CLOSED] Mixing real-camera acquisition with test harness/detector overrides is strictly forbidden. "
                "Real-camera execution must run directly without test overrides."
            )
    else:
        provenance = "live_camera_observation"

    # Prevent unsupported unattended real-camera headless mode
    if headless and camera_stream is None and key_provider is None:
        raise ValueError(
            "[SAFETY FAIL-CLOSED] Unattended real-camera headless mode without key/test harness is unsupported. "
            "Trial starts in WAITING phase requiring interactive user SPACE key."
        )

    # Generate default output paths under reports/acc06 if not provided
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"shadow_obs_{timestamp_str}_{uuid.uuid4().hex[:6]}"

    acc06_dir = os.path.join(PROJECT_ROOT, "reports", "acc06")
    os.makedirs(acc06_dir, exist_ok=True)

    output_jsonl = output_jsonl or os.path.join(acc06_dir, f"trial_{session_id}.jsonl")
    output_summary = output_summary or os.path.join(acc06_dir, f"summary_{session_id}.json")

    # 3. Path guards and existence validations BEFORE reserving outputs or opening camera
    validate_shadow_trial_paths(
        output_jsonl=output_jsonl,
        output_summary=output_summary,
        candidate_bundle_path=candidate_bundle_path,
        task_model_path=task_model_path,
        baseline_path=DEFAULT_BASELINE_PATH,
    )

    # 4. Verify candidate hash and load model bundle BEFORE reserving output handles
    # Real run always loads and uses the verified candidate bundle
    classifier, scaler, candidate_bundle = validate_and_load_candidate_bundle(
        candidate_bundle_path,
        expected_sha256=PINNED_CANDIDATE_SHA256,
    )

    detector_hash = compute_file_sha256(task_model_path)
    candidate_hash = compute_file_sha256(candidate_bundle_path)

    # Clock provider: returns (wall_clock, monotonic_clock)
    def get_clocks() -> Tuple[float, float]:
        if clock_provider is not None:
            return clock_provider()
        return (time.time(), time.monotonic())

    wall_start, mono_start = get_clocks()
    start_time_iso = datetime.datetime.now().isoformat()

    trial_metadata = {
        "session_id": session_id,
        "user_id": user_id,
        "role": "shadow_observation",
        "provenance": provenance,
        "created_at": start_time_iso,
        "candidate_bundle_path": os.path.relpath(candidate_bundle_path, PROJECT_ROOT).replace("\\", "/"),
        "candidate_bundle_sha256": candidate_hash,
        "detector_model_path": os.path.relpath(task_model_path, PROJECT_ROOT).replace("\\", "/"),
        "detector_model_sha256": detector_hash,
        "decision_threshold": DECISION_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "prep_duration_sec": prep_duration,
        "obs_duration_sec": obs_duration,
        "total_guided_stages": len(GUIDED_STAGES),
        "commands_dispatched": 0,
        "status": "INITIALIZING",
    }

    # 5. Exclusive reservation of BOTH output handles using mode 'x' BEFORE camera/detector
    # Reservation of the pair is sequential; if the second open fails, the first is cleaned up.
    f_jsonl, f_summary = reserve_output_handles(output_jsonl, output_summary)

    # 6. Protect EVERY operation after handle reservation with try/finally
    cap = None
    detector = detector_override
    records: List[Dict[str, Any]] = []
    trial_status = "INCOMPLETE"
    status_reason = "Run in progress"
    summary_data: Optional[Dict[str, Any]] = None
    write_error: Optional[Exception] = None

    try:
        # Write and flush metadata HEADER before the first frame
        header_record = {
            "record_type": "header",
            "session_id": session_id,
            "user_id": user_id,
            "role": "shadow_observation",
            "provenance": provenance,
            "candidate_bundle_path": trial_metadata["candidate_bundle_path"],
            "candidate_bundle_sha256": candidate_hash,
            "detector_model_path": trial_metadata["detector_model_path"],
            "detector_model_sha256": detector_hash,
            "decision_threshold": DECISION_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "prep_duration_sec": prep_duration,
            "obs_duration_sec": obs_duration,
            "guided_stages": GUIDED_STAGES,
            "created_at": start_time_iso,
            "commands_dispatched": 0,
        }
        f_jsonl.write(json.dumps(header_record) + "\n")
        f_jsonl.flush()
        records.append(header_record)

        # 7. Initialize detector and camera
        if detector is None:
            detector = MediaPipeAdapter(
                max_num_hands=1,
                min_detection_confidence=0.5,
                task_model_path=task_model_path,
            )

        if camera_stream is None:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"[FAIL-CLOSED] Unable to open camera device index {camera_id}")

        # 8. State machine variables
        phase = "WAITING"
        current_stage_idx = 0
        phase_start_mono = mono_start
        frame_index = 0
        last_pred = "NONE"
        last_margin = None

        while True:
            # Acquire frame
            if camera_stream is not None:
                try:
                    ret, bgr_frame = next(camera_stream)
                except StopIteration:
                    status_reason = "Camera stream ended"
                    break
            else:
                ret, bgr_frame = cap.read()

            if not ret or bgr_frame is None:
                status_reason = "Camera read failure"
                break

            wall_now, mono_now = get_clocks()

            # Preprocessing: horizontal flip + explicit BGR->RGB conversion
            flipped_bgr, rgb_frame = pre_process_frame_bgr_to_rgb(bgr_frame)

            # Detect hand
            hands = detector.detect(rgb_frame)
            hand_detected = bool(len(hands) > 0)

            # Classify if hand detected
            if hand_detected:
                classif_result = classify_hand(
                    hands[0],
                    classifier,
                    scaler,
                    decision_threshold=DECISION_THRESHOLD,
                    margin_threshold=MARGIN_THRESHOLD,
                )
                raw_pred = classif_result["raw_prediction"]
                rej_pred = classif_result["rejected_prediction"]
                best_score = classif_result["best_score"]
                margin = classif_result["margin"]
                last_pred = rej_pred
                last_margin = margin
            else:
                classif_result = {
                    "raw_landmarks": None,
                    "canonical_landmarks": None,
                    "handedness": None,
                    "hand_confidence": None,
                    "raw_prediction": "NO_HAND",
                    "rejected_prediction": "NO_HAND",
                    "best_score": None,
                    "margin": None,
                    "all_scores": None,
                }
                raw_pred = "NO_HAND"
                rej_pred = "NO_HAND"
                best_score = None
                margin = None
                last_pred = "NO_HAND"
                last_margin = None

            # State transitions using monotonic time for countdown accuracy
            if phase == "WAITING":
                stage_info = {"stage_index": -1, "stage_id": 0, "intended_label": "WAITING", "display_name": "WAITING"}
                is_transition = True
                time_remaining = 0.0
            else:
                stage_info = GUIDED_STAGES[current_stage_idx]
                elapsed_in_phase_mono = mono_now - phase_start_mono

                if phase == "PREPARATION":
                    is_transition = True
                    time_remaining = prep_duration - elapsed_in_phase_mono
                    if elapsed_in_phase_mono >= prep_duration:
                        phase = "OBSERVATION"
                        phase_start_mono = mono_now
                        is_transition = False
                        time_remaining = obs_duration
                elif phase == "OBSERVATION":
                    is_transition = False
                    time_remaining = obs_duration - elapsed_in_phase_mono
                    if elapsed_in_phase_mono >= obs_duration:
                        current_stage_idx += 1
                        if current_stage_idx >= len(GUIDED_STAGES):
                            trial_status = "COMPLETED"
                            status_reason = "All 12 guided stages completed successfully"
                            time_remaining = 0.0
                        else:
                            phase = "PREPARATION"
                            phase_start_mono = mono_now
                            stage_info = GUIDED_STAGES[current_stage_idx]
                            is_transition = True
                            time_remaining = prep_duration

            # Log frame to JSONL with full session linkage
            frame_record = {
                "record_type": "frame",
                "session_id": session_id,
                "user_id": user_id,
                "frame_index": frame_index,
                "timestamp": wall_now,
                "stage_index": stage_info["stage_index"],
                "stage_id": stage_info["stage_id"],
                "stage_display_name": stage_info["display_name"],
                "intended_label": stage_info["intended_label"],
                "phase": phase,
                "is_transition": is_transition,
                "hand_detected": hand_detected,
                "handedness": classif_result["handedness"],
                "hand_confidence": classif_result["hand_confidence"],
                "raw_prediction": raw_pred,
                "rejected_prediction": rej_pred,
                "best_score": best_score,
                "margin": margin,
                "all_scores": classif_result["all_scores"],
                "raw_landmarks": classif_result["raw_landmarks"],
                "canonical_landmarks": classif_result["canonical_landmarks"],
                "commands_dispatched": 0,
            }
            records.append(frame_record)

            f_jsonl.write(json.dumps(frame_record) + "\n")
            f_jsonl.flush()
            frame_index += 1

            if trial_status == "COMPLETED":
                break

            # Handle UI and Key Input
            if key_provider is not None:
                key = key_provider()
            elif not headless:
                hud_frame = draw_hud_overlay(
                    frame_bgr=flipped_bgr,
                    stage_name=stage_info["display_name"],
                    phase=phase,
                    time_remaining=time_remaining,
                    current_prediction=last_pred,
                    current_margin=last_margin,
                    hand_detected=hand_detected,
                    commands_count=0,
                )
                cv2.imshow(window_title, hud_frame)

                # Check if window was closed by user
                try:
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                        status_reason = "User closed preview window"
                        break
                except Exception:
                    pass

                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            # Key actions
            if key in (27, ord("q"), ord("Q")):
                status_reason = "User pressed quit key (ESC/Q)"
                break
            elif key == ord(" ") and phase == "WAITING":
                phase = "PREPARATION"
                current_stage_idx = 0
                phase_start_mono = mono_now

    except Exception as e:
        trial_status = "ERROR"
        status_reason = f"Exception: {type(e).__name__}: {str(e)}"
        raise
    finally:
        # Resilient resource cleanup: ensure failures in one resource do not block others
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if detector is not None and hasattr(detector, "close"):
            try:
                detector.close()
            except Exception:
                pass
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        _safe_close(f_jsonl)

        # Update metadata and save summary into reserved handle
        trial_metadata["end_time_iso"] = datetime.datetime.now().isoformat()
        trial_metadata["trial_status"] = trial_status
        trial_metadata["status_reason"] = status_reason

        try:
            summary_data = compute_shadow_trial_summary(records, trial_metadata)
            if f_summary is not None and not getattr(f_summary, "closed", True):
                f_summary.write(json.dumps(summary_data, indent=2))
                f_summary.flush()
        except Exception as se:
            write_error = se
        finally:
            _safe_close(f_summary)

        if write_error is not None:
            raise RuntimeError(
                f"[FAIL-CLOSED] Failed to write summary JSON to '{output_summary}': {write_error}. "
                f"Handles closed; JSONL was preserved."
            ) from write_error

    if summary_data is None:
        raise RuntimeError(f"[FAIL-CLOSED] Trial completed without computing summary data for '{output_summary}'")

    return summary_data


def main() -> None:
    """CLI entrypoint for standalone shadow trial tool."""
    check_forbidden_cli_args()

    parser = argparse.ArgumentParser(
        description="AirPad ACC-06 Bounded Shadow-Only Guided Observation Trial (Zero Action Dispatch)."
    )
    parser.add_argument(
        "--candidate-bundle",
        default=DEFAULT_CANDIDATE_PATH,
        help=f"Path to candidate SVM bundle (default: {DEFAULT_CANDIDATE_PATH})",
    )
    parser.add_argument(
        "--task-model",
        default=DEFAULT_DETECTOR_TASK_PATH,
        help=f"Path to local MediaPipe HandLandmarker task model (default: {DEFAULT_DETECTOR_TASK_PATH})",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Path for incremental JSONL frame logs (auto-generated in reports/acc06 if omitted)",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Path for final trial summary JSON (auto-generated in reports/acc06 if omitted)",
    )
    parser.add_argument("--user-id", default="tuana", help="User identifier (default: tuana)")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument(
        "--prep-duration",
        type=float,
        default=3.0,
        help="Preparation duration in seconds per stage (default: 3.0)",
    )
    parser.add_argument(
        "--obs-duration",
        type=float,
        default=5.0,
        help="Observation duration in seconds per stage (default: 5.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV window",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("      AIRPAD ACC-06 SHADOW-ONLY TRIAL (ZERO DISPATCH)")
    print("=" * 70)
    print(f"Candidate: {args.candidate_bundle}")
    print(f"Detector:  {args.task_model}")
    print(f"User ID:   {args.user_id}")
    print(f"Timing:    Prep: {args.prep_duration}s, Obs: {args.obs_duration}s per stage")
    print("Safety:    Action controllers DISABLED by construction (0 commands)")
    print("=" * 70)

    summary = run_shadow_trial(
        candidate_bundle_path=args.candidate_bundle,
        task_model_path=args.task_model,
        output_jsonl=args.output_jsonl,
        output_summary=args.output_summary,
        user_id=args.user_id,
        camera_id=args.camera_id,
        prep_duration=args.prep_duration,
        obs_duration=args.obs_duration,
        headless=args.headless,
    )

    print("\n[TRIAL FINISHED]")
    print(f"Status:   {summary['metadata']['trial_status']} ({summary['metadata']['status_reason']})")
    print(f"Frames:   {summary['frame_counts']['total_frames_captured']} total frames logged")
    print(f"Summary:  {args.output_summary or 'reports/acc06'}")


if __name__ == "__main__":
    main()
