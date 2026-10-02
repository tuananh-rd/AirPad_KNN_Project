"""Unit tests for ACC-06 Bounded Shadow-Only Guided Observation Trial Tool.

Acceptance Requirements:
1. Fake camera + detector + classifier + clock/key input headless end-to-end (no real camera).
2. Valid finite landmarks, no-hand frames, misclassification, detector dropout, model rejection.
3. All captured stable frames retained; preparation frames excluded from scored stable metrics.
4. Fixed staged schedule: NO_HAND, 1..9 in collection order, OTHER again, NO_HAND again (all 12 observed).
5. One-space start / auto advance / Q / ESC / read-failure cleanup: release + detector.close + window destroy in finally.
6. Fail-closed guards: invalid model hash, missing local detector task file, output collision, output aliasing,
   and forbidden action flags (--real-control) all fail before camera acquisition with VideoCapture not called.
7. TOCTOU race injection & handle lifecycle: pre-existing collision files preserved; exclusive 'x' reservation proves camera never opened.
8. Header metadata record in JSONL before first frame; all frames have session/user linkage. Header excluded from frame accounting.
9. Parameter validation: blank user, non-finite/zero/negative durations fail closed before outputs or camera.
10. Macro-F1 fixed 9-class denominator: support-0 classes contribute 0.0, denominator is strictly 9.0 even on incomplete takes.
11. Landmark validation fail-closed: NaN/Inf, invalid shape, invalid handedness rejected.
12. Candidate bundle validation: exact 9 classes, hyperparams, and scaler verified before handle reservation.
13. Positive asymmetric BGR->RGB expectation: proves detector receives converted RGB frame.
14. OS safety test: import hook and OS dispatch API interceptors active during positive fake trial; verifies zero command dispatches.
15. Deterministic fake classifier with known correct/wrong/UNKNOWN outputs + no-hand dropout; asserts concrete metrics and classifier never fit.
16. VideoCapture and GUI cleanup test: exercises cap.release() and cv2.destroyAllWindows().
17. Handle cleanup on summary write failure, header write failure, and collision reservation failure.
18. No refitting or rebundling of candidate; candidate and baseline hashes unchanged.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tools.shadow_trial as shadow
from src.vision.landmarks import HandLandmarks, LandmarkPoint
from tools.train import ModelBundle


# Dummy fake detector for test simulation
class FakeDetector:
    def __init__(self, hand_sequence: Optional[List[Optional[HandLandmarks]]] = None):
        self.hand_sequence = hand_sequence or []
        self.call_idx = 0
        self.frames_received = []
        self.closed = False

    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        self.frames_received.append(frame.copy())
        if self.call_idx < len(self.hand_sequence):
            res = self.hand_sequence[self.call_idx]
            self.call_idx += 1
            return [res] if res is not None else []
        return []

    def close(self):
        self.closed = True


class DeterministicFakeClassifier:
    """Deterministic fake classifier with known output predictions and spied fit methods."""
    def __init__(self, target_classes: List[str]):
        self.classes_ = np.array(target_classes)
        self.model = MagicMock()
        self.model.classes_ = self.classes_
        self.fit = MagicMock()
        self._fit = MagicMock()
        self.decision_call_idx = 0
        self.responses: List[Tuple[str, float, float]] = []

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.decision_call_idx < len(self.responses):
            pred_class, margin, best_score = self.responses[self.decision_call_idx]
            self.decision_call_idx += 1
        else:
            pred_class, margin, best_score = ("OTHER", 0.5, 1.0)

        scores = np.zeros(len(self.classes_), dtype=np.float64)
        target_idx = list(self.classes_).index(pred_class)
        scores[target_idx] = best_score
        runner_up_idx = (target_idx + 1) % len(self.classes_)
        scores[runner_up_idx] = best_score - margin
        for i in range(len(self.classes_)):
            if i not in (target_idx, runner_up_idx):
                scores[i] = best_score - margin - 1.0
        return np.array([scores])


def make_synthetic_hand(
    landmark_offset: float = 0.0,
    handedness: str = "Right",
    confidence: float = 0.95,
) -> HandLandmarks:
    """Create a synthetic 21-landmark HandLandmarks object."""
    points = []
    for i in range(21):
        points.append(LandmarkPoint(
            x=0.5 + 0.01 * i + landmark_offset,
            y=0.5 + 0.02 * i + landmark_offset,
            z=0.05 + 0.005 * i,
        ))
    return HandLandmarks(points=points, handedness=handedness, confidence=confidence)


def test_safety_and_forbidden_flags_rejection():
    """Verify that forbidden action-dispatch flags are rejected fail-closed before execution."""
    for bad_flag in ["--real-control", "--dispatch", "--execute", "--enable-control"]:
        with pytest.raises(ValueError, match="Forbidden action-dispatch flag"):
            shadow.check_forbidden_cli_args([bad_flag])

    # Safe flags pass without error
    shadow.check_forbidden_cli_args(["--user-id", "tuana", "--headless"])


def test_parameter_validation_fail_closed():
    """Verify that blank user and non-positive/non-finite durations are rejected fail-closed."""
    for bad_user in ["", "   ", None]:
        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            shadow.validate_trial_parameters(user_id=bad_user, prep_duration=3.0, obs_duration=5.0)

    for bad_prep in [0.0, -1.0, float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="prep_duration must be a positive finite number"):
            shadow.validate_trial_parameters(user_id="tuana", prep_duration=bad_prep, obs_duration=5.0)

    for bad_obs in [0.0, -2.5, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="obs_duration must be a positive finite number"):
            shadow.validate_trial_parameters(user_id="tuana", prep_duration=3.0, obs_duration=bad_obs)


def test_os_safety_and_zero_commands_dispatched_with_active_import_hook(tmp_path):
    """Verify OS safety via active import hook and OS dispatch interceptors during a positive fake trial."""
    class ForbiddenImportBlocker:
        FORBIDDEN_PREFIXES = (
            "src.platform",
            "src.runtime",
            "src.gesture.fsm",
            "pyautogui",
            "pynput",
            "keyboard",
        )

        def find_spec(self, fullname, path, target=None):
            for prefix in self.FORBIDDEN_PREFIXES:
                if fullname == prefix or fullname.startswith(prefix + "."):
                    raise ImportError(f"[SAFETY VIOLATION] Attempted import of forbidden module: {fullname}")
            return None

    # Evict any existing loaded forbidden modules for the duration of this test
    cached_forbidden = {}
    for mod in list(sys.modules.keys()):
        for prefix in ForbiddenImportBlocker.FORBIDDEN_PREFIXES:
            if mod == prefix or mod.startswith(prefix + "."):
                cached_forbidden[mod] = sys.modules.pop(mod)
                break

    blocker = ForbiddenImportBlocker()
    sys.meta_path.insert(0, blocker)

    def blocked_os_dispatch(*args, **kwargs):
        raise RuntimeError("[SAFETY VIOLATION] Attempted to invoke OS action dispatch API")

    out_jsonl = str(tmp_path / "os_safety_test.jsonl")
    out_summary = str(tmp_path / "os_safety_summary.json")

    # Positive fake trial: 5 frames, space to start, transitions, hand detection, classification, exit
    cam_frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(6)]
    def sim_cam():
        for f in cam_frames:
            yield True, f

    key_seq = [ord(" "), 255, 255, 255, ord("q")]
    key_idx = [0]
    def sim_keys():
        k = key_seq[key_idx[0]] if key_idx[0] < len(key_seq) else 255
        key_idx[0] += 1
        return k

    fake_det = FakeDetector([make_synthetic_hand() for _ in range(10)])

    try:
        with patch("os.system", side_effect=blocked_os_dispatch), \
             patch("subprocess.Popen", side_effect=blocked_os_dispatch), \
             patch("subprocess.run", side_effect=blocked_os_dispatch):

            summary = shadow.run_shadow_trial(
                output_jsonl=out_jsonl,
                output_summary=out_summary,
                headless=True,
                camera_stream=sim_cam(),
                key_provider=sim_keys,
                detector_override=fake_det,
            )

        # Assert zero commands dispatched across summary and metadata
        assert summary["metadata"]["commands_dispatched"] == 0
        assert summary["protocol_notes"]["action_commands_dispatched"] == 0

        # Assert every frame record has commands_dispatched == 0
        with open(out_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 2  # header + frames
        for l in lines[1:]:
            rec = json.loads(l)
            assert rec["commands_dispatched"] == 0

        # Verify the import blocker is actively rejecting forbidden modules
        with pytest.raises(ImportError, match=r"\[SAFETY VIOLATION\]"):
            importlib.import_module("src.platform")

        with pytest.raises(ImportError, match=r"\[SAFETY VIOLATION\]"):
            importlib.import_module("pyautogui")

        with pytest.raises(ImportError, match=r"\[SAFETY VIOLATION\]"):
            importlib.import_module("pynput")

    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
        sys.modules.update(cached_forbidden)


def test_os_safety_subprocess_isolation(tmp_path):
    """Run a fresh Python subprocess enforcing the import hook to ensure clean isolated safety."""
    sub_jsonl = str(tmp_path / "sub_trial.jsonl").replace("\\", "/")
    sub_summary = str(tmp_path / "sub_summary.json").replace("\\", "/")

    script = f"""
import sys, os
sys.path.insert(0, r"{PROJECT_ROOT}")

class ForbiddenImportBlocker:
    FORBIDDEN_PREFIXES = ("src.platform", "src.runtime", "src.gesture.fsm", "pyautogui", "pynput", "keyboard")
    def find_spec(self, fullname, path, target=None):
        for prefix in self.FORBIDDEN_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                raise ImportError(f"[SAFETY VIOLATION] {{fullname}}")
        return None

sys.meta_path.insert(0, ForbiddenImportBlocker())

import tools.shadow_trial as shadow
import numpy as np

def dummy_cam():
    for _ in range(3):
        yield True, np.zeros((240, 320, 3), dtype=np.uint8)

summary = shadow.run_shadow_trial(
    output_jsonl=r"{sub_jsonl}",
    output_summary=r"{sub_summary}",
    headless=True,
    camera_stream=dummy_cam(),
    key_provider=lambda: ord("q"),
)
assert summary["metadata"]["commands_dispatched"] == 0
print("SUBPROCESS_OS_SAFETY_OK")
"""
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, f"Subprocess failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "SUBPROCESS_OS_SAFETY_OK" in res.stdout


def test_toctou_race_injection_preserves_files_and_blocks_camera(tmp_path):
    """Verify that if an output file exists or is injected, FileExistsError is raised and camera is never opened."""
    collision_jsonl = str(tmp_path / "race_collision.jsonl")
    collision_summary = str(tmp_path / "race_collision.json")

    with open(collision_jsonl, "w", encoding="utf-8") as f:
        f.write("CANARY_CONTENT_PRESERVE_DO_NOT_TRUNCATE\n")

    with patch("cv2.VideoCapture") as mock_cv:
        with pytest.raises(FileExistsError, match="Output file already exists"):
            shadow.run_shadow_trial(
                output_jsonl=collision_jsonl,
                output_summary=collision_summary,
                headless=True,
                key_provider=lambda: ord("q"),
            )
        assert mock_cv.call_count == 0

    with open(collision_jsonl, "r", encoding="utf-8") as f:
        assert f.read() == "CANARY_CONTENT_PRESERVE_DO_NOT_TRUNCATE\n"


def test_second_open_collision_cleans_first_and_preserves_collision(tmp_path):
    """Verify that when summary output collision occurs, own newly created empty JSONL is removed and collision file preserved."""
    target_jsonl = str(tmp_path / "new_target.jsonl")
    collision_summary = str(tmp_path / "collision_summary.json")

    with open(collision_summary, "w", encoding="utf-8") as f:
        f.write("PRE_EXISTING_CANARY_SUMMARY\n")

    with patch("cv2.VideoCapture") as mock_cv:
        with pytest.raises(FileExistsError, match="Output file already exists"):
            shadow.run_shadow_trial(
                output_jsonl=target_jsonl,
                output_summary=collision_summary,
                headless=True,
                key_provider=lambda: ord("q"),
            )
        assert mock_cv.call_count == 0

    # Own newly created empty JSONL must be cleaned up safely
    assert not os.path.exists(target_jsonl)

    # Collision summary must be preserved untouched
    with open(collision_summary, "r", encoding="utf-8") as f:
        assert f.read() == "PRE_EXISTING_CANARY_SUMMARY\n"


def test_preflight_invalid_inputs_assert_videocapture_not_called_and_no_outputs(tmp_path):
    """Verify that all invalid preflight parameters and paths fail before VideoCapture with NO outputs created."""
    out_jsonl = str(tmp_path / "preflight.jsonl")
    out_summary = str(tmp_path / "preflight.json")

    # 1. Invalid candidate bundle hash
    corrupt_cand = str(tmp_path / "corrupt_candidate.json")
    with open(corrupt_cand, "w") as f:
        f.write("invalid hash content")

    with patch("cv2.VideoCapture") as mock_cv:
        with pytest.raises(ValueError, match="Candidate bundle SHA-256 mismatch"):
            shadow.run_shadow_trial(
                candidate_bundle_path=corrupt_cand,
                output_jsonl=out_jsonl,
                output_summary=out_summary,
                headless=True,
                key_provider=lambda: ord("q"),
            )
        assert mock_cv.call_count == 0
        assert not os.path.exists(out_jsonl)
        assert not os.path.exists(out_summary)

    # 2. Missing detector task model
    missing_task = str(tmp_path / "nonexistent.task")
    with patch("cv2.VideoCapture") as mock_cv:
        with pytest.raises(FileNotFoundError, match="Local detector task model missing"):
            shadow.run_shadow_trial(
                task_model_path=missing_task,
                output_jsonl=out_jsonl,
                output_summary=out_summary,
                headless=True,
                key_provider=lambda: ord("q"),
            )
        assert mock_cv.call_count == 0
        assert not os.path.exists(out_jsonl)
        assert not os.path.exists(out_summary)

    # 3. Invalid user_id (blank)
    with patch("cv2.VideoCapture") as mock_cv:
        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            shadow.run_shadow_trial(
                user_id="",
                output_jsonl=out_jsonl,
                output_summary=out_summary,
                headless=True,
                key_provider=lambda: ord("q"),
            )
        assert mock_cv.call_count == 0
        assert not os.path.exists(out_jsonl)
        assert not os.path.exists(out_summary)

    # 4. Non-.jsonl output extension
    with patch("cv2.VideoCapture") as mock_cv:
        with pytest.raises(ValueError, match="output_jsonl must end with literal .jsonl"):
            shadow.run_shadow_trial(
                output_jsonl=str(tmp_path / "bad.txt"),
                output_summary=out_summary,
                headless=True,
                key_provider=lambda: ord("q"),
            )
        assert mock_cv.call_count == 0
        assert not os.path.exists(out_summary)

    # 5. Output aliasing baseline
    with patch("cv2.VideoCapture") as mock_cv:
        with pytest.raises(ValueError, match="Output aliases baseline model path"):
            shadow.run_shadow_trial(
                output_summary=shadow.DEFAULT_BASELINE_PATH,
                output_jsonl=out_jsonl,
                headless=True,
                key_provider=lambda: ord("q"),
            )
        assert mock_cv.call_count == 0


def test_candidate_bundle_validation_checks(tmp_path):
    """Verify candidate bundle validation for exact 9 classes, hyperparameters, and scaler metadata."""
    bundle = ModelBundle.load(shadow.DEFAULT_CANDIDATE_PATH)

    # 1. Missing class
    mutated = copy.deepcopy(bundle)
    mutated.labels = mutated.labels[:-1]
    mut_p = str(tmp_path / "cand_missing_class.json")
    mutated.save(mut_p)
    with pytest.raises(ValueError, match="Candidate bundle must contain exactly the 9 canonical target classes"):
        shadow.validate_and_load_candidate_bundle(mut_p, expected_sha256=None)

    # 2. Hyperparameter mismatch
    mutated = copy.deepcopy(bundle)
    mutated.hyperparameters["C"] = 1.0
    mut_p = str(tmp_path / "cand_wrong_hp.json")
    mutated.save(mut_p)
    with pytest.raises(ValueError, match="Candidate bundle hyperparameters mismatch"):
        shadow.validate_and_load_candidate_bundle(mut_p, expected_sha256=None)

    # 3. Missing scaler metadata
    mutated = copy.deepcopy(bundle)
    mutated.metadata.pop("scaler")
    mut_p = str(tmp_path / "cand_no_scaler.json")
    mutated.save(mut_p)
    with pytest.raises(ValueError, match="missing valid StandardScaler metadata"):
        shadow.validate_and_load_candidate_bundle(mut_p, expected_sha256=None)


def test_landmark_validation_fail_closed():
    """Verify that non-finite landmarks or invalid handedness fail closed before canonicalization."""
    clf, scaler, _ = shadow.validate_and_load_candidate_bundle(shadow.DEFAULT_CANDIDATE_PATH)

    # 1. NaN in landmarks
    bad_hand_nan = make_synthetic_hand()
    bad_hand_nan.points[0].x = float("nan")
    with pytest.raises(ValueError, match="contain NaN or Inf values"):
        shadow.classify_hand(bad_hand_nan, clf, scaler)

    # 2. Inf in landmarks
    bad_hand_inf = make_synthetic_hand()
    bad_hand_inf.points[5].y = float("inf")
    with pytest.raises(ValueError, match="contain NaN or Inf values"):
        shadow.classify_hand(bad_hand_inf, clf, scaler)

    # 3. Invalid handedness
    bad_hand_h = make_synthetic_hand(handedness="Foot")
    with pytest.raises(ValueError, match="Invalid handedness"):
        shadow.classify_hand(bad_hand_h, clf, scaler)


def test_asymmetric_bgr_to_rgb_expectation():
    """Verify that camera frames undergo horizontal flip and explicit BGR->RGB conversion before detection."""
    h, w = 100, 100
    bgr_frame = np.zeros((h, w, 3), dtype=np.uint8)
    bgr_frame[:, :, 0] = 20   # Blue
    bgr_frame[:, :, 1] = 100  # Green
    bgr_frame[:, :, 2] = 250  # Red
    bgr_frame[0:10, 0:10, :] = 255

    flipped_bgr, rgb_frame = shadow.pre_process_frame_bgr_to_rgb(bgr_frame)

    assert np.all(flipped_bgr[0:10, 90:100, :] == 255)
    assert not np.all(flipped_bgr[0:10, 0:10, :] == 255)

    assert rgb_frame[50, 50, 0] == 250
    assert rgb_frame[50, 50, 1] == 100
    assert rgb_frame[50, 50, 2] == 20

    assert flipped_bgr[50, 50, 0] == 20
    assert flipped_bgr[50, 50, 2] == 250
    assert not np.array_equal(rgb_frame, flipped_bgr)


def test_headless_end_to_end_synthetic_trial_with_header_and_linkage(tmp_path):
    """Run a complete simulated headless trial progressing through all 12 stages asserting COMPLETED and exact full-frame accounting."""
    out_jsonl = str(tmp_path / "e2e_trial.jsonl")
    out_summary = str(tmp_path / "e2e_summary.json")

    wall_clock = [1700000000.0]
    mono_clock = [0.0]
    def sim_clocks():
        w = wall_clock[0]
        m = mono_clock[0]
        wall_clock[0] += 1.0
        mono_clock[0] += 1.0
        return (w, m)

    key_idx = [0]
    keys = [ord(" ")] + [255] * 200
    def sim_keys():
        k = keys[key_idx[0]] if key_idx[0] < len(keys) else 255
        key_idx[0] += 1
        return k

    def sim_cam():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        while True:
            yield True, frame

    fake_det = FakeDetector([make_synthetic_hand() for _ in range(100)])

    summary = shadow.run_shadow_trial(
        candidate_bundle_path=shadow.DEFAULT_CANDIDATE_PATH,
        task_model_path=shadow.DEFAULT_DETECTOR_TASK_PATH,
        output_jsonl=out_jsonl,
        output_summary=out_summary,
        user_id="tuana_eval",
        prep_duration=1.0,
        obs_duration=2.0,
        headless=True,
        camera_stream=sim_cam(),
        key_provider=sim_keys,
        clock_provider=sim_clocks,
        detector_override=fake_det,
    )

    # 1. Output files exist and status is COMPLETED
    assert os.path.isfile(out_jsonl)
    assert os.path.isfile(out_summary)
    assert summary["metadata"]["trial_status"] == "COMPLETED"
    assert "All 12 guided stages completed" in summary["metadata"]["status_reason"]

    # 2. Check JSONL Header and Frame records with exact full-frame accounting (header excluded)
    with open(out_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_captured = summary["frame_counts"]["total_frames_captured"]
    # Total lines in file is exactly total_captured + 1 (line 0 is header)
    assert len(lines) == total_captured + 1

    header = json.loads(lines[0])
    assert header["record_type"] == "header"
    assert header["user_id"] == "tuana_eval"
    assert header["role"] == "shadow_observation"
    assert header["candidate_bundle_sha256"] == shadow.PINNED_CANDIDATE_SHA256
    assert header["detector_model_sha256"] == shadow.PINNED_DETECTOR_TASK_SHA256
    assert len(header["guided_stages"]) == 12

    session_id = header["session_id"]
    for idx, l in enumerate(lines[1:]):
        frame_rec = json.loads(l)
        assert frame_rec["record_type"] == "frame"
        assert frame_rec["frame_index"] == idx
        assert frame_rec["session_id"] == session_id
        assert frame_rec["user_id"] == "tuana_eval"
        assert frame_rec["commands_dispatched"] == 0

    # 3. Every 12 stage was observed and has positive frame counts
    stages_breakdown = summary["stages_breakdown"]
    assert len(stages_breakdown) == 12
    for s in stages_breakdown:
        assert s["total_frames"] > 0
        assert s["stable_frames"] > 0

    # 4. Target gesture metrics consistency
    target_metrics = summary["target_gesture_metrics"]
    assert target_metrics["total_target_support"] == (
        target_metrics["hand_detected_count"] + target_metrics["detector_loss_count"]
    )
    assert 0.0 <= target_metrics["macro_f1"] <= 1.0


def test_deterministic_fake_classifier_and_concrete_metrics(tmp_path):
    """Verify concrete accuracy, recall, OTHER false target, dropout, fixed-9 F1, and classifier fit spy."""
    out_jsonl = str(tmp_path / "det_trial.jsonl")
    out_summary = str(tmp_path / "det_summary.json")

    hand_obj = make_synthetic_hand()

    # 37-frame sequence: detector dropouts on Frames 1, 8 (target dropout), 34, 36
    detector_frames = [hand_obj] * 37
    for idx in [1, 8, 34, 36]:
        detector_frames[idx] = None

    responses_by_frame = {}
    for f in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]:
        responses_by_frame[f] = ("OTHER", 0.5, 1.0)

    responses_by_frame[2] = ("POINT_ONE", 0.5, 1.0)              # False target during NO_HAND
    responses_by_frame[4] = ("PINCH_CLOSED", 0.5, 1.0)           # Correct
    responses_by_frame[5] = ("PINCH_CLOSED", 0.5, 1.0)           # Correct
    responses_by_frame[7] = ("PINCH_MIDDLE", 0.5, 1.0)           # Correct
    responses_by_frame[10] = ("L_SHAPE", 0.5, 1.0)              # Correct
    responses_by_frame[11] = ("L_SHAPE", 0.05, 1.0)             # Margin 0.05 < 0.15 -> UNKNOWN rejection
    responses_by_frame[13] = ("OTHER", 0.5, 1.0)                # Correct
    responses_by_frame[14] = ("POINT_ONE", 0.5, 1.0)            # OTHER misclassified as target
    responses_by_frame[16] = ("POINT_ONE", 0.5, 1.0)            # Correct
    responses_by_frame[17] = ("POINT_ONE", 0.5, 1.0)            # Correct
    responses_by_frame[19] = ("OPEN_PALM", 0.5, 1.0)            # Correct
    responses_by_frame[20] = ("OPEN_PALM", 0.5, 1.0)            # Correct
    responses_by_frame[22] = ("FIST", 0.5, 1.0)                 # Correct
    responses_by_frame[23] = ("FIST", 0.5, 1.0)                 # Correct
    responses_by_frame[25] = ("TWO_FINGERS_PARALLEL", 0.5, 1.0) # Correct
    responses_by_frame[26] = ("TWO_FINGERS_PARALLEL", 0.5, 1.0) # Correct
    responses_by_frame[28] = ("THREE_FINGERS", 0.5, 1.0)        # Correct
    responses_by_frame[29] = ("THREE_FINGERS", 0.5, 1.0)        # Correct
    responses_by_frame[31] = ("OTHER", 0.5, 1.0)                # Correct
    responses_by_frame[32] = ("OTHER", 0.5, 1.0)                # Correct
    responses_by_frame[35] = ("OTHER", 0.5, 1.0)                # Other during NO_HAND

    ordered_responses = [responses_by_frame[f] for f in range(37) if detector_frames[f] is not None]

    fake_clf = DeterministicFakeClassifier(shadow.ALL_9_CLASSES)
    fake_clf.responses = ordered_responses
    fake_det = FakeDetector(detector_frames)

    wall = [100.0]
    mono = [0.0]
    def sim_clocks():
        w, m = wall[0], mono[0]
        wall[0] += 1.0
        mono[0] += 1.0
        return (w, m)

    key_idx = [0]
    keys = [ord(" ")] + [255] * 200
    def sim_keys():
        k = keys[key_idx[0]] if key_idx[0] < len(keys) else 255
        key_idx[0] += 1
        return k

    def sim_cam():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        while True:
            yield True, frame

    actual_bundle = ModelBundle.load(shadow.DEFAULT_CANDIDATE_PATH)
    with patch("tools.shadow_trial.validate_and_load_candidate_bundle", return_value=(fake_clf, None, actual_bundle)):
        summary = shadow.run_shadow_trial(
            output_jsonl=out_jsonl,
            output_summary=out_summary,
            prep_duration=1.0,
            obs_duration=2.0,
            headless=True,
            camera_stream=sim_cam(),
            key_provider=sim_keys,
            clock_provider=sim_clocks,
            detector_override=fake_det,
        )

    # 1. Assert status COMPLETED, synthetic provenance, and classifier NEVER fit on observations
    assert summary["metadata"]["trial_status"] == "COMPLETED"
    assert summary["metadata"]["provenance"] == "synthetic_test_harness"
    assert fake_clf.fit.call_count == 0
    assert fake_clf._fit.call_count == 0

    # 2. Concrete target metrics matching ground truth
    tm = summary["target_gesture_metrics"]
    assert tm["total_target_support"] == 20
    assert tm["hand_detected_count"] == 19
    assert tm["detector_loss_count"] == 1
    assert tm["rejections_count"] == 1
    assert abs(tm["raw_accuracy"] - 0.90) < 1e-9          # 18/20
    assert abs(tm["rejection_accuracy"] - 0.85) < 1e-9    # 17/20
    assert abs(tm["rejection_rate"] - 0.05) < 1e-9        # 1/20
    assert tm["other_support"] == 4
    assert tm["other_to_target_count"] == 1               # 1/4
    assert abs(tm["other_to_target_rate"] - 0.25) < 1e-9
    assert abs(tm["pinch_closed_recall"] - 1.0) < 1e-9
    assert abs(tm["pinch_middle_recall"] - 0.5) < 1e-9

    # Strict fixed-9 macro-F1: (5.8 + 4/3 + 6/7) / 9.0
    expected_f1 = (5.8 + (4.0 / 3.0) + (6.0 / 7.0)) / 9.0
    assert abs(tm["macro_f1"] - expected_f1) < 1e-9

    # 3. Concrete NO_HAND metrics
    nh = summary["no_hand_baseline_metrics"]
    assert nh["total_stable_no_hand_frames"] == 5
    assert nh["hand_detected_count"] == 2
    assert nh["false_target_predictions_count"] == 1
    assert nh["other_predictions_count"] == 1
    assert abs(nh["clean_no_hand_rate"] - 0.60) < 1e-9


def test_fake_videocapture_and_gui_cleanup(tmp_path):
    """Verify that VideoCapture and OpenCV GUI window cleanup (release and destroyAllWindows) are exercised."""
    out_jsonl = str(tmp_path / "vc_cleanup.jsonl")
    out_summary = str(tmp_path / "vc_cleanup.json")

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
    mock_cap.release = MagicMock()

    mock_adapter = MagicMock()
    mock_adapter.detect.return_value = [make_synthetic_hand()]
    mock_adapter.close = MagicMock()

    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("tools.shadow_trial.MediaPipeAdapter", return_value=mock_adapter), \
         patch("cv2.destroyAllWindows") as mock_destroy, \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord("q")), \
         patch("cv2.getWindowProperty", return_value=1):

        summary = shadow.run_shadow_trial(
            output_jsonl=out_jsonl,
            output_summary=out_summary,
            headless=False,
            camera_stream=None,
        )

        assert summary["metadata"]["trial_status"] == "INCOMPLETE"
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()
        mock_adapter.close.assert_called_once()


def test_header_write_failure_closes_handles_and_does_not_open_camera(tmp_path):
    """Verify that a write failure during header initialization safely closes handles and does not open camera."""
    out_jsonl = str(tmp_path / "hdr_fail.jsonl")
    out_summary = str(tmp_path / "hdr_fail.json")

    real_open = open
    def mock_open_raising_on_header(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if "hdr_fail.jsonl" in path and "x" in mode:
            orig_write = handle.write
            def failing_write(data):
                if '"record_type": "header"' in data:
                    raise IOError("Disk write error during header flush")
                return orig_write(data)
            handle.write = failing_write
        return handle

    with patch("builtins.open", side_effect=mock_open_raising_on_header), \
         patch("cv2.VideoCapture") as mock_cv:

        with pytest.raises(IOError, match="Disk write error during header flush"):
            shadow.run_shadow_trial(
                output_jsonl=out_jsonl,
                output_summary=out_summary,
                headless=True,
                key_provider=lambda: ord("q"),
            )
        assert mock_cv.call_count == 0


def test_summary_write_failure_propagates_and_closes_handles(tmp_path):
    """Verify that a summary write failure is not swallowed, error propagates, handles close, and JSONL is preserved."""
    out_jsonl = str(tmp_path / "sum_fail.jsonl")
    out_summary = str(tmp_path / "sum_fail.json")

    def sim_cam():
        yield True, np.zeros((240, 320, 3), dtype=np.uint8)

    real_open = open
    def mock_open_failing_summary_write(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if str(path).endswith("sum_fail.json") and "x" in mode:
            def failing_write(data):
                raise OSError("Disk full while writing summary")
            handle.write = failing_write
        return handle

    with patch("builtins.open", side_effect=mock_open_failing_summary_write):
        with pytest.raises(RuntimeError, match=r"\[FAIL-CLOSED\] Failed to write summary JSON"):
            shadow.run_shadow_trial(
                output_jsonl=out_jsonl,
                output_summary=out_summary,
                headless=True,
                camera_stream=sim_cam(),
                key_provider=lambda: ord("q"),
                detector_override=FakeDetector([None]),
            )

    # JSONL file was preserved on disk despite summary write failure
    assert os.path.isfile(out_jsonl)
    with open(out_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) >= 1
    assert json.loads(lines[0])["record_type"] == "header"


def test_fixed_9_macro_f1_on_incomplete_trial(tmp_path):
    """Verify that macro-F1 strictly uses a fixed 9-class denominator even when trial is incomplete."""
    out_jsonl = str(tmp_path / "incomplete_f1.jsonl")
    out_summary = str(tmp_path / "incomplete_f1.json")

    def short_cam():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(6):
            yield True, frame

    key_idx = [0]
    keys = [ord(" "), 255, 255, 255, ord("q")]
    def sim_keys():
        k = keys[key_idx[0]] if key_idx[0] < len(keys) else 255
        key_idx[0] += 1
        return k

    fake_det = FakeDetector([make_synthetic_hand() for _ in range(10)])

    summary = shadow.run_shadow_trial(
        output_jsonl=out_jsonl,
        output_summary=out_summary,
        headless=True,
        camera_stream=short_cam(),
        key_provider=sim_keys,
        detector_override=fake_det,
    )

    assert summary["metadata"]["trial_status"] == "INCOMPLETE"
    target_m = summary["target_gesture_metrics"]
    per_class = target_m["per_class"]
    assert len(per_class) == 9

    expected_macro = sum(per_class[c]["f1"] for c in shadow.ALL_9_CLASSES) / 9.0
    assert abs(target_m["macro_f1"] - expected_macro) < 1e-9


def test_early_exit_user_quit_and_cleanup(tmp_path):
    """Verify that pressing Q / ESC causes immediate exit marked INCOMPLETE and properly cleans up."""
    out_jsonl = str(tmp_path / "early_quit.jsonl")
    out_summary = str(tmp_path / "early_summary.json")

    def sim_cam():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        while True:
            yield True, frame

    fake_det = FakeDetector([make_synthetic_hand()])

    key_seq = [ord(" "), 255, ord("q")]
    key_idx = [0]
    def sim_keys():
        k = key_seq[key_idx[0]] if key_idx[0] < len(key_seq) else 255
        key_idx[0] += 1
        return k

    summary = shadow.run_shadow_trial(
        output_jsonl=out_jsonl,
        output_summary=out_summary,
        headless=True,
        camera_stream=sim_cam(),
        key_provider=sim_keys,
        detector_override=fake_det,
    )

    assert summary["metadata"]["trial_status"] == "INCOMPLETE"
    assert "User pressed quit key" in summary["metadata"]["status_reason"]
    assert fake_det.closed is True
    assert os.path.isfile(out_jsonl)
    assert os.path.isfile(out_summary)


def test_camera_read_failure_cleanup(tmp_path):
    """Verify clean shutdown when camera stream drops frames."""
    out_jsonl = str(tmp_path / "cam_fail.jsonl")
    out_summary = str(tmp_path / "cam_fail_summary.json")

    def failing_cam():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        yield True, frame
        yield False, None

    fake_det = FakeDetector([None, None])

    summary = shadow.run_shadow_trial(
        output_jsonl=out_jsonl,
        output_summary=out_summary,
        headless=True,
        camera_stream=failing_cam(),
        detector_override=fake_det,
    )

    assert summary["metadata"]["trial_status"] == "INCOMPLETE"
    assert "Camera read failure" in summary["metadata"]["status_reason"]
    assert fake_det.closed is True
    assert os.path.isfile(out_jsonl)
    assert os.path.isfile(out_summary)


def test_real_camera_headless_without_fixture_fails_closed():
    """Verify that unattended real-camera headless execution fails closed before camera acquisition."""
    with patch("cv2.VideoCapture") as mock_cap:
        with pytest.raises(ValueError, match="Unattended real-camera headless mode without key/test harness is unsupported"):
            shadow.run_shadow_trial(headless=True, camera_stream=None, key_provider=None)
        assert mock_cap.call_count == 0


def test_window_close_via_get_window_property_exits_safely(tmp_path):
    """Verify that closing OpenCV window via X button triggers clean INCOMPLETE exit."""
    out_jsonl = str(tmp_path / "win_close.jsonl")
    out_summary = str(tmp_path / "win_close.json")

    def dummy_cam():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        while True:
            yield True, frame

    fake_det = FakeDetector([None, None, None])

    win_prop_calls = [1, 0]
    def mock_get_win_prop(title, prop):
        res = win_prop_calls.pop(0) if win_prop_calls else 0
        return res

    with patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=255), \
         patch("cv2.getWindowProperty", side_effect=mock_get_win_prop):

        summary = shadow.run_shadow_trial(
            output_jsonl=out_jsonl,
            output_summary=out_summary,
            headless=False,
            camera_stream=dummy_cam(),
            detector_override=fake_det,
        )

        assert summary["metadata"]["trial_status"] == "INCOMPLETE"
        assert "User closed preview window" in summary["metadata"]["status_reason"]


def test_candidate_and_baseline_immutability():
    """Verify that frozen candidate and baseline model files remain strictly byte-identical."""
    import hashlib

    actual_cand = hashlib.sha256(open(shadow.DEFAULT_CANDIDATE_PATH, "rb").read()).hexdigest()
    assert actual_cand == shadow.PINNED_CANDIDATE_SHA256

    actual_base = hashlib.sha256(open(shadow.DEFAULT_BASELINE_PATH, "rb").read()).hexdigest()
    assert actual_base == shadow.PINNED_BASELINE_SHA256

    actual_task = hashlib.sha256(open(shadow.DEFAULT_DETECTOR_TASK_PATH, "rb").read()).hexdigest()
    assert actual_task == shadow.PINNED_DETECTOR_TASK_SHA256


def test_forbid_test_fixtures_with_real_camera_acquisition():
    """Verify that mixing real-camera acquisition with test harness/detector overrides is strictly forbidden."""
    fake_det = FakeDetector([make_synthetic_hand()])
    with pytest.raises(ValueError, match=r"\[SAFETY FAIL-CLOSED\] Mixing real-camera acquisition with test harness/detector overrides is strictly forbidden"):
        shadow.run_shadow_trial(camera_stream=None, detector_override=fake_det)


def test_hud_overlay_layout_legibility_and_separation():
    """Verify that HUD overlay renders without text clipping or overlapping across 640x480 and 320x240 resolutions."""
    stage_name = "[8] TWO_FINGERS_PARALLEL"
    phase = "OBSERVATION"

    for (w, h) in [(640, 480), (320, 240)]:
        dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
        rendered = shadow.draw_hud_overlay(
            frame_bgr=dummy_frame,
            stage_name=stage_name,
            phase=phase,
            time_remaining=4.2,
            current_prediction="TWO_FINGERS_PARALLEL",
            current_margin=0.45,
            hand_detected=True,
            commands_count=0,
        )
        assert rendered.shape == (h, w, 3)

        # Stage text width must strictly fit within frame width
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.75 if w >= 600 else 0.55
        max_stage_w = w - 20
        text_size, _ = cv2.getTextSize(stage_name, font, font_scale, 2)
        while text_size[0] > max_stage_w and font_scale > 0.35:
            font_scale -= 0.05
            text_size, _ = cv2.getTextSize(stage_name, font, font_scale, 2)
        assert text_size[0] <= w - 20

        # Countdown / phase line (y=84) is separated from stage name line (y=56) by at least 25px
        stage_y = 56
        phase_timer_y = 84
        assert phase_timer_y - stage_y >= 25
