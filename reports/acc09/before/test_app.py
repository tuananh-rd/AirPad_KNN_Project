"""Tests for app.py CLI commands and safety checks.

Covers:
- `--self-check` execution without webcam returning exit code 0.
- `--replay fixtures/synthetic_replay.json` execution in dry-run mode.
- `--real-control` printing safety warning banner.
- `--camera` graceful failure when dependencies are mocked as missing.
- Default `main()` with no arguments returning 0 and displaying help.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from app import main, run_self_check, run_replay, run_camera, create_pipeline, WARNING_BANNER


def test_self_check_succeeds():
    """--self-check must execute end-to-end without webcam and return 0."""
    exit_code = run_self_check()
    assert exit_code == 0


def test_main_cli_self_check():
    exit_code = main(["--self-check"])
    assert exit_code == 0


def test_main_cli_replay(tmp_path):
    fixture_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "fixtures", "synthetic_replay.json"
    )
    assert os.path.exists(fixture_path)
    exit_code = main(["--replay", fixture_path])
    assert exit_code == 0


def test_main_cli_no_args_prints_help(capsys):
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "options:" in captured.out.lower()


def test_real_control_prints_warning(capsys):
    fixture_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "fixtures", "synthetic_replay.json"
    )
    # Replay with --real-control: should print warning banner
    # We mock WindowsController to avoid actually moving user's OS mouse
    with patch("src.platform.windows.WindowsController") as mock_ctrl_class:
        mock_instance = mock_ctrl_class.return_value
        mock_instance.held_buttons = set()
        exit_code = main(["--replay", fixture_path, "--real-control"])
        assert exit_code == 0

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "REAL OS CONTROL" in captured.out


def test_camera_missing_dependencies(capsys):
    # Hide cv2 to test graceful degradation
    with patch.dict("sys.modules", {"cv2": None}):
        exit_code = run_camera()
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "opencv-python" in captured.out


def test_replay_nonexistent_fixture():
    exit_code = run_replay("nonexistent_fixture_file.json")
    assert exit_code == 1


class FakeVideoCapture:
    """Mock OpenCV VideoCapture for unit tests."""
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


import numpy as np

class FakeDetector:
    """Mock HandDetector tracking received frames and lifecycle."""
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


class FakePipeline:
    """Mock AirPadPipeline stub to avoid loading heavy production SVM in isolated contract tests."""
    def __init__(self):
        from src.platform.dry_run import DryRunController
        from src.gesture.pose import HandPose
        from src.gesture.fsm import FSMState
        self.controller = DryRunController()
        self.cleaned = False
        self.is_emergency_stopped = False

    def process_landmarks(self, landmarks=None, handedness="Right", timestamp_ms=0.0):
        from src.gesture.pose import HandPose
        from src.gesture.fsm import FSMState
        res = MagicMock()
        res.filtered_pose = HandPose.POINT_ONE
        res.fsm_state = FSMState.MOVING
        res.action_fired = None
        return res

    def cleanup(self):
        self.cleaned = True



def make_dummy_hand(x: float = 0.5, y: float = 0.5, z: float = 0.0, handedness: str = "Right"):
    from src.vision.adapter import HandLandmarks, LandmarkPoint
    pts = [LandmarkPoint(x=x, y=y, z=z) for _ in range(21)]
    return HandLandmarks(points=pts, handedness=handedness, confidence=0.95)


def test_run_camera_asymmetric_color_contract_rgb_to_detector_bgr_to_preview():
    """Asymmetric color regression test:

    Webcam provides BGR frames. MediaPipe requires RGB frames.
    Display HUD/OpenCV requires BGR frames.
    
    Using an asymmetric color [B=30, G=120, R=210]:
    - On old code: detector received BGR directly (channel 0 = 30).
    - On fixed code: detector receives RGB (channel 0 = 210, channel 2 = 30).
    - Preview imshow receives BGR (channel 0 = 30, channel 2 = 210).
    """
    import cv2
    import numpy as np

    # Asymmetric BGR test frame where R != B
    bgr_frame = np.full((100, 100, 3), [30, 120, 210], dtype=np.uint8)
    fake_cap = FakeVideoCapture([bgr_frame])
    fake_detector = FakeDetector()
    shown_frames = []

    def fake_imshow(win_name, img):
        shown_frames.append(img.copy())

    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("app.create_pipeline", return_value=FakePipeline()), \
         patch("cv2.imshow", side_effect=fake_imshow), \
         patch("cv2.waitKey", return_value=27):

        exit_code = run_camera(allow_real_control=False)
        assert exit_code == 0

    # 1. Detector MUST receive RGB
    assert len(fake_detector.frames_received) == 1
    rgb_received = fake_detector.frames_received[0]
    assert rgb_received[0, 0, 0] == 210, "Red channel must be at index 0 for MediaPipe detector"
    assert rgb_received[0, 0, 1] == 120, "Green channel must be at index 1"
    assert rgb_received[0, 0, 2] == 30, "Blue channel must be at index 2 for MediaPipe detector"

    # 2. Preview imshow MUST receive BGR
    assert len(shown_frames) == 1
    bgr_shown = shown_frames[0]
    assert bgr_shown[0, 0, 0] == 30, "Blue channel must remain at index 0 for OpenCV BGR preview"
    assert bgr_shown[0, 0, 1] == 120, "Green channel at index 1"
    assert bgr_shown[0, 0, 2] == 210, "Red channel must remain at index 2 for OpenCV BGR preview"


def test_asymmetric_color_regression_reproduced_on_saved_initial_app():
    """Reproduce RGB regression against saved original app via isolated module load.

    Verifies side-by-side without overwriting app.py:
    - Saved app.py.initial directly passed BGR [30, 120, 210] to detector (channel 0 = 30).
    - Fixed app.py passes converted RGB [210, 120, 30] to detector (channel 0 = 210).
    """
    import importlib.util
    from importlib.machinery import SourceFileLoader
    import numpy as np

    initial_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "reports", "acc07", "app.py.initial"
    )
    assert os.path.isfile(initial_path)
    loader = SourceFileLoader("app_initial_isolated", initial_path)
    spec = importlib.util.spec_from_loader("app_initial_isolated", loader)
    app_initial = importlib.util.module_from_spec(spec)
    loader.exec_module(app_initial)

    # 1. Exercise saved initial app
    bgr_frame_initial = np.full((50, 50, 3), [30, 120, 210], dtype=np.uint8)
    fake_cap_initial = FakeVideoCapture([bgr_frame_initial])
    fake_det_initial = FakeDetector()

    with patch("cv2.VideoCapture", return_value=fake_cap_initial), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_det_initial), \
         patch.object(app_initial, "create_pipeline", return_value=FakePipeline()), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=27):

        app_initial.run_camera(allow_real_control=False)

    assert len(fake_det_initial.frames_received) == 1
    # On the original unpatched code, detector receives BGR (channel 0 is Blue=30, channel 2 is Red=210)
    assert fake_det_initial.frames_received[0][0, 0, 0] == 30, "Original app passed BGR channel 0 directly"
    assert fake_det_initial.frames_received[0][0, 0, 2] == 210, "Original app passed BGR channel 2 directly"

    # 2. Exercise fixed current app
    bgr_frame_fixed = np.full((50, 50, 3), [30, 120, 210], dtype=np.uint8)
    fake_cap_fixed = FakeVideoCapture([bgr_frame_fixed])
    fake_det_fixed = FakeDetector()

    with patch("cv2.VideoCapture", return_value=fake_cap_fixed), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_det_fixed), \
         patch("app.create_pipeline", return_value=FakePipeline()), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=27):

        run_camera(allow_real_control=False)

    assert len(fake_det_fixed.frames_received) == 1
    # On fixed code, detector receives RGB (channel 0 is Red=210, channel 2 is Blue=30)
    assert fake_det_fixed.frames_received[0][0, 0, 0] == 210, "Fixed app passes RGB channel 0 (Red)"
    assert fake_det_fixed.frames_received[0][0, 0, 2] == 30, "Fixed app passes RGB channel 2 (Blue)"


def test_run_camera_model_path_forwarding_and_fail_closed():
    """Verify explicit model path loading and fail-closed validation with bundle assertions."""
    import numpy as np
    import app as app_module
    import tools.train as train_module

    candidate_path = "reports/acc05/candidate_personal_svm.json"
    assert os.path.exists(candidate_path)

    bgr_frame = np.zeros((60, 60, 3), dtype=np.uint8)
    fake_cap = FakeVideoCapture([bgr_frame])
    fake_detector = FakeDetector()

    load_bundle_calls = []
    create_pipeline_calls = []

    real_load_bundle = train_module.load_bundle
    def spy_load_bundle(path):
        load_bundle_calls.append(path)
        return real_load_bundle(path)

    real_create_pipeline = app_module.create_pipeline
    def spy_create_pipeline(config, allow_real_control=False, model_bundle=None):
        create_pipeline_calls.append({"allow_real_control": allow_real_control, "model_bundle": model_bundle})
        return real_create_pipeline(config, allow_real_control=allow_real_control, model_bundle=model_bundle)

    with patch("tools.train.load_bundle", side_effect=spy_load_bundle), \
         patch("app.create_pipeline", side_effect=spy_create_pipeline), \
         patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord('q')):

        exit_code = run_camera(allow_real_control=False, model_path=candidate_path)
        assert exit_code == 0

    assert len(load_bundle_calls) == 1
    assert load_bundle_calls[0] == candidate_path
    assert len(create_pipeline_calls) == 1
    passed_bundle = create_pipeline_calls[0]["model_bundle"]
    assert passed_bundle is not None
    assert passed_bundle.algorithm == "rbf_svm"
    assert passed_bundle.hyperparameters.get("C") == 10.0
    assert passed_bundle.hyperparameters.get("gamma") == 0.01

    # Invalid nonexistent model bundle path must fail closed without opening camera or pipeline
    with patch("cv2.VideoCapture") as mock_vc, patch("app.create_pipeline") as mock_cp:
        exit_code = run_camera(allow_real_control=False, model_path="models/nonexistent_candidate.json")
        assert exit_code == 1
        mock_vc.assert_not_called()
        mock_cp.assert_not_called()


def test_run_camera_raw_landmarks_and_handedness_forwarded_to_pipeline():
    """Verify detected raw landmarks and handedness are faithfully forwarded to pipeline."""
    import numpy as np
    from app import create_pipeline
    from src.config.loader import load_config

    hand = make_dummy_hand(x=0.42, y=0.58, z=0.05, handedness="Left")
    bgr_frame = np.zeros((80, 80, 3), dtype=np.uint8)
    fake_cap = FakeVideoCapture([bgr_frame])
    fake_detector = FakeDetector(detected_hands=[[hand]])

    processed_calls = []

    config = load_config()
    real_pipeline = create_pipeline(config, allow_real_control=False)
    original_process = real_pipeline.process_landmarks

    def spy_process_landmarks(landmarks=None, handedness="Right", timestamp_ms=0.0):
        processed_calls.append({
            "landmarks": landmarks.copy() if landmarks is not None else None,
            "handedness": handedness,
            "timestamp_ms": timestamp_ms,
        })
        return original_process(landmarks=landmarks, handedness=handedness, timestamp_ms=timestamp_ms)

    real_pipeline.process_landmarks = spy_process_landmarks

    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("app.create_pipeline", return_value=real_pipeline), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=27):

        exit_code = run_camera(allow_real_control=False)
        assert exit_code == 0

    assert len(processed_calls) == 1
    call_info = processed_calls[0]
    assert call_info["handedness"] == "Left"
    assert call_info["landmarks"] is not None
    assert call_info["landmarks"].shape == (21, 3)
    assert np.isclose(call_info["landmarks"][0, 0], 0.42)
    assert np.isclose(call_info["landmarks"][0, 1], 0.58)


def test_run_camera_resilient_cleanup_paths():
    """Verify resilient resource cleanup across success, camera open failure, and errors."""
    import numpy as np

    # 1. Success exit
    fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_detector = FakeDetector()
    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("app.create_pipeline", return_value=FakePipeline()), \
         patch("cv2.destroyAllWindows") as mock_destroy, \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord('q')):

        exit_code = run_camera(allow_real_control=False)
        assert exit_code == 0
        assert fake_detector.closed
        assert fake_cap.released
        mock_destroy.assert_called_once()

    # 2. Camera device open failure: must return 1 and cleanup initialized resources
    closed_cap = FakeVideoCapture([], is_opened=False)
    fake_detector_fail = FakeDetector()
    with patch("cv2.VideoCapture", return_value=closed_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector_fail), \
         patch("app.create_pipeline", return_value=FakePipeline()), \
         patch("cv2.destroyAllWindows") as mock_destroy_fail:

        exit_code = run_camera(allow_real_control=False)
        assert exit_code == 1
        assert fake_detector_fail.closed
        assert closed_cap.released
        mock_destroy_fail.assert_called_once()


def test_run_camera_cleanup_failure_on_success_path_raises():
    """Verify cleanup exceptions during normal exit are propagated, not swallowed with return 0."""
    import numpy as np

    fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_detector = FakeDetector()

    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord('q')), \
         patch("cv2.destroyAllWindows") as mock_destroy, \
         patch("src.runtime.pipeline.AirPadPipeline.cleanup", side_effect=RuntimeError("Pipeline held-input cleanup failed")):

        with pytest.raises(RuntimeError, match="Pipeline held-input cleanup failed"):
            run_camera(allow_real_control=False)

        # All cleanups must still have been attempted
        assert fake_detector.closed
        assert fake_cap.released
        mock_destroy.assert_called_once()


def test_run_camera_combined_original_and_cleanup_failures_preserves_original_exception():
    """Verify original loop exception is preserved even when cleanup functions also fail."""
    import numpy as np

    fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_detector = FakeDetector()
    fake_detector.close = MagicMock(side_effect=RuntimeError("Secondary detector close fault"))

    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("cv2.destroyAllWindows") as mock_destroy, \
         patch("src.runtime.pipeline.AirPadPipeline.cleanup", side_effect=RuntimeError("Secondary pipeline cleanup fault")), \
         patch("cv2.imshow", side_effect=ValueError("Primary frame loop display fault")):

        # Must raise the primary exception, not the secondary cleanup exception
        with pytest.raises(ValueError, match="Primary frame loop display fault"):
            run_camera(allow_real_control=False)

        # All cleanups must still have been attempted
        assert fake_cap.released
        mock_destroy.assert_called_once()


def test_run_camera_detector_exception_propagates_and_cleans_up():
    """MediaPipe detector exception during detection loop must propagate out and trigger full cleanup."""
    import numpy as np

    fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_detector = FakeDetector()
    fake_detector.detect = MagicMock(side_effect=RuntimeError("MediaPipe detection hardware fault"))
    fake_pipeline = FakePipeline()

    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("app.create_pipeline", return_value=fake_pipeline), \
         patch("cv2.destroyAllWindows") as mock_destroy, \
         patch("cv2.imshow"):

        with pytest.raises(RuntimeError, match="MediaPipe detection hardware fault"):
            run_camera(allow_real_control=False)

    assert fake_detector.closed
    assert fake_pipeline.cleaned
    assert fake_cap.released
    mock_destroy.assert_called_once()


def test_run_camera_read_failure_exits_cleanly_with_code_0():
    """Camera frame read failure (EOF mid-stream) exits cleanly code 0 with full cleanup."""
    import numpy as np

    # Frame 1 is valid, frame 2 read returns (False, None) to trigger loop break
    fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_detector = FakeDetector()
    fake_pipeline = FakePipeline()

    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("app.create_pipeline", return_value=fake_pipeline), \
         patch("cv2.destroyAllWindows") as mock_destroy, \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=0):  # Normal frame processing, breaks on EOF

        exit_code = run_camera(allow_real_control=False)
        assert exit_code == 0

    assert fake_detector.closed
    assert fake_pipeline.cleaned
    assert fake_cap.released
    mock_destroy.assert_called_once()


def test_run_camera_default_dry_run_and_no_real_os_controller():
    """Verify default run_camera uses DryRunController and NEVER instantiates WindowsController."""
    import numpy as np
    from src.platform.dry_run import DryRunController

    fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_detector = FakeDetector()

    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_detector), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord('q')), \
         patch("src.platform.windows.WindowsController", side_effect=AssertionError("WindowsController must never be instantiated!")):

        exit_code = run_camera(allow_real_control=False)
        assert exit_code == 0


def test_synthetic_dynamic_fsm_sequence_in_dry_run():
    """Exercise full pipeline with real default config through controlled classifier sequence.

    Tests:
    - Entry into MOVING requires stable POINT_ONE (>= 2 updates across point_one_stable_ms=100ms)
    - Short PINCH_CLOSED release to POINT_ONE (80ms <= elapsed <= 300ms) triggers exactly one LEFT_CLICK
    - Sustained PINCH_CLOSED (> drag_threshold_ms=400ms) enters DRAGGING with exactly one LEFT_DOWN
    - Drag release to POINT_ONE fires exactly one LEFT_UP and returns to MOVING
    - Subsequent hand loss (landmarks=None) immediately resets to PAUSED with zero held inputs
    - Immediate first reacquisition frame gates cursor movement and permits zero movement/held inputs
    - Subsequent stable frames satisfy dwell and re-enable cursor movement
    - Pipeline cleanup confirms held buttons empty
    """
    import numpy as np
    from app import create_pipeline
    from src.config.loader import load_config
    from src.platform.dry_run import DryRunController
    from src.gesture.fsm import FSMState
    from src.gesture.pose import HandPose
    from src.platform.interface import MouseEvent

    config = load_config()  # Real default configuration (no altered thresholds!)
    pipeline = create_pipeline(config, allow_real_control=False)
    ctrl = pipeline.controller
    assert isinstance(ctrl, DryRunController)

    dummy_lms = np.full((21, 3), 0.5, dtype=np.float32)

    class SequenceClassifier:
        def __init__(self):
            self.current_pose = "POINT_ONE"

        def predict_with_rejection(self, canon_flat, decision_threshold=0.0, margin_threshold=0.15, unknown_label="UNKNOWN"):
            return [self.current_pose]

    seq_clf = SequenceClassifier()
    pipeline.classifier = seq_clf

    # 1. Entry into MOVING: requires stable POINT_ONE through temporal filters
    seq_clf.current_pose = "POINT_ONE"
    for step in range(10):
        res = pipeline.process_landmarks(dummy_lms, handedness="Right", timestamp_ms=1000.0 + step * 33.3)
    assert res.fsm_state == FSMState.MOVING
    assert pipeline.fsm.can_move_cursor(HandPose.POINT_ONE) is True

    # 2. Click gesture: short pinch (80ms <= elapsed <= 300ms) released to POINT_ONE
    seq_clf.current_pose = "PINCH_CLOSED"
    for step in range(5):
        res = pipeline.process_landmarks(dummy_lms, handedness="Right", timestamp_ms=1350.0 + step * 33.3)
    assert res.fsm_state == FSMState.CLICK_PENDING

    seq_clf.current_pose = "POINT_ONE"
    for step in range(5):
        res = pipeline.process_landmarks(dummy_lms, handedness="Right", timestamp_ms=1550.0 + step * 33.3)
    assert res.fsm_state == FSMState.MOVING
    click_events = [e for e in ctrl.event_log if e.get("type") == MouseEvent.LEFT_CLICK]
    assert len(click_events) == 1, f"Short pinch release must emit exactly one LEFT_CLICK, got {len(click_events)}"

    # 3. Drag gesture: sustained pinch hold (> drag_threshold_ms=400ms) followed by clean drag release
    seq_clf.current_pose = "PINCH_CLOSED"
    for step in range(20):
        res = pipeline.process_landmarks(dummy_lms, handedness="Right", timestamp_ms=1750.0 + step * 33.3)
    assert res.fsm_state == FSMState.DRAGGING
    assert "left" in ctrl.held_buttons
    down_events = [e for e in ctrl.event_log if e.get("type") == MouseEvent.LEFT_DOWN]
    assert len(down_events) == 1, f"Drag initiation must emit exactly one LEFT_DOWN, got {len(down_events)}"

    # Drag release (drag up): release back to POINT_ONE
    seq_clf.current_pose = "POINT_ONE"
    for step in range(10):
        res = pipeline.process_landmarks(dummy_lms, handedness="Right", timestamp_ms=2450.0 + step * 33.3)
    assert res.fsm_state == FSMState.MOVING
    assert "left" not in ctrl.held_buttons
    up_events = [e for e in ctrl.event_log if e.get("type") == MouseEvent.LEFT_UP]
    assert len(up_events) == 1, f"Drag release to POINT_ONE must emit exactly one LEFT_UP, got {len(up_events)}"

    # 4. Hand loss: landmarks=None immediately releases held inputs and returns to PAUSED
    res_loss = pipeline.process_landmarks(None, handedness="Right", timestamp_ms=2800.0)
    assert res_loss.fsm_state == FSMState.PAUSED
    assert len(ctrl.held_buttons) == 0, "Held buttons must be cleared immediately on hand loss"
    assert len(ctrl.held_keys) == 0, "Held keys must be cleared immediately on hand loss"

    # 5. Post-loss conservative reacquisition:
    # First frame of hand returning with POINT_ONE does NOT immediately grant cursor movement or dispatch moves
    moves_before = len([e for e in ctrl.event_log if e.get("type") == MouseEvent.MOVE])
    seq_clf.current_pose = "POINT_ONE"
    res_reacq1 = pipeline.process_landmarks(dummy_lms, handedness="Right", timestamp_ms=2900.0)
    assert res_reacq1.fsm_state == FSMState.PAUSED
    assert pipeline.fsm.can_move_cursor(HandPose.POINT_ONE) is False, "Cursor movement must be gated during settling"
    assert res_reacq1.action_fired is None
    moves_after = len([e for e in ctrl.event_log if e.get("type") == MouseEvent.MOVE])
    assert moves_after == moves_before, "No cursor movement events permitted on immediate first reacquisition frame"
    assert len(ctrl.held_buttons) == 0, "No held buttons permitted on immediate first reacquisition frame"
    assert len(ctrl.held_keys) == 0, "No held keys permitted on immediate first reacquisition frame"

    # Subsequent frames satisfy temporal filter and dwell
    for step in range(1, 10):
        res = pipeline.process_landmarks(dummy_lms, handedness="Right", timestamp_ms=2900.0 + step * 33.3)
    assert res.fsm_state == FSMState.MOVING
    assert pipeline.fsm.can_move_cursor(HandPose.POINT_ONE) is True

    # Audit total event counts across entire test: exactly one click, exactly one down, exactly one up!
    assert len([e for e in ctrl.event_log if e.get("type") == MouseEvent.LEFT_CLICK]) == 1
    assert len([e for e in ctrl.event_log if e.get("type") == MouseEvent.LEFT_DOWN]) == 1
    assert len([e for e in ctrl.event_log if e.get("type") == MouseEvent.LEFT_UP]) == 1

    # 6. Pipeline cleanup keeps held buttons clean
    pipeline.cleanup()
    assert len(ctrl.held_buttons) == 0
    assert len(ctrl.held_keys) == 0


def test_camera_cursor_direction_regression_reproduced_on_saved_initial_app():
    """Reproduce reversed cursor direction regression on saved original app vs fixed app.

    Problem:
    - OpenCV preview flips webcam horizontally (cv2.flip(frame, 1)). Hand moving to user's
      right appears on the right side of the flipped frame (x increases from 0.35 to 0.65).
    - In initial app.py, run_camera passes config with default mirror_x=True, causing CursorMapper
      to invert x AGAIN (cur_x = 1.0 - x). Thus, moving hand right moved cursor left (reversed!).
    - In fixed app.py, run_camera creates cam_config with mirror_x=False so that increasing
      flipped-preview x directly increases screen x (moving cursor right).
    """
    import importlib.util
    from importlib.machinery import SourceFileLoader

    initial_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "reports", "acc08", "app.py.initial"
    )
    assert os.path.isfile(initial_path)
    loader = SourceFileLoader("app_initial_acc08", initial_path)
    spec = importlib.util.spec_from_loader("app_initial_acc08", loader)
    app_initial = importlib.util.module_from_spec(spec)
    loader.exec_module(app_initial)

    created_initial_pipelines = []
    real_create_pipeline_initial = app_initial.create_pipeline
    def spy_create_pipeline_initial(config, allow_real_control=False, model_bundle=None):
        pipe = real_create_pipeline_initial(config, allow_real_control=allow_real_control, model_bundle=model_bundle)
        created_initial_pipelines.append(pipe)
        return pipe

    fake_cap_initial = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_det_initial = FakeDetector()

    with patch("cv2.VideoCapture", return_value=fake_cap_initial), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_det_initial), \
         patch.object(app_initial, "create_pipeline", side_effect=spy_create_pipeline_initial), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord('q')):

        exit_code_initial = app_initial.run_camera(allow_real_control=False)
        assert exit_code_initial == 0

    assert len(created_initial_pipelines) == 1
    pipe_initial = created_initial_pipelines[0]
    # On initial app, mirror_x was True
    assert pipe_initial.cursor_mapper.mirror_x is True
    # Demonstrating the reversal on initial app:
    # Hand moving right on flipped preview (x: 0.35 -> 0.65) resulted in DECREASING screen x (cursor moving left!)
    pos_left_hand = pipe_initial.cursor_mapper.map_point(0.35, 0.5, timestamp=1.0)
    pos_right_hand = pipe_initial.cursor_mapper.map_point(0.65, 0.5, timestamp=1.05)
    assert pos_left_hand[0] > pos_right_hand[0], (
        f"Initial app exhibited reversed cursor: screen x decreased ({pos_left_hand[0]} -> {pos_right_hand[0]}) "
        f"when hand moved right on flipped preview"
    )

    # Now verify fixed app
    created_fixed_pipelines = []
    real_create_pipeline_fixed = create_pipeline
    def spy_create_pipeline_fixed(config, allow_real_control=False, model_bundle=None):
        pipe = real_create_pipeline_fixed(config, allow_real_control=allow_real_control, model_bundle=model_bundle)
        created_fixed_pipelines.append(pipe)
        return pipe

    fake_cap_fixed = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_det_fixed = FakeDetector()

    with patch("cv2.VideoCapture", return_value=fake_cap_fixed), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_det_fixed), \
         patch("app.create_pipeline", side_effect=spy_create_pipeline_fixed), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord('q')):

        exit_code_fixed = run_camera(allow_real_control=False)
        assert exit_code_fixed == 0

    assert len(created_fixed_pipelines) == 1
    pipe_fixed = created_fixed_pipelines[0]
    # On fixed app, mirror_x must be False for camera
    assert pipe_fixed.cursor_mapper.mirror_x is False
    # Correct direction:
    # Hand moving right on flipped preview (x: 0.35 -> 0.65) results in INCREASING screen x (cursor moving right!)
    pos_left_hand_fixed = pipe_fixed.cursor_mapper.map_point(0.35, 0.5, timestamp=1.0)
    pos_right_hand_fixed = pipe_fixed.cursor_mapper.map_point(0.65, 0.5, timestamp=1.05)
    assert pos_right_hand_fixed[0] > pos_left_hand_fixed[0], (
        f"Fixed app must move cursor right: screen x increased ({pos_left_hand_fixed[0]} -> {pos_right_hand_fixed[0]}) "
        f"when hand moved right on flipped preview"
    )


def test_run_camera_end_to_end_cursor_direction_x_and_y():
    """Verify live run_camera end-to-end with phase-specific RIGHT, LEFT, DOWN, UP controller MOVE deltas.

    Exercises:
    - Real CursorMapper (with camera-specific mirror_x=False)
    - Real MajorityFilter, StabilityGate, and GestureFSM in MOVING state
    - Real DryRunController recording platform events
    - Phase-specific directional validation:
        1. RIGHT phase: each consecutive MOVE event has dx > 0
        2. LEFT phase: each consecutive MOVE event has dx < 0
        3. DOWN phase: each consecutive MOVE event has dy > 0
        4. UP phase: each consecutive MOVE event has dy < 0
    """
    from src.platform.interface import MouseEvent
    from src.vision.adapter import LandmarkPoint, HandLandmarks

    def make_positioned_hand(x: float, y: float):
        pts = [LandmarkPoint(x=x, y=y, z=0.0) for _ in range(21)]
        return HandLandmarks(points=pts, handedness="Right", confidence=0.95)

    # 1. Settling phase: 8 frames at (0.50, 0.50) to allow MajorityFilter (5), StabilityGate (2),
    # and FSM dwell (100ms) to transition to MOVING and establish stable baseline
    settle_frames = [(0.50, 0.50)] * 8

    # 2. RIGHT phase: 5 steps moving right from 0.50 to 0.65
    right_frames = [(0.50 + i * 0.03, 0.50) for i in range(1, 6)]

    # 3. LEFT phase: 5 steps moving left from 0.65 back to 0.40
    left_frames = [(0.65 - i * 0.05, 0.50) for i in range(1, 6)]

    # 4. Vertical settle: 4 frames at (0.50, 0.50) to settle horizontal filter momentum
    vert_settle = [(0.50, 0.50)] * 4

    # 5. DOWN phase: 5 steps moving down from 0.50 to 0.65
    down_frames = [(0.50, 0.50 + i * 0.03) for i in range(1, 6)]

    # 6. UP phase: 5 steps moving up from 0.65 back to 0.40
    up_frames = [(0.50, 0.65 - i * 0.05) for i in range(1, 6)]

    all_coords = settle_frames + right_frames + left_frames + vert_settle + down_frames + up_frames
    hands_seq = [[make_positioned_hand(x, y)] for x, y in all_coords]

    fake_cap = FakeVideoCapture([np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(len(hands_seq))])
    fake_det = FakeDetector(hands_seq)

    captured_pipelines = []
    real_create_pipeline = create_pipeline

    class AlwaysPointOneClassifier:
        def predict_with_rejection(self, canon_flat, decision_threshold=0.0, margin_threshold=0.15, unknown_label="UNKNOWN"):
            return ["POINT_ONE"]

    def spy_create_pipeline(config, allow_real_control=False, model_bundle=None):
        pipe = real_create_pipeline(config, allow_real_control=allow_real_control, model_bundle=model_bundle)
        pipe.classifier = AlwaysPointOneClassifier()
        # Test instrumentation: rate_limiter.allow is mocked to True strictly in this isolated test
        # to allow deterministic per-frame event dispatch without depending on wall-clock sleep timing,
        # preserving all production filtering, FSM gating, and coordinate transformation logic.
        if pipe.rate_limiter:
            pipe.rate_limiter.allow = lambda: True
        captured_pipelines.append(pipe)
        return pipe

    simulated_time = 1000.0
    def advancing_time():
        nonlocal simulated_time
        simulated_time += 0.0333
        return simulated_time

    with patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_det), \
         patch("app.create_pipeline", side_effect=spy_create_pipeline), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=0), \
         patch("time.time", side_effect=advancing_time):

        exit_code = run_camera(allow_real_control=False)
        assert exit_code == 0

    assert len(captured_pipelines) == 1
    pipe = captured_pipelines[0]
    ctrl = pipe.controller
    move_events = [e for e in ctrl.event_log if e.get("type") == MouseEvent.MOVE]
    assert len(move_events) >= 20, f"Expected >= 20 MOVE events, got {len(move_events)}"

    deltas = [(move_events[i]["x"] - move_events[i-1]["x"], move_events[i]["y"] - move_events[i-1]["y"])
              for i in range(1, len(move_events))]

    # Phase-specific delta validations based on known sequence indices
    # RIGHT phase deltas (indices 2 to 6 in delta list)
    right_deltas = [d[0] for d in deltas[2:7]]
    assert len(right_deltas) == 5
    assert all(dx > 0 for dx in right_deltas), f"RIGHT phase must have all positive dx, got {right_deltas}"

    # LEFT phase deltas (indices 8 to 12 in delta list)
    left_deltas = [d[0] for d in deltas[8:12]]
    assert len(left_deltas) == 4
    assert all(dx < 0 for dx in left_deltas), f"LEFT phase must have all negative dx, got {left_deltas}"

    # DOWN phase deltas (indices 16 to 21 in delta list)
    down_deltas = [d[1] for d in deltas[16:21]]
    assert len(down_deltas) == 5
    assert all(dy > 0 for dy in down_deltas), f"DOWN phase must have all positive dy, got {down_deltas}"

    # UP phase deltas (indices 22 to 26 in delta list)
    up_deltas = [d[1] for d in deltas[22:26]]
    assert len(up_deltas) == 4
    assert all(dy < 0 for dy in up_deltas), f"UP phase must have all negative dy, got {up_deltas}"


def test_run_camera_does_not_mutate_config_and_preserves_generic_pipeline():
    """Verify run_camera does not mutate the loaded config dictionary object in-place,
    and preserves generic pipeline mirror_x=True/False opposite mapping semantics.
    """
    from src.config.loader import load_config

    # 1. Load config and keep a pristine deep-copied copy of the exact dictionary
    initial_cfg = load_config()
    assert initial_cfg.get("motion", {}).get("mirror_x") is True, "Default config in default.yaml must have mirror_x=True"
    pristine_cfg = copy.deepcopy(initial_cfg)

    # 2. Patch app.load_config to return the EXACT initial_cfg object (same instance in memory)
    fake_cap = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_det = FakeDetector()

    with patch("app.load_config", return_value=initial_cfg), \
         patch("cv2.VideoCapture", return_value=fake_cap), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_det), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord('q')):

        exit_code = run_camera(allow_real_control=False)
        assert exit_code == 0

    # Assert that initial_cfg was NOT mutated in-place by run_camera
    assert initial_cfg == pristine_cfg, "run_camera must not mutate the loaded config dictionary in-place"
    assert initial_cfg["motion"]["mirror_x"] is True, "initial_cfg['motion']['mirror_x'] must remain True"

    # 3. Assert effective camera pipeline has mirror_x=False
    captured_pipelines = []
    real_create_pipeline = create_pipeline
    def spy_create_pipeline(config, allow_real_control=False, model_bundle=None):
        pipe = real_create_pipeline(config, allow_real_control=allow_real_control, model_bundle=model_bundle)
        captured_pipelines.append(pipe)
        return pipe

    fake_cap2 = FakeVideoCapture([np.zeros((40, 40, 3), dtype=np.uint8)])
    fake_det2 = FakeDetector()
    with patch("cv2.VideoCapture", return_value=fake_cap2), \
         patch("src.vision.adapter.MediaPipeAdapter", return_value=fake_det2), \
         patch("app.create_pipeline", side_effect=spy_create_pipeline), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord('q')):

        exit_code2 = run_camera(allow_real_control=False)
        assert exit_code2 == 0

    assert len(captured_pipelines) == 1
    cam_pipe = captured_pipelines[0]
    assert cam_pipe.cursor_mapper.mirror_x is False, "Camera pipeline must have effective mirror_x=False"

    # 4. Verify generic create_pipeline with mirror_x=True vs mirror_x=False maps opposite x directions
    pipe_mirrored = create_pipeline(pristine_cfg, allow_real_control=False)
    assert pipe_mirrored.cursor_mapper.mirror_x is True

    unmirrored_cfg = copy.deepcopy(pristine_cfg)
    unmirrored_cfg["motion"]["mirror_x"] = False
    pipe_unmirrored = create_pipeline(unmirrored_cfg, allow_real_control=False)
    assert pipe_unmirrored.cursor_mapper.mirror_x is False

    # When landmark moves left-to-right (0.35 -> 0.65):
    # - Mirrored pipeline moves screen x in opposite direction (decreases screen x)
    # - Unmirrored pipeline moves screen x in same direction (increases screen x)
    x_mirrored_start = pipe_mirrored.cursor_mapper.map_point(0.35, 0.5, timestamp=1.0)[0]
    x_mirrored_end = pipe_mirrored.cursor_mapper.map_point(0.65, 0.5, timestamp=1.05)[0]
    assert x_mirrored_end < x_mirrored_start, "Generic mirror_x=True maps increasing preview x to decreasing screen x"

    x_unmirrored_start = pipe_unmirrored.cursor_mapper.map_point(0.35, 0.5, timestamp=1.0)[0]
    x_unmirrored_end = pipe_unmirrored.cursor_mapper.map_point(0.65, 0.5, timestamp=1.05)[0]
    assert x_unmirrored_end > x_unmirrored_start, "Generic mirror_x=False maps increasing preview x to increasing screen x"



