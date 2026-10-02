import pytest
import numpy as np
from unittest.mock import MagicMock

from src.runtime.pipeline import AirPadPipeline, PipelineResult
from src.gesture.pose import HandPose
from src.gesture.fsm import GestureFSM, FSMState
from src.platform.dry_run import DryRunController

def test_pipeline_feature_version_mismatch():
    mock_bundle = MagicMock()
    mock_bundle.feature_version = 2
    with pytest.raises(ValueError, match="feature version mismatch"):
        AirPadPipeline(config={}, model_bundle=mock_bundle)

def test_pipeline_no_hand_resets():
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
        'unknown_timeout_frames': 5
    }
    controller = DryRunController()
    fsm = GestureFSM(controller, config)
    fsm.state = FSMState.MOVING
    
    pipeline = AirPadPipeline(config=config, controller=controller, fsm=fsm)
    result = pipeline.process_landmarks(None)
    
    assert result.raw_pose == HandPose.NO_HAND
    assert result.fsm_state == FSMState.PAUSED
    assert result.action_fired is None

def test_pipeline_end_to_end_synthetic():
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
        'unknown_timeout_frames': 5
    }
    controller = DryRunController()
    fsm = GestureFSM(controller, config)
    
    mock_classifier = MagicMock()
    mock_classifier.predict_with_rejection.return_value = np.array(["POINT_ONE"])
    
    # Mock CursorMapper
    mock_mapper = MagicMock()
    mock_mapper.map_point.return_value = (500, 500)
    
    pipeline = AirPadPipeline(
        config=config,
        controller=controller,
        classifier=mock_classifier,
        fsm=fsm,
        cursor_mapper=mock_mapper
    )
    
    synthetic_landmarks = np.zeros((21, 3))
    
    # Send frames to reach POINT_ONE stable
    for i in range(3):
        res = pipeline.process_landmarks(synthetic_landmarks, timestamp_ms=i*50.0)
    
    # Frame 4 should hit MOVING and emit MOVE action
    res = pipeline.process_landmarks(synthetic_landmarks, timestamp_ms=200.0)
    assert res.fsm_state == FSMState.MOVING
    assert res.action_fired == "MOVE"
    assert len(controller.event_log) > 0
    assert controller.event_log[-1]['type'].name == "MOVE"
    
    # Cleanup should release controller
    pipeline.cleanup()
    assert len(controller.held_buttons) == 0

def test_pipeline_pinch_click():
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
        'unknown_timeout_frames': 5
    }
    controller = DryRunController()
    fsm = GestureFSM(controller, config)
    fsm.state = FSMState.MOVING
    fsm._reacquire_count = 2
    
    mock_classifier = MagicMock()
    mock_classifier.predict_with_rejection.return_value = np.array(["PINCH_CLOSED"])
    
    pipeline = AirPadPipeline(config=config, controller=controller, classifier=mock_classifier, fsm=fsm)
    synthetic_landmarks = np.zeros((21, 3))
    
    # Send PINCH_CLOSED
    res = pipeline.process_landmarks(synthetic_landmarks, timestamp_ms=0.0)
    assert res.fsm_state == FSMState.CLICK_PENDING
    
    # Wait longer than drag threshold
    res = pipeline.process_landmarks(synthetic_landmarks, timestamp_ms=450.0)
    assert res.fsm_state == FSMState.DRAGGING
    assert fsm.is_mouse_down is True
    assert 'left' in controller.held_buttons
    
    pipeline.cleanup()
    assert len(controller.held_buttons) == 0


def test_pipeline_uses_nested_ml_rejection_thresholds():
    classifier = MagicMock()
    classifier.predict_with_rejection.return_value = np.array(["UNKNOWN"])
    pipeline = AirPadPipeline(
        config={"ml": {"decision_threshold": 1.25, "margin_threshold": 0.35}},
        classifier=classifier,
    )

    pipeline.process_landmarks(np.zeros((21, 3)))

    classifier.predict_with_rejection.assert_called_once()
    kwargs = classifier.predict_with_rejection.call_args.kwargs
    assert kwargs["decision_threshold"] == 1.25
    assert kwargs["margin_threshold"] == 0.35


def test_pipeline_failsafe_corner_triggers_emergency_stop():
    from src.safety.failsafe import FailsafeMonitor
    controller = DryRunController()
    fsm = GestureFSM(controller, {})
    fsm.state = FSMState.MOVING
    fsm._reacquire_count = 2
    fsm._point_one_start_ms = 0.0
    
    mock_classifier = MagicMock()
    mock_classifier.predict_with_rejection.return_value = np.array(["POINT_ONE"])
    mock_mapper = MagicMock()
    mock_mapper.map_point.return_value = (5, 5)  # Top-left corner
    
    failsafe = FailsafeMonitor(1920, 1080, corner_size=10)
    pipeline = AirPadPipeline(
        config={},
        controller=controller,
        classifier=mock_classifier,
        fsm=fsm,
        cursor_mapper=mock_mapper,
        failsafe_monitor=failsafe,
    )
    
    res = pipeline.process_landmarks(np.zeros((21, 3)), timestamp_ms=100.0)
    assert res.action_fired == "EMERGENCY_STOP"
    assert pipeline.is_emergency_stopped is True
    assert failsafe.is_stopped() is True
    assert fsm.state == FSMState.PAUSED


def test_pipeline_emergency_hotkey_stops_pipeline():
    from src.safety.failsafe import EmergencyHotkey
    controller = DryRunController()
    fsm = GestureFSM(controller, {})
    fsm.state = FSMState.MOVING
    controller.mouse_down('left')
    
    hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=True)
    pipeline = AirPadPipeline(
        config={},
        controller=controller,
        fsm=fsm,
        emergency_hotkey=hotkey,
    )
    
    # Trigger emergency via hotkey
    hotkey.trigger_emergency()
    assert pipeline.is_emergency_stopped is True
    assert fsm.state == FSMState.PAUSED
    assert len(controller.held_buttons) == 0
    pipeline.cleanup()
