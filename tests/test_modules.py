"""Tests for temporal filters, evaluation, platform, safety, and config modules.

Covers:
- MajorityFilter / StabilityGate
- compute_metrics / GroupKFold / LOSO
- DryRunController events and cleanup
- RateLimiter / FailsafeMonitor / EmergencyHotkey
- Config loading + schema validation
- ModelBundle save/load round-trip
- Data schema
"""
from __future__ import annotations

import json
import os
import tempfile
import numpy as np
import pytest

from src.gesture.temporal import MajorityFilter, StabilityGate
from src.gesture.pose import HandPose
from src.platform.dry_run import DryRunController
from src.platform.interface import MouseEvent, KeyEvent
from src.ml.evaluation import compute_metrics, group_kfold_split, leave_one_subject_out
from src.ml.bundle import ModelBundle
from src.ml.data_schema import SampleMetadata, LABEL_NAMES
from src.safety.failsafe import RateLimiter, FailsafeMonitor, EmergencyHotkey


# ===================================================================
# Temporal filters
# ===================================================================
class TestMajorityFilter:
    def test_initial_is_unknown(self):
        mf = MajorityFilter(window_size=5, majority_ratio=0.6)
        assert mf.current_stable_pose == HandPose.UNKNOWN

    def test_majority_reached(self):
        mf = MajorityFilter(window_size=5, majority_ratio=0.6)
        # 3 out of 5 = 0.6 → should switch
        for _ in range(3):
            result = mf.update(HandPose.POINT_ONE)
        assert result == HandPose.POINT_ONE

    def test_no_majority_keeps_stable(self):
        mf = MajorityFilter(window_size=5, majority_ratio=0.6)
        mf.update(HandPose.POINT_ONE)
        mf.update(HandPose.OPEN_PALM)
        mf.update(HandPose.FIST)
        result = mf.update(HandPose.PINCH_CLOSED)
        assert result == HandPose.UNKNOWN  # no majority reached

    def test_reset_clears(self):
        mf = MajorityFilter(window_size=3)
        for _ in range(3):
            mf.update(HandPose.POINT_ONE)
        mf.reset()
        assert mf.current_stable_pose == HandPose.UNKNOWN
        assert len(mf.window) == 0


class TestStabilityGate:
    def test_newly_stable_fires_once(self):
        sg = StabilityGate(required_count=3)
        _, s1 = sg.update(HandPose.POINT_ONE)
        assert not s1
        _, s2 = sg.update(HandPose.POINT_ONE)
        assert not s2
        pose, s3 = sg.update(HandPose.POINT_ONE)
        assert s3  # newly stable on 3rd frame
        assert pose == HandPose.POINT_ONE

        # 4th frame: not newly stable
        _, s4 = sg.update(HandPose.POINT_ONE)
        assert not s4

    def test_interruption_resets_count(self):
        sg = StabilityGate(required_count=3)
        sg.update(HandPose.POINT_ONE)
        sg.update(HandPose.POINT_ONE)
        sg.update(HandPose.FIST)  # reset
        sg.update(HandPose.POINT_ONE)
        _, stable = sg.update(HandPose.POINT_ONE)
        assert not stable  # only 2 consecutive POINT_ONE after reset

    def test_reset(self):
        sg = StabilityGate()
        sg.update(HandPose.POINT_ONE)
        sg.reset()
        assert sg.count == 0
        assert sg.stable_pose == HandPose.UNKNOWN


# ===================================================================
# DryRunController
# ===================================================================
class TestDryRunController:
    def test_mouse_move_logged(self):
        ctrl = DryRunController()
        ctrl.mouse_move(100, 200)
        assert len(ctrl.event_log) == 1
        assert ctrl.event_log[0]['type'] == MouseEvent.MOVE
        assert ctrl.event_log[0]['x'] == 100

    def test_mouse_down_up_tracked(self):
        ctrl = DryRunController()
        ctrl.mouse_down('left')
        assert 'left' in ctrl.held_buttons
        ctrl.mouse_up('left')
        assert 'left' not in ctrl.held_buttons

    def test_cleanup_releases_all(self):
        ctrl = DryRunController()
        ctrl.mouse_down('left')
        ctrl.key_down('shift')
        assert len(ctrl.held_buttons) == 1
        assert len(ctrl.held_keys) == 1

        ctrl.cleanup()
        assert len(ctrl.held_buttons) == 0
        assert len(ctrl.held_keys) == 0

        # Verify cleanup events logged
        cleanup_events = [e for e in ctrl.event_log if e.get('cleanup')]
        assert len(cleanup_events) == 2

    def test_cleanup_idempotent(self):
        ctrl = DryRunController()
        ctrl.cleanup()  # nothing held
        assert len(ctrl.event_log) == 0

    def test_screen_size(self):
        ctrl = DryRunController()
        assert ctrl.get_screen_size() == (1920, 1080)


# ===================================================================
# Evaluation metrics
# ===================================================================
class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['macro_f1'] == 1.0
        for label_metrics in metrics['per_class'].values():
            assert label_metrics['precision'] == 1.0
            assert label_metrics['recall'] == 1.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 2])
        metrics = compute_metrics(y_true, y_pred, labels=[0, 1, 2])
        cm = metrics['confusion_matrix']
        assert len(cm) == 3
        assert len(cm[0]) == 3

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        metrics = compute_metrics(y_true, y_pred, labels=[0, 1])
        assert metrics['macro_f1'] == 0.0


class TestGroupKFold:
    def test_no_group_leakage(self):
        X = np.ones((10, 2))
        y = np.zeros(10)
        groups = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])

        for train_idx, test_idx in group_kfold_split(X, y, groups, n_splits=2):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups), "Groups must not leak between splits"

    def test_all_indices_covered(self):
        X = np.ones((8, 2))
        y = np.zeros(8)
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        all_test = []
        for _, test_idx in group_kfold_split(X, y, groups, n_splits=2):
            all_test.extend(test_idx.tolist())
        assert sorted(all_test) == list(range(8))


class TestLOSO:
    def test_one_subject_per_fold(self):
        X = np.ones((6, 2))
        y = np.zeros(6)
        subjects = np.array(['A', 'A', 'B', 'B', 'C', 'C'])

        folds = list(leave_one_subject_out(X, y, subjects))
        assert len(folds) == 3  # one per unique subject

        for train_idx, test_idx in folds:
            test_subjects = set(subjects[test_idx])
            assert len(test_subjects) == 1


# ===================================================================
# Model bundle
# ===================================================================
class TestModelBundle:
    def test_save_load_roundtrip(self, tmp_path):
        bundle = ModelBundle(
            feature_version=1,
            labels=['OPEN_PALM', 'POINT_ONE', 'PINCH_CLOSED'],
            algorithm='rbf_svm',
            hyperparameters={'C': 10.0, 'gamma': 'scale'},
            train_samples=100,
            created_at='2024-01-01T00:00:00',
            metadata={'author': 'test'},
            model_data={'X_train': [[1, 2], [3, 4]], 'y_train': [0, 1]},
        )
        path = str(tmp_path / 'model.json')
        bundle.save(path)

        loaded = ModelBundle.load(path)
        assert loaded.feature_version == 1
        assert loaded.labels == ['OPEN_PALM', 'POINT_ONE', 'PINCH_CLOSED']
        assert loaded.algorithm == 'rbf_svm'
        assert loaded.hyperparameters == {'C': 10.0, 'gamma': 'scale'}
        assert loaded.model_data == {'X_train': [[1, 2], [3, 4]], 'y_train': [0, 1]}


# ===================================================================
# Data schema
# ===================================================================
class TestDataSchema:
    def test_label_names_include_required(self):
        for name in ['OPEN_PALM', 'POINT_ONE', 'PINCH_CLOSED', 'UNKNOWN', 'OTHER']:
            assert name in LABEL_NAMES

    def test_sample_metadata_fields(self):
        meta = SampleMetadata(
            user_id='u1',
            session_id='s1',
            handedness='Right',
            distance_cm=50.0,
            lighting='bright',
            timestamp=1234567890.0,
            label='OPEN_PALM',
        )
        assert meta.is_transition == False
        assert meta.user_id == 'u1'


# ===================================================================
# Safety
# ===================================================================
class TestRateLimiter:
    def test_first_event_allowed(self):
        rl = RateLimiter(max_events_per_second=10)
        assert rl.allow() is True

    def test_rapid_events_throttled(self):
        rl = RateLimiter(max_events_per_second=1)
        assert rl.allow() is True
        assert rl.allow() is False  # immediate second call blocked


class TestFailsafeMonitor:
    def test_corner_triggers_stop(self):
        monitor = FailsafeMonitor(1920, 1080, corner_size=10)
        assert monitor.check_position(5, 5) is True
        assert monitor.is_stopped()

    def test_center_safe(self):
        monitor = FailsafeMonitor(1920, 1080, corner_size=10)
        assert monitor.check_position(960, 540) is False
        assert not monitor.is_stopped()


class TestEmergencyHotkey:
    def test_trigger_calls_callback(self):
        called = []
        hotkey = EmergencyHotkey("ctrl+alt+q", lambda: called.append(True), dry_run=True)
        hotkey.trigger_emergency()
        assert len(called) == 1

    def test_dry_run_no_registration(self):
        hotkey = EmergencyHotkey("ctrl+alt+q", lambda: None, dry_run=True)
        hotkey.register()  # should be no-op
        assert not hotkey._registered


# ===================================================================
# Config loading
# ===================================================================
class TestConfigLoading:
    def test_load_default_config(self):
        from src.config.loader import load_config
        import src.config.loader as loader_module
        # Reset singleton
        loader_module._config_instance = None

        config = load_config()
        assert 'feature' in config
        assert 'ml' in config
        assert 'gesture' in config
        assert 'platform' in config
        assert 'safety' in config

        # Check specific values
        assert config['ml']['algorithm'] == 'rbf_svm'
        assert config['ml']['C'] == 10.0
        assert config['feature']['version'] == 1

        # Cleanup singleton
        loader_module._config_instance = None

    def test_get_config_singleton(self):
        from src.config.loader import get_config, load_config
        import src.config.loader as loader_module
        loader_module._config_instance = None

        config1 = get_config()
        config2 = get_config()
        assert config1 is config2  # same object

        loader_module._config_instance = None
