"""Unit tests for tools.collect_data, tools.train, and synthetic replay fixtures.

Covers:
- Dataset metadata validation and feature shape checks (tools/collect_data.py).
- Save and load round-trip for .npz and .csv (tools/collect_data.py).
- Dataset validation rejecting NaNs, infs, and invalid feature versions (tools/train.py).
- Grouped splitting (GroupKFold & LOSO) ensuring zero group overlap between train and test folds.
- Scaler fitting on train fold only with proof of zero test data leakage.
- Hyperparameter tuning (tune_svm) returning best parameters and measurement logs.
- ModelBundle export and compatibility with RBF-SVM inference.
- Verification of fixtures/synthetic_replay.json structural integrity and phase sequence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.gesture.pose import HandPose
from src.ml.bundle import ModelBundle
from src.ml.evaluation import group_kfold_split, leave_one_subject_out
from src.ml.svm import SVMClassifier
from tools.collect_data import (
    LABEL_NAMES,
    REQUIRED_METADATA_KEYS,
    collect_data_interactive,
    create_sample_metadata,
    export_dataset_csv,
    load_dataset_csv,
    load_dataset_npz,
    save_dataset_npz,
    validate_sample,
)
from tools.train import (
    FeatureScaler,
    create_model_bundle,
    evaluate_fold,
    extract_groups,
    load_bundle,
    load_model_from_bundle,
    main as train_main,
    save_bundle,
    train_from_dataset,
    tune_svm,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_metadata() -> Dict[str, Any]:
    return {
        "user_id": "subject_01",
        "session_id": "session_2026_09_29_a",
        "timestamp": 123456.78,
        "handedness": "Right",
        "distance_cm": 45.0,
        "lighting": "bright",
        "label": "POINT_ONE",
        "is_transition": False,
        "feature_version": 1,
    }


@pytest.fixture
def valid_features() -> np.ndarray:
    # 63 features (21 landmarks x 3 coords)
    rng = np.random.RandomState(42)
    return rng.uniform(0.0, 1.0, size=(63,)).astype(np.float32)


@pytest.fixture
def small_dataset() -> tuple[np.ndarray, list[str], list[dict[str, Any]], np.ndarray]:
    """Synthetic dataset with 60 samples, 3 classes, 4 distinct sessions."""
    rng = np.random.RandomState(42)
    n_samples = 60
    classes = ["OPEN_PALM", "POINT_ONE", "PINCH_CLOSED"]
    sessions = ["sess_A", "sess_B", "sess_C", "sess_D"]

    features = rng.normal(loc=0.5, scale=0.2, size=(n_samples, 63)).astype(np.float32)
    labels = [classes[i % len(classes)] for i in range(n_samples)]
    session_groups = np.array([sessions[i // 15] for i in range(n_samples)])

    metadata = []
    for i in range(n_samples):
        meta = create_sample_metadata(
            user_id=f"user_{session_groups[i][-1]}",
            session_id=str(session_groups[i]),
            timestamp=1000.0 + i * 33.3,
            handedness="Right" if i % 2 == 0 else "Left",
            distance_cm=40.0 + (i % 10),
            lighting="bright" if i % 3 == 0 else "dim",
            label=labels[i],
            is_transition=False,
            feature_version=1,
        )
        metadata.append(meta)

    return features, labels, metadata, session_groups


# ---------------------------------------------------------------------------
# 1. Tests for tools/collect_data.py
# ---------------------------------------------------------------------------
class TestCollectData:
    def test_validate_sample_valid(self, valid_features, valid_metadata):
        assert validate_sample(valid_features, valid_metadata) is True

    def test_validate_sample_omitted_feature_version_defaults_to_1(self, valid_features, valid_metadata):
        meta = valid_metadata.copy()
        del meta["feature_version"]
        assert validate_sample(valid_features, meta) is True

    @pytest.mark.parametrize("bad_shape", [(62,), (64,), (21, 3), (1, 63), (0,)])
    def test_validate_sample_invalid_feature_shapes(self, bad_shape, valid_metadata):
        bad_features = np.zeros(bad_shape, dtype=np.float32)
        assert validate_sample(bad_features, valid_metadata) is False

    def test_validate_sample_not_numpy(self, valid_metadata):
        assert validate_sample([0.1] * 63, valid_metadata) is False  # type: ignore

    def test_validate_sample_nan_and_inf(self, valid_features, valid_metadata):
        feats_nan = valid_features.copy()
        feats_nan[10] = np.nan
        assert validate_sample(feats_nan, valid_metadata) is False

        feats_inf = valid_features.copy()
        feats_inf[20] = np.inf
        assert validate_sample(feats_inf, valid_metadata) is False

    @pytest.mark.parametrize("missing_key", sorted(list(REQUIRED_METADATA_KEYS)))
    def test_validate_sample_missing_required_keys(self, valid_features, valid_metadata, missing_key):
        bad_meta = valid_metadata.copy()
        del bad_meta[missing_key]
        assert validate_sample(valid_features, bad_meta) is False

    def test_validate_sample_invalid_label(self, valid_features, valid_metadata):
        bad_meta = valid_metadata.copy()
        bad_meta["label"] = "UNREGISTERED_POSE_XYZ"
        assert validate_sample(valid_features, bad_meta) is False

    def test_validate_sample_invalid_handedness(self, valid_features, valid_metadata):
        bad_meta = valid_metadata.copy()
        bad_meta["handedness"] = "Ambidextrous"
        assert validate_sample(valid_features, bad_meta) is False

    def test_validate_sample_invalid_lighting(self, valid_features, valid_metadata):
        bad_meta = valid_metadata.copy()
        bad_meta["lighting"] = "pitch_black"
        assert validate_sample(valid_features, bad_meta) is False

    @pytest.mark.parametrize("bad_version", [2, 0, -1, "1", None])
    def test_validate_sample_invalid_feature_version(self, valid_features, valid_metadata, bad_version):
        bad_meta = valid_metadata.copy()
        bad_meta["feature_version"] = bad_version
        assert validate_sample(valid_features, bad_meta) is False

    def test_npz_save_and_load_round_trip(self, tmp_path, small_dataset):
        features, labels, metadata, _ = small_dataset
        npz_path = str(tmp_path / "dataset.npz")

        save_dataset_npz(npz_path, features, labels, metadata)
        assert os.path.exists(npz_path)

        loaded_feats, loaded_labels, loaded_meta = load_dataset_npz(npz_path)

        np.testing.assert_allclose(loaded_feats, features, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(loaded_labels, labels)
        assert len(loaded_meta) == len(metadata)
        for original, loaded in zip(metadata, loaded_meta):
            assert original["user_id"] == loaded["user_id"]
            assert original["session_id"] == loaded["session_id"]
            assert original["label"] == loaded["label"]
            assert original["feature_version"] == loaded["feature_version"]

    def test_npz_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset_npz("nonexistent_dataset_file.npz")

    def test_npz_length_mismatch_raises(self, small_dataset):
        features, labels, metadata, _ = small_dataset
        with pytest.raises(ValueError):
            save_dataset_npz("dummy.npz", features[:10], labels[:5], metadata[:10])

    def test_csv_save_and_load_round_trip(self, tmp_path, small_dataset):
        features, labels, metadata, _ = small_dataset
        csv_path = str(tmp_path / "dataset.csv")

        export_dataset_csv(csv_path, features, labels, metadata)
        assert os.path.exists(csv_path)

        loaded_feats, loaded_labels, loaded_meta = load_dataset_csv(csv_path)

        np.testing.assert_allclose(loaded_feats, features, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(loaded_labels, labels)
        assert len(loaded_meta) == len(metadata)
        for orig, loaded in zip(metadata, loaded_meta):
            assert orig["session_id"] == loaded["session_id"]
            assert orig["label"] == loaded["label"]
            assert orig["is_transition"] == loaded["is_transition"]
            assert loaded["feature_version"] == 1

    def test_collect_data_mock_session(self, tmp_path):
        output_npz = str(tmp_path / "mock_custom.npz")
        rng = np.random.RandomState(42)
        dummy_landmarks = rng.uniform(0.1, 0.9, size=(21, 3))
        mock_frames = [(dummy_landmarks, "Right"), (dummy_landmarks, "Left")]

        exit_code = collect_data_interactive(
            output_path=output_npz,
            user_id="subject_alpha",
            session_id="sess_001",
            mock_frames=mock_frames,
        )
        assert exit_code == 0
        assert os.path.exists(output_npz)

        features, labels, metadata = load_dataset_npz(output_npz)
        assert len(features) == 2
        assert len(labels) == 2
        assert labels[0] == "PINCH_CLOSED"
        assert metadata[0]["user_id"] == "subject_alpha"
        assert metadata[0]["session_id"] == "sess_001"
        assert metadata[0]["handedness"] == "Right"
        assert metadata[1]["handedness"] == "Left"

    def test_collect_data_append_session(self, tmp_path):
        output_npz = str(tmp_path / "mock_append.npz")
        rng = np.random.RandomState(42)
        dummy_landmarks = rng.uniform(0.1, 0.9, size=(21, 3))

        # First session
        collect_data_interactive(
            output_path=output_npz,
            user_id="subject_alpha",
            session_id="sess_001",
            mock_frames=[(dummy_landmarks, "Right")],
        )
        # Second session appended
        collect_data_interactive(
            output_path=output_npz,
            user_id="subject_beta",
            session_id="sess_002",
            mock_frames=[(dummy_landmarks, "Left")],
        )

        features, labels, metadata = load_dataset_npz(output_npz)
        assert len(features) == 2
        assert metadata[0]["user_id"] == "subject_alpha"
        assert metadata[1]["user_id"] == "subject_beta"

    def test_collect_data_opencv_bgr_to_mediapipe_rgb_contract(self, tmp_path):
        """Assert exact RGB-channel order reaches detector after flip while displayed frame stays BGR (ACC-03)."""
        output_npz = str(tmp_path / "camera_contract.npz")

        # Create synthetic BGR frame with distinct channel values and left/right spatial asymmetry
        raw_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        raw_bgr[:, :100, 0] = 10   # Blue channel (left)
        raw_bgr[:, 100:, 0] = 30   # Blue channel (right)
        raw_bgr[:, :, 1] = 50      # Green channel
        raw_bgr[:, :100, 2] = 200  # Red channel (left)
        raw_bgr[:, 100:, 2] = 240  # Red channel (right)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, raw_bgr.copy()), (True, raw_bgr.copy()), (False, None)]

        detected_frames = []
        mock_detector = MagicMock()
        def mock_detect(frame_arg):
            detected_frames.append(frame_arg.copy())
            return []
        mock_detector.detect.side_effect = mock_detect
        mock_detector_cls = MagicMock(return_value=mock_detector)

        displayed_frames = []
        def mock_imshow(win_name, frame_arg):
            displayed_frames.append(frame_arg.copy())

        # First waitKey iteration returns -1 (allows rendering), second returns 27 (ESC exit)
        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("src.vision.adapter.MediaPipeAdapter", mock_detector_cls), \
             patch("cv2.imshow", side_effect=mock_imshow), \
             patch("cv2.waitKey", side_effect=[-1, 27]):
            exit_code = collect_data_interactive(output_path=output_npz)

        assert exit_code == 0
        assert len(detected_frames) >= 1
        assert len(displayed_frames) >= 1

        det_frame = detected_frames[0]
        disp_frame = displayed_frames[0]

        # 1. Shape check
        assert det_frame.shape == (200, 200, 3)
        assert disp_frame.shape == (200, 200, 3)

        # 2. Channel order check: detector receives RGB, displayed frame remains BGR
        # In raw BGR frame:
        # Left half (:100) has B=10, G=50, R=200
        # Right half (100:) has B=30, G=50, R=240
        # After horizontal flip cv2.flip(frame, 1):
        # Left half (:100) gets the old right half (B=30, G=50, R=240)
        # Right half (100:) gets the old left half (B=10, G=50, R=200)

        # For displayed frame, HUD text is drawn at y<=70 and y>=180.
        # Inspect region y in [100:150] which is guaranteed text-free:
        np.testing.assert_array_equal(disp_frame[100:150, :100, 0], 30)   # Blue
        np.testing.assert_array_equal(disp_frame[100:150, :100, 1], 50)   # Green
        np.testing.assert_array_equal(disp_frame[100:150, :100, 2], 240)  # Red

        np.testing.assert_array_equal(disp_frame[100:150, 100:, 0], 10)   # Blue
        np.testing.assert_array_equal(disp_frame[100:150, 100:, 1], 50)   # Green
        np.testing.assert_array_equal(disp_frame[100:150, 100:, 2], 200)  # Red

        # In detected frame (converted to RGB before any HUD drawing):
        # Channel 0 must be Red
        # Channel 1 must be Green
        # Channel 2 must be Blue
        np.testing.assert_array_equal(det_frame[:, :100, 0], 240)  # Red
        np.testing.assert_array_equal(det_frame[:, :100, 1], 50)   # Green
        np.testing.assert_array_equal(det_frame[:, :100, 2], 30)   # Blue

        np.testing.assert_array_equal(det_frame[:, 100:, 0], 200)  # Red
        np.testing.assert_array_equal(det_frame[:, 100:, 1], 50)   # Green
        np.testing.assert_array_equal(det_frame[:, 100:, 2], 10)   # Blue

        # Explicit channel-swap correspondence between detected RGB and displayed BGR in text-free region
        np.testing.assert_array_equal(det_frame[100:150, :, 0], disp_frame[100:150, :, 2])
        np.testing.assert_array_equal(det_frame[100:150, :, 1], disp_frame[100:150, :, 1])
        np.testing.assert_array_equal(det_frame[100:150, :, 2], disp_frame[100:150, :, 0])


# ---------------------------------------------------------------------------
# 2. Tests for tools/train.py - Dataset Validation
# ---------------------------------------------------------------------------
class TestTrainDataValidation:
    def test_validate_dataset_valid(self, small_dataset):
        features, labels, metadata, _ = small_dataset
        assert validate_dataset(features, labels, metadata) is True

    def test_validate_dataset_rejects_nans(self, small_dataset):
        features, labels, metadata, _ = small_dataset
        corrupt_features = features.copy()
        corrupt_features[5, 12] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_dataset(corrupt_features, labels, metadata)

    def test_validate_dataset_rejects_infs(self, small_dataset):
        features, labels, metadata, _ = small_dataset
        corrupt_features = features.copy()
        corrupt_features[3, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            validate_dataset(corrupt_features, labels, metadata)

    def test_validate_dataset_rejects_invalid_feature_version(self, small_dataset):
        features, labels, metadata, _ = small_dataset
        corrupt_metadata = [dict(m) for m in metadata]
        corrupt_metadata[10]["feature_version"] = 2  # version 2 is invalid (only version 1 accepted)

        with pytest.raises(ValueError, match="feature_version"):
            validate_dataset(features, labels, corrupt_metadata)

    def test_validate_dataset_rejects_missing_feature_version(self, small_dataset):
        features, labels, metadata, _ = small_dataset
        corrupt_metadata = [dict(m) for m in metadata]
        del corrupt_metadata[0]["feature_version"]

        with pytest.raises(ValueError, match="feature_version"):
            validate_dataset(features, labels, corrupt_metadata)

    def test_validate_dataset_rejects_invalid_label(self, small_dataset):
        features, labels, metadata, _ = small_dataset
        corrupt_labels = list(labels)
        corrupt_labels[2] = "INVALID_NON_EXISTENT_GESTURE"

        with pytest.raises(ValueError, match="invalid label"):
            validate_dataset(features, corrupt_labels, metadata)

    def test_validate_dataset_rejects_shape_mismatch(self, small_dataset):
        features, labels, metadata, _ = small_dataset
        bad_features = np.zeros((len(features), 20), dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            validate_dataset(bad_features, labels, metadata)


# ---------------------------------------------------------------------------
# 3. Tests for Grouped Splitting (GroupKFold / LOSO)
# ---------------------------------------------------------------------------
class TestGroupedSplitting:
    def test_group_kfold_zero_group_overlap(self, small_dataset):
        """Verify GroupKFold ensures zero overlap of groups between train and test."""
        features, labels, metadata, groups = small_dataset
        y = np.array(labels)

        split_count = 0
        for train_idx, test_idx in group_kfold_split(features, y, groups, n_splits=3):
            split_count += 1
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # Invariant: zero group leakage between folds
            overlap = train_groups.intersection(test_groups)
            assert len(overlap) == 0, f"Found leaked groups across split: {overlap}"
            assert len(train_idx) + len(test_idx) == len(groups)

        assert split_count >= 2

    def test_loso_zero_group_overlap(self, small_dataset):
        """Leave-one-subject-out CV must isolate exactly one subject per test fold."""
        features, labels, metadata, _ = small_dataset
        subjects = np.array([m["user_id"] for m in metadata])
        unique_subjects = np.unique(subjects)
        y = np.array(labels)

        fold_count = 0
        for train_idx, test_idx in leave_one_subject_out(features, y, subjects):
            fold_count += 1
            train_subjects = set(subjects[train_idx])
            test_subjects = set(subjects[test_idx])

            assert len(test_subjects) == 1, "LOSO test fold must contain exactly one subject"
            assert train_subjects.isdisjoint(test_subjects), "Subject leaked across train and test folds"

        assert fold_count == len(unique_subjects)

    def test_grouped_split_preserves_temporal_sessions(self, small_dataset):
        """Consecutive frames from the same session must never be split across train and test."""
        features, labels, metadata, groups = small_dataset
        y = np.array(labels)

        for train_idx, test_idx in group_kfold_split(features, y, groups, n_splits=4):
            train_set = set(train_idx)
            test_set = set(test_idx)

            for g in np.unique(groups):
                g_indices = set(np.where(groups == g)[0])
                # All indices for group g must be completely in train OR completely in test
                in_train = g_indices.issubset(train_set)
                in_test = g_indices.issubset(test_set)
                assert in_train or in_test, f"Session {g} was split across train and test folds!"


# ---------------------------------------------------------------------------
# 4. Tests for Scaler Zero-Leakage Fitting
# ---------------------------------------------------------------------------
class TestScalerNoLeakage:
    def test_scaler_fit_on_train_fold_only(self):
        """Proves that test statistics DO NOT leak into the fitted scaler.

        Train fold has mean=0.0; Test fold has extreme mean=1000.0.
        Fitting on train fold ONLY must keep mean_ near 0.0.
        """
        rng = np.random.RandomState(42)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(100, 63))
        X_test = rng.normal(loc=1000.0, scale=1.0, size=(50, 63))

        scaler = FeatureScaler().fit(X_train)

        # Scaler mean must match X_train mean (~0.0), NOT contaminated by X_test (~1000.0)
        assert np.all(np.abs(scaler.mean_) < 0.5), "Scaler mean was contaminated or incorrect"
        np.testing.assert_allclose(scaler.mean_, np.mean(X_train, axis=0))

        # Transform test data using TRAIN statistics
        X_test_scaled = scaler.transform(X_test)
        # Transformed test data should have mean ~1000.0 (since normalized with train mean 0.0)
        assert np.all(np.mean(X_test_scaled, axis=0) > 800.0)

        # Contrast: if whole dataset was leaked during fitting, mean would be ~333.3
        leaked_scaler = FeatureScaler().fit(np.vstack([X_train, X_test]))
        assert np.all(leaked_scaler.mean_ > 200.0), "Sanity check on data leakage simulation failed"

    def test_scaler_serialization_round_trip(self):
        rng = np.random.RandomState(42)
        X = rng.normal(loc=2.5, scale=1.8, size=(40, 63))

        scaler = FeatureScaler().fit(X)
        scaler_dict = scaler.to_dict()

        restored_scaler = FeatureScaler.from_dict(scaler_dict)

        np.testing.assert_allclose(scaler.mean_, restored_scaler.mean_)
        np.testing.assert_allclose(scaler.scale_, restored_scaler.scale_)

        transformed_orig = scaler.transform(X[:5])
        transformed_restored = restored_scaler.transform(X[:5])
        np.testing.assert_allclose(transformed_orig, transformed_restored)

    def test_scaler_zero_variance_guard(self):
        """Constant features (zero variance) must scale safely without division by zero or NaNs."""
        X = np.ones((20, 63), dtype=float)
        scaler = FeatureScaler().fit(X)
        transformed = scaler.transform(X)

        assert not np.any(np.isnan(transformed))
        assert not np.any(np.isinf(transformed))
        np.testing.assert_allclose(transformed, 0.0)


# ---------------------------------------------------------------------------
# 5. Tests for Hyperparameter Tuning and ModelBundle
# ---------------------------------------------------------------------------
class TestModelBundleAndSVM:
    def test_tune_svm(self, small_dataset):
        features, labels, metadata, groups = small_dataset
        res = tune_svm(
            features,
            labels,
            groups,
            C_values=[1.0, 10.0],
            gamma_values=["scale", 0.01],
            n_splits=2,
            use_scaler=True,
        )

        assert "best_params" in res
        assert "best_score" in res
        assert "measurements" in res
        assert res["best_params"]["C"] in [1.0, 10.0]
        assert res["best_params"]["gamma"] in ["scale", 0.01]
        assert len(res["measurements"]) == 4

        for m in res["measurements"]:
            assert "C" in m
            assert "gamma" in m
            assert "mean_macro_f1" in m
            assert "std_macro_f1" in m
            assert isinstance(m["fold_scores"], list)

    def test_model_bundle_export_and_load_compatibility(self, tmp_path, small_dataset):
        features, labels, metadata, _ = small_dataset
        bundle_path = str(tmp_path / "model_bundle.json")

        bundle = create_model_bundle(
            X=features,
            y=labels,
            C=10.0,
            gamma="scale",
            validation_metrics={"macro_f1": 0.95},
            extra_metadata={"trained_by": "test_suite"},
        )

        assert bundle.feature_version == 1
        assert bundle.algorithm == "rbf_svm"
        assert bundle.hyperparameters["C"] == 10.0
        assert bundle.train_samples == len(features)
        assert "scaler" in bundle.metadata
        assert bundle.metadata["validation_metrics"]["macro_f1"] == 0.95

        save_bundle(bundle, bundle_path)
        assert os.path.exists(bundle_path)

        loaded_bundle = load_bundle(bundle_path)
        assert loaded_bundle.feature_version == 1
        assert loaded_bundle.algorithm == "rbf_svm"
        assert loaded_bundle.hyperparameters["gamma"] == "scale"

        classifier, restored_scaler, _ = load_model_from_bundle(loaded_bundle)
        assert isinstance(classifier, SVMClassifier)
        assert restored_scaler is not None

        # Re-predict on a subset and ensure deterministic, valid class output
        test_query = features[:5]
        preds = classifier.predict(test_query)

        assert len(preds) == 5
        for p in preds:
            assert p in loaded_bundle.labels

    def test_bundle_excludes_runtime_labels_before_scaler_fit(self):
        features = np.vstack([
            np.full((2, 63), -1.0),
            np.full((2, 63), 1.0),
            np.full((1, 63), 1000.0),
            np.full((1, 63), 2000.0),
        ])
        labels = ["OPEN_PALM", "OPEN_PALM", "POINT_ONE", "POINT_ONE", "UNKNOWN", "NO_HAND"]

        bundle = create_model_bundle(features, labels)

        assert bundle.train_samples == 4
        assert set(bundle.model_data["y_train"]) == {"OPEN_PALM", "POINT_ONE"}
        np.testing.assert_allclose(bundle.metadata["scaler"]["mean"], 0.0)


# ---------------------------------------------------------------------------
# 6. Tests for Training CLI (tools/train.py)
# ---------------------------------------------------------------------------
class TestTrainCLI:
    def test_train_cli_end_to_end(self, tmp_path, small_dataset):
        features, labels, metadata, _ = small_dataset
        npz_path = str(tmp_path / "test_data.npz")
        bundle_path = str(tmp_path / "output_bundle.json")
        summary_path = str(tmp_path / "summary.json")

        save_dataset_npz(npz_path, features, labels, metadata)

        exit_code = train_main([
            npz_path,
            bundle_path,
            "--group-key", "user_id",
            "--n-splits", "3",
            "--c-values", "1.0", "10.0",
            "--gamma-values", "scale",
            "--summary-path", summary_path,
        ])
        assert exit_code == 0
        assert os.path.exists(bundle_path)
        assert os.path.exists(summary_path)

        bundle = load_bundle(bundle_path)
        assert bundle.feature_version == 1
        assert bundle.algorithm == "rbf_svm"
        assert bundle.train_samples == len(features)
        assert "validation_metrics" in bundle.metadata
        assert "macro_f1" in bundle.metadata["validation_metrics"]

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["n_samples"] == len(features)
        assert "macro_f1" in summary
        assert "per_class" in summary
        assert "confusion_matrix" in summary

    def test_train_cli_missing_dataset(self, tmp_path):
        bundle_path = str(tmp_path / "output_bundle.json")
        exit_code = train_main([
            "nonexistent_dataset.npz",
            bundle_path,
        ])
        assert exit_code == 1

    def test_train_cli_filters_unknown_and_no_hand(self, tmp_path, small_dataset):
        features, labels, metadata, _ = small_dataset
        extra_features = np.zeros((2, 63), dtype=np.float32)
        extra_labels = ["UNKNOWN", "NO_HAND"]
        extra_meta = [
            create_sample_metadata(
                user_id="user_X",
                session_id="sess_X",
                timestamp=9999.0,
                handedness="Right",
                distance_cm=40.0,
                lighting="bright",
                label="UNKNOWN",
                is_transition=False,
                feature_version=1,
            ),
            create_sample_metadata(
                user_id="user_X",
                session_id="sess_X",
                timestamp=10000.0,
                handedness="Right",
                distance_cm=40.0,
                lighting="bright",
                label="NO_HAND",
                is_transition=False,
                feature_version=1,
            ),
        ]
        all_features = np.vstack([features, extra_features])
        all_labels = labels + extra_labels
        all_meta = metadata + extra_meta

        npz_path = str(tmp_path / "mixed_data.npz")
        bundle_path = str(tmp_path / "filtered_bundle.json")
        save_dataset_npz(npz_path, all_features, all_labels, all_meta)

        exit_code = train_main([
            npz_path,
            bundle_path,
            "--group-key", "user_id",
            "--n-splits", "3",
        ])
        assert exit_code == 0
        bundle = load_bundle(bundle_path)
        assert bundle.train_samples == len(features)
        assert "UNKNOWN" not in bundle.labels
        assert "NO_HAND" not in bundle.labels


# ---------------------------------------------------------------------------
# 7. Tests for Synthetic Replay Fixture
# ---------------------------------------------------------------------------
class TestSyntheticReplayFixture:
    @pytest.fixture
    def replay_fixture_path(self) -> Path:
        base_dir = Path(__file__).parent.parent
        return base_dir / "fixtures" / "synthetic_replay.json"

    def test_fixture_file_exists(self, replay_fixture_path):
        assert replay_fixture_path.exists(), f"Fixture file not found at {replay_fixture_path}"

    def test_fixture_structure_and_frame_count(self, replay_fixture_path):
        with open(replay_fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list), "synthetic_replay.json must be a JSON array of frames"
        n_frames = len(data)
        assert 30 <= n_frames <= 40, f"Expected 30-40 frames, got {n_frames}"

        # Verify monotonicity of timestamps
        timestamps = [f["timestamp_ms"] for f in data]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], f"Non-monotonic timestamp at frame {i}"

    def test_fixture_phases_sequence(self, replay_fixture_path):
        """Verify the 5 distinct phases specified in the requirements:
        - Frame 0-5: No hand / open palm
        - Frame 6-15: POINT_ONE hovering (cursor movement)
        - Frame 16-22: PINCH_CLOSED quick click
        - Frame 23-30: PINCH_CLOSED hold for drag
        - Frame 31-35: Release / hand loss
        """
        with open(replay_fixture_path, "r", encoding="utf-8") as f:
            frames = json.load(f)

        # Phase 1: Frame 0-5
        for i in range(0, 6):
            assert frames[i]["pose"] in ("NO_HAND", "OPEN_PALM")

        # Phase 2: Frame 6-15
        for i in range(6, 16):
            assert frames[i]["pose"] == "POINT_ONE"

        # Phase 3: Frame 16-22
        # Includes pinch closed and clean release to point one
        pinch_click_poses = [frames[i]["pose"] for i in range(16, 23)]
        assert "PINCH_CLOSED" in pinch_click_poses
        assert "POINT_ONE" in pinch_click_poses

        # Phase 4: Frame 23-30
        drag_poses = [frames[i]["pose"] for i in range(23, 31)]
        assert "PINCH_CLOSED" in drag_poses

        # Phase 5: Frame 31-35
        end_poses = [frames[i]["pose"] for i in range(31, len(frames))]
        assert "NO_HAND" in end_poses

    def test_fixture_landmarks_validity(self, replay_fixture_path):
        with open(replay_fixture_path, "r", encoding="utf-8") as f:
            frames = json.load(f)

        for frame in frames:
            landmarks = frame.get("landmarks")
            if frame["pose"] == "NO_HAND":
                assert landmarks is None or len(landmarks) == 0
            else:
                assert landmarks is not None
                assert len(landmarks) == 21, f"Hand frame {frame['frame_index']} must have 21 landmarks"

                # Check coordinates in [0, 1] for x and y
                for pt in landmarks:
                    x, y, z = pt[0], pt[1], pt[2]
                    assert 0.0 <= x <= 1.0, f"Frame {frame['frame_index']} x coord out of range: {x}"
                    assert 0.0 <= y <= 1.0, f"Frame {frame['frame_index']} y coord out of range: {y}"


# ---------------------------------------------------------------------------
# 8. Tests for tools/merge_datasets.py
# ---------------------------------------------------------------------------
class TestMergeDatasets:
    def test_merge_npz_datasets(self, tmp_path, small_dataset):
        from tools.merge_datasets import merge_npz_datasets
        features, labels, metadata, _ = small_dataset

        ds1_path = str(tmp_path / "ds1.npz")
        ds2_path = str(tmp_path / "ds2.npz")
        merged_path = str(tmp_path / "merged.npz")

        half = len(features) // 2
        save_dataset_npz(ds1_path, features[:half], labels[:half], metadata[:half])
        save_dataset_npz(ds2_path, features[half:], labels[half:], metadata[half:])

        exit_code = merge_npz_datasets([ds1_path, ds2_path], merged_path)
        assert exit_code == 0
        assert os.path.exists(merged_path)

        m_feat, m_lbl, m_meta = load_dataset_npz(merged_path)
        assert len(m_feat) == len(features)
        assert len(m_lbl) == len(labels)
        assert len(m_meta) == len(metadata)
