"""Focused tests for the production RBF-SVM pose classifier."""
from unittest.mock import MagicMock

import numpy as np

from src.gesture.pose import HandPose
from src.ml.svm import SVMClassifier


def test_svm_scales_once_per_rejected_prediction():
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    y = np.array(["LEFT", "LEFT", "RIGHT", "RIGHT"])
    classifier = SVMClassifier(C=10.0).fit(X, y)
    original_transform = classifier.scaler.transform
    classifier.scaler.transform = MagicMock(wraps=original_transform)

    result = classifier.predict_with_rejection(
        np.array([[1.5, 1.5]]), decision_threshold=-1.0, margin_threshold=0.0
    )

    assert result.tolist() == ["RIGHT"]
    assert classifier.scaler.transform.call_count == 1


def test_svm_excludes_runtime_only_labels_for_strings_and_enums():
    X = np.array([
        [-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0],
        [50.0, 50.0], [60.0, 60.0],
    ])
    y = np.array([
        int(HandPose.OPEN_PALM), int(HandPose.OPEN_PALM),
        int(HandPose.POINT_ONE), int(HandPose.POINT_ONE),
        int(HandPose.UNKNOWN), int(HandPose.NO_HAND),
    ])

    classifier = SVMClassifier().fit(X, y)

    assert set(classifier.model.classes_) == {
        int(HandPose.OPEN_PALM), int(HandPose.POINT_ONE)
    }


def test_svm_rejects_small_margin_as_unknown():
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array(["A", "A", "B", "B"])
    classifier = SVMClassifier().fit(X, y)

    result = classifier.predict_with_rejection(
        np.array([[0.0]]), decision_threshold=-10.0, margin_threshold=10.0
    )

    assert result.tolist() == ["UNKNOWN"]
