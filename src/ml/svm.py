"""RBF-SVM classifier for canonicalized static hand poses."""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.gesture.pose import HandPose


class SVMClassifier:
    """StandardScaler + RBF SVC with score- and margin-based rejection."""

    EXCLUDED_TRAINING_LABELS = {"NO_HAND", "UNKNOWN"}
    EXCLUDED_TRAINING_VALUES = {int(HandPose.NO_HAND), int(HandPose.UNKNOWN)}

    def __init__(
        self,
        C: float = 10.0,
        gamma: Union[str, float] = "scale",
        class_weight: Optional[Union[str, dict]] = "balanced",
    ) -> None:
        self.C = float(C)
        self.gamma = gamma
        self.class_weight = class_weight
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[SVC] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "SVMClassifier":
        return self._fit(X, y, scaler=None)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler: Optional[StandardScaler],
    ) -> "SVMClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2 or len(X_arr) != len(y_arr) or len(X_arr) == 0:
            raise ValueError("X and y must contain the same non-zero number of samples")

        def is_trainable(label: Any) -> bool:
            if str(label) in self.EXCLUDED_TRAINING_LABELS:
                return False
            try:
                return int(label) not in self.EXCLUDED_TRAINING_VALUES
            except (TypeError, ValueError):
                return True

        keep = np.array([is_trainable(label) for label in y_arr])
        X_arr, y_arr = X_arr[keep], y_arr[keep]
        if len(np.unique(y_arr)) < 2:
            raise ValueError("RBF-SVM training requires at least two trainable pose classes")

        # Normal training always fits after filtering. The private scaler path is
        # used only when restoring the exact train-only scaler from a bundle.
        self.scaler = scaler or StandardScaler().fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)
        self.model = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            decision_function_shape="ovr",
        ).fit(X_scaled, y_arr)
        return self

    def _require_fitted(self) -> tuple[StandardScaler, SVC]:
        if self.scaler is None or self.model is None:
            raise RuntimeError("SVMClassifier must be fit before prediction")
        return self.scaler, self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        scaler, model = self._require_fitted()
        return model.predict(scaler.transform(np.asarray(X, dtype=float)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scaler, model = self._require_fitted()
        scores = np.asarray(
            model.decision_function(scaler.transform(np.asarray(X, dtype=float))),
            dtype=float,
        )
        if scores.ndim == 1:
            # sklearn returns one signed score for binary SVC. Expand it so the
            # same top-score and top-two-margin policy applies to both classes.
            scores = np.column_stack((-scores, scores))
        return scores

    def predict_with_rejection(
        self,
        X: np.ndarray,
        decision_threshold: float = 0.0,
        margin_threshold: float = 0.15,
        unknown_label: Any = "UNKNOWN",
    ) -> np.ndarray:
        _, model = self._require_fitted()
        scores = self.decision_function(X)
        best_indices = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(len(scores)), best_indices]
        if scores.shape[1] > 1:
            top_two = np.partition(scores, -2, axis=1)[:, -2:]
            margins = top_two[:, 1] - top_two[:, 0]
        else:  # Defensive; fitted SVC always has at least two classes.
            margins = np.full(len(scores), np.inf)

        predictions = np.asarray(model.classes_[best_indices], dtype=object)
        reject = (best_scores < decision_threshold) | (margins < margin_threshold)
        predictions[reject] = unknown_label
        return predictions
