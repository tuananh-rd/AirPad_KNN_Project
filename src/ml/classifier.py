"""Common contract for static-pose classifiers used by the runtime pipeline."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class PoseClassifier(Protocol):
    """Classifier that can reject uncertain static-pose predictions."""

    def predict_with_rejection(
        self,
        X: np.ndarray,
        decision_threshold: float,
        margin_threshold: float,
        unknown_label: Any = "UNKNOWN",
    ) -> np.ndarray:
        ...
