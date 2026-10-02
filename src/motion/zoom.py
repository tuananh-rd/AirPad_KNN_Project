"""Zoom engine for ZOOM_LATCHED mode (entered via L_SHAPE dwell).

Contract:
- Tracks landmark distance delta (e.g. thumb tip to index tip aperture).
- Emits semantic zoom steps (+1 zoom in, -1 zoom out).
- Does NOT trigger pinch clicks.
- Baseline resets cleanly on entering/exiting zoom mode.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np


class ZoomEngine:
    """Computes discrete semantic zoom steps from distance changes in ZOOM_LATCHED."""

    def __init__(
        self,
        step_threshold: float = 0.035,
        dead_zone: float = 0.01,
        cooldown_ms: float = 80.0,
    ):
        self.step_threshold = float(step_threshold)
        self.dead_zone = float(dead_zone)
        self.cooldown_ms = float(cooldown_ms)

        self._baseline_distance: Optional[float] = None
        self._accum_delta: float = 0.0
        self._last_step_time_ms: float = 0.0

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline_distance

    def set_baseline(self, distance: float) -> None:
        """Explicitly set or recalibrate baseline aperture distance."""
        self._baseline_distance = float(distance)
        self._accum_delta = 0.0

    def reset(self) -> None:
        """Reset baseline and accumulator when exiting ZOOM mode."""
        self._baseline_distance = None
        self._accum_delta = 0.0
        self._last_step_time_ms = 0.0

    @staticmethod
    def compute_aperture(landmarks: np.ndarray, thumb_idx: int = 4, index_idx: int = 8) -> float:
        """Helper: compute Euclidean distance between thumb tip and index tip from (21, 3) landmarks."""
        if landmarks.shape[0] <= max(thumb_idx, index_idx):
            return 0.0
        pt_thumb = landmarks[thumb_idx, :2]
        pt_index = landmarks[index_idx, :2]
        return float(np.linalg.norm(pt_thumb - pt_index))

    def update(self, current_distance: float, timestamp_ms: float) -> int:
        """Update zoom distance and return integer zoom steps (+N for zoom in, -N for zoom out, 0 for none)."""
        current_distance = float(current_distance)
        timestamp_ms = float(timestamp_ms)

        # First frame: capture baseline
        if self._baseline_distance is None:
            self._baseline_distance = current_distance
            return 0

        # Delta relative to baseline
        delta = current_distance - self._baseline_distance

        # Dead-zone check
        if abs(delta) < self.dead_zone:
            return 0

        # Cooldown check
        if timestamp_ms - self._last_step_time_ms < self.cooldown_ms:
            return 0

        steps = 0
        if delta >= self.step_threshold:
            steps = int(delta / self.step_threshold)
            self._baseline_distance += steps * self.step_threshold
            self._last_step_time_ms = timestamp_ms
        elif delta <= -self.step_threshold:
            steps = -int(abs(delta) / self.step_threshold)
            self._baseline_distance += steps * self.step_threshold
            self._last_step_time_ms = timestamp_ms

        return steps
