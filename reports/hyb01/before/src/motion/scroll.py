"""Scroll engine for TWO_FINGERS_PARALLEL gesture mode.

Features:
- Dominant-axis lock: prevents diagonal wobble by locking motion to X or Y.
- Dead-zone thresholding: ignores minor tremor while holding pose.
- Configurable gain and rate limiting to prevent event queue flooding.
- Reset capability upon exiting scroll mode.
- Sentinel initialization for first-event dispatch without false rate-limit suppression.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional


class ScrollEngine:
    """Computes discrete scroll events (dx, dy) from hand motion.

    In standard platforms (e.g. Windows/macOS):
    - Positive dy = scroll up (or down depending on natural scrolling convention).
    - Positive dx = scroll right.
    """

    def __init__(
        self,
        dead_zone: float = 0.005,
        gain: float = 120.0,
        axis_lock_ratio: float = 1.2,
        rate_limit_interval_ms: float = 30.0,
    ):
        self.dead_zone = float(dead_zone)
        self.gain = float(gain)
        self.axis_lock_ratio = float(axis_lock_ratio)
        self.rate_limit_interval_ms = float(rate_limit_interval_ms)

        self._locked_axis: Optional[str] = None  # 'X', 'Y', or None
        self._last_scroll_time_ms: Optional[float] = None
        self._accum_dx: float = 0.0
        self._accum_dy: float = 0.0

    @property
    def locked_axis(self) -> Optional[str]:
        return self._locked_axis

    def reset(self) -> None:
        """Reset dominant-axis lock and accumulator when entering/exiting scroll mode."""
        self._locked_axis = None
        self._accum_dx = 0.0
        self._accum_dy = 0.0
        self._last_scroll_time_ms = None

    def update(
        self,
        dx: float,
        dy: float,
        timestamp_ms: float,
    ) -> Tuple[int, int]:
        """Compute scroll steps (scroll_dx, scroll_dy) from displacement deltas.

        Returns (0, 0) if below dead-zone or rate-limited.
        The first scroll event after init/reset is not rate-limited.
        """
        dx = float(dx)
        dy = float(dy)
        timestamp_ms = float(timestamp_ms)

        # 1. Dead-zone check on input delta
        magnitude = math.hypot(dx, dy)
        if magnitude < self.dead_zone:
            return (0, 0)

        # 2. Dominant-axis lock
        if self._locked_axis is None:
            abs_x = abs(dx)
            abs_y = abs(dy)
            if abs_y >= abs_x * self.axis_lock_ratio:
                self._locked_axis = "Y"
            elif abs_x >= abs_y * self.axis_lock_ratio:
                self._locked_axis = "X"

        if self._locked_axis == "Y":
            dx = 0.0
        elif self._locked_axis == "X":
            dy = 0.0

        # Accumulate
        self._accum_dx += dx * self.gain
        self._accum_dy += dy * self.gain

        # 3. Rate limit check (first event is allowed via None sentinel)
        if self._last_scroll_time_ms is not None:
            if timestamp_ms - self._last_scroll_time_ms < self.rate_limit_interval_ms:
                return (0, 0)

        # Compute integer scroll units
        steps_x = int(self._accum_dx)
        steps_y = int(self._accum_dy)

        if steps_x != 0 or steps_y != 0:
            self._accum_dx -= steps_x
            self._accum_dy -= steps_y
            self._last_scroll_time_ms = timestamp_ms
            return (steps_x, steps_y)

        return (0, 0)
