"""Swipe engine for THREE_FINGERS gesture mode.

Contract:
- Monitors displacement and velocity during THREE_FINGERS pose.
- Emits BACK or FORWARD semantic event EXACTLY ONCE per swipe gesture (one-shot).
- Cooldown and re-arm logic prevents accidental multiple triggers until the hand
  either returns to neutral or exits the THREE_FINGERS pose.
"""
from __future__ import annotations

import math
from typing import Optional, Literal


class SwipeEngine:
    """Detects directional swipes (BACK / FORWARD) from three-finger movements."""

    def __init__(
        self,
        min_displacement: float = 0.08,    # normalized screen displacement
        min_velocity: float = 0.0003,       # units per ms (0.3 units/sec)
        cooldown_ms: float = 400.0,
        rearm_threshold: float = 0.04,     # must return within this distance to rearm
    ):
        self.min_displacement = float(min_displacement)
        self.min_velocity = float(min_velocity)
        self.cooldown_ms = float(cooldown_ms)
        self.rearm_threshold = float(rearm_threshold)

        self._start_x: Optional[float] = None
        self._start_time_ms: float = 0.0
        self._last_event_time_ms: float = 0.0
        self._is_armed: bool = True
        self._fired: bool = False

    @property
    def is_armed(self) -> bool:
        return self._is_armed

    def reset(self) -> None:
        """Reset internal swipe tracking."""
        self._start_x = None
        self._start_time_ms = 0.0
        self._is_armed = True
        self._fired = False

    def rearm(self) -> None:
        """Explicitly re-arm swipe detection."""
        self._is_armed = True
        self._fired = False
        self._start_x = None

    def update(
        self,
        x: float,
        timestamp_ms: float,
        is_three_fingers: bool = True,
    ) -> Optional[Literal["BACK", "FORWARD"]]:
        """Process hand X coordinate.

        Returns:
        - "BACK" if swiped left (or backward depending on convention).
        - "FORWARD" if swiped right.
        - None if no swipe or not armed.
        """
        x = float(x)
        timestamp_ms = float(timestamp_ms)

        # If user left three_fingers pose, automatically re-arm for next time
        if not is_three_fingers:
            self.rearm()
            return None

        # If fired, check if cooldown passed and hand returned toward neutral to rearm
        if self._fired or not self._is_armed:
            if timestamp_ms - self._last_event_time_ms >= self.cooldown_ms:
                if self._start_x is not None and abs(x - self._start_x) <= self.rearm_threshold:
                    self.rearm()
            return None

        # Start tracking trajectory
        if self._start_x is None:
            self._start_x = x
            self._start_time_ms = timestamp_ms
            return None

        dx = x - self._start_x
        dt = timestamp_ms - self._start_time_ms

        if dt <= 0.0:
            return None

        velocity = abs(dx) / dt

        # Check threshold conditions
        if abs(dx) >= self.min_displacement and velocity >= self.min_velocity:
            # Direction: positive dx = right/forward, negative dx = left/back
            event: Literal["BACK", "FORWARD"] = "FORWARD" if dx > 0 else "BACK"
            self._fired = True
            self._is_armed = False
            self._last_event_time_ms = timestamp_ms
            return event

        return None
