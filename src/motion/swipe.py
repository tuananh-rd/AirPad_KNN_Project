"""Swipe engine for THREE_FINGERS gesture mode.

Contract:
- Monitors displacement and velocity during THREE_FINGERS pose.
- Emits BACK or FORWARD semantic event EXACTLY ONCE per swipe gesture (one-shot).
- Cooldown and re-arm logic prevents accidental multiple triggers until the hand
  either returns to neutral or exits the THREE_FINGERS pose.
- In Hybrid mode: Uses a recent-motion trajectory sliding window to calculate
  velocity and displacement from recent baseline, allowing users to hold the
  pose stationary before swiping without diluting velocity.
"""
from __future__ import annotations

import math
from typing import Optional, Literal, Union, List, Tuple


class SwipeEngine:
    """Detects directional swipes (BACK / FORWARD) from three-finger movements."""

    def __init__(
        self,
        min_displacement: float = 0.08,    # normalized screen displacement
        min_velocity: float = 0.0003,       # units per ms (0.3 units/sec)
        cooldown_ms: float = 400.0,
        rearm_threshold: float = 0.04,     # must return within this distance to rearm
        mode: str = "legacy",
        window_ms: float = 500.0,          # recent-motion trajectory window for hybrid mode
    ):
        self.min_displacement = float(min_displacement)
        self.min_velocity = float(min_velocity)
        self.cooldown_ms = float(cooldown_ms)
        self.rearm_threshold = float(rearm_threshold)
        self.mode = str(mode).lower()
        self.window_ms = float(window_ms)

        self._start_x: Optional[float] = None
        self._start_time_ms: float = 0.0
        self._history: List[Tuple[float, float]] = []  # (timestamp_ms, x) for hybrid mode
        self._last_event_time_ms: float = -1e9
        self._is_armed: bool = True
        self._fired: bool = False
        self._exit_observed_since_fire: bool = False

    @property
    def is_armed(self) -> bool:
        return self._is_armed

    def reset(self) -> None:
        """Reset internal swipe tracking."""
        self._start_x = None
        self._start_time_ms = 0.0
        self._history.clear()
        if self.mode == "hybrid":
            if not self._fired:
                self._is_armed = True
            else:
                self._exit_observed_since_fire = True
        else:
            self._is_armed = True
            self._fired = False

    def rearm(self) -> None:
        """Explicitly re-arm swipe detection."""
        self._is_armed = True
        self._fired = False
        self._start_x = None
        self._history.clear()
        self._exit_observed_since_fire = False

    def update(
        self,
        x: float,
        timestamp_ms: float,
        is_three_fingers: bool = True,
    ) -> Optional[Union[Literal["BACK", "FORWARD"], Literal["PREV", "NEXT"]]]:
        """Process hand X coordinate.

        Returns:
        - In legacy mode: "BACK" or "FORWARD"
        - In hybrid mode: "PREV" (preview left) or "NEXT" (preview right)
        - None if no swipe or not armed.
        """
        x = float(x)
        timestamp_ms = float(timestamp_ms)

        # If user left three_fingers pose
        if not is_three_fingers:
            self._history.clear()
            if self.mode == "hybrid":
                self._start_x = None
                if self._fired:
                    self._exit_observed_since_fire = True
                    if timestamp_ms - self._last_event_time_ms >= self.cooldown_ms:
                        self.rearm()
                else:
                    self._is_armed = True
                return None
            else:
                self.rearm()
                return None

        # If fired, check if cooldown passed and hand returned toward neutral to rearm
        if self._fired or not self._is_armed:
            if timestamp_ms - self._last_event_time_ms >= self.cooldown_ms:
                if self.mode == "hybrid":
                    if self._exit_observed_since_fire:
                        self.rearm()
                else:
                    if self._start_x is not None and abs(x - self._start_x) <= self.rearm_threshold:
                        self.rearm()
            # Legacy: unconditionally return None from this block (matches
            # saved-original behavior where rearm frame never falls through).
            # Hybrid: only return if still fired/unarmed, allowing fall-through
            # after successful rearm to seed _history with current sample.
            if self.mode != "hybrid":
                return None
            if self._fired or not self._is_armed:
                return None

        # Hybrid mode: Recent-motion trajectory sliding window
        if self.mode == "hybrid":
            self._history.append((timestamp_ms, x))
            cutoff = timestamp_ms - self.window_ms
            self._history = [(t, val) for t, val in self._history if t >= cutoff]

            for t_base, x_base in self._history:
                dt = timestamp_ms - t_base
                if dt <= 0.0:
                    continue
                dx = x - x_base
                if abs(dx) >= self.min_displacement:
                    velocity = abs(dx) / dt
                    if velocity >= self.min_velocity:
                        event: Union[Literal["BACK", "FORWARD"], Literal["PREV", "NEXT"]] = (
                            "NEXT" if dx > 0 else "PREV"
                        )
                        self._fired = True
                        self._is_armed = False
                        self._last_event_time_ms = timestamp_ms
                        self._exit_observed_since_fire = False
                        self._history.clear()
                        return event
            return None

        # Legacy mode: Single start_x and start_time_ms
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
            event = "FORWARD" if dx > 0 else "BACK"
            self._fired = True
            self._is_armed = False
            self._last_event_time_ms = timestamp_ms
            return event

        return None

    def update_tab(
        self,
        x: float,
        timestamp_ms: float,
        is_three_fingers: bool = True,
    ) -> Optional[Literal["PREV", "NEXT"]]:
        """Convenience method for tab switching (PREV / NEXT)."""
        evt = self.update(x, timestamp_ms, is_three_fingers=is_three_fingers)
        if evt in ("NEXT", "FORWARD"):
            return "NEXT"
        if evt in ("PREV", "BACK"):
            return "PREV"
        return None
