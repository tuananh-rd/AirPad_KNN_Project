"""Smoothing and jitter reduction filters for cursor motion.

Includes:
- EMAFilter: Exponential Moving Average filter.
- OneEuroFilter: Speed-adaptive low-pass filter (Casiez et al., CHI 2012)
  for reducing jitter at low speeds while maintaining responsive lag-free
  tracking at high speeds.

Contract:
- OneEuroFilter operates with timestamps in SECONDS (float), matching frequency cutoffs in Hz.
  If caller uses milliseconds, convert via: `timestamp_s = timestamp_ms / 1000.0`.
- Non-monotonic timestamps (dt <= 0 or timestamp going backwards) are rejected safely
  without modifying internal timestamp baseline (_t_prev).
"""
from __future__ import annotations

import math
from typing import Tuple, Optional


class EMAFilter:
    """Exponential Moving Average filter for 2D coordinates.

    Formula: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
    """

    def __init__(self, alpha: float = 0.5):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._val: Optional[Tuple[float, float]] = None

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        """Apply EMA smoothing to point (x, y)."""
        if self._val is None:
            self._val = (float(x), float(y))
        else:
            prev_x, prev_y = self._val
            new_x = self.alpha * float(x) + (1.0 - self.alpha) * prev_x
            new_y = self.alpha * float(y) + (1.0 - self.alpha) * prev_y
            self._val = (new_x, new_y)
        return self._val

    def reset(self) -> None:
        """Clear filter state."""
        self._val = None

    @property
    def value(self) -> Optional[Tuple[float, float]]:
        return self._val


class _Scalar1EuroFilter:
    """1D 1€ filter helper operating in seconds and Hz."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: float) -> float:
        """Filter scalar x given monotonic timestamp in SECONDS.

        Contract:
        - If dt <= 1e-6 (duplicate timestamp or timestamp going backwards),
          returns previous value without updating _t_prev.
        """
        x = float(x)
        timestamp = float(timestamp)

        if self._t_prev is None or self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return x

        dt = timestamp - self._t_prev

        # Guard against zero/negative dt or non-monotonic timestamps
        # Contract requirement: timestamp đi lùi không được cập nhật mốc nội bộ!
        if dt <= 1e-6:
            return self._x_prev

        # Update baseline timestamp only on valid forward step
        self._t_prev = timestamp

        # Estimate derivative (speed) in units/second
        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev
        self._dx_prev = dx_hat

        # Compute dynamic cutoff frequency (Hz)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)

        # Filter signal
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev
        self._x_prev = x_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class OneEuroFilter:
    """2D Speed-adaptive 1€ filter.

    Parameters:
    - min_cutoff: Minimum cutoff frequency in Hz.
    - beta: Speed coefficient for frequency adaptation.
    - d_cutoff: Cutoff frequency for derivative filtering in Hz.

    Timestamps MUST be passed in SECONDS (e.g. 0.016 for 16ms delta).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._fx = _Scalar1EuroFilter(min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
        self._fy = _Scalar1EuroFilter(min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)

    def filter(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        """Filter a 2D point (x, y) given timestamp in SECONDS."""
        smooth_x = self._fx.filter(x, timestamp)
        smooth_y = self._fy.filter(y, timestamp)
        return (smooth_x, smooth_y)

    def reset(self) -> None:
        """Reset internal filter states."""
        self._fx.reset()
        self._fy.reset()
