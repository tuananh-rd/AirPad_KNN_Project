"""Cursor mapper: maps normalized hand landmark coordinates to screen coordinates.

Features:
- Active ROI (Region of Interest) mapping with configurable bounds.
- Camera horizontal mirroring (webcam mirror).
- Dead-zone thresholding to eliminate micro-jitter while resting.
- Screen boundary clamping.
- Sensitivity / gain scaling.
- Configurable smoothing filter (One Euro Filter, EMA, or None).
- Baseline reset on mode changes to prevent cursor jumps.
- Monotonic synthetic clock when timestamp=None to prevent stall.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional
from src.motion.filters import EMAFilter, OneEuroFilter


class CursorMapper:
    """Maps normalized hand coordinates (typically index fingertip in [0, 1])
    to absolute OS screen coordinates (x_screen, y_screen).
    """

    def __init__(
        self,
        screen_size: Tuple[int, int] = (1920, 1080),
        roi_x_min: float = 0.1,
        roi_x_max: float = 0.9,
        roi_y_min: float = 0.1,
        roi_y_max: float = 0.9,
        mirror_x: bool = True,
        dead_zone: float = 0.002,
        gain_x: float = 1.0,
        gain_y: float = 1.0,
        filter_type: str = "one_euro",
        ema_alpha: float = 0.5,
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.007,
        one_euro_d_cutoff: float = 1.0,
    ):
        if roi_x_min >= roi_x_max:
            raise ValueError(f"roi_x_min ({roi_x_min}) must be < roi_x_max ({roi_x_max})")
        if roi_y_min >= roi_y_max:
            raise ValueError(f"roi_y_min ({roi_y_min}) must be < roi_y_max ({roi_y_max})")

        self.screen_width = int(screen_size[0])
        self.screen_height = int(screen_size[1])

        self.roi_x_min = float(roi_x_min)
        self.roi_x_max = float(roi_x_max)
        self.roi_y_min = float(roi_y_min)
        self.roi_y_max = float(roi_y_max)

        self.mirror_x = bool(mirror_x)
        self.dead_zone = float(dead_zone)
        self.gain_x = float(gain_x)
        self.gain_y = float(gain_y)

        self.filter_type = filter_type.lower()
        if self.filter_type == "ema":
            self._filter = EMAFilter(alpha=ema_alpha)
        elif self.filter_type == "one_euro":
            self._filter = OneEuroFilter(
                min_cutoff=one_euro_min_cutoff,
                beta=one_euro_beta,
                d_cutoff=one_euro_d_cutoff,
            )
        else:
            self._filter = None

        self._prev_roi_pos: Optional[Tuple[float, float]] = None
        self._last_screen_pos: Optional[Tuple[int, int]] = None
        self._internal_time_s: float = 0.0
        self._default_dt_s: float = 1.0 / 60.0

    def set_screen_size(self, width: int, height: int) -> None:
        """Update screen dimensions."""
        self.screen_width = int(width)
        self.screen_height = int(height)

    def reset_baseline(self) -> None:
        """Reset internal filter states and previous positions to prevent cursor jumps."""
        if self._filter is not None:
            self._filter.reset()
        self._prev_roi_pos = None
        self._last_screen_pos = None
        self._internal_time_s = 0.0

    def map_point(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Transform raw normalized camera coordinate (x, y) in [0, 1] to screen (sx, sy).

        Parameters:
        - x, y: Normalized camera landmark coordinates in [0, 1].
        - timestamp: Monotonic timestamp in SECONDS. If None, advances internal
          virtual clock at 60 Hz (1/60s) to prevent zero-dt lockup.

        Steps:
        1. Camera mirror (if mirror_x=True, x = 1.0 - x).
        2. Normalize relative to active ROI.
        3. Clamp normalized coordinates to [0.0, 1.0].
        4. Apply dead-zone filter against previous position.
        5. Apply smoothing filter (EMA / 1-Euro).
        6. Apply sensitivity/gain and map to screen coordinates.
        7. Clamp to [0, screen_width - 1] and [0, screen_height - 1].
        """
        # Resolve timestamp in seconds
        if timestamp is not None:
            t = float(timestamp)
        else:
            self._internal_time_s += self._default_dt_s
            t = self._internal_time_s

        # 1. Camera horizontal mirroring
        cur_x = (1.0 - float(x)) if self.mirror_x else float(x)
        cur_y = float(y)

        # 2. Map into ROI [0, 1]
        norm_u = (cur_x - self.roi_x_min) / (self.roi_x_max - self.roi_x_min)
        norm_v = (cur_y - self.roi_y_min) / (self.roi_y_max - self.roi_y_min)

        # 3. Clamp to [0.0, 1.0]
        clamped_u = max(0.0, min(1.0, norm_u))
        clamped_v = max(0.0, min(1.0, norm_v))

        # 4. Dead-zone check
        if self._prev_roi_pos is not None:
            prev_u, prev_v = self._prev_roi_pos
            dist = math.hypot(clamped_u - prev_u, clamped_v - prev_v)
            if dist < self.dead_zone:
                clamped_u, clamped_v = prev_u, prev_v
            else:
                self._prev_roi_pos = (clamped_u, clamped_v)
        else:
            self._prev_roi_pos = (clamped_u, clamped_v)

        # 5. Smoothing filter
        if self._filter is not None:
            if isinstance(self._filter, OneEuroFilter):
                smooth_u, smooth_v = self._filter.filter(clamped_u, clamped_v, t)
            elif isinstance(self._filter, EMAFilter):
                smooth_u, smooth_v = self._filter.filter(clamped_u, clamped_v)
            else:
                smooth_u, smooth_v = clamped_u, clamped_v
        else:
            smooth_u, smooth_v = clamped_u, clamped_v

        # 6. Gain & screen projection
        u_scaled = 0.5 + (smooth_u - 0.5) * self.gain_x
        v_scaled = 0.5 + (smooth_v - 0.5) * self.gain_y

        u_screen = max(0.0, min(1.0, u_scaled))
        v_screen = max(0.0, min(1.0, v_scaled))

        screen_x = int(round(u_screen * (self.screen_width - 1)))
        screen_y = int(round(v_screen * (self.screen_height - 1)))

        # 7. Clamp to screen bounds
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        self._last_screen_pos = (screen_x, screen_y)
        return (screen_x, screen_y)
