"""Motion processing components for AirPad.

Includes:
- EMAFilter, OneEuroFilter: Jitter reduction & smoothing filters
- CursorMapper: ROI mapping, camera mirror, clamp, dead-zone, sensitivity
- ScrollEngine: Dominant-axis lock, dead-zone, gain/rate limit
- ZoomEngine: Distance delta tracking in ZOOM_LATCHED, semantic zoom steps
- SwipeEngine: Three-finger displacement/velocity tracking with cooldown & re-arm
"""
from __future__ import annotations

from src.motion.filters import EMAFilter, OneEuroFilter
from src.motion.cursor import CursorMapper
from src.motion.scroll import ScrollEngine
from src.motion.zoom import ZoomEngine
from src.motion.swipe import SwipeEngine

__all__ = [
    "EMAFilter",
    "OneEuroFilter",
    "CursorMapper",
    "ScrollEngine",
    "ZoomEngine",
    "SwipeEngine",
]
