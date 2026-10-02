"""
Hand landmark representations and conversions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class LandmarkPoint:
    """A single 3D landmark point."""
    x: float
    y: float
    z: float


@dataclass
class HandLandmarks:
    """A collection of 21 landmark points representing a hand."""
    points: List[LandmarkPoint]
    handedness: str
    confidence: float

    def to_numpy(self) -> np.ndarray:
        """Convert this HandLandmarks instance to a numpy array of shape (21, 3)."""
        return to_numpy(self)


LANDMARK_NAMES = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP"
}


def to_numpy(hand: HandLandmarks) -> np.ndarray:
    """
    Convert HandLandmarks to a numpy array of shape (21, 3).
    """
    arr = np.zeros((21, 3), dtype=np.float32)
    for i, p in enumerate(hand.points):
        arr[i] = [p.x, p.y, p.z]
    return arr


def from_numpy(arr: np.ndarray, handedness: str, confidence: float) -> HandLandmarks:
    """
    Convert a numpy array of shape (21, 3) to HandLandmarks.
    """
    if arr.shape != (21, 3):
        raise ValueError(f"Expected shape (21, 3), got {arr.shape}")
        
    points = [LandmarkPoint(x=float(arr[i, 0]), y=float(arr[i, 1]), z=float(arr[i, 2])) for i in range(21)]
    return HandLandmarks(points=points, handedness=handedness, confidence=confidence)
