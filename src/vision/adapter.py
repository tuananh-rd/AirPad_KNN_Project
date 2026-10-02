"""
MediaPipe adapter and interfaces.
"""
from __future__ import annotations

import os
import typing
from typing import List, Optional, Protocol
import numpy as np

from src.vision.landmarks import HandLandmarks, LandmarkPoint


class HandDetector(Protocol):
    """Protocol for hand detectors."""
    
    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """Detect hands in an image frame."""
        ...


class MediaPipeAdapter(HandDetector):
    """MediaPipe hand detection adapter supporting both modern Tasks API and legacy solutions API."""
    
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        static_image_mode: bool = False,
        task_model_path: Optional[str] = None,
    ):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("MediaPipe is not installed. Install with 'pip install mediapipe'")
            
        self.mp = mp
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = static_image_mode
        self._backend = "unknown"

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
            )
            self._backend = "solutions"
        else:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if task_model_path is None:
                default_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "models",
                    "hand_landmarker.task",
                )
                if not os.path.exists(default_path):
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    os.makedirs(os.path.dirname(default_path), exist_ok=True)
                    import urllib.request
                    urllib.request.urlretrieve(url, default_path)
                task_model_path = default_path

            base_options = python.BaseOptions(model_asset_path=task_model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self._backend = "tasks"

    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """
        Process a frame and return HandLandmarks.
        Note: frame should be RGB.
        """
        if self._backend == "solutions":
            results = self.hands.process(frame)
            hands_list = []
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    points = [LandmarkPoint(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks.landmark]
                    label = handedness.classification[0].label
                    score = handedness.classification[0].score
                    hands_list.append(HandLandmarks(points=points, handedness=label, confidence=score))
            return hands_list
        elif self._backend == "tasks":
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame)
            results = self.landmarker.detect(mp_image)
            hands_list = []
            if results.hand_landmarks and results.handedness:
                for hand_lms, hand_cats in zip(results.hand_landmarks, results.handedness):
                    points = [LandmarkPoint(x=lm.x, y=lm.y, z=lm.z) for lm in hand_lms]
                    label = hand_cats[0].category_name if hand_cats else "Right"
                    score = float(hand_cats[0].score) if hand_cats else 0.99
                    hands_list.append(HandLandmarks(points=points, handedness=label, confidence=score))
            return hands_list
        return []

    def close(self) -> None:
        """Release underlying MediaPipe detector resources."""
        if self._backend == "solutions" and hasattr(self, "hands"):
            self.hands.close()
        elif self._backend == "tasks" and hasattr(self, "landmarker"):
            self.landmarker.close()


class DummyDetector(HandDetector):
    """Dummy detector for testing."""
    
    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """Returns dummy landmarks."""
        points = [LandmarkPoint(x=0.1, y=0.1, z=0.1) for _ in range(21)]
        return [HandLandmarks(points=points, handedness="Right", confidence=0.99)]
