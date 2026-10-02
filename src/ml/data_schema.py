from dataclasses import dataclass
from typing import List

@dataclass 
class SampleMetadata:
    user_id: str
    session_id: str
    handedness: str  # 'Left' | 'Right'
    distance_cm: float
    lighting: str  # 'bright' | 'dim' | 'mixed'
    timestamp: float
    label: str
    is_transition: bool = False  # transition frame between gestures

LABEL_NAMES: List[str] = [
    'OPEN_PALM', 'POINT_ONE', 'PINCH_CLOSED', 'PINCH_MIDDLE', 'TWO_FINGERS_PARALLEL', 
    'L_SHAPE', 'THREE_FINGERS', 'FIST', 'UNKNOWN', 'OTHER'
]
