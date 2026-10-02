from collections import deque
from typing import Deque, Optional
from .pose import HandPose

class MajorityFilter:
    """Temporal majority voting over a sliding window.
    
    Only outputs a pose if it has >= majority_ratio of votes
    in the window, otherwise outputs previous stable pose.
    """
    def __init__(self, window_size: int = 5, majority_ratio: float = 0.6):
        self.window_size = window_size
        self.majority_ratio = majority_ratio
        self.window: Deque[HandPose] = deque(maxlen=window_size)
        self.current_stable_pose = HandPose.UNKNOWN

    def update(self, pose: HandPose) -> HandPose:
        self.window.append(pose)
        
        counts: dict[HandPose, int] = {}
        for p in self.window:
            counts[p] = counts.get(p, 0) + 1
            
        threshold = self.window_size * self.majority_ratio
        
        for p, count in counts.items():
            if count >= threshold:
                self.current_stable_pose = p
                break
                
        return self.current_stable_pose

    def reset(self) -> None:
        self.window.clear()
        self.current_stable_pose = HandPose.UNKNOWN

class StabilityGate:
    """Requires a pose to be stable for N consecutive frames."""
    def __init__(self, required_count: int = 3, required_stable_frames: Optional[int] = None):
        self.required_count = (
            required_stable_frames if required_stable_frames is not None else required_count
        )
        self.current_pose = HandPose.UNKNOWN
        self.count = 0
        self.stable_pose = HandPose.UNKNOWN

    def update(self, pose: HandPose) -> tuple[HandPose, bool]:
        # Returns (pose, is_newly_stable)
        is_newly_stable = False
        if pose == self.current_pose:
            self.count += 1
        else:
            self.current_pose = pose
            self.count = 1
            
        if self.count == self.required_count and self.current_pose != self.stable_pose:
            self.stable_pose = self.current_pose
            is_newly_stable = True
            
        if self.count >= self.required_count:
            return self.stable_pose, is_newly_stable
            
        return self.stable_pose, False

    def reset(self) -> None:
        self.current_pose = HandPose.UNKNOWN
        self.count = 0
        self.stable_pose = HandPose.UNKNOWN
