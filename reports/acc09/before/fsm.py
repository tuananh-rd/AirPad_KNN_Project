from __future__ import annotations

from enum import IntEnum, auto
from typing import Any, Optional
import numpy as np
from src.gesture.pose import HandPose


class FSMState(IntEnum):
    PAUSED = 0                   # Startup state, no mouse control
    MOVING = auto()              # Cursor tracking active (POINT_ONE stable)
    CLICK_PENDING = auto()       # Thumb-index pinch detected, waiting to see if click or drag
    DRAGGING = auto()            # Drag in progress (mouseDown sent)
    SCROLLING = auto()           # Scroll mode (TWO_FINGERS_PARALLEL)
    ZOOM_LATCHED = auto()        # Zoom mode latched (L_SHAPE dwell)
    ZOOM_DWELL = auto()          # L_SHAPE detected, waiting for dwell time
    RIGHT_CLICK_PENDING = auto() # Thumb-middle pinch detected, waiting for release


class GestureFSM:
    """Finite State Machine for gesture-to-action mapping.

    Key invariants:
    - LEFT_CLICK fires exactly once on clean pinch release to POINT_ONE (if duration 80-300ms)
    - RIGHT_CLICK fires on clean thumb-middle pinch release to POINT_ONE (configurable, edge-triggered)
    - DRAG: mouseDown fires once on hold threshold, mouseUp on ANY exit from DRAGGING
    - Hand loss or error ALWAYS triggers cleanup (mouseUp, key release)
    - OPEN_PALM / NO_HAND -> PAUSED, releases everything and resets all pending states (even if already PAUSED)
    - UNKNOWN / OTHER never issue commands; reacquisition requires stable consecutive POINT_ONE frames
    - Edge-triggered: actions fire on transitions, not while held
    """

    def __init__(self, controller: Any, config: dict):
        self.controller = controller
        self.state = FSMState.PAUSED

        self._pinch_start_ms: Optional[float] = None
        self._right_pinch_start_ms: Optional[float] = None
        self._l_shape_start_ms: Optional[float] = None
        self._point_one_start_ms: Optional[float] = None
        self._hysteresis_count: int = 0
        self._last_pose = HandPose.UNKNOWN
        self._last_action_ms: float = 0.0
        self._is_mouse_down: bool = False

        # Configuration parameters
        self.drag_threshold_ms = float(config.get('drag_threshold_ms', 400.0))
        self.click_min_ms = float(config.get('click_min_ms', 80.0))
        self.click_max_ms = float(config.get('click_max_ms', 300.0))
        self.dwell_zoom_ms = float(config.get('dwell_zoom_ms', 250.0))
        self.hysteresis_exit_count = int(config.get('hysteresis_exit_count', 3))
        self.debounce_ms = float(config.get('debounce_ms', 50.0))
        self.point_one_stable_ms = float(config.get('point_one_stable_ms', 100.0))

        # Phase 2 additions
        self.enable_right_click = bool(config.get('enable_right_click', True))
        self.reacquire_stable_frames = int(config.get('reacquire_stable_frames', 2))
        self.unknown_timeout_frames = int(config.get('unknown_timeout_frames', 5))

        # Config validation
        if self.reacquire_stable_frames < 1:
            raise ValueError(f"reacquire_stable_frames must be >= 1, got {self.reacquire_stable_frames}")
        if not (self.click_min_ms <= self.click_max_ms < self.drag_threshold_ms):
            raise ValueError(
                f"Invalid gesture timing hierarchy: expected click_min_ms ({self.click_min_ms}) "
                f"<= click_max_ms ({self.click_max_ms}) < drag_threshold_ms ({self.drag_threshold_ms})"
            )

        self._reacquire_count: int = 0
        self._unknown_count: int = 0

    @property
    def is_mouse_down(self) -> bool:
        return self._is_mouse_down

    @property
    def is_reacquired(self) -> bool:
        return self._reacquire_count >= self.reacquire_stable_frames and self._unknown_count == 0

    def can_move_cursor(self, pose: HandPose) -> bool:
        """Check if cursor motion is currently valid and safe.

        Requires explicit HandPose enum (no pose=None bypass allowed).
        Returns True ONLY when:
        - In MOVING state with POINT_ONE pose and stable reacquisition.
        - In DRAGGING state with PINCH_CLOSED pose and confirmed button down.
        - Not in UNKNOWN or degraded tracking state.
        """
        if not isinstance(pose, HandPose):
            return False

        if self._is_mouse_down:
            return self.state == FSMState.DRAGGING and pose == HandPose.PINCH_CLOSED

        if self.state == FSMState.MOVING:
            if self._unknown_count > 0:
                return False
            if self._reacquire_count < self.reacquire_stable_frames:
                return False
            return pose == HandPose.POINT_ONE

        return False

    def is_scrolling_active(self, pose: HandPose) -> bool:
        """Check if scrolling output is active. UNKNOWN/OTHER immediately halts output."""
        if not isinstance(pose, HandPose):
            return False
        if pose in (HandPose.UNKNOWN, HandPose.OTHER):
            return False
        return self.state == FSMState.SCROLLING and pose == HandPose.TWO_FINGERS_PARALLEL

    def is_zoom_active(self, pose: HandPose) -> bool:
        """Check if zoom output is active. UNKNOWN/OTHER immediately halts output."""
        if not isinstance(pose, HandPose):
            return False
        if pose in (HandPose.UNKNOWN, HandPose.OTHER):
            return False
        return self.state == FSMState.ZOOM_LATCHED and pose == HandPose.L_SHAPE

    def _exit_dragging(self) -> None:
        """Ensure mouseUp is released when leaving DRAGGING.

        If mouse_up fails:
        - _is_mouse_down remains True
        - Error is propagated to prevent unconfirmed release state
        """
        if self._is_mouse_down or self.state == FSMState.DRAGGING:
            self.controller.mouse_up('left')
            self._is_mouse_down = False

    def update(
        self,
        pose: HandPose,
        timestamp_ms: float,
        landmarks: Optional[np.ndarray] = None,
    ) -> FSMState:
        """Process one frame. Returns new state.

        This method is the ONLY entry point for state transitions.
        All side effects (mouse/key events) happen here.
        """
        timestamp_ms = float(timestamp_ms)

        # Global override: NO_HAND or OPEN_PALM always resets pending and triggers cleanup,
        # even if already in PAUSED state.
        if pose in (HandPose.NO_HAND, HandPose.OPEN_PALM):
            self._force_cleanup()
            self.state = FSMState.PAUSED
            return self.state

        if self.state == FSMState.PAUSED:
            if pose == HandPose.POINT_ONE:
                if self._point_one_start_ms is None:
                    self._point_one_start_ms = timestamp_ms
                elif timestamp_ms - self._point_one_start_ms >= self.point_one_stable_ms:
                    self.state = FSMState.MOVING
                    self._reacquire_count = self.reacquire_stable_frames
                    self._unknown_count = 0
            else:
                self._point_one_start_ms = None

        elif self.state == FSMState.MOVING:
            self._point_one_start_ms = None

            if pose == HandPose.POINT_ONE:
                self._unknown_count = 0
                self._reacquire_count += 1

            elif pose == HandPose.FIST:
                # FIST resets the reacquisition sequence
                self._reacquire_count = 0

            elif pose in (HandPose.UNKNOWN, HandPose.OTHER):
                # Tracking degraded: reset reacquisition, freeze active gestures
                self._unknown_count += 1
                self._reacquire_count = 0
                if self._unknown_count >= self.unknown_timeout_frames:
                    self._force_cleanup()
                    self.state = FSMState.PAUSED

            elif pose == HandPose.PINCH_CLOSED:
                # Can only initiate click/drag if fully reacquired
                if self.is_reacquired:
                    self._pinch_start_ms = timestamp_ms
                    self.state = FSMState.CLICK_PENDING
                    self._unknown_count = 0
                else:
                    self._reacquire_count = 0

            elif pose == HandPose.PINCH_MIDDLE:
                # Can only initiate right-click if enabled and fully reacquired
                if self.enable_right_click and self.is_reacquired:
                    self._right_pinch_start_ms = timestamp_ms
                    self.state = FSMState.RIGHT_CLICK_PENDING
                    self._unknown_count = 0
                else:
                    self._reacquire_count = 0

            elif pose == HandPose.TWO_FINGERS_PARALLEL:
                if self.is_reacquired:
                    self.state = FSMState.SCROLLING
                    self._hysteresis_count = 0
                    self._unknown_count = 0
                else:
                    self._reacquire_count = 0

            elif pose == HandPose.L_SHAPE:
                if self.is_reacquired:
                    self._l_shape_start_ms = timestamp_ms
                    self.state = FSMState.ZOOM_DWELL
                    self._unknown_count = 0
                else:
                    self._reacquire_count = 0

            else:
                pass

        elif self.state == FSMState.CLICK_PENDING:
            if self._pinch_start_ms is None:
                self._pinch_start_ms = timestamp_ms
            elapsed = timestamp_ms - self._pinch_start_ms

            if pose == HandPose.PINCH_CLOSED:
                if elapsed >= self.drag_threshold_ms:
                    self.state = FSMState.DRAGGING
                    self.controller.mouse_down('left')
                    self._is_mouse_down = True
            elif pose == HandPose.POINT_ONE:
                # Only clean release to POINT_ONE can fire left click
                if self.click_min_ms <= elapsed <= self.click_max_ms:
                    if timestamp_ms - self._last_action_ms >= self.debounce_ms:
                        self.controller.mouse_click('left')
                        self._last_action_ms = timestamp_ms
                self.state = FSMState.MOVING
                self._pinch_start_ms = None
            else:
                # Any other pose (PINCH_MIDDLE, UNKNOWN, OTHER, FIST) cancels click
                self.state = FSMState.MOVING
                self._pinch_start_ms = None
                self._reacquire_count = 0

        elif self.state == FSMState.RIGHT_CLICK_PENDING:
            if self._right_pinch_start_ms is None:
                self._right_pinch_start_ms = timestamp_ms
            elapsed = timestamp_ms - self._right_pinch_start_ms

            if pose == HandPose.PINCH_MIDDLE:
                # Still holding thumb-middle pinch
                pass
            elif pose == HandPose.POINT_ONE:
                # Only clean release to POINT_ONE can fire right click
                if self.click_min_ms <= elapsed <= self.click_max_ms:
                    if timestamp_ms - self._last_action_ms >= self.debounce_ms:
                        self.controller.mouse_click('right')
                        self._last_action_ms = timestamp_ms
                self.state = FSMState.MOVING
                self._right_pinch_start_ms = None
            else:
                # Any other pose (PINCH_CLOSED, UNKNOWN, OTHER, FIST) cancels right click
                self.state = FSMState.MOVING
                self._right_pinch_start_ms = None
                self._reacquire_count = 0

        elif self.state == FSMState.DRAGGING:
            if pose != HandPose.PINCH_CLOSED:
                # Attempt to exit dragging. If controller raises, exception propagates
                # and state/is_mouse_down is NOT marked released.
                self._exit_dragging()
                if pose in (HandPose.UNKNOWN, HandPose.OTHER):
                    self.state = FSMState.MOVING
                    self._reacquire_count = 0
                    self._unknown_count = 1
                else:
                    self.state = FSMState.MOVING

        elif self.state == FSMState.SCROLLING:
            if pose != HandPose.TWO_FINGERS_PARALLEL:
                if pose in (HandPose.UNKNOWN, HandPose.OTHER):
                    self._unknown_count += 1

                if pose == self._last_pose:
                    self._hysteresis_count += 1
                else:
                    self._hysteresis_count = 1

                if self._hysteresis_count >= self.hysteresis_exit_count:
                    self.state = FSMState.MOVING
                    self._reacquire_count = 0
            else:
                self._hysteresis_count = 0
                self._unknown_count = 0

        elif self.state == FSMState.ZOOM_DWELL:
            if pose == HandPose.L_SHAPE:
                if self._l_shape_start_ms is None:
                    self._l_shape_start_ms = timestamp_ms
                elapsed = timestamp_ms - self._l_shape_start_ms
                if elapsed >= self.dwell_zoom_ms:
                    self.state = FSMState.ZOOM_LATCHED
            else:
                self.state = FSMState.MOVING
                self._l_shape_start_ms = None

        elif self.state == FSMState.ZOOM_LATCHED:
            if pose != HandPose.L_SHAPE:
                if pose in (HandPose.UNKNOWN, HandPose.OTHER):
                    self._unknown_count += 1

                if pose == self._last_pose:
                    self._hysteresis_count += 1
                else:
                    self._hysteresis_count = 1

                if self._hysteresis_count >= self.hysteresis_exit_count:
                    self.state = FSMState.MOVING
                    self._reacquire_count = 0
            else:
                self._hysteresis_count = 0
                self._unknown_count = 0

        self._last_pose = pose
        return self.state

    def on_hand_loss(self) -> None:
        """Called when hand tracking is lost. MUST cleanup."""
        self._force_cleanup()
        self.state = FSMState.PAUSED

    def on_error(self) -> None:
        """Called on any error. MUST cleanup."""
        self._force_cleanup()
        self.state = FSMState.PAUSED

    def on_emergency_stop(self) -> None:
        """Emergency stop hotkey pressed."""
        self._force_cleanup()
        self.state = FSMState.PAUSED

    def _force_cleanup(self) -> None:
        """Release all held buttons/keys via controller.cleanup() and reset state."""
        self._exit_dragging()
        self.controller.cleanup()
        self.state = FSMState.PAUSED
        self._pinch_start_ms = None
        self._right_pinch_start_ms = None
        self._l_shape_start_ms = None
        self._point_one_start_ms = None
        self._hysteresis_count = 0
        self._reacquire_count = 0
        self._unknown_count = 0
        self._last_pose = HandPose.UNKNOWN
