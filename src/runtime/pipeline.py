from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np

from src.gesture.pose import HandPose
from src.gesture.fsm import GestureFSM, FSMState
from src.gesture.temporal import MajorityFilter, StabilityGate
from src.platform.interface import PlatformController
from src.platform.dry_run import DryRunController
from src.ml.classifier import PoseClassifier
from src.ml.bundle import ModelBundle
from src.motion.cursor import CursorMapper
from src.motion.scroll import ScrollEngine
from src.motion.zoom import ZoomEngine
from src.motion.swipe import SwipeEngine

from src.features.canonicalize import canonicalize
from src.safety.failsafe import RateLimiter, FailsafeMonitor, EmergencyHotkey, SafetyGate


@dataclass
class PipelineResult:
    raw_pose: HandPose
    filtered_pose: HandPose
    fsm_state: FSMState
    cursor_pos: Optional[Tuple[int, int]]
    action_fired: Optional[str]
    telemetry: Dict[str, Any] = field(default_factory=dict)


class AirPadPipeline:
    def __init__(
        self,
        config: dict,
        controller: Optional[PlatformController] = None,
        model_bundle: Optional[ModelBundle] = None,
        classifier: Optional[PoseClassifier] = None,
        cursor_mapper: Optional[CursorMapper] = None,
        scroll_engine: Optional[ScrollEngine] = None,
        zoom_engine: Optional[ZoomEngine] = None,
        swipe_engine: Optional[SwipeEngine] = None,
        fsm: Optional[GestureFSM] = None,
        majority_filter: Optional[MajorityFilter] = None,
        stability_gate: Optional[StabilityGate] = None,
        rate_limiter: Optional[RateLimiter] = None,
        failsafe_monitor: Optional[FailsafeMonitor] = None,
        emergency_hotkey: Optional[EmergencyHotkey] = None,
        gate: Optional[SafetyGate] = None,
        profile: str = "legacy",
    ):
        self.config = config
        self.profile = str(profile).lower()
        if self.profile not in ("legacy", "hybrid"):
            raise ValueError(f"Unknown profile '{profile}'; must be 'legacy' or 'hybrid'")
        self.assist_state = "OFF" if self.profile == "hybrid" else "LEGACY"

        self.controller = controller or DryRunController()
        self.model_bundle = model_bundle
        self.classifier = classifier
        self.cursor_mapper = cursor_mapper
        self.scroll_engine = scroll_engine
        self.zoom_engine = zoom_engine
        self.swipe_engine = swipe_engine
        self.fsm = fsm
        self.majority_filter = majority_filter
        self.stability_gate = stability_gate
        self.rate_limiter = rate_limiter
        self.failsafe_monitor = failsafe_monitor
        self.emergency_hotkey = emergency_hotkey

        if self.profile == "hybrid":
            if self.scroll_engine is not None:
                self.scroll_engine.mode = "hybrid"
            if self.swipe_engine is not None:
                self.swipe_engine.mode = "hybrid"

        # Validate Gate consistency across controller, FSM, and explicit gate
        ctrl_gate = getattr(self.controller, 'gate', None)
        fsm_gate = getattr(self.fsm, 'gate', None) if self.fsm else None

        if gate is not None:
            if ctrl_gate is not None and ctrl_gate is not gate:
                raise ValueError("SafetyGate mismatch between explicit gate and controller.gate")
            if fsm_gate is not None and fsm_gate is not gate:
                raise ValueError("SafetyGate mismatch between explicit gate and fsm.gate")
            effective_gate = gate
        elif ctrl_gate is not None:
            if fsm_gate is not None and fsm_gate is not ctrl_gate:
                raise ValueError("SafetyGate mismatch between controller.gate and fsm.gate")
            effective_gate = ctrl_gate
        elif fsm_gate is not None:
            effective_gate = fsm_gate
        else:
            effective_gate = SafetyGate()

        self.gate: SafetyGate = effective_gate
        self.controller.gate = effective_gate
        if self.fsm is not None:
            self.fsm.gate = effective_gate
        self._emergency_stopped = False

        # Fail-closed admission contract for real OS control
        is_real = (
            getattr(self.controller, 'allow_real_control', False)
            or (self.controller.__class__.__name__ == 'WindowsController')
            or bool(self.config.get("safety", {}).get("allow_real_control", False))
        )
        if is_real:
            if self.emergency_hotkey is None:
                try:
                    self.controller.cleanup()
                except Exception:
                    pass
                raise RuntimeError("Real OS control admission refused: emergency hotkey is required")
            if self.emergency_hotkey.dry_run:
                try:
                    self.controller.cleanup()
                except Exception:
                    pass
                raise RuntimeError("Real OS control admission refused: emergency hotkey cannot be dry_run for real control")
            self.emergency_hotkey.callback = self.trigger_emergency_stop
            try:
                self.emergency_hotkey.register()
            except Exception as e:
                try:
                    self.emergency_hotkey.unregister()
                except Exception:
                    pass
                try:
                    self.controller.cleanup()
                except Exception:
                    pass
                raise RuntimeError(f"Real OS control admission refused: emergency hotkey registration failed ({e})") from e
            if not self.emergency_hotkey.is_registered:
                try:
                    self.emergency_hotkey.unregister()
                except Exception:
                    pass
                try:
                    self.controller.cleanup()
                except Exception:
                    pass
                raise RuntimeError("Real OS control admission refused: emergency hotkey not registered")
        else:
            if self.emergency_hotkey is not None:
                self.emergency_hotkey.callback = self.trigger_emergency_stop
                self.emergency_hotkey.register()

        # Constructor model bundle validation with rollback
        try:
            if self.model_bundle is not None:
                if self.model_bundle.feature_version != 1:
                    raise ValueError(f"Model feature version mismatch: expected 1, got {self.model_bundle.feature_version}")
        except Exception:
            try:
                self.cleanup()
            except Exception:
                pass
            raise

    @property
    def is_emergency_stopped(self) -> bool:
        if self._emergency_stopped:
            return True
        if self.gate and self.gate.is_stopped:
            return True
        if self.failsafe_monitor and self.failsafe_monitor.is_stopped():
            return True
        if self.fsm and getattr(self.fsm, '_emergency_stopped', False):
            return True
        return False

    def trigger_emergency_stop(self) -> None:
        self._emergency_stopped = True
        if self.profile == "hybrid":
            self.assist_state = "EMERGENCY_STOPPED"

        def _do_stop():
            cleanup_exc: Optional[Exception] = None
            if self.failsafe_monitor:
                try:
                    self.failsafe_monitor.emergency_stop()
                except Exception as e:
                    if cleanup_exc is None:
                        cleanup_exc = e
            if self.fsm:
                try:
                    self.fsm.on_emergency_stop()
                except Exception as e:
                    if cleanup_exc is None:
                        cleanup_exc = e
            try:
                self.controller.cleanup()
            except Exception as e:
                if cleanup_exc is None:
                    cleanup_exc = e
            if cleanup_exc is not None:
                raise cleanup_exc

        if self.gate is not None:
            self.gate.trigger_stop(_do_stop)
        else:
            _do_stop()

    def toggle_assist(self) -> str:
        """Toggle assist state between OFF and READY in hybrid profile under shared SafetyGate.

        Emergency stop is latched and cannot be cleared by SPACE.
        In legacy profile, returns 'LEGACY'.
        """
        def _do_toggle():
            if self.is_emergency_stopped:
                return "EMERGENCY_STOPPED"
            if self.profile != "hybrid":
                return "LEGACY"
            if self.assist_state == "OFF":
                self.assist_state = "READY"
            else:
                self.assist_state = "OFF"
                self._prev_scroll_pos = None
                if self.scroll_engine:
                    self.scroll_engine.reset()
                if self.swipe_engine:
                    self.swipe_engine.reset()
            return self.assist_state

        gate = getattr(self, "gate", None)
        gate_lock = getattr(gate, "_lock", None) if gate else None
        if gate_lock is not None:
            with gate_lock:
                return _do_toggle()
        return _do_toggle()

    def process_landmarks(self, landmarks: Optional[np.ndarray], handedness: str = "Right", timestamp_ms: float = 0.0) -> PipelineResult:
        if self.profile == "hybrid":
            return self._process_landmarks_hybrid(landmarks, handedness=handedness, timestamp_ms=timestamp_ms)

        if self.is_emergency_stopped:
            return PipelineResult(
                raw_pose=HandPose.NO_HAND,
                filtered_pose=HandPose.NO_HAND,
                fsm_state=FSMState.PAUSED,
                cursor_pos=None,
                action_fired="EMERGENCY_STOP",
                telemetry={"status": "emergency_stopped"}
            )

        if landmarks is None:
            if self.majority_filter:
                self.majority_filter.update(HandPose.NO_HAND)
            if self.stability_gate:
                self.stability_gate.update(HandPose.NO_HAND)
            if self.fsm:
                self.fsm.update(HandPose.NO_HAND, timestamp_ms, None)

            self._prev_scroll_pos = None
            if self.cursor_mapper:
                self.cursor_mapper.reset_baseline()
            if self.scroll_engine:
                self.scroll_engine.reset()
            if self.zoom_engine:
                self.zoom_engine.reset()
            if self.swipe_engine:
                self.swipe_engine.reset()

            return PipelineResult(
                raw_pose=HandPose.NO_HAND,
                filtered_pose=HandPose.NO_HAND,
                fsm_state=self.fsm.state if self.fsm else FSMState.PAUSED,
                cursor_pos=None,
                action_fired=None,
                telemetry={}
            )

        # 1. Canonicalize
        canon = canonicalize(landmarks, handedness=handedness)
        canon_flat = canon.flatten().reshape(1, -1) if canon.ndim == 2 else canon.reshape(1, -1)

        # 2. Classify
        raw_pose = HandPose.UNKNOWN
        if self.classifier:
            ml_cfg = self.config.get("ml", {})
            decision_thresh = ml_cfg.get("decision_threshold", 0.0)
            margin_thresh = ml_cfg.get("margin_threshold", 0.15)
            preds = self.classifier.predict_with_rejection(
                canon_flat,
                decision_threshold=decision_thresh,
                margin_threshold=margin_thresh,
                unknown_label="UNKNOWN",
            )
            pred_val = preds[0]
            if isinstance(pred_val, str):
                try:
                    raw_pose = HandPose[pred_val]
                except KeyError:
                    raw_pose = HandPose.UNKNOWN
            else:
                try:
                    raw_pose = HandPose(int(pred_val))
                except ValueError:
                    raw_pose = HandPose.UNKNOWN

        # 3. Temporal Filter
        filtered_pose = raw_pose
        if self.majority_filter:
            filtered_pose = self.majority_filter.update(filtered_pose)
        if self.stability_gate:
            filtered_pose, _ = self.stability_gate.update(filtered_pose)

        # 4. FSM Update
        new_state = FSMState.PAUSED
        if self.fsm:
            new_state = self.fsm.update(filtered_pose, timestamp_ms, landmarks)

        action_fired = None
        cursor_pos = None

        # 5. Motion and gesture action dispatch under shared SafetyGate transaction
        action_fired = None
        cursor_pos = None

        def _execute_motion():
            nonlocal action_fired, cursor_pos
            if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                return

            if self.fsm and self.fsm.can_move_cursor(filtered_pose) and self.cursor_mapper:
                self._prev_scroll_pos = None
                # Use index tip (8)
                index_tip = landmarks[8]
                x, y = index_tip[0], index_tip[1]
                mapped_pos = self.cursor_mapper.map_point(x, y, timestamp=timestamp_ms / 1000.0)
                if mapped_pos:
                    if self.failsafe_monitor and self.failsafe_monitor.check_position(mapped_pos[0], mapped_pos[1]):
                        self.trigger_emergency_stop()
                        return
                    if not self.rate_limiter or self.rate_limiter.allow():
                        if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                            return
                        self.controller.mouse_move(mapped_pos[0], mapped_pos[1])
                        if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                            return
                        cursor_pos = mapped_pos
                        action_fired = "MOVE"

            elif self.fsm and self.fsm.is_scrolling_active(filtered_pose) and self.scroll_engine:
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                cx = float(index_tip[0] + middle_tip[0]) / 2.0
                cy = float(index_tip[1] + middle_tip[1]) / 2.0
                if getattr(self, '_prev_scroll_pos', None) is not None:
                    dx = cx - self._prev_scroll_pos[0]
                    dy = cy - self._prev_scroll_pos[1]
                    steps = self.scroll_engine.update(dx, dy, timestamp_ms)
                    if steps != (0, 0):
                        if not self.rate_limiter or self.rate_limiter.allow():
                            if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                                return
                            self.controller.scroll(steps[0], steps[1])
                            if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                                return
                            action_fired = "SCROLL"
                self._prev_scroll_pos = (cx, cy)

            elif self.fsm and self.fsm.is_zoom_active(filtered_pose) and self.zoom_engine:
                aperture = self.zoom_engine.compute_aperture(landmarks)
                steps = self.zoom_engine.update(aperture, timestamp_ms)
                if steps != 0:
                    if not self.rate_limiter or self.rate_limiter.allow():
                        if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                            return
                        if hasattr(self.controller, 'zoom'):
                            self.controller.zoom(steps)
                        else:
                            self.controller.scroll(0, steps)
                        if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                            return
                        action_fired = "ZOOM"

            elif self.fsm and self.fsm.state == FSMState.MOVING and filtered_pose == HandPose.THREE_FINGERS and self.swipe_engine:
                # Wrist or palm x (using 0 for wrist)
                wrist_x = landmarks[0][0]
                event = self.swipe_engine.update(wrist_x, timestamp_ms)
                if event in ("BACK", "FORWARD"):
                    if not self.rate_limiter or self.rate_limiter.allow():
                        if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                            return
                        if hasattr(self.controller, 'swipe'):
                            self.controller.swipe(event)
                        if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
                            return
                        action_fired = f"SWIPE_{event}"

        if self.gate is not None and hasattr(self.gate, "transaction"):
            with self.gate.transaction() as active:
                if active and not self.is_emergency_stopped and not (self.fsm and self.fsm.is_stopped):
                    _execute_motion()
        else:
            _execute_motion()

        if self.is_emergency_stopped or (self.fsm and self.fsm.is_stopped):
            return PipelineResult(
                raw_pose=raw_pose,
                filtered_pose=filtered_pose,
                fsm_state=FSMState.PAUSED,
                cursor_pos=None,
                action_fired="EMERGENCY_STOP",
                telemetry={"status": "emergency_stopped"}
            )
        
        telemetry = {
            "timestamp_ms": timestamp_ms,
            "raw_pose": raw_pose.name,
            "filtered_pose": filtered_pose.name,
            "fsm_state": new_state.name,
            "action": action_fired
        }

        return PipelineResult(
            raw_pose=raw_pose,
            filtered_pose=filtered_pose,
            fsm_state=new_state,
            cursor_pos=cursor_pos,
            action_fired=action_fired,
            telemetry=telemetry
        )

    def _process_landmarks_hybrid(
        self,
        landmarks: Optional[np.ndarray],
        handedness: str = "Right",
        timestamp_ms: float = 0.0,
    ) -> PipelineResult:
        if self.is_emergency_stopped:
            return PipelineResult(
                raw_pose=HandPose.NO_HAND,
                filtered_pose=HandPose.NO_HAND,
                fsm_state=FSMState.PAUSED,
                cursor_pos=None,
                action_fired="EMERGENCY_STOP",
                telemetry={"status": "emergency_stopped", "assist_state": "EMERGENCY_STOPPED", "profile": "hybrid"}
            )

        gate_lock = getattr(self.gate, "_lock", None) if getattr(self, "gate", None) else None
        lock_ctx = gate_lock if gate_lock is not None else contextlib.nullcontext()

        if self.is_emergency_stopped:
            return PipelineResult(
                raw_pose=HandPose.NO_HAND,
                filtered_pose=HandPose.NO_HAND,
                fsm_state=FSMState.PAUSED,
                cursor_pos=None,
                action_fired="EMERGENCY_STOP",
                telemetry={"status": "emergency_stopped", "assist_state": "EMERGENCY_STOPPED", "profile": "hybrid"}
            )

        if landmarks is None:
            with lock_ctx:
                if self.majority_filter:
                    self.majority_filter.update(HandPose.NO_HAND)
                if self.stability_gate:
                    self.stability_gate.update(HandPose.NO_HAND)

                self._prev_scroll_pos = None
                if self.scroll_engine:
                    self.scroll_engine.reset()
                if self.swipe_engine:
                    self.swipe_engine.reset()

                # Serialize assist_state check and update:
                # Invariant: If assist_state is OFF, NEVER silently re-enable to READY!
                # Only explicit toggle_assist() may transition OFF -> READY.
                if self.is_emergency_stopped:
                    self.assist_state = "EMERGENCY_STOPPED"
                elif self.assist_state != "OFF":
                    self.assist_state = "READY"

                current_assist = self.assist_state

            return PipelineResult(
                raw_pose=HandPose.NO_HAND,
                filtered_pose=HandPose.NO_HAND,
                fsm_state=FSMState.PAUSED,
                cursor_pos=None,
                action_fired=None,
                telemetry={"assist_state": current_assist, "profile": "hybrid"}
            )

        # 1. Canonicalize
        canon = canonicalize(landmarks, handedness=handedness)
        canon_flat = canon.flatten().reshape(1, -1) if canon.ndim == 2 else canon.reshape(1, -1)

        # 2. Classify
        raw_pose = HandPose.UNKNOWN
        if self.classifier:
            ml_cfg = self.config.get("ml", {})
            decision_thresh = ml_cfg.get("decision_threshold", 0.0)
            margin_thresh = ml_cfg.get("margin_threshold", 0.15)
            preds = self.classifier.predict_with_rejection(
                canon_flat,
                decision_threshold=decision_thresh,
                margin_threshold=margin_thresh,
                unknown_label="UNKNOWN",
            )
            pred_val = preds[0]
            if isinstance(pred_val, str):
                try:
                    raw_pose = HandPose[pred_val]
                except KeyError:
                    raw_pose = HandPose.UNKNOWN
            else:
                try:
                    raw_pose = HandPose(int(pred_val))
                except ValueError:
                    raw_pose = HandPose.UNKNOWN

        # 3. Temporal filter
        filtered_pose = raw_pose
        if self.majority_filter:
            filtered_pose = self.majority_filter.update(filtered_pose)
        if self.stability_gate:
            filtered_pose, _ = self.stability_gate.update(filtered_pose)

        # 4. Hybrid Assist active (whitelisted: TWO_FINGERS_PARALLEL -> scroll, THREE_FINGERS -> tab switch)
        is_scroll_eligible = (
            self.classifier is not None
            and filtered_pose == HandPose.TWO_FINGERS_PARALLEL
            and raw_pose == HandPose.TWO_FINGERS_PARALLEL
        )
        is_tab_eligible = (
            self.classifier is not None
            and filtered_pose == HandPose.THREE_FINGERS
            and raw_pose == HandPose.THREE_FINGERS
        )

        action_fired = None

        with lock_ctx:
            if self.is_emergency_stopped:
                return PipelineResult(
                    raw_pose=raw_pose,
                    filtered_pose=filtered_pose,
                    fsm_state=FSMState.PAUSED,
                    cursor_pos=None,
                    action_fired="EMERGENCY_STOP",
                    telemetry={"status": "emergency_stopped", "assist_state": "EMERGENCY_STOPPED", "profile": "hybrid"}
                )

            # If assist is OFF: zero action dispatch, reset engines under lock, return HUD telemetry
            if self.assist_state == "OFF":
                self._prev_scroll_pos = None
                if self.scroll_engine:
                    self.scroll_engine.reset()
                if self.swipe_engine:
                    self.swipe_engine.reset()
                return PipelineResult(
                    raw_pose=raw_pose,
                    filtered_pose=filtered_pose,
                    fsm_state=FSMState.PAUSED,
                    cursor_pos=None,
                    action_fired=None,
                    telemetry={"assist_state": "OFF", "profile": "hybrid"}
                )

            # Mismatch reset under lock: if raw pose no longer matches filtered eligible mode, reset baseline
            if filtered_pose == HandPose.TWO_FINGERS_PARALLEL and not is_scroll_eligible:
                self._prev_scroll_pos = None
                if self.scroll_engine:
                    self.scroll_engine.reset()
            if filtered_pose == HandPose.THREE_FINGERS and not is_tab_eligible:
                if self.swipe_engine:
                    self.swipe_engine.reset()

            def _execute_hybrid_action():
                nonlocal action_fired
                if self.is_emergency_stopped or self.assist_state == "OFF":
                    return

                if is_scroll_eligible:
                    if self.assist_state != "OFF" and not self.is_emergency_stopped:
                        self.assist_state = "SCROLL"
                    if self.swipe_engine:
                        self.swipe_engine.reset()

                    if self.scroll_engine and landmarks is not None and len(landmarks) > 12:
                        index_tip = landmarks[8]
                        middle_tip = landmarks[12]
                        cy = float(index_tip[1] + middle_tip[1]) / 2.0
                        cx = float(index_tip[0] + middle_tip[0]) / 2.0

                        if getattr(self, '_prev_scroll_pos', None) is not None:
                            # Mirrored preview coordinate contract:
                            # Hand UP in preview -> cy decreases -> dy_cam < 0.
                            # Target: hand UP -> scroll UP (positive scroll units).
                            # dy_scroll = prev_cy - cy
                            dy_scroll = self._prev_scroll_pos[1] - cy
                            steps = self.scroll_engine.update(0.0, dy_scroll, timestamp_ms)
                            steps_y = steps[1] if isinstance(steps, tuple) else steps
                            if steps_y != 0:
                                if not self.rate_limiter or self.rate_limiter.allow():
                                    # Re-check assist_state and emergency stop after rate_limiter callback
                                    if self.is_emergency_stopped or self.assist_state != "SCROLL":
                                        return
                                    self.controller.scroll(0, steps_y)
                                    if self.is_emergency_stopped or self.assist_state != "SCROLL":
                                        return
                                    action_fired = "SCROLL"
                        self._prev_scroll_pos = (cx, cy)

                elif is_tab_eligible:
                    if self.assist_state != "OFF" and not self.is_emergency_stopped:
                        self.assist_state = "TAB"
                    self._prev_scroll_pos = None
                    if self.scroll_engine:
                        self.scroll_engine.reset()

                    if self.swipe_engine and landmarks is not None and len(landmarks) > 0:
                        wrist_x = landmarks[0][0]
                        event = self.swipe_engine.update(wrist_x, timestamp_ms, is_three_fingers=True)
                        if event in ("FORWARD", "NEXT"):
                            if not self.rate_limiter or self.rate_limiter.allow():
                                # Re-check assist_state and emergency stop after rate_limiter callback
                                if self.is_emergency_stopped or self.assist_state != "TAB":
                                    return
                                if hasattr(self.controller, 'switch_tab'):
                                    self.controller.switch_tab("NEXT")
                                if self.is_emergency_stopped or self.assist_state != "TAB":
                                    return
                                action_fired = "TAB_NEXT"
                        elif event in ("BACK", "PREV"):
                            if not self.rate_limiter or self.rate_limiter.allow():
                                # Re-check assist_state and emergency stop after rate_limiter callback
                                if self.is_emergency_stopped or self.assist_state != "TAB":
                                    return
                                if hasattr(self.controller, 'switch_tab'):
                                    self.controller.switch_tab("PREV")
                                if self.is_emergency_stopped or self.assist_state != "TAB":
                                    return
                                action_fired = "TAB_PREV"

                else:
                    # All other poses: physical mouse retains exclusive ownership of pointer and clicking/dragging.
                    # Zero actions emitted.
                    if self.assist_state != "OFF" and not self.is_emergency_stopped:
                        self.assist_state = "READY"
                    self._prev_scroll_pos = None
                    if self.scroll_engine:
                        self.scroll_engine.reset()
                    if self.swipe_engine and landmarks is not None and len(landmarks) > 0:
                        self.swipe_engine.update(landmarks[0][0], timestamp_ms, is_three_fingers=False)

            if self.gate is not None and hasattr(self.gate, "transaction"):
                with self.gate.transaction() as active:
                    if active and not self.is_emergency_stopped and self.assist_state != "OFF":
                        _execute_hybrid_action()
            else:
                if not self.is_emergency_stopped and self.assist_state != "OFF":
                    _execute_hybrid_action()

            if self.assist_state == "OFF":
                action_fired = None

            current_assist = self.assist_state

        if self.is_emergency_stopped:
            return PipelineResult(
                raw_pose=raw_pose,
                filtered_pose=filtered_pose,
                fsm_state=FSMState.PAUSED,
                cursor_pos=None,
                action_fired="EMERGENCY_STOP",
                telemetry={"status": "emergency_stopped", "assist_state": "EMERGENCY_STOPPED", "profile": "hybrid"}
            )

        telemetry = {
            "timestamp_ms": timestamp_ms,
            "raw_pose": raw_pose.name,
            "filtered_pose": filtered_pose.name,
            "fsm_state": FSMState.PAUSED.name,
            "action": action_fired,
            "assist_state": current_assist,
            "profile": "hybrid",
        }

        return PipelineResult(
            raw_pose=raw_pose,
            filtered_pose=filtered_pose,
            fsm_state=FSMState.PAUSED,
            cursor_pos=None,
            action_fired=action_fired,
            telemetry=telemetry,
        )

    def cleanup(self) -> None:
        cleanup_exc: Optional[Exception] = None
        if self.emergency_hotkey:
            try:
                self.emergency_hotkey.unregister()
            except Exception as e:
                cleanup_exc = e
        if self.fsm and self.profile != "hybrid":
            try:
                self.fsm.on_hand_loss()
            except Exception as e:
                if cleanup_exc is None:
                    cleanup_exc = e
        try:
            self.controller.cleanup()
        except Exception as e:
            if cleanup_exc is None:
                cleanup_exc = e
        if cleanup_exc is not None:
            raise cleanup_exc
