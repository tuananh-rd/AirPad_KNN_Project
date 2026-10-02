"""
Dry run platform controller for testing and safe execution.
"""
from __future__ import annotations

import threading
from typing import Dict, Any, List, Set, Tuple, Optional
from src.platform.interface import PlatformController, MouseEvent, KeyEvent


class DryRunController(PlatformController):
    """
    Dry run controller that logs events instead of executing them.
    Thread-safe.
    """
    
    def __init__(self):
        self.event_log: List[Dict[str, Any]] = []
        self.held_buttons: Set[str] = set()
        self.held_keys: Set[str] = set()
        self.gate: Optional[Any] = None
        self._lock = threading.Lock()
        
    def _log_event(self, event_type: MouseEvent | KeyEvent, **kwargs) -> None:
        with self._lock:
            self._log_event_unlocked(event_type, **kwargs)

    def _log_event_unlocked(self, event_type: MouseEvent | KeyEvent, **kwargs) -> None:
        """Log event without acquiring lock (caller must hold lock)."""
        self.event_log.append({
            "type": event_type,
            **kwargs
        })

    def _dispatch(self, action_fn) -> Any:
        if getattr(self, "gate", None) is not None:
            return self.gate.dispatch(action_fn)
        return action_fn()

    def _release(self, release_fn) -> Any:
        if getattr(self, "gate", None) is not None:
            return self.gate.release(release_fn)
        return release_fn()

    def mouse_move(self, x: int, y: int) -> None:
        self._dispatch(lambda: self._log_event(MouseEvent.MOVE, x=x, y=y))

    def mouse_down(self, button: str = 'left') -> None:
        def _do_down():
            with self._lock:
                self.held_buttons.add(button)
                self._log_event_unlocked(
                    MouseEvent.LEFT_DOWN if button == 'left' else MouseEvent.LEFT_DOWN,
                    button=button,
                )
        self._dispatch(_do_down)

    def mouse_up(self, button: str = 'left') -> None:
        def _do_up():
            with self._lock:
                self.held_buttons.discard(button)
                self._log_event_unlocked(
                    MouseEvent.LEFT_UP if button == 'left' else MouseEvent.LEFT_UP,
                    button=button,
                )
        self._release(_do_up)

    def mouse_click(self, button: str = 'left') -> None:
        evt = MouseEvent.LEFT_CLICK if button == 'left' else MouseEvent.RIGHT_CLICK
        self._dispatch(lambda: self._log_event(evt, button=button))

    def scroll(self, dx: int, dy: int) -> None:
        self._dispatch(lambda: self._log_event(MouseEvent.SCROLL, dx=dx, dy=dy))

    def key_down(self, key: str) -> None:
        def _do_key_down():
            with self._lock:
                self.held_keys.add(key)
                self._log_event_unlocked(KeyEvent.KEY_DOWN, key=key)
        self._dispatch(_do_key_down)

    def key_up(self, key: str) -> None:
        def _do_key_up():
            with self._lock:
                self.held_keys.discard(key)
                self._log_event_unlocked(KeyEvent.KEY_UP, key=key)
        self._release(_do_key_up)

    def cleanup(self) -> None:
        def _do_cleanup():
            with self._lock:
                buttons_to_release = list(self.held_buttons)
                keys_to_release = list(self.held_keys)
                
                for btn in buttons_to_release:
                    self._log_event_unlocked(
                        MouseEvent.LEFT_UP if btn == 'left' else MouseEvent.LEFT_UP,
                        button=btn, cleanup=True,
                    )
                    self.held_buttons.discard(btn)
                    
                for key in keys_to_release:
                    self._log_event_unlocked(KeyEvent.KEY_UP, key=key, cleanup=True)
                    self.held_keys.discard(key)
        self._release(_do_cleanup)

    def switch_tab(self, direction: str) -> None:
        """Dry-run simulation of application tab switching with conservative modifier tracking."""
        is_prev = direction.upper() in ("PREV", "LEFT", "BACK")
        modifiers = ['ctrl', 'shift'] if is_prev else ['ctrl']
        key = 'tab'

        def _do_switch():
            gate = getattr(self, "gate", None)
            if gate is not None and getattr(gate, "is_stopped", False):
                return

            acquired_mods = []
            try:
                for mod in modifiers:
                    if gate is not None and getattr(gate, "is_stopped", False):
                        return
                    self.key_down(mod)
                    acquired_mods.append(mod)

                if gate is not None and getattr(gate, "is_stopped", False):
                    return

                with self._lock:
                    self.held_keys.add(key)
                    self._log_event_unlocked(KeyEvent.KEY_DOWN, key=key)

                with self._lock:
                    self._log_event_unlocked(KeyEvent.KEY_UP, key=key)
                    self.held_keys.discard(key)
            finally:
                cleanup_exc: Optional[Exception] = None
                for mod in reversed(acquired_mods):
                    try:
                        self.key_up(mod)
                    except Exception as e:
                        if cleanup_exc is None:
                            cleanup_exc = e
                if cleanup_exc is not None:
                    raise cleanup_exc

        self._dispatch(_do_switch)

    def get_screen_size(self) -> Tuple[int, int]:
        return (1920, 1080)
