"""
Dry run platform controller for testing and safe execution.
"""
from __future__ import annotations

import threading
from typing import Dict, Any, List, Set, Tuple
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

    def mouse_move(self, x: int, y: int) -> None:
        self._log_event(MouseEvent.MOVE, x=x, y=y)

    def mouse_down(self, button: str = 'left') -> None:
        with self._lock:
            self.held_buttons.add(button)
            self._log_event_unlocked(
                MouseEvent.LEFT_DOWN if button == 'left' else MouseEvent.LEFT_DOWN,
                button=button,
            )

    def mouse_up(self, button: str = 'left') -> None:
        with self._lock:
            self.held_buttons.discard(button)
            self._log_event_unlocked(
                MouseEvent.LEFT_UP if button == 'left' else MouseEvent.LEFT_UP,
                button=button,
            )

    def mouse_click(self, button: str = 'left') -> None:
        evt = MouseEvent.LEFT_CLICK if button == 'left' else MouseEvent.RIGHT_CLICK
        self._log_event(evt, button=button)

    def scroll(self, dx: int, dy: int) -> None:
        self._log_event(MouseEvent.SCROLL, dx=dx, dy=dy)

    def key_down(self, key: str) -> None:
        with self._lock:
            self.held_keys.add(key)
            self._log_event_unlocked(KeyEvent.KEY_DOWN, key=key)

    def key_up(self, key: str) -> None:
        with self._lock:
            self.held_keys.discard(key)
            self._log_event_unlocked(KeyEvent.KEY_UP, key=key)

    def cleanup(self) -> None:
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

    def get_screen_size(self) -> Tuple[int, int]:
        return (1920, 1080)
