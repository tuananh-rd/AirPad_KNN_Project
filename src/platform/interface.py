"""
Platform interface for mouse and keyboard control.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Protocol, Tuple


class MouseEvent(Enum):
    """Mouse event types."""
    MOVE = auto()
    LEFT_DOWN = auto()
    LEFT_UP = auto()
    LEFT_CLICK = auto()
    RIGHT_CLICK = auto()
    SCROLL = auto()


class KeyEvent(Enum):
    """Key event types."""
    KEY_DOWN = auto()
    KEY_UP = auto()


class PlatformController(Protocol):
    """Protocol for platform interaction (mouse/keyboard)."""
    
    def mouse_move(self, x: int, y: int) -> None:
        """Move mouse to absolute coordinates."""
        ...
        
    def mouse_down(self, button: str = 'left') -> None:
        """Press and hold mouse button."""
        ...
        
    def mouse_up(self, button: str = 'left') -> None:
        """Release mouse button."""
        ...
        
    def mouse_click(self, button: str = 'left') -> None:
        """Click mouse button."""
        ...
        
    def scroll(self, dx: int, dy: int) -> None:
        """Scroll wheel."""
        ...
        
    def key_down(self, key: str) -> None:
        """Press and hold key."""
        ...
        
    def key_up(self, key: str) -> None:
        """Release key."""
        ...
        
    def cleanup(self) -> None:
        """Release ALL held buttons/keys."""
        ...
        
    def switch_tab(self, direction: str) -> None:
        """Switch application tab (direction: 'PREV' | 'NEXT')."""
        ...
        
    def get_screen_size(self) -> Tuple[int, int]:
        """Return screen size as (width, height)."""
        ...
