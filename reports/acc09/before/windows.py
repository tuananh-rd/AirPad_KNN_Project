"""
Windows-specific platform controller.
"""
from __future__ import annotations

import logging
from typing import Set, Tuple
from src.platform.interface import PlatformController

logger = logging.getLogger(__name__)


class WindowsController(PlatformController):
    """
    Windows platform controller for real mouse and keyboard actions.
    Protected by explicit `allow_real_control=True` flag to prevent accidental execution.
    """
    def __init__(self, allow_real_control: bool = False):
        if not allow_real_control:
            raise RuntimeError(
                "WindowsController requires explicit allow_real_control=True; "
                "use DryRunController for safe execution."
            )
            
        self._held_buttons: Set[str] = set()
        self._held_keys: Set[str] = set()
        
        # Lazy imports to avoid side effects on import and allow optional installation
        try:
            import ctypes
            import pyautogui
        except ImportError as e:
            raise ImportError(
                "Windows real control requires 'pyautogui'. "
                "Install with `pip install pyautogui` or `pip install .[platform-real]`."
            ) from e
        
        self.ctypes = ctypes
        self.pyautogui = pyautogui
        self.pyautogui.FAILSAFE = False

    def mouse_move(self, x: int, y: int) -> None:
        self.pyautogui.moveTo(int(x), int(y))

    def mouse_down(self, button: str = 'left') -> None:
        self.pyautogui.mouseDown(button=button)
        self._held_buttons.add(button)

    def mouse_up(self, button: str = 'left') -> None:
        self.pyautogui.mouseUp(button=button)
        self._held_buttons.discard(button)

    def mouse_click(self, button: str = 'left') -> None:
        self.pyautogui.click(button=button)

    def scroll(self, dx: int, dy: int) -> None:
        if dy != 0:
            self.pyautogui.scroll(int(dy))
        if hasattr(self.pyautogui, 'hscroll') and dx != 0:
            self.pyautogui.hscroll(int(dx))

    def key_down(self, key: str) -> None:
        self.pyautogui.keyDown(key)
        self._held_keys.add(key)

    def key_up(self, key: str) -> None:
        self.pyautogui.keyUp(key)
        self._held_keys.discard(key)

    def cleanup(self) -> None:
        """Idempotent release of all held buttons and keys using try/finally."""
        buttons = list(self._held_buttons)
        keys = list(self._held_keys)
        
        for btn in buttons:
            try:
                self.pyautogui.mouseUp(button=btn)
            except Exception as e:
                logger.error(f"Failed to release button {btn}: {e}")
            finally:
                self._held_buttons.discard(btn)
                
        for key in keys:
            try:
                self.pyautogui.keyUp(key)
            except Exception as e:
                logger.error(f"Failed to release key {key}: {e}")
            finally:
                self._held_keys.discard(key)

    def get_screen_size(self) -> Tuple[int, int]:
        size = self.pyautogui.size()
        return (int(size[0]), int(size[1]))

    def zoom(self, steps: int) -> None:
        if steps == 0:
            return
        try:
            self.pyautogui.keyDown('ctrl')
            self.pyautogui.scroll(int(steps))
        finally:
            self.pyautogui.keyUp('ctrl')

    def swipe(self, direction: str) -> None:
        key = 'left' if direction.upper() == 'BACK' else 'right'
        try:
            self.pyautogui.keyDown('alt')
            self.pyautogui.press(key)
        finally:
            self.pyautogui.keyUp('alt')
