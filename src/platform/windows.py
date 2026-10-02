"""
Windows-specific platform controller.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Set, Tuple
from src.platform.interface import PlatformController
from src.safety.failsafe import SafetyGate

logger = logging.getLogger(__name__)


class WindowsController(PlatformController):
    """
    Windows platform controller for real mouse and keyboard actions.
    Protected by explicit `allow_real_control=True` flag to prevent accidental execution.
    Synchronized via SafetyGate to guarantee fail-closed dispatch and clean held-input release.
    """
    def __init__(self, allow_real_control: bool = False, gate: Optional[SafetyGate] = None):
        if not allow_real_control:
            raise RuntimeError(
                "WindowsController requires explicit allow_real_control=True; "
                "use DryRunController for safe execution."
            )
        self.allow_real_control = True
        self.gate: SafetyGate = gate if gate is not None else SafetyGate()
        self._lock = threading.RLock()

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

    @property
    def held_buttons(self) -> Set[str]:
        return set(self._held_buttons)

    @property
    def held_keys(self) -> Set[str]:
        return set(self._held_keys)

    def _dispatch(self, action_fn, *args, **kwargs) -> Any:
        return self.gate.dispatch(action_fn, *args, **kwargs)

    def _release(self, release_fn, *args, **kwargs) -> Any:
        return self.gate.release(release_fn, *args, **kwargs)

    def mouse_move(self, x: int, y: int) -> None:
        self._dispatch(lambda: self.pyautogui.moveTo(int(x), int(y)))

    def mouse_down(self, button: str = 'left') -> None:
        def _do_down():
            self._held_buttons.add(button)
            self.pyautogui.mouseDown(button=button)
        self._dispatch(_do_down)

    def mouse_up(self, button: str = 'left') -> None:
        def _do_up():
            try:
                self.pyautogui.mouseUp(button=button)
                self._held_buttons.discard(button)
            except Exception:
                raise
        self._release(_do_up)

    def mouse_click(self, button: str = 'left') -> None:
        def _do_click():
            self._held_buttons.add(button)
            try:
                self.pyautogui.click(button=button)
                self._held_buttons.discard(button)
            except Exception:
                # Partial down failure: conservatively retain button for cleanup retry
                raise
        self._dispatch(_do_click)

    def scroll(self, dx: int, dy: int) -> None:
        def _do_scroll():
            if dy != 0:
                self.pyautogui.scroll(int(dy))
            if hasattr(self.pyautogui, 'hscroll') and dx != 0:
                self.pyautogui.hscroll(int(dx))
        self._dispatch(_do_scroll)

    def key_down(self, key: str) -> None:
        def _do_key_down():
            self._held_keys.add(key)
            self.pyautogui.keyDown(key)
        self._dispatch(_do_key_down)

    def key_up(self, key: str) -> None:
        def _do_key_up():
            try:
                self.pyautogui.keyUp(key)
                self._held_keys.discard(key)
            except Exception:
                raise
        self._release(_do_key_up)

    def cleanup(self) -> None:
        """Idempotent release of all held buttons and keys.

        Attempts all cleanup releases. Retains failed buttons/keys for subsequent
        cleanup retry, and raises the first cleanup exception if any failed.
        """
        def _do_cleanup():
            buttons = list(self._held_buttons)
            keys = list(self._held_keys)
            cleanup_exc: Optional[Exception] = None

            for btn in buttons:
                try:
                    self.pyautogui.mouseUp(button=btn)
                    self._held_buttons.discard(btn)
                except Exception as e:
                    logger.error(f"Failed to release button {btn}: {e}")
                    if cleanup_exc is None:
                        cleanup_exc = e

            for key in keys:
                try:
                    self.pyautogui.keyUp(key)
                    self._held_keys.discard(key)
                except Exception as e:
                    logger.error(f"Failed to release key {key}: {e}")
                    if cleanup_exc is None:
                        cleanup_exc = e

            if cleanup_exc is not None:
                raise cleanup_exc

        self._release(_do_cleanup)

    def get_screen_size(self) -> Tuple[int, int]:
        size = self.pyautogui.size()
        return (int(size[0]), int(size[1]))

    def zoom(self, steps: int) -> None:
        if steps == 0:
            return
        def _do_zoom():
            self.key_down('ctrl')
            try:
                self.pyautogui.scroll(int(steps))
            finally:
                if 'ctrl' in self._held_keys:
                    self.key_up('ctrl')
        self._dispatch(_do_zoom)

    def swipe(self, direction: str) -> None:
        key = 'left' if direction.upper() == 'BACK' else 'right'
        def _do_swipe():
            self.key_down('alt')
            try:
                self._held_keys.add(key)
                try:
                    self.pyautogui.press(key)
                    self._held_keys.discard(key)
                except Exception:
                    raise
            finally:
                if 'alt' in self._held_keys:
                    self.key_up('alt')
        self._dispatch(_do_swipe)

    def switch_tab(self, direction: str) -> None:
        """Switch application tab with conservative modifier tracking and fail-closed safety.

        - 'PREV', 'LEFT', 'BACK': previous tab via Ctrl+Shift+Tab
        - 'NEXT', 'RIGHT', 'FORWARD': next tab via Ctrl+Tab
        Modifiers ('ctrl', 'shift') and 'tab' are tracked before OS invocation.
        Gate stop is rechecked before each action.
        Modifiers are guaranteed released in finally block only if held tracking confirms acquisition.
        Failed releases or unconfirmed press keys are retained in _held_keys for cleanup retry.
        """
        is_prev = direction.upper() in ("PREV", "LEFT", "BACK")
        modifiers = ['ctrl', 'shift'] if is_prev else ['ctrl']
        key = 'tab'

        def _do_switch_tab():
            if self.gate.is_stopped:
                return

            attempted_mods = []
            try:
                for mod in modifiers:
                    if self.gate.is_stopped:
                        return
                    attempted_mods.append(mod)
                    self.key_down(mod)

                if self.gate.is_stopped:
                    return

                self._held_keys.add(key)
                try:
                    self.pyautogui.press(key)
                    self._held_keys.discard(key)
                except Exception:
                    raise
            finally:
                cleanup_exc: Optional[Exception] = None
                for mod in reversed(attempted_mods):
                    if mod in self._held_keys:
                        try:
                            self.key_up(mod)
                        except Exception as e:
                            if cleanup_exc is None:
                                cleanup_exc = e
                if cleanup_exc is not None:
                    raise cleanup_exc

        self._dispatch(_do_switch_tab)

