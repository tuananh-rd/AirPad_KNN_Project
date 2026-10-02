"""
Safety and failsafe mechanisms.
"""
from __future__ import annotations

import time
import threading
from typing import Callable, Optional


class RateLimiter:
    """Rate limiter to ensure we don't spam OS events."""
    
    def __init__(self, max_events_per_second: int):
        self.max_events_per_second = max_events_per_second
        self.interval = 1.0 / max_events_per_second
        self.last_event_time = 0.0
        self._lock = threading.Lock()
        
    def allow(self) -> bool:
        """Check if an event is allowed based on rate limits."""
        with self._lock:
            now = time.time()
            if now - self.last_event_time >= self.interval:
                self.last_event_time = now
                return True
            return False


class FailsafeMonitor:
    """Monitors for failsafe conditions like cursor in corner."""
    
    def __init__(self, screen_width: int, screen_height: int, corner_size: int = 10):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.corner_size = corner_size
        self._is_stopped = False
        
    def check_position(self, x: int, y: int) -> bool:
        """
        Check if position is in a failsafe corner (top-left or top-right etc).
        Returns True if failsafe triggered, False otherwise.
        """
        if self._is_stopped:
            return True
            
        if (
            (x <= self.corner_size and y <= self.corner_size)
            or (x >= self.screen_width - self.corner_size and y <= self.corner_size)
            or (x <= self.corner_size and y >= self.screen_height - self.corner_size)
            or (x >= self.screen_width - self.corner_size and y >= self.screen_height - self.corner_size)
        ):
            self.emergency_stop()
            return True
            
        return False
        
    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._is_stopped = True
        
    def is_stopped(self) -> bool:
        """Check if emergency stop was triggered."""
        return self._is_stopped


class EmergencyHotkey:
    """Emergency hotkey listener."""
    
    def __init__(self, hotkey: str, callback: Callable[[], None], dry_run: bool = False):
        self.hotkey = hotkey
        self.callback = callback
        self.dry_run = dry_run
        self._registered = False
        
        if not dry_run:
            try:
                import keyboard
                self.keyboard = keyboard
            except ImportError:
                print("Warning: 'keyboard' module not installed, hotkey disabled.")
                self.dry_run = True

    def register(self) -> None:
        """Register the hotkey."""
        if self.dry_run or self._registered:
            return
            
        try:
            self.keyboard.add_hotkey(self.hotkey, self.callback)
            self._registered = True
        except Exception as e:
            print(f"Failed to register hotkey: {e}")
            
    def unregister(self) -> None:
        """Unregister the hotkey."""
        if self.dry_run or not self._registered:
            return
            
        try:
            self.keyboard.remove_hotkey(self.hotkey)
            self._registered = False
        except Exception:
            pass
            
    def trigger_emergency(self) -> None:
        """Manually trigger for testing or dry_run."""
        self.callback()
