import sys
import pytest
from unittest.mock import MagicMock

from src.platform.windows import WindowsController


@pytest.fixture
def mock_gui(monkeypatch):
    mock = MagicMock()
    mock.size.return_value = (1920, 1080)
    mock.hscroll = MagicMock()
    monkeypatch.setitem(sys.modules, 'pyautogui', mock)
    return mock


def test_windows_controller_requires_allow_real_control():
    with pytest.raises(RuntimeError, match="requires explicit allow_real_control=True"):
        WindowsController(allow_real_control=False)


def test_windows_controller_missing_pyautogui(monkeypatch):
    # Hide pyautogui from sys.modules
    monkeypatch.setitem(sys.modules, 'pyautogui', None)
    with pytest.raises(ImportError, match="Windows real control requires 'pyautogui'"):
        WindowsController(allow_real_control=True)


def test_windows_controller_mouse_move(mock_gui):
    controller = WindowsController(allow_real_control=True)
    controller.mouse_move(100, 200)
    mock_gui.moveTo.assert_called_once_with(100, 200)


def test_windows_controller_mouse_actions(mock_gui):
    controller = WindowsController(allow_real_control=True)
    
    controller.mouse_down('left')
    mock_gui.mouseDown.assert_called_once_with(button='left')
    assert 'left' in controller._held_buttons
    
    controller.mouse_up('left')
    mock_gui.mouseUp.assert_called_once_with(button='left')
    assert 'left' not in controller._held_buttons
    
    controller.mouse_click('right')
    mock_gui.click.assert_called_once_with(button='right')


def test_windows_controller_scroll(mock_gui):
    controller = WindowsController(allow_real_control=True)
    
    controller.scroll(0, 10)
    mock_gui.scroll.assert_called_with(10)
    
    controller.scroll(5, 0)
    mock_gui.hscroll.assert_called_with(5)


def test_windows_controller_zoom_swipe(mock_gui):
    controller = WindowsController(allow_real_control=True)
    
    controller.zoom(3)
    mock_gui.keyDown.assert_any_call('ctrl')
    mock_gui.scroll.assert_called_with(3)
    mock_gui.keyUp.assert_any_call('ctrl')
    
    controller.swipe('BACK')
    mock_gui.keyDown.assert_any_call('alt')
    mock_gui.press.assert_any_call('left')
    mock_gui.keyUp.assert_any_call('alt')


def test_windows_controller_cleanup(mock_gui):
    controller = WindowsController(allow_real_control=True)
    
    controller.mouse_down('left')
    controller.mouse_down('right')
    controller.key_down('shift')
    
    # Make first mouseUp raise an exception; subsequent mouseUp calls succeed
    mock_gui.mouseUp.side_effect = [Exception("error"), None, None]
    
    with pytest.raises(Exception, match="error"):
        controller.cleanup()
    
    # The failed button is retained for retry; the other button and key succeeded
    assert len(controller._held_buttons) == 1
    assert len(controller._held_keys) == 0
    
    # Second cleanup retries and clears the remaining button
    controller.cleanup()
    assert len(controller._held_buttons) == 0
    assert len(controller._held_keys) == 0
    
    # Check idempotent
    controller.cleanup()


def test_windows_controller_screen_size(mock_gui):
    controller = WindowsController(allow_real_control=True)
    assert controller.get_screen_size() == (1920, 1080)
