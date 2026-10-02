import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import numpy as np
    print("numpy:", np.__version__)
except Exception as e:
    print("numpy FAILED:", e)

try:
    import yaml
    print("yaml:", yaml.__version__)
except Exception as e:
    print("yaml FAILED:", e)

try:
    import jsonschema
    print("jsonschema:", jsonschema.__version__)
except Exception as e:
    print("jsonschema FAILED:", e)

try:
    import sklearn
    print("sklearn:", sklearn.__version__)
except Exception as e:
    print("sklearn FAILED:", e)

try:
    import cv2
    print("cv2:", cv2.__version__)
except Exception as e:
    print("cv2 FAILED:", e)

try:
    import mediapipe as mp
    print("mediapipe:", mp.__version__)
    # Verify mp.solutions.hands exists
    print("mediapipe hands:", hasattr(mp.solutions, "hands"))
except Exception as e:
    print("mediapipe FAILED:", e)

try:
    import pyautogui
    print("pyautogui:", pyautogui.__version__)
except Exception as e:
    print("pyautogui FAILED:", e)

try:
    import keyboard
    print("keyboard:", keyboard.__version__)
except Exception as e:
    print("keyboard FAILED:", e)

try:
    import pytest
    print("pytest:", pytest.__version__)
except Exception as e:
    print("pytest FAILED:", e)

print("IMPORT_VERIFICATION_COMPLETE")
