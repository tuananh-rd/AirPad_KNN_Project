import sys

def test_import(name, stmt):
    print(f"Testing {name}...", end=" ", flush=True)
    try:
        exec(stmt)
        print("OK", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

test_import("numpy", "import numpy")
test_import("yaml", "import yaml")
test_import("jsonschema", "import jsonschema")
test_import("sklearn", "import sklearn")
test_import("cv2", "import cv2")
test_import("mediapipe", "import mediapipe as mp")
test_import("pyautogui", "import pyautogui")
test_import("keyboard", "import keyboard")
test_import("pytest", "import pytest")

print("ALL_TESTS_DONE", flush=True)
