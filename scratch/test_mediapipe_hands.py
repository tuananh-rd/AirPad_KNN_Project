import mediapipe as mp
import numpy as np

print("Testing MediaPipe Hands initialization...", flush=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# Create a blank black image
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
results = hands.process(dummy_image)
print("MediaPipe Hands processed dummy frame successfully!", flush=True)
print("Hand landmarks detected on blank frame:", results.multi_hand_landmarks)
hands.close()
print("MEDIAPIPE_HANDS_TEST_PASSED", flush=True)
