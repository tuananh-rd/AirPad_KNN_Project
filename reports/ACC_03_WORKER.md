# ACC-03 Worker Implementation Report: OpenCV BGR to MediaPipe RGB Contract Repair

**Worker**: Antigravity in airpad-worker  
**Task ID**: `task_90b6735ee553`  
**Dispatch ID**: `ctx_88794eec8d79`  
**Coordinator Terminal**: `term_eb1d1c40-1bf5-4fd5-ab4e-0d3083b40673`  
**Worker Terminal**: `term_e986c838-27b3-4389-9d6a-260fa2ab2324`  
**Date**: 2026-10-01  

---

## 1. Executive Summary

1. **What was done**:
   Repaired the camera input contract in `tools/collect_data.py` so captured OpenCV BGR frames are horizontally flipped, converted to a dedicated RGB copy via `rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`, and passed to `detector.detect(rgb_frame)`, while keeping the displayed frame (`frame`) strictly in BGR format for HUD text annotations and `cv2.imshow`.

2. **What was found**:
   Verified that MediaPipe HandLandmarker natively requires RGB color channel ordering (as standardized across `src/vision/adapter.py`), whereas `tools/collect_data.py` had previously passed raw BGR frames directly to the detector, causing inverted color channels during interactive webcam collection; regression testing confirmed all 56 tests in `tests/test_tools.py` and the complete 261-test project suite pass cleanly with zero regressions.

3. **What is next / left**:
   All requirements for ACC-03 are fulfilled; dataset files (`custom_airpad.npz`, `hagrid_airpad.npz`, `combined_airpad.npz`) and model weights (`airpad_svm.json`) remain byte-for-byte identical, with zero camera operations or Git repository initialization, ready for Coordinator review.

---

## 2. Baseline and Pre-Edit Artifact Verification

### 2.1 Brain Baseline Dataset and Model Hashes
All baseline artifacts were verified prior to making edits and re-verified post-implementation; zero bytes were changed:

| Dataset / Model | File Path | Expected SHA-256 Checksum | Verified SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- | :--- |
| `custom` | `data/custom_airpad.npz` | `498699e1f28a78031193476fbafff2a2cd307bb9d82a8baee6a57f3097019169` | `498699e1f28a78031193476fbafff2a2cd307bb9d82a8baee6a57f3097019169` | MATCH |
| `hagrid` | `data/hagrid_airpad.npz` | `ec38045d608aa6d30386796e4e0586ff619468d4abfd755619835bb8fa61c081` | `ec38045d608aa6d30386796e4e0586ff619468d4abfd755619835bb8fa61c081` | MATCH |
| `combined` | `data/combined_airpad.npz` | `c0c6d4e26c26f8a303077c6de971a54c2d4085444f9479cef9e021986cc9e8bf` | `c0c6d4e26c26f8a303077c6de971a54c2d4085444f9479cef9e021986cc9e8bf` | MATCH |
| `model` | `models/airpad_svm.json` | `4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977` | `4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977` | MATCH |

### 2.2 Pre-Edit Backups (`reports/acc03/before/`)
Before modifying any source code, read-only copies and checksum manifests were created:
- Backup: `reports/acc03/before/tools/collect_data.py`
  - SHA-256: `1a1c43225045a979957aa65e183fc9f9f8b52d88924b7f55e75d776d0daeda17`
- Manifest: `reports/acc03/before/manifest.sha256` and `reports/acc03/before/MANIFEST.md`

---

## 3. Implementation Details (`tools/collect_data.py`)

### 3.1 BGR-to-RGB Conversion Contract
In `tools/collect_data.py` inside the live interactive loop (`collect_data_interactive`):
- Captured frames from `cv2.VideoCapture` are read in OpenCV default BGR format.
- The frame is horizontally mirrored for user-intuitive selfie camera interaction: `frame = cv2.flip(frame, 1)`.
- Prior code passed `frame` directly to `detector.detect(frame)`. Because `detector` is an instance of `src.vision.adapter.MediaPipeAdapter`, MediaPipe converts the numpy array to `mp.Image(image_format=mp.ImageFormat.SRGB, data=...)`. Passing BGR caused MediaPipe's underlying ML model to process Red as Blue and Blue as Red.
- The code was updated to convert a separate copy to RGB:
  ```python
  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  hands = detector.detect(rgb_frame)
  ```
- The original `frame` remains untouched BGR. All subsequent HUD rendering (`cv2.putText`, `cv2.line`, `cv2.circle`) draws directly onto `frame` using standard BGR colors:
  - Green status text and keypoints: `(0, 255, 0)`
  - Yellow target label: `(0, 255, 255)`
  - Blue/Orange skeleton lines: `(255, 200, 0)`
  - Red recording warning: `(0, 0, 255)`
- `cv2.imshow("AirPad Pose Collector", frame)` displays the BGR frame without color distortion.

### 3.2 Mock Mode Backward Compatibility
- For unit testing without hardware cameras, `mock_frames` accepts pre-computed landmarks and handedness tuples: `List[Tuple[np.ndarray, str]]`.
- The live camera loop is only executed when `mock_frames is None`. Mock mode remains completely unaffected and preserved.

---

## 4. Verification and Test Coverage

### 4.1 Unit Test Design (`tests/test_tools.py`)
Added `test_collect_data_opencv_bgr_to_mediapipe_rgb_contract` to `TestCollectData` in `tests/test_tools.py`:
1. **Synthetic Asymmetric Frame**: Created a $(200, 200, 3)$ `raw_bgr` frame with asymmetric spatial and channel properties:
   - Left half ($x \in [0, 100)$): $B=10, G=50, R=200$
   - Right half ($x \in [100, 200)$): $B=30, G=50, R=240$
2. **Camera Mocking**: Mocked `cv2.VideoCapture` returning `raw_bgr`, with `cv2.waitKey` controlling the loop termination.
3. **Detection and Display Spy**: Spied on arguments passed to `detector.detect` (`det_frame`) and `cv2.imshow` (`disp_frame`).
4. **Assertions**:
   - Both frames have shape $(200, 200, 3)$.
   - After horizontal flip (`cv2.flip(frame, 1)`):
     - Left half receives right half pixels ($B=30, G=50, R=240$).
     - Right half receives left half pixels ($B=10, G=50, R=200$).
   - `disp_frame` retains BGR channel order in unannotated middle rows ($y \in [100, 150)$):
     - Left half: $B=30, G=50, R=240$.
     - Right half: $B=10, G=50, R=200$.
   - `det_frame` receives strict RGB channel order:
     - Left half: $R=240, G=50, B=30$.
     - Right half: $R=200, G=50, B=10$.
   - Channel-swap identity:
     - `det_frame[100:150, :, 0] == disp_frame[100:150, :, 2]` (Red)
     - `det_frame[100:150, :, 1] == disp_frame[100:150, :, 1]` (Green)
     - `det_frame[100:150, :, 2] == disp_frame[100:150, :, 0]` (Blue)

### 4.2 Test Execution Results
- `tests/test_tools.py`:
  - Command: `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests\test_tools.py`
  - Result: **56 passed in 3.10s** (100%)
- Full Test Suite:
  - Command: `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider`
  - Result: **261 passed in 104.64s** (100%)

---

## 5. File Manifest & Checksums

| File Path | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `reports/acc03/before/tools/collect_data.py` | Pre-edit collector backup | `1a1c43225045a979957aa65e183fc9f9f8b52d88924b7f55e75d776d0daeda17` |
| `reports/acc03/before/manifest.sha256` | Pre-edit SHA-256 manifest | `8448ebf5fe81da00851ec6522c0fc44026ae67b36f52e3e57f208197779d7ce7` |
| `reports/acc03/before/MANIFEST.md` | Pre-edit baseline documentation | `7eb08cbf7f174c43144243a0595300305f8179261ff2f4bf6c63b4fe76d7fbbe` |
| `tools/collect_data.py` | Repaired collector (BGR to RGB) | `98f1d390dc8f51c54fa0ace66e906e897597d2bb0173620f0d978fc96f774455` |
| `tests/test_tools.py` | Regression test suite with camera contract test | `bbcc9943d3497ab36a5dfe78e06da2a7350a319bef344bab0ff49b60deb57783` |
| `reports/acc03/before_after.diff` | Unified diff of source changes | `bb1172e387f9b055f6451923a30c159084281a40133be341f87c8bc1545aab9d` |
| `reports/ACC_03_WORKER.md` | This implementation report | (Current document) |

---

## 6. Constraints & Commitments Confirmation

- **No Git Repository Initialization**: `.git` was checked and verified non-existent; no git operations were executed.
- **No Live Webcam / Recording**: Zero hardware cameras were opened; all testing utilized isolated synthetic arrays and mock objects.
- **Production Artifact Integrity**: `data/custom_airpad.npz`, `data/hagrid_airpad.npz`, `data/combined_airpad.npz`, and `models/airpad_svm.json` were strictly untouched.
- **Scope Discipline**: Changes were strictly limited to `tools/collect_data.py`, focused unit test additions in `tests/test_tools.py`, and documentation in `reports/`.
