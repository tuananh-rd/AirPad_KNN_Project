# AirPad RBF-SVM Project

**Virtual air-mouse pad for Windows** — local processing, camera-based hand tracking.

An RBF-SVM classifies **static hand poses only**; all dynamic gestures (click, drag, scroll, zoom, swipe) are handled by the existing temporal FSM and motion pipeline.

## Architecture

```
config/default.yaml        ← YAML config + JSON schema validation
src/
├── config/                ← Config loader (singleton, thread-safe, semantic validation)
├── vision/                ← Landmark types, MediaPipe adapter (safe import)
├── features/              ← Canonicalization: wrist-origin, scale, mirror, rotation, z-weight
├── ml/                    ← RBF-SVM classifier, evaluation, and model bundle
├── gesture/               ← HandPose enum, temporal filters, GestureFSM
├── motion/                ← Filters (EMA, OneEuro), CursorMapper, ScrollEngine, ZoomEngine, SwipeEngine
├── platform/              ← Controller interface, DryRunController, WindowsController (lazy import)
├── runtime/               ← AirPadPipeline orchestrator & PipelineResult telemetry
└── safety/                ← Rate limiter, failsafe monitor, emergency hotkey
tools/
├── collect_data.py        ← Schema-validated dataset collection (.npz / .csv)
└── train.py               ← Leak-free grouped training (GroupKFold/LOSO), scaler, bundle export
fixtures/
└── synthetic_replay.json  ← Deterministic 36-frame 5-phase landmark test sequence
app.py                     ← CLI: --self-check, --replay, --camera, --real-control
tests/                     ← 100% deterministic unit + replay tests (no OS side effects)
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Install minimal (development + test)
pip install -e ".[benchmark]"
pip install pytest

# Install full camera preview and real control (optional)
pip install -e ".[all]"
```

### Optional Dependencies

| Extra | Package | Purpose |
|-------|---------|---------|
| `vision` | mediapipe, opencv-python | Hand detection & camera preview |
| `camera` | opencv-python, mediapipe | Alias for camera support |
| — | scikit-learn | Required for StandardScaler and the production RBF-SVM |
| `platform-real` | pyautogui | Real Windows mouse & keyboard control |
| `hotkey-real` | keyboard | OS global emergency hotkey |
| `all` | all above | Full environment |

## CLI Quickstart

```bash
# 1. Run internal self-check (no camera required, no OS side effects):
python app.py --self-check

# 2. Replay landmark fixture in dry-run mode:
python app.py --replay fixtures/synthetic_replay.json

# 3. Run camera tracking preview (mirror preview with status overlay, Esc to exit):
python app.py --camera

# 4. Enable real OS mouse control (explicit flag required, prints safety banner):
python app.py --camera --real-control
```

> [!WARNING]
> `--real-control` enables actual Windows OS mouse and keyboard control via `WindowsController`.
> In case of unexpected behavior, trigger the **Emergency Stop Hotkey: Ctrl+Alt+Q** or push your mouse into any screen corner.

## Gesture Contract (MVP)

| # | Pose/Action | Behavior |
|---|-------------|----------|
| 1 | Startup | PAUSED state |
| 2 | OPEN_PALM / NO_HAND | Clutch/pause, release all, reset baseline |
| 3 | POINT_ONE stable ~100ms | → MOVE mode (cursor tracking active) |
| 4 | Quick pinch 80–300ms | LEFT_CLICK (clean release to POINT_ONE) |
| 5 | Pinch hold >400ms | DRAG: mouseDown once, mouseUp guaranteed on release/loss/unknown |
| 6 | PINCH_MIDDLE quick pinch | RIGHT_CLICK (clean release to POINT_ONE, edge-triggered) |
| 7 | TWO_FINGERS_PARALLEL | SCROLL mode (dominant-axis lock, dead-zone, gain) |
| 8 | L_SHAPE dwell 250ms | ZOOM latch (aperture distance delta → semantic zoom steps) |
| 9 | THREE_FINGERS swipe | BACK / FORWARD (displacement + velocity + cooldown + re-arm) |
| 10 | Emergency hotkey | Immediate cleanup + PAUSED |

**Key invariants:**
- RBF-SVM only classifies static poses — no `PINCH_DYNAMIC` class
- `NO_HAND` is detected before classification and `UNKNOWN` is a runtime rejection; neither is trained as a class
- `OTHER` remains a trainable catch-all class
- Click / right-click fires exactly once on release
- Drag: mouseDown exactly once; mouseUp guaranteed on release, hand loss, error, or unknown
- UNKNOWN / OTHER never issue commands; reacquisition requires stable consecutive frames
- Hand loss / error / exit → cleanup (release all buttons/keys)

## Testing

All tests run completely offline and deterministically without touching OS hardware:

```bash
pytest -v                        # Run all tests
pytest tests/test_svm.py         # Production RBF-SVM core
pytest tests/test_fsm.py         # Phase 1 FSM contract
pytest tests/test_motion.py      # Motion pipeline (filters, cursor, scroll, zoom, swipe)
pytest tests/test_fsm_phase2.py    # 2A-SAFETY FSM contract & guards
pytest tests/test_windows.py     # Mocked WindowsController
pytest tests/test_pipeline.py    # Pipeline integration & synthetic replay
pytest tests/test_tools.py       # Data tools, leak-free scaler, bundle save/load
pytest tests/test_app.py         # CLI commands (--self-check, --replay, --camera)
```

## Project Status

- ✅ **PHASE 1**: Foundation + static-pose ML core + Temporal Core + test fixtures.
- ✅ **PHASE 2A & 2A-SAFETY**:
  - `src/motion`: Filters (EMA, OneEuro in seconds/Hz), CursorMapper (ROI, mirror, clamp, dead-zone, gain), ScrollEngine (dominant-axis lock, sentinel rate limit), ZoomEngine (aperture delta), SwipeEngine (one-shot, re-arm).
  - `src/gesture/fsm.py`: Dedicated `PINCH_MIDDLE` right-click, clean release to `POINT_ONE`, mouseUp guaranteed on all DRAGGING exits, UNKNOWN/OTHER safety, reacquisition stability gating, removal of `pose=None` bypass.
  - `config`: default.yaml & schema.py extended with semantic timing hierarchy validation.
- ✅ **PHASE 2B**:
  - `src/platform/windows.py`: `WindowsController` with lazy import, strict `--real-control` guard, idempotent `try/finally` cleanup.
  - `src/runtime/pipeline.py`: Dependency-injected `AirPadPipeline` linking detector, canonicalizer, classifier, temporal filter, FSM, and motion engines. Fail-fast on `feature_version != 1`.
  - `tools/`: `collect_data.py` (schema validation, `.npz`/`.csv`) & `train.py` (GroupKFold/LOSO without temporal leakage, train-only scaler fit, bundle export).
  - `fixtures/synthetic_replay.json`: 36-frame 5-phase landmark test sequence without personal data.
  - `app.py`: CLI supporting `--self-check`, `--replay`, `--camera`, `--real-control`, mirror preview, status overlay, and Esc/finally cleanup.
  - Complete mock and unit tests for Windows platform, pipeline, tools, and CLI application.

## Static-pose model

Production inference uses `SVMClassifier`, which owns both a scikit-learn
`StandardScaler` and an `SVC(kernel="rbf")`. Raw 63-value canonical features are
passed to the classifier; scaling is therefore identical during training and
runtime. A prediction becomes `UNKNOWN` when either the winning decision score
is below `ml.decision_threshold` or the top-two score margin is below
`ml.margin_threshold`.

`tools/train.py` tunes a deliberately small `C`/`gamma` grid with grouped folds,
so frames from one session or user do not leak between train and validation.
SVM decision scores are not probabilities: calibrate both rejection thresholds
on held-out users/sessions from the real dataset, with false command activation
as a safety metric. The checked-in defaults are starting values, not universal
confidence guarantees.

## Importing HaGRIDv2 landmarks

AirPad can bootstrap six conservative pose classes from the official HaGRIDv2
annotation JSON files without downloading the full image archive:
`palm`, `one`, `fist`, `two_up`, `three`, and `no_gesture`. Ambiguous gestures
such as `ok` are deliberately not converted to pinch classes.

```bash
python -m tools.import_hagrid data/hagrid_v2/annotations data/hagrid_airpad.npz \
  --max-per-class 5000 --seed 42
```

## Training the RBF-SVM Model

Train the model bundle using zero-leakage GroupKFold cross-validation:

```bash
python -m tools.train data/hagrid_airpad.npz models/airpad_svm.json \
  --group-key user_id \
  --summary-path models/train_summary.json
```

## Custom Pose Data Collection

Collect custom pose data (such as `PINCH_CLOSED`, `PINCH_MIDDLE`, and `L_SHAPE`) directly from your webcam:

```bash
python -m tools.collect_data --output data/custom_airpad.npz --user-id user_01
```

**Controls during collection:**
- `[1]` PINCH_CLOSED, `[2]` PINCH_MIDDLE, `[3]` L_SHAPE, `[4]` OTHER, `[5]` POINT_ONE
- `[SPACE]` Capture single frame
- `[R]` Toggle continuous recording
- `[T]` Toggle transition flag
- `[S]` Save dataset to disk
- `[ESC / Q]` Save and exit

## Live Camera Execution

1. **Dry-Run Mode (Recommended first step)**:
   ```bash
   python app.py --camera --model models/airpad_svm.json
   ```
   Visualizes hand tracking, recognized pose, FSM state, and action events without sending OS input.

2. **Real OS Control Mode**:
   ```bash
   python app.py --camera --model models/airpad_svm.json --real-control
   ```
   *Emergency Stop*: Press **Ctrl+Alt+Q** or push the cursor to any screen corner.
