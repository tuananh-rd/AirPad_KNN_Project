# AirPad Shadow-Only Guided Observation Trial Guide

**Tool**: `tools/shadow_trial.py`  
**Purpose**: Safe, zero-action live (or recorded) camera observation trial for evaluating personal hand pose classifiers before production promotion.  
**Safety Status**: Zero action dispatch by construction (no platform controllers, no OS input synthesis, 0 commands dispatched).

---

## 1. Safety Guarantees & Resource Lifecycle

1. **Zero Action Dispatching by Construction**:
   - The shadow trial tool contains no imports or references to `src.platform`, `src.runtime.pipeline`, `src.gesture.fsm`, `pyautogui`, `pynput`, `keyboard`, or any OS input generation APIs.
   - It performs hand detection, landmark canonicalization, feature scaling, model scoring, and file logging **only**.
   - Verified via automated import hooks and API interceptors: `commands_dispatched` is strictly `0` across all metadata, summary, and frame records.
2. **Pre-Flight Flag Rejection**:
   - The CLI strictly rejects `--real-control`, `--dispatch`, `--execute`, or any flag attempting action execution before opening the camera device.
3. **Model & Detector Verification Before Handle Reservation**:
   - Candidate bundle SHA-256 and detector task model existence/hash are verified before any output handles are reserved and before camera acquisition.
   - If candidate or detector verification fails, no output files are created and the camera is never opened.
4. **Collision & Resource Lifecycle Guards**:
   - Output handles (`.jsonl` and summary `.json`) are reserved sequentially using exclusive creation mode `'x'` (`open(..., 'x')`).
   - If the second open fails (e.g. summary file collision), the newly created empty JSONL file is closed and safely cleaned up, while the pre-existing collision summary file is strictly preserved.
   - All operations after handle reservation are wrapped in a robust `try/finally` block.
   - If writing the final summary JSON fails, the error is propagated (never silently swallowed), both handles are safely closed, and the JSONL frame log is preserved on disk.
5. **Provenance Tracking & Real-Camera Safety**:
   - Each trial records `"provenance"` in both header line and summary metadata: `"live_camera_observation"` for physical camera runs, or `"synthetic_test_harness"` when synthetic fixtures are present.
   - Mixing test harness fixtures or detector overrides with real-camera acquisition (`camera_stream=None`) is strictly forbidden fail-closed. Real runs always execute the verified, loaded candidate bundle directly.

---

## 2. Guided Trial Workflow & Responsive HUD Layout

The tool guides the operator through a predeclared 12-stage protocol using a high-visibility, responsive Heads-Up Display (HUD) engineered to prevent text clipping across common resolutions (640x480, 320x240):

```
+-------------------------------------------------------------------------------+
| AIRPAD SHADOW TRIAL [OBSERVATION ONLY | 0 ACTIONS DISPATCHED]                 |
| [8] TWO_FINGERS_PARALLEL                                                      |
| Phase: OBSERVATION                                     Time Left: 4.2s        |
| STABLE OBSERVATION -- RECORDING ACTIVE                                        |
|                                                                               |
|                                [Camera Preview]                               |
|                                                                               |
| Hand: YES        | Pred: TWO_FINGERS_PARALLEL (m=0.45)   | Cmds: 0 (LOCKED)   |
+-------------------------------------------------------------------------------+
```

### Multi-Line Banner Architecture
- **Line 1 (y22)**: Header title and zero-action disclaimer.
- **Line 2 (y56)**: Intended stage name, dynamically scaled via `cv2.getTextSize` so long names like `[8] TWO_FINGERS_PARALLEL` never clip or overlap.
- **Line 3 (y84)**: Dedicated line for current phase (`PREPARATION` vs `OBSERVATION`) and countdown timer, separated vertically from the stage label.
- **Line 4 (y108)**: Instructional status banner.
- **Bottom Status Bar (h-40 to h)**: Hand detection indicator (`YES` / `NO`), model prediction with decision margin, and locked 0-command count.

### Stages Sequence (12 Blocks)
1. **NO_HAND (Initial Baseline)**: Rest hand out of camera frame.
2. **[1] PINCH_CLOSED**: Thumb-index tip contact pinch.
3. **[2] PINCH_MIDDLE**: Thumb-middle tip contact pinch.
4. **[3] L_SHAPE**: Thumb and index extended at right angle.
5. **[4] OTHER**: Natural neutral poses or non-target hand postures.
6. **[5] POINT_ONE**: Index finger extended, others curled.
7. **[6] OPEN_PALM**: All five fingers extended flat towards camera.
8. **[7] FIST**: All fingers curled tightly into fist.
9. **[8] TWO_FINGERS_PARALLEL**: Index and middle fingers extended parallel.
10. **[9] THREE_FINGERS**: Index, middle, and ring fingers extended.
11. **OTHER (Post-Gesture Catch-All)**: Hand movement, transitional poses, non-target postures.
12. **NO_HAND (Final Post-Trial Negative)**: Hand removed from camera frame.

### Stage Phases & Timing
- **WAITING Phase**: Camera preview runs while waiting for user readiness. No trial frames are scored. Press **[SPACE]** once to begin Stage 1.
- **PREPARATION Phase** ($\ge 3.0$ seconds default): Overlay displays amber background. Hand transitions into indicated pose. Logged with `is_transition = True` and **excluded** from scored stable metrics.
- **OBSERVATION Phase** ($\ge 5.0$ seconds default): Overlay turns green. Hand holds pose steady. Logged with `is_transition = False` and **included** in scored stable metrics.
- **Automatic Stage Progression**: Auto-advances across stages. No per-gesture hotkeys or manual save keystrokes required.
- **Immediate Exit**: Pressing **[Q]**, **[ESC]**, or clicking the window close button (`X`) triggers immediate safe exit marked `INCOMPLETE` with flushed logs.

---

## 3. Video Pre-Processing Pipeline Contract

To prevent color space and coordinate distortion:
1. Raw frame captured from OpenCV `cap.read()` in BGR format.
2. Frame flipped horizontally (`cv2.flip(frame, 1)`) for intuitive mirror preview.
3. Flipped frame explicitly converted to RGB (`cv2.cvtColor(flipped_bgr, cv2.COLOR_BGR2RGB)`).
4. MediaPipe `HandLandmarker` processes **only** the RGB frame.
5. Preview HUD is rendered on the BGR canvas for display.

---

## 4. Observability & Output Format

All trial outputs are saved by default in `reports/acc06/`:

### 1. Incremental JSONL Frame Log (`trial_<session_id>.jsonl`)
- **Line 0 Header Record**: Written and flushed before the first frame, recording `session_id`, `user_id`, `provenance`, model/detector SHA-256 hashes, frozen decision thresholds (`0.0` / `0.15`), and the complete 12-stage schedule.
- **Frame Records (Lines 1..N)**: Each frame logs `frame_index`, `timestamp`, `session_id`, `user_id`, `stage_id`, `intended_label`, `phase`, `is_transition`, `hand_detected`, `handedness`, raw landmarks, canonical landmarks, predictions, margin, and `commands_dispatched: 0`. Header is strictly excluded from frame indexing.

### 2. Summary Metrics JSON (`summary_<session_id>.json`)
Emitted on completion or early exit:
- **Metadata**: Provenance (`live_camera_observation` vs `synthetic_test_harness`), user ID, session ID, candidate bundle hash, detector model hash, trial status (`COMPLETED` vs `INCOMPLETE`), 0 commands confirmed.
- **NO_HAND Baseline Metrics**: False detections count/rate, false target action predictions count, other predictions count, clean no-hand rate.
- **Target Gesture Metrics (Full Denominator)**:
  - *Full Denominator Rule*: Detector losses (`hand_detected == False`) during target stages are strictly included in the denominator as errors.
  - Overall raw and rejection accuracy across target gestures.
  - Strict fixed 9-class Macro-F1 denominator (support-0 classes contribute 0.0).
  - Per-class recall table, with special emphasis on `PINCH_CLOSED` and `PINCH_MIDDLE`.
  - `OTHER`-to-target false action trigger counts.
- **Stages Breakdown**: Frame counts (total, prep, stable) and detection statistics per stage.

---

## 5. Usage & Execution Command

Launch command for live human operator observation:
```powershell
.venv\Scripts\python.exe tools/shadow_trial.py `
  --candidate-bundle reports/acc05/candidate_personal_svm.json `
  --task-model models/hand_landmarker.task `
  --user-id tuana `
  --prep-duration 3.0 `
  --obs-duration 5.0
```

### Available CLI Flags
- `--candidate-bundle PATH`: Path to frozen candidate bundle (default: `reports/acc05/candidate_personal_svm.json`).
- `--task-model PATH`: Path to local MediaPipe model (default: `models/hand_landmarker.task`).
- `--output-jsonl PATH`: Explicit path for JSONL frame output (default: auto under `reports/acc06/`).
- `--output-summary PATH`: Explicit path for JSON summary output (default: auto under `reports/acc06/`).
- `--user-id STRING`: Operator identifier (default: `tuana`).
- `--camera-id INT`: Camera device index (default: `0`).
- `--prep-duration FLOAT`: Preparation duration in seconds (default: `3.0`).
- `--obs-duration FLOAT`: Observation duration in seconds (default: `5.0`).
- `--headless`: Run without GUI preview window (for automated replay or testing).

---

## 6. Important Caveats & Scientific Limitations

1. **Declared Intended Poses vs Ground Truth**:
   - The stage labels represent *intended* poses declared by the guided protocol; they are not independently verified physical ground truth.
   - Metrics are observational indicators of personal consistency, not calibrated laboratory benchmark truth.
2. **Single-User One-Take Scope**:
   - Observations reflect performance on a single user in same-day lighting conditions. They do not demonstrate generalization to other individuals, varying hand morphology, or different environments.
3. **No Frame-i.i.d. Significance**:
   - Video frames within a held gesture burst are temporally correlated. Statistical significance claims (e.g. p-values) cannot be derived from consecutive camera frames.
4. **No Model Promotion**:
   - This tool is strictly observational. Production model `models/airpad_svm.json` remains untouched.
