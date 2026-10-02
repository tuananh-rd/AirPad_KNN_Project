# ACC-07 Worker Report: Bounded Runtime Dry-Run Readiness

**Task ID**: ACC-07 (Supervised Bounded Runtime Dry-Run Readiness)  
**Execution Context**: Direct-terminal supervised fallback under Brain/Coordinator supervision  
**Worker**: Antigravity  
**Date**: 2026-10-01  
**Target Artifacts**: `app.py`, `tests/test_app.py`, `reports/acc07/*`, `reports/ACC_07_WORKER_REPORT.md`  
**Status**: COMPLETED  

---

## 1. Executive Summary

Under direct-terminal supervised fallback authorization, this task prepared and validated the existing runtime pipeline for dry-run execution using the frozen personal SVM candidate bundle (`reports/acc05/candidate_personal_svm.json`).

The scope comprised four critical operational milestones:
1. **Camera Color Contract Defect Correction**: Fixed the BGR-to-RGB contract mismatch in `app.run_camera`. OpenCV frame capture yields BGR images, while `MediaPipeAdapter.detect()` requires RGB images. We implemented an explicit `cv2.cvtColor(flipped_bgr, cv2.COLOR_BGR2RGB)` conversion exclusively for the detector while preserving `flipped_bgr` for on-screen HUD drawing and `cv2.imshow` presentation. An isolated module test reproduces the defect side-by-side on `reports/acc07/app.py.initial` without temporary file overwrites.
2. **Resilient Bounded Resource Cleanup with Priority Exception Propagation**: Restructured `app.run_camera` so that all initialized resources (`detector`, `pipeline`, `cap`, and OpenCV window handles) are cleanly released via independent, fault-tolerant cleanup steps in a `finally:` block across all normal and abnormal exit paths. Cleanup failures on otherwise successful runs are propagated as exceptions rather than returning success (exit code 0); active primary exceptions from the loop, detector hardware faults, or initialization are strictly preserved and not masked by secondary cleanup faults. Mid-stream read failure (EOF) exits cleanly with code 0 and executes full cleanup.
3. **Dry-Run Physical Retake Replay with Strict No-Overwrite Invariants**: Enhanced `reports/acc07/replay_acc06_dryrun.py` to replay all 3,320 frames of raw landmarks from the audited ACC-06 physical shadow take (`reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl`) through `create_pipeline(allow_real_control=False)` with the frozen candidate bundle. The run achieved 100% (3,320/3,320) prediction agreement with the shadow observation log, verified zero held inputs on every single `NO_HAND` frame, eliminated accelerated-replay speed dependence by binding the RateLimiter to recorded capture timestamps, installed active positive OS dispatch interceptors, removed all deletion flags (no `--force` or file removal), and wrote results exclusively (`open(..., 'x')`) to `reports/acc07/dryrun_replay_events_v2.json`.
4. **Enhanced Dynamic Gating and Fast Isolated Contract Verification**: Updated `test_synthetic_dynamic_fsm_sequence_in_dry_run` to assert exactly one `LEFT_CLICK`, exactly one `LEFT_DOWN`, exactly one `LEFT_UP`, and verified that the immediate first reacquisition frame permits zero cursor movement events and zero held inputs while remaining `PAUSED`. Isolated contract and cleanup tests utilize stub pipelines to avoid repeatedly loading heavy production models, ensuring sub-second test execution.

Coordinator final independent verification confirmed:
- Full test suite: **327 passed in 92.32s** (execution74778, exit 0).
- Internal self-check: `python app.py --self-check` verified **5/5 stages** (exit 0).
- Real candidate replay: matches every single 3,320 frame, timestamp, and prediction, and all 156 events in the v2 artifact (`dryrun_replay_events_v2.json`), with 1,445 no-detection frames strictly `PAUSED`, zero events, and zero held inputs.
- Protected 81-snapshot: only `app.py` and `tests/test_app.py` modified in codebase root; zero missing originals.

---

## 2. Actual Modified and Created Files

### 2.1. Codebase Files Modified
* [`app.py`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/app.py): Runtime entrypoint updated with BGR-to-RGB conversion for detector, resilient cleanup with exception priority, and fail-closed `--model` path validation.
* [`tests/test_app.py`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/tests/test_app.py): Test suite expanded to 18 unit tests covering color contract, cleanup exception propagation, detector exceptions, read EOF, candidate bundle forwarding, and dynamic FSM gating.

### 2.2. Task Artifacts Created in `reports/acc07/`
* [`reports/acc07/app.py.initial`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/acc07/app.py.initial): Unmodified pre-task snapshot of `app.py`.
* [`reports/acc07/app.py.diff`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/acc07/app.py.diff): Unified diff of changes to `app.py`.
* [`reports/acc07/test_app.py.initial`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/acc07/test_app.py.initial): Unmodified pre-task snapshot of `tests/test_app.py`.
* [`reports/acc07/test_app.py.diff`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/acc07/test_app.py.diff): Unified diff of changes to `tests/test_app.py`.
* [`reports/acc07/replay_acc06_dryrun.py`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/acc07/replay_acc06_dryrun.py): Landmark replay driver with positive OS guards, timestamp-driven rate limiting, and strict exclusive output creation.
* [`reports/acc07/dryrun_replay_events.json`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/acc07/dryrun_replay_events.json): Earlier dry-run replay event telemetry artifact (overwritten during initial iterations before deletion branch was removed).
* [`reports/acc07/dryrun_replay_events_v2.json`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/acc07/dryrun_replay_events_v2.json): Final verified replay artifact with recorded real guard targets, generated exclusively and preserved separately.
* [`reports/acc07/test_acc07_dryrun.py`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/acc07/test_acc07_dryrun.py): 8 audit and invariant unit tests.

### 2.3. Task Report
* [`reports/ACC_07_WORKER_REPORT.md`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/reports/ACC_07_WORKER_REPORT.md): This report.

---

## 3. Cryptographic Fingerprints and Provenance

| Artifact | Role | SHA256 Checksum |
|:---|:---|:---|
| `reports/acc05/candidate_personal_svm.json` | Frozen Candidate Model Bundle | `66091d52d9dd6848fd8e35fc2641a31c4c2ffc9008329a4d2e9f85f8dd1246be` |
| `models/airpad_svm.json` | Unchanged Production Model | `4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977` |
| `models/hand_landmarker.task` | MediaPipe Hand Landmarker Task | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl` | Audited ACC-06 Retake Log | `2e2753def2d95bf9a33cc88c55eaba6f19560b7cd5df2815cc037baa3abc8cd2` |
| `reports/acc07/app.py.initial` | Pre-modification snapshot of `app.py` | `ea6e46ede12798e6162924e66acceef9a8b693c5c44aea4849a00db953a25144` |
| `app.py` | Modified runtime entrypoint | `687ad66a16907cc580709a53608bb292ac522d2a603f4cb56a4860ea3fd24b34` |
| `reports/acc07/app.py.diff` | Unified diff of `app.py` | `96d038bebe44fb75f0c78a81fdb18d27acf621e18bcf12e3c3f68f9f04b02637` |
| `reports/acc07/test_app.py.initial` | Pre-modification snapshot of `tests/test_app.py` | `950f0fcbf3e63aa728e85f4c77e4e0e3cfcddadeb8a8956266a15c8803860980` |
| `tests/test_app.py` | Updated CLI & camera unit tests (18 tests) | `12c56ebb5d7a5a7c028cfc4a51e0f348df761e1a3377d08cd8b34abf593e91fe` |
| `reports/acc07/test_app.py.diff` | Unified diff of `tests/test_app.py` | `dc7941eb07560ff07ac9f45a9f5eff6b64f10c1644e3484993538c0a6e37d213` |
| `reports/acc07/replay_acc06_dryrun.py` | Landmark dry-run replay script | `32ff970bcb1b9a54ca41f351c8f86618598e3443e42847ac9c2af948a885ad8d` |
| `reports/acc07/dryrun_replay_events_v2.json` | Audited replay event trace & telemetry | `6bf9abd3e8ca96b8bdcbd928e97a11a5b997e9c99072158a95a1aa97b03c3646` |
| `reports/acc07/test_acc07_dryrun.py` | Replay artifact invariant & safety tests (8 tests) | `4b1d5463b9154eac83b0e38f6924246636407d16cbcac8353dae3b63d697e6b2` |

---

## 4. Implementation Details

### 4.1. Camera Color Contract Correction (`app.py`)
In OpenCV, `cap.read()` yields an 8-bit BGR image array. `cv2.flip(frame, 1)` mirrors the frame horizontally for natural interaction, remaining in BGR format. Prior code passed this flipped BGR frame directly to `detector.detect(frame)`. Because `MediaPipeAdapter` expects RGB frames, the red and blue channels were inverted during hand tracking, impacting landmark detection accuracy.

In the corrected implementation:
```python
# Mirror view for natural interaction
flipped_bgr = cv2.flip(frame, 1)
h, w, _ = flipped_bgr.shape
timestamp_ms = time.time() * 1000.0

# Hand detection requires RGB, but OpenCV provides BGR
rgb_frame = cv2.cvtColor(flipped_bgr, cv2.COLOR_BGR2RGB)
detected_hands = detector.detect(rgb_frame)
```
All visual landmark rendering (`cv2.line`, `cv2.circle`), HUD overlays (`cv2.putText`), and active ROI rectangles (`cv2.rectangle`) are drawn on `flipped_bgr`, and `cv2.imshow("AirPad Virtual Pad", flipped_bgr)` displays this BGR buffer. This ensures accurate colors on screen and accurate channel ordering for MediaPipe.

### 4.2. Resilient Resource Management and Exception Priority (`app.py`)
`run_camera` manages resources with guaranteed cleanup attempts across all initialized components and strict exception priority:
```python
primary_exc: Optional[BaseException] = None
cleanup_exceptions: List[Exception] = []

try:
    pipeline = create_pipeline(config, allow_real_control=allow_real_control, model_bundle=bundle)
    detector = MediaPipeAdapter(static_image_mode=False, max_num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam device index 0.")
        return 1

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        ...
    return 0
except BaseException as e:
    primary_exc = e
    raise
finally:
    if detector is not None and hasattr(detector, "close"):
        try:
            detector.close()
        except Exception as e:
            logger.warning(f"Error during detector cleanup: {e}")
            cleanup_exceptions.append(e)
    if pipeline is not None and hasattr(pipeline, "cleanup"):
        try:
            pipeline.cleanup()
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")
            cleanup_exceptions.append(e)
    if cap is not None and hasattr(cap, "release"):
        try:
            cap.release()
        except Exception as e:
            logger.warning(f"Error releasing camera: {e}")
            cleanup_exceptions.append(e)
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        logger.warning(f"Error destroying windows: {e}")
        cleanup_exceptions.append(e)

    # If no primary exception was active, propagate the first cleanup failure
    if primary_exc is None and cleanup_exceptions:
        raise cleanup_exceptions[0]
```
Key properties:
- **Comprehensive attempts**: Every cleanup function (`detector.close`, `pipeline.cleanup`, `cap.release`, `cv2.destroyAllWindows`) is attempted even if earlier ones raise.
- **No false success**: If the acquisition loop finished normally but `pipeline.cleanup()` failed (e.g. failing to release a held mouse button), the cleanup exception is raised rather than returning exit code 0.
- **Preserved primary exception**: If an exception occurred inside the frame loop, detector, or initialization, that primary exception propagates out unimpeded; secondary cleanup faults do not mask the root cause.
- **Mid-stream Read Failure**: If `cap.read()` fails mid-stream (`ret=False`), the loop breaks gracefully and exits with code 0 while executing all cleanups.

### 4.3. Explicit Model Path Validation (`app.py`)
`run_camera` validates that if an explicit `model_path` is provided, it must exist on disk; if not found, it reports an error and returns exit code 1 fail-closed, rather than silently falling back to a missing bundle.

---

## 5. Test Verification and Evidence

### 5.1. Asymmetric Color Regression Test Side-by-Side Reproduction
Two dedicated regression tests in `tests/test_app.py` verify the color contract:
1. `test_run_camera_asymmetric_color_contract_rgb_to_detector_bgr_to_preview`:
   - Uses an asymmetric synthetic frame `[B=30, G=120, R=210]` where $R \neq B$.
   - Asserts `fake_detector.frames_received[0]` receives channel 0 as 210 (Red) and channel 2 as 30 (Blue) [RGB].
   - Asserts `cv2.imshow` receives channel 0 as 30 (Blue) and channel 2 as 210 (Red) [BGR].
2. `test_asymmetric_color_regression_reproduced_on_saved_initial_app`:
   - Loads the unmodified `reports/acc07/app.py.initial` via `importlib.machinery.SourceFileLoader` in an isolated namespace without overwriting `app.py`.
   - Passes the identical asymmetric frame `[B=30, G=120, R=210]` to `app_initial.run_camera()`.
   - Asserts that `app_initial` passed channel 0 as 30 (Blue) and channel 2 as 210 (Red) to the detector, **proving the defect on the original code**.
   - Asserts that fixed `app.run_camera()` passes channel 0 as 210 (Red) and channel 2 as 30 (Blue), **proving the fix**.

### 5.2. Cleanup and Exception Propagation Tests
1. `test_run_camera_cleanup_failure_on_success_path_raises`:
   - Simulates normal loop termination, but `pipeline.cleanup()` raises `RuntimeError("Pipeline held-input cleanup failed")`.
   - Asserts that `run_camera` raises `RuntimeError`, preventing false success exit.
   - Asserts that `detector.close()`, `cap.release()`, and `cv2.destroyAllWindows()` were all still called.
2. `test_run_camera_combined_original_and_cleanup_failures_preserves_original_exception`:
   - Loop raises `ValueError("Primary frame loop display fault")`.
   - `detector.close()` and `pipeline.cleanup()` also raise secondary runtime errors.
   - Asserts that `run_camera` raises the original `ValueError("Primary frame loop display fault")`.
   - Asserts that `cap.release()` and `cv2.destroyAllWindows()` were still executed.
3. `test_run_camera_detector_exception_propagates_and_cleans_up`:
   - `detector.detect()` raises `RuntimeError("MediaPipe detection hardware fault")`.
   - Asserts exception propagates and `detector.close()`, `pipeline.cleanup()`, `cap.release()`, and `cv2.destroyAllWindows()` are executed.
4. `test_run_camera_read_failure_exits_cleanly_with_code_0`:
   - First frame is valid, second frame returns `(False, None)`.
   - Asserts clean return 0 and all cleanup functions executed.

### 5.3. Model Bundle Forwarding Spies
`test_run_camera_model_path_forwarding_and_fail_closed`:
- Uses spies wrapping `tools.train.load_bundle` and `app.create_pipeline`.
- Asserts `load_bundle` was called with the exact candidate path `"reports/acc05/candidate_personal_svm.json"`.
- Asserts `create_pipeline` received the loaded bundle with hyperparameters $C=10.0$ and $\gamma=0.01$.
- Asserts nonexistent model paths return exit code 1 fail-closed, with `create_pipeline` and `cv2.VideoCapture` never invoked.

### 5.4. Full Dynamic FSM Sequence in Dry Run
`test_synthetic_dynamic_fsm_sequence_in_dry_run`:
- Uses the **real default configuration** from `load_config()` (no hacked thresholds).
- Feeds controlled landmark sequences with regular 33.3ms timestamps across temporal stages:
  1. **Entry into `MOVING`**: steady `POINT_ONE` sequence satisfies 5-frame MajorityFilter (3/5 votes), 2-frame StabilityGate, and $100\text{ms}$ dwell (`point_one_stable_ms`), granting cursor movement (`can_move_cursor` is True).
  2. **Click Gesture**: short pinch ($80\text{ms} \le \Delta t \le 300\text{ms}$) released to `POINT_ONE`, returning to `MOVING` and emitting **exactly one** `LEFT_CLICK`.
  3. **Drag Initiation & Release**: sustained pinch hold ($> 400\text{ms}$ `drag_threshold_ms`) enters `DRAGGING`, emits **exactly one** `LEFT_DOWN` and adds `'left'` to `ctrl.held_buttons`. Subsequent release to `POINT_ONE` exits `DRAGGING`, emits **exactly one** `LEFT_UP`, and clears `'left'` from `ctrl.held_buttons`.
  4. **Hand Loss**: hand disappears (`landmarks=None`), immediately resetting to `PAUSED` and clearing all held inputs.
  5. **Conservative Reacquisition Gating**: on the immediate first frame of hand returning with `POINT_ONE`, the state remains `PAUSED`, cursor movement remains disabled (`can_move_cursor` is False), no movement events are emitted, and held buttons/keys remain empty. Only after subsequent frames satisfy temporal filter settling and $100\text{ms}$ dwell does FSM return to `MOVING` and re-enable cursor movement (`can_move_cursor` is True).
  6. **Final Invariant Audit**: asserts across the entire test that `LEFT_CLICK == 1`, `LEFT_DOWN == 1`, `LEFT_UP == 1`, and `pipeline.cleanup()` confirms clean held sets.

### 5.5. ACC-07 Replay and Safety Audit Suite (`reports/acc07/test_acc07_dryrun.py`)
All 8 unit and audit tests verify core replay invariants:
1. `test_acc07_file_hashes_and_pins`: asserts candidate bundle and shadow log match exact cryptographic SHA-256 pins.
2. `test_acc07_replay_events_json_audit`: verifies generated replay artifact (`dryrun_replay_events_v2.json`) has 3,320 replayed frames, 100% agreement, 0 real OS events, recorded real guard targets, and clean cleanup sets.
3. `test_acc07_replay_collision_preserves_existing_file_and_raises`: verifies exclusive creation raises `FileExistsError` on existing output file and preserves existing file bytes without deletion or truncation.
4. `test_acc07_replay_anti_aliasing_fail_closed`: verifies `ValueError` if output path aliases candidate bundle or observation log.
5. `test_acc07_replay_fail_closed_on_tampered_hash`: verifies failure on modified bundle without creating output.
6. `test_acc07_active_positive_os_guards_intercept_forbidden_dispatch`: forced call control verifying positive OS guards intercept real `WindowsController` calls and raise `AssertionError`.
7. `test_acc07_active_positive_os_guards_intercept_pyautogui_dispatch`: forced call control verifying positive OS guards intercept `pyautogui.scroll` and raise `AssertionError`.
8. `test_acc07_rate_limiter_bound_to_timestamps`: verifies `RateLimiter` enforces intervals using provided timestamps via `RateLimiter(max_events_per_second=30)`.

### 5.6. Comprehensive Test Suite Results
Command:
```
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests reports/acc04/test_repair_validation.py reports/acc05/test_acc05_benchmark.py reports/acc07/test_acc07_dryrun.py
```
**Results**:
- Coordinator Execution (execution74778): **327 passed in 92.32s** (exit code 0).
- Worker Execution (task-1932): **327 passed in 117.52s** (exit code 0).
- Breakdown: 310 in `tests/` (including 18 in `tests/test_app.py`) + 4 in `reports/acc04/` + 5 in `reports/acc05/` + 8 in `reports/acc07/`.

Internal Self-Check:
```
.venv/Scripts/python.exe -B app.py --self-check
All self-checks PASSED. AirPad RBF-SVM core is operational. (5/5 stages verified)
```

---

## 6. ACC-06 Physical Retake Dry-Run Replay Audit

The replay driver `reports/acc07/replay_acc06_dryrun.py` was executed against the audited physical retake log `reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl` using the frozen candidate bundle `reports/acc05/candidate_personal_svm.json`.

Full results were recorded exclusively (`open(..., 'x')`) to `reports/acc07/dryrun_replay_events_v2.json`.

### 6.1. Replay Safety Architecture
1. **Positive OS Guards**: Interceptors were installed on `src.platform.windows.WindowsController` and `pyautogui` (covering `moveTo`, `move`, `click`, `mouseDown`, `mouseUp`, `keyDown`, `keyUp`, `press`, `scroll`, `hscroll`, and `hotkey`). Any attempt to dispatch real OS commands raises an immediate assertion error. The summary records the exact list of targets actually patched. Recorded attempts: **0**.
2. **RateLimiter Capture-Clock Binding**: Because replaying 3,320 frames in memory executes at $>1,000\text{ fps}$, tying `RateLimiter` to `time.time()` would drop events that would not be dropped at real $30\text{ fps}$ acquisition. The replay driver bound the RateLimiter to the frame's recorded capture timestamp without modifying production code.
3. **Per-Row Accounting**: All 3,320 rows record `timestamp`, `held_buttons`, `held_keys`, `raw_pose`, `filtered_pose`, `shadow_rejected_prediction`, and `matches_shadow_rejected`.
4. **Continuous NO_HAND Invariant**: On every frame where hand detection was absent or the intended pose was `NO_HAND`, the replay asserted `len(held_buttons) == 0` and `len(held_keys) == 0`. Violations: **0**.
5. **No Force/Overwrite Branch**: Removed all `--force` / `-f` branches; the script strictly refuses to delete or overwrite existing output files, failing closed with `FileExistsError` to preserve prior artifacts.

### 6.2. Replay Summary Statistics (`dryrun_replay_events_v2.json`):
- **Total Frames Replayed**: 3,320
- **Elapsed Time**: 3.07 seconds (1,080.5 fps)
- **Hand Detected Frames**: 1,875 (56.48%)
- **Hand Absent Frames**: 1,445 (43.52%) — all strictly `PAUSED`, zero events, zero held inputs
- **Real OS Commands Dispatched**: **0**
- **Dry-Run Platform Events**:
  - `MOVE`: 151
  - `SCROLL`: 5
  - **Total Events**: 156
- **Prediction Agreement with Shadow Retake Log**: **3,320 / 3,320 (100.00%)**
- **FSM State Distribution**:
  - `PAUSED`: 3,159 frames (95.15%)
  - `MOVING`: 153 frames (4.61%)
  - `SCROLLING`: 8 frames (0.24%)
  - `DRAGGING` / `CLICK_PENDING`: 0 frames
- **Safety Invariant Audit**:
  - `held_buttons_before_cleanup`: `[]` (empty)
  - `held_buttons_after_cleanup`: `[]` (empty)
  - `held_keys_before_cleanup`: `[]` (empty)
  - `held_keys_after_cleanup`: `[]` (empty)
  - `no_hand_held_input_violations`: 0
  - `real_controller_invoked`: `False`
  - `zero_os_control_verified`: `True`

All 156 emitted events occurred strictly on the in-memory event log of `DryRunController`. Zero OS mouse or keyboard drivers were engaged.

### 6.3. Scientific Limitation and Protocol Disclosure
This trial replayed recorded landmarks from a guided static pose observation session (`trial_shadow_obs_20261001_162213_fb6d49.jsonl`). While replay verifies pipeline processing integrity, temporal filter convergence, and zero accidental OS actions under static observation, **frame records from static observation do not independently establish dynamic user intent or dynamic gesture safety**. Interactive physical dynamics (such as intentional drag-and-drop, pointing velocity, and user-initiated gesture timing) remain unproven and require dedicated dynamic user evaluation sessions.

---

## 7. Safety Invariants and Operational Limitations

1. **Hardware Isolation**: No physical webcam was opened or queried by the agent during this task. Mocks and synthetic streams were used for unit tests, and recorded physical landmarks were replayed for offline pipeline verification.
2. **Model Promotion Frozen**: The candidate bundle remains located strictly at `reports/acc05/candidate_personal_svm.json`. The production model `models/airpad_svm.json` remains untouched.
3. **No Retraining or Tuning**: No observation frames were appended to training datasets, and no hyperparameters or thresholds were adjusted.
4. **Dry-Run Enforcement**: All replay and test execution operated under `allow_real_control=False`.
5. **No Version Control Mutation**: No Git repository operations (`git init`, `git checkout`, `git reset`) were performed.

---

## 8. Process Deviation Disclosure

During the execution of task ACC-07:
* **Intermediate JSON Execution-Artifact Overwrite Deviation**: Early versions of the replay script wrote `reports/acc07/dryrun_replay_events.json` using `open(..., "w")`, and intermediate iterations incorporated pre-deletion code (`if os.path.exists: os.remove`) and were executed via `python replay_acc06_dryrun.py --force`. Consequently, earlier intermediate execution JSON outputs were overwritten in-place and are not recoverable as independent archived benchmarks.
* **Separately Preserved Final Artifact**: To resolve this, the final replay driver execution introduced a distinct output file, `reports/acc07/dryrun_replay_events_v2.json`, which was generated exclusively (`open(..., "x")`) and preserved independently without modifying or deleting earlier files.
* **Strict Upstream and Model Preservation**: All raw source observations (`reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl`), frozen candidate models (`reports/acc05/candidate_personal_svm.json`), production models (`models/airpad_svm.json`), and baseline snapshots (`reports/acc07/app.py.initial`, `reports/acc07/test_app.py.initial`) were strictly preserved and never mutated or deleted.
* **Permanent Deletion Branch Removal**: The `--force`/`-f` flag and any file deletion logic were completely removed from `replay_acc06_dryrun.py`, enforcing true non-destructive fail-closed behavior (`FileExistsError`) on existing files.


---

## 9. Conclusion

ACC-07 is complete. The runtime entrypoint `app.py` has resolved the camera color contract defect and established resilient cleanup with strict exception priority. The personal candidate model has been verified end-to-end through the existing FSM and motion pipeline in dry-run mode with 3,320 physical observation frames, confirming pipeline processing integrity, 100% prediction agreement with the audited shadow retake, and zero unintended OS actions under static observation replay.
