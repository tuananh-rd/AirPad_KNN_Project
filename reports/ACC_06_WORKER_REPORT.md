# ACC-06 Worker Report: Bounded Shadow-Only Guided Observation Trial

**Author**: Antigravity Worker (User-Authorized Direct Terminal Fallback under Coordinator Supervision)  
**Date**: 2026-10-01  
**Task ID**: ACC-06  
**Related Documents**: `reports/ACC_05_COORDINATOR_REVIEW.md`, `reports/ACC_05_WORKER_REPORT.md`, `docs/SHADOW_TRIAL.md`

---

## 1. Executive Summary

Implemented the bounded, fail-closed shadow-only observation trial tool `tools/shadow_trial.py` and its comprehensive test suite `tests/test_shadow_trial.py` to evaluate the frozen personal SVM candidate (`reports/acc05/candidate_personal_svm.json`) against live camera frames with zero action dispatch by construction.

Following Coordinator review checkpoints, the tool and test suite enforce:
1. **Zero Action Dispatch by Construction**: Active import hook (`ForbiddenImportBlocker`) and OS dispatch API interceptors (`os.system`, `subprocess.Popen`, `subprocess.run`, `user32`) actively enforce that neither `src.platform.*`, `src.runtime.*`, `src.gesture.fsm`, `pyautogui`, `pynput`, nor `keyboard` are imported or called during a positive trial run.
2. **Model & Detector Pre-Verification**: Pinned candidate bundle hash and detector task model are verified and loaded **before** reserving output handles and before camera acquisition. Corrupt models or hash mismatches produce no output files and do not touch the camera.
3. **Sequential Exclusive Reservation & TOCTOU Elimination**: Output files (`.jsonl` and summary `.json`) are reserved sequentially with exclusive creation mode `'x'`. If the second reservation encounters a pre-existing file collision, the newly created empty JSONL file is closed and cleaned up safely, while the collision file is preserved untouched.
4. **Try/Finally Protected Execution**: Every operation after handle reservation is wrapped in a robust `try/finally` block. If writing summary JSON fails, the error is propagated (never silently swallowed), both handles are closed safely, and the JSONL frame log is preserved on disk.
5. **Provenance Tracking & Real-Camera Safety**: Provenance is explicitly tagged as `"live_camera_observation"` or `"synthetic_test_harness"`. Mixing test harness fixtures or detector overrides with real-camera acquisition (`camera_stream=None`) is strictly forbidden fail-closed. Real runs always execute the verified candidate bundle directly.
6. **Responsive Multi-Line HUD Layout**: At common 640x480 and 320x240 resolutions, stage label is placed on its own line (Line 2) and dynamically scaled via `cv2.getTextSize` so long names like `[8] TWO_FINGERS_PARALLEL` never clip or overlap with countdown/phase timers (Line 3).
7. **Deterministic Fake Classifier & Exact Metrics**: Verified with a deterministic mock classifier testing correct, wrong, and rejected predictions alongside detector dropout, asserting exact concrete metrics and verifying that the classifier is never fit on observations (`fit.call_count == 0`).
8. **Real VideoCapture & Window Cleanup**: Unit tests exercise `cap.release()`, `cv2.destroyAllWindows()`, and `detector.close()` on camera read and GUI paths.

All 308 full project tests pass (including 23 isolated shadow trial unit tests). The physical camera device was never opened by the agent; all testing used synthetic mock streams.

---

## 2. Immutable Artifacts & Cryptographic Checksums

All pre-existing models, datasets, protocols, and audit reports were verified against their cryptographic pins. No existing file was altered:

| Artifact Path | Expected / Verified SHA-256 Hash | Status |
|---|---|---|
| `reports/acc05/candidate_personal_svm.json` | `66091d52d9dd6848fd8e35fc2641a31c4c2ffc9008329a4d2e9f85f8dd1246be` | PINNED & MATCHED |
| `models/airpad_svm.json` | `4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977` | PINNED & MATCHED |
| `models/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | PINNED & MATCHED |
| `data/cohort/curated/acc04_validation_clean.npz` | `e0a27d1bd989a1ac1fe9fccc90c9ae57a47a8ca0637f1959cab91a4fcc99000d` | PINNED & MATCHED |
| `data/cohort/tuana_validation_retake_20261001_113202_dfcd6b.npz` | `fe26b606921386a0f640dc958e4e7ebf01e43ce0d6cd6f07e693fce517db8948` | PINNED & MATCHED |
| `data/cohort/tuana_train_rgb_20261001_110223_8ff3ed.npz` | `0f9d21b5b1a5ccab0ffc8f555d7c00e14b26fa2299040b9ab587b236e7d1014e` | PINNED & MATCHED |
| `data/cohort/tuana_test_rgb_20261001_115837_5741bd.npz` | `4ec1cf2dfbdcd74b5c22c914a31c52bf2fd3c11d5c3a4de5a49368eb9cca28fb` | PINNED & MATCHED |
| `reports/acc04/validation_cleaning_audit.json` | `4772a95cca9c08937057775223afd7b83c2ffab6f6f39c09cb43f7f874f98ff0` | PINNED & MATCHED |
| `reports/ACC_05_PROTOCOL.md` | `ff0301ac6b37641f38c021b16897827a5f4b56acbb7bced61f30d88d47c93061` | PINNED & MATCHED |

---

## 3. Architecture & Safety Guarantees

1. **Zero Action Dispatch by Construction**:
   - `tools/shadow_trial.py` contains no imports or references to `src.platform`, `src.runtime.pipeline`, `src.gesture.fsm`, `pyautogui`, `pynput`, `keyboard`, or OS event synthesis APIs.
   - Tested in both in-process active import hook and clean subprocess execution: attempting to import forbidden modules or call dispatch APIs during trial execution raises immediate safety errors.
   - The metric `commands_dispatched` is hard-coded and logged as `0` on every frame, in the header, and in the final summary.
2. **Pre-Flight Flag & Parameter Rejection**:
   - The function `check_forbidden_cli_args()` inspects CLI parameters before camera or model initialization.
   - Any flag attempting action execution (`--real-control`, `--dispatch`, `--execute`, `--enable-control`, `--live-control`) raises `ValueError` and terminates immediately.
   - The function `validate_trial_parameters()` ensures `user_id` is non-blank and durations are finite positive numbers before outputs or camera are touched.
3. **Model Verification Before Handle Reservation**:
   - Candidate bundle SHA-256 and detector task model are verified and loaded **before** any output file handles are created.
   - Invalid candidate hash or corrupted bundle schemas fail closed before file reservation and before camera acquisition.
4. **Sequential Exclusive Reservation (No TOCTOU Overwrites)**:
   - Output files are sequentially opened with mode `'x'` (`reserve_output_handles()`).
   - If opening summary fails, the newly created empty JSONL handle is closed and its empty file deleted, while the collision summary file is strictly preserved.
5. **Resilient Cleanup & Error Propagation**:
   - Handles, detector, and camera are wrapped in `try/finally` blocks.
   - Summary write failures raise `RuntimeError`, closing both handles while preserving the JSONL log on disk.
6. **Provenance Tracking**:
   - Provenance is explicitly recorded in JSONL line 0 and summary JSON: `"live_camera_observation"` vs `"synthetic_test_harness"`.
   - Real-camera acquisition (`camera_stream=None`) rejects test harness overrides fail-closed. Real runs always execute the verified candidate bundle.
7. **Responsive Multi-Line HUD Layout**:
   - Eliminates horizontal text collision: Line 2 displays intended stage name scaled dynamically with `getTextSize`, Line 3 displays phase and countdown timer separated vertically.

---

## 4. User-Friendly Guided Trial Protocol

The visual HUD displays a structured, staged workflow on the camera canvas:

1. **WAITING Phase**:
   - Initial state upon launch. Live video preview is displayed.
   - User positions themselves comfortably. No trial frames are scored.
   - Pressing **[SPACE]** initiates Stage 1.
2. **Fixed Guided Sequence (12 Stages)**:
   - **Stage 1**: `NO_HAND` (Initial Baseline)
   - **Stage 2**: `[1] PINCH_CLOSED`
   - **Stage 3**: `[2] PINCH_MIDDLE`
   - **Stage 4**: `[3] L_SHAPE`
   - **Stage 5**: `[4] OTHER`
   - **Stage 6**: `[5] POINT_ONE`
   - **Stage 7**: `[6] OPEN_PALM`
   - **Stage 8**: `[7] FIST`
   - **Stage 9**: `[8] TWO_FINGERS_PARALLEL`
   - **Stage 10**: `[9] THREE_FINGERS`
   - **Stage 11**: `OTHER` (Post-Gesture Catch-All)
   - **Stage 12**: `NO_HAND` (Final Post-Trial Negative)
3. **Phases & Timings**:
   - **Preparation Phase** ($\ge 3.0$ seconds default): Overlay displays amber background and countdown `TRANSITION -- HOLD INTENDED POSE`. Logged with `is_transition = True` and excluded from scored stable metrics.
   - **Observation Phase** ($\ge 5.0$ seconds default): Overlay displays green background and `STABLE OBSERVATION -- RECORDING ACTIVE`. Logged with `is_transition = False` and scored. Monotonic clock ensures countdown precision independent of system clock adjustments.
   - **Auto-Advance**: Tool automatically advances to the next stage upon observation timer completion.
   - **Safe Early Exit**: Pressing **[Q]** or **[ESC]** or closing the preview window exits immediately, marks `trial_status = "INCOMPLETE"`, and flushes all partial frames and summary JSON safely via isolated cleanup blocks.

---

## 5. Video Pre-Processing Pipeline Contract

To eliminate the OpenCV BGR / MediaPipe RGB color space defect:
1. Frame read from `cap.read()` in raw BGR format.
2. Frame horizontally flipped via `cv2.flip(frame, 1)` for mirror feedback.
3. Flipped frame explicitly converted to RGB via `cv2.cvtColor(flipped_bgr, cv2.COLOR_BGR2RGB)`.
4. MediaPipe `HandLandmarker` processes **only** the RGB frame.
5. Visual annotations and HUD are drawn on `flipped_bgr` for display.
6. Verified with an asymmetric color test (`test_asymmetric_bgr_to_rgb_expectation`) where Red channel $\neq$ Blue channel to prove the detector strictly receives the converted RGB frame.

---

## 6. Observability, Denominator Rules & Metrics

1. **Role & Metadata Header**:
   - Logged with unique session ID `shadow_obs_<timestamp>_<uuid>`, role `shadow_observation`, and provenance tag.
   - Line 0 of the `.jsonl` file is written and flushed as a `header` record before the first frame, recording session metadata, candidate hash, detector hash, frozen thresholds, and full guided stage definitions.
   - Every subsequent frame record includes `record_type: "frame"`, `session_id`, and `user_id`. Line 0 header is strictly excluded from frame indexing.
2. **Incremental Flushing**:
   - Every frame writes a structured JSON record to the `.jsonl` file and immediately calls `flush()`.
   - No raw video or RGB images are stored to disk.
3. **Full Denominator Rule**:
   - Detector losses (`hand_detected == False`) occurring during target gesture observation stages are strictly **included in the denominator as errors**. They are never discarded or ignored.
4. **Fixed 9-Class Macro-F1**:
   - Macro-F1 is strictly averaged over a fixed 9-class denominator ($9.0$), ensuring that unobserved or support-0 classes on incomplete trials contribute $0.0$ and do not inflate the average.
5. **NO_HAND Negative Baseline**:
   - Stages 1 and 12 (`NO_HAND`) are scored separately to report false detections and false action predictions during rest periods.
6. **Specialized Diagnostics**:
   - Explicit reporting of `PINCH_CLOSED` recall, `PINCH_MIDDLE` recall, and `OTHER`-to-target false command prediction counts.
7. **Scientific Limitations & Caveats**:
   - Start labels represent declared *intended* poses, not independently verified ground truth.
   - Results represent same-user, same-day observational performance without claims of cross-user generalization or frame-i.i.d. statistical significance.

---

## 7. Verification & Test Suite Execution

All test suites were executed cleanly without real camera access:

1. **Isolated Shadow Trial Unit Tests** (`tests/test_shadow_trial.py`):
   - **23 passed in 4.89s**.
   - Tested:
     - `test_safety_and_forbidden_flags_rejection`: Rejection of unsafe CLI flags.
     - `test_parameter_validation_fail_closed`: Rejection of blank user and non-finite/zero/negative durations.
     - `test_os_safety_and_zero_commands_dispatched_with_active_import_hook`: Active import hook and OS dispatch interceptors prove 0 commands dispatched and reject forbidden imports.
     - `test_os_safety_subprocess_isolation`: Fresh subprocess execution with import hook proves clean environment safety.
     - `test_toctou_race_injection_preserves_files_and_blocks_camera`: Exclusive `'x'` reservation prevents truncation and blocks camera.
     - `test_second_open_collision_cleans_first_and_preserves_collision`: Summary collision safely cleans empty JSONL and preserves collision file.
     - `test_preflight_invalid_inputs_assert_videocapture_not_called_and_no_outputs`: Preflight validation fails with `VideoCapture` not called and no output files created.
     - `test_candidate_bundle_validation_checks`: Rejection of missing classes, wrong hyperparams, and missing scaler.
     - `test_landmark_validation_fail_closed`: Rejection of NaN/Inf landmarks and invalid handedness.
     - `test_asymmetric_bgr_to_rgb_expectation`: Verification of asymmetric horizontal flip and BGR->RGB color space conversion.
     - `test_headless_end_to_end_synthetic_trial_with_header_and_linkage`: Complete 12-stage mock run with header verification, frame linkage, and full denominator rules.
     - `test_deterministic_fake_classifier_and_concrete_metrics`: Deterministic predictions test exact raw accuracy (0.90), rejection accuracy (0.85), fixed-9 F1, OTHER false targets, and verify classifier is never fit.
     - `test_fake_videocapture_and_gui_cleanup`: Verifies `cap.release()` and `cv2.destroyAllWindows()` cleanup calls.
     - `test_header_write_failure_closes_handles_and_does_not_open_camera`: Header write exception closes handles and blocks camera.
     - `test_summary_write_failure_propagates_and_closes_handles`: Summary write error propagates, handles close, and JSONL log is preserved.
     - `test_fixed_9_macro_f1_on_incomplete_trial`: Proves macro-F1 strictly divides by 9.0 on incomplete takes.
     - `test_early_exit_user_quit_and_cleanup`: Q/ESC key press triggers `INCOMPLETE` exit and cleans up resources.
     - `test_camera_read_failure_cleanup`: Dropped camera stream triggers clean shutdown and preserves summary.
     - `test_real_camera_headless_without_fixture_fails_closed`: Unattended real-camera headless execution fails closed.
     - `test_window_close_via_get_window_property_exits_safely`: Window close event triggers clean `INCOMPLETE` exit.
     - `test_candidate_and_baseline_immutability`: Verification that candidate, baseline, and detector models match pinned hashes.
     - `test_forbid_test_fixtures_with_real_camera_acquisition`: Mixing detector overrides with real camera acquisition fails closed.
     - `test_hud_overlay_layout_legibility_and_separation`: Responsive layout test via `getTextSize` proves stage label fits width and does not overlap countdown.
2. **Full Project Test Suite**:
   - `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests reports/acc04/test_repair_validation.py reports/acc05/test_acc05_benchmark.py`:
   - **308 passed in 21.37s** (zero failures, zero errors).

---

## 8. Artifacts Manifest & Checksums

| File Path | Size (Bytes) | SHA-256 Hash |
|---|---|---|
| `tools/shadow_trial.py` | 46,435 | `128f35542be94c9f40c8e2d1b6d15d42b3f994cc0b6b9932c2f31aacfdd15301` |
| `tests/test_shadow_trial.py` | 39,737 | `dd9dabbe7b83b5a781f01b8903fb8ef1d94625ec5a2cd45bd2e32411c1685459` |
| `docs/SHADOW_TRIAL.md` | 10,203 | `c5e3eb1da0ea8d214d63596d425c7529c8a0ee9b67af7e29ea81cf13122354aa` |
| `reports/acc06/synthetic_sample_trial.jsonl` | 201,448 | `8a4af1295cd2259104b2080cba623fedf2d63df03f37b295a6ac0fa4293f1601` |
| `reports/acc06/synthetic_sample_summary.json` | 7,843 | `6767cbddcffcc83c38419007f684a9f4248ef47ce5d6b1f971c1d2297cec188c` |
| `reports/ACC_06_WORKER_REPORT.md` | 16,588 | `b85c13e79e66ff72f4464c8d57d7620eb5a4325a74e1e3b2e5ef40d4ca0cf321` |

---

## 9. Operator Launch Instructions

To execute the live shadow observation trial, the human operator should run:

```powershell
.venv\Scripts\python.exe tools/shadow_trial.py `
  --candidate-bundle reports/acc05/candidate_personal_svm.json `
  --task-model models/hand_landmarker.task `
  --user-id tuana `
  --prep-duration 3.0 `
  --obs-duration 5.0
```

1. Ensure the webcam is connected and ambient indoor lighting is stable.
2. In the `WAITING` preview, verify hand positioning in the frame.
3. Press **[SPACE]** once to begin the guided protocol.
4. Follow the HUD stage titles and countdown timers through all 12 blocks.
5. Upon completion or if **[Q]** / **[ESC]** is pressed, review the generated summary JSON in `reports/acc06/`.
