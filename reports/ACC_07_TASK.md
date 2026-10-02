# ACC-07 — camera contract and existing runtime dry-run

Status: assigned to the existing sole Antigravity Worker through the previously approved Orca terminal fallback. Coordinator retains review authority; no concurrent editor, camera launch or production promotion.

+ACC-07: supervised bounded runtime dry-run readiness. You are the sole authorized Antigravity code Worker. Coordinator uses the previously user-approved Orca direct-terminal fallback; do NOT reuse old structured Dispatch IDs, spawn agents or emit fake worker_done.

Read reports/ACC_06_RETAKE_REVIEW.md and reports/ACC_06_COORDINATOR_REVIEW.md. Evidence: unchanged code passed 308 tests; frozen candidate achieved 814/905 (89.9448%) versus production 253/905 (27.9558%) on identical live target rows; numbered gestures 793/795. Static recognition improved, temporal action safety unproven. app.run_camera currently sends BGR to the RGB adapter, unlike fixed collector/shadow.

ONE TASK: prepare and validate existing runtime end-to-end dry-run using the frozen personal candidate, correcting camera color contract and bounded resource cleanup.

OWNERSHIP: edit only app.py, tests/test_app.py, reports/acc07/* and reports/ACC_07_WORKER_REPORT.md. Preserve initial app.py/test_app.py copies under reports/acc07 for no-index diff. Coordinator alone edits task/coordinator reports. No Git repository: no init/reset/checkout. Preserve all other edits. Do not edit src, tools, config, datasets, model bundles or thresholds; do not train/promote models or introduce architecture. If an existing src bug blocks safety, report evidence and stop for Coordinator scope decision.

IMPLEMENT:
1. Flip camera BGR frame, convert to separate RGB frame for detector; retain BGR preview. Add asymmetric color regression that demonstrably fails against original app code. Preserve API/default dry-run and explicit model argument.
2. Ensure detector.close, pipeline.cleanup, camera.release and window teardown on normal stop, read failure, detection exception and failed camera open, as applicable to initialized resources. Preserve original error; all cleanup attempted even if one cleanup raises. Bound refactoring to camera lifecycle.
3. Focused fake-camera tests: actual detector input RGB, preview BGR; model path forwarding; raw landmarks/handedness reach pipeline; safe cleanup paths; default DryRunController and no real WindowsController/OS injection. Use mocks/positive guards, not counters alone.
4. Test existing FSM with controlled synthetic dynamic sequence (click/drag/hand loss/reacquire) in dry-run, including held-button release on loss/cleanup. Do not claim static capture establishes dynamic intent.
5. Produce read-only end-to-end replay evidence using REAL frozen candidate through app.create_pipeline(allow_real_control=False) and RAW landmark arrays from reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl, original timestamps/order including missing detections. Do NOT feed canonical NPZ features through canonicalizer again. Capture raw predictions, filtered poses/FSM and DryRunController events, prove held inputs cleared on loss/cleanup and real OS APIs not invoked. Compare stable observation raw predictions against shadow log. No tuning, relabeling or excluding failed rows. Clearly distinguish action events from classification and explain unexpected events, if any. Minimal experiment driver only under reports/acc07; write new outputs without overwriting inputs.
6. Run focused tests, complete pytest suite (308 existing + additions), app.py --self-check. No physical camera launch and NO --real-control. Coordinator reviews before opening camera.

Pins: candidate SHA256 66091d52d9dd6848fd8e35fc2641a31c4c2ffc9008329a4d2e9f85f8dd1246be; production 4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977; retake log 2e2753def2d95bf9a33cc88c55eaba6f19560b7cd5df2815cc037baa3abc8cd2. Never overwrite these.

ACCEPTANCE: reproducible failing-before/passing-after RGB regression, all tests pass, paired predictions match frozen candidate, safety tests actually prohibit OS control, complete resource cleanup, explicit limitations/no promotion. Report exact commands/results, modified files, evidence paths/pins, risks and any blocker. End with ACC07_DIRECT_DONE and idle. This is the only active code task.
