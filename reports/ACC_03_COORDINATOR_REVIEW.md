# ACC-03 Coordinator review — 2026-10-01

Outcome: **accepted as a verified camera input-contract repair; user qualitatively confirms substantially more stable live hand detection after the repair.**

## Evidence

- Coordinator contract test before dispatch established that OpenCV's BGR frame reached `MediaPipeAdapter.detect` unchanged while the adapter documents RGB input and constructs `mp.Image` tagged `SRGB`.
- Antigravity (sole code editor) changed only the collector's live loop: it flips the BGR frame, creates `rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`, and passes that copy only to detection. BGR `frame` remains the rendering/HUD image.
- Diff review confirms that exact scoped change; no detector API, threshold, model, feature or metadata behavior changed.
- Added synthetic asymmetric-frame regression verifies spatial flip, exact RGB channels at the detector, and BGR preservation in a text-free rendered region.
- Coordinator independently ran focused suite: **56 passed in 5.17s**. Worker reports full suite: **261 passed in 104.64s**.
- After completion of the post-fix train recording, Coordinator independently reran `.venv/Scripts/python.exe -B -m pytest -q`: **261 passed in 49.00s**, exit code 0 (execution session 57297). This verifies regressions, not live detection or classification accuracy.
- Coordinator independently rechecked baseline hashes: custom/HAGRID/combined datasets, production model and ACC-01 evaluator all match prior values.

## Interpretation

The BGR/RGB mismatch is a real software defect. After the post-fix live diagnostic, the user reported "ổn định hơn rất nhiều rồi đó" (substantially more stable). This qualitative before/after observation supports the mismatch contributing to the observed instability, but is not a controlled detector benchmark, proof that every illumination issue is resolved, or evidence that gesture-classification accuracy exceeds baseline. Process inspection after feedback found no running collector. No new code task is warranted from this feedback alone. The next priority is resuming gesture 5 collection in a fresh post-fix train/development session when the user resumes recording; the pre-fix morning session remains on hold. Preserve earlier recordings and distinguish pre/post-fix provenance. Separate complete validation/test sessions are still required for an accuracy comparison.

Orca worker task `task_90b6735ee553` / dispatch `ctx_88794eec8d79` succeeded. Completion was acknowledged; release retained the pre-existing terminal without stopping it; no reclaimable worker remains.
