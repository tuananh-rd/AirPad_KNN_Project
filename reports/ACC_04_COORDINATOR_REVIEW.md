# ACC-04 independent Coordinator review — 2026-10-01

Outcome: accepted as bounded validation-data repair, not an accuracy improvement. Sole code author: Antigravity. Coordinator reviewed scripts/artifacts and ran independent tests; Coordinator edits only Markdown reports.

## Delivery and ownership

Structured task `task_040f287b68e6` failed readiness before injection in three attempts (`ctx_8648f6e12396`, `ctx_aa58f34c82c4`, `ctx_bf9799e782a1`). User explicitly authorized direct Orca terminal fallback. One direct task delivered to existing Antigravity terminal, request `3f2407bc-92ee-40eb-8303-9992e0825f38`; Worker replied `ACC04_DIRECT_DONE` with actual report/files. This is terminal-evidenced completion, not a fabricated successful Dispatch. Existing terminal remains user-owned; no duplicate editor, process stop or new Run. Reclaimable-worker list empty.

Workspace has no Git repository. No Git diff is available or Git initialized. Only new task-scoped script/tests/artifacts were introduced; final sources read directly, production code unchanged by this task. The intermediate script allowed overwrite/hash bypass and did not protect audit paths or NumPy suffix expansion. Coordinator required two same-task corrections; final script removes bypass/overwrite options and guards both requested and effective filenames, output/audit aliases and existing targets before writes.

## Data evidence

- Raw validation: 1713 rows, hash `9758a42469e9bc3e07fa7ee18993189cdf47938a3fd46262a943553b5a0fe01e`, preserved unchanged.
- Exclusions: all 359 uncertain L_SHAPE rows (not proven individually wrong), plus source rows 1477–1499 inclusive, 23 user-reported three-finger samples labelled TWO_FINGERS_PARALLEL. No guessed relabeling.
- Clean validation: 1331 rows, `data/cohort/curated/acc04_validation_clean.npz`, hash `e0a27d1bd989a1ac1fe9fccc90c9ae57a47a8ca0637f1959cab91a4fcc99000d`.
- Separate unchanged retake: 415 rows, L_SHAPE 209 / OTHER 206, hash `fe26b606921386a0f640dc958e4e7ebf01e43ce0d6cd6f07e693fce517db8948`.
- Validation total 1746: PINCH_CLOSED 173, PINCH_MIDDLE 170, L_SHAPE 209, OTHER 206, POINT_ONE 193, OPEN_PALM 185, FIST 195, TWO_FINGERS_PARALLEL 202, THREE_FINGERS 213.
- Coordinator independently verified complete retained/excluded source-row partition, order, feature bytes, labels, full metadata, pinned source hashes and retake provenance. Both actual session identities preserved and assigned validation; raw unclean file excluded from readiness inputs.
- Train 1692-row file retains hash `0f9d21b5b1a5ccab0ffc8f555d7c00e14b26fa2299040b9ab587b236e7d1014e`. Original custom/HaGRID/combined/model/evaluator hashes independently match prior baseline records.

Worker regenerated its initial derived NPZ/audit during correction, removing only its own intermediate outputs. No raw recording deleted; accepted outputs are reproducible from preserved sources. Scripts are bounded to pinned inputs, not general dataset cleanup tools.

## Independent verification

- `python -B -m pytest -q -p no:cacheprovider`: 261 passed in 42.43s.
- `python -B -m pytest -q -p no:cacheprovider reports/acc04/test_repair_validation.py tests/test_cohort_sessions.py tests/test_tools.py`: final 102 passed in 7.45s.
- `python -B -m pytest -q -p no:cacheprovider tests reports/acc04/test_repair_validation.py`: final 272 passed in 36.60s, exit 0 (execution session 47711). Includes all 261 project tests plus 11 bounded repair tests; default testpaths do not include reports tests automatically.
- Existing readiness function independently rerun in memory with explicit train + clean validation + retake and plan: NOT_READY, sole missing criterion is test sessions for tuana. No READY manifest emitted.

Worker report's combined-focused result 101 is an intermediate 10-repair-test result, not final 11-test total. Final authoritative result is Coordinator's 102; do not add stale totals. A Coordinator diagnostic initially failed shell quoting, made no changes, then succeeded with corrected quoting. Final script hash `9dd84e977f4f8e920a8c848539cc924fbf3945bb1e0fc6ae7e4a0d011953274b`; tests hash `7079f02b591d649eb9e8cdc145243347ae242205ce10445b77f9670fe645d3b8`.

## Remaining risks and exactly one next priority

No classifier candidate trained or evaluated: no comparable accuracy/F1 improvement established. Frame correlation, manual physical-label correctness and same-day/lighting dependence remain; schema/row counts do not measure detection failures or generalization across friends or lighting. The two validation takes are one evaluation role, not two independent held-out roles. Script guards are preflight, not a concurrent-writer/transactional I/O system; do not reuse it as a production ingestion architecture.

Next priority: obtain one complete genuinely new nine-label test take, role fixed before collection/performance review. Keep it out of training/tuning; preserve all raw metadata. No further code task or architecture change warranted before this human collection dependency is met.
