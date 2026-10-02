# ACC-05 independent Brain/Coordinator review — 2026-10-01

## Status and authority

ACC-05 CLOSED: accepted as a descriptive experiment with an explicitly documented protocol-order deviation, NOT a pristine confirmatory execution or production promotion. The fixed personal candidate exceeds the unchanged production baseline on the same personal test take. Worker recovered from server connection errors, corrected its reports, emitted ACC05_DIRECT_DONE and returned idle; Coordinator inspected the corrected report and chronology. The original baseline-exceeding loop has reached its observed numerical stopping condition; no additional code task was launched.

Only Antigravity edited experiment code, under reports/acc05. Coordinator performed read-only checks/tests and Markdown reporting. One task/editor throughout. ACC-05 used the user-authorized Orca terminal fallback, not a successful structured Dispatch; initial request cbbb1a1a-4fcd-40a7-b7a7-13bdc79fca48. Structured ACC-04 readiness failures remain failures. This folder is not a Git repository: no honest git diff is available, no repository was initialized. New script/test sources were reviewed directly; pinned hashes protect existing inputs and baseline.

## Fixed experiment and independent evidence

Protocol SHA256 ff0301ac6b37641f38c021b16897827a5f4b56acbb7bced61f30d88d47c93061 was frozen before viewing predictions. One RBF SVM, C=10, gamma=0.01, class_weight=balanced; scaler fitted on 1692 train rows only. Validation 1746, test 1663. Both models use decision=0 and margin=0.15. No architecture change, parameter search, threshold tuning, train+validation refit or promotion.

Coordinator independently verified:

- Explicit role readiness READY, 5101 stable samples, four real session identities. Manifest fingerprint 3c3ca26a9bda2e46fc698512ee78c35a7a47937e04383b87642626acc676e1fe.
- No train/validation/test session overlap or exact cross-role feature duplicates. This is SAME-USER, not user-disjoint. No exact new-test feature overlap with embedded baseline training rows either; absence of exact duplicates alone is not proof of independence.
- Candidate embedded X/y exactly match the 1692 train rows; embedded scaler mean matches train-only calculation. Hyperparameters match the frozen protocol.
- Restored candidate and baseline separately with the existing bundle loader and reproduced all 1663 raw and rejected test predictions exactly. The loader reconstructs SVC from embedded training rows and retains the embedded scaler: deterministic restoration is not adding held-out data or model selection.
- Recomputed accuracy, macro-F1 and balanced accuracy with sklearn for all eight combinations of validation/test, baseline/candidate, raw/rejected; matched exports to 1e-12 with full support and fixed nine target classes. UNKNOWN is an error/FN, not dropped.
- Audited all 3409 exported validation/test prediction records: complete unique source-row coverage, hashes, true labels, identity, timestamps and curated-to-raw ACC-04 audit mapping.
- Final combined suite: `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests reports/acc04/test_repair_validation.py reports/acc05/test_acc05_benchmark.py` — **285 passed in 41.04s**, execution session 97052 exit 0. Includes a real synthetic end-to-end orchestration test with train X/y, scaler, restored fit and identical held-out-input assertions; synthetic train has 40 rows, not 1692.

## Comparable results

Primary rejection-aware test comparison, same 1663 rows:

| Measure | Production baseline | Fixed personal candidate |
|---|---:|---:|
| Accuracy | 0.2507516536380036 (417/1663) | 0.9158147925435959 (1523/1663) |
| Macro-F1 | 0.16310594760016325 | 0.9136945288983959 |
| Balanced accuracy | 0.2639752362215273 | 0.9069587619553475 |
| Coverage | 0.9711365003006615 | 1.0 |
| Rejections | 48 | 0 |
| OTHER predicted as a target | 115/191 | 33/191 |

Accuracy gain 66.5063 percentage points; macro-F1 gain 75.0589 points. Raw baseline accuracy 0.2555622369212267 (425/1663), macro-F1 0.1641986222389224; candidate raw equals rejected. Validation rejected accuracy baseline 0.18442153493699887 versus candidate 0.9478808705612829, macro-F1 0.12036278295541335 versus 0.9478818729871755.

Paired test correctness: both correct359, candidate only1164, baseline only58, both wrong82. All relative gates pass: accuracy and macro-F1 strictly higher, no per-class recall drop over0.02, OTHER-to-target count lower. These are frame predictions, NOT measured application action events.

## Deviations, root-cause limits and residual risks

Worker ran an ad-hoc in-memory preliminary benchmark before implementing the final script and before saving/hashing candidate artifact, contrary to the frozen artifact-before-test requirement. Coordinator observed and immediately required disclosure, preserving identical prescribed parameters/data/gates and prohibiting post-result tuning. Final script saves/hashes before its reproducible export, but this does NOT erase the earlier deviation. Acceptable evidence of observed descriptive gain; not a pristine confirmatory protocol execution. This test is now consumed.

Script class order follows collection order rather than protocol alphabetical order: PINCH_CLOSED, PINCH_MIDDLE, L_SHAPE, OTHER, POINT_ONE, OPEN_PALM, FIST, TWO_FINGERS_PARALLEL, THREE_FINGERS. Both models and metrics consistently use all nine; scalar results unaffected. Document rather than alter artifact after results.

Baseline embedded y contains all nine classes. A claim that baseline lacked the other classes is false and was returned to Worker for correction. The BGR-to-RGB collector bug was established in ACC-03; this experiment does not isolate its contribution to classifier improvement. Personal/domain calibration is a plausible explanation, not proven causal attribution. Historical six-class HaGRID and leaky nine-class metrics are not comparable to this test.

All takes are from one user on one day, with correlated frames within held poses. No frame-independent significance/p-value claim is justified. No generalization evidence for friends, another day/lighting, no-hand negatives, detector dropout or end-to-end commands. Candidate PINCH_CLOSED recall104/161=64.60%, PINCH_MIDDLE111/161=68.94%; OTHER33/191=17.28% target-frame false positives remain material risks. Relative safety improvement is not absolute deployment safety.

## Protected artifacts

Final hashes verified unchanged during review:

- Production models/airpad_svm.json: 4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977.
- reports/acc05/benchmark_personal_svm.py: 6fccfee2be2c9f029bc02badd691486d72af90f06f422ce53d2d503256dea6d4.
- Candidate bundle: 66091d52d9dd6848fd8e35fc2641a31c4c2ffc9008329a4d2e9f85f8dd1246be.
- Predictions: 71f04150db7f3195bcb407f190e0f1cdebd00d01390d06ca6d2202241faf6dee.
- Metrics: 5d8dcf852d1c48983875a39d46abe87b490d0a866934ccec19ab905b1fdd9a51.
- Final experiment tests: fadd6e2e4a7cd5ba33b18577be77e629eb50c7c7593b4240c788cac6b9af1454.

All six metrics-input hash checks passed, including all four datasets. ACC-04 audit hash is pinned and verified. Raw/rejected/retake data preserved.

Final collector, evaluator, existing tests/test_tools.py and all three original datasets also match their pre-task hashes. Final Worker report hash b5fdad7413046c79d416b935fccf5bba04bd8480d38cf5e5ffb4160f1745dfa6; chronology dd521d55a88fbe19fe9252e20b569cfb6173f6d1a06bd8bd5df84923afdccf8a. Worker completion reports its own full run285 in19.16s; independent Coordinator result285 in41.04s is authoritative for this review. Final Orca resource enumeration shows historical failed/succeeded dispatches retained with absent/external resources, no disposable new supervised worker. Existing Antigravity terminal is deliberately preserved, not stopped or released through old capabilities.

## Exactly one next priority

The highest-priority follow-up is a safe shadow-only trial of this unchanged candidate on a fresh independent take, with action dispatch disabled, including non-target/no-hand periods and pinch gestures. Scope a separate Worker task only when moving beyond the original baseline-exceeding loop; do not tune using consumed test or silently promote the model. No second task launched. Physical performance by the user would be needed for live evidence; existing offline data cannot establish action safety.
