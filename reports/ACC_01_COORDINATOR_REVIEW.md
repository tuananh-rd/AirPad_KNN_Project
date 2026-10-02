# ACC-01 independent Brain / Coordinator review

Date: 2026-09-30. Run `run_0cbe816bafce`.

Decision: accept the evaluation implementation and six-class baseline, with limitations below. Do not claim improved nine-class AirPad accuracy and do not promote a production model.

Only Antigravity edited code. Coordinator read code/diffs, ran independent verification and wrote this review. Workspace has no Git repository; reviewed existing-file changes using `git diff --no-index` against Worker pre-edit copies and read newly created sources. No repository was initialized.

## Verified changes

- `src/ml/evaluation.py`: UNKNOWN contributes false negatives and preserves full support; explicit prediction rejection column; unexpected label and missing/insufficient group validation.
- `tools/train.py`: missing groups no longer become independent unknown_i groups; insufficient user groups fail instead of silently falling back to sessions.
- `tools/audit/provenance.py`: verifies pinned datasets and maps combined rows to original user/session provenance; checks preserved metadata and identifies 43 duplicate custom feature/label keys.
- `tools/evaluate.py`: fixed-parameter grouped evaluation, original source-row tracking, user/session disjointness, full test-fold coverage, manifest checks, insufficient per-class user rejection and fingerprints.
- `tests/test_evaluation.py`: 18 additional regression tests, including actual evaluator scaler-state checks and isolated artifact output.

## Independent checks

Command: `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider`.
Result: **225 passed in 66.56s** (207 original + 18 additional).

Coordinator compared metrics on 1000 randomized predictions including UNKNOWN against sklearn. Accuracy, macro-F1 and balanced accuracy match; all observations remain counted.

Coordinator verified all 60,000 full benchmark prediction source IDs and their labels/users/sessions against original HaGRID NPZ. Each source row appears once as test; each fold has disjoint true users and sessions, and complete training class support. Aggregate/fold metrics below were independently recomputed from predictions.

Dataset: 60,000 rows, six HaGRID classes, 9,170 users. Fixed C=10, gamma=.01, class_weight=balanced; thresholds 0 and .15; seed 42; five user-grouped folds. No tuning or candidate comparison occurred.

| Metric | Before rejection | After rejection |
|---|---|---|
| Accuracy | 0.9917333333333334 | 0.9917333333333334 |
| Macro-F1 | 0.9917306463380989 | 0.9918544754844651 |
| Balanced accuracy | 0.9917333333333334 | 0.9917333333333334 |
| Support | 60000 | 60000 |
| Coverage | 1.0 | 0.99975 |

15 predictions were rejected. No accepted label changed. Rejection changed macro-F1 by removing some false positives; it did not increase total correct predictions or accuracy. This is not a model improvement experiment.

| Fold (zero based) | Train rows | Test rows | Accuracy after rejection | Rejections |
|---|---|---|---|---|
| 0 | 48036 | 11964 | 0.9938147776663323 | 2 |
| 1 | 48006 | 11994 | 0.9904118726029681 | 6 |
| 2 | 47838 | 12162 | 0.9909554349613551 | 3 |
| 3 | 47938 | 12062 | 0.9914607859393135 | 0 |
| 4 | 48182 | 11818 | 0.9920460314774073 | 4 |

Split fingerprint: `9f8d1131ecaee79bf3bff3a52090adac9ff083fa59bd98fc4ad8bd552c8afe7e`.
Evaluator SHA-256: `fa9995adcda6ec425431362c92baa62f2013c062e7704cf44104f42f829ad67c`; matches benchmark record and reviewed evaluator.

Original custom, HaGRID, combined and production model SHA-256 values are unchanged from the pre-task records in `BRAIN_COORDINATOR_LOOP.md`.

## Comparison and unresolved risks

Historical nine-class macro-F1=0.9672760042 and accuracy=0.9901418138 came from a different evaluation with fake custom user/session chunks. They are not directly comparable with this clean six-class baseline. No candidate was trained or substituted, so an accuracy gain over a comparable baseline has not been established.

PINCH_CLOSED, PINCH_MIDDLE and L_SHAPE have only one real user/session. A clean nine-class unseen-user benchmark is unavailable. Additional independent webcam sessions are required even for personal-use evaluation; multi-user evaluation also requires additional real users. HaGRID landmark accuracy does not measure live detector reliability, temporal action correctness, latency or false command activation.

The original combined NPZ and legacy preparation script were intentionally preserved. `tools/train.py` still cannot infer real identities from arbitrary supplied metadata and its historical tuning/OOF protocol is not a nested evaluation. Use the audited evaluator for baseline claims; old trainer scores must not be presented as independent accuracy evidence. Audit currently pins the existing dataset hashes and is not yet a general-purpose importer for future data.

The Worker report contains stale values in its raw-accuracy and per-fold tables. Its artifact JSON and prediction records are consistent with the independent recomputation here. Documentation-only repair task ACC-01-R failed twice at Orca agent readiness before input delivery; this review supplies the authoritative correction and does not claim that repair completed.

## Loop checkpoint

No active code dispatch remains. Original Worker terminal was preserved by Orca as pre-existing; failed starts owned no resource; reclaimable enumeration is empty.

Next required user decision: target personal use with additional independent sessions, or general use with additional real users and sessions. New recordings require people and camera activity not supplied by offline artifacts. Further model optimization or architecture changes cannot bypass this evidence gap. A separate Worker readiness issue also needs recovery before the next dispatch.
