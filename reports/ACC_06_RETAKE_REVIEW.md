# ACC-06 focused retake review — 2026-10-01

Outcome: complete physical shadow retake audited; frozen personal candidate exceeds unchanged production baseline on the SAME905 intended-target observation frames. No model promotion or real-control readiness claimed. Initial noncompliant take remains excluded from quality conclusions, unmodified. No code task or camera restart in this review.

## Integrity and completion

Execution87005 exited0, COMPLETED all12 stages,3320 frames. Session shadow_obs_20261001_162213_fb6d49, usertuana, live_camera_observation, role shadow_observation. Log reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl SHA2562e2753def2d95bf9a33cc88c55eaba6f19560b7cd5df2815cc037baa3abc8cd2; summary SHA2569ec436037be8e8a6fba4d20ff7d8d961d6fe27a8564c8e4057c82a43b3e3bf8f.

Independent checks: contiguous indices0..3319, session/user linkage, zero commands, finite detected raw/canonical landmarks, fixed-nine sklearn accuracy/F1 match summary. Waiting1484/preparation697 excluded; stable1139=target905+NO_HAND234. Candidate66091d52d9dd6848fd8e35fc2641a31c4c2ffc9008329a4d2e9f85f8dd1246be, baseline4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977 and tool128f35542be94c9f40c8e2d1b6d15d42b3f994cc0b6b9932c2f31aacfdd15301 unchanged. Earlier308 tests verify this unchanged code; no fresh code changes to test.

## Paired comparison, not historical cross-protocol numbers

Restored both actual bundles independently, used the exact816 logged canonical feature rows for detected target frames, same decision0/margin.15, and assigned NO_HAND for all89 missing-detection frames for BOTH models. No double-canonicalization, extra fitting on observations or threshold selection. Restored candidate predictions exactly reproduce saved log. Diagnostic command execution40037 exited0; metrics computed read-only in memory.

| Metric on same905 frames | Production baseline | Personal candidate |
|---|---:|---:|
| Full-support accuracy | 253/905=27.9558% | 814/905=89.9448% |
| Fixed9 macro-F1 | .1816308487082773 | .966000818361046 |
| OTHER predicted as a target | 109/198 | 0/198 |

Accuracy gain61.9890 percentage points; macro-F1 gain78.4370 points. Paired correctness both251/candidate-only563/baseline-only2/both-wrong89. This is a valid same-row DESCRIPTIVE comparison of frozen models using intended labels, not independently verified laboratory truth or frame-iid significance.

## What changed in the live evidence

Numbered1–9 stable blocks:795/795 hand detection;793/795=99.7484% intended-pose agreement. PINCH_CLOSED89/89 correct, PINCH_MIDDLE86/88=97.7273%; the two errors predict OTHER, not another action target. L_SHAPE, POINT_ONE, OPEN_PALM, FIST, TWO_FINGERS_PARALLEL and THREE_FINGERS all correct on this short take. Numbered OTHER88/88 detected/correct.

All89 detector losses occur in the EXTRA post-gestureOTHER block (21/110 detected), accounting for89 of91 full-support candidate mismatches. Keep these rows as errors under the declared protocol; do not silently remove or relabel. Movement/occlusion/hand leaving frame in this non-target stage cannot be distinguished from detector failure without physical evidence; no camera images were stored. Safe inactivity is not an observed OS action, so neither false-command rate nor user interaction safety is established.

BothNO_HAND blocks234/234 no hand detected and zero target predictions; OTHER0/198 target predictions. Target detection overall816/905=90.1657%, detected-only agreement814/816=99.7549% is supplementary, NOT replacement for89.9448% full support. This positive retake supports satisfactory static recognition in these conditions, not every-day/lighting/friend generalization or temporal action safety.

## Exactly one next priority

An end-to-end dry-run of the frozen candidate through the existing gesture/temporal/FSM runtime, with real OS control forbidden, is the next priority. The existing app.run_camera BGR-to-RGB contract defect must be corrected and regression-tested by Antigravity as part of that bounded runtime trial before using that entrypoint; shadow/collector paths already correct. Scope app.py + focused tests/test_app.py and task reports, preserve models/config/datasets and thresholds; confirm DryRunController only, RGB detector/BGR preview, no real-controller creation, candidate explicit model path, cleanup/no-hand/reacquire behavior and existing308 tests. Do not transfer observations into training or promote the production bundle. This is proposed, NOT assigned/launched by a status-review turn. No further static recordings needed merely to reconfirm this take; physical dynamic-action dry-run is a distinct validation step.
