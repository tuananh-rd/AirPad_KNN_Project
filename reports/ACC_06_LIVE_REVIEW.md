# ACC-06 completed physical shadow take: independent review

Date2026-10-01, user reports done. Execution session27496 exited0: COMPLETED, all12 stages,9134 logged frames. No live OS input actions; production/candidate unchanged. This is a completed schedule, not proof every physical intended pose was performed correctly.

## Provenance and checks

Session shadow_obs_20261001_135704_8e95a2, usertuana, role shadow_observation, header provenance live_camera_observation. Source reports/acc06/trial_shadow_obs_20261001_135704_8e95a2.jsonl SHA256855797f165a054d436eb6b6aa82a9018a67199acac6b03c24f00ce6f87eac0f0; summary SHA25637de5f6f2f4505380b27ce5a08269dbc89874b1e76783564ff5c102776f026ce. Immutable candidate/detector hashes match header.

Independently verified contiguous frame indices0..9133, session/user linkage, commands0, finite detected landmarks/features, nondecreasing timestamps, all12 stage supports and roughly8-second stage intervals. Independently recomputed full-support accuracy and fixed9 macro-F1 with sklearn, matching saved summary. No summaries altered or frames discarded. Waiting7074, preparation753 excluded from scored stable observations; stable1307 = target1101 + intendedNO_HAND206. Previous308 tests passed for the unchanged tool; no new code changes or rerun needed to audit this log.

## Observed results against intended labels

| Measure | Result |
|---|---:|
| Target hand detections | 327/1101 =29.70% |
| Target detector losses | 774/1101 =70.30% |
| Full-support rejected prediction agreement | 273/1101 =24.80% |
| Fixed9 macro-F1 | .3434148003986435 |
| PINCH_CLOSED detections / intended support | 0/112 |
| PINCH_MIDDLE detections / intended support | 1/114 |
| L_SHAPE detections and correct predictions | 80/80 |
| OTHER-to-target frames | 36/240 =15.00% |
| Hand detected during intendedNO_HAND | 112/206 =54.37% |
| Target predicted during intendedNO_HAND | 94/206 =45.63% |

Of828 target mismatches,774 are missing detection;54 occur on detected frames. Conditional detected-only agreement273/327=83.49% is diagnostic, NOT replacement for24.80% full-denominator result. Detector/capture/performance-of-prompt is the main observed bottleneck; classifier-only tuning cannot recover missing landmarks.

Initial intendedNO_HAND block has87/87 detected frames with multiple target predictions (POINT_ONE53, FIST15, PINCH_MIDDLE7, TWO_FINGERS_PARALLEL6, OTHER3, L_SHAPE2, UNKNOWN1). Final intendedNO_HAND block has25/119 detections. This creates a necessary physical-protocol ambiguity: hand may not have been removed, or detector may have produced detections on something else. Landmark-only logs cannot establish which; no camera images were stored by design. Do not label these as proven false detections without confirmation.

## Comparison and decision

ACC05 .9158147925435959 is accuracy on a different test take of successfully collected landmark rows. This shadow take includes detector losses and intended prompts rather than independently verified truth;24.80% is NOT a directly comparable paired model regression against ACC05 or production baseline. The earlier offline baseline-exceeding conclusion remains bounded to that take. Live trial does not meet readiness for real control; do not promote or enable OS actions.

Collector and shadow tool both use max_num_hands1/min_detection_confidence.5, flippedBGR-to-RGB detection, same existing adapter/local asset. No obvious configuration mismatch found. No evidence to blame laptop hardware, change architecture or lower thresholds (which may worsen intendedNO_HAND detections).

Exactly one next priority: verify physical execution of negative/pinch blocks before assigning a code change or a controlled detector diagnostic. Need user confirmation: at each NO_HAND did both hands leave the camera frame; during numbered stages, especially1/2, was the prompted hand held visibly through the entire green OBSERVATION period? Preserve this take unmodified/never silently relabel. Await this factual information, not routine permission. No new Worker task/camera started at this checkpoint.

## User-confirmed protocol deviation and retake

User subsequently confirms talking with Coordinator while performing the take and not following instructions correctly, and requests a retry. Mark first take protocol-noncompliant: retain original log/summary unchanged for audit, exclude from model-quality conclusions and tuning/training. The numerical accounting above remains an accurate description of that log, not valid measured detector/classifier quality or evidence of laptop failure.

Verified no active camera process and unchanged tool/candidate/production hashes. Launched identical shadow-only tool with new automatic session shadow_obs_20261001_162213_fb6d49; execution session87005. Log reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl; verified live_camera_observation header, last checked frame398 WAITING, commands0. New reserved summary pending completion. No code task/model changes/overwrites. User presses SPACE once, follows12 stages without chatting during the timed take, both hands outside frame forNO_HAND, holds intended pose during green OBSERVATION. No new performance claim before completed retake audit.
