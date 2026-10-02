# ACC-02 Coordinator review — 2026-10-01

Outcome: **accepted for collection safety and known-cohort readiness only**. Accuracy above baseline is **not demonstrated**. The autonomous loop is at a human-data blocker, not a model/architecture decision.

Run `run_0cbe816bafce`; task `task_8d7f4243f92c`; dispatch `ctx_0571b9f0320a`. Antigravity was the only code editor. Coordinator reviewed code, ran diagnostics/tests and wrote Markdown reports. Worker_done succeeded with matching lifecycle IDs; delivery acknowledged after worker-release returned retained/external_terminal, processAction=none. Reclaimable enumeration returned zero. No code task remains active.

## Scope and review evidence

- Modified `tools/collect_data.py`: unreadable existing output fails before camera/save; valid append retained; all nine hotkeys documented. Reviewed `git diff --no-index` against the pre-edit backup. There is no Git repository; no Git initialization or reset was performed.
- New `tools/cohort_sessions.py`: explicit roster and pair-keyed session roles; independent sessions across train/validation/test; stable nine-pose coverage per intended participant/role; transitions separated; original file hashes and row indices; duplicate/invalid inputs rejected before output; path-alias safeguards; deterministic manifest fingerprint; no manifest emitted unless READY.
- New `tests/test_cohort_sessions.py`, Vietnamese `docs/KNOWN_COHORT_COLLECTION.md`, and Worker report/audit/diff under `reports/acc02`.
- Review rejected successive unsafe drafts using concrete in-memory probes: implicit test-session selection, duplicate assignments/JSON keys, direct helper validation bypasses, incomplete provenance, malformed hashes, and a transition-only friend incorrectly marked READY. Corrections stayed within this one dispatch. Final cohort probe returns NOT_READY for the transition-only friend and READY when both users have full stable coverage.

Independent final verification:

| Check | Result |
|---|---|
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider` | **260 passed in 85.79s**; previous baseline 225 passed |
| Focused suite, same options, `tests/test_cohort_sessions.py` | **35 passed in 1.09s** |
| Collector module invocation with `--help` | Exit 0, no camera launched |
| Real custom-data audit with explicit tuana/train plan and `--fail-on-not-ready` | **NOT_READY**, exit 1 |
| Original three NPZ files, production model and ACC-01 evaluator hashes | Match preserved baseline |

Final source SHA-256 (unchanged across final independent test and completion review):

- Collector: `1a1c43225045a979957aa65e183fc9f9f8b52d88924b7f55e75d776d0daeda17`.
- Cohort utility: `a6cd8e9fa753cbb587477f4e8e3865b88da1ee56e0d761484158c8b678893c7f`.
- New tests: `249daf9ada74a99041ddb28ac77a69c57a8f6c3552fbaf9f97712b1f830056c3`.
- Collector pre-edit backup: `9c90d5acf46768e25d9054463478945cd1381d5a0b1b525f1e72082277f7bd6f`.

## Baseline comparison and accuracy decision

ACC-01 established a fixed six-class HaGRID baseline: 60,000 original rows, five true-user-disjoint folds; raw accuracy/balanced accuracy **0.9917333333333334**, raw macro-F1 **0.9917306463380989**. After rejection, macro-F1 is **0.9918544754844651**, coverage **0.99975**. This is not a nine-class personal/cohort baseline, and cannot be compared to the historical leaky nine-class score.

ACC-02 did not change features/classifier/runtime, train any candidate, export a model or run a new accuracy benchmark. Therefore no classification-accuracy improvement claim is warranted. It improves protection of collection files and the validity of the next experiment.

Real custom data still has 1066 stable rows from tuana's single session, covering only PINCH_CLOSED, PINCH_MIDDLE and L_SHAPE. The six remaining poses, validation/test sessions and friends' recordings are absent. Existing data stays development/train-only.

## Remaining limitations

- Readiness checks validate declared IDs and coverage; they cannot prove that a person actually recorded independent sessions. Inventing new IDs for chunks can still defeat provenance without human/source audit. Interpret the Worker report's claim about rejecting pseudo-independent folds with this limitation.
- READY is structural readiness, not an accuracy guarantee or sufficient statistical power. One frame per class meets coverage but is not a robust experiment. Use varied short bursts, real separate recording occasions and per-person/per-class metrics later.
- Collector fail-closed behavior protects the reproduced unreadable-file case, not every possible crash/concurrent-write scenario. No universal zero-data-loss guarantee is established.
- Extra unrostered participants trigger NOT_READY; diagnostic totals still describe all loaded rows. The Worker report's statement that they are excluded from totals is stronger than the implementation.
- NOT_READY does not create a new manifest; an old manifest already at that path is not deleted. Future consumers must check the fresh readiness status, source hashes and fingerprint; never treat file existence as approval.
- Known-cohort session evaluation is the agreed target. The ACC-01 unseen-user evaluator is not yet a nine-pose personalized evaluation workflow; do not bypass its guards.

## Exactly one highest-priority next task

**Collect genuine known-cohort sessions**, with human participation. No further code/model optimization is dispatched before this blocker is resolved.

Scope: tuana and the intended friends, stable anonymous IDs, at least three genuinely separate recording sessions per person, full nine-pose coverage per role, OTHER negatives and separately flagged transitions. Save each new session to a new NPZ under `data/cohort`; preserve originals. Assign train/validation/test before inspecting model performance. The existing incomplete session is development data, not a held-out test.

Files/workflow: follow `docs/KNOWN_COHORT_COLLECTION.md` using `python -m tools.collect_data`, then audit through `python -m tools.cohort_sessions` with explicit roster and plan. Camera recording requires the participants' presence and human initiation; Coordinator has not opened a camera.

Acceptance: real sessions confirmed by participants, explicit pair-keyed plan, READY for every intended person and pose, separate transitions, pinned source hashes/original rows, no input/model overwrite. Only then can a single Worker evaluation task establish a comparable nine-pose known-cohort baseline and measure a candidate against it.

Question needed from user: availability of tuana and the friends to record the required genuine sessions. The purpose (personal plus friends) is already settled and needs no reconfirmation.
