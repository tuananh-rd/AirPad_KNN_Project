# ACC-07 preliminary physical dry-run feedback

User reports almost all gestures work, with intermittent intended two-finger gesture recognized as three fingers. This is qualitative user feedback, not a saved runtime benchmark: app preview has no persistent frame/event log. Do not infer a confusion percentage, raw-classifier error, lighting cause or hardware failure from it.

Read-only comparison of frozen ACC05 test_predictions: intended TWO_FINGERS_PARALLEL181 rows, candidate181 correct (baseline181 PINCH_CLOSED); THREE_FINGERS186 rows, candidate186 correct (baseline131 PINCH_CLOSED/54OTHER/1correct). ACC06 compliant retake stable OBSERVATION replay: two-finger93/93 raw+filtered correct; three-finger87/87 raw+filtered correct. Different conditions/short static samples cannot rule out this new reported confusion during movement or pose transition. No training or threshold adjustments based on consumed test data.

App displays filtered pose. MajorityFilter5/.6 retains previous stable pose when no majority; StabilityGate2 retains previous pose during settling. Hence brief THREE_FINGERS display after transitioning from three to two can be filter lag; persistent THREE_FINGERS while holding intended two may instead be detector/pose/classifier ambiguity. Existing evidence does not distinguish these explanations.

Next priority is clarify duration under a stationary intended two-finger hold, starting from hands outside camera rather than from three fingers. Ask user whether incorrect THREE_FINGERS disappears during5s hold or persists/reappears while stationary. Keep audited candidate and dry-run only; no new code task or extra camera launched, no real control/model promotion. Need affected runtime evidence before selecting a corrective implementation.

## Follow-up and physical sequence completion

User clarified the two-to-three display only flashes briefly and returns to correct two-finger pose. This is consistent with transition/filter settling, not proof of that cause: no saved raw runtime frames/scores establish whether raw predictions also flickered. No persistent stationary misclassification reported, so no threshold/model change justified.

Coordinator then requested three repetitions of pointing/moving, short pinch/release, sustained pinch/drag/release, hand removal/reacquisition in existing dry-run preview. User replied "ok rồi đó" (completed). Record completion, not independently verified exact event counts or a new benchmark; no detailed faults reported. Read-only process check finds no project camera Python process, only antigravity-mcp helper, so preview no longer running; exact camera exit code not claimed.

ACC07 implementation remains accepted with327 tests and audited offline replay; physical feedback is qualitative. Frozen candidate already exceeded paired baseline on prior datasets. No new code task, model promotion, retuning, architecture change or real control enabled. Real OS control would be a materially different trial requiring an explicit user decision and bounded setup; completion of dry-run does not authorize it.
