# Two/three-finger flicker — initial Brain diagnosis

Date: 2026-10-02. User reports that the two-finger pose still briefly flashes as three fingers during the post-ACC09 dry-run test. Physical acceptance is withheld; real OS control remains unauthorized. No code, model or configuration edits, and no new Worker implementation task in this review.

## Evidence

Current app HUD displays PipelineResult.filtered_pose, not the raw classifier prediction. The same filtered pose also feeds GestureFSM; therefore do not assume the observed flash is merely cosmetic. No unintended OS action was observed or claimed: the launched session was DryRun only.

Current configuration: MajorityFilter window5 / majority_ratio0.6, followed by StabilityGate required_count2. Read-only inline reproduction uses these existing classes, first primes eight TWO_FINGERS_PARALLEL frames, then supplies three THREE_FINGERS raw frames and five TWO_FINGERS_PARALLEL frames. Command chunk00810d, exit0:

| Frame after priming | Raw | Majority | Final filtered |
|---|---|---|---|
|1|THREE|TWO|TWO|
|2|THREE|TWO|TWO|
|3|THREE|THREE|TWO|
|4|TWO|THREE|THREE|
|5|TWO|THREE|THREE|
|6|TWO|TWO|THREE|
|7|TWO|TWO|TWO|
|8|TWO|TWO|TWO|

A three-frame raw confusion burst can therefore pass both existing filters and remain visible briefly even after raw classification returns correct. This is demonstrated filter behavior, NOT proof that this exact burst occurred in the current physical test, nor proof that the filter implementation violates its configured contract. Raising persistence blindly could delay valid gesture changes, click timing and safe release behavior.

Prior accepted ACC06 retake had93/93 intended two-finger and87/87 intended three-finger observation agreement in its short stable blocks. Those frozen conditions do not establish today's lighting/dynamic transitions. ACC09 intentionally changed safety synchronization, not classifier/temporal thresholds; its identical frozen replay is not evidence that today's flicker vanished.

## Missing evidence and next decision

Need distinguish flicker during a genuinely held-still two-finger pose from brief transition ambiguity while raising/lowering the third finger. Also need actual frame timing, hand-detection continuity, raw/rejected versus filtered predictions and classifier margins before choosing model rejection or temporal/action persistence changes. The ordinary app preview does not persist these traces.

Ask user the single materially relevant question: does it flash while holding two fingers still, or only during a gesture transition? If held-still flicker persists, highest-priority next investigation is a focused, labeled, dry-run two/three-finger trace in current conditions, retaining missing detections and excluding declared transitions from stable-pose scores. Do not request all nine gestures or collect new training data merely to diagnose this pair. Existing shadow tool has a fixed12-stage schedule rather than a focused pair CLI; do not silently alter its schedule/protocol from Brain or tune on its prior scored observation data.

No fresh accuracy figure, physical safety acceptance or corrective filter parameter is claimed. Only Antigravity may implement any needed diagnostic/code change after a bounded evidence-backed task is chosen.
