# HYB-01 — Hybrid assist: physical mouse + two-finger scroll + three-finger tabs

2026-10-02. Brain/Coordinator approved implementation after user explicitly requested retaining THREE_FINGERS for tab switching. Exactly ONE task, sole existing Antigravity in airpad-worker. Coordinator edits reports only. Supervised direct Orca terminal fallback previously authorized after structured readiness failed; this prompt carries no old Dispatch authority. Do not fabricate orchestration worker_done.

## Evidence and baseline

Read reports/HYBRID_DIRECTION_PROPOSAL_2026-10-02.md and latest ACC_09_WORKER_REPORT.md / ACC_09_COORDINATOR_REVIEW.md (including final acceptance checkpoints, not only historical HOLD).

Independent pre-task focused suite: 104 passed, 5.99s, chunk30e48a. ACC09 accepted full baseline:356 passed, self-check5/5. Diagnostic f2fcf2 demonstrated perfect raw PINCH still emits three stale POINT_ONE cursor moves before CLICK_PENDING; scroll100 frames dy=.002 yields zero units despite cumulative movement .2 (large-motion control emits23). WindowsController.swipe currently maps BACK/FORWARD to Alt+Left/Right (history), not tabs; pipeline never forwards swipe pose exit. These justify interaction changes, not retraining or recognition accuracy claims.

Preserve legacy behavior without selecting Hybrid. Frozen ACC07 legacy replay:3320frames,156events=151MOVE+5SCROLL,1445no-hand frames, exact per-frame/header/stages/actions match. Preserve all previous reports/artifacts; use new exclusive paths.

## Product contract

1. Add explicit --profile hybrid (and a clearly named legacy/airmouse profile; old CLI/default behavior must remain compatible). Hybrid starts OFF. SPACE in the focused camera preview toggles assist; say explicitly it is preview-focused, not a global hotkey. HUD shows profile and OFF/READY/SCROLL/TAB state, DRY RUN versus REAL. Ctrl+Alt+Q emergency remains global and latched; SPACE cannot recover emergency stop.
2. Physical mouse owns pointer, left/right click and dragging in Hybrid. For ALL poses, stale filtered labels, transitions and long holds: zero generated MOVE, CLICK, mouseDown, zoom, browser-history or window/desktop-switch events. Apply whitelist BEFORE calling legacy FSM with direct click/down side effects, not only after FSM dispatch. Do not release physical mouse buttons that AirPad never pressed. No global mouse listener or input ownership redesign.
3. TWO_FINGERS_PARALLEL directly enters vertical scroll after stable recognition, no POINT_ONE arming. Slow cumulative movement must scroll, while bounded stationary jitter/forward-return subthreshold noise does not creep. Do not simply remove the dead-zone. Warm baseline on entry; reset on exit/no-hand/OFF/profile lifecycle; no stale movement burst or scrolling during raw pose transition. Define and test direction: hand up in mirrored preview => scroll up, hand down => scroll down. Horizontal scrolling is out of scope for Hybrid v1. Compare same path at15/30/60FPS; bound gain/output differences with a declared tolerance.
4. THREE_FINGERS directly supports deliberate left/right swipes: preview LEFT => previous application tab (Ctrl+Shift+Tab), RIGHT => next (Ctrl+Tab). Not Alt+arrows/history, not Alt+Tab/windows. Require stable pose AND real trajectory displacement/velocity; stationary pose, brief2/3 flicker without deliberate swipe, transition geometry and cooldown cannot switch tabs. One shot per swipe; exit pose/reenter permits another deliberate swipe but does not erase global cooldown. Reset baseline at lifecycle boundaries; returning neutral while still THREE must not cause unintended reverse/duplicate. Foreground application must support these shortcuts; browser/editor support is not universal OS behavior.
5. SafetyGate remains shared, fail-closed real admission unchanged. OFF transition and dispatch serialized under gate; OFF means no future forward event after transition completes. Tab composite tracks Ctrl/Shift/Tab conservatively BEFORE potential OS effects, forward steps recheck stop; release remains permitted stopped. Partial down/press/up failure retains unresolved keys, attempts all acquired modifier releases, cleanup retries and propagates error. No untracked pyautogui.hotkey shortcut. DryRun semantic/events reflect same intended tab contract with zero actual hooks/OS input.
6. Zoom and gesture click/drag unavailable in Hybrid, retained legacy. No classifier/filter threshold changes, flicker tuning, retraining, new ML architecture, screen mapper change, backend PAUSE tuning, model promotion or default config rewrite.

## Bounded file scope

Allowed only as needed: app.py, src/runtime/pipeline.py, ONE small src/runtime/hybrid.py helper if it simplifies isolated profile policy (same classifier/pipeline/gate, not a new architecture), src/gesture/fsm.py for profile gating only if necessary, src/motion/scroll.py and src/motion/swipe.py for opt-in/lifecycle behavior preserving legacy, src/platform/{interface,windows,dry_run}.py, src/config/schema.py if required; relevant existing tests plus new tests/test_hybrid.py; docs/AIRPAD_FUNCTION_GUIDE_VI.md; isolated reports/hyb01/* and reports/HYB_01_WORKER_REPORT.md. Prefer minimal files and legacy FSM unchanged. src/safety/failsafe.py protected: request Coordinator scope amendment with reproducer before changing it. No edits to models, data, detector, temporal filter, config/default.yaml or frozen benchmarks/reports.

Before first edit, save exact originals (preserve relative path hierarchy) under reports/hyb01/before and SHA256 manifest, including protected inputs. This is NOT a Git repository. Produce git diff --no-index snapshots versus final code; exit1 means differences, not test failure. Never initialize Git or reset/overwrite originals. Never launch webcam/real control.

## Tests and benchmark required

All automation camera/backend/hotkey mocked before imports; no OS hooks. Deterministic label-controlled landmark streams distinguish interaction testing from model accuracy.

New tests cover OFF start, toggle, all9poses incl pinch click/drag holds, transitions with stale filter, scroll immediate/direct entry, slow100x.002, jitter/return-to-origin, both vertical directions, horizontal suppression, entry/exit/no-hand baseline reset,15/30/60FPS parity, deliberate3finger left/right, stationary and brief flicker, one-shot/cooldown and exit/reentry, no accidental history/zoom, physical mouse never released, emergency stopped forward suppression, reentrant stop during modifier down/Tab press, each partial press/down/up fault and cleanup retry, real admission fails without emergency hook, CLI profile wiring and mocked camera HUD/SPACE.

Create a reproducible bounded headless action benchmark under reports/hyb01 (new artifacts only) comparing pre-change legacy and Hybrid on declared identical synthetic intent scenarios, plus legacy frozen replay separately. Report intended versus emitted action counts, forbidden events, slow-scroll units and direction, jitter events, tab one-shot/rearm outcomes. No claims of real user completion rates or improved recognition accuracy from synthetic actions.

Commands:

    .venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests/test_hybrid.py tests/test_fsm.py tests/test_fsm_phase2.py tests/test_pipeline.py tests/test_motion.py tests/test_windows.py tests/test_safety_contract.py tests/test_app.py
    .venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests reports/acc04/test_repair_validation.py reports/acc05/test_acc05_benchmark.py reports/acc07/test_acc07_dryrun.py
    .venv/Scripts/python.exe -B app.py --self-check

Fresh guarded replay via unchanged reports/acc07/replay_acc06_dryrun.py into NEW exclusive reports/hyb01 artifact, compare frozen3320frames/156actions exact. Models/candidate/cohort/detector/defaultconfig/log/frozenreplay SHA unchanged.

Acceptance: all regression suites and self-check pass, default legacy replay exact, Hybrid benchmark produces zero forbidden events, slow-scroll nonzero in correct direction and bounded jitter no events, repeat deliberate tabs both ways once each and cooldown safe; all safety fault/stop cases pass with no leaked holds on successful cleanup. Document actual executed commands/counts, source hashes, remaining physical-trial uncertainty and foreground/modifier shortcut limitations. Include Vietnamese function table and dry-run/real launch instructions with explicit profile and existing personal candidate path reports/acc05/candidate_personal_svm.json. Coordinator independently reviews final stable version before camera.

## Submission and quota

### Same-task amendment 11:00: delayed deliberate THREE swipe

Independent RED201fad: Hybrid engine holds x=.5 from1000..3000ms in40ms frames, then x=.62@3190ms; no event despite rapid latest190ms displacement. Velocity averages from pose-entry time, so waiting to stabilize can permanently impair later swipes. This is new mechanism evidence, not permission for a new architecture or classifier tuning. Source freeze lifted ONLY for src/motion/swipe.py and tests/test_hybrid.py; other production files/tests remain frozen. Allow Hybrid-only recent-motion trajectory/window or stationary-baseline refresh while retaining displacement .08 and velocity .0003 and legacy behavior. Tests:2s/10s stationary hold then rapid .12/150–190ms both directions =>exactlyone; stationary jitter/slow drift=>none; existing one-shot/neutral-return/pose-exit/cooldown intact; default-filter pipeline regression and matching synthetic benchmark. Worker remains sole editor, same task, previous benchmark/guide repairs also required. Brain must rerun focused/full final stable version; previous401pass verifies previous source, not this upcoming amendment.

Coordinator additional read-only pre-implementation check: chunkccbde0, exit0, pre-import fake pyautogui records BACK=[down alt,press left,up alt], FORWARD=[down alt,press right,up alt], holds empty; default slow100x.002 returns0. No OS input generated. Confirms tab-semantic mismatch and slow-scroll mechanism independently this turn. The submitted task text predates this appended evidence; this addition does not alter its scope or criteria.

Fresh pre-edit focused rerun while terminal prompt still being entered:104 passed8.77s, exit0, session32756/chunk7a2efe. Worker has not started edits at this checkpoint; no camera.

Report final diffs, tests/benchmark artifacts, limitations in reports/HYB_01_WORKER_REPORT.md; then reply HYB01_DIRECT_DONE and idle. Do not start another task or camera. If a genuine blocker arises, report it without widening scope. On included usage/quota exhaustion STOP immediately, checkpoint incomplete work; never use paid credits, reset credits, account/model switch or paid fallback. Do not assert a global billing lock. This is the single current task; subsequent feedback is same-task repair.
