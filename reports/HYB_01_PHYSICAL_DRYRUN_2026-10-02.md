# HYB-01 physical dry-run — user reports OK; real-control trial not yet authorized

## User feedback checkpoint, 2026-10-02 21:53

User responds `ok rồi đó` to the four-step dry-run protocol. Record qualitative overall positive feedback only: no exact attempt counts, individual check results, raw-frame traces, recognition accuracy or true OS behavior were provided. No failure evidence justifies a new code task. Automated acceptance remains408tests+self-check+benchmarks; this user feedback adds preliminary physical dry-run confidence, not quantified physical acceptance.

Read-only process/window check55be4e confirms original launcher8204/child24872 still running exact Hybrid command WITHOUT --real-control; AirPad Virtual Pad responds. Leave this session intact. No new camera instance, code/test/config/model modification or Worker task. Coordinator's opening commentary proposed switching to real control, but NO switch/OS command was performed: real control is materially different from this explicitly dry-run session, so request specific user authorization before launching it. Proposed sole next step is a short real Hybrid browser scroll/tab trial on safe disposable pages, with focusedSPACE enable then physicalmouse select target, no held physicalCtrl/Shift/Alt, globalCtrlAltQ emergency and focusedQ/Esc exit. Do not close/restart the camera or claim this step completed until authorized.

Earlier startup/protocol details below retained for provenance.

Latest lifecycle observation: terminalb016ef now returns to PowerShell prompt, so the earlier55be4e running check no longer represents current state. Camera session ended while this checkpoint was being recorded; do not claim it remains open or infer why it ended. Final process/window check confirms current state separately. No automatic restart or real-control launch while awaiting authorization.

2026-10-02 21:49 Asia/Saigon. User asks to continue after automated HYB-01 acceptance. Exactly one next priority: supervised physical Hybrid dry-run; no code task or Worker prompt needed. Brain edits Markdown reports only. No retraining, source/test/config/model/guide edits, paid fallback or real-control launch.

## Preconditions and startup evidence

Read final Coordinator/Worker acceptance and Hybrid guide section9. Accepted evidence remains408passed183.57s, self-check5/5, bounded synthetic and captured benchmarks; not physical usability or recognition certification. Independent current hashes match accepted app.py, src/runtime/pipeline.py, src/motion/{scroll,swipe}.py, WindowsController, defaultconfig, explicit personal candidate and detector. No existing AirPad camera or pytest process before launch. Orca runtime restarted; no terminal in inspected main or old auxiliary camera context, so created ONE visible app terminal (not another agent/Worker).

Command executed through Orca in original checkout:

    .venv\Scripts\python.exe -u -B app.py --camera --profile hybrid --model reports/acc05/candidate_personal_svm.json

No --real-control: factory selects DryRunController; OS mouse/keyboard actions are simulated, emergency hook is dry-run rather than a real global hook. Profile startsOFF by code; user controls focused-previewSPACE. Do not promise physical CtrlAltQ stop during dry-run; focusedQ/Esc exits, SPACE disables assist. No global focus/key ownership or app instrumentation added.

Orca create receipt555516bb-8e04-4ebf-85ac-47b05cd76269, runtime6e6114a7-3148-4dbe-b563-c2a58c95ba4d, app terminalterm_ea40e4e6-41c0-4c98-9870-f6c8ea1b2cf2. Read9ae36f positively shows live camera loop started; launcher8204/child24872 match exact command and parent relationship; child24872 has responding AirPad Virtual Pad window. These prove startup, NOT good recognition/FPS or test success. Retained for authorized user participation. Startup detector warnings are recorded in terminal; no crash observed and no tuning inferred.

## Short user protocol

Keep preview focused for HUD and SPACE; no application tab actually switches in dry-run.

1. With assistOFF, move hand and try TWO/THREE: no Action should fire. Click preview and pressSPACE to enableREADY.
2. Hold TWO_FINGERS_PARALLEL, move slowly up then down 3times; HUD should enterSCROLL and show ActionSCROLL during intended movement. Hold still briefly: no repeated scrolling. Dry-runHUD does not display signed wheel units, so actual OS direction/usability remains for later controlled real trial.
3. Hold THREE_FINGERS still for2s, then swipe previewLEFT: oneTAB_PREV. Return toward neutral WITHOUT changing pose: noTAB_NEXT/duplicate. Drop hand/change pose before next attempt; holdTHREE2s then swipeRIGHT: oneTAB_NEXT. Repeat each direction3times, resting between.
4. PressSPACE toOFF and retry gestures: no Action. Move/click physical mouse normally: AirPad should not move/click it. This observation alone does not certify actual OS coexistence because this session is dry-run.

Exit withQ/Esc in preview, or leave running for follow-up. User reports scroll/tab missed events, accidental repeats/held-return behavior, and detection stability separately. No current raw-frame/action log is being persisted by this unmodified app; do not claim saved quantitative trials, exact counts, or classifier improvement. User observations are pending; do not infer pass from opening preview. Next code task ONLY if physical failure evidence supports one bounded repair; sole existing Antigravity remains only editor.
