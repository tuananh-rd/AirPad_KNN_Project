# ACC-08 Coordinator review — 2026-10-01

Outcome: ACCEPTED for restarting the already-authorized, bounded real-control trial. Only Antigravity edited code. Coordinator inspected source/diffs, requested same-task corrections, independently ran tests, and wrote this review. No new code task or model promotion.

## Evidence and scope

Before: independent dry-run direction assertion failed (preview x .35→.65 mapped screen x 1319→698, mirror True). The camera flips images before detection; the generic mapper then mirrored x again.

After: reviewed no-index diff against exact ACC-07 originals (workspace is not a Git repository). app.py adds deepcopy of camera-local configuration, sets mirror_x=False and passes that copy into camera pipeline/ROI rendering. Generic create_pipeline, replay, RGB conversion, BGR preview, features, detector, FSM, model and safety thresholds unchanged.

Independent full command:

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests reports/acc04/test_repair_validation.py reports/acc05/test_acc05_benchmark.py reports/acc07/test_acc07_dryrun.py
```

Execution 10843: **330 passed in 311.96s**, exit 0. Prior baseline: **327 passed in 92.32s**; durations are not comparable performance benchmarks (concurrent Worker testing). Self-check independently **5/5**, exit 0.

Three added regressions cover isolated loading of saved original camera code, real mapper/temporal filters/FSM/DryRun MOVE deltas for right/left/down/up, and exact input-config object non-mutation with generic mirror True/False mapping preserved. Classifier is stubbed for direction; rate limiter is instrumented only in this focused test. These prove coordinate behavior, NOT physical-camera recognition accuracy or real OS release safety. Phase slices deliberately omit the first reversal frame while the default motion filter settles; Worker report wording “every event” in left/up phases is broader than these assertions. Saved mapper sample values also depend on timestamps/filter state; sign assertions are the acceptance evidence, not report pixel numerals.

Compared all 87 pre-task project/evidence hashes (excluding environment packages): only app.py and tests/test_app.py changed; other 85 entries unchanged. Final hashes:

- app.py: B4852C0483A0D89AABBE973ECDDFE3AC8AD4295BEE222DDDF3D2A6880CE55DC6
- tests/test_app.py: BBF787F46E28900B535BAD1F772BDCC7E3D61C3110BD82BA9CBFFC7105FD3A14
- new guide: A61E0AF0A40A096F77668D21FF3E3CDAF4D784C15018522979508746925E2BE1
- candidate: 66091D52D9DD6848FD8E35FC2641A31C4C2FFC9008329A4D2E9F85F8DD1246BE (unchanged)
- production: 4CC3C55F5E4B571C5462DE31AFD6D5C4DB64C803466D306CDEFAB34B31D7B977 (unchanged)

## Documentation and limitations

Reviewed Vietnamese guide against actual FSM, pipeline, scroll and platform behavior; required corrections to initial reversed scroll instructions, temporal-filter caveats, arming, emergency-stop latch/restart, application-dependent zoom and explicit candidate commands. The camera app explicitly constructs a 15px mapped-coordinate corner monitor (class constructor default 10px); it does not monitor the physical cursor. Native PyAutoGUI failsafe remains disabled as before. Hotkey registration and OS cleanup are best-effort, not hardware safety guarantees.

Current scroll convention remains hand-down→wheel-up, hand-up→wheel-down; no unrelated scroll code changed. FIST is not a global pause command. OPEN_PALM has temporal recognition delay. Q/Esc requires preview focus; Ctrl+Alt+Q is latched until restart.

Previous paired held-out accuracy remains candidate 1523/1663 (91.5815%) versus baseline 417/1663 (25.0752%); prior live-shadow stable-target result remains 814/905 (89.9448%) versus 253/905 (27.9558%). No new recognition benchmark or generalization claim from this camera-coordinate patch. User-specific one-user data limitation remains.

Worker ACC08_DIRECT_DONE report inspected; final documentation-only latch amendment also requested. Before camera restart, process enumeration found no project camera/pytest process, only the Antigravity helper. Retained sole Worker terminal; reclaimable-worker query empty, no new lifecycle authority or worker created. Next step is the already-authorized empty-Notepad physical trial and user direction feedback, not a new architecture/code task.

Final amended ACC08_DIRECT_DONE and idle prompt observed. Global emergency-hotkey register/unregister preflight returned registered=True, exit0. Restarted exactly one existing-app trial, execution11582: `.venv/Scripts/python.exe -u -B app.py --camera --real-control --model reports/acc05/candidate_personal_svm.json`. Real-control warning and live-camera-start banner confirmed; process intentionally remains running for the user's authorized trial. User instructed to keep hand out before selecting empty Notepad and use Ctrl+Alt+Q to stop. Physical direction feedback still pending; startup is not proof of user-task success or recognition accuracy.
