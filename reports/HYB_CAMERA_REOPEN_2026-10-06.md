# Hybrid camera reopen — blocked by Windows Application Control

2026-10-06 07:56 Asia/Saigon. User asks to reopen camera. Reuse previously explicitly approved Hybrid real-control mode and personal candidate, with assistOFF at startup. No code task, Worker prompt, security weakening, dependency installation or model/config change authorized/performed.

## Checks and attempted launch

No existing AirPad camera process at precheck. Eight key accepted source/safety/motion/config/candidate SHA values match prior accepted version. Old app terminal handle stale after Orca runtime restart; main-worktree terminal list empty. Created exactly ONE visible app terminal (not agent/Worker): term_bb0778bd-41a4-42ba-8d40-ce3205fff9f8, receipt43336e43-beef-43f7-a75d-27bb291499fe, runtimeaf381052-306d-4038-9b49-dfaadebf3017.

Attempted unchanged command:

    .venv\Scripts\python.exe -u -B app.py --camera --profile hybrid --real-control --model reports/acc05/candidate_personal_svm.json

Launcher11912/child10952 existed briefly, no preview window. Fresh terminalb2722b then25de40 proves app exited to PowerShell with:

    ImportError: DLL load failed while importing _solve_toeplitz: An Application Control policy has blocked this file.

Failure chain: run_camera -> tools.train.load_bundle -> src.ml.svm -> sklearn -> scipy.special -> scipy.linalg -> _solve_toeplitz. Failure occurs before real pipeline/controller creation and webcam acquisition. Do NOT report camera opened, live OS-control admission completed or recognition regression. Startup receipt/process creation alone was not treated as success.

## Independent blocker evidence

Code Integrity operational events3033/3077 at the matching launch time explicitly identify `.venv/Lib/site-packages/scipy/linalg/_solve_toeplitz.cp314-win_amd64.pyd` as failing Enterprise signing/code-integrity requirements; policy ID0283ac0f-fff1-49ae-ada1-8a933130cad6. This corroborates OS policy rejection rather than missing webcam. Exact policy origin, reason for change sinceOct2 and remediation are NOT established; do not guess Smart App Control, malware or laptop hardware failure.

Final read-only process/window audit25de40: zero AirPad app processes and zero AirPad Virtual Pad windows. App terminal retained idle; no repeated launch, automatic retry loop, alternative unsigned DLL, DLL renaming/moving, policy bypass/disable, admin elevation, Defender exclusions, credit/reset/paid fallback or account/model switch. Only Markdown checkpoint reports edited by Brain. Need a separately scoped environment/security-policy remediation decision before widening camera-only request; Antigravity remains only code editor if a code change is eventually justified. Camera CLOSED; user informed of blocker.
