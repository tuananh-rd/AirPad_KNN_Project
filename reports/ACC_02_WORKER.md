# ACC-02 Worker Implementation Report: Known-Cohort Session Readiness & Collector Safeguards

**Worker**: Antigravity in airpad-worker  
**Task ID**: `task_8d7f4243f92c`  
**Dispatch ID**: `ctx_0571b9f0320a`  
**Coordinator Terminal**: `term_e268eedf-b7d0-463a-8c49-447c54ce01e7`  
**Worker Terminal**: `term_fbabd736-4c4f-49b1-99c9-2ae8307f76e0`  
**Date**: 2026-10-01  

---

## 1. Executive Summary

1. **What was done**:
   - Fixed `tools/collect_data.py` to fail closed with a nonzero exit code on corrupt or unreadable existing output files before camera initialization, save routines, or mock processing, preserving file bytes and hashes intact while maintaining valid append backward compatibility and accurately displaying all 9 pose hotkeys (`[1]`..`[9]`).
   - Implemented `tools/cohort_sessions.py` enforcing strict pair-keyed `(user_id, session_id)` assignments, disallowing delimiter `::` within identities, requiring explicit participant roster and session-role plans for `READY` (rejecting heuristic auto-assignment/roster inference), and implementing centralized validation across public helpers (`validate_cohort_sessions` and `generate_split_manifest`).
   - Hardened integrity guards: strictly rejects JSON duplicate keys via `object_pairs_hook`, rejects output report and manifest paths aliasing ANY input dataset or each other before any writes (using `normcase(realpath)` and `samefile` to protect Windows case aliases and hardlinks), enforces unconditional mandatory provenance validation for every sample (nonblank `source_file`, non-negative integer not bool `source_row_idx`, 64-hex-char `source_hash`, and source map coverage), and ensures split manifests are never emitted when the dataset is `NOT_READY` (with manifest computation and validation occurring in memory before writing either output).
   - Enforced safe session-split evaluation rules: minimum 3 independent disjoint sessions per participant across `train`, `validation`, and `test` roles, single-user personal-use readiness (without imposing the $\ge 2$-users cohort rule), mandatory stable-sample coverage of all 9 static target classes *per participant per role* (preventing transition-only participants or cross-user coverage masking), explicit rejection and diagnostic reporting of extra unrostered participants, separate tracking and exclusion of transition frames (`is_transition=True`) from default stable lists into a dedicated transition list, mandatory source row IDs and source hashes, and deterministic fingerprinting excluding wall-clock timestamps.
   - Audited existing custom data (`data/custom_airpad.npz`) using `tools/cohort_sessions.py` with an explicit train plan (`reports/acc02/plan_existing_custom.json`), generating `reports/acc02/readiness_existing_custom.json` which authoritatively declared `NOT_READY` due to having only 1 session, missing 6 of 9 target classes, and having zero sessions in validation and test roles.
   - Authored the comprehensive Vietnamese collection guide in `docs/KNOWN_COHORT_COLLECTION.md` using standard `python -m tools...` module invocation, safe PowerShell array expansion for multiple files, pair-keyed plan examples, and accurate guidance on handedness, privacy, and readiness verification.
   - Implemented 35 regression tests in `tests/test_cohort_sessions.py` with isolated `tmp_path` fixtures; verified that all **260/260 tests pass** (225 legacy + 35 new) and all original dataset and model SHA-256 hashes remain identical.

2. **What was found**:
   - The existing collector previously caught exceptions on corrupt files, printed a warning, and silently overwrote existing files with single frames. The fail-closed repair guarantees zero accidental data loss or corruption overwriting.
   - The existing dataset `data/custom_airpad.npz` has exactly 1 real session (`sess_20260929_132034_7a13ed`) and 3 classes (`PINCH_CLOSED`, `PINCH_MIDDLE`, `L_SHAPE`). It cannot satisfy independent session splitting across 3 roles or 9-class coverage, confirming that new genuine collection per `docs/KNOWN_COHORT_COLLECTION.md` is required before known-cohort or personal-use evaluation can proceed.
   - Artificially partitioning single sessions into pseudo-independent folds is strictly rejected by `tools/cohort_sessions.py`.

3. **What is next / left**:
   - All code changes, collector safeguards, cohort readiness tools, audit reports, documentation, tests, and diffs are complete and verified.
   - No classifier, evaluator, or runtime architecture modifications were performed.
   - Ready for Coordinator review; submitting `worker_done`.

---

## 2. Baseline and Pre-Edit Artifact Verification

### 2.1 Brain Baseline Dataset and Model Hashes
All baseline hashes were verified prior to and following implementation; zero bytes changed:
- `data/custom_airpad.npz`: `498699e1f28a78031193476fbafff2a2cd307bb9d82a8baee6a57f3097019169`
- `data/hagrid_airpad.npz`: `ec38045d608aa6d30386796e4e0586ff619468d4abfd755619835bb8fa61c081`
- `data/combined_airpad.npz`: `c0c6d4e26c26f8a303077c6de971a54c2d4085444f9479cef9e021986cc9e8bf`
- `models/airpad_svm.json`: `4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977`

### 2.2 Pre-Edit Backups (`reports/acc02/before/`)
Read-only backups created prior to modifying any source code:
- `reports/acc02/before/tools/collect_data.py` (`9c90d5acf46768e25d9054463478945cd1381d5a0b1b525f1e72082277f7bd6f`)
- Manifest: `reports/acc02/before/manifest.sha256` and `reports/acc02/before/MANIFEST.md`

---

## 3. Collector Safeguards (`tools/collect_data.py`)

### 3.1 Fail-Closed on Corrupt/Unreadable Datasets
In `tools/collect_data.py`:
- Previously, lines 368–376 caught all exceptions when attempting to load an existing `.npz` file, printed `"[WARN] Could not load existing dataset: {e}. Starting fresh."`, and subsequently overwrote the file upon recording or saving.
- The new implementation fails closed immediately:
  ```python
  if os.path.exists(output_path):
      try:
          ex_feat, ex_lbl, ex_meta = load_dataset_npz(output_path)
          collected_features.extend([ex_feat[i] for i in range(len(ex_feat))])
          collected_labels.extend(ex_lbl.tolist())
          collected_metadata.extend(ex_meta)
          print(f"[INFO] Loaded existing dataset with {len(ex_lbl)} samples from {output_path}")
      except Exception as e:
          print(f"[ERROR] Existing dataset at {output_path} is corrupt or unreadable: {e}", file=sys.stderr)
          if raise_on_error:
              raise ValueError(f"Existing dataset at {output_path} is corrupt or unreadable: {e}") from e
          return 1
  ```
- This check executes before any mock frame processing, camera opening (`cv2.VideoCapture`), or disk write operations. The corrupt file remains bit-for-bit identical.

### 3.2 Display of All 9 Pose Hotkeys
The HUD and console control text in `tools/collect_data.py` now accurately lists all 9 target poses:
- `[1] PINCH_CLOSED`, `[2] PINCH_MIDDLE`, `[3] L_SHAPE`
- `[4] OTHER`, `[5] POINT_ONE`, `[6] OPEN_PALM`
- `[7] FIST`, `[8] TWO_FINGERS_PARALLEL`, `[9] THREE_FINGERS`

### 3.3 Valid Append Backward Compatibility
When `output_path` points to a valid existing dataset, existing samples, labels, and metadata are cleanly loaded, and new frames captured during the session are appended without data loss.

---

## 4. Cohort Session Readiness Utility (`tools/cohort_sessions.py`)

### 4.1 Strict Pair-Keyed Assignments
Sessions are identified by `(user_id, session_id)` pairs, preventing collisions when two users share identical session naming conventions (e.g. `user_A::sess_01` in train vs `user_B::sess_01` in test). Identities containing `::` are strictly rejected.

### 4.2 Centralized Validation for Both Public Helpers
`validate_samples_integrity` provides a single centralized validation routine used by both `validate_cohort_sessions` and `generate_split_manifest`. It enforces:
- Strict sample count equality between features, labels, and metadata.
- Identity validation: non-blank, non-null string without `::` delimiter.
- Exact agreement between label array and metadata label (`labels[idx] == meta["label"]`), rejecting mismatches.
- Mandatory boolean transition flag (`isinstance(meta["is_transition"], bool)`).
- Unconditional mandatory provenance: non-blank `source_file`, non-negative integer (not bool) `source_row_idx`, 64-char hexadecimal `source_hash`, and duplicate `(source_file, source_row_idx)` detection.
- Complete plan validation: all sessions in dataset must be assigned; all sessions in plan must exist in dataset; all roles must be valid.

### 4.3 Participant-Level Stable Representation and Coverage
- Each roster participant must have stable samples in each role; participants with only transition frames are rejected.
- Each roster participant must cover all 9 static target classes in stable samples for each role; cross-participant masking is prohibited.
- Extra unrostered participants in the dataset are diagnosed and excluded from readiness totals.

### 4.4 Output Path Alias Guards & JSON Duplicate Rejection
- Before reading or writing files, `validate_output_paths_no_alias` verifies that output report and manifest paths do not resolve to any input dataset file or to each other (using `normcase(realpath)` and `samefile`).
- All JSON parsing utilizes `object_pairs_hook=strict_json_object_pairs_hook` to reject duplicate keys.
- Split manifests are never emitted if status is `NOT_READY`. When requested on a `READY` dataset, manifests are computed and validated in memory before writing either the report or manifest. Direct calls to `generate_split_manifest` on `NOT_READY` input strictly raise `ValueError`.

---

## 5. Authoritative Audit of Existing Custom Data

Running `.venv/Scripts/python.exe -m tools.cohort_sessions --data data/custom_airpad.npz --roster tuana --mode personal --plan reports/acc02/plan_existing_custom.json --output reports/acc02/readiness_existing_custom.json` produced:

```json
{
  "status": "NOT_READY",
  "mode": "personal",
  "target_classes": [
    "PINCH_CLOSED", "PINCH_MIDDLE", "L_SHAPE", "OTHER",
    "POINT_ONE", "OPEN_PALM", "FIST", "TWO_FINGERS_PARALLEL", "THREE_FINGERS"
  ],
  "roster": ["tuana"],
  "total_samples": 1066,
  "stable_samples": 1066,
  "transition_samples": 0,
  "source_files": {
    "data/custom_airpad.npz": "498699e1f28a78031193476fbafff2a2cd307bb9d82a8baee6a57f3097019169"
  },
  "missing_criteria": [
    "Participant 'tuana' has only 1 session(s) (['sess_20260929_132034_7a13ed']); minimum 3 independent sessions required (at least 1 each for train, validation, test).",
    "Dataset contains only 3 of 9 target classes: missing ['FIST', 'OPEN_PALM', 'OTHER', 'POINT_ONE', 'THREE_FINGERS', 'TWO_FINGERS_PARALLEL']",
    "Participant 'tuana' in role 'train' missing stable samples for classes: ['OTHER', 'POINT_ONE', 'OPEN_PALM', 'FIST', 'TWO_FINGERS_PARALLEL', 'THREE_FINGERS']",
    "Role 'validation' missing sessions from participant 'tuana'",
    "Role 'test' missing sessions from participant 'tuana'"
  ]
}
```

The audit explicitly confirms:
- Only 1 session exists (`sess_20260929_132034_7a13ed`).
- Only 3 of 9 static target classes exist.
- Disjoint 3-role assignment is mathematically impossible without new sessions.
- No artificial re-chunking was performed.

---

## 6. Vietnamese Collection Guide (`docs/KNOWN_COHORT_COLLECTION.md`)

The Vietnamese collection guide in `docs/KNOWN_COHORT_COLLECTION.md` details:
- **Module execution syntax**: Standard `.venv/Scripts/python.exe -m tools.collect_data` and `.venv/Scripts/python.exe -m tools.cohort_sessions` ensuring `src` is always on Python module path.
- **PowerShell array expansion**: Safe multi-file handling avoiding Windows wildcard expansion pitfalls:
  ```powershell
  $dataFiles = (Get-Item data/cohort/*.npz | Select-Object -ExpandProperty FullName)
  .\.venv\Scripts\python.exe -m tools.cohort_sessions --data $dataFiles --roster tuana,friend_01 --mode cohort --plan plan_cohort.json --output reports/cohort_readiness_group.json --manifest-out reports/manifest_cohort.json
  ```
- **Explicit pair-keyed plan format**: Documentation and examples using `user_id::session_id` pair keys.
- **Handedness and sample planning**: Handedness guidance focused on application needs (single vs two-handed use); explicit disclaimer that sample counts are planning guidelines rather than accuracy guarantees.
- **Privacy guarantees**: Strictly local landmark storage without video recording.

---

## 7. Test Suite & Verification Results

### 7.1 Regression Tests (`tests/test_cohort_sessions.py`)
35 unit tests were implemented across four test classes:
- **TestCollectDataSafeguards**:
  - `test_corrupt_file_fail_closed_no_overwrite`: Nonzero exit code, error logging, bit-for-bit unchanged hash on corrupt files.
  - `test_valid_append_backward_compatibility`: Sequential sessions append correctly.
- **TestCentralizedIntegrityValidation**:
  - `test_direct_validate_cohort_sessions_label_mismatch_raises`: In-memory label mismatch raises ValueError.
  - `test_direct_validate_cohort_sessions_non_boolean_transition_raises`: Non-boolean is_transition raises ValueError.
  - `test_direct_identity_containing_delimiter_raises`: Identity with `::` raises ValueError.
  - `test_absent_source_provenance_raises`: Absent source provenance fields raise ValueError.
  - `test_non_hex_source_hash_raises`: Non-hexadecimal source hash raises ValueError.
  - `test_bool_source_row_idx_raises`: Boolean source row index raises ValueError.
  - `test_direct_generate_split_manifest_rejects_not_ready_input`: Rejects NOT_READY input directly.
  - `test_json_duplicate_keys_rejected`: Duplicate JSON object key raises ValueError.
  - `test_output_path_aliasing_rejected`: Input dataset path aliasing (including case differences) and identical output paths raise ValueError.
- **TestIntegrityViolations**:
  - `test_duplicate_input_files_by_path`: Raises ValueError on duplicate input paths.
  - `test_duplicate_input_files_by_hash`: Raises ValueError on duplicate file content.
  - `test_duplicate_raw_session_ingestion_across_files`: Raises ValueError when raw session appears in multiple files.
  - `test_duplicate_roster_entries`: Raises ValueError on duplicate roster participants.
  - `test_blank_or_null_roster_entries`: Raises ValueError on blank/null roster entries.
  - `test_unknown_role_key`: Raises ValueError on invalid role strings.
  - `test_duplicate_session_assignment_even_same_role`: Raises ValueError on duplicate assignments.
  - `test_unknown_session_in_plan`: Raises ValueError when plan references nonexistent data session.
  - `test_unassigned_session_in_dataset`: Raises ValueError when dataset contains unassigned session.
- **TestCohortSessionRules**:
  - `test_pair_keyed_assignments_two_users_sharing_session_name`: Verifies same session text across different users assigned to different roles.
  - `test_no_roster_or_no_plan_results_in_diagnostic_not_ready_never_auto_ready`: Verifies absent inputs produce NOT_READY diagnostic.
  - `test_manifest_not_emitted_on_not_ready`: Verifies manifest file is never written when status is NOT_READY.
  - `test_personal_use_acceptance_three_sessions_nine_classes`: Confirms `READY` status for 1 user with 3 sessions and 9 classes.
  - `test_single_user_rejection_insufficient_sessions`: Confirms `NOT_READY` when user has fewer than 3 sessions.
  - `test_multi_user_cohort_acceptance`: Confirms `READY` status for multi-user cohort.
  - `test_cohort_mode_rejection_missing_roster_participant`: Rejects datasets missing roster members.
  - `test_missing_target_classes_rejection`: Rejects datasets with missing static classes.
  - `test_transition_frames_excluded_from_stable_counts`: Verifies separate transition counts.
  - `test_existing_custom_dataset_is_not_ready`: Authoritative test verifying `data/custom_airpad.npz` produces `NOT_READY`.
  - `test_transition_only_friend_in_cohort_is_not_ready`: Verifies participant with only transition frames produces NOT_READY.
  - `test_friend_missing_pose_in_role_is_not_ready`: Verifies participant missing a pose in a role produces NOT_READY.
  - `test_extra_unrostered_participant_diagnostic`: Verifies extra unrostered participants are diagnosed.
- **TestManifestAndFingerprint**:
  - `test_manifest_provenance_and_transition_separation`: Verifies manifest output structure and separation of stable vs transition lists.
  - `test_fingerprint_deterministic_and_excludes_timestamps`: Confirms deterministic fingerprint without timestamps.

### 7.2 Full Test Suite Execution
- **Command**: `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider`
- **Result**: **260 passed in 84.68s** (225 legacy + 35 new, 0 failures, 0 regressions).

---

## 8. Summary of Modified and Created Files

| File Path | Action | Description |
| :--- | :---: | :--- |
| `tools/collect_data.py` | Modified | Fail closed on corrupt files before camera/save; accurate 9 hotkey HUD. |
| `tools/cohort_sessions.py` | Created | Offline cohort-session readiness and assignment validation utility with centralized validation, alias guards, and manifest safety. |
| `docs/KNOWN_COHORT_COLLECTION.md` | Created | Vietnamese guide for known-cohort and personal webcam collection. |
| `tests/test_cohort_sessions.py` | Created | 35 comprehensive regression tests for ACC-02 safeguards and readiness. |
| `reports/acc02/plan_existing_custom.json` | Created | Explicit pair-keyed session plan assigning existing custom session to train. |
| `reports/acc02/readiness_existing_custom.json` | Created | Authoritative audit artifact declaring existing custom dataset NOT_READY. |
| `reports/acc02/before/tools/collect_data.py` | Created | Read-only pre-edit backup of `tools/collect_data.py`. |
| `reports/acc02/before/manifest.sha256` | Created | Pre-edit checksum manifest. |
| `reports/acc02/before/MANIFEST.md` | Created | Pre-edit documentation manifest. |
| `reports/acc02/before_after.diff` | Created | Unified diff of all modifications and additions for ACC-02. |
| `reports/ACC_02_WORKER.md` | Created | This implementation report. |
