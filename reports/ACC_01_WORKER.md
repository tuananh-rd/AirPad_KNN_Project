# ACC-01 Worker Implementation Report: Trustworthy Evaluation & Clean Baseline

**Worker**: Antigravity in airpad-worker  
**Task ID**: `task_b59fa9dfa606`  
**Dispatch ID**: `ctx_50b8f4408d16`  
**Coordinator Terminal**: `term_e268eedf-b7d0-463a-8c49-447c54ce01e7`  
**Worker Terminal**: `term_fbabd736-4c4f-49b1-99c9-2ae8307f76e0`  
**Date**: 2026-09-30  

---

## 1. Executive Summary

1. **What was done**: 
   - Created read-only pre-edit backups and SHA256 manifests under `reports/acc01/before/`.
   - Executed a rigorous provenance audit of `data/custom_airpad.npz`, `data/hagrid_airpad.npz`, and `data/combined_airpad.npz`, proving 1-to-1 exact correspondence, accounting for 43 duplicate feature+label keys (max multiplicity 2, all timestamp-disambiguated with 100% group agreement), and exposing synthetic chunk leakage (`tuana_chunk0..4`). Exported `reports/acc01/corrected_group_manifest.json` and `reports/acc01/audit_report.json`.
   - Modified `src/ml/evaluation.py` and `tools/train.py` to fix rejection/UNKNOWN denominator accounting (0.5 recall, 2/3 F1 on regression spec), reject missing/empty/insufficient groups, enforce zero user and real-session leakage per fold, and validate corrected manifest coverage and row labels.
   - Built the fixed-seed evaluation CLI `tools/evaluate.py` tracking original source row IDs, recording split fingerprints and evaluator source hashes, and executed both a stratified 600-sample smoke benchmark (`reports/acc01/smoke/`) and the approved full 60,000-row 5-fold baseline benchmark (`reports/acc01/full/`).
   - Implemented 18 regression tests in `tests/test_evaluation.py` (all using isolated `tmp_path` fixtures and an evaluator scaler spy); verified that all **225/225 tests pass**.

2. **What was found**:
   - The custom dataset contains exactly 1 real user (`tuana`) and 1 real session (`sess_20260929_132034_7a13ed`). Artificially splitting this into `tuana_chunk0..4` in `combined_airpad.npz` caused 100% user/session leakage across all 5 folds in historical training.
   - Clean unseen-user evaluation for the 3 custom gestures (`PINCH_CLOSED`, `PINCH_MIDDLE`, `L_SHAPE`) is strictly **UNAVAILABLE**; clean 9-class generalization cannot be evaluated without additional multi-user data.
   - Historical macro-F1 (0.967276) is confirmed leaky and cannot serve as a valid improvement baseline or comparison target.
   - The clean 6-class HaGRID baseline on 60,000 samples (9,170 real users, 5-fold GroupKFold, zero leakage) achieved **Macro-F1 = 0.9918544754844651** (post-rejection) and **Accuracy = 0.9917333333333334** with 99.98% coverage. This is strictly a 6-class baseline and must not be called a 9-class improvement.

3. **What is next / left**:
   - All code modifications, tests, audits, manifests, smoke benchmarks, and full baseline benchmarks are complete and verified green.
   - Deliverable complete; submitting final `worker_done` report to Coordinator.

---

## 2. Artifact and Manifest Verification

### 2.1 Brain Baseline Dataset and Model Hashes
All baseline hashes match the Coordinator's expected digests exactly:
- `data/custom_airpad.npz`: `498699e1f28a78031193476fbafff2a2cd307bb9d82a8baee6a57f3097019169`
- `data/hagrid_airpad.npz`: `ec38045d608aa6d30386796e4e0586ff619468d4abfd755619835bb8fa61c081`
- `data/combined_airpad.npz`: `c0c6d4e26c26f8a303077c6de971a54c2d4085444f9479cef9e021986cc9e8bf`
- `models/airpad_svm.json`: `4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977`

### 2.2 Pre-Edit Backups (`reports/acc01/before/`)
Read-only backups created prior to modifying any source code:
- `reports/acc01/before/src/ml/evaluation.py` (`d9a0b3faf732af4bab2d0242cf29c7bc26eaa355ef3aa8fd459e35e28c13f25d`)
- `reports/acc01/before/tools/train.py` (`3e5053074d89333c19ab3000c083481a55fef086e39d107fd66a398f30c0c705`)
- Manifest: `reports/acc01/before/manifest.sha256` and `reports/acc01/before/MANIFEST.md`

### 2.3 File Ownership & Changes
- **Modified Existing Files**:
  - `src/ml/evaluation.py`: Fixed UNKNOWN false negative denominator accounting, confusion matrix UNKNOWN column, unexpected label rejection, and strict group validation.
  - `tools/train.py`: Fixed `extract_groups` to fail clearly on missing/empty/None/NaN values; required sufficient groups in `tune_svm` and `train_from_dataset`.
- **Created New Files**:
  - `tools/audit/__init__.py`: Package init.
  - `tools/audit/provenance.py`: Provenance audit, multiplicity accounting, and corrected group manifest generator.
  - `tools/evaluate.py`: Clean evaluation CLI with source row ID tracking, session/user leakage assertions, split fingerprinting, manifest validation, and synthetic group rejection.
  - `tests/test_evaluation.py`: 18 comprehensive regression tests (using `tmp_path` and scaler spy).
- **Unified Diff**:
  - `reports/acc01/before_after.diff` (1,628 lines, unified diff of all modifications and additions).

---

## 3. Provenance and Multiplicity Audit Results

Generated via `python tools/audit/provenance.py`:
- **HaGRID Segment (rows 0 to 59,999)**: 100% exact match in features, labels, and original metadata (`timestamp`, `handedness`, `label`) to `data/hagrid_airpad.npz`. 9,170 unique users, 29,893 sessions across 6 classes.
- **Custom Segment (rows 60,000 to 61,065)**: 100% exact match in features, labels, and non-group metadata (`timestamp`, `handedness`, `label`, `distance_cm`, `lighting`, `is_transition`, `feature_version`) to `data/custom_airpad.npz`. 1 real user (`tuana`), 1 real session (`sess_20260929_132034_7a13ed`).
- **Multiplicity Accounting**:
  - Exactly 43 duplicate feature+label keys detected in `custom_airpad.npz`.
  - Max multiplicity: 2.
  - All 43 pairs are fully disambiguated by timestamp.
  - 100% of candidate duplicate matches agree on the identical user (`tuana`) and session (`sess_20260929_132034_7a13ed`).
- **Leakage Analysis**:
  - In `combined_airpad.npz`, custom rows were artificially assigned to `tuana_chunk0..4`.
  - In a 5-fold CV using these chunk IDs, 100% of folds (5/5) leak the real user and real session between train and test.
  - When evaluated using true groups, the 1 real user cannot be partitioned across 5 folds (`ValueError: Insufficient unique groups: found 1 unique groups`).
- **Audit Deliverables**:
  - `reports/acc01/corrected_group_manifest.json`: Authoritative row-by-row mapping of all 61,066 samples with true IDs, source dataset, preserved metadata, and leak flags.
  - `reports/acc01/audit_report.json`: Detailed summary statistics and verification status.

---

## 4. Evaluation Engine Implementation & Verification

### 4.1 Metric Accounting
- `compute_metrics`:
  - When predictions contain `UNKNOWN` (rejection), they are treated as false negatives in the full denominator:
    $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad \text{where } FN = \sum_{j \ne i} cm[i, j] + cm[i, \text{UNKNOWN}]$$
  - Tested on `[POINT_ONE, POINT_ONE]` vs `[POINT_ONE, UNKNOWN]` with `labels=[POINT_ONE]`:
    - `recall = 0.5000`
    - `f1 = 2/3 = 0.6667`
    - `support = 2`
    - `coverage = 0.5000`
    - `rejection_rate = 0.5000`
    - `confusion_matrix = [[1, 1]]` (columns: `['POINT_ONE', 'UNKNOWN']`)
  - UNKNOWN is never added as an extra macro target class.

### 4.2 Group Extraction & Fold Isolation
- `extract_groups`: Rejects missing `group_key`, `None`, empty string, whitespace, or `NaN`.
- `group_kfold_split`: Rejects `unique_groups < n_splits` and missing group elements.
- `tools/evaluate.py`:
  - Preserves exact source dataset row IDs (`selected_source_indices`, `train_source_row_ids`, `test_source_row_ids`).
  - Asserts that every evaluated source row appears exactly once in the test set across all folds:
    `assert sorted(all_tested_source_ids) == sorted(selected_source_indices)`
  - Explicitly asserts zero user leakage (`train_users.isdisjoint(test_users)`) AND zero session leakage (`train_sessions.isdisjoint(test_sessions)`) on every fold.
  - Computes and exports deterministic split fingerprint:
    `split_fingerprint = compute_split_fingerprint(fold_test_source_ids)`
  - Validates corrected manifest: checks hash binding, exact record count, unique complete sample_id coverage, and row label matching.
  - Requires >= 2 real users per target class; rejects custom classes with single user.
  - Asserts no absent training classes per fold.

---

## 5. Benchmark Results

### 5.1 Stratified Smoke Benchmark (`reports/acc01/smoke/`)
- Command: `python tools/evaluate.py --sample-limit 600 --output-dir reports/acc01/smoke`
- Evaluator Source Hash: `fa9995adcda6ec42...`
- Split Fingerprint: `fe9aaed09f701e139befa96668d851233e7c88e6d4df7e6f951f28029072f026`
- Samples: 600 (100 per class, 283 unique users).
- **Exact Smoke Results**:
  - Macro-F1 (post-rejection): **0.9816312779659467**
  - Accuracy (post-rejection): **0.9816666666666667**
  - Balanced Accuracy: **0.9816666666666668**
  - Coverage: **1.0000** (0 rejections)
  - Raw Macro-F1 (pre-rejection): **0.9816312779659467**
  - Artifacts: `reports/acc01/smoke/split_manifest.json`, `reports/acc01/smoke/predictions.jsonl`, `reports/acc01/smoke/clean_baseline_metrics.json`.

---

### 5.2 Full 60,000-Row Five-Fold HaGRID Baseline Benchmark (`reports/acc01/full/`)
- Command: `python tools/evaluate.py --dataset data/hagrid_airpad.npz --output-dir reports/acc01/full`
- Evaluator Source Hash: `fa9995adcda6ec42...`
- Split Fingerprint: `9f8d1131ecaee79bf3bff3a52090adac9ff083fa59bd98fc4ad8bd552c8afe7e`
- Dataset: `data/hagrid_airpad.npz` (60,000 samples, 10,000 per class, 9,170 unique users, 29,893 sessions).
- Protocol: 5-fold GroupKFold by `user_id`, seed 42. Strict zero user and session leakage.
- Scaler: `StandardScaler` fit exclusively on train fold (48,000 samples/fold).
- Classifier: `SVC(C=10.0, gamma=0.01, class_weight='balanced')`.
- Rejection Policy: `decision_threshold=0.0`, `margin_threshold=0.15`.
- Artifacts: `reports/acc01/full/split_manifest.json`, `reports/acc01/full/predictions.jsonl`, `reports/acc01/full/clean_baseline_metrics.json`.

#### Full Benchmark Aggregate Metrics
| Metric | Post-Rejection (Clean Baseline) | Pre-Rejection (Raw) |
|---|---|---|
| **Macro-F1** | **0.9918544754844651** | 0.9917306463380989 |
| **Accuracy** | **0.9917333333333334** | 0.9916000000000000 |
| **Balanced Accuracy** | **0.9917333333333334** | 0.9916000000000000 |
| **Coverage** | **0.9997500000000000** | 1.0000000000000000 |
| **Rejection Rate** | **0.0002500000000000** (15 rejected) | 0.0000000000000000 |
| **Total Support** | **60,000** | 60,000 |

#### Full Benchmark Per-Class Performance (Post-Rejection)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **FIST** | 0.9889 | 0.9923 | 0.9906 | 10,000 |
| **OPEN_PALM** | 0.9936 | 0.9991 | 0.9964 | 10,000 |
| **OTHER** | 0.9875 | 0.9837 | 0.9856 | 10,000 |
| **POINT_ONE** | 0.9925 | 0.9902 | 0.9913 | 10,000 |
| **THREE_FINGERS** | 0.9967 | 0.9927 | 0.9947 | 10,000 |
| **TWO_FINGERS_PARALLEL** | 0.9927 | 0.9924 | 0.9925 | 10,000 |

#### Per-Fold Stability Summary
| Fold | Train Samples | Test Samples | Clean Macro-F1 | Clean Accuracy | Rejections | User Leakage | Session Leakage |
|---|---|---|---|---|---|---|---|
| Fold 1 | 48,036 | 11,964 | 0.9909 | 0.9908 | 3 | 0 | 0 |
| Fold 2 | 48,006 | 11,994 | 0.9914 | 0.9913 | 4 | 0 | 0 |
| Fold 3 | 47,838 | 12,162 | 0.9932 | 0.9931 | 1 | 0 | 0 |
| Fold 4 | 47,938 | 12,062 | 0.9917 | 0.9915 | 4 | 0 | 0 |
| Fold 5 | 48,182 | 11,818 | 0.9922 | 0.9920 | 3 | 0 | 0 |

---

## 6. Custom Poses & 9-Class Blockers / Unsupported Metrics

| Dimension | Custom Poses (`PINCH_CLOSED`, `PINCH_MIDDLE`, `L_SHAPE`) | HaGRID Poses (6 classes) |
|---|---|---|
| Real Users | **1** (`tuana`) | **9,170** |
| Real Sessions | **1** (`sess_20260929_132034_7a13ed`) | **29,893** |
| Unseen-User CV | **UNAVAILABLE** (Cannot form 5 disjoint folds) | **AVAILABLE** (5-fold clean CV) |
| Clean 9-Class Claim | **STRICTLY BLOCKED** (Lack of multi-subject data) | N/A |
| Historical Leaky Macro-F1 | 0.967276 (leaked across folds via chunks) | Do not compare directly |

**Required Remediation**: Multi-user, multi-session recording for the 3 custom gestures is an absolute prerequisite before claiming clean 9-class unseen-user performance.

---

## 7. Full Test Suite Status

Command: `.\.venv\Scripts\python.exe -m pytest`
- **Result**: **225 passed in 71.17s** (207 existing tests + 18 new regression tests).
- All tests use isolated `tmp_path` fixtures to protect review artifacts.
- Evaluator scaler spy confirms train-fold-only fitting across all folds.
- Zero regressions across existing pipeline, FSM, geometry, KNN, SVM, or data tools.

---

## 8. Summary of Generated Review Artifacts

```
reports/
├── ACC_01_WORKER.md
└── acc01/
    ├── before/
    │   ├── MANIFEST.md
    │   ├── manifest.sha256
    │   ├── src/ml/evaluation.py
    │   └── tools/train.py
    ├── before_after.diff
    ├── corrected_group_manifest.json
    ├── audit_report.json
    ├── smoke/
    │   ├── split_manifest.json
    │   ├── predictions.jsonl
    │   └── clean_baseline_metrics.json
    └── full/
        ├── split_manifest.json
        ├── predictions.jsonl
        └── clean_baseline_metrics.json
```
