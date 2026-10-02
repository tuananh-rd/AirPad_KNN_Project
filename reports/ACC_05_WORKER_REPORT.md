# ACC-05 Worker Report: Single Frozen Personal SVM Paired Benchmark

**Author**: Antigravity Worker (User-Authorized Direct Terminal Fallback under Coordinator Supervision)  
**Date**: 2026-10-01  
**Task ID**: ACC-05  
**Protocol Document**: `reports/ACC_05_PROTOCOL.md` (SHA-256: `ff0301ac6b37641f38c021b16897827a5f4b56acbb7bced61f30d88d47c93061`)

---

## 1. Executive Summary

Executed the ACC-05 single frozen personal SVM paired benchmark comparing the production baseline (`models/airpad_svm.json`) against a candidate personal SVM trained strictly on 1692 post-RGB train frames with a train-only fitted `StandardScaler`, evaluated on 1746 validation rows (clean + retake) and 1663 held-out test rows. Following an initial preliminary in-memory run via an ad-hoc `python -c` command, the formal benchmark script `reports/acc05/benchmark_personal_svm.py` was authored to serialize and hash the candidate bundle before generating reproducible predictions and metrics exports. On held-out test data, the candidate achieved 91.58% accuracy and 91.37% macro-F1 (an increase of +66.03 percentage points raw accuracy over baseline raw 25.56% [425/1663, F1 16.42%], and +66.51 percentage points rejection accuracy over baseline rejection 25.08% [417/1663, F1 16.31%]), meeting all primary and secondary protocol gates while reducing offline OTHER-to-target frame misclassifications from 115 to 33 (-71.3%), though a residual risk of 33/191 (17.28%) OTHER frames predicted as `OPEN_PALM` and lower pinch recalls (`PINCH_CLOSED` 64.60%, `PINCH_MIDDLE` 68.94%) remain. The test suite passes completely (285/285 tests passing), leaving production model promotion for subsequent coordinator determination.

---

## 2. Input Integrity & Cryptographic Checksums

All immutable inputs were verified with SHA-256 hashes prior to benchmark execution. No input file was modified or overwritten:

| Input Artifact | Description | Expected / Verified SHA-256 Hash | Status |
|---|---|---|---|
| `reports/ACC_05_PROTOCOL.md` | Frozen benchmark protocol | `ff0301ac6b37641f38c021b16897827a5f4b56acbb7bced61f30d88d47c93061` | MATCH |
| `data/cohort/tuana_train_rgb_20261001_110223_8ff3ed.npz` | Training set (1692 rows) | `0f9d21b5b1a5ccab0ffc8f555d7c00e14b26fa2299040b9ab587b236e7d1014e` | MATCH |
| `data/cohort/curated/acc04_validation_clean.npz` | ACC-04 clean validation (1331 rows) | `e0a27d1bd989a1ac1fe9fccc90c9ae57a47a8ca0637f1959cab91a4fcc99000d` | MATCH |
| `data/cohort/tuana_validation_retake_20261001_113202_dfcd6b.npz` | ACC-04 retake validation (415 rows) | `fe26b606921386a0f640dc958e4e7ebf01e43ce0d6cd6f07e693fce517db8948` | MATCH |
| `data/cohort/tuana_test_rgb_20261001_115837_5741bd.npz` | Held-out test set (1663 rows) | `4ec1cf2dfbdcd74b5c22c914a31c52bf2fd3c11d5c3a4de5a49368eb9cca28fb` | MATCH |
| `models/airpad_svm.json` | Production baseline model bundle | `4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977` | MATCH |
| `reports/acc04/validation_cleaning_audit.json` | ACC-04 validation repair audit | `4772a95cca9c08937057775223afd7b83c2ffab6f6f39c09cb43f7f874f98ff0` | MATCH |

---

## 3. Session Roster, Manifest & Splitting

Using `tools.cohort_sessions`, a formal session plan (`reports/acc05/plan_acc05.json`) mapped the four dataset partitions using their exact session identifiers:
- **Train**: `tuana::tuana_train_rgb_20261001_110223_8ff3ed` (1692 samples)
- **Validation**: `tuana::tuana_validation_rgb_20261001_112128_2a3fea` (1331 clean samples via `acc04_validation_clean.npz`) + `tuana::tuana_validation_retake_20261001_113202_dfcd6b` (415 retake samples) = 1746 samples
- **Test**: `tuana::tuana_test_rgb_20261001_115837_5741bd` (1663 samples)

Key cohort attributes:
- **Participant Scope**: All sessions originate from user `tuana`. The partitions are **session-disjoint** across distinct temporal capture blocks, but **not user-disjoint**.
- Total cohort samples: **5101** across 4 temporal sessions.
- Transition frames: **0** across all partitions.
- Readiness status: **`READY`** recorded in `reports/acc05/readiness_acc05.json`.
- Manifest fingerprint: **`3c3ca26a9bda2e46fc698512ee78c35a7a47937e04383b87642626acc676e1fe`** recorded in `reports/acc05/manifest_acc05.json`.

---

## 4. Candidate Model Specification, Restoration & Anti-Leakage Controls

1. **Model Configuration**:
   - Architecture: Radial Basis Function (RBF) Support Vector Machine (`sklearn.svm.SVC`).
   - Hyperparameters (fixed by protocol, zero grid search or tuning): $C=10.0, \gamma=0.01$, `class_weight='balanced'`, `probability=False`.
   - Feature Preprocessing: `sklearn.preprocessing.StandardScaler` fit **exclusively** on the 1692 train rows.
   - Target Classes: Exactly 9 static classes matching script `STATIC_CLASSES`:
     `["PINCH_CLOSED", "PINCH_MIDDLE", "L_SHAPE", "OTHER", "POINT_ONE", "OPEN_PALM", "FIST", "TWO_FINGERS_PARALLEL", "THREE_FINGERS"]`.
2. **Bundle Serialization & Deterministic Restoration**:
   - Candidate Bundle Path: `reports/acc05/candidate_personal_svm.json` (SHA-256: `66091d52d9dd6848fd8e35fc2641a31c4c2ffc9008329a4d2e9f85f8dd1246be`).
   - By design in the codebase (`tools.train.load_model_from_bundle`), restoring a bundle executes `classifier._fit(X_train, y_train, scaler=scaler)`. This retrains a deterministic RBF-SVC on the exact embedded train-only rows using the restored train-only scaler. No hyperparameter tuning, model selection, or held-out data leakage occurred.
3. **Verification via Unit Test Spies vs Actual Bundle**:
   - The synthetic E2E unit test (`test_mocked_end_to_end_execute_benchmark_spies_scaler_and_fit`) exercises benchmark orchestration using 40 synthetic train rows, spying `_fit` and `predict_with_rejection` to prove zero leakage of validation/test features.
   - Separately, inspection of the actual on-disk candidate bundle confirms all 1692 real training rows are embedded, with scaler parameters matching the mean and scale of the real train set.

---

## 5. Paired Benchmark Results

Evaluations were performed simultaneously on the production baseline (`models/airpad_svm.json`) and the frozen candidate personal SVM (`reports/acc05/candidate_personal_svm.json`) across both Validation (1746 rows) and Held-Out Test (1663 rows).

Rejection policy parameters (fixed by protocol): `decision_threshold = 0.0`, `margin_threshold = 0.15`. Any prediction rejected or labeled `UNKNOWN` is treated as an error/FN in the full fixed denominator.

### 5.1. Held-Out Test Set Performance (1663 rows)

#### Rejection Policy (Primary Deployment Evaluation)

| Metric | Baseline Production Model | Candidate Personal Model | Delta ($\text{Cand} - \text{Base}$) | Protocol Gate |
|---|---|---|---|---|
| **Accuracy (Full Denominator)** | **25.08%** (417 / 1663) | **91.58%** (1523 / 1663) | **+66.51%** (+66.51 pp) | **PASSED** (Gate: strictly above baseline) |
| **Macro-F1 (Full Denominator)** | **16.31%** | **91.37%** | **+75.06%** (+75.06 pp) | **PASSED** (Gate: strictly above baseline) |
| **Balanced Accuracy** | 26.40% | 90.70% | **+64.30%** | Information |
| **Coverage** | 97.11% (1615 / 1663) | 100.00% (1663 / 1663) | +2.89% | Information |
| **Rejection Count** | 48 | 0 | -48 | Information |
| **OTHER-to-Target Misclassifications** | 115 | 33 | **-82 (-71.3%)** | **PASSED** (Gate: no increase) |

#### Raw / Argmax Policy (Informational)

| Metric | Baseline Production Model | Candidate Personal Model | Delta ($\text{Cand} - \text{Base}$) |
|---|---|---|---|
| **Accuracy** | **25.56%** (425 / 1663) | **91.58%** (1523 / 1663) | **+66.03%** (+66.03 pp) |
| **Macro-F1** | **16.42%** | **91.37%** | **+74.95%** (+74.95 pp) |
| **Balanced Accuracy** | 26.87% | 90.70% | +63.84% |
| **Coverage** | 100.00% | 100.00% | 0.00% |

---

### 5.2. Combined Validation Set Performance (1746 rows)

| Metric | Baseline Rejection | Candidate Rejection | Delta |
|---|---|---|---|
| **Accuracy (Full Denominator)** | 18.44% (322 / 1746) | 94.79% (1655 / 1746) | +76.35% |
| **Macro-F1 (Full Denominator)** | 12.04% | 94.79% | +82.75% |
| **Balanced Accuracy** | 17.75% | 95.16% | +77.41% |
| **Coverage** | 97.37% | 100.00% | +2.63% |
| **Rejections** | 46 | 0 | -46 |

---

## 6. Acceptance Gate Evaluations

### Gate 1: Primary Superiority Gate — **MET (PASSED)**
- **Requirement**: Candidate test full-denominator accuracy and macro-F1 must be strictly above baseline on 1663 test rows.
- **Results**:
  - Rejection policy: $\Delta \text{Accuracy} = +66.51$ percentage points ($91.58\%$ vs $25.08\%$).
  - Rejection policy: $\Delta \text{Macro-F1} = +75.06$ percentage points ($91.37\%$ vs $16.31\%$).
- **Verdict**: Strongly MET.

### Gate 2: Secondary No-Regression Recall Gate — **MET (PASSED)**
- **Requirement**: No class may suffer a recall drop exceeding 0.02 (2.0 percentage points) under the rejection policy on the held-out test set.
- **Per-Class Recall Comparison**:

| Class | Baseline Recall | Candidate Recall | Delta Recall | Drop > 0.02? | Status |
|---|---|---|---|---|---|
| `PINCH_CLOSED` | 63.98% (103/161) | 64.60% (104/161) | **+0.62%** | No | PASSED |
| `PINCH_MIDDLE` | 34.16% (55/161) | 68.94% (111/161) | **+34.78%** | No | PASSED |
| `L_SHAPE` | 100.00% (184/184) | 100.00% (184/184) | **+0.00%** | No | PASSED |
| `OTHER` | 36.13% (69/191) | 82.72% (158/191) | **+46.59%** | No | PASSED |
| `POINT_ONE` | 0.00% (0/199) | 100.00% (199/199) | **+100.00%** | No | PASSED |
| `OPEN_PALM` | 0.00% (0/220) | 100.00% (220/220) | **+100.00%** | No | PASSED |
| `FIST` | 2.78% (5/180) | 100.00% (180/180) | **+97.22%** | No | PASSED |
| `TWO_FINGERS_PARALLEL` | 0.00% (0/181) | 100.00% (181/181) | **+100.00%** | No | PASSED |
| `THREE_FINGERS` | 0.54% (1/186) | 100.00% (186/186) | **+99.46%** | No | PASSED |

- **Verdict**: Zero classes experienced a recall drop. The minimum delta was +0.00% (`L_SHAPE`), and all other 8 classes improved.

### Gate 3: Secondary OTHER-to-Target Misclassification Gate — **MET (PASSED)**
- **Requirement**: No increase in OTHER frames misclassified as target action classes ($1..8$).
- **Results**:
  - Baseline OTHER frame misclassifications: **115** out of 191 OTHER frames.
  - Candidate OTHER frame misclassifications: **33** out of 191 OTHER frames (all 33 predicted as `OPEN_PALM`).
  - Delta: **-82 misclassified frames** (-71.3% reduction).
- **Residual Operational Risk**:
  - This metric evaluates offline frame-level predictions, not observed live runtime command triggers.
  - A residual risk remains: **33 / 191 (17.28%)** of test OTHER frames were misclassified as `OPEN_PALM`.
  - In addition, fine-motor pinch gestures exhibit lower recall than full-hand poses: `PINCH_CLOSED` recall is 64.60% and `PINCH_MIDDLE` recall is 68.94%, compared to 100% for `POINT_ONE`, `OPEN_PALM`, `FIST`, `L_SHAPE`, `TWO_FINGERS_PARALLEL`, and `THREE_FINGERS`.

---

## 7. Paired Contingency Accounting (Observational)

Cross-tabulation of sample-by-sample correctness on the 1663 held-out test frames:

| Model Predictions | Baseline Correct | Baseline Incorrect | Total |
|---|---|---|---|
| **Candidate Correct** | 359 (21.59%) | **1164 (69.99%)** | 1523 (91.58%) |
| **Candidate Incorrect** | **58 (3.49%)** | 82 (4.93%) | 140 (8.42%) |
| **Total** | 417 (25.08%) | 1246 (74.92%) | 1663 (100.0%) |

- **Observational Disagreement**: Candidate was uniquely correct on **1164** frames where Baseline failed; Baseline was uniquely correct on **58** frames where Candidate failed.
- **Statistical Consideration**: Because video frames within gesture bursts exhibit high temporal correlation (consecutive frames are not independent and identically distributed), standard i.i.d. assumptions required for McNemar asymptotic significance testing are violated. This table is reported as descriptive sample agreement only, without significance claims.

---

## 8. Baseline Model Inspection & Explanatory Discussion

1. **Bundle Label Verification**:
   - Direct inspection of `models/airpad_svm.json` confirms that the baseline model bundle embeds **all 9 classes** in its `labels` and `model_data["y_train"]` metadata across 61,066 training samples.
   - Consequently, the hypothesis that baseline errors were caused by missing class definitions is factually disproven.
2. **Explanatory Hypotheses**:
   - The OpenCV BGR vs MediaPipe RGB color space discrepancy in legacy custom collection was a confirmed implementation defect. However, this benchmark does not isolate its individual causal impact from broader domain shift factors, such as personal morphological adaptation, sensor noise, background lighting, or training distribution shifts.
   - The superior performance of the personal model is consistent with personalized domain calibration on user `tuana`'s specific hand geometry and sensor capture conditions, but isolating the exact contributions of each factor remains a hypothesis requiring ablation.

---

## 9. Chronology & Protocol Deviations Disclosure

Full chronological details are documented in `reports/acc05/CHRONOLOGY_AND_DEVIATIONS.md`:
1. **Preliminary Ad-Hoc Execution**: Prior to the creation of `benchmark_personal_svm.py` and the serialization of `candidate_personal_svm.json`, an exploratory benchmark was run via an ad-hoc `python -c` invocation. Following Coordinator instructions, the dedicated benchmark script was authored to enforce proper sequencing (persisting and hashing the candidate artifact first before computing official predictions).
2. **Deterministic Re-fitting on Bundle Load**: Reconstituting the candidate via `load_model_from_bundle` deterministically retrains an RBF-SVC on the embedded train-only data rows by design. No hyperparameter tuning or held-out data leakage occurred.
3. **Class Ordering**: `STATIC_CLASSES` in `benchmark_personal_svm.py` follows `["PINCH_CLOSED", "PINCH_MIDDLE", "L_SHAPE", "OTHER", "POINT_ONE", "OPEN_PALM", "FIST", "TWO_FINGERS_PARALLEL", "THREE_FINGERS"]`. All models, predictions, and metrics used this identical class sequence.

---

## 10. Verification & Test Suite Execution

All test suites pass cleanly:
1. **ACC-05 Isolated Tests** (`reports/acc05/test_acc05_benchmark.py`, SHA-256: `fadd6e2e4a7cd5ba33b18577be77e629eb50c7c7593b4240c788cac6b9af1454`):
   - **13 passed** in 2.44s.
   - Tested: synthetic E2E benchmark orchestration with spies verifying scaler fit exclusively on 40 train rows, restored candidate receiving exact train X/y and frozen hyperparameters, identical held-out X features for both models, 40 validation / 30 test support verification, source row coverage, rejection metrics vs sklearn, pinned hash enforcement, path collision guards, cross-role session overlap, and cross-role feature duplicate rejection.
2. **Focused Regression Suite** (`reports/acc04/test_repair_validation.py`, `reports/acc05/test_acc05_benchmark.py`, `tests/test_cohort_sessions.py`, `tests/test_tools.py`):
   - **115 passed** in 7.24s.
3. **Full Project Test Suite** (`pytest -q tests reports/acc04/test_repair_validation.py reports/acc05/test_acc05_benchmark.py`):
   - **285 passed** in 41.04s.

---

## 11. Artifact Manifest & Checksums

| File Path | Size (Bytes) | SHA-256 Hash |
|---|---|---|
| `reports/acc05/plan_acc05.json` | 264 | `eab02f26af2b1ef36263489d09dbb5a9f5c0b8fdf822e55df52dcfbdc6bd4168` |
| `reports/acc05/readiness_acc05.json` | 7,219 | `c61326759c8b2070a79ee8b06053ff980959af362aab165a4b9d97f28ac0d3fc` |
| `reports/acc05/manifest_acc05.json` | 1,993,159 | `0ec9a1ebd34ca26bb92cc338e27fdfe0983f734c64f41da9f5067624d5ea6cda` |
| `reports/acc05/benchmark_personal_svm.py` | 34,755 | `6fccfee2be2c9f029bc02badd691486d72af90f06f422ce53d2d503256dea6d4` |
| `reports/acc05/candidate_personal_svm.json` | 3,227,388 | `66091d52d9dd6848fd8e35fc2641a31c4c2ffc9008329a4d2e9f85f8dd1246be` |
| `reports/acc05/benchmark_predictions.json` | 3,473,187 | `71f04150db7f3195bcb407f190e0f1cdebd00d01390d06ca6d2202241faf6dee` |
| `reports/acc05/benchmark_metrics.json` | 40,828 | `5d8dcf852d1c48983875a39d46abe87b490d0a866934ccec19ab905b1fdd9a51` |
| `reports/acc05/test_acc05_benchmark.py` | 25,499 | `fadd6e2e4a7cd5ba33b18577be77e629eb50c7c7593b4240c788cac6b9af1454` |
| `reports/acc05/CHRONOLOGY_AND_DEVIATIONS.md` | 8,407 | `dd521d55a88fbe19fe9252e20b569cfb6173f6d1a06bd8bd5df84923afdccf8a` |
| `reports/ACC_05_WORKER_REPORT.md` | (current file) | (signed below) |

---

## 12. Operational Limitations & Next Steps

1. **Scope of Generalization**:
   - The candidate model was evaluated exclusively on user `tuana` same-day personal capture sessions under consistent indoor webcam conditions.
   - It is not certified for multi-user cohort deployment or diverse environmental lighting without further multi-user data collection.
2. **Production Status**:
   - `models/airpad_svm.json` remains unchanged. No model promotion or production deployment was executed.
   - Promotion to production runtime remains under Coordinator authorization.
