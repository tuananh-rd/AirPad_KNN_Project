# ACC-05 protocol frozen before test predictions — 2026-10-01

One experiment: unchanged production RBF-SVM versus one personal train-only RBF-SVM on the identical held-out personal test take. No test predictions have been viewed at this checkpoint. No parameter search, architecture change, threshold tuning, train+validation refit or model promotion in this task.

Candidate: StandardScaler fitted exclusively to the 1692 post-RGB train rows; SVM C=10.0, gamma=0.01, class_weight=balanced, matching baseline hyperparameters. All nine classes in fixed sorted label order. Rejection thresholds decision=0.0, margin=0.15 for BOTH models. Report raw and rejection-aware metrics; primary comparison is rejection-aware full-denominator accuracy and nine-class macro-F1. Validation is diagnostic only, not training or selection.

Baseline bundle: models/airpad_svm.json SHA256 4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977, reconstructed with tools.train.load_model_from_bundle preserving its embedded scaler. Candidate serialized separately under reports/acc05 and frozen before test predictions.

Immutable role inputs:

- Train: data/cohort/tuana_train_rgb_20261001_110223_8ff3ed.npz, 1692 rows, SHA256 0f9d21b5b1a5ccab0ffc8f555d7c00e14b26fa2299040b9ab587b236e7d1014e.
- Validation: data/cohort/curated/acc04_validation_clean.npz, 1331 rows, SHA256 e0a27d1bd989a1ac1fe9fccc90c9ae57a47a8ca0637f1959cab91a4fcc99000d; and unchanged retake data/cohort/tuana_validation_retake_20261001_113202_dfcd6b.npz, 415 rows, SHA256 fe26b606921386a0f640dc958e4e7ebf01e43ce0d6cd6f07e693fce517db8948.
- Test: data/cohort/tuana_test_rgb_20261001_115837_5741bd.npz, 1663 rows, SHA256 4ec1cf2dfbdcd74b5c22c914a31c52bf2fd3c11d5c3a4de5a49368eb9cca28fb.

Independent readiness audit is structurally READY. Role IDs were assigned before recording/results; validation retake remains validation. Preserve curated-to-raw source provenance from ACC-04. No raw unclean validation, earlier night train, rejected session or fake chunks in candidate inputs. Reject session overlap or exact cross-role feature duplicates rather than silently filtering held-out rows.

Observed baseline-exceeding outcome requires candidate test full-denominator accuracy AND macro-F1 strictly above baseline on these same1663 rows. Eligibility for later live trial additionally requires no per-class recall drop exceeding0.02 and no increase in OTHER-to-target false-command count; report those separately even if primary gain achieved. UNKNOWN counted as error/FN, all support retained. No guarantee of achieving gates; failure is valid evidence, not permission to optimize on this test.

These are descriptive personal same-day metrics from correlated frames/one held-out take, not confidence evidence across days, lighting, friends, detector failures or end-to-end actions. Historical six-class HaGRID scores and leaky nine-class scores are not comparable. Once viewed, this test is consumed for this frozen experiment; any materially selected/tuned follow-up needs validation-only reasoning and a new independent final test.
