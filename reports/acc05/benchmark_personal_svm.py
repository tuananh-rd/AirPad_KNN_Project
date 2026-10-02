"""ACC-05 Single Frozen Personal SVM Paired Benchmark Script.

Executes the frozen ACC-05 protocol:
1. Validates pinned hashes of all 6 immutable inputs before processing.
2. Enforces strict path collision and alias guards:
   - Output files must not exist before the run (fails closed immediately).
   - Output paths must not alias input datasets or each other.
3. Pre-validates sample schema, lengths, exact user/session IDs, and zero transitions.
4. Validates cross-role session disjointness and exact zero cross-role feature overlap.
5. Fits candidate StandardScaler and SVM (C=10.0, gamma=0.01, balanced) exclusively on 1692 train rows.
6. Serializes candidate model bundle to reports/acc05/candidate_personal_svm.json and hashes it.
7. Reconstitutes candidate and baseline models from frozen bundles.
8. Evaluates both models on identical validation (1746 rows) and test (1663 rows) under raw and rejection policies.
9. Computes fixed 9-label metrics (accuracy, macro-F1, balanced accuracy, coverage, confusion matrix, per-class stats).
10. Evaluates decision gates (primary accuracy/macro-F1 delta, recall drop <= 0.02, OTHER-to-target false commands).
11. Saves full prediction provenance and metrics JSON to reports/acc05/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml.evaluation import compute_metrics
from src.ml.svm import SVMClassifier
from tools.cohort_sessions import norm_path, paths_alias_each_other, validate_samples_integrity
from tools.collect_data import LABEL_NAMES, load_dataset_npz
from tools.train import create_model_bundle, load_model_from_bundle, save_bundle

# Canonical 9 static target classes in AirPad
STATIC_CLASSES = [
    "PINCH_CLOSED",
    "PINCH_MIDDLE",
    "L_SHAPE",
    "OTHER",
    "POINT_ONE",
    "OPEN_PALM",
    "FIST",
    "TWO_FINGERS_PARALLEL",
    "THREE_FINGERS",
]

TARGET_ACTION_CLASSES = set(STATIC_CLASSES) - {"OTHER"}

# Pinned immutable input hashes
PINNED_HASHES = {
    "train_rgb": {
        "rel_path": "data/cohort/tuana_train_rgb_20261001_110223_8ff3ed.npz",
        "sha256": "0f9d21b5b1a5ccab0ffc8f555d7c00e14b26fa2299040b9ab587b236e7d1014e",
        "expected_rows": 1692,
        "expected_user_id": "tuana",
        "expected_session_id": "tuana_train_rgb_20261001_110223_8ff3ed",
    },
    "val_clean": {
        "rel_path": "data/cohort/curated/acc04_validation_clean.npz",
        "sha256": "e0a27d1bd989a1ac1fe9fccc90c9ae57a47a8ca0637f1959cab91a4fcc99000d",
        "expected_rows": 1331,
        "expected_user_id": "tuana",
        "expected_session_id": "tuana_validation_rgb_20261001_112128_2a3fea",
    },
    "val_retake": {
        "rel_path": "data/cohort/tuana_validation_retake_20261001_113202_dfcd6b.npz",
        "sha256": "fe26b606921386a0f640dc958e4e7ebf01e43ce0d6cd6f07e693fce517db8948",
        "expected_rows": 415,
        "expected_user_id": "tuana",
        "expected_session_id": "tuana_validation_retake_20261001_113202_dfcd6b",
    },
    "test_rgb": {
        "rel_path": "data/cohort/tuana_test_rgb_20261001_115837_5741bd.npz",
        "sha256": "4ec1cf2dfbdcd74b5c22c914a31c52bf2fd3c11d5c3a4de5a49368eb9cca28fb",
        "expected_rows": 1663,
        "expected_user_id": "tuana",
        "expected_session_id": "tuana_test_rgb_20261001_115837_5741bd",
    },
    "baseline_bundle": {
        "rel_path": "models/airpad_svm.json",
        "sha256": "4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977",
    },
    "acc04_audit": {
        "rel_path": "reports/acc04/validation_cleaning_audit.json",
        "sha256": "4772a95cca9c08937057775223afd7b83c2ffab6f6f39c09cb43f7f874f98ff0",
    },
}

DEFAULT_CANDIDATE_BUNDLE_PATH = os.path.join(PROJECT_ROOT, "reports", "acc05", "candidate_personal_svm.json")
DEFAULT_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "reports", "acc05", "benchmark_predictions.json")
DEFAULT_METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "acc05", "benchmark_metrics.json")


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_pinned_hash(filepath: str, expected_sha256: str) -> str:
    """Verify that file exists and matches the pinned SHA-256 hash; fails closed on discrepancy."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[FAIL-CLOSED] Required input file not found: {filepath}")
    actual_sha256 = compute_file_sha256(filepath)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"[FAIL-CLOSED] Hash mismatch for {filepath}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def validate_benchmark_paths(
    input_paths: List[Tuple[str, str]],
    output_paths: List[Tuple[str, str]],
) -> None:
    """Strictly validate path safety and collision guards before any benchmark processing or writes.

    Enforces:
    - Output target files must not already exist.
    - Output target paths must not alias each other.
    - Output target paths must not alias ANY input dataset or bundle.
    """
    # 1. Output targets must not already exist
    for out_name, out_path in output_paths:
        if os.path.exists(out_path):
            raise FileExistsError(
                f"[FAIL-CLOSED] Target output file already exists: {out_path} ({out_name}). "
                "Refusing to overwrite existing benchmark artifact."
            )
        # Check potential numpy.savez suffix appending if someone passed .npz target
        if out_path.endswith(".npz"):
            effective_target = out_path if out_path.endswith(".npz") else f"{out_path}.npz"
            if os.path.exists(effective_target) and effective_target != out_path:
                raise FileExistsError(
                    f"[FAIL-CLOSED] Suffix-appended target output file already exists: {effective_target}. "
                    "Refusing to overwrite."
                )

    # 2. Output targets must not alias each other
    for i, (name1, p1) in enumerate(output_paths):
        for j, (name2, p2) in enumerate(output_paths):
            if i < j and paths_alias_each_other(p1, p2):
                raise ValueError(
                    f"[FAIL-CLOSED] Target paths {name1} ('{p1}') and {name2} ('{p2}') "
                    "resolve to the same filesystem location."
                )

    # 3. Output targets must not alias any input dataset
    for out_name, out_path in output_paths:
        for in_name, in_path in input_paths:
            if paths_alias_each_other(out_path, in_path):
                raise ValueError(
                    f"[FAIL-CLOSED] Target path {out_name} ('{out_path}') aliases "
                    f"input path {in_name} ('{in_path}'). Writes strictly prohibited."
                )


def validate_sample_schema_and_integrity(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    expected_rows: int,
    expected_user_id: str,
    expected_session_id: str,
    source_path: str,
    source_hash: str,
    dataset_name: str,
) -> None:
    """Validate sample schema, lengths, identities, and zero transitions in memory."""
    n_samples = len(metadata)
    if n_samples != expected_rows:
        raise ValueError(
            f"[FAIL-CLOSED] {dataset_name} row count mismatch: expected {expected_rows}, got {n_samples}"
        )
    if len(labels) != expected_rows:
        raise ValueError(
            f"[FAIL-CLOSED] {dataset_name} labels count mismatch: expected {expected_rows}, got {len(labels)}"
        )
    if len(features) != expected_rows:
        raise ValueError(
            f"[FAIL-CLOSED] {dataset_name} features count mismatch: expected {expected_rows}, got {len(features)}"
        )

    if features.ndim != 2 or features.shape[1] != 63:
        raise ValueError(
            f"[FAIL-CLOSED] {dataset_name} features must have shape ({expected_rows}, 63), got {features.shape}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"[FAIL-CLOSED] {dataset_name} features must have float dtype, got {features.dtype}")
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError(f"[FAIL-CLOSED] {dataset_name} features contain NaN or infinite values")

    for idx, meta in enumerate(metadata):
        if not isinstance(meta, dict):
            raise ValueError(f"[FAIL-CLOSED] {dataset_name} row {idx}: metadata is not a dictionary")

        u = meta.get("user_id")
        if u != expected_user_id:
            raise ValueError(
                f"[FAIL-CLOSED] {dataset_name} row {idx}: unexpected user_id '{u}', expected '{expected_user_id}'"
            )
        s = meta.get("session_id")
        if s != expected_session_id:
            raise ValueError(
                f"[FAIL-CLOSED] {dataset_name} row {idx}: unexpected session_id '{s}', expected '{expected_session_id}'"
            )

        lbl_arr = str(labels[idx])
        meta_lbl = meta.get("label")
        if lbl_arr != meta_lbl:
            raise ValueError(
                f"[FAIL-CLOSED] {dataset_name} row {idx}: label mismatch between array ('{lbl_arr}') and metadata ('{meta_lbl}')"
            )
        if lbl_arr not in LABEL_NAMES:
            raise ValueError(f"[FAIL-CLOSED] {dataset_name} row {idx}: unrecognized label '{lbl_arr}'")

        is_trans = meta.get("is_transition")
        if is_trans is not False:
            raise ValueError(
                f"[FAIL-CLOSED] {dataset_name} row {idx}: is_transition must be strictly False, "
                f"got {type(is_trans).__name__} ({is_trans!r})"
            )

    # Verification via existing validator: validate_samples_integrity
    prov_meta = []
    for idx, m in enumerate(metadata):
        m_copy = copy.deepcopy(m)
        m_copy["source_file"] = source_path
        m_copy["source_hash"] = source_hash
        m_copy["source_row_idx"] = m.get("original_row_idx", idx)
        prov_meta.append(m_copy)

    validate_samples_integrity(
        prov_meta,
        labels,
        features,
        source_hashes={source_path: source_hash},
    )


def compute_prediction_margins(scores: np.ndarray) -> np.ndarray:
    """Compute margin between highest decision score and second highest decision score."""
    if scores.shape[1] > 1:
        top_two = np.partition(scores, -2, axis=1)[:, -2:]
        margins = top_two[:, 1] - top_two[:, 0]
    else:
        margins = np.full(len(scores), np.inf)
    return margins.astype(float)


def compute_paired_contingency(
    y_true: List[str],
    y_pred_baseline: List[str],
    y_pred_candidate: List[str],
) -> Dict[str, int]:
    """Compute 2x2 contingency table for paired McNemar-style comparison."""
    both_correct = 0
    candidate_only = 0
    baseline_only = 0
    both_wrong = 0

    for t, b, c in zip(y_true, y_pred_baseline, y_pred_candidate):
        b_corr = (b == t)
        c_corr = (c == t)
        if b_corr and c_corr:
            both_correct += 1
        elif c_corr and not b_corr:
            candidate_only += 1
        elif b_corr and not c_corr:
            baseline_only += 1
        else:
            both_wrong += 1

    return {
        "both_correct": both_correct,
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "both_wrong": both_wrong,
        "total": len(y_true),
    }


def compute_other_to_target_count(
    y_true: List[str],
    y_pred: List[str],
) -> int:
    """Count instances where ground truth is OTHER but model predicts an active command class."""
    count = 0
    for t, p in zip(y_true, y_pred):
        if t == "OTHER" and p in TARGET_ACTION_CLASSES:
            count += 1
    return count


def execute_acc05_benchmark(
    train_path: Optional[str] = None,
    val_clean_path: Optional[str] = None,
    val_retake_path: Optional[str] = None,
    test_path: Optional[str] = None,
    baseline_bundle_path: Optional[str] = None,
    acc04_audit_path: Optional[str] = None,
    candidate_bundle_path: Optional[str] = None,
    predictions_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the single frozen personal SVM paired benchmark."""
    train_path = train_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["train_rgb"]["rel_path"])
    val_clean_path = val_clean_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["val_clean"]["rel_path"])
    val_retake_path = val_retake_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["val_retake"]["rel_path"])
    test_path = test_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["test_rgb"]["rel_path"])
    baseline_bundle_path = baseline_bundle_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["baseline_bundle"]["rel_path"])
    acc04_audit_path = acc04_audit_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["acc04_audit"]["rel_path"])

    candidate_bundle_path = candidate_bundle_path or DEFAULT_CANDIDATE_BUNDLE_PATH
    predictions_path = predictions_path or DEFAULT_PREDICTIONS_PATH
    metrics_path = metrics_path or DEFAULT_METRICS_PATH

    # 1. Validate paths and collision guards before any execution
    input_paths = [
        ("train_rgb", train_path),
        ("val_clean", val_clean_path),
        ("val_retake", val_retake_path),
        ("test_rgb", test_path),
        ("baseline_bundle", baseline_bundle_path),
        ("acc04_audit", acc04_audit_path),
    ]
    output_paths = [
        ("candidate_bundle", candidate_bundle_path),
        ("predictions_json", predictions_path),
        ("metrics_json", metrics_path),
    ]
    validate_benchmark_paths(input_paths, output_paths)

    # 2. Mandatory verification of all pinned hashes
    train_hash = verify_pinned_hash(train_path, PINNED_HASHES["train_rgb"]["sha256"])
    val_clean_hash = verify_pinned_hash(val_clean_path, PINNED_HASHES["val_clean"]["sha256"])
    val_retake_hash = verify_pinned_hash(val_retake_path, PINNED_HASHES["val_retake"]["sha256"])
    test_hash = verify_pinned_hash(test_path, PINNED_HASHES["test_rgb"]["sha256"])
    baseline_bundle_hash = verify_pinned_hash(baseline_bundle_path, PINNED_HASHES["baseline_bundle"]["sha256"])
    acc04_audit_hash = verify_pinned_hash(acc04_audit_path, PINNED_HASHES["acc04_audit"]["sha256"])

    # Load ACC-04 audit for raw source row mapping of clean validation
    with open(acc04_audit_path, "r", encoding="utf-8") as f:
        acc04_audit = json.load(f)
    acc04_retained_mapping = acc04_audit.get("retained_samples_mapping", [])
    if len(acc04_retained_mapping) != PINNED_HASHES["val_clean"]["expected_rows"]:
        raise ValueError(
            f"[FAIL-CLOSED] ACC-04 audit retained_samples_mapping length mismatch: "
            f"expected {PINNED_HASHES['val_clean']['expected_rows']}, got {len(acc04_retained_mapping)}"
        )

    # 3. Load datasets and validate sample integrity
    train_f, train_l, train_m = load_dataset_npz(train_path)
    validate_sample_schema_and_integrity(
        train_f, train_l, train_m,
        expected_rows=PINNED_HASHES["train_rgb"]["expected_rows"],
        expected_user_id=PINNED_HASHES["train_rgb"]["expected_user_id"],
        expected_session_id=PINNED_HASHES["train_rgb"]["expected_session_id"],
        source_path=train_path, source_hash=train_hash, dataset_name="train_rgb",
    )

    val_clean_f, val_clean_l, val_clean_m = load_dataset_npz(val_clean_path)
    validate_sample_schema_and_integrity(
        val_clean_f, val_clean_l, val_clean_m,
        expected_rows=PINNED_HASHES["val_clean"]["expected_rows"],
        expected_user_id=PINNED_HASHES["val_clean"]["expected_user_id"],
        expected_session_id=PINNED_HASHES["val_clean"]["expected_session_id"],
        source_path=val_clean_path, source_hash=val_clean_hash, dataset_name="val_clean",
    )

    val_retake_f, val_retake_l, val_retake_m = load_dataset_npz(val_retake_path)
    validate_sample_schema_and_integrity(
        val_retake_f, val_retake_l, val_retake_m,
        expected_rows=PINNED_HASHES["val_retake"]["expected_rows"],
        expected_user_id=PINNED_HASHES["val_retake"]["expected_user_id"],
        expected_session_id=PINNED_HASHES["val_retake"]["expected_session_id"],
        source_path=val_retake_path, source_hash=val_retake_hash, dataset_name="val_retake",
    )

    test_f, test_l, test_m = load_dataset_npz(test_path)
    validate_sample_schema_and_integrity(
        test_f, test_l, test_m,
        expected_rows=PINNED_HASHES["test_rgb"]["expected_rows"],
        expected_user_id=PINNED_HASHES["test_rgb"]["expected_user_id"],
        expected_session_id=PINNED_HASHES["test_rgb"]["expected_session_id"],
        source_path=test_path, source_hash=test_hash, dataset_name="test_rgb",
    )

    # 4. Check cross-role session disjointness
    train_sessions = set(m["session_id"] for m in train_m)
    val_sessions = set(m["session_id"] for m in val_clean_m).union(set(m["session_id"] for m in val_retake_m))
    test_sessions = set(m["session_id"] for m in test_m)

    if train_sessions.intersection(val_sessions):
        raise ValueError(f"[FAIL-CLOSED] Overlapping sessions between train and validation: {train_sessions.intersection(val_sessions)}")
    if train_sessions.intersection(test_sessions):
        raise ValueError(f"[FAIL-CLOSED] Overlapping sessions between train and test: {train_sessions.intersection(test_sessions)}")
    if val_sessions.intersection(test_sessions):
        raise ValueError(f"[FAIL-CLOSED] Overlapping sessions between validation and test: {val_sessions.intersection(test_sessions)}")

    # 5. Check exact cross-role feature duplicates (zero tolerance, never silently filter)
    val_f = np.vstack([val_clean_f, val_retake_f])
    val_l = np.concatenate([val_clean_l, val_retake_l])
    val_m = val_clean_m + val_retake_m

    train_tuples = set(tuple(row) for row in train_f)
    val_tuples = set(tuple(row) for row in val_f)
    test_tuples = set(tuple(row) for row in test_f)

    tv_overlap = train_tuples.intersection(val_tuples)
    if tv_overlap:
        raise ValueError(f"[FAIL-CLOSED] Detected {len(tv_overlap)} exact feature duplicates between train and validation.")
    tt_overlap = train_tuples.intersection(test_tuples)
    if tt_overlap:
        raise ValueError(f"[FAIL-CLOSED] Detected {len(tt_overlap)} exact feature duplicates between train and test.")
    vt_overlap = val_tuples.intersection(test_tuples)
    if vt_overlap:
        raise ValueError(f"[FAIL-CLOSED] Detected {len(vt_overlap)} exact feature duplicates between validation and test.")

    # 6. Fit candidate model and serialize candidate bundle
    # C=10.0, gamma=0.01, class_weight='balanced', StandardScaler fitted exclusively on 1692 train rows
    candidate_bundle = create_model_bundle(
        X=train_f,
        y=train_l,
        C=10.0,
        gamma=0.01,
        class_weight="balanced",
        labels=STATIC_CLASSES,
        extra_metadata={
            "experiment": "ACC-05",
            "task": "Single frozen personal SVM paired benchmark",
            "train_input_path": os.path.relpath(train_path, PROJECT_ROOT).replace("\\", "/"),
            "train_input_sha256": train_hash,
            "train_rows": len(train_f),
        },
    )

    out_bundle_dir = os.path.dirname(candidate_bundle_path)
    if out_bundle_dir:
        os.makedirs(out_bundle_dir, exist_ok=True)
    save_bundle(candidate_bundle, candidate_bundle_path)
    candidate_bundle_hash = compute_file_sha256(candidate_bundle_path)

    # 7. Reconstitute candidate and baseline models from frozen bundles
    cand_clf, cand_scaler, cand_b = load_model_from_bundle(candidate_bundle_path)
    base_clf, base_scaler, base_b = load_model_from_bundle(baseline_bundle_path)

    # Verify candidate scaler was fit exclusively on train set
    train_mean = np.mean(train_f, axis=0)
    if not np.allclose(cand_scaler.mean_, train_mean, atol=1e-5):
        raise ValueError("[FAIL-CLOSED] Candidate scaler mean does not match train set mean.")

    # 8. Run predictions on Validation (1746 rows) and Test (1663 rows)
    # Validation predictions
    b_val_raw = base_clf.predict(val_f)
    b_val_rej = base_clf.predict_with_rejection(val_f, decision_threshold=0.0, margin_threshold=0.15, unknown_label="UNKNOWN")
    b_val_scores = base_clf.decision_function(val_f)
    b_val_margins = compute_prediction_margins(b_val_scores)
    b_val_best = b_val_scores[np.arange(len(b_val_scores)), np.argmax(b_val_scores, axis=1)]

    c_val_raw = cand_clf.predict(val_f)
    c_val_rej = cand_clf.predict_with_rejection(val_f, decision_threshold=0.0, margin_threshold=0.15, unknown_label="UNKNOWN")
    c_val_scores = cand_clf.decision_function(val_f)
    c_val_margins = compute_prediction_margins(c_val_scores)
    c_val_best = c_val_scores[np.arange(len(c_val_scores)), np.argmax(c_val_scores, axis=1)]

    # Test predictions
    b_test_raw = base_clf.predict(test_f)
    b_test_rej = base_clf.predict_with_rejection(test_f, decision_threshold=0.0, margin_threshold=0.15, unknown_label="UNKNOWN")
    b_test_scores = base_clf.decision_function(test_f)
    b_test_margins = compute_prediction_margins(b_test_scores)
    b_test_best = b_test_scores[np.arange(len(b_test_scores)), np.argmax(b_test_scores, axis=1)]

    c_test_raw = cand_clf.predict(test_f)
    c_test_rej = cand_clf.predict_with_rejection(test_f, decision_threshold=0.0, margin_threshold=0.15, unknown_label="UNKNOWN")
    c_test_scores = cand_clf.decision_function(test_f)
    c_test_margins = compute_prediction_margins(c_test_scores)
    c_test_best = c_test_scores[np.arange(len(c_test_scores)), np.argmax(c_test_scores, axis=1)]

    # 9. Compute evaluation metrics across all four settings
    val_true_list = [str(x) for x in val_l]
    test_true_list = [str(x) for x in test_l]

    metrics_results = {
        "validation": {
            "baseline_raw": compute_metrics(val_true_list, b_val_raw.tolist(), labels=STATIC_CLASSES),
            "baseline_rejection": compute_metrics(val_true_list, b_val_rej.tolist(), labels=STATIC_CLASSES),
            "candidate_raw": compute_metrics(val_true_list, c_val_raw.tolist(), labels=STATIC_CLASSES),
            "candidate_rejection": compute_metrics(val_true_list, c_val_rej.tolist(), labels=STATIC_CLASSES),
        },
        "test": {
            "baseline_raw": compute_metrics(test_true_list, b_test_raw.tolist(), labels=STATIC_CLASSES),
            "baseline_rejection": compute_metrics(test_true_list, b_test_rej.tolist(), labels=STATIC_CLASSES),
            "candidate_raw": compute_metrics(test_true_list, c_test_raw.tolist(), labels=STATIC_CLASSES),
            "candidate_rejection": compute_metrics(test_true_list, c_test_rej.tolist(), labels=STATIC_CLASSES),
        },
    }

    # Add rejection counts explicitly
    for split_name in ["validation", "test"]:
        for pol in ["baseline_rejection", "candidate_rejection"]:
            m = metrics_results[split_name][pol]
            rej_count = int(round(m["rejection_rate"] * m["support"]))
            m["rejection_count"] = rej_count

    # 10. Paired comparisons and Gate Evaluations
    b_test_rej_m = metrics_results["test"]["baseline_rejection"]
    c_test_rej_m = metrics_results["test"]["candidate_rejection"]

    delta_acc = float(c_test_rej_m["accuracy"] - b_test_rej_m["accuracy"])
    delta_macro_f1 = float(c_test_rej_m["macro_f1"] - b_test_rej_m["macro_f1"])
    delta_balanced_acc = float(c_test_rej_m["balanced_accuracy"] - b_test_rej_m["balanced_accuracy"])
    delta_coverage = float(c_test_rej_m["coverage"] - b_test_rej_m["coverage"])
    delta_rejection_count = int(c_test_rej_m["rejection_count"] - b_test_rej_m["rejection_count"])

    # Per-class recall deltas on test rejection
    per_class_recall_deltas = {}
    per_class_drops_exceeding_2pct = []
    for cls_name in STATIC_CLASSES:
        b_rec = b_test_rej_m["per_class"][cls_name]["recall"]
        c_rec = c_test_rej_m["per_class"][cls_name]["recall"]
        d_rec = float(c_rec - b_rec)
        per_class_recall_deltas[cls_name] = {
            "baseline_recall": b_rec,
            "candidate_recall": c_rec,
            "delta_recall": d_rec,
            "drop_exceeds_0_02": bool(d_rec < -0.02),
        }
        if d_rec < -0.02:
            per_class_drops_exceeding_2pct.append(cls_name)

    # OTHER-to-target false command count
    b_other_to_target = compute_other_to_target_count(test_true_list, b_test_rej.tolist())
    c_other_to_target = compute_other_to_target_count(test_true_list, c_test_rej.tolist())
    delta_other_to_target = c_other_to_target - b_other_to_target

    # Paired contingency table on test rejection
    test_contingency = compute_paired_contingency(test_true_list, b_test_rej.tolist(), c_test_rej.tolist())

    # Gate evaluations
    primary_gate_met = bool(delta_acc > 0.0 and delta_macro_f1 > 0.0)
    secondary_recall_gate_met = bool(len(per_class_drops_exceeding_2pct) == 0)
    secondary_other_gate_met = bool(delta_other_to_target <= 0)
    secondary_gates_met = bool(secondary_recall_gate_met and secondary_other_gate_met)

    # 11. Compile prediction records with full provenance
    val_pred_records = []
    for i in range(len(val_f)):
        # Curated validation partition (rows 0..1330)
        if i < len(val_clean_f):
            audit_item = acc04_retained_mapping[i]
            src_file = os.path.relpath(val_clean_path, PROJECT_ROOT).replace("\\", "/")
            src_hash = val_clean_hash
            src_row = i
            raw_src_file = audit_item["original_source_file"]
            raw_src_hash = audit_item["original_source_hash"]
            raw_src_row = audit_item["original_row_idx"]
            sess_id = val_clean_m[i]["session_id"]
            u_id = val_clean_m[i]["user_id"]
            ts = val_clean_m[i]["timestamp"]
        else:
            # Retake validation partition (rows 1331..1745)
            r_idx = i - len(val_clean_f)
            src_file = os.path.relpath(val_retake_path, PROJECT_ROOT).replace("\\", "/")
            src_hash = val_retake_hash
            src_row = r_idx
            raw_src_file = src_file
            raw_src_hash = src_hash
            raw_src_row = r_idx
            sess_id = val_retake_m[r_idx]["session_id"]
            u_id = val_retake_m[r_idx]["user_id"]
            ts = val_retake_m[r_idx]["timestamp"]

        val_pred_records.append({
            "evaluation_role": "validation",
            "sample_index": i,
            "source_file": src_file,
            "source_hash": src_hash,
            "source_row_idx": src_row,
            "raw_source_file": raw_src_file,
            "raw_source_hash": raw_src_hash,
            "raw_source_row_idx": raw_src_row,
            "user_id": u_id,
            "session_id": sess_id,
            "timestamp": ts,
            "true_label": val_true_list[i],
            "baseline_raw_pred": str(b_val_raw[i]),
            "baseline_rej_pred": str(b_val_rej[i]),
            "baseline_best_score": float(b_val_best[i]),
            "baseline_margin": float(b_val_margins[i]),
            "candidate_raw_pred": str(c_val_raw[i]),
            "candidate_rej_pred": str(c_val_rej[i]),
            "candidate_best_score": float(c_val_best[i]),
            "candidate_margin": float(c_val_margins[i]),
        })

    test_pred_records = []
    for i in range(len(test_f)):
        test_pred_records.append({
            "evaluation_role": "test",
            "sample_index": i,
            "source_file": os.path.relpath(test_path, PROJECT_ROOT).replace("\\", "/") ,
            "source_hash": test_hash,
            "source_row_idx": i,
            "raw_source_file": os.path.relpath(test_path, PROJECT_ROOT).replace("\\", "/"),
            "raw_source_hash": test_hash,
            "raw_source_row_idx": i,
            "user_id": test_m[i]["user_id"],
            "session_id": test_m[i]["session_id"],
            "timestamp": test_m[i]["timestamp"],
            "true_label": test_true_list[i],
            "baseline_raw_pred": str(b_test_raw[i]),
            "baseline_rej_pred": str(b_test_rej[i]),
            "baseline_best_score": float(b_test_best[i]),
            "baseline_margin": float(b_test_margins[i]),
            "candidate_raw_pred": str(c_test_raw[i]),
            "candidate_rej_pred": str(c_test_rej[i]),
            "candidate_best_score": float(c_test_best[i]),
            "candidate_margin": float(c_test_margins[i]),
        })

    # 12. Save prediction records JSON to disk
    predictions_payload = {
        "task_id": "ACC-05",
        "description": "ACC-05 Paired Model Prediction Records with Provenance",
        "validation_samples_count": len(val_pred_records),
        "test_samples_count": len(test_pred_records),
        "validation_predictions": val_pred_records,
        "test_predictions": test_pred_records,
    }
    pred_dir = os.path.dirname(predictions_path)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f, indent=2)

    # 13. Save benchmark metrics JSON to disk
    metrics_payload = {
        "task_id": "ACC-05",
        "description": "ACC-05 Single Frozen Personal SVM Paired Benchmark Metrics",
        "inputs": {
            "train_rgb": {
                "path": os.path.relpath(train_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": train_hash,
                "rows": len(train_f),
            },
            "val_clean": {
                "path": os.path.relpath(val_clean_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": val_clean_hash,
                "rows": len(val_clean_f),
            },
            "val_retake": {
                "path": os.path.relpath(val_retake_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": val_retake_hash,
                "rows": len(val_retake_f),
            },
            "test_rgb": {
                "path": os.path.relpath(test_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": test_hash,
                "rows": len(test_f),
            },
            "baseline_bundle": {
                "path": os.path.relpath(baseline_bundle_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": baseline_bundle_hash,
            },
            "candidate_bundle": {
                "path": os.path.relpath(candidate_bundle_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": candidate_bundle_hash,
            },
        },
        "gates": {
            "primary_gate_met": primary_gate_met,
            "primary_gate_description": "Candidate test full-denominator accuracy and macro-F1 strictly above baseline on 1663 test rows",
            "delta_accuracy": delta_acc,
            "delta_macro_f1": delta_macro_f1,
            "secondary_recall_gate_met": secondary_recall_gate_met,
            "secondary_recall_gate_description": "No per-class recall drop exceeding 0.02",
            "per_class_drops_exceeding_2pct": per_class_drops_exceeding_2pct,
            "secondary_other_gate_met": secondary_other_gate_met,
            "secondary_other_gate_description": "No increase in OTHER-to-target false command count",
            "baseline_other_to_target": b_other_to_target,
            "candidate_other_to_target": c_other_to_target,
            "delta_other_to_target": delta_other_to_target,
            "secondary_gates_met": secondary_gates_met,
        },
        "paired_deltas_test_rejection": {
            "delta_accuracy": delta_acc,
            "delta_macro_f1": delta_macro_f1,
            "delta_balanced_accuracy": delta_balanced_acc,
            "delta_coverage": delta_coverage,
            "delta_rejection_count": delta_rejection_count,
            "contingency_table": test_contingency,
            "per_class_recall_deltas": per_class_recall_deltas,
        },
        "metrics": metrics_results,
    }

    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="ACC-05 Frozen Personal SVM Paired Benchmark")
    parser.add_argument("--train", default=None, help="Path to train RGB dataset NPZ")
    parser.add_argument("--val-clean", default=None, help="Path to cleaned validation dataset NPZ")
    parser.add_argument("--val-retake", default=None, help="Path to validation retake NPZ")
    parser.add_argument("--test", default=None, help="Path to test RGB dataset NPZ")
    parser.add_argument("--baseline", default=None, help="Path to baseline model bundle JSON")
    parser.add_argument("--audit", default=None, help="Path to ACC-04 validation cleaning audit JSON")
    parser.add_argument("--candidate-out", default=None, help="Path to write candidate bundle JSON")
    parser.add_argument("--predictions-out", default=None, help="Path to write predictions JSON")
    parser.add_argument("--metrics-out", default=None, help="Path to write metrics JSON")
    args = parser.parse_args()

    try:
        print("[INFO] Executing ACC-05 paired benchmark...")
        result = execute_acc05_benchmark(
            train_path=args.train,
            val_clean_path=args.val_clean,
            val_retake_path=args.val_retake,
            test_path=args.test,
            baseline_bundle_path=args.baseline,
            acc04_audit_path=args.audit,
            candidate_bundle_path=args.candidate_out,
            predictions_path=args.predictions_out,
            metrics_path=args.metrics_out,
        )
        print("[SUCCESS] ACC-05 benchmark completed successfully.")
        print(f"          Primary Gate Met: {result['gates']['primary_gate_met']}")
        print(f"          Delta Accuracy:   {result['gates']['delta_accuracy']:+.6f}")
        print(f"          Delta Macro-F1:   {result['gates']['delta_macro_f1']:+.6f}")
        print(f"          Secondary Gates:  {result['gates']['secondary_gates_met']}")
        return 0
    except Exception as e:
        print(f"[ERROR] ACC-05 benchmark failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
