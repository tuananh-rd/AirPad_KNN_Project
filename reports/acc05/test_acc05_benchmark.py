"""ACC-05 Isolated Benchmark Unit Tests.

Proves:
1. End-to-end benchmark orchestration with spies asserting:
   - Only train rows reach candidate training and scaler fit (zero leakage of val/test).
   - Both predictors see identical held-out row counts/IDs and frozen thresholds (decision=0.0, margin=0.15).
2. Frozen hyperparameters (C=10.0, gamma=0.01, class_weight='balanced').
3. Rejection metrics computation with UNKNOWN handling verified against sklearn.metrics.
4. Fail-closed guards:
   - Wrong pinned hash
   - Pre-existing output collision
   - Target path aliasing input dataset
   - Targets aliasing each other
   - Cross-role session overlap
   - Cross-role exact feature duplicate
   - Transition frame rejection
5. OTHER-to-target false command counter accuracy.
6. Paired contingency table accounting.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import reports.acc05.benchmark_personal_svm as bench
from reports.acc05.benchmark_personal_svm import (
    PINNED_HASHES,
    STATIC_CLASSES,
    TARGET_ACTION_CLASSES,
    compute_file_sha256,
    compute_other_to_target_count,
    compute_paired_contingency,
    compute_prediction_margins,
    execute_acc05_benchmark,
    validate_benchmark_paths,
    validate_sample_schema_and_integrity,
    verify_pinned_hash,
)
from src.ml.evaluation import compute_metrics
from src.ml.svm import SVMClassifier
from tools.collect_data import save_dataset_npz
from tools.train import FeatureScaler, ModelBundle, create_model_bundle, load_model_from_bundle


def test_train_only_scaler_and_model_fit():
    """Verify that candidate scaler and model fit exclusively on training data."""
    rng = np.random.RandomState(42)
    n_train = 50
    n_val = 30
    X_train = rng.randn(n_train, 63).astype(np.float32)
    y_train = np.array(["PINCH_CLOSED"] * 25 + ["POINT_ONE"] * 25)

    X_val = rng.randn(n_val, 63).astype(np.float32) + 10.0  # distinct distribution
    y_val = np.array(["PINCH_CLOSED"] * 15 + ["POINT_ONE"] * 15)

    # 1. Fit scaler on train only
    scaler = FeatureScaler().fit(X_train)
    assert np.allclose(scaler.mean_, np.mean(X_train, axis=0), atol=1e-5)
    assert np.allclose(scaler.scale_, np.std(X_train, axis=0), atol=1e-5)
    # Ensure val data did not affect scaler parameters
    assert not np.allclose(scaler.mean_, np.mean(np.vstack([X_train, X_val]), axis=0))

    # 2. Check bundle serialization retains train-only scaler and training rows
    bundle = create_model_bundle(
        X=X_train,
        y=y_train,
        C=10.0,
        gamma=0.01,
        class_weight="balanced",
        labels=["PINCH_CLOSED", "POINT_ONE"],
    )
    assert bundle.train_samples == n_train
    assert np.allclose(bundle.metadata["scaler"]["mean"], scaler.mean_.tolist())
    assert bundle.hyperparameters == {"C": 10.0, "gamma": 0.01, "class_weight": "balanced"}


def test_mocked_end_to_end_execute_benchmark_spies_scaler_and_fit(tmp_path):
    """Spy execute_acc05_benchmark asserting train-only fit, identical held-out rows, and frozen thresholds."""
    rng = np.random.RandomState(123)

    # Create synthetic datasets with distinct distributions
    n_train, n_val_clean, n_val_retake, n_test = 40, 25, 15, 30

    def make_dataset(n_samples, offset, user_id, session_id):
        feats = (rng.randn(n_samples, 63) + offset).astype(np.float32)
        half = n_samples // 2
        labels = np.array(["PINCH_CLOSED"] * half + ["POINT_ONE"] * (n_samples - half))
        meta = [
            {"user_id": user_id, "session_id": session_id, "label": labels[i], "is_transition": False, "timestamp": 1000.0 + i}
            for i in range(n_samples)
        ]
        return feats, labels, meta

    train_f, train_l, train_m = make_dataset(n_train, 0.0, "tuana", "sess_syn_train")
    val_c_f, val_c_l, val_c_m = make_dataset(n_val_clean, 5.0, "tuana", "sess_syn_val_clean")
    val_r_f, val_r_l, val_r_m = make_dataset(n_val_retake, 10.0, "tuana", "sess_syn_val_retake")
    test_f, test_l, test_m = make_dataset(n_test, 15.0, "tuana", "sess_syn_test")

    train_p = str(tmp_path / "syn_train.npz")
    val_c_p = str(tmp_path / "syn_val_clean.npz")
    val_r_p = str(tmp_path / "syn_val_retake.npz")
    test_p = str(tmp_path / "syn_test.npz")

    save_dataset_npz(train_p, train_f, train_l, train_m)
    save_dataset_npz(val_c_p, val_c_f, val_c_l, val_c_m)
    save_dataset_npz(val_r_p, val_r_f, val_r_l, val_r_m)
    save_dataset_npz(test_p, test_f, test_l, test_m)

    train_h = compute_file_sha256(train_p)
    val_c_h = compute_file_sha256(val_c_p)
    val_r_h = compute_file_sha256(val_r_p)
    test_h = compute_file_sha256(test_p)

    audit_p = str(tmp_path / "syn_audit.json")
    with open(audit_p, "w", encoding="utf-8") as f:
        json.dump({
            "retained_samples_mapping": [
                {"output_row_idx": i, "original_row_idx": i, "original_source_file": "raw.npz", "original_source_hash": "abc"}
                for i in range(n_val_clean)
            ]
        }, f)
    audit_h = compute_file_sha256(audit_p)

    # Synthetic baseline bundle with distinct training distribution
    base_train_f, base_train_l, _ = make_dataset(20, -20.0, "tuana", "sess_syn_baseline")
    dummy_bundle_p = str(tmp_path / "syn_baseline.json")
    base_bundle = create_model_bundle(base_train_f, base_train_l, C=10.0, gamma=0.01, labels=STATIC_CLASSES)
    base_bundle.save(dummy_bundle_p)
    base_h = compute_file_sha256(dummy_bundle_p)

    # Patch pinned hashes dict for isolated synthetic run
    mock_pinned = {
        "train_rgb": {"rel_path": train_p, "sha256": train_h, "expected_rows": n_train, "expected_user_id": "tuana", "expected_session_id": "sess_syn_train"},
        "val_clean": {"rel_path": val_c_p, "sha256": val_c_h, "expected_rows": n_val_clean, "expected_user_id": "tuana", "expected_session_id": "sess_syn_val_clean"},
        "val_retake": {"rel_path": val_r_p, "sha256": val_r_h, "expected_rows": n_val_retake, "expected_user_id": "tuana", "expected_session_id": "sess_syn_val_retake"},
        "test_rgb": {"rel_path": test_p, "sha256": test_h, "expected_rows": n_test, "expected_user_id": "tuana", "expected_session_id": "sess_syn_test"},
        "baseline_bundle": {"rel_path": dummy_bundle_p, "sha256": base_h},
        "acc04_audit": {"rel_path": audit_p, "sha256": audit_h},
    }

    out_cand_p = str(tmp_path / "out_candidate.json")
    out_pred_p = str(tmp_path / "out_pred.json")
    out_met_p = str(tmp_path / "out_metrics.json")

    original_fit = SVMClassifier._fit
    fit_calls = []

    def spy_fit(self, X, y, scaler=None):
        fit_calls.append({
            "self": self,
            "C": self.C,
            "gamma": self.gamma,
            "class_weight": self.class_weight,
            "X": np.copy(X),
            "y": np.copy(y),
            "scaler": scaler,
        })
        return original_fit(self, X, y, scaler=scaler)

    original_pwr = SVMClassifier.predict_with_rejection
    pwr_calls = []

    def spy_pwr(self, X, *args, **kwargs):
        preds = original_pwr(self, X, *args, **kwargs)
        pwr_calls.append({
            "self": self,
            "X": np.copy(X),
            "preds": preds,
            "args": args,
            "kwargs": kwargs,
        })
        return preds

    with patch.dict(bench.PINNED_HASHES, mock_pinned, clear=False), \
         patch("reports.acc05.benchmark_personal_svm.create_model_bundle", wraps=create_model_bundle) as spy_bundle, \
         patch.object(SVMClassifier, "_fit", autospec=True, side_effect=spy_fit), \
         patch.object(SVMClassifier, "predict_with_rejection", autospec=True, side_effect=spy_pwr):

        res = execute_acc05_benchmark(
            train_path=train_p,
            val_clean_path=val_c_p,
            val_retake_path=val_r_p,
            test_path=test_p,
            baseline_bundle_path=dummy_bundle_p,
            acc04_audit_path=audit_p,
            candidate_bundle_path=out_cand_p,
            predictions_path=out_pred_p,
            metrics_path=out_met_p,
        )

        # 1. Assert create_model_bundle was called with exact train X values and y labels
        assert spy_bundle.call_count == 1
        call_kwargs = spy_bundle.call_args[1] if spy_bundle.call_args[1] else {}
        call_args = spy_bundle.call_args[0]
        X_passed = call_kwargs.get("X", call_args[0] if len(call_args) > 0 else None)
        y_passed = call_kwargs.get("y", call_args[1] if len(call_args) > 1 else None)
        assert np.array_equal(X_passed, train_f)
        assert np.array_equal(y_passed, train_l)

        # 2. Assert candidate bundle artifact embeds exact train rows and train-only scaler
        cand_b_loaded = ModelBundle.load(out_cand_p)
        assert np.array_equal(cand_b_loaded.model_data["X_train"], train_f)
        assert np.array_equal(cand_b_loaded.model_data["y_train"], train_l)
        assert np.allclose(cand_b_loaded.metadata["scaler"]["mean"], np.mean(train_f, axis=0))
        assert np.allclose(cand_b_loaded.metadata["scaler"]["scale"], np.std(train_f, axis=0))
        assert cand_b_loaded.hyperparameters == {"C": 10.0, "gamma": 0.01, "class_weight": "balanced"}

        # 3. Assert real _fit spy verified restored candidate receives exact train X/y and frozen params
        restored_cand_calls = [
            c for c in fit_calls
            if c["scaler"] is not None and np.array_equal(c["X"], train_f) and np.array_equal(c["y"], train_l)
        ]
        assert len(restored_cand_calls) == 1
        restored_call = restored_cand_calls[0]
        assert restored_call["C"] == 10.0
        assert restored_call["gamma"] == 0.01
        assert restored_call["class_weight"] == "balanced"
        assert np.allclose(restored_call["scaler"].mean_, np.mean(train_f, axis=0))
        assert np.allclose(restored_call["scaler"].scale_, np.std(train_f, axis=0))

        # 4. Assert predict_with_rejection executed 4 times (baseline val, cand val, baseline test, cand test)
        assert len(pwr_calls) == 4
        for c in pwr_calls:
            assert c["kwargs"].get("decision_threshold") == 0.0
            assert c["kwargs"].get("margin_threshold") == 0.15

        # 5. Assert identical held-out X values for both models on validation and test
        val_expected_X = np.vstack([val_c_f, val_r_f])
        assert np.array_equal(pwr_calls[0]["X"], pwr_calls[1]["X"])
        assert np.array_equal(pwr_calls[0]["X"], val_expected_X)
        assert np.array_equal(pwr_calls[2]["X"], pwr_calls[3]["X"])
        assert np.array_equal(pwr_calls[2]["X"], test_f)

        # 6. Assert all four returned supports are 40 validation and 30 test
        assert res["metrics"]["validation"]["baseline_raw"]["support"] == 40
        assert res["metrics"]["validation"]["baseline_rejection"]["support"] == 40
        assert res["metrics"]["validation"]["candidate_raw"]["support"] == 40
        assert res["metrics"]["validation"]["candidate_rejection"]["support"] == 40

        assert res["metrics"]["test"]["baseline_raw"]["support"] == 30
        assert res["metrics"]["test"]["baseline_rejection"]["support"] == 30
        assert res["metrics"]["test"]["candidate_raw"]["support"] == 30
        assert res["metrics"]["test"]["candidate_rejection"]["support"] == 30

        # 7. Assert source row coverage and concrete non-mock string predictions in saved artifacts
        with open(out_pred_p, "r", encoding="utf-8") as pf:
            saved_preds_payload = json.load(pf)
        val_records = saved_preds_payload["validation_predictions"]
        test_records = saved_preds_payload["test_predictions"]
        assert len(val_records) == 40
        assert len(test_records) == 30
        assert saved_preds_payload["validation_samples_count"] == 40
        assert saved_preds_payload["test_samples_count"] == 30

        val_clean_rel = os.path.relpath(val_c_p, bench.PROJECT_ROOT).replace("\\", "/")
        val_retake_rel = os.path.relpath(val_r_p, bench.PROJECT_ROOT).replace("\\", "/")
        test_rel = os.path.relpath(test_p, bench.PROJECT_ROOT).replace("\\", "/")

        val_clean_rows = [r["source_row_idx"] for r in val_records if r["source_file"] == val_clean_rel]
        val_retake_rows = [r["source_row_idx"] for r in val_records if r["source_file"] == val_retake_rel]
        test_rows = [r["source_row_idx"] for r in test_records if r["source_file"] == test_rel]

        assert sorted(val_clean_rows) == list(range(n_val_clean))
        assert sorted(val_retake_rows) == list(range(n_val_retake))
        assert sorted(test_rows) == list(range(n_test))

        for r in val_records + test_records:
            assert isinstance(r["baseline_raw_pred"], str)
            assert isinstance(r["baseline_rej_pred"], str)
            assert isinstance(r["candidate_raw_pred"], str)
            assert isinstance(r["candidate_rej_pred"], str)
            assert r["candidate_rej_pred"] in STATIC_CLASSES + ["UNKNOWN"]


def test_rejection_metrics_and_unknown_handling():
    """Verify compute_metrics treats UNKNOWN as false negative in full denominator."""
    y_true = ["PINCH_CLOSED", "POINT_ONE", "FIST", "OTHER"]
    y_pred = ["PINCH_CLOSED", "UNKNOWN", "FIST", "PINCH_CLOSED"]
    labels = ["PINCH_CLOSED", "POINT_ONE", "FIST", "OTHER"]

    metrics = compute_metrics(y_true, y_pred, labels=labels, unknown_label="UNKNOWN")

    assert metrics["support"] == 4
    assert metrics["accuracy"] == 2 / 4  # 0.5
    assert metrics["coverage"] == 3 / 4  # 0.75
    assert metrics["rejection_rate"] == 1 / 4  # 0.25

    per_class = metrics["per_class"]
    assert per_class["POINT_ONE"]["recall"] == 0.0
    assert per_class["FIST"]["recall"] == 1.0


def test_rejection_metrics_comparison_with_sklearn():
    """Compare compute_metrics calculation directly against sklearn.metrics."""
    y_true = ["PINCH_CLOSED"] * 30 + ["POINT_ONE"] * 30 + ["FIST"] * 20 + ["OTHER"] * 20
    # Create predictions where some are correct, some swapped, some UNKNOWN
    y_pred = (
        ["PINCH_CLOSED"] * 25 + ["UNKNOWN"] * 5 +
        ["POINT_ONE"] * 20 + ["FIST"] * 5 + ["UNKNOWN"] * 5 +
        ["FIST"] * 18 + ["UNKNOWN"] * 2 +
        ["OTHER"] * 15 + ["POINT_ONE"] * 5
    )

    labels = ["PINCH_CLOSED", "POINT_ONE", "FIST", "OTHER"]

    custom_m = compute_metrics(y_true, y_pred, labels=labels, unknown_label="UNKNOWN")

    # 1. Compare Accuracy: sklearn accuracy_score on full denominator
    sk_acc = accuracy_score(y_true, y_pred)
    assert abs(custom_m["accuracy"] - sk_acc) < 1e-6

    # 2. Compare Macro-F1: sklearn precision_recall_fscore_support on target labels
    # When UNKNOWN is not in `labels`, sklearn treats it as a prediction outside target classes (FP for none, FN for true class)
    sk_prec, sk_rec, sk_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    for idx, cls_name in enumerate(labels):
        assert abs(custom_m["per_class"][cls_name]["precision"] - sk_prec[idx]) < 1e-6
        assert abs(custom_m["per_class"][cls_name]["recall"] - sk_rec[idx]) < 1e-6
        assert abs(custom_m["per_class"][cls_name]["f1"] - sk_f1[idx]) < 1e-6

    sk_macro_f1 = np.mean(sk_f1)
    assert abs(custom_m["macro_f1"] - sk_macro_f1) < 1e-6


def test_fails_closed_on_wrong_pinned_hash(tmp_path):
    """Must fail closed with ValueError when SHA-256 does not match pinned hash."""
    fake_file = tmp_path / "fake.npz"
    fake_file.write_bytes(b"corrupted contents")

    with pytest.raises(ValueError, match=r"\[FAIL-CLOSED\] Hash mismatch"):
        verify_pinned_hash(str(fake_file), "0000000000000000000000000000000000000000000000000000000000000000")


def test_fails_closed_when_output_already_exists(tmp_path):
    """Must fail closed when any benchmark output target already exists."""
    existing_file = tmp_path / "candidate_personal_svm.json"
    existing_file.write_text("{}", encoding="utf-8")

    input_paths = [("train_rgb", "dummy_train.npz")]
    output_paths = [
        ("candidate_bundle", str(existing_file)),
        ("predictions_json", str(tmp_path / "pred.json")),
    ]

    with pytest.raises(FileExistsError, match=r"\[FAIL-CLOSED\] Target output file already exists"):
        validate_benchmark_paths(input_paths, output_paths)


def test_fails_closed_when_output_aliases_input(tmp_path):
    """Must fail closed when an output path aliases an input dataset."""
    in_path = str(tmp_path / "train.npz")
    input_paths = [("train_rgb", in_path)]
    output_paths = [("candidate_bundle", in_path)]

    with pytest.raises(ValueError, match=r"\[FAIL-CLOSED\] Target path candidate_bundle .* aliases input path train_rgb"):
        validate_benchmark_paths(input_paths, output_paths)


def test_fails_closed_when_outputs_alias_each_other(tmp_path):
    """Must fail closed when two output paths resolve to the same location."""
    same_path = str(tmp_path / "output.json")
    input_paths = [("train_rgb", "train.npz")]
    output_paths = [
        ("predictions_json", same_path),
        ("metrics_json", same_path),
    ]

    with pytest.raises(ValueError, match=r"resolve to the same filesystem location"):
        validate_benchmark_paths(input_paths, output_paths)


def test_fails_closed_on_cross_role_session_overlap(tmp_path):
    """Must fail closed when train and test share a session ID."""
    rng = np.random.RandomState(42)
    feats = rng.randn(10, 63).astype(np.float32)
    labels = np.array(["PINCH_CLOSED"] * 10)
    meta = [
        {"user_id": "tuana", "session_id": "shared_session_xyz", "label": "PINCH_CLOSED", "is_transition": False, "timestamp": 1000.0}
        for _ in range(10)
    ]

    train_p = str(tmp_path / "t.npz")
    val_c_p = str(tmp_path / "vc.npz")
    val_r_p = str(tmp_path / "vr.npz")
    test_p = str(tmp_path / "te.npz")

    save_dataset_npz(train_p, feats, labels, meta)
    save_dataset_npz(val_c_p, feats, labels, [{"user_id": "tuana", "session_id": "val_sess_c", "label": "PINCH_CLOSED", "is_transition": False, "timestamp": 1000.0}] * 10)
    save_dataset_npz(val_r_p, feats, labels, [{"user_id": "tuana", "session_id": "val_sess_r", "label": "PINCH_CLOSED", "is_transition": False, "timestamp": 1000.0}] * 10)
    save_dataset_npz(test_p, feats, labels, meta)  # shared session

    audit_p = str(tmp_path / "audit.json")
    with open(audit_p, "w", encoding="utf-8") as f:
        json.dump({"retained_samples_mapping": [{"output_row_idx": i, "original_row_idx": i, "original_source_file": "raw.npz", "original_source_hash": "abc"} for i in range(10)]}, f)

    mock_pinned = {
        "train_rgb": {"rel_path": train_p, "sha256": compute_file_sha256(train_p), "expected_rows": 10, "expected_user_id": "tuana", "expected_session_id": "shared_session_xyz"},
        "val_clean": {"rel_path": val_c_p, "sha256": compute_file_sha256(val_c_p), "expected_rows": 10, "expected_user_id": "tuana", "expected_session_id": "val_sess_c"},
        "val_retake": {"rel_path": val_r_p, "sha256": compute_file_sha256(val_r_p), "expected_rows": 10, "expected_user_id": "tuana", "expected_session_id": "val_sess_r"},
        "test_rgb": {"rel_path": test_p, "sha256": compute_file_sha256(test_p), "expected_rows": 10, "expected_user_id": "tuana", "expected_session_id": "shared_session_xyz"},
        "baseline_bundle": {"rel_path": "dummy.json", "sha256": "dummy"},
        "acc04_audit": {"rel_path": audit_p, "sha256": compute_file_sha256(audit_p)},
    }

    with patch.dict(bench.PINNED_HASHES, mock_pinned, clear=False), \
         patch("reports.acc05.benchmark_personal_svm.verify_pinned_hash", return_value="0" * 64):
        with pytest.raises(ValueError, match=r"\[FAIL-CLOSED\] Overlapping sessions"):
            execute_acc05_benchmark(
                train_path=train_p,
                val_clean_path=val_c_p,
                val_retake_path=val_r_p,
                test_path=test_p,
                acc04_audit_path=audit_p,
                candidate_bundle_path=str(tmp_path / "cand.json"),
                predictions_path=str(tmp_path / "pred.json"),
                metrics_path=str(tmp_path / "met.json"),
            )


def test_fails_closed_on_cross_role_feature_duplicates(tmp_path):
    """Must fail closed when an exact feature vector appears across train and test."""
    rng = np.random.RandomState(42)
    feats = rng.randn(10, 63).astype(np.float32)
    labels = np.array(["PINCH_CLOSED"] * 10)

    train_p = str(tmp_path / "t.npz")
    val_c_p = str(tmp_path / "vc.npz")
    val_r_p = str(tmp_path / "vr.npz")
    test_p = str(tmp_path / "te.npz")

    save_dataset_npz(train_p, feats, labels, [{"user_id": "tuana", "session_id": "sess_t", "label": "PINCH_CLOSED", "is_transition": False, "timestamp": 1000.0}] * 10)
    save_dataset_npz(val_c_p, feats + 1.0, labels, [{"user_id": "tuana", "session_id": "sess_vc", "label": "PINCH_CLOSED", "is_transition": False, "timestamp": 1000.0}] * 10)
    save_dataset_npz(val_r_p, feats + 2.0, labels, [{"user_id": "tuana", "session_id": "sess_vr", "label": "PINCH_CLOSED", "is_transition": False, "timestamp": 1000.0}] * 10)
    # test has exact same features as train
    save_dataset_npz(test_p, feats, labels, [{"user_id": "tuana", "session_id": "sess_te", "label": "PINCH_CLOSED", "is_transition": False, "timestamp": 1000.0}] * 10)

    audit_p = str(tmp_path / "audit.json")
    with open(audit_p, "w", encoding="utf-8") as f:
        json.dump({"retained_samples_mapping": [{"output_row_idx": i, "original_row_idx": i, "original_source_file": "raw.npz", "original_source_hash": "abc"} for i in range(10)]}, f)

    mock_pinned = {
        "train_rgb": {"rel_path": train_p, "sha256": compute_file_sha256(train_p), "expected_rows": 10, "expected_user_id": "tuana", "expected_session_id": "sess_t"},
        "val_clean": {"rel_path": val_c_p, "sha256": compute_file_sha256(val_c_p), "expected_rows": 10, "expected_user_id": "tuana", "expected_session_id": "sess_vc"},
        "val_retake": {"rel_path": val_r_p, "sha256": compute_file_sha256(val_r_p), "expected_rows": 10, "expected_user_id": "tuana", "expected_session_id": "sess_vr"},
        "test_rgb": {"rel_path": test_p, "sha256": compute_file_sha256(test_p), "expected_rows": 10, "expected_user_id": "tuana", "expected_session_id": "sess_te"},
        "baseline_bundle": {"rel_path": "dummy.json", "sha256": "dummy"},
        "acc04_audit": {"rel_path": audit_p, "sha256": compute_file_sha256(audit_p)},
    }

    with patch.dict(bench.PINNED_HASHES, mock_pinned, clear=False), \
         patch("reports.acc05.benchmark_personal_svm.verify_pinned_hash", return_value="0" * 64):
        with pytest.raises(ValueError, match=r"\[FAIL-CLOSED\] Detected .* exact feature duplicates"):
            execute_acc05_benchmark(
                train_path=train_p,
                val_clean_path=val_c_p,
                val_retake_path=val_r_p,
                test_path=test_p,
                acc04_audit_path=audit_p,
                candidate_bundle_path=str(tmp_path / "cand.json"),
                predictions_path=str(tmp_path / "pred.json"),
                metrics_path=str(tmp_path / "met.json"),
            )


def test_fails_closed_on_transition_frame():
    """Must fail closed if any frame in validation or test is marked is_transition=True."""
    feats = np.zeros((5, 63), dtype=np.float32)
    labels = np.array(["PINCH_CLOSED"] * 5)
    meta = [
        {"user_id": "tuana", "session_id": "sess_1", "label": "PINCH_CLOSED", "is_transition": False},
        {"user_id": "tuana", "session_id": "sess_1", "label": "PINCH_CLOSED", "is_transition": True},  # violation
        {"user_id": "tuana", "session_id": "sess_1", "label": "PINCH_CLOSED", "is_transition": False},
        {"user_id": "tuana", "session_id": "sess_1", "label": "PINCH_CLOSED", "is_transition": False},
        {"user_id": "tuana", "session_id": "sess_1", "label": "PINCH_CLOSED", "is_transition": False},
    ]

    with pytest.raises(ValueError, match=r"is_transition must be strictly False"):
        validate_sample_schema_and_integrity(
            feats, labels, meta, expected_rows=5, expected_user_id="tuana",
            expected_session_id="sess_1", source_path="dummy.npz", source_hash="dummy",
            dataset_name="test_ds",
        )


def test_other_to_target_count_calculation():
    """Prove OTHER-to-target false command counter identifies dangerous false triggers."""
    y_true = ["OTHER", "OTHER", "PINCH_CLOSED", "OTHER", "POINT_ONE"]
    y_pred = ["PINCH_CLOSED", "UNKNOWN", "PINCH_CLOSED", "OTHER", "FIST"]

    count = compute_other_to_target_count(y_true, y_pred)
    assert count == 1


def test_paired_contingency_table():
    """Prove contingency table accounts for both models correctly."""
    y_true = ["A", "B", "C", "D"]
    y_base = ["A", "B", "X", "Y"]  # A, B correct
    y_cand = ["A", "X", "C", "Y"]  # A, C correct

    table = compute_paired_contingency(y_true, y_base, y_cand)
    assert table["both_correct"] == 1
    assert table["baseline_only_correct"] == 1
    assert table["candidate_only_correct"] == 1
    assert table["both_wrong"] == 1
    assert table["total"] == 4


if __name__ == "__main__":
    pytest.main(["-v", __file__])
