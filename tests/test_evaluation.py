"""Regression tests for ACC-01 evaluation fixes and provenance audit.

Tests verify:
- UNKNOWN prediction handled as false negative / rejection (0.5 recall, 2/3 F1, support=2).
- Rejection column in confusion matrix without adding an extra macro target class.
- Rejection of unexpected ground truth and predicted labels.
- Rejection of missing, empty, or insufficient groups in extract_groups and group_kfold_split.
- Strict zero-leakage fold isolation.
- Both user and session leakage detection for combined vs true custom data.
- Exact 1-to-1 provenance reconstruction between combined, HaGRID, and custom datasets.
- Scaler is fit exclusively on train fold with zero test leakage.
- Deterministic split manifest and reproducible prediction outputs.
"""
from __future__ import annotations

import json
import os
import tempfile
import numpy as np
import pytest

from src.gesture.pose import HandPose
from src.ml.evaluation import compute_metrics, group_kfold_split, leave_one_subject_out
from src.ml.svm import SVMClassifier
from tools.audit.provenance import audit_datasets
from tools.collect_data import load_dataset_npz
from tools.train import extract_groups, evaluate_fold


class TestUnknownRejectionMetrics:
    def test_unknown_exact_regression_spec(self):
        """Exact regression test from task specification:
        compute_metrics([POINT_ONE, POINT_ONE], [POINT_ONE, UNKNOWN], labels=[POINT_ONE])
        must return recall=0.5, F1=2/3, support=2.
        """
        # Test with string labels
        res_str = compute_metrics(
            ["POINT_ONE", "POINT_ONE"],
            ["POINT_ONE", "UNKNOWN"],
            labels=["POINT_ONE"],
        )
        assert res_str["macro_f1"] == pytest.approx(2 / 3, abs=1e-6)
        assert res_str["accuracy"] == 0.5
        assert res_str["balanced_accuracy"] == 0.5
        assert res_str["support"] == 2
        assert res_str["coverage"] == 0.5
        assert res_str["rejection_rate"] == 0.5

        pc = res_str["per_class"]["POINT_ONE"]
        assert pc["precision"] == 1.0
        assert pc["recall"] == 0.5
        assert pc["f1"] == pytest.approx(2 / 3, abs=1e-6)
        assert pc["support"] == 2

        # Confusion matrix has shape (1, 2) with UNKNOWN column
        assert res_str["confusion_matrix"] == [[1, 1]]
        assert res_str["confusion_col_labels"] == ["POINT_ONE", "UNKNOWN"]
        assert res_str["confusion_row_labels"] == ["POINT_ONE"]

        # Test with HandPose enum
        res_enum = compute_metrics(
            [HandPose.POINT_ONE, HandPose.POINT_ONE],
            [HandPose.POINT_ONE, HandPose.UNKNOWN],
            labels=[HandPose.POINT_ONE],
        )
        assert res_enum["macro_f1"] == pytest.approx(2 / 3, abs=1e-6)
        pc_enum = res_enum["per_class"][HandPose.POINT_ONE]
        assert pc_enum["recall"] == 0.5
        assert pc_enum["f1"] == pytest.approx(2 / 3, abs=1e-6)
        assert pc_enum["support"] == 2

    def test_unknown_does_not_add_extra_macro_target_class(self):
        """UNKNOWN must NOT be counted as an extra class in macro_f1."""
        y_true = ["A", "B"]
        y_pred = ["A", "UNKNOWN"]
        res = compute_metrics(y_true, y_pred, labels=["A", "B"])

        # Target classes must only be A and B
        assert set(res["per_class"].keys()) == {"A", "B"}
        # A: precision=1, recall=1, f1=1
        # B: precision=0, recall=0, f1=0 (rejected as UNKNOWN)
        # Macro-F1 = (1.0 + 0.0) / 2 = 0.5
        assert res["macro_f1"] == 0.5
        assert res["confusion_matrix"] == [[1, 0, 0], [0, 0, 1]]
        assert res["confusion_col_labels"] == ["A", "B", "UNKNOWN"]

    def test_unexpected_labels_fail_clearly(self):
        """Unexpected labels must raise ValueError."""
        # Unexpected prediction
        with pytest.raises(ValueError, match="Unexpected prediction label"):
            compute_metrics(["A", "B"], ["A", "UNEXPECTED_LABEL"], labels=["A", "B"])

        # Unexpected ground truth
        with pytest.raises(ValueError, match="Unexpected ground truth label"):
            compute_metrics(["A", "INVALID_TRUE"], ["A", "B"], labels=["A", "B"])


class TestGroupExtractionAndSplitting:
    def test_extract_groups_rejects_missing_key(self):
        """extract_groups must fail clearly when group_key is missing."""
        bad_metadata = [
            {"session_id": "s1", "user_id": "u1"},
            {"session_id": "s2"},  # missing user_id
        ]
        with pytest.raises(ValueError, match="missing required group_key 'user_id'"):
            extract_groups(bad_metadata, group_key="user_id")

    def test_extract_groups_rejects_empty_or_none_value(self):
        """extract_groups must fail clearly when group value is None, empty string, or NaN."""
        bad_metadata_none = [{"user_id": None}]
        with pytest.raises(ValueError, match="empty or invalid group_key"):
            extract_groups(bad_metadata_none, group_key="user_id")

        bad_metadata_empty = [{"user_id": "   "}]
        with pytest.raises(ValueError, match="empty or invalid group_key"):
            extract_groups(bad_metadata_empty, group_key="user_id")

        bad_metadata_nan = [{"user_id": float("nan")}]
        with pytest.raises(ValueError, match="empty or invalid group_key"):
            extract_groups(bad_metadata_nan, group_key="user_id")

    def test_group_kfold_split_rejects_insufficient_groups(self):
        """group_kfold_split must fail clearly when unique groups < n_splits."""
        X = np.ones((10, 2))
        y = np.zeros(10)
        groups = np.array(["g1", "g1", "g2", "g2", "g1", "g2", "g1", "g2", "g1", "g2"])  # only 2 groups

        with pytest.raises(ValueError, match="Insufficient group count"):
            list(group_kfold_split(X, y, groups, n_splits=5))

    def test_group_kfold_split_rejects_missing_or_empty_groups(self):
        """group_kfold_split must fail clearly if any group is None, empty, or NaN."""
        X = np.ones((4, 2))
        y = np.zeros(4)

        with pytest.raises(ValueError, match="None"):
            list(group_kfold_split(X, y, np.array(["g1", None, "g2", "g3"]), n_splits=2))

        with pytest.raises(ValueError, match="empty or whitespace"):
            list(group_kfold_split(X, y, np.array(["g1", "", "g2", "g3"]), n_splits=2))

    def test_real_group_split_isolation_zero_leakage(self):
        """True grouped folds must guarantee zero group leakage between train and test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 10)
        y = rng.choice(["A", "B", "C"], size=n_samples)
        groups = np.array([f"user_{i % 20}" for i in range(n_samples)])

        for train_idx, test_idx in group_kfold_split(X, y, groups, n_splits=5, seed=42):
            tr_g = set(groups[train_idx])
            te_g = set(groups[test_idx])
            assert tr_g.isdisjoint(te_g), f"Leaked groups: {tr_g.intersection(te_g)}"
            assert len(train_idx) + len(test_idx) == n_samples


class TestProvenanceAndLeakage:
    def test_exact_provenance_reconstruction(self):
        """Combined dataset rows must match HaGRID and Custom datasets 1-to-1 exactly."""
        h_feat, h_lbl, h_meta = load_dataset_npz("data/hagrid_airpad.npz")
        c_feat, c_lbl, c_meta = load_dataset_npz("data/custom_airpad.npz")
        comb_feat, comb_lbl, comb_meta = load_dataset_npz("data/combined_airpad.npz")

        n_h = len(h_lbl)
        n_c = len(c_lbl)
        assert len(comb_lbl) == n_h + n_c

        # HaGRID segment exact match
        assert np.array_equal(comb_feat[:n_h], h_feat)
        assert np.array_equal(comb_lbl[:n_h], h_lbl)

        # Custom segment exact match
        assert np.array_equal(comb_feat[n_h:], c_feat)
        assert np.array_equal(comb_lbl[n_h:], c_lbl)

        # Custom rows have true user 'tuana' and true session
        true_users = {m["user_id"] for m in c_meta}
        assert true_users == {"tuana"}
        true_sessions = {m["session_id"] for m in c_meta}
        assert len(true_sessions) == 1

    def test_synthetic_chunks_leak_user_and_session(self):
        """Demonstrate that synthetic chunks in combined_airpad.npz leak real user and session.

        When splitting the 1066 custom samples using the synthetic chunk identities (tuana_chunk0..4),
        every single fold leaks the real user ('tuana') and the real session ('sess_20260929_132034_7a13ed').
        """
        _, _, c_meta = load_dataset_npz("data/custom_airpad.npz")
        _, _, comb_meta = load_dataset_npz("data/combined_airpad.npz")
        custom_comb_meta = comb_meta[60000:]

        synthetic_users = np.array([m["user_id"] for m in custom_comb_meta])
        true_users = np.array([m["user_id"] for m in c_meta])
        true_sessions = np.array([m["session_id"] for m in c_meta])

        assert set(synthetic_users) == {f"tuana_chunk{i}" for i in range(5)}
        assert set(true_users) == {"tuana"}

        # Simulate 5-fold CV using the synthetic chunk IDs
        X_dummy = np.zeros((len(synthetic_users), 2))
        y_dummy = np.zeros(len(synthetic_users))

        leaked_folds_user = 0
        leaked_folds_session = 0

        for tr_idx, te_idx in group_kfold_split(X_dummy, y_dummy, synthetic_users, n_splits=5, seed=42):
            # Synthetic users appear disjoint:
            assert set(synthetic_users[tr_idx]).isdisjoint(set(synthetic_users[te_idx]))

            # BUT true user leaks completely:
            tr_true_users = set(true_users[tr_idx])
            te_true_users = set(true_users[te_idx])
            if not tr_true_users.isdisjoint(te_true_users):
                leaked_folds_user += 1

            # AND true session leaks completely:
            tr_true_sessions = set(true_sessions[tr_idx])
            te_true_sessions = set(true_sessions[te_idx])
            if not tr_true_sessions.isdisjoint(te_true_sessions):
                leaked_folds_session += 1

        assert leaked_folds_user == 5, "All 5 folds must exhibit true user leakage under synthetic chunks!"
        assert leaked_folds_session == 5, "All 5 folds must exhibit true session leakage under synthetic chunks!"

    def test_custom_classes_have_insufficient_users_for_cv(self):
        """Under true user groups, 1 user cannot be partitioned into 5 folds."""
        c_feat, c_lbl, c_meta = load_dataset_npz("data/custom_airpad.npz")
        true_users = np.array([m["user_id"] for m in c_meta])
        assert len(np.unique(true_users)) == 1

        with pytest.raises(ValueError, match="Insufficient group count"):
            list(group_kfold_split(c_feat, c_lbl, true_users, n_splits=5))


class TestScalerAndReproducibility:
    def test_scaler_fit_exclusively_on_train_evaluator_spy(self, tmp_path):
        """Verify scaler is fit exclusively on train fold through actual evaluator fit path."""
        from tools.evaluate import run_evaluation
        spy: list[dict] = []
        run_evaluation(
            dataset_path="data/hagrid_airpad.npz",
            sample_limit=300,
            output_dir=str(tmp_path),
            scaler_spy=spy,
        )
        assert len(spy) == 5, "Must capture scaler state across all 5 folds"
        for fold_info in spy:
            assert np.allclose(fold_info["scaler_mean"], fold_info["train_mean"], atol=1e-6), (
                f"Fold {fold_info['fold']} scaler mean does not match training fold mean"
            )
            # Scaler mean must NOT match test fold mean
            assert not np.allclose(fold_info["scaler_mean"], fold_info["test_mean"]), (
                f"Fold {fold_info['fold']} scaler mean leaked test fold data"
            )
            # Positive variance features match train std
            pos_var = fold_info["train_std"] > 1e-5
            assert np.allclose(fold_info["scaler_scale"][pos_var], fold_info["train_std"][pos_var], atol=1e-5), (
                f"Fold {fold_info['fold']} scaler scale does not match training fold std on varying features"
            )

    def test_deterministic_split_and_predictions(self, tmp_path):
        """Running evaluation with fixed seed must generate identical folds and predictions."""
        from tools.evaluate import run_evaluation

        out1 = str(tmp_path / "run1")
        out2 = str(tmp_path / "run2")

        res1 = run_evaluation(
            dataset_path="data/hagrid_airpad.npz",
            sample_limit=300,
            seed=42,
            output_dir=out1,
        )
        res2 = run_evaluation(
            dataset_path="data/hagrid_airpad.npz",
            sample_limit=300,
            seed=42,
            output_dir=out2,
        )

        assert res1["clean_baseline_post_rejection"]["macro_f1"] == res2["clean_baseline_post_rejection"]["macro_f1"]
        assert res1["clean_baseline_post_rejection"]["accuracy"] == res2["clean_baseline_post_rejection"]["accuracy"]

        # Compare predictions files
        preds1 = open(os.path.join(out1, "predictions.jsonl")).read()
        preds2 = open(os.path.join(out2, "predictions.jsonl")).read()
        assert preds1 == preds2

    def test_synthetic_chunk_rejection_without_manifest(self):
        """Passing combined dataset with synthetic chunks must fail if no corrected manifest."""
        from tools.evaluate import run_evaluation

        with pytest.raises(ValueError, match="synthetic chunk groupings"):
            run_evaluation(
                dataset_path="data/combined_airpad.npz",
                corrected_manifest_path=None,
            )

    def test_source_row_ids_preserved_after_subsampling(self, tmp_path):
        """Prediction records and manifest must retain exact original dataset row IDs."""
        from tools.evaluate import run_evaluation

        out = str(tmp_path / "eval")
        res = run_evaluation(
            dataset_path="data/hagrid_airpad.npz",
            sample_limit=600,
            output_dir=out,
        )
        # Read predictions
        pred_lines = [json.loads(line) for line in open(os.path.join(out, "predictions.jsonl"))]
        assert len(pred_lines) == 600

        # Source row IDs must not simply be 0..599 (they span the 60000-sample dataset)
        source_ids = [p["source_row_id"] for p in pred_lines]
        assert max(source_ids) > 10000, "Source row IDs must reflect original positions across classes"
        assert len(set(source_ids)) == 600, "All source row IDs must be unique"

        # Check split manifest has matching test_source_row_ids
        manifest = json.load(open(os.path.join(out, "split_manifest.json")))
        all_manifest_test_ids = []
        for fold in manifest["folds"]:
            all_manifest_test_ids.extend(fold["test_source_row_ids"])
        assert sorted(all_manifest_test_ids) == sorted(source_ids)

    def test_multiplicity_audit_exact_duplicates(self, tmp_path):
        """Provenance audit must identify exact duplicates and use tmp_path for test outputs."""
        manifest_tmp = str(tmp_path / "manifest.json")
        report_tmp = str(tmp_path / "report.json")
        report = audit_datasets(
            combined_path="data/combined_airpad.npz",
            hagrid_path="data/hagrid_airpad.npz",
            custom_path="data/custom_airpad.npz",
            manifest_out=manifest_tmp,
            report_out=report_tmp,
        )
        assert report["multiplicity_audit"]["duplicate_feature_label_keys"] == 43
        assert report["multiplicity_audit"]["max_multiplicity"] == 2
        assert report["multiplicity_audit"]["disambiguated_by_timestamp"] == 43
        assert report["multiplicity_audit"]["all_duplicates_agree_on_real_user_and_session"] is True
        assert os.path.exists(manifest_tmp)
        assert os.path.exists(report_tmp)

    def test_target_class_with_single_user_fails_user_support_check(self, tmp_path):
        """Target class with < 2 users must fail user support check before evaluation."""
        from tools.evaluate import run_evaluation

        with pytest.raises(ValueError, match="lacks independent user support"):
            run_evaluation(
                dataset_path="data/custom_airpad.npz",
                output_dir=str(tmp_path),
            )

    def test_manifest_validation_rejects_corrupted_records(self, tmp_path):
        """Corrected manifest validation must reject wrong count, coverage gaps, or label mismatch."""
        from tools.evaluate import apply_and_validate_corrected_manifest

        # Load small dataset
        c_feat, c_lbl, c_meta = load_dataset_npz("data/custom_airpad.npz")
        h = "dummy_hash"

        # 1. Test hash mismatch
        bad_manifest_path = str(tmp_path / "bad_hash.json")
        with open(bad_manifest_path, "w") as f:
            json.dump({"metadata": {"combined_hash": "wrong_hash"}, "records": []}, f)
        with pytest.raises(ValueError, match="hash mismatch"):
            apply_and_validate_corrected_manifest(c_meta, c_lbl, bad_manifest_path, dataset_hash=h)

        # 2. Test record count mismatch
        bad_count_path = str(tmp_path / "bad_count.json")
        with open(bad_count_path, "w") as f:
            json.dump({"metadata": {"combined_hash": h}, "records": [{"sample_id": 0}]}, f)
        with pytest.raises(ValueError, match="record count mismatch"):
            apply_and_validate_corrected_manifest(c_meta, c_lbl, bad_count_path, dataset_hash=h)
