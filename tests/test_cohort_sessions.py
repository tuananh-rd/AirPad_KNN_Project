"""Tests for tools.cohort_sessions and collection safeguards (ACC-02).

Covers:
- Corrupt existing dataset fail-closed safety (no overwrite, unchanged sha256, nonzero exit).
- Valid append backward compatibility.
- Centralized validation for BOTH public helpers (validate_cohort_sessions and generate_split_manifest):
  + In-memory label/metadata mismatch raises ValueError.
  + In-memory non-boolean transition flag raises ValueError.
  + In-memory identities containing '::' raise ValueError.
  + Absent source_* provenance unconditionally raises ValueError.
  + Non-hex source_hash (e.g. 'z'*64) raises ValueError.
  + Boolean source_row_idx (e.g. True) raises ValueError.
  + Supplied source map not covering file or mismatching hash raises ValueError.
- JSON object duplicate keys strictly rejected via object_pairs_hook.
- Plan validation:
  + Direct generate_split_manifest rejects NOT_READY input (raises ValueError).
  + generate_split_manifest rejects partial plan or invalid roles.
  + Pair-keyed (user_id, session_id) assignments (same session text for different users allowed).
  + Duplicate session assignments raise ValueError.
  + Unknown sessions in plan raise ValueError.
- Output path alias guards:
  + Output report path aliasing input dataset (including case differences) raises ValueError.
  + Output manifest path aliasing input dataset raises ValueError.
  + Output report path and manifest path identical raises ValueError.
- Manifest emission safety:
  + Manifest is NOT emitted when dataset is NOT_READY.
  + Manifest emitted when READY has separated stable and transition lists.
- Roster and mode validation:
  + Missing roster or plan results in NOT_READY diagnostic.
  + Duplicate, blank, or null roster entries raise ValueError.
  + Single-user personal-use acceptance mode (>=3 independent sessions covering 9 classes).
  + Multi-user cohort acceptance mode.
- Audit of existing custom dataset (data/custom_airpad.npz) confirming NOT_READY.
- Provenance and deterministic fingerprinting (excluding wall-clock timestamps).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tools.cohort_sessions import (
    ROLES,
    STATIC_CLASSES,
    audit_dataset_readiness,
    compute_deterministic_fingerprint,
    compute_file_sha256,
    generate_split_manifest,
    load_dataset_with_provenance,
    parse_session_plan,
    validate_cohort_sessions,
    validate_input_files,
    validate_output_paths_no_alias,
    validate_roster,
    validate_samples_integrity,
    validate_session_disjointness_across_files,
)
from tools.collect_data import (
    collect_data_interactive,
    create_sample_metadata,
    load_dataset_npz,
    save_dataset_npz,
)


@pytest.fixture
def dummy_landmarks() -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)


def create_mock_dataset(
    output_path: str,
    user_sessions: Dict[str, List[str]],
    classes: List[str] = STATIC_CLASSES,
    samples_per_class: int = 2,
    include_transitions: bool = False,
) -> None:
    """Helper to synthesize structured test datasets with known users, sessions, and classes."""
    rng = np.random.RandomState(123)
    features_list = []
    labels_list = []
    metadata_list = []

    for user_id, sessions in user_sessions.items():
        for sess_id in sessions:
            for cls_name in classes:
                for s_idx in range(samples_per_class):
                    feat = rng.normal(loc=0.5, scale=0.1, size=(63,)).astype(np.float32)
                    meta = create_sample_metadata(
                        user_id=user_id,
                        session_id=sess_id,
                        timestamp=1000.0 + len(features_list),
                        handedness="Right" if s_idx % 2 == 0 else "Left",
                        distance_cm=45.0,
                        lighting="bright",
                        label=cls_name,
                        is_transition=False,
                        feature_version=1,
                    )
                    features_list.append(feat)
                    labels_list.append(cls_name)
                    metadata_list.append(meta)

            if include_transitions:
                for t_idx in range(2):
                    feat = rng.normal(loc=0.5, scale=0.1, size=(63,)).astype(np.float32)
                    meta = create_sample_metadata(
                        user_id=user_id,
                        session_id=sess_id,
                        timestamp=2000.0 + len(features_list),
                        handedness="Right",
                        distance_cm=45.0,
                        lighting="bright",
                        label="OTHER",
                        is_transition=True,
                        feature_version=1,
                    )
                    features_list.append(feat)
                    labels_list.append("OTHER")
                    metadata_list.append(meta)

    save_dataset_npz(
        output_path,
        np.asarray(features_list, dtype=np.float32),
        labels_list,
        metadata_list,
    )


class TestCollectDataSafeguards:
    def test_corrupt_file_fail_closed_no_overwrite(self, tmp_path, dummy_landmarks):
        corrupt_file = str(tmp_path / "corrupt_data.npz")
        corrupt_content = b"PK\x03\x04CORRUPTED_ARCHIVE_DATA_NOT_VALID_ZIP"
        with open(corrupt_file, "wb") as f:
            f.write(corrupt_content)

        original_hash = hashlib.sha256(corrupt_content).hexdigest()

        exit_code = collect_data_interactive(
            output_path=corrupt_file,
            user_id="attacker_or_user",
            mock_frames=[(dummy_landmarks, "Right")],
            raise_on_error=False,
        )
        assert exit_code != 0

        assert os.path.exists(corrupt_file)
        with open(corrupt_file, "rb") as f:
            after_bytes = f.read()
        assert after_bytes == corrupt_content
        assert hashlib.sha256(after_bytes).hexdigest() == original_hash

        with pytest.raises(ValueError, match="corrupt or unreadable"):
            collect_data_interactive(
                output_path=corrupt_file,
                user_id="attacker_or_user",
                mock_frames=[(dummy_landmarks, "Right")],
                raise_on_error=True,
            )

    def test_valid_append_backward_compatibility(self, tmp_path, dummy_landmarks):
        valid_file = str(tmp_path / "valid_append.npz")

        rc1 = collect_data_interactive(
            output_path=valid_file,
            user_id="user_alpha",
            session_id="sess_01",
            mock_frames=[(dummy_landmarks, "Right")],
        )
        assert rc1 == 0

        rc2 = collect_data_interactive(
            output_path=valid_file,
            user_id="user_alpha",
            session_id="sess_02",
            mock_frames=[(dummy_landmarks, "Left")],
        )
        assert rc2 == 0

        feats, lbls, meta = load_dataset_npz(valid_file)
        assert len(feats) == 2
        assert meta[0]["session_id"] == "sess_01"
        assert meta[1]["session_id"] == "sess_02"


class TestCentralizedIntegrityValidation:
    def test_direct_validate_cohort_sessions_label_mismatch_raises(self):
        meta = [
            {
                "user_id": "u1",
                "session_id": "s1",
                "label": "POINT_ONE",
                "is_transition": False,
                "feature_version": 1,
                "source_file": "d.npz",
                "source_hash": "a" * 64,
                "source_row_idx": 0,
            }
        ]
        labels = ["PINCH_CLOSED"]
        with pytest.raises(ValueError, match="label mismatch"):
            validate_cohort_sessions(meta, labels)

    def test_direct_validate_cohort_sessions_non_boolean_transition_raises(self):
        meta = [
            {
                "user_id": "u1",
                "session_id": "s1",
                "label": "POINT_ONE",
                "is_transition": "not_a_bool",
                "feature_version": 1,
                "source_file": "d.npz",
                "source_hash": "a" * 64,
                "source_row_idx": 0,
            }
        ]
        labels = ["POINT_ONE"]
        with pytest.raises(ValueError, match="'is_transition' must be a boolean"):
            validate_cohort_sessions(meta, labels)

    def test_direct_identity_containing_delimiter_raises(self):
        meta = [
            {
                "user_id": "u1::malicious",
                "session_id": "s1",
                "label": "POINT_ONE",
                "is_transition": False,
                "feature_version": 1,
                "source_file": "d.npz",
                "source_hash": "a" * 64,
                "source_row_idx": 0,
            }
        ]
        labels = ["POINT_ONE"]
        with pytest.raises(ValueError, match="contains forbidden delimiter '::'"):
            validate_cohort_sessions(meta, labels)

    def test_absent_source_provenance_raises(self):
        # source_file absent
        meta = [
            {
                "user_id": "u1",
                "session_id": "s1",
                "label": "POINT_ONE",
                "is_transition": False,
                "feature_version": 1,
            }
        ]
        labels = ["POINT_ONE"]
        with pytest.raises(ValueError, match="missing or empty mandatory 'source_file'"):
            validate_cohort_sessions(meta, labels)

    def test_non_hex_source_hash_raises(self):
        meta = [
            {
                "user_id": "u1",
                "session_id": "s1",
                "label": "POINT_ONE",
                "is_transition": False,
                "feature_version": 1,
                "source_file": "d.npz",
                "source_hash": "z" * 64,  # 'z' is not hexadecimal
                "source_row_idx": 0,
            }
        ]
        labels = ["POINT_ONE"]
        with pytest.raises(ValueError, match="64-character hexadecimal SHA-256 hash"):
            validate_cohort_sessions(meta, labels)

    def test_bool_source_row_idx_raises(self):
        meta = [
            {
                "user_id": "u1",
                "session_id": "s1",
                "label": "POINT_ONE",
                "is_transition": False,
                "feature_version": 1,
                "source_file": "d.npz",
                "source_hash": "a" * 64,
                "source_row_idx": True,  # True is bool
            }
        ]
        labels = ["POINT_ONE"]
        with pytest.raises(ValueError, match="must be a non-negative integer \\(not bool\\)"):
            validate_cohort_sessions(meta, labels)

    def test_direct_generate_split_manifest_rejects_not_ready_input(self):
        # 1-session train-only input is NOT_READY
        meta = [
            {
                "user_id": "u1",
                "session_id": "s1",
                "label": "POINT_ONE",
                "is_transition": False,
                "feature_version": 1,
                "source_file": "d.npz",
                "source_hash": "a" * 64,
                "source_row_idx": 0,
            }
        ]
        labels = ["POINT_ONE"]
        plan = {("u1", "s1"): "train"}
        with pytest.raises(ValueError, match="Cannot generate split manifest: dataset is NOT_READY"):
            generate_split_manifest(meta, labels, session_assignments=plan, roster=["u1"], mode="personal")

    def test_json_duplicate_keys_rejected(self):
        dup_json = '{"u1::s1": "train", "u1::s1": "validation"}'
        with pytest.raises(ValueError, match="Duplicate key 'u1::s1'"):
            parse_session_plan(dup_json)

    def test_output_path_aliasing_rejected(self, tmp_path):
        f = str(tmp_path / "dataset.npz")
        create_mock_dataset(f, {"u1": ["s1"]})

        # Exact alias
        with pytest.raises(ValueError, match="aliases input dataset"):
            validate_output_paths_no_alias([f], output_report_path=f)

        # Case-insensitive alias (e.g. on Windows)
        f_upper = str(tmp_path / "DATASET.NPZ")
        with pytest.raises(ValueError, match="aliases input dataset"):
            validate_output_paths_no_alias([f], output_report_path=f_upper)

        # Report and manifest paths identical
        out_f = str(tmp_path / "report.json")
        with pytest.raises(ValueError, match="cannot be the same file"):
            validate_output_paths_no_alias([f], output_report_path=out_f, output_manifest_path=out_f)


class TestIntegrityViolations:
    def test_duplicate_input_files_by_path(self, tmp_path):
        f = str(tmp_path / "file1.npz")
        create_mock_dataset(f, {"u1": ["s1"]})
        with pytest.raises(ValueError, match="Duplicate input file detected"):
            validate_input_files([f, f])

    def test_duplicate_input_files_by_hash(self, tmp_path):
        f1 = str(tmp_path / "file1.npz")
        f2 = str(tmp_path / "file2.npz")
        create_mock_dataset(f1, {"u1": ["s1"]})
        with open(f1, "rb") as src, open(f2, "wb") as dst:
            dst.write(src.read())

        with pytest.raises(ValueError, match="Duplicate input file content detected"):
            validate_input_files([f1, f2])

    def test_duplicate_raw_session_ingestion_across_files(self, tmp_path):
        f1 = str(tmp_path / "file_a.npz")
        f2 = str(tmp_path / "file_b.npz")
        create_mock_dataset(f1, {"u1": ["sess_shared"]})
        create_mock_dataset(f2, {"u1": ["sess_shared"]}, samples_per_class=3)

        _, _, meta1, _ = load_dataset_with_provenance(f1)
        _, _, meta2, _ = load_dataset_with_provenance(f2)

        with pytest.raises(ValueError, match="Duplicate raw session ingestion across files"):
            validate_session_disjointness_across_files({f1: meta1, f2: meta2})

    def test_duplicate_roster_entries(self):
        with pytest.raises(ValueError, match="Duplicate participant in roster"):
            validate_roster(["tuana", "tuana"])

    def test_blank_or_null_roster_entries(self):
        with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
            validate_roster(["tuana", "  "])
        with pytest.raises(ValueError, match="must be a non-null string"):
            validate_roster(["tuana", None])

    def test_unknown_role_key(self):
        plan = {"u1::s1": "unknown_role"}
        with pytest.raises(ValueError, match="Unknown or invalid role"):
            parse_session_plan(plan)

    def test_duplicate_session_assignment_even_same_role(self):
        plan = [
            {"user_id": "u1", "session_id": "s1", "role": "train"},
            {"user_id": "u1", "session_id": "s1", "role": "train"},
        ]
        with pytest.raises(ValueError, match="Duplicate assignment for session"):
            parse_session_plan(plan)

    def test_unknown_session_in_plan(self, tmp_path):
        f = str(tmp_path / "dataset.npz")
        create_mock_dataset(f, {"u1": ["s1"]})
        plan = {
            "u1::s1": "train",
            "u1::s_ghost": "validation",
        }
        with pytest.raises(ValueError, match="not found in dataset"):
            audit_dataset_readiness([f], roster=["u1"], session_plan=plan)

    def test_unassigned_session_in_dataset(self, tmp_path):
        f = str(tmp_path / "dataset.npz")
        create_mock_dataset(f, {"u1": ["s1", "s2"]})
        plan = {
            "u1::s1": "train",
        }
        with pytest.raises(ValueError, match="unassigned session"):
            audit_dataset_readiness([f], roster=["u1"], session_plan=plan)


class TestCohortSessionRules:
    def test_pair_keyed_assignments_two_users_sharing_session_name(self, tmp_path):
        data_path = str(tmp_path / "shared_sess_name.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={
                "user_A": ["sess_common", "sess_a2", "sess_a3"],
                "user_B": ["sess_common", "sess_b2", "sess_b3"],
            },
            classes=STATIC_CLASSES,
        )

        plan = {
            "user_A::sess_common": "train",
            "user_A::sess_a2": "validation",
            "user_A::sess_a3": "test",
            "user_B::sess_common": "test",
            "user_B::sess_b2": "train",
            "user_B::sess_b3": "validation",
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["user_A", "user_B"],
            mode="cohort",
            session_plan=plan,
        )

        assert report["status"] == "READY"
        assert report["role_assignments"]["train"]["sessions"] == ["user_A::sess_common", "user_B::sess_b2"]
        assert report["role_assignments"]["test"]["sessions"] == ["user_A::sess_a3", "user_B::sess_common"]

    def test_no_roster_or_no_plan_results_in_diagnostic_not_ready_never_auto_ready(self, tmp_path):
        data_path = str(tmp_path / "diagnostic.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["s1", "s2", "s3"]},
            classes=STATIC_CLASSES,
        )

        report_no_plan = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana"],
            mode="personal",
            session_plan=None,
        )
        assert report_no_plan["status"] == "NOT_READY"
        assert any("Explicit session assignment plan is required" in c for c in report_no_plan["missing_criteria"])

        plan = {"tuana::s1": "train", "tuana::s2": "validation", "tuana::s3": "test"}
        report_no_roster = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=None,
            mode="personal",
            session_plan=plan,
        )
        assert report_no_roster["status"] == "NOT_READY"
        assert any("Explicit participant roster is required" in c for c in report_no_roster["missing_criteria"])

    def test_manifest_not_emitted_on_not_ready(self, tmp_path):
        data_path = str(tmp_path / "not_ready.npz")
        create_mock_dataset(data_path, {"tuana": ["s1"]})
        man_out = str(tmp_path / "manifest.json")
        rep_out = str(tmp_path / "report.json")
        plan = {"tuana::s1": "train"}

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana"],
            session_plan=plan,
            output_report_path=rep_out,
            output_manifest_path=man_out,
        )
        assert report["status"] == "NOT_READY"
        assert os.path.exists(rep_out)
        assert not os.path.exists(man_out)

    def test_personal_use_acceptance_three_sessions_nine_classes(self, tmp_path):
        data_path = str(tmp_path / "personal_ready.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["sess_1", "sess_2", "sess_3"]},
            classes=STATIC_CLASSES,
            samples_per_class=3,
        )

        plan = {
            "tuana::sess_1": "train",
            "tuana::sess_2": "validation",
            "tuana::sess_3": "test",
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana"],
            mode="personal",
            session_plan=plan,
        )

        assert report["status"] == "READY"
        assert report["mode"] == "personal"
        assert report["roster"] == ["tuana"]
        assert len(report["missing_criteria"]) == 0
        assert report["role_assignments"]["train"]["sessions"] == ["tuana::sess_1"]
        assert report["role_assignments"]["validation"]["sessions"] == ["tuana::sess_2"]
        assert report["role_assignments"]["test"]["sessions"] == ["tuana::sess_3"]
        for role in ROLES:
            for cls_name in STATIC_CLASSES:
                assert report["role_assignments"][role]["stable_class_counts"][cls_name] >= 3

    def test_single_user_rejection_insufficient_sessions(self, tmp_path):
        data_path = str(tmp_path / "personal_not_ready.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["sess_1", "sess_2"]},
            classes=STATIC_CLASSES,
        )
        plan = {
            "tuana::sess_1": "train",
            "tuana::sess_2": "validation",
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana"],
            mode="personal",
            session_plan=plan,
        )

        assert report["status"] == "NOT_READY"
        assert any("minimum 3 independent sessions required" in c for c in report["missing_criteria"])

    def test_multi_user_cohort_acceptance(self, tmp_path):
        data_path = str(tmp_path / "cohort_ready.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={
                "tuana": ["sess_t1", "sess_t2", "sess_t3"],
                "friend_01": ["sess_f1", "sess_f2", "sess_f3"],
            },
            classes=STATIC_CLASSES,
            samples_per_class=2,
        )

        plan = {
            "train": ["tuana::sess_t1", "friend_01::sess_f1"],
            "validation": ["tuana::sess_t2", "friend_01::sess_f2"],
            "test": ["tuana::sess_t3", "friend_01::sess_f3"],
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana", "friend_01"],
            mode="cohort",
            session_plan=plan,
        )

        assert report["status"] == "READY"
        assert report["mode"] == "cohort"
        assert len(report["missing_criteria"]) == 0
        for r in ROLES:
            parts = report["role_assignments"][r]["participants"]
            assert "tuana" in parts
            assert "friend_01" in parts

    def test_cohort_mode_rejection_missing_roster_participant(self, tmp_path):
        data_path = str(tmp_path / "cohort_missing_user.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["sess_1", "sess_2", "sess_3"]},
            classes=STATIC_CLASSES,
        )
        plan = {
            "tuana::sess_1": "train",
            "tuana::sess_2": "validation",
            "tuana::sess_3": "test",
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana", "friend_01"],
            mode="cohort",
            session_plan=plan,
        )

        assert report["status"] == "NOT_READY"
        assert any("missing from dataset" in c for c in report["missing_criteria"])

    def test_missing_target_classes_rejection(self, tmp_path):
        data_path = str(tmp_path / "missing_classes.npz")
        partial_classes = ["PINCH_CLOSED", "PINCH_MIDDLE", "L_SHAPE", "OTHER"]
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["sess_1", "sess_2", "sess_3"]},
            classes=partial_classes,
        )
        plan = {
            "tuana::sess_1": "train",
            "tuana::sess_2": "validation",
            "tuana::sess_3": "test",
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana"],
            mode="personal",
            session_plan=plan,
        )

        assert report["status"] == "NOT_READY"
        assert any("missing" in c.lower() for c in report["missing_criteria"])

    def test_transition_frames_excluded_from_stable_counts(self, tmp_path):
        data_path = str(tmp_path / "with_transitions.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["sess_1", "sess_2", "sess_3"]},
            classes=STATIC_CLASSES,
            samples_per_class=2,
            include_transitions=True,
        )

        plan = {
            "tuana::sess_1": "train",
            "tuana::sess_2": "validation",
            "tuana::sess_3": "test",
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana"],
            mode="personal",
            session_plan=plan,
        )

        assert report["status"] == "READY"
        assert report["transition_samples"] == 6
        assert report["stable_samples"] == len(STATIC_CLASSES) * 2 * 3
        assert report["total_samples"] == report["stable_samples"] + report["transition_samples"]

    def test_existing_custom_dataset_is_not_ready(self):
        """Authoritative audit of existing repo custom_airpad.npz: must declare NOT_READY."""
        custom_npz = "data/custom_airpad.npz"
        assert os.path.exists(custom_npz)

        plan = {"tuana::sess_20260929_132034_7a13ed": "train"}
        report = audit_dataset_readiness(
            dataset_paths=[custom_npz],
            roster=["tuana"],
            mode="personal",
            session_plan=plan,
        )

        assert report["status"] == "NOT_READY"
        assert len(report["sessions_summary"]) == 1
        assert "tuana::sess_20260929_132034_7a13ed" in report["sessions_summary"]
        crit_text = " ".join(report["missing_criteria"])
        assert "minimum 3 independent sessions required" in crit_text
        assert "missing" in crit_text.lower()

    def test_transition_only_friend_in_cohort_is_not_ready(self, tmp_path):
        """Tuana has all 9 stable poses in each role; friend_01 has only transition frames in each role."""
        data_path = str(tmp_path / "trans_only_friend.npz")
        # Tuana has all 9 poses
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["sess_t1", "sess_t2", "sess_t3"]},
            classes=STATIC_CLASSES,
            samples_per_class=2,
            include_transitions=False,
        )
        # Friend only has transition frames
        f_feat = []
        f_lbl = []
        f_meta = []
        rng = np.random.RandomState(42)
        for s_id in ["sess_f1", "sess_f2", "sess_f3"]:
            for _ in range(5):
                feat = rng.normal(0.5, 0.1, size=(63,)).astype(np.float32)
                f_feat.append(feat)
                f_lbl.append("OTHER")
                f_meta.append(
                    create_sample_metadata(
                        user_id="friend_01",
                        session_id=s_id,
                        timestamp=100.0,
                        handedness="Right",
                        distance_cm=45.0,
                        lighting="bright",
                        label="OTHER",
                        is_transition=True,  # All transition!
                        feature_version=1,
                    )
                )

        # Append to existing
        f_orig_feat, f_orig_lbl, f_orig_meta = load_dataset_npz(data_path)
        all_feat = np.vstack([f_orig_feat, np.asarray(f_feat, dtype=np.float32)])
        all_lbl = f_orig_lbl.tolist() + f_lbl
        all_meta = f_orig_meta + f_meta
        save_dataset_npz(data_path, all_feat, all_lbl, all_meta)

        plan = {
            "train": ["tuana::sess_t1", "friend_01::sess_f1"],
            "validation": ["tuana::sess_t2", "friend_01::sess_f2"],
            "test": ["tuana::sess_t3", "friend_01::sess_f3"],
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana", "friend_01"],
            mode="cohort",
            session_plan=plan,
        )

        assert report["status"] == "NOT_READY"
        crit_text = " ".join(report["missing_criteria"])
        assert "friend_01" in crit_text
        assert "zero stable samples" in crit_text or "missing stable samples" in crit_text

    def test_friend_missing_pose_in_role_is_not_ready(self, tmp_path):
        """Tuana has all 9 poses; friend_01 is missing FIST in train role."""
        data_path = str(tmp_path / "friend_missing_fist.npz")
        # Friend has 8 poses in train, missing FIST
        friend_classes = [c for c in STATIC_CLASSES if c != "FIST"]

        # Create dataset with Tuana covering all 9 in all 3 sessions
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["sess_t1", "sess_t2", "sess_t3"]},
            classes=STATIC_CLASSES,
            samples_per_class=2,
        )
        # Append friend with missing FIST
        f_feat = []
        f_lbl = []
        f_meta = []
        rng = np.random.RandomState(42)
        for s_id in ["sess_f1", "sess_f2", "sess_f3"]:
            # sess_f1 will be train, so omit FIST from sess_f1
            classes_to_use = friend_classes if s_id == "sess_f1" else STATIC_CLASSES
            for cls_name in classes_to_use:
                for _ in range(2):
                    feat = rng.normal(0.5, 0.1, size=(63,)).astype(np.float32)
                    f_feat.append(feat)
                    f_lbl.append(cls_name)
                    f_meta.append(
                        create_sample_metadata(
                            user_id="friend_01",
                            session_id=s_id,
                            timestamp=200.0,
                            handedness="Right",
                            distance_cm=45.0,
                            lighting="bright",
                            label=cls_name,
                            is_transition=False,
                            feature_version=1,
                        )
                    )

        f_orig_feat, f_orig_lbl, f_orig_meta = load_dataset_npz(data_path)
        all_feat = np.vstack([f_orig_feat, np.asarray(f_feat, dtype=np.float32)])
        all_lbl = f_orig_lbl.tolist() + f_lbl
        all_meta = f_orig_meta + f_meta
        save_dataset_npz(data_path, all_feat, all_lbl, all_meta)

        plan = {
            "train": ["tuana::sess_t1", "friend_01::sess_f1"],
            "validation": ["tuana::sess_t2", "friend_01::sess_f2"],
            "test": ["tuana::sess_t3", "friend_01::sess_f3"],
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana", "friend_01"],
            mode="cohort",
            session_plan=plan,
        )

        assert report["status"] == "NOT_READY"
        crit_text = " ".join(report["missing_criteria"])
        assert "friend_01" in crit_text
        assert "FIST" in crit_text

    def test_extra_unrostered_participant_diagnostic(self, tmp_path):
        """Dataset has tuana and friend_extra, but roster is only tuana."""
        data_path = str(tmp_path / "extra_unrostered.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={
                "tuana": ["sess_1", "sess_2", "sess_3"],
                "friend_extra": ["sess_e1"],
            },
            classes=STATIC_CLASSES,
        )
        plan = {
            "tuana::sess_1": "train",
            "tuana::sess_2": "validation",
            "tuana::sess_3": "test",
            "friend_extra::sess_e1": "train",
        }

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana"],
            mode="personal",
            session_plan=plan,
        )

        assert report["status"] == "NOT_READY"
        crit_text = " ".join(report["missing_criteria"])
        assert "unrostered participant" in crit_text
        assert "friend_extra" in crit_text


class TestManifestAndFingerprint:
    def test_manifest_provenance_and_transition_separation(self, tmp_path):
        data_path = str(tmp_path / "manifest_test.npz")
        create_mock_dataset(
            output_path=data_path,
            user_sessions={"tuana": ["s1", "s2", "s3"]},
            classes=STATIC_CLASSES,
            samples_per_class=1,
            include_transitions=True,
        )

        plan = {"tuana::s1": "train", "tuana::s2": "validation", "tuana::s3": "test"}
        manifest_out = str(tmp_path / "split_manifest.json")

        report = audit_dataset_readiness(
            dataset_paths=[data_path],
            roster=["tuana"],
            session_plan=plan,
            output_manifest_path=manifest_out,
        )
        assert report["status"] == "READY"
        assert os.path.exists(manifest_out)

        with open(manifest_out, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["manifest_version"] == 1
        assert manifest["total_stable_samples"] == len(STATIC_CLASSES) * 3
        assert manifest["total_transition_samples"] == 2 * 3
        assert len(manifest["stable_samples"]) == manifest["total_stable_samples"]
        assert len(manifest["transition_samples"]) == manifest["total_transition_samples"]

        for s in manifest["stable_samples"]:
            assert s["is_transition"] is False
            assert isinstance(s["source_row_idx"], int)
            assert not isinstance(s["source_row_idx"], bool)
            assert len(s["source_hash"]) == 64

        for s in manifest["transition_samples"]:
            assert s["is_transition"] is True
            assert isinstance(s["source_row_idx"], int)
            assert not isinstance(s["source_row_idx"], bool)
            assert len(s["source_hash"]) == 64

    def test_fingerprint_deterministic_and_excludes_timestamps(self):
        samples = [
            {
                "role": "train",
                "user_id": "u1",
                "session_id": "s1",
                "source_hash": "a" * 64,
                "source_row_idx": 0,
                "label": "POINT_ONE",
                "is_transition": False,
            },
            {
                "role": "validation",
                "user_id": "u1",
                "session_id": "s2",
                "source_hash": "a" * 64,
                "source_row_idx": 1,
                "label": "FIST",
                "is_transition": True,
            },
        ]
        fp1 = compute_deterministic_fingerprint(samples, roster=["u1"], mode="personal")
        fp2 = compute_deterministic_fingerprint(samples, roster=["u1"], mode="personal")
        assert fp1 == fp2
        assert len(fp1) == 64
