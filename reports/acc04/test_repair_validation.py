"""ACC-04 Negative-case and Positive-case verification tests.

Proves that repair_validation.py fails closed on:
1. Wrong pinned SHA-256 hash (mandatory pins, no bypass).
2. Extensionless output_npz_path (proves extensionless output cannot change existing sibling .npz and cannot create audit).
3. Audit JSON aliasing any input dataset (e.g. --audit-json pointing at raw validation NPZ).
4. Existing audit collision (audit JSON already exists on disk).
5. Existing output NPZ collision (clean NPZ already exists on disk).
6. Output NPZ aliasing input dataset.
7. Output NPZ and audit JSON aliasing each other.
8. Missing input file.
9. Schema / transition violation (e.g. is_transition=True).
10. Identity mismatch (unexpected user_id or session_id).
And proves that valid execution produces exact 1331 clean rows, 382 excluded rows,
415 retake rows, and all 9 target classes in combined validation.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reports.acc04.repair_validation import (
    EXPECTED_CLEAN_COUNTS,
    EXPECTED_RETAKE_COUNTS,
    PINNED_HASHES,
    compute_file_sha256,
    execute_validation_repair,
    validate_dataset_samples_integrity,
    validate_paths_and_collision_guards,
    verify_pinned_hash,
)
from tools.collect_data import load_dataset_npz, save_dataset_npz


def test_fails_closed_on_wrong_pinned_hash(tmp_path):
    """Must fail closed with ValueError when SHA-256 does not match pinned hash."""
    fake_file = tmp_path / "fake.npz"
    fake_file.write_bytes(b"corrupted contents")

    with pytest.raises(ValueError, match=r"\[FAIL-CLOSED\] Hash mismatch"):
        verify_pinned_hash(str(fake_file), "0000000000000000000000000000000000000000000000000000000000000000")


def test_fails_closed_when_input_missing():
    """Must fail closed with FileNotFoundError when input file is missing."""
    with pytest.raises(FileNotFoundError, match=r"\[FAIL-CLOSED\] Required input file not found"):
        verify_pinned_hash("data/nonexistent_file_xyz.npz", "0000000000000000000000000000000000000000000000000000000000000000")


def test_fails_closed_when_output_npz_lacks_extension_even_with_existing_sibling(tmp_path):
    """Proves extensionless output path cannot modify an existing sibling .npz and cannot create audit JSON."""
    sibling_npz = tmp_path / "target_sibling.npz"
    sentinel_bytes = b"SENTINEL_IMMUTABLE_SIBLING_DATA_12345"
    sibling_npz.write_bytes(sentinel_bytes)
    sibling_hash_before = compute_file_sha256(str(sibling_npz))

    extensionless_output = str(tmp_path / "target_sibling")
    audit_out = str(tmp_path / "test_audit.json")

    with pytest.raises(ValueError, match=r"must explicitly end with '\.npz'"):
        execute_validation_repair(
            output_npz_path=extensionless_output,
            audit_json_path=audit_out,
        )

    # Prove existing sibling .npz was not modified or overwritten by numpy.savez suffix appending
    assert sibling_npz.read_bytes() == sentinel_bytes
    assert compute_file_sha256(str(sibling_npz)) == sibling_hash_before

    # Prove NO audit JSON was created
    assert not os.path.exists(audit_out)


def test_fails_closed_when_audit_json_aliases_input(tmp_path):
    """Must fail closed when --audit-json points at raw input dataset before ANY write operations."""
    raw_path = os.path.join(PROJECT_ROOT, PINNED_HASHES["validation_raw"]["rel_path"])
    raw_hash_before = compute_file_sha256(raw_path)

    out_clean = str(tmp_path / "clean_output.npz")

    # Pass raw validation NPZ as audit JSON path
    with pytest.raises((ValueError, FileExistsError), match=r"\[FAIL-CLOSED\]"):
        execute_validation_repair(
            output_npz_path=out_clean,
            audit_json_path=raw_path,
        )

    # Prove raw input file was completely untouched
    raw_hash_after = compute_file_sha256(raw_path)
    assert raw_hash_after == raw_hash_before

    # Prove NO partial clean NPZ output was created
    assert not os.path.exists(out_clean)


def test_fails_closed_when_audit_json_already_exists(tmp_path):
    """Must fail closed when audit JSON already exists; must NOT create partial clean NPZ."""
    existing_audit = tmp_path / "existing_audit.json"
    existing_audit.write_text('{"sentinel": "untouched"}', encoding="utf-8")

    out_clean = str(tmp_path / "clean_output.npz")

    with pytest.raises(FileExistsError, match=r"\[FAIL-CLOSED\] Target audit file already exists"):
        execute_validation_repair(
            output_npz_path=out_clean,
            audit_json_path=str(existing_audit),
        )

    # Prove existing audit was untouched and NO partial clean NPZ was created
    assert json.loads(existing_audit.read_text(encoding="utf-8")) == {"sentinel": "untouched"}
    assert not os.path.exists(out_clean)


def test_fails_closed_when_output_npz_already_exists(tmp_path):
    """Must fail closed when output NPZ already exists; must NOT create audit JSON."""
    existing_npz = tmp_path / "existing_clean.npz"
    existing_npz.write_bytes(b"sentinel_data")

    audit_out = str(tmp_path / "new_audit.json")

    with pytest.raises(FileExistsError, match=r"\[FAIL-CLOSED\] Target output file already exists"):
        execute_validation_repair(
            output_npz_path=str(existing_npz),
            audit_json_path=audit_out,
        )

    # Prove existing NPZ was untouched and NO audit JSON was created
    assert existing_npz.read_bytes() == b"sentinel_data"
    assert not os.path.exists(audit_out)


def test_fails_closed_when_output_npz_aliases_input(tmp_path):
    """Must fail closed when output NPZ points to an input dataset."""
    raw_path = os.path.join(PROJECT_ROOT, PINNED_HASHES["validation_raw"]["rel_path"])
    raw_hash_before = compute_file_sha256(raw_path)
    audit_out = str(tmp_path / "new_audit.json")

    with pytest.raises((FileExistsError, ValueError), match=r"\[FAIL-CLOSED\]"):
        execute_validation_repair(
            output_npz_path=raw_path,
            audit_json_path=audit_out,
        )

    assert compute_file_sha256(raw_path) == raw_hash_before
    assert not os.path.exists(audit_out)


def test_fails_closed_when_output_and_audit_alias_each_other(tmp_path):
    """Must fail closed when output NPZ and audit JSON resolve to the same path."""
    same_path = str(tmp_path / "same_target_file.npz")

    with pytest.raises(ValueError, match=r"resolve to the same filesystem path"):
        validate_paths_and_collision_guards(
            raw_val_path="dummy_raw.npz",
            retake_val_path="dummy_retake.npz",
            train_path="dummy_train.npz",
            output_npz_path=same_path,
            audit_json_path=same_path,
        )


def test_fails_closed_on_transition_frame():
    """Pre-write validator must reject any sample with is_transition=True."""
    raw_path = os.path.join(PROJECT_ROOT, PINNED_HASHES["validation_raw"]["rel_path"])
    feats, lbls, meta = load_dataset_npz(raw_path)

    sample_feats = feats[:10].copy()
    sample_lbls = lbls[:10].copy()
    sample_meta = [copy.deepcopy(m) for m in meta[:10]]
    sample_meta[0]["is_transition"] = True

    with pytest.raises(ValueError, match=r"is_transition must be strictly False"):
        validate_dataset_samples_integrity(
            features=sample_feats,
            labels=sample_lbls,
            metadata=sample_meta,
            expected_rows=10,
            expected_user_id="tuana",
            expected_session_id=sample_meta[0]["session_id"],
            source_path=raw_path,
            source_hash=PINNED_HASHES["validation_raw"]["sha256"],
            dataset_name="test_dataset",
        )


def test_fails_closed_on_unexpected_identity():
    """Pre-write validator must reject any sample with unexpected user_id or session_id."""
    raw_path = os.path.join(PROJECT_ROOT, PINNED_HASHES["validation_raw"]["rel_path"])
    feats, lbls, meta = load_dataset_npz(raw_path)

    sample_feats = feats[:10].copy()
    sample_lbls = lbls[:10].copy()
    sample_meta = [copy.deepcopy(m) for m in meta[:10]]
    sample_meta[0]["user_id"] = "impostor"

    with pytest.raises(ValueError, match=r"unexpected user_id 'impostor'"):
        validate_dataset_samples_integrity(
            features=sample_feats,
            labels=sample_lbls,
            metadata=sample_meta,
            expected_rows=10,
            expected_user_id="tuana",
            expected_session_id=sample_meta[0]["session_id"],
            source_path=raw_path,
            source_hash=PINNED_HASHES["validation_raw"]["sha256"],
            dataset_name="test_dataset",
        )


def test_positive_repair_in_temp_dir(tmp_path):
    """Proves end-to-end repair execution on real inputs in a temporary destination."""
    out_file = str(tmp_path / "clean_validation.npz")
    audit_file = str(tmp_path / "audit.json")

    result = execute_validation_repair(
        output_npz_path=out_file,
        audit_json_path=audit_file,
    )

    # 1. Output files exist
    assert os.path.exists(out_file)
    assert os.path.exists(audit_file)

    # 2. Row count and class distribution
    clean_feats, clean_lbls, clean_meta = load_dataset_npz(out_file)
    assert len(clean_feats) == 1331
    assert len(clean_lbls) == 1331
    assert len(clean_meta) == 1331

    counts = dict(Counter(clean_lbls))
    assert counts == EXPECTED_CLEAN_COUNTS
    assert "L_SHAPE" not in counts
    assert "OTHER" not in counts

    # 3. Retained session ID preserved
    for m in clean_meta:
        assert m["session_id"] == "tuana_validation_rgb_20261001_112128_2a3fea"
        assert m["user_id"] == "tuana"
        assert m["is_transition"] is False

    # 4. JSON audit structure
    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["output_clean_validation"]["row_count"] == 1331
    assert audit["exclusion_summary"]["total_excluded"] == 382
    assert audit["exclusion_summary"]["l_shape_uncertain_count"] == 359
    assert audit["exclusion_summary"]["burst_mislabeled_count"] == 23
    assert len(audit["retained_samples_mapping"]) == 1331
    assert len(audit["excluded_samples"]) == 382
    assert len(audit["retake_provenance"]) == 415

    # 5. Combined validation totals
    assert audit["combined_validation_summary"]["total_rows"] == 1746
    assert audit["combined_validation_summary"]["classes_covered_count"] == 9


if __name__ == "__main__":
    pytest.main(["-v", __file__])
