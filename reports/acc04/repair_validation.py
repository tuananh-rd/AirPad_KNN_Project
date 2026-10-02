"""ACC-04 Bounded Reproducible Validation Repair Script.

Repairs validation artifacts by:
1. Validating exact pinned SHA-256 hashes of immutable inputs before any processing (mandatory, no bypass).
2. Enforcing strict path alias guards and collision checks for BOTH output NPZ and audit JSON:
   - Requiring output_npz_path to explicitly end with literal '.npz' before any writes.
   - Normalizing numpy output paths to prevent numpy.savez suffix appending from bypassing alias checks.
   - Neither target may overwrite existing files.
   - Neither target may alias any input dataset or each other (protecting case variants and hardlinks).
   - All alias/collision checks fail closed BEFORE any write operations occur.
3. Excluding the exact UNION of:
   - All old rows labelled L_SHAPE (359 rows, hotkey 4 selection uncertain)
   - Rows 1477..1499 inclusive (23 rows, fourth burst under TWO_FINGERS_PARALLEL performed as THREE_FINGERS)
   Total excluded: 382 rows. Never relabels guesses.
4. Preserving the remaining 1331 rows in original order, byte/value equivalent, preserving session IDs.
5. Performing comprehensive pre-write validation on clean validation, retake, and train datasets:
   - Feature shape, dtype float32, no NaN/inf.
   - Metadata and label length equality.
   - Exact user_id ("tuana") and session_id matching.
   - Explicit zero transitions check (is_transition strictly False for all rows).
   - Centralized validation through existing validator validate_samples_integrity from tools.cohort_sessions.
6. Emitting new cleaned dataset to data/cohort/curated/acc04_validation_clean.npz.
7. Emitting detailed JSON audit mapping to reports/acc04/validation_cleaning_audit.json.
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

from tools.collect_data import LABEL_NAMES, load_dataset_npz, save_dataset_npz
from tools.cohort_sessions import norm_path, paths_alias_each_other, validate_samples_integrity

# Pinned immutable input hashes (strictly mandatory)
PINNED_HASHES = {
    "validation_raw": {
        "rel_path": "data/cohort/tuana_validation_rgb_20261001_112128_2a3fea.npz",
        "sha256": "9758a42469e9bc3e07fa7ee18993189cdf47938a3fd46262a943553b5a0fe01e",
        "expected_rows": 1713,
        "expected_user_id": "tuana",
        "expected_session_id": "tuana_validation_rgb_20261001_112128_2a3fea",
    },
    "validation_retake": {
        "rel_path": "data/cohort/tuana_validation_retake_20261001_113202_dfcd6b.npz",
        "sha256": "fe26b606921386a0f640dc958e4e7ebf01e43ce0d6cd6f07e693fce517db8948",
        "expected_rows": 415,
        "expected_user_id": "tuana",
        "expected_session_id": "tuana_validation_retake_20261001_113202_dfcd6b",
    },
    "train_rgb": {
        "rel_path": "data/cohort/tuana_train_rgb_20261001_110223_8ff3ed.npz",
        "sha256": "0f9d21b5b1a5ccab0ffc8f555d7c00e14b26fa2299040b9ab587b236e7d1014e",
        "expected_rows": 1692,
        "expected_user_id": "tuana",
        "expected_session_id": "tuana_train_rgb_20261001_110223_8ff3ed",
    },
}

DEFAULT_OUTPUT_NPZ = os.path.join(PROJECT_ROOT, "data", "cohort", "curated", "acc04_validation_clean.npz")
DEFAULT_AUDIT_JSON = os.path.join(PROJECT_ROOT, "reports", "acc04", "validation_cleaning_audit.json")

EXPECTED_CLEAN_COUNTS = {
    "PINCH_CLOSED": 173,
    "PINCH_MIDDLE": 170,
    "POINT_ONE": 193,
    "OPEN_PALM": 185,
    "FIST": 195,
    "TWO_FINGERS_PARALLEL": 202,
    "THREE_FINGERS": 213,
}

EXPECTED_RETAKE_COUNTS = {
    "L_SHAPE": 209,
    "OTHER": 206,
}


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_pinned_hash(filepath: str, expected_sha256: str) -> str:
    """Verify that file exists and matches the pinned SHA-256 hash; fails closed on any discrepancy."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[FAIL-CLOSED] Required input file not found: {filepath}")
    actual_sha256 = compute_file_sha256(filepath)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"[FAIL-CLOSED] Hash mismatch for {filepath}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def validate_paths_and_collision_guards(
    raw_val_path: str,
    retake_val_path: str,
    train_path: str,
    output_npz_path: str,
    audit_json_path: str,
) -> str:
    """Strictly validate path safety and collision guards before any processing or write operations.

    Enforces:
    - output_npz_path must explicitly end with literal '.npz' (rejects extensionless paths).
    - Normalized numpy target filename (accounting for numpy.savez .npz appending) is evaluated against all guards.
    - Target paths must not already exist (neither NPZ nor audit JSON may overwrite existing targets).
    - Target paths must not alias each other.
    - Neither target path may alias ANY input dataset (raw validation, retake, or train).

    Returns:
        The normalized output NPZ path ending with .npz.
    """
    # 1. Require literal .npz extension on output_npz_path to prevent numpy.savez suffix appending
    if not output_npz_path.lower().endswith(".npz"):
        raise ValueError(
            f"[FAIL-CLOSED] Target output NPZ path must explicitly end with '.npz', got '{output_npz_path}'. "
            "Extensionless paths prohibited to prevent unintentional suffix appending."
        )

    # Compute effective filesystem target paths (both explicitly and as numpy.savez would write)
    effective_npz_target = output_npz_path if output_npz_path.endswith(".npz") else f"{output_npz_path}.npz"

    inputs = [
        ("raw_validation", raw_val_path),
        ("retake_validation", retake_val_path),
        ("train_rgb", train_path),
    ]
    targets = [
        ("output_clean_npz", output_npz_path),
        ("effective_npz_target", effective_npz_target),
        ("audit_json", audit_json_path),
    ]

    # 2. Output targets must not already exist
    if os.path.exists(output_npz_path):
        raise FileExistsError(
            f"[FAIL-CLOSED] Target output file already exists: {output_npz_path}. "
            "Refusing to overwrite existing curated validation artifact."
        )
    if os.path.exists(effective_npz_target) and effective_npz_target != output_npz_path:
        raise FileExistsError(
            f"[FAIL-CLOSED] Suffix-appended target output file already exists: {effective_npz_target}. "
            "Refusing to overwrite existing file."
        )
    if os.path.exists(audit_json_path):
        raise FileExistsError(
            f"[FAIL-CLOSED] Target audit file already exists: {audit_json_path}. "
            "Refusing to overwrite existing audit report."
        )

    # 3. Targets must not alias each other
    if paths_alias_each_other(output_npz_path, audit_json_path) or \
       paths_alias_each_other(effective_npz_target, audit_json_path):
        raise ValueError(
            f"[FAIL-CLOSED] Target output NPZ ({output_npz_path}) and audit JSON ({audit_json_path}) "
            "resolve to the same filesystem path."
        )

    # 4. Targets must not alias ANY input path
    for target_name, target_path in targets:
        for input_name, input_path in inputs:
            if paths_alias_each_other(target_path, input_path):
                raise ValueError(
                    f"[FAIL-CLOSED] Target path {target_name} ('{target_path}') aliases "
                    f"input path {input_name} ('{input_path}'). Writes strictly prohibited."
                )

    return effective_npz_target


def validate_dataset_samples_integrity(
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
    """Comprehensive in-memory validation of dataset samples before any disk write.

    Enforces:
    - Length equality across features, labels, and metadata.
    - Feature shape (expected_rows, 63), dtype float, no NaN or infinite values.
    - Exact user_id and session_id matching for all samples.
    - Exact label agreement between label array and metadata.
    - All labels belonging to canonical LABEL_NAMES.
    - Zero transitions: is_transition must be boolean False for all rows.
    - Centralized validation through existing validator validate_samples_integrity from tools.cohort_sessions.
    """
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

    # Schema checks
    if features.ndim != 2 or features.shape[1] != 63:
        raise ValueError(
            f"[FAIL-CLOSED] {dataset_name} features must have shape ({expected_rows}, 63), got {features.shape}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"[FAIL-CLOSED] {dataset_name} features must have float dtype, got {features.dtype}")
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError(f"[FAIL-CLOSED] {dataset_name} features contain NaN or infinite values")

    # Sample-level checks
    for idx, meta in enumerate(metadata):
        if not isinstance(meta, dict):
            raise ValueError(f"[FAIL-CLOSED] {dataset_name} row {idx}: metadata is not a dictionary")

        # Exact user_id and session_id validation
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

        # Label validation
        lbl_arr = str(labels[idx])
        meta_lbl = meta.get("label")
        if lbl_arr != meta_lbl:
            raise ValueError(
                f"[FAIL-CLOSED] {dataset_name} row {idx}: label mismatch between array ('{lbl_arr}') and metadata ('{meta_lbl}')"
            )
        if lbl_arr not in LABEL_NAMES:
            raise ValueError(f"[FAIL-CLOSED] {dataset_name} row {idx}: unrecognized label '{lbl_arr}'")

        # Zero transition frames allowed
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


def execute_validation_repair(
    raw_val_path: Optional[str] = None,
    retake_val_path: Optional[str] = None,
    train_path: Optional[str] = None,
    output_npz_path: Optional[str] = None,
    audit_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the bounded ACC-04 validation repair workflow with strict fail-closed guards."""
    raw_val_path = raw_val_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["validation_raw"]["rel_path"])
    retake_val_path = retake_val_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["validation_retake"]["rel_path"])
    train_path = train_path or os.path.join(PROJECT_ROOT, PINNED_HASHES["train_rgb"]["rel_path"])
    output_npz_path = output_npz_path or DEFAULT_OUTPUT_NPZ
    audit_json_path = audit_json_path or DEFAULT_AUDIT_JSON

    # 1. Path safety and collision checks (strictly before any reading or writing)
    effective_npz_path = validate_paths_and_collision_guards(
        raw_val_path=raw_val_path,
        retake_val_path=retake_val_path,
        train_path=train_path,
        output_npz_path=output_npz_path,
        audit_json_path=audit_json_path,
    )

    # 2. Verify all immutable inputs against pinned hashes (mandatory, no bypass)
    raw_val_hash = verify_pinned_hash(raw_val_path, PINNED_HASHES["validation_raw"]["sha256"])
    retake_val_hash = verify_pinned_hash(retake_val_path, PINNED_HASHES["validation_retake"]["sha256"])
    train_hash = verify_pinned_hash(train_path, PINNED_HASHES["train_rgb"]["sha256"])

    # 3. Load and pre-validate train dataset
    train_feats, train_lbls, train_meta = load_dataset_npz(train_path)
    validate_dataset_samples_integrity(
        features=train_feats,
        labels=train_lbls,
        metadata=train_meta,
        expected_rows=PINNED_HASHES["train_rgb"]["expected_rows"],
        expected_user_id=PINNED_HASHES["train_rgb"]["expected_user_id"],
        expected_session_id=PINNED_HASHES["train_rgb"]["expected_session_id"],
        source_path=train_path,
        source_hash=train_hash,
        dataset_name="train_rgb",
    )

    # 4. Load and pre-validate retake validation dataset
    retake_feats, retake_lbls, retake_meta = load_dataset_npz(retake_val_path)
    validate_dataset_samples_integrity(
        features=retake_feats,
        labels=retake_lbls,
        metadata=retake_meta,
        expected_rows=PINNED_HASHES["validation_retake"]["expected_rows"],
        expected_user_id=PINNED_HASHES["validation_retake"]["expected_user_id"],
        expected_session_id=PINNED_HASHES["validation_retake"]["expected_session_id"],
        source_path=retake_val_path,
        source_hash=retake_val_hash,
        dataset_name="validation_retake",
    )
    retake_counts = dict(Counter(retake_lbls))
    for label_name, expected_count in EXPECTED_RETAKE_COUNTS.items():
        actual_count = retake_counts.get(label_name, 0)
        if actual_count != expected_count:
            raise ValueError(
                f"[FAIL-CLOSED] Retake label count mismatch for {label_name}: "
                f"expected {expected_count}, got {actual_count}"
            )

    # 5. Load raw validation dataset
    raw_feats, raw_lbls, raw_meta = load_dataset_npz(raw_val_path)
    if len(raw_feats) != PINNED_HASHES["validation_raw"]["expected_rows"] or \
       len(raw_lbls) != PINNED_HASHES["validation_raw"]["expected_rows"] or \
       len(raw_meta) != PINNED_HASHES["validation_raw"]["expected_rows"]:
        raise ValueError(
            f"[FAIL-CLOSED] Raw validation dataset length mismatch: "
            f"expected {PINNED_HASHES['validation_raw']['expected_rows']}, "
            f"got {len(raw_feats)} features, {len(raw_lbls)} labels, {len(raw_meta)} metadata"
        )

    # 6. Identify exact exclusion sets
    # Set A: All rows labelled L_SHAPE (user uncertain about hotkey 4)
    l_shape_indices = [i for i, lbl in enumerate(raw_lbls) if lbl == "L_SHAPE"]
    if len(l_shape_indices) != 359:
        raise ValueError(
            f"[FAIL-CLOSED] Expected exactly 359 rows with label L_SHAPE in raw validation, "
            f"found {len(l_shape_indices)}"
        )
    if min(l_shape_indices) != 343 or max(l_shape_indices) != 701:
        raise ValueError(
            f"[FAIL-CLOSED] Unexpected L_SHAPE index range: "
            f"expected 343..701, got {min(l_shape_indices)}..{max(l_shape_indices)}"
        )

    # Set B: Rows 1477..1499 inclusive (fourth burst under TWO_FINGERS_PARALLEL performed as THREE_FINGERS)
    burst_indices = list(range(1477, 1500))
    if len(burst_indices) != 23:
        raise ValueError(f"[FAIL-CLOSED] Expected 23 burst rows, got {len(burst_indices)}")

    for b_idx in burst_indices:
        if raw_lbls[b_idx] != "TWO_FINGERS_PARALLEL":
            raise ValueError(
                f"[FAIL-CLOSED] Burst row {b_idx} expected label TWO_FINGERS_PARALLEL, "
                f"got {raw_lbls[b_idx]}"
            )

    burst_start_time = raw_meta[1477]["timestamp"]
    burst_end_time = raw_meta[1499]["timestamp"]
    if abs(burst_start_time - 1790828834.976209) > 1e-4 or abs(burst_end_time - 1790828836.193892) > 1e-4:
        raise ValueError(
            f"[FAIL-CLOSED] Burst timestamp range mismatch: "
            f"expected [1790828834.976209, 1790828836.193892], "
            f"got [{burst_start_time}, {burst_end_time}]"
        )

    # Check temporal gaps bounding burst
    pre_gap = burst_start_time - raw_meta[1476]["timestamp"]
    post_gap = raw_meta[1500]["timestamp"] - burst_end_time
    if pre_gap < 0.5 or post_gap < 1.0:
        raise ValueError(
            f"[FAIL-CLOSED] Temporal gaps bounding burst invalid: "
            f"pre_gap={pre_gap:.2f}s (<0.5s), post_gap={post_gap:.2f}s (<1.0s)"
        )

    # Disjointness check between exclusion sets
    overlap = set(l_shape_indices).intersection(set(burst_indices))
    if overlap:
        raise ValueError(f"[FAIL-CLOSED] Overlap detected between L_SHAPE and burst exclusion sets: {overlap}")

    all_excluded_indices = sorted(set(l_shape_indices).union(set(burst_indices)))
    if len(all_excluded_indices) != 382:
        raise ValueError(
            f"[FAIL-CLOSED] Total excluded rows expected 382 (359 + 23), got {len(all_excluded_indices)}"
        )

    # 7. Extract retained rows in original order
    retained_indices = [i for i in range(len(raw_lbls)) if i not in set(all_excluded_indices)]
    if len(retained_indices) != 1331:
        raise ValueError(
            f"[FAIL-CLOSED] Retained row count expected 1331 (1713 - 382), got {len(retained_indices)}"
        )

    clean_feats = raw_feats[retained_indices].copy()
    clean_lbls = raw_lbls[retained_indices].copy()
    clean_meta = [copy.deepcopy(raw_meta[i]) for i in retained_indices]

    # Verify label distribution of cleaned dataset matches exact expected counts
    clean_counts = dict(Counter(clean_lbls))
    for label_name, expected_count in EXPECTED_CLEAN_COUNTS.items():
        actual_count = clean_counts.get(label_name, 0)
        if actual_count != expected_count:
            raise ValueError(
                f"[FAIL-CLOSED] Label count mismatch for {label_name}: "
                f"expected {expected_count}, got {actual_count}"
            )

    # Ensure zero L_SHAPE or OTHER present in clean_lbls
    if "L_SHAPE" in clean_counts or "OTHER" in clean_counts:
        raise ValueError(
            f"[FAIL-CLOSED] Cleaned dataset must not contain L_SHAPE or OTHER, "
            f"got counts: L_SHAPE={clean_counts.get('L_SHAPE', 0)}, OTHER={clean_counts.get('OTHER', 0)}"
        )

    # Attach original_row_idx to clean_meta copy for provenance validation
    for out_i, orig_i in enumerate(retained_indices):
        clean_meta[out_i]["original_row_idx"] = orig_i

    # 8. Comprehensive pre-write validation of cleaned dataset
    validate_dataset_samples_integrity(
        features=clean_feats,
        labels=clean_lbls,
        metadata=clean_meta,
        expected_rows=1331,
        expected_user_id=PINNED_HASHES["validation_raw"]["expected_user_id"],
        expected_session_id=PINNED_HASHES["validation_raw"]["expected_session_id"],
        source_path=raw_val_path,
        source_hash=raw_val_hash,
        dataset_name="clean_validation",
    )

    # Clean up temporary field before saving
    for m in clean_meta:
        m.pop("original_row_idx", None)

    combined_val_total = len(clean_lbls) + len(retake_lbls)
    if combined_val_total != 1746:
        raise ValueError(
            f"[FAIL-CLOSED] Combined validation row total expected 1746 (1331 + 415), "
            f"got {combined_val_total}"
        )

    # 9. Build detailed JSON audit mapping in memory before writing anything
    retained_audit = []
    for out_idx, orig_idx in enumerate(retained_indices):
        m = raw_meta[orig_idx]
        retained_audit.append({
            "output_row_idx": out_idx,
            "original_row_idx": orig_idx,
            "original_source_file": os.path.relpath(raw_val_path, PROJECT_ROOT).replace("\\", "/"),
            "original_source_hash": raw_val_hash,
            "user_id": m.get("user_id"),
            "session_id": m.get("session_id"),
            "label": str(raw_lbls[orig_idx]),
            "timestamp": m.get("timestamp"),
            "handedness": m.get("handedness"),
            "is_transition": m.get("is_transition", False),
        })

    excluded_audit = []
    for orig_idx in all_excluded_indices:
        m = raw_meta[orig_idx]
        lbl = str(raw_lbls[orig_idx])
        if orig_idx in l_shape_indices:
            reason = "Uncertain L_SHAPE label: user cannot recall selecting hotkey 4 during session"
            category = "UNCERTAIN_LABEL_L_SHAPE"
        else:
            reason = (
                "Mislabeled burst: fourth burst under TWO_FINGERS_PARALLEL reported as THREE_FINGERS "
                f"(timestamps {burst_start_time:.6f}..{burst_end_time:.6f})"
            )
            category = "MISLABELED_BURST_THREE_FINGERS"

        excluded_audit.append({
            "original_row_idx": orig_idx,
            "original_source_file": os.path.relpath(raw_val_path, PROJECT_ROOT).replace("\\", "/"),
            "original_source_hash": raw_val_hash,
            "label": lbl,
            "timestamp": m.get("timestamp"),
            "category": category,
            "reason": reason,
        })

    retake_provenance = []
    for r_idx, m in enumerate(retake_meta):
        retake_provenance.append({
            "retake_row_idx": r_idx,
            "source_file": os.path.relpath(retake_val_path, PROJECT_ROOT).replace("\\", "/"),
            "source_hash": retake_val_hash,
            "user_id": m.get("user_id"),
            "session_id": m.get("session_id"),
            "label": str(retake_lbls[r_idx]),
            "timestamp": m.get("timestamp"),
            "handedness": m.get("handedness"),
            "is_transition": m.get("is_transition", False),
        })

    combined_counts = {k: clean_counts.get(k, 0) + retake_counts.get(k, 0) for k in [
        "PINCH_CLOSED", "PINCH_MIDDLE", "L_SHAPE", "OTHER", "POINT_ONE",
        "OPEN_PALM", "FIST", "TWO_FINGERS_PARALLEL", "THREE_FINGERS"
    ]}

    # 10. Write clean NPZ dataset to disk
    out_dir = os.path.dirname(effective_npz_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_dataset_npz(effective_npz_path, clean_feats, clean_lbls, clean_meta)
    clean_npz_hash = compute_file_sha256(effective_npz_path)

    # 11. Write JSON audit file to disk
    audit_payload = {
        "task_id": "ACC-04",
        "description": "ACC-04 Validation Artifact Repair and Provenance Audit",
        "inputs": {
            "validation_raw": {
                "path": os.path.relpath(raw_val_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": raw_val_hash,
                "row_count": len(raw_lbls),
            },
            "validation_retake": {
                "path": os.path.relpath(retake_val_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": retake_val_hash,
                "row_count": len(retake_lbls),
            },
            "train_rgb": {
                "path": os.path.relpath(train_path, PROJECT_ROOT).replace("\\", "/"),
                "sha256": train_hash,
                "row_count": PINNED_HASHES["train_rgb"]["expected_rows"],
            },
        },
        "output_clean_validation": {
            "path": os.path.relpath(effective_npz_path, PROJECT_ROOT).replace("\\", "/"),
            "sha256": clean_npz_hash,
            "row_count": len(clean_lbls),
            "class_distribution": clean_counts,
        },
        "validation_retake_summary": {
            "row_count": len(retake_lbls),
            "class_distribution": retake_counts,
        },
        "combined_validation_summary": {
            "total_rows": combined_val_total,
            "clean_rows": len(clean_lbls),
            "retake_rows": len(retake_lbls),
            "classes_covered_count": len([c for c, cnt in combined_counts.items() if cnt > 0]),
            "class_distribution": combined_counts,
        },
        "exclusion_summary": {
            "total_excluded": len(all_excluded_indices),
            "l_shape_uncertain_count": len(l_shape_indices),
            "burst_mislabeled_count": len(burst_indices),
            "reasons": {
                "UNCERTAIN_LABEL_L_SHAPE": "All 359 original L_SHAPE rows excluded (hotkey 4 selection uncertain)",
                "MISLABELED_BURST_THREE_FINGERS": "23 rows (1477..1499) under TWO_FINGERS_PARALLEL excluded (performed as THREE_FINGERS)",
            },
        },
        "retained_samples_mapping": retained_audit,
        "excluded_samples": excluded_audit,
        "retake_provenance": retake_provenance,
    }

    audit_dir = os.path.dirname(audit_json_path)
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    return audit_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair validation artifacts for ACC-04.")
    parser.add_argument("--raw-validation", default=None, help="Path to raw validation dataset NPZ")
    parser.add_argument("--retake-validation", default=None, help="Path to validation retake NPZ")
    parser.add_argument("--train-rgb", default=None, help="Path to train RGB dataset NPZ")
    parser.add_argument("--output-clean", default=None, help="Path to output cleaned validation NPZ")
    parser.add_argument("--audit-json", default=None, help="Path to output JSON audit file")
    args = parser.parse_args()

    try:
        print("[INFO] Executing ACC-04 validation repair...")
        result = execute_validation_repair(
            raw_val_path=args.raw_validation,
            retake_val_path=args.retake_validation,
            train_path=args.train_rgb,
            output_npz_path=args.output_clean,
            audit_json_path=args.audit_json,
        )
        print(f"[SUCCESS] Cleaned NPZ created: {result['output_clean_validation']['path']}")
        print(f"          Cleaned SHA-256: {result['output_clean_validation']['sha256']}")
        print(f"          Retained rows: {result['output_clean_validation']['row_count']}")
        print(f"          Excluded rows: {result['exclusion_summary']['total_excluded']}")
        print(f"          Combined validation rows: {result['combined_validation_summary']['total_rows']}")
        return 0
    except Exception as e:
        print(f"[ERROR] ACC-04 validation repair failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
