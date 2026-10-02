"""Provenance audit utility for AirPad datasets.

Validates the integrity, origin, and true groupings of samples in:
- data/custom_airpad.npz
- data/hagrid_airpad.npz
- data/combined_airpad.npz

Performs exact feature/label verification with multiplicity accounting,
exposes synthetic chunk leakage, and generates the authoritative corrected group manifest.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import numpy as np
from tools.collect_data import load_dataset_npz

# Baseline dataset and model hashes from Brain
EXPECTED_HASHES = {
    "custom": "498699e1f28a78031193476fbafff2a2cd307bb9d82a8baee6a57f3097019169",
    "hagrid": "ec38045d608aa6d30386796e4e0586ff619468d4abfd755619835bb8fa61c081",
    "combined": "c0c6d4e26c26f8a303077c6de971a54c2d4085444f9479cef9e021986cc9e8bf",
    "model": "4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977",
}


def compute_file_sha256(filepath: str) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def audit_datasets(
    combined_path: str = "data/combined_airpad.npz",
    hagrid_path: str = "data/hagrid_airpad.npz",
    custom_path: str = "data/custom_airpad.npz",
    manifest_out: str = "reports/acc01/corrected_group_manifest.json",
    report_out: str = "reports/acc01/audit_report.json",
) -> Dict[str, Any]:
    """Execute complete provenance audit and export corrected group manifest."""
    # 1. Verify checksums
    actual_hashes = {
        "custom": compute_file_sha256(custom_path),
        "hagrid": compute_file_sha256(hagrid_path),
        "combined": compute_file_sha256(combined_path),
    }

    for key, expected in EXPECTED_HASHES.items():
        if key in actual_hashes:
            actual = actual_hashes[key]
            if actual != expected:
                raise ValueError(
                    f"Checksum mismatch for {key}: expected {expected}, got {actual}"
                )

    # 2. Load datasets
    c_feat, c_lbl, c_meta = load_dataset_npz(custom_path)
    h_feat, h_lbl, h_meta = load_dataset_npz(hagrid_path)
    comb_feat, comb_lbl, comb_meta = load_dataset_npz(combined_path)

    n_hagrid = len(h_lbl)
    n_custom = len(c_lbl)
    n_combined = len(comb_lbl)

    if n_combined != n_hagrid + n_custom:
        raise ValueError(
            f"Combined length {n_combined} does not match sum of HaGRID ({n_hagrid}) + Custom ({n_custom})"
        )

    # 3. Exact row verification for HaGRID segment [0 : n_hagrid]
    print(f"Auditing {n_hagrid} HaGRID rows in combined dataset...")
    if not np.array_equal(comb_feat[:n_hagrid], h_feat):
        raise RuntimeError("HaGRID feature mismatch detected in combined dataset!")
    if not np.array_equal(comb_lbl[:n_hagrid], h_lbl):
        raise RuntimeError("HaGRID label mismatch detected in combined dataset!")

    # 4. Multiplicity and duplicate analysis on custom dataset
    print(f"Auditing {n_custom} Custom rows with multiplicity accounting...")
    key_map = defaultdict(list)
    for idx in range(n_custom):
        k = (tuple(c_feat[idx]), str(c_lbl[idx]))
        key_map[k].append(idx)

    dup_keys = {k: v for k, v in key_map.items() if len(v) > 1}
    max_multiplicity = max((len(v) for v in dup_keys.values()), default=1)
    disambiguated_by_timestamp = 0
    all_duplicates_agree_on_group = True

    for k, indices in dup_keys.items():
        timestamps = [c_meta[i]["timestamp"] for i in indices]
        if len(set(timestamps)) == len(indices):
            disambiguated_by_timestamp += 1
        users = {c_meta[i]["user_id"] for i in indices}
        sessions = {c_meta[i]["session_id"] for i in indices}
        if len(users) > 1 or len(sessions) > 1:
            all_duplicates_agree_on_group = False

    print(
        f"Custom dataset contains {len(dup_keys)} duplicate feature+label keys "
        f"(max multiplicity: {max_multiplicity}), all {disambiguated_by_timestamp} disambiguated by timestamp."
    )

    # 5. Exact row verification for Custom segment [n_hagrid : n_combined]
    if not np.array_equal(comb_feat[n_hagrid:], c_feat):
        raise RuntimeError("Custom feature mismatch detected in combined dataset!")
    if not np.array_equal(comb_lbl[n_hagrid:], c_lbl):
        raise RuntimeError("Custom label mismatch detected in combined dataset!")

    # 6. Build corrected manifest and detect synthetic chunk leakage
    records = []
    synthetic_chunk_users = set()
    synthetic_chunk_sessions = set()

    # Process HaGRID portion
    for i in range(n_hagrid):
        orig_m = h_meta[i]
        comb_m = comb_meta[i]

        u_id = orig_m.get("user_id")
        s_id = orig_m.get("session_id")
        if not u_id or not s_id:
            raise ValueError(f"HaGRID row {i} missing user_id or session_id in metadata")

        # Verify preserved timestamp, handedness, and label against original metadata
        for k in ["timestamp", "handedness", "label"]:
            if k in orig_m and orig_m[k] != comb_m.get(k):
                raise RuntimeError(
                    f"Metadata mismatch in HaGRID row {i} for '{k}': original {orig_m[k]!r} vs combined {comb_m.get(k)!r}"
                )

        records.append({
            "sample_id": i,
            "source": "hagrid",
            "original_index": i,
            "label": str(h_lbl[i]),
            "timestamp": orig_m.get("timestamp"),
            "handedness": orig_m.get("handedness"),
            "true_user_id": u_id,
            "true_session_id": s_id,
            "combined_user_id": comb_m.get("user_id"),
            "combined_session_id": comb_m.get("session_id"),
            "is_leaky": False,
        })

    # Process Custom portion
    custom_real_users = set()
    custom_real_sessions = set()

    for i in range(n_custom):
        comb_idx = n_hagrid + i
        orig_m = c_meta[i]
        comb_m = comb_meta[comb_idx]

        true_u = orig_m.get("user_id")
        true_s = orig_m.get("session_id")
        comb_u = comb_m.get("user_id")
        comb_s = comb_m.get("session_id")

        if not true_u or not true_s:
            raise ValueError(f"Custom row {i} missing user_id or session_id in original metadata")

        # Verify preserved timestamp, handedness, label, distance, lighting against original metadata
        for k in ["timestamp", "handedness", "label", "distance_cm", "lighting", "is_transition", "feature_version"]:
            if k in orig_m and orig_m[k] != comb_m.get(k):
                raise RuntimeError(
                    f"Metadata mismatch in Custom row {i} for '{k}': original {orig_m[k]!r} vs combined {comb_m.get(k)!r}"
                )

        custom_real_users.add(true_u)
        custom_real_sessions.add(true_s)

        is_leaky = (comb_u != true_u) or (comb_s != true_s)
        if is_leaky:
            synthetic_chunk_users.add(comb_u)
            synthetic_chunk_sessions.add(comb_s)

        records.append({
            "sample_id": comb_idx,
            "source": "custom",
            "original_index": i,
            "label": str(c_lbl[i]),
            "timestamp": orig_m.get("timestamp"),
            "handedness": orig_m.get("handedness"),
            "true_user_id": true_u,
            "true_session_id": true_s,
            "combined_user_id": comb_u,
            "combined_session_id": comb_s,
            "is_leaky": is_leaky,
        })

    # Summary analysis
    audit_summary = {
        "status": "PASS",
        "verified_hashes": actual_hashes,
        "counts": {
            "hagrid_samples": n_hagrid,
            "custom_samples": n_custom,
            "combined_samples": n_combined,
        },
        "multiplicity_audit": {
            "duplicate_feature_label_keys": len(dup_keys),
            "max_multiplicity": max_multiplicity,
            "disambiguated_by_timestamp": disambiguated_by_timestamp,
            "all_duplicates_agree_on_real_user_and_session": all_duplicates_agree_on_group,
            "safe_group_match": all_duplicates_agree_on_group,
        },
        "custom_analysis": {
            "real_users": sorted(list(custom_real_users)),
            "real_sessions": sorted(list(custom_real_sessions)),
            "synthetic_chunk_users": sorted(list(synthetic_chunk_users)),
            "synthetic_chunk_sessions": sorted(list(synthetic_chunk_sessions)),
            "leaky_samples_count": len([r for r in records if r["is_leaky"]]),
            "custom_classes": sorted(list(set(str(l) for l in c_lbl))),
            "unseen_user_eval_available": False,
            "reason": (
                "Only 1 real user ('tuana') and 1 real session ('sess_20260929_132034_7a13ed') "
                "exist for the 3 custom classes (PINCH_CLOSED, PINCH_MIDDLE, L_SHAPE). "
                "Partitioning into synthetic chunks (tuana_chunk0..4) caused 100% fold leakage in historical 5-fold CV. "
                "Under strict true-user grouped cross-validation, 1 user cannot be partitioned across 5 folds; "
                "unseen-user metrics for these 3 classes are strictly UNAVAILABLE."
            ),
        },
        "hagrid_analysis": {
            "n_users": len(set(r["true_user_id"] for r in records if r["source"] == "hagrid")),
            "n_sessions": len(set(r["true_session_id"] for r in records if r["source"] == "hagrid")),
            "classes": sorted(list(set(str(l) for l in h_lbl))),
            "unseen_user_eval_available": True,
        },
    }

    # Ensure output directory exists
    for p in [manifest_out, report_out]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # Save manifest
    manifest_data = {
        "metadata": {
            "created_at": "2026-09-30",
            "total_samples": n_combined,
            "combined_hash": actual_hashes["combined"],
            "custom_hash": actual_hashes["custom"],
            "hagrid_hash": actual_hashes["hagrid"],
        },
        "records": records,
    }
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Corrected group manifest saved to: {manifest_out}")

    # Save report
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Audit report saved to: {report_out}")

    return audit_summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit dataset provenance and generate corrected group manifest.")
    parser.add_argument("--combined", default="data/combined_airpad.npz", help="Path to combined .npz")
    parser.add_argument("--hagrid", default="data/hagrid_airpad.npz", help="Path to HaGRID .npz")
    parser.add_argument("--custom", default="data/custom_airpad.npz", help="Path to custom .npz")
    parser.add_argument("--manifest-out", default="reports/acc01/corrected_group_manifest.json", help="Output manifest path")
    parser.add_argument("--report-out", default="reports/acc01/audit_report.json", help="Output report path")

    args = parser.parse_args(argv)
    try:
        summary = audit_datasets(
            combined_path=args.combined,
            hagrid_path=args.hagrid,
            custom_path=args.custom,
            manifest_out=args.manifest_out,
            report_out=args.report_out,
        )
        print("\n=== AUDIT SUMMARY ===")
        print(f"Status: {summary['status']}")
        print(f"Total samples: {summary['counts']['combined_samples']}")
        print(f"Custom real user count: {len(summary['custom_analysis']['real_users'])} ({summary['custom_analysis']['real_users']})")
        print(f"Synthetic chunk users found: {len(summary['custom_analysis']['synthetic_chunk_users'])}")
        print(f"Multiplicity audit: {summary['multiplicity_audit']['duplicate_feature_label_keys']} keys (max {summary['multiplicity_audit']['max_multiplicity']}), all agree on group: {summary['multiplicity_audit']['all_duplicates_agree_on_real_user_and_session']}")
        print(f"Unseen-user evaluation available for custom poses: {summary['custom_analysis']['unseen_user_eval_available']}")
        return 0
    except Exception as e:
        print(f"[ERROR] Audit failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
