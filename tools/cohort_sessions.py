"""Offline cohort-session readiness and assignment validation utility for AirPad.

Enforces safe session-split evaluation rules:
- Known-cohort generalization / personal-use evaluation across independent sessions.
- Strict pair-keyed (user_id, session_id) assignments: two users may have sessions with
  the same name but different roles. Forbidden delimiter '::' in user_id/session_id.
- Strict session disjointness: each (user_id, session_id) belongs to exactly one role.
- Unconditional mandatory provenance validation for EVERY sample: non-blank source_file,
  non-negative integer (not bool) source_row_idx, 64-char hexadecimal source_hash,
  full source_hashes coverage, and duplicate (source_file, source_row_idx) rejection.
- Centralized validation across public helpers (validate_cohort_sessions and generate_split_manifest).
- generate_split_manifest enforces full readiness validation with explicit roster and strictly
  rejects NOT_READY inputs (raising ValueError).
- Strict JSON duplicate key rejection via object_pairs_hook.
- Output path alias guards protecting against Windows case aliases (normcase(realpath))
  and hardlinks (samefile) across inputs and outputs.
- Mandatory 3 disjoint sessions per participant in roster across train, validation, and test.
- Single-user personal-use mode allowed without >=2 users rule, provided >=3 sessions
  and full 9-class coverage across all roles.
- Class coverage: all 9 static target classes must be present in stable frames for each role.
- Transition separation: transition frames (is_transition=True) are tracked separately,
  excluded from default stable sample lists, and retained in a separate transition list.
- Deterministic fingerprinting excluding wall-clock timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from tools.collect_data import (
    LABEL_NAMES,
    load_dataset_csv,
    load_dataset_npz,
)

# Canonical 9 static gesture target classes in AirPad
STATIC_CLASSES: List[str] = [
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

ROLES: Tuple[str, str, str] = ("train", "validation", "test")

ROLE_ALIASES: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}

HEX_DIGITS: Set[str] = set(string.hexdigits)


def norm_path(p: str) -> str:
    """Normalize filesystem path for case-insensitive and link-safe comparison."""
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def paths_alias_each_other(p1: str, p2: str) -> bool:
    """Check if two file paths resolve to the same file via path normalization or samefile."""
    if norm_path(p1) == norm_path(p2):
        return True
    if os.path.exists(p1) and os.path.exists(p2):
        try:
            return os.path.samefile(p1, p2)
        except (OSError, ValueError):
            pass
    return False


def strict_json_object_pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """JSON object pairs hook that raises ValueError on duplicate keys."""
    res: Dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate key '{k}' detected in JSON object")
        res[k] = v
    return res


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_deterministic_fingerprint(
    samples: List[Dict[str, Any]],
    roster: List[str],
    mode: str,
) -> str:
    """Compute deterministic SHA-256 fingerprint over assigned samples and plan.

    Excludes wall-clock timestamps to ensure reproducibility across runs.
    """
    hasher = hashlib.sha256()
    hasher.update(f"mode={mode}|roster={sorted(roster)}".encode("utf-8"))

    sorted_samples = sorted(
        samples,
        key=lambda s: (
            s.get("role", ""),
            s.get("user_id", ""),
            s.get("session_id", ""),
            s.get("source_hash", ""),
            int(s.get("source_row_idx", 0)),
        ),
    )

    for s in sorted_samples:
        entry_str = (
            f"{s.get('role')}|{s.get('user_id')}|{s.get('session_id')}|"
            f"{s.get('source_hash')}|{s.get('source_row_idx')}|"
            f"{s.get('label')}|{bool(s.get('is_transition', False))}"
        )
        hasher.update(entry_str.encode("utf-8"))

    return hasher.hexdigest()


def normalize_identity(val: Any, field_name: str, context: str = "") -> str:
    """Validate and normalize identity string.

    Raises ValueError on null, non-string, blank, or values containing forbidden '::'.
    """
    ctx_msg = f" in {context}" if context else ""
    if val is None or not isinstance(val, str):
        raise ValueError(
            f"Invalid {field_name}{ctx_msg}: must be a non-null string, got {type(val).__name__} ({val!r})"
        )
    val_str = val.strip()
    if not val_str:
        raise ValueError(f"Invalid {field_name}{ctx_msg}: cannot be empty or whitespace-only")
    if "::" in val_str:
        raise ValueError(f"Invalid {field_name}{ctx_msg}: '{val_str}' contains forbidden delimiter '::'")
    return val_str


def parse_session_plan(plan_input: Union[str, Dict[str, Any], List[Any]]) -> Dict[Tuple[str, str], str]:
    """Parse session assignment plan into a mapping of (user_id, session_id) -> normalized role.

    Accepts:
    1. Mapping of "user_id::session_id" -> role, or JSON string/file
    2. Mapping of role -> list of {"user_id": ..., "session_id": ...} or "user_id::session_id"
    3. Nested dict: {user_id: {session_id: role}}
    4. List of dicts: [{"user_id": ..., "session_id": ..., "role": ...}]

    Uses strict_json_object_pairs_hook to reject duplicate keys in JSON inputs.
    Raises ValueError on unknown roles, duplicate assignments, or invalid identities.
    """
    if isinstance(plan_input, str):
        if os.path.exists(plan_input):
            with open(plan_input, "r", encoding="utf-8") as f:
                raw_plan = json.load(f, object_pairs_hook=strict_json_object_pairs_hook)
        else:
            raw_plan = json.loads(plan_input, object_pairs_hook=strict_json_object_pairs_hook)
    elif isinstance(plan_input, (dict, list)):
        raw_plan = plan_input
    else:
        raise ValueError(f"Invalid plan_input type: {type(plan_input)}")

    session_to_role: Dict[Tuple[str, str], str] = {}
    seen_assignments: Set[Tuple[str, str]] = set()

    def record_assignment(u_id: str, s_id: str, raw_r: Any, ctx: str) -> None:
        norm_u = normalize_identity(u_id, "user_id", ctx)
        norm_s = normalize_identity(s_id, "session_id", ctx)

        if raw_r is None or not isinstance(raw_r, str):
            raise ValueError(f"Invalid role for session ({norm_u}, {norm_s}) in {ctx}: got {raw_r!r}")

        raw_r_str = raw_r.strip().lower()
        norm_r = ROLE_ALIASES.get(raw_r_str)
        if not norm_r:
            raise ValueError(
                f"Unknown or invalid role '{raw_r}' for session ({norm_u}, {norm_s}) in {ctx}. "
                f"Allowed roles: {list(ROLES)}"
            )

        pair = (norm_u, norm_s)
        if pair in seen_assignments:
            raise ValueError(
                f"Duplicate assignment for session ({norm_u}, {norm_s}) in session plan: "
                f"already assigned to '{session_to_role[pair]}', attempted re-assignment to '{norm_r}'"
            )
        seen_assignments.add(pair)
        session_to_role[pair] = norm_r

    if isinstance(raw_plan, list):
        for idx, item in enumerate(raw_plan):
            if not isinstance(item, dict):
                raise ValueError(f"Item {idx} in session plan list must be a dictionary, got {type(item)}")
            u = item.get("user_id")
            s = item.get("session_id")
            r = item.get("role")
            record_assignment(u, s, r, f"plan list index {idx}")

    elif isinstance(raw_plan, dict):
        keys_lower = {str(k).lower(): k for k in raw_plan.keys()}
        is_role_keyed = any(k in ROLE_ALIASES for k in keys_lower)

        if is_role_keyed:
            for raw_role, sessions in raw_plan.items():
                norm_role = ROLE_ALIASES.get(str(raw_role).lower())
                if not norm_role:
                    raise ValueError(f"Unknown or invalid role key '{raw_role}' in role-keyed session plan")

                if not isinstance(sessions, (list, tuple, set)):
                    raise ValueError(f"Value for role '{raw_role}' must be a list of sessions, got {type(sessions)}")

                for s_item in sessions:
                    if isinstance(s_item, dict):
                        u = s_item.get("user_id")
                        s = s_item.get("session_id")
                    elif isinstance(s_item, str):
                        if "::" in s_item:
                            u, s = s_item.split("::", 1)
                        elif "/" in s_item:
                            u, s = s_item.split("/", 1)
                        else:
                            raise ValueError(
                                f"String session entry '{s_item}' under role '{raw_role}' must be pair-keyed "
                                f"format 'user_id::session_id'"
                            )
                    else:
                        raise ValueError(f"Invalid session entry {s_item!r} under role '{raw_role}'")

                    record_assignment(u, s, norm_role, f"role '{raw_role}' list")
        else:
            first_val = next(iter(raw_plan.values())) if raw_plan else None
            if isinstance(first_val, dict):
                for u_id, sess_map in raw_plan.items():
                    if not isinstance(sess_map, dict):
                        raise ValueError(f"Expected dict of sessions for user '{u_id}', got {type(sess_map)}")
                    for s_id, r in sess_map.items():
                        record_assignment(u_id, s_id, r, f"user '{u_id}' map")
            else:
                for pair_key, r in raw_plan.items():
                    pair_str = str(pair_key)
                    if "::" in pair_str:
                        u, s = pair_str.split("::", 1)
                    elif "/" in pair_str:
                        u, s = pair_str.split("/", 1)
                    else:
                        raise ValueError(
                            f"Plan assignment key '{pair_key}' is not pair-keyed. "
                            f"Must be format 'user_id::session_id' to distinguish sessions across users."
                        )
                    record_assignment(u, s, r, f"pair key '{pair_key}'")

    return session_to_role


def validate_samples_integrity(
    metadata: List[Dict[str, Any]],
    labels: Sequence[str],
    features: Optional[np.ndarray] = None,
    source_hashes: Optional[Dict[str, str]] = None,
) -> None:
    """Centralized strict validator for in-memory and loaded samples.

    Enforces:
    - Counts and types.
    - Non-blank identities without '::'.
    - Labels and label agreement.
    - Strict boolean is_transition.
    - Unconditional mandatory provenance:
      + non-blank source_file
      + non-negative integer (not bool) source_row_idx
      + 64-character hexadecimal source_hash
      + coverage and match in supplied source_hashes
      + unique (source_file, source_row_idx)
    """
    n_samples = len(metadata)
    if len(labels) != n_samples:
        raise ValueError(f"Sample length mismatch: labels ({len(labels)}) != metadata ({n_samples})")

    if features is not None:
        if len(features) != n_samples:
            raise ValueError(f"Sample length mismatch: features ({len(features)}) != metadata ({n_samples})")
        if features.ndim != 2 or features.shape[1] != 63:
            raise ValueError(f"Features must have shape (N, 63), got {features.shape}")
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Features contain NaN or infinite values")

    seen_source_rows: Set[Tuple[str, int]] = set()

    for idx, meta in enumerate(metadata):
        if not isinstance(meta, dict):
            raise ValueError(f"Row {idx}: metadata entry is not a dictionary")

        u = normalize_identity(meta.get("user_id"), "user_id", f"row {idx}")
        s = normalize_identity(meta.get("session_id"), "session_id", f"row {idx}")
        meta["user_id"] = u
        meta["session_id"] = s

        # Label validation and agreement
        meta_lbl = meta.get("label")
        if meta_lbl is None or not isinstance(meta_lbl, str):
            raise ValueError(f"Row {idx}: invalid label in metadata: {meta_lbl!r}")
        if meta_lbl not in LABEL_NAMES:
            raise ValueError(f"Row {idx}: unrecognized label '{meta_lbl}' in metadata")

        lbl_arr = labels[idx]
        if not isinstance(lbl_arr, str):
            raise ValueError(f"Row {idx}: label array value must be a string, got {type(lbl_arr)}")
        if lbl_arr != meta_lbl:
            raise ValueError(
                f"Row {idx}: label mismatch between label array ('{lbl_arr}') and metadata ('{meta_lbl}')"
            )

        # Boolean transition validation
        if "is_transition" not in meta:
            raise ValueError(f"Row {idx}: missing mandatory 'is_transition' flag in metadata")
        is_trans = meta["is_transition"]
        if not isinstance(is_trans, bool):
            raise ValueError(f"Row {idx}: 'is_transition' must be a boolean, got {type(is_trans).__name__} ({is_trans!r})")

        # Feature version
        feat_ver = meta.get("feature_version", 1)
        if feat_ver != 1:
            raise ValueError(f"Row {idx}: invalid feature_version {feat_ver}, expected 1")

        # Unconditional mandatory provenance validation
        src_file = meta.get("source_file")
        if not src_file or not isinstance(src_file, str) or not src_file.strip():
            raise ValueError(f"Row {idx}: missing or empty mandatory 'source_file' in metadata ({src_file!r})")

        src_row = meta.get("source_row_idx")
        # Ensure integer and strictly NOT boolean (bool is a subclass of int in Python)
        if not isinstance(src_row, int) or isinstance(src_row, bool) or src_row < 0:
            raise ValueError(
                f"Row {idx}: 'source_row_idx' must be a non-negative integer (not bool), "
                f"got {type(src_row).__name__} ({src_row!r})"
            )

        src_hash = meta.get("source_hash")
        if not src_hash or not isinstance(src_hash, str) or len(src_hash) != 64 or not all(c in HEX_DIGITS for c in src_hash):
            raise ValueError(
                f"Row {idx}: 'source_hash' must be a 64-character hexadecimal SHA-256 hash, "
                f"got {src_hash!r}"
            )

        if source_hashes is not None:
            norm_file_map = {norm_path(k): v for k, v in source_hashes.items()}
            src_norm = norm_path(src_file)
            if src_norm not in norm_file_map:
                raise ValueError(
                    f"Row {idx}: source_file '{src_file}' not found in supplied source_hashes map"
                )
            expected_h = norm_file_map[src_norm]
            if src_hash.lower() != expected_h.lower():
                raise ValueError(
                    f"Row {idx}: source_hash mismatch for '{src_file}': "
                    f"sample has {src_hash[:12]}..., map has {expected_h[:12]}..."
                )

        row_key = (norm_path(src_file), src_row)
        if row_key in seen_source_rows:
            raise ValueError(f"Row {idx}: duplicate provenance row key detected: ({src_file}, {src_row})")
        seen_source_rows.add(row_key)


def validate_input_files(dataset_paths: Sequence[str]) -> Dict[str, str]:
    """Validate input file paths and ensure no duplicates by resolved path, normcase, or hash."""
    if not dataset_paths:
        raise ValueError("No dataset paths provided")

    file_hashes: Dict[str, str] = {}
    validated_paths: List[str] = []

    for path in dataset_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: '{path}'")

        for existing_path in validated_paths:
            if paths_alias_each_other(path, existing_path):
                raise ValueError(
                    f"Duplicate input file detected: '{path}' aliases '{existing_path}'"
                )

        h = compute_file_sha256(path)
        for existing_path, existing_hash in file_hashes.items():
            if h == existing_hash:
                raise ValueError(
                    f"Duplicate input file content detected: '{path}' has identical SHA-256 hash "
                    f"({h[:12]}...) to '{existing_path}'"
                )

        file_hashes[path] = h
        validated_paths.append(path)

    return file_hashes


def validate_output_paths_no_alias(
    dataset_paths: Sequence[str],
    output_report_path: Optional[str] = None,
    output_manifest_path: Optional[str] = None,
) -> None:
    """Ensure output report and manifest paths do not alias ANY input dataset or each other."""
    for in_p in dataset_paths:
        if output_report_path and paths_alias_each_other(output_report_path, in_p):
            raise ValueError(f"Output report path '{output_report_path}' aliases input dataset '{in_p}'")
        if output_manifest_path and paths_alias_each_other(output_manifest_path, in_p):
            raise ValueError(f"Output manifest path '{output_manifest_path}' aliases input dataset '{in_p}'")

    if output_report_path and output_manifest_path:
        if paths_alias_each_other(output_report_path, output_manifest_path):
            raise ValueError(
                f"Output report path and manifest path cannot be the same file: '{output_report_path}'"
            )


def load_dataset_with_provenance(
    filepath: str,
    expected_hash: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], str]:
    """Load dataset from .npz or .csv, enforcing integrity and strict provenance fields."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    sha256_hash = compute_file_sha256(filepath)
    if expected_hash and sha256_hash.lower() != expected_hash.lower():
        raise ValueError(
            f"File '{filepath}' hash mismatch: expected {expected_hash}, calculated {sha256_hash}"
        )

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".npz":
        features, labels, metadata = load_dataset_npz(filepath)
    elif ext == ".csv":
        features, labels, metadata = load_dataset_csv(filepath)
    else:
        raise ValueError(f"Unsupported dataset format '{ext}'. Must be .npz or .csv")

    for idx, meta in enumerate(metadata):
        meta["source_file"] = filepath
        meta["source_hash"] = sha256_hash
        meta["source_row_idx"] = idx

    validate_samples_integrity(metadata, labels, features, {filepath: sha256_hash})

    return features, labels, metadata, sha256_hash


def validate_session_disjointness_across_files(
    metadata_by_file: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Ensure that the same raw session (user_id, session_id) does not appear across multiple files."""
    session_to_files: Dict[Tuple[str, str], List[str]] = {}
    for f_path, meta_list in metadata_by_file.items():
        for m in meta_list:
            pair = (m["user_id"], m["session_id"])
            if pair not in session_to_files:
                session_to_files[pair] = []
            if f_path not in session_to_files[pair]:
                session_to_files[pair].append(f_path)

    for pair, files in session_to_files.items():
        if len(files) > 1:
            raise ValueError(
                f"Duplicate raw session ingestion across files: session {pair} appears in multiple files: {files}"
            )


def validate_roster(raw_roster: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Validate participant roster."""
    if raw_roster is None:
        return None

    seen: Set[str] = set()
    cleaned_roster: List[str] = []

    for idx, entry in enumerate(raw_roster):
        u = normalize_identity(entry, "roster participant", f"roster index {idx}")
        if u in seen:
            raise ValueError(f"Duplicate participant in roster: '{u}'")
        seen.add(u)
        cleaned_roster.append(u)

    return cleaned_roster


def validate_cohort_sessions(
    metadata: List[Dict[str, Any]],
    labels: Sequence[str],
    session_assignments: Optional[Dict[Tuple[str, str], str]] = None,
    roster: Optional[Sequence[str]] = None,
    mode: str = "auto",
    source_files: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate cohort sessions readiness and assignment constraints.

    Centralized sample integrity validation runs first, unconditionally validating provenance
    and raising ValueError immediately on corruptions, mismatches, or invalid types.
    """
    validate_samples_integrity(metadata, labels, source_hashes=source_files)

    cleaned_roster = validate_roster(roster)

    detected_user_sessions: Dict[Tuple[str, str], List[int]] = {}
    detected_users: Set[str] = set()

    for idx, meta in enumerate(metadata):
        u = meta["user_id"]
        s = meta["session_id"]
        pair = (u, s)
        detected_users.add(u)
        if pair not in detected_user_sessions:
            detected_user_sessions[pair] = []
        detected_user_sessions[pair].append(idx)

    # Validate session assignment integrity against data sessions
    if session_assignments is not None:
        for (u, s), r in session_assignments.items():
            normalize_identity(u, "user_id in plan")
            normalize_identity(s, "session_id in plan")
            if r not in ROLES:
                raise ValueError(f"Invalid role '{r}' for session ({u}, {s}) in plan")

        plan_sessions = set(session_assignments.keys())
        data_sessions = set(detected_user_sessions.keys())

        unknown_in_plan = plan_sessions - data_sessions
        if unknown_in_plan:
            sample_unknown = sorted(list(unknown_in_plan))[:5]
            raise ValueError(
                f"Session plan references {len(unknown_in_plan)} session(s) not found in dataset: {sample_unknown}"
            )

        unassigned_in_data = data_sessions - plan_sessions
        if unassigned_in_data:
            sample_unassigned = sorted(list(unassigned_in_data))[:5]
            raise ValueError(
                f"Dataset contains {len(unassigned_in_data)} unassigned session(s): {sample_unassigned}"
            )

    missing_criteria: List[str] = []

    if cleaned_roster is None or len(cleaned_roster) == 0:
        missing_criteria.append("Explicit participant roster is required; none provided.")
        eval_roster = sorted(list(detected_users))
    else:
        eval_roster = cleaned_roster

    if mode == "auto":
        mode_val = "personal" if len(eval_roster) <= 1 else "cohort"
    else:
        mode_val = mode.lower()

    if mode_val not in ("personal", "cohort"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'personal', 'cohort', or 'auto'")

    if mode_val == "cohort" and len(eval_roster) < 2:
        missing_criteria.append(
            f"Cohort mode requires at least 2 participants in roster; got {len(eval_roster)}: {eval_roster}"
        )
    elif mode_val == "personal" and len(eval_roster) != 1:
        missing_criteria.append(
            f"Personal mode requires exactly 1 participant in roster; got {len(eval_roster)}: {eval_roster}"
        )

    missing_from_data = [u for u in eval_roster if u not in detected_users]
    if missing_from_data:
        missing_criteria.append(f"Roster participants missing from dataset: {missing_from_data}")

    extra_users = sorted(list(detected_users - set(eval_roster)))
    if extra_users:
        missing_criteria.append(f"Dataset contains unrostered participant(s): {extra_users}")

    sessions_summary: Dict[str, Dict[str, Any]] = {}
    user_session_counts: Dict[str, int] = {u: 0 for u in eval_roster}

    for (u_id, s_id), indices in detected_user_sessions.items():
        sess_key = f"{u_id}::{s_id}"
        s_entry = {
            "session_key": sess_key,
            "session_id": s_id,
            "user_id": u_id,
            "total_samples": len(indices),
            "stable_samples": 0,
            "transition_samples": 0,
            "classes": {},
            "stable_classes": {},
        }

        if u_id in user_session_counts:
            user_session_counts[u_id] += 1

        for idx in indices:
            meta = metadata[idx]
            lbl = labels[idx]
            is_trans = bool(meta.get("is_transition", False))

            s_entry["classes"][lbl] = s_entry["classes"].get(lbl, 0) + 1
            if is_trans:
                s_entry["transition_samples"] += 1
            else:
                s_entry["stable_samples"] += 1
                if lbl in STATIC_CLASSES:
                    s_entry["stable_classes"][lbl] = s_entry["stable_classes"].get(lbl, 0) + 1

        sessions_summary[sess_key] = s_entry

    for u in eval_roster:
        n_sess = user_session_counts.get(u, 0)
        if n_sess < 3:
            u_sessions = [s for (user, s) in detected_user_sessions.keys() if user == u]
            missing_criteria.append(
                f"Participant '{u}' has only {n_sess} session(s) ({u_sessions}); "
                f"minimum 3 independent sessions required (at least 1 each for train, validation, test)."
            )

    present_static_classes = set()
    for s_info in sessions_summary.values():
        if s_info["user_id"] in eval_roster:
            present_static_classes.update(s_info["stable_classes"].keys())

    missing_classes = sorted(list(set(STATIC_CLASSES) - present_static_classes))
    if missing_classes:
        missing_criteria.append(
            f"Dataset contains only {len(present_static_classes)} of {len(STATIC_CLASSES)} target classes: "
            f"missing {missing_classes}"
        )

    if session_assignments is None:
        missing_criteria.append("Explicit session assignment plan is required; none provided.")
        effective_plan: Dict[Tuple[str, str], str] = {}
    else:
        effective_plan = session_assignments

    role_assignments: Dict[str, Dict[str, Any]] = {
        r: {
            "sessions": [],
            "participants": [],
            "total_samples": 0,
            "stable_samples": 0,
            "transition_samples": 0,
            "stable_class_counts": {c: 0 for c in STATIC_CLASSES},
            "participant_stable_class_counts": {u: {c: 0 for c in STATIC_CLASSES} for u in eval_roster},
        }
        for r in ROLES
    }

    if effective_plan:
        for (u_id, s_id), role in effective_plan.items():
            sess_key = f"{u_id}::{s_id}"
            s_info = sessions_summary.get(sess_key)
            if not s_info:
                continue

            r_entry = role_assignments[role]
            r_entry["sessions"].append(sess_key)
            if u_id not in r_entry["participants"]:
                r_entry["participants"].append(u_id)

            r_entry["total_samples"] += s_info["total_samples"]
            r_entry["stable_samples"] += s_info["stable_samples"]
            r_entry["transition_samples"] += s_info["transition_samples"]
            for c, cnt in s_info["stable_classes"].items():
                r_entry["stable_class_counts"][c] = r_entry["stable_class_counts"].get(c, 0) + cnt
                if u_id in eval_roster:
                    r_entry["participant_stable_class_counts"][u_id][c] += cnt

        for r in ROLES:
            r_entry = role_assignments[r]
            r_parts = set(r_entry["participants"])
            for u in eval_roster:
                if u not in r_parts:
                    missing_criteria.append(f"Role '{r}' missing sessions from participant '{u}'")
                else:
                    u_counts = r_entry["participant_stable_class_counts"][u]
                    u_stable_total = sum(u_counts.values())
                    if u_stable_total == 0:
                        missing_criteria.append(
                            f"Participant '{u}' in role '{r}' has zero stable samples (only transition frames or empty)"
                        )
                    else:
                        missing_for_u = [c for c in STATIC_CLASSES if u_counts.get(c, 0) == 0]
                        if missing_for_u:
                            missing_criteria.append(
                                f"Participant '{u}' in role '{r}' missing stable samples for classes: {missing_for_u}"
                            )

    total_samples = len(metadata)
    transition_samples = sum(1 for m in metadata if bool(m.get("is_transition", False)))
    stable_samples = total_samples - transition_samples

    status = "READY" if len(missing_criteria) == 0 else "NOT_READY"
    serialized_plan = {f"{u}::{s}": r for (u, s), r in effective_plan.items()}

    report: Dict[str, Any] = {
        "status": status,
        "mode": mode_val,
        "target_classes": STATIC_CLASSES,
        "roster": eval_roster,
        "total_samples": total_samples,
        "stable_samples": stable_samples,
        "transition_samples": transition_samples,
        "source_files": source_files or {},
        "sessions_summary": sessions_summary,
        "role_assignments": role_assignments,
        "session_plan": serialized_plan,
        "missing_criteria": missing_criteria,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return report


def generate_split_manifest(
    metadata: List[Dict[str, Any]],
    labels: Sequence[str],
    session_assignments: Dict[Tuple[str, str], str],
    roster: Sequence[str],
    source_hashes: Optional[Dict[str, str]] = None,
    mode: str = "auto",
) -> Dict[str, Any]:
    """Generate reproducible split manifest, validating and separating transitions from stable samples.

    Calls full readiness validation with explicit roster and rejects NOT_READY inputs with ValueError.
    """
    cleaned_roster = validate_roster(roster)
    if not cleaned_roster:
        raise ValueError("Explicit participant roster is required to generate split manifest")

    # Run full readiness validation
    report = validate_cohort_sessions(
        metadata=metadata,
        labels=labels,
        session_assignments=session_assignments,
        roster=cleaned_roster,
        mode=mode,
        source_files=source_hashes,
    )

    if report["status"] != "READY":
        raise ValueError(
            f"Cannot generate split manifest: dataset is NOT_READY ({report['missing_criteria']})"
        )

    stable_samples: List[Dict[str, Any]] = []
    transition_samples: List[Dict[str, Any]] = []

    for idx, meta in enumerate(metadata):
        u = meta["user_id"]
        s = meta["session_id"]
        pair = (u, s)
        role = session_assignments[pair]

        src_file = meta["source_file"]
        src_hash = meta["source_hash"]
        src_row = meta["source_row_idx"]

        sample_record = {
            "source_file": src_file,
            "source_hash": src_hash,
            "source_row_idx": src_row,
            "user_id": u,
            "session_id": s,
            "label": str(labels[idx]),
            "role": role,
            "is_transition": bool(meta.get("is_transition", False)),
        }

        if sample_record["is_transition"]:
            transition_samples.append(sample_record)
        else:
            stable_samples.append(sample_record)

    all_assigned = stable_samples + transition_samples
    eval_roster = cleaned_roster
    fingerprint = compute_deterministic_fingerprint(all_assigned, eval_roster, report["mode"])

    roles_summary: Dict[str, Dict[str, Any]] = {
        r: {
            "stable_count": sum(1 for s in stable_samples if s["role"] == r),
            "transition_count": sum(1 for s in transition_samples if s["role"] == r),
        }
        for r in ROLES
    }

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "fingerprint": fingerprint,
        "mode": report["mode"],
        "roster": eval_roster,
        "total_stable_samples": len(stable_samples),
        "total_transition_samples": len(transition_samples),
        "source_files": source_hashes or {},
        "roles": roles_summary,
        "stable_samples": stable_samples,
        "transition_samples": transition_samples,
    }
    return manifest


def audit_dataset_readiness(
    dataset_paths: Sequence[str],
    roster: Optional[Sequence[str]] = None,
    mode: str = "auto",
    session_plan: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
    output_report_path: Optional[str] = None,
    output_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit one or more dataset files for cohort session readiness.

    Enforces output path alias guards, strict integrity checks, and manifest safety.
    """
    # 1. Output path alias guards (fail fast before reading or writing)
    validate_output_paths_no_alias(dataset_paths, output_report_path, output_manifest_path)

    # 2. Validate input files (no duplicates by path, normcase, or hash)
    source_files = validate_input_files(dataset_paths)

    # 3. Parse session plan if provided (rejects duplicate JSON keys)
    parsed_plan = parse_session_plan(session_plan) if session_plan else None

    # 4. Load datasets with provenance
    combined_metadata: List[Dict[str, Any]] = []
    combined_labels: List[str] = []
    metadata_by_file: Dict[str, List[Dict[str, Any]]] = {}

    for path in dataset_paths:
        _, lbls, meta, file_hash = load_dataset_with_provenance(path, expected_hash=source_files[path])
        combined_metadata.extend(meta)
        combined_labels.extend(lbls.tolist() if hasattr(lbls, "tolist") else list(lbls))
        metadata_by_file[path] = meta

    # 5. Check for duplicate session ingestion across distinct files
    validate_session_disjointness_across_files(metadata_by_file)

    # 6. Validate cohort sessions & assignments
    report = validate_cohort_sessions(
        metadata=combined_metadata,
        labels=combined_labels,
        session_assignments=parsed_plan,
        roster=roster,
        mode=mode,
        source_files=source_files,
    )

    # 7. Compute and validate requested manifest in memory BEFORE writing either output
    manifest: Optional[Dict[str, Any]] = None
    if output_manifest_path:
        if report["status"] == "READY":
            if parsed_plan is None:
                raise ValueError("Cannot generate split manifest without an explicit session plan")
            manifest = generate_split_manifest(
                metadata=combined_metadata,
                labels=combined_labels,
                session_assignments=parsed_plan,
                roster=report["roster"],
                source_hashes=source_files,
                mode=report["mode"],
            )
        else:
            # Do NOT emit manifest on NOT_READY
            manifest = None

    # 8. Write report if requested
    if output_report_path:
        out_dir = os.path.dirname(os.path.abspath(output_report_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # 9. Write manifest if READY and computed
    if output_manifest_path and manifest is not None:
        man_dir = os.path.dirname(os.path.abspath(output_manifest_path))
        if man_dir:
            os.makedirs(man_dir, exist_ok=True)
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit AirPad custom datasets for known-cohort / personal session readiness."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="One or more dataset paths (.npz or .csv)",
    )
    parser.add_argument(
        "--roster",
        default=None,
        help="Comma-separated list of participant user_ids (e.g. tuana or tuana,friend_01)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "personal", "cohort"],
        default="auto",
        help="Evaluation mode: personal (single-user), cohort (multi-user), or auto",
    )
    parser.add_argument(
        "--plan",
        default=None,
        help="Path to JSON file specifying pair-keyed session assignment plan",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON readiness report",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Path to write JSON split manifest if dataset is READY and valid plan exists",
    )
    parser.add_argument(
        "--fail-on-not-ready",
        action="store_true",
        help="Exit with nonzero status code (1) if status is NOT_READY",
    )

    args = parser.parse_args(argv)

    roster = [u.strip() for u in args.roster.split(",")] if args.roster else None

    report = audit_dataset_readiness(
        dataset_paths=args.data,
        roster=roster,
        mode=args.mode,
        session_plan=args.plan,
        output_report_path=args.output,
        output_manifest_path=args.manifest_out,
    )

    print("\n" + "=" * 60)
    print("      AIRPAD COHORT SESSIONS READINESS AUDIT")
    print("=" * 60)
    print(f"Status:             {report['status']}")
    print(f"Mode:               {report['mode']}")
    print(f"Roster:             {report['roster']}")
    print(f"Total Samples:      {report['total_samples']}")
    print(f"Stable Samples:     {report['stable_samples']}")
    print(f"Transition Samples: {report['transition_samples']}")
    print(f"Distinct Sessions:  {len(report['sessions_summary'])}")

    if report["missing_criteria"]:
        print("\nMissing Criteria / Blockers:")
        for idx, crit in enumerate(report["missing_criteria"], 1):
            print(f"  {idx}. {crit}")

    if args.output:
        print(f"\nReport written to: {args.output}")
    print("=" * 60 + "\n")

    if args.fail_on_not_ready and report["status"] != "READY":
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
