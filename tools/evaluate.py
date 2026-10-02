"""Clean fixed-seed evaluation CLI for AirPad RBF-SVM baseline.

Strict zero-leakage evaluation protocol:
- Scaler is fit exclusively on training fold.
- Fixed hyperparameters (C=10.0, gamma=0.01, class_weight='balanced').
- Rejection policy: decision_threshold=0.0, margin_threshold=0.15.
- Metrics compute UNKNOWN as false negatives in full denominator.
- Rejects synthetic chunk identities (tuana_chunk0..4) and binds to dataset hash.
- Rigorously validates corrected manifest record count, complete coverage, and row labels.
- Enforces >=2 distinct real users per target class; rejects custom classes with single user.
- Asserts zero user leakage AND zero session leakage per fold.
- Asserts each evaluated source row appears exactly once as test sample.
- Reports fold classes absent from train if any.
- Exports split fingerprint, evaluator source hash, and detailed predictions.
- Explicitly declares custom 3-class unseen-user metrics as UNAVAILABLE; never claims 9-class improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.gesture.pose import HandPose
from src.ml.evaluation import compute_metrics, group_kfold_split
from src.ml.svm import SVMClassifier
from tools.collect_data import load_dataset_npz
from tools.train import extract_groups, validate_dataset

# Known baseline dataset hashes
DATASET_HASHES = {
    "custom": "498699e1f28a78031193476fbafff2a2cd307bb9d82a8baee6a57f3097019169",
    "hagrid": "ec38045d608aa6d30386796e4e0586ff619468d4abfd755619835bb8fa61c081",
    "combined": "c0c6d4e26c26f8a303077c6de971a54c2d4085444f9479cef9e021986cc9e8bf",
}


def compute_file_sha256(filepath: str) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_split_fingerprint(fold_test_source_ids: List[List[int]]) -> str:
    """Compute deterministic SHA-256 fingerprint of fold test splits."""
    payload = json.dumps(fold_test_source_ids, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_and_validate_corrected_manifest(
    metadata: List[Dict[str, Any]],
    labels: List[str] | np.ndarray,
    manifest_path: str,
    dataset_hash: str,
) -> Tuple[str, Dict[str, Any]]:
    """Validate and apply corrected provenance manifest.

    Validates:
    - manifest combined_hash matches dataset_hash
    - exact record count matches dataset length
    - unique and complete sample_id coverage [0..N-1]
    - row label mapping matches dataset labels exactly
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Corrected group manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_json = json.load(f)

    manifest_meta = manifest_json.get("metadata", {})
    if manifest_meta.get("combined_hash") != dataset_hash:
        raise ValueError(
            f"Corrected manifest hash mismatch: manifest bound to {manifest_meta.get('combined_hash')}, "
            f"dataset has {dataset_hash}"
        )

    records = manifest_json.get("records", [])
    n_expected = len(labels)
    if len(records) != n_expected:
        raise ValueError(
            f"Manifest record count mismatch: expected {n_expected} records, found {len(records)}"
        )

    # Validate unique complete sample_id coverage and row label matching
    seen_ids = set()
    for rec in records:
        sid = rec.get("sample_id")
        if sid is None or not isinstance(sid, int):
            raise ValueError(f"Invalid sample_id {sid!r} in manifest record: {rec}")
        if sid in seen_ids or sid < 0 or sid >= n_expected:
            raise ValueError(f"Duplicate or out-of-range sample_id {sid} in manifest")
        seen_ids.add(sid)

        # Check label matching
        expected_lbl = str(labels[sid])
        rec_lbl = str(rec.get("label"))
        if rec_lbl != expected_lbl:
            raise ValueError(
                f"Label mismatch in manifest at sample {sid}: expected '{expected_lbl}', got '{rec_lbl}'"
            )

    if seen_ids != set(range(n_expected)):
        raise ValueError("Manifest does not provide complete 0..N-1 sample_id coverage")

    # Apply true groups by sample_id mapping (not blind enumeration)
    for rec in records:
        sid = rec["sample_id"]
        metadata[sid]["user_id"] = rec["true_user_id"]
        metadata[sid]["session_id"] = rec["true_session_id"]

    manifest_hash = compute_file_sha256(manifest_path)
    return manifest_hash, manifest_meta


def run_evaluation(
    dataset_path: str = "data/hagrid_airpad.npz",
    group_key: str = "user_id",
    n_splits: int = 5,
    seed: int = 42,
    sample_limit: Optional[int] = None,
    C: float = 10.0,
    gamma: float | str = 0.01,
    class_weight: Optional[str] = "balanced",
    decision_threshold: float = 0.0,
    margin_threshold: float = 0.15,
    output_dir: str = "reports/acc01/smoke",
    split_manifest_name: str = "split_manifest.json",
    predictions_name: str = "predictions.jsonl",
    metrics_name: str = "clean_baseline_metrics.json",
    corrected_manifest_path: Optional[str] = "reports/acc01/corrected_group_manifest.json",
    scaler_spy: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute clean baseline evaluation with strict zero-leakage protocol."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_hash = compute_file_sha256(dataset_path)
    evaluator_source_hash = compute_file_sha256(__file__)

    # 1. Load and validate dataset
    features, labels, metadata = load_dataset_npz(dataset_path)
    validate_dataset(features, labels, metadata, raise_on_error=True)
    n_total_source_rows = len(labels)

    # Check for known synthetic chunk leakage
    raw_user_ids = [m.get("user_id", "") for m in metadata]
    has_synthetic_chunks = any(
        isinstance(u, str) and u.startswith("tuana_chunk") for u in raw_user_ids
    )

    manifest_hash = None
    if has_synthetic_chunks:
        if not corrected_manifest_path:
            raise ValueError(
                "Dataset contains synthetic chunk groupings ('tuana_chunk*') which cause 100% fold leakage! "
                "Leaky CV evaluation is strictly rejected. Provide a valid corrected group manifest."
            )
        manifest_hash, _ = apply_and_validate_corrected_manifest(
            metadata=metadata,
            labels=labels,
            manifest_path=corrected_manifest_path,
            dataset_hash=dataset_hash,
        )
        print(f"[INFO] Successfully applied and validated corrected manifest (hash: {manifest_hash[:16]}...)")

    # 2. Filter trainable classes and track exact source row IDs
    trainable_source_indices: List[int] = []
    for idx, lbl in enumerate(labels):
        lbl_str = str(lbl)
        if lbl_str not in {"NO_HAND", "UNKNOWN", str(int(HandPose.NO_HAND)), str(int(HandPose.UNKNOWN))}:
            trainable_source_indices.append(idx)

    # Apply deterministic stratified sample limit if provided (for smoke testing)
    if sample_limit is not None and sample_limit < len(trainable_source_indices):
        trainable_labels = [str(labels[i]) for i in trainable_source_indices]
        unique_cls = sorted(list(set(trainable_labels)))
        per_class_limit = max(2, sample_limit // len(unique_cls))
        selected_source_indices = []
        for cls in unique_cls:
            cls_source_idxs = [
                src_i for src_i, l in zip(trainable_source_indices, trainable_labels) if l == cls
            ]
            selected_source_indices.extend(cls_source_idxs[:per_class_limit])
        selected_source_indices = sorted(selected_source_indices)
        print(
            f"[INFO] Applying deterministic stratified sample limit: {len(selected_source_indices)} / "
            f"{len(trainable_source_indices)} samples ({per_class_limit} per class)"
        )
    else:
        selected_source_indices = trainable_source_indices

    # Extract evaluated subset with exact source tracking
    X = features[selected_source_indices]
    y = np.array([str(labels[i]) for i in selected_source_indices])
    meta = [metadata[i] for i in selected_source_indices]
    n_evaluated_samples = len(X)

    target_classes = sorted(list(set(y)))
    print(
        f"Evaluating {n_evaluated_samples} samples across {len(target_classes)} classes: {target_classes}"
    )

    # 3. Extract and validate groups
    groups = extract_groups(meta, group_key=group_key)
    sessions = extract_groups(meta, group_key="session_id")
    users = extract_groups(meta, group_key="user_id")
    unique_groups = np.unique(groups)
    print(f"Groups: {len(unique_groups)} unique groups by '{group_key}'")
    # Validate independent user support per target class (>= 2 real users per class required)
    for cls in target_classes:
        cls_mask = (y == cls)
        cls_users = np.unique(users[cls_mask])
        if len(cls_users) < 2:
            raise ValueError(
                f"Target class '{cls}' lacks independent user support: found only {len(cls_users)} real user(s) "
                f"({cls_users.tolist()}). Clean grouped cross-validation requires at least 2 distinct real users per class. "
                f"Evaluation for this class is strictly UNAVAILABLE; clean 9-class evaluation is blocked."
            )

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Insufficient unique groups ({len(unique_groups)}) for n_splits={n_splits}. "
            f"Grouped CV requires at least {n_splits} groups."
        )

    # 4. Generate folds with strict user AND session disjointness
    fold_splits = list(group_kfold_split(X, y, groups, n_splits=n_splits, seed=seed))
    fold_test_source_ids: List[List[int]] = []
    all_tested_source_ids: List[int] = []

    split_manifest_folds = []
    for fold_idx, (tr_idx, te_idx) in enumerate(fold_splits):
        tr_source_ids = [selected_source_indices[i] for i in tr_idx]
        te_source_ids = [selected_source_indices[i] for i in te_idx]
        fold_test_source_ids.append(te_source_ids)
        all_tested_source_ids.extend(te_source_ids)

        # Assertion: zero user leakage
        tr_users = set(users[tr_idx])
        te_users = set(users[te_idx])
        user_overlap = tr_users.intersection(te_users)
        if len(user_overlap) > 0:
            raise RuntimeError(f"User leakage detected in fold {fold_idx}: {user_overlap}")

        # Assertion: zero real-session leakage
        tr_sessions = set(sessions[tr_idx])
        te_sessions = set(sessions[te_idx])
        session_overlap = tr_sessions.intersection(te_sessions)
        if len(session_overlap) > 0:
            raise RuntimeError(f"Session leakage detected in fold {fold_idx}: {session_overlap}")

        # Compute per-class support and assert no absent training classes
        tr_labels = y[tr_idx]
        te_labels = y[te_idx]
        tr_supp = {cls: int(np.sum(tr_labels == cls)) for cls in target_classes}
        te_supp = {cls: int(np.sum(te_labels == cls)) for cls in target_classes}

        for cls in target_classes:
            if tr_supp[cls] == 0:
                raise ValueError(
                    f"Fold {fold_idx} has absent training class: '{cls}' has 0 training samples."
                )

        split_manifest_folds.append({
            "fold_idx": fold_idx,
            "train_samples": len(tr_idx),
            "test_samples": len(te_idx),
            "train_users": len(tr_users),
            "test_users": len(te_users),
            "train_sessions": len(tr_sessions),
            "test_sessions": len(te_sessions),
            "user_leakage_count": 0,
            "session_leakage_count": 0,
            "train_support": tr_supp,
            "test_support": te_supp,
            "train_source_row_ids": tr_source_ids,
            "test_source_row_ids": te_source_ids,
        })

    # Assertion: each evaluated source row appears EXACTLY ONCE as a test sample
    assert len(all_tested_source_ids) == len(selected_source_indices), (
        f"Mismatch in test sample count: {len(all_tested_source_ids)} vs {len(selected_source_indices)}"
    )
    assert sorted(all_tested_source_ids) == sorted(selected_source_indices), (
        "Every evaluated source row must appear exactly once in the test set across all folds!"
    )

    split_fingerprint = compute_split_fingerprint(fold_test_source_ids)

    split_manifest = {
        "dataset_path": dataset_path,
        "dataset_hash": dataset_hash,
        "evaluator_source_hash": evaluator_source_hash,
        "manifest_hash": manifest_hash,
        "split_fingerprint": split_fingerprint,
        "group_key": group_key,
        "n_splits": n_splits,
        "seed": seed,
        "sample_limit": sample_limit,
        "n_total_source_rows": n_total_source_rows,
        "n_evaluated_samples": n_evaluated_samples,
        "selected_source_indices": selected_source_indices,
        "target_classes": target_classes,
        "folds": split_manifest_folds,
    }

    split_manifest_path = os.path.join(output_dir, split_manifest_name)
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"Split manifest saved to: {split_manifest_path} (Fingerprint: {split_fingerprint[:16]}...)")

    # 5. Run grouped CV evaluation
    all_true: List[str] = []
    all_pred_raw: List[str] = []
    all_pred_rejected: List[str] = []
    prediction_records: List[Dict[str, Any]] = []

    per_fold_metrics_clean: List[Dict[str, Any]] = []
    per_fold_metrics_raw: List[Dict[str, Any]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(fold_splits):
        print(f"Running Fold {fold_idx + 1}/{n_splits} (train: {len(tr_idx)}, test: {len(te_idx)})...")
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        te_source_ids = [selected_source_indices[i] for i in te_idx]

        # Fit SVMClassifier: StandardScaler is strictly fit on X_tr only
        clf = SVMClassifier(
            C=C,
            gamma=gamma,
            class_weight=class_weight,
        ).fit(X_tr, y_tr)

        # Scaler spy hook for tests
        if scaler_spy is not None:
            scaler_spy.append({
                "fold": fold_idx,
                "scaler_mean": clf.scaler.mean_.copy(),
                "scaler_scale": clf.scaler.scale_.copy(),
                "train_mean": X_tr.mean(axis=0),
                "train_std": X_tr.std(axis=0),
                "test_mean": X_te.mean(axis=0),
            })

        # Raw predictions
        y_pred_raw = clf.predict(X_te)
        # Predictions with rejection
        y_pred_rej = clf.predict_with_rejection(
            X_te,
            decision_threshold=decision_threshold,
            margin_threshold=margin_threshold,
            unknown_label="UNKNOWN",
        )

        # Compute fold metrics
        m_raw = compute_metrics(y_te, y_pred_raw, labels=target_classes)
        m_rej = compute_metrics(y_te, y_pred_rej, labels=target_classes)
        per_fold_metrics_raw.append(m_raw)
        per_fold_metrics_clean.append(m_rej)

        for i, idx in enumerate(te_idx):
            src_row_id = selected_source_indices[idx]
            record = {
                "source_row_id": int(src_row_id),
                "sample_id": int(src_row_id),
                "fold": fold_idx,
                "user_id": str(meta[idx].get("user_id")),
                "session_id": str(meta[idx].get("session_id")),
                "true_label": str(y_te[i]),
                "pred_raw": str(y_pred_raw[i]),
                "pred_rejected": str(y_pred_rej[i]),
                "is_rejected": bool(y_pred_rej[i] == "UNKNOWN"),
            }
            prediction_records.append(record)
            all_true.append(str(y_te[i]))
            all_pred_raw.append(str(y_pred_raw[i]))
            all_pred_rejected.append(str(y_pred_rej[i]))

    # 6. Export prediction records
    pred_path = os.path.join(output_dir, predictions_name)
    with open(pred_path, "w", encoding="utf-8") as f:
        for rec in prediction_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Prediction records saved to: {pred_path} ({len(prediction_records)} records)")

    # 7. Compute overall out-of-fold metrics
    oof_metrics_raw = compute_metrics(all_true, all_pred_raw, labels=target_classes)
    oof_metrics_clean = compute_metrics(all_true, all_pred_rejected, labels=target_classes)

    overall_results = {
        "evaluation_name": "ACC-01 Clean Six-Class HaGRID Baseline Evaluation",
        "dataset_path": dataset_path,
        "dataset_hash": dataset_hash,
        "evaluator_source_hash": evaluator_source_hash,
        "manifest_hash": manifest_hash,
        "split_fingerprint": split_fingerprint,
        "sample_limit": sample_limit,
        "n_total_source_rows": n_total_source_rows,
        "n_evaluated_samples": n_evaluated_samples,
        "target_classes": target_classes,
        "group_key": group_key,
        "n_splits": n_splits,
        "seed": seed,
        "hyperparameters": {
            "C": C,
            "gamma": gamma,
            "class_weight": class_weight,
            "decision_threshold": decision_threshold,
            "margin_threshold": margin_threshold,
        },
        "custom_classes_unseen_user_status": {
            "status": "UNAVAILABLE",
            "reason": (
                "Only 1 real user ('tuana') and 1 real session ('sess_20260929_132034_7a13ed') "
                "exist for PINCH_CLOSED, PINCH_MIDDLE, and L_SHAPE. "
                "Clean unseen-user generalization metrics are strictly unavailable without additional user data. "
                "This benchmark evaluates the 6 HaGRID classes only; 9-class clean performance cannot be claimed."
            ),
            "affected_classes": ["PINCH_CLOSED", "PINCH_MIDDLE", "L_SHAPE"],
        },
        "clean_baseline_post_rejection": {
            "macro_f1": oof_metrics_clean["macro_f1"],
            "accuracy": oof_metrics_clean["accuracy"],
            "balanced_accuracy": oof_metrics_clean["balanced_accuracy"],
            "support": oof_metrics_clean["support"],
            "coverage": oof_metrics_clean["coverage"],
            "rejection_rate": oof_metrics_clean["rejection_rate"],
            "per_class": oof_metrics_clean["per_class"],
            "confusion_matrix": oof_metrics_clean["confusion_matrix"],
            "confusion_row_labels": oof_metrics_clean["confusion_row_labels"],
            "confusion_col_labels": oof_metrics_clean["confusion_col_labels"],
        },
        "raw_baseline_pre_rejection": {
            "macro_f1": oof_metrics_raw["macro_f1"],
            "accuracy": oof_metrics_raw["accuracy"],
            "balanced_accuracy": oof_metrics_raw["balanced_accuracy"],
            "support": oof_metrics_raw["support"],
            "per_class": oof_metrics_raw["per_class"],
            "confusion_matrix": oof_metrics_raw["confusion_matrix"],
        },
        "per_fold_summary": [
            {
                "fold": idx,
                "clean_macro_f1": per_fold_metrics_clean[idx]["macro_f1"],
                "clean_accuracy": per_fold_metrics_clean[idx]["accuracy"],
                "raw_macro_f1": per_fold_metrics_raw[idx]["macro_f1"],
                "raw_accuracy": per_fold_metrics_raw[idx]["accuracy"],
                "rejection_rate": per_fold_metrics_clean[idx]["rejection_rate"],
                "user_leakage": 0,
                "session_leakage": 0,
            }
            for idx in range(n_splits)
        ],
        "historical_comparison_notes": {
            "historical_macro_f1": 0.967276,
            "historical_is_leaky": True,
            "historical_reason": "Evaluated on synthetic chunks (tuana_chunk0..4) with 100% user/session leakage.",
            "direct_comparison_valid": False,
            "note": "Clean six-class baseline score must not be directly compared to leaky nine-class historical score.",
        },
    }

    metrics_path = os.path.join(output_dir, metrics_name)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)
    print(f"Metrics report saved to: {metrics_path}")

    # 8. Print summary report
    print("\n" + "=" * 70)
    print("           ACC-01 CLEAN SIX-CLASS BASELINE REPORT")
    print("=" * 70)
    print(f"Dataset:            {dataset_path}")
    print(f"Evaluator Hash:     {evaluator_source_hash[:16]}...")
    print(f"Samples Evaluated:  {n_evaluated_samples} ({len(target_classes)} classes)")
    print(f"Grouping:           {group_key} ({len(unique_groups)} groups, {n_splits} folds)")
    print(f"Split Fingerprint:  {split_fingerprint}")
    print(f"Parameters:         C={C}, gamma={gamma}, margin_threshold={margin_threshold}")
    print("-" * 70)
    print(f"CLEAN Macro-F1 (post-rejection):    {oof_metrics_clean['macro_f1']:.16f}")
    print(f"CLEAN Accuracy (post-rejection):    {oof_metrics_clean['accuracy']:.16f}")
    print(f"Balanced Accuracy:                 {oof_metrics_clean['balanced_accuracy']:.16f}")
    print(f"Coverage:                          {oof_metrics_clean['coverage']:.4f} ({oof_metrics_clean['rejection_rate']*100:.2f}% rejected)")
    print(f"Raw Macro-F1 (pre-rejection):      {oof_metrics_raw['macro_f1']:.16f}")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for cls in target_classes:
        pc = oof_metrics_clean["per_class"][cls]
        print(f"{cls:<20} {pc['precision']:<12.4f} {pc['recall']:<12.4f} {pc['f1']:<12.4f} {pc['support']:<10}")
    print("=" * 70 + "\n")

    return overall_results


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean fixed-seed evaluation CLI for AirPad baseline.")
    parser.add_argument("--dataset", default="data/hagrid_airpad.npz", help="Path to input .npz dataset")
    parser.add_argument("--group-key", default="user_id", help="Group key for cross-validation")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--sample-limit", type=int, default=None, help="Deterministic sample limit for smoke benchmark")
    parser.add_argument("--c", type=float, default=10.0, help="SVC C parameter")
    parser.add_argument("--gamma", default="0.01", help="SVC gamma parameter")
    parser.add_argument("--class-weight", default="balanced", help="SVC class_weight")
    parser.add_argument("--decision-threshold", type=float, default=0.0, help="Rejection decision threshold")
    parser.add_argument("--margin-threshold", type=float, default=0.15, help="Rejection margin threshold")
    parser.add_argument("--output-dir", default="reports/acc01/smoke", help="Output directory")
    parser.add_argument("--split-manifest-name", default="split_manifest.json", help="Split manifest filename")
    parser.add_argument("--predictions-name", default="predictions.jsonl", help="Predictions filename")
    parser.add_argument("--metrics-name", default="clean_baseline_metrics.json", help="Metrics report filename")
    parser.add_argument("--corrected-manifest", default="reports/acc01/corrected_group_manifest.json", help="Path to corrected group manifest")

    args = parser.parse_args(argv)

    try:
        gamma_val: float | str
        try:
            gamma_val = float(args.gamma)
        except ValueError:
            gamma_val = str(args.gamma)

        run_evaluation(
            dataset_path=args.dataset,
            group_key=args.group_key,
            n_splits=args.n_splits,
            seed=args.seed,
            sample_limit=args.sample_limit,
            C=args.c,
            gamma=gamma_val,
            class_weight=args.class_weight,
            decision_threshold=args.decision_threshold,
            margin_threshold=args.margin_threshold,
            output_dir=args.output_dir,
            split_manifest_name=args.split_manifest_name,
            predictions_name=args.predictions_name,
            metrics_name=args.metrics_name,
            corrected_manifest_path=args.corrected_manifest,
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
