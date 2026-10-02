"""RBF-SVM training, grouped cross-validation, and model bundle export.

Ensures strict zero-leakage training protocol:
- Grouped cross-validation (GroupKFold / LOSO) ensures no frame session leakage.
- Scaler / normalizer is fit on train fold ONLY, then applied to test fold.
- RBF-SVM evaluation using macro-F1, per-class metrics, and confusion matrix.
- Minimal hyperparameter tuning across (C, gamma) combinations.
- Self-contained ModelBundle serialization for production deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from src.gesture.pose import HandPose
from src.ml.bundle import ModelBundle
from src.ml.evaluation import compute_metrics, group_kfold_split, leave_one_subject_out
from src.ml.svm import SVMClassifier
from sklearn.preprocessing import StandardScaler
from tools.collect_data import LABEL_NAMES, load_dataset_npz


class SerializableStandardScaler(StandardScaler):
    """scikit-learn StandardScaler with JSON serialization helpers."""

    def to_dict(self) -> Dict[str, Any]:
        if not hasattr(self, "mean_") or not hasattr(self, "scale_"):
            raise RuntimeError("FeatureScaler must be fit before exporting to dict.")
        return {
            "type": "standard",
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializableStandardScaler":
        scaler = cls()
        scaler.mean_ = np.asarray(data["mean"], dtype=float)
        scaler.scale_ = np.asarray(data["scale"], dtype=float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        scaler.n_samples_seen_ = 1
        return scaler


# Kept as a source-compatible import for existing data tooling.
FeatureScaler = SerializableStandardScaler


def validate_dataset(
    features: np.ndarray,
    labels: Union[List[str], np.ndarray],
    metadata: List[Dict[str, Any]],
    raise_on_error: bool = True,
) -> bool:
    """Validate dataset integrity prior to training.

    Checks:
    - features has shape (N, 63)
    - no NaNs or infinite values in features
    - labels length matches features
    - all labels are valid members of LABEL_NAMES
    - metadata length matches features
    - all metadata entries have feature_version == 1

    Raises:
        ValueError: If validation fails and raise_on_error is True.

    Returns:
        True if valid, False if invalid and raise_on_error is False.
    """
    try:
        if not isinstance(features, np.ndarray):
            raise ValueError(f"Features must be a numpy.ndarray, got {type(features)}")

        if len(features) == 0:
            raise ValueError("Dataset features array is empty")

        if features.ndim != 2 or features.shape[1] != 63:
            raise ValueError(f"Features array must have shape (N, 63), got {features.shape}")

        if np.any(np.isnan(features)):
            raise ValueError("Features array contains NaN values")

        if np.any(np.isinf(features)):
            raise ValueError("Features array contains infinite values")

        n_samples = len(features)
        if len(labels) != n_samples:
            raise ValueError(f"Length mismatch: {n_samples} features vs {len(labels)} labels")

        if len(metadata) != n_samples:
            raise ValueError(f"Length mismatch: {n_samples} features vs {len(metadata)} metadata entries")

        for idx, lbl in enumerate(labels):
            lbl_str = str(lbl)
            if lbl_str not in LABEL_NAMES:
                raise ValueError(f"Sample {idx} has invalid label '{lbl_str}' not in LABEL_NAMES")

        for idx, meta in enumerate(metadata):
            if not isinstance(meta, dict):
                raise ValueError(f"Metadata entry {idx} is not a dict")
            version = meta.get("feature_version")
            if version != 1:
                raise ValueError(
                    f"Metadata entry {idx} has invalid feature_version: {version} (expected 1)"
                )
        return True

    except ValueError as e:
        if raise_on_error:
            raise
        return False


def extract_groups(
    metadata: List[Dict[str, Any]],
    group_key: str = "session_id",
) -> np.ndarray:
    """Extract grouping IDs (e.g. session_id or user_id) from sample metadata."""
    return np.array([m.get(group_key, f"unknown_{i}") for i, m in enumerate(metadata)])


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 10.0,
    gamma: Union[str, float] = "scale",
    use_scaler: bool = True,
) -> Dict[str, Any]:
    """Train an RBF-SVM on one fold with no scaler leakage.

    Scaler is strictly fitted on X_train only and used to transform X_test.
    """
    if not use_scaler:
        raise ValueError("RBF-SVM evaluation requires StandardScaler")
    classifier = SVMClassifier(C=C, gamma=gamma).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    return metrics


def tune_svm(
    X: np.ndarray,
    y: Union[List[str], np.ndarray],
    groups: np.ndarray,
    C_values: List[float] = [1.0, 10.0],
    gamma_values: List[Union[str, float]] = ["scale", 0.01],
    n_splits: int = 5,
    use_scaler: bool = True,
) -> Dict[str, Any]:
    """Evaluate a small (C, gamma) grid using grouped CV.

    Ensures zero group overlap between train and test folds.
    Returns best parameters and complete measurements table.
    """
    y_arr = np.asarray(y)
    trainable = ~np.isin(
        y_arr.astype(str),
        ["NO_HAND", "UNKNOWN", str(int(HandPose.NO_HAND)), str(int(HandPose.UNKNOWN))],
    )
    X, y_arr, groups = X[trainable], y_arr[trainable], groups[trainable]
    unique_groups = len(np.unique(groups))
    effective_splits = min(n_splits, unique_groups)
    if effective_splits < 2:
        effective_splits = 2

    measurements: List[Dict[str, Any]] = []
    best_score = -1.0
    best_params = {"C": C_values[0], "gamma": gamma_values[0]}

    for gamma in gamma_values:
        for C in C_values:
            fold_f1_scores: List[float] = []

            for train_idx, test_idx in group_kfold_split(X, y_arr, groups, n_splits=effective_splits):
                if len(test_idx) == 0:
                    continue

                X_tr, y_tr = X[train_idx], y_arr[train_idx]
                X_te, y_te = X[test_idx], y_arr[test_idx]

                # Check if train fold has samples
                if len(X_tr) == 0:
                    continue

                fold_result = evaluate_fold(
                    X_tr, y_tr, X_te, y_te,
                    C=C,
                    gamma=gamma,
                    use_scaler=use_scaler,
                )
                fold_f1_scores.append(fold_result["macro_f1"])

            mean_f1 = float(np.mean(fold_f1_scores)) if fold_f1_scores else 0.0
            std_f1 = float(np.std(fold_f1_scores)) if fold_f1_scores else 0.0

            measurement = {
                "C": C,
                "gamma": gamma,
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "fold_scores": fold_f1_scores,
            }
            measurements.append(measurement)

            if mean_f1 > best_score:
                best_score = mean_f1
                best_params = {"C": C, "gamma": gamma}

    return {
        "best_params": best_params,
        "best_score": best_score,
        "measurements": measurements,
    }


def create_model_bundle(
    X: np.ndarray,
    y: Union[List[str], np.ndarray],
    C: float = 10.0,
    gamma: Union[str, float] = "scale",
    class_weight: Optional[str] = "balanced",
    labels: Optional[List[str]] = None,
    validation_metrics: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> ModelBundle:
    """Create a serialized ModelBundle for deployment.

    Args:
        X: Training features array of shape (N, 63).
        y: Target label array of length N.
        C: SVC regularization parameter.
        gamma: RBF kernel coefficient.
        labels: List of class names (defaults to sorted unique labels in y).
        validation_metrics: Validation metrics summary (macro-F1, etc.).
        extra_metadata: Additional arbitrary metadata.

    Returns:
        ModelBundle instance ready for serialization.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=str)
    keep = ~np.isin(
        y_arr,
        ["NO_HAND", "UNKNOWN", str(int(HandPose.NO_HAND)), str(int(HandPose.UNKNOWN))],
    )
    X_arr, y_arr = X_arr[keep], y_arr[keep]
    if len(np.unique(y_arr)) < 2:
        raise ValueError("Bundle requires at least two trainable pose classes")

    if labels is None:
        labels_list = sorted(list(set(y_arr)))
    else:
        labels_list = [
            str(lbl) for lbl in labels
            if str(lbl) not in {"NO_HAND", "UNKNOWN", str(int(HandPose.NO_HAND)), str(int(HandPose.UNKNOWN))}
        ]

    # Always fit on the filtered deployment training set. Accepting an external
    # scaler here would make it possible to leak held-out/runtime-only samples.
    fitted_scaler = FeatureScaler().fit(X_arr)
    scaler_data = {
        "type": "standard",
        "mean": fitted_scaler.mean_.tolist(),
        "scale": fitted_scaler.scale_.tolist(),
    }

    metadata: Dict[str, Any] = {
        "scaler": scaler_data,
        "validation_metrics": validation_metrics,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return ModelBundle(
        feature_version=1,
        labels=labels_list,
        algorithm="rbf_svm",
        hyperparameters={"C": float(C), "gamma": gamma, "class_weight": class_weight},
        train_samples=len(X_arr),
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata=metadata,
        model_data={
            "X_train": X_arr.tolist(),
            "y_train": y_arr.tolist(),
        },
    )


def save_bundle(bundle: ModelBundle, filepath: str) -> None:
    """Serialize and save ModelBundle to disk."""
    parent_dir = os.path.dirname(os.path.abspath(filepath))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    bundle.save(filepath)


def load_bundle(filepath: str) -> ModelBundle:
    """Load ModelBundle from disk."""
    return ModelBundle.load(filepath)


def load_model_from_bundle(
    bundle_or_path: Union[ModelBundle, str],
) -> Tuple[SVMClassifier, FeatureScaler, ModelBundle]:
    """Reconstitute a fitted RBF-SVM (including its StandardScaler) from a bundle.

    Returns:
        (classifier, scaler, bundle)
    """
    if isinstance(bundle_or_path, str):
        bundle = load_bundle(bundle_or_path)
    else:
        bundle = bundle_or_path

    scaler_info = bundle.metadata.get("scaler")
    if scaler_info is None:
        raise ValueError("RBF-SVM bundle is missing its required StandardScaler")
    scaler = FeatureScaler.from_dict(scaler_info)

    if bundle.algorithm != "rbf_svm":
        raise ValueError(f"Unsupported production classifier: {bundle.algorithm}")
    params = bundle.hyperparameters
    classifier = SVMClassifier(
        C=params.get("C", 10.0),
        gamma=params.get("gamma", "scale"),
        class_weight=params.get("class_weight", "balanced"),
    )
    X_train = np.asarray(bundle.model_data["X_train"], dtype=float)
    y_train = np.asarray(bundle.model_data["y_train"], dtype=str)
    classifier._fit(X_train, y_train, scaler=scaler)

    return classifier, scaler, bundle


def train_from_dataset(
    dataset_path: str,
    output_bundle_path: str,
    group_key: str = "user_id",
    c_values: Optional[List[float]] = None,
    gamma_values: Optional[List[Union[str, float]]] = None,
    n_splits: int = 5,
    summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train RBF-SVM model bundle end-to-end from an .npz dataset.

    Workflow:
    1. Load .npz dataset and validate integrity.
    2. Filter out runtime-only classes (UNKNOWN, NO_HAND).
    3. Extract grouping IDs based on group_key.
    4. Tune C and gamma using zero-leakage grouped cross-validation.
    5. Evaluate out-of-fold metrics (macro-F1, per-class metrics, confusion matrix).
    6. Fit final model on all trainable samples.
    7. Save ModelBundle to output_bundle_path.
    8. Return summary dictionary and optionally write summary JSON.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    features, labels, metadata = load_dataset_npz(dataset_path)
    validate_dataset(features, labels, metadata, raise_on_error=True)

    y_arr = np.asarray(labels, dtype=str)
    trainable_mask = ~np.isin(
        y_arr,
        ["NO_HAND", "UNKNOWN", str(int(HandPose.NO_HAND)), str(int(HandPose.UNKNOWN))],
    )
    X_trainable = features[trainable_mask]
    y_trainable = y_arr[trainable_mask]
    meta_trainable = [m for idx, m in enumerate(metadata) if trainable_mask[idx]]

    if len(X_trainable) == 0:
        raise ValueError(f"No trainable samples found in dataset '{dataset_path}'")
    unique_classes = sorted(list(set(y_trainable)))
    if len(unique_classes) < 2:
        raise ValueError(
            f"Need at least 2 distinct trainable classes, found: {unique_classes}"
        )

    groups = extract_groups(meta_trainable, group_key=group_key)
    unique_groups = np.unique(groups)
    effective_group_key = group_key
    if len(unique_groups) < 2 and group_key == "user_id":
        alt_groups = extract_groups(meta_trainable, group_key="session_id")
        if len(np.unique(alt_groups)) >= 2:
            print(
                f"[WARN] Only 1 user_id found. Falling back to session_id grouping ({len(np.unique(alt_groups))} sessions)."
            )
            groups = alt_groups
            unique_groups = np.unique(groups)
            effective_group_key = "session_id"

    c_vals = c_values if c_values is not None else [1.0, 10.0]
    gamma_vals = gamma_values if gamma_values is not None else ["scale", 0.01]

    tuning_res = tune_svm(
        X=X_trainable,
        y=y_trainable,
        groups=groups,
        C_values=c_vals,
        gamma_values=gamma_vals,
        n_splits=n_splits,
        use_scaler=True,
    )
    best_params = tuning_res["best_params"]
    best_C = float(best_params["C"])
    best_gamma = best_params["gamma"]

    # Evaluate out-of-fold metrics with best parameters
    effective_splits = max(2, min(n_splits, len(unique_groups)))
    oof_preds = np.empty_like(y_trainable)
    oof_mask = np.zeros(len(y_trainable), dtype=bool)

    for tr_idx, te_idx in group_kfold_split(X_trainable, y_trainable, groups, n_splits=effective_splits):
        if len(te_idx) == 0 or len(tr_idx) == 0:
            continue
        X_tr, y_tr = X_trainable[tr_idx], y_trainable[tr_idx]
        X_te = X_trainable[te_idx]
        clf = SVMClassifier(C=best_C, gamma=best_gamma).fit(X_tr, y_tr)
        oof_preds[te_idx] = clf.predict(X_te)
        oof_mask[te_idx] = True

    if np.any(oof_mask):
        metrics = compute_metrics(y_trainable[oof_mask], oof_preds[oof_mask], labels=unique_classes)
    else:
        metrics = {
            "macro_f1": float(tuning_res["best_score"]),
            "per_class": {},
            "confusion_matrix": [],
        }

    bundle = create_model_bundle(
        X=X_trainable,
        y=y_trainable,
        C=best_C,
        gamma=best_gamma,
        labels=unique_classes,
        validation_metrics=metrics,
        extra_metadata={
            "group_key": effective_group_key,
            "unique_groups_count": int(len(unique_groups)),
            "tuning_measurements": tuning_res["measurements"],
        },
    )
    save_bundle(bundle, output_bundle_path)

    summary = {
        "dataset_path": dataset_path,
        "output_bundle_path": output_bundle_path,
        "n_samples": int(len(X_trainable)),
        "classes": unique_classes,
        "group_key": effective_group_key,
        "n_groups": int(len(unique_groups)),
        "best_params": best_params,
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "per_class": metrics.get("per_class", {}),
        "confusion_matrix": metrics.get("confusion_matrix", []),
    }

    if summary_path:
        parent_dir = os.path.dirname(os.path.abspath(summary_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def print_training_report(summary: Dict[str, Any]) -> None:
    """Format and print an executive training report to stdout."""
    print("\n" + "=" * 65)
    print("           AIRPAD RBF-SVM TRAINING REPORT")
    print("=" * 65)
    print(f"Dataset:      {summary['dataset_path']}")
    print(f"Samples:      {summary['n_samples']} ({len(summary['classes'])} classes)")
    print(f"Groups:       {summary['n_groups']} (grouped by {summary['group_key']})")
    print(f"Best Params:  C={summary['best_params']['C']}, gamma={summary['best_params']['gamma']}")
    print(f"Macro-F1:     {summary['macro_f1']:.4f}")
    print("-" * 65)
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 65)
    per_class = summary.get("per_class", {})
    for cls_name in summary["classes"]:
        m = per_class.get(cls_name, {})
        p = m.get("precision", 0.0)
        r = m.get("recall", 0.0)
        f1 = m.get("f1", 0.0)
        print(f"{cls_name:<25} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
    print("-" * 65)
    print(f"Model saved:  {summary['output_bundle_path']}")
    print("=" * 65 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train and export AirPad RBF-SVM model bundle from an .npz dataset."
    )
    parser.add_argument("dataset_path", help="Path to input .npz dataset file")
    parser.add_argument("output_bundle_path", help="Path to output .json model bundle file")
    parser.add_argument(
        "--group-key",
        default="user_id",
        help="Metadata key for grouped cross-validation (default: user_id)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[1.0, 10.0],
        help="C grid values to tune (default: 1.0 10.0)",
    )
    parser.add_argument(
        "--gamma-values",
        nargs="+",
        default=["scale", "0.01"],
        help="gamma grid values to tune (default: scale 0.01)",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path to write summary JSON report",
    )

    args = parser.parse_args(argv)

    # Parse gamma values to handle float conversion where applicable
    parsed_gamma: List[Union[str, float]] = []
    for g in args.gamma_values:
        try:
            parsed_gamma.append(float(g))
        except ValueError:
            parsed_gamma.append(str(g))

    try:
        summary = train_from_dataset(
            dataset_path=args.dataset_path,
            output_bundle_path=args.output_bundle_path,
            group_key=args.group_key,
            c_values=args.c_values,
            gamma_values=parsed_gamma,
            n_splits=args.n_splits,
            summary_path=args.summary_path,
        )
        print_training_report(summary)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
