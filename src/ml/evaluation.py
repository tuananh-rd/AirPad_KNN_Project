import numpy as np
from typing import Dict, Any, Tuple, Generator, List, Optional, Union
from src.gesture.pose import HandPose


def is_unknown_prediction(
    prediction: Any,
    target_labels: Optional[List[Any]] = None,
    unknown_label: Any = "UNKNOWN",
) -> bool:
    """Determine whether a predicted label represents an UNKNOWN / rejected prediction.

    If the label is in target_labels, it is treated as a valid target class,
    not an unknown/rejection.
    """
    if target_labels is not None and prediction in target_labels:
        return False
    if prediction is None:
        return True
    if isinstance(prediction, str) and prediction.strip().upper() == "UNKNOWN":
        return True
    if hasattr(prediction, "name") and getattr(prediction, "name") == "UNKNOWN":
        return True
    if isinstance(prediction, HandPose) and prediction == HandPose.UNKNOWN:
        return True
    if unknown_label is not None and prediction == unknown_label:
        return True
    return False


def compute_metrics(
    y_true: Union[List[Any], np.ndarray],
    y_pred: Union[List[Any], np.ndarray],
    labels: Optional[List[Any]] = None,
    unknown_label: Any = "UNKNOWN",
) -> Dict[str, Any]:
    """Compute trustworthy classification metrics with proper UNKNOWN / rejection handling.

    - Computes macro-F1 strictly across the fixed target classes in `labels`.
    - UNKNOWN predictions are treated as false negatives / rejections in the full denominator.
    - An UNKNOWN prediction column is included in the confusion matrix when rejections occur,
      without introducing UNKNOWN as an extra target class in macro-F1.
    - Unexpected true or predicted labels raise ValueError clearly.
    - Provides accuracy, balanced accuracy, support, precision/recall/F1,
      coverage, rejection rate, and row/column labels for the confusion matrix.
    """
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    if len(y_true_list) != len(y_pred_list):
        raise ValueError(
            f"Length mismatch: {len(y_true_list)} ground truth labels vs {len(y_pred_list)} predictions"
        )

    if labels is None:
        # Infer target classes from ground truth (excluding UNKNOWN if present)
        inferred = []
        for t in y_true_list:
            if not is_unknown_prediction(t, None, unknown_label) and t not in inferred:
                inferred.append(t)
        if not inferred:
            # Fallback if y_true only contains unknown
            for t in y_true_list:
                if t not in inferred:
                    inferred.append(t)
        labels = inferred
    else:
        labels = list(labels)

    n_classes = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Validate ground truth labels
    for t in y_true_list:
        if t not in label_to_idx:
            raise ValueError(f"Unexpected ground truth label '{t}' not in target classes: {labels}")

    # Check for presence of UNKNOWN predictions
    has_unknown = any(is_unknown_prediction(p, labels, unknown_label) for p in y_pred_list)

    # Validate predicted labels
    for p in y_pred_list:
        if p not in label_to_idx and not is_unknown_prediction(p, labels, unknown_label):
            raise ValueError(
                f"Unexpected prediction label '{p}' not in target classes {labels} and not recognizable UNKNOWN"
            )

    # Construct confusion matrix
    # Rows: ground truth classes in `labels`
    # Columns: predicted classes in `labels`, plus UNKNOWN column if present
    n_cols = n_classes + (1 if has_unknown else 0)
    cm = np.zeros((n_classes, n_cols), dtype=int)

    for t, p in zip(y_true_list, y_pred_list):
        row_idx = label_to_idx[t]
        if p in label_to_idx:
            col_idx = label_to_idx[p]
            cm[row_idx, col_idx] += 1
        else:
            # Rejection / UNKNOWN column
            cm[row_idx, n_classes] += 1

    per_class = {}
    f1_sum = 0.0
    recalls_with_support = []
    total_correct = 0

    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        total_correct += tp

        # False positives: other classes predicted as class i
        fp = int(cm[:, i].sum() - tp)

        # False negatives: class i predicted as any other class OR rejected as UNKNOWN
        fn = int(cm[i, :].sum() - tp)
        support = tp + fn

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        }

        f1_sum += f1
        if support > 0:
            recalls_with_support.append(recall)

    macro_f1 = float(f1_sum / n_classes) if n_classes > 0 else 0.0
    balanced_acc = float(np.mean(recalls_with_support)) if recalls_with_support else 0.0

    total_samples = len(y_true_list)
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0

    rejected_count = int(cm[:, n_classes].sum()) if has_unknown else 0
    rejection_rate = float(rejected_count / total_samples) if total_samples > 0 else 0.0
    coverage = float(1.0 - rejection_rate)

    row_labels = [lbl.name if hasattr(lbl, "name") else lbl for lbl in labels]
    col_labels = list(row_labels)
    if has_unknown:
        col_labels.append("UNKNOWN")

    return {
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'support': total_samples,
        'coverage': coverage,
        'rejection_rate': rejection_rate,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'confusion_row_labels': row_labels,
        'confusion_col_labels': col_labels,
        'confusion_matrix_row_labels': row_labels,
        'confusion_matrix_col_labels': col_labels,
    }


def group_kfold_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """GroupKFold split - isolates groups between train and test folds with zero leakage.

    Validation rules:
    - Missing or empty group identifiers raise ValueError clearly.
    - Insufficient unique group count (< n_splits) raises ValueError clearly.
    - Deterministic split ordering controlled by seed.

    Yields (train_idx, test_idx) tuples.
    """
    if groups is None:
        raise ValueError("groups array cannot be None")

    n_samples = len(X)
    if len(groups) != n_samples:
        raise ValueError(
            f"Length mismatch: {n_samples} samples in X vs {len(groups)} group entries"
        )

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    # Validate no missing/empty/NaN groups
    for idx, g in enumerate(groups):
        if g is None:
            raise ValueError(f"Group identifier at sample index {idx} is None")
        if isinstance(g, str) and not g.strip():
            raise ValueError(f"Group identifier at sample index {idx} is empty or whitespace string")
        if isinstance(g, float) and np.isnan(g):
            raise ValueError(f"Group identifier at sample index {idx} is NaN")

    unique_groups = np.unique(groups)
    n_unique = len(unique_groups)

    if n_unique < n_splits:
        raise ValueError(
            f"Insufficient group count: found {n_unique} unique groups, "
            f"which is less than n_splits={n_splits}. Cannot create {n_splits} disjoint group folds."
        )

    # Deterministic shuffle of unique groups
    shuffled_groups = unique_groups.copy()
    np.random.RandomState(seed).shuffle(shuffled_groups)

    folds = np.array_split(shuffled_groups, n_splits)

    for fold_groups in folds:
        test_mask = np.isin(groups, fold_groups)
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        yield train_idx, test_idx


def leave_one_subject_out(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """LOSO cross-validation. Yields (train_idx, test_idx) for each unique subject.

    Validation rules:
    - Missing or empty subject identifiers raise ValueError.
    - Insufficient unique subject count (< 2) raises ValueError.
    """
    if subjects is None:
        raise ValueError("subjects array cannot be None")

    n_samples = len(X)
    if len(subjects) != n_samples:
        raise ValueError(
            f"Length mismatch: {n_samples} samples in X vs {len(subjects)} subject entries"
        )

    for idx, s in enumerate(subjects):
        if s is None:
            raise ValueError(f"Subject identifier at sample index {idx} is None")
        if isinstance(s, str) and not s.strip():
            raise ValueError(f"Subject identifier at sample index {idx} is empty or whitespace string")
        if isinstance(s, float) and np.isnan(s):
            raise ValueError(f"Subject identifier at sample index {idx} is NaN")

    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError(
            f"LOSO cross-validation requires at least 2 unique subjects, found {len(unique_subjects)}"
        )

    for subject in unique_subjects:
        test_mask = (subjects == subject)
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        yield train_idx, test_idx
