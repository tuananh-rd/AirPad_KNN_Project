import numpy as np
from typing import Dict, Any, Tuple, Generator, List, Optional

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Compute macro-F1, per-class precision/recall, confusion matrix.
    All from scratch using NumPy (no sklearn dependency)."""
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred))).tolist()
    
    n_classes = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
            
    per_class = {}
    f1_sum = 0.0
    valid_classes = 0
    
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        f1_sum += f1
        valid_classes += 1
        
    macro_f1 = f1_sum / valid_classes if valid_classes > 0 else 0.0
    
    return {
        'macro_f1': float(macro_f1),
        'per_class': per_class,
        'confusion_matrix': cm.tolist()
    }

def group_kfold_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """GroupKFold split - never splits consecutive frames from same session.
    groups: array of group IDs (user_id or session_id).
    Yields (train_idx, test_idx) tuples."""
    unique_groups = np.unique(groups)
    np.random.RandomState(42).shuffle(unique_groups) # Deterministic for consistent splits
    
    folds = np.array_split(unique_groups, n_splits)
    
    for fold_groups in folds:
        test_mask = np.isin(groups, fold_groups)
        train_mask = ~test_mask
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        yield train_idx, test_idx

def leave_one_subject_out(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """LOSO cross-validation. Yields (train_idx, test_idx) for each unique subject."""
    unique_subjects = np.unique(subjects)
    
    for subject in unique_subjects:
        test_mask = (subjects == subject)
        train_mask = ~test_mask
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        yield train_idx, test_idx
