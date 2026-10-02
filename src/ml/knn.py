import numpy as np
from typing import Any, Tuple

class ScratchKNN:
    """K-Nearest Neighbors classifier from scratch using NumPy.
    
    Supports L1 (Manhattan), L2 (Euclidean), and cosine distance metrics.
    Deterministic tie-breaking by smallest distance sum, then lowest label index.
    """
    
    def __init__(self, k: int = 5, metric: str = 'L2'):
        self.k = k
        self.metric = metric.upper()
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ScratchKNN':
        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
            
        self.X_train = np.asarray(X, dtype=float).copy()
        y_arr = np.asarray(y)
        self.classes_, self.y_train = np.unique(y_arr, return_inverse=True)
        
        if self.metric == 'COSINE':
            norms = np.linalg.norm(self.X_train, axis=1, keepdims=True)
            self.X_train = self.X_train / (norms + 1e-10)
            
        return self
        
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0, len(self.X_train)))
            
        if self.metric == 'L1':
            return np.sum(np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=-1)
        elif self.metric == 'L2':
            return np.sqrt(np.sum((X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])**2, axis=-1))
        elif self.metric == 'COSINE':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_norm = X / (norms + 1e-10)
            dot = np.dot(X_norm, self.X_train.T)
            return 1.0 - dot
        else:
            raise ValueError(f"Unknown metric {self.metric}")
            
    def _get_top_k(self, distances: np.ndarray) -> np.ndarray:
        n_query, n_train = distances.shape
        k = min(self.k, n_train)
        if k == 0:
            return np.zeros((n_query, 0), dtype=int)
            
        if k == n_train:
            top_k_idx_unsorted = np.tile(np.arange(n_train), (n_query, 1))
        else:
            top_k_idx_unsorted = np.argpartition(distances, k - 1, axis=1)[:, :k]
            
        top_k_distances_unsorted = np.take_along_axis(distances, top_k_idx_unsorted, axis=1)
        sort_idx = np.argsort(top_k_distances_unsorted, axis=1)
        top_k_idx = np.take_along_axis(top_k_idx_unsorted, sort_idx, axis=1)
        
        return top_k_idx
        
    def _predict_details(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        if len(X_arr) == 0:
            return np.array([]), np.array([]), np.array([])
            
        distances = self._compute_distances(X_arr)
        top_k_idx = self._get_top_k(distances)
        
        n_query = X_arr.shape[0]
        k = top_k_idx.shape[1]
        
        if k == 0:
            return np.zeros(n_query, dtype=self.classes_.dtype), np.zeros(n_query), np.zeros(n_query)

        top_k_labels = self.y_train[top_k_idx]
        top_k_distances = np.take_along_axis(distances, top_k_idx, axis=1)
        
        n_classes = len(self.classes_)
        
        counts = np.zeros((n_query, n_classes), dtype=int)
        dist_sums = np.zeros((n_query, n_classes), dtype=float)
        
        for i in range(k):
            labels_i = top_k_labels[:, i]
            dist_i = top_k_distances[:, i]
            np.add.at(counts, (np.arange(n_query), labels_i), 1)
            np.add.at(dist_sums, (np.arange(n_query), labels_i), dist_i)
            
        dt = np.dtype([('neg_count', int), ('dist_sum', float), ('label_idx', int)])
        structured = np.empty(counts.shape, dtype=dt)
        structured['neg_count'] = -counts
        structured['dist_sum'] = dist_sums
        structured['label_idx'] = np.arange(n_classes)
        
        sorted_indices = np.argsort(structured, axis=1)
        best_label_indices = sorted_indices[:, 0]
        predictions = self.classes_[best_label_indices]
        
        sorted_counts = np.take_along_axis(counts, sorted_indices, axis=1)
        top_count = sorted_counts[:, 0]
        second_count = sorted_counts[:, 1] if n_classes > 1 else np.zeros_like(top_count)
        
        confidences = top_count / k
        margins = (top_count - second_count) / k
        
        return predictions, confidences, margins

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds, _, _ = self._predict_details(X)
        return preds
        
    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_details(X)
        
    def predict_with_rejection(self, X: np.ndarray, 
                                confidence_threshold: float = 0.55,
                                margin_threshold: float = 0.15,
                                unknown_label: Any = -1) -> np.ndarray:
        preds, confs, margins = self._predict_details(X)
        if len(preds) == 0:
            return preds
            
        reject = (confs < confidence_threshold) | (margins < margin_threshold)
        return np.where(reject, unknown_label, preds)
