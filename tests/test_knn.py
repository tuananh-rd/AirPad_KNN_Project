"""Tests for src.ml.knn – ScratchKNN classifier.

Covers:
- L1, L2, cosine metrics
- Deterministic tie-breaking
- predict / predict_proba / predict_with_rejection
- Unknown rejection
- Edge cases (k > n_train, empty input, zero vectors for cosine)
- ScratchKNN vs sklearn fixture comparison
"""
from __future__ import annotations

import numpy as np
import pytest

from src.ml.knn import ScratchKNN


# ---------------------------------------------------------------------------
# Fixtures – small deterministic datasets
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_dataset():
    """3 classes, 4 samples each, well-separated clusters."""
    X = np.array([
        # Class 0 cluster near (0,0)
        [0.1, 0.1], [0.2, 0.0], [0.0, 0.2], [0.1, 0.15],
        # Class 1 cluster near (5,5)
        [5.0, 5.0], [5.1, 5.1], [4.9, 5.0], [5.0, 4.9],
        # Class 2 cluster near (10,0)
        [10.0, 0.0], [10.1, 0.1], [9.9, 0.0], [10.0, 0.1],
    ], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    return X, y


@pytest.fixture
def tie_dataset():
    """Dataset that creates deterministic tie scenarios."""
    # 2 classes, 2 samples each, query equidistant from one of each
    X = np.array([
        [0.0, 0.0],  # class 0
        [2.0, 0.0],  # class 1
        [0.0, 2.0],  # class 0
        [2.0, 2.0],  # class 1
    ], dtype=float)
    y = np.array([0, 1, 0, 1])
    return X, y


# ---------------------------------------------------------------------------
# Basic prediction tests
# ---------------------------------------------------------------------------
class TestPredictBasic:
    @pytest.mark.parametrize("metric", ["L1", "L2"])
    def test_predict_correct_clusters(self, simple_dataset, metric):
        X, y = simple_dataset
        knn = ScratchKNN(k=3, metric=metric).fit(X, y)

        queries = np.array([[0.15, 0.1], [5.05, 5.0], [10.0, 0.05]])
        preds = knn.predict(queries)
        np.testing.assert_array_equal(preds, [0, 1, 2])

    def test_predict_correct_clusters_cosine(self):
        """Cosine measures direction — use directionally separated clusters."""
        X = np.array([
            # Class 0: pointing right (+x direction)
            [5.0, 0.1], [4.0, 0.2], [6.0, 0.0],
            # Class 1: pointing up (+y direction)
            [0.1, 5.0], [0.2, 4.0], [0.0, 6.0],
            # Class 2: pointing diagonal
            [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],
        ], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        knn = ScratchKNN(k=3, metric='cosine').fit(X, y)

        queries = np.array([[10.0, 0.0], [0.0, 10.0], [7.0, 7.0]])
        preds = knn.predict(queries)
        np.testing.assert_array_equal(preds, [0, 1, 2])

    def test_predict_k_equals_1(self, simple_dataset):
        X, y = simple_dataset
        knn = ScratchKNN(k=1, metric='L2').fit(X, y)
        preds = knn.predict(X)
        np.testing.assert_array_equal(preds, y)  # perfect recall with k=1

    def test_predict_empty_query(self, simple_dataset):
        X, y = simple_dataset
        knn = ScratchKNN(k=3).fit(X, y)
        preds = knn.predict(np.empty((0, 2)))
        assert len(preds) == 0


# ---------------------------------------------------------------------------
# Confidence and margin
# ---------------------------------------------------------------------------
class TestPredictProba:
    def test_confidence_is_ratio(self, simple_dataset):
        X, y = simple_dataset
        knn = ScratchKNN(k=3, metric='L2').fit(X, y)
        query = np.array([[0.1, 0.1]])  # clearly class 0
        preds, confs, margins = knn.predict_proba(query)
        assert preds[0] == 0
        assert confs[0] == 1.0  # all 3 neighbors are class 0
        assert margins[0] == 1.0

    def test_low_confidence_at_boundary(self):
        # Place query equidistant between two classes
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, -1]])
        y = np.array([0, 1, 0, 1, 0])
        knn = ScratchKNN(k=5, metric='L2').fit(X, y)
        query = np.array([[0.5, 0.5]])
        _, confs, margins = knn.predict_proba(query)
        assert confs[0] < 1.0  # not fully confident


# ---------------------------------------------------------------------------
# Unknown rejection
# ---------------------------------------------------------------------------
class TestRejection:
    def test_rejects_low_confidence(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, -1]])
        y = np.array([0, 1, 0, 1, 0])
        knn = ScratchKNN(k=5, metric='L2').fit(X, y)
        query = np.array([[0.5, 0.5]])

        # Very high threshold should reject
        result = knn.predict_with_rejection(query, confidence_threshold=0.99,
                                             margin_threshold=0.0, unknown_label=-1)
        assert result[0] == -1

    def test_accepts_high_confidence(self, simple_dataset):
        X, y = simple_dataset
        knn = ScratchKNN(k=3).fit(X, y)
        query = np.array([[0.1, 0.1]])
        result = knn.predict_with_rejection(query, confidence_threshold=0.5,
                                             margin_threshold=0.1)
        assert result[0] == 0  # should not reject

    def test_custom_unknown_label(self):
        """Custom unknown label should be returned on rejection."""
        # Use boundary data where query is ambiguous
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, -1]])
        y = np.array([0, 1, 0, 1, 0])
        knn = ScratchKNN(k=5).fit(X, y)
        query = np.array([[0.5, 0.5]])
        result = knn.predict_with_rejection(query, confidence_threshold=0.99,
                                             margin_threshold=0.99, unknown_label=999)
        assert result[0] == 999


# ---------------------------------------------------------------------------
# Deterministic tie-breaking
# ---------------------------------------------------------------------------
class TestTieBreaking:
    def test_deterministic_same_votes(self, tie_dataset):
        """When two classes tie in votes, pick by smallest distance sum, then lowest label."""
        X, y = tie_dataset
        knn = ScratchKNN(k=4, metric='L2').fit(X, y)

        # Query at (1, 1) — equidistant from all 4 points
        # class 0: points (0,0) and (0,2), distances = sqrt(2), sqrt(2), sum = 2*sqrt(2)
        # class 1: points (2,0) and (2,2), distances = sqrt(2), sqrt(2), sum = 2*sqrt(2)
        # Both tie in votes AND distance → should pick label 0 (lowest index)
        query = np.array([[1.0, 1.0]])
        preds = knn.predict(query)
        assert preds[0] == 0, "Tie should break to lowest label index"

    def test_tie_break_by_distance(self):
        """When votes tie, closer class wins."""
        X = np.array([
            [0.0, 0.0],  # class 0
            [0.5, 0.0],  # class 1
        ])
        y = np.array([0, 1])
        knn = ScratchKNN(k=2, metric='L2').fit(X, y)

        # Query at (0.1, 0) → closer to class 0
        query = np.array([[0.1, 0.0]])
        preds = knn.predict(query)
        assert preds[0] == 0, "Tie should break to closer class"

    def test_determinism_repeated(self, tie_dataset):
        """Same query produces same result every time."""
        X, y = tie_dataset
        knn = ScratchKNN(k=4, metric='L2').fit(X, y)
        query = np.array([[1.0, 1.0]])

        results = [knn.predict(query)[0] for _ in range(100)]
        assert len(set(results)) == 1, "Predictions should be deterministic"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_k_greater_than_n_train(self):
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        knn = ScratchKNN(k=10, metric='L2').fit(X, y)
        preds = knn.predict(np.array([[0.1, 0.1]]))
        assert preds[0] in [0, 1]  # should still work

    def test_fit_rejects_empty(self):
        with pytest.raises(ValueError):
            ScratchKNN().fit(np.empty((0, 2)), np.array([]))

    def test_fit_rejects_mismatch(self):
        with pytest.raises(ValueError):
            ScratchKNN().fit(np.ones((3, 2)), np.array([0, 1]))

    def test_cosine_with_zero_vectors(self):
        """Zero vectors should not crash cosine metric."""
        X = np.array([[1, 0], [0, 1]])
        y = np.array([0, 1])
        knn = ScratchKNN(k=2, metric='cosine').fit(X, y)
        query = np.array([[0.0, 0.0]])  # zero vector
        preds = knn.predict(query)
        assert np.isfinite(preds[0])


# ---------------------------------------------------------------------------
# ScratchKNN vs sklearn comparison
# ---------------------------------------------------------------------------
class TestSklearnComparison:
    """Compare ScratchKNN against sklearn on same data. Skip if sklearn unavailable."""

    @pytest.fixture
    def sklearn_available(self):
        pytest.importorskip("sklearn")

    def test_predictions_match_l2(self, sklearn_available, simple_dataset):
        from sklearn.neighbors import KNeighborsClassifier

        X, y = simple_dataset
        k = 3
        queries = np.array([[0.15, 0.1], [5.05, 5.0], [10.0, 0.05]])

        scratch = ScratchKNN(k=k, metric='L2').fit(X, y)
        sk = KNeighborsClassifier(n_neighbors=k, metric='euclidean').fit(X, y)

        scratch_preds = scratch.predict(queries)
        sk_preds = sk.predict(queries)
        np.testing.assert_array_equal(scratch_preds, sk_preds)

    def test_predictions_match_l1(self, sklearn_available, simple_dataset):
        from sklearn.neighbors import KNeighborsClassifier

        X, y = simple_dataset
        k = 3
        queries = np.array([[0.15, 0.1], [5.05, 5.0], [10.0, 0.05]])

        scratch = ScratchKNN(k=k, metric='L1').fit(X, y)
        sk = KNeighborsClassifier(n_neighbors=k, metric='manhattan').fit(X, y)

        scratch_preds = scratch.predict(queries)
        sk_preds = sk.predict(queries)
        np.testing.assert_array_equal(scratch_preds, sk_preds)
