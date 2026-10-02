import time
from typing import Dict, Any

try:
    from sklearn.neighbors import KNeighborsClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def create_sklearn_knn(k: int = 5, metric: str = 'minkowski', p: int = 2, algorithm: str = 'auto') -> Any:
    """Create sklearn KNN for benchmarking."""
    if not HAS_SKLEARN:
        raise ImportError('scikit-learn required for benchmark')
    return KNeighborsClassifier(n_neighbors=k, metric=metric, p=p, algorithm=algorithm)

def benchmark_algorithms(X_train: Any, y_train: Any, X_test: Any, y_test: Any, k: int = 5) -> Dict[str, Dict[str, float]]:
    """Benchmark brute/kd_tree/ball_tree. Returns dict of {algorithm: {accuracy, time_fit, time_predict}}.
    Does NOT claim any is faster; just reports measurements."""
    if not HAS_SKLEARN:
        raise ImportError('scikit-learn required for benchmark')
    
    results = {}
    for algo in ['brute', 'kd_tree', 'ball_tree']:
        model = create_sklearn_knn(k=k, algorithm=algo)
        
        t0 = time.time()
        model.fit(X_train, y_train)
        t_fit = time.time() - t0
        
        t0 = time.time()
        preds = model.predict(X_test)
        t_predict = time.time() - t0
        
        acc = float((preds == y_test).mean())
        
        results[algo] = {
            'accuracy': acc,
            'time_fit': t_fit,
            'time_predict': t_predict
        }
    return results
