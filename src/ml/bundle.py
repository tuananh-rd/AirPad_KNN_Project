import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

@dataclass
class ModelBundle:
    feature_version: int
    labels: List[str]  # class names
    algorithm: str
    hyperparameters: Dict[str, Any]
    train_samples: int
    created_at: str  # ISO timestamp
    metadata: Dict[str, Any]  # arbitrary extra info
    model_data: Dict[str, Any]  # Raw X_train and y_train lists for model reconstruction
    
    def save(self, path: str) -> None:
        """Save the bundle as a JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'ModelBundle':
        """Load the bundle from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "algorithm" not in data:
            raise ValueError("Legacy KNN bundles are not supported; retrain an RBF-SVM bundle")
        return cls(**data)
