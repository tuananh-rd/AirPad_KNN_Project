"""
Configuration loader and manager.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional
import yaml
import jsonschema

from src.config.schema import CONFIG_SCHEMA


_config_instance: Optional[Dict[str, Any]] = None
_config_lock = threading.Lock()


def validate_config_semantics(config_data: Dict[str, Any]) -> None:
    """Validate cross-field timing hierarchy and logical invariants."""
    gesture = config_data.get("gesture", {})
    click_min = gesture.get("click_min_ms")
    click_max = gesture.get("click_max_ms")
    drag_threshold = gesture.get("drag_threshold_ms")
    reacquire_stable_frames = gesture.get("reacquire_stable_frames", 1)

    if reacquire_stable_frames is not None and reacquire_stable_frames < 1:
        raise ValueError(f"reacquire_stable_frames must be >= 1, got {reacquire_stable_frames}")

    if click_min is not None and click_max is not None and drag_threshold is not None:
        if not (click_min <= click_max < drag_threshold):
            raise ValueError(
                f"Invalid gesture timing hierarchy: expected click_min_ms ({click_min}) "
                f"<= click_max_ms ({click_max}) < drag_threshold_ms ({drag_threshold})"
            )


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads default config and optionally merges with user config at path.
    Validates against schema and semantic timing invariants.
    """
    global _config_instance
    
    # Locate default.yaml
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_path = os.path.join(base_dir, "config", "default.yaml")
    
    with open(default_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_data = yaml.safe_load(f)
            # Simple merge at top level keys
            if user_data:
                for k, v in user_data.items():
                    if isinstance(v, dict) and k in config_data and isinstance(config_data[k], dict):
                        config_data[k].update(v)
                    else:
                        config_data[k] = v
                        
    # Validate against JSON schema
    jsonschema.validate(instance=config_data, schema=CONFIG_SCHEMA)
    
    # Validate semantic timing hierarchy and minimums
    validate_config_semantics(config_data)
    
    with _config_lock:
        _config_instance = config_data
        
    return config_data


def get_config() -> Dict[str, Any]:
    """
    Returns the loaded configuration. Thread-safe singleton.
    """
    global _config_instance
    
    with _config_lock:
        if _config_instance is None:
            # Need to release lock before loading to prevent deadlock if load_config uses lock
            pass
            
    if _config_instance is None:
        load_config()
        
    with _config_lock:
        if _config_instance is None:
            raise RuntimeError("Failed to load config")
        return _config_instance
