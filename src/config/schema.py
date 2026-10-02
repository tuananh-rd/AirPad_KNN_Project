"""
JSON Schema for validating the AirPad configuration.
"""

CONFIG_SCHEMA = {
    "type": "object",
    "required": ["feature", "ml", "gesture", "platform", "safety"],
    "properties": {
        "feature": {
            "type": "object",
            "required": ["version", "scale_epsilon", "mirror_to_right", "normalize_rotation", "z_weight"],
            "properties": {
                "version": {"type": "integer"},
                "scale_epsilon": {"type": "number"},
                "mirror_to_right": {"type": "boolean"},
                "normalize_rotation": {"type": "boolean"},
                "z_weight": {"type": "number"}
            }
        },
        "ml": {
            "type": "object",
            "required": ["algorithm", "C", "gamma", "class_weight", "decision_threshold", "margin_threshold"],
            "properties": {
                "algorithm": {"type": "string", "enum": ["rbf_svm"]},
                "C": {"type": "number", "exclusiveMinimum": 0},
                "gamma": {"oneOf": [
                    {"type": "string", "enum": ["scale", "auto"]},
                    {"type": "number", "exclusiveMinimum": 0}
                ]},
                "class_weight": {"type": ["string", "null"], "enum": ["balanced", None]},
                "decision_threshold": {"type": "number"},
                "margin_threshold": {"type": "number"}
            }
        },
        "gesture": {
            "type": "object",
            "required": [
                "stability_window", "majority_ratio", "debounce_ms", "click_min_ms", 
                "click_max_ms", "drag_threshold_ms", "dwell_zoom_ms", "hysteresis_exit_count"
            ],
            "properties": {
                "stability_window": {"type": "integer", "minimum": 1},
                "majority_ratio": {"type": "number"},
                "debounce_ms": {"type": "integer", "minimum": 0},
                "click_min_ms": {"type": "integer", "minimum": 0},
                "click_max_ms": {"type": "integer", "minimum": 0},
                "drag_threshold_ms": {"type": "integer", "minimum": 1},
                "dwell_zoom_ms": {"type": "integer", "minimum": 1},
                "hysteresis_exit_count": {"type": "integer", "minimum": 1},
                "point_one_stable_ms": {"type": "integer", "minimum": 1},
                "enable_right_click": {"type": "boolean"},
                "enable_swipe": {"type": "boolean"},
                "reacquire_stable_frames": {"type": "integer", "minimum": 1}
            }
        },
        "motion": {
            "type": "object",
            "properties": {
                "roi_x_min": {"type": "number"},
                "roi_x_max": {"type": "number"},
                "roi_y_min": {"type": "number"},
                "roi_y_max": {"type": "number"},
                "mirror_x": {"type": "boolean"},
                "dead_zone": {"type": "number"},
                "gain_x": {"type": "number"},
                "gain_y": {"type": "number"},
                "filter_type": {"type": "string", "enum": ["none", "ema", "one_euro"]},
                "ema_alpha": {"type": "number"},
                "one_euro_min_cutoff": {"type": "number"},
                "one_euro_beta": {"type": "number"},
                "one_euro_d_cutoff": {"type": "number"},
                "scroll_dead_zone": {"type": "number"},
                "scroll_gain": {"type": "number"},
                "scroll_axis_lock_ratio": {"type": "number"},
                "zoom_step_threshold": {"type": "number"},
                "zoom_cooldown_ms": {"type": "number"},
                "swipe_min_displacement": {"type": "number"},
                "swipe_min_velocity": {"type": "number"},
                "swipe_cooldown_ms": {"type": "number"}
            }
        },
        "platform": {
            "type": "object",
            "required": ["dry_run", "rate_limit_hz"],
            "properties": {
                "dry_run": {"type": "boolean"},
                "rate_limit_hz": {"type": "integer", "minimum": 1}
            }
        },
        "safety": {
            "type": "object",
            "required": ["failsafe_corner", "max_events_per_second", "emergency_hotkey"],
            "properties": {
                "failsafe_corner": {"type": "boolean"},
                "max_events_per_second": {"type": "integer", "minimum": 1},
                "emergency_hotkey": {"type": "string"}
            }
        }
    }
}
