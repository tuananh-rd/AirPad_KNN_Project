"""Data collection and dataset management utilities for the AirPad pose classifier.

Provides functionality for recording, validating, and persisting hand landmark
feature datasets (.npz and .csv) along with full experimental metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np

from src.features.canonicalize import canonicalize
from src.gesture.pose import HandPose

# Valid pose labels recognized by the system
LABEL_NAMES: List[str] = [p.name for p in HandPose]

# Valid domain values for metadata fields
VALID_HANDEDNESS = {"Left", "Right"}
VALID_LIGHTING = {"bright", "dim", "mixed"}

# Required metadata keys that must be present in every sample
REQUIRED_METADATA_KEYS = {
    "user_id",
    "session_id",
    "timestamp",
    "handedness",
    "distance_cm",
    "lighting",
    "label",
    "is_transition",
}

# Optional keys with default values
OPTIONAL_METADATA_KEYS = {
    "feature_version": 1,
}

ALL_METADATA_KEYS = REQUIRED_METADATA_KEYS | set(OPTIONAL_METADATA_KEYS.keys())


def create_sample_metadata(
    user_id: str,
    session_id: str,
    timestamp: float,
    handedness: str,
    distance_cm: float,
    lighting: str,
    label: str,
    is_transition: bool = False,
    feature_version: int = 1,
) -> Dict[str, Any]:
    """Helper to construct a validated metadata dictionary for a single sample."""
    return {
        "user_id": str(user_id),
        "session_id": str(session_id),
        "timestamp": float(timestamp),
        "handedness": str(handedness),
        "distance_cm": float(distance_cm),
        "lighting": str(lighting),
        "label": str(label),
        "is_transition": bool(is_transition),
        "feature_version": int(feature_version),
    }


def validate_sample(features: np.ndarray, metadata: Dict[str, Any]) -> bool:
    """Validate a single feature vector and its associated metadata.

    Invariants checked:
    - features is a numpy array with shape (63,)
    - features contains no NaNs or infinite values
    - metadata is a dictionary containing all required schema keys:
      user_id, session_id, timestamp, handedness, distance_cm, lighting, label, is_transition
    - metadata['label'] is an accepted pose in LABEL_NAMES
    - metadata['handedness'] in ('Left', 'Right')
    - metadata['lighting'] in ('bright', 'dim', 'mixed')
    - metadata['is_transition'] is a boolean
    - metadata['feature_version'] == 1 (if present; defaults to 1)

    Returns:
        True if the sample passes all checks, False otherwise.
    """
    if not isinstance(features, np.ndarray):
        return False
    if features.shape != (63,):
        return False
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return False

    if not isinstance(metadata, dict):
        return False

    # Check required keys presence
    if not REQUIRED_METADATA_KEYS.issubset(metadata.keys()):
        return False

    # Check types and value domains
    if not isinstance(metadata["user_id"], str) or not metadata["user_id"].strip():
        return False
    if not isinstance(metadata["session_id"], str) or not metadata["session_id"].strip():
        return False
    if not isinstance(metadata["timestamp"], (int, float)):
        return False
    if metadata["handedness"] not in VALID_HANDEDNESS:
        return False
    if not isinstance(metadata["distance_cm"], (int, float)) or metadata["distance_cm"] < 0:
        return False
    if metadata["lighting"] not in VALID_LIGHTING:
        return False
    if metadata["label"] not in LABEL_NAMES:
        return False
    if not isinstance(metadata["is_transition"], bool):
        return False

    # Feature version must be 1 if specified
    version = metadata.get("feature_version", 1)
    if not isinstance(version, int) or version != 1:
        return False

    return True


def save_dataset_npz(
    filepath: str,
    features: np.ndarray,
    labels: Union[List[str], np.ndarray],
    metadata: List[Dict[str, Any]],
) -> None:
    """Save features, labels, and JSON-encoded metadata to an .npz archive.

    Args:
        filepath: Destination path for the .npz file.
        features: Numpy array of shape (N, 63).
        labels: List or array of N string labels.
        metadata: List of N metadata dictionaries.

    Raises:
        ValueError: If array lengths do not match or features shape is invalid.
    """
    features_arr = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=str)

    n_samples = len(features_arr)
    if len(labels_arr) != n_samples or len(metadata) != n_samples:
        raise ValueError(
            f"Length mismatch: features={n_samples}, labels={len(labels_arr)}, metadata={len(metadata)}"
        )

    if n_samples > 0 and (features_arr.ndim != 2 or features_arr.shape[1] != 63):
        raise ValueError(f"Features must have shape (N, 63), got {features_arr.shape}")

    # Ensure every metadata dict has default feature_version=1
    normalized_meta = []
    for meta in metadata:
        m = dict(meta)
        m.setdefault("feature_version", 1)
        normalized_meta.append(m)

    meta_json = json.dumps(normalized_meta)

    parent_dir = os.path.dirname(os.path.abspath(filepath))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    np.savez_compressed(
        filepath,
        features=features_arr,
        labels=labels_arr,
        metadata=meta_json,
    )


def load_dataset_npz(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load and validate dataset from an .npz archive.

    Args:
        filepath: Path to the .npz archive.

    Returns:
        tuple (features: (N, 63) float32, labels: (N,) string array, metadata: list[dict])

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If archive format or data validation fails.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with np.load(filepath, allow_pickle=True) as data:
        for required_key in ("features", "labels", "metadata"):
            if required_key not in data.files:
                raise ValueError(f"Missing required key '{required_key}' in .npz archive")

        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=str)
        raw_meta = data["metadata"]

        if isinstance(raw_meta, np.ndarray):
            meta_str = str(raw_meta.item()) if raw_meta.ndim == 0 else str(raw_meta[0])
        else:
            meta_str = str(raw_meta)

        try:
            metadata: List[Dict[str, Any]] = json.loads(meta_str)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON metadata: {e}") from e

    n_samples = len(features)
    if len(labels) != n_samples or len(metadata) != n_samples:
        raise ValueError(
            f"Length mismatch in archive: features={n_samples}, labels={len(labels)}, metadata={len(metadata)}"
        )

    if n_samples > 0:
        if features.ndim != 2 or features.shape[1] != 63:
            raise ValueError(f"Features must have shape (N, 63), got {features.shape}")

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Features contain NaNs or infinite values")

    for i, meta in enumerate(metadata):
        if not isinstance(meta, dict):
            raise ValueError(f"Metadata entry {i} is not a dictionary")
        if meta.get("label") not in LABEL_NAMES:
            raise ValueError(f"Metadata entry {i} has invalid label: '{meta.get('label')}'")
        if meta.get("feature_version", 1) != 1:
            raise ValueError(f"Metadata entry {i} has invalid feature_version: {meta.get('feature_version')}")

    return features, labels, metadata


def export_dataset_csv(
    filepath: str,
    features: np.ndarray,
    labels: Union[List[str], np.ndarray],
    metadata: List[Dict[str, Any]],
) -> None:
    """Export dataset to CSV format for tabular analysis and inspection.

    Columns include all metadata fields followed by 63 feature columns (f_0 .. f_62).
    """
    features_arr = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=str)

    n_samples = len(features_arr)
    if len(labels_arr) != n_samples or len(metadata) != n_samples:
        raise ValueError(
            f"Length mismatch: features={n_samples}, labels={len(labels_arr)}, metadata={len(metadata)}"
        )

    if n_samples > 0 and (features_arr.ndim != 2 or features_arr.shape[1] != 63):
        raise ValueError(f"Features must have shape (N, 63), got {features_arr.shape}")

    parent_dir = os.path.dirname(os.path.abspath(filepath))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    meta_fields = [
        "user_id",
        "session_id",
        "timestamp",
        "handedness",
        "distance_cm",
        "lighting",
        "label",
        "is_transition",
        "feature_version",
    ]
    feature_fields = [f"f_{i}" for i in range(63)]
    fieldnames = meta_fields + feature_fields

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for i in range(n_samples):
            meta = metadata[i]
            row = [
                meta.get("user_id", ""),
                meta.get("session_id", ""),
                float(meta.get("timestamp", 0.0)),
                meta.get("handedness", "Right"),
                float(meta.get("distance_cm", 0.0)),
                meta.get("lighting", "bright"),
                labels_arr[i],
                bool(meta.get("is_transition", False)),
                int(meta.get("feature_version", 1)),
            ]
            row.extend(features_arr[i].tolist())
            writer.writerow(row)


def load_dataset_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load dataset from a CSV file exported by export_dataset_csv.

    Returns:
        tuple (features: (N, 63) float32, labels: (N,) string array, metadata: list[dict])
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    features_list: List[List[float]] = []
    labels_list: List[str] = []
    metadata_list: List[Dict[str, Any]] = []

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV file is empty")

        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if len(row) < 72:  # 9 metadata fields + 63 feature columns
                raise ValueError(f"Row {row_idx + 1} has insufficient columns ({len(row)} < 72)")

            meta = {
                "user_id": str(row[0]),
                "session_id": str(row[1]),
                "timestamp": float(row[2]),
                "handedness": str(row[3]),
                "distance_cm": float(row[4]),
                "lighting": str(row[5]),
                "label": str(row[6]),
                "is_transition": row[7].strip().lower() in ("true", "1", "t", "yes"),
                "feature_version": int(row[8]),
            }
            feat = [float(v) for v in row[9:72]]

            features_list.append(feat)
            labels_list.append(meta["label"])
            metadata_list.append(meta)

    features = np.asarray(features_list, dtype=np.float32) if features_list else np.empty((0, 63), dtype=np.float32)
    labels = np.asarray(labels_list, dtype=str) if labels_list else np.empty(0, dtype=str)

    return features, labels, metadata_list


def collect_data_interactive(
    output_path: str = "data/custom_airpad.npz",
    user_id: str = "user_01",
    camera_index: int = 0,
    distance_cm: float = 45.0,
    lighting: str = "mixed",
    session_id: Optional[str] = None,
    mock_frames: Optional[List[Any]] = None,
    raise_on_error: bool = False,
) -> int:
    """Run interactive webcam data collection session to capture custom pose datasets."""
    if session_id is None:
        session_id = f"sess_{datetime_now_str()}_{uuid.uuid4().hex[:6]}"

    collected_features: List[np.ndarray] = []
    collected_labels: List[str] = []
    collected_metadata: List[Dict[str, Any]] = []

    # If dataset already exists, load it to append. Fail closed if corrupt.
    if os.path.exists(output_path):
        try:
            ex_feat, ex_lbl, ex_meta = load_dataset_npz(output_path)
            collected_features.extend([ex_feat[i] for i in range(len(ex_feat))])
            collected_labels.extend(ex_lbl.tolist())
            collected_metadata.extend(ex_meta)
            print(f"[INFO] Loaded existing dataset with {len(ex_lbl)} samples from {output_path}")
        except Exception as e:
            print(f"[ERROR] Existing dataset at {output_path} is corrupt or unreadable: {e}", file=sys.stderr)
            if raise_on_error:
                raise ValueError(f"Existing dataset at {output_path} is corrupt or unreadable: {e}") from e
            return 1

    # Hotkey map to selectable poses
    pose_keys = {
        ord('1'): "PINCH_CLOSED",
        ord('2'): "PINCH_MIDDLE",
        ord('3'): "L_SHAPE",
        ord('4'): "OTHER",
        ord('5'): "POINT_ONE",
        ord('6'): "OPEN_PALM",
        ord('7'): "FIST",
        ord('8'): "TWO_FINGERS_PARALLEL",
        ord('9'): "THREE_FINGERS",
    }

    current_label = "PINCH_CLOSED"
    is_transition = False
    continuous_recording = False
    new_samples_count = 0

    if mock_frames is not None:
        # Mock mode for non-interactive test runs
        for frame_landmarks, handedness in mock_frames:
            feat = canonicalize(frame_landmarks, handedness=handedness).astype(np.float32)
            meta = create_sample_metadata(
                user_id=user_id,
                session_id=session_id,
                timestamp=time.time(),
                handedness=handedness,
                distance_cm=distance_cm,
                lighting=lighting,
                label=current_label,
                is_transition=is_transition,
                feature_version=1,
            )
            collected_features.append(feat)
            collected_labels.append(current_label)
            collected_metadata.append(meta)
            new_samples_count += 1

        if collected_features:
            save_dataset_npz(
                output_path,
                np.asarray(collected_features, dtype=np.float32),
                collected_labels,
                collected_metadata,
            )
        return 0

    try:
        import cv2
        from src.vision.adapter import MediaPipeAdapter
    except ImportError as e:
        print(f"[ERROR] Webcam collection requires cv2 and mediapipe: {e}", file=sys.stderr)
        return 1

    detector = MediaPipeAdapter(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("      AIRPAD CUSTOM POSE DATA COLLECTION TOOL")
    print("=" * 60)
    print(f"Output:   {output_path}")
    print(f"User ID:  {user_id}")
    print(f"Session:  {session_id}")
    print("\nControls:")
    print("  [1] PINCH_CLOSED          [2] PINCH_MIDDLE          [3] L_SHAPE")
    print("  [4] OTHER                 [5] POINT_ONE             [6] OPEN_PALM")
    print("  [7] FIST                  [8] TWO_FINGERS_PARALLEL  [9] THREE_FINGERS")
    print("  [SPACE] Record 1 frame    [R] Toggle Continuous Recording")
    print("  [T] Toggle Transition     [S] Save Dataset")
    print("  [ESC / Q] Save and Exit")
    print("=" * 60 + "\n")

    def save_current() -> None:
        if collected_features:
            save_dataset_npz(
                output_path,
                np.asarray(collected_features, dtype=np.float32),
                collected_labels,
                collected_metadata,
            )
            print(f"[INFO] Saved {len(collected_labels)} total samples ({new_samples_count} new in this session) to {output_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = detector.detect(rgb_frame)

            detected_hand = hands[0] if hands else None
            record_this_frame = False

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in pose_keys:
                current_label = pose_keys[key]
            elif key == ord(' '):
                record_this_frame = True
            elif key == ord('r'):
                continuous_recording = not continuous_recording
            elif key == ord('t'):
                is_transition = not is_transition
            elif key == ord('s'):
                save_current()

            if continuous_recording:
                record_this_frame = True

            # Record frame if hand is present and recording triggered
            if record_this_frame and detected_hand is not None:
                lms = detected_hand.to_numpy()
                handedness = detected_hand.handedness
                feat = canonicalize(lms, handedness=handedness).astype(np.float32)
                meta = create_sample_metadata(
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=time.time(),
                    handedness=handedness,
                    distance_cm=distance_cm,
                    lighting=lighting,
                    label=current_label,
                    is_transition=is_transition,
                    feature_version=1,
                )
                collected_features.append(feat)
                collected_labels.append(current_label)
                collected_metadata.append(meta)
                new_samples_count += 1

            # Draw hand skeleton overlay
            if detected_hand is not None:
                lms_pts = detected_hand.points
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17),
                ]
                for p1, p2 in connections:
                    pt1 = (int(lms_pts[p1].x * w), int(lms_pts[p1].y * h))
                    pt2 = (int(lms_pts[p2].x * w), int(lms_pts[p2].y * h))
                    cv2.line(frame, pt1, pt2, (255, 200, 0), 1)
                for pt in lms_pts:
                    px, py = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            # HUD Display
            status_color = (0, 0, 255) if continuous_recording else ((0, 255, 0) if detected_hand else (128, 128, 128))
            hand_str = f"Hand: {detected_hand.handedness} ({detected_hand.confidence:.2f})" if detected_hand else "No Hand Detected"
            cv2.putText(frame, f"TARGET: {current_label}  [1-9 to switch]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"{hand_str} | Session: {new_samples_count} new | Total: {len(collected_labels)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            rec_status = "REC: ON (Continuous)" if continuous_recording else "REC: Press SPACE to capture"
            trans_status = "TRANSITION: YES" if is_transition else "TRANSITION: NO"
            cv2.putText(frame, f"{rec_status} | {trans_status} [T] | Dist: {distance_cm}cm", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("AirPad Pose Collector", frame)

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        save_current()

    return 0


def datetime_now_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collect custom gesture pose samples from webcam.")
    parser.add_argument("--output", default="data/custom_airpad.npz", help="Output .npz dataset path")
    parser.add_argument("--user-id", default="user_01", help="User identifier (default: user_01)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--distance", type=float, default=45.0, help="Approx distance in cm (default: 45.0)")
    parser.add_argument("--lighting", default="mixed", choices=["bright", "dim", "mixed"], help="Lighting condition")
    parser.add_argument("--session-id", default=None, help="Explicit session ID")

    args = parser.parse_args(argv)
    return collect_data_interactive(
        output_path=args.output,
        user_id=args.user_id,
        camera_index=args.camera,
        distance_cm=args.distance,
        lighting=args.lighting,
        session_id=args.session_id,
    )


if __name__ == "__main__":
    sys.exit(main())
