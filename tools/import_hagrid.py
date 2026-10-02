"""Convert HaGRIDv2 landmark annotations to the AirPad dataset schema."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.features.canonicalize import canonicalize
from tools.collect_data import create_sample_metadata, save_dataset_npz


HAGRID_TO_AIRPAD = {
    "palm": "OPEN_PALM",
    "one": "POINT_ONE",
    "fist": "FIST",
    "two_up": "TWO_FINGERS_PARALLEL",
    "three": "THREE_FINGERS",
    "no_gesture": "OTHER",
}


def _annotation_files(input_path: Path) -> Tuple[Path, List[Path]]:
    if not input_path.exists():
        raise FileNotFoundError(f"HaGRID input does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError("HaGRID input file must use the .json extension")
        return input_path.parent, [input_path]
    files = sorted(path for path in input_path.rglob("*.json") if path.is_file())
    if not files:
        raise ValueError(f"No annotation JSON files found under: {input_path}")
    return input_path, files


def _coerce_landmarks(value: Any) -> Optional[np.ndarray]:
    try:
        landmarks = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if landmarks.shape == (21, 2):
        landmarks = np.column_stack((landmarks, np.zeros(21, dtype=float)))
    if landmarks.shape != (21, 3) or not np.all(np.isfinite(landmarks)):
        return None
    return landmarks


def _iter_hands(
    root: Path,
    files: Iterable[Path],
    stats: Counter,
) -> Iterable[Dict[str, Any]]:
    for annotation_path in files:
        try:
            with annotation_path.open("r", encoding="utf-8") as stream:
                records = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            stats["invalid_files"] += 1
            continue
        if not isinstance(records, dict):
            stats["invalid_files"] += 1
            continue

        source = annotation_path.relative_to(root).as_posix()
        stats["files_processed"] += 1
        if stats["files_processed"] % 5 == 0 or stats["files_processed"] == 1:
            print(f"[{stats['files_processed']}] Processing {source} ... (usable: {stats['usable_hands']})", file=sys.stderr, flush=True)
        for image_id in sorted(records, key=str):
            stats["images_seen"] += 1
            record = records[image_id]
            if not isinstance(record, dict):
                stats["malformed_hands"] += 1
                continue
            labels = record.get("labels")
            hands = record.get("hand_landmarks")
            if not isinstance(labels, list) or not isinstance(hands, list):
                stats["malformed_hands"] += 1
                continue
            if len(labels) != len(hands):
                stats["malformed_hands"] += abs(len(labels) - len(hands))

            user_id = str(record.get("user_id") or "hagrid_unknown")
            session_id = f"hagrid:{source}:{image_id}"
            for source_label, raw_landmarks in zip(labels, hands):
                target_label = HAGRID_TO_AIRPAD.get(str(source_label))
                if target_label is None:
                    stats["unmapped_hands"] += 1
                    continue
                landmarks = _coerce_landmarks(raw_landmarks)
                if landmarks is None:
                    stats["malformed_hands"] += 1
                    continue
                stats["usable_hands"] += 1
                yield {
                    "label": target_label,
                    "landmarks": landmarks,
                    "user_id": user_id,
                    "session_id": session_id,
                }


def _cap_per_class(
    samples: Iterable[Dict[str, Any]],
    max_per_class: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be at least 1")
    if max_per_class is None:
        return list(samples)

    # Per-class reservoir sampling keeps memory proportional to the requested
    # cap instead of the 1.3M+ hands in the complete HaGRIDv2 annotations.
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: Counter = Counter()
    generators: Dict[str, random.Random] = {}
    for sample in samples:
        label = sample["label"]
        seen[label] += 1
        reservoir = grouped[label]
        if len(reservoir) < max_per_class:
            reservoir.append(sample)
            continue
        rng = generators.setdefault(label, random.Random(f"{seed}:{label}"))
        replacement_index = rng.randrange(seen[label])
        if replacement_index < max_per_class:
            reservoir[replacement_index] = sample

    selected: List[Dict[str, Any]] = []
    for label in sorted(grouped):
        selected.extend(grouped[label])
    return selected


def import_hagrid_annotations(
    input_path: str | Path,
    output_path: str | Path,
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Import HaGRID annotations and save a schema-valid AirPad NPZ file.

    ``max_per_class`` counts source hands. Each retained hand emits two feature
    rows (Right and Left canonicalization) because HaGRID does not provide
    reliable handedness metadata.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    root, files = _annotation_files(input_path)
    stats: Counter = Counter()
    retained = _cap_per_class(_iter_hands(root, files, stats), max_per_class, seed)
    if not retained:
        raise ValueError("HaGRID annotations contained zero usable mapped hands")

    features: List[np.ndarray] = []
    labels: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for sample in retained:
        for handedness in ("Right", "Left"):
            features.append(
                canonicalize(sample["landmarks"], handedness=handedness).astype(np.float32)
            )
            labels.append(sample["label"])
            metadata.append(
                create_sample_metadata(
                    user_id=sample["user_id"],
                    session_id=sample["session_id"],
                    timestamp=0.0,
                    handedness=handedness,
                    distance_cm=0.0,
                    lighting="mixed",
                    label=sample["label"],
                    is_transition=False,
                    feature_version=1,
                )
            )

    feature_array = np.asarray(features, dtype=np.float32).reshape(-1, 63)
    label_array = np.asarray(labels, dtype=str)
    save_dataset_npz(str(output_path), feature_array, label_array, metadata)

    class_counts = Counter(labels)
    return {
        "files_processed": stats["files_processed"],
        "invalid_files": stats["invalid_files"],
        "images_seen": stats["images_seen"],
        "malformed_hands": stats["malformed_hands"],
        "unmapped_hands": stats["unmapped_hands"],
        "usable_hands": stats["usable_hands"],
        "retained_hands": len(retained),
        "output_samples": len(labels),
        "class_counts": dict(sorted(class_counts.items())),
        "output": str(output_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert HaGRIDv2 landmark annotations to an AirPad NPZ dataset."
    )
    parser.add_argument("input", help="Annotation JSON file or directory")
    parser.add_argument("output", help="Destination .npz file")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum source hands per AirPad class before two-way mirroring",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        summary = import_hagrid_annotations(
            args.input,
            args.output,
            max_per_class=args.max_per_class,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
