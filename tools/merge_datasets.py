"""Utility to merge multiple AirPad .npz datasets while preserving metadata and schema."""
from __future__ import annotations

import argparse
from collections import Counter
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

from tools.collect_data import load_dataset_npz, save_dataset_npz
from tools.train import validate_dataset


def merge_npz_datasets(
    input_paths: List[str],
    output_path: str,
) -> int:
    """Merge multiple .npz dataset files into a single consolidated dataset."""
    all_features: List[np.ndarray] = []
    all_labels: List[str] = []
    all_metadata: List[dict] = []

    print(f"Merging {len(input_paths)} datasets into {output_path}...")

    for path in input_paths:
        if not os.path.exists(path):
            print(f"[ERROR] Input file does not exist: {path}", file=sys.stderr)
            return 1
        feats, lbls, meta = load_dataset_npz(path)
        print(f"  Loaded {len(lbls)} samples from {path}")
        all_features.append(feats)
        all_labels.extend(lbls.tolist() if isinstance(lbls, np.ndarray) else lbls)
        all_metadata.extend(meta)

    if not all_features:
        print("[ERROR] No data found to merge.", file=sys.stderr)
        return 1

    merged_features = np.vstack(all_features).astype(np.float32)
    merged_labels = np.asarray(all_labels, dtype=str)

    validate_dataset(merged_features, merged_labels, all_metadata, raise_on_error=True)

    save_dataset_npz(output_path, merged_features, merged_labels, all_metadata)

    class_counts = Counter(all_labels)
    print("\n" + "=" * 50)
    print("        DATASET MERGE COMPLETE")
    print("=" * 50)
    print(f"Output:       {output_path}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Total classes: {len(class_counts)}")
    print("-" * 50)
    for cls_name, count in sorted(class_counts.items()):
        print(f"  {cls_name:<25} {count:>8} samples")
    print("=" * 50 + "\n")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge multiple AirPad .npz datasets into one.")
    parser.add_argument("inputs", nargs="+", help="Input .npz files to merge")
    parser.add_argument("--output", "-o", required=True, help="Destination .npz path")

    args = parser.parse_args(argv)
    return merge_npz_datasets(args.inputs, args.output)


if __name__ == "__main__":
    sys.exit(main())
