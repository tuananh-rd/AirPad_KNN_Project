import json

import numpy as np

from tools.collect_data import load_dataset_npz
from tools.import_hagrid import import_hagrid_annotations, main


def _hand(offset: float = 0.0, dimensions: int = 2):
    points = []
    for index in range(21):
        point = [
            0.2 + offset + index * 0.013 + (index % 3) * 0.004,
            0.3 + index * 0.009 + (index % 5) * 0.003,
        ]
        if dimensions == 3:
            point.append(index * -0.002)
        points.append(point)
    return points


def _write_annotation(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def test_import_mapping_variants_metadata_and_roundtrip(tmp_path):
    annotation = tmp_path / "annotations" / "train" / "palm.json"
    output = tmp_path / "airpad.npz"
    _write_annotation(
        annotation,
        {
            "image_b": {
                "labels": ["palm", "not_mapped"],
                "hand_landmarks": [_hand(dimensions=2), _hand(0.02)],
                "user_id": "person_7",
            },
            "image_bad": {
                "labels": ["fist"],
                "hand_landmarks": [[[0.0, 0.0]]],
                "user_id": "person_8",
            },
        },
    )

    summary = import_hagrid_annotations(annotation.parent.parent, output)
    features, labels, metadata = load_dataset_npz(str(output))

    assert summary["unmapped_hands"] == 1
    assert summary["malformed_hands"] == 1
    assert summary["output_samples"] == 2
    assert features.shape == (2, 63)
    assert features.dtype == np.float32
    assert labels.tolist() == ["OPEN_PALM", "OPEN_PALM"]
    assert {item["handedness"] for item in metadata} == {"Right", "Left"}
    assert {item["user_id"] for item in metadata} == {"person_7"}
    assert len({item["session_id"] for item in metadata}) == 1
    assert np.all(np.isfinite(features))


def test_deterministic_per_class_cap_preserves_mirror_pairs(tmp_path):
    annotation = tmp_path / "annotations" / "one.json"
    records = {
        f"image_{index}": {
            "labels": ["one"],
            "hand_landmarks": [_hand(index * 0.001, dimensions=3)],
            "user_id": f"person_{index}",
        }
        for index in range(5)
    }
    _write_annotation(annotation, records)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    summary_a = import_hagrid_annotations(annotation, first, max_per_class=2, seed=9)
    summary_b = import_hagrid_annotations(annotation, second, max_per_class=2, seed=9)
    features_a, labels_a, metadata_a = load_dataset_npz(str(first))
    features_b, labels_b, metadata_b = load_dataset_npz(str(second))

    assert summary_a["retained_hands"] == 2
    assert summary_a["output_samples"] == 4
    np.testing.assert_allclose(features_a, features_b)
    assert labels_a.tolist() == labels_b.tolist() == ["POINT_ONE"] * 4
    assert metadata_a == metadata_b
    sessions = [item["session_id"] for item in metadata_a]
    assert all(sessions.count(session) == 2 for session in set(sessions))


def test_cli_reports_zero_usable_samples(tmp_path, capsys):
    annotation = tmp_path / "unknown.json"
    _write_annotation(
        annotation,
        {"image": {"labels": ["ok"], "hand_landmarks": [_hand()], "user_id": "u"}},
    )

    exit_code = main([str(annotation), str(tmp_path / "out.npz")])

    assert exit_code == 1
    assert "zero usable" in capsys.readouterr().err
