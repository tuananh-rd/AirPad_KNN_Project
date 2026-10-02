import numpy as np
from tools.collect_data import load_dataset_npz, save_dataset_npz
from tools.train import validate_dataset

# 1. Load HaGRID
h_feat, h_lbl, h_meta = load_dataset_npz("data/hagrid_airpad.npz")
print(f"Loaded HaGRID: {len(h_lbl)} samples")

# 2. Load custom data
c_feat, c_lbl, c_meta = load_dataset_npz("data/custom_airpad.npz")
print(f"Loaded Custom: {len(c_lbl)} samples")

# Segment custom data into 5 distinct temporal chunks per class
# so that each fold gets an independent contiguous time block
new_c_meta = []
n_chunks = 5

for cls_name in ["PINCH_CLOSED", "PINCH_MIDDLE", "L_SHAPE"]:
    cls_indices = [i for i, l in enumerate(c_lbl) if l == cls_name]
    splits = np.array_split(cls_indices, n_chunks)
    for chunk_idx, idx_list in enumerate(splits):
        for idx in idx_list:
            m = dict(c_meta[idx])
            m["user_id"] = f"tuana_chunk{chunk_idx}"
            m["session_id"] = f"tuana_{cls_name}_c{chunk_idx}"
            new_c_meta.append((idx, m))

new_c_meta.sort(key=lambda x: x[0])
c_meta_stratified = [m for _, m in new_c_meta]

# Combine
all_features = np.vstack([h_feat, c_feat]).astype(np.float32)
all_labels = np.concatenate([h_lbl, c_lbl])
all_metadata = h_meta + c_meta_stratified

validate_dataset(all_features, all_labels, all_metadata)
save_dataset_npz("data/combined_airpad.npz", all_features, all_labels, all_metadata)

print("Saved data/combined_airpad.npz successfully!")
from collections import Counter
counts = Counter(all_labels)
for c, n in sorted(counts.items()):
    print(f"  {c:<25} {n:>5}")
