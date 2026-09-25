import os

import numpy as np


def voxel_positive_indices(voxel, threshold=0.5):
    voxel = np.asarray(voxel)
    if voxel.ndim == 4:
        voxel = voxel[0]
    x, y, z = np.nonzero(voxel > float(threshold))
    res = int(voxel.shape[-1])
    return (x * res * res + y * res + z).astype(np.int32, copy=False)


def load_sparse_occupancy_index(path):
    payload = np.load(path, allow_pickle=True)
    index = {key: payload[key] for key in payload.files if key != "metadata_json"}
    index["num_voxels"] = int(np.asarray(index["num_voxels"]).item())
    return index


def nearest_neighbor_occupancy_metrics(prediction_positive_indices, index):
    voxel_offsets = np.asarray(index["voxel_offsets"], dtype=np.int64)
    voxel_sample_ids = np.asarray(index["voxel_sample_ids"], dtype=np.int32)
    positive_counts = np.asarray(index["positive_counts"], dtype=np.float32)
    rows = []
    for values in prediction_positive_indices:
        pred = np.unique(np.asarray(values, dtype=np.int32).reshape(-1))
        pred_count = int(pred.size)
        if pred_count == 0:
            rows.append((0.0, 0.0, 0.0, 0.0))
            continue
        posting = [
            voxel_sample_ids[voxel_offsets[int(v)] : voxel_offsets[int(v) + 1]]
            for v in pred
            if 0 <= int(v) < len(voxel_offsets) - 1
        ]
        candidates = np.concatenate(posting) if posting else np.zeros(0, dtype=np.int32)
        if candidates.size == 0:
            rows.append((0.0, 0.0, 0.0, 0.0))
            continue
        sample_ids, intersections = np.unique(candidates, return_counts=True)
        tp = intersections.astype(np.float32, copy=False)
        gt_count = positive_counts[sample_ids]
        fp = float(pred_count) - tp
        fn = gt_count - tp
        iou = tp / np.maximum(tp + fp + fn, 1.0)
        best = int(np.argmax(iou))
        precision = float(tp[best] / max(tp[best] + fp[best], 1.0))
        recall = float(tp[best] / max(tp[best] + fn[best], 1.0))
        f1 = float((2.0 * tp[best]) / max(2.0 * tp[best] + fp[best] + fn[best], 1.0))
        rows.append((recall, precision, f1, float(iou[best])))
    values = np.asarray(rows or [(0.0, 0.0, 0.0, 0.0)], dtype=np.float32)
    return {
        "recall": float(values[:, 0].mean()),
        "precision": float(values[:, 1].mean()),
        "f1": float(values[:, 2].mean()),
        "iou": float(values[:, 3].mean()),
    }


def evaluate_voxel_folder(folder, train_index="", test_index="", threshold=0.5):
    files = sorted(name for name in os.listdir(folder) if name.endswith(".npy"))
    positives = [voxel_positive_indices(np.load(os.path.join(folder, name)), threshold) for name in files]
    counts = np.asarray([len(values) for values in positives], dtype=np.float32)
    metrics = {
        "pred_positive_count_mean": float(counts.mean()) if counts.size else 0.0,
        "pred_positive_count_min": float(counts.min()) if counts.size else 0.0,
        "pred_positive_count_max": float(counts.max()) if counts.size else 0.0,
    }
    for split, path in (("train", train_index), ("test", test_index)):
        if path and os.path.exists(path):
            for name, value in nearest_neighbor_occupancy_metrics(positives, load_sparse_occupancy_index(path)).items():
                metrics[f"nn_{split}_{name}"] = value
    return metrics
