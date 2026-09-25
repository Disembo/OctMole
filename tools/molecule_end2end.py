import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.molecule import MoleculeDataset
from models.denoiser_molecule import MoleculeOctreeDenoiser, MoleculeFeatureDenoiser
from util.molecule_export import (
    ATOM_LIST,
    export_molecule_sample,
    molecule_sample_arrays,
    write_samples_xyz,
)
from util.molecule_metrics import evaluate_xyz_file


def setup_dist(device_arg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_arg.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_arg)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and device_arg.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, device


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def load_stage1(path, args, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["args"].train_mode == "molecule_octree"
    assert ckpt["args"].octree_depth == args.octree_depth
    args.octree_depth_increment = int(
        getattr(ckpt["args"], "octree_depth_increment", 1)
    )
    model = MoleculeOctreeDenoiser(ckpt["args"]).to(device).eval()
    model.load_state_dict(ckpt["model_ema1"])
    if is_main_process():
        print(f"Loaded stage1 ckpt: {path} at epoch {ckpt['epoch']}")
    return model


def load_stage2(path, resolution, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["args"].train_mode == "molecule_feature"
    assert ckpt["args"].voxel_size == resolution
    model = MoleculeFeatureDenoiser(ckpt["args"]).to(device).eval()
    model.load_state_dict(ckpt["model_ema1"])
    if is_main_process():
        print(f"Loaded stage2 ckpt: {path} at epoch {ckpt['epoch']}")
    return model


def batch_from_sparse(items, device):
    max_len = max(len(item["spvoxel"]) for item in items)
    batch = {
        "spvoxel": torch.zeros(len(items), max_len, 3, dtype=torch.long, device=device),
        "padding_mask": torch.ones(len(items), max_len, dtype=torch.bool, device=device),
    }
    for i, item in enumerate(items):
        n = len(item["spvoxel"])
        batch["spvoxel"][i, :n] = torch.as_tensor(item["spvoxel"], device=device)
        batch["padding_mask"][i, :n] = False
    return batch


def octree_stats(leaves, max_depth):
    leaves = np.unique(np.asarray(leaves, dtype=np.int64).reshape(-1, 3), axis=0)
    node_counts = np.zeros(max_depth + 1, dtype=np.int64)
    branching = np.zeros((max_depth, 9), dtype=np.int64)
    node_counts[0] = 1
    for d in range(1, max_depth + 1):
        child_width = 1 << (max_depth - d)
        parent_width = child_width << 1
        parents = np.unique((leaves // parent_width) * parent_width, axis=0)
        children = np.unique((leaves // child_width) * child_width, axis=0)
        node_counts[d] = len(children)
        counts = np.zeros(len(parents), dtype=np.int64)
        for i, parent in enumerate(parents):
            counts[i] = np.all((children // parent_width) * parent_width == parent, axis=1).sum()
        branching[d - 1] = np.bincount(counts, minlength=9)
    return node_counts, branching


def save_stage1(root, item, args, source):
    root.mkdir(parents=True, exist_ok=True)
    sample_id = item["id"]
    np.savez_compressed(
        root / f"{sample_id:06d}.npz",
        spvoxel=item["spvoxel"],
        node_counts=item["node_counts"],
        branching_hist=item["branching_hist"],
    )
    rank = item.get("rank", 0)
    metadata = {
        "dataset_id": item.get("dataset_id"),
        "seed": args.seed + rank if source == "sample" else args.seed,
        "rank": rank,
        "source": source,
        "stage1_checkpoint": args.stage1_model or None,
        "stage2_checkpoint": args.stage2_model,
        "octree_depth": args.octree_depth,
        "octree_depth_increment": args.octree_depth_increment,
        "box_extent": args.box_extent,
    }
    (root / f"{sample_id:06d}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@torch.no_grad()
def sample_stage1(model, ids, args, device):
    out = []
    for offset in tqdm(range(0, len(ids), args.stage1_bsz), desc="Sample stage1", disable=not is_main_process()):
        chunk = ids[offset:offset + args.stage1_bsz]
        result = model.generate(len(chunk), device)
        for row, sample_id in enumerate(chunk):
            valid = ~result["padding_mask"][row]
            item = {
                "id": sample_id,
                "spvoxel": result["spvoxel"][row, valid].cpu().numpy(),
                "node_counts": result["node_counts"][row].cpu().numpy(),
                "branching_hist": result["branching_hist"][row].cpu().numpy(),
                "box_extent": args.box_extent,
                "rank": int(os.environ.get("RANK", "0")),
                "source": "sample",
            }
            out.append(item)
    return out


def sample_gt(ids, selected, dataset, args):
    out = []
    for sample_id, dataset_id in zip(ids, selected):
        sample = dataset[int(dataset_id)]
        leaves = sample["spvoxel"].numpy().astype(np.int64)
        node_counts, branching = octree_stats(leaves, args.octree_depth)
        item = {
            "id": sample_id,
            "spvoxel": leaves,
            "node_counts": node_counts,
            "branching_hist": branching,
            "box_extent": float(sample["box_extent"]),
            "coord_feature_gt": sample["coord_feature"].numpy(),
            "rank": int(os.environ.get("RANK", "0")),
            "source": "ground_truth",
            "dataset_id": int(dataset_id),
        }
        out.append(item)
    return out


@torch.no_grad()
def sample_clamped(model, ids, selected, dataset, args, device):
    clamp = args.stage1_clamp_depth
    shift = args.octree_depth - clamp
    init_nodes = []
    meta = []
    for sample_id, dataset_id in zip(ids, selected):
        sample = dataset[int(dataset_id)]
        leaves = np.unique(sample["spvoxel"].numpy().astype(np.int64), axis=0)
        init = np.unique((leaves >> shift) << shift, axis=0)
        kept_leaves = None
        if args.stage1_edit_mode == "plane":
            centers = leaves + 0.5
            axis = int(np.argmax(centers.max(0) - centers.min(0)))
            keep = centers[:, axis] < np.median(centers[:, axis])
            kept = leaves[keep]
            kept_ancestors = set(map(tuple, np.unique((kept >> shift) << shift, axis=0)))
            pruned = [c for c in map(tuple, init) if c not in kept_ancestors]
            if pruned:
                kept_leaves = kept
                init = np.asarray(pruned, dtype=np.int64)
        init_nodes.append(torch.as_tensor(init, dtype=torch.long))
        meta.append((sample_id, float(sample["box_extent"]), int(dataset_id), kept_leaves))
    result = model.generate(0, device, init_nodes=init_nodes, init_depth=clamp)
    out = []
    for row, (sample_id, box, gt_dataset_id, kept_leaves) in enumerate(meta):
        valid = ~result["padding_mask"][row]
        spvoxel = result["spvoxel"][row, valid].cpu().numpy()
        node_counts = result["node_counts"][row].cpu().numpy()
        branching = result["branching_hist"][row].cpu().numpy()
        if kept_leaves is not None and len(kept_leaves):
            spvoxel = np.unique(np.concatenate([spvoxel, kept_leaves], axis=0), axis=0)
            node_counts, branching = octree_stats(spvoxel, args.octree_depth)
        out.append({
            "id": sample_id,
            "spvoxel": spvoxel,
            "node_counts": node_counts,
            "branching_hist": branching,
            "box_extent": box,
            "rank": int(os.environ.get("RANK", "0")),
            "source": f"clamp{clamp}_{args.stage1_edit_mode}" if args.stage1_edit_mode else f"clamp{clamp}",
            "dataset_id": gt_dataset_id,
        })
    return out


@torch.no_grad()
def sample_stage2(model, sparse_items, args, device):
    resolution = 1 << args.octree_depth
    nonempty = [item for item in sparse_items if len(item["spvoxel"])]
    out = []
    for offset in tqdm(range(0, len(nonempty), args.stage2_bsz), desc="Sample stage2", disable=not is_main_process()):
        items = nonempty[offset:offset + args.stage2_bsz]
        result = model.generate(batch_from_sparse(items, device))
        result = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in result.items()}
        for row, item in enumerate(items):
            sample = {k: v[row] for k, v in result.items()} | {
                "coord_feature_gt": item.get("coord_feature_gt")
            }
            if args.snap_offsets and "coord_feature" in sample:
                sample["coord_feature"] = sample["coord_feature"].copy()
                sample["coord_feature"][:, 0] = 0.0
            atom_type, _, _, points, score = molecule_sample_arrays(
                sample, resolution, item["box_extent"]
            )
            out.append({"id": item["id"], "stage1": item, "sample": sample,
                        "xyz": (atom_type, points, score)})
    return out


def summarize_stage1(counts, branching):
    counts = np.stack(counts)
    leaf_counts = counts[:, -1]
    values, frequencies = np.unique(leaf_counts, return_counts=True)
    return {
        "num_samples": len(counts),
        "node_count_mean": counts.mean(axis=0).tolist(),
        "leaf_count_mean": float(leaf_counts.mean()),
        "leaf_count_std": float(leaf_counts.std()),
        "leaf_count_percentiles": np.percentile(leaf_counts, [0, 25, 50, 75, 100]).tolist(),
        "leaf_count_histogram": {str(k): int(v) for k, v in zip(values, frequencies)},
        "atom_count_histogram": {str(k): int(v) for k, v in zip(values, frequencies)},
        "branching_histogram": branching.tolist(),
    }


def atom_type_distribution(counts):
    total = counts.sum()
    return {ATOM_LIST[i]: float(value / total) for i, value in enumerate(counts) if value}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--octree-depth", type=int, required=True)
    parser.add_argument("--box-extent", type=float, required=True)
    parser.add_argument("--metrics-dataset", required=True, choices=("qm9", "drug"))
    parser.add_argument("--stage1-gt", action="store_true")
    parser.add_argument("--stage1-model", default="")
    parser.add_argument("--stage2-model", required=True)
    parser.add_argument("--stage1-bsz", type=int, default=128)
    parser.add_argument("--stage2-bsz", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--snap-offsets", action="store_true",
                        help="Snap stage2 atoms to cell centers (drop intra-cell offsets).")
    parser.add_argument("--stage1-clamp-depth", type=int, default=0,
                        help="Clamp stage1 occupancy up to this depth from GT "
                             "molecules, regenerate deeper levels only.")
    parser.add_argument("--gt-pool", type=int, default=0,
                        help="restrict GT conditioning to this many distinct molecules, cycled")
    parser.add_argument("--stage1-edit-mode", default="", choices=("", "plane"),
                        help="plane: keep GT leaves on one side of a median plane exactly, "
                             "regrow the other half (fragment editing demo)")
    parser.add_argument("--save-extra", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    args.octree_depth_increment = 1
    rank, world_size, device = setup_dist(args.device)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    out_dir = Path(args.out_dir)
    stage1_dir = out_dir / "stage1"
    sample_dir = out_dir / "samples"
    xyz_path = out_dir / "samples.xyz"
    all_xyz_path = out_dir / "samples_all.xyz"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_samples_xyz([], xyz_path)
        write_samples_xyz([], all_xyz_path)
        if args.save_extra:
            stage1_dir.mkdir(parents=True, exist_ok=True)
            sample_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    if args.stage1_gt and args.stage1_clamp_depth:
        raise ValueError("--stage1-clamp-depth is incompatible with --stage1-gt")
    if not args.stage1_gt and not args.stage1_model:
        raise ValueError("--stage1-model is required unless --stage1-gt is set")
    stage1_model = None if args.stage1_gt else load_stage1(args.stage1_model, args, device)
    if args.stage1_clamp_depth:
        if args.stage1_clamp_depth % stage1_model.depth_increment:
            raise ValueError("clamp depth must be a multiple of the stage1 depth increment")
    stage2_model = load_stage2(args.stage2_model, 1 << args.octree_depth, device)
    ratio = 0.2 if args.metrics_dataset == "drug" else 0.1
    rerank_chunk = 512
    selected_per_chunk = max(1, math.ceil(rerank_chunk * ratio))
    candidate_id = selected_count = 0
    stage1_counts = []
    stage1_branching = np.zeros((args.octree_depth, 9), dtype=np.int64)
    atom_counts = np.zeros(len(ATOM_LIST), dtype=np.int64)
    num_candidates = 0

    gt_dataset = gt_selected = None
    if args.stage1_gt or args.stage1_clamp_depth:
        gt_dataset = MoleculeDataset(
            args.data_path, train_mode="molecule_feature",
            voxel_size=1 << args.octree_depth, split="valid",
            box_extent=args.box_extent,
        )
        full_chunks, remainder = divmod(args.num_samples, selected_per_chunk)
        candidate_count = full_chunks * rerank_chunk
        if remainder:
            candidate_count += math.floor(remainder / ratio)
        rng = np.random.default_rng(args.seed)
        if args.gt_pool:
            pool = rng.choice(len(gt_dataset), size=min(args.gt_pool, len(gt_dataset)),
                              replace=False)
            gt_selected = np.resize(pool, candidate_count)
        else:
            gt_selected = rng.choice(
                len(gt_dataset), size=candidate_count,
                replace=candidate_count > len(gt_dataset),
            )

    while selected_count < args.num_samples:
        remaining = args.num_samples - selected_count
        chunk_size = (
            rerank_chunk if remaining >= selected_per_chunk
            else math.floor(remaining / ratio)
        )
        chunk = []
        while len(chunk) < chunk_size:
            count = chunk_size - len(chunk)
            ids = list(range(candidate_id, candidate_id + count))
            local_ids = ids[rank::world_size]
            if args.stage1_gt:
                local_selected = gt_selected[candidate_id:candidate_id + count][rank::world_size]
                sparse_items = sample_gt(local_ids, local_selected, gt_dataset, args)
            elif args.stage1_clamp_depth:
                local_selected = gt_selected[candidate_id:candidate_id + count][rank::world_size]
                sparse_items = sample_clamped(
                    stage1_model, local_ids, local_selected, gt_dataset, args, device
                )
            else:
                sparse_items = sample_stage1(stage1_model, local_ids, args, device)
            local_rows = sample_stage2(stage2_model, sparse_items, args, device)
            gathered = [None] * world_size if rank == 0 else None
            if world_size > 1:
                dist.gather_object((sparse_items, local_rows), gathered, dst=0)
            else:
                gathered = [(sparse_items, local_rows)]
            if rank == 0:
                stage1_rows = [item for part, _ in gathered for item in part]
                rows = [row for _, part in gathered for row in part]
                for item in stage1_rows:
                    stage1_counts.append(item["node_counts"])
                    stage1_branching += item["branching_hist"]
                chunk.extend(sorted(rows, key=lambda row: row["id"]))
                valid_count = len(chunk)
            else:
                valid_count = 0
            candidate_id += count
            count_tensor = torch.tensor(valid_count, device=device)
            if world_size > 1:
                dist.broadcast(count_tensor, src=0)
            valid_count = int(count_tensor.item())
            if rank != 0:
                chunk = [None] * valid_count

        if rank == 0:
            chunk_xyz = [row["xyz"] for row in chunk]
            ranked = sorted(chunk, key=lambda row: row["xyz"][2], reverse=True)
            selected = ranked[:math.ceil(len(ranked) * ratio)]
            write_samples_xyz(chunk_xyz, all_xyz_path, append=True)
            write_samples_xyz([row["xyz"] for row in selected], xyz_path, append=True)
            num_candidates += len(chunk_xyz)
            for atom_type, _, _ in chunk_xyz:
                atom_counts += np.bincount(
                    atom_type, minlength=len(ATOM_LIST)
                )[:len(ATOM_LIST)]
            if args.save_extra:
                for index, row in enumerate(selected, start=selected_count):
                    item = row["stage1"] | {"id": index}
                    save_stage1(stage1_dir, item, args, item["source"])
                    export_molecule_sample(
                        sample_dir, row["sample"], index,
                        1 << args.octree_depth, item["box_extent"],
                    )
            selected_count += len(selected)
        count_tensor = torch.tensor(selected_count, device=device)
        if world_size > 1:
            dist.broadcast(count_tensor, src=0)
        selected_count = int(count_tensor.item())

    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        metrics = evaluate_xyz_file(all_xyz_path, dataset=args.metrics_dataset)
        metrics["num_exported"] = selected_count
        metrics["num_candidates"] = num_candidates
        metrics["rerank_ratio"] = ratio
        metrics["rerank_chunk"] = rerank_chunk
        metrics["stage1"] = summarize_stage1(stage1_counts, stage1_branching)
        metrics["atom_type_distribution"] = atom_type_distribution(atom_counts)
        print(json.dumps(metrics, indent=2))
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
