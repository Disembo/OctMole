import json
import os
from pathlib import Path

import numpy as np


ATOM_LIST = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def coord_feature_to_points(spvoxel, coord_feature, voxel_size=32, box_extent=15.36):
    spvoxel = np.asarray(spvoxel, dtype=np.float32).reshape(-1, 3)
    coord_feature = np.asarray(coord_feature, dtype=np.float32).reshape(-1, 4)
    offset = coord_feature[:, :1] * coord_feature[:, 1:4]
    grid_pos = spvoxel + 0.5 + offset
    return grid_pos / float(voxel_size) * float(box_extent) - float(box_extent) * 0.5


def _write_obj(path, points):
    with open(path, "w", encoding="utf-8") as f:
        for p in np.asarray(points):
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def molecule_sample_arrays(result, voxel_size=32, box_extent=15.36):
    mask = ~result["padding_mask"]
    atom_type = result["atom_type"][mask].astype(np.int64, copy=False)
    coord = result["coord_feature"][mask].astype(np.float32, copy=False)
    spvoxel = result["spvoxel"][mask].astype(np.int64, copy=False)
    points = coord_feature_to_points(spvoxel, coord, voxel_size, box_extent)
    return atom_type, coord, spvoxel, points, float(result.get("score", 0.0))


def export_molecule_sample(sample_dir, result, index, voxel_size=32, box_extent=15.36):
    sample_dir = Path(sample_dir) / f"{index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    atom_type, coord, spvoxel, points, score = molecule_sample_arrays(
        result, voxel_size, box_extent
    )
    mask = ~result["padding_mask"]
    visual_points = coord_feature_to_points(spvoxel, coord, voxel_size, box_extent=1.0)
    feature = np.concatenate([atom_type[:, None].astype(np.float32), coord], axis=1)
    np.save(sample_dir / "feature_pred.npy", feature)
    np.save(sample_dir / "spvoxel.npy", spvoxel)
    _write_obj(sample_dir / "points_pred.obj", visual_points)
    
    # save gt points if provided
    if result.get("coord_feature_gt") is not None:
        coord_gt = result["coord_feature_gt"].astype(np.float32, copy=False)
        if coord_gt.shape[0] == mask.shape[0]:  # padded GT
            coord_gt = coord_gt[mask]
        elif coord_gt.shape[0] != spvoxel.shape[0]:
            raise ValueError(
                "coord_feature_gt must match either padding_mask "
                "or the number of valid spvoxels"
            )
        visual_points_gt = coord_feature_to_points(spvoxel, coord_gt, voxel_size, box_extent=1.0)
        _write_obj(sample_dir / "points_gt.obj", visual_points_gt)
    
    metadata = {
        "score": score,
        "voxel_size": int(voxel_size),
        "box_extent": float(box_extent),
    }
    (sample_dir / "sample_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return atom_type, points, metadata["score"]


def write_samples_xyz(rows, path, append=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for atom_type, points, score in rows:
            f.write(f"{len(atom_type)}\n")
            f.write(f"score: {float(score):.9g}\n")
            for type_id, point in zip(atom_type, points):
                symbol = ATOM_LIST[int(type_id)] if 0 <= int(type_id) < len(ATOM_LIST) else "X"
                f.write(f"{symbol}\t{point[0]:.8f}\t{point[1]:.8f}\t{point[2]:.8f}\n")


def write_xyz_from_sample_dirs(sample_root, xyz_path):
    rows = []
    for child in sorted(Path(sample_root).iterdir()):
        if not child.is_dir():
            continue
        feature_path = child / "feature_pred.npy"
        spvoxel_path = child / "spvoxel.npy"
        meta_path = child / "sample_metadata.json"
        if not (feature_path.exists() and spvoxel_path.exists() and meta_path.exists()):
            continue
        feature = np.load(feature_path)
        spvoxel = np.load(spvoxel_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        points = coord_feature_to_points(
            spvoxel,
            feature[:, 1:],
            voxel_size=metadata["voxel_size"],
            box_extent=metadata["box_extent"],
        )
        rows.append((np.rint(feature[:, 0]).astype(np.int64), points, float(metadata.get("score", 0.0))))
    write_samples_xyz(rows, xyz_path)
    return xyz_path, len(rows)


def export_stage1_voxel_ply(voxel, path, threshold=0.5):
    voxel = np.asarray(voxel)
    if voxel.ndim == 4:
        voxel = voxel[0]
    coords = np.argwhere(voxel > float(threshold))
    size = int(voxel.shape[0])
    points = (coords.astype(np.float32) + 0.5) / float(size) - 0.5
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
