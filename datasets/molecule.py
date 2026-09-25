import gzip
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


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
ATOM_TO_ID = {name: i for i, name in enumerate(ATOM_LIST)}

COVALENT_RADIUS = {
    "H": 0.31, "B": 0.85, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "As": 1.19,
    "Br": 1.20, "I": 1.39, "Hg": 1.32, "Bi": 1.48,
}


def _normalize_symbol(symbol):
    symbol = str(symbol)
    return symbol[:1].upper() + symbol[1:].lower()


def _rotation_matrix(rng):
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _line_voxels(start, end, size):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    voxel = np.clip(np.floor(start).astype(np.int32), 0, size - 1)
    last = np.clip(np.floor(end).astype(np.int32), 0, size - 1)
    direction = end - start
    step = np.sign(direction).astype(np.int32)
    t_delta = np.full(3, np.inf, dtype=np.float32)
    t_max = np.full(3, np.inf, dtype=np.float32)

    for axis in range(3):
        if direction[axis] == 0:
            continue
        boundary = voxel[axis] + (1 if step[axis] > 0 else 0)
        t_max[axis] = (boundary - start[axis]) / direction[axis]
        t_delta[axis] = abs(1.0 / direction[axis])

    out = []
    while True:
        out.append(tuple(int(x) for x in voxel))
        if np.array_equal(voxel, last):
            break
        axis = int(np.argmin(t_max))
        voxel[axis] += step[axis]
        if np.any(voxel < 0) or np.any(voxel >= size):
            break
        t_max[axis] += t_delta[axis]
    return out


def _infer_bonds(atom_types, atom_pos, grid_pos, voxel_size):
    bond_voxels = []
    for i in range(len(atom_types)):
        ri = COVALENT_RADIUS.get(atom_types[i], 0.77)
        for j in range(i):
            rj = COVALENT_RADIUS.get(atom_types[j], 0.77)
            distance = float(np.linalg.norm(atom_pos[i] - atom_pos[j]))
            if distance < 0.4 or distance > 1.25 * (ri + rj):
                continue
            bond_voxels.extend(_line_voxels(grid_pos[i], grid_pos[j], voxel_size))
    if not bond_voxels:
        return np.zeros((0, 3), dtype=np.int32)
    return np.unique(np.asarray(bond_voxels, dtype=np.int32), axis=0)


def _crop_to_box(atom_types, atom_pos, box_extent):
    centered = atom_pos - atom_pos.mean(axis=0, keepdims=True)
    half = float(box_extent) * 0.5
    in_box = np.all((centered >= -half) & (centered < half), axis=1)
    if in_box.all():
        return atom_types, centered

    center_idx = int(np.argmin(np.linalg.norm(centered, axis=1)))
    centered = atom_pos - atom_pos[center_idx : center_idx + 1]
    in_box = np.all((centered >= -half) & (centered < half), axis=1)
    if not in_box.any():
        return atom_types, atom_pos - atom_pos.mean(axis=0, keepdims=True)
    keep = np.nonzero(in_box)[0]
    return [atom_types[i] for i in keep], centered[keep]


def _octree_training_level(spvoxel, depth, max_depth, depth_increment=1):
    leaves = np.asarray(spvoxel, dtype=np.int64).reshape(-1, 3)
    leaves = np.unique(leaves, axis=0)
    side = 1 << depth_increment
    child_width = 1 << (max_depth - depth)
    parent_width = child_width * side
    leaf_parents = (leaves // parent_width) * parent_width
    parents, inverse = np.unique(leaf_parents, axis=0, return_inverse=True)

    child_offsets = (leaves - leaf_parents) // child_width
    child_indices = child_offsets @ np.array([1, side, side * side], dtype=np.int64)
    target = np.zeros((len(parents), side ** 3), dtype=np.float32)
    target[inverse, child_indices] = 1.0
    return parents, target


def _random_cell_shift(spvoxel, delta, resolution, rng):
    num_shifts = int(rng.integers(len(spvoxel) // 10 + 1))
    if num_shifts == 0:
        return

    directions = np.array(
        [
            [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, 1],
        ],
        dtype=np.int32,
    )
    selected = rng.choice(len(spvoxel), size=num_shifts, replace=False)
    shifts = directions[rng.integers(len(directions), size=num_shifts)]
    occupied = {tuple(voxel) for voxel in spvoxel}

    for atom_idx, shift in zip(selected, shifts):
        source = spvoxel[atom_idx]
        target = source + shift
        if np.any(target < 0) or np.any(target >= resolution) or tuple(target) in occupied:
            continue

        occupied.remove(tuple(source))
        occupied.add(tuple(target))
        spvoxel[atom_idx] = target
        delta[atom_idx] -= shift


class MoleculeDataset(Dataset):
    def __init__(
        self, root, train_mode, voxel_size, split, box_extent,
        include_bonds=False, random_rotation=False, sample_conformer=False, seed=0,
        octree_depth=None, octree_depth_increment=1, random_cell_shift=False,
    ):
        self.root = Path(root)
        self.train_mode = train_mode
        self.voxel_size = None if train_mode == "molecule_octree" else int(voxel_size)
        self.box_extent = float(box_extent)
        self.include_bonds = bool(include_bonds) and train_mode != "molecule_octree"
        self.random_rotation = bool(random_rotation)
        self.random_cell_shift = bool(random_cell_shift)
        self.sample_conformer = bool(sample_conformer)
        self.seed = int(seed)
        self.octree_depth = int(octree_depth) if octree_depth is not None else None
        self.octree_depth_increment = int(octree_depth_increment)
        if self.train_mode == "molecule_octree" and self.octree_depth is None:
            raise ValueError("octree_depth is required for molecule_octree")
        self.resolution = (
            1 << self.octree_depth
            if self.train_mode == "molecule_octree"
            else self.voxel_size
        )
        self.epoch = 0
        self.env = None
        self.env_pid = None

        if self.train_mode not in {"molecule_voxel", "molecule_feature", "molecule_octree"}:
            raise ValueError(f"Unsupported train_mode: {self.train_mode}")
        if self.box_extent <= 0.0:
            raise ValueError("box_extent must be positive for LMDB molecule data.")

        self.lmdb_path = self.root / f"{split}.lmdb"
        if not self.lmdb_path.is_file():
            raise FileNotFoundError(f"{self.lmdb_path} not found")

        env = self._connect(save=False)
        with env.begin() as txn:
            self.length = int(txn.stat()["entries"])
            first = next(txn.cursor().iternext(values=False))
        env.close()
        first_text = first.decode("ascii")
        self.key_width = len(first_text) if first_text.startswith("0") and len(first_text) > 1 else 0

    def _connect(self, save=True):
        env = lmdb.open(
            str(self.lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if save:
            self.env = env
            self.env_pid = os.getpid()
        return env

    def _env(self):
        if self.env is None or self.env_pid != os.getpid():
            self.close()
            self._connect()
        return self.env

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = None
        self.env_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        state["env_pid"] = None
        return state

    def __del__(self):
        self.close()

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rng(self, index):
        return np.random.default_rng(self.seed + self.epoch * 1000003 + int(index))

    def _key(self, index):
        if self.key_width:
            return f"{int(index):0{self.key_width}d}".encode("ascii")
        return str(int(index)).encode("ascii")

    def _load(self, index):
        with self._env().begin() as txn:
            payload = txn.get(self._key(index))
        if payload is None:
            raise KeyError(f"Missing LMDB key for index {index}")
        try:
            payload = gzip.decompress(payload)
        except OSError:
            pass
        return pickle.loads(payload)

    def _payload(self, index):
        sample = self._load(index)
        rng = self._rng(index)
        atom_types = [_normalize_symbol(x) for x in sample["atom_type"]]
        atom_pos = sample["atom_pos"]
        if isinstance(atom_pos, (list, tuple)):
            conf = int(rng.integers(len(atom_pos))) if self.sample_conformer else 0
            atom_pos = atom_pos[conf]
        atom_pos = np.asarray(atom_pos, dtype=np.float32).reshape(-1, 3)
        if self.random_rotation and len(atom_pos):
            atom_pos = atom_pos - atom_pos.mean(axis=0, keepdims=True)
            atom_pos = atom_pos @ _rotation_matrix(rng).T
        atom_types, atom_pos = _crop_to_box(atom_types, atom_pos, self.box_extent)

        grid_pos = (atom_pos + self.box_extent * 0.5) / self.box_extent * self.resolution
        keep = np.all((grid_pos >= 0.0) & (grid_pos < float(self.resolution)), axis=1)
        atom_types = [atom_types[i] for i in np.nonzero(keep)[0]]
        atom_pos = atom_pos[keep]
        grid_pos = np.clip(grid_pos[keep], 0.0, np.nextafter(float(self.resolution), 0.0))

        spvoxel = np.floor(grid_pos).astype(np.int32)
        delta = grid_pos - (spvoxel.astype(np.float32) + 0.5)
        if self.train_mode == "molecule_feature" and self.random_cell_shift:
            _random_cell_shift(spvoxel, delta, self.resolution, rng)
        length = np.linalg.norm(delta, axis=1, keepdims=True).astype(np.float32)
        direction = np.zeros_like(delta, dtype=np.float32)
        nonzero = length[:, 0] > 0.0
        direction[nonzero] = delta[nonzero] / length[nonzero]
        atom_type = np.asarray([ATOM_TO_ID[x] for x in atom_types], dtype=np.int64)
        bonds = _infer_bonds(atom_types, atom_pos, grid_pos, self.resolution) if self.include_bonds else np.zeros((0, 3), dtype=np.int32)

        return {
            "spvoxel": spvoxel.astype(np.int32, copy=False),
            "spvoxel_bond": bonds,
            "coord_feature": np.concatenate([length, direction], axis=1).astype(np.float32),
            "atom_type": atom_type,
            "uid": np.asarray(index, dtype=np.int64),
            "box_extent": np.asarray(self.box_extent, dtype=np.float32),
        }

    def _dense_voxel(self, data):
        voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.float32)
        bond = np.asarray(data["spvoxel_bond"], dtype=np.int64).reshape(-1, 3)
        atom = np.asarray(data["spvoxel"], dtype=np.int64).reshape(-1, 3)
        if len(bond):
            voxel[bond[:, 0], bond[:, 1], bond[:, 2]] = 0.5
        if len(atom):
            voxel[atom[:, 0], atom[:, 1], atom[:, 2]] = 1.0
        return torch.from_numpy(voxel).unsqueeze(0)

    def __getitem__(self, index):
        data = self._payload(index)
        if self.train_mode == "molecule_voxel":
            return self._dense_voxel(data)
        if self.train_mode == "molecule_octree":
            depth_rng = self._rng(index + self.length)
            depth = self.octree_depth_increment * int(
                depth_rng.integers(1, self.octree_depth // self.octree_depth_increment + 1)
            )
            spvoxel, target = _octree_training_level(
                data["spvoxel"], depth=depth, max_depth=self.octree_depth,
                depth_increment=self.octree_depth_increment,
            )
            return {
                "spvoxel": torch.as_tensor(spvoxel, dtype=torch.long),
                "target": torch.as_tensor(target, dtype=torch.float32),
                "depth": torch.tensor(depth, dtype=torch.long),
                "uid": torch.tensor(int(data["uid"]), dtype=torch.long),
                "box_extent": torch.tensor(
                    float(data["box_extent"]), dtype=torch.float32
                ),
            }

        atom_type = torch.as_tensor(data["atom_type"], dtype=torch.long)
        coord_feature = torch.as_tensor(data["coord_feature"], dtype=torch.float32)
        spvoxel = torch.as_tensor(data["spvoxel"], dtype=torch.long)
        feature = torch.cat([atom_type.float().unsqueeze(1), coord_feature], dim=1)
        return {
            "feature": feature,
            "coord_feature": coord_feature,
            "spvoxel": spvoxel,
            "atom_type": atom_type,
            "uid": torch.tensor(int(data["uid"]), dtype=torch.long),
            "box_extent": torch.tensor(float(data["box_extent"]), dtype=torch.float32),
        }


def molecule_collate(batch):
    if isinstance(batch[0], torch.Tensor):  # mode = molecule_voxel
        return torch.stack(batch, dim=0)

    if "target" in batch[0]:  # mode = molecule_octree
        max_len = max(item["target"].shape[0] for item in batch)
        bsz = len(batch)
        target_dim = batch[0]["target"].shape[1]
        out = {
            "spvoxel": torch.zeros(bsz, max_len, 3, dtype=torch.long),
            "target": torch.zeros(bsz, max_len, target_dim, dtype=torch.float32),
            "padding_mask": torch.ones(bsz, max_len, dtype=torch.bool),
            "depth": torch.stack([item["depth"] for item in batch]),
            "uid": torch.stack([item["uid"] for item in batch]),
            "box_extent": torch.stack([item["box_extent"] for item in batch]),
        }
        for i, item in enumerate(batch):
            n = item["target"].shape[0]
            out["spvoxel"][i, :n] = item["spvoxel"]
            out["target"][i, :n] = item["target"]
            out["padding_mask"][i, :n] = False
        return out

    # mode = molecule_features
    max_len = max(item["feature"].shape[0] for item in batch)
    bsz = len(batch)
    feat_dim = batch[0]["feature"].shape[1]
    out = {
        "feature": torch.zeros(bsz, max_len, feat_dim, dtype=torch.float32),
        "coord_feature": torch.zeros(bsz, max_len, 4, dtype=torch.float32),
        "spvoxel": torch.zeros(bsz, max_len, 3, dtype=torch.long),
        "atom_type": torch.zeros(bsz, max_len, dtype=torch.long),
        "padding_mask": torch.ones(bsz, max_len, dtype=torch.bool),
        "uid": torch.stack([item["uid"] for item in batch]),
        "box_extent": torch.stack([item["box_extent"] for item in batch]),
    }
    for i, item in enumerate(batch):
        n = item["feature"].shape[0]
        out["feature"][i, :n] = item["feature"]
        out["coord_feature"][i, :n] = item["coord_feature"]
        out["spvoxel"][i, :n] = item["spvoxel"]
        out["atom_type"][i, :n] = item["atom_type"]
        out["padding_mask"][i, :n] = False
    return out
