#!/usr/bin/env python3
"""Evaluate a multi-record XYZ file with the VoxMol/MiDi OpenBabel protocol.

The input order is preserved in ``manifest.jsonl``.  OpenBabel conversion
failures have a null ``sdf_record_index`` and their XYZ record is copied to
``failed/``.  Reference-dependent distribution distances are emitted only
when ``--reference-stats`` is supplied.

Reference statistics are a JSON object with the following MiDi-style fields:
``num_nodes``, ``atom_types``, ``bond_types``, ``valencies``,
``bond_lengths``, and ``bond_angles``. Histograms may contain counts or
probabilities. Numeric histogram keys are interpreted as node count, valency,
Angstrom, and degrees, respectively.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Geometry import Point3D


RDLogger.DisableLog("rdApp.*")

ALLOWED_BONDS: dict[str, Any] = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}

BOND_TYPES = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}

DATASET_ATOM_VOCABS = {
    "qm9": {"H", "C", "N", "O", "F"},
    "drug": {"H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi"},
}


@dataclass
class XYZRecord:
    sample_index: int
    comment: str
    atoms: list[str]
    positions: np.ndarray
    raw_xyz: str
    score: float | None


@dataclass
class MoleculeGraph:
    sample_index: int
    atoms: list[str]
    positions: np.ndarray
    bonds: list[tuple[int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OctMole XYZ samples with OpenBabel and MiDi legacy metrics."
    )
    parser.add_argument("--xyz", type=Path, required=True, help="Multi-record XYZ input.")
    parser.add_argument(
        "--reference-stats",
        type=Path,
        default=None,
        help="Optional MiDi-style reference-statistics JSON. W1/TV metrics are omitted without it.",
    )
    parser.add_argument(
        "--rerank-ratio",
        type=float,
        default=1.0,
        help="Fraction retained per rerank chunk; 1.0 disables reranking (default: 1.0).",
    )
    parser.add_argument(
        "--rerank-chunk",
        type=int,
        default=500,
        help="Candidates ranked together, matching Uni-3DAR's default batch size (default: 500).",
    )
    parser.add_argument(
        "--clash-threshold",
        type=float,
        default=0.8,
        help="Remove atoms closer than this distance in Angstrom (default: 0.8, as in VoxMol).",
    )
    parser.add_argument(
        "--no-clash-filter",
        action="store_true",
        help="Disable VoxMol's close-atom removal step.",
    )
    parser.add_argument(
        "--reject-oov",
        choices=sorted(DATASET_ATOM_VOCABS),
        default=None,
        metavar="DATASET",
        help=(
            "Count records containing elements outside DATASET's vocabulary as "
            "conversion failures. This check is applied before clash filtering."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing openbabel output directory.",
    )
    return parser.parse_args()


def _canonical_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if not re.fullmatch(r"[A-Za-z]{1,3}", symbol):
        raise ValueError(f"invalid element symbol {symbol!r}")
    return symbol[0].upper() + symbol[1:].lower()


def read_xyz_records(path: Path) -> list[XYZRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[XYZRecord] = []
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        start = cursor
        try:
            n_atoms = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"line {cursor + 1}: expected atom count") from exc
        if n_atoms < 1:
            raise ValueError(f"line {cursor + 1}: atom count must be positive")
        if cursor + n_atoms + 1 >= len(lines):
            raise ValueError(f"record {len(records)} is truncated")
        comment = lines[cursor + 1]
        atom_lines = lines[cursor + 2 : cursor + 2 + n_atoms]
        atoms: list[str] = []
        positions: list[list[float]] = []
        for offset, line in enumerate(atom_lines, start=cursor + 3):
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"line {offset}: expected element and three coordinates")
            atoms.append(_canonical_symbol(fields[0]))
            try:
                xyz = [float(value) for value in fields[1:4]]
            except ValueError as exc:
                raise ValueError(f"line {offset}: invalid coordinate") from exc
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError(f"line {offset}: coordinate must be finite")
            positions.append(xyz)
        cursor += n_atoms + 2
        match = re.match(r"\s*score\s*:\s*([^\s]+)", comment, flags=re.IGNORECASE)
        score = None
        if match:
            try:
                score = float(match.group(1))
            except ValueError:
                score = None
        raw = "\n".join(lines[start:cursor]) + "\n"
        records.append(
            XYZRecord(
                sample_index=len(records),
                comment=comment,
                atoms=atoms,
                positions=np.asarray(positions, dtype=np.float64),
                raw_xyz=raw,
                score=score,
            )
        )
    return records


def format_xyz(atoms: list[str], positions: np.ndarray, comment: str) -> str:
    rows = [str(len(atoms)), comment]
    rows.extend(
        f"{atom}\t{x:.8f}\t{y:.8f}\t{z:.8f}"
        for atom, (x, y, z) in zip(atoms, positions)
    )
    return "\n".join(rows) + "\n"


def rerank_records(
    records: list[XYZRecord], ratio: float, chunk_size: int
) -> list[XYZRecord]:
    """Apply Uni-3DAR's per-generation-batch selection and floor rounding."""
    selected: list[XYZRecord] = []
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        missing_scores = [record.sample_index for record in chunk if record.score is None]
        if missing_scores:
            preview = ", ".join(str(index) for index in missing_scores[:5])
            raise ValueError(f"rerank requires a numeric score; missing for samples {preview}")
        ranked = sorted(chunk, key=lambda record: float(record.score), reverse=True)
        selected.extend(ranked[: int(len(ranked) * ratio)])
    return selected


def oov_statistics(records: list[XYZRecord], vocabulary: set[str]) -> dict[str, Any]:
    atom_counts = Counter(atom for record in records for atom in record.atoms)
    oov_counts = {atom: count for atom, count in sorted(atom_counts.items()) if atom not in vocabulary}
    oov_molecules = sum(any(atom not in vocabulary for atom in record.atoms) for record in records)
    total_atoms = sum(atom_counts.values())
    return {
        "num_atoms": sum(oov_counts.values()),
        "atom_fraction": _ratio(sum(oov_counts.values()), total_atoms),
        "num_molecules": oov_molecules,
        "molecule_fraction": _ratio(oov_molecules, len(records)),
        "elements": oov_counts,
    }


def remove_close_atoms(
    atoms: list[str], positions: np.ndarray, threshold: float
) -> tuple[list[str], np.ndarray, list[int]]:
    """Match VoxMol: repeatedly remove the first atom involved in a clash."""
    kept = list(range(len(atoms)))
    removed: list[int] = []
    while len(kept) > 1:
        pos = positions[kept]
        distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        clashing = np.argwhere((distances > 0) & (distances < threshold))
        if clashing.size == 0:
            break
        local_index = int(np.unique(clashing.ravel())[0])
        removed.append(kept.pop(local_index))
    return [atoms[index] for index in kept], positions[kept], removed


def openbabel_graph(ob: Any, mol: Any, sample_index: int) -> MoleculeGraph:
    atoms = [ob.GetSymbol(atom.GetAtomicNum()) for atom in ob.OBMolAtomIter(mol)]
    positions = np.asarray(
        [[atom.GetX(), atom.GetY(), atom.GetZ()] for atom in ob.OBMolAtomIter(mol)],
        dtype=np.float64,
    )
    bonds: list[tuple[int, int, int]] = []
    for bond in ob.OBMolBondIter(mol):
        # OpenBabel writes XYZ-derived aromatic systems to SDF in Kekule form.
        # GetBondOrder therefore matches the actual single/double bonds in SDF;
        # overriding it with IsAromatic would make metrics disagree with output.
        order = int(bond.GetBondOrder())
        if order not in BOND_TYPES:
            raise ValueError(f"unsupported OpenBabel bond order {order}")
        bonds.append((bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, order))
    return MoleculeGraph(sample_index, atoms, positions, bonds)


def graph_to_rdkit(graph: MoleculeGraph) -> Chem.Mol:
    editable = Chem.RWMol()
    for symbol in graph.atoms:
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(0)
        editable.AddAtom(atom)
    for begin, end, order in graph.bonds:
        editable.AddBond(begin, end, BOND_TYPES[order])
    mol = editable.GetMol()
    conformer = Chem.Conformer(len(graph.atoms))
    for index, (x, y, z) in enumerate(graph.positions):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conformer)
    return mol


def atom_is_stable(symbol: str, valency: int) -> bool:
    allowed = ALLOWED_BONDS.get(symbol)
    if allowed is None:
        return False
    if isinstance(allowed, dict):
        allowed = allowed.get(0)
    if isinstance(allowed, list):
        return valency in allowed
    return valency == allowed


def stability(graph: MoleculeGraph) -> tuple[int, bool, list[int]]:
    # MiDi's integer bond tensor turns aromatic 1.5 into 1 before summation.
    valencies = [0] * len(graph.atoms)
    for begin, end, order in graph.bonds:
        contribution = 1 if order == 4 else order
        valencies[begin] += contribution
        valencies[end] += contribution
    stable = [atom_is_stable(symbol, valency) for symbol, valency in zip(graph.atoms, valencies)]
    return sum(stable), bool(stable) and all(stable), valencies


def validity(mol: Chem.Mol) -> tuple[str | None, int | None, str | None]:
    count: int | None = None
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        count = len(fragments)
        largest = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
        Chem.SanitizeMol(largest)
        return Chem.MolToSmiles(largest, canonical=True), count, None
    except Chem.rdchem.AtomValenceException:
        return None, count, "atom_valence"
    except Chem.rdchem.KekulizeException:
        return None, count, "kekulize"
    except (Chem.rdchem.AtomKekulizeException, ValueError, RuntimeError) as exc:
        return None, count, type(exc).__name__


def _normalize(histogram: dict[float, float]) -> dict[float, float]:
    total = float(sum(histogram.values()))
    if total <= 0:
        return {}
    return {key: float(value) / total for key, value in histogram.items()}


def _normalize_array(histogram: np.ndarray) -> np.ndarray:
    total = float(histogram.sum())
    return histogram / total if total > 0 else histogram


def _dense_integer_w1(generated: dict[float, float], target: dict[float, float]) -> float:
    """MiDi wasserstein1d for integer-spaced node-count/valency bins."""
    keys = set(generated) | set(target)
    if not keys:
        return 0.0
    integer_keys: list[int] = []
    for key in keys:
        if key < 0 or not float(key).is_integer():
            raise ValueError(f"dense integer W1 received non-integer bin {key}")
        integer_keys.append(int(key))
    generated_dense = np.zeros(max(integer_keys) + 1, dtype=np.float64)
    target_dense = np.zeros_like(generated_dense)
    for key, value in generated.items():
        generated_dense[int(key)] = value
    for key, value in target.items():
        target_dense[int(key)] = value
    generated_dense = _normalize_array(generated_dense)
    target_dense = _normalize_array(target_dense)
    return float(np.abs(np.cumsum(generated_dense) - np.cumsum(target_dense)).sum())


def _dense_bond_length_w1(
    generated_by_type: dict[str, Any], target_by_type: dict[str, Any]
) -> dict[str, float]:
    """MiDi bond_length_distance using 0.01 Angstrom dense bins."""
    generated = {
        str(order): _normalize(_numeric_histogram(generated_by_type.get(str(order), {})))
        for order in range(1, 5)
    }
    target = {
        str(order): _normalize(_numeric_histogram(target_by_type.get(str(order), {})))
        for order in range(1, 5)
    }
    nonempty = [histogram for histogram in [*generated.values(), *target.values()] if histogram]
    if not nonempty:
        return {str(order): 0.0 for order in range(1, 5)}
    min_length = min(min(histogram) for histogram in nonempty)
    max_length = max(max(histogram) for histogram in nonempty)
    num_bins = int((max_length - min_length) * 100) + 1
    generated_dense = np.zeros((4, num_bins), dtype=np.float64)
    target_dense = np.zeros_like(generated_dense)
    for order in range(1, 5):
        for distance, value in generated[str(order)].items():
            generated_dense[order - 1, int((distance - min_length) * 100)] = value
        for distance, value in target[str(order)].items():
            target_dense[order - 1, int((distance - min_length) * 100)] = value
    distances = np.abs(
        np.cumsum(generated_dense, axis=1) - np.cumsum(target_dense, axis=1)
    ).sum(axis=1) / 100.0
    return {str(order): float(distances[order - 1]) for order in range(1, 5)}


def _dense_angle_w1(
    generated: dict[str, Any], target: dict[str, Any], symbols: Iterable[str]
) -> dict[str, float]:
    """MiDi angle_distance using its fixed 0..180 degree, 0.1-degree bins."""
    distances: dict[str, float] = {}
    for symbol in symbols:
        generated_dense = np.zeros(1801, dtype=np.float64)
        target_dense = np.zeros(1801, dtype=np.float64)
        for angle, value in _numeric_histogram(generated.get(symbol, {})).items():
            index = int(round(angle * 10))
            if 0 <= index <= 1800:
                generated_dense[index] = value
        for angle, value in _numeric_histogram(target.get(symbol, {})).items():
            index = int(round(angle * 10))
            if 0 <= index <= 1800:
                target_dense[index] = value
        generated_dense = _normalize_array(generated_dense)
        target_dense = _normalize_array(target_dense)
        distances[symbol] = float(
            np.abs(np.cumsum(generated_dense) - np.cumsum(target_dense)).sum() / 10.0
        )
    return distances


def _numeric_histogram(raw: dict[str, Any]) -> dict[float, float]:
    return {float(key): float(value) for key, value in raw.items()}


def _categorical_tv(generated: dict[str, float], target: dict[str, Any]) -> float:
    gen = _normalize({key: float(value) for key, value in generated.items()})
    ref = _normalize({key: float(value) for key, value in target.items()})
    return 0.5 * sum(abs(gen.get(key, 0.0) - ref.get(key, 0.0)) for key in set(gen) | set(ref))


def generated_statistics(graphs: Iterable[MoleculeGraph]) -> dict[str, Any]:
    node_counts: Counter[str] = Counter()
    atom_counts: Counter[str] = Counter()
    bond_counts: Counter[str] = Counter()
    valencies: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: dict[str, Counter[str]] = defaultdict(Counter)
    angles: dict[str, Counter[str]] = defaultdict(Counter)
    for graph in graphs:
        n_atoms = len(graph.atoms)
        node_counts[str(n_atoms)] += 1
        atom_counts.update(graph.atoms)
        total_pairs = n_atoms * (n_atoms - 1) // 2
        bond_counts["0"] += total_pairs - len(graph.bonds)
        adjacency: list[list[int]] = [[] for _ in graph.atoms]
        graph_valencies = [0] * n_atoms
        for begin, end, order in graph.bonds:
            bond_counts[str(order)] += 1
            graph_valencies[begin] += 1 if order == 4 else order
            graph_valencies[end] += 1 if order == 4 else order
            adjacency[begin].append(end)
            adjacency[end].append(begin)
            distance = round(float(np.linalg.norm(graph.positions[begin] - graph.positions[end])), 2)
            lengths[str(order)][f"{distance:.2f}"] += 1
        for symbol, valency in zip(graph.atoms, graph_valencies):
            valencies[symbol][str(valency)] += 1
        for center, neighbors in enumerate(adjacency):
            for left_index, left in enumerate(neighbors):
                for right in neighbors[left_index + 1 :]:
                    first = graph.positions[left] - graph.positions[center]
                    second = graph.positions[right] - graph.positions[center]
                    first_unit = first / (np.linalg.norm(first) + 1e-6)
                    second_unit = second / (np.linalg.norm(second) + 1e-6)
                    cosine = float(np.dot(first_unit, second_unit))
                    if not -1.0 <= cosine <= 1.0:
                        continue
                    angle = float(np.round(math.degrees(math.acos(cosine)), 1))
                    angles[graph.atoms[center]][f"{angle:.1f}"] += 1
    return {
        "num_nodes": dict(node_counts),
        "atom_types": dict(atom_counts),
        "bond_types": dict(bond_counts),
        "valencies": {key: dict(value) for key, value in valencies.items()},
        "bond_lengths": {key: dict(value) for key, value in lengths.items()},
        "bond_angles": {key: dict(value) for key, value in angles.items()},
    }


def reference_distances(graphs: list[MoleculeGraph], path: Path) -> dict[str, float]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    required = {"num_nodes", "atom_types", "bond_types", "valencies", "bond_lengths", "bond_angles"}
    missing = sorted(required - reference.keys())
    if missing:
        raise ValueError(f"reference statistics missing fields: {', '.join(missing)}")
    generated = generated_statistics(graphs)
    atom_weights = _normalize({key: float(value) for key, value in reference["atom_types"].items()})
    bond_weights = _normalize({key: float(value) for key, value in reference["bond_types"].items()})
    metrics = {
        "num_nodes_w1": _dense_integer_w1(
            _numeric_histogram(generated["num_nodes"]), _numeric_histogram(reference["num_nodes"])
        ),
        "atom_types_tv": _categorical_tv(generated["atom_types"], reference["atom_types"]),
        "bond_types_tv": _categorical_tv(generated["bond_types"], reference["bond_types"]),
    }
    metrics["valency_w1"] = sum(
        atom_weights.get(symbol, 0.0)
        * _dense_integer_w1(
            _numeric_histogram(generated["valencies"].get(symbol, {})),
            _numeric_histogram(histogram),
        )
        for symbol, histogram in reference["valencies"].items()
    )
    bond_length_distances = _dense_bond_length_w1(
        generated["bond_lengths"], reference["bond_lengths"]
    )
    metrics["bond_lengths_w1"] = sum(
        bond_weights.get(str(order), 0.0)
        * bond_length_distances.get(str(order), 0.0)
        for order in range(1, 5)
    )
    angle_distances = _dense_angle_w1(
        generated["bond_angles"], reference["bond_angles"], reference["bond_angles"]
    )
    angle_terms: list[tuple[float, float]] = []
    for symbol in reference["bond_angles"]:
        ref_valencies = _normalize(_numeric_histogram(reference["valencies"].get(symbol, {})))
        angle_weight = atom_weights.get(symbol, 0.0) * (1.0 - ref_valencies.get(0.0, 0.0) - ref_valencies.get(1.0, 0.0))
        angle_terms.append((angle_weight, angle_distances[symbol]))
    total_angle_weight = sum(weight for weight, _ in angle_terms)
    metrics["angles_w1"] = (
        sum(weight * distance for weight, distance in angle_terms) / (total_angle_weight + 1e-5)
    )
    return metrics


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from openbabel import openbabel as ob
    except ImportError as exc:
        raise RuntimeError(
            "OpenBabel is not installed; run `uv sync --extra openbabel-eval`."
        ) from exc

    xyz_path = args.xyz.expanduser().resolve()
    if not xyz_path.is_file():
        raise FileNotFoundError(xyz_path)
    if args.clash_threshold <= 0:
        raise ValueError("--clash-threshold must be positive")
    if not 0 < args.rerank_ratio <= 1:
        raise ValueError("--rerank-ratio must be in (0, 1]")
    if args.rerank_chunk <= 0:
        raise ValueError("--rerank-chunk must be positive")
    output_dir = xyz_path.parent / "openbabel"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    failed_dir = output_dir / "failed"
    failed_dir.mkdir(parents=True)

    records = read_xyz_records(xyz_path)
    rerank_enabled = args.rerank_ratio < 1.0
    selected_records = (
        rerank_records(records, args.rerank_ratio, args.rerank_chunk)
        if rerank_enabled
        else records
    )
    selected_indices = {record.sample_index for record in selected_records}
    rerank_positions = {
        record.sample_index: index for index, record in enumerate(selected_records)
    }
    oov_by_dataset = {
        dataset: oov_statistics(selected_records, vocabulary)
        for dataset, vocabulary in DATASET_ATOM_VOCABS.items()
    }
    converter = ob.OBConversion()
    if not converter.SetInFormat("xyz"):
        raise RuntimeError("OpenBabel XYZ reader is unavailable")
    sdf_serializer = ob.OBConversion()
    if not sdf_serializer.SetOutFormat("sdf"):
        raise RuntimeError("OpenBabel SDF serializer is unavailable")
    sdf_reader = ob.OBConversion()
    if not sdf_reader.SetInFormat("sdf"):
        raise RuntimeError("OpenBabel SDF reader is unavailable")
    sdf_path = output_dir / "molecules.sdf"
    stable_sdf_path = output_dir / "molecules_stable.sdf"
    unstable_sdf_path = output_dir / "molecules_unstable.sdf"
    sdf_records: list[str] = []
    stable_sdf_records: list[str] = []
    unstable_sdf_records: list[str] = []
    graphs: list[MoleculeGraph] = []
    manifest_by_index: dict[int, dict[str, Any]] = {}
    valid_smiles: list[str] = []
    error_counts: Counter[str] = Counter()
    stable_atoms = stable_molecules = valid_count = connected_count = 0
    total_atoms = 0
    component_counts: list[int] = []

    for record in records:
        if record.sample_index not in selected_indices:
            manifest_by_index[record.sample_index] = {
                "sample_index": record.sample_index,
                "score": record.score,
                "input_atom_count": len(record.atoms),
                "selected_for_evaluation": False,
                "rerank_position": None,
                "converted": False,
                "sdf_record_index": None,
                "valid": False,
                "smiles": None,
            }

    for record in selected_records:
        atoms = record.atoms
        positions = record.positions
        removed: list[int] = []
        if not args.no_clash_filter:
            atoms, positions, removed = remove_close_atoms(atoms, positions, args.clash_threshold)
        filtered_xyz = format_xyz(atoms, positions, record.comment)
        row: dict[str, Any] = {
            "sample_index": record.sample_index,
            "score": record.score,
            "selected_for_evaluation": True,
            "rerank_position": rerank_positions[record.sample_index] if rerank_enabled else None,
            "input_atom_count": len(record.atoms),
            "evaluated_atom_count": len(atoms),
            "removed_atom_indices": removed,
            "converted": False,
            "sdf_record_index": None,
            "valid": False,
            "smiles": None,
        }
        mol = ob.OBMol()
        try:
            if args.reject_oov is not None:
                unexpected = sorted(
                    set(record.atoms) - DATASET_ATOM_VOCABS[args.reject_oov]
                )
                if unexpected:
                    raise ValueError(
                        f"elements outside {args.reject_oov} vocabulary: {unexpected}"
                    )
            converted = bool(converter.ReadString(mol, filtered_xyz))
            if not converted or mol.NumAtoms() != len(atoms):
                raise ValueError(
                    f"OpenBabel returned {mol.NumAtoms()} atoms for {len(atoms)} input atoms"
                )
            # VoxMol evaluates the OpenBabel-produced SDF, not the in-memory
            # XYZ molecule. Round-trip once so bond orders and 4-decimal SDF
            # coordinates used by metrics exactly match the saved artifact.
            sdf_text = sdf_serializer.WriteString(mol)
            sdf_mol = ob.OBMol()
            if not sdf_text or not sdf_reader.ReadString(sdf_mol, sdf_text):
                raise ValueError("could not round-trip OpenBabel molecule through SDF")
            graph = openbabel_graph(ob, sdf_mol, record.sample_index)
            if graph.atoms != atoms:
                raise ValueError(f"element mismatch: input={atoms}, OpenBabel={graph.atoms}")
            stable_count, molecule_stable, valencies = stability(graph)
            rdkit_mol = graph_to_rdkit(graph)
            smiles, components, validity_error = validity(rdkit_mol)
            # Serialize only after graph construction and metric preparation
            # succeeded, so sdf_record_index always addresses molecules.sdf.
            sdf_mol.SetTitle(f"sample_{record.sample_index:06d}")
            output_sdf_record = sdf_serializer.WriteString(sdf_mol)
            if not output_sdf_record:
                raise RuntimeError("could not serialize molecule for aggregate SDF")
            row["converted"] = True
            row["sdf_record_index"] = len(graphs)
            row["bond_count"] = len(graph.bonds)
            row["stable_atom_count"] = stable_count
            row["molecule_stable"] = molecule_stable
            row["valencies"] = valencies
            stable_atoms += stable_count
            stable_molecules += int(molecule_stable)
            total_atoms += len(graph.atoms)
            sdf_records.append(output_sdf_record)
            if molecule_stable:
                stable_sdf_records.append(output_sdf_record)
            else:
                unstable_sdf_records.append(output_sdf_record)
            if components is not None:
                component_counts.append(components)
                connected_count += int(components == 1)
                row["num_components"] = components
            if smiles is not None:
                valid_count += 1
                valid_smiles.append(smiles)
                row["valid"] = True
                row["smiles"] = smiles
            elif validity_error:
                error_counts[validity_error] += 1
                row["validity_error"] = validity_error
            graphs.append(graph)
        except Exception as exc:  # preserve correspondence even for per-record failures
            error_name = type(exc).__name__
            error_counts[f"conversion:{error_name}"] += 1
            row["error"] = f"{error_name}: {exc}"
            (failed_dir / f"sample_{record.sample_index:06d}.xyz").write_text(
                record.raw_xyz, encoding="utf-8"
            )
        manifest_by_index[record.sample_index] = row

    sdf_path.write_text("".join(sdf_records), encoding="utf-8")
    stable_sdf_path.write_text("".join(stable_sdf_records), encoding="utf-8")
    unstable_sdf_path.write_text("".join(unstable_sdf_records), encoding="utf-8")

    converted_count = len(graphs)
    unique_count = len(set(valid_smiles))
    metrics: dict[str, Any] = {
        "input_xyz": str(xyz_path),
        "output_dir": str(output_dir),
        "versions": {
            "python": platform.python_version(),
            "openbabel": ob.OBReleaseVersion(),
            "rdkit": rdBase.rdkitVersion,
        },
        "configuration": {
            "clash_filter": not args.no_clash_filter,
            "clash_threshold_angstrom": None if args.no_clash_filter else args.clash_threshold,
            "reject_oov": args.reject_oov,
            "rerank_enabled": rerank_enabled,
            "rerank_ratio": args.rerank_ratio,
            "rerank_chunk": args.rerank_chunk,
        },
        "num_input": len(records),
        "num_selected": len(selected_records),
        "num_converted": converted_count,
        "num_failed": len(selected_records) - converted_count,
        "conversion_rate": _ratio(converted_count, len(selected_records)),
        "num_atoms_input": sum(len(record.atoms) for record in selected_records),
        "num_atoms_input_total": sum(len(record.atoms) for record in records),
        "num_atoms_evaluated": total_atoms,
        "oov": oov_by_dataset,
        "atom_stable": _ratio(stable_atoms, total_atoms),
        "mol_stable": _ratio(stable_molecules, converted_count),
        "valid": _ratio(valid_count, converted_count),
        "unique": _ratio(unique_count, valid_count),
        "connected": _ratio(connected_count, converted_count),
        "mean_num_components": float(np.mean(component_counts)) if component_counts else 0.0,
        "max_num_components": max(component_counts, default=0),
        "stable_molecule_count": stable_molecules,
        "valid_count": valid_count,
        "unique_count": unique_count,
        "error_counts": dict(error_counts),
    }
    if args.reference_stats is not None:
        reference_path = args.reference_stats.expanduser().resolve()
        metrics["reference_stats"] = str(reference_path)
        metrics.update(reference_distances(graphs, reference_path))

    manifest = [manifest_by_index[index] for index in range(len(records))]
    write_jsonl(output_dir / "manifest.jsonl", manifest)
    (output_dir / "valid_smiles.txt").write_text(
        "".join(f"{smiles}\n" for smiles in valid_smiles), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    try:
        metrics = evaluate(parse_args())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
