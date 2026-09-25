#!/usr/bin/env python3
"""Build validation reference statistics from a OctMole molecule LMDB.

Each LMDB record is converted through the same XYZ -> OpenBabel -> SDF
round-trip used by ``evaluate_openbabel.py``.  The resulting JSON can be
passed directly to that script with ``--reference-stats``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

import lmdb
import numpy as np
from tqdm import tqdm

if __package__:
    from .evaluate_openbabel import (
        MoleculeGraph,
        _canonical_symbol,
        format_xyz,
        generated_statistics,
        openbabel_graph,
    )
else:
    from evaluate_openbabel import (
        MoleculeGraph,
        _canonical_symbol,
        format_xyz,
        generated_statistics,
        openbabel_graph,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenBabel reference statistics from validation LMDB data."
    )
    parser.add_argument(
        "--lmdb",
        type=Path,
        required=True,
        help="Validation LMDB file; its filename must be valid.lmdb.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output reference-statistics JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def _decode_record(payload: bytes) -> dict[str, Any]:
    try:
        payload = gzip.decompress(payload)
    except OSError:
        pass
    record = pickle.loads(payload)
    if not isinstance(record, dict):
        raise TypeError(f"expected a dictionary, got {type(record).__name__}")
    if "atom_type" not in record or "atom_pos" not in record:
        raise KeyError("record must contain atom_type and atom_pos")
    return record


def _atoms_and_positions(record: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    atoms = [_canonical_symbol(symbol) for symbol in record["atom_type"]]
    positions = np.asarray(record["atom_pos"], dtype=np.float64)
    if positions.ndim == 3:
        # Validation uses the deterministic first conformer, as OctMole does
        # when sample_conformer is disabled.
        positions = positions[0]
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError(
            f"atom_pos must have shape [N, 3] or [C, N, 3], got {positions.shape}"
        )
    if len(atoms) != len(positions):
        raise ValueError(
            f"atom_type has {len(atoms)} entries but atom_pos has {len(positions)}"
        )
    if not np.isfinite(positions).all():
        raise ValueError("atom_pos contains a non-finite coordinate")
    return atoms, positions


def _graphs_from_lmdb(
    lmdb_path: Path, ob: Any, counters: dict[str, int]
) -> Iterator[MoleculeGraph]:
    converter = ob.OBConversion()
    serializer = ob.OBConversion()
    reader = ob.OBConversion()
    if not converter.SetInFormat("xyz"):
        raise RuntimeError("OpenBabel XYZ reader is unavailable")
    if not serializer.SetOutFormat("sdf"):
        raise RuntimeError("OpenBabel SDF serializer is unavailable")
    if not reader.SetInFormat("sdf"):
        raise RuntimeError("OpenBabel SDF reader is unavailable")

    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    try:
        with env.begin() as transaction:
            counters["num_records"] = int(transaction.stat()["entries"])
            values = transaction.cursor().iternext(keys=False, values=True)
            progress = tqdm(values, total=counters["num_records"], unit="molecule")
            for index, payload in enumerate(progress):
                try:
                    record = _decode_record(payload)
                    atoms, positions = _atoms_and_positions(record)
                    xyz = format_xyz(atoms, positions, f"validation_{index}")

                    molecule = ob.OBMol()
                    if not converter.ReadString(molecule, xyz):
                        raise ValueError("OpenBabel could not read XYZ")
                    if molecule.NumAtoms() != len(atoms):
                        raise ValueError(
                            f"OpenBabel returned {molecule.NumAtoms()} atoms "
                            f"for {len(atoms)} input atoms"
                        )

                    # The SDF round-trip fixes the representation consumed by
                    # evaluation, including Kekule single/double bond orders.
                    sdf = serializer.WriteString(molecule)
                    sdf_molecule = ob.OBMol()
                    if not sdf or not reader.ReadString(sdf_molecule, sdf):
                        raise ValueError("could not round-trip molecule through SDF")
                    graph = openbabel_graph(ob, sdf_molecule, index)
                    if graph.atoms != atoms:
                        raise ValueError(
                            f"element mismatch: input={atoms}, OpenBabel={graph.atoms}"
                        )
                except Exception as exc:
                    counters["num_failed"] += 1
                    if counters["num_failed"] <= 10:
                        print(
                            f"warning: record {index}: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                    continue

                counters["num_converted"] += 1
                yield graph
    finally:
        env.close()


def build_stats(lmdb_path: Path) -> dict[str, Any]:
    try:
        from openbabel import openbabel as ob
    except ImportError as exc:
        raise RuntimeError(
            "OpenBabel is not installed; run `uv sync --extra openbabel-eval`."
        ) from exc

    # OpenBabel emits many non-fatal aromatic/stereochemistry diagnostics for
    # coordinate-only input. Conversion failures are reported by this script.
    ob.obErrorLog.StopLogging()
    counters = {"num_records": 0, "num_converted": 0, "num_failed": 0}
    statistics = generated_statistics(_graphs_from_lmdb(lmdb_path, ob, counters))
    if counters["num_converted"] == 0:
        raise RuntimeError("OpenBabel did not convert any validation records")
    statistics["_meta"] = {
        "split": "validation",
        "graph_source": f"openbabel-{ob.OBReleaseVersion()}-from-xyz-sdf",
        "num_records": counters["num_records"],
        "num_converted": counters["num_converted"],
    }
    if counters["num_failed"]:
        print(
            f"warning: omitted {counters['num_failed']} of {counters['num_records']} records",
            file=sys.stderr,
        )
    return statistics


def main() -> int:
    args = parse_args()
    lmdb_path = args.lmdb.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    try:
        if lmdb_path.name != "valid.lmdb":
            raise ValueError("--lmdb must point to the validation split named valid.lmdb")
        if not lmdb_path.is_file():
            raise FileNotFoundError(lmdb_path)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists; pass --overwrite to replace it"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = build_stats(lmdb_path)
        output_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {"output": str(output_path), **stats["_meta"]},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
