import contextlib
import math
import os
import tempfile

import numpy as np

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)


BONDS1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92, "B": 119, "Si": 148, "P": 144, "As": 152, "S": 134, "Cl": 127, "Br": 141, "I": 161},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135, "Si": 185, "P": 184, "S": 182, "Cl": 177, "Br": 194, "I": 214},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136, "Cl": 175, "Br": 214, "S": 168, "I": 222, "P": 177},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142, "Br": 172, "S": 151, "P": 163, "Si": 163, "Cl": 164, "I": 194},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142, "S": 158, "Si": 160, "Cl": 166, "Br": 178, "P": 156, "I": 187},
    "B": {"H": 119, "Cl": 175},
    "Si": {"Si": 233, "H": 148, "C": 185, "O": 163, "S": 200, "F": 160, "Cl": 202, "Br": 215, "I": 243},
    "Cl": {"Cl": 199, "H": 127, "C": 177, "N": 175, "O": 164, "P": 203, "S": 207, "B": 175, "Si": 202, "F": 166, "Br": 214},
    "S": {"H": 134, "C": 182, "N": 168, "O": 151, "S": 204, "F": 158, "Cl": 207, "Br": 225, "Si": 200, "P": 210, "I": 234},
    "Br": {"Br": 228, "H": 141, "C": 194, "O": 172, "N": 214, "Si": 215, "S": 225, "F": 178, "Cl": 214, "P": 222},
    "P": {"P": 221, "H": 144, "C": 184, "O": 163, "Cl": 203, "S": 210, "F": 156, "N": 177, "Br": 222},
    "I": {"H": 161, "C": 214, "Si": 243, "N": 222, "O": 194, "S": 234, "F": 187, "I": 266},
    "As": {"H": 152},
}
BONDS2 = {"C": {"C": 134, "N": 129, "O": 120, "S": 160}, "N": {"C": 129, "N": 125, "O": 121}, "O": {"C": 120, "N": 121, "O": 121, "P": 150}, "P": {"O": 150, "S": 186}, "S": {"P": 186}}
BONDS3 = {"C": {"C": 120, "N": 116, "O": 113}, "N": {"C": 116, "N": 110}, "O": {"C": 113}}
ALLOWED = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Al": 3, "Si": 4, "P": [3, 5], "S": 4, "Cl": 1, "As": 3, "Br": 1, "I": 1, "Hg": [1, 2], "Bi": [3, 5]}


def read_xyz_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        n = int(lines[i])
        score_line = lines[i + 1]
        score = float(score_line.split(":", 1)[1]) if score_line.startswith("score:") else 0.0
        i += 2
        atoms, points = [], []
        for _ in range(n):
            symbol, x, y, z = lines[i].split()[:4]
            atoms.append(symbol)
            points.append([float(x), float(y), float(z)])
            i += 1
        rows.append((np.asarray(points, dtype=np.float32), atoms, score))
    return rows


def get_bond_order(a, b, distance, single_bond=False):
    d = 100.0 * float(distance)
    if a not in BONDS1 or b not in BONDS1[a]:
        return 0
    if d >= BONDS1[a][b] + 10:
        return 0
    if a in BONDS2 and b in BONDS2[a] and d < BONDS2[a][b] + 5:
        if a in BONDS3 and b in BONDS3[a] and d < BONDS3[a][b] + 3:
            return 1 if single_bond else 3
        return 1 if single_bond else 2
    return 1


@contextlib.contextmanager
def _quiet_rdkit():
    tmp = tempfile.SpooledTemporaryFile()
    out, err = os.dup(1), os.dup(2)
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    try:
        yield
    finally:
        try:
            os.dup2(out, 1)
            os.dup2(err, 2)
        finally:
            os.close(out)
            os.close(err)
            tmp.close()


def _molecule(points, atoms, single_bond=False):
    points = np.asarray(points, dtype=np.float32)
    bond_count = np.zeros(len(atoms), dtype=np.int32)
    mol = Chem.RWMol()
    for a in atoms:
        mol.AddAtom(Chem.Atom(a))
    bond_types = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
    for i in range(len(atoms)):
        for j in range(i):
            order = get_bond_order(atoms[i], atoms[j], np.linalg.norm(points[i] - points[j]), single_bond=single_bond)
            if order:
                bond_count[i] += order; bond_count[j] += order
                mol.AddBond(i, j, bond_types[int(order)])
    stable = 0
    for atom, count in zip(atoms, bond_count):
        allowed = ALLOWED.get(atom, 0)
        stable += int(count in allowed if isinstance(allowed, list) else count == allowed)
    return mol, stable, len(atoms)


def _smiles(mol):
    def mol_to_smiles(m):
        try:
            Chem.SanitizeMol(m)
        except ValueError:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    with _quiet_rdkit():
        if mol_to_smiles(mol) is None:
            return None
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest = max(frags, default=mol, key=lambda x: x.GetNumAtoms())
        return mol_to_smiles(largest)


def _metric_summary(rows, valid_single_bond):
    total_atoms = stable_atoms = stable_mols = valid = 0
    smiles = []
    for row in rows:
        points, atoms = row[:2]
        mol, stable, n = _molecule(points, atoms, single_bond=False)
        total_atoms += n
        stable_atoms += stable
        stable_mols += int(stable == n and n > 0)
        valid_mol = _molecule(points, atoms, single_bond=True)[0] if valid_single_bond else mol
        s = _smiles(valid_mol)
        valid += int(s is not None)
        if s is not None:
            smiles.append(s)
    count = max(len(rows), 1)
    return {
        "atom_stable": stable_atoms / max(total_atoms, 1),
        "mol_stable": stable_mols / count,
        "valid": valid / count,
        "unique": len(set(smiles)) / max(len(smiles), 1),
    }


def _rerank_rows(rows, rerank_ratio, rerank_chunk):
    selected = []
    for start in range(0, len(rows), rerank_chunk):
        chunk = rows[start : start + rerank_chunk]
        keep = max(1, int(math.ceil(len(chunk) * rerank_ratio)))
        selected.extend(sorted(chunk, key=lambda row: row[2] if len(row) > 2 else 0.0, reverse=True)[:keep])
    return selected


def compute_molecule_metrics(rows, dataset="qm9", rerank_chunk=512):
    valid_single_bond = str(dataset).lower() == "drug"
    rerank_ratio = 0.2 if str(dataset).lower() == "drug" else 0.1
    metrics = _metric_summary(rows, valid_single_bond)
    rerank_metrics = _metric_summary(_rerank_rows(rows, rerank_ratio, rerank_chunk), valid_single_bond)
    metrics.update({
        "rerank_atom_stable": rerank_metrics["atom_stable"],
        "rerank_mol_stable": rerank_metrics["mol_stable"],
        "rerank_valid": rerank_metrics["valid"],
        "rerank_unique": rerank_metrics["unique"],
    })
    return metrics


def evaluate_xyz_file(path, dataset="qm9", rerank_chunk=512):
    return compute_molecule_metrics(read_xyz_file(path), dataset, rerank_chunk)
