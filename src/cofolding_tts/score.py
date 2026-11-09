"""Symmetry-corrected ligand RMSD against a crystal reference, after protein alignment.

This is the narrow path the 8SGE rescue needs, lifted from a much larger
multi-system scorer. Two deliberate simplifications:

* **One protein chain.** The original ran a global pairwise sequence alignment to
  decide which predicted chain was the protein. For a single known system that is
  unnecessary, and dropping it removes a biopython dependency. Here the sole
  protein chain is taken, and more than one is an error.
* **No manifest layer.** Paths are passed explicitly rather than resolved through a
  benchmark manifest.

What is NOT simplified is the measurement itself. The ligand RMSD is computed after
superposing the *protein* on the reference, never by fitting the ligand to itself,
because a ligand-only fit would hide exactly the placement error being studied. It
is minimised over every complete graph automorphism of the ligand, so a symmetric
group that can be relabelled does not inflate the number.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem

from .pose import apply_transform, kabsch, minimum_mapped_rmsd

__all__ = [
    "best_atoms",
    "one_letter",
    "chain_coordinates",
    "reference_chain",
    "reference_ligand_molecule",
    "graph_mappings",
    "molecule_coordinates",
    "remove_tokenization_hydrogens",
    "sole_protein_chain",
    "predicted_ligand_coordinates",
    "ligand_rmsd",
]

LIGAND_RESNAME = "LIG1"


def atom_position(atom: gemmi.Atom) -> np.ndarray:
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)


def best_atoms(residue: gemmi.Residue, *, heavy_only: bool = False) -> dict[str, gemmi.Atom]:
    """Choose the highest-occupancy alternate location for each atom name."""

    selected: dict[str, gemmi.Atom] = {}
    for atom in residue:
        if heavy_only and atom.element.atomic_number == 1:
            continue
        name = atom.name.strip()
        current = selected.get(name)
        if current is None or float(atom.occ) > float(current.occ):
            selected[name] = atom
    return selected


def one_letter(residue: gemmi.Residue) -> str:
    info = gemmi.find_tabulated_residue(residue.name)
    code = info.one_letter_code if info else None
    return code if code and code != " " else "X"


def chain_coordinates(chain: gemmi.Chain, sequence: str) -> np.ndarray:
    """Sequence-indexed C-alpha coordinates; NaN where the residue is absent."""

    ca = np.full((len(sequence), 3), np.nan, dtype=float)
    for residue in chain:
        index = residue.seqid.num - 1
        if index < 0 or index >= len(sequence):
            continue
        # Gemmi uses lowercase one-letter codes for several modified residues, so
        # compare the parent identity case-insensitively.
        observed = one_letter(residue).upper()
        if observed != "X" and observed != sequence[index]:
            raise ValueError(
                f"sequence mismatch at {chain.name}:{index + 1}: "
                f"expected={sequence[index]} observed={observed}"
            )
        atoms = best_atoms(residue, heavy_only=True)
        if "CA" in atoms:
            ca[index] = atom_position(atoms["CA"])
    return ca


def reference_chain(model: gemmi.Model, chain_id: str) -> gemmi.Chain:
    matches = [chain for chain in model if chain.name == chain_id]
    if len(matches) != 1:
        raise ValueError(f"expected one reference chain {chain_id}, found {len(matches)}")
    return matches[0]


def sole_protein_chain(model: gemmi.Model) -> gemmi.Chain:
    """The single protein chain of a prediction. More than one is an error here."""

    chains = [
        chain for chain in model
        if any(
            (info := gemmi.find_tabulated_residue(r.name)) is not None and info.is_amino_acid()
            for r in chain
        )
    ]
    if len(chains) != 1:
        raise ValueError(f"expected exactly one protein chain, found {len(chains)}")
    return chains[0]


def sole_ligand_residue(model: gemmi.Model, resname: str = LIGAND_RESNAME) -> gemmi.Residue:
    residues = [r for chain in model for r in chain if r.name == resname]
    if len(residues) != 1:
        raise ValueError(f"expected one {resname} residue, found {len(residues)}")
    return residues[0]


def reference_ligand_molecule(path: Path | str) -> tuple[Chem.Mol, np.ndarray]:
    molecule = Chem.MolFromMolFile(str(path), removeHs=True, sanitize=True)
    if molecule is None:
        raise ValueError(f"RDKit could not parse {path}")
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    coordinates = np.asarray(
        [tuple(conformer.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())],
        dtype=float,
    )
    return molecule, coordinates


def remove_tokenization_hydrogens(molecule: Chem.Mol) -> Chem.Mol:
    """Remove every atom the tokenizer skips, keeping heavy-atom order.

    The tokenizer drops all hydrogens, including an explicit one used only to
    define double-bond stereochemistry, which RDKit would otherwise keep.
    """

    parameters = Chem.RemoveHsParameters()
    parameters.removeDefiningBondStereo = True
    heavy = Chem.RemoveHs(molecule, parameters, sanitize=False)
    Chem.SanitizeMol(heavy)
    return heavy


def _topology_copy(molecule: Chem.Mol) -> Chem.Mol:
    result = Chem.RWMol(Chem.Mol(molecule))
    for atom in result.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
    for bond in result.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.SINGLE)
    return result.GetMol()


def graph_mappings(predicted: Chem.Mol, reference: Chem.Mol) -> tuple[list[tuple[int, ...]], str]:
    """Every complete atom correspondence between the two molecules.

    Falls back to element-and-connectivity matching when bond orders disagree,
    which happens when a toolkit perceives aromaticity differently; the mode is
    returned so the caller can record which comparison was actually made.
    """

    if predicted.GetNumAtoms() != reference.GetNumAtoms():
        raise ValueError(
            "ligand heavy-atom counts differ: "
            f"predicted {predicted.GetNumAtoms()}, reference {reference.GetNumAtoms()}"
        )
    matches = reference.GetSubstructMatches(
        predicted, useChirality=True, uniquify=False, maxMatches=100000
    )
    mode = "bond_order_and_chirality"
    if not matches:
        matches = _topology_copy(reference).GetSubstructMatches(
            _topology_copy(predicted), useChirality=True, uniquify=False, maxMatches=100000
        )
        mode = "element_connectivity_and_chirality"
    full = [tuple(m) for m in matches if len(m) == predicted.GetNumAtoms()]
    if not full:
        raise ValueError("no complete chirality-compatible ligand graph mapping")
    return full, mode


def molecule_coordinates(molecule: Chem.Mol, residue: gemmi.Residue) -> np.ndarray:
    """Coordinates of `residue`, reordered into `molecule`'s atom order by name."""

    atoms = best_atoms(residue, heavy_only=True)
    missing = [
        molecule.GetAtomWithIdx(i).GetProp("name")
        for i in range(molecule.GetNumAtoms())
        if molecule.GetAtomWithIdx(i).GetProp("name") not in atoms
    ]
    if missing:
        raise ValueError(f"residue is missing ligand atoms: {missing}")
    return np.array([
        atom_position(atoms[molecule.GetAtomWithIdx(i).GetProp("name")])
        for i in range(molecule.GetNumAtoms())
    ])


def predicted_ligand_coordinates(
    prediction_root: Path, sample_id: str, reference_receptor: Path, sequence: str,
    reference_chain_id: str, *, pose_index: int = 0,
) -> tuple[Chem.Mol, np.ndarray]:
    """The predicted ligand, superposed into the reference frame on protein C-alphas.

    Returns the predicted molecule (for graph matching) and its aligned coordinates
    in the predicted molecule's own atom order.
    """

    prediction_root = Path(prediction_root)
    cif = prediction_root / "predictions" / sample_id / f"{sample_id}_model_{pose_index}.cif"
    structure = gemmi.read_structure(str(cif))
    model = structure[0]
    predicted_ca = chain_coordinates(sole_protein_chain(model), sequence)

    reference_structure = gemmi.read_structure(str(reference_receptor))
    reference_ca = chain_coordinates(
        reference_chain(reference_structure[0], reference_chain_id), sequence
    )
    usable = np.all(np.isfinite(predicted_ca), axis=1) & np.all(np.isfinite(reference_ca), axis=1)
    if usable.sum() < 3:
        raise ValueError(f"only {usable.sum()} aligned C-alpha pairs; need at least 3")
    rotation, translation, _ = kabsch(predicted_ca[usable], reference_ca[usable])

    with (prediction_root / "processed" / "mols" / f"{sample_id}.pkl").open("rb") as handle:
        molecule = remove_tokenization_hydrogens(pickle.load(handle)[LIGAND_RESNAME])  # noqa: S301
    coordinates = molecule_coordinates(molecule, sole_ligand_residue(model))
    return molecule, apply_transform(coordinates, rotation, translation)


def ligand_rmsd(
    predicted_molecule: Chem.Mol, aligned_coordinates: np.ndarray,
    reference_molecule: Chem.Mol, reference_coordinates: np.ndarray,
) -> tuple[float, tuple[int, ...], str]:
    """Symmetry-corrected ligand RMSD, with no ligand-only fit.

    `fit_mobile=False` is the point: the protein alignment has already been applied
    and fitting the ligand to itself would measure conformation rather than
    placement.
    """

    mappings, mode = graph_mappings(predicted_molecule, reference_molecule)
    value, mapping = minimum_mapped_rmsd(
        aligned_coordinates, reference_coordinates, mappings, fit_mobile=False
    )
    return value, mapping, mode
