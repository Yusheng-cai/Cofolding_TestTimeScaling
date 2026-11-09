"""Axial (atropisomer) stereochemistry: perceiving the axis, and reading it off a pose.

Two separate jobs live here.

**Perception** (`find_axis_candidates`, `has_axis`) answers "does this ligand have a
stereogenic axis at all?" from the molecular graph alone. The code below is carried
over unchanged from the original implementation and its behaviour is pinned by tests.

**Labelling** (`label_pose`, `axis_correct`) answers "which way round is this
particular pose?" by measuring the torsion in 3D. For 8SGE the axis is the
C12-C11-N1-C3 dihedral and the crystal sits at -101.4 degrees; a pose is on the
crystal's side when its torsion is within 90 degrees of that, measured circularly.

The original perception rationale follows verbatim.
Detect candidate stereogenic axes (atropisomer axes) in a ligand.

Why this exists: nothing in the standard pipeline perceives one. RDKit 2026.03
defines a ``Bond_Atropisomer`` stereo type, but ``FindPotentialStereo`` does not
flag an axis from a SMILES string even for textbook cases - BINOL and
2,2'-dinitro-6,6'-dimethylbiphenyl both come back with no stereo found. The
descriptor only appears once the configuration is already specified by wedge
bonds or 3D coordinates. So a ligand whose SMILES omits the axial configuration
is indistinguishable, to the toolkit, from one that has no axis at all.

That is the same blind spot exp096 found in the models, and it is why 8SGE's
``(5P)`` survives only in the deposited chemical name. This module perceives the
axis from the molecular graph alone, so a benchmark can be screened for ligands
that could behave the same way.

What counts as a candidate axis, in order:

  1. a SINGLE bond, not itself in a ring,
  2. joining two atoms that are each aromatic and in a ring - a biaryl, N-aryl
     or heteroaryl-aryl linkage,
  3. with two ortho positions on each side (so the sides can be compared), and
  4. NEITHER side locally symmetric. If an atom's two ortho neighbours are
     equivalent by canonical rank, rotating the axis by 180 degrees maps the
     molecule onto itself and the axis is not stereogenic. This is what
     correctly rejects biphenyl and 2-methylbiphenyl.

Rule 4 decides stereogenicity and is exact. Whether an axis is actually
*hindered* enough to give separable atropisomers is a question about a rotation
barrier, which a graph cannot answer; ``blocked_ortho`` is reported as a crude
steric proxy (how many of the four ortho positions carry no hydrogen) and must
not be read as a barrier. Classic C-C atropisomers usually need three or four;
N-aryl and five-membered-ring cases can be hindered with fewer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from rdkit import Chem


@dataclass(frozen=True)
class AxisCandidate:
    """One candidate stereogenic axis."""

    bond_idx: int
    begin_idx: int
    end_idx: int
    kind: str                 # e.g. "C-C", "N-C"
    blocked_ortho: int        # 0-4 ortho positions bearing no hydrogen
    begin_ring_size: int
    end_ring_size: int
    fused_either_side: bool   # an ortho position is a ring-fusion atom

    def as_dict(self) -> dict:
        return asdict(self)


def _ortho(atom: Chem.Atom, partner: Chem.Atom) -> list[Chem.Atom]:
    """The ring neighbours of `atom` other than its axis partner."""
    return [n for n in atom.GetNeighbors() if n.GetIdx() != partner.GetIdx()]


def _smallest_ring(mol: Chem.Mol, idx: int) -> int:
    sizes = [len(r) for r in mol.GetRingInfo().AtomRings() if idx in r]
    return min(sizes) if sizes else 0


def find_axis_candidates(mol: Chem.Mol) -> list[AxisCandidate]:
    """Candidate stereogenic axes, perceived from the graph alone.

    Hydrogens may be implicit or explicit; counts use ``GetTotalNumHs``.
    """
    if mol is None:
        return []
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    ring_info = mol.GetRingInfo()
    found: list[AxisCandidate] = []

    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (a.IsInRing() and b.IsInRing()):
            continue
        if not (a.GetIsAromatic() and b.GetIsAromatic()):
            continue

        blocked, fused, symmetric, usable = 0, False, False, True
        for x, y in ((a, b), (b, a)):
            nb = _ortho(x, y)
            if len(nb) != 2:
                usable = False
                break
            if ranks[nb[0].GetIdx()] == ranks[nb[1].GetIdx()]:
                symmetric = True
            for o in nb:
                if o.GetTotalNumHs() == 0:
                    blocked += 1
                if ring_info.NumAtomRings(o.GetIdx()) > 1:
                    fused = True
        if not usable or symmetric:
            continue

        found.append(AxisCandidate(
            bond_idx=bond.GetIdx(),
            begin_idx=a.GetIdx(), end_idx=b.GetIdx(),
            kind=f"{a.GetSymbol()}-{b.GetSymbol()}",
            blocked_ortho=blocked,
            begin_ring_size=_smallest_ring(mol, a.GetIdx()),
            end_ring_size=_smallest_ring(mol, b.GetIdx()),
            fused_either_side=fused,
        ))
    return found


def has_axis(smiles: str) -> bool:
    """Convenience wrapper: does this SMILES contain any candidate axis?"""
    mol = Chem.MolFromSmiles(smiles)
    return bool(find_axis_candidates(mol))

# --- 8SGE axial convention, carried over from the four-model comparison -----
# The deposited ligand is ZWS. The torsion below is the hindered axis; the sp3
# centre is reported alongside it so a pose that flips the axis can be told apart
# from one that inverts the ordinary tetrahedral centre.
CRYSTAL_TORSION = -101.4          # degrees, C12-C11-N1-C3
SP3_ATOM = "C7"
TORSION_ATOMS = ("C12", "C11", "N1", "C3")
CRYSTAL_SP3_CIP = "R"


def crystal_atom_names(reference_xyz, system_cif) -> dict[str, int]:
    """Map deposited ZWS atom names onto reference-SDF atom order.

    The SDF carries coordinates but not the deposited atom names, and the torsion
    is defined by name. Both describe the same molecule in the same frame, so the
    nearest deposited atom to each SDF atom gives the correspondence.
    """

    import gemmi
    import numpy as np

    structure = gemmi.read_structure(str(system_cif))
    structure.setup_entities()
    deposited = [a for ch in structure[0] for r in ch if r.name == "ZWS" for a in r]
    if not deposited:
        raise ValueError(f"no ZWS residue in {system_cif}")
    positions = np.array([a.pos.tolist() for a in deposited])
    out: dict[str, int] = {}
    for i in range(len(reference_xyz)):
        j = int(np.argmin(np.linalg.norm(positions - reference_xyz[i], axis=1)))
        out[deposited[j].name] = i
    return out


def label_pose(mol, xyz, sp3_index: int, torsion_indices) -> tuple[str, float]:
    """(sp3 CIP code, torsion in degrees) for one pose, both read from the geometry."""

    from rdkit.Chem import rdMolTransforms as rmt

    work = Chem.Mol(mol)
    conformer = Chem.Conformer(work.GetNumAtoms())
    for i, position in enumerate(xyz):
        conformer.SetAtomPosition(i, [float(v) for v in position])
    work.RemoveAllConformers()
    work.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(work)
    cip = work.GetAtomWithIdx(sp3_index).GetPropsAsDict().get("_CIPCode", "?")
    return cip, float(rmt.GetDihedralDeg(work.GetConformer(), *torsion_indices))


def axis_correct(torsion_degrees: float, crystal_degrees: float = CRYSTAL_TORSION) -> bool:
    """Is this pose on the crystal's side of the axis?

    Compared circularly, so a pose at +179 and one at -179 are two degrees apart,
    not 358.
    """

    return abs((torsion_degrees - crystal_degrees + 180.0) % 360.0 - 180.0) < 90.0
