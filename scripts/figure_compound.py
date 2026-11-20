#!/usr/bin/env python3
"""[inspect] Draw the 8SGE ligand and mark the bond the whole result turns on.

The compound is ZWS, as deposited with PDB 8SGE. Rotation about the N1-C11 bond
is hindered, so the two senses of the C12-C11-N1-C3 torsion are separate
conformers rather than two points on a freely turning bond. The crystal is at
-101.4 degrees. Which side of that bond a prediction lands on decides which way
round the molecule sits in the pocket.

    python scripts/figure_compound.py

Needs rdkit and gemmi; both are inspect-tier dependencies, no GPU and no model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem                                    # noqa: E402
from rdkit.Chem import AllChem, Draw                      # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D                   # noqa: E402

from cofolding_tts.axis import (                          # noqa: E402
    CRYSTAL_TORSION, SP3_ATOM, TORSION_ATOMS, crystal_atom_names,
)

INK = (0.165, 0.165, 0.157)
BLUE = (0.0, 0.447, 0.698)       # Okabe-Ito blue, the hindered axis
VERM = (0.835, 0.369, 0.0)       # Okabe-Ito vermillion, the sp3 centre
SURFACE = (0.988, 0.988, 0.980)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ref = ROOT / "data" / "reference" / "8sge"
    ap.add_argument("--sdf", default=str(ref / "ligand_1H.sdf"))
    ap.add_argument("--system-cif", default=str(ref / "system.cif"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "8sge_compound.png"))
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=440)
    args = ap.parse_args()

    mol = Chem.MolFromMolFile(args.sdf, removeHs=False)
    if mol is None:
        raise SystemExit(f"could not read {args.sdf}")
    xyz = mol.GetConformer().GetPositions()
    names = crystal_atom_names(xyz, args.system_cif)
    missing = [n for n in (*TORSION_ATOMS, SP3_ATOM) if n not in names]
    if missing:
        raise SystemExit(f"atom names absent from {args.system_cif}: {missing}")
    torsion = [names[n] for n in TORSION_ATOMS]
    sp3 = names[SP3_ATOM]

    # Draw the connectivity, not the crystal pose: a 2-D depiction says which bond
    # is hindered far more legibly than a projection of the 3-D coordinates.
    flat = Chem.RemoveHs(Chem.Mol(mol))
    heavy = {a.GetIdx(): i for i, a in enumerate(
        [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1])}
    # RemoveHs preserves the order of the heavy atoms, so the index of a heavy
    # atom in `flat` is its rank among heavy atoms in `mol`.
    rank = {}
    seen = 0
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1:
            rank[a.GetIdx()] = seen
            seen += 1
    torsion_flat = [rank[i] for i in torsion if i in rank]
    sp3_flat = rank.get(sp3)
    AllChem.Compute2DCoords(flat)

    axis_bond = flat.GetBondBetweenAtoms(torsion_flat[1], torsion_flat[2])
    if axis_bond is None:
        raise SystemExit("the two central torsion atoms are not bonded")

    for n in TORSION_ATOMS:
        i = rank.get(names[n])
        if i is not None:
            flat.GetAtomWithIdx(i).SetProp("atomNote", n)

    drawer = rdMolDraw2D.MolDraw2DCairo(args.width, args.height)
    opts = drawer.drawOptions()
    opts.setBackgroundColour(SURFACE)
    opts.addStereoAnnotation = True
    opts.annotationFontScale = 0.70
    opts.bondLineWidth = 2
    opts.highlightBondWidthMultiplier = 20
    opts.setHighlightColour(BLUE)
    highlight_atoms = list(torsion_flat) + ([sp3_flat] if sp3_flat is not None else [])
    colours = {i: BLUE for i in torsion_flat}
    if sp3_flat is not None:
        colours[sp3_flat] = VERM
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, flat,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=colours,
        highlightBonds=[axis_bond.GetIdx()],
        highlightBondColors={axis_bond.GetIdx(): BLUE})
    drawer.FinishDrawing()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(drawer.GetDrawingText())
    print(f"wrote {out}")
    print(f"  hindered bond      {TORSION_ATOMS[1]}-{TORSION_ATOMS[2]} "
          f"(heavy-atom indices {torsion_flat[1]}, {torsion_flat[2]})")
    print(f"  torsion            {'-'.join(TORSION_ATOMS)} = {CRYSTAL_TORSION}° in the crystal")
    print(f"  sp3 centre         {SP3_ATOM}")
    print(f"  heavy atoms        {flat.GetNumAtoms()}   formula "
          f"{Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
