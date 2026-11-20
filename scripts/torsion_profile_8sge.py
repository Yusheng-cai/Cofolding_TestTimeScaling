"""[inspect] MMFF relaxed torsion profile about the 8SGE hindered axis, isolated ligand.

One question: is Boltz-2's +60.6 deg attractor a physical minimum of this bond, or
off the minima -- the latter would say the model applies a generic rotatable-bond
prior instead of modelling a hindered axis. Atom indices come from the same
crystal_atom_names helper every exp096/exp097 scorer uses, so the dihedral is the
identical four atoms as every torsion number in the reports.
"""
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms as rmt

sys.path.insert(0, "src")
from cofolding_tts import axis as AX
from cofolding_tts import score as S

REF = Path("data/reference/8sge")
ref_mol, ref_xyz = S.reference_ligand_molecule(REF / "ligand_1H.sdf")
names = AX.crystal_atom_names(ref_xyz, REF / "system.cif")
q = [names[n] for n in AX.TORSION_ATOMS]
print(f"dihedral {'-'.join(AX.TORSION_ATOMS)} = SDF indices {q}")

mol = Chem.AddHs(Chem.Mol(ref_mol), addCoords=True)   # H appended, heavy idx preserved
print(f"crystal torsion read back: {rmt.GetDihedralDeg(mol.GetConformer(), *q):+.2f} deg "
      f"(expected {AX.CRYSTAL_TORSION:+.1f})\n")

rows = []
for target in range(-180, 181, 5):
    m = Chem.Mol(mol)
    rmt.SetDihedralDeg(m.GetConformer(), *q, float(target))
    props = AllChem.MMFFGetMoleculeProperties(m)
    ff = AllChem.MMFFGetMoleculeForceField(m, props)
    ff.MMFFAddTorsionConstraint(*q, False, float(target) - 0.05, float(target) + 0.05, 1e4)
    ff.Minimize(maxIts=4000)
    rows.append((target, ff.CalcEnergy()))

T = np.array([r[0] for r in rows], float)
E = np.array([r[1] for r in rows], float); E -= E.min()

mins = [i for i in range(len(E)) if E[i] < E[(i-1) % len(E)] and E[i] < E[(i+1) % len(E)] and E[i] < 5]
print("minima below 5 kcal/mol: " + (", ".join(f"{T[i]:+.0f} ({E[i]:.2f})" for i in mins) or "none"))
print(f"barrier (max of profile): {E.max():.1f} kcal/mol\n")
for label, val in (("crystal", AX.CRYSTAL_TORSION), ("Boltz-2 attractor", 60.6),
                   ("frozen input conformer", 101.78)):
    print(f"  {label:24} {val:+8.2f} deg -> MMFF {np.interp(val, T, E):6.2f} kcal/mol")
print(f"\n{'deg':>6} {'kcal':>7}")
for t, e in zip(T, E):
    mark = ""
    for label, val in (("crystal", AX.CRYSTAL_TORSION), ("BOLTZ", 60.6), ("input", 101.78)):
        if abs(t - val) <= 2.5:
            mark = "  <-- " + label
    print(f"{t:+6.0f} {e:7.2f}  {'#' * int(round(min(e, 40)))}{mark}")

# --- recorded output -------------------------------------------------------
import csv as _csv
_out = Path("results/8sge_torsion_profile.csv")
_out.parent.mkdir(parents=True, exist_ok=True)
with _out.open("w", newline="") as fh:
    w = _csv.writer(fh); w.writerow(["torsion_deg", "mmff_kcal_above_min"])
    w.writerows([[int(t), round(float(e), 3)] for t, e in zip(T, E)])
print(f"\nwrote {_out}")
