#!/usr/bin/env python3
"""[reproduce] Score 8SGE runs and export the compact bundle the inspect tier reads.

For every run directory it finds, this computes

  * the symmetry-corrected ligand RMSD to the crystal, after superposing the
    protein (never the ligand) onto the reference, and
  * the axial torsion and which side of the hindered axis the pose sits on.

It writes two files. `results/8sge_per_pose.csv` is one row per draw. The pose
bundle `data/poses/8sge_poses.npz` holds the protein-aligned ligand coordinates
themselves, which is what lets the figure and the viewer work on a clone that has
no run directories and no GPU.

    python scripts/score_8sge.py --runs runs
    python scripts/score_8sge.py --runs /elsewhere/runs --include '^(control|gauss-s_lig)'
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cofolding_tts import axis as AX          # noqa: E402
from cofolding_tts import score as S          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference" / "8sge"
SAMPLE = "8sge__1__1.A__1.H__1.H"
PROTEIN_CHAIN = "1.A"


def sequence_from_fasta(path: Path) -> str:
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    out, taking = [], False
    for line in lines:
        if line.startswith(">"):
            if taking:
                break
            taking = True
            continue
        if taking:
            out.append(line)
    if not out:
        raise ValueError(f"no sequence in {path}")
    return "".join(out)


def arm_of(run_name: str) -> str:
    """The arm a run belongs to: the name with the draw and lambda suffixes removed.

    Order matters. The draw suffix is last, so it has to come off before the
    lambda suffix is at the end of the string and can be matched.
    """

    stripped = re.sub(r"[_-]d\d+$", "", run_name)
    stripped = re.sub(r"_lam\d+p\d+$", "", stripped)
    return stripped or run_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--include", default=None,
                    help="regex; only run directories whose name matches are scored")
    ap.add_argument("--out-csv", default=str(ROOT / "results" / "8sge_per_pose.csv"))
    ap.add_argument("--out-npz", default=str(ROOT / "data" / "poses" / "8sge_poses.npz"))
    args = ap.parse_args()

    pattern = re.compile(args.include) if args.include else None
    run_dirs = sorted(
        d.parent for d in Path(args.runs).glob("*/boltz_results_inputs")
        if pattern is None or pattern.search(d.parent.name)
    )
    if not run_dirs:
        print(f"no run directories under {args.runs}", file=sys.stderr)
        return 1

    sequence = sequence_from_fasta(REFERENCE / "sequences.fasta")
    ref_mol, ref_xyz = S.reference_ligand_molecule(REFERENCE / "ligand_1H.sdf")

    # The torsion is defined by deposited atom name; map those onto SDF order once.
    names = AX.crystal_atom_names(ref_xyz, REFERENCE / "system.cif")
    atom_names = [""] * ref_mol.GetNumAtoms()
    for deposited, index in names.items():
        atom_names[index] = deposited
    sp3 = names[AX.SP3_ATOM]
    torsion = [names[n] for n in AX.TORSION_ATOMS]
    cry_cip, cry_torsion = AX.label_pose(ref_mol, ref_xyz, sp3, torsion)
    if cry_cip != AX.CRYSTAL_SP3_CIP or abs(cry_torsion - AX.CRYSTAL_TORSION) > 0.5:
        raise SystemExit(
            f"crystal self-check failed: sp3={cry_cip} torsion={cry_torsion:+.1f}; "
            f"expected {AX.CRYSTAL_SP3_CIP} and {AX.CRYSTAL_TORSION:+.1f}"
        )
    print(f"crystal: sp3 {AX.SP3_ATOM}={cry_cip}, "
          f"{'-'.join(AX.TORSION_ATOMS)}={cry_torsion:+.1f} deg")

    rows, coords, kept = [], [], []
    for run in run_dirs:
        try:
            mol, xyz = S.predicted_ligand_coordinates(
                run / "boltz_results_inputs", SAMPLE,
                REFERENCE / "receptor.cif", sequence, PROTEIN_CHAIN)
            rmsd, mapping, mode = S.ligand_rmsd(mol, xyz, ref_mol, ref_xyz)
            # Reorder the pose into reference-SDF order so the torsion indices,
            # which are defined on the reference, address the right atoms.
            mappings, _ = S.graph_mappings(mol, ref_mol)
            # A symmetric ligand can admit several mappings. If they disagree about
            # the torsion then the axial label is not well defined for this pose,
            # and saying so is better than silently picking one.
            torsions, cips = set(), set()
            for candidate in mappings:
                reordered = np.empty_like(ref_xyz)
                reordered[np.asarray(candidate, int)] = xyz
                c, tt = AX.label_pose(ref_mol, reordered, sp3, torsion)
                torsions.add(round(tt, 1)); cips.add(c)
            ambiguous = int(len(torsions) > 1 or len(cips) > 1)
            tor = sorted(torsions)[0] if len(torsions) == 1 else float("nan")
            cip = sorted(cips)[0] if len(cips) == 1 else "?"
            in_ref_order = np.empty_like(ref_xyz)
            in_ref_order[np.asarray(mapping, int)] = xyz
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {run.name}: SKIPPED ({exc})", file=sys.stderr)
            continue
        rows.append({
            "run": run.name,
            "arm": arm_of(run.name),
            "ligand_rmsd_angstrom": round(float(rmsd), 3),
            "placed_under_2A": int(rmsd < 2.0),
            "axial_torsion_deg": round(tor, 1),
            "axis_correct": int(AX.axis_correct(tor)) if tor == tor else -1,
            "sp3_cip": cip,
            "n_graph_mappings": len(mappings),
            "mapping_ambiguous": ambiguous,
            "mapping_mode": mode,
        })
        coords.append(in_ref_order)
        kept.append(run.name)

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    out_npz = Path(args.out_npz); out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        run=np.array(kept),
        arm=np.array([r["arm"] for r in rows]),
        coords=np.array(coords, dtype=np.float32),
        crystal=ref_xyz.astype(np.float32),
        rmsd=np.array([r["ligand_rmsd_angstrom"] for r in rows], dtype=np.float32),
        torsion=np.array([r["axial_torsion_deg"] for r in rows], dtype=np.float32),
        axis_correct=np.array([r["axis_correct"] for r in rows], dtype=np.int8),
        atom_names=np.array(atom_names),
        elements=np.array([ref_mol.GetAtomWithIdx(i).GetSymbol()
                           for i in range(ref_mol.GetNumAtoms())]),
    )
    print(f"scored {len(rows)} runs -> {out_csv}")
    print(f"pose bundle ({np.array(coords).shape}) -> {out_npz} "
          f"[{out_npz.stat().st_size / 1024:.0f} KB]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
