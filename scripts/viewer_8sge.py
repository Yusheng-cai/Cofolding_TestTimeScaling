#!/usr/bin/env python3
"""[inspect] Build a standalone 3D page showing the two 8SGE placements in the pocket.

Writes one self-contained HTML file. It needs network access only when opened, to
pull the viewer library from a CDN; the structures themselves are embedded.

The page shows the crystal ligand, a representative rescued pose, and a
representative unperturbed pose, all superposed into the crystal receptor frame by
the protein alignment the scorer uses, with the pocket residues drawn around them.

    python scripts/viewer_8sge.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cofolding_tts.pdbio import record                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference" / "8sge"
POCKET_RADIUS = 5.0      # A from any crystal ligand atom
CONTEXT_RADIUS = 14.0    # A, for the backbone ribbon


def representative(rows, arm, want_axis_correct):
    """The pose of `arm` closest to the median RMSD of its axis-correct subset."""

    group = [r for r in rows
             if r["arm"] == arm and int(r["axis_correct"]) == want_axis_correct]
    if not group:
        return None
    values = sorted(float(r["ligand_rmsd_angstrom"]) for r in group)
    median = values[len(values) // 2]
    return min(group, key=lambda r: abs(float(r["ligand_rmsd_angstrom"]) - median))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", default=str(ROOT / "data" / "poses" / "8sge_poses.npz"))
    ap.add_argument("--per-pose", default=str(ROOT / "results" / "8sge_per_pose.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "8sge_viewer.html"))
    args = ap.parse_args()

    bundle = np.load(args.poses, allow_pickle=True)
    runs = list(bundle["run"])
    index = {r: i for i, r in enumerate(runs)}
    names = [str(n) for n in bundle["atom_names"]]
    elements = [str(e) for e in bundle["elements"]]
    crystal = bundle["crystal"]
    rows = list(csv.DictReader(open(args.per_pose)))

    chosen = [("crystal", "Crystal 8SGE", crystal, 0.0, "crystal")]
    for arm, axis_ok, label, kind in (
        ("gauss-s_lig", 1, "Rescued, gauss s_lig", "correct"),
        ("control", 0, "Unperturbed control", "wrong"),
    ):
        pick = representative(rows, arm, axis_ok)
        if pick is None:
            continue
        chosen.append((pick["run"], f"{label}", bundle["coords"][index[pick["run"]]],
                       float(pick["ligand_rmsd_angstrom"]), kind))

    structure = gemmi.read_structure(str(REFERENCE / "receptor.cif"))
    structure.remove_hydrogens()
    pocket, context = [], []
    for chain in structure[0]:
        for residue in chain:
            if residue.name == "HOH":
                continue
            pts = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in residue])
            if not len(pts):
                continue
            closest = np.sqrt(((pts[:, None] - crystal[None]) ** 2).sum(2)).min()
            if closest < POCKET_RADIUS:
                pocket.append(residue.seqid.num)
            if closest < CONTEXT_RADIUS:
                context.append(residue)

    serial, lines = 1, []
    for residue in context:
        for atom in residue:
            lines.append(record("ATOM", serial, atom.name, residue.name, "A",
                                residue.seqid.num,
                                (atom.pos.x, atom.pos.y, atom.pos.z),
                                element=atom.element.name))
            serial += 1
    protein_pdb = "\n".join(lines) + "\nTER\nEND\n"

    blocks, meta = [f'<script type="text/plain" id="pdb_protein">{protein_pdb}</script>'], []
    for i, (run, label, xyz, rmsd, kind) in enumerate(chosen, start=1):
        body = "\n".join(
            record("HETATM", j, names[j - 1] or elements[j - 1], "LIG", "L", 1, p,
                   bfactor=rmsd, element=elements[j - 1])
            for j, p in enumerate(xyz, start=1))
        blocks.append(f'<script type="text/plain" id="pdb_{i}">{body}\nEND\n</script>')
        meta.append({"id": i, "run": run, "label": label,
                     "rmsd": round(rmsd, 2), "kind": kind})

    template = (ROOT / "scripts" / "viewer_template.html").read_text()
    html = (template
            .replace("__DATABLOCKS__", "\n".join(blocks))
            .replace("__META__", json.dumps({"models": meta, "pocket": sorted(set(pocket))},
                                            separators=(",", ":"))))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} [{out.stat().st_size / 1024:.0f} KB], "
          f"{len(meta)} placements, {len(set(pocket))} pocket residues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
