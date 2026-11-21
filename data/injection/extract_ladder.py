#!/usr/bin/env python3
"""[provenance] Build ladder.csv: the step-140 injection ladder, both conditionings.

Reads the experiment tree this work came from; it cannot run on a fresh clone.

The experiment: take the crystal ligand, rotate ONLY the hindered axis to each of 13
angles, overwrite the ligand's coordinates at denoising step 140 of 200, and let the
sampler finish. 20 replicates an angle. The readout is the final axial torsion, which
says which rotamer the trajectory committed to -- so the fraction landing in the
crystal basin, as a function of the angle injected, is a committor.

Two conditionings, the same ladder:

    intact    the trunk's ligand rows untouched
    ablated   those rows replaced with matched Gaussian noise

The no-injection references for both conditionings live in data/tcp/arm_poses.csv as
base_noinj and slig_noinj; they are not duplicated here.
"""
from __future__ import annotations
import argparse, csv, hashlib, pathlib

SOURCES = {
    "intact":  ("exp116_7ftm_coordinate_injection/lddt_pli_torsion.csv", "tors_s140_"),
    "ablated": ("exp117_8sge_slig_torsion_scan/lddt_pli_torsion_slig.csv", "slig_s140_"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the reports/ directory to read")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "ladder.csv"))
    args = ap.parse_args()
    src = pathlib.Path(args.source)

    rows, digests = [], {}
    for conditioning, (rel, prefix) in SOURCES.items():
        path = src / rel
        digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for row in csv.DictReader(path.open()):
            if not row["run"].startswith(prefix):
                continue
            rows.append({"conditioning": conditioning,
                         "injected_torsion_deg": f'{float(row["injected_torsion"]):.1f}',
                         "pose_index": row["pose_index"],
                         "final_torsion_deg": row["final_torsion"],
                         "lddt_pli": row["lddt_pli"]})

    rows.sort(key=lambda r: (r["conditioning"], float(r["injected_torsion_deg"]),
                             int(r["pose_index"])))
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    angles = {r["injected_torsion_deg"] for r in rows}
    print(f"{len(rows)} poses, {len(angles)} injected angles, 2 conditionings -> {args.out}")
    for rel, digest in sorted(digests.items()):
        print(f"  {digest}  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
