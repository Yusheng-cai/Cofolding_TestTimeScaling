#!/usr/bin/env python3
"""[provenance] Build arm_poses.csv: the five 8SGE arms, one row per pose.

Like data/baselines/extract_baselines.py, this reads the experiment tree this work
came from and cannot run on a fresh clone. --source points at its reports/ directory.

The five arms differ one factor at a time, which is the only reason they can be read
against each other:

    base_noinj       no intervention
    slig_noinj       our ablation: the ligand rows of s_trunk replaced with matched
                     Gaussian noise, drawn once per run
    tcp_c11          Boltz-Perturb's C11: s perturbed over steps 80-159, redrawn
                     every step
    tcp_c11_frozen   C11 with the direction drawn ONCE and reused
    tcp_c10          C11's magnitude over the whole trajectory

Every arm shares seed 42 and every Boltz flag, so the per-pose atom-noise draws match
across arms and a difference is the intervention rather than the sampling.
"""
from __future__ import annotations
import argparse, csv, hashlib, pathlib

# arm -> (directory holding per_pose_metrics.csv, torsion table, key in that table)
ARMS = {
    "base_noinj":     ("exp117_8sge_slig_torsion_scan/base_noinj",
                       "exp117_8sge_slig_torsion_scan/lddt_pli_torsion_slig.csv", "base_noinj"),
    "slig_noinj":     ("exp117_8sge_slig_torsion_scan/slig_noinj",
                       "exp117_8sge_slig_torsion_scan/lddt_pli_torsion_slig.csv", "slig_noinj"),
    "tcp_c11":        ("exp118_8sge_tcp_replication/tcp_c11",
                       "exp118_8sge_tcp_replication/lddt_pli_torsion.csv", "tcp_c11"),
    "tcp_c11_frozen": ("exp118_8sge_tcp_replication/tcp_c11_frozen",
                       "exp118_8sge_tcp_replication/lddt_pli_torsion.csv", "tcp_c11_frozen"),
    "tcp_c10":        ("exp118_8sge_tcp_replication/tcp_c10",
                       "exp118_8sge_tcp_replication/lddt_pli_torsion.csv", "tcp_c10"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the reports/ directory to read")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "arm_poses.csv"))
    args = ap.parse_args()
    src = pathlib.Path(args.source)

    out_rows, digests = [], {}
    for arm, (run_dir, torsion_rel, key) in ARMS.items():
        metrics = src / run_dir / "per_pose_metrics.csv"
        torsion_path = src / torsion_rel
        digests[f"{run_dir}/per_pose_metrics.csv"] = hashlib.sha256(
            metrics.read_bytes()).hexdigest()[:16]
        digests[torsion_rel] = hashlib.sha256(torsion_path.read_bytes()).hexdigest()[:16]

        angles = {r["pose_index"]: r for r in csv.DictReader(torsion_path.open())
                  if r["run"] == key}
        for row in csv.DictReader(metrics.open()):
            extra = angles.get(row["pose_index"])
            if extra is None:
                raise SystemExit(f"{arm}: no torsion for pose {row['pose_index']}")
            out_rows.append({
                "arm": arm,
                "pose_index": row["pose_index"],
                "ligand_rmsd_angstrom": f'{float(row["ligand_rmsd_angstrom"]):.3f}',
                "final_torsion_deg": extra["final_torsion"],
                "lddt_pli": extra["lddt_pli"],
                "confidence_score": f'{float(row["confidence_score"]):.4f}',
            })

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"{len(out_rows)} poses across {len(ARMS)} arms -> {args.out}")
    for rel, digest in sorted(digests.items()):
        print(f"  {digest}  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
