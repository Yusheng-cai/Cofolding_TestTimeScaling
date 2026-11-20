#!/usr/bin/env python3
"""[provenance] Build control_poses.csv from the experiment tree this work came from.

This script does NOT run on a fresh clone: it reads the source experiments, which are
not part of this repository. It is committed so the committed CSV has a recipe rather
than only a description. --source points at the reports/ directory of that tree.

One row per UNPERTURBED draw. Nothing here is an intervention arm; these are the
baselines every arm is read against.
"""
from __future__ import annotations
import argparse, csv, hashlib, pathlib

# pdb id -> (source csv under reports/, row filter, rmsd column, draw column, note)
SOURCES = {
    "8sge": ("exp034_boltz2_8sge_multisample/per_pose_metrics.csv",
             None, "ligand_rmsd_angstrom", "pose_index",
             "25 model seeds, one pose each"),
    "8cjg": ("exp104_8cjg_slig_ablation/per_run.csv",
             ("arm", "control"), "ligand_rmsd_angstrom", "run",
             "20 diffusion draws under one conditioning"),
    "8jjq": ("exp105_8jjq_slig_harm_control/per_run.csv",
             ("arm", "control"), "ligand_rmsd_angstrom", "run",
             "20 diffusion draws under one conditioning"),
    "7ftm": ("exp033_boltz2_7ftm_multisample/per_pose_metrics.csv",
             None, "ligand_rmsd_angstrom", "pose_index",
             "32 poses across model seeds"),
}


def familiarity(manifests: pathlib.Path, wanted: set[str]) -> dict[str, dict]:
    """Per-system familiarity, as recorded in the input manifests.

    The score is SuCOS shape x pocket quality-coverage against the closest system
    released BEFORE the structural training cutoff, on 0-100. It says how much like
    its training set a target looks, which is the first thing anyone asks when a
    co-folding model fails.
    """
    import json
    out: dict[str, dict] = {}
    for path in sorted(manifests.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for sample in payload.get("samples", []):
            pdb = str(sample.get("sample_id", ""))[:4].lower()
            rnp = sample.get("runs_n_poses") or {}
            record = rnp.get("nesso1_sep2021_familiarity")
            if pdb in wanted and record and pdb not in out:
                out[pdb] = dict(record, manifest=path.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the reports/ directory to read")
    ap.add_argument("--manifests", default=None,
                    help="the data/manifests/ directory; familiarity is read from there")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "control_poses.csv"))
    args = ap.parse_args()
    src = pathlib.Path(args.source)

    out_rows, digests = [], {}
    for pdb, (rel, keep, rmsd_col, draw_col, note) in sorted(SOURCES.items()):
        path = src / rel
        digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for row in csv.DictReader(path.open()):
            if keep and row[keep[0]] != keep[1]:
                continue
            out_rows.append({"pdb": pdb, "draw": row[draw_col],
                             "ligand_rmsd_angstrom": f"{float(row[rmsd_col]):.3f}",
                             "source_correct": row.get("correct", ""),
                             "source": rel})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    if args.manifests:
        systems = familiarity(pathlib.Path(args.manifests), set(SOURCES))
        counts = {pdb: sum(r["pdb"] == pdb for r in out_rows) for pdb in SOURCES}
        path = pathlib.Path(args.out).with_name("systems.csv")
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pdb", "familiarity_0_to_100", "closest_pre_cutoff_system",
                        "closest_system_release_date", "structural_training_cutoff",
                        "n_control_draws", "manifest"])
            for pdb, rec in sorted(systems.items()):
                w.writerow([pdb, f'{rec["similarity_score_0_to_100"]:.1f}',
                            rec["closest_pre_cutoff_system"],
                            rec["closest_system_release_date"],
                            rec["structural_training_cutoff"],
                            counts[pdb], rec["manifest"]])
        print(f"{len(systems)} systems -> {path}")

    print(f"{len(out_rows)} rows -> {args.out}")
    for rel, h in digests.items():
        print(f"  {h}  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
