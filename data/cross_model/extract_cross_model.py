#!/usr/bin/env python3
"""Rebuild data/cross_model/ from the experiment tree this repository was extracted from.

Two tables, both consolidations of work that already exists; nothing is recomputed
from raw predictions here.

  pose_lddt.csv   one row per scored 8SGE pose, three models x two MSA conditions
                  (exp096), carrying ligand RMSD, lDDT-PLI and the axial torsion.
  systems.csv     one row per (system, model) cell this project has scored
                  (reports/cross_model_cofolding), carrying the < 2 A hit count.
  system_poses.csv  one row per DRAW behind those cells, carrying its ligand RMSD.
                  Followed from each cell's own source_csv pointer, so the counts
                  recomputed from it must reproduce systems.csv exactly; the
                  extractor asserts that and refuses to write if they disagree.

This script CANNOT run on a fresh clone -- its sources are not vendored. It is
committed so the vendored tables have a recipe rather than only a description.

    python data/cross_model/extract_cross_model.py --source /path/to/BindingAffinityPredictor
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

POSE_SRC = "reports/exp096_8sge_four_model/pli_per_pose.csv"
SYS_SRC = "reports/cross_model_cofolding/cross_model_pose_results.csv"

POSE_COLS = ["model", "msa", "seed", "sample", "ligand_rmsd_angstrom",
             "lddt_pli", "axial_torsion_deg", "crystal_side_of_axis"]
SYS_COLS = ["pdb", "familiarity", "model", "experiment_id", "n_poses",
            "hits_lt2", "best_rmsd", "median_rmsd"]
DRAW_COLS = ["pdb", "familiarity", "model", "experiment_id", "pose",
             "ligand_rmsd_angstrom"]

# The per-pose files were written by six different experiments and name the system
# and the model in whatever way suited each. These three readers are the whole of
# the difference; everything else about a draw is read the same way.
MODEL_NAMES = {"alphafold3": "AlphaFold3", "boltz2": "Boltz-2",
               "protenix": "Protenix", "boltz-2": "Boltz-2"}


def system_of(row: dict) -> str | None:
    """The PDB id a row belongs to, or None in a single-system file."""

    if row.get("pdb_id"):
        return row["pdb_id"].upper()
    for key in ("target", "sample_id"):
        if row.get(key):
            return row[key][:4].upper()
    return None


def model_of(row: dict) -> str | None:
    """The model a row came from, or None in a single-model file."""

    name = row.get("sampling_model") or row.get("model")
    return MODEL_NAMES.get((name or "").lower(), name) if name else None


def pose_of(row: dict, index: int) -> str:
    """A stable identifier for the draw: its seed, or its prediction path."""

    if row.get("seed"):
        return row["seed"]
    ref = row.get("file") or row.get("prediction_file") or ""
    m = re.search(r"seed[_-](\d+)", ref)
    return m.group(1) if m else str(index)


def draws_for_cell(cell: dict, source: Path) -> list[dict]:
    """Every draw behind one (system, model) cell, read from its own source file."""

    rows = list(csv.DictReader(open(source / cell["source_csv"])))
    out = []
    for i, r in enumerate(rows):
        sys_id, model = system_of(r), model_of(r)
        if sys_id is not None and sys_id != cell["pdb"]:
            continue
        if model is not None and model != cell["model"]:
            continue
        out.append({"pdb": cell["pdb"], "familiarity": cell["familiarity"],
                    "model": cell["model"],
                    "experiment_id": cell["experiment_id"],
                    "pose": pose_of(r, i),
                    "ligand_rmsd_angstrom": f"{float(r['ligand_rmsd_angstrom']):.3f}"})
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_and_sample(row: dict) -> tuple[str, str]:
    """exp096 records AF3/Protenix seeds in a column and Boltz-2's in the path."""
    seed, ref = row.get("seed", ""), row.get("file", "")
    if not seed:
        m = re.search(r"seed[_-](\d+)", ref)
        seed = m.group(1) if m else ""
    m = re.search(r"(?:sample[_-]|model_)(\d+)", ref)
    return seed, (m.group(1) if m else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path,
                    help="checkout of the experiment tree holding reports/")
    args = ap.parse_args()

    poses_in = args.source / POSE_SRC
    systems_in = args.source / SYS_SRC

    rows = list(csv.DictReader(open(poses_in)))
    out = []
    for r in rows:
        seed, sample = seed_and_sample(r)
        out.append({"model": r["model"], "msa": r["msa"], "seed": seed,
                    "sample": sample,
                    "ligand_rmsd_angstrom": r["ligand_rmsd_angstrom"],
                    "lddt_pli": r["lddt_pli"],
                    "axial_torsion_deg": r["axial_torsion_deg"],
                    "crystal_side_of_axis": r["crystal_side_of_axis"]})
    out.sort(key=lambda r: (r["model"], r["msa"], int(r["seed"] or 0),
                            int(r["sample"] or 0)))
    with open(HERE / "pose_lddt.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=POSE_COLS)
        w.writeheader()
        w.writerows(out)
    print(f"pose_lddt.csv   {len(out)} poses   source sha256 {sha256(poses_in)[:16]}")

    cells = []
    for r in csv.DictReader(open(systems_in)):
        # Drop the frozen-MSA arms. They exist to ask a different question and are
        # not comparable with the ordinary full-MSA sweeps the other cells carry.
        if r["msa"].startswith("frozen"):
            continue
        cells.append({"pdb": r["pdb_id"], "familiarity": r["familiarity"],
                      "model": r["model"], "experiment_id": r["experiment_id"],
                      "n_poses": r["n_poses"], "hits_lt2": r["hits_lt2"],
                      "best_rmsd": r["best_rmsd"], "median_rmsd": r["median_rmsd"],
                      "source_csv": r["source_csv"]})
    # One cell per (system, model), the largest full-MSA sweep.
    best: dict[tuple[str, str], dict] = {}
    for c in cells:
        key = (c["pdb"], c["model"])
        if key not in best or int(c["n_poses"]) > int(best[key]["n_poses"]):
            best[key] = c
    cells = sorted(best.values(), key=lambda c: (float(c["familiarity"]), c["pdb"],
                                                 c["model"]))
    with open(HERE / "systems.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SYS_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(cells)
    n_sys = len({c["pdb"] for c in cells})
    print(f"systems.csv     {len(cells)} cells, {n_sys} systems   "
          f"source sha256 {sha256(systems_in)[:16]}")

    draws, disagreements = [], []
    for c in cells:
        got = draws_for_cell(c, args.source)
        hits = sum(1 for d in got if float(d["ligand_rmsd_angstrom"]) < 2.0)
        if (len(got), hits) != (int(c["n_poses"]), int(c["hits_lt2"])):
            disagreements.append(
                f"{c['pdb']}/{c['model']} ({c['experiment_id']}): "
                f"read {len(got)} draws with {hits} hits, "
                f"summary says {c['n_poses']} with {c['hits_lt2']}")
        draws.extend(got)
    if disagreements:
        raise SystemExit("per-pose files do not reproduce the summary table:\n  "
                         + "\n  ".join(disagreements))
    with open(HERE / "system_poses.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DRAW_COLS)
        w.writeheader()
        w.writerows(draws)
    print(f"system_poses.csv {len(draws)} draws over {len(cells)} cells   "
          f"counts match systems.csv exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
