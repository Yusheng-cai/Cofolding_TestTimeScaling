#!/usr/bin/env python3
r"""[inspect] Every unperturbed draw this project has scored: 645 poses, 8 systems, 3 models.

One strip per (system, model) cell, every draw in it drawn. A rate would hide the
thing worth seeing, which is the SHAPE of each cell:

  * a tight clump far from the crystal -- the model is not uncertain, it is
    confidently wrong, and repeating the draw returns the same wrong answer
    (7FTM at 31 A, 8TQV at 15, 8H7R at 8);
  * two clumps with a gap -- the model is choosing between two answers, and the
    fraction in the near one is what a hit rate reports (8CJG and 8AM2 under
    Boltz-2, 8SGE under AlphaFold3);
  * one clump under 2 A -- the system is solved (8RDK).

Models occupy fixed slots within a system, top to bottom, so an empty slot means
that model was never run on that system rather than that it scored badly.

Rows are ordered by familiarity, on the right axis. Sampling differs between cells
and Protenix was only ever run on 8SGE; read data/cross_model/PROVENANCE.md before
quoting any of this.

    python scripts/figure_cross_model.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cofolding_tts import figstyle as fs              # noqa: E402

HIT = 2.0
# model -> (slot within the system's row, colour, marker). Fixed, so a missing
# cell reads as a gap rather than as a different model.
SLOTS = {"Boltz-2": (0.26, fs.BLUE, "o"),
         "AlphaFold3": (0.0, fs.ORANGE, "s"),
         "Protenix": (-0.26, fs.PINK, "D")}


def load(path: Path):
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    familiarity: dict[str, float] = {}
    for r in csv.DictReader(open(path)):
        cells[(r["pdb"], r["model"])].append(float(r["ligand_rmsd_angstrom"]))
        familiarity[r["pdb"]] = float(r["familiarity"])
    return cells, familiarity


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses",
                    default=str(ROOT / "data" / "cross_model" / "system_poses.csv"))
    ap.add_argument("--out",
                    default=str(ROOT / "results" / "figures" / "cross_model.png"))
    args = ap.parse_args()

    cells, familiarity = load(Path(args.poses))
    systems = sorted(familiarity, key=lambda p: (familiarity[p], p))

    fs.use(15.0)
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    fig.subplots_adjust(left=0.105, right=0.915, top=0.900, bottom=0.118)

    rng = np.random.default_rng(0)
    ax.axvspan(0.6, HIT, color=fs.GREEN, alpha=0.07, lw=0, zorder=0)
    ax.axvline(HIT, color=fs.MUTED, lw=1.3, ls=(0, (3, 3)), zorder=1)

    for i, pdb in enumerate(systems):
        if i:
            ax.axhline(i - 0.5, color="#e8e8e8", lw=1.0, zorder=0)
        for model, (slot, colour, marker) in SLOTS.items():
            draws = cells.get((pdb, model))
            if not draws:
                continue
            v = np.asarray(draws)
            y = i + slot + rng.uniform(-0.062, 0.062, v.size)
            ax.scatter(v, y, s=26, c=colour, marker=marker, alpha=0.5,
                       linewidths=0.0, zorder=3)
            ax.scatter([np.median(v)], [i + slot], s=150, c=colour, marker=marker,
                       linewidths=1.3, edgecolors=fs.SURFACE, zorder=4)

    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems)
    ax.set_ylim(-0.56, len(systems) - 0.44)
    ax.set_xscale("log")
    ax.set_xlim(0.62, 42)
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    ax.set_xlabel(fs.tex(r"ligand RMSD to the crystal (\AA), every draw"))
    fs.box(ax, minor=False)

    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(range(len(systems)))
    right.set_yticklabels([f"{familiarity[p]:.0f}" for p in systems])
    right.set_ylabel("familiarity")
    right.tick_params(direction="in", length=7.0, width=1.3, pad=7.0)

    handles = [plt.Line2D([], [], ls="", marker=m, color=c, markersize=11,
                          markeredgecolor=fs.SURFACE, markeredgewidth=1.2, label=k)
               for k, (_, c, m) in SLOTS.items()]
    handles.append(plt.Line2D([], [], ls="", marker="o", color=fs.MUTED,
                              markersize=11, markeredgecolor=fs.SURFACE,
                              markeredgewidth=1.2, label="median of the cell"))
    ax.legend(handles=handles, frameon=False, ncol=4, fontsize=14.0,
              loc="lower left", bbox_to_anchor=(0.0, 1.012), handletextpad=0.3,
              columnspacing=1.5, borderaxespad=0.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"wrote {out}")
    for pdb in systems:
        for model in SLOTS:
            v = cells.get((pdb, model))
            if not v:
                continue
            v = np.asarray(v)
            print(f"  {pdb}  fam {familiarity[pdb]:4.1f}  {model:<11} n={v.size:3d}  "
                  f"median {np.median(v):6.2f}  spread {v.max() - v.min():6.2f}  "
                  f"< 2 A {int((v < HIT).sum()):3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
