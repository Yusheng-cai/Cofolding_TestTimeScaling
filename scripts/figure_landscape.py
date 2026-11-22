#!/usr/bin/env python3
r"""[inspect] 8SGE pose landscape: lDDT-PLI against ligand RMSD, three co-folding models.

600 poses from exp096 (three models x two MSA conditions x 20 seeds x 5 samples);
this figure draws the 300 full-MSA poses, one panel per model.

Two things the panels are drawn to show:

  * the draws are BIMODAL, not diffuse. The marginal histogram over each panel has
    two lobes with a gap between roughly 2.5 and 5 A, so a model is not imprecise
    about where the ligand goes -- it chooses between two answers.
  * which lobe a model lands in is a property of the MODEL, not of the target. Same
    molecule, same pocket, same MSA: Boltz-2 sits in the wrong lobe almost always,
    Protenix in the right one almost always, AlphaFold3 splits.

Colour is the axial torsion C12-C11-N1-C3, the order parameter that distinguishes
the two rotamers, not a restatement of the x axis.

The horizontal line is Runs N' Poses' own success criterion, lDDT-PLI > 0.8, which
that benchmark applies together with RMSD < 2 A. The two dashed lines therefore cut
each panel into the benchmark's four outcomes, and the top-left box is what it would
score as a success -- a stricter count than RMSD alone, and the one to quote when
comparing against that paper.

For reference the script also fits the cut that best separates RMSD < 2 A from the
rest over these 300 poses (Youden's J) and prints it. It lands at 0.76, close to the
published 0.8, which is a check on the criterion rather than a proposal: a cut fitted
on the labels it is scored against is optimistic, and lDDT-PLI cannot select anything
in any case, since it needs the deposited complex to compute.

    python scripts/figure_landscape.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cofolding_tts import figstyle as fs              # noqa: E402
from cofolding_tts.lddt import youden_threshold        # noqa: E402

MODELS = ["Boltz-2", "AlphaFold3", "Protenix"]
MSA = "full 2633"
RMSD_MAX = 10.0
HIT = 2.0
RNP_PLI = 0.8          # Runs N' Poses: success is RMSD < 2 A AND lDDT-PLI > 0.8


def load(path: Path, msa: str) -> dict[str, dict[str, np.ndarray]]:
    by: dict[str, list[dict]] = {m: [] for m in MODELS}
    for r in csv.DictReader(open(path)):
        if r["msa"] == msa and r["model"] in by:
            by[r["model"]].append(r)
    return {m: {"rmsd": np.array([float(r["ligand_rmsd_angstrom"]) for r in rows]),
                "pli": np.array([float(r["lddt_pli"]) for r in rows]),
                "axis": np.array([int(r["crystal_side_of_axis"]) for r in rows], bool)}
            for m, rows in by.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", default=str(ROOT / "data" / "cross_model" / "pose_lddt.csv"))
    ap.add_argument("--msa", default=MSA)
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "landscape.png"))
    args = ap.parse_args()

    data = load(Path(args.poses), args.msa)
    pooled_rmsd = np.concatenate([d["rmsd"] for d in data.values()])
    pooled_pli = np.concatenate([d["pli"] for d in data.values()])
    sep = youden_threshold(pooled_pli, pooled_rmsd < HIT)
    fs.use(15.0)

    fig = plt.figure(figsize=(15.0, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 4.6], hspace=0.07, wspace=0.075,
                          left=0.055, right=0.992, top=0.855, bottom=0.165)
    bins = np.arange(0.0, RMSD_MAX + 0.5, 0.5)
    hmax = max(np.histogram(d["rmsd"], bins=bins)[0].max() for d in data.values())

    share = None
    for col, (model, letter) in enumerate(zip(MODELS, "abc")):
        d = data[model]
        top = fig.add_subplot(gs[0, col])
        ax = fig.add_subplot(gs[1, col], sharey=share)
        share = share or ax

        top.hist(d["rmsd"], bins=bins, color="#d2d2d2", edgecolor=fs.SURFACE, lw=0.5)
        top.set_xlim(0, RMSD_MAX)
        top.set_ylim(0, hmax * 1.12)
        top.axis("off")

        for mask, colour, label in ((~d["axis"], fs.VERM, "the other rotamer"),
                                    (d["axis"], fs.GREEN, "crystal's rotamer")):
            ax.scatter(d["rmsd"][mask], d["pli"][mask], s=115, c=colour, alpha=0.85,
                       linewidths=0.9, edgecolors=fs.SURFACE, zorder=3, label=label)
        ax.axvline(HIT, color=fs.MUTED, lw=1.3, ls=(0, (3, 3)), zorder=1)
        ax.axhline(RNP_PLI, color=fs.MUTED, lw=1.3, ls=(0, (3, 3)), zorder=1)
        ax.set_xlim(0, RMSD_MAX)
        ax.set_ylim(0.15, 0.97)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        fs.box(ax)

        hit = d["rmsd"] < HIT
        n_hit, n_rnp = int(hit.sum()), int((hit & (d["pli"] > RNP_PLI)).sum())
        fs.panel(fig, top, letter, dx=0.004 if col else -0.047, dy=0.020, text=model)
        ax.annotate(fs.tex(rf"{n_hit}/{d['rmsd'].size} under 2\,\AA, "
                           rf"{n_rnp} also above 0.8"),
                    (0.965, 0.95), xycoords="axes fraction", ha="right", va="top",
                    fontsize=13.5, color=fs.MUTED)
        if col == 0:
            ax.set_ylabel("lDDT-PLI")
            ax.annotate(fs.tex(r"Runs N' Poses success"), (9.75, RNP_PLI - 0.016),
                        ha="right", va="top", fontsize=13.5, color=fs.MUTED)
            ax.legend(frameon=False, loc="lower right", fontsize=13.5,
                      handletextpad=0.25, borderaxespad=0.55, scatterpoints=1)
        else:
            ax.tick_params(labelleft=False)

    fig.text(0.523, 0.030, fs.tex(r"ligand RMSD to the crystal (\AA)"),
             ha="center", va="bottom", fontsize=plt.rcParams["axes.labelsize"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"wrote {out}")
    rnp = (pooled_rmsd < HIT) & (pooled_pli > RNP_PLI)
    print(f"  Runs N' Poses success (RMSD < 2 A and lDDT-PLI > 0.8): "
          f"{int(rnp.sum())}/{rnp.size}, against {int((pooled_rmsd < HIT).sum())}"
          f"/{rnp.size} on RMSD alone")
    print(f"  best-fit lDDT-PLI cut {sep.cut:.3f}  J={sep.youden_j:.3f}  "
          f"wrong poses above it={sep.accepted_negatives}  "
          f"right poses below it={sep.rejected_positives}  "
          f"(agrees with RMSD < 2 A on "
          f"{pooled_pli.size - sep.accepted_negatives - sep.rejected_positives}"
          f"/{pooled_pli.size})")
    for model in MODELS:
        d = data[model]
        gap = int(((d["rmsd"] >= 2.5) & (d["rmsd"] <= 5.0)).sum())
        hit = d["rmsd"] < HIT
        above = d["pli"] >= sep.cut
        print(f"  {model:<12} n={d['rmsd'].size:3d}  <2A={int(hit.sum()):3d}  "
              f"crystal rotamer={int(d['axis'].sum()):3d}  in the 2.5-5 A gap={gap:3d}  "
              f"RNP success={int((hit & (d['pli'] > RNP_PLI)).sum()):3d}  "
              f"miss above fitted cut={int((above & ~hit).sum()):3d}  "
              f"hit below it={int((~above & hit).sum()):3d}  "
              f"misses top out at {d['pli'][~hit].max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
