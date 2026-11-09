#!/usr/bin/env python3
"""[inspect] The 8SGE rescue figure, from committed data. No GPU, no run directories.

Two panels against the same arm rows:

  left   where the ligand ends up, as RMSD to the crystal
  right  which way round the hindered axis is, as the C12-C11-N1-C3 torsion

Reading them together is the point. The axis is not a correlate of the placement;
it is the placement. A pose on the wrong side of the axis cannot be seated, because
the two halves of the molecule have swapped sides of the pocket.

    python scripts/figure_8sge.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CRYSTAL_TORSION = -101.4

# Okabe-Ito blue and vermillion: distinguishable under every common form of
# colour-vision deficiency, and checked against the light surface below.
BLUE, VERM, INK, MUT = "#0072B2", "#D55E00", "#2a2a28", "#7a7a76"
SURFACE = "#fcfcfb"

HEADLINE = ["control", "gauss-s_lig"]
LABELS = {
    "control": "control\nunperturbed",
    "gauss-s_lig": "gauss s_lig\nseed 42",
    "gauss-s_lig-s12": "seed 12",
    "gauss-s_lig-s59": "seed 59",
    "gauss-s_lig-s14": "seed 14",
    "gauss-s_lig-s05": "seed 5",
}


def style() -> None:
    matplotlib.rcdefaults()
    fs = 9
    plt.rcParams.update({
        "text.usetex": False, "font.family": "DejaVu Sans",
        "font.size": fs, "axes.titlesize": fs + 0.5, "axes.labelsize": fs - 0.5,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "axes.edgecolor": "#8a8a85", "axes.labelcolor": INK, "axes.linewidth": 0.8,
        "xtick.color": "#6a6a66", "ytick.color": "#6a6a66",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-pose", default=str(ROOT / "results" / "8sge_per_pose.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "8sge_rescue.png"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.per_pose)) if r["arm"] in LABELS]
    if not rows:
        raise SystemExit(f"no usable rows in {args.per_pose}")
    order = HEADLINE + [a for a in LABELS if a not in HEADLINE
                        and any(r["arm"] == a for r in rows)]

    style()
    fig = plt.figure(figsize=(10.4, 0.62 * len(order) + 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.14,
                          left=0.135, right=0.985, top=0.80, bottom=0.17)
    rng = np.random.default_rng(0)

    panels = [
        (fig.add_subplot(gs[0, 0]), "ligand_rmsd_angstrom",
         "Where the ligand ends up", "RMSD to crystal  (Å)", 2.0),
        (fig.add_subplot(gs[0, 1]), "axial_torsion_deg",
         "Which way round the axis is", "C12–C11–N1–C3 torsion  (degrees)", None),
    ]
    for ax, column, title, xlabel, marker in panels:
        for i, arm in enumerate(order):
            y = len(order) - 1 - i
            arm_rows = [r for r in rows if r["arm"] == arm]
            for correct, colour in ((1, BLUE), (0, VERM)):
                v = np.array([float(r[column]) for r in arm_rows
                              if int(r["axis_correct"]) == correct])
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=22, color=colour,
                           alpha=0.8, linewidths=0.5, edgecolor=SURFACE, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABELS[a] for a in reversed(order)], fontsize=7.6,
                           color=INK, linespacing=1.3)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(title, color=INK, loc="left", pad=7)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.10, lw=0.5)
        if marker is not None:
            ax.axvline(marker, color=MUT, lw=1, ls=(0, (4, 3)), zorder=2)
            ax.annotate("2 Å", (marker, len(order) - 0.55), textcoords="offset points",
                        xytext=(5, 0), fontsize=7.2, color=MUT, va="center")
            ax.set_xlim(left=0)
        else:
            ax.axvline(CRYSTAL_TORSION, color=MUT, lw=1, ls=(0, (4, 3)), zorder=2)
            ax.annotate("crystal", (CRYSTAL_TORSION, len(order) - 0.55),
                        textcoords="offset points", xytext=(5, 0), fontsize=7.2,
                        color=MUT, va="center")
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticklabels([])

    counts = {a: (sum(int(r["placed_under_2A"]) for r in rows if r["arm"] == a),
                  sum(r["arm"] == a for r in rows)) for a in HEADLINE if a in order}
    headline = "   ".join(
        f"{LABELS[a].splitlines()[0]}: {hit} of {n} placed under 2 Å"
        for a, (hit, n) in counts.items())
    fig.suptitle("8SGE: destroying the ligand's trunk representation recovers the pose",
                 x=0.135, ha="left", y=0.965, fontsize=11, color=INK)
    fig.text(0.135, 0.895, headline, ha="left", fontsize=8.2, color=MUT)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, markersize=5,
                               label="axis matches the crystal"),
                        Line2D([], [], marker="o", ls="", color=VERM, markersize=5,
                               label="axis flipped")],
               frameon=False, fontsize=7.8, labelcolor=INK, ncol=2,
               loc="upper right", bbox_to_anchor=(0.985, 0.985))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
