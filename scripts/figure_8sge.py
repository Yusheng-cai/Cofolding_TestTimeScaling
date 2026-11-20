#!/usr/bin/env python3
"""[inspect] The 8SGE rescue figure, from committed data. No GPU, no run directories.

Two panels over the same rows:

  left   where the ligand ends up, as RMSD to the crystal
  right  which way round the hindered axis is, as the C12-C11-N1-C3 torsion

The unit of a row is a *conditioning* (one model seed), not a replicate. The top
row is the unperturbed scan: 41 conditionings, one pose each, because an
unperturbed run has no other source of randomness. Every row below it is 40 draws
at one fixed conditioning, differing only in the noise that replaced the ligand's
trunk rows, with that conditioning's own unperturbed pose marked as a diamond.

Read the two panels together. The axis usually decides the placement -- but seed
14 is the exception that keeps the claim honest, where the axis flips and the
ligand still does not seat.

    python scripts/figure_8sge.py
"""

from __future__ import annotations

import argparse
import csv
import re
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

SCAN = "control_scan"

# Row order, and the control pose that belongs beside each row. Grouped by what
# the unperturbed model does at that conditioning: four it gets wrong, then the
# one it already gets right, which is a harm control rather than a rescue.
ORDER = [SCAN, "gauss-s_lig", "gauss-s_lig-s12", "gauss-s_lig-s14",
         "gauss-s_lig-s05", "gauss-s_lig-s59"]
LABELS = {
    SCAN:              "unperturbed scan\n41 conditionings",
    "gauss-s_lig":     "seed 42",
    "gauss-s_lig-s12": "seed 12",
    "gauss-s_lig-s14": "seed 14",
    "gauss-s_lig-s05": "seed 5",
    "gauss-s_lig-s59": "seed 59\n(already correct)",
}
CONTROL_OF = {"gauss-s_lig": "baseline_lam0", "gauss-s_lig-s12": "control_lam0p00_d12",
              "gauss-s_lig-s14": "control_lam0p00_d14", "gauss-s_lig-s05": "control_lam0p00_d05",
              "gauss-s_lig-s59": "control_lam0p00_d59"}
# The unbiased scan is model seeds 1-40 (the `control` arm) plus seed 42, which was
# run separately as `baseline_lam0`. Seed 59 is deliberately NOT in it: that
# conditioning was added later because the model already succeeds there, so counting
# it among the scan would inflate the unperturbed success rate.
SCAN_MAX = 40


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


def load(path: str):
    """Group the per-pose table into plot rows, and pick out each row's control."""
    raw = list(csv.DictReader(open(path)))
    by_run = {r["run"]: r for r in raw}
    groups: dict[str, list] = {a: [] for a in ORDER}
    for r in raw:
        if r["arm"] in groups:
            groups[r["arm"]].append(r)
        elif r["arm"] == "control":
            m = re.search(r"_d(\d+)$", r["run"])
            if m and int(m.group(1)) <= SCAN_MAX:
                groups[SCAN].append(r)
    if "baseline_lam0" in by_run:
        groups[SCAN].append(by_run["baseline_lam0"])
    controls = {a: by_run[k] for a, k in CONTROL_OF.items() if k in by_run}
    return [a for a in ORDER if groups[a]], groups, controls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-pose", default=str(ROOT / "results" / "8sge_per_pose.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "8sge_rescue.png"))
    args = ap.parse_args()

    order, groups, controls = load(args.per_pose)
    if not order:
        raise SystemExit(f"no usable rows in {args.per_pose}")

    style()
    fig = plt.figure(figsize=(10.4, 0.70 * len(order) + 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.14,
                          left=0.155, right=0.985, top=0.775, bottom=0.155)
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
            for correct, colour in ((1, BLUE), (0, VERM)):
                v = np.array([float(r[column]) for r in groups[arm]
                              if int(r["axis_correct"]) == correct])
                v = v[np.isfinite(v)]
                if len(v):
                    ax.scatter(v, y + rng.uniform(-0.19, 0.19, len(v)), s=22,
                               color=colour, alpha=0.8, linewidths=0.5,
                               edgecolor=SURFACE, zorder=3)
            c = controls.get(arm)
            if c is not None and np.isfinite(float(c[column])):
                ax.scatter([float(c[column])], [y], s=46, marker="D",
                           facecolor="none", edgecolor=INK, linewidths=1.1, zorder=5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABELS[a] for a in reversed(order)], fontsize=7.6,
                           color=INK, linespacing=1.3)
        ax.set_ylim(-0.6, len(order) - 0.4)
        # Separate the unperturbed scan from the fixed-conditioning rows below it.
        ax.axhline(len(order) - 1.5, color="#d8d8d2", lw=0.8, zorder=1)
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

    scan_hits = sum(int(r["placed_under_2A"]) for r in groups[SCAN])
    rescued = [a for a in order if a in CONTROL_OF
               and not int(controls[a]["placed_under_2A"])
               and sum(int(r["placed_under_2A"]) for r in groups[a]) > 1]
    sub = (f"unperturbed: {scan_hits} of {len(groups[SCAN])} conditionings placed "
           f"under 2 Å    with s_lig destroyed: {len(rescued)} of "
           f"{len(CONTROL_OF) - 1} wrong conditionings become reachable in 40 draws")
    fig.suptitle("8SGE: destroying the ligand's trunk representation makes the "
                 "right pose reachable",
                 x=0.155, ha="left", y=0.975, fontsize=11, color=INK)
    fig.text(0.155, 0.925, sub, ha="left", fontsize=8.2, color=MUT)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, markersize=5,
                               label="axis matches the crystal"),
                        Line2D([], [], marker="o", ls="", color=VERM, markersize=5,
                               label="axis flipped"),
                        Line2D([], [], marker="D", ls="", markerfacecolor="none",
                               markeredgecolor=INK, markersize=6,
                               label="unperturbed pose at this conditioning")],
               frameon=False, fontsize=7.8, labelcolor=INK, ncol=3,
               loc="upper left", bbox_to_anchor=(0.148, 0.898),
               handletextpad=0.4, columnspacing=1.6)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    for a in order:
        n = len(groups[a])
        hit = sum(int(r["placed_under_2A"]) for r in groups[a])
        ax_ok = sum(int(r["axis_correct"]) for r in groups[a])
        print(f"  {LABELS[a].splitlines()[0]:<24} n={n:>3}  axis {ax_ok:>3}  under2A {hit:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
