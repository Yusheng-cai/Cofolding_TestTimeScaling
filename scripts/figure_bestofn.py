#!/usr/bin/env python3
"""[inspect] Best-of-N reachability, from committed data. No GPU, no run directories.

For each conditioning, the chance that at least one of N draws reaches the target,
computed exactly by sampling the recorded draws without replacement:

    P(hit within N) = 1 - C(n - k, N) / C(n, N)

Two panels, because the two metrics disagree in an informative way:

  left   reaching the crystal pose          ligand RMSD < 2 A
  right  reaching the crystal's axial sense  C12-C11-N1-C3 on the crystal's side

This is ORACLE best-of-N: it says the right answer is now *reachable*, not that it
can be picked out. Nothing in this repository ranks the draws.

    python scripts/figure_bestofn.py
"""

from __future__ import annotations

import argparse
import csv
import re
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK, MUT, SURFACE = "#2a2a28", "#7a7a76", "#fcfcfb"
# Okabe-Ito, assigned in a fixed order and never cycled.
BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"

SCAN = "control_scan"
SCAN_MAX = 40
# (arm, label, colour, linestyle). Seed 59 is dashed for two reasons: it lands on
# exactly the same 21/40 as seed 42 and would otherwise be hidden under it, and it
# is the one conditioning the model already placed correctly, so it reads as a cost
# rather than a rescue.
ARMS = [("gauss-s_lig-s12", "seed 12", BLUE, "-"),
        ("gauss-s_lig", "seed 42", ORANGE, "-"),
        ("gauss-s_lig-s59", "seed 59 (already correct)", GREEN, (0, (5, 2))),
        ("gauss-s_lig-s14", "seed 14", PURPLE, "-"),
        ("gauss-s_lig-s05", "seed 5", VERM, "-")]


def reach(n: int, k: int, upto: int):
    """Exact P(at least one of the k successes appears in N draws of n), N = 1..upto."""
    return [1.0 - (comb(n - k, N) / comb(n, N) if N <= n - k else 0.0)
            for N in range(1, upto + 1)]


def load(path):
    raw = list(csv.DictReader(open(path)))
    groups = {a: [] for a, _, _, _ in ARMS}
    groups[SCAN] = []
    for r in raw:
        if r["arm"] in groups:
            groups[r["arm"]].append(r)
        elif r["arm"] == "control":
            m = re.search(r"_d(\d+)$", r["run"])
            if m and int(m.group(1)) <= SCAN_MAX:
                groups[SCAN].append(r)
        elif r["arm"] == "baseline_lam0":
            groups[SCAN].append(r)
    return groups


def style():
    matplotlib.rcdefaults()
    fs = 9
    plt.rcParams.update({
        "text.usetex": False, "font.family": "DejaVu Sans", "font.size": fs,
        "axes.titlesize": fs + 0.5, "axes.labelsize": fs - 0.5,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "axes.edgecolor": "#8a8a85", "axes.labelcolor": INK, "axes.linewidth": 0.8,
        "xtick.color": "#6a6a66", "ytick.color": "#6a6a66",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-pose", default=str(ROOT / "results" / "8sge_per_pose.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "8sge_best_of_n.png"))
    ap.add_argument("--max-n", type=int, default=40)
    args = ap.parse_args()

    groups = load(args.per_pose)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.715, bottom=0.135, wspace=0.09)

    panels = [(axes[0], "placed_under_2A", "Reaching the crystal pose",
               "ligand RMSD < 2 Å"),
              (axes[1], "axis_correct", "Reaching the crystal's axial sense",
               "C12–C11–N1–C3 on the crystal's side")]
    xs = range(1, args.max_n + 1)
    summary = {}
    for ax, column, title, sub in panels:
        n_scan = len(groups[SCAN])
        k_scan = sum(int(r[column]) for r in groups[SCAN])
        ax.plot(xs, reach(n_scan, k_scan, args.max_n), color=MUT, lw=1.6,
                ls=(0, (4, 2.5)), zorder=2,
                label=f"unperturbed scan  ({k_scan}/{n_scan})")
        for arm, label, colour, dash in ARMS:
            rows = groups[arm]
            if not rows:
                continue
            n, k = len(rows), sum(int(r[column]) for r in rows)
            ax.plot(xs, reach(n, k, args.max_n), color=colour, lw=2.0, ls=dash,
                    zorder=3, label=f"{label}  ({k}/{n})")
            summary.setdefault(arm, {})[column] = (k, n)
        ax.axhline(0.9, color=MUT, lw=0.9, ls=(0, (2, 2.5)), zorder=1)
        ax.annotate("90%", (args.max_n, 0.9), textcoords="offset points",
                    xytext=(-2, 5), ha="right", fontsize=7.2, color=MUT)
        ax.set_xlim(1, args.max_n)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("model runs, N")
        ax.set_title(title, color=INK, loc="left", pad=20)
        ax.annotate(sub, (0, 1), xycoords="axes fraction", textcoords="offset points",
                    xytext=(0, 5), fontsize=7.4, color=MUT, va="bottom")
        ax.grid(alpha=0.10, lw=0.5)
        ax.legend(frameon=False, fontsize=7.4, labelcolor=INK, loc="lower right",
                  handlelength=1.8, borderaxespad=0.6)
    axes[0].set_ylabel("chance at least one draw hits")

    fig.suptitle("8SGE: with s_lig destroyed, the crystal pose becomes reachable "
                 "at three conditionings of five",
                 x=0.075, ha="left", y=0.965, fontsize=11, color=INK)
    fig.text(0.075, 0.892,
             "equal compute: N runs means N model seeds unperturbed, N noise draws "
             "at one seed perturbed.  Oracle best-of-N —",
             ha="left", fontsize=8.2, color=MUT)
    fig.text(0.075, 0.855,
             "the right answer becomes reachable, not identified; nothing here ranks "
             "the draws.",
             ha="left", fontsize=8.2, color=MUT)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    for arm, label, _, _ in ARMS:
        if arm not in summary:
            continue
        line = []
        for column in ("placed_under_2A", "axis_correct"):
            k, n = summary[arm][column]
            need = next((N for N in xs if reach(n, k, args.max_n)[N - 1] >= 0.9), None)
            line.append(f"{column}={k}/{n} N90={need if need else '>%d' % args.max_n}")
        print(f"  {label:<26} " + "  ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
