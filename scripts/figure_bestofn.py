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
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cofolding_tts import figstyle as fs              # noqa: E402

MUT = fs.MUTED
BLUE, ORANGE, GREEN, VERM, PURPLE = (fs.BLUE, fs.ORANGE, fs.GREEN, fs.VERM,
                                     fs.PURPLE)

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




def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-pose", default=str(ROOT / "results" / "8sge_per_pose.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "8sge_best_of_n.png"))
    ap.add_argument("--max-n", type=int, default=40)
    args = ap.parse_args()

    groups = load(args.per_pose)
    fs.use(15.0)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.992, top=0.885, bottom=0.135,
                        wspace=0.065)

    panels = [(axes[0], "a", "placed_under_2A",
               fs.tex(r"the crystal pose (RMSD $<$ 2\,\AA)")),
              (axes[1], "b", "axis_correct",
               fs.tex(r"the crystal's axial sense"))]
    xs = range(1, args.max_n + 1)
    summary = {}
    for ax, letter, column, title in panels:
        n_scan = len(groups[SCAN])
        k_scan = sum(int(r[column]) for r in groups[SCAN])
        ax.plot(xs, reach(n_scan, k_scan, args.max_n), color=MUT, lw=2.4,
                ls=(0, (4, 2.5)), zorder=2,
                label=f"unperturbed  ({k_scan}/{n_scan})")
        for arm, label, colour, dash in ARMS:
            rows = groups[arm]
            if not rows:
                continue
            n, k = len(rows), sum(int(r[column]) for r in rows)
            ax.plot(xs, reach(n, k, args.max_n), color=colour, lw=2.8, ls=dash,
                    zorder=3, label=f"{label}  ({k}/{n})")
            summary.setdefault(arm, {})[column] = (k, n)
        ax.axhline(0.9, color=MUT, lw=1.1, ls=(0, (2, 2.5)), zorder=1)
        ax.set_xlim(1, args.max_n)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("model runs, $N$")
        fs.box(ax)
        fs.panel(fig, ax, letter, dx=0.004 if letter == "b" else -0.052, dy=0.015,
                 text=title)
        ax.legend(frameon=False, loc="lower right", fontsize=13.0,
                  handlelength=2.0, borderaxespad=0.7)
    axes[0].set_ylabel("chance at least one draw hits")
    axes[1].tick_params(labelleft=False)

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
