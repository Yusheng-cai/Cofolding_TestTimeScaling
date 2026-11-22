#!/usr/bin/env python3
r"""[inspect] Eight systems, three models: how often an unperturbed draw lands under 2 A.

One row per deposited system, one marker per model, drawn from the per-pose sweeps
already scored in the work this repository was extracted from. Systems are ordered
by FAMILIARITY -- how much the target resembles the closest complex released before
the 2021-09-30 structural training cutoff -- because the obvious explanation for a
failure is that the model has never seen anything like the target, and this ordering
shows that it is not the explanation.

Read it as three claims:

  * the failure is not one structure. Four of the eight are missed by every model
    that has been run on them.
  * the failure is not one model. Where models disagree they disagree sharply, and
    not consistently in anyone's favour.
  * familiarity does not order the outcome. The most familiar system here is solved
    and the second most familiar is missed by both models.

    python scripts/figure_cross_model.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cofolding_tts import figstyle as fs              # noqa: E402

MODELS = [("Boltz-2", fs.BLUE, "o"), ("AlphaFold3", fs.ORANGE, "s"),
          ("Protenix", fs.PURPLE, "D")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems",
                    default=str(ROOT / "data" / "cross_model" / "systems.csv"))
    ap.add_argument("--out",
                    default=str(ROOT / "results" / "figures" / "cross_model.png"))
    args = ap.parse_args()

    cells: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    familiarity: dict[str, float] = {}
    for r in csv.DictReader(open(args.systems)):
        familiarity[r["pdb"]] = float(r["familiarity"])
        cells[r["pdb"]][r["model"]] = (int(r["hits_lt2"]), int(r["n_poses"]))
    order = sorted(familiarity, key=lambda p: (familiarity[p], p))

    fs.use(16.0)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    fig.subplots_adjust(left=0.115, right=0.885, top=0.885, bottom=0.115)

    for i, pdb in enumerate(order):
        rates = sorted(cells[pdb][m][0] / cells[pdb][m][1]
                       for m, _, _ in MODELS if m in cells[pdb])
        ax.plot([-0.055, 1.055], [i, i], color="#ededed", lw=1.2, zorder=1)
        if len(rates) > 1:                                  # span between models
            ax.plot([rates[0], rates[-1]], [i, i], color="#bdbdbd", lw=3.0,
                    solid_capstyle="round", zorder=2)

    for model, colour, marker in MODELS:
        pts = [(cells[p][model][0] / cells[p][model][1], i)
               for i, p in enumerate(order) if model in cells[p]]
        ax.scatter([x for x, _ in pts], [y for _, y in pts], s=265, c=colour,
                   marker=marker, zorder=4, edgecolors=fs.SURFACE, linewidths=1.4,
                   label=model)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.055, 1.055)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([fs.tex(s) for s in ("0", r"25\%", r"50\%", r"75\%", r"100\%")])
    ax.set_xlabel(fs.tex(r"unperturbed draws under 2\,\AA"))
    fs.box(ax)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    # Familiarity on the right, so the ordering is readable without a second label
    # line under every system name.
    rhs = ax.twinx()
    rhs.set_ylim(*ax.get_ylim())
    rhs.set_yticks(range(len(order)))
    rhs.set_yticklabels([f"{familiarity[p]:.0f}" for p in order])
    rhs.set_ylabel("familiarity", labelpad=10)
    fs.box(rhs, minor=False)
    rhs.tick_params(axis="y", direction="in", left=False, right=True)
    rhs.tick_params(axis="x", top=False, bottom=False)

    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
              columnspacing=1.9, handletextpad=0.3, borderaxespad=0.0,
              scatterpoints=1, fontsize=15)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"wrote {out}")
    for pdb in order:
        bits = "  ".join(f"{m}={cells[pdb][m][0]}/{cells[pdb][m][1]}"
                         for m, _, _ in MODELS if m in cells[pdb])
        print(f"  {pdb}  familiarity {familiarity[pdb]:>5.1f}   {bits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
