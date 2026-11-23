# Cofolding_TestTimeScaling

[Runs N' Poses](https://doi.org/10.1038/s41594-026-01797-5) showed that co-folding
accuracy scales with **familiarity** — how close a complex sits to something the
model was trained on. Read as a coverage problem, that points at more data and more
post-training.

![success rate against similarity to the training set](results/figures/rnp_familiarity.jpg)

Figure 1 of [Škrinjar, Eberhardt, Durairaj and
Schwede](https://doi.org/10.1101/2025.02.03.636309), reproduced unmodified under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — the one figure here that
is not ours; see `results/figures/PROVENANCE.md`. Success is RMSD < 2 Å **and**
lDDT-PLI > 0.8, and the four methods are AlphaFold3, Protenix, Chai-1 and Boltz-1.

The grey line is the one to look at. It is the best of the four methods' top-ranked
models, picked with the answer in hand, and it runs above every individual method in
every bin: the models fail on different systems, so a good fraction of the misses are
already being produced by *somebody*. That is a selection and sampling statement, not
a coverage one, and it is the opening this repository works in.

This repository tests the other reading: that for at least some of these misses the
right pose is already inside the model's reach and the sampler never visits it. If
that holds, the miss is a **search** failure at inference, and the lever is
test-time compute rather than another training run.

It is a falsifiable claim, not a hopeful one, and what follows is the evidence for
its premise: the shape of the failure across eight complexes and three models, and
the one system where it has been taken apart far enough to name the coordinate that
decides it. What to *do* about it is a separate question, and not this README's;
`docs/method.md` and the notebooks carry where that has got to.

## Sometimes the right answer is already in the output

![every scored draw, eight systems, three models](results/figures/cross_model.png)

645 unperturbed draws, 8 deposited complexes, 3 models, no oracle — every one of
them drawn, because the shape of a strip says what a hit rate cannot.

- **In 5 of the 16 cells the crystal pose is already there, and rare.** 10 of 25 on
  8CJG under Boltz-2 against a 5.7 Å median; 19 of 100 and 1 of 100 on 8SGE. Nothing
  is missing from those models. Which draw you get is the whole of the difference,
  and that is a search problem.
- **In 7 cells no draw ever came within 2 Å** — closest 2.10 Å, Boltz-2 on 8JMP.
  Nothing here says the answer is reachable in those, in either direction.
- **A tight clump far from the crystal is not uncertainty.** 7FTM's 25 draws span
  0.99 Å about a median of 31 Å. Drawing again returns the same wrong answer.
- **Familiarity, on the right axis, does not order the outcome here.** These eight
  were picked for being hard, so read that as a population trend giving no grip on
  an individual complex, not as a count against it.

The spread within a strip is the seed being re-rolled, not diffusion noise. Cells
are not uniformly sampled, a hit is RMSD < 2 Å alone rather than the benchmark's
stricter combined criterion, and Protenix has only been run on 8SGE — read
`data/cross_model/PROVENANCE.md` before quoting a rate.

## There are two answers, not a spread

![lDDT-PLI against ligand RMSD, three models](results/figures/landscape.png)

Co-folding models do not place a ligand imprecisely. They choose between a small
number of placements, commit early, and are often confident and wrong.

300 poses on 8SGE, same molecule, same pocket and same MSA for all three models
(each at its own default recycling, so this is three published models, not a
controlled compute comparison). The RMSD marginal has two lobes with almost nothing between them: 0, 2 and 13 of 100 draws
land in the 2.5–5 Å gap. **Colour is the axial torsion** C12–C11–N1–C3 — the two
lobes are the two rotamers of one hindered bond, and the model picks one.

Two things fall out of that panel and set up everything else here:

- **The dashed lines are Runs N' Poses' own success criterion**, RMSD < 2 Å *and*
  lDDT-PLI > 0.8, so the top-left box of each panel is what that benchmark would
  score. It is stricter than RMSD alone and the gap is not small: Protenix's 64
  sub-2 Å poses are 25 successes, because a pose can be close and still have given up
  contacts. The best cut these 300 poses can support is 0.76, near the published 0.8
  — but no cut on lDDT-PLI selects anything in practice, since it draws its contact
  pairs from the deposited complex and so needs the answer it would stand in for.
- **The right axis is necessary, not sufficient.** AlphaFold3's green points at
  6–9 Å are the crystal's rotamer in the wrong place.

## Running it

**Inspect** — no GPU, no model, no checkpoint; committed data only.

```bash
pip install -e .
python scripts/figure_cross_model.py   # the two figures above
python scripts/figure_landscape.py

python scripts/figure_compound.py      # not shown here: the ligand, the
python scripts/figure_bestofn.py       # intervention arms, the 8SGE rescue,
python scripts/figure_8sge.py          # and a standalone 3D viewer of both
python scripts/viewer_8sge.py          # basins in the pocket

python -m pytest -q
```

Figures are drawn through `cofolding_tts.figstyle`: Computer Modern, boxed axes,
inward ticks. It renders text with LaTeX when a `latex` binary is on PATH and falls
back to matplotlib's own fonts when it is not, so a bare clone still produces every
figure.

**Reproduce** — needs a GPU. Boltz-2 is not vendored; one script installs it.

```bash
bash scripts/setup_boltz.sh            # a separate venv with boltz==2.2.1
export BOLTZ_CHECKPOINT=/path/to/boltz2_conf.ckpt
export BOLTZ_CACHE=/path/to/boltz/cache

python scripts/run_8sge.py --arm control --draws 30 --python .venv-boltz/bin/python
python scripts/run_8sge.py --arm gauss   --draws 40
python scripts/run_8sge.py --arm tcp     --draws 20     # also tcp-frozen, tcp-c10
python scripts/score_8sge.py --runs runs
```

Every number here was produced with **boltz 2.2.1**. The intervention patches named
attributes of named Boltz classes, so a release that moved one would install cleanly
and silently do nothing; the runner refuses any other version unless you pass
`--allow-any-boltz`. Runs land under `runs/`, which is not tracked.

## Notebooks

Four, self-contained, no GPU — `numpy`, `matplotlib` and this package over committed
data.

| notebook | what it shows |
| --- | --- |
| `01_baseline.ipynb` | the failure on 8SGE, recomputed from coordinates |
| `02_baselines.ipynb` | the unperturbed rate on four systems, with familiarity beside it |
| `03_tcp_arms.ipynb` | Boltz-Perturb's TCP against ours, and the frozen control |
| `04_committor.ipynb` | the committor along the hindered axis, and the tilt of the landscape |

## What is in here

```
src/cofolding_tts/   intervene.py (the hook + CLI), tcp.py, score.py, axis.py,
                     lddt.py, pose.py, pdbio.py, basin.py, figstyle.py
data/cross_model/    645 draws over 8 systems; 600 scored poses on 8SGE
data/baselines/      unperturbed draws and familiarity, four systems
data/tcp/            the five 8SGE arms, one row per pose
data/injection/      the step-140 coordinate-injection ladder
data/reference/      the 8SGE inputs a run needs
pdb/                 deposited structures for eight systems, with checksums
docs/method.md       what the intervention is and why it is where it is
```

What this README does not cover is what to do about the problem. The intervention,
its arms and the landscape they move on are in `docs/method.md`, `data/tcp/` and
notebooks `03`–`04`; read **Limits** before quoting any of it.

## Limits

- **The sampling claim holds in five of sixteen cells, not everywhere.** In seven
  cells no draw ever reached the crystal pose, and those say nothing about whether
  it is reachable at all.
- **The failure is shown across systems and models; the intervention is
  characterised on one system with one model.** Every per-pose arm table committed
  here is Boltz-2 on 8SGE.
- **Nothing ranks the draws.** A rare correct draw is worth nothing without a
  reference-free way to pick it out, which is separate and unsolved.
- **The intervention has a cost.** 8JJQ goes 18/20 → 27/40; on 8CJG it goes the
  other way (7/20 → 40/40 and 39/40 under two conditionings); on 8AM2 it moves ~20%
  of draws in *both* directions. Those draws are not vendored here — pointers, not
  results you can check from this clone.
- **Ligand RMSD is always measured against the deposited ligand** after superposing
  the protein, never by fitting the ligand to itself.

## License

MIT, except `results/figures/rnp_familiarity.jpg`, which is someone else's figure
reproduced under CC BY 4.0 and attributed in `results/figures/PROVENANCE.md`. See
`LICENSE`.
