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

## The failure is not one structure, and not one model

![every scored draw, eight systems, three models](results/figures/cross_model.png)

Every unperturbed draw this project has scored: 645 poses, 8 deposited complexes, 3
models, no oracle. A hit rate would hide the thing worth seeing, which is the shape
of each strip.

- **A tight clump far from the crystal is not uncertainty.** 7FTM's 25 Boltz-2 draws
  span 0.99 Å about a median of 31 Å; 8H7R's 30 span 0.56 Å about 8.2 Å. Drawing
  again returns the same wrong answer, which is why more samples do not help.
- **Two clumps with a gap is a choice between two answers** — 8CJG and 8AM2 under
  Boltz-2, 8SGE under AlphaFold3, 8RDK under both. What a hit rate reports is just
  the fraction that landed in the near clump.
- **One clump inside the green band is a solved system** — 8RDK under both models,
  8CJG under AlphaFold3. Protenix on 8SGE is the in-between case: most of its draws
  are inside the band and a tail of them is not. Where models disagree they disagree
  sharply and not in anyone's consistent favour. An empty slot is a model that was
  never run on that system, not one that scored badly.

**In five of the sixteen cells the right answer is already in the model's own
output — it is just rare.** Boltz-2 puts 10 of 25 8CJG draws under 2 Å while its
median sits at 5.7 Å; on 8SGE it reaches 0.93 Å once in 100. AlphaFold3 reaches
1.02 Å on 8SGE 19 times in 100, and 1.32 Å on 8AM2 once in 25. Nothing is missing
from the model in those cells. The crystal pose is in its distribution, the draws
around it are not, and which one you get is the whole of the difference — that is
the sampling failure this repository is about, and it is why spending compute at
inference is a sensible move there rather than a hope.

**In the other four systems no draw ever came within 2 Å.** The closest was 2.10 Å,
Boltz-2 on 8JMP; the rest never got nearer than 4.4 Å, and 7FTM never left 30 Å.
For those, nothing here shows the answer is reachable at all, and drawing more
samples is not what would find out.

Familiarity is on the right axis, and inside this set it does not order the outcome.
That is not a count against the Runs N' Poses trend: these eight were selected for
being hard, so the honest reading is that a population-level trend gives you no grip
on an individual complex — which is the case in which you would want to spend
inference compute.

Cells are not uniformly sampled, a hit here is RMSD < 2 Å alone rather than the
benchmark's stricter combined criterion, and Protenix has only been run on 8SGE —
read `data/cross_model/PROVENANCE.md` before quoting a rate.

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

## You cannot sample your way out of it

Within one conditioning the diffusion is near-deterministic. A given Boltz-2 seed
returns the same pose every time, and the unperturbed 8SGE draws agree with *each
other* to 0.11 Å while disagreeing with the crystal by 6.8 Å. Best-of-N over seeds
re-rolls the conditioning, not the pose.

So the rare correct draws two figures up are rare in a particular way. They are not
a sampler exploring and occasionally finding the answer; they are a handful of
conditionings that happened to land on the other side, drawn from a model that will
then reproduce that answer exactly as faithfully as it reproduces the wrong one.
That is what makes this a search problem rather than a precision problem, and it is
where the repository's remaining work starts — `docs/method.md` for the intervention
that follows from it, `03_tcp_arms.ipynb` and `04_committor.ipynb` for its arms and
the landscape they move on, `data/tcp/` for the per-pose tables. Read **Limits**
before quoting any of that.

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
