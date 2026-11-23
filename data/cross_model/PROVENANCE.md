# Cross-model tables

Consolidations of sweeps that already existed. Nothing here is recomputed from raw
predictions; a disagreement with a source report is a bug in the extraction, not a
new result. `extract_cross_model.py` rebuilds all three files and prints the source
checksums.

## `pose_lddt.csv` — 600 poses, 8SGE, three models

One row per scored pose from exp096: three models × two MSA conditions × 20 seeds ×
5 diffusion samples. No intervention; these are ordinary unperturbed draws.

| column | meaning |
| --- | --- |
| `model` | Boltz-2, AlphaFold3 or Protenix |
| `msa` | `full 2633` (the ordinary input) or `frozen 1024` (a fixed depth-1024 subsample shared by all three models) |
| `seed`, `sample` | model seed, and the diffusion sample index within it |
| `ligand_rmsd_angstrom` | symmetry-corrected ligand RMSD **to the deposited ligand**, after protein alignment |
| `lddt_pli` | lDDT of the protein–ligand interface |
| `axial_torsion_deg` | C12–C11–N1–C3, the order parameter separating the two rotamers; crystal **−101.4°** |
| `crystal_side_of_axis` | 1 if `|wrap(torsion − (−101.4))| < 90°` |

Recycling differs between models and is not a free parameter: Boltz-2 ran 3 recycles,
AlphaFold3 and Protenix 10, each at its own default. The arms are each model as
published, not a controlled compute comparison.

Source: `reports/exp096_8sge_four_model/pli_per_pose.csv`,
sha256 `67512578449277fd…`.

`scripts/figure_landscape.py` draws the `full 2633` arms.

## `systems.csv` — 16 cells, 8 systems

One row per (system, model) cell, the largest **full-MSA** sweep for that pair. The
frozen-MSA arms are dropped here: they exist to ask a different question and are not
comparable with the ordinary sweeps.

| column | meaning |
| --- | --- |
| `pdb` | deposited complex |
| `familiarity` | SuCOS shape × pocket quality-coverage against the closest system released before the 2021-09-30 structural training cutoff, 0–100 |
| `model`, `experiment_id` | which model, and the sweep the cell came from |
| `n_poses`, `hits_lt2` | poses scored, and how many landed under 2 Å |
| `best_rmsd`, `median_rmsd` | over those poses |

**The cells are not uniformly sampled.** Some vary the model seed (one pose each),
some vary the diffusion sample under a fixed seed, and the counts range 25–100. Rates
are solid within a cell and only loosely comparable across them — enough to separate
"never" from "usually", not enough for a two-point difference to mean anything.

Protenix has been run on 8SGE only, so its single marker is not evidence about the
other seven systems.

Source: `reports/cross_model_cofolding/cross_model_pose_results.csv`,
sha256 `15c357215c486c6e…`, itself built from the per-pose CSVs of exp040, exp041,
exp077, exp079, exp095 and exp096.

It is the index for `system_poses.csv` below, and the two must agree.

## `system_poses.csv` — 645 draws, the same 16 cells

One row per **draw** behind those cells: the distribution each cell's summary line
was computed from. A rate hides the shape, and the shape is what distinguishes a
model that is uncertain from one that is confidently wrong.

| column | meaning |
| --- | --- |
| `pdb`, `familiarity`, `model`, `experiment_id` | the cell this draw belongs to |
| `pose` | the draw's seed, or its index where the source recorded no seed |
| `ligand_rmsd_angstrom` | symmetry-corrected ligand RMSD **to the deposited ligand**, after protein alignment |

Sources: followed from each cell's own `source_csv` pointer — the per-pose CSVs of
exp040, exp041, exp077, exp095 and exp096. Those files were written by five
different experiments and name the system and the model differently; the extractor's
`system_of` / `model_of` / `pose_of` readers are the whole of that difference.

**The extractor checks itself.** For every cell it recomputes `n_poses` and
`hits_lt2` from the draws it read and refuses to write the file if either disagrees
with `systems.csv`. The committed table passes that check on all 16 cells, which is
what makes it safe to treat these two files as one table.

There is no lDDT-PLI column here. That metric was only ever computed for 8SGE, where
it is in `pose_lddt.csv`; the other seven systems have ligand RMSD alone.

`scripts/figure_cross_model.py` draws it.
