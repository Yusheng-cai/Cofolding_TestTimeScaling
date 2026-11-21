# The step-140 injection ladder

`ladder.csv` — 520 rows, one per pose: 13 injected angles x 20 replicates x 2
conditionings. Built by `extract_ladder.py` from the experiment tree this work came
from; it cannot run on a fresh clone.

## What was done

The crystal ligand is rotated about the hindered axis (C12-C11-N1-C3) to each of 13
angles — nothing else about it changes, so each injected pose is the deposited
geometry with one bond turned. At denoising **step 140 of 200** the ligand's 33 rows
of coordinates are overwritten with that pose, Kabsch-placed into the sampler's
current frame and carrying `sigma_tm * randn` so the denoiser sees the noise level it
expects. The protein is never touched. The sampler then finishes unmodified.

| conditioning | meaning | source |
| --- | --- | --- |
| `intact` | the trunk's ligand rows untouched | `exp116_7ftm_coordinate_injection/lddt_pli_torsion.csv` |
| `ablated` | those rows replaced with matched Gaussian noise | `exp117_8sge_slig_torsion_scan/lddt_pli_torsion_slig.csv` |

Checksums of those two tables are printed by the extractor. The no-injection
references for both conditionings are `base_noinj` and `slig_noinj` in
`data/tcp/arm_poses.csv`, not duplicated here.

## Why step 140

Below about step 137 the trajectory still carries enough noise to erase the
injection: at step 120 every injected angle returns byte-identical output. Above
about 160 every injection is kept whatever it is, because the sampler stops injecting
noise at step 154 and the tail is deterministic. Steps 137-154 are the only window
where an injection both survives and still has thermal noise to relax into. The basin
is normally chosen by step 133, so the injection arrives *after* the model has
decided and overrides a commitment rather than guiding an open one.

## What the table is for

The fraction of replicates finishing in the crystal basin, as a function of the angle
injected, is a committor along a real internal coordinate. `notebooks/04_committor.ipynb`
computes it: surfaces at -19 and -170 degrees against the -23/+157 a symmetric
landscape would give, so 146 degrees of the circle lead to the crystal basin and 214
to the model's.

**Every RMSD in this repository is measured against the crystal, never against an
injected pose.** The injected pose is an input, not a reference.
