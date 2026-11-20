# Baseline draws, four systems

`control_poses.csv` — one row per **unperturbed** Boltz-2 draw. No intervention arms
are in this file; these are the numbers every arm has to be read against.

| column | meaning |
| --- | --- |
| `pdb` | PDB id of the system |
| `draw` | the run or pose identifier in the source experiment |
| `ligand_rmsd_angstrom` | symmetry-corrected ligand RMSD **to the deposited ligand**, after protein alignment |
| `source_correct` | the source experiment's own correctness flag, where it had one; its criterion is per-system, so prefer recomputing from the RMSD |
| `source` | the file the row came from |

`systems.csv` — one row per system: the **familiarity** score (SuCOS shape x pocket
quality-coverage against the closest system released before the 2021-09-30 structural
training cutoff, on 0-100), which system that was and when it was released, and how
many control draws are in `control_poses.csv`.

| PDB | familiarity | closest pre-cutoff system | released |
| --- | --- | --- | --- |
| 8SGE | 29.5 | `6do5__1__1.A__1.D` | 2018-12-19 |
| 7FTM | 0.0 | `5vdt__1__1.A__1.D` | 2017-09-27 |
| 8CJG | 0.0 | `3f8r__1__1.A_1.B__1.E` | 2009-01-13 |
| 8JJQ | 0.0 | `7br3__1__1.A__1.L` | 2020-10-07 |

Read that column beside the rates: the three systems at 0.0 include the one the model
handles best, and the only system with real familiarity is the one it fails 24 times
in 25. Familiarity is not what is failing here.

## Where each system's draws come from

| PDB | n | sampling | source file | sha256 (first 16) |
| --- | --- | --- | --- | --- |
| 8SGE | 25 | 25 model seeds, one pose each | `exp034_boltz2_8sge_multisample/per_pose_metrics.csv` | `1ea1752313fbe578` |
| 8CJG | 20 | 20 diffusion draws under one conditioning | `exp104_8cjg_slig_ablation/per_run.csv` | `f8ef8d77fd92fc27` |
| 8JJQ | 20 | 20 diffusion draws under one conditioning | `exp105_8jjq_slig_harm_control/per_run.csv` | `c98e0e00e52cffcb` |
| 7FTM | 32 | 32 poses across model seeds | `exp033_boltz2_7ftm_multisample/per_pose_metrics.csv` | `6b810478294db0c3` |

**The sampling differs between systems**, and that is not cosmetic: 8SGE and 7FTM
vary the model seed, which redraws the conditioning, while 8CJG and 8JJQ vary only
the diffusion noise under one fixed conditioning. Rates are comparable within a
system and only loosely across them.

`extract_baselines.py` rebuilds this file from that experiment tree. It cannot run on
a fresh clone — the sources are not vendored — and is committed so the table has a
recipe rather than only a description.

## Systems with no baseline table here

- **8AM2** has no single baseline rate. The model is right under one conditioning and
  wrong under another, which is the whole point of that system, so a per-draw control
  file would misrepresent it.
- **8ATB** is used only as the decoy null in the injection work; its unperturbed
  25/25 is recorded there, not as a scored baseline sweep.
- **8RDK** and **8PFL** are prepared but unrun.
