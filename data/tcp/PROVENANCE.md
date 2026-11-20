# The five 8SGE arms

`arm_poses.csv` — one row per pose, 20 poses an arm, 100 in all. Built by
`extract_tcp_arms.py` from the experiment tree this work came from; it cannot run on
a fresh clone.

| arm | intervention | source |
| --- | --- | --- |
| `base_noinj` | none | `exp117_8sge_slig_torsion_scan/base_noinj` |
| `slig_noinj` | ours: ligand rows of `s_trunk` replaced with matched Gaussian noise, drawn once | `exp117_8sge_slig_torsion_scan/slig_noinj` |
| `tcp_c11` | Boltz-Perturb C11: `s` perturbed over steps 80–159, redrawn every step | `exp118_8sge_tcp_replication/tcp_c11` |
| `tcp_c11_frozen` | C11 with the direction drawn once and reused | `exp118_8sge_tcp_replication/tcp_c11_frozen` |
| `tcp_c10` | C11's magnitude over the whole trajectory | `exp118_8sge_tcp_replication/tcp_c10` |

Columns: `ligand_rmsd_angstrom` against the deposited ligand after protein alignment,
`final_torsion_deg` on the hindered axis, `lddt_pli`, and Boltz's own
`confidence_score`.

All five arms share seed 42 and every Boltz flag, so the per-pose atom-noise draws
match across arms and a difference between them is the intervention rather than the
sampling. Scored with `cofolding_tts.basin`, these reproduce the recorded summary
exactly: 1, 9, 11, 7 and 5 of 20 in the correct basin, correct-basin median RMSD
0.92, 2.98, 1.25, 2.47 and 2.85 Å.

## What is missing

TCP has been run on **8SGE only**. The arm that matters most is unrun: 8JJQ, where
the model is right 18 times in 20 unperturbed and our own ablation costs accuracy
(90% to 68%). Until that arm exists, nothing here says what TCP costs where the model
is already right. 7FTM and 8CJG are also unrun.

Those systems are not runnable from this repository either: only 8SGE's inputs are
vendored.
