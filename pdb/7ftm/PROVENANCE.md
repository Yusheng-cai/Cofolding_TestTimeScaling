# 7FTM

Runs N' Poses system `7ftm__1__1.A__1.C`, taken from the ground-truth bundle rather than
re-downloaded, so the coordinates are byte-identical to the ones every number
in this repository was scored against.

Why this system is here: the 31 A displacement: one reproducible wrong answer in the wrong cavity, used for the coordinate-injection sweep.

| file | sha256 (first 16) |
| --- | --- |
| `receptor.cif` | `fdf1baeb16f1cb17` |
| `system.cif` | `f2e02aeb3c36fc5f` |
| `sequences.fasta` | `a13ea94c7f2afe0f` |
| `ligand_1.C.sdf` | `c8c5b222d83d4581` |

`system.cif` is the deposited complex, `receptor.cif` the protein alone,
and `ligand_*.sdf` the deposited ligand with its bond orders — the reference
every ligand RMSD in this repository is measured against.
