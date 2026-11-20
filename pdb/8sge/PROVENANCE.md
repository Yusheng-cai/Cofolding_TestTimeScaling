# 8SGE

Runs N' Poses system `8sge__1__1.A__1.H`, taken from the ground-truth bundle rather than
re-downloaded, so the coordinates are byte-identical to the ones every number
in this repository was scored against.

Why this system is here: the atropisomer; two rotamers of one hindered bond, and the system every figure in this repository is built on.

| file | sha256 (first 16) |
| --- | --- |
| `receptor.cif` | `0c668b7e7df23583` |
| `system.cif` | `8692ab235d245c74` |
| `sequences.fasta` | `3b73d766de0f4ae6` |
| `ligand_1.H.sdf` | `c4525a0ac68025fb` |

`system.cif` is the deposited complex, `receptor.cif` the protein alone,
and `ligand_*.sdf` the deposited ligand with its bond orders — the reference
every ligand RMSD in this repository is measured against.
