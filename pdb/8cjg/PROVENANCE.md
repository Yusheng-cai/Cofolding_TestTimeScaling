# 8CJG

Runs N' Poses system `8cjg__1__1.A__1.D`, taken from the ground-truth bundle rather than
re-downloaded, so the coordinates are byte-identical to the ones every number
in this repository was scored against.

Why this system is here: rescued cleanly: the indole leaves its slot in the wrong pose, so a contact metric sees the error.

| file | sha256 (first 16) |
| --- | --- |
| `receptor.cif` | `36cec2b894aea378` |
| `system.cif` | `e9cc7dd0079c2e20` |
| `sequences.fasta` | `876b39f24a0445db` |
| `ligand_1.D.sdf` | `4c02b2bc601ada01` |

`system.cif` is the deposited complex, `receptor.cif` the protein alone,
and `ligand_*.sdf` the deposited ligand with its bond orders — the reference
every ligand RMSD in this repository is measured against.
