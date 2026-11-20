# 8RDK

Runs N' Poses system `8rdk__1__1.A__1.D`, taken from the ground-truth bundle rather than
re-downloaded, so the coordinates are byte-identical to the ones every number
in this repository was scored against.

Why this system is here: watchlist: a second placement that is reachable.

| file | sha256 (first 16) |
| --- | --- |
| `receptor.cif` | `5976d848abcfeead` |
| `system.cif` | `c8979d79d8d4f90f` |
| `sequences.fasta` | `95bbc40f3e884c2c` |
| `ligand_1.D.sdf` | `bdd8e3a14f576a7c` |

`system.cif` is the deposited complex, `receptor.cif` the protein alone,
and `ligand_*.sdf` the deposited ligand with its bond orders — the reference
every ligand RMSD in this repository is measured against.
