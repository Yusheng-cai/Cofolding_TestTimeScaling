# 8ATB

Runs N' Poses system `8atb__1__1.A__1.C`, taken from the ground-truth bundle rather than
re-downloaded, so the coordinates are byte-identical to the ones every number
in this repository was scored against.

Why this system is here: the decoy null; 25/25 correct unperturbed, so an injected wrong pose has something to be pushed back from.

| file | sha256 (first 16) |
| --- | --- |
| `receptor.cif` | `433112955de05713` |
| `system.cif` | `4a4c7fc0c3445123` |
| `sequences.fasta` | `1de0fbeb6cb34d36` |
| `ligand_1.C.sdf` | `d1ab07987f40e070` |

`system.cif` is the deposited complex, `receptor.cif` the protein alone,
and `ligand_*.sdf` the deposited ligand with its bond orders — the reference
every ligand RMSD in this repository is measured against.
