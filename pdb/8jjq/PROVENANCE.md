# 8JJQ

Runs N' Poses system `8jjq__1__1.A__1.D`, taken from the ground-truth bundle rather than
re-downloaded, so the coordinates are byte-identical to the ones every number
in this repository was scored against.

Why this system is here: the harm control; Boltz-2 is right ~90% of the time and the intervention costs accuracy here.

| file | sha256 (first 16) |
| --- | --- |
| `receptor.cif` | `1b4de0e562b89a41` |
| `system.cif` | `51d0063bca76197e` |
| `sequences.fasta` | `4be7a2d83cea908f` |
| `ligand_1.D.sdf` | `120ddfe460ee5a2a` |

`system.cif` is the deposited complex, `receptor.cif` the protein alone,
and `ligand_*.sdf` the deposited ligand with its bond orders — the reference
every ligand RMSD in this repository is measured against.
