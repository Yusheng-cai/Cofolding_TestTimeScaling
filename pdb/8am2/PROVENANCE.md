# 8AM2

Runs N' Poses system `8am2__1__1.A__1.D`, taken from the ground-truth bundle rather than
re-downloaded, so the coordinates are byte-identical to the ones every number
in this repository was scored against.

Why this system is here: the symmetric case; the intervention moves ~20% of draws in BOTH directions and the arms never meet.

| file | sha256 (first 16) |
| --- | --- |
| `receptor.cif` | `03e1824de220f4bb` |
| `system.cif` | `a6386e1bad2a05ad` |
| `sequences.fasta` | `b51677e3a7247182` |
| `ligand_1.D.sdf` | `c973daa92dba50aa` |

`system.cif` is the deposited complex, `receptor.cif` the protein alone,
and `ligand_*.sdf` the deposited ligand with its bond orders — the reference
every ligand RMSD in this repository is measured against.
