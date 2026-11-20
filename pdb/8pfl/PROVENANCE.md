# 8PFL

Runs N' Poses system `8pfl__1__1.A__1.C`, taken from the ground-truth bundle rather than
re-downloaded, so the coordinates are byte-identical to the ones every number
in this repository was scored against.

Why this system is here: prepared, not yet run.

| file | sha256 (first 16) |
| --- | --- |
| `receptor.cif` | `c004864924ba422f` |
| `system.cif` | `95decca25909cf6c` |
| `sequences.fasta` | `f323453b5793bfb0` |
| `ligand_1.C.sdf` | `0c006e7c510fd725` |

`system.cif` is the deposited complex, `receptor.cif` the protein alone,
and `ligand_*.sdf` the deposited ligand with its bond orders — the reference
every ligand RMSD in this repository is measured against.
