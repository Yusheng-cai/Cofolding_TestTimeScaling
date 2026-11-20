# Crystal structures

The deposited structures this repository scores against, vendored so a clone is
runnable and checkable offline. Each folder carries `receptor.cif` (protein),
`system.cif` (deposited complex), `ligand_*.sdf` (deposited ligand, bond orders
intact), `sequences.fasta`, and a `PROVENANCE.md` with per-file checksums.

Source: the Runs N' Poses ground-truth bundle, copied rather than re-downloaded,
so these are the exact coordinates behind every number here. **Every ligand RMSD
in this repository is measured against `ligand_*.sdf`, never against a predicted
or injected pose.**

| PDB | system id | files | size | why it is here |
| --- | --- | --- | --- | --- |
| **7FTM** | `7ftm__1__1.A__1.C` | 4 | 462 KB | the 31 A displacement: one reproducible wrong answer in the wrong cavity, used for the coordinate-injection sweep |
| **8AM2** | `8am2__1__1.A__1.D` | 4 | 674 KB | the symmetric case; the intervention moves ~20% of draws in BOTH directions and the arms never meet |
| **8ATB** | `8atb__1__1.A__1.C` | 4 | 346 KB | the decoy null; 25/25 correct unperturbed, so an injected wrong pose has something to be pushed back from |
| **8CJG** | `8cjg__1__1.A__1.D` | 4 | 833 KB | rescued cleanly: the indole leaves its slot in the wrong pose, so a contact metric sees the error |
| **8JJQ** | `8jjq__1__1.A__1.D` | 4 | 377 KB | the harm control; Boltz-2 is right ~90% of the time and the intervention costs accuracy here |
| **8PFL** | `8pfl__1__1.A__1.C` | 4 | 524 KB | prepared, not yet run |
| **8RDK** | `8rdk__1__1.A__1.D` | 4 | 412 KB | watchlist: a second placement that is reachable |
| **8SGE** | `8sge__1__1.A__1.H` | 4 | 412 KB | the atropisomer; two rotamers of one hindered bond, and the system every figure in this repository is built on |

Total: 4044 KB across 8 systems.
