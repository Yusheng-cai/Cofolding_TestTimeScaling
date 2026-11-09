# 8SGE reference data

One system, vendored so that the repository is runnable on a fresh clone.

| file | what it is | source |
|---|---|---|
| `receptor.cif` | protein chain `1.A` of the deposited complex | PDB entry **8SGE** |
| `system.cif` | the full deposited complex, including ligand `ZWS` | PDB entry **8SGE** |
| `ligand_1H.sdf` | the crystal ligand, chain `1.H`, as the benchmark processes it | Runs N' Poses |
| `sequences.fasta` | the modelled sequence | Runs N' Poses |
| `msa_1A.csv.gz` | the MSA used for every run in this repository | generated for the parent project with the Boltz-2 MSA pipeline; gzipped here |
| `input.yaml.template` | the Boltz-2 input, with `{MSA_PATH}` filled in at run time | written for this repository |

**Why these are committed.** The rest of the parent project never commits raw
datasets. This is a deliberate exception for a single public PDB entry, about
430 KB in total, because without it nothing in this repository can be scored or
drawn and the result cannot be checked by a reader.

**The MSA is stored gzipped and the input YAML as a template** because a committed
YAML would have to carry an absolute path to the MSA and would then only work on
the machine that wrote it. `scripts/run_8sge.py` unpacks the MSA and fills the
template into `runs/_inputs/` at launch.

**Citation.** PDB 8SGE is the KLHDC2 complex with the degrader ZWS. The deposited
chemical name carries the `(5P)` axial descriptor that the isomeric SMILES does
not, which is the whole subject of this repository. Cite the PDB entry and the
Runs N' Poses benchmark if you use these files.
