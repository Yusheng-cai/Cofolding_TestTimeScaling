"""Test-time interventions on co-folding models, and the metrics to judge them.

The package is deliberately split so that reading the result does not require the
machinery to produce it:

    intervene  installs the hook that edits the trunk representation (needs torch
               and a Boltz-2 install)
    score      ligand RMSD against a crystal reference, after protein alignment
    axis       atropisomer-axis perception, and the 8SGE axial label
    lddt       lDDT-PLI
    pose       Kabsch and symmetry-corrected RMSD, numpy only
    pdbio      column-correct PDB records
"""

__version__ = "0.1.0"
